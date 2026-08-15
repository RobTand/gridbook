"""Quantized ``VocabParallelEmbedding``: the lookup table as a paid-for unit.

WHY THIS EXISTS
    On a 27B-class model the token embedding is ~1.27G parameters — 2.37 GiB in
    BF16, which is *larger than the entire lm_head* and a sixth of a 16 GB card.
    Every other weight in the checkpoint is priced by the allocator and shipped
    in whichever format the measurement says it deserves; the embedding alone
    has been exempt, not because it earned the exemption but because no runtime
    would serve it any other way.

    vLLM's compressed-tensors will not close that gap.  Its embedding path
    accepts weight-only INT group/channel schemes and *raises* for anything
    else, so NVFP4 — the format that actually wins here — has no stock route.
    Gridbook already owns an out-of-tree dispatch for exactly this class of
    problem, so this module adds the missing method rather than asking for a
    change to vLLM.

WHAT THE MEASUREMENT SAID (Qwen3.8-27B, exact full-vocab KL, 4088 positions)
    The embedding is not terminal, so its cost was measured end to end: perturb
    the embedding, re-run the forward, and compare the resulting logits against
    the BF16 control through an *untouched* head.  Two passes, because layer 0
    consumes the embedding's output through an NVFP4 (W4A4) projection on the
    shipping recipe and therefore already quantizes it:

        arm                 GiB      KL (BF16 body)   KL (4-bit act floor)
        INT8   W8A16        1.185       0.000331          0.000706
        INT4   W4A16 g128   0.611       0.001963          0.002664
        FP8    per-row      1.185       0.000342          0.000890
        NVFP4  group-16     0.666       0.001063          0.000948

    The 8-bit arms get *worse* under the activation floor and NVFP4 gets
    *better* — the same 4-bit grid that the activation is about to be snapped
    to absorbs part of the weight error instead of adding to it.  Under the
    floor the model will actually serve at, NVFP4 matches FP8's KL for 44%
    fewer bytes, which is why this module implements NVFP4 first.

    Against the body's own real-KL frontier (0.0315 KL/GiB at 4.5 bpp), the
    1.70 GiB this frees is worth far more spent on decoder weights than on a
    lossless lookup table.  That comparison — marginal KL per marginal byte,
    every consumer priced on the same axis — is the whole argument.

WHY A UNIFORM PER-ROW FORMAT IS SAFE ON A ZIPFIAN VOCABULARY
    An end-to-end KL measurement only exercises rows whose tokens appear in the
    calibration windows: 1,446 of 248,320 here (0.58%).  Rows never emitted
    contribute exactly zero, so the measured KL is a LOWER bound over the
    vocabulary, and no amount of extra calibration closes a tail that long.

    It closes structurally instead.  Every row gets its own group scales and
    the same bit width regardless of how often it is used, so per-row
    reconstruction error is a property of the row, not of its frequency.
    Measured over all 248,320 rows, tokens that never appear in the corpus at
    all carry 1.011x the relative error of the measured population under NVFP4
    (1.004x under FP8), flat at p99 and at max.  The rare rows are not damaged
    disproportionately, so the blind part of the vocabulary is no worse off
    than the part the KL run saw.

    The Zipf caution is real, but it applies to per-row *allocation* — spending
    fewer bits on rare rows because no evidence defends them.  This module does
    not allocate per row.

WHY GATHER-THEN-DEQUANTIZE
    ``embedding()`` receives token ids, so only the selected rows are ever
    needed: T rows per forward, not 248,320.  Dequantizing the gathered rows is
    both cheaper than materializing the table and the *reason* the memory
    saving is real — a method that dequantized the whole table on load would
    ship the bytes and then spend them again.

WHAT THIS MODULE DOES NOT OWN
    ``ParallelLMHead`` subclasses ``VocabParallelEmbedding`` in vLLM but is a
    Linear in every way that matters here, and compressed-tensors already
    serves it quantized through its own linear method.  Dispatch must exclude
    it; see the caller in ``config.get_quant_method``.  This module refuses a
    ``ParallelLMHead`` defensively rather than trusting that seam to hold.

ABSENCE MEANS LEGACY
    An artifact with no declaration behaves exactly as it did before this
    module existed.  Every published artifact predates the schema, so absence
    stays a no-op rather than becoming a new failure mode.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

import torch


__all__ = [
    "EmbeddingFormatError",
    "SCHEMA_KEY",
    "SUPPORTED_SCHEMA_VERSIONS",
    "EmbeddingFormat",
    "FORMATS",
    "parse_declaration",
    "GridbookNVFP4EmbeddingMethod",
]


class EmbeddingFormatError(ValueError):
    """An embedding declaration Gridbook refuses to serve."""


#: Top-level key in ``quant_config.json`` carrying the declaration.
SCHEMA_KEY = "quantized_embedding"

#: Schema versions this build understands.  A newer artifact must fail loudly
#: rather than be read with older rules -- a version bump is how the producer
#: says "the meaning changed".
SUPPORTED_SCHEMA_VERSIONS = (1,)


class EmbeddingFormat(NamedTuple):
    """One audited embedding format and the method that serves it."""

    #: Stable id written by the producer.  Never reuse an id for new semantics.
    id: str
    #: Human description, quoted verbatim into refusal messages.
    description: str
    #: Group size along the embedding dimension.
    group_size: int


#: Audited formats.  Anything absent is UNKNOWN, which fails.  NVFP4 is first
#: because it is what the measurement in this module's header selected; adding
#: a rung here is a serving promotion and needs its own end-to-end KL evidence.
FORMATS: Mapping[str, EmbeddingFormat] = {
    "nvfp4": EmbeddingFormat(
        id="nvfp4",
        description="NVFP4 E2M1 group-16 with FP8-E4M3 group scales and one "
                    "FP32 per-tensor global scale (compressed-tensors "
                    "nvfp4-pack-quantized layout)",
        group_size=16,
    ),
}


def parse_declaration(
    cfg: Mapping[str, Any],
    *,
    canonicalize: Callable[[str], str],
    cb_targets: frozenset[str] = frozenset(),
) -> dict[str, EmbeddingFormat]:
    """Resolve *cfg*'s embedding declaration to ``{target: EmbeddingFormat}``.

    Fail-closed on every axis the producer could get wrong: unknown schema
    version, unknown format id, a unit claimed by both this declaration and the
    CB vocabulary.  Absence returns ``{}`` -- see the module header.
    """
    raw = cfg.get(SCHEMA_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise EmbeddingFormatError(
            f"{SCHEMA_KEY!r} must be a mapping; got {type(raw).__name__}")

    version = raw.get("version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise EmbeddingFormatError(
            f"{SCHEMA_KEY!r} schema version {version!r} is not supported by "
            f"this Gridbook build (understands {list(SUPPORTED_SCHEMA_VERSIONS)}). "
            "A version bump means the producer changed what the fields mean; "
            "upgrade Gridbook rather than reading new bytes with old rules.")

    units = raw.get("units")
    if not isinstance(units, Mapping) or not units:
        raise EmbeddingFormatError(
            f"{SCHEMA_KEY!r} declares no units; omit the key entirely to serve "
            "the embedding unquantized.")

    out: dict[str, EmbeddingFormat] = {}
    for target, fmt_id in units.items():
        key = canonicalize(str(target))
        if key in cb_targets:
            raise EmbeddingFormatError(
                f"unit {key!r} is claimed by both {SCHEMA_KEY!r} and the CB "
                "config_groups vocabulary; exactly one dispatch may own a "
                "unit's resident weights.")
        fmt = FORMATS.get(str(fmt_id).strip().lower())
        if fmt is None:
            raise EmbeddingFormatError(
                f"unit {key!r} declares embedding format {fmt_id!r}, which "
                f"this Gridbook build has not audited "
                f"(known: {sorted(FORMATS)}).")
        out[key] = fmt
    return out


# ---------------------------------------------------------------------------
# NVFP4 decode.
#
# The on-disk contract is compressed-tensors' ``nvfp4-pack-quantized``, written
# by prismaquant's ``quantize_dequantize_nvfp4``:
#
#   weight_packed        uint8   [rows, cols/2]   two E2M1 nibbles per byte,
#                                                 EVEN element in the LOW nibble
#   weight_scale         fp8e4m3 [rows, cols/16]  s_g_real / global_real
#   weight_global_scale  fp32    [1]              1 / global_real  (DIVISOR)
#
#   value = sign(code) * E2M1[code & 7] * weight_scale.float() * global_real
#
# The divisor convention on the global scale is not ours to change -- vLLM's own
# NVFP4 linear path inverts on load the same way, and the exporter writes one
# number read by both.  Getting it backwards is silent and catastrophic, so the
# parity test renders a whole tensor both ways rather than checking a formula.
# ---------------------------------------------------------------------------

#: E2M1 magnitudes indexed by the low three bits of a nibble; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_VALUE_LUT_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _value_lut(device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """16-entry signed E2M1 value table, one lookup per nibble.

    Folding the sign into the table keeps the hot path a single gather instead
    of a mask/shift/negate chain, and keeps it free of any op whose shape
    depends on the data -- this runs inside a CUDA-graph-captured decode step.
    """
    key = (str(device), dtype)
    lut = _VALUE_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(
            list(_E2M1_MAGNITUDES) + [-v for v in _E2M1_MAGNITUDES],
            device=device, dtype=dtype)
        _VALUE_LUT_CACHE[key] = lut
    return lut


def nvfp4_gather_dequant(
    packed_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    global_real: torch.Tensor,
    rows: torch.Tensor,
    *,
    group_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize only the rows named by *rows*.

    ``scale_u8`` is the FP8-E4M3 group-scale plane reinterpreted as bytes:
    ``index_select`` has no FP8 kernel, and a byte view is exact and zero-copy
    where a dtype conversion at load time would have spent back a third of the
    memory this format saves.
    """
    idx = rows.reshape(-1)
    n = idx.numel()
    cols = packed_u8.shape[1] * 2
    n_groups = cols // group_size

    pk = packed_u8.index_select(0, idx)                       # [n, cols/2] u8
    lo = (pk & 0x0F).to(torch.long)
    hi = (pk >> 4).to(torch.long)
    codes = torch.stack((lo, hi), dim=-1).reshape(n, cols)

    values = _value_lut(packed_u8.device)[codes]              # [n, cols] f32
    scale = (scale_u8.index_select(0, idx)
             .view(torch.float8_e4m3fn)
             .to(torch.float32))                              # [n, n_groups]
    eff = scale * global_real.to(torch.float32)

    out = (values.reshape(n, n_groups, group_size)
           * eff.unsqueeze(-1)).reshape(n, cols)
    return out.to(out_dtype).reshape(*rows.shape, cols)


def _bind_to_quantize_method_base() -> None:
    """Make this a VIRTUAL subclass of vLLM's ``QuantizeMethodBase``.

    Not decoration -- load-bearing.  vLLM's post-load sweep
    (``model_loader/utils.process_weights_after_loading``) selects the modules
    to finalize with ``isinstance(quant_method, QuantizeMethodBase)``.  That is
    a NOMINAL check, so a class that implements the whole surface structurally
    is skipped in silence: the weights load, dispatch is correct, every
    inspection of the artifact looks right, and the model dies on its FIRST
    forward with a missing derived attribute.  That is exactly how this was
    found (2026-08-15 load smoke), and no unit test in either repository could
    have found it, because the sweep only exists inside vLLM.

    Virtual registration rather than real inheritance because the base's
    ``apply`` is a Linear's GEMM: inheriting it would give this class a
    plausible-looking answer to a question a lookup table must refuse.  We
    implement every member the base declares, so the ABC contract holds; what
    we decline is the semantics of being a Linear.

    Bound from ``create_weights`` rather than at import: this module must stay
    importable without vLLM (the declaration half is parsed in the build venv,
    which has no vLLM), and every path that reaches the sweep passes through
    ``create_weights`` first, so this is always in time.
    """
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )

    if not issubclass(GridbookNVFP4EmbeddingMethod, QuantizeMethodBase):
        QuantizeMethodBase.register(GridbookNVFP4EmbeddingMethod)


class GridbookNVFP4EmbeddingMethod:
    """Serve a ``VocabParallelEmbedding`` whose table is NVFP4 on disk.

    A VIRTUAL subclass of ``QuantizeMethodBase`` -- see
    ``_bind_to_quantize_method_base`` for why it must be a subclass at all
    (vLLM's post-load sweep is an isinstance check) and why the subclassing is
    virtual rather than real (the base's ``apply`` is a Linear's GEMM, and an
    embedding that answered it would be claiming an operation it does not
    implement).
    """

    def __init__(self, fmt: EmbeddingFormat, prefix: str) -> None:
        self.fmt = fmt
        self.prefix = prefix

    # -- weights ----------------------------------------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        from torch.nn import Parameter
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )
        from vllm.model_executor.utils import set_weight_attrs

        # Must happen before the post-load sweep, which is the only thing that
        # calls process_weights_after_loading. Idempotent.
        _bind_to_quantize_method_base()

        if isinstance(layer, ParallelLMHead):
            raise EmbeddingFormatError(
                f"{self.prefix!r} is a ParallelLMHead, which vLLM serves "
                "through its compressed-tensors LINEAR method; declaring it "
                f"under {SCHEMA_KEY!r} would take the output projection off "
                "the GEMM path and run it as a lookup.")

        group = self.fmt.group_size
        rows = sum(output_partition_sizes)
        cols = input_size_per_partition
        if cols % (2 * group) != 0:
            raise EmbeddingFormatError(
                f"{self.prefix!r} embedding dim {cols} is not a multiple of "
                f"{2 * group} (group size {group}, two elements per byte); "
                "the packed layout has no truthful spelling for it.")

        # Both planes shard along dim 0 (vocab), which is also vLLM's
        # ``output_dim`` for an embedding -- so its own weight_loader shards
        # them correctly with no override from us.  ``packed_dim`` is 1, i.e.
        # NOT the shard dim, which is why the loader's whole-row assertion is
        # the right one and its pack_factor branch must not fire.
        packed = Parameter(
            torch.empty(rows, cols // 2, dtype=torch.uint8),
            requires_grad=False)
        set_weight_attrs(packed, {"input_dim": 1, "output_dim": 0,
                                  "packed_dim": 1, "pack_factor": 2})
        layer.register_parameter("weight_packed", packed)
        set_weight_attrs(packed, extra_weight_attrs)

        scale = Parameter(
            torch.empty(rows, cols // group, dtype=torch.float8_e4m3fn),
            requires_grad=False)
        set_weight_attrs(scale, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_scale", scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # No output_dim: replicated on every shard.  vLLM's embedding loader
        # takes its `output_dim is None` branch and copies it wholesale.
        gscale = Parameter(torch.empty(1, dtype=torch.float32),
                           requires_grad=False)
        layer.register_parameter("weight_global_scale", gscale)
        set_weight_attrs(gscale, extra_weight_attrs)

        layer.gridbook_embed_dtype = params_dtype
        layer.gridbook_embed_group = group

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The exporter writes 1/global_real (the compressed-tensors divisor
        # convention); invert ONCE here so the hot path never divides and the
        # convention is applied in exactly one place.
        gs = layer.weight_global_scale.data.to(torch.float32).reshape(-1)
        if gs.numel() != 1:
            raise EmbeddingFormatError(
                f"{self.prefix!r} weight_global_scale has {gs.numel()} "
                "elements; the NVFP4 embedding layout is per-tensor.")
        if not bool(torch.isfinite(gs).all()) or float(gs) <= 0.0:
            raise EmbeddingFormatError(
                f"{self.prefix!r} weight_global_scale is {float(gs)!r}; a "
                "nonpositive or non-finite divisor means the producer wrote a "
                "degenerate tensor, and inverting it would poison every token.")
        layer.gridbook_global_real = (1.0 / gs).to(layer.weight_packed.device)
        # Byte alias of the FP8 scale plane: index_select has no FP8 kernel,
        # and this view is exact and allocation-free.
        layer.gridbook_scale_u8 = layer.weight_scale.data.view(torch.uint8)

    # -- serve ------------------------------------------------------------
    def embedding(self, layer: torch.nn.Module,
                  input_: torch.Tensor) -> torch.Tensor:
        return nvfp4_gather_dequant(
            layer.weight_packed.data,
            layer.gridbook_scale_u8,
            layer.gridbook_global_real,
            input_,
            group_size=layer.gridbook_embed_group,
            out_dtype=layer.gridbook_embed_dtype,
        )

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        raise EmbeddingFormatError(
            f"{self.prefix!r} is served as a quantized embedding LOOKUP; it "
            "has no GEMM path. A caller reaching apply() has routed a Linear "
            "to the embedding method.")

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: Any) -> Any:
        raise EmbeddingFormatError(
            f"{self.prefix!r} is a quantized embedding and this model ties its "
            "output projection to it. A tied lm_head would inherit the "
            "embedding's NVFP4 rounding, which is not what was measured and "
            "not what the allocator priced; ship the embedding unquantized on "
            "tied-weight models, or untie them first.")
