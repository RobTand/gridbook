"""``PrismaQuantCBLinearMethod`` — weight loading + apply for CB Linears.

Load-time (LAYOUT.md §3): a byte-shaped uint8 ``cb_qweight`` per Linear, an
fp8-only per-output-channel ``weight_scale``, and the model-level shared
``cb_codebook.*`` sidecars (loaded once by the config). Apply: emulate the
served W4A4/W8A8 activation bucket, then dispatch to the native CUDA/CUTLASS
kernel for the format and activation-row count. FP8-CB has no Triton serving
fallback: an unavailable native extension is a load/runtime error.

Fused vLLM modules (qkv_proj, gate_up_proj) hold several roles' output rows in
one weight; per-role shared codebooks are concatenated and addressed by a
per-output-row offset (``cb_row_offset``) so fusion stays correct.
"""
from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from . import codec
from .bf16_grouped_lane import (dense_mm as bf16_sm120_dense_mm,
                                require_lane as bf16_sm120_require_lane,
                                requested as bf16_sm120_requested)
from .expand import expand_fp4_v2_to_weight
from .fp8_fused_lane import rung_eligible as fp8_fused_rung_eligible
from .fp4v2_fused_midm_lane import (eligible as fp4v2_midm_eligible,
                                    fused_mm as fp4v2_midm_fused_mm,
                                    require_lane as fp4v2_midm_require_lane,
                                    requested as fp4v2_midm_requested,
                                    supports as fp4v2_midm_supports)
from .native_cutlass import (native_cutlass_scaled_mm, native_fp4_quant,
                             native_fp8_quant, require_native_fp4_quant,
                             require_native_fp8_cutlass)
from .ops import (cb_bf16_grouped_mm, cb_gemv_fp4_v2, cb_gemv_fp8,
                  fp4_act_qdq_or_codec)
from .nvfp4_activation_contract import (
    CONTRACT_KEY as _NVFP4_ACTIVATION_CONTRACT_KEY,
    bridge_contract as _route_bridge_contract,
    emit_route,
    fused_fp4_contract as _route_fused_fp4_contract,
    reciprocal_vector as _nvfp4_reciprocal_vector,
    require_identical_loaded_scales,
    rowwise_range_multiplier as _nvfp4_rowwise_range_multiplier,
)

# NOTE: the fused-sibling fallback map (qkv_proj, gate_up_proj, in_proj_qkvz,
# in_proj_ba) used to be duplicated here. It — and the namespace handling around
# it — now live once, in ``config`` (``_FUSED_FALLBACK`` +
# ``PrismaQuantConfig.shard_target_keys``), because this module's copy and the
# config's copy answered "which shards does this fused module have?" separately
# and drifted (issue #1).

# Native decode/prefill boundary. Both CB grids use their owned CUDA GEMV
# through eight activation rows. FP4-v2 product mode uses exact BF16 expansion
# plus Gridbook's owned CUTLASS grouped GEMM (with E=1) above the boundary.
# This is deliberately fixed rather than environment-selectable: an artifact's
# serving kernel family must not change underneath a throughput/quality run.
CUDA_GEMV_M_MAX = 8

# fp4-MMA fused prefill gate (OPT-IN, default OFF): "" = off (the shipping
# exact quality route remains the default); "1"/"midm" use the artifact's
# attested static activation scalar; ``static_lsq`` keeps that exact scalar and
# payload but fits the existing EVT residual independently per row; "rowwise"
# derives an independent native-NVFP4 scalar for every runtime row and therefore
# needs no serialized activation metadata. The ``*_midm`` modes cover only
# 16 < M <= 128. Every mode remains an explicit experiment until its served
# quality/performance gate passes.
_FP4_FUSED_MODE: list = []

_FP4_FUSED_STATIC_MODES = frozenset(("1", "midm"))
_FP4_FUSED_STATIC_LSQ_MODES = frozenset((
    "static_lsq", "static_lsq_midm",
))
_FP4_FUSED_ROWWISE_MODES = frozenset(("rowwise", "rowwise_midm"))
_FP4_FUSED_MODES = (
    _FP4_FUSED_STATIC_MODES
    | _FP4_FUSED_STATIC_LSQ_MODES
    | _FP4_FUSED_ROWWISE_MODES
)
_FP4_FUSED_ALLOWED_MODES = frozenset(("",)) | _FP4_FUSED_MODES


def _fp4_fused_mode() -> str:
    # Explicit opt-in: what this changes for a served artifact is the fp4
    # activation bucket moving from the exact BF16 quality route's fp32
    # emulation scales to the format's native ue4m3 scale factors. The fused
    # kernel is bit-exact against the native NVFP4 collective, but it is a
    # change in served numerics. The same-process A/B failed the equivalence
    # and dense-timing promotion screens, while representative
    # teacher/task/served-SLO evidence remains incomplete. See
    # docs/audits/fused_nvfp4_enablement_2026-07-31.md.
    current = os.environ.get("PRISMAQUANT_CB_FUSED_FP4", "").strip()
    if current not in _FP4_FUSED_ALLOWED_MODES:
        raise ValueError(
            "invalid PRISMAQUANT_CB_FUSED_FP4="
            f"{current!r}; expected '', '1', 'midm', 'static_lsq', "
            "'static_lsq_midm', 'rowwise', or 'rowwise_midm'"
        )
    if not _FP4_FUSED_MODE:
        _FP4_FUSED_MODE.append(current)
    elif current != _FP4_FUSED_MODE[0]:
        raise RuntimeError(
            "PRISMAQUANT_CB_FUSED_FP4 changed after Gridbook dispatch was "
            "fixed; restart the process instead of mixing activation contracts"
        )
    return _FP4_FUSED_MODE[0]


# FP8-CB's native CUDA GEMV handles M<=8. The boundary is measured on GB10
# (bench_cuda_gemv.py) and is part of the native dispatch contract, not a
# runtime selector: the weight-stationary GEMV wins through M=8 but loses at
# M=16. Larger M goes to native CUTLASS (fused when eligible, otherwise CUDA
# expand + direct CUTLASS W8A8).
FP8_CUDA_GEMV_M_MAX = 8

# The dense native-NVFP4 collective owns one 128-row N tile per CTA.  A fused
# projection may therefore use more than one interned codebook block as long as
# every tile resolves to exactly one block.  Keep this in lockstep with
# ``TileShapeFp4`` in csrc/cb_fused_fp4_gemm.cu; the extension independently
# validates the resulting tile map at its public boundary.
FP4_FUSED_TILE_N = 128
FP4_FUSED_TILE_M_WIDE = 256

# One entry per CUDA device. ``get_device_properties`` queries cached runtime
# metadata and does not synchronize the device; keeping the result here also
# removes that query from steady-state dispatch.  A failed query is cached as
# zero so shape selection fails closed to the long-standing TileM=128 kernel.
_FP4_DENSE_SM_COUNTS: dict[int, int] = {}


def _shard_degree(full_size: int, local_size: int) -> int:
    """The TP degree implied by vLLM's per-partition constructor arguments.

    vLLM hands ``create_weights`` BOTH the whole-tensor sizes and this rank's
    partition of them; their quotient is the serving degree, independent of
    any distributed state (and therefore well-defined in offline construction
    too). Non-divisible shapes cannot come from a healthy engine; the ceil
    keeps the reported degree usable in a refusal message anyway.
    """
    if local_size > 0 and full_size >= local_size:
        return max(1, -(-full_size // local_size))
    return 1


class ShardGroupAlignmentError(ValueError):
    """A tensor-parallel shard boundary would cut a CB group or alignment law.

    Raised at weight CONSTRUCTION — before any buffer exists and long before
    any byte is copied — by the dense CB loader's shard-legality gate. The
    exception IS the structured fact (walker-R5 invariant 5 as data): gates
    and reports read the fields below; ``str(exc)`` merely renders them for
    humans and never carries information the fields do not.

    Attributes:
        qname: quantized target (or fused role) the shard belongs to.
        axis: ``"input"`` (row-parallel, K sharded) or ``"output"``
            (column-parallel, N sharded).
        group_size: the alignment quantum the shard violates — 256
            (the packed superblock) on the input axis; the native kernel
            row-alignment quantum (8 fp4 / 16 fp8) or the sharding degree on
            the output axis.
        tp_degree: live serving degree the geometry implies.
        shard_size: this rank's extent along the offending axis.
        detail: machine-readable reason code sentence.
    """

    def __init__(self, *, qname: str, axis: str, group_size: int,
                 tp_degree: int, shard_size: int, detail: str) -> None:
        self.qname = qname
        self.axis = axis
        self.group_size = int(group_size)
        self.tp_degree = int(tp_degree)
        self.shard_size = int(shard_size)
        self.detail = detail
        super().__init__(
            f"{qname}: {axis}-axis shard of {shard_size} per rank at "
            f"TP={tp_degree} would split a {group_size}-wide alignment "
            f"group ({detail}); refusing at weight construction instead of "
            "serving a mis-sharded CB group"
        )


def _fp8_fused_midm_mode() -> str:
    """Return the process-stable FP8 fused-mid-M request.

    Unset means ``auto``: enable only on the exact Blackwell capabilities the
    fused extension compiles for.  ``0`` means off and ``1`` means require.
    Keeping ``1`` distinct from ``auto`` matters on Ada: an explicit request
    must fail at MODEL LOAD instead of silently substituting expand+CUTLASS,
    while the default must never ask the Blackwell-only loader to JIT there.
    """
    from .lane_select import latched_choice

    return latched_choice(
        "PRISMAQUANT_CB_FUSED_MIDM",
        spellings={"": "auto", "0": "off", "1": "require"},
        meaning="the FP8-CB fused mid-M decode-in-prologue lane",
    )


def _fp8_fused_midm_enabled(device: torch.device) -> bool:
    """Resolve the fused-mid-M request for ``device`` at model load.

    The optional fused implementation is an sm12x kernel.  Ada's native
    FP8-CB path is CUDA GEMV through M=8 and expand + native W8A8 CUTLASS above
    it; resolving ``auto`` to false here prevents a doomed ``get_fused_ext``
    build before that route can be reached.  Capability discovery is metadata
    only and performs no tensor read or device synchronization.
    """
    from .cuda_ext import NativeKernelUnavailableError
    from .lane_select import device_capability

    mode = _fp8_fused_midm_mode()
    if mode == "off":
        return False
    capability = device_capability(device)
    if capability in ((12, 0), (12, 1)):
        return True
    if mode == "require":
        got = ("unavailable" if capability is None
               else f"{capability[0]}.{capability[1]}")
        raise NativeKernelUnavailableError(
            "PRISMAQUANT_CB_FUSED_MIDM=1 requires the fused FP8-CB "
            "decode-in-prologue kernel on compute capability 12.0 or 12.1, "
            f"but the loading device reports {got}; disable the explicit "
            "request or use the native FP8 expand+CUTLASS route"
        )
    # Fail closed when CUDA cannot identify the device as an attested sm12x.
    return False


def _fp4_dense_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    try:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        index = int(index)
    except Exception:  # noqa: BLE001 - optional optimization, fail closed
        return 0
    cached = _FP4_DENSE_SM_COUNTS.get(index)
    if cached is not None:
        return cached
    try:
        count = int(torch.cuda.get_device_properties(index).multi_processor_count)
        if count <= 0:
            count = 0
    except Exception:  # noqa: BLE001 - optional optimization, fail closed
        count = 0
    _FP4_DENSE_SM_COUNTS[index] = count
    return count


def _fp4_dense_tile_m(M: int, N: int, sm_count: int) -> int:
    """Select the measured dense fused tile without under-filling the GPU.

    TileM=256 halves weight decode work per output row, but loses by about 20%
    when its grid has at most half of GB10's SM count.  Interleaved K24 A/Bs
    crossed over once the grid reached two thirds of the device: 32 CTAs on a
    48-SM GB10.  Keep that occupancy floor explicit and device-relative.
    """
    if M < FP4_FUSED_TILE_M_WIDE or sm_count <= 0:
        return 128
    grid_ctas = ((M + FP4_FUSED_TILE_M_WIDE - 1)
                 // FP4_FUSED_TILE_M_WIDE) * ((N + FP4_FUSED_TILE_N - 1)
                                               // FP4_FUSED_TILE_N)
    occupancy_floor = (2 * sm_count + 2) // 3
    return FP4_FUSED_TILE_M_WIDE if grid_ctas >= occupancy_floor else 128

# NOTE on a rejected variant (2026-07-18): N-chunking the transient expand +
# GEMM with side-stream overlap measured 0.46x (small-N GEMMs lose more than
# the overlap hides) AND is not bit-exact — cutlass_scaled_mm picks different
# configs per shape (split-K on narrow N), which breaks the served-KL
# contract. Prefill overlap needs cross-layer prefetch or the fused
# decode-in-prologue kernel, not N-chunking.


@register_weight_loader_v2_supported_method
class PrismaQuantCBLinearMethod(LinearMethodBase):
    def __init__(self, quant_config, scheme: dict, prefix: str) -> None:
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        self.has_static_fp4_activation = (
            self.is_fp4
            and scheme.get("activation_contract")
            == _NVFP4_ACTIVATION_CONTRACT_KEY
        )
        # Two-tier v2 scale coding (fp4 only) — absence of scale_coding ⇒ v1.
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        elif isinstance(sc, str):
            self.is_v2 = sc == codec.SCALE_CODING_TWO_TIER
            self._sub_table = codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = False
            self._sub_table = None
        if self.is_v2 and not self.is_fp4:
            raise ValueError(f"{prefix}: two-tier scale coding is fp4-only")

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        del params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        K = input_size_per_partition
        partitions = [int(width) for width in output_partition_sizes]
        rows = sum(partitions)
        self._require_shard_group_alignment(
            K, partitions, input_size=input_size, output_size=output_size)
        row_bytes = (K // codec.SUPERBLOCK) * self.type_size
        layer.logical_widths = partitions
        layer._cb_input_size = K
        # The serving degrees implied by vLLM's constructor arguments. Load
        # finalization re-derives rank-local role geometry from the output
        # degree; storing it here keeps that arithmetic on the same authority
        # as the construction gate above.
        layer._cb_input_tp_degree = _shard_degree(input_size, K)
        layer._cb_output_tp_degree = _shard_degree(output_size, rows)

        cb_qweight = ModelWeightParameter(
            data=torch.empty(rows, row_bytes, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("cb_qweight", cb_qweight)

        if getattr(self, "has_static_fp4_activation", False):
            # One slot per vLLM logical shard is the standard
            # compressed-tensors scalar ABI.  NaN initialization makes an
            # absent checkpoint tensor (or a missed shard) observable at the
            # load-finalization gate instead of becoming an arbitrary scale.
            input_global_scale = PerTensorScaleParameter(
                data=torch.full(
                    (len(output_partition_sizes),),
                    float("nan"),
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            layer.register_parameter(
                "input_global_scale", input_global_scale
            )

        if not self.is_fp4:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(rows, dtype=torch.float32),
                output_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight_scale", weight_scale)

    def _require_shard_group_alignment(self, K, partitions, *, input_size,
                                       output_size) -> None:
        """Refuse, at construction, any TP shard that would split a group.

        Two alignment laws, both evaluated on PER-RANK geometry (walker-R5
        invariant 5 as a structured gate, never prose):

        * ``input`` axis (row-parallel): a packed row is a chain of
          256-weight superblocks and codewords/scale planes never straddle
          one, so a K-shard must contain whole superblocks. When the WHOLE
          tensor already violates the producer's ``in_features % 256`` SPEC
          requirement, the artifact error is raised exactly as the unsharded
          path always raised it; only a legal whole tensor with an illegal
          shard gets the structured shard refusal.
        * ``output`` axis (column-parallel): packed rows are independent
          byte streams, but the native decode/prefill kernel families index
          rows in 8-wide (fp4) / 16-wide (fp8) quanta. At TP=1 this is
          enforced exactly where it has always lived — the post-load native
          attestation in ``process_weights_after_loading`` — so the
          single-device path is untouched; a sharded rank refuses HERE,
          before any parameter exists.
        """
        if K % codec.SUPERBLOCK != 0:
            if input_size % codec.SUPERBLOCK != 0:
                # The whole tensor is out of producer SPEC — the artifact
                # error, identical to the unsharded path (where K is the
                # full input size).
                raise ValueError(
                    f"{self.prefix}: in_features {input_size} not a multiple "
                    f"of {codec.SUPERBLOCK}")
            raise ShardGroupAlignmentError(
                qname=self.prefix, axis="input",
                group_size=codec.SUPERBLOCK,
                tp_degree=_shard_degree(input_size, K), shard_size=K,
                detail="row-parallel K-shard would cut a packed superblock")
        alignment = 8 if self.is_fp4 else 16
        misaligned = [width for width in partitions if width % alignment]
        if misaligned and sum(partitions) != output_size:
            raise ShardGroupAlignmentError(
                qname=self.prefix, axis="output", group_size=alignment,
                tp_degree=_shard_degree(output_size, sum(partitions)),
                shard_size=misaligned[0],
                detail=("column-parallel N-shard violates the native kernel "
                        f"row-alignment quantum; offending logical shards "
                        f"{misaligned} of {partitions}"))

    def _rank_local_role_widths(self, layer, shard_prefixes,
                                widths, ckpt_rows) -> list[int]:
        """Rank-local row width of every merged CB role.

        Checkpoint role rows are FULL-tensor counts while this rank received
        its ``1/t`` slice of every role independently (vLLM narrows each
        role at ``tp_rank * shard_size`` with per-rank coordinates), so the
        per-role boundary on THIS rank is ``checkpoint rows // output TP
        degree``. At degree 1 this is the identity — the exact widths the
        pre-TP code asserted.
        """
        full_role_rows = [self._lookup_ckpt_rows(sp, ckpt_rows)
                          for sp in shard_prefixes]
        degree = int(getattr(layer, "_cb_output_tp_degree", 1) or 1)
        local_widths: list[int] = []
        for sp, count in zip(shard_prefixes, full_role_rows):
            if count % degree:
                raise ShardGroupAlignmentError(
                    qname=f"{sp} (role of {self.prefix})", axis="output",
                    group_size=degree, tp_degree=degree, shard_size=count,
                    detail=("merged-role checkpoint row count does not "
                            "divide evenly across ranks; a rank-local role "
                            "boundary would fall inside the role"))
            local_widths.append(count // degree)
        assert sum(local_widths) == sum(widths), (
            f"{self.prefix}: rank-local checkpoint role rows {local_widths} "
            f"(full {full_role_rows} at TP={degree}; "
            f"sum {sum(local_widths)}) != logical width sum {sum(widths)}")
        return local_widths

    # -- shard-role resolution for a (possibly fused) layer -----------------
    def _shard_roles(self):
        """The ``target_scheme`` keys of this (possibly fused) module's CB
        roles, in shard order — the per-role codebooks
        ``process_weights_after_loading`` concatenates.

        Delegated to the config so this and ``_scheme_for_prefix`` can never
        disagree about which shards a fused module has, or about which
        namespace vintage the keys are in. ``unfused_fallback=True`` keeps this
        call site's semantics: a plain Linear is its own single role (the
        scheme lookup deliberately has no such rung). Serving prefixes may
        carry a class wrapper (``language_model.model.*``) OR the mapper's own
        namespace while the stored keys sit in another — both the 0.8B hybrid
        (resolved ZERO roles, died on the width assert) and issue #1 were that
        one question answered locally and wrongly."""
        return self.quant_config.shard_target_keys(self.prefix,
                                                   unfused_fallback=True)

    def _ckpt_cb_rows(self) -> dict[str, int]:
        """Row count of every ``*.cb_qweight`` tensor in the on-disk checkpoint
        (header-only read, cached once on the shared quant_config). Used to
        recover the per-ROLE row split of a vLLM-merged linear whose roles have
        separate codebooks (e.g. GDN ``in_proj_qkvz`` = ``in_proj_qkv`` +
        ``in_proj_z``); ``logical_widths`` does not expose that boundary."""
        qc = self.quant_config
        cache = getattr(qc, "_ckpt_cb_row_cache", None)
        if cache is not None:
            return cache
        import glob
        import json
        import struct
        model_dir, revision = qc._get_sidecar_source()
        if not os.path.isdir(model_dir):
            # Hub repo id: vLLM already downloaded the weights; reuse its
            # exact revision's snapshot dir instead of a raw path join.  The
            # shared config source also pins quant_config.json and the codebook
            # to this same immutable commit.
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                model_dir,
                revision=revision,
                allow_patterns=["*.safetensors"],
            )
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        cache = {}
        for st in files:
            with open(st, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name != "__metadata__" and name.endswith(".cb_qweight"):
                    cache[name] = int(meta["shape"][0])
        qc._ckpt_cb_row_cache = cache
        return cache

    def _lookup_ckpt_rows(self, shard_prefix: str, ckpt_rows: dict) -> int:
        """Rows of ``<shard_prefix>.cb_qweight`` in the checkpoint. shard_prefix
        is a vLLM-mapped name; the on-disk tensor keeps the export name, which
        differs only by the model-nesting prefix (Qwen3-VL: ``model.language_
        model.`` vs ``language_model.model.``). Match on the module-local tail
        (from ``.layers.`` on, which is nesting-invariant), requiring a unique
        hit so a mis-mapped name fails loudly rather than silently truncating."""
        want = shard_prefix + ".cb_qweight"
        if want in ckpt_rows:
            return ckpt_rows[want]
        cut = shard_prefix.rfind(".layers.")
        tail = (shard_prefix[cut:] if cut >= 0
                else "." + shard_prefix.split(".")[-1]) + ".cb_qweight"
        hits = [r for name, r in ckpt_rows.items() if name.endswith(tail)]
        assert len(hits) == 1, (
            f"{shard_prefix}: expected exactly one checkpoint cb_qweight ending "
            f"'{tail}', found {len(hits)} — cannot resolve merged-role row split")
        return hits[0]

    def _finalize_static_activation_scale(self, layer,
                                          shard_prefixes: list[str]) -> None:
        """Fail-closed load gate for the contracted dense scale parameter."""

        if not getattr(self, "has_static_fp4_activation", False):
            return
        if not hasattr(layer, "input_global_scale"):
            raise ValueError(
                f"{self.prefix}: contracted FP4-CB layer has no "
                "input_global_scale parameter"
            )
        expected_scales = self.quant_config.activation_scales_for_targets(
            shard_prefixes
        )
        layer._cb_fp4_input_global_scale = require_identical_loaded_scales(
            layer.input_global_scale.data,
            prefix=self.prefix,
            expected=expected_scales,
        )
        # The static-LSQ binding takes the already-attested F32 by value.  Keep
        # vLLM's device scalar above for the ordinary static ABI, while avoiding
        # a steady-state device-to-host read (and making malformed direct fixed-G
        # calls reject synchronously before kernel launch).
        layer._cb_fp4_input_global_scale_f32 = float(expected_scales[0])

    def process_weights_after_loading(self, layer):
        # vLLM stamps grouped-BMM geometry only after the Linear constructor
        # (and therefore after get_quant_method/create_weights).  A dense CB
        # method must not reinterpret [T,G,K] as T*G independent dense rows:
        # that would multiply every group by all G output blocks and return
        # [T,G,G*N] instead of the required [T,G,N].  Keep this load-time and
        # fail-closed until Gridbook owns a measured grouped-CB kernel.  The
        # released DeepSeek-V4 wo_a route is source FP8 W8A16, whose grouped
        # BMM contract is separately qualified.
        if bool(getattr(layer, "is_bmm", False)):
            raise RuntimeError(
                f"{self.prefix}: dense CB Linear does not implement grouped "
                "BMM semantics; declare this projection as source FP8 W8A16 "
                "instead of serving a shape-incorrect CB fallback"
            )
        dev = layer.cb_qweight.device
        codebooks = self.quant_config.get_codebooks()

        # Build the concatenated flat codebook + per-row base offset. The
        # per-row offset MUST cover every output row (its length == N), else the
        # decode/expand kernels read cb_row_offset out of bounds.
        shard_prefixes = self._shard_roles()
        self._finalize_static_activation_scale(layer, shard_prefixes)
        widths = list(layer.logical_widths)
        if len(shard_prefixes) == len(widths):
            # One CB role per logical shard: a genuinely fused module (qkv_proj /
            # gate_up_proj) with a per-role codebook, or a plain single Linear.
            # logical_widths is the authoritative per-shard row count.
            shard_widths = widths
        else:
            # vLLM merged MORE logical shards than we have CB roles: e.g. a
            # Gated-DeltaNet ``in_proj_qkvz`` (logical_widths=[q,k,v,z], 4 chunks)
            # that the export packed as TWO targets ``in_proj_qkv``(=q+k+v) +
            # ``in_proj_z``(=z), each with its own codebook. logical_widths does
            # NOT expose the per-ROLE boundary, so derive each role's RANK-LOCAL
            # row count from the checkpoint's separate ``<role>.cb_qweight``
            # tensor (vLLM merges them on load, but they are distinct on disk),
            # divided across ranks by this layer's output TP degree — at TP=1
            # that division is the identity and the widths are exactly what the
            # checkpoint holds. The old code used widths[0] for the single role
            # -> cb_row_offset was short -> illegal memory access in the
            # decode/expand kernels.
            ckpt_rows = self._ckpt_cb_rows()
            shard_widths = self._rank_local_role_widths(
                layer, shard_prefixes, widths, ckpt_rows)
        blocks, row_offsets = [], []
        # Fused projections commonly reuse the exact same shared codebook for
        # every role.  Keep one physical block per exact reference tuple and
        # point all matching roles at it.  Comparing references (rather than
        # tensor values) is deliberate: differently named codebooks remain
        # distinct even when their current contents happen to match.
        block_offsets: dict[tuple[str, ...], int] = {}
        block_ids: dict[tuple[str, ...], int] = {}
        block_ranges: list[tuple[int, int]] = []
        role_block_ids: list[tuple[int, int]] = []
        cb_cumoffset = 0
        for i, sp in enumerate(shard_prefixes):
            ref = self.quant_config.target_scheme[sp]["codebook_ref"]
            names = tuple(ref) if isinstance(ref, (list, tuple)) else (ref,)
            if names not in block_offsets:
                subs = [codebooks[n].to(dev) for n in names]
                flat = codec.build_flat_product_codebook(
                    subs, self.k, self.n_sub, self.prefix,
                    "fp4" if self.is_fp4 else "fp8")
                block_offsets[names] = cb_cumoffset
                block_ids[names] = len(blocks)
                block_ranges.append((cb_cumoffset, flat.numel()))
                blocks.append(flat)
                cb_cumoffset += flat.numel()
            w = shard_widths[i]
            # Exact-reference duplicates reuse their first block; distinct
            # references retain a cumulative base (rather than i*cb_total),
            # which remains correct when blocks differ in size.
            row_offsets.append(torch.full((w,), block_offsets[names],
                                          dtype=torch.int32, device=dev))
            role_block_ids.append((w, block_ids[names]))
        cb_flat = torch.cat(blocks).contiguous()
        cb_row_offset = torch.cat(row_offsets).contiguous()

        qw = layer.cb_qweight.data
        rows, row_bytes = int(qw.shape[0]), int(qw.shape[1])
        assert cb_row_offset.numel() == rows, (
            f"{self.prefix}: cb_row_offset has {cb_row_offset.numel()} rows but "
            f"the packed weight has {rows} — per-row offset must cover "
            "every output row (kernels index it by row).")
        layer._cb_flat = cb_flat
        layer._cb_row_offset = cb_row_offset
        # The native-FP4 decode-in-prologue kernel stages one value LUT per
        # 128-row N tile.  Preserve the interned block ranges so it can build
        # one packed LUT per unique reference (rather than re-materializing the
        # role copies), and derive a tiny tile->block identity tensor without a
        # CUDA->host synchronization.  A role boundary is harmless when both
        # sides share the same interned block; a boundary between DIFFERENT
        # blocks must land on a tile edge.  Otherwise fail closed to the exact
        # shipping path instead of decoding any row through the wrong LUT.
        layer._cb_fp4_lut_ranges = tuple(block_ranges)
        if self.is_fp4:
            # The exact BF16 quality bridge expands one physical codebook at a
            # time.  Preserve a coalesced host-side row plan so fused modules
            # with distinct role codebooks (for example A/B/A qkv shards) can
            # launch the native expander against the matching dictionary for
            # every contiguous row segment.  This metadata is derived entirely
            # from the loader contract; steady-state dispatch never reads a
            # device offset back to the host.
            quality_segments: list[tuple[int, int, int]] = []
            row_start = 0
            for width, block_id in role_block_ids:
                if (quality_segments
                        and quality_segments[-1][2] == block_id):
                    start, previous_width, _ = quality_segments[-1]
                    quality_segments[-1] = (
                        start, previous_width + width, block_id)
                else:
                    quality_segments.append((row_start, width, block_id))
                row_start += width
            assert row_start == rows
            layer._cb_fp4_quality_segments = tuple(quality_segments)

            row_block_ids: list[int] = []
            for width, block_id in role_block_ids:
                row_block_ids.extend([block_id] * width)
            tile_ids = []
            tile_safe = len(row_block_ids) == rows
            for start in range(0, rows, FP4_FUSED_TILE_N):
                tile = row_block_ids[start:start + FP4_FUSED_TILE_N]
                if not tile or any(block_id != tile[0] for block_id in tile):
                    tile_safe = False
                    break
                tile_ids.append(tile[0])
            layer._cb_fp4_fused_lut_ok = tile_safe
            layer._cb_fp4_lut_tile_ids = (
                torch.tensor(tile_ids, dtype=torch.int32, device=dev)
                if tile_safe else None)
        # The fp8 mid-M fused kernel has no row-offset input: every row uses
        # LUT base zero.  Derive and cache its safety fact without a CUDA->host
        # sync.  One interned block means every role necessarily points at the
        # first (zero-based) block; two or more blocks must use the offset-aware
        # exact native route.
        layer._cb_fp8_fused_lut_ok = len(blocks) == 1
        dummy = torch.zeros(1, dtype=torch.float32, device=dev)
        if self.is_fp4 and self.is_v2:
            # v2: NO resident fp32 plane (docs/SPEC.md §7, INV-1). The kernel
            # composes the E4M3 scales in-register from the packed 9 bytes via
            # this (256,16) table; the 9-byte plane stays inside cb_qweight.
            layer._cb_compose = codec.build_compose_table(self._sub_table).to(dev)
            layer._cb_scale = dummy
            # Warm the CUDA-GEMV JIT build at LOAD time — otherwise the ~30 s
            # in-container build fires on the first decode step and poisons the
            # first served request's latency (same reason as the fp8 branch
            # below).
            self._cuda_gemv_ok()
        elif self.is_fp4:
            # LEGACY FP4-CB v1. This branch used to materialize
            # ``decode_fp4_scale_plane(qw, k)`` — a DENSE fp32 ``[rows, K/16]``
            # tensor, held on the layer for its lifetime, which is precisely
            # the "resident per-superblock FP32 scale plane" docs/SPEC.md §7
            # INV-1 forbids by name. At K=12 it was as large as the packed
            # weight it accompanied.
            #
            # It was also read by nothing: every ``_cb_scale`` read in this
            # file sits in ``_apply_inline`` BELOW the ``if self.is_fp4``
            # branch, i.e. on the FP8 path. And ``_require_fp4_v2_product``
            # a few dozen lines below rejects this exact layout, so a v1 layer
            # allocated the plane and then failed the load anyway. Removing it
            # cannot change any behaviour that survives the load.
            layer._cb_scale = dummy
            layer._cb_compose = dummy
        else:
            layer._cb_scale = layer.weight_scale.data.reshape(-1).to(
                torch.float32)
            layer._cb_compose = dummy
            # E4M3-byte codebook for the fp8-direct transient expand and the
            # CUDA GEMV (exact: every codebook value is on the e4m3 grid, so
            # bf16 -> fp8 is a lossless re-encoding of the same table).
            layer._cb_flat_fp8 = codec.flat_codebook_fp8(
                cb_flat, self.prefix)
            # Warm the CUDA-GEMV JIT build at LOAD time — otherwise the ~30 s
            # in-container build fires on the first decode step and poisons
            # the first served request's latency (seen live: 1.89 tok/s rep 1).
            self._cuda_gemv_ok()

        # ONE resident copy of the packed weight (issue #1, 2026-07-25).
        #
        # ``pad_qweight`` allocates a padded COPY, and the registered
        # ``cb_qweight`` parameter used to stay live alongside it for the whole
        # serve — every dense CB weight was resident TWICE. Measured on the
        # shipped Qwen3.6-27B gridbook artifact (safetensors header): 15.07 GiB
        # of dense ``cb_qweight`` out of 21.38 GiB total, i.e. ~36.5 GiB of
        # weights served instead of ~21.4 GiB. It went unnoticed because the
        # reference box is a 128 GB unified-memory DGX Spark, where the slack
        # simply absorbed it; it only bites on 24/32 GB cards (reported from a
        # 32 GB RTX 5090). MoE stacks were never affected — ``moe.py`` pads only
        # bounded per-forward transients, never the resident expert stack.
        #
        # The fix is a narrow VIEW, not a ``del``: ``cb_qweight.data`` is still
        # read by the fp8 mid-M fused prefill entry,
        # both of which take the row stride explicitly and only require
        # ``stride(1) == 1`` and a 16-byte-multiple ``stride(0)`` — which the
        # 16-byte pad preserves (see ``codec.pad_qweight``). Dropping the local
        # ``qw`` then releases the original storage.
        layer._cb_qw_padded = codec.pad_qweight(qw)
        layer.cb_qweight.data = layer._cb_qw_padded.narrow(1, 0, row_bytes)
        del qw
        layer._cb_N = rows
        layer._cb_K = layer._cb_input_size

        # Production construction attests every native kernel that this layer
        # can reach. This intentionally builds/loads outside first forward and
        # outside CUDA-graph capture; an incomplete serving image fails while
        # the model is loading rather than silently changing kernel family.
        from .cuda_ext import (NativeKernelUnavailableError, get_fused_ext,
                               require_ext)
        required_n_alignment = 8 if self.is_fp4 else 16
        if rows % required_n_alignment:
            family = "BF16 grouped" if self.is_fp4 else "FP8 CUTLASS"
            raise NativeKernelUnavailableError(
                f"{self.prefix}: native {family} quality prefill requires "
                f"N divisible by {required_n_alignment}, got N={rows}")
        require_ext(f"{self.prefix} dense CB decode/QDQ/expansion")
        # PARSE EVERY DISPATCH FLAG, on every layer, regardless of format. A
        # typo must raise on an FP8-only model too: these selectors used to be
        # read only inside the ``is_fp4`` branch below, so on an FP8 serve
        # ``PRISMAQUANT_CB_FP4_FUSED_MIDM=ture`` was never parsed and the
        # operator got a silent baseline run instead of an error. Parsing is
        # separated from ACTING on the value, which stays per-format.
        fused_mode = _fp4_fused_mode()
        midm_requested = fp4v2_midm_requested()
        sm120_requested = bf16_sm120_requested()
        if self.is_fp4:
            layer._cb_fused_fp4_mode = fused_mode
            self._require_fp4_v2_product("model load")
            from .cuda_ext import (require_bf16_grouped_ext,
                                   require_fp4_v2_expander)
            require_fp4_v2_expander(
                f"{self.prefix} dense FP4-v2 expansion", device=dev)
            require_bf16_grouped_ext(
                f"{self.prefix} dense FP4-v2 quality prefill")
            # OPT-IN sm12x-native bridge lane. Resolved HERE, at model load:
            # the repo's rule is that nothing resolves at first forward, and
            # with the flag on an unavailable lane must fail the load rather
            # than silently serve the SM80 schedule (a different reduction
            # order than the one the operator asked to measure).
            layer._cb_bf16_sm120 = None
            if sm120_requested:
                layer._cb_bf16_sm120 = bf16_sm120_require_lane(
                    f"{self.prefix} dense FP4-v2 quality prefill", device=dev)
            # OPT-IN contract-preserving fused mid-M lane (audit §3 P2a):
            # decode the packed CB rows inside the CUTLASS prologue instead of
            # materializing the [N,K] BF16 transient in HBM. Resolved HERE for
            # the same two reasons as the lane above — nothing resolves at
            # first forward, and with the flag on an unavailable lane must fail
            # the LOAD rather than quietly serve the expand + bridge route,
            # which would answer a different question than the operator asked.
            layer._cb_fp4v2_midm = None
            if midm_requested:
                lane = fp4v2_midm_require_lane(
                    f"{self.prefix} dense FP4-v2 fused mid-M prefill",
                    device=dev)
                # PER-LAYER gate, at LOAD, exactly as the sibling persistent-B
                # lane does (moe.py). Attesting only that the EXTENSION exists
                # left every M-independent layer property — an uncompiled rung,
                # K not superblock-aligned, N not 8-aligned, a projection
                # spanning two interned dictionaries — to be discovered per
                # call and answered by silently serving expand + bridge for
                # every request, which is the substitution this flag exists to
                # forbid. The M band stays a call-time fall-through: that is a
                # property of the REQUEST, it is what makes this a mid-M lane,
                # and PLUGIN.md documents it.
                reason = fp4v2_midm_supports(
                    lane, layer, N=layer._cb_N, K=layer._cb_K, k_bits=self.k,
                    n_sub=self.n_sub, type_size=self.type_size,
                    is_v2=self.is_v2)
                if reason is not None:
                    from .cuda_ext import NativeKernelUnavailableError
                    raise NativeKernelUnavailableError(
                        f"{self.prefix}: requested the FP4-CB fused mid-M lane "
                        f"(PRISMAQUANT_CB_FP4_FUSED_MIDM=1), but this layer "
                        f"cannot use it ({reason}); Gridbook does not "
                        "substitute a different kernel behind an explicit lane "
                        "selection")
                layer._cb_fp4v2_midm = lane
            if fused_mode:
                rowwise = fused_mode in _FP4_FUSED_ROWWISE_MODES
                static_lsq = fused_mode in _FP4_FUSED_STATIC_LSQ_MODES
                if not self._fused_fp4_ok(
                    layer, layer._cb_K, rowwise=rowwise,
                    static_lsq=static_lsq,
                ):
                    from .cuda_ext import NativeKernelUnavailableError
                    raise NativeKernelUnavailableError(
                        f"{self.prefix}: requested native fused dense FP4 "
                        f"mode {fused_mode!r} is unavailable; changing to "
                        "the exact BF16 bridge would violate the explicit "
                        "activation contract"
                    )
        else:
            self._require_fp8_cuda_ext("model-load attestation")
            require_native_fp8_cutlass(
                f"{self.prefix} dense FP8 quality prefill")
            # The optional decode-in-prologue specialization is sm12x-only.
            # Resolve AUTO/explicit policy against the loading device before
            # touching its loader: on Ada AUTO is false and get_fused_ext must
            # never be called, so the native M>8 expand+CUTLASS route remains
            # reachable. On sm12x, get_fused_ext memoizes the load result, so
            # first prefill can neither JIT nor discover a different path.
            fused_midm = _fp8_fused_midm_enabled(dev)
            layer._cb_fp8_fused_midm = fused_midm
            if fused_midm:
                fused_ext = get_fused_ext()
                if (fused_ext is None
                        and _fp8_fused_midm_mode() == "require"):
                    raise NativeKernelUnavailableError(
                        f"{self.prefix}: PRISMAQUANT_CB_FUSED_MIDM=1 "
                        "requires the fused FP8-CB decode-in-prologue "
                        "extension, but it did not load; Gridbook does not "
                        "substitute expand+CUTLASS behind an explicit lane "
                        "requirement"
                    )
        from .ops import register_cb_layer
        layer._cb_layer_id = register_cb_layer(self, layer)

    def _cuda_gemv_ok(self) -> bool:
        """CUDA decode-GEMV eligibility (fp8 n_sub=4 rungs, or fp4 two-tier v2
        n_sub=2 rungs; native extension built). Cached — the answer never changes
        within a process."""
        ok = getattr(self, "_cuda_gemv_cached", None)
        if ok is None:
            fp8_ok = not self.is_fp4 and self.n_sub == 4
            fp4v2_ok = self.is_fp4 and self.is_v2 and self.n_sub == 2
            ok = fp8_ok or fp4v2_ok
            if ok:
                from .cuda_ext import get_ext
                ok = get_ext() is not None
            self._cuda_gemv_cached = ok
        return ok

    def _require_fp8_cuda_ext(self, operation: str) -> None:
        """Fail closed when an FP8-CB CUDA operation cannot be served.

        FP8-CB used to fall through to the Triton decode/expand kernels when
        ``get_ext()`` failed (or when the legacy decode selector disabled it).
        That made a nominal Gridbook serve silently run a different kernel
        family. Native CUDA is now part of the FP8-CB runtime contract, so a
        missing/disabled extension is an explicit error instead.
        """
        if self._cuda_gemv_ok():
            return
        from .cuda_ext import NativeKernelUnavailableError
        raise NativeKernelUnavailableError(
            f"{self.prefix}: FP8-CB {operation} requires Gridbook's native "
            "CUDA extension (n_sub=4), but it is unavailable "
            f"(n_sub={self.n_sub}). An alternate fallback is forbidden; fix the CUDA "
            "extension build before serving this artifact."
        )

    def _require_fp4_v2_product(self, operation: str) -> None:
        """Pin dense quality serving to the native FP4-v2 product layout.

        The legacy v1 layout has decode support in older kernels but no exact
        native BF16 expansion contract. Selecting a numerically different
        implementation for larger M would make one layer change format with
        batch size, so the entire dense serving lane rejects that layout
        until its native expansion is implemented.
        """
        if (self.is_fp4 and self.is_v2 and self.n_sub == 2
                and self.type_size == 4 * self.k + 9):
            return
        from .cuda_ext import NativeKernelUnavailableError
        raise NativeKernelUnavailableError(
            f"{self.prefix}: {operation} requires FP4-CB-v2 product layout "
            "(n_sub=2, type_size=4*k+9); legacy v1 layouts have no native "
            "quality-preserving dense prefill kernel")

    def _require_fp4_cuda_gemv(self) -> None:
        """Require the owned FP4-v2 product GEMV for M<=8."""
        self._require_fp4_v2_product("dense execution")
        if self._cuda_gemv_ok():
            return
        from .cuda_ext import NativeKernelUnavailableError
        raise NativeKernelUnavailableError(
            f"{self.prefix}: FP4-CB-v2 decode requires Gridbook's native "
            "CUDA GEMV, but it is unavailable; no alternate kernel is "
            "permitted")

    def _fused_fp8_lut_ok(self, layer) -> bool:
        """Whether the offset-free fp8 fused kernel can decode every row.

        ``cb_fused_prefill_mm_scaled`` receives the flat LUT but no per-row
        offsets, so it always addresses LUT base zero.  A uniform non-zero
        base is therefore not sufficient unless the caller also slices the
        LUT (which this path intentionally does not do).  Production layers
        cache this fact while interning blocks at load time; the tensor check
        is a defensive/testing fallback for manually constructed layers.  The
        format checks pin the four-subtable fp8 layout hard-coded by the CUDA
        prologue rather than relying on the shipped scheme menu implicitly.
        """
        ok = getattr(layer, "_cb_fp8_fused_lut_ok", None)
        if ok is None:
            ro = getattr(layer, "_cb_row_offset", None)
            ok = (ro is not None and ro.numel() == layer._cb_N
                  and ro.numel() > 0
                  and bool((ro == 0).all().item()))
            layer._cb_fp8_fused_lut_ok = ok
        return (not self.is_fp4
                and self.n_sub == 4
                and self.type_size == 4 * self.k
                and bool(ok))

    # -- fp4-MMA fused prefill (opt-in; see the dispatch site below) ---------
    def _fused_fp4_ok(self, layer, K: int, *, rowwise: bool = False,
                      static_lsq: bool = False) -> bool:
        """Eligibility for the fused fp4 block-scaled prefill (cached per
        layer). Mirrors the kernel's TORCH_CHECKs so a miss is a silent
        fall-through, never a crash. Static, static-LSQ, and rowwise activation
        families are cached separately so probing one cannot authorize another.
        """
        if rowwise and static_lsq:
            return False
        cache_attr = (
            "_cb_fp4_fused_rowwise_ok" if rowwise else
            "_cb_fp4_fused_static_lsq_ok" if static_lsq else
            "_cb_fp4_fused_static_ok"
        )
        ok = getattr(layer, cache_attr, None)
        if ok is not None:
            return ok
        ok = (self.is_fp4
              and (rowwise or (
                  getattr(self, "has_static_fp4_activation", False)
                  and getattr(layer, "_cb_fp4_input_global_scale", None)
                  is not None))
              and 12 <= self.k <= 24
              and self.n_sub == 2
              and K % 256 == 0
              and layer._cb_N % 8 == 0
              and self.type_size == 4 * self.k + (9 if self.is_v2 else 16))
        if ok and static_lsq:
            # This host scalar exists only after the producer contract payload
            # and loaded tensor bits have passed the fail-closed load gate.
            ok = getattr(
                layer, "_cb_fp4_input_global_scale_f32", None
            ) is not None
        if ok:
            ok = (codec.fp4_value_lut_nbytes(self.k, self.n_sub)
                  <= codec.FP4_FUSED_LUT_MAX_BYTES)
        if ok:
            # Every 128-row N tile must resolve to one interned codebook block.
            # Production layers cache this from shard metadata at load time.
            # The tensor fallback keeps manually constructed test layers safe;
            # it is evaluated once and cached, just like the old uniform-LUT
            # check, and therefore cannot add a sync to steady-state serving.
            ok = getattr(layer, "_cb_fp4_fused_lut_ok", None)
            if ok is None:
                ro = layer._cb_row_offset
                ok = (ro.numel() == layer._cb_N and ro.numel() > 0)
                tile_bases = []
                if ok:
                    for start in range(0, layer._cb_N, FP4_FUSED_TILE_N):
                        tile = ro[start:start + FP4_FUSED_TILE_N]
                        base = int(tile[0])
                        if not bool((tile == base).all().item()):
                            ok = False
                            break
                        tile_bases.append(base)
                if ok:
                    unique_bases = list(dict.fromkeys(tile_bases))
                    base_to_id = {base: i for i, base in enumerate(unique_bases)}
                    layer._cb_fp4_lut_tile_ids = torch.tensor(
                        [base_to_id[base] for base in tile_bases],
                        dtype=torch.int32, device=ro.device)
                layer._cb_fp4_fused_lut_ok = bool(ok)
            ok = bool(ok)
        if ok and not rowwise and not static_lsq:
            from .cuda_ext import NativeKernelUnavailableError
            try:
                require_native_fp4_quant(
                    f"{self.prefix} dense static NVFP4 quantization")
            except NativeKernelUnavailableError:
                ok = False
        if ok:
            from .cuda_ext import get_fused_fp4_ext
            fext = get_fused_fp4_ext()
            ok = (fext is not None
                  and hasattr(fext, "cb_fused_fp4_prefill_mm_scaled")
                  and (not rowwise
                       or hasattr(fext, "cb_nvfp4_quantize_rows"))
                  and (not static_lsq or hasattr(
                      fext, "cb_nvfp4_quantize_static_lsq")))
        setattr(layer, cache_attr, bool(ok))
        return bool(ok)

    def _try_fused_fp4(self, layer, x, N: int, K: int, M: int, *,
                       rowwise: bool = False, static_lsq: bool = False):
        """One fused NVF4 block-scaled GEMM over this layer's packed rows, or
        None if ineligible (the caller then runs the shipping path)."""
        if static_lsq:
            eligible = self._fused_fp4_ok(layer, K, static_lsq=True)
        else:
            # Preserve the historical call shape for instrumentation and
            # compatibility wrappers that know only the rowwise flag.
            eligible = self._fused_fp4_ok(layer, K, rowwise=rowwise)
        # K0.4 dense record for the fused NVFP4 lane. The lane carried only the
        # three TileM integers, so a report could see WHICH tile ran but not
        # whether the fused route ran at all — and the alternative serves a
        # DIFFERENT activation contract, which is the one distinction the
        # record exists to make. Tensor-free and sync-free, like the FP8 twin.
        route_common = dict(
            kind="dense",
            policy=(getattr(layer, "_cb_fused_fp4_mode", None)
                    or ("rowwise" if rowwise else
                        "static_lsq" if static_lsq else "1")),
            contract=_route_fused_fp4_contract(
                rowwise=rowwise, static_lsq=static_lsq),
            shape=f"M{int(M)}:N{int(N)}:K{int(K)}")
        if not eligible:
            emit_route(layer, symbol="", state="fallback", tile_m=0,
                       reason="fused fp4 dense eligibility gate declined",
                       **route_common)
            return None
        from .cuda_ext import get_fused_fp4_ext
        fext = get_fused_fp4_ext()
        lut = getattr(layer, "_cb_fp4_lut", None)
        ranges = getattr(layer, "_cb_fp4_lut_ranges", None)
        if not ranges:
            # Defensive compatibility for manually constructed single-LUT
            # layers. Production loaders always provide exact ranges.
            # If the LUT was injected directly (as in compatibility callers
            # and tests), the packed sidecar need not be present at all.
            ranges = ((0, (layer._cb_flat.numel() if lut is None
                           else lut.numel())),)
        if lut is None:
            dev = layer._cb_flat.device
            lut_bytes = codec.fp4_value_lut_nbytes(self.k, self.n_sub)
            lut_blocks = []
            for offset, length in ranges:
                block = layer._cb_flat.narrow(0, offset, length)
                block_lut = codec.build_fp4_value_lut(
                    block, self.k, self.n_sub).to(dev)
                if block_lut.numel() != lut_bytes:
                    raise RuntimeError(
                        f"{self.prefix}: FP4 fused LUT block has "
                        f"{block_lut.numel()} bytes, expected {lut_bytes}")
                lut_blocks.append(block_lut)
            lut = torch.cat(lut_blocks).contiguous()
            layer._cb_fp4_lut = lut
            layer._cb_fp4_compose_u8 = (
                codec.build_compose_u8(self._sub_table).to(dev) if self.is_v2
                else torch.zeros(1, dtype=torch.uint8, device=dev))
            layer._cb_fp4_ones = torch.ones(N, dtype=torch.float32, device=dev)
        x2 = x.reshape(-1, K)
        if rowwise:
            if x2.dtype not in (torch.bfloat16, torch.float16):
                emit_route(layer, symbol="", state="fallback", tile_m=0,
                           reason=(f"activation dtype {x2.dtype} is not half "
                                   f"precision; cb_nvfp4_quantize_rows "
                                   f"requires it"),
                           **route_common)
                return None
            # Full UE4M3 range per row. The extension returns the exact three
            # operands consumed by the existing fused GEMM; no weight decoder,
            # GEMM, or artifact representation is duplicated for this mode.
            aq, sfa, a_scales = fext.cb_nvfp4_quantize_rows(
                x2.contiguous(), _nvfp4_rowwise_range_multiplier()
            )
        elif static_lsq:
            if x2.dtype not in (torch.bfloat16, torch.float16):
                emit_route(layer, symbol="", state="fallback", tile_m=0,
                           reason=(f"activation dtype {x2.dtype} is not half "
                                   f"precision; "
                                   f"cb_nvfp4_quantize_static_lsq requires it"),
                           **route_common)
                return None
            # Keep the producer-attested G and its native E2M1/SFA payload.
            # The shared activation kernel changes only the existing per-row
            # EVT residual to the least-squares optimum for those fixed bytes.
            gs = layer._cb_fp4_input_global_scale_f32
            aq, sfa, a_scales = fext.cb_nvfp4_quantize_static_lsq(
                x2.contiguous(), gs
            )
        else:
            gs = layer._cb_fp4_input_global_scale
            aq, sfa = native_fp4_quant(x2, gs)
            a_scales = _nvfp4_reciprocal_vector(
                layer, which="dense", scale=gs, rows=M
            )
        # Preserve the original single-LUT launch contract (and its overlap of
        # LUT staging with the first A/SFA TMA). Only a genuinely multi-block
        # projection needs the per-N-tile indirection.
        lut_tile_ids = (layer._cb_fp4_lut_tile_ids
                        if len(ranges) > 1 else None)
        sm_count = _fp4_dense_sm_count(aq.device)
        tile_m = _fp4_dense_tile_m(M, N, sm_count)
        # Latest-route telemetry is intentionally tensor-free and sync-free.
        # Validation and serve logs can attest which concrete kernel ran,
        # while ordinary calls pay only integer arithmetic after the cached
        # device-property lookup.
        layer._cb_fp4_fused_tile_m = tile_m
        layer._cb_fp4_fused_tile_candidate_ctas = (
            ((M + FP4_FUSED_TILE_M_WIDE - 1) // FP4_FUSED_TILE_M_WIDE)
            * ((N + FP4_FUSED_TILE_N - 1) // FP4_FUSED_TILE_N)
        )
        layer._cb_fp4_fused_sm_count = sm_count
        route_common["tile_m"] = int(tile_m)
        route_common["tile_candidate_ctas"] = int(
            layer._cb_fp4_fused_tile_candidate_ctas)
        route_common["tile_sm_count"] = int(sm_count)
        route_common["tile_compiled"] = f"{FP4_FUSED_TILE_M_WIDE},128"
        emit_route(layer, symbol="cb_fused_fp4_prefill_mm_scaled",
                   state="error", reason="launch did not return",
                   **route_common)
        y = fext.cb_fused_fp4_prefill_mm_scaled(
            aq, sfa.view(torch.uint8).reshape(-1), layer._cb_qw_padded,
            layer._cb_fp4_lut, layer._cb_fp4_compose_u8, a_scales,
            layer._cb_fp4_ones, N, K, self.k, self.n_sub, self.type_size,
            self.is_v2, lut_tile_ids, tile_m)
        emit_route(layer, symbol="cb_fused_fp4_prefill_mm_scaled",
                   state="served", reason=None, **route_common)
        return y.reshape(*x.shape[:-1], N)

    def _expand_fp4_quality_weight(self, layer, N: int,
                                   K: int) -> torch.Tensor:
        """Expand every dense FP4 role through its matching native LUT.

        ``cb_expand_v2`` accepts one zero-based physical product dictionary.
        Fused vLLM modules may concatenate roles backed by different shared
        dictionaries, so the loader records contiguous row segments and this
        bridge expands each segment against the corresponding interned block.
        Concatenation stays on-device and produces the same contiguous BF16
        ``[N, K]`` transient consumed by the owned CUTLASS quality GEMM.
        """
        segments = getattr(layer, "_cb_fp4_quality_segments", None)
        ranges = getattr(layer, "_cb_fp4_lut_ranges", None)
        if not segments or not ranges:
            # Compatibility for manually constructed single-LUT layers. Every
            # production layer receives both metadata tuples at model load.
            return expand_fp4_v2_to_weight(
                layer._cb_qw_padded, layer._cb_flat,
                layer._cb_row_offset, layer._cb_compose,
                N, K, self.k, self.n_sub, self.type_size)

        expanded = []
        covered = 0
        for row_start, nrows, block_id in segments:
            if row_start != covered or nrows <= 0 or block_id >= len(ranges):
                raise RuntimeError(
                    f"{self.prefix}: invalid FP4 quality segment "
                    f"{(row_start, nrows, block_id)!r}")
            cb_start, cb_length = ranges[block_id]
            expanded.append(expand_fp4_v2_to_weight(
                layer._cb_qw_padded.narrow(0, row_start, nrows),
                layer._cb_flat.narrow(0, cb_start, cb_length),
                layer._cb_row_offset.narrow(0, row_start, nrows),
                layer._cb_compose,
                nrows, K, self.k, self.n_sub, self.type_size))
            covered += nrows
        if covered != N:
            raise RuntimeError(
                f"{self.prefix}: FP4 quality segments cover {covered} rows, "
                f"expected N={N}")
        return expanded[0] if len(expanded) == 1 else torch.cat(expanded, dim=0)

    def apply(self, layer, x, bias=None):
        # M-branch hoist (compile lane): dispatch through ONE opaque custom op
        # so torch.compile never traces the M-branch (which otherwise bakes
        # the prefill expand path into the decode graph — see ops.py). Eager
        # serving behavior is identical: the op impl calls _apply_inline.
        # This is mandatory: exposing the Python/ATen body to Inductor could
        # generate Triton, violating the native-only serving contract.
        from .cuda_ext import NativeKernelUnavailableError
        if bias is not None:
            raise NativeKernelUnavailableError(
                f"{self.prefix}: biased CB Linear has no opaque native "
                "Gridbook serving operator")
        lid = getattr(layer, "_cb_layer_id", None)
        if lid is None:
            raise NativeKernelUnavailableError(
                f"{self.prefix}: CB Linear was not registered during "
                "process_weights_after_loading")
        from .ops import cb_linear_forward
        return cb_linear_forward(x, lid)

    def _apply_inline(self, layer, x, bias=None):
        N, K = layer._cb_N, layer._cb_K
        M = x.reshape(-1, K).shape[0]

        if self.is_fp4:
            # Opt-in native NVFP4 prefill decodes packed CB rows in the CUTLASS
            # block-scaled prologue. Static, static-LSQ, and rowwise modes keep
            # the v0.4.2 activation contracts; the static quantizer now calls
            # the registered CUDA op directly. Explicit modes are attested at
            # model load; the selector call here only enforces the existing
            # same-process environment immutability contract.
            requested_mode = _fp4_fused_mode()
            fused_mode = getattr(
                layer, "_cb_fused_fp4_mode", requested_mode
            ) if bias is None and M > 16 else ""
            if (fused_mode in _FP4_FUSED_MODES
                    and not (fused_mode.endswith("midm") and M > 128)):
                mode_kwargs = {}
                if fused_mode in _FP4_FUSED_ROWWISE_MODES:
                    mode_kwargs["rowwise"] = True
                elif fused_mode in _FP4_FUSED_STATIC_LSQ_MODES:
                    mode_kwargs["static_lsq"] = True
                y = self._try_fused_fp4(
                    layer, x, N, K, M, **mode_kwargs)
                if y is not None:
                    return y
                # SAME rule as the MoE twin (moe.py::_apply_inline). The mode
                # was ATTESTED at model load, so a None here means the lane
                # declined this concrete CALL — in practice the rowwise /
                # static-LSQ activation quantizers' half-precision guard
                # (``_try_fused_fp4``'s dtype checks). Falling through would
                # serve the exact BF16 quality route, whose activation bucket
                # is the fp32-emulated group QDQ rather than the format's
                # native ue4m3 scale factors: a DIFFERENT served activation
                # contract than the one the operator explicitly selected, and
                # silently so. Gridbook does not substitute an activation
                # contract, so this is an error, exactly as it is for MoE.
                from .cuda_ext import NativeKernelUnavailableError
                raise NativeKernelUnavailableError(
                    f"{self.prefix}: requested native fused dense FP4 mode "
                    f"{fused_mode!r} became unavailable after model load "
                    f"(activation dtype {x.dtype}, M={M}, N={N}, K={K})")

            self._require_fp4_v2_product("dense execution")
            xq = fp4_act_qdq_or_codec(x)
            if M <= CUDA_GEMV_M_MAX:
                self._require_fp4_cuda_gemv()
                y = cb_gemv_fp4_v2(
                    xq, layer._cb_qw_padded, layer._cb_flat,
                    layer._cb_row_offset, layer._cb_compose,
                    N, K, self.k, self.n_sub, self.type_size)
            else:
                # OPT-IN fused mid-M lane (P2a). At 9 <= M <= 128 ONE M-tile
                # covers the batch, so decoding B inside the CUTLASS prologue
                # costs no redundant per-M-tile decode and the [N,K] BF16
                # transient never reaches HBM at all. CONTRACT-PRESERVING: the
                # decoded weights are bit-identical to
                # expand_fp4_v2_to_weight's and the activation is this same
                # xq, so only the FP32 reduction order differs. Unset (the
                # default) leaves the route below byte-for-byte unchanged.
                midm = getattr(layer, "_cb_fp4v2_midm", None)
                # K0.4: all three routes below serve the SAME activation
                # contract and differ only in GEMM schedule, so the symbol is
                # the only field that separates them in a report — which is
                # exactly what an opt-in schedule A/B needs to read back.
                quality_route = dict(
                    kind="dense", shape=f"M{int(M)}:N{int(N)}:K{int(K)}",
                    contract=_route_bridge_contract(True),
                    state="served", reason=None, tile_m=0)
                if midm is not None and fp4v2_midm_eligible(
                        midm, layer, M=M, N=N, K=K, k_bits=self.k,
                        n_sub=self.n_sub, type_size=self.type_size,
                        is_v2=self.is_v2):
                    y = fp4v2_midm_fused_mm(
                        midm, xq.reshape(M, K).contiguous(), layer,
                        N=N, K=K, k_bits=self.k)
                    emit_route(layer, policy="fp4v2_midm",
                               symbol="cb_fused_fp4v2_prefill_mm",
                               **quality_route)
                else:
                    # Exact FP4-CB-v2 -> BF16 expansion followed by the same
                    # owned device-scheduled CUTLASS grouped kernel as MoE
                    # quality prefill. E=1 gives a dense GEMM without routing
                    # metadata or a cuBLAS/F.linear fallback. The QDQ and
                    # expanded weight are bit-identical to the established
                    # quality reference; only FP32 accumulation order may
                    # differ.
                    W = self._expand_fp4_quality_weight(layer, N, K)
                    xq2 = xq.reshape(M, K).contiguous()
                    lane = getattr(layer, "_cb_bf16_sm120", None)
                    if lane is not None:
                        # OPT-IN sm12x-native lane: one expert, M padded up to
                        # a tile, every tile's expert id 0. Same operands, same
                        # single bf16 round; only the fp32 reduction order
                        # differs.
                        y = bf16_sm120_dense_mm(lane, xq2, W)
                        emit_route(
                            layer, policy="bf16_sm120",
                            symbol="cb_bf16_grouped_mm_sm120_gather",
                            **quality_route)
                    else:
                        expert_ends = torch.full(
                            (1,), M, dtype=torch.int32, device=x.device)
                        y = cb_bf16_grouped_mm(
                            xq2, W.unsqueeze(0), expert_ends, 0)
                        emit_route(layer, policy="bf16_grouped_bridge",
                                   symbol="cb_bf16_grouped_mm",
                                   **quality_route)
                    del W
                y = y.reshape(*x.shape[:-1], N)
            if bias is not None:
                y = y + bias
            return y

        # FP8-CB decode: the bandwidth-bound native CUDA GEMV owns M<=8. It
        # takes raw activations and fuses the dynamic FP8 QDQ. There is no
        # Triton fallback: running a different kernel family silently would
        # invalidate both the serving contract and any throughput comparison.
        if not self.is_fp4 and M <= FP8_CUDA_GEMV_M_MAX:
            route_shape = (
                f"FP8_CB_K{int(self.k)}:M{int(M)}:N{int(N)}:K{int(K)}"
            )
            emit_route(
                layer, kind="dense", policy="fp8_cb_cuda_gemv",
                symbol="cb_gemv_fp8", tile_m=0, shape=route_shape,
                contract=_route_bridge_contract(False), state="error",
                reason="launch did not return")
            self._require_fp8_cuda_ext("decode GEMV")
            y = cb_gemv_fp8(x, layer._cb_qw_padded, layer._cb_flat_fp8,
                            layer._cb_row_offset, layer._cb_scale,
                            N, K, self.k, self.n_sub, self.type_size)
            if bias is not None:
                y = y + bias
            emit_route(
                layer, kind="dense", policy="fp8_cb_cuda_gemv",
                symbol="cb_gemv_fp8", tile_m=0, shape=route_shape,
                contract=_route_bridge_contract(False), state="served",
                reason=None)
            return y

        # FP8_CB prefill (M large): transiently expand THIS layer's packed
        # weight into a native fp8 tile and call the directly registered
        # CUTLASS per-channel W8A8 op, then free the tile. An expanded
        # FP8_CB weight IS a standard per-channel fp8 checkpoint (codebook
        # values on the e4m3 grid; layer.weight_scale per output channel).
        # The expand writes the fp8 bytes directly (no bf16 intermediate, no
        # cast pass) — byte-identical to the old bf16-expand + cast, at a third
        # of the expand-side HBM traffic.
        # INV-1: the [N,K] tile is bounded to one layer (expand -> GEMM ->
        # free), never resident/model-wide (the NVINT2 OOM trap). `ops` is
        # imported lazily so the module still imports without vLLM (venv tests).
        x2 = x.reshape(-1, K)
        xq, sa = native_fp8_quant(x2)
        # scale_b is the per-output-channel weight scale as [N, 1] (matches
        # the standard per-channel fp8 contract; verified against a fp32 dequant
        # reference in tests/test_transient_fp8.py::test_transient_gemm_*).
        ws = layer._cb_scale.reshape(N, 1)

        # Mid-M fused decode-in-prologue: at M in [9, 128] ONE M-tile covers
        # the batch, so decoding B inside the CUTLASS prologue has no redundant
        # M-tile work. This native route now owns M=9..16 as well.
        # AUTO on sm12x, or explicitly requested there. Ada always uses CUDA
        # expand + direct CUTLASS for this band; an explicit fused request on
        # Ada was already refused at model load.
        # Numerics: the _scaled entry applies BOTH the per-token activation
        # scale and the per-channel weight scale inside its fp32 EVT epilogue
        # and rounds once to bf16 — the same rounding ORDER as
        # cutlass_scaled_mm (the older unscaled entry rounded first and scaled
        # in python, which moved served prompt logprobs by up to 0.86 nats).
        # RUNG COVERAGE (K1.2). The lane preserves the historical optimized
        # reader surface — codec.FP8_FUSED_KBITS, K28..K48 step 4 — independent
        # of the narrower K40/K44/K48 producer menu, and never a literal ladder
        # copied from the .cu (this used to be an inline
        # `self.k in (28, 32, 36, 40, 44, 48)`, one of two copies that could
        # each drift from the kernel independently). The law is the cheap gate
        # because asking the MODULE first would force a JIT build at first
        # forward for rungs that can never take this path; once the module is
        # in hand, `fused_fp8_kbits` is the authority and the law is only a
        # filter over it. Other reader rungs off the law are not
        # "unsupported" — they take the exact expand + CUTLASS route below.
        # ONE latched read (see ``_fp8_fused_midm_mode``), taken on the
        # dispatch path as well as at load so a mid-serve change RAISES instead
        # of being silently ignored. The value USED is the layer's, fixed at
        # load; this read exists to make the change loud. That was previously
        # three unlatched reads of the raw environment compared against each
        # other — the right instinct with the wrong mechanism, since the
        # comparison could only notice a change straddling those two lines, and
        # every one of the three accepted "false" and "off" as ENABLED because
        # they tested ``!= "0"``.
        _fp8_fused_midm_mode()
        fused_midm = bool(getattr(layer, "_cb_fp8_fused_midm", False))
        if (bias is None and FP8_CUDA_GEMV_M_MAX < x2.shape[0] <= 128
                and codec.fp8_fused_rung_supported(self.k)
                and fused_midm
                and self._fused_fp8_lut_ok(layer)):
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            if fp8_fused_rung_eligible(fext, self.k):
                # K0.4 dense route record. The dense mid-M FP8 lane had NO
                # telemetry at all (only the fp4 lane's three tile attributes),
                # so a served FP8 prefill was indistinguishable in a dispatch
                # report from one that quietly took the expand+GEMM route.
                shape = (f"FP8_CB_K{int(self.k)}:M{int(x2.shape[0])}:"
                         f"N{int(N)}:K{int(K)}")
                emit_route(layer, kind="dense", policy="fp8_cb_midm",
                           symbol="cb_fused_prefill_mm_scaled", tile_m=128,
                           shape=shape, contract="fp8_per_token_dynamic",
                           state="error", reason="launch did not return")
                y = fext.cb_fused_prefill_mm_scaled(
                    xq, layer.cb_qweight.data, layer._cb_flat_fp8,
                    sa.reshape(-1).to(torch.float32).contiguous(),
                    layer._cb_scale.reshape(-1).to(torch.float32).contiguous(),
                    N, K, self.k)
                emit_route(layer, kind="dense", policy="fp8_cb_midm",
                           symbol="cb_fused_prefill_mm_scaled", tile_m=128,
                           shape=shape, contract="fp8_per_token_dynamic",
                           state="served", reason=None)
                return y.reshape(*x.shape[:-1], N)

        # Native exact route for every FP8-CB shape the fused kernel cannot
        # take (including offset-bearing fused roles, bias, unsupported rungs,
        # a disabled/unavailable fused extension, and M>128). Gridbook's CUDA
        # extension expands the packed rows; the directly registered CUTLASS
        # op consumes the resulting [N,K] e4m3 tile. Missing CUDA support is
        # fatal, never a
        # hidden Triton byte-gather.
        shape = (f"FP8_CB_K{int(self.k)}:M{int(x2.shape[0])}:"
                 f"N{int(N)}:K{int(K)}")
        emit_route(
            layer, kind="dense", policy="fp8_cb_expand_cutlass_w8a8",
            symbol="cb_expand_fp8+_C.cutlass_scaled_mm", tile_m=0,
            shape=shape, contract=_route_bridge_contract(False),
            state="error", reason="native composition did not return")
        self._require_fp8_cuda_ext("weight expansion")
        from .ops import cb_expand_fp8
        W_e4m3 = cb_expand_fp8(
            layer._cb_qw_padded, layer._cb_flat_fp8,
            layer._cb_row_offset, N, K, self.k, self.n_sub,
            self.type_size)
        out = native_cutlass_scaled_mm(
            xq, W_e4m3.t(), sa, ws, torch.bfloat16)
        del W_e4m3
        emit_route(
            layer, kind="dense", policy="fp8_cb_expand_cutlass_w8a8",
            symbol="cb_expand_fp8+_C.cutlass_scaled_mm", tile_m=0,
            shape=shape, contract=_route_bridge_contract(False),
            state="served", reason=None)
        return out.reshape(*x.shape[:-1], N)
