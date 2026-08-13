"""SOURCE-format passthrough: the consumer half of a mixed Gridbook artifact.

A Gridbook artifact may ship some units as CB (our NVFP4-CB / FP8-CB codebook
vocabulary) and others as **verbatim copies of the source checkpoint's own
quantized tensors** — no requantization, no repacking, byte-for-byte.  Serving
such a unit means handing it back to the *native* vLLM method that already
understands that source format, exactly as if the stock checkpoint had been
loaded.

This module owns three things, and deliberately nothing else:

1. **The declaration schema** (:func:`parse_declaration`).  Which units pass
   through, and in which source format, is a producer-written fact in the
   artifact's ``quant_config.json`` — never inferred from tensor dtypes.  A
   loader that sniffs safetensors answers a different question than "what did
   the producer promise", and the two diverge exactly when it matters.
2. **The format registry** (:data:`FORMATS`).  One entry per source format we
   have *audited end to end on real hardware*: which vLLM method serves it,
   which backends that method may resolve to, and which device capabilities the
   audit covered.
3. **Fail-closed lookup.**  An unknown schema version, an unknown format id, an
   unaudited device, or a unit that is both CB and passthrough is a load-time
   refusal.  There is no environment-variable bypass, for the same reason
   ``delegated_preflight`` has none: an escape hatch here reintroduces the
   silent degradation the policy exists to prevent.

**Absence means legacy.**  An artifact with no declaration behaves exactly as
before this module existed — all-CB plus the pre-existing stock
compressed-tensors delegation.  Every published artifact predates the schema,
so absence must stay a no-op rather than becoming a new failure mode.

**Why a registry rather than a predicate.**  Gridbook's headline claim is a
native CUDA/CUTLASS operator lane.  "Does vLLM accept this format" and "does
vLLM execute it on a native kernel *on this device*" are different questions,
and vLLM 0.24 answers them differently for the very format that motivated this
module: its MXFP4 MoE oracle selects ``DEEPGEMM_MXFP4`` by default on the whole
sm12x family, and that rung then dies inside DeepGEMM's scale-layout transform
on sm_121.  A selector predicate would have called that a pass.  The tables
below are therefore hand-audited *outcomes*, and anything absent from them is
UNKNOWN — which fails.

This module is standard-library-only at import time on purpose: the schema and
the policy can be unit-tested on CPU with neither torch nor vLLM present.  The
one vLLM touchpoint, :func:`build_delegated_method`, imports lazily inside the
call.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple


__all__ = [
    "SourcePassthroughError",
    "SCHEMA_KEY",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SourceFormat",
    "FORMATS",
    "parse_declaration",
    "format_for",
    "require_audited_device",
    "build_delegated_method",
]


class SourcePassthroughError(ValueError):
    """A passthrough declaration Gridbook refuses to serve."""


#: Top-level key in ``quant_config.json`` carrying the declaration.
SCHEMA_KEY = "source_passthrough"

#: Schema versions this build understands.  A newer artifact must fail loudly
#: rather than be read with older rules — a version bump is how the producer
#: says "the meaning changed".
SUPPORTED_SCHEMA_VERSIONS = (1,)


class SourceFormat(NamedTuple):
    """One audited source format and the native route that serves it."""

    #: Stable id written by the producer.  Never reuse an id for new semantics.
    id: str
    #: What kind of unit carries this format: ``"moe_experts"`` or ``"linear"``.
    unit_kind: str
    #: Human description, quoted verbatim into refusal messages.
    description: str
    #: ``(major, minor)`` capabilities the native route was audited on.
    audited_capabilities: tuple[tuple[int, int], ...]
    #: Backend labels (``Enum.MEMBER`` or bare/qualified class name) audited to
    #: execute this format on a native non-Triton kernel.
    audited_backends: frozenset[str]
    #: Backends measured to FAIL for this format, mapped to the measured
    #: symptom.  Present so the refusal names the real problem and its fix
    #: instead of a generic "unaudited backend".
    known_broken_backends: Mapping[str, str]
    #: Whether the format's declared contract quantizes activations.  MXFP4
    #: expert weights with BF16 activations declare W4A16, so a weight-only
    #: backend honors the declaration rather than silently rewriting it.
    quantizes_activations: bool
    #: Operator-facing hint appended to refusals: how to reach the audited rung.
    remedy: str
    #: Import path of the vLLM method that serves this format, resolved lazily.
    method_factory: str


# --- the audited registry ----------------------------------------------------
#
# Audited against vLLM 0.24.0 (torch 2.11.0+cu130) on NVIDIA GB10, compute
# capability (12, 1).  Each entry records a MEASURED outcome — a real layer
# built, its weights processed and a forward profiled on the device — not a
# reading of vLLM's selector predicates, which disagree with the measurement
# for the very first format below.

_MXFP4_EXPERTS = SourceFormat(
    id="mxfp4_e2m1_ue8m0_g32",
    unit_kind="moe_experts",
    description=(
        "source MXFP4 routed experts: E2M1 nibble pairs packed two-per-byte "
        "along K, with one F8_E8M0 (UE8M0) exponent scale per 32 logical K "
        "elements"
    ),
    audited_capabilities=((12, 1),),
    # MARLIN is vLLM's `moe_wna16_marlin_gemm` CUDA kernel
    # (`marlin_moe_wna16::Marlin<...>`); a profiled forward on sm_121 showed it
    # plus `vllm::act_and_mul_kernel`, `moe_align_block_size` and `topkGating`,
    # and zero Triton kernels.
    audited_backends=frozenset({
        "Mxfp4MoeBackend.MARLIN",
        "MarlinExperts",
    }),
    known_broken_backends={
        # vLLM 0.24's DSV4 MXFP4 priority list puts this rung ahead of Marlin
        # and its device gate admits the whole sm12x family
        # (`is_device_capability_family(120)`), so it is what `auto` picks here.
        # It then raises inside DeepGEMM while repacking the block scales.
        "Mxfp4MoeBackend.DEEPGEMM_MXFP4": (
            "vLLM 0.24 selects this rung by default on sm12x, but DeepGEMM's "
            "scale-layout transform rejects sm_121: "
            "process_weights_after_loading raises "
            "'Assertion error (csrc/apis/layout.hpp:59): Unknown SF "
            "transformation'"
        ),
        "DeepGemmFP4Experts": (
            "DeepGEMM's MXFP4 scale repack does not support sm_121 "
            "(layout.hpp:59 'Unknown SF transformation')"
        ),
    },
    quantizes_activations=False,
    remedy=(
        "Pin the audited rung for this device with "
        "--kernel-config '{\"moe_backend\":\"marlin\"}' (equivalently "
        "--moe-backend marlin). Setting VLLM_USE_DEEP_GEMM=0 also reaches it, "
        "but is a process-wide switch that changes unrelated FP8 paths."
    ),
    method_factory="gridbook.source_passthrough:_build_vllm_mxfp4_moe_method",
)

_FP8_BLOCK_LINEAR = SourceFormat(
    id="fp8_e4m3_ue8m0_block128",
    unit_kind="linear",
    description=(
        "source block-quantized FP8 linear weights: F8_E4M3 values with one "
        "F8_E8M0 (UE8M0) exponent scale per 128x128 block"
    ),
    audited_capabilities=((12, 1),),
    # The route is GRIDBOOK-OWNED: every rung in vLLM 0.24's block-scaled FP8
    # linear ladder fails for this UE8M0 wire on sm_121 (recorded below).
    # Gridbook keeps the source E4M3 + block128 UE8M0 planes verbatim and
    # serves BF16 activations through a raw-plane native decode GEMV or a
    # caller-scoped BF16 transient consumed by its CUTLASS bridge.  Release
    # evidence is pending; admission is intentionally limited to Spark.
    audited_backends=frozenset({"Fp8SourceW8A16LinearMethod"}),
    known_broken_backends={
        "DeepGemmFp8BlockScaledMMKernel": (
            "DeepGEMM's UE8M0 scale-layout transform rejects sm_121 "
            "(layout.hpp:59 'Unknown SF transformation')"
        ),
        "CutlassFp8BlockScaledMMKernel": (
            "torch.ops._C.cutlass_scaled_mm has no sm_121 block-scaled "
            "dispatch (scaled_mm_helper.hpp:17 'dispatch_scaled_mm')"
        ),
        "TritonFp8BlockScaledMMKernel": (
            "Triton 3.6.0 cannot bind the float8_e8m0fnu scale dtype "
            "(KeyError 'float8_e8m0fnu'), and a Triton lane is refused by "
            "Gridbook doctrine regardless"
        ),
        "FlashInferFp8BlockScaledMMKernel": (
            "gated on is_device_capability(90) exactly, and the underlying "
            "FlashInfer entry point is named fp8_blockscale_gemm_sm90"
        ),
        "MarlinFP8ScaledMMLinearKernel": (
            "self-excludes above sm_89 unless VLLM_TEST_FORCE_FP8_MARLIN=1, "
            "and its can_implement does not verify UE8M0 128x128 handling"
        ),
    },
    quantizes_activations=False,
    remedy=(
        "No native block-scaled FP8 kernel serves UE8M0 scales on sm_121 in "
        "vLLM 0.24; Gridbook's raw-resident W8A16 route preserves the source "
        "planes and BF16 activations. Use the pinned Gridbook runtime carrying "
        "source_fp8_block128_w8a16 ABI feature 1, export the unit as Gridbook "
        "FP8-CB, or serve it on an admitted device."
    ),
    method_factory=(
        "gridbook.source_passthrough:"
        "_build_gridbook_fp8_source_w8a16_method"
    ),
)

_MXFP8_LINEAR = SourceFormat(
    id="mxfp8_e4m3_e8m0_g32",
    unit_kind="linear",
    description=(
        "MXFP8 linear weights: F8_E4M3 values with one F8_E8M0 (UE8M0) "
        "exponent scale per 32 contiguous K elements, scales stored row-major"
    ),
    audited_capabilities=((12, 1),),
    # Distinct from the block-128 W8A16 route above: this direct per-32 wire
    # deliberately enters the dynamically quantized W8A8 MXFP8 collective.
    audited_backends=frozenset({"Mxfp8DenseLinearMethod"}),
    known_broken_backends={},
    quantizes_activations=True,
    remedy=(
        "Set GRIDBOOK_MXFP8_DENSE=1 to opt in to Gridbook's MXFP8 dense lane "
        "(correctness-audited; serve-parity bench pending)."
    ),
    method_factory="gridbook.source_passthrough:_build_gridbook_mxfp8_direct_method",
)

#: format id -> audited route.  A producer id absent from this mapping is a
#: hard refusal: an unknown format cannot be given a native route by guessing.
FORMATS: dict[str, SourceFormat] = {
    _MXFP4_EXPERTS.id: _MXFP4_EXPERTS,
    _FP8_BLOCK_LINEAR.id: _FP8_BLOCK_LINEAR,
    _MXFP8_LINEAR.id: _MXFP8_LINEAR,
}


def format_for(format_id: Any, *, unit: str | None = None) -> SourceFormat:
    """Look up an audited format, or refuse.

    ``unit`` only decorates the message; resolution is by id alone.
    """

    where = f" declared for {unit!r}" if unit else ""
    if not isinstance(format_id, str) or not format_id:
        raise SourcePassthroughError(
            f"source-passthrough format{where} must be a nonempty string, got "
            f"{format_id!r}"
        )
    try:
        return FORMATS[format_id]
    except KeyError:
        known = ", ".join(sorted(FORMATS))
        raise SourcePassthroughError(
            f"unknown source-passthrough format {format_id!r}{where}. This "
            f"build audits only: {known}. Gridbook serves a native "
            f"CUDA/CUTLASS operator lane and refuses a format whose native "
            f"route it has not measured; there is no environment-variable "
            f"bypass. Either export the unit in an audited format, or audit "
            f"the new one and extend FORMATS in gridbook/source_passthrough.py."
        ) from None


# --- the declaration ---------------------------------------------------------


def parse_declaration(
    config: Mapping[str, Any],
    *,
    canonicalize: Callable[[str], str] | None = None,
    cb_targets: frozenset[str] | set[str] = frozenset(),
) -> dict[str, SourceFormat]:
    """Read the versioned passthrough map, or return ``{}`` for a legacy artifact.

    The shape, under ``quant_config.json``::

        "source_passthrough": {
          "version": 1,
          "units": {
            "model.layers.7.mlp.experts": "mxfp4_e2m1_ue8m0_g32",
            "model.layers.9.mlp.experts": "mxfp4_e2m1_ue8m0_g32"
          }
        }

    Every failure mode below is a refusal rather than a default, because each
    one is a producer/consumer disagreement about what the bytes mean:

    * absent key -> ``{}`` (legacy; the ONLY silent path, and it changes nothing);
    * present but not an object, or missing/unknown ``version`` -> refuse;
    * ``units`` absent, not an object, or empty -> refuse (an artifact that
      declares the key means to use it; an empty map is a producer bug, not a
      request for legacy behaviour);
    * a unit name that is not a nonempty string -> refuse;
    * a format id absent from :data:`FORMATS` -> refuse;
    * a unit that is ALSO a CB target -> refuse: the two vocabularies would
      each claim the same tensors, and whichever won would be an accident of
      dispatch order.
    """

    raw = config.get(SCHEMA_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SourcePassthroughError(
            f"{SCHEMA_KEY!r} must be an object with 'version' and 'units', got "
            f"{type(raw).__name__}"
        )

    if "version" not in raw:
        raise SourcePassthroughError(
            f"{SCHEMA_KEY!r} is missing 'version'. Gridbook refuses to read an "
            f"unversioned passthrough declaration, because a later schema "
            f"change would then be silently reinterpreted under older rules."
        )
    version = raw["version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise SourcePassthroughError(
            f"{SCHEMA_KEY!r} declares schema version {version!r}, which this "
            f"Gridbook build does not understand (supported: {supported}). "
            f"Upgrade Gridbook to serve this artifact; Gridbook does not "
            f"guess at a schema it was not built for."
        )

    units = raw.get("units")
    if not isinstance(units, Mapping):
        raise SourcePassthroughError(
            f"{SCHEMA_KEY!r}.units must be an object mapping unit prefix -> "
            f"format id, got {type(units).__name__}"
        )
    if not units:
        raise SourcePassthroughError(
            f"{SCHEMA_KEY!r}.units is empty. Omit {SCHEMA_KEY!r} entirely for "
            f"an all-CB artifact; an empty map is a producer error rather than "
            f"a request for legacy behaviour."
        )

    resolved: dict[str, SourceFormat] = {}
    for raw_unit, raw_format in units.items():
        if not isinstance(raw_unit, str) or not raw_unit:
            raise SourcePassthroughError(
                f"{SCHEMA_KEY!r}.units key must be a nonempty string, got "
                f"{raw_unit!r}"
            )
        unit = canonicalize(raw_unit) if canonicalize is not None else raw_unit
        fmt = format_for(raw_format, unit=raw_unit)
        if unit in cb_targets:
            raise SourcePassthroughError(
                f"unit {raw_unit!r} is declared BOTH as a CB target and as "
                f"source-passthrough {fmt.id!r}. One unit has one meaning; "
                f"remove it from whichever vocabulary does not describe the "
                f"tensors the artifact actually stores."
            )
        previous = resolved.get(unit)
        if previous is not None and previous.id != fmt.id:
            raise SourcePassthroughError(
                f"unit {raw_unit!r} resolves to serving prefix {unit!r}, which "
                f"is already declared as source-passthrough {previous.id!r}; "
                f"the two declarations disagree."
            )
        resolved[unit] = fmt
    return resolved


# --- device attestation ------------------------------------------------------


def require_audited_device(
    fmt: SourceFormat, *, prefix: str, capability: tuple[int, int] | None
) -> None:
    """Refuse a passthrough unit on a device the format was not audited on.

    The native route is a *measured* property of a (format, device) pair, not
    of the format alone: the same vLLM build that serves MXFP4 experts on a
    Blackwell consumer part picks a rung on other Blackwell parts that dies in
    DeepGEMM.  Attesting the device here, at model load, is what keeps that
    from becoming a first-forward surprise.
    """

    audited = ", ".join(f"sm_{major}{minor}"
                        for major, minor in fmt.audited_capabilities)
    if capability is None:
        raise SourcePassthroughError(
            f"source-passthrough unit {prefix!r} declares {fmt.id!r}, but "
            f"Gridbook could not read this device's compute capability, so it "
            f"cannot attest that the audited native route applies. The format "
            f"is audited for {audited}."
        )
    if tuple(capability) not in fmt.audited_capabilities:
        raise SourcePassthroughError(
            f"source-passthrough unit {prefix!r} declares {fmt.id!r}, which "
            f"Gridbook has audited only on {audited}; this device reports "
            f"sm_{capability[0]}{capability[1]}. The native backend a format "
            f"resolves to is device-dependent, so an unaudited device is "
            f"UNKNOWN rather than assumed-good. Audit this device and extend "
            f"audited_capabilities in gridbook/source_passthrough.py."
        )


# --- native method construction ----------------------------------------------


def _build_vllm_mxfp4_moe_method(layer: Any, prefix: str) -> Any:
    """vLLM's own MXFP4 MoE method, constructed exactly as vLLM would.

    ``Mxfp4MoEMethod`` creates ``w13_weight`` / ``w13_weight_scale`` /
    ``w2_weight`` / ``w2_weight_scale`` as ``uint8`` in the source packing, so
    an artifact that copied the checkpoint's expert tensors verbatim loads
    through vLLM's stock weight loader untouched — which is the whole point of
    a passthrough unit.
    """

    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    return Mxfp4MoEMethod(layer.moe_config)


def _require_mxfp8_opt_in(fmt_id: str, prefix: str) -> None:
    """The OPT-IN gate for the Gridbook-owned MXFP8 dense lane.

    Correctness parity (kernel vs fp32 oracle over the real body shapes) is
    audited; NATIVE-PARITY *served* evidence is pending, and Gridbook does not
    default-enable a lane on correctness evidence alone.  With the flag unset
    the unit refuses at load, with the flag named — the same fail-closed
    shape as every other lane, just one rung earlier in its promotion life.
    """

    from .mxfp8_dense_lane import MXFP8_DENSE_FLAG, mxfp8_dense_enabled

    if not mxfp8_dense_enabled():
        raise SourcePassthroughError(
            f"source-passthrough unit {prefix!r} declares {fmt_id!r}, served "
            f"by Gridbook's OPT-IN MXFP8 dense lane, and {MXFP8_DENSE_FLAG} "
            f"is not set. The lane is correctness-audited on sm_121 but its "
            f"serve-parity bench is pending, so it does not enable itself. "
            f"Set {MXFP8_DENSE_FLAG}=1 to serve this unit."
        )


def _build_gridbook_fp8_source_w8a16_method(layer: Any, prefix: str) -> Any:
    """Gridbook's raw-resident W8A16 route for DeepSeek block-128 FP8."""

    del layer, prefix
    from .fp8_source_w8a16 import (
        WIRE_FP8_BLOCK128,
        build_fp8_source_w8a16_method,
    )

    return build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)


def _build_gridbook_mxfp8_direct_method(layer: Any, prefix: str) -> Any:
    """Gridbook's MXFP8 dense lane reading the native per-32 wire."""

    _require_mxfp8_opt_in(_MXFP8_LINEAR.id, prefix)
    from .mxfp8_dense_lane import WIRE_MXFP8_G32, build_mxfp8_dense_method

    return build_mxfp8_dense_method(WIRE_MXFP8_G32)


_FACTORIES: dict[str, Callable[[Any, str], Any]] = {
    "gridbook.source_passthrough:_build_vllm_mxfp4_moe_method":
        _build_vllm_mxfp4_moe_method,
    ("gridbook.source_passthrough:"
     "_build_gridbook_fp8_source_w8a16_method"):
        _build_gridbook_fp8_source_w8a16_method,
    "gridbook.source_passthrough:_build_gridbook_mxfp8_direct_method":
        _build_gridbook_mxfp8_direct_method,
}


def build_delegated_method(fmt: SourceFormat, layer: Any, prefix: str) -> Any:
    """Construct the native vLLM method that serves *fmt*.

    Resolution is through an explicit table rather than ``importlib`` on the
    stored string: the registry is data, and data must not be able to name an
    arbitrary callable.
    """

    try:
        factory = _FACTORIES[fmt.method_factory]
    except KeyError:  # pragma: no cover - registry/table drift
        raise SourcePassthroughError(
            f"source format {fmt.id!r} names method factory "
            f"{fmt.method_factory!r}, which is not registered in "
            f"gridbook/source_passthrough.py"
        ) from None
    return factory(layer, prefix)
