"""Fail-closed preflight for **delegated** compressed-tensors groups (D0.2).

A mixed Gridbook artifact keeps its CB groups and hands every remaining
``compressed-tensors`` group back to vLLM (see ``config.py`` and
[`docs/DELEGATED-NVFP4-MOE.md`](../docs/DELEGATED-NVFP4-MOE.md)).  vLLM then
runs its own backend ladder, and two of its rungs break promises Gridbook makes
on the tin:

* **Marlin silently rewrites the declared contract.** Its NVFP4 selector reads
  the weight key and ignores the activation key, so a group that declares W4A4
  passes selection and is then converted to ``nvfp4_w4a16_moe_quant_config``
  with both activation-scale arguments set to ``None``.  The run is weight-only
  W4A16 arithmetic reported as if the declaration had been honored.
* **``emulation`` is Triton.**  Its experts class derives from vLLM's
  ``TritonExperts``.  Gridbook's headline claim is a native CUDA/CUTLASS
  operator lane; a delegated group dispatching Triton would make that claim
  true only by absence of such an artifact.

No published Gridbook artifact has a delegated group in either state, so both
are *latent* today.  This module makes them impossible rather than unlikely:
the policy is evaluated at model load, the moment vLLM hands back a resolved
method, and it **fails closed**.  There is deliberately no environment-variable
bypass — an escape hatch here would reintroduce exactly the silent degradation
the policy exists to prevent.

Design notes, in the order they matter:

1. **The declaration is the config group, not the tensors.**  Runtime parameter
   creation can allocate or synthesize scale tensors, so a preflight that scans
   safetensors names answers a different question.  Callers pass the resolved
   group dict.
2. **Backend semantics are a versioned table, not an inferred predicate.**
   ``docs/DELEGATED-NVFP4-MOE.md`` is explicit about this: a selector predicate
   (``_supports_quant_scheme``) is not a runtime semantics contract, and a
   future backend can accept a declaration and still transform its scales.  The
   tables below are therefore hand-audited names, and anything absent from them
   is **UNKNOWN**, which for an activation-quantized declaration means "fail".
3. **Triton is detected structurally *and* by name.**  The structural test —
   the ``triton`` token anywhere in the resolved class's module path or in any
   name along its MRO — catches ``Nvfp4QuantizationEmulationTritonExperts``
   (whose base is ``…experts.triton_moe.TritonExperts``) and the
   ``scaled_mm/triton.py`` linear kernels without naming either.  The versioned
   set catches the enum spelling (``EMULATION``), which carries no such token.

This module is standard-library-only on purpose: it imports neither torch nor
vLLM, so the policy can be unit-tested on CPU against stub backend classes that
mirror the vLLM shapes described above.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping, NamedTuple


__all__ = [
    "DelegatedBackendError",
    "declared_contract",
    "require_native_delegated_backend",
    "require_native_passthrough_backend",
]


class DelegatedBackendError(ValueError):
    """A delegated group resolved to a backend Gridbook refuses to serve."""


# --- versioned backend tables ------------------------------------------------
#
# Audited against the two vLLM builds named in docs/DELEGATED-NVFP4-MOE.md
# (`0.23.1rc1.dev764+g54b16d8a9` and `0.24.0`).  Names are matched against both
# the fully-qualified ``module.QualName`` and the bare ``QualName`` so that a
# module move upstream does not silently defeat the table; a genuine *rename*
# is meant to fall through to the UNKNOWN branch and be re-audited.

#: Backends documented to discard a declared activation quantization and run
#: weight-only arithmetic instead.  Value is the contract they rewrite it to.
_DROPS_ACTIVATION_SCALES: dict[str, str] = {
    # NVFP4 MoE ladder (oracle/nvfp4.py -> prepare_nvfp4_moe_layer_for_marlin,
    # which sets both activation-scale arguments to None and builds
    # nvfp4_w4a16_moe_quant_config).
    "NvFp4MoeBackend.MARLIN": "weight-only W4A16",
    "MarlinExperts": "weight-only W4A16",
    # Dense NVFP4 linear ladder (kernels/linear/nvfp4/marlin.py); vLLM forces
    # this kernel for a use_a16 declaration, which is legitimate — it is only a
    # violation when the group declared quantized activations.
    "MarlinNvFp4LinearKernel": "weight-only W4A16",
}

#: Backends whose *identity* is Triton even though no name along their MRO
#: carries the token (the NVFP4 MoE oracle spells this one as an enum).
_TRITON_BACKED_BACKENDS: frozenset[str] = frozenset({
    "NvFp4MoeBackend.EMULATION",
})

#: Backends audited to honor a declared NVFP4 W4A4 contract.  Everything else
#: — including ``flashinfer_b12x``, which is opt-in, version-sensitive, and
#: dynamically quantizes BF16 activations in-kernel — is UNKNOWN and fails.
_PRESERVES_NVFP4_W4A4: frozenset[str] = frozenset({
    # NVFP4 MoE oracle backends and their experts classes.
    "NvFp4MoeBackend.FLASHINFER_TRTLLM",
    "NvFp4MoeBackend.FLASHINFER_CUTEDSL",
    "NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED",
    "NvFp4MoeBackend.FLASHINFER_CUTLASS",
    "NvFp4MoeBackend.VLLM_CUTLASS",
    "NvFp4MoeBackend.EMULATION",
    "TrtLlmNvFp4ExpertsMonolithic",
    "TrtLlmNvFp4ExpertsModular",
    "FlashInferExperts",
    "FlashInferCuteDSLExperts",
    "FlashInferCuteDSLBatchedExperts",
    "CutlassExpertsFp4",
    "Nvfp4QuantizationEmulationTritonExperts",
    # Dense NVFP4 linear kernels.
    "CutlassNvFp4LinearKernel",
    "EmulationNvFp4LinearKernel",
})

#: Attributes on a resolved vLLM method (or on the ``scheme`` vLLM attaches to
#: the layer) that name the kernel it actually selected.  Extraction is by an
#: explicit list, never by scanning ``vars()``: an unrelated attribute that
#: happens to hold a class must not become a policy input.
_BACKEND_ATTRS: tuple[str, ...] = (
    "nvfp4_backend",     # NVFP4 MoE oracle enum
    "experts_cls",       # NVFP4 MoE resolved experts class
    "kernel",            # dense linear resolved kernel instance
    "fused_experts",     # modular-kernel experts instance, where present
)

_TRITON_TOKEN = "triton"


class _Identity(NamedTuple):
    """One resolved object the policy is allowed to judge."""

    role: str          # where it came from, for the error message
    label: str         # ``module.QualName`` (classes) or ``Enum.MEMBER``
    names: tuple[str, ...]   # every name/module the token test may inspect
    is_backend: bool   # True for a selected kernel, False for a wrapper
    is_enum: bool      # an oracle's *spelling* of a backend, not the class


def _describe(identities: Iterable[_Identity]) -> str:
    """Name every violating identity, most concrete first.

    A reader needs the resolved backend **class** — the enum spelling names the
    ladder rung, but the class names the code that would run — so classes are
    reported ahead of the enum member that selected them.
    """

    ordered = sorted(identities,
                     key=lambda i: (not i.is_backend, i.is_enum, i.label))
    return ", ".join(f"{i.label} (via {i.role})" for i in ordered)


def _class_names(cls: type) -> tuple[str, ...]:
    """Module paths and qualified names along *cls*'s MRO.

    The MRO is what makes the token test honest: vLLM's NVFP4 emulation experts
    class does not carry ``Triton`` in its own module path, but its base does.
    """

    names: list[str] = []
    for entry in getattr(cls, "__mro__", (cls,)):
        module = getattr(entry, "__module__", "") or ""
        qualname = getattr(entry, "__qualname__", "") or getattr(
            entry, "__name__", "")
        if module:
            names.append(module)
        if qualname:
            names.append(qualname)
            if module:
                names.append(f"{module}.{qualname}")
    return tuple(names)


def _identity(role: str, value: Any, *, is_backend: bool) -> _Identity | None:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return None
    if isinstance(value, Enum):
        label = f"{type(value).__qualname__}.{value.name}"
        names = (label, str(value.value)) + _class_names(type(value))
        return _Identity(role, label, names, is_backend, True)
    cls = value if isinstance(value, type) else type(value)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", "") or getattr(cls, "__name__", "?")
    label = f"{module}.{qualname}" if module else qualname
    return _Identity(role, label, _class_names(cls), is_backend, False)


def _identities(method: Any, layer: Any = None) -> list[_Identity]:
    """Every object that determines what a delegated group will execute.

    The wrapper vLLM returns (``CompressedTensorsLinearMethod``,
    ``CompressedTensors*MoEMethod``) is recorded but is *not* a backend: it
    dispatches to one.  The dense path stores its resolved scheme on the layer
    (``layer.scheme``) before returning the wrapper, and the scheme owns the
    kernel — so the layer is part of the resolution, not an extra.
    """

    found: list[_Identity] = []
    seen: set[tuple[str, bool]] = set()

    def add(identity: _Identity | None) -> None:
        if identity is None:
            return
        key = (identity.label, identity.is_backend)
        if key in seen:
            return
        seen.add(key)
        found.append(identity)

    add(_identity("resolved method", method, is_backend=False))
    for source, holder in (("method", method), ("layer scheme",
                                                getattr(layer, "scheme", None))):
        if holder is None:
            continue
        if source == "layer scheme":
            add(_identity("resolved scheme", holder, is_backend=False))
        for attr in _BACKEND_ATTRS:
            add(_identity(f"{source}.{attr}", getattr(holder, attr, None),
                          is_backend=True))
    return found


def _is_triton_backed(identity: _Identity) -> bool:
    if identity.label in _TRITON_BACKED_BACKENDS:
        return True
    return any(_TRITON_TOKEN in name.lower() for name in identity.names)


def _table_hit(identity: _Identity, table: Mapping[str, str] | Iterable[str]):
    """Look *identity* up by fully-qualified label and by bare class name."""

    keys = {identity.label, identity.label.rsplit(".", 1)[-1]}
    for key in keys:
        if key in table:
            return key
    return None


# --- declared contract -------------------------------------------------------


def _quant_entry(group: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = group.get(key)
    return value if isinstance(value, Mapping) else None


def _is_nvfp4(entry: Mapping[str, Any] | None) -> bool:
    """NVFP4 in the compressed-tensors vocabulary: 4-bit float, group 16."""

    if entry is None:
        return False
    return (str(entry.get("type", "")).lower() == "float"
            and entry.get("num_bits") == 4)


def declared_contract(group: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize what a stock ``compressed-tensors`` group *declares*.

    ``None`` (no group resolved) yields an "unknown" summary that quantizes
    nothing: the caller may still refuse a Triton-backed backend, but it must
    not claim a contract it could not read.
    """

    if group is None:
        return {"known": False, "quantizes_activations": False,
                "nvfp4_w4a4": False, "text": "no resolved config group"}
    weights = _quant_entry(group, "weights")
    activations = _quant_entry(group, "input_activations")
    quantizes_activations = activations is not None
    nvfp4_w4a4 = _is_nvfp4(weights) and _is_nvfp4(activations)
    if nvfp4_w4a4:
        text = "NVFP4 W4A4 (NVFP4 weights + NVFP4 input_activations)"
    elif _is_nvfp4(weights):
        text = "NVFP4 W4A16 (NVFP4 weights, no input_activations)"
    elif weights is not None and quantizes_activations:
        text = (f"W{weights.get('num_bits', '?')}A"
                f"{activations.get('num_bits', '?')} "
                f"({weights.get('type', '?')} weights + "
                f"{activations.get('type', '?')} input_activations)")
    elif weights is not None:
        text = (f"weight-only W{weights.get('num_bits', '?')} "
                f"({weights.get('type', '?')} weights)")
    else:
        text = "no weight quantization declared"
    return {"known": True, "quantizes_activations": quantizes_activations,
            "nvfp4_w4a4": nvfp4_w4a4, "text": text}


# --- the policy --------------------------------------------------------------


def _fail(prefix: str, group_name: str | None, contract: str, verdict: str,
          remedy: str) -> None:
    where = f"group {group_name!r}" if group_name else "an unnamed group"
    raise DelegatedBackendError(
        f"delegated compressed-tensors {where} at {prefix!r}: {verdict}. "
        f"The group declares {contract}. Gridbook serves a native "
        f"CUDA/CUTLASS operator lane and fails closed rather than serving a "
        f"contract it did not declare; there is no environment-variable "
        f"bypass. {remedy}"
    )


def require_native_delegated_backend(
    *,
    prefix: str,
    group_name: str | None,
    group: Mapping[str, Any] | None,
    method: Any,
    layer: Any = None,
) -> None:
    """Raise unless a delegated group's resolved backend is safe to serve.

    Three rules, in the order a reader should think about them:

    **T — no Triton lane.** Unconditional: any resolved identity whose MRO
    reaches Triton is refused, whatever the group declares.  This is the rule
    that makes the README's no-Triton sentence enforceable instead of merely
    true-by-absence.

    **A — no silent contract rewrite.** A group that declares quantized
    activations may not resolve to a backend documented to discard them.  It
    does *not* fire for a weight-only declaration: NVFP4 W4A16 on Marlin is
    exactly what vLLM's dense ladder forces, and it honors the declaration.

    **U — unaudited is not a pass.** For the one declaration class this repo
    has audited end to end (NVFP4 W4A4), the resolved kernel must appear in the
    audited table.  An unrecognized backend, or a resolved method whose kernel
    we cannot even name, is UNKNOWN — and UNKNOWN fails closed rather than
    becoming a false pass.
    """

    if method is None:
        return
    contract = declared_contract(group)
    identities = _identities(method, layer)

    # T — Triton lane.
    triton = [identity for identity in identities if _is_triton_backed(identity)]
    if triton:
        _fail(prefix, group_name, contract["text"],
              f"vLLM resolved it to Triton-backed {_describe(triton)}",
              "Pin a native backend for this group — e.g. "
              "--kernel-config '{\"moe_backend\":\"flashinfer_cutlass\"}' "
              "or '{\"moe_backend\":\"cutlass\"}' — or remove the group "
              "from the artifact. See docs/DELEGATED-NVFP4-MOE.md.")

    # A — declared activation quantization must survive backend selection.
    if contract["quantizes_activations"]:
        dropping = {identity: _table_hit(identity, _DROPS_ACTIVATION_SCALES)
                    for identity in identities}
        dropping = {identity: key for identity, key in dropping.items()
                    if key is not None}
        if dropping:
            rewritten = sorted({_DROPS_ACTIVATION_SCALES[key]
                                for key in dropping.values()})
            _fail(prefix, group_name, contract["text"],
                  f"vLLM resolved it to {_describe(dropping)}, which discards "
                  f"the declared activation scales and runs "
                  f"{' / '.join(rewritten)}",
                  "Pin a backend that honors the declared activation "
                  "contract, or re-export the group as the weight-only "
                  "contract it would actually execute. See "
                  "docs/DELEGATED-NVFP4-MOE.md.")

    # U — audited-or-fail, for the declaration class this repo has audited.
    if contract["nvfp4_w4a4"]:
        backends = [identity for identity in identities if identity.is_backend]
        if not backends:
            _fail(prefix, group_name, contract["text"],
                  f"Gridbook could not determine which backend "
                  f"{type(method).__module__}.{type(method).__qualname__} "
                  f"selected, so the declared activation contract cannot be "
                  f"proven to survive",
                  "This usually means an unaudited vLLM version. Audit the "
                  "backend and extend the table in "
                  "gridbook/delegated_preflight.py.")
        unaudited = [identity for identity in backends
                     if _table_hit(identity, _PRESERVES_NVFP4_W4A4) is None]
        if unaudited:
            _fail(prefix, group_name, contract["text"],
                  f"vLLM resolved it to {_describe(unaudited)}, which is not "
                  f"in Gridbook's audited set of backends that honor an NVFP4 "
                  f"W4A4 declaration",
                  "Audit the backend against docs/DELEGATED-NVFP4-MOE.md and "
                  "extend the table in gridbook/delegated_preflight.py, or "
                  "pin an audited backend for this group.")


# --- source-format passthrough ----------------------------------------------


def require_native_passthrough_backend(
    *,
    prefix: str,
    source_format: Any,
    method: Any,
    layer: Any = None,
) -> None:
    """Raise unless a SOURCE-PASSTHROUGH unit resolved to its audited backend.

    The sibling policy above judges a *compressed-tensors* group against a
    declaration it reads out of the group dict.  A passthrough unit has no such
    dict: the producer named a source format, and that format's audited route
    is a table in ``gridbook.source_passthrough``.  So the rules are the same
    three in spirit but keyed differently — the format supplies the contract.

    ``source_format`` is a ``gridbook.source_passthrough.SourceFormat``; it is
    taken structurally (``.id``, ``.audited_backends``, …) rather than by
    import, so this module stays torch-free and vLLM-free and the policy can be
    tested against stub formats.

    **T — no Triton lane.** Unconditional and first, exactly as above.  A
    passthrough unit that resolved to Triton would make the README's no-Triton
    sentence false, whatever the format declares.

    **B — a backend measured to break is named, not merely "unaudited".**
    vLLM's MXFP4 oracle picks ``DEEPGEMM_MXFP4`` by default on the whole sm12x
    family and that rung then raises inside DeepGEMM on sm_121.  Without this
    rule the operator would get Gridbook's generic UNKNOWN message for a
    failure we have already diagnosed, and would have to rediscover the fix.

    **U — unaudited is not a pass.** The resolved backend must appear in the
    format's audited set.  A format with an EMPTY audited set therefore refuses
    every delegation, which is how a BLOCKED verdict is encoded as data rather
    than as a comment.
    """

    if method is None:
        return
    fmt_id = getattr(source_format, "id", "<unknown>")
    described = getattr(source_format, "description", "")
    remedy = getattr(source_format, "remedy", "")
    audited = frozenset(getattr(source_format, "audited_backends", ()) or ())
    broken = dict(getattr(source_format, "known_broken_backends", {}) or {})

    identities = _identities(method, layer)

    def fail(verdict: str, fix: str) -> None:
        raise DelegatedBackendError(
            f"source-passthrough unit {prefix!r} ({fmt_id}): {verdict}. The "
            f"unit stores {described}. Gridbook serves a native CUDA/CUTLASS "
            f"operator lane and fails closed rather than serving a kernel it "
            f"has not audited for this format on this device; there is no "
            f"environment-variable bypass. {fix}"
        )

    # T — Triton lane.
    triton = [identity for identity in identities if _is_triton_backed(identity)]
    if triton:
        fail(f"vLLM resolved it to Triton-backed {_describe(triton)}", remedy)

    backends = [identity for identity in identities if identity.is_backend]

    # B — a rung we have already measured to fail for this format.
    for identity in backends:
        key = _table_hit(identity, broken)
        if key is not None:
            fail(f"vLLM resolved it to {_describe([identity])}, which "
                 f"Gridbook has measured to fail for this format: "
                 f"{broken[key]}", remedy)

    # U — audited-or-fail.
    if not backends:
        fail(f"Gridbook could not determine which backend "
             f"{type(method).__module__}.{type(method).__qualname__} selected, "
             f"so the audited native route cannot be proven to be the one that "
             f"will run",
             "This usually means an unaudited vLLM version. Audit it and "
             "extend gridbook/source_passthrough.py.")
    unaudited = [identity for identity in backends
                 if _table_hit(identity, audited) is None]
    if unaudited:
        known = ", ".join(sorted(audited)) or "(none — this format has no " \
                                             "audited native route on any device yet)"
        fail(f"vLLM resolved it to {_describe(unaudited)}, which is not in "
             f"Gridbook's audited set for this format. Audited: {known}",
             remedy)
