"""D0.2 preflight: a delegated group may not resolve to Triton or lose scales.

CPU-only and vLLM-free by construction. ``gridbook.delegated_preflight`` imports
neither torch nor vLLM, so the policy is exercised directly against stub classes
that mirror the vLLM shapes documented in ``docs/DELEGATED-NVFP4-MOE.md``:

  * NVFP4 MoE — ``CompressedTensorsW4A4Nvfp4MoEMethod`` stores the oracle's
    verdict on ``self.nvfp4_backend`` (an enum) and ``self.experts_cls``;
  * dense NVFP4 — ``CompressedTensorsLinearMethod`` carries nothing, and vLLM
    attaches the resolved scheme to the *layer*, whose ``.kernel`` is the
    selected linear kernel.

The stubs reproduce those two shapes and nothing else; what is pinned here is
the POLICY and its DIAGNOSTICS, not any vLLM behavior.
"""
from __future__ import annotations

import enum

import pytest

from gridbook.delegated_preflight import (
    DelegatedBackendError,
    declared_contract,
    require_native_delegated_backend,
)


# --- stubs mirroring the audited vLLM shapes ---------------------------------

_FUSED_MOE = "vllm.model_executor.layers.fused_moe.experts"
_LINEAR_KERNELS = "vllm.model_executor.kernels.linear.nvfp4"


def _cls(name: str, module: str, bases: tuple[type, ...] = ()) -> type:
    cls = type(name, bases, {})
    cls.__module__ = module
    cls.__qualname__ = name
    return cls


TritonExperts = _cls("TritonExperts", f"{_FUSED_MOE}.triton_moe")
# Named so the token test CANNOT pass on this class's own name: only the MRO
# reaches Triton, which is the case the structural rule exists for.
EmulationExperts = _cls("Nvfp4QuantizationEmulationExperts",
                        f"{_FUSED_MOE}.nvfp4_emulation_moe", (TritonExperts,))
FlashInferExperts = _cls("FlashInferExperts", f"{_FUSED_MOE}.flashinfer_cutlass_moe")
CutlassExpertsFp4 = _cls("CutlassExpertsFp4", f"{_FUSED_MOE}.cutlass_moe")
MarlinExperts = _cls("MarlinExperts", f"{_FUSED_MOE}.marlin_moe")
MysteryExperts = _cls("SomeFutureNvFp4Experts", f"{_FUSED_MOE}.some_future_moe")

MarlinNvFp4LinearKernel = _cls("MarlinNvFp4LinearKernel",
                               f"{_LINEAR_KERNELS}.marlin")
CutlassNvFp4LinearKernel = _cls("CutlassNvFp4LinearKernel",
                                f"{_LINEAR_KERNELS}.cutlass")
# The kernel vLLM 0.26's ladder actually selects for a delegated dense W4A4
# group. Note the module: FlashInfer is the WRAPPER, CUTLASS is the operator.
FlashInferCutlassNvFp4LinearKernel = _cls(
    "FlashInferCutlassNvFp4LinearKernel", f"{_LINEAR_KERNELS}.flashinfer")


class NvFp4MoeBackend(enum.Enum):
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"
    EMULATION = "EMULATION"
    FLASHINFER_B12X = "FLASHINFER_B12X"


class _MoEMethod:
    """``CompressedTensorsW4A4Nvfp4MoEMethod``'s post-``__init__`` shape."""

    def __init__(self, backend: NvFp4MoeBackend, experts_cls: type) -> None:
        self.nvfp4_backend = backend
        self.experts_cls = experts_cls


_MoEMethod.__module__ = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4")
_MoEMethod.__qualname__ = "CompressedTensorsW4A4Nvfp4MoEMethod"


class _Scheme:
    def __init__(self, kernel: type) -> None:
        self.kernel = kernel()


_Scheme.__qualname__ = "CompressedTensorsW4A4Fp4"


class _Layer:
    def __init__(self, scheme=None) -> None:
        if scheme is not None:
            self.scheme = scheme


class _LinearMethod:
    pass


_LinearMethod.__qualname__ = "CompressedTensorsLinearMethod"


_NVFP4 = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
          "group_size": 16, "symmetric": True, "dynamic": False}
_NVFP4_ACT = dict(_NVFP4, dynamic="local")
W4A4_GROUP = {"format": "nvfp4-pack-quantized", "weights": _NVFP4,
              "input_activations": _NVFP4_ACT,
              "targets": ["re:.*mlp.experts.*"]}
W4A16_GROUP = {"format": "nvfp4-pack-quantized", "weights": _NVFP4,
               "input_activations": None, "targets": ["re:.*visual.*"]}


def _check(group, method, layer=None, *, name="group_nvfp4",
           prefix="model.layers.0.mlp.experts"):
    require_native_delegated_backend(prefix=prefix, group_name=name,
                                     group=group, method=method, layer=layer)


# --- the declaration ---------------------------------------------------------


def test_declared_contract_reads_the_config_group():
    w4a4 = declared_contract(W4A4_GROUP)
    assert w4a4["nvfp4_w4a4"] and w4a4["quantizes_activations"]
    assert "W4A4" in w4a4["text"]
    w4a16 = declared_contract(W4A16_GROUP)
    assert not w4a16["nvfp4_w4a4"] and not w4a16["quantizes_activations"]
    assert "W4A16" in w4a16["text"]
    unknown = declared_contract(None)
    assert not unknown["known"] and not unknown["quantizes_activations"]


# --- rule T: no Triton lane --------------------------------------------------


def test_triton_backed_moe_backend_is_rejected():
    method = _MoEMethod(NvFp4MoeBackend.EMULATION, EmulationExperts)
    with pytest.raises(DelegatedBackendError) as excinfo:
        _check(W4A4_GROUP, method)
    message = str(excinfo.value)
    # The resolved backend class, the group, and the violated contract.
    assert "Nvfp4QuantizationEmulationExperts" in message
    assert "group_nvfp4" in message
    assert "model.layers.0.mlp.experts" in message
    assert "W4A4" in message
    assert "Triton" in message


def test_triton_rule_fires_without_a_resolved_declaration():
    """Triton is refused even when the declaring group cannot be resolved.

    A group we cannot read is not a licence to serve Triton: rule T is
    unconditional, and only the contract-preserving rules need the declaration.
    """

    method = _MoEMethod(NvFp4MoeBackend.EMULATION, EmulationExperts)
    with pytest.raises(DelegatedBackendError, match="Triton"):
        _check(None, method, name=None)


def test_triton_enum_spelling_is_rejected_on_its_own():
    """``EMULATION`` carries no ``triton`` token; the versioned set catches it."""

    method = _MoEMethod(NvFp4MoeBackend.EMULATION, CutlassExpertsFp4)
    with pytest.raises(DelegatedBackendError, match="NvFp4MoeBackend.EMULATION"):
        _check(W4A4_GROUP, method)


# --- rule A: no silent contract rewrite --------------------------------------


def test_marlin_on_a_declared_w4a4_group_is_rejected():
    method = _MoEMethod(NvFp4MoeBackend.MARLIN, MarlinExperts)
    with pytest.raises(DelegatedBackendError) as excinfo:
        _check(W4A4_GROUP, method)
    message = str(excinfo.value)
    assert "MarlinExperts" in message or "NvFp4MoeBackend.MARLIN" in message
    assert "activation scales" in message
    assert "weight-only W4A16" in message
    assert "W4A4" in message


def test_marlin_on_a_declared_w4a16_group_is_accepted():
    """The published 27B's stock vision tower is exactly this shape.

    vLLM *forces* the Marlin kernel for a weight-only NVFP4 declaration. Marlin
    drops nothing there, because nothing was declared; a preflight that
    rejected it would break a shipping artifact to prevent an impossibility.
    """

    layer = _Layer(_Scheme(MarlinNvFp4LinearKernel))
    _check(W4A16_GROUP, _LinearMethod(), layer,
           name="group_vision", prefix="visual.blocks.0.attn.qkv")


def test_dense_marlin_on_a_declared_w4a4_group_is_rejected():
    layer = _Layer(_Scheme(MarlinNvFp4LinearKernel))
    with pytest.raises(DelegatedBackendError) as excinfo:
        _check(W4A4_GROUP, _LinearMethod(), layer, name="group_dense",
               prefix="model.layers.0.self_attn.qkv_proj")
    assert "MarlinNvFp4LinearKernel" in str(excinfo.value)


# --- rule U: unaudited is not a pass -----------------------------------------


def test_audited_native_moe_backends_are_accepted():
    for backend, experts in (
        (NvFp4MoeBackend.FLASHINFER_CUTLASS, FlashInferExperts),
        (NvFp4MoeBackend.VLLM_CUTLASS, CutlassExpertsFp4),
    ):
        _check(W4A4_GROUP, _MoEMethod(backend, experts))


def test_audited_native_dense_kernel_is_accepted():
    layer = _Layer(_Scheme(CutlassNvFp4LinearKernel))
    _check(W4A4_GROUP, _LinearMethod(), layer, name="group_dense",
           prefix="model.layers.0.self_attn.qkv_proj")


def test_flashinfer_cutlass_dense_kernel_is_accepted():
    """Re-audited 2026-08-14; before that, this build could serve no stock
    NVFP4 W4A4 Linear at all.

    vLLM 0.26 selects ``FlashInferCutlassNvFp4LinearKernel`` where the table
    only knew ``CutlassNvFp4LinearKernel``, so the policy failed CLOSED on
    every delegated dense W4A4 group -- correct behaviour for an unknown name,
    and exactly the re-audit branch this module documents. The audit checked
    the two properties the policy protects: ``input_quant_key()`` is
    ``kNvfp4Dynamic`` and ``apply_weights`` passes a real ``x_blockscale`` into
    the GEMM (the activation contract is honoured, not rewritten weight-only
    the way Marlin rewrites it), and the operator is
    ``flashinfer_scaled_fp4_mm(..., backend="cutlass")`` with no ``triton``
    token anywhere on the class, module, or MRO.
    """
    layer = _Layer(_Scheme(FlashInferCutlassNvFp4LinearKernel))
    _check(W4A4_GROUP, _LinearMethod(), layer, name="group_dense",
           prefix="model.layers.0.self_attn.qkv_proj")


def test_unaudited_backend_on_a_declared_w4a4_group_is_rejected():
    method = _MoEMethod(NvFp4MoeBackend.FLASHINFER_B12X, MysteryExperts)
    with pytest.raises(DelegatedBackendError) as excinfo:
        _check(W4A4_GROUP, method)
    message = str(excinfo.value)
    assert "SomeFutureNvFp4Experts" in message or "B12X" in message
    assert "audited" in message


def test_unnameable_backend_on_a_declared_w4a4_group_is_rejected():
    """UNKNOWN must not become a false pass (DELEGATED-NVFP4-MOE §Preflight)."""

    with pytest.raises(DelegatedBackendError, match="could not determine"):
        _check(W4A4_GROUP, _LinearMethod(), _Layer())


def test_unaudited_backend_on_a_weight_only_group_is_accepted():
    """Rule U is scoped to the declaration class this repo actually audited."""

    layer = _Layer(_Scheme(MarlinNvFp4LinearKernel))
    _check(W4A16_GROUP, _LinearMethod(), layer, name="group_vision",
           prefix="visual.blocks.0.mlp.fc1")


# --- fail-closed means fail-closed -------------------------------------------


def test_no_environment_variable_bypasses_the_policy(monkeypatch):
    for name in ("PRISMAQUANT_ALLOW_TRITON", "GRIDBOOK_ALLOW_TRITON",
                 "PRISMAQUANT_DELEGATION_UNSAFE", "GRIDBOOK_SKIP_PREFLIGHT",
                 "PRISMAQUANT_ALLOW_MARLIN_W4A4"):
        monkeypatch.setenv(name, "1")
    with pytest.raises(DelegatedBackendError):
        _check(W4A4_GROUP, _MoEMethod(NvFp4MoeBackend.MARLIN, MarlinExperts))
    with pytest.raises(DelegatedBackendError):
        _check(W4A4_GROUP, _MoEMethod(NvFp4MoeBackend.EMULATION,
                                      EmulationExperts))


def test_policy_module_reads_no_environment():
    """The bypass test above only proves five names; this proves the class."""

    import inspect

    from gridbook import delegated_preflight

    source = inspect.getsource(delegated_preflight)
    assert "os.environ" not in source and "getenv" not in source


def test_absent_method_is_not_a_violation():
    """``get_quant_method`` returning ``None`` means "not mine", not "unsafe"."""

    _check(W4A4_GROUP, None)
