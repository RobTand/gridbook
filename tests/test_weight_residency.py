"""Dense CB weights must be resident exactly ONCE (issue #1, 2026-07-25).

``process_weights_after_loading`` derives a right-padded copy of the packed
weight for the decode/expand kernels. Before the fix the registered
``cb_qweight`` parameter stayed live alongside it, so every dense CB Linear was
resident twice — 15.07 GiB of duplicated ``cb_qweight`` on the shipped
Qwen3.6-27B gridbook artifact (21.38 GiB of weights served as ~36.5 GiB).
Invisible on the 128 GB unified-memory reference box; fatal on a 32 GB card.

The parameter is now a narrow VIEW of the padded buffer, which is only legal
because the pad is 16 bytes wide: the fp8 CUTLASS prefill entries still read
``layer.cb_qweight.data`` and TORCH_CHECK ``stride(0) % 16 == 0``. Both halves
of that contract are pinned here.

CPU-only; vLLM symbols are stubbed when unavailable (same idiom as
``test_target_namespace_compat``). No CUDA is touched: ``PRISMAQUANT_CB_DECODE``
is forced off the ``cuda`` default so the load-time JIT warm never fires.
"""
import os
import sys
import types

import pytest

os.environ.setdefault("PRISMAQUANT_CB_DECODE", "triton")

torch = pytest.importorskip("torch")

if "vllm" not in sys.modules:
    try:
        import vllm  # noqa: F401
    except Exception:
        def _mod(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        _mod("vllm")
        _mod("vllm.model_executor")
        _mod("vllm.model_executor.layers")
        _mod("vllm.model_executor.layers.quantization")
        lin = _mod("vllm.model_executor.layers.linear")
        lin.LinearBase = type("LinearBase", (), {})
        lin.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
        lin.LinearMethodBase = type("LinearMethodBase", (), {})
        lin.register_weight_loader_v2_supported_method = lambda cls: cls
        bc = _mod("vllm.model_executor.layers.quantization.base_config")

        class QuantizationConfig:
            def __init__(self):
                pass

        bc.QuantizationConfig = QuantizationConfig
        bc.QuantizeMethodBase = object
        vpe = _mod("vllm.model_executor.layers.vocab_parallel_embedding")
        vpe.UnquantizedEmbeddingMethod = type("UEM", (), {})
        vpe.VocabParallelEmbedding = type("VPE", (), {})
        fm = _mod("vllm.model_executor.layers.fused_moe")
        fm.RoutedExperts = type("RoutedExperts", (), {})
        par = _mod("vllm.model_executor.parameter")

        class _StubParam(torch.nn.Parameter):
            """vLLM's ModelWeightParameter/ChannelQuantScaleParameter stand-in:
            only the ``.data`` / ``register_parameter`` behaviour matters here."""

            def __new__(cls, data, **kw):
                return super().__new__(cls, data, requires_grad=False)

            def __init__(self, data, **kw):
                pass

        par.ModelWeightParameter = _StubParam
        par.ChannelQuantScaleParameter = _StubParam

from gridbook import codec                                        # noqa: E402
from gridbook.config import PrismaQuantConfig                      # noqa: E402
from gridbook import linear as cb_linear                           # noqa: E402
from gridbook.linear import PrismaQuantCBLinearMethod              # noqa: E402


@pytest.fixture(autouse=True)
def _reset_fp4_fused_mode_cache():
    """Keep process-global dispatch policy independent between tests."""
    cb_linear._FP4_FUSED_MODE.clear()
    yield
    cb_linear._FP4_FUSED_MODE.clear()


def test_dense_fp4_fused_prefill_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4", raising=False)
    assert cb_linear._fp4_fused_mode() == ""


@pytest.mark.parametrize("value", ["1", "midm"])
def test_dense_fp4_fused_prefill_explicit_modes(monkeypatch, value):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", value)
    assert cb_linear._fp4_fused_mode() == value


@pytest.fixture(autouse=True)
def _single_process_tp(monkeypatch):
    """Pin tensor-parallel rank/size to a single process for the duration of
    each test.

    Real vLLM's ``BasevLLMParameter.__init__`` reads the TP rank and world size
    from the process-global TP group, which only exists inside a launched
    engine — so constructing a ``ModelWeightParameter`` outside one dies with
    "tensor model parallel group is not initialized".

    This is what lets the file run against the REAL parameter class wherever
    vLLM is installed (the GPU container), instead of only against the stub
    above. That distinction is the whole point: the contract under test is a
    property of ``ModelWeightParameter.data`` — rebinding it to a narrow view
    must release the original storage — so verifying it only against a
    ``torch.nn.Parameter`` stub would verify nothing about the code that ships.
    No-op when vLLM is absent and the stub is in force.
    """
    par = sys.modules.get("vllm.model_executor.parameter")
    if par is None or not hasattr(par, "get_tensor_model_parallel_rank"):
        return
    monkeypatch.setattr(par, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(par, "get_tensor_model_parallel_world_size", lambda: 1)


# The real shipped fp8 rungs. type_size == 4*k, all 16-byte multiples — the
# property the padded-stride invariant rests on.
_FP8_RUNGS = [(28, 112), (32, 128), (36, 144), (40, 160), (44, 176), (48, 192)]


# ---------------------------------------------------------------------------
# 1. pad invariant: >= 8 bytes of read slack AND a 16-byte-multiple row stride
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,type_size", _FP8_RUNGS)
@pytest.mark.parametrize("K", [256, 1024, 2048, 5120])
def test_pad_qweight_keeps_stride_16_aligned_and_gives_slack(k, type_size, K):
    assert type_size == 4 * k, "fp8 rung table drifted from type_size = 4k"
    row_bytes = (K // codec.SUPERBLOCK) * type_size
    assert row_bytes % 16 == 0, "unpadded fp8 row stride must already be 16B"
    qw = torch.zeros(8, row_bytes, dtype=torch.uint8)
    padded = codec.pad_qweight(qw)
    assert padded.shape == (8, row_bytes + codec.PAD_BYTES)
    assert padded.stride(1) == 1
    assert padded.stride(0) % 16 == 0, (
        "the fp8 CUTLASS prefill entries TORCH_CHECK stride(0) % 16 == 0")
    assert padded.stride(0) - row_bytes >= 8, (
        "the 8-byte codeword window reads past the last packed byte")
    assert padded.stride(0) >= (K // 256) * 4 * k       # the kernel's own check
    assert torch.equal(padded[:, :row_bytes], qw)
    assert int(padded[:, row_bytes:].max()) == 0        # pad is zeroed


def test_pad_width_is_a_16_multiple():
    """A pad that is not itself a 16-byte multiple cannot preserve a
    16-byte-aligned row stride — that was exactly the old +8's failure."""
    assert codec.PAD_BYTES % 16 == 0 and codec.PAD_BYTES >= 8


# ---------------------------------------------------------------------------
# 2. one storage: cb_qweight is a view of _cb_qw_padded, not a second copy
# ---------------------------------------------------------------------------

_K, _N = 512, 64
_SCHEME = {"grid": "fp8", "mode": "product", "k": 44, "n_sub": 4,
           "type_size": 176, "group_size": 0, "vec_dim": 8,
           "codebook_group": "mlp", "codebook_source": "learned",
           "codebook_ref": ["cb.a", "cb.b", "cb.c", "cb.d"]}
_TARGET = "model.layers.0.mlp.down_proj"


class _Layer(torch.nn.Module):
    """Stands in for vLLM's ``LinearBase``: only parameter registration and
    attribute assignment are exercised by process_weights_after_loading."""


def _loaded_layer():
    """A CB Linear taken through create_weights -> (weight load) ->
    process_weights_after_loading, entirely on CPU."""
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {"g": {"format": "FP8_CB_K44",
                                "targets": [_TARGET],
                                "scheme": dict(_SCHEME)}},
        "ignore": ["lm_head"],
    })
    cfg._ensure_resolved()
    # k=44, n_sub=4 -> ceil-first widths [11,11,11,11], sub_dim = 8//4 = 2.
    cfg.get_codebooks = lambda: {
        n: torch.zeros(2 ** 11, 2, dtype=torch.bfloat16)
        for n in _SCHEME["codebook_ref"]}

    method = PrismaQuantCBLinearMethod(cfg, dict(_SCHEME), _TARGET)
    layer = _Layer()
    method.create_weights(layer, _K, [_N], _K, _N, torch.bfloat16,
                          weight_loader=None)
    # "Load" the checkpoint tensors.
    layer.cb_qweight.data.copy_(torch.randint(
        0, 256, layer.cb_qweight.shape, dtype=torch.uint8))
    layer.weight_scale.data.copy_(torch.rand(_N))
    original = layer.cb_qweight.data
    method.process_weights_after_loading(layer)
    return layer, original


def test_cb_qweight_and_padded_share_one_storage():
    layer, _original = _loaded_layer()
    qw, padded = layer.cb_qweight.data, layer._cb_qw_padded
    assert qw.data_ptr() == padded.data_ptr()
    assert (qw.untyped_storage().data_ptr()
            == padded.untyped_storage().data_ptr())
    assert qw.untyped_storage().nbytes() == padded.untyped_storage().nbytes()


def test_layer_no_longer_references_the_original_storage():
    """The layer must hold NO handle on the pre-pad allocation — that handle is
    what kept the second 15.07 GiB alive for the whole serve. (The only other
    handle was a local in ``process_weights_after_loading``, dropped on return.)
    Compared by storage identity, not by value."""
    layer, original = _loaded_layer()
    assert (layer.cb_qweight.data.untyped_storage().data_ptr()
            != original.untyped_storage().data_ptr())
    assert (layer._cb_qw_padded.untyped_storage().data_ptr()
            != original.untyped_storage().data_ptr())


def test_repointed_cb_qweight_still_satisfies_the_fp8_kernel_checks():
    """``layer.cb_qweight.data`` is still passed to cb_fused_prefill_mm_scaled
    (mid-M, default-on) and the persistent-TC path. Mirror their TORCH_CHECKs."""
    layer, _ = _loaded_layer()
    qw = layer.cb_qweight.data
    row_bytes = (_K // codec.SUPERBLOCK) * _SCHEME["type_size"]
    assert qw.dim() == 2 and qw.shape == (_N, row_bytes)
    assert qw.stride(1) == 1
    assert qw.stride(0) % 16 == 0
    assert qw.stride(0) >= (_K // 256) * 4 * _SCHEME["k"]
    assert qw.stride(0) == row_bytes + codec.PAD_BYTES     # padded, not copied


def test_repointed_cb_qweight_holds_the_loaded_bytes():
    """A view must not change what the kernels read."""
    layer, original = _loaded_layer()
    assert torch.equal(layer.cb_qweight.data, original)


def test_weight_scale_is_not_duplicated():
    """``_cb_scale`` aliases ``weight_scale.data`` (already fp32 and 1-D, so
    reshape(-1).to(float32) is a no-op view) — there is no second copy to free,
    which is why ``weight_scale`` is deliberately left registered."""
    layer, _ = _loaded_layer()
    assert (layer._cb_scale.untyped_storage().data_ptr()
            == layer.weight_scale.data.untyped_storage().data_ptr())
