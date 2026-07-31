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


# ---------------------------------------------------------------------------
# 3. fused roles: exact codebook references share one LUT block safely
# ---------------------------------------------------------------------------

_FUSED_K = 28
_FUSED_TYPE_SIZE = 4 * _FUSED_K
_FUSED_SCHEME = {
    "grid": "fp8", "mode": "product", "k": _FUSED_K, "n_sub": 4,
    "type_size": _FUSED_TYPE_SIZE, "group_size": 0, "vec_dim": 8,
    "codebook_group": "attn", "codebook_source": "learned",
}
_FUSED_PREFIX = "model.layers.0.self_attn.qkv_proj"
_FUSED_ROLES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
]
_REF_A = tuple(f"cb.a.sub{i}" for i in range(4))
_REF_B = tuple(f"cb.b.sub{i}" for i in range(4))
_FUSED_WIDTHS = (3, 5, 4)
_FUSED_BLOCK_VALUES = 4 * (2 ** (_FUSED_K // 4)) * 2


def _fused_loaded_layer(role_refs):
    """Load a three-role fused dense layer without touching CUDA or vLLM TP."""
    target_scheme = {
        role: {**_FUSED_SCHEME, "codebook_ref": list(ref)}
        for role, ref in zip(_FUSED_ROLES, role_refs)
    }
    all_names = {name for ref in role_refs for name in ref}
    codebooks = {
        name: torch.full(
            (2 ** (_FUSED_K // 4), 2),
            # Deliberately identical contents across differently named refs:
            # deduplication is by provenance identity, never by current value.
            1.0,
            dtype=torch.bfloat16,
        )
        for name in all_names
    }

    class _FusedConfig:
        def __init__(self):
            self.target_scheme = target_scheme

        def shard_target_keys(self, prefix, *, unfused_fallback=False):
            assert prefix == _FUSED_PREFIX
            assert unfused_fallback
            return list(_FUSED_ROLES)

        def get_codebooks(self):
            return codebooks

    method = PrismaQuantCBLinearMethod(
        _FusedConfig(),
        {**_FUSED_SCHEME, "codebook_ref": list(role_refs[0])},
        _FUSED_PREFIX,
    )
    layer = _Layer()
    rows = sum(_FUSED_WIDTHS)
    row_bytes = (_K // codec.SUPERBLOCK) * _FUSED_TYPE_SIZE
    layer.cb_qweight = torch.nn.Parameter(
        torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.linspace(0.5, 1.5, rows), requires_grad=False)
    layer.logical_widths = list(_FUSED_WIDTHS)
    layer._cb_input_size = _K
    method.process_weights_after_loading(layer)
    return method, layer, codebooks


def _flat_for_ref(ref, codebooks):
    return codec.build_flat_codebook(
        [codebooks[name] for name in ref], _FUSED_PREFIX, "fp8")


def test_identical_fused_refs_deduplicate_block_and_offsets():
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    assert layer._cb_flat.numel() == _FUSED_BLOCK_VALUES
    assert torch.equal(
        layer._cb_row_offset,
        torch.zeros(sum(_FUSED_WIDTHS), dtype=torch.int32),
    )
    assert layer._cb_fp8_fused_lut_ok is True
    assert method._fused_fp8_lut_ok(layer) is True


def test_distinct_fused_refs_keep_distinct_blocks_and_offsets():
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_B, _REF_A])
    expected_offsets = torch.tensor(
        [0] * _FUSED_WIDTHS[0]
        + [_FUSED_BLOCK_VALUES] * _FUSED_WIDTHS[1]
        + [0] * _FUSED_WIDTHS[2],
        dtype=torch.int32,
    )
    assert layer._cb_flat.numel() == 2 * _FUSED_BLOCK_VALUES
    assert torch.equal(layer._cb_row_offset, expected_offsets)
    assert layer._cb_fp8_fused_lut_ok is False
    assert method._fused_fp8_lut_ok(layer) is False


def test_codebook_ref_order_is_part_of_dedup_identity():
    reversed_ref = tuple(reversed(_REF_A))
    method, layer, _ = _fused_loaded_layer([_REF_A, reversed_ref, _REF_A])
    assert layer._cb_flat.numel() == 2 * _FUSED_BLOCK_VALUES
    assert torch.equal(
        layer._cb_row_offset,
        torch.tensor(
            [0] * _FUSED_WIDTHS[0]
            + [_FUSED_BLOCK_VALUES] * _FUSED_WIDTHS[1]
            + [0] * _FUSED_WIDTHS[2],
            dtype=torch.int32,
        ),
    )
    assert method._fused_fp8_lut_ok(layer) is False


@pytest.mark.parametrize("role_refs", [
    (_REF_A, _REF_A, _REF_A),
    (_REF_A, _REF_B, _REF_A),
])
def test_dedup_preserves_every_rows_addressed_lut(role_refs):
    """The compact layout addresses the same LUT as the old concatenation."""
    _method, layer, codebooks = _fused_loaded_layer(role_refs)
    legacy_blocks = [_flat_for_ref(ref, codebooks) for ref in role_refs]
    legacy_flat = torch.cat(legacy_blocks)
    legacy_base = 0
    row = 0
    for width, block in zip(_FUSED_WIDTHS, legacy_blocks):
        new_bases = layer._cb_row_offset[row:row + width]
        assert bool((new_bases == new_bases[0]).all())
        new_base = int(new_bases[0])
        assert torch.equal(
            layer._cb_flat[new_base:new_base + block.numel()],
            legacy_flat[legacy_base:legacy_base + block.numel()],
        )
        legacy_base += block.numel()
        row += width


def _mock_fp8_ops(monkeypatch, N):
    vops = types.ModuleType("vllm._custom_ops")
    vops.scaled_fp8_quant = lambda x, **kw: (
        x, torch.ones(x.shape[0], 1, dtype=torch.float32))
    vops.cutlass_scaled_mm = lambda xq, wt, sa, ws, dtype, bias: torch.full(
        (xq.shape[0], N), 7.0, dtype=torch.bfloat16)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", vops)
    monkeypatch.setattr(sys.modules["vllm"], "_custom_ops", vops,
                        raising=False)
    return vops


def test_same_ref_fused_roles_enter_midm_kernel(monkeypatch):
    from gridbook import cuda_ext

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K, M = layer._cb_N, layer._cb_K, 32
    _mock_fp8_ops(monkeypatch, N)
    calls = []

    class _FusedExt:
        @staticmethod
        def cb_fused_prefill_mm_scaled(xq, qw, cb, sa, ws, n, k, k_bits):
            calls.append((qw, cb, n, k, k_bits))
            return torch.full((M, N), 11.0, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: _FusedExt())
    monkeypatch.setattr(
        sys.modules["gridbook.linear"], "expand_cb_to_fp8",
        lambda *a, **kw: pytest.fail("eligible shared LUT fell back"),
    )
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert len(calls) == 1
    assert calls[0][1] is layer._cb_flat_fp8
    assert torch.equal(out, torch.full_like(out, 11.0))


def test_distinct_ref_fused_roles_fall_back_before_extension(monkeypatch):
    from gridbook import cuda_ext

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_B, _REF_A])
    N, K, M = layer._cb_N, layer._cb_K, 32
    _mock_fp8_ops(monkeypatch, N)
    monkeypatch.setattr(
        cuda_ext, "get_fused_ext",
        lambda: pytest.fail("offset-unsafe fused extension was queried"),
    )
    fallback_calls = []

    def _expand(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return torch.zeros(N, K, dtype=torch.float32)

    monkeypatch.setattr(
        sys.modules["gridbook.linear"], "expand_cb_to_fp8", _expand)
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert len(fallback_calls) == 1
    assert torch.equal(out, torch.full_like(out, 7.0))


def test_fused_lut_guard_rejects_uniform_nonzero_base():
    """The kernel starts at cb[0]; merely uniform offsets are insufficient."""
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    del layer._cb_fp8_fused_lut_ok
    layer._cb_row_offset.fill_(_FUSED_BLOCK_VALUES)
    assert method._fused_fp8_lut_ok(layer) is False


@pytest.mark.parametrize("attr,value", [
    ("n_sub", 2),
    ("type_size", _FUSED_TYPE_SIZE + 16),
])
def test_fused_lut_guard_rejects_incompatible_fp8_layout(attr, value):
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    setattr(method, attr, value)
    assert method._fused_fp8_lut_ok(layer) is False
