"""CPU gate for safe fp4-CB stock prefill (no GPU, no vLLM needed).

Three things are decidable on CPU from shapes and config ints alone:

  1. ``_apply_inline``'s mode precedence — fused FP4 is explicitly opted in;
     an unset request takes the conservative loop, and an explicit stock/loop
     request remains authoritative.
  2. ``_stock_chunk_default`` — the fp4 bf16 chunk transient is 2 B/elt, so the
     fp8 lane's flat 256 holds every expert of a large MoE live at once. The
     fp4 chunk is byte-budgeted instead; fp8 keeps its 256 verbatim.
  3. ``_expand_stack_slice``'s fp4 branch — ``type_size = 4k+9`` puts the
     expanders' 8-byte codeword window inside the superblock for every k, so
     the ``codec.pad_qweight`` copy is provably dead weight there and the RAW
     slice view is handed to the expander. The predicate is checked, not
     assumed: a rung it does not hold for falls back to the pad.

vLLM is stubbed when absent (the same technique
``tests/test_target_namespace_compat.py`` already uses), and the stubs are
removed again in teardown so a later file's ``importorskip("vllm")`` is not
satisfied by our leftovers.
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

# The R2/Hy3 CB-band shape the byte budget was sized against: E=192,
# w13 = (2*inter=3072, hidden=4096), w2 = (hidden=4096, inter=1536).
HY3 = dict(E=192, hidden=4096, inter=1536)
GIB = 1 << 30


def _install_vllm_stubs():
    """Minimal stand-ins for the vLLM symbols ``gridbook.moe`` imports."""
    def _mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _mod("vllm")
    _mod("vllm.model_executor")
    me_utils = _mod("vllm.model_executor.utils")
    me_utils.set_weight_attrs = lambda p, d: [setattr(p, k, v)
                                              for k, v in d.items()]
    _mod("vllm.model_executor.layers")
    fm = _mod("vllm.model_executor.layers.fused_moe")
    fm.RoutedExperts = type("RoutedExperts", (), {})
    cfg = _mod("vllm.model_executor.layers.fused_moe.config")
    cfg.FusedMoEConfig = type("FusedMoEConfig", (), {})
    cfg.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
    base = _mod("vllm.model_executor.layers.fused_moe.fused_moe_method_base")
    base.FusedMoEMethodBase = type(
        "FusedMoEMethodBase", (), {"__init__": lambda self, moe=None: None})
    actm = _mod("vllm.model_executor.layers.fused_moe.activation")
    actm.MoEActivation = type(
        "MoEActivation", (), {"from_str": staticmethod(lambda s: s)})
    actm.apply_moe_activation = lambda act, out, x: None


@pytest.fixture(scope="module")
def moe(request):
    """``gridbook.moe``, importable with or without a real vLLM install."""
    import importlib

    # Preserve complete module objects, not only names. Another CPU test may
    # have installed a deliberately smaller vLLM stub first; ``import vllm``
    # alone would then succeed even though the MoE submodules are absent.
    before = {
        name: module for name, module in sys.modules.items()
        if name.startswith("vllm") or name.startswith("gridbook")
    }
    stubbed = False
    try:
        from vllm.model_executor.layers.fused_moe.config import (  # noqa: F401
            FusedMoEConfig,
        )
    except Exception:
        _install_vllm_stubs()
        stubbed = True
    mod = importlib.import_module("gridbook.moe")
    yield mod
    if stubbed:
        for name in list(sys.modules):
            if name.startswith("vllm") or name.startswith("gridbook"):
                sys.modules.pop(name, None)
        sys.modules.update(before)


def _method(moe, *, grid, k, n_sub, type_size, is_v2=True):
    """A ``PrismaQuantCBMoEMethod`` with ``__init__`` bypassed (the fixture
    pattern the other MoE test files use) carrying just the format fields the
    sizing / expand code reads."""
    m = moe.PrismaQuantCBMoEMethod.__new__(moe.PrismaQuantCBMoEMethod)
    m.prefix = "test"
    m.scheme = {"grid": grid, "k": k, "n_sub": n_sub, "type_size": type_size}
    m.is_fp4 = grid == "fp4"
    m.is_v2 = is_v2
    m.k = k
    m.n_sub = n_sub
    m.type_size = type_size
    m._sub_table = None
    return m


def _fp4(moe, k=16, n_sub=2):
    return _method(moe, grid="fp4", k=k, n_sub=n_sub, type_size=4 * k + 9)


def _fp8(moe, k=44):
    return _method(moe, grid="fp8", k=k, n_sub=4, type_size=4 * k,
                   is_v2=False)


def _layer(E=HY3["E"], hidden=HY3["hidden"], inter=HY3["inter"]):
    lay = types.SimpleNamespace()
    lay._cb_E, lay._cb_hidden, lay._cb_inter = E, hidden, inter
    return lay


# --------------------------------------------------------------------------- #
# 1 — the prefill-mode default.                                                 #
# --------------------------------------------------------------------------- #
def _dispatch_mode(moe, monkeypatch, m, fused_fp4="fused_fp4"):
    """Run ``_apply_inline`` far enough to see which prefill arm it picks.

    32 tokens (> the 16-token decode cut-off) with every arm replaced by a
    sentinel: no CUDA, no vLLM kernels, just the mode decision.
    """
    cls = moe.PrismaQuantCBMoEMethod
    for arm in ("auto", "stock", "loop", "batched", "grouped_fused",
                "grouped_fused_v2", "l2_pipeline"):
        monkeypatch.setattr(
            cls, f"_apply_prefill_{arm}",
            (lambda tag: lambda self, *a, **kw: tag)(arm),
            raising=True)
    # ``grouped_fused`` first tries its v2 implementation; report the public
    # selector name so this helper describes dispatch policy, not internals.
    monkeypatch.setattr(
        cls, "_apply_prefill_grouped_fused_v2",
        lambda self, *a, **kw: "grouped_fused", raising=True)
    monkeypatch.setattr(
        cls, "_apply_prefill_grouped_fused_fp4",
        lambda self, *a, **kw: fused_fp4, raising=True)
    monkeypatch.setattr(cls, "_cuda_moe_ok", lambda self, layer: False)
    lay = _layer()
    lay._cb_layer_id = None
    lay.apply_router_weight_on_input = False
    lay.activation = types.SimpleNamespace(value="silu")
    x = torch.zeros(32, HY3["hidden"])
    ids = torch.zeros(32, 4, dtype=torch.int32)
    w = torch.ones(32, 4)
    return m._apply_inline(lay, x, w, ids)


def test_unset_fp4_uses_conservative_loop(moe, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4_MOE", raising=False)
    assert _dispatch_mode(moe, monkeypatch, _fp4(moe)) == "loop"


@pytest.mark.parametrize("value", ["1", "128", "256"])
def test_fp4_fused_prefill_is_explicitly_opted_in(moe, monkeypatch, value):
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL", raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", value)
    assert _dispatch_mode(moe, monkeypatch, _fp4(moe)) == "fused_fp4"


def test_opted_in_fp4_fused_miss_falls_back_to_loop(moe, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL", raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", "1")
    assert _dispatch_mode(
        moe, monkeypatch, _fp4(moe), fused_fp4=None) == "loop"


def test_default_prefill_mode_fp8_is_still_auto(moe, monkeypatch):
    """REGRESSION GUARD for de10a2d: the fp8-CB lane keeps its measured
    per-layer 'auto'. Scoping the fp4 change with an unconditional
    ``or 'stock'`` would clobber it, which is the failure this test exists
    to catch."""
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL", raising=False)
    assert _dispatch_mode(moe, monkeypatch, _fp8(moe)) == "auto"


@pytest.mark.parametrize("mode", ["stock", "loop", "batched",
                                  "grouped_fused", "l2_pipeline"])
def test_explicit_fp4_mode_bypasses_fused_default(moe, monkeypatch, mode):
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", mode)
    assert _dispatch_mode(moe, monkeypatch, _fp4(moe)) == mode


def test_fp8_explicit_mode_still_overrides_auto(moe, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "loop")
    assert _dispatch_mode(moe, monkeypatch, _fp8(moe)) == "loop"


# --------------------------------------------------------------------------- #
# 2 — _stock_chunk_default: fp8 unchanged, fp4 byte-budgeted.                    #
# --------------------------------------------------------------------------- #
def test_stock_chunk_fp8_is_unchanged(moe, monkeypatch):
    """fp8-CB keeps the flat 256 the fp8 lane was measured on."""
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    assert _fp8(moe)._stock_chunk_default(_layer(), torch.bfloat16) == 256


def test_stock_chunk_fp4_is_byte_budgeted(moe, monkeypatch):
    """The fp4 transient is bf16 with no CUDA expander, so a flat 256 holds all
    192 Hy3 experts live (4.83 GB analytic per stage). The byte budget picks the
    largest chunk that fits 1 GiB instead."""
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    c = _fp4(moe)._stock_chunk_default(_layer(), torch.bfloat16)
    tile = 2 * HY3["inter"] * HY3["hidden"] * 2          # w13 bf16 bytes/expert
    assert c == 42
    assert c * tile <= GIB < (c + 1) * tile              # maximal fit
    assert HY3["E"] * tile > 4.8e9                       # the balloon avoided


def test_stock_chunk_env_overrides(moe, monkeypatch):
    """Both knobs still work, and the explicit expert-chunk wins outright."""
    m = _fp4(moe)
    lay = _layer()
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", str(1 << 28))
    assert m._stock_chunk_default(lay, torch.bfloat16) == 10
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", "7")
    assert m._stock_chunk_default(lay, torch.bfloat16) == 7


@pytest.mark.parametrize("name", ["PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
                                  "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES"])
@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_stock_chunk_rejects_invalid_environment(moe, monkeypatch, name,
                                                  value):
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        _fp4(moe)._stock_chunk_default(_layer(), torch.bfloat16)


def test_stock_chunk_never_zero_or_over_e(moe, monkeypatch, capsys):
    """A budget smaller than one expert still runs (chunk 1), and a budget
    bigger than the whole stack never exceeds E."""
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    m = _fp4(moe)
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod,
                        "_STOCK_BUDGET_WARNED", False)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", "1")
    assert m._stock_chunk_default(_layer(), torch.bfloat16) == 1
    assert "budget cannot be met" in capsys.readouterr().err
    assert m._stock_chunk_default(_layer(), torch.bfloat16) == 1
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", str(1 << 40))
    assert m._stock_chunk_default(_layer(), torch.bfloat16) == HY3["E"]


def test_stock_chunk_uses_bf16_tile_size_for_fp32_inputs(moe, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    m = _fp4(moe)
    assert m._stock_chunk_default(_layer(), torch.float32) == 42
    assert m._stock_chunk_default(_layer(), torch.bfloat16) == 42


# --------------------------------------------------------------------------- #
# 3 — the pad contract and the raw slice view.                                  #
# --------------------------------------------------------------------------- #
def test_window_fits_superblock_arithmetic(moe):
    """The window the expanders read is bytes ``(31k)//8 .. +7`` of a
    superblock. fp4 v2 (type_size = 4k+9) is always in bounds -> the raw view is
    safe; fp8 (type_size = 4k) overruns at every shipped k -> its Triton branch
    must KEEP the pad."""
    fits = moe._window_fits_superblock
    for k in range(1, 65):
        assert fits(k, 4 * k + 9), f"fp4 v2 k={k} should not need the pad"
    for k in (28, 32, 36, 40, 44, 48):                  # the shipped fp8 rungs
        assert not fits(k, 4 * k), f"fp8 k={k} must keep the pad"
    assert fits(64, 4 * 64)                             # the k > 56 crossover
    assert not fits(56, 4 * 56)


def _slice_probe(moe, monkeypatch, which, k, type_size,
                 E=4, hidden=512, inter=256):
    """Drive ``_expand_stack_slice`` with the fp4 expander replaced by a
    recorder, and report the qweight tensor it was actually handed."""
    m = _fp4(moe, k=k)
    m.type_size = type_size
    lay = _layer(E=E, hidden=hidden, inter=inter)
    lay._cb_flat = torch.zeros(8)
    lay._cb_compose = torch.zeros(4096)
    out_f = 2 * inter if which == "w13" else hidden
    in_f = hidden if which == "w13" else inter
    row_bytes = (in_f // 256) * type_size
    setattr(lay, f"{which}_cb_qweight",
            torch.randint(0, 256, (E, out_f, row_bytes), dtype=torch.uint8))
    rec = {}

    def _fake(qw, cb_flat, row0, compose, N, K, k_bits, n_sub, ts):
        rec.update(qw=qw, N=N, K=K, k=k_bits, ts=ts)
        return torch.zeros(N, K, dtype=torch.bfloat16)

    monkeypatch.setattr(moe, "expand_fp4_v2_to_weight", _fake)
    W = m._expand_stack_slice(lay, which, 1, 3, to_fp8=False)
    return rec, getattr(lay, f"{which}_cb_qweight"), W


def test_expand_stack_slice_fp4_is_raw_view(moe, monkeypatch):
    """The fp4 branch hands the expander the RAW slice view: no pad, no
    ``.contiguous()`` copy, same storage as the resident buffer."""
    monkeypatch.delenv("PRISMAQUANT_CB_EXPAND", raising=False)
    rec, packed, W = _slice_probe(moe, monkeypatch, "w13", 16, 73)
    qw = rec["qw"]
    assert qw.shape == (2 * packed.shape[1], packed.shape[2])   # UNPADDED
    assert qw.data_ptr() == packed[1:3].data_ptr()              # a view
    assert (qw.untyped_storage().data_ptr()
            == packed.untyped_storage().data_ptr())
    assert qw.dtype == torch.uint8 and qw.stride(1) == 1
    assert qw.stride(0) == packed.shape[2]        # unpadded row stride is legal
    assert W.shape == (2, packed.shape[1], 512)


def test_expand_stack_slice_fp4_pad_escape(moe, monkeypatch):
    """PRISMAQUANT_CB_EXPAND=pad restores the padded copy (bisection escape)."""
    from gridbook import codec
    monkeypatch.setenv("PRISMAQUANT_CB_EXPAND", "pad")
    rec, packed, _ = _slice_probe(moe, monkeypatch, "w13", 16, 73)
    assert rec["qw"].shape[1] == packed.shape[2] + codec.PAD_BYTES
    assert rec["qw"].data_ptr() != packed[1:3].data_ptr()


def test_expand_stack_slice_fp4_pads_when_window_would_overrun(moe, monkeypatch):
    """The raw view is taken because the arithmetic says it is safe, not because
    the branch is fp4. A (hypothetical) fp4 rung whose type_size leaves no tail
    slack falls back to the pad instead of reading out of bounds."""
    from gridbook import codec
    monkeypatch.delenv("PRISMAQUANT_CB_EXPAND", raising=False)
    assert not moe._window_fits_superblock(16, 4 * 16)      # premise
    rec, packed, _ = _slice_probe(moe, monkeypatch, "w13", 16, 4 * 16)
    assert rec["qw"].shape[1] == packed.shape[2] + codec.PAD_BYTES


def test_expand_stack_slice_fp4_per_stack_shapes(moe, monkeypatch):
    """w2's slice is sized from ``_cb_inter``, w13's from ``_cb_hidden`` — the
    raw view must not disturb that."""
    monkeypatch.delenv("PRISMAQUANT_CB_EXPAND", raising=False)
    rec, packed, W = _slice_probe(moe, monkeypatch, "w2", 16, 73)
    assert rec["K"] == 256 and rec["N"] == 2 * packed.shape[1]
    assert rec["qw"].shape[1] == packed.shape[2]
    assert W.shape == (2, packed.shape[1], 256)


def test_expand_stack_slice_under_bf16_default_dtype(moe, monkeypatch):
    """vLLM runs the forward under ``torch.set_default_dtype(bfloat16)``; the
    raw view must stay a uint8 view of the resident buffer regardless."""
    monkeypatch.delenv("PRISMAQUANT_CB_EXPAND", raising=False)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rec, packed, _ = _slice_probe(moe, monkeypatch, "w13", 16, 73)
    finally:
        torch.set_default_dtype(old)
    assert rec["qw"].dtype == torch.uint8
    assert rec["qw"].data_ptr() == packed[1:3].data_ptr()
