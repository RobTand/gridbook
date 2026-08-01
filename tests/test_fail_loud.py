"""Fail-closed contracts in the native CB MoE path.

CPU-only, no CUDA, no nvcc, no vLLM: the expanders and the JIT-extension probe
are monkeypatched, so what is pinned here is CONTROL FLOW and DIAGNOSTICS, not
kernel numerics.

  1. FP8 expansion must raise ``NativeKernelUnavailableError`` when its native
     extension cannot load. There is no interpreted serving fallback.
  2. ``_assert_cast_lossless`` must reject a codebook table that the bf16 /
     e4m3 cast rounds, and must say by how much. The casts are exact only for
     tables already on those grids.
  3. ``_cuda_moe_ok`` must say so on stderr when the grouped CUDA decode path
     disengages, and print one ``decode=grouped-cuda`` line per process when it
     engages, so a log can prove which decode path a run actually used.

Instances are built with ``__new__`` and explicit attributes, the same
init-bypass used by ``test_moe_grouped_fused`` and ``test_moe_stacked``:
nothing here exercises ``__init__``.

Run: ``python -m pytest tests/test_fail_loud.py -q``. When vLLM is absent the
module-scoped runtime fixture installs private stubs and restores the exact
prior import graph afterward, so this file is safe in a combined pytest run.
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

def _install_vllm_stubs():
    """Install the vLLM surface imported by Gridbook's MoE runtime."""
    def _mod(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, module)
        return module

    _mod("vllm")
    _mod("vllm.model_executor")
    model_utils = _mod("vllm.model_executor.utils")
    model_utils.set_weight_attrs = lambda p, attrs: None
    _mod("vllm.model_executor.layers")
    fused_moe = _mod("vllm.model_executor.layers.fused_moe")
    fused_moe.RoutedExperts = type("RoutedExperts", (), {})
    config = _mod("vllm.model_executor.layers.fused_moe.config")
    config.FusedMoEConfig = type("FusedMoEConfig", (), {})
    config.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
    base = _mod("vllm.model_executor.layers.fused_moe.fused_moe_method_base")
    base.FusedMoEMethodBase = type("FusedMoEMethodBase", (), {})


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    """Import Gridbook against a private real-or-stubbed vLLM graph."""
    del isolated_gridbook_runtime_imports
    try:
        from vllm.model_executor.layers.fused_moe.config import (  # noqa: F401
            FusedMoEConfig,
        )
    except Exception:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                sys.modules.pop(name, None)
        _install_vllm_stubs()

    globals()["codec"] = pytest.importorskip("gridbook.codec")
    globals()["moe"] = pytest.importorskip("gridbook.moe")
    globals()["cuda_ext"] = pytest.importorskip("gridbook.cuda_ext")

K, N_SUB = 44, 4                       # the shipped fp8-CB rung
TYPE_SIZE = 4 * K                      # fp8-CB superblock byte width


@pytest.fixture(autouse=True)
def _reset_process_state(monkeypatch):
    """The two decode latches are PROCESS-wide by design (one line per
    process, not per layer). Reset them around every test so ordering cannot
    make a later test pass on an earlier test's cached probe."""
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod,
                        "_DECODE_ENGAGED_LOGGED", False)
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod,
                        "_DECODE_DISABLED_LOGGED", False)
    monkeypatch.delenv("PRISMAQUANT_SKIP_CB_CAST_CHECK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_DECODE", raising=False)


def _method(prefix="model.layers.0.mlp.experts", is_fp4=False, is_v2=False):
    m = moe.PrismaQuantCBMoEMethod.__new__(moe.PrismaQuantCBMoEMethod)
    m.quant_config = None
    m.scheme = {"grid": "fp8", "mode": "product", "k": K, "n_sub": N_SUB,
                "type_size": TYPE_SIZE}
    m.prefix = prefix
    m.is_fp4 = is_fp4
    m.is_v2 = is_v2
    m.k = K
    m.n_sub = N_SUB
    m.type_size = TYPE_SIZE
    m._sub_table = None
    return m


def _layer(E=2, hidden=256, inter=256, flat=None):
    lay = types.SimpleNamespace()
    lay._cb_E = E
    lay._cb_hidden = hidden
    lay._cb_inter = inter
    lay._cb_flat = _on_grid_flat() if flat is None else flat
    rb = moe._row_bytes(hidden, TYPE_SIZE)
    lay.w13_cb_qweight = torch.zeros(E, 2 * inter, rb, dtype=torch.uint8)
    lay.w2_cb_qweight = torch.zeros(
        E, hidden, moe._row_bytes(inter, TYPE_SIZE), dtype=torch.uint8)
    return lay


def _on_grid_flat(n=64):
    """A LATTICE-style table: every value already sits on the e4m3 grid, so
    both the bf16 and the e4m3 cast are exact (this is why every artifact
    shipped so far passes the check by construction)."""
    v = torch.arange(n, dtype=torch.float32) / 64.0 - 0.5
    return v.to(torch.float8_e4m3fn).to(torch.bfloat16)


# --------------------------------------------------------------- (2) the cast
def test_cast_check_accepts_an_on_grid_table():
    flat = _on_grid_flat()
    cast = flat.to(torch.float8_e4m3fn)
    moe._assert_cast_lossless(flat, cast, "e4m3 flat", "t")   # must not raise


def test_cast_check_rejects_off_grid_values_with_diagnostics():
    """A malformed learned table can carry values between e4m3 grid points.
    Rounding them at load is invisible downstream -- the packed weight bytes are
    untouched and every shape/uniformity assert still passes -- so the cast has
    to refuse, and say how bad it is."""
    flat = _on_grid_flat(8).float()
    # 0.5 -> next e4m3 code is 0.5625; land halfway, off-grid by construction.
    flat[3] = 0.53125
    cast = flat.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError) as e:
        moe._assert_cast_lossless(flat, cast, "e4m3 flat",
                                  "model.layers.3.mlp.experts.w13")
    msg = str(e.value)
    assert "LOSSY" in msg
    assert "model.layers.3.mlp.experts.w13" in msg
    assert "e4m3 flat" in msg
    assert "1 of 8 table values do not survive it" in msg   # element count
    assert "max abs" in msg and "max rel" in msg            # both error fields
    assert "PRISMAQUANT_SKIP_CB_CAST_CHECK=1" in msg        # the escape hatch


def test_cast_check_reports_real_error_magnitudes():
    """The two numbers in the message are the measured errors, not adjectives."""
    flat = torch.tensor([0.53125], dtype=torch.float32)
    cast = flat.to(torch.float8_e4m3fn)
    got = float(cast.float()[0])
    with pytest.raises(ValueError) as e:
        moe._assert_cast_lossless(flat, cast, "e4m3 flat", "t")
    msg = str(e.value)
    assert f"max abs {abs(0.53125 - got):.3e}" in msg
    assert f"max rel {abs(0.53125 - got) / 0.53125:.3e}" in msg


def test_cast_check_env_escape_warns_instead_of_raising(monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_SKIP_CB_CAST_CHECK", "1")
    flat = torch.tensor([0.53125], dtype=torch.float32)
    moe._assert_cast_lossless(flat, flat.to(torch.float8_e4m3fn),
                              "e4m3 flat", "t")              # must not raise
    err = capsys.readouterr().err
    assert "WARNING" in err and "LOSSY" in err


def test_stock_flat_fp8_checks_the_e4m3_cast():
    """The load-time e4m3 re-encode in _stock_cb_flat_fp8 is the second cast;
    "every CB value is on the e4m3 grid -- lossless" used to be a comment
    there, with nothing enforcing it."""
    m = _method()
    off_grid = torch.tensor([0.53125, 0.25, -0.125], dtype=torch.bfloat16)
    lay = _layer(flat=off_grid)
    with pytest.raises(ValueError, match="e4m3 flat"):
        m._stock_cb_flat_fp8(lay)
    # an on-grid table still builds, and is cached on the layer
    lay2 = _layer()
    cb = m._stock_cb_flat_fp8(lay2)
    assert cb.dtype is torch.uint8
    assert lay2._cb_flat_fp8 is cb


def test_process_weights_bf16_cast_is_checked():
    """codec.build_flat_codebook casts each sub-table to bf16; the same
    argument applies, so process_weights_after_loading checks it too."""
    subs = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)]
    moe._assert_cast_lossless(torch.cat([t.reshape(-1).float() for t in subs]),
                              codec.build_flat_codebook(subs), "bf16 flat", "t")
    subs_off = [torch.tensor([[1.0000001, 2.0]], dtype=torch.float32),
                torch.tensor([[0.5]], dtype=torch.bfloat16)]   # mixed dtypes
    with pytest.raises(ValueError, match="bf16 flat"):
        moe._assert_cast_lossless(
            torch.cat([t.reshape(-1).float() for t in subs_off]),
            codec.build_flat_codebook(subs_off), "bf16 flat", "t")


def test_build_flat_codebook_checks_every_runtime_caller():
    """The bf16 check belongs in the shared codec helper, not only MoE."""
    with pytest.raises(ValueError, match="bf16 flat"):
        codec.build_flat_codebook(
            [torch.tensor([1.0000001], dtype=torch.float32)], "dense.layer")


def test_build_flat_codebook_does_not_pre_round_float64_input():
    raw = torch.tensor([1.0 + 2.0 ** -40], dtype=torch.float64)
    with pytest.raises(ValueError, match="bf16 flat"):
        codec.build_flat_codebook([raw], "float64.layer")


def test_shared_fp8_reencode_helper_checks_dense_and_moe_callers():
    flat = torch.tensor([0.53125], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="e4m3 flat"):
        codec.flat_codebook_fp8(flat, "dense.layer")


@pytest.mark.parametrize("value", [0.53125, float("nan"), float("inf")])
def test_fp4_grid_validation_rejects_malformed_tables_at_load(value):
    with pytest.raises(ValueError, match="E2M1 grid|non-finite"):
        codec.build_flat_codebook(
            [torch.tensor([0.0, -6.0, value], dtype=torch.float32)],
            "fp4.layer", "fp4")


def test_fp4_grid_validation_accepts_learned_but_grid_valued_table():
    # A learned table is valid; it need not equal the deterministic lattice.
    flat = codec.build_flat_codebook(
        [torch.tensor([-6.0, -1.5, -0.0, 0.5, 3.0])],
        "fp4.learned", "fp4")
    assert flat.dtype is torch.bfloat16


@pytest.mark.parametrize("value", [0.53125, float("nan"), float("inf")])
def test_fp8_grid_validation_rejects_malformed_tables_at_load(value):
    with pytest.raises(ValueError, match="E4M3 grid|non-finite"):
        codec.build_flat_codebook(
            [torch.tensor([0.0, -0.5, value], dtype=torch.float32)],
            "fp8.layer", "fp8")


# ------------------------------------------------ (1) native expand fail-closed
def test_ops_cb_expand_fp8_without_ext_raises(monkeypatch):
    """The raw op reports a native-kernel failure, never AttributeError."""
    ops = pytest.importorskip("gridbook.ops")
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    qw = torch.zeros(4, TYPE_SIZE, dtype=torch.uint8)
    with pytest.raises(cuda_ext.NativeKernelUnavailableError,
                       match="FP8-CB transient expansion"):
        ops.cb_expand_fp8(qw, torch.zeros(64, dtype=torch.uint8),
                          torch.zeros(4, dtype=torch.int32),
                          4, 256, K, N_SUB, TYPE_SIZE)


# --------------------------------------------- (3) decode-path disengagement
def test_decode_disengagement_is_loud(monkeypatch, capsys):
    """No extension must identify disengagement without claiming a fallback."""
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    m = _method()
    lay = _layer()
    assert m._cuda_moe_ok(lay) is False
    out = capsys.readouterr()
    assert "grouped CUDA decode" in out.err and "ERROR" in out.err
    assert "decode=grouped-cuda" not in out.out       # never claim engagement


def test_decode_disengagement_warning_is_process_deduplicated(monkeypatch,
                                                               capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    assert _method()._cuda_moe_ok(_layer()) is False
    assert "ERROR" in capsys.readouterr().err
    assert _method(prefix="model.layers.1.mlp.experts")._cuda_moe_ok(
        _layer()) is False
    assert capsys.readouterr().err == ""


def test_decode_engagement_line_is_printed_once(monkeypatch, capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: object())
    m = _method()
    assert m._cuda_moe_ok(_layer()) is True
    assert "[prismaquant-cb] decode=grouped-cuda" in capsys.readouterr().out
    # per PROCESS, not per layer: a 48-layer model must not print 48 lines
    m2 = _method(prefix="model.layers.1.mlp.experts")
    assert m2._cuda_moe_ok(_layer()) is True
    assert "decode=grouped-cuda" not in capsys.readouterr().out


def test_cuda_gate_builds_the_fp8_lut_through_the_checked_helper(monkeypatch):
    """The gate used to duplicate the e4m3 cast inline, which would have
    skipped the check. It now goes through _stock_cb_flat_fp8."""
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: object())
    m = _method()
    lay = _layer(flat=torch.tensor([0.53125], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="e4m3 flat"):
        m._cuda_moe_ok(lay)
