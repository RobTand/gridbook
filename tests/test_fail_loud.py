"""Three silent degrades in the CB MoE path, made loud.

CPU-only, no CUDA, no nvcc, no vLLM: the expanders and the JIT-extension probe
are monkeypatched, so what is pinned here is CONTROL FLOW and DIAGNOSTICS, not
kernel numerics.

  1. ``_expand_stack_slice``'s fp8 branch must consult ``_cb_expand_ext_ok()``
     before reaching for ``ops.cb_expand_fp8``. That op dereferences
     ``cuda_ext.get_ext()`` with no None check of its own, so an unguarded call
     raises ``AttributeError`` mid-prefill on any host where the JIT build
     failed -- instead of the Triton fallback ``cuda_ext``'s module docstring
     advertises. ``test_ops_cb_expand_fp8_without_ext_raises`` pins the raw
     failure mode; the two tests after it pin the guard.
  2. ``_assert_cast_lossless`` must reject a codebook table that the bf16 /
     e4m3 cast rounds, and must say by how much. The casts are exact only for
     tables already on those grids.
  3. ``_cuda_moe_ok`` must say so on stderr when the grouped CUDA decode path
     disengages, and print one ``decode=grouped-cuda`` line per process when it
     engages, so a log can prove which decode path a run actually used.

Instances are built with ``__new__`` and explicit attributes, the same
init-bypass the other MoE test files use (``test_moe_stock_prefill._build``,
``test_moe_stacked``): nothing here exercises ``__init__``.

Run: ``python -m pytest tests/test_fail_loud.py -q``. Like
tests/test_target_namespace_compat.py this file injects stub ``vllm.*`` modules
into ``sys.modules`` when vLLM is absent, so it must get its OWN pytest process
-- which is what .github/scripts/run_cpu_tests.sh already does for every file.
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

# vLLM symbols gridbook.moe imports at module scope. Stubbed only when vLLM is
# genuinely absent, exactly as tests/test_target_namespace_compat.py does it.
if "vllm" not in sys.modules:
    try:
        import vllm  # noqa: F401
    except Exception:
        def _mod(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], leaf, m)
            return m

        _mod("vllm")
        _mod("vllm.model_executor")
        me_utils = _mod("vllm.model_executor.utils")
        me_utils.set_weight_attrs = lambda p, attrs: None
        _mod("vllm.model_executor.layers")
        fm = _mod("vllm.model_executor.layers.fused_moe")
        fm.RoutedExperts = type("RoutedExperts", (), {})
        fmc = _mod("vllm.model_executor.layers.fused_moe.config")
        fmc.FusedMoEConfig = type("FusedMoEConfig", (), {})
        fmc.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
        fmb = _mod("vllm.model_executor.layers.fused_moe.fused_moe_method_base")
        fmb.FusedMoEMethodBase = type("FusedMoEMethodBase", (), {})
        fma = _mod("vllm.model_executor.layers.fused_moe.activation")
        fma.MoEActivation = type("MoEActivation", (), {})
        fma.apply_moe_activation = lambda *a, **kw: None

codec = pytest.importorskip("gridbook.codec")
moe = pytest.importorskip("gridbook.moe")
cuda_ext = pytest.importorskip("gridbook.cuda_ext")

K, N_SUB = 44, 4                       # the shipped fp8-CB rung
TYPE_SIZE = 4 * K                      # fp8-CB superblock byte width


@pytest.fixture(autouse=True)
def _reset_process_state(monkeypatch):
    """The two fail-loud latches are PROCESS-wide by design (one line per
    process, not per layer). Reset them around every test so ordering cannot
    make a later test pass on an earlier test's cached probe."""
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod, "_EXPAND_EXT_OK", None)
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod,
                        "_DECODE_ENGAGED_LOGGED", False)
    monkeypatch.setattr(moe.PrismaQuantCBMoEMethod,
                        "_DECODE_DISABLED_LOGGED", False)
    monkeypatch.delenv("PRISMAQUANT_SKIP_CB_CAST_CHECK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_EXPAND", raising=False)
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


# ------------------------------------------------- (1) the expand fail-soft
def test_ops_cb_expand_fp8_without_ext_raises(monkeypatch):
    """THE failure this PR is about: ops.cb_expand_fp8 has no None check --
    its docstring pushes that onto the caller (gridbook/ops.py:107) -- so an
    unguarded call on an ext-less host is an AttributeError, not a fallback."""
    ops = pytest.importorskip("gridbook.ops")
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    qw = torch.zeros(4, TYPE_SIZE, dtype=torch.uint8)
    with pytest.raises(AttributeError):
        ops.cb_expand_fp8(qw, torch.zeros(64, dtype=torch.uint8),
                          torch.zeros(4, dtype=torch.int32),
                          4, 256, K, N_SUB, TYPE_SIZE)


def test_expand_ext_probe_is_false_and_loud(monkeypatch, capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    cls = moe.PrismaQuantCBMoEMethod
    assert cls._cb_expand_ext_ok() is False
    err = capsys.readouterr().err
    assert "CUDA expand extension" in err and "WARNING" in err
    # cached: probed once per process, no second warning, no second get_ext call
    monkeypatch.setattr(cuda_ext, "get_ext",
                        lambda: pytest.fail("probe was not cached"))
    assert cls._cb_expand_ext_ok() is False
    assert capsys.readouterr().err == ""


def test_expand_ext_probe_requires_the_called_symbol(monkeypatch, capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: object())
    cls = moe.PrismaQuantCBMoEMethod
    assert cls._cb_expand_ext_ok() is False
    assert "CUDA expand extension" in capsys.readouterr().err


def test_expand_stack_slice_falls_back_to_triton_without_ext(monkeypatch,
                                                             capsys):
    """With no extension the fp8 branch must take the padded Triton expander
    and keep going -- never AttributeError out of the middle of a prefill."""
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    ops = pytest.importorskip("gridbook.ops")
    monkeypatch.setattr(ops, "cb_expand_fp8", lambda *a, **kw: pytest.fail(
        "unguarded CUDA expander call on an ext-less host"))

    seen = {}

    def _fake_triton(qwp, lut, row0, N, Kdim, k, n_sub, ts):
        seen["rows"] = N
        seen["pad"] = qwp.shape[1] - TYPE_SIZE * (Kdim // codec.SUPERBLOCK)
        return torch.zeros(N, Kdim, dtype=torch.uint8)

    monkeypatch.setattr(moe, "expand_cb_to_fp8", _fake_triton)

    m = _method()
    lay = _layer(E=2, hidden=256, inter=256)
    W = m._expand_stack_slice(lay, "w13", 0, 2, to_fp8=True)

    assert seen["rows"] == 2 * (2 * 256)          # nE * out_f
    assert seen["pad"] == codec.PAD_BYTES         # the padded Triton contract
    assert tuple(W.shape) == (2, 2 * 256, 256)
    assert "CUDA expand extension" in capsys.readouterr().err


def test_expand_stack_slice_uses_the_cuda_op_when_the_ext_is_present(
        monkeypatch):
    """The guard must not cost the fast path: with an extension loaded the
    branch still goes to ops.cb_expand_fp8 on the RAW unpadded slice view."""
    monkeypatch.setattr(
        cuda_ext, "get_ext", lambda: types.SimpleNamespace(cb_expand_fp8=True))
    ops = pytest.importorskip("gridbook.ops")
    monkeypatch.setattr(moe, "expand_cb_to_fp8", lambda *a, **kw: pytest.fail(
        "fell back to Triton with the extension available"))

    seen = {}

    def _fake_op(qw, lut, row0, N, Kdim, k, n_sub, ts):
        seen["rows"] = N
        seen["row_bytes"] = qw.shape[1]
        return torch.zeros(N, Kdim, dtype=torch.uint8)

    monkeypatch.setattr(ops, "cb_expand_fp8", _fake_op)

    m = _method()
    lay = _layer(E=2, hidden=256, inter=256)
    W = m._expand_stack_slice(lay, "w13", 0, 2, to_fp8=True)
    assert seen["rows"] == 2 * (2 * 256)
    assert seen["row_bytes"] == moe._row_bytes(256, TYPE_SIZE)   # NOT padded
    assert tuple(W.shape) == (2, 2 * 256, 256)


def test_expand_env_override_still_forces_triton(monkeypatch):
    """PRISMAQUANT_CB_EXPAND=triton keeps working with the extension loaded
    (the bisection lever upstream documents)."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXPAND", "triton")
    monkeypatch.setattr(
        cuda_ext, "get_ext", lambda: types.SimpleNamespace(cb_expand_fp8=True))
    ops = pytest.importorskip("gridbook.ops")
    monkeypatch.setattr(ops, "cb_expand_fp8", lambda *a, **kw: pytest.fail(
        "env override ignored"))
    seen = {}
    monkeypatch.setattr(moe, "expand_cb_to_fp8",
                        lambda qwp, lut, row0, N, Kd, *a: (
                            seen.setdefault("hit", True),
                            torch.zeros(N, Kd, dtype=torch.uint8))[1])
    _method()._expand_stack_slice(_layer(), "w13", 0, 2, to_fp8=True)
    assert seen.get("hit") is True


# --------------------------------------------- (3) decode-path disengagement
def test_decode_disengagement_is_loud(monkeypatch, capsys):
    """No extension must identify disengagement without claiming a fallback."""
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    m = _method()
    lay = _layer()
    assert m._cuda_moe_ok(lay) is False
    out = capsys.readouterr()
    assert "grouped CUDA decode" in out.err and "WARNING" in out.err
    assert "decode=grouped-cuda" not in out.out       # never claim engagement


def test_decode_disengagement_warning_is_process_deduplicated(monkeypatch,
                                                               capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: None)
    assert _method()._cuda_moe_ok(_layer()) is False
    assert "WARNING" in capsys.readouterr().err
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


def test_decode_env_disable_is_also_reported(monkeypatch, capsys):
    """PRISMAQUANT_CB_DECODE=triton is a deliberate override, so it gets no
    extension WARNING -- but it must still not print the engagement line."""
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE", "triton")
    monkeypatch.setattr(cuda_ext, "get_ext",
                        lambda: pytest.fail("probed despite the env gate"))
    assert _method()._cuda_moe_ok(_layer()) is False
    out = capsys.readouterr()
    assert "decode=grouped-cuda" not in out.out


def test_cuda_gate_builds_the_fp8_lut_through_the_checked_helper(monkeypatch):
    """The gate used to duplicate the e4m3 cast inline, which would have
    skipped the check. It now goes through _stock_cb_flat_fp8."""
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: object())
    m = _method()
    lay = _layer(flat=torch.tensor([0.53125], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="e4m3 flat"):
        m._cuda_moe_ok(lay)
