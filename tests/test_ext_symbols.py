"""Loader-side symbol assertion for the JIT extensions (``gridbook.cuda_ext``).

GPU-FREE and TOOLCHAIN-FREE: nothing here builds or loads a real extension.
``_require_symbols`` is exercised against stand-in module objects, and the
required-symbol tuples are cross-checked against the ``m.def(...)`` bindings in
the packaged ``.cu`` sources so the two cannot drift apart. Runs anywhere the
package imports (``cuda_ext`` itself imports only ``os`` and ``sys``); the one
test that drives ``get_ext()`` end to end needs ``torch`` importable but never
a GPU, an ``nvcc`` or a build -- ``load`` is monkeypatched.

    python -m pytest tests/test_ext_symbols.py -q
"""
import re
import sys
import types

import pytest

cuda_ext = pytest.importorskip(
    "gridbook.cuda_ext", reason="gridbook not importable")


def _stub(name, symbols):
    """A stand-in for a torch-JIT extension module exporting ``symbols``."""
    mod = types.ModuleType(name)
    for s in symbols:
        setattr(mod, s, lambda *a, **k: None)
    return mod


# --------------------------------------------------------------------------
# _require_symbols
# --------------------------------------------------------------------------

def test_complete_module_is_returned_unchanged():
    mod = _stub("prismaquant_cb_ext", cuda_ext._EXT_SYMBOLS)
    got = cuda_ext._require_symbols(mod, cuda_ext._EXT_SYMBOLS,
                                    build_dir="/opt/gridbook/ext-cache",
                                    source="cb_gemv.cu")
    assert got is mod


def test_extra_symbols_do_not_matter():
    """The check is a lower bound, not an equality: an extension carrying the
    optional bindings on top of the required set is the normal case."""
    mod = _stub("prismaquant_cb_ext",
                (*cuda_ext._EXT_SYMBOLS, "cb_expand_fp8_into", "l2_unpin"))
    assert cuda_ext._require_symbols(
        mod, cuda_ext._EXT_SYMBOLS, build_dir="/cache",
        source="cb_gemv.cu") is mod


def test_missing_symbol_raises_and_names_it_and_the_directory():
    """The whole point of the error: it must say WHAT is missing and WHICH
    directory to delete. A bare AttributeError from a call site says
    neither."""
    present = [s for s in cuda_ext._EXT_SYMBOLS if s != "cb_moe_combine"]
    mod = _stub("prismaquant_cb_ext", present)
    with pytest.raises(cuda_ext.StaleExtensionError) as ei:
        cuda_ext._require_symbols(mod, cuda_ext._EXT_SYMBOLS,
                                  build_dir="/opt/gridbook/ext-cache",
                                  source="cb_gemv.cu")
    msg = str(ei.value)
    assert "cb_moe_combine" in msg
    assert "/opt/gridbook/ext-cache" in msg
    assert "cb_gemv.cu" in msg
    assert "PRISMAQUANT_CB_EXT_DIR" in msg


def test_every_missing_symbol_is_reported_not_just_the_first():
    """A stale .so usually lags by more than one binding; reporting one at a
    time would mean one restart per symbol to find out."""
    mod = _stub("prismaquant_cb_ext", ("fp8_act_qdq",))
    with pytest.raises(cuda_ext.StaleExtensionError) as ei:
        cuda_ext._require_symbols(mod, cuda_ext._EXT_SYMBOLS,
                                  build_dir="/cache", source="cb_gemv.cu")
    msg = str(ei.value)
    for s in cuda_ext._EXT_SYMBOLS:
        if s != "fp8_act_qdq":
            assert s in msg, f"{s} missing from the error message"


def test_stale_extension_error_is_not_confused_with_the_other_two():
    """cuda_ext reports three different defects three different ways; the
    except arms rely on the classes being disjoint."""
    assert issubclass(cuda_ext.StaleExtensionError, RuntimeError)
    assert not issubclass(cuda_ext.StaleExtensionError,
                          cuda_ext.IncompleteInstallError)
    assert not issubclass(cuda_ext.IncompleteInstallError,
                          cuda_ext.StaleExtensionError)


# --------------------------------------------------------------------------
# the required sets themselves
# --------------------------------------------------------------------------

_M_DEF = re.compile(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def _exports(cu_name):
    """Symbol names bound by ``m.def("...")`` in a packaged CUDA source."""
    import os
    path = os.path.join(cuda_ext.csrc_dir(), cu_name)
    if not os.path.isfile(path):
        pytest.skip(f"{cu_name} not present in this install")
    with open(path, encoding="utf-8") as fh:
        return set(_M_DEF.findall(fh.read()))


@pytest.mark.parametrize("required,cu_name", [
    (cuda_ext._EXT_SYMBOLS, "cb_gemv.cu"),
    (cuda_ext._PTC_SYMBOLS, "cb_persistent_tc.cu"),
    (cuda_ext._FUSED_SYMBOLS, "cb_fused_gemm.cu"),
])
def test_required_symbols_exist_in_the_packaged_source(required, cu_name):
    """The assertion lists must not drift from the kernels they describe. A
    typo here would make a HEALTHY build fail to load on every host, which is a
    worse outcome than the bug this check exists to catch -- so pin it."""
    exported = _exports(cu_name)
    missing = sorted(set(required) - exported)
    assert not missing, (
        f"{cu_name} does not m.def() {missing}; the required-symbol tuple in "
        f"cuda_ext is stale or misspelled")


def test_optional_bindings_stay_optional():
    """These have call-site probes that treat absence as 'an older build' and
    degrade correctly. Promoting any of them to required would turn a working
    degrade into a hard fallback to Triton for every older extension."""
    for name in ("cb_expand_fp8_into",      # ops.cb_expand_fp8_into_available
                 "l2_pin_region", "l2_reset_window", "l2_unpin",
                 "l2_persisting_max_bytes", "l2_max_window_bytes"):
        assert name not in cuda_ext._EXT_SYMBOLS, (
            f"{name} is probed at its call site and must stay optional")
    for name in ("cb_fused_moe_grouped", "cb_fused_moe_tile_m",
                 "cb_fused_moe_tile_sizes",
                 "cb_fused_moe_tile_sizes_for_kbits"):
        assert name not in cuda_ext._FUSED_SYMBOLS, (
            f"{name} ships independently and must stay optional")


# --------------------------------------------------------------------------
# end to end through get_ext(), with the build monkeypatched out
# --------------------------------------------------------------------------

@pytest.fixture
def fresh_get_ext():
    """Reset get_ext's one-shot memo around a test, and restore it after."""
    saved = (cuda_ext._ext, cuda_ext._tried)
    cuda_ext._ext, cuda_ext._tried = None, False
    yield
    cuda_ext._ext, cuda_ext._tried = saved


def _patch_load(monkeypatch, result):
    torch = pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    assert torch is not None
    calls = {}

    def fake_load(*args, **kwargs):
        calls["kwargs"] = kwargs
        return result

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    return calls


def test_get_ext_refuses_a_stale_module(monkeypatch, capsys, tmp_path,
                                        fresh_get_ext):
    """The regression this PR is about: a module that loaded but lacks a symbol
    ops.py dereferences unconditionally must NOT become get_ext()'s value."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    stale = _stub("prismaquant_cb_ext",
                  [s for s in cuda_ext._EXT_SYMBOLS if s != "cb_gemv_fp4_v2"])
    _patch_load(monkeypatch, stale)

    assert cuda_ext.get_ext() is None
    err = capsys.readouterr().err
    assert "stale" in err.lower()
    assert "cb_gemv_fp4_v2" in err
    assert str(tmp_path) in err


def test_get_ext_accepts_a_complete_module(monkeypatch, tmp_path,
                                           fresh_get_ext):
    """...and the check does not reject a healthy build."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    good = _stub("prismaquant_cb_ext", cuda_ext._EXT_SYMBOLS)
    _patch_load(monkeypatch, good)

    assert cuda_ext.get_ext() is good


def test_get_ext_is_still_memoised(monkeypatch, tmp_path, fresh_get_ext):
    """One build attempt per process, unchanged: the second call must not
    re-enter load()."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    good = _stub("prismaquant_cb_ext", cuda_ext._EXT_SYMBOLS)
    n = {"calls": 0}
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")

    def counting_load(*args, **kwargs):
        n["calls"] += 1
        return good

    monkeypatch.setattr(cpp_extension, "load", counting_load)
    assert cuda_ext.get_ext() is good
    assert cuda_ext.get_ext() is good
    assert n["calls"] == 1


def test_sys_modules_untouched():
    """Guard against this file leaking stubs into a shared pytest process the
    way tests/test_target_namespace_compat.py does (see
    .github/scripts/run_cpu_tests.sh)."""
    assert "prismaquant_cb_ext" not in sys.modules
