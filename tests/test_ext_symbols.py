"""CPU-only contract tests for :mod:`gridbook.cuda_ext` loaders.

No extension is compiled. ``torch.utils.cpp_extension.load`` and device/CUTLASS
discovery are replaced with stand-ins, while required symbol declarations are
also checked against the packaged ``m.def`` bindings.
"""
from __future__ import annotations

import concurrent.futures
import os
import re
import sys
import threading
import types

import pytest

cuda_ext = pytest.importorskip(
    "gridbook.cuda_ext", reason="gridbook not importable")


def _stub(name, symbols, *, path=None):
    mod = types.ModuleType(name)
    if path is not None:
        mod.__file__ = os.fspath(path)
    for symbol in symbols:
        setattr(mod, symbol, lambda *args, **kwargs: None)
    return mod


_LOADER_STATE = (
    ("_ext", "_tried"),
    ("_ptc", "_ptc_tried"),
    ("_fused", "_fused_tried"),
    ("_fused_fp4", "_fused_fp4_tried"),
)


@pytest.fixture(autouse=True)
def fresh_loaders():
    saved = {name: getattr(cuda_ext, name)
             for pair in _LOADER_STATE for name in pair}
    for value_name, tried_name in _LOADER_STATE:
        setattr(cuda_ext, value_name, None)
        setattr(cuda_ext, tried_name, False)
    yield
    for name, value in saved.items():
        setattr(cuda_ext, name, value)


def _patch_load(monkeypatch, result=None, *, error=None):
    pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if error is not None:
            raise error
        if result is not None and "name" in kwargs:
            # torch's real loader imports the module under the requested
            # TORCH_EXTENSION_NAME. Preserve that ABI behavior in the stub.
            result.__name__ = kwargs["name"]
        return result

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    return calls


def _prepare_cutlass(monkeypatch, tmp_path):
    cutlass = tmp_path / "cutlass" / "include"
    cutlass.mkdir(parents=True)
    monkeypatch.setattr(cuda_ext, "_find_cutlass_include",
                        lambda: str(cutlass))
    return cutlass


def _prepare_fp4_loader(monkeypatch, tmp_path):
    import torch

    cutlass = _prepare_cutlass(monkeypatch, tmp_path)
    monkeypatch.setenv("PRISMAQUANT_CUTLASS_INCLUDE", str(cutlass))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))


@pytest.mark.parametrize(
    "loader_name,value_name,tried_name,lock_name,symbols,prepare",
    [
        ("get_ext", "_ext", "_tried", "_ext_lock", "_EXT_SYMBOLS",
         "main"),
        ("get_persistent_ext", "_ptc", "_ptc_tried", "_ptc_lock",
         "_PTC_SYMBOLS", "ptc"),
        ("get_fused_fp4_ext", "_fused_fp4", "_fused_fp4_tried",
         "_fused_fp4_lock", "_FUSED_FP4_SYMBOL_FAMILIES", "fp4"),
        ("get_fused_ext", "_fused", "_fused_tried", "_fused_lock",
         "_FUSED_SYMBOLS", "fused"),
    ],
)
def test_cold_load_is_published_once_to_concurrent_callers(
        monkeypatch, tmp_path, loader_name, value_name, tried_name, lock_name,
        symbols, prepare):
    """No caller may observe the in-progress loader's initial ``None``.

    Model construction can ask several layers for the same extension at once.
    The first cold JIT is deliberately held in ``load`` while a second caller
    arrives; both must receive the one validated module and compilation must
    happen once.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    if prepare == "ptc":
        monkeypatch.setenv("PRISMAQUANT_ENABLE_PTC", "1")
        _prepare_cutlass(monkeypatch, tmp_path)
    elif prepare == "fp4":
        _prepare_fp4_loader(monkeypatch, tmp_path)
    elif prepare == "fused":
        _prepare_cutlass(monkeypatch, tmp_path)

    declared = getattr(cuda_ext, symbols)
    if symbols == "_FUSED_FP4_SYMBOL_FAMILIES":
        declared = declared[0][1]
    good = _stub(loader_name, declared)
    started = threading.Event()
    release = threading.Event()
    waiter_blocked = threading.Event()
    calls = []

    raw_lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if raw_lock.locked():
                waiter_blocked.set()
            raw_lock.acquire()
            return self

        def __exit__(self, *_args):
            raw_lock.release()
            return False

    monkeypatch.setattr(cuda_ext, lock_name, ObservedLock())

    pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")

    def blocking_load(*args, **kwargs):
        calls.append((args, kwargs))
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the cold JIT")
        good.__name__ = kwargs["name"]
        return good

    monkeypatch.setattr(cpp_extension, "load", blocking_load)
    loader = getattr(cuda_ext, loader_name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(loader)
        assert started.wait(timeout=10), "cold loader never reached JIT load"
        # Terminal state is published after load and symbol validation, not at
        # entry. This also makes the test distinguish the historical race even
        # if the second worker happens to start late on a loaded CI host.
        assert getattr(cuda_ext, tried_name) is False
        second = pool.submit(loader)
        assert waiter_blocked.wait(timeout=10), \
            "second caller did not wait for the cold loader"
        release.set()
        assert first.result(timeout=10) is good
        assert second.result(timeout=10) is good

    assert getattr(cuda_ext, value_name) is good
    assert getattr(cuda_ext, tried_name) is True
    assert len(calls) == 1


def test_concurrent_failed_load_is_memoized_once(monkeypatch, capsys,
                                                   tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    calls = []

    pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")

    def failing_load(*args, **kwargs):
        calls.append((args, kwargs))
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the cold JIT")
        raise RuntimeError("deliberate cold-build failure")

    monkeypatch.setattr(cpp_extension, "load", failing_load)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cuda_ext.get_ext)
        assert started.wait(timeout=10), "cold loader never reached JIT load"
        assert cuda_ext._tried is False
        second = pool.submit(cuda_ext.get_ext)
        release.set()
        assert first.result(timeout=10) is None
        assert second.result(timeout=10) is None

    assert cuda_ext._tried is True
    assert len(calls) == 1
    assert "deliberate cold-build failure" in capsys.readouterr().err


@pytest.mark.parametrize(
    "loader_name,value_name,tried_name,lock_name",
    [
        ("get_ext", "_ext", "_tried", "_ext_lock"),
        ("get_persistent_ext", "_ptc", "_ptc_tried", "_ptc_lock"),
        ("get_fused_fp4_ext", "_fused_fp4", "_fused_fp4_tried",
         "_fused_fp4_lock"),
        ("get_fused_ext", "_fused", "_fused_tried", "_fused_lock"),
    ],
)
def test_completed_loader_fast_path_does_not_take_mutex(
        monkeypatch, loader_name, value_name, tried_name, lock_name):
    class ExplodingLock:
        def __enter__(self):
            raise AssertionError("memoized loader acquired the cold-path lock")

        def __exit__(self, *_args):
            return False

    sentinel = object()
    setattr(cuda_ext, value_name, sentinel)
    setattr(cuda_ext, tried_name, True)
    monkeypatch.setattr(cuda_ext, lock_name, ExplodingLock())

    assert getattr(cuda_ext, loader_name)() is sentinel


# ---------------------------------------------------------------------------
# Symbol validation and diagnostics
# ---------------------------------------------------------------------------


def test_complete_module_is_returned_unchanged(tmp_path):
    mod = _stub("main", cuda_ext._EXT_SYMBOLS)
    assert cuda_ext._require_symbols(
        mod, cuda_ext._EXT_SYMBOLS, build_dir=str(tmp_path),
        source="cb_gemv.cu") is mod


def test_extra_symbols_do_not_matter(tmp_path):
    mod = _stub("main", (*cuda_ext._EXT_SYMBOLS, "future_binding"))
    assert cuda_ext._require_symbols(
        mod, cuda_ext._EXT_SYMBOLS, build_dir=str(tmp_path),
        source="cb_gemv.cu") is mod


def test_require_symbols_accepts_a_one_shot_iterable(tmp_path):
    names = (name for name in ("first", "second"))
    mod = _stub("generated", ("first",))
    with pytest.raises(cuda_ext.StaleExtensionError) as exc_info:
        cuda_ext._require_symbols(
            mod, names, build_dir=str(tmp_path), source="generated.cu")
    message = str(exc_info.value)
    assert "second" in message
    assert "['first', 'second']" in message


def test_missing_symbols_report_module_and_cache_diagnostics(tmp_path):
    module_path = tmp_path / "unexpected" / "main.so"
    mod = _stub("main", ("fp8_act_qdq",), path=module_path)
    with pytest.raises(cuda_ext.StaleExtensionError) as exc_info:
        cuda_ext._require_symbols(
            mod, cuda_ext._EXT_SYMBOLS, build_dir=str(tmp_path),
            source="cb_gemv.cu")
    message = str(exc_info.value)
    for symbol in cuda_ext._EXT_SYMBOLS[1:]:
        assert symbol in message
    assert str(module_path) in message
    assert str(tmp_path) in message
    assert "mode" in message and "owner uid:gid" in message
    assert "lock/build time" in message
    assert "PRISMAQUANT_CB_EXT_DIR" in message


def test_cache_diagnostics_handles_a_missing_directory(tmp_path):
    missing = tmp_path / "not-created"
    diagnostic = cuda_ext._cache_diagnostics(str(missing))
    assert str(missing) in diagnostic
    assert "cannot be stat'ed" in diagnostic
    assert "FileNotFoundError" in diagnostic


def test_symbol_family_accepts_either_complete_family_and_partial_other(
        tmp_path):
    families = (("alpha", ("a", "b")), ("gamma", ("g",)))
    gamma = _stub("gamma", ("a", "g"))
    alpha = _stub("alpha", ("a", "b"))
    for mod in (gamma, alpha):
        assert cuda_ext._require_any_symbol_family(
            mod, families, build_dir=str(tmp_path), source="fused.cu") is mod


def test_symbol_family_rejects_module_with_no_complete_family(tmp_path):
    families = (("alpha", ("a", "b")), ("gamma", ("g",)))
    mod = _stub("partial", ("a", "unrelated"))
    with pytest.raises(cuda_ext.StaleExtensionError) as exc_info:
        cuda_ext._require_any_symbol_family(
            mod, families, build_dir=str(tmp_path), source="fused.cu")
    message = str(exc_info.value)
    assert "no usable symbol family" in message
    assert "alpha" in message and "gamma" in message
    assert "b" in message and "g" in message


@pytest.mark.parametrize("families", [(), (("empty", ()),)])
def test_symbol_family_rejects_an_invalid_contract(tmp_path, families):
    with pytest.raises(ValueError, match="non-empty"):
        cuda_ext._require_any_symbol_family(
            _stub("mod", ()), families, build_dir=str(tmp_path),
            source="fused.cu")


def test_error_classes_are_disjoint():
    assert issubclass(cuda_ext.StaleExtensionError, RuntimeError)
    assert not issubclass(cuda_ext.StaleExtensionError,
                          cuda_ext.IncompleteInstallError)
    assert not issubclass(cuda_ext.IncompleteInstallError,
                          cuda_ext.StaleExtensionError)


# ---------------------------------------------------------------------------
# Declared contracts agree with the packaged extensions and call-site policy
# ---------------------------------------------------------------------------


_M_DEF = re.compile(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def _exports(cu_name):
    path = os.path.join(cuda_ext.csrc_dir(), cu_name)
    if not os.path.isfile(path):
        pytest.skip(f"{cu_name} not present in this install")
    with open(path, encoding="utf-8") as source:
        return set(_M_DEF.findall(source.read()))


@pytest.mark.parametrize("required,cu_name", [
    (cuda_ext._EXT_SYMBOLS, "cb_gemv.cu"),
    (cuda_ext._PTC_SYMBOLS, "cb_persistent_tc.cu"),
    (cuda_ext._FUSED_SYMBOLS, "cb_fused_gemm.cu"),
])
def test_strict_symbols_exist_in_packaged_source(required, cu_name):
    assert not (set(required) - _exports(cu_name))


def test_fp4_symbol_families_exist_in_packaged_source():
    exported = _exports("cb_fused_fp4_gemm.cu")
    for _label, required in cuda_ext._FUSED_FP4_SYMBOL_FAMILIES:
        assert not (set(required) - exported)


def test_main_optional_bindings_stay_optional():
    for name in ("cb_expand_fp8_into", "l2_pin_region", "l2_reset_window",
                 "l2_unpin", "l2_persisting_max_bytes",
                 "l2_max_window_bytes"):
        assert name not in cuda_ext._EXT_SYMBOLS


def test_main_contract_rejects_pre_fp4_qdq_revision(monkeypatch, capsys,
                                                     tmp_path):
    """The activation binding arrived after the original decode extension.

    A cached module from that earlier revision must be diagnosed as stale
    instead of being accepted and silently dropping the single-launch QDQ.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    old_symbols = tuple(name for name in cuda_ext._EXT_SYMBOLS
                        if name != "fp4_act_qdq")
    stale = _stub("main", old_symbols,
                  path=tmp_path / "prismaquant_cb_ext.so")
    _patch_load(monkeypatch, stale)

    assert cuda_ext.get_ext() is None
    error = capsys.readouterr().err
    assert "incompatible CUDA decode-GEMV" in error
    assert "fp4_act_qdq" in error


def test_fp8_grouped_bindings_stay_optional_after_dense_prerequisite():
    for name in ("cb_fused_moe_grouped", "cb_fused_moe_tile_m",
                 "cb_fused_moe_tile_sizes",
                 "cb_fused_moe_tile_sizes_for_kbits"):
        assert name not in cuda_ext._FUSED_SYMBOLS


# ---------------------------------------------------------------------------
# Main decode extension
# ---------------------------------------------------------------------------


def test_get_ext_accepts_complete_module(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    good = _stub("main", cuda_ext._EXT_SYMBOLS,
                 path=tmp_path / "prismaquant_cb_ext.so")
    calls = _patch_load(monkeypatch, good)
    assert cuda_ext.get_ext() is good
    assert calls[0][1]["build_directory"] == str(tmp_path)


def test_get_ext_refuses_incompatible_module(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    stale = _stub("main", cuda_ext._EXT_SYMBOLS[:-1],
                  path=tmp_path / "prismaquant_cb_ext.so")
    _patch_load(monkeypatch, stale)
    assert cuda_ext.get_ext() is None
    error = capsys.readouterr().err
    assert "incompatible CUDA decode-GEMV" in error
    assert "cb_moe_combine" in error
    assert "prismaquant_cb_ext.so" in error


def test_get_ext_is_memoized(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    good = _stub("main", cuda_ext._EXT_SYMBOLS)
    calls = _patch_load(monkeypatch, good)
    assert cuda_ext.get_ext() is good
    assert cuda_ext.get_ext() is good
    assert len(calls) == 1


def test_get_ext_reports_build_exception_separately(monkeypatch, capsys,
                                                     tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_load(monkeypatch, error=PermissionError("cannot create lock"))
    assert cuda_ext.get_ext() is None
    error = capsys.readouterr().err
    assert "could not be built (PermissionError: cannot create lock)" in error
    assert "incompatible" not in error


def test_get_ext_reports_incomplete_install_separately(monkeypatch, capsys):
    _patch_load(monkeypatch, _stub("unused", ()))
    monkeypatch.setattr(
        cuda_ext, "_require_csrc",
        lambda *_names: (_ for _ in ()).throw(
            cuda_ext.IncompleteInstallError("missing cb_gemv.cu")))
    assert cuda_ext.get_ext() is None
    error = capsys.readouterr().err
    assert "broken gridbook install" in error
    assert "missing cb_gemv.cu" in error


# ---------------------------------------------------------------------------
# Persistent tensor-core extension
# ---------------------------------------------------------------------------


def test_persistent_loader_is_opt_in_and_does_not_build(monkeypatch):
    calls = _patch_load(monkeypatch, _stub("ptc", cuda_ext._PTC_SYMBOLS))
    monkeypatch.delenv("PRISMAQUANT_ENABLE_PTC", raising=False)
    assert cuda_ext.get_persistent_ext() is None
    assert not calls


def test_persistent_loader_accepts_complete_module(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_ENABLE_PTC", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    good = _stub("ptc", cuda_ext._PTC_SYMBOLS)
    calls = _patch_load(monkeypatch, good)
    assert cuda_ext.get_persistent_ext() is good
    assert calls[0][1]["build_directory"] == str(tmp_path / "ptc")


def test_persistent_loader_refuses_missing_binding(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_ENABLE_PTC", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    _patch_load(monkeypatch, _stub("ptc", ()))
    with pytest.warns(UserWarning, match="incompatible persistent-TC") as seen:
        assert cuda_ext.get_persistent_ext() is None
    assert "cb_prefill_persistent_tc" in str(seen[0].message)


def test_persistent_loader_reports_build_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_ENABLE_PTC", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    _patch_load(monkeypatch, error=RuntimeError("nvcc failed"))
    with pytest.warns(UserWarning, match="persistent-TC ext unavailable"):
        assert cuda_ext.get_persistent_ext() is None


# ---------------------------------------------------------------------------
# Fused FP4 extension: dense and grouped families are independent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbols", [
    ("cb_fused_fp4_prefill_mm_scaled",),
    ("cb_fused_fp4_moe_grouped",),
    ("cb_fused_fp4_prefill_mm_scaled", "cb_fused_fp4_moe_grouped"),
])
def test_fused_fp4_accepts_each_useful_partial_module(
        monkeypatch, tmp_path, symbols):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_fp4_loader(monkeypatch, tmp_path)
    good = _stub("fused_fp4", symbols)
    calls = _patch_load(monkeypatch, good)
    assert cuda_ext.get_fused_fp4_ext() is good
    kwargs = calls[0][1]
    assert re.fullmatch(r"pq_cb_fused_fp4_[0-9a-f]{64}", kwargs["name"])
    identity = kwargs["name"].removeprefix("pq_cb_fused_fp4_")
    assert kwargs["build_directory"] == str(
        tmp_path / "fused_fp4" / identity)
    assert good.__gridbook_jit_identity__ == identity
    assert good.__gridbook_jit_abi_schema__ == \
        cuda_ext._FUSED_FP4_ABI_SCHEMA
    assert "-gencode=arch=compute_121a,code=sm_121a" in \
        kwargs["extra_cuda_cflags"]


def _identity_fixture(tmp_path):
    src = tmp_path / "csrc"
    header = src / "cutlass_fork" / "sm120_cb_fused_fp4_mma.hpp"
    header.parent.mkdir(parents=True)
    (src / "cb_fused_fp4_gemm.cu").write_text("source-v1")
    header.write_text("header-v1")

    cutlass = tmp_path / "cutlass" / "include"
    (cutlass / "cutlass").mkdir(parents=True)
    (cutlass / "cutlass" / "cutlass.h").write_text("cutlass-v1")
    (cutlass / "cutlass" / "version.h").write_text("version-v1")
    util = tmp_path / "cutlass" / "tools" / "util" / "include"
    packed = util / "cutlass" / "util" / "packed_stride.hpp"
    packed.parent.mkdir(parents=True)
    packed.write_text("packed-v1")
    return src, header, cutlass, util


def _fake_torch_identity():
    c_api = types.SimpleNamespace(
        _GLIBCXX_USE_CXX11_ABI=True,
        _PYBIND11_COMPILER_TYPE="gcc",
        _PYBIND11_STDLIB="libstdcpp",
        _PYBIND11_BUILD_ABI="cxxabi1011",
        _cuda_getCompiledVersion=lambda: 12080,
        _cuda_getRuntimeVersion=lambda: 12080,
        _cuda_getDriverVersion=lambda: 12090,
    )
    return types.SimpleNamespace(
        __version__="2.9.0+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
        compiled_with_cxx11_abi=lambda: True,
        _C=c_api,
    )


def test_fused_fp4_identity_covers_sources_arch_and_toolchain(
        monkeypatch, tmp_path):
    src, header, cutlass, util = _identity_fixture(tmp_path)
    fake_torch = _fake_torch_identity()
    fake_cpp = types.SimpleNamespace(
        CUDA_HOME="/toolkit", get_cxx_compiler=lambda: "c++")
    compiler_revision = {"value": "v1"}
    monkeypatch.setattr(
        cuda_ext, "_compiler_identity",
        lambda command: {"command": command,
                         "revision": compiler_revision["value"]})

    def identity(capability=(12, 1)):
        return cuda_ext._fused_fp4_build_identity(
            fake_torch, fake_cpp, src_dir=str(src),
            cutlass_include=str(cutlass), util_include=str(util),
            capability=capability)

    initial, payload = identity()
    assert payload["target"]["code"] == "sm_121a"
    assert payload["torch"]["version"] == "2.9.0+cu128"
    assert payload["torch"]["cuda"] == "12.8"
    assert payload["torch"]["glibcxx_cxx11_abi"] is True
    assert payload["cuda"]["runtime_version"] == 12080
    assert payload["cuda"]["nvcc"]["command"] == "/toolkit/bin/nvcc"
    assert payload["host_compiler"]["command"] == "c++"

    (src / "cb_fused_fp4_gemm.cu").write_text("source-v2")
    source_changed, _ = identity()
    assert source_changed != initial
    (src / "cb_fused_fp4_gemm.cu").write_text("source-v1")
    header.write_text("header-v2")
    header_changed, _ = identity()
    assert header_changed != initial
    header.write_text("header-v1")
    arch_changed, _ = identity((12, 0))
    assert arch_changed != initial

    fake_torch.__version__ = "2.10.0+cu128"
    torch_changed, _ = identity()
    assert torch_changed != initial
    fake_torch.__version__ = "2.9.0+cu128"
    fake_torch._C._GLIBCXX_USE_CXX11_ABI = False
    abi_changed, _ = identity()
    assert abi_changed != initial
    fake_torch._C._GLIBCXX_USE_CXX11_ABI = True
    fake_torch.version.cuda = "13.0"
    cuda_changed, _ = identity()
    assert cuda_changed != initial
    fake_torch.version.cuda = "12.8"
    compiler_revision["value"] = "v2"
    compiler_changed, _ = identity()
    assert compiler_changed != initial


def test_fused_fp4_rejects_wrong_identity_module(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_fp4_loader(monkeypatch, tmp_path)
    wrong = _stub("pq_cb_fused_fp4_wrong",
                  ("cb_fused_fp4_prefill_mm_scaled",))
    pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    monkeypatch.setattr(cpp_extension, "load", lambda *args, **kwargs: wrong)

    assert cuda_ext.get_fused_fp4_ext() is None
    error = capsys.readouterr().err
    assert "JIT ABI identity mismatch" in error
    assert "pq_cb_fused_fp4_wrong" in error


def test_fused_fp4_refuses_module_with_only_optional_symbols(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_fp4_loader(monkeypatch, tmp_path)
    useless = _stub("fused_fp4", ("cb_fused_fp4_moe_tile_sizes",))
    _patch_load(monkeypatch, useless)
    assert cuda_ext.get_fused_fp4_ext() is None
    error = capsys.readouterr().err
    assert "incompatible fused fp4" in error
    assert "dense prefill" in error and "grouped MoE prefill" in error


def test_fused_fp4_reports_build_exception(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_fp4_loader(monkeypatch, tmp_path)
    _patch_load(monkeypatch, error=RuntimeError("CUTLASS failed"))
    assert cuda_ext.get_fused_fp4_ext() is None
    error = capsys.readouterr().err
    assert "unavailable (RuntimeError: CUTLASS failed)" in error
    assert "incompatible" not in error


# ---------------------------------------------------------------------------
# Fused FP8 extension: scaled dense binding is the grouped prerequisite
# ---------------------------------------------------------------------------


def test_fused_fp8_accepts_dense_only_module(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    dense = _stub("fused", cuda_ext._FUSED_SYMBOLS)
    calls = _patch_load(monkeypatch, dense)
    assert cuda_ext.get_fused_ext() is dense
    assert calls[0][1]["build_directory"] == str(tmp_path / "fused")


def test_fused_fp8_refuses_grouped_without_dense_prerequisite(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    grouped = _stub(
        "fused", ("cb_fused_moe_grouped", "cb_fused_moe_tile_m"))
    _patch_load(monkeypatch, grouped)
    assert cuda_ext.get_fused_ext() is None
    error = capsys.readouterr().err
    assert "incompatible fused prefill" in error
    assert "cb_fused_prefill_mm_scaled" in error


def test_fused_fp8_reports_build_exception(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _prepare_cutlass(monkeypatch, tmp_path)
    _patch_load(monkeypatch, error=RuntimeError("CUTLASS failed"))
    assert cuda_ext.get_fused_ext() is None
    error = capsys.readouterr().err
    assert "fused prefill extension unavailable" in error
    assert "CUTLASS failed" in error


def test_no_fake_extension_modules_leak_into_sys_modules():
    for name in ("main", "ptc", "fused", "fused_fp4"):
        assert name not in sys.modules
