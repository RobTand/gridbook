"""CPU-only contract tests for the JIT build surface fixed on 2026-08-01.

Four properties of :mod:`gridbook.cuda_ext`, none of which needs a compiler:

* every module is compiled for the LIVE device and in its OWN build
  subdirectory (audit §3 P0.1 / P0.5);
* the fused-FP4 loader rejects a non-Blackwell device BEFORE any build work,
  instead of dying minutes deep in nvcc (§3 P0.2, ROADMAP "architecture
  precheck");
* the fused-FP8 module's ``cutlass_fork`` headers participate in its identity,
  so a header edit cannot serve a cached kernel built from the old one
  (§3 P0.3, ROADMAP K0.3);
* ``PRISMAQUANT_CUTLASS_INCLUDE`` is honoured by every CUTLASS loader, and a
  wrong value fails loudly rather than silently using vLLM's copy (§3 P0.4).

No extension is compiled: ``torch.utils.cpp_extension.load`` and device /
CUTLASS discovery are replaced with stand-ins.
"""
from __future__ import annotations

import os
import re
import types

import pytest

cuda_ext = pytest.importorskip(
    "gridbook.cuda_ext", reason="gridbook not importable")


_LOADER_STATE = (
    ("_ext", "_tried"),
    ("_ext_v2", "_tried_v2"),
    ("_bf16_grouped", "_bf16_grouped_tried"),
    ("_fused", "_fused_tried"),
    ("_fused_fp4", "_fused_fp4_tried"),
)


@pytest.fixture(autouse=True)
def fresh_loaders():
    """Give each case a process that has never attempted a load.

    Every loader memoizes its terminal state, including failures, so without
    this a case would observe whatever an earlier one (or an earlier test
    file) resolved. The exact prior values are restored afterwards.
    """
    saved = {name: getattr(cuda_ext, name)
             for pair in _LOADER_STATE for name in pair}
    for value_name, tried_name in _LOADER_STATE:
        setattr(cuda_ext, value_name, None)
        setattr(cuda_ext, tried_name, False)
    yield
    for name, value in saved.items():
        setattr(cuda_ext, name, value)


def _stub(symbols, *, path=None):
    mod = types.ModuleType("gridbook_test_stub")
    if path is not None:
        mod.__file__ = os.fspath(path)
    for symbol in symbols:
        setattr(mod, symbol, lambda *args, **kwargs: None)
    return mod


def _patch_load(monkeypatch, result=None, *, error=None):
    """Record ``load`` calls; optionally fail, or refuse to be called at all."""
    pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if error is not None:
            raise error
        if result is not None and "name" in kwargs:
            # torch imports the module under the requested
            # TORCH_EXTENSION_NAME; the identity check depends on that.
            result.__name__ = kwargs["name"]
        return result

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    return calls


def _patch_capability(monkeypatch, capability):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda *args, **kwargs: capability)


def _cutlass_tree(tmp_path, *, complete=True):
    """A minimal directory that looks like a CUTLASS ``include`` tree."""
    include = tmp_path / "cutlass-src" / "include"
    (include / "cutlass").mkdir(parents=True)
    if complete:
        (include / "cutlass" / "cutlass.h").write_text("// sentinel\n")
        (include / "cutlass" / "version.h").write_text("// sentinel\n")
    return include


# ---------------------------------------------------------------------------
# P0.1 / P0.5: live-device target, own build subdirectory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loader_name,subdir,module_name", [
    ("get_ext", "main", "prismaquant_cb_ext"),
    ("get_ext_v2", "v2", "prismaquant_cb_v2_ext"),
])
def test_hot_modules_target_the_live_device_in_their_own_subdirectory(
        monkeypatch, tmp_path, loader_name, subdir, module_name):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (8, 9))
    symbols = (cuda_ext._EXT_SYMBOLS if loader_name == "get_ext"
               else cuda_ext._V2_SYMBOLS)
    good = _stub(symbols)
    calls = _patch_load(monkeypatch, good)

    assert getattr(cuda_ext, loader_name)() is good
    kwargs = calls[0][1]
    assert kwargs["name"] == module_name
    assert kwargs["build_directory"] == str(tmp_path / subdir)
    assert "-gencode=arch=compute_89,code=sm_89" in kwargs["extra_cuda_cflags"]
    # Architecture-GENERIC sources: no arch-conditional target, which would
    # refuse to load on any other capability.
    assert not any("a," in flag or flag.endswith("a")
                   for flag in kwargs["extra_cuda_cflags"])


@pytest.mark.parametrize("loader_name", ["get_ext", "get_ext_v2"])
def test_hot_modules_fail_soft_when_the_device_cannot_be_queried(
        monkeypatch, capsys, tmp_path, loader_name):
    """No visible GPU means no defensible target, and the reason is printed.

    Compile-only environments (the image build) pin the capability instead;
    see the Dockerfile's ``load_for_build``.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    import torch

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("no CUDA GPUs are available")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    calls = _patch_load(monkeypatch, _stub(cuda_ext._EXT_SYMBOLS))

    assert getattr(cuda_ext, loader_name)() is None
    assert calls == []
    error = capsys.readouterr().err
    assert "no CUDA GPUs are available" in error
    assert "TORCH_CUDA_ARCH_LIST" in error


# ---------------------------------------------------------------------------
# P0.2: the fused-FP4 loader rejects non-Blackwell before building anything
# ---------------------------------------------------------------------------


def test_fused_fp4_rejects_non_blackwell_before_any_build_work(
        monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (8, 9))

    def refuse_discovery():
        raise AssertionError(
            "CUTLASS discovery ran before the capability precheck")

    monkeypatch.setattr(cuda_ext, "_find_cutlass_include", refuse_discovery)
    calls = _patch_load(
        monkeypatch, error=AssertionError("nvcc was invoked on Ada"))

    assert cuda_ext.get_fused_fp4_ext() is None
    assert calls == [], "the loader must not reach torch's JIT build"
    error = capsys.readouterr().err
    assert "requires compute capability 12.0 or 12.1, got 8.9" in error


def test_fused_fp4_and_fp8_prechecks_agree(monkeypatch, capsys, tmp_path):
    """Both fused loaders reject the same set, with parallel wording."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (9, 0))
    monkeypatch.setattr(cuda_ext, "_find_cutlass_include",
                        lambda: str(_cutlass_tree(tmp_path)))
    calls = _patch_load(monkeypatch, _stub(cuda_ext._FUSED_SYMBOLS))

    assert cuda_ext.get_fused_ext() is None
    assert cuda_ext.get_fused_fp4_ext() is None
    assert calls == []
    error = capsys.readouterr().err
    assert "fused FP8-CB prefill requires compute capability 12.0 or 12.1, " \
        "got 9.0" in error
    assert "fused NVFP4-CB prefill requires compute capability 12.0 or 12.1, " \
        "got 9.0" in error


# ---------------------------------------------------------------------------
# P0.3: the fused-FP8 module's fork headers key its identity
# ---------------------------------------------------------------------------


def _fake_torch():
    c_api = types.SimpleNamespace(
        _GLIBCXX_USE_CXX11_ABI=True,
        _PYBIND11_COMPILER_TYPE="gcc",
        _PYBIND11_STDLIB="libstdcpp",
        _PYBIND11_BUILD_ABI="cxxabi1011",
        _cuda_getCompiledVersion=lambda: 13000,
        _cuda_getRuntimeVersion=lambda: 13000,
        _cuda_getDriverVersion=lambda: 13000,
    )
    return types.SimpleNamespace(
        __version__="2.9.0+cu130",
        version=types.SimpleNamespace(cuda="13.0"),
        compiled_with_cxx11_abi=lambda: True,
        _C=c_api,
    )


def _fp8_identity_fixture(tmp_path):
    """A stand-in packaged tree holding the .cu and all three fork headers."""
    src = tmp_path / "csrc"
    (src / "cutlass_fork").mkdir(parents=True)
    for name in cuda_ext._FUSED_BUILD_INPUTS:
        (src / name).write_text(f"// {name} v1\n")
    cutlass = _cutlass_tree(tmp_path)
    util = tmp_path / "cutlass-src" / "tools" / "util" / "include"
    packed = util / "cutlass" / "util" / "packed_stride.hpp"
    packed.parent.mkdir(parents=True)
    packed.write_text("// packed v1\n")
    return src, cutlass, util


def test_fp8_identity_names_every_packaged_build_input():
    """The declared inputs are the .cu plus the headers it includes.

    Read from the packaged source rather than restated, so adding a
    ``cutlass_fork`` include to ``cb_fused_gemm.cu`` without declaring it here
    fails instead of quietly leaving that header out of the cache key.
    """
    path = os.path.join(cuda_ext.csrc_dir(), "cb_fused_gemm.cu")
    if not os.path.isfile(path):
        pytest.skip("cb_fused_gemm.cu not present in this install")
    with open(path, encoding="utf-8") as source:
        included = set(re.findall(
            r'#include\s+"(cutlass_fork/[^"]+)"', source.read()))
    declared = set(cuda_ext._FUSED_BUILD_INPUTS)
    assert "cb_fused_gemm.cu" in declared
    assert included <= declared, (
        f"cb_fused_gemm.cu includes {sorted(included - declared)}, which do "
        f"not key its build identity — a cached kernel built from an older "
        f"copy of those headers would be served")


@pytest.mark.parametrize("mutated", [
    name for name in cuda_ext._FUSED_BUILD_INPUTS
    if name.startswith("cutlass_fork/")
])
def test_fp8_identity_changes_when_a_fork_header_changes(tmp_path, mutated):
    src, cutlass, util = _fp8_identity_fixture(tmp_path)
    fake_cpp = types.SimpleNamespace(
        CUDA_HOME="/toolkit", get_cxx_compiler=lambda: "c++")

    def identity():
        digest, payload = cuda_ext._fused_build_identity_fp8(
            _fake_torch(), fake_cpp, src_dir=str(src),
            cutlass_include=str(cutlass), util_include=str(util),
            capability=(12, 1))
        return digest, payload

    before, payload = identity()
    assert payload["target"]["code"] == "sm_121a"
    assert payload["schema"] == cuda_ext._FUSED_ABI_SCHEMA
    assert set(payload["inputs"]) == set(cuda_ext._FUSED_BUILD_INPUTS)

    (src / mutated).write_text(f"// {mutated} v2 — one byte of schedule\n")
    after, _ = identity()
    assert after != before, (
        f"editing {mutated} left the module identity unchanged; the stale "
        f"cached kernel would still be loaded")

    (src / mutated).write_text(f"// {mutated} v1\n")
    restored, _ = identity()
    assert restored == before


def test_fp8_identity_keys_both_the_module_name_and_the_build_dir(
        monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (12, 1))
    monkeypatch.setattr(cuda_ext, "_find_cutlass_include",
                        lambda: str(_cutlass_tree(tmp_path)))
    good = _stub(cuda_ext._FUSED_SYMBOLS)
    calls = _patch_load(monkeypatch, good)

    assert cuda_ext.get_fused_ext() is good
    kwargs = calls[0][1]
    assert re.fullmatch(r"pq_cb_fused_[0-9a-f]{64}", kwargs["name"])
    identity = kwargs["name"].removeprefix("pq_cb_fused_")
    assert kwargs["build_directory"] == str(tmp_path / "fused" / identity)
    assert good.__gridbook_jit_identity__ == identity
    assert good.__gridbook_jit_abi_schema__ == cuda_ext._FUSED_ABI_SCHEMA
    assert "-gencode=arch=compute_121a,code=sm_121a" in \
        kwargs["extra_cuda_cflags"]


def test_fp8_contract_requires_the_grouped_family(monkeypatch, capsys,
                                                  tmp_path):
    """Dense-only is no longer a legitimate build, so it is not accepted.

    The identity above guarantees a loaded module was compiled from the
    current source; a missing grouped binding therefore means a broken build,
    and accepting it would silently downgrade routed prefill to the BF16
    bridge with no diagnostic.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (12, 1))
    monkeypatch.setattr(cuda_ext, "_find_cutlass_include",
                        lambda: str(_cutlass_tree(tmp_path)))
    _patch_load(monkeypatch, _stub(("cb_fused_prefill_mm_scaled",)))

    assert cuda_ext.get_fused_ext() is None
    error = capsys.readouterr().err
    assert "incompatible fused prefill" in error
    for name in ("cb_fused_moe_grouped", "cb_fused_moe_tile_m",
                 "cb_fused_moe_tile_sizes",
                 "cb_fused_moe_tile_sizes_for_kbits"):
        assert name in error


# ---------------------------------------------------------------------------
# P0.4: PRISMAQUANT_CUTLASS_INCLUDE, honoured everywhere and never guessed at
# ---------------------------------------------------------------------------


def test_cutlass_include_override_is_honoured(monkeypatch, tmp_path):
    include = _cutlass_tree(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_CUTLASS_INCLUDE", str(include))
    assert cuda_ext._find_cutlass_include() == str(include)


def test_cutlass_include_override_reaches_every_cutlass_loader(
        monkeypatch, tmp_path):
    """bf16-grouped, fused-FP8 and fused-FP4 all read the same override.

    Before 2026-08-01 only the fused-FP4 loader did, so the other two simply
    could not build in a venv without vLLM's bundled tree.
    """
    include = _cutlass_tree(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    monkeypatch.setenv("PRISMAQUANT_CUTLASS_INCLUDE", str(include))
    # Deliberately NOT stubbing _find_cutlass_include: the override is what is
    # under test, and it is read inside that function.
    _patch_capability(monkeypatch, (12, 1))

    for loader, symbols in (
        # cc 12.1 also compiles the sm12x-native lane, whose bindings the
        # loader requires strictly.
        (cuda_ext.get_bf16_grouped_ext,
         cuda_ext._BF16_GROUPED_SYMBOLS + cuda_ext._BF16_GROUPED_SM120_SYMBOLS),
        (cuda_ext.get_fused_ext, cuda_ext._FUSED_SYMBOLS),
        (cuda_ext.get_fused_fp4_ext,
         cuda_ext._FUSED_FP4_SYMBOL_FAMILIES[0][1]),
    ):
        for value_name, tried_name in _LOADER_STATE:
            setattr(cuda_ext, value_name, None)
            setattr(cuda_ext, tried_name, False)
        calls = _patch_load(monkeypatch, _stub(symbols))
        assert loader() is not None
        assert str(include) in calls[0][1]["extra_include_paths"]


def test_cutlass_include_override_fails_loudly_when_wrong(
        monkeypatch, tmp_path):
    bogus = tmp_path / "not-cutlass"
    bogus.mkdir()
    monkeypatch.setenv("PRISMAQUANT_CUTLASS_INCLUDE", str(bogus))
    with pytest.raises(FileNotFoundError) as exc_info:
        cuda_ext._find_cutlass_include()
    message = str(exc_info.value)
    assert str(bogus) in message
    assert "cutlass/cutlass.h" in message
    # A silent fall-through to vLLM's bundled copy would compile the operator
    # against different headers than the one the operator asked for.
    assert "does not fall back silently" in message


def test_wrong_cutlass_override_fails_the_grouped_bridge_closed(
        monkeypatch, capsys, tmp_path):
    bogus = tmp_path / "not-cutlass"
    bogus.mkdir()
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    monkeypatch.setenv("PRISMAQUANT_CUTLASS_INCLUDE", str(bogus))
    _patch_capability(monkeypatch, (12, 1))
    calls = _patch_load(monkeypatch, _stub(cuda_ext._BF16_GROUPED_SYMBOLS))

    assert cuda_ext.get_bf16_grouped_ext() is None
    assert calls == []
    error = capsys.readouterr().err
    assert "cutlass/cutlass.h" in error
    assert "serving will fail closed" in error
