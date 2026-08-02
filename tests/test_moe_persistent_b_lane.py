"""CPU contract gates for the OPT-IN persistent-B grouped MoE lane.

The lane itself (``gridbook/csrc/cb_moe_persistent_b.cu``, ROADMAP K1.1) is a
cc-12.x kernel and its numerics belong to the GPU suite.  What is testable
without a device — and what carries the whole opt-in promise — is the part that
decides WHETHER it runs and WHICH binary it would run:

* with ``PRISMAQUANT_CB_MOE_PERSISTENT_B`` unset nothing about the dispatch
  changes, nothing probes and nothing builds;
* with it set on a machine or a layer that cannot serve the lane, the model
  LOAD fails with an actionable sentence instead of quietly running the
  expand + grouped-bridge route — which would answer a different question than
  the operator asked;
* the selector is process-stable and rejects typos, so an intended A/B can
  never become an unlabelled baseline run;
* the loader rejects a non-Blackwell device BEFORE any build work, and its
  build digest keys both the module name and the build directory so a source
  edit can never be served from a stale cached kernel.

No extension is compiled and no device is touched: ``cpp_extension.load`` and
capability discovery are replaced with stand-ins, which is also why the file
passes unchanged on a host with a real GPU.
"""
from __future__ import annotations

import importlib.util
import os
import re
import types

import pytest

torch = pytest.importorskip("torch")
cuda_ext = pytest.importorskip(
    "gridbook.cuda_ext", reason="gridbook not importable")

from gridbook import moe_persistent_b_lane as lane  # noqa: E402
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402

_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B"
_CFG_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG"

# ``gridbook.moe`` imports vLLM at module scope; the dispatch section below is
# skipped rather than skipping this whole file, because sections A-C and E are
# what a vLLM-free CI host can still gate.
_HAS_VLLM = importlib.util.find_spec("vllm") is not None
requires_vllm = pytest.mark.skipif(
    not _HAS_VLLM,
    reason="vLLM not importable here; gridbook.moe imports it at module scope")


# ---------------------------------------------------------------------------
# Harness.  ``_LOADER_STATE`` / ``fresh_loaders`` / ``_stub`` / ``_patch_load``
# / ``_patch_capability`` / ``_fake_torch`` are the minimal copies of the
# helpers in tests/test_ext_build_identity.py, with this module's own loader
# pair appended.  They are duplicated rather than imported so that file stays
# free to change without silently changing what this one asserts.
# ---------------------------------------------------------------------------

_LOADER_STATE = (
    ("_ext", "_tried"),
    ("_ext_v2", "_tried_v2"),
    ("_bf16_grouped", "_bf16_grouped_tried"),
    ("_fused", "_fused_tried"),
    ("_fused_fp4", "_fused_fp4_tried"),
    ("_moe_persistent_b", "_moe_persistent_b_tried"),
)


@pytest.fixture(autouse=True)
def fresh_loaders():
    """Give each case a process that has never attempted a load."""
    saved = {name: getattr(cuda_ext, name)
             for pair in _LOADER_STATE for name in pair}
    for value_name, tried_name in _LOADER_STATE:
        setattr(cuda_ext, value_name, None)
        setattr(cuda_ext, tried_name, False)
    yield
    for name, value in saved.items():
        setattr(cuda_ext, name, value)


@pytest.fixture(autouse=True)
def _fresh_flag(monkeypatch):
    """Unset both selectors and clear the process-stable latch."""
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.delenv(_CFG_FLAG, raising=False)
    lane._reset_for_tests()
    yield
    lane._reset_for_tests()


def _stub(symbols, *, path=None):
    mod = types.ModuleType("gridbook_test_stub")
    if path is not None:
        mod.__file__ = os.fspath(path)
    for symbol in symbols:
        setattr(mod, symbol, lambda *args, **kwargs: None)
    return mod


def _patch_load(monkeypatch, result=None, *, error=None):
    """Record ``load`` calls; optionally fail, or refuse to be called at all."""
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
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda *args, **kwargs: capability)


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


def _complete_stub():
    return types.SimpleNamespace(
        cb_moe_persistent_b_prefill=lambda *a, **k: None,
        cb_moe_persistent_b_decode=lambda *a, **k: None,
        cb_moe_persistent_b_configs=lambda: ((128, 64), (64, 128), (128, 128)),
        cb_moe_persistent_b_tile_k=lambda: 64,
        cb_moe_persistent_b_is_moe_only=lambda: True,
    )


def _source_path():
    path = os.path.join(cuda_ext.csrc_dir(), "cb_moe_persistent_b.cu")
    if not os.path.isfile(path):
        pytest.skip("cb_moe_persistent_b.cu not present in this install")
    return path


# ---------------------------------------------------------------------------
# A. Selector contract
# ---------------------------------------------------------------------------


def test_flag_defaults_to_the_expand_plus_bridge_route(monkeypatch):
    assert lane.requested() is False
    lane._reset_for_tests()
    monkeypatch.setenv(_FLAG, "0")
    assert lane.requested() is False
    lane._reset_for_tests()
    monkeypatch.setenv(_FLAG, "")
    assert lane.requested() is False


def test_flag_enables_the_lane(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert lane.requested() is True


@pytest.mark.parametrize("value",
                         ["yes", "true", "on", "persistent", "128"])
def test_typos_are_refused_rather_than_silently_ignored(monkeypatch, value):
    """A misspelled opt-in must not resolve to the baseline schedule."""
    monkeypatch.setenv(_FLAG, value)
    with pytest.raises(ValueError, match=_FLAG):
        lane.requested()


def test_selector_cannot_change_mid_process(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert lane.requested() is True
    monkeypatch.setenv(_FLAG, "0")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        lane.requested()


def test_require_lane_fails_closed_without_the_extension(monkeypatch):
    """No persistent-B module at all is a fail-closed model load."""
    # ``require_lane`` imports cuda_ext lazily, so the STRING path is what has
    # to be patched — patching an already-imported binding would miss it.
    monkeypatch.setattr("gridbook.cuda_ext.get_moe_persistent_b_ext",
                        lambda: None)
    with pytest.raises(NativeKernelUnavailableError,
                       match="cb_moe_persistent_b.cu"):
        lane.require_lane("routed quality prefill")


def test_require_lane_fails_closed_without_the_bindings(monkeypatch):
    """A module without the entry points is refused, not fallen back."""
    stub = types.SimpleNamespace(cb_moe_persistent_b_tile_k=lambda: 64)
    monkeypatch.setattr("gridbook.cuda_ext.get_moe_persistent_b_ext",
                        lambda: stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("routed quality prefill")
    message = str(exc_info.value)
    assert f"{_FLAG}=1" in message
    assert "cb_moe_persistent_b_prefill" in message
    assert "cb_moe_persistent_b_decode" in message
    assert "cb_moe_persistent_b_configs" in message
    assert "compute capability 12.0/12.1" in message
    assert "does not substitute a different kernel" in message


def test_require_lane_names_the_device_that_cannot_serve_it(monkeypatch):
    """On a visible non-Blackwell device the message says which one it is."""
    stub = types.SimpleNamespace(cb_moe_persistent_b_tile_k=lambda: 64)
    monkeypatch.setattr("gridbook.cuda_ext.get_moe_persistent_b_ext",
                        lambda: stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_capability(monkeypatch, (8, 9))
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("routed quality prefill")
    assert "this device reports 8.9" in str(exc_info.value)


def test_require_lane_accepts_a_complete_module(monkeypatch):
    stub = _complete_stub()
    monkeypatch.setattr("gridbook.cuda_ext.get_moe_persistent_b_ext",
                        lambda: stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert lane.require_lane("routed quality prefill") is stub


def test_config_enumerates_the_compiled_tiles(monkeypatch):
    """Python may only ask for a (TM, TN) the module was actually built with."""
    configs = lane.config(_complete_stub())
    assert isinstance(configs, list) and configs
    for row in configs:
        assert isinstance(row, list)
        assert row and all(isinstance(value, int) for value in row)
    assert configs[0] == [128, 64]


# ---------------------------------------------------------------------------
# B. supports(): the load-time predicate, one readable sentence per rejection
# ---------------------------------------------------------------------------

_GOOD_LAYER = dict(is_fp4=True, is_v2=True, n_sub=2, k_bits=16,
                   type_size=73, hidden=2048, inter=1024)


def test_supports_accepts_a_plain_fp4_cb_v2_layer():
    assert lane.supports(**_GOOD_LAYER) is None
    # 73 == 4*16 + 9: the fixture is a real v2 row, not an arbitrary number.
    assert _GOOD_LAYER["type_size"] == 4 * _GOOD_LAYER["k_bits"] + 9


@pytest.mark.parametrize("override,mentions", [
    (dict(is_fp4=False), "FP4"),
    (dict(is_v2=False), "v2"),
    (dict(n_sub=1), "n_sub=1"),
    (dict(n_sub=4), "n_sub=4"),
    (dict(k_bits=0, type_size=9), "k=0"),
    (dict(k_bits=25, type_size=109), "k=25"),
    (dict(type_size=72), "type_size"),
    (dict(hidden=2000), "2000"),
    (dict(inter=1000), "1000"),
], ids=["fp8", "not-v2", "n_sub-1", "n_sub-4", "k-0", "k-25",
        "type_size", "hidden-unaligned", "inter-unaligned"])
def test_supports_rejects_with_a_readable_reason(override, mentions):
    """Every rejection is diagnosed at model load, never inside a request."""
    reason = lane.supports(**{**_GOOD_LAYER, **override})
    assert isinstance(reason, str) and reason.strip(), (
        f"{override} must be refused with a sentence, got {reason!r}")
    assert mentions in reason, (
        f"the reason for {override} does not name the offending fact "
        f"({mentions!r}): {reason!r}")


def test_supports_mirrors_the_kernels_own_shape_checks():
    """The k range and the 4*k+9 row size are the kernel's TORCH_CHECKs."""
    source = open(_source_path(), encoding="utf-8").read()
    assert "k_bits >= 1 && k_bits <= 24" in source
    assert "type_size == 4 * k_bits + 9" in source
    for k_bits in (1, 24):
        assert lane.supports(**{**_GOOD_LAYER, "k_bits": k_bits,
                                "type_size": 4 * k_bits + 9}) is None


# ---------------------------------------------------------------------------
# C. Loader identity: capability gate, digest, strict symbols, build inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability,buildable", [
    ((12, 0), True),
    ((12, 1), True),
    ((8, 9), False),
    ((9, 0), False),
    ((12, 2), False),
])
def test_buildable_is_exactly_the_two_blackwell_capabilities(capability,
                                                             buildable):
    assert cuda_ext.moe_persistent_b_buildable(capability) is buildable
    # torch hands back a tuple, but a list must resolve identically.
    assert cuda_ext.moe_persistent_b_buildable(list(capability)) is buildable


def test_loader_rejects_non_blackwell_before_any_build_work(
        monkeypatch, capsys, tmp_path):
    """One ``if``, not minutes deep in nvcc: no include work, no compile."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (8, 9))

    def refuse_source_probe(*_args, **_kwargs):
        raise AssertionError("source discovery ran before the arch precheck")

    monkeypatch.setattr(cuda_ext, "_require_csrc", refuse_source_probe)
    calls = _patch_load(
        monkeypatch, error=AssertionError("nvcc was invoked on Ada"))

    assert cuda_ext.get_moe_persistent_b_ext() is None
    assert calls == [], "the loader must not reach torch's JIT build"
    error = capsys.readouterr().err
    assert "compute capability 12.0/12.1" in error
    assert "got 8.9" in error


def test_identity_keys_both_the_module_name_and_the_build_dir(
        monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (12, 1))
    good = _stub(cuda_ext._MOE_PERSISTENT_B_SYMBOLS)
    calls = _patch_load(monkeypatch, good)

    assert cuda_ext.get_moe_persistent_b_ext() is good
    kwargs = calls[0][1]
    assert re.fullmatch(r"pq_cb_moe_persistent_b_[0-9a-f]{64}", kwargs["name"])
    identity = kwargs["name"].removeprefix("pq_cb_moe_persistent_b_")
    assert kwargs["build_directory"] == str(
        tmp_path / "moe_persistent_b" / identity)
    assert good.__gridbook_jit_identity__ == identity
    assert good.__gridbook_jit_abi_schema__ == \
        cuda_ext._MOE_PERSISTENT_B_ABI_SCHEMA
    # The arch-CONDITIONAL target: a plain sm_121 build of this schedule can
    # load and then abort at launch (the loader's own recorded reason).
    assert "-gencode=arch=compute_121a,code=sm_121a" in \
        kwargs["extra_cuda_cflags"]
    # Memoized: a second call neither rebuilds nor re-enters the JIT.
    assert cuda_ext.get_moe_persistent_b_ext() is good
    assert len(calls) == 1


def test_identity_is_memoized_after_a_failure(monkeypatch, tmp_path):
    """A failed load is terminal for the process, not retried per request."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (12, 1))
    calls = _patch_load(monkeypatch, error=RuntimeError("no nvcc here"))

    assert cuda_ext.get_moe_persistent_b_ext() is None
    assert cuda_ext.get_moe_persistent_b_ext() is None
    assert len(calls) == 1


@pytest.mark.parametrize("mutated", cuda_ext._MOE_PERSISTENT_B_BUILD_INPUTS)
def test_identity_moves_when_any_build_input_changes(tmp_path, mutated):
    """A source edit must not be servable from the cached kernel."""
    src = tmp_path / "csrc"
    src.mkdir()
    for name in cuda_ext._MOE_PERSISTENT_B_BUILD_INPUTS:
        (src / name).write_text(f"// {name} v1\n")
    fake_cpp = types.SimpleNamespace(
        CUDA_HOME="/toolkit", get_cxx_compiler=lambda: "c++")

    def identity():
        return cuda_ext._moe_persistent_b_build_identity(
            _fake_torch(), fake_cpp, src_dir=str(src), capability=(12, 1))

    before, payload = identity()
    assert payload["schema"] == cuda_ext._MOE_PERSISTENT_B_ABI_SCHEMA
    assert set(payload["inputs"]) == set(
        cuda_ext._MOE_PERSISTENT_B_BUILD_INPUTS)
    assert payload["target"]["code"] == "sm_121a"
    # This translation unit includes no CUTLASS at all; the payload records
    # that honestly as three absent sentinels rather than omitting the field.
    assert set(payload["cutlass_inputs"].values()) == {None}
    assert payload["bindings"] == [
        ["persistent-B grouped MoE",
         list(cuda_ext._MOE_PERSISTENT_B_SYMBOLS)]]

    (src / mutated).write_text(f"// {mutated} v2 — one byte of schedule\n")
    after, _ = identity()
    assert after != before, (
        f"editing {mutated} left the module identity unchanged; the stale "
        f"cached kernel would still be loaded")

    (src / mutated).write_text(f"// {mutated} v1\n")
    restored, _ = identity()
    assert restored == before


def test_identity_separates_the_two_supported_capabilities(tmp_path):
    """A cc-12.0 binary can never be served to a cc-12.1 process."""
    src = tmp_path / "csrc"
    src.mkdir()
    for name in cuda_ext._MOE_PERSISTENT_B_BUILD_INPUTS:
        (src / name).write_text(f"// {name} v1\n")
    fake_cpp = types.SimpleNamespace(
        CUDA_HOME="/toolkit", get_cxx_compiler=lambda: "c++")

    def identity(capability):
        return cuda_ext._moe_persistent_b_build_identity(
            _fake_torch(), fake_cpp, src_dir=str(src), capability=capability)

    assert identity((12, 0))[0] != identity((12, 1))[0]
    assert identity((12, 0))[1]["target"]["code"] == "sm_120a"


def test_loader_refuses_a_module_missing_a_strict_symbol(
        monkeypatch, capsys, tmp_path):
    """The digest proves the source; a missing binding is a broken build."""
    monkeypatch.setenv("PRISMAQUANT_CB_EXT_DIR", str(tmp_path))
    _patch_capability(monkeypatch, (12, 1))
    partial = _stub([name for name in cuda_ext._MOE_PERSISTENT_B_SYMBOLS
                     if name != "cb_moe_persistent_b_configs"])
    _patch_load(monkeypatch, partial)

    assert cuda_ext.get_moe_persistent_b_ext() is None
    error = capsys.readouterr().err
    assert "incompatible persistent-B grouped MoE extension" in error
    assert "cb_moe_persistent_b_configs" in error


def test_declared_build_inputs_cover_every_gridbook_include():
    """Declared inputs are read from the source, not restated.

    Adding a Gridbook include to ``cb_moe_persistent_b.cu`` without declaring
    it here would leave that header out of the cache key — the stale-kernel
    class the identity mechanism exists to prevent.
    """
    with open(_source_path(), encoding="utf-8") as source:
        included = set(re.findall(
            r'#include\s+"((?:cutlass_fork/|cb_)[^"]+)"', source.read()))
    declared = set(cuda_ext._MOE_PERSISTENT_B_BUILD_INPUTS)
    assert "cb_moe_persistent_b.cu" in declared
    assert included <= declared, (
        f"cb_moe_persistent_b.cu includes {sorted(included - declared)}, "
        f"which do not key its build identity")
    # The other direction: every declared input must be a packaged file, or
    # the loader's own _require_csrc gate fails the build for a name nobody
    # ships.
    missing = [name for name in declared
               if not os.path.isfile(os.path.join(cuda_ext.csrc_dir(), name))]
    assert not missing, f"declared build inputs are not packaged: {missing}"


def test_every_strict_symbol_is_exported_by_the_packaged_source():
    """The loader's contract cannot demand a binding the .cu never defines."""
    with open(_source_path(), encoding="utf-8") as source:
        exports = set(re.findall(
            r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', source.read()))
    required = set(cuda_ext._MOE_PERSISTENT_B_SYMBOLS)
    assert required <= exports, (
        f"the loader requires {sorted(required - exports)}, which "
        f"cb_moe_persistent_b.cu does not export; every build would be "
        f"refused as incompatible")


def test_require_ext_message_names_the_source_and_the_capability(monkeypatch):
    monkeypatch.setattr(cuda_ext, "_moe_persistent_b_tried", True)
    monkeypatch.setattr(cuda_ext, "_moe_persistent_b", None)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        cuda_ext.require_moe_persistent_b_ext("routed quality prefill")
    message = str(exc_info.value)
    assert "cb_moe_persistent_b.cu" in message
    assert "compute capability 12.0/12.1" in message
    assert "does not fall back to Triton" in message


# ---------------------------------------------------------------------------
# D. Dispatch gating (needs gridbook.moe, hence vLLM)
# ---------------------------------------------------------------------------


def _bare_method():
    """A method object with no ``__init__`` side effects; stays CPU-only."""
    from gridbook.moe import PrismaQuantCBMoEMethod

    return object.__new__(PrismaQuantCBMoEMethod)


@requires_vllm
def test_moe_dispatch_reads_this_exact_selector():
    """The gating in moe.py is this module's functions, not a second copy."""
    from gridbook import moe

    assert moe.persistent_b_requested is lane.requested
    assert moe.persistent_b_require_lane is lane.require_lane
    assert moe.persistent_b_supports is lane.supports


@requires_vllm
def test_cfg_defaults_to_the_kernels_own_choice():
    assert _bare_method()._persistent_b_cfg() == 0


@requires_vllm
def test_cfg_parses_the_requested_index(monkeypatch):
    monkeypatch.setenv(_CFG_FLAG, "3")
    assert _bare_method()._persistent_b_cfg() == 3


@requires_vllm
def test_cfg_refuses_a_non_integer(monkeypatch):
    monkeypatch.setenv(_CFG_FLAG, "fastest")
    with pytest.raises(ValueError, match="must be an integer"):
        _bare_method()._persistent_b_cfg()


@requires_vllm
def test_cfg_refuses_a_negative_index(monkeypatch):
    monkeypatch.setenv(_CFG_FLAG, "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _bare_method()._persistent_b_cfg()


@requires_vllm
def test_cfg_is_read_once_and_cached(monkeypatch):
    """The two GEMM stages of one forward cannot see different tile configs."""
    method = _bare_method()
    monkeypatch.setenv(_CFG_FLAG, "2")
    assert method._persistent_b_cfg() == 2
    monkeypatch.setenv(_CFG_FLAG, "5")
    assert method._persistent_b_cfg() == 2
    monkeypatch.delenv(_CFG_FLAG)
    assert method._persistent_b_cfg() == 2


@requires_vllm
@pytest.mark.parametrize("layer_attrs", [
    {},                              # flag off at load: attribute never set
    {"_cb_moe_persistent_b": None},  # flag off at load: attribute set to None
], ids=["absent", "none"])
def test_prefill_dispatch_is_unchanged_while_the_lane_is_off(monkeypatch,
                                                             layer_attrs):
    """With no lane object on the layer, neither opt-in helper is entered."""
    from gridbook.moe import PrismaQuantCBMoEMethod as Method

    taken = []
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_persistent_b",
        lambda self, *a, **k: taken.append("persistent-b"))
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_sm120",
        lambda self, *a, **k: taken.append("sm120"))

    layer = types.SimpleNamespace(**layer_attrs)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    # The default route then reads layer._cb_E, which this bare stand-in does
    # not carry: reaching that read is exactly the proof it was taken.
    with pytest.raises(AttributeError, match="_cb_E"):
        Method._apply_prefill_native_bf16(
            _bare_method(), layer, x, None, None, "silu")
    assert taken == []


@requires_vllm
def test_prefill_dispatch_takes_the_lane_when_it_is_resolved(monkeypatch):
    """A resolved lane object wins, and wins over the sm12x bridge lane."""
    from gridbook.moe import PrismaQuantCBMoEMethod as Method

    taken = []
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_persistent_b",
        lambda self, *a, **k: taken.append("persistent-b") or "pb")
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_sm120",
        lambda self, *a, **k: taken.append("sm120") or "sm120")

    layer = types.SimpleNamespace(_cb_moe_persistent_b=_complete_stub(),
                                  _cb_bf16_sm120=_complete_stub())
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert Method._apply_prefill_native_bf16(
        _bare_method(), layer, x, None, None, "silu") == "pb"
    assert taken == ["persistent-b"]


@requires_vllm
def test_prefill_still_requires_bf16_activations(monkeypatch):
    """The lane does not relax the payload contract the default route has."""
    from gridbook.moe import PrismaQuantCBMoEMethod as Method

    layer = types.SimpleNamespace(_cb_moe_persistent_b=_complete_stub())
    with pytest.raises(TypeError, match="BF16 activations"):
        Method._apply_prefill_native_bf16(
            _bare_method(), layer, torch.zeros(2, 4), None, None, "silu")


# ---------------------------------------------------------------------------
# E. The additive-block invariant
# ---------------------------------------------------------------------------


def test_persistent_b_loader_is_one_additive_block_at_the_end_of_cuda_ext():
    """The loader is appended, never interleaved.

    Everything this lane adds to ``cuda_ext`` lives between two marker
    comments at the very end of the file, so the module keeps a single
    contiguous, self-contained diff surface: nothing above the BEGIN marker
    mentions the lane, and nothing follows the END marker.
    """
    path = os.path.abspath(cuda_ext.__file__)
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    begin = [i for i, line in enumerate(lines)
             if "BEGIN ADDITIVE BLOCK — persistent-B grouped MoE loader" in
             line]
    end = [i for i, line in enumerate(lines)
           if "END ADDITIVE BLOCK — persistent-B grouped MoE loader" in line]
    assert len(begin) == 1, f"expected one BEGIN marker, found {len(begin)}"
    assert len(end) == 1, f"expected one END marker, found {len(end)}"
    assert begin[0] < end[0]

    above = "\n".join(lines[:begin[0]]).lower()
    assert "persistent_b" not in above
    assert "persistent-b" not in above

    trailer = lines[end[0] + 1:]
    for line in trailer:
        stripped = line.strip()
        assert not stripped or set(stripped) <= set("#= "), (
            f"{stripped!r} follows the persistent-B block; the block must be "
            f"the last thing in cuda_ext.py so concurrent edits above it "
            f"cannot conflict")

    block = "\n".join(lines[begin[0]:end[0]])
    for name in ("_MOE_PERSISTENT_B_BUILD_INPUTS", "_MOE_PERSISTENT_B_SYMBOLS",
                 "_MOE_PERSISTENT_B_ABI_SCHEMA",
                 "_MOE_PERSISTENT_B_CAPABILITIES",
                 "def moe_persistent_b_buildable",
                 "def get_moe_persistent_b_ext",
                 "def require_moe_persistent_b_ext",
                 "def _moe_persistent_b_build_identity",
                 "def _load_moe_persistent_b_ext_locked"):
        assert name in block, f"{name} is not inside the additive block"
