"""CPU contract gates for the OPT-IN FP4-CB v2 fused mid-M lane.

The lane itself is a CUTLASS kernel that decodes packed CB rows inside the
producer/consumer stage instead of materializing the ``[N, K]`` BF16 tile in
HBM first (2026-08-01 performance audit §3 P2a). It is CONTRACT-PRESERVING —
the decoded values are bit-identical to ``cb_expand_v2`` and the activations
are the same BF16 group-16 QDQ output the shipping bridge consumes, so the
only thing that moves is the FP32 GEMM reduction order — and it nevertheless
ships OPT-IN behind ``PRISMAQUANT_CB_FP4_FUSED_MIDM=1`` until the served
NATIVE-PARITY gate has been run against it.

None of that needs a device. What is testable here is everything that decides
WHETHER the kernel runs, and that carries the whole opt-in promise:

* with the flag unset nothing about the dispatch changes and nothing probes or
  builds — the lane is inert at import;
* with it set on a machine that cannot serve the lane, the model LOAD fails
  with an actionable message rather than quietly running the expand + BF16
  bridge, which would answer a different question than the operator asked;
* the selector is process-stable and rejects typos, so an intended A/B can
  never become an unlabelled baseline run;
* eligibility is decided from load-time metadata alone and every rejection
  falls through to the shipping route instead of raising — above all the HARD
  mid-M window and the "exactly ONE zero-based product dictionary" rule that
  keeps a fused module with per-role codebooks off this kernel;
* the module's JIT identity covers its fork header and the shared grouping
  glue, so a header edit can never serve a kernel cached from the old bytes.

Nothing here compiles an extension: the build identity is computed against a
stand-in source tree, and every extension is a ``types.SimpleNamespace`` stub.
"""
from __future__ import annotations

import importlib.util
import os
import re
import types

import pytest

from conftest import gridbook_include_closure

torch = pytest.importorskip("torch")

from gridbook import cuda_ext  # noqa: E402
from gridbook import fp4v2_fused_midm_lane as lane  # noqa: E402
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402
# The FP8/BF16 identity gates already model a packaged source tree and a torch
# whose ABI knobs are answerable without a build; reuse them rather than grow a
# second, silently diverging description of the same environment.
from test_ext_build_identity import _cutlass_tree, _fake_torch  # noqa: E402

_FLAG = "PRISMAQUANT_CB_FP4_FUSED_MIDM"

# Four rungs spanning the residency ladder the kernel documents: k12 stages the
# whole codebook, k24 stages sub0 only. ``cb_elems`` differs at each, so a
# helper that got the size wrong could not pass all four.
_RUNGS = (12, 16, 20, 24)

# Every rung the shipped module compiles a kernel for.
_COMPILED_KBITS = tuple(range(12, 25))


@pytest.fixture(autouse=True)
def _fresh_flag(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    lane._reset_for_tests()
    yield
    lane._reset_for_tests()


# ---------------------------------------------------------------------------
# The selector: default-off, typo-intolerant, process-stable
# ---------------------------------------------------------------------------


def test_flag_defaults_to_the_expand_plus_bridge_route(monkeypatch):
    # The name is spelled out in this file so a rename cannot quietly turn
    # every case below into a test of an environment variable nothing reads.
    assert lane._FLAG == _FLAG
    assert lane.requested() is False
    monkeypatch.setenv(_FLAG, "0")
    lane._reset_for_tests()
    assert lane.requested() is False


def test_flag_enables_the_lane(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert lane.requested() is True


@pytest.mark.parametrize("value", ["yes", "true", "128", "midm"])
def test_typos_are_refused_rather_than_silently_ignored(monkeypatch, value):
    monkeypatch.setenv(_FLAG, value)
    with pytest.raises(ValueError, match=_FLAG):
        lane.requested()


def test_selector_cannot_change_mid_process(monkeypatch):
    """Two reduction orders inside one run would make an A/B unreadable."""
    monkeypatch.setenv(_FLAG, "1")
    assert lane.requested() is True
    monkeypatch.setenv(_FLAG, "0")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        lane.requested()


def test_the_lane_probes_nothing_while_the_flag_is_unset(monkeypatch):
    """Import and flag read must not reach the loader, let alone nvcc.

    The opt-in promise is that with the flag off the dispatch is byte-for-byte
    what it was; a probe at import would already have cost a build attempt on
    every host that merely has Gridbook installed.
    """
    def refuse():
        raise AssertionError(
            "the fused FP4-v2 loader ran without the lane being requested")

    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext", refuse)
    # A genuine top-level execution, but into a throwaway module object: a
    # reload would publish a second copy of the process-stable latch and let
    # this case leak a dispatch decision into the rest of the session.
    spec = importlib.util.find_spec("gridbook.fp4v2_fused_midm_lane")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    assert fresh.requested() is False
    assert lane.requested() is False


# ---------------------------------------------------------------------------
# require_lane: an explicit lane selection never resolves to another kernel
# ---------------------------------------------------------------------------


def _ext_stub(*, max_m: int = 128, kbits=_COMPILED_KBITS, omit=(),
              prepare=None):
    """A stand-in for the loaded fused FP4-v2 module.

    Carries the loader's FULL strict tuple rather than the three entry points
    this module dereferences: ``require_lane`` checks against
    ``cuda_ext._FUSED_FP4V2_SYMBOLS`` since 2026-08-02, precisely so a local
    list cannot drift below what the loader enforces.

    ``cb_fused_fp4v2_kbits`` deliberately returns a LIST: the accessor's job
    includes normalizing whatever pybind hands back into a tuple of ints.
    """
    symbols = {name: (lambda *a, **k: None)
               for name in cuda_ext._FUSED_FP4V2_SYMBOLS}
    symbols["cb_fused_fp4v2_max_m"] = lambda: max_m
    symbols["cb_fused_fp4v2_kbits"] = lambda: list(kbits)
    if prepare is not None:
        symbols["cb_fused_fp4v2_prepare"] = prepare
    for name in omit:
        del symbols[name]
    return types.SimpleNamespace(**symbols)


def test_require_lane_fails_closed_without_the_extension(monkeypatch):
    """No fused FP4-v2 module at all: name the source, not a fallback."""
    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext", lambda: None)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("FP4 quality dense prefill")
    message = str(exc_info.value)
    assert "cb_fused_fp4v2_gemm.cu" in message
    assert f"{_FLAG}=1" in message
    assert "compute capability 12.0/12.1" in message
    assert "does not substitute a different kernel" in message


@pytest.mark.parametrize("missing", cuda_ext._FUSED_FP4V2_SYMBOLS)
def test_require_lane_fails_closed_on_an_incomplete_module(monkeypatch,
                                                           missing):
    """A partial module is a broken build, not an older one.

    The loader keys both the module name and the build directory on the source
    identity, so anything that imports at all was compiled from exactly these
    sources; a missing binding therefore cannot mean "a previous release".

    Parametrized over the LOADER's tuple, not a local restatement: the lane's
    private list used to be a strict subset, so a module missing a binding the
    loader required could still pass this gate.
    """
    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext",
                        lambda: _ext_stub(omit=(missing,)))
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("FP4 quality dense prefill")
    assert missing in str(exc_info.value)


def test_require_lane_opts_the_smem_in_at_load_not_first_launch(monkeypatch):
    """``cb_fused_fp4v2_prepare`` must be called by the LOAD-time resolution.

    Every compiled class exceeds the 48 KiB static limit, so CUTLASS's
    ``initialize()`` performs a ``cudaFuncSetAttribute`` — not stream-ordered
    work — from inside ``run_fused``, i.e. inside a forward, possibly a
    CUDA-graph capture. Attesting here is what moves it.
    """
    calls = []
    stub = _ext_stub(prepare=lambda: calls.append("prepare"))
    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext", lambda: stub)

    assert lane.require_lane("FP4 quality dense prefill") is stub
    assert calls == ["prepare"]


def test_require_lane_fails_closed_when_prepare_rejects_the_device(monkeypatch):
    """The device gate is the kernel's; a failure is normalized, never
    deferred to the first prefill."""
    def prepare():
        raise RuntimeError("needs 51200 B of opt-in shared memory")

    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext",
                        lambda: _ext_stub(prepare=prepare))
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("FP4 quality dense prefill")
    message = str(exc_info.value)
    assert "load-time device attestation failed" in message
    assert "51200" in message
    assert "does not defer this failure to first prefill" in message


def test_require_lane_accepts_a_complete_module(monkeypatch):
    stub = _ext_stub()
    monkeypatch.setattr(cuda_ext, "get_fused_fp4v2_ext", lambda: stub)

    assert lane.require_lane("FP4 quality dense prefill") is stub
    # The KERNEL is authoritative about the ceiling; the python constant only
    # quotes it, so the two must agree or the docs describe another kernel.
    assert lane.max_m(stub) == 128
    assert lane.MID_M_MAX == 128
    rungs = lane.kbits(stub)
    assert isinstance(rungs, tuple)
    assert all(type(k) is int for k in rungs)


# ---------------------------------------------------------------------------
# eligible(): every rejection falls through, none of them raises
# ---------------------------------------------------------------------------


def _layer_stub(k_bits: int, *, elems: int | None = None, segments=None,
                cb_flat: bool = True):
    """A layer carrying only the load-time metadata ``eligible`` may read."""
    flat = None
    if cb_flat:
        count = lane.cb_elems(k_bits) if elems is None else elems
        flat = torch.zeros(count, dtype=torch.bfloat16)
    return types.SimpleNamespace(_cb_flat=flat,
                                 _cb_fp4_quality_segments=segments)


def _shape(k_bits: int, **overrides):
    """A shape that is eligible in every respect before ``overrides``."""
    kwargs = dict(M=64, N=2048, K=4096, k_bits=k_bits, n_sub=2,
                  type_size=4 * k_bits + 9, is_v2=True)
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("k_bits", _RUNGS)
def test_a_mid_m_quality_shape_is_eligible(k_bits):
    """The positive control: without it every rejection below is vacuous."""
    assert lane.eligible(_ext_stub(), _layer_stub(k_bits),
                         **_shape(k_bits)) is True


def test_a_missing_extension_is_never_eligible():
    """The capability probe may legitimately hand back ``None``."""
    assert lane.eligible(None, _layer_stub(16), **_shape(16)) is False


def test_the_mid_m_window_is_hard_at_both_ends():
    """M=8 belongs to the decode GEMV; M=129 is a second M-tile.

    Both directions are asserted because a one-sided gate is how an
    out-of-range M reaches a kernel that TORCH_CHECKs it and aborts the
    request instead of falling through to the shipping route.
    """
    ext, layer = _ext_stub(), _layer_stub(16)
    assert lane.eligible(ext, layer, **_shape(16, M=8)) is False
    assert lane.eligible(ext, layer, **_shape(16, M=lane.MID_M_MIN)) is True
    assert lane.eligible(ext, layer, **_shape(16, M=128)) is True
    assert lane.eligible(ext, layer, **_shape(16, M=129)) is False


def test_k_must_be_a_whole_number_of_cb_superblocks():
    """The packed stream is addressed in 256-weight superblocks."""
    ext, layer = _ext_stub(), _layer_stub(16)
    for bad_k in (255, 257, 4095):
        assert lane.eligible(ext, layer, **_shape(16, K=bad_k)) is False


def test_n_must_be_aligned_to_the_bf16_epilogue_access():
    ext, layer = _ext_stub(), _layer_stub(16)
    for bad_n in (7, 12, 2047):
        assert lane.eligible(ext, layer, **_shape(16, N=bad_n)) is False


def test_degenerate_shapes_are_rejected():
    """A zero or negative extent is a caller bug, answered by falling back."""
    ext, layer = _ext_stub(), _layer_stub(16)
    assert lane.eligible(ext, layer, **_shape(16, N=0)) is False
    assert lane.eligible(ext, layer, **_shape(16, N=-8)) is False
    assert lane.eligible(ext, layer, **_shape(16, K=0)) is False
    assert lane.eligible(ext, layer, **_shape(16, K=-256)) is False


@pytest.mark.parametrize("override", [
    {"is_v2": False},
    {"n_sub": 1},
    {"type_size": 4 * 16 + 16},
], ids=["v1", "single-sub-table", "wrong-type-size"])
def test_only_the_fp4_v2_product_mode_is_eligible(override):
    """The kernel decodes v2's two sub-tables and 9-byte scale plane only.

    v1 layouts and signed n_sub=1 rungs are a different decode entirely, so a
    shape that merely looks close must stay on the shipping route.
    """
    assert lane.eligible(_ext_stub(), _layer_stub(16),
                         **_shape(16, **override)) is False


def test_a_rung_the_module_did_not_compile_is_rejected():
    """``kbits()`` is read back from the module, not assumed."""
    ext = _ext_stub(kbits=(12, 13, 14, 15))
    assert lane.eligible(ext, _layer_stub(20), **_shape(20)) is False
    assert lane.eligible(ext, _layer_stub(12), **_shape(12)) is True


@pytest.mark.parametrize("k_bits", _RUNGS)
def test_a_codebook_of_the_wrong_size_is_rejected(k_bits):
    """The kernel indexes one zero-based ``[sub0 | sub1]`` dictionary.

    A flat table that is not exactly ``cb_elems(k_bits)`` long is either a
    different rung's dictionary or several interned ones concatenated; both
    would be gathered with the wrong stride and silently decode wrong values.
    """
    exact = lane.cb_elems(k_bits)
    ext = _ext_stub()
    assert lane.eligible(ext, _layer_stub(k_bits, elems=exact - 1),
                         **_shape(k_bits)) is False
    assert lane.eligible(ext, _layer_stub(k_bits, elems=exact + 1),
                         **_shape(k_bits)) is False
    assert lane.eligible(ext, _layer_stub(k_bits, elems=2 * exact),
                         **_shape(k_bits)) is False


def test_a_layer_without_a_staged_codebook_is_rejected():
    """fp8 and v1 layers have no ``_cb_flat``; absence must not raise."""
    assert lane.eligible(_ext_stub(), _layer_stub(16, cb_flat=False),
                         **_shape(16)) is False


def test_two_role_codebooks_stay_on_the_segmented_bridge():
    """A fused qkv shard with A/B role dictionaries is not this kernel's shape.

    ``cb_expand_v2`` takes a single physical codebook with no per-row offset;
    the segmented bridge exists precisely to expand one dictionary at a time.
    """
    segments = ((0, 1024, 0), (1024, 1024, 1))
    assert lane.eligible(_ext_stub(), _layer_stub(16, segments=segments),
                         **_shape(16)) is False


@pytest.mark.parametrize("segments", [((0, 2048, 0),), None],
                         ids=["one-segment", "no-segment-metadata"])
def test_a_single_interned_dictionary_is_eligible(segments):
    """Absent metadata means the loader never split this layer's rows."""
    assert lane.eligible(_ext_stub(), _layer_stub(16, segments=segments),
                         **_shape(16)) is True


# ---------------------------------------------------------------------------
# fused_mm: the binding's positional contract
# ---------------------------------------------------------------------------


def test_fused_mm_passes_the_binding_its_exact_argument_order():
    """Every operand is positional and untyped at the pybind boundary.

    Two of the three tensors are uint8 and two of the three ints are extents,
    so a transposed pair would not raise — it would decode garbage. Pin the
    order here rather than discover it on a served request.
    """
    seen = {}
    result = object()

    def fake_mm(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return result

    ext = types.SimpleNamespace(cb_fused_fp4v2_prefill_mm=fake_mm)
    layer = types.SimpleNamespace(_cb_qw_padded=object(), _cb_flat=object(),
                                  _cb_compose=object())
    a = object()

    out = lane.fused_mm(ext, a, layer, N=2048, K=4096, k_bits=16)

    assert out is result, "the kernel's output must reach the caller untouched"
    assert seen["kwargs"] == {}
    assert seen["args"] == (a, layer._cb_qw_padded, layer._cb_flat,
                            layer._cb_compose, 2048, 4096, 16)


# ---------------------------------------------------------------------------
# Build identity: a header edit can never serve a stale cached kernel
# ---------------------------------------------------------------------------


def _fp4v2_identity_fixture(tmp_path):
    """A stand-in packaged tree holding every declared build input."""
    src = tmp_path / "csrc"
    (src / "cutlass_fork").mkdir(parents=True)
    for name in cuda_ext._FUSED_FP4V2_BUILD_INPUTS:
        (src / name).write_text(f"// {name} v1\n")
    cutlass = _cutlass_tree(tmp_path)
    util = tmp_path / "cutlass-src" / "tools" / "util" / "include"
    packed = util / "cutlass" / "util" / "packed_stride.hpp"
    packed.parent.mkdir(parents=True)
    packed.write_text("// packed v1\n")
    return src, cutlass, util


def _identity(src, cutlass, util, *, capability=(12, 1)):
    fake_cpp = types.SimpleNamespace(CUDA_HOME="/toolkit",
                                     get_cxx_compiler=lambda: "c++")
    return cuda_ext._fused_fp4v2_build_identity(
        _fake_torch(), fake_cpp, src_dir=str(src),
        cutlass_include=str(cutlass), util_include=str(util),
        capability=capability)


def test_every_declared_build_input_is_packaged():
    """A declared input that is absent breaks the load, not just the cache."""
    src = cuda_ext.csrc_dir()
    if not os.path.isfile(os.path.join(src, "cb_fused_fp4v2_gemm.cu")):
        pytest.skip("cb_fused_fp4v2_gemm.cu not present in this install")
    missing = [name for name in cuda_ext._FUSED_FP4V2_BUILD_INPUTS
               if not os.path.isfile(os.path.join(src, name))]
    assert missing == [], (
        f"{missing} are keyed into the fused FP4-v2 build identity but are "
        f"not packaged under {src}")


def test_the_declared_inputs_cover_the_fork_and_the_shared_glue():
    """torch's extension versioner hashes only the ``.cu`` it is handed.

    The decode-in-prologue mainloop and the shared ``AssertSmemFits`` glue are
    where this lane's behaviour actually lives, so leaving either out of the
    identity is exactly how an edited header gets served from cache.
    """
    declared = set(cuda_ext._FUSED_FP4V2_BUILD_INPUTS)
    for name in ("cb_fused_fp4v2_gemm.cu",
                 "cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp",
                 "cb_grouped_common.hpp"):
        assert name in declared, (
            f"{name} does not key the fused FP4-v2 build identity")


def test_the_declared_inputs_are_read_from_the_source_not_restated():
    """Adding a Gridbook include without declaring it must fail here.

    The same gate the FP8 and grouped-BF16 modules carry, and TRANSITIVE since
    2026-08-02: a header reached only through ``cb_grouped_common.hpp`` changes
    this binary exactly as a directly included one does, and reading the
    translation unit's own include list alone is how the grouped-BF16 module
    lost ``sm120_expert_row_broadcast.hpp`` from its cache key.
    """
    path = os.path.join(cuda_ext.csrc_dir(), "cb_fused_fp4v2_gemm.cu")
    if not os.path.isfile(path):
        pytest.skip("cb_fused_fp4v2_gemm.cu not present in this install")
    included = gridbook_include_closure(cuda_ext.csrc_dir(),
                                        "cb_fused_fp4v2_gemm.cu")
    declared = set(cuda_ext._FUSED_FP4V2_BUILD_INPUTS)
    assert included <= declared, (
        f"cb_fused_fp4v2_gemm.cu reaches {sorted(included - declared)}, "
        f"which do not key its build identity — a cached kernel built from "
        f"an older copy of those headers would be served")


@pytest.mark.parametrize("mutated", cuda_ext._FUSED_FP4V2_BUILD_INPUTS)
def test_identity_moves_when_any_build_input_changes(tmp_path, mutated):
    src, cutlass, util = _fp4v2_identity_fixture(tmp_path)

    before, payload = _identity(src, cutlass, util)
    assert payload["schema"] == cuda_ext._FUSED_FP4V2_ABI_SCHEMA
    assert set(payload["inputs"]) == set(cuda_ext._FUSED_FP4V2_BUILD_INPUTS)

    (src / mutated).write_text(f"// {mutated} v2 — one byte of schedule\n")
    after, _ = _identity(src, cutlass, util)
    assert after != before, (
        f"editing {mutated} left the module identity unchanged; the stale "
        f"cached kernel would still be loaded")

    (src / mutated).write_text(f"// {mutated} v1\n")
    assert _identity(src, cutlass, util)[0] == before


def test_identity_is_stable_across_identical_inputs(tmp_path):
    """An unstable digest would rebuild for minutes on every process start."""
    src, cutlass, util = _fp4v2_identity_fixture(tmp_path)
    first, _ = _identity(src, cutlass, util)
    for _ in range(3):
        assert _identity(src, cutlass, util)[0] == first


def test_identity_separates_the_two_supported_targets(tmp_path):
    """sm_120a and sm_121a SASS are not interchangeable."""
    src, cutlass, util = _fp4v2_identity_fixture(tmp_path)
    sm120, _ = _identity(src, cutlass, util, capability=(12, 0))
    sm121, _ = _identity(src, cutlass, util, capability=(12, 1))
    assert sm120 != sm121


@pytest.mark.parametrize("capability,code", [((12, 0), "sm_120a"),
                                             ((12, 1), "sm_121a")])
def test_identity_records_the_architecture_accelerated_target(tmp_path,
                                                              capability,
                                                              code):
    """The sm90-family kernel layer compiles its body only under ``a``.

    A plain ``sm_121`` build of this module loads and then aborts at launch,
    so the recorded target must carry the accelerated suffix.
    """
    src, cutlass, util = _fp4v2_identity_fixture(tmp_path)
    _, payload = _identity(src, cutlass, util, capability=capability)
    assert payload["target"]["code"] == code
    assert payload["target"]["compute"] == code.replace("sm_", "compute_")
    assert payload["target"]["capability"] == list(capability)


@pytest.mark.parametrize("capability,buildable", [
    ((12, 0), True),
    ((12, 1), True),
    ((8, 9), False),
    ((9, 0), False),
    ((12, 2), False),
])
def test_only_the_blackwell_pair_is_buildable(capability, buildable):
    """Checked BEFORE include discovery, so a doomed nvcc run never starts."""
    assert cuda_ext.fused_fp4v2_buildable(capability) is buildable


# ---------------------------------------------------------------------------
# supports(): the LOAD-time per-layer gate (audit follow-up #7)
# ---------------------------------------------------------------------------


def test_supports_accepts_a_plain_fp4_cb_v2_layer():
    ext = _ext_stub()
    assert lane.supports(ext, _layer_stub(16), N=2048, K=4096, k_bits=16,
                         n_sub=2, type_size=73, is_v2=True) is None


@pytest.mark.parametrize("override,mentions", [
    ({"k_bits": 26, "type_size": 4 * 26 + 9}, "not a rung this build"),
    ({"K": 4095}, "superblock"),
    ({"N": 2044}, "8-element"),
    ({"n_sub": 1}, "two-tier v2 product mode"),
    ({"type_size": 72}, "two-tier v2 product mode"),
    ({"is_v2": False}, "two-tier v2 product mode"),
])
def test_supports_rejects_with_a_readable_reason(override, mentions):
    """Each M-independent miss must be diagnosable at LOAD, in a sentence.

    Before this gate existed the lane attested only that the EXTENSION was
    present, so a layer that could never take it served the expand + bridge
    route for every request while the load reported success — the silent
    substitution behind an explicit selection that the flag forbids.
    """
    kwargs = dict(N=2048, K=4096, k_bits=16, n_sub=2, type_size=73,
                  is_v2=True)
    kwargs.update(override)
    reason = lane.supports(_ext_stub(), _layer_stub(kwargs["k_bits"]),
                           **kwargs)
    assert reason is not None and mentions in reason


def test_supports_rejects_a_multi_dictionary_fused_projection():
    reason = lane.supports(
        _ext_stub(), _layer_stub(16, segments=((0, 8, 0), (8, 8, 1))),
        N=2048, K=4096, k_bits=16, n_sub=2, type_size=73, is_v2=True)
    assert reason is not None and "interned codebook blocks" in reason


def test_supports_deliberately_excludes_the_m_band():
    """The M band is a property of the REQUEST, not of the layer.

    Falling through for an out-of-band M is the documented behaviour and is
    what makes this a mid-M lane; making it a load failure would reject every
    layer, since one layer serves every M.
    """
    import inspect

    source = inspect.getsource(lane.supports)
    assert "MID_M_MIN" not in source and "max_m(" not in source


def test_eligible_is_supports_plus_the_m_band():
    """One predicate, not two that can disagree."""
    ext = _ext_stub()
    layer = _layer_stub(16)
    shape = dict(N=2048, K=4096, k_bits=16, n_sub=2, type_size=73,
                 is_v2=True)
    assert lane.supports(ext, layer, **shape) is None
    assert lane.eligible(ext, layer, M=64, **shape)
    assert not lane.eligible(ext, layer, M=8, **shape)
    assert not lane.eligible(ext, layer, M=129, **shape)
    # A layer-level miss makes every M ineligible, including in-band ones.
    bad = _layer_stub(16, segments=((0, 8, 0), (8, 8, 1)))
    assert lane.supports(ext, bad, **shape) is not None
    assert not lane.eligible(ext, bad, M=64, **shape)


def test_facts_is_guarded_on_the_dispatch_path():
    """A module that will not answer reads as "offers nothing", never raises.

    ``eligible`` runs per prefill and its docstring promises a miss falls
    through to the shipping route, so an unguarded read here would raise
    mid-request at exactly the site that promises not to.
    """
    class Hostile:
        def cb_fused_fp4v2_max_m(self):
            raise RuntimeError("module went away")

        def cb_fused_fp4v2_kbits(self):
            raise RuntimeError("module went away")

    ext = Hostile()
    assert lane.max_m(ext) == 0
    assert lane.kbits(ext) == ()
    assert not lane.eligible(ext, _layer_stub(16), M=64, N=2048, K=4096,
                             k_bits=16, n_sub=2, type_size=73, is_v2=True)
