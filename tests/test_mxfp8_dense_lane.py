"""MXFP8 dense lane policy: flag semantics, registry wiring, OPT-IN gate.

CPU-only, vLLM-free: everything here is refusal/policy behaviour that must be
testable without a GPU (the kernel itself is covered by
``test_mxfp8_dense_gemm.py`` on the GPU image).
"""
import pytest

torch = pytest.importorskip("torch")

from gridbook import lane_select  # noqa: E402
from gridbook import source_passthrough as sp  # noqa: E402
from gridbook.mxfp8_dense_lane import (  # noqa: E402
    MXFP8_DENSE_FLAG,
    WIRE_MXFP8_G32,
    build_mxfp8_dense_method,
    mxfp8_dense_enabled,
)

WIRE_FP8_BLOCK128 = "fp8_e4m3_ue8m0_block128"


@pytest.fixture(autouse=True)
def _fresh_flag(monkeypatch):
    lane_select.reset_for_tests(MXFP8_DENSE_FLAG)
    monkeypatch.delenv(MXFP8_DENSE_FLAG, raising=False)
    yield
    lane_select.reset_for_tests(MXFP8_DENSE_FLAG)


def test_flag_defaults_off_and_parses_strictly(monkeypatch):
    assert mxfp8_dense_enabled() is False
    lane_select.reset_for_tests(MXFP8_DENSE_FLAG)
    monkeypatch.setenv(MXFP8_DENSE_FLAG, "1")
    assert mxfp8_dense_enabled() is True
    lane_select.reset_for_tests(MXFP8_DENSE_FLAG)
    monkeypatch.setenv(MXFP8_DENSE_FLAG, "true")
    with pytest.raises(ValueError, match=MXFP8_DENSE_FLAG):
        mxfp8_dense_enabled()


def test_unknown_wire_id_refused_before_any_vllm_import():
    with pytest.raises(ValueError, match="unknown MXFP8 dense wire id"):
        build_mxfp8_dense_method("mxfp8_e5m2_g32")


def test_block128_hard_refuses_the_w8a8_mxfp8_method():
    with pytest.raises(ValueError, match="W8A16.*cannot enter.*W8A8"):
        build_mxfp8_dense_method(WIRE_FP8_BLOCK128)


def test_registry_splits_block_w8a16_from_direct_g32_w8a8():
    assert WIRE_FP8_BLOCK128 in sp.FORMATS
    assert WIRE_MXFP8_G32 in sp.FORMATS
    direct = sp.FORMATS[WIRE_MXFP8_G32]
    assert direct.unit_kind == "linear"
    assert direct.audited_backends == frozenset({"Mxfp8DenseLinearMethod"})
    assert direct.audited_capabilities == ((12, 1),)
    block = sp.FORMATS[WIRE_FP8_BLOCK128]
    assert block.audited_backends == frozenset({
        "Fp8SourceW8A16LinearMethod"})
    assert block.quantizes_activations is False
    assert direct.quantizes_activations is True
    for fmt in (block, direct):
        assert fmt.method_factory in sp._FACTORIES


def test_parse_declaration_accepts_mxfp8_linear_units():
    resolved = sp.parse_declaration({
        sp.SCHEMA_KEY: {
            "version": 1,
            "units": {
                "model.layers.3.self_attn.wq_a": WIRE_FP8_BLOCK128,
                "model.layers.3.mlp.shared_experts.w1": WIRE_MXFP8_G32,
            },
        }
    })
    assert resolved["model.layers.3.self_attn.wq_a"].id == WIRE_FP8_BLOCK128
    assert resolved["model.layers.3.mlp.shared_experts.w1"].id == WIRE_MXFP8_G32


def test_direct_g32_opt_in_gate_refuses_with_flag_named():
    """OPT-IN as a load-time refusal: correctness is audited, the serve-parity
    bench is pending, and the lane must not enable itself."""
    factory = sp._FACTORIES[
        "gridbook.source_passthrough:_build_gridbook_mxfp8_direct_method"]
    with pytest.raises(sp.SourcePassthroughError) as exc:
        factory(None, "model.layers.3.self_attn.wq_a")
    msg = str(exc.value)
    assert MXFP8_DENSE_FLAG in msg
    assert "serve-parity bench is pending" in msg


def test_opt_in_gate_opens_with_flag_and_fails_only_on_vllm(monkeypatch):
    """With the flag set, the gate passes and construction proceeds to the
    lazy vLLM import — proving the refusal above is the flag, not an
    accident of a missing dependency."""
    monkeypatch.setenv(MXFP8_DENSE_FLAG, "1")
    factory = sp._FACTORIES[
        "gridbook.source_passthrough:_build_gridbook_mxfp8_direct_method"]
    try:
        method = factory(None, "u")
    except ModuleNotFoundError as exc:
        assert "vllm" in str(exc)
    else:
        assert type(method).__name__ == "Mxfp8DenseLinearMethod"


def test_audited_and_broken_sets_stay_disjoint_for_fp8_source_entries():
    for wire in (WIRE_FP8_BLOCK128, WIRE_MXFP8_G32):
        fmt = sp.FORMATS[wire]
        assert not (fmt.audited_backends & set(fmt.known_broken_backends))


# --- A-side offset pre-warm: the CUDA-graph capture prerequisite ------------
#
# ``_SfOffsetCache.get`` computes offsets on the host and moves them with an
# unpinned ``.to(device)``. Inside a CUDA graph capture that copy is a hard
# error, so every key must already exist before capture starts. The B side was
# always resolved at load from weight dimensions; the A side is keyed by the
# runtime row count, which under FULL_DECODE_ONLY is first seen INSIDE the
# capture region. These cover the reader and the pre-warm on CPU; the served
# proof is a graph capture on the GPU image.

class _FakeExt:
    """Records the offset requests the lane makes."""

    def __init__(self):
        self.calls = []

    def mxfp8_sf_offsets(self, rows, k, is_b):
        self.calls.append((int(rows), int(k), bool(is_b)))
        return torch.zeros(4, dtype=torch.int32)


def test_capture_sizes_reader_is_silent_without_vllm():
    """No vLLM in the CPU tier: the reader must degrade to (), not raise --
    a load must never fail for a compilation-config schema reason."""
    from gridbook.mxfp8_dense_lane import _cudagraph_capture_sizes
    assert _cudagraph_capture_sizes() == ()


def test_capture_sizes_reader_dedups_sorts_and_drops_nonpositive(monkeypatch):
    """Exercise the REAL reader against a stub config.

    ``monkeypatch.setitem`` on ``sys.modules`` is deliberate: it restores the
    entries afterwards, so this does not leak stub ``vllm`` modules into later
    files the way CONTRIBUTING.md warns about.
    """
    import sys
    import types

    import gridbook.mxfp8_dense_lane as lane

    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            cudagraph_capture_sizes=[8, 2, 2, 0, 4, 1, -3]))
    stub = types.ModuleType("vllm.config")
    stub.get_current_vllm_config = lambda: config
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", stub)

    assert lane._cudagraph_capture_sizes() == (1, 2, 4, 8)


def test_capture_sizes_reader_degrades_to_silence_on_a_hostile_config(
        monkeypatch):
    """Schema drift must not fail a load -- the reason ops.py warns instead of
    raising. A config whose accessor explodes yields (), not an exception."""
    import sys
    import types

    import gridbook.mxfp8_dense_lane as lane

    def boom():
        raise RuntimeError("compilation_config moved again")

    stub = types.ModuleType("vllm.config")
    stub.get_current_vllm_config = boom
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", stub)

    assert lane._cudagraph_capture_sizes() == ()


def test_prewarm_populates_a_side_for_every_capture_size(monkeypatch):
    """The fix proper: one A-side entry per capture size, at the layer's K."""
    import gridbook.mxfp8_dense_lane as lane

    monkeypatch.setattr(lane, "_OFFSETS", lane._SfOffsetCache())
    monkeypatch.setattr(lane, "_cudagraph_capture_sizes", lambda: (1, 2, 4, 8))
    ext = _FakeExt()
    lane._prewarm_activation_offsets(ext, 4096, torch.device("cpu"))

    assert ext.calls == [(1, 4096, False), (2, 4096, False),
                         (4, 4096, False), (8, 4096, False)]
    # and a subsequent forward-time lookup is a cache HIT, i.e. no host copy
    # would be issued inside a capture
    before = len(ext.calls)
    lane._OFFSETS.get(ext, 4, 4096, is_b=False, device=torch.device("cpu"))
    assert len(ext.calls) == before


def test_prewarm_is_a_noop_when_capture_sizes_are_unknown(monkeypatch):
    """Eager serving (or an unreadable config) must not pay for this."""
    import gridbook.mxfp8_dense_lane as lane

    monkeypatch.setattr(lane, "_OFFSETS", lane._SfOffsetCache())
    monkeypatch.setattr(lane, "_cudagraph_capture_sizes", tuple)
    ext = _FakeExt()
    lane._prewarm_activation_offsets(ext, 4096, torch.device("cpu"))
    assert ext.calls == []
