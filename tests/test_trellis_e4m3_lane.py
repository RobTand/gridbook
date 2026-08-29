"""The E4M3 trellis W8A8 serving lane.

Scope, stated up front so a green run is not over-read. What is exercised for
real: the wire decode, the code-plane -> ``float8_e4m3fn`` reinterpretation,
the row-scale-as-``scale_b`` identity, ``torch._scaled_mm`` itself, both
residency modes, and the refusals. What is STUBBED: vLLM's
``LinearMethodBase`` / ``ModelWeightParameter`` (absent from the build venv --
serving runs in its own container) and vLLM's per-token FP8 quantizer, which is
an audited upstream CUDA op, not this lane's contribution. So these tests
establish the lane's NUMERICS and CONTROL FLOW. They do not establish that
vLLM loads it; that owes a container run.

``monkeypatch.setitem(sys.modules, ...)`` follows ``test_mxfp8_dense_lane.py``:
it restores the entries afterwards so stub modules do not leak.

FALSIFIED, 2026-08-29 -- the survivors, published because a green suite proves
a check RUNS, not that it is load-bearing. Each mutant was applied to the LANE
(the driver), not to a fixture, and the suite was re-run:

  1. ``scale_b := ones`` (drop the trellis row scale)        -> 2 failed
  2. resident mode decodes a zero plane instead of the wire  -> 2 failed
  3. streamed mode skips ``decode_native_packed_out``        -> 2 failed
  4. unset mode silently defaults to ``resident``            -> 1 failed

Restored: 11 passed. So the numerics gate, both residency paths and the
no-default rule are each individually load-bearing.
"""
from __future__ import annotations

import struct
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import lane_select, trellis                      # noqa: E402
from gridbook import trellis_decode_pool as decode_pool        # noqa: E402
from gridbook.trellis_e4m3_lane import (                        # noqa: E402
    MODE_RESIDENT,
    MODE_STREAMED,
    TRELLIS_E4M3_FLAG,
    TRELLIS_E4M3_MODE_ENV,
    build_trellis_e4m3_method,
    trellis_e4m3_enabled,
    trellis_e4m3_mode,
)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    # The flag is LATCHED process-wide on purpose (dispatch may not change
    # mid-run), so a test that flips it must clear the latch, exactly as
    # ``test_mxfp8_dense_lane._fresh_flag`` does.
    decode_pool.reset_for_tests()
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.delenv(TRELLIS_E4M3_FLAG, raising=False)
    monkeypatch.delenv(TRELLIS_E4M3_MODE_ENV, raising=False)
    yield
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)


# --- flags and refusals ----------------------------------------------------

def test_flag_defaults_off_and_parses_strictly(monkeypatch):
    assert trellis_e4m3_enabled() is False
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "1")
    assert trellis_e4m3_enabled() is True
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "true")
    with pytest.raises(ValueError, match=TRELLIS_E4M3_FLAG):
        trellis_e4m3_enabled()


def test_mode_has_no_default_and_says_why(monkeypatch):
    """An unset mode is an ERROR, not a pick.

    The two modes differ in the footprint the artifact OCCUPIES, not in its
    numbers, so a default would silently decide a memory claim on the
    operator's behalf. Principle 2: no heuristic where an explicit exists.
    """
    with pytest.raises(ValueError) as exc:
        trellis_e4m3_mode()
    assert TRELLIS_E4M3_MODE_ENV in str(exc.value)
    assert "resident" in str(exc.value) and "streamed" in str(exc.value)
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, "sometimes")
    with pytest.raises(ValueError):
        trellis_e4m3_mode()
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, MODE_STREAMED)
    assert trellis_e4m3_mode() == MODE_STREAMED


def test_lane_refuses_before_any_vllm_import_when_flag_unset():
    """The refusal must beat the vLLM import, or an operator on a box without
    vLLM gets an ImportError instead of the flag's name."""
    with pytest.raises(RuntimeError) as exc:
        build_trellis_e4m3_method(
            {"family": "TCQ_E4M3_R256", "body_rate_q256": 1152, "rows": 4,
             "columns": 256, "wire_bytes": 4096})
    assert TRELLIS_E4M3_FLAG in str(exc.value)
    assert "fail-closed" in str(exc.value)


# --- the numerics ----------------------------------------------------------

def _e4m3_wire(rows: int, columns: int, q256: int):
    """A valid E4M3 wire, built through the packer's own public path."""
    template = trellis.build_q256_schedule(trellis.TCQ_E4M3_R256, q256, 256)
    expanded = tuple(template[i % 256] for i in range(columns))
    terminal = trellis.native_bits(trellis.TCQ_E4M3_R256)
    alphabets = {
        rate: trellis.canonical_full_alphabet(
            trellis.TCQ_E4M3_R256)[:1 << (rate + 1)]
        for rate in sorted({r for r in expanded if r < terminal})
    }
    import random
    rng = random.Random(q256 * 1009 + columns)
    u = [[rng.getrandbits(1) for _ in range(columns)] for _ in range(rows)]
    points = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    finite = [c for c in range(1 << terminal) if c not in (0x7f, 0xff)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                bypass[row][column] = rng.choice(finite)
            else:
                points[row][column] = rng.randrange(1 << (rate - 1))
    scale_blob = struct.pack(
        f"<{rows}f", *[0.5 * (2 ** (row % 4)) for row in range(rows)])
    return trellis.pack_planes(
        family=trellis.TCQ_E4M3_R256, body_rate_q256=q256, schedule=expanded,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256, u_bits=u,
        point_indices=points, bypass_codes=bypass, alphabets=alphabets,
        scale_blob=scale_blob, global_scale_real=1.0)


def _install_vllm_stubs(monkeypatch):
    """Minimal stand-ins for the two vLLM symbols the lane imports."""
    class _LinearMethodBase:
        pass

    def _model_weight_parameter(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _model_weight_parameter
    # The blob parameter carries NO shard dimensions on purpose (a wire has no
    # splittable axis), so the lane uses the dimension-less base class.
    parameter.BasevLLMParameter = _model_weight_parameter
    for name, mod in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
        ("vllm.model_executor.layers",
         types.ModuleType("vllm.model_executor.layers")),
        ("vllm.model_executor.layers.linear", linear),
        ("vllm.model_executor.parameter", parameter),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _reference_quant(x):
    """Per-token dynamic E4M3, the arithmetic vLLM's CUDA op implements.

    Stubbed rather than skipped: an ``importorskip`` here would make the whole
    numerics gate vanish on this box, which is the failure mode where a test
    file is green because it never ran.
    """
    amax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / 448.0
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


class _Layer(torch.nn.Module):
    pass


def _scheme_for(wire):
    """The sidecar declaration a checkpoint carries for this wire."""
    return {"family": wire.family, "body_rate_q256": wire.body_rate_q256,
            "rows": wire.rows, "columns": wire.columns,
            "wire_bytes": len(wire.to_bytes())}


def _load_blob(layer, wire, dev):
    """Do exactly what vLLM's weight loader does: fill the declared blob."""
    blob = wire.to_bytes()
    layer.wire_bytes.data = torch.frombuffer(
        bytearray(blob), dtype=torch.uint8).clone()
    layer.to(dev)


def _drive(monkeypatch, mode, rows=128, columns=512, q256=512, m=32,
           seed=0):
    """Run the lane's three hooks for real; return (got, want, layer, method)."""
    from gridbook import native_cutlass
    from gridbook.trellis_ops import prepare_wire_cuda

    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_cutlass, "native_fp8_quant", _reference_quant)

    wire = _e4m3_wire(rows, columns, q256)
    dev = torch.device("cuda")
    scheme = _scheme_for(wire)
    method = build_trellis_e4m3_method(scheme, "test.layer")

    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[wire.rows], input_size=wire.columns,
        output_size=wire.rows, params_dtype=torch.bfloat16)
    # THE ONLY THING THE LOADER PROVIDES is the blob. The row scale, the
    # prepared wire and the body geometry are all DERIVED from it by the
    # lane -- an earlier revision hand-bound them here, which left the whole
    # load path untested because the test was doing the load path's job.
    _load_blob(layer, wire, dev)
    method.process_weights_after_loading(layer)

    # Seeded: a cross-mode comparison is only meaningful on the SAME input,
    # and an unseeded draw made the two modes look divergent when they were
    # merely fed different activations.
    generator = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, wire.columns, dtype=torch.bfloat16, device=dev,
                    generator=generator)
    got = method.apply(layer, x)

    ref_w = trellis.decode_values_torch(wire, device=dev)
    a_q, a_scale = _reference_quant(x)
    want = ((a_q.float() * a_scale) @ ref_w.t()).to(torch.bfloat16)
    return got, want, layer, method


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_apply_matches_the_wire_contract(monkeypatch, mode):
    """The served output is the wire's own contracted weight, both modes."""
    got, want, _, _ = _drive(monkeypatch, mode)
    rel = ((got.float() - want.float()).abs().max()
           / want.float().abs().max().clamp_min(1e-30)).item()
    assert rel < 8e-3, f"{mode}: relative error {rel:.3e}"


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    """Residency is a FOOTPRINT choice and must never be a numerics choice.

    Stronger than either mode matching the reference: if these two ever
    diverge, one of them is serving different bytes than it priced, which is
    the whole class of defect this lane exists to avoid.
    """
    resident, _, _, _ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    streamed, _, _, _ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(resident, streamed)


@requires_cuda
def test_neither_mode_keeps_the_loaded_wire(monkeypatch):
    """The footprint claim each mode makes must be true of the live layer.

    An earlier revision asserted streamed KEEPS ``trellis_payload``, which
    encoded the defect as the contract: ``prepare_wire_cuda`` clones every wire
    tensor onto the device, so the parameter was a second copy, and with a
    per-layer decode target on top streamed occupied MORE than resident."""
    _, _, resident, _ = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(resident, "trellis_payload")
    assert resident.weight_fp8.dtype is torch.float8_e4m3fn
    assert resident.gridbook_trellis_prepared is None

    _, _, streamed, _ = _drive(monkeypatch, MODE_STREAMED)
    assert not hasattr(streamed, "trellis_payload"), \
        "streamed held the wire twice: the parameter AND the prepared clone"
    assert not hasattr(streamed, "weight_fp8")
    assert streamed.gridbook_trellis_prepared is not None


@requires_cuda
def test_streamed_scratch_is_shared_across_layers(monkeypatch):
    """One decode target per device, not one per layer -- the other half of
    the footprint fix, and the half a per-layer assertion cannot see."""
    _, _, a, _ = _drive(monkeypatch, MODE_STREAMED)
    one = decode_pool.pool_bytes(a.decode_buf.device)
    assert one == a.decode_buf.numel()
    _, _, b, _ = _drive(monkeypatch, MODE_STREAMED)
    assert decode_pool.pool_bytes(b.decode_buf.device) == one, \
        "the pool grew with the layer count; the scratch is not shared"
    # Same UNDERLYING storage, not merely equal pool_bytes -- the weaker form
    # survived mutation because a reallocate-and-rebind also passes it.
    assert a.decode_buf.untyped_storage().data_ptr() == \
        b.decode_buf.untyped_storage().data_ptr()


@requires_cuda
def test_streamed_decode_target_is_allocated_once(monkeypatch):
    """No allocation in the forward: a graph capture would refuse it."""
    _, _, layer, method = _drive(monkeypatch, MODE_STREAMED)
    before = layer.decode_buf.data_ptr()
    # A second forward through the SAME method, at a different row count, so
    # this exercises the real reuse path rather than a fabricated one.
    x = torch.randn(8, layer.trellis_columns, dtype=torch.bfloat16,
                    device=layer.decode_buf.device)
    method.apply(layer, x)
    assert layer.decode_buf.data_ptr() == before


@requires_cuda
def test_shape_mismatch_is_refused_at_create_weights(monkeypatch):
    """A wire is bound to its shape; a mismatched layer is a load error."""
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    method = build_trellis_e4m3_method(
        {"family": "TCQ_E4M3_R256", "body_rate_q256": 1152, "rows": 128,
         "columns": 512, "wire_bytes": 4096})
    with pytest.raises(ValueError, match="bound to its shape"):
        method.create_weights(
            _Layer(), input_size_per_partition=256,
            output_partition_sizes=[128], input_size=256, output_size=128,
            params_dtype=torch.bfloat16)


@requires_cuda
def test_a_wire_that_disagrees_with_its_scheme_is_refused(monkeypatch):
    """The sidecar is a GATE INPUT, not prose: the blob must be what it says.

    Nothing else in the load path compares the two. Without this check a
    checkpoint could serve one artifact while every receipt, byte budget and
    shipcard described another -- which is the failure mode that makes a
    provenance field a confession log instead of a gate.
    """
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    wire = _e4m3_wire(8, 256, 1152)
    dev = torch.device("cuda")

    # A scheme that lies about the row count. create_weights sizes the blob
    # from the scheme, so the length check fires first; make the length agree
    # and the geometry disagree to reach the identity check.
    honest = _scheme_for(wire)
    lying = dict(honest, rows=16)
    method = build_trellis_e4m3_method(lying, "test.layer")
    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[16], input_size=wire.columns,
        output_size=16, params_dtype=torch.bfloat16)
    _load_blob(layer, wire, dev)
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        method.process_weights_after_loading(layer)


@requires_cuda
def test_a_truncated_blob_is_refused_by_length(monkeypatch):
    """Length is checked before parsing, so a short read cannot be reinterpreted."""
    lane_select.reset_for_tests(TRELLIS_E4M3_FLAG)
    monkeypatch.setenv(TRELLIS_E4M3_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E4M3_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    wire = _e4m3_wire(8, 256, 1152)
    scheme = _scheme_for(wire)
    method = build_trellis_e4m3_method(scheme, "test.layer")
    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[wire.rows], input_size=wire.columns,
        output_size=wire.rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.zeros(scheme["wire_bytes"] - 1,
                                        dtype=torch.uint8)
    layer.to(torch.device("cuda"))
    with pytest.raises(ValueError, match="wire_bytes"):
        method.process_weights_after_loading(layer)


def test_the_wire_format_is_what_pins_the_e4m3_global_scale():
    """Where the gsr rule lives, pinned so the lane does not grow a dead copy.

    ``decoded_scales`` reads the per-row fp32 plane and does NOT apply
    ``global_scale_real``, so a wire carrying any other value would be served
    as if it were 1.0. That would be a silent factor -- except the wire's own
    ``validate`` refuses to construct one, and ``from_bytes`` validates. So
    the lane holds no check of its own: this test is the record of why, and
    it fails if the wire format ever stops enforcing it.
    """
    import dataclasses
    wire = _e4m3_wire(8, 256, 1152)
    with pytest.raises(ValueError, match="global_scale_real is fixed at 1.0"):
        dataclasses.replace(wire, global_scale_real=0.5).validate()


@requires_cuda
def test_scale_b_is_derived_from_the_wire_not_loaded(monkeypatch):
    """The served row scale IS the wire's own decoded plane, bit for bit.

    The loader supplies only the blob; a separately carried scale tensor could
    disagree with the bytes it scales and nothing would notice. Deriving makes
    that state unrepresentable, and this is the assertion that says so.
    """
    _g, _w, layer, _m = _drive(monkeypatch, MODE_STREAMED)
    wire = _e4m3_wire(128, 512, 512)
    want = torch.tensor([row[0] for row in trellis.decoded_scales(wire)],
                        dtype=torch.float32, device=layer.scale_b.device)
    assert torch.equal(layer.scale_b.reshape(-1), want)
