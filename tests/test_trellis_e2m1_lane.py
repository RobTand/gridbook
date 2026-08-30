"""The E2M1 trellis W4A4 serving lane.

Scope, stated up front so a green run is not over-read. Exercised for real: the
wire decode, the nibble-plane -> ``float4_e2m1fn_x2`` reinterpretation, the
group-16 ue4m3 plane as ``scale_b`` through the cuBLAS 128x4 blocking,
``torch._scaled_mm`` itself, the scalar epilogue, both residency modes, and the
refusals. STUBBED: vLLM's ``LinearMethodBase`` / ``ModelWeightParameter``
(absent from the build venv) and the NVFP4 activation quantizer, which is an
audited upstream CUDA op, not this lane's contribution. So these tests
establish the lane's NUMERICS and CONTROL FLOW. **They do not establish that
vLLM loads it; that owes a container run.**

The blocking check is the load-bearing one, and it is pinned against the wire's
own contracted product rather than against another implementation of the
layout: an unswizzled plane is ACCEPTED by ``_scaled_mm`` and silently
miscomputes by 67-70% -- at aligned shapes as well as unaligned ones, so no
shape is a safe place to skip the swizzle. Ground truth, not agreement, is what
separates those two.

FALSIFIED, 2026-08-29 -- the survivors, published because a green suite proves
a check RUNS, not that it is load-bearing. Each mutant was applied to the LANE
(the driver), not to a fixture, and the suite was re-run:

  1. ``blocked_scales`` degraded to plain zero-padding      -> see below
  2. ``scale_b := ones`` (drop the group scale plane)       -> see below
  3. resident mode decodes a zero plane instead of the wire -> see below
  4. streamed mode skips ``decode_native_packed_out``       -> see below
  5. epilogue multiply dropped (global_scale_real ignored)  -> see below
"""
from __future__ import annotations

import random
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import lane_select, trellis                       # noqa: E402
from gridbook import trellis_decode_pool as decode_pool        # noqa: E402
from gridbook.trellis_e2m1_lane import (                        # noqa: E402
    MODE_RESIDENT,
    MODE_STREAMED,
    TRELLIS_E2M1_FLAG,
    TRELLIS_E2M1_MODE_ENV,
    ACTIVATION_CONTRACT,
    blocked_scales,
    build_trellis_e2m1_method,
    trellis_e2m1_enabled,
    trellis_e2m1_mode,
)
from gridbook.qtip_hadamard import (                            # noqa: E402
    ONLINE_HADAMARD_FLAG, SIGN_GENERATOR, TRANSFORM_ALGORITHM,
    TRANSFORM_NORMALIZATION, TRANSFORM_PADDING, TRANSFORM_SCHEMA,
    apply_input_transform, apply_inverse_output_transform,
    online_transform_digest, seeded_sign_digest, seeded_signs)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
GROUP = 16


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    decode_pool.reset_for_tests()
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.delenv(TRELLIS_E2M1_FLAG, raising=False)
    monkeypatch.delenv(TRELLIS_E2M1_MODE_ENV, raising=False)
    lane_select.reset_for_tests(ONLINE_HADAMARD_FLAG)
    monkeypatch.delenv(ONLINE_HADAMARD_FLAG, raising=False)
    yield
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    decode_pool.reset_for_tests()
    lane_select.reset_for_tests(ONLINE_HADAMARD_FLAG)


# --- flags and refusals ----------------------------------------------------

def test_flag_defaults_off_and_parses_strictly(monkeypatch):
    assert trellis_e2m1_enabled() is False
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "1")
    assert trellis_e2m1_enabled() is True
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "yes")
    with pytest.raises(ValueError, match=TRELLIS_E2M1_FLAG):
        trellis_e2m1_enabled()


def test_mode_has_no_default_and_says_why(monkeypatch):
    with pytest.raises(ValueError) as exc:
        trellis_e2m1_mode()
    assert TRELLIS_E2M1_MODE_ENV in str(exc.value)
    assert "resident" in str(exc.value) and "streamed" in str(exc.value)
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, "whichever")
    with pytest.raises(ValueError):
        trellis_e2m1_mode()
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, MODE_RESIDENT)
    assert trellis_e2m1_mode() == MODE_RESIDENT


def test_lane_refuses_before_any_vllm_import_when_flag_unset():
    with pytest.raises(RuntimeError) as exc:
        build_trellis_e2m1_method(
            {"family": "TCQ_E2M1_R256", "body_rate_q256": 512, "rows": 4,
             "columns": 256, "wire_bytes": 4096})
    assert TRELLIS_E2M1_FLAG in str(exc.value)
    assert "fail-closed" in str(exc.value)


def test_activation_contract_is_the_shared_vocabulary():
    """The stamped contract must be a name the route probe already knows."""
    from gridbook import nvfp4_activation_contract as nac
    assert ACTIVATION_CONTRACT == nac.EXECUTION_CONTRACT
    assert "e2m1" in ACTIVATION_CONTRACT and "group16" in ACTIVATION_CONTRACT


# --- the blocked scale layout ---------------------------------------------

def test_blocked_scales_is_not_mere_padding():
    """At rows%128 != 0 the blocked layout must REORDER, not just pad.

    This is the check that separates the correct layout from the one
    ``_scaled_mm`` accepts and silently misreads.
    """
    rows, groups = 192, 32
    plane = torch.arange(rows * groups, dtype=torch.int32).remainder(200) \
                 .add(1).to(torch.uint8).view(rows, groups)
    out = blocked_scales(plane.view(torch.float8_e4m3fn))
    padded = torch.zeros(256, groups, dtype=torch.uint8)
    padded[:rows] = plane
    assert out.numel() == 256 * groups
    assert not torch.equal(out.view(torch.uint8), padded.flatten()), (
        "blocked_scales returned row-major padding; _scaled_mm accepts that "
        "and returns ~66% relative error")


@pytest.mark.parametrize("rows,groups", [(128, 64), (128, 4), (256, 64)])
def test_blocked_scales_is_never_row_major(rows, groups):
    """Even at a perfectly aligned shape the blocked layout is NOT row-major.

    An earlier revision of this test asserted the opposite -- that the two
    coincide when aligned, which was the stated reason a row-major plane
    "appears to work" at aligned shapes. Measured
    (``dq-runs/trellis-kernel-20260829/_q9_rowmajor_at_aligned.py``) that is
    false in both halves: the 128x4 tile is swizzled internally (the 32x4x4
    shuffle), so no shape makes it row-major, and a row-major plane is 67-70%
    wrong at ALIGNED shapes too. It never worked anywhere; it only failed to
    raise. That makes the trap worse than documented, not milder."""
    plane = torch.arange(rows * groups, dtype=torch.int32).remainder(200) \
                 .add(1).to(torch.uint8).view(rows, groups)
    out = blocked_scales(plane.view(torch.float8_e4m3fn)).view(torch.uint8)
    assert out.numel() == rows * groups
    assert not torch.equal(out, plane.flatten())
    # A permutation, not a transform: same multiset of bytes.
    assert torch.equal(out.sort().values, plane.flatten().sort().values)


# --- the numerics ----------------------------------------------------------

def _e2m1_wire(rows: int, columns: int, q256: int):
    """A valid E2M1 wire, built through the packer's own public path."""
    family = trellis.TCQ_E2M1_R256
    template = trellis.build_q256_schedule(family, q256, 256)
    expanded = tuple(template[i % 256] for i in range(columns))
    terminal = trellis.native_bits(family)
    alphabets = {
        rate: trellis.canonical_full_alphabet(family)[:1 << (rate + 1)]
        for rate in sorted({r for r in expanded if r < terminal})
    }
    rng = random.Random(q256 * 1009 + columns)
    u = [[rng.getrandbits(1) for _ in range(columns)] for _ in range(rows)]
    points = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                bypass[row][column] = rng.randrange(1 << terminal)
            else:
                points[row][column] = rng.randrange(1 << (rate - 1))
    groups = (columns + GROUP - 1) // GROUP
    # e4m3 codes in a few binades, none of them the NaN codes.
    scale_blob = bytes(0x30 + ((row * groups + g) % 0x14)
                       for row in range(rows) for g in range(groups))
    return trellis.pack_planes(
        family=family, body_rate_q256=q256, schedule=expanded,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256, u_bits=u,
        point_indices=points, bypass_codes=bypass, alphabets=alphabets,
        scale_blob=scale_blob, global_scale_real=0.5)


def _install_vllm_stubs(monkeypatch):
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


_LAST_A_DEQUANT = {}


def _reference_fp4_quant(x, global_scale):
    """Group-16 static-global-scale E2M1, the arithmetic the CUDA op implements.

    Stubbed rather than skipped: an ``importorskip`` here would make the whole
    numerics gate vanish on this box -- the failure mode where a file is green
    because it never ran. Records the exact value it represents so the test's
    expectation is the A side the lane actually consumed, not a re-derivation.
    """
    m, k = x.shape
    _LAST_A_DEQUANT["input"] = x.clone()
    groups = k // GROUP
    xf = x.float().view(m, groups, GROUP)
    amax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    # NVFP4: per-group scale is stored as e4m3 of (amax/6 * global_scale).
    sf = (amax / 6.0 * global_scale.float()).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    q = (xf * global_scale.float() / sf_f).clamp(-6.0, 6.0)
    e2m1_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=x.device)
    mag = q.abs().unsqueeze(-1)
    idx = (mag - e2m1_levels).abs().argmin(dim=-1)
    vals = e2m1_levels[idx] * torch.sign(q)
    codes = idx.to(torch.uint8) | (q < 0).to(torch.uint8) * 8
    codes = torch.where(vals == 0, torch.zeros_like(codes), codes)
    flat = codes.view(m, k)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).contiguous()
    # What these bytes REPRESENT, before the 1/global_scale epilogue.
    _LAST_A_DEQUANT["value"] = (vals * sf_f).view(m, k)
    plane = sf.view(m, groups)
    # PLAIN uint8, matching vLLM's production ``native_fp4_quant``. An earlier
    # revision returned ``.view(torch.float4_e2m1fn_x2)`` -- a difference that
    # looks cosmetic and is not: ``_scaled_mm`` rejects a uint8 A against a
    # float4 B, so the stub was hiding a real load-path defect that only the
    # container run could surface. A stub that is easier than production is a
    # gate with a hole in it.
    return packed, blocked_scales(plane)


class _Layer(torch.nn.Module):
    pass


def _scheme_for(wire):
    """The sidecar declaration a checkpoint carries for this wire."""
    return {"family": wire.family, "body_rate_q256": wire.body_rate_q256,
            "rows": wire.rows, "columns": wire.columns,
            "wire_bytes": len(wire.to_bytes())}


def _online_transform(rows, columns):
    def side(role, dimension, block_size, seed):
        return {
            "dimension": dimension, "block_size": block_size, "seed": seed,
            "sign_generator": SIGN_GENERATOR,
            "sign_sha256": seeded_sign_digest(role, dimension, seed),
        }
    transform = {
        "schema": TRANSFORM_SCHEMA,
        "algorithm": TRANSFORM_ALGORITHM,
        "normalization": TRANSFORM_NORMALIZATION,
        "padding": TRANSFORM_PADDING,
        "input": side("input", columns, 256, 11),
        "output": side("output", rows, 128, 29),
    }
    transform["transform_sha256"] = online_transform_digest(transform)
    return transform


def _load_blob(layer, wire, dev, input_global_scale=4.0):
    """Do exactly what vLLM's weight loader does: fill the declared params."""
    blob = wire.to_bytes()
    layer.wire_bytes.data = torch.frombuffer(
        bytearray(blob), dtype=torch.uint8).clone()
    layer.trellis_input_global_scale.data = torch.tensor(
        [float(input_global_scale)], dtype=torch.float32)
    layer.to(dev)


def _drive(monkeypatch, mode, rows=256, columns=1024, q256=512, m=32,
           seed=0, online_transform=False):
    """Run the lane's three hooks for real; return (got, want, layer, method)."""
    from gridbook import native_cutlass
    from gridbook.trellis_ops import prepare_wire_cuda

    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, mode)
    lane_select.reset_for_tests(ONLINE_HADAMARD_FLAG)
    if online_transform:
        monkeypatch.setenv(ONLINE_HADAMARD_FLAG, "1")
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_cutlass, "native_fp4_quant",
                        _reference_fp4_quant)

    wire = _e2m1_wire(rows, columns, q256)
    dev = torch.device("cuda")
    scheme = _scheme_for(wire)
    if online_transform:
        scheme["online_transform"] = _online_transform(rows, columns)
    method = build_trellis_e2m1_method(scheme, "test.layer")

    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[wire.rows], input_size=wire.columns,
        output_size=wire.rows, params_dtype=torch.bfloat16)
    # THE LOADER PROVIDES THE BLOB AND THE A-SIDE SCALE, nothing else. The
    # group-scale plane, global_scale_real, the prepared wire and the epilogue
    # scalar are all DERIVED -- an earlier revision hand-bound four of them
    # here, which left the load path untested because the test performed it.
    _load_blob(layer, wire, dev, input_global_scale=4.0)
    method.process_weights_after_loading(layer)
    gs = layer.trellis_input_global_scale.data.reshape(())

    generator = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, wire.columns, dtype=torch.bfloat16, device=dev,
                    generator=generator)
    got = method.apply(layer, x)

    # WANT: the A side the lane actually consumed, times the wire's OWN
    # contracted weight. decode_values_torch already folds global_scale_real,
    # so this is the whole product with no re-derivation of the lane's math.
    ref_w = trellis.decode_values_torch(wire, device=dev)
    a_deq = _LAST_A_DEQUANT["value"] / float(gs)
    want = (a_deq @ ref_w.t()).to(torch.bfloat16)
    if online_transform:
        signs = seeded_signs("output", rows, 29, device=dev,
                             dtype=torch.bfloat16)
        want = apply_inverse_output_transform(want, signs, 128)
    return got, want, layer, method


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_lane_reproduces_the_wire_contract(monkeypatch, mode):
    got, want, _layer, _m = _drive(monkeypatch, mode)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    err = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1e-9)
    assert err / scale < 8e-3, f"{mode}: rel err {err / scale:.3e}"


@requires_cuda
@pytest.mark.parametrize("rows", [192, 320])
def test_unaligned_rows_are_exact_too(monkeypatch, rows):
    """rows%128 != 0 is where a padded-but-unswizzled plane silently lies."""
    got, want, _layer, _m = _drive(monkeypatch, MODE_STREAMED, rows=rows,
                                   columns=512)
    err = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1e-9)
    assert err / scale < 8e-3, f"rows={rows}: rel err {err / scale:.3e}"


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    a, _w, _l, _m = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, _w2, _l2, _m2 = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b), "residency must not change the numbers"


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_qtip_online_transform_wraps_native_w4a4_apply(monkeypatch, mode):
    got, want, layer, _method = _drive(
        monkeypatch, mode, online_transform=True)
    assert torch.equal(got, want)
    assert layer.gridbook_online_transform_contract == TRANSFORM_SCHEMA
    assert layer.qtip_input_signs.shape == (layer.trellis_columns,)
    assert layer.qtip_output_signs.shape == (layer.trellis_rows,)
    # The quantizer must consume R_in*x, not the original BF16 activation.
    # Reconstruct its source from the deterministic test generator.
    generator = torch.Generator(device="cuda").manual_seed(0)
    original = torch.randn(
        32, layer.trellis_columns, dtype=torch.bfloat16, device="cuda",
        generator=generator)
    expected_quant_input = apply_input_transform(
        original, layer.qtip_input_signs, layer.qtip_input_block_size)
    assert torch.equal(_LAST_A_DEQUANT["input"], expected_quant_input)


@requires_cuda
def test_neither_mode_keeps_the_loaded_wire(monkeypatch):
    """Both modes drop ``trellis_payload``, for the same reason.

    An earlier revision asserted streamed KEEPS it, encoding the defect as the
    contract. ``prepare_wire_cuda`` clones every wire tensor onto the device,
    so holding the parameter as well stored the wire twice -- and with a
    per-layer decode target on top, streamed occupied MORE than resident,
    inverting the whole point of the mode."""
    _g, _w, res, _m = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(res, "trellis_payload")
    assert res.gridbook_trellis_prepared is None
    assert res.weight_fp4.numel() == res.trellis_rows * res.trellis_columns // 2
    _g2, _w2, stm, _m2 = _drive(monkeypatch, MODE_STREAMED)
    assert not hasattr(stm, "trellis_payload"), \
        "streamed held the wire twice: the parameter AND the prepared clone"
    assert stm.gridbook_trellis_prepared is not None
    assert not hasattr(stm, "weight_fp4")


@requires_cuda
def test_streamed_scratch_is_shared_across_layers(monkeypatch):
    """The decode target is ONE buffer per device, not one per layer.

    This is the other half of the footprint fix, and it is the half a
    per-layer test cannot see: each layer's ``decode_buf`` looked correct in
    isolation while the total scaled with the layer count."""
    _g, _w, a, _m = _drive(monkeypatch, MODE_STREAMED, rows=128, columns=512)
    one = decode_pool.pool_bytes(a.decode_buf.device)
    assert one == a.decode_buf.numel()
    _g2, _w2, b, _m2 = _drive(monkeypatch, MODE_STREAMED, rows=128,
                              columns=512, seed=1)
    assert decode_pool.pool_bytes(b.decode_buf.device) == one, \
        "the pool grew with the layer count; the scratch is not shared"
    # The load-bearing assertion is SAME UNDERLYING STORAGE, not equal
    # pool_bytes: pool_bytes reports only the current allocation, so it cannot
    # see buffers orphaned in an earlier one. (Both weaker forms survived
    # mutation; this one does not.)
    assert a.decode_buf.untyped_storage().data_ptr() == \
        b.decode_buf.untyped_storage().data_ptr()


@requires_cuda
def test_pool_growth_rebinds_earlier_layers(monkeypatch):
    """A layer finalized before a LARGER one must not keep a freed view."""
    _g, _w, small, _m = _drive(monkeypatch, MODE_STREAMED, rows=128,
                               columns=512)
    _g2, _w2, big, _m2 = _drive(monkeypatch, MODE_STREAMED, rows=256,
                                columns=1024)
    total = decode_pool.pool_bytes(big.decode_buf.device)
    assert total == big.decode_buf.numel() > small.decode_buf.numel()
    # The small layer must be re-sliced INTO the grown storage. Without the
    # rebind it keeps a live view of the old one -- which does not crash and
    # does not change pool_bytes, it just quietly keeps both allocations, i.e.
    # exactly the per-layer footprint the pool exists to remove.
    assert small.decode_buf.untyped_storage().data_ptr() == \
        big.decode_buf.untyped_storage().data_ptr()
    assert small.decode_buf.numel() == 128 * 512 // 2
    x = torch.randn(4, small.trellis_columns, dtype=torch.bfloat16,
                    device=small.decode_buf.device)
    _m.apply(small, x)


@requires_cuda
def test_epilogue_is_derived_not_accepted(monkeypatch):
    """A loader cannot bind an epilogue that disagrees with the two scales."""
    _g, _w, layer, _m = _drive(monkeypatch, MODE_STREAMED)
    # global_scale_real is no longer bindable at all -- it is read from the
    # wire header. The fixture is seeded, so this is the SAME wire the lane
    # parsed, and the only remaining loader input is the A-side scale.
    wire = _e2m1_wire(256, 1024, 512)
    want = (float(wire.global_scale_real)
            / float(layer.trellis_input_global_scale.data.reshape(())))
    assert layer.trellis_epilogue_scale == want


@requires_cuda
def test_streamed_decode_target_is_allocated_once(monkeypatch):
    _g, _w, layer, method = _drive(monkeypatch, MODE_STREAMED)
    ptr = layer.decode_buf.data_ptr()
    x = torch.randn(8, layer.trellis_columns, dtype=torch.bfloat16,
                    device=layer.decode_buf.device)
    method.apply(layer, x)
    assert layer.decode_buf.data_ptr() == ptr, \
        "the forward reallocated its decode target; that is not graph-safe"


@requires_cuda
def test_wire_shape_must_match_the_layer(monkeypatch):
    from gridbook import native_cutlass
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_cutlass, "native_fp4_quant",
                        _reference_fp4_quant)
    method = build_trellis_e2m1_method(
        {"family": "TCQ_E2M1_R256", "body_rate_q256": 512, "rows": 256,
         "columns": 1024, "wire_bytes": 4096})
    with pytest.raises(ValueError, match="bound to its shape"):
        method.create_weights(_Layer(), input_size_per_partition=512,
                              output_partition_sizes=[256], input_size=512,
                              output_size=256, params_dtype=torch.bfloat16)


@requires_cuda
def test_a_wire_that_disagrees_with_its_scheme_is_refused(monkeypatch):
    """The sidecar is a GATE INPUT, not prose: the blob must be what it says."""
    from gridbook import native_cutlass
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_cutlass, "native_fp4_quant",
                        _reference_fp4_quant)
    wire = _e2m1_wire(128, 512, 512)
    lying = dict(_scheme_for(wire), rows=64)
    method = build_trellis_e2m1_method(lying, "test.layer")
    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[64], input_size=wire.columns,
        output_size=64, params_dtype=torch.bfloat16)
    _load_blob(layer, wire, torch.device("cuda"))
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        method.process_weights_after_loading(layer)


@requires_cuda
def test_a_nonpositive_input_global_scale_is_refused(monkeypatch):
    """The A-side scale DIVIDES the activations, so it must fail at load.

    It is the one quantity here that is genuinely not a wire fact, so it is
    the one that is loaded rather than derived -- which means it is also the
    one a checkpoint can get wrong. Failing at the first forward could happen
    inside a graph capture.
    """
    from gridbook import native_cutlass
    lane_select.reset_for_tests(TRELLIS_E2M1_FLAG)
    monkeypatch.setenv(TRELLIS_E2M1_FLAG, "1")
    monkeypatch.setenv(TRELLIS_E2M1_MODE_ENV, MODE_STREAMED)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_cutlass, "native_fp4_quant",
                        _reference_fp4_quant)
    wire = _e2m1_wire(128, 512, 512)
    method = build_trellis_e2m1_method(_scheme_for(wire), "test.layer")
    layer = _Layer()
    method.create_weights(
        layer, input_size_per_partition=wire.columns,
        output_partition_sizes=[wire.rows], input_size=wire.columns,
        output_size=wire.rows, params_dtype=torch.bfloat16)
    _load_blob(layer, wire, torch.device("cuda"), input_global_scale=0.0)
    with pytest.raises(ValueError, match="trellis_input_global_scale"):
        method.process_weights_after_loading(layer)
