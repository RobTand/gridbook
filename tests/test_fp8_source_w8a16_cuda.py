"""Spark release gate for raw-resident block128 source-FP8 W8A16.

The oracle decodes the raw E4M3 and UE8M0 bytes here, independently of every
Gridbook format helper and native symbol.  The tests exercise the installed
wheel's two low-level CUDA bindings, the opaque whole-method dispatch, DSV4
group mapping, fullgraph opacity, changed-input CUDA-graph replay, fail-closed
contracts, bounded transient lifetime, and non-gating representative timings.
"""
from __future__ import annotations

import gc
import types
import weakref

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("source-FP8 W8A16 release gate needs CUDA",
                allow_module_level=True)
if tuple(torch.cuda.get_device_capability()) != (12, 1):
    pytest.skip("source-FP8 W8A16 release target is Spark / sm_121",
                allow_module_level=True)

from gridbook import cuda_ext, dsv4_woa, ops  # noqa: E402
from gridbook import fp8_source_w8a16 as source_lane  # noqa: E402
from gridbook.fp8_source_w8a16 import (  # noqa: E402
    WIRE_FP8_BLOCK128,
    build_fp8_source_w8a16_method,
)

SOURCE_EXT = cuda_ext.get_fp8_source_w8a16_ext()
if SOURCE_EXT is None:
    pytest.skip("installed source-FP8 W8A16 extension is unavailable",
                allow_module_level=True)
GROUPED_EXT = cuda_ext.get_bf16_grouped_ext()
if GROUPED_EXT is None:
    pytest.skip("installed grouped-BF16 bridge is unavailable",
                allow_module_level=True)

DEV = torch.device("cuda")


def _raw_planes(n: int, k: int, *, seed: int = 1):
    """Finite E4M3 bytes plus nontrivial block128 UE8M0 bytes on CUDA."""

    generator = torch.Generator(device=DEV).manual_seed(seed)
    raw_q = torch.randint(
        0, 256, (n, k), dtype=torch.uint8, device=DEV,
        generator=generator)
    # E4M3FN reserves sign-agnostic exponent=15,mantissa=7 as NaN.
    raw_q.masked_fill_((raw_q & 0x7f) == 0x7f, 0x7e)
    scale_shape = ((n + 127) // 128, (k + 127) // 128)
    raw_scale = torch.randint(
        118, 126, scale_shape, dtype=torch.uint8, device=DEV,
        generator=generator)
    # Make every block visibly distinct so a row/column indexing swap cannot
    # cancel under random data, including partial edge blocks.
    row = torch.arange(scale_shape[0], device=DEV, dtype=torch.int16)[:, None]
    col = torch.arange(scale_shape[1], device=DEV, dtype=torch.int16)[None, :]
    raw_scale.copy_(((119 + 3 * row + 5 * col) % 13 + 113).to(torch.uint8))
    return (raw_q.view(torch.float8_e4m3fn),
            raw_scale.view(torch.float8_e8m0fnu))


def _decode_e4m3_bytes(raw: torch.Tensor) -> torch.Tensor:
    """Independent E4M3FN bit decoder producing FP32."""

    bits = raw.to(torch.int16)
    sign = torch.where((bits & 0x80) != 0, -1.0, 1.0)
    exponent = (bits >> 3) & 0x0f
    mantissa = bits & 0x07
    subnormal = mantissa.float() * (2.0 ** -9)
    normal = torch.ldexp(
        1.0 + mantissa.float() / 8.0,
        exponent.to(torch.int32) - 7,
    )
    decoded = torch.where(exponent == 0, subnormal, normal) * sign
    return torch.where(
        (exponent == 15) & (mantissa == 7),
        torch.full_like(decoded, float("nan")), decoded)


def _decode_ue8m0_bytes(raw: torch.Tensor) -> torch.Tensor:
    """Independent UE8M0 bit decoder producing FP32."""

    bits = raw.to(torch.int16)
    decoded = torch.ldexp(
        torch.ones_like(bits, dtype=torch.float32),
        bits.to(torch.int32) - 127,
    )
    return torch.where(bits == 0xff,
                       torch.full_like(decoded, float("nan")), decoded)


def _weight_oracle(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The exact W8A16 weight: one BF16 round after raw-byte decode."""

    n, k = q.shape
    sf = _decode_ue8m0_bytes(scales.view(torch.uint8))
    sf = sf.repeat_interleave(128, 0).repeat_interleave(128, 1)
    decoded = _decode_e4m3_bytes(q.view(torch.uint8)) * sf[:n, :k]
    return decoded.to(torch.bfloat16)


def _reference(x: torch.Tensor, q: torch.Tensor,
               scales: torch.Tensor, groups: int = 1) -> torch.Tensor:
    weight = _weight_oracle(q, scales)
    rows = q.shape[0] // groups
    pieces = [
        x[:, group].float() @
        weight[group * rows:(group + 1) * rows].float().t()
        for group in range(groups)
    ]
    return torch.stack(pieces, dim=1).to(torch.bfloat16)


def _rel_l2(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got.float() - expected.float()).norm()
                 / expected.float().norm().clamp_min(1e-30))


def _assert_reassociated_close(got: torch.Tensor, expected: torch.Tensor,
                               *, limit: float = 3e-3) -> None:
    assert got.dtype is torch.bfloat16
    assert torch.isfinite(got).all()
    error = _rel_l2(got, expected)
    assert error <= limit, f"relative L2 {error:.7g} exceeds {limit}"


def _layer_for(q: torch.Tensor, scales: torch.Tensor, *, groups: int = 1,
               monkeypatch=None, shard_degree: int = 1):
    method = build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)
    layer = torch.nn.Module()
    layer.tp_size = shard_degree
    if shard_degree > 1:
        # What create_weights would have stamped for a column-parallel plane
        # at this degree; the finalize path reads the plan, never tp_size.
        setattr(layer, source_lane._SHARD_ATTR,
                source_lane.ShardPlan(row_degree=1, col_degree=shard_degree))
    layer.register_parameter(
        "weight", torch.nn.Parameter(q, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    if groups > 1:
        layer.is_bmm = True
        layer.bmm_batch_size = groups
        if monkeypatch is not None:
            monkeypatch.setattr(
                dsv4_woa, "install_dsv4_woa_adapter", lambda: None)
    # Mirror vLLM's authoritative ownership edge in addition to the local
    # variable: the weak custom-op registry must never own model storage.
    layer.quant_method = method
    method.process_weights_after_loading(layer)
    return method, layer


def test_expander_is_bit_exact_at_both_partial_block_boundaries():
    q, scales = _raw_planes(257, 385, seed=11)
    got = SOURCE_EXT.fp8_source_expand_bf16(q, scales)
    expected = _weight_oracle(q, scales)
    assert tuple(got.shape) == (257, 385)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_raw_gemv_m1_m2_m4_m8_matches_independent_oracle(m):
    # Partial block128 edges remain supported within the native bridge's
    # eight-element alignment contract.
    q, scales = _raw_planes(264, 392, seed=100 + m)
    x = torch.randn(m, 1, 392, device=DEV, dtype=torch.bfloat16) * 0.125
    got = SOURCE_EXT.fp8_source_gemv(x.contiguous(), q, scales, 1)
    expected = _reference(x, q, scales).reshape(m, 264)
    _assert_reassociated_close(got, expected)


@pytest.mark.parametrize("n,k", [
    # Whole-model (TP=1) planes.
    (8192, 4096),
    (32768, 1024),
    (4096, 8192),
    (2048, 4096),
    (4096, 2048),
    # The per-rank planes a TP=2 serve actually hands the kernel: wq_b
    # (32768,1024) and wo_a (8192,4096) cut on the output axis, wo_b
    # (4096,8192) and down_proj (4096,2048) cut on the input axis, and one
    # shared-expert gate/up role (2048,4096) cut on the output axis. Every
    # local extent is a multiple of the 128 source block, so these are the
    # shapes the shard law admits rather than shapes chosen for coverage.
    (16384, 1024),
    (4096, 4096),
    (4096, 1024),
    (1024, 4096),
])
def test_decode_covers_each_distinct_dsv4_source_shape(n, k):
    q, scales = _raw_planes(n, k, seed=n + k)
    x = torch.randn(1, 1, k, device=DEV, dtype=torch.bfloat16) * 0.03125
    got = SOURCE_EXT.fp8_source_gemv(x, q, scales, 1)
    expected = _reference(x, q, scales).reshape(1, n)
    _assert_reassociated_close(got, expected, limit=4e-3)


def test_low_level_ops_honor_a_nondefault_stream():
    q, scales = _raw_planes(256, 384, seed=31)
    x = torch.randn(4, 1, 384, device=DEV, dtype=torch.bfloat16)
    producer = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    worker.wait_stream(producer)
    with torch.cuda.stream(worker):
        expanded = SOURCE_EXT.fp8_source_expand_bf16(q, scales)
        got = SOURCE_EXT.fp8_source_gemv(x, q, scales, 1)
    producer.wait_stream(worker)
    expected_w = _weight_oracle(q, scales)
    expected_y = _reference(x, q, scales).reshape(4, 256)
    assert torch.equal(expanded.view(torch.int16),
                       expected_w.view(torch.int16))
    _assert_reassociated_close(got, expected_y)


def test_dense_whole_method_decode_and_prefill_keep_activation_bf16():
    q, scales = _raw_planes(256, 256, seed=41)
    q_bits = q.view(torch.uint8).clone()
    scale_bits = scales.view(torch.uint8).clone()
    method, layer = _layer_for(q, scales)

    # Model loading preserves only the exact source planes as resident state.
    assert torch.equal(layer.weight.view(torch.uint8), q_bits)
    assert torch.equal(layer.weight_scale_inv.view(torch.uint8), scale_bits)
    assert layer._fp8_source_resident_bytes == q.numel() + scales.numel()
    assert not [tensor for tensor in layer.parameters()
                if tensor.dtype is torch.bfloat16]

    for m in (1, 8, 9, 33):
        x = torch.randn(m, 256, device=DEV, dtype=torch.bfloat16)
        before = x.view(torch.int16).clone()
        got = method.apply(layer, x)
        expected = _reference(x.view(m, 1, 256), q, scales).reshape(m, 256)
        _assert_reassociated_close(got, expected)
        assert torch.equal(x.view(torch.int16), before)


def test_large_m_transient_is_not_retained(monkeypatch):
    q, scales = _raw_planes(256, 256, seed=43)
    method, layer = _layer_for(q, scales)
    seen = []
    original = ops.fp8_source_expand_bf16

    def tracked(*args):
        transient = original(*args)
        seen.append(weakref.ref(transient))
        return transient

    monkeypatch.setattr(ops, "fp8_source_expand_bf16", tracked)
    output = method.apply(
        layer, torch.randn(9, 256, device=DEV, dtype=torch.bfloat16))
    torch.cuda.synchronize()
    gc.collect()
    assert output.shape == (9, 256)
    assert len(seen) == 1 and seen[0]() is None
    assert not any(t.dtype is torch.bfloat16 for t in layer.parameters())


def test_grouped_whole_method_decode_and_prefill_are_isolated(monkeypatch):
    groups, rows, k = 8, 128, 256
    # This compact case isolates group indexing and leakage. Exact release
    # geometry is exercised below; CPU refusal coverage pins every near miss.
    monkeypatch.setattr(source_lane, "_DSV4_BMM_GROUPS", groups)
    monkeypatch.setattr(source_lane, "_DSV4_BMM_ROWS", rows)
    monkeypatch.setattr(source_lane, "_DSV4_BMM_K", k)
    q, scales = _raw_planes(groups * rows, k, seed=51)
    method, layer = _layer_for(
        q, scales, groups=groups, monkeypatch=monkeypatch)
    assert getattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR) == \
        dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI

    for m in (1, 8, 9, 32):
        x = torch.randn(m, groups, k, device=DEV, dtype=torch.bfloat16)
        got = method.apply(layer, x)
        expected = _reference(x, q, scales, groups=groups)
        _assert_reassociated_close(got, expected)
        assert tuple(got.shape) == (m, groups, rows)

    # One nonzero group cannot leak into any other output group.
    isolated = torch.zeros(1, groups, k, device=DEV, dtype=torch.bfloat16)
    isolated[:, 3] = torch.randn(1, k, device=DEV, dtype=torch.bfloat16)
    got = method.apply(layer, isolated)
    assert torch.count_nonzero(got[:, :3]) == 0
    assert torch.count_nonzero(got[:, 4:]) == 0


@pytest.mark.parametrize("shard_degree", [1, 2, 4])
def test_exact_dsv4_wo_a_geometry_decode_and_prefill(monkeypatch,
                                                    shard_degree):
    """The whole-method path at every qualified shard degree.

    Degree 1 is the whole plane; degrees 2 and 4 are one rank's contiguous
    group slice, which is what the lane finalizes for a TP=2 / TP=4 serve.
    Both the decode (m=1) and prefill (m=9) arms run.
    """

    groups, rows, k = 8, 1024, 4096
    q, scales = _raw_planes(groups * rows, k, seed=61)
    local_groups = groups // shard_degree
    local_rows = local_groups * rows
    q = q[:local_rows].contiguous()
    scales = scales[:local_rows // 128].contiguous()
    method, layer = _layer_for(
        q, scales, groups=local_groups, monkeypatch=monkeypatch,
        shard_degree=shard_degree)
    assert layer._fp8_source_shard_degree == shard_degree
    assert layer._fp8_source_groups == local_groups
    for m in (1, 9):
        x = torch.randn(
            m, local_groups, k, device=DEV, dtype=torch.bfloat16) * 0.125
        got = method.apply(layer, x)
        expected = _reference(x, q, scales, groups=local_groups)
        _assert_reassociated_close(got, expected, limit=4e-3)


# The DSv4 wo_a tolerance: the same 4e-3 the unsharded geometry above uses,
# so a sharded call is held to the qualified call's bar and not a looser one.
_DSV4_WO_A_REL_L2_LIMIT = 4e-3


@pytest.mark.parametrize("shard_degree", [2, 4])
def test_sharded_wo_a_reproduces_the_unsharded_group_slice(shard_degree):
    """Column-sharding DSv4 `wo_a` is the same call over fewer groups.

    vLLM narrows a column-parallel plane at `rank * N_local`, and the wo_a
    plane is group-major with 1024 rows per group, so rank `r` of a degree-`d`
    shard holds exactly groups `[r*G/d, (r+1)*G/d)` of the full G=8 plane.
    This measures the claim the qualification rests on: every rank -- not just
    rank 0, so the group-offset arithmetic is exercised -- reproduces its
    columns of the *full* G=8 call, and agrees with the independent oracle on
    its own slice.  Bitwise on the expand path, since decode is elementwise;
    within the qualified tolerance on the GEMV path, whose K-reduction may
    reassociate.
    """

    groups, rows, k = 8, 1024, 4096
    q, scales = _raw_planes(groups * rows, k, seed=61)
    local_groups = groups // shard_degree
    local_rows = local_groups * rows
    local_scale_rows = local_rows // 128
    x = torch.randn(1, groups, k, device=DEV, dtype=torch.bfloat16) * 0.125

    full_weight = SOURCE_EXT.fp8_source_expand_bf16(q, scales)
    full_out = SOURCE_EXT.fp8_source_gemv(
        x, q, scales, groups).reshape(1, groups, rows)

    for rank in range(shard_degree):
        value_rows = slice(rank * local_rows, (rank + 1) * local_rows)
        # Exactly vLLM's block-scale narrow: start = rank * ceil(N_local/128).
        scale_rows = slice(rank * local_scale_rows,
                           (rank + 1) * local_scale_rows)
        held_groups = slice(rank * local_groups, (rank + 1) * local_groups)
        q_local = q[value_rows].contiguous()
        scales_local = scales[scale_rows].contiguous()
        x_local = x[:, held_groups].contiguous()

        local_weight = SOURCE_EXT.fp8_source_expand_bf16(q_local, scales_local)
        assert torch.equal(local_weight.view(torch.int16),
                           full_weight[value_rows].view(torch.int16)), \
            f"rank {rank} expand differs from the full plane's rows"
        assert torch.equal(
            local_weight.view(torch.int16),
            _weight_oracle(q_local, scales_local).view(torch.int16))

        local_out = SOURCE_EXT.fp8_source_gemv(
            x_local, q_local, scales_local,
            local_groups).reshape(1, local_groups, rows)
        _assert_reassociated_close(
            local_out, _reference(x_local, q_local, scales_local,
                                  groups=local_groups),
            limit=_DSV4_WO_A_REL_L2_LIMIT)
        _assert_reassociated_close(local_out, full_out[:, held_groups],
                                   limit=_DSV4_WO_A_REL_L2_LIMIT)


def test_fullgraph_contains_only_the_outer_source_dispatch_node():
    q, scales = _raw_planes(256, 256, seed=71)
    method, layer = _layer_for(q, scales)
    traced = []

    def backend(graph_module, example_inputs):
        del example_inputs
        traced.append(graph_module)
        return graph_module.forward

    def run(value):
        return method.apply(layer, value)

    compiled = torch.compile(run, backend=backend, fullgraph=True)
    first = torch.randn(2, 256, device=DEV, dtype=torch.bfloat16)
    second = torch.randn(2, 256, device=DEV, dtype=torch.bfloat16)
    _assert_reassociated_close(compiled(first), run(first))
    _assert_reassociated_close(compiled(second), run(second))
    assert len(traced) == 1
    nodes = [node for node in traced[0].graph.nodes
             if node.op == "call_function" and
             "prismaquant" in str(node.target)]
    assert len(nodes) == 1
    assert "fp8_source_linear_forward" in str(nodes[0].target)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_cuda_graph_replay_uses_changed_inputs_and_matches_eager(m):
    q, scales = _raw_planes(256, 256, seed=80 + m)
    method, layer = _layer_for(q, scales)
    static_x = torch.randn(m, 256, device=DEV, dtype=torch.bfloat16)

    # Warm all Python/module-load work before capture.
    method.apply(layer, static_x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, static_x)

    previous = None
    for seed in (901 + m, 1901 + m):
        generator = torch.Generator(device=DEV).manual_seed(seed)
        changed = torch.randn(
            static_x.shape, generator=generator, device=DEV,
            dtype=torch.bfloat16)
        static_x.copy_(changed)
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.clone()
        eager = method.apply(layer, changed)
        assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16))
        if previous is not None:
            assert not torch.equal(replayed.view(torch.int16), previous)
        previous = replayed.view(torch.int16).clone()


def test_low_level_invalid_contracts_fail_closed():
    q, scales = _raw_planes(256, 256, seed=91)
    x = torch.zeros(1, 1, 256, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="BF16 activations"):
        SOURCE_EXT.fp8_source_gemv(x.float(), q, scales, 1)
    with pytest.raises(RuntimeError, match="1 <= M <= 8"):
        SOURCE_EXT.fp8_source_gemv(
            torch.zeros(9, 1, 256, device=DEV, dtype=torch.bfloat16),
            q, scales, 1)
    with pytest.raises(RuntimeError, match="divisible by groups"):
        SOURCE_EXT.fp8_source_gemv(
            torch.zeros(1, 3, 256, device=DEV, dtype=torch.bfloat16),
            q, scales, 3)
    q_bad_k, scales_bad_k = _raw_planes(256, 255, seed=92)
    with pytest.raises(RuntimeError, match="K divisible by 8"):
        SOURCE_EXT.fp8_source_gemv(
            torch.zeros(1, 1, 255, device=DEV, dtype=torch.bfloat16),
            q_bad_k, scales_bad_k, 1)
    q_bad_rows, scales_bad_rows = _raw_planes(258, 256, seed=94)
    with pytest.raises(RuntimeError, match="per-group N divisible by 8"):
        SOURCE_EXT.fp8_source_gemv(
            torch.zeros(1, 3, 256, device=DEV, dtype=torch.bfloat16),
            q_bad_rows, scales_bad_rows, 3)
    with pytest.raises(RuntimeError, match="float8_e4m3fn"):
        SOURCE_EXT.fp8_source_expand_bf16(q.float(), scales)
    with pytest.raises(RuntimeError, match="scale shape"):
        SOURCE_EXT.fp8_source_expand_bf16(q, scales[:-1])
    with pytest.raises(RuntimeError, match="contiguous"):
        SOURCE_EXT.fp8_source_expand_bf16(q.t(), scales)
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        SOURCE_EXT.fp8_source_expand_bf16(q.cpu(), scales.cpu())


def test_method_and_loader_invalid_contracts_fail_closed(monkeypatch):
    q, scales = _raw_planes(256, 256, seed=93)
    method, layer = _layer_for(q, scales)
    with pytest.raises(TypeError, match="preserves BF16"):
        method.apply(layer, torch.zeros(1, 256, device=DEV))
    with pytest.raises(ValueError, match="does not serve biased"):
        method.apply(
            layer, torch.zeros(1, 256, device=DEV, dtype=torch.bfloat16),
            torch.zeros(256, device=DEV, dtype=torch.bfloat16))

    incomplete = types.SimpleNamespace(fp8_source_gemv=lambda *args: None)
    monkeypatch.setattr(
        cuda_ext, "get_fp8_source_w8a16_ext", lambda: incomplete)
    with pytest.raises(cuda_ext.NativeKernelUnavailableError,
                       match="fp8_source_expand_bf16"):
        cuda_ext.require_fp8_source_w8a16_ext(
            "missing-symbol gate", device=DEV)

    stale = types.SimpleNamespace(
        fp8_source_gemv=lambda *args: None,
        fp8_source_expand_bf16=lambda *args: None,
        __gridbook_jit_capability__=(12, 0),
    )
    monkeypatch.setattr(cuda_ext, "get_fp8_source_w8a16_ext", lambda: stale)
    with pytest.raises(cuda_ext.NativeKernelUnavailableError,
                       match="built for compute capability 12.0"):
        cuda_ext.require_fp8_source_w8a16_ext(
            "stale-capability gate", device=DEV)


def test_model_load_refuses_missing_prefill_bridge_and_bad_geometry(
        monkeypatch):
    q, scales = _raw_planes(256, 256, seed=95)
    method = build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(q, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    monkeypatch.setattr(
        cuda_ext, "require_bf16_grouped_ext",
        lambda operation="": (_ for _ in ()).throw(
            cuda_ext.NativeKernelUnavailableError("bridge unavailable")))
    with pytest.raises(cuda_ext.NativeKernelUnavailableError,
                       match="bridge unavailable"):
        method.process_weights_after_loading(layer)

    bad_method = build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)
    bad = torch.nn.Module()
    bad.is_bmm = True
    bad.bmm_batch_size = 3
    bad.tp_size = 1
    bad.register_parameter(
        "weight", torch.nn.Parameter(q, requires_grad=False))
    bad.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    with pytest.raises(ValueError, match="dividing N"):
        bad_method.process_weights_after_loading(bad)

    cpu_method = build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)
    cpu_layer = torch.nn.Module()
    cpu_layer.register_parameter(
        "weight", torch.nn.Parameter(q.cpu(), requires_grad=False))
    cpu_layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(scales.cpu(), requires_grad=False))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        cpu_method.process_weights_after_loading(cpu_layer)


def _time_ms(fn, *, warmup: int = 2, iterations: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def test_reports_non_gating_representative_dsv4_timings():
    """Print release-device microtimings; no promotion threshold is inferred."""

    n, k = 8192, 4096
    q, scales = _raw_planes(n, k, seed=101)
    decode_x = torch.randn(1, 1, k, device=DEV, dtype=torch.bfloat16)
    prefill_x = torch.randn(64, k, device=DEV, dtype=torch.bfloat16)
    ends_dense = torch.tensor([64], device=DEV, dtype=torch.int32)
    grouped_x = torch.randn(64, 8, k, device=DEV, dtype=torch.bfloat16)
    grouped_a = grouped_x.permute(1, 0, 2).contiguous().view(8 * 64, k)
    ends_grouped = torch.arange(
        1, 9, device=DEV, dtype=torch.int32) * 64

    decode_ms = _time_ms(
        lambda: SOURCE_EXT.fp8_source_gemv(decode_x, q, scales, 1))

    def dense_prefill():
        weight = ops.fp8_source_expand_bf16(q, scales)
        return ops.cb_bf16_grouped_mm(
            prefill_x, weight.view(1, n, k), ends_dense, 0)

    def grouped_prefill():
        weight = ops.fp8_source_expand_bf16(q, scales)
        return ops.cb_bf16_grouped_mm(
            grouped_a, weight.view(8, 1024, k), ends_grouped, 0)

    dense_ms = _time_ms(dense_prefill)
    grouped_ms = _time_ms(grouped_prefill)
    print(
        "source_fp8_w8a16_non_gating_timing "
        f"device={torch.cuda.get_device_name()} cc=12.1 "
        f"decode_M1_N8192_K4096_ms={decode_ms:.4f} "
        f"dense_M64_N8192_K4096_ms={dense_ms:.4f} "
        f"wo_a_M64_G8_N1024_K4096_ms={grouped_ms:.4f}",
        flush=True,
    )
    assert decode_ms > 0 and dense_ms > 0 and grouped_ms > 0
