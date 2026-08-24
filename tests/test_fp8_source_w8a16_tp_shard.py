"""Rank-summed parity for source-FP8 W8A16 planes sharded at DSv4 shapes.

One process, no collectives, no second device: vLLM's OWN narrowing
arithmetic is driven once per simulated rank with the global rank/world-size
patched, and the rank-local planes are then combined the way the collective
would.  That proves two things and deliberately not a third:

1. **Byte placement.**  Every rank's value plane AND its UE8M0 block-scale
   plane are the exact contiguous slices of the checkpoint tensors, for
   column-parallel, row-parallel and merged-column planes at the real DSv4
   TP=2 shapes.  The scale plane is the interesting half: vLLM narrows it
   with the SAME arithmetic as the value plane over the block grid
   (``BlockQuantScaleParameter`` is ``_ColumnvLLMParameter, RowvLLMParameter``
   with ``pass``), converting element counts to block counts by CEIL division
   (``adjust_block_scale_shard``), so the narrow start is
   ``rank * ceil(local / 128)`` -- correct only where the local extent is a
   whole multiple of 128.

2. **Arithmetic identity.**  Column shards concatenate to the whole-tensor
   product BITWISE; row shards SUM to it within the reassociation tolerance
   this lane already uses, because a row split changes only the order of a
   reduction.

It does NOT prove a distributed run: no engine, no collectives, no second
device, and no kernel -- the products here are fp32 reference matmuls over
the decoded planes, which is what isolates the SHARD LAW from the kernel.

Requires the real vLLM parameter classes; skipped where vLLM is absent (the
authoritative environment is the pinned container, not a host venv).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_vllm_parameter = pytest.importorskip("vllm.model_executor.parameter")
_vllm_linear = pytest.importorskip("vllm.model_executor.layers.linear")

from gridbook import fp8_source_w8a16 as lane  # noqa: E402

BLOCK = 128

#: Reassociation tolerance, the same figure the lane's CUDA parity uses for
#: DSv4-sized reductions.
_REL_L2_LIMIT = 4e-3


# --- an independent decoder --------------------------------------------------
#
# Deliberately restates the wire format rather than importing the lane's
# helpers: a shard law proven against the lane's own decoder would only prove
# self-consistency.


def _decode(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """E4M3 values times their UE8M0 block-128 exponents, in fp32."""

    n, k = q.shape
    values = q.to(torch.float32)
    exponents = scales.view(torch.uint8).to(torch.int32) - 127
    factors = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32),
                          exponents)
    full = factors.repeat_interleave(BLOCK, dim=0).repeat_interleave(
        BLOCK, dim=1)
    return values * full[:n, :k]


def _synth(n: int, k: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(n, k, generator=generator) * 0.05
    q = values.to(torch.float8_e4m3fn)
    # UE8M0 exponents around 1.0, so the decode spans several binades.
    raw = torch.randint(120, 134, ((n + BLOCK - 1) // BLOCK,
                                   (k + BLOCK - 1) // BLOCK),
                        generator=generator, dtype=torch.uint8)
    scales = raw.view(torch.float8_e8m0fnu)
    return q, scales


def _rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    denominator = want.to(torch.float64).norm()
    if float(denominator) == 0.0:
        return float((got.to(torch.float64) - want.to(torch.float64)).norm())
    return float((got.to(torch.float64) - want.to(torch.float64)).norm()
                 / denominator)


# --- driving the lane's construction at a simulated rank ---------------------


def _method():
    return lane.build_fp8_source_w8a16_method(lane.WIRE_FP8_BLOCK128)


def _built(monkeypatch, *, rank, world_size, input_size,
           input_size_per_partition, output_size, output_partition_sizes,
           prefix="model.layers.0.unit"):
    """Construct this lane's parameters as rank *rank* of *world_size*."""

    monkeypatch.setattr(_vllm_parameter, "get_tensor_model_parallel_rank",
                        lambda: rank)
    monkeypatch.setattr(
        _vllm_parameter, "get_tensor_model_parallel_world_size",
        lambda: world_size)
    layer = torch.nn.Module()
    layer.prefix = prefix
    # vLLM stamps the world size on EVERY layer, sharded or not; the lane must
    # ignore it and read the structural sizes instead.
    layer.tp_size = world_size
    _method().create_weights(
        layer,
        input_size_per_partition,
        list(output_partition_sizes),
        input_size,
        output_size,
        torch.bfloat16,
        weight_loader=lambda param, loaded, *args, **kwargs: None,
    )
    return layer


def _load_column(layer, q, scales):
    layer.weight.load_column_parallel_weight(q)
    layer.weight_scale_inv.load_column_parallel_weight(scales)


def _load_row(layer, q, scales):
    layer.weight.load_row_parallel_weight(q)
    layer.weight_scale_inv.load_row_parallel_weight(scales)


def _load_merged(layer, q, scales, *, full_role_sizes, world_size):
    """vLLM's merged-column v1 path, applied by hand.

    Restated from ``MergedColumnParallelLinear.weight_loader`` (linear.py
    :786-800): the per-role offset and size are divided by the world size
    first, then converted to block units by ``adjust_block_scale_shard`` for
    the scale plane, and the parameter narrows the loaded role at
    ``tp_rank * shard_size``.
    """

    offset = 0
    for role_size in full_role_sizes:
        local_offset = offset // world_size
        local_size = role_size // world_size
        layer.weight.load_merged_column_weight(
            loaded_weight=q.narrow(0, offset, role_size),
            shard_offset=local_offset,
            shard_size=local_size,
        )
        scale_size, scale_offset = _vllm_linear.adjust_block_scale_shard(
            layer.weight_block_size, local_size, local_offset)
        layer.weight_scale_inv.load_merged_column_weight(
            loaded_weight=scales.narrow(
                0, offset // BLOCK, role_size // BLOCK),
            shard_offset=scale_offset,
            shard_size=scale_size,
        )
        offset += role_size


# --- 1. byte placement -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "full_n", "full_k"),
    (
        ("attn.wq_b", 32768, 1024),
        ("attn.wo_a", 8192, 4096),
    ),
)
def test_column_shard_narrows_both_planes_to_exact_slices(
        monkeypatch, name, full_n, full_k):
    world_size = 2
    q, scales = _synth(full_n, full_k, seed=11)
    local_n = full_n // world_size
    for rank in range(world_size):
        layer = _built(monkeypatch, rank=rank, world_size=world_size,
                       input_size=full_k, input_size_per_partition=full_k,
                       output_size=full_n, output_partition_sizes=[local_n],
                       prefix=name)
        _load_column(layer, q, scales)

        assert torch.equal(
            layer.weight.data.view(torch.uint8),
            q.narrow(0, rank * local_n, local_n).view(torch.uint8))
        blocks = local_n // BLOCK
        assert torch.equal(
            layer.weight_scale_inv.data.view(torch.uint8),
            scales.narrow(0, rank * blocks, blocks).view(torch.uint8))


@pytest.mark.parametrize(
    ("name", "full_n", "full_k"),
    (
        ("self_attn.wo_b", 4096, 8192),
        ("shared_experts.down_proj", 4096, 2048),
    ),
)
def test_row_shard_narrows_both_planes_to_exact_slices(
        monkeypatch, name, full_n, full_k):
    world_size = 2
    q, scales = _synth(full_n, full_k, seed=12)
    local_k = full_k // world_size
    for rank in range(world_size):
        layer = _built(monkeypatch, rank=rank, world_size=world_size,
                       input_size=full_k, input_size_per_partition=local_k,
                       output_size=full_n, output_partition_sizes=[full_n],
                       prefix=name)
        _load_row(layer, q, scales)

        assert torch.equal(
            layer.weight.data.view(torch.uint8),
            q.narrow(1, rank * local_k, local_k).view(torch.uint8))
        blocks = local_k // BLOCK
        assert torch.equal(
            layer.weight_scale_inv.data.view(torch.uint8),
            scales.narrow(1, rank * blocks, blocks).view(torch.uint8))


def test_merged_column_shard_places_every_role_at_exact_block_offsets(
        monkeypatch):
    """DSv4's shared-experts gate_up_proj: two 2048-wide roles at TP=2."""

    world_size = 2
    full_roles = [2048, 2048]
    full_n = sum(full_roles)
    full_k = 4096
    q, scales = _synth(full_n, full_k, seed=13)
    local_roles = [size // world_size for size in full_roles]

    for rank in range(world_size):
        layer = _built(
            monkeypatch, rank=rank, world_size=world_size,
            input_size=full_k, input_size_per_partition=full_k,
            output_size=full_n, output_partition_sizes=local_roles,
            prefix="mlp.shared_experts.gate_up_proj")
        _load_merged(layer, q, scales, full_role_sizes=full_roles,
                     world_size=world_size)

        expected = torch.cat([
            q.narrow(0, offset + rank * local, local)
            for offset, local in zip((0, full_roles[0]), local_roles)
        ], dim=0)
        assert torch.equal(layer.weight.data.view(torch.uint8),
                           expected.view(torch.uint8))
        expected_scales = torch.cat([
            scales.narrow(0, (offset + rank * local) // BLOCK, local // BLOCK)
            for offset, local in zip((0, full_roles[0]), local_roles)
        ], dim=0)
        assert torch.equal(layer.weight_scale_inv.data.view(torch.uint8),
                           expected_scales.view(torch.uint8))


def test_the_narrow_start_is_what_the_128_law_protects():
    """Why the law is 128 and not "any even split", stated as arithmetic.

    ``load_column_parallel_weight`` narrows the SCALE plane at
    ``tp_rank * ceil(local_n / 128)``.  Where the local extent is a whole
    multiple of 128 that start is exactly the rank's first block; one element
    below it, rank 1 re-reads a block rank 0 already owns.  The lane refuses
    the second case at construction, which is what this arithmetic justifies.
    """
    world_size = 2
    for local_n, aligned in ((2048, True), (2048 - 8, False)):
        full_blocks = -(-(local_n * world_size) // BLOCK)
        local_blocks = -(-local_n // BLOCK)
        starts = [rank * local_blocks for rank in range(world_size)]
        tiles_exactly = (starts[-1] + local_blocks == full_blocks
                         and local_n % BLOCK == 0)
        assert tiles_exactly is aligned


# --- 2. arithmetic identity --------------------------------------------------


def test_column_shards_concatenate_bitwise_to_the_whole_product(monkeypatch):
    world_size = 2
    full_n, full_k = 8192, 4096
    q, scales = _synth(full_n, full_k, seed=21)
    x = torch.randn(4, full_k, dtype=torch.float32)
    whole = x @ _decode(q, scales).t()

    shards = []
    local_n = full_n // world_size
    for rank in range(world_size):
        layer = _built(monkeypatch, rank=rank, world_size=world_size,
                       input_size=full_k, input_size_per_partition=full_k,
                       output_size=full_n, output_partition_sizes=[local_n])
        _load_column(layer, q, scales)
        shards.append(x @ _decode(layer.weight.data,
                                  layer.weight_scale_inv.data).t())

    # Each output column is the same reduction over the same K, so the
    # concatenation is EXACT, not merely close.
    assert torch.equal(torch.cat(shards, dim=1), whole)


def test_row_shards_sum_to_the_whole_product_within_tolerance(monkeypatch):
    world_size = 2
    full_n, full_k = 4096, 8192
    q, scales = _synth(full_n, full_k, seed=22)
    x = torch.randn(4, full_k, dtype=torch.float32)
    whole = x @ _decode(q, scales).t()

    partial = torch.zeros_like(whole)
    local_k = full_k // world_size
    for rank in range(world_size):
        layer = _built(monkeypatch, rank=rank, world_size=world_size,
                       input_size=full_k, input_size_per_partition=local_k,
                       output_size=full_n, output_partition_sizes=[full_n])
        _load_row(layer, q, scales)
        x_local = x.narrow(1, rank * local_k, local_k)
        partial = partial + x_local @ _decode(
            layer.weight.data, layer.weight_scale_inv.data).t()

    # A row split reassociates one reduction; nothing else changes.
    assert _rel_l2(partial, whole) < _REL_L2_LIMIT


def test_row_shard_bias_is_not_double_counted_because_bias_is_refused():
    """The all-reduce hazard this lane cannot have.

    A row-parallel Linear adds its bias AFTER the reduce; a method that added
    its own would count it once per rank.  This lane refuses biased Linears
    outright, so the hazard is structurally absent rather than handled.
    """
    method = _method()
    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="does not serve biased linears"):
        method.apply(layer, torch.zeros(1, 8, dtype=torch.bfloat16),
                     bias=torch.zeros(8))


# --- 3. the refusals, through the real parameter surface ---------------------


def test_a_misaligned_column_shard_refuses_before_any_parameter_exists(
        monkeypatch):
    with pytest.raises(lane.ShardAlignmentError, match="column-parallel"):
        _built(monkeypatch, rank=0, world_size=2,
               input_size=4096, input_size_per_partition=4096,
               output_size=2 * (2048 - 8), output_partition_sizes=[2048 - 8])


def test_a_misaligned_row_shard_refuses_before_any_parameter_exists(
        monkeypatch):
    with pytest.raises(lane.ShardAlignmentError, match="row-parallel"):
        _built(monkeypatch, rank=0, world_size=2,
               input_size=2 * (2048 - 8),
               input_size_per_partition=2048 - 8,
               output_size=4096, output_partition_sizes=[4096])


def test_a_replicated_plane_constructs_at_any_world_size(monkeypatch):
    """DSv4's fused wqa_wkv (disable_tp) and indexer wq_b (Replicated).

    Both report ``tp_size = world_size`` while holding whole tensors, so a
    lane that gated on ``tp_size`` would refuse them.
    """
    for world_size in (2, 4, 8):
        layer = _built(monkeypatch, rank=0, world_size=world_size,
                       input_size=4096, input_size_per_partition=4096,
                       output_size=1536, output_partition_sizes=[1024, 512],
                       prefix="attn.fused_wqa_wkv")
        assert layer.tp_size == world_size
        assert getattr(layer, lane._SHARD_ATTR).degree == 1
        assert tuple(layer.weight.shape) == (1536, 4096)
