"""``gridbook.moe_ep`` — the two expert-parallel mechanisms, on CPU.

``moe_ep`` is deliberately torch-only (the ``moe_routing`` precedent), so the
load-time gather rule and the forward-time id remap are testable with no vLLM
import and no GPU. The kernel-level consequences — that a remote pair really
contributes nothing to a CB MoE forward — are proved separately, on real
stacks, in ``tests/test_moe_ep_exactness.py``.

Run: ``python -m pytest tests/test_moe_ep.py -q``.
"""
from __future__ import annotations

import pytest
import torch

from gridbook.moe_ep import (
    ExpertParallelError,
    ep_shape_note,
    gather_expert_major,
    local_expert_gather_index,
    remap_local_expert_ids,
)


def _linear_map(e_global: int, ep_size: int, rank: int) -> torch.Tensor:
    """vLLM's ``linear`` placement: contiguous blocks of experts per rank."""
    per = e_global // ep_size
    em = torch.full((e_global,), -1, dtype=torch.int32)
    lo = rank * per
    em[lo:lo + per] = torch.arange(per, dtype=torch.int32)
    return em


def _round_robin_map(e_global: int, ep_size: int, rank: int) -> torch.Tensor:
    """vLLM's ``round_robin`` placement: interleaved, non-contiguous."""
    em = torch.full((e_global,), -1, dtype=torch.int32)
    owned = torch.arange(rank, e_global, ep_size)
    em[owned] = torch.arange(owned.numel(), dtype=torch.int32)
    return em


def _stamp(param: torch.Tensor, idx, e_global: int) -> torch.Tensor:
    param._gridbook_ep_gather = idx
    param._gridbook_ep_global_experts = e_global
    return param


# --- the gather index ---------------------------------------------------------


def test_world_size_one_has_no_map_and_no_gather():
    assert local_expert_gather_index(
        None, 8, surface="s", prefix="p") is None


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_contiguous_placement_yields_that_ranks_block(rank):
    em = _linear_map(32, 4, rank)
    idx = local_expert_gather_index(em, 8, surface="s", prefix="p")
    assert torch.equal(idx, torch.arange(rank * 8, rank * 8 + 8))


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_interleaved_placement_is_gathered_generally(rank):
    """The rule is nonzero-ordered-by-local-slot, never a contiguous range."""
    em = _round_robin_map(32, 4, rank)
    idx = local_expert_gather_index(em, 8, surface="s", prefix="p")
    assert torch.equal(idx, torch.arange(rank, 32, 4))
    # Local slot l really is global id idx[l].
    for slot, gid in enumerate(idx.tolist()):
        assert int(em[gid]) == slot


def test_every_global_expert_is_owned_exactly_once_across_ranks():
    for maker in (_linear_map, _round_robin_map):
        seen = torch.cat([
            local_expert_gather_index(maker(32, 4, r), 8,
                                      surface="s", prefix="p")
            for r in range(4)])
        assert torch.equal(torch.sort(seen).values, torch.arange(32))


def test_a_map_that_owns_the_wrong_count_refuses():
    em = _linear_map(32, 4, 1)
    with pytest.raises(ExpertParallelError, match="bijection"):
        local_expert_gather_index(em, 7, surface="s", prefix="lay.0")


def test_a_map_with_a_duplicated_local_slot_refuses():
    em = _linear_map(32, 4, 0)
    em[3] = 0                       # slot 0 claimed twice, slot 3 vacated
    with pytest.raises(ExpertParallelError, match="bijection"):
        local_expert_gather_index(em, 8, surface="s", prefix="lay.0")


def test_a_map_with_an_out_of_range_local_slot_refuses():
    em = _linear_map(32, 4, 0)
    em[0] = 99
    with pytest.raises(ExpertParallelError, match="bijection"):
        local_expert_gather_index(em, 8, surface="s", prefix="lay.0")


def test_a_non_monotone_map_refuses():
    """Not a kernel requirement — the law the exactness evidence assumes."""
    em = _linear_map(32, 4, 0)
    em[0], em[1] = 1, 0
    with pytest.raises(ExpertParallelError, match="monotone"):
        local_expert_gather_index(em, 8, surface="s", prefix="lay.0")


def test_refusals_name_the_surface_and_the_layer():
    em = _linear_map(32, 4, 0)
    em[0], em[1] = 1, 0
    with pytest.raises(ExpertParallelError) as excinfo:
        local_expert_gather_index(
            em, 8, surface="CB MoE expert stacks", prefix="model.layers.3.mlp")
    message = str(excinfo.value)
    assert "model.layers.3.mlp" in message
    assert "CB MoE expert stacks" in message


def test_a_single_local_expert_is_trivially_monotone():
    em = torch.full((8,), -1, dtype=torch.int32)
    em[5] = 0
    idx = local_expert_gather_index(em, 1, surface="s", prefix="p")
    assert torch.equal(idx, torch.tensor([5]))


# --- the loader gather --------------------------------------------------------


def test_gather_takes_this_ranks_rows_from_a_whole_stack():
    idx = local_expert_gather_index(_round_robin_map(8, 2, 1), 4,
                                    surface="s", prefix="p")
    param = _stamp(torch.zeros(4, 6, 10, dtype=torch.uint8), idx, 8)
    incoming = torch.arange(8 * 6 * 10, dtype=torch.uint8).reshape(8, 6, 10)
    got = gather_expert_major(param, incoming)
    assert got.shape == param.shape
    for slot, gid in enumerate([1, 3, 5, 7]):
        assert torch.equal(got[slot], incoming[gid])


def test_gather_is_a_no_op_at_world_size_one():
    param = torch.zeros(8, 6, 10, dtype=torch.uint8)     # never stamped
    incoming = torch.ones(8, 6, 10, dtype=torch.uint8)
    assert gather_expert_major(param, incoming) is incoming


def test_gather_leaves_a_non_expert_major_tensor_alone():
    """The per-layer ``(1,)`` input scales are never stamped, so never cut."""
    idx = local_expert_gather_index(_linear_map(8, 2, 0), 4,
                                    surface="s", prefix="p")
    scale = torch.zeros(1, dtype=torch.float32)          # unstamped by design
    incoming = torch.tensor([0.25], dtype=torch.float32)
    assert gather_expert_major(scale, incoming) is incoming
    # Even if something did stamp it, a leading dim of 1 != E_global is left
    # for the caller's shape check rather than silently indexed.
    _stamp(scale, idx, 8)
    assert gather_expert_major(scale, incoming) is incoming


def test_an_already_local_stack_is_not_gathered_twice():
    idx = local_expert_gather_index(_linear_map(8, 2, 0), 4,
                                    surface="s", prefix="p")
    param = _stamp(torch.zeros(4, 6, 10, dtype=torch.uint8), idx, 8)
    incoming = torch.ones(4, 6, 10, dtype=torch.uint8)
    assert gather_expert_major(param, incoming) is incoming


def test_a_wrong_sized_stack_is_left_for_the_shape_refusal():
    idx = local_expert_gather_index(_linear_map(8, 2, 0), 4,
                                    surface="s", prefix="p")
    param = _stamp(torch.zeros(4, 6, 10, dtype=torch.uint8), idx, 8)
    incoming = torch.ones(6, 6, 10, dtype=torch.uint8)   # neither 8 nor 4
    assert gather_expert_major(param, incoming) is incoming


def test_shape_note_is_empty_at_world_size_one_and_names_ep_above_it():
    plain = torch.zeros(8, 4, dtype=torch.uint8)
    assert ep_shape_note(plain) == ""
    idx = local_expert_gather_index(_linear_map(8, 2, 0), 4,
                                    surface="s", prefix="p")
    param = _stamp(torch.zeros(4, 4, dtype=torch.uint8), idx, 8)
    note = ep_shape_note(param)
    assert "expert parallelism" in note and "4 of 8" in note


# --- the forward-time remap ---------------------------------------------------


def _routing(tokens: int, e_global: int, topk: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([
        torch.randperm(e_global, generator=g)[:topk] for _ in range(tokens)
    ]).to(torch.int32)
    w = torch.rand(tokens, topk, generator=g, dtype=torch.float32)
    return ids, w / w.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("maker", [_linear_map, _round_robin_map])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_local_pairs_keep_their_weight_and_take_their_local_slot(maker, rank):
    em = maker(32, 4, rank)
    ids, w = _routing(24, 32, 8, seed=rank)
    local, wl = remap_local_expert_ids(em, ids, w, 8)
    owned = em[ids.to(torch.long)] >= 0
    assert torch.equal(local[owned], em[ids.to(torch.long)][owned].long())
    assert torch.equal(wl[owned], w[owned])


@pytest.mark.parametrize("maker", [_linear_map, _round_robin_map])
def test_remote_pairs_are_exactly_zero_weighted(maker):
    em = maker(32, 4, 2)
    ids, w = _routing(24, 32, 8, seed=5)
    _, wl = remap_local_expert_ids(em, ids, w, 8)
    remote = em[ids.to(torch.long)] < 0
    assert remote.any(), "test routing must contain remote pairs"
    assert torch.equal(wl[remote], torch.zeros_like(wl[remote]))
    # Exactly zero, in the bit pattern, and positive — not -0.0, not denormal.
    assert (wl[remote].view(torch.int32) == 0).all()


def test_every_remapped_id_is_a_valid_local_slot():
    for rank in range(4):
        em = _round_robin_map(32, 4, rank)
        ids, w = _routing(40, 32, 8, seed=rank)
        local, _ = remap_local_expert_ids(em, ids, w, 8)
        assert int(local.min()) >= 0 and int(local.max()) < 8


def test_a_remote_pair_aliases_to_the_tokens_own_smallest_local_expert():
    em = _linear_map(8, 2, 0)               # rank 0 owns globals 0..3
    ids = torch.tensor([[6, 2, 3, 7]], dtype=torch.int32)
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    local, wl = remap_local_expert_ids(em, ids, w, 4)
    # local pairs: global 2 -> slot 2, global 3 -> slot 3; smallest is 2.
    assert local.tolist() == [[2, 2, 3, 2]]
    # Surviving weights are the ORIGINAL bits, not a rounded copy.
    assert torch.equal(wl, torch.tensor([[0.0, w[0, 1], w[0, 2], 0.0]]))


def test_a_token_with_no_local_pair_falls_back_to_expert_zero():
    em = _linear_map(8, 2, 0)
    ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.int32)
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    local, wl = remap_local_expert_ids(em, ids, w, 4)
    assert local.tolist() == [[0, 0, 0, 0]]
    assert wl.tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_the_padding_sentinel_composes_with_the_remap():
    """vLLM's -1 routing padding is neutralised to (expert 0, weight 0) first.

    On a rank where global expert 0 is REMOTE the pair is re-aliased and its
    weight zeroed again; the composition must stay exactly zero either way.
    """
    ids = torch.tensor([[0, 5, 6, 7]], dtype=torch.int32)   # pair 0 = padding
    w = torch.tensor([[0.0, 0.5, 0.3, 0.2]])
    for rank in (0, 1):
        em = _linear_map(8, 2, rank)
        local, wl = remap_local_expert_ids(em, ids, w, 4)
        assert float(wl[0, 0]) == 0.0
        assert 0 <= int(local[0, 0]) < 4


def test_shapes_and_dtypes_survive_the_remap():
    em = _round_robin_map(32, 4, 1)
    ids, w = _routing(13, 32, 6, seed=9)
    local, wl = remap_local_expert_ids(em, ids, w, 8)
    assert local.shape == ids.shape and wl.shape == w.shape
    assert local.dtype == ids.dtype and wl.dtype == w.dtype


def test_the_partition_is_exact_across_ranks():
    """Every (token, pair) is live on exactly one rank, at its own weight."""
    e_global, ep = 32, 4
    ids, w = _routing(40, e_global, 8, seed=3)
    live = torch.zeros_like(w)
    for rank in range(ep):
        _, wl = remap_local_expert_ids(
            _round_robin_map(e_global, ep, rank), ids, w, e_global // ep)
        live += (wl != 0).float() * wl
        # A pair is live on this rank iff this rank owns it.
        owned = _round_robin_map(e_global, ep, rank)[ids.to(torch.long)] >= 0
        assert torch.equal(wl != 0, owned & (w != 0))
    assert torch.equal(live, w)


def test_the_remap_is_shape_static_and_free_of_host_reads():
    """Capture safety, checked rather than asserted.

    Under ``FakeTensorMode`` a tensor has metadata but NO data, so any op that
    would read a value on the host — ``.item()``, ``nonzero``, ``bincount``,
    a data-dependent size — raises instead of silently working. Completing the
    remap with the output shapes intact is therefore evidence the path is
    capturable. The real CUDA-graph capture is exercised in
    ``tests/test_moe_ep_exactness.py``; this runs on CPU with no GPU.
    """
    em = _round_robin_map(32, 4, 0)
    ids, w = _routing(16, 32, 8, seed=1)
    with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
        fem = mode.from_tensor(em)
        fids = mode.from_tensor(ids)
        fw = mode.from_tensor(w)
        local, wl = remap_local_expert_ids(fem, fids, fw, 8)
        assert tuple(local.shape) == tuple(ids.shape)
        assert tuple(wl.shape) == tuple(w.shape)
        assert local.dtype == ids.dtype and wl.dtype == w.dtype


def test_a_data_dependent_alternative_would_fail_that_check():
    """Negative control: prove the fake-tensor probe can actually fail.

    Compaction — dropping remote pairs instead of zero-weighting them — is the
    design this lane rejected. It cannot survive the same probe, which is why
    the passing test above is evidence and not decoration.
    """
    em = _round_robin_map(32, 4, 0)
    ids, _ = _routing(16, 32, 8, seed=1)
    with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
        fem = mode.from_tensor(em)
        fids = mode.from_tensor(ids)
        with pytest.raises(Exception):
            keep = torch.nonzero(
                fem.index_select(0, fids.reshape(-1).long()) >= 0)
            _ = int(keep.numel())


# --- default-device context ---------------------------------------------------
# vLLM constructs the model inside ``torch.device(<gpu>)``; every device-less
# constructor in a construction-time path then lands on the GPU. The gather
# index moves the map to the CPU on purpose, so its comparison tensor must be
# built on the CPU explicitly or ``torch.equal`` refuses the cross-device
# compare. The first two-node DSv4 serve died exactly there (2026-08-23).

def _contiguous_rank1_of_2(e_global: int = 8) -> torch.Tensor:
    em = torch.full((e_global,), -1, dtype=torch.int64)
    em[e_global // 2:] = torch.arange(e_global // 2, dtype=torch.int64)
    return em


def test_gather_index_is_built_under_a_meta_default_device():
    em = _contiguous_rank1_of_2()
    plain = local_expert_gather_index(em, 4, surface="s", prefix="p")
    with torch.device("meta"):
        under_context = local_expert_gather_index(
            em, 4, surface="s", prefix="p")
    assert under_context.device.type == "cpu"
    assert torch.equal(under_context, plain)


def test_gather_index_refusals_still_fire_under_a_default_device_context():
    em = _contiguous_rank1_of_2()
    with torch.device("meta"), pytest.raises(ExpertParallelError,
                                             match="not a bijection"):
        local_expert_gather_index(em, 3, surface="s", prefix="lay.0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gather_index_is_built_under_a_cuda_default_device_from_a_cuda_map():
    em = _contiguous_rank1_of_2().cuda()
    plain = local_expert_gather_index(em.cpu(), 4, surface="s", prefix="p")
    with torch.device("cuda"):
        under_context = local_expert_gather_index(
            em, 4, surface="s", prefix="p")
    assert under_context.device.type == "cpu"
    assert torch.equal(under_context, plain)
