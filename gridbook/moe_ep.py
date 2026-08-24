"""Expert parallelism for stacked CB expert weights (torch-only, no vLLM).

vLLM serves MoE either tensor-parallel (every rank holds every expert, each
expert's rows/columns sharded) or expert-parallel (each rank holds a disjoint
SUBSET of whole experts). ``--enable-expert-parallel`` selects the second:
``FusedMoEParallelConfig.make`` returns ``tp_size=1, ep_size=world,
ep_rank=rank`` for the MoE layers while dense Linears stay tensor-parallel at
the full world size.

Expert parallelism is the only multi-rank MoE mode Gridbook serves, because it
is the only one that does not shard a CB expert. A CB expert stack is a byte
tensor whose last dimension is ``(in/256)·type_size`` — superblock bytes, not
input columns — so a TP row/column split would cut a superblock in half and
there is no partial-superblock decode. Under EP each rank keeps whole experts,
whole superblocks, and the identical per-expert numerics it serves at world
size 1; the only per-rank difference is WHICH experts are resident.

Two mechanisms live here, both pure ``torch`` so they are testable on CPU with
no vLLM import and no GPU (the ``moe_routing`` precedent):

``local_expert_gather_index``
    Load-time. Turns ``layer.expert_map`` into the ``(E_local,)`` index of
    global expert ids this rank owns, ordered by local slot, so a whole-stack
    checkpoint tensor of ``E_global`` experts can be gathered down to this
    rank's ``E_local`` rows by ``w.index_select(0, idx)``. It refuses any map
    that is not a monotone bijection onto ``range(E_local)``.

``remap_local_expert_ids``
    Forward-time, and it must run INSIDE the opaque custom op, never in
    ``apply()`` — see ``ops.cb_moe_forward``. The router emits GLOBAL expert
    ids; this rank's stacks are indexed by LOCAL slot. Remote pairs are rewritten
    to an expert this rank owns and their router weight is set to exactly 0.0,
    so every kernel keeps a dense, static ``(T, k)`` routing tensor and the
    remote pairs contribute an exact ``+0.0`` to the combine.

Why zero-weight aliasing rather than compaction: the alternative is to drop
remote pairs, which makes the pair count data-dependent — unrepresentable under
CUDA-graph capture, which is the shipped decode regime (see ``moe._padded_route``
and gridbook#47). Aliasing is capture-safe: every shape stays static, no host
read occurs, and correctness rests on ``x + 0.0 == x``, an IEEE identity in
every accumulation order. It is exact rather than approximate because
``apply_router_weight_on_input`` is refused outright for CB MoE
(``moe.py`` ``apply()``), so the router weight is ALWAYS applied in the combine
and never folded into the activation before the GEMM.

The cost is arithmetic, not accuracy: a rank still computes ``T·k`` pairs when
only ``T·k/EP`` of them are its own. At decode that is free — the grouped GEMV
is memory-bound on the resident expert bytes and an aliased pair reads a stack
the token already reads. At prefill it is real wasted FLOPs, bounded by
``(EP-1)/EP`` of the routed prefill cost. Compaction on the prefill lanes only,
where shapes are already dynamic, is recorded as v2 debt.
"""
from __future__ import annotations

import torch

__all__ = [
    "ExpertParallelError",
    "ep_shape_note",
    "gather_expert_major",
    "local_expert_gather_index",
    "remap_local_expert_ids",
]


class ExpertParallelError(ValueError):
    """Fail-closed refusal on a CB expert-parallel surface.

    Structured, not prose: every message names the surface, the layer prefix
    and the observed geometry, so an operator can act on it without reading
    Gridbook source.
    """


def local_expert_gather_index(
    expert_map: torch.Tensor | None,
    local_num_experts: int,
    *,
    surface: str,
    prefix: str,
) -> torch.Tensor | None:
    """Global expert ids owned by this rank, ordered by local slot.

    ``expert_map`` is vLLM's ``(E_global,)`` int32 placement tensor: entry ``g``
    is this rank's local slot for global expert ``g``, or ``-1`` when the expert
    lives on another rank (``vllm/model_executor/layers/fused_moe/layer.py``
    ``determine_expert_map``). Returns ``None`` when ``expert_map is None``,
    which is world size 1 — the whole checkpoint stack is this rank's stack and
    no gather happens.

    The returned index is built by ``nonzero`` and reordered by local slot; it
    never assumes the owned ids are a contiguous range. vLLM's ``round_robin``
    placement strategy produces genuinely non-contiguous maps.

    Refuses:

    * a map that is not a bijection onto ``range(local_num_experts)`` — a rank
      that owns a slot twice, or owns fewer/more experts than its stacks hold,
      would load the wrong bytes into the right shape and serve silently wrong;
    * a NON-MONOTONE map (owned global ids not in ascending local-slot order).
      This is not a requirement of any kernel — every CB MoE path is
      self-consistent in local-slot space. It is the law under which this
      lane's exactness evidence was taken: rank-local expert-ascending
      accumulation equals global-expert-ascending accumulation on the same
      subset only when the map preserves order. Both stock placement
      strategies are monotone, and the mechanism that could break it (EPLB) is
      refused by vLLM before Gridbook sees the layer
      (``supports_eplb`` defaults False). A non-monotone map is therefore an
      untested surface, and untested surfaces refuse.
    """
    if expert_map is None:
        return None
    if not isinstance(expert_map, torch.Tensor):
        raise ExpertParallelError(
            f"{prefix}: {surface}: expert_map must be a torch.Tensor, got "
            f"{type(expert_map).__name__}")
    em = expert_map.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    e_global = int(em.numel())
    e_local = int(local_num_experts)
    owned = torch.nonzero(em >= 0, as_tuple=False).reshape(-1)
    slots = em.index_select(0, owned)
    # Explicit device: vLLM builds the model inside a ``torch.device(<gpu>)``
    # context, under which a device-less constructor lands on the GPU while
    # ``em`` was deliberately moved to the CPU — ``torch.equal`` then refuses
    # the cross-device compare and construction dies on every rank
    # (observed on the first two-node DSv4 serve, 2026-08-23).
    expected = torch.arange(e_local, dtype=torch.int64, device="cpu")
    if int(owned.numel()) != e_local or not torch.equal(
            torch.sort(slots).values, expected):
        raise ExpertParallelError(
            f"{prefix}: {surface}: expert_map is not a bijection onto "
            f"range({e_local}) — {int(owned.numel())} of {e_global} global "
            f"experts map to local slots "
            f"{sorted(int(v) for v in slots.tolist())[:8]}"
            f"{'...' if int(slots.numel()) > 8 else ''}, expected each of "
            f"0..{e_local - 1} exactly once. Gridbook serves whole CB expert "
            "stacks under expert parallelism and cannot resolve an ambiguous "
            "placement.")
    if e_local > 1 and not bool(torch.all(slots[1:] > slots[:-1])):
        raise ExpertParallelError(
            f"{prefix}: {surface}: expert_map is not monotone — global expert "
            "ids owned by this rank are not in ascending local-slot order. "
            "Gridbook's CB MoE exactness evidence is taken under a monotone "
            "placement (vLLM's linear and round_robin strategies are both "
            "monotone; EPLB, which is not, is already refused by vLLM). Serve "
            "with a stock placement strategy.")
    return owned.index_select(0, torch.argsort(slots)).contiguous()


def gather_expert_major(param: torch.Tensor,
                        incoming: torch.Tensor) -> torch.Tensor:
    """Gather a whole-stack checkpoint tensor down to this rank's experts.

    Both CB expert loaders — the ``RoutedExperts`` instance wrapper in
    ``moe.py`` and the top-level ``load_weights`` wrapper in
    ``moe_toplevel_loader.py`` — funnel expert-major tensors through here so
    there is exactly ONE gather rule. The rule reads only what
    ``create_weights`` stamped on the destination param, so neither loader has
    to reach back into the module for the placement.

    A tensor is gathered only when the destination carries a gather index AND
    the incoming leading dimension is the GLOBAL expert count. Anything else is
    returned untouched, so the caller's shape check still produces the refusal
    it would have produced at world size 1. Non-expert-major params (the
    per-layer ``(1,)`` input scales) are never stamped and so never gathered.
    """
    idx = getattr(param, "_gridbook_ep_gather", None)
    if idx is None or incoming.ndim == 0:
        return incoming
    e_global = int(getattr(param, "_gridbook_ep_global_experts", 0))
    if int(incoming.shape[0]) != e_global:
        return incoming
    if int(param.shape[0]) == e_global:
        return incoming
    return incoming.index_select(0, idx.to(incoming.device))


def ep_shape_note(param: torch.Tensor) -> str:
    """One clause naming the EP placement, for a loader's shape refusal.

    Empty at world size 1, so refusal text is unchanged on the path every
    shipped artifact takes today.
    """
    idx = getattr(param, "_gridbook_ep_gather", None)
    if idx is None:
        return ""
    e_global = int(getattr(param, "_gridbook_ep_global_experts", 0))
    return (f" — expert parallelism is active: this rank holds "
            f"{int(idx.numel())} of {e_global} experts, and a whole-stack "
            f"tensor of {e_global} experts is gathered by expert_map, so the "
            f"checkpoint leading dimension must be either {e_global} or "
            f"{int(param.shape[0])}")


def remap_local_expert_ids(
    expert_map: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewrite global router ids to local slots, zeroing remote pairs.

    ``topk_ids`` ``(T, k)`` carries GLOBAL expert ids; this rank's stacks are
    indexed 0..``local_num_experts``-1. Every pair this rank owns keeps its
    router weight and gets its local slot. Every pair it does not own is
    aliased to an expert this rank DOES own for that same token — the token's
    smallest local id — and its weight is set to exactly ``0.0``.

    Aliasing to a stack the token already reads is deliberate: the aliased pair
    then re-reads bytes already in cache, and (unlike an arbitrary target) it
    cannot introduce a fresh ``0.0 * inf`` or ``0.0 * NaN``, because the token
    is already reading that expert with a nonzero weight. A token with NO local
    pair at all has no such target and falls back to local expert 0; its whole
    row contributes exactly zero, so this rank adds nothing for it and the
    other ranks supply its output through the stock final all-reduce
    (``vllm/model_executor/layers/fused_moe/runner/moe_runner.py``
    ``_maybe_reduce_final_output``).

    Every operation here is static-shape and free of host reads, so the result
    is CUDA-graph capturable.

    Preconditions (both held by the caller, ``ops.cb_moe_forward``): ids are
    already non-negative — vLLM's ``-1`` routing-padding sentinel is
    neutralised to expert 0 with weight 0 first, and this function composes
    exactly with that (global 0 may be remote here, in which case the pair is
    re-aliased and its already-zero weight is set to zero again) — and ids are
    less than ``expert_map.numel()``.
    """
    shape = topk_ids.shape
    flat = topk_ids.reshape(-1)
    if flat.dtype != torch.long:
        flat = flat.to(torch.long)
    local = expert_map.index_select(0, flat).to(torch.long).view(shape)
    is_local = local >= 0
    e_local = int(local_num_experts)
    # Smallest local id this token routes to, or e_local when it has none.
    sentinel = torch.full_like(local, e_local)
    first = torch.where(is_local, local, sentinel).amin(dim=-1, keepdim=True)
    first = torch.where(first == e_local, torch.zeros_like(first), first)
    local = torch.where(is_local, local, first.expand_as(local))
    # Exactly 0.0, and BEFORE any lower-precision cast downstream, so the
    # combine adds an exact zero in every dtype it may be narrowed to.
    weights = torch.where(is_local, topk_weights,
                          torch.zeros_like(topk_weights))
    return local.to(topk_ids.dtype), weights
