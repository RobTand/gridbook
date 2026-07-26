"""L2-resident scratch policy for the ROUND-4 MoE prefill pipeline
(``PRISMAQUANT_CB_PREFILL=l2_pipeline``).

WHY IT EXISTS. Rounds 1-3 attacked the launch count and the padded-tile
redundancy; on a large-expert MoE the dominant remaining cost is the decoded
weight ROUND-TRIP itself — the stock path writes each expert's ``N_e x K`` e4m3
tile to HBM and the very next kernel reads it straight back (~17 ms/layer of the
~42 ms). Round 4 keeps the decode (bit-identical bytes, the promoted numerics)
but aims the write at a small pair of scratch buffers that a CUDA L2
persisting-access window keeps resident, so the GEMM's read is served from L2
rather than from HBM.

WHY ITS OWN MODULE. ``moe.py`` imports vLLM at module scope, so nothing in it is
importable in the build venv. The parts with the arithmetic and therefore the
risk — how big the window may be, how many experts fit in one scratch fill, when
the whole path must fall through to stock — live here, torch-free and
CUDA-free, exactly as ``moe_routing.py`` and ``moe_autotune.py`` do for R2/R3.

NO SHAPE HEURISTICS. The window cap is DERIVED from what the device reports
(``l2_persisting_max_bytes``, halved because we pin a rotating PAIR) and the
group size is derived from that cap and the artifact's own expert tile size. No
MB constant and no model table appears anywhere on this path; an operator may
override the cap with ``PRISMAQUANT_CB_L2_WINDOW_MB`` for bisection only.
"""
from __future__ import annotations

from typing import NamedTuple

_MB = 1024 * 1024


class L2Plan(NamedTuple):
    """The per-(layer, stage-set) scratch plan.

    ``group_size``  experts decoded per scratch fill (>= 1).
    ``groups``      ``[(e0, e1)]`` contiguous half-open expert ranges.
    ``buffer_bytes``  bytes of ONE scratch half (group_size * expert_bytes).
    ``arena_bytes``   bytes of the whole arena (2 * buffer_bytes) — allocated as
                      ONE contiguous block so a SINGLE pinning window covers
                      both halves (the window is one address range per stream).
    """

    group_size: int
    groups: list
    buffer_bytes: int
    arena_bytes: int


def cb_l2_cap_bytes(persisting_max_bytes, *, env_mb=None,
                    max_window_bytes=None) -> int:
    """Bytes allowed in ONE scratch half.

    Derived, never guessed: the device's persisting-L2 capacity halved (we pin a
    rotating pair inside one window, so each half may claim at most half of it),
    intersected with the driver's maximum window extent if the extension reports
    one. ``env_mb`` (``PRISMAQUANT_CB_L2_WINDOW_MB``) is an operator override of
    the SAME quantity — a per-half cap — and can only lower the cap it replaces
    when the device number is smaller, so a too-large override can never make us
    ask for a window the hardware cannot back.

    Returns 0 when nothing usable is reported, which the caller reads as "no L2
    lever available".
    """
    caps = []
    if persisting_max_bytes:
        caps.append(int(persisting_max_bytes) // 2)
    if max_window_bytes:
        caps.append(int(max_window_bytes) // 2)
    if env_mb is not None:
        try:
            env_bytes = int(float(env_mb) * _MB)
        except (TypeError, ValueError):
            env_bytes = 0
        if env_bytes <= 0:
            return 0
        caps.append(env_bytes)
    if not caps:
        return 0
    return max(0, min(caps))


def cb_l2_group_size(expert_bytes: int, cap_bytes: int,
                     force: int | None = None) -> int:
    """Experts per scratch fill = ``floor(cap / expert_bytes)``.

    0 means FALL THROUGH: either the device reported no usable window, or the
    LARGEST expert tile alone exceeds the cap, in which case no rotation of this
    pair can ever keep a tile resident and the honest answer is to run the stock
    path rather than pay the pipeline's bookkeeping for no L2 benefit.

    Small experts are the opposite regime: a single tile leaves most of the
    window idle and one decode launch per expert is pure overhead, so the group
    grows to fill the window. Both regimes come out of the same division — the
    device's capacity and the artifact's own shapes — with no branch on model.
    """
    if force is not None:                 # bisection override (tests, A/B)
        return max(0, int(force))
    if expert_bytes <= 0 or cap_bytes <= 0 or expert_bytes > cap_bytes:
        return 0
    return int(cap_bytes // expert_bytes)


def cb_l2_plan(num_experts: int, expert_bytes: int, cap_bytes: int,
               force_group: int | None = None):
    """The full plan, or ``None`` when the path must fall through to stock.

    ``expert_bytes`` must be the LARGEST expert tile across the projection
    stages the plan will serve (w13's ``2*inter*hidden`` and w2's
    ``hidden*inter``): one arena is allocated and pinned per layer and reused by
    both stages, so it has to hold the worst case, and the group size that
    follows is the one both stages can honour.
    """
    gs = cb_l2_group_size(expert_bytes, cap_bytes, force_group)
    if gs <= 0 or num_experts <= 0:
        return None
    gs = min(gs, int(num_experts))
    groups = [(g0, min(int(num_experts), g0 + gs))
              for g0 in range(0, int(num_experts), gs)]
    buf = gs * int(expert_bytes)
    return L2Plan(group_size=gs, groups=groups, buffer_bytes=buf,
                  arena_bytes=2 * buf)


def cb_l2_live_groups(groups, bounds):
    """Drop expert groups no token routed to, using ONLY the E+1 segment
    offsets the routing already fetched (R1's single host sync) — so the skip
    costs no additional device read.

    Returns ``[(e0, e1, p0, p1)]``. A group is decoded ONLY if it has rows: at
    E=256 with a narrow top-k most groups are empty, and decoding them would
    burn both the launch and the window's residency on weights no GEMM reads.
    Dropping whole units (rather than individual experts inside a group) keeps
    the buffer rotation a simple alternation over the surviving units, which is
    what the event structure is proved against.
    """
    live = []
    for e0, e1 in groups:
        p0, p1 = bounds[e0], bounds[e1]
        if p1 > p0:
            live.append((e0, e1, p0, p1))
    return live
