"""K1.5: swizzle-group packing applied WITHIN each expert chunk.

``moe._apply_prefill_native_bf16_sm120`` used to offer the packed expert
order only when ONE decode chunk covered every expert (``chunk >= E``),
because the chunk loop slices blocks as ``block_off[c0]..block_off[c1]``
against a weight stack expanded in EXPERT-ID order for ``[c0, c1)``: a
globally packed order would put foreign experts' blocks inside a chunk's
range. That gate left the production multi-chunk shapes (E=128 at chunks=2)
with the gather but not the 13.9-16.9% tile-order win measured at T=512
(docs/BENCHMARKS.md, 2026-08-02 section).

The fix packs each chunk's OWN expert subrange (``pack_chunks``), so range
membership is invariant under the permutation and every chunk gets the
aligned order. The sections here gate, top to bottom:

* the host-math helper (pure Python ints — it must stay off the device, since
  its only input is the block-offset host read the caller already paid for);
* ``_padded_route``'s per-chunk permutation structure (CPU tensors): range
  boundaries and membership preserved, whole blocks only, deterministic;
* the wired operator on GPU: a multi-chunk layer's output is ``torch.equal``
  to both the single-chunk reference and the unpacked multi-chunk control,
  the capture refusal still fires, and the top_k>1 combine sits in the
  suite's documented reassociation envelope (which is why the bit gates use
  top_k=1 exact-count routing).

The harness helpers are the minimal copies of tests/
test_bf16_grouped_sm120_lane.py's, duplicated rather than imported for the
reason that file states about its own borrowed helpers: that file stays free
to change without silently changing what this one asserts.
"""
from __future__ import annotations

import importlib.util
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import bf16_grouped_lane as lane  # noqa: E402
from gridbook import lane_select  # noqa: E402

_CHUNK_FLAG = "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK"

# The suite's documented reassociation envelope (tests/
# test_bf16_grouped_sm120_lane.py's ``_REL``): two arms that quantize the same
# values and round once to bf16 but may sum several contributions per token
# through CUDA atomics. Only the top_k>1 case below is held to it.
_REL = 2.1e-2


# ===========================================================================
# The host-math helper. No device, no ext, no vLLM.
# ===========================================================================


def _blocks(*per_expert):
    return list(per_expert)


def test_none_and_full_width_delegate_exactly():
    """chunk=None and chunk>=E are the historical whole-layer packing."""
    counts = _blocks(0, 100, 0, 500, 64, 0, 1, 320)
    base = lane.pack_expert_blocks(counts, 64, 8)
    assert lane.pack_expert_blocks_chunked(counts, 64, 8, None) == base
    assert lane.pack_expert_blocks_chunked(counts, 64, 8, len(counts)) == base
    assert lane.pack_expert_blocks_chunked(counts, 64, 8,
                                           len(counts) + 3) == base


def test_per_chunk_order_is_the_subrange_order_shifted():
    """Each chunk's slice of the order is pack_expert_blocks of ITS counts."""
    counts = _blocks(130, 260, 40, 70, 195, 5, 135, 64)
    order, touched, minimum = lane.pack_expert_blocks_chunked(
        counts, 64, 8, 4)
    head = lane.pack_expert_blocks(counts[:4], 64, 8)
    tail = lane.pack_expert_blocks(counts[4:], 64, 8)
    assert order == [e for e in head[0]] + [e + 4 for e in tail[0]]
    assert touched == head[1] + tail[1]
    assert minimum == head[2] + tail[2]
    # A permutation of the non-empty experts, deterministically.
    assert sorted(order) == [e for e, c in enumerate(counts) if c]
    assert order == lane.pack_expert_blocks_chunked(counts, 64, 8, 4)[0]


def test_width_one_is_the_identity_on_non_empty_experts():
    """One expert per chunk has nothing to reorder within its own range."""
    counts = _blocks(3, 0, 9, 1, 5)
    order, _, _ = lane.pack_expert_blocks_chunked(counts, 64, 8, 1)
    assert order == [0, 2, 3, 4]


def test_every_chunk_aligns_when_independently_packable():
    """Telemetry: groups-touched reaches the minimum iff each chunk does."""
    # Blocks per chunk of 4: {2,5,1} fills a group of 8 exactly;
    # {3,1,3,1} fills two groups exactly (3+1 | 3+1 after FFD).
    counts = _blocks(70, 260, 40, 0, 130, 5, 190, 64)
    order, touched, minimum = lane.pack_expert_blocks_chunked(
        counts, 64, 8, 4)
    assert sorted(order) == [0, 1, 2, 4, 5, 6, 7]
    assert touched == minimum


def test_invalid_width_refuses_rather_than_degrading():
    with pytest.raises(ValueError, match="chunk width"):
        lane.pack_expert_blocks_chunked([1, 2], 64, 8, 0)


def test_packing_helpers_never_touch_a_device_or_host_read():
    """Graph-capture safety by construction: pure int math, no tensor calls.

    The counts arrive as Python ints from the block-offset ``tolist()`` the
    caller already paid for; anything that looked like a tensor op here would
    either sync under capture or duplicate that read.
    """
    for fn in (lane.pack_expert_blocks, lane.pack_expert_blocks_chunked):
        source = inspect.getsource(fn)
        for banned in (".item(", ".tolist(", ".cpu(", ".numpy(",
                       "synchronize", "torch.", "cuda"):
            assert banned not in source, (
                f"{fn.__name__} must stay pure host math; found {banned!r}")


# ===========================================================================
# ``_padded_route`` with ``pack_chunks`` — CPU tensors, real routing ops.
# ``gridbook.moe`` imports vLLM at module scope, so these skip where it is
# absent rather than skipping the file.
# ===========================================================================

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
requires_vllm = pytest.mark.skipif(
    not _HAS_VLLM,
    reason="vLLM not importable here; gridbook.moe imports it at module scope")

_TILE_M = 64
_GROUP = 8
_WIDTH = 4


def _skewed_counts(tile_m=_TILE_M):
    """Ragged E=8 histogram whose BOTH width-4 chunks non-trivially reorder.

    Padded block counts per expert: {2, 5, 1, 0 | 3, 1, 3, 1}. At group 8,
    first-fit-decreasing orders the first chunk [1, 0, 2] and the second
    [4, 6, 5, 7] — neither ascending — while one expert stays empty (a
    zero-M problem) and one is a hair under a whole tile.
    """
    return (tile_m + 6, 4 * tile_m + 4, tile_m - 24, 0,
            2 * tile_m + 2, 5, 2 * tile_m + 62, tile_m)


def _routing_counts(counts, seed, device="cpu"):
    """``top_k=1`` routing realizing an EXACT per-expert histogram.

    ``top_k=1`` is load-bearing wherever a case asserts ``torch.equal`` end to
    end: every token then receives exactly ONE routed contribution, so the
    closing ``index_add_`` is a pure scatter into a zeroed tensor and neither
    the row ORDER of the padded layout nor run-to-run atomic scheduling can
    reassociate it. (At top_k>1 the combine sums several rows per token with
    CUDA atomics, which reassociate — that case is held to ``_REL``.)
    """
    ids = torch.cat([torch.full((int(c),), e, dtype=torch.int32)
                     for e, c in enumerate(counts) if c])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = ids[torch.randperm(ids.numel(), generator=generator)].reshape(-1, 1)
    weights = (torch.rand(ids.shape[0], 1, generator=generator) + 0.1).float()
    return ids.to(device), weights.to(device)


@requires_vllm
def test_padded_route_pack_chunks_permutes_only_within_chunks():
    """The K1.5 layout contract: every chunk keeps exactly its own blocks.

    Against the UNPACKED same-chunking control: identical host block offsets
    (so every ``block_off[c0]..block_off[c1]`` slice the loop takes lands on
    the same tiles), identical per-chunk owner MULTISETS, no foreign expert
    inside any range, at least one range actually reordered (else the gate
    below would be vacuous), and the whole-block invariants the global
    packing already guarantees — same (owner, source row) pairs over real
    rows, padding monotone inside each tile and pointing at the throwaway
    row, router weights untouched, and the result deterministic.
    """
    from gridbook.moe import _padded_route

    counts = _skewed_counts()
    ids, weights = _routing_counts(counts, seed=51)
    E = len(counts)
    kw = dict(trim=True, block_offsets=True)

    plain = _padded_route(ids, weights, E, _TILE_M, **kw)
    packed = _padded_route(ids, weights, E, _TILE_M,
                           pack_chunks=(_WIDTH, _GROUP), **kw)
    again = _padded_route(ids, weights, E, _TILE_M,
                          pack_chunks=(_WIDTH, _GROUP), **kw)

    assert torch.equal(packed.expert_ids, again.expert_ids)
    assert torch.equal(packed.row_src, again.row_src)
    assert torch.equal(packed.dest, again.dest)
    assert packed.block_offsets == plain.block_offsets

    T = int(ids.shape[0])
    reordered = 0
    for c0 in range(0, E, _WIDTH):
        c1 = min(E, c0 + _WIDTH)
        b0, b1 = packed.block_offsets[c0], packed.block_offsets[c1]
        got = packed.expert_ids[b0:b1].tolist()
        want = plain.expert_ids[b0:b1].tolist()
        assert set(got) <= set(range(c0, c1)), "a foreign block entered the chunk"
        assert sorted(got) == want, "range membership not preserved"
        reordered += got != want
        # Whole BLOCKS moved, so each tile still holds its real rows first,
        # and padding rows still name the throwaway destination row T.
        flags = packed.is_pad[b0 * _TILE_M:b1 * _TILE_M] \
            .reshape(-1, _TILE_M).int()
        assert bool((flags[:, 1:] >= flags[:, :-1]).all())
        pad_dest = packed.dest[b0 * _TILE_M:b1 * _TILE_M][
            packed.is_pad[b0 * _TILE_M:b1 * _TILE_M]]
        assert bool((pad_dest == T).all())
    assert reordered >= 1, "the fixture must actually reorder some chunk"

    keep = ~packed.is_pad
    owners = packed.expert_ids.repeat_interleave(_TILE_M)
    pairs = set(zip(owners[keep].tolist(), packed.row_src[keep].tolist()))
    owners_plain = plain.expert_ids.repeat_interleave(_TILE_M)
    keep_plain = ~plain.is_pad
    pairs_plain = set(zip(owners_plain[keep_plain].tolist(),
                          plain.row_src[keep_plain].tolist()))
    assert pairs == pairs_plain
    assert torch.equal(packed.pw_sorted, plain.pw_sorted)
    assert int(packed.is_pad.sum()) == int(plain.is_pad.sum())


@requires_vllm
def test_padded_route_global_and_chunked_packaging_disagree():
    """A width narrower than E must NOT reproduce the global packed order.

    The global spelling moves segments across chunk boundaries (legal only
    when one chunk covers E); the chunked spelling never does. Same routing,
    different orders — this is what makes the two spellings distinct knobs.
    """
    from gridbook.moe import _padded_route

    counts = _skewed_counts()
    ids, weights = _routing_counts(counts, seed=53)
    E = len(counts)
    global_pack = _padded_route(ids, weights, E, _TILE_M, trim=True,
                                block_offsets=True, pack_group=_GROUP)
    chunked_pack = _padded_route(ids, weights, E, _TILE_M, trim=True,
                                 block_offsets=True,
                                 pack_chunks=(_WIDTH, _GROUP))
    assert global_pack.block_offsets == chunked_pack.block_offsets
    assert global_pack.expert_ids.tolist() != chunked_pack.expert_ids.tolist()


@requires_vllm
def test_padded_route_pack_chunks_is_inert_without_the_block_offsets():
    """No host block read, no packing — parity with ``pack_group``."""
    from gridbook.moe import _padded_route

    ids, weights = _routing_counts(_skewed_counts(), seed=55)
    E = len(_skewed_counts())
    plain = _padded_route(ids, weights, E, _TILE_M, trim=True)
    asked = _padded_route(ids, weights, E, _TILE_M, trim=True,
                          pack_chunks=(_WIDTH, _GROUP))
    assert torch.equal(plain.expert_ids, asked.expert_ids)
    assert torch.equal(plain.row_src, asked.row_src)
    assert plain.block_offsets is None and asked.block_offsets is None


@requires_vllm
def test_padded_route_refuses_both_pack_spellings_at_once():
    """One spelling at a time — a silent precedence rule would be a trap."""
    from gridbook.moe import _padded_route

    ids, weights = _routing_counts(_skewed_counts(), seed=57)
    with pytest.raises(ValueError, match="never both"):
        _padded_route(ids, weights, 8, _TILE_M, trim=True,
                      block_offsets=True, pack_group=_GROUP,
                      pack_chunks=(_WIDTH, _GROUP))


@requires_vllm
def test_capture_still_refuses_the_offsets_read_with_pack_chunks(monkeypatch):
    """The K1.5 spelling must not sneak a routing-dependent read into a
    graph: ``block_offsets`` under capture is refused before any packing,
    whatever spelling asked for it (gridbook#47 seam)."""
    from gridbook.moe import _padded_route

    ids, weights = _routing_counts(_skewed_counts(), seed=59)
    monkeypatch.setattr("gridbook.moe._capturing_now", lambda t: True)
    with pytest.raises(RuntimeError, match="gridbook#47"):
        _padded_route(ids, weights, 8, _TILE_M, trim=True,
                      block_offsets=True, pack_chunks=(_WIDTH, _GROUP))


# ===========================================================================
# THE WIRED OPERATOR — ``_apply_prefill_native_bf16`` on the sm12x lane.
# Needs CUDA, the grouped-BF16 extension and the sm12x lane; skips alone.
# ===========================================================================

DEV = "cuda"


def _require_stack():
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the wired K1.5 forward gates")


def _sm120_lane():
    _require_stack()
    from gridbook.cuda_ext import get_bf16_grouped_ext

    ext = get_bf16_grouped_ext()
    if ext is None:
        pytest.skip("owned grouped-BF16 CUTLASS extension unavailable")
    if not hasattr(ext, "cb_bf16_grouped_mm_sm120_gather_out"):
        pytest.skip("this build carries no sm12x lane (needs cc 12.0/12.1)")
    return ext


@pytest.fixture(autouse=True)
def _fresh_flags(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_BF16_SM120", raising=False)
    monkeypatch.delenv(_CHUNK_FLAG, raising=False)
    lane._reset_for_tests()
    lane_select.reset_for_tests(_CHUNK_FLAG)
    yield
    lane._reset_for_tests()
    lane_select.reset_for_tests(_CHUNK_FLAG)


def _build(*, experts=8, hidden=512, inter=512, seed=0, k=44):
    """A synthetic FP8-CB MoE layer, without invoking vLLM weight loading."""
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    codec = pytest.importorskip("gridbook.codec")
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 4
    type_size = fmt.nvfp4_cb_type_size(k, "fp8")
    codebook = fmt._resolve_codebook(
        k, "fp8", "product", None, torch.device(DEV))

    torch.manual_seed(seed)
    w13 = torch.randn(experts, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=DEV) * 0.05
    p13, f13 = fmt.nvfp4_cb_pack(
        w13, k, grid="fp8", mode="product", codebook=codebook)
    p2, f2 = fmt.nvfp4_cb_pack(
        w2, k, grid="fp8", mode="product", codebook=codebook)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {"grid": "fp8", "mode": "product", "k": k,
                     "n_sub": n_sub, "type_size": type_size}
    method.prefix = "test.k15_chunk_packing"
    method.is_fp4 = False
    method.is_v2 = False
    method.k = k
    method.n_sub = n_sub
    method.type_size = type_size
    method._sub_table = None

    layer = types.SimpleNamespace(
        _cb_E=experts,
        _cb_hidden=hidden,
        _cb_inter=inter,
        w13_cb_qweight=p13.reshape(experts, 2 * inter, -1).contiguous(),
        w2_cb_qweight=p2.reshape(experts, hidden, -1).contiguous(),
        w13_weight_scale=f13["scales"].reshape(
            experts, 2 * inter).to(DEV).float(),
        w2_weight_scale=f2["scales"].reshape(experts, hidden).to(DEV).float(),
        _cb_flat=codec.build_flat_codebook([t.to(DEV) for t in codebook]),
        _cb_compose=torch.zeros(1, device=DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer, {"E": experts, "hidden": hidden, "inter": inter}


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 - enum spelling differs across vLLM
        return MoEActivation.SILU


def _routing(tokens, experts, topk, seed, device=DEV):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=generator)[:topk]
                       for _ in range(tokens)]).to(torch.int32)
    weights = (torch.rand(tokens, topk, generator=generator) + 0.1).float()
    return ids.to(device), weights.to(device)


def _resolve_lane_at_load(layer, device):
    """Exactly what ``process_weights_after_loading`` does for this lane."""
    from gridbook import moe

    layer._cb_bf16_sm120 = None
    if moe.bf16_sm120_requested():
        layer._cb_bf16_sm120 = moe.bf16_sm120_require_lane(
            "test routed quality prefill", device=device)
    return layer._cb_bf16_sm120


def _spy_padded_route(monkeypatch):
    """Record every ``_padded_route`` call the prefill makes and its result."""
    from gridbook import moe

    real = moe._padded_route
    seen = []

    def spy(*args, **kwargs):
        route = real(*args, **kwargs)
        seen.append(types.SimpleNamespace(
            pack_group=kwargs.get("pack_group"),
            pack_chunks=kwargs.get("pack_chunks"), route=route))
        return route

    monkeypatch.setattr(moe, "_padded_route", spy)
    return seen


def _set_chunk(monkeypatch, value):
    """Set the chunk override and clear its process-stable latch.

    The knob raises if it moves mid-process; clearing the latch is the
    documented stand-in for the restart an operator would perform (see
    tests/test_bf16_grouped_sm120_lane.py, which does the same).
    """
    monkeypatch.setenv(_CHUNK_FLAG, str(value))
    lane_select.reset_for_tests(_CHUNK_FLAG)


def _report(tag, reference, candidate):
    rel = ((candidate.float() - reference.float()).norm()
           / reference.float().norm().clamp_min(1e-6)).item()
    maxabs = (candidate.float() - reference.float()).abs().max().item()
    print(f"{tag}: rel={rel:.3e} maxabs={maxabs:.3e}")
    return rel


@requires_vllm
def test_multichunk_packed_output_equals_the_single_chunk_reference(
        monkeypatch):
    """THE K1.5 GATE: chunks=4 with per-chunk packing is ``torch.equal`` to
    both the one-chunk reference AND the unpacked multi-chunk control.

    WHY FULL-OUTPUT EQUALITY IS THE HONEST GATE HERE (verified from the code,
    not assumed): chunking splits the EXPERT dimension only, and each routed
    pair-row belongs to exactly one expert, hence exactly one chunk in ANY
    chunking — no reduction ever straddles a chunk boundary. Each padded row's
    operands (bit-exact expanded weight slice, QDQ'd activation row, router
    weight) are the same in every arm, blocks move whole so each M-tile keeps
    its 64 rows, and the collective's FP32 K-reduction order is a property of
    the compiled mainloop, not of the tile's position in a launch. The closing
    ``index_add_`` cannot reassociate anything at top_k=1 (one contribution
    per token — a pure scatter), which is why this case routes with
    ``_routing_counts``. Equality therefore holds end to end, across BOTH
    changes at once (chunk count and packed order).

    Non-vacuity is asserted structurally: against the unpacked control, at
    least one chunk's tile owners are reordered, and every chunk keeps
    exactly its own experts' blocks with the same host offsets.
    """
    from gridbook import moe

    ext = _sm120_lane()
    group = lane.swizzle_group(ext)
    counts = _skewed_counts(lane.tile_m(ext))
    method, layer, dims = _build(seed=61)
    act = _silu_act()
    ids, weights = _routing_counts(counts, seed=63, device=DEV)
    tokens = int(ids.shape[0])
    torch.manual_seed(62)
    x = torch.randn(tokens, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext
    seen = _spy_padded_route(monkeypatch)

    # Arm R: one chunk covers every expert — the legacy global-packing path.
    _set_chunk(monkeypatch, dims["E"])
    single = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    assert seen[-1].pack_group == group
    assert seen[-1].pack_chunks is None, (
        "chunk >= E must take the byte-identical legacy path")

    # Arm C: half the experts per chunk — the K1.5 per-chunk packing path.
    half = dims["E"] // 2
    _set_chunk(monkeypatch, half)
    multi = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    assert seen[-1].pack_group is None
    assert seen[-1].pack_chunks == (half, group)
    packed_route = seen[-1].route

    # Arm U: same chunking, packing OFF (group 1 disables the policy) — the
    # unpacked multi-chunk control the structural claims are made against.
    monkeypatch.setattr(moe, "bf16_sm120_swizzle_group", lambda _ext: 1)
    unpacked = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    assert seen[-1].pack_chunks == (half, 1)
    plain_route = seen[-1].route

    assert single.shape == multi.shape == unpacked.shape == x.shape
    assert torch.isfinite(multi).all()

    # Structure: per-chunk range membership and boundaries are the control's.
    off_p, off_u = packed_route.block_offsets, plain_route.block_offsets
    assert off_p == off_u
    reordered = 0
    for c0 in range(0, dims["E"], half):
        c1 = min(dims["E"], c0 + half)
        b0, b1 = off_p[c0], off_p[c1]
        got = packed_route.expert_ids[b0:b1].tolist()
        want = plain_route.expert_ids[b0:b1].tolist()
        assert set(got) <= set(range(c0, c1)), "foreign blocks in the chunk"
        assert sorted(got) == want
        reordered += got != want
    assert reordered >= 1, "the packing did not move anything"

    # The bit gates.
    assert torch.equal(multi, unpacked), (
        "per-chunk packing changed the multi-chunk result")
    assert torch.equal(multi, single), (
        "the chunked-and-packed result differs from the one-chunk reference")


@requires_vllm
def test_topk_two_multichunk_sits_in_the_reassociation_envelope(monkeypatch):
    """At top_k>1 the closing ``index_add_`` sums several rows per token via
    CUDA atomics, so run-to-run ordering — not this change — bounds the
    comparison. Both arms here differ ONLY in chunk width and packed order,
    i.e. in nothing the combine sees; they must land inside the suite's
    documented envelope, exactly as the sibling file's flag-flip cases do."""
    ext = _sm120_lane()
    method, layer, dims = _build(seed=65)
    act = _silu_act()
    ids, weights = _routing(48, dims["E"], 2, seed=67)
    torch.manual_seed(66)
    x = torch.randn(48, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext

    _set_chunk(monkeypatch, dims["E"])
    single = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    _set_chunk(monkeypatch, dims["E"] // 2)
    multi = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    assert multi.shape == single.shape
    assert _report("k15-multichunk-vs-single[topk=2]", single,
                   multi) <= _REL