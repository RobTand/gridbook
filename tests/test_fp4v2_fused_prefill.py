"""Gates for the FP4-CB v2 fused mid-M lane (csrc/cb_fused_fp4v2_gemm.cu).

The lane's whole claim is CONTRACT PRESERVATION: the weights it decodes inside
the CUTLASS prologue are bit-identical to what ``cb_expand_v2`` writes to HBM
today, and the activations are the same BF16 group-16 QDQ output the shipping
bridge already consumes — so the ONLY thing that moves is the FP32 GEMM
reduction order (2026-08-01 performance audit §3 P2a). This file is the
evidence for that claim, in three layers of decreasing strength:

1. **DIRECT decode read-out (the primary gate).** With a one-hot ``A``
   (``a[m, k0+m] = 1``) the GEMM degenerates to
   ``y[m, n] = W_decoded[n, k0+m]`` EXACTLY: the fp32 accumulator sums one
   ``1.0 * f32(w)`` term and a column of zeros, and
   ``bf16(f32(bf16 w)) == w``. Sweeping ``k0`` over every window of
   a small problem therefore reads back the ENTIRE decoded ``[N, K]`` tile and
   compares it to ``cb_expand_v2``'s output **bit-for-bit**. Nothing here is a
   tolerance; a single wrong codeword, sub-table base, scale nibble or compose
   gather changes a bf16 pattern and fails.

2. **Whole-tile GEMM equality against the passthrough oracle.**
   ``sm120_fp4v2_bf16_mm_fork`` is the same 128x64x64 config with plain BF16 B
   through TMA, so its FP32 accumulation ORDER is identical to the fused
   lane's. Feeding it the ``cb_expand_v2`` tile must reproduce the fused
   output bit-for-bit. This is the fp8 twin's gate
   (``tests/test_fused_prefill.py::test_fused_bitexact_synth``) transposed to
   this payload, and it also catches decode errors at K positions a one-hot
   sweep of a small shape would not visit.

3. **End-to-end numerics vs the SHIPPING route** under the reduction-order
   tolerance discipline this repo already uses for the sm12x BF16 lane
   (``tests/test_bf16_grouped_cutlass.py``): relative L2 against an fp32
   reference, capped at 2e-3 and required to be no worse than a BF16
   ``F.linear`` computing the same product.

The packed bytes come from the REAL prismaquant encoder (``scale_coding=
"two_tier"``), never fabricated: only the encoder guarantees every
``(super, sub)`` scale pair is legal/E4M3-exact, and a fabricated illegal pair
would make this file's "bit-exact" claim vacuous.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

codec = pytest.importorskip("gridbook.codec")
pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")

from gridbook.expand import expand_fp4_v2_to_weight  # noqa: E402
from gridbook.ops import cb_bf16_grouped_mm  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

from gridbook.moe_gemv_select import cb_gemv_v2_device_support  # noqa: E402

_supported, _reason, _index = cb_gemv_v2_device_support("cuda:0")
if not _supported:
    pytest.skip(_reason, allow_module_level=True)

from gridbook.cuda_ext import (get_fused_fp4v2_ext,  # noqa: E402
                               require_fp4_v2_expander)

ext = get_fused_fp4v2_ext()
if ext is None:
    pytest.skip("fp4-v2 fused mid-M ext unavailable (no nvcc / CUTLASS?)",
                allow_module_level=True)

# The expander oracle needs its opt-in smem carve-out set before first launch.
require_fp4_v2_expander("fp4-v2 fused mid-M decode gate", device="cuda:0")

DEV = "cuda"
# The whole product ladder the lane claims to serve.
RUNGS = tuple(range(12, 25))
# The two rungs whose codebook cannot be staged whole (sub0 only, sub1 stays in
# global) plus the one that exactly fills the largest compiled stage — the
# smem-tightest cells of the shipped residency ladder.
TIGHT_RUNGS = (22, 23, 24)


def _prep(k, N, K, seed=0):
    """One dense fp4-v2 layer's tensors, straight from the REAL encoder."""
    cb = pq._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(1, N, K, generator=g) * 0.02).to(DEV)
    fields = pq.nvfp4_cb_fields(w, k, grid="fp4", mode="product", codebook=cb,
                                scale_coding="two_tier", encode_tier="fast")
    raw = pq.nvfp4_cb_assemble_bytes(fields, k, grid="fp4", mode="product")
    ts = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")            # 4k + 9
    n_sb = K // codec.SUPERBLOCK
    packed = raw.reshape(N, n_sb * ts).contiguous().to(DEV)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    return dict(
        qwp=codec.pad_qweight(packed),
        cb_flat=codec.build_flat_codebook(subs),
        compose=codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV),
        row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
        N=N, K=K, k=k, ts=ts)


def _expanded(p):
    """The SHIPPING decoded tile — this lane's bit-exactness target."""
    return expand_fp4_v2_to_weight(p["qwp"], p["cb_flat"], p["row_off"],
                                   p["compose"], p["N"], p["K"], p["k"], 2,
                                   p["ts"])


def _fused(p, a, force_lut_bytes=-1):
    return ext.cb_fused_fp4v2_prefill_mm(
        a, p["qwp"], p["cb_flat"], p["compose"], p["N"], p["K"], p["k"],
        force_lut_bytes)


def _one_hot(M, K, k0):
    """``a[m, k0+m] = 1`` — the probe that turns the GEMM into a read-out."""
    a = torch.zeros(M, K, dtype=torch.bfloat16, device=DEV)
    a[torch.arange(M, device=DEV), k0 + torch.arange(M, device=DEV)] = 1.0
    return a


def _rel_l2(y, reference):
    return ((y.float() - reference).norm()
            / reference.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# 1. DIRECT decode read-out — the primary contract-preservation gate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", RUNGS)
def test_prologue_decode_is_bit_exact_vs_cb_expand_v2(k):
    """Every decoded weight of a whole [N,K] tile equals cb_expand_v2's byte.

    M = 128 = TileM, so each call reads back 128 consecutive K columns; four
    calls cover K = 512 completely. That is EVERY codeword slot of both
    superblocks, both product sub-tables, and all sixteen scale groups.
    """
    N, K, M = 64, 512, 128
    p = _prep(k, N, K, seed=k)
    W = _expanded(p)
    assert W.shape == (N, K) and W.dtype is torch.bfloat16
    for k0 in range(0, K, M):
        y = _fused(p, _one_hot(M, K, k0))
        got = y.t().contiguous()                       # [N, M] = W[:, k0:k0+M]
        want = W[:, k0:k0 + M].contiguous()
        assert torch.equal(got.view(torch.uint16), want.view(torch.uint16)), (
            f"k={k}: in-prologue decode differs from cb_expand_v2 in the "
            f"K window [{k0}, {k0 + M})")


@pytest.mark.parametrize("k", TIGHT_RUNGS)
def test_decode_bit_exact_with_the_codebook_forced_to_global(k):
    """The smem residency choice may not change a single decoded bit.

    ``force_lut_bytes=0`` compiles out the codebook stage entirely, so both
    sub-table gathers go to global. Agreement with the auto (staged) class is
    what proves the pointer-select in the mainloop is the only difference.
    """
    N, K, M = 64, 512, 128
    p = _prep(k, N, K, seed=100 + k)
    W = _expanded(p)
    for k0 in range(0, K, M):
        a = _one_hot(M, K, k0)
        auto = _fused(p, a).t().contiguous()
        glob = _fused(p, a, force_lut_bytes=0).t().contiguous()
        want = W[:, k0:k0 + M].contiguous()
        assert torch.equal(glob.view(torch.uint16), want.view(torch.uint16))
        assert torch.equal(auto.view(torch.uint16), glob.view(torch.uint16))


def test_decode_bit_exact_across_superblocks_and_an_n_residue():
    """A shape that is not a whole number of N tiles, over 3 superblocks.

    N = 136 leaves an 8-row residue in the last 64-wide tile (the predicated
    epilogue path) and K = 768 makes the K-tile counter walk three superblocks,
    which is where an off-by-one in the superblock/quarter split would show.
    """
    k, N, K, M = 16, 136, 768, 128
    p = _prep(k, N, K, seed=7)
    W = _expanded(p)
    for k0 in range(0, K, M):
        y = _fused(p, _one_hot(M, K, k0))
        assert torch.equal(y.t().contiguous().view(torch.uint16),
                           W[:, k0:k0 + M].contiguous().view(torch.uint16))


@pytest.mark.parametrize("mode,name", [(1, "tile row"),
                                       (2, "column within the K-tile"),
                                       (3, "absolute K-tile index")])
def test_the_decode_write_view_addresses_exactly_what_the_mma_reads(mode,
                                                                    name):
    """REGRESSION for a measured swizzle-view bug, pinned by construction.

    ``debug_mode`` makes the decoder store the COORDINATE it used instead of
    the decoded value, so the one-hot read-out recovers the decoder's
    (row, column, K-tile) -> smem mapping directly. All three must come back
    as the identity.

    The first implementation built the decode-write view by slicing the pipe
    mode BEFORE calling ``as_position_independent_swizzle_tensor`` — the
    expression the fp8 fork uses, whose layout aliases differently. For this
    bf16 128-byte swizzle atom that produced a DIFFERENT physical mapping than
    the reader's: columns permuted by XOR 8 on odd rows, and rows 8/9
    transposed. Nothing in a value-level comparison says WHICH of the codeword
    extraction, the scale compose or the address arithmetic is wrong; this
    probe separates them, and keeping it means the two views cannot silently
    diverge again.
    """
    N, K, M = 64, 512, 128
    p = _prep(16, N, K, seed=11)
    for k0 in range(0, K, M):
        y = ext.cb_fused_fp4v2_prefill_mm(
            _one_hot(M, K, k0), p["qwp"], p["cb_flat"], p["compose"],
            N, K, 16, -1, mode)
        got = y.t().contiguous().float()             # [N, M] over (n, kglob)
        cols = torch.arange(M, device=DEV) + k0
        if mode == 1:
            want = torch.arange(N, device=DEV, dtype=torch.float32)
            want = want[:, None].expand(N, M)
        elif mode == 2:
            want = (cols % 64).to(torch.float32)[None, :].expand(N, M)
        else:
            want = (cols // 64).to(torch.float32)[None, :].expand(N, M)
        assert torch.equal(got, want.contiguous()), (
            f"decode write view disagrees with the MMA read view in {name} "
            f"(K window [{k0}, {k0 + M}))")


# ---------------------------------------------------------------------------
# 2. Whole-tile GEMM equality against the passthrough oracle.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", (12, 16, 20, 24))
@pytest.mark.parametrize("M", (9, 32, 64, 128))
def test_fused_equals_the_passthrough_oracle_bit_for_bit(k, M):
    """Same tile, same TiledMma, same epilogue -> same reduction order.

    So any difference is the decode, and there is none.
    """
    N, K = 128, 512
    p = _prep(k, N, K, seed=k + M)
    W = _expanded(p)
    g = torch.Generator(device=DEV).manual_seed(M)
    a = (torch.randn(M, K, generator=g, device=DEV) * 0.1).to(torch.bfloat16)
    y_ref = ext.sm120_fp4v2_bf16_mm_fork(a, W)
    y_f = _fused(p, a)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16)), (
        f"k={k} M={M}: the fused lane and the passthrough oracle on the "
        f"cb_expand_v2 tile are not bit-identical")


# ---------------------------------------------------------------------------
# 3. End-to-end numerics vs the SHIPPING expand + bridge route.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", (9, 16, 32, 64, 128))
def test_end_to_end_matches_the_bridge_within_the_reduction_order_band(M):
    """The lane's fp32-accumulate error may not exceed a BF16 F.linear's.

    Same discipline as the sm12x BF16 grouped lane's gate: relative L2 against
    an fp32 reference, an absolute backstop for BF16 output rounding, and a
    relative comparison to ``F.linear`` on the SAME operands. The fused lane,
    the shipping bridge and F.linear compute the same product in three
    different fp32 orders; none may be meaningfully worse than the others.
    """
    k, N, K = 16, 512, 1024
    p = _prep(k, N, K, seed=3)
    W = _expanded(p)
    g = torch.Generator(device=DEV).manual_seed(M)
    a = (torch.randn(M, K, generator=g, device=DEV) * 0.1).to(torch.bfloat16)

    fp32 = (a.float() @ W.float().t())
    fused = _fused(p, a)
    bf16_linear = F.linear(a, W)
    bridge = cb_bf16_grouped_mm(
        a.contiguous(), W.unsqueeze(0),
        torch.full((1,), M, dtype=torch.int32, device=DEV), 0)

    fused_rel = _rel_l2(fused, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)
    bridge_rel = _rel_l2(bridge, fp32)
    assert fused_rel <= 2e-3, (
        f"M={M}: fused relative L2 {fused_rel:.6e} exceeds the BF16 output "
        f"rounding backstop")
    assert fused_rel <= torch.maximum(1.25 * linear_rel, linear_rel + 2e-5), (
        f"M={M}: fused {fused_rel:.6e} vs BF16 F.linear {linear_rel:.6e}: the "
        f"reduction-order difference is larger than the bf16 rounding floor")
    assert fused_rel <= torch.maximum(1.25 * bridge_rel, bridge_rel + 2e-5), (
        f"M={M}: fused {fused_rel:.6e} vs the shipping bridge "
        f"{bridge_rel:.6e}")


# ---------------------------------------------------------------------------
# Binding contract: the HARD gate and the input validation.
# ---------------------------------------------------------------------------
def test_m_above_the_mid_m_ceiling_is_refused_not_served_slowly():
    """The kernel enforces its own gate, so a drifted python gate cannot
    quietly route a large-M prefill into a schedule that re-decodes B per
    M-tile (the fp8 twin measured 0.22x at M ~ 1400)."""
    k, N, K = 16, 64, 512
    p = _prep(k, N, K, seed=1)
    max_m = int(ext.cb_fused_fp4v2_max_m())
    assert max_m == 128
    a = torch.zeros(max_m + 1, K, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="fused mid-M lane serves"):
        _fused(p, a)


@pytest.mark.parametrize("k", (11, 25))
def test_rungs_outside_the_product_ladder_are_refused(k):
    p = _prep(16, 64, 512, seed=1)
    a = torch.zeros(16, 512, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="unsupported k_bits"):
        ext.cb_fused_fp4v2_prefill_mm(a, p["qwp"], p["cb_flat"], p["compose"],
                                      64, 512, k, -1)


def test_a_wrong_sized_codebook_is_refused():
    """One zero-based product dictionary, exactly — the same rule
    cb_expand_v2 and expand_fp4_v2_to_weight enforce."""
    p = _prep(16, 64, 512, seed=1)
    a = torch.zeros(16, 512, dtype=torch.bfloat16, device=DEV)
    doubled = torch.cat([p["cb_flat"], p["cb_flat"]]).contiguous()
    with pytest.raises(RuntimeError, match="cb_flat must be"):
        ext.cb_fused_fp4v2_prefill_mm(a, p["qwp"], doubled, p["compose"],
                                      64, 512, 16, -1)


def test_an_uncompiled_codebook_stage_class_is_refused():
    p = _prep(16, 64, 512, seed=1)
    a = torch.zeros(16, 512, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="not a compiled codebook-stage"):
        _fused(p, a, force_lut_bytes=49152)


# ---------------------------------------------------------------------------
# Attestation: what was actually compiled, and the shipped tables.
# ---------------------------------------------------------------------------
def test_config_is_the_shipped_tile():
    tile_m, tile_n, tile_k, stages, cap = ext.cb_fused_fp4v2_config()
    assert (tile_m, tile_n, tile_k, stages) == (128, 64, 64, 2)
    assert cap == 101376
    assert int(ext.cb_fused_fp4v2_max_m()) == tile_m


def test_compiled_rungs_are_the_whole_product_ladder():
    assert tuple(int(k) for k in ext.cb_fused_fp4v2_kbits()) == RUNGS


def test_smem_report_matches_the_shipped_table():
    """The table in cb_fused_fp4v2_gemm.cu's header comment is the OUTPUT of
    csrc/tools/smem_probe_fp4v2_bf16.cu; assert the built module agrees, so a
    stale comment cannot survive a config change."""
    flat = [int(v) for v in ext.cb_fused_fp4v2_smem_report()]
    pairs = {flat[i]: flat[i + 1] for i in range(0, len(flat) - 2, 2)}
    assert pairs[0] == 52224
    assert pairs[4096] == 56320
    assert pairs[16384] == 68608
    assert pairs[32768] == 84992
    assert flat[-1] == 101376
    for lut_bytes, size in pairs.items():
        assert size <= 101376, (lut_bytes, size)


def test_lut_residency_ladder_matches_the_shipped_table():
    """Full staging up to k22; sub0-only at k23/k24 (their tables are 48 and
    64 KiB, over every compiled stage class)."""
    flat = [int(v) for v in ext.cb_fused_fp4v2_lut_plan()]
    plan = {flat[i]: tuple(flat[i + 1:i + 4]) for i in range(0, len(flat), 4)}
    assert set(plan) == set(RUNGS)
    for k, (cb_bytes, lut_bytes, stage_bytes) in plan.items():
        w0, w1 = (k + 1) // 2, k // 2
        assert cb_bytes == (8 << w0) + (8 << w1)
        assert lut_bytes in tuple(
            int(c) for c in ext.cb_fused_fp4v2_lut_classes())
        assert stage_bytes <= lut_bytes
        if cb_bytes <= 32768:
            assert stage_bytes == cb_bytes, f"k={k} should stage fully"
        else:
            assert stage_bytes == (8 << w0), f"k={k} should stage sub0 only"
    assert plan[24][1:] == (32768, 32768)
    assert plan[12][1:] == (4096, 1024)
