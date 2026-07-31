"""Correctness + CAPTURE-SAFETY gate for the STOCK-KERNEL MoE PREFILL path
(``moe.PrismaQuantCBMoEMethod._apply_prefill_stock``, PRISMAQUANT_CB_PREFILL=
stock — task 15 in docs/lanes/nvfp4-cb/prod_hy3_results.md).

The stock path transiently expands each expert-CHUNK into a HARDWARE format
(fp8-CB -> fp8 bytes; fp4-CB-v2 -> bf16) and drives vLLM's OWN fused-MoE grouped
Triton kernel with DEVICE-SIDE routing (moe_align_block_size). Unlike the
_apply_prefill_batched path it replaces, it does NO host reads of device data and
has a FIXED python trip count, so it is CUDA-graph capturable — the prerequisite
for cudagraph_mode=FULL. These tests pin, in order:

  1. test_stock_fp8_quant_matches_codec — the stock W8A8 kernel's built-in
     per-token fp8 activation quant IS codec.fp8_dynamic_act_qdq (the equivalence
     that lets fp8 reuse the kernel's quant for input AND intermediate).
  2. test_stock_routing_matches_loop — moe_align's device routing reproduces the
     loop's (token,slot)->expert assignment.
  3. test_stock_expand_matches_decode — the sliced chunk expand is bit-identical
     to stacking _decode_expert (fp8-byte + bf16 rungs).
  4. test_loop_vs_stock_parity / _chunk_boundaries / _ragged — stock output vs
     _apply_prefill_loop within the suite's 2e-2 reassociation tolerance, across
     rungs, ragged distributions and chunk boundaries (1, E, uneven); exact-match
     rate + max|Δ| reported.
  5. test_stock_capture_replay — THE definitive no-host-reads proof: capture the
     path on fixed shapes, then REPLAY with DIFFERENT routing values and assert
     the replayed output matches the eager result on that new routing.

Run (serving container: vLLM + triton + prismaquant + CUDA):
    docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
      --entrypoint bash vllm-node:latest -c 'pip install -q pytest; \\
      PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
      /repo/plugins/gridbook/tests/test_moe_stock_prefill.py -v -s'
"""
import types

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")

DEV = "cuda"
# Suite tolerance (test_transient_fp8 / test_moe_batched_prefill): bf16-GEMM
# rel-norm 1e-2, + act-quant reassociation 2e-2. The fp4-v2 stock rung uses the
# SAME bf16 grouped GEMM as the loop, so it matches it tightly (~4.6e-3 measured)
# and holds to _REL.
_REL = 2e-2
# The fp8 stock rung runs the NATIVE fp8 tensor-core GEMM (fp8xfp8 -> fp32 accum,
# scales deferred post-accum — the dense linear.py fp8 prefill's GEMM), whereas
# _apply_prefill_loop dequantises val*scale / x*a_scale to bf16 and does a bf16
# MMA. Both quantise to the SAME fp8 grid (test_stock_fp8_quant_matches_codec +
# the weight bit-identity test), so the residual is pure accumulation class. It
# is NOT a fidelity loss: measured against a faithful fp32-dequant reference the
# stock path sits 2.03e-2 away and the loop 1.92e-2 away — equally faithful; the
# 2.04e-2 loop-vs-stock gap is just their mutual bf16-vs-fp8 accumulation. 3e-2
# bounds that fp8-accumulation class (max 2.04e-2 observed + single-seed churn).
_REL_FP8 = 3e-2


def _tol(cfg):
    return _REL_FP8 if cfg == "fp8" else _REL

_CONFIGS = {
    "fp8": dict(grid="fp8", k=44, n_sub=4, scale_coding=None),
    "fp4v2": dict(grid="fp4", k=16, n_sub=2, scale_coding="two_tier"),
}


def _require_stack():
    # Probe the submodule the MoE path actually imports, not just "vllm":
    # test_target_namespace_compat installs STUB vllm modules into sys.modules
    # at collection time when real vLLM is absent, so a bare importorskip
    # passes in a full build-venv suite run and the real import then explodes
    # inside the test — a missing dependency wearing a regression's clothes.
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the stock-prefill forward tests")


def _build(cfg_name, E=8, hidden=512, inter=768, seed=0):
    """PrismaQuantCBMoEMethod (init bypassed) + a namespace layer carrying
    synthetic stacked CB expert weights on CUDA (mirrors test_moe_batched)."""
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    from gridbook.moe import PrismaQuantCBMoEMethod

    cfg = _CONFIGS[cfg_name]
    grid, k, n_sub = cfg["grid"], cfg["k"], cfg["n_sub"]
    is_fp4 = grid == "fp4"
    is_v2 = cfg["scale_coding"] == "two_tier"
    sc = fmt.SCALE_CODING_TWO_TIER if is_v2 else None
    ts = (fmt.nvfp4_cb_type_size(k, grid, sc) if is_fp4
          else fmt.nvfp4_cb_type_size(k, grid))
    cb = fmt._resolve_codebook(k, grid, "product", None, torch.device(DEV))

    torch.manual_seed(seed)
    w13 = torch.randn(E, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(E, hidden, inter, device=DEV) * 0.05

    def _pack(w):
        if is_fp4:
            return fmt.nvfp4_cb_pack(w, k, grid=grid, mode="product",
                                    codebook=cb, scale_coding=sc)
        return fmt.nvfp4_cb_pack(w, k, grid=grid, mode="product", codebook=cb)

    p13, f13 = _pack(w13)
    p2, f2 = _pack(w2)

    m = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    m.quant_config = None
    m.scheme = {"grid": grid, "mode": "product", "k": k, "n_sub": n_sub,
                "type_size": ts}
    m.prefix = "test"
    m.is_fp4 = is_fp4
    m.is_v2 = is_v2
    m.k = k
    m.n_sub = n_sub
    m.type_size = ts
    m._sub_table = codec.TWO_TIER_SUB_TABLE if is_v2 else None

    layer = types.SimpleNamespace()
    layer._cb_E = E
    layer._cb_hidden = hidden
    layer._cb_inter = inter
    layer.w13_cb_qweight = p13.reshape(E, 2 * inter, -1).contiguous()
    layer.w2_cb_qweight = p2.reshape(E, hidden, -1).contiguous()
    layer._cb_flat = codec.build_flat_codebook([t.to(DEV) for t in cb])
    if is_fp4:
        layer._cb_compose = codec.build_compose_table(
            codec.TWO_TIER_SUB_TABLE).to(DEV)
    else:
        layer._cb_compose = torch.zeros(1, device=DEV)
        layer.w13_weight_scale = f13["scales"].reshape(E, 2 * inter).to(
            DEV).float()
        layer.w2_weight_scale = f2["scales"].reshape(E, hidden).to(DEV).float()
    layer.apply_router_weight_on_input = False
    layer.activation = types.SimpleNamespace(value="silu")
    return m, layer, dict(E=E, hidden=hidden, inter=inter)


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 — enum spelling differs across vLLM
        return MoEActivation.SILU


def _routing(T, E, topk, dist, seed):
    """(topk_ids int32 [T,topk] distinct-per-token, topk_weights f32 [T,topk])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dist == "uniform":
        ids = torch.stack([torch.randperm(E, generator=g)[:topk]
                           for _ in range(T)])
    elif dist == "subset":                          # only experts [0, pool) hit
        pool = max(topk, E // 2)
        ids = torch.stack([torch.randperm(pool, generator=g)[:topk]
                           for _ in range(T)])
    elif dist == "one_expert":
        assert topk == 1
        ids = torch.full((T, 1), 3, dtype=torch.long)
    else:
        raise ValueError(dist)
    w = torch.rand(T, topk, generator=g) + 0.1
    return ids.to(torch.int32).to(DEV), w.float().to(DEV)


def _report(tag, o_ref, o_new):
    rel = ((o_new.float() - o_ref.float()).norm()
           / o_ref.float().norm().clamp_min(1e-6)).item()
    exact = (o_new == o_ref).float().mean().item()
    maxabs = (o_new.float() - o_ref.float()).abs().max().item()
    print(f"{tag}: rel={rel:.3e} exact-match={exact:.3f} max|Δ|={maxabs:.3e}")
    return rel


# --------------------------------------------------------------------------- #
# 1 — the stock kernel's fp8 activation quant IS codec.fp8_dynamic_act_qdq.      #
# --------------------------------------------------------------------------- #
def test_stock_fp8_quant_matches_codec():
    """vLLM's per-token fp8-dynamic activation quant (moe_kernel_quantize_input,
    which the fp8 stock path uses for the input AND intermediate), dequantised,
    equals codec.fp8_dynamic_act_qdq. This is the equivalence that authorises
    reusing the kernel's built-in quant instead of running ours explicitly."""
    _require_stack()
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.fused_moe.utils import (
        moe_kernel_quantize_input,
    )
    torch.manual_seed(0)
    x = (torch.randn(96, 512, device=DEV) * 0.7).to(torch.bfloat16)
    xq, xs = moe_kernel_quantize_input(
        x, None, current_platform.fp8_dtype(), per_act_token_quant=True)
    deq = (xq.to(torch.float32) * xs.to(torch.float32)).to(x.dtype)
    ref = codec.fp8_dynamic_act_qdq(x)
    rel = _report("fp8-quant vs codec", ref, deq)
    assert rel <= _REL, f"stock fp8 quant diverges from codec: rel {rel:.3e}"


# --------------------------------------------------------------------------- #
# 2 — moe_align device routing reproduces the loop's (token,slot)->expert.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dist", ["uniform", "subset"])
def test_stock_routing_matches_loop(dist):
    """A single all-E chunk (identity expert_map): decode moe_align's
    sorted_token_ids / expert_ids back to a (token,slot)->expert map and assert
    it equals topk_ids — i.e. the device routing lands every routed pair on the
    same expert the loop's torch.where(topk_ids==e) does. No -1 padding blocks
    (ignore_invalid_experts=True), padding token ids skipped."""
    _require_stack()
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    E, T, topk = 8, 40, 4
    ti, _ = _routing(T, E, topk, dist, seed=3)
    ti_i = ti.to(torch.int32)
    block_m = 16
    expert_map = torch.arange(E, dtype=torch.int32, device=DEV)   # identity
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        ti_i, block_m, E, expert_map, ignore_invalid_experts=True)
    n_valid = T * topk
    npad = int(num_pad.item())
    seen = torch.zeros(n_valid, dtype=torch.bool)
    for b in range(npad // block_m):
        e = int(expert_ids[b].item())
        blk = sorted_ids[b * block_m:(b + 1) * block_m]
        for p in blk.tolist():
            if p >= n_valid:                          # padding slot
                continue
            token, slot = p // topk, p % topk
            assert int(ti[token, slot].item()) == e, (
                f"pair {p} (tok {token},slot {slot}) routed to {e} != "
                f"topk_ids {int(ti[token, slot].item())}")
            seen[p] = True
    assert bool(seen.all()), "some routed pair never appeared in moe_align output"


# --------------------------------------------------------------------------- #
# 3 — sliced chunk expand == stacking _decode_expert (fp8-byte + bf16 rungs).    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("which", ["w13", "w2"])
def test_stock_expand_matches_decode(cfg, which):
    _require_stack()
    m, layer, d = _build(cfg)
    is_fp8 = cfg == "fp8"
    c0, c1 = 2, 6
    stacked = m._expand_stack_slice(layer, which, c0, c1, to_fp8=is_fp8)
    for i, e in enumerate(range(c0, c1)):
        per = m._decode_expert(layer, which, e)               # [out, in] bf16
        got = stacked[i]
        if is_fp8:
            got = got.to(torch.float32) * getattr(
                layer, f"{which}_weight_scale")[e][:, None].to(torch.float32)
            got = got.to(torch.bfloat16)
        assert torch.equal(got, per), (
            f"{cfg}/{which}: sliced expand expert {e} != _decode_expert")


# --------------------------------------------------------------------------- #
# 4 — loop vs stock parity across rungs and token distributions.                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_loop_vs_stock_parity(cfg, dist, topk):
    _require_stack()
    m, layer, d = _build(cfg, seed=1)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], topk, dist, seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_stk = m._apply_prefill_stock(layer, x, tw, ti, act, chunk=3)
    assert o_stk.shape == o_loop.shape == (T, d["hidden"])
    rel = _report(f"parity[{cfg},{dist},topk={topk}]", o_loop, o_stk)
    tol = _tol(cfg)
    assert rel <= tol, f"{cfg}/{dist}/topk={topk}: stock rel {rel:.3e} > {tol}"


# --------------------------------------------------------------------------- #
# 5 — chunk boundaries: 1, E//2+1 (uneven), E, > E all match the loop.           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("chunk", [1, 5, 8, 1000])
def test_stock_chunk_boundaries(cfg, chunk):
    _require_stack()
    m, layer, d = _build(cfg, seed=6)
    act = _silu_act()
    T, topk = 48, 4
    ti, tw = _routing(T, d["E"], topk, "uniform", seed=13)
    torch.manual_seed(7)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_stk = m._apply_prefill_stock(layer, x, tw, ti, act, chunk=chunk)
    rel = _report(f"chunk[{cfg},chunk={chunk}]", o_loop, o_stk)
    tol = _tol(cfg)
    assert rel <= tol, f"{cfg}/chunk={chunk}: rel {rel:.3e} > {tol}"


# --------------------------------------------------------------------------- #
# 6 — ragged: one expert ALL tokens; some experts 0 tokens.                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
def test_stock_ragged_zero_and_all(cfg):
    _require_stack()
    m, layer, d = _build(cfg, seed=4)
    act = _silu_act()
    T = 40
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=11)
    torch.manual_seed(5)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_stk = m._apply_prefill_stock(layer, x, tw, ti, act, chunk=3)
    rel = _report(f"ragged-one-expert[{cfg}]", o_loop, o_stk)
    tol = _tol(cfg)
    assert rel <= tol

    ti2, tw2 = _routing(T, d["E"], 2, "subset", seed=12)
    o_loop2 = m._apply_prefill_loop(layer, x, tw2, ti2, act)
    o_stk2 = m._apply_prefill_stock(layer, x, tw2, ti2, act, chunk=3)
    rel2 = _report(f"ragged-subset[{cfg}]", o_loop2, o_stk2)
    assert rel2 <= tol


# --------------------------------------------------------------------------- #
# 7 — THE capture-safety proof: capture on fixed shapes, replay with DIFFERENT   #
#     routing, assert both replays match the eager result on that routing.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
def test_stock_capture_replay(cfg):
    """If the path did ANY host read of device data (.tolist/.item/.cpu/nonzero
    -> python) the captured graph would bake the capture-time routing and a
    replay with different routing would produce the WRONG output. We capture on
    routing #A, then replay after copying routing #B into the static inputs and
    require the output to equal the EAGER stock result on #B (and same for #A).
    Passing across two distinct routings is the definitive no-host-reads proof."""
    _require_stack()
    m, layer, d = _build(cfg, seed=8)
    act = _silu_act()
    E, hidden = d["E"], d["hidden"]
    T, topk, chunk = 32, 4, 3           # chunk=3 over E=8 -> 3 chunks, uneven last

    tiA, twA = _routing(T, E, topk, "uniform", seed=21)
    tiB, twB = _routing(T, E, topk, "subset", seed=22)     # DIFFERENT routing
    torch.manual_seed(23)
    xA = torch.randn(T, hidden, dtype=torch.bfloat16, device=DEV) * 0.5
    xB = torch.randn(T, hidden, dtype=torch.bfloat16, device=DEV) * 0.5

    # Eager references (also warm _cb_flat_fp8 + triton JIT for both routings).
    refA = m._apply_prefill_stock(layer, xA, twA, tiA, act, chunk=chunk)
    refB = m._apply_prefill_stock(layer, xB, twB, tiB, act, chunk=chunk)

    # Static input buffers the graph reads from every replay.
    x_s = xA.clone()
    ti_s = tiA.clone()
    tw_s = twA.clone()

    # Warm up on a side stream (JIT/autotune must not happen during capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m._apply_prefill_stock(layer, x_s, tw_s, ti_s, act, chunk=chunk)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = m._apply_prefill_stock(layer, x_s, tw_s, ti_s, act, chunk=chunk)

    # Replay #A (captured routing).
    x_s.copy_(xA); ti_s.copy_(tiA); tw_s.copy_(twA)
    g.replay()
    torch.cuda.synchronize()
    relA = _report(f"capture-replay-A[{cfg}]", refA, out_s)
    assert relA <= _REL, f"{cfg}: replay #A rel {relA:.3e} > {_REL}"

    # Replay #B (DIFFERENT routing copied into the static inputs) — the proof.
    x_s.copy_(xB); ti_s.copy_(tiB); tw_s.copy_(twB)
    g.replay()
    torch.cuda.synchronize()
    relB = _report(f"capture-replay-B[{cfg}]", refB, out_s)
    assert relB <= _REL, (
        f"{cfg}: replay with different routing rel {relB:.3e} > {_REL} — "
        "a host read baked the capture-time routing (capture-safety broken)")
