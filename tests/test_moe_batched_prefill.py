"""Correctness gate for the BATCHED-EXPERT MoE PREFILL path
(``moe.PrismaQuantCBMoEMethod._apply_prefill_batched``, task 15 in
docs/nvfp4-cb-plan/prod_hy3_results.md §"open").

The batched path replaces the per-expert prefill loop's E Triton expands + E
activation-QDQ passes with ONE act-QDQ over all tokens and ONE expand per
projection per expert-CHUNK, then a grouped/segmented bf16 GEMM. Its numerics
must match the loop within the suite's tolerance class:

  * weights are bit-identical (same expander) — test_batched_expand_matches_decode;
  * per-token QDQ is bit-identical because both activation QDQs are per-row ops
    — test_qdq_once_equals_per_selection (the 1a property);
  * only the GEMM accumulation + cross-expert combine reassociate — the
    loop-vs-batched parity tests assert rel-norm <= the fp8/fp4 tolerance and
    REPORT the exact-match rate + max|Δ| (the served logprob A/B is the adoption
    gate).

Two run scopes:

* ``-k qdq_once`` (build venv, NO vLLM, CPU ok): the per-row QDQ property.
    PYTHONPATH=/home/rob/prismaquant:/home/rob/prismaquant/plugins/gridbook \\
      /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest \\
      plugins/gridbook/tests/test_moe_batched_prefill.py -q -k qdq_once

* everything else (serving container: vLLM + triton + prismaquant + CUDA):
    docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
      --entrypoint bash vllm-node-tf5-cu132-lfm:latest -c 'pip install -q pytest; \\
      PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
      /repo/plugins/gridbook/tests/test_moe_batched_prefill.py -v'
"""
import types

import pytest
import torch

# codec is torch-only (no triton / no vLLM) so the QDQ property test runs
# anywhere; expand/moe/fmt are imported lazily inside the GPU tests.
codec = pytest.importorskip("gridbook.codec")


# --------------------------------------------------------------------------- #
# 1a — the per-row QDQ property (CPU ok; the whole batched path relies on it).  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_qdq_once_equals_per_selection(kind):
    """Both activation QDQs are PER-TOKEN-ROW ops, so quantising once over all
    rows then gathering a selection == quantising the gathered rows:
    ``QDQ(x)[sel] == QDQ(x[sel])`` BIT-EXACTLY. This is what lets the batched
    path do a single act-QDQ instead of one per expert."""
    torch.manual_seed(0)
    T, in_f = 40, 512                              # in_f % 16 == 0 for fp4 groups
    x = torch.randn(T, in_f) * 0.7
    qdq = (codec.fp4_group16_act_qdq if kind == "fp4"
           else codec.fp8_dynamic_act_qdq)
    xq_full = qdq(x)
    # A ragged selection with repeats (a token routed to several experts) and
    # a reordering — the batched path gathers rows in an arbitrary order.
    sel = torch.tensor([0, 5, 5, 39, 12, 12, 12, 1, 38, 0])
    lhs = qdq(x)[sel]
    rhs = qdq(x[sel])
    assert torch.equal(lhs, rhs), (
        f"{kind}: QDQ(x)[sel] != QDQ(x[sel]) — QDQ is not a pure per-row op")
    # And the full pass is self-consistent row-for-row.
    assert torch.equal(xq_full[sel], rhs)


# --------------------------------------------------------------------------- #
# GPU + vLLM + prismaquant synthetic-layer builder for the forward tests.       #
# --------------------------------------------------------------------------- #
DEV = "cuda"
# Established tolerances (test_transient_fp8 / test_cb_kernels): bf16-GEMM
# rel-norm 1e-2; +act-quant reassociation 2e-2. Loop-vs-batched differs only by
# GEMM/combine reassociation, so 2e-2 is the contract; segmented (default) is
# far tighter (bit-identical GEMM, only combine reassociates).
_REL = 2e-2

_CONFIGS = {
    # grid, k, n_sub, scale_coding, method attrs
    "fp8": dict(grid="fp8", k=44, n_sub=4, scale_coding=None),
    "fp4v2": dict(grid="fp4", k=16, n_sub=2, scale_coding="two_tier"),
}


def _require_stack():
    # NOT a bare importorskip("vllm"): test_target_namespace_compat installs
    # STUB vllm modules into sys.modules at collection time when real vLLM is
    # absent, so in a full build-venv suite run "vllm" imports fine and the
    # real submodule import then explodes INSIDE the test (a failure that looks
    # like a code regression but is only a missing dependency). Probe the
    # submodule the MoE path actually needs, so a stubbed env skips like an
    # empty one. In the serving container the real package wins and nothing
    # here changes.
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the batched-prefill forward tests")


def _build(cfg_name, E=8, hidden=512, inter=768, seed=0):
    """Construct a PrismaQuantCBMoEMethod (init bypassed, like test_moe_stacked)
    + a namespace layer carrying synthetic stacked CB expert weights on CUDA."""
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
    w13 = torch.randn(E, 2 * inter, hidden, device=DEV) * 0.05   # gate_up
    w2 = torch.randn(E, hidden, inter, device=DEV) * 0.05        # down

    def _pack(w):
        if is_fp4:
            packed, fields = fmt.nvfp4_cb_pack(
                w, k, grid=grid, mode="product", codebook=cb, scale_coding=sc)
        else:
            packed, fields = fmt.nvfp4_cb_pack(
                w, k, grid=grid, mode="product", codebook=cb)
        return packed, fields

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
    # for apply() (shared-experts test)
    layer.apply_router_weight_on_input = False
    layer.activation = types.SimpleNamespace(value="silu")
    return m, layer, dict(E=E, hidden=hidden, inter=inter)


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 - enum spelling differs across vLLM
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
    elif dist == "one_expert":                      # every pair -> one expert
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
# 1b — the batched expand is bit-identical to stacking _decode_expert.          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("which", ["w13", "w2"])
def test_batched_expand_matches_decode(cfg, which):
    _require_stack()
    m, layer, d = _build(cfg)
    experts = [0, 3, 4, 7]
    stacked = m._expand_expert_stack(layer, which, experts)   # [C, out, in]
    for i, e in enumerate(experts):
        per = m._decode_expert(layer, which, e)               # [out, in]
        assert torch.equal(stacked[i], per), (
            f"{cfg}/{which}: batched expand expert {e} != _decode_expert "
            "(1b bit-identity broken)")


# --------------------------------------------------------------------------- #
# 3 — loop vs batched parity across rungs and token distributions.              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_loop_vs_batched_parity(cfg, dist, topk):
    _require_stack()
    m, layer, d = _build(cfg, seed=1)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], topk, dist, seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_bat = m._apply_prefill_batched(layer, x, tw, ti, act, use_grouped_mm=False)
    assert o_bat.shape == o_loop.shape == (T, d["hidden"])
    rel = _report(f"parity[{cfg},{dist},topk={topk},segmented]", o_loop, o_bat)
    assert rel <= _REL, f"{cfg}/{dist}/topk={topk}: batched rel {rel:.3e} > {_REL}"


# --------------------------------------------------------------------------- #
# 3b — the opt-in torch._grouped_mm single-launch GEMM path.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
def test_grouped_mm_parity(cfg):
    _require_stack()
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm not present on this build")
    m, layer, d = _build(cfg, seed=1)
    act = _silu_act()
    T, topk = 48, 4
    ti, tw = _routing(T, d["E"], topk, "uniform", seed=9)
    torch.manual_seed(3)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_gmm = m._apply_prefill_batched(layer, x, tw, ti, act, use_grouped_mm=True)
    rel = _report(f"parity[{cfg},grouped_mm]", o_loop, o_gmm)
    # grouped_mm falls back to segmented if the build rejects it; either way the
    # result must satisfy the tolerance contract vs the loop.
    assert rel <= _REL, f"{cfg}: grouped_mm rel {rel:.3e} > {_REL}"


# --------------------------------------------------------------------------- #
# 4 — ragged: some experts 0 tokens; one expert ALL tokens.                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
def test_ragged_zero_and_all(cfg):
    _require_stack()
    m, layer, d = _build(cfg, seed=4)
    act = _silu_act()
    T = 40
    # one_expert: expert 3 owns every pair, experts {0,1,2,4,5,6,7} own none.
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=11)
    torch.manual_seed(5)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_bat = m._apply_prefill_batched(layer, x, tw, ti, act, use_grouped_mm=False)
    rel = _report(f"ragged-one-expert[{cfg}]", o_loop, o_bat)
    assert rel <= _REL
    # Only expert 3's tokens are nonzero; but all T tokens route to expert 3
    # here, so every row is populated — a plain sanity check that nothing is
    # dropped and the shapes line up.
    assert o_bat.shape == (T, d["hidden"])

    # subset: experts >= E//2 get zero tokens (0-length segments skipped).
    ti2, tw2 = _routing(T, d["E"], 2, "subset", seed=12)
    o_loop2 = m._apply_prefill_loop(layer, x, tw2, ti2, act)
    o_bat2 = m._apply_prefill_batched(layer, x, tw2, ti2, act,
                                      use_grouped_mm=False)
    rel2 = _report(f"ragged-subset[{cfg}]", o_loop2, o_bat2)
    assert rel2 <= _REL


# --------------------------------------------------------------------------- #
# 5 — chunk boundaries: chunk=1, chunk=E, chunk=E//2+1 all match the loop and    #
#     each other (chunking only changes combine order = reassociation-class).    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8", "fp4v2"])
@pytest.mark.parametrize("chunk", [1, 5, 8, 1000])
def test_chunk_boundaries(cfg, chunk):
    _require_stack()
    m, layer, d = _build(cfg, seed=6)
    act = _silu_act()
    T, topk = 48, 4
    ti, tw = _routing(T, d["E"], topk, "uniform", seed=13)
    torch.manual_seed(7)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_loop = m._apply_prefill_loop(layer, x, tw, ti, act)
    o_bat = m._apply_prefill_batched(layer, x, tw, ti, act, chunk=chunk,
                                     use_grouped_mm=False)
    rel = _report(f"chunk[{cfg},chunk={chunk}]", o_loop, o_bat)
    assert rel <= _REL, f"{cfg}/chunk={chunk}: rel {rel:.3e} > {_REL}"


# --------------------------------------------------------------------------- #
# 6 — shared-experts present: apply() returns ROUTED-ONLY (shared args del'd)    #
#     and equals the batched routed compute; the env switch selects the path.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", ["fp8"])
def test_shared_experts_present(cfg, monkeypatch):
    _require_stack()
    m, layer, d = _build(cfg, seed=8)
    act = _silu_act()
    T, topk = 48, 2                                # T > 16 -> prefill dispatch
    ti, tw = _routing(T, d["E"], topk, "uniform", seed=15)
    torch.manual_seed(9)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    # Non-None shared-expert args must NOT change the routed-only return
    # (apply() del's them; the SharedExperts wrapper is run separately by the
    # vLLM runner — see the apply() docstring).
    shared = types.SimpleNamespace()
    shared_in = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV)

    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "batched")
    o_apply = m.apply(layer, x, tw, ti, shared, shared_in)
    o_bat = m._apply_prefill_batched(layer, x, tw, ti, act, use_grouped_mm=False)
    assert torch.equal(o_apply, o_bat), (
        "apply() with shared args != direct batched routed compute "
        "(shared expert leaked into the routed return?)")

    # The loop switch is honoured and stays within tolerance of the default.
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "loop")
    o_loop = m.apply(layer, x, tw, ti, shared, shared_in)
    rel = _report(f"shared-present[{cfg}]", o_loop, o_apply)
    assert rel <= _REL
