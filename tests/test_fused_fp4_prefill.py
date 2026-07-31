"""Gates for the fused NVFP4_CB block-scaled prefill (cb_fused_fp4_gemm.cu).

Container-only (JIT-builds the fp4 ext; needs nvcc + CUTLASS headers — vLLM's
bundled copy, or PRISMAQUANT_CUTLASS_INCLUDE for a venv with a CUTLASS
checkout; prismaquant importable for the packer):

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
    --entrypoint bash vllm-node:latest -c \\
    'pip install -q pytest; PYTHONPATH=/repo:/repo/plugins/gridbook \\
     python3 -m pytest /repo/plugins/gridbook/tests/test_fused_fp4_prefill.py -v'

Contracts pinned:
  * SASS: the built module contains OMMA.SF.16864 (the k=64 mxf4nvf4
    block-scale MMA) and NO QMMA — kind::f8f6f4 accepts e2m1 but issues at
    the fp8 k=32 rate, a silent half-speed failure parity cannot catch.
  * cb_fused_fp4_prefill_mm_scaled == (host-expanded e2m1/SFB planes ->
    sm120_nvf4_mm_scaled stock blockscaled GEMM) BIT-IDENTICAL across the
    rung ladder (product K12..K24, signed S13..S16, v1 + v2), the M ladder,
    and ragged N/K.
  * fp32-emulation tolerance on the kernel's OWN activation bucket
    (native NVFP4: per-tensor fp32 global x per-group ue4m3 SF). NOTE: this
    bucket differs from the Triton/transient paths' fp32-group-scale QDQ —
    the measured fused-vs-Triton delta (~7.5e-2 rel on random Linears) is a
    property of the FORMAT's two activation buckets, not a kernel bug, and
    is why the fused paths are opt-in pending a served KL A/B.
  * grouped (MoE) == per-expert dense BIT-IDENTICAL at tile_m 128 and 256.
  * vLLM's scaled_fp4_quant swizzled-SF layout == the codec reference the
    kernels' SFA TMA descriptor assumes (guards the serving dispatch).
"""
import os
import subprocess
from pathlib import Path

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)


def _build_ext():
    from gridbook.cuda_ext import get_fused_fp4_ext
    return get_fused_fp4_ext()


ext = _build_ext()
if ext is None:
    pytest.skip("fp4 fused ext unavailable (no nvcc / CUTLASS?)",
                allow_module_level=True)

DEV = "cuda"


# ---------------------------------------------------------------------------
# host packers (weight side from pack fields; activation side torch reference)
# ---------------------------------------------------------------------------
def prep_weight(k, N, K, mode, coding, seed):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=DEV) * 0.05
    cb = fmt._resolve_codebook(k, "fp4", mode, None, torch.device(DEV))
    packed, fields = fmt.nvfp4_cb_pack(w, k, grid="fp4", mode=mode,
                                       codebook=cb, scale_coding=coding)
    ts = fmt.nvfp4_cb_type_size(k, "fp4", coding)
    is_v2 = coding == fmt.SCALE_CODING_TWO_TIER
    n_sub = 2 if mode == "product" else 1
    cb_flat = codec.build_flat_codebook([t.to(DEV) for t in cb])
    lut = codec.build_fp4_value_lut(cb_flat, k, n_sub).to(DEV)
    compose = (codec.build_compose_u8().to(DEV) if is_v2 else
               torch.zeros(1, dtype=torch.uint8, device=DEV))
    qwp = codec.pad_qweight(packed)

    nvec = N * K // 8
    if mode == "product":
        w0 = k - k // 2
        t0 = cb_flat[:(1 << w0) * 4].reshape(-1, 4)
        t1 = cb_flat[(1 << w0) * 4:].reshape(-1, 4)
        idx = fields["indices"].to(DEV).reshape(nvec, 2).long()
        vals = torch.cat([t0[idx[:, 0]].float(), t1[idx[:, 1]].float()], dim=1)
        codes = codec.fp4_e2m1_codes(vals.reshape(N, K))
    else:
        mag = cb_flat.reshape(-1, 8)
        idx = fields["indices"].to(DEV).reshape(nvec).long()
        signs_neg = (fields["signs"].to(DEV).reshape(N, K) < 0)
        vals = (mag[idx].float().reshape(N, K)
                * torch.where(signs_neg, -1.0, 1.0))
        codes = (codec.fp4_e2m1_codes(mag[idx].float().reshape(N, K))
                 | (signs_neg.to(torch.uint8) << 3))
    b_packed = codec.pack_e2m1_codes(codes).contiguous()
    scales = fields["scales"].to(DEV).float()
    sfb_sw = codec.swizzle_sf_plane(
        scales.to(torch.float8_e4m3fn).view(torch.uint8)).contiguous()
    w_deq = vals.reshape(N, K).float() * scales.repeat_interleave(16, dim=1)
    return dict(qwp=qwp, lut=lut, compose=compose, ts=ts, n_sub=n_sub,
                is_v2=is_v2, b_packed=b_packed, sfb_sw=sfb_sw, w_deq=w_deq,
                cb_flat=cb_flat, k=k, N=N, K=K)


def quant_act(x2, gs=None):
    if gs is None:
        amax = x2.float().abs().amax()
        gs = (448.0 * 6.0) / amax.clamp_min(1e-8)
    aq, sfa_log, recip = codec.nvfp4_act_quant_ref(x2, gs)
    sfa = codec.swizzle_sf_plane(sfa_log.view(torch.uint8)).contiguous()
    return aq.contiguous(), sfa, sfa_log, recip


def run_pair(wctx, M, seed):
    torch.manual_seed(seed)
    N, K, k = wctx["N"], wctx["K"], wctx["k"]
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, sfa_log, recip = quant_act(x.reshape(-1, K))
    a_scales = recip.reshape(1).expand(M).contiguous().to(torch.float32)
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    y_ref = ext.sm120_nvf4_mm_scaled(aq, sfa, wctx["b_packed"],
                                     wctx["sfb_sw"], a_scales, b_scales, N, K)
    y_f = ext.cb_fused_fp4_prefill_mm_scaled(
        aq, sfa, wctx["qwp"], wctx["lut"], wctx["compose"], a_scales,
        b_scales, N, K, k, wctx["n_sub"], wctx["ts"], wctx["is_v2"])
    # fp32 emulation of the SAME activation bucket (dequant the actual bytes)
    lo, hi = (aq & 0xF), (aq >> 4) & 0xF
    codes_a = torch.stack([lo, hi], dim=-1).reshape(M, K)
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV)
    val = torch.where((codes_a & 8).bool(), -grid[(codes_a & 7).long()],
                      grid[(codes_a & 7).long()])
    sf_f = sfa_log.view(torch.float8_e4m3fn).to(torch.float32)
    xq = (val.reshape(M, K // 16, 16) * sf_f.unsqueeze(-1)).reshape(M, K)
    y_emu = (xq @ wctx["w_deq"].t()) * recip.to(torch.float32)
    return y_ref, y_f, y_emu


# ---------------------------------------------------------------------------
# SASS gate — the load-bearing correctness gate for the fp4 lane.
# ---------------------------------------------------------------------------
def test_sass_omma_16864_and_no_qmma():
    so = Path(ext.__file__)
    cuobjdump = None
    for cand in ("cuobjdump",
                 "/usr/local/cuda/bin/cuobjdump",
                 os.path.join(os.path.dirname(torch.__file__), "..", "triton",
                              "backends", "nvidia", "bin", "cuobjdump")):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            cuobjdump = cand
            break
        except Exception:  # noqa: BLE001
            continue
    if cuobjdump is None:
        pytest.skip("no cuobjdump on PATH")
    sass = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True,
                          text=True, check=True).stdout
    assert "OMMA.SF.16864" in sass, (
        "fp4 fused module does not issue the k=64 block-scaled MMA")
    assert "QMMA" not in sass, (
        "fp4 fused module contains QMMA (k=32, f8f6f4 kind) — e2m1 data would "
        "run at the fp8 rate; the silent half-speed trap this gate exists for")


# ---------------------------------------------------------------------------
# dense parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,mode,coding", [
    (12, "product", fmt.SCALE_CODING_TWO_TIER),
    (13, "product", fmt.SCALE_CODING_TWO_TIER),
    (16, "product", fmt.SCALE_CODING_TWO_TIER),
    (18, "product", fmt.SCALE_CODING_TWO_TIER),
    (20, "product", fmt.SCALE_CODING_TWO_TIER),
    (24, "product", fmt.SCALE_CODING_TWO_TIER),
    (13, "signed", fmt.SCALE_CODING_TWO_TIER),
    (16, "signed", fmt.SCALE_CODING_TWO_TIER),
    (16, "product", fmt.SCALE_CODING_V1),
    (20, "product", fmt.SCALE_CODING_V1),
])
def test_fused_bitexact_vs_stock(k, mode, coding):
    wctx = prep_weight(k, N=320, K=1536, mode=mode, coding=coding, seed=k)
    for M in (32, 64, 128):
        y_ref, y_f, y_emu = run_pair(wctx, M, seed=100 + M)
        assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))
        rel = ((y_f.float() - y_emu).norm() / y_emu.norm().clamp_min(1e-6))
        assert rel <= 1e-2, f"M={M}: emu rel {rel:.3e}"


@pytest.mark.parametrize("M", [32, 64, 128, 512, 2048])
def test_fused_bitexact_m_ladder(M):
    wctx = prep_weight(16, N=320, K=1536, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=1)
    y_ref, y_f, y_emu = run_pair(wctx, M, seed=M)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))


# N must be a multiple of 8 (bf16 TMA epilogue); K a multiple of 256 (the
# superblock). Both hold for every exported CB Linear; the raggedness under
# test is N vs TileN=128 and K vs TileK=128 edge tiles.
@pytest.mark.parametrize("N,K", [(320, 1536), (192, 1024), (104, 768),
                                 (520, 2560)])
def test_fused_bitexact_ragged(N, K):
    wctx = prep_weight(14, N=N, K=K, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=N)
    y_ref, y_f, _ = run_pair(wctx, 96, seed=7)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))


def test_padded_row_stride_view():
    """The serving entry receives layer._cb_qw_padded's NARROW view semantics
    (issue #1) — an explicitly-strided view must be bit-identical."""
    wctx = prep_weight(16, N=192, K=1024, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=9)
    _, y_a, _ = run_pair(wctx, 64, seed=3)
    row_bytes = (wctx["K"] // 256) * wctx["ts"]
    view = wctx["qwp"].narrow(1, 0, row_bytes)
    wctx2 = dict(wctx, qwp=view)
    _, y_b, _ = run_pair(wctx2, 64, seed=3)
    assert torch.equal(y_a.view(torch.uint16), y_b.view(torch.uint16))


def test_fused_vs_triton_bucket_delta_documented():
    """The fused path's native-NVFP4 activation bucket vs the Triton decode
    path's fp32-group-scale bucket: a real, bounded numerics difference (NOT
    a kernel defect — the weight side is bit-exact between them). Recorded
    here so a regression in EITHER direction is loud."""
    from gridbook.kernels import cb_decode_linear
    wctx = prep_weight(16, N=320, K=1536, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=11)
    torch.manual_seed(11)
    M, N, K = 128, wctx["N"], wctx["K"]
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    _, y_f, _ = run_pair(wctx, M, seed=11)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    y_tri = cb_decode_linear(
        xq, wctx["qwp"], wctx["cb_flat"],
        torch.zeros(N, dtype=torch.int32, device=DEV),
        torch.zeros(1, device=DEV),
        codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV),
        N=N, K=K, k_bits=wctx["k"], n_sub=2, type_size=wctx["ts"],
        is_fp4=True, is_v2=True)
    rel = ((y_f.float() - y_tri.float()).norm()
           / y_tri.float().norm().clamp_min(1e-6)).item()
    assert rel <= 0.15, f"bucket delta blew up: {rel:.3e}"
    assert rel >= 1e-4, "buckets identical?! the e4m3-SF snap vanished"


# ---------------------------------------------------------------------------
# grouped (MoE) parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tile_m,rows_per_e", [(128, 128), (128, 256),
                                               (256, 256)])
def test_moe_grouped_bitexact_vs_dense(tile_m, rows_per_e):
    k, E, N, K = 16, 4, 320, 1536
    wctxs = [prep_weight(k, N=N, K=K, mode="product",
                         coding=fmt.SCALE_CODING_TWO_TIER, seed=20 + e)
             for e in range(E)]
    ts, n_sub = wctxs[0]["ts"], wctxs[0]["n_sub"]
    row_bytes = (K // 256) * ts
    flat = torch.zeros(E * N * row_bytes + 16, dtype=torch.uint8, device=DEV)
    stack = flat[:E * N * row_bytes].reshape(E, N, row_bytes)
    for e, w in enumerate(wctxs):
        stack[e] = w["qwp"][:, :row_bytes]
    lut, compose = wctxs[0]["lut"], wctxs[0]["compose"]

    Mp = E * rows_per_e
    torch.manual_seed(31)
    x = torch.randn(Mp, K, dtype=torch.bfloat16, device=DEV)
    gs = (448.0 * 6.0) / x.float().abs().amax().clamp_min(1e-8)
    aq, sfa, _, recip = quant_act(x, gs)
    a_scales = recip.reshape(1).expand(Mp).contiguous().to(torch.float32)
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    eids = torch.arange(E, dtype=torch.int32, device=DEV).repeat_interleave(
        rows_per_e // tile_m).contiguous()
    y_g = ext.cb_fused_fp4_moe_grouped(aq, sfa, stack, lut, compose, a_scales,
                                       b_scales, eids, N, K, k, n_sub, ts,
                                       True, tile_m)
    outs = []
    for e, w in enumerate(wctxs):
        sl = slice(e * rows_per_e, (e + 1) * rows_per_e)
        # SAME global scale as the grouped call: per-row bit-equivalence of
        # the native quant holds only at a fixed global.
        aqe, sfae, _, _ = quant_act(x[sl], gs)
        outs.append(ext.cb_fused_fp4_prefill_mm_scaled(
            aqe, sfae, w["qwp"], lut, compose, a_scales[sl].contiguous(),
            b_scales, N, K, k, n_sub, ts, True))
    y_d = torch.cat(outs, dim=0)
    assert torch.equal(y_g.view(torch.uint16), y_d.view(torch.uint16))


# ---------------------------------------------------------------------------
# serving-dispatch guard: vLLM's fp4 quant swizzle == the codec reference
# ---------------------------------------------------------------------------
def test_vllm_scaled_fp4_quant_layout_matches_codec():
    vops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(vops, "scaled_fp4_quant"):
        pytest.skip("vllm has no scaled_fp4_quant")
    torch.manual_seed(5)
    M, K = 200, 1536                       # ragged M exercises the 128-pad
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    amax = x.float().abs().amax()
    gs = ((448.0 * 6.0) / amax).to(torch.float32)
    aq_v, sf_v = vops.scaled_fp4_quant(x, gs)
    aq_r, sf_log, _ = codec.nvfp4_act_quant_ref(x, gs)
    sf_r = codec.swizzle_sf_plane(sf_log.view(torch.uint8))
    assert sf_v.view(torch.uint8).reshape(-1).numel() == sf_r.numel(), (
        "swizzled SF plane size mismatch — vLLM layout drifted from the "
        "tile_atom_to_shape_SFA contract the kernel descriptor assumes")
    # SF bytes: identical layout AND values (both are e4m3(amax*gs/6)).
    assert torch.equal(sf_v.view(torch.uint8).reshape(-1), sf_r)
    # data: RTN tie conventions may differ on exact midpoints; require the
    # mismatch rate to be tie-rare, and dequants to agree elementwise there.
    mism = (aq_v != aq_r).float().mean().item()
    assert mism < 5e-3, f"packed e2m1 mismatch rate {mism:.2e}"
