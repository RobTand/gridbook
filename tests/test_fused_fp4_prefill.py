"""Gates for the fused NVFP4_CB block-scaled prefill (cb_fused_fp4_gemm.cu).

Container-only (JIT-builds the fp4 ext; needs nvcc + CUTLASS headers — vLLM's
bundled copy, or PRISMAQUANT_CUTLASS_INCLUDE for a venv with a CUTLASS
checkout; prismaquant importable for the packer):

  docker run --rm --gpus all -v /home/rob/gridbook:/gridbook \\
    -v /home/rob/prismaquant:/prismaquant \\
    --entrypoint bash vllm-node:latest -c \\
    'pip install -q pytest; PYTHONPATH=/gridbook:/prismaquant \\
     python3 -m pytest /gridbook/tests/test_fused_fp4_prefill.py -v'

Contracts pinned:
  * SASS: the built module contains OMMA.SF.16864 (the k=64 mxf4nvf4
    block-scale MMA) and NO QMMA — kind::f8f6f4 accepts e2m1 but issues at
    the fp8 k=32 rate, a silent half-speed failure parity cannot catch.
  * cb_fused_fp4_prefill_mm_scaled == (host-expanded e2m1/SFB planes ->
    sm120_nvf4_mm_scaled stock blockscaled GEMM) BIT-IDENTICAL across the
    full fused-eligible rung ladder (product K12..K24, signed S12..S20,
    every rung under both v1 and v2), the M ladder,
    and ragged N/K.
  * fp32-emulation tolerance on the kernel's OWN activation bucket
    (native NVFP4: per-tensor fp32 global x per-group ue4m3 SF). NOTE: this
    bucket differs from the Triton/transient paths' fp32-group-scale QDQ —
    the measured fused-vs-Triton delta (~7.5e-2 rel on random Linears) is a
    property of the FORMAT's two activation buckets, not a kernel bug. The
    model A/B is now red, so the fused paths remain explicit opt-ins pending
    the audit's full reconsideration gates.
  * grouped (MoE) == per-expert dense BIT-IDENTICAL at tile_m 128 and 256.
  * vLLM's scaled_fp4_quant swizzled-SF layout == the codec reference the
    kernels' SFA TMA descriptor assumes for a representative random sample
    (guards serving dispatch, not arbitrary packed-byte equivalence). Exact
    midpoint/underflow operator oracles live in test_nvfp4_act_quant_ref.py.
"""
import os
import re
import shutil
import subprocess
import sys
import types
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
                cb_flat=cb_flat, scales=scales, k=k, N=N, K=K)


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
def test_sass_fused_symbols_issue_omma_16864_and_no_qmma():
    so = Path(ext.__file__)
    cuobjdumps = []
    for cand in ("cuobjdump",
                 "/usr/local/cuda/bin/cuobjdump",
                 os.path.join(os.path.dirname(torch.__file__), "..", "triton",
                              "backends", "nvidia", "bin", "cuobjdump")):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            resolved = shutil.which(cand) or str(Path(cand).resolve())
            if resolved not in cuobjdumps:
                cuobjdumps.append(resolved)
        except Exception:  # noqa: BLE001
            continue
    assert cuobjdumps, (
        "enablement attestation requires cuobjdump; none is executable")
    # CUDA's split toolkit makes cuobjdump invoke nvdisasm as a separate
    # executable.  Serving images may intentionally omit it from PATH while
    # shipping compatible copies inside Triton packages; CUDA 12.8's copy,
    # for example, cannot decode an sm_121 cubin, while tokenspeed_triton's
    # CUDA 13.1 copy can. Try every available candidate and accept only a
    # successful disassembly, preserving the instruction gate rather than
    # skipping or weakening it because the first nvdisasm is too old.
    site_root = Path(torch.__file__).resolve().parent.parent
    nv_candidates = [
        shutil.which("nvdisasm"),
        site_root / "tokenspeed_triton/backends/nvidia/bin/nvdisasm",
        site_root / "triton/backends/nvidia/bin/nvdisasm",
    ]
    errors = []
    sass = None
    nvdisasms = []
    for cand in nv_candidates:
        if cand is not None and Path(cand).is_file():
            resolved = str(Path(cand).resolve())
            if resolved not in nvdisasms:
                nvdisasms.append(resolved)
    assert nvdisasms, (
        "enablement attestation requires an nvdisasm capable of decoding the "
        "built cubin; none is available")
    for cuobjdump in cuobjdumps:
        for nvdisasm in nvdisasms:
            env = os.environ.copy()
            env["PATH"] = f"{Path(nvdisasm).parent}:{env.get('PATH', '')}"
            env["NVDISASM_PATH"] = str(Path(nvdisasm).parent)
            proc = subprocess.run(
                [cuobjdump, "-sass", str(so)], capture_output=True, text=True,
                env=env)
            if proc.returncode == 0:
                sass = proc.stdout
                break
            errors.append(
                f"cuobjdump={cuobjdump}, nvdisasm={nvdisasm}: "
                f"{proc.stderr.strip()}")
        if sass is not None:
            break
    assert sass is not None, (
        "no available nvdisasm could decode the fused cubin: "
        + "; ".join(errors))
    # Attribute the instruction to the two concrete fused device kernels. A
    # module-wide OMMA search is insufficient because the stock/reference
    # kernel is deliberately linked into this same extension and supplies its
    # own OMMA instructions even if a fused kernel silently scalarizes.
    markers = list(re.finditer(
        r"(?m)^\s*Function : (?P<symbol>\S+)\s*$", sass))
    functions = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(sass)
        functions.append((marker.group("symbol"), sass[marker.end():end]))
    fused = [(symbol, body) for symbol, body in functions
             if "MainloopSm120CbFusedFp4TmaWarpSpecialized" in symbol]
    assert len(fused) == 2, (
        "expected exactly the concrete fused TileM=128 and TileM=256 device "
        f"kernels, found {len(fused)}")
    tile256 = [(symbol, body) for symbol, body in fused
               if "ILi256EE" in symbol]
    tile128 = [(symbol, body) for symbol, body in fused
               if "ILi256EE" not in symbol]
    assert len(tile128) == 1 and len(tile256) == 1, (
        "could not uniquely identify concrete fused TileM=128/256 symbols")
    for label, (_symbol, body) in (
        ("dense/grouped TileM=128", tile128[0]),
        ("grouped TileM=256", tile256[0]),
    ):
        assert "OMMA.SF.16864" in body, (
            f"{label} fused kernel does not issue the k=64 block-scaled MMA")
        assert "QMMA" not in body, (
            f"{label} fused kernel issues QMMA instead of native NVFP4 OMMA")
    assert "QMMA" not in sass, (
        "fp4 fused module contains QMMA (k=32, f8f6f4 kind) — e2m1 data would "
        "run at the fp8 rate; the silent half-speed trap this gate exists for")


# ---------------------------------------------------------------------------
# dense parity
# ---------------------------------------------------------------------------
_FUSED_ELIGIBLE_RUNG_CASES = [
    (k, mode, coding)
    for coding in (fmt.SCALE_CODING_V1, fmt.SCALE_CODING_TWO_TIER)
    for mode, rungs in (
        ("product", range(12, 25)),
        ("signed", range(12, 21)),
    )
    for k in rungs
]


@pytest.mark.parametrize("k,mode,coding", _FUSED_ELIGIBLE_RUNG_CASES)
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


def test_fused_multilut_distinct_merged_roles_bitexact_vs_stock():
    """Each 128-row N tile must stage the LUT named by its merged role.

    The second block flips every E2M1 value's sign while retaining the exact
    packed index and scale planes.  A kernel that accidentally reuses LUT zero
    for both tiles therefore cannot pass this bitwise stock-NVFP4 oracle.
    """
    k, role_n, K, M = 16, 128, 1024, 64
    wctx = prep_weight(k, N=role_n, K=K, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=109)
    packed = torch.cat((wctx["qwp"], wctx["qwp"]), dim=0).contiguous()
    lut0 = wctx["lut"]
    # Product LUT entries are u16 nibble quads. XORing bit 3 of every nibble
    # produces the matching sign-flipped codebook without changing indices.
    lut1 = torch.bitwise_xor(lut0, torch.tensor(0x88, device=DEV,
                                                dtype=torch.uint8))
    luts = torch.cat((lut0, lut1)).contiguous()
    tile_ids = torch.tensor([0, 1], dtype=torch.int32, device=DEV)

    b_packed = torch.cat((wctx["b_packed"], wctx["b_packed"] ^ 0x88),
                         dim=0).contiguous()
    scales = torch.cat((wctx["scales"], wctx["scales"]), dim=0)
    sfb = codec.swizzle_sf_plane(
        scales.to(torch.float8_e4m3fn).view(torch.uint8)).contiguous()
    N = 2 * role_n

    torch.manual_seed(110)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, _, recip = quant_act(x)
    a_scales = recip.reshape(1).expand(M).contiguous().float()
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    y_ref = ext.sm120_nvf4_mm_scaled(
        aq, sfa, b_packed, sfb, a_scales, b_scales, N, K)
    y_fused = ext.cb_fused_fp4_prefill_mm_scaled(
        aq, sfa, packed, luts, wctx["compose"], a_scales, b_scales,
        N, K, k, wctx["n_sub"], wctx["ts"], True, tile_ids)
    assert torch.equal(y_fused.view(torch.uint16), y_ref.view(torch.uint16))
    assert not torch.equal(y_fused[:, :role_n], y_fused[:, role_n:])


def test_fused_multilut_persistent_cta_reuse_bitexact_vs_stock():
    """A persistent CTA must reload the LUT when its next N tile changes.

    Two N tiles are too small to exercise CUTLASS's persistent scheduler:
    each resident CTA sees at most one role and a lifetime ``lut_resident``
    boolean appears correct.  Two hundred fifty-six alternating role tiles
    exceed the one-large-CTA-per-SM residency of every qualified Blackwell
    target and deterministically expose stale-LUT reuse.
    """
    k, role_n, ntiles, K, M = 16, 128, 256, 256, 128
    wctx = prep_weight(k, N=role_n, K=K, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=1801)
    packed = wctx["qwp"].repeat(ntiles, 1).contiguous()
    lut0 = wctx["lut"]
    lut1 = torch.bitwise_xor(
        lut0, torch.tensor(0x88, device=DEV, dtype=torch.uint8))
    luts = torch.cat((lut0, lut1)).contiguous()
    tile_ids = (torch.arange(ntiles, dtype=torch.int32, device=DEV) & 1) \
        .contiguous()

    b_tiles = [wctx["b_packed"] if tile % 2 == 0
               else wctx["b_packed"] ^ 0x88
               for tile in range(ntiles)]
    b_packed = torch.cat(b_tiles, dim=0).contiguous()
    scales = wctx["scales"].repeat(ntiles, 1)
    sfb = codec.swizzle_sf_plane(
        scales.to(torch.float8_e4m3fn).view(torch.uint8)).contiguous()
    N = role_n * ntiles

    torch.manual_seed(1802)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, _, recip = quant_act(x)
    a_scales = recip.reshape(1).expand(M).contiguous().float()
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    y_ref = ext.sm120_nvf4_mm_scaled(
        aq, sfa, b_packed, sfb, a_scales, b_scales, N, K)
    y_fused = ext.cb_fused_fp4_prefill_mm_scaled(
        aq, sfa, packed, luts, wctx["compose"], a_scales, b_scales,
        N, K, k, wctx["n_sub"], wctx["ts"], True, tile_ids)
    assert torch.equal(y_fused.view(torch.uint16), y_ref.view(torch.uint16))


def test_rowwise_quantized_fused_gemm_bitexact_vs_stock():
    """Per-row activation reciprocals must index the same rows as SFA/A.

    The rowwise quantizer and fused GEMM each have independent live oracles,
    but a constant ``a_scales`` vector cannot expose an epilogue row-indexing
    error at their integration boundary.  Exercise a wide range of row
    amplitudes, feed the quantizer's heterogeneous reciprocals directly into
    both collectives, and require exact BF16 agreement with stock NVFP4.
    """
    k, M, N, K = 16, 128, 256, 1024
    wctx = prep_weight(k, N=N, K=K, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=1811)
    torch.manual_seed(1812)
    amplitudes = torch.logspace(
        -3, 3, M, dtype=torch.float32, device=DEV).reshape(M, 1)
    x = (torch.randn(M, K, dtype=torch.float32, device=DEV)
         * amplitudes).to(torch.bfloat16).contiguous()
    aq, sfa, a_scales = ext.cb_nvfp4_quantize_rows(x, 448.0)
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)

    # This also makes the test load-bearing: a broadcast scalar would let a
    # row-indexing bug pass while pretending to cover the rowwise contract.
    assert torch.unique(a_scales.view(torch.int32)).numel() > M // 2
    y_ref = ext.sm120_nvf4_mm_scaled(
        aq, sfa, wctx["b_packed"], wctx["sfb_sw"], a_scales, b_scales,
        N, K)
    y_fused = ext.cb_fused_fp4_prefill_mm_scaled(
        aq, sfa, wctx["qwp"], wctx["lut"], wctx["compose"], a_scales,
        b_scales, N, K, k, wctx["n_sub"], wctx["ts"], wctx["is_v2"])
    assert torch.equal(y_fused.view(torch.uint16), y_ref.view(torch.uint16))


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


def test_bindings_reject_invalid_tensor_contracts():
    """Malformed public-op inputs fail before a CUDA launch."""
    k, N, K = 16, 128, 256
    wctx = prep_weight(k, N=N, K=K, mode="product",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=61)
    torch.manual_seed(62)
    x = torch.randn(32, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, _, recip = quant_act(x)
    a_scales = recip.reshape(1).expand(x.shape[0]).contiguous().float()
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    dense_args = (
        aq, sfa, wctx["qwp"], wctx["lut"], wctx["compose"], a_scales,
        b_scales, N, K, k, wctx["n_sub"], wctx["ts"], True,
    )
    row_bytes = (K // 256) * wctx["ts"]

    bad_sfa = sfa.to(torch.float16)
    with pytest.raises(RuntimeError, match="sfa must be contiguous uint8"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            dense_args[0], bad_sfa, *dense_args[2:])
    with pytest.raises(RuntimeError, match="compose must be contiguous CUDA"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:4], dense_args[4].cpu(), *dense_args[5:])
    multi_lut = torch.cat((wctx["lut"], wctx["lut"])).contiguous()
    with pytest.raises(RuntimeError, match="multiple value LUT blocks require"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:3], multi_lut, *dense_args[4:])
    with pytest.raises(RuntimeError, match="lut_tile_ids must be contiguous"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:3], multi_lut, *dense_args[4:],
            torch.zeros(1, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="lut_tile_ids must be contiguous"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:3], multi_lut, *dense_args[4:],
            torch.zeros(2, dtype=torch.int32, device=DEV))
    short_visible_row = wctx["qwp"].narrow(1, 0, row_bytes - 1)
    with pytest.raises(RuntimeError, match="packed visible row width"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:2], short_visible_row, *dense_args[3:])

    # A tensor view can outlive an external resize of its backing storage.
    # Keep the visible shape/stride otherwise valid and prove the binding
    # catches that hostile state before the producer can read the final byte.
    row_stride = row_bytes + codec.PAD_BYTES
    storage_need = (N - 1) * row_stride + row_bytes
    unsafe_storage = torch.empty(
        storage_need, dtype=torch.uint8, device=DEV)
    storage_short_view = torch.as_strided(
        unsafe_storage, (N, row_bytes), (row_stride, 1))
    unsafe_storage.untyped_storage().resize_(storage_need - 1)
    with pytest.raises(RuntimeError, match="backing storage is too small"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args[:2], storage_short_view, *dense_args[3:])
    # The old optional debug pointer had no capacity/device check and allowed a
    # short tensor to be overwritten by the fixed-size smem dump. The new final
    # argument is the checked LUT tile map; a further tensor is still refused.
    with pytest.raises(TypeError):
        ext.cb_fused_fp4_prefill_mm_scaled(
            *dense_args, torch.zeros(1, dtype=torch.int32, device=DEV),
            torch.zeros(1, dtype=torch.uint8, device=DEV))

    E, tile_m = 2, 128
    stack = torch.empty(E, N, row_bytes, dtype=torch.uint8, device=DEV)
    stack[0].copy_(wctx["qwp"][:, :row_bytes])
    stack[1].copy_(wctx["qwp"][:, :row_bytes])
    xg = torch.randn(tile_m, K, dtype=torch.bfloat16, device=DEV)
    aqg, sfag, _, recipg = quant_act(xg)
    grouped_args = (
        aqg, sfag, stack, wctx["lut"], wctx["compose"],
        recipg.reshape(1).expand(tile_m).contiguous().float(), b_scales,
        torch.zeros(1, dtype=torch.int32, device=DEV), N, K, k,
        wctx["n_sub"], wctx["ts"], True, tile_m,
    )
    with pytest.raises(RuntimeError, match="sfa must be contiguous uint8"):
        ext.cb_fused_fp4_moe_grouped(
            grouped_args[0], grouped_args[1].to(torch.float16),
            *grouped_args[2:])
    with pytest.raises(RuntimeError, match="expert_ids must be contiguous"):
        ext.cb_fused_fp4_moe_grouped(
            *grouped_args[:7], grouped_args[7].cpu(), *grouped_args[8:])

    # A one-GPU enablement box cannot construct a different-CUDA-device input;
    # exercise the explicit same-device contract whenever CI provides one.
    if torch.cuda.device_count() > 1:
        foreign_lut = wctx["lut"].to("cuda:1")
        with pytest.raises(RuntimeError, match="same CUDA device as a"):
            ext.cb_fused_fp4_prefill_mm_scaled(
                *dense_args[:3], foreign_lut, *dense_args[4:])


def test_binding_rejects_first_oversized_signed_lut():
    """S21 is decoder-compatible but exceeds the fused 16-KiB LUT carve."""

    k, N, K = 21, 128, 256
    wctx = prep_weight(k, N=N, K=K, mode="signed",
                       coding=fmt.SCALE_CODING_TWO_TIER, seed=63)
    x = torch.randn(32, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, _, recip = quant_act(x)
    with pytest.raises(RuntimeError, match="value LUT exceeds the smem carve"):
        ext.cb_fused_fp4_prefill_mm_scaled(
            aq, sfa, wctx["qwp"], wctx["lut"], wctx["compose"],
            recip.reshape(1).expand(x.shape[0]).contiguous().float(),
            torch.ones(N, dtype=torch.float32, device=DEV),
            N, K, k, wctx["n_sub"], wctx["ts"], True,
        )


def _run_invalid_expert_id_child(bad_eid):
    """Launch one malformed route in a disposable CUDA process.

    The producer's device-side trap intentionally poisons this process's CUDA
    context. The parent test therefore invokes this helper via ``runpy`` in a
    subprocess and requires CUDA's illegal-instruction failure.
    """
    k, E, N, K, tile_m = 16, 2, 128, 256, 128
    wctxs = [prep_weight(k, N=N, K=K, mode="product",
                         coding=fmt.SCALE_CODING_TWO_TIER, seed=90 + e)
             for e in range(E)]
    row_bytes = (K // 256) * wctxs[0]["ts"]
    stack = torch.empty(E, N, row_bytes, dtype=torch.uint8, device=DEV)
    for e, wctx in enumerate(wctxs):
        stack[e].copy_(wctx["qwp"][:, :row_bytes])
    x = torch.randn(tile_m, K, dtype=torch.bfloat16, device=DEV)
    aq, sfa, _, recip = quant_act(x)
    ext.cb_fused_fp4_moe_grouped(
        aq, sfa, stack, wctxs[0]["lut"], wctxs[0]["compose"],
        recip.reshape(1).expand(tile_m).contiguous().float(),
        torch.ones(N, dtype=torch.float32, device=DEV),
        torch.tensor([bad_eid], dtype=torch.int32, device=DEV),
        N, K, k, wctxs[0]["n_sub"], wctxs[0]["ts"], True, tile_m)
    torch.cuda.synchronize()


@pytest.mark.parametrize("bad_eid", [-2, 2], ids=["below-pad", "past-E"])
def test_moe_grouped_invalid_expert_id_traps_fail_closed(bad_eid):
    child_code = (
        "import runpy; "
        f"ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"ns['_run_invalid_expert_id_child']({bad_eid})"
    )
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", child_code], capture_output=True, text=True,
        env=env, timeout=180)
    combined = (proc.stdout + "\n" + proc.stderr).lower()
    assert proc.returncode != 0, (
        f"out-of-range expert id {bad_eid} reached a successful launch")
    # With CUDA_LAUNCH_BLOCKING the CUTLASS adapter may translate the trap to
    # kErrorInternal before the runtime reports it as an illegal instruction.
    # Both are the same fail-closed launch; an unchecked/clamped route returns
    # successfully (which the return-code assertion above rejects).
    assert ("illegal instruction" in combined or
            "fused fp4 moe launch failed: error internal" in combined), (
        "invalid expert ID did not hit the explicit device-side trap; output: "
        + combined[-2000:])


@pytest.mark.parametrize(
    "N,K,tile_m,capture_graph",
    [(2048, 3072, 128, False), (3072, 1024, 256, True)],
    ids=["jason-stage1-m128", "jason-stage2-m256-graph"],
)
def test_moe_grouped_jason_stage_shapes_stream_and_graph(
        N, K, tile_m, capture_graph):
    """Gate the two Laguna expert projection shapes used in final serving.

    The tile schedule is deliberately non-monotonic and selects the final
    expert twice.  This catches expert-stride bugs that sequential toy IDs or
    expert zero alone cannot expose.  Both shapes run on a non-default stream;
    stage 2 additionally proves capture and replay of the grouped entry.
    """
    k, E = 16, 4
    wctxs = [prep_weight(k, N=N, K=K, mode="product",
                         coding=fmt.SCALE_CODING_TWO_TIER, seed=70 + e)
             for e in range(E)]
    ts, n_sub = wctxs[0]["ts"], wctxs[0]["n_sub"]
    row_bytes = (K // 256) * ts
    stack = torch.empty(E, N, row_bytes, dtype=torch.uint8, device=DEV)
    for e, wctx in enumerate(wctxs):
        stack[e].copy_(wctx["qwp"][:, :row_bytes])

    route = [E - 1, 1, 0, E - 1]
    assert len(set(route)) > 1 and max(route) == E - 1
    Mp = len(route) * tile_m
    torch.manual_seed(81)
    x = torch.randn(Mp, K, dtype=torch.bfloat16, device=DEV)
    gs = (448.0 * 6.0) / x.float().abs().amax().clamp_min(1e-8)
    aq, sfa, _, recip = quant_act(x, gs)
    a_scales = recip.reshape(1).expand(Mp).contiguous().to(torch.float32)
    b_scales = torch.ones(N, dtype=torch.float32, device=DEV)
    eids = torch.tensor(route, dtype=torch.int32, device=DEV)
    lut, compose = wctxs[0]["lut"], wctxs[0]["compose"]

    # Precompute each dense reference's activation layout on the producer
    # stream, then make the non-default launch stream wait on all inputs.
    dense_inputs = []
    for tile, expert in enumerate(route):
        sl = slice(tile * tile_m, (tile + 1) * tile_m)
        aqe, sfae, _, _ = quant_act(x[sl], gs)
        dense_inputs.append((expert, sl, aqe, sfae))

    launch_args = (aq, sfa, stack, lut, compose, a_scales, b_scales, eids,
                   N, K, k, n_sub, ts, True, tile_m)
    producer = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    assert worker.cuda_stream != producer.cuda_stream
    worker.wait_stream(producer)
    with torch.cuda.stream(worker):
        y_grouped = ext.cb_fused_fp4_moe_grouped(*launch_args)
        dense_out = []
        for expert, sl, aqe, sfae in dense_inputs:
            dense_out.append(ext.cb_fused_fp4_prefill_mm_scaled(
                aqe, sfae, wctxs[expert]["qwp"], lut, compose,
                a_scales[sl].contiguous(), b_scales, N, K, k, n_sub, ts,
                True))
        y_dense = torch.cat(dense_out, dim=0)
    worker.synchronize()
    assert torch.equal(y_grouped.view(torch.uint16),
                       y_dense.view(torch.uint16))

    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=worker):
            y_graph = ext.cb_fused_fp4_moe_grouped(*launch_args)
        graph.replay()
        worker.synchronize()
        assert torch.equal(y_graph.view(torch.uint16),
                           y_grouped.view(torch.uint16))
        graph.replay()
        worker.synchronize()
        assert torch.equal(y_graph.view(torch.uint16),
                           y_grouped.view(torch.uint16))


def _synthetic_route(case, T, E):
    """Independent token-major top-2 routes for the full-MoE oracle."""
    token = torch.arange(T, dtype=torch.long)
    if case == "empty":
        # Only two experts are hit; all other expert stacks must consume no
        # blocks and must not perturb the result.
        ids = torch.stack((torch.full_like(token, 1),
                           torch.full_like(token, E - 2)), dim=1)
    elif case == "hotspot":
        # One expert crosses the TileM boundary while the other slot fans out.
        ids = torch.stack((torch.full_like(token, E - 1), token % (E - 1)),
                          dim=1)
    elif case == "nonmonotonic":
        pattern = torch.tensor(
            [[E - 1, 1], [0, E - 2], [E - 2, 2], [1, E - 1], [2, 0]],
            dtype=torch.long)
        ids = pattern[token % pattern.shape[0]]
    elif case == "balanced":
        # A single round-robin sequence split into top-2 rows keeps total
        # expert counts within one for every T, including TileM +/- 1.  Since
        # E is odd, adjacent entries never repeat within a token.
        ids = (torch.arange(2 * T, dtype=torch.long) % E).reshape(T, 2)
    else:  # pragma: no cover - the test's literal case table is exhaustive
        raise ValueError(case)
    assert bool((ids[:, 0] != ids[:, 1]).all())
    # Exact binary fractions avoid a router-weight representation confound.
    weights = torch.tensor([0.625, 0.375], dtype=torch.float32).expand(T, 2).clone()
    return ids, weights


def _independent_padded_segments(topk_ids, topk_weights, E, tile_m, device):
    """Build expert-major padded rows without using cb_grouped_pad_routing.

    Iterating token then slot reproduces the public loop contract's stable
    within-expert order, but the construction is deliberately independent of
    the grouped path's sort/cumsum/index arithmetic.
    """
    T, topk = topk_ids.shape
    destinations, weights, segments = [], [], []
    cursor = 0
    for expert in range(E):
        pairs = [
            (token, slot)
            for token in range(T)
            for slot in range(topk)
            if int(topk_ids[token, slot]) == expert
        ]
        if not pairs:
            continue
        padded = ((len(pairs) + tile_m - 1) // tile_m) * tile_m
        destinations.extend(token for token, _slot in pairs)
        weights.extend(float(topk_weights[token, slot])
                       for token, slot in pairs)
        destinations.extend([T] * (padded - len(pairs)))
        weights.extend([0.0] * (padded - len(pairs)))
        segments.append((expert, cursor, cursor + padded))
        cursor += padded
    return (
        torch.tensor(destinations, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        segments,
    )


def _dense_native_role_reference(
        activations, segments, packed, lut, compose, N, K, k, n_sub,
        type_size, global_scale):
    """Run one dense native-NVFP4 GEMM per independently routed expert."""
    vops = pytest.importorskip("vllm._custom_ops")
    outputs = []
    b_scales = torch.ones(N, dtype=torch.float32, device=activations.device)
    for expert, start, end in segments:
        rows = activations[start:end].contiguous()
        aq, sfa = vops.scaled_fp4_quant(rows, global_scale)
        a_scales = (1.0 / global_scale).reshape(1).expand(
            rows.shape[0]).contiguous()
        outputs.append(ext.cb_fused_fp4_prefill_mm_scaled(
            aq, sfa.view(torch.uint8).reshape(-1), packed[expert], lut,
            compose, a_scales, b_scales, N, K, k, n_sub, type_size, True))
    return torch.cat(outputs, dim=0)


def _decode_native_fp4_activation(aq, sfa, M, K, global_scale):
    """Decode vLLM's actual E2M1 + swizzled UE4M3 payload to BF16.

    This is the B-side activation oracle: it consumes the bytes returned by
    ``scaled_fp4_quant`` rather than recomputing quantization. Indexing the
    swizzled plane with the descriptor's inverse logical offsets makes every
    scale value used here the one supplied to the native MMA.
    """
    packed = aq.view(torch.uint8).reshape(M, -1)
    assert packed.shape[1] * 2 == K
    codes = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=-1) \
        .reshape(M, K)
    mag = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=aq.device)
    values = torch.where((codes & 8).bool(),
                         -mag[(codes & 7).long()],
                         mag[(codes & 7).long()])

    offsets = codec.sf_swizzle_offsets(M, K // 16, aq.device)
    sf_flat = sfa.view(torch.uint8).reshape(-1)
    assert int(offsets.max()) < sf_flat.numel()
    sf_bytes = sf_flat.index_select(0, offsets.reshape(-1)).reshape(
        M, K // 16).contiguous()
    sf = sf_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    decoded = values.reshape(M, K // 16, 16) * sf.unsqueeze(-1)
    decoded.mul_((1.0 / global_scale).to(torch.float32))
    return decoded.reshape(M, K).to(torch.bfloat16)


def _segmented_decoded_weight_bf16_gemm(activations, segments, wctxs):
    """One ordinary BF16 GEMM per expert using independently decoded weights."""
    outputs = []
    for expert, start, end in segments:
        weight = wctxs[expert]["w_deq"].to(torch.bfloat16)
        outputs.append(activations[start:end].to(torch.bfloat16) @ weight.t())
    return torch.cat(outputs, dim=0)


def _stagewise_fp4_decomposition(
        activations, segments, wctxs, packed, lut, compose, tile_m):
    """Return A/B/C/D for one projection stage on identical routed rows."""
    vops = pytest.importorskip("vllm._custom_ops")
    M, K = activations.shape
    N = wctxs[0]["N"]
    k = wctxs[0]["k"]
    n_sub = wctxs[0]["n_sub"]
    type_size = wctxs[0]["ts"]
    global_scale = ((448.0 * 6.0) /
                    activations.float().abs().amax().clamp_min(1e-12)).float()

    # A: the legacy fp32-group-scale activation bucket, followed by a normal
    # BF16 GEMM over decoded BF16 weights.
    legacy_qdq = codec.fp4_group16_act_qdq(activations).to(torch.bfloat16)
    a_out = _segmented_decoded_weight_bf16_gemm(
        legacy_qdq, segments, wctxs)

    # B: decode the exact native quantizer payload back to BF16, then use the
    # same BF16 GEMM as A. Thus A->B isolates activation representation.
    aq, sfa = vops.scaled_fp4_quant(activations, global_scale)
    native_qdq = _decode_native_fp4_activation(
        aq, sfa, M, K, global_scale)
    b_out = _segmented_decoded_weight_bf16_gemm(
        native_qdq, segments, wctxs)

    # C: native activation bytes and native block-scaled MMA, one independent
    # dense invocation per expert. B->C isolates native weight/accumulation.
    c_out = _dense_native_role_reference(
        activations, segments, packed, lut, compose, N, K, k, n_sub,
        type_size, global_scale)

    # D: the production grouped native invocation over the identical bytes.
    # C->D therefore isolates only grouped dispatch/expert addressing.
    expert_ids = []
    for expert, start, end in segments:
        assert (end - start) % tile_m == 0
        expert_ids.extend([expert] * ((end - start) // tile_m))
    expert_ids = torch.tensor(
        expert_ids, dtype=torch.int32, device=activations.device)
    recip = (1.0 / global_scale).reshape(1)
    d_out = ext.cb_fused_fp4_moe_grouped(
        aq, sfa.view(torch.uint8).reshape(-1), packed, lut, compose,
        recip.expand(M).contiguous(),
        torch.ones(N, dtype=torch.float32, device=activations.device),
        expert_ids, N, K, k, n_sub, type_size, True, tile_m)
    return a_out, b_out, c_out, d_out


def _relative_l2(before, after):
    """Relative L2 delta for an explicitly directed before -> after edge."""
    return ((after.float() - before.float()).norm()
            / before.float().norm().clamp_min(1e-6)).item()


def test_real_moe_grouped_full_routing_bitexact_vs_per_expert_dense_native(
        monkeypatch):
    """Full two-stage MoE routing oracle across the production TileM cliffs.

    The candidate is the real Gridbook method: stable expert routing, padding,
    native activation quantization, both grouped projection kernels, activation,
    router weighting and index_add combine. The reference independently builds
    expert segments and invokes only the dense native-NVFP4 kernel per expert.
    Equality is bitwise BF16, so no routing, padding, stage-2 expert identity or
    combine drift can hide behind a numerical tolerance.
    """
    from gridbook.moe import (
        MoEActivation,
        PrismaQuantCBMoEMethod,
        apply_moe_activation,
    )

    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "1")
    k, n_sub, E, hidden, inter = 16, 2, 5, 256, 256
    type_size = fmt.nvfp4_cb_type_size(
        k, "fp4", fmt.SCALE_CODING_TWO_TIER)
    w13_ctx = [
        prep_weight(k, N=2 * inter, K=hidden, mode="product",
                    coding=fmt.SCALE_CODING_TWO_TIER, seed=120 + expert)
        for expert in range(E)
    ]
    w2_ctx = [
        prep_weight(k, N=hidden, K=inter, mode="product",
                    coding=fmt.SCALE_CODING_TWO_TIER, seed=140 + expert)
        for expert in range(E)
    ]
    w13_row_bytes = (hidden // 256) * type_size
    w2_row_bytes = (inter // 256) * type_size
    w13 = torch.empty(E, 2 * inter, w13_row_bytes,
                      dtype=torch.uint8, device=DEV)
    w2 = torch.empty(E, hidden, w2_row_bytes,
                     dtype=torch.uint8, device=DEV)
    for expert in range(E):
        w13[expert].copy_(w13_ctx[expert]["qwp"][:, :w13_row_bytes])
        w2[expert].copy_(w2_ctx[expert]["qwp"][:, :w2_row_bytes])

    # One shared codebook per layer is the production union-find contract.
    cb_flat = w13_ctx[0]["cb_flat"]
    assert all(torch.equal(ctx["cb_flat"], cb_flat)
               for ctx in (*w13_ctx, *w2_ctx))
    lut = codec.build_fp4_value_lut(cb_flat, k, n_sub).to(DEV)
    compose = codec.build_compose_u8(codec.TWO_TIER_SUB_TABLE).to(DEV)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.prefix = "test.full_routing"
    method.is_fp4 = True
    method.is_v2 = True
    method.k = k
    method.n_sub = n_sub
    method.type_size = type_size
    method.has_static_fp4_activation = True
    method._sub_table = codec.TWO_TIER_SUB_TABLE
    # This is a routing/grouped-dispatch oracle, so give both the production
    # method and the independent dense reference the same fixed, artifact-like
    # activation contract.  Deriving a fresh batch amax here would exercise the
    # retired dynamic contract and make the test bypass the static safety gate.
    stage1_scale = torch.tensor([512.0], dtype=torch.float32, device=DEV)
    stage2_scale = torch.tensor([256.0], dtype=torch.float32, device=DEV)
    layer = types.SimpleNamespace(
        _cb_E=E,
        _cb_hidden=hidden,
        _cb_inter=inter,
        _cb_flat=cb_flat,
        _cb_fp4_input_global_scale_w13=stage1_scale,
        _cb_fp4_input_global_scale_w2=stage2_scale,
        w13_cb_qweight=w13,
        w2_cb_qweight=w2,
    )
    act = MoEActivation.from_str("silu")
    cases = ("empty", "hotspot", "nonmonotonic", "balanced")

    for tile_m in (128, 256):
        for T in (tile_m - 1, tile_m, tile_m + 1):
            for case_index, case in enumerate(cases):
                ids_cpu, weights_cpu = _synthetic_route(case, T, E)
                counts = torch.bincount(ids_cpu.reshape(-1), minlength=E)
                if case == "empty":
                    assert int((counts == 0).sum()) == E - 2
                elif case == "hotspot":
                    assert int(counts[E - 1]) == T
                elif case == "balanced":
                    assert int(counts.max() - counts.min()) <= 1

                torch.manual_seed(1000 + tile_m + T + case_index)
                x = torch.randn(T, hidden, dtype=torch.bfloat16, device=DEV)
                topk_ids = ids_cpu.to(device=DEV, dtype=torch.int32)
                topk_weights = weights_cpu.to(DEV)
                grouped = method._apply_prefill_grouped_fused_fp4(
                    layer, x, topk_weights, topk_ids, act, tile_m=tile_m)
                assert grouped is not None

                dest, route_weights, segments = _independent_padded_segments(
                    ids_cpu, weights_cpu, E, tile_m, x.device)
                assert segments and all((end - start) % tile_m == 0
                                        for _expert, start, end in segments)
                x_pad = torch.cat((x, x.new_zeros((1, hidden)))) \
                    .index_select(0, dest).contiguous()

                gs1 = stage1_scale
                gate_up = _dense_native_role_reference(
                    x_pad, segments, w13, lut, compose, 2 * inter, hidden,
                    k, n_sub, type_size, gs1)
                intermediate = torch.empty(
                    (x_pad.shape[0], inter), dtype=gate_up.dtype, device=DEV)
                apply_moe_activation(act, intermediate, gate_up)

                gs2 = stage2_scale
                dense = _dense_native_role_reference(
                    intermediate, segments, w2, lut, compose, hidden, inter,
                    k, n_sub, type_size, gs2)
                dense.mul_(route_weights[:, None].to(dense.dtype))
                reference = torch.zeros(
                    (T + 1, hidden), dtype=x.dtype, device=DEV)
                reference.index_add_(0, dest, dense.to(reference.dtype))
                reference = reference[:T]

                same = torch.equal(grouped.view(torch.uint16),
                                   reference.view(torch.uint16))
                assert same, (
                    f"full routing mismatch case={case} T={T} tile_m={tile_m}; "
                    f"bit mismatches="
                    f"{int((grouped.view(torch.uint16) != reference.view(torch.uint16)).sum())}"
                )


def test_stagewise_activation_accumulation_decomposition(
        record_testsuite_property):
    """Decompose the activation-bucket, MMA, and grouping deltas by MoE stage.

    A->B is evidence, not an enablement gate: it is the intentional fp32-scale
    versus UE4M3-scale activation-contract change and requires served quality
    validation. B->C uses the pre-existing 1e-2 native-fused emulation bound.
    C->D must remain bitwise BF16 zero.
    """
    from gridbook.moe import MoEActivation, apply_moe_activation

    k, n_sub, E, hidden, inter = 16, 2, 3, 256, 256
    type_size = fmt.nvfp4_cb_type_size(
        k, "fp4", fmt.SCALE_CODING_TWO_TIER)
    w13_ctx = [
        prep_weight(k, N=2 * inter, K=hidden, mode="product",
                    coding=fmt.SCALE_CODING_TWO_TIER, seed=220 + expert)
        for expert in range(E)
    ]
    w2_ctx = [
        prep_weight(k, N=hidden, K=inter, mode="product",
                    coding=fmt.SCALE_CODING_TWO_TIER, seed=240 + expert)
        for expert in range(E)
    ]
    assert all(ctx["n_sub"] == n_sub for ctx in (*w13_ctx, *w2_ctx))
    cb_flat = w13_ctx[0]["cb_flat"]
    assert all(torch.equal(ctx["cb_flat"], cb_flat)
               for ctx in (*w13_ctx, *w2_ctx))
    lut = codec.build_fp4_value_lut(cb_flat, k, n_sub).to(DEV)
    compose = codec.build_compose_u8(codec.TWO_TIER_SUB_TABLE).to(DEV)

    def _packed_stack(contexts, K):
        row_bytes = (K // 256) * type_size
        return torch.stack(
            [ctx["qwp"][:, :row_bytes] for ctx in contexts]).contiguous()

    w13 = _packed_stack(w13_ctx, hidden)
    w2 = _packed_stack(w2_ctx, inter)
    reports = []
    for tile_m in (128, 256):
        T = tile_m + 1
        ids_cpu, weights_cpu = _synthetic_route("hotspot", T, E)
        dest, _route_weights, segments = _independent_padded_segments(
            ids_cpu, weights_cpu, E, tile_m, torch.device(DEV))
        torch.manual_seed(2000 + tile_m)
        x = torch.randn(T, hidden, dtype=torch.bfloat16, device=DEV)
        x_pad = torch.cat((x, x.new_zeros((1, hidden)))) \
            .index_select(0, dest).contiguous()

        stage1 = _stagewise_fp4_decomposition(
            x_pad, segments, w13_ctx, w13, lut, compose, tile_m)
        # A common native stage-1 result feeds stage 2, so that stage-2 deltas
        # are not contaminated by propagation of an earlier branch's error.
        intermediate = torch.empty(
            (x_pad.shape[0], inter), dtype=stage1[2].dtype, device=DEV)
        apply_moe_activation(MoEActivation.SILU, intermediate, stage1[2])
        stage2 = _stagewise_fp4_decomposition(
            intermediate, segments, w2_ctx, w2, lut, compose, tile_m)

        for stage_name, outputs in (("stage1", stage1), ("stage2", stage2)):
            a_out, b_out, c_out, d_out = outputs
            assert all(bool(torch.isfinite(out).all()) for out in outputs)
            a_to_b = _relative_l2(a_out, b_out)
            b_to_c = _relative_l2(b_out, c_out)
            c_to_d = _relative_l2(c_out, d_out)
            bit_mismatches = int((c_out.view(torch.uint16) !=
                                  d_out.view(torch.uint16)).sum())
            prefix = f"tile{tile_m}_{stage_name}"
            record_testsuite_property(
                f"{prefix}_a_to_b_rel_l2", f"{a_to_b:.9e}")
            record_testsuite_property(
                f"{prefix}_b_to_c_rel_l2", f"{b_to_c:.9e}")
            record_testsuite_property(
                f"{prefix}_c_to_d_rel_l2", f"{c_to_d:.9e}")
            record_testsuite_property(
                f"{prefix}_c_to_d_bit_mismatches", bit_mismatches)
            reports.append(
                f"{prefix}: A->B={a_to_b:.9e}, B->C={b_to_c:.9e}, "
                f"C->D={c_to_d:.9e}, C/D bits={bit_mismatches}")
            # This is the established native-fused-vs-emulation bound used by
            # test_fused_bitexact_vs_stock; do not create a new looser gate.
            assert b_to_c <= 1e-2, (
                f"{prefix}: native accumulation delta {b_to_c:.3e}")
            assert bit_mismatches == 0, (
                f"{prefix}: grouped native differs from dense native")
            assert c_to_d == 0.0

    print("NVFP4 stagewise decomposition | " + " | ".join(reports),
          flush=True)


# ---------------------------------------------------------------------------
# Representative serving-dispatch guard: vLLM's fp4 quant swizzle == the
# codec reference. This is not an exhaustive numerical oracle: vLLM uses
# hardware approximate reciprocals for arbitrary nonzero scale factors.
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
    # Data: vLLM preserves the input sign when a value rounds to zero, while
    # the codec reference canonicalizes both +0 and -0 to nibble zero.  The
    # hardware values are identical, so compare decoded E2M1 values rather
    # than treating the sign bit of zero as a numerical mismatch.  Any other
    # code difference remains a hard failure.
    def _decode(packed):
        u8 = packed.view(torch.uint8)
        codes = torch.stack((u8 & 0xF, (u8 >> 4) & 0xF), dim=-1).reshape(M, K)
        mag = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV)
        return torch.where((codes & 8).bool(),
                           -mag[(codes & 7).long()],
                           mag[(codes & 7).long()])

    assert torch.equal(_decode(aq_v), _decode(aq_r)), (
        "vLLM packed E2M1 values differ from the codec reference")
