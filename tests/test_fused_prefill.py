"""Gates for the fused decode-in-prologue FP8_CB GEMM (cb_fused_gemm.cu).

Container-only (JIT-builds two extensions; needs nvcc + /artifacts):

  docker run --rm --gpus all -v /home/rob/gridbook:/gridbook \\
    -v /home/rob/dq-runs/nvfp4-cb-phase0/serve:/artifacts \\
    --entrypoint bash vllm-node:latest -c \\
    'pip install -q pytest; PYTHONPATH=/gridbook \\
     python3 -m pytest /gridbook/tests/test_fused_prefill.py -v'

Contracts pinned:
  * cb_fused_prefill_mm == (cb_expand_fp8 tile -> fork64 passthrough GEMM)
    BIT-IDENTICAL — the decoded smem tile is provably the same bytes through
    the same MMA config. All four rungs + non-multiple N/M + real artifact.
  * chunked-overlap driver (N-chunks through the SAME fork config) ==
    monolithic fork GEMM BIT-IDENTICAL — the property cutlass_scaled_mm
    lacks (its heuristics reconfigure on narrow N), which is what makes
    overlap serving-safe with OUR kernel only.
"""
import json
from pathlib import Path

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
from gridbook.cuda_ext import get_ext, get_fused_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

gemv_ext = get_ext()
if gemv_ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)", allow_module_level=True)


fused = get_fused_ext()
if fused is None:
    pytest.skip("fused ext unavailable", allow_module_level=True)

DEV = "cuda"

# K1.2. Every per-rung gate below is parametrized over what the LOADED MODULE
# says it compiled, not over a literal ladder. That is the point of the change:
# a hand-written `[36, 40, 44, 48]` tested four of the six compiled rungs and
# could not notice a seventh appearing or a fourth disappearing. Reading the
# binding makes the coverage follow the kernel by construction.
COMPILED_RUNGS = sorted(int(k) for k in fused.cb_fused_kbits())


# ---------------------------------------------------------------------------
# K1.2 — the RUNG SURFACE itself.
#
# The product FP8-CB ladder is every integer k in [28, 48]; this lane backs the
# multiples of 4, because type_size = 4k must be a 16-byte TMA box multiple AND
# the mainloop's single CbSubW = k/4 sub-table width is the format's real layout
# only then (csrc/cb_gemv.cu `SubSplit` splits raggedly otherwise). Three things
# then have to agree, forever: the kernel's compiled set, the Python law
# dispatch gates on, and what dispatch actually accepts.
# ---------------------------------------------------------------------------
def test_compiled_rungs_are_exactly_the_law():
    """The module's compiled set == codec.FP8_FUSED_KBITS.

    This is the join that makes the cheap host-side gate safe. `linear.py` and
    `moe.py` filter on the LAW before touching the extension (asking the module
    first would force a JIT build at first forward merely to learn a rung is
    ineligible), and only then confirm against the module. If the law were ever
    a superset, dispatch would offer an uncompiled rung; if a subset, it would
    silently miss a compiled one. Both are caught here.
    """
    assert tuple(COMPILED_RUNGS) == codec.FP8_FUSED_KBITS


def test_law_is_the_multiples_of_four_in_the_product_range():
    """...and the law is the FORMAT's law, stated independently of the kernel."""
    assert COMPILED_RUNGS == [k for k in range(28, 49) if 4 * k % 16 == 0]
    # The ragged-split half of the law: n_sub=4 gives one uniform sub-table
    # width only on these rungs (csrc/cb_gemv.cu SubSplit: base + (i < extra)).
    for k in COMPILED_RUNGS:
        assert len({k // 4 + (1 if i < k % 4 else 0) for i in range(4)}) == 1


@pytest.mark.parametrize("k", [29, 30, 31, 37, 47])
def test_rungs_off_the_law_are_rejected_by_the_law(k):
    """A product rung this lane cannot serve must be REFUSED, with the reason.

    These are real production rungs (the published 27B artifact's K36-K47
    ladder contains several), so the failure an operator can hit must name the
    law and say the rung is still served elsewhere — not "unsupported k_bits".
    """
    assert not codec.fp8_fused_rung_supported(k)
    packed, cb8 = _synth(32, N=256, K=1024, seed=5)   # shapes are irrelevant
    xq = torch.zeros(16, 1024, device=DEV).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError) as exc:
        fused.cb_fused_prefill_mm(xq, packed, cb8, 256, 1024, k)
    message = str(exc.value)
    assert "rung law" in message
    assert "16-byte" in message and "CbSubW" in message


def test_grouped_tile_sizes_are_reported_per_rung():
    """Every compiled rung offers at least TileM=128, and the smem-bound tiles
    are reported per rung rather than as a union — so a caller enumerating
    `cb_fused_moe_tile_sizes_for_kbits` can never select an uncompiled cell."""
    union = set(int(t) for t in fused.cb_fused_moe_tile_sizes())
    for k in COMPILED_RUNGS:
        tiles = [int(t) for t in fused.cb_fused_moe_tile_sizes_for_kbits(k)]
        assert tiles and tiles == sorted(tiles)
        assert set(tiles) <= union
        assert 128 in tiles
        # The measured smem ceiling: TileM=256 fits only the two lowest rungs
        # (256/k32 lands on EXACTLY 101,376 B — the zero-margin cell).
        assert (256 in tiles) == (k <= 32)


def test_smem_report_covers_every_compiled_rung():
    """The per-rung smem report is generated from the same list the switches
    are, so it cannot fall behind the dispatch (it used to be hand-written)."""
    flat = [int(v) for v in fused.smem_report_rungs()]
    rows = [flat[i:i + 4] for i in range(0, len(flat), 4)]
    assert [r[0] for r in rows[:-1]] == COMPILED_RUNGS
    ceiling = rows[-1][1]
    for k, total, lut_bytes, subs in rows[:-1]:
        assert 0 < total <= ceiling
        assert lut_bytes == subs * (1 << (k // 4)) * 2


def _synth(k, N, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts = 4 * k
    packed = torch.randint(0, 256, (N, (K // 256) * ts), generator=g,
                           dtype=torch.uint8).to(DEV)
    subs = [(torch.randn(1 << (k // 4), 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    cb8 = codec.build_flat_codebook(subs).to(torch.float8_e4m3fn).view(
        torch.uint8).contiguous()
    return packed, cb8


def _ref_and_fused(packed, cb8, N, K, k, M, seed=1):
    ts = 4 * k
    qwp = codec.pad_qweight(packed)
    off = torch.zeros(N, dtype=torch.int32, device=DEV)
    W = gemv_ext.cb_expand_fp8(qwp, cb8, off, N, K, k, 4, ts)
    torch.manual_seed(seed)
    xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    y_ref = fused.sm120_fp8_mm_fork64(xq, W)
    y_f = fused.cb_fused_prefill_mm(xq, packed, cb8, N, K, k)
    return y_ref, y_f


@pytest.mark.parametrize("k", COMPILED_RUNGS)
def test_fused_bitexact_synth(k):
    packed, cb8 = _synth(k, N=256, K=1024, seed=k)
    y_ref, y_f = _ref_and_fused(packed, cb8, 256, 1024, k, M=384)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))


def test_fused_bitexact_ragged_shapes():
    packed, cb8 = _synth(44, N=320, K=1536, seed=7)
    y_ref, y_f = _ref_and_fused(packed, cb8, 320, 1536, 44, M=1400)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))


def test_fused_bitexact_real_artifact():
    d = Path("/artifacts/fp8cb_k44")
    if not (d / "model.safetensors").exists():
        pytest.skip("fp8cb_k44 artifact not mounted")
    from safetensors.torch import load_file
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    tensors = load_file(str(d / "model.safetensors"))
    codebooks = load_file(str(d / cfg.get("codebook_file", "cb_codebooks.pqcb")))
    qname = "model.layers.5.mlp.gate_proj"
    sch = next(g["scheme"] for g in cfg["config_groups"].values()
               if qname in g["targets"])
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    N = packed.shape[0]
    K = (packed.shape[1] // sch["type_size"]) * 256
    ref = sch["codebook_ref"]
    subs = [codebooks[n].to(DEV).float()
            for n in (ref if isinstance(ref, list) else [ref])]
    cb8 = codec.build_flat_codebook(subs).to(torch.float8_e4m3fn).view(
        torch.uint8).contiguous()
    y_ref, y_f = _ref_and_fused(packed, cb8, N, K, int(sch["k"]), M=512)
    assert torch.equal(y_ref.view(torch.uint16), y_f.view(torch.uint16))


def test_chunked_overlap_bitexact_vs_monolithic():
    """N-chunked expand+GEMM through the SAME fork config must equal the
    monolithic result bit-for-bit (per-output K-reduction is config-fixed)."""
    k, N, K, M = 48, 8192, 5120, 640
    packed, cb8 = _synth(k, N=N, K=K, seed=3)
    ts = 4 * k
    qwp = codec.pad_qweight(packed)
    off = torch.zeros(N, dtype=torch.int32, device=DEV)
    torch.manual_seed(3)
    xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    W = gemv_ext.cb_expand_fp8(qwp, cb8, off, N, K, k, 4, ts)
    y_mono = fused.sm120_fp8_mm_fork(xq, W)

    es = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    # the side stream must order after main's prior writes (off/xq/packed)
    es.wait_stream(main)
    outs = []
    rows = 4096
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        with torch.cuda.stream(es):
            Wc = gemv_ext.cb_expand_fp8(qwp[r0:r1], cb8, off[r0:r1],
                                        r1 - r0, K, k, 4, ts)
            ev = torch.cuda.Event()
            ev.record(es)
        main.wait_event(ev)
        Wc.record_stream(main)
        outs.append(fused.sm120_fp8_mm_fork(xq, Wc))
    y_ch = torch.cat(outs, dim=-1)
    torch.cuda.synchronize()
    assert torch.equal(y_mono.view(torch.uint16), y_ch.view(torch.uint16))


# ---------------------------------------------------------------------------
# Scaled epilogue (cb_fused_prefill_mm_scaled): the per-token activation scale
# and the per-channel weight scale are applied in the kernel's fp32 EVT
# epilogue, rounding ONCE to bf16 — the rounding ORDER ops.cutlass_scaled_mm
# uses. The old unscaled entry + a python multiply rounds twice, which moved
# served 27B prompt logprobs by mean 0.10 / max 0.86 nats. The GATE reference
# is therefore cutlass_scaled_mm, with the fp32 GEMM as ground truth.
# ---------------------------------------------------------------------------
def _scaled_case(k, N, K, M, seed):
    packed, cb8 = _synth(k, N=N, K=K, seed=seed)
    ts = 4 * k
    qwp = codec.pad_qweight(packed)
    off = torch.zeros(N, dtype=torch.int32, device=DEV)
    W = gemv_ext.cb_expand_fp8(qwp, cb8, off, N, K, k, 4, ts)
    torch.manual_seed(seed)
    xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    sa = (torch.rand(M, 1, device=DEV) * 0.02 + 0.01).float()
    ws = (torch.rand(N, 1, device=DEV) * 0.02 + 0.01).float()
    y_scaled = fused.cb_fused_prefill_mm_scaled(
        xq, packed, cb8, sa.reshape(-1).contiguous(),
        ws.reshape(-1).contiguous(), N, K, k)
    y_unscaled = (fused.cb_fused_prefill_mm(xq, packed, cb8, N, K, k).float()
                  * sa * ws.reshape(1, -1)).to(torch.bfloat16)
    ref32 = (xq.float() @ W.float().t()) * sa * ws.reshape(1, -1)
    return xq, W, sa, ws, y_scaled, y_unscaled, ref32


@pytest.mark.parametrize("k", COMPILED_RUNGS)
def test_fused_scaled_matches_fp32_reference(k):
    """In-epilogue scaling must be a correct-rounding of the fp32 product, and
    never worse than the round-then-scale path it replaces."""
    _, _, _, _, y_s, y_u, ref32 = _scaled_case(k, N=256, K=1024, M=96, seed=k)
    err_s = (y_s.float() - ref32).abs().max().item()
    err_u = (y_u.float() - ref32).abs().max().item()
    scale = ref32.abs().max().item()
    assert err_s <= 0.005 * scale, (err_s, scale)
    assert err_s <= err_u + 1e-6, (err_s, err_u)


def test_fused_scaled_matches_cutlass_scaled_mm():
    """The promotion gate: same value as the shipping comparator path
    (ops.cutlass_scaled_mm) to within one bf16 ulp — same scale semantics,
    same rounding order, only the K-reduction schedule differs."""
    ops = pytest.importorskip("vllm._custom_ops")
    k, N, K, M = 44, 320, 1536, 96
    xq, W, sa, ws, y_s, y_u, ref32 = _scaled_case(k, N, K, M, seed=11)
    y_c = ops.cutlass_scaled_mm(xq, W.t(), sa, ws, torch.bfloat16, None)
    tol = 4.0 * torch.finfo(torch.bfloat16).eps * ref32.abs().max().item()
    assert (y_s.float() - y_c.float()).abs().max().item() <= tol


# ---------------------------------------------------------------------------
# Padded-row-stride acceptance (issue #1, 2026-07-25).
#
# `process_weights_after_loading` no longer keeps a second contiguous copy of
# the packed weight: `layer.cb_qweight.data` is now a NARROW VIEW of the
# 16-byte-padded buffer, so what reaches these entries has
# `stride(0) == row_bytes + codec.PAD_BYTES`, not `row_bytes`. The kernel
# already takes the row stride explicitly (`packed.stride(0)` -> the packed-B
# TMA descriptor's row stride, cb_fused_gemm.cu run_fused*/to_underlying_
# arguments), and check_fused_inputs never required contiguity — only
# `stride(1) == 1`, `stride(0) % 16 == 0` and `stride(0) >= (K/256)*4*k`, all of
# which the 16-byte pad preserves. These gates make that a tested contract
# rather than a read of the source: same bytes in, bit-identical bytes out.
# ---------------------------------------------------------------------------

def _padded_view(packed):
    view = codec.pad_qweight(packed).narrow(1, 0, packed.shape[1])
    assert view.stride(0) == packed.shape[1] + codec.PAD_BYTES
    assert view.stride(0) % 16 == 0 and view.stride(1) == 1
    assert torch.equal(view, packed)          # same bytes, different stride
    return view


@pytest.mark.parametrize("k", COMPILED_RUNGS)
def test_fused_padded_view_bitexact_vs_contiguous(k):
    N, K, M = 320, 1536, 96
    packed, cb8 = _synth(k, N=N, K=K, seed=100 + k)
    torch.manual_seed(k)
    xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    y_c = fused.cb_fused_prefill_mm(xq, packed, cb8, N, K, k)
    y_v = fused.cb_fused_prefill_mm(xq, _padded_view(packed), cb8, N, K, k)
    assert torch.equal(y_c.view(torch.uint16), y_v.view(torch.uint16))


@pytest.mark.parametrize("k", COMPILED_RUNGS)
def test_fused_scaled_padded_view_bitexact_vs_contiguous(k):
    """The default-on mid-M serving entry (linear.py:_apply_inline)."""
    N, K, M = 320, 1536, 96
    packed, cb8 = _synth(k, N=N, K=K, seed=200 + k)
    torch.manual_seed(k)
    xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    sa = (torch.rand(M, device=DEV) * 0.02 + 0.01).float().contiguous()
    ws = (torch.rand(N, device=DEV) * 0.02 + 0.01).float().contiguous()
    y_c = fused.cb_fused_prefill_mm_scaled(xq, packed, cb8, sa, ws, N, K, k)
    y_v = fused.cb_fused_prefill_mm_scaled(
        xq, _padded_view(packed), cb8, sa, ws, N, K, k)
    assert torch.equal(y_c.view(torch.uint16), y_v.view(torch.uint16))
