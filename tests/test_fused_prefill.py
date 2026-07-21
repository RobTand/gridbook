"""Gates for the fused decode-in-prologue FP8_CB GEMM (cb_fused_gemm.cu).

Container-only (JIT-builds two extensions; needs nvcc + /artifacts):

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
    -v /home/rob/dq-runs/nvfp4-cb-phase0/serve:/artifacts \\
    --entrypoint bash vllm-node:latest -c \\
    'pip install -q pytest; PYTHONPATH=/repo:/repo/plugins/gridbook \\
     python3 -m pytest /repo/plugins/gridbook/tests/test_fused_prefill.py -v'

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
import os
from pathlib import Path

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
from gridbook.cuda_ext import get_ext  # noqa: E402

gemv_ext = get_ext()
if gemv_ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)", allow_module_level=True)


def _build_fused():
    from torch.utils.cpp_extension import load
    build = os.path.join(os.path.expanduser("~"), ".cache", "pq-fused-build")
    os.makedirs(build, exist_ok=True)
    cut = ("/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
           "fmha_sm100/cutlass")
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "csrc")
    return load(name="pq_cb_fused",
                sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
                extra_include_paths=[f"{cut}/include",
                                     f"{cut}/tools/util/include", src_dir],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                build_directory=build, verbose=False)


try:
    fused = _build_fused()
except Exception as exc:  # pragma: no cover - env dependent
    pytest.skip(f"fused ext build failed: {exc}", allow_module_level=True)

DEV = "cuda"


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


@pytest.mark.parametrize("k", [36, 40, 44, 48])
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
