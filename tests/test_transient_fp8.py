"""Correctness gate for the FP8_CB TRANSIENT-EXPANSION prefill path
(docs/nvfp4-cb-plan/serving-kernel.md §1a, prototype ii+).

Two parts, split by the ``-k`` selector so each runs where it can:

* Part A -- ``-k value_expand`` (build venv, NO vLLM):
    ``expand_cb_to_value`` decodes the codebook VALUE for every (n, j). It must
    equal ``nvfp4_cb_reconstruct / weight_scale`` (reconstruct = value * scale),
    and the tile must be exactly on the e4m3 grid (so the fp8 cast is lossless).
      PYTHONPATH=/home/rob/prismaquant:/home/rob/prismaquant/plugins/gridbook \\
        /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest \\
        plugins/gridbook/tests/test_transient_fp8.py -q -k value_expand

* Part B -- ``-k gemm`` (serving container, needs vLLM):
    the full transient GEMM (``scaled_fp8_quant`` + ``cutlass_scaled_mm`` over the
    expanded e4m3 tile) matches a fp32 dequant reference; and the per-layer
    transient is bounded + steady across forwards (INV-1).
      docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
        -v /home/rob/dq-runs/nvfp4-cb-phase0/serve:/artifacts \\
        --entrypoint bash vllm-node:latest -c 'pip install -e \\
        /repo/plugins/gridbook --no-deps -q; \\
        PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
        /repo/plugins/gridbook/tests/test_transient_fp8.py -q -k gemm'
"""
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

# The plugin sits outside the repo PYTHONPATH and needs triton (present in both
# the build venv and the serving container). Skip cleanly if unavailable.
codec = pytest.importorskip(
    "gridbook.codec",
    reason="gridbook plugin not importable (needs the plugin on "
           "PYTHONPATH + triton)")
_expand = pytest.importorskip("gridbook.expand")
expand_cb_to_value = _expand.expand_cb_to_value

# The independent Part-A reference uses prismaquant's emulation codec. Guard it:
# in the container the heavy import may be unavailable, and Part B does not need
# it (its reference is built from the already-validated expanded tile).
try:
    from prismaquant.nvfp4_cb_formats import (
        nvfp4_cb_reconstruct, nvfp4_cb_unpack,
    )
    _HAVE_PRISMAQUANT = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_PRISMAQUANT = False

ART = "fp8cb_k44"
# One MLP layer per orientation (down: N<K ; gate: N>K) -- both are the largest
# FP8_CB rung in the 0.6B artifact (N*K = 3.1M -> ~9 MiB transient).
PICK = ["model.layers.5.mlp.down_proj", "model.layers.5.mlp.gate_proj"]
DEV = "cuda"
_VALUE_REL = 1e-2      # Part A tolerance
_GEMM_REL = 2e-2       # Part B tolerance (fp8 act quant + bf16 accum)


def _serve_root() -> Path:
    """Locate the CB serve artifacts in either the venv (real path) or the
    container (bind-mounted at /artifacts)."""
    for p in (os.environ.get("CB_SERVE_ROOT"),
              "/home/rob/dq-runs/nvfp4-cb-phase0/serve",
              "/artifacts"):
        if p and (Path(p) / ART / "model.safetensors").exists():
            return Path(p)
    pytest.skip("CB serve artifacts (fp8cb_k44) not found")


def _load():
    d = _serve_root() / ART
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    tensors = load_file(str(d / "model.safetensors"))
    codebooks = load_file(str(d / cfg.get("codebook_file", "cb_codebooks.pqcb")))
    q2s = {}
    for g in cfg["config_groups"].values():
        for t in g["targets"]:
            q2s[t] = g["scheme"]
    return tensors, codebooks, q2s


def _subtables(scheme, codebooks):
    ref = scheme["codebook_ref"]
    names = ref if isinstance(ref, list) else [ref]
    return [codebooks[n].to(DEV).float() for n in names]


def _prep(qname):
    """Load a single-role FP8_CB layer and derive every tensor the transient
    path consumes. Returns a dict."""
    tensors, codebooks, q2s = _load()
    if qname not in q2s:
        pytest.skip(f"{qname} not a CB target in {ART}")
    sch = q2s[qname]
    assert sch["grid"] == "fp8", "transient path is FP8_CB only"
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    N = packed.shape[0]
    K = (packed.shape[1] // sch["type_size"]) * codec.SUPERBLOCK
    ws = tensors[qname + ".weight_scale"].to(DEV).float()          # (N,)
    subs = _subtables(sch, codebooks)
    cb_flat = codec.build_flat_codebook(subs)
    row_off = torch.zeros(N, dtype=torch.int32, device=DEV)        # single role
    qwp = codec.pad_qweight(packed)
    return dict(packed=packed, qwp=qwp, cb_flat=cb_flat, row_off=row_off,
                N=N, K=K, k=int(sch["k"]), n_sub=int(sch["n_sub"]),
                ts=int(sch["type_size"]), ws=ws, subs=subs)


# --------------------------------------------------------------------------- #
# Part A -- value expansion (no vLLM).                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("qname", PICK)
def test_value_expand_matches_reconstruct(qname):
    if not _HAVE_PRISMAQUANT:
        pytest.skip("prismaquant not importable; Part A reference unavailable")
    p = _prep(qname)
    N, K, k = p["N"], p["K"], p["k"]
    # Reference: reconstruct = value * weight_scale  =>  value = recon / scale.
    fields = nvfp4_cb_unpack(p["packed"], k, "fp8", "product", (N, K),
                             codebook=p["subs"], scales=p["ws"].reshape(-1, 1))
    recon = nvfp4_cb_reconstruct(fields, k, grid="fp8", mode="product",
                                 codebook=p["subs"])
    w_value_ref = recon / p["ws"].reshape(-1, 1)                   # (N, K)

    w_value = expand_cb_to_value(p["qwp"], p["cb_flat"], p["row_off"],
                                 N, K, k, p["n_sub"], p["ts"], is_fp4=False)
    assert w_value.shape == (N, K) and w_value.dtype == torch.bfloat16

    rel = ((w_value.float() - w_value_ref.float()).norm()
           / w_value_ref.float().norm().clamp_min(1e-6))
    assert rel <= _VALUE_REL, f"{qname}: value-expand rel err {rel:.4e}"

    # The transient must be exactly on the e4m3 grid -- this is what makes the
    # `.to(float8_e4m3fn)` cast in the prefill path lossless.
    cast_err = (w_value.float()
                - w_value.to(torch.float8_e4m3fn).float()).abs().max()
    assert cast_err <= 1e-3, (
        f"{qname}: expanded tile not on the e4m3 grid (max |Δ| {cast_err:.3e})")


@pytest.mark.parametrize("qname", PICK)
def test_fp8_direct_expand_bitexact(qname):
    """The fp8-direct expander (byte-gather, no bf16 intermediate) must produce
    byte-identical output to the old two-pass path (bf16 expand + lossless
    e4m3 cast) — same tile, a third of the expand-side traffic."""
    p = _prep(qname)
    N, K, k = p["N"], p["K"], p["k"]
    ref = expand_cb_to_value(p["qwp"], p["cb_flat"], p["row_off"],
                             N, K, k, p["n_sub"], p["ts"], is_fp4=False
                             ).to(torch.float8_e4m3fn)
    cb_fp8 = p["cb_flat"].to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    out = _expand.expand_cb_to_fp8(p["qwp"], cb_fp8, p["row_off"],
                                   N, K, k, p["n_sub"], p["ts"])
    assert out.dtype == torch.float8_e4m3fn and out.shape == (N, K)
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8)), (
        f"{qname}: fp8-direct expand bytes != bf16-expand+cast bytes")


def test_value_expand_rejects_fp4():
    """NVFP4_CB must stay on the Triton decode path -- the expander refuses it
    rather than silently producing a scale-less fp4 tile."""
    dummy = torch.zeros(4, 8, dtype=torch.uint8, device=DEV)
    cb = torch.zeros(16, dtype=torch.bfloat16, device=DEV)
    off = torch.zeros(4, dtype=torch.int32, device=DEV)
    with pytest.raises(NotImplementedError):
        expand_cb_to_value(dummy, cb, off, 4, 256, 16, 2, 80, is_fp4=True)


# --------------------------------------------------------------------------- #
# Part B -- transient fp8 GEMM (needs vLLM).                                   #
# --------------------------------------------------------------------------- #
def _scale_b_candidates(ws, N):
    """The two plausible per-channel scale_b shapes for cutlass_scaled_mm."""
    return [("[N,1]", ws.reshape(N, 1)), ("[1,N]", ws.reshape(1, N))]


@pytest.mark.parametrize("qname", PICK)
def test_transient_gemm_matches_fp32_dequant(qname):
    pytest.importorskip("vllm")
    import vllm._custom_ops as ops

    p = _prep(qname)
    N, K = p["N"], p["K"]
    ws = p["ws"]
    w_value = expand_cb_to_value(p["qwp"], p["cb_flat"], p["row_off"],
                                 N, K, p["k"], p["n_sub"], p["ts"], is_fp4=False)
    w_e4m3 = w_value.to(torch.float8_e4m3fn)

    torch.manual_seed(0)
    M = 64                                       # prefill regime (> threshold)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    xq, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

    # fp32 dequant reference: (act_deq) @ (weight_deq).t(), weight_deq = the
    # e4m3 codebook value * per-channel weight_scale (== the true reconstruct).
    ref = (xq.float() * sa) @ (w_e4m3.float() * ws.reshape(-1, 1)).t()
    ref_norm = ref.norm().clamp_min(1e-6)

    # Determine which per-channel scale_b shape cutlass broadcasts correctly.
    matched = []
    for name, sb in _scale_b_candidates(ws, N):
        try:
            out = ops.cutlass_scaled_mm(xq, w_e4m3.t(), sa, sb,
                                        torch.bfloat16)
        except Exception as e:                   # noqa: BLE001 - report + skip
            print(f"{qname}: scale_b {name} raised: {type(e).__name__}: {e}")
            continue
        rel = (out.float() - ref).norm() / ref_norm
        print(f"{qname}: scale_b {name} -> rel {rel:.4e}")
        if rel <= _GEMM_REL:
            matched.append((name, float(rel)))
    assert matched, (f"{qname}: no scale_b shape reproduced the fp32 dequant "
                     f"reference within {_GEMM_REL}")


def test_transient_gemm_memory_steady():
    """INV-1: the per-forward transient is bounded (one layer) and does not
    grow across forwards. Reports the peak transient MiB for the largest layer."""
    pytest.importorskip("vllm")
    import vllm._custom_ops as ops

    p = _prep(PICK[0])                            # down_proj: largest FP8_CB rung
    N, K = p["N"], p["K"]
    ws2 = p["ws"].reshape(N, 1)                   # [N,1] per-channel scale_b
    torch.manual_seed(0)
    M = 64
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)

    def one_forward():
        w_value = expand_cb_to_value(p["qwp"], p["cb_flat"], p["row_off"],
                                     N, K, p["k"], p["n_sub"], p["ts"],
                                     is_fp4=False)
        w_e4m3 = w_value.to(torch.float8_e4m3fn)
        xq, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        out = ops.cutlass_scaled_mm(xq, w_e4m3.t(), sa, ws2, torch.bfloat16)
        del w_value, w_e4m3, xq, sa               # free the transient
        return out

    for _ in range(3):                            # warmup (kernels, autotune)
        del_out = one_forward(); del del_out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    live = []
    for _ in range(20):
        out = one_forward()
        live.append(torch.cuda.memory_allocated())   # 'out' still alive
        del out
    torch.cuda.synchronize()

    peak_transient_mib = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    growth_mib = (live[-1] - live[0]) / 2 ** 20
    print(f"{PICK[0]} N={N} K={K}: peak transient {peak_transient_mib:.2f} MiB; "
          f"live-alloc growth over 20 forwards {growth_mib:.4f} MiB")

    # No per-forward growth (a leak would climb monotonically).
    assert growth_mib <= 1.0, f"per-forward memory grew {growth_mib:.3f} MiB (leak?)"
    # Peak is bounded by ~one layer's expansion (W_value bf16 + W_e4m3 e4m3 =
    # 3*N*K bytes) + the small activation tensors -- not a model-wide dense
    # expansion. Generous slack for act/out buffers + allocator rounding.
    one_layer_mib = 3 * N * K / 2 ** 20
    assert peak_transient_mib < one_layer_mib + 32, (
        f"peak transient {peak_transient_mib:.1f} MiB exceeds the one-layer "
        f"bound {one_layer_mib:.1f}+32 MiB (INV-1 violation?)")
