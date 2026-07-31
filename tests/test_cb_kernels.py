"""Correctness gate for the CB Triton decode-GEMM, on the REAL exported 0.6B
tensors (docs/lanes/nvfp4-cb/serving-kernel.md prototype (i)).

Run:
  PYTHONPATH=/home/rob/prismaquant:/home/rob/prismaquant/plugins/gridbook \
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest \
    plugins/gridbook/tests/test_cb_kernels.py -v

Two checks per artifact/Linear:
  * exact-match unpack — the kernel's byte-window codeword extraction is
    bit-identical to `nvfp4_cb_unpack`;
  * decode-GEMM matches `nvfp4_cb_reconstruct @ x` to <=1e-2 rel (bf16 accum).
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from prismaquant.nvfp4_cb_formats import (
    _bit_split, nvfp4_cb_reconstruct, nvfp4_cb_unpack,
)

# The plugin package sits outside the repo's PYTHONPATH and needs triton (and,
# transitively for the full plugin, vLLM) — present in the serving container,
# not necessarily in the build venv. Skip cleanly instead of breaking
# collection of the main suite.
codec = pytest.importorskip(
    "gridbook.codec",
    reason="gridbook plugin not importable in this environment "
           "(run inside the serving container / with plugins on PYTHONPATH)",
)
kernels = pytest.importorskip("gridbook.kernels")
cb_decode_linear = kernels.cb_decode_linear

SERVE = Path("/home/rob/dq-runs/nvfp4-cb-phase0/serve")
ARTIFACTS = ["fp8cb_k44", "nvfp4cb_k16"]
# One single-role Linear per common role (single codebook -> row offset 0).
PICK = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.5.mlp.gate_proj",
    "model.layers.5.mlp.down_proj",
]
DEV = "cuda"


def _load(art):
    d = SERVE / art
    if not (d / "model.safetensors").exists():
        pytest.skip(f"artifact {art} not exported yet")
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    tensors = load_file(str(d / "model.safetensors"))
    codebooks = load_file(str(d / cfg.get("codebook_file", "cb_codebooks.pqcb")))
    # qname -> scheme
    q2s = {}
    for g in cfg["config_groups"].values():
        for t in g["targets"]:
            q2s[t] = g["scheme"]
    return tensors, codebooks, q2s


def _subtables(scheme, codebooks):
    ref = scheme["codebook_ref"]
    names = ref if isinstance(ref, list) else [ref]
    return [codebooks[n].to(DEV).float() for n in names]


def _extract_codes_ref(packed, k, type_size):
    """Pure-torch mirror of the kernel's 8-byte LE window extraction."""
    n, rb = packed.shape
    n_sb = rb // type_size
    pp = torch.cat([packed.to(torch.int64),
                    torch.zeros(n, 8, dtype=torch.int64, device=packed.device)],
                   dim=1)
    codes = torch.zeros(n, n_sb * 32, dtype=torch.int64, device=packed.device)
    for s in range(n_sb):
        for v in range(32):
            bitpos = v * k
            base = s * type_size + bitpos // 8
            val = torch.zeros(n, dtype=torch.int64, device=packed.device)
            for i in range(8):
                val |= pp[:, base + i] << (8 * i)
            codes[:, s * 32 + v] = (val >> (bitpos % 8)) & ((1 << k) - 1)
    return codes


@pytest.mark.parametrize("art", ARTIFACTS)
@pytest.mark.parametrize("qname", PICK)
def test_unpack_bitexact(art, qname):
    tensors, codebooks, q2s = _load(art)
    if qname not in q2s:
        pytest.skip(f"{qname} not a CB target in {art}")
    sch = q2s[qname]
    k, grid, mode, n_sub = sch["k"], sch["grid"], sch["mode"], sch["n_sub"]
    ts, is_fp4 = sch["type_size"], grid == "fp4"
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    N = packed.shape[0]
    in_f = (packed.shape[1] // ts) * codec.SUPERBLOCK

    subs = _subtables(sch, codebooks)
    ws = tensors.get(qname + ".weight_scale")
    ws = ws.to(DEV).float() if ws is not None else None
    fields = nvfp4_cb_unpack(packed, k, grid, mode, (N, in_f),
                             codebook=subs, scales=ws)
    idx = fields["indices"].to(torch.int64)              # (N, nvec, n_sub)
    widths = _bit_split(k, n_sub)
    off, code_ref = 0, torch.zeros_like(idx[..., 0])
    for i in range(n_sub):
        code_ref |= idx[..., i] << off
        off += widths[i]

    codes = _extract_codes_ref(packed, k, ts)
    assert torch.equal(codes, code_ref), (
        f"{art}/{qname}: kernel codeword extraction != nvfp4_cb_unpack")


@pytest.mark.parametrize("art", ARTIFACTS)
@pytest.mark.parametrize("qname", PICK)
@pytest.mark.parametrize("M", [1, 17])
def test_gemm_matches_reconstruct(art, qname, M):
    tensors, codebooks, q2s = _load(art)
    if qname not in q2s:
        pytest.skip(f"{qname} not a CB target in {art}")
    sch = q2s[qname]
    k, grid, mode, n_sub = sch["k"], sch["grid"], sch["mode"], sch["n_sub"]
    ts, is_fp4 = sch["type_size"], grid == "fp4"
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    N = packed.shape[0]
    K = (packed.shape[1] // ts) * codec.SUPERBLOCK
    subs = _subtables(sch, codebooks)
    ws = tensors.get(qname + ".weight_scale")
    ws = ws.to(DEV).float() if ws is not None else None

    # reconstruct wants fp8 per-channel scales as (rows, 1); the kernel wants
    # them flat (rows,).
    ws2d = ws.reshape(-1, 1) if ws is not None else None
    fields = nvfp4_cb_unpack(packed, k, grid, mode, (N, K),
                             codebook=subs, scales=ws2d)
    w_ref = nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode,
                                 codebook=subs).to(torch.bfloat16)  # (N,K)

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    y_ref = x.float() @ w_ref.float().t()

    cb_flat = codec.build_flat_codebook(subs)
    cb_row_offset = torch.zeros(N, dtype=torch.int32, device=DEV)
    qwp = codec.pad_qweight(packed)
    scale = (codec.decode_fp4_scale_plane(packed, k) if is_fp4
             else ws.reshape(-1))
    y = cb_decode_linear(x, qwp, cb_flat, cb_row_offset, scale,
                         torch.zeros(1, device=DEV), N=N, K=K,
                         k_bits=k, n_sub=n_sub, type_size=ts, is_fp4=is_fp4)

    rel = (y.float() - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
    assert rel <= 1e-2, f"{art}/{qname} M={M}: rel err {rel:.4e} > 1e-2"


def test_fused_row_offset():
    """Two different roles concatenated into one weight decode with their own
    codebooks (the qkv/gate_up fusion path) via cb_row_offset."""
    art = "nvfp4cb_k16"
    tensors, codebooks, q2s = _load(art)
    qa, qb = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
    if qa not in q2s or qb not in q2s:
        pytest.skip("q/k proj not present")
    sch = q2s[qa]
    k, grid, mode, n_sub, ts = sch["k"], sch["grid"], sch["mode"], \
        sch["n_sub"], sch["type_size"]
    is_fp4 = grid == "fp4"
    pa = tensors[qa + ".cb_qweight"].to(DEV)
    pb = tensors[qb + ".cb_qweight"].to(DEV)
    Na, Nb = pa.shape[0], pb.shape[0]
    K = (pa.shape[1] // ts) * codec.SUPERBLOCK
    subs_a = _subtables(q2s[qa], codebooks)
    subs_b = _subtables(q2s[qb], codebooks)

    def ref(p, subs):
        f = nvfp4_cb_unpack(p, k, grid, mode, (p.shape[0], K), codebook=subs)
        return nvfp4_cb_reconstruct(f, k, grid=grid, mode=mode,
                                    codebook=subs).to(torch.bfloat16)
    w_ref = torch.cat([ref(pa, subs_a), ref(pb, subs_b)], dim=0)   # (Na+Nb, K)

    packed = torch.cat([pa, pb], dim=0)
    cb_flat_a = codec.build_flat_codebook(subs_a)
    cb_flat_b = codec.build_flat_codebook(subs_b)
    cb_flat = torch.cat([cb_flat_a, cb_flat_b])
    off = torch.cat([torch.zeros(Na, dtype=torch.int32, device=DEV),
                     torch.full((Nb,), cb_flat_a.numel(), dtype=torch.int32,
                                device=DEV)])
    qwp = codec.pad_qweight(packed)
    scale = codec.decode_fp4_scale_plane(packed, k)
    torch.manual_seed(1)
    x = torch.randn(8, K, dtype=torch.bfloat16, device=DEV)
    y = cb_decode_linear(x, qwp, cb_flat, off, scale,
                         torch.zeros(1, device=DEV), N=Na + Nb, K=K,
                         k_bits=k, n_sub=n_sub, type_size=ts, is_fp4=is_fp4)
    y_ref = x.float() @ w_ref.float().t()
    rel = (y.float() - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
    assert rel <= 1e-2, f"fused rel err {rel:.4e} > 1e-2"
