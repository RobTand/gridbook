"""CB MoE — stacked-expert layout contract (LAYOUT.md §3, moe_cb_design.md §4).

CPU-only (no triton, no vLLM) — validates that each expert's slice of a stacked
pack is exactly the dense §1 superblock layout, so the MoE method's per-expert
decode (which reuses the dense, separately bit-exact-tested, expand kernels) is
correct. The vLLM-dependent MoE-method construction/forward test is skip-guarded
and runs in the container in the post-27B GPU window (resource-discipline hold).

  PYTHONPATH=/home/rob/prismaquant:/home/rob/prismaquant/plugins/vllm_prismaquant \
    CUDA_VISIBLE_DEVICES= /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
    -m pytest plugins/vllm_prismaquant/tests/test_moe_stacked.py -q
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-only; the GPU is the 27B's

import pytest
import torch

fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")

CPU = torch.device("cpu")


@pytest.mark.parametrize("scale_coding", ["v1", "two_tier"])
def test_stacked_fp4_expert_slices_are_dense(scale_coding):
    """Pack a stacked (E, out, in) fp4 weight; each expert's row-slice
    (out, bytes) unpacks + reconstructs to that expert's dense weight."""
    E, out, in_f, k = 4, 64, 512, 16
    torch.manual_seed(0)
    w = torch.randn(E, out, in_f) * 0.05
    cb = fmt._resolve_codebook(k, "fp4", "product", None, CPU)
    packed, fields = fmt.nvfp4_cb_pack(
        w, k, grid="fp4", mode="product", codebook=cb, scale_coding=scale_coding)
    ts = fmt.nvfp4_cb_type_size(k, "fp4", scale_coding)
    # The exporter reshapes (E*out, bytes) -> (E, out, bytes); cb_qweight[e] is
    # then rows [e*out:(e+1)*out] of the flat pack.
    assert packed.shape == (E * out, (in_f // 256) * ts)
    w_full = fmt.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode="product",
                                      codebook=cb)          # (E, out, in)
    assert tuple(w_full.shape) == (E, out, in_f)
    for e in range(E):
        p_e = packed[e * out:(e + 1) * out].contiguous()    # (out, bytes)
        f_e = fmt.nvfp4_cb_unpack(p_e, k, "fp4", "product", (out, in_f),
                                  codebook=cb, scale_coding=scale_coding)
        w_e = fmt.nvfp4_cb_reconstruct(f_e, k, grid="fp4", mode="product",
                                       codebook=cb)
        assert torch.equal(w_e, w_full[e]), f"expert {e} slice != dense reconstruct"


def test_stacked_fp8_expert_scale_slices():
    """fp8 stacked: the per-(expert,out) weight_scale slices align with the
    packed rows (the MoE buffer is weight_scale (E, out))."""
    E, out, in_f, k = 3, 48, 256, 44
    torch.manual_seed(1)
    w = torch.randn(E, out, in_f) * 0.05
    cb = fmt._resolve_codebook(k, "fp8", "product", None, CPU)
    packed, fields = fmt.nvfp4_cb_pack(w, k, grid="fp8", mode="product",
                                       codebook=cb)
    ts = fmt.nvfp4_cb_type_size(k, "fp8")
    assert packed.shape == (E * out, (in_f // 256) * ts)    # 4k bytes, no plane
    scales = fields["scales"].reshape(E * out, -1)          # per-channel (E*out,1)
    assert scales.shape[0] == E * out
    w_full = fmt.nvfp4_cb_reconstruct(fields, k, grid="fp8", mode="product",
                                      codebook=cb)
    for e in range(E):
        p_e = packed[e * out:(e + 1) * out].contiguous()
        s_e = scales[e * out:(e + 1) * out]
        f_e = fmt.nvfp4_cb_unpack(p_e, k, "fp8", "product", (out, in_f),
                                  codebook=cb, scales=s_e)
        w_e = fmt.nvfp4_cb_reconstruct(f_e, k, grid="fp8", mode="product",
                                       codebook=cb)
        assert torch.equal(w_e, w_full[e])


def test_moe_method_buffer_shapes():
    """vLLM-dependent: construct the CB MoE method + create_weights, assert the
    stacked w13/w2 buffer shapes. Deferred to the container (post-27B window)."""
    pytest.importorskip("vllm")
    import types
    from vllm_prismaquant.moe import PrismaQuantCBMoEMethod, _row_bytes

    E, hidden, inter, k, ts = 8, 512, 1024, 44, 176   # fp8 k44 -> ts=4k=176
    scheme = {"grid": "fp8", "mode": "product", "k": k, "n_sub": 4,
              "type_size": ts, "codebook_ref": ["cb.x"]}
    moe_cfg = types.SimpleNamespace()                 # only stored, not read here
    m = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    # bypass FusedMoEMethodBase.__init__ (needs a real FusedMoEConfig); test the
    # buffer-shape logic directly.
    m.quant_config = None; m.scheme = scheme; m.prefix = "x"
    m.is_fp4 = False; m.k = k; m.n_sub = 4; m.type_size = ts; m.is_v2 = False
    layer = types.SimpleNamespace()
    layer.register_parameter = lambda n, p: setattr(layer, n, p)
    m.create_weights(layer, E, hidden, inter, torch.bfloat16, weight_loader=None)
    assert layer.w13_cb_qweight.shape == (E, 2 * inter, _row_bytes(hidden, ts))
    assert layer.w2_cb_qweight.shape == (E, hidden, _row_bytes(inter, ts))
    assert layer.w13_weight_scale.shape == (E, 2 * inter)
    assert layer.w2_weight_scale.shape == (E, hidden)
