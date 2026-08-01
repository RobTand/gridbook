"""Live gates for the experimental fixed-G NVFP4 LSQ residual policy."""

from __future__ import annotations

import pytest
import torch

from gridbook import codec


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="static-LSQ NVFP4 quantization needs CUDA"
)


@pytest.fixture(scope="module")
def ext():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion needs Blackwell")
    from gridbook.cuda_ext import get_fused_fp4_ext

    module = get_fused_fp4_ext()
    if module is None or not hasattr(module, "cb_nvfp4_quantize_static_lsq"):
        pytest.skip("static-LSQ fused FP4 extension unavailable")
    return module


def _logical_sfa(sfa: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    offsets = codec.sf_swizzle_offsets(rows, groups, sfa.device)
    return sfa.reshape(-1)[offsets]


def _decode_native_payload(
    packed: torch.Tensor, sfa: torch.Tensor, rows: int, k: int
) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(rows, k)
    magnitude = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=packed.device,
        dtype=torch.float32,
    )
    values = magnitude[(codes & 7).long()]
    values = torch.where((codes & 8).bool(), -values, values)
    sf = _logical_sfa(sfa, rows, k // 16).view(torch.float8_e4m3fn).float()
    return (values.reshape(rows, k // 16, 16) * sf.unsqueeze(-1)).reshape(
        rows, k
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("shape", ((1, 256), (17, 1024), (129, 256)))
def test_static_lsq_fixed_g_payload_matches_vllm(ext, dtype, shape):
    vops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(vops, "scaled_fp4_quant"):
        pytest.skip("vLLM scaled_fp4_quant unavailable")

    torch.manual_seed(414)
    x = (torch.randn(*shape, device="cuda") * 1.75).to(dtype).contiguous()
    global_scale = 7.25
    global_scale_tensor = torch.tensor(
        global_scale, device="cuda", dtype=torch.float32
    )
    packed, sfa, _ = ext.cb_nvfp4_quantize_static_lsq(x, global_scale)
    packed_ref, sfa_ref = vops.scaled_fp4_quant(x, global_scale_tensor)

    assert torch.equal(packed, packed_ref)
    assert torch.equal(sfa, sfa_ref.view(torch.uint8).reshape(-1))


def test_static_lsq_is_exact_row_optimum_and_never_regresses_raw_mse(ext):
    torch.manual_seed(415)
    amplitudes = torch.tensor(
        (1.0e-3, 0.25, 2.0, 32.0, 2.0e3), device="cuda"
    ).reshape(-1, 1)
    x = (torch.randn(5, 1024, device="cuda") * amplitudes).to(
        torch.bfloat16
    ).contiguous()
    global_scale = 1.75
    packed, sfa, scales = ext.cb_nvfp4_quantize_static_lsq(x, global_scale)
    q_raw = _decode_native_payload(packed, sfa, x.shape[0], x.shape[1])
    xf = x.float()

    dot_xq = (xf * q_raw).sum(dim=1)
    dot_qq = q_raw.square().sum(dim=1)
    static_recip = torch.tensor(
        1.0 / global_scale, device="cuda", dtype=torch.float32
    )
    expected = torch.where(dot_qq > 0, dot_xq / dot_qq, static_recip)
    torch.testing.assert_close(scales, expected, rtol=2.0e-5, atol=2.0e-7)

    static_error = (xf - q_raw * static_recip).square().sum(dim=1)
    lsq_error = (xf - q_raw * scales[:, None]).square().sum(dim=1)
    # The closed-form row optimum cannot make any row worse.  Keep only a tiny
    # reduction-order allowance rather than hiding a substantive regression.
    torch.testing.assert_close(
        torch.minimum(lsq_error, static_error), lsq_error,
        rtol=2.0e-6, atol=2.0e-6,
    )


def test_static_lsq_keeps_large_valid_residual_unclipped(ext):
    # A deliberately saturated fixed-G payload has a valid LS multiplier well
    # above two.  Clipping this is not a numerical-safety measure: heldout model
    # evidence shows that even rare large valid residuals can materially matter.
    x = torch.full(
        (1, 256), 12000.0, device="cuda", dtype=torch.bfloat16
    )
    global_scale = 1.0
    packed, sfa, scales = ext.cb_nvfp4_quantize_static_lsq(x, global_scale)
    q_raw = _decode_native_payload(packed, sfa, 1, 256)
    expected = (x.float() * q_raw).sum() / q_raw.square().sum()

    assert expected > 2.0
    torch.testing.assert_close(scales[0], expected, rtol=2.0e-6, atol=0.0)
    assert scales[0] > 2.0 / global_scale


def test_static_lsq_zero_denominator_falls_back_to_original_static_residual(ext):
    x = torch.zeros(3, 256, device="cuda", dtype=torch.bfloat16)
    global_scale = 3.25
    packed, sfa, scales = ext.cb_nvfp4_quantize_static_lsq(x, global_scale)

    assert not bool(packed.any())
    assert not bool(sfa.any())
    torch.testing.assert_close(
        scales, torch.full_like(scales, 1.0 / global_scale),
        rtol=2.0e-7, atol=0.0,
    )


def test_static_lsq_out_nondefault_stream_and_graph(ext):
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    global_scale = 6.0
    packed = torch.empty(32, 512, device="cuda", dtype=torch.uint8)
    sfa = torch.empty(128 * 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(32, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ext.cb_nvfp4_quantize_static_lsq_out(
            x, global_scale, packed, sfa, scales
        )
    stream.synchronize()
    expected = ext.cb_nvfp4_quantize_static_lsq(x, global_scale)
    assert torch.equal(packed, expected[0])
    assert torch.equal(sfa, expected[1])
    assert torch.equal(scales, expected[2])

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ext.cb_nvfp4_quantize_static_lsq_out(
            x, global_scale, packed, sfa, scales
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(packed, expected[0])
    assert torch.equal(sfa, expected[1])
    assert torch.equal(scales, expected[2])


def test_static_lsq_binding_rejects_malformed_inputs(ext):
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    global_scale = 4.0
    packed = torch.empty(2, 128, device="cuda", dtype=torch.uint8)
    sfa = torch.empty(128 * 16, device="cuda", dtype=torch.uint8)
    scales = torch.empty(2, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="input must be CUDA"):
        ext.cb_nvfp4_quantize_static_lsq(x.cpu(), global_scale)
    with pytest.raises(RuntimeError, match="bfloat16 or float16"):
        ext.cb_nvfp4_quantize_static_lsq(x.float(), global_scale)
    with pytest.raises(RuntimeError, match="contiguous"):
        ext.cb_nvfp4_quantize_static_lsq(x[:, ::2], global_scale)
    with pytest.raises(RuntimeError, match="multiple of 256"):
        ext.cb_nvfp4_quantize_static_lsq(
            x[:, :240].contiguous(), global_scale
        )
    for bad_scale in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(RuntimeError, match="global_scale must"):
            ext.cb_nvfp4_quantize_static_lsq(x, bad_scale)
    with pytest.raises(RuntimeError, match="packed must"):
        ext.cb_nvfp4_quantize_static_lsq_out(
            x, global_scale, packed[:, :-1].contiguous(), sfa, scales
        )
    with pytest.raises(RuntimeError, match="sfa must"):
        ext.cb_nvfp4_quantize_static_lsq_out(
            x, global_scale, packed, sfa[:-1], scales
        )
    with pytest.raises(RuntimeError, match="a_scales must"):
        ext.cb_nvfp4_quantize_static_lsq_out(
            x, global_scale, packed, sfa, scales.half()
        )
