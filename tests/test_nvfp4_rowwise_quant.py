"""Live gates for the shared row-wise native-NVFP4 activation quantizer."""

from __future__ import annotations

import math

import pytest
import torch

from gridbook import codec


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="row-wise NVFP4 quantization needs CUDA"
)


@pytest.fixture(scope="module")
def ext():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native FP4 conversion needs Blackwell")
    from gridbook.cuda_ext import get_fused_fp4_ext

    module = get_fused_fp4_ext()
    if module is None or not hasattr(module, "cb_nvfp4_quantize_rows"):
        pytest.skip("fused FP4 extension unavailable")
    return module


def _logical_sfa(sfa: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    offsets = codec.sf_swizzle_offsets(rows, groups, sfa.device)
    return sfa.reshape(-1)[offsets]


def _unpack(packed: torch.Tensor, k: int) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    return torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(-1, k)


def _outputs(ext, x: torch.Tensor, multiplier: float = 448.0):
    packed, sfa, scales = ext.cb_nvfp4_quantize_rows(x, multiplier)
    return packed, _logical_sfa(sfa, x.shape[0], x.shape[1] // 16), scales


def test_rowwise_quant_matches_vllm_one_row_calls(ext):
    vops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(vops, "scaled_fp4_quant"):
        pytest.skip("vLLM scaled_fp4_quant unavailable")

    torch.manual_seed(931)
    rows = []
    for row_max in (1.0, 2.0, 4.0):
        row = (torch.randn(256, device="cuda") * (row_max / 5.0)).clamp(
            -row_max, row_max
        )
        row[0] = row_max
        rows.append(row.to(torch.bfloat16))
    x = torch.stack(rows).contiguous()
    packed, sfa, scales = _outputs(ext, x)

    for index, row_max in enumerate((1.0, 2.0, 4.0)):
        global_scale = torch.tensor(
            448.0 * 6.0 / row_max, dtype=torch.float32, device="cuda"
        )
        packed_ref, sfa_ref = vops.scaled_fp4_quant(
            x[index : index + 1], global_scale
        )
        logical_ref = _logical_sfa(sfa_ref.view(torch.uint8), 1, 16)
        assert torch.equal(packed[index], packed_ref[0])
        assert torch.equal(sfa[index], logical_ref[0])
        assert scales[index].item() == torch.tensor(
            1.0 / float(global_scale), dtype=torch.float32
        ).item()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_rowwise_quant_matches_vllm_for_arbitrary_row_ranges(ext, dtype):
    vops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(vops, "scaled_fp4_quant"):
        pytest.skip("vLLM scaled_fp4_quant unavailable")

    torch.manual_seed(1801)
    amplitudes = torch.tensor(
        (1.0e-3, 0.25, 7.0, 1.0e3), device="cuda"
    ).reshape(-1, 1)
    x = (torch.randn(4, 1024, device="cuda") * amplitudes).to(
        dtype
    ).contiguous()
    packed, sfa, scales = _outputs(ext, x)

    for index in range(x.shape[0]):
        row_max = x[index].abs().float().max()
        global_scale = (torch.tensor(
            448.0 * 6.0, dtype=torch.float32, device="cuda"
        ) / row_max).to(torch.float32)
        packed_ref, sfa_ref = vops.scaled_fp4_quant(
            x[index : index + 1], global_scale
        )
        logical_ref = _logical_sfa(sfa_ref.view(torch.uint8), 1, 64)
        assert torch.equal(packed[index], packed_ref[0])
        assert torch.equal(sfa[index], logical_ref[0])
        torch.testing.assert_close(
            scales[index], 1.0 / global_scale, rtol=2.0e-7, atol=0.0
        )


def test_rowwise_quant_is_batch_and_chunk_invariant(ext):
    torch.manual_seed(932)
    target = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    peers_a = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16) * 0.01
    peers_b = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16) * 100.0

    one = _outputs(ext, target)
    batch_a = _outputs(ext, torch.cat((target, peers_a), dim=0))
    batch_b = _outputs(ext, torch.cat((target, peers_b), dim=0))
    moved = _outputs(ext, torch.cat((peers_b[:1], target, peers_b[1:]), dim=0))

    assert torch.equal(one[0][0], batch_a[0][0])
    assert torch.equal(one[0][0], batch_b[0][0])
    assert torch.equal(one[0][0], moved[0][1])
    assert torch.equal(one[1][0], batch_a[1][0])
    assert torch.equal(one[1][0], batch_b[1][0])
    assert torch.equal(one[1][0], moved[1][1])
    scale_bits = one[2][0:1].view(torch.uint8)
    assert torch.equal(scale_bits, batch_a[2][0:1].view(torch.uint8))
    assert torch.equal(scale_bits, batch_b[2][0:1].view(torch.uint8))
    assert torch.equal(scale_bits, moved[2][1:2].view(torch.uint8))


def test_rowwise_quant_zero_and_underflow_boundaries(ext):
    # A row maximum of 2688 makes G=448*6/2688 exactly one. Reuse the live
    # underflow boundary from the scalar-G reference and reserve the final
    # group for the value that fixes the row maximum.
    tiny = torch.zeros(256, dtype=torch.bfloat16, device="cuda")
    half_min_subnormal_sf = 6.0 * (2.0 ** -10)
    tiny[:16] = half_min_subnormal_sf
    tiny[16:32] = torch.nextafter(
        torch.tensor(half_min_subnormal_sf, device="cuda"),
        torch.tensor(math.inf, device="cuda"),
    ).to(torch.bfloat16)
    tiny[-16] = 2688.0
    zero = torch.zeros_like(tiny)
    packed, sfa, scales = _outputs(ext, torch.stack((tiny, zero)))

    assert sfa[0, 0].item() == 0
    assert scales[0].item() == 1.0
    assert not bool(sfa[1].any())
    assert not bool((_unpack(packed[1:2], 256) & 7).any())
    assert scales[1].item() == 0.0


def test_rowwise_quant_out_nondefault_stream_and_graph(ext):
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty(32, 512, device="cuda", dtype=torch.uint8)
    sfa = torch.empty(128 * 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(32, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ext.cb_nvfp4_quantize_rows_out(x, packed, sfa, scales, 448.0)
    stream.synchronize()
    expected = ext.cb_nvfp4_quantize_rows(x, 448.0)
    assert torch.equal(packed, expected[0])
    assert torch.equal(sfa, expected[1])
    assert torch.equal(scales, expected[2])

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ext.cb_nvfp4_quantize_rows_out(x, packed, sfa, scales, 448.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(packed, expected[0])
    assert torch.equal(sfa, expected[1])
    assert torch.equal(scales, expected[2])


def test_rowwise_quant_binding_rejects_malformed_inputs(ext):
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty(2, 128, device="cuda", dtype=torch.uint8)
    sfa = torch.empty(128 * 16, device="cuda", dtype=torch.uint8)
    scales = torch.empty(2, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="input must be CUDA"):
        ext.cb_nvfp4_quantize_rows(x.cpu(), 448.0)
    with pytest.raises(RuntimeError, match="bfloat16 or float16"):
        ext.cb_nvfp4_quantize_rows(x.float(), 448.0)
    with pytest.raises(RuntimeError, match="contiguous"):
        ext.cb_nvfp4_quantize_rows(x[:, ::2], 448.0)
    with pytest.raises(RuntimeError, match="multiple of 256"):
        ext.cb_nvfp4_quantize_rows(x[:, :240].contiguous(), 448.0)
    with pytest.raises(RuntimeError, match="range_multiplier"):
        ext.cb_nvfp4_quantize_rows(x, 0.0)
    with pytest.raises(RuntimeError, match="range_multiplier"):
        ext.cb_nvfp4_quantize_rows(x, 449.0)
    with pytest.raises(RuntimeError, match="packed must"):
        ext.cb_nvfp4_quantize_rows_out(
            x, packed[:, :-1].contiguous(), sfa, scales, 448.0
        )
    with pytest.raises(RuntimeError, match="sfa must"):
        ext.cb_nvfp4_quantize_rows_out(x, packed, sfa[:-1], scales, 448.0)
    with pytest.raises(RuntimeError, match="a_scales must"):
        ext.cb_nvfp4_quantize_rows_out(x, packed, sfa, scales.half(), 448.0)
