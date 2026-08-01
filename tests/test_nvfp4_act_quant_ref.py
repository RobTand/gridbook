"""Focused oracle tests for native NVFP4 activation quantization.

These cases pin the two boundaries that random normal inputs almost never
exercise: E2M1 midpoint ties and E4M3 scale-factor underflow.  CPU tests keep
the reference contract continuously covered; Blackwell tests compare it to
vLLM's live ``scaled_fp4_quant`` operator without building Gridbook's fused
CUDA extension.  They deliberately do not claim arbitrary packed-byte
equivalence: vLLM applies nonzero scale factors with hardware approximate
reciprocals that a plain Torch reference cannot reproduce bit-for-bit.
"""

import pytest
import torch

from gridbook import codec


MIDPOINT_ROW = [
    0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
    -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0,
    6.0, 0.0,
]
MIDPOINT_CODES_RNE = [
    0, 2, 2, 4, 4, 6, 6,
    8, 10, 10, 12, 12, 14, 14,
    7, 0,
]

# The smallest E4M3 subnormal is 2^-9.  Its round-to-nearest-even underflow
# boundary is half that, so the corresponding group amax boundary is 6*2^-10.
UNDERFLOW_VALUES = [
    0.0,
    2.0 ** -20,
    0.001,
    0.003,
    0.005,
    0.0058,
    6.0 * 2.0 ** -10,
    0.006,
    0.008,
    -(2.0 ** -20),
    -0.001,
    -0.003,
    -0.005,
    -0.0058,
    -(6.0 * 2.0 ** -10),
    -0.006,
]
UNDERFLOW_GROUPS = [*range(7), *range(9, 15)]
NONZERO_SF_GROUPS = [7, 8, 15]


def _unpack_codes(packed: torch.Tensor, k: int) -> torch.Tensor:
    u8 = packed.view(torch.uint8)
    return torch.stack((u8 & 0xF, (u8 >> 4) & 0xF), dim=-1).reshape(-1, k)


def _underflow_row(device: torch.device) -> torch.Tensor:
    groups = [torch.full((codec.FP4_GROUP,), value,
                         dtype=torch.bfloat16, device=device)
              for value in UNDERFLOW_VALUES]
    groups.extend(torch.zeros(codec.FP4_GROUP, dtype=torch.bfloat16,
                              device=device)
                  for _ in range(16 - len(groups)))
    return torch.cat(groups)


def test_nvfp4_ref_rounds_every_e2m1_midpoint_to_even():
    x = torch.tensor([MIDPOINT_ROW], dtype=torch.bfloat16)
    packed, sf, reciprocal = codec.nvfp4_act_quant_ref(
        x, torch.tensor(1.0, dtype=torch.float32))

    assert _unpack_codes(packed, x.shape[1])[0].tolist() == MIDPOINT_CODES_RNE
    assert sf.view(torch.uint8).tolist() == [[
        int(torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8))
    ]]
    assert reciprocal.item() == 1.0


def test_nvfp4_ref_e4m3_underflow_boundary_emits_zero_group():
    x = _underflow_row(torch.device("cpu")).reshape(1, -1)
    packed, sf, _ = codec.nvfp4_act_quant_ref(
        x, torch.tensor(1.0, dtype=torch.float32))
    sf_codes = sf.view(torch.uint8)[0]
    payload = _unpack_codes(packed, x.shape[1])[0].reshape(-1, codec.FP4_GROUP)

    assert sf_codes[UNDERFLOW_GROUPS].tolist() == [0] * len(UNDERFLOW_GROUPS)
    assert sf_codes[NONZERO_SF_GROUPS].tolist() == [1, 1, 1]
    assert not bool(payload[:7].any())
    assert payload[9:15].unique().tolist() == [8]  # signed numerical zero


def _require_live_scaled_fp4_quant():
    if not torch.cuda.is_available():
        pytest.skip("live vLLM oracle needs CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("live scaled FP4 quantization needs Blackwell")
    vops = pytest.importorskip("vllm._custom_ops")
    if not hasattr(vops, "scaled_fp4_quant"):
        pytest.skip("vLLM has no scaled_fp4_quant")
    return vops


def _logical_vllm_sf(sf: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    offsets = codec.sf_swizzle_offsets(rows, groups, sf.device)
    return sf.view(torch.uint8).reshape(-1)[offsets]


def test_live_vllm_rounds_every_e2m1_midpoint_to_even():
    vops = _require_live_scaled_fp4_quant()
    row = torch.tensor(MIDPOINT_ROW * 16, dtype=torch.bfloat16, device="cuda")
    x = row.reshape(1, -1).expand(128, -1).contiguous()
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    packed_live, sf_live = vops.scaled_fp4_quant(x, global_scale)
    packed_ref, sf_ref, _ = codec.nvfp4_act_quant_ref(x, global_scale)
    sf_live_logical = _logical_vllm_sf(sf_live, *sf_ref.shape)

    assert torch.equal(sf_live_logical, sf_ref.view(torch.uint8))
    assert torch.equal(packed_live.view(torch.uint8), packed_ref)
    assert (_unpack_codes(packed_live, x.shape[1])[0, :16].tolist()
            == MIDPOINT_CODES_RNE)


def test_live_vllm_e4m3_underflow_boundary_matches_reference():
    vops = _require_live_scaled_fp4_quant()
    row = _underflow_row(torch.device("cuda"))
    x = row.reshape(1, -1).expand(128, -1).contiguous()
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    packed_live, sf_live = vops.scaled_fp4_quant(x, global_scale)
    packed_ref, sf_ref, _ = codec.nvfp4_act_quant_ref(x, global_scale)
    sf_live_logical = _logical_vllm_sf(sf_live, *sf_ref.shape)

    assert (sf_live_logical[0, UNDERFLOW_GROUPS].tolist()
            == [0] * len(UNDERFLOW_GROUPS))
    assert sf_live_logical[0, NONZERO_SF_GROUPS].tolist() == [1, 1, 1]
    assert torch.equal(sf_live_logical, sf_ref.view(torch.uint8))
    assert torch.equal(packed_live.view(torch.uint8), packed_ref)


def test_live_static_lsq_preserves_midpoint_and_underflow_payload_boundaries():
    """Fixed G=1 must not let residual fitting perturb native payload bytes."""

    vops = _require_live_scaled_fp4_quant()
    from gridbook.cuda_ext import get_fused_fp4_ext

    ext = get_fused_fp4_ext()
    if ext is None or not hasattr(ext, "cb_nvfp4_quantize_static_lsq"):
        pytest.skip("static-LSQ fused FP4 extension unavailable")
    midpoint = torch.tensor(
        MIDPOINT_ROW * 16, dtype=torch.bfloat16, device="cuda"
    )
    underflow = _underflow_row(torch.device("cuda"))
    x = torch.stack((midpoint, underflow, torch.zeros_like(midpoint))).contiguous()
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    packed, sfa, residuals = ext.cb_nvfp4_quantize_static_lsq(x, 1.0)
    packed_live, sfa_live = vops.scaled_fp4_quant(x, global_scale)
    logical = _logical_vllm_sf(sfa, x.shape[0], x.shape[1] // codec.FP4_GROUP)
    logical_live = _logical_vllm_sf(
        sfa_live, x.shape[0], x.shape[1] // codec.FP4_GROUP
    )

    assert torch.equal(packed, packed_live)
    assert torch.equal(logical, logical_live)
    assert torch.isfinite(residuals).all()
    assert (residuals[:2] > 0).all()
    # q dot q is exactly zero only for the final all-zero row, so r=1 and the
    # original static residual r/G remains one rather than becoming NaN/zero.
    assert residuals[2].item() == 1.0
