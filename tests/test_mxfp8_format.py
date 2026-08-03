"""MXFP8 reference semantics and the DeepSeek block-128 embedding (CPU-only).

The embedding claim these tests pin: DeepSeek's FP8 convention (E4M3 values,
one UE8M0 scale per 128x128 tile) embeds EXACTLY into MXFP8 (same values, one
UE8M0 scale per 32 K-elements) by scale replication — 128 = 4 * 32, so a chunk
never straddles a tile, and the map ``SF_mx[n, c] = S_ds[n // 128, c // 4]``
is total, arithmetic-free and bit-exact.
"""
import pytest

torch = pytest.importorskip("torch")

from gridbook.mxfp8 import (  # noqa: E402
    E4M3_MAX,
    SFVEC,
    broadcast_block128_scales,
    dequant_mxfp8,
    fill_sf_plane,
    quantize_mxfp8,
)


def test_quantize_never_saturates_under_ceil_rule():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * torch.exp2(
        torch.randint(-20, 20, (64, 1)).float())
    q, sf = quantize_mxfp8(x)
    scale = torch.exp2(sf.to(torch.int16).float() - 127.0)
    scaled = (x.unflatten(-1, (256 // SFVEC, SFVEC))
              / scale.unsqueeze(-1)).abs()
    assert float(scaled.max()) <= E4M3_MAX + 1e-6
    assert not q.to(torch.float32).isinf().any()


def test_quantize_round_trip_error_is_e4m3_grade():
    torch.manual_seed(1)
    x = torch.randn(32, 128)
    q, sf = quantize_mxfp8(x)
    y = dequant_mxfp8(q, sf)
    rel = ((y - x).abs() / x.abs().clamp_min(1e-6)).median()
    # e4m3 carries 3 mantissa bits: median relative error well under 1/16.
    assert float(rel) < 1 / 16


def test_all_zero_group_is_exact_with_minimum_scale():
    x = torch.zeros(4, SFVEC * 2)
    q, sf = quantize_mxfp8(x)
    assert torch.equal(q.to(torch.float32), x)
    assert int(sf.max()) == 0  # 2**-127, the UE8M0 minimum


def test_k_not_multiple_of_group_refused():
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp8(torch.randn(4, 33))
    with pytest.raises(ValueError, match="divisible by 32"):
        broadcast_block128_scales(torch.zeros(1, 1, dtype=torch.uint8), 4, 33)


@pytest.mark.parametrize("rows,k", [(136, 576), (128, 128), (300, 224),
                                    (513, 4096)])
def test_broadcast_matches_naive_loop_bit_exactly(rows, k):
    torch.manual_seed(2)
    s = torch.randint(0, 255, ((rows + 127) // 128, (k + 127) // 128),
                      dtype=torch.uint8)
    fast = broadcast_block128_scales(s, rows, k)
    slow = torch.empty(rows, k // SFVEC, dtype=torch.uint8)
    for n in range(rows):
        for c in range(k // SFVEC):
            slow[n, c] = s[n // 128, (c * SFVEC) // 128]
    assert torch.equal(fast, slow)


def test_broadcast_shape_mismatch_refused():
    with pytest.raises(ValueError, match="does not match"):
        broadcast_block128_scales(torch.zeros(2, 2, dtype=torch.uint8),
                                  128, 128)


def test_block128_values_requantize_bit_exactly_under_broadcast_scales():
    """The value half of the embedding proof: dividing the dequantized tensor
    by the broadcast scale recovers the ORIGINAL e4m3 bytes, because the
    multiplier is the exact power of two the bytes were stored under."""
    torch.manual_seed(3)
    n, k = 136, 576
    w_q = (torch.randn(n, k) * 64).to(torch.float8_e4m3fn)
    s = torch.randint(112, 140, ((n + 127) // 128, (k + 127) // 128),
                      dtype=torch.uint8)
    sf_rm = broadcast_block128_scales(s, n, k)
    scale = torch.exp2(sf_rm.to(torch.int16).float() - 127.0)
    w_deq = (w_q.to(torch.float32).unflatten(-1, (k // SFVEC, SFVEC))
             * scale.unsqueeze(-1)).flatten(-2)
    w_back = (w_deq.unflatten(-1, (k // SFVEC, SFVEC))
              / scale.unsqueeze(-1)).flatten(-2).to(torch.float8_e4m3fn)
    assert torch.equal(w_back.view(torch.uint8), w_q.view(torch.uint8))


def test_fill_sf_plane_contracts():
    sf = torch.zeros(4, 4, dtype=torch.uint8)
    offs = torch.arange(16)
    plane = fill_sf_plane(sf, offs, 32)
    assert plane.numel() == 32 and plane.dtype == torch.uint8
    with pytest.raises(ValueError, match="offsets numel"):
        fill_sf_plane(sf, torch.arange(15), 32)
    with pytest.raises(ValueError, match="plane_numel"):
        fill_sf_plane(sf, offs, 32, out=torch.zeros(31, dtype=torch.uint8))
