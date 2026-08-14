"""Edge values, malformed input, scale grouping, E2M1/E8M0 specifics."""
import torch
import pytest

from research.mxfp4_cb.format import Mxfp4CbFormat, SUPERBLOCK
from research.mxfp4_cb.e2m1 import (
    e2m1_encode,
    e2m1_decode,
    e8m0_encode_amax,
    e8m0_decode,
    quantize_mxfp4_block,
    dequant_mxfp4_block,
    MX_BLOCK,
)
from research.mxfp4_cb.codec import encode_mxfp4_cb, decode_mxfp4_cb
from research.mxfp4_cb.e2m1 import E2M1_MAGNITUDES


def test_e2m1_all_grid_values_roundtrip():
    vals = torch.tensor(sorted({v for m in E2M1_MAGNITUDES for v in (m, -m)}), dtype=torch.float32)
    codes = e2m1_encode(vals)
    got = e2m1_decode(codes)
    assert torch.equal(vals, got)
    # Nibbles 0..15 all decode; encode recovers same nibble except code 8 (-0) canonicalizes to 0
    all_codes = torch.arange(16, dtype=torch.uint8)
    vals2 = e2m1_decode(all_codes)
    codes2 = e2m1_encode(vals2)
    expected = all_codes.clone()
    expected[8] = 0  # -0 -> +0 canonical
    assert torch.equal(codes2, expected)


def test_e2m1_zero_sign():
    pos0 = torch.tensor([0.0])
    neg0 = torch.tensor([-0.0])
    # Both should encode to code 0 (mag 0, sign 0) — negative zero not preserved for zero magnitude
    assert int(e2m1_encode(pos0).item()) == 0
    assert int(e2m1_encode(neg0).item()) == 0
    assert float(e2m1_decode(torch.tensor([0], dtype=torch.uint8)).item()) == 0.0


def test_e2m1_nonfinite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        e2m1_encode(torch.tensor([float("inf")]))
    with pytest.raises(ValueError, match="non-finite"):
        e2m1_encode(torch.tensor([float("nan")]))


def test_e2m1_decode_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        e2m1_decode(torch.tensor([16], dtype=torch.uint8))


def test_e8m0_zero_group():
    amax = torch.tensor([0.0, 0.0])
    sf = e8m0_encode_amax(amax)
    assert torch.equal(sf, torch.tensor([0, 0], dtype=torch.uint8))
    scales = e8m0_decode(sf)
    # 2^-127
    assert float(scales[0].item()) == 2 ** (-127)
    # dequant of zeros stays zero
    x = torch.zeros(1, 32)
    q, sf2 = quantize_mxfp4_block(x)
    assert int(sf2[0, 0].item()) == 0
    dq = dequant_mxfp4_block(q, sf2)
    assert torch.equal(dq, torch.zeros_like(dq))


def test_e8m0_nan_reserved():
    # Byte 0xFF decodes to NaN
    assert torch.isnan(e8m0_decode(torch.tensor([255], dtype=torch.uint8))).all()
    # Encoder never emits 0xFF — test with largest finite amax that still encodes
    # (1e38 is < 3.4e38 float32 max)
    big = torch.tensor([1e38])
    sf = e8m0_encode_amax(big)
    assert int(sf.item()) != 255
    assert not torch.isnan(e8m0_decode(sf)).any()
    mid = torch.tensor([1e30])
    sf_mid = e8m0_encode_amax(mid)
    assert int(sf_mid.item()) != 255
    assert int(sf_mid.item()) < 254
    # Zero -> 0
    assert int(e8m0_encode_amax(torch.tensor([0.0])).item()) == 0


def test_e8m0_saturating_threshold():
    # amax exactly 6 * 2^s should map to that s, amax just above -> s+1
    s = 5
    scale = 2 ** s
    amax_exact = 6.0 * scale
    sf = e8m0_encode_amax(torch.tensor([amax_exact]))
    assert int(sf.item()) == s + 127
    amax_plus = amax_exact * (1 + 2 ** -20)
    sf2 = e8m0_encode_amax(torch.tensor([amax_plus]))
    assert int(sf2.item()) == s + 1 + 127


def test_mx_block_quant_never_clips():
    torch.manual_seed(1)
    x = torch.randn(4, 64) * torch.exp2(torch.randint(-10, 10, (4, 1)).float())
    q, sf = quantize_mxfp4_block(x)
    dq = dequant_mxfp4_block(q, sf)
    # Dequant should be finite and not huge vs orig
    assert torch.isfinite(dq).all()
    # Each group's scaled max <=6
    for r in range(4):
        for g in range(2):
            block = x[r, g * 32 : (g + 1) * 32]
            s = float(e8m0_decode(sf[r, g]).item())
            scaled = (block / s).abs().max().item() if s != 0 else 0
            # after encode, the quantized scaled values <=6
            assert scaled <= 6 + 1e-5 or s == 2 ** -127


def test_codec_rejects_bad_shapes():
    fmt = Mxfp4CbFormat(k=16)
    cbs = [torch.zeros((1 << 8, 4)), torch.zeros((1 << 8, 4))]
    # K not multiple of 256
    with pytest.raises(ValueError, match="multiple of superblock"):
        encode_mxfp4_cb(torch.zeros(2, 300), fmt, cbs)
    # non-2-D
    with pytest.raises(ValueError, match="2-D"):
        encode_mxfp4_cb(torch.zeros(512), fmt, cbs)
    # wrong dtype
    with pytest.raises(TypeError, match="float"):
        encode_mxfp4_cb(torch.zeros(2, 512, dtype=torch.int32), fmt, cbs)
    # decode with wrong qweight bytes
    w = torch.zeros(2, 512, dtype=torch.float32)
    qw = encode_mxfp4_cb(w, fmt, cbs)
    with pytest.raises(ValueError, match="row_bytes"):
        decode_mxfp4_cb(qw[:, :10], fmt, cbs, 2, 512)
    with pytest.raises(ValueError, match="K.*multiple"):
        decode_mxfp4_cb(qw, fmt, cbs, 2, 300)


def test_codec_rejects_wrong_codebook_shape():
    fmt = Mxfp4CbFormat(k=16)
    # product k=16 => w0=8,w1=8 => each 256x4; give wrong shape
    bad = [torch.zeros((10, 4)), torch.zeros((256, 4))]
    w = torch.randn(1, 256)
    with pytest.raises(ValueError):
        encode_mxfp4_cb(w, fmt, bad)


def test_format_validation():
    with pytest.raises(ValueError):
        Mxfp4CbFormat(k=0)
    with pytest.raises(ValueError):
        Mxfp4CbFormat(k=16, mode="product", n_sub=1)
    with pytest.raises(ValueError):
        Mxfp4CbFormat(k=16, group_size=16)
    fmt = Mxfp4CbFormat(k=12)
    assert fmt.type_size == 4 * 12 + 8
    assert abs(fmt.bpw - (12 / 8 + 0.25)) < 1e-9
