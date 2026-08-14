"""Exact wire roundtrip and deterministic reconstruction."""
import torch
import pytest

from research.mxfp4_cb.format import Mxfp4CbFormat, SUPERBLOCK, type_size_for_k
from research.mxfp4_cb.codec import pack_indices, unpack_indices, encode_mxfp4_cb, decode_mxfp4_cb
from research.mxfp4_cb.e2m1 import e2m1_encode, e2m1_decode


def _make_product_codebook(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w0 = (k + 1) // 2  # ceil
    w1 = k // 2
    cb0 = torch.randn((1 << w0, 4), generator=g)
    # snap to E2M1 grid so codebook is legal
    cb0 = e2m1_decode(e2m1_encode(cb0)).to(torch.float32)
    cb1 = torch.randn((1 << w1, 4), generator=g)
    cb1 = e2m1_decode(e2m1_encode(cb1)).to(torch.float32)
    return [cb0, cb1]


@pytest.mark.parametrize("k", [12, 13, 16, 20, 24])
def test_pack_unpack_wire_roundtrip(k):
    num_sb = 3
    g = torch.Generator().manual_seed(k)
    codes = torch.randint(0, 1 << k, (num_sb, 32), generator=g, dtype=torch.int64)
    packed = pack_indices(codes, k)
    assert packed.shape == (num_sb, 4 * k)
    assert packed.dtype == torch.uint8
    got = unpack_indices(packed, k)
    assert torch.equal(got, codes)


def test_pack_unpack_byte_boundary_exact():
    """Hand-picked wire bytes for k=12: first code 0b101010101010 etc."""
    k = 12
    codes = torch.tensor([[0xABC, 0x123] + [0] * 30], dtype=torch.int64)
    packed = pack_indices(codes, k)
    # Re-derive via from_bytes and compare
    raw = int.from_bytes(bytes(packed[0].tolist()), "little")
    assert (raw & ((1 << k) - 1)) == 0xABC
    assert ((raw >> k) & ((1 << k) - 1)) == 0x123
    assert torch.equal(unpack_indices(packed, k), codes)


@pytest.mark.parametrize("k", [12, 16])
def test_encode_decode_deterministic_roundtrip(k):
    fmt = Mxfp4CbFormat(k=k, mode="product")
    rows, K = 2, 512
    cbs = _make_product_codebook(k, seed=k)
    g = torch.Generator().manual_seed(k * 10)
    w = torch.randn((rows, K), generator=g, dtype=torch.float32) * 0.05
    # outlier stress
    w[0, ::64] *= 4
    qw = encode_mxfp4_cb(w, fmt, cbs)
    assert qw.shape == (rows, (K // SUPERBLOCK) * fmt.type_size)
    assert qw.dtype == torch.uint8
    w2 = decode_mxfp4_cb(qw, fmt, cbs, rows, K)
    # deterministic: re-encode same w yields identical bytes
    qw2 = encode_mxfp4_cb(w, fmt, cbs)
    assert torch.equal(qw, qw2)
    # decode is deterministic
    w3 = decode_mxfp4_cb(qw, fmt, cbs, rows, K)
    assert torch.equal(w2, w3)
    # sandwich decode: re-encode decoded weight yields *some* bytes (not necessarily same),
    # but decode of that must be stable (idempotent under optimal re-encode? not guaranteed)
    # Instead check wire is self-consistent: decode(encode(w)) is stable on second decode
    # Already above.


def test_row_major_byte_layout():
    """Each row's bytes are contiguous SBs: index bytes then 8 scale bytes."""
    k = 16
    fmt = Mxfp4CbFormat(k=k)
    assert fmt.type_size == 4 * k + 8
    rows, K = 1, 256
    cbs = _make_product_codebook(k)
    w = torch.zeros((rows, K), dtype=torch.float32)
    qw = encode_mxfp4_cb(w, fmt, cbs)
    # For all-zero weight, amax=0 => E8M0 byte 0 for all 8 scales, so scale plane is 8 zeros
    # Index plane is deterministic (nearest to zero vector => smallest distance to zero code)
    assert qw.shape == (1, fmt.type_size)
    # scale bytes are the last 8
    scale_bytes = qw[0, 4 * k :]
    # zero group => E8M0 byte 0
    assert torch.equal(scale_bytes, torch.zeros(8, dtype=torch.uint8))
