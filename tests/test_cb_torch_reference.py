"""CPU checks for the independent packed-format oracle used by CUDA tests."""
from __future__ import annotations

import torch
import pytest

from cb_torch_reference import (
    decode_cb_values,
    extract_codewords,
    reconstruct_cb_weight,
)


def _pack_row(codes: list[int], k_bits: int, tail: bytes = b"") -> torch.Tensor:
    stream = 0
    for vector, code in enumerate(codes):
        stream |= int(code) << (vector * k_bits)
    index_bytes = stream.to_bytes(4 * k_bits, "little")
    return torch.tensor(list(index_bytes + tail), dtype=torch.uint8)


_FP8_PRODUCER_RUNGS = tuple(range(4, 49, 4))
_FP8_LEGACY_IRREGULAR_RUNGS = (29, 33, 47)


@pytest.mark.parametrize("k_bits", _FP8_PRODUCER_RUNGS)
def test_fp8_product_decode_covers_every_v10_producer_rung(k_bits):
    """Exact 4*k-byte rows decode at K4..K48 without implicit tail bytes."""

    mask = (1 << k_bits) - 1
    codes = [((vector * 0x9E3779B1) ^ (mask >> (vector % 5))) & mask
             for vector in range(32)]
    # Make the final K4 codeword consume the final nibble of the exact body;
    # this is the lane whose native aligned widx+1 read must be predicated.
    codes[-1] = mask
    packed = _pack_row(codes, k_bits).reshape(1, -1)
    assert packed.shape[1] == 4 * k_bits

    width = k_bits // 4
    tables = []
    for sub in range(4):
        table = (torch.arange((1 << width) * 2, dtype=torch.float32)
                 + sub * 10000).to(torch.bfloat16)
        tables.append(table)
    codebook = torch.cat(tables)
    offsets = torch.zeros(1, dtype=torch.int32)

    got_codes = extract_codewords(
        packed, N=1, K=256, k_bits=k_bits, type_size=4 * k_bits)
    assert torch.equal(got_codes[0, 0], torch.tensor(codes))
    got = decode_cb_values(
        packed, codebook, offsets, N=1, K=256, k_bits=k_bits,
        n_sub=4, type_size=4 * k_bits)

    expected = []
    sub_mask = (1 << width) - 1
    for code in codes:
        for sub, table in enumerate(tables):
            index = (code >> (sub * width)) & sub_mask
            expected.extend(table[index * 2:index * 2 + 2])
    assert torch.equal(got[0], torch.stack(expected))


@pytest.mark.parametrize("k_bits", _FP8_LEGACY_IRREGULAR_RUNGS)
def test_fp8_legacy_irregular_reader_keeps_ceil_first_split(k_bits):
    """v10 producer law narrows; the older K28..K48 reader ABI does not."""

    widths = [k_bits // 4 + (1 if i < k_bits % 4 else 0)
              for i in range(4)]
    codes = [((vector + 1) * 0x123456789ABC) & ((1 << k_bits) - 1)
             for vector in range(32)]
    packed = _pack_row(codes, k_bits).reshape(1, -1)
    tables = [
        (torch.arange((1 << width) * 2, dtype=torch.float32)
         + sub * 10000).to(torch.bfloat16)
        for sub, width in enumerate(widths)
    ]
    got = decode_cb_values(
        packed, torch.cat(tables), torch.zeros(1, dtype=torch.int32),
        N=1, K=256, k_bits=k_bits, n_sub=4, type_size=4 * k_bits)

    expected = []
    bit_offset = 0
    for code in codes:
        bit_offset = 0
        for width, table in zip(widths, tables):
            index = (code >> bit_offset) & ((1 << width) - 1)
            expected.extend(table[index * 2:index * 2 + 2])
            bit_offset += width
    assert torch.equal(got[0], torch.stack(expected))


def test_uneven_product_decode_and_row_offsets():
    k_bits, n_sub, N, K = 13, 2, 2, 256
    codes = [((vector * 3) & 0x7F) | (((31 - vector) & 0x3F) << 7)
             for vector in range(32)]
    row = _pack_row(codes, k_bits)
    packed = torch.stack((row, row))

    table0 = torch.arange(128 * 4, dtype=torch.float32).to(torch.bfloat16)
    table1 = (-torch.arange(64 * 4, dtype=torch.float32)).to(torch.bfloat16)
    block_a = torch.cat((table0, table1))
    block_b = (block_a.float() * 0.25).to(torch.bfloat16)
    codebook = torch.cat((block_a, block_b))
    offsets = torch.tensor([0, block_a.numel()], dtype=torch.int32)

    got_codes = extract_codewords(
        packed, N=N, K=K, k_bits=k_bits, type_size=4 * k_bits)
    assert torch.equal(got_codes[0, 0], torch.tensor(codes))

    got = decode_cb_values(
        packed, codebook, offsets, N=N, K=K, k_bits=k_bits,
        n_sub=n_sub, type_size=4 * k_bits)
    expected_rows = []
    for block in (block_a, block_b):
        vectors = []
        for code in codes:
            i0, i1 = code & 0x7F, (code >> 7) & 0x3F
            vectors.append(torch.cat((block[i0 * 4:(i0 + 1) * 4],
                                      block[128 * 4 + i1 * 4:
                                            128 * 4 + (i1 + 1) * 4])))
        expected_rows.append(torch.cat(vectors))
    assert torch.equal(got, torch.stack(expected_rows))


def test_fp8_and_two_tier_weight_rounding():
    k_bits, n_sub, N, K = 13, 2, 1, 256
    codes = [((vector * 5) & 0x7F) | (((vector + 7) & 0x3F) << 7)
             for vector in range(32)]
    body = _pack_row(codes, k_bits)
    codebook = torch.linspace(-2, 2, 128 * 4 + 64 * 4).to(torch.bfloat16)
    offsets = torch.zeros(1, dtype=torch.int32)
    dummy = torch.zeros(1)

    values = decode_cb_values(
        body.reshape(1, -1), codebook, offsets, N=N, K=K,
        k_bits=k_bits, n_sub=n_sub, type_size=4 * k_bits)
    fp8 = reconstruct_cb_weight(
        body.reshape(1, -1), codebook, offsets, torch.tensor([0.75]), dummy,
        N=N, K=K, k_bits=k_bits, n_sub=n_sub, type_size=4 * k_bits,
        is_fp4=False)
    assert torch.equal(fp8, (values.float() * 0.75).to(torch.bfloat16))

    # E=127 and sub-code 0 select scale 1.0 for all sixteen groups.
    v2 = torch.cat((body, torch.tensor([127] + [0] * 8, dtype=torch.uint8)))
    compose = torch.zeros(256, 16)
    compose[127, 0] = 1.0
    fp4 = reconstruct_cb_weight(
        v2.reshape(1, -1), codebook, offsets, dummy, compose.reshape(-1),
        N=N, K=K, k_bits=k_bits, n_sub=n_sub,
        type_size=4 * k_bits + 9, is_fp4=True, is_v2=True)
    assert torch.equal(fp4, values.to(torch.bfloat16))
