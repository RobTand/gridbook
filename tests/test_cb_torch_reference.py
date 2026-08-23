"""CPU checks for the independent packed-format oracle used by CUDA tests."""
from __future__ import annotations

import torch

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
