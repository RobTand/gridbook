"""Public K1..K25 laws and direct-kernel research laws through K32."""
from __future__ import annotations

import pytest
import torch

from gridbook import codec
from gridbook.runtime_contract import load_runtime_contract


def test_runtime_publishes_only_the_supported_product_domain():
    row = next(item for item in load_runtime_contract()["formats"]
               if item["family"] == "NVFP4_CB_K")
    assert row["rungs"] == list(range(1, 26))
    assert row["producer_rungs"] == list(range(1, 26))
    assert row["mode"] == "product"
    assert row["n_sub"] == 2


def test_k26_k32_are_below_the_public_contract():
    row = next(item for item in load_runtime_contract()["formats"]
               if item["family"] == "NVFP4_CB_K")
    assert set(range(26, 33)).isdisjoint(row["rungs"])
    assert set(range(26, 33)).isdisjoint(row["producer_rungs"])


@pytest.mark.parametrize("k_bits", range(1, 33))
def test_product_subtable_shapes_and_flat_size(k_bits):
    """Exercise the codec primitive, including research-only K26..K32."""
    shapes = codec.product_subtable_shapes(k_bits, 2)
    assert shapes == (
        (1 << ((k_bits + 1) // 2), 4),
        (1 << (k_bits // 2), 4),
    )
    tables = [torch.zeros(shape, dtype=torch.bfloat16) for shape in shapes]
    flat = codec.build_flat_product_codebook(
        tables, k_bits, 2, f"NVFP4_CB_K{k_bits}", "fp4")
    assert flat.numel() == sum(rows * cols for rows, cols in shapes)


def test_k1_zero_width_and_k32_lattice_ceiling_shapes():
    assert codec.product_subtable_shapes(1, 2) == ((2, 4), (1, 4))
    assert codec.product_subtable_shapes(32, 2) == (
        (65536, 4), (65536, 4))
    assert sum(r * c * 2 for r, c in
               codec.product_subtable_shapes(32, 2)) == 1_048_576


def test_product_codebook_shape_mismatch_fails_before_cuda():
    with pytest.raises(ValueError, match=r"sub1 must have shape \(1, 4\)"):
        codec.build_flat_product_codebook(
            [torch.zeros(2, 4), torch.zeros(0, 4)], 1, 2,
            "NVFP4_CB_K1", "fp4")
    with pytest.raises(ValueError, match="requires 2 sub-codebooks"):
        codec.build_flat_product_codebook(
            [torch.zeros(65536, 4)], 32, 2, "NVFP4_CB_K32", "fp4")


@pytest.mark.parametrize("k_bits,n_sub", [
    (1, 2), (16, 2), (25, 2), (26, 2), (32, 2),
    (4, 4), (44, 4), (48, 4),
])
def test_strict_product_builder_preserves_legacy_flat_bytes(k_bits, n_sub):
    """The strict loader is an ABI check, not a new flat representation.

    Canonical sidecars keep producing exactly the tensor the historical
    ``build_flat_codebook`` path did, including dtype, order and contiguity.
    Tuple input is intentional: callers have never been required to allocate
    a mutable list merely to flatten an ordered product book.
    """

    tables = tuple(
        torch.zeros(shape, dtype=torch.float16)
        for shape in codec.product_subtable_shapes(k_bits, n_sub)
    )
    legacy = codec.build_flat_codebook(tables)
    strict = codec.build_flat_product_codebook(tables, k_bits, n_sub)
    assert strict.dtype == legacy.dtype == torch.bfloat16
    assert strict.is_contiguous()
    assert torch.equal(strict.view(torch.uint16), legacy.view(torch.uint16))


@pytest.mark.parametrize("k_bits,staged", [(1, True), (24, True),
                                             (25, True), (26, False),
                                             (32, False)])
def test_expander_policy_closed_form(k_bits, staged):
    dictionary_bytes = 2 * sum(
        rows * cols
        for rows, cols in codec.product_subtable_shapes(k_bits, 2))
    assert (dictionary_bytes <= 99 * 1024) is staged
