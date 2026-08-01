"""Producer/runtime ABI gates for static native-NVFP4 activations."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gridbook.nvfp4_activation_contract import (
    CONTRACT_KEY,
    CONTRACT_SCHEMA,
    EXECUTION_CONTRACT,
    FULL_E4M3_POLICY,
    GROUP_SIZE,
    LEGACY_POLICY,
    TENSOR_SUFFIX,
    VALUE_DTYPE,
    parse_contract,
    reciprocal_vector,
    require_identical_loaded_scales,
    target_values_sha256,
    validate_payload,
)


def _record(scales=None, *, policy=LEGACY_POLICY):
    scales = {"a": 1.0, "b": 2.0} if scales is None else scales
    return {
        "schema": CONTRACT_SCHEMA,
        "contract": EXECUTION_CONTRACT,
        "group_size": GROUP_SIZE,
        "tensor_suffix": TENSOR_SUFFIX,
        "value_dtype": VALUE_DTYPE,
        "input_global_scale_policy": policy,
        "target_count": len(scales),
        "target_names": sorted(scales),
        "target_values_sha256": target_values_sha256(scales, policy=policy),
    }


def test_digest_matches_producer_cross_repo_vector():
    assert target_values_sha256(
        {"b": 2.0, "a": 1.0}, policy=LEGACY_POLICY
    ) == "5207c30737409ae6d16586f1f169efc8f56948bee51031e1610683f0fee08d0f"


def test_absent_contract_is_legacy_compatible():
    assert parse_contract({}) is None
    assert parse_contract({"execution_contracts": {"future": {}}}) is None


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema", "v2", "schema"),
        ("contract", "dynamic", "contract"),
        ("group_size", 32, "group_size"),
        ("tensor_suffix", "other", "tensor_suffix"),
        ("value_dtype", "float16", "value_dtype"),
        ("input_global_scale_policy", "unknown", "policy"),
        ("target_count", 0, "positive integer"),
        ("target_names", ["b", "a"], "sorted"),
        ("target_values_sha256", "ABC", "lowercase hexadecimal"),
    ],
)
def test_malformed_contract_fails_closed(field, value, match):
    record = _record()
    record[field] = value
    with pytest.raises(ValueError, match=match):
        parse_contract({"execution_contracts": {CONTRACT_KEY: record}})


def test_payload_rejects_missing_extra_bad_dtype_shape_value_and_digest():
    record = _record()
    assert validate_payload(record, {
        "a": torch.tensor([1.0], dtype=torch.float32),
        "b": torch.tensor([2.0], dtype=torch.float32),
    }) == {"a": 1.0, "b": 2.0}

    with pytest.raises(ValueError, match="target_count"):
        validate_payload(record, {"a": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="target_count"):
        validate_payload(record, {
            "a": torch.tensor([1.0]), "b": torch.tensor([2.0]),
            "c": torch.tensor([3.0]),
        })
    wrong_names = _record({"a": 1.0, "c": 2.0})
    with pytest.raises(ValueError, match="physical target set mismatch"):
        validate_payload(wrong_names, {
            "a": torch.tensor([1.0]), "b": torch.tensor([2.0]),
        })
    with pytest.raises(ValueError, match="float32 shape"):
        validate_payload(record, {
            "a": torch.tensor([1.0], dtype=torch.float16),
            "b": torch.tensor([2.0]),
        })
    with pytest.raises(ValueError, match="float32 shape"):
        validate_payload(record, {
            "a": torch.tensor(1.0), "b": torch.tensor([2.0]),
        })
    with pytest.raises(ValueError, match="finite and > 0"):
        validate_payload(record, {
            "a": torch.tensor([float("nan")]), "b": torch.tensor([2.0]),
        })
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_payload(record, {
            "a": torch.tensor([1.0]), "b": torch.tensor([3.0]),
        })


def test_merged_scale_slots_and_physical_targets_require_identical_f32_bits():
    scale = require_identical_loaded_scales(
        torch.tensor([3.25, 3.25]), prefix="qkv", expected=[3.25, 3.25]
    )
    assert torch.equal(scale, torch.tensor([3.25]))
    with pytest.raises(ValueError, match="non-identical"):
        require_identical_loaded_scales(
            torch.tensor([3.25, 3.5]), prefix="qkv", expected=[3.25, 3.25]
        )
    with pytest.raises(ValueError, match="finite and > 0"):
        require_identical_loaded_scales(
            torch.tensor([float("nan")]), prefix="qkv", expected=[3.25]
        )
    with pytest.raises(ValueError, match="non-identical"):
        require_identical_loaded_scales(
            torch.tensor([3.25]), prefix="qkv", expected=[3.5]
        )


def test_reciprocal_vector_is_cached_and_bounded():
    layer = type("Layer", (), {})()
    scale = torch.tensor([4.0], dtype=torch.float32)
    first = reciprocal_vector(layer, which="dense", scale=scale, rows=17)
    assert first.data_ptr() == reciprocal_vector(
        layer, which="dense", scale=scale, rows=17
    ).data_ptr()
    assert torch.equal(first, torch.full((17,), 0.25))
    for rows in range(1, 20):
        reciprocal_vector(layer, which="w13", scale=scale, rows=rows)
    assert len(layer._cb_fp4_reciprocal_cache) <= 8


def test_policy_is_digest_bound():
    scales = {"a": 1.0}
    assert target_values_sha256(scales, policy=LEGACY_POLICY) != \
        target_values_sha256(scales, policy=FULL_E4M3_POLICY)
