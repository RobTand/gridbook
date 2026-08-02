"""Producer/runtime ABI gates for static native-NVFP4 activations."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gridbook.nvfp4_activation_contract import (
    CONTRACT_KEY,
    CONTRACT_SCHEMA,
    CONTRACT_SCHEMA_V2,
    EXECUTION_CONTRACT,
    FULL_E4M3_POLICY,
    GROUP_SIZE,
    LEGACY_POLICY,
    MSE_GRID_POLICY,
    ROUTED_MOE_STAGES,
    ROUTED_MOE_STAGE_KEY,
    ROUTED_MOE_STAGE_SCHEMA,
    SOURCE_PARENT_MODULE_CACHE,
    SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    TENSOR_SUFFIX,
    VALUE_DTYPE,
    parse_contract,
    parse_routed_moe_stages,
    reciprocal_vector,
    require_identical_loaded_scales,
    routed_moe_stages_sha256,
    stage_values_sha256,
    target_values_sha256,
    validate_payload,
    verify_routed_moe_stages,
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


_W13 = "m.experts.gate_up_proj"
_W2 = "m.experts.down_proj"
_STAGE_SOURCES = {
    "w13": SOURCE_PARENT_MODULE_CACHE,
    "w2": SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
}


def _stage_modules(scales, *, policy=LEGACY_POLICY, sources=None):
    sources = sources or _STAGE_SOURCES
    targets = {"w13": _W13, "w2": _W2}
    return {
        "m.experts": {
            stage: {
                "stage": stage,
                "target": targets[stage],
                "input_global_scale_policy": policy,
                "calibration_source": sources[stage],
                "stage_values_sha256": stage_values_sha256(
                    stage=stage,
                    target=targets[stage],
                    policy=policy,
                    calibration_source=sources[stage],
                    value=scales[targets[stage]],
                ),
            }
            for stage in ROUTED_MOE_STAGES
        }
    }


def _stage_record(scales=None, *, policy=LEGACY_POLICY, sources=None,
                  modules=None):
    scales = {_W13: 1.0, _W2: 2.0} if scales is None else scales
    modules = modules if modules is not None else _stage_modules(
        scales, policy=policy, sources=sources
    )
    record = _record(scales, policy=policy)
    record["schema"] = CONTRACT_SCHEMA_V2
    record[ROUTED_MOE_STAGE_KEY] = {
        "schema": ROUTED_MOE_STAGE_SCHEMA,
        "stages": list(ROUTED_MOE_STAGES),
        "module_count": len(modules),
        "module_names": sorted(modules),
        "modules": modules,
        "stages_sha256": routed_moe_stages_sha256(modules),
    }
    return record


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
    assert scale.ndim == 0
    assert torch.equal(scale, torch.tensor(3.25))
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
    assert target_values_sha256(scales, policy=MSE_GRID_POLICY) not in {
        target_values_sha256(scales, policy=LEGACY_POLICY),
        target_values_sha256(scales, policy=FULL_E4M3_POLICY),
    }


def test_mse_grid_is_valid_v1_static_policy():
    record = _record({"a": 1.25}, policy=MSE_GRID_POLICY)
    parsed = parse_contract({"execution_contracts": {CONTRACT_KEY: record}})
    assert parsed["input_global_scale_policy"] == MSE_GRID_POLICY
    assert validate_payload(parsed, {"a": torch.tensor([1.25])}) == {"a": 1.25}


# ---------------------------------------------------------------------------
# ROADMAP K0.2 — routed-MoE stage attestation
# ---------------------------------------------------------------------------

def test_stage_schema_literals_are_pinned_cross_repo():
    # PrismaQuant pins the identical literals in
    # ``tests/test_nvfp4_activation_contract.py``.  They are a cross-repo ABI.
    assert CONTRACT_SCHEMA == "prismaquant.nvfp4_w4a4_activation.v1"
    assert CONTRACT_SCHEMA_V2 == "prismaquant.nvfp4_w4a4_activation.v2"
    assert ROUTED_MOE_STAGE_SCHEMA == (
        "prismaquant.nvfp4_w4a4_activation_stages.v1"
    )
    assert ROUTED_MOE_STAGE_KEY == "routed_moe_stages"
    assert ROUTED_MOE_STAGES == ("w13", "w2")


def test_stage_digest_matches_producer_cross_repo_vector():
    # Byte-identical to PrismaQuant's pinned stage vectors.
    assert stage_values_sha256(
        stage="w13",
        target="m.experts.gate_up_proj",
        policy=LEGACY_POLICY,
        calibration_source=SOURCE_PARENT_MODULE_CACHE,
        value=1.0,
    ) == "c15c44ac3c290d4e596967218b41ffba2f12a857a2cc2356a1ed4e159a40e630"
    assert stage_values_sha256(
        stage="w2",
        target="m.experts.down_proj",
        policy=LEGACY_POLICY,
        calibration_source=SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        value=2.0,
    ) == "91f005ef177c3c8ccfb1f25a528d0a9a601ef4bdde61db8d33947a1a951cfe2e"
    assert routed_moe_stages_sha256(
        _stage_modules({_W13: 1.0, _W2: 2.0})
    ) == "77c830f2b1989a9a0069dcc7afabbe73f0913ccbfb634287346a0c097e231882"


def test_v1_dense_record_stays_valid_and_carries_no_stage_section():
    record = _record()
    parsed = parse_contract({"execution_contracts": {CONTRACT_KEY: record}})
    assert parsed["schema"] == CONTRACT_SCHEMA
    assert parse_routed_moe_stages(parsed) is None
    verdict = verify_routed_moe_stages(parsed, {"a": 1.0, "b": 2.0})
    assert verdict["verdict"] == "not_attested"
    assert verdict["attested"] is False


def test_stage_attested_v2_record_parses_and_verifies():
    scales = {_W13: 1.0, _W2: 2.0}
    record = _stage_record(scales)
    parsed = parse_contract({"execution_contracts": {CONTRACT_KEY: record}})
    assert parsed["schema"] == CONTRACT_SCHEMA_V2
    # The whole-model digest fields are exactly the v1 fields.
    assert parsed["target_values_sha256"] == target_values_sha256(
        scales, policy=LEGACY_POLICY
    )
    assert validate_payload(parsed, {
        _W13: torch.tensor([1.0]), _W2: torch.tensor([2.0])
    }) == scales
    stages = parse_routed_moe_stages(parsed)
    assert sorted(stages) == ["m.experts"]
    assert sorted(stages["m.experts"]) == ["w13", "w2"]
    verdict = verify_routed_moe_stages(parsed, scales)
    assert verdict["verdict"] == "attested_and_verified"
    assert verdict["modules"] == ["m.experts"]
    assert verdict["failing_module"] is None


def test_schema_and_stage_section_are_one_claim():
    record = _record()
    record[ROUTED_MOE_STAGE_KEY] = _stage_record()[ROUTED_MOE_STAGE_KEY]
    with pytest.raises(ValueError, match="requires schema"):
        parse_contract({"execution_contracts": {CONTRACT_KEY: record}})
    orphan = _stage_record()
    orphan.pop(ROUTED_MOE_STAGE_KEY)
    with pytest.raises(ValueError, match="carries no routed_moe_stages"):
        parse_contract({"execution_contracts": {CONTRACT_KEY: orphan}})
    unknown = _record()
    unknown["schema"] = "prismaquant.nvfp4_w4a4_activation.v3"
    with pytest.raises(ValueError, match="schema"):
        parse_contract({"execution_contracts": {CONTRACT_KEY: unknown}})


def test_half_attested_module_and_illegal_stage_source_fail_closed():
    half = _stage_record()
    half[ROUTED_MOE_STAGE_KEY]["modules"]["m.experts"].pop("w2")
    with pytest.raises(ValueError, match="both .*stages are required"):
        parse_routed_moe_stages(half)
    # w2 calibrated from the experts-module input is exactly the defect the
    # attestation exists to expose.
    with pytest.raises(ValueError, match="not a legal input for that stage"):
        parse_routed_moe_stages(_stage_record(sources={
            "w13": SOURCE_PARENT_MODULE_CACHE,
            "w2": SOURCE_PARENT_MODULE_CACHE,
        }))
    # A stage naming a target the whole-model mapping never declared.
    stray = _stage_modules({_W13: 1.0, _W2: 2.0})
    record = _stage_record(modules=stray)
    record["target_names"] = [_W13]
    record["target_count"] = 1
    with pytest.raises(ValueError, match="not in the contract's target_names"):
        parse_routed_moe_stages(record)
    # A tampered section digest.
    tampered = _stage_record()
    tampered[ROUTED_MOE_STAGE_KEY]["stages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stages_sha256 mismatch"):
        parse_routed_moe_stages(tampered)


def test_stage_verdicts_separate_missing_stages_from_digest_mismatch():
    record = _stage_record({_W13: 1.0, _W2: 2.0})
    missing = verify_routed_moe_stages(record, {_W13: 1.0})
    assert missing["verdict"] == "missing_stages"
    assert missing["failing_module"] == "m.experts"
    assert missing["failing_stage"] == "w2"

    mismatch = verify_routed_moe_stages(record, {_W13: 1.0, _W2: 3.0})
    assert mismatch["verdict"] == "digest_mismatch"
    assert mismatch["failing_stage"] == "w2"

    # Mutating one stage's scale moves only that stage's digest.
    moved = _stage_record({_W13: 1.0, _W2: 4.0})
    base_module = record[ROUTED_MOE_STAGE_KEY]["modules"]["m.experts"]
    moved_module = moved[ROUTED_MOE_STAGE_KEY]["modules"]["m.experts"]
    assert base_module["w13"]["stage_values_sha256"] == (
        moved_module["w13"]["stage_values_sha256"]
    )
    assert base_module["w2"]["stage_values_sha256"] != (
        moved_module["w2"]["stage_values_sha256"]
    )

    malformed = _stage_record()
    malformed[ROUTED_MOE_STAGE_KEY]["module_count"] = 7
    assert verify_routed_moe_stages(malformed, {_W13: 1.0, _W2: 2.0})[
        "verdict"
    ] == "malformed_stage_attestation"
