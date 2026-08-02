"""Standalone contract between Gridbook's runtime and artifact producers."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path
import subprocess
import sys

import pytest

from gridbook.runtime_contract import (
    RUNTIME_CONTRACT_SCHEMA,
    RuntimeContractError,
    load_runtime_contract,
    validate_runtime_contract,
)


def _plugin_source() -> Path:
    spec = importlib.util.find_spec("gridbook.plugin")
    assert spec is not None and spec.origin
    return Path(spec.origin)


def test_packaged_contract_loads_and_validates():
    resource = files("gridbook").joinpath("runtime_contract.json")
    assert resource.is_file()
    raw = json.loads(resource.read_text(encoding="utf-8"))
    contract = load_runtime_contract()
    assert contract == raw
    assert contract["schema"] == RUNTIME_CONTRACT_SCHEMA


def test_loader_is_vllm_torch_and_prismaquant_free(tmp_path):
    package = Path(importlib.util.find_spec("gridbook").origin).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(package.parent), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    probe = r'''
import builtins
import sys

blocked = {"prismaquant", "torch", "vllm"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("forbidden import: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from gridbook.runtime_contract import load_runtime_contract
contract = load_runtime_contract()
assert contract["quant_method"]["canonical"] == "gridbook"
assert not blocked.intersection(sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_contract_matches_runtime_registration_and_loader_table():
    contract = load_runtime_contract()
    source = _plugin_source().read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The plugin consumes the contract rather than maintaining literal aliases
    # or architecture module paths beside it.
    assert "load_runtime_contract" in source
    assert "register_quantization_config(quant_method)" in source
    assert "vllm.model_executor.models." not in source
    literal_registrations = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_quantization_config"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert literal_registrations == []

    assert contract["quant_method"] == {
        "canonical": "gridbook",
        "accepted": ["gridbook", "prismaquant"],
        "legacy": ["prismaquant"],
    }
    profiles = contract["producer_profiles"]
    assert profiles["top_level_loader_modules"] == [
        "vllm.model_executor.models.hy_v3",
        "vllm.model_executor.models.hy_v3_mtp",
        "vllm.model_executor.models.laguna",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.model_executor.models.lfm2_moe",
        # DeepSeek-V4 is a per-platform package in vLLM 0.24; the entrypoint
        # class is DEFINED in the platform submodule, which is what plugin.py
        # has to match on (ROADMAP D0.1).
        "vllm.models.deepseek_v4.nvidia.model",
    ]
    assert set(profiles["supported_ids"]) == {
        "deepseek_v4", "hy_v3", "laguna", "qwen3", "qwen3_5", "qwen3_5_dense",
    }


def test_contract_pins_current_format_ladders_and_layout_restrictions():
    by_family = {
        item["family"]: item for item in load_runtime_contract()["formats"]
    }
    assert set(by_family) == {"NVFP4_CB_K", "NVFP4_CB_S", "FP8_CB_K"}
    for family in ("NVFP4_CB_K", "NVFP4_CB_S"):
        assert by_family[family]["rungs"] == list(range(12, 25))
        assert by_family[family]["layout_versions"] == [1, 2]
        assert by_family[family]["moe_layout_versions"] == [2]
    assert by_family["FP8_CB_K"]["rungs"] == list(range(28, 49))
    assert by_family["FP8_CB_K"]["layout_versions"] == [1]
    assert by_family["FP8_CB_K"]["moe_layout_versions"] == [1]
    assert load_runtime_contract()["packing"] == {
        "vector_dim": 8,
        "superblock_weights": 256,
        "codewords_per_superblock": 32,
        "index_bytes_per_k": 4,
        "index_bit_order": "lsb_first",
        "subindex_split": "ceil_first",
    }


def _wrong_schema(contract):
    contract["schema"] = "gridbook.runtime-contract.v999"


def _canonical_not_accepted(contract):
    contract["quant_method"]["accepted"] = ["prismaquant"]


def _duplicate_rung(contract):
    contract["formats"][0]["rungs"].append(
        contract["formats"][0]["rungs"][-1]
    )


def _unsupported_family_layout(contract):
    contract["formats"][0]["layout_versions"].append(3)


def _wrong_loader_module(contract):
    contract["producer_profiles"]["top_level_loader_modules"][0] = "hy_v3"


def _non_vllm_loader_root(contract):
    # The two accepted roots are an allow-list, not a bare "vllm." prefix: the
    # entries become dynamic imports into the serving process.
    contract["producer_profiles"]["top_level_loader_modules"][0] = (
        "vllm.model_executor.layers.hy_v3"
    )


def _empty_supported_ids(contract):
    contract["producer_profiles"]["supported_ids"] = []


def _wrong_scale_plane(contract):
    contract["layout"]["type_size_rules"][0]["scale_plane_bytes"] = 15


def _unsupported_format_mode(contract):
    contract["formats"][0]["mode"] = "full"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_wrong_schema, "contract.schema"),
        (_canonical_not_accepted, "canonical must be accepted"),
        (_duplicate_rung, "sorted, unique"),
        (_unsupported_family_layout, "unsupported ABI layout"),
        (_wrong_loader_module, "vllm.model_executor.models"),
        (_non_vllm_loader_root, "vllm.model_executor.models or vllm.models"),
        (_empty_supported_ids, "non-empty JSON array"),
        (_wrong_scale_plane, "type_size_rules"),
        (_unsupported_format_mode, "unsupported grid/mode pair"),
    ],
)
def test_validation_rejects_incompatible_contracts(mutate, message):
    contract = copy.deepcopy(load_runtime_contract())
    mutate(contract)
    with pytest.raises(RuntimeContractError, match=message):
        validate_runtime_contract(contract)
