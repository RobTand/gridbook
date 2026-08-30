"""Standalone contract between Gridbook's runtime and artifact producers."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import re
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


_FP8_READER_RUNGS = list(range(28, 49))
_FP8_PRODUCER_RUNGS = [40, 44, 48]
_NVFP4_READER_RUNGS = list(range(12, 25))
_NVFP4_PRODUCER_RUNGS = list(range(12, 25))


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
    assert contract["contract_version"] == 12
    assert contract["abi_features"] == {
        "dspark_construction_physical_bridge": 1,
        "routed_moe_per_role_codebook_lut": 1,
        "source_fp8_block128_w8a16": 1,
    }


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
        # The DSpark draft is a separate entrypoint with its own top-level
        # loader and physical ``mtp.*`` namespace.
        "vllm.models.deepseek_v4.nvidia.dspark",
    ]
    assert set(profiles["supported_ids"]) == {
        "deepseek_v4", "hy_v3", "laguna", "qwen3", "qwen3_5", "qwen3_5_dense",
    }


def test_contract_pins_current_format_ladders_and_layout_restrictions():
    by_family = {
        item["family"]: item for item in load_runtime_contract()["formats"]
    }
    assert set(by_family) == {
        "NVFP4_CB_K", "FP8_CB_K", "TCQ_E2M1_R256", "TCQ_E4M3_R256",
    }
    # A row's kind decides its vocabulary; the two CB rows keep theirs.
    assert by_family["NVFP4_CB_K"]["kind"] == "cb_product"
    assert by_family["FP8_CB_K"]["kind"] == "cb_product"
    # The signed NVFP4_CB_S family was removed from the runtime (2026-08-23);
    # a row for it must not outlive its enforcement sites.
    assert "NVFP4_CB_S" not in by_family
    assert by_family["NVFP4_CB_K"]["rungs"] == _NVFP4_READER_RUNGS
    assert (by_family["NVFP4_CB_K"]["producer_rungs"]
            == _NVFP4_PRODUCER_RUNGS)
    assert by_family["NVFP4_CB_K"]["layout_versions"] == [1, 2]
    assert by_family["NVFP4_CB_K"]["moe_layout_versions"] == [2]
    assert by_family["FP8_CB_K"]["rungs"] == _FP8_READER_RUNGS
    assert by_family["FP8_CB_K"]["producer_rungs"] == _FP8_PRODUCER_RUNGS
    # Historical FP8 readers remain independently loadable even when they are
    # not canonical producer choices.  The pre-release low-rung candidates are
    # absent from both artifact domains.
    assert {28, 32, 36} <= (
        set(_FP8_READER_RUNGS) - set(_FP8_PRODUCER_RUNGS))
    assert set((4, 8, 12, 16, 20, 24)).isdisjoint(_FP8_READER_RUNGS)
    assert set(range(1, 12)).isdisjoint(_NVFP4_READER_RUNGS)
    assert 25 not in _NVFP4_READER_RUNGS
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


def test_platform_lanes_pin_structural_routes_without_device_claims():
    """Cross-compilation is neither a 4090 nor RTX 50 serve qualification."""

    lanes = load_runtime_contract()["lane_eligibility"]
    assert lanes["schema"] == "gridbook.lane-eligibility.v3"
    assert lanes["platforms"] == {
        "sm_89": {"compute_capability": [8, 9]},
        "sm_120": {"compute_capability": [12, 0]},
        "sm_121": {"compute_capability": [12, 1]},
    }
    assert lanes["regimes"] == ["decode", "batch"]
    assert lanes["structures"] == ["dense", "routed_moe"]
    by_id = {cell["id"]: cell for cell in lanes["cells"]}
    assert set(by_id) == {
        "fp8_cb_dense_sm89_decode_cuda_gemv",
        "fp8_cb_dense_sm89_batch_expand_cutlass_w8a8",
        "nvfp4_cb_dense_sm120_decode_cuda_gemv",
        "nvfp4_cb_dense_sm120_batch_expand_bf16",
        "nvfp4_cb_routed_sm120_decode_cuda_gemv",
        "nvfp4_cb_routed_sm120_batch_persistent_b",
        "nvfp4_cb_routed_sm120_batch_expand_bf16",
        "fp8_cb_dense_sm120_decode_cuda_gemv",
        "fp8_cb_dense_sm120_batch_expand_cutlass_w8a8",
        "fp8_cb_routed_sm120_decode_cuda_gemv",
        "fp8_cb_routed_sm120_batch_persistent_b",
        "fp8_cb_routed_sm120_batch_expand_bf16",
        "trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8",
        "trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8",
        "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4",
        "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4",
    }
    cb_cells = [cell for cell in by_id.values()
                if cell["platform"] != "sm_121"]

    sm89 = [cell for cell in by_id.values() if cell["platform"] == "sm_89"]
    assert {(cell["structure"], cell["regime"], cell["route_status"])
            for cell in sm89} == {
        ("dense", "decode", "backed"),
        ("dense", "batch", "backed"),
    }
    assert all(cell["family"] == "FP8_CB_K" for cell in sm89)
    assert all(cell["rungs"] == _FP8_PRODUCER_RUNGS for cell in sm89)

    sm120_nvfp4 = [cell for cell in by_id.values()
                   if cell["platform"] == "sm_120"
                   and cell["family"] == "NVFP4_CB_K"]
    assert {(cell["structure"], cell["regime"], cell["route_status"])
            for cell in sm120_nvfp4} == {
        ("dense", "decode", "backed"),
        ("dense", "batch", "fallback"),
        ("routed_moe", "decode", "backed"),
        ("routed_moe", "batch", "backed"),
        ("routed_moe", "batch", "fallback"),
    }
    assert all(cell["rungs"] == _NVFP4_PRODUCER_RUNGS
               for cell in sm120_nvfp4)

    sm120_fp8 = [cell for cell in by_id.values()
                 if cell["platform"] == "sm_120"
                 and cell["family"] == "FP8_CB_K"]
    assert {(cell["structure"], cell["regime"], cell["route_status"])
            for cell in sm120_fp8} == {
        ("dense", "decode", "backed"),
        ("dense", "batch", "backed"),
        ("routed_moe", "decode", "backed"),
        ("routed_moe", "batch", "backed"),
        ("routed_moe", "batch", "fallback"),
    }
    assert all(cell["rungs"] == _FP8_PRODUCER_RUNGS for cell in sm120_fp8)
    persistent = by_id["nvfp4_cb_routed_sm120_batch_persistent_b"]
    assert persistent["predicates"] == [
        {"fact": "role_split", "op": "equals", "value": False}
    ]
    fp8_persistent = by_id["fp8_cb_routed_sm120_batch_persistent_b"]
    assert fp8_persistent["predicates"] == [
        {"fact": "role_split", "op": "equals", "value": False}
    ]
    # Every CB cell stays a cross-compilation claim. The four trellis cells
    # are the only device-qualified ones, and they are checked against their
    # own laws in tests/test_runtime_contract_trellis.py.
    assert all(cell["qualification"] == "compile_only" for cell in cb_cells)
    assert all(cell["requires_serve_flags"] == [] for cell in cb_cells)
    assert all(cell["platform"] in {"sm_89", "sm_120"} for cell in cb_cells)


def _wrong_schema(contract):
    contract["schema"] = "gridbook.runtime-contract.v999"


def _wrong_contract_version(contract):
    contract["contract_version"] = 2


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


def _missing_per_role_lut_capability(contract):
    del contract["abi_features"]["routed_moe_per_role_codebook_lut"]


def _wrong_per_role_lut_capability(contract):
    contract["abi_features"]["routed_moe_per_role_codebook_lut"] = 2


def _missing_dspark_bridge_capability(contract):
    del contract["abi_features"]["dspark_construction_physical_bridge"]


def _wrong_dspark_bridge_capability(contract):
    contract["abi_features"]["dspark_construction_physical_bridge"] = 2


def _missing_source_fp8_w8a16_capability(contract):
    del contract["abi_features"]["source_fp8_block128_w8a16"]


def _wrong_source_fp8_w8a16_capability(contract):
    contract["abi_features"]["source_fp8_block128_w8a16"] = 2


def _unsupported_format_mode(contract):
    contract["formats"][0]["mode"] = "full"


def _producer_rung_off_law(contract):
    contract["formats"][1]["producer_rungs"].remove(48)


def _manual_cross_family_preference(contract):
    contract["formats"][0]["prefer_over"] = "FP8_CB_K"


def _legacy_irregular_rung_claimed_by_lane(contract):
    contract["lane_eligibility"]["cells"][0]["rungs"] = [29]


def _unsupported_nvfp4_rung_claimed_by_lane(contract):
    contract["lane_eligibility"]["cells"][2]["rungs"].append(26)


def _explicit_unbacked_lane(contract):
    contract["lane_eligibility"]["cells"][0]["route_status"] = "unbacked"


def _unknown_lane_key(contract):
    contract["lane_eligibility"]["cells"][0]["detail"] = "not schema"


def _wrong_platform_capability(contract):
    contract["lane_eligibility"]["platforms"]["sm_89"][
        "compute_capability"
    ] = [9, 0]


def _duplicate_lane_id(contract):
    contract["lane_eligibility"]["cells"][1]["id"] = (
        contract["lane_eligibility"]["cells"][0]["id"]
    )


def _flag_on_unflagged_lane(contract):
    contract["lane_eligibility"]["cells"][0][
        "requires_serve_flags"
    ] = ["GRIDBOOK_TEST=1"]


def _unknown_lane_predicate_fact(contract):
    contract["lane_eligibility"]["cells"][0]["predicates"] = [
        {"fact": "model_id", "op": "equals", "value": "anything"}
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_wrong_schema, "contract.schema"),
        (_wrong_contract_version, "contract.contract_version"),
        (_canonical_not_accepted, "canonical must be accepted"),
        (_duplicate_rung, "sorted, unique"),
        (_unsupported_family_layout, "unsupported ABI layout"),
        (_wrong_loader_module, "vllm.model_executor.models"),
        (_non_vllm_loader_root, "vllm.model_executor.models or vllm.models"),
        (_empty_supported_ids, "non-empty JSON array"),
        (_wrong_scale_plane, "type_size_rules"),
        (_missing_per_role_lut_capability,
         "routed_moe_per_role_codebook_lut"),
        (_wrong_per_role_lut_capability, "must be 1"),
        (_missing_dspark_bridge_capability,
         "dspark_construction_physical_bridge"),
        (_wrong_dspark_bridge_capability, "must be 1"),
        (_missing_source_fp8_w8a16_capability,
         "source_fp8_block128_w8a16"),
        (_wrong_source_fp8_w8a16_capability, "must be 1"),
        (_unsupported_format_mode, "unsupported grid/mode pair"),
        (_producer_rung_off_law, "canonical producer ladder"),
        (_manual_cross_family_preference, "unknown field"),
        (_legacy_irregular_rung_claimed_by_lane, "non-producer rungs"),
        (_unsupported_nvfp4_rung_claimed_by_lane, "non-producer rungs"),
        (_explicit_unbacked_lane, "closed-world absence"),
        (_unknown_lane_key, "unknown field"),
        (_wrong_platform_capability, "platform id must be the exact"),
        (_duplicate_lane_id, "duplicate cell id"),
        (_flag_on_unflagged_lane, "requires_serve_flags must be empty"),
        (_unknown_lane_predicate_fact, "must be one of"),
    ],
)
def test_validation_rejects_incompatible_contracts(mutate, message):
    contract = copy.deepcopy(load_runtime_contract())
    mutate(contract)
    with pytest.raises(RuntimeContractError, match=message):
        validate_runtime_contract(contract)


# --- the version-pin set is derived, not remembered --------------------------

#: Every file that pins the contract schema or its version number, relative to
#: the repository root.  Bumping the schema means editing all of them in ONE
#: commit: they are coupled by meaning, not by imports, so a missed one breaks
#: nothing at the time and ships a stale pin instead.  The set was verified
#: identical on both sides of the v7 -> v8 bump (T1/T2, 2026-08-23) and again
#: at v8 -> v9; the literal below exists so that adding a NEW pin forces a
#: conscious edit here rather than a silent miss at v10.
_VERSION_PIN_FILES = frozenset({
    "CHANGELOG.md",
    "docs/PLUGIN.md",
    "gridbook/runtime_contract.json",
    "gridbook/runtime_contract.py",
    "tests/test_runtime_contract.py",
    "tests/test_runtime_contract_ep.py",
    "tests/test_runtime_contract_tp.py",
    # Added at v9.  This file asserts the SHAPE of every tensor-parallel row
    # but pinned its schema only in a prose docstring, which no grep sees:
    # the v8 bump left it reading "v7", and the v9 bump broke it for real
    # (a third unit kind fell through its two-kind branch).  It now asserts
    # the schema string, so it is in this set and the next bump must visit it.
    "tests/test_target_namespace_compat.py",
})

#: The changelog is the ONE place a superseded schema string is still true:
#: its entries are history and must keep naming the version they landed in.
_STALE_SCHEMA_EXEMPT = frozenset({"CHANGELOG.md"})

_PIN_PATTERN = re.compile(r"contract_version|runtime-contract\.v")
_SCHEMA_PATTERN = re.compile(r"gridbook\.runtime-contract\.v(\d+)")
_PIN_SUFFIXES = (".py", ".json", ".md")


def _repo_root() -> Path:
    """The source checkout to walk for pins.

    In-tree that is this file's grandparent. The installed-wheel release gate
    stages ``tests/`` outside the checkout so the local package cannot shadow
    the wheel, and exports ``GRIDBOOK_SOURCE_ROOT`` as a data-file locator
    (never on ``PYTHONPATH``); ``GITHUB_WORKSPACE`` is the CI spelling.
    """
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        if (root / "gridbook" / "__init__.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "no Gridbook source checkout: run in-tree or set GRIDBOOK_SOURCE_ROOT")


def _scan_for_version_pins() -> dict[str, str]:
    """Every tracked-looking source file whose text pins the contract version.

    Deliberately a walk rather than a shell-out: the point is that the set is
    DERIVED from the tree at test time and compared against the literal above,
    so a new pin cannot appear without this test failing.
    """

    root = _repo_root()
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if not name.startswith(".")
            and name not in {"__pycache__", "build"}
        )
        for filename in sorted(filenames):
            if not filename.endswith(_PIN_SUFFIXES):
                continue
            path = Path(dirpath) / filename
            text = path.read_text(encoding="utf-8", errors="ignore")
            if _PIN_PATTERN.search(text):
                found[str(path.relative_to(root))] = text
    return found


def test_every_contract_version_pin_is_in_the_declared_set():
    assert set(_scan_for_version_pins()) == _VERSION_PIN_FILES


def test_no_pin_file_carries_a_stale_schema_string():
    """The current version is spelled out everywhere; older ones nowhere.

    A pin that still names an older schema is the failure mode this guards:
    it is not an import error, not a test failure anywhere else, and it ships.
    """

    assert RUNTIME_CONTRACT_SCHEMA == "gridbook.runtime-contract.v12"
    current = int(_SCHEMA_PATTERN.fullmatch(RUNTIME_CONTRACT_SCHEMA).group(1))

    for name, text in _scan_for_version_pins().items():
        assert RUNTIME_CONTRACT_SCHEMA in text, \
            f"{name} pins the contract but never names {RUNTIME_CONTRACT_SCHEMA}"
        if name in _STALE_SCHEMA_EXEMPT:
            continue
        stale = sorted({int(v) for v in _SCHEMA_PATTERN.findall(text)
                        if int(v) < current})
        assert not stale, \
            f"{name} still names superseded contract schema version(s) {stale}"
