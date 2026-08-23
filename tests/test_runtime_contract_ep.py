"""The expert-parallel capability table in the packaged runtime contract.

Companion to ``test_runtime_contract_tp.py``, on the other parallelism axis.
Tensor parallelism splits a unit's rows/columns; expert parallelism splits a
routed MoE layer's EXPERTS and leaves every expert whole. CB expert stacks
serve on the second axis and never on the first, so they need their own table
rather than a relaxed row in the existing one.

What is pinned here, without torch, vLLM, or CUDA — stdlib, pytest, and the
stdlib-only contract loader only:

1. The published table matches what the enforcement sites actually enforce.
   The topology predicate is checked against the AST of
   ``gridbook/config.py::_require_ep_moe_serving`` (every field of the
   contract's ``requires`` object must correspond to an attribute that
   function reads), and the admission laws against the AST of
   ``gridbook/moe_ep.py`` and the two loader call sites.
2. The closed-world reading: a unit with no row is REFUSED, and the mixed
   per-expert-format MoE unit publishes an explicit cap of 1 rather than
   simply being absent, so a consumer can tell "refused" from "unknown".
3. ``schema`` and ``contract_version`` move together, and the validator
   refuses a table that drifts from the code in either direction.

Run: ``python -m pytest tests/test_runtime_contract_ep.py -q``.
"""
from __future__ import annotations

import ast
import copy
import json
from importlib.resources import files
from pathlib import Path

import pytest

from gridbook.runtime_contract import (
    RUNTIME_CONTRACT_SCHEMA,
    RuntimeContractError,
    load_runtime_contract,
    validate_runtime_contract,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _packaged_contract() -> dict:
    raw = files("gridbook").joinpath("runtime_contract.json")
    return json.loads(raw.read_text(encoding="utf-8"))


def _ep_table(contract: dict) -> dict:
    return contract["expert_parallel"]


def _units_by_id(contract: dict) -> dict[str, dict]:
    return {row["unit"]: row for row in _ep_table(contract)["units"]}


def _source(rel: str) -> tuple[str, ast.Module]:
    text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
    return text, ast.parse(text)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"no function {name!r}")


# --- 1. The published table ---------------------------------------------------


def test_packaged_table_declares_exactly_the_enforced_capability():
    ep = _ep_table(_packaged_contract())
    assert ep["axis"] == "vllm_expert_parallel_size"
    assert ep["semantics"] == "closed_world"

    units = _units_by_id(load_runtime_contract())
    assert set(units) == {
        "FP8_CB_K",
        "NVFP4_CB_K",
        "cb_moe_per_expert_format_groups",
    }
    # Both CB families serve expert-parallel under identical laws: nothing
    # inside an expert is sharded, so the format grid is irrelevant here
    # (contrast the TP table, where fp4 and fp8 publish different row quanta).
    admission = {
        "shard_axis": "expert",
        "sharded_dims": "none",
        "placement": "monotone_bijection",
        "checkpoint_leading_dim": "global_expert_count",
        "remote_pair_handling": "zero_weight_alias",
        "cross_rank_reduction": "vllm_final_all_reduce",
    }
    for family in ("FP8_CB_K", "NVFP4_CB_K"):
        assert units[family]["kind"] == "cb_moe_expert_stack"
        assert units[family]["expert_admission"] == admission
        # No numeric ceiling: admission is the laws, evaluated per rank at
        # weight construction, not a world size anything compares against.
        assert "max_world_size" not in units[family]

    refused = units["cb_moe_per_expert_format_groups"]
    assert refused["kind"] == "cb_moe_expert_stack_refused"
    assert refused["max_world_size"] == 1
    assert "expert_admission" not in refused


def test_requires_block_matches_the_topology_predicate_source():
    """Every published requirement is a branch of the refusal site.

    The contract may not claim a topology restriction the code does not
    enforce, and the code may not enforce one the contract does not publish —
    the second half is what makes the table usable as a pre-check.
    """
    requires = _ep_table(load_runtime_contract())["requires"]
    assert requires == {
        "vllm_flag": "--enable-expert-parallel",
        "moe_tensor_parallel_size": 1,
        "all2all_kernels": False,
        "expert_load_balancing": False,
        "skip_final_all_reduce": False,
    }

    _, tree = _source("gridbook/config.py")
    gate = _function(tree, "_require_ep_moe_serving")
    # The attributes the gate actually reads off vLLM's parallel config.
    read = {node.value for node in ast.walk(gate)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for attr, field in (
            ("use_ep", "vllm_flag"),
            ("tp_size", "moe_tensor_parallel_size"),
            ("use_all2all_kernels", "all2all_kernels"),
            ("enable_eplb", "expert_load_balancing"),
            ("skip_final_all_reduce", "skip_final_all_reduce"),
    ):
        assert attr in read, (
            f"contract publishes requires.{field} but "
            f"_require_ep_moe_serving never reads {attr!r}")
    # It must raise, and it must name the flag an operator has to add.
    assert any(isinstance(node, ast.Raise) for node in ast.walk(gate))
    text = " ".join(c.value for c in ast.walk(gate)
                    if isinstance(c, ast.Constant)
                    and isinstance(c.value, str))
    assert "--enable-expert-parallel" in text
    assert requires["vllm_flag"] in text


def test_admission_laws_name_real_enforcement_sites():
    """Each ``expert_admission`` value corresponds to code, not prose."""
    _, moe_ep = _source("gridbook/moe_ep.py")
    fns = {node.name for node in ast.walk(moe_ep)
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"local_expert_gather_index", "remap_local_expert_ids",
            "gather_expert_major"} <= fns

    # placement: monotone_bijection — the gather-index builder refuses both a
    # non-bijection and a non-monotone map.
    gather = _function(moe_ep, "local_expert_gather_index")
    gather_text = " ".join(c.value for c in ast.walk(gather)
                           if isinstance(c, ast.Constant)
                           and isinstance(c.value, str))
    assert "bijection" in gather_text and "monotone" in gather_text
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(gather)) >= 3

    # remote_pair_handling: zero_weight_alias — the remap zeroes rather than
    # dropping, which is what keeps every shape static under graph capture.
    remap = _function(moe_ep, "remap_local_expert_ids")
    calls = {node.func.attr for node in ast.walk(remap)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert "where" in calls and "zeros_like" in calls
    assert "nonzero" not in calls, \
        "a data-dependent compaction would not be capturable"

    # checkpoint_leading_dim: global_expert_count — BOTH loaders funnel
    # through the one gather rule.
    for rel in ("gridbook/moe.py", "gridbook/moe_toplevel_loader.py"):
        text, tree = _source(rel)
        assert "gather_expert_major" in text, \
            f"{rel} does not gather expert-major checkpoint tensors"


def test_schema_and_contract_version_move_together():
    contract = load_runtime_contract()
    assert RUNTIME_CONTRACT_SCHEMA == "gridbook.runtime-contract.v7"
    assert contract["schema"] == RUNTIME_CONTRACT_SCHEMA
    assert contract["contract_version"] == 7


# --- 2. Closed-world reading: absence means REFUSED ---------------------------


def _permitted(table: dict, unit: str) -> bool:
    """The documented consumer lookup for the expert-parallel axis.

    Mirrors docs/PLUGIN.md: find exactly one row named ``unit``; no row is a
    refusal, a ``cb_moe_expert_stack_refused`` row is a refusal above its cap,
    and a ``cb_moe_expert_stack`` row admits subject to the table's
    ``requires`` topology, which the runtime re-checks against the live worker.
    """
    rows = [row for row in table["units"] if row["unit"] == unit]
    if len(rows) != 1:
        return False
    row = rows[0]
    if row["kind"] == "cb_moe_expert_stack_refused":
        return False
    return row["kind"] == "cb_moe_expert_stack"


def test_absence_is_a_refusal_not_a_default():
    table = _ep_table(load_runtime_contract())
    assert _permitted(table, "FP8_CB_K")
    assert _permitted(table, "NVFP4_CB_K")
    assert not _permitted(table, "cb_moe_per_expert_format_groups")
    # Units that exist on the OTHER axis are not thereby expert-parallel.
    for unit in ("fp8_e4m3_ue8m0_block128", "mxfp4_e2m1_ue8m0_g32",
                 "mxfp8_e4m3_e8m0_g32", "NVFP4_CB_K_but_typoed"):
        assert not _permitted(table, unit)


# --- 3. The validator refuses drift ------------------------------------------


def _mutated(**edits):
    contract = copy.deepcopy(load_runtime_contract())
    ep = contract["expert_parallel"]
    for key, value in edits.items():
        if value is None:
            ep.pop(key, None)
        else:
            ep[key] = value
    return contract


def test_validator_refuses_an_open_world_reading():
    with pytest.raises(RuntimeContractError, match="closed_world"):
        validate_runtime_contract(_mutated(semantics="open_world"))


def test_validator_refuses_a_wrong_axis():
    with pytest.raises(RuntimeContractError, match="axis"):
        validate_runtime_contract(_mutated(axis="tp_world_size"))


def test_validator_refuses_a_relaxed_topology_requirement():
    for field, value in (
            ("moe_tensor_parallel_size", 2),
            ("all2all_kernels", True),
            ("expert_load_balancing", True),
            ("skip_final_all_reduce", True),
            ("vllm_flag", "--enable-ep"),
    ):
        contract = copy.deepcopy(load_runtime_contract())
        contract["expert_parallel"]["requires"][field] = value
        with pytest.raises(RuntimeContractError, match=field):
            validate_runtime_contract(contract)


def test_validator_refuses_a_bool_smuggled_in_as_an_int():
    """``True == 1`` in Python; the contract must not accept it as the cap."""
    contract = copy.deepcopy(load_runtime_contract())
    contract["expert_parallel"]["requires"]["moe_tensor_parallel_size"] = True
    with pytest.raises(RuntimeContractError,
                       match="moe_tensor_parallel_size"):
        validate_runtime_contract(contract)


def test_validator_refuses_an_unenforced_admission_law():
    contract = copy.deepcopy(load_runtime_contract())
    for row in contract["expert_parallel"]["units"]:
        if row["unit"] == "NVFP4_CB_K":
            row["expert_admission"]["remote_pair_handling"] = "compaction"
    with pytest.raises(RuntimeContractError, match="remote_pair_handling"):
        validate_runtime_contract(contract)


def test_validator_refuses_a_missing_cb_family_row():
    contract = copy.deepcopy(load_runtime_contract())
    contract["expert_parallel"]["units"] = [
        row for row in contract["expert_parallel"]["units"]
        if row["unit"] != "FP8_CB_K"
    ]
    with pytest.raises(RuntimeContractError, match="FP8_CB_K"):
        validate_runtime_contract(contract)


def test_validator_refuses_promoting_the_refused_mixed_unit():
    contract = copy.deepcopy(load_runtime_contract())
    for row in contract["expert_parallel"]["units"]:
        if row["unit"] == "cb_moe_per_expert_format_groups":
            row["max_world_size"] = 8
    with pytest.raises(RuntimeContractError, match="max_world_size"):
        validate_runtime_contract(contract)


def test_validator_refuses_an_invented_unit():
    contract = copy.deepcopy(load_runtime_contract())
    contract["expert_parallel"]["units"].append({
        "unit": "INT4_CB_K",
        "kind": "cb_moe_expert_stack",
        "expert_admission": dict(
            contract["expert_parallel"]["units"][0]["expert_admission"]),
    })
    with pytest.raises(RuntimeContractError, match="INT4_CB_K"):
        validate_runtime_contract(contract)


def test_validator_refuses_a_missing_table():
    contract = copy.deepcopy(load_runtime_contract())
    del contract["expert_parallel"]
    with pytest.raises(RuntimeContractError):
        validate_runtime_contract(contract)
