"""The tensor-parallel capability table in the packaged runtime contract.

Gridbook refuses a live tensor-parallel world size above 1 on every dispatch
path.  As of schema ``gridbook.runtime-contract.v5`` the packaged contract
publishes that fact as machine-readable per-unit rows, so a producer gate can
branch on fields instead of prose (principle: an attested claim, never an
asserted one).

What is pinned here, without importing anything beyond the standard library,
pytest, and the stdlib-only contract loader:

1. The packaged table matches what the enforcement sites actually refuse.
   Each row is checked against the SOURCE TEXT of its refusal site — the AST
   of ``gridbook/config.py``, ``gridbook/fp8_source_w8a16.py``, and
   ``gridbook/mxfp8_dense_lane.py`` — so a row nobody enforces cannot ship.
2. Absence means REFUSED.  The documented closed-world lookup never defaults
   to permitted, and the packaged validator rejects a table that drops a
   mandatory field, omits or invents a unit, or widens a claim past the
   enforced maximum.  A vacuous pass is unrepresentable in both directions:
   publishing and reading.

Run: ``python -m pytest tests/test_runtime_contract_tp.py -q``.
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


def _tp_table(contract: dict) -> dict:
    return contract["tensor_parallel"]


def _units_by_id(contract: dict) -> dict[str, dict]:
    return {row["unit"]: row for row in _tp_table(contract)["units"]}


# --- 1. The published table ---------------------------------------------------


def test_packaged_table_declares_exactly_the_enforced_capability():
    tp = _tp_table(_packaged_contract())
    assert tp["axis"] == "vllm_tensor_parallel_world_size"
    assert tp["semantics"] == "closed_world"
    # Every dispatch path refuses a live world size above 1 today; the root
    # cap states that whole-model fact.
    assert tp["max_world_size"] == 1

    units = _units_by_id(load_runtime_contract())
    assert set(units) == {
        "FP8_CB_K",
        "NVFP4_CB_K",
        "NVFP4_CB_S",
        "fp8_e4m3_ue8m0_block128",
        "mxfp4_e2m1_ue8m0_g32",
        "mxfp8_e4m3_e8m0_g32",
    }
    for family in ("FP8_CB_K", "NVFP4_CB_K", "NVFP4_CB_S"):
        assert units[family]["kind"] == "cb_format_family"
        assert units[family]["max_world_size"] == 1
        assert "arms" not in units[family]
    for fmt in ("fp8_e4m3_ue8m0_block128", "mxfp4_e2m1_ue8m0_g32",
                "mxfp8_e4m3_e8m0_g32"):
        assert units[fmt]["kind"] == "source_passthrough_format"
        assert units[fmt]["max_world_size"] == 1
    # The two Gridbook-owned lanes branch on an execution arm with its own
    # refusal site; the vLLM-native MXFP4 route does not.
    assert "arms" not in units["mxfp4_e2m1_ue8m0_g32"]
    for fmt in ("fp8_e4m3_ue8m0_block128", "mxfp8_e4m3_e8m0_g32"):
        arms = {arm["arm"]: arm for arm in units[fmt]["arms"]}
        assert set(arms) == {"dense", "bmm"}
        assert all(arm["max_world_size"] == 1 for arm in arms.values())
    # The one pinned execution geometry: FP8-source W8A16 grouped BMM.
    bmm = {arm["arm"]: arm
           for arm in units["fp8_e4m3_ue8m0_block128"]["arms"]}["bmm"]
    assert bmm["requires_geometry"] == {
        "bmm_groups": 8,
        "rows_per_group": 1024,
        "k": 4096,
    }
    dense = {arm["arm"]: arm
             for arm in units["fp8_e4m3_ue8m0_block128"]["arms"]}["dense"]
    assert "requires_geometry" not in dense


def test_schema_and_contract_version_move_together():
    contract = load_runtime_contract()
    assert RUNTIME_CONTRACT_SCHEMA == "gridbook.runtime-contract.v5"
    assert contract["schema"] == RUNTIME_CONTRACT_SCHEMA
    assert contract["contract_version"] == 5


# --- 2. Closed-world reading: absence means REFUSED ---------------------------


def _permitted(table: dict, unit: str, world_size: int, *, arm=None,
               geometry: dict | None = None) -> bool:
    """The documented consumer lookup, verbatim from docs/PLUGIN.md.

    Exactly one row named *unit* must exist (and the exact *arm* row for an
    armed unit); the claim must cover *world_size* and any pinned geometry
    must match exactly.  Anything else is a refusal.  There is no default.
    """
    rows = [row for row in table["units"] if row["unit"] == unit]
    if len(rows) != 1:
        return False
    row = rows[0]
    if world_size > row["max_world_size"]:
        return False
    if "arms" not in row:
        return geometry is None
    if arm is None:
        return False
    matches = [a for a in row["arms"] if a["arm"] == arm]
    if len(matches) != 1:
        return False
    arm_row = matches[0]
    if world_size > arm_row["max_world_size"]:
        return False
    if "requires_geometry" in arm_row:
        return geometry == arm_row["requires_geometry"]
    return geometry is None


def test_closed_world_lookup_refuses_what_the_table_does_not_claim():
    table = _tp_table(load_runtime_contract())

    # Positive controls: the claims the runtime stands behind.
    assert _permitted(table, "NVFP4_CB_K", 1)
    assert _permitted(
        table, "fp8_e4m3_ue8m0_block128", 1, arm="bmm",
        geometry={"bmm_groups": 8, "rows_per_group": 1024, "k": 4096})

    # An unknown unit id has no row -> REFUSED.
    assert not _permitted(table, "NVFP4_CB_K99", 1)
    assert not _permitted(table, "fp8_e4m3_ue8m0_block128_extra", 1)
    # A world size no row covers -> REFUSED.
    assert not _permitted(table, "NVFP4_CB_K", 2)
    assert not _permitted(table, "NVFP4_CB_K", 8)
    # An armed unit consulted without naming the arm -> REFUSED.
    assert not _permitted(table, "mxfp8_e4m3_e8m0_g32", 1)
    # An arm name the unit does not publish -> REFUSED.
    assert not _permitted(table, "mxfp8_e4m3_e8m0_g32", 1, arm="rowwise")


def test_removed_unit_entry_is_a_refusal_not_a_default():
    contract = load_runtime_contract()
    mutated = copy.deepcopy(contract)
    mutated["tensor_parallel"]["units"] = [
        row for row in mutated["tensor_parallel"]["units"]
        if row["unit"] != "FP8_CB_K"
    ]
    # The consumer side: with the row gone there is nothing to permit with.
    assert not _permitted(mutated["tensor_parallel"], "FP8_CB_K", 1)
    # The publisher side: shipping such a table fails validation.
    with pytest.raises(RuntimeContractError, match="missing.*FP8_CB_K"):
        validate_runtime_contract(mutated)


# --- 3. The validator refuses incomplete or widened tables --------------------


def _drop_tensor_parallel(contract):
    del contract["tensor_parallel"]


def _drop_one_unit(contract):
    contract["tensor_parallel"]["units"] = [
        row for row in contract["tensor_parallel"]["units"]
        if row["unit"] != "mxfp4_e2m1_ue8m0_g32"
    ]


def _rename_unit(contract):
    units = contract["tensor_parallel"]["units"]
    units[0]["unit"] = "NOT_A_REAL_FAMILY"


def _relabel_kind(contract):
    units = _units_by_id(contract)
    units["mxfp4_e2m1_ue8m0_g32"]["kind"] = "cb_format_family"


def _drop_root_cap(contract):
    del contract["tensor_parallel"]["max_world_size"]


def _widen_root_cap(contract):
    contract["tensor_parallel"]["max_world_size"] = 2


def _widen_unit_cap(contract):
    _units_by_id(contract)["FP8_CB_K"]["max_world_size"] = 2


def _drop_unit_cap(contract):
    del _units_by_id(contract)["FP8_CB_K"]["max_world_size"]


def _strip_arms_from_pinned_lane(contract):
    row = _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]
    del row["arms"]


def _add_arms_to_flat_unit(contract):
    _units_by_id(contract)["mxfp4_e2m1_ue8m0_g32"]["arms"] = [
        {"arm": "dense", "max_world_size": 1},
    ]


def _drop_bmm_arm_claim(contract):
    row = _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]
    row["arms"] = [arm for arm in row["arms"] if arm["arm"] != "bmm"]


def _unknown_arm_name(contract):
    row = _units_by_id(contract)["mxfp8_e4m3_e8m0_g32"]
    row["arms"][1]["arm"] = "tensor"


def _corrupt_geometry_pin(contract):
    row = _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]
    bmm = next(arm for arm in row["arms"] if arm["arm"] == "bmm")
    bmm["requires_geometry"]["rows_per_group"] = 2048


def _drop_geometry_pin(contract):
    row = _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]
    bmm = next(arm for arm in row["arms"] if arm["arm"] == "bmm")
    del bmm["requires_geometry"]


def _invent_geometry_pin(contract):
    row = _units_by_id(contract)["mxfp8_e4m3_e8m0_g32"]
    row["arms"][1]["requires_geometry"] = {
        "bmm_groups": 8, "rows_per_group": 1024, "k": 4096,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_tensor_parallel, "tensor_parallel"),
        (_drop_one_unit, "missing.*mxfp4_e2m1_ue8m0_g32"),
        (_rename_unit, "unknown.*NOT_A_REAL_FAMILY"),
        (_relabel_kind, "unknown \\['mxfp4"),
        (_drop_root_cap, "max_world_size"),
        (_widen_root_cap, "no dispatch path"),
        (_widen_unit_cap, "no enforcement site"),
        (_drop_unit_cap, "max_world_size"),
        (_strip_arms_from_pinned_lane, "per-arm TP refusal sites"),
        (_add_arms_to_flat_unit, "one flat TP refusal site"),
        (_drop_bmm_arm_claim, "missing arm claim"),
        (_unknown_arm_name, "must be one of"),
        (_corrupt_geometry_pin, "must equal"),
        (_drop_geometry_pin, "must equal"),
        (_invent_geometry_pin, "not pinned"),
    ],
)
def test_validator_rejects_tables_that_overclaim_or_underclaim(
        mutate, message):
    contract = copy.deepcopy(load_runtime_contract())
    mutate(contract)
    with pytest.raises(RuntimeContractError, match=message):
        validate_runtime_contract(contract)


# --- 4. Rows are derived from the enforcement sites' source -------------------


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name!r}")


def _source(path: str) -> tuple[str, ast.Module]:
    full = _REPO_ROOT / path
    text = full.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(full))


def test_general_gate_refuses_above_one_before_dispatch():
    """Every dispatched unit passes through the config-level gate first."""
    source, tree = _source("gridbook/config.py")

    helper = _function(tree, "_require_supported_tensor_parallel")
    comparisons = [node for node in ast.walk(helper) if
                   isinstance(node, ast.Compare)]
    assert any(isinstance(cmp.ops[0], ast.NotEq)
               and isinstance(cmp.comparators[0], ast.Constant)
               and cmp.comparators[0].value == 1
               for cmp in comparisons), "gate must refuse world sizes != 1"
    assert any(isinstance(node, ast.Raise) for node in ast.walk(helper)), \
        "the out-of-policy branch must raise"
    assert "supports tensor-parallel size 1 only" in source

    get_quant_method = _function(tree, "get_quant_method")
    first = get_quant_method.body[0]
    assert (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Call)
            and isinstance(first.value.func, ast.Attribute)
            and first.value.func.attr == "_require_supported_tensor_parallel"
            ), "the TP gate must fire before any dispatch decision"


def test_fp8_source_geometry_rows_match_the_lane_constants():
    """The geometry pin restates fp8_source_w8a16's own constants."""
    source, tree = _source("gridbook/fp8_source_w8a16.py")
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)):
            name = node.targets[0].id
            if name.startswith("_DSV4_"):
                constants[name] = node.value.value
    assert constants == {
        "_DSV4_BMM_GROUPS": 8,
        "_DSV4_BMM_ROWS": 1024,
        "_DSV4_BMM_K": 4096,
        "_DSV4_RELEASE_TP": 1,
    }

    row = _units_by_id(load_runtime_contract())["fp8_e4m3_ue8m0_block128"]
    arms = {arm["arm"]: arm for arm in row["arms"]}
    assert arms["bmm"]["requires_geometry"] == {
        "bmm_groups": constants["_DSV4_BMM_GROUPS"],
        "rows_per_group": constants["_DSV4_BMM_ROWS"],
        "k": constants["_DSV4_BMM_K"],
    }
    assert arms["dense"]["max_world_size"] == constants["_DSV4_RELEASE_TP"]
    # The refusal texts those rows attest to.
    assert "qualified only for grouped" in source
    assert "release-gated only" in source


def test_mxfp8_bmm_row_matches_the_lane_audit_gate():
    source, _tree = _source("gridbook/mxfp8_dense_lane.py")
    assert 'int(getattr(layer, "tp_size", 1)) != 1' in source
    assert '"MXFP8 BMM is audited only for TP=1"' in source

    row = _units_by_id(load_runtime_contract())["mxfp8_e4m3_e8m0_g32"]
    arms = {arm["arm"]: arm for arm in row["arms"]}
    assert arms["bmm"]["max_world_size"] == 1


def test_passthrough_rows_match_the_source_format_registry():
    """Every audited passthrough format id carries exactly one TP row.

    The ids are read out of ``FORMATS``' module-level ``SourceFormat(...)``
    constructions in ``gridbook/source_passthrough.py``, so adding a format
    there without a contract row fails here instead of shipping an
    unattested unit.
    """
    _text, tree = _source("gridbook/source_passthrough.py")
    registry_ids = set()
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "SourceFormat"):
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "id":
                registry_ids.add(keyword.value.value)
    assert registry_ids == {"mxfp4_e2m1_ue8m0_g32",
                            "fp8_e4m3_ue8m0_block128",
                            "mxfp8_e4m3_e8m0_g32"}

    units = _units_by_id(load_runtime_contract())
    published = {unit for unit, row in units.items()
                 if row["kind"] == "source_passthrough_format"}
    assert published == registry_ids
