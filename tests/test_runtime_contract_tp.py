"""The tensor-parallel capability table in the packaged runtime contract.

Since the shard-aware loading wave, dense CB Linears construct above one
tensor-parallel rank under structural shard-alignment gates, while every
other surface refuses by name at a numeric TP=1 ceiling.  As of schema
``gridbook.runtime-contract.v11`` the packaged contract publishes exactly
that split as machine-readable per-unit rows, so a producer gate can branch
on fields instead of prose (principle: an attested claim, never an asserted
one).

The same schema adds a THIRD claim shape, the composite: a
``mixed_fused_projection`` row for one vLLM merged module whose roles have
different Gridbook formats.  It is not a format and owns no law, so it
publishes the axis it admits and the fact that each role's legality is that
role's own row above — an "inherited" claim rather than a cap the composite
would be asserting on its roles' behalf.

An earlier schema added the second law-admitted surface: the FP8-source W8A16
lane's DENSE arm, which enforces its own whole-128 shard laws at weight
construction, while the same unit's grouped-BMM arm is admitted only at
shard degrees whose divided group count was measured, because
column-sharding a grouped plane divides the kernel's group count (a kernel
qualification, not a law).  Admission for an armed unit is therefore
published PER ARM.

What is pinned here, without importing anything beyond the standard library,
pytest, and the stdlib-only contract loader:

1. The packaged table matches what the enforcement sites actually enforce.
   Each row is checked against the SOURCE TEXT of its site — the AST of
   ``gridbook/config.py`` and ``gridbook/linear.py`` for the dense CB
   admission laws and the per-surface refusals, plus
   ``gridbook/fp8_source_w8a16.py``, ``gridbook/mxfp8_dense_lane.py``, and
   ``gridbook/source_passthrough.py`` for the numeric TP=1 lanes — so a row
   nobody enforces cannot ship.
2. Absence means REFUSED.  The documented closed-world lookup never defaults
   to permitted where the runtime refuses, and the packaged validator rejects
   a table that drops a mandatory field, omits or invents a unit, publishes a
   numeric cap for the capless dense CB surface, or widens a numeric claim
   past the enforced maximum.  A vacuous pass is unrepresentable in both
   directions: publishing and reading.

Run: ``python -m pytest tests/test_runtime_contract_tp.py -q``.
"""
from __future__ import annotations

import ast
import copy
import json
import os
from importlib.resources import files
from pathlib import Path

import pytest

from gridbook.runtime_contract import (
    RUNTIME_CONTRACT_SCHEMA,
    RuntimeContractError,
    load_runtime_contract,
    validate_runtime_contract,
)


def _repo_root(test_file: Path | None = None) -> Path:
    """The source checkout whose enforcement sites this file reads.

    In-tree that is this file's grandparent. The installed-wheel release gate
    stages ``tests/`` outside the checkout so the local package cannot shadow
    the wheel, and exports ``GRIDBOOK_SOURCE_ROOT`` as a data-file locator
    (never on ``PYTHONPATH``); ``GITHUB_WORKSPACE`` is the CI spelling.
    """
    location = Path(__file__) if test_file is None else Path(test_file)
    roots = [location.resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        if (root / "gridbook" / "__init__.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "no Gridbook source checkout: run in-tree or set GRIDBOOK_SOURCE_ROOT")


_REPO_ROOT = _repo_root()


@pytest.mark.parametrize(
    "variable", ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE")
)
def test_repo_root_honors_installed_wheel_source_locator(
    monkeypatch, tmp_path, variable,
):
    source_root = tmp_path / "checkout"
    package = source_root / "gridbook"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("GRIDBOOK_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)
    monkeypatch.setenv(variable, str(source_root))

    staged_test = tmp_path / "wheel-tests" / "tests" / "test_contract.py"
    assert _repo_root(staged_test) == source_root.resolve()


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
    # No whole-model cap exists to publish: dense CB dispatch paths no longer
    # refuse above 1, so a single scalar would claim more than any one number
    # enforces.  v6 removed the field; its presence is a validator error.
    assert "max_world_size" not in tp

    units = _units_by_id(load_runtime_contract())
    assert set(units) == {
        "FP8_CB_K",
        "NVFP4_CB_K",
        "fp8_e4m3_ue8m0_block128",
        "mxfp4_e2m1_ue8m0_g32",
        "mxfp8_e4m3_e8m0_g32",
        "mixed_fused_projection",
    }
    # Dense CB units admit TP>1 subject to structural shard laws, so they
    # publish those laws instead of a numeric cap.  A cap here would be a
    # number no enforcement site stands behind.
    expected_admission = {
        "FP8_CB_K": {"input_axis_group": 256, "output_axis_quantum": 16,
                     "merged_roles": "even_division"},
        "NVFP4_CB_K": {"input_axis_group": 256, "output_axis_quantum": 8,
                       "merged_roles": "even_division"},
    }
    for family, admission in expected_admission.items():
        assert units[family]["kind"] == "cb_format_family"
        assert units[family]["shard_admission"] == admission
        assert "max_world_size" not in units[family]
        assert "arms" not in units[family]
    for fmt in ("fp8_e4m3_ue8m0_block128", "mxfp4_e2m1_ue8m0_g32",
                "mxfp8_e4m3_e8m0_g32"):
        assert units[fmt]["kind"] == "source_passthrough_format"
        assert "shard_admission" not in units[fmt]
    # The two Gridbook-owned lanes branch on an execution arm with its own
    # refusal site; the vLLM-native MXFP4 route does not.  An armed unit
    # publishes admission per arm and carries NO unit-level number: one
    # scalar cannot cover a law-admitted arm and a capped arm at once.
    assert "arms" not in units["mxfp4_e2m1_ue8m0_g32"]
    assert units["mxfp4_e2m1_ue8m0_g32"]["max_world_size"] == 1
    for fmt in ("fp8_e4m3_ue8m0_block128", "mxfp8_e4m3_e8m0_g32"):
        arms = {arm["arm"]: arm for arm in units[fmt]["arms"]}
        assert set(arms) == {"dense", "bmm"}
    # MXFP8 has no sharded audit on either arm: both stay numerically capped,
    # and so the unit-level cap of 1 remains a number a site enforces.
    mxfp8_arms = {arm["arm"]: arm
                  for arm in units["mxfp8_e4m3_e8m0_g32"]["arms"]}
    assert units["mxfp8_e4m3_e8m0_g32"]["max_world_size"] == 1
    assert all(arm["max_world_size"] == 1 for arm in mxfp8_arms.values())
    # The FP8-source unit publishes NO unit-level number: admission differs
    # per arm, and one scalar could not carry both arms' rules.
    assert "max_world_size" not in units["fp8_e4m3_ue8m0_block128"]
    fp8_arms = {arm["arm"]: arm
                for arm in units["fp8_e4m3_ue8m0_block128"]["arms"]}
    # The one pinned execution geometry: FP8-source W8A16 grouped BMM.  The
    # geometry names the UNSHARDED plane; the degrees say which shards of it
    # were measured.  Both are needed -- alignment alone cannot admit a group
    # count nobody ran.
    bmm = fp8_arms["bmm"]
    assert "max_world_size" not in bmm
    assert bmm["requires_geometry"] == {
        "bmm_groups": 8,
        "rows_per_group": 1024,
        "k": 4096,
    }
    assert bmm["shard_admission"] == {
        "input_axis_group": 128,
        "output_axis_quantum": 128,
        "merged_roles": "per_role_group_multiple",
        "qualified_shard_degrees": [1, 2, 4],
    }
    # The dense arm publishes the same laws and NO degree list: alignment is
    # the whole rule there, exactly like the dense CB families.
    dense = fp8_arms["dense"]
    assert "max_world_size" not in dense
    assert "requires_geometry" not in dense
    assert dense["shard_admission"] == {
        "input_axis_group": 128,
        "output_axis_quantum": 128,
        "merged_roles": "per_role_group_multiple",
    }

    # The composite: a merged module whose roles have different formats. It
    # publishes no quantum of its own -- gridbook/mixed_linear.py builds each
    # role's carrier at that role's whole-tensor output size, so the rows
    # above are what decides -- and no cap, for the same reason the dense CB
    # rows carry none. The axis list is the one law it does own: a merged
    # projection is column-parallel, and a row-parallel split of it is
    # refused by the composer itself.
    mixed = units["mixed_fused_projection"]
    assert mixed["kind"] == "mixed_fused_projection"
    assert "max_world_size" not in mixed
    assert "arms" not in mixed
    assert mixed["shard_admission"] == {
        "axes": ["output"],
        "per_role_law": "inherited",
    }


def test_schema_and_contract_version_move_together():
    contract = load_runtime_contract()
    assert RUNTIME_CONTRACT_SCHEMA == "gridbook.runtime-contract.v11"
    assert contract["schema"] == RUNTIME_CONTRACT_SCHEMA
    assert contract["contract_version"] == 11


# --- 2. Closed-world reading: absence means REFUSED ---------------------------


def _permitted(table: dict, unit: str, world_size: int, *, arm=None,
               geometry: dict | None = None) -> bool:
    """The documented consumer lookup, verbatim from docs/PLUGIN.md.

    Exactly one row named *unit* must exist (and the exact *arm* row for an
    armed unit); a numeric claim must cover *world_size*, any pinned geometry
    must match exactly, and a row (or arm) that publishes ``shard_admission``
    publishes no numeric claim at all — admission above one rank is decided by
    its laws at the runtime's weight construction, which fail closed.
    Anything else is a refusal.  There is no default.
    """
    rows = [row for row in table["units"] if row["unit"] == unit]
    if len(rows) != 1:
        return False
    row = rows[0]
    if "shard_admission" in row:
        # Dense CB: no numeric cap exists to compare against; the table
        # defers to the construction-time alignment gates for any size.
        return geometry is None and "max_world_size" not in row
    if "max_world_size" in row and world_size > row["max_world_size"]:
        return False
    if "arms" not in row:
        return geometry is None
    # A unit with a law-admitted arm carries no unit-level number at all;
    # either way the arm decides.
    if arm is None:
        return False
    matches = [a for a in row["arms"] if a["arm"] == arm]
    if len(matches) != 1:
        return False
    arm_row = matches[0]
    if "shard_admission" in arm_row:
        if "max_world_size" in arm_row:
            return False
        degrees = arm_row["shard_admission"].get("qualified_shard_degrees")
        if degrees is not None and world_size not in degrees:
            return False
        if "requires_geometry" in arm_row:
            return geometry == arm_row["requires_geometry"]
        return geometry is None
    if world_size > arm_row["max_world_size"]:
        return False
    if "requires_geometry" in arm_row:
        return geometry == arm_row["requires_geometry"]
    return geometry is None


def test_closed_world_lookup_refuses_what_the_table_does_not_claim():
    table = _tp_table(load_runtime_contract())

    # Positive controls: the claims the runtime stands behind.
    assert _permitted(table, "NVFP4_CB_K", 1)
    # The lifted surface: dense CB above one rank is admitted by the table
    # (the runtime's construction-time shard laws decide the rest).
    assert _permitted(table, "NVFP4_CB_K", 2)
    assert _permitted(table, "FP8_CB_K", 4)
    assert _permitted(
        table, "fp8_e4m3_ue8m0_block128", 1, arm="bmm",
        geometry={"bmm_groups": 8, "rows_per_group": 1024, "k": 4096})
    # The second lifted surface: the FP8-source DENSE arm above one rank
    # (its whole-128 construction laws decide the rest).
    assert _permitted(table, "fp8_e4m3_ue8m0_block128", 2, arm="dense")
    assert _permitted(table, "fp8_e4m3_ue8m0_block128", 8, arm="dense")
    # The composite reads like the other law-admitted rows: the table defers,
    # and the per-role construction gates decide.
    assert _permitted(table, "mixed_fused_projection", 2)
    assert _permitted(table, "mixed_fused_projection", 8)

    # An unknown unit id has no row -> REFUSED.
    assert not _permitted(table, "NVFP4_CB_K99", 1)
    assert not _permitted(table, "fp8_e4m3_ue8m0_block128_extra", 1)
    # A REMOVED unit has no row either: the signed NVFP4_CB_S family left the
    # runtime (2026-08-23), and the closed-world reading refuses its artifacts
    # without any new machinery — absence IS the refusal.
    assert not _permitted(table, "NVFP4_CB_S", 1)
    assert not _permitted(table, "NVFP4_CB_S", 2)
    # A world size no numeric claim covers -> REFUSED.
    for capped in ("fp8_e4m3_ue8m0_block128", "mxfp4_e2m1_ue8m0_g32",
                   "mxfp8_e4m3_e8m0_g32"):
        assert not _permitted(table, capped, 2)
        assert not _permitted(table, capped, 8)
    for arm in ("dense", "bmm"):
        assert not _permitted(table, "mxfp8_e4m3_e8m0_g32", 2, arm=arm)
    # The FP8-source BMM arm names the UNSHARDED plane, so a consumer asks
    # with G=8 and the degree it wants; only measured degrees are admitted.
    dsv4_wo_a = {"bmm_groups": 8, "rows_per_group": 1024, "k": 4096}
    assert _permitted(
        table, "fp8_e4m3_ue8m0_block128", 2, arm="bmm", geometry=dsv4_wo_a)
    assert _permitted(
        table, "fp8_e4m3_ue8m0_block128", 4, arm="bmm", geometry=dsv4_wo_a)
    # A degree past the measured list -> REFUSED, however aligned it is.
    assert not _permitted(
        table, "fp8_e4m3_ue8m0_block128", 8, arm="bmm", geometry=dsv4_wo_a)
    # The per-rank group count is NOT what the row publishes; asking with it
    # is asking about a plane the table never claimed -> REFUSED.
    assert not _permitted(
        table, "fp8_e4m3_ue8m0_block128", 2, arm="bmm",
        geometry={"bmm_groups": 4, "rows_per_group": 1024, "k": 4096})
    # And the geometry still has to match: a different plane -> REFUSED.
    assert not _permitted(
        table, "fp8_e4m3_ue8m0_block128", 2, arm="bmm",
        geometry={"bmm_groups": 8, "rows_per_group": 2048, "k": 4096})
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


def _restore_removed_root_cap(contract):
    contract["tensor_parallel"]["max_world_size"] = 2


def _widen_passthrough_cap(contract):
    _units_by_id(contract)["mxfp4_e2m1_ue8m0_g32"]["max_world_size"] = 2


def _drop_passthrough_cap(contract):
    del _units_by_id(contract)["mxfp4_e2m1_ue8m0_g32"]["max_world_size"]


def _cap_a_cb_row(contract):
    # Dense CB publishes no numeric cap because no code enforces one; adding
    # one asserts a number nothing stands behind.
    _units_by_id(contract)["NVFP4_CB_K"]["max_world_size"] = 8


def _drop_cb_admission(contract):
    del _units_by_id(contract)["FP8_CB_K"]["shard_admission"]


def _corrupt_input_axis_group(contract):
    _units_by_id(contract)["FP8_CB_K"]["shard_admission"][
        "input_axis_group"] = 128


def _corrupt_output_axis_quantum(contract):
    _units_by_id(contract)["NVFP4_CB_K"]["shard_admission"][
        "output_axis_quantum"] = 4


def _relax_merged_roles(contract):
    _units_by_id(contract)["NVFP4_CB_K"]["shard_admission"][
        "merged_roles"] = "best_effort"


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


def _fp8_arm(contract, name):
    row = _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]
    return next(arm for arm in row["arms"] if arm["arm"] == name)


def _restore_unit_cap_on_an_armed_row(contract):
    # An armed unit's admission is per arm; a unit-level scalar could not
    # cover a law-admitted arm and a capped arm at once.
    _units_by_id(contract)["fp8_e4m3_ue8m0_block128"]["max_world_size"] = 1


def _cap_the_law_admitted_arm(contract):
    _fp8_arm(contract, "dense")["max_world_size"] = 8


def _drop_the_dense_shard_law(contract):
    del _fp8_arm(contract, "dense")["shard_admission"]


def _corrupt_the_dense_input_axis_law(contract):
    _fp8_arm(contract, "dense")["shard_admission"]["input_axis_group"] = 256


def _relax_the_dense_merged_role_law(contract):
    _fp8_arm(contract, "dense")["shard_admission"]["merged_roles"] = \
        "even_division"


def _law_on_an_unadmitted_arm(contract):
    # MXFP8 has no sharded audit on either arm, so neither may publish a law
    # in place of the number a site actually enforces.
    row = _units_by_id(contract)["mxfp8_e4m3_e8m0_g32"]
    row["arms"][0]["shard_admission"] = {
        "input_axis_group": 128,
        "output_axis_quantum": 128,
        "merged_roles": "per_role_group_multiple",
    }


def _cap_the_bmm_arm_again(contract):
    _fp8_arm(contract, "bmm")["max_world_size"] = 2


def _widen_the_bmm_degrees(contract):
    # TP=8 leaves one group per rank; the kernel was never run there.
    _fp8_arm(contract, "bmm")["shard_admission"][
        "qualified_shard_degrees"] = [1, 2, 4, 8]


def _drop_the_bmm_degree_list(contract):
    # Without it the arm would read as "any aligned degree", which is exactly
    # the claim the grouped kernel does not support.
    del _fp8_arm(contract, "bmm")["shard_admission"][
        "qualified_shard_degrees"]


def _cap_the_composite(contract):
    # A number here would claim a ceiling for every role at once; no site
    # enforces one, and the roles' own rows are where admission lives.
    _units_by_id(contract)["mixed_fused_projection"]["max_world_size"] = 2


def _drop_the_composite_row(contract):
    contract["tensor_parallel"]["units"] = [
        row for row in contract["tensor_parallel"]["units"]
        if row["unit"] != "mixed_fused_projection"
    ]


def _admit_the_composite_on_the_input_axis(contract):
    _units_by_id(contract)["mixed_fused_projection"]["shard_admission"][
        "axes"] = ["output", "input"]


def _invent_a_composite_law(contract):
    _units_by_id(contract)["mixed_fused_projection"]["shard_admission"][
        "per_role_law"] = "output_axis_quantum_128"


def _invent_geometry_pin(contract):
    row = _units_by_id(contract)["mxfp8_e4m3_e8m0_g32"]
    row["arms"][1]["requires_geometry"] = {
        "bmm_groups": 8, "rows_per_group": 1024, "k": 4096,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_tensor_parallel, "tensor_parallel"),
        (_drop_one_unit, "missing.*mxfp4"),
        (_rename_unit, "no CB format family.*NOT_A_REAL_FAMILY"),
        (_relabel_kind, "missing field.*shard_admission"),
        (_restore_removed_root_cap, "unknown field.*max_world_size"),
        (_widen_passthrough_cap, "no enforcement site"),
        (_drop_passthrough_cap, "max_world_size"),
        (_cap_a_cb_row, "unknown field.*max_world_size"),
        (_drop_cb_admission, "missing field.*shard_admission"),
        (_corrupt_input_axis_group, "must equal 256"),
        (_corrupt_output_axis_quantum, "must equal 8 for grid 'fp4'"),
        (_relax_merged_roles, "must be 'even_division'"),
        (_strip_arms_from_pinned_lane, "per-arm TP refusal sites"),
        (_add_arms_to_flat_unit, "one flat TP refusal site"),
        (_drop_bmm_arm_claim, "missing arm claim"),
        (_unknown_arm_name, "must be one of"),
        (_corrupt_geometry_pin, "must equal"),
        (_drop_geometry_pin, "must equal"),
        (_invent_geometry_pin, "not pinned"),
        (_restore_unit_cap_on_an_armed_row, "unknown field.*max_world_size"),
        (_cap_the_law_admitted_arm, "unknown field.*max_world_size"),
        (_drop_the_dense_shard_law, "missing field.*shard_admission"),
        (_corrupt_the_dense_input_axis_law, "must equal 128"),
        (_relax_the_dense_merged_role_law, "must equal"),
        (_law_on_an_unadmitted_arm, "unknown field.*shard_admission"),
        (_cap_the_bmm_arm_again, "unknown field.*max_world_size"),
        (_widen_the_bmm_degrees, "must equal"),
        (_drop_the_bmm_degree_list, "missing field"),
        (_cap_the_composite, "unknown field.*max_world_size"),
        (_drop_the_composite_row, "mixed_fused_projection"),
        (_admit_the_composite_on_the_input_axis, "must equal"),
        (_invent_a_composite_law, "must equal"),
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


def test_dense_cb_admission_rows_match_the_linear_gates():
    """The shard-admission laws restate linear.py's construction gates.

    Derives each published constant from the enforcement site instead of
    asserting it: ``codec.SUPERBLOCK`` for the input axis, the kernel
    row-alignment conditional (``8 if self.is_fp4 else 16``) for the output
    axis, and the merged-role even-division refusal in
    ``_rank_local_role_widths``.
    """
    _text, codec_tree = _source("gridbook/codec.py")
    superblock = None
    for node in codec_tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SUPERBLOCK"
                and isinstance(node.value, ast.Constant)):
            superblock = node.value.value
    assert superblock == 256, "codec.SUPERBLOCK must stay the input law"

    source, tree = _source("gridbook/linear.py")
    error_cls = next((node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "ShardGroupAlignmentError"), None)
    assert error_cls is not None, \
        "the structured shard refusal must exist"
    assert any(isinstance(base, ast.Name) and base.id == "ValueError"
               for base in error_cls.bases), \
        "callers may catch the structured refusal as ValueError"

    create_weights = _function(tree, "create_weights")
    assert any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == "_require_shard_group_alignment"
               for node in ast.walk(create_weights)), \
        "weight construction must run the shard-legality gate first"
    gate = _function(tree, "_require_shard_group_alignment")
    gate_source = ast.get_source_segment(source, gate) or ""
    assert "codec.SUPERBLOCK" in gate_source, \
        "the input-axis law must be the packed superblock"
    assert "8 if self.is_fp4 else 16" in gate_source, \
        "the output-axis law must be the native kernel row quantum"
    role_gate = _function(tree, "_rank_local_role_widths")
    assert any(isinstance(node, ast.Raise) for node in ast.walk(role_gate)), \
        "uneven merged-role division must refuse"

    units = _units_by_id(load_runtime_contract())
    quanta = {"fp4": 8, "fp8": 16}
    for family, grid in (("FP8_CB_K", "fp8"), ("NVFP4_CB_K", "fp4")):
        admission = units[family]["shard_admission"]
        assert admission["input_axis_group"] == superblock
        assert admission["output_axis_quantum"] == quanta[grid]
        assert admission["merged_roles"] == "even_division"


def test_config_dispatch_policy_matches_the_published_split():
    """config.py refuses every non-dense surface by name; dense CB defers.

    The v5 blanket pre-dispatch gate is gone: no single statement refuses
    before dispatch, which is why CB rows publish laws instead of a cap.
    Every remaining ``_require_tp1_serving`` site must name ITS OWN non-dense
    surface, and together they must cover exactly the surfaces that stay
    closed.
    """
    source, tree = _source("gridbook/config.py")

    # The blanket gate is gone outright.
    names = {node.name for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_require_supported_tensor_parallel" not in names
    assert "supports tensor-parallel size 1 only" not in source

    # Its replacement raises, names the surface, and states the split.
    # The message literals are line-wrapped, so derive the sentence from the
    # raise site's string constants rather than raw source text.
    helper = _function(tree, "_require_tp1_serving")
    assert any(isinstance(node, ast.Raise) for node in ast.walk(helper)), \
        "the out-of-policy branch must raise"
    helper_text = " ".join(
        c.value for c in ast.walk(helper)
        if isinstance(c, ast.Constant) and isinstance(c.value, str))
    assert "Dense CB Linears are the only supported tensor-parallel surface" \
        in helper_text

    # Dispatch does not open with a TP gate any more: policy lives per arm.
    get_quant_method = _function(tree, "get_quant_method")
    first = get_quant_method.body[0]
    assert not (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Call)
                and isinstance(first.value.func, ast.Attribute)
                and "tensor_parallel" in first.value.func.attr), \
        "no blanket TP gate may fire before dispatch"

    # Collect the static text every refusal site passes as its own name.
    # Surfaces may be f-strings or concatenated literals, so walk each call
    # subtree for string constants instead of reading one argument shape.
    refusals = [node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_require_tp1_serving"]
    fragments: list[str] = []
    for node in refusals:
        parts = [c.value for c in ast.walk(node)
                 if isinstance(c, ast.Constant)
                 and isinstance(c.value, str) and c.value.strip()]
        assert parts, f"refusal at line {node.lineno} must name its surface"
        fragments.extend(parts)
    joined = "\n".join(fragments)
    # Five, not six: the stacked whole-tensor CB MoE lane moved off the TP=1
    # helper onto the expert-parallel gate below, and mixed-format fused
    # projections moved off it entirely — that composite owns no law of its
    # own, so each role's existing shard gate refuses per role instead. Trellis
    # dense is a separate opaque-blob research lane: its wire has no splittable
    # axis, so it correctly keeps its own named TP=1 refusal. The MIXED
    # per-expert-format MoE site also stays.
    assert len(refusals) == 5
    for surface in (
            "source-passthrough unit format",
            "delegated stock compressed-tensors groups",
            "quantized embedding units",
            "Gridbook trellis dense lanes",
            "CB MoE expert stacks",
    ):
        assert surface in joined, \
            f"no refusal site names the {surface!r} surface"
    assert "mixed-format fused projections" not in joined, \
        "the mixed fused surface is admitted per role; no site may gate it"
    assert "dense CB" not in joined and "CB Linear" not in joined, \
        "no refusal site may name the admitted dense CB surface"
    # T6: the surviving MoE refusal must say what DOES serve above one rank,
    # or an operator reads it as "no multi-rank MoE at all".
    assert "--enable-expert-parallel" in joined, \
        "the mixed-MoE refusal must name the mode that does serve"

    # And the admitted surface really is dispatched: the fused, mixed-role,
    # and plain dense CB arms all construct PrismaQuantCBLinearMethod (the
    # mixed arm is guarded by its own named refusal above).
    cb_arms = [node for node in ast.walk(get_quant_method)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "PrismaQuantCBLinearMethod"]
    assert len(cb_arms) == 3


def _lane_constants(tree, prefix: str) -> dict:
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, (ast.Constant, ast.Tuple))):
            name = node.targets[0].id
            if name.startswith(prefix):
                constants[name] = ast.literal_eval(node.value)
    return constants


def test_fp8_source_geometry_rows_match_the_lane_constants():
    """The BMM geometry pin restates fp8_source_w8a16's own constants."""
    source, tree = _source("gridbook/fp8_source_w8a16.py")
    constants = _lane_constants(tree, "_DSV4_")
    assert constants == {
        "_DSV4_BMM_GROUPS": 8,
        "_DSV4_BMM_ROWS": 1024,
        "_DSV4_BMM_K": 4096,
        "_DSV4_BMM_QUALIFIED_SHARD_DEGREES": (1, 2, 4),
    }

    row = _units_by_id(load_runtime_contract())["fp8_e4m3_ue8m0_block128"]
    arms = {arm["arm"]: arm for arm in row["arms"]}
    assert arms["bmm"]["requires_geometry"] == {
        "bmm_groups": constants["_DSV4_BMM_GROUPS"],
        "rows_per_group": constants["_DSV4_BMM_ROWS"],
        "k": constants["_DSV4_BMM_K"],
    }
    # The BMM arm publishes the MEASURED shard degrees rather than a cap:
    # sharding a grouped plane divides the kernel's group count, so each
    # degree is its own qualification and the list is the enforcement site's.
    assert "max_world_size" not in arms["bmm"]
    assert arms["bmm"]["shard_admission"]["qualified_shard_degrees"] == \
        list(constants["_DSV4_BMM_QUALIFIED_SHARD_DEGREES"])
    assert "qualified only for grouped" in source

    # And the qualified table the refusal consults is built from exactly
    # those constants, so a row cannot outlive the geometry it names.
    qualified = _function(tree, "_qualified_bmm_geometries")
    named = {node.id for node in ast.walk(qualified)
             if isinstance(node, ast.Name)}
    assert set(constants) <= named, \
        "the qualified geometry table must be built from the pinned constants"


def test_fp8_source_dense_shard_laws_match_the_lane_gate():
    """The dense arm's published laws restate the lane's construction gate.

    The lane is the enforcement site, so the number is derived from its
    module constant rather than asserted here, and the structural facts the
    law depends on -- degrees taken from ``create_weights``' arguments, not
    from ``layer.tp_size``, and the refusal raised at construction -- are
    pinned against the source.
    """
    source, tree = _source("gridbook/fp8_source_w8a16.py")
    alignment = _lane_constants(tree, "_TP_SHARD_ALIGNMENT")
    assert alignment == {"_TP_SHARD_ALIGNMENT": 128}
    # The literal is tied to the source block size by an import-time guard,
    # so it cannot drift away from the format it describes.
    assert "if _TP_SHARD_ALIGNMENT != DS_BLOCK:" in source

    error_cls = next((node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "ShardAlignmentError"), None)
    assert error_cls is not None, "the structured shard refusal must exist"
    assert any(isinstance(base, ast.Name) and base.id == "ValueError"
               for base in error_cls.bases), \
        "callers may catch the structured refusal as ValueError"

    create_weights = _function(tree, "create_weights")
    called = {node.func.id for node in ast.walk(create_weights)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"_resolve_shard_plan", "_require_shard_alignment"} <= called, \
        "weight construction must resolve and gate the shard plan"
    # The degrees come from the ARGUMENTS.  vLLM stamps tp_size on replicated
    # layers too (DSv4 has 64 of them), so tp_size cannot decide this.
    plan = _function(tree, "_resolve_shard_plan")
    plan_source = ast.get_source_segment(source, plan) or ""
    assert "input_size_per_partition" in plan_source
    assert "output_partition_sizes" in plan_source
    assert "tp_size" not in plan_source

    gate = _function(tree, "_require_shard_alignment")
    gate_source = ast.get_source_segment(source, gate) or ""
    assert "_TP_SHARD_ALIGNMENT" in gate_source
    assert "row_degree" in gate_source and "col_degree" in gate_source

    arms = {arm["arm"]: arm for arm
            in _units_by_id(load_runtime_contract())[
                "fp8_e4m3_ue8m0_block128"]["arms"]}
    admission = arms["dense"]["shard_admission"]
    assert admission["input_axis_group"] == alignment["_TP_SHARD_ALIGNMENT"]
    assert admission["output_axis_quantum"] == alignment["_TP_SHARD_ALIGNMENT"]
    assert admission["merged_roles"] == "per_role_group_multiple"
    assert "max_world_size" not in arms["dense"]


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
