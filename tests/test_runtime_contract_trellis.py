"""The trellis half of the runtime contract, and what it may not be made to say.

Contract v12 is the first to publish ``device_qualified`` cells.  They come
from one physical receipt (2026-08-29, GB10, ``vllm/vllm-openai:qwen38-flash-
next``): four combinations of {E4M3, E2M1} x {resident, streamed}, four dense
Linears each, every code plane and ``scale_b`` operand byte-exact against the
wire, dispatched to ``TrellisE4M3LinearMethod`` / ``TrellisE2M1LinearMethod``.

That receipt covers strictly less than the lanes could serve, so the tests here
are arranged in two layers, because one alone is not enough:

* the packaged VALIDATOR refuses cells that are structurally wrong -- a rate
  off the candidate ladder, an activation contract that is not the lane's, a
  prose key, a family carrying CB fields;
* a LAWS TABLE below pins the four cells field for field, because no generic
  rule can know which rate a receipt covered.  ``rungs_q256: [384]`` is a
  perfectly well-formed cell and a false claim, and only an exact comparison
  against the receipt catches it.

The laws table is deliberately the same shape as
``gridbook.sm120_preflight._SM120_CELL_LAWS``, which pins the compile-only CB
cells for the same reason.
"""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

import pytest

from gridbook.runtime_contract import (
    RuntimeContractError,
    load_runtime_contract,
    validate_runtime_contract,
)
from gridbook.trellis import FAMILIES, RUNG_POLICIES


#: The four cells the 2026-08-29 GB10 receipt covers, field for field.
#: Widening ANY value here without a new receipt is the failure this pins.
_TRELLIS_CELL_LAWS: dict[str, dict[str, object]] = {
    "trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8": {
        "platform": "sm_121",
        "family": "TCQ_E4M3_R256",
        "structure": "dense",
        "regime": "decode",
        "rungs_q256": [1152],
        "activation_contract": "fp8_per_token_dynamic",
        "route_status": "backed_with_serve_flag",
        "qualification": "device_qualified",
        "requires_serve_flags": [
            "GRIDBOOK_TRELLIS_E4M3=1",
            "GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed",
        ],
        "predicates": [],
    },
    "trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8": {
        "platform": "sm_121",
        "family": "TCQ_E4M3_R256",
        "structure": "dense",
        "regime": "batch",
        "rungs_q256": [1152],
        "activation_contract": "fp8_per_token_dynamic",
        "route_status": "backed_with_serve_flag",
        "qualification": "device_qualified",
        "requires_serve_flags": [
            "GRIDBOOK_TRELLIS_E4M3=1",
            "GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed",
        ],
        "predicates": [],
    },
    "trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4": {
        "platform": "sm_121",
        "family": "TCQ_E2M1_R256",
        "structure": "dense",
        "regime": "decode",
        "rungs_q256": [512],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "route_status": "backed_with_serve_flag",
        "qualification": "device_qualified",
        "requires_serve_flags": [
            "GRIDBOOK_TRELLIS_E2M1=1",
            "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed",
        ],
        "predicates": [],
    },
    "trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4": {
        "platform": "sm_121",
        "family": "TCQ_E2M1_R256",
        "structure": "dense",
        "regime": "batch",
        "rungs_q256": [512],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "route_status": "backed_with_serve_flag",
        "qualification": "device_qualified",
        "requires_serve_flags": [
            "GRIDBOOK_TRELLIS_E2M1=1",
            "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed",
        ],
        "predicates": [],
    },
}


def _cells() -> dict[str, dict]:
    return {cell["id"]: cell
            for cell in load_runtime_contract()["lane_eligibility"]["cells"]}


def _formats() -> dict[str, dict]:
    return {row["family"]: row for row in load_runtime_contract()["formats"]}


def _repo_root() -> Path:
    """The source checkout whose dispatch site this file reads.

    In-tree that is this file's grandparent. The installed-wheel release gate
    stages ``tests/`` outside the checkout so the local package cannot shadow
    the wheel, and exports ``GRIDBOOK_SOURCE_ROOT`` as a data-file locator
    (never on ``PYTHONPATH``); ``GITHUB_WORKSPACE`` is the CI spelling. Same
    resolution order as ``test_runtime_contract_tp.py``.
    """
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        if (root / "gridbook" / "config.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "no Gridbook source checkout: run in-tree or set GRIDBOOK_SOURCE_ROOT")


def _config_source() -> tuple[str, ast.Module]:
    path = _repo_root() / "gridbook" / "config.py"
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


# ---------------------------------------------------------------------------
# What the contract publishes, checked against the package's own sources
# ---------------------------------------------------------------------------
def test_trellis_format_rows_restate_the_reader_domain():
    """The JSON ladder is ``RUNG_POLICIES``, not a second opinion of it.

    ``runtime_contract.py`` is standard-library-only and cannot import
    ``trellis.py``'s policies at validation time, so it restates them.  This is
    the assertion that keeps a restatement from becoming a drift.
    """
    rows = _formats()
    assert set(FAMILIES) <= set(rows)
    for family in FAMILIES:
        row = rows[family]
        policy = RUNG_POLICIES[family]
        assert row["kind"] == "tcq_trellis"
        assert row["name_pattern"] == f"{family.removesuffix('_R256')}_R{{k}}"
        assert tuple(row["candidate_rungs_q256"]) == policy.candidate_q256
        assert row["reader_rate_range_q256"] == [
            policy.research_floor_q256, policy.research_ceiling_q256]
        assert row["native_terminal_q256"] == policy.native_terminal_q256
        assert row["residency_modes"] == ["resident", "streamed"]
        # A trellis row must not borrow the CB vocabulary: its ladder is a
        # rate, not a codebook index, and publishing a grid/mode/n_sub would
        # describe a product codebook that does not exist here.
        for cb_only in ("grid", "mode", "n_sub", "rungs", "producer_rungs",
                        "layout_versions", "moe_layout_versions"):
            assert cb_only not in row


def test_trellis_cells_equal_the_device_receipt():
    """The four cells are pinned field for field against what actually ran."""
    cells = _cells()
    sm121 = {cid: cell for cid, cell in cells.items()
             if cell["platform"] == "sm_121"}
    assert set(sm121) == set(_TRELLIS_CELL_LAWS), (
        "the sm_121 cell set must equal the receipt's; a new cell needs a new "
        "receipt")
    for cell_id, law in _TRELLIS_CELL_LAWS.items():
        actual = {field: sm121[cell_id][field] for field in law}
        assert actual == law, f"{cell_id} drifted from the receipt"


def test_only_the_receipted_rungs_are_qualified():
    """Four of the five E2M1 candidate rates carry NO cell, on purpose.

    The receipt served ``body_rate_q256=512`` and nothing else on that ladder.
    An unserved rate is unattested by absence -- not ``compile_only``, which
    would still be a claim about a route nobody compiled a receipt for.
    """
    ladder = set(_formats()["TCQ_E2M1_R256"]["candidate_rungs_q256"])
    assert ladder == {384, 512, 640, 768, 896}
    qualified = {rate for cell in _cells().values()
                 if cell.get("family") == "TCQ_E2M1_R256"
                 for rate in cell["rungs_q256"]}
    assert qualified == {512}
    assert ladder - qualified == {384, 640, 768, 896}

    e4m3 = {rate for cell in _cells().values()
            if cell.get("family") == "TCQ_E4M3_R256"
            for rate in cell["rungs_q256"]}
    assert e4m3 == {1152}


def test_no_routed_moe_trellis_cell_and_the_code_agrees():
    """Closed-world absence, backed by the one dispatch site that exists.

    ``_build_trellis_method`` is called from inside ``get_quant_method``'s
    ``isinstance(layer, LinearBase)`` arm and nowhere else, so a FusedMoE
    expert stack cannot reach a trellis lane at all.  The missing
    ``routed_moe`` cell is therefore a fact about the code, not an omission.
    """
    assert not [cell for cell in _cells().values()
                if cell.get("family") in FAMILIES
                and cell["structure"] != "dense"]

    _, tree = _config_source()
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_build_trellis_method"]
    assert len(calls) == 1, "more than one trellis dispatch site"
    guards = [node for node in ast.walk(tree)
              if isinstance(node, ast.If)
              and isinstance(node.test, ast.Call)
              and getattr(node.test.func, "id", "") == "isinstance"
              and getattr(node.test.args[1], "id", "") == "LinearBase"]
    assert len(guards) == 1
    guard = guards[0]
    assert guard.lineno < calls[0].lineno <= guard.end_lineno, (
        "the trellis dispatch site left the LinearBase arm; a routed MoE "
        "stack could now reach it and the dense-only cells would overclaim")


def test_no_cell_can_carry_a_quality_or_performance_claim():
    """The table says which route EXECUTES. It has nowhere to say it is good.

    The trellis receipt ran on a synthetic checkpoint with RANDOM weights, so
    it could not have measured quality even if a field existed.  This asserts
    the field does not exist -- on any cell, of either kind.
    """
    forbidden = {
        "detail", "rationale", "note", "comment", "quality", "accuracy",
        "perplexity", "kl", "speedup", "throughput", "benchmark", "measured",
    }
    for cell in _cells().values():
        assert not (forbidden & set(cell)), f"{cell['id']} carries prose"


def test_serve_flags_and_activation_contracts_are_the_lanes_own_constants():
    """The published strings come from the lane modules, not from prose.

    This is what makes ``activation_contract`` an attestation: the value in the
    JSON is compared against the constant the lane's ``apply()`` stamps on the
    layer, so a lane that changed its A-side route without changing the
    contract fails here rather than shipping a stale claim.
    """
    pytest.importorskip("torch")
    from gridbook import trellis_e2m1_lane as e2m1
    from gridbook import trellis_e4m3_lane as e4m3

    cells = _cells()
    by_family = {
        "TCQ_E4M3_R256": (
            e4m3.ACTIVATION_CONTRACT,
            e4m3.TRELLIS_E4M3_FLAG,
            e4m3.TRELLIS_E4M3_MODE_ENV,
            (e4m3.MODE_RESIDENT, e4m3.MODE_STREAMED),
        ),
        "TCQ_E2M1_R256": (
            e2m1.ACTIVATION_CONTRACT,
            e2m1.TRELLIS_E2M1_FLAG,
            e2m1.TRELLIS_E2M1_MODE_ENV,
            (e2m1.MODE_RESIDENT, e2m1.MODE_STREAMED),
        ),
    }
    seen = 0
    for cell in cells.values():
        law = by_family.get(cell.get("family"))
        if law is None:
            continue
        contract, flag, mode_env, modes = law
        assert cell["activation_contract"] == contract
        assert cell["requires_serve_flags"] == [
            f"{flag}=1", f"{mode_env}={'|'.join(modes)}"]
        seen += 1
    assert seen == len(_TRELLIS_CELL_LAWS)

    # Both lanes default OFF, which is why no trellis cell may say "backed".
    assert not e4m3.trellis_e4m3_enabled()
    assert not e2m1.trellis_e2m1_enabled()


def test_the_residency_modes_the_contract_names_are_the_lanes_modes():
    pytest.importorskip("torch")
    from gridbook import trellis_e2m1_lane as e2m1
    from gridbook import trellis_e4m3_lane as e4m3

    rows = _formats()
    assert rows["TCQ_E4M3_R256"]["residency_modes"] == [
        e4m3.MODE_RESIDENT, e4m3.MODE_STREAMED]
    assert rows["TCQ_E2M1_R256"]["residency_modes"] == [
        e2m1.MODE_RESIDENT, e2m1.MODE_STREAMED]


# ---------------------------------------------------------------------------
# What the validator refuses: every scoping field, mutated one at a time
# ---------------------------------------------------------------------------
def _trellis_cell(contract, family="TCQ_E4M3_R256"):
    for cell in contract["lane_eligibility"]["cells"]:
        if cell.get("family") == family:
            return cell
    raise AssertionError(f"no {family} cell")


def _rung_off_the_ladder(contract):
    _trellis_cell(contract)["rungs_q256"] = [999]


def _empty_rung_set(contract):
    _trellis_cell(contract)["rungs_q256"] = []


def _rungs_in_cb_spelling(contract):
    cell = _trellis_cell(contract)
    cell["rungs"] = cell.pop("rungs_q256")


def _swapped_activation_contract(contract):
    _trellis_cell(contract)["activation_contract"] = "e2m1_group16_ue4m3_static"


def _invented_activation_contract(contract):
    _trellis_cell(contract)["activation_contract"] = "bf16_preserved"


def _plain_backed_without_flags(contract):
    cell = _trellis_cell(contract)
    cell["route_status"] = "backed"
    cell["requires_serve_flags"] = []


def _dropped_one_serve_flag(contract):
    cell = _trellis_cell(contract)
    cell["requires_serve_flags"] = cell["requires_serve_flags"][:1]


def _prose_on_a_trellis_cell(contract):
    _trellis_cell(contract)["detail"] = "serves great"


def _unknown_platform(contract):
    _trellis_cell(contract)["platform"] = "sm_121a"


def _trellis_family_with_cb_keys(contract):
    for row in contract["formats"]:
        if row["family"] == "TCQ_E4M3_R256":
            row["grid"] = "fp8"
            row["mode"] = "product"


def _unknown_format_kind(contract):
    contract["formats"][0]["kind"] = "codebook"


def _widened_candidate_ladder(contract):
    for row in contract["formats"]:
        if row["family"] == "TCQ_E2M1_R256":
            row["candidate_rungs_q256"] = [384, 512, 640, 768, 896, 1016]


def _trellis_reader_band_drift(contract):
    for row in contract["formats"]:
        if row["family"] == "TCQ_E2M1_R256":
            row["reader_rate_range_q256"] = [256, 2040]


def _missing_trellis_tp_row(contract):
    contract["tensor_parallel"]["units"] = [
        row for row in contract["tensor_parallel"]["units"]
        if row["unit"] != "TCQ_E4M3_R256"]


def _trellis_tp_above_one_rank(contract):
    for row in contract["tensor_parallel"]["units"]:
        if row["unit"] == "TCQ_E4M3_R256":
            row["max_world_size"] = 2


def _trellis_family_dropped_from_formats(contract):
    contract["formats"] = [row for row in contract["formats"]
                           if row["family"] != "TCQ_E2M1_R256"]


def _stale_lane_eligibility_schema(contract):
    contract["lane_eligibility"]["schema"] = "gridbook.lane-eligibility.v2"


@pytest.mark.parametrize("mutate, message", [
    (_rung_off_the_ladder, "candidate_rungs_q256"),
    (_empty_rung_set, "rungs_q256"),
    (_rungs_in_cb_spelling, "rungs_q256"),
    (_swapped_activation_contract, "activation_contract"),
    (_invented_activation_contract, "activation_contract"),
    (_plain_backed_without_flags, "requires_serve_flags"),
    (_dropped_one_serve_flag, "requires_serve_flags"),
    (_prose_on_a_trellis_cell, "unknown field"),
    (_unknown_platform, "platform"),
    (_trellis_family_with_cb_keys, "unknown field"),
    (_unknown_format_kind, "kind"),
    (_widened_candidate_ladder, "candidate_rungs_q256"),
    (_trellis_reader_band_drift, "reader_rate_range_q256"),
    (_missing_trellis_tp_row, "trellis_format_family"),
    (_trellis_tp_above_one_rank, "max_world_size"),
    (_trellis_family_dropped_from_formats, "trellis format families"),
    (_stale_lane_eligibility_schema, "lane-eligibility.v3"),
])
def test_validator_refuses_an_overclaiming_trellis_contract(mutate, message):
    contract = copy.deepcopy(load_runtime_contract())
    mutate(contract)
    with pytest.raises(RuntimeContractError) as excinfo:
        validate_runtime_contract(contract)
    assert message in str(excinfo.value)


@pytest.mark.parametrize("field, value", [
    ("structure", "routed_moe"),
    ("regime", "decode"),
    ("qualification", "compile_only"),
    ("rungs_q256", [384]),
])
def test_the_laws_table_is_the_layer_generic_validation_cannot_be(field, value):
    """A well-formed cell can still be a false claim, and this is the catch.

    Each mutation below leaves the contract structurally VALID -- the
    validator has no way to know which structure, regime, rate or
    qualification a receipt covered. Only the pinned laws table does, which is
    why both layers exist.  ``rungs_q256: [384]`` is the sharpest case: it is
    on the candidate ladder, so nothing generic can refuse it, and it names a
    rate nobody ever served.
    """
    contract = copy.deepcopy(load_runtime_contract())
    cell = _trellis_cell(contract, "TCQ_E2M1_R256")
    if cell[field] == value:                       # pick the other cell
        cell = [c for c in contract["lane_eligibility"]["cells"]
                if c.get("family") == "TCQ_E2M1_R256" and c is not cell][0]
    cell[field] = value
    # Structurally fine: the generic validator has no receipt to compare to.
    if field == "regime":
        cell["id"] = cell["id"] + "_dup"
    validate_runtime_contract(contract)

    # The laws table is what refuses it.
    law = _TRELLIS_CELL_LAWS.get(cell["id"])
    if law is None:
        assert cell["id"] not in _TRELLIS_CELL_LAWS
        return
    actual = {name: cell[name] for name in law}
    assert actual != law, (
        f"mutating {field} must diverge from the pinned receipt, or the laws "
        "table is not load-bearing for that field")
