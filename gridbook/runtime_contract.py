"""Load and validate Gridbook's packaged producer/runtime contract.

This module is deliberately standard-library-only.  Exporters and compatibility
checks can inspect the installed Gridbook contract without importing torch,
vLLM, CUDA extensions, or any producing project.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping


RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v12"
_RESOURCE_NAME = "runtime_contract.json"

#: A ``formats`` row's kind decides which vocabulary the rest of the row --
#: and every lane cell naming its family -- is written in.  The two are not
#: variants of one thing: a CB row describes a product codebook over an fp4 or
#: fp8 grid indexed by an integer K, while a TCQ row describes a
#: self-describing trellis wire whose ladder is a body-bit rate per 256
#: weights.  Sharing one key set would have forced a trellis row to publish a
#: ``grid``/``mode``/``n_sub`` it does not have, so the kinds carry different
#: keys and each is validated against its own law.
_FORMAT_KINDS = ("cb_product", "tcq_trellis")

#: The CB reader domain, unchanged since v11: it keeps the historical reader
#: menu separate from the canonical producer one.  The broad low-rung
#: expansion developed after v0.9.0 never shipped and
#: is deliberately absent here: NVFP4 reads and produces K12..K24, while FP8
#: reads every integer K28..K48 and produces only K40/K44/K48.  Generic CUDA
#: bindings retain wider direct-kernel ranges for research, but low-level kernel
#: coverage is not artifact authority and therefore does not appear here.
_FORMAT_RUNGS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "NVFP4_CB_K": (tuple(range(12, 25)), tuple(range(12, 25))),
    "FP8_CB_K": (
        tuple(range(28, 49)),
        (40, 44, 48),
    ),
}
# Mapping order is serialization/validation plumbing, never a cross-family
# preference.  AQUA resolves overlapping FP8/NVFP4 choices from registered
# activation contracts; this runtime contract intentionally has no manual
# "prefer FP8" field.

#: The TCQ trellis reader domain, restating ``gridbook.trellis.RUNG_POLICIES``
#: field for field: the candidate ladder a producer may target, the inclusive
#: research rate range the reader accepts, and the native terminal rate.  All
#: rates are BODY BITS PER 256 WEIGHTS -- never a rounded decimal bpw, and
#: never a CB codebook K.  This module stays standard-library-only, so the
#: numbers are restated here rather than imported from ``trellis.py`` (which
#: pulls in torch); ``tests/test_runtime_contract_trellis.py`` asserts the two
#: agree, which is what makes this a restatement rather than a second opinion.
_TRELLIS_RUNG_LAW: dict[str, dict[str, Any]] = {
    "TCQ_E2M1_R256": {
        "candidate_rungs_q256": (384, 512, 640, 768, 896),
        "reader_rate_range_q256": (256, 1016),
        "native_terminal_q256": 1024,
    },
    "TCQ_E4M3_R256": {
        "candidate_rungs_q256": (1152,),
        "reader_rate_range_q256": (256, 2040),
        "native_terminal_q256": 2048,
    },
}

#: Both trellis lanes implement the same two residency modes, and the mode is
#: a FOOTPRINT choice rather than a numerics one -- the two are asserted
#: bit-identical by ``tests/test_trellis_*_lane.py`` -- so it is published
#: per family rather than per lane cell.
_TRELLIS_RESIDENCY_MODES = ("resident", "streamed")

#: What each trellis lane executes on the A side, keyed by family.  These are
#: the lanes' own ``ACTIVATION_CONTRACT`` constants
#: (``trellis_e4m3_lane.py``, ``trellis_e2m1_lane.py``); the trellis lane cells
#: carry the value so a producer can compare a PRICED activation contract
#: against the SERVED one instead of assuming they match.  Neither lane has a
#: bf16 A-side route: ``_scaled_mm`` on this hardware refuses a mixed
#: bf16 x fp4 / bf16 x fp8 pair outright, so W4A4 and W8A8 are the only shapes
#: these lanes can take.  A CB cell publishes no such field because a CB lane's
#: A side is mode-selected at serve time and one value would be wrong.
_TRELLIS_ACTIVATION_CONTRACTS: dict[str, str] = {
    "TCQ_E2M1_R256": "e2m1_group16_ue4m3_static",
    "TCQ_E4M3_R256": "fp8_per_token_dynamic",
}

#: Both trellis lanes are OPT-IN and refuse construction with the flag unset
#: (``build_trellis_*_method``), and neither mode env var has a default -- an
#: unset mode is an error, because defaulting would pick the artifact's memory
#: footprint for the operator silently.  A trellis cell therefore cannot be
#: ``backed``: the route does not execute on a default serve, so the honest
#: status is ``backed_with_serve_flag`` and these are the flags it names.
_TRELLIS_SERVE_FLAGS: dict[str, tuple[str, ...]] = {
    "TCQ_E2M1_R256": (
        "GRIDBOOK_TRELLIS_E2M1=1",
        "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed",
    ),
    "TCQ_E4M3_R256": (
        "GRIDBOOK_TRELLIS_E4M3=1",
        "GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed",
    ),
}

#: v3, not v2: a trellis cell carries a rate ladder in different units and an
#: executed activation contract that a CB cell has no truthful value for, so
#: the cell vocabulary is no longer one key set.  A consumer written against
#: v2 must refuse this table whole rather than read the keys it recognises --
#: it would silently mistake ``rungs_q256`` for absent rungs.
_LANE_ELIGIBILITY_SCHEMA = "gridbook.lane-eligibility.v3"
_LANE_STRUCTURES = ("dense", "routed_moe")
_LANE_REGIMES = ("decode", "batch")
_LANE_ROUTE_STATUSES = frozenset({
    "backed", "backed_with_serve_flag", "fallback",
})
_LANE_QUALIFICATIONS = frozenset({"compile_only", "device_qualified"})
_LANE_PREDICATE_FACTS = frozenset({
    "role_split", "in_features", "out_features",
})
_LANE_PREDICATE_OPS = frozenset({
    "equals", "in", "multiple_of", "at_least", "at_most",
})

#: The vLLM package roots a ``top_level_loader_modules`` entry may name. The
#: historical root is ``vllm.model_executor.models``; vLLM 0.24 additionally
#: ships per-architecture packages under ``vllm.models`` whose entrypoint class
#: lives in a platform submodule (``vllm.models.deepseek_v4.nvidia.model``
#: defines ``DeepseekV4ForCausalLM``; ``vllm/models/deepseek_v4/__init__.py``
#: only re-exports it). ``plugin.py`` matches on the module that DEFINES a class
#: (``__module__`` guard), so the contract has to be able to name that
#: submodule. The list stays an explicit two-entry allow-list rather than a
#: bare ``vllm.`` prefix: every entry here becomes a dynamic import into the
#: serving process, which is exactly what ``tests/test_no_triton_runtime.py``
#: pins against a reviewed allow-list.
_LOADER_MODULE_ROOTS = ("vllm.model_executor.models.", "vllm.models.")

#: The one parallelism axis this contract makes claims about.  The value is
#: vLLM's live tensor-parallel world size as read by
#: ``gridbook.config._initialized_tensor_parallel_world_size`` — not a CLI
#: argument, which can disagree with the running worker.
_TP_AXIS = "vllm_tensor_parallel_world_size"

#: Serving-unit ids that are NOT CB format families, i.e. the keys of
#: ``FORMATS`` in ``gridbook/source_passthrough.py``.  Every unit a producer
#: can address needs an explicit tensor-parallel row: the completeness checks
#: below refuse a table that omits or invents one.
_PASSTHROUGH_TP_UNITS = frozenset({
    "fp8_e4m3_ue8m0_block128",
    "mxfp4_e2m1_ue8m0_g32",
    "mxfp8_e4m3_e8m0_g32",
})

#: Units whose serving method branches on an execution arm with its own TP
#: refusal site: the FP8-source W8A16 lane gates dense and grouped-BMM arms
#: separately (``gridbook/fp8_source_w8a16.py``), and the MXFP8 lane audits its
#: BMM arm separately (``gridbook/mxfp8_dense_lane.py``).  A unit not listed
#: here must publish ONE flat claim and must not carry ``arms``.
_TP_ARMED_UNITS: dict[str, tuple[str, ...]] = {
    "fp8_e4m3_ue8m0_block128": ("dense", "bmm"),
    "mxfp8_e4m3_e8m0_g32": ("dense", "bmm"),
}

#: Arms whose serving method admits MORE than one rank under structural shard
#: laws it enforces itself, keyed by (unit, arm) and pinned to the enforcement
#: site's own constants.  The FP8-source W8A16 DENSE arm is the one entry
#: (v7, 2026-08-23): ``gridbook/fp8_source_w8a16.py`` derives each Linear's
#: shard degree from its ``create_weights`` arguments (never from
#: ``layer.tp_size``, which vLLM stamps on replicated layers too) and refuses,
#: before any parameter exists, any shard whose per-rank extent on the sharded
#: axis is not a whole multiple of the 128-element source block — the exact
#: condition under which vLLM's ``BlockQuantScaleParameter`` narrow indexes the
#: UE8M0 block grid correctly.  Merged planes must satisfy it per fused role.
#: An arm listed here publishes those LAWS and no numeric cap, because no
#: enforcement site imposes one; an arm not listed here publishes a cap.
_TP_LAW_ADMITTED_ARMS: dict[tuple[str, str], dict[str, Any]] = {
    ("fp8_e4m3_ue8m0_block128", "dense"): {
        "input_axis_group": 128,
        "output_axis_quantum": 128,
        "merged_roles": "per_role_group_multiple",
    },
    # The grouped arm obeys the same alignment law AND a closed list of
    # measured shard degrees, because column-sharding a grouped plane divides
    # the kernel's group count: alignment alone cannot admit a degree whose
    # geometry nobody ran.  The list mirrors
    # ``fp8_source_w8a16._DSV4_BMM_QUALIFIED_SHARD_DEGREES``.
    ("fp8_e4m3_ue8m0_block128", "bmm"): {
        "input_axis_group": 128,
        "output_axis_quantum": 128,
        "merged_roles": "per_role_group_multiple",
        "qualified_shard_degrees": [1, 2, 4],
    },
}

#: The exact grouped-BMM geometry each armed unit qualifies, keyed by
#: (unit, arm).  Today only the FP8-source W8A16 BMM arm is pinned, to the
#: constants in ``gridbook/fp8_source_w8a16.py`` (``_DSV4_BMM_GROUPS``,
#: ``_DSV4_BMM_ROWS``, ``_DSV4_BMM_K``).  The geometry is the UNSHARDED plane;
#: which shard degrees of it are admitted is the separate
#: ``qualified_shard_degrees`` list on the arm's ``shard_admission``, because
#: column-sharding a grouped plane divides the kernel's group count and each
#: degree is its own measurement.  An arm without an entry here must omit
#: ``requires_geometry`` entirely.
_TP_REQUIRED_GEOMETRY: dict[tuple[str, str], dict[str, int]] = {
    ("fp8_e4m3_ue8m0_block128", "bmm"): {
        "bmm_groups": 8,
        "rows_per_group": 1024,
        "k": 4096,
    },
}

#: The shard-admission laws every ``cb_format_family`` unit publishes instead
#: of a numeric cap, pinned from the dense-CB enforcement site
#: ``gridbook/linear.py``.  ``_TP_CB_INPUT_AXIS_GROUP`` is ``codec.SUPERBLOCK``
#: (gridbook/codec.py:21): a row-parallel K-shard must contain whole packed
#: superblocks, checked in
#: ``PrismaQuantCBLinearMethod._require_shard_group_alignment``.  The output-
#: axis quanta are the native kernel row-alignment quanta (fp4: 8-wide rows,
#: fp8: 16-wide) that a column-parallel logical shard must not cut, enforced
#: by the same method for sharded layers.  ``_TP_CB_MERGED_ROLES`` restates
#: that a merged checkpoint role must divide evenly across ranks
#: (``PrismaQuantCBLinearMethod._rank_local_role_widths``).  Dense CB units
#: publish NO numeric cap because no dispatch path enforces one: above one
#: rank, admission IS these laws, evaluated per rank at weight construction
#: and raised as ``ShardGroupAlignmentError`` before any buffer exists.
_TP_CB_INPUT_AXIS_GROUP = 256
_TP_CB_OUTPUT_AXIS_QUANTA: dict[str, int] = {"fp4": 8, "fp8": 16}
_TP_CB_MERGED_ROLES = "even_division"

#: The composite surface: one vLLM merged projection whose roles have
#: DIFFERENT Gridbook formats (DeepSeek-V4's shared-expert ``gate_up_proj``
#: fuses a CB gate with a block-FP8 source-passthrough up).  It is not a
#: format, so it is neither a CB family nor a passthrough unit, and it has no
#: alignment law of its own to publish: ``gridbook/mixed_linear.py`` derives
#: the column degree from vLLM's own ``create_weights`` arguments, refuses a
#: row-parallel split of a merged plane (hence ``axes``), and builds every
#: role's carrier at that ROLE's whole-tensor output size so the role's
#: existing gate — a CB family's ``shard_admission`` above, or a passthrough
#: arm's — decides that carrier's legality.  ``per_role_law: "inherited"`` is
#: that fact as data: a consumer pre-checking a mixed module evaluates each
#: role against the role's own row, and there is no number here to check
#: instead.  Publishing a numeric cap would assert more than any site
#: enforces, which is the same reason the dense CB rows carry none.
_TP_MIXED_FUSED_UNIT = "mixed_fused_projection"
_TP_MIXED_FUSED_ADMISSION: dict[str, Any] = {
    "axes": ["output"],
    "per_role_law": "inherited",
}

#: Expert parallelism is a SECOND axis, not a relaxation of the first. The
#: ``tensor_parallel`` table above is about splitting one unit's rows/columns
#: across ranks; this one is about splitting a routed MoE layer's EXPERTS,
#: leaving every expert whole. A CB expert stack cannot appear in the first
#: table — its last dimension is superblock bytes, not input columns — and
#: needs no shard laws in this one, because nothing is sharded within an
#: expert.
_EP_AXIS = "vllm_expert_parallel_size"

#: The topology predicate ``config.py::_require_ep_moe_serving`` enforces at
#: method construction, restated field by field. Each entry is one refusal
#: branch there: ``use_ep`` must be on (``vllm_flag``); the MoE layer's own
#: ``tp_size`` must be 1; ``use_all2all_kernels`` must be false, which is
#: vLLM's own way of saying dp/pcp/sequence-parallel EP is off and no
#: dispatch/combine token exchange is expected of the method; EPLB must be off
#: (Gridbook holds whole stacks resident and cannot follow a re-placement);
#: and ``skip_final_all_reduce`` must be off, because Gridbook returns this
#: rank's partial and relies on vLLM's stock final all-reduce
#: (``fused_moe/runner/moe_runner.py::_maybe_reduce_final_output``) to sum the
#: ranks.
_EP_REQUIRES: dict[str, object] = {
    "vllm_flag": "--enable-expert-parallel",
    "moe_tensor_parallel_size": 1,
    "all2all_kernels": False,
    "expert_load_balancing": False,
    "skip_final_all_reduce": False,
}

#: The per-unit admission laws for a CB MoE expert stack above one rank. Every
#: value names an enforcement site: ``placement`` and
#: ``checkpoint_leading_dim`` are ``moe_ep.local_expert_gather_index`` and
#: ``moe_ep.gather_expert_major`` (both loaders funnel through the latter);
#: ``remote_pair_handling`` is ``moe_ep.remap_local_expert_ids``, applied
#: inside the opaque custom op before every forward path;
#: ``cross_rank_reduction`` is vLLM's, not Gridbook's. ``sharded_dims: none``
#: is the substantive claim — a rank's expert bytes are byte-identical to the
#: corresponding slice of the world-size-1 stack, so per-expert numerics do
#: not change with the world size.
_EP_CB_ADMISSION: dict[str, str] = {
    "shard_axis": "expert",
    "sharded_dims": "none",
    "placement": "monotone_bijection",
    "checkpoint_leading_dim": "global_expert_count",
    "remote_pair_handling": "zero_weight_alias",
    "cross_rank_reduction": "vllm_final_all_reduce",
}

#: Units this build refuses above one rank on the expert-parallel axis, with
#: the cap each publishes. Mixed per-expert-format stacks declare their format
#: partition over GLOBAL expert ids, so a rank owning a subset can neither
#: size nor fill the per-format sub-stacks
#: (``config.py``, the ``per-expert format groups`` refusal site).
_EP_REFUSED_UNITS: dict[str, int] = {
    "cb_moe_per_expert_format_groups": 1,
}


def _has_law_admitted_arm(unit_id: str, arms) -> bool:
    """Whether any arm of *unit_id* is admitted above one rank by shard laws."""

    return any((unit_id, arm) in _TP_LAW_ADMITTED_ARMS for arm in (arms or ()))


class RuntimeContractError(ValueError):
    """The packaged runtime contract is malformed or internally inconsistent."""


def _fail(path: str, message: str) -> None:
    raise RuntimeContractError(f"{path}: {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        _fail(path, f"missing field(s): {missing}")
    if extra:
        _fail(path, f"unknown field(s): {extra}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(path, "must be a positive integer")
    return value


def _non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a non-negative integer")
    return value


def _unique_strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(path, "must be a non-empty JSON array")
    result = [_string(item, f"{path}[{index}]")
              for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        _fail(path, "must not contain duplicates")
    return result


def _unique_strings_allow_empty(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be a JSON array")
    result = [_string(item, f"{path}[{index}]")
              for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        _fail(path, "must not contain duplicates")
    return result


def _sorted_unique_ints(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        _fail(path, "must be a non-empty JSON array")
    result = [_positive_int(item, f"{path}[{index}]")
              for index, item in enumerate(value)]
    if result != sorted(set(result)):
        _fail(path, "must contain sorted, unique positive integers")
    return result


def _validate_version_set(value: Any, path: str) -> set[int]:
    obj = _object(value, path)
    _keys(obj, path, {"current", "supported"})
    current = _positive_int(obj["current"], f"{path}.current")
    supported = set(_sorted_unique_ints(obj["supported"],
                                        f"{path}.supported"))
    if current not in supported:
        _fail(path, "current must be listed in supported")
    return supported


def _validate_tensor_parallel(root: Mapping[str, Any],
                              family_grids: Mapping[str, str],
                              trellis_families: set[str]) -> None:
    """Validate the tensor-parallel capability table.

    The table is ATTESTATION, not aspiration: every row must restate what a
    refusal site in this package actually enforces, and the closed-world
    reading makes absence a refusal.  A consumer serving unit *U* at tensor-
    parallel size *t* looks up exactly one row named ``U`` (and, for an armed
    unit, the row for the arm it will execute); no match, or a numeric claim
    that does not cover *t*, or a pinned geometry that does not match exactly
    is a REFUSAL.  There is no default, wildcard, or inheritance anywhere in
    this section.

    Two claim shapes exist, matching the two enforcement shapes in the
    runtime:

    * ``source_passthrough_format`` units (and every Gridbook-owned lane
      behind them) are gated by a NUMERIC world-size ceiling, so their rows
      carry ``max_world_size`` — pinned to 1 below.
    * ``cb_format_family`` units are the dense CB Linears lifted above one
      rank by the shard-aware loading wave.  No dispatch path enforces a
      numeric ceiling for them, so publishing a number would assert more
      than the code stands behind; their rows instead carry
      ``shard_admission`` — the structural laws per-rank geometry must
      satisfy, enforced at weight construction by
      ``ShardGroupAlignmentError``.  A shard that violates them is refused
      by the runtime regardless of what any table says; the row exists so a
      producer can pre-check the same laws.
    * the ``mixed_fused_projection`` unit is a COMPOSITE of the rows above,
      not a format: it publishes the axis it admits and the fact that each
      role's legality is the role's own row.  Neither a cap nor a law of its
      own would be true of it.
    * ``trellis_format_family`` units are the TCQ lanes, and they take the
      NUMERIC shape: ``config.py::_build_trellis_method`` calls
      ``_require_tp1_serving("Gridbook trellis dense lanes", ...)`` at method
      construction, before a parameter exists.  A trellis wire's rows are
      bit-packed against a shared per-column rate schedule and carry their own
      alphabets, so a rank's slice is not a byte range of the whole; a sharded
      artifact needs per-rank wires.  The cap restates that refusal site.
    """

    tp = _object(root["tensor_parallel"], "contract.tensor_parallel")
    _keys(tp, "contract.tensor_parallel", {"axis", "semantics", "units"})
    if tp["axis"] != _TP_AXIS:
        _fail("contract.tensor_parallel.axis", f"must be {_TP_AXIS!r}")
    if tp["semantics"] != "closed_world":
        _fail("contract.tensor_parallel.semantics",
              "must be 'closed_world': an unclaimed unit is refused, never "
              "permitted by default")

    units = tp["units"]
    if not isinstance(units, list) or not units:
        _fail("contract.tensor_parallel.units",
              "must be a non-empty JSON array")
    seen: set[str] = set()
    cb_units: set[str] = set()
    passthrough_units: set[str] = set()
    mixed_units: set[str] = set()
    trellis_units: set[str] = set()
    for index, item in enumerate(units):
        path = f"contract.tensor_parallel.units[{index}]"
        row = _object(item, path)
        unit_id = _string(row["unit"], f"{path}.unit")
        kind = _string(row["kind"], f"{path}.kind")
        if kind == "trellis_format_family":
            trellis_units.add(unit_id)
            _keys(row, path, {"unit", "kind", "max_world_size"})
            if unit_id in seen:
                _fail("contract.tensor_parallel.units",
                      f"duplicate unit {unit_id!r}")
            seen.add(unit_id)
            if unit_id not in trellis_families:
                _fail(f"{path}.unit",
                      f"no trellis format family {unit_id!r} in "
                      "contract.formats")
            cap = _positive_int(row["max_world_size"],
                                f"{path}.max_world_size")
            if cap != 1:
                _fail(f"{path}.max_world_size",
                      "must be 1; config.py::_build_trellis_method refuses "
                      "every trellis target above one rank")
            continue
        if kind == "cb_format_family":
            cb_units.add(unit_id)
            expected = {"unit", "kind", "shard_admission"}
        elif kind == "source_passthrough_format":
            passthrough_units.add(unit_id)
            # Whether a unit is armed is a property of its enforcement sites,
            # so a mismatch is named here rather than as a generic
            # missing/unknown field.
            armed = _TP_ARMED_UNITS.get(unit_id)
            if "arms" in row:
                if armed is None:
                    _fail(f"{path}.arms",
                          f"unit {unit_id!r} has one flat TP refusal site; "
                          "publish one flat claim without 'arms'")
                expected = {"unit", "kind", "arms"}
            else:
                if armed is not None:
                    _fail(path,
                          f"unit {unit_id!r} has per-arm TP refusal sites; it "
                          f"must declare arms {sorted(armed)}")
                expected = {"unit", "kind", "max_world_size"}
            # A unit with a law-admitted arm publishes NO unit-level number:
            # one scalar cannot cover an arm admitted by laws and an arm
            # capped at one rank at the same time.  A unit whose every arm is
            # capped keeps the cap, because a site does enforce it.
            if not _has_law_admitted_arm(unit_id, armed):
                expected = expected | {"max_world_size"}
        elif kind == "mixed_fused_projection":
            mixed_units.add(unit_id)
            expected = {"unit", "kind", "shard_admission"}
        else:
            _fail(f"{path}.kind",
                  "must be 'cb_format_family', 'source_passthrough_format', "
                  "'mixed_fused_projection' or 'trellis_format_family'")
        _keys(row, path, expected)
        if unit_id in seen:
            _fail("contract.tensor_parallel.units",
                  f"duplicate unit {unit_id!r}")
        seen.add(unit_id)

        if kind == "mixed_fused_projection":
            if unit_id != _TP_MIXED_FUSED_UNIT:
                _fail(f"{path}.unit",
                      f"must be {_TP_MIXED_FUSED_UNIT!r}: there is one "
                      "composite surface, and it is named for what it is "
                      "rather than for any format")
            admission_path = f"{path}.shard_admission"
            admission = _object(row["shard_admission"], admission_path)
            _keys(admission, admission_path, set(_TP_MIXED_FUSED_ADMISSION))
            for field, pinned in _TP_MIXED_FUSED_ADMISSION.items():
                if admission[field] != pinned:
                    _fail(f"{admission_path}.{field}",
                          f"must equal {pinned!r} (gridbook/mixed_linear.py "
                          "admits the output axis only and inherits every "
                          "role's own law)")
            continue

        if kind == "cb_format_family":
            grid = family_grids.get(unit_id)
            quantum = (_TP_CB_OUTPUT_AXIS_QUANTA.get(grid)
                       if grid is not None else None)
            if quantum is None:
                _fail(f"{path}.unit",
                      f"no CB format family {unit_id!r} in contract.formats")
            admission_path = f"{path}.shard_admission"
            admission = _object(row["shard_admission"], admission_path)
            _keys(admission, admission_path,
                  {"input_axis_group", "output_axis_quantum", "merged_roles"})
            if (_positive_int(admission["input_axis_group"],
                              f"{admission_path}.input_axis_group")
                    != _TP_CB_INPUT_AXIS_GROUP):
                _fail(f"{admission_path}.input_axis_group",
                      f"must equal {_TP_CB_INPUT_AXIS_GROUP} "
                      "(codec.SUPERBLOCK; linear.py refuses a K-shard that "
                      "cuts a packed superblock)")
            if (_positive_int(admission["output_axis_quantum"],
                              f"{admission_path}.output_axis_quantum")
                    != quantum):
                _fail(f"{admission_path}.output_axis_quantum",
                      f"must equal {quantum} for grid {grid!r} "
                      "(linear.py native kernel row-alignment quantum)")
            if admission["merged_roles"] != _TP_CB_MERGED_ROLES:
                _fail(f"{admission_path}.merged_roles",
                      f"must be {_TP_CB_MERGED_ROLES!r} "
                      "(linear.py refuses merged checkpoint roles that do "
                      "not divide evenly across ranks)")
            continue

        required_arms = _TP_ARMED_UNITS.get(unit_id)
        if "max_world_size" in row:
            cap = _positive_int(row["max_world_size"],
                                f"{path}.max_world_size")
            if cap != 1:
                _fail(f"{path}.max_world_size",
                      "must be 1; no enforcement site in this build allows "
                      "more")
        if required_arms is None:
            continue
        arms = row["arms"]
        if not isinstance(arms, list) or not arms:
            _fail(f"{path}.arms", "must be a non-empty JSON array")
        arm_names: list[str] = []
        for arm_index, arm_item in enumerate(arms):
            arm_path = f"{path}.arms[{arm_index}]"
            arm = _object(arm_item, arm_path)
            arm_name = _string(arm.get("arm", ""), f"{arm_path}.arm")
            law = _TP_LAW_ADMITTED_ARMS.get((unit_id, arm_name))
            if law is None:
                arm_expected = {"arm", "max_world_size"}
            else:
                arm_expected = {"arm", "shard_admission"}
            if "requires_geometry" in arm:
                arm_expected = arm_expected | {"requires_geometry"}
            _keys(arm, arm_path, arm_expected)
            if arm_name not in required_arms:
                _fail(f"{arm_path}.arm",
                      f"must be one of {sorted(required_arms)}")
            if arm_name in arm_names:
                _fail(f"{path}.arms", f"duplicate arm {arm_name!r}")
            arm_names.append(arm_name)
            # The geometry pin is independent of how the arm is admitted: a
            # law-admitted arm still names the unsharded plane it qualifies.
            geometry_key = (unit_id, arm_name)
            if geometry_key in _TP_REQUIRED_GEOMETRY:
                pinned = _TP_REQUIRED_GEOMETRY[geometry_key]
                if arm.get("requires_geometry") != pinned:
                    _fail(f"{arm_path}.requires_geometry",
                          f"must equal {pinned}")
            elif "requires_geometry" in arm:
                _fail(f"{arm_path}.requires_geometry",
                      f"is not pinned for {unit_id!r} arm {arm_name!r}; "
                      "omit it rather than under-specifying")
            if law is not None:
                admission_path = f"{arm_path}.shard_admission"
                admission = _object(arm["shard_admission"], admission_path)
                _keys(admission, admission_path, set(law))
                for field, pinned in law.items():
                    if admission[field] != pinned:
                        _fail(f"{admission_path}.{field}",
                              f"must equal {pinned!r} (the shard law "
                              f"{unit_id!r} arm {arm_name!r} enforces at "
                              "weight construction)")
                continue
            arm_cap = _positive_int(arm["max_world_size"],
                                    f"{arm_path}.max_world_size")
            if arm_cap != 1:
                _fail(f"{arm_path}.max_world_size",
                      "must be 1; no enforcement site in this build allows "
                      "more")
        missing = sorted(set(required_arms) - set(arm_names))
        if missing:
            _fail(f"{path}.arms", f"missing arm claim(s): {missing}")

    if cb_units != set(family_grids):
        _fail("contract.tensor_parallel.units",
              f"every CB format family must carry exactly one "
              f"'cb_format_family' claim; missing "
              f"{sorted(set(family_grids) - cb_units)}, unknown "
              f"{sorted(cb_units - set(family_grids))}")
    if mixed_units != {_TP_MIXED_FUSED_UNIT}:
        _fail("contract.tensor_parallel.units",
              "the composite mixed-format fused surface must carry exactly "
              f"one {_TP_MIXED_FUSED_UNIT!r} claim; got "
              f"{sorted(mixed_units)}")
    if passthrough_units != set(_PASSTHROUGH_TP_UNITS):
        _fail("contract.tensor_parallel.units",
              f"source-passthrough claims must equal {sorted(_PASSTHROUGH_TP_UNITS)}; "
              f"missing "
              f"{sorted(set(_PASSTHROUGH_TP_UNITS) - passthrough_units)}, "
              f"unknown {sorted(passthrough_units - set(_PASSTHROUGH_TP_UNITS))}")
    if trellis_units != trellis_families:
        _fail("contract.tensor_parallel.units",
              f"every trellis format family must carry exactly one "
              f"'trellis_format_family' claim; missing "
              f"{sorted(trellis_families - trellis_units)}, unknown "
              f"{sorted(trellis_units - trellis_families)}")


def _validate_expert_parallel(root: Mapping[str, Any],
                              family_grids: Mapping[str, str]) -> None:
    """Validate the expert-parallel capability table.

    Same doctrine as ``_validate_tensor_parallel`` — attestation, closed
    world, no default or wildcard — on a different axis. A consumer asking
    "can this build serve unit *U* with ``--enable-expert-parallel``?" looks up
    exactly one row named *U*: a ``cb_moe_expert_stack`` row is a yes under the
    published ``requires`` topology and ``expert_admission`` laws, a
    ``cb_moe_expert_stack_refused`` row is an explicit no above its cap, and no
    row at all is a refusal.

    Unlike the tensor-parallel table, admission here is TOPOLOGY-wide as well
    as per-unit: expert parallelism only keeps whole experts if no other
    parallel axis is also splitting the layer, so ``requires`` is validated
    against the exact predicate the refusal site enforces.
    """

    ep = _object(root["expert_parallel"], "contract.expert_parallel")
    _keys(ep, "contract.expert_parallel",
          {"axis", "semantics", "requires", "units"})
    if ep["axis"] != _EP_AXIS:
        _fail("contract.expert_parallel.axis", f"must be {_EP_AXIS!r}")
    if ep["semantics"] != "closed_world":
        _fail("contract.expert_parallel.semantics",
              "must be 'closed_world': an unclaimed unit is refused, never "
              "permitted by default")
    requires = _object(ep["requires"], "contract.expert_parallel.requires")
    _keys(requires, "contract.expert_parallel.requires", set(_EP_REQUIRES))
    for key, pinned in _EP_REQUIRES.items():
        got = requires[key]
        if got != pinned or type(got) is not type(pinned):
            _fail(f"contract.expert_parallel.requires.{key}",
                  f"must be {pinned!r} (config.py::_require_ep_moe_serving "
                  "refuses otherwise)")

    units = ep["units"]
    if not isinstance(units, list) or not units:
        _fail("contract.expert_parallel.units",
              "must be a non-empty JSON array")
    seen: set[str] = set()
    cb_units: set[str] = set()
    refused_units: dict[str, int] = {}
    for index, item in enumerate(units):
        path = f"contract.expert_parallel.units[{index}]"
        row = _object(item, path)
        unit_id = _string(row["unit"], f"{path}.unit")
        kind = _string(row["kind"], f"{path}.kind")
        if kind == "cb_moe_expert_stack":
            cb_units.add(unit_id)
            _keys(row, path, {"unit", "kind", "expert_admission"})
            if unit_id not in family_grids:
                _fail(f"{path}.unit",
                      f"no CB format family {unit_id!r} in contract.formats")
            admission_path = f"{path}.expert_admission"
            admission = _object(row["expert_admission"], admission_path)
            _keys(admission, admission_path, set(_EP_CB_ADMISSION))
            for key, pinned in _EP_CB_ADMISSION.items():
                if admission[key] != pinned:
                    _fail(f"{admission_path}.{key}", f"must be {pinned!r}")
        elif kind == "cb_moe_expert_stack_refused":
            _keys(row, path, {"unit", "kind", "max_world_size"})
            cap = _positive_int(row["max_world_size"],
                                f"{path}.max_world_size")
            expected_cap = _EP_REFUSED_UNITS.get(unit_id)
            if expected_cap is None:
                _fail(f"{path}.unit",
                      f"unknown refused expert-parallel unit {unit_id!r}")
            if cap != expected_cap:
                _fail(f"{path}.max_world_size",
                      f"must be {expected_cap} for {unit_id!r}")
            refused_units[unit_id] = cap
        else:
            _fail(f"{path}.kind",
                  "must be 'cb_moe_expert_stack' or "
                  "'cb_moe_expert_stack_refused'")
        if unit_id in seen:
            _fail("contract.expert_parallel.units",
                  f"duplicate unit {unit_id!r}")
        seen.add(unit_id)

    if cb_units != set(family_grids):
        _fail("contract.expert_parallel.units",
              f"every CB format family must carry exactly one "
              f"'cb_moe_expert_stack' claim; missing "
              f"{sorted(set(family_grids) - cb_units)}, unknown "
              f"{sorted(cb_units - set(family_grids))}")
    if set(refused_units) != set(_EP_REFUSED_UNITS):
        _fail("contract.expert_parallel.units",
              f"refused expert-parallel claims must equal "
              f"{sorted(_EP_REFUSED_UNITS)}; missing "
              f"{sorted(set(_EP_REFUSED_UNITS) - set(refused_units))}, "
              f"unknown {sorted(set(refused_units) - set(_EP_REFUSED_UNITS))}")


def _validate_trellis_format(
    fmt: Mapping[str, Any],
    path: str,
    family: str,
) -> None:
    """Validate one TCQ trellis ``formats`` row against the reader domain.

    A trellis row publishes a RATE ladder, not a codebook ladder, so it shares
    no numeric field with a CB row.  ``candidate_rungs_q256`` is the ladder a
    producer may target; ``reader_rate_range_q256`` is the inclusive band this
    package's own wire reader accepts, which is far wider and is a reader fact
    rather than a production one.  Neither is an eligibility claim: which of
    those rates a lane has actually been seen to serve is decided cell by cell
    in ``lane_eligibility``, and a rate on this ladder with no cell naming it
    is unattested.
    """

    _keys(fmt, path, {
        "kind", "family", "name_pattern", "candidate_rungs_q256",
        "reader_rate_range_q256", "native_terminal_q256", "residency_modes",
    })
    law = _TRELLIS_RUNG_LAW.get(family)
    if law is None:
        _fail(f"{path}.family",
              f"must be one of {sorted(_TRELLIS_RUNG_LAW)}")
    pattern = _string(fmt["name_pattern"], f"{path}.name_pattern")
    if pattern.count("{k}") != 1:
        _fail(f"{path}.name_pattern", "must contain exactly one '{k}'")
    expected_pattern = f"{family.removesuffix('_R256')}_R{{k}}"
    if pattern != expected_pattern:
        _fail(f"{path}.name_pattern",
              f"must be {expected_pattern!r}: the rung id is "
              "``trellis.rung_id``'s own spelling, and a second spelling "
              "would name rates no receipt describes")

    candidates = _sorted_unique_ints(
        fmt["candidate_rungs_q256"], f"{path}.candidate_rungs_q256")
    if tuple(candidates) != law["candidate_rungs_q256"]:
        _fail(f"{path}.candidate_rungs_q256",
              f"must equal the candidate ladder "
              f"{list(law['candidate_rungs_q256'])} "
              "(gridbook.trellis.RUNG_POLICIES)")
    band = fmt["reader_rate_range_q256"]
    if not isinstance(band, list) or len(band) != 2:
        _fail(f"{path}.reader_rate_range_q256",
              "must be [floor, ceiling] in body bits per 256 weights")
    floor = _positive_int(band[0], f"{path}.reader_rate_range_q256[0]")
    ceiling = _positive_int(band[1], f"{path}.reader_rate_range_q256[1]")
    if (floor, ceiling) != law["reader_rate_range_q256"]:
        _fail(f"{path}.reader_rate_range_q256",
              f"must equal {list(law['reader_rate_range_q256'])} "
              "(gridbook.trellis.RUNG_POLICIES)")
    if not floor <= candidates[0] or not candidates[-1] <= ceiling:
        _fail(f"{path}.candidate_rungs_q256",
              "every candidate rate must lie inside the reader band")
    terminal = _positive_int(
        fmt["native_terminal_q256"], f"{path}.native_terminal_q256")
    if terminal != law["native_terminal_q256"]:
        _fail(f"{path}.native_terminal_q256",
              f"must equal {law['native_terminal_q256']} "
              "(gridbook.trellis.RUNG_POLICIES)")
    modes = _unique_strings(
        fmt["residency_modes"], f"{path}.residency_modes")
    if tuple(modes) != _TRELLIS_RESIDENCY_MODES:
        _fail(f"{path}.residency_modes",
              f"must equal {list(_TRELLIS_RESIDENCY_MODES)}")


def _validate_trellis_cell(
    cell: Mapping[str, Any],
    path: str,
    family: str,
) -> None:
    """Validate one TCQ trellis lane cell, including what it may NOT say.

    The scoping fields are the whole point of the cell, so each is checked
    against a law rather than merely typed:

    * ``rungs_q256`` must be a subset of the family's CANDIDATE ladder, in the
      same body-bits-per-256-weights units the wire header uses.  The cell
      names the rates a receipt covers, and every other rate on the ladder is
      unattested by absence.
    * ``activation_contract`` must be the lane's own executed A-side contract.
      It records which route RAN, never that the route is good: no quality,
      accuracy or speed claim is representable in this table at all.
    * ``route_status`` must be ``backed_with_serve_flag`` naming the family's
      opt-in flags, because the lane refuses construction with them unset.
      A plain ``backed`` would claim a default serve reaches this route.
    """

    _keys(cell, path, {
        "id", "platform", "family", "structure", "regime", "rungs_q256",
        "activation_contract", "route_status", "qualification",
        "requires_serve_flags", "predicates",
    })
    law_rungs = _TRELLIS_RUNG_LAW[family]["candidate_rungs_q256"]
    rungs = set(_sorted_unique_ints(
        cell["rungs_q256"], f"{path}.rungs_q256"))
    if not rungs:
        _fail(f"{path}.rungs_q256",
              "must name at least one rate; an empty cell attests nothing")
    if not rungs <= set(law_rungs):
        _fail(f"{path}.rungs_q256",
              "must be a subset of formats[family].candidate_rungs_q256; "
              f"non-candidate rates {sorted(rungs - set(law_rungs))}")
    contract = _string(
        cell["activation_contract"], f"{path}.activation_contract")
    expected_contract = _TRELLIS_ACTIVATION_CONTRACTS[family]
    if contract != expected_contract:
        _fail(f"{path}.activation_contract",
              f"must be {expected_contract!r}: the {family} lane's apply() "
              "takes exactly one A-side route and this table restates it")
    flags = tuple(_unique_strings(
        cell["requires_serve_flags"], f"{path}.requires_serve_flags"))
    expected_flags = _TRELLIS_SERVE_FLAGS[family]
    if flags != expected_flags:
        _fail(f"{path}.requires_serve_flags",
              f"must equal {list(expected_flags)}: the lane is opt-in and "
              "its residency mode has no default")


def _validate_lane_eligibility(
    root: Mapping[str, Any],
    family_rungs: Mapping[str, set[int]],
    trellis_families: set[str],
) -> None:
    """Validate platform-scoped route facts without claiming graph support.

    Gridbook owns the byte layout and dispatch route, so it publishes those
    facts here.  A producer's graph/capture policy remains outside this
    contract: no cell says anything about CUDA-graph capture, and a consumer
    must not read one as though it did.

    ``qualification`` separates the two ways a cell can have come to exist:

    * ``compile_only`` -- the route is structurally backed and its kernels
      cross-compile, but no device has been seen to take it.  Every CB cell is
      this, and the sm_120 rows are explicitly a cross-compilation receipt
      rather than an RTX 50 serve.
    * ``device_qualified`` -- a real vLLM process on that exact compute
      capability loaded an artifact, dispatched these Linears to this lane,
      and generated.  The four sm_121 trellis cells are the first, from the
      2026-08-29 GB10 container receipt (4 combinations of
      {E4M3, E2M1} x {resident, streamed}, code planes and scale operands
      byte-exact against the wire).

    What ``device_qualified`` does NOT mean, and what nothing in this table
    can be read to mean: that the route is accurate, fast, or worth choosing.
    ``route_status`` and ``activation_contract`` say which route EXECUTES.
    Model quality is not representable here, and the trellis receipt was taken
    on a synthetic checkpoint with random weights, so it could not have
    established quality even if the schema had somewhere to put it.

    Scope is carried by the cell's own typed fields and by closed-world
    absence, never by prose -- ``detail``/``rationale`` keys are refused
    outright.  A rung with no cell naming it, a structure with no cell, a
    platform with no cell: all unattested.  The trellis cells are therefore
    ``dense`` only (routed MoE never reaches ``_build_trellis_method``, which
    is called only under ``isinstance(layer, LinearBase)``), name one rate
    each out of their family's candidate ladder, and are pinned to TP=1 by
    their ``tensor_parallel`` row rather than by anything said here.
    """

    eligibility = _object(
        root["lane_eligibility"], "contract.lane_eligibility")
    _keys(eligibility, "contract.lane_eligibility", {
        "schema", "platforms", "regimes", "structures", "cells",
    })
    if eligibility["schema"] != _LANE_ELIGIBILITY_SCHEMA:
        _fail("contract.lane_eligibility.schema",
              f"must be {_LANE_ELIGIBILITY_SCHEMA!r}")

    regimes = _unique_strings(
        eligibility["regimes"], "contract.lane_eligibility.regimes")
    if regimes != list(_LANE_REGIMES):
        _fail("contract.lane_eligibility.regimes",
              f"must equal {list(_LANE_REGIMES)}")
    structures = _unique_strings(
        eligibility["structures"], "contract.lane_eligibility.structures")
    if structures != list(_LANE_STRUCTURES):
        _fail("contract.lane_eligibility.structures",
              f"must equal {list(_LANE_STRUCTURES)}")

    platforms = _object(
        eligibility["platforms"], "contract.lane_eligibility.platforms")
    if not platforms:
        _fail("contract.lane_eligibility.platforms",
              "must name at least one exact platform")
    for platform, item in platforms.items():
        platform_path = f"contract.lane_eligibility.platforms.{platform}"
        _string(platform, platform_path)
        platform_data = _object(item, platform_path)
        _keys(platform_data, platform_path, {"compute_capability"})
        capability = platform_data["compute_capability"]
        if not isinstance(capability, list) or len(capability) != 2:
            _fail(f"{platform_path}.compute_capability",
                  "must be [major, minor]")
        major = _positive_int(
            capability[0], f"{platform_path}.compute_capability[0]")
        minor = _non_negative_int(
            capability[1], f"{platform_path}.compute_capability[1]")
        if platform != f"sm_{major}{minor}":
            _fail(platform_path,
                  f"platform id must be the exact capability name "
                  f"'sm_{major}{minor}'")

    cells = eligibility["cells"]
    if not isinstance(cells, list) or not cells:
        _fail("contract.lane_eligibility.cells",
              "must be a non-empty JSON array")
    seen_ids: set[str] = set()
    for index, item in enumerate(cells):
        path = f"contract.lane_eligibility.cells[{index}]"
        cell = _object(item, path)
        family = _string(cell.get("family", ""), f"{path}.family")
        is_trellis = family in trellis_families
        if is_trellis:
            _validate_trellis_cell(cell, path, family)
        else:
            _keys(cell, path, {
                "id", "platform", "family", "structure", "regime", "rungs",
                "route_status", "qualification", "requires_serve_flags",
                "predicates",
            })
        cell_id = _string(cell["id"], f"{path}.id")
        if cell_id in seen_ids:
            _fail("contract.lane_eligibility.cells",
                  f"duplicate cell id {cell_id!r}")
        seen_ids.add(cell_id)

        platform = _string(cell["platform"], f"{path}.platform")
        if platform not in platforms:
            _fail(f"{path}.platform",
                  f"must name one of {sorted(platforms)}")
        if not is_trellis:
            rung_law = _FORMAT_RUNGS.get(family)
            if rung_law is None or family not in family_rungs:
                _fail(f"{path}.family",
                      f"must name one of "
                      f"{sorted(set(family_rungs) | trellis_families)}")
            producer_rungs = set(rung_law[1])
            rungs = set(_sorted_unique_ints(cell["rungs"], f"{path}.rungs"))
            if not rungs <= producer_rungs:
                _fail(f"{path}.rungs",
                      "must be a subset of formats[family].producer_rungs; "
                      f"non-producer rungs {sorted(rungs - producer_rungs)}")
        structure = _string(cell["structure"], f"{path}.structure")
        if structure not in _LANE_STRUCTURES:
            _fail(f"{path}.structure",
                  f"must be one of {list(_LANE_STRUCTURES)}")
        regime = _string(cell["regime"], f"{path}.regime")
        if regime not in _LANE_REGIMES:
            _fail(f"{path}.regime",
                  f"must be one of {list(_LANE_REGIMES)}")
        route_status = _string(
            cell["route_status"], f"{path}.route_status")
        if route_status not in _LANE_ROUTE_STATUSES:
            _fail(f"{path}.route_status",
                  f"must be one of {sorted(_LANE_ROUTE_STATUSES)}; an "
                  "unbacked route is represented by closed-world absence")
        qualification = _string(
            cell["qualification"], f"{path}.qualification")
        if qualification not in _LANE_QUALIFICATIONS:
            _fail(f"{path}.qualification",
                  f"must be one of {sorted(_LANE_QUALIFICATIONS)}")

        flags = _unique_strings_allow_empty(
            cell["requires_serve_flags"], f"{path}.requires_serve_flags")
        if route_status == "backed_with_serve_flag" and not flags:
            _fail(path, "backed_with_serve_flag must name a serve flag")
        if route_status != "backed_with_serve_flag" and flags:
            _fail(path, "requires_serve_flags must be empty unless "
                  "route_status is 'backed_with_serve_flag'")

        predicates = cell["predicates"]
        if not isinstance(predicates, list):
            _fail(f"{path}.predicates", "must be a JSON array")
        for predicate_index, item in enumerate(predicates):
            predicate_path = f"{path}.predicates[{predicate_index}]"
            predicate = _object(item, predicate_path)
            _keys(predicate, predicate_path, {"fact", "op", "value"})
            fact = _string(predicate["fact"], f"{predicate_path}.fact")
            if fact not in _LANE_PREDICATE_FACTS:
                _fail(f"{predicate_path}.fact",
                      f"must be one of {sorted(_LANE_PREDICATE_FACTS)}")
            op = _string(predicate["op"], f"{predicate_path}.op")
            if op not in _LANE_PREDICATE_OPS:
                _fail(f"{predicate_path}.op",
                      f"must be one of {sorted(_LANE_PREDICATE_OPS)}")
            value = predicate["value"]
            if op == "in":
                if not isinstance(value, list) or not value:
                    _fail(f"{predicate_path}.value",
                          "must be a non-empty JSON array for op 'in'")
            elif op == "multiple_of":
                _positive_int(value, f"{predicate_path}.value")
            elif op in {"at_least", "at_most"}:
                if isinstance(value, bool) or not isinstance(value, int):
                    _fail(f"{predicate_path}.value", "must be an integer")


def validate_runtime_contract(contract: Any) -> None:
    """Raise :class:`RuntimeContractError` unless *contract* is self-consistent."""

    root = _object(contract, "contract")
    _keys(root, "contract", {
        "schema", "contract_version", "abi_features", "quant_method",
        "packing", "layout", "formats", "lane_eligibility",
        "tensor_parallel", "expert_parallel", "producer_profiles",
    })
    if root["schema"] != RUNTIME_CONTRACT_SCHEMA:
        _fail("contract.schema", f"must be {RUNTIME_CONTRACT_SCHEMA!r}")
    contract_version = _positive_int(
        root["contract_version"], "contract.contract_version")
    if contract_version != 12:
        _fail("contract.contract_version", "must be 12 for this schema")

    features = _object(root["abi_features"], "contract.abi_features")
    _keys(features, "contract.abi_features", {
        "dspark_construction_physical_bridge",
        "routed_moe_per_role_codebook_lut",
        "source_fp8_block128_w8a16",
    })
    dspark_bridge = _positive_int(
        features["dspark_construction_physical_bridge"],
        "contract.abi_features.dspark_construction_physical_bridge",
    )
    if dspark_bridge != 1:
        _fail(
            "contract.abi_features.dspark_construction_physical_bridge",
            "must be 1",
        )
    per_role_lut = _positive_int(
        features["routed_moe_per_role_codebook_lut"],
        "contract.abi_features.routed_moe_per_role_codebook_lut",
    )
    if per_role_lut != 1:
        _fail(
            "contract.abi_features.routed_moe_per_role_codebook_lut",
            "must be 1",
        )
    source_fp8_w8a16 = _positive_int(
        features["source_fp8_block128_w8a16"],
        "contract.abi_features.source_fp8_block128_w8a16",
    )
    if source_fp8_w8a16 != 1:
        _fail(
            "contract.abi_features.source_fp8_block128_w8a16",
            "must be 1",
        )

    quant = _object(root["quant_method"], "contract.quant_method")
    _keys(quant, "contract.quant_method", {"canonical", "accepted", "legacy"})
    canonical = _string(quant["canonical"], "contract.quant_method.canonical")
    accepted = set(_unique_strings(
        quant["accepted"], "contract.quant_method.accepted"))
    legacy = set(_unique_strings(
        quant["legacy"], "contract.quant_method.legacy"))
    if canonical not in accepted:
        _fail("contract.quant_method", "canonical must be accepted")
    if canonical in legacy:
        _fail("contract.quant_method", "canonical must not be legacy")
    if not legacy < accepted:
        _fail("contract.quant_method", "legacy must be a proper subset of accepted")

    packing = _object(root["packing"], "contract.packing")
    _keys(packing, "contract.packing", {
        "vector_dim", "superblock_weights", "codewords_per_superblock",
        "index_bytes_per_k", "index_bit_order", "subindex_split",
    })
    for field, expected in {
        "vector_dim": 8,
        "superblock_weights": 256,
        "codewords_per_superblock": 32,
        "index_bytes_per_k": 4,
    }.items():
        actual = _positive_int(packing[field], f"contract.packing.{field}")
        if actual != expected:
            _fail(f"contract.packing.{field}", f"must be {expected}")
    if packing["index_bit_order"] != "lsb_first":
        _fail("contract.packing.index_bit_order", "must be 'lsb_first'")
    if packing["subindex_split"] != "ceil_first":
        _fail("contract.packing.subindex_split", "must be 'ceil_first'")
    if (packing["codewords_per_superblock"] * packing["vector_dim"]
            != packing["superblock_weights"]):
        _fail("contract.packing",
              "codewords_per_superblock * vector_dim must equal "
              "superblock_weights")

    layout = _object(root["layout"], "contract.layout")
    _keys(layout, "contract.layout", {
        "field", "current", "supported", "default_when_absent",
        "scale_coding_field", "scale_coding_default_when_absent",
        "type_size_rules",
    })
    if layout["field"] != "layout_version":
        _fail("contract.layout.field", "must be 'layout_version'")
    layout_versions = _validate_version_set({
        "current": layout["current"], "supported": layout["supported"]},
        "contract.layout",
    )
    default_layout = _positive_int(
        layout["default_when_absent"],
        "contract.layout.default_when_absent",
    )
    if default_layout not in layout_versions:
        _fail("contract.layout", "default_when_absent must be supported")
    if layout["scale_coding_field"] != "scale_coding.kind":
        _fail("contract.layout.scale_coding_field",
              "must be 'scale_coding.kind'")
    if layout["scale_coding_default_when_absent"] != "v1":
        _fail("contract.layout.scale_coding_default_when_absent",
              "must be 'v1'")
    rules = layout["type_size_rules"]
    if not isinstance(rules, list) or not rules:
        _fail("contract.layout.type_size_rules",
              "must be a non-empty JSON array")
    actual_rules: set[tuple[str, int, str, int]] = set()
    for index, item in enumerate(rules):
        path = f"contract.layout.type_size_rules[{index}]"
        rule = _object(item, path)
        _keys(rule, path, {
            "grid", "layout_version", "scale_coding", "scale_plane_bytes",
        })
        grid = _string(rule["grid"], f"{path}.grid")
        version = _positive_int(
            rule["layout_version"], f"{path}.layout_version")
        coding = _string(rule["scale_coding"], f"{path}.scale_coding")
        scale_bytes = rule["scale_plane_bytes"]
        if (isinstance(scale_bytes, bool) or not isinstance(scale_bytes, int)
                or scale_bytes < 0):
            _fail(f"{path}.scale_plane_bytes",
                  "must be a non-negative integer")
        value = (grid, version, coding, scale_bytes)
        if value in actual_rules:
            _fail("contract.layout.type_size_rules", f"duplicate rule {value}")
        actual_rules.add(value)
    expected_rules = {
        ("fp4", 1, "v1", 16),
        ("fp4", 2, "two_tier", 9),
        ("fp8", 1, "v1", 0),
    }
    if actual_rules != expected_rules:
        _fail("contract.layout.type_size_rules",
              f"must equal {sorted(expected_rules)}")

    formats = root["formats"]
    if not isinstance(formats, list) or not formats:
        _fail("contract.formats", "must be a non-empty JSON array")
    family_grids: dict[str, str] = {}
    family_rungs: dict[str, set[int]] = {}
    trellis_families: set[str] = set()
    seen_families: set[str] = set()
    for index, item in enumerate(formats):
        path = f"contract.formats[{index}]"
        fmt = _object(item, path)
        kind = _string(fmt.get("kind", ""), f"{path}.kind")
        if kind not in _FORMAT_KINDS:
            _fail(f"{path}.kind", f"must be one of {list(_FORMAT_KINDS)}")
        family = _string(fmt.get("family", ""), f"{path}.family")
        if family in seen_families:
            _fail("contract.formats", f"duplicate family {family!r}")
        seen_families.add(family)
        if kind == "tcq_trellis":
            _validate_trellis_format(fmt, path, family)
            trellis_families.add(family)
            continue
        _keys(fmt, path, {
            "kind", "family", "name_pattern", "grid", "mode", "n_sub", "rungs",
            "producer_rungs", "layout_versions", "moe_layout_versions",
        })
        family_grids[family] = _string(fmt["grid"], f"{path}.grid")
        pattern = _string(fmt["name_pattern"], f"{path}.name_pattern")
        if pattern.count("{k}") != 1:
            _fail(f"{path}.name_pattern", "must contain exactly one '{k}'")
        grid = _string(fmt["grid"], f"{path}.grid")
        mode = _string(fmt["mode"], f"{path}.mode")
        n_sub = _positive_int(fmt["n_sub"], f"{path}.n_sub")
        if grid not in {"fp4", "fp8"}:
            _fail(f"{path}.grid", "must be 'fp4' or 'fp8'")
        expected_n_sub = {
            ("fp4", "product"): 2,
            ("fp8", "product"): 4,
        }.get((grid, mode))
        if expected_n_sub is None:
            _fail(path, f"unsupported grid/mode pair {(grid, mode)!r}")
        if n_sub != expected_n_sub:
            _fail(f"{path}.n_sub",
                  f"must be {expected_n_sub} for {grid}/{mode}")
        rungs = _sorted_unique_ints(fmt["rungs"], f"{path}.rungs")
        producer_rungs = _sorted_unique_ints(
            fmt["producer_rungs"], f"{path}.producer_rungs")
        if not set(producer_rungs) <= set(rungs):
            _fail(f"{path}.producer_rungs",
                  "must be a subset of the accepted reader rungs")
        rung_law = _FORMAT_RUNGS.get(family)
        if rung_law is None:
            _fail(f"{path}.family",
                  f"must be one of {sorted(_FORMAT_RUNGS)}")
        accepted_law, producer_law = rung_law
        if tuple(rungs) != accepted_law:
            _fail(f"{path}.rungs",
                  f"must equal the accepted reader domain {list(accepted_law)}")
        if tuple(producer_rungs) != producer_law:
            _fail(f"{path}.producer_rungs",
                  f"must equal the canonical producer ladder "
                  f"{list(producer_law)}")
        family_rungs[family] = set(rungs)
        family_layouts = set(_sorted_unique_ints(
            fmt["layout_versions"], f"{path}.layout_versions"))
        moe_layouts = set(_sorted_unique_ints(
            fmt["moe_layout_versions"], f"{path}.moe_layout_versions"))
        if not family_layouts <= layout_versions:
            _fail(path, "layout_versions contains an unsupported ABI layout")
        if not moe_layouts <= family_layouts:
            _fail(path, "moe_layout_versions must be a subset of layout_versions")

    if set(family_grids) != set(_FORMAT_RUNGS):
        _fail("contract.formats",
              f"CB format families must equal {sorted(_FORMAT_RUNGS)}")
    if trellis_families != set(_TRELLIS_RUNG_LAW):
        _fail("contract.formats",
              f"trellis format families must equal "
              f"{sorted(_TRELLIS_RUNG_LAW)}")

    _validate_lane_eligibility(root, family_rungs, trellis_families)
    _validate_tensor_parallel(root, family_grids, trellis_families)
    _validate_expert_parallel(root, family_grids)

    profiles = _object(root["producer_profiles"],
                       "contract.producer_profiles")
    _keys(profiles, "contract.producer_profiles", {
        "supported_ids", "top_level_loader_modules",
    })
    supported_profile_ids = set(_unique_strings(
        profiles["supported_ids"],
        "contract.producer_profiles.supported_ids",
    ))
    modules = _unique_strings(
        profiles["top_level_loader_modules"],
        "contract.producer_profiles.top_level_loader_modules",
    )
    for index, module in enumerate(modules):
        path = f"contract.producer_profiles.top_level_loader_modules[{index}]"
        if not module.startswith(_LOADER_MODULE_ROOTS):
            _fail(path,
                  "must be a vllm.model_executor.models or vllm.models module")
    if not supported_profile_ids:
        _fail("contract.producer_profiles.supported_ids",
              "must declare at least one producer profile")


def load_runtime_contract() -> dict[str, Any]:
    """Return the validated contract bundled with the installed distribution."""

    resource = files("gridbook").joinpath(_RESOURCE_NAME)
    try:
        with resource.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            f"cannot load packaged {_RESOURCE_NAME}: {exc}") from exc
    validate_runtime_contract(contract)
    return contract


__all__ = [
    "RUNTIME_CONTRACT_SCHEMA",
    "RuntimeContractError",
    "load_runtime_contract",
    "validate_runtime_contract",
]
