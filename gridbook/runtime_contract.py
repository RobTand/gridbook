"""Load and validate Gridbook's packaged producer/runtime contract.

This module is deliberately standard-library-only.  Exporters and compatibility
checks can inspect the installed Gridbook contract without importing torch,
vLLM, CUDA extensions, or any producing project.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping


RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v9"
_RESOURCE_NAME = "runtime_contract.json"

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


def _unique_strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(path, "must be a non-empty JSON array")
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
                              family_grids: Mapping[str, str]) -> None:
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
    for index, item in enumerate(units):
        path = f"contract.tensor_parallel.units[{index}]"
        row = _object(item, path)
        unit_id = _string(row["unit"], f"{path}.unit")
        kind = _string(row["kind"], f"{path}.kind")
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
                  "must be 'cb_format_family', 'source_passthrough_format' "
                  "or 'mixed_fused_projection'")
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


def validate_runtime_contract(contract: Any) -> None:
    """Raise :class:`RuntimeContractError` unless *contract* is self-consistent."""

    root = _object(contract, "contract")
    _keys(root, "contract", {
        "schema", "contract_version", "abi_features", "quant_method",
        "packing", "layout", "formats", "tensor_parallel", "expert_parallel",
        "producer_profiles",
    })
    if root["schema"] != RUNTIME_CONTRACT_SCHEMA:
        _fail("contract.schema", f"must be {RUNTIME_CONTRACT_SCHEMA!r}")
    contract_version = _positive_int(
        root["contract_version"], "contract.contract_version")
    if contract_version != 9:
        _fail("contract.contract_version", "must be 9 for this schema")

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
    for index, item in enumerate(formats):
        path = f"contract.formats[{index}]"
        fmt = _object(item, path)
        _keys(fmt, path, {
            "family", "name_pattern", "grid", "mode", "n_sub", "rungs",
            "layout_versions", "moe_layout_versions",
        })
        family = _string(fmt["family"], f"{path}.family")
        if family in family_grids:
            _fail("contract.formats", f"duplicate family {family!r}")
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
        _sorted_unique_ints(fmt["rungs"], f"{path}.rungs")
        family_layouts = set(_sorted_unique_ints(
            fmt["layout_versions"], f"{path}.layout_versions"))
        moe_layouts = set(_sorted_unique_ints(
            fmt["moe_layout_versions"], f"{path}.moe_layout_versions"))
        if not family_layouts <= layout_versions:
            _fail(path, "layout_versions contains an unsupported ABI layout")
        if not moe_layouts <= family_layouts:
            _fail(path, "moe_layout_versions must be a subset of layout_versions")

    _validate_tensor_parallel(root, family_grids)
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
