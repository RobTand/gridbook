"""Load and validate Gridbook's packaged producer/runtime contract.

This module is deliberately standard-library-only.  Exporters and compatibility
checks can inspect the installed Gridbook contract without importing torch,
vLLM, CUDA extensions, or any producing project.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping


RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v5"
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

#: The exact grouped-BMM geometry each armed unit qualifies, keyed by
#: (unit, arm).  Today only the FP8-source W8A16 BMM arm is pinned, to the
#: constants in ``gridbook/fp8_source_w8a16.py`` (``_DSV4_BMM_GROUPS``,
#: ``_DSV4_BMM_ROWS``, ``_DSV4_BMM_K``, enforced alongside
#: ``_DSV4_RELEASE_TP``).  An arm without an entry here must omit
#: ``requires_geometry`` entirely.
_TP_REQUIRED_GEOMETRY: dict[tuple[str, str], dict[str, int]] = {
    ("fp8_e4m3_ue8m0_block128", "bmm"): {
        "bmm_groups": 8,
        "rows_per_group": 1024,
        "k": 4096,
    },
}


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
                              families: set[str]) -> None:
    """Validate the tensor-parallel capability table.

    The table is ATTESTATION, not aspiration: every row must restate what a
    refusal site in this package actually enforces, and the closed-world
    reading makes absence a refusal.  A consumer serving unit *U* at tensor-
    parallel size *t* looks up exactly one row named ``U`` (and, for an armed
    unit, the row for the arm it will execute); no match, or ``t`` above the
    matched claim, or a pinned geometry that does not match exactly is a
    REFUSAL.  There is no default, wildcard, or inheritance anywhere in this
    section.
    """

    tp = _object(root["tensor_parallel"], "contract.tensor_parallel")
    _keys(tp, "contract.tensor_parallel",
          {"axis", "semantics", "max_world_size", "units"})
    if tp["axis"] != _TP_AXIS:
        _fail("contract.tensor_parallel.axis", f"must be {_TP_AXIS!r}")
    if tp["semantics"] != "closed_world":
        _fail("contract.tensor_parallel.semantics",
              "must be 'closed_world': an unclaimed unit is refused, never "
              "permitted by default")
    root_cap = _positive_int(
        tp["max_world_size"], "contract.tensor_parallel.max_world_size")
    # Pinned to the enforced fact: every dispatch path through
    # PrismaQuantConfig.get_quant_method refuses a live world size above 1,
    # so a wider number here would be a claim no code stands behind.  Widening
    # is a schema-version event, made together with the lifting commit.
    if root_cap != 1:
        _fail("contract.tensor_parallel.max_world_size",
              "must be 1; no dispatch path in this build enforces more")

    units = tp["units"]
    if not isinstance(units, list) or not units:
        _fail("contract.tensor_parallel.units",
              "must be a non-empty JSON array")
    seen: set[str] = set()
    cb_units: set[str] = set()
    passthrough_units: set[str] = set()
    for index, item in enumerate(units):
        path = f"contract.tensor_parallel.units[{index}]"
        row = _object(item, path)
        expected = {"unit", "kind", "max_world_size"}
        if "arms" in row:
            expected = expected | {"arms"}
        _keys(row, path, expected)
        unit_id = _string(row["unit"], f"{path}.unit")
        kind = _string(row["kind"], f"{path}.kind")
        if kind == "cb_format_family":
            cb_units.add(unit_id)
        elif kind == "source_passthrough_format":
            passthrough_units.add(unit_id)
        else:
            _fail(f"{path}.kind",
                  "must be 'cb_format_family' or 'source_passthrough_format'")
        if unit_id in seen:
            _fail("contract.tensor_parallel.units",
                  f"duplicate unit {unit_id!r}")
        seen.add(unit_id)
        cap = _positive_int(row["max_world_size"], f"{path}.max_world_size")
        if cap != 1:
            _fail(f"{path}.max_world_size",
                  "must be 1; no enforcement site in this build allows more")
        if cap > root_cap:
            _fail(f"{path}.max_world_size",
                  "exceeds contract.tensor_parallel.max_world_size")

        required_arms = _TP_ARMED_UNITS.get(unit_id)
        if "arms" not in row:
            if required_arms is not None:
                _fail(path,
                      f"unit {unit_id!r} has per-arm TP refusal sites; it "
                      f"must declare arms {sorted(required_arms)}")
            continue
        if required_arms is None:
            _fail(f"{path}.arms",
                  f"unit {unit_id!r} has one flat TP refusal site; publish "
                  "one flat claim without 'arms'")
        arms = row["arms"]
        if not isinstance(arms, list) or not arms:
            _fail(f"{path}.arms", "must be a non-empty JSON array")
        arm_names: list[str] = []
        for arm_index, arm_item in enumerate(arms):
            arm_path = f"{path}.arms[{arm_index}]"
            arm = _object(arm_item, arm_path)
            arm_expected = {"arm", "max_world_size"}
            if "requires_geometry" in arm:
                arm_expected = arm_expected | {"requires_geometry"}
            _keys(arm, arm_path, arm_expected)
            arm_name = _string(arm["arm"], f"{arm_path}.arm")
            if arm_name not in required_arms:
                _fail(f"{arm_path}.arm",
                      f"must be one of {sorted(required_arms)}")
            if arm_name in arm_names:
                _fail(f"{path}.arms", f"duplicate arm {arm_name!r}")
            arm_names.append(arm_name)
            arm_cap = _positive_int(arm["max_world_size"],
                                    f"{arm_path}.max_world_size")
            if arm_cap != 1:
                _fail(f"{arm_path}.max_world_size",
                      "must be 1; no enforcement site in this build allows "
                      "more")
            if arm_cap > cap:
                _fail(f"{arm_path}.max_world_size", "exceeds the unit claim")
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
        missing = sorted(set(required_arms) - set(arm_names))
        if missing:
            _fail(f"{path}.arms", f"missing arm claim(s): {missing}")

    if cb_units != families:
        _fail("contract.tensor_parallel.units",
              f"every CB format family must carry exactly one "
              f"'cb_format_family' claim; missing "
              f"{sorted(families - cb_units)}, unknown "
              f"{sorted(cb_units - families)}")
    if passthrough_units != set(_PASSTHROUGH_TP_UNITS):
        _fail("contract.tensor_parallel.units",
              f"source-passthrough claims must equal {sorted(_PASSTHROUGH_TP_UNITS)}; "
              f"missing "
              f"{sorted(set(_PASSTHROUGH_TP_UNITS) - passthrough_units)}, "
              f"unknown {sorted(passthrough_units - set(_PASSTHROUGH_TP_UNITS))}")


def validate_runtime_contract(contract: Any) -> None:
    """Raise :class:`RuntimeContractError` unless *contract* is self-consistent."""

    root = _object(contract, "contract")
    _keys(root, "contract", {
        "schema", "contract_version", "abi_features", "quant_method",
        "packing", "layout", "formats", "tensor_parallel", "producer_profiles",
    })
    if root["schema"] != RUNTIME_CONTRACT_SCHEMA:
        _fail("contract.schema", f"must be {RUNTIME_CONTRACT_SCHEMA!r}")
    contract_version = _positive_int(
        root["contract_version"], "contract.contract_version")
    if contract_version != 5:
        _fail("contract.contract_version", "must be 5 for this schema")

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
    families: set[str] = set()
    for index, item in enumerate(formats):
        path = f"contract.formats[{index}]"
        fmt = _object(item, path)
        _keys(fmt, path, {
            "family", "name_pattern", "grid", "mode", "n_sub", "rungs",
            "layout_versions", "moe_layout_versions",
        })
        family = _string(fmt["family"], f"{path}.family")
        if family in families:
            _fail("contract.formats", f"duplicate family {family!r}")
        families.add(family)
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
            ("fp4", "signed"): 1,
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

    _validate_tensor_parallel(root, families)

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
