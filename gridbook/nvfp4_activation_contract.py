"""Runtime consumer for PrismaQuant's static NVFP4 W4A4 contract.

The producer owns calibration and serialization.  Gridbook deliberately owns
only the strict reader: schema validation, the canonical payload digest, and
small helpers used by both dense and MoE dispatch.  Keeping these rules here
prevents the two serving paths from growing independent interpretations of the
same artifact fields.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import re
import struct
from typing import Any

import torch


CONTRACT_KEY = "nvfp4_w4a4"
CONTRACT_SCHEMA = "prismaquant.nvfp4_w4a4_activation.v1"
# Record schema declared by a producer that also attests the routed-MoE stage
# section (ROADMAP K0.2).  The whole-model fields are identical in both
# versions -- ``target_values_sha256`` is still framed with the v1 literal --
# so a v2 record verifies exactly what a v1 record always verified, plus the
# per-stage attestation.  A reader that does not know v2 fails closed rather
# than accepting a fused-MoE readiness claim it cannot check.
CONTRACT_SCHEMA_V2 = "prismaquant.nvfp4_w4a4_activation.v2"
SUPPORTED_SCHEMAS = (CONTRACT_SCHEMA, CONTRACT_SCHEMA_V2)
ROUTED_MOE_STAGE_SCHEMA = "prismaquant.nvfp4_w4a4_activation_stages.v1"
ROUTED_MOE_STAGE_KEY = "routed_moe_stages"
STAGE_W13 = "w13"
STAGE_W2 = "w2"
ROUTED_MOE_STAGES = (STAGE_W13, STAGE_W2)
EXECUTION_CONTRACT = "e2m1_group16_ue4m3_static"
GROUP_SIZE = 16
# Rowwise activation quantization owns a phase/range multiplier that is
# deliberately separate from ``codec.FP8_ELEMENT_MAX``.  The latter is also
# the serialized E4M3 ceiling used to compose FP4-CB weight scales; changing it
# to run an activation experiment would silently change the decoded weights.
# Keep the production default at the native full-range endpoint.  A dedicated
# helper gives dense and MoE one shared runtime owner and lets validation
# override only this activation choice in-process.
ROWWISE_RANGE_MULTIPLIER = 448.0
TENSOR_SUFFIX = "input_global_scale"
VALUE_DTYPE = "float32"
LEGACY_POLICY = "legacy_6_over_calibration_amax.v1"
FULL_E4M3_POLICY = (
    "full_e4m3_range_448x6_over_calibration_amax.v1"
)
MSE_GRID_POLICY = "mse_grid_calibrated.v1"
SUPPORTED_POLICIES = frozenset((
    LEGACY_POLICY,
    FULL_E4M3_POLICY,
    MSE_GRID_POLICY,
))

# Producer calibration-source vocabulary, kept identical to PrismaQuant's
# ``nvfp4_activation_contract``.  ``w13`` consumes the experts-module input and
# ``w2`` the routed intermediate, so the legal source per stage differs; a
# record that calibrates ``w2`` from the module input is exactly the defect the
# stage attestation exists to expose.
SOURCE_TARGET_CACHE = "target_activation_cache"
SOURCE_PARENT_MODULE_CACHE = "parent_module_activation_cache"
SOURCE_SUPPLEMENTAL_MODULE_INPUT = "supplemental_module_input_sample"
SOURCE_SUPPLEMENTAL_ROUTED_REPLAY = "supplemental_routed_intermediate_replay"
SOURCE_SUPPLEMENTAL_MAX_ABS = "supplemental_max_abs"
SOURCE_PACKED_EXPERT_RENDER = "packed_expert_render_max_abs"
# ===========================================================================
# K0.4 — LATEST-ROUTE DISPATCH TELEMETRY (dense AND routed).
#
# What K0.4 asks to be attestable on every fused call: the requested activation
# POLICY, the ACTUAL kernel symbol invoked, TileM, the problem SHAPE, the
# activation CONTRACT, the fallback STATE, and the exact fallback REASON.
#
# MECHANISM: the one already in the tree. The 0.4.2 dense TileM route telemetry
# is three plain integers written onto the ``layer`` object at each fused
# success, which ``scripts/validate_fused_nvfp4_ab.py``'s probe reads back
# immediately after the call it wrapped. That is deliberately tensor-free and
# sync-free — a probe must not be able to change what the model executed — and
# it is last-write-wins per layer, because "the latest route" is the question.
# This extends that surface to the full field list and to the routed lanes; it
# does NOT introduce a parallel registry, logger, or counter store, and the
# three original dense attributes keep their names and meanings so the existing
# gate and its tests are untouched.
#
# It lives in this module because ``linear.py`` and ``moe.py`` both already
# import it, so no new module edge is created — and because the activation
# CONTRACT vocabulary belongs with the contract.
# ===========================================================================

# What RAN, as opposed to what was asked for (that is ``policy``). Closed set:
# both the writer and the report validate against this one tuple, so a typo
# cannot invent a contract that no gate is checking.
ROUTE_CONTRACTS = frozenset((
    "nvfp4_static_G",
    "nvfp4_static_lsq",
    "nvfp4_rowwise",
    "fp8_per_token_dynamic",
    "fp4_group16_rtn",
    "fp32_emulated_group_qdq",
))

ROUTE_STATES = frozenset(("served", "fallback", "error"))

# The record's field names, in report order. The probe reads exactly these.
ROUTE_FIELDS = (
    "kind",                  # "dense" | "moe"
    "policy",                # requested activation policy, verbatim
    "symbol",                # the kernel entry point actually invoked
    "tile_m",                # int; 0 where the route has no tile (GEMV)
    "shape",                 # compact problem-shape key
    "contract",              # what RAN, from ROUTE_CONTRACTS
    "state",                 # from ROUTE_STATES
    "reason",                # exact fallback reason; None when served
    "tile_candidate_ctas",   # selector provenance
    "tile_sm_count",         # selector provenance
    "tile_rho",              # selector provenance: P // E (0 for dense)
    "tile_compiled",         # selector provenance: "128,256"
)


# The contract each fused-NVFP4 activation MODE actually runs, so the dense and
# routed dispatch sites cannot describe the same mode differently. The mode
# strings are the env vocabulary (``rowwise256``, ``static_lsq_midm``, ...);
# what RAN is one of three things regardless of tile or M band.
def fused_fp4_contract(*, rowwise: bool, static_lsq: bool) -> str:
    """The ``ROUTE_CONTRACTS`` member a fused NVFP4 lane serves."""
    if rowwise:
        return "nvfp4_rowwise"
    if static_lsq:
        return "nvfp4_static_lsq"
    return "nvfp4_static_G"


def bridge_contract(is_fp4: bool) -> str:
    """The contract every QUALITY route serves — bridge, sm12x, fused mid-M.

    All of them consume the exact native QDQ the decode path uses, so they
    share one contract and differ only in GEMM schedule. Recording that is the
    point: a report reader must be able to see that an opt-in schedule lane did
    NOT change the activation contract, which is exactly what distinguishes it
    from the fused-NVFP4 modes above.
    """
    return "fp4_group16_rtn" if is_fp4 else "fp8_per_token_dynamic"


def emit_route(layer, *, kind: str, policy: str, symbol: str, tile_m: int = 0,
               shape: str = "", contract: str = "", state: str = "served",
               reason=None, tile_candidate_ctas: int = 0,
               tile_sm_count: int = 0, tile_rho: int = 0,
               tile_compiled: str = "") -> None:
    """Record the latest dispatch route on ``layer``. Never raises.

    Twelve ``setattr``s of Python scalars — no tensor is touched, so this can
    sit on the hot path without a sync and cannot perturb what executed.

    TWO-PHASE USE. Write ``state="error"`` with a reason before a launch and
    rewrite ``state="served"`` after it returns; then "raised mid-launch" is
    distinguishable from "never launched", and ``symbol`` stays an honest record
    of what was INVOKED even when it threw. A gate miss writes ``symbol=""``,
    ``state="fallback"`` and the exact reason.

    Selector provenance (``tile_rho`` above all) is what makes a tile choice
    auditable offline: given rho, the candidate CTA count, the SM count and the
    compiled set, a report reader can re-derive the verdict without the GPU.
    """
    try:
        values = {
            "kind": str(kind), "policy": str(policy), "symbol": str(symbol),
            "tile_m": int(tile_m), "shape": str(shape),
            "contract": str(contract), "state": str(state),
            "reason": None if reason is None else str(reason),
            "tile_candidate_ctas": int(tile_candidate_ctas),
            "tile_sm_count": int(tile_sm_count), "tile_rho": int(tile_rho),
            "tile_compiled": str(tile_compiled),
        }
        for field in ROUTE_FIELDS:
            setattr(layer, f"_cb_route_{field}", values[field])
    except Exception:  # noqa: BLE001 — telemetry must never break a request
        pass


def read_route(layer):
    """The latest route record as a plain dict, or ``None`` if never written.

    Pure ``getattr`` over Python scalars. Returning ``None`` (rather than a
    partial dict) is what lets a consumer count a MISSING record as a probe
    error instead of silently passing a gate that never observed a route.
    """
    if getattr(layer, "_cb_route_state", None) is None:
        return None
    return {f: getattr(layer, f"_cb_route_{f}", None) for f in ROUTE_FIELDS}


SUPPORTED_CALIBRATION_SOURCES = frozenset((
    SOURCE_TARGET_CACHE,
    SOURCE_PARENT_MODULE_CACHE,
    SOURCE_SUPPLEMENTAL_MODULE_INPUT,
    SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
    SOURCE_SUPPLEMENTAL_MAX_ABS,
    SOURCE_PACKED_EXPERT_RENDER,
))
STAGE_CALIBRATION_SOURCES = {
    STAGE_W13: frozenset((
        SOURCE_TARGET_CACHE,
        SOURCE_PARENT_MODULE_CACHE,
        SOURCE_SUPPLEMENTAL_MODULE_INPUT,
        SOURCE_SUPPLEMENTAL_MAX_ABS,
        SOURCE_PACKED_EXPERT_RENDER,
    )),
    STAGE_W2: frozenset((
        SOURCE_TARGET_CACHE,
        SOURCE_SUPPLEMENTAL_ROUTED_REPLAY,
        SOURCE_SUPPLEMENTAL_MAX_ABS,
        SOURCE_PACKED_EXPERT_RENDER,
    )),
}

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_STAGE_ENTRY_FIELDS = (
    "stage",
    "target",
    "input_global_scale_policy",
    "calibration_source",
    "stage_values_sha256",
)


def rowwise_range_multiplier() -> float:
    """Return the runtime-only UE4M3 phase/range multiplier.

    This value is not an artifact field and cannot modify the producer-owned
    static activation contract.  Validation may temporarily replace the
    module constant before serving starts; malformed values fail before the
    CUDA binding is entered.
    """

    value = float(ROWWISE_RANGE_MULTIPLIER)
    if not math.isfinite(value) or not 0.0 < value <= 448.0:
        raise ValueError(
            "NVFP4 rowwise range multiplier must be finite and in (0,448], "
            f"got {value!r}"
        )
    return value


def parse_contract(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and return the one supported top-level contract record.

    Absence is the backwards-compatible legacy state.  Once the key is
    declared every field is mandatory and exact: a future schema must not be
    silently interpreted as this one.
    """

    records = config.get("execution_contracts")
    if records is None:
        return None
    if not isinstance(records, Mapping):
        raise ValueError("execution_contracts must be an object")
    if CONTRACT_KEY not in records:
        return None
    raw = records[CONTRACT_KEY]
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY} must be an object"
        )
    record = dict(raw)
    schema = record.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.schema={schema!r}; expected "
            f"one of {list(SUPPORTED_SCHEMAS)}"
        )
    exact = {
        "contract": EXECUTION_CONTRACT,
        "group_size": GROUP_SIZE,
        "tensor_suffix": TENSOR_SUFFIX,
        "value_dtype": VALUE_DTYPE,
    }
    for field, expected in exact.items():
        value = record.get(field)
        # bool is an int subclass; accepting True as group_size=1 would be a
        # particularly unhelpful malformed-metadata failure mode.
        if field == "group_size" and isinstance(value, bool):
            value = None
        if value != expected:
            raise ValueError(
                f"execution_contracts.{CONTRACT_KEY}.{field}={value!r}; "
                f"expected {expected!r}"
            )
    policy = record.get("input_global_scale_policy")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}."
            f"input_global_scale_policy={policy!r}; expected one of "
            f"{sorted(SUPPORTED_POLICIES)}"
        )
    count = record.get("target_count")
    if (isinstance(count, bool) or not isinstance(count, int)
            or count <= 0):
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.target_count must be a "
            f"positive integer, got {count!r}"
        )
    target_names = record.get("target_names")
    if (not isinstance(target_names, list)
            or any(not isinstance(name, str) or not name
                   for name in target_names)):
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.target_names must be a "
            "list of nonempty physical target strings"
        )
    if target_names != sorted(set(target_names)):
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.target_names must be sorted "
            "and duplicate-free"
        )
    if len(target_names) != count:
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.target_names has "
            f"{len(target_names)} entries, but target_count={count}"
        )
    digest = record.get("target_values_sha256")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY}.target_values_sha256 must "
            "be 64 lowercase hexadecimal characters"
        )
    # The stage section and the record schema are one claim: a v1 record must
    # not smuggle stages past an older reader, and a v2 record without a
    # well-formed section is a fused-readiness claim with nothing behind it.
    has_section = ROUTED_MOE_STAGE_KEY in record
    if schema == CONTRACT_SCHEMA and has_section:
        raise ValueError(
            f"execution_contracts.{CONTRACT_KEY} declares schema "
            f"{CONTRACT_SCHEMA} but carries {ROUTED_MOE_STAGE_KEY}; the stage "
            f"attestation requires schema {CONTRACT_SCHEMA_V2}"
        )
    if schema == CONTRACT_SCHEMA_V2:
        if not has_section:
            raise ValueError(
                f"execution_contracts.{CONTRACT_KEY} declares schema "
                f"{CONTRACT_SCHEMA_V2} but carries no {ROUTED_MOE_STAGE_KEY} "
                "section"
            )
        parse_routed_moe_stages(record)
    return record


def parse_routed_moe_stages(
    record: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, Any]]] | None:
    """Validate and return the per-module ``w13``/``w2`` stage attestation.

    ``None`` is the dense-only state: a dense artifact carries no stage
    section and stays valid.  Once the section exists every packed FusedMoE
    module must attest BOTH stages, every stage must name a physical target
    declared by the whole-model mapping, and every calibration source must be
    legal for its stage.
    """

    section = record.get(ROUTED_MOE_STAGE_KEY)
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError(f"{ROUTED_MOE_STAGE_KEY} must be an object")
    if section.get("schema") != ROUTED_MOE_STAGE_SCHEMA:
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.schema={section.get('schema')!r}; "
            f"expected {ROUTED_MOE_STAGE_SCHEMA!r}"
        )
    if list(section.get("stages") or ()) != list(ROUTED_MOE_STAGES):
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.stages must be {list(ROUTED_MOE_STAGES)}"
        )
    modules = section.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.modules must be a nonempty object"
        )
    names = section.get("module_names")
    if names != sorted(str(name) for name in modules):
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.module_names must be the sorted module "
            "set"
        )
    count = section.get("module_count")
    if isinstance(count, bool) or count != len(modules):
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.module_count={count!r} does not match "
            f"{len(modules)} attested modules"
        )
    declared_targets = set(record.get("target_names") or ())
    policy = record.get("input_global_scale_policy")
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_module, raw_entries in modules.items():
        module = str(raw_module)
        if not isinstance(raw_entries, Mapping):
            raise ValueError(f"{ROUTED_MOE_STAGE_KEY}.{module} must be object")
        missing = [
            stage for stage in ROUTED_MOE_STAGES if stage not in raw_entries
        ]
        if missing or len(raw_entries) != len(ROUTED_MOE_STAGES):
            raise ValueError(
                f"routed-MoE module {module!r} attests "
                f"{sorted(str(key) for key in raw_entries)}; both "
                f"{list(ROUTED_MOE_STAGES)} stages are required"
            )
        entries: dict[str, dict[str, Any]] = {}
        for stage in ROUTED_MOE_STAGES:
            entry = raw_entries[stage]
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} must be an "
                    "object"
                )
            if sorted(str(key) for key in entry) != sorted(
                _STAGE_ENTRY_FIELDS
            ):
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} must declare "
                    f"exactly {sorted(_STAGE_ENTRY_FIELDS)}"
                )
            if entry.get("stage") != stage:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} is labelled "
                    f"{entry.get('stage')!r}"
                )
            target = entry.get("target")
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} has no "
                    "physical target"
                )
            if target.rsplit(".", 1)[0] != module:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} names "
                    f"unrelated target {target!r}"
                )
            if declared_targets and target not in declared_targets:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} target "
                    f"{target!r} is not in the contract's target_names"
                )
            if entry.get("input_global_scale_policy") != policy:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} declares "
                    f"policy {entry.get('input_global_scale_policy')!r}, but "
                    f"the contract declares {policy!r}"
                )
            source = entry.get("calibration_source")
            if source not in STAGE_CALIBRATION_SOURCES[stage]:
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} was "
                    f"calibrated from {source!r}, which is not a legal input "
                    f"for that stage"
                )
            stage_digest = entry.get("stage_values_sha256")
            if (not isinstance(stage_digest, str)
                    or _DIGEST_RE.fullmatch(stage_digest) is None):
                raise ValueError(
                    f"routed-MoE module {module!r} stage {stage} "
                    "stage_values_sha256 must be 64 lowercase hexadecimal "
                    "characters"
                )
            entries[stage] = dict(entry)
        parsed[module] = entries
    expected = routed_moe_stages_sha256(parsed)
    declared = section.get("stages_sha256")
    if declared != expected:
        raise ValueError(
            f"{ROUTED_MOE_STAGE_KEY}.stages_sha256 mismatch: declared "
            f"{declared!r}, recomputed {expected}"
        )
    return parsed


def stage_values_sha256(
    *,
    stage: str,
    target: str,
    policy: str,
    calibration_source: str,
    value: Any,
) -> str:
    """Producer-canonical digest over one routed-MoE stage's attested fields."""

    if stage not in ROUTED_MOE_STAGES:
        raise ValueError(f"unknown routed-MoE stage {stage!r}")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported NVFP4 scale policy {policy!r}")
    if calibration_source not in SUPPORTED_CALIBRATION_SOURCES:
        raise ValueError(
            f"unsupported NVFP4 calibration source {calibration_source!r}"
        )
    digest = hashlib.sha256()
    for field in (
        ROUTED_MOE_STAGE_SCHEMA,
        policy,
        str(stage),
        str(target),
        str(calibration_source),
    ):
        encoded = field.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(
        struct.pack("<f", scale_f32(value, target=str(target)))
    )
    return digest.hexdigest()


def routed_moe_stages_sha256(
    modules: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> str:
    """Producer-canonical digest over the whole stage section."""

    digest = hashlib.sha256()
    encoded = ROUTED_MOE_STAGE_SCHEMA.encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    for module in sorted(modules):
        encoded = str(module).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        entries = modules[module]
        for stage in ROUTED_MOE_STAGES:
            if stage not in entries:
                raise ValueError(
                    f"routed-MoE module {module!r} has no {stage} stage"
                )
            encoded = str(stage).encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            digest.update(
                bytes.fromhex(str(entries[stage]["stage_values_sha256"]))
            )
    return digest.hexdigest()


def verify_routed_moe_stages(
    record: Mapping[str, Any], scales: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a stage attestation against the artifact's serialized scalars.

    Returns a machine-readable result rather than raising, because the caller
    is a validation harness whose job is to LABEL an artifact, not to load it.
    The three failure classes are kept apart on purpose: a missing stage means
    the producer never calibrated it, while a digest mismatch means the record
    and the serialized tensors disagree.
    """

    normalized = {
        str(target): scale_f32(value, target=str(target))
        for target, value in scales.items()
    }
    try:
        parsed = parse_routed_moe_stages(record)
    except ValueError as exc:
        return {
            "attested": True,
            "verdict": "malformed_stage_attestation",
            "detail": str(exc),
            "modules": [],
            "failing_module": None,
            "failing_stage": None,
        }
    if parsed is None:
        return {
            "attested": False,
            "verdict": "not_attested",
            "detail": (
                "the contract declares no routed-MoE stage section; a "
                "fused-MoE comparison against it is fallback telemetry, not "
                "evidence"
            ),
            "modules": [],
            "failing_module": None,
            "failing_stage": None,
        }
    policy = str(record["input_global_scale_policy"])
    for module in sorted(parsed):
        entries = parsed[module]
        for stage in ROUTED_MOE_STAGES:
            entry = entries[stage]
            target = str(entry["target"])
            if target not in normalized:
                return {
                    "attested": True,
                    "verdict": "missing_stages",
                    "detail": (
                        f"routed-MoE module {module!r} attests stage {stage} "
                        f"on {target!r}, but the artifact serializes no "
                        f"{target}.{TENSOR_SUFFIX} tensor"
                    ),
                    "modules": sorted(parsed),
                    "failing_module": module,
                    "failing_stage": stage,
                }
            actual = stage_values_sha256(
                stage=stage,
                target=target,
                policy=policy,
                calibration_source=str(entry["calibration_source"]),
                value=normalized[target],
            )
            if actual != str(entry["stage_values_sha256"]):
                return {
                    "attested": True,
                    "verdict": "digest_mismatch",
                    "detail": (
                        f"routed-MoE module {module!r} stage {stage} declares "
                        f"{entry['stage_values_sha256']}, but the serialized "
                        f"{target}.{TENSOR_SUFFIX} digests to {actual}"
                    ),
                    "modules": sorted(parsed),
                    "failing_module": module,
                    "failing_stage": stage,
                }
    return {
        "attested": True,
        "verdict": "attested_and_verified",
        "detail": (
            f"{len(parsed)} packed FusedMoE modules attest both "
            f"{list(ROUTED_MOE_STAGES)} stages, and every stage digest "
            "matches the serialized scalar"
        ),
        "modules": sorted(parsed),
        "failing_module": None,
        "failing_stage": None,
    }


def scale_f32(value: Any, *, target: str) -> float:
    """Return one exact finite positive F32 scalar or fail closed."""

    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or tuple(value.shape) != (1,):
            raise ValueError(
                f"{target}.{TENSOR_SUFFIX} must be float32 shape [1], got "
                f"dtype={value.dtype}, shape={tuple(value.shape)}"
            )
        value = value.detach().cpu().item()
    try:
        rounded = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{target}.{TENSOR_SUFFIX} is not an F32 scalar: {value!r}"
        ) from exc
    if not math.isfinite(rounded) or rounded <= 0.0:
        raise ValueError(
            f"{target}.{TENSOR_SUFFIX} must be finite and > 0, got "
            f"{rounded!r}"
        )
    return rounded


def target_values_sha256(
    scales: Mapping[str, Any], *, policy: str
) -> str:
    """Producer-canonical digest over physical names and serialized F32s."""

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported NVFP4 scale policy {policy!r}")
    digest = hashlib.sha256()
    for field in (CONTRACT_SCHEMA, policy):
        encoded = field.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    for target in sorted(scales):
        name = str(target)
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<f", scale_f32(scales[target],
                                                   target=name)))
    return digest.hexdigest()


def validate_payload(
    record: Mapping[str, Any], scales: Mapping[str, Any]
) -> dict[str, float]:
    """Validate complete artifact membership/count/digest and normalize F32s."""

    normalized = {
        str(target): scale_f32(value, target=str(target))
        for target, value in scales.items()
    }
    count = int(record["target_count"])
    if len(normalized) != count:
        raise ValueError(
            f"NVFP4 activation contract declares target_count={count}, but "
            f"the artifact contains {len(normalized)} .{TENSOR_SUFFIX} tensors"
        )
    declared_names = set(record["target_names"])
    actual_names = set(normalized)
    if actual_names != declared_names:
        missing = sorted(declared_names - actual_names)
        extra = sorted(actual_names - declared_names)
        raise ValueError(
            "NVFP4 activation contract physical target set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    actual = target_values_sha256(
        normalized, policy=str(record["input_global_scale_policy"])
    )
    expected = str(record["target_values_sha256"])
    if actual != expected:
        raise ValueError(
            "NVFP4 activation contract target_values_sha256 mismatch: "
            f"declared {expected}, artifact {actual}"
        )
    return normalized


def require_identical_loaded_scales(
    values: torch.Tensor, *, prefix: str, expected: list[float]
) -> torch.Tensor:
    """Validate loaded merged slots and return the single device scalar.

    vLLM gives a fused Linear one slot per logical shard.  Native activation
    quantization happens before the projection split, so every slot and every
    producer physical target participating in that module must carry the exact
    same IEEE-F32 bits.  Numeric equality is insufficient (notably for NaNs and
    signed zero), hence the explicit byte comparison.
    """

    data = values.detach()
    if data.dtype != torch.float32 or data.ndim != 1 or data.numel() == 0:
        raise ValueError(
            f"{prefix}.{TENSOR_SUFFIX} parameter must be nonempty 1-D "
            f"float32, got dtype={data.dtype}, shape={tuple(data.shape)}"
        )
    host = data.to(device="cpu", dtype=torch.float32).contiguous()
    loaded = [scale_f32(host[i:i + 1], target=prefix)
              for i in range(host.numel())]
    expected_f32 = [scale_f32(value, target=prefix) for value in expected]
    if not expected_f32:
        raise ValueError(f"{prefix}: activation contract has no physical target")
    bits = [struct.pack("<f", value) for value in loaded + expected_f32]
    if any(value != bits[0] for value in bits[1:]):
        raise ValueError(
            f"{prefix}: fused logical shards have non-identical "
            f"{TENSOR_SUFFIX} values"
        )
    # vLLM's native scaled_fp4_quant runtime ABI consumes a scalar tensor.  The
    # checkpoint/PerTensorScaleParameter ABI is [logical_shards], so retain the
    # registered parameter but expose one 0-D view to the fused call sites.
    return data.reshape(-1)[0].reshape(())


def reciprocal_vector(layer, *, which: str, scale: torch.Tensor,
                      rows: int) -> torch.Tensor:
    """Return a small bounded cache of contiguous per-row output scales."""

    if rows <= 0:
        raise ValueError(f"NVFP4 activation row count must be positive, got {rows}")
    cache = getattr(layer, "_cb_fp4_reciprocal_cache", None)
    if cache is None:
        cache = {}
        layer._cb_fp4_reciprocal_cache = cache
    device = scale.device
    key = (which, int(rows), device.type, device.index)
    cached = cache.get(key)
    if cached is not None:
        return cached
    value = torch.reciprocal(scale.to(torch.float32)).expand(rows).contiguous()
    # Prompt lengths are unbounded.  This cache is an optimization, not a
    # reason to retain one tensor for every request shape for process lifetime.
    if len(cache) >= 8:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


__all__ = [
    "CONTRACT_KEY",
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_V2",
    "EXECUTION_CONTRACT",
    "ROUTE_CONTRACTS",
    "ROUTE_FIELDS",
    "ROUTE_STATES",
    "bridge_contract",
    "emit_route",
    "fused_fp4_contract",
    "read_route",
    "FULL_E4M3_POLICY",
    "GROUP_SIZE",
    "LEGACY_POLICY",
    "MSE_GRID_POLICY",
    "ROUTED_MOE_STAGES",
    "ROUTED_MOE_STAGE_KEY",
    "ROUTED_MOE_STAGE_SCHEMA",
    "ROWWISE_RANGE_MULTIPLIER",
    "SOURCE_PACKED_EXPERT_RENDER",
    "SOURCE_PARENT_MODULE_CACHE",
    "SOURCE_SUPPLEMENTAL_MAX_ABS",
    "SOURCE_SUPPLEMENTAL_MODULE_INPUT",
    "SOURCE_SUPPLEMENTAL_ROUTED_REPLAY",
    "SOURCE_TARGET_CACHE",
    "STAGE_CALIBRATION_SOURCES",
    "STAGE_W2",
    "STAGE_W13",
    "SUPPORTED_CALIBRATION_SOURCES",
    "SUPPORTED_POLICIES",
    "SUPPORTED_SCHEMAS",
    "TENSOR_SUFFIX",
    "VALUE_DTYPE",
    "parse_contract",
    "parse_routed_moe_stages",
    "reciprocal_vector",
    "routed_moe_stages_sha256",
    "rowwise_range_multiplier",
    "require_identical_loaded_scales",
    "scale_f32",
    "stage_values_sha256",
    "target_values_sha256",
    "validate_payload",
    "verify_routed_moe_stages",
]
