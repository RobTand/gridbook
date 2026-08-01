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
EXECUTION_CONTRACT = "e2m1_group16_ue4m3_static"
GROUP_SIZE = 16
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

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


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
    exact = {
        "schema": CONTRACT_SCHEMA,
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
    return record


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
    "EXECUTION_CONTRACT",
    "FULL_E4M3_POLICY",
    "GROUP_SIZE",
    "LEGACY_POLICY",
    "MSE_GRID_POLICY",
    "SUPPORTED_POLICIES",
    "TENSOR_SUFFIX",
    "VALUE_DTYPE",
    "parse_contract",
    "reciprocal_vector",
    "require_identical_loaded_scales",
    "scale_f32",
    "target_values_sha256",
    "validate_payload",
]
