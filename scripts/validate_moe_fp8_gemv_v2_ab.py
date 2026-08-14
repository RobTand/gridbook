#!/usr/bin/env python3
"""Same-engine exact-quality A/B for routed FP8-CB whole-row GEMV.

This is the full-model invariance gate for
``PRISMAQUANT_CB_FP8_GEMV_V2=1`` on the exact dsv4flash0731 artifact.  One
eager, in-process vLLM engine is loaded with both routed FP8 implementations
resident.  Requests alternate between:

* ``baseline``: ``cb_moe_gemv_fp8``; and
* ``fused``: ``cb_moe_gemv_fp8_v2`` (the whole-row sibling).

The process selector stays pinned to ``1``.  A request-scoped controller
changes only ``_cb_use_fp8_v2_w13`` and ``_cb_use_fp8_v2_w2`` on the exact
eight routed-FP8 layers, then restores their load-resolved values.  The 35
FP4 layers and all other selectors remain fixed.  Eager execution is
mandatory because these Python booleans are trace-time constants under CUDA
graph capture.

The workload and scoring implementation are shared with
``validate_moe_gemv_v2_ab.py``: two counterbalanced repeats of the canonical
eight sealed 16-token WikiText prefixes, with every vocabulary logprob
cardinality-checked and compacted one request at a time.  The actual-artifact
operator gate requires BF16 storage-bit equality, so this model-level gate
requires exact row/score digests both across repeats and across arms.  KL,
target NLL and PPL deltas are retained as diagnostics, not substitutes for
exact equality.

The harness additionally fingerprints router ids and weights at every loaded
MoE layer, proves exact projection/operator cardinalities, refuses any mixed
or fallback route, records a closed source/binary manifest, and requires every
request, selector mutation, dispatch observation and engine to stay in the
same PID.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _load_fp4_gemv_helpers() -> Any:
    path = Path(__file__).with_name("validate_moe_gemv_v2_ab.py")
    module_name = "_gridbook_validate_moe_gemv_v2_for_fp8_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shared GEMV A/B helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = _load_fp4_gemv_helpers()
pb = base.pb
v5 = base.v5
validation_common = base.validation_common

SCHEMA = "gridbook.moe-fp8-gemv-v2-ab.v1"
SOURCE_MANIFEST_SCHEMA = "gridbook.routed-fp8-v2-source-manifest.v1"
DSV4_KV_CACHE_DTYPE = base.DSV4_KV_CACHE_DTYPE
ARMS = v5.ARMS
ARM_LABELS = {
    "baseline": "inherited_routed_fp8_cb_gemv",
    "fused": "whole_row_routed_fp8_cb_gemv_v2",
}

FP8_GEMV_ENV = "PRISMAQUANT_CB_FP8_GEMV_V2"
GEMV_ENV = base.GEMV_ENV
DECODE_CONTRACT_ENV = base.DECODE_CONTRACT_ENV
W2_SCHEDULE_ENVS = base.W2_SCHEDULE_ENVS

_PROFILE_SAMPLES = base._PROFILE_SAMPLES
_PROFILE_SEQLEN = base._PROFILE_SEQLEN
_QUALITY_REPEATS = base._QUALITY_REPEATS
_EXPECTED_TOPK = base._EXPECTED_TOPK
_EXPECTED_FP4_LAYER_IDS = base._EXPECTED_FP4_LAYER_IDS
_EXPECTED_FP8_LAYER_IDS = base._EXPECTED_FP8_LAYER_IDS
_EXPECTED_K16_LAYER_IDS = base._EXPECTED_K16_LAYER_IDS
_EXPECTED_ARTIFACT = base._EXPECTED_ARTIFACT
_EXPECTED_PREFIX_TENSOR_SHA256 = base._EXPECTED_PREFIX_TENSOR_SHA256
_EXPECTED_PREFIX_JSON_SHA256 = base._EXPECTED_PREFIX_JSON_SHA256
_LAYER_PREFIX_RE = base._LAYER_PREFIX_RE

_REQUIRED_MANIFEST_LABELS = frozenset({
    "gridbook/__init__.py",
    "gridbook/config.py",
    "gridbook/linear.py",
    "gridbook/moe.py",
    "gridbook/moe_gemv_select.py",
    "gridbook/moe_routing.py",
    "gridbook/moe_toplevel_loader.py",
    "gridbook/ops.py",
    "gridbook/cuda_ext.py",
    "gridbook/codec.py",
    "gridbook/cb_digest.py",
    "gridbook/cb_fill_guard.py",
    "gridbook/native_cutlass.py",
    "gridbook/plugin.py",
    "gridbook/_fused_nvfp4_validation.py",
    "gridbook/csrc/cb_gemv.cu",
    "gridbook/csrc/cb_gemv_v2.cu",
    "scripts/validate_fused_nvfp4_ab.py",
    "scripts/validate_moe_persistent_b_ab.py",
    "scripts/validate_moe_gemv_v2_ab.py",
    "scripts/validate_moe_fp8_gemv_v2_ab.py",
    "extension/main",
    "extension/fp4_v2",
})


def _required_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), **v5._required_file_record(resolved)}


def _build_source_manifest(
    runtime: MutableMapping[str, Any], *, main_extension: Any, fp4_extension: Any
) -> dict[str, Any]:
    package_root = Path(runtime["gridbook"]["package_root"]).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    expected_package_root = (repository_root / "gridbook").resolve()
    if package_root != expected_package_root:
        raise RuntimeError(
            "mixed harness/runtime source roots: imported Gridbook is "
            f"{package_root}, harness expects {expected_package_root}"
        )
    paths = {
        "gridbook/__init__.py": package_root / "__init__.py",
        "gridbook/config.py": package_root / "config.py",
        "gridbook/linear.py": package_root / "linear.py",
        "gridbook/moe.py": package_root / "moe.py",
        "gridbook/moe_gemv_select.py": package_root / "moe_gemv_select.py",
        "gridbook/moe_routing.py": package_root / "moe_routing.py",
        "gridbook/moe_toplevel_loader.py": (
            package_root / "moe_toplevel_loader.py"
        ),
        "gridbook/ops.py": package_root / "ops.py",
        "gridbook/cuda_ext.py": package_root / "cuda_ext.py",
        "gridbook/codec.py": package_root / "codec.py",
        "gridbook/cb_digest.py": package_root / "cb_digest.py",
        "gridbook/cb_fill_guard.py": package_root / "cb_fill_guard.py",
        "gridbook/native_cutlass.py": package_root / "native_cutlass.py",
        "gridbook/plugin.py": package_root / "plugin.py",
        "gridbook/_fused_nvfp4_validation.py": (
            package_root / "_fused_nvfp4_validation.py"
        ),
        "gridbook/csrc/cb_gemv.cu": package_root / "csrc" / "cb_gemv.cu",
        "gridbook/csrc/cb_gemv_v2.cu": (
            package_root / "csrc" / "cb_gemv_v2.cu"
        ),
        "scripts/validate_fused_nvfp4_ab.py": Path(v5.__file__),
        "scripts/validate_moe_persistent_b_ab.py": Path(pb.__file__),
        "scripts/validate_moe_gemv_v2_ab.py": Path(base.__file__),
        "scripts/validate_moe_fp8_gemv_v2_ab.py": Path(__file__),
        "extension/main": Path(main_extension.__file__),
        "extension/fp4_v2": Path(fp4_extension.__file__),
    }
    labels = frozenset(paths)
    if labels != _REQUIRED_MANIFEST_LABELS:
        raise RuntimeError(
            "routed-FP8 source manifest declaration drifted: "
            f"observed={sorted(labels)} expected={sorted(_REQUIRED_MANIFEST_LABELS)}"
        )
    records = {label: _required_record(path) for label, path in paths.items()}
    source_roots = {
        "imported_gridbook_package": str(package_root),
        "harness_repository": str(repository_root),
    }
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "required_labels": sorted(_REQUIRED_MANIFEST_LABELS),
        "source_roots": source_roots,
        "files": records,
    }
    runtime["harness"]["source_manifest"] = manifest
    for label in (
        "moe_gemv_select.py",
        "native_cutlass.py",
        "cb_gemv.cu",
        "cb_gemv_v2.cu",
    ):
        manifest_label = (
            f"gridbook/csrc/{label}" if label.endswith(".cu")
            else f"gridbook/{label}"
        )
        runtime["source_files"][label] = records[manifest_label]
        runtime["source_sha256"][label] = records[manifest_label]["sha256"]
    return manifest


def _source_manifest_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    records = manifest.get("files")
    labels = set(records) if isinstance(records, Mapping) else set()
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        violations.append({"reason": "schema mismatch"})
    if labels != _REQUIRED_MANIFEST_LABELS:
        violations.append({
            "reason": "required label set mismatch",
            "missing": sorted(_REQUIRED_MANIFEST_LABELS - labels),
            "unexpected": sorted(labels - _REQUIRED_MANIFEST_LABELS),
        })
    if isinstance(records, Mapping):
        for label in sorted(labels & _REQUIRED_MANIFEST_LABELS):
            record = records[label]
            path_raw = record.get("path") if isinstance(record, Mapping) else None
            if not isinstance(path_raw, str):
                violations.append({"label": label, "reason": "missing path"})
                continue
            try:
                current = _required_record(Path(path_raw))
            except Exception as exc:  # noqa: BLE001 - report exact source failure
                violations.append({
                    "label": label,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            if (
                record.get("sha256") != current["sha256"]
                or record.get("bytes") != current["bytes"]
                or str(Path(path_raw).resolve()) != current["path"]
            ):
                violations.append({
                    "label": label,
                    "reason": "file changed after manifest capture",
                    "captured": dict(record),
                    "current": current,
                })
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "required_labels": sorted(_REQUIRED_MANIFEST_LABELS),
        "observed_labels": sorted(labels),
        "violations": violations,
        "pass": not violations,
    }


@dataclass(frozen=True)
class _LayerBinding:
    layer_id: int
    layer_index: int
    prefix: str
    method: Any
    layer: Any
    original_fp4_w13: bool
    original_fp4_w2: bool
    original_fp8_w13: bool
    original_fp8_w2: bool
    k_bits: int
    n_sub: int
    type_size: int
    hidden_size: int
    intermediate_size: int
    role_split: bool


class FP8GemvV2ArmController:
    """Switch only the two FP8 GEMV booleans on each of eight FP8 layers."""

    def __init__(self, *, ops: Any, moe: Any) -> None:
        bindings: list[_LayerBinding] = []
        malformed_prefixes: list[str] = []
        for layer_id in sorted(ops._LAYER_REGISTRY):
            try:
                method, layer = ops._lookup_cb_layer(layer_id)
            except RuntimeError:
                continue
            if not isinstance(method, moe.PrismaQuantCBMoEMethod):
                continue
            prefix = str(method.prefix)
            match = _LAYER_PREFIX_RE.search(prefix)
            if match is None:
                malformed_prefixes.append(prefix)
                continue
            selectors = {
                "fp4_w13": getattr(layer, "_cb_use_v2_w13", None),
                "fp4_w2": getattr(layer, "_cb_use_v2_w2", None),
                "fp8_w13": getattr(layer, "_cb_use_fp8_v2_w13", None),
                "fp8_w2": getattr(layer, "_cb_use_fp8_v2_w2", None),
            }
            if not all(isinstance(value, bool) for value in selectors.values()):
                raise RuntimeError(
                    f"{prefix}: load did not resolve all four GEMV booleans"
                )
            bindings.append(_LayerBinding(
                layer_id=int(layer_id),
                layer_index=int(match.group(1)),
                prefix=prefix,
                method=method,
                layer=layer,
                original_fp4_w13=selectors["fp4_w13"],
                original_fp4_w2=selectors["fp4_w2"],
                original_fp8_w13=selectors["fp8_w13"],
                original_fp8_w2=selectors["fp8_w2"],
                k_bits=int(method.k),
                n_sub=int(method.n_sub),
                type_size=int(method.type_size),
                hidden_size=int(getattr(layer, "_cb_hidden", -1)),
                intermediate_size=int(getattr(layer, "_cb_inter", -1)),
                role_split=bool(getattr(layer, "_cb_role_split", False)),
            ))

        self.fp4 = tuple(
            binding for binding in bindings if bool(binding.method.is_fp4)
        )
        self.fp8 = tuple(
            binding for binding in bindings if not bool(binding.method.is_fp4)
        )
        self._all = tuple(bindings)
        prefixes = [binding.prefix for binding in bindings]
        fp4_ids = {binding.layer_index for binding in self.fp4}
        fp8_ids = {binding.layer_index for binding in self.fp8}
        checks = {
            "no_malformed_moe_prefix": not malformed_prefixes,
            "unique_prefixes": len(prefixes) == len(set(prefixes)),
            "unique_layer_indices": len(bindings) == len({
                binding.layer_index for binding in bindings
            }),
            "exact_total_moe_layers": len(bindings) == 43,
            "exact_fp4_layer_ids": fp4_ids == _EXPECTED_FP4_LAYER_IDS,
            "exact_fp8_layer_ids": fp8_ids == _EXPECTED_FP8_LAYER_IDS,
            "all_shapes_exact": all(
                binding.hidden_size == 4096
                and binding.intermediate_size == 2048
                for binding in bindings
            ),
            "all_fp4_formats_exact": all(
                bool(binding.method.is_v2)
                and binding.n_sub == 2
                and binding.k_bits == (
                    16 if binding.layer_index in _EXPECTED_K16_LAYER_IDS else 18
                )
                and binding.type_size == 4 * binding.k_bits + 9
                for binding in self.fp4
            ),
            "all_fp4_selectors_fixed_v2": all(
                binding.original_fp4_w13
                and binding.original_fp4_w2
                and not binding.original_fp8_w13
                and not binding.original_fp8_w2
                for binding in self.fp4
            ),
            "all_fp8_formats_exact": all(
                not bool(binding.method.is_v2)
                and binding.k_bits == 28
                and binding.n_sub == 4
                and binding.type_size == 112
                and binding.role_split
                for binding in self.fp8
            ),
            "all_fp8_selected_v2_at_load": all(
                not binding.original_fp4_w13
                and not binding.original_fp4_w2
                and binding.original_fp8_w13
                and binding.original_fp8_w2
                for binding in self.fp8
            ),
        }
        self.inventory_gate = {
            "expected_fp4_layer_ids": sorted(_EXPECTED_FP4_LAYER_IDS),
            "observed_fp4_layer_ids": sorted(fp4_ids),
            "expected_fp8_layer_ids": sorted(_EXPECTED_FP8_LAYER_IDS),
            "observed_fp8_layer_ids": sorted(fp8_ids),
            "mutable_fp8_layer_count": len(self.fp8),
            "mutable_fp8_attribute_count": len(self.fp8) * 2,
            "mutable_attributes": [
                "_cb_use_fp8_v2_w13", "_cb_use_fp8_v2_w2"
            ],
            "malformed_prefixes": malformed_prefixes,
            "fp4": [self._binding_record(binding) for binding in self.fp4],
            "fp8": [self._binding_record(binding) for binding in self.fp8],
            "checks": checks,
            "pass": all(checks.values()),
        }
        if not self.inventory_gate["pass"]:
            raise RuntimeError(
                "loaded DSV4 routed-FP8 inventory is not exact: "
                f"{self.inventory_gate}"
            )

    @staticmethod
    def _binding_record(binding: _LayerBinding) -> dict[str, Any]:
        return {
            "layer_id": binding.layer_id,
            "layer_index": binding.layer_index,
            "prefix": binding.prefix,
            "is_fp4": bool(binding.method.is_fp4),
            "is_v2_format": bool(binding.method.is_v2),
            "k_bits": binding.k_bits,
            "n_sub": binding.n_sub,
            "type_size": binding.type_size,
            "hidden_size": binding.hidden_size,
            "intermediate_size": binding.intermediate_size,
            "role_split": binding.role_split,
            "resolved_fp4_w13_v2": binding.original_fp4_w13,
            "resolved_fp4_w2_v2": binding.original_fp4_w2,
            "resolved_fp8_w13_v2": binding.original_fp8_w13,
            "resolved_fp8_w2_v2": binding.original_fp8_w2,
        }

    @property
    def prefixes(self) -> frozenset[str]:
        return frozenset(binding.prefix for binding in self._all)

    @property
    def fp4_prefixes(self) -> frozenset[str]:
        return frozenset(binding.prefix for binding in self.fp4)

    @property
    def fp8_prefixes(self) -> frozenset[str]:
        return frozenset(binding.prefix for binding in self.fp8)

    def _drift(self) -> list[dict[str, Any]]:
        violations = []
        for binding in self._all:
            expected = {
                "_cb_use_v2_w13": binding.original_fp4_w13,
                "_cb_use_v2_w2": binding.original_fp4_w2,
                "_cb_use_fp8_v2_w13": binding.original_fp8_w13,
                "_cb_use_fp8_v2_w2": binding.original_fp8_w2,
            }
            for attribute, value in expected.items():
                observed = getattr(binding.layer, attribute, None)
                if observed is not value:
                    violations.append({
                        "prefix": binding.prefix,
                        "attribute": attribute,
                        "expected": value,
                        "observed": observed,
                    })
        return violations

    @contextmanager
    def arm(self, arm: str, *, label: str) -> Iterator[dict[str, Any]]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        drift = self._drift()
        if drift:
            raise RuntimeError(
                f"GEMV selectors were not restored before {label}: {drift}"
            )
        requested = arm == "fused"
        for binding in self.fp8:
            binding.layer._cb_use_fp8_v2_w13 = requested
            binding.layer._cb_use_fp8_v2_w2 = requested
        record = {
            "pid": os.getpid(),
            "label": label,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "mutated_layer_count": len(self.fp8),
            "mutated_attribute_count": len(self.fp8) * 2,
            "expected_fp8_v2_stack_booleans": 16 if requested else 0,
            "observed_fp8_v2_stack_booleans": sum(
                int(getattr(binding.layer, "_cb_use_fp8_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_fp8_v2_w2", False))
                for binding in self.fp8
            ),
            "fp4_v2_stack_booleans": sum(
                int(getattr(binding.layer, "_cb_use_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_v2_w2", False))
                for binding in self.fp4
            ),
            "fp4_fp8_v2_stack_booleans": sum(
                int(getattr(binding.layer, "_cb_use_fp8_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_fp8_v2_w2", False))
                for binding in self.fp4
            ),
            "fp8_fp4_v2_stack_booleans": sum(
                int(getattr(binding.layer, "_cb_use_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_v2_w2", False))
                for binding in self.fp8
            ),
            "restored_after_request": False,
        }
        record["selection_pass"] = bool(
            record["mutated_layer_count"] == 8
            and record["mutated_attribute_count"] == 16
            and record["observed_fp8_v2_stack_booleans"]
            == record["expected_fp8_v2_stack_booleans"]
            and record["fp4_v2_stack_booleans"] == 70
            and record["fp4_fp8_v2_stack_booleans"] == 0
            and record["fp8_fp4_v2_stack_booleans"] == 0
        )
        if not record["selection_pass"]:
            self.restore_all()
            raise RuntimeError(f"routed-FP8 arm switch failed: {record}")
        try:
            yield record
        finally:
            self.restore_all()
            record["restored_after_request"] = self.restoration_gate()["pass"]
            record["pass"] = bool(
                record["selection_pass"] and record["restored_after_request"]
            )

    def restore_all(self) -> None:
        for binding in self.fp8:
            binding.layer._cb_use_fp8_v2_w13 = binding.original_fp8_w13
            binding.layer._cb_use_fp8_v2_w2 = binding.original_fp8_w2

    def restoration_gate(self) -> dict[str, Any]:
        violations = self._drift()
        return {"violations": violations, "pass": not violations}


def _tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    import torch

    cpu = tensor.detach().contiguous().to(device="cpu")
    raw = bytes(cpu.view(torch.uint8).reshape(-1).tolist())
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class FP8GemvDispatchProbe:
    """Request-scoped proof of layer routes and all grouped GEMV calls."""

    _OP_ATTRS = {
        "fp4_inherited": "cb_moe_gemv_fp4_v2",
        "fp4_v2": "cb_moe_gemv_v2",
        "fp8_inherited": "cb_moe_gemv_fp8",
        "fp8_v2": "cb_moe_gemv_fp8_v2",
    }

    def __init__(
        self, *, ops: Any, moe: Any, controller: FP8GemvV2ArmController
    ) -> None:
        self.ops = ops
        self.moe = moe
        self.controller = controller
        self.records: list[dict[str, Any]] = []
        self.unscoped_calls: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._current_layer: dict[str, Any] | None = None
        self._original_ops = {
            name: getattr(ops, attribute)
            for name, attribute in self._OP_ATTRS.items()
        }
        self._original_grouped = moe.PrismaQuantCBMoEMethod._apply_grouped_decode
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("routed-FP8 dispatch probe is already installed")
        for name, attribute in self._OP_ATTRS.items():
            original = self._original_ops[name]

            def wrapper(*args: Any, _name=name, _original=original, **kwargs: Any):
                self._record_op(_name, args)
                return _original(*args, **kwargs)

            wrapper.__name__ = f"gridbook_fp8_ab_probe_{attribute}"
            setattr(self.ops, attribute, wrapper)

        original_grouped = self._original_grouped

        def grouped_wrapper(
            method: Any,
            layer: Any,
            x: Any,
            topk_weights: Any,
            topk_ids: Any,
            act: Any,
        ) -> Any:
            if self._current_layer is not None:
                raise RuntimeError("nested grouped-decode probes are forbidden")
            self._record_layer(method, layer, x, topk_weights, topk_ids)
            self._current_layer = {
                "prefix": str(method.prefix),
                "is_fp4": bool(method.is_fp4),
            }
            try:
                return original_grouped(
                    method, layer, x, topk_weights, topk_ids, act
                )
            finally:
                self._current_layer = None

        grouped_wrapper.__name__ = "gridbook_fp8_ab_probe_grouped_decode"
        self.moe.PrismaQuantCBMoEMethod._apply_grouped_decode = grouped_wrapper
        self._installed = True

    @contextmanager
    def request(
        self, *, label: str, arm: str, expected_tokens: int
    ) -> Iterator[dict[str, Any]]:
        if not self._installed:
            raise RuntimeError("routed-FP8 dispatch probe is not installed")
        if self._active is not None:
            raise RuntimeError("nested dispatch request scopes are forbidden")
        record = {
            "pid": os.getpid(),
            "label": label,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "expected_tokens": int(expected_tokens),
            "layer_calls": [],
            "op_calls": [],
        }
        self._active = record
        try:
            yield record
        finally:
            self._active = None
            self._finalize_record(record)
            self.records.append(record)

    def _require_active(self, kind: str) -> dict[str, Any]:
        if self._active is None:
            event = {"kind": kind, "reason": "call outside request scope"}
            self.unscoped_calls.append(event)
            raise RuntimeError(
                f"routed-FP8 probe observed {kind} outside a request scope"
            )
        return self._active

    def _record_layer(
        self, method: Any, layer: Any, x: Any, topk_weights: Any, topk_ids: Any
    ) -> None:
        record = self._require_active("grouped_decode")
        record["layer_calls"].append({
            "prefix": str(method.prefix),
            "tokens": int(x.shape[0]),
            "is_fp4": bool(method.is_fp4),
            "fp4_w13_v2": bool(getattr(layer, "_cb_use_v2_w13", False)),
            "fp4_w2_v2": bool(getattr(layer, "_cb_use_v2_w2", False)),
            "fp8_w13_v2": bool(
                getattr(layer, "_cb_use_fp8_v2_w13", False)
            ),
            "fp8_w2_v2": bool(
                getattr(layer, "_cb_use_fp8_v2_w2", False)
            ),
            "topk_ids": _tensor_fingerprint(topk_ids),
            "topk_weights": _tensor_fingerprint(topk_weights),
        })

    def _record_op(self, op: str, args: Sequence[Any]) -> None:
        record = self._require_active(op)
        if self._current_layer is None:
            event = {
                "kind": op,
                "reason": "CUDA op outside grouped-decode layer scope",
            }
            self.unscoped_calls.append(event)
            raise RuntimeError(
                f"routed-FP8 probe observed {op} outside a layer scope"
            )
        if len(args) < 6:
            raise RuntimeError(f"{op} probe received a truncated signature")
        xq, qw, _cb, _scale, pair_expert, pair_xrow = args[:6]
        pair_count = int(pair_expert.shape[0])
        input_rows = int(xq.shape[0])
        tokens = (
            pair_count // _EXPECTED_TOPK
            if pair_count % _EXPECTED_TOPK == 0 else -1
        )
        stage = (
            "w13" if input_rows == tokens
            else "w2" if input_rows == pair_count
            else "unknown"
        )
        scalars = [int(value) for value in args[6:] if isinstance(value, int)]
        record["op_calls"].append({
            "op": op,
            "prefix": self._current_layer["prefix"],
            "is_fp4": self._current_layer["is_fp4"],
            "stage": stage,
            "pair_count": pair_count,
            "pair_xrow_count": int(pair_xrow.shape[0]),
            "input_rows": input_rows,
            "tokens": tokens,
            "output_columns": int(qw.shape[1]),
            "layout_scalars": scalars,
        })

    def _finalize_record(self, record: MutableMapping[str, Any]) -> None:
        violations: list[str] = []
        expected_tokens = int(record["expected_tokens"])
        expected_pairs = expected_tokens * _EXPECTED_TOPK
        requested = record["arm"] == "fused"
        layer_calls = list(record["layer_calls"])
        op_calls = list(record["op_calls"])
        prefix_counts = Counter(call["prefix"] for call in layer_calls)
        if set(prefix_counts) != self.controller.prefixes:
            violations.append("grouped-decode layer prefix set differs")
        if any(count != 1 for count in prefix_counts.values()):
            violations.append("a grouped-decode layer ran other than once")
        if len(layer_calls) != 43:
            violations.append("request did not traverse exactly 43 MoE layers")
        for call in layer_calls:
            if call["tokens"] != expected_tokens:
                violations.append("a layer received the wrong token cardinality")
            expected_route_shape = [expected_tokens, _EXPECTED_TOPK]
            if (
                call["topk_ids"]["shape"] != expected_route_shape
                or call["topk_weights"]["shape"] != expected_route_shape
            ):
                violations.append("a layer exposed the wrong top-k route shape")

        fp4_layers = [call for call in layer_calls if call["is_fp4"]]
        fp8_layers = [call for call in layer_calls if not call["is_fp4"]]
        if len(fp4_layers) != 35 or len(fp8_layers) != 8:
            violations.append("request did not traverse exact 35 FP4 + 8 FP8")
        if any(
            not call["fp4_w13_v2"] or not call["fp4_w2_v2"]
            or call["fp8_w13_v2"] or call["fp8_w2_v2"]
            for call in fp4_layers
        ):
            violations.append("an FP4 selector changed or entered FP8-v2")
        if any(
            call["fp4_w13_v2"] or call["fp4_w2_v2"]
            or call["fp8_w13_v2"] is not requested
            or call["fp8_w2_v2"] is not requested
            for call in fp8_layers
        ):
            violations.append("an FP8 layer exposed the wrong selector")

        counts = Counter(call["op"] for call in op_calls)
        selected_fp8 = "fp8_v2" if requested else "fp8_inherited"
        expected_counts = {"fp4_v2": 70, selected_fp8: 24}
        if dict(counts) != expected_counts:
            violations.append(
                f"CUDA op counts {dict(counts)} != expected {expected_counts}"
            )
        for call in op_calls:
            if (
                call["pair_count"] != expected_pairs
                or call["pair_xrow_count"] != expected_pairs
                or call["tokens"] != expected_tokens
                or call["stage"] == "unknown"
            ):
                violations.append("a CUDA op had the wrong routed cardinality")
            if call["is_fp4"]:
                if call["op"] != "fp4_v2":
                    violations.append("an FP4 layer used a fallback/wrong op")
            else:
                if call["op"] != selected_fp8:
                    violations.append("an FP8 layer used a fallback/wrong op")
                if call["layout_scalars"] != [28, 4, 112]:
                    violations.append("an FP8 op used a non-K28 layout")

        fp4_ops = [call for call in op_calls if call["is_fp4"]]
        fp8_ops = [call for call in op_calls if not call["is_fp4"]]
        if Counter(call["stage"] for call in fp4_ops) != {"w13": 35, "w2": 35}:
            violations.append("FP4 route lacks exact 35 w13 + 35 w2")
        if Counter(call["stage"] for call in fp8_ops) != {"w13": 16, "w2": 8}:
            violations.append("FP8 route lacks exact 16 role-w13 + 8 w2")
        for prefix in self.controller.fp8_prefixes:
            observed = Counter(
                call["stage"] for call in fp8_ops if call["prefix"] == prefix
            )
            if observed != {"w13": 2, "w2": 1}:
                violations.append(f"{prefix}: incomplete FP8 projection route")

        record["observed_op_counts"] = dict(counts)
        record["expected_op_counts"] = expected_counts
        record["observed_layer_count"] = len(layer_calls)
        record["expected_layer_count"] = 43
        record["expected_pair_count_per_op"] = expected_pairs
        record["fallback_or_wrong_route_count"] = sum(
            call["op"] not in expected_counts for call in op_calls
        )
        record["route_signature"] = [
            {key: call[key] for key in (
                "prefix", "tokens", "is_fp4", "topk_ids", "topk_weights"
            )}
            for call in layer_calls
        ]
        record["violations"] = sorted(set(violations))
        record["pass"] = not violations

    def restore(self) -> None:
        if not self._installed:
            return
        for name, attribute in self._OP_ATTRS.items():
            setattr(self.ops, attribute, self._original_ops[name])
        self.moe.PrismaQuantCBMoEMethod._apply_grouped_decode = (
            self._original_grouped
        )
        self._installed = False

    def restoration_gate(self) -> dict[str, Any]:
        checks = {
            attribute: getattr(self.ops, attribute) is self._original_ops[name]
            for name, attribute in self._OP_ATTRS.items()
        }
        checks["_apply_grouped_decode"] = (
            self.moe.PrismaQuantCBMoEMethod._apply_grouped_decode
            is self._original_grouped
        )
        return {
            "checks": checks,
            "unscoped_calls": list(self.unscoped_calls),
            "pass": all(checks.values()) and not self.unscoped_calls,
        }


def _assert_measurement_environment() -> None:
    expected = {
        FP8_GEMV_ENV: "1",
        GEMV_ENV: "v2",
        DECODE_CONTRACT_ENV: "v1",
        **{name: None for name in W2_SCHEDULE_ENVS},
    }
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"routed-FP8 measurement environment changed mid-run: {observed}"
        )


def _generation_digest(output: Any) -> dict[str, Any]:
    choices = getattr(output, "outputs", None)
    if not isinstance(choices, Sequence) or len(choices) != 1:
        raise RuntimeError("vLLM result did not contain exactly one generation")
    token_ids = tuple(int(token) for token in choices[0].token_ids)
    raw = struct.pack(f"<{len(token_ids)}q", *token_ids)
    return {
        "token_count": len(token_ids),
        "token_ids_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _run_generate(
    *, llm: Any, sampling: Any, prompt_ids: list[int], arm: str, label: str,
    controller: FP8GemvV2ArmController, probe: FP8GemvDispatchProbe,
    synchronize: Any,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    _assert_measurement_environment()
    with controller.arm(arm, label=label) as selector:
        with probe.request(
            label=label, arm=arm, expected_tokens=len(prompt_ids)
        ) as dispatch:
            synchronize()
            started = time.perf_counter()
            output = llm.generate(
                [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
            )[0]
            synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not selector["pass"]:
        raise RuntimeError(f"selector restore gate failed: {selector}")
    if not dispatch["pass"]:
        raise RuntimeError(f"request dispatch gate failed: {dispatch}")
    return output, elapsed_ms, selector, dispatch


def _records_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [
        {"label": record["label"], "violations": record.get("violations", [])}
        for record in records if not record.get("pass")
    ]
    return {
        "requests": len(records),
        "failed_requests": failures,
        "pass": bool(records) and not failures,
    }


def _selector_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [dict(record) for record in records if not record.get("pass")]
    return {
        "measurements": len(records),
        "failed": failures,
        "pass": bool(records) and not failures,
    }


def _route_invariance_gate(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches = []
    for pair in pairs:
        baseline = pair["dispatch"]["baseline"]["route_signature"]
        candidate = pair["dispatch"]["fused"]["route_signature"]
        if baseline != candidate:
            mismatches.append({
                "source_prompt_index": pair["source_prompt_index"],
                "repeat_index": pair["repeat_index"],
                "reason": "router id/weight fingerprints differ across arms",
            })
    return {
        "paired_requests": len(pairs),
        "comparison": "exact per-layer topk id and weight storage digests",
        "mismatches": mismatches,
        "pass": bool(pairs) and not mismatches,
    }


def _cross_arm_digest_gate(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches = []
    fields = (
        "prompt_token_ids_sha256", "score_sha256", "row_sha256"
    )
    for pair in pairs:
        baseline = pair["score_digests"]["baseline"]
        candidate = pair["score_digests"]["fused"]
        bad = [field for field in fields if baseline[field] != candidate[field]]
        if pair["generation_digests"]["baseline"] != pair[
            "generation_digests"
        ]["fused"]:
            bad.append("generation_token_ids_sha256")
        if bad:
            mismatches.append({
                "source_prompt_index": pair["source_prompt_index"],
                "repeat_index": pair["repeat_index"],
                "fields": bad,
                "baseline_score_sha256": baseline["score_sha256"],
                "candidate_score_sha256": candidate["score_sha256"],
            })
    return {
        "paired_requests": len(pairs),
        "comparison": (
            "exact full-vocabulary float32 rows, target float64 values, "
            "prompt ids, and generated token ids"
        ),
        "mismatches": mismatches,
        "pass": bool(pairs) and not mismatches,
    }


def _repeat_digest_gate(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_gate = pb._full_vocab_repeat_determinism_gate(
        pairs, n_prompts=_PROFILE_SAMPLES, repeats=_QUALITY_REPEATS
    )
    generation_mismatches = []
    for prompt_index in range(_PROFILE_SAMPLES):
        prompt_pairs = sorted(
            (pair for pair in pairs
             if pair["source_prompt_index"] == prompt_index),
            key=lambda pair: pair["repeat_index"],
        )
        if len(prompt_pairs) != _QUALITY_REPEATS:
            continue
        for arm in ARMS:
            expected = prompt_pairs[0]["generation_digests"][arm]
            for pair in prompt_pairs[1:]:
                if pair["generation_digests"][arm] != expected:
                    generation_mismatches.append({
                        "source_prompt_index": prompt_index,
                        "repeat_index": pair["repeat_index"],
                        "arm": arm,
                    })
    return {
        "full_vocab": base_gate,
        "generation_mismatches": generation_mismatches,
        "pass": base_gate["pass"] and not generation_mismatches,
    }


def _extension_residency_gate(
    cuda_ext: Any, bootstrap_extension: Mapping[str, Any], *,
    main_extension: Any, fp4_extension: Any,
) -> dict[str, Any]:
    current_main = cuda_ext.get_ext()
    current_fp4 = cuda_ext.get_ext_v2()
    checks = {
        "main_extension_identity_stable": current_main is main_extension,
        "fp4_extension_identity_stable": current_fp4 is fp4_extension,
        "main_preloaded_before_model": bool(
            bootstrap_extension.get("preloaded_before_model")
        ),
        "main_inherited_fp8_symbol": hasattr(
            main_extension, "cb_moe_gemv_fp8"
        ),
        "main_whole_row_fp8_symbol": hasattr(
            main_extension, "cb_moe_gemv_fp8_v2"
        ),
        "main_qdq_symbol": hasattr(main_extension, "fp8_act_qdq"),
        "main_combine_symbol": hasattr(main_extension, "cb_moe_combine"),
        "fixed_fp4_v2_symbol": hasattr(fp4_extension, "cb_gemv_v2"),
    }
    modules = {
        "main": _required_record(Path(main_extension.__file__)),
        "fp4_v2": _required_record(Path(fp4_extension.__file__)),
    }
    checks["bootstrap_main_path_matches"] = (
        modules["main"]["path"] == bootstrap_extension.get("path")
    )
    checks["bootstrap_main_sha256_matches"] = (
        modules["main"]["sha256"] == bootstrap_extension.get("sha256")
    )
    return {"modules": modules, "checks": checks, "pass": all(checks.values())}


def _same_pid_gate(
    *, runtime: Mapping[str, Any], llm: Any,
    selectors: Sequence[Mapping[str, Any]],
    dispatch: Sequence[Mapping[str, Any]], engine_object_id: int,
) -> dict[str, Any]:
    expected = os.getpid()
    selector_pid_set = {record.get("pid") for record in selectors}
    dispatch_pid_set = {record.get("pid") for record in dispatch}
    selector_pids = sorted(str(value) for value in selector_pid_set)
    dispatch_pids = sorted(str(value) for value in dispatch_pid_set)
    checks = {
        "runtime_pid": runtime.get("pid") == expected,
        "selector_pids": selector_pid_set == {expected},
        "dispatch_pids": dispatch_pid_set == {expected},
        "engine_object_stable": id(llm) == engine_object_id,
        "v1_multiprocessing_disabled": os.environ.get(v5.VLLM_MP_ENV) == "0",
    }
    return {
        "expected_pid": expected,
        "runtime_pid": runtime.get("pid"),
        "selector_pids": selector_pids,
        "dispatch_pids": dispatch_pids,
        "engine_object_id": engine_object_id,
        "model_load_count": 1,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _configured_numerical_gates(
    args: argparse.Namespace, quality: Mapping[str, Any]
) -> dict[str, Any]:
    delta = quality["delta"]
    gates = {}
    definitions = (
        (
            "max_mean_kl", args.max_mean_kl,
            delta["kl_baseline_to_fused"]["mean"],
        ),
        (
            "max_mean_nll_regression", args.max_mean_nll_regression,
            delta["mean_nll_fused_minus_baseline"],
        ),
        (
            "max_ppl_relative_regression", args.max_ppl_relative_regression,
            delta["ppl_relative_regression"],
        ),
    )
    for name, limit, observed in definitions:
        if limit is not None:
            gates[name] = {
                "limit": limit,
                "observed": observed,
                "pass": observed is not None and observed <= limit,
            }
    return gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.kv_cache_dtype != DSV4_KV_CACHE_DTYPE:
        raise RuntimeError(
            "routed-FP8-v2 DSV4 validation requires "
            f"kv_cache_dtype={DSV4_KV_CACHE_DTYPE!r}, got "
            f"{args.kv_cache_dtype!r}"
        )
    if "vllm" in sys.modules or "gridbook.moe_gemv_select" in sys.modules:
        raise RuntimeError(
            "vLLM/Gridbook selector was imported before the harness could "
            f"pin its process contract; launch {Path(__file__).name} fresh"
        )

    measured_envs = (
        FP8_GEMV_ENV, GEMV_ENV, DECODE_CONTRACT_ENV, *W2_SCHEDULE_ENVS
    )
    inherited = {name: os.environ.get(name) for name in measured_envs}
    os.environ[v5.VLLM_MP_ENV] = "0"
    os.environ[FP8_GEMV_ENV] = "1"
    os.environ[GEMV_ENV] = "v2"
    os.environ[DECODE_CONTRACT_ENV] = "v1"
    for name in W2_SCHEDULE_ENVS:
        os.environ.pop(name, None)
    os.environ[pb.PERSISTENT_B_ENV] = "0"
    os.environ[pb.BF16_SM120_ENV] = "0"
    os.environ[v5.FUSED_MOE_ENV] = ""
    os.environ.pop(v5.PREFILL_ENV, None)
    started = time.monotonic()

    bootstrap = validation_common.prepare_validation(
        args,
        harness_path=Path(__file__),
        helpers=v5,
        extension_none_message=(
            "main Gridbook extension did not load; refusing a fallback A/B"
        ),
        extension_loader="get_ext",
        required_symbol="cb_moe_gemv_fp8_v2",
        validation_name="routed-FP8 whole-row GEMV-v2",
        prompt_loader=base._fixed_decode_prompt_loader,
    )
    torch = bootstrap.torch
    runtime = bootstrap.runtime
    moe = bootstrap.moe
    main_extension = bootstrap.cuda_ext.require_ext(
        "routed-FP8 same-process quality A/B"
    )
    fp4_extension = bootstrap.cuda_ext.get_ext_v2()
    if fp4_extension is None or not hasattr(fp4_extension, "cb_gemv_v2"):
        raise RuntimeError(
            "fixed FP4-v2 extension is unavailable; refusing a mixed/fallback A/B"
        )
    source_manifest = _build_source_manifest(
        runtime,
        main_extension=main_extension,
        fp4_extension=fp4_extension,
    )

    from gridbook import moe_gemv_select, ops
    from vllm import envs as vllm_envs

    skip_padding_gate = base._moe_skip_padding_gate(
        vllm_envs.VLLM_MOE_SKIP_PADDING
    )
    if not skip_padding_gate["pass"]:
        raise RuntimeError(
            "exact routed-FP8 validation requires VLLM_MOE_SKIP_PADDING=True"
        )
    model_gate = pb._model_contract_gate(bootstrap.candidate_config, args)
    raw_config = v5._config_dict(bootstrap.candidate_config)
    model_gate["checks"]["num_experts_per_tok"] = (
        raw_config.get("num_experts_per_tok") == _EXPECTED_TOPK
    )
    model_gate["expected"]["num_experts_per_tok"] = _EXPECTED_TOPK
    model_gate["observed"]["num_experts_per_tok"] = raw_config.get(
        "num_experts_per_tok"
    )
    model_gate["pass"] = all(model_gate["checks"].values())
    if not model_gate["pass"]:
        raise RuntimeError(f"loaded model is not exact DSV4: {model_gate}")
    artifact_gate = base._artifact_gate(
        bootstrap.candidate_artifact_provenance
    )
    if not artifact_gate["pass"]:
        raise RuntimeError(
            f"candidate is not exact dsv4flash0731: {artifact_gate}"
        )
    if moe_gemv_select.cb_gemv_mode() != "v2":
        raise RuntimeError("fixed FP4 GEMV selector did not resolve to v2")
    if moe_gemv_select.cb_fp8_gemv_v2_requested() is not True:
        raise RuntimeError("routed-FP8 selector did not resolve to enabled")

    class _LoadProbe:
        @staticmethod
        def restore() -> None:
            return None

    engine = validation_common.load_candidate_engine(
        bootstrap, args, probe=_LoadProbe(), attest_chunked_prefill=True
    )
    if engine.chunked_prefill_contract is None:
        raise RuntimeError("chunked-prefill contract attestation did not run")
    extension_gate = _extension_residency_gate(
        bootstrap.cuda_ext,
        bootstrap.extension,
        main_extension=main_extension,
        fp4_extension=fp4_extension,
    )
    if not extension_gate["pass"]:
        raise RuntimeError(
            f"extension residency could not be proven: {extension_gate}"
        )

    controller = FP8GemvV2ArmController(ops=ops, moe=moe)
    probe = FP8GemvDispatchProbe(ops=ops, moe=moe, controller=controller)
    probe.install()
    llm = engine.llm
    engine_object_id = id(llm)
    synchronize = torch.cuda.synchronize
    warmup_records: list[dict[str, Any]] = []
    selectors: list[dict[str, Any]] = []
    quality_pairs: list[dict[str, Any]] = []
    accumulator = v5._new_quality_accumulator(kl_mode=v5.KL_FULL_VOCAB)

    try:
        for warmup_index in range(args.warmup_pairs):
            prompt_index = warmup_index % len(bootstrap.prompts)
            for arm in v5.paired_arm_order(warmup_index):
                output, wall_ms, selector, dispatch = _run_generate(
                    llm=llm,
                    sampling=engine.timing_sampling,
                    prompt_ids=bootstrap.prompts[prompt_index],
                    arm=arm,
                    label=f"warmup:{warmup_index}:{prompt_index}:{arm}",
                    controller=controller,
                    probe=probe,
                    synchronize=synchronize,
                )
                selectors.append(selector)
                warmup_records.append({
                    "warmup_index": warmup_index,
                    "source_prompt_index": prompt_index,
                    "arm": arm,
                    "wall_ms": wall_ms,
                    "generation_digest": _generation_digest(output),
                    "selector": selector,
                    "dispatch": dispatch,
                })
                del output

        for repeat in range(_QUALITY_REPEATS):
            for prompt_index, prompt_ids in enumerate(bootstrap.prompts):
                block_index = repeat * len(bootstrap.prompts) + prompt_index
                order = v5.paired_arm_order(repeat + prompt_index)
                compact_scores: dict[str, Any] = {}
                score_digests: dict[str, dict[str, Any]] = {}
                generation_digests: dict[str, dict[str, Any]] = {}
                dispatches: dict[str, dict[str, Any]] = {}
                selector_pair: dict[str, dict[str, Any]] = {}
                walls: dict[str, float] = {}
                for arm in order:
                    output, wall_ms, selector, dispatch = _run_generate(
                        llm=llm,
                        sampling=engine.quality_sampling,
                        prompt_ids=prompt_ids,
                        arm=arm,
                        label=f"quality:{repeat}:{prompt_index}:{arm}",
                        controller=controller,
                        probe=probe,
                        synchronize=synchronize,
                    )
                    selectors.append(selector)
                    selector_pair[arm] = selector
                    dispatches[arm] = dispatch
                    walls[arm] = wall_ms
                    generation_digests[arm] = _generation_digest(output)
                    score = pb._compact_full_vocab_score(
                        output,
                        prompt_ids,
                        expected_vocab_size=bootstrap.candidate_vocab_size,
                    )
                    compact_scores[arm] = score
                    score_digests[arm] = score.digest_record()
                    del output
                scoring_pair = {
                    "prompt_index": block_index,
                    "pair_order": list(order),
                    "scores": compact_scores,
                }
                v5._accumulate_quality_pair(accumulator, scoring_pair)
                quality_pairs.append({
                    "prompt_index": block_index,
                    "block_index": block_index,
                    "repeat_index": repeat,
                    "source_prompt_index": prompt_index,
                    "pair_order": list(order),
                    "score_digests": score_digests,
                    "generation_digests": generation_digests,
                    "wall_ms": walls,
                    "selectors": selector_pair,
                    "dispatch": dispatches,
                })
                del scoring_pair, compact_scores, score
    finally:
        probe.restore()
        controller.restore_all()

    quality = v5._finish_quality_accumulator(accumulator)
    quality["arm_labels"] = dict(ARM_LABELS)
    all_dispatch = list(probe.records)
    profile = bootstrap.dataset["decode_profile"]
    exact_repeat_gate = _repeat_digest_gate(quality_pairs)
    exact_cross_arm_gate = _cross_arm_digest_gate(quality_pairs)
    route_gate = _route_invariance_gate(quality_pairs)
    core_gates = {
        "exact_runtime_source_and_binary_manifest": _source_manifest_gate(
            source_manifest
        ),
        "exact_model_contract": model_gate,
        "exact_artifact": artifact_gate,
        "canonical_sealed_8x16_workload": {
            "expected_tensor_sha256": _EXPECTED_PREFIX_TENSOR_SHA256,
            "observed_tensor_sha256": profile["token_ids_tensor_sha256"],
            "expected_json_sha256": _EXPECTED_PREFIX_JSON_SHA256,
            "observed_json_sha256": profile["token_ids_json_sha256"],
            "pass": (
                profile["token_ids_tensor_sha256"]
                == _EXPECTED_PREFIX_TENSOR_SHA256
                and profile["token_ids_json_sha256"]
                == _EXPECTED_PREFIX_JSON_SHA256
                and profile["n_samples"] == _PROFILE_SAMPLES
                and profile["seqlen"] == _PROFILE_SEQLEN
            ),
        },
        "exact_loaded_layer_and_selector_inventory": controller.inventory_gate,
        "same_pid_one_engine_one_load": _same_pid_gate(
            runtime=runtime,
            llm=llm,
            selectors=selectors,
            dispatch=all_dispatch,
            engine_object_id=engine_object_id,
        ),
        "vllm_moe_skip_padding_contract": skip_padding_gate,
        "matched_extension_residency": extension_gate,
        "chunked_prefill_execution_contract": (
            validation_common.chunked_prefill_integrity_gate(
                engine.chunked_prefill_contract
            )
        ),
        "selector_switch_and_per_request_restore": _selector_gate(selectors),
        "exact_routes_cardinality_and_no_fallback": _records_gate(all_dispatch),
        "cross_arm_router_route_invariance": route_gate,
        "quality_arm_order_counterbalanced": pb._pair_order_gate(quality_pairs),
        "exact_repeat_full_vocab_digests": exact_repeat_gate,
        "exact_cross_arm_full_vocab_digests": exact_cross_arm_gate,
        "full_vocab_cardinality_and_finiteness": base._score_cardinality_gate(
            quality_pairs, vocab_size=bootstrap.candidate_vocab_size
        ),
        "controller_final_restore": controller.restoration_gate(),
        "probe_final_restore_and_scope": probe.restoration_gate(),
    }
    numerical_gates = _configured_numerical_gates(args, quality)
    measurement_valid = all(bool(gate.get("pass")) for gate in core_gates.values())
    numerical_gates_pass = all(
        bool(gate.get("pass")) for gate in numerical_gates.values()
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": v5._utc_now(),
        "scope": (
            "exact dsv4flash0731 target decode at M=16; TP=1; one eager "
            "same-PID vLLM engine; inherited vs whole-row routed-FP8 GEMV"
        ),
        "arm_labels": dict(ARM_LABELS),
        "comparison_contract": {
            "packed_weights_scales_codebooks": "identical actual artifact",
            "fp4_layers": "fixed CB-GEMV-v2 route in both arms",
            "fp8_difference": (
                "only 16 w13/w2 booleans on eight routed-FP8 layers"
            ),
            "expected_output_relation": "exact full-vocabulary equality",
            "primary_promotion_criteria": [
                "exact_repeat_full_vocab_digests",
                "exact_cross_arm_full_vocab_digests",
                "cross_arm_router_route_invariance",
            ],
            "numerical_metrics": "diagnostic/backstop only",
        },
        "settings": {
            "model": args.model,
            "model_resolved": str(bootstrap.candidate_path.resolve()),
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "v1_multiprocessing": False,
            "prefix_caching": False,
            "kv_cache_dtype": args.kv_cache_dtype,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup_pairs": args.warmup_pairs,
            "quality_repeats": _QUALITY_REPEATS,
            "quality_kl_mode": v5.KL_FULL_VOCAB,
            "quality_prompt_logprobs_request": -1,
            "candidate_vocab_size": bootstrap.candidate_vocab_size,
            "decode_profile": profile,
            "seed": args.seed,
        },
        "environment_contract": {
            "inherited_before_sanitization": inherited,
            "during_load_and_measurement": {
                name: os.environ.get(name) for name in measured_envs
            },
            "process_fp8_selector_remained_enabled": True,
            "request_exception": (
                "only _cb_use_fp8_v2_w13/_w2 on eight FP8 layers"
            ),
        },
        "runtime": runtime,
        "source_manifest": source_manifest,
        "extension": extension_gate,
        "candidate_artifact_provenance": (
            bootstrap.candidate_artifact_provenance
        ),
        "dataset": bootstrap.dataset,
        "model_load_seconds": engine.model_load_seconds,
        "warmup": warmup_records,
        "quality": quality,
        "numerical_diagnostics": {
            "target_mean_nll": {
                arm: quality["arms"][arm]["mean_nll"] for arm in ARMS
            },
            "target_mean_nll_delta_candidate_minus_baseline": (
                quality["delta"]["mean_nll_fused_minus_baseline"]
            ),
            "ppl_relative_regression": quality["delta"][
                "ppl_relative_regression"
            ],
            "kl_baseline_to_candidate": quality["delta"][
                "kl_baseline_to_fused"
            ],
        },
        "full_vocab_streaming_evidence": {
            "profile": profile,
            "pairs": quality_pairs,
            "storage_contract": (
                "vLLM FlatLogprobs -> paired float32 rows -> shared v5 "
                "incremental scorer; raw rows released after each pair"
            ),
        },
        "dispatch": {
            "layer_inventory": controller.inventory_gate,
            "request_attestations": all_dispatch,
            "selector_attestations": selectors,
        },
        "core_integrity_gates": core_gates,
        "configured_numerical_backstop_gates": numerical_gates,
        "measurement_only": bool(args.measurement_only),
        "measurement_valid": measurement_valid,
        "configured_gates_pass": numerical_gates_pass,
        "promotion_contract": {
            "same_process_without_reload": True,
            "exact_actual_artifact": True,
            "exact_full_vocab_cross_arm_required": True,
            "exact_full_vocab_repeat_required": True,
            "exact_router_route_required": True,
            "served_graph_performance_concurrency_soak_still_required": True,
            "complete": measurement_valid and numerical_gates_pass,
        },
        "limitations": [
            "This quality gate exercises eager grouped decode at M=16.",
            "CUDA-graph parity is covered by the separate direct operator gate.",
            "Served speed, concurrency, long-prefill, soak and memory remain separate.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    report["promotion_recommendation"] = (
        "measurement_failed"
        if not measurement_valid
        else "numerical_backstop_failed"
        if not numerical_gates_pass
        else "measurement_only"
        if args.measurement_only
        else "quality_candidate_only_requires_served_release_gates"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-token-ids-json", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--kv-cache-memory-bytes", type=v5._positive_int, default=268_435_456
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=(DSV4_KV_CACHE_DTYPE,),
        default=DSV4_KV_CACHE_DTYPE,
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=v5._unit_interval, default=0.90
    )
    parser.add_argument("--warmup-pairs", type=v5._positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measurement-only", action="store_true")
    parser.add_argument("--max-mean-kl", type=v5._nonnegative_float)
    parser.add_argument(
        "--max-mean-nll-regression", type=v5._nonnegative_float
    )
    parser.add_argument(
        "--max-ppl-relative-regression", type=v5._nonnegative_float
    )
    args = parser.parse_args(argv)

    model = Path(args.model).expanduser()
    if not model.is_dir():
        parser.error("routed-FP8-v2 validation requires a local --model")
    args.revision = None
    args.allow_downloads = False
    args.teacher_model = None
    args.teacher_revision = None
    args.teacher_dtype = "bfloat16"
    args.teacher_full_vocab_kl = True
    args.mode = "moe256"
    args.n_samples = _PROFILE_SAMPLES
    args.seqlen = _PROFILE_SEQLEN
    args.top_k = 129_280
    args.dtype = "bfloat16"
    args.quantization = "gridbook"
    args.tokenizer_mode = "deepseek_v4"
    args.max_num_batched_tokens = None
    args.enable_chunked_prefill = None
    args.dataset_cache_dir = None
    args.dataset_split = "train"
    args.wikitext_text = None
    args.expected_architecture = "DeepseekV4ForCausalLM"
    args.expected_model_type = "deepseek_v4"
    args.expected_hidden_layers = 43
    args.expected_hidden_size = 4096
    args.expected_moe_intermediate_size = 2048
    args.expected_routed_experts = 256
    args.expected_vocab_size = 129_280
    args.max_teacher_fused_mean_kl = None
    args.max_teacher_fused_kl_regression = None
    args.min_timing_speedup = None
    if args.measurement_only and any(
        value is not None for value in (
            args.max_mean_kl,
            args.max_mean_nll_regression,
            args.max_ppl_relative_regression,
        )
    ):
        parser.error("--measurement-only cannot be combined with thresholds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - preserve machine-readable failure
        failure = {
            "schema": SCHEMA,
            "created_at": v5._utc_now(),
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        }
        v5._atomic_json(args.output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1
    if not report["measurement_valid"] or not report["configured_gates_pass"]:
        report["status"] = "gate_failed"
    elif report["measurement_only"]:
        report["status"] = "measurement_only"
    else:
        report["status"] = "ok"
    v5._atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_delta": report["quality"]["delta"],
    }, indent=2), flush=True)
    return 0 if report["status"] in ("ok", "measurement_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
