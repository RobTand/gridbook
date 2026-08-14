#!/usr/bin/env python3
"""Same-engine DSV4 quality A/B for the alternate FP4-CB decode GEMV.

This is the promotion-quality gate for ``PRISMAQUANT_CB_GEMV=v2`` on the
exact dsv4flash0731 target artifact.  A fresh eager vLLM process loads the
candidate once with both the inherited and v2 CUDA modules resident.  The two
arms then toggle only the load-resolved Python booleans on the 35 target
FP4-CB MoE layers:

* ``baseline`` dispatches ``cb_moe_gemv_fp4_v2`` (the shipping inherited
  schedule); and
* ``fused`` dispatches ``cb_moe_gemv_v2`` (the smem-resident dictionary
  schedule).

The process-wide environment selector remains pinned to ``v2``.  Changing it
after load is both unsupported by Gridbook and an invalid A/B: the selector is
resolved once per process.  The harness instead switches the already-attested
``layer._cb_use_v2_w13`` and ``layer._cb_use_v2_w2`` booleans, restoring them
after every request and at shutdown.  Eager execution is mandatory because
those Python booleans are trace-time constants under graph capture.

The workload is value-closed: the first 16 tokens of every one of the eight
producer-sealed DSV4 WikiText windows.  Every request therefore reaches the
grouped decode branch at exactly M=16.  Full-vocabulary prompt logprobs are
streamed through the established v5 scalar scorer one pair at a time; raw rows
are released before the next pair.  Two counterbalanced repeats are fixed by
the script and exact score/row digest repeatability is a core gate.

Example::

    python3 scripts/validate_moe_gemv_v2_ab.py \
      --model /models/dsv4flash0731 \
      --prompt-token-ids-json /evidence/dsv4-wikitext-inputs-v1.json \
      --output /evidence/dsv4-gemv-v2-ab.json \
      --max-mean-kl 1e-4 \
      --max-mean-nll-regression 0.00498754 \
      --max-ppl-relative-regression 0.005

A pass qualifies model-level quality for this teacher-forced decode profile.
The separate operator gate requires exact BF16 equality on the DSV4 release
widths; the numeric thresholds here remain fail-closed corruption backstops.
Served graph throughput, concurrency, long prefill, soak, and memory remain
separate release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


def _load_persistent_b_helpers() -> Any:
    path = Path(__file__).with_name("validate_moe_persistent_b_ab.py")
    module_name = "_gridbook_validate_moe_persistent_b_for_gemv_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load persistent-B validation helpers from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pb = _load_persistent_b_helpers()
v5 = pb.v5
validation_common = pb.validation_common

SCHEMA = "gridbook.moe-gemv-v2-ab.v1"
DSV4_KV_CACHE_DTYPE = "fp8"
ARMS = v5.ARMS
ARM_LABELS = {
    "baseline": "inherited_default_grouped_fp4_v2",
    "fused": "smem_resident_dictionary_v2",
}

GEMV_ENV = "PRISMAQUANT_CB_GEMV"
DECODE_CONTRACT_ENV = "PRISMAQUANT_CB_DECODE_CONTRACT"
W2_SCHEDULE_ENVS = (
    "PRISMAQUANT_CB_W2_SCHED",
    "PRISMAQUANT_CB_W2_ROWS",
    "PRISMAQUANT_CB_W2_WARPS",
)

_PROFILE_SAMPLES = 8
_PROFILE_SEQLEN = 16
_QUALITY_REPEATS = 2
_EXPECTED_TOPK = 6
_EXPECTED_FP4_LAYER_IDS = frozenset({
    *range(0, 18), 20, 21, *range(23, 33), 37, 38, 40, 41, 42,
})
_EXPECTED_FP8_LAYER_IDS = frozenset({18, 19, 22, 33, 34, 35, 36, 39})
_EXPECTED_K16_LAYER_IDS = frozenset({0, 2})
_EXPECTED_PREFIX_TENSOR_SHA256 = (
    "11c40cbfd3819f72f18507f359787f479ff06d30fd6e30f697c3bc4e0b4b99f7"
)
_EXPECTED_PREFIX_JSON_SHA256 = (
    "9ed265d2e7202f2282a225929e55faef2a0f87b4508fe8fca10378de500b8c85"
)

# Exact dsv4flash0731 target artifact identity.  The shared provenance reader
# independently hashes every file, including the 112 GB safetensors payload;
# these constants make the gate compare those observations rather than merely
# record them.
_EXPECTED_ARTIFACT = {
    "config_sha256": (
        "cf07dd1184989ce21a3aa8a21815feb234bcec6385763ae2fcb2781bde015a89"
    ),
    "quant_config_sha256": (
        "6ecb5ffaca90a9ca3f5095df0d788381f9d3451dcb2c0ec5426ec736b26844ef"
    ),
    "codebook_sha256": (
        "ec893b2e56354d1ce8f8a5f613937a8b60b7cf7d2e08e590e114376cbabecc0c"
    ),
    "weight_sha256": (
        "d347c32304e50aa2e8904593744f1e38f7fa1a4a54aece47ae9b3b3e6c8b5334"
    ),
    "weight_bytes": 112_349_037_959,
    "codebook_relative_path": "cb_codebooks.pqcb",
}
_PAIRWISE_QUALITY_GATES = (
    "max_mean_kl",
    "max_mean_nll_regression",
    "max_ppl_relative_regression",
)
_LAYER_PREFIX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts$")


def _decode_profile(
    windows: Sequence[Sequence[int]],
) -> tuple[list[list[int]], dict[str, Any]]:
    if len(windows) != _PROFILE_SAMPLES:
        raise RuntimeError(
            f"decode profile requires {_PROFILE_SAMPLES} sealed windows"
        )
    prompts = [list(map(int, window[:_PROFILE_SEQLEN])) for window in windows]
    if any(len(prompt) != _PROFILE_SEQLEN for prompt in prompts):
        raise RuntimeError("a sealed window is shorter than the decode profile")
    if len({tuple(prompt) for prompt in prompts}) != len(prompts):
        raise RuntimeError("decode profile prompts are not content-distinct")
    flattened = [token for prompt in prompts for token in prompt]
    tensor_digest = hashlib.sha256(
        struct.pack(f"<{len(flattened)}q", *flattened)
    ).hexdigest()
    json_digest = pb._canonical_sha256(prompts)
    if tensor_digest != _EXPECTED_PREFIX_TENSOR_SHA256:
        raise RuntimeError(
            "derived 8x16 token tensor is not the canonical DSV4 decode profile"
        )
    if json_digest != _EXPECTED_PREFIX_JSON_SHA256:
        raise RuntimeError(
            "derived 8x16 JSON value digest is not the canonical DSV4 profile"
        )
    return prompts, {
        "schema": "gridbook.dsv4-gemv-v2-prefix-profile.v1",
        "selection": "first_16_tokens_of_every_sealed_full_kl_window",
        "n_samples": _PROFILE_SAMPLES,
        "seqlen": _PROFILE_SEQLEN,
        "quality_repeats": _QUALITY_REPEATS,
        "positions_per_repeat": _PROFILE_SAMPLES * (_PROFILE_SEQLEN - 1),
        "total_scored_positions": (
            _PROFILE_SAMPLES * (_PROFILE_SEQLEN - 1) * _QUALITY_REPEATS
        ),
        "token_ids_tensor_sha256": tensor_digest,
        "token_ids_json_sha256": json_digest,
        "prompt_sha256": [
            hashlib.sha256(
                struct.pack(f"<{len(prompt)}q", *prompt)
            ).hexdigest()
            for prompt in prompts
        ],
    }


def _fixed_decode_prompt_loader(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[list[list[int]], dict[str, Any]]:
    # The producer-owned loader intentionally requires the complete sealed
    # 8x512 payload.  Give it that immutable source contract, then derive the
    # independently digest-pinned 8x16 decode view used by this harness.
    source_args = SimpleNamespace(**vars(args))
    source_args.n_samples = pb._CANONICAL_FULL_KL_SELECTION["n_samples"]
    source_args.seqlen = pb._CANONICAL_FULL_KL_SELECTION["seqlen"]
    windows, source_record = pb._fixed_prompt_loader(source_args, tokenizer)
    prompts, profile = _decode_profile(windows)
    return prompts, {
        "name": "wikitext",
        "source": "producer_sealed_token_ids_derived_decode_prefix",
        "sealed_source": source_record,
        "decode_profile": profile,
    }


def _artifact_gate(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    if isinstance(provenance, Mapping):
        config = provenance.get("config")
        quant = provenance.get("quant_config")
        codebook = provenance.get("codebook_file")
        weights = provenance.get("weight_files")
        weight = (
            weights.get("model.safetensors")
            if isinstance(weights, Mapping)
            else None
        )
        observed = {
            "config_sha256": (
                config.get("sha256") if isinstance(config, Mapping) else None
            ),
            "quant_config_sha256": (
                quant.get("sha256") if isinstance(quant, Mapping) else None
            ),
            "codebook_sha256": (
                codebook.get("sha256")
                if isinstance(codebook, Mapping) else None
            ),
            "weight_sha256": (
                weight.get("sha256") if isinstance(weight, Mapping) else None
            ),
            "weight_bytes": (
                weight.get("bytes") if isinstance(weight, Mapping) else None
            ),
            "codebook_relative_path": (
                codebook.get("relative_path")
                if isinstance(codebook, Mapping) else None
            ),
        }
    checks = {
        key: observed.get(key) == expected
        for key, expected in _EXPECTED_ARTIFACT.items()
    }
    return {
        "expected": dict(_EXPECTED_ARTIFACT),
        "observed": observed,
        "checks": checks,
        "pass": bool(checks) and all(checks.values()),
    }


def _moe_skip_padding_gate(observed: Any) -> dict[str, Any]:
    """Attest the static routed-token contract of the qualified DSV4 image."""

    return {
        "expected": True,
        "observed": observed,
        "pass": observed is True,
    }


@dataclass(frozen=True)
class _LayerBinding:
    layer_id: int
    layer_index: int
    prefix: str
    method: Any
    layer: Any
    original_w13: bool
    original_w2: bool
    k_bits: int
    n_sub: int
    type_size: int
    hidden_size: int
    intermediate_size: int
    role_split: bool


class GemvV2ArmController:
    """Exception-safe switch over the exact loaded target-layer booleans."""

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
            original_w13 = getattr(layer, "_cb_use_v2_w13", None)
            original_w2 = getattr(layer, "_cb_use_v2_w2", None)
            if not isinstance(original_w13, bool) or not isinstance(
                original_w2, bool
            ):
                raise RuntimeError(
                    f"{prefix}: load did not resolve both CB-GEMV booleans"
                )
            bindings.append(_LayerBinding(
                layer_id=int(layer_id),
                layer_index=int(match.group(1)),
                prefix=prefix,
                method=method,
                layer=layer,
                original_w13=original_w13,
                original_w2=original_w2,
                k_bits=int(method.k),
                n_sub=int(method.n_sub),
                type_size=int(method.type_size),
                hidden_size=int(getattr(layer, "_cb_hidden", -1)),
                intermediate_size=int(getattr(layer, "_cb_inter", -1)),
                role_split=bool(getattr(layer, "_cb_role_split", False)),
            ))
        prefixes = [binding.prefix for binding in bindings]
        fp4 = tuple(binding for binding in bindings if bool(binding.method.is_fp4))
        fp8 = tuple(binding for binding in bindings if not bool(binding.method.is_fp4))
        self.fp4 = fp4
        self.fp8 = fp8
        self._all = (*fp4, *fp8)

        fp4_ids = {binding.layer_index for binding in fp4}
        fp8_ids = {binding.layer_index for binding in fp8}
        fp4_records = [self._binding_record(binding) for binding in fp4]
        fp8_records = [self._binding_record(binding) for binding in fp8]
        checks = {
            "no_malformed_moe_prefix": not malformed_prefixes,
            "unique_prefixes": len(prefixes) == len(set(prefixes)),
            "unique_layer_indices": len(bindings) == len({
                binding.layer_index for binding in bindings
            }),
            "exact_fp4_layer_ids": fp4_ids == _EXPECTED_FP4_LAYER_IDS,
            "exact_fp8_layer_ids": fp8_ids == _EXPECTED_FP8_LAYER_IDS,
            "exact_total_moe_layers": len(bindings) == 43,
            "all_fp4_two_tier_v2": all(
                bool(binding.method.is_v2)
                and binding.n_sub == 2
                and binding.type_size == 4 * binding.k_bits + 9
                for binding in fp4
            ),
            "exact_fp4_rungs": all(
                binding.k_bits
                == (16 if binding.layer_index in _EXPECTED_K16_LAYER_IDS else 18)
                for binding in fp4
            ),
            "all_fp4_shapes": all(
                binding.hidden_size == 4096
                and binding.intermediate_size == 2048
                for binding in fp4
            ),
            "all_fp4_v2_selected_at_load": all(
                binding.original_w13 is True and binding.original_w2 is True
                for binding in fp4
            ),
            "all_fp8_k28_product": all(
                not bool(binding.method.is_v2)
                and binding.k_bits == 28
                and binding.n_sub == 4
                and binding.type_size == 112
                for binding in fp8
            ),
            "all_fp8_role_split": all(binding.role_split for binding in fp8),
            "all_fp8_excluded_from_v2": all(
                binding.original_w13 is False and binding.original_w2 is False
                for binding in fp8
            ),
        }
        self.inventory_gate = {
            "expected_fp4_layer_ids": sorted(_EXPECTED_FP4_LAYER_IDS),
            "observed_fp4_layer_ids": sorted(fp4_ids),
            "expected_fp8_layer_ids": sorted(_EXPECTED_FP8_LAYER_IDS),
            "observed_fp8_layer_ids": sorted(fp8_ids),
            "expected_fp4_stack_booleans": 70,
            "observed_fp4_stack_booleans": len(fp4) * 2,
            "expected_fp8_stack_booleans": 16,
            "observed_fp8_stack_booleans": len(fp8) * 2,
            "malformed_prefixes": malformed_prefixes,
            "fp4": fp4_records,
            "fp8": fp8_records,
            "checks": checks,
            "pass": all(checks.values()),
        }
        if not self.inventory_gate["pass"]:
            raise RuntimeError(
                "loaded DSV4 target CB MoE inventory is not exact: "
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
            "resolved_w13_v2": binding.original_w13,
            "resolved_w2_v2": binding.original_w2,
        }

    @property
    def fp4_prefixes(self) -> frozenset[str]:
        return frozenset(binding.prefix for binding in self.fp4)

    @property
    def fp8_prefixes(self) -> frozenset[str]:
        return frozenset(binding.prefix for binding in self.fp8)

    @contextmanager
    def arm(self, arm: str, *, label: str) -> Iterator[dict[str, Any]]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        drift = [
            binding.prefix
            for binding in self.fp4
            if (
                getattr(binding.layer, "_cb_use_v2_w13", None)
                is not binding.original_w13
                or getattr(binding.layer, "_cb_use_v2_w2", None)
                is not binding.original_w2
            )
        ]
        if drift:
            raise RuntimeError(
                f"CB-GEMV selector booleans were not restored before {label}: "
                f"{drift}"
            )
        requested = arm == "fused"
        for binding in self.fp4:
            binding.layer._cb_use_v2_w13 = requested
            binding.layer._cb_use_v2_w2 = requested
        record = {
            "label": label,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "expected_v2_stack_booleans": 70 if requested else 0,
            "observed_v2_stack_booleans": sum(
                int(getattr(binding.layer, "_cb_use_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_v2_w2", False))
                for binding in self.fp4
            ),
            "fp8_stack_booleans_true": sum(
                int(getattr(binding.layer, "_cb_use_v2_w13", False))
                + int(getattr(binding.layer, "_cb_use_v2_w2", False))
                for binding in self.fp8
            ),
            "restored_after_request": False,
        }
        record["selection_pass"] = bool(
            record["observed_v2_stack_booleans"]
            == record["expected_v2_stack_booleans"]
            and record["fp8_stack_booleans_true"] == 0
        )
        if not record["selection_pass"]:
            self.restore_all()
            raise RuntimeError(f"CB-GEMV arm switch failed: {record}")
        try:
            yield record
        finally:
            self.restore_all()
            record["restored_after_request"] = self.restoration_gate()["pass"]
            record["pass"] = bool(
                record["selection_pass"] and record["restored_after_request"]
            )

    def restore_all(self) -> None:
        for binding in self.fp4:
            binding.layer._cb_use_v2_w13 = binding.original_w13
            binding.layer._cb_use_v2_w2 = binding.original_w2

    def restoration_gate(self) -> dict[str, Any]:
        violations = []
        for binding in self._all:
            for stack, expected in (
                ("w13", binding.original_w13), ("w2", binding.original_w2)
            ):
                observed = getattr(binding.layer, f"_cb_use_v2_{stack}", None)
                if observed is not expected:
                    violations.append({
                        "prefix": binding.prefix,
                        "stack": stack,
                        "expected": expected,
                        "observed": observed,
                    })
        return {"violations": violations, "pass": not violations}


class GemvDispatchProbe:
    """Eager-only, request-scoped proof of every layer and CUDA-op route."""

    _OP_ATTRS = {
        "inherited": "cb_moe_gemv_fp4_v2",
        "v2": "cb_moe_gemv_v2",
        "fp8": "cb_moe_gemv_fp8",
    }

    def __init__(self, *, ops: Any, moe: Any, controller: GemvV2ArmController):
        self.ops = ops
        self.moe = moe
        self.controller = controller
        self.records: list[dict[str, Any]] = []
        self.unscoped_calls: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._current_layer: dict[str, Any] | None = None
        self._original_ops = {
            name: getattr(ops, attr) for name, attr in self._OP_ATTRS.items()
        }
        self._original_grouped = moe.PrismaQuantCBMoEMethod._apply_grouped_decode
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("GEMV dispatch probe is already installed")
        for name, attr in self._OP_ATTRS.items():
            original = self._original_ops[name]

            def wrapper(*args: Any, _name=name, _original=original, **kwargs: Any):
                self._record_op(_name, args)
                return _original(*args, **kwargs)

            wrapper.__name__ = f"gridbook_ab_probe_{attr}"
            setattr(self.ops, attr, wrapper)
        original_grouped = self._original_grouped

        def grouped_wrapper(
            method: Any,
            layer: Any,
            x: Any,
            topk_weights: Any,
            topk_ids: Any,
            act: Any,
        ) -> Any:
            self._record_layer(method, layer, x)
            if self._current_layer is not None:
                raise RuntimeError("nested grouped-decode layer probes are forbidden")
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

        grouped_wrapper.__name__ = "gridbook_ab_probe_apply_grouped_decode"
        self.moe.PrismaQuantCBMoEMethod._apply_grouped_decode = grouped_wrapper
        self._installed = True

    @contextmanager
    def request(
        self, *, label: str, arm: str, expected_tokens: int
    ) -> Iterator[dict[str, Any]]:
        if not self._installed:
            raise RuntimeError("GEMV dispatch probe is not installed")
        if self._active is not None:
            raise RuntimeError("nested GEMV dispatch probe scopes are forbidden")
        record = {
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
                f"GEMV dispatch probe observed {kind} outside a request scope"
            )
        return self._active

    def _record_layer(self, method: Any, layer: Any, x: Any) -> None:
        record = self._require_active("grouped_decode")
        record["layer_calls"].append({
            "prefix": str(method.prefix),
            "tokens": int(x.shape[0]),
            "is_fp4": bool(method.is_fp4),
            "w13_v2": bool(getattr(layer, "_cb_use_v2_w13", False)),
            "w2_v2": bool(getattr(layer, "_cb_use_v2_w2", False)),
        })

    def _record_op(self, op: str, args: Sequence[Any]) -> None:
        record = self._require_active(op)
        if self._current_layer is None:
            event = {
                "kind": op,
                "reason": "CUDA op outside a grouped-decode layer scope",
            }
            self.unscoped_calls.append(event)
            raise RuntimeError(
                f"GEMV dispatch probe observed {op} outside a layer scope"
            )
        if len(args) < 6:
            raise RuntimeError(f"{op} probe received a truncated call signature")
        xq, qw, _cb, _compose_or_scale, pair_expert, _pair_xrow = args[:6]
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
        record["op_calls"].append({
            "op": op,
            "prefix": self._current_layer["prefix"],
            "is_fp4": self._current_layer["is_fp4"],
            "stage": stage,
            "pair_count": pair_count,
            "input_rows": input_rows,
            "tokens": tokens,
            "output_columns": int(qw.shape[1]),
        })

    def _finalize_record(self, record: MutableMapping[str, Any]) -> None:
        violations: list[str] = []
        expected_tokens = int(record["expected_tokens"])
        layer_calls = list(record["layer_calls"])
        op_calls = list(record["op_calls"])
        prefix_counts = Counter(call["prefix"] for call in layer_calls)
        expected_prefixes = (
            self.controller.fp4_prefixes | self.controller.fp8_prefixes
        )
        if set(prefix_counts) != expected_prefixes:
            violations.append("grouped-decode layer prefix set differs")
        if any(count != 1 for count in prefix_counts.values()):
            violations.append("a grouped-decode layer ran other than once")
        if any(call["tokens"] != expected_tokens for call in layer_calls):
            violations.append("a grouped-decode layer did not receive exact M=16")

        fp4_calls = [call for call in layer_calls if call["is_fp4"]]
        fp8_calls = [call for call in layer_calls if not call["is_fp4"]]
        requested = record["arm"] == "fused"
        if len(fp4_calls) != 35 or len(fp8_calls) != 8:
            violations.append("request did not traverse exact 35 FP4 + 8 FP8 layers")
        if any(
            call["w13_v2"] is not requested or call["w2_v2"] is not requested
            for call in fp4_calls
        ):
            violations.append("an FP4 layer exposed the wrong per-stack selector")
        if any(call["w13_v2"] or call["w2_v2"] for call in fp8_calls):
            violations.append("an FP8 layer entered the FP4 v2 selector")

        counts = Counter(call["op"] for call in op_calls)
        expected_counts = {
            "inherited": 0 if requested else 70,
            "v2": 70 if requested else 0,
            # Exact artifact has per-role FP8 books: gate/up plus w2 per layer.
            "fp8": 24,
        }
        if dict(counts) != {k: v for k, v in expected_counts.items() if v}:
            violations.append(
                f"CUDA op counts {dict(counts)} != expected {expected_counts}"
            )
        if any(call["tokens"] != expected_tokens for call in op_calls):
            violations.append("a CUDA op call did not represent exact M=16")
        if any(call["stage"] == "unknown" for call in op_calls):
            violations.append("a CUDA op call had an unknown projection stage")
        fp4_ops = [call for call in op_calls if call["op"] != "fp8"]
        fp8_ops = [call for call in op_calls if call["op"] == "fp8"]
        if Counter(call["stage"] for call in fp4_ops) != {"w13": 35, "w2": 35}:
            violations.append("FP4 dispatch did not contain exact 35 w13 + 35 w2")
        if Counter(call["stage"] for call in fp8_ops) != {"w13": 16, "w2": 8}:
            violations.append("FP8 dispatch did not contain exact 16 w13 + 8 w2")
        for prefix in self.controller.fp4_prefixes:
            observed = Counter(
                (call["op"], call["stage"])
                for call in fp4_ops if call["prefix"] == prefix
            )
            selected = "v2" if requested else "inherited"
            if observed != {(selected, "w13"): 1, (selected, "w2"): 1}:
                violations.append(
                    f"{prefix}: per-layer FP4 dispatch {dict(observed)} differs"
                )
        for prefix in self.controller.fp8_prefixes:
            observed = Counter(
                call["stage"] for call in fp8_ops if call["prefix"] == prefix
            )
            if observed != {"w13": 2, "w2": 1}:
                violations.append(
                    f"{prefix}: per-layer FP8 dispatch {dict(observed)} differs"
                )

        record["observed_op_counts"] = dict(counts)
        record["expected_op_counts"] = expected_counts
        record["observed_layer_count"] = len(layer_calls)
        record["expected_layer_count"] = 43
        record["fp8_signature"] = [
            {key: call[key] for key in (
                "prefix", "stage", "pair_count", "input_rows", "tokens",
                "output_columns",
            )}
            for call in fp8_ops
        ]
        record["violations"] = violations
        record["pass"] = not violations

    def restore(self) -> None:
        if not self._installed:
            return
        for name, attr in self._OP_ATTRS.items():
            setattr(self.ops, attr, self._original_ops[name])
        self.moe.PrismaQuantCBMoEMethod._apply_grouped_decode = (
            self._original_grouped
        )
        self._installed = False

    def restoration_gate(self) -> dict[str, Any]:
        checks = {
            attr: getattr(self.ops, attr) is self._original_ops[name]
            for name, attr in self._OP_ATTRS.items()
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
        GEMV_ENV: "v2",
        DECODE_CONTRACT_ENV: "v1",
        **{name: None for name in W2_SCHEDULE_ENVS},
    }
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"decode measurement environment changed mid-run: {observed}"
        )


def _run_generate(
    *,
    llm: Any,
    sampling: Any,
    prompt_ids: list[int],
    arm: str,
    label: str,
    controller: GemvV2ArmController,
    probe: GemvDispatchProbe,
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


def _dispatch_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        {
            "label": record["label"],
            "arm": record["arm"],
            "violations": list(record.get("violations", ())),
        }
        for record in records if not record.get("pass")
    ]
    return {
        "requests": len(records),
        "failed_requests": failed,
        "pass": bool(records) and not failed,
    }


def _selector_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [dict(record) for record in records if not record.get("pass")]
    return {
        "measurements": len(records),
        "failed": failed,
        "pass": bool(records) and not failed,
    }


def _fp8_dispatch_invariance_gate(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for pair in pairs:
        baseline = pair["dispatch"]["baseline"]["fp8_signature"]
        fused = pair["dispatch"]["fused"]["fp8_signature"]
        if baseline != fused:
            mismatches.append({
                "source_prompt_index": pair["source_prompt_index"],
                "repeat_index": pair["repeat_index"],
                "baseline": baseline,
                "fused": fused,
            })
    return {
        "paired_requests": len(pairs),
        "mismatches": mismatches,
        "pass": bool(pairs) and not mismatches,
    }


def _score_cardinality_gate(
    pairs: Sequence[Mapping[str, Any]], *, vocab_size: int
) -> dict[str, Any]:
    expected_positions = _PROFILE_SEQLEN - 1
    failures: list[dict[str, Any]] = []
    observations = 0
    for pair in pairs:
        for arm in ARMS:
            observations += 1
            score = pair["score_digests"][arm]
            if (
                score.get("positions") != expected_positions
                or score.get("vocab_size") != vocab_size
                or len(score.get("row_sha256", ())) != expected_positions
                or not math.isfinite(float(score.get("mean_nll", math.nan)))
                or not math.isfinite(float(score.get("ppl", math.nan)))
            ):
                failures.append({
                    "source_prompt_index": pair["source_prompt_index"],
                    "repeat_index": pair["repeat_index"],
                    "arm": arm,
                    "score": score,
                })
    expected_observations = _PROFILE_SAMPLES * _QUALITY_REPEATS * len(ARMS)
    return {
        "expected_positions_per_score": expected_positions,
        "expected_vocab_size": vocab_size,
        "expected_score_observations": expected_observations,
        "observed_score_observations": observations,
        "failures": failures,
        "pass": observations == expected_observations and not failures,
    }


def _extension_residency_gate(
    cuda_ext: Any, bootstrap_extension: Mapping[str, Any]
) -> dict[str, Any]:
    main = cuda_ext.get_ext()
    v2_ext = cuda_ext.get_ext_v2()
    checks = {
        "main_extension_resident": main is not None,
        "v2_extension_resident": v2_ext is not None,
        "main_inherited_symbol": (
            main is not None and hasattr(main, "cb_moe_gemv_fp4_v2")
        ),
        "main_fp8_symbol": (
            main is not None and hasattr(main, "cb_moe_gemv_fp8")
        ),
        "v2_symbol": v2_ext is not None and hasattr(v2_ext, "cb_gemv_v2"),
        "v2_preloaded_before_model": bool(
            bootstrap_extension.get("preloaded_before_model")
        ),
    }
    records: dict[str, Any] = {}
    for label, module in (("main", main), ("v2", v2_ext)):
        path_raw = getattr(module, "__file__", None) if module is not None else None
        if path_raw is not None:
            path = Path(path_raw).resolve()
            records[label] = {
                "path": str(path), **v5._required_file_record(path)
            }
    return {"modules": records, "checks": checks, "pass": all(checks.values())}


def _augment_provenance(runtime: MutableMapping[str, Any]) -> None:
    package_root = Path(runtime["gridbook"]["package_root"])
    additions = {
        "moe_gemv_select.py": package_root / "moe_gemv_select.py",
        "cb_gemv.cu": package_root / "csrc" / "cb_gemv.cu",
        "cb_gemv_v2.cu": package_root / "csrc" / "cb_gemv_v2.cu",
    }
    for label, path in additions.items():
        record = {"path": str(path.resolve()), **v5._required_file_record(path)}
        runtime["source_files"][label] = record
        runtime["source_sha256"][label] = record["sha256"]
    helper_path = Path(pb.__file__).resolve()
    runtime["harness"]["shared_helpers"]["persistent_b_validation_api"] = {
        "path": str(helper_path), **v5._required_file_record(helper_path)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.kv_cache_dtype != DSV4_KV_CACHE_DTYPE:
        raise RuntimeError(
            "CB-GEMV-v2 DSV4 validation requires "
            f"kv_cache_dtype={DSV4_KV_CACHE_DTYPE!r}, got "
            f"{args.kv_cache_dtype!r}"
        )
    if "vllm" in sys.modules or "gridbook.moe_gemv_select" in sys.modules:
        raise RuntimeError(
            "vLLM/Gridbook selector was imported before the harness could pin "
            f"the same-process contract; launch {Path(__file__).name} fresh"
        )
    measured_envs = (GEMV_ENV, DECODE_CONTRACT_ENV, *W2_SCHEDULE_ENVS)
    inherited = {name: os.environ.get(name) for name in measured_envs}
    os.environ[v5.VLLM_MP_ENV] = "0"
    os.environ[GEMV_ENV] = "v2"
    os.environ[DECODE_CONTRACT_ENV] = "v1"
    for name in W2_SCHEDULE_ENVS:
        os.environ.pop(name, None)
    # These prefill selectors are out of scope and must not allocate an
    # unrelated experimental route while this decode-only gate loads.
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
            "CB-GEMV-v2 extension did not build/load; refusing a fallback A/B"
        ),
        extension_loader="get_ext_v2",
        required_symbol="cb_gemv_v2",
        validation_name="CB-GEMV-v2",
        prompt_loader=_fixed_decode_prompt_loader,
    )
    torch = bootstrap.torch
    moe = bootstrap.moe
    runtime = bootstrap.runtime
    _augment_provenance(runtime)
    from gridbook import moe_gemv_select, ops
    from vllm import envs as vllm_envs

    skip_padding_gate = _moe_skip_padding_gate(
        vllm_envs.VLLM_MOE_SKIP_PADDING
    )
    if not skip_padding_gate["pass"]:
        raise RuntimeError(
            "exact DSV4 CB-GEMV-v2 validation requires "
            f"VLLM_MOE_SKIP_PADDING=True: {skip_padding_gate}"
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
    artifact_gate = _artifact_gate(bootstrap.candidate_artifact_provenance)
    if not artifact_gate["pass"]:
        raise RuntimeError(
            f"candidate is not the exact dsv4flash0731 artifact: {artifact_gate}"
        )
    if moe_gemv_select.cb_gemv_mode() != "v2":
        raise RuntimeError("CB-GEMV process selector did not resolve to v2")

    class _LoadProbe:
        @staticmethod
        def restore() -> None:
            return None

    engine = validation_common.load_candidate_engine(
        bootstrap, args, probe=_LoadProbe(), attest_chunked_prefill=True
    )
    chunked_contract = engine.chunked_prefill_contract
    if chunked_contract is None:
        raise RuntimeError("chunked-prefill contract attestation did not run")
    extension_gate = _extension_residency_gate(
        bootstrap.cuda_ext, bootstrap.extension
    )
    if not extension_gate["pass"]:
        raise RuntimeError(
            f"matched extension residency could not be proven: {extension_gate}"
        )

    controller = GemvV2ArmController(ops=ops, moe=moe)
    probe = GemvDispatchProbe(ops=ops, moe=moe, controller=controller)
    probe.install()
    llm = engine.llm
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
    core_gates = {
        "model_contract": model_gate,
        "exact_artifact": artifact_gate,
        "canonical_8x16_workload": {
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
        "loaded_layer_inventory": controller.inventory_gate,
        "vllm_moe_skip_padding_contract": skip_padding_gate,
        "matched_extension_residency": extension_gate,
        "same_process_one_engine": {
            "expected_pid": os.getpid(),
            "runtime_pid": runtime["pid"],
            "engine_object_id": id(llm),
            "model_load_count": 1,
            "pass": runtime["pid"] == os.getpid(),
        },
        "eager_toggle_contract": {
            "enforce_eager": True,
            "toggle_attributes": ["_cb_use_v2_w13", "_cb_use_v2_w2"],
            "environment_selector_remained": os.environ.get(GEMV_ENV),
            "pass": os.environ.get(GEMV_ENV) == "v2",
        },
        "chunked_prefill_execution_contract": (
            validation_common.chunked_prefill_integrity_gate(chunked_contract)
        ),
        "selector_switch_and_per_request_restore": _selector_gate(selectors),
        "exact_dispatch_and_no_fallback": _dispatch_gate(all_dispatch),
        "quality_arm_order_counterbalanced": pb._pair_order_gate(quality_pairs),
        "quality_repeat_digest_determinism": (
            pb._full_vocab_repeat_determinism_gate(
                quality_pairs,
                n_prompts=_PROFILE_SAMPLES,
                repeats=_QUALITY_REPEATS,
            )
        ),
        "full_vocab_cardinality_and_finiteness": _score_cardinality_gate(
            quality_pairs, vocab_size=bootstrap.candidate_vocab_size
        ),
        "fp8_dispatch_unchanged_between_arms": (
            _fp8_dispatch_invariance_gate(quality_pairs)
        ),
        "controller_final_restore": controller.restoration_gate(),
        "probe_final_restore_and_scope": probe.restoration_gate(),
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": v5._utc_now(),
        "scope": (
            "exact dsv4flash0731 target decode at M=16; TP=1; one eager "
            "in-process vLLM engine; inherited vs CB-GEMV-v2 boolean A/B"
        ),
        "arm_labels": dict(ARM_LABELS),
        "comparison_contract": {
            "weights": "identical packed FP4-CB bytes and resident codebooks",
            "activations": "identical native group-16 FP4 activation QDQ",
            "router_and_fp8_layers": "identical and dispatch-attested",
            "permitted_difference": (
                "none expected on the exact DSV4 K=2048/4096 "
                "default-schedule specializations; numeric gates are "
                "fail-closed corruption backstops"
            ),
            "same_contract": True,
        },
        "settings": {
            "model": args.model,
            "model_resolved": str(bootstrap.candidate_path.resolve()),
            "quantization": args.quantization,
            "dtype": args.dtype,
            "tokenizer_mode": args.tokenizer_mode,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "v1_multiprocessing": False,
            "prefix_caching": False,
            "vllm_moe_skip_padding": skip_padding_gate["observed"],
            "max_model_len": args.seqlen + 16,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup_pairs": args.warmup_pairs,
            "quality_repeats": _QUALITY_REPEATS,
            "quality_kl_mode": v5.KL_FULL_VOCAB,
            "quality_prompt_logprobs_request": -1,
            "candidate_vocab_size": bootstrap.candidate_vocab_size,
            "decode_profile": profile,
            "chunked_prefill_contract": chunked_contract,
            "seed": args.seed,
        },
        "environment_contract": {
            "inherited_before_sanitization": inherited,
            "during_model_load_and_measurement": {
                name: os.environ.get(name) for name in measured_envs
            },
            "inherited_w2_schedule_is_unset_default": True,
            "decode_contract": "v1",
            "selector": "v2",
            "measurement_exception": (
                "only the 70 captured target FP4 stack booleans are scoped; "
                "the process selector and extension residency never change"
            ),
        },
        "runtime": runtime,
        "extension": bootstrap.extension,
        "candidate_artifact_provenance": (
            bootstrap.candidate_artifact_provenance
        ),
        "dataset": bootstrap.dataset,
        "model_load_seconds": engine.model_load_seconds,
        "warmup": warmup_records,
        "quality": quality,
        "full_vocab_streaming_evidence": {
            "profile": profile,
            "pairs": quality_pairs,
            "storage_contract": (
                "vLLM FlatLogprobs -> paired float32 rows -> shared v5 "
                "incremental scorer -> scalar/digest evidence; raw rows "
                "released before the next pair"
            ),
        },
        "dispatch": {
            "layer_inventory": controller.inventory_gate,
            "request_attestations": all_dispatch,
            "selector_attestations": selectors,
        },
        "core_integrity_gates": core_gates,
        "limitations": [
            "This gate exercises grouped decode at exactly M=16, not prefill.",
            "Eager quality evidence does not prove CUDA-graph replay.",
            "Served speed, concurrency, long prefill, soak, and memory gates remain.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    configured = v5._configured_gates(args, report)
    report["configured_promotion_gates"] = configured
    report["measurement_only"] = bool(args.measurement_only)
    report["measurement_valid"] = all(
        bool(gate.get("pass")) for gate in core_gates.values()
    )
    report["configured_gates_pass"] = (
        None
        if args.measurement_only
        else all(bool(gate.get("pass")) for gate in configured.values())
    )
    present = [
        name for name in _PAIRWISE_QUALITY_GATES
        if getattr(args, name) is not None
    ]
    missing = [name for name in _PAIRWISE_QUALITY_GATES if name not in present]
    report["promotion_contract"] = {
        "exact_full_vocab_pairwise_quality": True,
        "same_process_without_reload": True,
        "exact_8x16_two_repeat_profile": True,
        "required_pairwise_quality_thresholds": list(_PAIRWISE_QUALITY_GATES),
        "present_pairwise_quality_thresholds": present,
        "missing_pairwise_quality_thresholds": missing,
        "served_graph_performance_concurrency_soak_still_required": True,
        "complete": not missing,
    }
    report["promotion_recommendation"] = (
        "measurement_failed"
        if not report["measurement_valid"]
        else "measurement_only_no_promotion_thresholds_configured"
        if args.measurement_only
        else "configured_gates_failed"
        if report["configured_gates_pass"] is False
        else "screening_only_all_pairwise_thresholds_required"
        if missing
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
        help="fixed DSV4 sparse-MLA KV-cache dtype",
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
        parser.error("CB-GEMV-v2 DSV4 validation requires a local --model")
    # Fixed fields consumed by the shared bootstrap/engine and threshold API.
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

    thresholds = [getattr(args, name) for name in _PAIRWISE_QUALITY_GATES]
    if not any(value is not None for value in thresholds) and not args.measurement_only:
        parser.error(
            "no thresholds configured; pass --measurement-only for evidence-only use"
        )
    if any(value is not None for value in thresholds) and args.measurement_only:
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
    if not report["measurement_valid"] or report["configured_gates_pass"] is False:
        report["status"] = "gate_failed"
    elif report["measurement_only"]:
        report["status"] = "measurement_only"
    elif not report["promotion_contract"]["complete"]:
        report["status"] = "screening_only"
    else:
        report["status"] = "ok"
    v5._atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_contract": report["promotion_contract"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_delta": report["quality"]["delta"],
        "layer_inventory": report["dispatch"]["layer_inventory"],
    }, indent=2), flush=True)
    return 0 if report["status"] in ("ok", "measurement_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
