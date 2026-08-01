#!/usr/bin/env python3
"""Prepare a small expert-only LFM2 MoE assignment for fused-NVFP4 A/Bs.

The generated column weights are deliberately uniform: this is a controlled
kernel/activation validation artifact, not a production-quality allocation.
All non-selected tensors remain BF16 in the exported checkpoint so the local
BF16 model remains an informative, fit-able teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from array import array
from pathlib import Path

from safetensors import safe_open


PROJECTIONS = ("gate_up_proj", "down_proj")
_HASH_BLOCK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("at least one nonnegative layer is required")
    return layers


def prepare(
    model_dir: Path,
    output_dir: Path,
    layers: tuple[int, ...],
    format_name: str,
) -> dict:
    model_file = model_dir / "model.safetensors"
    config_file = model_dir / "config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError("expected config.json and model.safetensors")

    assignment: dict[str, str] = {}
    col_weights: dict[str, array] = {}
    shapes: dict[str, list[int]] = {}
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        for layer in layers:
            parent = f"model.layers.{layer}.feed_forward.experts"
            for projection in PROJECTIONS:
                qname = f"{parent}.{projection}"
                if qname in available:
                    shape = tuple(
                        int(v) for v in checkpoint.get_slice(qname).get_shape()
                    )
                else:
                    # Standard HF LFM checkpoints store one w1/w2/w3 tensor
                    # per expert; the exporter assembles the same packed
                    # gate_up/down tensors that vLLM owns.
                    leaves = ("w1", "w3") if projection == "gate_up_proj" else ("w2",)
                    expert = 0
                    member_shapes = []
                    while f"{parent}.{expert}.{leaves[0]}.weight" in available:
                        member_shapes.append([
                            tuple(int(v) for v in checkpoint.get_slice(
                                f"{parent}.{expert}.{leaf}.weight"
                            ).get_shape())
                            for leaf in leaves
                        ])
                        expert += 1
                    if not member_shapes:
                        raise KeyError(
                            f"missing packed and per-expert sources for {qname!r}"
                        )
                    first = member_shapes[0]
                    if any(group != first for group in member_shapes):
                        raise ValueError(f"{qname}: experts have inconsistent shapes")
                    in_features = first[0][1]
                    if any(member[1] != in_features for member in first):
                        raise ValueError(f"{qname}: fused siblings disagree on input width")
                    shape = (expert, sum(member[0] for member in first), in_features)
                if len(shape) != 3 or shape[-1] % 256:
                    raise ValueError(
                        f"{qname}: expected [E,N,K] with K divisible by 256, got {shape}"
                    )
                assignment[qname] = format_name
                col_weights[qname] = array("f", [1.0]) * shape[-1]
                shapes[qname] = list(shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "layer_config.json"
    weights_path = output_dir / "uniform_col_weights.pkl"
    manifest_path = output_dir / "prepare_manifest.json"
    assignment_path.write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with weights_path.open("wb") as handle:
        pickle.dump(col_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": "gridbook.lfm-fused-validation-inputs.v1",
        "purpose": "diagnostic only; uniform column weights, selected experts only",
        "source_model": str(model_dir.resolve()),
        "source_config_sha256": _sha256(config_file),
        "source_model_sha256": _sha256(model_file),
        "source_model_bytes": model_file.stat().st_size,
        "layers": list(layers),
        "format": format_name,
        "targets": shapes,
        "layer_config_sha256": _sha256(assignment_path),
        "col_weights_sha256": _sha256(weights_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("2,8,14,23"))
    parser.add_argument("--format", default="NVFP4_CB_K16")
    args = parser.parse_args()
    manifest = prepare(args.model_dir, args.output_dir, args.layers, args.format)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
