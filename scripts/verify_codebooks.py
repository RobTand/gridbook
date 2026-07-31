#!/usr/bin/env python3
"""Verify a local model's codebook sidecar against quant_config provenance.

This auditor is intentionally read-only.  It never stamps or rewrites a
sidecar: deriving a new digest from the file under inspection would merely make
an arbitrary or stale sidecar self-consistent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    # Prefer the installed package.  CI deliberately runs outside the checkout
    # so its wheel, rather than ./gridbook, is what this exercises.
    from gridbook.cb_digest import load_codebooks
except ModuleNotFoundError as exc:
    if exc.name not in {"gridbook", "gridbook.cb_digest"}:
        raise
    # Still support invoking the script directly from an uninstalled checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.modules.pop("gridbook", None)
    from gridbook.cb_digest import load_codebooks


def verify_model(model_dir: Path, config_name: str = "quant_config.json") -> int:
    config_path = model_dir / config_name
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"{config_path}: cannot read quant config: {exc}", file=sys.stderr)
        return 1

    if not isinstance(config, dict):
        print(f"{config_path}: quant config must be a JSON object",
              file=sys.stderr)
        return 1
    provenance = config.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        print(f"{config_path}: provenance must be a JSON object",
              file=sys.stderr)
        return 1
    expected = (provenance or {}).get("codebook_sha256")
    if expected is None:
        if provenance is not None and "codebook_sha256" in provenance:
            print(
                f"{config_path}: provenance.codebook_sha256 must be a JSON "
                "object, not null", file=sys.stderr)
            return 1
        print(
            f"{config_path}: no provenance.codebook_sha256 mapping; this "
            "legacy artifact has no external codebook binding",
            file=sys.stderr)
        return 2

    sidecar_name = config.get("codebook_file", "cb_codebooks.pqcb")
    if not isinstance(sidecar_name, str) or not sidecar_name:
        print(f"{config_path}: codebook_file must be a non-empty string",
              file=sys.stderr)
        return 1
    sidecar_path = model_dir / sidecar_name
    try:
        tables = load_codebooks(sidecar_path, expected_sha256=expected)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"{sidecar_path}: OK, verified {len(tables)} codebook table(s) "
        f"against {config_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=Path,
                        help="local model directory containing both sidecars")
    parser.add_argument("--config", default="quant_config.json",
                        help="quant config filename relative to MODEL_DIR")
    args = parser.parse_args(argv)
    return verify_model(args.model_dir, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
