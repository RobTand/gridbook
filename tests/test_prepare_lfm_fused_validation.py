"""CPU-only tests for the small LFM fused-NVFP4 validation fixture."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
from array import array
from pathlib import Path

import pytest


def _script_path(name: str) -> Path:
    # Release jobs stage tests outside the checkout to exercise the installed
    # wheel; the validation scripts remain source utilities in that checkout.
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    candidates = [root / "scripts" / name for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not locate {name!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


SCRIPT = _script_path("prepare_lfm_fused_validation.py")
SPEC = importlib.util.spec_from_file_location("prepare_lfm_fused_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class _Slice:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _Checkpoint:
    def __init__(self, tensors):
        self.tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return self.tensors.keys()

    def get_slice(self, name):
        return _Slice(self.tensors[name])


def test_prepare_supports_per_expert_hf_layout(tmp_path, monkeypatch):
    model = tmp_path / "model"
    output = tmp_path / "inputs"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"lfm2_moe"}\n')
    (model / "model.safetensors").write_bytes(b"fixture")
    tensors = {}
    for expert in range(2):
        parent = f"model.layers.2.feed_forward.experts.{expert}"
        tensors[f"{parent}.w1.weight"] = (1792, 2048)
        tensors[f"{parent}.w3.weight"] = (1792, 2048)
        tensors[f"{parent}.w2.weight"] = (2048, 1792)
    monkeypatch.setattr(prep, "safe_open", lambda *_args, **_kwargs: _Checkpoint(tensors))

    manifest = prep.prepare(model, output, (2,), "NVFP4_CB_K16")
    assignment = json.loads((output / "layer_config.json").read_text())
    with (output / "uniform_col_weights.pkl").open("rb") as handle:
        weights = pickle.load(handle)

    gate_up = "model.layers.2.feed_forward.experts.gate_up_proj"
    down = "model.layers.2.feed_forward.experts.down_proj"
    assert assignment == {gate_up: "NVFP4_CB_K16", down: "NVFP4_CB_K16"}
    assert manifest["targets"][gate_up] == [2, 3584, 2048]
    assert manifest["targets"][down] == [2, 2048, 1792]
    assert isinstance(weights[gate_up], array)
    assert weights[gate_up].typecode == "f"
    assert len(weights[gate_up]) == 2048
    assert len(weights[down]) == 1792
    assert set(weights[gate_up]) == {1.0}
    assert set(weights[down]) == {1.0}
    torch = pytest.importorskip("torch")
    assert tuple(torch.as_tensor(weights[gate_up]).shape) == (2048,)
    assert torch.as_tensor(weights[gate_up]).dtype == torch.float32
    assert manifest["source_config_sha256"] == prep._sha256(model / "config.json")
    assert manifest["source_model_sha256"] == prep._sha256(
        model / "model.safetensors"
    )

    original_hash = manifest["source_model_sha256"]
    original_bytes = manifest["source_model_bytes"]
    (model / "model.safetensors").write_bytes(b"changed")
    changed_manifest = prep.prepare(
        model, tmp_path / "changed-inputs", (2,), "NVFP4_CB_K16"
    )
    assert changed_manifest["source_model_bytes"] == original_bytes
    assert changed_manifest["source_model_sha256"] != original_hash


def test_parse_layers_is_sorted_unique_and_rejects_bad_values():
    assert prep.parse_layers("23,2,8,2") == (2, 8, 23)
    with pytest.raises(Exception):
        prep.parse_layers("")
    with pytest.raises(Exception):
        prep.parse_layers("-1,2")
