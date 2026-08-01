"""Integration coverage through PrismaQuantConfig.get_codebooks().

The tests are CPU-only.  Minimal vLLM symbols are stubbed when the serving
stack is unavailable so the real lazy config/path/memoization code executes.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")


def _install_vllm_stubs() -> None:
    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value

    module("vllm")
    config_module = module("vllm.config")
    module("vllm.model_executor")
    module("vllm.model_executor.layers")
    module("vllm.model_executor.layers.quantization")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    base = module("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def __init__(self):
            pass

    base.QuantizationConfig = QuantizationConfig
    base.QuantizeMethodBase = object
    embedding = module(
        "vllm.model_executor.layers.vocab_parallel_embedding")
    embedding.UnquantizedEmbeddingMethod = type(
        "UnquantizedEmbeddingMethod", (), {})
    embedding.VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})
    fused_moe = module("vllm.model_executor.layers.fused_moe")
    fused_moe.RoutedExperts = type("RoutedExperts", (), {})
    config_module.get_current_vllm_config = lambda: None


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    """Import Gridbook against a private real-or-stubbed vLLM graph."""
    del isolated_gridbook_runtime_imports
    try:
        import vllm.config as config_module
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,  # noqa: F401
        )
    except Exception:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                sys.modules.pop(name, None)
        _install_vllm_stubs()
        import vllm.config as config_module

    from gridbook.cb_digest import codebook_tensor_sha256 as digest
    from gridbook.config import PrismaQuantConfig as config_class

    globals()["vllm_config"] = config_module
    globals()["codebook_tensor_sha256"] = digest
    globals()["PrismaQuantConfig"] = config_class

_NAME = "cb_codebook.lattice.NVFP4_CB_K16.sub0"


def _table(value: float = 1.0) -> torch.Tensor:
    return torch.full((16, 8), value, dtype=torch.float16)


def _write_sidecar(path: Path, value: float = 1.0) -> torch.Tensor:
    table = _table(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes(table.view(torch.uint8).reshape(-1).tolist())
    header = json.dumps({
        _NAME: {
            "dtype": "F16",
            "shape": list(table.shape),
            "data_offsets": [0, len(payload)],
        },
    }, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return table


def _full_config(expected: str | None, *, codebook_file="cb_codebooks.pqcb"):
    config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "codebook_file": codebook_file,
        "config_groups": {
            "cb": {
                "targets": ["model.layers.0.mlp.down_proj"],
                "scheme": {"grid": "fp4", "mode": "product", "k": 16},
            },
        },
        "ignore": [],
    }
    if expected is not None:
        config["provenance"] = {"codebook_sha256": {_NAME: expected}}
    return config


def _set_model(
    monkeypatch,
    model: str,
    *,
    resolved_commit: str | None = None,
    revision: str | None = None,
):
    model_config = types.SimpleNamespace(
        model=model,
        hf_config=types.SimpleNamespace(_commit_hash=resolved_commit),
        revision=revision,
    )
    current = types.SimpleNamespace(model_config=model_config)
    monkeypatch.setattr(
        vllm_config, "get_current_vllm_config", lambda: current)
    return model_config


def test_inline_config_verifies_then_memoizes(tmp_path, monkeypatch):
    sidecar = tmp_path / "cb_codebooks.pqcb"
    table = _write_sidecar(sidecar)
    _set_model(monkeypatch, str(tmp_path))
    cfg = PrismaQuantConfig.from_config(
        _full_config(codebook_tensor_sha256(table)))

    first = cfg.get_codebooks()
    assert torch.equal(first[_NAME], table)

    # A second call must not re-resolve or re-open the path.
    _write_sidecar(sidecar, value=99.0)
    monkeypatch.setattr(
        vllm_config, "get_current_vllm_config",
        lambda: (_ for _ in ()).throw(AssertionError("not memoized")))
    assert cfg.get_codebooks() is first
    assert torch.equal(first[_NAME], table)


def test_pointer_config_resolves_hashes_and_sidecar_path(tmp_path, monkeypatch):
    relative_sidecar = "tables/codebooks.pqcb"
    table = _write_sidecar(tmp_path / relative_sidecar)
    (tmp_path / "quant_config.json").write_text(json.dumps(
        _full_config(codebook_tensor_sha256(table),
                     codebook_file=relative_sidecar)), encoding="utf-8")
    _set_model(
        monkeypatch,
        str(tmp_path),
        resolved_commit="ignored-for-local-model",
        revision="also-ignored-for-local-model",
    )

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("local sidecars must not use the Hub"),
    )

    cfg = PrismaQuantConfig.from_config({"config_file": "quant_config.json"})
    got = cfg.get_codebooks()
    assert cfg.codebook_file == relative_sidecar
    assert torch.equal(got[_NAME], table)
    assert cfg._sidecar_source == (str(tmp_path), None)


@pytest.mark.parametrize(
    "resolved_commit,requested_revision,expected_revision",
    [
        ("B" * 40, "mutable-release-tag", "b" * 40),
        (None, "A" * 40, "a" * 40),
        ("malformed-resolved-commit", "C" * 40, "c" * 40),
    ],
)
def test_hub_id_pins_both_external_files_to_one_model_revision(
    tmp_path,
    monkeypatch,
    resolved_commit,
    requested_revision,
    expected_revision,
):
    config_path = tmp_path / "downloaded-quant-config.json"
    sidecar_path = tmp_path / "downloaded-codebooks.pqcb"
    table = _write_sidecar(sidecar_path)
    config_path.write_text(json.dumps(
        _full_config(codebook_tensor_sha256(table))), encoding="utf-8")
    model_config = _set_model(
        monkeypatch,
        "owner/model",
        resolved_commit=resolved_commit,
        revision=requested_revision,
    )

    import huggingface_hub
    calls = []

    def fake_download(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        # The selected source must be cached before the first lazy download;
        # a moving tag/config cannot change the codebook's revision afterward.
        model_config.hf_config._commit_hash = "moved-after-first-download"
        model_config.revision = "also-moved-after-first-download"
        return str({
            "quant_config.json": config_path,
            "cb_codebooks.pqcb": sidecar_path,
        }[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cfg = PrismaQuantConfig.from_config({"config_file": "quant_config.json"})
    assert torch.equal(cfg.get_codebooks()[_NAME], table)
    assert calls == [
        ("owner/model", "quant_config.json", expected_revision),
        ("owner/model", "cb_codebooks.pqcb", expected_revision),
    ]


@pytest.mark.parametrize(
    "resolved_commit,requested_revision",
    [
        (None, None),
        (None, ""),
        (None, "main"),
        (None, "release/v1"),
        (None, "a" * 39),
        (None, "g" * 40),
        ("malformed-resolved-commit", "main"),
        ("f" * 39, "release/v1"),
    ],
)
def test_unpinned_or_mutable_hub_sidecars_fail_before_download(
    monkeypatch, resolved_commit, requested_revision
):
    _set_model(
        monkeypatch,
        "owner/unpinned",
        resolved_commit=resolved_commit,
        revision=requested_revision,
    )

    import huggingface_hub
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: calls.append(kwargs),
    )
    cfg = PrismaQuantConfig.from_config({"config_file": "quant_config.json"})
    with pytest.raises(
        RuntimeError, match="has no immutable revision for Gridbook sidecars"
    ):
        cfg.get_codebooks()
    assert calls == []


def test_config_without_hash_mapping_remains_loadable(tmp_path, monkeypatch):
    table = _write_sidecar(tmp_path / "cb_codebooks.pqcb")
    _set_model(monkeypatch, str(tmp_path))
    cfg = PrismaQuantConfig.from_config(_full_config(None))
    assert torch.equal(cfg.get_codebooks()[_NAME], table)


def test_get_codebooks_refuses_stale_sidecar(tmp_path, monkeypatch):
    intended = _table(1.0)
    _write_sidecar(tmp_path / "cb_codebooks.pqcb", value=2.0)
    _set_model(monkeypatch, str(tmp_path))
    cfg = PrismaQuantConfig.from_config(
        _full_config(codebook_tensor_sha256(intended)))
    with pytest.raises(ValueError, match="provenance mismatch"):
        cfg.get_codebooks()
    assert cfg._codebooks is None


@pytest.mark.parametrize("provenance, message", [
    ("invalid", "provenance must be an object"),
    ({"codebook_sha256": None}, "must be an object when declared"),
    ({"codebook_sha256": {}}, "present but empty"),
    ({"codebook_sha256": {_NAME: "bad"}}, "exactly 64"),
])
def test_malformed_config_provenance_fails_closed(
        tmp_path, monkeypatch, provenance, message):
    _write_sidecar(tmp_path / "cb_codebooks.pqcb")
    _set_model(monkeypatch, str(tmp_path))
    config = _full_config(None)
    config["provenance"] = provenance
    cfg = PrismaQuantConfig.from_config(config)
    with pytest.raises(ValueError, match=message):
        cfg.get_codebooks()
