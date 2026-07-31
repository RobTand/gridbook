"""CPU coverage for external per-table codebook provenance."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from gridbook.cb_digest import (  # noqa: E402
    codebook_tensor_sha256,
    load_codebooks,
    verify_codebook_hashes,
)

_SUB0 = "cb_codebook.lattice.NVFP4_CB_K16.sub0"
_SUB1 = "cb_codebook.lattice.NVFP4_CB_K16.sub1"


def _audit_script() -> Path:
    # CI deliberately stages tests outside the checkout to exercise the wheel;
    # GITHUB_WORKSPACE is the authoritative location of repository scripts.
    root = Path(os.environ.get(
        "GITHUB_WORKSPACE", Path(__file__).resolve().parents[1]))
    script = root / "scripts" / "verify_codebooks.py"
    if not script.is_file():
        pytest.skip("repository audit script is not available")
    return script


def _tables(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        _SUB0: torch.randn(16, 8, generator=generator).to(torch.float16),
        _SUB1: torch.randn(16, 8, generator=generator).to(torch.float16),
    }


def _hashes(tables: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: codebook_tensor_sha256(table)
            for name, table in tables.items()}


def _self_digest_from_submitted_patch(
        tables: dict[str, torch.Tensor]) -> str:
    """Reproduce the aggregate digest that bundle 02 originally proposed."""
    digest = hashlib.sha256()
    for name in sorted(tables):
        table = tables[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(table.dtype).encode())
        digest.update(str(tuple(table.shape)).encode())
        digest.update(bytes(table.view(torch.uint8).reshape(-1)))
    return digest.hexdigest()


def _write(directory: Path, tables: dict[str, torch.Tensor], *,
           metadata: dict[str, str] | None = None) -> Path:
    """Write the tiny F16 fixture without safetensors' optional NumPy writer.

    Gridbook only reads sidecars, and NumPy is deliberately absent from its
    runtime dependencies. Building fixtures through ``save_file`` would make
    this suite stricter than the installed package: safetensors currently
    imports NumPy only on its write path.
    """
    path = directory / "cb_codebooks.pqcb"
    header: dict[str, object] = {}
    payload = bytearray()
    for name, table in tables.items():
        data = table.detach().to(device="cpu", dtype=torch.float16).contiguous()
        raw = bytes(data.view(torch.uint8).reshape(-1).tolist())
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "F16",
            "shape": list(data.shape),
            "data_offsets": [start, len(payload)],
        }
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def test_fixed_vector_matches_exporter_bytes():
    table = torch.tensor([0.0, 1.0, -2.0, 0.5], dtype=torch.float32)
    assert struct.pack("<4e", 0.0, 1.0, -2.0, 0.5).hex() == \
        "0000003c00c00038"
    assert codebook_tensor_sha256(table) == \
        "58e0b63dabbfe62f04be297478992b1c1d301fc4c3c4c1c8abe3f431b322cce5"


def test_hash_matches_exporter_conversion_and_needs_no_numpy(monkeypatch):
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    assert not source.is_contiguous()
    expected_bytes = struct.pack(
        "<24e", *source.to(torch.float16).contiguous().reshape(-1).tolist())
    expected = hashlib.sha256(expected_bytes).hexdigest()

    def forbidden_numpy(self):  # pragma: no cover - only called on regression
        raise AssertionError("NumPy must not be required for provenance")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    assert codebook_tensor_sha256(source) == expected
    # The published producer deliberately normalizes dtype to fp16.
    assert codebook_tensor_sha256(source.to(torch.float64)) == expected


def test_empty_tensor_has_standard_empty_sha256():
    assert codebook_tensor_sha256(torch.empty(0)) == hashlib.sha256(b"").hexdigest()


def test_matching_external_mapping_loads_from_one_open(tmp_path, monkeypatch):
    tables = _tables()
    path = _write(tmp_path, tables)

    import safetensors
    real_safe_open = safetensors.safe_open
    opens = []

    def counted_safe_open(*args, **kwargs):
        opens.append(args[0])
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counted_safe_open)
    got = load_codebooks(path, _hashes(tables))
    assert opens == [str(path)]
    assert sorted(got) == sorted(tables)
    assert all(torch.equal(got[name], tables[name]) for name in tables)


def test_absent_mapping_is_backward_compatible(tmp_path):
    tables = _tables()
    got = load_codebooks(_write(tmp_path, tables), expected_sha256=None)
    assert all(torch.equal(got[name], tables[name]) for name in tables)


def test_intact_wrong_sidecar_with_matching_self_digest_is_refused(tmp_path):
    """A self-digest cannot bind a sidecar to the intended artifact."""
    intended = _tables(seed=1)
    wrong = _tables(seed=2)
    wrong_self_digest = _self_digest_from_submitted_patch(wrong)
    path = _write(
        tmp_path, wrong, metadata={"cb_tables_sha256": wrong_self_digest})

    with pytest.raises(ValueError, match="provenance mismatch") as exc:
        load_codebooks(path, _hashes(intended))
    assert _SUB0 in str(exc.value)


def test_length_preserving_value_change_is_refused(tmp_path):
    intended = _tables()
    changed = {name: table.clone() for name, table in intended.items()}
    changed[_SUB1][3, 4] += 1
    path = _write(tmp_path, changed)
    with pytest.raises(ValueError, match="stale, corrupt"):
        load_codebooks(path, _hashes(intended))


@pytest.mark.parametrize("declared, message", [
    ({_SUB0}, "not bound by provenance"),
    ({_SUB0, _SUB1, "cb.missing"}, "missing from sidecar"),
])
def test_present_mapping_must_cover_sidecar_exactly(tmp_path, declared, message):
    tables = _tables()
    all_hashes = _hashes(tables)
    expected = {name: all_hashes.get(name, "0" * 64) for name in declared}
    with pytest.raises(ValueError, match=message):
        load_codebooks(_write(tmp_path, tables), expected)


@pytest.mark.parametrize("expected, message", [
    ({}, "present but empty"),
    ([], "must be an object"),
    ({_SUB0: "A" * 64}, "lowercase"),
    ({_SUB0: "0" * 63}, "exactly 64"),
    ({_SUB0: 123}, "exactly 64"),
    ({"": "0" * 64}, "invalid tensor name"),
])
def test_malformed_present_mapping_fails_closed(expected, message):
    with pytest.raises(ValueError, match=message):
        verify_codebook_hashes({_SUB0: torch.zeros(1)}, expected)


def test_malformed_mapping_is_rejected_before_path_is_opened(monkeypatch):
    import safetensors
    monkeypatch.setattr(
        safetensors, "safe_open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("malformed provenance must fail before file I/O")))
    with pytest.raises(ValueError, match="present but empty"):
        load_codebooks("attacker-controlled.pqcb", {})


def test_loaded_tables_are_cloned_before_handle_closes(monkeypatch):
    original = torch.tensor([1.0, 2.0], dtype=torch.float16)
    mapped = original.clone()

    class MutatingHandle:
        def __enter__(self):
            return self

        def keys(self):
            return [_SUB0]

        def get_tensor(self, name):
            assert name == _SUB0
            return mapped

        def __exit__(self, *_exc):
            mapped.fill_(99)

    import safetensors
    monkeypatch.setattr(safetensors, "safe_open",
                        lambda *_args, **_kwargs: MutatingHandle())
    got = load_codebooks("replaced-after-open.pqcb",
                         {_SUB0: codebook_tensor_sha256(original)})
    assert torch.equal(got[_SUB0], original)
    assert not torch.equal(got[_SUB0], mapped)


def test_read_only_cli_verifies_and_never_offers_stamp(tmp_path):
    tables = _tables()
    _write(tmp_path, tables)
    (tmp_path / "quant_config.json").write_text(json.dumps({
        "codebook_file": "cb_codebooks.pqcb",
        "provenance": {"codebook_sha256": _hashes(tables)},
    }), encoding="utf-8")
    script = _audit_script()
    victim = tmp_path / "must-not-be-overwritten"
    victim.write_text("sentinel", encoding="utf-8")
    old_stamp_path = tmp_path / "cb_codebooks.pqcb.stamp.tmp"
    old_stamp_path.symlink_to(victim)

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "verified 2 codebook table(s)" in result.stdout
    assert victim.read_text(encoding="utf-8") == "sentinel"
    assert old_stamp_path.is_symlink()

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True, capture_output=True, check=False)
    assert help_result.returncode == 0
    assert "stamp" not in help_result.stdout.lower()


def test_read_only_cli_rejects_mismatch_and_reports_unbound_legacy(tmp_path):
    tables = _tables()
    _write(tmp_path, tables)
    config_path = tmp_path / "quant_config.json"
    script = _audit_script()

    config_path.write_text(json.dumps({
        "provenance": {"codebook_sha256": {
            name: "0" * 64 for name in tables}},
    }), encoding="utf-8")
    mismatch = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True, capture_output=True, check=False)
    assert mismatch.returncode == 1
    assert "provenance mismatch" in mismatch.stderr

    config_path.write_text("{}", encoding="utf-8")
    legacy = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True, capture_output=True, check=False)
    assert legacy.returncode == 2
    assert "has no external codebook binding" in legacy.stderr


def test_digest_module_imports_no_vllm():
    check = subprocess.run([
        sys.executable, "-c",
        "import sys, gridbook.cb_digest; "
        "assert not any(n == 'vllm' or n.startswith('vllm.') "
        "for n in sys.modules)",
    ], text=True, capture_output=True, check=False)
    assert check.returncode == 0, check.stderr
