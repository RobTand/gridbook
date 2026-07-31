"""Verify codebook sidecars against artifact-external provenance.

PrismaQuant exporters record one SHA-256 per sidecar tensor in
``quant_config.json["provenance"]["codebook_sha256"]``.  Keeping the expected
digest outside ``cb_codebooks.pqcb`` is essential: a digest stored only in the
file it describes can be updated along with wrong or stale table values and
therefore does not bind the sidecar to the rest of the model artifact.

This module deliberately has no vLLM dependency.  The serving config and the
read-only audit script both use the same verifier.
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from typing import TypeAlias

import torch

CodebookHashes: TypeAlias = Mapping[str, str]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def codebook_tensor_sha256(table: torch.Tensor) -> str:
    """Return the digest emitted by PrismaQuant's codebook exporters.

    The producer construction is SHA-256 over the tensor converted to a
    contiguous CPU ``float16`` array, in C order, with no name/shape framing.
    The mapping is per tensor, so framing multiple tables is neither needed nor
    compatible with already-published artifacts.

    ``ctypes.string_at`` copies the contiguous buffer without NumPy.  NumPy is
    intentionally not a gridbook dependency, and ``Tensor.numpy()`` is not
    available for every PyTorch dtype.
    """
    if not isinstance(table, torch.Tensor):
        raise TypeError("codebook table must be a torch.Tensor")
    data = table.detach().to(device="cpu", dtype=torch.float16).contiguous()
    nbytes = data.numel() * data.element_size()
    raw = ctypes.string_at(data.data_ptr(), nbytes) if nbytes else b""
    return hashlib.sha256(raw).hexdigest()


def _validated_hashes(expected: CodebookHashes) -> dict[str, str]:
    if not isinstance(expected, Mapping):
        raise ValueError(
            "provenance.codebook_sha256 must be an object mapping tensor "
            "names to lowercase SHA-256 digests")
    if not expected:
        raise ValueError(
            "provenance.codebook_sha256 is present but empty; omit the field "
            "for an unbound legacy artifact")

    validated: dict[str, str] = {}
    for name, digest in expected.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "provenance.codebook_sha256 contains an invalid tensor name")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(
                "provenance.codebook_sha256[%r] must be exactly 64 lowercase "
                "hexadecimal characters" % name)
        validated[name] = digest
    return validated


def verify_codebook_hashes(
    tables: Mapping[str, torch.Tensor],
    expected: CodebookHashes,
    *,
    source: str | os.PathLike[str] = "cb_codebooks.pqcb",
) -> None:
    """Verify a complete per-table provenance mapping.

    A present provenance mapping is required to cover the sidecar exactly.
    Silently accepting an unbound extra table would recreate the wrong-values
    hole for any scheme referencing that table.  Legacy configs remain
    supported by passing ``None`` to :func:`load_codebooks` instead.
    """
    want = _validated_hashes(expected)
    actual_names = set(tables)
    expected_names = set(want)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unbound = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append("missing from sidecar: " + ", ".join(missing))
        if unbound:
            details.append("not bound by provenance: " + ", ".join(unbound))
        raise ValueError(
            f"[prismaquant-cb] ERROR: codebook provenance for {source} does "
            f"not cover the sidecar exactly ({'; '.join(details)})")

    mismatches: list[tuple[str, str, str]] = []
    for name in sorted(want):
        got = codebook_tensor_sha256(tables[name])
        if not hmac.compare_digest(got, want[name]):
            mismatches.append((name, want[name], got))
    if mismatches:
        preview = "; ".join(
            f"{name}: expected {declared}, computed {computed}"
            for name, declared, computed in mismatches[:3])
        if len(mismatches) > 3:
            preview += f"; and {len(mismatches) - 3} more"
        raise ValueError(
            f"[prismaquant-cb] ERROR: codebook provenance mismatch for "
            f"{source} ({preview}). The sidecar is stale, corrupt, or belongs "
            f"to another artifact; refusing to decode with wrong tables.")


def load_codebooks(
    path: str | os.PathLike[str],
    expected_sha256: CodebookHashes | None = None,
) -> dict[str, torch.Tensor]:
    """Load one stable sidecar snapshot and optionally verify its provenance.

    ``safe_open`` opens the sidecar once.  Each table is cloned while that one
    handle is live, so verification and later decoding use the same in-memory
    bytes even if the path is atomically replaced after the load.  This is an
    accidental-corruption check, not an authenticity or hostile-filesystem
    boundary.

    ``expected_sha256=None`` is the backward-compatible legacy path.  A mapping
    that is present but malformed, incomplete, or mismatched fails closed.
    """
    from safetensors import safe_open

    path = os.fspath(path)
    # Reject malformed declarations before opening a config-controlled path.
    expected = (_validated_hashes(expected_sha256)
                if expected_sha256 is not None else None)
    with safe_open(path, framework="pt", device="cpu") as sidecar:
        tables = {
            name: sidecar.get_tensor(name).detach().clone().contiguous()
            for name in sidecar.keys()
        }
    if expected is not None:
        verify_codebook_hashes(tables, expected, source=path)
    return tables
