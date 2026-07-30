"""Integrity binding for the ``cb_codebooks.pqcb`` sidecar (docs/SPEC.md §4.1).

**Wrong codebook VALUES are undetectable at decode time.** A ``k``-bit codeword
indexes a ``2^k``-row table (SPEC §1.1), so every index is in range by
construction: substitute a table of the right names and shapes but the wrong
numbers and decode still produces a correctly-shaped, correctly-typed tensor —
filled with structured garbage. The server starts, the weights "load", and
generation is quietly wrong. That is the same silent-wrong-weights family
docs/TROUBLESHOOTING.md "The model loads but generates garbage" already warns
about.

The reachable cases are mundane: a ``.pqcb`` copied in from a *different*
checkpoint at the same rung; a stale sidecar left behind when a model was
re-encoded with re-fit codebooks; length-preserving data corruption (measured:
safetensors loads a byte-flipped payload without complaint, names and shapes
unchanged).

Structural damage is already caught and this module claims no credit for it:
byte-truncation raises ``SafetensorError: MetadataIncompleteBuffer`` in
safetensors itself, and a table missing by name raises downstream. Values are
the hole.

The sidecar therefore declares a digest of its own tables in its safetensors
``__metadata__`` header, and :func:`load_codebooks` refuses to serve a file that
does not hash to its own declaration.

Deliberately vLLM-free: this module is importable (and testable) without the
serving stack, so producer-side tooling can share exactly the runtime's digest
construction rather than reimplementing it. See ``scripts/cb_digest.py``.
"""
from __future__ import annotations

import hashlib
import os
import sys

import torch

# safetensors metadata key holding the digest of the tables the file carries.
# OPTIONAL: a sidecar without it still loads (every artifact published before
# the binding existed has no key).
CB_DIGEST_META_KEY = "cb_tables_sha256"

# Downgrades a digest mismatch to a warning. Debug only — it re-opens exactly
# the silent-wrong-weights hole the check exists to close.
SKIP_DIGEST_ENV = "PRISMAQUANT_SKIP_CB_DIGEST"


def codebook_digest(tables: dict[str, torch.Tensor]) -> str:
    """sha256 over the codebook tables (docs/SPEC.md §4.1, normative).

    For each tensor name in ``sorted()`` order, absorb the name as UTF-8, then
    ``str(dtype)``, then ``str(tuple(shape))``, then the tensor's raw contiguous
    C-order bytes.

    Sorting makes the digest independent of dict insertion order. Absorbing
    dtype and shape makes it a *layout* digest rather than merely a byte
    digest: a sidecar reshaped or re-typed under a fixed byte budget is a
    different codebook and must not hash the same.

    The raw bytes are taken through a ``uint8`` view rather than
    ``Tensor.numpy()`` only so that ``bfloat16`` tables can be hashed —
    ``numpy()`` raises ``TypeError: Got unsupported ScalarType BFloat16``. For
    every dtype numpy *does* support the two yield identical bytes (one
    contiguous buffer, read as-is), so this cannot change the digest of a
    sidecar an existing encoder already stamped.
    """
    h = hashlib.sha256()
    for n in sorted(tables):
        t = tables[n].detach().cpu().contiguous()
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_declared_digest(path: str) -> str | None:
    """The digest the ``.pqcb`` at *path* declares, or ``None`` if it declares
    none. Reads only the safetensors header, not the tensor data."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    return meta.get(CB_DIGEST_META_KEY)


def load_codebooks(path: str) -> dict[str, torch.Tensor]:
    """Load the codebook tables from *path*, bound to their declared digest.

    Backward-compatible by construction: the check can only fire on a file that
    asked to be checked. A sidecar with no declared digest loads exactly as
    before (a plain ``safetensors.torch.load_file``), with one informational
    line on stderr.

    Raises ``ValueError`` when a declared digest does not match the tables,
    unless ``PRISMAQUANT_SKIP_CB_DIGEST=1``.
    """
    from safetensors.torch import load_file

    tables = load_file(path)
    want = read_declared_digest(path)
    if want is None:
        print(f"[prismaquant-cb] {path}: no {CB_DIGEST_META_KEY} metadata "
              f"(sidecar predates digest binding) — integrity check skipped.",
              file=sys.stderr, flush=True)
        return tables
    got = codebook_digest(tables)
    if got == want:
        return tables
    if os.environ.get(SKIP_DIGEST_ENV) == "1":
        print(f"[prismaquant-cb] WARNING: {path}: {CB_DIGEST_META_KEY} "
              f"mismatch (declared {want[:16]}..., computed {got[:16]}...) ignored "
              f"because {SKIP_DIGEST_ENV}=1. Output is not trustworthy.",
              file=sys.stderr, flush=True)
        return tables
    raise ValueError(
        f"[prismaquant-cb] ERROR: {path}: the codebook tables hash to {got} "
        f"but the file's own {CB_DIGEST_META_KEY} metadata declares {want} — "
        f"this sidecar is corrupted, or belongs to a different checkpoint or a "
        f"different encode of this one. Serving it would decode every codebook "
        f"weight to structured garbage, so it is refused. Re-download the "
        f"artifact; set {SKIP_DIGEST_ENV}=1 to downgrade this to a warning "
        f"(debug only — the resulting output is not trustworthy).")
