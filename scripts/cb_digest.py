#!/usr/bin/env python3
"""Stamp or verify the ``cb_tables_sha256`` digest of a ``.pqcb`` sidecar.

The runtime refuses a sidecar whose tables do not hash to the digest the file
itself declares (docs/SPEC.md 4.1, gridbook/cb_digest.py). This is the
producer / auditor side of that binding.

    # what does this file declare, and what do its tables actually hash to?
    python scripts/cb_digest.py show   cb_codebooks.pqcb

    # exit 0 if they agree (or nothing is declared), 1 if they do not
    python scripts/cb_digest.py verify cb_codebooks.pqcb

    # rewrite the file with the digest of its own tables stamped in
    python scripts/cb_digest.py stamp  cb_codebooks.pqcb

``stamp`` rewrites the container (safetensors metadata is not editable in
place), preserving every tensor bit-for-bit and every other metadata key. It
writes to a temporary file in the same directory and renames over the original
only after the rewrite has been read back and re-verified, so an interrupted
stamp cannot leave a half-written sidecar.

Uses only torch + safetensors — no vLLM, no GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running straight from a checkout without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridbook.cb_digest import (  # noqa: E402
    CB_DIGEST_META_KEY,
    codebook_digest,
    read_declared_digest,
)


def _read(path: str):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
    return load_file(path), meta


def cmd_show(args) -> int:
    tables, meta = _read(args.path)
    declared = meta.get(CB_DIGEST_META_KEY)
    print(f"path      : {args.path}")
    print(f"tables    : {len(tables)}")
    for n in sorted(tables):
        t = tables[n]
        print(f"  {n}  {str(t.dtype).replace('torch.', '')} {tuple(t.shape)}")
    print(f"declared  : {declared or '(none)'}")
    print(f"computed  : {codebook_digest(tables)}")
    other = {k: v for k, v in meta.items() if k != CB_DIGEST_META_KEY}
    if other:
        print(f"other meta: {sorted(other)}")
    return 0


def cmd_verify(args) -> int:
    tables, _ = _read(args.path)
    declared = read_declared_digest(args.path)
    computed = codebook_digest(tables)
    if declared is None:
        print(f"{args.path}: no {CB_DIGEST_META_KEY} declared "
              f"(computed {computed}) — nothing to verify against.")
        return 0
    if declared == computed:
        print(f"{args.path}: OK, tables match declared {CB_DIGEST_META_KEY} "
              f"{declared}")
        return 0
    print(f"{args.path}: MISMATCH — declared {declared}, computed {computed}",
          file=sys.stderr)
    return 1


def cmd_stamp(args) -> int:
    from safetensors.torch import save_file

    tables, meta = _read(args.path)
    computed = codebook_digest(tables)
    declared = meta.get(CB_DIGEST_META_KEY)
    if declared == computed:
        print(f"{args.path}: already stamped with {computed}")
        return 0
    if declared is not None and not args.force:
        print(f"{args.path}: refusing to overwrite an existing "
              f"{CB_DIGEST_META_KEY} ({declared}) that does not match the "
              f"tables ({computed}). This file is corrupt or mismatched — "
              f"re-stamping would launder it. Pass --force if you are certain "
              f"the tables are the intended ones.", file=sys.stderr)
        return 1

    meta[CB_DIGEST_META_KEY] = computed
    tmp = args.path + ".stamp.tmp"
    save_file(tables, tmp, metadata=meta)
    # Read back before replacing the original: a stamp that does not verify is
    # worse than no stamp at all.
    back_tables, back_meta = _read(tmp)
    ok = (back_meta.get(CB_DIGEST_META_KEY) == computed
          and codebook_digest(back_tables) == computed
          and sorted(back_tables) == sorted(tables))
    if not ok:
        os.unlink(tmp)
        print(f"{args.path}: rewrite did not verify; original left untouched",
              file=sys.stderr)
        return 1
    os.replace(tmp, args.path)
    print(f"{args.path}: stamped {CB_DIGEST_META_KEY}={computed}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
            ("show", cmd_show, "print declared and computed digests"),
            ("verify", cmd_verify, "exit non-zero on a mismatch"),
            ("stamp", cmd_stamp, "write the digest into the file")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("path", help="path to a .pqcb codebook sidecar")
        if name == "stamp":
            sp.add_argument("--force", action="store_true",
                            help="overwrite a mismatched existing digest")
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
