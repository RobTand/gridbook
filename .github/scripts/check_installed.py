#!/usr/bin/env python3
"""Post-install gate: run against an INSTALLED gridbook, never a checkout.

CI installs the built wheel with ``--no-deps`` into a clean interpreter and runs
this.  Two properties are asserted that a file-listing check on the wheel cannot
prove:

1. ``import gridbook`` works with **no torch, no triton, no vLLM installed**.
   ``gridbook/__init__.py`` keeps ``register()`` lazy precisely so the package
   imports on a GPU-less machine; if that regresses, ``pip install gridbook``
   starts pulling multi-GB wheels just to be importable and every no-GPU
   consumer breaks.
2. The packaged CUDA sources resolve **from site-packages**, through the same
   ``importlib.resources`` lookup the runtime JIT builder uses.  The historical
   bug was a repo-root-relative ``os.pardir`` join that resolved to
   ``<site-packages>/csrc`` (nonexistent). Older releases silently selected a
   slow Triton fallback; the native-only runtime now fails the affected CB
   operator closed.

Deliberately NOT asserted here: that the extension compiles.  That needs nvcc
and is a manual pre-release gate -- see docs/RELEASING.md.

Usage:  python .github/scripts/check_installed.py
Run it from a directory that is NOT the repo root, or the checkout will shadow
site-packages and the check becomes meaningless (this is asserted below).
"""
from __future__ import annotations

import importlib.metadata as md
import os
import pathlib
import sys
from importlib.resources import files

PKG = "gridbook"
ENTRY_POINT_GROUP = "vllm.general_plugins"

# The runtime-required floor. Keep in lockstep with check_dist.py's REQUIRED
# and tests/test_release_metadata.py's WHEEL_REQUIRED — three mirrors of one
# list, and the 2026-08-02 reconciliation found this one a wave behind.
REQUIRED_SOURCES = [
    "csrc/cb_gemv.cu",
    "csrc/cb_gemv_v2.cu",
    "csrc/cb_bf16_grouped_gemm.cu",
    "csrc/cb_fused_gemm.cu",
    "csrc/cb_fused_fp4_gemm.cu",
    "csrc/cb_fused_fp4v2_gemm.cu",
    "csrc/cb_moe_persistent_b.cu",
    # Shared by all four fused/grouped loaders (EVT trees, AssertSmemFits, the
    # grouped host validation). Missing from the wheel until 2026-08-02.
    "csrc/cb_grouped_common.hpp",
    "csrc/cutlass_fork/sm120_cb_mma_tma.hpp",
    "csrc/cutlass_fork/sm120_cb_fused_mma.hpp",
    "csrc/cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    "csrc/cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp",
    "csrc/cutlass_fork/sm120_bf16_expert_mma.hpp",
    "csrc/cutlass_fork/sm120_expert_row_broadcast.hpp",
]

_errors: list[str] = []


def err(msg: str) -> None:
    _errors.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def _uses_os_pardir(path: pathlib.Path) -> bool:
    """True if the module contains a live ``os.pardir`` *expression*.

    Token-based, not a substring search: cuda_ext.py's own docstring says "do
    not reintroduce os.pardir", and a naive `in` test flags that warning as the
    defect it warns about.
    """
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "pardir"
                and isinstance(node.value, ast.Name) and node.value.id == "os"):
            return True
    return False


def main() -> int:
    # -- the package must come from the install, not from a sibling checkout ---
    import gridbook

    origin = pathlib.Path(gridbook.__file__).resolve()
    print(f"      gridbook.__file__ = {origin}")
    if "site-packages" not in origin.parts and "dist-packages" not in origin.parts:
        err(f"gridbook was imported from {origin}, not from site-packages. "
            f"cwd={os.getcwd()} is shadowing the installed package; this check "
            f"proves nothing when run from the repo root.")
        return 1
    ok("imported from site-packages")

    # -- import must not require torch/triton/vLLM ---------------------------
    for heavy in ("torch", "triton", "vllm"):
        if heavy in sys.modules:
            err(f"importing {PKG} pulled in {heavy!r}; register() must stay lazy "
                f"so the package imports on a machine with no GPU stack")
    if not _errors:
        ok("import pulled in none of torch/triton/vllm")

    version = getattr(gridbook, "__version__", None)
    dist_version = md.version(PKG)
    if version is None:
        err("gridbook.__version__ is missing")
    elif version != dist_version:
        err(f"gridbook.__version__ ({version}) != installed dist version "
            f"({dist_version}); the two version literals have drifted")
    else:
        ok(f"version {version} agrees with dist metadata")

    # -- entry point discoverable and loadable -------------------------------
    eps = [e for e in md.entry_points(group=ENTRY_POINT_GROUP) if e.name == PKG]
    if not eps:
        found = sorted(e.name for e in md.entry_points(group=ENTRY_POINT_GROUP))
        err(f"no {ENTRY_POINT_GROUP!r} entry point named {PKG!r} "
            f"(group contains: {found}); vLLM would never load the plugin")
    else:
        ep = eps[0]
        ok(f"entry point {ENTRY_POINT_GROUP}:{ep.name} = {ep.value}")
        try:
            fn = ep.load()  # loads gridbook:register -- does NOT call it
        except Exception as exc:  # noqa: BLE001
            err(f"entry point failed to load: {exc!r}")
        else:
            if not callable(fn):
                err(f"entry point resolved to a non-callable: {fn!r}")
            else:
                ok("entry point loads to a callable (not invoked: needs vLLM)")

    # -- packaged CUDA sources resolve from the installed package ------------
    csrc = files(PKG) / "csrc"
    print(f"      csrc = {csrc}")
    if not csrc.is_dir():
        err(f"{csrc} is not a directory. The wheel did not ship "
            f"{PKG}/csrc -- every CUDA extension build will fail with "
            f"FileNotFoundError and native CB execution will fail closed.")
    else:
        missing = [r for r in REQUIRED_SOURCES if not (files(PKG) / r).is_file()]
        if missing:
            err(f"packaged sources missing from the install: {missing}")
        else:
            ok(f"all {len(REQUIRED_SOURCES)} runtime-required sources resolve "
               f"from site-packages")

    # -- the runtime resolver ------------------------------------------------
    # NOT torch-gated. Every torch import inside cuda_ext is function-local
    # (inside the JIT loader functions), so the module imports and csrc_dir()
    # resolves in a bare `--no-deps` install -- measured in a torch-less venv,
    # 2026-07-28. Gating this on torch would have made the check vacuous in
    # exactly the install job most likely to be missing packaged sources.
    from gridbook import cuda_ext

    resolver = getattr(cuda_ext, "csrc_dir", None)
    if resolver is None:
        err("gridbook.cuda_ext has no csrc_dir(); the JIT builder is still "
            "resolving sources by hand and can drift from the packaged "
            "layout again")
    else:
        d = pathlib.Path(resolver())
        if not (d / "cb_gemv.cu").is_file():
            err(f"cuda_ext.csrc_dir() -> {d}, which has no cb_gemv.cu")
        else:
            ok(f"cuda_ext.csrc_dir() -> {d}")
    if _uses_os_pardir(pathlib.Path(cuda_ext.__file__)):
        err("gridbook/cuda_ext.py has live `os.pardir` code: repo-root "
            "relative source resolution is the bug this gate exists for")
    else:
        ok("cuda_ext.py contains no os.pardir path arithmetic")

    print()
    if _errors:
        print(f"{len(_errors)} error(s)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
