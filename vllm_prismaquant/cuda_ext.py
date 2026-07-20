"""JIT build/load of the CUDA decode-GEMV extension (``csrc/cb_gemv.cu``).

Loaded lazily on first use. Needs nvcc (present in the serving container;
absent in the build venv) — if the build fails the caller falls back to the
Triton decode path with a loud one-time warning, so the plugin keeps serving
everywhere the Triton prototype did.

Build cache: ``PRISMAQUANT_CB_EXT_DIR`` if set, else ``~/.cache/prismaquant-
cb-ext`` (inside the container that is ephemeral — one ~40 s build per
container start; mount a host dir over it to persist). Never ``/tmp``.

No fast-math: the QDQ kernel's division/conversion rounding must match torch
bit-for-bit.
"""
from __future__ import annotations

import os
import sys

_ext = None
_tried = False


def get_ext():
    """The compiled extension module, or None if unavailable."""
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    try:
        import torch  # noqa: F401  (must import before cpp_extension)
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "csrc", "cb_gemv.cu")
        build_dir = os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext")
        os.makedirs(build_dir, exist_ok=True)
        _ext = load(name="prismaquant_cb_ext", sources=[src],
                    build_directory=build_dir,
                    extra_cuda_cflags=["-O3"], verbose=False)
    except Exception as exc:  # noqa: BLE001 — any build/env failure -> fallback
        print(f"[prismaquant-cb] WARNING: CUDA decode-GEMV extension "
              f"unavailable ({type(exc).__name__}: {exc}); falling back to the "
              f"Triton decode path (slow prototype).",
              file=sys.stderr, flush=True)
        _ext = None
    return _ext
