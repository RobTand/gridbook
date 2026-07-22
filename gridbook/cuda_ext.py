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


_fused = None
_fused_tried = False


def get_fused_ext():
    """The CUTLASS decode-in-prologue prefill extension (cb_fused_gemm.cu),
    or None. Separate module from the GEMV ext: it needs the CUTLASS headers
    (taken from the vLLM install's bundled copy) and a longer JIT build.
    Fail-soft like get_ext — serving falls back to the transient-expand path."""
    global _fused, _fused_tried
    if _fused_tried:
        return _fused
    _fused_tried = True
    try:
        import glob

        import torch  # noqa: F401
        import vllm
        from torch.utils.cpp_extension import load

        vroot = os.path.dirname(os.path.abspath(vllm.__file__))
        cut = None
        for pat in ("third_party/fmha_sm100/cutlass", "third_party/cutlass"):
            cand = os.path.join(vroot, pat)
            if os.path.isdir(os.path.join(cand, "include")):
                cut = cand
                break
        if cut is None:
            hits = glob.glob(os.path.join(
                vroot, "third_party", "**", "cutlass", "include"),
                recursive=True)
            if hits:
                cut = os.path.dirname(hits[0])
        if cut is None:
            raise FileNotFoundError("no bundled CUTLASS under vllm/third_party")
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "csrc")
        build_dir = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_dir, "fused")
        os.makedirs(build_dir, exist_ok=True)
        _fused = load(name="pq_cb_fused",
                      sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
                      extra_include_paths=[os.path.join(cut, "include"),
                                           os.path.join(cut, "tools", "util",
                                                        "include"),
                                           src_dir],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                      build_directory=build_dir, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused prefill extension unavailable "
              f"({type(exc).__name__}: {exc}); mid-M stays on the transient "
              f"expand path.", file=sys.stderr, flush=True)
        _fused = None
    return _fused
