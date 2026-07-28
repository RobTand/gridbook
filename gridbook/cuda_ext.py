"""JIT build/load of the CUDA decode-GEMV extension (``gridbook/csrc/cb_gemv.cu``).

Loaded lazily on first use. Needs nvcc (present in the serving container;
absent in the build venv) — if the build fails the caller falls back to the
Triton decode path with a loud one-time warning, so the plugin keeps serving
everywhere the Triton prototype did.

The ``.cu``/``.hpp`` sources ship *inside* the package (``gridbook/csrc``) and
are located with :func:`csrc_dir`, so an in-repo checkout, ``pip install -e``
and a wheel install all resolve identically. Do not reintroduce ``os.pardir``
repo-root arithmetic here: under a non-editable install only the package lands
in site-packages, so a repo-root-relative path does not exist and every
extension build fails silently into the slow Triton path.

Build cache: ``PRISMAQUANT_CB_EXT_DIR`` if set, else ``~/.cache/prismaquant-
cb-ext`` (inside the container that is ephemeral — one ~30 s build per
container start; mount a host dir over it to persist). Never ``/tmp``.
(~30 s is measured: cold ``get_ext()`` in ``vllm-node:latest``, no ``--gpus``,
``TORCH_CUDA_ARCH_LIST=12.1`` -> 29.4 s / 29.7 s on two runs.)

No fast-math: the QDQ kernel's division/conversion rounding must match torch
bit-for-bit.
"""
from __future__ import annotations

import os
import sys

_ext = None
_tried = False

_NVCC_HINT = (
    "install a CUDA toolchain matching your torch build (distro `cuda-toolkit` "
    "or the `nvidia-cuda-nvcc-*` wheel) and make sure `nvcc` is on PATH or "
    "CUDA_HOME points at it; set PRISMAQUANT_CB_EXT_DIR to a writable, "
    "persistent directory to keep the one-time ~30 s JIT build across restarts"
)


class IncompleteInstallError(FileNotFoundError):
    """A packaged CUDA source is missing from the installed package.

    Distinct from "no nvcc": this is a packaging/installation defect, not a
    property of the user's machine, and it is reported differently so the two
    are never confused.
    """


# Anchor for importlib.resources. ``__spec__.parent`` is the supported way to
# name the containing package; ``__package__`` is deprecated as a module
# attribute since 3.12 and can be None, in which case ``files(None)`` raises a
# bare AttributeError that would land in the generic "no CUDA toolchain" arm
# and print exactly the wrong diagnosis. The literal is the last resort.
_PKG = getattr(__spec__, "parent", None) or __package__ or "gridbook"


def csrc_dir() -> str:
    """Absolute path to the packaged CUDA sources (``gridbook/csrc``).

    Resolved through ``importlib.resources`` so it is identical for an in-repo
    checkout, an editable install (both setuptools editable modes) and a wheel
    install.
    """
    from importlib.resources import files

    try:
        return os.fspath(files(_PKG) / "csrc")
    except Exception as exc:  # noqa: BLE001 — any anchor/loader failure
        # Still a packaging defect, never a toolchain problem: report it as one.
        raise IncompleteInstallError(
            f"cannot locate the gridbook package resources (anchor {_PKG!r}): "
            f"{type(exc).__name__}: {exc}. This is a packaging defect, not a "
            f"missing CUDA toolchain — reinstall gridbook.") from exc


def _require_csrc(*names: str) -> str:
    """Return :func:`csrc_dir`, asserting the named sources are present."""
    d = csrc_dir()
    missing = [n for n in names if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise IncompleteInstallError(
            f"gridbook is installed without its CUDA sources: {missing} not "
            f"found under {d}. This is a packaging defect, not a missing CUDA "
            f"toolchain — reinstall gridbook (`pip install --force-reinstall "
            f"gridbook`) or install from a checkout.")
    return d


def get_ext():
    """The compiled extension module, or None if unavailable."""
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    try:
        import torch  # noqa: F401  (must import before cpp_extension)
        from torch.utils.cpp_extension import load

        src = os.path.join(_require_csrc("cb_gemv.cu"), "cb_gemv.cu")
        build_dir = os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext")
        os.makedirs(build_dir, exist_ok=True)
        _ext = load(name="prismaquant_cb_ext", sources=[src],
                    build_directory=build_dir,
                    extra_cuda_cflags=["-O3"], verbose=False)
    except IncompleteInstallError as exc:
        # The speed figure is sourced, not asserted: docs/BENCHMARKS.md records
        # the CUDA grouped-GEMV decode path taking the reference 35B MoE
        # artifact from 3.5 to ~33 served tok/s. Do not restate it as a
        # round number without re-measuring.
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Falling back to the Triton decode path: correct, but not "
              f"production-eligible — the CUDA decode path measured ~9x higher "
              f"served decode throughput on the reference MoE artifact "
              f"(3.5 -> 33 tok/s; docs/BENCHMARKS.md).",
              file=sys.stderr, flush=True)
        _ext = None
    except Exception as exc:  # noqa: BLE001 — any build/env failure -> fallback
        print(f"[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV "
              f"extension could not be built ({type(exc).__name__}: {exc}); "
              f"falling back to the Triton decode path (slow prototype). To "
              f"get the CUDA path: {_NVCC_HINT}.",
              file=sys.stderr, flush=True)
        _ext = None
    return _ext


_fused = None
_fused_tried = False


def _find_cutlass_include() -> str:
    """Locate the vLLM-bundled CUTLASS include dir (same discovery the
    fused ext uses)."""
    import glob

    import vllm

    vroot = os.path.dirname(os.path.abspath(vllm.__file__))
    for pat in ("third_party/fmha_sm100/cutlass", "third_party/cutlass"):
        cand = os.path.join(vroot, pat)
        if os.path.isdir(os.path.join(cand, "include")):
            return os.path.join(cand, "include")
    hits = glob.glob(os.path.join(
        vroot, "third_party", "**", "cutlass", "include"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("no bundled CUTLASS under vllm/third_party")


_ptc = None
_ptc_tried = False


def get_persistent_ext():
    """JIT build/load of the persistent-N tensor-core prefill kernel
    (gridbook/csrc/cb_persistent_tc.cu, #4b v1). Fail-soft like the fused ext."""
    global _ptc, _ptc_tried
    if _ptc_tried:
        return _ptc
    _ptc_tried = True
    # QUARANTINE (2026-07-23): a boot wedged minutes after this kernel's
    # bench container exited; until the canary ladder clears it, the ext
    # builds only on explicit opt-in.
    if os.environ.get("PRISMAQUANT_ENABLE_PTC") != "1":
        _ptc = None
        return None
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load

        src_dir = _require_csrc("cb_persistent_tc.cu")
        build_dir = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_dir, "ptc")
        os.makedirs(build_dir, exist_ok=True)
        cut = _find_cutlass_include()
        _ptc = load(name="pq_cb_ptc",
                    sources=[os.path.join(src_dir, "cb_persistent_tc.cu")],
                    extra_include_paths=[cut, src_dir],
                    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                    build_directory=build_dir, verbose=False)
    except IncompleteInstallError as exc:
        import warnings
        warnings.warn(f"broken gridbook install — {exc}")
        _ptc = None
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"persistent-TC ext unavailable: {exc}")
        _ptc = None
    return _ptc


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
        src_dir = _require_csrc("cb_fused_gemm.cu")
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
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Mid-M prefill stays on the transient expand path.",
              file=sys.stderr, flush=True)
        _fused = None
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused prefill extension unavailable "
              f"({type(exc).__name__}: {exc}); mid-M stays on the transient "
              f"expand path (the shipping default — this is expected on "
              f"non-sm_120 GPUs and without nvcc).",
              file=sys.stderr, flush=True)
        _fused = None
    return _fused
