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

Because that cache is persistent and reused, every loader here asserts the
symbol set its callers need before returning the module — see
:func:`_require_symbols`.
A ``.so`` left in the cache by an older ``.cu`` would otherwise be handed back
unexamined, and the miss would surface as an ``AttributeError`` mid-forward —
or, worse, as a silent degrade where an optional-binding probe reads the stale
module as "an older build".

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


class StaleExtensionError(RuntimeError):
    """An extension loaded, but does not export a symbol its callers need.

    A third, distinct failure class: the toolchain worked and the package is
    complete, but the binary that came out of the build cache is not the one
    the current sources describe. That is a *deployment* defect, like
    :class:`IncompleteInstallError` and unlike "this machine has no nvcc", so
    it is reported at ERROR and it names the directory to delete.
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


def _require_symbols(mod, names, *, build_dir: str, source: str):
    """Return ``mod``, refusing it unless it exports every name in ``names``.

    ``torch.utils.cpp_extension.load`` hands back whatever the build cache
    produced, and until this check nothing in the package ever looked at the
    result. A cache that yields an OLD ``.so`` therefore surfaces only when a
    call site dereferences a symbol that the old source never had: an
    ``AttributeError`` raised from inside a ``torch.library.custom_op`` body
    (``ops.py``), i.e. mid-forward and possibly mid-capture, whose message
    names the attribute but not the directory that has to be deleted. Some
    misses do not even raise — the optional-binding probes
    (``ops.cb_expand_fp8_into_available``) read a missing symbol as "older
    build", so a stale cache silently costs a fast path instead.

    The cache is *designed* to be persistent and reused across restarts
    (``PRISMAQUANT_CB_EXT_DIR``; docs/INSTALL.md "Persisting the JIT build
    cache"; the reference image pins it to ``/opt/gridbook/ext-cache``), so
    "the cache outlived the sources" is a supported configuration reached by
    following the docs, not an exotic one.

    ``names`` is deliberately only the symbols the package calls WITHOUT a
    guard. Bindings that ship independently keep their existing call-site
    ``hasattr`` probes and must stay optional, or this check would turn a
    working degrade into a hard fallback: ``cb_expand_fp8_into``
    (``ops.cb_expand_fp8_into_available``), the ``l2_*`` window API
    (``moe.PrismaQuantCBMoEMethod._l2_ext_call``), and the grouped-MoE fused
    bindings (``moe.PrismaQuantCBMoEMethod._gf2_ok`` / ``_gf2_tile_sizes``).
    """
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise StaleExtensionError(
            f"the extension built from {source} loaded from {build_dir}, but "
            f"does not export {missing} (needs {list(names)}). The JIT build "
            f"cache is persistent and shared by design, so the usual cause is "
            f"a STALE .so left there by an older {source} — the current "
            f"source was never compiled. Delete {build_dir} and start again, "
            f"or point PRISMAQUANT_CB_EXT_DIR at a fresh dir. Check that the "
            f"serving user can WRITE that directory: a cache dir owned by "
            f"another user (e.g. created by an earlier `docker run` without "
            f"`--user`) is the common way a rebuild does not happen.")
    return mod


# Symbols ``get_ext()``'s module must export, asserted on EVERY load rather
# than only at build time. Each is dereferenced unconditionally by a custom op
# in ``ops.py`` -- there is no probe upstream of these, so a module missing one
# is strictly worse than no module at all:
_EXT_SYMBOLS = (
    "fp8_act_qdq",          # ops.fp8_act_qdq
    "cb_gemv_fp8",          # ops.cb_gemv_fp8
    "cb_gemv_fp4_v2",       # ops.cb_gemv_fp4_v2
    "cb_expand_fp8",        # ops.cb_expand_fp8
    "cb_moe_gemv_fp8",      # ops.cb_moe_gemv_fp8
    "cb_moe_gemv_fp4_v2",   # ops.cb_moe_gemv_fp4_v2
    "cb_moe_combine",       # ops.cb_moe_combine
)


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
        mod = load(name="prismaquant_cb_ext", sources=[src],
                   build_directory=build_dir,
                   extra_cuda_cflags=["-O3"], verbose=False)
        # Assign only after the symbol set checks out: a module missing one of
        # these must never become the value `get_ext()` returns.
        _ext = _require_symbols(mod, _EXT_SYMBOLS, build_dir=build_dir,
                                source="cb_gemv.cu")
    except StaleExtensionError as exc:
        # Same severity as IncompleteInstallError and for the same reason: a
        # deployment defect the operator can fix, not a property of the box.
        # The cost of the fallback is the figure sourced in the arm below.
        print(f"[prismaquant-cb] ERROR: stale CUDA decode-GEMV extension — "
              f"{exc} Falling back to the Triton decode path until the cache "
              f"is cleared: correct, but not production-eligible "
              f"(docs/BENCHMARKS.md).",
              file=sys.stderr, flush=True)
        _ext = None
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

# ops.cb_prefill_persistent_tc dereferences this straight after the None check
# (it has no probe), and it is the only binding cb_persistent_tc.cu exports.
_PTC_SYMBOLS = ("cb_prefill_persistent_tc",)


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
        mod = load(name="pq_cb_ptc",
                   sources=[os.path.join(src_dir, "cb_persistent_tc.cu")],
                   extra_include_paths=[cut, src_dir],
                   extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                   build_directory=build_dir, verbose=False)
        _ptc = _require_symbols(mod, _PTC_SYMBOLS, build_dir=build_dir,
                                source="cb_persistent_tc.cu")
    except StaleExtensionError as exc:
        # Worth a warning even though this ext is opt-in: without it the None
        # return makes ops.cb_prefill_persistent_tc report "ext not enabled
        # (PRISMAQUANT_ENABLE_PTC=1)", which is the wrong diagnosis for someone
        # who has just set that variable.
        import warnings
        warnings.warn(f"stale persistent-TC ext — {exc}")
        _ptc = None
    except IncompleteInstallError as exc:
        import warnings
        warnings.warn(f"broken gridbook install — {exc}")
        _ptc = None
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"persistent-TC ext unavailable: {exc}")
        _ptc = None
    return _ptc


_fused_fp4 = None
_fused_fp4_tried = False


def get_fused_fp4_ext():
    """The NVFP4_CB fused BLOCK-SCALED prefill extension
    (cb_fused_fp4_gemm.cu), or None. Separate module from the fp8 fused ext
    so the SASS gate (OMMA.SF.16864 present / QMMA absent — the fp4-at-fp8-
    rate trap, docs/lanes/nvfp4-cb/fp4-fused-prefill.md) can be run against
    the fp4 module alone. Needs the CUTLASS headers (vLLM's bundled copy, or
    PRISMAQUANT_CUTLASS_INCLUDE for venv builds) and the sm_121a/sm_120a
    arch-specific target — the block-scaled MMA is an arch-'a' instruction,
    so the build pins the current device's compute_XYa. Fail-soft: serving
    falls back to the Triton/transient fp4 paths."""
    global _fused_fp4, _fused_fp4_tried
    if _fused_fp4_tried:
        return _fused_fp4
    _fused_fp4_tried = True
    try:
        import torch
        from torch.utils.cpp_extension import load

        cut_inc = os.environ.get("PRISMAQUANT_CUTLASS_INCLUDE")
        if not cut_inc:
            cut_inc = _find_cutlass_include()
        # cutlass/util/packed_stride.hpp lives in the tools tree of a CUTLASS
        # checkout (vLLM's bundled copy keeps the same shape).
        incs = [cut_inc]
        util_inc = os.path.join(os.path.dirname(cut_inc), "tools", "util",
                                "include")
        if os.path.isdir(util_inc):
            incs.append(util_inc)
        src_dir = _require_csrc("cb_fused_fp4_gemm.cu")
        build_dir = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_dir, "fused_fp4")
        os.makedirs(build_dir, exist_ok=True)
        cc = torch.cuda.get_device_capability()
        arch = f"compute_{cc[0]}{cc[1]}a"
        code = f"sm_{cc[0]}{cc[1]}a"
        _fused_fp4 = load(
            name="pq_cb_fused_fp4",
            sources=[os.path.join(src_dir, "cb_fused_fp4_gemm.cu")],
            extra_include_paths=incs + [src_dir],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               f"-gencode=arch={arch},code={code}"],
            build_directory=build_dir, verbose=False)
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"fp4 prefill stays on the Triton/transient paths.",
              file=sys.stderr, flush=True)
        _fused_fp4 = None
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused fp4 prefill extension "
              f"unavailable ({type(exc).__name__}: {exc}); fp4 prefill stays "
              f"on the Triton/transient paths (expected off Blackwell or "
              f"without nvcc).", file=sys.stderr, flush=True)
        _fused_fp4 = None
    return _fused_fp4


# The one binding this module exists for: BOTH fused entry points gate the
# whole path on it (moe.PrismaQuantCBMoEMethod._gf_ok and the mid-M branch of
# linear.PrismaQuantCBLinearMethod._apply_inline), so a module without it is
# functionally identical to None -- except that it costs a multi-minute CUTLASS
# build and hides the reason for the degrade. Everything else cb_fused_gemm.cu
# exports stays OPTIONAL and keeps its call-site probe: the grouped-MoE
# bindings ship independently of this one by design (moe.py, _gf2_ok).
_FUSED_SYMBOLS = ("cb_fused_prefill_mm_scaled",)


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
        mod = load(name="pq_cb_fused",
                   sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
                   extra_include_paths=[os.path.join(cut, "include"),
                                        os.path.join(cut, "tools", "util",
                                                     "include"),
                                        src_dir],
                   extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                   build_directory=build_dir, verbose=False)
        _fused = _require_symbols(mod, _FUSED_SYMBOLS, build_dir=build_dir,
                                  source="cb_fused_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: stale fused prefill extension — {exc} "
              f"Mid-M prefill stays on the transient expand path.",
              file=sys.stderr, flush=True)
        _fused = None
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
