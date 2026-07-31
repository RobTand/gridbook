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

Every loader validates the symbols its callers will use before returning a
module. Strict call contracts use :func:`_require_symbols`; fused FP4 uses
independent symbol families because its dense and grouped call sites are
separately guarded. An incompatible module would otherwise fail with
``AttributeError`` mid-forward, or silently disable a probed fast path.

No fast-math: the QDQ kernel's division/conversion rounding must match torch
bit-for-bit.
"""
from __future__ import annotations

import os
import sys
import threading

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
    """A loaded extension does not satisfy its current Python call contract.

    Kept separate from :class:`IncompleteInstallError` and compiler failures:
    loading succeeded, but the resulting Python module is incompatible. A
    stale/corrupt cache entry is one possible cause; so is an unexpected module
    on the import path. The diagnostic names both the module and build cache.
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


def _cache_diagnostics(build_dir: str) -> str:
    """Describe a JIT build directory without claiming why a load went stale.

    An unwritable directory normally fails while acquiring the build lock or
    writing build products; it is not evidence that ``load`` reused an old
    module. Reporting the observed path/mode/owner/access lets an operator
    distinguish that build failure from a binary API mismatch.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(build_dir)))
    try:
        info = os.stat(path)
    except OSError as exc:
        return (f"requested build directory {path!r} cannot be stat'ed "
                f"({type(exc).__name__}: {exc})")
    writable = os.access(path, os.W_OK | os.X_OK)
    return (f"requested build directory {path!r} has mode "
            f"{info.st_mode & 0o7777:04o}, owner uid:gid "
            f"{info.st_uid}:{info.st_gid}, and is "
            f"{'writable' if writable else 'not writable'} by this process")


def _module_location(mod) -> str:
    path = getattr(mod, "__file__", None)
    return repr(os.fspath(path)) if path is not None else "<unknown>"


def _mismatch_message(mod, *, build_dir: str, source: str,
                      requirement: str) -> str:
    return (
        f"the module loaded for {source} from {_module_location(mod)} does not "
        f"satisfy the current call contract: {requirement}. "
        f"{_cache_diagnostics(build_dir)}. Clear this extension's build "
        f"directory (or choose a fresh PRISMAQUANT_CB_EXT_DIR) and restart. "
        f"A stale/corrupt cache or unexpected module is possible. If the "
        f"build itself reported PermissionError, fix the directory ownership "
        f"or mode; an unwritable cache ordinarily fails at lock/build time "
        f"rather than proving that an old binary was reused."
    )


def _require_symbols(mod, names, *, build_dir: str, source: str):
    """Return ``mod``, refusing it unless it exports every name in ``names``.

    ``names`` must contain only symbols called without an upstream capability
    probe. The helper intentionally keeps a small, stable signature so new JIT
    loaders can share the same diagnostics.
    """
    required = tuple(names)
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        requirement = (f"missing {missing}; every required symbol is "
                       f"{list(required)}")
        raise StaleExtensionError(_mismatch_message(
            mod, build_dir=build_dir, source=source,
            requirement=requirement))
    return mod


def _require_any_symbol_family(mod, families, *, build_dir: str, source: str):
    """Return ``mod`` when at least one independent call family is complete.

    ``families`` is an iterable of ``(label, required_symbols)`` pairs. A
    fused module may legitimately carry dense-prefill bindings, grouped-MoE
    bindings, or both. Missing an entire family preserves its call-site
    fallback; a module with no complete useful family is refused.
    """
    normalized = tuple((label, tuple(names)) for label, names in families)
    if not normalized or any(not names for _label, names in normalized):
        raise ValueError("symbol families must be non-empty")
    if any(all(hasattr(mod, name) for name in names)
           for _label, names in normalized):
        return mod
    detail = "; ".join(
        f"{label} needs {list(names)} (missing "
        f"{[name for name in names if not hasattr(mod, name)]})"
        for label, names in normalized)
    raise StaleExtensionError(_mismatch_message(
        mod, build_dir=build_dir, source=source,
        requirement=f"no usable symbol family; {detail}"))


# Symbols ``get_ext()``'s module must export, asserted on EVERY load rather
# than only at build time. Most are dereferenced unconditionally by a custom op
# in ``ops.py``. ``fp4_act_qdq`` does have a correct codec fallback, but keeping
# it in the same revision contract prevents a pre-QDQ cache from being accepted
# as the current main extension and silently losing the single-launch path.
_EXT_SYMBOLS = (
    "fp8_act_qdq",          # ops.fp8_act_qdq
    "fp4_act_qdq",          # ops.fp4_act_qdq
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
        # Loading succeeded, so distinguish an incompatible module from source
        # packaging and compiler/toolchain failures.
        print(f"[prismaquant-cb] ERROR: incompatible CUDA decode-GEMV "
              f"extension — "
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


_ext_v2 = None
_tried_v2 = False
_ext_v2_lock = threading.Lock()

# Symbols cb_gemv_v2.cu must export, checked on EVERY load rather than only in
# a build script. ``PRISMAQUANT_CB_EXT_DIR`` is normally a PERSISTENT directory
# (that is the whole point of the env var — see the module docstring), so a .so
# left there by an OLDER cb_gemv_v2.cu is reused silently by
# ``torch.utils.cpp_extension.load`` and the dispatch would then call a symbol
# that is not in it — a stale-artefact failure, mid-forward, on a JIT artefact.
# Failing here instead routes every CB layer to the inherited kernel with a
# loud reason.
_V2_SYMBOLS = (
    "cb_gemv_v2",
    "cb_gemv_v2_prefers_inherited",
    "cb_gemv_v2_prepare",
    "cb_expand_v2",
)


def get_ext_v2():
    """The compiled CB-GEMV-v2 extension (``gridbook/csrc/cb_gemv_v2.cu``).

    A SEPARATE JIT MODULE from :func:`get_ext`, not a second source file of it:
    both ``.cu`` files define ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)``
    (cb_gemv.cu / cb_gemv_v2.cu), so compiling them into one module collides at
    link. Separate name, separate build subdirectory, same cache root — so the
    one-time build persists exactly as the inherited ext's does.

    Fail-soft, same contract as :func:`get_ext`: any build/env failure returns
    None and the caller (``moe_gemv_select.cb_gemv_v2_available``) routes every
    CB layer to the INHERITED kernel — correct, unchanged serving, just without
    the v2 delta. The warning is loud because a silent degrade would turn a
    v2-vs-inherited A/B into two identical arms and read as "v2 buys nothing".
    """
    global _ext_v2, _tried_v2
    if _tried_v2:
        return _ext_v2
    with _ext_v2_lock:
        if _tried_v2:
            return _ext_v2
        try:
            import torch  # noqa: F401  (must import before cpp_extension)
            from torch.utils.cpp_extension import load

            src = os.path.join(
                _require_csrc("cb_gemv_v2.cu"), "cb_gemv_v2.cu")
            build_dir = os.environ.get(
                "PRISMAQUANT_CB_EXT_DIR") or os.path.join(
                    os.path.expanduser("~"), ".cache", "prismaquant-cb-ext")
            build_dir = os.path.join(build_dir, "v2")
            os.makedirs(build_dir, exist_ok=True)
            mod = load(name="prismaquant_cb_v2_ext", sources=[src],
                       build_directory=build_dir,
                       extra_cuda_cflags=["-O3"], verbose=False)
            _ext_v2 = _require_symbols(
                mod, _V2_SYMBOLS, build_dir=build_dir,
                source="cb_gemv_v2.cu")
        except StaleExtensionError as exc:
            print(f"[prismaquant-cb] ERROR: incompatible CB-GEMV-v2 "
                  f"extension — {exc} Every CB layer decodes on the inherited "
                  f"GEMV kernel.", file=sys.stderr, flush=True)
            _ext_v2 = None
        except IncompleteInstallError as exc:
            print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
                  f"Every CB layer decodes on the inherited GEMV kernel.",
                  file=sys.stderr, flush=True)
            _ext_v2 = None
        except Exception as exc:  # noqa: BLE001 — build/env failure -> fallback
            print(f"[prismaquant-cb] WARNING: the CB-GEMV-v2 extension could "
                  f"not be built ({type(exc).__name__}: {exc}); every CB layer "
                  f"decodes on the inherited GEMV kernel (this is expected on "
                  f"non-sm_120/sm_121 GPUs and without nvcc). To get the v2 "
                  f"path: {_NVCC_HINT}.", file=sys.stderr, flush=True)
            _ext_v2 = None
        finally:
            _tried_v2 = True
    return _ext_v2


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
        warnings.warn(f"incompatible persistent-TC ext — {exc}")
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


# Both families are guarded independently at their call sites
# (linear._try_fused_fp4 and moe._gf4_ok). A binary containing either family
# remains useful and must not be rejected merely because the other was built
# from a different revision.
_FUSED_FP4_SYMBOL_FAMILIES = (
    ("dense prefill", ("cb_fused_fp4_prefill_mm_scaled",)),
    ("grouped MoE prefill", ("cb_fused_fp4_moe_grouped",)),
)


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
        mod = load(
            name="pq_cb_fused_fp4",
            sources=[os.path.join(src_dir, "cb_fused_fp4_gemm.cu")],
            extra_include_paths=incs + [src_dir],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               f"-gencode=arch={arch},code={code}"],
            build_directory=build_dir, verbose=False)
        _fused_fp4 = _require_any_symbol_family(
            mod, _FUSED_FP4_SYMBOL_FAMILIES, build_dir=build_dir,
            source="cb_fused_fp4_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: incompatible fused fp4 prefill "
              f"extension — {exc} fp4 prefill stays on the Triton/transient "
              f"paths.", file=sys.stderr, flush=True)
        _fused_fp4 = None
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


# `_gf_ok` is the prerequisite for both the dense and grouped FP8 fused paths,
# and it dereferences this binding after its capability probe. Grouped bindings
# remain optional additions: `_gf2_ok` requires `_gf_ok` first, then probes
# `cb_fused_moe_grouped` and `cb_fused_moe_tile_m` separately.
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
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load

        cut_inc = _find_cutlass_include()
        cut_root = os.path.dirname(cut_inc)
        src_dir = _require_csrc("cb_fused_gemm.cu")
        build_dir = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_dir, "fused")
        os.makedirs(build_dir, exist_ok=True)
        mod = load(name="pq_cb_fused",
                   sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
                   extra_include_paths=[cut_inc,
                                        os.path.join(cut_root, "tools", "util",
                                                     "include"),
                                        src_dir],
                   extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                   build_directory=build_dir, verbose=False)
        _fused = _require_symbols(mod, _FUSED_SYMBOLS,
                                  build_dir=build_dir,
                                  source="cb_fused_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: incompatible fused prefill "
              f"extension — {exc} Fused dense and grouped prefill stay on "
              f"their fallback paths.",
              file=sys.stderr, flush=True)
        _fused = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Fused dense and grouped prefill stay on their fallback paths.",
              file=sys.stderr, flush=True)
        _fused = None
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused prefill extension unavailable "
              f"({type(exc).__name__}: {exc}); fused dense and grouped "
              f"prefill stay on their fallback paths (this is expected on "
              f"non-sm_120 GPUs and without nvcc).",
              file=sys.stderr, flush=True)
        _fused = None
    return _fused
