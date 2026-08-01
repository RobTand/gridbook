"""JIT build/load of the CUDA decode-GEMV extension (``gridbook/csrc/cb_gemv.cu``).

The low-level loader is lazy. Production model construction resolves every
reachable module before serving, outside first forward and CUDA-graph capture.
Compilation needs nvcc (present in the serving container; often absent in
build-only environments). Capability probes may receive ``None`` when a build
is unavailable, but production call sites use
:func:`require_ext` / :func:`require_ext_v2` and fail closed. Gridbook has no
interpreted-kernel runtime fallback.

The ``.cu``/``.hpp`` sources ship *inside* the package (``gridbook/csrc``) and
are located with :func:`csrc_dir`, so an in-repo checkout, ``pip install -e``
and a wheel install all resolve identically. Do not reintroduce ``os.pardir``
repo-root arithmetic here: under a non-editable install only the package lands
in site-packages, so a repo-root-relative path does not exist and every native
extension build fails.

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

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading

_ext = None
_tried = False
_ext_lock = threading.Lock()

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


class NativeKernelUnavailableError(RuntimeError):
    """A required Gridbook native CUDA/CUTLASS kernel cannot be called.

    Loader functions retain ``None`` as a useful capability-probe result, but
    serving code must raise this error instead of selecting an unoptimized or
    numerically different implementation.
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


def _sha256_file(path: str) -> str:
    """Return the content identity of one JIT build input."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiler_identity(command: str | None) -> dict[str, object]:
    """Best-effort identity for a compiler command, without using a shell.

    Compiler discovery must never become a new reason for a fail-soft JIT to
    fail.  An absent or unqueryable executable is therefore represented in the
    identity instead of raising.  ``shlex`` supports safe ``CXX=ccache g++``
    style commands while ``shell=False`` prevents command substitution.
    """
    if not command:
        return {"argv": [], "path": None, "version": None}
    try:
        argv = shlex.split(os.fspath(command))
    except ValueError as exc:
        return {"argv": [os.fspath(command)], "path": None,
                "version": f"{type(exc).__name__}: {exc}"}
    if not argv:
        return {"argv": [], "path": None, "version": None}
    resolved = shutil.which(argv[0])
    if resolved is None and os.path.isfile(argv[0]):
        resolved = os.path.abspath(argv[0])
    if resolved is None:
        return {"argv": argv, "path": None, "version": "not found"}
    probe = [resolved, *argv[1:], "--version"]
    try:
        result = subprocess.run(
            probe, check=False, capture_output=True, text=True, timeout=10)
        output = (result.stdout or result.stderr).strip()
        version = f"exit={result.returncode}: {output}"
    except (OSError, subprocess.SubprocessError) as exc:
        version = f"{type(exc).__name__}: {exc}"
    return {"argv": argv, "path": os.path.realpath(resolved),
            "version": version}


def _optional_runtime_value(owner, name: str):
    """Read or call an optional runtime ABI field, returning ``None``."""
    value = getattr(owner, name, None)
    if value is None:
        return None
    try:
        return value() if callable(value) else value
    except Exception:  # noqa: BLE001 — optional telemetry cannot break a JIT
        return None


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
    bindings, or both. Missing an entire family preserves its separate exact
    native route; a module with no complete useful family is refused.
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
# in ``ops.py``. Keeping ``fp4_act_qdq`` in the same revision contract prevents
# a pre-QDQ cache from being accepted as the current main extension and
# silently losing a required native serving operation.
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
    if _tried:
        return _ext
    with _ext_lock:
        if _tried:
            return _ext
        return _load_ext_locked()


def require_ext(operation: str = "this operation"):
    """Return the main native extension or raise a production-facing error."""
    ext = get_ext()
    if ext is None:
        raise NativeKernelUnavailableError(
            f"{operation} requires Gridbook's native CUDA extension "
            f"(cb_gemv.cu), but it is unavailable. Gridbook does not fall "
            f"back to Triton. To enable the native path: {_NVCC_HINT}.")
    return ext


def _load_ext_locked():
    """Build and publish the main extension with ``_ext_lock`` held."""
    global _ext, _tried
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
              f"extension — {exc} Native Gridbook execution is unavailable "
              f"until the extension cache is cleared; serving will fail "
              f"closed.",
              file=sys.stderr, flush=True)
        _ext = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Native Gridbook execution is unavailable and serving will "
              f"fail closed.",
              file=sys.stderr, flush=True)
        _ext = None
    except Exception as exc:  # noqa: BLE001 — loader reports unavailable
        print(f"[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV "
              f"extension could not be built ({type(exc).__name__}: {exc}); "
              f"native Gridbook execution is unavailable and serving will "
              f"fail closed. To enable the native path: {_NVCC_HINT}.",
              file=sys.stderr, flush=True)
        _ext = None
    finally:
        _tried = True
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

    This low-level probe is fail-soft: any build/environment failure returns
    ``None``. The optional decode selector may then keep its inherited GEMV,
    but production FP4-v2 loaders separately call
    :func:`require_fp4_v2_expander` and fail closed because this module also
    owns the exact quality expander. The warning is loud because a silent
    decode-selector miss would turn a v2-vs-inherited A/B into two identical
    arms and read as "v2 buys nothing".
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
                  f"extension — {exc} The optional decode selector stays on "
                  f"the inherited GEMV; FP4-v2 quality serving fails closed.",
                  file=sys.stderr, flush=True)
            _ext_v2 = None
        except IncompleteInstallError as exc:
            print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
                  f"The optional decode selector stays inherited; FP4-v2 "
                  f"quality serving fails closed.",
                  file=sys.stderr, flush=True)
            _ext_v2 = None
        except Exception as exc:  # noqa: BLE001 — build/env failure -> fallback
            print(f"[prismaquant-cb] WARNING: the CB-GEMV-v2 extension could "
                  f"not be built ({type(exc).__name__}: {exc}); the optional "
                  f"decode selector stays on the inherited GEMV, while any "
                  f"FP4-v2 quality-serving load fails closed. To enable the "
                  f"native module: {_NVCC_HINT}.", file=sys.stderr, flush=True)
            _ext_v2 = None
        finally:
            _tried_v2 = True
    return _ext_v2


def require_ext_v2(operation: str = "this operation"):
    """Return the native FP4-v2 extension or fail closed."""
    ext = get_ext_v2()
    if ext is None:
        raise NativeKernelUnavailableError(
            f"{operation} requires Gridbook's native FP4-v2 CUDA extension "
            f"(cb_gemv_v2.cu), but it is unavailable. Gridbook does not fall "
            f"back to Triton. To enable the native path: {_NVCC_HINT}.")
    return ext


def require_fp4_v2_expander(operation: str = "this operation", *, device=None):
    """Return and device-attest the required FP4-v2 quality expander.

    Building ``cb_gemv_v2.cu`` proves that the symbols exist, but the current
    expander shares the v2 module's 99 KiB dynamic-shared-memory preparation
    contract.  That contract is qualified only for CUDA compute capability
    12.0/12.1 and is device-specific, so production loaders call this function
    before first forward or graph capture.
    """
    ext = require_ext_v2(operation)
    try:
        if device is None:
            ext.cb_gemv_v2_prepare()
        else:
            import torch

            with torch.cuda.device(device):
                ext.cb_gemv_v2_prepare()
    except Exception as exc:  # noqa: BLE001 — normalize the load-time gate
        raise NativeKernelUnavailableError(
            f"{operation} requires Gridbook's native FP4-v2 quality expander "
            "on CUDA compute capability 12.0 or 12.1 with at least 99 KiB "
            "of opt-in shared memory, but load-time device attestation failed "
            f"({type(exc).__name__}: {exc}). Gridbook does not defer this "
            "failure to first prefill or fall back to Triton.") from exc
    return ext


_fused = None
_fused_tried = False
_fused_lock = threading.Lock()


def _find_cutlass_include() -> str:
    """Locate the vLLM-bundled CUTLASS include dir (same discovery the
    fused ext uses), without importing vLLM's runtime package.

    Importing ``vllm`` merely to locate its files eagerly initializes optional
    compiler backends, including Triton on some releases. Gridbook does not
    need any of that state to compile an owned CUTLASS translation unit; module
    discovery gives us the package directory without executing ``__init__``.
    """
    import glob
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        raise FileNotFoundError("vLLM package not found; cannot locate CUTLASS")
    if spec.submodule_search_locations:
        vroot = os.path.abspath(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        vroot = os.path.dirname(os.path.abspath(spec.origin))
    else:
        raise FileNotFoundError(
            "vLLM package has no filesystem location; cannot locate CUTLASS")
    for pat in ("third_party/fmha_sm100/cutlass", "third_party/cutlass"):
        cand = os.path.join(vroot, pat)
        if os.path.isdir(os.path.join(cand, "include")):
            return os.path.join(cand, "include")
    hits = glob.glob(os.path.join(
        vroot, "third_party", "**", "cutlass", "include"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("no bundled CUTLASS under vllm/third_party")


_bf16_grouped = None
_bf16_grouped_tried = False
_bf16_grouped_lock = threading.Lock()
_BF16_GROUPED_SYMBOLS = (
    "cb_bf16_grouped_mm",
    "cb_bf16_grouped_mm_out",
)


def get_bf16_grouped_ext():
    """Load Gridbook's native CUTLASS grouped BF16 bridge.

    This is the quality-preserving prefill backend for a BF16 activation QDQ
    and a bit-exact BF16 FP4-CB-v2 weight expansion. Capability probes may
    inspect the ``None`` result; production calls use
    :func:`require_bf16_grouped_ext` and fail closed.
    """
    if _bf16_grouped_tried:
        return _bf16_grouped
    with _bf16_grouped_lock:
        if _bf16_grouped_tried:
            return _bf16_grouped
        return _load_bf16_grouped_ext_locked()


def require_bf16_grouped_ext(operation: str = "this operation"):
    """Return the native CUTLASS BF16 grouped extension or fail closed."""
    ext = get_bf16_grouped_ext()
    if ext is None:
        raise NativeKernelUnavailableError(
            f"{operation} requires Gridbook's native CUTLASS grouped BF16 "
            "extension (cb_bf16_grouped_gemm.cu), but it is unavailable. "
            "Gridbook does not fall back to Triton, torch._grouped_mm, "
            f"F.linear, or cuBLAS. To enable the native path: {_NVCC_HINT}.")
    return ext


def _load_bf16_grouped_ext_locked():
    """Build and publish grouped BF16 with ``_bf16_grouped_lock`` held."""
    global _bf16_grouped, _bf16_grouped_tried
    build_dir = "<unresolved>"
    try:
        import torch
        from torch.utils.cpp_extension import load

        cc = torch.cuda.get_device_capability()
        if cc < (8, 0):
            raise RuntimeError(
                f"CUTLASS grouped BF16 requires compute capability >= 8.0, "
                f"got {cc[0]}.{cc[1]}")
        src_dir = _require_csrc("cb_bf16_grouped_gemm.cu")
        cut_inc = _find_cutlass_include()
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "bf16_grouped")
        os.makedirs(build_dir, exist_ok=True)
        arch = f"compute_{cc[0]}{cc[1]}"
        code = f"sm_{cc[0]}{cc[1]}"
        mod = load(
            name="pq_cb_bf16_grouped",
            sources=[os.path.join(src_dir, "cb_bf16_grouped_gemm.cu")],
            extra_include_paths=[cut_inc],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               f"-gencode=arch={arch},code={code}"],
            build_directory=build_dir,
            verbose=False)
        _bf16_grouped = _require_symbols(
            mod, _BF16_GROUPED_SYMBOLS, build_dir=build_dir,
            source="cb_bf16_grouped_gemm.cu")
    except StaleExtensionError as exc:
        print("[prismaquant-cb] ERROR: incompatible CUTLASS grouped BF16 "
              f"extension — {exc} Native quality prefill is unavailable and "
              "serving will fail closed.", file=sys.stderr, flush=True)
        _bf16_grouped = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              "Native quality prefill is unavailable and serving will fail "
              "closed.", file=sys.stderr, flush=True)
        _bf16_grouped = None
    except Exception as exc:  # noqa: BLE001 — capability probe is fail-soft
        print("[prismaquant-cb] WARNING: CUTLASS grouped BF16 extension "
              f"unavailable ({type(exc).__name__}: {exc}); native quality "
              "prefill is unavailable and serving will fail closed. To "
              f"enable the native path: {_NVCC_HINT}.",
              file=sys.stderr, flush=True)
        _bf16_grouped = None
    finally:
        _bf16_grouped_tried = True
    return _bf16_grouped


_fused_fp4 = None
_fused_fp4_tried = False
_fused_fp4_lock = threading.Lock()


# These two packaged files define the Gridbook-owned fused FP4 implementation.
# Both are explicit build inputs: torch's extension versioner hashes the source
# passed to ``load`` but does not make an included header part of its stable
# module name or caller-selected build-directory identity.
_FUSED_FP4_BUILD_INPUTS = (
    "cb_fused_fp4_gemm.cu",
    "cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
)
_FUSED_FP4_ABI_SCHEMA = 4


# All families are guarded independently at their call sites
# (linear._try_fused_fp4 and moe._gf4_ok). A binary containing any one family
# remains useful and must not be rejected merely because another was built
# from a different revision.
_FUSED_FP4_SYMBOL_FAMILIES = (
    ("row-wise activation", (
        "cb_nvfp4_quantize_rows", "cb_nvfp4_quantize_rows_out")),
    ("static-LSQ activation", (
        "cb_nvfp4_quantize_static_lsq",
        "cb_nvfp4_quantize_static_lsq_out")),
    ("dense prefill", ("cb_fused_fp4_prefill_mm_scaled",)),
    ("grouped MoE prefill", ("cb_fused_fp4_moe_grouped",)),
)


def _fused_fp4_build_identity(torch, cpp_extension, *, src_dir: str,
                              cutlass_include: str,
                              util_include: str,
                              capability: tuple[int, int]):
    """Return ``(digest, payload)`` for every practical binary ABI input.

    The digest keys both the build directory and extension module name.  That
    keeps persistent caches from reusing an FP4 binary across changes to the
    included Gridbook header, target architecture, Python/Torch ABI, CUDA
    toolkit, or host compiler.  A small set of external CUTLASS sentinels is
    also hashed; Ninja's generated dependency file remains authoritative for
    the complete external include graph inside a keyed directory.
    """
    c_ext = getattr(torch, "_C", None)
    cuda_home = getattr(cpp_extension, "CUDA_HOME", None)
    nvcc = os.environ.get("CUDACXX")
    if not nvcc and cuda_home:
        nvcc = os.path.join(os.fspath(cuda_home), "bin", "nvcc")
    if not nvcc:
        nvcc = "nvcc"

    cxx = os.environ.get("CXX")
    if not cxx:
        cxx = _optional_runtime_value(cpp_extension, "get_cxx_compiler")

    packaged_inputs = {
        name: _sha256_file(os.path.join(src_dir, name))
        for name in _FUSED_FP4_BUILD_INPUTS
    }
    cutlass_inputs = {}
    for label, path in (
        ("cutlass/cutlass.h",
         os.path.join(cutlass_include, "cutlass", "cutlass.h")),
        ("cutlass/version.h",
         os.path.join(cutlass_include, "cutlass", "version.h")),
        ("cutlass/util/packed_stride.hpp",
         os.path.join(util_include, "cutlass", "util",
                      "packed_stride.hpp")),
    ):
        cutlass_inputs[label] = (
            _sha256_file(path) if os.path.isfile(path) else None)

    major, minor = capability
    payload = {
        "schema": _FUSED_FP4_ABI_SCHEMA,
        "bindings": [
            [label, list(names)]
            for label, names in _FUSED_FP4_SYMBOL_FAMILIES
        ],
        "inputs": packaged_inputs,
        "cutlass_inputs": cutlass_inputs,
        "target": {
            "capability": [major, minor],
            "compute": f"compute_{major}{minor}a",
            "code": f"sm_{major}{minor}a",
        },
        "python": {
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "torch": {
            "version": str(getattr(torch, "__version__", "unknown")),
            "cuda": (None if getattr(getattr(torch, "version", None),
                                     "cuda", None) is None else
                     str(torch.version.cuda)),
            "cxx11_abi": _optional_runtime_value(
                torch, "compiled_with_cxx11_abi"),
            "glibcxx_cxx11_abi": _optional_runtime_value(
                c_ext, "_GLIBCXX_USE_CXX11_ABI"),
            "pybind11_compiler": _optional_runtime_value(
                c_ext, "_PYBIND11_COMPILER_TYPE"),
            "pybind11_stdlib": _optional_runtime_value(
                c_ext, "_PYBIND11_STDLIB"),
            "pybind11_build_abi": _optional_runtime_value(
                c_ext, "_PYBIND11_BUILD_ABI"),
        },
        "cuda": {
            "compiled_version": _optional_runtime_value(
                c_ext, "_cuda_getCompiledVersion"),
            "runtime_version": _optional_runtime_value(
                c_ext, "_cuda_getRuntimeVersion"),
            "driver_version": _optional_runtime_value(
                c_ext, "_cuda_getDriverVersion"),
            "nvcc": _compiler_identity(nvcc),
        },
        "host_compiler": _compiler_identity(
            None if cxx is None else os.fspath(cxx)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _require_fused_fp4_identity(mod, *, expected_name: str, identity: str,
                                build_dir: str, source: str):
    """Validate the identity-bearing ``PyInit_*`` module ABI.

    ``TORCH_EXTENSION_NAME`` makes the requested name part of the compiled
    module's exported initializer.  Requiring that same name after import is a
    practical ABI attestation without maintaining a second C++ version
    literal.  The full digest is then exposed for release/runtime telemetry.
    """
    actual = getattr(mod, "__name__", None)
    if actual != expected_name:
        raise StaleExtensionError(_mismatch_message(
            mod, build_dir=build_dir, source=source,
            requirement=(f"JIT ABI identity mismatch: expected module "
                         f"{expected_name!r}, loaded {actual!r}")))
    mod.__gridbook_jit_identity__ = identity
    mod.__gridbook_jit_abi_schema__ = _FUSED_FP4_ABI_SCHEMA
    return mod


def get_fused_fp4_ext():
    """The NVFP4_CB fused BLOCK-SCALED prefill extension
    (cb_fused_fp4_gemm.cu), or None. Separate module from the fp8 fused ext
    so the SASS gate (OMMA.SF.16864 present / QMMA absent — the fp4-at-fp8-
    rate trap, docs/lanes/nvfp4-cb/fp4-fused-prefill.md) can be run against
    the fp4 module alone. Needs the CUTLASS headers (vLLM's bundled copy, or
    PRISMAQUANT_CUTLASS_INCLUDE for venv builds) and the sm_121a/sm_120a
    arch-specific target — the block-scaled MMA is an arch-'a' instruction,
    so the build pins the current device's compute_XYa. Fail-soft: this
    research specialization may be absent while production stays on exact
    native BF16 expansion plus the owned grouped CUTLASS bridge."""
    if _fused_fp4_tried:
        return _fused_fp4
    with _fused_fp4_lock:
        if _fused_fp4_tried:
            return _fused_fp4
        return _load_fused_fp4_ext_locked()


def preload_fused_extensions(*, strict: bool = False) -> dict[str, bool]:
    """Attempt both independent fused JIT modules for residency-matched A/Bs.

    Each loader is fail-soft in normal operation, but keep the attempts
    independent even if a monkeypatch, import failure, or future strict mode
    raises.  Loading only the FP8 module does not residency-match an NVFP4
    baseline with its fused candidate.  The returned status lets validation
    code prove residency; ``strict=True`` raises only after both attempts.
    """
    status: dict[str, bool] = {}
    errors: dict[str, Exception] = {}
    for family, load_ext in (
        ("fp8", get_fused_ext),
        ("fp4", get_fused_fp4_ext),
    ):
        try:
            status[family] = load_ext() is not None
        except Exception as exc:  # noqa: BLE001 — report after both attempts
            status[family] = False
            errors[family] = exc
    if strict and not all(status.values()):
        detail = ", ".join(
            f"{family}={errors.get(family, 'unavailable')}"
            for family, loaded in status.items() if not loaded
        )
        raise RuntimeError(f"fused extension preload failed: {detail}")
    return status


def _load_fused_fp4_ext_locked():
    """Build and publish fused FP4 with ``_fused_fp4_lock`` held."""
    global _fused_fp4, _fused_fp4_tried
    try:
        import torch
        from torch.utils import cpp_extension

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
        src_dir = _require_csrc(*_FUSED_FP4_BUILD_INPUTS)
        cc = torch.cuda.get_device_capability()
        identity, _identity_payload = _fused_fp4_build_identity(
            torch, cpp_extension, src_dir=src_dir,
            cutlass_include=cut_inc, util_include=util_inc,
            capability=cc)
        module_name = f"pq_cb_fused_fp4_{identity}"
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "fused_fp4", identity)
        os.makedirs(build_dir, exist_ok=True)
        arch = f"compute_{cc[0]}{cc[1]}a"
        code = f"sm_{cc[0]}{cc[1]}a"
        mod = cpp_extension.load(
            name=module_name,
            sources=[os.path.join(src_dir, "cb_fused_fp4_gemm.cu")],
            extra_include_paths=incs + [src_dir],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               f"-gencode=arch={arch},code={code}"],
            build_directory=build_dir, verbose=False)
        mod = _require_any_symbol_family(
            mod, _FUSED_FP4_SYMBOL_FAMILIES, build_dir=build_dir,
            source="cb_fused_fp4_gemm.cu")
        _fused_fp4 = _require_fused_fp4_identity(
            mod, expected_name=module_name, identity=identity,
            build_dir=build_dir, source="cb_fused_fp4_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: incompatible fused fp4 prefill "
              f"extension — {exc} fp4 prefill stays on the exact native "
              f"BF16/CUTLASS bridge.", file=sys.stderr, flush=True)
        _fused_fp4 = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"fp4 prefill stays on the exact native BF16/CUTLASS bridge.",
              file=sys.stderr, flush=True)
        _fused_fp4 = None
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused fp4 prefill extension "
              f"unavailable ({type(exc).__name__}: {exc}); fp4 prefill stays "
              f"on the exact native BF16/CUTLASS bridge (expected off Blackwell or "
              f"without nvcc).", file=sys.stderr, flush=True)
        _fused_fp4 = None
    finally:
        _fused_fp4_tried = True
    return _fused_fp4


# `_gf_ok` is the prerequisite for both the dense and grouped FP8 fused paths,
# and it dereferences this binding after its capability probe. Grouped bindings
# remain optional additions: `_gf2_ok` requires `_gf_ok` first, then probes
# `cb_fused_moe_grouped` and `cb_fused_moe_tile_m` separately.
_FUSED_SYMBOLS = ("cb_fused_prefill_mm_scaled",)


def get_fused_ext():
    """The CUTLASS decode-in-prologue prefill extension (cb_fused_gemm.cu),
    or None. Separate module from the GEMV ext: it needs the CUTLASS headers
    (taken from the vLLM install's bundled copy), a longer JIT build, and the
    architecture-accelerated ``sm_120a``/``sm_121a`` target required by its
    conditional tensor-core instructions. Fail-soft like get_ext — serving
    falls back to the exact native expansion path."""
    if _fused_tried:
        return _fused
    with _fused_lock:
        if _fused_tried:
            return _fused
        return _load_fused_ext_locked()


def _load_fused_ext_locked():
    """Build and publish fused FP8 with ``_fused_lock`` held."""
    global _fused, _fused_tried
    try:
        import torch
        from torch.utils.cpp_extension import load

        cc = torch.cuda.get_device_capability()
        if cc not in ((12, 0), (12, 1)):
            raise RuntimeError(
                "fused FP8-CB prefill requires compute capability 12.0 or "
                f"12.1, got {cc[0]}.{cc[1]}")
        cut_inc = _find_cutlass_include()
        cut_root = os.path.dirname(cut_inc)
        src_dir = _require_csrc("cb_fused_gemm.cu")
        build_dir = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_dir, "fused")
        os.makedirs(build_dir, exist_ok=True)
        arch = f"compute_{cc[0]}{cc[1]}a"
        code = f"sm_{cc[0]}{cc[1]}a"
        mod = load(name="pq_cb_fused",
                   sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
                   extra_include_paths=[cut_inc,
                                        os.path.join(cut_root, "tools", "util",
                                                     "include"),
                                        src_dir],
                   extra_cuda_cflags=[
                       "-O3", "--expt-relaxed-constexpr",
                       f"-gencode=arch={arch},code={code}",
                   ],
                   build_directory=build_dir, verbose=False)
        _fused = _require_symbols(mod, _FUSED_SYMBOLS,
                                  build_dir=build_dir,
                                  source="cb_fused_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: incompatible fused prefill "
              f"extension — {exc} Fused dense and grouped prefill stay on "
              f"their exact native routes.",
              file=sys.stderr, flush=True)
        _fused = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"Fused dense and grouped prefill stay on their exact native routes.",
              file=sys.stderr, flush=True)
        _fused = None
    except Exception as exc:  # noqa: BLE001
        print(f"[prismaquant-cb] WARNING: fused prefill extension unavailable "
              f"({type(exc).__name__}: {exc}); fused dense and grouped "
              f"prefill stay on their exact native routes (this is expected on "
              f"non-sm_120 GPUs and without nvcc).",
              file=sys.stderr, flush=True)
        _fused = None
    finally:
        _fused_tried = True
    return _fused
