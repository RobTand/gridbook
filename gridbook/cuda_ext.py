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
cb-ext``. EVERY module builds in its own subdirectory of that root — ``main``,
``v2``, ``bf16_grouped/<identity>``, ``fused/<identity>``,
``fused_fp4/<identity>`` — so no two ninja workspaces share artefacts. Inside the container the root is
ephemeral (one ~30 s build per container start; mount a host dir over it to
persist). Never ``/tmp``. (~30 s is measured: cold ``get_ext()`` in
``vllm-node:latest`` with ``TORCH_CUDA_ARCH_LIST=12.1`` -> 29.4 s / 29.7 s on
two runs. That measurement predates both the ``main`` subdirectory and the
device-derived ``-gencode`` below, each of which costs one further rebuild of
an existing cache — accepted once per user.)

Architecture: every module is compiled for exactly the live device's compute
capability instead of inheriting ``TORCH_CUDA_ARCH_LIST``. The stock vLLM base
image ships ``"8.0 8.7 8.9 9.0 10.0 11.0 12.0"``, which OMITS 12.1, so outside
the Gridbook Dockerfile (which bakes ``12.1a`` globally) an inherited target
left a GB10 running production decode from PTX JIT or a mismatched SASS target
— the 2026-08-01 performance audit's §3 P0.1. A build host with no visible GPU
consequently has no defensible target and each loader reports itself
unavailable; a compile-only environment with nvcc and no GPU pins the target by
overriding ``torch.cuda.get_device_capability`` for the duration of the build
(see the Dockerfile's ``load_for_build``).

Every loader validates the symbols its callers will use before returning a
module. Strict call contracts use :func:`_require_symbols`; fused FP4 uses
independent symbol families because its dense and grouped call sites are
separately guarded. An incompatible module would otherwise fail with
``AttributeError`` mid-forward, or silently disable a probed fast path. The
two fused modules and the grouped BF16 bridge additionally hash their packaged
sources, Gridbook headers, target, compiled-in lane macros and toolchain ABI
into the module name and build directory, so a loaded module of those three is
always built from the current sources.

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


def _target_capability(module: str) -> tuple[int, int]:
    """The compute capability a native module is compiled for.

    Gridbook targets the live device explicitly rather than inheriting
    ``TORCH_CUDA_ARCH_LIST`` (the module docstring records the measured
    reason).
    A host with no visible CUDA device therefore has no defensible target: the
    failure is normalized here into one actionable sentence and lands in the
    caller's fail-soft arm, which reports the module unavailable rather than
    shipping SASS for the wrong chip.
    """
    import torch

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # noqa: BLE001 — one diagnosis for every cause
        raise RuntimeError(
            f"cannot determine which CUDA architecture to compile {module} "
            f"for ({type(exc).__name__}: {exc}); Gridbook compiles for the "
            f"live device instead of inheriting TORCH_CUDA_ARCH_LIST, so a "
            f"visible GPU is required at build time. A compile-only build "
            f"host (nvcc, no GPU) must pin torch.cuda.get_device_capability "
            f"to its intended target for the duration of the build — see the "
            f"Dockerfile's prewarm step.") from exc
    return int(major), int(minor)


def _arch_target(capability: tuple[int, int], *,
                 accelerated: bool) -> tuple[str, str]:
    """``(compute_XY, sm_XY)`` for one capability.

    ``accelerated`` appends nvcc's ``a`` suffix, selecting the architecture-
    CONDITIONAL target. Only the two fused CUTLASS modules may use it: their
    tensor-core instructions (the block-scaled MMA above all) exist solely in
    that arch-specific family. The architecture-generic modules must not, since
    an ``a`` binary refuses to load on any other capability at all.
    """
    major, minor = capability
    suffix = "a" if accelerated else ""
    return (f"compute_{major}{minor}{suffix}", f"sm_{major}{minor}{suffix}")


def _gencode_flag(capability: tuple[int, int], *, accelerated: bool) -> str:
    """The one ``-gencode`` nvcc flag that pins a build to a single target."""
    arch, code = _arch_target(capability, accelerated=accelerated)
    return f"-gencode=arch={arch},code={code}"


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
        # cb_gemv.cu is architecture-GENERIC (sm_80+): plain compute_XY/sm_XY,
        # no arch-conditional 'a' target and no cc floor beyond what the source
        # itself supports. This is the hot decode module, so an inherited
        # TORCH_CUDA_ARCH_LIST miss costs production throughput directly.
        cc = _target_capability("the CUDA decode-GEMV extension (cb_gemv.cu)")
        build_root = os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext")
        # Own subdirectory, like every sibling module: building at the cache
        # root shares one ninja workspace with nothing to distinguish it, and
        # invites artefact collisions with any module added later.
        build_dir = os.path.join(build_root, "main")
        os.makedirs(build_dir, exist_ok=True)
        mod = load(name="prismaquant_cb_ext", sources=[src],
                   build_directory=build_dir,
                   extra_cuda_cflags=[
                       "-O3", _gencode_flag(cc, accelerated=False)],
                   verbose=False)
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
            # Architecture-generic like the main module: the 99 KiB dynamic
            # smem contract is a DEVICE attestation done at load
            # (require_fp4_v2_expander), not a compile-time arch pin.
            cc = _target_capability(
                "the CB-GEMV-v2 extension (cb_gemv_v2.cu)")
            build_dir = os.environ.get(
                "PRISMAQUANT_CB_EXT_DIR") or os.path.join(
                    os.path.expanduser("~"), ".cache", "prismaquant-cb-ext")
            build_dir = os.path.join(build_dir, "v2")
            os.makedirs(build_dir, exist_ok=True)
            mod = load(name="prismaquant_cb_v2_ext", sources=[src],
                       build_directory=build_dir,
                       extra_cuda_cflags=[
                           "-O3", _gencode_flag(cc, accelerated=False)],
                       verbose=False)
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

    KNOWN COUPLING (2026-08-01 performance audit, §4 "Dispatch-surface gaps"):
    because every FP4-CB load path needs this exact expander, this one device
    attestation narrows FP4-CB's hardware floor to cc 12.0/12.1 for the WHOLE
    format — including layers that would otherwise serve happily from the
    inherited (non-v2) GEMV, which has no such requirement. That makes FP4-CB's
    floor strictly narrower than FP8-CB's. It is deliberate: Blackwell
    (sm_120/sm_121) is the target and the v2 module owns the bit-exact
    expander, so there is no second exact FP4 decode to fall back to. Recorded
    here so it stays a documented coupling rather than an accidental discovery
    on an Ada or Hopper box.
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
    """Locate the CUTLASS include dir every owned CUTLASS module compiles with.

    ``PRISMAQUANT_CUTLASS_INCLUDE`` wins when set — it is the only way to build
    these modules in a venv that has no vLLM wheel, or against a CUTLASS newer
    than the bundled copy — and it is read HERE rather than in one loader so
    the grouped-BF16, fused-FP8 and fused-FP4 modules all honour it (before
    2026-08-01 only the fused-FP4 loader did, so the other two simply could not
    build without vLLM's bundled tree). A set-but-wrong override FAILS instead
    of falling back to the bundled copy: silently compiling against different
    headers than the operator asked for is exactly the class of surprise the
    override exists to prevent.

    Otherwise discover vLLM's bundled copy WITHOUT importing vLLM's runtime
    package: importing ``vllm`` merely to locate its files eagerly initializes
    optional compiler backends, including Triton on some releases. Gridbook
    needs none of that state to compile an owned CUTLASS translation unit;
    module discovery gives us the package directory without executing
    ``__init__``.
    """
    import glob
    import importlib.util

    override = os.environ.get("PRISMAQUANT_CUTLASS_INCLUDE")
    if override:
        override = os.path.abspath(os.path.expanduser(os.fspath(override)))
        if not os.path.isfile(
                os.path.join(override, "cutlass", "cutlass.h")):
            raise FileNotFoundError(
                f"PRISMAQUANT_CUTLASS_INCLUDE={override!r} does not contain "
                f"cutlass/cutlass.h. Point it at the `include` directory of a "
                f"CUTLASS checkout (the one holding `cutlass/cutlass.h`), or "
                f"unset it to use the CUTLASS bundled with vLLM. Gridbook "
                f"does not fall back silently: an override that names the "
                f"wrong tree must not compile against a different CUTLASS "
                f"than the one requested.")
        return override

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

# The packaged files that define the grouped BF16 bridge: the translation unit,
# the shared grouping glue and the expert-indexed mainloop fork. All three are
# hashed into the module identity — the module is now header-bearing, and
# torch's extension versioner hashes only the ``.cu`` it is handed (2026-08-01
# performance audit, §3 P0.3/P1).
_BF16_GROUPED_BUILD_INPUTS = (
    "cb_bf16_grouped_gemm.cu",
    "cb_grouped_common.hpp",
    "cutlass_fork/sm120_bf16_expert_mma.hpp",
)
# Starts at 1: this module's identity payload is new.
_BF16_GROUPED_ABI_SCHEMA = 1

# Every device gets the SM80-compatible device-scheduled lane; it is the
# default and its two entry points are dereferenced without a probe.
_BF16_GROUPED_SYMBOLS = (
    "cb_bf16_grouped_mm",
    "cb_bf16_grouped_mm_out",
)
# Additionally required when the module was BUILT for cc 12.x, where the
# sm12x-native lane is compiled in. Strict, not "any useful family": the
# identity keys both the module name and the build directory, so a module that
# loads at all was built from exactly these sources — a missing sm120 binding
# is a broken build, not an older one.
_BF16_GROUPED_SM120_SYMBOLS = (
    "cb_bf16_grouped_mm_sm120",
    "cb_bf16_grouped_mm_sm120_out",
    "cb_bf16_grouped_sm120_tile_m",
    "cb_bf16_grouped_sm120_tile_sizes",
    "cb_bf16_grouped_sm120_config",
)
# The capabilities whose kernel layer the sm12x lane needs. Same set the two
# fused modules gate on.
_BF16_GROUPED_SM120_CAPABILITIES = ((12, 0), (12, 1))
# nvcc macro that compiles the sm12x lane in. It is part of the build identity
# (see ``_bf16_grouped_build_identity``) so a cache entry built without the
# lane can never be served as one that has it.
_BF16_GROUPED_SM120_DEFINE = "PRISMAQUANT_CB_BF16_SM120"


def bf16_grouped_sm120_buildable(capability) -> bool:
    """Whether the sm12x-native BF16 lane is compiled for ``capability``."""
    return tuple(capability) in _BF16_GROUPED_SM120_CAPABILITIES


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


def _bf16_grouped_build_identity(torch, cpp_extension, *, src_dir: str,
                                 cutlass_include: str, util_include: str,
                                 capability: tuple[int, int],
                                 sm120_lane: bool):
    """``(digest, payload)`` for the grouped BF16 module's binary ABI."""
    return _fused_build_identity(
        torch, cpp_extension, src_dir=src_dir,
        cutlass_include=cutlass_include, util_include=util_include,
        capability=capability, build_inputs=_BF16_GROUPED_BUILD_INPUTS,
        bindings=(("grouped BF16 bridge", _BF16_GROUPED_SYMBOLS),
                  ("sm12x-native lane",
                   _BF16_GROUPED_SM120_SYMBOLS if sm120_lane else ())),
        abi_schema=_BF16_GROUPED_ABI_SCHEMA,
        # The sm12x lane needs the architecture-CONDITIONAL target — see the
        # loader below for the measured reason — while every other capability
        # keeps the portable generic one.
        accelerated=sm120_lane,
        defines=({_BF16_GROUPED_SM120_DEFINE: "1"} if sm120_lane else {}))


def _load_bf16_grouped_ext_locked():
    """Build and publish grouped BF16 with ``_bf16_grouped_lock`` held."""
    global _bf16_grouped, _bf16_grouped_tried
    build_dir = "<unresolved>"
    try:
        import torch  # noqa: F401  (must import before cpp_extension)
        from torch.utils import cpp_extension

        cc = _target_capability(
            "the CUTLASS grouped BF16 extension (cb_bf16_grouped_gemm.cu)")
        if cc < (8, 0):
            raise RuntimeError(
                f"CUTLASS grouped BF16 requires compute capability >= 8.0, "
                f"got {cc[0]}.{cc[1]}")
        sm120_lane = bf16_grouped_sm120_buildable(cc)
        src_dir = _require_csrc(*_BF16_GROUPED_BUILD_INPUTS)
        cut_inc = _find_cutlass_include()
        util_inc = os.path.join(os.path.dirname(cut_inc), "tools", "util",
                                "include")
        identity, _identity_payload = _bf16_grouped_build_identity(
            torch, cpp_extension, src_dir=src_dir, cutlass_include=cut_inc,
            util_include=util_inc, capability=cc, sm120_lane=sm120_lane)
        module_name = f"pq_cb_bf16_grouped_{identity}"
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "bf16_grouped", identity)
        os.makedirs(build_dir, exist_ok=True)
        flags = ["-O3", "--expt-relaxed-constexpr"]
        if sm120_lane:
            flags.append(f"-D{_BF16_GROUPED_SM120_DEFINE}=1")
        # MEASURED (GB10, cc 12.1, CUTLASS 4.3.4): the sm12x lane needs the
        # ``a``-suffixed target even though its MMA (`m16n8k16` bf16) is
        # architecture-GENERIC. The reason is the kernel LAYER, not the
        # instruction: sm90_gemm_tma_warpspecialized_pingpong.hpp compiles
        # its operator() body only under __CUDA_ARCH_FEAT_SM90/120/121_ALL (or
        # a conditional/family target), and otherwise emits
        # CUTE_INVALID_CONTROL_PATH. Built as plain ``sm_121`` the module
        # compiles and loads, then every launch aborts with "Arch conditional
        # MMA instruction used without targeting appropriate compute
        # capability". Built as ``sm_121a`` the same source passes its bit
        # tests. Non-12.x devices keep the portable generic target, since they
        # compile only the SM80 lane.
        flags.append(_gencode_flag(cc, accelerated=sm120_lane))
        mod = cpp_extension.load(
            name=module_name,
            sources=[os.path.join(src_dir, "cb_bf16_grouped_gemm.cu")],
            extra_include_paths=[cut_inc, util_inc, src_dir],
            extra_cuda_cflags=flags,
            build_directory=build_dir,
            verbose=False)
        required = _BF16_GROUPED_SYMBOLS + (
            _BF16_GROUPED_SM120_SYMBOLS if sm120_lane else ())
        mod = _require_symbols(
            mod, required, build_dir=build_dir,
            source="cb_bf16_grouped_gemm.cu")
        _bf16_grouped = _require_fused_identity(
            mod, expected_name=module_name, identity=identity,
            abi_schema=_BF16_GROUPED_ABI_SCHEMA,
            build_dir=build_dir, source="cb_bf16_grouped_gemm.cu")
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


# --------------------------------------------------------------------------
# Shared fused-JIT build identity (ROADMAP K0.3)
#
# torch's extension versioner hashes the ``.cu`` passed to ``load``, but an
# INCLUDED header is invisible to the stable module name and to a
# caller-selected build directory. Both fused modules include Gridbook-owned
# ``cutlass_fork`` headers, so without this a header edit silently serves the
# previously cached kernel. The fused-FP4 loader solved that in 0.4.2; the
# facility is parameterized here and applied to the fused-FP8 module too
# (2026-08-01 performance audit, §3 P0.3).
#
# The digest keys BOTH the extension module name and the build directory, so a
# persistent cache can never reuse a binary across a change to the packaged
# sources/headers, the target architecture, the Python/Torch ABI, the CUDA
# toolkit, or the host compiler. A small set of external CUTLASS sentinels is
# hashed as well; inside a digest-keyed directory ninja's generated dependency
# file remains authoritative for the complete external include graph.
# --------------------------------------------------------------------------
def _fused_build_identity(torch, cpp_extension, *, src_dir: str,
                          cutlass_include: str,
                          util_include: str,
                          capability: tuple[int, int],
                          build_inputs: tuple[str, ...],
                          bindings,
                          abi_schema: int,
                          accelerated: bool = True,
                          defines: dict[str, str] | None = None):
    """Return ``(digest, payload)`` for every practical binary ABI input.

    ``build_inputs`` are package-relative paths under ``src_dir`` (the ``.cu``
    plus every Gridbook header it includes). ``bindings`` is the module's
    symbol contract as ``(label, names)`` pairs, so a change to what the loader
    requires also invalidates the cache. ``abi_schema`` is the module's own
    revision counter for this payload's SHAPE — bump it when the meaning of a
    field changes rather than its value.

    ``accelerated`` selects the architecture-conditional (``a``-suffixed)
    target recorded in the identity; both fused modules always use it, while
    the grouped BF16 module uses it only where it compiles its sm12x lane.
    ``defines`` are nvcc ``-D`` macros that change which code is compiled —
    a lane compiled out is a different binary and must be a different cache
    entry. The key is OMITTED when empty so a module that passes no macros
    keeps the payload shape it had before this parameter existed.
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
        for name in build_inputs
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
    compute, code = _arch_target((major, minor), accelerated=accelerated)
    payload = {
        "schema": abi_schema,
        "bindings": [
            [label, list(names)] for label, names in bindings
        ],
        "inputs": packaged_inputs,
        "cutlass_inputs": cutlass_inputs,
        "target": {
            "capability": [major, minor],
            "compute": compute,
            "code": code,
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
    if defines:
        payload["defines"] = {str(k): str(v) for k, v in defines.items()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _require_fused_identity(mod, *, expected_name: str, identity: str,
                            abi_schema: int, build_dir: str, source: str):
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
    mod.__gridbook_jit_abi_schema__ = abi_schema
    return mod


_fused_fp4 = None
_fused_fp4_tried = False
_fused_fp4_lock = threading.Lock()


# The packaged files that define the Gridbook-owned fused FP4 implementation.
# All are explicit build inputs (see the shared identity block above);
# ``cb_grouped_common.hpp`` joined them when the row-padded tile-indexed
# grouping glue was extracted, and ``sm120_expert_row_broadcast.hpp`` comes in
# through it — exactly the stale-kernel class this mechanism exists to prevent.
_FUSED_FP4_BUILD_INPUTS = (
    "cb_fused_fp4_gemm.cu",
    "cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    "cb_grouped_common.hpp",
    "cutlass_fork/sm120_expert_row_broadcast.hpp",
)
# Unchanged at 4: extracting the mechanism altered neither this module's digest
# inputs nor the meaning of any payload field, so every already-built FP4 cache
# entry stays valid. Bumping it here would have thrown away good binaries.
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
    """``(digest, payload)`` for the fused NVFP4-CB module's binary ABI."""
    return _fused_build_identity(
        torch, cpp_extension, src_dir=src_dir,
        cutlass_include=cutlass_include, util_include=util_include,
        capability=capability, build_inputs=_FUSED_FP4_BUILD_INPUTS,
        bindings=_FUSED_FP4_SYMBOL_FAMILIES,
        abi_schema=_FUSED_FP4_ABI_SCHEMA)


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

        # FIRST, before any include discovery or build work: the block-scaled
        # MMA this module is built around exists only on sm_120a/sm_121a, so
        # every other capability is a doomed multi-minute nvcc run inside the
        # user's first request. Same early rejection the fused FP8 loader has
        # (2026-08-01 performance audit §3 P0.2; ROADMAP "architecture
        # precheck before the fused CUTLASS build").
        cc = _target_capability(
            "the fused NVFP4-CB prefill extension (cb_fused_fp4_gemm.cu)")
        if cc not in ((12, 0), (12, 1)):
            raise RuntimeError(
                "fused NVFP4-CB prefill requires compute capability 12.0 or "
                f"12.1, got {cc[0]}.{cc[1]}")
        cut_inc = _find_cutlass_include()
        # cutlass/util/packed_stride.hpp lives in the tools tree of a CUTLASS
        # checkout (vLLM's bundled copy keeps the same shape).
        incs = [cut_inc]
        util_inc = os.path.join(os.path.dirname(cut_inc), "tools", "util",
                                "include")
        if os.path.isdir(util_inc):
            incs.append(util_inc)
        src_dir = _require_csrc(*_FUSED_FP4_BUILD_INPUTS)
        identity, _identity_payload = _fused_fp4_build_identity(
            torch, cpp_extension, src_dir=src_dir,
            cutlass_include=cut_inc, util_include=util_inc,
            capability=cc)
        module_name = f"pq_cb_fused_fp4_{identity}"
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "fused_fp4", identity)
        os.makedirs(build_dir, exist_ok=True)
        mod = cpp_extension.load(
            name=module_name,
            sources=[os.path.join(src_dir, "cb_fused_fp4_gemm.cu")],
            extra_include_paths=incs + [src_dir],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               _gencode_flag(cc, accelerated=True)],
            build_directory=build_dir, verbose=False)
        mod = _require_any_symbol_family(
            mod, _FUSED_FP4_SYMBOL_FAMILIES, build_dir=build_dir,
            source="cb_fused_fp4_gemm.cu")
        _fused_fp4 = _require_fused_identity(
            mod, expected_name=module_name, identity=identity,
            abi_schema=_FUSED_FP4_ABI_SCHEMA,
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


# The packaged files that define the fused FP8-CB implementation: the
# translation unit plus every Gridbook-owned header it includes. Each is hashed
# into the module identity, so editing a fork header can no longer serve a
# stale cached kernel (2026-08-01 performance audit §3 P0.3).
_FUSED_BUILD_INPUTS = (
    "cb_fused_gemm.cu",
    "cutlass_fork/sm120_cb_mma_tma.hpp",
    "cutlass_fork/sm120_cb_fused_mma.hpp",
    "cutlass_fork/sm120_expert_row_broadcast.hpp",
    # Shared grouping glue (EVT trees, smem gate, host validation).
    "cb_grouped_common.hpp",
)
# Starts at 1: this module's identity payload is new, so there is no older
# schema of it to distinguish from. (FP4 keeps its own counter at 4.)
_FUSED_ABI_SCHEMA = 1


# STRICT, not "any useful family". Because the identity above keys both the
# module name and the build directory, a module that loads at all was built
# from exactly these sources — so a missing grouped binding is not an older
# build, it is a broken one, and accepting it would silently downgrade routed
# prefill. `moe._gf2_ok` therefore dereferences the grouped family after a
# capability probe alone, and `linear` dereferences the dense entry point.
_FUSED_SYMBOLS = (
    "cb_fused_prefill_mm_scaled",          # dense mid-M + linear.py prefill
    "cb_fused_moe_grouped",                # routed prefill, one launch/stage
    "cb_fused_moe_tile_m",                 # the kernel's default TileM
    "cb_fused_moe_tile_sizes",             # every compiled TileM
    "cb_fused_moe_tile_sizes_for_kbits",   # ... that this rung can serve
)


def get_fused_ext():
    """The CUTLASS decode-in-prologue prefill extension (cb_fused_gemm.cu),
    or None. Separate module from the GEMV ext: it needs the CUTLASS headers
    (``PRISMAQUANT_CUTLASS_INCLUDE``, else the vLLM install's bundled copy), a
    longer JIT build, and the architecture-accelerated ``sm_120a``/``sm_121a``
    target required by its conditional tensor-core instructions. Its packaged
    sources, ``cutlass_fork`` headers and toolchain ABI key both the module
    name and the build directory, so what loads is always built from the
    current sources — which is what lets the dense AND grouped bindings be
    required strictly. Fail-soft like get_ext — serving falls back to the exact
    native expansion path."""
    if _fused_tried:
        return _fused
    with _fused_lock:
        if _fused_tried:
            return _fused
        return _load_fused_ext_locked()


def _fused_build_identity_fp8(torch, cpp_extension, *, src_dir: str,
                              cutlass_include: str,
                              util_include: str,
                              capability: tuple[int, int]):
    """``(digest, payload)`` for the fused FP8-CB module's binary ABI."""
    return _fused_build_identity(
        torch, cpp_extension, src_dir=src_dir,
        cutlass_include=cutlass_include, util_include=util_include,
        capability=capability, build_inputs=_FUSED_BUILD_INPUTS,
        bindings=(("fused FP8-CB prefill", _FUSED_SYMBOLS),),
        abi_schema=_FUSED_ABI_SCHEMA)


def _load_fused_ext_locked():
    """Build and publish fused FP8 with ``_fused_lock`` held."""
    global _fused, _fused_tried
    try:
        import torch
        from torch.utils import cpp_extension

        cc = _target_capability(
            "the fused FP8-CB prefill extension (cb_fused_gemm.cu)")
        if cc not in ((12, 0), (12, 1)):
            raise RuntimeError(
                "fused FP8-CB prefill requires compute capability 12.0 or "
                f"12.1, got {cc[0]}.{cc[1]}")
        cut_inc = _find_cutlass_include()
        cut_root = os.path.dirname(cut_inc)
        util_inc = os.path.join(cut_root, "tools", "util", "include")
        src_dir = _require_csrc(*_FUSED_BUILD_INPUTS)
        identity, _identity_payload = _fused_build_identity_fp8(
            torch, cpp_extension, src_dir=src_dir,
            cutlass_include=cut_inc, util_include=util_inc,
            capability=cc)
        module_name = f"pq_cb_fused_{identity}"
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "fused", identity)
        os.makedirs(build_dir, exist_ok=True)
        mod = cpp_extension.load(
            name=module_name,
            sources=[os.path.join(src_dir, "cb_fused_gemm.cu")],
            extra_include_paths=[cut_inc, util_inc, src_dir],
            extra_cuda_cflags=[
                "-O3", "--expt-relaxed-constexpr",
                _gencode_flag(cc, accelerated=True),
            ],
            build_directory=build_dir, verbose=False)
        mod = _require_symbols(mod, _FUSED_SYMBOLS,
                               build_dir=build_dir,
                               source="cb_fused_gemm.cu")
        _fused = _require_fused_identity(
            mod, expected_name=module_name, identity=identity,
            abi_schema=_FUSED_ABI_SCHEMA,
            build_dir=build_dir, source="cb_fused_gemm.cu")
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


# ===========================================================================
# BEGIN P2a BLOCK — FP4-CB v2 fused mid-M lane loader
# (2026-08-01 performance audit §3 P2a; csrc/cb_fused_fp4v2_gemm.cu).
# Purely additive: nothing above this line is touched.
# ===========================================================================
_fused_fp4v2 = None
_fused_fp4v2_tried = False
_fused_fp4v2_lock = threading.Lock()


# The packaged files that define the contract-preserving fused FP4-v2 quality
# lane: the translation unit, its CB->BF16 decode-in-prologue mainloop fork and
# the shared grouping glue it takes ``AssertSmemFits`` from (which pulls in the
# expert-row-broadcast header). All are hashed into the module identity — the
# module is header-bearing and torch's extension versioner hashes only the
# ``.cu`` it is handed.
_FUSED_FP4V2_BUILD_INPUTS = (
    "cb_fused_fp4v2_gemm.cu",
    "cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp",
    "cutlass_fork/sm120_bf16_expert_mma.hpp",
    "cb_grouped_common.hpp",
    "cutlass_fork/sm120_expert_row_broadcast.hpp",
)
# Starts at 1: this module's identity payload is new.
_FUSED_FP4V2_ABI_SCHEMA = 1

# STRICT, not "any useful family". The identity keys both the module name and
# the build directory, so a module that loads at all was built from exactly
# these sources — a missing binding is a broken build, not an older one, and
# accepting it would let an explicit lane selection serve a different kernel.
_FUSED_FP4V2_SYMBOLS = (
    "cb_fused_fp4v2_prefill_mm",       # the dense mid-M entry point
    "sm120_fp4v2_bf16_mm_fork",        # the decode bit-exactness oracle
    "cb_fused_fp4v2_max_m",            # the HARD mid-M ceiling
    "cb_fused_fp4v2_kbits",            # every compiled rung
    "cb_fused_fp4v2_lut_classes",      # compiled codebook smem-stage classes
    "cb_fused_fp4v2_config",           # [tile_m, tile_n, tile_k, stages, cap]
    "cb_fused_fp4v2_smem_report",      # measured SharedStorageSize per class
    "cb_fused_fp4v2_lut_plan",         # the codebook residency ladder
)

# The capabilities whose kernel layer this lane needs. Same set the two fused
# modules and the sm12x BF16 lane gate on.
_FUSED_FP4V2_CAPABILITIES = ((12, 0), (12, 1))


def fused_fp4v2_buildable(capability) -> bool:
    """Whether the fused FP4-v2 quality lane is compiled for ``capability``."""
    return tuple(capability) in _FUSED_FP4V2_CAPABILITIES


def _fused_fp4v2_build_identity(torch, cpp_extension, *, src_dir: str,
                                cutlass_include: str,
                                util_include: str,
                                capability: tuple[int, int]):
    """``(digest, payload)`` for the fused FP4-v2 module's binary ABI."""
    return _fused_build_identity(
        torch, cpp_extension, src_dir=src_dir,
        cutlass_include=cutlass_include, util_include=util_include,
        capability=capability, build_inputs=_FUSED_FP4V2_BUILD_INPUTS,
        bindings=(("fused FP4-CB v2 quality mid-M lane",
                   _FUSED_FP4V2_SYMBOLS),),
        abi_schema=_FUSED_FP4V2_ABI_SCHEMA)


def get_fused_fp4v2_ext():
    """The CONTRACT-PRESERVING fused FP4-CB v2 quality prefill extension
    (cb_fused_fp4v2_gemm.cu), or None.

    A separate module from both ``get_fused_ext`` (FP8-CB) and
    ``get_fused_fp4_ext`` (the native-NVFP4 block-scaled lane, a DIFFERENT
    served activation contract). This one decodes packed CB rows to BF16
    values bit-identical to ``cb_expand_v2`` and multiplies them against the
    same BF16 group-16-QDQ'd activations the shipping bridge consumes, so the
    only requalification surface is the FP32 reduction order.

    Needs the CUTLASS headers (``PRISMAQUANT_CUTLASS_INCLUDE``, else the vLLM
    install's bundled copy) and the architecture-accelerated
    ``sm_120a``/``sm_121a`` target: the sm90-family cooperative kernel layer
    compiles its body only under the arch-feature macro, so a plain ``sm_121``
    build loads and then aborts at launch. Fail-soft like every other loader —
    a capability probe may inspect the ``None``; the dispatch calls
    ``fp4v2_fused_midm_lane.require_lane`` and fails closed."""
    if _fused_fp4v2_tried:
        return _fused_fp4v2
    with _fused_fp4v2_lock:
        if _fused_fp4v2_tried:
            return _fused_fp4v2
        return _load_fused_fp4v2_ext_locked()


def _load_fused_fp4v2_ext_locked():
    """Build and publish fused FP4-v2 with ``_fused_fp4v2_lock`` held."""
    global _fused_fp4v2, _fused_fp4v2_tried
    try:
        import torch
        from torch.utils import cpp_extension

        # FIRST, before any include discovery or build work: this lane's
        # kernel layer exists only on sm_120a/sm_121a, so every other
        # capability is a doomed multi-minute nvcc run inside the user's first
        # request (2026-08-01 audit §3 P0.2).
        cc = _target_capability(
            "the fused FP4-CB v2 quality prefill extension "
            "(cb_fused_fp4v2_gemm.cu)")
        if not fused_fp4v2_buildable(cc):
            raise RuntimeError(
                "the fused FP4-CB v2 quality mid-M lane requires compute "
                f"capability 12.0 or 12.1, got {cc[0]}.{cc[1]}")
        cut_inc = _find_cutlass_include()
        incs = [cut_inc]
        util_inc = os.path.join(os.path.dirname(cut_inc), "tools", "util",
                                "include")
        if os.path.isdir(util_inc):
            incs.append(util_inc)
        src_dir = _require_csrc(*_FUSED_FP4V2_BUILD_INPUTS)
        identity, _identity_payload = _fused_fp4v2_build_identity(
            torch, cpp_extension, src_dir=src_dir,
            cutlass_include=cut_inc, util_include=util_inc, capability=cc)
        module_name = f"pq_cb_fused_fp4v2_{identity}"
        build_root = (os.environ.get("PRISMAQUANT_CB_EXT_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "prismaquant-cb-ext"))
        build_dir = os.path.join(build_root, "fused_fp4v2", identity)
        os.makedirs(build_dir, exist_ok=True)
        mod = cpp_extension.load(
            name=module_name,
            sources=[os.path.join(src_dir, "cb_fused_fp4v2_gemm.cu")],
            extra_include_paths=incs + [src_dir],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                               _gencode_flag(cc, accelerated=True)],
            build_directory=build_dir, verbose=False)
        mod = _require_symbols(
            mod, _FUSED_FP4V2_SYMBOLS, build_dir=build_dir,
            source="cb_fused_fp4v2_gemm.cu")
        _fused_fp4v2 = _require_fused_identity(
            mod, expected_name=module_name, identity=identity,
            abi_schema=_FUSED_FP4V2_ABI_SCHEMA,
            build_dir=build_dir, source="cb_fused_fp4v2_gemm.cu")
    except StaleExtensionError as exc:
        print(f"[prismaquant-cb] ERROR: incompatible fused FP4-v2 quality "
              f"prefill extension — {exc} FP4 prefill stays on the exact "
              f"native BF16 expand + CUTLASS bridge.",
              file=sys.stderr, flush=True)
        _fused_fp4v2 = None
    except IncompleteInstallError as exc:
        print(f"[prismaquant-cb] ERROR: broken gridbook install — {exc} "
              f"FP4 prefill stays on the exact native BF16 expand + CUTLASS "
              f"bridge.", file=sys.stderr, flush=True)
        _fused_fp4v2 = None
    except Exception as exc:  # noqa: BLE001 — capability probe is fail-soft
        print(f"[prismaquant-cb] WARNING: fused FP4-v2 quality prefill "
              f"extension unavailable ({type(exc).__name__}: {exc}); FP4 "
              f"prefill stays on the exact native BF16 expand + CUTLASS "
              f"bridge (expected off Blackwell or without nvcc).",
              file=sys.stderr, flush=True)
        _fused_fp4v2 = None
    finally:
        _fused_fp4v2_tried = True
    return _fused_fp4v2
# =========================== END P2a BLOCK =================================
