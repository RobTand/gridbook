"""One implementation of Gridbook's environment-flag and lane-attestation rules.

Three opt-in lanes (``bf16_grouped_lane``, ``fp4v2_fused_midm_lane``,
``moe_persistent_b_lane``) each carried a byte-identical ~30-line copy of the
same two things: a process-stable flag read that raises on a typo, and a
load-time attestation that fails closed rather than substituting the other
route. Copies drift, and they had: one lane's symbol list was a strict subset
of the one ``cuda_ext`` enforces while its comment claimed "the two lists now
agree", and two of the three computed the live device's compute capability and
then DISCARDED it unless a symbol was missing. Both defects are structural —
they are what a copy is for — so the mechanism lives here once.

The OLDER flags were separately inconsistent with the newer lanes' semantics,
in ways that all failed OPEN:

* ``PRISMAQUANT_PRELOAD_FUSED`` compared ``== "1"``, so ``true`` and ``" 1 "``
  silently warmed nothing and the operator got an unlabelled unmatched A/B;
* ``PRISMAQUANT_CB_FUSED_MIDM`` compared ``!= "0"``, so ``false`` and ``off``
  ENABLED the lane they read as disabling, and it was read unlatched at three
  sites, so two of them could disagree inside one forward;
* ``PRISMAQUANT_CB_GROUPED_TRIM`` compared ``== "1"``, so any typo silently
  selected the non-default arm;
* ``PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK`` / ``..._CHUNK_BYTES`` were read per
  CALL. The chunk gates the swizzle-group packed expert ORDER, so changing
  either mid-process silently changes the FP32 reduction order between two
  forwards of one run — the exact class of thing the lane flags are latched to
  prevent.

Every flag now goes through :func:`latched_bool` / :func:`latched_int`: parsed
strictly (a typo raises and names the accepted spellings), and latched to the
first value observed (a later change raises rather than mixing two dispatch
behaviours inside one run).
"""
from __future__ import annotations

import os

# flag name -> the raw string this process latched. One dict rather than a
# module-level list per flag, so ``reset_for_tests`` cannot miss one.
_LATCHED: dict[str, str] = {}

_TRUE = ("1",)
_FALSE = ("", "0")


def reset_for_tests(flag: str | None = None) -> None:
    """Clear one latch, or all of them (tests only)."""
    if flag is None:
        _LATCHED.clear()
    else:
        _LATCHED.pop(flag, None)


def _latch(flag: str, current: str) -> str:
    """Pin ``flag`` to its first observed value; raise if it changes."""
    if flag not in _LATCHED:
        _LATCHED[flag] = current
    elif current != _LATCHED[flag]:
        raise RuntimeError(
            f"{flag} changed after Gridbook dispatch was fixed (was "
            f"{_LATCHED[flag]!r}, now {current!r}); restart the process "
            f"instead of changing dispatch behaviour within one run")
    return _LATCHED[flag]


def latched_bool(flag: str, *, default: bool = False,
                 meaning: str = "this lane") -> bool:
    """A strictly parsed, process-stable on/off flag.

    Accepts only ``''`` (unset), ``'0'`` and ``'1'``, surrounding whitespace
    stripped. Anything else raises and names the accepted spellings: a flag
    that silently ignores ``true`` or treats ``off`` as ON turns an intended
    enablement A/B into an unlabelled run of the other arm, which is worse than
    a crash because the numbers look fine.

    ``default`` is what an UNSET flag means; ``'0'`` and ``'1'`` always mean
    exactly themselves, so an opt-out flag is spelled ``default=True`` rather
    than by inverting the comparison at the call site.
    """
    current = os.environ.get(flag, "").strip()
    if current not in _TRUE + _FALSE:
        raise ValueError(
            f"invalid {flag}={current!r}; expected '1' to enable {meaning}, "
            f"'0' to disable it, or leave it unset for the default "
            f"({'enabled' if default else 'disabled'})")
    value = _latch(flag, current)
    if value == "":
        return default
    return value in _TRUE


def latched_int(flag: str, *, default: int, minimum: int = 1,
                meaning: str = "this value") -> int:
    """A strictly parsed, process-stable integer knob.

    Latched for the same reason the lane selectors are: these knobs feed
    dispatch decisions (the expert-chunk width gates the packed expert ORDER,
    hence the FP32 reduction order), so a value that changes between two
    forwards of one run makes the run's numbers describe neither setting.
    """
    current = os.environ.get(flag, "").strip()
    value = _latch(flag, current)
    if value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid {flag}={value!r}; expected an integer >= {minimum} for "
            f"{meaning}, or leave it unset for the default {default}") from exc
    if parsed < minimum:
        raise ValueError(
            f"{flag}={parsed} is below the minimum {minimum} for {meaning}")
    return parsed


def _device_capability(device):
    """``(major, minor)`` for ``device``, or ``None`` if CUDA cannot say."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(device)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 — reported by the caller as "unavailable"
        return None


def require_lane(operation: str, *, flag: str, lane: str, source: str,
                 alternative: str, get_ext, symbols, buildable,
                 device=None, prepare: str | None = None):
    """Return the extension carrying an opt-in lane, or fail closed.

    Called at MODEL LOAD, never at first forward. Failing closed is the point:
    with the flag on, quietly serving ``alternative`` would produce a run whose
    numbers describe a kernel the operator did not select.

    ``symbols`` is passed by the caller as ``cuda_ext``'s own strict tuple, not
    a local restatement — a lane whose local list was one name short of the
    loader's, under a comment claiming the two agreed, is why this argument
    exists rather than a per-lane literal.

    THE DEVICE IS ACTUALLY CHECKED. Two of the three lanes used to compute the
    live capability and then use it only to decorate a symbols-missing message,
    so on a mixed-capability box a module built for device 0 was attested for
    device N and failed at first launch — the one place this function exists to
    make impossible. Both the capability and, when the module records it, the
    capability the module was BUILT for are compared against ``device`` here.

    ``prepare`` names an optional per-device binding (the shared-memory opt-in).
    Calling it here is what keeps ``cudaFuncSetAttribute`` — which is not
    stream-ordered work — out of a first forward and out of a graph capture.
    """
    from .cuda_ext import NativeKernelUnavailableError

    capability = _device_capability(device)
    ext = get_ext()
    missing = [name for name in symbols
               if ext is None or not hasattr(ext, name)]
    if missing:
        raise NativeKernelUnavailableError(
            f"{operation} requested {lane} ({flag}=1), but Gridbook's "
            f"{source} is unavailable or does not carry it (missing "
            f"{missing}; device capability {capability}). The lane is compiled "
            f"only for compute capability 12.0/12.1"
            + ("" if capability is None or buildable(capability)
               else f", and this device reports "
                    f"{capability[0]}.{capability[1]}")
            + f". Unset {flag} to use {alternative}; Gridbook does not "
            f"substitute a different kernel behind an explicit lane selection.")

    if capability is not None and not buildable(capability):
        raise NativeKernelUnavailableError(
            f"{operation} requested {lane} ({flag}=1), and the loaded {source} "
            f"carries it, but THIS device reports compute capability "
            f"{capability[0]}.{capability[1]}, which the lane is not compiled "
            f"for. On a mixed-capability host the module is built for one "
            f"device and would abort at launch on another. Unset {flag} to "
            f"use {alternative}.")

    built_for = getattr(ext, "__gridbook_jit_capability__", None)
    if (built_for is not None and capability is not None
            and tuple(built_for) != tuple(capability)):
        raise NativeKernelUnavailableError(
            f"{operation} requested {lane} ({flag}=1), but the loaded {source} "
            f"was compiled for compute capability {built_for[0]}."
            f"{built_for[1]} while this device reports {capability[0]}."
            f"{capability[1]}. These lanes pin an architecture-ACCELERATED "
            f"target (sm_120a / sm_121a), so the binary is not portable "
            f"between them and would abort at launch rather than run slowly. "
            f"Serve this device from its own process, or unset {flag} to use "
            f"{alternative}.")

    if prepare is not None:
        # DEVICE ATTESTATION. Loading proves the symbols exist; this proves
        # THIS device can serve them, and opts every compiled configuration in
        # to its dynamic shared-memory budget before any forward runs.
        try:
            if device is None:
                getattr(ext, prepare)()
            else:
                import torch

                with torch.cuda.device(device):
                    getattr(ext, prepare)()
        except Exception as exc:  # noqa: BLE001 — normalize the load-time gate
            raise NativeKernelUnavailableError(
                f"{operation} requested {lane} ({flag}=1), but load-time "
                f"device attestation failed ({type(exc).__name__}: {exc}). "
                f"The schedule needs CUDA compute capability 12.0 or 12.1 with "
                f"enough opt-in shared memory for its largest compiled tile; "
                f"the kernel's own check reports the exact bound. Gridbook "
                f"does not defer this failure to first prefill or serve "
                f"{alternative} instead.") from exc
    return ext
