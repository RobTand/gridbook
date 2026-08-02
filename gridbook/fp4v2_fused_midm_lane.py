"""Selection and attestation of the FP4-CB v2 fused mid-M lane.

The quality FP4-CB dense path materializes the decoded ``[N, K]`` BF16 tile in
HBM before every prefill GEMM. ``csrc/cb_fused_fp4v2_gemm.cu`` removes that
transient at mid M by decoding the packed CB rows inside the CUTLASS
producer/consumer stage — the mechanism the promoted FP8-CB mid-M kernel
already ships (2026-08-01 performance audit §3 P2a, structural cause (c)).

The decoded values are **bit-identical to** ``cb_expand_v2`` and the
activations are the same BF16 group-16 QDQ output the bridge already consumes,
so the lane is CONTRACT-PRESERVING: only the FP32 GEMM reduction order moves.
It is nevertheless **OPT-IN** behind ``PRISMAQUANT_CB_FP4_FUSED_MIDM=1`` until
the served NATIVE-PARITY gate is run, per
[NATIVE-PARITY](../docs/NATIVE-PARITY.md).
Nothing here runs, probes or builds anything when the flag is unset — with the
flag off the dispatch is byte-for-byte what it was.

This module owns four things so the dispatch site stays small: the
process-stable flag read, the load-time attestation (the repo convention is
that nothing resolves at first forward), the per-layer eligibility predicate,
and the call wrapper.
"""
from __future__ import annotations

import os

_FLAG = "PRISMAQUANT_CB_FP4_FUSED_MIDM"
_ALLOWED = frozenset(("", "0", "1"))
_STATE: list[str] = []

# The lane's hard mid-M ceiling, kept here only as the value the tests and the
# docs quote. The KERNEL is authoritative: ``max_m()`` reads it back from the
# loaded module, and the binding itself TORCH_CHECKs it, so a drifted python
# constant can never route an out-of-range M into the kernel.
MID_M_MAX = 128
# Below this the bandwidth-bound decode GEMV owns the shape (linear.py's
# ``CUDA_GEMV_M_MAX``); the fused lane starts one row above it.
MID_M_MIN = 9


def requested() -> bool:
    """Whether the operator asked for the fused mid-M FP4 lane.

    Process-stable like every other Gridbook dispatch selector: a value that
    changed after dispatch was fixed would silently mix two reduction orders
    inside one run and make an A/B unreadable. A typo raises rather than
    quietly selecting the baseline.
    """
    current = os.environ.get(_FLAG, "").strip()
    if current not in _ALLOWED:
        raise ValueError(
            f"invalid {_FLAG}={current!r}; expected '1' to enable the "
            f"contract-preserving FP4-CB fused mid-M lane, or '' / '0' for "
            f"the shipping expand + BF16 bridge route")
    if not _STATE:
        _STATE.append(current)
    elif current != _STATE[0]:
        raise RuntimeError(
            f"{_FLAG} changed after Gridbook dispatch was fixed; restart the "
            f"process instead of mixing GEMM reduction orders within one run")
    return _STATE[0] == "1"


def _reset_for_tests() -> None:
    """Clear the process-stable latch (tests only)."""
    _STATE.clear()


# Exactly the entry points THIS module dereferences. The loader's
# ``_require_symbols`` is what enforces the module's full binding contract (it
# is strict, because the build identity keys the module name and directory);
# this is the lane's own defence in depth, so it stays scoped to what a miss
# here would actually break.
_REQUIRED_SYMBOLS = (
    "cb_fused_fp4v2_prefill_mm",       # fused_mm
    "cb_fused_fp4v2_max_m",            # eligible (via _facts)
    "cb_fused_fp4v2_kbits",            # eligible (via _facts)
)


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension carrying the lane, or fail closed.

    Called at model load, never at first forward. Failing closed is the point:
    with the flag on, quietly serving the expand + bridge route would produce a
    run whose numbers describe the wrong kernel.
    """
    from .cuda_ext import (NativeKernelUnavailableError,
                           fused_fp4v2_buildable, get_fused_fp4v2_ext)

    capability = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 — reported below as "unavailable"
        capability = None

    ext = get_fused_fp4v2_ext()
    missing = [name for name in _REQUIRED_SYMBOLS
               if ext is None or not hasattr(ext, name)]
    if missing:
        raise NativeKernelUnavailableError(
            f"{operation} requested the FP4-CB fused mid-M lane ({_FLAG}=1), "
            f"but Gridbook's fused FP4-v2 quality extension "
            f"(cb_fused_fp4v2_gemm.cu) is unavailable or incomplete "
            f"(missing {missing}; device capability {capability}). "
            f"The lane is compiled only for compute capability 12.0/12.1"
            + ("" if capability is None or fused_fp4v2_buildable(capability)
               else f", and this device reports "
                    f"{capability[0]}.{capability[1]}")
            + f". Unset {_FLAG} to use the shipping expand + BF16 bridge "
            f"route; Gridbook does not substitute a different kernel behind "
            f"an explicit lane selection.")
    return ext


def _facts(ext) -> tuple[int, tuple[int, ...]]:
    """``(max_m, compiled rungs)``, read from the module ONCE.

    ``eligible`` runs on the per-prefill dispatch path, so the two attestation
    queries are memoized on the extension object itself: they are compile-time
    constants of a loaded binary, and this lane exists to save microseconds at
    mid M. The cache lives on the module (not in a dict keyed by identity) so
    it dies exactly when the module does.
    """
    cached = getattr(ext, "_gridbook_fp4v2_facts", None)
    if cached is None:
        cached = (int(ext.cb_fused_fp4v2_max_m()),
                  tuple(int(k) for k in ext.cb_fused_fp4v2_kbits()))
        try:
            ext._gridbook_fp4v2_facts = cached
        except Exception:  # noqa: BLE001 — a read-only stub still works
            pass
    return cached


def max_m(ext) -> int:
    """The lane's hard mid-M ceiling, as the kernel reports it."""
    return _facts(ext)[0]


def kbits(ext) -> tuple[int, ...]:
    """Every rung the loaded module actually compiled a kernel for."""
    return _facts(ext)[1]


def cb_elems(k_bits: int) -> int:
    """Element count of one zero-based fp4-v2 product dictionary."""
    return (4 << ((k_bits + 1) // 2)) + (4 << (k_bits // 2))


def eligible(ext, layer, *, M: int, N: int, K: int, k_bits: int,
             n_sub: int, type_size: int, is_v2: bool) -> bool:
    """Whether THIS call may take the fused mid-M lane.

    Every condition is a property the kernel itself would TORCH_CHECK, checked
    here so a miss falls through to the shipping route instead of raising:

    * mid-M by construction — one M-tile, ``9 <= M <= max_m`` (HARD);
    * fp4-v2 product mode at a compiled rung, ``K % 256 == 0``;
    * N aligned to the BF16 epilogue's 8-element access;
    * exactly ONE zero-based product dictionary. ``cb_expand_v2`` — and hence
      this kernel, which must reproduce it bit-for-bit — takes a single
      physical codebook with no per-row offset, so a fused module whose roles
      are backed by DIFFERENT interned dictionaries stays on the segmented
      bridge. This is the fp4 twin of ``_cb_fp8_fused_lut_ok`` and it is
      answered from load-time metadata, never a device read.
    """
    if ext is None:
        return False
    if not (is_v2 and n_sub == 2 and type_size == 4 * k_bits + 9):
        return False
    if not (MID_M_MIN <= M <= max_m(ext)):
        return False
    if K <= 0 or K % 256 != 0 or N <= 0 or N % 8 != 0:
        return False
    if k_bits not in kbits(ext):
        return False
    cb_flat = getattr(layer, "_cb_flat", None)
    if cb_flat is None or cb_flat.numel() != cb_elems(k_bits):
        return False
    segments = getattr(layer, "_cb_fp4_quality_segments", None)
    if segments is not None and len(segments) != 1:
        return False
    return True


def fused_mm(ext, a, layer, *, N: int, K: int, k_bits: int):
    """One fused decode-in-prologue GEMM over this layer's packed rows.

    ``a`` is the ``[M, K]`` BF16 activation the quality path already produces
    (group-16 QDQ), contiguous. Returns ``[M, N]`` BF16.
    """
    return ext.cb_fused_fp4v2_prefill_mm(
        a, layer._cb_qw_padded, layer._cb_flat, layer._cb_compose,
        N, K, k_bits)
