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

from . import lane_select

_FLAG = "PRISMAQUANT_CB_FP4_FUSED_MIDM"

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
    return lane_select.latched_bool(
        _FLAG, meaning="the contract-preserving FP4-CB fused mid-M lane")


def _reset_for_tests() -> None:
    """Clear the process-stable latch (tests only)."""
    lane_select.reset_for_tests(_FLAG)


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension carrying the lane, or fail closed.

    Called at model load, never at first forward. Failing closed is the point:
    with the flag on, quietly serving the expand + bridge route would produce a
    run whose numbers describe the wrong kernel.

    The symbol list is ``cuda_ext``'s own strict tuple rather than a local
    subset, and ``cb_fused_fp4v2_prepare`` is called here so the lane's 99 KiB
    dynamic-shared-memory opt-in — a ``cudaFuncSetAttribute``, which is not
    stream-ordered work — happens at LOAD instead of inside whichever forward
    happens to be first, possibly a CUDA-graph capture.
    """
    from .cuda_ext import (_FUSED_FP4V2_SYMBOLS, fused_fp4v2_buildable,
                           get_fused_fp4v2_ext)

    return lane_select.require_lane(
        operation, flag=_FLAG,
        lane="the FP4-CB fused mid-M lane",
        source="fused FP4-v2 quality extension (cb_fused_fp4v2_gemm.cu)",
        alternative="the shipping expand + BF16 bridge route",
        get_ext=get_fused_fp4v2_ext,
        symbols=_FUSED_FP4V2_SYMBOLS,
        buildable=fused_fp4v2_buildable,
        device=device,
        prepare="cb_fused_fp4v2_prepare")


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
        # GUARDED, on the model of ``fp8_fused_lane.compiled_kbits``. This runs
        # on the per-prefill dispatch path via ``eligible``, where the
        # docstring promises a miss falls through to the shipping route — so a
        # module that will not answer must read as "offers nothing", not raise
        # mid-request. ``require_lane`` already dereferenced both entry points
        # at load, so reaching this arm in production means something changed
        # under us, and the safe reading of that is the conservative one.
        try:
            cached = (int(ext.cb_fused_fp4v2_max_m()),
                      tuple(int(k) for k in ext.cb_fused_fp4v2_kbits()))
        except Exception:  # noqa: BLE001 — treat as "not offered"
            return (0, ())
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


def supports(ext, layer, *, N: int, K: int, k_bits: int, n_sub: int,
             type_size: int, is_v2: bool) -> str | None:
    """``None`` when this LAYER can use the lane, else the reason it cannot.

    Every clause is an M-INDEPENDENT property known at model load, which is why
    it is separated from the per-call ``eligible`` below: with the flag on, a
    layer that can never take the lane would otherwise serve the expand +
    bridge route for every request while the load reported success — the same
    silent substitution behind an explicit selection that the flag exists to
    prevent, and that the sibling persistent-B lane already fails the load for
    (``moe_persistent_b_lane.supports``). ``linear.py`` calls this at load and
    raises.

    The M band is deliberately NOT here: ``9 <= M <= max_m`` is a property of
    the REQUEST, and falling through for an out-of-band M is the documented
    behaviour (docs/PLUGIN.md), not a silent substitution — the lane is a mid-M
    lane and says so.
    """
    if ext is None:
        return "the fused FP4-v2 quality extension is unavailable"
    if not (is_v2 and n_sub == 2 and type_size == 4 * k_bits + 9):
        return ("format is not FP4-CB two-tier v2 product mode (n_sub=2, "
                f"type_size=4k+9), got n_sub={n_sub}, type_size={type_size}")
    if k_bits not in kbits(ext):
        return (f"k={k_bits} is not a rung this build compiled "
                f"({list(kbits(ext))})")
    if K <= 0 or K % 256 != 0:
        return f"K={K} is not a positive multiple of the 256-element superblock"
    if N <= 0 or N % 8 != 0:
        return (f"N={N} is not a positive multiple of the BF16 epilogue's "
                f"8-element access")
    # Exactly ONE zero-based product dictionary. ``cb_expand_v2`` — and hence
    # this kernel, which must reproduce it bit-for-bit — takes a single
    # physical codebook with no per-row offset, so a fused module whose roles
    # are backed by DIFFERENT interned dictionaries can never take this lane.
    # The fp4 twin of ``_cb_fp8_fused_lut_ok``; answered from load-time
    # metadata, never a device read.
    cb_flat = getattr(layer, "_cb_flat", None)
    if cb_flat is None or cb_flat.numel() != cb_elems(k_bits):
        return ("the layer's flat product dictionary is missing or is not the "
                f"single zero-based one this lane decodes ({cb_elems(k_bits)} "
                f"elements)")
    segments = getattr(layer, "_cb_fp4_quality_segments", None)
    if segments is not None and len(segments) != 1:
        return (f"this fused projection spans {len(segments)} interned "
                f"codebook blocks; the lane decodes exactly one")
    return None


def eligible(ext, layer, *, M: int, N: int, K: int, k_bits: int,
             n_sub: int, type_size: int, is_v2: bool) -> bool:
    """Whether THIS call may take the fused mid-M lane.

    The layer-level conditions are :func:`supports`, which ``linear.py``
    enforces at model LOAD; re-checked here so a manually constructed test
    layer, or anything that reached dispatch without the load gate, falls
    through to the shipping route instead of raising. What this adds is the
    per-request condition: mid-M by construction, one M-tile,
    ``9 <= M <= max_m`` (HARD, and enforced again by the kernel's own
    TORCH_CHECK so a drifted python constant cannot route past it).
    """
    if ext is None:
        return False
    if not (MID_M_MIN <= M <= max_m(ext)):
        return False
    return supports(ext, layer, N=N, K=K, k_bits=k_bits, n_sub=n_sub,
                    type_size=type_size, is_v2=is_v2) is None


def fused_mm(ext, a, layer, *, N: int, K: int, k_bits: int):
    """One fused decode-in-prologue GEMM over this layer's packed rows.

    ``a`` is the ``[M, K]`` BF16 activation the quality path already produces
    (group-16 QDQ), contiguous. Returns ``[M, N]`` BF16.
    """
    return ext.cb_fused_fp4v2_prefill_mm(
        a, layer._cb_qw_padded, layer._cb_flat, layer._cb_compose,
        N, K, k_bits)
