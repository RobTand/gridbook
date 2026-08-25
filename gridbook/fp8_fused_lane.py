"""One answer about the FP8-CB fused mid-M lane's RUNG surface (K1.2).

``linear.py`` (dense) and ``moe.py`` (routed) both decide whether a layer's
``k_bits`` can take the decode-in-prologue fused path.  Its historical reader
surface is ``(28, 32, 36, 40, 44, 48)``: K28/K32/K36 remain load-compatible
optimized paths even though the canonical producer menu is now K40/K44/K48.
Before K1.2 the literal was transcribed at both call sites and could drift from
the compiled module.  This module owns the decision, the way
``bf16_grouped_lane`` and ``fp4v2_fused_midm_lane`` own theirs.

TWO-STEP, DELIBERATELY. The law (``codec.fp8_fused_rung_supported``, over
``codec.FP8_FUSED_KBITS``) is checked first because it is free: it needs no
extension, and asking the MODULE first would force a JIT build at first forward
merely to learn that a rung can never take this path. Once a module is in hand,
``compiled_kbits`` is the AUTHORITY — it is what this build actually
instantiated. The law can only ever be a filter over that answer, never a
substitute for it, so dispatch can neither select an uncompiled rung nor
silently miss a compiled one. ``rung_eligible`` is the only public answer; the
law is consulted THROUGH ``codec``, so this module never re-exports it.
"""
from __future__ import annotations

from . import codec


def compiled_kbits(ext) -> tuple[int, ...]:
    """The rungs ``ext`` actually compiled, or ``()`` if it will not say.

    Fail-soft on purpose, and for a live reason rather than a compatibility
    one: ``cb_fused_kbits`` is in the fused module's STRICT symbol contract, so
    a module that loaded has it. But dispatch treats an unexpected answer
    mid-forward as "not offered" and falls through to the exact native route,
    exactly as ``moe._gf2_tile_sizes`` does for the tile query — a surprise
    here must cost a slower kernel, never the request.
    """
    if ext is None:
        return ()
    try:
        from .cuda_ext import fused_fp8_kbits

        return fused_fp8_kbits(ext)
    except Exception:  # noqa: BLE001 — treat as "not offered"
        return ()


def rung_eligible(ext, k: int) -> bool:
    """Whether this build can run rung ``k`` through the fused lane."""
    return codec.fp8_fused_rung_supported(k) and int(k) in compiled_kbits(ext)
