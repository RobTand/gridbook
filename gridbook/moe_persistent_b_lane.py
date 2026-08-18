"""Selector for the persistent-B grouped MoE decode-in-mainloop lane.

Two routes serve the CB MoE quality prefill above the M<=16 GEMV band, for
BOTH payload families (FP4-CB two-tier v2, and — since ROADMAP K1.2 — stock
FP8-CB):

* the DEFAULT one — the native expander (``cb_expand_fp4_v2`` /
  ``cb_expand_fp8`` + the fp32 row-scale multiply) materializes each expert
  chunk's weights as BF16 in HBM and an owned grouped CUTLASS GEMM reads them
  back (``moe.py::_apply_prefill_native_bf16``);
* this one — ``csrc/cb_moe_persistent_b.cu`` decodes the packed CB bytes
  inside the mainloop and streams the expert's routed rows through the decoded
  tile, so the ``[E, N, K]`` BF16 transient never exists.

Both consume the SAME activation payload (the exact native QDQ of the family —
group-16 RTN for FP4, e4m3 for FP8) and the same packed weights, accumulate in
FP32 and round once to BF16.  What differs is the FP32 reduction order —
reassociation-class.  The lane was opt-in until that surface cleared served
validation — the FP4 arm's same-session served A/B on the DSv4 92 GB body
(kl_mean −0.051%, PPL −0.30% — arithmetic noise; the dated entry in
CHANGELOG.md names the rest of the record) on top of each arm's bitwise
decode-identity CUDA suite.  **Default is now "auto"** (0.8.9): each routed CB
layer engages its family's arm where the load-time predicate and the extension
attest, and keeps the expand+bridge route where they do not, with the route
telemetry naming which.  The bridge is exactly the previous default —
degrading is correct, only slower.

ONE flag covers both families, three spellings:

* unset / ``auto`` — persistent-B where supported, bridge where not (DEFAULT);
* ``1`` / ``require`` — every routed CB layer must take the lane; a layer it
  cannot serve (per-role FP8-CB splits, for example) fails the LOAD, never a
  silent baseline run.  This is the A/B-integrity semantics 0.8.8 spelled as
  ``=1``: with the lane explicitly requested, quietly serving the bridge
  would produce a run whose numbers describe the wrong schedule;
* ``0`` / ``off`` — the expand+bridge route everywhere.

The module mirrors ``bf16_grouped_lane`` deliberately: one process-stable
selector, resolution at model load and never at first forward, and a failure
that is loud rather than a silent substitution of the other route.
"""
from __future__ import annotations

import os

from . import lane_select

_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B"
_D2R_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R"

_MODE_SPELLINGS = {
    "": "auto",
    "auto": "auto",
    "1": "require",
    "require": "require",
    "0": "off",
    "off": "off",
}


def mode() -> str:
    """The lane's process-wide mode: ``"auto"`` | ``"require"`` | ``"off"``.

    Process-stable like every other Gridbook dispatch selector: a value that
    changed after dispatch was fixed would mix two reduction orders inside one
    run and make an A/B unreadable.  A typo raises rather than quietly
    selecting a mode.  Unset means ``auto`` (the default since 0.8.9); ``1``
    keeps its 0.8.8 meaning exactly — an explicit lane REQUIREMENT that fails
    the load rather than substituting the bridge.
    """
    return lane_select.latched_choice(
        _FLAG, spellings=_MODE_SPELLINGS,
        meaning="the persistent-B grouped MoE decode-in-mainloop lane")


def requested() -> bool:
    """Whether the lane may engage at all (any mode but ``off``)."""
    return mode() != "off"


def d2r_requested() -> bool:
    """Whether the nested direct-to-register experiment was requested.

    This is deliberately a second switch UNDER persistent-B: setting it
    without ``PRISMAQUANT_CB_MOE_PERSISTENT_B=1`` is rejected by the model-load
    wiring instead of becoming an ignored, falsely-labelled experiment.
    """
    return lane_select.latched_bool(
        _D2R_FLAG,
        meaning="persistent-B's experimental BF16 direct-to-register B path")


def _reset_for_tests() -> None:
    """Clear the process-stable latch (tests only)."""
    lane_select.reset_for_tests(_FLAG)
    lane_select.reset_for_tests(_D2R_FLAG)


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension carrying the lane, or fail closed.

    Called at model load, never at first forward.  Failing closed is the
    point: with the flag on, quietly serving the expand + bridge route would
    produce a run whose numbers describe the wrong schedule.

    ``cb_moe_persistent_b_prepare`` is the per-device attestation: loading the
    module proves the symbols exist, and this proves THIS device can serve
    them, opting every compiled configuration in to its dynamic shared-memory
    budget.  Doing it at load rather than lazily on first launch is what keeps
    ``cudaFuncSetAttribute`` — which is not stream-ordered work — out of a
    first forward and out of a CUDA-graph capture.
    """
    from .cuda_ext import (_MOE_PERSISTENT_B_SYMBOLS,
                           get_moe_persistent_b_ext,
                           moe_persistent_b_buildable,
                           require_moe_persistent_b_ext)

    # Keep the module-level fail-closed diagnostic for "no module at all": it
    # names the nvcc hint, which a lane-level message would not.
    require_moe_persistent_b_ext(operation)
    return lane_select.require_lane(
        operation, flag=_FLAG,
        lane="the persistent-B grouped MoE lane",
        source="persistent-B extension (cb_moe_persistent_b.cu)",
        alternative="the default expand + grouped-bridge route",
        get_ext=get_moe_persistent_b_ext,
        symbols=_MOE_PERSISTENT_B_SYMBOLS,
        buildable=moe_persistent_b_buildable,
        device=device,
        prepare="cb_moe_persistent_b_prepare")


def probe_lane(operation: str = "this operation", *, device=None):
    """``require_lane``'s degrade sibling for ``mode() == "auto"``.

    Returns ``(ext, None)`` when the lane attests on this device, or
    ``(None, why)`` when it cannot — the caller keeps the expand+bridge route
    and SAYS so.  Auto must never fail a load the bridge can serve: the
    bridge is exactly the pre-0.8.9 default, only slower, so degrading is
    correct (the same posture as the GEMV v2 availability probe).
    """
    try:
        return require_lane(operation, device=device), None
    except Exception as exc:  # noqa: BLE001 — any attestation failure means
        # the bridge serves; the reason is surfaced by the caller's telemetry.
        return None, f"{type(exc).__name__}: {exc}"


def require_d2r_lane(operation: str = "this operation", *, device=None):
    """Attest the nested D2R symbols in the existing persistent-B module.

    There is intentionally no second loader, build directory or extension
    cache.  The established module's source digest keys the candidate code;
    this narrower symbol/prepare gate runs only when the nested flag is on, so
    an unset experiment retains the production ABI tuple unchanged.
    """
    from .cuda_ext import (_MOE_PERSISTENT_B_D2R_SYMBOLS,
                           get_moe_persistent_b_ext,
                           moe_persistent_b_buildable)

    return lane_select.require_lane(
        operation, flag=_D2R_FLAG,
        lane="persistent-B's experimental BF16 direct-to-register B path",
        source="the existing persistent-B extension (cb_moe_persistent_b.cu)",
        alternative="the established persistent-B shared-B schedule",
        get_ext=get_moe_persistent_b_ext,
        symbols=_MOE_PERSISTENT_B_D2R_SYMBOLS,
        buildable=moe_persistent_b_buildable,
        device=device,
        prepare="cb_moe_persistent_b_d2r_prepare")


def supports(*, is_fp4: bool, is_v2: bool, n_sub: int, k_bits: int,
             type_size: int, hidden: int, inter: int) -> str | None:
    """``None`` when the lane can serve this layer, else the reason it cannot.

    Mirrors ``cb_moe_persistent_b_prefill``'s own TORCH_CHECKs so a layer the
    kernel would reject is diagnosed at model load with a readable sentence
    instead of aborting a request.  Every clause is a shape/format fact known
    at load; nothing here reads the routing or the device.
    """
    if not is_fp4:
        return "format is not FP4-CB"
    if not is_v2:
        return "format is not two-tier layout v2"
    if n_sub != 2:
        return f"n_sub={n_sub} is not the product-mode 2 the decode assumes"
    if not 1 <= k_bits <= 24:
        return f"k={k_bits} is outside the FP4-CB v2 range [1, 24]"
    if type_size != 4 * k_bits + 9:
        return "serialized row type_size is not FP4-CB layout v2 (4*k+9)"
    # Superblock alignment (256) is the binding constraint and it implies the
    # kernel's own `N % 8 == 0` check for both projection widths, since 2*inter
    # and hidden are then multiples of 256.  There is deliberately no minimum
    # on N: a projection narrower than the narrowest compiled tile is served by
    # clamping to that tile, which the kernel's column masking makes correct.
    if hidden % 256 or inter % 256:
        return ("hidden and intermediate sizes must be superblock aligned "
                f"(256), got hidden={hidden}, intermediate={inter}")
    return None


def supports_fp8(*, is_fp4: bool, n_sub: int, k_bits: int, type_size: int,
                 hidden: int, inter: int, role_split: bool) -> str | None:
    """``None`` when the FP8-CB arm can serve this layer, else the reason.

    Mirrors ``cb_moe_persistent_b_prefill_fp8``'s TORCH_CHECKs, plus the one
    layout fact the kernel cannot see: a per-role w13 split has no single
    stacked qweight/book pair, so the lane refuses it BY NAME at model load —
    with the flag set, a role-split FP8-CB layer fails the load rather than
    silently keeping the bridge (Gridbook does not substitute a route behind
    an explicit lane selection).
    """
    if is_fp4:
        return "format is FP4-CB (use the FP4 arm's supports())"
    if role_split:
        return ("per-role FP8-CB codebooks: the persistent-B decode consumes "
                "one stacked stock book per projection, and this layer's w13 "
                "splits gate/up into two books")
    if n_sub != 4:
        return f"n_sub={n_sub} is not the product-mode 4 the decode assumes"
    if not 1 <= k_bits <= 48:
        return f"k={k_bits} is outside the FP8-CB range [1, 48]"
    if type_size != 4 * k_bits:
        return "serialized row type_size is not FP8-CB layout (4*k)"
    if hidden % 256 or inter % 256:
        return ("hidden and intermediate sizes must be superblock aligned "
                f"(256), got hidden={hidden}, intermediate={inter}")
    return None


def config(ext) -> list[list[int]]:
    """The tile configs the extension was actually COMPILED for.

    Always enumerated from the module — never a hardcoded list — because a
    (TM, TN) pair can be dropped for shared-memory reasons and python must not
    be able to request one that does not exist.
    """
    return [list(map(int, row)) for row in ext.cb_moe_persistent_b_configs()]


def resolve_cfg(ext, *, fp8_type_size: int | None = None) -> int:
    """Validate ``PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG`` against this build.

    Called at model LOAD, like every other decision this lane makes.  A tile
    index the module was not compiled for, or a typo, must fail the load — the
    kernel would otherwise abort the first request that carried routed rows,
    which is both later and harder to read.  ``0`` means "let the kernel pick
    from the shapes", which is the production setting; anything else is a
    measurement override.

    ``fp8_type_size`` names the FP8-CB rung the layer serves, and adds the
    2-CTAs/SM occupancy floor to the load-time validation: FP8's wider packed
    superblocks push the wide tiles past the floor above k=33/k=31, and the
    kernel TORCH_CHECKs exactly that at launch — which for an explicit
    override would be the first routed request, the failure mode load-time
    resolution exists to prevent.  The predicate is the extension's own
    (``cb_moe_persistent_b_fp8_cfg_eligible``), never re-derived here.  FP4
    passes ``None``: every compiled tile holds the floor at every FP4 rung.
    """
    raw = os.environ.get("PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG", "").strip()
    if not raw:
        return 0
    try:
        cfg = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG={raw!r}; expected "
            f"0 for the kernel's own shape-driven choice, or a 1-based index "
            f"into cb_moe_persistent_b_configs()") from exc
    compiled = config(ext)
    if not 0 <= cfg <= len(compiled):
        raise ValueError(
            f"PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG={cfg} is not a compiled "
            f"tile config; this build offers 0 (auto) or 1..{len(compiled)} "
            f"({compiled})")
    if cfg and fp8_type_size is not None and not bool(
            ext.cb_moe_persistent_b_fp8_cfg_eligible(cfg, fp8_type_size)):
        raise ValueError(
            f"PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG={cfg} does not hold two "
            f"CTAs per SM at this layer's FP8-CB type_size={fp8_type_size}; "
            f"pick an eligible tile or 0 (auto), which filters on the same "
            f"predicate")
    return cfg
