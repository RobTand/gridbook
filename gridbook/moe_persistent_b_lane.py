"""OPT-IN selector for the persistent-B grouped MoE decode-in-mainloop lane.

Two routes serve the FP4-CB MoE quality prefill above the M<=16 GEMV band:

* the DEFAULT one — ``cb_expand_fp4_v2`` materializes each expert chunk's
  weights as BF16 in HBM and an owned grouped CUTLASS GEMM reads them back
  (``moe.py::_apply_prefill_native_bf16``);
* this one — ``csrc/cb_moe_persistent_b.cu`` decodes the packed CB bytes
  inside the mainloop and streams the expert's routed rows through the decoded
  tile, so the ``[E, N, K]`` BF16 transient never exists.

Both consume the SAME activation payload (the exact group-16 RTN QDQ) and the
same packed weights, accumulate in FP32 and round once to BF16.  What differs
is the FP32 reduction order — reassociation-class, the requalification surface
W4's sm12x-native BF16 lane and the promoted FP8 mid-M fused kernel cleared —
so the lane stays opt-in until the served
[NATIVE-PARITY](../docs/NATIVE-PARITY.md) protocol runs on the WHOLE routed
operator.

The module mirrors ``bf16_grouped_lane`` deliberately: one process-stable
selector, resolution at model load and never at first forward, and a failure
that is loud rather than a silent substitution of the other route.
"""
from __future__ import annotations

import os

_FLAG = "PRISMAQUANT_CB_MOE_PERSISTENT_B"
_ALLOWED = frozenset(("", "0", "1"))
_STATE: list[str] = []


def requested() -> bool:
    """Whether the operator asked for the persistent-B MoE lane.

    Process-stable like every other Gridbook dispatch selector: a value that
    changed after dispatch was fixed would mix two reduction orders inside one
    run and make an A/B unreadable.  A typo raises rather than quietly
    selecting the baseline.
    """
    current = os.environ.get(_FLAG, "").strip()
    if current not in _ALLOWED:
        raise ValueError(
            f"invalid {_FLAG}={current!r}; expected '1' to enable the "
            f"persistent-B grouped MoE decode-in-mainloop lane, or '' / '0' "
            f"for the default expand + grouped-bridge route")
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


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension carrying the lane, or fail closed.

    Called at model load, never at first forward.  Failing closed is the
    point: with the flag on, quietly serving the expand + bridge route would
    produce a run whose numbers describe the wrong schedule.
    """
    from .cuda_ext import (NativeKernelUnavailableError,
                           moe_persistent_b_buildable,
                           require_moe_persistent_b_ext)

    ext = require_moe_persistent_b_ext(operation)
    capability = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 — reported below as "unavailable"
        capability = None
    from .cuda_ext import _MOE_PERSISTENT_B_SYMBOLS

    # The loader already enforces this strictly, so the re-check can never be
    # the difference — but it is spelled against the SAME tuple rather than a
    # local subset, so the two lists cannot drift into meaning nothing.
    missing = [name for name in _MOE_PERSISTENT_B_SYMBOLS
               if not hasattr(ext, name)]
    if missing:
        raise NativeKernelUnavailableError(
            f"{operation} requested the persistent-B grouped MoE lane "
            f"({_FLAG}=1), but the loaded extension does not carry it "
            f"(missing {missing}; device capability {capability}). The lane "
            f"is compiled only for compute capability 12.0/12.1"
            + ("" if capability is None or
               moe_persistent_b_buildable(capability)
               else f", and this device reports "
                    f"{capability[0]}.{capability[1]}")
            + f". Unset {_FLAG} to use the default expand + grouped-bridge "
            f"route; Gridbook does not substitute a different kernel behind "
            f"an explicit lane selection.")
    return ext


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
    # and hidden are then multiples of 256.
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


def resolve_cfg(ext) -> int:
    """Validate ``PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG`` against this build.

    Called at model LOAD, like every other decision this lane makes.  A tile
    index the module was not compiled for, or a typo, must fail the load — the
    kernel would otherwise abort the first request that carried routed rows,
    which is both later and harder to read.  ``0`` means "let the kernel pick
    from the shapes", which is the production setting; anything else is a
    measurement override.
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
    return cfg
