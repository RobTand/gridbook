"""Selection and attestation of the sm12x-native BF16 grouped lane.

Gridbook's quality-preserving prefill bridge has two compiled lanes (see
``csrc/cb_bf16_grouped_gemm.cu``):

* the DEFAULT device-scheduled CUTLASS 2.x ``DefaultGemmGrouped`` on an
  ``arch::Sm80`` schedule, which takes exact per-expert segments; and
* the sm12x-native CUTLASS 3.x collective (TMA warp-specialized mainloop,
  row-padded tile-indexed grouping), compiled only on cc 12.x.

The second is **OPT-IN** behind ``PRISMAQUANT_CB_BF16_SM120=1``, per
[NATIVE-PARITY](../docs/NATIVE-PARITY.md): it is bit-gated against the torch
reference but has not been through the served grouped-MoE protocol, and it
changes the FP32 REDUCTION ORDER of every default NVFP4-CB prefill. Nothing
here runs, probes or builds anything when the flag is unset — with the flag
off the dispatch is byte-for-byte what it was.

This module owns three things so ``linear.py`` and ``moe.py`` share one
answer: the process-stable flag read, the load-time attestation (the repo
convention is that nothing resolves at first forward), and the dense
``E=1`` padding helper.
"""
from __future__ import annotations

import os

_FLAG = "PRISMAQUANT_CB_BF16_SM120"
_ALLOWED = frozenset(("", "0", "1"))
_STATE: list[str] = []


def requested() -> bool:
    """Whether the operator asked for the sm12x-native lane.

    Process-stable like every other Gridbook dispatch selector: a value that
    changes after dispatch was fixed would silently mix two reduction orders
    inside one run and make an A/B unreadable. A typo raises rather than
    quietly selecting the baseline.
    """
    current = os.environ.get(_FLAG, "").strip()
    if current not in _ALLOWED:
        raise ValueError(
            f"invalid {_FLAG}={current!r}; expected '1' to enable the "
            f"sm12x-native BF16 grouped lane, or '' / '0' for the default "
            f"SM80-schedule bridge")
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
    """Return the extension whose sm12x lane is usable, or fail closed.

    Called at model load, never at first forward. Failing closed is the point:
    with the flag on, quietly serving the SM80 lane would produce a run whose
    numbers describe the wrong kernel.
    """
    from .cuda_ext import (NativeKernelUnavailableError,
                           bf16_grouped_sm120_buildable,
                           require_bf16_grouped_ext)

    ext = require_bf16_grouped_ext(operation)
    capability = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 — reported below as "unavailable"
        capability = None
    missing = [name for name in ("cb_bf16_grouped_mm_sm120",
                                 "cb_bf16_grouped_mm_sm120_out",
                                 "cb_bf16_grouped_sm120_tile_m")
               if not hasattr(ext, name)]
    if missing:
        raise NativeKernelUnavailableError(
            f"{operation} requested the sm12x-native BF16 grouped lane "
            f"({_FLAG}=1), but the loaded grouped-BF16 extension does not "
            f"carry it (missing {missing}; device capability {capability}). "
            f"The lane is compiled only for compute capability 12.0/12.1"
            + ("" if capability is None or
               bf16_grouped_sm120_buildable(capability)
               else f", and this device reports "
                    f"{capability[0]}.{capability[1]}")
            + f". Unset {_FLAG} to use the default SM80-schedule bridge; "
            f"Gridbook does not substitute a different kernel behind an "
            f"explicit lane selection.")
    return ext


def tile_m(ext) -> int:
    """The lane's row-padding granularity, as the kernel reports it."""
    return int(ext.cb_bf16_grouped_sm120_tile_m())


def dense_mm(ext, a, weight):
    """Dense ``E=1`` GEMM through the padded lane.

    ``a`` is ``[M, K]`` BF16 and ``weight`` is ``[N, K]`` BF16. M is padded up
    to one tile with zero rows (an activation-side transient of the same class
    the promoted fused routed path already allocates — no second weight copy,
    no new packer), every tile carries expert id 0, and the padded rows are
    sliced off. Returns ``[M, N]``.
    """
    import torch

    from .ops import cb_bf16_grouped_mm_sm120

    granularity = tile_m(ext)
    m = int(a.shape[0])
    blocks = (m + granularity - 1) // granularity
    padded = blocks * granularity
    if padded != m:
        a = torch.cat(
            [a, a.new_zeros((padded - m, a.shape[1]))]) if m else a.new_zeros(
                (padded, a.shape[1]))
    expert_ids = torch.zeros(blocks, dtype=torch.int32, device=a.device)
    y = cb_bf16_grouped_mm_sm120(a.contiguous(), weight.unsqueeze(0),
                                 expert_ids, granularity)
    return y[:m]
