"""Which grouped fp4-v2 decode GEMV a (layer, stack) runs, and the process-wide
kernel selector behind it (``PRISMAQUANT_CB_GEMV``).

WHY ITS OWN MODULE. ``moe.py`` imports vLLM at module scope, so nothing in it is
importable in the build venv. The part with the contract — the env whitelist,
the once-per-process resolution, the availability probe and the per-(layer,
stack) fallback decision — lives here, vLLM-free and importable with nothing
but the standard library, exactly as ``moe_routing.py`` / ``moe_autotune.py`` /
``moe_l2.py`` do for the prefill policies. ``moe.py`` is a thin adapter that
calls :func:`cb_gemv_choice` once per stack at load and stores the answer on
the layer.

Nothing here imports torch at module scope either: the two functions that need
the extension (:func:`cb_gemv_v2_available`, :func:`cb_gemv_choice`) import it
inside the call, so a process that never serves a CB MoE layer never pays the
JIT build.
"""
from __future__ import annotations

import os
import sys


# CB-GEMV-v2 DISPATCH — which grouped fp4-v2 decode GEMV a (layer, stack) runs.
# Two kernels now exist for the SAME job:
#   * ``cb_moe_gemv_fp4_v2`` — the INHERITED kernel (csrc/cb_gemv.cu). Global
#     dictionary, no staged smem; its rowpack schedule caps its shared-memory
#     request at 48 KB (cb_gemv.cu:1445).
#   * ``cb_moe_gemv_v2``     — the smem-resident-dictionary kernel
#     (csrc/cb_gemv_v2.cu). Opts in to the sm_121a 99 KB dynamic-smem budget
#     and stages the product sub-tables when they fit, which is what buys the
#     decode-efficiency delta on the low rungs.
# Same byte format, same decode semantics, same FMA chain as the inherited
# ROWPACK schedule (csrc/cb_gemv_v2.cu:40-49); only WHERE the dictionary is
# read from changes. It is NOT bit-exact against the inherited DEFAULT
# schedule — that is a reassociation-class difference.
#
# Selector (resolve once per process, whitelist, fail loud, one log line):
#   auto       — the shipping default. NOT "prefer v2": see cb_gemv_choice —
#                auto only reaches v2 when the extension built AND the
#                occupancy predicate says it wins.
#   v2         — as auto, but self-describing in the log so an A/B arm is
#                identifiable from the serve output alone.
#   inherited  — the KILL SWITCH and the A/B control. Never probes, never
#                builds the v2 extension, and reproduces today's dispatch
#                exactly. This is the arm that proves a v2 regression is a v2
#                regression.
# The switch exists so an A/B is ONE serve session per arm with an otherwise
# byte-identical process shape — swapping plugin trees between arms would
# confound the measurement with a rebuild.
_CB_GEMV_VALUES = ("auto", "v2", "inherited")
_CB_GEMV: str | None = None


def cb_gemv_mode() -> str:
    """The process-wide CB GEMV kernel selection ("auto" | "v2" |
    "inherited"), resolved once from ``PRISMAQUANT_CB_GEMV`` (unset -> "auto").

    Raises ``ValueError`` on an unknown spelling and ``RuntimeError`` on a
    mid-process change. Both matter: under mixed dispatch one model runs BOTH
    kernels, so (a) a typo must not silently degrade to the inherited kernel
    and then be read as "v2 buys nothing", and (b) a selector that moved
    mid-serve would split the dispatch across kernels and make every number
    taken from that process unattributable.
    """
    global _CB_GEMV
    raw = os.environ.get("PRISMAQUANT_CB_GEMV")
    val = (raw if raw is not None else "auto").strip().lower()
    if _CB_GEMV is None:
        if val not in _CB_GEMV_VALUES:
            raise ValueError(
                f"[prismaquant-cb] $PRISMAQUANT_CB_GEMV={raw!r} is not a known "
                f"CB GEMV selection; expected one of {list(_CB_GEMV_VALUES)}. "
                f"Refusing to serve: a typo must not silently degrade to the "
                f"inherited kernel and be read as a null result.")
        _CB_GEMV = val
        print(f"[prismaquant-cb] cb_gemv={val}"
              + (" (env unset -> auto default)" if raw is None else ""),
              flush=True)
    elif val != _CB_GEMV:
        raise RuntimeError(
            f"[prismaquant-cb] PRISMAQUANT_CB_GEMV changed mid-process: "
            f"resolved {_CB_GEMV!r} at first use, env now says {val!r}. The "
            f"selection is resolved ONCE per process; a mid-serve change would "
            f"split the dispatch across kernels. Restart with the env pinned.")
    return _CB_GEMV


def decode_contract_v2_arg() -> int:
    """The int64 ``v2`` argument a ``cb_moe_gemv_v2`` call MUST pass.

    The inherited kernel resolves the decode contract itself, from the
    environment, PER LAUNCH — ``pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT",
    "v2")`` at cb_gemv.cu:471/760/1357/1437. The v2 kernel takes it as an
    argument instead, so under a MIXED dispatch (one model, both kernels) the
    two must agree or the serve runs v1 per-weight rounding on some layers and
    v2 on others — a systematic, silent, ulp-class divergence.

    Agreement is therefore by CONSTRUCTION here: ``pq_env_is`` is a bare
    ``strcmp`` against ``"v2"``, so this is a bare ``== "v2"``. Deliberately no
    ``.strip()`` and no ``.lower()`` — "V2" or " v2" is v1 to the inherited
    kernel, and normalising here would make it v2 to ours, which is exactly the
    split this function exists to prevent.
    """
    return 1 if os.environ.get("PRISMAQUANT_CB_DECODE_CONTRACT") == "v2" else 0


# Availability probe. An OLD extension (or an old plugin tree) under a NEW
# dispatch is a real deployment state — the v2 kernel is a SEPARATE JIT module
# (``prismaquant_cb_v2_ext``) from the inherited one, and either half of the
# plumbing can be missing independently:
#   * ``gridbook.cuda_ext`` predating ``get_ext_v2()``            (old tree)
#   * ``gridbook.ops`` predating the ``cb_moe_gemv_v2`` custom op (old tree)
#   * ``get_ext_v2()`` -> None: nvcc absent / JIT build failed    (old/no ext)
#   * a stale ``prismaquant_cb_v2_ext.so`` in the build cache that predates a
#     symbol                                                     (old .so)
# Every one of those degrades to the inherited kernel with a loud one-time
# warning rather than raising — the inherited path is exactly today's serve, so
# degrading is CORRECT, only slower.
_CB_GEMV_V2_OK: bool | None = None


def cb_gemv_v2_available() -> bool:
    """True iff the whole v2 path — custom op, JIT module and both host
    symbols — is present and loadable. Probed once, never raises."""
    global _CB_GEMV_V2_OK
    if _CB_GEMV_V2_OK is not None:
        return _CB_GEMV_V2_OK
    why = None
    try:
        from . import cuda_ext
        from . import ops as pq_ops
        get_v2 = getattr(cuda_ext, "get_ext_v2", None)
        if get_v2 is None:
            why = ("gridbook.cuda_ext has no get_ext_v2() — plugin tree "
                   "predates cb_gemv_v2")
        elif getattr(pq_ops, "cb_moe_gemv_v2", None) is None:
            why = ("gridbook.ops has no cb_moe_gemv_v2 custom op — plugin tree "
                   "predates cb_gemv_v2")
        else:
            ext = get_v2()
            if ext is None:
                why = "get_ext_v2() returned None (JIT build unavailable)"
            else:
                for sym in ("cb_gemv_v2", "cb_gemv_v2_prefers_inherited"):
                    if not hasattr(ext, sym):
                        why = (f"the loaded prismaquant_cb_v2_ext exports no "
                               f"{sym}() — stale .so in the build cache")
                        break
    except Exception as exc:                  # noqa: BLE001 — probe never raises
        why = f"{type(exc).__name__}: {exc}"
    if why is not None:
        print(f"[prismaquant-cb] WARNING: CB-GEMV-v2 unavailable ({why}); "
              f"every CB layer decodes on the INHERITED kernel. Serving is "
              f"correct and unchanged, but this serve carries no v2 delta — "
              f"do not report it as a v2 arm.", file=sys.stderr, flush=True)
    _CB_GEMV_V2_OK = why is None
    return _CB_GEMV_V2_OK


def cb_gemv_choice(k_bits: int, n_sub: int, type_size: int,
                   in_features: int) -> tuple[bool, str]:
    """``(use_v2, reason)`` for one (layer, stack).

    THE FALLBACK PREDICATE. Resolved on the HOST, ONCE, at load — never at
    apply — so the call-site ``if`` is a trace-time constant and no host work,
    and in particular no ``.item()``/``.tolist()``, can enter a captured
    region. That is what lets FULL-decode cudagraphs coexist with a mixed
    dispatch.

    The occupancy verdict is delegated to the COMPILED
    ``cb_gemv_v2_prefers_inherited`` (csrc/cb_gemv_v2.cu) rather than
    re-implemented here, so there is exactly ONE arithmetic and it is the one
    the binary that actually launches was compiled from. A Python
    re-implementation would be free to drift from it.

    WHY THE FALLBACK EXISTS — THE PHYSICAL REASON. v2's win comes from staging
    the product sub-tables in shared memory. At k=24 those tables are
    2 x 2^12 x 4 x 2 B = 65,536 B, and sm_121a's opt-in dynamic-smem budget is
    99 KiB (101,376 B), so two resident blocks would need <= 50,688 B each: the
    dictionary ALONE busts a 2-block bill at every slot size and warp count.
    The block therefore runs 1/SM = 8 of 48 warps = 16.7 % occupancy BY
    CONSTRUCTION, and with no second block resident to overlap its dictionary
    burst against, the DRAM duty cycle caps near 50 % (measured 48.8 %). Decode
    is bandwidth-bound, so that is a pure throughput loss. Below K=2048 the
    shorter weight stream still leaves v2 ahead; at K>=2048 it does not.
    Nothing ERRORS at the wall — it is a speed decision — which is why a test
    that trips it must assert the DISPATCH DECISION and not an exception (see
    the ``reason`` string, logged per (layer, stack)).

    ``in_features >= 2048`` inside the predicate is MEASURED AT k=24 ONLY. Its
    extrapolation to k=22/23 is unmeasured; the predicate's verdict there
    routes those to the inherited kernel, which is the conservative direction
    (the proven kernel), so the untested extrapolation cannot produce a wrong
    answer, only a missed win.
    """
    mode = cb_gemv_mode()
    if mode == "inherited":
        return False, "mode=inherited (kill switch)"
    # v2 is PRODUCT-MODE ONLY. `_cuda_moe_ok` admits n_sub in (1, 2) for
    # fp4-v2; a signed n_sub=1 rung would trip v2's own `cb_elems` TORCH_CHECK
    # loudly rather than silently, but it must never get that far.
    if n_sub != 2:
        return False, f"n_sub={n_sub} (v2 is product-mode only)"
    if type_size != 4 * k_bits + 9:
        return False, f"type_size={type_size} != 4k+9 (not fp4-v2)"
    if not cb_gemv_v2_available():
        return False, "v2 extension unavailable"
    from .cuda_ext import get_ext_v2
    if get_ext_v2().cb_gemv_v2_prefers_inherited(
            k_bits, type_size, in_features):
        return False, "predicate: occupancy wall (staged dict starves blocks/SM)"
    return True, f"mode={mode}"
