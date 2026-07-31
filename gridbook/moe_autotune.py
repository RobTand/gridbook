"""MEASURED per-layer prefill-path selection for the CB MoE method
(``PRISMAQUANT_CB_PREFILL=auto``).

WHY IT EXISTS. The grouped-fused prefill WINS on MoE models with small experts
(35B-A3B: 4,285 vs stock 3,932 tok/s) and LOSES on models with large ones
(Laguna-class: 1,503 vs stock 1,821 @8k): round 2 re-decodes each expert's B
operand once per M-tile and pads each expert's rows up to a tile multiple, so
both taxes scale with expert SIZE while the saved e4m3 expand round-trip does
not. There is therefore no static right answer — the crossover is a property of
the model's expert shapes and of the compiled TileM. Per the platform rule the
answer is MEASURED on the real inputs, never guessed from shapes or model names.

WHY ITS OWN MODULE. ``moe.py`` imports vLLM at module scope, so nothing in it is
importable in the build venv. Keeping the *policy* here — torch-only, with the
candidate thunks and the timer injected — makes the part with all the caching,
threshold and determinism logic testable on CPU with no vLLM and no GPU, exactly
as ``moe_routing.py`` does for the padded routing. ``moe.py`` is a thin adapter
that supplies the candidates and the layer object.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time

import torch

STOCK = "stock"

# ---------------------------------------------------------------------------
# Durable timing sink (re-vet R21)
# ---------------------------------------------------------------------------
# The tuner's per-candidate milliseconds were a `print()` to stderr and nothing
# else, so the only record of what the box measured was a serve log somebody
# had to still have. This appends one JSON row per tuned layer:
# (format, shape regime, box) -> {candidate: ms}.
#
# WHY THIS AND NOT THE lambda TERM. A time term in the allocator's objective
# would optimize against a serving-cost table that does not exist, on a lane
# that is not default, against a tax the kernel campaign moved 12x -> 1.75x in
# one week — a heuristic in an objective's clothes. The forcing function for
# lambda is TWO measured tables that disagree in RANKING, and this file is what
# makes that observable. lambda stays specified-not-implemented (R21 ruling,
# recorded in the re-vet outcome); nothing here feeds an allocator.
#
# Path: PRISMAQUANT_CB_AUTOTUNE_LOG, else `<artifact dir>/cb_autotune_timings.jsonl`
# when the caller knows it (`sink_dir`), else off. Append-only JSONL: a serve
# process crashing must not lose the rows already measured, and two processes
# appending is a merge rather than a lost file.
_SINK_LOCK = threading.Lock()


def autotune_sink_path(sink_dir=None):
    """Resolve the durable sink path, or ``None`` when there is nowhere to put
    it (no env override and the caller does not know the artifact dir)."""
    env = os.environ.get("PRISMAQUANT_CB_AUTOTUNE_LOG")
    if env:
        return env if env not in {"0", "off", "false"} else None
    if sink_dir:
        return os.path.join(str(sink_dir), "cb_autotune_timings.jsonl")
    return None


def box_id():
    """Coarse identity of the machine a timing was taken on — the third key of
    the (format, shape regime, box) table. A ms number is meaningless without
    it, and 'two boxes disagree in ranking' is unanswerable without it."""
    name = None
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:                          # noqa: BLE001 — never fail a serve
        name = None
    return name or f"cpu:{platform.machine()}"


def shape_regime(num_tokens, n_experts=None, intermediate=None):
    """The shape bucket a timing belongs to. Tokens are bucketed by power of
    two because the crossover this table exists to locate is a property of the
    M tile, not of an exact token count; expert count and intermediate size are
    recorded exactly because the round-2 taxes scale with expert SIZE."""
    m = max(int(num_tokens), 1)
    bucket = 1 << (m.bit_length() - 1)
    return {
        "num_tokens": int(num_tokens),
        "m_bucket": int(bucket),
        "n_experts": None if n_experts is None else int(n_experts),
        "intermediate": None if intermediate is None else int(intermediate),
    }


def record_autotune_timings(best, timings, *, layer_prefix, fmt, regime,
                            forced=False, sink_dir=None, path=None):
    """Append one row to the durable sink. Returns the path written, or None.

    Never raises: a serve must not die because a log directory is read-only.
    """
    target = path or autotune_sink_path(sink_dir)
    if not target or not timings and not forced:
        return None
    row = {
        "schema": "prismaquant.cb_autotune.v1",
        "ts": time.time(),
        "box": box_id(),
        "format": fmt,
        "layer": layer_prefix,
        "regime": regime,
        "chosen": best,
        "forced": bool(forced),
        "ms": {str(k): float(v) for k, v in (timings or {}).items()},
    }
    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(row, sort_keys=True) + "\n"
        with _SINK_LOCK:
            with open(target, "a") as fh:
                fh.write(line)
        return target
    except Exception as exc:                   # noqa: BLE001
        print(f"[prismaquant-cb] autotune sink unavailable ({exc}); "
              "timings stay on stderr only", file=sys.stderr, flush=True)
        return None


def cb_time_candidate(fn):
    """Run ``fn`` ONCE and return ``(output, milliseconds)``, or ``None`` if the
    candidate is unavailable (returned ``None``) or raised.

    ISOLATION. The candidate is bracketed by a full ``torch.cuda.synchronize()``
    on both sides of a CUDA-event pair, so the number contains this candidate's
    device work and nothing queued before or after it. Events (not the wall
    clock) are what make it a device duration rather than a launch duration.
    Without CUDA it falls back to a wall clock — meaningless as a real decision,
    but it keeps the machinery runnable off-GPU.

    A raising candidate is DISQUALIFIED rather than fatal: an extension build may
    not have compiled some (TileM, k_bits) pair (shared-memory limits), and that
    must degrade to the candidates that do work, like every other gate here.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return None if out is None else (out, start.elapsed_time(end))
        t0 = time.perf_counter()
        out = fn()
        return None if out is None else (out, (time.perf_counter() - t0) * 1e3)
    except Exception as exc:  # noqa: BLE001 — a bad candidate is not a bad serve
        print(f"[prismaquant-cb] prefill candidate disqualified "
              f"({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return None


def cb_autotune_prefill(candidates, keep: str = STOCK, timer=None):
    """Time every ``(name, fn)`` in ``candidates`` once; return
    ``(best_name, {name: ms}, kept_output)``.

    DETERMINISM. The tuning call runs several candidates, all of them valid under
    the REASSOCIATION-CLASS contract — but they are not bit-equal, so returning
    "whichever ran last" would make a given input's output depend on which path
    happened to win. We always return the ``keep`` candidate's output (in
    practice 'stock', the default path), so the tuning call is bit-identical to
    an ordinary call. Timing ties break on the candidate NAME, not on iteration
    order, for the same reason.

    ``(None, {}, None)`` means every candidate was disqualified; the caller runs
    its default path. ``timer`` is injectable for testing.
    """
    timer = timer or cb_time_candidate
    timings: dict[str, float] = {}
    kept = None
    for name, fn in candidates:
        res = timer(fn)
        if res is None:
            continue
        out, ms = res
        timings[name] = float(ms)
        if name == keep:
            kept = out
    if not timings:
        return None, {}, None
    best = min(timings, key=lambda n: (timings[n], n))
    return best, timings, kept


def resolve_candidate(name, candidates):
    """Map a cached or forced choice onto this call's candidate thunks. Exact
    name first; a bare FAMILY name ('grouped_fused') resolves to that family's
    first (smallest-TileM) candidate, which is what an operator pinning a path
    for A/B bisection means. ``None`` if nothing matches."""
    for cname, fn in candidates:
        if cname == name:
            return fn
    for cname, fn in candidates:
        if cname.split(":", 1)[0] == name:
            return fn
    return None


def cb_prefill_auto(layer, num_tokens, build_candidates, run_stock, *,
                    min_m=1024, forced=None, keep=STOCK, timer=None, log=None):
    """The 'auto' policy: tune once per layer, then dispatch straight to the
    winner.

    WHEN IT TUNES. Only at ``num_tokens >= min_m``
    (``PRISMAQUANT_CB_AUTOTUNE_MIN_M``, default 1024): a short prefill is not the
    steady-state shape the choice has to serve, and tuning on it would cache a
    decision taken at the wrong M. Below the threshold the call is plain stock
    and NO choice is cached, so the first qualifying call still gets to tune.

    COST. The tuning call runs every candidate once — ~2-3x one layer forward,
    ONCE per layer per process. Every later call is a name lookup and a direct
    dispatch, with no timing at all. The cache lives on the LAYER, not globally:
    rungs, expert counts and intermediate sizes differ per layer, so a global
    winner would be a heuristic wearing a measurement's clothes.

    DETERMINISM. The tuning call returns the ``keep`` candidate's output (see
    ``cb_autotune_prefill``), so a given input yields the same bytes whether or
    not this was the call that tuned.

    ``forced`` (``PRISMAQUANT_CB_PREFILL_AUTO_FORCE``) pins a winner with no
    timing, for A/B bisection of a suspect choice.
    """
    def _emit(best, timings, was_forced):
        """Call the log callback, passing the layer/shape context when it can
        take it (R21's durable sink needs it) and staying compatible with the
        3-positional-argument callbacks that predate the sink."""
        if not log:
            return
        try:
            import inspect

            params = inspect.signature(log).parameters
            rich = "layer" in params and "num_tokens" in params
        except (TypeError, ValueError):
            rich = False
        if rich:
            log(best, timings, was_forced, layer=layer, num_tokens=num_tokens)
        else:
            log(best, timings, was_forced)

    choice = getattr(layer, "_cb_prefill_choice", None)
    if choice is None and forced:
        choice = forced
        layer._cb_prefill_choice = choice
        _emit(choice, {}, True)

    if choice is None:
        if num_tokens < min_m:
            return run_stock()
        best, timings, kept = cb_autotune_prefill(
            build_candidates(), keep=keep, timer=timer)
        if best is None:                       # every candidate disqualified
            return run_stock()
        layer._cb_prefill_choice = best
        _emit(best, timings, False)
        return kept if kept is not None else run_stock()

    fn = resolve_candidate(choice, build_candidates())
    out = fn() if fn is not None else None
    # The winner became unavailable (or a forced name matches nothing): serve
    # from the default rather than fail the request.
    return run_stock() if out is None else out
