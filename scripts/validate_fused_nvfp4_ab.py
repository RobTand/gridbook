#!/usr/bin/env python3
"""Same-process quality/performance A/B for fused NVFP4-CB prefill.

This is a promotion-gate *measurement* tool, not a microbenchmark.  It loads
one vLLM engine, replays identical deterministic WikiText windows through the
shipping baseline and opt-in fused path, and changes the dispatch environment
between synchronous requests.  Loading once removes model/session drift; the
fused extension is loaded before the model so extension residency is also
identical in both arms.

The two arms intentionally do not have the same activation contract:

* baseline: Gridbook's fp32-emulated group-scale activation QDQ;
* fused static modes: native NVFP4 attested per-tensor global scale + UE4M3
  group factors;
* fused rowwise modes: native NVFP4 independent runtime scale per row + UE4M3
  group factors.

Consequently the useful quality outputs are target-token NLL/PPL and paired
top-K coarse KL, not tensor equality.  Optional one-token request wall timing
includes scheduling, prefill, logits, and output processing; it is neither a
streaming TTFT measurement nor a substitute for the served workload matrix.
For small models, ``--teacher-model`` additionally scores both arms against a
local or revision-pinned, identity-matched, dtype-attested BF16 Transformers
reference after all timing measurements complete.  Teacher comparison is
cross-runtime.  Its
default KL remains top-K/coarse; ``--teacher-full-vocab-kl`` explicitly requests
and cardinality-checks every vLLM vocabulary logprob for exact KL, at substantial
memory cost on 128k-vocabulary models.  Coarse KL can reject a candidate but
cannot produce a promotion-eligible status; that requires the explicit exact
full-vocabulary teacher contract and both teacher-quality limits.

``--mode dense`` instruments Gridbook's dense linear fused path.
``--mode moe128`` and ``--mode moe256`` instrument the real grouped-MoE
routing, two-stage GEMMs, activation, and combine path.  The MoE baseline keeps
``PRISMAQUANT_CB_PREFILL`` unset and proves that the normal FP4 loop ran; the
candidate also keeps it unset and selects only ``PRISMAQUANT_CB_FUSED_FP4_MOE``.

Example (inside the pinned vLLM/CUDA container)::

    python3 scripts/validate_fused_nvfp4_ab.py \
      --model /models/nvfp4cb_k16 \
      --dataset-cache-dir /hfcache/datasets \
      --output /evidence/dense-k16-fused-ab.json \
      --n-samples 4 --seqlen 128 --top-k 1024 --timing-repeats 3 \
      --measurement-only

The script must own the Python process.  Importing vLLM before it runs would
freeze vLLM's multiprocessing environment before the same-process control is
set, so that condition is refused rather than silently producing empty
dispatch telemetry.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_COMMON_PATH = (
    Path(__file__).resolve().parents[1]
    / "gridbook"
    / "_fused_nvfp4_validation.py"
)
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "_gridbook_fused_nvfp4_validation_common", _COMMON_PATH
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError(f"could not load validation common module from {_COMMON_PATH}")
validation_common = sys.modules.get(_COMMON_SPEC.name)
if validation_common is None:
    validation_common = importlib.util.module_from_spec(_COMMON_SPEC)
    sys.modules[_COMMON_SPEC.name] = validation_common
    _COMMON_SPEC.loader.exec_module(validation_common)


SCHEMA = "gridbook.fused-nvfp4-ab.v5"
ARMS = ("baseline", "fused")
KL_COARSE_TOPK = "coarse_topk_tail"
KL_FULL_VOCAB = "exact_full_vocab"
FUSED_ENV = "PRISMAQUANT_CB_FUSED_FP4"
FUSED_MOE_ENV = "PRISMAQUANT_CB_FUSED_FP4_MOE"
PREFILL_ENV = "PRISMAQUANT_CB_PREFILL"
VLLM_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
DENSE_FUSED_MODES = ("1", "midm", "rowwise", "rowwise_midm")
_TEACHER_QUALITY_PROMOTION_GATES = (
    "max_teacher_fused_mean_kl",
    "max_teacher_fused_kl_regression",
)

_CONFIG_IDENTITY_IGNORED_KEYS = frozenset({
    # These identify the artifact/location, not the underlying architecture.
    "_name_or_path",
    # The candidate is intentionally quantized and the teacher must explicitly
    # be unquantized.  That asymmetry is attested separately.
    "quantization_config",
})
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and less than 1")
    return parsed


def _closed_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def paired_arm_order(prompt_index: int) -> tuple[str, str]:
    """Counterbalance which arm runs first while still pairing each prompt."""

    if prompt_index < 0:
        raise ValueError("prompt_index must be nonnegative")
    return ARMS if prompt_index % 2 == 0 else tuple(reversed(ARMS))


def activate_arm(
    arm: str,
    *,
    fused_mode: str,
    environ: MutableMapping[str, str],
    mode_cache: list,
) -> str:
    """Select one arm and invalidate Gridbook's process-lifetime env cache."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if fused_mode not in DENSE_FUSED_MODES:
        raise ValueError(
            "fused_mode must be one of: " + ", ".join(DENSE_FUSED_MODES)
        )
    selected = fused_mode if arm == "fused" else ""
    environ[FUSED_ENV] = selected
    mode_cache.clear()
    return selected


def activate_execution_arm(
    arm: str,
    *,
    execution_mode: str,
    dense_fused_mode: str,
    environ: MutableMapping[str, str],
    dense_mode_cache: list,
    moe_mode_cache: list,
) -> str:
    """Select one intentional A/B arm and invalidate both fused selectors.

    Production selectors remain process-stable and fail closed on mutation.
    This harness is the one controlled exception: it changes both environment
    gates and clears both selector caches together so no stale dense or MoE
    decision can leak across arms.  :func:`scoped_execution_arm` owns the
    corresponding exception-safe restore.
    """

    if execution_mode == "dense":
        environ[FUSED_MOE_ENV] = ""
        moe_mode_cache.clear()
        return activate_arm(
            arm,
            fused_mode=dense_fused_mode,
            environ=environ,
            mode_cache=dense_mode_cache,
        )
    if execution_mode not in ("moe128", "moe256"):
        raise ValueError(
            "execution_mode must be one of: dense, moe128, moe256"
        )
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if PREFILL_ENV in environ:
        raise RuntimeError(
            f"{PREFILL_ENV} must remain unset for the grouped-MoE A/B"
        )
    environ[FUSED_ENV] = ""
    selected = execution_mode.removeprefix("moe") if arm == "fused" else ""
    environ[FUSED_MOE_ENV] = selected
    dense_mode_cache.clear()
    moe_mode_cache.clear()
    return selected


_ENV_MISSING = object()


@contextmanager
def scoped_execution_arm(
    arm: str,
    *,
    execution_mode: str,
    dense_fused_mode: str,
    environ: MutableMapping[str, str],
    dense_mode_cache: list,
    moe_mode_cache: list,
    dense_selector: Callable[[], str],
    moe_selector: Callable[[], str],
) -> Iterator[str]:
    """Temporarily select and attest one same-process A/B execution arm.

    The scope restores the exact pre-arm environment and cache contents even
    when selection, dispatch, or synchronization raises.  Calling the real
    production selectors inside the scope proves that the arm did not merely
    mutate environment strings while leaving a stale cached decision active.
    """

    env_before = {
        name: environ[name] if name in environ else _ENV_MISSING
        for name in (FUSED_ENV, FUSED_MOE_ENV)
    }
    dense_before = list(dense_mode_cache)
    moe_before = list(moe_mode_cache)
    try:
        selected = activate_execution_arm(
            arm,
            execution_mode=execution_mode,
            dense_fused_mode=dense_fused_mode,
            environ=environ,
            dense_mode_cache=dense_mode_cache,
            moe_mode_cache=moe_mode_cache,
        )
        expected_dense = selected if execution_mode == "dense" else ""
        expected_moe = selected if execution_mode != "dense" else ""
        observed_dense = dense_selector()
        observed_moe = moe_selector()
        if observed_dense != expected_dense:
            raise RuntimeError(
                "dense fused-mode cache did not switch arms: expected "
                f"{expected_dense!r}, observed {observed_dense!r}"
            )
        if observed_moe != expected_moe:
            raise RuntimeError(
                "MoE fused-mode cache did not switch arms: expected "
                f"{expected_moe!r}, observed {observed_moe!r}"
            )
        yield selected
    finally:
        for name, previous in env_before.items():
            if previous is _ENV_MISSING:
                environ.pop(name, None)
            else:
                environ[name] = previous
        dense_mode_cache[:] = dense_before
        moe_mode_cache[:] = moe_before


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile with a defined empty-input result."""

    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    fraction = index - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    if not all(math.isfinite(v) for v in clean):
        raise ValueError("cannot summarize non-finite values")
    return {
        "count": len(clean),
        "mean": float(statistics.fmean(clean)),
        "min": min(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean),
    }


def _entry_logprob(value: Any) -> float:
    logprob = getattr(value, "logprob", None)
    if logprob is None and isinstance(value, Mapping):
        logprob = value.get("logprob")
    if logprob is None and isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("empty logprob tuple/list")
        logprob = value[0]
    if logprob is None:
        logprob = value
    parsed = float(logprob)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite logprob {parsed!r}")
    return parsed


def target_logprob(entries: Mapping[Any, Any] | None, token_id: int) -> float:
    """Extract vLLM's prompt-target logprob across supported value shapes."""

    if entries is None:
        raise KeyError(token_id)
    value = entries.get(token_id)
    if value is None:
        value = entries.get(str(token_id))
    if value is None:
        raise KeyError(token_id)
    return _entry_logprob(value)


@dataclass(frozen=True)
class TopKRow:
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]

    @property
    def coverage(self) -> float:
        return float(sum(math.exp(lp) for lp in self.logprobs))


def topk_row(entries: Mapping[Any, Any] | None, top_k: int) -> TopKRow:
    if entries is None:
        raise RuntimeError("vLLM returned no prompt logprobs for a position")
    parsed: list[tuple[int, float]] = []
    for raw_token, raw_value in entries.items():
        token = int(raw_token)
        if token < 0:
            continue
        parsed.append((token, _entry_logprob(raw_value)))
    parsed.sort(key=lambda item: (-item[1], item[0]))
    parsed = parsed[: int(top_k)]
    if not parsed:
        raise RuntimeError("vLLM returned an empty finite top-K row")
    return TopKRow(
        token_ids=tuple(item[0] for item in parsed),
        logprobs=tuple(item[1] for item in parsed),
    )


def full_vocab_row(
    entries: Mapping[Any, Any] | None, expected_vocab_size: int
) -> TopKRow:
    """Materialize one vLLM all-logprobs row in exact token-id order."""

    if entries is None:
        raise RuntimeError("vLLM returned no prompt logprobs for a position")
    if expected_vocab_size <= 0:
        raise ValueError("expected_vocab_size must be positive")
    values: list[float | None] = [None] * expected_vocab_size
    seen = 0
    for raw_token, raw_value in entries.items():
        token = int(raw_token)
        if token < 0:
            continue
        if token >= expected_vocab_size:
            raise RuntimeError(
                f"vLLM full-vocab row returned out-of-range token id {token} "
                f"for vocab_size={expected_vocab_size}"
            )
        if values[token] is not None:
            raise RuntimeError(
                f"vLLM full-vocab row returned duplicate token id {token}"
            )
        values[token] = _entry_logprob(raw_value)
        seen += 1
    if seen != expected_vocab_size or any(value is None for value in values):
        raise RuntimeError(
            "vLLM full-vocab row cardinality mismatch: expected exactly "
            f"{expected_vocab_size} token ids [0,{expected_vocab_size - 1}], "
            f"observed {seen}"
        )
    return TopKRow(
        token_ids=tuple(range(expected_vocab_size)),
        logprobs=tuple(float(value) for value in values if value is not None),
    )


def coarse_topk_kl(reference: TopKRow, candidate: TopKRow) -> float:
    """KL(reference || candidate) on reference top-K plus one tail bucket.

    vLLM does not materialize full-vocabulary prompt logprobs at every prompt
    position.  Missing candidate tokens therefore use its lowest retained
    top-K logprob, matching PrismaQuant's existing relative-A/B convention.
    The unrepresented probability mass is compared as one tail bucket.  This
    is a deterministic *coarse* KL suitable for paired regression checks, not
    an absolute full-vocabulary KL claim.
    """

    if not reference.token_ids or not candidate.token_ids:
        raise ValueError("KL rows must not be empty")
    candidate_map = dict(zip(candidate.token_ids, candidate.logprobs))
    floor = min(candidate.logprobs)
    p_values = [math.exp(lp) for lp in reference.logprobs]
    q_logprobs = [candidate_map.get(token, floor) for token in reference.token_ids]
    kl = sum(
        p * (lp - qlp)
        for p, lp, qlp in zip(p_values, reference.logprobs, q_logprobs)
    )
    epsilon = 1e-12
    p_tail = max(1.0 - sum(p_values), epsilon)
    q_tail = max(1.0 - sum(math.exp(lp) for lp in q_logprobs), epsilon)
    kl += p_tail * (math.log(p_tail) - math.log(q_tail))
    if not math.isfinite(kl):
        raise RuntimeError("coarse top-K KL became non-finite")
    # Tiny negative values are possible from summation when rows are equal.
    return float(max(kl, 0.0))


def exact_full_vocab_kl(reference: TopKRow, candidate: TopKRow) -> float:
    """Exact KL(reference || candidate) over attested complete vocab rows."""

    if reference.token_ids != candidate.token_ids or not reference.token_ids:
        raise RuntimeError("full-vocab KL rows have different token-id support")
    expected = tuple(range(len(reference.token_ids)))
    if reference.token_ids != expected:
        raise RuntimeError("full-vocab KL rows are not complete token-id order")
    for label, row in (("reference", reference), ("candidate", candidate)):
        mass = math.fsum(math.exp(logprob) for logprob in row.logprobs)
        if not math.isfinite(mass) or abs(mass - 1.0) > 1e-4:
            raise RuntimeError(
                f"{label} full-vocab row probability mass is {mass}, not 1"
            )
    kl = math.fsum(
        math.exp(ref_lp) * (ref_lp - cand_lp)
        for ref_lp, cand_lp in zip(reference.logprobs, candidate.logprobs)
    )
    if not math.isfinite(kl):
        raise RuntimeError("full-vocab KL became non-finite")
    return float(max(kl, 0.0))


def _row_kl(reference: TopKRow, candidate: TopKRow, kl_mode: str) -> float:
    if kl_mode == KL_COARSE_TOPK:
        return coarse_topk_kl(reference, candidate)
    if kl_mode == KL_FULL_VOCAB:
        return exact_full_vocab_kl(reference, candidate)
    raise ValueError(f"unknown KL mode {kl_mode!r}")


@dataclass(frozen=True)
class PromptScore:
    target_logprobs: tuple[float, ...]
    rows: tuple[TopKRow, ...]

    @property
    def mean_nll(self) -> float:
        return -statistics.fmean(self.target_logprobs)

    @property
    def ppl(self) -> float:
        return math.exp(self.mean_nll)


def score_prompt_output(
    output: Any,
    prompt_ids: Sequence[int],
    top_k: int,
    *,
    full_vocab: bool = False,
    expected_vocab_size: int | None = None,
) -> PromptScore:
    returned_prompt_ids = getattr(output, "prompt_token_ids", None)
    if returned_prompt_ids is None:
        raise RuntimeError("vLLM did not return prompt_token_ids for attestation")
    try:
        returned_prompt_ids = [int(token) for token in returned_prompt_ids]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM returned invalid prompt_token_ids") from exc
    expected_prompt_ids = [int(token) for token in prompt_ids]
    if returned_prompt_ids != expected_prompt_ids:
        raise RuntimeError(
            "vLLM returned prompt_token_ids different from the submitted prompt: "
            f"expected {expected_prompt_ids}, observed {returned_prompt_ids}"
        )
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs")
    if len(prompt_logprobs) != len(prompt_ids):
        raise RuntimeError(
            f"vLLM returned {len(prompt_logprobs)} prompt positions for "
            f"exactly {len(prompt_ids)} submitted input tokens"
        )
    targets: list[float] = []
    rows: list[TopKRow] = []
    for position in range(1, len(prompt_ids)):
        entries = prompt_logprobs[position]
        targets.append(target_logprob(entries, int(prompt_ids[position])))
        if full_vocab:
            if expected_vocab_size is None:
                raise RuntimeError("full-vocab scoring requires expected_vocab_size")
            rows.append(full_vocab_row(entries, expected_vocab_size))
        else:
            rows.append(topk_row(entries, top_k))
    if not targets:
        raise RuntimeError("a quality prompt must contain at least two tokens")
    return PromptScore(tuple(targets), tuple(rows))


def score_teacher_prompt(
    model: Any,
    prompt_ids: Sequence[int],
    top_k: int,
    torch: Any,
    *,
    full_vocab: bool = False,
    expected_vocab_size: int | None = None,
) -> PromptScore:
    """Score exact target NLL plus top-K or explicit full-vocabulary rows."""

    if len(prompt_ids) < 2:
        raise RuntimeError("a teacher prompt must contain at least two tokens")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [list(map(int, prompt_ids))], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1]
        if logits.ndim != 2 or logits.shape[0] != len(prompt_ids) - 1:
            raise RuntimeError(
                "teacher returned incompatible logits shape "
                f"{tuple(logits.shape)} for prompt length {len(prompt_ids)}"
            )
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        vocab_size = int(logprobs.shape[-1])
        if expected_vocab_size is not None and vocab_size != expected_vocab_size:
            raise RuntimeError(
                f"teacher logits vocab={vocab_size} differs from attested "
                f"candidate vocab={expected_vocab_size}"
            )
        targets = input_ids[0, 1:]
        target_values = logprobs.gather(1, targets[:, None]).squeeze(1)
        if full_vocab:
            values = logprobs
            token_ids = None
        else:
            retained = min(int(top_k), vocab_size)
            values, token_ids = torch.topk(
                logprobs, k=retained, dim=-1, largest=True, sorted=True
            )
    target_cpu = target_values.detach().cpu().tolist()
    values_cpu = values.detach().cpu().tolist()
    if full_vocab:
        full_ids = tuple(range(vocab_size))
        rows = tuple(
            TopKRow(
                token_ids=full_ids,
                logprobs=tuple(float(value) for value in row_values),
            )
            for row_values in values_cpu
        )
    else:
        assert token_ids is not None
        tokens_cpu = token_ids.detach().cpu().tolist()
        rows = tuple(
            TopKRow(
                token_ids=tuple(int(token) for token in row_tokens),
                logprobs=tuple(float(value) for value in row_values),
            )
            for row_tokens, row_values in zip(tokens_cpu, values_cpu)
        )
    return PromptScore(
        target_logprobs=tuple(float(value) for value in target_cpu),
        rows=rows,
    )


_COUNT_FIELDS = (
    "apply_calls",
    "fp4_apply_calls",
    "fp4_prefill_calls",
    "candidate_gate_opportunities",
    "fused_attempts",
    "fused_successes",
    "fused_fallbacks",
    "fused_errors",
    "loop_calls",
    "loop_successes",
    "loop_errors",
    "probe_errors",
)


def _empty_dispatch_record(label: str, arm: str | None) -> dict[str, Any]:
    return {
        "label": label,
        "arm": arm,
        **{field: 0 for field in _COUNT_FIELDS},
        "pids": set(),
        "apply_shapes": Counter(),
        "success_shapes": Counter(),
        "fallback_shapes": Counter(),
        "loop_shapes": Counter(),
        "success_prefixes": Counter(),
        "fallback_prefixes": Counter(),
        "loop_prefixes": Counter(),
    }


def _json_dispatch_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": record["label"],
        "arm": record["arm"],
        **{field: int(record[field]) for field in _COUNT_FIELDS},
        "pids": sorted(int(pid) for pid in record["pids"]),
        "apply_shapes": dict(sorted(record["apply_shapes"].items())),
        "success_shapes": dict(sorted(record["success_shapes"].items())),
        "fallback_shapes": dict(sorted(record["fallback_shapes"].items())),
        "loop_shapes": dict(sorted(record["loop_shapes"].items())),
        "success_prefixes": dict(sorted(record["success_prefixes"].items())),
        "fallback_prefixes": dict(sorted(record["fallback_prefixes"].items())),
        "loop_prefixes": dict(sorted(record["loop_prefixes"].items())),
    }


def aggregate_dispatch(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _empty_dispatch_record("aggregate", None)
    for record in records:
        for field in _COUNT_FIELDS:
            merged[field] += int(record[field])
        merged["pids"].update(record["pids"])
        for field in (
            "apply_shapes",
            "success_shapes",
            "fallback_shapes",
            "loop_shapes",
            "success_prefixes",
            "fallback_prefixes",
            "loop_prefixes",
        ):
            merged[field].update(record[field])
    result = _json_dispatch_record(merged)
    result.pop("label")
    result.pop("arm")
    attempts = result["fused_attempts"]
    result["fused_success_fraction"] = (
        result["fused_successes"] / attempts if attempts else None
    )
    return result


class DenseDispatchProbe:
    """Temporary class wrapper proving dense fused dispatch in this PID."""

    def __init__(self, method_class: type, *, prefill_threshold: int, fused_mode: str):
        self.method_class = method_class
        self.prefill_threshold = int(prefill_threshold)
        self.fused_mode = fused_mode
        self.current: dict[str, Any] | None = None
        self.unscoped = _empty_dispatch_record("unscoped", None)
        self._original_apply = None
        self._original_try = None

    def _record(self) -> dict[str, Any]:
        return self.current if self.current is not None else self.unscoped

    def _on_apply(self, method: Any, layer: Any, x: Any, bias: Any) -> None:
        record = self._record()
        record["pids"].add(os.getpid())
        record["apply_calls"] += 1
        try:
            if not bool(method.is_fp4):
                return
            record["fp4_apply_calls"] += 1
            n = int(layer._cb_N)
            k = int(layer._cb_K)
            m = int(x.numel()) // k
            shape = f"M{m}:N{n}:K{k}"
            record["apply_shapes"][shape] += 1
            if m > self.prefill_threshold:
                record["fp4_prefill_calls"] += 1
                mode_allows = self.fused_mode in ("1", "rowwise") or (
                    self.fused_mode in ("midm", "rowwise_midm") and m <= 128
                )
                if record["arm"] == "fused" and bias is None and mode_allows:
                    record["candidate_gate_opportunities"] += 1
        except Exception:  # noqa: BLE001 - probe must not change model execution
            record["probe_errors"] += 1

    def _on_try_start(self, method: Any, layer: Any, n: int, k: int, m: int) -> str:
        del method
        record = self._record()
        record["pids"].add(os.getpid())
        record["fused_attempts"] += 1
        return f"M{int(m)}:N{int(n)}:K{int(k)}"

    def _on_try_finish(
        self, method: Any, shape: str, output: Any, *, error: bool = False
    ) -> None:
        record = self._record()
        prefix = str(getattr(method, "prefix", "<unknown>"))
        if error:
            record["fused_errors"] += 1
        elif output is None:
            record["fused_fallbacks"] += 1
            record["fallback_shapes"][shape] += 1
            record["fallback_prefixes"][prefix] += 1
        else:
            record["fused_successes"] += 1
            record["success_shapes"][shape] += 1
            record["success_prefixes"][prefix] += 1

    def install(self) -> None:
        if self._original_apply is not None:
            raise RuntimeError("dispatch probe is already installed")
        probe = self
        self._original_apply = self.method_class._apply_inline
        self._original_try = self.method_class._try_fused_fp4
        original_apply = self._original_apply
        original_try = self._original_try

        def wrapped_apply(method, layer, x, bias=None):
            probe._on_apply(method, layer, x, bias)
            return original_apply(method, layer, x, bias)

        def wrapped_try(method, layer, x, n, k, m, *, rowwise=False):
            shape = probe._on_try_start(method, layer, n, k, m)
            try:
                if rowwise:
                    output = original_try(
                        method, layer, x, n, k, m, rowwise=True
                    )
                else:
                    output = original_try(method, layer, x, n, k, m)
            except Exception:
                probe._on_try_finish(method, shape, None, error=True)
                raise
            probe._on_try_finish(method, shape, output)
            return output

        self.method_class._apply_inline = wrapped_apply
        self.method_class._try_fused_fp4 = wrapped_try

    def restore(self) -> None:
        if self._original_apply is None:
            return
        self.method_class._apply_inline = self._original_apply
        self.method_class._try_fused_fp4 = self._original_try
        self._original_apply = None
        self._original_try = None

    @contextlib.contextmanager
    def measurement(self, arm: str, label: str):
        if arm not in ARMS:
            raise ValueError(f"invalid arm {arm!r}")
        if self.current is not None:
            raise RuntimeError("dispatch measurements may not overlap")
        record = _empty_dispatch_record(label, arm)
        self.current = record
        try:
            yield record
        finally:
            self.current = None


class MoEDispatchProbe:
    """Temporary wrappers proving loop-vs-grouped-fused MoE dispatch."""

    def __init__(self, method_class: type):
        self.method_class = method_class
        self.current: dict[str, Any] | None = None
        self.unscoped = _empty_dispatch_record("unscoped", None)
        self._original_fused = None
        self._original_loop = None

    def _record(self) -> dict[str, Any]:
        return self.current if self.current is not None else self.unscoped

    @staticmethod
    def _shape(layer: Any, x: Any, topk_ids: Any, tile_m: int | None) -> str:
        base = (
            f"T{int(x.shape[0])}:E{int(layer._cb_E)}:"
            f"H{int(layer._cb_hidden)}:I{int(layer._cb_inter)}:"
            f"topk{int(topk_ids.shape[-1])}"
        )
        return f"{base}:tile{int(tile_m)}" if tile_m is not None else base

    def install(self) -> None:
        if self._original_fused is not None:
            raise RuntimeError("dispatch probe is already installed")
        probe = self
        self._original_fused = (
            self.method_class._apply_prefill_grouped_fused_fp4
        )
        self._original_loop = self.method_class._apply_prefill_loop
        original_fused = self._original_fused
        original_loop = self._original_loop

        def wrapped_fused(
            method, layer, x, topk_weights, topk_ids, act, *, tile_m=128,
            rowwise=False,
        ):
            record = probe._record()
            record["pids"].add(os.getpid())
            record["fused_attempts"] += 1
            try:
                shape = probe._shape(layer, x, topk_ids, tile_m)
            except Exception:  # noqa: BLE001 - telemetry must not alter serving
                shape = f"tile{int(tile_m)}:<probe-error>"
                record["probe_errors"] += 1
            prefix = str(getattr(method, "prefix", "<unknown>"))
            try:
                kwargs = {"tile_m": tile_m}
                if rowwise:
                    kwargs["rowwise"] = True
                output = original_fused(
                    method, layer, x, topk_weights, topk_ids, act, **kwargs
                )
            except Exception:
                record["fused_errors"] += 1
                raise
            if output is None:
                record["fused_fallbacks"] += 1
                record["fallback_shapes"][shape] += 1
                record["fallback_prefixes"][prefix] += 1
            else:
                record["fused_successes"] += 1
                record["success_shapes"][shape] += 1
                record["success_prefixes"][prefix] += 1
            return output

        def wrapped_loop(method, layer, x, topk_weights, topk_ids, act):
            record = probe._record()
            record["pids"].add(os.getpid())
            record["loop_calls"] += 1
            try:
                shape = probe._shape(layer, x, topk_ids, None)
            except Exception:  # noqa: BLE001 - telemetry must not alter serving
                shape = "<probe-error>"
                record["probe_errors"] += 1
            prefix = str(getattr(method, "prefix", "<unknown>"))
            try:
                output = original_loop(
                    method, layer, x, topk_weights, topk_ids, act
                )
            except Exception:
                record["loop_errors"] += 1
                raise
            record["loop_successes"] += 1
            record["loop_shapes"][shape] += 1
            record["loop_prefixes"][prefix] += 1
            return output

        self.method_class._apply_prefill_grouped_fused_fp4 = wrapped_fused
        self.method_class._apply_prefill_loop = wrapped_loop

    def restore(self) -> None:
        if self._original_fused is None:
            return
        self.method_class._apply_prefill_grouped_fused_fp4 = self._original_fused
        self.method_class._apply_prefill_loop = self._original_loop
        self._original_fused = None
        self._original_loop = None

    @contextlib.contextmanager
    def measurement(self, arm: str, label: str):
        if arm not in ARMS:
            raise ValueError(f"invalid arm {arm!r}")
        if self.current is not None:
            raise RuntimeError("dispatch measurements may not overlap")
        record = _empty_dispatch_record(label, arm)
        self.current = record
        try:
            yield record
        finally:
            self.current = None


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*argv: str) -> str | None:
        try:
            return subprocess.run(
                [*argv], cwd=repo, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    root = run("git", "rev-parse", "--show-toplevel")
    status = run("git", "status", "--porcelain") if root is not None else None
    return {
        "root": root,
        "commit": run("git", "rev-parse", "HEAD") if root is not None else None,
        "branch": (
            run("git", "branch", "--show-current") if root is not None else None
        ),
        # A wheel/site-packages install is commonly outside a Git worktree.
        # Reporting that as clean would falsely attest an unrelated checkout.
        "dirty": bool(status) if status is not None else None,
    }


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _required_file_record(path: Path) -> dict[str, Any]:
    """Hash one required provenance file, failing instead of recording null."""

    digest = _sha256(path)
    if digest is None:
        raise RuntimeError(f"required provenance file is unreadable: {path}")
    return {"bytes": path.stat().st_size, "sha256": digest}


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _config_dict(config: Any) -> dict[str, Any]:
    raw = config.to_dict() if callable(getattr(config, "to_dict", None)) else config
    if not isinstance(raw, Mapping):
        raise RuntimeError("model config does not expose a mapping/to_dict identity")
    return dict(raw)


def _config_identity(config: Any) -> dict[str, Any]:
    raw = _config_dict(config)
    canonical = {
        key: value
        for key, value in raw.items()
        if key not in _CONFIG_IDENTITY_IGNORED_KEYS
    }
    return {
        "sha256": _json_sha256(canonical),
        "model_type": raw.get("model_type"),
        "architectures": raw.get("architectures"),
        "vocab_size": raw.get("vocab_size"),
        "canonical": canonical,
    }


def _assert_config_identity(candidate: Any, teacher: Any) -> dict[str, Any]:
    teacher_raw = _config_dict(teacher)
    teacher_quant = teacher_raw.get("quantization_config")
    if teacher_quant not in (None, {}, False):
        raise RuntimeError(
            "teacher config declares quantization_config; refusing to label it "
            "an unquantized BF16 reference"
        )
    candidate_id = _config_identity(candidate)
    teacher_id = _config_identity(teacher)
    if candidate_id["canonical"] != teacher_id["canonical"]:
        keys = sorted(
            key
            for key in set(candidate_id["canonical"]) | set(teacher_id["canonical"])
            if candidate_id["canonical"].get(key)
            != teacher_id["canonical"].get(key)
        )
        raise RuntimeError(
            "candidate/teacher base configs differ after excluding only "
            f"{sorted(_CONFIG_IDENTITY_IGNORED_KEYS)}; differing keys: {keys[:20]}"
        )
    return {
        "match": True,
        "canonical_sha256": candidate_id["sha256"],
        "model_type": candidate_id["model_type"],
        "architectures": candidate_id["architectures"],
        "vocab_size": candidate_id["vocab_size"],
        "ignored_keys": sorted(_CONFIG_IDENTITY_IGNORED_KEYS),
        "teacher_quantization_config_absent": True,
    }


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RuntimeError("tokenizer does not expose get_vocab()")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise RuntimeError("tokenizer returned an empty/non-mapping vocabulary")
    vocab_digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items(), key=lambda item: str(item[0])):
        token_bytes = str(token).encode("utf-8")
        vocab_digest.update(len(token_bytes).to_bytes(8, "little"))
        vocab_digest.update(token_bytes)
        vocab_digest.update(int(token_id).to_bytes(8, "little", signed=True))

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    backend_sha256 = None
    if callable(backend_to_str):
        backend_sha256 = hashlib.sha256(
            backend_to_str().encode("utf-8")
        ).hexdigest()

    special_ids = {
        name: getattr(tokenizer, f"{name}_token_id", None)
        for name in ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
    }
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_sha256": vocab_digest.hexdigest(),
        "backend_sha256": backend_sha256,
        "get_vocab_size": len(vocab),
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(vocab))),
        "length": int(len(tokenizer)),
        "special_token_ids": special_ids,
    }
    identity["sha256"] = _json_sha256(identity)
    return identity


def _assert_tokenizer_identity(candidate: Any, teacher: Any) -> dict[str, Any]:
    candidate_id = _tokenizer_identity(candidate)
    teacher_id = _tokenizer_identity(teacher)
    if candidate_id != teacher_id:
        fields = sorted(
            key
            for key in set(candidate_id) | set(teacher_id)
            if candidate_id.get(key) != teacher_id.get(key)
        )
        raise RuntimeError(
            "candidate/teacher tokenizers are not identical; differing "
            f"identity fields: {fields}"
        )
    return {"match": True, **candidate_id}


def _tokenizer_file_records(model_dir: Path) -> dict[str, dict[str, Any]]:
    if not model_dir.is_dir():
        return {}
    return {
        name: _required_file_record(model_dir / name)
        for name in _TOKENIZER_FILES
        if (model_dir / name).is_file()
    }


def _assert_local_tokenizer_files_match(
    candidate_dir: Path, teacher_dir: Path
) -> dict[str, Any] | None:
    if not candidate_dir.is_dir() or not teacher_dir.is_dir():
        return None
    candidate = _tokenizer_file_records(candidate_dir)
    teacher = _tokenizer_file_records(teacher_dir)
    if not candidate or not teacher:
        raise RuntimeError(
            "local candidate/teacher must both contain tokenizer provenance files"
        )
    if candidate != teacher:
        differing = sorted(
            name
            for name in set(candidate) | set(teacher)
            if candidate.get(name) != teacher.get(name)
        )
        raise RuntimeError(
            "local candidate/teacher tokenizer files differ: "
            f"{differing}"
        )
    return {"match": True, "files": candidate}


def _local_model_provenance(
    model_dir: Path, *, role: str
) -> dict[str, Any] | None:
    """Hash the exact local config, tokenizer, weights, and CB sidecars."""

    if not model_dir.is_dir():
        return None
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"local {role} has no config.json: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    index_record = None
    if index_path.is_file():
        index_record = _required_file_record(index_path)
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid {role} safetensors index: {index_path}"
            ) from exc
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RuntimeError(f"{role} safetensors index has an empty weight_map")
        relative_weights = sorted({str(name) for name in weight_map.values()})
    else:
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise RuntimeError(
                f"local {role} must use model.safetensors or a safetensors index"
            )
        relative_weights = ["model.safetensors"]

    resolved_root = model_dir.resolve()
    weights: dict[str, dict[str, Any]] = {}
    for relative in relative_weights:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(
                f"{role} safetensors index escapes model directory: {relative}"
            )
        # Hugging Face snapshot files are commonly symlinks into the cache's
        # blob store.  The index path must remain lexical-relative, but the
        # payload symlink itself is valid provenance and is hashed by content.
        path = model_dir / relative_path
        if not path.is_file():
            raise RuntimeError(f"{role} weight shard is missing: {path}")
        weights[relative] = _required_file_record(path)

    tokenizer_files = _tokenizer_file_records(model_dir)
    if not tokenizer_files:
        raise RuntimeError(f"local {role} has no tokenizer provenance files")

    quant_config_path = model_dir / "quant_config.json"
    quant_config_record = None
    codebook_record = None
    if quant_config_path.is_file():
        quant_config_record = _required_file_record(quant_config_path)
        try:
            quant_config = json.loads(
                quant_config_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid {role} quant_config.json: {quant_config_path}"
            ) from exc
        if not isinstance(quant_config, Mapping):
            raise RuntimeError(
                f"{role} quant_config.json must contain a JSON object"
            )
        codebook_file = quant_config.get("codebook_file")
        if not isinstance(codebook_file, str) or not codebook_file.strip():
            raise RuntimeError(
                f"{role} quant_config.json must declare a nonempty "
                "codebook_file"
            )
        relative_codebook = Path(codebook_file)
        if relative_codebook.is_absolute() or ".." in relative_codebook.parts:
            raise RuntimeError(
                f"{role} codebook_file escapes model directory: "
                f"{codebook_file}"
            )
        try:
            resolved_codebook = (model_dir / relative_codebook).resolve(
                strict=True
            )
            resolved_codebook.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{role} codebook_file is missing or resolves outside the "
                f"model directory: {codebook_file}"
            ) from exc
        if not resolved_codebook.is_file():
            raise RuntimeError(
                f"{role} codebook_file is not a file: {resolved_codebook}"
            )
        codebook_record = {
            "relative_path": relative_codebook.as_posix(),
            "resolved_path": str(resolved_codebook),
            **_required_file_record(resolved_codebook),
        }
    elif role == "candidate":
        raise RuntimeError(
            f"local candidate has no required quant_config.json: {model_dir}"
        )

    return {
        "kind": "local_directory",
        "role": role,
        "path": str(resolved_root),
        "config": _required_file_record(config_path),
        "safetensors_index": index_record,
        "weight_files": weights,
        "weight_bytes": sum(record["bytes"] for record in weights.values()),
        "tokenizer_files": tokenizer_files,
        "quant_config": quant_config_record,
        "codebook_file": codebook_record,
    }


def _local_teacher_provenance(model_dir: Path) -> dict[str, Any] | None:
    """Compatibility wrapper for callers/tests using the original helper name."""

    return _local_model_provenance(model_dir, role="teacher")


def _hub_model_provenance(
    config: Any,
    *,
    role: str,
    model_id: str,
    requested_revision: str,
) -> dict[str, Any]:
    """Attest the immutable Hub commit actually selected by Transformers."""

    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise RuntimeError(
            f"non-local {role} {model_id!r} has no explicit requested revision"
        )
    resolved_commit = getattr(config, "_commit_hash", None)
    immutable_commit = None
    if isinstance(resolved_commit, str):
        candidate = resolved_commit.strip()
        if len(candidate) == 40 and all(
            char in "0123456789abcdefABCDEF" for char in candidate
        ):
            immutable_commit = candidate.lower()
    if immutable_commit is None:
        raise RuntimeError(
            f"non-local {role} {model_id!r} did not expose a resolved full "
            "40-hex Transformers _commit_hash; refusing mutable or "
            f"unattested revision {requested_revision!r}"
        )
    return {
        "kind": "hub_model_id",
        "role": role,
        "model_id": model_id,
        "requested_revision": requested_revision.strip(),
        "resolved_commit_hash": immutable_commit,
    }


def _attest_teacher_model(model: Any, torch: Any) -> dict[str, Any]:
    """Require an actually-loaded, unquantized BF16-parameter teacher."""

    raw_config = _config_dict(model.config)
    if raw_config.get("quantization_config") not in (None, {}, False):
        raise RuntimeError("loaded teacher unexpectedly declares quantization")
    for attr in ("hf_quantizer", "quantization_method"):
        if getattr(model, attr, None) is not None:
            raise RuntimeError(f"loaded teacher exposes {attr}; refusing quantized reference")

    parameter_dtypes: Counter[str] = Counter()
    parameter_numel: Counter[str] = Counter()
    parameter_devices: Counter[str] = Counter()
    bad_parameters: list[str] = []
    for name, parameter in model.named_parameters():
        dtype_name = str(parameter.dtype)
        parameter_dtypes[dtype_name] += 1
        parameter_numel[dtype_name] += int(parameter.numel())
        parameter_devices[str(parameter.device)] += 1
        if parameter.dtype != torch.bfloat16:
            bad_parameters.append(f"{name}:{dtype_name}")
    if not parameter_dtypes:
        raise RuntimeError("teacher has no parameters to attest")
    if bad_parameters:
        raise RuntimeError(
            "teacher parameters are not uniformly BF16: "
            f"{bad_parameters[:20]}"
        )

    buffer_dtypes: Counter[str] = Counter()
    buffer_numel: Counter[str] = Counter()
    for _name, buffer in model.named_buffers():
        dtype_name = str(buffer.dtype)
        buffer_dtypes[dtype_name] += 1
        buffer_numel[dtype_name] += int(buffer.numel())
    return {
        "parameters_all_bfloat16": True,
        "parameter_tensors_by_dtype": dict(sorted(parameter_dtypes.items())),
        "parameter_elements_by_dtype": dict(sorted(parameter_numel.items())),
        "parameter_tensors_by_device": dict(sorted(parameter_devices.items())),
        "buffer_tensors_by_dtype": dict(sorted(buffer_dtypes.items())),
        "buffer_elements_by_dtype": dict(sorted(buffer_numel.items())),
        "quantization_config_absent": True,
        "quantizer_attributes_absent": True,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _command_output(*argv: str) -> str | None:
    try:
        output = subprocess.run(
            list(argv), check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0].strip() if output else None


def _required_module_path(module: Any, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if raw is None:
        raise RuntimeError(f"imported {label} module has no filesystem __file__")
    path = Path(raw).resolve()
    if not path.is_file():
        raise RuntimeError(f"imported {label} module file is unreadable: {path}")
    return path


def _runtime_provenance(
    torch: Any,
    vllm: Any,
    gridbook: Any,
    gridbook_config: Any,
    linear: Any,
    moe: Any,
    moe_toplevel_loader: Any,
    cb_fill_guard: Any,
    cuda_ext: Any,
    harness_path: Path,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gridbook_init = _required_module_path(gridbook, "gridbook")
    package_root = gridbook_init.parent
    imported_modules = {
        "__init__.py": gridbook_init,
        "config.py": _required_module_path(gridbook_config, "gridbook.config"),
        "linear.py": _required_module_path(linear, "gridbook.linear"),
        "moe.py": _required_module_path(moe, "gridbook.moe"),
        "moe_toplevel_loader.py": _required_module_path(
            moe_toplevel_loader, "gridbook.moe_toplevel_loader"
        ),
        "cb_fill_guard.py": _required_module_path(
            cb_fill_guard, "gridbook.cb_fill_guard"
        ),
        "cuda_ext.py": _required_module_path(cuda_ext, "gridbook.cuda_ext"),
    }
    for label, path in imported_modules.items():
        if path.parent != package_root:
            raise RuntimeError(
                "mixed gridbook package provenance: "
                f"{label} resolved to {path}, outside {package_root}"
            )
    csrc_root = Path(cuda_ext.csrc_dir()).resolve()
    source_paths = {
        **imported_modules,
        "plugin.py": package_root / "plugin.py",
        "ops.py": package_root / "ops.py",
        "codec.py": package_root / "codec.py",
        "cb_fused_fp4_gemm.cu": csrc_root / "cb_fused_fp4_gemm.cu",
        "sm120_cb_fused_fp4_mma.hpp": (
            csrc_root / "cutlass_fork" / "sm120_cb_fused_fp4_mma.hpp"
        ),
    }
    source_files = {
        label: {"path": str(path.resolve()), **_required_file_record(path)}
        for label, path in source_paths.items()
    }
    harness_path = harness_path.resolve()
    harness = {
        "path": str(harness_path),
        **_required_file_record(harness_path),
        "shared_helpers": {
            "fused_nvfp4_validation": {
                "path": str(_COMMON_PATH),
                **_required_file_record(_COMMON_PATH),
            }
        },
    }
    gridbook_version = getattr(gridbook, "__version__", None)
    if not isinstance(gridbook_version, str) or not gridbook_version:
        raise RuntimeError("imported gridbook package has no nonempty __version__")
    vllm_version = getattr(vllm, "__version__", None)
    if vllm_version is None:
        try:
            vllm_version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            vllm_version = None
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "vllm": vllm_version,
        "vllm_module_path": str(_required_module_path(vllm, "vllm")),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_driver": _command_output(
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader"
        ),
        "gpu": {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": int(props.total_memory),
        },
        "gridbook": {
            "version": gridbook_version,
            "module_path": str(gridbook_init),
            "package_root": str(package_root),
            "git": _git_state(package_root),
        },
        "source_files": source_files,
        # Retained as a compact compatibility view, but derived from the exact
        # imported package rather than from the harness checkout.
        "source_sha256": {
            label: record["sha256"] for label, record in source_files.items()
        },
        "harness": harness,
        "environment": {
            VLLM_MP_ENV: os.environ.get(VLLM_MP_ENV),
            "PRISMAQUANT_CB_DISPATCH": os.environ.get(
                "PRISMAQUANT_CB_DISPATCH", "op"
            ),
            "PRISMAQUANT_CB_EXT_DIR": os.environ.get("PRISMAQUANT_CB_EXT_DIR"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "GRIDBOOK_VALIDATION_IMAGE": os.environ.get(
                "GRIDBOOK_VALIDATION_IMAGE"
            ),
        },
    }


def _load_wikitext_windows(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[list[list[int]], dict[str, Any]]:
    text_path = args.wikitext_text
    if text_path is not None:
        text_path = text_path.resolve()
        text = text_path.read_text(encoding="utf-8")
        text_record = _required_file_record(text_path)
        source = {
            "source": "raw_text",
            "text_path": str(text_path),
            "text_bytes": text_record["bytes"],
            "text_sha256": text_record["sha256"],
        }
    else:
        try:
            from datasets import DownloadConfig, load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "the serving image does not include `datasets`; pass "
                "--wikitext-text with a cached WikiText-2 raw split"
            ) from exc
        download_config = DownloadConfig(
            local_files_only=not args.allow_downloads
        )
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            download_config=download_config,
        )
        dataset_fingerprint = getattr(dataset, "_fingerprint", None)
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
            raise RuntimeError(
                "loaded WikiText dataset did not expose a nonempty fingerprint"
            )
        text = "\n\n".join(
            row["text"] for row in dataset if row.get("text", "").strip()
        )
        source = {
            "source": "huggingface_datasets",
            "cache_dir": str(Path(args.dataset_cache_dir).resolve()),
            "downloads_allowed": bool(args.allow_downloads),
            "dataset_fingerprint": dataset_fingerprint,
        }
    corpus_bytes = text.encode("utf-8")
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token) for token in ids]
    if len(ids) < args.seqlen:
        raise RuntimeError(
            f"WikiText produced {len(ids)} tokens, fewer than --seqlen={args.seqlen}"
        )
    max_start = len(ids) - args.seqlen
    available_windows = max_start + 1
    if available_windows < args.n_samples:
        raise RuntimeError(
            "WikiText does not contain enough distinct start positions: "
            f"need {args.n_samples}, have {available_windows} for "
            f"seqlen={args.seqlen}"
        )
    rng = random.Random(args.window_seed)
    starts = rng.sample(range(available_windows), args.n_samples)
    windows = [ids[start : start + args.seqlen] for start in starts]
    if len({tuple(window) for window in windows}) != len(windows):
        raise RuntimeError(
            "sampled WikiText windows are not content-distinct; choose a "
            "different --window-seed/corpus or reduce --n-samples"
        )
    digest = hashlib.sha256(
        json.dumps(windows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return windows, {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": args.dataset_split,
        **source,
        "corpus_utf8_bytes": len(corpus_bytes),
        "corpus_text_sha256": corpus_sha256,
        "total_tokenized_tokens": len(ids),
        "window_seed": args.window_seed,
        "starts": starts,
        "n_samples": args.n_samples,
        "seqlen": args.seqlen,
        "prompt_token_ids_sha256": digest,
        "prompt_window_sha256": [
            hashlib.sha256(
                json.dumps(window, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for window in windows
        ],
    }


def _select_arm(
    linear: Any,
    moe: Any,
    arm: str,
    execution_mode: str,
    dense_fused_mode: str,
) -> Any:
    return scoped_execution_arm(
        arm,
        execution_mode=execution_mode,
        dense_fused_mode=dense_fused_mode,
        environ=os.environ,
        dense_mode_cache=linear._FP4_FUSED_MODE,
        moe_mode_cache=moe._FUSED_FP4_MOE_STATE,
        dense_selector=linear._fp4_fused_mode,
        moe_selector=moe._requested_fused_fp4_moe_mode,
    )


def _run_generate(
    *,
    llm: Any,
    sampling: Any,
    prompt_ids: list[int],
    arm: str,
    label: str,
    execution_mode: str,
    dense_fused_mode: str,
    linear: Any,
    moe: Any,
    probe: Any,
    synchronize: Any,
) -> tuple[Any, float, dict[str, Any]]:
    with _select_arm(linear, moe, arm, execution_mode, dense_fused_mode):
        synchronize()
        started = time.perf_counter()
        with probe.measurement(arm, label) as raw_record:
            result = llm.generate(
                [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
            )[0]
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result, elapsed_ms, _json_dispatch_record(raw_record)


def _kl_convention(kl_mode: str) -> str:
    if kl_mode == KL_FULL_VOCAB:
        return (
            "exact KL over every token id in the model vocabulary; both row "
            "cardinality and token-id coverage are fail-closed"
        )
    if kl_mode == KL_COARSE_TOPK:
        return (
            "reference top-K support plus one tail bucket; missing candidate "
            "tokens use the candidate's lowest retained top-K logprob"
        )
    raise ValueError(f"unknown KL mode {kl_mode!r}")


def _quality_summary(
    pairs: Sequence[Mapping[str, Any]], *, kl_mode: str = KL_COARSE_TOPK
) -> dict[str, Any]:
    arm_targets: dict[str, list[float]] = {arm: [] for arm in ARMS}
    coverages: dict[str, list[float]] = {arm: [] for arm in ARMS}
    forward_kl: list[float] = []
    reverse_kl: list[float] = []
    target_abs_delta: list[float] = []
    confident_forward: list[float] = []
    per_prompt = []
    for pair in pairs:
        baseline: PromptScore = pair["scores"]["baseline"]
        fused: PromptScore = pair["scores"]["fused"]
        if len(baseline.rows) != len(fused.rows):
            raise RuntimeError("paired arms returned different prompt-score lengths")
        prompt_fwd, prompt_rev = [], []
        for b_row, f_row, b_target, f_target in zip(
            baseline.rows,
            fused.rows,
            baseline.target_logprobs,
            fused.target_logprobs,
        ):
            fwd = _row_kl(b_row, f_row, kl_mode)
            rev = _row_kl(f_row, b_row, kl_mode)
            forward_kl.append(fwd)
            reverse_kl.append(rev)
            prompt_fwd.append(fwd)
            prompt_rev.append(rev)
            target_abs_delta.append(abs(b_target - f_target))
            if math.exp(max(b_row.logprobs)) > 0.5:
                confident_forward.append(fwd)
        for arm, score in (("baseline", baseline), ("fused", fused)):
            arm_targets[arm].extend(score.target_logprobs)
            coverages[arm].extend(row.coverage for row in score.rows)
        per_prompt.append({
            "prompt_index": pair["prompt_index"],
            "pair_order": pair["pair_order"],
            "positions": len(baseline.rows),
            "baseline_mean_nll": baseline.mean_nll,
            "fused_mean_nll": fused.mean_nll,
            "mean_nll_delta_fused_minus_baseline": (
                fused.mean_nll - baseline.mean_nll
            ),
            "baseline_to_fused_kl_mean": statistics.fmean(prompt_fwd),
            "fused_to_baseline_kl_mean": statistics.fmean(prompt_rev),
        })

    arm_metrics = {}
    for arm in ARMS:
        mean_nll = -statistics.fmean(arm_targets[arm])
        arm_metrics[arm] = {
            "tokens_scored": len(arm_targets[arm]),
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll),
            "topk_coverage": summarize_values(coverages[arm]),
        }
    nll_delta = arm_metrics["fused"]["mean_nll"] - arm_metrics["baseline"]["mean_nll"]
    ppl_ratio = arm_metrics["fused"]["ppl"] / arm_metrics["baseline"]["ppl"]
    return {
        "arms": arm_metrics,
        "delta": {
            "mean_nll_fused_minus_baseline": nll_delta,
            "ppl_fused_over_baseline": ppl_ratio,
            "ppl_relative_regression": ppl_ratio - 1.0,
            "target_logprob_abs_delta": summarize_values(target_abs_delta),
            "kl_baseline_to_fused": summarize_values(forward_kl),
            "kl_fused_to_baseline": summarize_values(reverse_kl),
            "kl_baseline_to_fused_confident_positions": summarize_values(
                confident_forward
            ),
        },
        "per_prompt": per_prompt,
        "kl_mode": kl_mode,
        "kl_convention": _kl_convention(kl_mode),
    }


def _pairwise_score_summary(
    reference_scores: Sequence[PromptScore],
    candidate_scores: Sequence[PromptScore],
    *,
    reference_name: str,
    candidate_name: str,
    kl_mode: str = KL_COARSE_TOPK,
) -> dict[str, Any]:
    if len(reference_scores) != len(candidate_scores) or not reference_scores:
        raise RuntimeError("teacher/candidate prompt-score sets must align")
    reference_targets: list[float] = []
    candidate_targets: list[float] = []
    forward_kl: list[float] = []
    reverse_kl: list[float] = []
    target_abs_delta: list[float] = []
    per_prompt = []
    for prompt_index, (reference, candidate) in enumerate(
        zip(reference_scores, candidate_scores)
    ):
        if (
            len(reference.rows) != len(candidate.rows)
            or len(reference.target_logprobs) != len(reference.rows)
            or len(candidate.target_logprobs) != len(candidate.rows)
        ):
            raise RuntimeError("teacher/candidate prompt-score lengths differ")
        prompt_forward: list[float] = []
        prompt_reverse: list[float] = []
        for ref_row, cand_row, ref_target, cand_target in zip(
            reference.rows,
            candidate.rows,
            reference.target_logprobs,
            candidate.target_logprobs,
        ):
            fwd = _row_kl(ref_row, cand_row, kl_mode)
            rev = _row_kl(cand_row, ref_row, kl_mode)
            forward_kl.append(fwd)
            reverse_kl.append(rev)
            prompt_forward.append(fwd)
            prompt_reverse.append(rev)
            target_abs_delta.append(abs(ref_target - cand_target))
        reference_targets.extend(reference.target_logprobs)
        candidate_targets.extend(candidate.target_logprobs)
        per_prompt.append({
            "prompt_index": prompt_index,
            "positions": len(reference.rows),
            f"{reference_name}_mean_nll": reference.mean_nll,
            f"{candidate_name}_mean_nll": candidate.mean_nll,
            "mean_nll_delta_candidate_minus_reference": (
                candidate.mean_nll - reference.mean_nll
            ),
            "reference_to_candidate_kl_mean": statistics.fmean(prompt_forward),
            "candidate_to_reference_kl_mean": statistics.fmean(prompt_reverse),
        })
    reference_nll = -statistics.fmean(reference_targets)
    candidate_nll = -statistics.fmean(candidate_targets)
    reference_ppl = math.exp(reference_nll)
    candidate_ppl = math.exp(candidate_nll)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "comparison_backend_contract": "Transformers reference vs vLLM candidate",
        "kl_mode": kl_mode,
        "kl_convention": _kl_convention(kl_mode),
        "tokens_scored": len(reference_targets),
        "reference_mean_nll": reference_nll,
        "candidate_mean_nll": candidate_nll,
        "mean_nll_delta_candidate_minus_reference": candidate_nll - reference_nll,
        "reference_ppl": reference_ppl,
        "candidate_ppl": candidate_ppl,
        "ppl_candidate_over_reference": candidate_ppl / reference_ppl,
        "target_logprob_abs_delta": summarize_values(target_abs_delta),
        "kl_reference_to_candidate": summarize_values(forward_kl),
        "kl_candidate_to_reference": summarize_values(reverse_kl),
        "per_prompt": per_prompt,
    }


def _timing_summary(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    arms = {arm: summarize_values(samples[arm]) for arm in ARMS}
    baseline_mean = arms["baseline"]["mean"]
    fused_mean = arms["fused"]["mean"]
    speedup = (
        float(baseline_mean) / float(fused_mean)
        if baseline_mean is not None and fused_mean not in (None, 0.0)
        else None
    )
    return {
        "metric": "offline_generate_one_token_wall_ms",
        "scope": "scheduler + prefill + logits + one-token output processing",
        "is_streaming_ttft": False,
        "arms": arms,
        "baseline_over_fused_speedup": speedup,
    }


def _configured_gates(args: argparse.Namespace, report: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    quality = report["quality"]["delta"]
    if args.max_mean_kl is not None:
        observed = quality["kl_baseline_to_fused"]["mean"]
        gates["max_mean_kl"] = {
            "limit": args.max_mean_kl,
            "observed": observed,
            "pass": observed is not None and observed <= args.max_mean_kl,
        }
    if args.max_mean_nll_regression is not None:
        observed = quality["mean_nll_fused_minus_baseline"]
        gates["max_mean_nll_regression"] = {
            "limit": args.max_mean_nll_regression,
            "observed": observed,
            "pass": observed <= args.max_mean_nll_regression,
        }
    if args.max_ppl_relative_regression is not None:
        observed = quality["ppl_relative_regression"]
        gates["max_ppl_relative_regression"] = {
            "limit": args.max_ppl_relative_regression,
            "observed": observed,
            "pass": observed <= args.max_ppl_relative_regression,
        }
    if args.min_timing_speedup is not None:
        observed = report.get("timing", {}).get("baseline_over_fused_speedup")
        gates["min_timing_speedup"] = {
            "limit": args.min_timing_speedup,
            "observed": observed,
            "pass": observed is not None and observed >= args.min_timing_speedup,
        }
    teacher_quality = report.get("teacher_quality")
    teacher_fused = (
        teacher_quality.get("fused")
        if isinstance(teacher_quality, Mapping)
        else None
    )
    teacher_baseline = (
        teacher_quality.get("baseline")
        if isinstance(teacher_quality, Mapping)
        else None
    )
    teacher_kl_limit = getattr(args, "max_teacher_fused_mean_kl", None)
    if teacher_kl_limit is not None:
        observed = (
            teacher_fused["kl_reference_to_candidate"]["mean"]
            if teacher_fused is not None
            else None
        )
        gates["max_teacher_fused_mean_kl"] = {
            "limit": teacher_kl_limit,
            "observed": observed,
            "kl_mode": teacher_fused.get("kl_mode") if teacher_fused else None,
            "pass": observed is not None and observed <= teacher_kl_limit,
        }
    teacher_regression_limit = getattr(
        args, "max_teacher_fused_kl_regression", None
    )
    if teacher_regression_limit is not None:
        fused_kl = (
            teacher_fused["kl_reference_to_candidate"]["mean"]
            if teacher_fused is not None
            else None
        )
        baseline_kl = (
            teacher_baseline["kl_reference_to_candidate"]["mean"]
            if teacher_baseline is not None
            else None
        )
        observed = (
            fused_kl - baseline_kl
            if fused_kl is not None and baseline_kl is not None
            else None
        )
        gates["max_teacher_fused_kl_regression"] = {
            "limit": teacher_regression_limit,
            "observed_fused_minus_baseline": observed,
            "teacher_to_fused_mean_kl": fused_kl,
            "teacher_to_baseline_mean_kl": baseline_kl,
            "kl_mode": teacher_fused.get("kl_mode") if teacher_fused else None,
            "pass": observed is not None and observed <= teacher_regression_limit,
        }
    return gates


def _configured_limit_values(args: argparse.Namespace) -> tuple[float | None, ...]:
    """Return every optional threshold that turns a measurement into a screen."""

    return tuple(
        getattr(args, name, None)
        for name in (
            "max_mean_kl",
            "max_mean_nll_regression",
            "max_ppl_relative_regression",
            "max_teacher_fused_mean_kl",
            "max_teacher_fused_kl_regression",
            "min_timing_speedup",
        )
    )


def _finalize_gate_report(
    args: argparse.Namespace,
    report: dict[str, Any],
    core_gates: Mapping[str, Mapping[str, Any]],
) -> None:
    """Attach integrity, screening, and promotion-contract outcomes in place."""

    configured_gates = _configured_gates(args, report)
    report["configured_promotion_gates"] = configured_gates
    report["measurement_valid"] = all(gate["pass"] for gate in core_gates.values())
    measurement_only = bool(getattr(args, "measurement_only", False))
    report["measurement_only"] = measurement_only
    report["configured_gates_pass"] = (
        None
        if measurement_only
        else all(gate["pass"] for gate in configured_gates.values())
    )
    full_vocab_teacher_requested = bool(args.teacher_full_vocab_kl)
    teacher_quality = report.get("teacher_quality")
    teacher_rows = (
        [teacher_quality.get(arm) for arm in ARMS]
        if isinstance(teacher_quality, Mapping)
        else []
    )
    full_vocab_teacher_observed = bool(teacher_rows) and all(
        isinstance(row, Mapping) and row.get("kl_mode") == KL_FULL_VOCAB
        for row in teacher_rows
    )
    full_vocab_teacher_present = (
        full_vocab_teacher_requested and full_vocab_teacher_observed
    )
    teacher_quality_thresholds_present = [
        name
        for name in _TEACHER_QUALITY_PROMOTION_GATES
        if getattr(args, name, None) is not None
    ]
    teacher_quality_thresholds_missing = [
        name
        for name in _TEACHER_QUALITY_PROMOTION_GATES
        if name not in teacher_quality_thresholds_present
    ]
    promotion_contract_complete = (
        full_vocab_teacher_present and not teacher_quality_thresholds_missing
    )
    report["promotion_contract"] = {
        "teacher_full_vocab_kl_required": True,
        "teacher_full_vocab_kl_requested": full_vocab_teacher_requested,
        "teacher_full_vocab_kl_observed": full_vocab_teacher_observed,
        "teacher_full_vocab_kl_present": full_vocab_teacher_present,
        "teacher_quality_thresholds_required": list(
            _TEACHER_QUALITY_PROMOTION_GATES
        ),
        "teacher_quality_thresholds_present": teacher_quality_thresholds_present,
        "teacher_quality_thresholds_missing": teacher_quality_thresholds_missing,
        "complete": promotion_contract_complete,
        "coarse_kl_may_reject_but_cannot_greenlight": True,
    }
    report["promotion_recommendation"] = (
        "measurement_failed"
        if not report["measurement_valid"]
        else "measurement_only_no_promotion_thresholds_configured"
        if measurement_only
        else "configured_gates_failed"
        if report["configured_gates_pass"] is False
        else "screening_only_full_vocab_teacher_required"
        if not full_vocab_teacher_present
        else "screening_only_teacher_quality_thresholds_required"
        if teacher_quality_thresholds_missing
        else "candidate_only_requires_served_validation"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if "vllm" in sys.modules:
        raise RuntimeError(
            "vLLM was imported before the harness could force same-process "
            f"execution; launch {Path(__file__).name} in a fresh Python process"
        )
    os.environ[VLLM_MP_ENV] = "0"
    os.environ[FUSED_ENV] = ""  # Model initialization starts in baseline mode.
    os.environ[FUSED_MOE_ENV] = ""
    inherited_prefill = os.environ.get(PREFILL_ENV)
    if args.mode != "dense":
        # Unset is the production FP4 policy entry point. Setting "loop" would
        # bypass the candidate before it can be measured.
        os.environ.pop(PREFILL_ENV, None)

    started = time.monotonic()
    bootstrap = validation_common.prepare_validation(
        args,
        harness_path=Path(__file__),
        helpers=sys.modules[__name__],
        extension_none_message=(
            "fused FP4 extension did not build/load; refusing a fallback A/B"
        ),
    )
    torch = bootstrap.torch
    linear = bootstrap.linear
    moe = bootstrap.moe
    runtime = bootstrap.runtime
    extension = bootstrap.extension
    candidate_vocab_size = bootstrap.candidate_vocab_size
    candidate_artifact_provenance = bootstrap.candidate_artifact_provenance
    prompts = bootstrap.prompts
    dataset = bootstrap.dataset
    quality_kl_mode = bootstrap.quality_kl_mode

    probe = (
        DenseDispatchProbe(
            linear.PrismaQuantCBLinearMethod,
            prefill_threshold=linear.PREFILL_M_THRESHOLD,
            fused_mode=args.fused_mode,
        )
        if args.mode == "dense"
        else MoEDispatchProbe(moe.PrismaQuantCBMoEMethod)
    )
    probe.install()
    engine = validation_common.load_candidate_engine(
        bootstrap, args, probe=probe
    )
    llm = engine.llm
    quality_sampling = engine.quality_sampling
    timing_sampling = engine.timing_sampling
    model_load_s = engine.model_load_seconds

    dispatch_records: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    quality_pairs: list[dict[str, Any]] = []
    timing_samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    synchronize = torch.cuda.synchronize

    try:
        # Both kernels and per-layer fused LUTs are resident before measured
        # requests. Prefix caching is disabled, so warmup cannot bypass a later
        # arm's prefill.
        for repeat in range(args.warmup_pairs):
            for arm in paired_arm_order(repeat):
                _result, elapsed_ms, record = _run_generate(
                    llm=llm,
                    sampling=timing_sampling,
                    prompt_ids=prompts[repeat % len(prompts)],
                    arm=arm,
                    label=f"warmup:{repeat}:{arm}",
                    execution_mode=args.mode,
                    dense_fused_mode=args.fused_mode,
                    linear=linear,
                    moe=moe,
                    probe=probe,
                    synchronize=synchronize,
                )
                record["wall_ms"] = elapsed_ms
                warmup_records.append(record)
                del _result

        # Exact-vocabulary quality materializes very large host objects. Run
        # timing first so deferred GC/lazy host work cannot land on one arm.
        for phase in validation_common.measurement_phase_order(
            args.timing_repeats
        ):
            if phase == "timing":
                validation_common.quiesce_before_timing(torch)
                for repeat in range(args.timing_repeats):
                    for prompt_index, prompt_ids in enumerate(prompts):
                        order = paired_arm_order(
                            repeat * len(prompts) + prompt_index
                        )
                        for arm in order:
                            _output, elapsed_ms, record = _run_generate(
                                llm=llm,
                                sampling=timing_sampling,
                                prompt_ids=prompt_ids,
                                arm=arm,
                                label=(
                                    f"timing:{repeat}:{prompt_index}:{arm}"
                                ),
                                execution_mode=args.mode,
                                dense_fused_mode=args.fused_mode,
                                linear=linear,
                                moe=moe,
                                probe=probe,
                                synchronize=synchronize,
                            )
                            timing_samples[arm].append(elapsed_ms)
                            record["wall_ms"] = elapsed_ms
                            dispatch_records.append(record)
                            del _output
                continue
            if phase != "quality":
                raise RuntimeError(f"unknown measurement phase {phase!r}")
            for prompt_index, prompt_ids in enumerate(prompts):
                scores: dict[str, PromptScore] = {}
                walls: dict[str, float] = {}
                order = paired_arm_order(prompt_index)
                for arm in order:
                    output, elapsed_ms, record = _run_generate(
                        llm=llm,
                        sampling=quality_sampling,
                        prompt_ids=prompt_ids,
                        arm=arm,
                        label=f"quality:{prompt_index}:{arm}",
                        execution_mode=args.mode,
                        dense_fused_mode=args.fused_mode,
                        linear=linear,
                        moe=moe,
                        probe=probe,
                        synchronize=synchronize,
                    )
                    scores[arm] = score_prompt_output(
                        output,
                        prompt_ids,
                        args.top_k,
                        full_vocab=args.teacher_full_vocab_kl,
                        expected_vocab_size=(
                            candidate_vocab_size
                            if args.teacher_full_vocab_kl
                            else None
                        ),
                    )
                    walls[arm] = elapsed_ms
                    record["wall_ms"] = elapsed_ms
                    dispatch_records.append(record)
                    del output
                quality_pairs.append({
                    "prompt_index": prompt_index,
                    "pair_order": list(order),
                    "scores": scores,
                    "wall_ms": walls,
                })
    finally:
        probe.restore()

    quality = _quality_summary(quality_pairs, kl_mode=quality_kl_mode)
    arm_scores = {
        arm: [pair["scores"][arm] for pair in quality_pairs]
        for arm in ARMS
    }
    teacher_quality, teacher_record = validation_common.score_teacher(
        bootstrap,
        args,
        arm_scores=arm_scores,
        arms=ARMS,
        helpers=sys.modules[__name__],
    )
    if teacher_quality is not None:
        teacher_baseline_kl = teacher_quality["baseline"][
            "kl_reference_to_candidate"
        ]["mean"]
        teacher_fused_kl = teacher_quality["fused"][
            "kl_reference_to_candidate"
        ]["mean"]
        teacher_quality["delta"] = {
            "teacher_to_candidate_mean_kl_fused_minus_baseline": (
                teacher_fused_kl - teacher_baseline_kl
            ),
            "kl_mode": quality_kl_mode,
        }
    baseline_records = [r for r in dispatch_records if r["arm"] == "baseline"]
    fused_records = [r for r in dispatch_records if r["arm"] == "fused"]
    baseline_dispatch = aggregate_dispatch(baseline_records)
    fused_dispatch = aggregate_dispatch(fused_records)
    observed_pids = sorted(set(baseline_dispatch["pids"] + fused_dispatch["pids"]))
    core_gates = {
        "same_process_dispatch": {
            "expected_pid": os.getpid(),
            "observed_pids": observed_pids,
            "pass": observed_pids == [os.getpid()],
        },
        "baseline_never_entered_fused_dispatch": {
            "observed_attempts": baseline_dispatch["fused_attempts"],
            "pass": baseline_dispatch["fused_attempts"] == 0,
        },
        "candidate_positive_fused_dispatch": {
            "observed_successes": fused_dispatch["fused_successes"],
            "pass": fused_dispatch["fused_successes"] > 0,
        },
        "candidate_fused_dispatch_no_errors": {
            "observed_errors": fused_dispatch["fused_errors"],
            "pass": fused_dispatch["fused_errors"] == 0,
        },
        "probe_no_errors": {
            "observed_errors": (
                baseline_dispatch["probe_errors"] + fused_dispatch["probe_errors"]
            ),
            "pass": (
                baseline_dispatch["probe_errors"] + fused_dispatch["probe_errors"]
            ) == 0,
        },
    }
    if args.mode != "dense":
        core_gates.update({
            "baseline_positive_loop_dispatch": {
                "observed_loop_calls": baseline_dispatch["loop_calls"],
                "pass": baseline_dispatch["loop_calls"] > 0,
            },
            "baseline_loop_no_errors": {
                "observed_errors": baseline_dispatch["loop_errors"],
                "pass": baseline_dispatch["loop_errors"] == 0,
            },
            "candidate_zero_fallbacks": {
                "observed_fallbacks": fused_dispatch["fused_fallbacks"],
                "pass": fused_dispatch["fused_fallbacks"] == 0,
            },
            "candidate_never_entered_loop": {
                "observed_loop_calls": fused_dispatch["loop_calls"],
                "pass": fused_dispatch["loop_calls"] == 0,
            },
        })
    else:
        opportunities = fused_dispatch["candidate_gate_opportunities"]
        attempts = fused_dispatch["fused_attempts"]
        success_fraction = fused_dispatch["fused_success_fraction"]
        core_gates["candidate_attempted_every_dispatch_opportunity"] = {
            "observed_opportunities": opportunities,
            "observed_attempts": attempts,
            "pass": opportunities > 0 and attempts == opportunities,
        }
        core_gates["candidate_minimum_fused_coverage"] = {
            "minimum_fraction": args.min_fused_success_fraction,
            "observed_fraction": success_fraction,
            "observed_successes": fused_dispatch["fused_successes"],
            "observed_fallbacks": fused_dispatch["fused_fallbacks"],
            "pass": (
                success_fraction is not None
                and success_fraction >= args.min_fused_success_fraction
            ),
        }
        if args.require_zero_fallbacks:
            core_gates["candidate_zero_fallbacks"] = {
                "observed_fallbacks": fused_dispatch["fused_fallbacks"],
                "pass": fused_dispatch["fused_fallbacks"] == 0,
            }

    kl_limitation = (
        "Exact full-vocabulary KL was requested and cardinality-attested; it "
        "materializes every vocabulary logprob and is intentionally opt-in."
        if args.teacher_full_vocab_kl
        else "Top-K KL is a paired coarse-support regression metric, not "
        "full-vocabulary KL; it may reject a candidate but cannot make this "
        "report promotion-eligible."
    )
    limitations = [
        kl_limitation,
        "Optional timing is offline one-token request wall time, not served TTFT.",
        (
            "Dense mode does not validate grouped-MoE routing or padding cliffs."
            if args.mode == "dense"
            else "One selected TileM does not cover the full routed-token/padding ladder."
        ),
        "A passing run does not replace served KL/PPL/tasks and workload-matrix gates.",
    ]
    if args.teacher_model is not None:
        limitations.append(
            "Teacher comparisons cross Transformers and vLLM runtimes; absolute "
            "deltas therefore include backend implementation numerics."
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "scope": (
            f"{args.mode} NVFP4-CB prefill; TP=1; one in-process vLLM engine"
        ),
        "activation_contract": {
            "baseline": "fp32-emulated group-scale FP4 activation QDQ",
            "fused": "native NVFP4 global FP32 scale + per-group UE4M3 factors",
            "same_contract": False,
        },
        "settings": validation_common.shared_report_settings(
            bootstrap,
            args,
            arm_settings={
                "fused_mode": args.fused_mode,
                "moe_fused_tile_m": (
                    int(args.mode.removeprefix("moe"))
                    if args.mode != "dense"
                    else None
                ),
            },
            inherited_prefill=inherited_prefill,
            prefill_threshold=linear.PREFILL_M_THRESHOLD,
            measurement_settings={
                "warmup_pairs": args.warmup_pairs,
                "timing_repeats": args.timing_repeats,
                "measurement_only": bool(
                    getattr(args, "measurement_only", False)
                ),
                "min_fused_success_fraction": (
                    args.min_fused_success_fraction
                ),
            },
        ),
        "runtime": runtime,
        "extension": extension,
        "candidate_artifact_provenance": candidate_artifact_provenance,
        "dataset": dataset,
        "model_load_seconds": model_load_s,
        "warmup_dispatch": warmup_records,
        "quality": quality,
        "teacher": teacher_record,
        "teacher_quality": teacher_quality,
        "dispatch": {
            "baseline": baseline_dispatch,
            "fused": fused_dispatch,
            "per_request": dispatch_records,
            "unscoped": _json_dispatch_record(probe.unscoped),
        },
        "core_integrity_gates": core_gates,
        "limitations": limitations,
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.timing_repeats:
        report["timing"] = _timing_summary(timing_samples)
    _finalize_gate_report(args, report, core_gates)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation_common.add_shared_cli_arguments(
        parser, helpers=sys.modules[__name__], n_samples_default=4
    )
    parser.add_argument(
        "--fused-mode", choices=DENSE_FUSED_MODES, default="1",
        help=(
            "dense activation family/range; MoE modes select their TileM "
            "by --mode"
        ),
    )
    parser.add_argument("--warmup-pairs", type=_positive_int, default=1)
    validation_common.add_shared_measurement_cli_arguments(
        parser, helpers=sys.modules[__name__]
    )
    parser.add_argument(
        "--require-zero-fallbacks",
        action="store_true",
        help="make any candidate _try_fused_fp4 -> None result an integrity failure",
    )
    parser.add_argument(
        "--min-fused-success-fraction",
        type=_closed_unit_interval,
        default=1.0,
        help=(
            "minimum dense candidate dispatch coverage; defaults fail-closed "
            "at 1.0, and any lower promotion threshold must be explicit"
        ),
    )
    parser.add_argument("--max-mean-kl", type=_nonnegative_float)
    parser.add_argument("--max-mean-nll-regression", type=_nonnegative_float)
    parser.add_argument("--max-ppl-relative-regression", type=_nonnegative_float)
    parser.add_argument(
        "--max-teacher-fused-mean-kl",
        type=_nonnegative_float,
        help=(
            "maximum absolute mean KL(teacher||fused); promotion eligibility "
            "requires this limit, --max-teacher-fused-kl-regression, and "
            "--teacher-full-vocab-kl"
        ),
    )
    parser.add_argument(
        "--max-teacher-fused-kl-regression",
        type=_nonnegative_float,
        help=(
            "maximum allowed mean KL(teacher||fused) minus mean "
            "KL(teacher||baseline); coarse top-K mode is a rejection screen "
            "only, and promotion eligibility also requires "
            "--max-teacher-fused-mean-kl and --teacher-full-vocab-kl"
        ),
    )
    parser.add_argument("--min-timing-speedup", type=_nonnegative_float)
    args = parser.parse_args(argv)
    teacher_gates = (
        args.max_teacher_fused_mean_kl,
        args.max_teacher_fused_kl_regression,
    )
    validation_common.validate_shared_cli_args(
        parser, args, teacher_gate_values=teacher_gates
    )
    has_thresholds = any(limit is not None for limit in _configured_limit_values(args))
    if not has_thresholds and not args.measurement_only:
        parser.error(
            "no promotion thresholds were configured; pass --measurement-only "
            "to collect non-promotional evidence explicitly"
        )
    if has_thresholds and args.measurement_only:
        parser.error("--measurement-only cannot be combined with promotion thresholds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure
        failure = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1
    if not report["measurement_valid"] or report["configured_gates_pass"] is False:
        report["status"] = "gate_failed"
    elif report["measurement_only"]:
        report["status"] = "measurement_only"
    elif not report["promotion_contract"]["complete"]:
        report["status"] = "screening_only"
    else:
        report["status"] = "ok"
    _atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_contract": report["promotion_contract"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_delta": report["quality"]["delta"],
        "timing": report.get("timing"),
        "dispatch": {
            "baseline": report["dispatch"]["baseline"],
            "fused": report["dispatch"]["fused"],
        },
    }, indent=2), flush=True)
    return 0 if report["status"] in ("ok", "measurement_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
