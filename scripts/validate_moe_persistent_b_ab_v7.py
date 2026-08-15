#!/usr/bin/env python3
"""Same-engine quality A/B for the persistent-B routed FP4-CB lane.

This is the whole-model quality gate for the opt-in
``PRISMAQUANT_CB_MOE_PERSISTENT_B`` schedule.  It loads the extension and the
candidate exactly once with persistent-B resolved for every compatible FP4-CB
MoE layer, then evaluates fixed, pre-tokenized WikiText windows in consecutive,
warmed baseline--persistent-B--baseline (B--PB--B) prompt blocks:

* ``baseline`` is Gridbook's default exact CB-to-BF16 expansion plus owned
  grouped-BF16 bridge; and
* ``fused`` is persistent-B's decode-in-mainloop schedule.

The two baseline observations bracket the candidate in the same PID.  Their
mean removes linear request-order drift from each prompt's NLL effect, while
their difference measures whether the sealed eight-prompt experiment can
resolve the promotion limit at all.  Both arms consume the same packed weights
and exact native group-16 activation QDQ; only the FP32 GEMM reduction order
changes.

The model is never reloaded.  The harness keeps the production selector at
``1`` and temporarily swaps only the already-attested per-layer extension
handle.  Before every request it clears latest-route telemetry, then requires
all loaded FP4-CB MoE layers to report the requested symbol and every loaded
FP8-CB MoE layer to report a served route.  FP8 routes must be byte-for-byte
identical within each triplet.  Layer counts and the DeepSeek-V4
architecture contract are machine gates, not log-inspection suggestions.

Target-token NLL/PPL are exact on all eight sealed 512-token windows.  Exact
symmetric full-vocabulary KL, ``0.5 * (KL(P||Q) + KL(Q||P))`` (half the
Jeffreys divergence), first runs on the digest-attested 64-token prefix of
every window as a rejection screen.  A candidate not rejected there must then
complete the same measurement over all 512 tokens of every sealed window;
only that confirmation can promote the lane.  vLLM's CPU ``LogprobsTensors``
are retained in a bounded transport, then compacted and cardinality/finiteness
checked one arm at a time; only prompt-level metrics and row/score digests
survive each triplet.  Token rows are never treated as independent
observations: the fixed eight prompts are the statistical units.

The v6 promotion decision is fixed, rather than operator-configurable.  It
uses one-sided 95% Student-t confidence bounds for prompt-block NLL and one
predeclared hybrid baseline-noise-relative symmetric-KL excess, a fail-closed
baseline-resolution check, and a per-prompt corruption backstop.  Integrity,
confidence-bounded harm, or corruption is FAIL; an unresolved interval is
INCONCLUSIVE; only every gate passing on the complete confirmation profile is
PROMOTION PASS.  An explicit campaign id and atomically reserved output bind
the result to one immutable attempt; a prior report is never overwritten.

Run this script in a fresh Python process inside the pinned Gridbook/vLLM CUDA
image.  Example for the exact dsv4flash artifact::

    python3 scripts/validate_moe_persistent_b_ab.py \
      --model /models/dsv4flash0731 \
      --prompt-token-ids-json /evidence/dsv4-wikitext-inputs-v1.json \
      --output /evidence/dsv4-persistent-b-ab.json \
      --campaign-id persistent-b-v6-20260813 \
      --enable-chunked-prefill --max-num-batched-tokens 256 \
      --top-k 256 --full-vocab-kl

A passing report remains candidate evidence; served TTFT/ITL/TPS and memory
headroom are separate NATIVE-PARITY gates.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import statistics
import struct
import sys
import time
import traceback
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _load_v5_helpers() -> Any:
    path = Path(__file__).with_name("validate_fused_nvfp4_ab.py")
    module_name = "_gridbook_validate_fused_nvfp4_ab_for_persistent_b"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load v5 validation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v5 = _load_v5_helpers()
validation_common = v5.validation_common

SCHEMA = "gridbook.moe-persistent-b-ab.v2"
ARMS = v5.ARMS
DSV4_KV_CACHE_DTYPE = "fp8"
ARM_LABELS = {
    "baseline": "expand_plus_grouped_bf16_bridge",
    "fused": "persistent_b_decode_in_mainloop",
}
PERSISTENT_B_ENV = "PRISMAQUANT_CB_MOE_PERSISTENT_B"
PERSISTENT_B_CFG_ENV = "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG"
BF16_SM120_ENV = "PRISMAQUANT_CB_BF16_SM120"
_FULL_KL_SCHEMA = "prismaquant.dsv4_wikitext_inputs/1"
_MAX_INPUT_BYTES = 1_048_576
_SHA256_CHARS = frozenset("0123456789abcdef")
# Closed release workload copied from PrismaQuant's producer-owned
# ``tools/dsv4_wikitext_inputs.py``. The self-digest in the JSON protects a
# payload against accidental damage; these constants protect the gate against
# an operator re-sealing a different, easier corpus under the same schema.
_CANONICAL_INPUT_SEMANTIC_SHA256 = (
    "3eedb1a879e8e9cf13a2a0f16d3fd01a0817d18de60a2dd743c9c4fc44a34680"
)
_CANONICAL_DATASETS_DISTRIBUTION = {
    "name": "datasets",
    "version": "4.6.0",
}
_CANONICAL_CORPUS_CONSTRUCTION = {
    "row_filter": "include iff bool(text.strip()); preserve text verbatim",
    "join_separator": "\n\n",
    "normalization": "none",
}
_CANONICAL_TOKENIZER_CONTENT_SHA256 = (
    "9f7ee7cb93b58bf30f278965547e7584b89c848e76c3adfeb92c070a88492de0"
)
_CANONICAL_TOKENIZER_VOCAB_SIZE = 129_280
_CANONICAL_WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
_CANONICAL_FULL_KL_DATASET = {
    "name": "wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "train",
    "revision": _CANONICAL_WIKITEXT_REVISION,
    "fingerprint": "7c4dea6941cc4a0a",
    "corpus_sha256": (
        "fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047"
    ),
    "total_tokens": 2_423_186,
}
_CANONICAL_FULL_KL_SELECTION = {
    "sampler": (
        "python.random.Random(seed).sample(range(max_start), n_samples)/v1"
    ),
    "window_seed": 42,
    "n_samples": 8,
    "seqlen": 512,
    "starts": [
        466_956,
        104_902,
        1_153_556,
        1_027_150,
        936_213,
        585_264,
        429_895,
        2_287_433,
    ],
}
_CANONICAL_FULL_KL_TOKEN_IDS_TENSOR_SHA256 = (
    "b3426e9bab87a1c444b04d0ce01fa9cba5ace313b91db2c3f77fc3525e732b22"
)
_CANONICAL_PPL_DATASET = {
    "name": "wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "test",
    "revision": _CANONICAL_WIKITEXT_REVISION,
    "fingerprint": "7ccd6deaa4fc56e5",
    "corpus_sha256": (
        "c5b5caea5bd655cb221545a484f2f0f59d35092a17a66840d7b9513d0b99687d"
    ),
    "total_tokens": 287_597,
}
_CANONICAL_PPL_SELECTION = {
    "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
    "n_tokens": 8_192,
}
_CANONICAL_PPL_TOKEN_IDS_SHA256 = (
    "6c23cefbd78c327d6edac566a5c6b419871021b6cf9890ec830713c1de704961"
)
_PROMOTION_PROTOCOL = "persistent-b-promotion-quality-v6"
_ATTEMPT_SCHEMA = "gridbook.persistent-b-validation-attempt.v1"
_SEALED_PROMPT_COUNT = 8
_SEALED_PROMPT_SEQLEN = 512
_EXACT_PILOT_SEQLEN = 64
_EXACT_CONFIRMATION_SEQLEN = 512
_ONE_SIDED_T_95_DF7 = 1.894578605061305
_NLL_LIMIT = math.log(1.005)
_CORRUPTION_LIMIT = math.log(1.01)
_J_ABSOLUTE_MARGIN = 1.0e-4
_J_NOISE_MULTIPLIER = 1.25
_CAMPAIGN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}\Z")
_TRIPLET_REQUESTS = (
    ("baseline_pre", "baseline"),
    ("candidate", "fused"),
    ("baseline_post", "baseline"),
)
_TRIPLET_ROLES = tuple(role for role, _arm in _TRIPLET_REQUESTS)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _campaign_id_arg(value: str) -> str:
    if not _CAMPAIGN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "campaign id must be 8..128 characters, start with an ASCII "
            "letter or digit, and contain only letters, digits, '.', '_', or '-'"
        )
    return value


@dataclass(frozen=True)
class _AttemptReservation:
    output: Path
    claim_path: Path
    attempt: Mapping[str, Any]
    claim_payload: Mapping[str, Any]
    output_payload: Mapping[str, Any]


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one durable JSON file without replacing any prior evidence."""

    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def _read_attempt_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable or invalid: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _reserve_attempt(output: Path, campaign_id: str) -> _AttemptReservation:
    """Atomically bind one campaign id to one output before loading the model.

    The campaign claim and output reservation deliberately survive a crash.  A
    crashed or inconclusive attempt therefore cannot be erased and retried
    under the same campaign identity without an explicit new protocol/campaign.
    """

    campaign_id = _campaign_id_arg(campaign_id)
    requested = output.expanduser()
    if not requested.name or requested.name in (".", ".."):
        raise ValueError("attempt output must name a file")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    resolved_output = parent / requested.name
    if resolved_output.exists() or resolved_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite prior persistent-B evidence: {resolved_output}"
        )

    campaign_sha256 = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()
    claim_path = parent / f".persistent-b-campaign-{campaign_sha256}.claim.json"
    reserved_at = v5._utc_now()
    attempt = {
        "schema": _ATTEMPT_SCHEMA,
        "protocol": _PROMOTION_PROTOCOL,
        "campaign_id": campaign_id,
        "campaign_id_sha256": campaign_sha256,
        "attempt_id": secrets.token_hex(16),
        "reserved_at": reserved_at,
        "pid": os.getpid(),
        "output": str(resolved_output),
        "claim_path": str(claim_path),
    }
    claim_payload = {
        "schema": _ATTEMPT_SCHEMA,
        "status": "campaign_claimed",
        "attempt": attempt,
    }
    output_payload = {
        "schema": SCHEMA,
        "created_at": reserved_at,
        "status": "attempt_reserved",
        "promotion_decision": "INCONCLUSIVE",
        "decision_reason": "attempt_in_progress_or_interrupted",
        "attempt": attempt,
    }

    # Claim first.  If a concurrent output creation wins after the precheck,
    # the surviving claim intentionally burns the campaign rather than making
    # a retry look like its first attempt.
    _exclusive_json(claim_path, claim_payload)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        _exclusive_json(resolved_output, output_payload)
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # The immutable claim is retained as evidence of the attempted run.
        raise
    return _AttemptReservation(
        output=resolved_output,
        claim_path=claim_path,
        attempt=attempt,
        claim_payload=claim_payload,
        output_payload=output_payload,
    )


def _finalize_attempt(
    reservation: _AttemptReservation, payload: Mapping[str, Any]
) -> None:
    """Replace only this process's live reservation, exactly once."""

    observed_claim = _read_attempt_json(
        reservation.claim_path, label="campaign claim"
    )
    if observed_claim != reservation.claim_payload:
        raise RuntimeError("campaign claim changed after reservation; refusing write")
    observed_output = _read_attempt_json(
        reservation.output, label="attempt output reservation"
    )
    if observed_output != reservation.output_payload:
        raise RuntimeError(
            "attempt output is no longer the live reservation; refusing overwrite"
        )
    final_payload = dict(payload)
    final_payload["attempt"] = dict(reservation.attempt)
    final_payload["attempt_immutability"] = {
        "single_attempt_per_campaign_in_evidence_directory": True,
        "prior_output_overwrite_refused": True,
        "campaign_claim_retained": True,
        "claim_path": str(reservation.claim_path),
        "claim_sha256": hashlib.sha256(
            _canonical_json_bytes(reservation.claim_payload)
        ).hexdigest(),
    }
    raw = json.dumps(
        final_payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = reservation.output.with_name(
        f".{reservation.output.name}.{reservation.attempt['attempt_id']}.final.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # The compare above proves ownership under the cooperative single-
        # harness protocol.  ``replace`` publishes the complete JSON in one
        # namespace operation; the directory fsync makes that transition
        # durable across a host crash.
        os.replace(temporary, reservation.output)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(reservation.output.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _strict_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = path.expanduser().resolve(strict=True)
    size = source.stat().st_size
    if not 0 < size <= _MAX_INPUT_BYTES:
        raise RuntimeError(
            f"fixed prompt payload must be 1..{_MAX_INPUT_BYTES} bytes, got {size}"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid fixed prompt payload {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("fixed prompt payload must contain a JSON object")
    return payload, {
        "path": str(source),
        "bytes": size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _fixed_prompt_loader(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[list[list[int]], dict[str, Any]]:
    """Read and attest the producer-sealed DSV4 full-KL token windows."""

    payload, file_record = _strict_json(args.prompt_token_ids_json)
    expected_top = {
        "schema", "datasets_distribution", "corpus_construction",
        "tokenizer", "full_kl", "ppl", "semantic_sha256",
    }
    if set(payload) != expected_top or payload.get("schema") != _FULL_KL_SCHEMA:
        raise RuntimeError(
            f"fixed prompt payload must be closed schema {_FULL_KL_SCHEMA!r}"
        )
    unsigned = {key: value for key, value in payload.items()
                if key != "semantic_sha256"}
    if (
        not _valid_sha256(payload.get("semantic_sha256"))
        or payload["semantic_sha256"] != _canonical_sha256(unsigned)
    ):
        raise RuntimeError("fixed prompt payload semantic digest differs")
    if payload["semantic_sha256"] != _CANONICAL_INPUT_SEMANTIC_SHA256:
        raise RuntimeError(
            "fixed prompt payload is not the canonical DSV4 WikiText release "
            "workload"
        )
    if payload.get("datasets_distribution") != _CANONICAL_DATASETS_DISTRIBUTION:
        raise RuntimeError("fixed prompt datasets producer identity differs")
    if payload.get("corpus_construction") != _CANONICAL_CORPUS_CONSTRUCTION:
        raise RuntimeError("fixed prompt corpus construction differs")

    tokenizer_record = payload.get("tokenizer")
    if (
        not isinstance(tokenizer_record, Mapping)
        or set(tokenizer_record) != {"schema", "content_sha256", "files"}
        or tokenizer_record.get("schema") != "prismaquant.tokenizer_identity/1"
        or not isinstance(tokenizer_record.get("files"), Mapping)
        or not tokenizer_record["files"]
    ):
        raise RuntimeError("fixed prompt tokenizer identity is malformed")
    files = dict(tokenizer_record["files"])
    if tokenizer_record.get("content_sha256") != _canonical_sha256({"files": files}):
        raise RuntimeError("fixed prompt tokenizer content digest differs")
    if (
        tokenizer_record.get("content_sha256")
        != _CANONICAL_TOKENIZER_CONTENT_SHA256
    ):
        raise RuntimeError("fixed prompt tokenizer value identity differs")
    model_root = Path(args.model).expanduser()
    if not model_root.is_dir():
        raise RuntimeError(
            "fixed DSV4 token inputs require a local --model so tokenizer "
            "files can be content-attested"
        )
    observed_files: dict[str, dict[str, Any]] = {}
    for name, descriptor in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"bytes", "sha256"}
            or isinstance(descriptor.get("bytes"), bool)
            or not isinstance(descriptor.get("bytes"), int)
            or int(descriptor["bytes"]) <= 0
            or not _valid_sha256(descriptor.get("sha256"))
        ):
            raise RuntimeError(f"malformed tokenizer descriptor {name!r}")
        candidate = model_root / name
        if not candidate.is_file():
            raise RuntimeError(f"candidate tokenizer file is missing: {candidate}")
        record = v5._required_file_record(candidate)
        observed_files[name] = record
        if record != dict(descriptor):
            raise RuntimeError(
                f"candidate tokenizer file differs from fixed inputs: {name}"
            )

    full_kl = payload.get("full_kl")
    if (
        not isinstance(full_kl, Mapping)
        or set(full_kl) != {
            "dataset", "selection", "token_ids", "token_ids_tensor_sha256"
        }
        or not isinstance(full_kl.get("selection"), Mapping)
        or not isinstance(full_kl.get("token_ids"), list)
    ):
        raise RuntimeError("fixed full-KL prompt section is malformed")
    selection = dict(full_kl["selection"])
    if full_kl.get("dataset") != _CANONICAL_FULL_KL_DATASET:
        raise RuntimeError("fixed full-KL dataset identity differs")
    if selection != _CANONICAL_FULL_KL_SELECTION:
        raise RuntimeError("fixed full-KL window selection differs")
    raw_windows = full_kl["token_ids"]
    source_samples = selection.get("n_samples")
    source_seqlen = selection.get("seqlen")
    if (
        isinstance(source_samples, bool)
        or not isinstance(source_samples, int)
        or isinstance(source_seqlen, bool)
        or not isinstance(source_seqlen, int)
        or source_samples <= 0
        or source_seqlen <= 1
        or len(raw_windows) != source_samples
    ):
        raise RuntimeError("fixed full-KL selection dimensions are malformed")
    if args.n_samples != source_samples or args.seqlen != source_seqlen:
        raise RuntimeError(
            "quality gate requires the complete sealed prompt selection: "
            f"--n-samples={source_samples} --seqlen={source_seqlen}"
        )
    vocab_size = int(len(tokenizer))
    if vocab_size != _CANONICAL_TOKENIZER_VOCAB_SIZE:
        raise RuntimeError(
            "fixed prompt tokenizer vocabulary differs from the canonical "
            f"DSV4 size {_CANONICAL_TOKENIZER_VOCAB_SIZE}: got {vocab_size}"
        )
    windows: list[list[int]] = []
    flattened: list[int] = []
    for index, raw in enumerate(raw_windows):
        if not isinstance(raw, list) or len(raw) != source_seqlen:
            raise RuntimeError(f"fixed prompt window {index} has wrong length")
        window: list[int] = []
        for value in raw:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= vocab_size
            ):
                raise RuntimeError(
                    f"fixed prompt window {index} contains invalid token id"
                )
            window.append(int(value))
        windows.append(window)
        flattened.extend(window)
    tensor_bytes = struct.pack(f"<{len(flattened)}q", *flattened)
    tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()
    if (
        not _valid_sha256(full_kl.get("token_ids_tensor_sha256"))
        or full_kl["token_ids_tensor_sha256"] != tensor_digest
    ):
        raise RuntimeError("fixed full-KL token tensor digest differs")
    if tensor_digest != _CANONICAL_FULL_KL_TOKEN_IDS_TENSOR_SHA256:
        raise RuntimeError("fixed full-KL token values are not canonical")
    if len({tuple(window) for window in windows}) != len(windows):
        raise RuntimeError("fixed prompt windows are not content-distinct")

    # The harness scores the full-KL windows, not the PPL prefix, but the one
    # canonical payload binds both release workloads. Validate the unused half
    # too so a report cannot cite the canonical semantic identity while carrying
    # substituted corpus metadata or token values elsewhere in the record.
    ppl = payload.get("ppl")
    if (
        not isinstance(ppl, Mapping)
        or set(ppl) != {"dataset", "selection", "token_ids", "token_ids_sha256"}
        or ppl.get("dataset") != _CANONICAL_PPL_DATASET
        or ppl.get("selection") != _CANONICAL_PPL_SELECTION
        or not isinstance(ppl.get("token_ids"), list)
        or len(ppl["token_ids"]) != _CANONICAL_PPL_SELECTION["n_tokens"]
    ):
        raise RuntimeError("fixed PPL corpus selection identity differs")
    ppl_ids: list[int] = []
    for value in ppl["token_ids"]:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= vocab_size
        ):
            raise RuntimeError("fixed PPL token prefix contains invalid token id")
        ppl_ids.append(int(value))
    ppl_digest = _canonical_sha256(ppl_ids)
    if (
        not _valid_sha256(ppl.get("token_ids_sha256"))
        or ppl["token_ids_sha256"] != ppl_digest
        or ppl_digest != _CANONICAL_PPL_TOKEN_IDS_SHA256
    ):
        raise RuntimeError("fixed PPL token values are not canonical")
    return windows, {
        "name": "wikitext",
        "source": "producer_sealed_token_ids",
        "input_file": file_record,
        "semantic_sha256": payload["semantic_sha256"],
        "datasets_distribution": payload["datasets_distribution"],
        "corpus_construction": payload["corpus_construction"],
        "dataset": full_kl["dataset"],
        "selection": selection,
        "tokenizer": {
            "content_sha256": tokenizer_record["content_sha256"],
            "files": observed_files,
            "runtime_vocab_size": vocab_size,
        },
        "prompt_token_ids_tensor_sha256": tensor_digest,
        "prompt_window_sha256": [
            hashlib.sha256(_canonical_json_bytes(window)).hexdigest()
            for window in windows
        ],
    }


@dataclass(frozen=True)
class _LayerBinding:
    layer_id: int
    prefix: str
    method: Any
    layer: Any
    persistent_handle: Any | None
    persistent_cfg: int | None


_FULL_VOCAB_TOKEN_IDS: dict[int, tuple[int, ...]] = {}


def _full_vocab_token_ids(vocab_size: int) -> tuple[int, ...]:
    token_ids = _FULL_VOCAB_TOKEN_IDS.get(vocab_size)
    if token_ids is None:
        token_ids = tuple(range(vocab_size))
        _FULL_VOCAB_TOKEN_IDS[vocab_size] = token_ids
    return token_ids


class _CompactFullVocabRows(Sequence[Any]):
    """Lazy ``TopKRow`` view over packed float32 full-vocabulary rows."""

    def __init__(self, values: array.array, *, rows: int, vocab_size: int):
        if values.typecode != "f":
            raise TypeError("compact full-vocabulary values must be float32")
        if len(values) != rows * vocab_size:
            raise ValueError("compact full-vocabulary row storage is truncated")
        self._values = values
        self._rows = int(rows)
        self._vocab_size = int(vocab_size)
        self._token_ids = _full_vocab_token_ids(vocab_size)

    def __len__(self) -> int:
        return self._rows

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(
                self._rows
            )))
        if index < 0:
            index += self._rows
        if not 0 <= index < self._rows:
            raise IndexError(index)
        start = index * self._vocab_size
        end = start + self._vocab_size
        return v5.TopKRow(
            token_ids=self._token_ids,
            logprobs=tuple(self._values[start:end]),
        )

    def __iter__(self) -> Iterator[Any]:
        for index in range(self._rows):
            yield self[index]


class _BoundedTensorPromptLogprobs:
    """Same-process full-vocabulary transport without Pythonizing every cell.

    vLLM normally calls ``Tensor.tolist()`` on the complete ``[positions,
    vocab + 1]`` prompt-logprob tensors, then copies those values into four
    growing Python lists.  At 511x129280 that transient is larger than the
    model's remaining UMA headroom.  The validation harness forces an in-proc
    engine, so retaining the already-produced CPU tensors until the caller
    compacts one result is both lossless and bounded.

    This deliberately is not a general vLLM ``PromptLogprobs`` implementation:
    only the validation compactor below may consume it.
    """

    schema = "gridbook.bounded-tensor-prompt-logprobs.v1"

    def __init__(self, tensors: Any):
        self.logprob_token_ids = tensors.logprob_token_ids
        self.logprobs = tensors.logprobs
        self.selected_token_ranks = tensors.selected_token_ranks
        self.cu_num_generated_tokens = tensors.cu_num_generated_tokens

    def __len__(self) -> int:
        # Prompt position zero has no conditional distribution.  vLLM's public
        # PromptLogprobs representation includes that leading empty position.
        return int(self.logprobs.shape[0]) + 1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(schema={self.schema!r}, "
            f"shape={tuple(self.logprobs.shape)!r})"
        )

    @property
    def transport_bytes(self) -> int:
        tensors = (
            self.logprob_token_ids,
            self.logprobs,
            self.selected_token_ranks,
        )
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in tensors
        )


def _install_bounded_full_vocab_transport(vllm_module: Any) -> dict[str, Any]:
    """Intercept only ``prompt_logprobs=-1`` before vLLM calls ``tolist``.

    The patch is intentionally local to this fresh, same-process validation
    run.  Coarse top-K requests retain vLLM's stock behavior.  Exact requests
    preserve the CPU tensors emitted by EngineCore and are rejected later if
    their dtype, shape, device, support, or finite-value contract differs.
    """

    del vllm_module  # Import provenance is attested separately by the harness.
    from vllm.logprobs import FlatLogprobs
    from vllm.v1.engine.logprobs import LogprobsProcessor

    marker = "_gridbook_bounded_full_vocab_v1"
    if getattr(LogprobsProcessor, marker, False):
        raise RuntimeError("bounded full-vocabulary transport already installed")
    original = LogprobsProcessor._update_prompt_logprobs

    def update_prompt_logprobs_bounded(self: Any, tensors: Any) -> None:
        if self.num_prompt_logprobs != -1:
            original(self, tensors)
            return
        if self.tokenizer is not None:
            raise RuntimeError(
                "bounded exact prompt-logprob transport requires detokenize=False"
            )
        initial = self.prompt_logprobs
        if not isinstance(initial, FlatLogprobs) or len(initial) != 1:
            raise RuntimeError(
                "bounded exact prompt-logprob transport did not receive the "
                "fresh one-position FlatLogprobs sentinel"
            )
        required = (
            "logprob_token_ids", "logprobs", "selected_token_ranks",
            "cu_num_generated_tokens",
        )
        if not all(hasattr(tensors, name) for name in required):
            raise RuntimeError(
                "vLLM exact prompt-logprob tensor ABI differs from the "
                "attested bounded transport"
            )
        if tensors.cu_num_generated_tokens is not None:
            raise RuntimeError(
                "exact prompt-logprob transport unexpectedly used a generated-"
                "token cumulative layout"
            )
        self.prompt_logprobs = _BoundedTensorPromptLogprobs(tensors)

    LogprobsProcessor._update_prompt_logprobs = update_prompt_logprobs_bounded
    setattr(LogprobsProcessor, marker, True)
    return {
        "schema": _BoundedTensorPromptLogprobs.schema,
        "installed": True,
        "scope": "same_process_prompt_logprobs_minus_one_only",
        "stock_topk_transport_unchanged": True,
        "python_cell_materialization": False,
    }


@dataclass(frozen=True)
class _CompactFullVocabScore:
    """One exact score in float32 rows plus stable evidence digests."""

    target_logprobs: tuple[float, ...]
    _row_values: array.array
    vocab_size: int
    prompt_token_ids_sha256: str
    row_sha256: tuple[str, ...]
    score_sha256: str
    transport_schema: str = "vllm.flat-logprobs.python-lists"
    transport_bytes: int | None = None
    transport_peak_compactor_scratch_bytes: int | None = None

    @property
    def rows(self) -> _CompactFullVocabRows:
        return _CompactFullVocabRows(
            self._row_values,
            rows=len(self.target_logprobs),
            vocab_size=self.vocab_size,
        )

    @property
    def mean_nll(self) -> float:
        return -v5.statistics.fmean(self.target_logprobs)

    @property
    def ppl(self) -> float:
        return math.exp(self.mean_nll)

    def digest_record(self) -> dict[str, Any]:
        return {
            "schema": "gridbook.compact-full-vocab-score.v1",
            "positions": len(self.target_logprobs),
            "vocab_size": self.vocab_size,
            "float_storage": "little_endian_ieee754_float32",
            "row_storage_bytes": len(self._row_values) * self._row_values.itemsize,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "row_sha256": list(self.row_sha256),
            "score_sha256": self.score_sha256,
            "mean_nll": self.mean_nll,
            "ppl": self.ppl,
            "transport_schema": self.transport_schema,
            "transport_bytes": self.transport_bytes,
            "transport_peak_compactor_scratch_bytes": (
                self.transport_peak_compactor_scratch_bytes
            ),
        }


def _compact_tensor_full_vocab_score(
    prompt_logprobs: _BoundedTensorPromptLogprobs,
    expected_ids: Sequence[int],
    *,
    expected_vocab_size: int,
) -> _CompactFullVocabScore:
    """Compact the bounded vLLM CPU tensor transport without ``tolist()``."""

    import torch

    token_ids = prompt_logprobs.logprob_token_ids
    logprobs = prompt_logprobs.logprobs
    ranks = prompt_logprobs.selected_token_ranks
    tensors = {"token ids": token_ids, "logprobs": logprobs, "ranks": ranks}
    if any(getattr(tensor, "device", None) is None for tensor in tensors.values()):
        raise RuntimeError("bounded full-vocabulary transport contains non-tensors")
    if any(tensor.device.type != "cpu" for tensor in tensors.values()):
        raise RuntimeError("bounded full-vocabulary transport must be on CPU")
    if not all(tensor.is_contiguous() for tensor in tensors.values()):
        raise RuntimeError("bounded full-vocabulary transport must be contiguous")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"bounded full-vocabulary token ids have dtype {token_ids.dtype}"
        )
    if logprobs.dtype != torch.float32:
        raise RuntimeError(
            f"bounded full-vocabulary logprobs have dtype {logprobs.dtype}"
        )
    if ranks.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"bounded full-vocabulary ranks have dtype {ranks.dtype}"
        )
    expected_rows = len(expected_ids) - 1
    expected_shape = (expected_rows, expected_vocab_size + 1)
    if tuple(token_ids.shape) != expected_shape:
        raise RuntimeError(
            "bounded full-vocabulary token-id tensor has shape "
            f"{tuple(token_ids.shape)}, expected {expected_shape}"
        )
    if tuple(logprobs.shape) != expected_shape:
        raise RuntimeError(
            "bounded full-vocabulary logprob tensor has shape "
            f"{tuple(logprobs.shape)}, expected {expected_shape}"
        )
    if tuple(ranks.shape) != (expected_rows,):
        raise RuntimeError(
            "bounded full-vocabulary rank tensor has shape "
            f"{tuple(ranks.shape)}, expected {(expected_rows,)}"
        )

    prompt_bytes = struct.pack(f"<{len(expected_ids)}q", *expected_ids)
    prompt_digest = hashlib.sha256(prompt_bytes).hexdigest()
    score_digest = hashlib.sha256()
    score_digest.update(b"gridbook.compact-full-vocab-score.v1\0")
    score_digest.update(struct.pack("<II", expected_rows, expected_vocab_size))
    score_digest.update(bytes.fromhex(prompt_digest))
    row_values = array.array("f")
    targets: list[float] = []
    row_digests: list[str] = []
    peak_compactor_scratch_bytes = (
        expected_vocab_size * 8  # topk ids converted to int64 for scatter
        + expected_vocab_size  # exact one-byte seen bitmap
        + expected_vocab_size * 4
        + expected_vocab_size * 4  # immutable digest bytes
    )
    for row_index, expected_target in enumerate(expected_ids[1:]):
        ids = token_ids[row_index]
        values = logprobs[row_index]
        if int(ids[0].item()) != int(expected_target):
            raise RuntimeError(
                "bounded full-vocabulary row target differs from the sealed "
                f"prompt at position {row_index + 1}"
            )
        topk_ids = ids[1:]
        topk_values = values[1:]
        if not bool(torch.isfinite(topk_values).all().item()):
            raise ValueError("bounded full-vocabulary row contains non-finite logprob")
        # Full-vocabulary topk(V) must be a permutation of [0,V).  With exactly
        # V cells, in-range support whose V-entry seen bitmap is all true is
        # exactly that permutation.  Keep the check and scatter index in native
        # tensor storage; never allocate one Python object per cell.
        scatter_ids = topk_ids.to(dtype=torch.int64, device=token_ids.device)
        if (
            int(scatter_ids.min().item()) < 0
            or int(scatter_ids.max().item()) >= expected_vocab_size
        ):
            raise RuntimeError(
                "bounded full-vocabulary row does not contain each canonical "
                "token exactly once"
            )
        seen = torch.zeros(
            expected_vocab_size, dtype=torch.bool, device=token_ids.device
        )
        seen.scatter_(0, scatter_ids, True)
        if int(torch.count_nonzero(seen).item()) != expected_vocab_size:
            raise RuntimeError(
                "bounded full-vocabulary row does not contain each canonical "
                "token exactly once"
            )
        packed = torch.empty(
            expected_vocab_size, dtype=torch.float32, device=logprobs.device
        )
        packed.scatter_(0, scatter_ids, topk_values)
        target = float(packed[int(expected_target)].item())
        if not math.isfinite(target):
            raise ValueError(f"non-finite logprob {target!r}")
        # ``Tensor.numpy()`` needs NumPy, which is deliberately not a
        # gridbook dependency and is absent from the release pipeline's
        # installed-wheel environment -- it failed the v0.8.7 gate here.
        # ``ctypes.string_at`` copies the same contiguous C-order buffer
        # without it; this is the idiom ``gridbook/cb_digest.py`` already
        # uses for exactly this reason, so the digest is unchanged.
        if packed.device.type != "cpu" or not packed.is_contiguous():
            raise RuntimeError(
                "bounded full-vocabulary row must be contiguous CPU "
                "memory to digest")
        raw = ctypes.string_at(
            packed.data_ptr(), packed.numel() * packed.element_size())
        row_digest = hashlib.sha256(raw).hexdigest()
        row_values.frombytes(raw)
        targets.append(target)
        row_digests.append(row_digest)
        score_digest.update(struct.pack("<d", target))
        score_digest.update(bytes.fromhex(row_digest))
        del ids, values, topk_ids, topk_values, scatter_ids, seen, packed, raw
    if not targets:
        raise RuntimeError("full-vocabulary quality prompt has no scored tokens")
    return _CompactFullVocabScore(
        target_logprobs=tuple(targets),
        _row_values=row_values,
        vocab_size=expected_vocab_size,
        prompt_token_ids_sha256=prompt_digest,
        row_sha256=tuple(row_digests),
        score_sha256=score_digest.hexdigest(),
        transport_schema=prompt_logprobs.schema,
        transport_bytes=prompt_logprobs.transport_bytes,
        transport_peak_compactor_scratch_bytes=peak_compactor_scratch_bytes,
    )


def _compact_full_vocab_score(
    output: Any,
    prompt_ids: Sequence[int],
    *,
    expected_vocab_size: int,
) -> _CompactFullVocabScore:
    """Cardinality-check one flat vLLM result into compact row storage.

    A single full-vocabulary request is already large.  Requiring vLLM's
    primitive ``FlatLogprobs`` representation prevents the much larger
    list-of-dictionaries object graph.  Read its primitive lists directly:
    integer ``__getitem__`` on vLLM's compatibility Sequence would otherwise
    recreate one ``Logprob`` plus one dictionary entry for every vocabulary
    cell.  The direct path deliberately preserves that mapping view's
    last-value-wins duplicate semantics before applying the established v5
    cardinality and finite-value rules.
    """

    if sys.byteorder != "little":
        raise RuntimeError("compact full-vocabulary evidence requires little endian")
    returned = getattr(output, "prompt_token_ids", None)
    try:
        returned_ids = [int(token) for token in returned]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM returned invalid prompt_token_ids") from exc
    expected_ids = [int(token) for token in prompt_ids]
    if returned_ids != expected_ids:
        raise RuntimeError("vLLM returned different prompt_token_ids")
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_ids):
        raise RuntimeError("vLLM returned the wrong prompt_logprobs length")
    if isinstance(prompt_logprobs, _BoundedTensorPromptLogprobs):
        return _compact_tensor_full_vocab_score(
            prompt_logprobs,
            expected_ids,
            expected_vocab_size=expected_vocab_size,
        )
    flat_fields = (
        "start_indices", "end_indices", "token_ids", "logprobs", "ranks",
        "decoded_tokens",
    )
    if not all(hasattr(prompt_logprobs, name) for name in flat_fields):
        raise RuntimeError(
            "full-vocabulary scoring requires vLLM FlatLogprobs; refusing "
            "the cardinality-sized list[dict] representation"
        )

    flat = {name: getattr(prompt_logprobs, name) for name in flat_fields}
    if not all(isinstance(flat[name], list) for name in flat_fields):
        raise RuntimeError(
            "vLLM FlatLogprobs fields must use the pinned primitive-list "
            "layout"
        )
    starts = flat["start_indices"]
    ends = flat["end_indices"]
    token_ids = flat["token_ids"]
    logprobs = flat["logprobs"]
    if len(starts) != len(expected_ids) or len(ends) != len(expected_ids):
        raise RuntimeError(
            "vLLM FlatLogprobs span count does not match prompt length"
        )
    flat_length = len(token_ids)
    for name in ("logprobs", "ranks", "decoded_tokens"):
        if len(flat[name]) != flat_length:
            raise RuntimeError(
                "vLLM FlatLogprobs primitive fields have different lengths"
            )
    if starts and starts[0] != ends[0]:
        raise RuntimeError(
            "vLLM FlatLogprobs position zero must have an empty span"
        )
    previous_end = 0
    for position, (start, end) in enumerate(zip(starts, ends)):
        if type(start) is not int or type(end) is not int:
            raise RuntimeError("vLLM FlatLogprobs spans must be Python ints")
        if start != previous_end:
            raise RuntimeError(
                "vLLM FlatLogprobs spans are not contiguous at position "
                f"{position}"
            )
        if not 0 <= start <= end <= flat_length:
            raise RuntimeError(
                "vLLM FlatLogprobs span is out of bounds at position "
                f"{position}: [{start},{end}) for flat_length={flat_length}"
            )
        previous_end = end
    if previous_end != flat_length:
        raise RuntimeError("vLLM FlatLogprobs has unreferenced trailing cells")

    prompt_bytes = struct.pack(f"<{len(expected_ids)}q", *expected_ids)
    prompt_digest = hashlib.sha256(prompt_bytes).hexdigest()
    score_digest = hashlib.sha256()
    score_digest.update(b"gridbook.compact-full-vocab-score.v1\0")
    score_digest.update(struct.pack("<II", len(expected_ids) - 1,
                                    expected_vocab_size))
    score_digest.update(bytes.fromhex(prompt_digest))
    row_values = array.array("f")
    targets: list[float] = []
    row_digests: list[str] = []
    for position in range(1, len(expected_ids)):
        start = starts[position]
        end = ends[position]
        expected_target = expected_ids[position]

        # FlatLogprobs.__getitem__ builds a dict comprehension.  A full-vocab
        # row normally contains the requested target first and again inside
        # topk(V), so the later value is authoritative.  Reverse lookup keeps
        # that last-value-wins behavior and, like the legacy path, validates
        # the target before any row-cardinality error.
        target = None
        for flat_index in range(end - 1, start - 1, -1):
            raw_token = token_ids[flat_index]
            if type(raw_token) is not int:
                continue
            if raw_token == expected_target:
                target = v5._entry_logprob(logprobs[flat_index])
                break
        if target is None:
            raise KeyError(expected_target)

        # Retain only the final primitive value for each canonical token while
        # remembering first-insertion order.  That order matters for the
        # legacy helper's failure precedence (finite value vs out-of-range),
        # even though successful rows are always packed in token-id order.
        retained: list[Any | None] = [None] * expected_vocab_size
        seen = bytearray(expected_vocab_size)
        first_order: list[int] = []
        observed = 0
        for flat_index in range(start, end):
            raw_token = token_ids[flat_index]
            if type(raw_token) is not int:
                raise RuntimeError(
                    "vLLM FlatLogprobs token_ids must contain Python ints"
                )
            if raw_token < 0 or raw_token >= expected_vocab_size:
                # Invalid ids have no destination slot.  Keeping them in the
                # order stream preserves the legacy negative-skip / positive-
                # out-of-range behavior without allocating a mapping.
                first_order.append(raw_token)
                continue
            if not seen[raw_token]:
                seen[raw_token] = 1
                observed += 1
                first_order.append(raw_token)
            retained[raw_token] = logprobs[flat_index]

        for token in first_order:
            if token < 0:
                continue
            if token >= expected_vocab_size:
                raise RuntimeError(
                    "vLLM full-vocab row returned out-of-range token id "
                    f"{token} for vocab_size={expected_vocab_size}"
                )
            retained[token] = v5._entry_logprob(retained[token])
        if observed != expected_vocab_size or any(
            value is None for value in retained
        ):
            raise RuntimeError(
                "vLLM full-vocab row cardinality mismatch: expected exactly "
                f"{expected_vocab_size} token ids "
                f"[0,{expected_vocab_size - 1}], observed {observed}"
            )
        packed = array.array("f", retained)
        if len(packed) != expected_vocab_size:
            raise RuntimeError("packed full-vocabulary row has wrong cardinality")
        raw = packed.tobytes()
        row_digest = hashlib.sha256(raw).hexdigest()
        row_values.extend(packed)
        targets.append(target)
        row_digests.append(row_digest)
        score_digest.update(struct.pack("<d", target))
        score_digest.update(bytes.fromhex(row_digest))
        del retained, seen, first_order, packed, raw
    if not targets:
        raise RuntimeError("full-vocabulary quality prompt has no scored tokens")
    return _CompactFullVocabScore(
        target_logprobs=tuple(targets),
        _row_values=row_values,
        vocab_size=expected_vocab_size,
        prompt_token_ids_sha256=prompt_digest,
        row_sha256=tuple(row_digests),
        score_sha256=score_digest.hexdigest(),
    )


def _full_vocab_profile(
    prompts: Sequence[Sequence[int]], *, seqlen: int
) -> tuple[list[list[int]], dict[str, Any]]:
    if seqlen <= 16:
        raise RuntimeError("full-vocabulary profile must cross prefill threshold 16")
    selected = [list(map(int, prompt[:seqlen])) for prompt in prompts]
    if any(len(prompt) != seqlen for prompt in selected):
        raise RuntimeError("full-vocabulary profile exceeds sealed prompt length")
    flattened = [token for prompt in selected for token in prompt]
    tensor = struct.pack(f"<{len(flattened)}q", *flattened)
    return selected, {
        "schema": "gridbook.dsv4-full-vocab-prefix-profile.v1",
        "selection": "first_n_tokens_of_every_sealed_full_kl_window",
        "n_samples": len(selected),
        "seqlen": seqlen,
        "positions_per_repeat": len(selected) * (seqlen - 1),
        "token_ids_tensor_sha256": hashlib.sha256(tensor).hexdigest(),
        "prompt_sha256": [
            hashlib.sha256(
                struct.pack(f"<{len(prompt)}q", *prompt)
            ).hexdigest()
            for prompt in selected
        ],
    }


class PersistentBArmController:
    """Controlled same-process switch over already-resolved layer handles."""

    def __init__(
        self,
        *,
        ops: Any,
        moe: Any,
        route_api: Any,
        expected_fp4: int,
        expected_fp8: int,
        expected_cfg: int,
    ) -> None:
        self.route_api = route_api
        self.expected_cfg = int(expected_cfg)
        bindings: list[_LayerBinding] = []
        for layer_id in sorted(ops._LAYER_REGISTRY):
            try:
                method, layer = ops._lookup_cb_layer(layer_id)
            except RuntimeError:
                continue
            if not isinstance(method, moe.PrismaQuantCBMoEMethod):
                continue
            is_fp4 = bool(method.is_fp4)
            handle = getattr(layer, "_cb_moe_persistent_b", None)
            cfg = (
                int(getattr(layer, "_cb_moe_persistent_b_cfg", -1))
                if handle is not None
                else None
            )
            bindings.append(_LayerBinding(
                layer_id=int(layer_id),
                prefix=str(method.prefix),
                method=method,
                layer=layer,
                persistent_handle=handle,
                persistent_cfg=cfg,
            ))
        prefixes = [binding.prefix for binding in bindings]
        if len(prefixes) != len(set(prefixes)):
            raise RuntimeError("loaded CB MoE prefixes are not unique")
        self.fp4 = tuple(binding for binding in bindings
                         if bool(binding.method.is_fp4))
        self.fp8 = tuple(binding for binding in bindings
                         if not bool(binding.method.is_fp4))
        self.inventory_gate = {
            "expected_fp4_cb_moe_layers": int(expected_fp4),
            "observed_fp4_cb_moe_layers": len(self.fp4),
            "expected_fp8_cb_moe_layers": int(expected_fp8),
            "observed_fp8_cb_moe_layers": len(self.fp8),
            "fp4_prefixes": [binding.prefix for binding in self.fp4],
            "fp8_prefixes": [binding.prefix for binding in self.fp8],
            "all_fp4_resolved_persistent_b_at_load": all(
                binding.persistent_handle is not None for binding in self.fp4
            ),
            "all_fp4_cfg_match": all(
                binding.persistent_cfg == self.expected_cfg for binding in self.fp4
            ),
            "all_fp8_excluded_from_persistent_b": all(
                binding.persistent_handle is None for binding in self.fp8
            ),
        }
        self.inventory_gate["pass"] = bool(
            len(self.fp4) == expected_fp4
            and len(self.fp8) == expected_fp8
            and self.inventory_gate[
                "all_fp4_resolved_persistent_b_at_load"
            ]
            and self.inventory_gate["all_fp4_cfg_match"]
            and self.inventory_gate["all_fp8_excluded_from_persistent_b"]
        )
        if not self.inventory_gate["pass"]:
            raise RuntimeError(
                "loaded DSV4 CB MoE inventory does not match the requested "
                f"contract: {self.inventory_gate}"
            )
        for binding in self.fp4:
            if getattr(binding.layer, "_cb_bf16_sm120", None) is not None:
                raise RuntimeError(
                    f"{binding.prefix}: sm120 bridge lane is active; baseline "
                    "must be the default expand+grouped-BF16 bridge"
                )
            if getattr(binding.layer, "_cb_fused_fp4_moe_mode", None):
                raise RuntimeError(
                    f"{binding.prefix}: fused activation-contract lane is active"
                )

    def clear_routes(self) -> None:
        for binding in (*self.fp4, *self.fp8):
            setattr(binding.layer, "_cb_route_state", None)

    @contextmanager
    def arm(self, arm: str, *, label: str) -> Iterator[dict[str, Any]]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        before = [
            getattr(binding.layer, "_cb_moe_persistent_b", None)
            for binding in self.fp4
        ]
        requested = arm == "fused"
        for binding in self.fp4:
            binding.layer._cb_moe_persistent_b = (
                binding.persistent_handle if requested else None
            )
        observed = [
            getattr(binding.layer, "_cb_moe_persistent_b", None)
            is binding.persistent_handle
            for binding in self.fp4
        ]
        selector = {
            "label": label,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "pid": os.getpid(),
            "expected_handle_present": requested,
            "observed_handle_present_count": (
                sum(observed) if requested else len(observed) - sum(observed)
            ),
            "expected_layer_count": len(self.fp4),
            "pass": all(observed) if requested else not any(observed),
        }
        if not selector["pass"]:
            raise RuntimeError(f"persistent-B arm switch failed: {selector}")
        self.clear_routes()
        try:
            yield selector
        finally:
            for binding, value in zip(self.fp4, before):
                binding.layer._cb_moe_persistent_b = value

    def attest_routes(self, arm: str, *, label: str) -> dict[str, Any]:
        expected = (
            {
                "policy": "moe_persistent_b",
                "symbol": "cb_moe_persistent_b_prefill",
                "tile_m": self.expected_cfg,
                "contract": "fp4_group16_rtn",
                "state": "served",
                "reason": None,
            }
            if arm == "fused"
            else {
                "policy": "bf16_grouped_bridge",
                "symbol": "cb_bf16_grouped_mm_out",
                "tile_m": 0,
                "contract": "fp4_group16_rtn",
                "state": "served",
                "reason": None,
            }
        )
        fp4_routes: dict[str, Any] = {}
        fp8_routes: dict[str, Any] = {}
        violations: list[str] = []
        for binding in self.fp4:
            route = self.route_api.read_route(binding.layer)
            fp4_routes[binding.prefix] = route
            if route is None:
                violations.append(f"{binding.prefix}: no current-request route")
                continue
            for field, wanted in expected.items():
                if route.get(field) != wanted:
                    violations.append(
                        f"{binding.prefix}: {field}={route.get(field)!r}, "
                        f"expected {wanted!r}"
                    )
            handle_present = (
                getattr(binding.layer, "_cb_moe_persistent_b", None)
                is binding.persistent_handle
            )
            if handle_present != (arm == "fused"):
                violations.append(
                    f"{binding.prefix}: dispatch handle changed during request"
                )
        for binding in self.fp8:
            route = self.route_api.read_route(binding.layer)
            fp8_routes[binding.prefix] = route
            if route is None:
                violations.append(f"{binding.prefix}: no current-request FP8 route")
            elif route.get("state") != "served" or route.get("reason") is not None:
                violations.append(
                    f"{binding.prefix}: FP8 route was not served cleanly: {route}"
                )
            if getattr(binding.layer, "_cb_moe_persistent_b", None) is not None:
                violations.append(
                    f"{binding.prefix}: FP8 layer acquired a persistent-B handle"
                )
        return {
            "label": label,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "expected_fp4_route": expected,
            "fp4_routes": fp4_routes,
            "fp8_routes": fp8_routes,
            "violations": violations,
            "pass": not violations,
        }


def _run_generate(
    *,
    llm: Any,
    sampling: Any,
    prompt_ids: list[int],
    arm: str,
    label: str,
    controller: PersistentBArmController,
    synchronize: Any,
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    with controller.arm(arm, label=label) as selector:
        synchronize()
        started = time.perf_counter()
        output = llm.generate(
            [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
        )[0]
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        routes = controller.attest_routes(arm, label=label)
    return output, elapsed_ms, selector, routes


def _model_contract_gate(
    config: Any, args: argparse.Namespace
) -> dict[str, Any]:
    raw = v5._config_dict(config)
    observed_architectures = list(raw.get("architectures") or ())
    checks = {
        "architecture": (
            observed_architectures == [args.expected_architecture]
        ),
        "model_type": raw.get("model_type") == args.expected_model_type,
        "num_hidden_layers": (
            raw.get("num_hidden_layers") == args.expected_hidden_layers
        ),
        "hidden_size": raw.get("hidden_size") == args.expected_hidden_size,
        "moe_intermediate_size": (
            raw.get("moe_intermediate_size") == args.expected_moe_intermediate_size
        ),
        "n_routed_experts": (
            raw.get("n_routed_experts") == args.expected_routed_experts
        ),
        "vocab_size": raw.get("vocab_size") == args.expected_vocab_size,
    }
    return {
        "expected": {
            "architectures": [args.expected_architecture],
            "model_type": args.expected_model_type,
            "num_hidden_layers": args.expected_hidden_layers,
            "hidden_size": args.expected_hidden_size,
            "moe_intermediate_size": args.expected_moe_intermediate_size,
            "n_routed_experts": args.expected_routed_experts,
            "vocab_size": args.expected_vocab_size,
        },
        "observed": {
            key: raw.get(key) for key in (
                "architectures", "model_type", "num_hidden_layers",
                "hidden_size", "moe_intermediate_size", "n_routed_experts",
                "vocab_size",
            )
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _triplet_order_gate(
    triplets: Sequence[Mapping[str, Any]],
    *,
    n_prompts: int,
    expected_pid: int,
    first_request_index: int,
) -> dict[str, Any]:
    """Require one uninterrupted B--PB--B block for every sealed prompt."""

    violations: list[dict[str, Any]] = []
    seen_prompts: list[int] = []
    expected_request = int(first_request_index)
    for block_position, triplet in enumerate(triplets):
        prompt_index = triplet.get("source_prompt_index")
        if type(prompt_index) is not int:
            violations.append({
                "block_position": block_position,
                "reason": "missing integer source_prompt_index",
            })
            continue
        seen_prompts.append(prompt_index)
        if list(triplet.get("request_order") or ()) != list(_TRIPLET_ROLES):
            violations.append({
                "prompt_index": prompt_index,
                "reason": "request roles are not consecutive B-PB-B",
                "observed": list(triplet.get("request_order") or ()),
            })
        expected_arms = [arm for _role, arm in _TRIPLET_REQUESTS]
        if list(triplet.get("arm_order") or ()) != expected_arms:
            violations.append({
                "prompt_index": prompt_index,
                "reason": "concrete arm order differs from baseline/fused/baseline",
                "observed": list(triplet.get("arm_order") or ()),
            })
        wanted_sequence = list(range(expected_request, expected_request + 3))
        observed_sequence = list(triplet.get("request_sequence") or ())
        if observed_sequence != wanted_sequence:
            violations.append({
                "prompt_index": prompt_index,
                "reason": "measured requests are not globally consecutive",
                "expected": wanted_sequence,
                "observed": observed_sequence,
            })
        expected_request += 3
        request_pids = triplet.get("request_pids")
        observed_pids = (
            [request_pids.get(role) for role in _TRIPLET_ROLES]
            if isinstance(request_pids, Mapping)
            else []
        )
        if observed_pids != [expected_pid] * 3:
            violations.append({
                "prompt_index": prompt_index,
                "reason": "triplet requests did not execute in the attested PID",
                "expected_pid": expected_pid,
                "observed_pids": observed_pids,
            })
    expected_prompts = list(range(n_prompts))
    if seen_prompts != expected_prompts:
        violations.append({
            "reason": "sealed prompts are missing, duplicated, or reordered",
            "expected": expected_prompts,
            "observed": seen_prompts,
        })
    expected_requests = n_prompts * 3
    return {
        "protocol": "consecutive_baseline_candidate_baseline_per_prompt",
        "expected_prompt_count": n_prompts,
        "observed_prompt_count": len(triplets),
        "expected_measured_requests": expected_requests,
        "observed_measured_requests": sum(
            len(triplet.get("request_sequence") or ()) for triplet in triplets
        ),
        "expected_pid": expected_pid,
        "first_request_index": first_request_index,
        "violations": violations,
        "pass": not violations,
    }


def _warmup_gate(
    records: Sequence[Mapping[str, Any]], *, profiles: Sequence[str],
    triplets_per_profile: int,
) -> dict[str, Any]:
    required_profiles = tuple(profiles)
    if not required_profiles or len(required_profiles) != len(set(required_profiles)):
        raise ValueError("warmup profiles must be nonempty and unique")
    expected = [
        (profile, repeat, role, arm)
        for profile in required_profiles
        for repeat in range(triplets_per_profile)
        for role, arm in _TRIPLET_REQUESTS
    ]
    observed = [
        (
            record.get("profile"),
            record.get("repeat_index"),
            record.get("role"),
            record.get("arm"),
        )
        for record in records
    ]
    return {
        "required_profiles": list(required_profiles),
        "triplets_per_profile": triplets_per_profile,
        "expected_requests": len(expected),
        "observed_requests": len(observed),
        "expected_order": [list(item) for item in expected],
        "observed_order": [list(item) for item in observed],
        "pass": observed == expected,
    }


def _fp8_invariance_gate(
    triplets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for triplet in triplets:
        routes = triplet.get("routes")
        role_routes = (
            [routes[role]["fp8_routes"] for role in _TRIPLET_ROLES]
            if isinstance(routes, Mapping)
            and all(role in routes for role in _TRIPLET_ROLES)
            else []
        )
        if len(role_routes) != 3 or not (
            role_routes[0] == role_routes[1] == role_routes[2]
        ):
            mismatches.append({
                "block_index": triplet.get("block_index"),
                "source_prompt_index": triplet.get("source_prompt_index"),
                "baseline_pre": role_routes[0] if len(role_routes) == 3 else None,
                "candidate": role_routes[1] if len(role_routes) == 3 else None,
                "baseline_post": role_routes[2] if len(role_routes) == 3 else None,
            })
    return {
        "triplet_blocks": len(triplets),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _route_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        {
            "label": record["label"],
            "arm": record["arm"],
            "violations": record["violations"],
        }
        for record in records if not record["pass"]
    ]
    return {
        "requests": len(records),
        "failed_requests": failed,
        "pass": bool(records) and not failed,
    }


def _selector_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [dict(record) for record in records if not record["pass"]]
    return {
        "measurements": len(records),
        "failed": failed,
        "pass": bool(records) and not failed,
    }


def _teacher_arm_scores(
    triplets: Sequence[Mapping[str, Any]], *, n_prompts: int
) -> dict[str, list[Any]]:
    """Use the leading baseline and candidate for an optional coarse teacher."""

    if len(triplets) != n_prompts:
        raise RuntimeError(
            f"teacher scoring needs {n_prompts} triplets, got {len(triplets)}"
        )
    return {
        "baseline": [triplet["scores"]["baseline_pre"] for triplet in triplets],
        "fused": [triplet["scores"]["candidate"] for triplet in triplets],
    }


def _finite_values(values: Sequence[float], *, label: str) -> list[float]:
    parsed = [float(value) for value in values]
    if not parsed or any(not math.isfinite(value) for value in parsed):
        raise RuntimeError(f"{label} must contain finite values")
    return parsed


def _fixed_prompt_t_interval(
    values: Sequence[float], *, limit: float, label: str
) -> dict[str, Any]:
    """One-sided t bounds with the sealed n=8/df=7 ternary contract."""

    parsed = _finite_values(values, label=label)
    if len(parsed) != _SEALED_PROMPT_COUNT:
        raise RuntimeError(
            f"{label} requires exactly {_SEALED_PROMPT_COUNT} prompt blocks, "
            f"got {len(parsed)}"
        )
    mean = statistics.fmean(parsed)
    sample_sd = statistics.stdev(parsed)
    standard_error = sample_sd / math.sqrt(_SEALED_PROMPT_COUNT)
    half_width = _ONE_SIDED_T_95_DF7 * standard_error
    lcb = mean - half_width
    ucb = mean + half_width
    passed = ucb <= limit
    failed = lcb > limit
    return {
        "unit": "sealed_prompt_block",
        "n": _SEALED_PROMPT_COUNT,
        "df": _SEALED_PROMPT_COUNT - 1,
        "confidence_per_bound": 0.95,
        "bounds": (
            "one-sided upper bound for PASS and one-sided lower bound for FAIL"
        ),
        "t_critical": _ONE_SIDED_T_95_DF7,
        "values": parsed,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "half_width": half_width,
        "lcb": lcb,
        "ucb": ucb,
        "limit": float(limit),
        "point_estimate_excess": mean > limit,
        "pass": passed,
        "fail": failed,
        "decision": (
            "PASS" if passed else "FAIL" if failed else "INCONCLUSIVE"
        ),
    }


def _symmetric_exact_kl(left: Any, right: Any) -> float:
    """Mean half-Jeffreys symmetric KL; positions are not replicates."""

    if len(left.rows) != len(right.rows) or not len(left.rows):
        raise RuntimeError("exact symmetric-KL prompt rows are empty or misaligned")
    position_values: list[float] = []
    for left_row, right_row in zip(left.rows, right.rows):
        forward = v5.exact_full_vocab_kl(left_row, right_row)
        reverse = v5.exact_full_vocab_kl(right_row, left_row)
        value = 0.5 * (forward + reverse)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("exact symmetric KL became invalid")
        position_values.append(value)
    return statistics.fmean(position_values)


def _exact_triplet_metrics(
    scores: Mapping[str, Any], *, prompt_index: int
) -> dict[str, Any]:
    if set(scores) != set(_TRIPLET_ROLES):
        raise RuntimeError("exact triplet score roles differ from B-PB-B")
    baseline_pre = scores["baseline_pre"]
    candidate = scores["candidate"]
    baseline_post = scores["baseline_post"]
    candidate_pre = _symmetric_exact_kl(candidate, baseline_pre)
    candidate_post = _symmetric_exact_kl(candidate, baseline_post)
    d_cb = 0.5 * (candidate_pre + candidate_post)
    d_bb = _symmetric_exact_kl(baseline_pre, baseline_post)
    return {
        "prompt_index": int(prompt_index),
        "positions": len(candidate.rows),
        "symmetric_kl_candidate_baseline_pre": candidate_pre,
        "symmetric_kl_candidate_baseline_post": candidate_post,
        "D_CB": d_cb,
        "D_BB": d_bb,
    }


def _exact_profile_completion_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    n_prompts: int,
    seqlen: int,
    executed: bool,
) -> dict[str, Any]:
    """Require every prompt and every scored position in an exact profile."""

    expected_prompt_indices = list(range(n_prompts))
    expected_positions = seqlen - 1
    observed_prompt_indices = [
        record.get("prompt_index") for record in records
    ]
    observed_positions = [record.get("positions") for record in records]
    complete = (
        executed
        and observed_prompt_indices == expected_prompt_indices
        and observed_positions == [expected_positions] * n_prompts
    )
    return {
        "required": True,
        "executed": bool(executed),
        "expected_prompts": n_prompts,
        "observed_prompts": len(records),
        "expected_prompt_indices": expected_prompt_indices,
        "observed_prompt_indices": observed_prompt_indices,
        "seqlen": seqlen,
        "expected_scored_positions_per_prompt": expected_positions,
        "observed_scored_positions_per_prompt": observed_positions,
        "pass": complete,
    }


def _nll_prompt_metrics(
    triplets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for triplet in triplets:
        scores = triplet.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(_TRIPLET_ROLES):
            raise RuntimeError("coarse triplet score roles differ from B-PB-B")
        values = {
            role: float(scores[role].mean_nll) for role in _TRIPLET_ROLES
        }
        _finite_values(list(values.values()), label="per-prompt NLL")
        baseline_center = 0.5 * (
            values["baseline_pre"] + values["baseline_post"]
        )
        result.append({
            "prompt_index": int(triplet["source_prompt_index"]),
            **{f"{role}_mean_nll": value for role, value in values.items()},
            "baseline_center_mean_nll": baseline_center,
            "effect_candidate_minus_baseline_center": (
                values["candidate"] - baseline_center
            ),
            "baseline_post_minus_pre": (
                values["baseline_post"] - values["baseline_pre"]
            ),
            "corruption_excess": (
                values["candidate"]
                - max(values["baseline_pre"], values["baseline_post"])
            ),
        })
    return result


def _promotion_quality_v6(
    nll_prompts: Sequence[Mapping[str, Any]],
    exact_prompts: Sequence[Mapping[str, Any]],
    *,
    integrity_pass: bool,
) -> dict[str, Any]:
    """Return the fixed v6 PASS/FAIL/INCONCLUSIVE promotion decision."""

    expected_prompt_indices = list(range(_SEALED_PROMPT_COUNT))
    for label, records in (
        ("NLL", nll_prompts), ("exact symmetric KL", exact_prompts)
    ):
        observed = [record.get("prompt_index") for record in records]
        if observed != expected_prompt_indices:
            raise RuntimeError(
                f"{label} prompt records must be exactly {expected_prompt_indices}, "
                f"got {observed}"
            )

    effects = _finite_values(
        [record["effect_candidate_minus_baseline_center"]
         for record in nll_prompts],
        label="prompt NLL effects",
    )
    baseline_deltas = _finite_values(
        [record["baseline_post_minus_pre"] for record in nll_prompts],
        label="baseline NLL differences",
    )
    corruption = _finite_values(
        [record["corruption_excess"] for record in nll_prompts],
        label="prompt corruption excess",
    )
    d_cb = _finite_values(
        [record["D_CB"] for record in exact_prompts], label="D_CB"
    )
    d_bb = _finite_values(
        [record["D_BB"] for record in exact_prompts], label="D_BB"
    )
    if any(value < 0.0 for value in (*d_cb, *d_bb)):
        raise RuntimeError("exact symmetric-KL values must be nonnegative")

    nll_gate = _fixed_prompt_t_interval(
        effects, limit=_NLL_LIMIT, label="prompt NLL effects"
    )
    baseline_mean_square = statistics.fmean(
        value * value for value in baseline_deltas
    )
    h_noise = _ONE_SIDED_T_95_DF7 * math.sqrt(
        0.75 * baseline_mean_square / _SEALED_PROMPT_COUNT
    )
    noise_gate = {
        "formula": (
            "t_0.95,df=7 * sqrt(0.75 * mean_i((Bpost_i-Bpre_i)^2) / 8)"
        ),
        "variance_derivation": (
            "Var[C-(Bpre+Bpost)/2]=1.5*sigma^2 and "
            "Var[Bpost-Bpre]=2*sigma^2"
        ),
        "assumptions": (
            "equal independent request-noise variance for C, Bpre, and Bpost; "
            "the main prompt-effect t interval still carries observed candidate "
            "variation, while this baseline-only diagnostic cannot identify "
            "candidate-specific heteroskedasticity or autocorrelation"
        ),
        "role": (
            "equal-variance baseline-resolution diagnostic, not a candidate "
            "variance estimate"
        ),
        "n": _SEALED_PROMPT_COUNT,
        "df": _SEALED_PROMPT_COUNT - 1,
        "t_critical": _ONE_SIDED_T_95_DF7,
        "baseline_differences": baseline_deltas,
        "baseline_difference_rms": math.sqrt(baseline_mean_square),
        "maximum_resolvable_baseline_difference_rms": (
            _NLL_LIMIT
            * math.sqrt(_SEALED_PROMPT_COUNT / 0.75)
            / _ONE_SIDED_T_95_DF7
        ),
        "h_noise": h_noise,
        "limit": _NLL_LIMIT,
        # The analytically inverted boundary can round one binary64 ULP above
        # itself after square/multiply/square-root.  Admit only that one ULP;
        # this preserves mathematical <= without widening the policy.
        "pass": h_noise <= math.nextafter(_NLL_LIMIT, math.inf),
    }
    corruption_max = max(corruption)
    corruption_gate = {
        "formula": "max_i[C_i - max(Bpre_i, Bpost_i)]",
        "values": corruption,
        "observed_max": corruption_max,
        "limit": _CORRUPTION_LIMIT,
        "pass": corruption_max <= _CORRUPTION_LIMIT,
    }

    absolute_allowances = [baseline + _J_ABSOLUTE_MARGIN for baseline in d_bb]
    relative_allowances = [
        _J_NOISE_MULTIPLIER * baseline for baseline in d_bb
    ]
    hybrid_allowances = [
        max(absolute, relative)
        for absolute, relative in zip(absolute_allowances, relative_allowances)
    ]
    hybrid_excess = [
        candidate - allowance
        for candidate, allowance in zip(d_cb, hybrid_allowances)
    ]
    hybrid_gate = _fixed_prompt_t_interval(
        hybrid_excess,
        limit=0.0,
        label=(
            "D_CB-max(D_BB+1e-4,1.25*D_BB)"
        ),
    )
    divergence_gate = {
        "metric": "prompt_mean_symmetric_exact_full_vocab_kl",
        "definition": "0.5 * (KL(P||Q) + KL(Q||P))",
        "normalization_note": "half the Jeffreys divergence",
        "D_CB_definition": (
            "0.5 * (symmetric_kl(C,Bpre) + symmetric_kl(C,Bpost))"
        ),
        "D_BB_definition": "symmetric_kl(Bpre,Bpost)",
        "D_CB": d_cb,
        "D_BB": d_bb,
        "absolute_margin": _J_ABSOLUTE_MARGIN,
        "relative_noise_multiplier": _J_NOISE_MULTIPLIER,
        "relative_excess_policy": (
            "The candidate may consume at most 25% more symmetric-KL noise "
            "than Bpre-vs-Bpost when that allowance exceeds the fixed 1e-4 "
            "margin. The 1.25 multiplier is a predeclared promotion policy, "
            "not a variance identity."
        ),
        "absolute_allowance_D_BB_plus_1e_4": absolute_allowances,
        "relative_allowance_1_25_times_D_BB": relative_allowances,
        "hybrid_allowance": hybrid_allowances,
        "hybrid_excess_D_CB_minus_allowance": hybrid_excess,
        "acceptance": (
            "one 95% prompt-unit t interval over "
            "g_i=D_CB_i-max(D_BB_i+1e-4,1.25*D_BB_i); "
            "UCB(g)<=0 PASS, LCB(g)>0 FAIL, otherwise INCONCLUSIVE"
        ),
        "gate": hybrid_gate,
        "pass": hybrid_gate["pass"],
        "fail": hybrid_gate["fail"],
    }

    fail_reasons: list[str] = []
    inconclusive_reasons: list[str] = []
    if not integrity_pass:
        fail_reasons.append("integrity_violation")
    if nll_gate["fail"]:
        fail_reasons.append("mean_nll_effect_lcb_exceeds_limit")
    if not corruption_gate["pass"]:
        fail_reasons.append("one_arm_corruption_backstop_exceeded")
    if hybrid_gate["fail"]:
        fail_reasons.append("exact_symmetric_kl_hybrid_excess_lcb_above_zero")
    if not noise_gate["pass"]:
        inconclusive_reasons.append("baseline_noise_exceeds_resolution_limit")
    if not nll_gate["pass"] and not nll_gate["fail"]:
        inconclusive_reasons.append("nll_confidence_interval_too_wide")
    if not hybrid_gate["pass"] and not hybrid_gate["fail"]:
        inconclusive_reasons.append("symmetric_kl_confidence_interval_too_wide")

    if fail_reasons:
        decision = "FAIL"
    elif inconclusive_reasons:
        decision = "INCONCLUSIVE"
    else:
        decision = "PROMOTION PASS"
    return {
        "protocol": _PROMOTION_PROTOCOL,
        "decision": decision,
        "statistical_unit": "sealed_prompt_block_not_token_row",
        "fail_reasons": fail_reasons,
        "inconclusive_reasons": inconclusive_reasons,
        "nll": {
            "per_prompt": [dict(record) for record in nll_prompts],
            "gate": nll_gate,
        },
        "baseline_noise_resolution": noise_gate,
        "corruption_backstop": corruption_gate,
        "exact_full_vocab_symmetric_kl": divergence_gate,
        "pass": decision == "PROMOTION PASS",
    }


def _augment_persistent_b_provenance(runtime: MutableMapping[str, Any]) -> None:
    package_root = Path(runtime["gridbook"]["package_root"])
    additions = {
        "moe_persistent_b_lane.py": package_root / "moe_persistent_b_lane.py",
        "cb_moe_persistent_b.cu": package_root / "csrc" / "cb_moe_persistent_b.cu",
    }
    for label, path in additions.items():
        record = {"path": str(path.resolve()), **v5._required_file_record(path)}
        runtime["source_files"][label] = record
        runtime["source_sha256"][label] = record["sha256"]


def _warm_profile(
    *,
    profile: str,
    prompts: Sequence[Sequence[int]],
    sampling: Any,
    repeats: int,
    llm: Any,
    controller: PersistentBArmController,
    synchronize: Any,
    warmup_records: list[dict[str, Any]],
    selector_records: list[dict[str, Any]],
    route_records: list[dict[str, Any]],
) -> None:
    """Warm one concrete request profile immediately before its measurement."""

    if not prompts:
        raise RuntimeError(f"{profile} warmup has no prompts")
    for repeat in range(repeats):
        prompt_ids = list(prompts[repeat % len(prompts)])
        for role, arm in _TRIPLET_REQUESTS:
            output, wall_ms, selector, route = _run_generate(
                llm=llm,
                sampling=sampling,
                prompt_ids=prompt_ids,
                arm=arm,
                label=f"warmup:{profile}:{repeat}:{role}",
                controller=controller,
                synchronize=synchronize,
            )
            selector_records.append(selector)
            route_records.append(route)
            warmup_records.append({
                "profile": profile,
                "repeat_index": repeat,
                "role": role,
                "arm": arm,
                "wall_ms": wall_ms,
                "selector": selector,
                "routes": route,
            })
            del output


def _measure_exact_profile(
    *,
    profile: str,
    prompts: Sequence[Sequence[int]],
    sampling: Any,
    llm: Any,
    controller: PersistentBArmController,
    synchronize: Any,
    expected_vocab_size: int,
    first_request_index: int,
    selector_records: list[dict[str, Any]],
    route_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Measure one exact-vocabulary B-PB-B profile and release rows promptly."""

    triplets: list[dict[str, Any]] = []
    prompt_metrics: list[dict[str, Any]] = []
    request_index = int(first_request_index)
    for prompt_index, raw_prompt_ids in enumerate(prompts):
        prompt_ids = list(raw_prompt_ids)
        compact_scores: dict[str, _CompactFullVocabScore] = {}
        score_digests: dict[str, dict[str, Any]] = {}
        routes: dict[str, Any] = {}
        selectors: dict[str, Any] = {}
        walls: dict[str, float] = {}
        request_sequence: list[int] = []
        for role, arm in _TRIPLET_REQUESTS:
            request_sequence.append(request_index)
            request_index += 1
            output, wall_ms, selector, route = _run_generate(
                llm=llm,
                sampling=sampling,
                prompt_ids=prompt_ids,
                arm=arm,
                label=f"{profile}:{prompt_index}:{role}",
                controller=controller,
                synchronize=synchronize,
            )
            selector_records.append(selector)
            route_records.append(route)
            selectors[role] = selector
            routes[role] = route
            walls[role] = wall_ms
            score = _compact_full_vocab_score(
                output,
                prompt_ids,
                expected_vocab_size=expected_vocab_size,
            )
            maximum_transport_bytes = (
                (len(prompt_ids) - 1)
                * (expected_vocab_size + 1)
                * (8 + 4)
                + (len(prompt_ids) - 1) * 8
            )
            if score.transport_schema != _BoundedTensorPromptLogprobs.schema:
                raise RuntimeError(
                    "exact full-vocabulary request bypassed the bounded tensor "
                    "transport"
                )
            if (
                score.transport_bytes is None
                or score.transport_bytes > maximum_transport_bytes
            ):
                raise RuntimeError(
                    "exact full-vocabulary tensor transport exceeds its sealed "
                    f"cardinality bound: {score.transport_bytes!r} > "
                    f"{maximum_transport_bytes}"
                )
            maximum_scratch_bytes = expected_vocab_size * (8 + 1 + 4 + 4)
            if (
                score.transport_peak_compactor_scratch_bytes is None
                or score.transport_peak_compactor_scratch_bytes
                > maximum_scratch_bytes
            ):
                raise RuntimeError(
                    "exact full-vocabulary compactor scratch exceeds its sealed "
                    f"one-row bound: "
                    f"{score.transport_peak_compactor_scratch_bytes!r} > "
                    f"{maximum_scratch_bytes}"
                )
            compact_scores[role] = score
            score_digests[role] = score.digest_record()
            del output
        metrics = _exact_triplet_metrics(
            compact_scores, prompt_index=prompt_index
        )
        prompt_metrics.append(metrics)
        triplets.append({
            "profile": profile,
            "prompt_index": prompt_index,
            "block_index": prompt_index,
            "source_prompt_index": prompt_index,
            "request_order": list(_TRIPLET_ROLES),
            "arm_order": [arm for _role, arm in _TRIPLET_REQUESTS],
            "request_sequence": request_sequence,
            "request_pids": {
                role: selectors[role]["pid"] for role in _TRIPLET_ROLES
            },
            "score_digests": score_digests,
            "prompt_metrics": metrics,
            "wall_ms": walls,
            "routes": routes,
        })
        # Three compact float32 row arrays are the only cardinality-sized
        # survivors during a triplet.  Release them before the next prompt.
        del compact_scores, score
    return triplets, prompt_metrics, request_index


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.kv_cache_dtype != DSV4_KV_CACHE_DTYPE:
        raise RuntimeError(
            "persistent-B DSV4 validation requires "
            f"kv_cache_dtype={DSV4_KV_CACHE_DTYPE!r}, got "
            f"{args.kv_cache_dtype!r}"
        )
    if "vllm" in sys.modules:
        raise RuntimeError(
            "vLLM was imported before the harness could force same-process "
            f"execution; launch {Path(__file__).name} in a fresh Python process"
        )
    inherited = {
        name: os.environ.get(name) for name in (
            PERSISTENT_B_ENV, PERSISTENT_B_CFG_ENV, BF16_SM120_ENV,
            v5.FUSED_ENV, v5.FUSED_MOE_ENV, v5.PREFILL_ENV,
        )
    }
    os.environ[v5.VLLM_MP_ENV] = "0"
    os.environ[PERSISTENT_B_ENV] = "1"
    os.environ[PERSISTENT_B_CFG_ENV] = str(args.persistent_b_cfg)
    os.environ[BF16_SM120_ENV] = "0"
    os.environ[v5.FUSED_ENV] = ""
    os.environ[v5.FUSED_MOE_ENV] = ""
    os.environ.pop(v5.PREFILL_ENV, None)
    started = time.monotonic()
    bootstrap = validation_common.prepare_validation(
        args,
        harness_path=Path(__file__),
        helpers=v5,
        extension_none_message=(
            "persistent-B extension did not build/load; refusing a fallback A/B"
        ),
        extension_loader="get_moe_persistent_b_ext",
        required_symbol="cb_moe_persistent_b_prefill",
        validation_name="persistent-B",
        prompt_loader=_fixed_prompt_loader,
    )
    bounded_transport = _install_bounded_full_vocab_transport(bootstrap.vllm)
    torch = bootstrap.torch
    moe = bootstrap.moe
    runtime = bootstrap.runtime
    confirmation_transport_bytes = (
        (_EXACT_CONFIRMATION_SEQLEN - 1)
        * (bootstrap.candidate_vocab_size + 1)
        * (8 + 4)
        + (_EXACT_CONFIRMATION_SEQLEN - 1) * 8
    )
    confirmation_triplet_compact_bytes = (
        3
        * (_EXACT_CONFIRMATION_SEQLEN - 1)
        * bootstrap.candidate_vocab_size
        * 4
    )
    confirmation_compactor_scratch_bytes = (
        bootstrap.candidate_vocab_size * (8 + 1 + 4 + 4)
    )
    runtime["bounded_full_vocab_transport"] = {
        **bounded_transport,
        "maximum_tensor_transport_bytes_8byte_ids_and_ranks": (
            confirmation_transport_bytes
        ),
        "three_compact_float32_scores_bytes": (
            confirmation_triplet_compact_bytes
        ),
        "maximum_cardinality_payload_bytes_during_third_arm": (
            confirmation_transport_bytes
            + confirmation_triplet_compact_bytes
            + confirmation_compactor_scratch_bytes
        ),
        "maximum_one_row_compactor_scratch_bytes": (
            confirmation_compactor_scratch_bytes
        ),
        "excludes": (
            "model, KV cache, allocator fragmentation, and vLLM's transient "
            "GPU chunk/cat plus asynchronous D2H-copy buffers before the "
            "hook receives the final CPU tensors"
        ),
    }
    _augment_persistent_b_provenance(runtime)
    from gridbook import nvfp4_activation_contract, ops

    model_gate = _model_contract_gate(bootstrap.candidate_config, args)
    if not model_gate["pass"]:
        raise RuntimeError(f"loaded model is not the requested DSV4: {model_gate}")

    class _LoadProbe:
        @staticmethod
        def restore() -> None:
            return None

    engine = validation_common.load_candidate_engine(
        bootstrap, args, probe=_LoadProbe(), attest_chunked_prefill=True
    )
    chunked_contract = engine.chunked_prefill_contract
    if chunked_contract is None:
        raise RuntimeError("chunked-prefill contract attestation did not run")
    controller = PersistentBArmController(
        ops=ops,
        moe=moe,
        route_api=nvfp4_activation_contract,
        expected_fp4=args.expected_persistent_b_layers,
        expected_fp8=args.expected_fp8_cb_moe_layers,
        expected_cfg=args.persistent_b_cfg,
    )

    exact_sampling = engine.quality_sampling
    coarse_sampling = bootstrap.sampling_params_class(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=args.top_k,
        flat_logprobs=False,
        detokenize=False,
    )
    pilot_prompts, pilot_profile = _full_vocab_profile(
        bootstrap.prompts, seqlen=args.pilot_full_vocab_seqlen
    )
    confirmation_prompts, confirmation_profile = _full_vocab_profile(
        bootstrap.prompts, seqlen=args.full_vocab_seqlen
    )

    llm = engine.llm
    synchronize = torch.cuda.synchronize
    warmup_records: list[dict[str, Any]] = []
    selector_records: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    quality_triplets: list[dict[str, Any]] = []
    measured_request_index = 0

    _warm_profile(
        profile="coarse_512",
        prompts=bootstrap.prompts,
        sampling=coarse_sampling,
        repeats=args.warmup_triplets,
        llm=llm,
        controller=controller,
        synchronize=synchronize,
        warmup_records=warmup_records,
        selector_records=selector_records,
        route_records=route_records,
    )

    for prompt_index, prompt_ids in enumerate(bootstrap.prompts):
        scores: dict[str, Any] = {}
        routes: dict[str, Any] = {}
        selectors: dict[str, Any] = {}
        walls: dict[str, float] = {}
        request_sequence: list[int] = []
        for role, arm in _TRIPLET_REQUESTS:
            request_sequence.append(measured_request_index)
            measured_request_index += 1
            output, wall_ms, selector, route = _run_generate(
                llm=llm,
                sampling=coarse_sampling,
                prompt_ids=prompt_ids,
                arm=arm,
                label=f"quality:{prompt_index}:{role}",
                controller=controller,
                synchronize=synchronize,
            )
            selector_records.append(selector)
            route_records.append(route)
            selectors[role] = selector
            routes[role] = route
            walls[role] = wall_ms
            scores[role] = v5.score_prompt_output(
                output,
                prompt_ids,
                args.top_k,
                full_vocab=False,
                expected_vocab_size=None,
            )
            del output
        quality_triplets.append({
            "profile": "coarse_512",
            "prompt_index": prompt_index,
            "block_index": prompt_index,
            "source_prompt_index": prompt_index,
            "request_order": list(_TRIPLET_ROLES),
            "arm_order": [arm for _role, arm in _TRIPLET_REQUESTS],
            "request_sequence": request_sequence,
            "request_pids": {
                role: selectors[role]["pid"] for role in _TRIPLET_ROLES
            },
            "scores": scores,
            "wall_ms": walls,
            "routes": routes,
        })

    nll_prompts = _nll_prompt_metrics(quality_triplets)

    # The 8x64 exact profile is only a rejection screen.  Warm it after the
    # coarse phase so its first measured triplet cannot inherit a stale warmup.
    _warm_profile(
        profile="exact_pilot_full_vocab_64",
        prompts=pilot_prompts,
        sampling=exact_sampling,
        repeats=args.warmup_triplets,
        llm=llm,
        controller=controller,
        synchronize=synchronize,
        warmup_records=warmup_records,
        selector_records=selector_records,
        route_records=route_records,
    )
    pilot_triplets, pilot_prompt_metrics, measured_request_index = (
        _measure_exact_profile(
            profile="exact_pilot_full_vocab_64",
            prompts=pilot_prompts,
            sampling=exact_sampling,
            llm=llm,
            controller=controller,
            synchronize=synchronize,
            expected_vocab_size=bootstrap.candidate_vocab_size,
            first_request_index=measured_request_index,
            selector_records=selector_records,
            route_records=route_records,
        )
    )

    base_integrity_gates = {
        "model_contract": model_gate,
        "loaded_layer_inventory": controller.inventory_gate,
        "chunked_prefill_execution_contract": (
            validation_common.chunked_prefill_integrity_gate(chunked_contract)
        ),
        "same_process_execution": {
            "expected_pid": os.getpid(),
            "runtime_pid": runtime["pid"],
            "pass": runtime["pid"] == os.getpid(),
        },
    }
    pilot_core_gates = {
        **base_integrity_gates,
        "selector_switch_attested": _selector_gate(selector_records),
        "route_and_fallback_attested": _route_gate(route_records),
        "coarse_and_pilot_profiles_warmed": _warmup_gate(
            warmup_records,
            profiles=("coarse_512", "exact_pilot_full_vocab_64"),
            triplets_per_profile=args.warmup_triplets,
        ),
        "coarse_B_PB_B_order_and_cardinality": _triplet_order_gate(
            quality_triplets,
            n_prompts=len(bootstrap.prompts),
            expected_pid=os.getpid(),
            first_request_index=0,
        ),
        "pilot_exact_B_PB_B_order_and_cardinality": _triplet_order_gate(
            pilot_triplets,
            n_prompts=len(pilot_prompts),
            expected_pid=os.getpid(),
            first_request_index=len(bootstrap.prompts) * 3,
        ),
        "fp8_routes_unchanged_within_pilot_triplets": _fp8_invariance_gate(
            [*quality_triplets, *pilot_triplets]
        ),
    }
    pilot_measurement_valid = all(
        bool(gate.get("pass")) for gate in pilot_core_gates.values()
    )
    pilot_quality = _promotion_quality_v6(
        nll_prompts,
        pilot_prompt_metrics,
        integrity_pass=pilot_measurement_valid,
    )
    pilot_quality["stage"] = "exact_8x64_rejection_screen"
    pilot_quality["can_promote"] = False
    pilot_quality["arm_labels"] = dict(ARM_LABELS)

    confirmation_triplets: list[dict[str, Any]] = []
    confirmation_prompt_metrics: list[dict[str, Any]] = []
    # ---- v7 AMENDMENT (2026-08-14), ESCALATION LOGIC ONLY ----------------
    # v6: confirmation_executed = pilot_quality["decision"] != "FAIL"
    #
    # The 8x64 screen hard-rejected Persistent-B while its OWN
    # baseline_noise_resolution gate reported pass:false
    # (baseline_difference_rms 0.02716 vs maximum_resolvable 0.00860), and
    # the report listed "baseline_noise_exceeds_resolution_limit" as an
    # inconclusive reason. A screen that cannot resolve its own baseline has
    # not resolved anything, and this project's standing rule is that cheap
    # screens are triage and never final selection.
    #
    # v7 therefore ALWAYS measures both stages, so the mandatory 8x512
    # confirmation evidence exists instead of being skipped.
    #
    # NO QUALITY THRESHOLD IS CHANGED. Every gate keeps its predeclared
    # limit and must still pass on its own terms. Both stages are reported
    # separately; a confirmation-stage PASS alongside a pilot-stage FAIL is
    # a result to adjudicate, NOT an automatic promotion.
    confirmation_executed = True
    if confirmation_executed:
        # The complete 8x512 profile is the only distribution measurement that
        # can promote.  Rewarm it immediately before measuring it; the pilot's
        # 24 requests are deliberately not treated as its warmup.
        _warm_profile(
            profile="exact_confirmation_full_vocab_512",
            prompts=confirmation_prompts,
            sampling=exact_sampling,
            repeats=args.warmup_triplets,
            llm=llm,
            controller=controller,
            synchronize=synchronize,
            warmup_records=warmup_records,
            selector_records=selector_records,
            route_records=route_records,
        )
        (
            confirmation_triplets,
            confirmation_prompt_metrics,
            measured_request_index,
        ) = _measure_exact_profile(
            profile="exact_confirmation_full_vocab_512",
            prompts=confirmation_prompts,
            sampling=exact_sampling,
            llm=llm,
            controller=controller,
            synchronize=synchronize,
            expected_vocab_size=bootstrap.candidate_vocab_size,
            first_request_index=measured_request_index,
            selector_records=selector_records,
            route_records=route_records,
        )

    if confirmation_executed:
        core_gates = {
            **base_integrity_gates,
            "selector_switch_attested": _selector_gate(selector_records),
            "route_and_fallback_attested": _route_gate(route_records),
            "all_measured_profiles_warmed_immediately_before_use": _warmup_gate(
                warmup_records,
                profiles=(
                    "coarse_512",
                    "exact_pilot_full_vocab_64",
                    "exact_confirmation_full_vocab_512",
                ),
                triplets_per_profile=args.warmup_triplets,
            ),
            "coarse_B_PB_B_order_and_cardinality": pilot_core_gates[
                "coarse_B_PB_B_order_and_cardinality"
            ],
            "pilot_exact_B_PB_B_order_and_cardinality": pilot_core_gates[
                "pilot_exact_B_PB_B_order_and_cardinality"
            ],
            "confirmation_exact_B_PB_B_order_and_cardinality": (
                _triplet_order_gate(
                    confirmation_triplets,
                    n_prompts=len(confirmation_prompts),
                    expected_pid=os.getpid(),
                    first_request_index=(
                        len(bootstrap.prompts) * 3 + len(pilot_prompts) * 3
                    ),
                )
            ),
            "fp8_routes_unchanged_within_all_triplets": _fp8_invariance_gate(
                [
                    *quality_triplets,
                    *pilot_triplets,
                    *confirmation_triplets,
                ]
            ),
            "complete_8x512_exact_confirmation": (
                _exact_profile_completion_gate(
                    confirmation_prompt_metrics,
                    n_prompts=_SEALED_PROMPT_COUNT,
                    seqlen=_EXACT_CONFIRMATION_SEQLEN,
                    executed=True,
                )
            ),
        }
        measurement_valid = all(
            bool(gate.get("pass")) for gate in core_gates.values()
        )
        quality = _promotion_quality_v6(
            nll_prompts,
            confirmation_prompt_metrics,
            integrity_pass=measurement_valid,
        )
        quality["stage"] = "exact_8x512_promotion_confirmation"
        quality["confirmation_complete"] = True
    else:
        core_gates = {
            **pilot_core_gates,
            "complete_8x512_exact_confirmation": {
                **_exact_profile_completion_gate(
                    (),
                    n_prompts=_SEALED_PROMPT_COUNT,
                    seqlen=_EXACT_CONFIRMATION_SEQLEN,
                    executed=False,
                ),
                "skip_reason": "8x64 rejection screen returned hard FAIL",
            },
        }
        measurement_valid = False
        quality = {
            **pilot_quality,
            "stage": "exact_8x64_hard_rejection",
            "confirmation_complete": False,
            "pass": False,
        }
    quality["arm_labels"] = dict(ARM_LABELS)

    teacher_arm_scores = _teacher_arm_scores(
        quality_triplets,
        n_prompts=len(bootstrap.prompts),
    )
    teacher_quality, teacher_record = validation_common.score_teacher(
        bootstrap,
        args,
        arm_scores=teacher_arm_scores,
        arms=ARMS,
        helpers=v5,
    )
    if teacher_quality is not None:
        teacher_quality["arm_labels"] = dict(ARM_LABELS)
        teacher_quality["delta"] = {
            "teacher_to_persistent_b_mean_kl_minus_bridge": (
                teacher_quality["fused"]["kl_reference_to_candidate"]["mean"]
                - teacher_quality["baseline"][
                    "kl_reference_to_candidate"
                ]["mean"]
            ),
            "kl_mode": bootstrap.quality_kl_mode,
        }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": v5._utc_now(),
        "scope": (
            "DSV4 routed FP4-CB quality prefill; TP=1; one in-process vLLM "
            "engine; warmed consecutive B-PB-B prompt blocks"
        ),
        "arm_labels": dict(ARM_LABELS),
        "activation_contract": {
            "baseline": "exact native group-16 FP4 activation QDQ",
            "fused": "exact native group-16 FP4 activation QDQ",
            "weights": "identical packed FP4-CB v2 bytes and codebooks",
            "same_contract": True,
            "permitted_difference": "FP32 GEMM reduction order only",
        },
        "settings": validation_common.shared_report_settings(
            bootstrap,
            args,
            arm_settings={
                "persistent_b_cfg": args.persistent_b_cfg,
                "concrete_arm_modes": dict(ARM_LABELS),
                "expected_persistent_b_layers": (
                    args.expected_persistent_b_layers
                ),
                "expected_fp8_cb_moe_layers": args.expected_fp8_cb_moe_layers,
            },
            inherited_prefill=inherited[v5.PREFILL_ENV],
            prefill_threshold=moe.MOE_PREFILL_M_THRESHOLD,
            measurement_settings={
                "chunked_prefill": chunked_contract["resolved_enabled"],
                "promotion_protocol": _PROMOTION_PROTOCOL,
                "warmup_triplets": args.warmup_triplets,
                "coarse_measured_requests": len(quality_triplets) * 3,
                "pilot_exact_measured_requests": len(pilot_triplets) * 3,
                "confirmation_exact_measured_requests": (
                    len(confirmation_triplets) * 3
                ),
                "full_vocab_kl": True,
                "pilot_full_vocab_seqlen": args.pilot_full_vocab_seqlen,
                "full_vocab_seqlen": args.full_vocab_seqlen,
                "pilot_full_vocab_profile": pilot_profile,
                "confirmation_full_vocab_profile": confirmation_profile,
                "confirmation_executed": confirmation_executed,
                "full_vocab_storage_contract": (
                    "vLLM CPU LogprobsTensors -> bounded tensor transport -> "
                    "three compact float32 scores -> prompt-unit exact "
                    "symmetric KL plus digest evidence; raw rows released "
                    "before the next prompt triplet"
                ),
                "chunked_prefill_contract": chunked_contract,
            },
        ),
        "environment_contract": {
            "inherited_before_sanitization": inherited,
            "during_model_load": {
                PERSISTENT_B_ENV: os.environ.get(PERSISTENT_B_ENV),
                PERSISTENT_B_CFG_ENV: os.environ.get(PERSISTENT_B_CFG_ENV),
                BF16_SM120_ENV: os.environ.get(BF16_SM120_ENV),
                v5.FUSED_ENV: os.environ.get(v5.FUSED_ENV),
                v5.FUSED_MOE_ENV: os.environ.get(v5.FUSED_MOE_ENV),
                v5.PREFILL_ENV: os.environ.get(v5.PREFILL_ENV),
            },
            "production_selector_remained_enabled": True,
            "measurement_exception": (
                "only the already-attested layer._cb_moe_persistent_b handle "
                "is scoped per arm; no extension is loaded or selector read "
                "inside a measured request"
            ),
        },
        "runtime": runtime,
        "extension": bootstrap.extension,
        "candidate_artifact_provenance": (
            bootstrap.candidate_artifact_provenance
        ),
        "dataset": bootstrap.dataset,
        "model_load_seconds": engine.model_load_seconds,
        "warmup": warmup_records,
        "coarse_prompt_evidence": nll_prompts,
        "pilot_quality": pilot_quality,
        "pilot_measurement_valid": pilot_measurement_valid,
        "quality": quality,
        "full_vocab_streaming_evidence": {
            "pilot_rejection_screen": {
                "profile": pilot_profile,
                "triplets": pilot_triplets,
            },
            "promotion_confirmation": (
                {
                    "profile": confirmation_profile,
                    "triplets": confirmation_triplets,
                }
                if confirmation_executed else None
            ),
        },
        "teacher": teacher_record,
        "teacher_quality": teacher_quality,
        "dispatch": {
            "layer_inventory": controller.inventory_gate,
            "selector_attestations": selector_records,
            "route_attestations": route_records,
        },
        "core_integrity_gates": core_gates,
        "limitations": [
            (
                "The 8x64 exact full-vocabulary profile is a rejection screen "
                "only. PROMOTION PASS additionally requires cardinality-attested "
                "half-Jeffreys measurements over every scored position of the "
                "complete sealed 8x512 profile."
            ),
            "A passing quality run does not replace served NATIVE-PARITY gates.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    report["measurement_valid"] = measurement_valid
    report["configured_gates_pass"] = quality["pass"]
    report["promotion_decision"] = quality["decision"]
    report["promotion_contract"] = {
        "protocol": _PROMOTION_PROTOCOL,
        "schema_transition": {
            "current_schema": SCHEMA,
            "retired": (
                "v1/v5 exact repeat-equality and repeated counterbalanced "
                "pair design"
            ),
            "replacement": (
                "warmed same-PID consecutive B-PB-B blocks with prompt-unit "
                "uncertainty and baseline-noise resolution"
            ),
        },
        "same_weight_and_activation_contract": True,
        "same_pid": core_gates["same_process_execution"]["pass"],
        "warmed": (
            core_gates.get("all_measured_profiles_warmed_immediately_before_use")
            or core_gates.get("coarse_and_pilot_profiles_warmed")
        ),
        "per_prompt_request_order": list(_TRIPLET_ROLES),
        "per_prompt_concrete_arm_order": [
            arm for _role, arm in _TRIPLET_REQUESTS
        ],
        "coarse_request_contract": "8 prompts * B-PB-B = 24",
        "coarse_measured_requests": len(quality_triplets) * 3,
        "pilot_exact_request_contract": "8x64 prefixes * B-PB-B = 24",
        "pilot_exact_measured_requests": len(pilot_triplets) * 3,
        "confirmation_exact_request_contract": (
            "8x512 complete windows * B-PB-B = 24; mandatory unless the "
            "8x64 rejection screen hard-FAILs"
        ),
        "confirmation_exact_measured_requests": (
            len(confirmation_triplets) * 3
        ),
        "confirmation_executed": confirmation_executed,
        "pilot_decision": pilot_quality["decision"],
        "predeclared_sealed_profiles": {
            "full_8x512": {
                "semantic_sha256": bootstrap.dataset["semantic_sha256"],
                "token_ids_tensor_sha256": bootstrap.dataset[
                    "prompt_token_ids_tensor_sha256"
                ],
                "prompt_sha256": bootstrap.dataset["prompt_window_sha256"],
            },
            "exact_full_vocab_pilot_8x64": pilot_profile,
            "exact_full_vocab_confirmation_8x512": confirmation_profile,
        },
        "statistical_unit": "sealed_prompt_block_not_token_row",
        "served_native_parity_still_required": True,
        "complete": (
            len(quality_triplets) == _SEALED_PROMPT_COUNT
            and len(pilot_triplets) == _SEALED_PROMPT_COUNT
            and len(confirmation_triplets) == _SEALED_PROMPT_COUNT
            and measurement_valid
        ),
    }
    report["promotion_recommendation"] = quality["decision"]
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--campaign-id",
        type=_campaign_id_arg,
        required=True,
        help=(
            "predeclared immutable release-campaign identity; one attempt is "
            "permitted in an evidence directory"
        ),
    )
    parser.add_argument("--prompt-token-ids-json", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-revision")
    parser.add_argument("--teacher-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--full-vocab-kl",
        action="store_true",
        default=True,
        help=(
            "compatibility spelling; v6 always runs the required exact "
            "full-vocabulary 8x64 rejection and 8x512 confirmation stages"
        ),
    )
    parser.add_argument(
        "--pilot-full-vocab-seqlen",
        type=v5._positive_int,
        default=64,
        help="fixed rejection-only exact full-vocabulary prefix length",
    )
    parser.add_argument(
        "--full-vocab-seqlen",
        type=v5._positive_int,
        default=512,
        help=(
            "fixed complete-window exact-vocabulary confirmation length; "
            "PROMOTION PASS requires all 512 sealed tokens"
        ),
    )
    parser.add_argument("--n-samples", type=v5._positive_int, default=8)
    parser.add_argument("--seqlen", type=v5._positive_int, default=512)
    parser.add_argument("--top-k", type=v5._positive_int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="gridbook")
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument(
        "--kv-cache-memory-bytes", type=v5._positive_int, default=268435456
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=(DSV4_KV_CACHE_DTYPE,),
        default=DSV4_KV_CACHE_DTYPE,
        help="fixed DSV4 sparse-MLA KV-cache dtype",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=v5._unit_interval, default=0.90
    )
    parser.add_argument("--max-num-batched-tokens", type=v5._positive_int)
    chunked = parser.add_mutually_exclusive_group()
    chunked.add_argument(
        "--enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_const",
        const=True,
        default=None,
    )
    chunked.add_argument(
        "--disable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_const",
        const=False,
    )
    parser.add_argument(
        "--warmup-triplets",
        type=v5._positive_int,
        default=1,
        help="separate B-PB-B warmups per coarse and exact request profile",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--persistent-b-cfg", type=v5._nonnegative_int, default=0)
    parser.add_argument("--expected-persistent-b-layers", type=v5._positive_int,
                        default=35)
    parser.add_argument("--expected-fp8-cb-moe-layers", type=v5._positive_int,
                        default=8)
    parser.add_argument("--expected-architecture",
                        default="DeepseekV4ForCausalLM")
    parser.add_argument("--expected-model-type", default="deepseek_v4")
    parser.add_argument("--expected-hidden-layers", type=v5._positive_int,
                        default=43)
    parser.add_argument("--expected-hidden-size", type=v5._positive_int,
                        default=4096)
    parser.add_argument("--expected-moe-intermediate-size", type=v5._positive_int,
                        default=2048)
    parser.add_argument("--expected-routed-experts", type=v5._positive_int,
                        default=256)
    parser.add_argument("--expected-vocab-size", type=v5._positive_int,
                        default=129280)
    args = parser.parse_args(argv)

    # Fields consumed by the shared loader/report API.  They are explicit here
    # rather than hidden in a second parser with unrelated fused-NVFP4 modes.
    args.mode = "moe256"
    args.teacher_full_vocab_kl = True
    args.dataset_cache_dir = None
    args.dataset_split = "train"
    args.wikitext_text = None

    if args.seqlen != _SEALED_PROMPT_SEQLEN:
        parser.error(
            f"v6 requires the sealed {_SEALED_PROMPT_COUNT}x"
            f"{_SEALED_PROMPT_SEQLEN} full-window profile"
        )
    if args.pilot_full_vocab_seqlen != _EXACT_PILOT_SEQLEN:
        parser.error(
            f"v6 requires the predeclared {_SEALED_PROMPT_COUNT}x"
            f"{_EXACT_PILOT_SEQLEN} exact full-vocabulary rejection profile"
        )
    if args.full_vocab_seqlen != _EXACT_CONFIRMATION_SEQLEN:
        parser.error(
            f"v6 requires the predeclared {_SEALED_PROMPT_COUNT}x"
            f"{_EXACT_CONFIRMATION_SEQLEN} exact full-vocabulary confirmation "
            "profile"
        )
    if args.n_samples != _SEALED_PROMPT_COUNT:
        parser.error(
            f"v6 requires exactly {_SEALED_PROMPT_COUNT} sealed prompt blocks"
        )
    candidate_is_local = Path(args.model).expanduser().is_dir()
    if not candidate_is_local:
        parser.error("persistent-B DSV4 validation requires a local --model")
    teacher_is_local = bool(
        args.teacher_model and Path(args.teacher_model).expanduser().is_dir()
    )
    if args.teacher_model and not teacher_is_local and not args.teacher_revision:
        parser.error(
            "a non-local --teacher-model requires an explicit --teacher-revision"
        )
    if args.teacher_model:
        parser.error(
            "streamed triplet full-vocab scoring does not retain candidate rows "
            "for a later teacher pass; run the identity-matched teacher as a "
            "separate predeclared gate"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reservation = _reserve_attempt(args.output, args.campaign_id)
    except Exception as exc:  # noqa: BLE001 - do not touch prior evidence
        refusal = {
            "schema": SCHEMA,
            "status": "attempt_refused",
            "promotion_decision": "FAIL",
            "decision_reason": "immutable_attempt_reservation_failed",
            "campaign_id": args.campaign_id,
            "output": str(args.output),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(refusal, indent=2), file=sys.stderr, flush=True)
        return 1
    args.output = reservation.output
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - preserve machine-readable failure
        failure = {
            "schema": SCHEMA,
            "created_at": v5._utc_now(),
            "status": "gate_failed",
            "promotion_decision": "FAIL",
            "decision_reason": "integrity_violation",
            "attempt": dict(reservation.attempt),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        }
        try:
            _finalize_attempt(reservation, failure)
        except Exception as write_exc:  # noqa: BLE001 - reservation stays visible
            failure["finalization_error"] = {
                "type": type(write_exc).__name__,
                "message": str(write_exc),
            }
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1
    if report["promotion_decision"] == "FAIL":
        report["status"] = "gate_failed"
    elif report["promotion_decision"] == "INCONCLUSIVE":
        report["status"] = "inconclusive"
    else:
        report["status"] = "ok"
    report["promotion_contract"]["immutable_attempt"] = {
        "campaign_id": args.campaign_id,
        "attempt_id": reservation.attempt["attempt_id"],
        "single_attempt": True,
        "prior_output_overwrite_refused": True,
    }
    try:
        _finalize_attempt(reservation, report)
    except Exception as exc:  # noqa: BLE001 - never bypass reservation ownership
        print(json.dumps({
            "schema": SCHEMA,
            "status": "gate_failed",
            "promotion_decision": "FAIL",
            "decision_reason": "immutable_attempt_finalization_failed",
            "attempt": dict(reservation.attempt),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, indent=2), file=sys.stderr, flush=True)
        return 1
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_decision": report["promotion_decision"],
        "promotion_contract": report["promotion_contract"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_gates": report["quality"],
        "layer_inventory": report["dispatch"]["layer_inventory"],
    }, indent=2), flush=True)
    return {"ok": 0, "gate_failed": 2, "inconclusive": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
