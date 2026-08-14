#!/usr/bin/env python3
"""Same-engine quality A/B for the persistent-B routed FP4-CB lane.

This is the whole-model quality gate for the opt-in
``PRISMAQUANT_CB_MOE_PERSISTENT_B`` schedule.  It loads the extension and the
candidate exactly once with persistent-B resolved for every compatible FP4-CB
MoE layer, then interleaves two arms over fixed, pre-tokenized WikiText windows:

* ``baseline`` is Gridbook's default exact CB-to-BF16 expansion plus owned
  grouped-BF16 bridge; and
* ``fused`` is persistent-B's decode-in-mainloop schedule.

Those report arm names intentionally match ``validate_fused_nvfp4_ab.py`` so
its established scoring and threshold machinery can be reused without a
second numerical implementation.  ``arm_labels`` always makes their concrete
meaning explicit.  Both arms consume the same packed weights and exact native
group-16 activation QDQ; only the FP32 GEMM reduction order changes.

The model is never reloaded.  The harness keeps the production selector at
``1`` and temporarily swaps only the already-attested per-layer extension
handle.  Before every request it clears latest-route telemetry, then requires
all loaded FP4-CB MoE layers to report the requested symbol and every loaded
FP8-CB MoE layer to report a served route.  FP8 routes must be byte-for-byte
identical between each paired request.  Layer counts and the DeepSeek-V4
architecture contract are machine gates, not log-inspection suggestions.

Target-token NLL/PPL are always exact.  Paired KL is top-K/coarse by default;
``--full-vocab-kl`` retains the complete sealed 8x512 coarse/NLL stage and
adds an exact-vocabulary stage over a digest-attested prefix of every window
(64 tokens by default).  vLLM's flat representation is compacted one arm at a
time; each pair is scored immediately through the shared v5 accumulator and
only scalar metrics plus row/score digests survive.  A coarse-only run can
reject a lane but cannot promote it.  Repeated arms are required and exact
repeat digests are a core gate.  An optional identity-matched BF16 teacher uses
the same shared teacher scorer in coarse mode, but is not required for a
transparent same-contract schedule A/B.

Run this script in a fresh Python process inside the pinned Gridbook/vLLM CUDA
image.  Example for the exact dsv4flash artifact::

    python3 scripts/validate_moe_persistent_b_ab.py \
      --model /models/dsv4flash0731 \
      --prompt-token-ids-json /evidence/dsv4-wikitext-inputs-v1.json \
      --output /evidence/dsv4-persistent-b-ab.json \
      --enable-chunked-prefill --max-num-batched-tokens 256 \
      --top-k 256 --full-vocab-kl \
      --quality-repeats 2 --timing-repeats 2 \
      --max-mean-kl 1e-4 --max-mean-nll-regression 0.00498754 \
      --max-ppl-relative-regression 0.005

A passing report remains candidate evidence; served TTFT/ITL/TPS and memory
headroom are separate NATIVE-PARITY gates.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import json
import math
import os
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

SCHEMA = "gridbook.moe-persistent-b-ab.v1"
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
_PAIRWISE_QUALITY_GATES = (
    "max_mean_kl",
    "max_mean_nll_regression",
    "max_ppl_relative_regression",
)


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


@dataclass(frozen=True)
class _CompactFullVocabScore:
    """One exact score in float32 rows plus stable evidence digests."""

    target_logprobs: tuple[float, ...]
    _row_values: array.array
    vocab_size: int
    prompt_token_ids_sha256: str
    row_sha256: tuple[str, ...]
    score_sha256: str

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
        }


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


def _pair_order_gate(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {"baseline/fused": 0, "fused/baseline": 0}
    unexpected: list[list[str]] = []
    per_prompt: dict[int, dict[str, int]] = {}
    missing_prompt_identity: list[int] = []
    for pair_index, pair in enumerate(pairs):
        key = "/".join(pair["pair_order"])
        if key in counts:
            counts[key] += 1
        else:
            unexpected.append(list(pair["pair_order"]))
            continue
        source_prompt = pair.get("source_prompt_index")
        if isinstance(source_prompt, bool) or not isinstance(source_prompt, int):
            missing_prompt_identity.append(pair_index)
            continue
        prompt_counts = per_prompt.setdefault(
            source_prompt, {"baseline/fused": 0, "fused/baseline": 0}
        )
        prompt_counts[key] += 1
    prompts_missing_crossover = [
        prompt_index
        for prompt_index, prompt_counts in sorted(per_prompt.items())
        if not all(prompt_counts.values())
    ]
    return {
        "counts": counts,
        "unexpected": unexpected,
        "per_source_prompt": {
            str(prompt_index): prompt_counts
            for prompt_index, prompt_counts in sorted(per_prompt.items())
        },
        "missing_source_prompt_identity_blocks": missing_prompt_identity,
        "source_prompts_missing_both_orders": prompts_missing_crossover,
        "pass": (
            not unexpected
            and not missing_prompt_identity
            and bool(per_prompt)
            and not prompts_missing_crossover
            and counts["baseline/fused"] > 0
            and counts["baseline/fused"] == counts["fused/baseline"]
        ),
    }


def _repeat_determinism_gate(
    pairs: Sequence[Mapping[str, Any]], *, n_prompts: int, repeats: int
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for prompt_index in range(n_prompts):
        prompt_pairs = [
            pair for pair in pairs if pair["source_prompt_index"] == prompt_index
        ]
        if len(prompt_pairs) != repeats:
            mismatches.append({
                "prompt_index": prompt_index,
                "reason": f"observed {len(prompt_pairs)} repeats, expected {repeats}",
            })
            continue
        for arm in ARMS:
            reference = prompt_pairs[0]["scores"][arm]
            for repeat_index, pair in enumerate(prompt_pairs[1:], start=1):
                if pair["scores"][arm] != reference:
                    mismatches.append({
                        "prompt_index": prompt_index,
                        "arm": arm,
                        "repeat_index": repeat_index,
                        "reason": "PromptScore differs from repeat zero",
                    })
    return {
        "required_repeats": repeats,
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _full_vocab_repeat_determinism_gate(
    pairs: Sequence[Mapping[str, Any]], *, n_prompts: int, repeats: int
) -> dict[str, Any]:
    """Compare compact bit digests, never cardinality-sized PromptScores."""

    mismatches: list[dict[str, Any]] = []
    for prompt_index in range(n_prompts):
        prompt_pairs = sorted(
            (
                pair for pair in pairs
                if pair["source_prompt_index"] == prompt_index
            ),
            key=lambda pair: pair["repeat_index"],
        )
        if len(prompt_pairs) != repeats:
            mismatches.append({
                "prompt_index": prompt_index,
                "reason": f"observed {len(prompt_pairs)} repeats, expected {repeats}",
            })
            continue
        for arm in ARMS:
            reference = prompt_pairs[0]["score_digests"][arm]
            for pair in prompt_pairs[1:]:
                observed = pair["score_digests"][arm]
                if (
                    observed["score_sha256"] != reference["score_sha256"]
                    or observed["row_sha256"] != reference["row_sha256"]
                    or observed["prompt_token_ids_sha256"]
                    != reference["prompt_token_ids_sha256"]
                ):
                    mismatches.append({
                        "prompt_index": prompt_index,
                        "arm": arm,
                        "repeat_index": pair["repeat_index"],
                        "expected_score_sha256": reference["score_sha256"],
                        "observed_score_sha256": observed["score_sha256"],
                        "reason": "full-vocabulary score/row digest differs",
                    })
    return {
        "required_repeats": repeats,
        "digest_schema": "gridbook.compact-full-vocab-score.v1",
        "comparison": "exact prompt, target-float64, and row-float32 digests",
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _fp8_invariance_gate(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for pair in pairs:
        baseline = pair["routes"]["baseline"]["fp8_routes"]
        fused = pair["routes"]["fused"]["fp8_routes"]
        if baseline != fused:
            mismatches.append({
                "block_index": pair["block_index"],
                "source_prompt_index": pair.get("source_prompt_index"),
                "baseline": baseline,
                "fused": fused,
            })
    return {
        "paired_blocks": len(pairs),
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


def _first_repeat_scores(
    pairs: Sequence[Mapping[str, Any]], *, n_prompts: int
) -> dict[str, list[Any]]:
    result = {arm: [] for arm in ARMS}
    for prompt_index in range(n_prompts):
        matches = [
            pair for pair in pairs
            if pair["source_prompt_index"] == prompt_index
            and pair["repeat_index"] == 0
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"prompt {prompt_index} has {len(matches)} repeat-zero blocks"
            )
        for arm in ARMS:
            result[arm].append(matches[0]["scores"][arm])
    return result


def _promotion_quality_summary(
    coarse: Mapping[str, Any],
    exact: Mapping[str, Any],
    *,
    full_vocab_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind full-workload NLL/PPL to streamed exact-profile KL.

    Pairwise target logprobs do not depend on how many non-target logprobs are
    returned.  We nevertheless source NLL/PPL and their regression gates from
    the complete sealed 8x512 screen, while the KL fields come only from the
    cardinality-attested exact stage.  Both component reports remain embedded
    so the metric provenance cannot be mistaken.
    """

    result = {
        "arms": dict(coarse["arms"]),
        "delta": dict(coarse["delta"]),
        "per_prompt": list(coarse["per_prompt"]),
        "kl_mode": v5.KL_FULL_VOCAB,
        "kl_convention": exact["kl_convention"],
        "metric_sources": {
            "target_nll_ppl_and_regression": (
                "coarse_full_sealed_8x512_stage"
            ),
            "pairwise_kl": "streamed_exact_full_vocab_profile_stage",
        },
        "coarse_full_workload": dict(coarse),
        "exact_full_vocab_profile": {
            "profile": dict(full_vocab_profile),
            "quality": dict(exact),
        },
    }
    for key in (
        "kl_baseline_to_fused",
        "kl_fused_to_baseline",
        "kl_baseline_to_fused_confident_positions",
    ):
        result["delta"][key] = exact["delta"][key]
    return result


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
    torch = bootstrap.torch
    moe = bootstrap.moe
    runtime = bootstrap.runtime
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

    coarse_sampling = engine.quality_sampling
    exact_sampling = None
    exact_prompts: list[list[int]] = []
    exact_profile: dict[str, Any] | None = None
    exact_accumulator: MutableMapping[str, Any] | None = None
    if args.full_vocab_kl:
        exact_sampling = engine.quality_sampling
        coarse_sampling = bootstrap.sampling_params_class(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=args.top_k,
            flat_logprobs=False,
            detokenize=False,
        )
        exact_prompts, exact_profile = _full_vocab_profile(
            bootstrap.prompts, seqlen=args.full_vocab_seqlen
        )
        exact_accumulator = v5._new_quality_accumulator(
            kl_mode=v5.KL_FULL_VOCAB
        )

    llm = engine.llm
    synchronize = torch.cuda.synchronize
    warmup_records: list[dict[str, Any]] = []
    selector_records: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    quality_pairs: list[dict[str, Any]] = []
    full_vocab_pairs: list[dict[str, Any]] = []
    timing_pairs: list[dict[str, Any]] = []
    timing_samples: dict[str, list[float]] = {arm: [] for arm in ARMS}

    for repeat in range(args.warmup_pairs):
        order = v5.paired_arm_order(repeat)
        routes: dict[str, Any] = {}
        for arm in order:
            output, wall_ms, selector, route = _run_generate(
                llm=llm,
                sampling=engine.timing_sampling,
                prompt_ids=bootstrap.prompts[repeat % len(bootstrap.prompts)],
                arm=arm,
                label=f"warmup:{repeat}:{arm}",
                controller=controller,
                synchronize=synchronize,
            )
            selector_records.append(selector)
            route_records.append(route)
            routes[arm] = route
            warmup_records.append({
                "repeat_index": repeat,
                "arm": arm,
                "wall_ms": wall_ms,
                "selector": selector,
                "routes": route,
            })
            del output

    for phase in validation_common.measurement_phase_order(args.timing_repeats):
        if phase == "timing":
            validation_common.quiesce_before_timing(torch)
            for repeat in range(args.timing_repeats):
                for prompt_index, prompt_ids in enumerate(bootstrap.prompts):
                    block_index = repeat * len(bootstrap.prompts) + prompt_index
                    # Cross each source prompt over both arm orders across
                    # repeats. Using block_index here aliases repeat parity when
                    # n_prompts is even, permanently confounding prompt and
                    # order despite an aggregate 50/50 count.
                    order = v5.paired_arm_order(repeat + prompt_index)
                    routes: dict[str, Any] = {}
                    walls: dict[str, float] = {}
                    for arm in order:
                        output, wall_ms, selector, route = _run_generate(
                            llm=llm,
                            sampling=engine.timing_sampling,
                            prompt_ids=prompt_ids,
                            arm=arm,
                            label=f"timing:{repeat}:{prompt_index}:{arm}",
                            controller=controller,
                            synchronize=synchronize,
                        )
                        selector_records.append(selector)
                        route_records.append(route)
                        routes[arm] = route
                        walls[arm] = wall_ms
                        timing_samples[arm].append(wall_ms)
                        del output
                    timing_pairs.append({
                        "block_index": block_index,
                        "repeat_index": repeat,
                        "source_prompt_index": prompt_index,
                        "pair_order": list(order),
                        "wall_ms": walls,
                        "routes": routes,
                    })
            continue
        if phase != "quality":
            raise RuntimeError(f"unknown measurement phase {phase!r}")
        for repeat in range(args.quality_repeats):
            for prompt_index, prompt_ids in enumerate(bootstrap.prompts):
                block_index = repeat * len(bootstrap.prompts) + prompt_index
                order = v5.paired_arm_order(repeat + prompt_index)
                scores: dict[str, Any] = {}
                routes: dict[str, Any] = {}
                walls: dict[str, float] = {}
                for arm in order:
                    output, wall_ms, selector, route = _run_generate(
                        llm=llm,
                        sampling=coarse_sampling,
                        prompt_ids=prompt_ids,
                        arm=arm,
                        label=f"quality:{repeat}:{prompt_index}:{arm}",
                        controller=controller,
                        synchronize=synchronize,
                    )
                    selector_records.append(selector)
                    route_records.append(route)
                    routes[arm] = route
                    walls[arm] = wall_ms
                    scores[arm] = v5.score_prompt_output(
                        output,
                        prompt_ids,
                        args.top_k,
                        full_vocab=False,
                        expected_vocab_size=None,
                    )
                    del output
                quality_pairs.append({
                    "prompt_index": block_index,
                    "block_index": block_index,
                    "repeat_index": repeat,
                    "source_prompt_index": prompt_index,
                    "pair_order": list(order),
                    "scores": scores,
                    "wall_ms": walls,
                    "routes": routes,
                })

        if args.full_vocab_kl:
            assert exact_sampling is not None
            assert exact_accumulator is not None
            assert exact_profile is not None
            for repeat in range(args.quality_repeats):
                for prompt_index, prompt_ids in enumerate(exact_prompts):
                    block_index = repeat * len(exact_prompts) + prompt_index
                    order = v5.paired_arm_order(repeat + prompt_index)
                    compact_scores: dict[str, _CompactFullVocabScore] = {}
                    score_digests: dict[str, dict[str, Any]] = {}
                    routes: dict[str, Any] = {}
                    walls: dict[str, float] = {}
                    for arm in order:
                        output, wall_ms, selector, route = _run_generate(
                            llm=llm,
                            sampling=exact_sampling,
                            prompt_ids=prompt_ids,
                            arm=arm,
                            label=(
                                f"full_vocab:{repeat}:{prompt_index}:{arm}"
                            ),
                            controller=controller,
                            synchronize=synchronize,
                        )
                        selector_records.append(selector)
                        route_records.append(route)
                        routes[arm] = route
                        walls[arm] = wall_ms
                        score = _compact_full_vocab_score(
                            output,
                            prompt_ids,
                            expected_vocab_size=bootstrap.candidate_vocab_size,
                        )
                        compact_scores[arm] = score
                        score_digests[arm] = score.digest_record()
                        del output
                    scoring_pair = {
                        "prompt_index": block_index,
                        "pair_order": list(order),
                        "scores": compact_scores,
                    }
                    v5._accumulate_quality_pair(
                        exact_accumulator, scoring_pair
                    )
                    full_vocab_pairs.append({
                        "prompt_index": block_index,
                        "block_index": block_index,
                        "repeat_index": repeat,
                        "source_prompt_index": prompt_index,
                        "pair_order": list(order),
                        "score_digests": score_digests,
                        "wall_ms": walls,
                        "routes": routes,
                    })
                    # The scalar accumulator and digest record now carry every
                    # value needed by the shared scorer and determinism gate.
                    # Drop both cardinality-sized float32 arrays before the
                    # next pair begins.
                    del scoring_pair, compact_scores, score

    coarse_quality = v5._quality_summary(
        quality_pairs, kl_mode=v5.KL_COARSE_TOPK
    )
    if args.full_vocab_kl:
        assert exact_accumulator is not None
        assert exact_profile is not None
        exact_quality = v5._finish_quality_accumulator(exact_accumulator)
        quality = _promotion_quality_summary(
            coarse_quality,
            exact_quality,
            full_vocab_profile=exact_profile,
        )
    else:
        exact_quality = None
        quality = coarse_quality
    quality["arm_labels"] = dict(ARM_LABELS)
    teacher_arm_scores = _first_repeat_scores(
        quality_pairs,
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

    all_pairs = [*quality_pairs, *full_vocab_pairs, *timing_pairs]
    core_gates = {
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
        "selector_switch_attested": _selector_gate(selector_records),
        "route_and_fallback_attested": _route_gate(route_records),
        "quality_arm_order_counterbalanced": _pair_order_gate(quality_pairs),
        "quality_repeat_determinism": _repeat_determinism_gate(
            quality_pairs,
            n_prompts=len(bootstrap.prompts),
            repeats=args.quality_repeats,
        ),
        "fp8_routes_unchanged_between_arms": _fp8_invariance_gate(all_pairs),
    }
    if args.timing_repeats:
        core_gates["timing_arm_order_counterbalanced"] = _pair_order_gate(
            timing_pairs
        )
    if args.full_vocab_kl:
        core_gates.update({
            "full_vocab_arm_order_counterbalanced": _pair_order_gate(
                full_vocab_pairs
            ),
            "full_vocab_repeat_digest_determinism": (
                _full_vocab_repeat_determinism_gate(
                    full_vocab_pairs,
                    n_prompts=len(exact_prompts),
                    repeats=args.quality_repeats,
                )
            ),
        })

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": v5._utc_now(),
        "scope": (
            "DSV4 routed FP4-CB quality prefill; TP=1; one in-process vLLM "
            "engine; persistent-B handle A/B"
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
                "warmup_pairs": args.warmup_pairs,
                "quality_repeats": args.quality_repeats,
                "timing_repeats": args.timing_repeats,
                "full_vocab_kl": args.full_vocab_kl,
                "full_vocab_seqlen": (
                    args.full_vocab_seqlen if args.full_vocab_kl else None
                ),
                "full_vocab_profile": exact_profile,
                "full_vocab_storage_contract": (
                    "vllm FlatLogprobs -> paired float32 rows -> shared "
                    "incremental scorer -> scalar/digest evidence; raw rows "
                    "released before next pair"
                    if args.full_vocab_kl else None
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
        "quality": quality,
        "full_vocab_streaming_evidence": (
            {
                "profile": exact_profile,
                "pairs": full_vocab_pairs,
            }
            if args.full_vocab_kl else None
        ),
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
                "Exact full-vocabulary KL was cardinality-attested on every "
                "position of the digest-bound prefix profile, while exact "
                "target NLL/PPL and coarse rejection KL cover the complete "
                "sealed 8x512 workload."
                if args.full_vocab_kl
                else "Top-K/tail KL is a rejection screen and cannot promote "
                "persistent-B."
            ),
            "Offline one-token wall timing is not streaming TTFT or TPOT.",
            "A passing quality run does not replace served NATIVE-PARITY gates.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.timing_repeats:
        report["timing"] = v5._timing_summary(timing_samples)

    configured = v5._configured_gates(args, report)
    report["configured_promotion_gates"] = configured
    report["measurement_only"] = bool(args.measurement_only)
    report["measurement_valid"] = all(
        bool(gate.get("pass")) for gate in core_gates.values()
    )
    report["configured_gates_pass"] = (
        None
        if args.measurement_only
        else all(bool(gate.get("pass")) for gate in configured.values())
    )
    present_pairwise = [
        name for name in _PAIRWISE_QUALITY_GATES
        if getattr(args, name) is not None
    ]
    missing_pairwise = [
        name for name in _PAIRWISE_QUALITY_GATES
        if name not in present_pairwise
    ]
    report["promotion_contract"] = {
        "same_weight_and_activation_contract": True,
        "exact_full_vocab_kl_required": True,
        "exact_full_vocab_kl_observed": (
            quality["kl_mode"] == v5.KL_FULL_VOCAB
        ),
        "complete_sealed_workload_screen_observed": bool(quality_pairs),
        "streamed_full_vocab_profile_observed": (
            bool(full_vocab_pairs) if args.full_vocab_kl else False
        ),
        "full_vocab_profile": exact_profile,
        "required_pairwise_quality_thresholds": list(_PAIRWISE_QUALITY_GATES),
        "present_pairwise_quality_thresholds": present_pairwise,
        "missing_pairwise_quality_thresholds": missing_pairwise,
        "served_native_parity_still_required": True,
        "complete": (
            quality["kl_mode"] == v5.KL_FULL_VOCAB
            and bool(quality_pairs)
            and bool(full_vocab_pairs)
            and not missing_pairwise
        ),
    }
    report["promotion_recommendation"] = (
        "measurement_failed"
        if not report["measurement_valid"]
        else "measurement_only_no_promotion_thresholds_configured"
        if args.measurement_only
        else "configured_gates_failed"
        if report["configured_gates_pass"] is False
        else "screening_only_full_vocab_required"
        if quality["kl_mode"] != v5.KL_FULL_VOCAB
        else "screening_only_pairwise_quality_thresholds_required"
        if missing_pairwise
        else "candidate_only_requires_served_native_parity"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        help=(
            "request every candidate vocabulary logprob (and teacher row when "
            "--teacher-model is supplied); memory-heavy on DSV4"
        ),
    )
    parser.add_argument(
        "--full-vocab-seqlen",
        type=v5._positive_int,
        default=64,
        help=(
            "digest-attested prefix length per sealed window for streamed "
            "exact-vocabulary KL; complete-window NLL/PPL and coarse KL "
            "always remain 512 tokens"
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
    parser.add_argument("--warmup-pairs", type=v5._positive_int, default=1)
    parser.add_argument("--quality-repeats", type=v5._positive_int, default=2)
    parser.add_argument("--timing-repeats", type=v5._nonnegative_int, default=0)
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
    parser.add_argument("--measurement-only", action="store_true")
    parser.add_argument("--max-mean-kl", type=v5._nonnegative_float)
    parser.add_argument("--max-mean-nll-regression", type=v5._nonnegative_float)
    parser.add_argument("--max-ppl-relative-regression", type=v5._nonnegative_float)
    parser.add_argument(
        "--max-teacher-persistent-b-mean-kl",
        dest="max_teacher_fused_mean_kl",
        type=v5._nonnegative_float,
    )
    parser.add_argument(
        "--max-teacher-persistent-b-kl-regression",
        dest="max_teacher_fused_kl_regression",
        type=v5._nonnegative_float,
    )
    parser.add_argument("--min-timing-speedup", type=v5._nonnegative_float)
    args = parser.parse_args(argv)

    # Fields consumed by the shared loader/report API.  They are explicit here
    # rather than hidden in a second parser with unrelated fused-NVFP4 modes.
    args.mode = "moe256"
    args.teacher_full_vocab_kl = bool(args.full_vocab_kl)
    args.dataset_cache_dir = None
    args.dataset_split = "train"
    args.wikitext_text = None

    if args.seqlen <= 16:
        parser.error("--seqlen must exceed the routed prefill threshold (16)")
    if args.full_vocab_seqlen <= 16:
        parser.error("--full-vocab-seqlen must exceed prefill threshold 16")
    if args.full_vocab_seqlen > args.seqlen:
        parser.error("--full-vocab-seqlen cannot exceed --seqlen")
    if args.quality_repeats < 2:
        parser.error("--quality-repeats must be at least 2")
    if args.n_samples * args.quality_repeats % 2:
        parser.error("n-samples * quality-repeats must be even for arm balance")
    if args.timing_repeats and args.n_samples * args.timing_repeats % 2:
        parser.error("n-samples * timing-repeats must be even for arm balance")
    if args.timing_repeats == 1:
        parser.error(
            "--timing-repeats must be 0 or at least 2 so every source prompt "
            "sees both arm orders"
        )
    if args.min_timing_speedup is not None and args.timing_repeats == 0:
        parser.error("--min-timing-speedup requires --timing-repeats > 0")
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
    if args.full_vocab_kl and args.teacher_model:
        parser.error(
            "streamed pairwise --full-vocab-kl does not retain candidate rows "
            "for a later teacher pass; run the identity-matched teacher as a "
            "separate predeclared gate"
        )
    teacher_limits = (
        args.max_teacher_fused_mean_kl,
        args.max_teacher_fused_kl_regression,
    )
    if any(value is not None for value in teacher_limits) and not args.teacher_model:
        parser.error("teacher-relative gates require --teacher-model")
    limits = v5._configured_limit_values(args)
    has_thresholds = any(value is not None for value in limits)
    if not has_thresholds and not args.measurement_only:
        parser.error(
            "no thresholds configured; pass --measurement-only for "
            "evidence-only use"
        )
    if has_thresholds and args.measurement_only:
        parser.error("--measurement-only cannot be combined with thresholds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - preserve machine-readable failure
        failure = {
            "schema": SCHEMA,
            "created_at": v5._utc_now(),
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        }
        v5._atomic_json(args.output, failure)
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
    v5._atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_contract": report["promotion_contract"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_delta": report["quality"]["delta"],
        "timing": report.get("timing"),
        "layer_inventory": report["dispatch"]["layer_inventory"],
    }, indent=2), flush=True)
    return 0 if report["status"] in ("ok", "measurement_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
