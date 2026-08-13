"""CPU contract tests for the persistent-B whole-model quality A/B."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _script_path() -> Path:
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        candidate = root / "scripts" / "validate_moe_persistent_b_ab.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("validate_moe_persistent_b_ab.py")


SCRIPT = _script_path()
SPEC = importlib.util.spec_from_file_location(
    "validate_moe_persistent_b_ab", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
pb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pb
SPEC.loader.exec_module(pb)


def _file_descriptor(path: Path) -> dict:
    return pb.v5._required_file_record(path)


def _write_gold_payload(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    windows: list[list[int]] | None = None,
) -> tuple[Path, Path]:
    model = root / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text('{"test":true}\n', encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    files = {
        name: _file_descriptor(model / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    tokenizer = {
        "schema": "prismaquant.tokenizer_identity/1",
        "content_sha256": pb._canonical_sha256({"files": files}),
        "files": files,
    }
    windows = windows or [[1, 2, 3], [4, 5, 6]]
    flattened = [token for window in windows for token in window]
    tensor_digest = pb.hashlib.sha256(
        struct.pack(f"<{len(flattened)}q", *flattened)
    ).hexdigest()
    full_selection = dict(pb._CANONICAL_FULL_KL_SELECTION)
    full_selection.update({
        "n_samples": len(windows),
        "seqlen": len(windows[0]),
        "starts": list(range(len(windows))),
    })
    ppl_ids = [1, 2]
    ppl_selection = dict(pb._CANONICAL_PPL_SELECTION)
    ppl_selection["n_tokens"] = len(ppl_ids)
    payload = {
        "schema": pb._FULL_KL_SCHEMA,
        "datasets_distribution": dict(pb._CANONICAL_DATASETS_DISTRIBUTION),
        "corpus_construction": dict(pb._CANONICAL_CORPUS_CONSTRUCTION),
        "tokenizer": tokenizer,
        "full_kl": {
            "dataset": dict(pb._CANONICAL_FULL_KL_DATASET),
            "selection": full_selection,
            "token_ids": windows,
            "token_ids_tensor_sha256": tensor_digest,
        },
        "ppl": {
            "dataset": dict(pb._CANONICAL_PPL_DATASET),
            "selection": ppl_selection,
            "token_ids": ppl_ids,
            "token_ids_sha256": pb._canonical_sha256(ppl_ids),
        },
    }
    payload["semantic_sha256"] = pb._canonical_sha256(payload)
    # Unit fixtures are deliberately tiny; patch only the immutable expected
    # values that their content necessarily changes. Separate tests below pin
    # every production constant and prove that re-sealing cannot evade them.
    monkeypatch.setattr(
        pb, "_CANONICAL_INPUT_SEMANTIC_SHA256", payload["semantic_sha256"]
    )
    monkeypatch.setattr(
        pb, "_CANONICAL_TOKENIZER_CONTENT_SHA256", tokenizer["content_sha256"]
    )
    monkeypatch.setattr(pb, "_CANONICAL_TOKENIZER_VOCAB_SIZE", 32)
    monkeypatch.setattr(pb, "_CANONICAL_FULL_KL_SELECTION", full_selection)
    monkeypatch.setattr(
        pb, "_CANONICAL_FULL_KL_TOKEN_IDS_TENSOR_SHA256", tensor_digest
    )
    monkeypatch.setattr(pb, "_CANONICAL_PPL_SELECTION", ppl_selection)
    monkeypatch.setattr(
        pb, "_CANONICAL_PPL_TOKEN_IDS_SHA256", payload["ppl"]["token_ids_sha256"]
    )
    path = root / "gold.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return model, path


class _Tokenizer:
    def __len__(self):
        return 32


def _prompt_args(model: Path, path: Path, *, n=2, seqlen=3):
    return SimpleNamespace(
        model=str(model),
        prompt_token_ids_json=path,
        n_samples=n,
        seqlen=seqlen,
    )


def test_canonical_release_workload_constants_are_value_closed():
    assert pb._CANONICAL_INPUT_SEMANTIC_SHA256 == (
        "3eedb1a879e8e9cf13a2a0f16d3fd01a0817d18de60a2dd743c9c4fc44a34680"
    )
    assert pb._CANONICAL_DATASETS_DISTRIBUTION == {
        "name": "datasets", "version": "4.6.0"
    }
    assert pb._CANONICAL_TOKENIZER_CONTENT_SHA256 == (
        "9f7ee7cb93b58bf30f278965547e7584b89c848e76c3adfeb92c070a88492de0"
    )
    assert pb._CANONICAL_FULL_KL_SELECTION["starts"] == [
        466_956, 104_902, 1_153_556, 1_027_150,
        936_213, 585_264, 429_895, 2_287_433,
    ]
    assert pb._CANONICAL_FULL_KL_TOKEN_IDS_TENSOR_SHA256 == (
        "b3426e9bab87a1c444b04d0ce01fa9cba5ace313b91db2c3f77fc3525e732b22"
    )


def test_fixed_prompt_loader_attests_closed_payload_tokens_and_tokenizer(
    tmp_path, monkeypatch
):
    model, path = _write_gold_payload(tmp_path, monkeypatch)
    windows, record = pb._fixed_prompt_loader(
        _prompt_args(model, path), _Tokenizer()
    )
    assert windows == [[1, 2, 3], [4, 5, 6]]
    assert record["source"] == "producer_sealed_token_ids"
    assert record["input_file"]["sha256"] == pb.hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert record["prompt_token_ids_tensor_sha256"] == (
        json.loads(path.read_text())["full_kl"]["token_ids_tensor_sha256"]
    )
    assert set(record["tokenizer"]["files"]) == {
        "tokenizer.json", "tokenizer_config.json"
    }


def test_fixed_prompt_loader_rejects_subset_and_resigned_token_values(
    tmp_path, monkeypatch
):
    model, path = _write_gold_payload(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="complete sealed prompt selection"):
        pb._fixed_prompt_loader(
            _prompt_args(model, path, n=1), _Tokenizer()
        )

    payload = json.loads(path.read_text())
    payload["full_kl"]["token_ids"][0][0] = 7
    flattened = [
        token for row in payload["full_kl"]["token_ids"] for token in row
    ]
    payload["full_kl"]["token_ids_tensor_sha256"] = pb.hashlib.sha256(
        struct.pack(f"<{len(flattened)}q", *flattened)
    ).hexdigest()
    payload["semantic_sha256"] = pb._canonical_sha256({
        key: value for key, value in payload.items()
        if key != "semantic_sha256"
    })
    # Even if the outer self-digest were accepted, the independently pinned
    # canonical token digest must reject the operator-created workload.
    monkeypatch.setattr(
        pb, "_CANONICAL_INPUT_SEMANTIC_SHA256", payload["semantic_sha256"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="token values are not canonical"):
        pb._fixed_prompt_loader(_prompt_args(model, path), _Tokenizer())


def test_fixed_prompt_loader_rejects_resigned_dataset_metadata(
    tmp_path, monkeypatch
):
    model, path = _write_gold_payload(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload["datasets_distribution"]["version"] = "999.0.0"
    payload["semantic_sha256"] = pb._canonical_sha256({
        key: value for key, value in payload.items()
        if key != "semantic_sha256"
    })
    monkeypatch.setattr(
        pb, "_CANONICAL_INPUT_SEMANTIC_SHA256", payload["semantic_sha256"]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="datasets producer identity differs"):
        pb._fixed_prompt_loader(_prompt_args(model, path), _Tokenizer())


def test_fixed_prompt_loader_rejects_model_tokenizer_drift(tmp_path, monkeypatch):
    model, path = _write_gold_payload(tmp_path, monkeypatch)
    (model / "tokenizer.json").write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="tokenizer file differs"):
        pb._fixed_prompt_loader(_prompt_args(model, path), _Tokenizer())


class _CBMethod:
    def __init__(self, prefix: str, is_fp4: bool):
        self.prefix = prefix
        self.is_fp4 = is_fp4


class _Layer:
    pass


class _RouteAPI:
    @staticmethod
    def read_route(layer):
        return getattr(layer, "route", None)


def _controller(*, fp4=2, fp8=1):
    registry = {}
    entries = {}
    for index in range(fp4 + fp8):
        is_fp4 = index < fp4
        method = _CBMethod(f"layer.{index}", is_fp4)
        layer = _Layer()
        layer._cb_moe_persistent_b = object() if is_fp4 else None
        if is_fp4:
            layer._cb_moe_persistent_b_cfg = 0
            layer._cb_bf16_sm120 = None
            layer._cb_fused_fp4_moe_mode = ""
        registry[index] = object()
        entries[index] = (method, layer)
    ops = SimpleNamespace(
        _LAYER_REGISTRY=registry,
        _lookup_cb_layer=lambda layer_id: entries[layer_id],
    )
    moe = SimpleNamespace(PrismaQuantCBMoEMethod=_CBMethod)
    controller = pb.PersistentBArmController(
        ops=ops,
        moe=moe,
        route_api=_RouteAPI,
        expected_fp4=fp4,
        expected_fp8=fp8,
        expected_cfg=0,
    )
    return controller


def _fp4_route(arm: str):
    if arm == "fused":
        return {
            "policy": "moe_persistent_b",
            "symbol": "cb_moe_persistent_b_prefill",
            "tile_m": 0,
            "contract": "fp4_group16_rtn",
            "state": "served",
            "reason": None,
        }
    return {
        "policy": "bf16_grouped_bridge",
        "symbol": "cb_bf16_grouped_mm_out",
        "tile_m": 0,
        "contract": "fp4_group16_rtn",
        "state": "served",
        "reason": None,
    }


def _fp8_route():
    return {
        "policy": "fp8_grouped",
        "symbol": "cb_fused_moe_grouped",
        "tile_m": 128,
        "contract": "fp8_per_token_dynamic",
        "state": "served",
        "reason": None,
    }


@pytest.mark.parametrize("arm", pb.ARMS)
def test_controller_switches_only_resolved_fp4_handles_and_proves_routes(arm):
    controller = _controller()
    initial = [binding.layer._cb_moe_persistent_b for binding in controller.fp4]
    with controller.arm(arm, label=f"test:{arm}") as selector:
        assert selector["pass"] is True
        assert all(
            (binding.layer._cb_moe_persistent_b is binding.persistent_handle)
            == (arm == "fused")
            for binding in controller.fp4
        )
        for binding in controller.fp4:
            binding.layer.route = _fp4_route(arm)
        for binding in controller.fp8:
            binding.layer.route = _fp8_route()
        routes = controller.attest_routes(arm, label=f"test:{arm}")
        assert routes["pass"] is True
        assert routes["violations"] == []
    assert [
        binding.layer._cb_moe_persistent_b for binding in controller.fp4
    ] == initial


def test_controller_route_attestation_fails_missing_or_fallback_route():
    controller = _controller(fp4=1, fp8=1)
    with controller.arm("fused", label="bad"):
        controller.fp4[0].layer.route = {
            **_fp4_route("fused"),
            "state": "fallback",
            "reason": "test miss",
        }
        # FP8 deliberately remains cleared/missing.
        record = controller.attest_routes("fused", label="bad")
    assert record["pass"] is False
    assert any("state='fallback'" in item for item in record["violations"])
    assert any("no current-request FP8 route" in item
               for item in record["violations"])


def test_controller_inventory_is_fail_closed():
    controller = _controller(fp4=1, fp8=1)
    assert controller.inventory_gate["pass"] is True
    with pytest.raises(RuntimeError, match="inventory does not match"):
        # Construct the same loaded registry while claiming a partial FP4 set.
        entries = {
            binding.layer_id: (binding.method, binding.layer)
            for binding in (*controller.fp4, *controller.fp8)
        }
        pb.PersistentBArmController(
            ops=SimpleNamespace(
                _LAYER_REGISTRY={key: object() for key in entries},
                _lookup_cb_layer=lambda key: entries[key],
            ),
            moe=SimpleNamespace(PrismaQuantCBMoEMethod=_CBMethod),
            route_api=_RouteAPI,
            expected_fp4=2,
            expected_fp8=1,
            expected_cfg=0,
        )


def _score(target=-1.0):
    return pb.v5.PromptScore(
        target_logprobs=(target,),
        rows=(pb.v5.TopKRow(
            token_ids=(0, 1),
            logprobs=(pb.math.log(0.7), pb.math.log(0.3)),
        ),),
    )


def _quality_pairs(*, mismatch=False):
    pairs = []
    for repeat in range(2):
        for prompt in range(2):
            block = repeat * 2 + prompt
            scores = {"baseline": _score(), "fused": _score()}
            if mismatch and repeat == 1 and prompt == 0:
                scores["fused"] = _score(-1.1)
            fp8 = {"fp8": _fp8_route()}
            pairs.append({
                "prompt_index": block,
                "block_index": block,
                "repeat_index": repeat,
                "source_prompt_index": prompt,
                "pair_order": list(pb.v5.paired_arm_order(repeat + prompt)),
                "scores": scores,
                "routes": {
                    "baseline": {"fp8_routes": fp8},
                    "fused": {"fp8_routes": dict(fp8)},
                },
            })
    return pairs


def test_interleaving_repeat_determinism_and_fp8_invariance_gates():
    pairs = _quality_pairs()
    order_gate = pb._pair_order_gate(pairs)
    assert order_gate["pass"] is True
    assert all(
        counts == {"baseline/fused": 1, "fused/baseline": 1}
        for counts in order_gate["per_source_prompt"].values()
    )
    assert pb._repeat_determinism_gate(
        pairs, n_prompts=2, repeats=2
    )["pass"] is True
    assert pb._fp8_invariance_gate(pairs)["pass"] is True

    nondeterministic = _quality_pairs(mismatch=True)
    gate = pb._repeat_determinism_gate(
        nondeterministic, n_prompts=2, repeats=2
    )
    assert gate["pass"] is False
    assert gate["mismatches"][0]["arm"] == "fused"
    nondeterministic[0]["routes"]["fused"]["fp8_routes"]["fp8"] = {
        **_fp8_route(), "tile_m": 256
    }
    assert pb._fp8_invariance_gate(nondeterministic)["pass"] is False


def test_aggregate_arm_balance_cannot_hide_missing_per_prompt_crossover():
    pairs = _quality_pairs()
    for pair in pairs:
        # This is the retired block-index schedule. With an even prompt count,
        # aggregate order remains 50/50 while each prompt always sees one order.
        pair["pair_order"] = list(
            pb.v5.paired_arm_order(pair["block_index"])
        )
    gate = pb._pair_order_gate(pairs)
    assert gate["counts"] == {"baseline/fused": 2, "fused/baseline": 2}
    assert gate["source_prompts_missing_both_orders"] == [0, 1]
    assert gate["pass"] is False


class _FlatLogprobs:
    def __init__(self, rows):
        self._rows = rows
        self.start_indices = []
        self.end_indices = []
        self.token_ids = []
        self.logprobs = []
        self.ranks = []
        self.decoded_tokens = []
        for row in rows:
            self.start_indices.append(len(self.token_ids))
            for token, logprob in row.items():
                self.token_ids.append(token)
                self.logprobs.append(logprob)
                self.ranks.append(1)
                self.decoded_tokens.append(None)
            self.end_indices.append(len(self.token_ids))

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return {
            token: {"logprob": logprob}
            for token, logprob in self._rows[index].items()
        }


class _PrimitiveFlatLogprobs:
    """Exact pinned vLLM layout, including duplicate raw token ids."""

    def __init__(self, rows, *, forbid_getitem=False):
        self._rows = rows
        self._forbid_getitem = forbid_getitem
        self.getitem_calls = 0
        self.start_indices = []
        self.end_indices = []
        self.token_ids = []
        self.logprobs = []
        self.ranks = []
        self.decoded_tokens = []
        for row in rows:
            self.start_indices.append(len(self.token_ids))
            for token, logprob in row:
                self.token_ids.append(token)
                self.logprobs.append(logprob)
                self.ranks.append(1)
                self.decoded_tokens.append(None)
            self.end_indices.append(len(self.token_ids))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, index):
        self.getitem_calls += 1
        if self._forbid_getitem:
            raise AssertionError("direct scorer materialized a mapping row")
        # This is vLLM FlatLogprobs.__getitem__ semantics: exact duplicate
        # keys retain their original insertion slot but their last value.
        return {
            token: {"logprob": logprob}
            for token, logprob in self._rows[index]
        }


def _primitive_output(rows, prompt_ids, *, forbid_getitem=False):
    return SimpleNamespace(
        prompt_token_ids=list(prompt_ids),
        prompt_logprobs=_PrimitiveFlatLogprobs(
            rows, forbid_getitem=forbid_getitem
        ),
    )


def _legacy_compact_full_vocab_score(output, prompt_ids, vocab_size):
    """The retired mapping-based implementation, retained only as an oracle."""

    expected_ids = [int(token) for token in prompt_ids]
    prompt_bytes = pb.struct.pack(f"<{len(expected_ids)}q", *expected_ids)
    prompt_digest = pb.hashlib.sha256(prompt_bytes).hexdigest()
    score_digest = pb.hashlib.sha256()
    score_digest.update(b"gridbook.compact-full-vocab-score.v1\0")
    score_digest.update(
        pb.struct.pack("<II", len(expected_ids) - 1, vocab_size)
    )
    score_digest.update(bytes.fromhex(prompt_digest))
    row_values = pb.array.array("f")
    targets = []
    row_digests = []
    for position in range(1, len(expected_ids)):
        entries = output.prompt_logprobs[position]
        target = pb.v5.target_logprob(entries, expected_ids[position])
        row = pb.v5.full_vocab_row(entries, vocab_size)
        packed = pb.array.array("f", row.logprobs)
        raw = packed.tobytes()
        row_digest = pb.hashlib.sha256(raw).hexdigest()
        row_values.extend(packed)
        targets.append(target)
        row_digests.append(row_digest)
        score_digest.update(pb.struct.pack("<d", target))
        score_digest.update(bytes.fromhex(row_digest))
    return pb._CompactFullVocabScore(
        target_logprobs=tuple(targets),
        _row_values=row_values,
        vocab_size=vocab_size,
        prompt_token_ids_sha256=prompt_digest,
        row_sha256=tuple(row_digests),
        score_sha256=score_digest.hexdigest(),
    )


def _full_vocab_output(*, delta=0.0):
    rows = [
        {},
        {0: pb.math.log(0.4), 1: pb.math.log(0.3),
         2: pb.math.log(0.2), 3: pb.math.log(0.1)},
        {0: pb.math.log(0.1), 1: pb.math.log(0.2),
         2: pb.math.log(0.3) + delta, 3: pb.math.log(0.4)},
    ]
    return SimpleNamespace(
        prompt_token_ids=[0, 1, 2],
        prompt_logprobs=_FlatLogprobs(rows),
    )


def test_streamed_full_vocab_compaction_reuses_shared_incremental_scorer():
    baseline = pb._compact_full_vocab_score(
        _full_vocab_output(), [0, 1, 2], expected_vocab_size=4
    )
    fused = pb._compact_full_vocab_score(
        _full_vocab_output(), [0, 1, 2], expected_vocab_size=4
    )
    assert len(baseline._row_values) == 8
    assert baseline.digest_record()["row_storage_bytes"] == 32
    assert baseline.score_sha256 == fused.score_sha256

    accumulator = pb.v5._new_quality_accumulator(
        kl_mode=pb.v5.KL_FULL_VOCAB
    )
    pb.v5._accumulate_quality_pair(accumulator, {
        "prompt_index": 0,
        "pair_order": ["baseline", "fused"],
        "scores": {"baseline": baseline, "fused": fused},
    })
    quality = pb.v5._finish_quality_accumulator(accumulator)
    assert quality["kl_mode"] == pb.v5.KL_FULL_VOCAB
    assert quality["delta"]["kl_baseline_to_fused"]["mean"] == 0.0
    assert quality["arms"]["baseline"]["tokens_scored"] == 2


def test_direct_flat_compaction_is_legacy_digest_identical_without_getitem():
    prompt_ids = [0, 1, 2]
    rows = [
        [],
        [
            (1, -8.0),
            (3, pb.math.log(0.1)),
            (0, pb.math.log(0.4)),
            (1, pb.math.log(0.3)),
            (2, pb.math.log(0.2)),
        ],
        [
            (2, -9.0),
            (1, pb.math.log(0.2)),
            (3, pb.math.log(0.4)),
            (2, pb.math.log(0.3)),
            (0, pb.math.log(0.1)),
        ],
    ]
    legacy_output = _primitive_output(rows, prompt_ids)
    expected = _legacy_compact_full_vocab_score(
        legacy_output, prompt_ids, 4
    )
    assert legacy_output.prompt_logprobs.getitem_calls == 2

    direct_output = _primitive_output(
        rows, prompt_ids, forbid_getitem=True
    )
    observed = pb._compact_full_vocab_score(
        direct_output, prompt_ids, expected_vocab_size=4
    )
    assert direct_output.prompt_logprobs.getitem_calls == 0
    assert observed.target_logprobs == expected.target_logprobs
    assert observed._row_values.tobytes() == expected._row_values.tobytes()
    assert observed.row_sha256 == expected.row_sha256
    assert observed.prompt_token_ids_sha256 == expected.prompt_token_ids_sha256
    assert observed.score_sha256 == expected.score_sha256
    assert tuple(observed.rows) == tuple(expected.rows)


def test_direct_flat_compaction_is_order_independent_and_last_value_wins():
    prompt_ids = [0, 1]
    canonical = [
        [],
        [
            (1, -20.0),
            (0, pb.math.log(0.6)),
            (1, pb.math.log(0.3)),
            (2, pb.math.log(0.1)),
        ],
    ]
    shuffled = [
        [],
        [
            (1, -20.0),
            (2, pb.math.log(0.1)),
            (1, pb.math.log(0.3)),
            (0, pb.math.log(0.6)),
        ],
    ]
    scores = [
        pb._compact_full_vocab_score(
            _primitive_output(rows, prompt_ids, forbid_getitem=True),
            prompt_ids,
            expected_vocab_size=3,
        )
        for rows in (canonical, shuffled)
    ]
    assert scores[0].target_logprobs == (pb.math.log(0.3),)
    assert scores[0]._row_values.tobytes() == scores[1]._row_values.tobytes()
    assert scores[0].row_sha256 == scores[1].row_sha256
    assert scores[0].score_sha256 == scores[1].score_sha256


def test_direct_flat_compaction_duplicate_nonfinite_uses_retained_value_only():
    prompt_ids = [0, 1]
    overwritten = [
        [],
        [
            (1, pb.math.log(0.5)),
            (0, float("nan")),
            (0, pb.math.log(0.3)),
            (1, pb.math.log(0.5)),
            (2, pb.math.log(0.2)),
        ],
    ]
    direct = pb._compact_full_vocab_score(
        _primitive_output(overwritten, prompt_ids, forbid_getitem=True),
        prompt_ids,
        expected_vocab_size=3,
    )
    legacy = _legacy_compact_full_vocab_score(
        _primitive_output(overwritten, prompt_ids), prompt_ids, 3
    )
    assert direct._row_values.tobytes() == legacy._row_values.tobytes()
    assert direct.score_sha256 == legacy.score_sha256

    retained_nonfinite = [
        [],
        [
            (1, pb.math.log(0.5)),
            (0, pb.math.log(0.3)),
            (0, float("nan")),
            (1, pb.math.log(0.5)),
            (2, pb.math.log(0.2)),
        ],
    ]
    with pytest.raises(ValueError, match="non-finite"):
        pb._compact_full_vocab_score(
            _primitive_output(
                retained_nonfinite, prompt_ids, forbid_getitem=True
            ),
            prompt_ids,
            expected_vocab_size=3,
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_direct_flat_compaction_rejects_nonfinite_target_first(nonfinite):
    prompt_ids = [0, 1]
    rows = [
        [],
        [(1, nonfinite), (0, -1.0), (1, nonfinite), (99, -2.0)],
    ]
    with pytest.raises(ValueError, match="non-finite"):
        pb._compact_full_vocab_score(
            _primitive_output(rows, prompt_ids, forbid_getitem=True),
            prompt_ids,
            expected_vocab_size=3,
        )


def test_direct_flat_compaction_preserves_row_failure_order():
    prompt_ids = [0, 1]
    nonfinite_before_oor = [
        [],
        [(1, -0.5), (0, float("nan")), (99, -1.0), (2, -2.0)],
    ]
    with pytest.raises(ValueError, match="non-finite"):
        pb._compact_full_vocab_score(
            _primitive_output(
                nonfinite_before_oor, prompt_ids, forbid_getitem=True
            ),
            prompt_ids,
            expected_vocab_size=3,
        )
    oor_before_nonfinite = [
        [],
        [(1, -0.5), (99, -1.0), (0, float("nan")), (2, -2.0)],
    ]
    with pytest.raises(RuntimeError, match="out-of-range token id 99"):
        pb._compact_full_vocab_score(
            _primitive_output(
                oor_before_nonfinite, prompt_ids, forbid_getitem=True
            ),
            prompt_ids,
            expected_vocab_size=3,
        )


def test_direct_flat_compaction_target_missing_precedes_cardinality_error():
    prompt_ids = [0, 2]
    rows = [[], [(0, -0.5), (1, -1.0), (1, -2.0)]]
    with pytest.raises(KeyError) as exc:
        pb._compact_full_vocab_score(
            _primitive_output(rows, prompt_ids, forbid_getitem=True),
            prompt_ids,
            expected_vocab_size=3,
        )
    assert exc.value.args == (2,)


def test_direct_flat_compaction_cardinality_out_of_range_and_negative_rules():
    prompt_ids = [0, 1]
    missing = [[], [(1, -0.5), (0, -1.0), (1, -0.5)]]
    with pytest.raises(RuntimeError, match="cardinality mismatch"):
        pb._compact_full_vocab_score(
            _primitive_output(missing, prompt_ids, forbid_getitem=True),
            prompt_ids,
            expected_vocab_size=3,
        )

    out_of_range = [
        [], [(1, -0.5), (0, -1.0), (3, -2.0), (2, -3.0)]
    ]
    with pytest.raises(RuntimeError, match="out-of-range token id 3"):
        pb._compact_full_vocab_score(
            _primitive_output(
                out_of_range, prompt_ids, forbid_getitem=True
            ),
            prompt_ids,
            expected_vocab_size=3,
        )

    with_negative = [
        [],
        [(-7, float("nan")), (1, -0.5), (2, -2.0), (0, -1.0)],
    ]
    direct = pb._compact_full_vocab_score(
        _primitive_output(with_negative, prompt_ids, forbid_getitem=True),
        prompt_ids,
        expected_vocab_size=3,
    )
    legacy = _legacy_compact_full_vocab_score(
        _primitive_output(with_negative, prompt_ids), prompt_ids, 3
    )
    assert direct.score_sha256 == legacy.score_sha256


def test_direct_flat_compaction_preserves_float32_row_bytes():
    prompt_ids = [0, 1]
    rows = [
        [],
        [
            (1, 1.0000000596046448),
            (3, -0.0),
            (2, 1.0e-45),
            (0, -87.3365447505531),
            (1, 1.0000000596046448),
        ],
    ]
    direct = pb._compact_full_vocab_score(
        _primitive_output(rows, prompt_ids, forbid_getitem=True),
        prompt_ids,
        expected_vocab_size=4,
    )
    legacy = _legacy_compact_full_vocab_score(
        _primitive_output(rows, prompt_ids), prompt_ids, 4
    )
    assert direct._row_values.tobytes() == legacy._row_values.tobytes()
    assert direct.row_sha256 == legacy.row_sha256
    assert direct.score_sha256 == legacy.score_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda flat: flat.logprobs.pop(), "different lengths"),
        (lambda flat: flat.end_indices.pop(), "span count"),
        (lambda flat: flat.start_indices.__setitem__(1, 1), "not contiguous"),
        (lambda flat: flat.end_indices.__setitem__(1, 999), "out of bounds"),
        (lambda flat: flat.end_indices.__setitem__(0, 1), "position zero"),
    ],
)
def test_direct_flat_compaction_rejects_malformed_primitive_layout(
    mutation, message
):
    prompt_ids = [0, 1]
    output = _primitive_output(
        [[], [(1, -0.5), (0, -1.0), (2, -2.0)]],
        prompt_ids,
        forbid_getitem=True,
    )
    mutation(output.prompt_logprobs)
    with pytest.raises(RuntimeError, match=message):
        pb._compact_full_vocab_score(
            output, prompt_ids, expected_vocab_size=3
        )


def test_full_vocab_repeat_gate_uses_score_and_row_digests():
    score = pb._compact_full_vocab_score(
        _full_vocab_output(), [0, 1, 2], expected_vocab_size=4
    ).digest_record()
    pairs = []
    for repeat in range(2):
        pairs.append({
            "source_prompt_index": 0,
            "repeat_index": repeat,
            "score_digests": {"baseline": score, "fused": score},
        })
    assert pb._full_vocab_repeat_determinism_gate(
        pairs, n_prompts=1, repeats=2
    )["pass"] is True
    pairs[1]["score_digests"] = {
        "baseline": score,
        "fused": {**score, "score_sha256": "0" * 64},
    }
    assert pb._full_vocab_repeat_determinism_gate(
        pairs, n_prompts=1, repeats=2
    )["pass"] is False


def test_promotion_summary_uses_full_workload_targets_and_exact_profile_kl():
    coarse = pb.v5._quality_summary(
        _quality_pairs(), kl_mode=pb.v5.KL_COARSE_TOPK
    )
    exact = pb.v5._quality_summary(
        _quality_pairs(), kl_mode=pb.v5.KL_FULL_VOCAB
    )
    combined = pb._promotion_quality_summary(
        coarse, exact, full_vocab_profile={"seqlen": 64}
    )
    assert combined["arms"] == coarse["arms"]
    assert combined["delta"]["mean_nll_fused_minus_baseline"] == (
        coarse["delta"]["mean_nll_fused_minus_baseline"]
    )
    assert combined["delta"]["kl_baseline_to_fused"] == (
        exact["delta"]["kl_baseline_to_fused"]
    )
    assert combined["kl_mode"] == pb.v5.KL_FULL_VOCAB


def test_model_contract_gate_is_exact_dsv4():
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "vocab_size": 129280,
    }
    args = SimpleNamespace(
        expected_architecture="DeepseekV4ForCausalLM",
        expected_model_type="deepseek_v4",
        expected_hidden_layers=43,
        expected_hidden_size=4096,
        expected_moe_intermediate_size=2048,
        expected_routed_experts=256,
        expected_vocab_size=129280,
    )
    assert pb._model_contract_gate(config, args)["pass"] is True
    config["num_hidden_layers"] = 42
    gate = pb._model_contract_gate(config, args)
    assert gate["pass"] is False
    assert gate["checks"]["num_hidden_layers"] is False


def test_harness_reuses_v5_scoring_and_threshold_implementation():
    source = inspect.getsource(pb.run)
    assert "v5._quality_summary" in source
    assert "v5._configured_gates" in source
    assert "validation_common.score_teacher" in source
    assert "prefill_threshold=moe.MOE_PREFILL_M_THRESHOLD" in source
    assert pb.ARMS == ("baseline", "fused")
    assert pb.ARM_LABELS["baseline"] == "expand_plus_grouped_bf16_bridge"
    assert pb.ARM_LABELS["fused"] == "persistent_b_decode_in_mainloop"


def test_report_and_runtime_share_exported_moe_prefill_threshold():
    """The CPU gate catches a report-only name drift before a model run."""
    moe_source = (
        SCRIPT.parents[1] / "gridbook" / "moe.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(moe_source)
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "MOE_PREFILL_M_THRESHOLD"
            for target in node.targets
        )
    ]
    assert len(definitions) == 1
    assert ast.literal_eval(definitions[0].value) == 16
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "num_tokens"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.LtE)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "MOE_PREFILL_M_THRESHOLD"
        for node in ast.walk(tree)
    )
    assert (
        "prefill_threshold=moe.MOE_PREFILL_M_THRESHOLD"
        in inspect.getsource(pb.run)
    )


def test_persistent_b_provenance_names_only_existing_hashed_sources(tmp_path):
    package = tmp_path / "gridbook"
    csrc = package / "csrc"
    csrc.mkdir(parents=True)
    lane = package / "moe_persistent_b_lane.py"
    kernel = csrc / "cb_moe_persistent_b.cu"
    lane.write_text("# lane\n", encoding="utf-8")
    kernel.write_text("// kernel\n", encoding="utf-8")
    runtime = {
        "gridbook": {"package_root": str(package)},
        "source_files": {},
        "source_sha256": {},
    }
    pb._augment_persistent_b_provenance(runtime)
    assert set(runtime["source_files"]) == {
        "moe_persistent_b_lane.py", "cb_moe_persistent_b.cu"
    }
    assert set(runtime["source_sha256"]) == set(runtime["source_files"])
    for label, record in runtime["source_files"].items():
        path = Path(record["path"])
        assert path.is_file(), label
        assert record["sha256"] == pb.hashlib.sha256(path.read_bytes()).hexdigest()
        assert runtime["source_sha256"][label] == record["sha256"]


def test_shared_candidate_loader_forwards_explicit_kv_cache_dtype():
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class FakeSampling:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    bootstrap = SimpleNamespace(
        candidate_load_revision=None,
        quality_logprobs=32,
        llm_class=FakeLLM,
        sampling_params_class=FakeSampling,
    )
    args = SimpleNamespace(
        model="candidate",
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        seqlen=64,
        enable_chunked_prefill=None,
        seed=0,
        quantization="gridbook",
        tokenizer_mode="deepseek_v4",
        kv_cache_memory_bytes=268435456,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=256,
    )
    pb.validation_common.load_candidate_engine(
        bootstrap,
        args,
        probe=SimpleNamespace(restore=lambda: None),
    )
    assert calls[0]["kv_cache_dtype"] == "fp8"


def test_parser_requires_repeats_balanced_arms_and_explicit_measurement(
    tmp_path, monkeypatch
):
    model, inputs = _write_gold_payload(tmp_path, monkeypatch)
    base = [
        "--model", str(model),
        "--prompt-token-ids-json", str(inputs),
        "--output", str(tmp_path / "out.json"),
    ]
    args = pb.parse_args([*base, "--measurement-only"])
    assert args.quality_repeats == 2
    assert args.expected_persistent_b_layers == 35
    assert args.expected_fp8_cb_moe_layers == 8
    assert args.tokenizer_mode == "deepseek_v4"
    assert args.kv_cache_memory_bytes == 268435456
    assert args.kv_cache_dtype == "fp8"
    assert args.full_vocab_seqlen == 64
    assert args.teacher_full_vocab_kl is False
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--measurement-only", "--quality-repeats", "1"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--measurement-only", "--timing-repeats", "1"])
    with pytest.raises(SystemExit):
        pb.parse_args(base)
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--measurement-only", "--kv-cache-dtype", "auto"])
    args.kv_cache_dtype = "auto"
    with pytest.raises(RuntimeError, match="requires kv_cache_dtype='fp8'"):
        pb.run(args)


def test_parser_allows_pairwise_full_vocab_without_impossible_teacher(
    tmp_path, monkeypatch
):
    model, inputs = _write_gold_payload(tmp_path, monkeypatch)
    args = pb.parse_args([
        "--model", str(model),
        "--prompt-token-ids-json", str(inputs),
        "--output", str(tmp_path / "out.json"),
        "--full-vocab-kl",
        "--max-mean-kl", "0.0001",
        "--max-mean-nll-regression", "0.0001",
        "--max-ppl-relative-regression", "0.0001",
    ])
    assert args.full_vocab_kl is True
    assert args.teacher_full_vocab_kl is True
    assert args.teacher_model is None


def test_shared_bootstrap_extension_is_backward_compatible_and_selectable():
    signature = inspect.signature(pb.validation_common.prepare_validation)
    assert signature.parameters["extension_loader"].default == (
        "get_fused_fp4_ext"
    )
    assert signature.parameters["required_symbol"].default is None
    assert signature.parameters["prompt_loader"].default is None
