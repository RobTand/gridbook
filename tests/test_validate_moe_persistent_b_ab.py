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
from types import ModuleType, SimpleNamespace

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


def _quality_triplets(n_prompts=8):
    triplets = []
    for prompt in range(n_prompts):
        fp8 = {"fp8": _fp8_route()}
        triplets.append({
            "prompt_index": prompt,
            "block_index": prompt,
            "source_prompt_index": prompt,
            "request_order": list(pb._TRIPLET_ROLES),
            "arm_order": [arm for _role, arm in pb._TRIPLET_REQUESTS],
            "request_sequence": list(range(prompt * 3, prompt * 3 + 3)),
            "request_pids": {
                role: os.getpid() for role in pb._TRIPLET_ROLES
            },
            "scores": {
                "baseline_pre": _score(),
                "candidate": _score(),
                "baseline_post": _score(),
            },
            "routes": {
                role: {"fp8_routes": dict(fp8)} for role in pb._TRIPLET_ROLES
            },
        })
    return triplets


def test_triplet_order_cardinality_pid_and_fp8_invariance_gates():
    triplets = _quality_triplets()
    gate = pb._triplet_order_gate(
        triplets,
        n_prompts=8,
        expected_pid=os.getpid(),
        first_request_index=0,
    )
    assert gate["pass"] is True
    assert gate["expected_measured_requests"] == 24
    assert gate["observed_measured_requests"] == 24
    assert pb._fp8_invariance_gate(triplets)["pass"] is True

    triplets[0]["routes"]["candidate"]["fp8_routes"]["fp8"] = {
        **_fp8_route(), "tile_m": 256
    }
    assert pb._fp8_invariance_gate(triplets)["pass"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda blocks: blocks[0].__setitem__(
            "request_order", ["candidate", "baseline_pre", "baseline_post"]
        ),
        lambda blocks: blocks[0].__setitem__(
            "arm_order", ["fused", "baseline", "baseline"]
        ),
        lambda blocks: blocks[1].__setitem__("request_sequence", [4, 3, 5]),
        lambda blocks: blocks[2]["request_pids"].__setitem__(
            "candidate", os.getpid() + 1
        ),
        lambda blocks: blocks.reverse(),
    ],
)
def test_triplet_order_gate_fails_closed_on_adversarial_order(mutation):
    triplets = _quality_triplets()
    mutation(triplets)
    gate = pb._triplet_order_gate(
        triplets,
        n_prompts=8,
        expected_pid=os.getpid(),
        first_request_index=0,
    )
    assert gate["pass"] is False
    assert gate["violations"]


def test_warmup_gate_requires_each_profile_in_measurement_order():
    profiles = (
        "coarse_512",
        "exact_pilot_full_vocab_64",
        "exact_confirmation_full_vocab_512",
    )
    records = [
        {
            "profile": profile,
            "repeat_index": repeat,
            "role": role,
            "arm": arm,
        }
        for profile in profiles
        for repeat in range(2)
        for role, arm in pb._TRIPLET_REQUESTS
    ]
    gate = pb._warmup_gate(
        records, profiles=profiles, triplets_per_profile=2
    )
    assert gate["pass"] is True
    assert gate["expected_requests"] == 18
    assert pb._warmup_gate(
        records[:-1], profiles=profiles, triplets_per_profile=2
    )["pass"] is False
    records[6], records[12] = records[12], records[6]
    assert pb._warmup_gate(
        records, profiles=profiles, triplets_per_profile=2
    )["pass"] is False


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
        {0: pb.math.log(0.1 - delta), 1: pb.math.log(0.2),
         2: pb.math.log(0.3 + delta), 3: pb.math.log(0.4)},
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


def test_bounded_tensor_transport_compacts_without_python_cell_lists():
    torch = pytest.importorskip("torch")
    prompt_ids = [0, 1, 2]
    # Column zero is vLLM's requested target.  Columns [1:] are topk(V), a
    # complete probability-ordered permutation of the vocabulary.
    token_ids = torch.tensor([
        [1, 3, 0, 1, 2],
        [2, 1, 3, 2, 0],
    ], dtype=torch.int64)
    row0 = [pb.math.log(value) for value in (0.1, 0.4, 0.3, 0.2)]
    row1 = [pb.math.log(value) for value in (0.2, 0.4, 0.3, 0.1)]
    logprobs = torch.tensor([
        [-99.0, *row0],
        [-98.0, *row1],
    ], dtype=torch.float32)
    tensors = SimpleNamespace(
        logprob_token_ids=token_ids,
        logprobs=logprobs,
        selected_token_ranks=torch.tensor([2, 3], dtype=torch.int64),
        cu_num_generated_tokens=None,
    )
    bounded = pb._BoundedTensorPromptLogprobs(tensors)
    output = SimpleNamespace(
        prompt_token_ids=prompt_ids,
        prompt_logprobs=bounded,
    )
    score = pb._compact_full_vocab_score(
        output, prompt_ids, expected_vocab_size=4
    )

    assert score.transport_schema == bounded.schema
    assert score.transport_bytes == (
        token_ids.numel() * token_ids.element_size()
        + logprobs.numel() * logprobs.element_size()
        + tensors.selected_token_ranks.numel()
        * tensors.selected_token_ranks.element_size()
    )
    assert score.target_logprobs == pytest.approx([
        float(logprobs[0, 3]), float(logprobs[1, 3])
    ])
    assert score.transport_peak_compactor_scratch_bytes == 4 * (8 + 1 + 4 + 4)
    assert list(score._row_values) == pytest.approx([
        float(logprobs[0, 2]), float(logprobs[0, 3]),
        float(logprobs[0, 4]), float(logprobs[0, 1]),
        float(logprobs[1, 4]), float(logprobs[1, 1]),
        float(logprobs[1, 3]), float(logprobs[1, 2]),
    ])
    assert not any(
        isinstance(value, list) for value in (
            bounded.logprob_token_ids,
            bounded.logprobs,
            bounded.selected_token_ranks,
        )
    )


def test_bounded_tensor_transport_is_flat_digest_byte_identical():
    torch = pytest.importorskip("torch")
    prompt_ids = [0, 1, 2]
    # The target column deliberately conflicts with its duplicate in topk(V).
    # vLLM's FlatLogprobs mapping and the bounded transport must both use the
    # latter value (last duplicate wins).
    raw_rows = [
        [],
        [(1, -99.0), (3, -3.25), (0, -0.75), (1, -1.25), (2, -2.0)],
        [(2, -98.0), (1, -1.5), (3, -0.5), (2, -2.5), (0, -3.5)],
    ]
    flat = pb._compact_full_vocab_score(
        _primitive_output(raw_rows, prompt_ids, forbid_getitem=True),
        prompt_ids,
        expected_vocab_size=4,
    )
    tensors = SimpleNamespace(
        logprob_token_ids=torch.tensor(
            [[token for token, _value in row] for row in raw_rows[1:]],
            dtype=torch.int32,
        ),
        logprobs=torch.tensor(
            [[value for _token, value in row] for row in raw_rows[1:]],
            dtype=torch.float32,
        ),
        selected_token_ranks=torch.tensor([2, 3], dtype=torch.int32),
        cu_num_generated_tokens=None,
    )
    bounded = pb._compact_full_vocab_score(
        SimpleNamespace(
            prompt_token_ids=prompt_ids,
            prompt_logprobs=pb._BoundedTensorPromptLogprobs(tensors),
        ),
        prompt_ids,
        expected_vocab_size=4,
    )
    assert bounded.target_logprobs == flat.target_logprobs
    assert bounded._row_values.tobytes() == flat._row_values.tobytes()
    assert bounded.row_sha256 == flat.row_sha256
    assert bounded.score_sha256 == flat.score_sha256


def test_bounded_tensor_transport_rejects_duplicate_support_and_nonfinite():
    torch = pytest.importorskip("torch")

    def output(token_ids, logprobs):
        return SimpleNamespace(
            prompt_token_ids=[0, 1],
            prompt_logprobs=pb._BoundedTensorPromptLogprobs(SimpleNamespace(
                logprob_token_ids=torch.tensor([token_ids], dtype=torch.int32),
                logprobs=torch.tensor([logprobs], dtype=torch.float32),
                selected_token_ranks=torch.tensor([1], dtype=torch.int32),
                cu_num_generated_tokens=None,
            )),
        )

    with pytest.raises(RuntimeError, match="each canonical token exactly once"):
        pb._compact_full_vocab_score(
            output([1, 0, 1, 1], [-1.0] * 4),
            [0, 1], expected_vocab_size=3,
        )
    with pytest.raises(ValueError, match="non-finite"):
        pb._compact_full_vocab_score(
            output([1, 0, 1, 2], [-1.0, -1.0, float("nan"), -1.0]),
            [0, 1], expected_vocab_size=3,
        )


def test_exact_transport_patch_intercepts_minus_one_before_tolist(monkeypatch):
    class FakeFlatLogprobs:
        def __len__(self):
            return 1

    original_calls = []

    class FakeProcessor:
        def _update_prompt_logprobs(self, tensors):
            original_calls.append(tensors)

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    logprobs = ModuleType("vllm.logprobs")
    logprobs.FlatLogprobs = FakeFlatLogprobs
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    engine = ModuleType("vllm.v1.engine")
    engine.__path__ = []
    engine_logprobs = ModuleType("vllm.v1.engine.logprobs")
    engine_logprobs.LogprobsProcessor = FakeProcessor
    for name, module in (
        ("vllm", vllm), ("vllm.logprobs", logprobs),
        ("vllm.v1", v1), ("vllm.v1.engine", engine),
        ("vllm.v1.engine.logprobs", engine_logprobs),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    contract = pb._install_bounded_full_vocab_transport(vllm)
    poison = SimpleNamespace(
        logprob_token_ids=object(),
        logprobs=SimpleNamespace(shape=(2, 5)),
        selected_token_ranks=object(),
        cu_num_generated_tokens=None,
    )
    exact = FakeProcessor()
    exact.num_prompt_logprobs = -1
    exact.tokenizer = None
    exact.prompt_logprobs = FakeFlatLogprobs()
    exact._update_prompt_logprobs(poison)
    assert isinstance(exact.prompt_logprobs, pb._BoundedTensorPromptLogprobs)
    assert exact.prompt_logprobs.logprobs is poison.logprobs
    assert original_calls == []
    assert contract["python_cell_materialization"] is False
    with pytest.raises(RuntimeError, match="fresh one-position"):
        exact._update_prompt_logprobs(poison)

    # The public final-output object can own the tensor wrapper after the
    # processor drops its reference; no tensor data are copied or released.
    final_prompt_logprobs = exact.prompt_logprobs
    exact.prompt_logprobs = []
    assert final_prompt_logprobs.logprobs is poison.logprobs

    coarse = FakeProcessor()
    coarse.num_prompt_logprobs = 256
    coarse._update_prompt_logprobs(poison)
    assert original_calls == [poison]


def test_exact_512_transport_payload_bound_is_below_two_gibibytes():
    positions = 511
    vocab = 129280
    # Worst supported tensor dtypes: int64 ids, float32 values, int64 ranks.
    transport = positions * (vocab + 1) * (8 + 4) + positions * 8
    compact_triplet = 3 * positions * vocab * 4
    assert transport == 792755180
    assert compact_triplet == 792744960
    assert transport + compact_triplet < 2 * 1024**3
    source = inspect.getsource(pb._install_bounded_full_vocab_transport)
    assert ".tolist(" not in source


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


def test_exact_triplet_uses_half_jeffreys_symmetric_kl_at_prompt_unit():
    baseline = pb._compact_full_vocab_score(
        _full_vocab_output(), [0, 1, 2], expected_vocab_size=4
    )
    candidate = pb._compact_full_vocab_score(
        _full_vocab_output(delta=-0.02), [0, 1, 2], expected_vocab_size=4
    )
    metrics = pb._exact_triplet_metrics({
        "baseline_pre": baseline,
        "candidate": candidate,
        "baseline_post": baseline,
    }, prompt_index=3)
    assert metrics["prompt_index"] == 3
    assert metrics["positions"] == 2
    assert metrics["D_CB"] > 0.0
    assert metrics["D_BB"] == 0.0
    assert metrics["symmetric_kl_candidate_baseline_pre"] == pytest.approx(
        metrics["symmetric_kl_candidate_baseline_post"]
    )


def test_exact_64_is_a_prefix_pilot_and_exact_512_covers_complete_windows():
    prompts = [list(range(512)) for _ in range(8)]
    pilot_prompts, pilot = pb._full_vocab_profile(prompts, seqlen=64)
    confirmation_prompts, confirmation = pb._full_vocab_profile(
        prompts, seqlen=512
    )
    assert pilot["n_samples"] == confirmation["n_samples"] == 8
    assert pilot["seqlen"] == 64
    assert pilot["positions_per_repeat"] == 8 * 63
    assert confirmation["seqlen"] == 512
    assert confirmation["positions_per_repeat"] == 8 * 511
    assert all(len(prompt) == 64 for prompt in pilot_prompts)
    assert confirmation_prompts == prompts
    assert pilot["token_ids_tensor_sha256"] != (
        confirmation["token_ids_tensor_sha256"]
    )


def test_exact_8x512_completion_gate_rejects_missing_reordered_or_short_rows():
    records = [
        {"prompt_index": index, "positions": 511} for index in range(8)
    ]
    passed = pb._exact_profile_completion_gate(
        records, n_prompts=8, seqlen=512, executed=True
    )
    assert passed["pass"] is True
    assert passed["expected_scored_positions_per_prompt"] == 511

    for mutation in (
        lambda values: values.pop(),
        lambda values: values.reverse(),
        lambda values: values[3].__setitem__("positions", 63),
    ):
        adversarial = [dict(record) for record in records]
        mutation(adversarial)
        assert pb._exact_profile_completion_gate(
            adversarial, n_prompts=8, seqlen=512, executed=True
        )["pass"] is False
    assert pb._exact_profile_completion_gate(
        records, n_prompts=8, seqlen=512, executed=False
    )["pass"] is False


def _v6_records(
    *,
    effects=None,
    baseline_deltas=None,
    corruption=None,
    d_cb=None,
    d_bb=None,
):
    effects = list(effects if effects is not None else [0.001] * 8)
    baseline_deltas = list(
        baseline_deltas if baseline_deltas is not None else [0.0] * 8
    )
    corruption = list(
        corruption if corruption is not None else effects
    )
    d_cb = list(d_cb if d_cb is not None else [0.0] * 8)
    d_bb = list(d_bb if d_bb is not None else [0.0] * 8)
    nll = [
        {
            "prompt_index": index,
            "effect_candidate_minus_baseline_center": effects[index],
            "baseline_post_minus_pre": baseline_deltas[index],
            "corruption_excess": corruption[index],
        }
        for index in range(8)
    ]
    exact = [
        {"prompt_index": index, "D_CB": d_cb[index], "D_BB": d_bb[index]}
        for index in range(8)
    ]
    return nll, exact


def _decision(**kwargs):
    nll, exact = _v6_records(**kwargs)
    return pb._promotion_quality_v6(nll, exact, integrity_pass=True)


def test_v6_promotion_pass_and_integrity_fail_are_ternary_and_fail_closed():
    passed = _decision()
    assert passed["decision"] == "PROMOTION PASS"
    assert passed["pass"] is True
    assert passed["statistical_unit"] == "sealed_prompt_block_not_token_row"

    nll, exact = _v6_records()
    failed = pb._promotion_quality_v6(nll, exact, integrity_pass=False)
    assert failed["decision"] == "FAIL"
    assert failed["fail_reasons"] == ["integrity_violation"]

    with pytest.raises(RuntimeError, match="exactly 8 prompt blocks"):
        pb._fixed_prompt_t_interval(
            [0.0] * 511, limit=pb._NLL_LIMIT, label="token rows"
        )


def test_v6_confidence_bounds_distinguish_pass_fail_and_inconclusive():
    passed = pb._fixed_prompt_t_interval(
        [pb._NLL_LIMIT] * 8,
        limit=pb._NLL_LIMIT,
        label="constant boundary",
    )
    assert passed["decision"] == "PASS"
    assert passed["sample_sd"] == 0.0
    assert passed["lcb"] == passed["ucb"] == pb._NLL_LIMIT

    nll_failed = _decision(effects=[pb._NLL_LIMIT + 1.0e-6] * 8)
    assert nll_failed["decision"] == "FAIL"
    assert nll_failed["nll"]["gate"]["fail"] is True
    assert "mean_nll_effect_lcb_exceeds_limit" in nll_failed["fail_reasons"]

    point_above_but_unresolved = _decision(
        effects=[pb._NLL_LIMIT - 0.004, pb._NLL_LIMIT + 0.006] * 4,
        corruption=[0.0] * 8,
    )
    gate = point_above_but_unresolved["nll"]["gate"]
    assert gate["mean"] > gate["limit"]
    assert gate["lcb"] <= gate["limit"] < gate["ucb"]
    assert gate["decision"] == "INCONCLUSIVE"
    assert point_above_but_unresolved["decision"] == "INCONCLUSIVE"

    point_below_but_unresolved = _decision(
        effects=[pb._NLL_LIMIT - 0.006, pb._NLL_LIMIT + 0.004] * 4,
        corruption=[0.0] * 8,
    )
    gate = point_below_but_unresolved["nll"]["gate"]
    assert gate["mean"] < gate["limit"]
    assert gate["lcb"] < gate["limit"] < gate["ucb"]
    assert gate["decision"] == "INCONCLUSIVE"


def test_v6_hybrid_symmetric_kl_uses_one_predeclared_prompt_excess_ci():
    j_failed = _decision(d_cb=[0.001] * 8, d_bb=[0.0] * 8)
    assert j_failed["decision"] == "FAIL"
    assert "exact_symmetric_kl_hybrid_excess_lcb_above_zero" in (
        j_failed["fail_reasons"]
    )

    relative_allowance = _decision(
        d_cb=[0.0012] * 8, d_bb=[0.0010] * 8
    )
    distribution = relative_allowance["exact_full_vocab_symmetric_kl"]
    assert distribution["definition"] == "0.5 * (KL(P||Q) + KL(Q||P))"
    assert distribution["normalization_note"] == "half the Jeffreys divergence"
    assert distribution["relative_noise_multiplier"] == 1.25
    assert "at most 25% more" in distribution["relative_excess_policy"]
    assert distribution["absolute_allowance_D_BB_plus_1e_4"] == pytest.approx(
        [0.0011] * 8
    )
    assert distribution["hybrid_allowance"] == pytest.approx([0.00125] * 8)
    assert distribution["hybrid_excess_D_CB_minus_allowance"] == pytest.approx(
        [-0.00005] * 8
    )
    assert set(distribution).isdisjoint({
        "absolute_noise_subtraction", "relative_noise_allowance"
    })
    assert distribution["gate"]["decision"] == "PASS"
    assert relative_allowance["decision"] == "PROMOTION PASS"

    unresolved = _decision(
        d_cb=[0.0008, 0.0018] * 4,
        d_bb=[0.0010] * 8,
    )
    gate = unresolved["exact_full_vocab_symmetric_kl"]["gate"]
    assert gate["mean"] > 0.0
    assert gate["lcb"] <= 0.0 < gate["ucb"]
    assert gate["decision"] == "INCONCLUSIVE"
    assert unresolved["decision"] == "INCONCLUSIVE"


def test_v6_wide_nll_and_symmetric_kl_intervals_are_inconclusive():
    nll_wide = _decision(
        effects=[-0.012, 0.012] * 4, corruption=[0.0] * 8
    )
    assert nll_wide["decision"] == "INCONCLUSIVE"
    assert "nll_confidence_interval_too_wide" in (
        nll_wide["inconclusive_reasons"]
    )

    j_wide = _decision(
        d_cb=[0.0005, 0.0020] * 4,
        d_bb=[0.0010] * 8,
    )
    assert j_wide["decision"] == "INCONCLUSIVE"
    assert "symmetric_kl_confidence_interval_too_wide" in (
        j_wide["inconclusive_reasons"]
    )


def test_v6_baseline_noise_resolution_formula_and_boundary():
    rms_boundary = (
        pb._NLL_LIMIT
        * pb.math.sqrt(8.0 / 0.75)
        / pb._ONE_SIDED_T_95_DF7
    )
    at_boundary = _decision(baseline_deltas=[rms_boundary] * 8)
    gate = at_boundary["baseline_noise_resolution"]
    assert gate["formula"] == (
        "t_0.95,df=7 * sqrt(0.75 * mean_i((Bpost_i-Bpre_i)^2) / 8)"
    )
    assert gate["t_critical"] == 1.894578605061305
    assert gate["n"] == 8 and gate["df"] == 7
    assert "equal independent request-noise variance" in gate["assumptions"]
    assert "candidate-specific heteroskedasticity" in gate["assumptions"]
    assert gate["h_noise"] == pytest.approx(pb.math.log(1.005))
    assert gate["baseline_difference_rms"] == pytest.approx(0.008598, rel=1e-3)
    assert gate["pass"] is True

    resolved = _decision(baseline_deltas=[rms_boundary * 0.999] * 8)
    assert resolved["baseline_noise_resolution"]["pass"] is True
    assert resolved["decision"] == "PROMOTION PASS"

    unresolved = _decision(baseline_deltas=[rms_boundary * 1.001] * 8)
    assert unresolved["decision"] == "INCONCLUSIVE"
    assert "baseline_noise_exceeds_resolution_limit" in (
        unresolved["inconclusive_reasons"]
    )


def test_v6_centered_triplet_cancels_linear_drift():
    triplets = _quality_triplets()
    for index, triplet in enumerate(triplets):
        offset = index * 0.1
        triplet["scores"] = {
            "baseline_pre": _score(-(1.00 + offset)),
            "candidate": _score(-(1.01 + offset)),
            "baseline_post": _score(-(1.02 + offset)),
        }
    records = pb._nll_prompt_metrics(triplets)
    assert all(
        record["effect_candidate_minus_baseline_center"]
        == pytest.approx(0.0, abs=1e-15)
        for record in records
    )
    assert all(
        record["candidate_mean_nll"] - record["baseline_pre_mean_nll"]
        == pytest.approx(0.01)
        for record in records
    )


def test_v6_one_arm_corruption_backstop_is_fail_closed():
    adversarial = [-0.001] * 7 + [0.015]
    result = _decision(effects=adversarial, corruption=adversarial)
    assert result["decision"] == "FAIL"
    assert result["corruption_backstop"]["limit"] == pytest.approx(
        pb.math.log(1.01)
    )
    assert result["corruption_backstop"]["observed_max"] == 0.015
    assert "one_arm_corruption_backstop_exceeded" in result["fail_reasons"]

    at_boundary = _decision(corruption=[pb._CORRUPTION_LIMIT] * 8)
    assert at_boundary["corruption_backstop"]["pass"] is True
    assert at_boundary["decision"] == "PROMOTION PASS"
    above_boundary = _decision(corruption=[
        pb.math.nextafter(pb._CORRUPTION_LIMIT, pb.math.inf)
    ] * 8)
    assert above_boundary["corruption_backstop"]["pass"] is False
    assert above_boundary["decision"] == "FAIL"


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


def test_harness_reuses_v5_scoring_but_owns_fixed_v6_thresholds():
    source = inspect.getsource(pb.run)
    assert "v5.score_prompt_output" in source
    assert "_promotion_quality_v6" in source
    assert "v5._configured_gates" not in source
    assert "validation_common.score_teacher" in source
    assert "prefill_threshold=moe.MOE_PREFILL_M_THRESHOLD" in source
    assert pb.SCHEMA == "gridbook.moe-persistent-b-ab.v2"
    assert pb._PROMOTION_PROTOCOL == "persistent-b-promotion-quality-v6"
    assert pb.ARMS == ("baseline", "fused")
    assert pb.ARM_LABELS["baseline"] == "expand_plus_grouped_bf16_bridge"
    assert pb.ARM_LABELS["fused"] == "persistent_b_decode_in_mainloop"

    pilot_warm = source.index(
        '_warm_profile(\n        profile="exact_pilot_full_vocab_64"'
    )
    pilot_measure = source.index("_measure_exact_profile(", pilot_warm)
    confirmation_warm = source.index(
        '_warm_profile(\n            profile="exact_confirmation_full_vocab_512"',
        pilot_measure,
    )
    confirmation_measure = source.index(
        "_measure_exact_profile(", confirmation_warm
    )
    assert pilot_warm < pilot_measure < confirmation_warm < confirmation_measure
    assert (
        'confirmation_executed = pilot_quality["decision"] != "FAIL"'
        in source
    )
    assert "nll_prompts,\n            confirmation_prompt_metrics" in source
    assert pb._EXACT_PILOT_SEQLEN == 64
    assert pb._EXACT_CONFIRMATION_SEQLEN == 512


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


def test_parser_fixes_v6_profile_and_retires_repeat_threshold_knobs(
    tmp_path, monkeypatch
):
    model, inputs = _write_gold_payload(tmp_path, monkeypatch)
    base = [
        "--model", str(model),
        "--prompt-token-ids-json", str(inputs),
        "--output", str(tmp_path / "out.json"),
        "--campaign-id", "campaign-test-001",
    ]
    args = pb.parse_args(base)
    assert args.campaign_id == "campaign-test-001"
    assert args.warmup_triplets == 1
    assert args.expected_persistent_b_layers == 35
    assert args.expected_fp8_cb_moe_layers == 8
    assert args.tokenizer_mode == "deepseek_v4"
    assert args.kv_cache_memory_bytes == 268435456
    assert args.kv_cache_dtype == "fp8"
    assert args.pilot_full_vocab_seqlen == 64
    assert args.full_vocab_seqlen == 512
    assert args.full_vocab_kl is True
    assert args.teacher_full_vocab_kl is True
    for retired in (
        ["--quality-repeats", "2"],
        ["--max-mean-kl", "0.0001"],
        ["--measurement-only"],
        ["--timing-repeats", "2"],
    ):
        with pytest.raises(SystemExit):
            pb.parse_args([*base, *retired])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--n-samples", "7"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--seqlen", "511"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--pilot-full-vocab-seqlen", "63"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--full-vocab-seqlen", "511"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--campaign-id", "bad/id"])
    with pytest.raises(SystemExit):
        pb.parse_args([*base, "--kv-cache-dtype", "auto"])
    args.kv_cache_dtype = "auto"
    with pytest.raises(RuntimeError, match="requires kv_cache_dtype='fp8'"):
        pb.run(args)


def test_parser_accepts_legacy_full_vocab_spelling_but_rejects_teacher(
    tmp_path, monkeypatch
):
    model, inputs = _write_gold_payload(tmp_path, monkeypatch)
    args = pb.parse_args([
        "--model", str(model),
        "--prompt-token-ids-json", str(inputs),
        "--output", str(tmp_path / "out.json"),
        "--campaign-id", "campaign-test-002",
        "--full-vocab-kl",
    ])
    assert args.full_vocab_kl is True
    assert args.teacher_full_vocab_kl is True
    assert args.teacher_model is None
    with pytest.raises(SystemExit):
        pb.parse_args([*[
            "--model", str(model),
            "--prompt-token-ids-json", str(inputs),
            "--output", str(tmp_path / "out.json"),
            "--campaign-id", "campaign-test-003",
        ], "--teacher-model", str(model)])


def test_attempt_reservation_is_single_use_and_never_overwrites(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    prior = evidence / "prior.json"
    prior.write_bytes(b"prior-evidence\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pb._reserve_attempt(prior, "release-campaign-prior")
    assert prior.read_bytes() == b"prior-evidence\n"

    output = evidence / "attempt.json"
    reservation = pb._reserve_attempt(output, "release-campaign-001")
    reserved = json.loads(output.read_text(encoding="utf-8"))
    claim = json.loads(reservation.claim_path.read_text(encoding="utf-8"))
    assert reserved["status"] == "attempt_reserved"
    assert reserved["attempt"] == claim["attempt"] == reservation.attempt
    assert len(reservation.attempt["attempt_id"]) == 32

    retry_output = evidence / "retry.json"
    with pytest.raises(FileExistsError):
        pb._reserve_attempt(retry_output, "release-campaign-001")
    assert not retry_output.exists()

    pb._finalize_attempt(reservation, {
        "schema": pb.SCHEMA,
        "status": "inconclusive",
        "promotion_decision": "INCONCLUSIVE",
    })
    finalized = output.read_bytes()
    payload = json.loads(finalized)
    assert payload["attempt"] == reservation.attempt
    assert payload["attempt_immutability"] == {
        "single_attempt_per_campaign_in_evidence_directory": True,
        "prior_output_overwrite_refused": True,
        "campaign_claim_retained": True,
        "claim_path": str(reservation.claim_path),
        "claim_sha256": pb.hashlib.sha256(
            pb._canonical_json_bytes(reservation.claim_payload)
        ).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="no longer the live reservation"):
        pb._finalize_attempt(reservation, {"status": "replacement"})
    assert output.read_bytes() == finalized


def test_main_refuses_existing_output_before_running_validation(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    output.write_bytes(b"immutable-prior-report\n")
    args = SimpleNamespace(
        output=output, campaign_id="release-campaign-refusal"
    )
    called = False

    def forbidden_run(_args):
        nonlocal called
        called = True
        raise AssertionError("run must not execute after reservation refusal")

    monkeypatch.setattr(pb, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(pb, "run", forbidden_run)
    assert pb.main([]) == 1
    assert called is False
    assert output.read_bytes() == b"immutable-prior-report\n"
    assert list(tmp_path.glob(".persistent-b-campaign-*.claim.json")) == []


def test_main_exception_finalizes_machine_readable_single_attempt_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "nested" / "failure.json"
    args = SimpleNamespace(
        output=output, campaign_id="release-campaign-failure"
    )
    monkeypatch.setattr(pb, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        pb, "run", lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert pb.main([]) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == pb.SCHEMA
    assert payload["status"] == "gate_failed"
    assert payload["promotion_decision"] == "FAIL"
    assert payload["decision_reason"] == "integrity_violation"
    assert payload["error"] == {"type": "RuntimeError", "message": "boom"}
    assert "RuntimeError: boom" in payload["traceback"]
    assert payload["attempt"]["campaign_id"] == "release-campaign-failure"
    assert payload["attempt_immutability"]["campaign_claim_retained"] is True
    claim_path = Path(payload["attempt_immutability"]["claim_path"])
    assert claim_path.is_file()
    assert json.loads(claim_path.read_text(encoding="utf-8"))["attempt"] == (
        payload["attempt"]
    )
    leftovers = [path for path in output.parent.iterdir() if path != output]
    assert leftovers == [claim_path]


def test_shared_bootstrap_extension_is_backward_compatible_and_selectable():
    signature = inspect.signature(pb.validation_common.prepare_validation)
    assert signature.parameters["extension_loader"].default == (
        "get_fused_fp4_ext"
    )
    assert signature.parameters["required_symbol"].default is None
    assert signature.parameters["prompt_loader"].default is None
