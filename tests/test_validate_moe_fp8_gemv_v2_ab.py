"""CPU contracts for the exact-DSV4 routed-FP8 whole-row quality A/B."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _script_path() -> Path:
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        candidate = root / "scripts" / "validate_moe_fp8_gemv_v2_ab.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("validate_moe_fp8_gemv_v2_ab.py")


SCRIPT = _script_path()
SPEC = importlib.util.spec_from_file_location(
    "validate_moe_fp8_gemv_v2_ab", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
fp8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fp8
SPEC.loader.exec_module(fp8)


def test_contract_is_exact_artifact_full_vocab_and_closed_source_set():
    assert fp8.SCHEMA == "gridbook.moe-fp8-gemv-v2-ab.v1"
    assert fp8._PROFILE_SAMPLES == 8
    assert fp8._PROFILE_SEQLEN == 16
    assert fp8._QUALITY_REPEATS == 2
    assert fp8._EXPECTED_FP8_LAYER_IDS == {
        18, 19, 22, 33, 34, 35, 36, 39
    }
    assert len(fp8._REQUIRED_MANIFEST_LABELS) == 23
    assert "gridbook/csrc/cb_gemv.cu" in fp8._REQUIRED_MANIFEST_LABELS
    assert "gridbook/moe_routing.py" in fp8._REQUIRED_MANIFEST_LABELS
    assert "extension/main" in fp8._REQUIRED_MANIFEST_LABELS
    assert fp8._EXPECTED_ARTIFACT["weight_sha256"] == (
        "d347c32304e50aa2e8904593744f1e38f7fa1a4a54aece47ae9b3b3e6c8b5334"
    )


class _Shape:
    def __init__(self, *shape: int):
        self.shape = shape


class _Layer:
    def __init__(self, *, fp4: bool):
        self._cb_use_v2_w13 = fp4
        self._cb_use_v2_w2 = fp4
        self._cb_use_fp8_v2_w13 = not fp4
        self._cb_use_fp8_v2_w2 = not fp4
        self._cb_hidden = 4096
        self._cb_inter = 2048
        self._cb_role_split = not fp4


def _invoke(op, *, stage: str, fp8_layout: bool):
    tokens = 16
    pairs = tokens * fp8._EXPECTED_TOPK
    xq = _Shape(tokens if stage == "w13" else pairs, 4096)
    qw = _Shape(256, 2048 if stage == "w13" else 4096, 1)
    pair = _Shape(pairs)
    if fp8_layout:
        return op(xq, qw, _Shape(1), _Shape(1), pair, pair, 28, 4, 112)
    return op(xq, qw, _Shape(1), _Shape(1), pair, pair, 18, 81, 0, 0)


class _Method:
    def __init__(self, index: int, *, is_fp4: bool):
        self.prefix = f"model.layers.{index}.ffn.experts"
        self.is_fp4 = is_fp4
        self.is_v2 = is_fp4
        self.k = 16 if index in fp8._EXPECTED_K16_LAYER_IDS else (
            18 if is_fp4 else 28
        )
        self.n_sub = 2 if is_fp4 else 4
        self.type_size = 4 * self.k + 9 if is_fp4 else 112

    def _apply_grouped_decode(
        self, layer, x, topk_weights, topk_ids, act
    ):
        if self.is_fp4:
            _invoke(layer._ops.cb_moe_gemv_v2, stage="w13", fp8_layout=False)
            _invoke(layer._ops.cb_moe_gemv_v2, stage="w2", fp8_layout=False)
        else:
            w13 = (
                layer._ops.cb_moe_gemv_fp8_v2
                if layer._cb_use_fp8_v2_w13
                else layer._ops.cb_moe_gemv_fp8
            )
            w2 = (
                layer._ops.cb_moe_gemv_fp8_v2
                if layer._cb_use_fp8_v2_w2
                else layer._ops.cb_moe_gemv_fp8
            )
            _invoke(w13, stage="w13", fp8_layout=True)
            _invoke(w13, stage="w13", fp8_layout=True)
            _invoke(w2, stage="w2", fp8_layout=True)
        return (layer, x, topk_weights, topk_ids, act)


def _fake_runtime():
    entries = {}
    registry = {}
    for index in range(43):
        is_fp4 = index in fp8._EXPECTED_FP4_LAYER_IDS
        entries[index] = (_Method(index, is_fp4=is_fp4), _Layer(fp4=is_fp4))
        registry[index] = object()

    def fake_op(name):
        def call(*args, **kwargs):
            return (name, args, kwargs)
        return call

    ops = SimpleNamespace(
        _LAYER_REGISTRY=registry,
        _lookup_cb_layer=lambda layer_id: entries[layer_id],
        cb_moe_gemv_fp4_v2=fake_op("fp4_inherited"),
        cb_moe_gemv_v2=fake_op("fp4_v2"),
        cb_moe_gemv_fp8=fake_op("fp8_inherited"),
        cb_moe_gemv_fp8_v2=fake_op("fp8_v2"),
    )
    for _method, layer in entries.values():
        layer._ops = ops
    moe = SimpleNamespace(PrismaQuantCBMoEMethod=_Method)
    return ops, moe, entries


def _simulate_request(*, arm, controller, probe, entries):
    ids = torch.arange(16 * 6, dtype=torch.int64).reshape(16, 6) % 256
    weights = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6) / 100
    with controller.arm(arm, label=f"test:{arm}") as selector:
        with probe.request(
            label=f"test:{arm}", arm=arm, expected_tokens=16
        ) as record:
            for index in range(43):
                method, layer = entries[index]
                method._apply_grouped_decode(
                    layer, _Shape(16, 4096), weights, ids, "silu"
                )
    return selector, record


def test_controller_mutates_only_sixteen_fp8_stack_booleans_and_restores():
    ops, moe, entries = _fake_runtime()
    controller = fp8.FP8GemvV2ArmController(ops=ops, moe=moe)
    assert controller.inventory_gate["pass"] is True
    assert controller.inventory_gate["mutable_fp8_layer_count"] == 8
    assert controller.inventory_gate["mutable_fp8_attribute_count"] == 16
    before_fp4 = [
        (
            layer._cb_use_v2_w13,
            layer._cb_use_v2_w2,
            layer._cb_use_fp8_v2_w13,
            layer._cb_use_fp8_v2_w2,
        )
        for index, (_method, layer) in entries.items()
        if index in fp8._EXPECTED_FP4_LAYER_IDS
    ]
    with controller.arm("baseline", label="cpu") as record:
        assert all(
            not binding.layer._cb_use_fp8_v2_w13
            and not binding.layer._cb_use_fp8_v2_w2
            for binding in controller.fp8
        )
    assert record["pass"] is True
    assert controller.restoration_gate()["pass"] is True
    after_fp4 = [
        (
            layer._cb_use_v2_w13,
            layer._cb_use_v2_w2,
            layer._cb_use_fp8_v2_w13,
            layer._cb_use_fp8_v2_w2,
        )
        for index, (_method, layer) in entries.items()
        if index in fp8._EXPECTED_FP4_LAYER_IDS
    ]
    assert after_fp4 == before_fp4


def test_inventory_rejects_one_fp8_stack_fallback_at_load():
    ops, moe, entries = _fake_runtime()
    entries[18][1]._cb_use_fp8_v2_w2 = False
    with pytest.raises(RuntimeError, match="inventory is not exact"):
        fp8.FP8GemvV2ArmController(ops=ops, moe=moe)


@pytest.mark.parametrize("arm", fp8.ARMS)
def test_dispatch_probe_proves_exact_routes_cardinality_and_no_fallback(arm):
    ops, moe, entries = _fake_runtime()
    controller = fp8.FP8GemvV2ArmController(ops=ops, moe=moe)
    probe = fp8.FP8GemvDispatchProbe(
        ops=ops, moe=moe, controller=controller
    )
    probe.install()
    selector, record = _simulate_request(
        arm=arm, controller=controller, probe=probe, entries=entries
    )
    assert selector["pass"] is True
    assert record["pass"] is True
    assert record["observed_layer_count"] == 43
    assert record["fallback_or_wrong_route_count"] == 0
    assert record["expected_pair_count_per_op"] == 96
    assert record["observed_op_counts"] == (
        {"fp4_v2": 70, "fp8_inherited": 24}
        if arm == "baseline" else {"fp4_v2": 70, "fp8_v2": 24}
    )
    probe.restore()
    assert probe.restoration_gate()["pass"] is True


def _digest_record(value: str = "a"):
    return {
        "positions": 15,
        "vocab_size": 129_280,
        "row_sha256": [value * 64] * 15,
        "score_sha256": value * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "mean_nll": 1.0,
        "ppl": 2.718281828,
    }


def _quality_pairs():
    pairs = []
    for repeat in range(2):
        for prompt in range(8):
            block = repeat * 8 + prompt
            route = [{"prefix": "layer", "topk_ids": {"sha256": "d" * 64}}]
            pairs.append({
                "prompt_index": block,
                "block_index": block,
                "repeat_index": repeat,
                "source_prompt_index": prompt,
                "pair_order": list(fp8.v5.paired_arm_order(repeat + prompt)),
                "score_digests": {
                    "baseline": _digest_record(),
                    "fused": _digest_record(),
                },
                "generation_digests": {
                    "baseline": {"token_ids_sha256": "e" * 64},
                    "fused": {"token_ids_sha256": "e" * 64},
                },
                "dispatch": {
                    "baseline": {"route_signature": route},
                    "fused": {"route_signature": list(route)},
                },
            })
    return pairs


def test_exact_cross_arm_repeat_and_route_digests_are_primary_gates():
    pairs = _quality_pairs()
    assert fp8._cross_arm_digest_gate(pairs)["pass"] is True
    assert fp8._repeat_digest_gate(pairs)["pass"] is True
    assert fp8._route_invariance_gate(pairs)["pass"] is True
    pairs[0]["score_digests"]["fused"] = _digest_record("0")
    assert fp8._cross_arm_digest_gate(pairs)["pass"] is False
    pairs = _quality_pairs()
    pairs[8]["score_digests"]["fused"] = _digest_record("0")
    assert fp8._repeat_digest_gate(pairs)["pass"] is False
    pairs = _quality_pairs()
    pairs[0]["dispatch"]["fused"]["route_signature"] = []
    assert fp8._route_invariance_gate(pairs)["pass"] is False


def test_source_manifest_is_exact_and_detects_post_capture_change(tmp_path):
    main = tmp_path / "main.so"
    fp4_ext = tmp_path / "fp4.so"
    main.write_bytes(b"main")
    fp4_ext.write_bytes(b"fp4")
    runtime = {
        "gridbook": {"package_root": str(SCRIPT.parents[1] / "gridbook")},
        "source_files": {},
        "source_sha256": {},
        "harness": {},
    }
    manifest = fp8._build_source_manifest(
        runtime,
        main_extension=SimpleNamespace(__file__=str(main)),
        fp4_extension=SimpleNamespace(__file__=str(fp4_ext)),
    )
    assert set(manifest["files"]) == fp8._REQUIRED_MANIFEST_LABELS
    assert fp8._source_manifest_gate(manifest)["pass"] is True
    main.write_bytes(b"changed")
    gate = fp8._source_manifest_gate(manifest)
    assert gate["pass"] is False
    assert gate["violations"][0]["reason"] == (
        "file changed after manifest capture"
    )


def test_parser_fixes_engine_workload_and_exact_digest_promotion(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    base_args = [
        "--model", str(model),
        "--prompt-token-ids-json", str(tmp_path / "inputs.json"),
        "--output", str(tmp_path / "out.json"),
    ]
    args = fp8.parse_args(base_args)
    assert args.n_samples == 8
    assert args.seqlen == 16
    assert args.teacher_full_vocab_kl is True
    assert args.quantization == "gridbook"
    assert args.kv_cache_dtype == "fp8"
    measured = fp8.parse_args([*base_args, "--measurement-only"])
    assert measured.measurement_only is True
    with pytest.raises(SystemExit):
        fp8.parse_args([
            *base_args, "--measurement-only", "--max-mean-kl", "0"
        ])


def test_harness_reuses_shared_engine_prompt_and_full_vocab_helpers():
    source = inspect.getsource(fp8.run)
    assert "validation_common.prepare_validation" in source
    assert "validation_common.load_candidate_engine" in source
    assert "base._fixed_decode_prompt_loader" in source
    assert "pb._compact_full_vocab_score" in source
    assert "v5._new_quality_accumulator" in source
    assert "v5._accumulate_quality_pair" in source
    assert "v5._finish_quality_accumulator" in source
    controller_source = inspect.getsource(fp8.FP8GemvV2ArmController.arm)
    assert "PRISMAQUANT" not in controller_source
    assert fp8.ARM_LABELS == {
        "baseline": "inherited_routed_fp8_cb_gemv",
        "fused": "whole_row_routed_fp8_cb_gemv_v2",
    }
