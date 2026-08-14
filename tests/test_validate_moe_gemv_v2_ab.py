"""CPU contract tests for the exact-DSV4 CB-GEMV-v2 quality A/B."""

from __future__ import annotations

import hashlib
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
        candidate = root / "scripts" / "validate_moe_gemv_v2_ab.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("validate_moe_gemv_v2_ab.py")


SCRIPT = _script_path()
SPEC = importlib.util.spec_from_file_location("validate_moe_gemv_v2_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gemv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gemv
SPEC.loader.exec_module(gemv)


def test_release_profile_and_artifact_constants_are_value_closed():
    assert gemv._PROFILE_SAMPLES == 8
    assert gemv._PROFILE_SEQLEN == 16
    assert gemv._QUALITY_REPEATS == 2
    assert gemv._EXPECTED_PREFIX_TENSOR_SHA256 == (
        "11c40cbfd3819f72f18507f359787f479ff06d30fd6e30f697c3bc4e0b4b99f7"
    )
    assert gemv._EXPECTED_PREFIX_JSON_SHA256 == (
        "9ed265d2e7202f2282a225929e55faef2a0f87b4508fe8fca10378de500b8c85"
    )
    assert gemv._EXPECTED_ARTIFACT == {
        "config_sha256": (
            "cf07dd1184989ce21a3aa8a21815feb234bcec6385763ae2fcb2781bde015a89"
        ),
        "quant_config_sha256": (
            "6ecb5ffaca90a9ca3f5095df0d788381f9d3451dcb2c0ec5426ec736b26844ef"
        ),
        "codebook_sha256": (
            "ec893b2e56354d1ce8f8a5f613937a8b60b7cf7d2e08e590e114376cbabecc0c"
        ),
        "weight_sha256": (
            "d347c32304e50aa2e8904593744f1e38f7fa1a4a54aece47ae9b3b3e6c8b5334"
        ),
        "weight_bytes": 112_349_037_959,
        "codebook_relative_path": "cb_codebooks.pqcb",
    }
    assert len(gemv._EXPECTED_FP4_LAYER_IDS) == 35
    assert len(gemv._EXPECTED_FP8_LAYER_IDS) == 8
    assert gemv._EXPECTED_FP4_LAYER_IDS.isdisjoint(
        gemv._EXPECTED_FP8_LAYER_IDS
    )
    assert gemv._EXPECTED_FP4_LAYER_IDS | gemv._EXPECTED_FP8_LAYER_IDS == set(
        range(43)
    )


def test_decode_profile_is_exact_8x16_and_digest_guarded(monkeypatch):
    windows = [
        list(range(index * 512, (index + 1) * 512))
        for index in range(8)
    ]
    prompts = [window[:16] for window in windows]
    flattened = [token for prompt in prompts for token in prompt]
    tensor_digest = hashlib.sha256(
        struct.pack(f"<{len(flattened)}q", *flattened)
    ).hexdigest()
    json_digest = gemv.pb._canonical_sha256(prompts)
    monkeypatch.setattr(gemv, "_EXPECTED_PREFIX_TENSOR_SHA256", tensor_digest)
    monkeypatch.setattr(gemv, "_EXPECTED_PREFIX_JSON_SHA256", json_digest)
    observed, record = gemv._decode_profile(windows)
    assert observed == prompts
    assert record["n_samples"] == 8
    assert record["seqlen"] == 16
    assert record["positions_per_repeat"] == 120
    assert record["total_scored_positions"] == 240
    windows[0][0] += 1
    with pytest.raises(RuntimeError, match="not the canonical DSV4 decode"):
        gemv._decode_profile(windows)


def _provenance() -> dict:
    expected = gemv._EXPECTED_ARTIFACT
    return {
        "config": {"sha256": expected["config_sha256"]},
        "quant_config": {"sha256": expected["quant_config_sha256"]},
        "codebook_file": {
            "sha256": expected["codebook_sha256"],
            "relative_path": expected["codebook_relative_path"],
        },
        "weight_files": {
            "model.safetensors": {
                "sha256": expected["weight_sha256"],
                "bytes": expected["weight_bytes"],
            }
        },
    }


def test_artifact_gate_matches_every_exact_file_field():
    assert gemv._artifact_gate(_provenance())["pass"] is True
    changed = _provenance()
    changed["weight_files"]["model.safetensors"]["sha256"] = "0" * 64
    gate = gemv._artifact_gate(changed)
    assert gate["pass"] is False
    assert gate["checks"]["weight_sha256"] is False
    assert gemv._artifact_gate(None)["pass"] is False


def test_qualified_runtime_requires_exact_true_moe_skip_padding():
    assert gemv._moe_skip_padding_gate(True)["pass"] is True
    assert gemv._moe_skip_padding_gate(False)["pass"] is False
    assert gemv._moe_skip_padding_gate(1)["pass"] is False
    source = inspect.getsource(gemv.run)
    assert "vllm_envs.VLLM_MOE_SKIP_PADDING" in source
    assert "VLLM_MOE_SKIP_PADDING=True" in source


def test_gemv_provenance_names_only_existing_hashed_sources(tmp_path):
    package = tmp_path / "gridbook"
    csrc = package / "csrc"
    csrc.mkdir(parents=True)
    sources = {
        "moe_gemv_select.py": package / "moe_gemv_select.py",
        "cb_gemv.cu": csrc / "cb_gemv.cu",
        "cb_gemv_v2.cu": csrc / "cb_gemv_v2.cu",
    }
    for label, path in sources.items():
        path.write_text(f"// {label}\n", encoding="utf-8")
    helper = tmp_path / "validate_moe_persistent_b_ab.py"
    helper.write_text("# helper\n", encoding="utf-8")
    runtime = {
        "gridbook": {"package_root": str(package)},
        "source_files": {},
        "source_sha256": {},
        "harness": {"shared_helpers": {}},
    }
    original_helper_file = gemv.pb.__file__
    try:
        gemv.pb.__file__ = str(helper)
        gemv._augment_provenance(runtime)
    finally:
        gemv.pb.__file__ = original_helper_file
    assert set(runtime["source_files"]) == set(sources)
    for label, record in runtime["source_files"].items():
        path = Path(record["path"])
        assert path.is_file(), label
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert runtime["source_sha256"][label] == record["sha256"]
    helper_record = runtime["harness"]["shared_helpers"][
        "persistent_b_validation_api"
    ]
    assert Path(helper_record["path"]).is_file()
    assert helper_record["sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()


class _Shape:
    def __init__(self, *shape: int):
        self.shape = shape


class _Method:
    def __init__(self, index: int, *, fp4: bool):
        self.prefix = f"model.layers.{index}.ffn.experts"
        self.is_fp4 = fp4
        self.is_v2 = fp4
        self.k = 16 if index in gemv._EXPECTED_K16_LAYER_IDS else (
            18 if fp4 else 28
        )
        self.n_sub = 2 if fp4 else 4
        self.type_size = 4 * self.k + 9 if fp4 else 112

    def _apply_grouped_decode(self, layer, x, topk_weights, topk_ids, act):
        if self.is_fp4:
            op = (
                layer._test_ops.cb_moe_gemv_v2
                if layer._cb_use_v2_w13
                else layer._test_ops.cb_moe_gemv_fp4_v2
            )
            _invoke_op(op, stage="w13")
            op = (
                layer._test_ops.cb_moe_gemv_v2
                if layer._cb_use_v2_w2
                else layer._test_ops.cb_moe_gemv_fp4_v2
            )
            _invoke_op(op, stage="w2")
        else:
            _invoke_op(layer._test_ops.cb_moe_gemv_fp8, stage="w13")
            _invoke_op(layer._test_ops.cb_moe_gemv_fp8, stage="w13")
            _invoke_op(layer._test_ops.cb_moe_gemv_fp8, stage="w2")
        return (layer, x, topk_weights, topk_ids, act)


class _Layer:
    def __init__(self, *, fp4: bool):
        self._cb_use_v2_w13 = fp4
        self._cb_use_v2_w2 = fp4
        self._cb_hidden = 4096
        self._cb_inter = 2048
        self._cb_role_split = not fp4


def _fake_runtime():
    entries = {}
    registry = {}
    for index in range(43):
        fp4 = index in gemv._EXPECTED_FP4_LAYER_IDS
        entries[index] = (_Method(index, fp4=fp4), _Layer(fp4=fp4))
        registry[index] = object()

    def _op(name):
        def call(*args, **kwargs):
            return (name, args, kwargs)
        return call

    ops = SimpleNamespace(
        _LAYER_REGISTRY=registry,
        _lookup_cb_layer=lambda layer_id: entries[layer_id],
        cb_moe_gemv_fp4_v2=_op("inherited"),
        cb_moe_gemv_v2=_op("v2"),
        cb_moe_gemv_fp8=_op("fp8"),
    )
    for _method, layer in entries.values():
        layer._test_ops = ops
    moe = SimpleNamespace(PrismaQuantCBMoEMethod=_Method)
    return ops, moe, entries


def test_controller_attests_exact_inventory_and_restores_every_stack():
    ops, moe, _entries = _fake_runtime()
    controller = gemv.GemvV2ArmController(ops=ops, moe=moe)
    assert controller.inventory_gate["pass"] is True
    assert controller.inventory_gate["observed_fp4_stack_booleans"] == 70
    assert controller.inventory_gate["observed_fp8_stack_booleans"] == 16
    with controller.arm("baseline", label="test") as record:
        assert all(
            binding.layer._cb_use_v2_w13 is False
            and binding.layer._cb_use_v2_w2 is False
            for binding in controller.fp4
        )
    assert record["pass"] is True
    assert controller.restoration_gate()["pass"] is True
    assert all(
        binding.layer._cb_use_v2_w13 is True
        and binding.layer._cb_use_v2_w2 is True
        for binding in controller.fp4
    )


def test_controller_rejects_one_selector_fallback_at_load():
    ops, moe, entries = _fake_runtime()
    entries[0][1]._cb_use_v2_w2 = False
    with pytest.raises(RuntimeError, match="inventory is not exact"):
        gemv.GemvV2ArmController(ops=ops, moe=moe)


def _invoke_op(op, *, stage: str):
    tokens = 16
    pairs = tokens * gemv._EXPECTED_TOPK
    xq = _Shape(tokens if stage == "w13" else pairs, 4096)
    qw = _Shape(256, 4096, 1)
    pair = _Shape(pairs)
    return op(xq, qw, _Shape(1), _Shape(1), pair, pair)


def _simulate_request(
    *, arm: str, controller, probe, entries, ops
) -> tuple[dict, dict]:
    with controller.arm(arm, label=f"test:{arm}") as selector:
        with probe.request(label=f"test:{arm}", arm=arm, expected_tokens=16) as rec:
            for index in range(43):
                method, layer = entries[index]
                method._apply_grouped_decode(
                    layer, _Shape(16, 4096), _Shape(16, 6), _Shape(16, 6), "silu"
                )
    return selector, rec


@pytest.mark.parametrize("arm", gemv.ARMS)
def test_dispatch_probe_proves_exact_per_layer_and_cuda_op_routes(arm):
    ops, moe, entries = _fake_runtime()
    controller = gemv.GemvV2ArmController(ops=ops, moe=moe)
    originals = {
        name: getattr(ops, attr)
        for name, attr in gemv.GemvDispatchProbe._OP_ATTRS.items()
    }
    probe = gemv.GemvDispatchProbe(
        ops=ops, moe=moe, controller=controller
    )
    probe.install()
    selector, record = _simulate_request(
        arm=arm, controller=controller, probe=probe, entries=entries, ops=ops
    )
    assert selector["pass"] is True
    assert record["pass"] is True
    assert record["observed_layer_count"] == 43
    assert record["observed_op_counts"] == (
        {"inherited": 70, "fp8": 24}
        if arm == "baseline" else {"v2": 70, "fp8": 24}
    )
    assert len(record["fp8_signature"]) == 24
    probe.restore()
    assert probe.restoration_gate()["pass"] is True
    assert all(
        getattr(ops, attr) is originals[name]
        for name, attr in gemv.GemvDispatchProbe._OP_ATTRS.items()
    )


def test_dispatch_probe_fails_closed_on_unscoped_or_incomplete_request():
    ops, moe, _entries = _fake_runtime()
    controller = gemv.GemvV2ArmController(ops=ops, moe=moe)
    probe = gemv.GemvDispatchProbe(ops=ops, moe=moe, controller=controller)
    probe.install()
    with pytest.raises(RuntimeError, match="outside a request scope"):
        _invoke_op(ops.cb_moe_gemv_v2, stage="w13")
    with probe.request(label="empty", arm="fused", expected_tokens=16) as record:
        pass
    assert record["pass"] is False
    assert record["violations"]
    probe.restore()
    assert probe.restoration_gate()["pass"] is False
    assert probe.restoration_gate()["unscoped_calls"]


def _digest_record(*, score="a" * 64):
    return {
        "positions": 15,
        "vocab_size": 129_280,
        "row_sha256": ["b" * 64] * 15,
        "score_sha256": score,
        "prompt_token_ids_sha256": "c" * 64,
        "mean_nll": 1.0,
        "ppl": 2.718281828,
    }


def _quality_pairs():
    pairs = []
    for repeat in range(2):
        for prompt in range(8):
            block = repeat * 8 + prompt
            signature = [{"stage": "w13", "tokens": 16}]
            pairs.append({
                "prompt_index": block,
                "block_index": block,
                "repeat_index": repeat,
                "source_prompt_index": prompt,
                "pair_order": list(gemv.v5.paired_arm_order(repeat + prompt)),
                "score_digests": {
                    "baseline": _digest_record(),
                    "fused": _digest_record(),
                },
                "dispatch": {
                    "baseline": {"fp8_signature": signature},
                    "fused": {"fp8_signature": list(signature)},
                },
            })
    return pairs


def test_quality_evidence_gates_are_per_source_full_vocab_and_fp8_invariant():
    pairs = _quality_pairs()
    assert gemv.pb._pair_order_gate(pairs)["pass"] is True
    assert gemv.pb._full_vocab_repeat_determinism_gate(
        pairs, n_prompts=8, repeats=2
    )["pass"] is True
    assert gemv._score_cardinality_gate(
        pairs, vocab_size=129_280
    )["pass"] is True
    assert gemv._fp8_dispatch_invariance_gate(pairs)["pass"] is True
    pairs[8]["score_digests"]["fused"]["score_sha256"] = "0" * 64
    assert gemv.pb._full_vocab_repeat_determinism_gate(
        pairs, n_prompts=8, repeats=2
    )["pass"] is False
    pairs[0]["dispatch"]["fused"]["fp8_signature"] = []
    assert gemv._fp8_dispatch_invariance_gate(pairs)["pass"] is False


def test_parser_fixes_workload_engine_and_requires_explicit_gate(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    base = [
        "--model", str(model),
        "--prompt-token-ids-json", str(tmp_path / "inputs.json"),
        "--output", str(tmp_path / "out.json"),
    ]
    measured = gemv.parse_args([*base, "--measurement-only"])
    assert measured.n_samples == 8
    assert measured.seqlen == 16
    assert measured.teacher_full_vocab_kl is True
    assert measured.quantization == "gridbook"
    assert measured.tokenizer_mode == "deepseek_v4"
    assert measured.kv_cache_dtype == "fp8"
    args = gemv.parse_args([
        *base,
        "--max-mean-kl", "0.0001",
        "--max-mean-nll-regression", "0.00498754",
        "--max-ppl-relative-regression", "0.005",
    ])
    assert args.measurement_only is False
    with pytest.raises(SystemExit):
        gemv.parse_args(base)
    with pytest.raises(SystemExit):
        gemv.parse_args([*base, "--measurement-only", "--max-mean-kl", "0.1"])
    with pytest.raises(SystemExit):
        gemv.parse_args([
            *base, "--measurement-only", "--kv-cache-dtype", "auto"
        ])
    measured.kv_cache_dtype = "auto"
    with pytest.raises(RuntimeError, match="requires kv_cache_dtype='fp8'"):
        gemv.run(measured)


def test_harness_reuses_shared_loader_streaming_scorer_and_thresholds():
    source = inspect.getsource(gemv.run)
    assert "validation_common.prepare_validation" in source
    assert "validation_common.load_candidate_engine" in source
    assert "pb._compact_full_vocab_score" in source
    assert "v5._new_quality_accumulator" in source
    assert "v5._accumulate_quality_pair" in source
    assert "v5._finish_quality_accumulator" in source
    assert "v5._configured_gates" in source
    assert "PRISMAQUANT_CB_GEMV" not in inspect.getsource(
        gemv.GemvV2ArmController.arm
    )
    assert gemv.ARM_LABELS == {
        "baseline": "inherited_default_grouped_fp4_v2",
        "fused": "smem_resident_dictionary_v2",
    }


def test_environment_guard_requires_unmodified_inherited_default(monkeypatch):
    monkeypatch.setenv(gemv.GEMV_ENV, "v2")
    monkeypatch.setenv(gemv.DECODE_CONTRACT_ENV, "v1")
    for name in gemv.W2_SCHEDULE_ENVS:
        monkeypatch.delenv(name, raising=False)
    gemv._assert_measurement_environment()
    monkeypatch.setenv("PRISMAQUANT_CB_W2_SCHED", "rowpack")
    with pytest.raises(RuntimeError, match="changed mid-run"):
        gemv._assert_measurement_environment()
