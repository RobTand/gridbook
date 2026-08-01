"""CPU-only tests for the same-process dense fused-NVFP4 A/B harness."""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _script_path(name: str) -> Path:
    # Release jobs stage tests outside the checkout to exercise the installed
    # wheel; the validation scripts remain source utilities in that checkout.
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    candidates = [root / "scripts" / name for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not locate {name!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


SCRIPT = _script_path("validate_fused_nvfp4_ab.py")
SPEC = importlib.util.spec_from_file_location("validate_fused_nvfp4_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ab
SPEC.loader.exec_module(ab)

V5_HELP_SHA256 = "c7ac242795ca86dd2afbe88103d6f646780229412eb08b7d6c6f2edf187f2696"


class _Logprob:
    def __init__(self, value):
        self.logprob = value


def test_v5_schema_and_cli_help_are_compatibility_locked():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
    )
    assert ab.SCHEMA == "gridbook.fused-nvfp4-ab.v5"
    assert ab.ARMS == ("baseline", "fused")
    assert ab.DENSE_FUSED_MODES == (
        "1", "midm", "static_lsq", "static_lsq_midm",
        "rowwise", "rowwise_midm",
    )
    assert hashlib.sha256(completed.stdout).hexdigest() == V5_HELP_SHA256


def test_timing_phase_is_executed_before_allocation_heavy_quality():
    common = ab.validation_common
    assert common.measurement_phase_order(0) == ("quality",)
    assert common.measurement_phase_order(1) == ("timing", "quality")
    assert common.measurement_phase_order(7) == ("timing", "quality")
    with pytest.raises(ValueError, match="nonnegative"):
        common.measurement_phase_order(-1)
    # Both entry points consume this single ordering contract; the v6 test
    # separately locks its runner to the same helper.
    assert "validation_common.measurement_phase_order" in inspect.getsource(ab.run)


def test_timing_quiescence_collects_host_garbage_before_cuda_sync(monkeypatch):
    events = []
    monkeypatch.setattr(
        ab.validation_common.gc,
        "collect",
        lambda: events.append("gc") or 17,
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: events.append("cuda_sync"))
    )
    assert ab.validation_common.quiesce_before_timing(torch) == 17
    assert events == ["gc", "cuda_sync"]


def test_shared_engine_bootstrap_preserves_v5_llm_and_sampling_contract():
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(("llm", kwargs))
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(
                        enable_chunked_prefill=kwargs["enable_chunked_prefill"]
                    ),
                    model_config=SimpleNamespace(
                        is_chunked_prefill_supported=False,
                        runner_type="generate",
                    ),
                )
            )

    class FakeSampling:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(("sampling", kwargs))

    bootstrap = SimpleNamespace(
        candidate_load_revision="resolved-commit",
        quality_logprobs=1024,
        llm_class=FakeLLM,
        sampling_params_class=FakeSampling,
    )
    args = SimpleNamespace(
        model="candidate",
        trust_remote_code=False,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        seqlen=128,
        enable_chunked_prefill=False,
        seed=7,
        quantization="gridbook",
        max_num_batched_tokens=256,
    )
    probe = SimpleNamespace(restore=lambda: calls.append(("restore", {})))
    engine = ab.validation_common.load_candidate_engine(
        bootstrap, args, probe=probe, attest_chunked_prefill=True
    )
    llm_kwargs = calls[0][1]
    assert llm_kwargs == {
        "model": "candidate",
        "trust_remote_code": False,
        "revision": "resolved-commit",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 144,
        "max_num_seqs": 1,
        "max_logprobs": 1024,
        "enforce_eager": True,
        "disable_log_stats": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "seed": 7,
        "quantization": "gridbook",
        "max_num_batched_tokens": 256,
    }
    assert engine.quality_sampling.kwargs == {
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 1024,
        "detokenize": False,
    }
    assert engine.timing_sampling.kwargs == {
        "max_tokens": 1,
        "temperature": 0.0,
        "detokenize": False,
    }
    assert engine.chunked_prefill_contract["requested"] == "disable"
    assert engine.chunked_prefill_contract["resolved_enabled"] is False
    assert engine.chunked_prefill_contract["promotion_compatible"] is True
    assert all(name != "restore" for name, _payload in calls)


def test_v5_run_requires_resolved_chunked_prefill_attestation():
    source = inspect.getsource(ab.run)
    assert "attest_chunked_prefill=True" in source
    assert "chunked_prefill_contract" in source


def test_shared_chunked_prefill_gate_is_fail_closed_and_schema_shared():
    compatible = {
        "requested": "auto",
        "resolved_enabled": True,
        "promotion_compatible": True,
    }
    incompatible = {
        **compatible,
        "promotion_compatible": False,
    }
    assert ab.validation_common.chunked_prefill_integrity_gate(
        compatible
    )["pass"] is True
    assert ab.validation_common.chunked_prefill_integrity_gate(
        incompatible
    )["pass"] is False
    assert "validation_common.chunked_prefill_integrity_gate" in inspect.getsource(
        ab.run
    )


def test_pair_order_is_prompt_paired_and_counterbalanced():
    assert ab.paired_arm_order(0) == ("baseline", "fused")
    assert ab.paired_arm_order(1) == ("fused", "baseline")
    assert ab.paired_arm_order(2) == ("baseline", "fused")
    with pytest.raises(ValueError, match="nonnegative"):
        ab.paired_arm_order(-1)


def test_activate_arm_mutates_env_and_clears_cached_mode():
    environ = {ab.FUSED_ENV: "stale"}
    cache = ["stale"]
    assert ab.activate_arm(
        "baseline", fused_mode="1", environ=environ, mode_cache=cache
    ) == ""
    assert environ[ab.FUSED_ENV] == ""
    assert cache == []

    cache.append("")
    assert ab.activate_arm(
        "fused", fused_mode="midm", environ=environ, mode_cache=cache
    ) == "midm"
    assert environ[ab.FUSED_ENV] == "midm"
    assert cache == []
    assert ab.activate_arm(
        "fused", fused_mode="rowwise", environ=environ, mode_cache=cache
    ) == "rowwise"
    assert environ[ab.FUSED_ENV] == "rowwise"
    assert cache == []
    assert ab.activate_arm(
        "fused", fused_mode="static_lsq", environ=environ, mode_cache=cache
    ) == "static_lsq"
    assert environ[ab.FUSED_ENV] == "static_lsq"
    assert cache == []
    with pytest.raises(ValueError, match="unknown arm"):
        ab.activate_arm("other", fused_mode="1", environ=environ, mode_cache=cache)


def test_activate_execution_arm_keeps_moe_prefill_unset():
    environ = {ab.FUSED_ENV: "stale", ab.FUSED_MOE_ENV: "stale"}
    dense_cache = ["stale"]
    moe_cache = ["stale"]
    selected = ab.activate_execution_arm(
        "baseline",
        execution_mode="moe128",
        dense_fused_mode="1",
        environ=environ,
        dense_mode_cache=dense_cache,
        moe_mode_cache=moe_cache,
    )
    assert selected == ""
    assert environ[ab.FUSED_MOE_ENV] == ""
    assert environ[ab.FUSED_ENV] == ""
    assert ab.PREFILL_ENV not in environ
    assert dense_cache == []
    assert moe_cache == []

    assert ab.activate_execution_arm(
        "fused",
        execution_mode="moe256",
        dense_fused_mode="1",
        environ=environ,
        dense_mode_cache=dense_cache,
        moe_mode_cache=moe_cache,
    ) == "256"
    assert environ[ab.FUSED_MOE_ENV] == "256"
    environ[ab.PREFILL_ENV] = "loop"
    with pytest.raises(RuntimeError, match="must remain unset"):
        ab.activate_execution_arm(
            "fused",
            execution_mode="moe128",
            dense_fused_mode="1",
            environ=environ,
            dense_mode_cache=dense_cache,
            moe_mode_cache=moe_cache,
        )


def test_scoped_arms_exercise_real_selectors_and_restore_on_error():
    """The A/B-only escape hatch must not weaken production selectors.

    Run in a child so the minimal vLLM import stubs and deliberate mutation of
    process-lifetime selector caches cannot leak into another test module.
    """

    source_root = SCRIPT.parents[1]
    child = r'''
import importlib.util
import os
import sys
import types

import torch


def module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value


module("vllm")
module("vllm.model_executor")
utils = module("vllm.model_executor.utils")
utils.set_weight_attrs = lambda *args, **kwargs: None
module("vllm.model_executor.layers")
linear_api = module("vllm.model_executor.layers.linear")
linear_api.LinearMethodBase = type("LinearMethodBase", (), {})
linear_api.register_weight_loader_v2_supported_method = lambda cls: cls
parameters = module("vllm.model_executor.parameter")
parameters.ChannelQuantScaleParameter = type("ChannelQuantScaleParameter", (), {})
parameters.ModelWeightParameter = type("ModelWeightParameter", (), {})
parameters.PerTensorScaleParameter = type("PerTensorScaleParameter", (), {})
fused_moe = module("vllm.model_executor.layers.fused_moe")
fused_moe.RoutedExperts = type("RoutedExperts", (), {})
moe_config = module("vllm.model_executor.layers.fused_moe.config")
moe_config.FusedMoEConfig = type("FusedMoEConfig", (), {})
moe_config.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
moe_base = module(
    "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
)
moe_base.FusedMoEMethodBase = type("FusedMoEMethodBase", (), {})
activation = module("vllm.model_executor.layers.fused_moe.activation")
activation.MoEActivation = type("MoEActivation", (), {})
activation.apply_moe_activation = lambda *args, **kwargs: None

script = os.environ["GRIDBOOK_AB_SCRIPT"]
spec = importlib.util.spec_from_file_location("gridbook_ab_child", script)
ab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ab
spec.loader.exec_module(ab)
from gridbook import linear, moe

os.environ[ab.FUSED_ENV] = "midm"
os.environ[ab.FUSED_MOE_ENV] = "256"
linear._FP4_FUSED_MODE[:] = ["midm"]
moe._FUSED_FP4_MOE_STATE[:] = ["256"]

class Expected(Exception):
    pass

try:
    with ab.scoped_execution_arm(
        "fused",
        execution_mode="dense",
        dense_fused_mode="1",
        environ=os.environ,
        dense_mode_cache=linear._FP4_FUSED_MODE,
        moe_mode_cache=moe._FUSED_FP4_MOE_STATE,
        dense_selector=linear._fp4_fused_mode,
        moe_selector=moe._requested_fused_fp4_moe_mode,
    ):
        assert linear._fp4_fused_mode() == "1"
        assert moe._requested_fused_fp4_moe_mode() == ""
        raise Expected
except Expected:
    pass

assert os.environ[ab.FUSED_ENV] == "midm"
assert os.environ[ab.FUSED_MOE_ENV] == "256"
assert linear._FP4_FUSED_MODE == ["midm"]
assert moe._FUSED_FP4_MOE_STATE == ["256"]

with ab.scoped_execution_arm(
    "fused",
    execution_mode="moe128",
    dense_fused_mode="1",
    environ=os.environ,
    dense_mode_cache=linear._FP4_FUSED_MODE,
    moe_mode_cache=moe._FUSED_FP4_MOE_STATE,
    dense_selector=linear._fp4_fused_mode,
    moe_selector=moe._requested_fused_fp4_moe_mode,
):
    assert linear._fp4_fused_mode() == ""
    assert moe._requested_fused_fp4_moe_mode() == "128"

# Outside the explicit harness scope, the real production selectors still
# reject a changed execution contract in this same process.
os.environ[ab.FUSED_ENV] = "1"
try:
    linear._fp4_fused_mode()
except RuntimeError:
    pass
else:
    raise AssertionError("dense selector stopped failing closed")
os.environ[ab.FUSED_ENV] = "midm"
os.environ[ab.FUSED_MOE_ENV] = "128"
try:
    moe._requested_fused_fp4_moe_mode()
except RuntimeError:
    pass
else:
    raise AssertionError("MoE selector stopped failing closed")
'''
    env = os.environ.copy()
    env["GRIDBOOK_AB_SCRIPT"] = str(SCRIPT)
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_logprob_and_topk_helpers_accept_vllm_shapes():
    entries = {
        7: _Logprob(-0.1),
        "8": {"logprob": -0.2},
        9: (-0.3, "decoded"),
    }
    assert ab.target_logprob(entries, 7) == pytest.approx(-0.1)
    assert ab.target_logprob(entries, 8) == pytest.approx(-0.2)
    row = ab.topk_row(entries, 2)
    assert row.token_ids == (7, 8)
    assert row.logprobs == pytest.approx((-0.1, -0.2))
    assert row.coverage == pytest.approx(math.exp(-0.1) + math.exp(-0.2))


def test_coarse_topk_kl_is_zero_for_equal_rows_and_positive_for_change():
    baseline = ab.TopKRow((1, 2), (math.log(0.6), math.log(0.3)))
    equal = ab.TopKRow((1, 2), (math.log(0.6), math.log(0.3)))
    changed = ab.TopKRow((1, 3), (math.log(0.45), math.log(0.35)))
    assert ab.coarse_topk_kl(baseline, equal) == pytest.approx(0.0, abs=1e-12)
    assert ab.coarse_topk_kl(baseline, changed) > 0.0
    assert math.isfinite(ab.coarse_topk_kl(changed, baseline))


def test_teacher_pairwise_summary_identifies_better_candidate():
    teacher = ab.PromptScore(
        (math.log(0.8), math.log(0.7)),
        (
            ab.TopKRow((1, 2), (math.log(0.8), math.log(0.1))),
            ab.TopKRow((3, 4), (math.log(0.7), math.log(0.2))),
        ),
    )
    close = teacher
    shifted = ab.PromptScore(
        (math.log(0.4), math.log(0.3)),
        (
            ab.TopKRow((1, 2), (math.log(0.4), math.log(0.3))),
            ab.TopKRow((3, 4), (math.log(0.3), math.log(0.3))),
        ),
    )
    exact = ab._pairwise_score_summary(
        [teacher], [close], reference_name="teacher", candidate_name="baseline"
    )
    changed = ab._pairwise_score_summary(
        [teacher], [shifted], reference_name="teacher", candidate_name="fused"
    )
    assert exact["mean_nll_delta_candidate_minus_reference"] == pytest.approx(0.0)
    assert exact["kl_reference_to_candidate"]["mean"] == pytest.approx(0.0)
    assert changed["mean_nll_delta_candidate_minus_reference"] > 0.0
    assert changed["kl_reference_to_candidate"]["mean"] > 0.0


def test_v5_quality_and_timing_report_shape_remains_compatible():
    row = ab.TopKRow((1, 2), (math.log(0.6), math.log(0.3)))
    score = ab.PromptScore((math.log(0.6),), (row,))
    quality = ab._quality_summary([{
        "prompt_index": 0,
        "pair_order": ["baseline", "fused"],
        "scores": {"baseline": score, "fused": score},
        "wall_ms": {"baseline": 2.0, "fused": 1.0},
    }])
    assert tuple(quality) == (
        "arms",
        "delta",
        "per_prompt",
        "kl_mode",
        "kl_convention",
    )
    assert tuple(quality["arms"]) == ("baseline", "fused")
    assert set(quality["delta"]) == {
        "mean_nll_fused_minus_baseline",
        "ppl_fused_over_baseline",
        "ppl_relative_regression",
        "target_logprob_abs_delta",
        "kl_baseline_to_fused",
        "kl_fused_to_baseline",
        "kl_baseline_to_fused_confident_positions",
    }
    timing = ab._timing_summary({"baseline": [2.0], "fused": [1.0]})
    assert tuple(timing) == (
        "metric",
        "scope",
        "is_streaming_ttft",
        "arms",
        "baseline_over_fused_speedup",
    )
    assert timing["baseline_over_fused_speedup"] == pytest.approx(2.0)


def test_score_prompt_output_scores_targets_and_drops_position_zero():
    class Output:
        prompt_token_ids = [1, 2, 3]
        prompt_logprobs = [
            None,
            {2: _Logprob(math.log(0.5)), 8: _Logprob(math.log(0.4))},
            {3: _Logprob(math.log(0.25)), 9: _Logprob(math.log(0.7))},
        ]

    score = ab.score_prompt_output(Output(), [1, 2, 3], top_k=1)
    assert score.target_logprobs == pytest.approx(
        (math.log(0.5), math.log(0.25))
    )
    assert tuple(row.token_ids for row in score.rows) == ((2,), (9,))
    assert score.mean_nll == pytest.approx(-math.log(math.sqrt(0.125)))


def test_full_vocab_rows_are_cardinality_checked_and_use_exact_kl():
    reference = ab.full_vocab_row(
        {
            0: _Logprob(math.log(0.5)),
            1: _Logprob(math.log(0.3)),
            2: _Logprob(math.log(0.2)),
        },
        3,
    )
    candidate = ab.full_vocab_row(
        {
            "0": _Logprob(math.log(0.4)),
            "1": _Logprob(math.log(0.4)),
            "2": _Logprob(math.log(0.2)),
        },
        3,
    )
    expected = 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.4)
    assert reference.token_ids == (0, 1, 2)
    assert ab.exact_full_vocab_kl(reference, candidate) == pytest.approx(expected)
    summary = ab._pairwise_score_summary(
        [ab.PromptScore((math.log(0.5),), (reference,))],
        [ab.PromptScore((math.log(0.4),), (candidate,))],
        reference_name="teacher",
        candidate_name="fused",
        kl_mode=ab.KL_FULL_VOCAB,
    )
    assert summary["kl_mode"] == ab.KL_FULL_VOCAB
    assert summary["kl_reference_to_candidate"]["mean"] == pytest.approx(expected)
    with pytest.raises(RuntimeError, match="cardinality mismatch"):
        ab.full_vocab_row({0: _Logprob(-0.1), 2: _Logprob(-0.2)}, 3)
    with pytest.raises(RuntimeError, match="out-of-range"):
        ab.full_vocab_row(
            {0: _Logprob(-0.1), 1: _Logprob(-0.2), 3: _Logprob(-0.3)}, 3
        )


def test_score_prompt_output_full_vocab_rejects_partial_vllm_rows():
    class Output:
        prompt_token_ids = [0, 1]
        prompt_logprobs = [
            None,
            {0: _Logprob(-1.0), 1: _Logprob(-0.5)},
        ]

    with pytest.raises(RuntimeError, match="cardinality mismatch"):
        ab.score_prompt_output(
            Output(),
            [0, 1],
            top_k=1,
            full_vocab=True,
            expected_vocab_size=3,
        )


def test_score_prompt_output_attests_exact_prompt_ids_and_position_count():
    class WrongIds:
        prompt_token_ids = [9, 2]
        prompt_logprobs = [None, {2: _Logprob(-0.1)}]

    with pytest.raises(RuntimeError, match="different from the submitted prompt"):
        ab.score_prompt_output(WrongIds(), [1, 2], top_k=1)

    class ExtraPosition:
        prompt_token_ids = [1, 2]
        prompt_logprobs = [
            None,
            {2: _Logprob(-0.1)},
            {3: _Logprob(-0.2)},
        ]

    with pytest.raises(RuntimeError, match="3 prompt positions for exactly 2"):
        ab.score_prompt_output(ExtraPosition(), [1, 2], top_k=1)

    class NoPromptIds:
        prompt_logprobs = [None, {2: _Logprob(-0.1)}]

    with pytest.raises(RuntimeError, match="did not return prompt_token_ids"):
        ab.score_prompt_output(NoPromptIds(), [1, 2], top_k=1)


def test_score_teacher_prompt_tiny_torch_fixture_topk_and_full_vocab():
    torch = pytest.importorskip("torch")

    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.bfloat16)
            )

        def forward(self, input_ids, use_cache=False):
            assert use_cache is False
            rows = torch.tensor(
                [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]],
                device=input_ids.device,
            )
            return SimpleNamespace(logits=rows[None, : input_ids.shape[1]])

    model = TinyTeacher()
    prompt = [0, 2, 1]
    full = ab.score_teacher_prompt(
        model,
        prompt,
        top_k=1,
        torch=torch,
        full_vocab=True,
        expected_vocab_size=3,
    )
    expected_logprobs = torch.log_softmax(
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]), dim=-1
    )
    assert full.target_logprobs == pytest.approx(
        (float(expected_logprobs[0, 2]), float(expected_logprobs[1, 1]))
    )
    assert all(row.token_ids == (0, 1, 2) for row in full.rows)
    assert full.rows[0].logprobs == pytest.approx(
        expected_logprobs[0].tolist()
    )

    topk = ab.score_teacher_prompt(
        model, prompt, top_k=1, torch=torch, expected_vocab_size=3
    )
    assert tuple(row.token_ids for row in topk.rows) == ((2,), (0,))
    with pytest.raises(RuntimeError, match="attested candidate vocab"):
        ab.score_teacher_prompt(
            model, prompt, top_k=1, torch=torch, expected_vocab_size=4
        )


class _Config:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


class _BackendTokenizer:
    def __init__(self, payload):
        self.payload = payload

    def to_str(self):
        return self.payload


class _Tokenizer:
    vocab_size = 3
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1
    unk_token_id = None
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None

    def __init__(self, backend="same"):
        self.backend_tokenizer = _BackendTokenizer(backend)

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2}

    def __len__(self):
        return 3


def test_teacher_config_and_tokenizer_identity_are_fail_closed():
    base = {
        "model_type": "tiny",
        "architectures": ["TinyForCausalLM"],
        "vocab_size": 3,
        "hidden_size": 4,
        "_name_or_path": "/candidate",
        "quantization_config": {"quant_method": "gridbook"},
    }
    teacher = {
        **base,
        "_name_or_path": "/teacher",
        "quantization_config": None,
    }
    identity = ab._assert_config_identity(_Config(base), _Config(teacher))
    assert identity["match"] is True
    assert identity["teacher_quantization_config_absent"] is True

    mismatched = {**teacher, "hidden_size": 8}
    with pytest.raises(RuntimeError, match="differing keys.*hidden_size"):
        ab._assert_config_identity(_Config(base), _Config(mismatched))
    quantized_teacher = {
        **teacher,
        "quantization_config": {"quant_method": "other"},
    }
    with pytest.raises(RuntimeError, match="declares quantization_config"):
        ab._assert_config_identity(_Config(base), _Config(quantized_teacher))

    assert ab._assert_tokenizer_identity(_Tokenizer(), _Tokenizer())["match"]
    with pytest.raises(RuntimeError, match="backend_sha256"):
        ab._assert_tokenizer_identity(_Tokenizer(), _Tokenizer("different"))


def test_local_model_provenance_hashes_weights_tokenizer_and_codebook(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text('{"model_type":"tiny"}\n')
    (teacher / "tokenizer.json").write_text('{"version":"1"}\n')
    (teacher / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (teacher / "model-00002-of-00002.safetensors").write_bytes(b"second")
    index = {
        "weight_map": {
            "model.a": "model-00001-of-00002.safetensors",
            "model.b": "model-00002-of-00002.safetensors",
        }
    }
    (teacher / "model.safetensors.index.json").write_text(json.dumps(index))
    codebook = teacher / "sidecars" / "tables.pqcb"
    codebook.parent.mkdir()
    codebook.write_bytes(b"exact codebook payload")
    quant_config = {"codebook_file": "sidecars/tables.pqcb"}
    (teacher / "quant_config.json").write_text(json.dumps(quant_config))

    provenance = ab._local_teacher_provenance(teacher)
    assert provenance["kind"] == "local_directory"
    assert provenance["role"] == "teacher"
    assert provenance["safetensors_index"]["sha256"]
    assert set(provenance["weight_files"]) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert provenance["weight_bytes"] == len(b"firstsecond")
    assert provenance["tokenizer_files"]["tokenizer.json"]["sha256"]
    candidate = ab._local_model_provenance(teacher, role="candidate")
    assert candidate["role"] == "candidate"
    assert candidate["config"] == provenance["config"]
    assert candidate["weight_files"] == provenance["weight_files"]
    assert candidate["quant_config"]["sha256"] == ab._sha256(
        teacher / "quant_config.json"
    )
    assert candidate["codebook_file"]["relative_path"] == (
        "sidecars/tables.pqcb"
    )
    assert candidate["codebook_file"]["sha256"] == ab._sha256(codebook)


def test_candidate_quant_sidecars_are_required_and_contained(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}\n')
    (model / "tokenizer.json").write_text('{}\n')
    (model / "model.safetensors").write_bytes(b"weights")

    teacher = ab._local_model_provenance(model, role="teacher")
    assert teacher["quant_config"] is None
    assert teacher["codebook_file"] is None
    with pytest.raises(RuntimeError, match="no required quant_config.json"):
        ab._local_model_provenance(model, role="candidate")

    quant_config_path = model / "quant_config.json"
    quant_config_path.write_text("[]")
    with pytest.raises(RuntimeError, match="must contain a JSON object"):
        ab._local_model_provenance(model, role="candidate")

    quant_config_path.write_text(json.dumps({"codebook_file": "../outside.pqcb"}))
    with pytest.raises(RuntimeError, match="escapes model directory"):
        ab._local_model_provenance(model, role="candidate")

    outside = tmp_path / "outside.pqcb"
    outside.write_bytes(b"outside")
    (model / "linked.pqcb").symlink_to(outside)
    quant_config_path.write_text(json.dumps({"codebook_file": "linked.pqcb"}))
    with pytest.raises(RuntimeError, match="resolves outside"):
        ab._local_model_provenance(model, role="candidate")

    inside = model / "inside.pqcb"
    inside.write_bytes(b"inside")
    quant_config_path.write_text(json.dumps({"codebook_file": "inside.pqcb"}))
    provenance = ab._local_model_provenance(model, role="candidate")
    assert provenance["codebook_file"]["sha256"] == ab._sha256(inside)


def test_hub_model_provenance_requires_resolved_transformers_commit():
    resolved_commit = "0123456789ABCDEF" * 2 + "01234567"
    record = ab._hub_model_provenance(
        SimpleNamespace(_commit_hash=f" {resolved_commit} "),
        role="candidate",
        model_id="org/model",
        requested_revision="release-tag",
    )
    assert record["requested_revision"] == "release-tag"
    assert record["resolved_commit_hash"] == resolved_commit.lower()

    with pytest.raises(RuntimeError, match="no explicit requested revision"):
        ab._hub_model_provenance(
            SimpleNamespace(_commit_hash=resolved_commit),
            role="candidate",
            model_id="org/model",
            requested_revision="",
        )
    for config in (
        SimpleNamespace(),
        SimpleNamespace(_commit_hash="   "),
        SimpleNamespace(_commit_hash="main"),
        SimpleNamespace(_commit_hash="a" * 39),
        SimpleNamespace(_commit_hash="a" * 41),
        SimpleNamespace(_commit_hash="a" * 64),
        SimpleNamespace(_commit_hash="a" * 20 + " " + "a" * 19),
        SimpleNamespace(_commit_hash="g" * 40),
    ):
        with pytest.raises(RuntimeError, match="full 40-hex.*_commit_hash"):
            ab._hub_model_provenance(
                config,
                role="teacher",
                model_id="org/teacher",
                requested_revision="release-tag",
            )


def test_local_tokenizer_files_must_match_exactly(tmp_path):
    candidate = tmp_path / "candidate"
    teacher = tmp_path / "teacher"
    candidate.mkdir()
    teacher.mkdir()
    for directory in (candidate, teacher):
        (directory / "tokenizer.json").write_text('{"same":true}\n')
        (directory / "tokenizer_config.json").write_text('{}\n')
    record = ab._assert_local_tokenizer_files_match(candidate, teacher)
    assert record["match"] is True
    (teacher / "tokenizer_config.json").write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="tokenizer files differ"):
        ab._assert_local_tokenizer_files_match(candidate, teacher)


def test_runtime_provenance_hashes_the_actual_imported_package(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "gridbook"
    csrc = package / "csrc"
    (csrc / "cutlass_fork").mkdir(parents=True)
    for relative in (
        "__init__.py",
        "config.py",
        "linear.py",
        "moe.py",
        "moe_toplevel_loader.py",
        "cb_fill_guard.py",
        "cuda_ext.py",
        "plugin.py",
        "ops.py",
        "codec.py",
        "csrc/cb_fused_fp4_gemm.cu",
        "csrc/cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    ):
        path = package / relative
        path.write_text(f"actual imported payload: {relative}\n", encoding="utf-8")
    vllm_path = tmp_path / "installed" / "vllm" / "__init__.py"
    vllm_path.parent.mkdir()
    vllm_path.write_text("__version__ = 'test'\n", encoding="utf-8")
    harness = tmp_path / "harness.py"
    harness.write_text("# exact harness\n", encoding="utf-8")

    gridbook = SimpleNamespace(
        __file__=str(package / "__init__.py"), __version__="9.9.test"
    )
    gridbook_config = SimpleNamespace(__file__=str(package / "config.py"))
    linear = SimpleNamespace(__file__=str(package / "linear.py"))
    moe = SimpleNamespace(__file__=str(package / "moe.py"))
    moe_toplevel_loader = SimpleNamespace(
        __file__=str(package / "moe_toplevel_loader.py")
    )
    cb_fill_guard = SimpleNamespace(__file__=str(package / "cb_fill_guard.py"))
    cuda_ext = SimpleNamespace(
        __file__=str(package / "cuda_ext.py"), csrc_dir=lambda: str(csrc)
    )
    vllm = SimpleNamespace(__file__=str(vllm_path), __version__="0.test")

    props = SimpleNamespace(name="fake-cpu-fixture", total_memory=123)
    fake_cuda = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _device: props,
        get_device_capability=lambda: (12, 1),
    )
    fake_torch = SimpleNamespace(
        __version__="torch.test",
        version=SimpleNamespace(cuda="cuda.test"),
        cuda=fake_cuda,
    )
    monkeypatch.setattr(ab, "_command_output", lambda *args: None)
    record = ab._runtime_provenance(
        fake_torch,
        vllm,
        gridbook,
        gridbook_config,
        linear,
        moe,
        moe_toplevel_loader,
        cb_fill_guard,
        cuda_ext,
        harness,
    )

    assert record["gridbook"]["module_path"] == str(
        (package / "__init__.py").resolve()
    )
    assert record["gridbook"]["version"] == "9.9.test"
    assert record["gridbook"]["git"]["dirty"] is None
    assert record["source_files"]["linear.py"]["path"] == str(
        (package / "linear.py").resolve()
    )
    assert record["source_sha256"]["linear.py"] == ab._sha256(
        package / "linear.py"
    )
    for relative in (
        "config.py",
        "moe_toplevel_loader.py",
        "cb_fill_guard.py",
    ):
        assert record["source_files"][relative]["path"] == str(
            (package / relative).resolve()
        )
        assert record["source_sha256"][relative] == ab._sha256(package / relative)
    assert record["harness"]["sha256"] == ab._sha256(harness)
    shared = record["harness"]["shared_helpers"]["fused_nvfp4_validation"]
    assert shared["path"] == str(ab._COMMON_PATH)
    assert shared["sha256"] == ab._sha256(ab._COMMON_PATH)

    (package / "moe_toplevel_loader.py").unlink()
    with pytest.raises(RuntimeError, match="moe_toplevel_loader.*unreadable"):
        ab._runtime_provenance(
            fake_torch,
            vllm,
            gridbook,
            gridbook_config,
            linear,
            moe,
            moe_toplevel_loader,
            cb_fill_guard,
            cuda_ext,
            harness,
        )


def test_teacher_actual_dtype_and_no_quantization_attestation():
    torch = pytest.importorskip("torch")

    class Teacher(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
            self.register_buffer("routing_bias", torch.zeros(2, dtype=torch.float32))
            self.config = _Config({"quantization_config": None})

    record = ab._attest_teacher_model(Teacher(torch.bfloat16), torch)
    assert record["parameters_all_bfloat16"] is True
    assert record["parameter_tensors_by_dtype"] == {"torch.bfloat16": 1}
    assert record["buffer_tensors_by_dtype"] == {"torch.float32": 1}
    with pytest.raises(RuntimeError, match="not uniformly BF16"):
        ab._attest_teacher_model(Teacher(torch.float32), torch)


def test_percentile_and_summary_are_deterministic():
    assert ab.percentile([], 0.5) is None
    assert ab.percentile([1.0], 0.95) == 1.0
    assert ab.percentile([1.0, 3.0], 0.5) == 2.0
    summary = ab.summarize_values([3.0, 1.0, 2.0])
    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["p50"] == 2.0
    with pytest.raises(ValueError, match="non-finite"):
        ab.summarize_values([float("nan")])


def test_raw_wikitext_path_needs_no_datasets_dependency(tmp_path):
    text_path = tmp_path / "wiki.test.raw"
    text_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text)))}

    args = SimpleNamespace(
        wikitext_text=text_path,
        allow_downloads=False,
        dataset_cache_dir="unused",
        dataset_split="test",
        seqlen=8,
        n_samples=3,
        window_seed=42,
    )
    first, first_meta = ab._load_wikitext_windows(args, Tokenizer())
    second, second_meta = ab._load_wikitext_windows(args, Tokenizer())
    assert first == second
    assert first_meta["starts"] == second_meta["starts"]
    assert first_meta["source"] == "raw_text"
    assert first_meta["text_sha256"] == ab._sha256(text_path)
    assert first_meta["corpus_text_sha256"] == ab.hashlib.sha256(
        text_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert len(first_meta["prompt_window_sha256"]) == 3
    assert len(set(first_meta["prompt_window_sha256"])) == 3
    assert all(len(window) == 8 for window in first)


def test_wikitext_windows_fail_when_distinct_sample_claim_is_impossible(tmp_path):
    text_path = tmp_path / "short.raw"
    text_path.write_text("abcdefghijklmnopq", encoding="utf-8")

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    args = SimpleNamespace(
        wikitext_text=text_path,
        allow_downloads=False,
        dataset_cache_dir="unused",
        dataset_split="test",
        seqlen=17,
        n_samples=2,
        window_seed=42,
    )
    with pytest.raises(RuntimeError, match="distinct start positions"):
        ab._load_wikitext_windows(args, Tokenizer())

    text_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    class RepeatingTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [7] * len(text)}

    args.seqlen = 8
    with pytest.raises(RuntimeError, match="not content-distinct"):
        ab._load_wikitext_windows(args, RepeatingTokenizer())


def test_datasets_source_records_fingerprint_and_exact_corpus_hash(
    tmp_path, monkeypatch
):
    class Dataset(list):
        _fingerprint = "wiki-fingerprint-123"

    dataset = Dataset([{"text": "abcdefghij"}, {"text": "klmnopqrst"}])
    fake_datasets = SimpleNamespace(
        DownloadConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        load_dataset=lambda *args, **kwargs: dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text)))}

    args = SimpleNamespace(
        wikitext_text=None,
        allow_downloads=False,
        dataset_cache_dir=tmp_path / "cache",
        dataset_split="test",
        seqlen=8,
        n_samples=2,
        window_seed=7,
    )
    _windows, meta = ab._load_wikitext_windows(args, Tokenizer())
    corpus = "abcdefghij\n\nklmnopqrst"
    assert meta["dataset_fingerprint"] == "wiki-fingerprint-123"
    assert meta["corpus_text_sha256"] == ab.hashlib.sha256(
        corpus.encode("utf-8")
    ).hexdigest()


def test_dispatch_probe_records_success_fallback_and_pid_without_torch():
    class X:
        def numel(self):
            return 64 * 256

    class Layer:
        _cb_N = 512
        _cb_K = 256

    class Method:
        is_fp4 = True
        prefix = "model.layers.0.mlp.down_proj"

        def _try_fused_fp4(self, layer, x, n, k, m):
            del x, n, k
            layer._cb_fp4_fused_tile_m = 128
            layer._cb_fp4_fused_tile_candidate_ctas = m
            layer._cb_fp4_fused_sm_count = 48
            return object()

        def _apply_inline(self, layer, x, bias=None):
            del bias
            return self._try_fused_fp4(
                layer, x, layer._cb_N, layer._cb_K, x.numel() // layer._cb_K
            )

    probe = ab.DenseDispatchProbe(Method, prefill_threshold=16, fused_mode="1")
    probe.install()
    method = Method()
    try:
        with probe.measurement("fused", "success") as success:
            assert method._apply_inline(Layer(), X()) is not None
        assert success["candidate_gate_opportunities"] == 1
        assert success["fused_attempts"] == 1
        assert success["fused_successes"] == 1
        assert success["fused_fallbacks"] == 0

        probe._original_try = lambda *args, **kwargs: None
        # Reinstall the wrapper around the fallback implementation.
        probe.restore()
        Method._try_fused_fp4 = lambda *args, **kwargs: None
        probe.install()
        with probe.measurement("fused", "fallback") as fallback:
            assert method._apply_inline(Layer(), X()) is None
        assert fallback["fused_attempts"] == 1
        assert fallback["fused_fallbacks"] == 1
        merged = ab.aggregate_dispatch((success, fallback))
        assert merged["fused_successes"] == 1
        assert merged["fused_fallbacks"] == 1
        assert merged["fused_success_fraction"] == pytest.approx(0.5)
        assert len(merged["pids"]) == 1
        assert merged["success_tile_m"] == {"128": 1}
        assert merged["success_tile_m_shapes"] == {
            "M64:N512:K256:tile128": 1
        }
        assert merged["success_tile_m_contracts"] == {
            "tile128:ctas64:sm48": 1
        }
        route_gate = ab.dense_tile_route_integrity_gate(merged)
        assert route_gate["pass"] is True
        assert route_gate["observed_routes"] == 1
    finally:
        probe.restore()


def test_dense_tile_route_integrity_gate_fails_closed():
    dispatch = ab.aggregate_dispatch((
        ab._empty_dispatch_record("missing", "fused"),
    ))
    dispatch["fused_successes"] = 1
    missing = ab.dense_tile_route_integrity_gate(dispatch)
    assert missing["pass"] is False
    assert missing["observed_routes"] == 0

    dispatch["success_tile_m"] = {"192": 1}
    dispatch["success_tile_m_shapes"] = {
        "M256:N512:K256:tile192": 1
    }
    dispatch["success_tile_m_contracts"] = {
        "tile192:ctas4:sm48": 1
    }
    invalid = ab.dense_tile_route_integrity_gate(dispatch)
    assert invalid["pass"] is False
    assert invalid["legal_tile_values"] is False


def test_dense_dispatch_probe_forwards_rowwise_family_without_losing_telemetry():
    observed = []

    class Layer:
        _cb_N = 512
        _cb_K = 256

    class X:
        def numel(self):
            return 64 * 256

    class Method:
        is_fp4 = True
        prefix = "model.layers.0.mlp.down_proj"

        def _try_fused_fp4(self, layer, x, n, k, m, *, rowwise=False):
            del layer, x, n, k, m
            observed.append(rowwise)
            return object()

        def _apply_inline(self, layer, x, bias=None):
            del bias
            return self._try_fused_fp4(
                layer, x, layer._cb_N, layer._cb_K,
                x.numel() // layer._cb_K, rowwise=True,
            )

    probe = ab.DenseDispatchProbe(
        Method, prefill_threshold=16, fused_mode="rowwise"
    )
    probe.install()
    try:
        with probe.measurement("fused", "rowwise") as record:
            assert Method()._apply_inline(Layer(), X()) is not None
        assert observed == [True]
        assert record["candidate_gate_opportunities"] == 1
        assert record["fused_attempts"] == 1
        assert record["fused_successes"] == 1
    finally:
        probe.restore()


def test_dense_dispatch_probe_forwards_static_lsq_without_losing_telemetry():
    observed = []

    class Layer:
        _cb_N = 512
        _cb_K = 256

    class X:
        def numel(self):
            return 64 * 256

    class Method:
        is_fp4 = True
        prefix = "model.layers.0.mlp.down_proj"

        def _try_fused_fp4(
            self, layer, x, n, k, m, *, rowwise=False, static_lsq=False,
        ):
            del layer, x, n, k, m
            observed.append((rowwise, static_lsq))
            return object()

        def _apply_inline(self, layer, x, bias=None):
            del bias
            return self._try_fused_fp4(
                layer, x, layer._cb_N, layer._cb_K,
                x.numel() // layer._cb_K, static_lsq=True,
            )

    probe = ab.DenseDispatchProbe(
        Method, prefill_threshold=16, fused_mode="static_lsq"
    )
    probe.install()
    try:
        with probe.measurement("fused", "static_lsq") as record:
            assert Method()._apply_inline(Layer(), X()) is not None
        assert observed == [(False, True)]
        assert record["candidate_gate_opportunities"] == 1
        assert record["fused_attempts"] == 1
        assert record["fused_successes"] == 1
    finally:
        probe.restore()


def test_moe_probe_proves_loop_and_grouped_fused_routes_without_torch():
    class X:
        shape = (128, 1024)

    class TopK:
        shape = (128, 8)

    class Layer:
        _cb_E = 256
        _cb_hidden = 1024
        _cb_inter = 512

    class Method:
        prefix = "model.layers.0.mlp.experts"

        def _apply_prefill_grouped_fused_fp4(
            self, layer, x, topk_weights, topk_ids, act, *, tile_m=128
        ):
            del layer, x, topk_weights, topk_ids, act, tile_m
            return object()

        def _apply_prefill_loop(self, layer, x, topk_weights, topk_ids, act):
            del layer, x, topk_weights, topk_ids, act
            return object()

    method = Method()
    probe = ab.MoEDispatchProbe(Method)
    probe.install()
    try:
        with probe.measurement("baseline", "loop") as baseline:
            assert method._apply_prefill_loop(Layer(), X(), None, TopK(), None)
        with probe.measurement("fused", "grouped") as fused:
            assert method._apply_prefill_grouped_fused_fp4(
                Layer(), X(), None, TopK(), None, tile_m=256
            )
        assert baseline["loop_calls"] == 1
        assert baseline["loop_successes"] == 1
        assert baseline["fused_attempts"] == 0
        assert fused["fused_attempts"] == 1
        assert fused["fused_successes"] == 1
        assert fused["fused_fallbacks"] == 0
        assert list(fused["success_shapes"]) == [
            "T128:E256:H1024:I512:topk8:tile256"
        ]
    finally:
        probe.restore()


def test_moe_dispatch_probe_forwards_rowwise_family_without_losing_telemetry():
    observed = []

    class X:
        shape = (128, 1024)

    class TopK:
        shape = (128, 8)

    class Layer:
        _cb_E = 256
        _cb_hidden = 1024
        _cb_inter = 512

    class Method:
        prefix = "model.layers.0.mlp.experts"

        def _apply_prefill_grouped_fused_fp4(
            self, layer, x, topk_weights, topk_ids, act, *, tile_m=128,
            rowwise=False,
        ):
            del layer, x, topk_weights, topk_ids, act
            observed.append((rowwise, tile_m))
            return object()

        def _apply_prefill_loop(self, layer, x, topk_weights, topk_ids, act):
            del layer, x, topk_weights, topk_ids, act
            return object()

    method = Method()
    probe = ab.MoEDispatchProbe(Method)
    probe.install()
    try:
        with probe.measurement("fused", "rowwise") as record:
            assert method._apply_prefill_grouped_fused_fp4(
                Layer(), X(), None, TopK(), None,
                tile_m=256, rowwise=True,
            )
        assert observed == [(True, 256)]
        assert record["fused_attempts"] == 1
        assert record["fused_successes"] == 1
    finally:
        probe.restore()


def test_teacher_kl_absolute_and_fused_regression_gates_can_fail_status():
    args = SimpleNamespace(
        max_mean_kl=None,
        max_mean_nll_regression=None,
        max_ppl_relative_regression=None,
        min_timing_speedup=None,
        max_teacher_fused_mean_kl=0.20,
        max_teacher_fused_kl_regression=0.03,
    )
    report = {
        "quality": {"delta": {}},
        "teacher_quality": {
            "baseline": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.10},
            },
            "fused": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.15},
            },
        },
    }
    gates = ab._configured_gates(args, report)
    assert gates["max_teacher_fused_mean_kl"]["pass"] is True
    regression = gates["max_teacher_fused_kl_regression"]
    assert regression["observed_fused_minus_baseline"] == pytest.approx(0.05)
    assert regression["pass"] is False
    assert all(
        gate["pass"]
        for name, gate in gates.items()
        if name != "max_teacher_fused_kl_regression"
    )
    assert not all(gate["pass"] for gate in gates.values())


def test_gate_finalization_distinguishes_measurement_screen_and_promotion():
    defaults = {
        "max_mean_kl": None,
        "max_mean_nll_regression": None,
        "max_ppl_relative_regression": None,
        "max_teacher_fused_mean_kl": None,
        "max_teacher_fused_kl_regression": None,
        "min_timing_speedup": None,
        "teacher_full_vocab_kl": False,
        "measurement_only": True,
    }
    core_gates = {"integrity": {"pass": True}}
    measurement = {"quality": {"delta": {}}}
    ab._finalize_gate_report(
        SimpleNamespace(**defaults), measurement, core_gates
    )
    assert measurement["measurement_valid"] is True
    assert measurement["configured_promotion_gates"] == {}
    assert measurement["configured_gates_pass"] is None
    assert measurement["promotion_contract"]["complete"] is False
    assert measurement["promotion_recommendation"] == (
        "measurement_only_no_promotion_thresholds_configured"
    )

    screening_args = {
        **defaults,
        "measurement_only": False,
        "max_mean_kl": 0.2,
    }
    screening = {
        "quality": {"delta": {"kl_baseline_to_fused": {"mean": 0.1}}}
    }
    ab._finalize_gate_report(
        SimpleNamespace(**screening_args), screening, core_gates
    )
    assert screening["configured_gates_pass"] is True
    assert screening["promotion_contract"]["complete"] is False
    assert screening["promotion_recommendation"] == (
        "screening_only_full_vocab_teacher_required"
    )

    full_vocab_without_teacher_limits = {
        "quality": {"delta": {"kl_baseline_to_fused": {"mean": 0.1}}},
        "teacher_quality": {
            arm: {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": value},
            }
            for arm, value in (("baseline", 0.10), ("fused", 0.12))
        },
    }
    ab._finalize_gate_report(
        SimpleNamespace(**{
            **screening_args,
            "teacher_full_vocab_kl": True,
        }),
        full_vocab_without_teacher_limits,
        core_gates,
    )
    contract = full_vocab_without_teacher_limits["promotion_contract"]
    assert full_vocab_without_teacher_limits["configured_gates_pass"] is True
    assert contract["complete"] is False
    assert contract["teacher_full_vocab_kl_observed"] is True
    assert contract["teacher_quality_thresholds_missing"] == [
        "max_teacher_fused_mean_kl",
        "max_teacher_fused_kl_regression",
    ]
    assert full_vocab_without_teacher_limits["promotion_recommendation"] == (
        "screening_only_teacher_quality_thresholds_required"
    )

    promotion = {
        "quality": {"delta": {"kl_baseline_to_fused": {"mean": 0.1}}},
        "settings": {
            "chunked_prefill_contract": {"promotion_compatible": True}
        },
        "teacher_quality": {
            "baseline": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.10},
            },
            "fused": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.12},
            },
        },
    }
    ab._finalize_gate_report(
        SimpleNamespace(**{
            **screening_args,
            "teacher_full_vocab_kl": True,
            "max_teacher_fused_mean_kl": 0.20,
            "max_teacher_fused_kl_regression": 0.03,
        }),
        promotion,
        core_gates,
    )
    assert promotion["configured_gates_pass"] is True
    assert promotion["promotion_contract"]["complete"] is True
    assert promotion["promotion_contract"]["teacher_quality_thresholds_present"] == [
        "max_teacher_fused_mean_kl",
        "max_teacher_fused_kl_regression",
    ]
    assert promotion["promotion_recommendation"] == (
        "candidate_only_requires_served_validation"
    )

    incompatible = {
        **promotion,
        "settings": {
            "chunked_prefill_contract": {"promotion_compatible": False}
        },
    }
    ab._finalize_gate_report(
        SimpleNamespace(**{
            **screening_args,
            "teacher_full_vocab_kl": True,
            "max_teacher_fused_mean_kl": 0.20,
            "max_teacher_fused_kl_regression": 0.03,
        }),
        incompatible,
        core_gates,
    )
    assert incompatible["promotion_contract"]["complete"] is False
    assert incompatible["promotion_contract"][
        "chunked_prefill_promotion_compatible"
    ] is False
    assert incompatible["promotion_recommendation"] == (
        "screening_only_chunked_prefill_contract_incompatible"
    )


@pytest.mark.parametrize(
    "configured_gate,missing_gate",
    [
        ("max_teacher_fused_mean_kl", "max_teacher_fused_kl_regression"),
        ("max_teacher_fused_kl_regression", "max_teacher_fused_mean_kl"),
    ],
)
def test_promotion_contract_requires_both_teacher_quality_thresholds(
    configured_gate,
    missing_gate,
):
    limits = {
        "max_mean_kl": None,
        "max_mean_nll_regression": None,
        "max_ppl_relative_regression": None,
        "max_teacher_fused_mean_kl": None,
        "max_teacher_fused_kl_regression": None,
        "min_timing_speedup": None,
        "teacher_full_vocab_kl": True,
        "measurement_only": False,
    }
    limits[configured_gate] = 0.20 if configured_gate.endswith("mean_kl") else 0.03
    report = {
        "quality": {"delta": {}},
        "teacher_quality": {
            "baseline": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.10},
            },
            "fused": {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": 0.12},
            },
        },
    }
    ab._finalize_gate_report(
        SimpleNamespace(**limits), report, {"integrity": {"pass": True}}
    )
    assert report["configured_gates_pass"] is True
    assert report["promotion_contract"]["complete"] is False
    assert report["promotion_contract"]["teacher_quality_thresholds_missing"] == [
        missing_gate
    ]


def test_timing_only_full_vocab_teacher_cannot_greenlight_bad_quality(
    tmp_path,
    monkeypatch,
):
    args = SimpleNamespace(
        max_mean_kl=None,
        max_mean_nll_regression=None,
        max_ppl_relative_regression=None,
        max_teacher_fused_mean_kl=None,
        max_teacher_fused_kl_regression=None,
        min_timing_speedup=1.0,
        teacher_full_vocab_kl=True,
        measurement_only=False,
        output=tmp_path / "timing-only.json",
    )
    report = {
        "quality": {"delta": {}},
        "teacher_quality": {
            arm: {
                "kl_mode": ab.KL_FULL_VOCAB,
                "kl_reference_to_candidate": {"mean": value},
            }
            for arm, value in (("baseline", 998.0), ("fused", 999.0))
        },
        "timing": {"baseline_over_fused_speedup": 1.1},
        "dispatch": {"baseline": {}, "fused": {}},
    }
    ab._finalize_gate_report(
        args, report, {"integrity": {"pass": True}}
    )
    assert report["configured_gates_pass"] is True
    assert report["promotion_contract"]["complete"] is False

    written = {}
    monkeypatch.setattr(ab, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(ab, "run", lambda parsed: dict(report))
    monkeypatch.setattr(
        ab, "_atomic_json", lambda path, payload: written.update(payload)
    )
    assert ab.main([]) == 2
    assert written["status"] == "screening_only"


def test_cli_refuses_nonprefill_window_and_speed_gate_without_timing(tmp_path):
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    model.mkdir()
    teacher.mkdir()
    screen_base = ["--model", str(model), "--output", str(tmp_path / "out.json")]
    base = [*screen_base, "--measurement-only"]
    args = ab.parse_args(base)
    assert args.min_fused_success_fraction == 1.0
    assert args.trust_remote_code is False
    assert args.measurement_only is True
    with pytest.raises(SystemExit):
        ab.parse_args(screen_base)
    with pytest.raises(SystemExit):
        ab.parse_args([*base, "--seqlen", "16"])
    with pytest.raises(SystemExit):
        ab.parse_args([*screen_base, "--min-timing-speedup", "1.1"])
    with pytest.raises(SystemExit):
        ab.parse_args([
            "--model", "org/model", "--output", str(tmp_path / "remote.json"),
            "--measurement-only",
        ])
    pinned = ab.parse_args([
        "--model", "org/model", "--output", str(tmp_path / "remote.json"),
        "--measurement-only", "--allow-downloads",
        "--revision", "0123456789abcdef",
        "--trust-remote-code", "--min-fused-success-fraction", "0.5",
    ])
    assert pinned.revision == "0123456789abcdef"
    assert pinned.trust_remote_code is True
    assert pinned.min_fused_success_fraction == 0.5
    local_teacher = ab.parse_args([*base, "--teacher-model", str(teacher)])
    assert local_teacher.teacher_model == str(teacher)
    assert local_teacher.teacher_dtype == "bfloat16"
    assert local_teacher.teacher_full_vocab_kl is False
    full_teacher = ab.parse_args([
        *screen_base,
        "--teacher-model", str(teacher),
        "--teacher-full-vocab-kl",
        "--max-teacher-fused-mean-kl", "0.2",
        "--max-teacher-fused-kl-regression", "0.01",
    ])
    assert full_teacher.teacher_full_vocab_kl is True
    assert full_teacher.max_teacher_fused_kl_regression == pytest.approx(0.01)
    with pytest.raises(SystemExit):
        ab.parse_args([*screen_base, "--teacher-full-vocab-kl"])
    with pytest.raises(SystemExit):
        ab.parse_args([
            *screen_base, "--max-teacher-fused-kl-regression", "0.01"
        ])
    with pytest.raises(SystemExit):
        ab.parse_args([
            *screen_base, "--teacher-model", "org/teacher",
            "--max-mean-kl", "0.1",
        ])
    coarse_screen = ab.parse_args([
        *screen_base, "--teacher-model", str(teacher),
        "--max-teacher-fused-kl-regression", "0.01",
    ])
    assert coarse_screen.teacher_full_vocab_kl is False
    assert coarse_screen.measurement_only is False
    with pytest.raises(SystemExit):
        ab.parse_args([*base, "--max-mean-kl", "0.1"])


@pytest.mark.parametrize(
    "measurement_only,configured_pass,contract_complete,expected_status,expected_rc",
    [
        (True, None, False, "measurement_only", 0),
        (False, True, False, "screening_only", 2),
        (False, True, True, "ok", 0),
        (False, False, True, "gate_failed", 2),
    ],
)
def test_main_status_separates_measurement_screening_and_promotion(
    tmp_path,
    monkeypatch,
    measurement_only,
    configured_pass,
    contract_complete,
    expected_status,
    expected_rc,
):
    output = tmp_path / f"{expected_status}.json"
    args = SimpleNamespace(output=output)
    report = {
        "measurement_valid": True,
        "configured_gates_pass": configured_pass,
        "measurement_only": measurement_only,
        "promotion_contract": {"complete": contract_complete},
        "promotion_recommendation": "fixture",
        "quality": {"delta": {}},
        "dispatch": {"baseline": {}, "fused": {}},
    }
    written = {}
    monkeypatch.setattr(ab, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(ab, "run", lambda parsed: dict(report))
    monkeypatch.setattr(
        ab, "_atomic_json", lambda path, payload: written.update(payload)
    )
    assert ab.main([]) == expected_rc
    assert written["status"] == expected_status
