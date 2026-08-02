"""Shared bootstrap and teacher plumbing for fused-NVFP4 validation tools.

The public validation entry points intentionally keep their arm scheduling,
dispatch gates, schemas, and CLIs separate.  This module owns only the costly
runtime-independent plumbing that must be identical across those entry points:
loading and attesting the extension/runtime, resolving candidate and teacher
artifacts, constructing one candidate engine, and scoring an optional teacher.

``helpers`` is the established v5 module.  Passing it explicitly avoids a
module cycle while retaining its fail-closed provenance and scoring routines.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MEASURED_PHASE_ORDER = ("timing", "quality")

# --- ROADMAP K0.2 readiness verdict ---------------------------------------
# A fused-MoE A/B is only evidence if the artifact's producer attested a
# stage-specific ``input_global_scale`` for BOTH routed stages.  Against an
# unattested artifact every fused attempt correctly fails closed to the
# baseline loop, so identical outputs and ~1.00x timing are *fallback
# telemetry, not evidence* (docs/audits/fused_nvfp4_enablement_2026-07-31.md).
# These verdicts let the A/B entry points say that in the report instead of
# silently publishing a zero-difference comparison.
K02_ATTESTED = "attested_and_verified"
K02_MISSING_STAGES = "missing_stages"
K02_DIGEST_MISMATCH = "digest_mismatch"
K02_NOT_ATTESTED = "not_attested"
K02_MALFORMED = "malformed_stage_attestation"
K02_CONTRACT_ABSENT = "contract_absent"
K02_ARTIFACT_UNREADABLE = "artifact_unreadable"
K02_VERDICTS = (
    K02_ATTESTED,
    K02_MISSING_STAGES,
    K02_DIGEST_MISMATCH,
    K02_NOT_ATTESTED,
    K02_MALFORMED,
    K02_CONTRACT_ABSENT,
    K02_ARTIFACT_UNREADABLE,
)
EVIDENCE_STAGE_ATTESTED = "stage_attested_fused_moe_evidence_eligible"
EVIDENCE_FALLBACK_TELEMETRY = "fallback_telemetry_not_evidence"
EVIDENCE_DENSE_SCOPE = "dense_scope_stage_attestation_not_applicable"


def measurement_phase_order(timing_repeats: int) -> tuple[str, ...]:
    """Keep allocation-heavy full-vocabulary scoring outside timing setup."""

    if timing_repeats < 0:
        raise ValueError("timing_repeats must be nonnegative")
    return MEASURED_PHASE_ORDER if timing_repeats else ("quality",)


def read_artifact_activation_contract(
    model_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Read one local artifact's activation contract and serialized scalars.

    Deliberately independent of the serving config reader: this runs before
    (and without) an engine, so a K0.2 verdict costs no GPU and no vLLM.
    """

    from safetensors import safe_open

    from gridbook import nvfp4_activation_contract as contract

    root = Path(model_dir)
    raw: Any = None
    quant_config_path = root / "quant_config.json"
    if quant_config_path.is_file():
        raw = json.loads(quant_config_path.read_text(encoding="utf-8"))
    else:
        config_path = root / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(config, Mapping):
                raw = config.get("quantization_config")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"artifact {root} declares no readable quantization config"
        )
    record = contract.parse_contract(raw)
    scales: dict[str, float] = {}
    suffix = "." + contract.TENSOR_SUFFIX
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"artifact {root} has no safetensors shard")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if not name.endswith(suffix):
                    continue
                target = name[: -len(suffix)]
                if target in scales:
                    raise RuntimeError(
                        f"artifact {root} serializes {name} more than once"
                    )
                scales[target] = contract.scale_f32(
                    reader.get_tensor(name), target=target
                )
    return (dict(record) if record is not None else None), scales


def k02_readiness_verdict(
    model_dir: Any, *, mode: str
) -> dict[str, Any]:
    """Return the machine-readable K0.2 stage-attestation verdict.

    ``pass`` is the precondition the A/B entry points consume: for a routed-MoE
    run it is True only when every packed FusedMoE module attests both stages
    AND every stage digest matches the artifact's serialized scalar.  Dense
    runs are out of scope by construction — a dense-only artifact carries no
    stage section and remains valid.
    """

    from gridbook import nvfp4_activation_contract as contract

    routed = mode != "dense"
    path = Path(model_dir).expanduser() if model_dir is not None else None
    result: dict[str, Any] = {
        "schema": contract.CONTRACT_SCHEMA_V2,
        "roadmap_item": "K0.2",
        "execution_mode": mode,
        "routed_moe_scope": routed,
        "artifact": str(path) if path is not None else None,
        "contract_schema": None,
        "modules": [],
        "failing_module": None,
        "failing_stage": None,
    }
    if path is None or not path.is_dir():
        result.update({
            "verdict": K02_ARTIFACT_UNREADABLE,
            "detail": (
                "a K0.2 verdict requires a local artifact directory; a Hub "
                "candidate cannot be stage-verified before download"
            ),
        })
    else:
        try:
            record, scales = read_artifact_activation_contract(path)
        except Exception as exc:  # unreadable/malformed artifact
            record = None
            scales = {}
            result.update({
                "verdict": K02_ARTIFACT_UNREADABLE,
                "detail": f"{type(exc).__name__}: {exc}",
            })
        else:
            result["contract_schema"] = (
                record.get("schema") if record is not None else None
            )
            if record is None:
                result.update({
                    "verdict": K02_CONTRACT_ABSENT,
                    "detail": (
                        "the artifact declares no "
                        f"execution_contracts.{contract.CONTRACT_KEY} record, "
                        "so it carries no lawful static activation scale for "
                        "either routed stage"
                    ),
                })
            else:
                verified = contract.verify_routed_moe_stages(record, scales)
                result.update({
                    "verdict": verified["verdict"],
                    "detail": verified["detail"],
                    "modules": verified["modules"],
                    "failing_module": verified["failing_module"],
                    "failing_stage": verified["failing_stage"],
                })
    attested = result["verdict"] == K02_ATTESTED
    result["pass"] = bool(attested) if routed else True
    result["evidence_class"] = (
        EVIDENCE_DENSE_SCOPE if not routed
        else EVIDENCE_STAGE_ATTESTED if attested
        else EVIDENCE_FALLBACK_TELEMETRY
    )
    result["fused_moe_ab_is_evidence"] = bool(attested) if routed else None
    return result


def quiesce_before_timing(torch: Any) -> int:
    """Force host garbage collection and prior CUDA work outside arm timers."""

    collected = gc.collect()
    torch.cuda.synchronize()
    return int(collected)


@dataclass
class ValidationBootstrap:
    torch: Any
    gridbook: Any
    gridbook_config: Any
    cuda_ext: Any
    linear: Any
    moe: Any
    vllm: Any
    llm_class: Any
    sampling_params_class: Any
    runtime: dict[str, Any]
    extension: dict[str, Any]
    candidate_path: Path
    candidate_config: Any
    candidate_vocab_size: int
    candidate_artifact_provenance: dict[str, Any] | None
    candidate_load_revision: str | None
    tokenizer: Any
    teacher_config: Any | None
    teacher_identity: dict[str, Any] | None
    teacher_artifact_provenance: dict[str, Any] | None
    teacher_load_revision: str | None
    teacher_path: Path | None
    prompts: list[list[int]]
    dataset: dict[str, Any]
    quality_kl_mode: str
    quality_logprobs: int


@dataclass
class CandidateEngine:
    llm: Any
    quality_sampling: Any
    timing_sampling: Any
    model_load_seconds: float
    chunked_prefill_contract: dict[str, Any] | None


def prepare_validation(
    args: Any,
    *,
    harness_path: Path,
    helpers: Any,
    extension_none_message: str,
) -> ValidationBootstrap:
    """Load and attest shared runtime, artifact, tokenizer, and dataset state."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("fused NVFP4 validation requires a CUDA GPU")

    import gridbook

    gridbook.register()
    from gridbook import cb_fill_guard, config as gridbook_config
    from gridbook import cuda_ext, linear, moe, moe_toplevel_loader

    extension_started = time.monotonic()
    fused_extension = cuda_ext.get_fused_fp4_ext()
    extension_load_s = time.monotonic() - extension_started
    if fused_extension is None:
        raise RuntimeError(extension_none_message)
    required_symbol = (
        "cb_fused_fp4_prefill_mm_scaled"
        if args.mode == "dense"
        else "cb_fused_fp4_moe_grouped"
    )
    if not hasattr(fused_extension, required_symbol):
        raise RuntimeError(
            f"fused FP4 extension lacks required {args.mode} symbol "
            f"{required_symbol!r}"
        )

    import vllm
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    runtime = helpers._runtime_provenance(
        torch,
        vllm,
        gridbook,
        gridbook_config,
        linear,
        moe,
        moe_toplevel_loader,
        cb_fill_guard,
        cuda_ext,
        harness_path,
    )
    helpers_path_raw = getattr(helpers, "__file__", None)
    if helpers_path_raw is None:
        raise RuntimeError("validation helper module has no filesystem __file__")
    helpers_path = Path(helpers_path_raw).resolve()
    if helpers_path != harness_path.resolve():
        runtime["harness"]["shared_helpers"]["v5_validation_api"] = {
            "path": str(helpers_path),
            **helpers._required_file_record(helpers_path),
        }
    extension_path_raw = getattr(fused_extension, "__file__", None)
    if extension_path_raw is None:
        raise RuntimeError("loaded fused FP4 extension has no filesystem __file__")
    extension_path = Path(extension_path_raw).resolve()
    extension_file = helpers._required_file_record(extension_path)
    extension = {
        "preloaded_before_model": True,
        "load_seconds": extension_load_s,
        "module": getattr(fused_extension, "__name__", None),
        "path": str(extension_path),
        "bytes": extension_file["bytes"],
        "sha256": extension_file["sha256"],
        "required_symbol": required_symbol,
        "required_symbol_present": True,
    }

    candidate_path = Path(args.model).expanduser()
    candidate_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        local_files_only=not args.allow_downloads,
    )
    candidate_vocab_size = int(getattr(candidate_config, "vocab_size", 0))
    if candidate_vocab_size <= 0:
        raise RuntimeError(
            f"candidate config has invalid vocab_size={candidate_vocab_size}"
        )
    if candidate_path.is_dir():
        candidate_artifact_provenance = helpers._local_model_provenance(
            candidate_path, role="candidate"
        )
        candidate_load_revision = args.revision
    else:
        candidate_artifact_provenance = helpers._hub_model_provenance(
            candidate_config,
            role="candidate",
            model_id=args.model,
            requested_revision=args.revision,
        )
        candidate_load_revision = candidate_artifact_provenance[
            "resolved_commit_hash"
        ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=candidate_load_revision,
        local_files_only=not args.allow_downloads,
        fix_mistral_regex=True,
    )

    teacher_config = None
    teacher_identity = None
    teacher_artifact_provenance = None
    teacher_load_revision = args.teacher_revision
    teacher_path = (
        Path(args.teacher_model).expanduser() if args.teacher_model else None
    )
    if args.teacher_model is not None:
        teacher_preflight_started = time.monotonic()
        teacher_config = AutoConfig.from_pretrained(
            args.teacher_model,
            trust_remote_code=args.trust_remote_code,
            revision=args.teacher_revision,
            local_files_only=not args.allow_downloads,
        )
        config_identity = helpers._assert_config_identity(
            candidate_config, teacher_config
        )
        assert teacher_path is not None
        if teacher_path.is_dir():
            teacher_artifact_provenance = helpers._local_model_provenance(
                teacher_path, role="teacher"
            )
        else:
            teacher_artifact_provenance = helpers._hub_model_provenance(
                teacher_config,
                role="teacher",
                model_id=args.teacher_model,
                requested_revision=args.teacher_revision,
            )
            teacher_load_revision = teacher_artifact_provenance[
                "resolved_commit_hash"
            ]
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model,
            trust_remote_code=args.trust_remote_code,
            revision=teacher_load_revision,
            local_files_only=not args.allow_downloads,
            fix_mistral_regex=True,
        )
        tokenizer_identity = helpers._assert_tokenizer_identity(
            tokenizer, teacher_tokenizer
        )
        tokenizer_files = helpers._assert_local_tokenizer_files_match(
            candidate_path, teacher_path
        )
        teacher_identity = {
            "config": config_identity,
            "tokenizer": tokenizer_identity,
            "local_tokenizer_files": tokenizer_files,
            "preflight_seconds": time.monotonic() - teacher_preflight_started,
        }
        del teacher_tokenizer

    prompts, dataset = helpers._load_wikitext_windows(args, tokenizer)
    quality_kl_mode = (
        helpers.KL_FULL_VOCAB
        if args.teacher_full_vocab_kl
        else helpers.KL_COARSE_TOPK
    )
    quality_logprobs = -1 if args.teacher_full_vocab_kl else args.top_k
    return ValidationBootstrap(
        torch=torch,
        gridbook=gridbook,
        gridbook_config=gridbook_config,
        cuda_ext=cuda_ext,
        linear=linear,
        moe=moe,
        vllm=vllm,
        llm_class=LLM,
        sampling_params_class=SamplingParams,
        runtime=runtime,
        extension=extension,
        candidate_path=candidate_path,
        candidate_config=candidate_config,
        candidate_vocab_size=candidate_vocab_size,
        candidate_artifact_provenance=candidate_artifact_provenance,
        candidate_load_revision=candidate_load_revision,
        tokenizer=tokenizer,
        teacher_config=teacher_config,
        teacher_identity=teacher_identity,
        teacher_artifact_provenance=teacher_artifact_provenance,
        teacher_load_revision=teacher_load_revision,
        teacher_path=teacher_path,
        prompts=prompts,
        dataset=dataset,
        quality_kl_mode=quality_kl_mode,
        quality_logprobs=quality_logprobs,
    )


def load_candidate_engine(
    bootstrap: ValidationBootstrap,
    args: Any,
    *,
    probe: Any,
    attest_chunked_prefill: bool = False,
) -> CandidateEngine:
    """Construct exactly one candidate engine while its dispatch probe is live."""

    load_started = time.monotonic()
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "revision": bootstrap.candidate_load_revision,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.seqlen + 16,
        "max_num_seqs": 1,
        "max_logprobs": bootstrap.quality_logprobs,
        "enforce_eager": True,
        "disable_log_stats": True,
        "enable_prefix_caching": False,
        "seed": args.seed,
    }
    chunked_prefill_request = args.enable_chunked_prefill
    if chunked_prefill_request is not None:
        llm_kwargs["enable_chunked_prefill"] = bool(
            chunked_prefill_request
        )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    try:
        llm = bootstrap.llm_class(**llm_kwargs)
    except Exception:
        probe.restore()
        raise
    model_load_s = time.monotonic() - load_started
    quality_sampling = bootstrap.sampling_params_class(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=bootstrap.quality_logprobs,
        detokenize=False,
    )
    timing_sampling = bootstrap.sampling_params_class(
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
    )
    try:
        chunked_prefill_contract = (
            attest_chunked_prefill_contract(llm, chunked_prefill_request)
            if attest_chunked_prefill
            else None
        )
    except Exception:
        probe.restore()
        raise
    return CandidateEngine(
        llm=llm,
        quality_sampling=quality_sampling,
        timing_sampling=timing_sampling,
        model_load_seconds=model_load_s,
        chunked_prefill_contract=chunked_prefill_contract,
    )


def attest_chunked_prefill_contract(
    llm: Any, requested: bool | None
) -> dict[str, Any]:
    """Read vLLM's resolved scheduler/model contract without log scraping."""

    if requested is not None and not isinstance(requested, bool):
        raise RuntimeError(
            "chunked prefill request must be auto/True/False, got "
            f"{requested!r}"
        )
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    resolved = getattr(scheduler_config, "enable_chunked_prefill", None)
    official_default = getattr(
        model_config, "is_chunked_prefill_supported", None
    )
    runner_type = getattr(model_config, "runner_type", None)
    if not isinstance(resolved, bool):
        raise RuntimeError(
            "vLLM scheduler config did not expose boolean "
            "enable_chunked_prefill"
        )
    if not isinstance(official_default, bool):
        raise RuntimeError(
            "vLLM model config did not expose boolean "
            "is_chunked_prefill_supported"
        )
    if not isinstance(runner_type, str) or not runner_type:
        raise RuntimeError("vLLM model config did not expose runner_type")

    explicit = requested is not None
    requested_matches_resolved = (
        None if requested is None else requested == resolved
    )
    conflicts_with_official_contract = bool(
        explicit and requested != official_default
    )
    would_trigger_vllm_warning = bool(
        explicit
        and (
            (
                runner_type == "generate"
                and requested is False
                and official_default is True
            )
            or (
                runner_type == "pooling"
                and requested is True
                and official_default is False
            )
        )
    )
    promotion_compatible = bool(
        not conflicts_with_official_contract
        and requested_matches_resolved is not False
    )
    return {
        "requested": (
            "auto" if requested is None
            else "enable" if requested
            else "disable"
        ),
        "engine_kwarg_omitted": requested is None,
        "resolved_enabled": resolved,
        "model_official_default_enabled": official_default,
        "model_runner_type": runner_type,
        "requested_matches_resolved": requested_matches_resolved,
        "explicit_override_conflicts_with_official_contract": (
            conflicts_with_official_contract
        ),
        "would_trigger_vllm_warning": would_trigger_vllm_warning,
        "promotion_compatible": promotion_compatible,
        "attestation_source": (
            "LLM.llm_engine.vllm_config.{scheduler_config."
            "enable_chunked_prefill,model_config."
            "is_chunked_prefill_supported,model_config.runner_type}"
        ),
    }


def chunked_prefill_integrity_gate(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the one promotion decision to an attested resolved contract.

    Both validation schemas use this exact gate so a warning/conflicting vLLM
    override cannot be informational in one report and fatal in the other.
    """

    return {
        **dict(contract),
        "pass": contract.get("promotion_compatible") is True,
    }


def score_teacher(
    bootstrap: ValidationBootstrap,
    args: Any,
    *,
    arm_scores: Mapping[str, Sequence[Any]],
    arms: Sequence[str],
    helpers: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load one attested BF16 teacher and compare it with every named arm."""

    if args.teacher_model is None:
        return None, None

    from transformers import AutoModelForCausalLM

    if bootstrap.teacher_config is None or bootstrap.teacher_identity is None:
        raise RuntimeError("teacher identity preflight did not complete")
    teacher_started = time.monotonic()
    teacher_dtype = getattr(bootstrap.torch, args.teacher_dtype, None)
    if teacher_dtype is None:
        raise RuntimeError(
            f"unsupported teacher torch dtype {args.teacher_dtype!r}"
        )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        config=bootstrap.teacher_config,
        revision=bootstrap.teacher_load_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_downloads,
        dtype=teacher_dtype,
    ).eval().to(bootstrap.torch.device("cuda"))
    dtype_attestation = helpers._attest_teacher_model(
        teacher, bootstrap.torch
    )
    architectures = list(
        getattr(teacher.config, "architectures", None) or []
    )
    if architectures and type(teacher).__name__ not in architectures:
        raise RuntimeError(
            f"loaded teacher class {type(teacher).__name__!r} is not the "
            f"attested config architecture {architectures}"
        )
    vocab_size = int(getattr(teacher.config, "vocab_size", 0))
    if vocab_size != bootstrap.candidate_vocab_size:
        raise RuntimeError(
            f"teacher vocab_size={vocab_size} differs from candidate "
            f"vocab_size={bootstrap.candidate_vocab_size}"
        )
    max_prompt_token = max(max(prompt) for prompt in bootstrap.prompts)
    if vocab_size <= max_prompt_token:
        raise RuntimeError(
            f"teacher vocab_size={vocab_size} cannot score token "
            f"id {max_prompt_token} from the candidate tokenizer"
        )
    teacher_scores = [
        helpers.score_teacher_prompt(
            teacher,
            prompt,
            args.top_k,
            bootstrap.torch,
            full_vocab=args.teacher_full_vocab_kl,
            expected_vocab_size=bootstrap.candidate_vocab_size,
        )
        for prompt in bootstrap.prompts
    ]
    teacher_quality = {
        arm: helpers._pairwise_score_summary(
            teacher_scores,
            arm_scores[arm],
            reference_name="teacher",
            candidate_name=arm,
            kl_mode=bootstrap.quality_kl_mode,
        )
        for arm in arms
    }
    teacher_path = bootstrap.teacher_path
    assert teacher_path is not None
    teacher_record = {
        "model": args.teacher_model,
        "model_resolved": (
            str(teacher_path.resolve()) if teacher_path.is_dir() else None
        ),
        "revision": args.teacher_revision,
        "resolved_load_revision": (
            bootstrap.teacher_load_revision
            if not teacher_path.is_dir()
            else None
        ),
        "load_and_score_seconds": time.monotonic() - teacher_started,
        "requested_dtype": args.teacher_dtype,
        "actual_dtype_attestation": dtype_attestation,
        "vocab_size": vocab_size,
        "model_class": f"{type(teacher).__module__}.{type(teacher).__qualname__}",
        "transformers_version": importlib.metadata.version("transformers"),
        "identity": bootstrap.teacher_identity,
        "artifact_provenance": bootstrap.teacher_artifact_provenance,
        "comparison_contract": {
            "reference_backend": "Transformers",
            "candidate_backend": "vLLM",
            "cross_runtime": True,
            "target_nll": "exact full-vocabulary normalization",
            "kl_mode": bootstrap.quality_kl_mode,
            "kl_convention": helpers._kl_convention(
                bootstrap.quality_kl_mode
            ),
        },
        "role": (
            "unquantized BF16-parameter Transformers reference; "
            "non-parameter buffers are dtype-attested separately; never "
            "used for timing"
        ),
    }
    del teacher, teacher_scores
    bootstrap.torch.cuda.empty_cache()
    return teacher_quality, teacher_record


def shared_report_settings(
    bootstrap: ValidationBootstrap,
    args: Any,
    *,
    arm_settings: Mapping[str, Any],
    inherited_prefill: str | None,
    prefill_threshold: int,
    measurement_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build schema-stable settings shared by both arm orchestrators."""

    candidate_path = bootstrap.candidate_path
    settings: dict[str, Any] = {
        "model": args.model,
        "model_resolved": (
            str(candidate_path.resolve()) if candidate_path.is_dir() else None
        ),
        "revision": args.revision,
        "resolved_load_revision": (
            bootstrap.candidate_load_revision
            if not candidate_path.is_dir()
            else None
        ),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_dtype": args.teacher_dtype,
        "teacher_full_vocab_kl": bool(args.teacher_full_vocab_kl),
        "trust_remote_code": bool(args.trust_remote_code),
        "downloads_allowed": bool(args.allow_downloads),
        "quantization": args.quantization,
        "dtype": args.dtype,
        "mode": args.mode,
        **dict(arm_settings),
        "prefill_env_inherited_then_removed": (
            inherited_prefill if args.mode != "dense" else None
        ),
        "prefill_env_during_measurement": os.environ.get(
            "PRISMAQUANT_CB_PREFILL"
        ),
        "prefill_m_threshold": int(prefill_threshold),
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "v1_multiprocessing": False,
        "prefix_caching": False,
        "chunked_prefill": bool(args.enable_chunked_prefill),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "top_k": args.top_k,
        "quality_prompt_logprobs_request": bootstrap.quality_logprobs,
        "quality_kl_mode": bootstrap.quality_kl_mode,
        "candidate_vocab_size": bootstrap.candidate_vocab_size,
        **dict(measurement_settings),
        "seed": args.seed,
    }
    return settings


def add_shared_cli_arguments(
    parser: Any,
    *,
    helpers: Any,
    n_samples_default: int,
    chunked_prefill_tristate: bool = False,
) -> None:
    """Add byte-for-byte-compatible common v5/v6 CLI arguments."""

    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--teacher-model",
        help=(
            "optional local path or revision-pinned Hub ID for an unquantized "
            "BF16 Transformers teacher"
        ),
    )
    parser.add_argument(
        "--teacher-revision",
        help="required explicit revision for a non-local --teacher-model",
    )
    parser.add_argument(
        "--teacher-dtype",
        choices=("bfloat16",),
        default="bfloat16",
        help="teacher parameter dtype; fixed to BF16 and attested after loading",
    )
    parser.add_argument(
        "--teacher-full-vocab-kl",
        action="store_true",
        help=(
            "opt into exact full-vocabulary KL by requesting every vLLM "
            "prompt logprob and enforcing exact vocabulary cardinality; this "
            "is memory-heavy for 128k-vocabulary models"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-cache-dir", default="/hfcache/datasets")
    parser.add_argument(
        "--wikitext-text",
        type=Path,
        help="cached WikiText-2 raw split; avoids a `datasets` dependency",
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="explicitly allow model/tokenizer repository Python code",
    )
    parser.add_argument(
        "--revision",
        help="required explicit revision for a non-local --model",
    )
    parser.add_argument(
        "--n-samples", type=helpers._positive_int, default=n_samples_default
    )
    parser.add_argument("--seqlen", type=helpers._positive_int, default=128)
    parser.add_argument("--window-seed", type=int, default=42)
    parser.add_argument("--top-k", type=helpers._positive_int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=helpers._unit_interval,
        default=0.80,
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=helpers._positive_int
    )
    if chunked_prefill_tristate:
        chunked = parser.add_mutually_exclusive_group()
        chunked.add_argument(
            "--enable-chunked-prefill",
            dest="enable_chunked_prefill",
            action="store_const",
            const=True,
            default=None,
            help="explicitly require chunked prefill",
        )
        chunked.add_argument(
            "--disable-chunked-prefill",
            dest="enable_chunked_prefill",
            action="store_const",
            const=False,
            help=(
                "explicitly disable chunked prefill; promotion fails if this "
                "conflicts with the model's official vLLM contract"
            ),
        )
    else:
        parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument(
        "--mode", choices=("dense", "moe128", "moe256"), default="dense"
    )


def add_shared_measurement_cli_arguments(parser: Any, *, helpers: Any) -> None:
    parser.add_argument(
        "--timing-repeats", type=helpers._nonnegative_int, default=0
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--measurement-only",
        action="store_true",
        help=(
            "explicitly collect evidence without promotion thresholds; the "
            "report status is measurement_only and configured_gates_pass is null"
        ),
    )


def validate_shared_cli_args(
    parser: Any,
    args: Any,
    *,
    teacher_gate_values: Sequence[float | None],
) -> None:
    """Apply identical model/teacher/timing CLI safety checks."""

    if args.seqlen <= 16:
        parser.error("--seqlen must exceed the fused prefill threshold (16)")
    if args.min_timing_speedup is not None and args.timing_repeats == 0:
        parser.error("--min-timing-speedup requires --timing-repeats > 0")
    candidate_is_local = Path(args.model).expanduser().is_dir()
    if not candidate_is_local and not args.revision:
        parser.error("a non-local --model requires an explicit --revision")
    teacher_is_local = bool(
        args.teacher_model
        and Path(args.teacher_model).expanduser().is_dir()
    )
    if args.teacher_model and not teacher_is_local and not args.teacher_revision:
        parser.error(
            "a non-local --teacher-model requires an explicit --teacher-revision"
        )
    if args.teacher_full_vocab_kl and not args.teacher_model:
        parser.error("--teacher-full-vocab-kl requires --teacher-model")
    if (
        any(limit is not None for limit in teacher_gate_values)
        and not args.teacher_model
    ):
        parser.error("teacher-relative gates require --teacher-model")
