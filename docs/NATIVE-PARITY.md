# Native-parity serving protocol

This is the reproducible performance gate for work intended to bring Gridbook
to native vLLM parity.  It compares the same model family, exact whole-artifact
byte budget, serving image, and controlled request distribution while changing
only the declared quantization/backend contract.  It is a protocol, not a
performance claim.

The runner is `gridbook-bench-serve` (or
`python -m gridbook.bench_serve` from a checkout).  It delegates every request
to the streaming client in **`vllm bench serve`**.  Consequently:

- TTFT is measured from request issue to the first streamed token;
- TPOT is `(E2EL - TTFT) / (output tokens - 1)` for multi-token responses;
- ITL is measured between streamed token arrivals; and
- E2EL is measured from request issue through the final stream event.

Do not replace TPOT with `whole request wall time / output tokens`.  That folds
prefill into decode and can make a faster prefill path look like a faster decode
kernel.  The report retains vLLM's raw detailed results as well as block-level
summaries.

The command surface in this protocol is parser-validated against vLLM
`0.23.1rc1.dev764+g54b16d8a9.d20260703`.  The report records the client version;
an older client missing a required streaming, warmup, detailed-result, or
sampling flag fails rather than silently changing the protocol.

## What must match

Before running either arm, record this checklist.  A row that differs is a new
experiment, not another sample of the same experiment.

| Dimension | Required match |
|---|---|
| Base model | Same architecture, tokenizer, revision, and non-quantized tensors |
| Byte budget | Exact whole served artifact is at or below the predeclared byte budget `B` |
| Software | Same immutable vLLM image digest, CUDA/driver, Gridbook commit, and package version |
| Server | Same model length, KV dtype, eager/graph mode, memory utilization, batching and scheduler flags |
| Execution | Explicit format/rung, serialized layout, scale coding, W/A contract, concrete backend, TP size, and fallback state |
| Features | Prefix caching off; speculative decode, LoRA, request logging, and observability identically configured or off |
| Hardware | Same GPU, power/clock policy, thermals, and no competing workload |
| Client | Same input/output lengths, prompts, dataset seed, arrival rate, concurrency, warmups, and blocks |

Use a native quantized artifact made from the same base model.  The formal size
gate is integer arithmetic: **`whole_served_artifact_bytes <= B`**.  “Whole
served artifact” means every model shard, config, tokenizer file, codebook,
scale/index sidecar, and other file actually shipped for serving.  Download,
JIT/compiler, engine, and KV caches are excluded.  Target bpp, quantizable-body
bytes, and resident VRAM remain useful diagnostics, but none substitutes for
the whole-artifact gate.  BF16 is a useful ceiling, not the native denominator;
a 27B Gridbook result cannot be compared with a differently sized native model.

The runner requires opaque `--image-id` and `--model-id` values and records them
verbatim.  Prefer an image digest and a model repository plus immutable commit,
not mutable tags such as `latest` or `main`.  `--server-arg` records the exact
server command line; it is metadata only and never starts or changes a server.
The runner and server may be different containers or hosts, so their
environments are never conflated.  `PRISMAQUANT_*` values visible to the
benchmark process are labelled **runner environment**; add other runner values
with `--runner-env`.  Record the actual separately managed server environment
explicitly with repeatable `--server-env NAME=VALUE`.  Credentials, URL user
info, sensitive URL query values, authorization headers, and secret-like
arguments/environment keys are redacted from metadata and recorded commands.

Format labels are not execution evidence.  Every report requires the
format/rung, serialized weight layout, scale coding, full weight/activation
contract (`W4A4`, `W8A8`, and so on), concrete kernel/backend, tensor-parallel
size, fallback state, exact server vLLM/runtime build, GPU identifier, driver,
and CUDA runtime.  Client and server runtime identifiers are separate required
fields because the benchmark may run outside the serving container.  Attach
server log lines proving the backend and fallback state; `--server-arg` and a
filename containing “native” do not prove dispatch.

One important delegated-NVFP4 trap is **Marlin contract drift**.  An auto/native
MoE path may select Marlin while dropping activation scales, making the executed
operator W4A16 rather than W4A4.  Such a run cannot be labelled W4A4 even if the
serialized weights are NVFP4.  Record the executed W/A contract and concrete
backend, and prove both from server logs.  A silent fallback is a different arm,
not a slower sample of the requested arm.

## Approximate 0.6B smoke matrix

Use the 0.6B pair for rapid iteration only.  The small model exposes launch and
dispatch overhead quickly, but its currently available arms are not a formal
native-parity pair:

| 0.6B arm | Whole shipped bytes | Executed contract |
|---|---:|---|
| Native | 870,290,032 B | W4A4 delegated NVFP4 (only if logs prove the non-Marlin W4A4 path) |
| Gridbook CB | 871,628,664 B | W8A8 FP8_CB_K36 |

The CB artifact is **+1,338,632 B / +0.154%**.  It misses a `<=0.1%` matching
tolerance and fails the exact `bytes <= B` gate when `B` is the native
870,290,032 B.  More fundamentally, W4A4 native versus W8A8 FP8_CB_K36 is a
different whole execution contract.  These runs can identify promising kernels
and regressions; they cannot establish release parity.

Run these fixed shapes as the quick smoke on 0.6B and as the first 27B check:

| Workload | Input | Output | Prompts/block | Concurrency | Primary metric |
|---|---:|---:|---:|---:|---|
| Prefill | 1400 | 1 | 16 | 1 | TTFT |
| Batch-1 decode | 32 | 256 | 16 | 1 | TPOT, ITL, output tok/s |
| Mixed online | 1024 | 128 | 32 | 4 | E2EL, TTFT, TPOT, p99 ITL |

For every fixed row use `--input-range-ratio 0`, `--request-rate inf`, four
unmeasured warmup prompts per block, dataset seed 1234, and **at least three
blocks**.  The runner derives a distinct prompt-dataset seed for each block
(`1234`, `1235`, `1236`, ...); use the same base seed for the native and
Gridbook arms so every block is paired.
This seed controls prompt construction, not generation.  Server sampling is
separately and unconditionally pinned to greedy decoding with `--temperature
0`.  Fixed lengths are enforced with vLLM's random dataset at range ratio zero
plus `ignore_eos`; the runner rejects a block if completed prompts, per-request
lengths, or total input/output tokens differ from the requested values.

For a prompt-length distribution, set `--input-range-ratio R` with
`0 <= R < 1`.  The pinned vLLM client receives
`{"input": R, "output": 0}`, so output length remains exact.  vLLM samples
integer input lengths up through `ceil(input_len * (1 + R))`; its lower endpoint
also depends on tokenizer-added special tokens and decode/re-encode correction,
neither of which is emitted in result JSON.  The harness therefore uses the safe
conservative envelope `[1, ceil(input_len * (1 + R))]`, requires integer positive
per-request lengths, and requires their sum to equal `total_input_tokens`.  The
exact per-request equality check remains exclusive to `R=0`.

Keep prefix caching off for this primary matrix.  vLLM's readiness probe and
warmups deliberately reuse the first request, so an enabled prefix cache would
turn that measured request into a cache-hit benchmark.  Prefix-cached serving
is valuable, but it belongs in a separately labelled workload with an explicit
hit-rate distribution.

Three blocks are the minimum smoke-quality measurement, not permission to
overstate a small delta.  Use five or more blocks near parity or when variance
is material.  Run the native and Gridbook arms back-to-back, reverse their order
in a second session, and include every block value rather than reporting only
the best one.  Restarting the server between final-validation sessions is
preferred; allow the same model-load/JIT warmup before each measured arm.

## Release matrix: 27B and shipped MoE behavior

A release claim repeats the smoke shapes on the exact byte-admissible 27B pair
and covers the serving scheduler, not just one batch-1 kernel point:

| Lane | Required sweep/evidence |
|---|---|
| Prompt distribution | Fixed rows plus at least one declared nonzero input-range ratio using identical seeds |
| Concurrency | Ladder `1, 2, 4, 8` plus the shipped maximum if higher; report every block and tail ITL |
| Chunked prefill | Matched runs with chunked prefill off and on, recording chunk size/scheduler flags |
| Plain decode | Non-speculative, batch-1 `M=1`; report TPOT/ITL and output throughput |
| Shipped decode | Speculative and/or batched decode at the production `M`, including acceptance rate/length and routed work |
| MoE routing | Per-layer/per-step routed-token histograms, expert occupancy, active experts, and grouped-operator shape |
| Grouped MoE | Time the whole routed/grouped operator including routing, packing, launches, kernel, and combine—not an isolated inner GEMM |

`vllm bench serve` supplies request timing but does not emit speculative
acceptance or MoE routing histograms.  Attach matched server telemetry/log
artifacts for those fields, keyed by run label, block seed, and timestamps.
Record the production batch/routed-token `M` seen in those attachments.  Never
extrapolate an `M=1` expert microbenchmark to grouped MoE serving: routing skew,
packing, occupancy, expert grouping, and combine overhead are part of the
operator that must reach parity.

Chunked-prefill on/off, eager/graph mode, speculation, and scheduler changes are
separate labelled cells.  They must not be averaged together.  Acceptance-rate
differences can move effective throughput even when the underlying decode
kernel is unchanged, so publish acceptance beside throughput for every
speculative arm.

## Invocation template

Supply model locations and revisions from the environment or your job runner;
the harness contains no machine-local model paths.  For example, the decode row
for either size/arm is:

```bash
gridbook-bench-serve \
  --base-url "$BASE_URL" \
  --model "$TOKENIZER_ID" \
  --tokenizer "$TOKENIZER_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --model-id "$MODEL_ID_AT_REVISION" \
  --image-id "$VLLM_IMAGE_DIGEST" \
  --git-commit "$GRIDBOOK_COMMIT" \
  --run-label "$ARM-$SIZE-decode" \
  --artifact-bytes "$WHOLE_ARTIFACT_BYTES" \
  --byte-budget "$BYTE_BUDGET_BYTES" \
  --format-rung "$FORMAT_RUNG" \
  --serialized-layout "$SERIALIZED_LAYOUT" \
  --scale-coding "$SCALE_CODING" \
  --quant-contract "$WEIGHT_ACTIVATION_CONTRACT" \
  --kernel-backend "$CONCRETE_KERNEL_BACKEND" \
  --tensor-parallel-size "$TP_SIZE" \
  --fallback-state "$FALLBACK_STATE" \
  --client-runtime-id "$BENCH_CLIENT_VLLM_RUNTIME_ID" \
  --server-runtime-id "$SERVER_VLLM_RUNTIME_ID" \
  --gpu-id "$GPU_ID" \
  --driver-version "$NVIDIA_DRIVER_VERSION" \
  --cuda-version "$SERVER_CUDA_VERSION" \
  --server-arg=--enforce-eager \
  --server-arg="--max-model-len=$MAX_MODEL_LEN" \
  --runner-env CUDA_VISIBLE_DEVICES \
  --server-env "CUDA_VISIBLE_DEVICES=$SERVER_CUDA_DEVICE" \
  --server-env PRISMAQUANT_CB_DECODE=cuda \
  --input-len 32 \
  --output-len 256 \
  --num-prompts 16 \
  --max-concurrency 1 \
  --warmups 4 \
  --blocks 3 \
  --dataset-seed 1234 \
  --input-range-ratio 0 \
  --request-rate inf \
  --output "results/$ARM-$SIZE-decode.json"
```

Use the same template with the prefill and mixed-online values from the table.
`--served-model-name` is the API alias; `--model-id` is the immutable artifact
identity.  They are deliberately separate because production servers often use
a short alias for a long Hub revision.  `--git-commit` may be omitted when the
runner is invoked from a Git checkout; an installed wheel has no repository
metadata, so supply the exact commit used to build it.  The byte and execution
identity arguments are required: the runner refuses an artifact larger than
the declared budget before issuing a request.

The benchmark does not launch a server.  Start native and Gridbook servers with
the recorded, matched arguments, wait until model loading and JIT compilation
finish, then point the runner at the active arm.  Never serve both large arms on
the same accelerator during a comparison.

## Structured report and acceptance

Each JSON report has schema `gridbook.vllm-bench-serve.v1` and contains:

- UTC start/end timestamps and wall duration;
- Gridbook git commit/dirty state and package, vLLM, Python, platform and host
  versions;
- supplied image, artifact, tokenizer, endpoint and served-model identities;
- exact whole-artifact bytes, budget, headroom, and byte-scope definition;
- structured format/rung, serialized layout/scale coding, W/A contract,
  concrete backend/fallback, TP size, client/server runtimes, GPU, driver, and
  CUDA;
- recorded server flags and dispatch environment;
- runner environment and explicitly supplied server environment in separate
  fields;
- the exact workload controls, block seeds, and full vLLM command for each block;
- the input-length ratio, validation envelope, and observed per-request lengths;
- raw vLLM detailed output for each block; and
- mean, median, min, max and sample standard deviation across block metrics.

A nonzero client exit, failed request, missing streaming metric, early EOS, or
length mismatch makes the report `failed`.  Required throughput and every
requested percentile must be finite; TTFT and E2EL must be positive.  A
multi-token response must also contain positive per-request ITL samples.
One-token prefill probes are explicitly exempt from ITL/TPOT positivity because
there is no post-first-token interval.  A partial/failure report retains the
redacted command, return code, raw result when one exists, and validation error
so the run cannot be mistaken for missing data.  The output name is atomically
reserved before probes begin, preventing concurrent clients from both claiming
it; existing reports are never overwritten unless `--overwrite` is explicit.

Detailed ITL entries are streamed **chunk** intervals, not a promise of one
entry per generated token: one chunk can carry several regular or
speculatively accepted tokens.  The harness therefore never requires
`len(itls) == output_len - 1`.  It reconciles TTFT and flattened ITL means and
medians against the detailed arrays, reconstructs per-request E2EL as
`ttft + sum(itls)` with a small terminal-event allowance, and checks mean TPOT
against `(mean E2EL - mean TTFT) / (fixed output_len - 1)`.  These consistency
checks reject contradictory summary data without inventing token arrival times
that the client did not observe.

Reports are created with mode `0600`, but treat them as **sensitive raw output**.
`--save-detailed` includes generated model text and server error strings, which
the harness cannot safely infer are public.  Review or remove those fields
before attaching a report to an issue or publishing it.  Metadata credentials
are redacted; arbitrary secrets echoed inside generated text or a server error
are not.

For each paired block compute `Gridbook / native` for throughput and
`native / Gridbook` for latency, so values above 1 consistently mean Gridbook
is faster.  Report the median paired ratio and every constituent ratio.  Define
the parity tolerance before looking at the results; do not call an interval
that crosses it a win.  Decode parity is judged on TPOT/ITL and output
throughput, prefill parity on TTFT, and mixed-load parity on all four latency
metrics plus request/output throughput.  A throughput gain does not excuse
regressed tail ITL or failed requests.

For a release-grade claim, attach both reports, the exact server logs containing
Gridbook dispatch lines, hardware clocks/power state, and the model/body-byte
calculation **plus the exact whole-shipped-artifact byte inventory**.  Attach
the matched acceptance/routing telemetry required by the release matrix.  Re-run
generated-text correctness or quality evaluation whenever
a performance change alters numerical association or dispatch, even if this
latency protocol passes.
