# Native-parity serving protocol

This is the reproducible performance gate for work intended to bring Gridbook
to native vLLM parity. It compares the same model family, exact whole-artifact
byte budget, serving image, and controlled request distribution while recording
the complete quantization/backend/activation contract. It is a protocol, not a
performance or quality claim.

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
| Software | Same immutable vLLM image digest, accelerator runtime/driver, Gridbook commit, and package version |
| Server | Same model length, KV dtype, eager/graph mode, memory utilization, batching and scheduler flags |
| Execution | Explicit format/rung, serialized layout, scale coding, W/A contract, concrete backend, TP size, and fallback state |
| Features | Prefix caching off; speculative decode, LoRA, request logging, and observability identically configured or off |
| Hardware | Same GPU, power/clock policy, thermals, and no competing workload |
| Client | Same requested/observed input contract, output length, prompts, dataset seed, arrival rate, concurrency, warmups, and blocks |

Use a native quantized artifact made from the same base model. The formal size
gate is integer arithmetic: **`whole_served_artifact_bytes <= B`**.  “Whole
served artifact” means every model shard, config, tokenizer file, codebook,
scale/index sidecar, and other file actually shipped for serving.  Download,
JIT/compiler, engine, and KV caches are excluded.  Target bpp, quantizable-body
bytes, and resident VRAM remain useful diagnostics, but none substitutes for
the whole-artifact gate. Model-payload bytes are recorded separately with an
explicit scope and are never used for this gate. BF16 is a useful ceiling, not
the native denominator;
a 27B Gridbook result cannot be compared with a differently sized native model.

The runner requires a SHA-256-bound inventory document, recomputes its integer
file-byte sum, requires that sum to equal `--artifact-bytes`, and then applies
the budget. It accepts a standalone `gridbook.artifact-inventory.v1` (canonical
relative `path`, non-negative integer `bytes`, and file `sha256` per entry), a
standalone `prismaquant.cb_export_artifact_inventory.v1`, or the producer's
`quant_config.json` containing that PrismaQuant inventory under
`provenance.artifact_inventory`. This lets new Gridbook exports feed their
fixed-point whole-directory inventory directly into the harness. It also binds
legacy/native reports to a reviewable document rather than an unaudited number
on the command line. The runner validates the document and digest, but does not
reopen terabytes of model shards; artifact publication must separately verify
file contents. A standalone Gridbook inventory carries per-file digests for
that purpose, while the current embedded PrismaQuant inventory carries sizes
and is bound as a whole by the `quant_config.json` digest.
An embedded export inventory is admissible only while the served file set and
sizes still match it. If publication adds a model card, `.gitattributes`, or any
other served-directory file, generate a standalone inventory from the final
downloaded snapshot; the pre-upload export inventory is then payload provenance,
not the formal whole-shipped-artifact gate.

The runner requires opaque `--image-id` and `--model-id` values and records them
verbatim.  Prefer an image digest and a model repository plus immutable commit,
not mutable tags such as `latest` or `main`.  `--server-arg` records the exact
server command line; it is metadata only and never starts or changes a server.
The runner and server may be different containers or hosts, so their
environments are never conflated.  `PRISMAQUANT_*` values visible to the
benchmark process are labelled **runner environment**; add other runner values
with `--runner-env`.  Record the actual separately managed server environment
explicitly with repeatable `--server-env NAME=VALUE`.  Credentials, URL user
info, sensitive URL query values, every nonempty URL fragment, authorization
headers, and secret-like arguments/environment keys are redacted from metadata
and recorded commands.

Format labels are not execution evidence. Every report requires the
format/rung, serialized weight layout, scale coding, full weight/activation
contract (`W4A4`, `W8A8`, and so on), concrete kernel/backend, tensor-parallel
size, fallback state, exact server vLLM/runtime build, GPU identifier, driver,
and accelerator runtime (for example CUDA or ROCm). Client and server runtime
identifiers are separate required fields because the benchmark may run outside
the serving container.  Attach
server log lines proving the backend and fallback state; `--server-arg` and a
filename containing “native” do not prove dispatch.

Every report also requires a SHA-256-bound
`gridbook.execution-manifest.v1`. Its `assignments` enumerate serving units and
their format/rung, layout, scale coding, W/A contract, concrete backend, and
fallback state, and it is bound to the artifact-inventory digest and TP size.
It must declare `coverage: "all_serving_units"`; a sampled-layer manifest is
not release evidence.
Uniform runs must repeat the manifest's one value on the corresponding CLI
identity fields. If an assignment uses more than one value, that CLI field must
say `mixed`. The manifest is the authoritative identity for mixed menus; a
single average bpp or headline format is not.

Minimal manifests look like this (real inventories enumerate every shipped
file, and real mixed menus enumerate every serving unit or an auditable complete
set of non-overlapping unit groups):

```json
{"schema":"gridbook.artifact-inventory.v1","total_bytes":123,"files":[{"path":"config.json","bytes":123,"sha256":"<64 hex>"}]}
```

For a new PrismaQuant export, pass its existing `quant_config.json` instead of
translating the embedded inventory into the standalone form.

```json
{"schema":"gridbook.execution-manifest.v1","coverage":"all_serving_units","artifact_inventory_sha256":"<inventory JSON SHA-256>","tensor_parallel_size":1,"assignments":[{"unit":"model.*","format_rung":"FP8_CB_K36","serialized_layout":"product-codebook-indices-v1","scale_coding":"e4m3-per-block","quant_contract":"W8A8","kernel_backend":"gridbook-cuda-cb-gemv-v2","fallback_state":"none-observed"}]}
```

The digest arguments are the SHA-256 of the exact JSON file bytes, not a digest
of a reserialized in-memory object.

One important delegated-NVFP4 trap is **Marlin contract drift**.  An auto/native
MoE path may select Marlin while dropping activation scales, making the executed
operator W4A16 rather than W4A4.  Such a run cannot be labelled W4A4 even if the
serialized weights are NVFP4.  Record the executed W/A contract and concrete
backend, and prove both from server logs.  A silent fallback is a different arm,
not a slower sample of the requested arm.
An intentional Gridbook W4A16 arm is valid when its exact packing/metadata,
profile, loader, and delegated backend are recorded as W4A16; it must never be
used as evidence for W4A4 merely because both reuse a native weight family.

The opt-in fused native-FP4 prefill path is another contract-changing cell, not
a transparent acceleration switch: it moves activation scaling from the
FP32-emulated group-scale bucket to native UE4M3 factors. Keep it explicitly
labelled and opt-in until its own same-session served KL/PPL/task and routing/
shape gates pass; arithmetic or latency parity alone cannot promote it.

## Approximate 0.6B smoke matrix

Use the 0.6B pair for rapid iteration only.  The small model exposes launch and
dispatch overhead quickly, but its currently available arms are not a formal
native-parity pair:

| 0.6B arm | Model payload scope | Exact whole directory | Executed contract |
|---|---:|---:|---|
| Native | 870,290,032 B | 886,191,111 B | W4A4 delegated NVFP4 (only if logs prove the non-Marlin W4A4 path) |
| Gridbook CB | 871,628,664 B (model plus `cb_codebooks.pqcb`) | 887,531,487 B | W8A8 FP8_CB_K36 |

At payload scope CB is **+1,338,632 B / +0.154%**. At exact whole-directory
scope it is **+1,340,376 B / +0.15125%**. Both miss a `<=0.1%` matching
tolerance, and CB fails the exact `bytes <= B` gate when `B` is the native
whole-directory size. More fundamentally, W4A4 native versus W8A8 FP8_CB_K36 is a
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
0`. `--input-len` is the value sent to vLLM as `--random-input-len`; it is a
dataset-generation request, not an observed-token assertion. The pinned random
dataset subtracts tokenizer special tokens, so a requested length of 64 can
legitimately appear as 63 in `input_lens`. For every range-ratio-zero cell,
determine that tokenizer-specific value in a dry run and provide it separately
as `--observed-input-len`. The runner records both values and rejects a block
unless every detailed input length and `total_input_tokens` match the declared
observed value exactly. Output length remains exact through `ignore_eos`.

For a prompt-length distribution, set `--input-range-ratio R` with
`0 <= R < 1`.  The pinned vLLM client receives
`{"input": R, "output": 0}`, so output length remains exact. Tokenizer-added
special tokens and decode/re-encode correction make the exact observed endpoints
tokenizer-specific. Determine them from the pinned client/tokenizer and declare
the inclusive contract with `--observed-input-len-min` and
`--observed-input-len-max`. The harness requires every detailed input length to
fall within that declared range and their sum to equal `total_input_tokens`; it
does not relabel the requested `--input-len` as an observed bound. The exact
per-request equality contract remains exclusive to `R=0`.

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

The pinned `vllm bench serve` emits speculative acceptance rate and length,
draft count, draft/accepted token counts, and per-position acceptance rates.
The runner requires and arithmetically reconciles every field when
`--speculative-mode=on`, and forbids those fields when the mode is `off`.
It does not emit MoE routing histograms. Attach matched server telemetry/log
artifacts for routing, keyed by run label, block seed, and timestamps.
Record the production batch/routed-token `M` seen in those attachments. Never
extrapolate an `M=1` expert microbenchmark to grouped MoE serving: routing skew,
packing, occupancy, expert grouping, and combine overhead are part of the
operator that must reach parity.

Chunked-prefill on/off, eager/graph mode, speculation, and scheduler changes are
separate labelled cells.  They must not be averaged together.  Acceptance-rate
differences can move effective throughput even when the underlying decode
kernel is unchanged, so publish acceptance beside throughput for every
speculative arm.

## Performance is a constraint, not the quality objective

A release assignment minimizes predicted quality loss subject to hard limits:
exact whole-artifact bytes `<= B`, p95 TTFT, p95 ITL and/or p05 throughput,
resident weights plus KV plus peak scratch memory, backend/shape/TP legality,
and serving-unit coupling. Do not replace those limits with
`quality + lambda * time`, one blended `serve_ms`, or the sum of each layer's
independently fastest format. That synthetic reference can exceed the byte
budget. The performance denominator is the fastest **globally feasible**
assignment under the same whole-artifact budget and execution constraints.

Kernel and per-layer timings can propose candidates; they cannot promote an
assignment. Final selection requires every arm to be evaluated in the same
session against the same BF16 teacher capture with served KL/PPL and the
declared downstream task suite, then measured end to end with this workload
matrix. This prevents cross-session KL drift, a changed activation bucket, or
an M=1 microbenchmark from being reported as a quality-preserving native-parity
win. If a dispatch change alters numerical association, repeat the same-session
quality gate even when generated smoke text looks unchanged.

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
  --artifact-inventory "$ARTIFACT_INVENTORY_JSON" \
  --artifact-inventory-sha256 "$ARTIFACT_INVENTORY_SHA256" \
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
  --driver-version "$SERVER_DRIVER_VERSION" \
  --accelerator-runtime "$SERVER_ACCELERATOR_RUNTIME" \
  --execution-manifest "$EXECUTION_MANIFEST_JSON" \
  --execution-manifest-sha256 "$EXECUTION_MANIFEST_SHA256" \
  --speculative-mode off \
  --server-arg=--enforce-eager \
  --server-arg="--max-model-len=$MAX_MODEL_LEN" \
  --runner-env CUDA_VISIBLE_DEVICES \
  --server-env "CUDA_VISIBLE_DEVICES=$SERVER_CUDA_DEVICE" \
  --server-env PRISMAQUANT_CB_DECODE=cuda \
  --input-len 32 \
  --observed-input-len "$OBSERVED_INPUT_LEN" \
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
metadata, so supply the exact commit used to build it. The byte inventory, its
digest, the execution manifest, and its digest are required. The runner verifies
their bindings and refuses an artifact larger than the declared budget before
issuing a request. For a speculative cell use `--speculative-mode on
--speculative-config '{"method":"...","num_speculative_tokens":K,...}'`; the
JSON must describe the exact server configuration, not merely say “enabled,”
and the same strict-JSON object must appear in the recorded
`--server-arg=--speculative-config=...`. A mismatch or a speculative server flag
in an `off` cell fails before requests are issued.
Optional payload bytes use `--payload-bytes` together with a precise
`--payload-scope`.

The benchmark does not launch a server.  Start native and Gridbook servers with
the recorded, matched arguments, wait until model loading and JIT compilation
finish, then point the runner at the active arm.  Never serve both large arms on
the same accelerator during a comparison.

## Structured report and acceptance

Each JSON report has schema `gridbook.vllm-bench-serve.v2` and contains:

- UTC start/end timestamps and wall duration;
- Gridbook git commit/dirty state and package, vLLM, Python, platform and host
  versions;
- supplied image, artifact, tokenizer, endpoint and served-model identities;
- exact inventory-derived whole-artifact bytes, inventory digest/reference,
  budget, headroom, and byte-scope definition, plus separately scoped optional
  payload bytes;
- structured format/rung, serialized layout/scale coding, W/A contract,
  concrete backend/fallback, TP size, client/server runtimes, GPU, driver, and
  accelerator runtime, plus the digest-bound per-serving-unit execution
  manifest;
- recorded server flags and dispatch environment;
- runner environment and explicitly supplied server environment in separate
  fields;
- the exact workload controls, structured speculation mode/config, block seeds,
  and full vLLM command for each block;
- the input-length ratio, validation envelope, and observed per-request lengths;
- raw vLLM detailed output for each block; and
- mean, median, min, max, NumPy-linear block p05/p95, and sample standard
  deviation across block metrics.

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
The CLI enforces at least four warmups, at least three independent blocks, and a
percentile set containing p95; a report cannot quietly weaken the minimum
TTFT/ITL gates.

Detailed ITL entries are streamed **chunk** intervals, not a promise of one
entry per generated token: one chunk can carry several regular or
speculatively accepted tokens.  The harness therefore never requires
`len(itls) == output_len - 1`. It reconstructs mean, median, population standard
deviation, and every requested percentile for TTFT and flattened ITL with
NumPy-linear percentile semantics. It reconstructs per-request E2EL as
`ttft + sum(itls)` and TPOT as `sum(itls) / (fixed output_len - 1)`, checking the
same complete aggregate set with a small declared terminal-event allowance.
Request, output-token, and total-token throughput are independently recomputed
from the finite positive raw duration and exact detailed totals. These checks
reject contradictory summary data without inventing one ITL per token.

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
