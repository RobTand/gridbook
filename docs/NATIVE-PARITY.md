# Native-parity serving protocol

This is the reproducible performance gate for work intended to bring Gridbook
to native vLLM parity.  It compares the same model family, byte budget, serving
image, and fixed request stream while changing only the weight format/plugin
dispatch under test.  It is a protocol, not a performance claim.

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

## What must match

Before running either arm, record this checklist.  A row that differs is a new
experiment, not another sample of the same experiment.

| Dimension | Required match |
|---|---|
| Base model | Same architecture, tokenizer, revision, and non-quantized tensors |
| Byte budget | Same target bpp and quantizable-body bytes; report the actual byte difference |
| Software | Same immutable vLLM image digest, CUDA/driver, Gridbook commit, and package version |
| Server | Same model length, KV dtype, eager/graph mode, memory utilization, batching and scheduler flags |
| Features | Prefix caching off; speculative decode, LoRA, request logging, and observability identically configured or off |
| Hardware | Same GPU, power/clock policy, thermals, and no competing workload |
| Client | Same input/output lengths, prompts, seed, arrival rate, concurrency, warmups, and number of blocks |

Use a native quantized artifact made from the same base model and matched body
bytes.  BF16 is a useful ceiling, but it is not the native denominator for this
test.  A 27B Gridbook result must not be compared with a differently sized
native model.

The runner requires opaque `--image-id` and `--model-id` values and records them
verbatim.  Prefer an image digest and a model repository plus immutable commit,
not mutable tags such as `latest` or `main`.  `--server-arg` records the exact
server command line; it is metadata only and never starts or changes a server.
All `PRISMAQUANT_*` environment values are captured automatically.  Add other
dispatch variables with `--dispatch-env`; names that look secret are redacted.

## Fixed 0.6B and 27B matrix

Use a matched 0.6B pair for rapid iteration, then repeat unchanged on the 27B
pair before accepting a change.  The small model exposes launch and dispatch
overhead quickly; the 27B gate establishes that the result survives realistic
weight traffic.  A win on 0.6B is a lead, not a final result.

Run all three shapes on both sizes:

| Workload | Input | Output | Prompts/block | Concurrency | Primary metric |
|---|---:|---:|---:|---:|---|
| Prefill | 1400 | 1 | 16 | 1 | TTFT |
| Batch-1 decode | 32 | 256 | 16 | 1 | TPOT, ITL, output tok/s |
| Mixed online | 1024 | 128 | 32 | 4 | E2EL, TTFT, TPOT, p99 ITL |

For every row use `--request-rate inf`, four unmeasured warmup prompts per
block, seed 1234, and **at least three blocks**.  The runner derives a distinct
seed for each block (`1234`, `1235`, `1236`, ...); use the same base seed for the
native and Gridbook arms so every block is paired.  Fixed lengths are enforced
with vLLM's random dataset at range ratio zero plus `ignore_eos`; the runner
rejects a block if completed prompts or total input/output tokens differ from
the requested totals.

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
  --server-arg=--enforce-eager \
  --server-arg="--max-model-len=$MAX_MODEL_LEN" \
  --dispatch-env CUDA_VISIBLE_DEVICES \
  --input-len 32 \
  --output-len 256 \
  --num-prompts 16 \
  --max-concurrency 1 \
  --warmups 4 \
  --blocks 3 \
  --seed 1234 \
  --request-rate inf \
  --output "results/$ARM-$SIZE-decode.json"
```

Use the same template with the prefill and mixed-online values from the table.
`--served-model-name` is the API alias; `--model-id` is the immutable artifact
identity.  They are deliberately separate because production servers often use
a short alias for a long Hub revision.  `--git-commit` may be omitted when the
runner is invoked from a Git checkout; an installed wheel has no repository
metadata, so supply the exact commit used to build it.

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
- recorded server flags and dispatch environment;
- the exact fixed workload, block seeds, and full vLLM command for each block;
- raw vLLM detailed output for each block; and
- mean, median, min, max and sample standard deviation across block metrics.

A nonzero client exit, failed request, missing streaming metric, early EOS, or
length mismatch makes the report `failed`.  A partial/failure report is still
written so the run cannot be mistaken for missing data.  Existing reports are
never overwritten unless `--overwrite` is explicit.

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
calculation.  Re-run generated-text correctness or quality evaluation whenever
a performance change alters numerical association or dispatch, even if this
latency protocol passes.
