# `rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm` — staged card block

Staged, **not uploaded**. Read the APPLY GATE in [`README.md`](README.md) first.

---

## 1. Front matter (replaces the existing YAML block)

```yaml
license: openmdw-1.1
base_model: poolside/Laguna-S-2.1
base_model_relation: quantized
library_name: gridbook
pipeline_tag: text-generation
tags:
  - gridbook
  - codebook-quantization
  - codebook
  - vector-quantization
  - prismaquant
  - vllm
  - quantized
  - mixed-precision
  - nvfp4
  - fp8
  - blackwell
  - dgx-spark
  - moe
  - long-context
  - single-gpu
  - code
  - conversational
```

## 2. Body block (insert after the intro bullets; replaces `## Serve`)

````markdown
## Serve this model

| | |
|---|---|
| **Plugin** | [`gridbook`](https://github.com/RobTand/gridbook) — an out-of-tree vLLM quantization plugin. Stock vLLM, no fork, no core patches. |
| **GPU** | NVIDIA Blackwell, compute capability `sm_120` / `sm_121`. Measured on GB10 / DGX Spark (`sm_121`). On older GPUs the plugin still loads but runs its **Triton fallback** kernels — correct, not fast, and not a production serving target. |
| **Memory** | 89.4 GB of weights on disk = **83.2 GiB** resident. Needs a ~128 GB unified/VRAM pool: measured on one DGX Spark at `--gpu-memory-utilization 0.85`. Only 12 of 48 layers are full-attention (the rest slide at w=512), so a full 256k request costs ~6 GiB of fp8 KV — model + 256k cache fit the Spark's pool. |
| **vLLM** | Version used for these measurements is not recorded on this card. |
| **Toolchain** | CUDA toolkit with `nvcc` on `PATH` **in the serving container** — the plugin JIT-builds its kernels on first model load (~30 s, cached). `nvcc` 13.0 is the tested toolchain. |
| **Parallelism** | Single GPU (`tp=1`). Tensor parallelism is not implemented in the plugin. |

```bash
pip install gridbook

vllm serve rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --max-num-batched-tokens 16384
```

Check it answers:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
```

**How to tell the fast path is active.** The plugin *fail-softs*: if the CUDA
extension cannot be built it still serves, through Triton fallback kernels, at a
fraction of the speed. That is not a crash — it is a line on stderr, tagged
`[prismaquant-cb]`:

```bash
vllm serve ... 2>&1 | grep '\[prismaquant-cb\]'
```

Any `WARNING` or `ERROR` on that tag means the CUDA decode path did **not**
load, and the numbers on this card are **not** reachable on your box. (The tag
is also used for a few harmless informational lines; it is the `WARNING` /
`ERROR` ones that matter.) The exact wording differs between plugin versions —
**grep the tag, not the sentence.** The two that matter read roughly:

```
[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV extension could not be built (<ExcType>: …); falling back to the Triton decode path (slow prototype). To get the CUDA path: …
[prismaquant-cb] ERROR: broken gridbook install — gridbook is installed without its CUDA sources: …
```

The first is an environment problem (usually no `nvcc` in the serving
container, or a torch/CUDA version mismatch); the second is a defect in the
install itself. Both are diagnosed in
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md#the-cuda-extension-did-not-load-triton-fallback).
Set `PRISMAQUANT_CB_EXT_DIR` to a writable, persistent directory to keep the
one-time JIT build across restarts (important in containers).

Notes for this model:

- **The command above is the fast-prefill configuration** and is the one the
  speed numbers on this card were measured with. It leaves 1.53× full-256k
  request concurrency. Dropping `--max-num-batched-tokens` and running default
  batching at `--gpu-memory-utilization 0.87` gives 3.28× concurrency instead,
  at ~6× slower long-prompt prefill. Pick per workload.
- **`--enforce-eager` is required for this artifact.** Its body is FP8-CB, which
  uses an M-gated decode dispatch — a host-side branch that CUDA-graph capture
  cannot see. Serving without it makes decode *worse*, because the server pads
  captured decode batches above the prefill-M threshold and every graphed step
  takes the expand path. See
  [`docs/KERNELS.md` §CUDA-graph safety rules](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md#cuda-graph-safety-rules).
- **Serve drafter-free.** A DFlash drafter was measured at 37.5% acceptance on
  this hardware — net-negative to parity. It adds nothing here.

**More:** [install + hardware matrix](https://github.com/RobTand/gridbook/blob/master/docs/INSTALL.md) ·
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md) ·
[format spec](https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md) ·
[kernels](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md) ·
[benchmarks + protocol](https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md) ·
[issues](https://github.com/RobTand/gridbook/issues)

**Other gridbook artifacts:**
[Qwen3.6-27B 5.5-bit](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) (23 GB, VL + MTP) ·
[Hy3-295B-A21B 2.9-bit](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) (106 GB, 295B MoE + MTP)
````

---

## Notes — NOT part of the card

**What changed vs. the live card (fetched 2026-07-28).**

| Live card | Staged | Why |
|---|---|---|
| `pip install git+https://github.com/RobTand/gridbook` | `pip install gridbook` | The live command is the **worst** of the three across the family: it installs cleanly and non-editably, so `<site-packages>/csrc` does not exist, every CUDA build fails, the plugin fail-softs to Triton — and the user gets a working server that is several times slower than this card's own 14.9 tok/s / 1,822 tok/s numbers, with only a stderr line to say so. It is also unpinned. |
| Serve flags spread across the intro bullet, the `## Serve` block, and a parenthetical under the A/B table | one block + one "pick per workload" note | The concurrency/prefill tradeoff is real and should be stated once, next to the command it modifies. |
| Target hardware in prose ("Blackwell (GB10 / sm_120+)") | requirements table | `sm_120+` is not the right statement — the plugin has a `sm_80` load floor and a `sm_120`/`sm_121` *fast* path; a Hopper user passes the floor and gets Triton. |
| No verification step | `curl` + a `[prismaquant-cb]` stderr check | Same reasoning as the other cards. |
| No cross-links | links to the other two gridbook artifacts | Every gridbook card is currently a dead end. |

**Plugin-version references need a tag.** The live card cites
"plugin 2026-07-23", "plugin >= 2026-07-23" and "plugin ≥2026-07-28". The last
one names something a user cannot obtain: the newest public commit on
`RobTand/gridbook` is `b4694e3`, pushed 2026-07-27, and the repo has **0 releases
and 0 tags**. Once a version is tagged, replace all three date references with
that tag — including the one inside the `## Measured A/B vs poolside's NVFP4
release` table ("prefill @8k 2,186 tok/s (plugin ≥2026-07-28)"), which is the one
a skeptical reader will try to reproduce.

**⚠ The live card's sizes are GiB values printed with a `GB` label.** The card
says "84 GB" for the artifact and "84 GB / 67 GB" in the weights row of the
`## Measured A/B vs poolside's NVFP4 release` table. Measured against the HF API
(`?blobs=true`, summed over `siblings`, 2026-07-28):

| | HF API bytes | as GB | as GiB | live card |
|---|---|---|---|---|
| `rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm` | 11 files | **89.37 GB** | **83.23 GiB** | "84 GB" |
| ↳ `model.safetensors` alone | 1 file | 89.36 GB | 83.22 GiB | — |
| `poolside/Laguna-S-2.1-NVFP4` | 27 files | **71.94 GB** | **67.00 GiB** | "67 GB" |

Both live-card figures are the **GiB** number wearing a `GB` label — 83.22 GiB
rounds to 84, and poolside's is 67.00 GiB to two decimals. The staged Memory row
therefore reads `89.4 GB … = 83.2 GiB`, matching the repo's own
[`README.md`](../../README.md) artifact table and
[`docs/INSTALL.md`](../INSTALL.md#per-artifact-requirements), which already state
`89.4 GB (83.2 GiB)`.

Two consequences for whoever applies this card:

- The sidecars cannot explain the gap and must not be blamed for it: every file
  other than `model.safetensors` sums to **0.000252 GB**, ~4 × 10⁵ times too
  small for a 5.4 GB discrepancy.
- **The A/B table's byte *ratio* is still correct** — 67.00 / 83.22 = 0.805, the
  "67/84 = 0.80" the card's per-byte decode reading rests on. Only the unit
  labels are wrong. Fix the labels in that table (`83.2 GiB` / `67.0 GiB`, or
  `89.4 GB` / `71.9 GB`) without touching the ratio or the conclusion drawn from
  it. That table is outside the staged block, so it is a separate edit to the
  same card.

**Left untouched.** The format-breakdown table, the honest-limitations section
(including "No quality claims" — correct and important), the measured A/B against
poolside's NVFP4 release with its per-byte-neutral decode reading, and the
attribution line.

**Sources.** Card text, serve command, speed numbers, and the concurrency
tradeoff: live card (all copied verbatim, nothing re-derived). Size: HF API
`?blobs=true` (see the table above); created 2026-07-23, modified 2026-07-27,
311 all-time downloads. `--enforce-eager`: `docs/KERNELS.md` §CUDA-graph safety
rules. Fail-soft stderr tag `[prismaquant-cb]`: `gridbook/cuda_ext.py`,
`get_ext()` — cite the symbol, not a line number; the message text and its line
number have both already moved once. Serve-by-Hub-id: public repo commit
`7f2b502` (the in-tree development hash `002471f` does not resolve in
`RobTand/gridbook`).
