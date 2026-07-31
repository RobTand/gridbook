# gridbook

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](docs/INSTALL.md)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](ROADMAP.md)

**Codebook-quantized LLM weights that serve at native tensor-core speed.**
`gridbook` is a **format specification** plus an **out-of-tree vLLM plugin** that
serves it: weights are stored as product-vector-quantization codebook indices
whose codebook entries lie exactly on a hardware grid (NVFP4 / FP8), so a decoded
tile *is* a native tile and feeds the same Blackwell tensor-core GEMM vLLM
already runs. No forked runtime, no patched vLLM core, no custom serving stack.

Ready-to-serve models are published on Hugging Face — [jump to the
list](#published-artifacts).

---

## What you need

| | Requirement |
|---|---|
| **GPU** | NVIDIA **Blackwell `sm_120` / `sm_121`** for the native-speed path (GB10 / DGX Spark is the reference; RTX 5090 is user-reported). Older NVIDIA cards partly work — read the [compatibility table](#compatibility) before assuming. |
| **CUDA toolchain** | `nvcc` on `PATH` **in the process that serves**, matching your torch build. Kernels are **JIT-compiled on first model load** (~30 s once, then cached), *not* at `pip install` time. `nvcc` 13.0 is the tested toolchain. |
| **PyTorch** | whatever build your vLLM uses (measured: `2.11.0+cu130`). |
| **vLLM** | already installed — gridbook is a plugin, not a runtime. Measured against `0.23.1rc1.dev764+g54b16d8a9`; see [compatibility](#compatibility). |
| **Parallelism** | **`tp=1` only.** The plugin has no tensor-parallel handling; multi-GPU sharding of CB weights is not implemented. |
| **Python** | ≥ 3.10 (measured on 3.12). |

Without `nvcc`, or on a non-Blackwell GPU, the plugin still loads and produces
**correct** output through fallback kernels — but they are a correctness path,
not a speed path. For dense decode the fallback is unchanged code and the gap is
measured: the Triton decode-GEMM did **4.20 tok/s** on the 27B where the CUDA
GEMV does **10.28**. For MoE the grouped CUDA GEMV is skipped entirely and the
regression is larger but artifact-dependent — that is a *historical*
before/after, not a fresh benchmark of today's fallback, and
[`BENCHMARKS.md`](docs/BENCHMARKS.md#what-the-fallback-costs) says exactly what
was and was not measured. See
[TROUBLESHOOTING](docs/TROUBLESHOOTING.md#the-cuda-extension-did-not-load-triton-fallback)
for how to tell which path you are on.

---

## Install

Install **into the environment that already has vLLM and torch** — gridbook does
not pin either.

```bash
pip install git+https://github.com/RobTand/gridbook
```

That is the only route that works today. `pip install gridbook` is **planned but
not yet published** — the PyPI name is unclaimed, so that command fails right
now. Track it on the [roadmap](ROADMAP.md#distribution).

There is nothing else to configure: the plugin registers itself through vLLM's
`vllm.general_plugins` entry point, and vLLM auto-detects a gridbook checkpoint
from its `config.json`. No `--quantization` flag.

Full matrix — toolchain versions, the three install routes, the first-run JIT
build, how to persist the build cache, and a copy-pasteable install check:
**[`docs/INSTALL.md`](docs/INSTALL.md)**. For the container route — a
[`Dockerfile`](Dockerfile) that builds vLLM + gridbook + the kernels in one
image — see **[`docs/CONTAINER.md`](docs/CONTAINER.md)**.

---

## Quickstart — serve a real model

```bash
pip install git+https://github.com/RobTand/gridbook

vllm serve rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm",
       "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}'
```

**How to tell it is actually working.** The failure is silent by design (the
plugin fail-softs to Triton rather than refusing to serve), so check for the
warning rather than for a crash:

- **Bad** — a `[prismaquant-cb] WARNING:` line on stderr saying the CUDA
  decode-GEMV extension could not be built means you are on the slow path.
  (Grep your vLLM log for `[prismaquant-cb]`, not for `gridbook` — the runtime
  log prefix still uses the project's older name.)
- **Good** — no such line, and the first model load pauses ~30 s the first time
  ever on that machine (the JIT build) and not on subsequent starts.
- A definitive check that does not require serving anything is in
  [`docs/INSTALL.md`](docs/INSTALL.md#verify-the-install).

Notes:

- **`--enforce-eager` is the tested configuration.** The decode dispatch branches
  on the token count on the host, which is capture-hostile; naive graph capture
  made decode *worse*, because vLLM pads captured decode batches past the
  prefill threshold and every graphed step then took the prefill path. Details
  and the exceptions: [`KERNELS.md`](docs/KERNELS.md#cuda-graph-safety-rules).
- **`--gpu-memory-utilization 0.85` is a deliberately conservative start**, not a
  tuned value. On a unified-memory box (GB10 / DGX Spark) that flag does not
  carve up private VRAM — it claims from the pool the host also lives in, and
  0.94/0.95 killed the whole machine under long-prefill spikes. Bring the server
  up, check several GiB of `MemAvailable` remain, then raise it. On a discrete
  GPU you can generally go higher. Details:
  [`INSTALL.md`](docs/INSTALL.md#per-artifact-requirements) and
  [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md#out-of-memory--box-hang-on-a-dgx-spark).
- Set `--max-model-len` to fit your KV budget. Adding `--kv-cache-dtype fp8`
  roughly halves KV bytes and is what the larger artifacts were measured with.
- Serving by Hub id (as above) or from a local directory both work — the
  codebook sidecar is fetched from the Hub when you pass a repo id.

---

## Published artifacts

All are standard Hugging Face directories (`config.json` + safetensors + the
`cb_codebooks.pqcb` codebook sidecar + tokenizer). Sizes are the sum of repo
files, read from the HF API on 2026-07-28.

| Model | Repo | Size on disk | bpp | Notes |
|---|---|---|---|---|
| Qwen3.6-27B (dense hybrid VL) | [`rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) | 23.0 GB (21.4 GiB) | 5.5 | The measured quality result below. Vision tower is stock NVFP4 W4A16, so it needs a vLLM NVFP4 backend as well as gridbook. |
| Laguna-S-2.1 (117B sparse MoE, coding) | [`rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm`](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) | 89.4 GB (83.2 GiB) | 6.0 | 256k context on one 128 GB box. **No quality claims** (no servable BF16 teacher at this scale). |
| Hy3-295B-A21B (295B MoE) | [`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) | 105.7 GB (98.5 GiB) | 2.9 | Serves on a **single** 128 GB box. **No quality-vs-teacher claim** at this scale. |

Each artifact needs enough memory for its weights **plus** KV cache and
activation headroom; see [`docs/INSTALL.md`](docs/INSTALL.md#per-artifact-requirements).

---

## Compatibility

The native path is Blackwell-targeted, but the decode kernel is *not*
arch-specific — this table separates what was **measured** from what is
**inferred from the code and untested**. Nothing here is inferred silently.

| GPU class | Decode (M ≤ 16) | Dense FP8-CB prefill | Mid-M fused prefill (16 < M ≤ 128) | Verdict |
|---|---|---|---|---|
| **GB10 / DGX Spark, `sm_121`** | CUDA GEMV | expand → stock W8A8 GEMM | CUTLASS `sm120` | **MEASURED.** Every published number comes from this box. |
| **RTX 5090, `sm_120`** | same | same | same | **USER-REPORTED** working (vLLM 0.25.1, 27B artifact, [issue #1](https://github.com/RobTand/gridbook/issues/1)) after that issue's memory patches. No speed numbers published; the exact `nvcc` arch flag (`sm_120` vs the measured `sm_121`) is untested here. |
| **H100 `sm_90`, RTX 4090 / L40S `sm_89`** | expected to work — the decode kernel contains no arch guards, no inline PTX and no `-arch` flag | expected to work (needs `sm_89+` fp8 support) | **fails, fail-softs** to the expand path with a warning (the fused kernel is `sm_120`-family only) | **INFERRED, UNTESTED.** Also gated by whatever the artifact's non-CB groups need (e.g. the 27B vision tower's NVFP4). |
| **A100 `sm_80`** | expected to work | **expected to fail**: the dense FP8-CB prefill calls `cutlass_scaled_mm` on fp8 with no capability guard and no fallback | n/a | **NOT RECOMMENDED.** The declared capability floor is 8.0, but a prompt longer than 16 tokens is expected to error. INFERRED from code; no A100 was tested. |
| **No `nvcc`, or non-NVIDIA** | Triton fallback | Triton fallback | n/a | **Correct, slow.** Use for numerics verification and CI, not for serving. |

Tested software stack (verified in-container, 2026-07-28): vLLM
`0.23.1rc1.dev764+g54b16d8a9`, torch `2.11.0+cu130`, triton `3.6.0`,
transformers `5.13.0`, CUDA `13.0.88`, Python `3.12.3`, **arm64**. vLLM `0.25.1`
and `0.23.1rc1.dev1060` are user-reported working on x86 + RTX 5090 (issue #1).
Everything else is untested. See
[`docs/INSTALL.md`](docs/INSTALL.md#tested-software-stack) for how versions can
drift and what that looks like.

---

## Why this exists

Below ~4 bits per weight, single-format scalar quantization loses quality fast,
and the methods that recover it (learned codebooks / vector quantization) have
historically been **slow to serve**: a VQ weight must be *decoded* before it can
be multiplied, and that decode usually runs on general CUDA cores, not tensor
cores.

Concretely, in the `llama.cpp` / GGUF ecosystem — today's most common low-bit
serving path:

- **k-quant** formats have a good fused CUDA matmul path (MMQ), so they serve
  reasonably fast, but their quality rungs stop around 3–4 bits.
- **IQ** formats (the sub-3-bit "I-quant" rungs) reach much better quality, but
  their decode falls to a **slower CUDA-core dequant path** — a *prefill tax* you
  pay on every prompt token.

`gridbook` targets **IQ-class quality at native-kernel speed**. The dtype in each
format name is the **grid the codebook's values live on, not the storage width**:
storage is a k-bit index per 8-weight vector, so `FP8_CB_K32` is a 4.0-bit/weight
encoding whose reconstructed values are exact fp8 numbers, and `NVFP4_CB_K16` is
a ~2.3-bit/weight encoding reconstructing exact NVFP4 codes. Because a decoded CB
tile *is* an NVFP4/FP8 tile:

1. **Decode (batch-1 / small M):** a bandwidth-bound fused GEMV streams the
   packed indices, expands per group in registers, and multiplies — fewer bytes
   per weight means *less* memory traffic than a wider format, so decode is at or
   above native-format speed.
2. **Prefill (large M):** the layer's weight tile is expanded **transiently** into
   a scratch buffer (bounded — never a resident dense weight), then run through
   the stock native GEMM.

Formats mix per-Linear with plain NVFP4, FP8 and BF16 inside one standard
`safetensors` checkpoint. See [`docs/MOTIVATION.md`](docs/MOTIVATION.md) for the
full argument and [`docs/KERNELS.md`](docs/KERNELS.md) for the kernel design.

---

## Headline results

Quality numbers are **at matched bits-per-weight** against a baseline quantized
to stock per-Linear NVFP4/FP8 (served natively by vLLM's `compressed-tensors`
path). Measured on a single **NVIDIA GB10 / DGX Spark** (Blackwell `sm_121`,
128 GB unified memory, ~273 GB/s), single calibration seed. Read the caveats in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — especially the ±17% measurement-
arithmetic band and the single-seed noise band.

| Model | bpp | Quality vs matched-bpp NVFP4/FP8 baseline | Serving speed |
|---|---|---|---|
| **27B dense hybrid** | 5.5 | **ALL-KL −58.3%**, confident-KL −52.9%, PPL gap to BF16 ~3× smaller | decode **at native parity** (10.3 vs 10.26 tok/s); dense prefill 1.44× the baseline's TTFT |
| **35B MoE**, 256 experts (**not published**) | 4.75 | **confident-KL −53%**, ALL-KL −43%, top-1 agreement higher on both slices | decode 32.6–33.3 tok/s — **faster than BF16** (28.4) |
| **295B MoE** (published) | 2.9 | *no quality-vs-teacher claim at this scale* | **serves on ONE DGX Spark**; prefill **~109 tok/s vs the matched-byte GGUF IQ build's 42 = ~2.6×**; base decode 14.6 tok/s (16.1 with the MTP draft) — below the GGUF build's ~18 |

- **The 27B row is the A/B build, not the published file — and the published
  file's own card reports different absolute numbers.** The −58.3% comes from
  the 2026-07-18 head-to-head, whose CB arm used a 4-rung menu
  (`FP8_CB_K36/K40/K44/K48`). The artifact actually on the Hub is a later,
  finer 8-rung ladder (`K36`–`K47`), measured in a **different** evaluation
  session, and its [model
  card](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm)
  reports ALL-KL **0.0049** and **−77%** where this table's session reads
  0.0134 and −58.3%. Both are real; **absolute KL is not comparable across
  sessions** (the *same unchanged* NVFP4/FP8 baseline artifact reads 0.02407
  confident-KL in one session and 0.01302 in the other), while the within-session
  relative verdict — CB wins big at matched bytes — reproduces in both. The
  arithmetic is worked through in
  [`BENCHMARKS.md`](docs/BENCHMARKS.md#two-sessions-two-builds-why-the-model-card-says-77-and-this-page-says-583).
  This page quotes the more conservative figure.
- The 35B MoE result is an internal measurement; **there is no downloadable 35B
  gridbook artifact**. It is reported because it is the evidence that the format
  win reproduces on Mixture-of-Experts, not as something you can reproduce
  without an encoder.
- The 295B result carries **no quality-vs-teacher claim**: a 295B BF16 reference
  cannot be run on a single box to compute KL against. What is validated is that
  it loads, serves, and generates coherent, arithmetically-correct output on one
  128 GB box, and its speed against a matched-byte GGUF build. Tool-use
  (ToolEvalBench) is **parity within the plateau band**, not a quality win: the
  shipped artifact scored 88/100 (130/148) on the ship config, against 87 for the
  GGUF IQ build and 86 for GGUF k-quant, with a measured single-seed churn band of
  ±2–3 points across serving configs.
- **MoE prefill is solved and on by default.** Measured on Laguna-S-2.1 (117B
  MoE): 293 → **1,821 tok/s** at 8k and 207 → **1,822 tok/s** at 63k after the
  CUDA chunk-expander landed (commit `8829c16`); the current default is a
  measured per-layer path selection. Docs that still describe MoE prefill as a
  per-expert loop are stale — see [`ROADMAP.md`](ROADMAP.md) for what is actually
  open.
- **Large-M dense prefill is the honest remaining gap**: ~1.44× the native GEMM's
  TTFT on the 27B.

---

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | Hardware/toolchain matrix, the three install routes, first-run JIT build, cache persistence, install verification. |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Real failure modes keyed to the exact message you see, with cause and fix. |
| [`docs/SPEC.md`](docs/SPEC.md) | **Normative format specification** — byte layout, product-VQ, v1/v2 scale coding, codebook sidecar, config vocabulary, extensibility contract. Implementation-independent, MUST/SHOULD language. |
| [`docs/MOTIVATION.md`](docs/MOTIVATION.md) | Why the 2–4 bpp regime needs codebooks, why serving has been the blocker, and an honest comparison to GGUF k-quant/IQ. |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Serving kernel design: transient-expand prefill, fused act-QDQ decode GEMV, two-tier in-register scale compose, grouped MoE GEMV, Triton fallbacks, CUDA-graph rules. |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | Operator reference for the plugin itself: dispatch, kernel set, environment switches. |
| [`docs/DELEGATED-NVFP4-MOE.md`](docs/DELEGATED-NVFP4-MOE.md) | Version-scoped backend selection for a **non-CB** NVFP4 MoE group on GB10 (`sm_121`), including Marlin's generic weight-only warning and a fail/unknown preflight policy. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | The measured results with full hardware/protocol context and caveats. |
| [`ROADMAP.md`](ROADMAP.md) | What is done, what is open, and what was measured and rejected. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to run the tests, what a good bug report contains, and what is in scope. |

---

## Naming

One project, several strings — this trips people up when grepping logs, so here
is the whole list:

| Where | String |
|---|---|
| Repo / PyPI package / Python import | `gridbook` |
| vLLM quantization registry key | `"gridbook"` — what every published artifact carries in `config.json` |
| Legacy registry key (still accepted) | `"prismaquant"` — artifacts exported before the rename |
| Custom-op namespace | `prismaquant::` |
| Environment variables | `PRISMAQUANT_*` |
| Log/warning prefix | `[prismaquant-cb]` |

The `prismaquant`-prefixed runtime names are kept because renaming them would
break compatibility with already-published artifacts and with in-flight A/B
tooling; a rename of the *non*-load-bearing ones (build-cache directory, log
prefix) is a pre-1.0 item on the [roadmap](ROADMAP.md). **PrismaQuant** is the
separate research pipeline that *produces* gridbook artifacts —
[github.com/RobTand/prismaquant](https://github.com/RobTand/prismaquant).

---

## Scope and honesty notes

- The plugin **does not patch vLLM core**. It does install a thin `load_weights`
  wrapper on specific *model classes* (HunYuan-V3, Laguna, Qwen3.5-MoE and their
  MTP drafters) whose loaders map MoE experts at the top level and would
  otherwise not recognise stacked codebook expert tensors. That wrap is inert for
  non-CB checkpoints. Nothing in `vllm/` is modified.
- This repository **serves** the format; it does not yet **produce** artifacts.
  A reference encoder is a roadmap item — until then, artifacts come from the
  authors' pipeline, and the [spec](docs/SPEC.md) is published so an encoder can
  be written independently.
- Every number in this repo comes from a measurement on the hardware named beside
  it. Where something is inferred rather than measured, it says so.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). The format specification in
[`docs/SPEC.md`](docs/SPEC.md) is published under the same license: the format is
free, open, and intended to be implemented by anyone, in any runtime, without
permission.

## Citation and attribution

Robert Tand — <robert.tand@icloud.com>. Machine-readable metadata in
[`CITATION.cff`](CITATION.cff).
