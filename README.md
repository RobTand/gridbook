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
| **GPU** | NVIDIA **Blackwell `sm_120` / `sm_121`** for the complete native path (GB10 / DGX Spark is the reference; RTX 5090 is user-reported). Some FP8-only paths may work on older NVIDIA cards; FP4-CB 0.5 serving does not — read the [compatibility table](#compatibility). |
| **CUDA toolchain** | `nvcc` on `PATH` **in the process that serves**, matching your torch build. Kernels are **JIT-compiled on first model load** (once per module, then cached), *not* at `pip install` time. `nvcc` 13.0 is the tested toolchain. A required extension that cannot be built is a serving error. |
| **PyTorch** | whatever build your vLLM uses (measured: `2.11.0+cu130`). |
| **vLLM** | already installed — gridbook is a plugin, not a runtime. Measured against `0.23.1rc1.dev764+g54b16d8a9`; see [compatibility](#compatibility). |
| **Parallelism** | **Dense CB Linears, dense FP8 source-passthrough Linears and mixed-format fused projections shard at `--tensor-parallel-size N`; routed CB MoE serves above one rank at `--tensor-parallel-size N --enable-expert-parallel`.** All load shard-aware with no exported-byte change; illegal boundaries are refused at construction (CB: whole packed groups; FP8 passthrough: whole 128-element source blocks per rank, per fused role). Grouped-BMM passthrough units shard only at a measured degree (1, 2, 4 at the DSv4 `wo_a` geometry, because sharding a grouped plane divides the kernel's group count). A mixed-format fused projection shards when every one of its roles does — the composer builds each carrier at that role's whole-tensor output size, so each role's own law decides. Routed CB MoE shards on the expert axis (whole experts per rank); `-tp N` without `--enable-expert-parallel` refuses and names the flag. Delegated groups, other passthrough formats and quantized embeddings still refuse above one. Measured two-node on 2026-08-23 (DeepSeek-V4 92 GB CB artifact across two DGX Sparks, KL within the CB kernel's nondeterminism floor); no cross-node speedup is claimed for CB artifacts. |
| **Python** | ≥ 3.10 (measured on 3.12). |

Gridbook has **no Triton dependency or serving lane**. Its production operators
are packaged native CUDA support kernels and CUTLASS GEMM/grouped-GEMM kernels;
if the native kernel required by an artifact, shape, or GPU cannot be loaded,
serving fails closed with the missing operation and build guidance. It does not
continue on a slower implementation. That guarantee is scoped to Gridbook's own
operators: a vLLM installation may still load and run Triton for components
Gridbook does not own — its attention backend, and under `VLLM_COMPILE` the
Inductor-lowered *surrounding* model graph, which Gridbook's opaque op bodies
are excluded from by design. Inside that lane the claim is enforced rather than
asserted: a static ratchet scans the package, `scripts/`, and `tests/` for any
Triton-reaching import; a GPU-lane check proves that executing a Gridbook op
imports no Triton module vLLM had not already loaded; and a model-load preflight
fails closed if a delegated `compressed-tensors` group resolves to a
Triton-backed backend or to one that would silently drop its declared activation
scales ([`tests/test_no_triton_runtime.py`](tests/test_no_triton_runtime.py),
[`gridbook/delegated_preflight.py`](gridbook/delegated_preflight.py)).
The old Triton prototype measured **4.20
tok/s** on the 27B where the native CUDA GEMV measured **10.28 tok/s**; that is
historical evidence from the retired path, not a selectable fallback or a
current benchmark. See
[TROUBLESHOOTING](docs/TROUBLESHOOTING.md#the-native-extension-did-not-load)
for build failures.

The serving boundary is permanently an opaque registered custom op. For FP8
transient paths Gridbook calls vLLM's registered native CUDA quantizer and
CUTLASS scaled-matmul operators directly, after checking their ABI and shape
contract; it does not call the fallback-capable `vllm._custom_ops` wrappers.
MoE activations likewise call registered `torch.ops._C` CUDA operators directly;
the vLLM helper with a Triton SWIGLUSTEP branch is not part of Gridbook dispatch.

---

## Install

Install **into the environment that already has vLLM and torch** — gridbook does
not pin either.

```bash
pip install gridbook
```

PyPI is the recommended stable install route. To test the current development
head instead, install directly from GitHub:

```bash
pip install git+https://github.com/RobTand/gridbook
```

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
pip install gridbook

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

**How to tell it is actually working.** Native-kernel availability is a serving
contract, so a missing required extension is explicit:

- **Bad** — a `[prismaquant-cb] ERROR:` or `WARNING:` line explicitly saying a
  **required** native operation is unavailable means Gridbook cannot serve that
  operation; the subsequent load/forward fails closed rather than selecting
  another backend. A shape-specialized optimization may be unavailable only
  when its diagnostic names a separately qualified native CUDA/CUTLASS route.
  (Grep your vLLM log for `[prismaquant-cb]`, not for `gridbook` — the runtime
  log prefix still uses the project's older name.)
- **Good** — no such line, and the first model load pauses while the JIT builds
  run, then does not pause again. Each native module is built and cached
  independently in its own subdirectory of the build cache, so what you pay is
  one build per module that load actually reaches — a cold container with an
  ephemeral cache pays them all; a persistent `PRISMAQUANT_CB_EXT_DIR` pays
  them once ever on that machine.
- A definitive check that does not require serving anything is in
  [`docs/INSTALL.md`](docs/INSTALL.md#verify-the-install).

Notes:

- **`--enforce-eager` remains the published 27B configuration.** The permanent
  opaque dispatch makes a mode-0 `FULL_DECODE_ONLY` graph a supported
  optimization candidate; on the dated close-rate 0.6B canary it improved
  latency by 20.1% and
  replayed changed inputs exactly. Keep eager for the quickstart until the same
  graph configuration clears the 27B streaming gate. Details and the exact
  flags: [`KERNELS.md`](docs/KERNELS.md#cuda-graph-safety-rules).
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

The FP4 native path is Blackwell-targeted; dense FP8-CB also has an Ada route.
The main decode translation unit is not arch-specific, but that alone does not
make an FP4-CB artifact portable:
every FP4-v2 quality path uses the v2 exact expander, whose device prepare
currently admits only CUDA cc 12.0/12.1. The owned grouped-BF16 CUTLASS GEMM is
SM80-compatible in isolation; the required expander sets the full FP4 serving
floor. This table separates what was **measured** from what is **inferred from
the code and untested**.

| GPU class | Dense small-M (M ≤ 8) | FP8-CB M = 9–128 | Native general / FP4 quality path | Verdict |
|---|---|---|---|---|
| **GB10 / DGX Spark, `sm_121`** | native CUDA GEMV | fused CUTLASS when eligible; otherwise CUDA expand → CUTLASS W8A8 | FP8: CUDA expand → CUTLASS; FP4 M>8: native BF16 expand → Gridbook CUTLASS grouped GEMM (`E=1`); fused native-NVFP4 remains opt-in | **MEASURED target.** Published results below predate the final native-only dispatch and remain tied to their recorded commits. |
| **RTX 50 family, `sm_120`** | public NVFP4 rows cover K12..K24; FP8-CB readers cover K28..K48 and producer-facing rows cover K40/K44/K48; wider generic bindings are research only | optimized subsets where independently eligible; otherwise exact expansion | the generic modules cross-compile as exact `sm_120` SASS, persistent-B/BF16 bridge as `sm_120a`, and the native vLLM FP8/CUTLASS ABI resolves without execution | **BOTH CB FAMILIES ARE COMPILE-ONLY ON RTX 50, NOT DEVICE-QUALIFIED.** Values outside the public reader sets remain unsupported artifacts despite retained low-level kernels. A 5090 user reported an older 27B path working ([issue #1](https://github.com/RobTand/gridbook/issues/1)); smaller cards have no physical correctness, graph, memory, or speed receipt yet. Contract v11 keeps all SM120 cells `compile_only`; dense NVFP4 expansion is a fallback, while dense FP8 expansion plus native CUTLASS W8A8 is structurally backed. |
| **RTX 4090 / L40S, `sm_89`** | FP8-CB CUDA GEMV cross-compiles; an FP4-CB layer is rejected at weight load | Blackwell fused kernel is load-time auto-off; CUDA expand → native CUTLASS W8A8 is the intended route | FP8 main extension + direct CUTLASS ABI cross-compile/registration preflight passed; FP4 has no SM89 activation registration and v2 expander prepare rejects the device | **FP8 COMPILE-ONLY, NOT DEVICE-QUALIFIED.** No physical Ada correctness, graph, memory or speed gate has run; contract v11 cells remain `compile_only`. NVFP4-CB is wholly unsupported. |
| **H100, `sm_90`** | FP8-CB decode is expected to work; an FP4-CB layer is rejected at weight load | Blackwell fused kernel ineligible; CUDA expand → CUTLASS expected for FP8 | FP8 native expansion + CUTLASS expected; FP4 unavailable because v2 expander prepare rejects the device | **FP8-ONLY IS INFERRED / UNTESTED.** No SM90 cross-compile or device gate is claimed. FP4-CB is unsupported. |
| **A100 `sm_80`** | no complete production lane: FP8 prefill needs `sm_89+`, and FP4 load requires cc 12.0/12.1 | FP8-CB rejected | grouped BF16 GEMM supports SM80, but the required FP4-v2 expander rejects the device | **UNSUPPORTED FOR PRODUCTION CB SERVING IN 0.5.** No slow fallback is selected. |
| **Supported NVIDIA GPU, no `nvcc`** | unavailable | unavailable | unavailable | **FAILS CLOSED.** Prebuild and package compatible native extensions, or provide `nvcc` in the serving environment. |
| **Non-NVIDIA** | unsupported | unsupported | unsupported | **UNSUPPORTED / UNQUALIFIED.** The canceled ROCm prototype is not shipped or dispatched. |

Tested software stack (verified in-container, 2026-07-28): vLLM
`0.23.1rc1.dev764+g54b16d8a9`, torch `2.11.0+cu130`, transformers `5.13.0`,
CUDA `13.0.88`, Python `3.12.3`, **arm64**. The image also contained Triton as
a vLLM dependency, but Gridbook neither depends on nor dispatches it. vLLM `0.25.1`
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
   CUTLASS. FP8 quantization/scaled GEMM use the directly registered native CUDA
   operators; FP4's quality path uses Gridbook's owned grouped-BF16 CUTLASS
   bridge.

Formats mix per-Linear with plain NVFP4, FP8 and BF16 inside one standard
`safetensors` checkpoint. The 0.5 native dense serving gate is deliberately
narrower than the format spec: CB Linears must be biasless, and FP4 must be an
unsigned product rung with v2 scale coding (the signed S-rung family was
deleted from the runtime on 2026-08-23). A non-`None` bias or FP4-v1 dense
layer is never routed through an unowned framework operation: the public dense
method rejects bias, and model load rejects the unsupported FP4-v1 family.
The unsigned NVFP4 public domain is every integer K12..K24: readers, producers,
artifact choosers, and SM120 lane rows use the same closed set. FP8-CB readers
retain every integer K28..K48 for historical artifacts, while new producers
emit only K40/K44/K48. The broader K1..K25 NVFP4 and K4..K48/4 FP8 menus were
pre-release candidates and were retracted before the planned 0.9.1 release.
Generic CUDA bindings retain NVFP4 K1..K32 and aligned low FP8 calls solely for kernel
research; they do not make those values valid artifacts. Each optimized fused
lane retains its own evidence-backed reader eligibility, including historical
FP8 K28/K32/K36.
See [`docs/MOTIVATION.md`](docs/MOTIVATION.md) for the
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
- **MoE prefill is native and fail-closed.** `M≤16` uses the owned grouped CUDA
  GEMV; above 16, FP8-CB uses a quality-green fused CUTLASS path when eligible
  and otherwise exact BF16 expansion + the owned CUTLASS grouped bridge, while
  FP4-CB uses that exact quality bridge directly. The predecessor CUDA
  chunk-expander path measured 293 → **1,821 tok/s** at 8k and 207 → **1,822
  tok/s** at 63k on Laguna-S-2.1 (commit `8829c16`); those historical numbers
  are not measurements of the new owned grouped bridge. The bridge's current
  generic SM80-compatible schedule is not Blackwell-optimized: on synthetic
  DSV4 shapes it measured **6–17% slower** than segmented BF16 matmuls at warm
  steady state. It is currently a quality/native-ownership result, not a speed
  claim. The v0.4.2 fused native-NVFP4 MoE route remains an explicit opt-in;
  its published evidence boundary does not qualify it as the production path.
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
| [`docs/KERNELS.md`](docs/KERNELS.md) | Serving kernel design: native transient-expand prefill, fused act-QDQ decode GEMV, two-tier in-register scale compose, CUTLASS GEMM/grouped GEMM, fail-closed dispatch, CUDA-graph rules. |
| [`docs/QTIP-NATIVE-NVFP4-RESEARCH.md`](docs/QTIP-NATIVE-NVFP4-RESEARCH.md) | Research-only ABI for deterministic online sign/Hadamard transforms around the native E2M1 trellis W4A4 lane. |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | Operator reference for the plugin itself: dispatch, kernel set, environment switches. |
| [`docs/DELEGATED-NVFP4-MOE.md`](docs/DELEGATED-NVFP4-MOE.md) | Version-scoped backend selection for a **non-CB** NVFP4 MoE group on GB10 (`sm_121`), including Marlin's generic weight-only warning and a fail/unknown preflight policy. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | The measured results with full hardware/protocol context and caveats. |
| [`ROADMAP.md`](ROADMAP.md) | Canonical kernel TODO, completed work, and measured-negative experiments. |
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
  MTP drafters, DeepSeek-V4) whose loaders map MoE experts at the top level and would
  otherwise not recognise stacked codebook expert tensors. That wrap is inert for
  non-CB checkpoints. Shared-CB projections, including MTP-nested shared
  experts, are aliased to native CB Linears; resolving one to a plain BF16
  Linear is a load error, not a compatibility fallback. Nothing in `vllm/` is
  modified.
- This repository **serves** the format; it does not produce artifacts.
  [PrismaQuant](https://github.com/RobTand/prismaquant) is the one canonical
  producer, while the [spec](docs/SPEC.md) and conformance fixtures keep the
  format independently implementable without maintaining a second encoder here.
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
