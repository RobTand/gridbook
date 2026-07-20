# Benchmarks

All results are from a **single NVIDIA GB10 / DGX Spark** (Blackwell `sm_121`,
128 GB unified memory, ~273 GB/s), serving through vLLM with `--enforce-eager`.
Read the [caveats](#caveats-read-these) — these are single-box, single-seed
measurements, and the 295B result carries **no quality-vs-teacher claim**.

## What is being compared, and how

- **Matched-bpp baseline.** Every quality comparison is against an artifact of the
  **same model at the same bits-per-weight**, quantized to stock per-Linear
  **NVFP4/FP8** and served natively by vLLM's `compressed-tensors` path. This
  isolates the one variable that matters: the *same byte budget* spent on codebook
  formats versus scalar formats. (It is **not** a comparison to BF16 or to a
  smaller/larger model.)
- **KL-vs-BF16.** Exact full-vocab (top-20) KL divergence between the quantized
  model's next-token distribution and the BF16 model's, on held-out WikiText
  (8176 positions), against the *same* BF16 reference dump per model. Lower is
  closer to the original model.
  - **ALL-KL** averages over all positions.
  - **confident-KL** averages over the positions where the BF16 reference is
    confident (places high mass on its top token) — the decision-relevant
    positions, where a probability shift is most likely to change an output.
- **top-1 agreement** = fraction of positions where the quantized model's argmax
  matches BF16's.
- **PPL / NLL** = direct perplexity / actual-token negative log-likelihood on the
  served artifact.
- **TTFT(1400)** = time to first token on a 1400-token prompt (prefill latency).
  **decode tok/s** = steady-state single-stream generation throughput.
- **ToolEvalBench (TEB)** = a tool-use / function-calling fidelity benchmark scored
  out of 100 (148 scenarios); used only where the base model supports tool calls.

Bits-per-weight (bpp) is reported over **quantizable parameters** (excludes
embeddings, `lm_head`, norms, and other non-CB tensors). The on-disk artifact is
larger than `bpp × params` because that excluded BF16 floor is still resident on
disk — noted per model below.

---

## 27B (Qwen3-class hybrid) — validated quality win at 5.5 bpp

A 27B hybrid model (attention + gated linear-attention, with a vision tower),
quantized at **5.5 bpp**. The allocator placed the **entire quantizable body on
FP8-CB codebook rungs** (a mix of `FP8_CB_K36/K40/K44/K48`) and **zero** stock
NVFP4/FP8 — the codebook formats won every Linear on cost.

**Matched-bpp denominator (from the safetensors headers):** the CB body is
**16.713 GB**, the NVFP4/FP8 baseline body is **16.707 GB** — a 0.04% difference.
This is a genuine matched-bytes comparison.

| | confident-KL | ALL-KL | conf top-1 | ALL top-1 | PPL | NLL |
|---|---|---|---|---|---|---|
| BF16 (reference) | — | — | — | — | 9.123 | 2.2108 |
| NVFP4/FP8 baseline | 0.02407 | 0.0321 | 98.77% | 92.3% | 9.251 | 2.2247 |
| **CB (ours)** | **0.01134** | **0.0134** | **99.56%** | **95.4%** | **9.166** | **2.2155** |

- **ALL-KL −58.3%**, confident-KL **−52.9%**, at matched bpp.
- top-1 agreement higher on both slices.
- PPL gap to BF16 is **~3× smaller** (CB +0.043 vs baseline +0.128).

The KL delta is far larger than the single-seed calibration noise band (~±10-40%
at this calibration size) and is corroborated by PPL, so the win is robust. A 0.04%
body-byte difference cannot explain a 58% KL move.

**Speed** (after the CUDA decode + expander kernels; same artifact, same harness):

| | TTFT(1400) | decode tok/s |
|---|---|---|
| BF16 | 1.269 s | 4.59 |
| NVFP4/FP8 baseline (native) | **0.746 s** | 10.26 |
| **CB (ours)** | 1.075 s | **10.27-10.30** |

- **Decode is at/above native parity** (10.3 vs 10.26). At matched body bytes,
  parity is the ceiling for a bandwidth-bound decode — reached.
- **Prefill is ~1.44× the native baseline** (1.075 s vs 0.746 s). The residual is
  the transient-expand path's doubled memory traffic; closing it is the
  fused-prologue / persistent-N kernel work in [`KERNELS.md`](KERNELS.md). (The
  initial Triton prototype was 1.62 s / 2.2×; the CUDA kernels closed most of it.)

**Measurement-arithmetic caveat.** Depending on whether an extra CUDA extension is
resident in the eval process, confident-KL reads **0.01134 or 0.01328** on this
artifact — a ±17% swing from allocator-address-induced reduction-order drift, not
from the CB kernels (both prefill paths are bit-identical offline). Under **either**
reading the verdict is unchanged: −45% to −53% confident-KL, −56% to −58% ALL-KL,
PPL gap 2-3× smaller. See [`KERNELS.md`](KERNELS.md#a-measurement-side-effect-worth-knowing).

---

## 35B MoE (Qwen3-MoE-class) — the win reproduces on Mixture-of-Experts

A 35B Mixture-of-Experts model (256 experts, top-8 routing, ~3B active per token,
hybrid linear-attention), quantized at **4.75 bpp** (achieved 4.758; 22 GB
artifact). Again the body went **all-CB** across six FP8-CB rungs, zero stock
formats on the experts.

**Gold metric** (vs the same BF16 dump, 8176 positions, top-20):

| | confident-KL | ALL-KL | conf top-1 | ALL top-1 | PPL |
|---|---|---|---|---|---|
| BF16 (reference) | — | — | — | — | 9.437 |
| NVFP4/FP8 baseline (native) | 0.03625 | 0.0492 | 98.44% | 89.9% | 9.587 |
| **CB (ours)** | **0.01706** | **0.0278** | **99.05%** | **92.6%** | **9.542** |

- **confident-KL −53%**, **ALL-KL −43%**, at matched bpp; top-1 better on both
  slices; PPL gap to BF16 ~30% smaller than the baseline's.

**Speed:**

| | TTFT(1400) | decode tok/s |
|---|---|---|
| BF16 | 0.484 s | 28.43 |
| NVFP4/FP8 baseline (native) | **0.325 s** | ~35.9 |
| **CB (ours)** | 3.46 s | **32.6-33.3** |

- **Decode: solved.** The grouped `(token, expert)` GEMV (one launch per
  projection, replacing ~10k host operations per token) took decode from 3.5 to
  ~33 tok/s — **faster than BF16**, within 8% of the native baseline, at 3× smaller
  and −43% ALL-KL.
- **Prefill is not yet solved for MoE:** TTFT is dominated by a per-expert launch
  storm in the correctness-path prefill loop (~71k launches per 1400-token
  prefill). A batched-expert expand + grouped GEMM is the remaining work; see
  [`KERNELS.md`](KERNELS.md#moe-path-grouped-token-expert-gemv).

---

## 295B MoE (Tencent-Hunyuan-class) — serves on ONE box; the prefill thesis

A 295B Mixture-of-Experts model (80 layers, 192 experts, top-8, ~21B active),
quantized at **2.9 bpp** (achieved 2.902) with a mixed two-tier FP4-CB + FP8-CB
body.

> **NO QUALITY-VS-TEACHER CLAIM.** A 295B BF16 reference cannot be run on a single
> box to compute KL against it, so there is **no KL or PPL-vs-BF16 number** at this
> scale. What is validated is: it loads, fits, serves, generates coherent and
> correct output, and its serving speed against the equivalent GGUF build.

**Footprint (read from the artifact header):** `model.safetensors` = **110.3 GB**
(102.7 GiB) = 99.68 GB packed `cb_qweight` + 10.52 GB BF16 sidecars (57 non-CB
layers) + 0.10 GB FP32 scales. The 2.902 bpp figure is over quantizable params; the
BF16 floor is bpp-excluded but disk-resident, which is why the total is 110 GB.

**Single-Spark fit:** 110.3 GB weights + ~2 GB framework on the ~121 GB usable
pool. Because CB decode is transient (no load-time expansion, INV-1), resident
weight footprint == disk. Serve with `--max-model-len 8192-16384`; the demo used
4096. Load 77 s; 44,272 tokens of KV at 4k context (10.8× concurrency). Native
262k context does not fit (true of any ~110 GB weight class on this box).

**Validation bar met** (no quality claims): coherent, factually and arithmetically
correct generation (e.g. `17×24=408`, `60 mi / 1.5 h = 40 mph`, `60 mi / 2 gal =
30 mpg`; capital-city chains; a correct recursive Fibonacci; RGB primaries).

**The speed thesis — prefill vs the GGUF IQ build at matched bytes:**

| 295B artifact @ ~2.8-2.9 bpp | prefill tok/s | decode tok/s | TEB |
|---|---|---|---|
| GGUF IQ 2.8 bpp (CUDA-core dequant) | 42 | ~17-18 | 87 (129/148) |
| GGUF k-quant | — | ~18-19 | 86 |
| **CB 2.9 bpp (ours, native tensor core)** | **89** | ~9-13 | **87 (129/148)** |

- **Prefill 89 tok/s vs the GGUF IQ build's 42 — ~2.1× faster.** This is the CB
  lane's reason to exist, proven at 300B class: the native-tile design removes the
  prefill dequant tax that IQ pays on general CUDA cores. (Subsequent perf levers
  pushed CB prefill toward ~115 and decode toward ~13 tok/s; the 2.1× figure is the
  locked serve-verdict number and the conservative one to cite.)
- **Decode currently *trails* the GGUF build here — honestly.** CB decode is
  ~10 tok/s (toward ~13 with levers) against the GGUF build's ~18. The dense FP4
  two-tier path still decodes through Triton on this artifact; a dense FP4-v2 CUDA
  decode kernel (the MoE routed-expert grouped kernel already exists) is the pending
  lever. So the 295B advantage is **prefill + single-box fit + MTP-head capability**,
  not decode. Decode-at-parity comes with that kernel.
- **ToolEvalBench 87/100 (129/148) — an exact tie with the GGUF IQ build** (87,
  129/148) and above k-quant (86), under the identical protocol below. Zero errors,
  all 74 scenarios ran. Honest reading: at matched body bytes, tool-use quality is
  **base-model-dominated — this is parity, not a quality win over IQ.** The
  case for the CB build over the GGUF one is: **same quality, ~2.1× prefill, native
  kernels, and it can additionally carry a BF16 multi-token-prediction head the GGUF
  build cannot.**

TEB protocol: 12288 context, FP8 KV cache, model-specific tool parsers, seed 1234,
`--no-think --hardmode --parallel 1`. Failures are the known cross-artifact
family failures shared by all three builds — scenario churn at the quality plateau.

---

## Caveats — read these

- **Single box, single seed.** All numbers are one DGX Spark, one calibration
  draw. Single-seed KL at this calibration size has a ~±10-40% noise band; the
  headline wins (−58% / −53%) are far outside it and corroborated by PPL/top-1, but
  smaller deltas in these tables should not be over-read.
- **Measurement-arithmetic sensitivity (~±17%).** Loading extra CUDA extensions
  shifts allocator addresses and perturbs FP reduction order, moving the *measured*
  KL even when the served bytes are identical. A/B arms must match extension
  residency. This is why the 27B confident-KL is quoted as a range under both
  readings.
- **The 295B has no quality-vs-teacher number** — only load/fit/coherence, TEB
  parity vs GGUF, and speed. Do not infer a KL or PPL win at that scale.
- **TEB is base-model-dominated at matched bytes.** Tool-use parity across
  quantizations of the same base model is the expected result; treat TEB as a "does
  not regress" gate, not a quantization-quality discriminator.
- **Speed status is honest and uneven.** Decode is at/above native parity (dense
  and MoE). Large-M **prefill is not yet at native parity**: ~1.44× on the 27B
  (traffic-bound transient expand) and materially worse on MoE (per-expert prefill
  loop). The GGUF prefill comparison (the 295B 2.1×) is against a *different*
  serving stack (llama.cpp CUDA-core IQ dequant), and is the clean "why native
  tiles win" number; it is not a claim of parity with vLLM's own native NVFP4/FP8
  GEMM at large M.
- **bpp vs disk size differ by the BF16 floor.** Reported bpp is over quantizable
  params; embeddings/`lm_head`/norms/non-CB Linears are excluded from bpp but
  resident on disk, so `bpp × params` understates the artifact size.
- **Blackwell-only.** The native-speed path targets `sm_120/121`. On other GPUs the
  Triton fallback runs (correct, not fast, not INV-2-eligible).
