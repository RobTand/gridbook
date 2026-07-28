# Benchmarks

All results are from a **single NVIDIA GB10 / DGX Spark** (Blackwell `sm_121`,
128 GB unified memory, ~273 GB/s), serving through vLLM with `--enforce-eager`.
Read the [caveats](#caveats--read-these) — these are single-box, single-seed
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
FP8-CB codebook rungs** and **zero** stock NVFP4/FP8 — the codebook formats won
every Linear on cost.

> **Which build is this?** The A/B in this section is the **2026-07-18** run,
> whose CB arm was allocated from a 4-rung menu (`FP8_CB_K36/K40/K44/K48`, 386 CB
> Linears). The artifact **published on the Hub** is a later 8-rung-ladder build
> (`K36`/`K40`/`K41`/`K43`/`K44`/`K45`/`K46`/`K47` — 427 CB targets, plus 110
> NVFP4 vision-tower targets; read from the shipped `quant_config.json`) measured
> in a **different session** with **different absolute KL**. See [two sessions,
> two builds](#two-sessions-two-builds-why-the-model-card-says-77-and-this-page-says-583)
> below before comparing this table with the model card.

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

### Two sessions, two builds: why the model card says 77% and this page says 58.3%

The [published 27B model
card](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm)
reports **ALL-KL 0.0049 / confident-KL 0.00295** and a **−77%** headline. This
page reports 0.0134 / 0.01134 and −58.3%. Both are real; they are not the same
measurement, and neither is a correction of the other.

| | build | eval session | CB ALL-KL | CB conf-KL | NVFP4/FP8 baseline ALL-KL | baseline conf-KL | relative |
|---|---|---|---|---|---|---|---|
| This page | 4-rung menu, 386 CB Linears | 2026-07-18 | 0.0134 | 0.01134 | 0.0321 | 0.02407 | **−58.3% / −52.9%** |
| Model card | 8-rung ladder (**the published file**) | 2026-07-22 | 0.0049 | 0.00295 | 0.0211 | 0.01302 | **−76.8% / −77.3%** |

Two things move between the rows, and only one of them is the format:

1. **Different artifacts.** The ladder build spends the same byte budget across
   eight rungs instead of four, so it is expected to be the better artifact.
2. **Different evaluation sessions, which shift the whole scale.** The *same
   unchanged* NVFP4/FP8 baseline artifact reads confident-KL **0.02407** in the
   2026-07-18 session and **0.01302** in the 2026-07-22 session — a 1.85× move
   with no change to any served byte. Absolute KL from one session is therefore
   **not** comparable to absolute KL from another; only within-session
   comparisons are. The ±17% extension-residency effect above is one identified
   contributing mechanism and does not account for the whole move; each session
   also took its own BF16 teacher dump, which every arm in that session is then
   scored against. The full mechanism is **not** fully characterised, which is
   exactly why cross-session absolute numbers are treated as incomparable rather
   than rescaled.

Both sessions are internally consistent — one corpus, one protocol, every arm
scored against a single BF16 teacher capture taken in that session — and both
return the same verdict at matched bytes. This page quotes the more conservative
one. The card's row was re-derived for this note by re-running the comparison
over that session's stored top-20 dumps; all four arms reproduce exactly.

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
- **Prefill (updated 2026-07-23): solved.** The TTFT above was measured against
  the old correctness-path per-expert prefill loop (~71k launches per 1400-token
  prefill) and **no longer describes the shipping default**. A CUDA chunk-expander
  feeding vLLM's own fused-MoE grouped kernel replaced it; measured on
  Laguna-S-2.1 (117B MoE): **293 → 1,821 tok/s at 8k** and **207 → 1,822 tok/s at
  63k** (commit `8829c16`). The current default is `auto` — a measured per-layer
  selection over the candidate prefill paths. The 35B TTFT row has not been
  re-measured on the new path.

---

## 295B MoE (Tencent-Hunyuan-class) — serves on ONE box; the prefill thesis

A 295B Mixture-of-Experts model (80 layers, 192 experts, top-8, ~21B active),
quantized at **2.9 bpp** (achieved 2.902) with a mixed two-tier FP4-CB + FP8-CB
body.

> **NO QUALITY-VS-TEACHER CLAIM.** A 295B BF16 reference cannot be run on a single
> box to compute KL against it, so there is **no KL or PPL-vs-BF16 number** at this
> scale. What is validated is: it loads, fits, serves, generates coherent and
> correct output, and its serving speed against the equivalent GGUF build.

**Footprint.** The **published** joint-menu artifact is **105.7 GB** (98.5 GiB)
across 5 shards — the figure to plan against, read from the Hugging Face file
listing. The composition breakdown below is from the earlier single-file body
(110.3 GB / 102.7 GiB): 99.68 GB packed `cb_qweight` + 10.52 GB BF16 sidecars (57
non-CB layers) + 0.10 GB FP32 scales. In both, the 2.902 bpp figure is over
quantizable params; the BF16 floor is bpp-excluded but disk-resident, which is why
the total exceeds `bpp × params`.

**Single-Spark fit:** ~105–110 GB weights + ~2 GB framework on the ~121 GB usable
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
| **gridbook 2.9 bpp (native tensor core, final)** | **~109** | 14.6 base / **16.1 with MTP spec** | **88 (130/148)*** |

\* The final joint-menu artifact scored 88 (130/148) on the ship config —
the highest of any Hy3 artifact on this box, and strictly above the shipped
GGUF IQ's 129/148 on the same seed/protocol. Caveat, stated plainly: across
serving configs of the earlier body, the same bytes measured 85–87 (scenario
churn at the capability plateau; the GGUF family's own band is 86–87), so a
single +1-to-+3 point read is at the top of the band, not a proven
directional gap. The joint artifact does carry genuinely more fidelity in
sensitive places (36 vanilla-FP8 Linears, fp8-K28 layer-21 experts, an
FP8-CB K44 MTP draft).

- **Prefill ~109 tok/s vs the GGUF IQ build's 42 — ~2.6× faster.** This is the
  format's reason to exist, proven at 300B class: the native-tile design removes
  the prefill dequant tax IQ pays on general CUDA cores.
- **Base decode still *trails* the GGUF build — honestly.** After three kernel
  rounds (dense FP4-v2 CUDA GEMV; a +50% w2 grouped schedule; double-buffered
  fp8 dense), base batch-1 decode is 14.6 vs the GGUF build's ~18. The wall is
  measured, not guessed: the fp4-CB decode chain is **compute-bound at GEMV
  shapes** (ncu: SM 71% vs memory 44%) under the bit-exact decode contract —
  not a bandwidth problem better staging can fix. **MTP speculative decoding
  (which GGUF cannot carry) closes most of the gap on natural text: 16.1
  tok/s** with the FP8-CB K44 draft at k=1, and becomes a straight multiplier
  once vLLM captures drafter CUDA graphs.
- The joint-menu allocation experiment is also worth citing: offered vanilla
  NVFP4 and FP8 alongside the codebook rungs, the measured allocator gave 36
  Linears to vanilla FP8 and **zero to vanilla NVFP4** — at matched bits the
  codebook dominates the fixed grid on every unit of a 295B.
- **ToolEvalBench: the shipped joint-menu artifact scored 88/100 (130/148)** on
  the ship config, against the GGUF IQ build's 87 (129/148) and k-quant's 86
  (128/148), under the identical protocol below. Zero errors, all 74 scenarios
  ran. **Read it as parity, not a win:** earlier bodies of the same bytes measured
  85–87 across serving configs and the GGUF family's own band is 86–87, so +1 sits
  inside the churn band. At matched body bytes tool-use quality is
  **base-model-dominated.** The case for the CB build over the GGUF one is:
  **same quality band, ~2.6× prefill, native kernels, and it can additionally
  carry a multi-token-prediction draft head the GGUF build cannot.**

TEB protocol: 12288 context, FP8 KV cache, model-specific tool parsers, seed 1234,
`--no-think --hardmode --parallel 1`. Failures are the known cross-artifact
family failures shared by all three builds — scenario churn at the quality plateau.

---

## What the fallback costs

If `nvcc` is missing, the JIT build fails, or `PRISMAQUANT_CB_DECODE=triton` is
set, the plugin keeps serving **correct** output on the pre-CUDA code paths.
**There is no fresh benchmark of that configuration.** What there is:

**Dense decode — measured, and the fallback code is unchanged.** On the 27B at
5.5 bpp, the Triton decode-GEMM measured **4.20 tok/s** and the CUDA GEMV that
replaced it measures **10.27–10.30 tok/s**. That Triton path is still what a
dense CB Linear executes when `_cuda_gemv_ok()` is false — the gate is
`PRISMAQUANT_CB_DECODE == "cuda"` **and** `get_ext() is not None`
(`gridbook/linear.py`), and the miss falls through to the same `cb_gemm` call.
The 4.20 figure was taken on the whole Triton-prototype configuration, which
also predates the fp8-direct expander, so treat it as the pessimistic end.

**MoE decode — the grouped CUDA GEMV is skipped entirely; the magnitude depends
on the artifact.** `_cuda_moe_ok()` in `gridbook/moe.py` returns `False` when
`get_ext() is None`, and the call falls through to the prefill-family paths:

- **fp4-CB artifacts** (the 295B) default to `PRISMAQUANT_CB_PREFILL=loop` —
  literally the per-expert transient loop that measured **3.52 tok/s** decoding
  the 35B before the grouped kernel replaced it (**32.6–33.3 tok/s** after). That
  loop costs ~10k host syncs/launches per token; it is a launch-storm, not a
  bandwidth problem.
- **fp8-CB artifacts** default to `auto`, a different (batched-expand) path.
  Expect a large regression — the one-launch-per-projection grouped GEMV is gone
  — but **its size has not been measured**, and the 3.52 figure above does *not*
  describe it.

Prefill degrades too and is not separately quantified for the fallback.

Sources: 27B dense before/after and 35B MoE before/after are the same served A/B
runs as the tables above.

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
  and MoE). Large-M dense **prefill is not yet at native parity**: ~1.44× on the
  27B (traffic-bound transient expand). MoE prefill was the worst gap and is now
  the default CUDA chunk-expander path (Laguna 117B: 293 → 1,821 tok/s at 8k) —
  the 35B TTFT row above predates it. The GGUF prefill comparison (the 295B
  ~2.6×) is against a *different* serving stack (llama.cpp CUDA-core IQ dequant),
  and is the clean "why native tiles win" number; it is not a claim of parity
  with vLLM's own native NVFP4/FP8 GEMM at large M.
- **bpp vs disk size differ by the BF16 floor.** Reported bpp is over quantizable
  params; embeddings/`lm_head`/norms/non-CB Linears are excluded from bpp but
  resident on disk, so `bpp × params` understates the artifact size.
- **Blackwell is the only measured target.** Every number on this page is from
  `sm_121`. The decode kernel itself carries no architecture guards and is
  *expected* to run from `sm_80` up, and only the mid-M fused prefill kernel is
  genuinely `sm_120`-family bound — but that is inferred from the code, not
  measured. Per-GPU expectations, with measured/inferred labelled, are in
  [`INSTALL.md`](INSTALL.md#hardware-matrix). Without `nvcc` the Triton fallback
  runs everywhere (correct, not fast).
- **One published artifact per headline is not the same as three.** The 27B and
  295B artifacts are downloadable; **the 35B MoE artifact is not published** — its
  numbers are an internal measurement showing the format win reproduces on MoE,
  not something you can reproduce without an encoder.
