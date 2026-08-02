# Benchmarks

For new native-performance work, use the reproducible streaming
[native-parity protocol](NATIVE-PARITY.md). It fixes request shapes and seeds,
runs independent blocks through `vllm bench serve`, and records the artifact,
image, dispatch, software, and server provenance alongside true TTFT/TPOT/ITL/
E2EL metrics.

All results are from a **single NVIDIA GB10 / DGX Spark** (Blackwell `sm_121`,
128 GB unified memory, ~273 GB/s). Published-model rows use vLLM with
`--enforce-eager`; the explicitly labelled CUDA-graph canary does not. Read the
[caveats](#caveats--read-these) — these are single-box, single-seed
measurements, and the 295B result carries **no quality-vs-teacher claim**.

## 2026-07-31 CUDA-graph canary — close-rate, not formal parity

This is historical evidence from the commit named below. Its then-new opaque
per-layer arm fixed the still-earlier host-branch graph-capture failure; current
Gridbook makes that opaque boundary unconditional and has removed the old
branch. On a Qwen3 0.6B iteration pair, vLLM `FULL_DECODE_ONLY` with compilation
mode 0 and capture sizes `[1,2,4,8]` measured:

Execution identity: GB10 `sm_121`, driver 595.84, TP=1, Gridbook 0.3.0 at
`aea57dcb`, vLLM `0.23.1rc1.dev764+g54b16d8a9` / Torch 2.11.0 / CUDA 13.0 in
image digest `sha256:d0840ff0e0ba1899a51bf4cb473f43d0c765288b8de708080ad9d95768615141`.
The Gridbook arm is `FP8_CB_K36` (W8A8), using that commit's opaque arm, with no
CUDA extension-build warning; the native arm is compressed-tensors NVFP4
(W4A4). The concrete native Linear backend and per-kernel fallback trace were
not retained. This run also predates Gridbook's direct registered-operator FP8
binding, which now bypasses the fallback-capable `vllm._custom_ops` wrapper.
Those limits make this a historical graph-capture canary, not release evidence
for the current operator stack.

| Workload | Gridbook | native | Gridbook / native |
|---|---:|---:|---:|
| 1400 input + 1 output, eager | 29.289 ms | 24.777 ms | 1.182× latency |
| 32 input + 256 output, eager | 1.4443 s | 2.8883 s | 0.500× latency |
| 32 input + 256 output, full-decode graph | 1.1534 s | 1.0895 s | **1.059× latency** |

The graph removes **20.1%** from Gridbook's end-to-end 32+256 latency and puts
it within **5.9%** of native in this canary. Four changed batch-1 prompts and a
four-request batch produced exactly equal text, token sequences, and per-token
logprobs between eager and graph replay (maximum absolute logprob difference
0). Capture took about 30 seconds and reported 2.64 GiB for sizes 1/2/4/8 in
the validation process.

This is deliberately **not** an exact-rate parity claim. The native model
payload is 870,290,032 bytes; the Gridbook model plus codebook sidecar is
871,628,664 bytes, **1,338,632 bytes (0.154%) larger**. It also compares the
whole execution contracts: native NVFP4 is W4A4, while `FP8_CB_K36` is W8A8.
Use this small pair for iteration; a release claim still requires exact
whole-artifact accounting, the same-session streaming protocol, and the 27B
gate. The timings above are offline whole-request latency with three warmups
and ten measured repetitions, not TTFT/TPOT measurements.

## 2026-08-01 DSV4-shaped grouped-BF16 bridge microbenchmark

This is a synthetic kernel measurement of the new quality-preserving bridge,
not a served throughput result. It isolates already-expanded BF16 expert
weights and already-QDQ'd BF16 activations; weight decode/expansion, activation
QDQ, routing, activation, router combine, and model execution are excluded.
Consequently it is independent of the FP4-CB `K14`/`K15` rung and is common
infrastructure for the quality-kernel Gridbook arms in the planned ~92 GB
comparison. It is not evidence for one allocation arm over another.

Execution identity: one GB10 (`sm_121`), Torch `2.11.0+cu130`, CUDA 13.0,
`vllm-node:latest` image ID `d0840ff0e0ba`. The extension was built and
attested before timing. A seed-731 uniform synthetic router used DSV4-shaped
`E=256`, `top_k=8`, `T=128`, hence `P=1024` routed pairs. The measured expert
chunk was `[0,32)`: all 32 experts were active and owned 117 pairs; 250 of 256
experts were active globally. The comparison is Gridbook's single owned
CUTLASS grouped launch against the retired segmented reference: up to 32
BF16 `F.linear` calls writing the same preallocated output segments.

“Cold” is the first data invocation after extension attestation, tensor
allocation, and a device synchronize; it excludes JIT compilation. “Warm” is
the median of 30 individual CUDA-event samples after 10 alternating warmups.
`reference / grouped` below is a speed ratio, so values below 1 mean the owned
grouped kernel is slower.

| DSV4 projection | BF16 GEMM shape per expert | grouped cold | segmented cold | grouped warm | segmented warm | reference / grouped, warm |
|---|---|---:|---:|---:|---:|---:|
| `w13` | `K=4096`, `N=4096` | 6.700 ms | 291.943 ms | 6.471 ms | 5.346 ms | 0.826× |
| `w2` | `K=2048`, `N=4096` | 2.872 ms | 2.677 ms | 2.792 ms | 2.569 ms | 0.920× |

The long-`K` `w13` routing sweep (router seed `731 + T`) remained a warm
regression rather than a hidden crossover:

| routed pairs `P` | pairs in measured chunk | grouped warm | segmented warm | reference / grouped |
|---:|---:|---:|---:|---:|
| 1,024 | 122 | 6.236 ms | 5.367 ms | 0.861× |
| 4,096 | 522 | 6.419 ms | 5.453 ms | 0.850× |
| 8,192 | 1,029 | 6.321 ms | 5.920 ms | 0.937× |

**Interpretation:** the owned bridge avoids the pathological first cuBLAS
initialization seen in the first `w13` reference call and removes the forbidden
vendor/runtime fallback, but its current generic SM80-compatible
`DefaultGemmGrouped` is **6–17% slower on warm GPU time** for these synthetic
DSV4 shapes. The 292 ms cold reference value is one-time library setup and must
not be multiplied by layers or requests. A `TileM=64` prototype **of this SM80
`DefaultGemmGrouped`** did not improve the warm result (the sm12x lane below
is a different collective, where TileM=64 does win — for a different reason:
row-padding granularity, not tile efficiency). The concrete optimization opportunity is a CUTLASS 3.x
SM100/SM121 grouped collective; the legacy grouped template has no SM100
specialization. Until that work is measured end-to-end, this bridge is a
quality/native-ownership result, not a prefill speed claim or a
Blackwell-optimized kernel.

That CUTLASS 3.x collective now exists as an **opt-in second lane**
(`PRISMAQUANT_CB_BF16_SM120`, [KERNELS](KERNELS.md#sm12x-native-grouped-bf16-opt-in-prismaquant_cb_bf16_sm120));
the measurements above are unchanged and still describe the DEFAULT lane. The
new lane's own numbers are immediately below.

## 2026-08-01 sm12x-native grouped-BF16 lane microbenchmark (PROPOSAL DATA)

Same kind of measurement and the same exclusions as the section above, so the
two are comparable, and the same caveat applies twice over: a kernel
microbenchmark **proposes**, only the [NATIVE-PARITY](NATIVE-PARITY.md) served
protocol promotes. Nothing here is a TTFT result or grounds for a default
change; the lane ships opt-in.

Execution identity: one GB10 (`sm_121`), Torch `2.11.0+cu130`, CUDA 13.0,
`gridbook:test`, CUTLASS 4.3.4. Lane config: **pingpong 64×128×64, 3 mainloop
stages, 83,968 B of the 101,376-byte sm120 budget**, tile-scheduler swizzle 1
below 64 padded M-tiles and 8 at or above. Reproduce with
`python3 scripts/bench_bf16_grouped_sm120.py [--tokens T]`. Seed-731 uniform
router, `E=32`, `top_k=8`; warm = median of 30 CUDA-event samples after 10
warmups, taken with the shared GB10 held exclusively. **The sm12x column
includes its padded-gather**, because the lane is what requires the padding;
`Mp` is the padded row count against `P` real routed rows. Ratios above 1 mean
the sm12x lane is faster.

| T | shape | P | Mp | sm12x warm | SM80 warm | segmented warm | sm12x / SM80 | sm12x / segmented |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | `w13` K=4096 N=4096 | 1,024 | 2,048 | 5.480 ms | 6.967 ms | 5.697 ms | **1.271×** | **1.040×** |
| 128 | `w2` K=2048 N=4096 | 1,024 | 2,048 | 2.519 ms | 2.985 ms | 2.603 ms | **1.185×** | **1.033×** |
| 128 | Laguna s1 K=3072 N=2048 | 1,024 | 2,048 | 1.948 ms | 2.291 ms | 1.985 ms | **1.176×** | **1.019×** |
| 128 | Laguna s2 K=1024 N=3072 | 1,024 | 2,048 | 1.015 ms | 1.251 ms | 1.067 ms | **1.233×** | **1.051×** |
| 512 | `w13` K=4096 N=4096 | 4,096 | 5,120 | 8.562 ms | 8.028 ms | 7.566 ms | 0.938× | 0.884× |
| 512 | `w2` K=2048 N=4096 | 4,096 | 5,120 | 3.333 ms | 3.488 ms | 3.054 ms | **1.046×** | 0.916× |
| 512 | Laguna s1 K=3072 N=2048 | 4,096 | 5,120 | 2.744 ms | 2.804 ms | 2.272 ms | **1.022×** | 0.828× |
| 512 | Laguna s2 K=1024 N=3072 | 4,096 | 5,120 | 1.316 ms | 1.390 ms | 1.208 ms | **1.056×** | 0.918× |
| 2048 | `w13` K=4096 N=4096 | 16,384 | 17,280 | 16.895 ms | 17.913 ms | 11.284 ms | **1.060×** | 0.668× |
| 2048 | `w2` K=2048 N=4096 | 16,384 | 17,280 | 6.410 ms | 6.977 ms | 4.603 ms | **1.088×** | 0.718× |
| 2048 | Laguna s1 K=3072 N=2048 | 16,384 | 17,280 | 5.188 ms | 4.661 ms | 3.234 ms | 0.898× | 0.623× |
| 2048 | Laguna s2 K=1024 N=3072 | 16,384 | 17,280 | 1.869 ms | 1.894 ms | 2.226 ms | **1.013×** | **1.191×** |

**Where this lands against the P1 target.** The target was "≥ segmented-BF16
parity warm". At `T=128` — the token count of the published DSV4 bridge
measurement above — it is **met on all four cells** (1.019–1.051×), and the
lane beats the SM80 bridge it would replace by 1.18–1.27×. At `T=512` it is
**not** met (0.83–0.92× of segmented) although the lane still beats the SM80
bridge on three of four cells. At `T=2048`, outside the target band, only the
short-`K` Laguna s2 cell reaches parity.

### The mechanism, isolated

This construction re-reads an expert's B slice **once per padded M-tile**, and
at these shapes the operator is bound by that traffic. Two controlled
measurements separate the SCHEDULE from the CONSTRUCTION, on the hardest cell
(`w13`, K=N=4096, 4,096 routed rows, GEMM only, no gather):

| routing | padding | sm12x | segmented | SM80 | sm12x / segmented |
|---|---:|---:|---:|---:|---:|
| ragged (real, `T=512`) | 1.25× | 7.941 ms | 7.496 ms | 7.964 ms | 0.94× |
| **packed, 128 rows/expert** | **1.00×** | 6.368 ms | 7.180 ms | 6.721 ms | **1.13×** |
| **packed, 64 rows/expert** | **1.00×** | 5.402 ms | 5.830 ms | 6.709 ms | **1.08×** |

With the ragged rounding removed the collective is **1.08–1.13× faster than
segmented cuBLAS** and 1.05–1.24× faster than the SM80 lane. The schedule is
not the deficit; the padding is. That is why the compiled rung is TileM=**64**
(halving the rounding granularity) even though the 128-row tile is the more
efficient GEMM per FLOP, and why the residual at `T=512` — where each expert
holds ~128 rows and rounding costs 1.25× — cannot be tuned away at a fixed
TileM.

### What was swept

Every candidate below was compiled and timed on all four shapes at `T=128` and
`T=512` (12 kernels × 4 swizzles × 3 raster orders); the winner is the row in
bold.

| kernel layer | tile | stages | smem | verdict |
|---|---|---:|---:|---|
| cooperative (4×2×1 warps) | 128×128×64 | 2 | 75,776 | the previous rung; 0.71–0.97× segmented |
| cooperative | 128×128×32 | 2 / 3 / 4 / 5 | 43k–92k | slower on every cell |
| cooperative | 128×64×64 | 2 | 59,392 | 0.32–0.93×; worst tested |
| **pingpong (2×2×1 warps)** | **64×128×64** | **3** | **83,968** | **winner: 1.02–1.05× at T=128** |
| pingpong | 64×128×64 | 2 | 59,392 | 0.90–0.96× of the 3-stage rung |
| pingpong | 64×256×64 | 2 | 92,160 | competitive at T=128, 0.75–0.86× at T=512 |
| pingpong | 64×64×64 | 4 | 75,776 | wins Laguna s1/s2 by ~2%, loses w13 by 30% |
| pingpong | 64×128×32 / 64×256×32 | 3 / 4 / 6 | 59k–84k | K-slice too thin; 0.50–0.93× |
| pingpong | 128×128×64 | 2 | 75,776 | ≈ cooperative — TileM, not the layer, is the lever |
| 256×128×64, 128×256×64, 128×64×128 | — | — | ≥48 KiB/stage | infeasible: auto-carves to ONE stage |

Raster order: CUTLASS's `Heuristic` (which resolves to `AlongN` here) matched
or beat `AlongN` everywhere and beat `AlongM` on every cell — `AlongM` cost up
to 2× at `T=512`. Swizzle: at 32 padded M-tiles swizzle 1 was fastest on three
of four shapes; at 80 M-tiles swizzle 8 was worth 1.36× on `w13`
(11.17 → 8.16 ms) and never cost more than 7% elsewhere, so the shipped policy
optimises the worst cell. Swizzle is a tile-ORDER argument and cannot move a
bit of output.

**Status.** The lane stays **opt-in**. It now strictly dominates the SM80
bridge it would replace on every cell at `T ≤ 512` except one (`w13` at
`T=512`, 0.938×), meets the segmented-parity target at `T=128`, and misses it
at `T=512`. The two remaining structural costs are named and measured: the
ragged padding tax above, and the padded activation gather (0.02–0.35 ms per
launch here) that the exact-segment lane does not pay. Closing them needs a
change of construction, not of schedule — a TileM ladder selected by measured
rows-per-expert (which the fused lanes already have, and which the `tile_m`
binding and dispatch helper already parameterise), or an A-side row-gather
inside the mainloop. The served [NATIVE-PARITY](NATIVE-PARITY.md) protocol has
not been run.

## 2026-08-02 FP4-CB v2 fused mid-M lane microbenchmark (PROPOSAL DATA)

**PROPOSAL DATA ONLY.** Per [NATIVE-PARITY](NATIVE-PARITY.md) a kernel
microbenchmark proposes; only the served protocol promotes. Nothing below is a
serving claim, a TTFT number, or grounds for changing a default — the lane
stays behind `PRISMAQUANT_CB_FP4_FUSED_MIDM`.

**What is timed.** The two ways the same dense FP4-CB v2 prefill GEMM can be
executed at mid M, each as the WHOLE operator the serving path would run:

* `fused` — one launch of `cb_fused_fp4v2_prefill_mm`; the packed CB rows are
  decoded inside the CUTLASS producer/consumer stage and the `[N,K]` BF16 tile
  never exists in HBM.
* `bridge` — today's shipping route: `expand_fp4_v2_to_weight` writes the
  decoded `[N,K]` BF16 tile to HBM, then the owned CUTLASS grouped kernel
  (`E=1`) multiplies it. **The expand is inside the timed region**, because the
  transient it writes is exactly what the lane removes; charging only the GEMM
  would measure a different claim.

The shared upstream work (activation group-16 QDQ, reshape) is excluded, as the
DSV4 bridge table above excludes it. Every cell is **bit-checked before it is
timed**: the harness asserts the fused result equals the passthrough oracle
(`sm120_fp4v2_bf16_mm_fork`, same tile/TiledMma/epilogue, plain BF16 B) fed the
expander's tile, so a regressed kernel cannot report a fast wrong number.

GB10, cc 12.1, CUTLASS 4.3.4, torch 2.11.0+cu130, k_bits = 16, tile
`128×64×64`/2 stages. Warm median of 30 CUDA-event samples after 10 warmups;
`scripts/bench_fp4v2_fused_midm.py`, run under the host GPU bench lock.

| shape | M=9 | M=16 | M=32 | M=64 | M=128 |
|---|---|---|---|---|---|
| 27B qkv/o `K=5376 N=5376` | **1.97–2.00×** | 3.54× | 3.55× | 3.54× | 3.57× |
| 27B gate `K=5376 N=14336` | 2.25× | 3.70× | 3.55× | 3.35× | 3.08× |
| 27B down `K=14336 N=5376` | 4.34× | 4.37× | 4.08× | 3.43× | 3.37× |
| DSV4 w13 `K=4096 N=4096` | 1.72× | 2.93× | 2.99× | 3.05× | 3.11× |
| DSV4 w2 `K=2048 N=4096` | 1.06× | 1.12× | 1.79× | 1.78× | 1.84× |

Ratios > 1 mean the fused lane is faster. All 30 cells bit-equal to the oracle.

**Why the band is wider than the fp8 mid-M twin's 1.04–1.45×.** Structural, not
tuning: the fp4 *quality* expand materializes **BF16** (2 bytes/weight — 4× the
transient traffic of fp8-CB's direct-to-E4M3 expand), so deleting it is worth
proportionally more. This is the same asymmetry the 2026-08-01 audit names as
cause (b).

**Open item — an unexplained M ≤ 12 cliff.** On the 27B qkv shape the fused
lane's warm median is ~0.37 ms for M = 8…12 and ~0.20 ms for M = 13…128:

| M | 8 | 9 | 10 | 11 | 12 | 13 | 16 | 24 | 32 | 9 (repeat) |
|---|---|---|---|---|---|---|---|---|---|---|
| fused warm (ms) | 0.372 | 0.367 | 0.382 | 0.365 | 0.342 | 0.201 | 0.201 | 0.201 | 0.201 | 0.359 |

The work is identical at every M in that range (one M-tile, the same
`N/64` CTAs, the same decode), the repeat rules out measurement ordering, and
the bridge column is flat across the same sweep — so this is a property of the
fused kernel, not of the harness. It costs the lane roughly half its win at
M ≤ 12 and it should be profiled (ncu) before any promotion argument is made.

## What is being compared, and how

- **Matched-rate baseline.** Quality rows compare the same model family against a
  stock per-Linear **NVFP4/FP8** assignment served by vLLM's
  `compressed-tensors` path. Historical tables generally match quantizable-body
  bpp or body bytes, not a canonical whole-artifact inventory. They also compare
  complete execution contracts—often native W4A4 against CB W8A8—not weight
  encoding alone. Treat each row according to its stated byte scope and W/A
  contract; only an inventory-derived whole-artifact match can pass the formal
  [native-parity gate](NATIVE-PARITY.md). The published “native” baselines are
  generally mixed NVFP4/FP8 assignments, not pure-NVFP4 artifacts.
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
FP8-CB codebook rungs** and **zero** stock NVFP4/FP8 under that run's
accuracy-only allocation objective. The teacher-backed comparison below is the
quality evidence; the zero-native assignment by itself is not a speed claim.

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
This is a close quantized-body byte match for the quality comparison, not a
canonical whole-artifact inventory match.

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

- **Measured batch-1 decode is at/above the native row in this historical cell**
  (10.3 vs 10.26). This does not establish parity for batched/speculative decode,
  tail ITL, other prompt distributions, or a formal whole-artifact byte match.
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
- **Prefill (historical update 2026-07-23).** The TTFT above was measured against
  the old correctness-path per-expert prefill loop (~71k launches per 1400-token
  prefill) and does not describe the current native-only path. A CUDA chunk-expander
  feeding vLLM's own fused-MoE grouped kernel replaced it; measured on
  Laguna-S-2.1 (117B MoE): **293 → 1,821 tok/s at 8k** and **207 → 1,822 tok/s at
  63k** (commit `8829c16`). Current Gridbook instead owns the exact-BF16 CUTLASS
  grouped bridge; neither the 35B TTFT row nor those Laguna figures have been
  re-measured on that bridge.

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
- The joint-menu allocation experiment gave 36 Linears to vanilla FP8 and zero
  to vanilla NVFP4. That is an accuracy-only allocator outcome, so zero NVFP4 is
  circular evidence about that objective—not proof that CB dominates native on
  latency or on every unit. This 295B result has fit/serve/TEB evidence and no
  BF16-teacher quality claim.
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

## Retired Triton path: historical cost

Current Gridbook has no Triton dependency or serving fallback. If a required
native CUDA/CUTLASS extension is unavailable, serving fails closed. The numbers
below are preserved because they motivated that policy; they are not a backend
selection operators can reproduce on the current release.

**Dense decode — historical measured comparison.** On the 27B at 5.5 bpp, the
old Triton decode-GEMM measured **4.20 tok/s** and the CUDA GEMV that replaced it
measured **10.27–10.30 tok/s**. The 4.20 figure came from the whole Triton-
prototype configuration and predates the FP8-direct native expander, so it is
not a fresh A/B against today's kernel set.

**MoE decode — historical measured comparison.** The per-expert transient loop
on the 35B measured **3.52 tok/s** before the native grouped kernel replaced it
at **32.6–33.3 tok/s**. That ~10k-host-operation-per-token result demonstrated a
launch problem, not a bandwidth limit. Its exact magnitude is artifact-specific
and must not be used as an estimate for current native paths.

Sources: the 27B dense and 35B MoE before/after figures are the same served A/B
runs as the tables above. No claim is made for a current fallback because none
exists.

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
- **Speed status is honest and uneven.** Historical batch-1 cells reached or
  exceeded their native decode rows, but batched/speculative decode, tail ITL,
  and the formal exact-byte workload matrix remain open. Large-M dense
  **prefill is not yet at native parity**: ~1.44× on the 27B (traffic-bound
  transient expand). The predecessor MoE CUDA chunk-expander path measured
  Laguna 117B at 293 → 1,821 tok/s at 8k; the 35B TTFT row predates it, and the
  current owned CUTLASS grouped bridge still needs its own measurement. The GGUF
  prefill comparison (the 295B ~2.6×) is against a
  *different* serving stack (llama.cpp CUDA-core IQ dequant), and is the clean
  "why native tiles win" number; it is not a claim of parity with vLLM's own
  native NVFP4/FP8 GEMM at large M.
- **bpp vs disk size differ by the BF16 floor.** Reported bpp is over quantizable
  params; embeddings/`lm_head`/norms/non-CB Linears are excluded from bpp but
  resident on disk, so `bpp × params` understates the artifact size.
- **Blackwell is the only measured target.** Every number on this page is from
  `sm_121`. The decode kernel itself carries no architecture guards and is
  *expected* to run from `sm_80` up, and only the mid-M fused prefill kernel is
  genuinely `sm_120`-family bound — but that is inferred from the code, not
  measured. Per-GPU expectations, with measured/inferred labelled, are in
  [`INSTALL.md`](INSTALL.md#hardware-matrix). Without `nvcc` (or compatible
  prebuilt native extensions), Gridbook cannot serve and fails closed.
- **One published artifact per headline is not the same as three.** The 27B and
  295B artifacts are downloadable; **the 35B MoE artifact is not published** — its
  numbers are an internal measurement showing the format win reproduces on MoE,
  not something you can reproduce without an encoder.
