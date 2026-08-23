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

## 2026-08-23 FP8-source grouped shard — decode microbenchmark

Why it exists: DeepSeek-V4's `wo_a` is a grouped-BMM source-passthrough unit,
and column-sharding it for tensor parallelism divides the kernel's group count
(G 8 -> 4 at TP=2). That is a new kernel geometry, so it was measured before
the lane admitted it. **This is a kernel microbenchmark, not a served result**:
no two-node serve has been run, and nothing here is a TP speedup claim.

Execution identity: GB10 `sm_121`, image `gridbook:0.8.12-r2ab`, branch
`tp-passthrough`. Geometry is the DSv4 `wo_a` plane — 1024 rows per group,
K 4096, batch 1. Driver `scripts/bench_fp8_source_grouped_shard.py`. Each arm:
500 iterations per timed run after 50 warmup, three A/B/A repeats, median,
CUDA-event timed, **rotating over a 268 MB working set** so no arm is served
from cache.

| call | per call | vs G=8 | bytes a rank holds |
|---|---:|---:|---:|
| G=8, whole plane (TP=1) | 163.34 µs | 1.000× | 33.6 MB |
| G=4, one rank at TP=2 | 80.90 µs | 0.496× | 16.8 MB |
| G=2, one rank at TP=4 | 41.23 µs | 0.253× | 8.4 MB |

Decode time falls with the bytes a rank holds, near-exactly linearly: there is
**no occupancy cliff** at the smaller group counts.

One trap worth publishing. A hot loop over a single resident plane — the
obvious way to write this bench — reports G=4 at 0.294× instead of 0.496×,
because 16.8 MB is small enough to be served from cache while 33.6 MB is not.
A real serve streams every other layer between two `wo_a` calls, so the
rotating number is the honest one. The cache-friendly number would have
overstated the sharded arm by 1.7×.

Exactness, measured alongside: at both degrees and on every rank, the sharded
call is bitwise equal to the corresponding columns of the unsharded G=8 call,
on the expand path and the GEMV path alike
(`tests/test_fp8_source_w8a16_cuda.py`).

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

**Status (superseded 2026-08-02).** This table describes the lane's original
PADDED-COPY mode with the natural expert order; those numbers still hold for
that mode and the analysis above (the padding tax, the sweep, the swizzle
policy) remains the measured record. The two structural costs it names —
the ragged padding tax and the padded activation gather — are closed by the
construction changes in the next section: an in-mainloop A-row gather and a
swizzle-group-aligned expert order. A `TileM` ladder was considered and is
measured-dead for these cells: every 128-row tile in the sweep above
(cooperative AND pingpong, all four swizzles, three raster orders) landed at
≤ 0.97× segmented, because the swizzle already recovers same-expert B reuse
in L2 while the 1.5× padding of a 128 rounding granularity is real work. The
served [NATIVE-PARITY](NATIVE-PARITY.md) protocol has not been run.

## 2026-08-02 sm12x grouped-BF16 lane: in-mainloop A-row gather + swizzle-aligned tile order (PROPOSAL DATA)

Same instrument, cells, seed and warm discipline as the table above
(`scripts/bench_bf16_grouped_sm120.py`, seed-731 router, `E=32`, `top_k=8`,
warm = median of 30 CUDA-event samples after 10 warmups, GB10 `sm_121`,
Torch 2.11.0+cu130, `gridbook:test`, GPU held exclusively under the bench
lock), so the rows are directly comparable. Two construction changes to the
SAME compiled collective — the kernel schedule, tile (pingpong 64×128×64),
stages (3) and shared memory (83,968 B) are unchanged:

1. **In-mainloop A-row gather** (`cb_bf16_grouped_mm_sm120_gather[_out]`).
   The producer warp reads each padded row through `row_src[m]` from the
   COMPACT activation with predicated, zero-filling 16-byte `cp.async` (ids
   outside `[0, S)` are the padding rows), instead of TMA-reading a
   materialized row-padded copy. The padded activation copy — an HBM write
   plus a padded re-read too large for L2 — no longer exists, and the compact
   source IS L2-resident at these sizes. Producer accounting follows
   upstream's own `sm120_mma_tma_blockwise_scaling.hpp` idiom (33 producer
   events; B-only transaction bytes). The smem stage bytes are identical to
   the padded-copy mode's, so the two modes are **bit-identical**
   (`tests/test_bf16_grouped_cutlass.py` asserts `torch.equal` on every
   routing shape, including K-residue).
2. **Swizzle-group-aligned expert order**
   (`bf16_grouped_lane.pack_expert_blocks`). The tile scheduler sweeps N with
   groups of 8 M-tiles (the measured large-grid swizzle), so an expert whose
   tiles straddle a group boundary has its whole B slice fetched from DRAM
   once per group touched. Packing experts into groups — deterministic
   first-fit-decreasing on the routing histogram, pure host math, telemetered
   as groups-touched vs minimum — aligns the boundaries: at `T=512` the
   seed-731 routing straddles 39 groups against a 32-group minimum, and
   packing reaches exactly 32/32 (the `grp` column). Tile order is scheduler
   order; the bit gate asserts a packed order is a pure block permutation.

`sm12x gather` below is the whole operator — ONE launch, no gather kernel, no
copy (the `row_src`/`expert_ids` layout vectors are routing products of the
same excluded class as every arm's routing). `sm12x padded` is the previous
mode re-measured in the same session, with its padded copy inside the timed
region as before.

| T | shape | P | Mp | sm12x gather | sm12x padded | SM80 | segmented | gather / SM80 | gather / segmented |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | `w13` K=4096 N=4096 | 1,024 | 2,048 | 5.591 ms | 5.496 ms | 6.670 ms | 5.865 ms | **1.193×** | **1.049×** |
| 128 | `w2` K=2048 N=4096 | 1,024 | 2,048 | 2.492 ms | 2.520 ms | 2.993 ms | 2.571 ms | **1.201×** | **1.032×** |
| 128 | Laguna s1 K=3072 N=2048 | 1,024 | 2,048 | 1.847 ms | 1.948 ms | 2.254 ms | 1.906 ms | **1.221×** | **1.032×** |
| 128 | Laguna s2 K=1024 N=3072 | 1,024 | 2,048 | 1.002 ms | 0.984 ms | 1.135 ms | 1.053 ms | **1.133×** | **1.051×** |
| 512 | `w13` K=4096 N=4096 | 4,096 | 5,120 | 6.870 ms | 8.057 ms | 8.089 ms | 7.606 ms | **1.177×** | **1.107×** |
| 512 | `w2` K=2048 N=4096 | 4,096 | 5,120 | 2.662 ms | 3.321 ms | 3.458 ms | 3.035 ms | **1.299×** | **1.140×** |
| 512 | Laguna s1 K=3072 N=2048 | 4,096 | 5,120 | 2.040 ms | 2.735 ms | 2.795 ms | 2.247 ms | **1.370×** | **1.102×** |
| 512 | Laguna s2 K=1024 N=3072 | 4,096 | 5,120 | 1.050 ms | 1.303 ms | 1.330 ms | 1.209 ms | **1.267×** | **1.151×** |

**Isolated effect of the tile order alone** (GEMM-only, padded-copy mode,
same iso protocol as the packed-vs-ragged table above, `T=512`): natural vs
packed order — `w13` 8.071 → 6.824 ms, `w2` 3.209 → 2.709 ms, s1
2.504 → 2.081 ms, s2 1.234 → 1.063 ms (13.9–16.9%), while at `T=128`
(32 tiles, swizzle 1) the order is neutral to <0.3%. The group-straddle
excess (39/32 ≈ 1.22× B-slice fetches) was the single largest remaining tax
at `T=512`; the gather then removes the padded copy and turns the A stream
into L2-resident compact reads (`s1` gather 2.040 ms vs 2.081 ms for the
packed GEMM alone — the gather mode is faster than the padded GEMM even
before counting the copy it deletes).

**Status of the 13.9–16.9% figure: measured at `E=32`, where the packing is
active.** `moe.py` applies `pack_expert_blocks` only when one expert chunk
covers the whole layer (`chunk >= E`), because a narrower chunk indexes blocks
as `block_off[c0]..block_off[c1]` and so assumes expert-major contiguity. The
chunk is `PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK` if set, else the
`PRISMAQUANT_CB_PREFILL_CHUNK_BYTES` budget (1 GiB) divided by one expert's
`w13` BF16 bytes — the same `_native_bf16_chunk` the persistent-B table below
runs, which is why its `E=128` cells show `chunks=2`. At those expert counts,
and under any lowered `..._CHUNK_BYTES`, the lane takes the gather but **not**
the tile order, so this row does not carry over to them. Packing within a
chunk is queued in [ROADMAP](../ROADMAP.md#p1--close-the-remaining-native-parity-gaps).

**Where this lands against the P1 target** ("≥ segmented-BF16 parity warm"):
**met on every cell at both token counts** — 1.032–1.051× segmented at
`T=128` and 1.102–1.151× at `T=512`, while beating the SM80 bridge it
replaces by 1.133–1.221× (`T=128`) and 1.177–1.370× (`T=512`). The
`T=512` cells that missed at 0.83–0.92× under the padded-copy construction
now clear parity by 10–15%.

**Status.** The lane stays **opt-in** (`PRISMAQUANT_CB_BF16_SM120`); these
are whole-operator microbenchmarks and PROPOSAL DATA — the served
[NATIVE-PARITY](NATIVE-PARITY.md) protocol has not been run. The gather mode
and the tile-order policy change no output bit relative to the padded mode
(gated `torch.equal`), so the lane's requalification surface is unchanged:
FP32 reduction order vs the SM80 lane, exactly as before.

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

## 2026-08-02 persistent-B grouped-MoE decode-in-mainloop microbenchmark (PROPOSAL DATA)

The two sections above time a **GEMM**. This one times the **whole routed
operator**, which is what [NATIVE-PARITY](NATIVE-PARITY.md) requires of a
grouped-MoE change: routing, activation QDQ, the gather, both projection
stages, the gated activation between them, the router-weight multiply and the
combine — everything from `(x, topk_ids, topk_weights)` to a combined
`[T, hidden]`. It is still a microbenchmark, so it still only **proposes**;
only the served protocol promotes, and the lane ships opt-in.

Execution identity: one GB10 (`sm_121`), Torch `2.11.0+cu130`, CUDA 13.0,
`gridbook:test`. FP4-CB two-tier v2, `k=16`, `type_size=73`; softmax-top-`k`
router over seed-731 `randn` logits, `top_k=8`; warm = median of 30 CUDA-event
samples after 10 warmups, taken with the shared GB10 held exclusively.
Reproduce with `python3 scripts/bench_moe_persistent_b.py`. Shapes are whole
MoE layers — `w13` is `[E, 2*inter, hidden]` and `w2` is `[E, hidden, inter]`,
so both stages run at their real shapes rather than one shape twice.

The three arms are the three routes a `T > 16` FP4-CB MoE prefill can take:

- **pb** — the persistent-B decode-in-mainloop lane
  (`PRISMAQUANT_CB_MOE_PERSISTENT_B=1`), exact `expert_ends` segments;
- **sm80** — the DEFAULT shipping path: `cb_expand_fp4_v2` per expert chunk
  into a BF16 `[Ec,N,K]` transient, then the SM80-schedule grouped bridge;
- **sm120** — the OPT-IN pingpong bridge lane, i.e. the same expansion feeding
  the freshly improved sm12x collective, **including** its padded gather and
  its one host read, because those are what that lane requires.

Every arm's expert-chunk loop is the serving one (`_native_bf16_chunk`'s 1 GiB
budget), which is why `E=128` shows `chunks=2`. **`expand%` is the share of the
DEFAULT operator spent inside the transient expansion the new lane deletes** —
the tax this kernel exists to remove. Ratios above 1 mean persistent-B is
faster.

| shape | E | T | P | pb warm | sm80 warm | sm120 warm | expand | expand% | sm80/pb | sm120/pb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSV4 `h=4096 i=2048` | 32 | 128 | 1,024 | 5.768 ms | 19.354 ms | 17.027 ms | 8.663 ms | 44.8% | **3.355×** | **2.952×** |
| DSV4 `h=4096 i=2048` | 32 | 512 | 4,096 | 10.181 ms | 21.921 ms | 22.550 ms | 8.666 ms | 39.5% | **2.153×** | **2.215×** |
| DSV4 `h=4096 i=2048` | 32 | 2,048 | 16,384 | 28.672 ms | 41.506 ms | 36.125 ms | 8.671 ms | 20.9% | **1.448×** | **1.260×** |
| Laguna `h=3072 i=1024` | 32 | 128 | 1,024 | 2.485 ms | 7.074 ms | 6.723 ms | 3.177 ms | 44.9% | **2.847×** | **2.705×** |
| Laguna `h=3072 i=1024` | 32 | 512 | 4,096 | 4.360 ms | 7.948 ms | 8.316 ms | 3.182 ms | 40.0% | **1.823×** | **1.907×** |
| Laguna `h=3072 i=1024` | 32 | 2,048 | 16,384 | 12.669 ms | 13.315 ms | 13.145 ms | 3.177 ms | 23.9% | **1.051×** | **1.038×** |
| Laguna `h=3072 i=1024` | 128 | 128 | 1,024 | 8.755 ms | 27.207 ms | 26.396 ms | 12.662 ms | 46.5% | **3.108×** | **3.015×** |
| Laguna `h=3072 i=1024` | 128 | 512 | 4,096 | 9.386 ms | 27.167 ms | 26.281 ms | 12.681 ms | 46.7% | **2.894×** | **2.800×** |
| Laguna `h=3072 i=1024` | 128 | 2,048 | 16,384 | 16.687 ms | 32.952 ms | 31.390 ms | 12.659 ms | 38.4% | **1.975×** | **1.881×** |

**The expand tax is real and it is the whole story at low `T`.** It measures
**20.9–46.7%** of the default operator, bracketing the ~35% the 2026-08-01
audit quotes, and — the part that matters for production — it does **not**
shrink with the expert count. At `E=128` it is 38–47% at every token count,
because the expansion pays for all 128 experts whether or not the router used
them, while the GEMM only pays for the routed rows. That is the asymmetry the
new schedule removes: its grid visits `(expert, N-tile)` pairs and an unrouted
expert costs two int32 loads and a CTA return.

**The lane wins every cell measured**, 1.05–3.36× over the default bridge and
1.04–3.02× over the pingpong bridge, and the two baselines are close enough to
each other that the win is against the *route*, not against a weak GEMM.

**Where the win narrows, and why.** The ratio falls with mean routed rows per
expert (`P/E`): the kernel decodes each weight tile once per `TM`-row M-tile,
so at `P/E = 512` (`E=32`, `T=2048`) it decodes four times where the expansion
decodes once, and the two costs nearly cancel (1.05×). The production-shaped
`E=128` row keeps `P/E = 128` even at `T=2048` and holds **1.98×**. The kernel
answers this with a shape-driven tile choice (cfg 4 below ~64 mean rows, cfg 1
above, selected from `P` and `E` alone so it stays capture-safe), and a
kernel-level sweep of the alternatives is recorded in
[KERNELS](KERNELS.md#persistent-b-decode-in-mainloop-default-auto-prismaquant_cb_moe_persistent_b):
a `256×64` tile halves the decode repetition in exactly this regime and still
lost, because it falls to one CTA per SM. Raising `TM` past 128 without losing
occupancy is the identified next step, and it is a change of *tile*, not of
schedule.

**Numerics were checked, not assumed.** The benchmark verifies all three arms
agree to a relative L2 of ≤ 4e-3 outside the timed region before reporting any
row; the observed worst case was 1.5e-3 and five of nine rows came out
bit-identical between pb and sm80. The weight decode itself is bit-exact
against `cb_expand_v2` under `torch.equal` in the test suite. The served
[NATIVE-PARITY](NATIVE-PARITY.md) protocol has **not** been run.

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
To match residency, set `PRISMAQUANT_PRELOAD_FUSED=1` on **both** arms: it now warms
*every* native extension family at plugin registration (previously only the two fused
ones), so neither arm can drift by loading a module the other never touched.

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

## 2026-08-21 kernel-eval branch: B1/B2/B7 measurements

All cells: GB10 (sm_121), `gridbook:0.8.11-clean-187c721` container, tree
sources JIT-built fresh, median-of-N CUDA-event timing, bit-equality asserted
per point before timing. Raw scripts kept outside the repo; branch
`perf/kernel-eval-2026-08-21`.

**B1 — dense FP4-v2 GEMV, round-2 backport (`PRISMAQUANT_CB_FP4V2_DENSE_R2`),
median of 5×100 after 50 warm, µs/call.** Legacy → R2 deltas across
K∈{2048,4096} × N∈{4096,12288} × M∈{1,2,4,8} × k∈{16,20}: R2 wins **32/32
points**. Representative cells: K2048/N4096/M1 k20 27.95→24.15 (−13.6%);
K2048/N12288/M2 k20 96.46→79.89 (−17.2%); K4096/N4096/M2 k20 57.40→45.84
(−20.2%); smallest win −0.9% (K2048/N12288/M1 k16). Gains grow with k —
consistent with the compute-bound decode profile (fewer issued load
instructions per codeword). Opt-in flag, still default OFF. A 2026-08-23 flip to
default-ON was attempted and REVERTED: R2 wins on large n_sb (−3.44…−10.17%
per-M aggregate at the gold artifact's real rungs, n_sb 20/68) but LOSES at
M=1 when n_sb < WARPS — +9.14% at k=12/K=768/n_sb=3 against a 0.36% control
spread, +4.50% at n_sb=5 — because the burst staging never amortizes over a
single warp iteration. The same cells win at M=2. Served NATIVE-PARITY was
also never run. See CHANGELOG "Measured negative result". Consensus re-measurement (2026-08-21, two independent
harnesses, fresh tree-private builds, `torch.equal` every cell): 32/32 on
this grid, **64/64** on the shipped Qwen3.8-27B dense shapes × k∈{12,14,16,18}
× M∈{1,2,4,8} (k12 −1…−8% … k18 −5…−16%), 47/48 on a small-K/untiled-N grid
(K∈{1024,1536}, N∈{4097,5000}) and 4/4 at M=16; the one losing cell,
**k=12 / K=1536 / M=1, +1.7…+2.4%**, reproduced three times on an idle GPU
— a known corner for the NATIVE-PARITY evaluation. The `R2=false`
instantiations are SASS-identical to the pre-commit build (20/20), and
`compute-sanitizer` memcheck is clean on the gate, the shipped shapes and
ten edge layouts (N=1/K=256, N=97, all-byte-fallback rows, fused two-role
`row_off`, M=16, decode-contract v2).

**B7 — sm12x grouped-BF16 per-chunk packing (K1.5), isolated stage-one gather
GEMM, median of 7/5.** Cell: E=64 mildly-skewed experts, w13 K=N=4096,
T=1024×top_k=8, tile_m=64, swizzle group 8, chunks=2 with tiles/chunk
[126,34] (natural-order straddle tax 75 groups vs 64 minimum = 1.17×
B-fetches): global-pack 13.39/13.63 ms; per-chunk pack **14.33/14.70 ms**;
natural order 16.11/16.39 ms ⇒ packed/natural **0.890 / 0.897**. Two-chunk
cost vs one launch ≈1.07×. Uniform-router control cell: packing inert
(1.001×) — the win appears exactly when segments straddle. *Consensus
re-measurement (2026-08-22, independent harness, three GPU runs):*
packed/natural 0.889 on the skewed cell and **0.896 on a random-uniform
router** (T=1024×top_k=8 over E=64, tiles/expert 2/3, natural-fetch factor
75 vs 64 minimum — the same straddle excess as the skewed cell), two-chunk
vs one-chunk 1.068; top_k=1 `torch.equal` gates chunks2-packed vs
chunks2-unpacked, vs chunks1-packed, and chunks4 vs chunks2 all True; the
1.001× control's construction is not recorded in the repo or the evidence
snapshot, so read "inert" as "no segment straddles a group boundary", not
as "uniform router".

**B2 — persistent-B staging vectorization (REJECTED on measurement).**
Byte-neutral by construction (staging theorem preconditions P1–P5 asserted
in-source; decode-probe `torch.equal` suites green), but whole-operator A/B
on DSV4 h4096/i2048 E=32 top_k=8 (`bench_moe_persistent_b.py`, warm ms,
median of 3) rejected it: at the DSv4-dominant k=12 the committed
(`DEVINL`) build runs +7…+11% slower than the byte-granular baseline
(pb T=512: 8.859/8.946/8.941 → 9.819/9.914; T=2048:
25.863/26.262/26.343 → 27.896/28.151), and the `__noinline__` spelling the
design called for is slower still (+14.1%). The recorded rationale —
inlining cost ~26 registers/thread and a ~23% whole-operator regression,
hence `__noinline__` scoping — is contradicted by `cuobjdump
--dump-resource-usage` of the built binaries: the inlined build allocates
FEWER registers than baseline on every fp4 instantiation — including
`<128,64,8>`, the tile `pick_cfg` actually selects at DSV4 shapes (112 → 80;
`<128,32,4>` 128 → 120, `<64,64,4>` 126 → 72) — and the noinline build more
(128/125). SASS of `<128,64,8>`: the mainloop is unchanged (32 HMMA, 16 LDSM,
8 LDGSTS in both builds; occupancy smem-capped at 2 CTAs/SM in both); the
baseline staging is one compiler-unrolled burst of 8 independent `LDG.E.U8`
+ 8 `STS.U8`, the replacement five `unroll 1` loops per row each carrying
`VOTE.ANY`/`WARPSYNC`/`SHFL.DOWN` and a reconvergent branch — a dependent
chain per word where the byte loop had memory-level parallelism. No isolated staging-share cell was
recorded. k≥16 cells were within noise (+0.4…+1.5%), which does not rescue
k=12; not merged.

**B2-S3 — FP8-family-only staging vectorization (`ox/pb-salvage-s3`,
SHIP consensus 2026-08-22 — coordinator + adversarial Ox Alpha; `f457de2`).**
The rejected B2's copy,
salvaged for the one family where it pays and gated at COMPILE TIME: both
mainloop staging sites dispatch on the kernel's existing `kFp8` trait; the
FP8 arm takes an aligned-u32 word copy (byte-loop fallback for odd `qw` base
pointers — `type_size == 4k` keeps every superblock offset 0 mod 4, so only
the base pointer can misalign) with word-granular slot zeroing, and the FP4
arm keeps the baseline loops verbatim. b2's ptxas mechanism (any text change
to shared inlined staging re-tunes the whole fp4 kernel schedule) is treated
as a gate, not a claim: against a fresh baseline build (`464d27f`, identity
`1546441…`) all 8 fp4 mainloop instantiations are instruction-stream- AND
resource-identical (hot `<128,64,8>` tile reads **112** registers, not 80),
and the three decode kernels are identical. Bit-neutrality: 25/25
`torch.equal` probe cells — fp4 decode/prefill k∈{12,14,16} × rows/expert
{4,128}, fp8 prefill at every shipped rung k∈{28,36,44,48} × E∈{256,32} ×
rows/expert {4,128}; plus a new `tests/test_persistent_b_fp8_staging.py`
that reads the entire staged plane bitwise through prefill via one-hot
activations at the four shipped FP8 rungs, on aligned and deliberately
misaligned sources (signed-zero-only pairs excepted: the epilogue turns
−0.0 weights into +0.0). Whole-operator A/B, DSV4 h4096/i2048 E=256
top_k=6, warm ms, iters 30/warmup 10, interleaved A/S3/A/S3 under clean
single-process GPU windows (baseline `1546441…`, S3 `5e193c33…`):

| lane | T | baseline (×2) | S3 (×2) | Δ |
|---|---|---|---|---|
| fp8-CB k=28 E=256 | 128 | 35.417 / 35.938 | 31.324 / 31.608 | −11.6…−12.0% |
| fp8-CB k=28 E=256 | 512 | 38.465 / 38.850 | 34.180 / 34.308 | −11.1…−11.7% |
| fp8-CB k=28 E=256 | 2048 | 41.910 / 42.288 | 37.650 / 37.837 | −10.2…−10.5% |
| fp8-CB k=36 E=256 | 512 | 89.097 | 76.141 | −14.5% |
| fp8-CB k=48 E=256 | 512 | 131.711 | 116.913 | −11.2% |
| fp4-CB k=12 E=32 | 512 | 8.975 | 8.972 | −0.03% (noise, as SASS identity demands) |

Small-E control (DSV4sm, E=32): k=28 −8.3…−10.9% across tokens; k=36 −7.3%;
k=48 −2.3%. An intermediate S3 build that kept the baseline byte zero-passes
inside the FP8 arm measured only −2.4…−3.0% at k=28 despite passing every
correctness gate — matching S1's word-granular slot zeroing recovers the
full win; that build (`9184bc59…`) was discarded and its logs preserved
outside the repo. Scope caveat: unchanged from the rejected-B2 entry above —
on the shipped DSv4 body the 11 FP8-CB routed layers run the bridge
(per-role split books), so this win reaches DSv4 only after the pooled-books
reburn; today it applies to any MoE artifact whose FP8-CB routed layers take
the persistent-B arm.

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
  residency — set `PRISMAQUANT_PRELOAD_FUSED=1` on both arms, which now warms every
  native extension family, not only the two fused ones. This is why the 27B
  confident-KL is quoted as a range under both readings.
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
