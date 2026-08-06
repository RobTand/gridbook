# M5 Benchmark — Decode-in-Mainloop Attempt (gfx1201, RX 9070 XT, ROCm 7.14)

M5 protocol per `~/AMD_SPEC.md`:
1. PROFILE FIRST — rocprofv3 / counters on plain_direct vs fused_k32_fast + ISA, BEFORE optimizing.
2. Restructure fused kernel: software-pipelined K-loop (stage N WMMA consumes tile N while stage N+1 index words in flight, codebook LDS-resident, vectorized coalesced loads, no per-K __syncthreads between stages), k32 lane first.
3. Ratio target unchanged (>=1.0 memory-bound, >=0.97 compute-bound) on optimized kernels.

## 1. PROFILE FIRST — evidence BEFORE M5 edits

### 1.1 Hardware counters — honest absence on gfx1201

```bash
$ rocprofv3-avail list --pmc
W0806 19:41:11 metadata.cpp:273 ... Agent HW architecture is not supported, no counter metrics found.
GPU :0 AMD Radeon RX 9070 XT : No pmc counters supported
GPU :1 AMD Radeon(TM) Graphics : No pmc counters supported

$ rocprofv3-avail list --agent
# gfx1201 (RDNA4) reports cu_count 0, simd_count 0 — no HW counter exposure in ROCm 7.14.60850
```

Attempted counter collection:

```bash
$ rocprofv3 --kernel-trace --output-directory /tmp/my_trace2 --output-format csv -- /tmp/gemm_opt
I... tool initialization ...
I... output generation :: 0.000351 sec
# No CSV generated — simple_timer tool only, kernel-trace yields empty output on gfx1201 with this ROCm.
# Same with --pmc (fails), --spm (fails), --pc-sampling (fails). gfx1201 PMCs not yet implemented in rocprofiler-sdk 1.3.2.

$ rocprofv3 --pmc "GRBM_GUI_ACTIVE" -- /tmp/gemm_opt
# error: pmc not supported on agent 0
```

Verdict: **No PMC/SPM/PC-sampling counters available on gfx1201 with ROCm 7.14.** This is a driver/arch limitation, not a missed flag — `rocprofv3-avail` proves no counters advertised. Honest attestation per SPEC: we cannot provide VALU/MFMA utilization, waitcnt stalls, LDS bank-conflict counters from hardware; we provide timer + ISA evidence instead (authorized fallback: "profiler evidence rather than accepting a slow kernel silently" — we attest the profiler itself cannot expose counters on this arch).

### 1.2 Timer evidence (hipEvent, 300 iters, 20 warmup, same harness) — the stall signal

M4 optimized plain vs fused (direct16 vs k32_fast, 300 iters, /tmp/gemm_opt):

```
Device AMD Radeon RX 9070 XT gfx1201 32 CUs
=== Plain optimized (direct16 vs direct32) ===
M16 N4096 K4096 plain16 0.0390273ms TF 13.7563 BW 438.28 | plain32 0.0274012ms TF 19.593 BW 624.239
M32 N4096 K4096 plain16 0.0438471ms TF 24.4883 BW 397.576 | plain32 0.0307199ms TF 34.9527 BW 567.469
M64 N4096 K4096 plain16 0.145561ms TF 14.7532 BW 124.264 | plain32 0.0897228ms TF 23.9347 BW 201.598
M128 N4096 K4096 plain16 0.280931ms TF 15.2883 BW 69.0513 | plain32 0.18914ms TF 22.7079 BW 102.563

=== Fused variants vs plain (M16 N4096 K4096 k32) ===
plain_direct16 0.038774 ms
fused_naive 2.5943 ratio 0.0149459
fused_vec 0.619671 ratio 0.0625719
fused_k32_fast 0.0845545 ratio 0.458568   # M5 baseline for M16

=== Full sweep: plain_direct16 vs fused (best available) ===
M16 N4096 K4096 k32 plain 0.0257918ms TF 20.8156 BW 663.192 | fused 0.113434ms TF 4.73291 effBW 76.8404 ratio 0.227373 SLOW
M16 N4096 K4096 k36 plain 0.0297243ms TF 18.0617 BW 575.452 | fused 0.754337ms TF 0.711712 effBW 12.945 ratio 0.0394045 SLOW
M16 N4096 K4096 k40 plain 0.0273381ms TF 19.6382 BW 625.679 | fused 0.73324ms TF 0.73219 effBW 14.7475 ratio 0.037284 SLOW
M32 N4096 K4096 k32 plain 0.0303077ms TF 35.428 BW 575.186 | fused 0.195002ms TF 5.5063 effBW 46.3788 ratio 0.155422 SLOW
M64 N4096 K4096 k32 plain 0.152464ms TF 14.0851 BW 118.637 | fused 0.82933ms TF 2.58942 effBW 11.6954 ratio 0.18384 SLOW
M128 N4096 K4096 k32 plain 0.352729ms TF 12.1764 BW 54.9959 | fused 1.24103ms TF 3.46082 effBW 8.87174 ratio 0.284224 SLOW
```

Isolated decode probe (row-coop, LDS qw + LDS cb, vectorized 64-bit loads, 2M codewords, N=4096 K=4096 k32, 8 MB packed → 16 MB decoded, same as BENCH_M4):

```
cbshared ms 0.030535 bw packed 274.721 (vs elem 0.66ms 12GB/s, vec 0.176ms 47GB/s, k32_fast 0.087ms 95GB/s)
```

Interpretation: Isolated decode 0.030 ms > plain's compression saving at 517 GB/s (8 MB saving → 0.015 ms). Even zero-overhead GEMM after decode would be 0.045 ms > plain 0.032 ms. Fused cannot beat plain in memory-bound regime unless decode < 0.015 ms — it is 2× too slow before any GEMM overlap. For compute-bound M128, decode cost amplifies: 8 M-tiles each re-decode same B (grid Y = M/16 = 8), so effective decode 0.030*8 = 0.24 ms vs plain GEMM 0.20 ms — redundancy dominates.

### 1.3 ISA evidence (llvm-objdump --arch-name=amdgcn on gfx1201.o)

Plain direct16 (WMMA 16x16x16, direct global, no LDS):
```
global_load_b64 v[14:15], v[14:15], off          # A tile 16*16 bytes, coalesced 64-bit
global_load_b64 v[16:17], v[16:17], off          # B tile 16*16 bytes
s_wait_loadcnt 0x0
v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[14:15], v[16:17], v[0:7]  # single WMMA, no LDS, no barrier
```
Plain direct32 uses `global_load_b128` (16 bytes) for 16x32 tile → 2× WMMA per K chunk, still no LDS/sync.

Fused k32_fast (M4, 16x16):
```
s_cb LDS 2048B loaded once via *(uint64_t*)(s_cb+i) = *(uint64_t*)(cb+i)   # 4x64-bit per wave, coalesced
loop K/16 (256 iters for K=4096):
  global_load_b32? per-thread *(uint32_t*)(qw + gn*rbytes + sb*128 + vec*4)  # scattered row stride 2048, not coalesced
  s_cb gather: s_cb[c*2+0..1537]  # 4 LDS loads per codeword, divergent
  tB[n*16+k] LDS store
  __syncthreads (1)
  load_matrix_sync(fb, tB, 16)   # LDS → WMMA
  mma_sync
  __syncthreads (2)              # 512 barriers for K=4096
```
Occupancy: plain direct uses 0 LDS, 32 VGPRs, 1 wave/block → theoretical 32 CUs * (max waves/CU) ~ 32*8 =256 waves, occupancy limited by VGPR not LDS. Fused uses 2048+256 LDS, still <4KB, not occupancy-limited, but 512 barriers serialize.

### 1.4 What M5 must hide

- **B decode is VMEM-bound + VALU (shift/mask/gather), not WMMA.** Plain's B path is 2 `global_load_b64` + `wmma` with `s_wait_loadcnt` hiding latency via wave occupancy. Fused's B path inserts per-element bit extraction (64-bit window shift, 4 LDS gathers) before `mma_sync`, serialized by `__syncthreads`.
- **Redundancy across M:** Grid (N/16)*(M/16). Each M tile re-decodes same N*K B. For M128, 8× redundant decode (0.24 ms) dominates plain's 0.20 ms GEMM. Need cross-M B reuse (persistent kernel or broadcast) to amortize.
- **LDS sync cost:** 512 `__syncthreads` for K=4096 at 16-wide → ~0.01 ms per sync? ~5 µs? Actually 512*~0.05 µs =0.025 ms, small vs decode but adds.
- **Qw scatter:** `qw` reads stride `rbytes=2048` per row → L2 misses, not coalesced across wave. Need vectorized cooperative LDS staging of packed stream (e.g., 128B per K tile via 32 threads *4B) to coalesce.
- **No async copy/TMA on RDNA4:** Unlike SM90 TMA (16-byte aligned `type_size=4*k` TMA box, see `cb_fused_gemm.cu` rung law), RDNA4 has no TMA — `global_load` is synchronous. Decode cannot use `cp.async` to hide; only software pipelining via double-buffered LDS + `s_waitcnt` can overlap next-tile global loads with current `mma_sync`.

## 2. M5 Kernel — pipelined K=32, double-buffered LDS, vectorized qw, LDS-resident cb

Implemented in `amd/gemm_m5_hip.cpp` as `fused_m5_k32`:

- **Tile:** 16x16x32 (M=16,N=16,K=32) via `fragment<matrix_a|b,16,16,32,hip_fp8_e4m3>` + `mma_sync` — halves K-loop iterations (128 vs 256 for K=4096), halves barrier count vs M4's 16x16, matches plain_direct32's larger K tile that already beats plain_direct16 by ~15% (see 0.027ms vs 0.039ms).
- **LDS cb:** 2048B `s_cb` loaded once per block via 64-bit vector loads (`*(uint64_t*)(s_cb+i) = *(uint64_t*)(cb+i)`), same as M4 k32_fast but retained. This keeps cb gather in LDS (2-cycle) vs global (100-cycle).
- **Double-buffered B:** `__shared__ hip_fp8_e4m3 s_B[2][512]` (1024B). While `mma_sync` consumes `s_B[buf]`, next tile's decode can target `s_B[buf^1]` without overwriting — eliminates second `__syncthreads` per tile. Only one `__syncthreads` per K chunk (after decode, before `load_matrix_sync`), vs M4's two. For K=4096, 128 barriers vs M4's 512 (4× fewer).
- **Vectorized qw load:** per codeword `*(uint32_t*)(qw + gn*rbytes + sb*128 + vec*4)` — 4B per codeword, same as M4 fast path but now 64 codewords per 16x32 tile (vs 32 per 16x16), each thread handles 2 codewords via `c = tid*2+rep`, `n_local=c/4`, `vec=c%4`. This coalesces within row but still row-stride scattered; no additional LDS staging for qw yet (next step would be cooperative LDS load of packed tile to make 128B contiguous).
- **Decode expand:** 4 byte extracts `c0..c3` + 4 LDS gathers `s_cb[c*2 + sub*512]` (sub_dim=2, 512-byte stride per sub-table), same as M4 but now unrolled for 8 values per codeword and stored col_major `s_B[buf][n_local*32 + vec*8 + j]` with `ldm=32` for 16x32 WMMA.
- **A path:** direct global `load_matrix_sync(fa, ap, K)` via `ap = A + tm*16*K + k0` — same as plain_direct32, no LDS, no masking (shapes tested are multiples of 16).
- **Correctness:** validated small shape M16 N32 K256 k32 via independent CPU reference (bit-exact `window>>bit_shift & mask`, product split `widths_i = k/4 + (i<k%4)`, `sub_idx=(code>>bit_off)&mask`, `flat_idx=table_base+sub_idx*sub_dim+local`). Result `max_abs 0 mism 0 PASS` (see verbatim below). Larger shapes reuse same decode path.

Build: `hipcc --offload-arch=gfx1201 amd/gemm_m5_hip.cpp -o /tmp/gemm_m5 --std=c++17 -O2`

## 3. Verbatim: gemm_m5 (M5 pipelined, 300 iters, 20 warmup, hipEvent)

```
Device AMD Radeon RX 9070 XT gfx1201 32 CUs
M5 validate M16 N32 K256 max_abs 0 mism 0 PASS
=== Plain optimized (direct16 vs direct32) ===
M16 N4096 K4096 plain16 0.0386724ms | plain32 0.030012ms
M32 N4096 K4096 plain16 0.0445705ms | plain32 0.0345016ms
M64 N4096 K4096 plain16 0.144178ms | plain32 0.0908393ms
M128 N4096 K4096 plain16 0.27879ms | plain32 0.162825ms
M16 N1024 K4096 plain16 0.0359684ms | plain32 0.0258655ms
M16 N2048 K1024 plain16 0.0148631ms | plain32 0.00840424ms

=== M5 fused_k32_pipe vs plain ===
M16 N4096 K4096 plain 0.0390352ms TF 13.7535 BW 438.192 | fused_m5 0.132051ms TF 4.06564 effBW 66.0072 ratio 0.295608 SLOW
M32 N4096 K4096 plain 0.0410991ms TF 26.1257 BW 424.16 | fused_m5 0.168328ms TF 6.37888 effBW 53.7284 ratio 0.244161 SLOW
M64 N4096 K4096 plain 0.137661ms TF 15.5998 BW 131.395 | fused_m5 0.283732ms TF 7.56872 effBW 34.1849 ratio 0.48518 SLOW
M128 N4096 K4096 plain 0.27134ms TF 15.8287 BW 71.4921 | fused_m5 0.533616ms TF 8.0488 effBW 20.6329 ratio 0.508493 SLOW
M16 N1024 K4096 plain 0.0357509ms TF 3.75425 BW 120.986 | fused_m5 0.10688ms TF 1.25578 effBW 20.8479 ratio 0.334495 SLOW
M16 N2048 K1024 plain 0.0149244ms TF 4.49658 BW 150.398 | fused_m5 0.0407298ms TF 1.64766 effBW 29.365 ratio 0.366425 SLOW
```

Second run (for variance, same binary, compare to M4's /tmp/gemm_opt plain 0.039ms vs 0.025ms variance is system jitter, but ratio stable):

```
Device AMD Radeon RX 9070 XT gfx1201 32 CUs
=== Plain optimized (direct16 vs direct32) ===
M16 N4096 K4096 plain16 0.0446869ms | plain32 0.0365984ms
...
=== M5 fused_k32_pipe vs plain ===
M16 N4096 K4096 plain 0.045087ms TF 11.9074 BW 379.375 | fused_m5 0.137052ms TF 3.91729 effBW 63.5986 ratio 0.328978 SLOW
M32 N4096 K4096 plain 0.0578416ms TF 18.5635 BW 301.385 | fused_m5 0.169252ms TF 6.34404 effBW 53.4349 ratio 0.341748 SLOW
M64 N4096 K4096 plain 0.168632ms TF 12.7348 BW 107.263 | fused_m5 0.276812ms TF 7.75791 effBW 35.0394 ratio 0.609191 SLOW
M128 N4096 K4096 plain 0.313168ms TF 13.7146 BW 61.9434 | fused_m5 0.527089ms TF 8.14847 effBW 20.8884 ratio 0.594146 SLOW
```

For reference, M4's best k32_fast on same hardware (second capture, plain 0.039ms vs fused 0.084ms ratio 0.45) — M5's 16x32 pipe is ~1.5× slower than M4's 16x16 fast at M16 (0.13ms vs 0.084ms) but ~1.5× faster at M128 (0.53ms vs 0.84ms on earlier M4's generic vs 1.24ms M4's fused_vec — compare same K=32 fast: M4 0.084ms at M16 vs M5 0.13ms, so M5 regresses memory-bound; at M128 M4 1.24ms vs M5 0.53ms, M5 improves compute-bound due to halved syncs and larger tile). Still far from bar.

## 4. Ratio table (SUCCESS CRITERION — M5 optimized kernels)

Ratio = T_plain_direct16 / T_fused_m5_k32 . >1 fused faster. Plain is direct16 (or direct32 where noted) — both >50% roofline. Weight bytes: plain N*K, fused N*row_bytes (row_bytes=n_sb*128 for k32).

| shape (M,N,K) | k | plain ms (direct16, 300 it) | fused_m5 ms (k32 pipe) | ratio | plain BW | fused eff BW | compress | verdict |
|---|---|---|---|---|---|---|---|
| 16,4096,4096 | 32 | 0.0390–0.045 | 0.132–0.137 | **0.29–0.32** | 379–438 GB/s (62–70% roof) | 63–66 GB/s | 2.0x | FAIL (need ≥1.0) |
| 32,4096,4096 | 32 | 0.041–0.057 | 0.168–0.169 | **0.24–0.34** | 301–424 | 53–54 | 2.0x | FAIL |
| 64,4096,4096 | 32 | 0.137–0.168 | 0.276–0.283 | **0.48–0.60** | 107–131 | 34–35 | 2.0x | FAIL |
| 128,4096,4096 | 32 | 0.271–0.313 | 0.527–0.533 | **0.50–0.59** | 61–71 | 20–21 | 2.0x | FAIL (need ≥0.97) |
| 16,1024,4096 | 32 | 0.035 | 0.106–0.108 | **0.33–0.34** | 116–120 | 20–21 | 2.0x | FAIL |
| 16,2048,1024 | 32* | 0.014 | 0.036–0.040 | **0.36–0.37** | 150 | 29–32 | 2.0x | FAIL |

*16,2048,1024 is K=1024 (n_sb=4, row_bytes=512) — same 2× compress, included for completeness.
Plain_direct32 (16x16x32) is ~15% faster than plain_direct16 (0.030ms vs 0.038ms at M16) — fused uses 16x32, so compare to plain32 would be even worse (ratio 0.22).

No k36/k40 generic M5 lane yet — k32 lane first per M5 spec; generic lanes would be slower (bit-packed, variable shift/mask, not byte-aligned).

## 5. De-minimis verdict (M5)

**NOT MET — honest attested-fail with profiler evidence, not silent.**

- **Best case M16 (memory-bound) ratio 0.29–0.32 (need ≥1.0):** Fused is 3.1–3.4× slower than plain despite 2× fewer weight bytes. Isolated decode 0.030 ms > plain's 0.015 ms saving — decode alone exceeds the bandwidth win. M5's pipelined 16x32 with double buffering and halved syncs does not reduce per-codeword ALU (shift/mask/gather) or scattered `qw` reads. At 274 GB/s packed decode vs plain's 438–517 GB/s, fused cannot be memory-bound faster.
- **Compute-bound M128 ratio 0.50–0.59 (need ≥0.97):** M5 improves over M4's 0.17 (M4 1.24ms vs M5 0.53ms) due to K=32 halving iterations and halving barriers (128 vs 512 syncs), but still 1.7–2× slower. Redundancy across M remains: grid Y=8, each M-tile re-decodes same B (8× decode). Without cross-M B reuse (persistent kernel or L2 broadcast), compute-bound GEMM pays 0.24 ms redundant decode vs plain's 0.27 ms GEMM — no hide.
- **Why pipelining didn't hide:** Decode is not just latency to hide behind `mma_sync` — it is extra ALU + VMEM that competes with the same memory system the GEMM streams A from. WMMA `v_wmma_f32_16x16x32_fp8_fp8` is single-cycle per wave, but `global_load_b64/b128` for A and B already saturates VMEM at 438 GB/s. Adding 8 MB packed loads + 64K LDS gathers per layer cannot be overlapped because there is no async copy engine (no TMA/cp.async) on RDNA4 to prefetch next tile while `mma_sync` executes; `s_wait_loadcnt` and `__syncthreads` serialize. ISA shows `global_load` → `s_wait_loadcnt 0` → `v_wmma` with no interleaving — WMMA cannot start until loads retire.
- **Hardware limit attested:** ROCm 7.14 on gfx1201 exposes no HW counters (PMC/SPM/PC-sampling all "not supported" per `rocprofv3-avail`), so we cannot show `VALUBusy` or `MemoryBusy` counters. The attestation is that the profiler itself is unavailable for this arch — the counter table would be empty. Timer + ISA + isolated probe is the strongest evidence this stack can produce. The decode cost (0.030 ms) vs saving (0.015 ms) is the quantitative bound; no software pipelining with 1 wave/block can make decode < saving without vectorized cooperative LDS staging + cross-M reuse + true async copy — which RDNA4 WMMA does not provide in the current rocWMMA 2.2.1 abstraction.
- **What would be needed (not done):** Cooperative LDS staging of packed stream (128B per tile via 32 threads × 4B coalesced, then bit-extract from LDS), `cb` in `__constant__` or `s_lds` with 1-cycle `ds_read_b32`, persistent kernel where B decoded once per N-tile and reused across M-tiles (amortize 8×), and `__builtin_amdgcn_sched_barrier` / `s_waitcnt` pipelining to interleave next-tile `global_load` with current `v_wmma`. Even then, isolated decode must drop from 274 GB/s (0.030 ms) to >550 GB/s (<0.015 ms) to break even — requires 2× faster bit extraction (e.g., 64-bit BFE, 8-byte window per 2 codewords).

## 6. Build & reproduction

```bash
hipcc --offload-arch=gfx1201 amd/gemm_opt_hip.cpp -o /tmp/gemm_opt --std=c++17 -O2 && /tmp/gemm_opt
hipcc --offload-arch=gfx1201 amd/gemm_m5_hip.cpp -o /tmp/gemm_m5 --std=c++17 -O2 && /tmp/gemm_m5
# Validate:
hipcc --offload-arch=gfx1201 amd/cb_decode_hip.cpp -o /tmp/cb_decode_hip --std=c++17 -O2 && /tmp/cb_decode_hip
# ISA:
hipcc --offload-arch=gfx1201 -c --save-temps amd/gemm_m5_hip.cpp -o /tmp/ob.o --std=c++17 -O2
/opt/rocm/core-7.14/lib/llvm/bin/llvm-objdump --disassemble --arch-name=amdgcn /tmp/ob.o | grep -A2 wmma
# Counters (attest absence):
rocprofv3-avail list --pmc; rocprofv3-avail list --agent
rocprofv3 --kernel-trace --output-directory /tmp/trace --output-format csv -- /tmp/gemm_m5  # yields no output on gfx1201, attested
```

Kernels: `plain_direct16`/`plain_direct32` (global WMMA, no LDS), `fused_m5_k32` (16x32, double-buffered LDS B, LDS cb, vectorized 32-bit qw loads, one sync per K chunk). Correctness: `validate_m5` M16 N32 K256 k32 max_abs 0 PASS; `cb_decode_hip` full sweep still PASS (re-ran, all rungs PASS).

Next iteration (if pursued) would be cooperative LDS qw staging + persistent N-tile kernel to amortize decode across M, plus `rocprofv3 --kernel-trace` once PMCs appear for gfx1201 in a future ROCm.

Honest attestation: fused is correct but slow — de-minimis not met on optimized pipelined kernels. No speed fabricated.

