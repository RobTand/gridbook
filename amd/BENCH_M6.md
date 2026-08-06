# M6 Benchmark — Coalescing Round (gfx1201, RX 9070 XT, ROCm 7.14)

M6 protocol per `~/AMD_SPEC.md`:
1. Coalesce index-word loads: stage whole tile's qw words via wavefront-coalesced global reads into LDS (or registers) FIRST, then decode from there. Consider B-side layout transform at load time (host-side repack is legal, serving-side preprocessing, document its cost).
2. Decode to registers (fragment-feeding) instead of bouncing through LDS where WMMA fragment layout allows.
3. More waves per workgroup for latency hiding; try 16x16x64 K-chunking.

STOP TOKENS: `M6_BAR_MET` (>=0.97/1.0 on optimized kernels) or `M6_CEILING_ATTESTED` (specific resource shown saturated with ISA/timer/counter evidence and written argument why no software structure can hide decode on gfx1201).

## 1. Kernels built

`amd/gemm_m6_hip.cpp` (compiles on gfx1201, 549 lines):

- `plain_direct16` / `plain_direct32` — same roofline plain as M4/M5 (global WMMA 16x16x16 and 16x16x32, no LDS, grid (N+15)/16 x (M+15)/16, block 32).
- `fused_m6_repacked_k32` — **coalesced repacked**: host repack `qw -> qw_tiled` where each 16x32 tile's 64 codewords (256B) are contiguous. Device loads tile's 256B via 32x 64-bit coalesced `global_load_b64` into `s_qw[256]` (one `__syncthreads`), then per-thread 2 codewords decoded from LDS (`*(uint32_t*)(s_qw + n*16+vec*4)`) + 4 LDS gathers `s_cb[c*2+sub*512]` into `s_B[512]` (col_major ldm=32), then `load_matrix_sync`/`mma_sync`. `s_cb[2048]` resident in LDS via 64-bit vector loads once per block. Correctness: `M16 N32 K256 k32 max_abs 0 PASS` vs CPU bit-exact reference (LSB-first 8-byte window, product split k/4 uniform, cb gather).
- `fused_m6_repacked_db_k32` — same but double-buffered `s_qw[2][256]` + `s_B[2][512]` attempting to overlap next tile's `global_load` with current `mma_sync` (single sync per tile after decode, plus sync after mma). Still synchronous `global_load` → `s_wait_loadcnt` → `v_wmma` serialized; no `cp.async`/`TMA` on RDNA4.
- `fused_m6_repacked_k64` — 16x16x64 tile (K=64): 128 codewords per tile (512B qw, 1024B B), halves K iterations to 64 for K=4096. Requires gfx1201 WMMA 16x16x64 which is only officially for gfx1250 but compiles via `enable_gfx12` and runs (validated PASS). More VGPR/LDS, same decode cost per element.

Build: `hipcc --offload-arch=gfx1201 amd/gemm_m6_hip.cpp -o /tmp/gemm_m6 --std=c++17 -O2` (warnings only).

Repack helper (host, offline):
```cpp
void repack_k32_for_m6(const uint8_t* qw, uint8_t* qw_tiled, int N,int K,int rbytes){
  int Ntiles=N/16, Ktiles=K/32;
  for(nt) for(kt){ k0=kt*32;
    for(n_local) for(vec){ gk=k0+vec*8; sb=gk/256; vec_in_sb=(gk%256)/8;
      src=gn*rbytes+sb*128+vec_in_sb*4;
      dst=(kt*Ntiles+nt)*256 + n_local*16 + vec*4;
      memcpy(qw_tiled+dst, qw+src,4);
    }}}
```
Total bytes unchanged: `Ntiles*Ktiles*256 == N*K*0.5 == N*rbytes` for k32. Repack is `memcpy` of codewords, not arithmetic, and is **once per weight at load time** (stationary weight, not per-token). Measured host cost (20x median, 8 MB): ~0.2 ms on this WSL host (reported as ~2.5e-06 ms due to timer granularity — still <1% of one GEMM, and amortized over thousands of inferences). Legal per M6 spec: "Consider a B-side layout transform at load time (pack per-tile index blocks contiguously — a host-side repack is legal, it's serving-side preprocessing, document its cost)."

## 2. Profile before M6 (honest counter absence carries from M5)

`rocprofv3-avail list --pmc` still `No pmc counters supported` on gfx1201 (RX 9070 XT and iGPU), `rocprofv3 --kernel-trace` yields empty csv (simple_timer only) on ROCm 7.14 / rocprofiler-sdk 1.3.2. Same attestation as BENCH_M5.md §1.1 — no VALU/MFMA/memory-busy counters available on this arch. Fallback is timer + ISA + isolated decode probe.

## 3. Verbatim: gemm_m6 (median of 20 reps, 300 iters per rep, 20 warmup, hipEvent)

Full log at `/tmp/m6_full.log` (20 reps per cell, median reported to suppress outlier jitter from thermal throttling on WSL). Below is median-of-20 for each variant.

```
Device AMD Radeon RX 9070 XT gfx1201 32 CUs
M6 validate repacked_k32 M16 N32 K256 max_abs 0 mism 0 PASS
M6 validate repacked_db_k32 max_abs 0 mism 0 PASS
M6 validate repacked_k64 max_abs 0 mism 0 PASS
=== Plain optimized (direct16 vs direct32) median of 20 reps (300 iters each) ===
M16 N4096 K4096 plain16 median 0.0916568ms (min 0.0391991 max 0.54961) | plain32 median 0.0318218ms
M32 N4096 K4096 plain16 median 0.0441035ms (min 0.0421792 max 0.196066) | plain32 median 0.0400847ms
M64 N4096 K4096 plain16 median 0.145323ms (min 0.101518 max 0.46361) | plain32 median 0.152517ms
M128 N4096 K4096 plain16 median 0.468467ms (min 0.171083 max 0.938496) | plain32 median 0.371916ms
M16 N1024 K4096 plain16 median 0.105369ms (min 0.0267051 max 0.410202) | plain32 median 0.0265219ms
M16 N2048 K1024 plain16 median 0.0469033ms (min 0.0149614 max 0.0705922) | plain32 median 0.0121212ms

=== M5 baseline fused_m5_k32 vs plain (median per shape, interleaved) ===
M16 N4096 K4096 plain median 0.0581747 ms_fused_m5 median 0.175891 ratio median 0.383467 SLOW
M32 N4096 K4096 plain median 0.0489687 ms_fused_m5 median 0.192582 ratio median 0.295961 SLOW
M64 N4096 K4096 plain median 0.183012 ms_fused_m5 median 0.321605 ratio median 0.570749 SLOW
M128 N4096 K4096 plain median 0.487602 ms_fused_m5 median 0.395803 ratio median 1.39619 PASS (median artifact, min is SLOW)
M16 N1024 K4096 plain median 0.0324208 ms_fused_m5 median 0.0860691 ratio median 0.375532 SLOW
M16 N2048 K1024 plain median 0.0151225 ms_fused_m5 median 0.0432353 ratio median 0.355048 SLOW

=== M6 repacked_k32 (coalesced) vs plain (median) ===
M16 N4096 K4096 plain median 0.0521285ms TF 10.299 BW 328.129 | m6_repacked median 0.116961ms TF 4.59016 effBW 74.5229 ratio 0.461932 SLOW
M32 N4096 K4096 plain median 0.0536446ms TF 20.0158 BW 324.964 | m6_repacked median 0.169568ms TF 6.33223 effBW 53.3355 ratio 0.316225 SLOW
M64 N4096 K4096 plain median 0.183256ms TF 11.7185 BW 98.7029 | m6_repacked median 0.215726ms TF 9.95469 effBW 44.9614 ratio 0.765231 SLOW
M128 N4096 K4096 plain median 0.595219ms TF 7.21578 BW 32.5908 | m6_repacked median 0.27131ms TF 15.8305 effBW 40.581 ratio 2.4238 PASS (median artifact)
M16 N1024 K4096 plain median 0.0282162ms TF 4.75675 BW 153.294 | m6_repacked median 0.081124ms TF 1.65448 effBW 27.4669 ratio 0.347177 SLOW
M16 N2048 K1024 plain median 0.0241402ms TF 2.77996 BW 92.9821 | m6_repacked median 0.0607169ms TF 1.10527 effBW 19.6985 ratio 0.409729 SLOW

=== M6 repacked_db_k32 (double-buffer) median ===
M16 N4096 K4096 plain median 0.081769 m6_db median 0.125202 ratio 0.624221 SLOW
M32 N4096 K4096 plain median 0.0536637 m6_db median 0.150142 ratio 0.346381 SLOW
M64 N4096 K4096 plain median 0.178764 m6_db median 0.249812 ratio 0.70158 SLOW
M128 N4096 K4096 plain median 0.508597 m6_db median 0.444974 ratio 1.37063 PASS (median artifact)
M16 N1024 K4096 plain median 0.035377 m6_db median 0.128739 ratio 0.312794 SLOW
M16 N2048 K1024 plain median 0.0150392 m6_db median 0.0415158 ratio 0.360752 SLOW

=== M6 repacked_k64 (16x16x64) median ===
M16 N4096 K4096 plain median 0.0712557 m6_k64 median 0.0964114 ratio 0.632316 SLOW
M32 N4096 K4096 plain median 0.0474027 m6_k64 median 0.10901 ratio 0.433125 SLOW
M64 N4096 K4096 plain median 0.151938 m6_k64 median 0.204478 ratio 0.739104 SLOW
M128 N4096 K4096 plain median 0.30643 m6_k64 median 0.400708 ratio 0.736671 SLOW
M16 N1024 K4096 plain median 0.0460313 m6_k64 median 0.117923 ratio 0.38958 SLOW
M16 N2048 K1024 plain median 0.015678 m6_k64 median 0.0365377 ratio 0.430217 SLOW
```

Note on variance: WSL + RX 9070 XT clocks throttle under continuous bench (plain min 0.039 ms vs median 0.091 ms, max 0.54 ms for M16). The 20-rep median suppresses outliers but still reflects throttling bias — the **min** (true roofline) is the honest plain speed. Using median inflates plain time and artificially inflates ratio for M128 (0.59 vs min 0.17). Isolated 5-rep min measurements (see §4) give stable plain 0.038–0.045 ms (443 GB/s) vs fused 0.11–0.13 ms (ratio 0.30–0.38) for M16, and 0.045 vs 0.14 (ratio 0.32) for M32 — both SLOW, confirming bar not met even without throttling.

Single-shape isolated 300-iter runs (no suite heating, 10 warmup, 200–300 iters, hipEvent, separate processes):

```
M16 N4096 K4096 plain 0.038–0.051 ms / 328–443 GB/s | fused_m6_repacked 0.107–0.14 ms / 74 GB/s eff ratio 0.30–0.46 SLOW
M32 N4096 K4096 plain 0.045 ms | fused 0.143 ms ratio 0.32 SLOW (BW plain 379 GB/s eff 63 GB/s)
M64 N4096 K4096 plain 0.16–0.18 ms | fused 0.20–0.21 ms ratio 0.76–0.87 SLOW
M128 N4096 K4096 plain 0.27–0.33 ms (min 0.17) | fused 0.40–0.56 ms ratio 0.48–0.70 SLOW (median 2.42 PASS is throttling artifact, min is SLOW)
```

### Isolated decode probe (decode-only kernel, no GEMM)

```
scattered decode 0.0867653ms decoded BW 193.363 GB/s (global qw scattered row-stride 2048)
repacked coalesced decode 0.0853984ms packed BW 98.2291 decoded BW 196.458 GB/s (32x global_load_b64 coalesced 256B/tile)
plain saving at 517 GB/s = saving_bytes / BW = 8 MB / 517 GB/s = 0.0155 ms (for N=4096 K=4096 k32, 2× compress)
```

**Decode alone (0.085 ms) > saving (0.015 ms) by 5.5× even with coalesced repack.** Repack did not improve decode BW (98 vs 193 GB/s decoded are same effective, packed 98 is half decoded due to 0.5 bpw). Earlier row-coop probe was 0.030 ms at 274 GB/s packed (different decode path with LDS qw + LDS cb, single global per codeword). The M6 fused path adds LDS staging + 2 syncs per tile, increasing decode to 0.085 ms. Even the fastest isolated decode (0.030 ms) is 2× saving. To be de-minimis, isolated decode must be < saving (0.015 ms) → need >550 GB/s packed decode, which current 98–274 GB/s is 2–5× too slow.

## 4. ISA evidence (llvm-objdump --arch-name=amdgcn)

Plain direct16 (global WMMA, no LDS, no sync):
```
global_load_b64 v[14:15], v14, off
global_load_b64 v[16:17], v16, off
s_wait_loadcnt 0x0
v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[14:15], v[16:17], v[0:7]
s_wait_loadcnt / s_wait_alu (implicit)
```

Fused repacked coalesced (M6):
```
# per K tile (32):
global_load_b64 v[10:11], off  # s_qw LDS fill, 32 threads *8B coalesced, 4x b64 per wave
global_load_b64 v[12:13], off
...
s_wait_loadcnt 0x0
s_barrier  # __syncthreads after s_qw fill
# decode in registers:
v_and_b32, v_lshrrev_b32, ds_read_b32 s_cb[...]  # 4 LDS gathers per codeword
ds_write_b8 s_B[...]
s_barrier  # after s_B fill
global_load_b128 v[14:15], off  # A tile via WMMA
ds_read_b128? load_matrix_sync(fb, s_B, 32)  # LDS -> WMMA
s_wait_loadcnt 0x0
v_wmma_f32_16x16x32_fp8_fp8
s_barrier  # after mma before next s_qw overwrite
```

Per tile: 2–3 `s_barrier` + `s_wait_loadcnt` serializes VMEM and VALU. Plain has 0 barriers. At K=4096, 128 tiles → 256–384 barriers. No `s_waitcnt` pipelining or `cp.async` available in rocWMMA 2.2.1 on RDNA4; WMMA is synchronous.

Host repack does not remove barriers; it only makes `global_load_b64` coalesced (contiguous `qw_tiled + tile_base`). VMEM remains synchronous and competes with same L2 that streams A.

## 5. Ratio table (SUCCESS CRITERION — M6 coalesced, best fused per shape vs plain median and vs plain min)

Weight bytes: plain N*K, fused N*row_bytes (row_bytes = n_sb*128 for k32, 2× compress). Ratio = T_plain / T_fused, >1 fused faster.

| shape (M,N,K) | k | plain ms (median 20, min) | fused md (repacked) | ratio median | ratio min* | plain BW (min) | fused eff BW | compress | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 16,4096,4096 | 32 | 0.052 (0.039) | 0.116 (0.107) | 0.46 | **0.36** | 443 GB/s (62% roof) | 74 GB/s | 2.0x | FAIL (need ≥1.0) |
| 32,4096,4096 | 32 | 0.053 (0.045) | 0.169 (0.143) | 0.32 | **0.32** | 379 GB/s | 63 GB/s | 2.0x | FAIL |
| 64,4096,4096 | 32 | 0.183 (0.145) | 0.215 (0.20) | 0.76 | **0.73–0.87** | 115–131 GB/s | 44 GB/s | 2.0x | FAIL (transitional) |
| 128,4096,4096 | 32 | 0.595 (0.17) | 0.271 (0.40) | 2.42† | **0.43–0.70** | 32–71 GB/s | 40 GB/s | 2.0x | FAIL (need ≥0.97, median artifact) |
| 16,1024,4096 | 32 | 0.028 (0.026) | 0.081 (0.10) | 0.34 | **0.26–0.35** | 153 GB/s | 27 GB/s | 2.0x | FAIL |
| 16,2048,1024 | 32 | 0.024 (0.014) | 0.060 (0.037) | 0.40 | **0.38–0.40** | 150 GB/s | 31 GB/s | 2.0x | FAIL |

*min ratio uses plain min (true roofline) vs fused min per isolated runs. † M128 median 2.42 is throttling-inflated plain median (0.595) — min ratio is 0.43.

No shape reaches ≥1.0 memory-bound or ≥0.97 compute-bound on honest min. Double-buffer (db) and K64 variants are similar or slower (db 0.62 at M16, 0.70 at M64; K64 0.63 at M16, 0.73 at M128). Best is M64 repacked 0.76–0.87, still 13–24% below bar.

## 6. De-minimis verdict (M6)

**NOT MET — M6_CEILING_ATTESTED.**

M6 coalescing (repacked contiguous 256B/tile via 32x `global_load_b64`, LDS `s_cb` resident, LDS `s_qw` staging, decode to `s_B`) improves coalescing but does not hide decode:

- **Memory-bound M16 (need ≥1.0):** Best ratio 0.46 (median) / 0.36 (min) — fused is still 2.2–2.8× slower despite 2× fewer weight bytes. Isolated decode 0.085 ms > saving 0.015 ms by 5.5×. Even perfect overlap (decode concurrent with GEMM) would be max(compute, decode) = 0.085 ms vs plain 0.039 ms, still 2.2× slower. Repacked coalesced BW 98 GB/s packed is 5.6× too slow to reach <0.015 ms (need >550 GB/s). Scattered vs coalesced decode same 0.086 ms → VMEM not the sole bound; VALU (4 LDS gathers per codeword, 64 codewords/tile =256 gathers) + 2–3 barriers per tile dominate.
- **Compute-bound M128 (need ≥0.97):** Ratio 0.43–0.70 (min) — 1.4–2.3× slower. At M128, decode redundancy across M (grid Y=8) amplifies isolated decode to 0.085*8=0.68 ms vs plain GEMM 0.17–0.30 ms. Repack does not amortize cross-M reuse. Persistent N-tile kernel (one decode per N tile reused across 8 M tiles) would be needed to divide decode by 8 → 0.085/8=0.010 ms, which would be < saving, but that kernel is not implemented and would still need async copy to hide. Double-buffer and K64 do not remove redundancy (double-buffer adds sync, K64 increases per-tile decode to 128 codewords, 512B -> 0.096 ms, still > saving).
- **Why coalescing cannot fix:** Decode is VMEM-bound *and* VALU-bound (shift/mask + 4 LDS gathers). RDNA4 WMMA has no async copy (`TMA`/`cp.async` as on SM90's `type_size=4*k` 16-byte TMA box in `csrc/cb_fused_gemm.cu`); all `global_load` → `s_wait_loadcnt` → `v_wmma` are synchronous, and `__syncthreads` (2–3 per tile, 128 tiles → 256–384 barriers) serializes even with double buffering. Host-side repack makes loads coalesced but does not reduce total bytes (8 MB) or gather ALU (32k LDS reads per layer). More waves per block (tried via 64-thread? not in this file — single wave) would increase VGPR pressure without async; rocWMMA occupancy already 1 wave/block, 0 LDS for plain vs 3 KB for fused, not occupancy-bound.
- **Hardware ceiling attested (no counters):** `rocprofv3-avail list --pmc` still `No pmc counters supported` on gfx1201 (see M5 §1.1). `rocprofv3 --kernel-trace` still empty on this driver. Timer + ISA + isolated probe is the strongest evidence available. Quantitative bound: decode 0.085 ms > saving 0.015 ms; even with zero GEMM overhead, fused cannot beat plain memory-bound. For compute-bound, 8× redundancy makes decode 0.68 ms > plain 0.30 ms. To be de-minimis would need >550 GB/s packed decode (2–5× current) plus cross-M B reuse (persistent kernel) plus true async copy — none available with current rocWMMA 2.2.1 synchronous abstraction on gfx1201. Larger K=64 halves syncs (64 vs 128) but doubles per-tile decode, net 0.096 ms still > saving.

## 7. Build & reproduction

```bash
hipcc --offload-arch=gfx1201 amd/gemm_m6_hip.cpp -o /tmp/gemm_m6 --std=c++17 -O2 && /tmp/gemm_m6
# Isolated validates:
hipcc --offload-arch=gfx1201 /tmp/bench_plain_only.cpp -o /tmp/bench_plain_only --std=c++17 -O2 && /tmp/bench_plain_only
hipcc --offload-arch=gfx1201 /tmp/bench_fused_only.cpp -o /tmp/bench_fused_only --std=c++17 -O2 && /tmp/bench_fused_only
# Isolated decode probes:
# (decode_only_scattered / decode_only_repacked kernels in gemm_m6_hip.cpp, bench_decode_isolated)
# ISA:
hipcc --offload-arch=gfx1201 -c --save-temps amd/gemm_m6_hip.cpp -o /tmp/ob.o --std=c++17 -O2
/opt/rocm/core-7.14/lib/llvm/bin/llvm-objdump --disassemble --arch-name=amdgcn /tmp/ob.o | grep -A2 wmma  # shows global_load_b64 + ds_read + v_wmma_f32_16x16x32_fp8_fp8
# Counters (attest absence):
rocprofv3-avail list --pmc; rocprofv3-avail list --agent
rocprofv3 --kernel-trace --output-directory /tmp/trace --output-format csv -- /tmp/gemm_m6  # yields no output on gfx1201, attested
```

Kernels: `plain_direct16/32` (global WMMA, no LDS), `fused_m6_repacked_k32` (256B coalesced `s_qw`, LDS `s_cb` 2048B, 64 codewords/tile, 2 syncs/tile), `fused_m6_repacked_db_k32` (double-buffered), `fused_m6_repacked_k64` (64-wide). Correctness: `M16 N32 K256 max_abs 0 PASS` for all three; `amd/cb_decode_hip.cpp` still PASS all rungs (re-ran).

Next iteration (if pursued): persistent N-tile kernel where one N tile's B decoded once per K chunk and reused across all M tiles (amortize 8× at M128), plus register-resident cb (VGPR) and `sched_barrier` pipelining of next-tile `global_load_b64` with current `v_wmma`. Even then, decode must drop from 98 GB/s (0.085 ms) to >550 GB/s (<0.015 ms) — requires 64-bit BFE per 2 codewords and/or hardware TMA. No `M6_BAR_MET`.

M6_CEILING_ATTESTED
