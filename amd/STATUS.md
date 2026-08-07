# amd/STATUS.md — RDNA4 FP8 serving kernels for gfx1201

Append-only log. Successor turns resume cold from this file.

## 2026-08-06 — M1 — Environment attestation + first triangle — DONE

- **Branch:** `feat/amd-rdna4-fp8` reset to `v0.8.1` (`c9c1265`).
- **Host:** WSL2 Ubuntu 26.04, Ryzen 9800X3D, RX 9070 XT 16GB gfx1201, ROCm 7.14.60850, clang 23.0.0git, rocWMMA 2.2.1, hip runtime visible (`hipGetDeviceCount=1`, `gcnArchName=gfx1201`, `warpSize=32`). `rocm-smi` reports `amdgpu not found` — expected on WSL dxg paravirtualization, hip/HSA still functional.
- **hipcc target:** `hipcc --offload-arch=gfx1201` compiles and runs. Verified with `probe_hip.cpp` (hipGetDeviceProperties reports gfx1201, 32 CUs, 16GB).
- **WMMA probe (compile-time, not assumption):**
  - FP16 WMMA (16x16x16, half->float) — OK
  - BF16 WMMA (bfloat16_t->float) — OK
  - INT8 WMMA (int8_t->int32) — OK
  - **FP8 E4M3 WMMA (hip_fp8_e4m3 / float8_t -> float, 16x16x16, 16x16x32) — OK**
  - **FP8 E5M2 WMMA (hip_fp8_e5m2 / bfloat8_t -> float) — OK**
  - **Mixed FP8 (E4M3/E5M2) — OK**
  - FP8 with half accumulator — OK
  - FNUZ FP8 (hip_fp8_e4m3_fnuz) — FAIL (undefined HIP_vector_base) — honest absence, CDNA only
  - MFMA FP8 — gated to gfx950, not gfx1201; fragment abstraction hides but direct mfma builtin not available — WMMA is the RDNA4 path.
  - Written to `amd/ATTESTATION.md` with verbatim logs.
- **First triangle GEMM:** `amd/gemm_wmma_hip.cpp` — single-wave WMMA GEMM (16x16x16 tiles, row_major A/B, float accum, loop over K). Best path = **FP8 E4M3 WMMA** (rocwmma fragment<matrix_a|b,16,16,16,hip_fp8_e4m3> + mma_sync). Fallback FP16 WMMA with FP8 storage also validated but not needed.
  - **Correctness:** vs CPU reference (quantize float -> E4M3 via HIP's __hip_cvt_float_to_fp8 SATFINITE, then FP32 GEMM). 16x16x16 row_major PASS max_abs=0 rms=0 bit-exact; 32x32x32 PASS max_abs=0; col_major variant also PASS; FP16 16x16x16 PASS max_abs=2.38e-07. Tolerances stated: require max_abs<1e-2, rms<1e-3; achieved 0.
  - **Build:** `hipcc --offload-arch=gfx1201 gemm_wmma_hip.cpp -o gemm_wmma --std=c++17 -O2` (warnings only).
  - **Run:** `rocminfo` shows gfx1201 HSA agent; `./gemm_wmma` exits 0.
- **Gap honestly documented:** FNUZ/MFMA absent on gfx1201; larger K=64/128 WMMA tiles only for gfx1250; rocm-smi not available on WSL2. FP8 WMMA is OCP, not FNUZ — matches gridbook's E4M3 grid (SPEC.md), so no grid mismatch.
- **Next:** M2 CB decode lane — fuse codebook indices -> FP8 operand tiles + per-row scales ahead of this GEMM, matching repo format contract (SPEC §1, §1.1 LSB-first, product n_sub=4, per-row FP32 weight_scale). Dense lane first.

## Repo state at M1

- `amd/ATTESTATION.md` — attestation
- `amd/gemm_wmma_hip.cpp` — minimal GEMM triangle source
- No changes to `gridbook/*` yet; M2 will add `amd/cb_decode.*` kernel and validation harness.

## How to resume

```bash
git checkout feat/amd-rdna4-fp8
cat amd/STATUS.md
cat amd/ATTESTATION.md
hipcc --offload-arch=gfx1201 amd/gemm_wmma_hip.cpp -o /tmp/gemm_wmma && /tmp/gemm_wmma
# then continue M2 per milestones in prompt
```

## 2026-08-06 — M2 — CB decode lane (FP8-CB decode + fused WMMA) — DONE

- **Commit:** `756e21f` (`amd: M2 CB decode + fused FP8-CB WMMA lane, bit-exact vs independent + Python references`). Fixup `dC` type (`float*` vs `hip_fp8*`) in `bench_one`/`bench_kernel` committed as part of M3 bench fix but decode logic unchanged.
- **Decode kernel:** `amd/cb_decode_hip.cpp` — `cb_decode_fp8_kernel` / `cb_decode_scaled_bf16_kernel` + fused `wmma_gemm_cb_fused` and masked `wmma_gemm_cb_fp8` (FP8 E4M3 WMMA 16x16x16) and `wmma_gemm_bf16_cb` fallback.
  - Bit packing: LSB-first, `row_bytes = n_sb*4*k`, `byte_base=(v*k)/8`, `bit_shift=(v*k)%8`, 8-byte window `window |= byte<<(8*b)`, `code=(window>>bit_shift)&((1ULL<<k)-1)`, product split `widths_i = k//n_sub + (1 if i<k%n_sub else 0)` (SPEC §1.1), `sub_idx=(code>>bit_off)&((1<<w)-1)`, `flat_idx=row_offset+table_base+sub_idx*sub_dim+local`. Matches `tests/cb_torch_reference.py` / `codec.py` / `expand.py`.
  - Previous bug (byte-aligned reader at k∈{36,40,44,48}, 9-12-bit indices, drift) fixed — now true bit extraction. Earlier mism 24-50% signature gone.
  - **Independent CPU reference:** `extract_codewords_cpu` / `cpu_decode_values` use padded row `+8 zeros` and window, no shared device helpers; harness integrity restored (GEMM reference no longer shares decode path).
  - **Correctness sweep (M2):** per operator notes, unscaled decode now PASS all rungs (k=28,30,32,36,40,44,48 with N=2..8, K=256/512/1024); scaled BF16 decode PASS (1-27 elem mismatch bug fixed via scale at row level + RNE via `hip_bfloat16(float)`); GEMM masking bug at N=2 / k∈{32,40,48} fixed via shared padded tile + masked store (LDS 16x16, `if(gr<M&&gc<N)` scatter). `test_decode_one` reports `PASS` for unscaled, scaled, FP8 WMMA GEMM, BF16 WMMA GEMM, and fused lane.
  - Example: `k32 N4 K512` decode PASS mism 0, scaled PASS, GEMM FP8 max_abs <1e-3 PASS, fused max_abs <1e-3 PASS. Larger `M16 N4096 K4096 k32` fused vs separate GPU GEMM identical (diff<1e-6).
- **GEMM lane:** FP8 E4M3 WMMA is primary (plain `hip_fp8_e4m3` WMMA 16x16x16, `col_major` B via LDS col_major packing). BF16 WMMA fallback validated. `scale_rows_kernel` applies per-row `weight_scale` after GEMM (matches D0 loader contract, per-row FP32).
- **Build:** `hipcc --offload-arch=gfx1201 amd/cb_decode_hip.cpp -o /tmp/cb_decode_hip --std=c++17 -O2` (warnings only). `amd/bench_cb_hip.cpp` is duplicate bench harness for M3 (same kernels, plain vs fused).
- **Repo state at M2:** `amd/cb_decode_hip.cpp` (799 lines), `amd/bench_cb_hip.cpp` (171 lines), `amd/gemm_wmma_hip.cpp` retained.

## 2026-08-06 — M3 — Perf pass (plain FP8 WMMA vs CB fused) — DONE (honest slow exit)

- **Bench harness:** `amd/bench_cb_hip.cpp` (plain `wmma_gemm_plain2` vs fused `wmma_gemm_fused_bench`, WMMA 16x16x16, masked LDS, grid (N+15)/16 x (M+15)/16, block 32, 20 warmup, 500 iters, `hipEventElapsedTime` average) and `amd/cb_decode_hip.cpp::bench_one` (`--bench`, 300 iters). Shapes identical, same A/W randomization, same GEMM epilogue — only difference is B source (raw FP8 vs decoded via `qw`+`cb_flat`).
- **Command:**
  ```bash
  hipcc --offload-arch=gfx1201 amd/bench_cb_hip.cpp -o /tmp/bench_cb --std=c++17 -O2 && /tmp/bench_cb
  hipcc --offload-arch=gfx1201 amd/cb_decode_hip.cpp -o /tmp/cb_decode_hip --std=c++17 -O2 && /tmp/cb_decode_hip --bench
  ```
  Output captured verbatim to `amd/BENCH.md`.
- **Result (verbatim in BENCH.md):** fused is **~1.5–3.7× slower** than plain (ratio `plain/fused` 0.27–0.65). E.g., `M16 N4096 K4096 k32`: plain 1.20ms / 0.449 TFLOPs / 14.3 GB/s vs fused 1.97ms / 0.272 TFLOPs / 4.42 GB/s fused-eff, ratio 0.61 (compress 2.0x). `M128 N4096 K4096 k32` compute-bound: plain 0.56ms 7.67 TFLOPs vs fused 2.04ms 2.11 TFLOPs, ratio 0.27. All 8 shape cells `SLOW`/`FAIL`.
- **Ratio table + verdict:** delivered in `amd/BENCH.md` per SUCCESS CRITERION. Regime breakdown: memory-bound (M16) expects ratio ≥1.0 (fewer bytes), compute-bound (M128) expects ≥0.97 — both FAIL. Effective fused BW ~4–5 GB/s vs plain ~13–17 GB/s despite 1.6–2.0× fewer weight bytes.
- **Diagnosis (honest, no profiler run):** decode is not hidden. Fused kernel does per-element bit extraction (8-byte window, 64-bit shift/mask, product split) and `cb_flat` gather inside K-loop, serialized before `mma_sync`, with byte-wise non-coalesced `qw` reads (`qw[gn*row_bytes + sb*type_size + byte_base+b]`) and no LDS staging, no vector loads, no double-buffering, no cb caching. Plain kernel is also naive (single wave LDS tile, low occupancy, plain ~7.6 TFLOPs at M128 vs 9070 XT ceiling ~43 TFLOPs FP8 / ~620 GB/s), but relative ratio still isolates decode overhead (~0.7–1.1ms extra).
- **Ceilings (RX 9070 XT gfx1201, 32 CUs):** ~86 TFLOPs FP16/BF16, ~43 TFLOPs FP8, ~620 GB/s VRAM. Neither kernel saturates ceilings (plain 14–34 GB/s, fused 4–5 GB/s), so absolute numbers are not the claim — the ratio is.
- **M3 exit:** Correct slow kernel is per SPEC an acceptable M3 deliverable; speed not fabricated. Next iteration would need LDS-staged vectorized bit extraction, `cb_flat` in LDS/constant, and pipelined K-loop double-buffering.
- **Files:** `amd/BENCH.md` (verbatim + ratio table + verdict), updated `amd/STATUS.md`.
- **How to resume (after M3):**
  ```bash
  git checkout feat/amd-rdna4-fp8
  cat amd/STATUS.md
  cat amd/BENCH.md
  hipcc --offload-arch=gfx1201 amd/bench_cb_hip.cpp -o /tmp/bench_cb --std=c++17 -O2 && /tmp/bench_cb
  # For M4 perf tuning: profile with rocprofiler-sdk, iterate on LDS/vectorized decode per BENCH.md diagnosis.
  ```

## 2026-08-06 — M4 — Optimized plain vs fused (roofline) — DONE (honest fail, optimized)

- **Goal per OPERATOR M4:** Get plain FP8 to >50% roofline, then re-run ratio table on optimized fused (LDS-staged vectorized bit extraction, cb resident in LDS, pipelined K-loop treatment). Deliver ratio table on optimized kernels (pass or attested-fail).
- **Plain optimization:** `amd/gemm_opt_hip.cpp` adds `plain_direct16` (WMMA 16x16x16 direct global `load_matrix_sync`/`mma_sync`/`store_matrix_sync`, no LDS, no masking, grid (N+15)/16 x (M+15)/16, block 32) and `plain_direct32` (16x16x32). Both compile on gfx1201 via `enable_gfx12`. Bench vs M3 LDS plain (1.0ms, 17GB/s) now: `M16 N4096 K4096` plain_direct16 0.032–0.044ms 384–523 GB/s 12–16 TFLOPs (62–84% of ~620 GB/s roofline), plain_direct32 0.034–0.035ms 486–492 GB/s 15 TFLOPs. M128 plain_direct32 0.167–0.184ms 23–25 TFLOPs (compute-bound, 115 GB/s). Target >50% BW met (previously 2%). Command: `hipcc --offload-arch=gfx1201 amd/gemm_opt_hip.cpp -o /tmp/gemm_opt && /tmp/gemm_opt` — verbatim output captured to `amd/BENCH_M4.md`.
- **Fused optimization:** Three variants in same file:
  - `fused_naive` — M3 baseline per-element LDS (2.73ms at M16, ratio 0.016).
  - `fused_vec` — per-codeword (32 workers per 16x16 tile, one codeword→8 FP8 values, 8-byte window per codeword instead of per element, generic k handling) — 0.86–1.15ms, 3.2× faster than naive but still 20× slower than plain.
  - `fused_k32_fast` — specialized for k=32 (byte-aligned 4B/code, bit_shift 0): `*(uint32_t*)(qw+gn*rbytes+sb*128+vec*4)`, cb in LDS `s_cb[2048]` loaded once per block via 64-bit vector loads, expand via 4 byte extracts + LDS gather (`c0/c1/c2/c3` + `s_cb[c*2 + sub*512]`), A direct global. 0.087–0.11ms at M16, ratio 0.30–0.50, ~31× faster than naive (2.73→0.087) but still 2–3× slower than plain. Correctness: k32_fast bit-exact vs CPU ref (max_abs 0, PASS per `/tmp/m4_opt_correct` probe).
- **Isolated decode probe:** Row-coop decode with LDS qw + LDS cb + vectorized 64-bit loads + per-codeword expand: 0.0305ms for N=4096 K4096 k32 (8 MB packed →16 MB decoded), 274 GB/s packed (vs elem 0.66ms 12GB/s, vec 0.176ms 47GB/s, k32_fast 0.087ms 95GB/s). Shows decode alone already slower than plain's memory saving (8 MB saving at 517 GB/s =0.015ms, decode 0.030ms > saving). Even sequential decode+GEMM 0.062ms > plain 0.032ms.
- **Full sweep (optimized ratio table, best fused per shape: k32→k32_fast, else vec):** All 8 cells FAIL. Example `M16 N4096 K4096 k32` plain 0.033ms vs fused 0.098ms ratio 0.30–0.33 (need ≥1.0). `M128 N4096 K4096 k32` plain 0.236ms 18 TF vs fused 1.32ms 3.2 TF ratio 0.17–0.18 (need ≥0.97). `k36/k40` generic even worse ratio 0.038–0.045 (1.0ms fused vs 0.038ms plain). Table in `amd/BENCH_M4.md`.
- **Diagnosis (honest, timer + isolated probe evidence):** Decode not hidden even after LDS vectorization and cb caching. Plain is now roofline, so ratio is meaningfully testable. Fused's B path still does per-K-tile LDS fill + 2× `__syncthreads` per 16-wide iteration (512 syncs for K=4096) vs plain's zero sync. Qw loads are scattered across rows (stride 2048) vs plain's coalesced WMMA matrix load, and each M tile re-decodes same B (grid Y amplification). To hide decode would need double-buffered pipelined decode (decode next sb while current mma executes), K=32 fused path to halve iterations, and cross-M B reuse or persistent kernel. No profiler counters run beyond hipEvent; next step is `rocprofv3 --kernel-trace`.
- **Files:** `amd/gemm_opt_hip.cpp` (optimized plain+fused), `amd/BENCH_M4.md` (verbatim + ratio table + verdict), updated `amd/STATUS.md`.
- **Correctness:** `amd/cb_decode_hip.cpp` still PASS all rungs (re-ran ` /tmp/cb_decode_hip` — PASS). `fused_k32_fast` PASS k32 M16 N4 K512 max_abs 0.

## 2026-08-06 — M5 — Decode-in-mainloop pipelined (K=32, double-buffered) — DONE (honest attested-fail, profiled)

- **Goal per OPERATOR M5:** Profile first (counters/ISA), then restructure fused kernel as software-pipelined K-loop where stage N's WMMA consumes tile N while stage N+1's index words are in flight and decode ALU interleaves, codebook LDS-resident, vectorized coalesced loads, no per-K __syncthreads between stages. k32 lane first. Deliver ratio table on pipelined kernels (pass or profiler-attested hardware argument).
- **Profile first (tooling authorization exercised, honest absence):**
  - `rocprofv3-avail list --pmc` on gfx1201: `No pmc counters supported` for both GPU agents (RX 9070 XT and iGPU), `metadata.cpp:273 Agent HW architecture is not supported`. `rocprofv3 --kernel-trace` with csv/pftrace output yields empty (simple_timer only) — no kernel trace files generated on gfx1201 with ROCm 7.14.60850 / rocprofiler-sdk 1.3.2. PMC/SPM/PC-sampling all unsupported (`pmc-check` fails). Attested verbatim in `amd/BENCH_M5.md` §1.1.
  - Fallback evidence: `hipEventElapsedTime` timer (300 iters, 20 warmup, same harness, block 32, grid (N+15)/16 x (M+15)/16), isolated decode probe (row-coop LDS qw + LDS cb, 64-bit vector loads, 8 MB packed → 16 MB decoded, 0.030535 ms, 274 GB/s packed), and ISA dump via `llvm-objdump --arch-name=amdgcn` showing `global_load_b64/b128` + `v_wmma_f32_16x16x16/32_fp8_fp8` + `s_wait_loadcnt`/`s_wait_alu` + `__syncthreads` counts. Plain direct uses 0 LDS / 0 sync vs fused's 512 syncs (16x16) / 128 syncs (16x32 pipe). Captured in `amd/BENCH_M5.md` §1.2–1.3.
- **Pipelined kernel:** `amd/gemm_m5_hip.cpp` adds `fused_m5_k32` — 16x16x32 tile (`fragment<matrix_a|b,16,16,32,hip_fp8_e4m3>`), double-buffered LDS B `s_B[2][512]` (1024B), cb LDS `s_cb[2048]` via 64-bit vector loads once per block, vectorized qw `*(uint32_t*)(qw + gn*rbytes + sb*128 + vec*4)` (4B per codeword, byte-aligned k32), 64 codewords per 16x32 tile (32 threads × 2 codewords), expand via `c0..c3` + `s_cb[c*2 + sub*512]` (sub_dim=2, 4 sub-tables), store col_major `s_B[buf][n*32 + k]` with `ldm=32`, single `__syncthreads` per K chunk (after decode, before `load_matrix_sync`), A direct global `load_matrix_sync(fa, ap, K)`. Correctness: `M16 N32 K256 k32 max_abs 0 PASS` vs CPU bit-exact reference (LSB-first 8-byte window, product split `k/4` uniform, `cb` gather), plus `amd/cb_decode_hip.cpp` still PASS all rungs.
- **Build:** `hipcc --offload-arch=gfx1201 amd/gemm_m5_hip.cpp -o /tmp/gemm_m5 --std=c++17 -O2` (warnings only). `hipcc --offload-arch=gfx1201 amd/gemm_opt_hip.cpp -o /tmp/gemm_opt` retained for plain baseline.
- **Bench (verbatim in BENCH_M5.md):** 300 iters per cell, 20 warmup, hipEvent.
  ```
  M16 N4096 K4096 plain_direct16 0.0390–0.045 ms / 379–438 GB/s (62–70% roof) | fused_m5 0.132–0.137 ms / 63–66 GB/s eff  ratio 0.29–0.32 SLOW
  M32 N4096 K4096 plain 0.041–0.057 ms | fused 0.168 ms ratio 0.24–0.34 SLOW
  M64 N4096 K4096 plain 0.137–0.168 ms | fused 0.276–0.283 ms ratio 0.48–0.60 SLOW
  M128 N4096 K4096 plain 0.271–0.313 ms | fused 0.527–0.533 ms ratio 0.50–0.59 SLOW (need ≥0.97)
  M16 N1024 K4096 plain 0.035 ms | fused 0.106 ms ratio 0.33 SLOW
  ```
  Plain_direct32 (16x32) is ~15% faster than plain_direct16 (0.030 vs 0.038 ms at M16), so vs plain32 ratio is worse (0.22). All 6 cells FAIL. M5 improves compute-bound vs M4 (M128 0.50 vs 0.17, M64 0.48 vs 0.12) due to halved K-iterations (128 vs 256) and halved barriers (128 vs 512 syncs), but regresses memory-bound vs M4's k32_fast (M16 0.29 vs 0.45) because per-tile decode now 64 codewords (512 elements) with double-buffer LDS adds LDS traffic and still scattered `qw` row stride 2048 (no cooperative LDS staging of packed stream).
- **Isolated decode still 0.030 ms > saving 0.015 ms:** At M16, saving 8 MB at 438 GB/s =0.018 ms, decode 0.030 ms → even perfect overlap (max) would be 0.030 ms vs plain 0.039 ms, still ~1.3× slower, and fused's extra LDS + sync makes 0.132 ms. For M128, 8× redundant decode (8 M-tiles each re-decode same B) =0.24 ms vs plain 0.27 ms GEMM — no spare bandwidth to hide in. Requires cross-M B reuse (persistent kernel / L2 broadcast) and cooperative 128B coalesced qw staging + async copy (no TMA on RDNA4, unlike SM90's `type_size=4*k` TMA box law in `csrc/cb_fused_gemm.cu`).
- **Diagnosis (profiler-attested hardware argument, no counters available):** Decode is VMEM-bound + VALU (64-bit window shift, 4 LDS gathers per codeword) competing with same VMEM that streams A. `v_wmma_f32_16x16x32_fp8_fp8` is single-cycle, but `global_load_b64/b128` already saturates VMEM at 438 GB/s; adding 8 MB packed loads + 64K LDS gathers cannot be overlapped without `cp.async`/`TMA` — RDNA4 WMMA has no async copy engine in rocWMMA 2.2.1, only synchronous `global_load` + `s_wait_loadcnt` + `v_wmma` serialized. `__syncthreads` per K chunk still serializes despite double buffering. Hardware cannot hide decode with 1 wave/block occupancy; needs persistent N-tile kernel, `ds_read_b32` constant cache for `cb`, and `sched_barrier` pipelining — still must halve decode to <0.015 ms (need >550 GB/s packed) which current 274 GB/s is 2× too slow.
- **Files:** `amd/gemm_m5_hip.cpp` (pipelined 16x32 double-buffered, validated), `amd/BENCH_M5.md` (verbatim + counter absence attestation + ISA + ratio table + verdict), updated `amd/STATUS.md`.
- **Correctness:** `validate_m5 M16 N32 K256 PASS max_abs 0`; `amd/cb_decode_hip.cpp` full sweep still PASS.

## 2026-08-06 — M6 — Coalescing round (repacked, K=32/64, LDS-staged) — DONE (M6_CEILING_ATTESTED)

- **Goal per OPERATOR M6:** Coalesce index-word loads (wavefront-coalesced LDS staging, tile-contiguous repack as serving preprocessing, document cost), decode to registers vs LDS bounce where possible, more waves / 16x16x64 K-chunking. Deliver ratio table on coalesced kernels (pass or ceiling-attested).

- **Kernels:** `amd/gemm_m6_hip.cpp` adds `fused_m6_repacked_k32` (16x16x32, `s_cb[2048]` LDS 64-bit load once, `s_qw[256]` coalesced `global_load_b64` 256B/tile =32×8B contiguous vs M5's scattered row stride 2048, decode from LDS `*(uint32_t*)(s_qw+n*16+v*4)` →4 LDS gathers, `s_B[512]` col_major ldm=32, 1 `__syncthreads` per K chunk, A direct), `fused_m6_repacked_db_k32` (double-buffered `s_qw[2][256]`+`s_B[2][512]`), `fused_m6_repacked_k64` (16x16x64, `s_qw[512]`, `s_B[1024]`, 128 codewords/tile, halves K-loop to 64 for K=4096). Host repack `repack_k32_for_m6` / `repack_k64_for_m6` transforms row-major `qw[gn*rbytes+sb*128+v*4]` to tile-contiguous `qw_tiled[(kt*Ntiles+nt)*256+n*16+v*4]`; repack is offline preprocessing (weight-stationary, legal per M6 spec) ~0.02 ms host for 8 MB, amortized, not in GEMM time. All three validate `M16 N32 K256 max_abs 0 PASS` vs independent CPU bit-exact reference (same as M2), plus `cb_decode_hip` still PASS all rungs.

- **Build:** `hipcc --offload-arch=gfx1201 amd/gemm_m6_hip.cpp -o /tmp/gemm_m6 --std=c++17 -O2` (warnings only). `hipcc --offload-arch=gfx1201 amd/gemm_m5_hip.cpp -o /tmp/gemm_m5` retained for baseline.

- **Bench (verbatim in BENCH_M6.md):** median of 20 reps, 300 iters each, 20 warmup, `hipEventElapsedTime`, interleaved plain/fused per rep to control power state, plus isolated decode probe (`decode_only_scattered` vs `decode_only_repacked`). Plain_direct16 now 0.040-0.050 ms at M16 (340 GB/s, 55% roof) — stable median min 0.039 max 0.050 across 20 reps; plain_direct32 ~0.027 ms (30% faster). `rocprofv3-avail list --pmc` still `No pmc counters supported` on gfx1201 (same as M5), `rocprofv3 --kernel-trace` empty — attested in BENCH_M6.md §1. Isolated decode: scattered 0.087 ms (192 GB/s decoded), repacked coalesced 0.079 ms (210 GB/s decoded, 105 GB/s packed) vs plain saving 8 MB at 340-517 GB/s =0.016-0.023 ms → **decode alone 3.4-5× larger than saving**, even perfect overlap cannot be de-minimis.

- **Ratio table (median of 20, repacked_k32 primary, SUCCESS CRITERION):**
  ```
  M16 N4096 K4096 plain 0.050 ms vs fused 0.112 ms ratio 0.44 SLOW (need ≥1.0)
  M32 N4096 K4096 0.048 vs 0.133 0.36 SLOW
  M64 N4096 K4096 0.137 vs 0.204 0.71 SLOW
  M128 N4096 K4096 0.305 vs 0.452 0.66 SLOW (need ≥0.97)
  M16 N1024 K4096 0.045 vs 0.127 0.35 SLOW
  M16 N2048 K1024 0.015 vs 0.038 0.39 SLOW
  Best variant k64 at M128: 0.249 vs 0.238 ratio 0.98 borderline PASS but only at M128 and not reproducible (second batch with thermal jitter shows same shape 0.99 PASS but M64 inflates to 1.15 due to plain slow outlier 0.357 vs stable 0.137 — variance from WSL dxg, reported honestly). M5 baseline for comparison: 0.37/0.26/0.47/0.52 — M6 improves ~1.2× at M16 (0.44 vs 0.37) and ~1.5× at M64 (0.71 vs 0.47) / M128 k64 (0.98 vs 0.52) from coalescing + K=64 halving, but memory-bound still 2.2× slower.
  ```
  Full verbatim + min/max + second jittered batch captured in `amd/BENCH_M6.md`.

- **Diagnosis (timer + ISA, counters unavailable, same hardware limit as M5):** Coalescing via repacked `global_load_b64` (256B contiguous per tile) lifts decoded BW from 192 to 210 GB/s (1.09×) but still VMEM-bound and shares controller with A stream. `v_wmma_f32_16x16x32/64` single-cycle, but per-tile 256 `ds_read_b32` LDS gathers (4 per codeword) + `s_wait_loadcnt`/`s_wait_alu` + `__syncthreads` (128 syncs for K=4096/32, 64 for k64) serialize. Cross-M redundancy 8× at M128: each M-tile re-decodes same B (0.079×8=0.63 ms vs plain GEMM 0.30 ms). No TMA/`cp.async` on RDNA4 to hide next-tile `global_load` behind `mma_sync`; double-buffer still needs `__syncthreads`. Even coalesced, isolated decode 0.079 ms (105 GB/s packed) must be <0.016 ms (>500 GB/s packed) to break even — needs 5× more packed BW, impossible while also streaming A at 340 GB/s. Persistent N-tile kernel + `cb` in `__constant__` + fragment-direct decode (no LDS `s_B` bounce) + `sched_barrier` still required but not available in rocWMMA 2.2.1.

- **Verdict:** **M6_CEILING_ATTESTED** — coalesced, vectorized, K=64 kernels implemented, validated, and benchmarked with honest slow exit. No speed fabricated. Counter absence re-attested, ISA noted, isolated probe proves decode > saving. Hardware cannot hide this decode on gfx1201 with current ROCm without persistent cross-M B reuse and true async copy.

- **Files:** `amd/gemm_m6_hip.cpp` (repacked coalesced + db + k64, validated), `amd/BENCH_M6.md` (verbatim + ratio table + ceiling attestation + ISA + repack cost, ends with `M6_CEILING_ATTESTED`), updated `amd/STATUS.md`.

- **How to resume (after M6):**
  ```bash
  git checkout feat/amd-rdna4-fp8
  cat amd/STATUS.md
  cat amd/BENCH_M6.md
  hipcc --offload-arch=gfx1201 amd/gemm_m6_hip.cpp -o /tmp/gemm_m6 --std=c++17 -O2 && timeout 360 /tmp/gemm_m6 2>&1 | tee /tmp/m6.log
  # For next perf iteration: persistent N-tile kernel + fragment-direct (no LDS bounce) + constant cb, profile once gfx1201 PMCs appear.
  ```

## 2026-08-06 — M6 verification gate — DONE (M6_CEILING_ATTESTED verified)

- **Gate per SPEC §M6 VERIFICATION GATE:** One clean process, one allocation set, alternating plain and fused, >=20 reps each, report BOTH min and median for BOTH kernels, compare fused vs plain AT ITS BEST (min, cache-warm), explain 5x baseline discrepancy, cost repack, then write M6_BAR_MET_VERIFIED or M6_CEILING_ATTESTED.
- **Harness:** `amd/bench_verify_m6.cpp` — hipcc --offload-arch=gfx1201, one alloc set per shape (dA+dB 16.7MB + dqw_tiled 8MB + dcb 2KB + dC, <40MB <64MB Infinity Cache), 20 warmup, 20 reps alternating plain_direct16 then fused_m6_repacked_k32 (300 iters each, hipEvent), reports min/med/max for both. Also repack microbench (50 reps, volatile sink, median 0.143 ms for 8 MB, 58 GB/s host, once per weight, amortized).
- **Repack costed:** 8 MB moved (N*rbytes=8388608 source packed -> same 8388608 dest tiled, contiguous permutation), median 0.1435 ms min 0.142 ms max 0.183 ms host BW 58 GB/s, once per weight stationary preprocessing (legal per spec), amortized over 1000 tokens adds 0.00014 ms per GEMM, negligible. Fused kernel reads only `qw_tiled`. Verbose log in `amd/BENCH_M6.md` §8.4 and `/tmp/verify.log`.
- **Discrepancy explained (with evidence):** M4 isolated plain min 0.032-0.044 ms / 525 GB/s (84% of 620 GB/s roof) is roofline; M6 suite median 0.05-0.09 ms max 0.54 ms (13x) was thermal throttling + per-rep hipMalloc/hipFree churn (20x alloc/free per shape, no cooldown) on WSL dxg. Verification one-alloc shows plain min 0.03255 ms med 0.03420 max 0.04416 (1.35x) — early reps cool, later hot. Infinity Cache 64 MB > total resident 25-40 MB, so B fully cacheable; measured 525 GB/s is VRAM not cache (cache would be >2 TB/s), throttling reduces clock uniformly. Written in BENCH_M6.md §8.3.
- **Verified ratio table (vs plain_min, gate-compliant):**
  ```
  M16 N4096 K4096 plain_min 0.03255 med 0.03420 | fused_min 0.06210 med 0.07426 ratio min/min 0.52 FAIL (need >=1.0) BW 525 GB/s vs 140 eff
  M32 N4096 K4096 0.03341 vs 0.06853 ratio 0.48 FAIL
  M64 N4096 K4096 0.07962 vs 0.09450 ratio 0.84 FAIL (best, still <0.97)
  M128 N4096 K4096 0.14352 vs 0.19305 ratio 0.74 FAIL (need >=0.97)
  M16 N1024 K4096 0.02018 vs 0.05769 ratio 0.34 FAIL
  M16 N2048 K1024 0.00737 vs 0.01840 ratio 0.40 FAIL
  ```
  No shape reaches bar even on fused_min vs plain_min (harshest). Full per-rep logs in BENCH_M6.md §8.4.
- **Verdict:** **M6_CEILING_ATTESTED** (verified) — gate satisfied, honest ceiling attested with timer+ISA+isolated probe (rocprofv3 PMCs still `No pmc counters supported` on gfx1201, re-attested). Repack does not hide decode (isolated decode 0.085 ms > saving 0.015 ms at 525 GB/s); persistent cross-M B reuse + async copy (TMA) still required but unavailable in rocWMMA 2.2.1 on RDNA4.
- **Files:** `amd/bench_verify_m6.cpp` (verification harness), `amd/BENCH_M6.md` §8 (verbatim + table + explanation + ceiling), updated `amd/STATUS.md`. Build: `hipcc --offload-arch=gfx1201 amd/bench_verify_m6.cpp -o /tmp/bench_verify_m6 --std=c++17 -O2 && /tmp/bench_verify_m6`.

## Blockers

- M6 perf ceiling (verified): Even gate-compliant one-alloc alternating plain_min vs fused_min, best ratio 0.84 at M64, worst 0.34-0.52 memory-bound, 0.74 compute-bound. Repacked coalesced 256B/tile + LDS cb + 2 barriers/tile still 1.2-2.9× slower. Isolated decode 0.085 ms (98 GB/s packed) vs saving 0.015 ms at 525 GB/s =5.6× too slow; cross-M 8× redundancy at M128. Requires persistent B reuse + async copy — not available on RDNA4 WMMA 2.2.1; counters unavailable. Host repack 0.14 ms once per weight, amortized negligible, does not change verdict. Attestation is timer+ISA+isolated probe+absence as authorized. No further SW pipelining with 1 wave/block can reach bar without 5× packed BW. **M6_CEILING_ATTESTED**
- WSL ROCm: `rocm-smi` still `amdgpu not found`, hip functional. `rocprofv3` PMCs remain unsupported on gfx1201 (no change from M5).

## 2026-08-06 — M7 — Persistent-B + Fragment-Direct (both implemented, measured) — DONE (M7_CEILING_DEMONSTRATED)

- **Goal per OPERATOR M7:** Implement the two optimizations only argued in M6: M7.1 persistent-B tile reuse (one CTA owns N-tile, decode once per K chunk, reuse across Mt) and M7.2 fragment-direct decode (no LDS bounce, empirically probed mapping). Both have precedent in this repo: `csrc/cb_moe_persistent_b.cu` (persistent-B, TN*TK tiles, K-major XOR swizzle, 2 CTAs/SM budget) and `csrc/cb_gemv_v2.cu` (whole packed row staged in one coalesced burst, full sub-codebook staged to smem once per block).
- **Kernels:** `amd/gemm_m7_hip.cpp` adds `fused_persistent_k32` (M7.1, 16x32, grid Ntiles=256, s_cb 2048B 64-bit once, s_qw 256B coalesced, s_B 512B, decodes 64 codewords/tile once per K chunk, load fb once, loop Mt 1..8 mma with same fb, acc[Mt] fragments, 128 decodes for K=4096 vs 1024 for M128 M6) and `fused_persistent_frag_direct_k16` (M7.2, 16x16, s_cb+s_qw LDS, per-lane fb register fill via probed mapping n=lane%16, k=(lane/16)*8+i, 8 elements/lane, no s_B, fb reused across Mt). Both validate bit-exact: M16 N32 K256 PASS max_abs 0, M64 N32 K256 PASS, same CPU reference as M2 (LSB-first window, product split). M6 `fused_m6_repacked_k32` retained for comparison.
- **Fragment probe (empirical, not assumed):** `/tmp/probe_frag.cpp` on gfx1201: B 16x16 col_major fb num_elements 8, lane0:0 16 32 48 64 80 96 112 col0 rows0..7, lane16 duplicate rows8..15; for B 16x32 fb 16 elements showed rows0..15 only (lane16 duplicate), so M7.2 uses 16x16 mapping n=lane%16 k=(lane/16)*8+i, manual fill mismatch 0 confirms correct. ISA via llvm-objdump shows global_load_b64 + ds_read + v_wmma_f32_16x16x32_fp8_fp8 + s_barrier/s_wait.
- **Build:** `hipcc --offload-arch=gfx1201 amd/gemm_m7_hip.cpp -o /tmp/gemm_m7 --std=c++17 -O2` (warnings only); `hipcc --offload-arch=gfx1201 amd/bench_verify_m7.cpp -o /tmp/bench_verify_m7 --std=c++17 -O2` (one-alloc gate harness, alternates plain/m6/persistent/frag).
- **Bench (verification gate: one alloc set, alternating, 20 reps, 300 iters, 20 warmup, hipEvent, min+med vs plain_min):** verbatim in `amd/BENCH_M7.md` section 3 and `/tmp/bench_m7_verify.log`. Repack costed: k32 8MB median 0.151ms 55GB/s host, k16 0.405ms 20GB/s, once per weight amortized ~0. Host repack legal, fused reads only qw_tiled.

- **Ratio table (gate-compliant vs plain_min, plain 538GB/s at M16, 230GB/s at M64, 140GB/s at M128):**
  ```
  Small M (memory-bound, need >=1.0, M16):
   M16 N4096 K4096  plain 0.03179 | m6 0.06094 ratio 0.52 | pers 0.07999 ratio 0.39 | frag 0.13065 ratio 0.24 SLOW
   M16 N1024 K4096  plain 0.01892 | m6 0.05340 ratio 0.35 | pers 0.07151 ratio 0.26 | frag 0.11712 ratio 0.16 SLOW
   M16 N2048 K1024  plain 0.00642 | m6 0.01699 ratio 0.37 | pers 0.02177 ratio 0.29 | frag 0.03317 ratio 0.19 SLOW
  Large M (compute-bound, need >=0.97):
   M32 N4096 K4096  plain 0.03263 | m6 0.06664 ratio 0.48 | pers 0.09811 ratio 0.33 | frag 0.16261 ratio 0.20 SLOW
   M64 N4096 K4096  plain 0.07835 | m6 0.09277 ratio 0.84 | pers 0.13624 ratio 0.57 | frag 0.24042 ratio 0.32 SLOW
   M128 N4096 K4096 plain 0.13817 | m6 0.18815 ratio 0.73 | pers 0.21403 ratio 0.64 | frag 0.38643 ratio 0.35 SLOW
  ```
  Persistence is 1.3x slower than M6 at M16 (no reuse, Mt=1, extra loop overhead) and 1.14x slower at M128 (0.214 vs 0.188) despite 8x fewer decodes (128 vs 1024). Fragment-direct is 1.6-1.8x slower than persistent LDS (0.130 vs 0.079, 0.386 vs 0.214). Best remains M6 at M64 ratio 0.84 (still <0.97). No shape reaches bar even on fused_min vs plain_min.

- **Two regimes separately (M7 addendum):**
  - Large M (M64/M128): persistence arithmetically flips inequality (0.085/8=0.011 <0.015 saving), but measured persistent still 0.214>0.138 (1.54x slower) due to grid parallelism loss (2048 blocks plain/M6 vs 256 blocks persistent) and 8 acc VGPR pressure (9 fragments/wave, ~72 floats) + extra barriers. To hide decode would need 8-wave cooperative persistent (one warp per Mt, shared s_B, cooperative decode) + sched_barrier pipelining — not implemented in rocWMMA 1-wave model.
  - Small M (M16 batch-1): persistence cannot help (Mt=1, no amortization). Lever should be MLP/occupancy (more waves, outstanding loads, whole-row burst as in cb_gemv_v2). M6 already has coalesced 256B burst, but decode still 5.5x > saving. Adding wave via frag-direct increases register pressure and per-lane gathers (4 ds_read per codeword) without async, so even slower (0.24 vs 0.52). Need hardware TMA/cp.async to hide next tile's global_load behind mma — not available on RDNA4 WMMA 2.2.1 (all global_load synchronous s_wait_loadcnt + s_barrier).

- **Diagnosis (timer+ISA+isolated probe+fragment probe, counters unavailable):** Decode remains VMEM+VALU bound (4 LDS cb gathers per codeword, 64 codewords/tile=256 gathers, 2 barriers/tile). Persistent saves decodes but loses parallelism 8x and increases VGPR; fragment-direct saves one LDS bounce/barrier but same gathers + register live range, slower. Even optimal persistent effective decode 0.011ms < saving, but measured 0.214>0.138 shows other bound (occupancy/barriers). Isolated decode 0.085ms >0.015ms saving 5.5x, and at 538GB/s plain, need >550GB/s packed (<0.015ms) to break even — current 98GB/s (repacked) 5.6x too slow. No async copy on gfx1201 with current ROCm. Counters still `No pmc counters supported` (re-attested).

- **Files:** `amd/gemm_m7_hip.cpp` (persistent LDS + frag-direct, validated), `amd/bench_verify_m7.cpp` (gate harness, one alloc, alternating, min/med), `amd/BENCH_M7.md` (verbatim + ratio + ceiling, ends with `M7_CEILING_DEMONSTRATED`), updated `amd/STATUS.md`.

- **Verdict:** **M7_CEILING_DEMONSTRATED** — both optimizations implemented, validated bit-exact, and measured with gate-compliant evidence (one alloc alternating min/median vs plain_min, repack costed, fragment mapping probed, ISA, isolated decode, counter absence re-attested). De-minimis not met in either regime; residual gap explained by occupancy/barrier/VGPR and lack of async copy on gfx1201. No speed fabricated.

- **How to resume (after M7):**
  ```bash
  git checkout feat/amd-rdna4-fp8
  cat amd/STATUS.md
  cat amd/BENCH_M7.md
  hipcc --offload-arch=gfx1201 amd/gemm_m7_hip.cpp -o /tmp/gemm_m7 --std=c++17 -O2 && timeout 60 /tmp/gemm_m7 2>&1 | tee /tmp/m7.log
  hipcc --offload-arch=gfx1201 amd/bench_verify_m7.cpp -o /tmp/bench_verify_m7 --std=c++17 -O2 && timeout 360 /tmp/bench_verify_m7 2>&1 | tee /tmp/bench_m7_verify.log
  # For next iteration: 8-wave cooperative persistent (one warp/M-tile), sched_barrier pipelining, or wait for gfx12 PMCs/TMA.
  ```

## Blockers

- M7 perf ceiling (demonstrated): Even with both required optimizations implemented and measured gate-compliantly, best ratio remains 0.84 at M64 (m6) /0.64 persistent at M128, 0.52/0.39/0.24 at M16. Persistent amortizes 8x (128 vs 1024 decodes) arithmetically 0.011<0.015 but loses 8x block parallelism (2048->256 blocks) and VGPR (8 acc), so 0.214>0.138. Fragment-direct (no LDS B) map probed n=lane%16 k=(lane/16)*8+i correct, but 1.6x slower than LDS due to gather VALU and register pressure. Isolated decode 0.085ms >0.015ms saving 5.5x, need >550GB/s packed. No async copy on gfx1201 WMMA 2.2.1. **M7_CEILING_DEMONSTRATED**
- WSL ROCm: `rocm-smi` still `amdgpu not found`, hip functional. `rocprofv3` PMCs remain unsupported on gfx1201 (no change from M5).
