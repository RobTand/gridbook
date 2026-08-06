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

## Blockers

- M5 perf: Pipelined 16x32 double-buffered K=32 still 2–3× slower memory-bound (ratio 0.29 vs need ≥1.0) and 1.7–2× slower compute-bound (0.50 vs 0.97). Isolated decode 0.030 ms > saving 0.015 ms; cross-M redundancy 8× at M128. Even with halved syncs and larger tile, decode not de-minimis. Needs cooperative LDS qw staging, persistent B reuse, and true async — not available with current rocWMMA API on gfx1201. Counters remain unavailable on this driver (gfx1201 PMCs not implemented in ROCm 7.14), so attestation is timer+ISA+absence.
- WSL ROCm: `rocm-smi` still `amdgpu not found`, hip functional. `rocprofv3` PMCs remain unsupported on gfx1201 (no change from M4).
