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

## Blockers

- M4 perf: Even optimized plain at >50% roofline (517 GB/s), fused decode cost (0.030ms isolated, 0.087ms fused) remains > plain's compression saving (0.015ms). De-minimis not met; needs pipelined LDS decode + K=32 fused + cross-M reuse.
- WSL ROCm: `rocm-smi` still `amdgpu not found`, hip functional. No rocprofv3 counter run — timer only.
