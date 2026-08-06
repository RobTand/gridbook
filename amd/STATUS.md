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

## Blockers

- None for M1. WSL ROCm is functional for hip/rocwmma. PyTorch-ROCm not installed (pip has no torch) — not needed for M1 numpy reference; may install for later validation if useful.
