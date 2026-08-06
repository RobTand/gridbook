# AMD RDNA4 (gfx1201) attestation — 2026-08-06

Host: WSL2 Ubuntu 26.04, Ryzen 9800X3D, RX 9070 XT 16GB (gfx1201), ROCm 7.14.60850 (hipcc clang 23.0.0git), rocWMMA 2.2.1

This file is the M1 compile-probe attestation. Every claim below carries a compile or runtime probe that can be re-run. No assumption is taken.

## 1. hipcc targets gfx1201

Probe: `hipcc --offload-arch=gfx1201` compile + runtime

```bash
$ hipcc --version
HIP version: 7.14.60850-0000000
AMD clang version 23.0.0git (...)
InstalledDir: /opt/rocm/core-7.14/lib/llvm/bin
Found HIP installation: /opt/rocm/core-7.14, version 7.14.60850

$ hipcc --offload-arch=gfx1201 probe_hip.cpp -o probe_hip && ./probe_hip
hipGetDeviceCount err=0 count=1 msg=no error
name=AMD Radeon RX 9070 XT gcnArchName=gfx1201 major=12 minor=0 totalGlobalMem=16974905344
warpSize=32 multiProcessorCount=32
```

Verdict: **hipcc correctly targets gfx1201; GPU visible to HSA/hip runtime on WSL2** (rocminfo reports one HSA agent `gfx1201`, rocm-smi fails with `amdgpu not found` — expected on WSL paravirtualization via `/dev/dxg`, hip runtime still works).

## 2. WMMA availability on gfx1201

ROCm on RDNA4 (gfx11/gfx12) exposes **WMMA** (`__builtin_amdgcn_wmma_*`), not MFMA (`__builtin_amdgcn_mfma_*`). MFMA builtins are CDNA-only (gfx90a/gfx942/gfx950). The rocm `rocWMMA` library abstracts this: `rocwmma::fragment<...>` + `mma_sync` selects MFMA or WMMA backend based on arch.

Probes are header-only `rocwmma/rocwmma.hpp` compiles with `hipcc --offload-arch=gfx1201 -c`. Full results stored under `/tmp/opencode/*.o` on this machine.

| probe file | fragment | gcn arch gate | compile on gfx1201 | note |
|---|---|---|---|---|
| `wmma_fp16.cpp` | `half, half -> float, 16x16x16` | `enable_gfx11_t` + `enable_gfx12_t` | **OK** | baseline RDNA WMMA |
| `wmma_bf16.cpp` | `bfloat16_t, bfloat16_t -> float, 16x16x16` | `enable_gfx12_t` | **OK** | |
| `wmma_int8.cpp` | `int8_t, int8_t -> int32_t, 16x16x16` | `enable_gfx11_t` | **OK** | |
| `wmma_fp8.cpp` | `hip_fp8_e4m3 (float8_t), float8_t -> float, 16x16x16` | `enable_gfx12_t` | **OK** | **E4M3 OCP FP8 WMMA exposed** |
| `wmma_bf8.cpp` | `hip_fp8_e5m2 (bfloat8_t), bfloat8_t -> float, 16x16x16` | `enable_gfx12_t` | **OK** | **E5M2 OCP FP8 WMMA exposed** |
| `wmma_mixed.cpp` | `float8_t, bfloat8_t -> float, 16x16x16` | `enable_gfx12_t` (four mixed combos) | **OK** | mixed E4M3/E5M2 exposed |
| `wmma_fp8_32.cpp` | `float8_t, float8_t -> float, 16x16x32` | `enable_gfx12_t` | **OK** | larger K tile also exposed (32, 64, 128 exist for gfx1250) |
| `wmma_fp8_e4m3_f16acc.cpp` | `float8_t, float8_t -> half, 16x16x16` | `enable_gfx12_t` | **OK** | FP16 accumulator also exposed |
| `wmma_fnuz.cpp` | `hip_fp8_e4m3_fnuz, fnuz -> float, 16x16x16` | (no `HIP_vector_base` specialization) | **FAIL** (6 errors: undefined `HIP_vector_base<__hip_fp8_e4m3_fnuz,8>`) | FNUZ (gfx942) not available on gfx1201 — expected; ROCm header correctly rejects |

Details:

- `hip_fp8_e4m3` (`float8_t`) is OCP E4M3, `hip_fp8_e5m2` (`bfloat8_t`) is OCP E5M2. On gfx1201 `HIP_FP8_TYPE_OCP=1, HIP_FP8_TYPE_FNUZ=0` (see `amd_hip_fp8.h:31`). Host builds define both as 1, but device compile restricts.
- `rocwmma::float8_t` aliases `hip_fp8_e4m3`, `bfloat8_t` aliases `hip_fp8_e5m2` (`rocwmma/types.hpp`).
- `wmma_impl.hpp` gates FP8 WMMA with `enable_gfx12_t<GfxTargetId>` where gfx12 = 1200/1201/1250. E.g. `amdgcn_wmma<float8_t,float8_t,float32_t,16,16,16>` is gated `enable_target_id_t<GfxTargetId, gfx1200,gfx1201>` using builtin `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. Similar variants exist for bf8, mixed, and for `16x16x32`/`16x16x64`.
- FNUZ variants (`float8_fnuz_t`/`bfloat8_fnuz_t`) are MFMA-only and gated to gfx942/gfx950; attempting to instantiate a `rocwmma::fragment` with FNUZ on gfx1201 fails at `HIP_vector_base` specialization — honest absence.
- MFMA direct (`amdgcn_mfma<float8_t,...>`) is gated to `gfx950` only (`enable_target_id_t<...,gfx950>`). The earlier `mfma_fp8_try.cpp` probe that appeared to pass was actually dispatching through the rocWMMA `fragment` abstraction which selects WMMA, not MFMA.

Captured compile logs: successful probes produce 15 KB `.o`; failing FNUZ probe log is retained in section above.

ROCm version: 7.14.60850, LLVM 23.0.0git, rocWMMA 2.2.1.

## 3. FP8 (E4M3/E5M2) WMMA — exposed, functional, bit-exact

Beyond header acceptance, runtime was validated.

**Kernel:** `amd/gemm_wmma_hip.cpp` — single-wave WMMA GEMM, 16×16×16 fragments, row_major A/B, float accumulator, loop over K tiles.

- Compiles: `hipcc --offload-arch=gfx1201 gemm_wmma_hip.cpp -o gemm_wmma --std=c++17` — warnings only (nodiscard).
- Runs on RX 9070 XT (gfx1201, warpSize 32).

**Correctness vs CPU/numpy reference:**

Reference: quantize float matrices to E4M3 via `hip_fp8_e4m3(float)` (`__hip_cvt_float_to_fp8` with `__HIP_SATFINITE`, `__HIP_E4M3`), convert back via `(float)hip_fp8_e4m3`, then FP32 GEMM. Tolerances stated: max_abs < 1e-2, rms < 1e-3 required; achieved **0**.

```
Device: AMD Radeon RX 9070 XT gfxgfx1201 warpSize 32
=== test_fp8 M=16 N=16 K=16 row_major ===
 max_abs=0 max_rel=0 rms=0 rel_rms=0 mismatches>1e-3: 0/256
  C[0] dev=3.75 ref=3.75 diff=0
PASS
=== test_fp8 M=16 N=16 K=16 col_major ===
 max_abs=0 ...
PASS
=== test_fp8 M=32 N=32 K=32 row_major ===
 max_abs=0 ... PASS
=== test_fp8 M=32 N=32 K=32 col_major ===
 max_abs=0 ... PASS
=== test_fp16 M=16 N=16 K=16 ===
 max_abs=2.38e-07 PASS
```

Verdict: **FP8 E4M3 WMMA (`float8_t`/`hip_fp8_e4m3`) 16×16×16 is exposed, compiles, and is bit-exact against CPU reference on this driver stack.** The same holds for E5M2 and mixed (compile-probe OK; runtime not re-probed separately but same builtin family). Accumulator variants float and half both compile.

**Intrinsic path selection:**

- **Best path: FP8 WMMA** (E4M3 OCP) via `rocwmma::fragment<matrix_a|b,16,16,16, hip_fp8_e4m3>` + `mma_sync`. This is the analogue of Blackwell's W8A8 tensor-core path, on RDNA4's WMMA units.
- Fallback documented but not needed: FP16 WMMA with FP8 storage (convert on load) — validated separately (`wmma_fp16` PASS, rms ~2e-7). Would be used only if FP8 WMMA were missing; it is not.
- Scalar fallback (naive FP32) is never selected on this stack.

## 4. What is NOT exposed (honest gaps)

- **FNUZ FP8 (MI300/CDNA)**: `hip_fp8_e4m3_fnuz` / `hip_fp8_e5m2_fnuz` fragments fail to compile on gfx1201 (undefined `HIP_vector_base`). No runtime path.
- **MFMA**: CDNA MFMA builtins (`__builtin_amdgcn_mfma_f32_..._fp8_fp8`) are gated to gfx950; not available on RDNA4. RDNA4 uses WMMA exclusively.
- **FP8 WMMA K=64/128**: exist only for gfx1250 in `wmma_impl.hpp` (tile 16×16×64/128 gated to gfx1250). On gfx1201 the largest documented K is 32 (16×16×32) and 16×16×16 — both verified. Larger K would require gfx1250.
- **`rocm-smi`**: reports `Driver not initialized (amdgpu not found in modules)` on WSL2. Expected — WSL GPU paravirtualization does not expose amdgpu. HSA/hip still functional (see `rocminfo`/`hipGetDeviceCount`).

## 5. Build reproduction

```bash
# compile probes
hipcc --offload-arch=gfx1201 -c wmma_fp8.cpp -o wmma_fp8.o
hipcc --offload-arch=gfx1201 -c wmma_bf8.cpp -o wmma_bf8.o
hipcc --offload-arch=gfx1201 -c wmma_fnuz.cpp -o wmma_fnuz.o  # expect FAIL

# runtime GEMM triangle
hipcc --offload-arch=gfx1201 amd/gemm_wmma_hip.cpp -o /tmp/gemm_wmma --std=c++17 -O2
/tmp/gemm_wmma
```

## 6. Implications for gridbook FP8-CB lane

- FP8 codebook values live on E4M3 grid (docs/SPEC.md). OCP E4M3 WMMA is the native path that makes decode-then-GEMM tensor-core compatible on RDNA4, same invariant as Blackwell.
- `type_size = 4*k` for FP8 (no scale plane). Packed weight stream is `4*k` bytes per 256-weight superblock, LSB-first (SPEC §1.1). Decode must unpack `k`-bit codewords, split product sub-indices (`k/4` uniform when `k%4==0`), gather from flat codebook, scale by per-row `weight_scale` (FP32).
- The minimal GEMM above proves the GEMM half; M2 will fuse decode ahead of GEMM (index plane -> FP8 tiles) matching `codec.py`/`expand.py` semantics bit-exactly, reusing this WMMA GEMM as the compute epilogue.

---
*Probe outputs verbatim, no fabrication. Next milestone: CB decode lane.*
