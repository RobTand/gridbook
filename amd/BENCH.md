# M3 Benchmark — plain FP8 WMMA vs CB decode-fused (gfx1201, RX 9070 XT, ROCm 7.14)

Captured verbatim from `bench_cb_hip` (500 iters per cell, 20 warmup, hipEvent timing) and `cb_decode_hip --bench` (300/500 iters, warmup 20). Both harnesses use identical shapes, same `hip_fp8_e4m3` A/W, WMMA 16x16x16, masked tile loads, `hipEventElapsedTime` averaging.

## Verbatim: bench_cb_hip (dedicated bench)

```
Device AMD Radeon RX 9070 XT gfx1201 32 CUs
Shape M16 N4096 K4096 k32 : plain 1.19521 ms tflops 0.449184 bw 14.3112GB/s | fused 1.97331 ms tflops 0.272067 eff_bw_fused 4.4171 ratio plain/fused 0.605691 (eff weight bytes 1.67772e+07 vs 8.38861e+06 ratio 2)
Shape M16 N4096 K4096 k36 : plain 1.05797 ms tflops 0.507452 bw 16.1676GB/s | fused 2.11557 ms tflops 0.253772 eff_bw_fused 4.61572 ratio plain/fused 0.50009 (eff weight bytes 1.67772e+07 vs 9.43718e+06 ratio 1.77778)
Shape M16 N4096 K4096 k40 : plain 1.03558 ms tflops 0.518423 bw 16.5171GB/s | fused 2.15247 ms tflops 0.249421 eff_bw_fused 5.02374 ratio plain/fused 0.481115 (eff weight bytes 1.67772e+07 vs 1.04858e+07 ratio 1.6)
Shape M32 N4096 K4096 k32 : plain 1.05232 ms tflops 1.02036 bw 16.5658GB/s | fused 1.92504 ms tflops 0.557775 eff_bw_fused 4.69806 ratio plain/fused 0.546648 (eff weight bytes 1.67772e+07 vs 8.38861e+06 ratio 2)
Shape M64 N4096 K4096 k32 : plain 1.04633 ms tflops 2.0524 bw 17.287GB/s | fused 2.0526 ms tflops 1.04623 eff_bw_fused 4.72539 ratio plain/fused 0.509759 (eff weight bytes 1.67772e+07 vs 8.38861e+06 ratio 2)
Shape M128 N4096 K4096 k32 : plain 0.560207 ms tflops 7.66675 bw 34.6276GB/s | fused 2.03954 ms tflops 2.10586 eff_bw_fused 5.39831 ratio plain/fused 0.274674 (eff weight bytes 1.67772e+07 vs 8.38861e+06 ratio 2)
Shape M16 N1024 K4096 k32 : plain 1.03652 ms tflops 0.129489 bw 4.17299GB/s | fused 1.96693 ms tflops 0.0682373 eff_bw_fused 1.13285 ratio plain/fused 0.526973 (eff weight bytes 4.1943e+06 vs 2.09715e+06 ratio 2)
Shape M16 N2048 K1024 k40 : plain 0.25542 ms tflops 0.262739 bw 8.78791GB/s | fused 0.596472 ms tflops 0.11251 eff_bw_fused 2.44467 ratio plain/fused 0.428218 (eff weight bytes 2.09715e+06 vs 1.31072e+06 ratio 1.6)
```

## Verbatim: cb_decode_hip --bench (same harness, 300 iters)

```
Device AMD Radeon RX 9070 XT gfx1201
=== M3 Bench: plain FP8 WMMA vs CB fused (identical shapes) ===
Bench M16 N4096 K4096 k32 : plain 1.27593ms tflops 0.420768 bw 13.4058GB/s | fused 1.96826ms tflops 0.272764 bw_eff 4.42842 ratio plain/fused 0.648254 (weight bytes 16777216 vs 8388608 2x) SLOW
Bench M16 N4096 K4096 k36 : plain 1.04435ms tflops 0.51407 bw 16.3784GB/s | fused 2.11832ms tflops 0.253442 bw_eff 4.60973 ratio plain/fused 0.493011 (weight bytes 16777216 vs 9437184 1.77778x) SLOW
Bench M16 N4096 K4096 k40 : plain 1.04704ms tflops 0.512752 bw 16.3365GB/s | fused 2.16579ms tflops 0.247887 bw_eff 4.99285 ratio plain/fused 0.483445 (weight bytes 16777216 vs 10485760 1.6x) SLOW
Bench M32 N4096 K4096 k32 : plain 1.03403ms tflops 1.03841 bw 16.8589GB/s | fused 1.92622ms tflops 0.557434 bw_eff 4.69519 ratio plain/fused 0.536815 (weight bytes 16777216 vs 8388608 2x) SLOW
Bench M64 N4096 K4096 k32 : plain 1.04149ms tflops 2.06194 bw 17.3674GB/s | fused 1.95563ms tflops 1.0981 bw_eff 4.95968 ratio plain/fused 0.532557 (weight bytes 16777216 vs 8388608 2x) SLOW
Bench M128 N4096 K4096 k32 : plain 0.596245ms tflops 7.20336 bw 32.5347GB/s | fused 2.03236ms tflops 2.11329 bw_eff 5.41736 ratio plain/fused 0.293375 (weight bytes 16777216 vs 8388608 2x) SLOW
Bench M16 N1024 K4096 k32 : plain 1.01229ms tflops 0.132588 bw 4.27285GB/s | fused 1.94724ms tflops 0.0689272 bw_eff 1.1443 ratio plain/fused 0.51986 (weight bytes 4194304 vs 2097152 2x) SLOW
Bench M16 N2048 K1024 k40 : plain 0.256309ms tflops 0.261828 bw 8.75743GB/s | fused 0.596061ms tflops 0.112587 bw_eff 2.44635 ratio plain/fused 0.430005 (weight bytes 2097152 vs 1310720 1.6x) SLOW
```

Build:
```
hipcc --offload-arch=gfx1201 amd/bench_cb_hip.cpp -o /tmp/bench_cb --std=c++17 -O2
hipcc --offload-arch=gfx1201 amd/cb_decode_hip.cpp -o /tmp/cb_decode_hip --std=c++17 -O2
/tmp/bench_cb
/tmp/cb_decode_hip --bench
```
Harness: grid (N+15)/16 x (M+15)/16, block 32 (one wave), WMMA 16x16x16 fp8_e4m3 row_major A, col_major B, float accum, masked LDS tile (shared 256). Fused decodes B tile on-the-fly: 8-byte window LSB-first, `code = (window>>bit_shift)&mask`, `sub_idx=(code>>bit_off)&mask_sub`, `flat_idx=row_offset+table_base+sub_idx*sub_dim+local`, gather `cb_flat`.

## Ratio table (SUCCESS CRITERION deliverable)

Ratio = T_plain / T_fused . >1 means fused faster. Effective weight bytes ratio = (N*K)/(N*row_bytes) where row_bytes = n_sb*4*k, n_sb=K/256. For k=32, 2.0x compression; k=36 1.778x; k=40 1.6x.

| shape (M,N,K) | k | n_sub | regime* | T_plain ms | T_fused ms | ratio plain/fused | weight bytes plain→fused | effective compression | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 16,4096,4096 | 32 | 4 | memory-bound (small M) | 1.20 | 1.97 | **0.61** | 16.8M → 8.4M | 2.0x | **FAIL** (should be ≥1.0) |
| 16,4096,4096 | 36 | 4 | memory-bound | 1.06 | 2.12 | **0.50** | 16.8M → 9.4M | 1.78x | FAIL |
| 16,4096,4096 | 40 | 4 | memory-bound | 1.04 | 2.15 | **0.48** | 16.8M → 10.5M | 1.6x | FAIL |
| 32,4096,4096 | 32 | 4 | memory-bound | 1.05 | 1.93 | **0.55** | 16.8M → 8.4M | 2.0x | FAIL |
| 64,4096,4096 | 32 | 4 | transitional | 1.05 | 2.05 | **0.51** | 16.8M → 8.4M | 2.0x | FAIL |
| 128,4096,4096 | 32 | 4 | compute-bound | 0.56 | 2.04 | **0.27** | 16.8M → 8.4M | 2.0x | FAIL (should be ≥0.97) |
| 16,1024,4096 | 32 | 4 | memory-bound | 1.04 | 1.97 | **0.53** | 4.19M → 2.10M | 2.0x | FAIL |
| 16,2048,1024 | 40 | 4 | memory-bound | 0.26 | 0.60 | **0.43** | 2.10M → 1.31M | 1.6x | FAIL |

*Regime heuristic: M=16 is decode-bound/memory-bound classic LLM decode batch; M=128 approaches compute-bound. 9070 XT ceilings: ~86 TFLOPs FP16/BF16, ~43 TFLOPs FP8 (WMMA), ~620 GB/s VRAM. Measured plain FP8 WMMA at M128 achieves ~7.6 TFLOPs / 34 GB/s in this naive single-wave LDS tile kernel — well below ceilings, but consistent baseline for comparison since both kernels share same GEMM epilogue structure.

## De-minimis verdict

**NOT MET.** The CB decode-fused GEMM is ~1.5–3.7× slower than the plain FP8 WMMA baseline of identical shape (ratio 0.27–0.65). Despite streaming 1.6–2.0× fewer weight bytes (indices vs raw FP8), effective fused bandwidth (`(M*K + N*row_bytes)/(time)`) is ~4–5 GB/s vs plain ~13–17 GB/s; the decode overhead dominates.

Diagnosis:

- The fused kernel decodes inside the K-loop per 16-wide tile: per B element it assembles an 8-byte window (`window |= byte << 8*b`), 64-bit shift, mask, product split (`bit_off` sum), and a global `cb_flat` gather, all in scalar ALU before the WMMA `mma_sync`. This decode is **not hoisted or double-buffered**, so it serializes ahead of `load_matrix_sync`/`mma_sync`. Plain kernel just does masked LDS loads.

- `qw` is read as `qw[gn*row_bytes + sb*type_size + byte_base+b]` per thread per K-tile — non-coalesced, byte-wise, no vectorized 32/64-bit loads, no LDS staging of the packed stream. Each wave re-reads the same `qw` bytes for different M tiles (no reuse across M).

- `cb_flat` is a small table (flat_size = Σ(1<<w)*sub_dim; e.g., k=32 → 4*256*2=2048 entries) but accessed via `cb_flat[flat_idx]` with divergent `sub_idx`, causing scalar memory accesses and no constant cache hint.

- The plain baseline itself is not yet tuned (single wave per 16×16 tile, LDS spilling, low occupancy), so absolute TFLOPs are low; but the *relative* comparison is still valid because both share that baseline structure — decode overhead would be proportionally larger even on a tuned GEMM.

What would be needed to hide decode (not done in this M3):

- Pre-stage packed `qw` tiles to LDS with 4-byte or 8-byte vector loads, then bit-extract from LDS with `__builtin_amdgcn_mov_b32`/`shift` — reduces global memory divergence.
- Hoist decode out of inner K-loop via double-buffering: decode next K-tile while current tile's `mma_sync` executes (requires pipelined loads, not possible with single shared tile synchronously).
- Use SGPR-stored `cb_flat` in LDS or constant memory, or embed as immediate `v_mad` for small k.
- For larger M, use persistent kernel / split-K to amortize decode across M.

Honest attestation: the current fused kernel is **correct** (bit-exact vs independent CPU reference per `cb_decode_hip` decode/GEMM tests) but **slow** — decode cost is not de-minimis. No profiler (rocprof/rocprofiler-sdk) was run in this pass; timing evidence above is the stall signal. A correct slow kernel is the prescribed M3 exit; fabricated speed would be dishonest.

Next iteration would require pipelined LDS decode + vectorized bit extraction + cb caching. No artificial speed numbers are reported.
