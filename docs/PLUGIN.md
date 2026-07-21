# vllm-prismaquant

Out-of-tree vLLM quantization plugin for PrismaQuant's **NVFP4-CB / FP8-CB**
codebook formats (docs/nvfp4-cb-plan/). The **Triton** path (`kernels.py`) was
**prototype (i)** — a correctness-first reference used to obtain the first served
KL-vs-BF16 and speed reading. The **current served path is the CUDA kernel set**
(`csrc/cb_gemv.cu`, JIT-built via `cuda_ext.get_ext()`), which honours INV-1
(decode in registers/smem, never a materialized `[N,K]` in HBM) at
bandwidth-bound decode speeds; the Triton path remains the bit-exact fallback
when nvcc is unavailable. See the CUDA kernel-set section below.

## Invariants

- **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) scales. The dense `[N,K]` weight is
  never materialized in HBM — each superblock's weight tile is expanded inside
  the kernel, in registers, then consumed by the matmul.
- **INV-2 (WAIVED):** we decode FP4/FP8 codes to bf16 and run `tl.dot` (bf16
  MMA). Triton cannot emit the Blackwell sm_121 block-scaled FP4 MMA, so this
  path will fail the prefill perf gate by construction. The production prefill is
  prototype (iii) (CUTLASS/CuTe fused-expand).

## Scope

- Even-split **product** mode only (NVFP4_CB_K16 → (8,8); FP8_CB_K44 →
  (11,11,11,11)). Uneven splits and signed/full modes: out of scope here.
- Uniform CB artifacts (all target Linears one format). Mixed containers would
  delegate plain NVFP4/FP8 to stock compressed-tensors — stubbed with a clear
  `NotImplementedError`.
- Single-GPU (tp=1); `--enforce-eager`.

## Layout / registration

- `register_quantization_config("prismaquant")` — vLLM auto-detects from
  `config.json["quantization_config"]["quant_method"] == "prismaquant"`.
- The exporter inlines the full quant config (config_groups + ignore) into
  `config.json`, so `from_config` gets everything; the shared `cb_codebook.*`
  tensors live in a sidecar `cb_codebooks.safetensors`, loaded **once** by the
  config via `get_current_vllm_config()` at weight-load time. **Zero vLLM-core
  monkeypatching.**

## CUDA kernel set (decode-GEMV + prefill)

Two JIT-built CUDA extensions. Numerics contract everywhere: identical weight
rounding to the reference decode (`w = bf16_rn(codebook · scale)`), fp8/fp4
activation QDQ bit-exact to the codec, and **fp32 accumulation** — so a CUDA
result differs from the reference only by summation **reassociation**, held to
`≤1 bf16 output ULP + a norm backstop` (the `_assert_triton_close` contract in
`tests/test_cuda_gemv.py`).

**`csrc/cb_gemv.cu`** (`cuda_ext.get_ext()`) — the decode path (M ≤ 16) + a
prefill expander:

- **Dense decode-GEMV** `cb_gemv_fp8` / `cb_gemv_fp4_v2`: one block per output
  row, warps stride the row's 256-weight superblocks, each superblock is decoded
  in registers (INV-1) and FMA'd against the activation. The **fp8** kernel runs
  a software-pipelined **double buffer** (prefetch superblock *s+WARPS* while
  decoding *s*): **bit-identical** and **+3–6%** (drift-immune interleaved A/B)
  because the dense kernel has few blocks and is latency-exposed. It is the
  default; `PRISMAQUANT_CB_FP8_SCHED=legacy` selects the single-buffer path for
  bisection. (The fp4-v2 dense kernel keeps single-buffer — double-buffering
  **regressed** it, since its heavier two-tier decode is compute-bound;
  `PRISMAQUANT_CB_FP4V2_SCHED=db` is the opt-in switch that measured the loss.)
- **Grouped MoE decode-GEMV** `cb_moe_gemv_fp8` / `cb_moe_gemv_fp4_v2`: one launch
  covers every routed *(token, expert)* pair of a layer (replaces a per-expert
  Python loop). The **fp4-v2 down-projection (w2)** grouped kernel has a
  schedule selector, `PRISMAQUANT_CB_W2_SCHED`:
  - **default** (unset) — the round-2 warp schedule: for a small superblock
    count (w2: `n_sb=6`) it drops the idle-warp 8-warp launch to ~3
    superblocks/warp (**2 warps**), **+50%** throughput on the Hy3 w2 shape
    (24.6–31.6% → 37.5–47.4% of the 273 GB/s bound). This **reassociates** the
    fp32 partial-sum vs `legacy` and is **served-KL-validated**. `w13` (`n_sb=16`)
    stays 8-warp, **bit-identical**. `PRISMAQUANT_CB_W2_WARPS` overrides the count.
  - **`legacy`** — the original 8/4-warp heuristic (numerics-preserving baseline;
    reproduces the pre-round-2 output exactly).
  - **`rowpack`** — round-3 experiment: one block owns `RPB` rows of one pair,
    staging the pair's activation in smem once and running `RPB` independent
    decode streams (targets the low K14/K16 rungs). Further-reassociated (same
    tolerance contract), pending its own served check. `PRISMAQUANT_CB_W2_ROWS`
    tunes `RPB ∈ {4,8,16}`.
  All schedule/double-buffer switches are read host-side in the launcher
  (CUDA-graph-capture-safe; no device reads, no new syncs).
- **`cb_expand_fp8`** — transient prefill expander: decodes the packed stream to
  a dense `[N,K]` e4m3 tile for a stock fp8 GEMM. This is the **large-M shipping
  answer** today (decode paid once), at the cost of materialising `[N,K]` in HBM
  (an INV-1 compromise the persistent-N work below aims to remove).
- `fp8_act_qdq`, `cb_moe_combine` — fused per-token fp8 QDQ and the deterministic
  expert-ascending bf16 combine.

**`csrc/cb_fused_gemm.cu`** (CUTLASS, separate JIT ext) — prefill:

- **`cb_fused_prefill_mm`** — decode-in-prologue fused GEMM (CUTLASS sm120
  collective: decode each B superblock into smem, then the FP8/FP4 tensor-core
  MMA — INV-1 **and** INV-2, bit-exact vs the passthrough `fork64`). Wins the
  **mid-M niche (17–128)** (1.04–1.45× vs serial); at large M every M-tile CTA
  re-decodes B, so transient-expand is preferred there.

**`csrc/cb_persistent_prefill.cu`** (experimental, off by default) — the
large-M endgame reference: a **persistent-N** kernel that decodes each B N-tile
**once** into smem and streams M through it (no `[N,K]` in HBM, INV-1). This is
an f32-FMA **schedule/correctness reference**, not the tensor-core perf path; the
go/no-go for the CUTLASS tensor-core version and the full design are in
`docs/nvfp4-cb-plan/persistent-n-prefill.md`.

## Tests

`tests/test_cb_kernels.py` runs the Triton path on the **real exported** 0.6B
tensors and (a) matches `nvfp4_cb_reconstruct @ x` to ≤1e-2 rel, (b) checks the
kernel's codeword extraction is bit-exact vs `nvfp4_cb_unpack`.
`tests/test_cuda_gemv.py` gates the `cb_gemv.cu` kernels (dense + grouped-MoE fp8
and fp4-v2, QDQ bit-exactness, the expander) against the Triton path and the
fp64 reconstruct; `tests/test_fused_prefill.py` and
`tests/test_persistent_prefill.py` gate the two prefill kernels.
