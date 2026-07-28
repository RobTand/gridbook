# Plugin reference

Operator-level reference for the out-of-tree vLLM plugin that serves the
**NVFP4-CB / FP8-CB** codebook formats. For installation and first-run problems
see [`INSTALL.md`](INSTALL.md) and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md);
for the format itself see [`SPEC.md`](SPEC.md).

The **served path is the CUDA kernel set** (`gridbook/csrc/cb_gemv.cu`, JIT-built
via `cuda_ext.get_ext()`, plus the CUTLASS fused prefill extension). The
**Triton** path (`kernels.py`) was the original correctness-first reference and
remains the bit-exact fallback when `nvcc` is unavailable — correct everywhere,
but several times slower and not a serving target.

## Invariants

- **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) scales. The dense `[N,K]` weight is
  never materialized in HBM as a *model-wide* tensor — each superblock's weight
  tile is expanded inside the kernel, in registers, then consumed by the matmul.
  The large-M transient-expand prefill path materializes one layer's `[N,K]` tile
  at a time (expand → GEMM → free), a bounded, deliberate relaxation.
- **INV-2 (decode feeds a native-format MMA):** honored by the CUTLASS
  decode-in-prologue prefill kernel, which decodes each superblock into smem and
  feeds the FP8/FP4 tensor-core MMA directly. **Waived on the Triton path**,
  which decodes to bf16 and runs `tl.dot` — Triton cannot emit the Blackwell
  `sm_120`-family block-scaled FP4 MMA, which is why that path is a correctness
  fallback rather than a performance one.

## Scope

- **Product** mode, ANY integer k: even splits (NVFP4_CB_K16 → (8,8);
  FP8_CB_K44 → (11,11,11,11)) and ceil-first uneven splits per SPEC §1
  `bit_split` (K13 → (7,6); FP8 K30 → (8,8,7,7)) — CUDA + Triton paths,
  encoder-anchored tests. **Signed** mode (S-rungs, n_sub=1: 8 sign bits +
  magnitude index into one half-grid table): CUDA + Triton decode across all
  paths, GPU-validated (18-test signed battery, 2026-07-22); production
  menus gate it on a measured K-vs-S win. Full mode: spec-reserved,
  unimplemented.
- **Mixed containers are supported and shipping.** A config group carrying a
  `"scheme"` key is a CB group and is served by this plugin; a group without one
  uses the stock `compressed-tensors` vocabulary and is delegated to a real
  `CompressedTensorsConfig` that the plugin constructs (NVFP4, FP8_DYNAMIC);
  `ignore` entries become BF16 passthrough. The shipped 295B artifact serves 36
  vanilla-FP8 Linears this way, and the 27B's vision tower is a stock NVFP4
  W4A16 group. **Consequence:** an artifact's hardware requirements are the union
  of gridbook's and those of its delegated groups.
- **Single-GPU (`tp=1`) only** — there is no tensor-parallel handling for CB
  weights. `--enforce-eager` is the tested serving configuration
  ([why](TROUBLESHOOTING.md#do-i-really-need---enforce-eager)).

## Layout / registration

- `register_quantization_config("gridbook")` — vLLM auto-detects from
  `config.json["quantization_config"]["quant_method"] == "gridbook"`.
  `"prismaquant"` is registered as a **legacy alias** for artifacts exported
  before the rename; both dispatch to the same config.
- Published artifacts write `config.json["quantization_config"]` as a **pointer
  stub** — `{quant_method, format, config_file, codebook_file}` — with the full
  `config_groups` / `ignore` / `layout_version` in the `config_file` sidecar
  (`quant_config.json`), resolved lazily via `get_current_vllm_config()`. A fully
  inlined config (with `config_groups` present) is also accepted.
- The shared `cb_codebook.*` tensors live in the sidecar named by
  `codebook_file`, default **`cb_codebooks.pqcb`**, loaded **once** at
  weight-load time. Both sidecars are fetched from the Hub when the model is
  given as a repo id rather than a local directory.
- **No vLLM-core files are patched.** The plugin does wrap `load_weights` on
  specific *model classes* (HunYuan-V3 + its MTP drafter, Laguna, Qwen3.5-MoE +
  its MTP) whose loaders map MoE experts at the top level and would otherwise not
  recognise stacked codebook expert tensors. The wrap is inert for non-CB
  checkpoints.

## CUDA kernel set (decode-GEMV + prefill)

Two JIT-built CUDA extensions. Numerics contract everywhere: identical weight
rounding to the reference decode (`w = bf16_rn(codebook · scale)`), fp8/fp4
activation QDQ bit-exact to the codec, and **fp32 accumulation** — so a CUDA
result differs from the reference only by summation **reassociation**, held to
`≤1 bf16 output ULP + a norm backstop` (the `_assert_triton_close` contract in
`tests/test_cuda_gemv.py`).

**`gridbook/csrc/cb_gemv.cu`** (`cuda_ext.get_ext()`) — the decode path (M ≤ 16) + a
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

**`gridbook/csrc/cb_fused_gemm.cu`** (CUTLASS, separate JIT ext) — prefill:

- **`cb_fused_prefill_mm`** — decode-in-prologue fused GEMM (CUTLASS sm120
  collective: decode each B superblock into smem, then the FP8/FP4 tensor-core
  MMA — INV-1 **and** INV-2, bit-exact vs the passthrough `fork64`). Wins the
  **mid-M niche (17–128)** (1.04–1.45× vs serial); at large M every M-tile CTA
  re-decodes B, so transient-expand is preferred there.

**`gridbook/csrc/cb_persistent_prefill.cu`** and
**`gridbook/csrc/cb_persistent_tc.cu`**
(experimental, **off by default**) — the persistent-N schedule: decode each B
N-tile **once** into smem and stream M through it (no `[N,K]` in HBM, INV-1). The
first is an f32-FMA schedule/correctness reference; the second is the tensor-core
build. **Verdict: measured negative for dense prefill** — parity-green but 2–5.7×
slower than expand-then-GEMM at 27B shapes, because the CUDA expander had already
cut the dense expand tax to ~10%. Retained as a schedule reference and quarantined
behind `PRISMAQUANT_ENABLE_PTC=1`; **do not enable it**. The MoE analog of the
idea is still open — see [`ROADMAP.md`](../ROADMAP.md).

## Environment switches

These are **escape hatches and A/B levers, not tuning knobs** — the defaults are
what every published number used, and each non-default was chosen by measurement.
Change one only to diagnose something or to reproduce a documented experiment.
(The prefix is `PRISMAQUANT_`, not `GRIDBOOK_`, for compatibility with existing
tooling and model cards — see the README's naming section.)

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_EXT_DIR` | `~/.cache/prismaquant-cb-ext` | Where the JIT extensions are built and cached. Point it at a persistent, writable directory in containers to avoid a ~30 s rebuild per start. |
| `PRISMAQUANT_CB_DECODE` | `cuda` | `triton` forces the Triton decode path (much slower; bisection only). |
| `PRISMAQUANT_CB_PREFILL` | `auto` (fp8-CB) / `loop` (fp4-CB) | MoE prefill path: `auto` \| `stock` \| `grouped_fused` \| `loop`. `auto` measures the candidates per layer on the first prefill and caches the choice. `l2_pipeline` exists but is **diagnostic-only** — it wedged live serving three times. |
| `PRISMAQUANT_CB_FUSED_MIDM` | `1` | `0` skips the CUTLASS mid-M fused prefill kernel — including its JIT build, which is worth doing on non-`sm_120` GPUs where the build is doomed anyway. |
| `PRISMAQUANT_PREFILL_M_THRESHOLD` | `16` | Token count above which a dense Linear takes the prefill path instead of the decode GEMV. |
| `PRISMAQUANT_CB_CUDA_M_MAX` | `8` | Within the decode regime, the largest M handled by the CUDA GEMV before the Triton decode-GEMM takes over (measured crossover on GB10). |
| `PRISMAQUANT_CB_DISPATCH` | `op` | `inline` restores in-graph host branching instead of the opaque custom-op dispatch (A/B only; less CUDA-graph-safe). |
| `PRISMAQUANT_CB_DECODE_CONTRACT` | `v1` | `v2` selects the scale-epilogue-hoist decode contract. Measured **null** on the served 27B; kept for reproducibility. |
| `PRISMAQUANT_CB_EXPAND` | CUDA | `triton` restores the previous Triton expander (bisection). |
| `PRISMAQUANT_DEBUG_PREFIXES` | off | `1` prints, per Linear, whether it resolved to a CB scheme or fell through — the first tool to reach for when memory use is higher than expected (silent BF16 fallback). |
| `PRISMAQUANT_CB_PREFILL_TIMING` | off | `1` emits per-stage prefill timers. |
| `PRISMAQUANT_PRELOAD_FUSED` | off | `1` force-builds the fused extension at registration so both arms of a served A/B carry identical extension residency (see the measurement side-effect in [`KERNELS.md`](KERNELS.md#a-measurement-side-effect-worth-knowing)). |
| `PRISMAQUANT_ENABLE_PTC` | off | Builds the quarantined persistent-N kernel. **Do not set this** — measured negative and under a stability quarantine. |

Kernel-schedule switches (`PRISMAQUANT_CB_FP8_SCHED`, `..._FP4V2_SCHED`,
`..._W2_SCHED`, `..._W2_WARPS`, `..._W2_ROWS`) are described inline in the kernel
sections above. All schedule switches are read host-side in the launcher, so they
are CUDA-graph-capture-safe.

## Tests

`tests/test_cb_kernels.py` runs the Triton path on the **real exported** 0.6B
tensors and (a) matches `nvfp4_cb_reconstruct @ x` to ≤1e-2 rel, (b) checks the
kernel's codeword extraction is bit-exact vs `nvfp4_cb_unpack`.
`tests/test_cuda_gemv.py` gates the `cb_gemv.cu` kernels (dense + grouped-MoE fp8
and fp4-v2, QDQ bit-exactness, the expander) against the Triton path and the
fp64 reconstruct; `tests/test_fused_prefill.py` and
`tests/test_persistent_prefill.py` gate the two prefill kernels.
