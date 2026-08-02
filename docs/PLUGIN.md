# Plugin reference

Operator-level reference for the out-of-tree vLLM plugin that serves the
**NVFP4-CB / FP8-CB** codebook formats. For installation and first-run problems
see [`INSTALL.md`](INSTALL.md) and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md);
for the format itself see [`SPEC.md`](SPEC.md).

The served path is Gridbook's packaged **native CUDA/CUTLASS kernel set** under
`gridbook/csrc/`. Native CUDA handles decode GEMV, codebook expansion,
activation QDQ, and routing/combine support; CUTLASS handles GEMM and grouped
GEMM. Gridbook defines, compiles, and dispatches no Triton operator and has no
Triton dependency or serving fallback. Required native operations fail closed
when unavailable. Importing the full plugin necessarily imports vLLM's Python
types; vLLM may load Triton for unrelated runtime components, but no CB group
dispatches through it. Non-NVIDIA
hardware is unsupported and unqualified; no ROCm backend or dispatch hook ships
in this release.

For FP8 activation quantization and scaled GEMM, Gridbook attests and calls the
registered `torch.ops._C` CUDA operators directly. It intentionally bypasses
`vllm._custom_ops`, whose convenience wrapper can select Triton on unsupported
shapes. Whole-Layer and whole-MoE dispatch permanently cross opaque Gridbook
custom ops; there is no inline-dispatch serving mode.

MoE activation dispatch follows the same direct-op rule. Gridbook never calls
vLLM's `apply_moe_activation` helper, whose SWIGLUSTEP arm is Triton. Supported
gated activations are attested during model load and invoke their registered
`torch.ops._C` CUDA operators directly; unsupported activations fail closed.

## Invariants

- **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) scales. The dense `[N,K]` weight is
  never materialized in HBM as a *model-wide* tensor — each superblock's weight
  tile is expanded inside the kernel, in registers, then consumed by the matmul.
  The large-M transient-expand prefill path materializes one layer's `[N,K]` tile
  at a time (expand → GEMM → free), a bounded, deliberate relaxation.
- **INV-2 (native tensor-core matrix kernels):** honored by the CUTLASS
  decode-in-prologue prefill kernel, which decodes each superblock into smem and
  feeds the FP8/FP4 tensor-core MMA directly. A quality-preserving FP4 path may
  use a native CUDA BF16 transient plus CUTLASS BF16 GEMM; that preserves the
  established activation bucket and is labelled separately from native W4A4.
- **INV-3 (native-only and fail closed):** capability probes may report a
  missing extension, but a serving caller requires the extension and raises an
  operation-specific error. No Gridbook CB call defines, compiles, or selects a
  Triton operator; ambient vLLM imports are outside this operator boundary.

## Scope

The producer/runtime boundary is machine-readable at
`gridbook/runtime_contract.json` and loadable without torch or vLLM through
`gridbook.runtime_contract.load_runtime_contract()`. It declares accepted
quantization names, serialized packing/type-size rules, supported CB rung
families, and producer-profile loader coverage. The plugin derives its own
registration aliases and top-level loader-module list from this file rather
than repeating them in Python. PrismaQuant pins an immutable Gridbook
commit and validates against this packaged contract; it does not vendor a
second runtime tree or maintain a parallel loader table.

- **Product** mode, ANY integer k in the format: even splits (NVFP4_CB_K16 → (8,8);
  FP8_CB_K44 → (11,11,11,11)) and ceil-first uneven splits per SPEC §1
  `bit_split` (K13 → (7,6); FP8 K30 → (8,8,7,7)) — encoder-anchored tests.
  Production availability is also gated by a native kernel contract for the
  concrete grid/mode/shape; a format-valid but unimplemented combination fails
  closed. Native dense FP4 serving in 0.5 accepts only unsigned product mode
  with v2 scale coding. A format-valid FP4-v1 product layer is rejected at
  model load because its exact every-M native quality expander is not
  implemented. **Signed** mode (S-rungs, n_sub=1: 8 sign bits + magnitude index into
  one half-grid table) was validated by a historical 18-test GPU battery on
  2026-07-22. It is **closed as research-only** by the K-vs-S head-to-head
  the same day: over 776 matched-rate per-(Linear, rung) comparisons the
  unsigned rungs won 79% of the time and the allocator placed 6 signed units
  against 147 unsigned, so S-rungs stay **off production menus**. The format
  stays in the spec for exotic weight geometries; no published artifact uses
  one. In 0.5, native dense serving also rejects a signed S-rung at model load:
  the format is valid and remains available for research codecs, but the exact
  every-M native prefill contract is not implemented for it. Full mode:
  spec-reserved, unimplemented.
- **Mixed containers are supported and shipping.** A config group carrying a
  `"scheme"` key is a CB group and is served by this plugin; a group without one
  uses the stock `compressed-tensors` vocabulary and is delegated to a real
  `CompressedTensorsConfig` that the plugin constructs (NVFP4, FP8_DYNAMIC);
  `ignore` entries become BF16 passthrough. The shipped 295B artifact serves 36
  vanilla-FP8 Linears this way, and the 27B's vision tower is a stock NVFP4
  W4A16 group. **Consequence:** an artifact's hardware requirements are the union
  of gridbook's and those of its delegated groups.
- **Single-GPU (`tp=1`) only** — there is no tensor-parallel handling for CB
  weights, and a live TP size above one now fails during model construction.
  `--enforce-eager` is the published-model configuration; mode-0
  `FULL_DECODE_ONLY` is also capture-correct with the permanent opaque dispatch
  and is being promoted through the model-size performance gates
  ([details](TROUBLESHOOTING.md#do-i-really-need---enforce-eager)).
- **BF16 activations only** — the shipping CUDA dense and grouped-MoE bindings
  require BF16. Gridbook no longer advertises FP16 to vLLM and therefore fails
  at dtype validation instead of crashing or changing dtype at a dispatch
  crossover.
- **Biasless public CB dense Linear only** — a non-`None` bias passed to the
  public dense CB method is rejected. The permanent opaque native operation
  does not expose a Gridbook-owned biased kernel, so 0.5 does not hide a
  framework bias add behind the serving boundary. Delegated non-CB groups keep
  their upstream method's contract.

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
  recognise stacked codebook expert tensors. HunYuan-V3-style shared-CB targets
  are aliased to the native CB Linear at construction, including the collapsed
  and nested MTP prefix forms. If the checkpoint's CB tensor can resolve only
  to a plain BF16 parameter, load fails with the target name; the retired
  decode-to-BF16 compatibility path is not available. The wrap is inert for
  non-CB checkpoints.

## CUDA kernel set (decode-GEMV + prefill)

JIT-built native CUDA/CUTLASS extensions. Numerics contract everywhere: identical weight
rounding to the reference decode (`w = bf16_rn(codebook · scale)`), fp8/fp4
activation QDQ bit-exact to the codec, and **fp32 accumulation** — so a CUDA
result differs from the independent PyTorch/FP64 reference only by summation
**reassociation**, held to `≤1 bf16 output ULP + a norm backstop` in the native
kernel tests.

**`gridbook/csrc/cb_gemv.cu`** (`cuda_ext.get_ext()`) — dense M≤8, grouped-MoE
M≤16, plus native expansion/support operators:

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
    tolerance contract), but measured negative and retained only for
    reproducibility. `PRISMAQUANT_CB_W2_ROWS` tunes `RPB ∈ {4,8,16}`.
  All schedule/double-buffer switches are read host-side in the launcher
  (CUDA-graph-capture-safe; no device reads, no new syncs).
- **`cb_expand_fp8`** — transient prefill expander: decodes the packed stream to
  a dense `[N,K]` e4m3 tile for CUTLASS W8A8. This is the **large-M shipping
  answer** today (decode paid once), at the cost of materialising `[N,K]` in HBM
  (an INV-1 compromise the persistent-N work below aims to remove).
- `fp8_act_qdq`, `cb_moe_combine` — fused per-token fp8 QDQ and the deterministic
  expert-ascending bf16 combine.

**`gridbook/csrc/cb_gemv_v2.cu`** (`cuda_ext.get_ext_v2()`) — required FP4-v2
quality expansion plus an optional alternate decode GEMV:

- **`cb_expand_v2`** is the exact BF16 expander used by every dense and routed
  FP4-v2 quality path. It is required independently of
  `PRISMAQUANT_CB_GEMV`; the selector changes decode GEMV scheduling, not this
  quality bridge.
- **`cb_gemv_v2_prepare`** performs the device-specific shared-memory
  attestation. The current implementation accepts only CUDA compute capability
  12.0 or 12.1, so 0.5 FP4-CB quality serving is Blackwell-only and fails at
  weight load elsewhere. The extension can be compiled and cached without a
  GPU; device preparation is intentionally deferred until model load.
- **`cb_gemv_v2`** is the optional smem-resident-dictionary decode schedule
  selected by `PRISMAQUANT_CB_GEMV=auto|v2`. The default `inherited` decode
  schedule still loads and prepares this extension because it needs
  `cb_expand_v2` for quality prefill.

**`gridbook/csrc/cb_fused_gemm.cu`** (CUTLASS, separate JIT ext) — prefill:

- **`cb_fused_prefill_mm`** — decode-in-prologue fused GEMM (CUTLASS sm120
  collective: decode each B superblock into smem, then the FP8/FP4 tensor-core
  MMA — INV-1 **and** INV-2, bit-exact vs the passthrough `fork64`). Wins the
  measured mid-M points (1.04–1.45× at M=32/64/128). Production FP8-CB dispatch
  considers it for **M=9–128** when its rung/layout/device predicates hold; at
  large M every M-tile CTA re-decodes B, so transient-expand is preferred. Its
  extension availability and selector mode are resolved during model load, not
  on the first eligible forward; changing the relevant environment value after
  load raises instead of silently changing residency or dispatch.

**`gridbook/csrc/cb_bf16_grouped_gemm.cu`** (CUTLASS, separate JIT ext) — the
quality-preserving BF16 bridge:

- **`cb_bf16_grouped_mm`** — consumes ragged expert segments and transiently
  expanded BF16 weights in one owned CUTLASS grouped GEMM. MoE prefill uses
  `E>1`; dense FP4-CB uses the same binding with `E=1` for every `M>8`. This
  retains Gridbook's established activation-QDQ contract and must not be
  described as the quality-red native-W4A4 experiment. Its current
  SM80-compatible `DefaultGemmGrouped` schedule is not Blackwell-optimized: on
  the recorded synthetic DSV4 shapes it measured 6–17% slower than segmented
  BF16 matmuls at warm steady state. It is a native-ownership/quality baseline,
  not yet a prefill-speed result.
  Its kernel schedule supports SM80+, but the FP4-v2 serving path as a whole
  still has the cc 12.0/12.1 floor imposed by `cb_expand_v2` device prepare.

**`gridbook/csrc/cb_fused_fp4_gemm.cu`** (CUTLASS, separate JIT ext) —
experimental native-FP4 prefill, **off by default**:

- `cb_nvfp4_quantize_rows` and `cb_nvfp4_quantize_static_lsq` are policies in
  one shared native-NVFP4 activation quantizer. `static_lsq` keeps the
  producer-attested `G` and vLLM-identical E2M1/SFA bytes, then computes the
  least-squares value for the existing row residual. No model payload changes.
- `cb_fused_fp4_prefill_mm_scaled` is the sole dense CB-weight decoder/GEMM for
  static, static-LSQ, and rowwise activation modes. The two-tier scale decoder
  coalesces four adjacent factors per lane-pair, and a host-side occupancy rule
  selects the existing concrete TileM128 or TileM256 runner. Multi-LUT layers
  use a checked N-tile map; single-LUT layers retain their original launch.
- `cb_fused_fp4_moe_grouped` is the qualified grouped arithmetic primitive with
  explicit TileM128/256 runners. MoE `static_lsq` calls the same shared
  quantizer and this unchanged grouped GEMM; it adds no second payload, weight,
  decoder, or matmul. Kernel parity and raw speed do not promote either flag;
  see the [dated served-evidence decision](audits/fused_nvfp4_enablement_2026-07-31.md).

**`gridbook/csrc/cb_persistent_tc.cu`**
(research source only, **not serving-reachable**) — the persistent-N schedule:
decode each B N-tile **once** into smem and stream M through it (no `[N,K]` in
HBM, INV-1), with phase 2 on the fp8 tensor cores. **Verdict: measured negative
for dense prefill** — parity-green but 2–5.7× slower than expand-then-GEMM at
27B shapes, because the CUDA expander had already cut the dense expand tax to
~10%. The serving selector, custom op, and JIT loader were deleted; this one
`.cu` remains for direct research tests behind
`GRIDBOOK_RESEARCH_PERSISTENT_TC=1` (`tests/test_persistent_tc.py` compiles it
itself). Its f32-FMA twin `cb_persistent_prefill.cu` — the schedule's
correctness reference, superseded verbatim by the tensor-core build — was
deleted on 2026-08-01 per
[`audits/ultraplan_perf_2026-08-01.md`](audits/ultraplan_perf_2026-08-01.md) §4.
The MoE analog of the idea is tracked in the canonical
[`kernel TODO`](../ROADMAP.md#kernel-todo-canonical).

## Environment switches

These are **escape hatches and A/B levers, not tuning knobs**. They describe the
current native-only dispatch; published numbers remain bound to the commit and
switches recorded with each run, including retired selectors where explicitly
labelled historical. Change one only to diagnose something or reproduce a dated
experiment.
(The prefix is `PRISMAQUANT_`, not `GRIDBOOK_`, for compatibility with existing
tooling and model cards — see the README's naming section.)

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_EXT_DIR` | `~/.cache/prismaquant-cb-ext` | Root of the JIT build cache. Point it at a persistent, writable directory in containers to avoid a ~30 s rebuild per start. Every module owns a SUBDIRECTORY of it — `main`, `v2`, and the identity-keyed `bf16_grouped/<digest>`, `fused/<digest>`, `fused_fp4/<digest>` — so no two ninja workspaces share artefacts and a changed source, header, lane macro, target or toolchain ABI lands in a new directory instead of serving a stale kernel. Upgrading Gridbook therefore costs ONE rebuild per affected module; the old directories are inert and can be deleted. |
| `PRISMAQUANT_CB_GEMV` | `inherited` | Which grouped FP4-CB **decode GEMV** serves a layer: `inherited` \| `auto` \| `v2`. All FP4-v2 quality paths build/load `cb_gemv_v2.cu` and run its cc 12.0/12.1 device prepare because that module also owns the required exact expander. Unset/`inherited` keeps the shipped decode schedule; `auto` or `v2` may additionally use the module's smem-resident decode GEMV where its occupancy predicate says it wins. That alternate decode is **not** bit-exact against `inherited` (reassociation class). FP8-only serves do not need the v2 module. An unknown spelling raises; changing it mid-process raises. |
| `PRISMAQUANT_CB_FUSED_FP4` | off | Dense fp4-CB native-FP4 prefill opt-in. `1`/`midm` use a fully attested artifact scalar for all prefill shapes / `16 < M <= 128`. `static_lsq`/`static_lsq_midm` keep that exact `G` and the native E2M1/SFA payload, but fit the existing per-row EVT residual by least squares; they add no model metadata, weight copy, decoder, or GEMM. `rowwise`/`rowwise_midm` instead derive an independent full-range scalar per runtime row and are the only fused choices accepted for legacy artifacts. All dense modes use one occupancy selector: TileM256 is chosen only for `M >= 256` and `ceil(M/256) * ceil(N/128) >= ceil(2*SM_count/3)`; otherwise TileM128 runs. The `*_midm` modes therefore always remain TileM128. All values are experimental and default-off; unknown spellings and mid-process changes fail. The K24 short exact gate passed, but long-context evidence is mixed and no >=4B/MoE served validation exists; see the [dated audit](audits/fused_nvfp4_enablement_2026-07-31.md). |
| `PRISMAQUANT_CB_FUSED_FP4_MOE` | off | Grouped-MoE fp4-CB native-FP4 prefill opt-in. Static `1`/`128` and `256` select TileM 128 and 256 and require both attested stage scalars. `static_lsq`/`static_lsq128` and `static_lsq256` select those same tiles while reusing the shared fixed-`G` LSQ quantizer. `rowwise`/`rowwise128` and `rowwise256` use independent runtime row scales and may serve legacy artifacts. An ineligible attempt records its cached reason and returns to the exact native BF16 quality bridge; unknown spellings and mid-process changes fail. Keep this off pending the dated audit's routed-quality, workload, and routing-policy gates. |
| `PRISMAQUANT_CB_BF16_SM120` | off | `1` routes the quality-preserving BF16 grouped bridge (every default NVFP4-CB prefill — dense `E=1` and routed MoE — plus the FP8-CB fallback) to the **sm12x-native** CUTLASS 3.x collective instead of the default SM80-schedule `DefaultGemmGrouped`. Compiled only for cc 12.0/12.1; resolved at model load and **fails the load** if unavailable rather than silently serving the other lane. Same operands, same single bf16 round, different FP32 reduction order — bit-gated against the torch reference, served protocol NOT run. The lane's collective has two A-source modes (bit-identical to each other, gated `torch.equal`): the row-padded copy, and an **in-mainloop A-row gather** that never materializes the padded activation; with the gather mode and the swizzle-group-aligned expert order, measured (GB10, pingpong 64×128×64): 1.13–1.37× the default bridge, and vs segmented BF16 matmuls **1.03–1.05× at T=128 and 1.10–1.15× at T=512** (the padded-copy mode's 0.83–0.92× T=512 deficit is closed at the construction level). The routed path still pays one host read of the per-expert block offsets per layer. Unknown spellings and mid-process changes raise. See [KERNELS](KERNELS.md#sm12x-native-grouped-bf16-opt-in-prismaquant_cb_bf16_sm120) and the [benchmark table](BENCHMARKS.md#2026-08-02-sm12x-grouped-bf16-lane-in-mainloop-a-row-gather--swizzle-aligned-tile-order-proposal-data). |
| `PRISMAQUANT_CB_FUSED_MIDM` | `1` | Resolved during model load. `0` skips the CUTLASS mid-M FP8 fused specialization and its JIT build; the exact native expansion/CUTLASS route remains. Any other supported setting is loaded/probed before the model becomes serve-ready. Changing the value later raises. |
| `PRISMAQUANT_CB_DECODE_CONTRACT` | `v1` | `v2` selects the scale-epilogue-hoist decode contract. Measured **null** on the served 27B; kept for reproducibility. |
| `PRISMAQUANT_DEBUG_PREFIXES` | off | `1` prints, per Linear, whether it resolved to a CB scheme or to a config-declared non-CB group — the first tool to reach for when memory use is higher than expected. |
| `PRISMAQUANT_PRELOAD_FUSED` | off | `1` independently attempts to build/preload both fused extensions (FP8-CB and NVFP4-CB) at registration so both arms of a served A/B can carry identical extension residency. Registration treats this as a capability probe; a serving caller still requires its selected native operation and fails closed (see the measurement side-effect in [`KERNELS.md`](KERNELS.md#a-measurement-side-effect-worth-knowing)). |

**Variables that no longer select anything.** `PRISMAQUANT_CB_DECODE` and
`PRISMAQUANT_CB_EXPAND` (whose `=triton` values are the ones you will find in
old scripts and model cards), the former
`PRISMAQUANT_CB_PREFILL={auto,stock,loop,batched,...}` family, and the former
dispatch-mode variable are **not retired *values* of live variables — no
dispatch reads any of them at all**, so setting one has no effect whatsoever,
not even a warning. Opaque whole-call dispatch is unconditional. Delete these
settings from old scripts and model-card commands rather than carrying them
forward as inert text.

The two `=triton` names have no reader whatsoever;
`PRISMAQUANT_CB_EXPAND`'s last one went with the Triton removal, and its
documentation row outlived it by a release until
[`audits/ultraplan_perf_2026-08-01.md`](audits/ultraplan_perf_2026-08-01.md)
§4 removed it. `PRISMAQUANT_CB_PREFILL` is still *named* in one place —
`scripts/validate_fused_nvfp4_ab.py` and the fused-NVFP4 validation harness
strip it from the environment and record that they did, so an inherited value
from an old shell cannot be mistaken for a measurement condition. That is a
sanitizer, not a selector. The regeneration command below is what keeps this
section honest: a documented variable the grep does not find is a ghost.

### The rest of them

The table above is what an operator touches. For completeness — because a
variable you find set in someone's script and cannot look up is worse than one
you can — this is **every** `PRISMAQUANT_*` variable in the tree. Regenerate the
list with:

```bash
grep -rho 'PRISMAQUANT_[A-Z0-9_]*' gridbook/*.py gridbook/csrc/*.cu | sort -u
```

**Kernel-schedule selectors** — read host-side in the launcher, so all of them
are CUDA-graph-capture-safe. Described inline in the kernel sections above.

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_FP8_SCHED` | auto | fp8-CB decode GEMV schedule variant. |
| `PRISMAQUANT_CB_FP4V2_SCHED` | auto | fp4-CB two-tier decode GEMV schedule variant. |
| `PRISMAQUANT_CB_W2_SCHED` | auto | `w2` grouped-MoE schedule. The rowpack variant measured negative and is kept as a recorded result. |
| `PRISMAQUANT_CB_W2_WARPS` / `..._W2_ROWS` | auto | Warp count / rows per block for that schedule (bisection). |

**MoE transient sizing overrides.** These change memory, not kernel family.
Production dispatch is fixed: grouped CUDA GEMV at M≤16; above 16, eligible
FP8-CB fused CUTLASS or exact expansion + owned CUTLASS grouped GEMM.

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK` | unset (byte-budget derived) | Explicit positive expert count per transient chunk. It overrides the byte budget; reduce it only when measured serve slack requires it. |
| `PRISMAQUANT_CB_PREFILL_CHUNK_BYTES` | `1073741824` (1 GiB) | Maximum BF16 weight transient used to derive the expert chunk. Activations, routing buffers, and allocator overhead are additional; if one expert exceeds the budget, chunk 1 is used with a warning. |

**Custom-op boundary**

The `prismaquant::cb_linear_forward` and `prismaquant::cb_moe_forward` opaque
ops are the sole production entry points. The switch that once exposed their
host-side branches to tracing no longer exists.

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE` | off | `1` restores the pre-hardening op-boundary behaviour. The name is the documentation: it re-opens a capture-unsafe boundary. |

## Tests

`tests/test_cb_kernels.py` uses an independent PyTorch decode on the **real
exported** 0.6B tensors and (a) matches `nvfp4_cb_reconstruct @ x` to ≤1e-2 rel,
(b) checks codeword extraction bit-exactness vs `nvfp4_cb_unpack`.
`tests/test_cuda_gemv.py` gates the `cb_gemv.cu` kernels (dense + grouped-MoE fp8
and fp4-v2, QDQ bit-exactness, the expander) against independent PyTorch/FP64
references. The grouped-BF16 bridge is gated against segmented BF16 matmul
references, while `tests/test_fused_prefill.py` gates the specialized prefill
kernels and `tests/test_persistent_tc.py` gates the research-only persistent-N
source behind its opt-in.
