# Ultraplan: performance gaps, redundancies, and the FP8-CB ↔ NVFP4-CB convergence plan

Date: 2026-08-01. Audited at commit `593f524` ("Merge native CUDA/CUTLASS-only
Gridbook 0.5.0 release"); §6 additionally audits the producer,
[prismaquant](https://github.com/RobTand/prismaquant), at `dca6f80` (0.5.2).
Static analysis of the full tree (Python dispatch,
`csrc/`, `cutlass_fork/`, build system, docs, tests); no GPU was available to
this audit, so every number quoted below is a *previously published* measurement
with its source cited, and every proposal states the evidence gate it must pass.
This document proposes; it does not promote. The [ROADMAP](../../ROADMAP.md)
status vocabulary (DEFAULT / OPT-IN / MEASURED NEGATIVE) and the
[NATIVE-PARITY](../NATIVE-PARITY.md) promotion rules apply to everything here.

Goals, as set for this audit:

1. Find gaps and redundancies.
2. Maximize performance; specifically **close the gap between FP8-CB and
   NVFP4-CB**.
3. Eliminate Triton usage; serve exclusively through owned CUDA support kernels
   and CUTLASS GEMMs.
4. Run **native CUTLASS kernels on `sm_120`/`sm_121`**. Other NVIDIA hardware is
   a secondary concern.
5. Establish how the producer decides **NVFP4 vs FP8-CB allocation at similar
   bit rate** — noting the competition runs everywhere up to **4.5 bpw**, since
   that is vanilla NVFP4's effective rate, not just at the CB ladder boundary
   (§6).

---

## 1. State of play (what is already settled)

- **Dense decode is at parity** (10.27–10.30 vs native 10.26 tok/s on the 27B;
  bandwidth-bound) — no work needed ([BENCHMARKS](../BENCHMARKS.md)).
- **FP8-CB mid-M (9 ≤ M ≤ 128) is fused CUTLASS and DEFAULT** (1.04×/1.26×/1.45×
  at M = 32/64/128), promoted through the quality gate.
- **Gridbook-owned Triton is gone.** 0.5.0 removed every Gridbook-defined
  Triton operator, dispatch, and fallback. What remains is *process-level*
  exposure through vLLM (§5) and stale documentation.
- **The measured-negative graveyard is closed** and stays closed: dense
  persistent-N (2–5.7× slower), blanket `grouped_fused` MoE default, w2
  rowpack, decode-contract-v2 hoist (null: 10.10 vs 10.13), L2-pinned pipeline
  (wedged serving 3×), N-chunked expand+GEMM overlap (0.46×, not bit-exact),
  naive inline graph capture. Nothing below reopens any of these without new
  profiling evidence, per [ROADMAP](../../ROADMAP.md).

The two headline measured gaps this plan attacks:

| Gap | Measured | Source |
|---|---|---|
| Dense large-M prefill (FP8-CB) | **1.44×** native TTFT (1.075 s vs 0.746 s, 27B) — transient expand doubles HBM traffic | BENCHMARKS |
| NVFP4-CB prefill quality path | BF16 bridge is **6–17% slower warm** than even segmented BF16 matmuls (0.826×/0.920× on DSV4 shapes); MoE expand is **~35% of layer time** at Laguna scale | BENCHMARKS 2026-08-01, KERNELS |

---

## 2. Why NVFP4-CB trails FP8-CB — the structural diagnosis

The gap is architectural, not a tuning miss. Four compounding causes:

**(a) Activation-contract asymmetry.** FP8-CB's fused prologue kernel consumes
the *same* per-token dynamic E4M3 QDQ as its non-fused route, so fusing it was a
pure kernel substitution and shipped as DEFAULT. The fused native-NVFP4 kernel
requires packed E2M1 activations + swizzled UE4M3 scale factors — a *different
served numeric contract* than the shipping fp32-emulated group-16 RTN QDQ.
Turning it on changes model output, which is why it sits behind
`PRISMAQUANT_CB_FUSED_FP4[_MOE]` and the six reconsideration gates in the
[2026-07-31 enablement audit](fused_nvfp4_enablement_2026-07-31.md).

**(b) The default NVFP4 path never executes an FP4 tensor-core GEMM.** For all
dense M > 8 and all MoE T > 16 it expands to **BF16** (2 bytes/weight — 4× the
transient traffic of FP8-CB's direct-to-E4M3 expand) and feeds
`cb_bf16_grouped_gemm.cu`, a **CUTLASS 2.x `DefaultGemmGrouped` on an
`arch::Sm80` schedule** (`m16n8k16` Ampere MMA, no TMA, no warp
specialization, no Blackwell smem budget). Dense FP4 additionally pays
grouped-kernel overhead for a plain GEMM by calling it with `E=1`.

**(c) FP4 has no mid-M lane at all.** FP8-CB owns M = 9–128 with the fused
kernel; the FP4 fused gate is `M > 16` *and opt-in*, so M = 9–16 always takes
the BF16 bridge, and with the flag unset so does everything above.

**(d) FP4 decode is compute-bound.** ncu on the 295B: SM 71% vs memory 44% —
the bit-exact two-tier scale compose costs ALU the smaller byte stream does not
repay. (FP8-CB decode is bandwidth-bound and at parity.)

Consequence: closing the gap = (1) replace the SM80 bridge with an sm120-native
CUTLASS BF16 kernel, (2) remove the transient HBM round trip on the quality
path with a contract-preserving CB→BF16 decode-in-mainloop schedule, (3) drive
the true-W4A4 fused lane through its existing promotion chain, (4) chip at the
decode ALU wall only with profiler evidence. In that order — (1) and (2) do not
change served numerics beyond reduction-order requalification, (3) does.

---

## 3. Plan of record

Phases are ordered by leverage ÷ risk. Each item names its evidence gate.
Per the one-payload rule (ROADMAP): no new packer, resident weight copy,
decoder, or second matmul per execution contract.

### P0 — hygiene and correctness bugs in the build/dispatch surface (days)

These are cheap, riskless, and some are latent correctness/perf bugs on the
primary target.

1. **Arch flags for the hot decode modules.** `get_ext()` and `get_ext_v2()`
   pass only `-O3` — no `-gencode` — so they inherit `TORCH_CUDA_ARCH_LIST`,
   and the stock vLLM base image's list (`"8.0 … 12.0"`) **omits 12.1**:
   outside the shipped Dockerfile (which bakes `12.1a` globally) a GB10 user's
   production GEMV runs from PTX JIT or a mismatched SASS target. Fix: derive
   `compute_XY/sm_XY` from the live device exactly as `get_bf16_grouped_ext`
   already does (`cuda_ext.py:560`). Gate: SASS check (`cuobjdump`) in the
   existing ext-symbol tests.
2. **Capability precheck on the fused-FP4 loader.** The FP8 fused loader
   rejects `cc ∉ {(12,0),(12,1)}` before building; the FP4 loader has no such
   check and dies minutes deep in nvcc on non-Blackwell. This is also ROADMAP's
   "arch precheck" item. One `if` + one test.
3. **Hash fork headers into the FP8 fused module identity.**
   `cb_fused_gemm.cu` includes three `cutlass_fork/` headers; none participate
   in its module name/build-dir identity, so editing a header can serve a stale
   cached kernel. The FP4 loader already solved this
   (`_FUSED_FP4_BUILD_INPUTS`); extract that mechanism and apply it to the FP8
   fused module (this is ROADMAP K0.3's "shared fused-JIT attestation",
   scoped to the build-identity half).
4. **Honour `PRISMAQUANT_CUTLASS_INCLUDE` in all CUTLASS loaders** (today only
   the FP4 loader reads it; the BF16-grouped and FP8-fused loaders cannot build
   without vLLM's bundled CUTLASS).
5. **Move `get_ext()`'s build dir into a subdirectory** of the cache root like
   every sibling module (pure tidiness, avoids ninja-artefact collisions).
6. **Dead-code removals** — see §4 ledger.
7. **Documentation truth repairs** — see §5 (stale HF cards describing a Triton
   fallback that no longer exists are actively wrong user-facing text).

### P1 — sm121-native CUTLASS BF16 grouped GEMM (the highest-leverage NVFP4 item)

Replace `cb_bf16_grouped_gemm.cu`'s SM80 `DefaultGemmGrouped` with a CUTLASS
3.x `CollectiveBuilder<arch::Sm120, OpClassTensorOp, bf16…>` kernel: TMA
warp-specialized mainloop, stages auto-carved against the 101,376-byte sm120
smem budget, persistent tile scheduler. Upstream CUTLASS 4.3.4 has **no SM120
ptr-array/grouped collective** (`cb_fused_gemm.cu:403-406` static_asserts
this), so grouping uses the **row-padded, tile-indexed, batch-mode-l-coordinate
construction that already exists twice in this tree** (FP8 fused grouped, FP4
fused grouped) — extract it into one shared header (§4, dedupe #1) and
instantiate it for BF16.

Why this first: it sits on the **default quality path** — every NVFP4-CB
prefill (dense `E=1` and MoE), plus the FP8-CB R3 fallback, flows through this
kernel — and it requires **no activation-contract change**, only
reduction-order requalification. It erases the measured 6–17% warm deficit and
plausibly more (the deficit is vs *segmented* matmuls; a native grouped
schedule also removes per-segment launch overhead).

- Expected: ≥ segmented-BF16 parity warm; target the fused-FP8 kernel's
  measured baseline-parity band (0.91–0.99× of native `cutlass_scaled_mm` for
  its analogous passthrough gate) as the sanity envelope.
- Gate: bit-level unit tests vs the torch reference (existing
  `test_bf16_grouped_cutlass.py` harness), then the NATIVE-PARITY grouped-MoE
  protocol — time the *whole routed operator*, never the inner GEMM alone.
- Keep the SM80 instantiation compiled-but-fallback for non-Blackwell (other
  hardware is secondary but the bridge is the one lane that is deliberately
  SM80-compatible today).
- The measured-negative `TileM=64` grouped prototype stays closed; this is a
  new collective, not a tile retune of the old one.

### P2 — contract-preserving CB→BF16 decode-in-mainloop (kill the transient round trip)

The quality path's remaining structural tax is materializing the decoded tile
in HBM (dense: ~2× GEMM bytes → the 1.44×; MoE: expand ≈ 35% of layer time).
The FP8 fused kernel already proves the mechanism: decode packed CB bytes into
the smem tile inside the CUTLASS producer stage. Do the same with a **BF16
mainloop** — decoded values are bit-identical to `cb_expand_fp4_v2` output, so
the served activation contract is untouched; only GEMM reduction order needs
requalification (same class of change the promoted FP8 mid-M kernel cleared).

Two sub-items, in order:

- **P2a — dense FP4 mid-M lane (9 ≤ M ≤ 128).** Mirror the FP8 fused mid-M
  win (1.04–1.45×) for FP4-CB with a CB→BF16 prologue on the P1 collective.
  This closes structural cause (c) outright and is DEFAULT-eligible because it
  is contract-preserving. Known ceiling: decode-in-prologue re-decodes B per
  M-tile, so it is mid-M-only by construction (FP8's measured 0.22× at M≈1400
  is the boundary evidence); gate it at M ≤ 128 like FP8.
- **P2b — ROADMAP K1.1: grouped MoE decode-in-mainloop for large M.** Decode an
  expert's weight tile **once**, stream routed/padded M rows through it —
  persistent-B along M, which is exactly what the roadmap already sanctions as
  "the next kernel" and is *not* the measured-negative dense persistent-N
  schedule (that one was CUDA-core/SM89-atom drafts with per-CTA N ownership).
  Gate: whole-routed-operator timing per NATIVE-PARITY, empty/uneven routing,
  stream- and graph-safety.
- **Dense large-M (the 1.44×) stays behind ROADMAP K1.3**: fresh roofline
  profiling *first*; only build a dense persistent-B/stream-M variant if the
  model shows a realizable win. The prior persistent-N implementation stays
  quarantined.

### P3 — true W4A4: the fused native-NVFP4 promotion chain (ROADMAP P0, unchanged order)

This is where the last of the byte advantage lives (E2M1 tensor-core rate, 4×
less activation traffic than W8A8 at the same M), and it is the only lane that
changes served numerics — so it moves strictly through the existing chain:

1. **K0.2** — produce a routed-MoE validation artifact with producer-attested
   stage-specific `input_global_scale` (today all fused MoE attempts correctly
   fail closed; the LFM A/B was fallback telemetry, not evidence).
2. **K0.5** — profile the raw operator gap vs stock `sm120_nvf4_mm_scaled`
   (split act-quant, packed-B decode, sync, MMA, epilogue, launch). The
   `static_lsq` screens (1.478×/1.741× offline wall) say the kernel is fast;
   the audit says it has *not* reached native-microkernel parity — find the
   delta before spending more quality-evidence budget.
3. **K0.4** — graph-safe TileM128/256 selector for grouped (dense selector
   exists), with full dispatch telemetry.
4. **K0.6** — the six-gate promotion run (≥4B dense + routed MoE, teacher KL /
   tasks, ≥10% representative p95 TTFT with no cell regressing >5%).
   `static_lsq` currently passes the short screen and fails the long-prompt
   NLL/PPL gate with a CI crossing the limit (p=0.3725) — the gate needs more
   windows and a ≥4B model, not relaxation.
5. Small dispatch completeness item while in there: the dense fused gate is
   `M > 16`; once P2a exists this is moot, but if P3 lands first, extend the
   fused-eligible range to M ≥ 9 so opt-in users don't hit the bridge at
   M = 9–16.

### P4 — orthogonal wins already sitting on the table

- **27B streaming gate for `FULL_DECODE_ONLY` CUDA graphs (K1.4).** The 0.6B
  canary measured **20.1%** end-to-end decode latency reduction (and +24%
  decode on the FP4 295B path) with exact replay. This is likely the single
  cheapest large decode win in the repo — it is a qualification run, not new
  code. Respect the ±17% extension-residency confound
  (`PRISMAQUANT_PRELOAD_FUSED=1`, matched arms); note `preload_fused_extensions`
  does **not** warm `get_ext`/`get_ext_v2`/`get_bf16_grouped_ext` today —
  extend it so "preload" is a full warm-up before any A/B.
- **`cb_gemv_v2` qualification on `sm_120` (K1.4)** — measured +5.97% live on
  Laguna decode; it is sm121-attested only. Same-session quality, soak, long
  prefill per roadmap.
- **K1.2 — FP8 fused rung coverage.** The fused mid-M kernel instantiates only
  K ∈ {28,32,36,40,44,48}; production permits every K28–K48, and the
  *published 27B artifact ships an 8-rung K36–K47 ladder* — so five of its
  eight rungs silently miss the fused mid-M lane and take expand+GEMM.
  Instantiate the missing rungs (smem table already parameterizes KBits) or
  encode the concrete route in the candidate identity so the allocator can't
  price an unbacked fast path.
- **FP4 decode ALU (structural cause d)**: no schedule work without a fresh ncu
  capture — the graveyard (decode-contract v2 null, fp4 double-buffer loss, w2
  rowpack negative) says speculative decode-GEMV retunes don't pay here. The
  v2 smem-dictionary lane is the one measured-positive direction; qualify it
  (above) before inventing new ones.

---

## 4. Redundancy and dead-code ledger

Reachability verified against `cuda_ext.py` loaders, Python dispatch, tests,
and CI scripts.

**Delete / archive (no runtime reachability):**

| Item | Evidence | Action |
|---|---|---|
| `csrc/sm120_fp8_gemm.cu` | Zero references repo-wide; superseded by `sm120_fp8_mm_fork`/`fork64` in `cb_fused_gemm.cu` (test-exercised). Also carries a misleading binding: `sm120_fp8_scaled_mm(a,b,scale_a,scale_b)` validates the scales then ignores them (epilogue hardcodes `{1.0f, 0.0f}`) | Delete; its historical role (baseline-parity gate, "build step 1") is documented in KERNELS |
| `cutlass_fork/sm120_cb_persistent_mma.hpp` | Included by nothing; draft of the abandoned persistent-N CUTLASS endgame | Delete (the *measured result* stays recorded in ROADMAP) |
| `csrc/cb_persistent_prefill.cu` | f32 correctness reference fully superseded by `cb_persistent_tc.cu` (same schedule, phase-1 "verbatim transplant") | Keep only `cb_persistent_tc.cu` under its research gate; delete the f32 twin and its test |
| `csrc/smem_probe_tilem.cu`, `csrc/toolchain_probe.cu` | Standalone `main()` binaries, no build integration | Keep (they generate the smem table baked into `cb_fused_gemm.cu`) but move under a `csrc/tools/` prefix and exclude from the wheel |
| `prismaquant::cb_expand_fp8_into` + `l2_pin_region`/`l2_reset_window`/`l2_unpin`/`l2_persisting_max_bytes` bindings | Registered, zero serving call sites; leftovers of the removed (thrice-wedged) L2 pipeline | Remove op + bindings + their tests; shrinks the contract-required symbol surface |
| `PRISMAQUANT_CB_EXPAND` in PLUGIN.md | Documented env var with **no reader in the code** | Delete the doc row (or the sanitizer list entry that keeps its name alive) |
| `cutlass_fork/*_orig.hpp` (67 KB) | Pristine diff baselines, compiled by nothing | Keep in repo for auditability; exclude from `package-data` so wheels shrink |

**Deduplicate (live code, two+ copies):**

1. **Row-padded tile-indexed grouped-GEMM construction** — implemented
   independently in `cb_fused_gemm.cu:400-560` and
   `cb_fused_fp4_gemm.cu:905-1110`. Extract one shared header; P1 becomes its
   third consumer, which is the forcing function.
2. **`ScaledFusion` EVT node tree** — verbatim ×3 (`cb_fused_gemm.cu`,
   `cb_fused_fp4_gemm.cu`, `smem_probe_tilem.cu`). Same shared header.
3. **MoE FP8 fused R1 (per-expert host loop) vs R2 (single grouped launch).**
   R2 supersedes R1; R1 survives only for extension builds lacking the grouped
   binding. Once P0.3's shared attestation guarantees the grouped symbols are
   present whenever the fused module is, retire R1 (one cascade branch, one
   `.tolist()` sync, and ~115 lines gone).
4. **Two FP4-v2 grouped decode GEMVs** (`cb_moe_gemv_fp4_v2` vs `cb_gemv_v2`)
   and **two expanders** — *intentional* (schedule A/B with load-time
   selection, and the v2 module owns the exact expander); keep, but this is
   the pair P4's v2 qualification should eventually collapse to one default.

**Dispatch-surface gaps (not dead code, but unbacked surface):** K1.2 rung
coverage (§3 P4); the doc-only ghost env vars; the `sm_120`-untested `v2`
device attestation narrowing the FP4 floor to cc 12.0/12.1 even on the
`inherited` GEMV path (acceptable today — Blackwell is the target — but it is
the single coupling that makes FP4-CB's hardware floor *narrower* than FP8-CB's,
worth a comment where it happens rather than an accidental discovery).

---

## 5. Triton elimination: what is actually left

The audit confirms **Gridbook defines, compiles, and dispatches zero Triton**
— and defends it well (opaque custom ops so Inductor never traces the bodies,
direct `torch.ops._C` binding instead of fallback-capable `vllm._custom_ops`,
hand-loading `vllm._C_stable_libtorch` by filename, an activation-op
allow-list that excludes the Triton SWIGLUSTEP branch, fail-closed `require_*`
everywhere, plus the AST ratchet in `test_no_triton_runtime.py`). "Use CUTLASS
exclusively" is therefore already true for every **GEMM**; the owned plain-CUDA
kernels that remain (decode GEMV, expanders, QDQ, combine) are the *correct*
tool for those bandwidth-bound / elementwise jobs — CUTLASS has no story for
M ≤ 8 GEMV, and rewriting them as CUTLASS would be motion, not progress.

What remains is **process-level and documentation-level**, in order of
actionability:

1. **Stale HF model cards (actively wrong).** All three staged cards in
   `docs/hf-cards/` still describe a live Triton decode fallback and quote a
   `[prismaquant-cb] … falling back to the Triton decode path` warning string
   that no longer exists in the code. Regenerate from `_TEMPLATE.md`.
2. **The Hy3 card recommends `--attention-backend TRITON_ATTN`** for the
   spec-decode config — the one place the repo tells users to turn Triton
   *on*. Re-qualify that config on a non-Triton attention backend (FlashInfer
   / FA family on sm12x) and update the card; if none passes, the card should
   say explicitly that attention is outside Gridbook's operator lane.
3. **Delegated compressed-tensors groups can dispatch Triton (and worse).**
   A mixed artifact's non-CB group goes to vLLM's backend ladder, where
   `emulation` is literally CUDA/Triton and Marlin silently converts a declared
   W4A4 into W4A16 (DELEGATED-NVFP4-MOE). No published artifact has such a
   group, so today this is latent. Implement the ROADMAP D0.2 preflight now,
   generalized: **fail closed at load if a delegated group resolves to a
   backend that either drops activation scales or is Triton-backed**, with the
   backend name in the error. That makes the README's no-Triton sentence
   enforceable rather than merely true-by-absence.
4. **Ratchet gaps in `test_no_triton_runtime.py`.** It bans only two `vllm.*`
   modules while `moe.py`/`config.py` legitimately import
   `vllm.model_executor.*` (which transitively initializes the `fused_moe`
   package whose members it bans), and `plugin.py`'s
   `importlib.import_module(module_path)` over `runtime_contract.json` model
   modules is invisible to the AST scan. Extend: (a) scan `scripts/` and
   `tests/` too, (b) assert the runtime-contract module list is exactly the
   allow-listed set, (c) add a GPU-lane runtime assertion that no
   `gridbook`-owned op execution imports `triton` into `sys.modules` beyond
   what bare vLLM already loaded (a delta assertion, since vLLM itself imports
   Triton unconditionally).
5. **The honest residual: Inductor.** Under `VLLM_COMPILE`, vLLM lowers the
   *surrounding* model graph to Triton kernels in-process. Gridbook's bodies
   are opaque to it by design, and this is outside the operator boundary the
   docs already scope. It cannot be eliminated from a vLLM serving process
   from plugin territory; the README's headline sentence should carry the same
   one-line scope qualifier the rest of the docs already use. (Torch and vLLM
   also pull `triton` as a transitive wheel dependency; the dependency test
   only guards the direct requirement, which is the correct claim to make.)

---

## 6. Producer-side allocation: NVFP4 vs FP8-CB at matched bytes (prismaquant)

Audited in the producer at `dca6f80`. The framing matters: this is **not** a
ladder-boundary question. Vanilla NVFP4 is **4.5 bpw effective** (E2M1 + one
UE4M3 scale per 16-block), so it is the standing alternative for *every* unit
at or below 4.5 bpw — `FP8_CB_K28`–`K36` (3.508–4.508) compete with it
per-layer at near-matched bytes (`NVFP4` = 4.500 sits between `FP8_CB_K35` =
4.383 and `K36` = 4.508), and below that the solver can fund a vanilla-NVFP4
promotion on one layer with cheaper CB rungs elsewhere. Gridbook ROADMAP D0.3
names exactly this pair of experiments.

### How the decision is made today (from the code, not the docs)

- **A pure quality-under-bytes optimizer.** `solve_allocation`
  (`allocator_solver.py:439`) is a multi-choice knapsack: minimize total
  predicted Δloss subject to a whole-artifact bpp/byte constraint (byte-budget
  mode at `allocator.py:3458`: min Δloss, ties → larger footprint), with an
  outer exact-payload loop because the DP's additive bytes model excludes the
  shared codebook sidecar. It is **global** — cross-layer byte-neutral trades
  are already expressible; each rung wins a unit iff its Δloss-per-extra-bin
  beats every other unit's claim on the same bins. **No latency or serving
  term exists anywhere in the objective or constraints.**
- **One flat candidate list, no family mechanics.** NVFP4_CB rungs, FP8_CB
  rungs, and vanilla NVFP4/FP8 are ordinary entries priced in the same
  ½·Fisher·MSE nats; exact ties resolve to the *cheaper* rung via menu order.
  Per-format calibrated-gain multipliers exist but production passes none
  (α = 1.0 everywhere).
- **Under shipped two-tier v2 coding the two CB ladders do not overlap.**
  NVFP4_CB tops out at K24 = 3.281 bpw; FP8_CB starts at K28 = 3.508 — a
  0.226 bpw hole (the K24 = K28-rate coincidence holds only under v1 coding).
  So within-CB "matched bpp" competition is nearly moot; the live contests are
  CB-vs-vanilla-NVFP4 at ≤ 4.5 bpw and global byte-neutral trades.
- **Menus are per-model script config, and outcomes are partly menu artifacts.**
  `FORMATS` env in the driver scripts: the 27B production run offered only
  four FP8-CB rungs (+ NVFP4/FP8/BF16); the 295B joint regen offered the full
  34-rung ladder + natives and assigned **36 dense/shared Linears to vanilla
  FP8 and zero to vanilla NVFP4 — "offered and never chosen."** Two caveats
  make that zero weaker than it looks: vanilla NVFP4 is **denied outright on
  packed MoE experts** (no stock-CT packed-expert emit path in the container),
  so the expert mass was never a fair contest; and the producer's own
  `format-speed-policy.md` flags the outcome as accuracy-only — *"zero
  selected NVFP4 units are circular evidence"* for speed.

### Three cost-model asymmetries that currently distort the call

1. **W4A4 vs W8A8 activation cost is priced only on the measured `output_mse`
   branch** of the cost precedence. Packed experts
   (`PRISMAQUANT_EXPERT_COST_SAMPLE=16` in every production script) and
   ladder-interpolated rungs (`CB_LADDER_INTERP=1`, ditto) fall to
   **weight-only Fisher pricing**, where the activation-contract difference is
   structurally invisible — NVFP4-CB gets credit for its cheaper index stream
   with none of its A-side cost, on most rows of a production run.
2. **Per-family fitted ladders, never cross-calibrated.** `_cb_ladder_split`
   fits NVFP4_CB_K and FP8_CB_K as separate curves (own anchors, holdout,
   law); anchors are activation-aware while interpolated rungs are
   weight-only, and the DP compares the mixed estimators as if identical.
3. **The producer models exactly one gridbook kernel gate** (`K % 256` for CB).
   It knows nothing of the `N % 8` (FP4) / `N % 16` (FP8) load gates,
   `n_sub`, the fused mid-M rung set, or LUT residency — it can legally assign
   a rung whose fast serving lane does not exist. (The 27B ladder pricing five
   unbacked fused-mid-M rungs, gridbook K1.2, is the same defect seen from the
   other end.)

### Plan — P5, producer-side (runs parallel to P1–P4; artifacts are where
allocation decisions become permanent)

- **P5a — make the quality axis fair first.** Extend activation-aware pricing
  (measured `output_mse`, or a per-family activation penalty calibrated once
  per model from a measured sample) to packed experts and interpolated rungs,
  and add a cross-family calibration check on the anchors. Gate: per-family
  predicted-vs-measured Δloss residuals on held-out layers must sit in
  family-symmetric bands before any cross-family verdict is published.
- **P5b — encode gridbook's real eligibility in the serving profile.**
  `out_features_multiple_of: 8` (FP4-CB) / `16` (FP8-CB), and the concrete
  fused-lane route (backed rung set, activation contract, fallback) as
  candidate metadata — the producer-side mirror of K1.2, so neither repo can
  price an unbacked lane.
- **P5c — implement the constrained Pareto solver the producer's
  `format-speed-policy.md` already specifies and defers** ("not yet
  implemented"): hard p95 TTFT / p95 ITL / p05 TPS / memory constraints as a
  second axis in `solve_allocation`, explicitly **no** λ-blended
  quality+latency objective (NATIVE-PARITY forbids it). Feed it the measured
  per-format × M-regime dispatch table from §2: until P1/P2 land, choosing
  FP8-CB over vanilla NVFP4 at ~4.5 bpw buys quality at a measured **1.44×
  dense-prefill cost**, and the allocator should see that trade rather than
  discover it at the release gate. The dormant precedent is in-repo:
  `mtp_rung_selection.py` already selects the MTP drafter rung
  throughput-optimally from served measurements.
- **P5d — run the D0.3 exact-rate experiments with the fixed pricing.**
  (i) `FP8_CB_K36` vs vanilla NVFP4 at matched exact whole-artifact bytes on
  dense units; (ii) below 4.5 bpw, byte-neutral sweeps where NVFP4 promotions
  are funded by lower CB rungs. Unlock packed experts for the contest via
  gridbook D0.2 (fail-closed packed-expert native delegation — which is also
  Triton-elimination item §5.3) so the 295B-class expert mass stops being
  decided by an emit-path gap.

The NATIVE-PARITY rule binds all of P5: kernel and per-layer timings propose;
only the served protocol promotes, and the performance denominator is the
fastest *globally feasible* assignment.

## 7. Explicit non-goals of this plan

- No revival of any measured-negative schedule without new profiling (§1 list).
- No second encoder, no TP > 1, no vLLM core patches (ROADMAP non-goals).
- No FP4-v1 performance work: resolve K2.1 as a load-time rejection unless a
  real v1 MoE artifact materializes.
- No new Ada/Hopper lanes: P0.2's precheck makes non-Blackwell fail fast and
  clean; qualification breadth (K2.3) stays behind the sm12x work.
- No relabeling: offline wall-clock and microbenchmarks in P1–P3 propose;
  only the NATIVE-PARITY served protocol promotes.

## 8. Sequenced summary

| Phase | Items | Served-path effect | Numerics risk |
|---|---|---|---|
| P0 | arch flags, FP4 loader cc gate, fused build-identity hashing, include override, dead code, doc truth | GB10-without-Docker perf bug fixed; stale-kernel class bug closed | none |
| P1 | sm121-native CUTLASS BF16 grouped GEMM (+ shared grouping header) | every default NVFP4 prefill, dense FP4 `E=1`, FP8 R3 fallback | reduction order only |
| P2 | a: dense FP4 fused mid-M CB→BF16; b: K1.1 grouped decode-in-mainloop; K1.3 dense roofline | removes FP4 mid-M hole; attacks the ~35% MoE expand tax and (evidence-gated) the dense 1.44× | reduction order only |
| P3 | K0.2 → K0.5 → K0.4 → K0.6 fused native-NVFP4 chain | true W4A4 rate where the gate passes | full quality-gate chain |
| P4 | 27B graph streaming gate, v2 GEMV sm_120 qualification, K1.2 rungs, preload completeness | ~20% decode latency candidate; +6% Laguna decode; 27B ladder gets its fused mid-M lane | qualification runs |
| P5 | producer: activation-fair pricing, eligibility metadata, constrained Pareto solver, D0.3 exact-rate runs | future artifacts allocate NVFP4-vs-FP8-CB on true quality *and* served latency across the whole ≤ 4.5 bpw regime | producer-side; measured-fit gated |

The FP8-CB ↔ NVFP4-CB gap closes from both ends: P1+P2 make the
*contract-preserving* NVFP4 path structurally identical to FP8-CB's (native
Blackwell collective, decode in the mainloop, no transient tile), and P3 gives
NVFP4 the *native-rate* lane FP8 already effectively has — but only through the
gates that keep the quality claim honest. P5 then makes the *allocation*
between them honest too: once the producer prices the activation contract on
every row, knows which serving lanes are actually backed, and carries served
latency as a hard constraint, "NVFP4 vs FP8-CB at matched bytes" stops being
circular evidence and becomes a measured decision.
