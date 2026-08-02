# Changelog

## Unreleased

- Add a **persistent-B grouped MoE decode-in-mainloop kernel**
  (`csrc/cb_moe_persistent_b.cu`, ROADMAP K1.1 / audit §3 P2b), the FP4-CB
  answer to the ~35%-of-layer-time MoE expand tax. A CTA owns one
  `(expert, N-tile)`, decodes that weight tile from packed CB bytes into
  shared memory **once**, and streams the expert's routed rows through it with
  the M-loop inside the kernel — so the `[E,N,K]` BF16 transient is never
  written or read back and unrouted experts cost two int32 loads. It consumes
  the **exact** `expert_ends` segments the default quality path already builds
  (no padded rows, no host read anywhere on the path), and its launch geometry
  is `E × ceil(N/TN)`, a function of the layer shape alone, so the call is
  CUDA-graph capturable as it stands. The mainloop is hand-assembled from
  `mma.sync.m16n8k16` / `ldmatrix` / `cp.async` rather than a CUTLASS
  collective, because no CUTLASS grouping construction on this architecture
  permits an in-kernel M-loop over one expert's segment. This is **not** a
  revival of the retired dense persistent-N schedule: it is MoE-only with no
  dense entry point in the translation unit, it uses tensor cores, and its win
  comes from amortizing a decode that kernel never had. Dense large-M stays
  behind K1.3.
  The weight decode is **bit-identical to `cb_expand_v2`** — a tested fact, not
  an asserted one: `cb_moe_persistent_b_decode` exposes the mainloop's own
  decode stage and the suite compares it to the expander with `torch.equal`
  across the k=12–24 rungs, on full and windowed row ranges. Activations are
  the same exact group-16 RTN QDQ payload and the epilogue rounds once to
  bf16, so only the FP32 reduction order changes (reassociation-class, gated
  at parity with per-segment `F.linear`). **Opt-in** behind
  `PRISMAQUANT_CB_MOE_PERSISTENT_B=1`, FP4-CB-v2 layers on cc 12.0/12.1,
  resolved at model load and failing the load — including the
  `..._CFG` tile override, validated against what the build compiled — rather
  than silently serving the other route. With the flag unset the dispatch is
  byte-for-byte what it was.
  Measured on nine **whole-routed-operator** cells (routing + QDQ + both
  projection stages + activation + combine, per the NATIVE-PARITY grouped-MoE
  rule): **1.05–3.36× over the default bridge and 1.04–3.02× over the pingpong
  bridge, winning every cell**, with the deleted expansion measuring
  **20.9–46.7%** of the default operator and *not* shrinking with expert count
  — at `E=128` it is 38–47% at every token count, because the expansion pays
  for every expert whether the router used it or not.
  `scripts/bench_moe_persistent_b.py` reproduces the table; proposal data only,
  served protocol not run.
- Harden that kernel against three defects a dedicated graph/stream audit
  found, two of which no test in the suite could have caught. (1) A **WAR race
  on the A double buffer**: the `cp.async` prefetch for stage `st+1` targets
  the same buffer `mma(st-1)` is reading and sat above the barrier that
  separates them, so a lagging warp could have its A fragment overwritten
  mid-MMA. Latent — not reproducible in ~1.15e9 stage×warp opportunities under
  SM contention — but reproducible at 400 cycles of injected warp skew, where
  relative error went from 1.6e-3 to 0.3-0.6. The prefetch now issues below the
  barrier. (2) `expert_ends` is DEVICE data whose **values** the host cannot
  check, and the kernel used them unclamped for both the row predicate and the
  store predicate; a hand-built out-of-range entry through the public binding
  produced an illegal access. The segment is now clamped to `[0, P]` with one
  register `min`. (3) The routing counted with `torch.bincount`, whose CUDA
  implementation host-syncs (it sizes its output from `self.max().item()`), so
  the lane's "no host read on this path" claim was false and the operator could
  not be captured end to end. It counts with `scatter_add_` now — identical
  integers, pure device work — and a test captures the whole routed operator
  and a negative control proves the avoided call really does break capture.
  Also added: alignment `TORCH_CHECK`s for the operands whose instruction
  selection assumes them, an `NATOM % 2` static assert, and a device gate that
  states the schedule's real shared-memory need instead of the whole budget.
- Record two measured-negative results from that kernel's tile sweep rather
  than only its winners: the `128×128` and `256×64` tiles both fall to one CTA
  per SM and never won a sweep cell — `256×64` halves the decode repetition at
  large rows-per-expert, exactly the regime it should own, and still lost — and
  a resident codebook in shared memory (`cb_gemv_v2.cu`'s DS=2 idea) cost 1.9×
  by tripping over the ~1 KiB the hardware reserves per CTA while buying ~1%
  where it fit. `csrc/tools/persistent_b_probe.cu` reproduces the budget table
  and the binding now `TORCH_CHECK`s the two-CTAs-per-SM floor that both
  results are about.

- Add an **sm12x-native CUTLASS 3.x lane** to the quality-preserving BF16
  grouped bridge (`csrc/cb_bf16_grouped_gemm.cu`, audit §3 P1): TMA
  warp-specialized mainloop, stages carved out of the sm120 shared-memory
  budget, and the row-padded tile-indexed grouping the two fused kernels
  already use — now extracted into `csrc/cb_grouped_common.hpp` and consumed by
  all three, with static_asserts proving the shared EVT/gate types are the ones
  each file spelled verbatim before. Upstream has no sm120 grouped collective
  and its sm120 builder refuses 16-bit input, so the collective is assembled
  from the 16-bit forms of the builder's own choices and the expert
  `l`-coordinate selection lives in a thin mainloop fork
  (`csrc/cutlass_fork/sm120_bf16_expert_mma.hpp`). The DEFAULT SM80-schedule
  lane is unchanged on every device. The new lane is **opt-in** behind
  `PRISMAQUANT_CB_BF16_SM120=1`, resolved at model load and failing closed:
  it is bit-gated against the torch reference (both lanes and a per-segment
  `F.linear` share one relative-L2 band), but it changes the FP32 reduction
  order. Measured on the GB10 with the compiled rung (pingpong 64×128×64, 3
  stages): 1.18–1.27× the bridge it would replace and 1.02–1.05× segmented
  BF16 matmuls at T=128, 0.83–0.92× segmented at T=512. The residual is the
  tile-indexed construction's ragged row padding, not the schedule — with the
  padding removed the same kernel runs 1.08–1.13× segmented — so the next step
  is a TileM ladder or an in-mainloop A gather, not more tuning. Proposal data
  only, served protocol not run; `scripts/bench_bf16_grouped_sm120.py`
  reproduces the tables.
- Key the grouped-BF16 module's JIT identity like the two fused modules: its
  packaged sources, Gridbook headers, compiled-in lane macro, target and
  toolchain ABI decide the module name and the `bf16_grouped/<digest>` build
  directory. One rebuild per user, once.

- Delete the unreachable native sources identified by the dead-code ledger in
  `docs/audits/ultraplan_perf_2026-08-01.md` §4: `csrc/sm120_fp8_gemm.cu` (the
  spent CUTLASS baseline-parity gate, whose binding validated per-token/
  per-channel scales it then ignored), `csrc/cutlass_fork/sm120_cb_persistent_mma.hpp`
  (an unbuilt draft of the abandoned persistent-N endgame), and
  `csrc/cb_persistent_prefill.cu` with its test (the f32 twin of the retained
  research kernel `cb_persistent_tc.cu`). The measured verdicts these kernels
  produced stay recorded in ROADMAP and KERNELS.
- Remove the `prismaquant::cb_expand_fp8_into` custom op and the
  `l2_pin_region` / `l2_reset_window` / `l2_unpin` / `l2_persisting_max_bytes` /
  `l2_max_window_bytes` extension bindings — registered surface with zero
  serving call sites, left behind when the L2-pinned per-expert scratch
  pipeline was removed from production dispatch. The allocating
  `prismaquant::cb_expand_fp8` prefill expander is unaffected.
- Shrink the wheel: standalone developer binaries move to
  `gridbook/csrc/tools/` and, with the pristine `cutlass_fork/*_orig.hpp` diff
  baselines, are now sdist-only. The repository and the source distribution
  keep them for auditability; `check_dist.py` and `tests/test_release_metadata.py`
  gate the split in both directions.
- Delete the `PRISMAQUANT_CB_EXPAND` row from the environment-switch table. The
  variable has had no reader since the Triton removal; documented switches that
  do nothing are worse than undocumented ones.
- Fail closed at model load when a delegated `compressed-tensors` group
  resolves to a backend that is Triton-backed, discards the declared activation
  scales (Marlin's NVFP4 W4A4 → W4A16 conversion), or is unaudited for an NVFP4
  W4A4 declaration. The error names the resolved backend class, the group, and
  the violated contract; there is no environment-variable bypass. Weight-only
  declarations are unaffected, including the published 27B's stock NVFP4 W4A16
  vision tower, which vLLM legitimately serves on Marlin.
- Extend the no-Triton ratchet to `scripts/` and `tests/`, pin
  `runtime_contract.json`'s dynamically imported model-module list to a
  reviewed allow-list, and add a GPU-lane assertion that executing a
  Gridbook-owned op imports no Triton module vLLM had not already loaded.
- Correct the staged HF card blocks and the README's no-Triton sentence: the
  cards no longer describe a Triton decode fallback or quote its retired
  warning string, the Hy3 card documents `TRITON_ATTN` as the attention backend
  measured at publication rather than a recommendation, and the README carries
  the operator-lane scope qualifier the other docs already use.
- Compile every JIT extension for the live device's compute capability instead
  of inheriting `TORCH_CUDA_ARCH_LIST`. The stock vLLM base image's list omits
  `12.1`, so outside Gridbook's own container a GB10 ran the production decode
  GEMV from PTX JIT or against a mismatched SASS target. A build host with no
  visible GPU now reports each module unavailable with an actionable reason;
  compile-only environments pin the capability for the duration of the build,
  as the image's prewarm step already did for the CUTLASS modules.
- Reject non-Blackwell devices in the fused NVFP4-CB loader before any CUTLASS
  include discovery or build work, as the fused FP8-CB loader already did. A
  non-Blackwell GPU no longer spends minutes inside nvcc during a first request.
- Hash `cb_fused_gemm.cu`'s three `cutlass_fork` headers into the fused FP8-CB
  module identity, which now keys both the extension name and the build
  directory (`fused/<digest>`), so editing one of those headers can no longer
  serve the previously cached kernel. Because a module that loads is therefore
  built from current sources, the fused FP8-CB contract now requires the
  grouped-MoE bindings alongside the dense entry point.
- Honour `PRISMAQUANT_CUTLASS_INCLUDE` in every CUTLASS loader; grouped-BF16
  and fused FP8-CB previously could not build without vLLM's bundled copy. A
  set-but-wrong value fails with the missing header named rather than silently
  compiling against a different CUTLASS.
- Build the main decode extension in `<cache>/main`, like every sibling module.
  An existing cache rebuilds it once.
- Retire the FP8-CB routed-MoE per-expert fused host loop. The single grouped
  launch supersedes it, and its only remaining reason to exist — an extension
  build carrying the dense fused binding without the grouped one — is now
  impossible. Constraint misses fall back directly to the exact native BF16
  expansion plus the owned CUTLASS grouped bridge.
- ROADMAP K0.2 (consumer half): recognise and verify PrismaQuant's routed-MoE
  stage attestation. `gridbook/nvfp4_activation_contract.py` accepts both the
  v1 record and the new stage-attested
  `prismaquant.nvfp4_w4a4_activation.v2` record, whose whole-model
  `target_names`/`target_count`/`target_values_sha256` fields are bit-identical
  to v1 (the digest framing constant stays pinned at the v1 literal). A v1
  record carrying a stage section, or a v2 record without one, fails closed.
  The new reader validates every packed FusedMoE module's `w13`/`w2` pair,
  their physical targets, policy coherence, stage-legal calibration sources,
  and both the per-stage and section digests.
- The fused-NVFP4 validation harness emits a machine-readable K0.2 readiness
  verdict (`attested_and_verified`, `missing_stages`, `digest_mismatch`,
  `not_attested`, `contract_absent`, `artifact_unreadable`) computed from the
  artifact alone — no GPU, no vLLM, no model load — naming the failing module
  and stage. Both `scripts/validate_fused_nvfp4_ab.py` and
  `scripts/validate_fused_nvfp4_three_arm.py` consume it as a core integrity
  gate and promotion-contract precondition: a routed-MoE A/B against an
  unattested artifact is now reported as `fallback_telemetry_not_evidence`
  rather than publishing a zero-difference fallback comparison. Dense runs are
  unaffected — a dense-only artifact carries no stage section and stays valid.
- Serving dispatch is unchanged: `moe.py`, `linear.py`, and `config.py` were
  not touched, and `validate_payload` still verifies exactly what it always
  verified.

## 0.5.1 — 2026-08-01

- Documentation-only release. Add the ultraplan performance audit
  (`docs/audits/ultraplan_perf_2026-08-01.md`): the FP8-CB ↔ NVFP4-CB
  convergence diagnosis and phased plan of record (P0–P5), the redundancy and
  dead-code ledger, the remaining process-level Triton surface, and the
  producer-side NVFP4-vs-FP8-CB allocation analysis at matched bytes
  (prismaquant @ `dca6f80`). No runtime, kernel, packaging, or dispatch
  changes; the served behavior of 0.5.0 is unchanged.

## 0.5.0 — 2026-08-01

- Remove Gridbook's Triton dependency and every Gridbook-defined Triton
  operator/dispatch/fallback path. Required CB operators now use native
  CUDA/CUTLASS and fail closed when the selected native extension is
  unavailable. Ambient vLLM may still import Triton for unrelated components.
- Route dense FP8-CB through CUDA GEMV at M≤8, fused CUTLASS at M=9–128 when
  eligible, and CUDA expansion + CUTLASS otherwise. Route dense FP4-CB M>8
  through exact BF16 expansion + the owned CUTLASS grouped bridge (`E=1`).
- Make the exact FP4-v2 quality path correct for fused projections with
  distinct per-role codebooks by coalescing contiguous row segments and
  expanding each segment against its matching native LUT before the CUTLASS
  bridge. Device-prepare the required v2 expander at model load; the 0.5 FP4
  serving floor is therefore CUDA cc 12.0/12.1, not the grouped GEMM's
  standalone SM80 floor.
- Replace selectable MoE stock/loop/auto/L2 prefill modes with fixed native
  dispatch: grouped CUDA GEMV at M≤16, then eligible quality-green FP8 fused
  CUTLASS or exact BF16 expansion + the owned CUTLASS grouped bridge. Prior
  Triton and Laguna chunk-expander results remain historical until this path is
  rebenchmarked.
- Resolve every serving-reachable extension and optional-kernel mode during
  model load, reject native shape-alignment misses before first forward, and
  reject mid-process FP8 fused-mode changes. Native dense 0.5 is intentionally
  biasless and accepts FP4 product-v2 only; signed S-rungs and FP4-v1 remain
  format-valid research inputs but fail the serving load gate.
- Make the whole-Layer/MoE opaque custom op the permanent dispatch boundary;
  remove the former inline-dispatch environment escape.
- Bind FP8 activation quantization and scaled matrix multiplication directly to
  vLLM's registered native CUDA operators after ABI/shape attestation. Gridbook
  no longer calls the fallback-capable `vllm._custom_ops` convenience wrappers.
- Compile the optional fused FP8-CB CUTLASS extension for the concrete
  `sm_120a`/`sm_121a` target derived from the serving GPU. A generic `sm_120`
  or `sm_121` build can load but device-assert on its first architecture-
  conditional tensor-core instruction, so it is no longer accepted.
- Alias HunYuan-V3 shared-CB projections, including their MTP-nested forms, to
  native CB Linears. If a CB tensor instead resolves to a plain BF16 parameter,
  model load now fails rather than decoding the weight into an upstream Linear.
- Keep the new grouped-BF16 bridge's performance status explicit: its generic
  SM80-compatible CUTLASS schedule is a quality/ownership baseline, not a
  Blackwell-optimized kernel, and measured 6–17% slower than segmented BF16
  matmuls on warm synthetic DSV4 shapes. This informational microbenchmark is
  not a DSV4 artifact qualification or a release promotion gate.
- Prewarm and verify the main, FP4-v2, and grouped-BF16 required extensions in
  the container image. FP8 fused prewarm remains a native optional
  specialization; experimental fused FP4 prewarm is an explicit build option.

[Full diff](https://github.com/RobTand/gridbook/compare/v0.4.2...v0.5.0)

## 0.4.2 — 2026-08-01

- Add a fixed-scale least-squares native-NVFP4 activation policy for dense and
  grouped-MoE prefill. It preserves the producer-attested global scale and
  vLLM-identical packed activation payload, changes only the existing per-row
  output residual, and reuses the same packed weights, decoder, and GEMM.
- Improve the fused two-tier scale decoder by 7.9–23.4% across the measured
  production-shape matrix, and select the existing TileM128/256 dense runners
  by occupancy. Selected cells improve 22.6–50.3% without copying a kernel
  body or changing output bits.
- Keep fused NVFP4 default-off while publishing the full evidence boundary.
  Qwen3-0.6B K24 passed its exact 6×128 quality gates at 1.478× offline speed;
  exact chunked 2×512 reached 1.741× but failed the PPL point gate, while the
  32-window point estimate passed without establishing non-inferiority.
- Extend the same-process A/B harness with exact chunked-prefill attestation,
  dense TileM route telemetry, MoE LSQ selection, and explicit fail-closed
  reasons. Legacy MoE artifacts without serialized activation scales are
  rejected rather than silently presented as fused measurements.
- Expand CPU/CUDA coverage for packed activation parity, static and LSQ scale
  contracts, multi-codebook routing, malformed storage, streams, graphs,
  grouped routing, selector isolation, and current-source provenance.

[Full diff](https://github.com/RobTand/gridbook/compare/v0.4.1...v0.4.2)

## 0.4.1 — 2026-08-01

- Keep dense and grouped-MoE fused native-NVFP4 prefill default-off after a
  teacher-backed LFM2.5 stop gate rejected promotion (mean full-vocabulary KL
  0.2472, PPL +5.65%, and teacher-to-candidate KL +0.1311 over the conservative
  path). The native activation bucket, not the decoded weight arithmetic, is
  the dominant change.
- Harden the opt-in fused CUDA implementation: validate device, dtype,
  contiguity, shape, storage, codebook, scale, and expert-ID contracts; launch
  on the active stream; trap invalid routed experts; and fix the concrete
  TileM128/TileM256 grouped runners for current CUDA toolchains.
- Add operator-level coverage for SASS instruction selection, every
  fused-eligible FP4 rung under both layouts, oversized-rung rejection, ragged
  shapes, non-default streams, CUDA graphs, malformed
  inputs, routing, real MoE stage shapes, and stagewise numerical attribution.
  The final CUDA gate passed 68/68 with no skips.
- Add immutable same-revision Hub loading for configuration, checkpoint
  headers, and codebook sidecars; preload both fused extensions independently;
  and wire the LFM top-level expert loader.
- Add fail-closed preparation and same-session teacher A/B tooling for fused
  enablement, with full-vocabulary metrics, exact artifact/runtime provenance,
  and explicit quality thresholds before a report can be promotion-complete.
- Make this repository the sole owner of Gridbook runtime code, CUDA sources,
  tests, packaging, and releases. Producer repositories consume a packaged
  runtime contract at an immutable commit; no Gridbook source tree is mirrored.
- Remove the unreachable, unvalidated ROCm delegation hook and its permanently
  skipped parity test. AMD support can return as a normal Gridbook feature once
  it has hardware access and served qualification.

[Full diff](https://github.com/RobTand/gridbook/compare/v0.4.0...v0.4.1)

## 0.4.0 — 2026-07-31

- Keep dense and grouped-MoE fused native-FP4 prefill behind explicit opt-in
  gates while its different activation contract awaits a served quality A/B.
  The conservative default also avoids the grouped kernel's severe padding
  amplification immediately above its 16-token dispatch boundary.
- Fail closed during model construction on unsupported serving contracts:
  Gridbook now advertises the BF16 dtype its native bindings actually accept,
  rejects FP8-CB below `sm_89`, rejects tensor parallelism above one, validates
  `PRISMAQUANT_CB_PREFILL`, and freezes that selector for the process lifetime.
- Deduplicate exact shared codebook references in fused dense layers, while
  routing layouts with distinct or offset codebooks away from kernels that
  cannot represent them.
- Release unloaded model layers from the compiled-dispatch registry with weak
  ownership and monotonic IDs, and serialize concurrent cold CUDA extension
  loads without adding locks to completed hot-path lookups.
- Reduce avoidable MoE-prefill selection/allocation overhead without changing
  quantization, routing, or kernel arithmetic.
- Require a dotted child boundary when resolving MoE targets, so sibling
  modules such as `experts2` cannot be mistaken for the live expert stack.
  This fix was extracted from Jason Wong's contribution series
  ([@jsconsultancy](https://github.com/jsconsultancy)).
- Add a fail-closed streaming native-parity harness that binds whole-artifact
  bytes, installed software and runtime identity, server evidence, workload,
  phase-specific latency, activation/backend/fallback contracts, and paired
  quality evidence. A successful single-arm report is measurement evidence,
  not by itself a parity or release claim.
- Revalidate `FULL_DECODE_ONLY` CUDA graphs with that release's then-default
  opaque dispatch:
  changed inputs replay exactly against eager at capture sizes 1 and 4, and a
  close-rate 0.6B canary narrows Gridbook decode to 5.9% of native. This remains
  approximate evidence: the CB artifact is 0.154% larger and compares W8A8
  against native W4A4 rather than isolating weight encoding alone.

[Full diff](https://github.com/RobTand/gridbook/compare/v0.3.0...v0.4.0)

## 0.3.0 — 2026-07-31

This release incorporates reviewed contributions from Jason Wong
([@jsconsultancy](https://github.com/jsconsultancy)), originally supplied as
the JW2026 patch set, together with integration fixes and expanded coverage.

- Align ignore matching with compressed-tensors and verify external codebook
  provenance before loading.
- Reject malformed codebooks and incompatible native extensions instead of
  silently degrading or risking incorrect numerics.
- Add single-launch CUDA FP4 activation QDQ, measured at 15.7–26.3× operator
  speedups on GB10.
- Add an opt-in shared-memory grouped-MoE decode kernel. Jason's Laguna release
  measured 24.993 versus 23.585 tok/s on GB10, a 5.97% end-to-end improvement.
- Add explicit, byte-budgeted FP4 stock prefill and stronger extension-cache
  diagnostics.
- Document the audited GB10 delegated-NVFP4 backend behavior.

The v2 decode kernel remains an explicit opt-in; inherited production dispatch
is unchanged. GB10 compute capability 12.1 is validated, while 12.0 remains
unexecuted.

Major contributions: Jason Wong (`@jsconsultancy`) — PRs #3, #4, and #6–#11.
Review, integration, and release validation: Robert Tand.

[Full diff](https://github.com/RobTand/gridbook/compare/v0.2.0...v0.3.0)
