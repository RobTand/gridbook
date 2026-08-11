# Changelog

## 0.8.3 — 2026-08-11

Per-role codebooks on routed MoE expert stacks, plus the fail-closed guard for
the hole they exposed. A uniform artifact — every artifact published to date —
resolves to the identical scheme object and decodes byte-for-byte as in 0.8.2.

- **Routed expert stacks may now carry one codebook per logical role**
  (`gate`, `up`, `down`) instead of one per layer. A learned codebook is fit
  per (layer, projection), so the three roles want three books at the same
  rung; before this, a learned routed artifact had to pool them. See
  `docs/SPEC.md` §5.1 for the artifact grammar. Scope: FP8-CB v1 product
  layout (`n_sub=4`), which is what the routed learned rungs use. An fp4 or
  v2 stack declaring per-role books is **refused at load**, not silently
  pooled — those lanes carry a second per-layer table and an in-kernel scale
  section, and shipping the split there untested would be worse than refusing.

- **SECURITY-ADJACENT FIX — the routed scheme resolver failed open.**
  `_moe_scheme_for_prefix` compared `(grid, mode, k, n_sub, type_size,
  activation_contract)` across a stack's targets — a tuple that does **not**
  include `codebook_ref` — and then returned the first target's scheme by
  sorted name, i.e. `…experts.down_proj`'s. An artifact naming different books
  per projection therefore **loaded without complaint and decoded every stack
  with down_proj's codebook**: silent numerical corruption, not a refusal. No
  Gridbook release could produce such an artifact (PrismaQuant's per-role
  emission self-gates on a `0.8.3` runtime), so no shipped checkpoint is
  affected — verified against a published routed-CB artifact, where 0 of 47
  routed layers disagree. The guard ships regardless of the feature: a role
  claimed twice with different books, or left without one, now refuses to load.

- **Per-role books and per-expert format groups are not composed.** A target
  such as `…experts.gate_proj.format_group_0` is refused by name. It would
  otherwise have been skipped in silence, because target matching keys on the
  final path component. No allocation produces the combination — routed layers
  take a single rung, since serving-unit promotion forces one format per
  routed stack — so this is an explicit non-feature, recorded rather than
  implemented.

- `cb_fused_moe_grouped` takes optional `out` / `n_offset`, letting a GEMM
  write its `N` columns into a column slice of a wider destination the caller
  owns. Splitting `N` preserves the tile count exactly
  (`ceil(Mp/TM) * ceil(N/TN)`), so a per-role gate/up stack costs one extra
  launch and no extra A traffic — versus a concat of both halves, which at
  DeepSeek-V4-Flash shapes would move ~0.6 GB per layer per forward. Existing
  callers pass neither argument and are unaffected.

- Per-role stacks materialise their `w13` halves and all weight scales as
  contiguous fp32 at load, releasing the fused stack. Residency is unchanged
  (the halves replace it), and the split path skips the per-forward
  `.to(float32).contiguous()` scale rebuild the fused path performs.

- **Both load-time gates are per-role aware, and that is tested on device.**
  Releasing the fused `w13` stack put two gates at risk of judging a tensor the
  lane no longer uses: `_cuda_moe_ok` materialises a LUT (the stock
  materialiser reads a `_cb_flat` a per-role layer does not have), and
  `_gf2_ok` validates `w13_cb_qweight`'s 3-D stride (which the split
  releases). Neither failure mode is loud — the second is a silent demotion to
  the BF16 quality bridge, ~663 ms/layer against ~29 ms on the fused lane at
  DeepSeek-V4-Flash shapes. Until a per-role artifact exists the standing
  evidence is an equivalence test: bind all three roles to the SAME book and
  the split must reproduce the uniform path bit for bit across decode, grouped
  prefill and the bridge expand, with both gates asserted to admit the split
  layer. `tests/test_routed_per_role_codebooks.py` §F.

## 0.8.2 — 2026-08-09

DeepSeek-V4-Flash serving enablement. Every entry below is additive or
fail-closed: an artifact that does not carry the new declarations keeps its
existing load path and identical bytes. The DSV4-Flash serving images were built
from `72d6650`; the two commits between it and this tag change prose only
(this changelog, module and test docstrings, and the CITATION date), so there is
no functional delta between what was served and what is tagged. The
served-parity gate named below is still open at tag time.

- DeepSeek-V4's grouped `attn.wo_a` source block-FP8 projection now runs
  through Gridbook's MXFP8 dense lane instead of vLLM's unconditional
  DeepGEMM `fp8_einsum` path. The qualified eugr Spark baseline, vLLM
  `0.26.1rc1.dev515+g653ebb52d.d20260808`, reads the weight and scale parameter
  directly for this projection, but Gridbook has already converted that scale
  tensor into its audited CuTe plane; an ABI-guarded adapter therefore performs
  vLLM's native inverse RoPE, groups heads in the original order, calls the
  owning Gridbook BMM method, and then calls `wo_b`. Stock DSV4 layers remain
  byte-for-byte on vLLM's original path. The grouped kernel is independently
  checked on GB10 at the artifact's exact `(G=8, N=1024, K=4096)` geometry:
  M=1 has max-abs error 0.03125 and relative Frobenius error 1.2460984e-5;
  M=64 has max-abs error 0.25 and relative Frobenius error 4.4897934e-5. Both
  satisfy the relative-Frobenius contract of at most 1e-4; this is not a
  bit-exactness claim. Full served parity is still a release gate.

- Semantically heterogeneous dense fusions now preserve per-Linear ownership.
  When vLLM merges checkpoint siblings such as gate/up or DeepSeek-V4's
  `wq_a`/`wkv`, Gridbook gives every role its existing native method and
  concatenates the results in vLLM shard order; it does not force one format,
  requantize a persistent common weight, or fall back to an upstream kernel.
  Matching format declarations also take this path when physical activation
  scalars differ per role; genuinely compatible roles retain vLLM's merged
  single-method path. The versioned, construction-scoped top-level loader
  stages every composite role plane and commits a fused module only after the
  complete transaction validates. An unwired model class fails at construction
  instead of making independently encoded roles appear invalid.
  Source MXFP4 and CB metadata are now explicitly rejected only when they
  claim the same routed-expert stack; separate routed, shared-expert, and
  attention modules may still use different formats within one decoder layer.

- DeepSeek-V4 source-passthrough lookup now bridges vLLM's `attn`/`ffn` and
  `w1`/`w3`/`w2` spellings to the producer declaration's
  `self_attn`/`mlp` and gate/up/down spellings. This covers both mixed fused
  projections and ordinary source-native body Linears/MoE stacks. The native
  block-FP8 merged path now declares its 128x128 scale block so vLLM places
  homogeneous source role scales at the correct offsets.

- The installed-wheel CPU release gate now locates sdist-only validation
  utilities explicitly instead of depending on GitHub's ambient
  `GITHUB_WORKSPACE`. The schema-v6 three-arm validation entry point is also
  included in the sdist and covered by the distribution-content gate.

- Source-format metadata groups are now kept out of compressed-tensors config
  parsing. Gridbook already owns these groups through the versioned
  `source_passthrough` declaration; handing their producer-only fields to
  compressed-tensors caused strict schema validation to reject mixed DeepSeek
  artifacts before model construction.

- Gridbook-native terminal lane methods now count as their own backend in the
  audited passthrough preflight. This lets the opt-in MXFP8 dense method satisfy
  the same fail-closed identity check its registry entry declares, while vLLM
  dispatching wrappers still require an inspected nested backend.

## 0.8.1 — 2026-08-05

- **Per-expert split-format MoE stacks (#40).** Gridbook now accepts the
  producer-owned `per_expert_format_groups` v1 declaration and serves one MoE
  layer whose `w13` and `w2` families are independently partitioned across CB
  formats and native MXFP4 passthrough groups. The loader validates complete,
  non-overlapping expert partitions and exact wire-id/scheme agreement, builds
  family-specific expert index maps, and refuses unknown or inconsistent
  declarations. Artifacts without the declaration retain the legacy uniform
  path and identical bytes.

- **Split-stack correctness follow-ups (#40, #41).** The primary mixed-vs-
  uniform oracle is now exact by construction: both sides use the same
  `dispatch_family_stages` and `torch.bmm` primitives in the same order, so its
  bit-equality claim no longer depends on BLAS reduction differences across
  torch versions; the byte-identity check also no longer needs undeclared
  NumPy. PR #41 adds a genuinely independent, explicit expert-loop witness
  which does not call the dispatcher, accepts at most 2 float64 ULP, and proves
  that both the exact assertion and tolerance bound reject deliberate
  perturbations.

## 0.8.0 — 2026-08-03

- **Release status note — read this before enabling anything below.** The MXFP8
  dense lane ships **opt-in behind `GRIDBOOK_MXFP8_DENSE=1` and refuses by
  default**, naming the flag in the refusal. Its correctness is audited and
  recorded; its *performance* is not. The NATIVE-PARITY served timing bench
  (`python -m gridbook.bench_mxfp8_dense`) had not been run at tag time — it
  refuses a non-idle GPU and the reference box was occupied — so this release
  makes no throughput claim for that lane whatsoever. The MXFP4 delegated route
  is a different case: it is **backed**, but only through vLLM's Marlin MoE
  backend, so it carries a *requirement* (`--moe-backend marlin`) rather than a
  refusal. Both statements are the honest state, not a hedge.

- **Delegated-native source-passthrough loading (#37).** A mixed Gridbook
  artifact may now ship some units as CB and others as verbatim copies of the
  source checkpoint's own quantized tensors; serving a passthrough unit means
  handing it to the native vLLM method that already understands that source
  format. `gridbook/source_passthrough.py` owns a versioned declaration schema
  (`source_passthrough`, version 1) and the audited format registry.
  **Absence of the key keeps the exact legacy all-CB meaning**, so every
  published artifact is unaffected. Unknown schema version, unknown format id,
  unaudited device, and a unit claimed by both the CB and passthrough
  vocabularies are each a load-time refusal with **no env bypass**.

- **The passthrough registry stores measured outcomes, not selector
  predicates.** Each entry records a real layer built, its weights processed
  and a forward profiled on the device — because on the very first format the
  measurement and the predicate disagree. `mxfp4_e2m1_ue8m0_g32` (routed
  experts) is native-confirmed via **Marlin** (`moe_wna16_marlin_gemm` /
  `marlin_moe_wna16::Marlin<…>`, alongside `vllm::act_and_mul_kernel`,
  `moe_align_block_size` and `topkGating`, with **zero Triton kernels** in the
  profile and finite output). The *default* rung is not that one: vLLM's DSV4
  MXFP4 oracle ranks `DEEPGEMM_MXFP4` ahead of Marlin and gates it on the whole
  sm12x family (`is_device_capability_family(120)`), then dies in
  `process_weights_after_loading` with DeepGEMM's `Unknown SF transformation`
  (`layout.hpp:59`). A predicate would have called that a pass. Serving MXFP4
  experts therefore **requires `--moe-backend marlin`**; `auto` is refused at
  construction with an actionable message rather than failing later.

- **`require_native_passthrough_backend` preflight.** A sibling of the
  compressed-tensors policy keyed on the *format* rather than on a config
  group: **Triton by MRO is refused unconditionally**, a rung measured to break
  is named with its symptom *and* its fix, and anything outside the audited set
  is UNKNOWN and fails. `config.py` routes declared units to the native method
  between the CB check and the ignore test, behind device attestation and that
  preflight — both at model load, the preflight the moment the constructor
  returns, since the MXFP4 method runs its oracle in `__init__` and that is the
  earliest point the backend is a fact rather than a prediction.

- **MXFP8 dense W8A8 lane (#38), and the route verdict it flips.** The DeepSeek
  body convention (E4M3 + UE8M0 per 128×128 tile) embeds *exactly* into MXFP8
  (E4M3 + UE8M0 per 32): `128 = 4 * 32`, so `SF_mx[n,c] = S_ds[n//128, c//4]`
  is a pure scale replication — no element byte changes, no scale arithmetic.
  The kernel is the **stock sm120 `OpClassBlockScaledTensorOp` CollectiveBuilder
  product** (`kind::mxf8f6f4`, SFVec 32) — no fork header, nothing vendored — and
  SF planes are filled by scattering with offsets computed from the mainloop's
  own CuTe layout, so the swizzle has exactly one spelling. On the strength of
  that, `fp8_e4m3_ue8m0_block128` moves from **BLOCKED** in #37 (empty audited
  set) to the **Gridbook-owned** route. The five measured-broken vLLM 0.24 rungs
  are *retained* in `known_broken_backends` with their symptoms rather than
  deleted (`DeepGemm…`, `Cutlass…`, `Triton…`, `FlashInfer…`,
  `MarlinFP8ScaledMMLinearKernel`). New wire id `mxfp8_e4m3_e8m0_g32` serves the
  same lane without the broadcast. Loader is the eighth digest-keyed JIT family.

- **MXFP8 correctness evidence, stated as a kernel claim.** Audited
  kernel-vs-fp32-oracle over the seven distinct DSV4-Flash body shapes from the
  **real checkpoint**: worst rel-Frobenius **5.9e-5** at M in {1, 64, 512} (M=1
  is covered by the same numeric contract, not a bit-exact contract); embedding
  chain worst 1.2e-4. The numbers live in a verdict-pin test, so they are
  asserted facts rather than prose. This is a correctness result and **not** a
  performance result — see the status note.

- **Activation UE8M0 exponent uses the exact frexp form.**
  `ceil(log2(amax/448))` in float32 is wrong about once in 4e5 group maxima:
  when `amax/448` rounds to exactly a power of two, `log2` lands on the integer,
  `ceil` keeps it, the exponent comes out one too small, and the group's max
  element **saturates at 448** — precisely what the ceil rule exists to prevent
  (measured: 15 saturations per 6e6 maxima; 0 for frexp). Replaced with the
  integer form the producer's encoder uses — `amax = f * 2^E`,
  `448 = 0.875 * 2^9`, exponent `E-9 + (f > 0.875)` — zeros pinned to byte 0, so
  both sides of the wire emit **identical scale bytes** for identical tensors.
  Pinned by a byte-equality cross-encoder test, a boundary regression over exact
  and near-miss power-of-two maxima, and a minimality assertion so the fix
  cannot overshoot. The recorded parity evidence above is unaffected: every
  comparison fed the same quantized operands to kernel and oracle, so the defect
  cancelled identically on both sides. Weight paths were never affected.

- **`cuda_ext.py` additive-block ordering restored.** The MXFP8 loader block
  initially landed *below* the persistent-B grouped-MoE block, breaking that
  block's file-tail invariant — the rule that keeps each lane revertible as one
  unit and stops concurrent branches appending loaders from conflicting. Pure
  code motion moved MXFP8 above it; the invariant test was not touched.

- **`lane_select._device_capability` becomes public `device_capability()`.**
  The lanes keep the old spelling as an alias.

- CI: bump `pypa/gh-action-pypi-publish` 1.14.1 → 1.14.2 in the actions group
  (#30). Action pins stay exact patch tags, not floating major refs.

## 0.7.0 — 2026-08-02

- Release status note: the `deepseek_v4` contract below is validated by
  fixture and instantiated-class tests; no released DSV4 artifact had
  loaded through it at tag time (the first CB artifact was still being
  produced). If the first real load surfaces contract gaps, a follow-up
  release will carry them.
- **D0.1 — register `deepseek_v4` and pin the DeepSeek-V4 serving contract.**
  Established against vLLM **0.24.0** and the released
  `deepseek-ai/DeepSeek-V4-Flash-0731` config (43 layers, all MoE, 256 routed +
  1 shared expert, top-6, MLA), by reading the class and instantiating it, not
  from the family resemblance to DeepSeek-V2/V3 — which turned out to mislead on
  every point that mattered. Four findings changed the wiring. (1) The class is
  not under `vllm/model_executor/models/` at all: vLLM 0.24 ships DSV4 as a
  per-platform package and `DeepseekV4ForCausalLM` is *defined* in
  `vllm/models/deepseek_v4/nvidia/model.py`, while the package `__init__` only
  re-exports it. `plugin.py` matches on the defining module, so the contract
  validator now takes a second root — `vllm.models.` — beside
  `vllm.model_executor.models.`, kept as a two-entry allow-list because each
  entry is a dynamic import into the serving process. (2) Its module attributes
  are `attn`/`ffn`, not `self_attn`/`mlp`, and the routed stack nests one level
  deeper again (`…ffn.experts.routed_experts.w13_*` under a FusedMoE prefix of
  `…ffn.experts`); the existing stem-plus-leaf `.experts.` anchor already spans
  both, so **no new per-model loader module was written** — the generic
  top-level wrap covers DSV4 as-is. (3) The checkpoint carries no `model.`
  component (keys start at `layers.N.`) and the class re-attaches it inside its
  own `hf_to_vllm_mapper`, i.e. after serving prefixes have been handed out;
  the loader already applied the model's mapper, and `_canonical_prefix` gained
  the matching source-namespace vintage so config-side target resolution
  crosses the same gap. (4) The class publishes **no**
  `packed_modules_mapping`, yet merges `attn.wq_a`+`attn.wkv` into
  `fused_wqa_wkv` and the shared expert's Mixtral-convention `w1`+`w3` into
  `gate_up_proj` — the merge lives only in `stacked_params_mapping`. Both fused
  fallback tables now carry those spellings, and `shard_target_keys` learned
  that one fused leaf can have more than one shard spelling (`gate_up_proj` is
  `gate_proj`/`up_proj` on Llama-class models and `w1`/`w3` here), trying each
  in order and taking the first with hits whole. Without that, a declared CB
  shard resolved to nothing and the layer fell silently through to BF16 or
  stock dispatch — the exact failure class this repo fails closed on.
  TP stays rejected above one, and D0.1 establishes that DSV4 needs no
  narrowly scoped TP implementation: it fits one GB10 at the planned 92 GB and
  every TP guard in the class is satisfied at `tp_size == 1`.
- **MTP and DSpark are passthrough, and the contract says so rather than
  implying support.** The artifact preserves `mtp.*` — 4,705 tensors across the
  three DSpark stages — verbatim, and
  `DeepseekV4ForCausalLM.load_weights` builds
  `AutoWeightsLoader(self, skip_substrs=["mtp."])`, so at plain serving time
  every one of them is dropped before any parameter lookup. The drafter
  `DeepSeekV4MTPModel` is reachable only under `--speculative-config` and
  consumes the payload in its source format, so no CB stacks are written for it
  and no loader module is registered for it; if one ever appears,
  `cb_fill_guard` fails the load and names the path to add. `dspark` occurs
  nowhere in the vLLM package, so none of the four `dspark_*` config keys is
  read at serving time. Hyper-connections (`hc_*`), the hash-routing
  `tid2eid` tables, `compressor.ape`, `attn_sink` and the router gate ride as
  unquantized parameters vLLM never routes through `get_quant_method`.
  `attn.wo_a` is documented as must-not-quantize for a subtler reason: it is
  created and post-processed through the quant-method contract but its forward
  **bypasses `apply()`**, reading `.weight`/`.weight_scale_inv` directly, so a
  CB layout there would serve wrong results silently instead of failing.
  D0.2 needed nothing new — the all-CB assignment delegates nothing, and the
  shipped `delegated_preflight` already refuses an unaudited or
  activation-scale-dropping backend for any stock region that might appear.

- Retire `PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE` and the `torch.Tag.cudagraph_unsafe`
  tagging it controlled. Nothing observable changes: the knob has been a silent
  no-op since the M-branch hoist (`5b4e5e7`, 2026-07-21). The tag only ever
  partitions an **inductor** graph, and only under
  `use_inductor_graph_partition=True` combined with piecewise cudagraphs —
  which is off at every vLLM optimization level, and which FULL capture ignores
  outright. Since the hoist, dynamo sees only `cb_linear_forward` /
  `cb_moe_forward`; the 18 tagged kernel ops run inside those two ops' eager
  implementations and are never graph nodes, so their tags were metadata no
  compiler read. The end state is that **no Gridbook op carries the tag, ever**
  — moving it onto the two whole-dispatch ops would not preserve anything, it
  would make every CB layer an eager partition boundary, i.e. the 2026-07-21
  corruption configuration at worse granularity. Those two ops are capture-safe
  by construction: the M-branch resolves host-side and the host-syncing arms are
  unreachable at captured decode sizes. Reproducing the historical corruption
  now needs the hoist reverted *and* inductor graph partitioning enabled with
  piecewise cudagraphs on a torch predating pytorch#165815.

## 0.6.0 — 2026-08-02

- **K1.2 — the FP8-CB fused mid-M rung surface, and why it is already
  complete.** The lane instantiated `k ∈ {28,32,36,40,44,48}` while production
  permits every integer K28–K48, which read like five missing instantiations on
  the published 27B artifact's 8-rung K36–K47 ladder. It is not: the compiled
  set is exactly what a **format + TMA law** admits. `type_size = 4k` is the
  packed-B TMA box's contiguous extent and must be a 16-byte multiple
  (`k % 4 == 0`); and the fused mainloop decodes with a *single* width
  `CbSubW = k/4`, while the format splits `k` over `n_sub = 4` **raggedly**
  (`csrc/cb_gemv.cu` `SubSplit`), so at k37 the true widths are `(10,9,9,9)` and
  a uniform decode would be *wrong*, not merely unaligned. ROADMAP K1.2's second
  arm is therefore the live one, and is implemented: the compiled set is
  queryable (`cb_fused_kbits()`), `linear.py` and `moe.py` stop carrying
  duplicate literal ladders and gate on the derived law
  (`codec.FP8_FUSED_KBITS`) before confirming against the module, every switch
  and report in the kernel is generated from one rung list, the rung check and
  `moe_tile_supported` became laws (the smem predicate is a closed form
  `static_assert`ed cell-by-cell against the probe's twelve measured numbers),
  and an off-law rung is refused with a message naming the law and the routes
  that *do* serve it. Per-rung bit-exact gates are now parametrized from the
  module's own reported surface rather than a hand-written four-of-six list.
  The published smem table was **regenerated** — it quoted the pre-R6 base and
  was stale by up to 16,384 B. No kernel instantiation changed; the cold-cache
  JIT build moves 71.4 s -> 76.0/75.7 s (GB10 container, 20 instantiations),
  i.e. +4.6 s of compile-time *evaluation* — the law predicates and the
  twelve-cell `static_assert` table pinning the smem closed form to the probe —
  not of code generation.
- **K0.4 — graph-safe grouped TileM selector and full dispatch telemetry.**
  `moe_routing.cb_grouped_tile_m` replaces a choice that was worse than manual:
  the FP8 grouped path resolved `tile_m=None` to the kernel's compiled default,
  so serving could never reach TileM=256 at all, and the FP4 path read its tile
  off the *suffix* of an activation-policy env string. The selector is
  CUDA-graph-safe **by construction** — every input is a host-known integer
  (`topk_ids.shape`, layer constants, the build's own compiled tile list, the
  cached non-synchronizing SM count), so there is no device read to sync on,
  which matters because `tile_m` fixes both the kernel symbol and every routing
  tensor's shape. The rule is `ρ = P/E > 512` plus the dense selector's
  `ceil(2·SM/3)` occupancy floor, derived from the exact
  `pad₂₅₆ − pad₁₂₈` padding lemma and a decode:MMA ratio that is `1:t`
  independent of N and K (so both projection stages give one condition); the
  threshold is calibrated by inverting the dense TileM A/B and remains proposal
  data for the grouped lanes. Telemetry now emits K0.4's full list — requested
  activation policy, actual kernel symbol, TileM, problem shape, activation
  contract, fallback state and the **exact** fallback reason — for dense *and*
  MoE calls, extending the 0.4.2 dense mechanism (plain scalars on the layer,
  tensor-free, sync-free, last-write-wins) rather than adding a parallel one,
  with two-phase writes and selector provenance so a tile choice is auditable
  offline. The FP8 grouped gate recorded a bare bool before this; it now names
  the failing clause. Both tiles are gated **bit-identical** pre-combine in
  stable-argsorted pair order — the selector is a pure performance choice.
- Split the fused MoE quality envelope from the reassociation one (tests only).
  `_REL = 2.1e-2` had been fitted to four hand-picked routing cases; across 224
  configurations the fused-vs-bridge disagreement runs 1.566e-2–2.316e-2 and is
  **cross-representation** — the two lanes' per-token E4M3 activation
  quantizers put 0.82% of elements on different codes — not reassociation, so
  the four cross-representation gates move to `_REL_FUSED = 2.5e-2` while
  `_REL` keeps the same-representation comparisons. It was never a TileM=256
  defect: at the failing cell both compiled tiles are bit-identical. Two
  sharper gates replace what the tolerance stopped gating — both lanes measured
  against the exact FP32 computation (0.89–1.05×, gated at 1.25×), and a ragged
  arm on the both-tiles-bit-identical test.
- **Fixed a CUDA-graph capture defect in the padded grouped routing.**
  `cb_grouped_pad_routing` documented "NO HOST READS" while calling
  `torch.bincount`, which sizes its CUDA output from `.max().item()` and
  therefore host-syncs — the exact trap the persistent-B lane hit and gated with
  a negative control. Every padded grouped lane was uncapturable as a result.
  Counts now come from the `scatter_add_` form that lane already proved
  (identical integers, static shape, pure device work).
- **Fixed a packaging bug that broke JIT builds from an installed wheel.**
  `csrc/*.hpp` was matched by no `package-data` glob (`csrc/*.h` does not match
  `.hpp`, and `csrc/cutlass_fork/*.hpp` is a different directory), so
  `cb_grouped_common.hpp` shipped in **neither** the wheel nor the sdist. It is
  a declared build input of all four fused/grouped loaders, so an installed
  wheel could not build any of them (`_require_csrc` raises
  `IncompleteInstallError`). Added the glob and reconciled all three mirrors of
  the runtime-required floor (`check_dist.py`, `check_installed.py`,
  `test_release_metadata.py`), which had also fallen a wave behind on the
  fp4v2, persistent-B and shared-header sources.
- **`PRISMAQUANT_PRELOAD_FUSED=1` now warms every native extension family**
  (decode GEMV, GEMV-v2, grouped BF16, both fused FP8/NVFP4 modules, fused
  FP4-v2, persistent-B MoE) instead of only the two fused ones, so a
  residency-matched A/B cannot be confounded by a module one arm never loaded
  — the audit's §3 P4 "preload must be a full warm-up" item, and the practical
  half of the documented ±17% measurement-arithmetic caveat. Each family is
  attempted independently and fail-soft; strict mode still reports every failure
  after every attempt. `preload_fused_extensions` remains as a delegating alias
  because it is the published name.
- The sm12x grouped-BF16 lane's in-mainloop A-row gather and swizzle-group
  expert packing are now **wired through `moe.py`**, not just available: stage
  one of the routed bridge takes the gather entry point (its `[Mp, K]` padded
  activation copy no longer exists — `dest` *is* the row-source vector), and
  `_padded_route` applies `pack_expert_blocks` using the per-expert block counts
  from the block-offset host read already taken, gated on `chunk >= E` because
  narrower expert chunks index blocks by expert-major contiguity. Stage two
  stays in padded mode by construction: its input is the padded intermediate.

- Close the sm12x grouped-BF16 lane's ragged-padding tax with two
  construction changes to the SAME compiled collective (schedule, tile,
  stages and smem untouched; audit §3 P1 follow-through, and the
  supersession of "Add an **sm12x-native CUTLASS 3.x lane**" below —
  its `T=512` deficit and its "next step" both resolve here):
  **an in-mainloop A-row gather** — the mainloop fork gains a fourth marked
  addition: the producer warp reads each padded row through ``row_src[m]``
  from the COMPACT activation with predicated zero-filling 16-byte
  ``cp.async`` (upstream's own `sm120_mma_tma_blockwise_scaling.hpp`
  producer idiom: 33 producer events, B-only transaction bytes), so the
  row-padded activation copy — an HBM write plus a padded re-read too large
  for L2 — no longer exists (`cb_bf16_grouped_mm_sm120_gather[_out]`; the
  padded-copy entry points are unchanged and the two modes are gated
  BIT-IDENTICAL, `torch.equal`, because they load byte-identical smem
  tiles); and **a swizzle-group-aligned expert order**
  (`bf16_grouped_lane.pack_expert_blocks`) — the persistent scheduler sweeps
  N in groups of 8 M-tiles, so an expert straddling a group boundary has its
  whole B slice fetched from DRAM once per group touched; packing experts
  into groups (deterministic first-fit-decreasing on the routing histogram,
  telemetered as groups-touched vs minimum) removes the straddle excess,
  measured 14–17% GEMM time at ``T=512`` and neutral at ``T=128``. The dense
  ``E=1`` helper now uses the gather mode too (``row_src = arange``; no
  padded copy, no concat). Measured on the GB10 (whole operator, warm
  medians, seed-731 DSV4/Laguna cells): **1.03–1.05× segmented BF16
  matmuls at ``T=128`` and 1.10–1.15× at ``T=512``** — the padded-copy
  construction's measured 0.83–0.92× ``T=512`` deficit is closed — while
  beating the SM80 bridge by 1.13–1.37× everywhere; a ``TileM`` ladder was
  evaluated against the sweep record instead and is measured-dead for these
  cells (every 128-row tile ≤ 0.97× segmented). Bit gates extended to
  every new boundary (gather==padded bitwise on every routing shape incl.
  K-residue, OOB ids read zeros, packed order is a pure block permutation);
  flag semantics, the SM80 lane, and flag-off dispatch are byte-for-byte
  unchanged. Proposal data; the served NATIVE-PARITY protocol has not run.
- Add the **FP4-CB v2 fused mid-M lane** (`csrc/cb_fused_fp4v2_gemm.cu` +
  `csrc/cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp`, audit §3 P2a), closing the
  audit's structural cause (c): FP8-CB owned M = 9–128 with a fused
  decode-in-prologue kernel while FP4 had **no mid-M lane at all**. The new
  lane decodes packed CB rows to BF16 inside the CUTLASS producer/consumer
  stage, so the `[N,K]` BF16 transient never reaches HBM. It is
  **contract-preserving**: the decoded values are proven bit-identical to
  `cb_expand_v2` — the whole decoded tile, at all 13 K12–K24 rungs, via a
  one-hot read-out that turns the GEMM into a direct dump of the decoded
  weights (no tolerance anywhere) — and the activations are the same BF16
  group-16 QDQ output the bridge already consumes, so only the FP32 GEMM
  reduction order moves. Separate module from the native-NVFP4 fused lane,
  whose payload and served numerics are different; nothing there is touched.
  fp4-v2's odd `type_size = 4k+9` makes TMA structurally unusable for B, so the
  producer publishes a per-stage descriptor under the mbarrier and the
  consumers gather packed bytes from gmem with aligned-u32 windows — which also
  makes `k_bits` a runtime parameter, so four compiled kernels serve the whole
  rung ladder (they differ only in the codebook smem-stage size). Shipped tile
  `128×64×64`/2 stages, measured against the 101,376-byte budget by the new
  host-only `csrc/tools/smem_probe_fp4v2_bf16.cu`; the zero-margin 48 KiB stage
  is deliberately not compiled. **Opt-in** behind
  `PRISMAQUANT_CB_FP4_FUSED_MIDM=1`, resolved at model load and failing closed;
  with the flag unset the dispatch is byte-for-byte what it was. `M ≤ 128` is a
  HARD gate the kernel enforces itself. Measured 1.06–4.37× the shipping expand
  + bridge route at M ∈ {9,16,32,64,128} on 27B/DSV4-class shapes, every cell
  bit-equal to a same-config passthrough oracle — a wider band than the fp8
  twin's 1.04–1.45× because the fp4 quality expand writes BF16, 4× the fp8
  expand's transient bytes. Proposal data only, served NATIVE-PARITY protocol
  not run, and an unexplained M ≤ 12 latency cliff is recorded as open;
  `scripts/bench_fp4v2_fused_midm.py` reproduces the table.
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
  (**Superseded within this release** by "Close the sm12x grouped-BF16 lane's
  ragged-padding tax" above: the gather was built and the TileM ladder was
  measured-dead, and the `T=512` numbers here are the pre-gather ones.)
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
- Make the no-Triton ratchet independent of the directory the suite is staged
  in. `release.yml`'s `verify installed wheel` job copies `tests` to
  `$RUNNER_TEMP/gbtests` — so the checkout cannot shadow the installed wheel —
  and failed there on the ratchet itself (`gbtests/test_no_triton_runtime.py:89
  [mention] executable definition _is_triton_module`, and 23 more across it and
  `test_delegated_preflight.py`) on a tree CI had just passed. The `mention`
  exemptions were keyed on a literal `tests/` prefix, so under any other
  staging name the two files that name Triton *because* they are the
  anti-Triton machinery stopped being exempt. Those keys are now anchors —
  resolved inside the scanned package and inside the ratchet's own directory,
  matched by resolved path rather than by a rendered string. The same three
  files are exempt, from the `mention` rules only, and the meta-test that an
  exemption can never hide a reaching import now runs on all three under
  staging instead of silently skipping two. Scan-root discovery treats a staged
  tree with no sibling `scripts/` as a smaller scan rather than a failed one,
  and a new test reproduces the release job's layout exactly: copy the suite
  into a `gbtests/`, load the copy, make it scan itself.
- Stop the CPU suite needing NumPy. `test_validate_fused_nvfp4_ab.py`'s K0.2
  fixture writer built its artifact with `safetensors.torch.save_file`, whose
  **write** path imports NumPy — which Gridbook deliberately does not depend on
  (`gridbook/cb_digest.py`), so it is absent from the one environment that
  matters here: the wheel's own closure, which is what `cpu-tests` and
  `release.yml`'s `verify` install the suite into from outside the checkout.
  The fixture therefore passed on every developer host and failed all four CI
  legs on master — `ModuleNotFoundError: No module named 'numpy'` from
  `safetensors/torch.py`, 4 failed / 49 passed. It now serializes its two F32
  scalars directly, exactly as `test_codebook_digest.py` has always done for
  its F16 sidecar and for the same written reason, and a scan asserts no test
  or script reaches for `save_file` again. Nothing in the wheel changes: the
  read path, which is all Gridbook ever uses, was never NumPy-bound.
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
