# Roadmap

What is done, what is open, and what was tried and rejected. Nothing here is a
commitment to a date. This is the **canonical queue for kernel work and its
enablement dependencies**;
[`docs/KERNELS.md`](docs/KERNELS.md) explains the design,
[`docs/PLUGIN.md`](docs/PLUGIN.md) documents the shipped controls, and dated
audits preserve evidence rather than maintaining parallel TODO lists. Kernel
statuses below track the internal kernel/format standard; a path is **DEFAULT**
(on, no flag), **OPT-IN** (behind an environment switch), or **MEASURED
NEGATIVE** (built, measured, and kept off).

---

## Done

These are listed because older documents — and older versions of this file —
still describe some of them as future work.

- **Dense decode.** CUDA decode GEMV for FP8-CB and FP4-CB (two-tier v2),
  DEFAULT. Measured at/above native-format parity on the 27B (10.3 vs 10.26
  tok/s).
- **MoE decode.** Grouped `(token, expert)` GEMV plus a deterministic combine,
  DEFAULT. Took 35B-MoE decode from 3.52 → ~33 tok/s (faster than BF16's 28.4).
- **MoE prefill.** Native CUDA expansion plus a Gridbook-owned CUTLASS grouped
  GEMM is the quality-preserving production lane. Its predecessor CUDA
  chunk-expander path measured 293 → **1,821 tok/s** at 8k and 207 → **1,822
  tok/s** at 63k on Laguna-S-2.1 (commit `8829c16`); those historical numbers
  are not measurements of the new owned grouped bridge. The bridge's current
  generic SM80-compatible schedule is not Blackwell-optimized and measured
  **6–17% slower** than segmented BF16 matmuls on warm synthetic DSV4 shapes.
- **Native FP8 operator ownership.** FP8 activation quantization and CUTLASS
  scaled GEMM call vLLM's registered native CUDA operators directly after ABI
  and shape attestation; the fallback-capable `vllm._custom_ops` wrapper is not
  in Gridbook's serving path.
- **Native shared-CB ownership.** HunYuan-V3 shared-expert prefixes, including
  nested/collapsed MTP forms, resolve to native CB Linears. A CB tensor that can
  resolve only to a plain BF16 Linear now stops model load.
- **Mid-M fused prefill** (FP8-CB dispatch band 9 ≤ M ≤ 128): a CUTLASS decode-in-prologue GEMM with
  an fp32 epilogue, promoted to DEFAULT after measuring **1.40× in its niche**
  at the promotion gate (1.04–1.45× across M = 32/64/128 on GB10) with the
  quality gate preserved. `sm_120`-family only; other devices use native CUDA
  expansion + CUTLASS.
- **Quantized MTP draft head.** The 295B artifact ships an FP8-CB K44 draft
  block; speculative decode at k=1 measured 14.6 → **16.1 tok/s** on prose. (The
  remaining upside needs vLLM to capture drafter CUDA graphs — upstream work, see
  below.)
- **Every integer rung** across both ladders (NVFP4-CB K12–K24, FP8-CB K28–K48)
  with ceil-first uneven index splits, encoder-anchored and frozen.
- **Packaging.** The CUDA sources ship inside the Python package, so a
  non-editable `pip install` produces a working CUDA path. Previously any
  non-editable install silently degraded to the Triton fallback in retired
  releases.
- **FP8-CB hardware floor.** An FP8-CB artifact now fails early with a clear
  `sm_89+` requirement rather than loading on `sm_80` and failing at its first
  large-M prefill.

---

## Open

### Kernel TODO (canonical)

Priority means dependency order: **P0** blocks a fused-NVFP4 promotion decision,
**P1** is the next native-parity work, and **P2** is useful but not on the
critical path. A checked item requires merged code, regression tests, and the
applicable evidence gate; a fast standalone kernel is not sufficient served
evidence.

The implementation rule is one payload, one activation quantizer, one weight
decoder, and one GEMM/grouped-GEMM per execution contract. New policies and
backends must compose those pieces; they must not introduce a second packer,
resident weight copy, decoder, or matmul merely to create another route.

#### P0 — decide fused NVFP4 safely

- [ ] **K0.1 — Align the producer and consumer releases.** In PrismaQuant, bump
  the immutable Gridbook runtime pin from 0.4.1 (`59cebf9`) to the final
  Gridbook 0.5.0 release commit, rerun cross-repository contract/provenance CI,
  and publish PrismaQuant 0.5.2. Do not copy the Gridbook runtime back into the
  producer.
- [ ] **K0.2 — Produce a valid routed-MoE validation artifact.** Re-export a
  manageable representative model with producer-attested, stage-specific
  `input_global_scale` values for both `w13` and `w2`. The existing partial LFM
  artifact has no such payload, so all fused attempts correctly fail closed;
  inventing a runtime scale is not an acceptable workaround.
  *Status: the attestation plumbing has landed on both sides and the remaining
  gate is the re-export run itself.* PrismaQuant's execution-contract record now
  carries a per-FusedMoE-module stage section
  (`prismaquant.nvfp4_w4a4_activation_stages.v1`, record schema bumped to
  `prismaquant.nvfp4_w4a4_activation.v2`) naming each stage's physical target,
  policy, calibration source (experts-module input vs routed-intermediate
  replay), and per-stage value digest; all three exporters build it from one
  shared builder and fail closed on a half-calibrated module. Gridbook's
  validation harness verifies that section against the serialized scalars
  before any engine loads and emits a machine-readable K0.2 verdict
  (`attested_and_verified` / `missing_stages` / `digest_mismatch` /
  `not_attested`) that both A/B entry points consume as a precondition: a
  routed-MoE A/B against an unattested artifact is now reported as
  `fallback_telemetry_not_evidence` instead of proceeding silently. What remains
  is the GPU work: pick a manageable representative routed-MoE model, run the
  activation probe plus routed-intermediate replay, re-export through the CB
  exporter, and confirm the harness returns `attested_and_verified`. Only then
  do K0.5/K0.6 have a lawful MoE artifact to measure.
- [x] **K0.3 — Finish shared fused-JIT attestation and fail-fast loading.
  IMPLEMENTED (2026-08-02).** The fused-FP4 source/header/ABI identity and
  strict two-module preload validation shipped in 0.4.2. That facility is now
  extracted (`cuda_ext._fused_build_identity`) and keys every header-bearing
  native module — fused FP8 (`fused/<digest>`), the grouped-BF16 bridge
  (`bf16_grouped/<digest>`), the fused FP4-v2 mid-M lane, and the persistent-B
  MoE lane — over packaged sources/headers (including the shared
  `cb_grouped_common.hpp`), target architecture and lane macros, Python/Torch/
  CUDA/compiler ABI, and the external CUTLASS sentinels. Non-`sm_120`/`sm_121`
  targets are rejected before either fused build starts (the FP4 loader gained
  the same precheck the FP8 loader had), and required validation now fails when
  the requested call route — not merely its module — does not execute: the FP8
  contract requires the grouped bindings, and each opt-in lane attests every
  symbol its forward path dereferences at model load, with negative-control
  tests. Evidence: `tests/test_ext_build_identity.py`,
  `tests/test_ext_symbols.py`, the per-lane attestation tests.
- [x] **K0.4 — Finish grouped-MoE routing and telemetry. IMPLEMENTED
  (2026-08-02).** `moe_routing.cb_grouped_tile_m` replaces the manual choice —
  which was worse than "manual": the FP8 grouped path resolved `tile_m=None` to
  the kernel's compiled default, so serving could never reach TileM=256 at all,
  and the FP4 path read its tile off the *suffix* of an activation-policy env
  string. The selector accounts for routed token counts (`P = tokens × top_k`,
  exactly host-known), per-expert padding waste at each tile (through the exact
  `pad₂₅₆ − pad₁₂₈` lemma), both projection stages' shapes (their decode:MMA
  ratio is `1:t` independent of N and K, so one condition serves both, and the
  narrower N bounds the occupancy), and occupancy (the dense selector's
  `ceil(2·SM/3)` floor). It is CUDA-graph-safe **by construction**: every input
  is a host-known integer, so there is no device read to sync on — which matters
  because `tile_m` fixes both the kernel symbol and every routing tensor's
  shape. Fixing this surfaced and repaired a real capture defect: the padded
  routing called `torch.bincount`, which host-syncs, while documenting "NO HOST
  READS". **K0.4's telemetry list is satisfied in full** — requested activation
  policy, actual kernel symbol, TileM, problem shape, activation contract,
  fallback state and the exact fallback reason are emitted for dense *and* MoE
  calls through `nvfp4_activation_contract.emit_route`, which extends the
  existing 0.4.2 dense mechanism rather than adding a parallel one, plus
  selector provenance so a tile choice is auditable offline. See
  [KERNELS](docs/KERNELS.md#grouped-moe-tilem-selection-k04). The calibrated
  `ρ > 512` threshold remains proposal data for the grouped lanes until a routed
  sweep pins it.
- [ ] **K0.5 — Profile and close the fused-NVFP4 raw operator gap.** Split
  activation quantization, packed-B decode, synchronization, MMA, epilogue,
  and launch costs and compare against the matching stock
  `sm120_nvf4_mm_scaled` execution contract. Optimize from that profile while
  retaining the shared quantizer, packed payload, decoder collective, and
  concrete GEMM runners.
- [ ] **K0.6 — Run the promotion gate and make an explicit decision.** In one
  pinned serving session, compare fused and current paths on a dense model of
  at least 4B plus a representative routed MoE. Cover full-vocabulary teacher
  KL/PPL/tasks, prompt-length distribution, concurrency, chunked prefill,
  plain and shipped batched/speculative decode, and routed-token histograms.
  Keep the flags default-off unless every
  [fused-NVFP4 reconsideration gate](docs/audits/fused_nvfp4_enablement_2026-07-31.md#reconsideration-gates)
  passes, including the p95 TTFT win, per-cell regression limit, zero
  unexplained fallback, and supported-runtime revalidation.

#### P1 — close the remaining native-parity gaps

- [ ] **K1.1 — Build large-M grouped MoE decode-in-mainloop.** Decode an expert
  weight tile once, stream its routed/padded M rows through it, and time the
  whole routed operator. It must preserve the selected activation payload,
  avoid an expanded `[E,N,K]` HBM tile, handle empty/uneven routing, and remain
  stream- and graph-safe. This is a new MoE schedule, not a revival of the
  measured-negative dense persistent-N kernel.
  **Kernel IMPLEMENTED behind `PRISMAQUANT_CB_MOE_PERSISTENT_B=1`**
  (`csrc/cb_moe_persistent_b.cu`, FP4-CB v2, cc 12.0/12.1): a CTA owns one
  (expert, N-tile), decodes that tile from packed CB bytes into shared memory
  once and streams the expert's exact routed segment through it, with the
  M-loop inside the kernel. No `[E,N,K]` transient, no padded rows, no host
  read; launch geometry is a function of `(E, N)` alone. Decode bit-identical
  to `cb_expand_v2` by test; only the FP32 reduction order changes. The
  whole-routed-operator microbenchmark is proposal data only —
  **what remains open on this item is the served
  [NATIVE-PARITY](docs/NATIVE-PARITY.md) gate, not the kernel.** See
  [KERNELS](docs/KERNELS.md#persistent-b-decode-in-mainloop-opt-in-prismaquant_cb_moe_persistent_b).
- [x] **K1.2 — Cover the complete FP8-CB mid-M production rung surface.
  RESOLVED (2026-08-02) — second arm.** Production does permit every K28–K48,
  but the first arm ("instantiate and test every product rung") is **closed by a
  format + TMA law**, not by effort: `type_size = 4k` is the packed-B TMA box's
  contiguous extent and must be a 16-byte multiple (`k % 4 == 0`), and the fused
  mainloop's single `CbSubW = k/4` sub-table width is the format's real layout
  only on those same rungs — the format splits `k` over `n_sub = 4` raggedly
  (`csrc/cb_gemv.cu` `SubSplit`), so a uniform decode at k37 would be *wrong*,
  not merely unaligned. The six compiled rungs were therefore already the
  maximum this collective admits, and the published 27B artifact's 8-rung
  K36–K47 ladder hits exactly the three multiples of 4 it contains.
  **The second arm is implemented:** the compiled set is queryable
  (`cb_fused_kbits()`), Python gates on the derived law and confirms against the
  module instead of carrying duplicated literals, every kernel switch is
  generated from one rung list, the smem feasibility predicate is a closed form
  `static_assert`ed against the probe's measured cells, the published smem table
  was regenerated (it quoted the stale pre-R6 base), an off-law rung is refused
  with a message naming the law and the routes that *do* serve it, and per-rung
  bit-exact gates are parametrized from the module's own reported surface. See
  [KERNELS](docs/KERNELS.md#rung-coverage-what-this-lane-can-and-cannot-serve-k12).
  Serving all 21 rungs through this lane would need a new packed-B TMA schedule
  *and* a ragged-width decode — a new kernel, tracked separately if ever wanted.
- [ ] **K1.3 — Reassess large-M dense FP8-CB from a fresh roofline.** The
  transient path remains about 1.44x native, but the existing persistent-N
  implementation was 2–5.7x slower. Profile current traffic and synchronization
  first; only build a replacement schedule if the model shows a realizable win.
  Do not continue or enable the quarantined implementation as unfinished work.
- [ ] **K1.4 — Complete graph and alternate-schedule qualification.** Run the
  exact-byte 27B streaming gate for `FULL_DECODE_ONLY` CUDA graphs, and qualify
  `cb_gemv_v2` on `sm_120` with same-session quality, telemetry, long prefill,
  concurrency, and soak coverage before considering a default change.

#### P2 — completeness and wider qualification

- [ ] **K2.1 — Resolve FP4-v1 MoE explicitly.** Either implement its transient
  and grouped path with the existing v1 decoder, or reject v1 expert artifacts
  at load with a precise support error. Production FP4-CB v2 remains the
  priority.
- [ ] **K2.2 — Revisit k24 long-K `cb_gemv_v2` staging only with evidence.** A
  double-buffered row stage is optional; implement it only if profiling predicts
  and an interleaved benchmark confirms a win over the safe compiled fallback.
- [ ] **K2.3 — Automate hardware qualification.** Add self-hosted CUDA compile,
  SASS, wheel-install, custom-op, non-default-stream, graph, and fail-if-skipped
  tests. Qualify both `sm_120` and `sm_121`, then extend only the paths that are
  legal on Ada/Hopper.

#### Blocked or deferred

- [ ] **KB.1 — Add approved W4A16 support when AMD validation hardware is
  available.** Gridbook should own one exact pack/schema/metadata/profile/loader
  and delegation path, initially reusing vLLM's upstream
  `RDNAHybridW4A16` execution backend. Do not author a duplicate W4A16 kernel
  or packer. The prior Strix Halo results are arithmetic bring-up evidence, not
  served validation, and no gfx1151/HIP kernel work is active while hardware
  access is unavailable.

#### Before DSV4 Flash (integration, not new kernels)

- [x] **D0.1 — Establish the exact serving contract.** Done for the body, MTP
  and DSpark, against vLLM **0.24.0** and the released
  `deepseek-ai/DeepSeek-V4-Flash-0731` config. `deepseek_v4` is now a
  registered producer profile and
  `vllm.models.deepseek_v4.nvidia.model` a registered top-level loader module.
  What the inspection actually established, each of which changed the wiring:
  - **The architecture is not where the contract could name it.** vLLM 0.24
    ships DSV4 as a per-platform *package*; `DeepseekV4ForCausalLM` is DEFINED
    in `vllm/models/deepseek_v4/nvidia/model.py`, and the package `__init__`
    only re-exports it. `plugin.py` installs on the defining module, so the
    contract validator now accepts a second root, `vllm.models.`, alongside
    `vllm.model_executor.models.` — still an explicit two-entry allow-list,
    because every entry is a dynamic import into the serving process.
  - **Module attributes are `attn`/`ffn`, not `self_attn`/`mlp`**, and the
    routed stack nests again: FusedMoE prefix `model.layers.N.ffn.experts`,
    parameters at `…ffn.experts.routed_experts.w13_*`. The shipped
    stem-plus-leaf `.experts.` anchor absorbs both, so **no new per-model
    loader module was needed** — the generic wrap covers DSV4.
  - **The checkpoint carries no `model.` component** (keys start at
    `layers.N.`); the class re-attaches it in its own `hf_to_vllm_mapper`
    *after* serving prefixes are handed out. The loader already applies the
    model's own mapper; `_canonical_prefix` gained the matching source-namespace
    vintage so config-side target resolution crosses the same gap.
  - **The class defines no `packed_modules_mapping`**, yet merges
    `attn.wq_a`+`attn.wkv` into `fused_wqa_wkv` and the shared expert's
    `w1`+`w3` into `gate_up_proj` (published only via
    `stacked_params_mapping`). Gridbook's fused fallback tables now carry both,
    so a CB shard resolves instead of silently falling through to BF16.
  - **All 43 layers are MoE** — there is no `first_k_dense_replace` in the
    config or in vLLM's DSV4 — so every layer contributes an expert stack the
    fill guard checks.
  - **TP stays 1.** Every TP guard in the class is a divisibility check that
    `tp_size == 1` satisfies (`64 % 1`, `256 % 1`, `8 // 1`); MLA never shards
    KV (`fused_wqa_wkv` is built `disable_tp=True`). No narrowly scoped TP
    implementation is required, so none was added and the existing rejection
    above one stands.
  - **MTP/DSpark are passthrough, not served.** `DeepseekV4ForCausalLM.
    load_weights` builds `AutoWeightsLoader(self, skip_substrs=["mtp."])`, so
    all 4,705 `mtp.*` tensors are dropped before any parameter lookup; the
    drafter `DeepSeekV4MTPModel` is reachable only through
    `--speculative-config`; and `dspark` appears **nowhere** in the vLLM
    package, so none of the four `dspark_*` config keys is read. The contract
    therefore preserves them as producer passthrough and claims none of them —
    it invents no support vLLM lacks.
  - **Not every DSV4 Linear is CB-eligible.** `ffn.gate`, both
    `compressor.fused_wkv_wgate`s, `indexer.weights_proj`, `lm_head` and
    `embed_tokens` are built with no quant config, and `attn.wo_a` is created
    and post-processed through the quant contract but **applied outside it**
    (`nvidia/ops/o_proj.py` reads `.weight`/`.weight_scale_inv` directly), so a
    CB layout there would serve silently wrong results. All of these are
    documented as must-not-quantize in [`docs/PLUGIN.md`](docs/PLUGIN.md).

  Remaining before a real artifact loads is not contract work — see the
  DSV4-Flash study's release gate (items 4-5: production calibration of the
  eight K14 expert groups, then export/load/generate/benchmark).
- [ ] **D0.2 — Complete packed-expert native delegation if the assignment needs
  it.** Reuse Gridbook's existing top-level expert-loader path, canonical
  producer packers/metadata, and a version-attested upstream vLLM backend for
  rank-3 stock NVFP4/FP8 experts. Do not create another packer, loader, or
  native kernel. Fail closed if a selected backend drops activation scales and
  changes a declared W4A4 unit into W4A16. **The fail-closed clause is
  SHIPPED and generalized** (`gridbook/delegated_preflight.py`, called from the
  single delegation choke point in `config.py`): at model load a delegated
  group whose resolved backend is Triton-backed, is documented to discard
  declared activation scales, or is simply unaudited for an NVFP4 W4A4
  declaration, raises with the backend class, the group, and the contract it
  would violate. No environment variable bypasses it. The rest of D0.2 — the
  packed-expert loader work itself — remains open and is only needed if the
  assignment calls for it.
- [ ] **D0.3 — Close the exact-rate evidence gap.** Rerun exact-byte
  0.6B/4B/27B endpoints and optimized menus over the representative workload
  matrix. At 4.5 bpp compare native NVFP4 with FP8-CB K36 using exact
  whole-artifact bytes; below 4.5 bpp evaluate byte-neutral assignments whose
  NVFP4 promotions are funded by lower CB rungs elsewhere. Record format/rung,
  layout, activation quantization, concrete backend, GPU/runtime identity, TP,
  and fallback state. This is an empirical release gate, not a reason to build
  another byte accountant or unconstrained allocator; follow
  [`docs/NATIVE-PARITY.md`](docs/NATIVE-PARITY.md).

Do not reopen a measured-negative schedule without new profiling evidence. In
particular, the dense persistent-N implementation, blanket `grouped_fused` MoE
default, w2 rowpack, decode-contract-v2 hoist, L2-pinned pipeline, and naive
inline CUDA-graph capture remain in [Measured and rejected](#measured-and-rejected).

### Conformance fixtures for independent implementers

The smallest downloadable artifact is 23 GB, and the spec ships no binary test
vectors. Anyone implementing a decoder from [`docs/SPEC.md`](docs/SPEC.md) has
nothing small to check against. Publishing a tiny CB artifact plus per-rung
decode vectors is the missing piece that makes "implementable by anyone" true in
practice rather than in principle.

### Distribution

- **PyPI — done.** Stable releases are published as `gridbook`; use
  `pip install gridbook` inside the environment that already owns the serving
  torch/vLLM stack.
- **Tagged releases — done.** Versioned GitHub releases carry wheel and sdist
  artifacts. Release highlights and contributor attribution are maintained in
  [`CHANGELOG.md`](CHANGELOG.md).
- **CI.** GitHub Actions now build the sdist and wheel, assert both really
  contain `gridbook/csrc/*.cu`, install the wheel **non-editably** into a clean
  environment and re-resolve the sources from `site-packages`, check the
  `vllm.general_plugins` entry point, and run the GPU-free tests
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). That closes the hole
  the packaging defect came through — nothing had ever exercised a non-editable
  install. **What CI still cannot do is compile the kernels**: free runners have
  no `nvcc`, so the compile-only extension build is a manual pre-tag gate
  documented in [`docs/RELEASING.md`](docs/RELEASING.md). A GPU-less
  self-hosted runner with a CUDA toolkit would automate it.
- **Container image.** A [`Dockerfile`](Dockerfile) that layers gridbook onto a
  pinned vLLM image is in the repo and documented in
  [`docs/CONTAINER.md`](docs/CONTAINER.md); no image is published to a registry
  yet.

### Widening measured hardware coverage

Every published number is from one GB10 / DGX Spark (`sm_121`, arm64). The
decode kernel is architecture-generic by construction and *should* run from
`sm_80` up, but that is inferred, not measured — see the
[hardware matrix](docs/INSTALL.md#hardware-matrix). Two concrete code items would
make the wider claim safe:

- **Wider execution validation for the dense FP8-CB capability guard.** The
  native FP8 prefill lane requires `sm_89+` and Gridbook rejects FP8-CB early on
  A100 rather than selecting a different kernel family. H100/Ada execution is
  still inferred rather than measured.
- **An architecture precheck before the fused CUTLASS build**, so a non-Blackwell
  GPU skips a doomed multi-minute compile inside the user's first request and
  selects the qualified native expand + CUTLASS route directly.

### vLLM compatibility preflight

The plugin imports vLLM internals (fused-MoE classes, the quantization registry,
the registered `vllm._C` CUDA operator ABI) that carry no stability promise, and vLLM
logs-and-continues when a plugin fails to load — so drift surfaces as an
unrelated "invalid quantization method" at model load. A symbol canary that fails
with one actionable sentence naming the missing symbol and the tested vLLM
version is the intended fix.

### Speculative decode throughput

Draft acceptance is already high (68–93% measured, model-dependent), but vLLM
runs the drafter uncaptured for this method, costing per-draft-token host
overhead that scales with k — so k=1 is today's throughput optimum. This becomes
a straight multiplier once drafter CUDA-graph capture lands upstream; no work is
needed here.

### Documentation and spec corrections

[`docs/SPEC.md`](docs/SPEC.md) still states that the vLLM registry key must be
`"prismaquant"` and that the quantization config is embedded in `config.json`.
Both are now wrong in the shipped world: every published artifact carries
`"quant_method": "gridbook"` with a pointer stub in `config.json` and the real
configuration in `quant_config.json`. The spec needs to be corrected to describe
what ships — an implementation written from the current text cannot load a
published artifact.

### Not planned

- **A second artifact encoder.** PrismaQuant is the canonical producer;
  Gridbook owns the serving contract, decoder, and conformance fixtures. Copying
  the producer's search, packer, or exporter here would create two sources of
  truth.
- **General tensor parallel.** No `tp > 1` support exists today and broad TP
  support is not committed. D0.1 established that DSV4 does **not** need a
  narrowly scoped implementation — the model fits one GB10 at the planned
  92 GB, and every TP guard in vLLM's DSV4 class is satisfied at `tp_size == 1`
  — so no TP work is accepted on its account.
- **A vLLM fork or core patches.** Running on stock vLLM is the point.

---

## Measured and rejected

Kept here because a rejected experiment with a number attached is more useful
than silence.

| Item | Verdict |
|---|---|
| **Persistent-N large-M dense prefill** | Built, parity-green, and **2–5.7× slower** than expand-then-GEMM at 27B shapes: the CUDA expander had already shrunk the dense expand tax to ~10%, removing the opportunity that motivated it. The serving selector, custom op, loader, and switch are deleted; only the `.cu` and an explicitly opted-in direct research test remain. |
| **Legacy `grouped_fused` / per-layer `auto` MoE selection** | The predecessor fused path won on small-expert MoE (35B class) and lost on large-expert Laguna. Those selectors are removed from production; the native quality contract is grouped CUDA GEMV at M≤16 and exact expansion + owned CUTLASS grouped GEMM above it. |
| **w2 rowpack decode schedule** | Measured negative; stays behind an environment switch as a recorded result. |
| **Decode contract v2** (scale-epilogue hoist) | Measured **null** on the served 27B (10.10 vs 10.13 tok/s, quality-neutral) — decode is bandwidth-bound at per-byte parity, so there was nothing for the hoist to recover. Default stays v1; v2 remains available. |
| **L2-pinned per-expert scratch pipeline** | Wedged live serving three times, including the serial variant. Removed from production dispatch and its selector surface; the underlying L2-residency hypothesis remains a historical unmeasured idea. |
| **Signed "S-rung" formats** | Serving correctness proven bit-exact end to end, but in a matched-rate head-to-head over 776 per-(Linear, rung) comparisons the unsigned rungs won 79% of the time and the allocator placed 6 signed units against 147 unsigned. Closed as research-only; the spec keeps them for exotic weight geometries. |
| **Retired host-branch CUDA-graph capture of the decode path** | Historical measurement, *worse*: a prefill-sized trace baked the expand arm into decode. That branch and its switch are removed. The later opaque whole-dispatch op fixed the mechanism; mode-0 `FULL_DECODE_ONLY` measured 20.1% faster on the dated close-rate 0.6B canary, pending a fresh 27B streaming gate on the current operator stack. |
