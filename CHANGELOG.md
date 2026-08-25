# Changelog

## 0.9.1 — 2026-08-24

### Pre-release broad-ladder retraction (intended 0.9.1 surface)

- Before the planned 0.9.1 release, Gridbook retracted the candidate public
  ladder expansions below because no released artifact or reader-compatibility
  need justified them. Contract schema v11 remains the authority without
  another schema bump: NVFP4-CB reads and produces exactly K12..K24; FP8-CB
  reads every integer K28..K48 and produces exactly K40/K44/K48.
- Lane-eligibility cells describe producer-facing routes and are narrowed to
  those producer sets. Historical FP8 fused readers
  K28/K32/K36/K40/K44/K48 remain compiled, including K28/K32/K36 reader-only
  compatibility paths. Generic NVFP4 K1..K32 and aligned low FP8 direct calls
  remain research primitives only; they cannot authorize a sidecar, chooser,
  producer, or lane claim.
- The two pre-release candidate sections immediately below are retained as
  development history, including their dated compile receipts and measurement
  evidence, but are superseded by this retraction and do not describe the
  intended public surface.

### Pre-release candidate (retracted above): NVFP4-CB K1..K25 expansion

- NVFP4-CB readers, producers, artifact choosers, and SM120 lane rows accept
  every integer rung K1..K25. There are no legacy K26..K32 artifacts to
  preserve, so those values are outside the public format contract. K1's
  ceil-first product split is the valid `(1, 0)` edge.
- The inherited dense and routed decode kernels are explicitly gated for the
  supported range. The exact BF16 transient expander stages the whole
  dictionary through K25 (98,304 bytes). Its LUT copy now handles K1's valid
  24-byte dictionary without an artificial 16-byte size rule. Direct CUDA
  bindings also admit K26..K32 runtime calls with global cached gathers for kernel
  research only; that scaffolding is not reader or artifact support.
- Optimized lanes remain independently qualified. A rung outside an optimized
  lane's compiled/eligible subset takes the exact generic decode or
  expand-plus-BF16 bridge; wire acceptance never implies fast-lane evidence.
- Overlapping FP8/NVFP4 rates carry no manual family priority. AQUA selection
  remains driven by registered activation contracts, not contract row order or
  an implicit FP8 preference. Model load now enforces those activation-family
  domains: FP8-CB requires `sm_89+`, while NVFP4-CB requires exact `sm_120` or
  `sm_121`, so RTX 40 and unsupported Blackwell capability families cannot
  acquire an NVFP4 candidate through a generic BF16-expansion fallback.
- The lane table now records ten closed-world SM120 cells, all `compile_only`.
  Five NVFP4 cells cover K1..K25: dense/routed decode are backed,
  role-unsplit routed batch is backed by persistent-B, and both
  expand-plus-BF16 routes are explicit fallbacks. Five FP8 cells cover the
  K4..K48 step-four producer ladder: dense/routed decode and dense
  expand-plus-native-CUTLASS W8A8 are backed, role-unsplit routed batch is
  backed by persistent-B, and routed expand-plus-BF16 is the fallback. The
  no-launch preflight cross-compiles the modules carrying both families,
  verifies all four modules as exact `sm_120`/`sm_120a` SASS, and resolves the
  vLLM native FP8/CUTLASS ABI. It does not promote NVFP4 K26..K32 research
  templates into the public contract.
- Schema and `contract_version` move together to
  `gridbook.runtime-contract.v11` / 11. All tensor/expert-parallel fields and
  FP8 format-ladder semantics retain their v10 meaning; lane eligibility gains
  the five compile-only SM120 FP8 cells above.

### Pre-release candidate (retracted above): FP8-CB K4..K48/4 expansion

- `formats[].rungs` is now explicitly the accepted **reader** domain and the
  new `formats[].producer_rungs` field is the canonical writer menu. FP8-CB
  readers accept K4/K8/K12/K16/K20/K24 plus every legacy K28..K48 artifact;
  v10 producers emit exactly K4..K48 in steps of four. NVFP4's reader and
  producer sets remain byte-for-byte the existing K12..K24 ladder.
- The generic dense/grouped FP8 CUDA decoder and transient expander admit the
  low reader rungs. Their aligned word extraction now predicates both tail
  words, including the K4/vector-31 case where the old unconditional `widx+1`
  read addressed one word beyond the exact packed body.
- The Blackwell fused source surface instantiates every producer rung and its
  low-rung LUT allocation remains 1 KiB-aligned. This is a compile/source
  surface change, not new GPU qualification evidence: low-rung device parity
  and served speed remain gates.
- `lane_eligibility` v2 publishes exact-platform, structure, regime and rung
  route facts separately from wire-format acceptance. The initial SM89 dense
  decode and batch rows are `compile_only`; they cannot make an artifact
  producer-legal. Graph/capture policy and per-run graph evidence deliberately
  remain outside Gridbook's packaged contract.
- `python -m gridbook.sm89_preflight --build-directory ... --receipt ...` is a callable,
  no-launch cross-compile/ABI preflight using explicit `sm_89` gencode. Its
  receipt ceiling is permanently `compile_only`. The 2026-08-24 no-device run
  produced an `sm_89` cubin and passed the full Gridbook main-extension symbol
  contract plus vLLM's direct FP8 quantization/CUTLASS ABI check. Durable
  receipt:
  `/home/rob/dq-runs/20260824_gridbook_sm89_v10_compile_only/receipt.json`,
  SHA-256
  `55a3679654db97617846f6addf46f8fbf617e16afbaeeebeea4ad2f454159bcd`.
  It loaded no model or tensor and is not device, graph, quality or speed
  evidence.
- Dense FP8 decode and expand + direct CUTLASS routes now write two-phase
  scalar-only route telemetry (error before launch, served after return),
  including the FP8-CB rung and exact M/N/K shape without a tensor read or
  synchronization.
- FP8 fused-mid-M selection is now platform-safe at model load: unset means
  auto (enabled only on cc 12.0/12.1), so an SM89 load never invokes the
  Blackwell-only fused JIT before reaching its native expand+CUTLASS route.
  Explicit `PRISMAQUANT_CB_FUSED_MIDM=1` on SM89 fails the load by name. The
  analogous routed fused eligibility check also rejects non-sm12x before
  querying that extension.
- Schema and `contract_version` move together to
  `gridbook.runtime-contract.v10` / 10. Tensor- and expert-parallel tables keep
  their v9 byte semantics unchanged.

## 0.9.0 — 2026-08-23

**Tensor parallelism is supported.** Gridbook artifacts serve above one rank
as a plain vLLM plugin — no forked runtime, no exported-byte change:

- **Dense CB Linears and dense FP8 source-passthrough Linears** shard at
  `--tensor-parallel-size N`.
- **Routed CB MoE expert stacks** shard on the expert axis at
  `--tensor-parallel-size N --enable-expert-parallel`; `-tp N` alone refuses
  at construction and names the flag.
- **Mixed-format fused projections** (a CB gate fused with a block-FP8 up)
  shard when every one of their roles does.

Measured, not simulated (2026-08-23): the DeepSeek-V4-Flash 92 GB CB artifact
— 176 dense CB Linears, 125 FP8 source-passthrough units, four mixed-format
fused projections and 43 routed CB MoE layers of 256 experts — served at
`-tp 2 --enable-expert-parallel` across two GB10 DGX Sparks (Ray, dual 200G
RoCE, the pinned vLLM `0.26.1rc1.dev693` image with Gridbook as a plugin).
Every MoE layer was admitted under expert parallelism with 128 of 256 experts
per rank, weights per rank fell to 41.9 GiB (about 82 GiB on one box), the
KV cache held 72,683 tokens at an 8k context, CUDA-graph capture completed,
greedy generation was coherent, and the same-corpus KL against two single-box
serves (889 positions, 746 BF16-confident) was 0.0505–0.0537 at 94.5–95.8%
top-1 agreement — inside the CB kernel's own nondeterminism floor (TP=2
same-session A/A 0.0405; single-box cross-session A/A 0.0483). Batch-1 decode
ran at 48.9 ms/token on GPUs shared with other work, against 47.6 ms/token
idle on one box: decode works, and no throughput claim is made for CB
artifacts.

Still refused above one rank, each by its own named gate: quantized-embedding
units (the Qwen3.8-27B CB artifacts), delegated compressed-tensors groups and
MXFP4/MXFP8 source passthrough. Grouped-BMM passthrough shards only at the
measured degrees 1, 2 and 4; expert parallelism is plain `use_ep` with the
MoE layer's own `tp_size` 1 (no all2all topologies, EPLB or
`skip_final_all_reduce`). Pipeline parallelism above one on DeepSeek-V4 is
blocked in stock vLLM (vllm-project/vllm#42769), independent of Gridbook —
use tensor parallelism. The contract schema is
`gridbook.runtime-contract.v9`; a producer pinned to an older schema refuses
this release whole until its pin is bumped deliberately.

Release-gate repair, tests only: five of this cycle's test files could not
run from the installed-wheel layout the release workflow uses (tests staged
outside the checkout, no vLLM). The three contract-enforcement tests now
locate the source checkout through `GRIDBOOK_SOURCE_ROOT` /
`GITHUB_WORKSPACE` like the other source-reading tests; the DSpark loader
and TP shard tests skip, by name, only the cases that need vLLM's linear
bases or compressed-tensors' `should_ignore_layer`; the EP exactness file
skips whole without vLLM. Runtime code is unchanged.

The sections below are the per-milestone record; where one says no two-node
serve has been measured, the measurement above supersedes it.

### Mixed-format fused projections serve above one tensor-parallel rank (contract schema v9)

One vLLM `MergedColumnParallelLinear` whose roles have DIFFERENT Gridbook
formats — DeepSeek-V4's shared-expert `gate_up_proj` fuses a CB gate with a
block-FP8 source-passthrough up — no longer refuses at TP>1. It was the last
refusing surface on the shipped DSv4 92 GB artifact.

- **The composite owned the defect, not either role.**
  `MixedFusedLinearMethod.create_weights` discarded `output_size` and handed
  every carrier its own per-rank width as that role's whole-tensor size, so
  each role's existing shard gate saw a degree-1 geometry and never fired. It
  now derives `row_degree` and `col_degree` from vLLM's own constructor
  arguments — never from `layer.tp_size`, which vLLM stamps on replicated
  planes too — and builds each carrier at `width * col_degree`. The CB
  `ShardGroupAlignmentError` and the source lane's `ShardAlignmentError` then
  decide legality per role, with their own structured fields, before any
  parameter exists.
- **The one law the composite does own** is the axis: a merged projection is
  column-parallel, and a row-parallel split of it would give every rank a
  partial sum of every role, which no role's format was qualified against.
  That is refused by name.
- **Narrowing.** The top-level mixed router is handed WHOLE checkpoint planes
  addressed by role name, so it takes this rank's slice before its shape gate:
  `narrow(output_dim, tp_rank * extent, extent)` where the extent is the
  destination parameter's own. That is exactly what vLLM's
  `load_column_parallel_weight` computes after its packing and block
  adjustment, so no packed factor, block size or superblock count is
  re-derived — and none can drift out of step with the role method that chose
  it. Per-tensor scalars replicate. A plane that is already rank-local is
  REFUSED rather than copied identically onto every rank.
- **Degree 1 is byte-identical.** The tensor-parallel rank is read once, in
  `create_weights`, and only when the plane is actually sharded; a test makes
  the accessor raise and requires construction to succeed.
- **Contract v9.** A third claim shape joins the two above: the
  `mixed_fused_projection` unit is a composite, not a format, so it publishes
  `shard_admission` `{axes: ["output"], per_role_law: "inherited"}` — the axis
  it admits, and the fact that each role's legality is that role's own row.
  Neither a numeric cap nor a law of its own would be true of it. Schema and
  `contract_version` move together to `gridbook.runtime-contract.v9` / 9.
- **The pin set is derived, not remembered.** `tests/test_runtime_contract.py`
  now walks the tree for every file that names `contract_version` or a schema
  string, compares the result against an explicit list, and requires each of
  them to name the CURRENT schema and (outside this changelog) no older one.
  A new pin now fails a test instead of shipping stale.

Measured on one process with the rank accessors pinned and construction run
per rank — no engine, no collectives, no second device. Every carrier plane
(CB `cb_qweight`/`weight_scale`, source `weight`/`weight_scale_inv`) is
byte-identical to what the role's own method plus vLLM's real
`load_column_parallel_weight` produce on that rank, at degrees 2 and 4, and
the ranks' slices tile the checkpoint plane exactly. Un-interleaving the
ranks' outputs reproduces the TP=1 output BITWISE for both roles, measured on
the decoded planes and fp32 reference products rather than on the serving
kernel. A two-node serve was not claimed here; it was measured on 2026-08-23
(the DeepSeek-V4 artifact's mixed-format fused projections served at TP=2) —
see the 0.9.0 lead above.

### FP8 source passthrough serves above one tensor-parallel rank

The `fp8_e4m3_ue8m0_block128` lane's dense arm no longer pins TP=1. The pin
(`_DSV4_RELEASE_TP`, plus the blanket refusal in `config._delegate_passthrough`)
was a release gate, not a measurement: the lane's own byte layout is exactly
what vLLM's stock narrowing already shards. It is replaced by the structural
law that makes that narrowing correct.

- **The law.** `BlockQuantScaleParameter` narrows the UE8M0 plane with the
  same arithmetic as the value plane, converting element counts to block
  counts by ceil division, so the narrow start is `rank * ceil(local / 128)`.
  That indexes the full plane's block grid correctly if and only if the
  per-rank extent on the sharded axis is a whole multiple of 128; below it,
  ranks silently read shifted blocks. `fp8_source_w8a16` now refuses anything
  else with `ShardAlignmentError` (a `ValueError`) inside `create_weights` —
  before any parameter exists for a loader to copy into, because a misaligned
  narrow is silently wrong rather than loud. On a merged plane the law is per
  fused role, since the block offsets are converted role by role.
- **The degree is structural.** Shard degrees come from `create_weights`' own
  arguments (`input_size` vs `input_size_per_partition`, `output_size` vs
  `sum(output_partition_sizes)`), never from `layer.tp_size`: vLLM's
  `LinearBase` stamps the world size onto every layer, including
  `ReplicatedLinear` and `disable_tp=True` merged planes. DeepSeek-V4 has 64
  such replicated passthrough planes per model that report `tp_size = N` while
  holding whole tensors; enforcing on `tp_size` would refuse them.
- **The grouped-BMM arm shards at measured degrees only, now 1, 2 and 4.**
  Column-sharding a grouped `wo_a` plane divides the kernel's group count
  (G 8 -> 4 at TP=2), so alignment cannot grant it and each degree is its own
  qualification. Degrees 2 and 4 were measured on the release device
  (2026-08-23, GB10 sm_121, DSv4 `wo_a` geometry: 1024 rows/group, K 4096,
  batch 1):

  | call | per call | vs G=8 |
  |---|---|---|
  | G=8, whole plane (TP=1) | 163.34 us | 1.000x |
  | G=4, one rank at TP=2 | 80.90 us | 0.496x |
  | G=2, one rank at TP=4 | 41.23 us | 0.253x |

  500 iterations per timed run after 50 warmup, three A/B/A repeats, median,
  CUDA-event timed, each arm rotating over a 268 MB working set so no arm is
  served from cache. **No occupancy cliff**: decode time falls with the bytes
  a rank holds, near-exactly linearly. (A hot loop over a single resident
  plane instead reports G=4 at 0.294x — a cache artifact of the microbench,
  not a speedup a serve would see. The honest number is 0.496x.) Exactness is
  stronger than the tolerance the qualification asked for: at both degrees and
  on **every** rank, the sharded call is BITWISE equal to the corresponding
  columns of the unsharded G=8 call, on the expand path and the GEMV path
  alike. Qualified degrees live in one place
  (`_DSV4_BMM_QUALIFIED_SHARD_DEGREES`), and the refusal message enumerates
  the whole qualified set rather than naming one geometry.
- **Other passthrough formats are unchanged.** `mxfp4_e2m1_ue8m0_g32` and
  `mxfp8_e4m3_e8m0_g32` have no sharded audit and keep refusing by name at
  dispatch. (Mixed-format fused planes kept their own separate refusal at the
  time of this entry; it was lifted below.)
- **Contract v7.** A passthrough unit whose method branches on an execution
  arm now publishes admission per arm, and a unit with a **law-admitted** arm
  carries no unit-level `max_world_size`: one scalar cannot cover a
  law-admitted arm and a capped arm at once. An armed unit whose every arm is
  capped keeps its unit-level 1. The FP8-source dense arm
  publishes `shard_admission` (`input_axis_group` 128, `output_axis_quantum`
  128, `merged_roles` `per_role_group_multiple`); its BMM arm publishes the
  same law plus `qualified_shard_degrees` `[1, 2, 4]`, keeping the pinned
  geometry (which names the unsharded plane). `mxfp8_e4m3_e8m0_g32` moves to
  the same shape with both arms still capped at 1 — no capability change.
  Schema and `contract_version` move together to v8 / 8.

### CB MoE expert stacks serve under expert parallelism (contract schema v8)

Routed CB MoE now serves above one rank with `-tp N --enable-expert-parallel`.
Tensor parallelism was never the right axis for it and still refuses: a CB
expert stack's last dimension is `(in/256)·type_size` superblock bytes, not
input columns, so vLLM's intermediate split would cut a packed superblock and
there is no partial-superblock decode. Expert parallelism shards the EXPERT
axis — the axis a stack is already indexed on — so every rank holds whole
experts, whole superblocks, and per-expert numerics byte-identical to a
single-rank serve. Dense Linears in the same model stay tensor-parallel at the
full world size, which is vLLM's own split.

- New torch-only `gridbook/moe_ep.py`. `local_expert_gather_index` turns
  `layer.expert_map` into this rank's global-id gather index ordered by local
  slot — a general `nonzero` gather, never a contiguous-range assumption,
  because vLLM's `round_robin` placement is genuinely interleaved — and
  refuses any map that is not a monotone bijection onto `range(E_local)`.
  `remap_local_expert_ids` rewrites global router ids to local slots inside
  the opaque custom op, aliasing remote pairs to the token's own smallest
  local expert at weight exactly `0.0` (expert 0 only when the token has no
  local pair). Compaction would make the pair count data-dependent and so
  uncapturable; aliasing keeps every shape static, and it is exact rather than
  approximate because `apply_router_weight_on_input` is refused outright for
  CB MoE, so the router weight is always applied in the combine.
- Both loaders — the `RoutedExperts` instance wrapper in `moe.py` and the
  top-level `load_weights` wrapper — gather a whole-stack `(E_global, …)`
  checkpoint tensor down to this rank through one shared rule that reads only
  what `create_weights` stamped on the destination param, so neither needs a
  module lookup. Their shape refusals now name the EP placement.
- `config.py::_require_ep_moe_serving` admits the stacked lane above one rank
  only under `use_ep` with MoE `tp_size` 1, no all2all topology (data-,
  pipeline-context- and sequence-parallel EP expect a dispatch/combine method
  Gridbook does not implement), no EPLB, and no `skip_final_all_reduce` —
  the last because Gridbook returns this rank's partial and relies on vLLM's
  stock final all-reduce. Mixed per-expert-format stacks keep refusing, now
  naming the mode that does serve. Each admitted layer announces itself.
- Contract: a new `expert_parallel` section on its own axis, with the topology
  predicate and per-unit `expert_admission` laws, validated field-for-field
  against the enforcement sites. `schema` and `contract_version` move together
  to `gridbook.runtime-contract.v7` / 7; a v6-pinned producer must refuse a v7
  contract whole until its pin is bumped deliberately.
- **Not measured**: no two-node serve has been run. The evidence is
  single-box, simulating the split in-process — per-rank stacks byte-identical
  to slices of the whole stack, remote pairs bitwise inert through the real
  decode and prefill kernels (proved by output invariance under three
  different alias targets), the remapped decode capturing and replaying as a
  CUDA graph, and per-rank partials summing to the whole-layer answer.
  Expert-parallel serving is correctness-only until the two-node gate runs.
- **2026-08-23 addendum**: the two-node gate has since run — the DeepSeek-V4
  92 GB CB artifact at `-tp 2 --enable-expert-parallel` across two DGX
  Sparks, every MoE layer admitted, KL against single-box serves within the
  CB kernel's nondeterminism floor. See the 0.9.0 lead above.

### Signed CB family (NVFP4_CB_S) deleted from the runtime

The sign-magnitude codebook family is gone from gridbook's runtime, not just
from the producer. PrismaQuant stopped emitting it on 2026-08-17 (an `n_sub=1`
codebook can never satisfy gridbook's native-FP4 predicate; "not performant,
we don't support them"), which left the runtime as a stale decoder whose own
test suite could no longer even build signed fixtures — the encoder refuses
`mode="signed"` at call time, so a full suite reported hundreds of failures.
Rob made the product call on 2026-08-23: **"The signed codebooks can be
deleted."**

Removed in one commit, so no contract row outlives its enforcement site:

- CUDA decode paths (`cb_gemv.cu`, `cb_fused_fp4_gemm.cu`,
  `cutlass_fork/sm120_cb_fused_fp4_mma.hpp`): the `fp4v2_signed_gather`
  routine, every `n_sub == 1` decode branch and LUT-sizing arm, and the
  CUTLASS `signed_mode` path are deleted. The bindings now REFUSE `n_sub != 2`
  by name — fail closed and loud, never a silent product-mode misread of
  signed bytes. Product-only kernels that already refused the layout
  (`cb_moe_persistent_b.cu`, `cb_gemv_v2.cu`) are untouched, as is every C
  type keyword (`unsigned` casts); none of the deleted code was keyword text.
- Python admission (`linear.py`, `moe.py`, `codec.py`): eligibility selectors
  admit only `n_sub in {fp8: 4, fp4: 2}`; the fused FP4 value-LUT builder
  drops its signed branch; the MoE constructor refuses non-product fp4
  schemes at load.
- Tests: all signed-fixture tests are deleted rather than skipped (their
  encoder anchor no longer exists), and two new refusal tests pin the
  removed family's loud failure at the dense GEMV and fused-prefill ABIs;
  CPU eligibility tests now assert n_sub=1 is ineligible outright.
- Contract: the `NVFP4_CB_S` format row and its closed-world
  `tensor_parallel` unit row are deleted from `runtime_contract.json`. The
  schema stays at v6: the table's SHAPE and reading rules are unchanged, and
  the closed-world semantics turns the missing row into a producer-side
  refusal by design — a reader pinned to v6 reads this contract correctly
  and refuses any stale NVFP4_CB_S artifact without a repin. A bump would
  have forced v6-pinned producers to refuse everything for no safety gain.
  No published artifact encodes an S-rung (all four published HF artifacts'
  sidecars verified clean; the only S-rung export ever found is the local
  2026-07-22 K-vs-S research bundle that the 0.5 load gate already refused).

### Contract schema v6: dense CB tensor-parallel admission is attested as laws, not a cap

The shard-aware loading lift above removed the blanket TP=1 gate that schema
v5's `tensor_parallel` section attested, so the table is reconciled with what
the code now enforces — and because the honest claim changes SHAPE, the schema
bumps to `gridbook.runtime-contract.v6` (`contract_version` 6) in the same
commit. Readers match the schema string exactly: a producer pinned to v5 must
refuse a v6 contract whole until its pin is bumped deliberately.

- Dense CB format families (`NVFP4_CB_K`, `NVFP4_CB_S`, `FP8_CB_K`) publish
  `shard_admission` instead of a numeric cap: `input_axis_group` 256
  (`codec.SUPERBLOCK`), `output_axis_quantum` 8 (fp4) / 16 (fp8) (the native
  kernel row quanta), and `merged_roles: "even_division"`. No dispatch path
  enforces a numeric ceiling for these units any more, so publishing any
  number would be an assertion; above one rank, admission IS these laws,
  evaluated per rank at weight construction (`ShardGroupAlignmentError`,
  derived from `gridbook/linear.py`). The validator now REFUSES a numeric cap
  on a CB row and pins every admission value to the enforcement site.
- Source-passthrough rows are unchanged and still capped at 1: MoE expert
  stacks, delegated stock compressed-tensors groups, source-passthrough units
  (FP8-source W8A16 dense release gate and pinned BMM geometry G=8/N=1024/
  K=4096/TP=1; MXFP8 BMM audited TP=1 only) keep refusing by name at their
  own sites.
- The root whole-model `max_world_size` field is removed: no single number is
  true of every dispatch path after the lift, and the closed-world reading
  never needed it (it was publisher-side symmetry only). Restoring one is a
  validator error.

`tests/test_runtime_contract_tp.py` derives every row from its enforcement
site's source text as before — the config-level derivation now pins the
per-surface split (blanket gate absent; all six refusal sites name non-dense
surfaces; dense CB arms construct without one) and a new derivation reads the
admission constants out of `codec.py`/`linear.py`.

### Contract schema v5: tensor-parallel capability is ATTESTED, not asserted

`runtime_contract.json` now carries a `tensor_parallel` section (schema
`gridbook.runtime-contract.v5`, `contract_version` 5): per-serving-unit rows —
CB format families and source-passthrough format ids, with per-arm rows and the
pinned grouped-BMM geometry for `fp8_e4m3_ue8m0_block128` — that restate what
the runtime's TP refusal sites actually enforce. Today that is a whole-model cap
of 1, enforced at `PrismaQuantConfig.get_quant_method` before dispatch, with
narrower arm-level gates in `fp8_source_w8a16.py` (dense release gate; BMM
geometry pin G=8, N=1024, K=4096, TP=1) and `mxfp8_dense_lane.py` (BMM audited
TP=1 only). The TP=1 behaviour itself is UNCHANGED — this change publishes the
fact, it does not lift the gate.

Reading is closed-world (`semantics: "closed_world"`): no matching row, an
unknown arm, a world size above the claim, or an off-pin geometry is a REFUSAL;
absence of a claim is never a clean bill. The packaged validator enforces the
publishing direction symmetrically: it refuses a contract that drops the table,
omits or invents a unit, drops a mandatory field, or publishes any number above
1. `tests/test_runtime_contract_tp.py` derives every row from the enforcement
sites' source text.

Compatibility rule: readers match the schema string exactly. A producer pinned
to `gridbook.runtime-contract.v4` must refuse a v5 contract whole and keep
serving against its pinned runtime until its pin is bumped deliberately.

### Shard-aware loading for dense CB Linears at TP>1 (2026-08-23)

Dense CB Linears now load correctly above one tensor-parallel rank with **no
change to any exported byte** — the campaign analysis (`tp-support-2026-08-22`)
established that CB artifacts shard purely at load time, and this wave
implements the loader side of that finding.

**What changed.**

- `config.py` no longer raises a blanket "TP=1 only" at method construction.
  The gate is replaced by per-surface policy: dense CB Linears construct at
  TP>1 under structured alignment gates; every other surface refuses AT
  CONSTRUCTION naming itself — MoE expert stacks (EP-first per `moetp.md`),
  delegated stock compressed-tensors groups, source-passthrough units,
  quantized embedding units and mixed-format fused projections. Ignored
  (BF16) targets keep vLLM-native sharding.
- New `linear.ShardGroupAlignmentError(ValueError)` with structured fields
  (`qname`, `axis`, `group_size`, `tp_degree`, `shard_size`, `detail`),
  raised at weight construction when a shard boundary would split a group:
  the input axis requires whole 256-weight superblocks; the output axis
  requires the native kernel row quantum (8 for fp4, 16 for fp8). The
  unsharded artifact error (`in_features % 256`) keeps its exact previous
  form.
- Merged GDN-style roles (`in_proj_qkvz` class) recover RANK-LOCAL role
  boundaries: checkpoint role rows are divided across ranks instead of being
  compared against local sums, which is true only at TP=1. A role whose
  checkpoint rows do not divide evenly is a structured refusal.

**What did not change.**

- The single-device path: TP=1 construction errors, load sequence and apply
  dispatch are byte-for-byte identical (the N-quantum check stays at its
  post-load home when unsharded).
- MoE stacked-expert loading, MXFP8 lane and FP8_SOURCE BMM remain refused
  at TP>1 by their own named gates.
- Unsupported CB layouts (legacy v1 fp4, signed rungs) refuse exactly as at
  TP=1, through the same model-load format gates.

**Honest scope.** Correctness only. No two-node serve has been measured on
this hardware (10 GbE Realtek, no RDMA), nothing here argues decode speed,
and per-token dynamic FP8 activation scales on row-parallel layers are
computed over each rank's local K window at TP>1 (as with stock W8A8
schemes), so served logits are not bit-identical to TP=1; quality claims
belong to the standing same-session KL gate. Loader-contract tests simulate
vLLM's attested narrowing in-process; no distributed run exists yet.
*(2026-08-23: superseded in part — the two-node serve has been measured, and
the link is dual 200G RoCE, not the 10 GbE assumed above; see the 0.9.0
lead. The activation-scale caveat stands.)*

### Measured negative result: the R2 default flip was attempted and REVERTED

`PRISMAQUANT_CB_FP4V2_DENSE_R2` stays **default OFF**. A flip to default-ON was
prepared, committed, and then reverted the same day when a wider grid found
regressions the promotion evidence had missed. Recorded here because the
measurement is the durable part.

**Where R2 wins.** On the Qwen3.8-27B CB gold artifact's four real fp4 rungs
(`NVFP4_CB_K12/14/16/18`, n_sb 20 and 68), all 40 (shape × M) points over
M∈{1,2,4,8,16} are bit-identical AND faster — per-M aggregate −3.44% to
−10.17%.

**Where R2 loses.** At **M=1 with `n_sb < WARPS`**, the aligned-down u64 burst
staging never amortizes: a warp does ONE iteration and some warps idle, so the
setup cost dominates. Measured with 5 interleaved A/B/A repeats:

| cell | n_sb | R2 delta | control spread |
|---|---|---|---|
| k=12 K=768 M=1 | 3 | **+9.14%** | 0.36% |
| k=12 K=1280 M=1 | 5 | **+4.50%** | 1.33% |
| k=12 K=1536 M=1 | 6 | +1.80% | 2.04% (borderline) |

`n_sb=4` is unaffected — it takes the `use4` branch (4 warps, all active). The
same cells at **M=2 all win** (−2.7% to −6.3%), so this is specifically the
single-row, few-superblock tail.

**Why the first evidence missed it.** The perf sweep used only the artifact's
real K∈{5120,17408} (n_sb 20/68); the 1728-config edge fuzz *did* cover small
K but asserted only **bit-identity**, never timing. Small `n_sb` was therefore
never perf-tested. The lesson is the B2 lesson again in a new costume: a grid
chosen from the shipping artifact's geometry cannot clear a kernel for
geometry the artifact does not contain.

**The crossover is RAGGED — a simple `n_sb >= X` dispatch would be fitting
noise.** Swept n_sb∈{1..24} at k∈{12,16}, N=4096, 4 interleaved A/B/A repeats:

| n_sb (M=1) | k=12 | k=16 | warps |
|---|---|---|---|
| 1, 2, 3 | LOSS +8.9…+11.8% | LOSS +7.5…+13.0% | 8 |
| 4 | ~ | win | **4** (`use4`) |
| 5, 6 | LOSS | LOSS(5) / ~(6) | 8 |
| 7, 8 | ~ | win | 8 |
| **9, 10** | **LOSS +2.4%** | win | 8 |
| 12, 20 | win | win | **4** (`use4`) |
| 16, 24 | ~ | win | 8 |

Two structural facts, and they point at the kernel rather than the launcher:

- **M=2 wins essentially everywhere** (only n_sb=1/k=16 loses, +1.08%). This
  is an M=1 problem.
- **The `use4` branch never loses** (n_sb∈{4,12,20}). The 8-warp branch is
  ragged: n_sb=8/16/24 are fine — every warp does equal iterations — but
  n_sb=9/10 LOSE at k=12, where one or two warps do TWO iterations and the
  rest do one. The straggler carries R2's un-amortized setup on the critical
  path.

So the mechanism is **warp-load imbalance at M=1**, not a size threshold, and
it is k-dependent (k=12 loses at n_sb=9,10 where k=16 wins). A `use4`-style
dispatch cannot express it cleanly.

**What a future flip needs:** a kernel-side fix for the M=1 imbalanced-tail
case — early-exit or a cheaper staging path when a warp's iteration count is 1
or the block is imbalanced — not a launcher threshold. Plus the served leg the
reverted attempt lacked. All 60 crossover cells were bit-identical, so the
correctness story is unchanged throughout.

Bit-identity itself is NOT in question and is now permanently gated:
`tests/test_dense_fp4v2_r2_edge_geometry.py` sweeps 1728 configs over
k∈{12..20} × K∈{512,1024,2048,3072} × N∈{1,17,48,96} × M∈{1,3,5,8,15,16} ×
sched∈{None,db} with zero mismatches, targeting the staging boundary that
moves with `type_size`. That test is kept.

### Fixed: the R2 M=1 imbalanced-tail regression, kernel-side (default stays OFF)

The regression above is fixed IN the `R2BACKPORT` instantiation — no
`n_sb`/warp-count dispatch was added — so the default flip is unblocked pending
only the served leg. Ingredient isolation over ONE binary with env-selectable
arms (interleaved A/B/A, every arm bit-compared per cell;
`dq-runs/r2-tail-exp`) attributed the loss to two staging-side details, and
**acquitted the packed gathers** (the arm that adds ONLY uint2 gathers tracks
legacy within noise at every cell, including n_sb=1):

| isolation arm (k=12, M=1) | n_sb=1 | n_sb=3 | n_sb=9 |
|---|---|---|---|
| legacy | 14.04 us | 15.38 | 27.53 |
| shipped R2 | 16.05 (+14.3%) | 17.11 (+11.2%) | 28.95 (+5.2%) |
| gathers only | 14.14 (+0.7%) | 15.28 (~) | 27.58 (~) |
| burst staging only (scalar gathers) | 16.21 (+15.5%) | 17.13 (+11.4%) | 29.17 (+6.0%) |
| R2 + last-sb burst | 14.76 (+5.1%) | 16.02 (+4.2%) | 27.90 (+1.3%) |
| R2 + last-sb burst + unconditional w2 | 14.58 (+3.9%) | 15.80 (+2.7%) | 27.41 (−0.4%) |

1. **Last-superblock byte-path fallback.** It put the straggler iteration of an
   imbalanced block (the ONLY iteration when `n_sb <= WARPS` for the row-final
   warp) on a slower path, and inlining both stage variants produced a
   multi-way specialized dispatch tree in SASS.
2. **Predicated third stage-word read.** The per-lane-divergent predicate on
   the hot path costs more than the smem load it saves on GB10; for every
   shipped rung k≤20 (`rem+k ≤ 51 < 64`) the word can never contribute, and
   the guarded compose discards whatever is read, so reading it unconditionally
   (legacy parity) is bit-safe at every k.

The landed fix: burst staging on EVERY superblock behind two fail-loud
launcher checks — `qw.stride(0) - n_sb*type_size >= 8` (pad_qweight's
documented ≥8-byte read-slack invariant covers the ≤7-byte window overrun of a
row's final superblock) and the worst-case burst window inside the 208-byte
stage slot — plus an UNCONDITIONAL third-word read, plus each iteration's head
offset derived from the row phase BEFORE the stage issues (an address
property), taking the off8→bitpos chain off the post-barrier critical path.
The grouped MoE kernel keeps its own last-superblock byte fallback (expert
stacks have no per-row padding to check).

Measured result (same harnesses as the negative result,
`dq-runs/r2-tail-exp/logs/both.log`):

- **Crossover sweep** n_sb∈{1..24 grid} × k∈{12,16} × M∈{1,2}, interleaved
  A/B/A: **zero loss cells**, every cell bit-identical. M=1 deltas span
 −0.03%…−18.91% (the +11.75%/+9.14%/+2.43% cells are now
 −0.78%/−0.52%/−7.48%); M=2 spans −2.39%…−17.27%.
- **Qwen3.8-27B real-shape bench**: 0 of 40 (shape × M) points regress beyond
  their control drift; per-M kernel-time aggregates M=1 −17.69%, M=2 −18.59%,
  M=4 −13.15%, M=8 −20.08%, M=16 −11.39%. The large-n_sb win is LARGER than
  the shipped backport's (−3.44%…−10.17%), not traded away.

`PRISMAQUANT_CB_FP4V2_DENSE_R2` remains default OFF; flipping it is now a
served-leg decision, not a correctness or shape-risk one. Callers hitting the
new pad-slack check were never safe under the old last-row arithmetic either;
all production paths build weights through `codec.pad_qweight`.

## 0.8.12 — 2026-08-22

- **Added: dense FP4-CB v2 GEMV round-2 backport behind
  `PRISMAQUANT_CB_FP4V2_DENSE_R2` (default off).** The grouped MoE kernel's
  three load-path optimizations — predicated spill-word read (the window
  lemma proves `rem+k > 64` is exactly the spill condition, live only at
  persistent-B k=44 among shipped rungs), packed `uint2` codebook gathers
  with the per-element bf16→f32 chains unchanged, and aligned-down u64 burst
  staging with last-superblock byte fallback — ported to the dense kernel as
  a second template instantiation; the legacy path text is untouched.
  Bit-exact in both modes (full `test_cuda_gemv.py` + a 15-case dual-mode
  gate); on GB10 the flag wins all 32 measured points, gains growing with k
  (−7%…−20% at k20), consistent with the decode chain's compute-bound ncu
  profile. Opt-in until the served NATIVE-PARITY protocol runs.
- **Proposed and REJECTED: persistent-B packed-superblock staging
  vectorization (B2)** — byte-granular to u32-interior copies with
  funnel-shift edges for the odd `4k+9` source phases and whole-slot zeroing.
  Byte-neutrality held (preconditions P1–P5 documented in-source per the
  staging-vectorization theorem; decode-probe `torch.equal` suites green),
  but independent A/B measurement on the DSv4-dominant k=12 rung showed the
  vectorized build +7…+11% whole-operator slower than the byte loop (the
  `__noinline__` spelling slower still, +14%), and the recorded rationale —
  that `__noinline__` scoping is load-bearing because inlining cost ~26
  registers/thread and a ~23% regression — is contradicted by `cuobjdump`
  dumps of the built binaries (the inlined build allocates FEWER registers
  than the byte-loop baseline). Not merged; the byte-granular staging
  remains.
- **SHIP consensus 2026-08-22 (coordinator + adversarial Ox Alpha): FP8-family-only persistent-B
  staging vectorization (B2-S3)** — the rejected B2's copy salvaged behind a
  compile-time family dispatch: at both mainloop staging sites, FP8-CB takes
  u32 words when the source plane is runtime word-aligned (the byte loop
  stays as a fallback so no host check refuses a currently-served layout)
  with word-granular slot zeroing, while the FP4 instantiation's statements
  remain the baseline loops verbatim. The fp4 kernels' compiled form is
  proven, not asserted: all eight fp4 mainloop instantiations are
  SASS- and resource-identical to baseline under `cuobjdump` (hot tile stays
  at 112 registers — the register cliff that killed B2 is avoided by
  construction). Bit-neutral on 25 `torch.equal` probe cells including every
  shipped FP8 rung (k∈{28,36,44,48}); whole-operator FP8-CB on DSv4 shapes
  measures −10.2…−12.0% at k=28 (T=128/512/2048), −14.5% at k=36 and −11.2%
  at k=48, with fp4 k=12 at noise (−0.03%). Reaches DSv4 serving only after
  the pooled-books reburn — today its FP8-CB routed layers use per-role
  split books and keep the bridge; the win applies now to any MoE artifact
  whose FP8-CB layers take the persistent-B arm.
- **Changed: sm12x grouped-BF16 lane packs expert blocks WITHIN each chunk**
  (ROADMAP K1.5): multi-chunk layers now get the swizzle-group tile-order win
  that previously required one chunk to cover every expert. Per-expert
  operands and reduction order are chunk-boundary-independent, so outputs are
  bit-identical (`torch.equal` end-to-end at top_k=1; atomicAdd-combine
  envelope respected at top_k>1); graph-capture refusal semantics preserved.
  Isolated stage-one gather measures −10.3…−11.0% on straddling segments
  (consensus re-measurement 2026-08-22: −9.1% and −11.1% on two independent
  draws); the author's "1.001× on a uniform-router control" has no recorded
  construction and a random-uniform router with ragged counts still measured
  −10.4%, so inertness is a property of no-straddle count vectors, not of
  router uniformity. Landed unconditional.
- **Added: `docs/audits/math_review_2026-08-21.md`** — byte-law triple
  agreement, decode-window containment/spill characterization theorems, the
  ρ-threshold tightness proof (with the residue-premise correction and the
  non-monotone advantage profile any calibration sweep must clear),
  empty-expert cost bound (~0.59 ns vs ms cells), staging byte-neutrality
  theorem, and the K1.3 roofline memo (conditional-GO shape: TM=256/TN=32,
  honest payoff band TTFT 1.05–1.17×, gates G0–G3 before any new schedule).

- **Corrected: the scope of the 0.8.11 capture refusal, and the end-to-end
  evidence the 0.8.11 entry lacked.** The 0.8.11 note said "artifacts
  carrying BF16-bridge layers still refuse `FULL_AND_PIECEWISE`". That is
  wrong for the DEFAULT bridge: `block_offsets` has exactly one consumer,
  the OPT-IN sm12x-native BF16 bridge (`PRISMAQUANT_CB_BF16_SM120=1`), and
  neither the default expand + grouped bridge nor the persistent-B lane
  calls `_padded_route` at all (they route with `_expert_counts`/`cumsum`,
  no host read). The refusal message now names the flag. Measured
  2026-08-21 on the published DSv4-Flash 87 GB body (32 persistent-B + 11
  announced-bridge layers, `gridbook:0.8.11-clean-187c721`, vLLM 0.26.1
  eugr build): `FULL_AND_PIECEWISE` with `cudagraph_capture_sizes` up to 64
  (`--max-num-seqs 32`) captures 11 piecewise + 7 full graphs and serves;
  single-stream decode 20.56–20.64 tok/s, identical to the card command's
  `FULL_DECODE_ONLY [1,2]` arm (20.53–20.61 on 0.8.11, 20.54–20.63 on
  0.8.10) because batch-1 decode is a full graph in both modes. The
  throughput consequence of capturing grouped decode sizes is reported in
  the PrismaQuant benchmark record, not here.
- **Testing: repaired three pre-existing test failures (all failing at
  0.8.11; no runtime behavior changed).** Two `test_moe_mixed.py` cases reset
  the FP8-GEMV-v2 dispatch flag via `moe_gemv_select._CB_FP8_GEMV_V2`, an
  attribute the lane-select latch refactor removed, and raised
  `AttributeError`; they now reset it through `lane_select.reset_for_tests`.
  One `test_nvfp4_static_runtime.py` case asserted a fused-fp4 layer was
  ineligible with "no loaded kernel state" but never forced the compiled
  extension absent, so it passed only where native fp4 never built and failed
  on the sm121 build host; it now forces `get_fused_fp4_ext` to return `None`.

## 0.8.11 — 2026-08-21

- **Fixed: the padded grouped MoE routing host-synced inside CUDA-graph
  capture (#47).** `_padded_route`'s two host reads — the optional trim count
  (`n_blocks.item()`) and the per-expert `block_offsets` (`.tolist()`) — are
  routing-dependent, so on vLLM 0.27, whose default `cudagraph_mode` is
  `FULL_AND_PIECEWISE` and captures routed prefill sizes, a stock
  `vllm serve` of any CB MoE artifact died at engine start with `operation
  not permitted when stream is capturing`. The crash was protective: even a
  permitted read would bake one capture-time routing's block count into
  every replay, so no captured trim can be correct. Under capture
  `_padded_route` now forces `trim=False` and launches the full
  static-capacity tail — the layout is data-independent (`cap_blocks =
  P//tile_m + E`, known from shapes alone) and is exactly the arm
  `PRISMAQUANT_CB_GROUPED_TRIM=0` already selects, so replays recompute the
  routing on device and stay correct for any token mix. The BF16 grouped
  bridge, whose expert-chunk launches slice by per-expert host boundaries no
  static shape can stand in for, now refuses capture with an error naming
  the remediation (capture sizes within the grouped-GEMV decode regime,
  `<= MOE_PREFILL_M_THRESHOLD = 16` tokens — e.g.
  `cudagraph_mode=FULL_DECODE_ONLY` — or `--enforce-eager`) instead of the
  runtime's opaque abort. Eager dispatch is unchanged, decode GEMVs were
  already sync-free, and fused-lane numerics are unchanged in both arms
  (padding rows scatter into the throwaway row; pad tiles carry expert id
  −1). Gated by `tests/test_padded_route_capture.py`: a real
  capture-and-replay of `_padded_route` against rewritten routing, host-read
  poisoning on the capture arm, bit-identity of the static-capacity layout
  with the eager `trim=False` arm, and a negative control proving the trim
  read still aborts capture on this torch.

  Verification scope: this is an operator-level fix with operator-level
  evidence — a real `torch.cuda.CUDAGraph` capture and replay of
  `_padded_route` on torch 2.11.0+cu130 (GB10, sm121), plus the contract tier
  against a stubbed vLLM surface. No end-to-end serve under stock vLLM 0.27
  defaults was run, so "the engine starts on 0.27 defaults" is the expected
  consequence, not a measured one. Artifacts whose layers all take the fused
  lanes are expected to capture; artifacts carrying BF16-bridge layers still
  refuse `FULL_AND_PIECEWISE` by design, now with the remediation named
  instead of the runtime's abort. Confirmation on a 0.27 stack is requested
  in #47.

- **Fixed: `FULL_DECODE_ONLY` capture aborted the load in the MXFP8 dense
  lane.** `_SfOffsetCache` computes swizzled-plane offsets on the host and
  moves them with an unpinned `.to(device)`.
  `process_weights_after_loading` resolved the **B side** from weight
  dimensions known at load, but the **A side** is keyed by the runtime row
  count, so under `cudagraph_mode=FULL_DECODE_ONLY` the first call at each
  capture size landed *inside* the capture region and raised `Cannot copy
  between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor
  is pinned`. It surfaced as `cudaErrorStreamCaptureUnjoined` at
  `capture_end()`, because DeepSeek-V4 reaches this lane from inside
  `maybe_execute_in_parallel`, leaving that side stream unjoined. The A-side
  offsets are now pre-warmed at load for every `cudagraph_capture_sizes` entry
  — `docs/KERNELS.md` CUDA-graph safety rule 3, mirroring `cb_gemv_v2_prepare`
  and `cb_moe_persistent_b_prepare`. **No numerics change**: the same offsets,
  computed earlier. Eager serving is unaffected (the reader returns `()` when
  no capture sizes are configured, and the pre-warm is then a no-op).

  Measured on one GB10 / DGX Spark (`sm_121`, arm64), vLLM
  0.26.1rc1.dev515, torch 2.11.0+cu130, a DeepSeek-V4-Flash-0731 NVFP4-CB
  artifact at TP=1, `--kv-cache-dtype fp8`, `--max-model-len 8192`,
  `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",
  "cudagraph_capture_sizes":[1,2,4,8]}`: capture previously failed the load
  outright; it now succeeds, and single-stream decode improves **12.05 ->
  14.08 tok/s (+16.8%)**, 83.0 -> 71.0 ms/token.

  Per-token logprobs under capture are **bit-equal** to `--enforce-eager` on
  7 of the 8 probe prompts. The 8th is excluded on the basis of an
  eager-vs-eager control run *before* the captured arm was compared: it is a
  near-tie that flips between two identical eager runs, so it cannot serve as
  a reference. Protocol: prefix caching off, serial single requests — under
  the serving default (prefix caching on) and concurrency, eager is not
  run-to-run reproducible either.

## 0.8.10 — 2026-08-18

- **0.8.9 could not load a per-expert split-format (mixed) expert bank at
  all**: the tri-state refactor renamed `cb_fp8_gemv_v2_requested` to
  `cb_fp8_gemv_v2_mode`, and `moe_mixed.py` still imported the old name, so
  `config.py`'s split-bank dispatch died with an ImportError on any artifact
  declaring `per_expert_format_groups`. Uniform stacks — every published
  artifact — were unaffected. The break shipped green because
  `tests/test_moe_mixed.py` used `pytest.importorskip("gridbook.moe_mixed")`,
  which turned our own ImportError into a silent skip on every vLLM-less
  runner; the file now skips only for a missing vLLM and fails loudly on
  gridbook's own import errors, and a CPU-suite guard statically parses
  `moe_mixed`'s imports from `moe_gemv_select` and holds them against the
  real module (resolved through `importlib.util.find_spec`, so the guard also
  runs in the release train's installed-wheel layout, where the repo tree is
  not at a path relative to the test file).
- The mixed method's FP8-v2 dispatch gate is tri-state-correct: `require`
  keeps the exact fail-load A/B-arm refusal on mixed FP8 groups, `auto` (the
  unset default) keeps the inherited kernel for them with the reason
  announced per stack — a mixed FP8 group is an off-cell, not an error — and
  `off` skips. Verified in the pinned serving container: `gridbook.moe_mixed`
  imports and the selector resolves `auto` with vLLM present.


## 0.8.9 — 2026-08-18

**The qualified CB kernels are now the default.** Three selectors gain a
tri-state with `auto` as the unset meaning; every previous explicit spelling
keeps its exact prior semantics, so no launch contract changes out from under
an operator — what changes is what an UNSET flag serves.

- **`PRISMAQUANT_CB_MOE_PERSISTENT_B` unset now means `auto`** (was: the
  expand+bridge route everywhere). Each routed CB layer engages its family's
  persistent-B decode-in-mainloop arm — FP4-CB two-tier v2 (ROADMAP K1.1) or
  stock FP8-CB (K1.1's second payload family) — where the load-time predicate
  and the extension attest, and keeps the expand+bridge route where they do
  not, with a per-layer fallback line naming why. `1`/`require` keeps the
  0.8.8 A/B-integrity semantics exactly: a layer no arm can serve fails the
  LOAD by name — on the shipped DSv4 87 GB body that is every FP8-CB expert
  layer (per-role split books), which is precisely why the default is `auto`
  and not `1`. `0`/`off` is the kill switch. Promotion record: bitwise decode
  identity per arm; whole-routed-operator microbenchmark wins every cell
  (FP4 1.05–3.36× over the bridge; FP8 15.8–18.4× at DSv4 E=256/K28/topk6
  shapes, rel-L2 ≤ 6.2e-04); served NATIVE-PARITY — the FP4 arm's
  same-session served A/B on the DSv4 92 GB body (kl_mean −0.051 %, direct
  PPL −0.30 % — arithmetic noise) and the 0.8.9 default-state served KL/PPL
  leg on the shipped clean 87 GB body against its gold record — env fully
  unset, 32 FP4-CB layers on the lane, 11 per-role FP8-CB layers on the
  announced bridge fallback: kl_mean +0.17 %, kl_p99 −0.03 %, direct PPL
  −0.06 % (inside the ±0.7 % cross-session KL envelope). The container build gains a soft Blackwell-gated prewarm for
  the module so the first routed CB load does not pay the JIT build on the
  request path.

- **`PRISMAQUANT_CB_GEMV` unset now means `auto`** (was `inherited`). The
  smem-resident-dictionary v2 decode GEMV engages only where the compiled
  occupancy predicate (`cb_gemv_v2_prefers_inherited` — the arithmetic of
  the binary that launches) says it wins; an unavailable extension degrades
  LOUDLY to the inherited kernel, which is exactly the pre-0.8.9 default.
  On K=2048/4096 shapes the compile-time virtual-warp specialization is
  bit-exact to the inherited kernel for every rung (all
  `PRISMAQUANT_CB_W2_*` overrides absent); other widths retain rowpack
  reduction order, may reassociate, and carry no served A/B — the predicate
  is the only gate there. `inherited` remains the kill switch and never
  probes or builds.

- **`PRISMAQUANT_CB_FP8_GEMV_V2` unset now means `auto`** (was disabled;
  the flag also gains the `require`/`off` word spellings). Exactly the
  qualified cell — `k=28/n_sub=4/type_size=112` at K in {2048, 4096} — takes
  the routed FP8-CB whole-row sibling; every other routed FP8-CB stack keeps
  the inherited kernel with the reason logged per stack. `1` keeps its
  strict pre-0.8.9 contract: an A/B-arm requirement under which any off-cell
  stack fails the load. Promotion record: 17/17 final-source operator gate;
  zero full-vocabulary and router-route delta over 240 scored positions on
  the exact artifact; the final-binary served rerun (B-v3, 2026-08-14,
  FULL_DECODE_ONLY graphs, route census, quiesced host) measured **+8.62 %
  decode throughput** with acceptance parity and attributed the earlier held
  signal's cross-arm integrity failure to host interference. Soak and
  high-concurrency protocols have not run as named gates; the sibling serves
  only the routed `M ≤ 16` GEMV band.

- `lane_select` grows `latched_choice` — the tri-state sibling of
  `latched_bool`, same strict-parse/latch/mid-process rules — and the
  persistent-B lane module gains `mode()` and `probe_lane()` (the
  non-raising resolution `auto` needs so its fallback telemetry can name
  the reason).

- Two d2r contract tests broken by the FP8 arm's ABI bump at `d9dfe53`
  (schema 1→2, three FP8 production symbols, the fp8 full-codeword decoder)
  are repaired; the persistent-B FP8 arm's stray "ROADMAP K1.2" comment
  labels are corrected to "K1.1's second payload family" (ROADMAP K1.2
  proper is the closed fused mid-M rung-surface item).

## 0.8.8 — 2026-08-15

- **The quantized embedding shipped in 0.8.7 could never be reached on
  Qwen3.5/3.6.** vLLM dispatches a quantized module from inside the layer's own
  constructor, so a model class that builds
  `VocabParallelEmbedding(vocab_size, hidden_size)` with neither a
  `quant_config` nor a `prefix` never calls `get_quant_method` for its lookup
  table. 0.8.7 shipped both the method and the dispatch branch, and on this
  architecture neither was ever reached — a mechanism can be complete and still
  be dead if nothing calls it. Most vLLM models pass both arguments
  (`deepseek_v2`, `arctic`, `dbrx`, …); `qwen3_5` passes neither. The failure is
  silent at construction and surfaces much later as vLLM refusing a parameter
  nothing claims: *"There is no module or parameter named
  `embed_tokens.weight_global_scale`"*.

  `embedding_construction.py` supplies the two arguments the model's own
  `__init__` already holds. It is inert unless the ambient config is ours, that
  config declares a unit for this exact prefix, and the embedding is being built
  without a `quant_config` — so another checkpoint, another architecture, a
  model class that already does this correctly, and a future vLLM that fixes it
  upstream all keep their current behaviour. `ParallelLMHead` is excluded; it
  subclasses the embedding but is an output projection served through
  `config_groups`. The arguments are substituted *during* construction rather
  than the module being rebuilt afterwards, because a rebuild would first
  allocate the full BF16 table and free it — and the artifact this exists for is
  the one being served under a deliberately tight memory budget.

  The embedding is the **first** module built, before any `get_quant_method`
  call has forced a pointer config to resolve, so the wrap resolves it itself;
  without that the units are still empty and a declared table is silently served
  unquantized.

  Verified on a real artifact: a 12.98 GB Qwen3.8-27B CB checkpoint now loads
  and serves on vLLM 0.26.1rc1 with 11.42 GiB of resident weights, and fits a
  16 GiB budget with 2.75 GiB of KV cache. Before this it did not load at all.

## 0.8.7 — 2026-08-15

- **Serve a quantized embedding table.** `model.embed_tokens` is 2.37 GiB on a
  27B-class vocabulary and nothing was pricing it: the producer could pack it,
  but every consumer path either ignored the declaration or routed it to
  vLLM's compressed-tensors embedding handler, which raises for FP4/FP8. The
  new declaration claims those units for Gridbook's own embedding method. It is
  **weight-only by construction** — a lookup has no input activation, so an
  emitted `input_global_scale` would be an unmatched checkpoint key at load —
  and a declared unit is parsed against the same target set as the passthrough
  units, so a unit claimed by two vocabularies fails at parse rather than in
  whichever branch of `get_quant_method` happens to run first.

- vLLM's post-load parameter sweep is an `isinstance` check against
  `VocabParallelEmbedding`, not duck typing. A structurally complete
  non-subclass is skipped **silently** and then dies on its first forward, so
  the embedding method subclasses rather than mimics, and `ParallelLMHead` is
  modelled as the subclass it actually is. `lm_head` remains refused here:
  the producer already declined it, and the consumer now declines it too
  instead of accepting a declaration nothing can load.

- **A stock NVFP4 W4A4 Linear could not be served at all on vLLM 0.26.** The
  audited kernel table knew `CutlassNvFp4LinearKernel`; 0.26's selection ladder
  picks `FlashInferCutlassNvFp4LinearKernel`, which was absent, so an
  activation-quantized declaration resolved UNKNOWN and failed closed on every
  delegated dense W4A4 group. The re-audited entry is added, with the audit
  recorded where the module says its authority lives.

- This tag also releases the routed-MoE kernel work that landed on `master`
  after 0.8.6 and had not yet been tagged: the persistent-B v6/v7 lane, the
  bf16 decode-to-register lane, and an exact FP8 CB GEMV v2. **Every one of
  them is opt-in and default-off**, so with no environment set this release
  dispatches exactly as 0.8.6 did and the only behaviour change is the
  embedding mechanism and the preflight entry above. `PRISMAQUANT_CB_GEMV`
  unset resolves to `inherited`, the shipping default, which never even probes
  or builds the v2 extension; `PRISMAQUANT_CB_FP8_GEMV_V2` unset is disabled;
  `PRISMAQUANT_CB_MOE_PERSISTENT_B` and its `_D2R` sibling are latched booleans
  defaulting to off. (`..._PERSISTENT_B_CFG=0` is the production setting *within*
  that lane — the kernel picking from the shapes — and is reached only once the
  lane is requested.) Each selector is resolved once per process and refuses
  both an unknown spelling and a mid-serve change, because either would make a
  number taken from that process unattributable.

- No format, contract, or ABI change: `gridbook.runtime-contract.v4` and the
  three-feature ABI closure are byte-identical to 0.8.6, so a consumer pinning
  this release moves only the version, commit and wheel hash.

## 0.8.6 — 2026-08-13

- **Load separately quantized DeepSeek-V4 DSpark drafts without rewriting
  namespaces.** DSpark constructs quantization methods under
  `model.layers.{L+stage}`, registers parameters under
  `model.layers.{stage}`, and stores draft tensors under `mtp.{stage}`.
  Gridbook now reuses DSpark's own `_remap_dspark_name` at the top-level load
  boundary, including routed-expert stacks and mixed fused roles, so those
  three namespaces reconcile explicitly and fail closed on ambiguity.

- DSpark construction now binds Gridbook's lazy `quant_config.json` and
  `cb_codebooks.pqcb` lookup to vLLM's explicit
  `speculative_config.draft_model_config`. The target body keeps its own
  `model_config` authority even when speculation is configured; a missing or
  malformed explicit draft config fails before draft initialization. This
  prevents a separately quantized draft from accidentally opening the target
  model's sidecars.

- The optional `gridbook.dspark-target-bridge.v1` record binds activation
  scalar targets to physical checkpoint tensors when an activation execution
  contract exists. Weight-only CB drafts do not invent an activation bridge;
  their construction-to-registered loader mapping is still covered by the
  new `abi_features.dspark_construction_physical_bridge = 1` runtime feature.
  The packaged loader allow-list now includes the unforked vLLM DSpark module.
  Adding this closed feature advances the producer/runtime contract from v3 to
  v4; consumers must not reinterpret the expanded key set as the older schema.

- This release adds no new quantization format. Target-only serving remains the
  DSV4-Flash shipping default. On the exact qualified candidate
  stack, native MXFP4 DSpark improved the fixed eager 8 x 128 suite by only
  `0.95%` while adding `10.13 GiB` of model residency, and K12-CB regressed
  throughput by `10.81%`; DSpark therefore remains an opt-in experimental
  companion rather than a production default. The final clean-commit image
  still has to repeat the release smoke before tagging.

- The optional smem-resident FP4-CB decode GEMV now preserves the inherited
  default's exact reduction order on DSV4 K=2048/4096 via eight named virtual
  warp accumulators. The final-source cc 12.1 gate (source SHA256
  `d72b15ecaad14e7af07f8af555259f5d1423cee2dacce160c13b3caf7b8bc92b`)
  passed 30/30 eager/graph operator tests for k12/k16/k18 and both decode
  contracts, measured a bit-exact 1.8175–1.9977x v1 direct-op speedup, and
  produced zero full-vocabulary KL, NLL, PPL, and target-logprob delta over 240
  same-process DSV4 positions. This is a release-shape candidate behind
  explicit `PRISMAQUANT_CB_GEMV=v2`, with every `PRISMAQUANT_CB_W2_*` override
  absent; the global default remains `inherited`, and final clean-wheel served
  graph/throughput qualification remains open. Exact evidence paths are in
  `docs/RELEASING.md`.

- Routed FP8-CB decode now has an independent, default-off whole-row sibling
  behind `PRISMAQUANT_CB_FP8_GEMV_V2=1`. It is deliberately closed to the
  DSV4 production cells `k=28`, `n_sub=4`, `type_size=112`, and projection
  widths `K in {2048,4096}`. The selection is fixed per stack at model load;
  unknown values, mid-process mutation, unsupported FP8-CB cells, and
  per-expert mixed FP8-CB groups all fail instead of silently serving the
  inherited kernel. FP4 layers and source-passthrough groups are outside this
  selector. The main-extension ABI now requires the sibling symbol, so a
  pre-sibling cached `.so` is rejected before model service.

- The final-source routed-FP8 operator gate passed 17 eager/registered-op,
  CUDA-graph replay, fullgraph, and rejection tests for both decode contracts.
  The exact-artifact eager quality report has SHA256
  `013ecf0efda1a707ead44fa9f57a94a017595aff2b65dc18cf142b97e8642314`
  and produced zero full-vocabulary KL, NLL, PPL, target-logprob, generation,
  and router-route delta over 240 scored DSV4 positions. An earlier served
  A/B/A2 showed an approximately 7.2% cycle-throughput signal, but it used a
  different main binary and its generated-content and speculative-acceptance
  summaries did not match; that result remains held and is not a served-speed
  claim. A final-binary served rerun plus graph, concurrency, soak,
  long-prefill, and memory gates remain open. Exact evidence paths and binary
  identities are in `docs/RELEASING.md`.

- Routed dispatch now handles vLLM's `VLLM_MOE_SKIP_PADDING` convention at
  Gridbook's opaque MoE boundary. The documented expert-id `-1` padding
  sentinel is mapped to a valid placeholder expert and its router weight is
  replaced with exact zero using device-only tensor operations; every other
  expert id remains unchanged, caller tensors are not mutated, and the
  separately scheduled shared-expert path is untouched. Previously a fully
  padded profile batch reached Gridbook's routed-count `scatter_add_` with
  `-1` indices and raised a CUDA device-side bounds assertion, later reported
  at the next codebook-expansion API call.

- Dense CB methods now reject `is_bmm` layers during weight finalization.
  DeepSeek-V4 `attn.wo_a` is a grouped BMM whose result shape cannot be
  represented by the dense CB method; released DSpark sidecars must keep its
  three projections on the already-qualified source-FP8 W8A16 route. A future
  grouped-CB implementation needs its own numeric and performance gate.

## 0.8.5 — 2026-08-12

- **Correct source block-FP8 serving to W8A16.** The
  `fp8_e4m3_ue8m0_block128` source-passthrough wire keeps its E4M3 weight bytes
  and UE8M0 128-by-128 scale blocks resident, but no longer dynamically
  quantizes BF16 activations to MXFP8. Decode-sized `M <= 8` calls use the owned
  `fp8_source_gemv` CUDA kernel directly on BF16 activations. Larger calls use
  `fp8_source_expand_bf16` to materialize one bounded BF16 weight transient,
  consume it with Gridbook's existing owned CUTLASS grouped-BF16 bridge, and
  release it immediately. DeepSeek-V4's grouped `attn.wo_a` uses the same
  contract, with the existing bridge expressed as one problem per group. There
  is no PyTorch, cuBLAS, Triton, or persistent BF16 fallback.

- The correction is deliberately scoped to the source block-128 wire. Direct
  `mxfp8_e4m3_e8m0_g32` remains the opt-in `Mxfp8DenseLinearMethod` W8A8 lane
  behind `GRIDBOOK_MXFP8_DENSE=1`; its resident format and dynamic MXFP8
  activation quantization are unchanged. The existing v0.8.0 W8A8 measurements
  remain historical evidence for that direct lane only, not evidence for the
  new W8A16 route. Served quality and performance evidence for the W8A16 route
  is pending the pre-tag gates in `docs/RELEASING.md`; this entry makes no
  throughput or served-parity claim.

- The packaged producer/runtime contract advances from closed schema v2 to v3
  and explicitly attests
  `abi_features.source_fp8_block128_w8a16 = 1`. Producers must require this
  capability rather than infer correct activation semantics from a package
  version. Missing, malformed, or future-valued declarations fail closed.

- The distribution floor now names both serving-reachable source files that
  were absent from its literal mirrors: the existing direct-MXFP8
  `mxfp8_dense_gemm.cu` and the new `fp8_source_w8a16.cu`. The built-artifact,
  installed-wheel, metadata, and GPU pre-tag compile gates therefore fail
  explicitly if either source or either strict native ABI is missing.

- Load and dispatch now fail closed at the remaining raw-wire boundaries:
  vLLM checkpoint loaders validate the E4M3 and UE8M0 source dtypes before a
  destination `copy_` can cast them; grouped DSV4 `wo_a` admits only
  `G=8,N=1024,K=4096,tp=1` (dense serving is separately TP=1); and the native
  extension must carry the exact JIT digest, ABI schema, and live-device build
  capability. Decode and prefill publish two-phase route telemetry identifying
  `fp8_source_gemv` versus
  `fp8_source_expand_bf16+cb_bf16_grouped_mm`, with the activation contract
  recorded as preserved BF16 rather than dynamic FP8 QDQ.

## 0.8.4 — 2026-08-12

- **Attest the routed per-role LUT ABI in the packaged runtime contract.**
  Gridbook 0.8.3 implemented and tested one FP8-CB v1 codebook per routed
  `gate`, `up`, and `down` role, but its producer-facing
  `runtime_contract.json` omitted the corresponding capability marker.
  Producers that correctly require explicit consumer attestation therefore
  refused the 0.8.3 release. The contract now declares and validates
  `abi_features.routed_moe_per_role_codebook_lut = 1`; malformed, missing, or
  future-valued declarations fail closed. This additive root member advances
  the closed contract schema from v1 to v2 rather than silently changing v1.
  No kernel, artifact layout, or serving behavior changes from 0.8.3.

- The routed per-role test module now skips cleanly when the optional vLLM
  serving dependency is absent. This restores the installed-wheel CPU release
  gate; the same tests still run in the qualified vLLM environment.

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
