# Plugin reference

Operator-level reference for the out-of-tree vLLM plugin that serves the
**NVFP4-CB / FP8-CB** codebook formats. For installation and first-run problems
see [`INSTALL.md`](INSTALL.md) and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md);
for the format itself see [`SPEC.md`](SPEC.md).

The served path is Gridbook's packaged **native CUDA/CUTLASS kernel set** under
`gridbook/csrc/`. Native CUDA handles decode GEMV, codebook/source expansion,
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

- **INV-1 (honored):** a CB resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) scales; a source block-FP8 resident
  weight is its raw E4M3 tensor + UE8M0 scale blocks. A dense BF16 `[N,K]`
  weight is never materialized in HBM as a *model-wide* tensor — each tile is
  expanded inside the kernel, in registers, then consumed by the matmul.
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
`gridbook.runtime_contract.load_runtime_contract()`. Closed schema v10 also
attests `abi_features.source_fp8_block128_w8a16 = 1`,
`abi_features.dspark_construction_physical_bridge = 1`, and a per-unit
tensor-parallel capability table, so a producer can require the BF16-activation
source route, the complete DSpark sidecar loader
and construction/physical namespace ABI (including weight-only drafts), or a
specific tensor-parallel size for a specific format without inferring
semantics from the package version. It
declares accepted quantization names, serialized packing/type-size
rules, accepted CB reader rungs, canonical producer rungs, exact-platform lane
eligibility, and producer-profile loader coverage. The plugin derives its own
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
  implemented. **Signed** mode (S-rungs, n_sub=1) was validated by a
  historical 18-test GPU battery on 2026-07-22 and closed as research-only by
  the same-day K-vs-S head-to-head: over 776 matched-rate per-(Linear, rung)
  comparisons the unsigned rungs won 79% of the time and the allocator placed
  6 signed units against 147 unsigned. The producer deleted the family on
  2026-08-17 ("not performant, we don't support them"); on 2026-08-23 the
  runtime followed — decode kernels, Python admission, tests, the packaged
  contract row, and the SPEC definition are gone, and a `"mode": "signed"`
  scheme now fails closed everywhere. No published artifact encodes an
  S-rung. Full mode: spec-reserved, unimplemented.
- **FP8-CB v10 reader and producer domains are intentionally different.**
  `formats[FP8_CB_K].rungs` accepts K4/K8/K12/K16/K20/K24 and every integer
  K28..K48, retaining older irregular artifacts. `producer_rungs` is the
  hardware-aligned K4..K48 step-4 menu. A producer MUST choose from the latter;
  a reader MUST continue to admit the former. NVFP4's two fields both remain
  K12..K24. `type_size` is still exactly `4*k` for FP8, so no legacy byte
  interpretation changed.
- **Mixed containers are supported and shipping.** A config group carrying a
  `"scheme"` key is a CB group and is served by this plugin; a group without one
  uses the stock `compressed-tensors` vocabulary and is delegated to a real
  `CompressedTensorsConfig` that the plugin constructs (NVFP4, FP8_DYNAMIC);
  `ignore` entries become BF16 passthrough. The shipped 295B artifact serves 36
  vanilla-FP8 Linears this way, and the 27B's vision tower is a stock NVFP4
  W4A16 group. **Consequence:** an artifact's hardware requirements are the union
  of gridbook's and those of its delegated groups.
- **Tensor parallel: dense CB and FP8 source-passthrough Linears, above one
  rank.** Since the
  shard-aware loading wave (2026-08-23), dense CB Linears load correctly at
  TP>1 with no change to any exported byte: vLLM's stock declared-dim
  narrowing slices whole packed rows on column-parallel layers and
  superblock-aligned byte windows on row-parallel layers; codebooks and
  compose tables replicate per rank from the immutable sidecar; and merged
  GDN-style roles recover rank-local boundaries. A shard that would split a
  CB group is refused at weight construction as a structured
  `ShardGroupAlignmentError` (qname / axis / group_size / tp_degree /
  shard_size fields — input axis requires whole 256-weight superblocks;
  output axis requires the native kernel's 8-wide (fp4) or 16-wide (fp8) row
  quantum). Everything else keeps refusing at construction, naming itself:
  delegated compressed-tensors groups and quantized embedding units; routed
  CB MoE expert stacks refuse on this axis and serve on the expert-parallel
  one instead (see below).
  `fp8_e4m3_ue8m0_block128` source-passthrough units are no longer among the
  refusals: dense ones shard when every rank's extent on the sharded axis is
  a whole multiple of the 128-element source block (per fused role on a
  merged plane, because vLLM narrows the UE8M0 scale plane over the block
  grid by ceil division), and grouped-BMM ones shard only at a degree whose
  divided group count was measured on the release device — 1, 2 and 4 at the
  DSv4 `wo_a` geometry; an unmeasured degree is refused even when perfectly
  aligned. Ignored (BF16) Linears stay on vLLM-native sharding. Dense
  TP>1 was measured two-node on 2026-08-23 (the DeepSeek-V4 92 GB CB
  artifact across two DGX Sparks, KL within the CB kernel's nondeterminism
  floor); nothing here claims a decode win for CB artifacts. The fact is
  published, not prose: every serving unit carries a `tensor_parallel` row
  in the contract (see [Tensor-parallel capability](#tensor-parallel-capability)).
  `--enforce-eager` is the published-model configuration; mode-0
  `FULL_DECODE_ONLY` is also capture-correct with the permanent opaque dispatch
  and is being promoted through the model-size performance gates
  ([details](TROUBLESHOOTING.md#do-i-really-need---enforce-eager)).
- **MoE expert stacks: expert parallel, not tensor parallel.** A CB expert
  stack's last dimension is superblock bytes, not input columns, so vLLM's
  tensor-parallel intermediate split would cut a packed superblock and there
  is no partial-superblock decode. Serve routed CB MoE above one rank with
  `-tp N --enable-expert-parallel`, which shards the EXPERT axis instead: each
  rank holds a disjoint subset of whole experts, whole superblocks, and
  per-expert numerics byte-identical to a single-rank serve. `-tp N` **without**
  `--enable-expert-parallel` refuses at method construction, naming the flag.
  So do the expert-parallel topologies whose premise Gridbook has not
  established — data-, pipeline-context- and sequence-parallel EP (vLLM
  switches those to all2all dispatch/combine kernels, which expect a MoE
  method that exchanges tokens), EPLB, and `skip_final_all_reduce` (Gridbook
  returns this rank's partial and relies on vLLM's stock final all-reduce).
  Mixed per-expert-format stacks also stay refused: their format partition is
  declared over global expert ids. The capability is published per unit in the
  contract's `expert_parallel` section (see
  [Expert-parallel capability](#expert-parallel-capability)). Like dense TP>1,
  this was measured two-node on 2026-08-23 (43 of 43 DeepSeek-V4 MoE layers
  admitted under EP, 128 of 256 experts per rank); nothing here claims a
  throughput result.
- **BF16 activations only** — the shipping CUDA dense and grouped-MoE bindings
  require BF16. Gridbook no longer advertises FP16 to vLLM and therefore fails
  at dtype validation instead of crashing or changing dtype at a dispatch
  crossover.
- **Biasless public CB dense Linear only** — a non-`None` bias passed to the
  public dense CB method is rejected. The permanent opaque native operation
  does not expose a Gridbook-owned biased kernel, so 0.5 does not hide a
  framework bias add behind the serving boundary. Delegated non-CB groups keep
  their upstream method's contract.

## Capability-scoped serving lanes

Contract v10 adds `lane_eligibility` schema
`gridbook.lane-eligibility.v2`. It does not widen the byte reader domain; it
states which exact platform/structure/regime/rung compositions have a native
route and how far their qualification has progressed.

- `platforms` maps a scalar id such as `sm_89` to one exact compute capability,
  never a `>=` family or wildcard.
- Every `cells` row names one scalar platform, format family, structure
  (`dense` or `routed_moe`), regime (`decode` or `batch`), non-empty subset of
  that family's `producer_rungs`, route status, qualification, serve flags and
  closed predicates. Absence is unbacked; an explicit `unbacked` row is invalid.
- `route_status` is `backed`, `backed_with_serve_flag`, or `fallback`.
  `qualification` is independent: `compile_only` proves no device execution
  and is never producer-legal; `device_qualified` requires the physical gate.
- The initial SM89 dense FP8-CB decode (`cb_gemv_fp8`) and batch
  (`cb_expand_fp8` + direct vLLM CUTLASS W8A8) rows cover the producer ladder
  but remain `compile_only`. This contract therefore does **not** authorize a
  4090 artifact yet.
- `python -m gridbook.sm89_preflight --build-directory <dedicated-dir>
  --receipt <receipt.json>`
  cross-compiles the production generic module with explicit
  `compute_89/sm_89` code, verifies its full symbol surface, and checks the
  direct vLLM FP8 quantization/CUTLASS ABI without invoking an operator. It
  never uses the production extension cache. The emitted JSON says
  `qualification_ceiling: compile_only` and explicitly excludes device,
  performance and graph claims.
- Gridbook intentionally carries no torch.compile or CUDA-graph configuration
  or attestation in this table. The serving producer owns its immutable graph
  requirement and references the per-run endpoint receipt that actually proved
  it; source compilation cannot establish graph correctness.

## Tensor-parallel capability

As of contract schema `gridbook.runtime-contract.v10`, the packaged contract
carries a `tensor_parallel` section that publishes, per serving unit, what the
runtime actually enforces. The table is an attestation: each row restates a
refusal or admission site in this package, and
`tests/test_runtime_contract_tp.py` checks every row against the source text
of that site.

- `axis` names the measured quantity: vLLM's live tensor-parallel world size,
  read from the running worker at model construction — not a CLI argument.
- There is no whole-model cap. The v5 blanket pre-dispatch gate is gone, so no
  single number is true of every dispatch path; publishing one would assert
  more than any site enforces.
- Each entry in `units` names one producer-addressable unit: a CB format family
  (`NVFP4_CB_K`, `FP8_CB_K`), one source-passthrough format id, or the single
  composite surface `mixed_fused_projection`.
  Three claim shapes exist, matching the three enforcement shapes in the
  runtime:
  - **Dense CB families** publish `shard_admission` instead of a cap, because
    no dispatch path enforces a numeric ceiling for them — above one rank,
    admission IS the laws, evaluated per rank at weight construction and
    raised as `ShardGroupAlignmentError`: `input_axis_group` (256; a
    row-parallel K-shard must contain whole packed superblocks),
    `output_axis_quantum` (8 fp4 / 16 fp8; a column-parallel logical shard
    must not cut the native kernel row quantum), and `merged_roles`
    (`even_division`; merged checkpoint roles divide across ranks). A shard
    that violates them is refused by the runtime regardless of the table; the
    row exists so a producer can pre-check the same laws.
  - **Source-passthrough formats** publish a numeric `max_world_size` of 1
    unless their lane enforces shard laws of its own. A unit whose method
    branches on an execution arm additionally carries per-arm rows, and each
    arm carries either `max_world_size` or `shard_admission`, never both nor
    neither. A unit with a **law-admitted** arm carries no unit-level number
    at all, because one scalar cannot cover a law-admitted arm and a capped
    arm at once; an armed unit whose every arm is capped keeps its unit cap of
    1, which its dispatch gate does enforce. Where a unit-level number is
    present it gates first, before the arm row is consulted.
  - **The one law-admitted passthrough arm** (v7) is
    `fp8_e4m3_ue8m0_block128` / `dense`. Its `shard_admission` publishes
    `input_axis_group` 128, `output_axis_quantum` 128 and `merged_roles`
    `per_role_group_multiple`: the lane derives each Linear's shard degree
    from its own `create_weights` arguments — never from `layer.tp_size`,
    which vLLM stamps onto replicated layers too — and refuses, before any
    parameter exists, any shard whose per-rank extent on the sharded axis is
    not a whole multiple of the 128-element source block. That is exactly the
    condition under which vLLM's `BlockQuantScaleParameter` narrow (start
    `rank * ceil(local / 128)`) indexes the UE8M0 block grid correctly; below
    it, ranks read shifted scale blocks with no error. On a merged plane the
    law applies per fused role, because the block offsets are converted role
    by role. The refusal is `ShardAlignmentError` (a `ValueError`).
  - **The same unit's `bmm` arm publishes the same law plus a closed list of
    measured shard degrees** (`qualified_shard_degrees` `[1, 2, 4]`) and pins
    the grouped geometry it qualifies (`bmm_groups` 8, `rows_per_group` 1024,
    `k` 4096 — the UNSHARDED plane). Column-sharding a grouped plane divides
    the kernel's group count (G 8 -> 4 at TP=2), so alignment alone cannot
    admit it: each degree is its own measurement. Degrees 1, 2 and 4 were
    measured on the release device on 2026-08-23 (every rank's call bitwise
    equal to the corresponding columns of the unsharded G=8 call); a degree
    outside the list is refused, however aligned the plane is.
  - **The composite** (v9) is `mixed_fused_projection`: one vLLM
    `MergedColumnParallelLinear` whose roles have DIFFERENT Gridbook formats,
    such as a CB `gate_proj` fused with a block-FP8 source-passthrough
    `up_proj`. It is not a format and owns no alignment law, so it publishes
    `shard_admission` `{axes: ["output"], per_role_law: "inherited"}` and no
    number. `axes` is the one law it does own: a merged projection is
    column-parallel, and a row-parallel split of it is refused by name.
    `per_role_law` `inherited` says where admission actually lives — the
    composer derives the column degree from vLLM's own `create_weights`
    arguments and builds each role's carrier at that ROLE's whole-tensor
    output size, so each role's row above is what a producer pre-checks, role
    by role, against the rank-local width that role will receive. Publishing a
    cap here would assert a ceiling on every role at once that no site
    enforces.
- `semantics` is `closed_world`. A consumer must read the table with the same
  rule the validator enforces for publishers:

  Serving unit *U* at world size *t* is permitted only if the table contains
  exactly one row named *U* and the exact arm row when the unit has arms; a
  numeric claim must cover *t*, and any pinned geometry must match exactly;
  a row (or arm) that publishes `shard_admission` defers to those laws at
  weight construction and carries no number to compare against — except that
  a `shard_admission` carrying `qualified_shard_degrees` admits *t* only when
  *t* is in that list. Every
  other outcome — no row, two rows, an unknown arm name, a larger *t* than a
  numeric claim, a different geometry — is a refusal. There is no default,
  wildcard, or inheritance.

The completeness direction is enforced too: the packaged validator rejects any
contract that omits a shipped unit, invents one, drops a mandatory field,
caps the capless dense CB surface with a number, or publishes a numeric claim
no enforcement site stands behind.

**Compatibility rule:** `schema` and `contract_version` move together (v10 / 10),
and readers match the schema string exactly. A producer pinned to the previous
schema (v9) must refuse a `gridbook.runtime-contract.v10` contract whole — no
partial parsing, no field-by-field salvage across versions — and keep producing
against its pinned runtime until its pin is deliberately bumped. Reading a v9
contract with a v10 reader fails the same way. Only the CURRENT schema string is
spelled in full anywhere outside the changelog, so a stale pin cannot hide in
prose.

## Expert-parallel capability

`expert_parallel` is the second axis, published beside `tensor_parallel` and
read the same way. Tensor parallelism splits one unit's rows and columns;
expert parallelism splits a routed MoE layer's experts and leaves every expert
whole. CB expert stacks serve on the second axis and never on the first, so
they carry rows here and none there. `tests/test_runtime_contract_ep.py`
checks every row against the source text of its enforcement site.

- `axis` is vLLM's expert-parallel size, and `semantics` is `closed_world` —
  same reading rule as above: exactly one row named *U*, or refusal.
- `requires` is the TOPOLOGY predicate, which is what makes this axis
  different: expert parallelism only keeps whole experts if no other parallel
  axis is splitting the layer too. Each field is one branch of
  `config.py::_require_ep_moe_serving` — `vllm_flag`
  (`--enable-expert-parallel`), `moe_tensor_parallel_size` 1,
  `all2all_kernels` false (vLLM's own derivation covering dp / pcp /
  sequence-parallel EP), `expert_load_balancing` false, and
  `skip_final_all_reduce` false.
- `cb_moe_expert_stack` rows (`NVFP4_CB_K`, `FP8_CB_K`) publish
  `expert_admission` laws rather than a numeric cap, for the same reason the
  dense CB rows do: `shard_axis` `expert`; `sharded_dims` `none` (a rank's
  expert bytes are byte-identical to the corresponding slice of the
  world-size-1 stack); `placement` `monotone_bijection` (the loaders refuse
  any other `expert_map`); `checkpoint_leading_dim` `global_expert_count`
  (both loaders gather a whole-stack checkpoint tensor down to this rank);
  `remote_pair_handling` `zero_weight_alias`; and `cross_rank_reduction`
  `vllm_final_all_reduce` — the reduction is vLLM's, not Gridbook's.
- `cb_moe_expert_stack_refused` rows publish an explicit `max_world_size` so a
  consumer can tell *refused* from *unknown*. Today that is
  `cb_moe_per_expert_format_groups`, capped at 1.

**Measured 2026-08-23:** the two-node gate — a real
`-tp 2 --enable-expert-parallel` serve of the DeepSeek-V4 92 GB CB artifact
across two DGX Sparks (Ray, dual 200G RoCE), with greedy generation and a
same-corpus KL check against two single-box serves — has run: every MoE
layer admitted, KL 0.0505–0.0537 at 94.5–95.8% top-1 agreement over 746
BF16-confident positions, inside the CB kernel's own nondeterminism floor
(A/A 0.0405–0.0483). Capacity is established; no throughput result is
claimed.

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
  its MTP, DeepSeek-V4) whose loaders map MoE experts at the top level and would
  otherwise not
  recognise stacked codebook expert tensors. HunYuan-V3-style shared-CB targets
  are aliased to the native CB Linear at construction, including the collapsed
  and nested MTP prefix forms. If the checkpoint's CB tensor can resolve only
  to a plain BF16 parameter, load fails with the target name; the retired
  decode-to-BF16 compatibility path is not available. The wrap is inert for
  non-CB checkpoints.

### Registered architectures

Both lists live in `gridbook/runtime_contract.json` and nowhere else;
`plugin.py` derives its install set from the second one. A `top_level_loader_modules`
entry names the module that **defines** the entrypoint class, which for a
per-platform vLLM package is a submodule rather than the package itself.

| Producer profile | vLLM entrypoint class | Loader module in the contract |
|---|---|---|
| `hy_v3` | `HYV3ForCausalLM`, `HYV3MTP` | `vllm.model_executor.models.hy_v3`, `…hy_v3_mtp` |
| `laguna` | `LagunaForCausalLM` | `vllm.model_executor.models.laguna` |
| `qwen3_5`, `qwen3_5_dense`, `qwen3` | Qwen3.5-MoE `ForCausalLM` / `ForConditionalGeneration` + MTP | `vllm.model_executor.models.qwen3_5`, `…qwen3_5_mtp`, `…lfm2_moe` |
| `deepseek_v4` | `DeepseekV4ForCausalLM`, `DSparkDeepseekV4ForCausalLM` | `vllm.models.deepseek_v4.nvidia.model`, `…nvidia.dspark` |

#### DeepSeek-V4 (`deepseek_v4`)

The current passing integration contract is the immutable eugr Spark image
`eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869`:
vLLM `0.26.1rc1.dev693+g7f7a32cfe.d20260812` at commit
`7f7a32cfec0f1bc5b73c37200b86631523a1ea8f`, torch `2.13.0+cu130`, and
FlashInfer `0.6.18` at commit
`9ffd99510d92b883f154fc9f2e3d5aac93e231ca`. The startup preflight must find
the native `(head_dim=64, topk=256)` entry in FlashInfer's SM120 DSV4 decode
dispatch and `VLLM_MOE_SKIP_PADDING=True`; Gridbook makes the resulting `-1`
routed-padding sentinel inert at its opaque MoE boundary. This contract uses
the released DSV4-Flash config (43 layers, all MoE, 256 routed experts + 1
shared, top-6, MLA with `q_lora_rank`/`o_lora_rank` 1024 and `o_groups` 8), one
GB10, and `tp=1`. The FlashInfer SM120 MLA backend **requires
`--kv-cache-dtype fp8`** — `auto` aborts model construction, independently of
Gridbook. Older EUGR/vLLM/FlashInfer combinations are not interchangeable with
this qualified tuple merely because they expose the same Python classes.

- **Native CB.** The routed expert stacks
  (`…ffn.experts.routed_experts.w13/w2`), the MLA projections `attn.wq_a` +
  `attn.wkv` (merged into `fused_wqa_wkv`), `attn.wq_b`, `attn.wo_b`,
  `indexer.wq_b`, and the shared expert's `w1`/`w3`/`w2`. Note the DSV4
  spellings: module attributes are `attn`/`ffn` (not `self_attn`/`mlp`), the
  routed stack nests under `routed_experts`, and shared-expert shards use the
  Mixtral `w1`/`w2`/`w3` convention while vLLM's merged leaf is `gate_up_proj`.
  The class publishes **no** `packed_modules_mapping`, so Gridbook's fused
  fallback tables supply the merge.
- **Independently owned dense fusions.** A fused output is not a single-format
  or single-metadata allocation unit. When source siblings have different
  CB/source encodings—or matching schemes but different physical activation
  scalars—Gridbook constructs private role carriers, runs each role's existing
  method unchanged, and concatenates results in `stacked_params_mapping` order.
  The top-level loader validates all packed/scalar planes before committing the
  composite. Truly compatible same-scheme/same-scalar CB roles and same-wire
  source roles retain vLLM's existing merged method. Composite dispatch is
  authorized only while a model class with mixed-fused loader ABI 1 is being
  constructed; an unwrapped class fails with that loader gap named. The ABI is
  scoped with a task-local construction gate, so target-model authority cannot
  leak into an MTP/draft constructor. No persistent common weight or
  cross-format requantization is created. The 2026-08-08 DSV4 artifact audit
  covers all 86 dense merges: 13 differ by format and 54 by contracted scalar
  (67 composite); 19 are physically compatible and retain the merged fast path.
- **Namespace.** The checkpoint has no `model.` component (keys start at
  `layers.N.`); the class re-attaches it in its own `hf_to_vllm_mapper`.
  Source-passthrough declarations use the producer/Transformers live spelling
  (`self_attn`/`mlp`, gate/up/down), while this vLLM class constructs
  `attn`/`ffn` and loads `w1`/`w3`/`w2`; config resolution bridges those exact
  structural aliases. Both the expert loader and config-side target resolution
  handle either outer namespace spelling.
- **MTP / DSpark is isolated from the target body.** The target artifact can
  preserve `mtp.*` (4,705 tensors across the three DSpark stages `mtp.0/1/2`).
  `DeepseekV4ForCausalLM.load_weights` still builds
  `AutoWeightsLoader(self, skip_substrs=["mtp."])`, so every one is dropped at
  body-only serving time and the target-body path is unchanged. Under DSpark
  speculative serving, the separate `DSparkDeepseekV4ForCausalLM` class in
  `vllm.models.deepseek_v4.nvidia.dspark` consumes that payload.
- **Target-only is the 0.8.6 shipping default; DSpark is experimental.** Omit
  `--speculative-config` for the release launch. On the exact runtime above,
  the same target artifact, eager 12,288-token configuration, and fixed
  sequential 8-prompt x 128-output-token suite measured `10.2389` output tok/s
  without MTP. The native MXFP4 draft measured `10.3364` tok/s (`+0.95%`) but
  raised model residency from `95.12` to `105.25` GiB; its acceptance was
  `42.289%` overall, `78.916%` at position zero, and `2.114` accepted
  speculative tokens per draft cycle. A K12-CB hybrid draft measured only
  `9.1319` tok/s (`-10.81%`). Those results prove that the loader can serve a
  coherent DSpark draft, but neither draft earns production-default status or
  justifies its memory cost. They are not graph-mode, concurrency, or 128k
  context claims.
- **Three DSpark namespaces stay distinct.** Decoder layers are constructed
  with quantization prefixes `model.layers.43/44/45` (more generally,
  `num_hidden_layers + stage`), registered by the three-element `ModuleList` as
  `model.layers.0/1/2`, and stored in the checkpoint as `mtp.0/1/2`. A DSpark
  Gridbook quantization config therefore declares the construction prefixes.
  At load time the wrapper calls the model's own callable
  `_remap_dspark_name` to obtain the registered prefix before running the
  existing mixed-fused and stacked-expert resolvers. It does not reproduce or
  guess that mapping. Mixed-fusion carriers retain their construction prefix;
  the transaction router derives the corresponding registered role prefix from
  the carrier's actual `named_parameters()` path and rejects inconsistent
  metadata.
- **Contracted activation scalars are explicitly bridged.** Contracted FP4-CB
  config targets remain in the construction namespace while serialized
  `input_global_scale` tensors and their digest-bound contract names remain
  physical `mtp.*`. `dspark_target_bridge` declares that one-to-one mapping
  for the complete activation contract (including a delegated stock-NVFP4
  target) with `L`, stage count, and same-tail validation. Gridbook accepts no
  inferred layer offset, undeclared construction target, incomplete map, or
  physical target outside the activation contract; delegated source-format
  `main_proj` is never included. Before consuming the
  checkpoint stream, the loader also compares stamped `L`/stage count with
  `config.num_hidden_layers` and the instantiated model's
  `num_dspark_layers`.
- **Stock ownership is preserved.** Stacked Gridbook expert planes and
  independently owned mixed-fusion planes are intercepted exactly as for the
  target body. Every other tensor is delegated under its original checkpoint
  name; draft tensors therefore retain their physical `mtp.*` spelling. This
  includes ordinary direct/fused dense CB planes, source-format tensors,
  model-level heads, non-MTP tensors, and the unwired confidence head. DSpark's
  stock loader remains their sole mapping/filtering authority. The
  draft reuses the existing Gridbook Linear/MoE methods and CUDA kernels; this
  loader registration adds no format or kernel.
- **Draft sidecars have draft authority.** vLLM intentionally leaves the
  target `model_config` installed while constructing the separate DSpark
  model. Gridbook scopes the explicit
  `speculative_config.draft_model_config` to that exact DSpark constructor, so
  its pointer `quant_config.json` and `cb_codebooks.pqcb` resolve from the
  draft rather than `/model`. Merely enabling speculation never redirects the
  target body, and absent or malformed draft authority fails before the
  constructor consumes weights.
- **Grouped `wo_a` is source W8A16, not dense CB.** The three DSpark
  `attn.wo_a` projections are grouped BMMs. The generic dense CB method has no
  grouped output contract and therefore rejects `is_bmm` at load instead of
  returning a shape-incorrect `[T,G,G*N]` result. These projections retain
  their E4M3/UE8M0 source planes and use the qualified grouped W8A16 adapter;
  the other eligible draft projections may use CB.

- **Evidence and provenance.** The DSpark figures above are pre-release
  integration/performance classification, not clean-tag provenance. Public
  release documentation does not treat operator-local paths or a rehearsal
  wheel identity as durable evidence. The clean committed wheel and derived
  image must satisfy the target-only 256k gate in
  [`RELEASING.md`](RELEASING.md); DSpark keeps its separate experimental gate.
- **Passed through unquantized.** Hyper-connection parameters (`hc_mult` 4:
  `hc_head_*`, `layers.N.hc_attn_*`, `layers.N.hc_ffn_*`), the hash-routing
  tables (`num_hash_layers` 3: `ffn.gate.tid2eid`), `ffn.gate` and its
  `e_score_correction_bias`, `compressor.ape`, `attn.attn_sink`, and all norms.
  vLLM builds these as raw parameters or with `quant_config=None`, so
  `get_quant_method` is never called for them.
- **Must not be CB-encoded.** `ffn.gate`, both
  `compressor.fused_wkv_wgate` modules, `indexer.weights_proj`, `lm_head` and
  `embed_tokens` have no quant config at all. `attn.wo_a` does have a quant
  config, but it remains source block-FP8 rather than CB: the qualified eugr
  vLLM baseline's grouped output projection bypasses `apply()` and reads
  `.weight` plus its scale parameter for DeepGEMM directly. On sm_121 that
  DeepGEMM scale transform is unsupported. Gridbook therefore keeps the raw
  E4M3 weight and block-128 UE8M0 scale resident and installs a narrow,
  ABI-guarded DSV4 adapter only on marked `wo_a` modules: native inverse RoPE,
  the Gridbook-owned grouped W8A16 method on unchanged BF16 activations, then
  `wo_b`. Every
  unmarked stock DSV4 layer continues through vLLM's original `_o_proj` method
  unchanged. Exact artifact geometry `(G=8, N=1024, K=4096)` is
  covered by the exact-artifact W8A16 gates in `RELEASING.md`. Historical
  grouped-MXFP8 W8A8 numbers are not evidence for this route. End-to-end served
  parity and performance remain release gates.
- **Fail-closed.** Outside `mtp.*`, vLLM's DSV4 loader looks parameters up
  unguarded, so any tensor the artifact emits with no matching parameter is a
  hard load failure rather than a silent skip. On the Gridbook side the usual
  three apply: a CB tensor that resolves only to a plain BF16 Linear stops the
  load, a shape mismatch against the stacked `(E, out, bytes)` contract raises,
  and a registered-but-never-filled expert stack is caught by `cb_fill_guard`.

## CUDA kernel set (decode-GEMV + prefill)

JIT-built native CUDA/CUTLASS extensions. For CB lanes, the numerics contract is
identical weight rounding to the reference decode
(`w = bf16_rn(codebook · scale)`), fp8/fp4 activation QDQ bit-exact to the codec,
and **fp32 accumulation**. The source block-FP8 W8A16 lane instead preserves its
BF16 activation tensor and decodes the raw E4M3/UE8M0 weight directly. CUDA
results are held against independent references with the operation-specific
exactness/tolerance gate; summation reassociation remains the only admitted
GEMM/GEMV difference.

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
- **Routed FP8-CB whole-row sibling** `cb_moe_gemv_fp8_v2`: an independent
  alternative selected by `PRISMAQUANT_CB_FP8_GEMV_V2` (default `auto` since
  0.8.9: engaged exactly on the qualified K28 cell, inherited elsewhere;
  `1` is the strict all-stacks A/B-arm spelling). It stages one complete
  packed weight row and
  the exact 1 KiB K28 E4M3 LUT in shared memory, then uses named virtual
  accumulators to preserve the inherited kernel's eight-chain reduction under
  both decode contracts. Its release surface is exactly
  `k=28/n_sub=4/type_size=112` at K=2048 or K=4096. The choice is fixed per
  `w13`/`w2` stack at model load; an opted-in unsupported uniform FP8 cell or
  any per-expert mixed FP8-CB group fails instead of silently serving the
  inherited kernel. FP4-CB and source-passthrough groups are outside this
  selector. The eager exact-artifact quality gate produced zero full-vocabulary
  and router-route delta over 240 positions; a prior approximately 7.2% served
  cycle-throughput signal used a different main binary and failed cross-arm
  content/acceptance integrity, so the selector remains off pending the
  final-binary served rerun and graph, concurrency, soak, long-prefill, and
  memory gates. See the exact evidence and rerun contract in
  [RELEASING](RELEASING.md#routed-fp8-cb-whole-row-dsv4-quality-and-served-gate).
- **`cb_expand_fp8`** — transient prefill expander: decodes the packed stream to
  a dense `[N,K]` e4m3 tile for CUTLASS W8A8. This is the **large-M shipping
  answer** today (decode paid once), at the cost of materialising `[N,K]` in HBM
  (an INV-1 compromise the persistent-N work below aims to remove).
- `fp8_act_qdq`, `cb_moe_combine` — fused per-token fp8 QDQ and the deterministic
  expert-ascending bf16 combine.

**`gridbook/csrc/fp8_source_w8a16.cu`**
(`cuda_ext.get_fp8_source_w8a16_ext()`) — source block-FP8 W8A16:

- `fp8_source_gemv` consumes BF16 activations and resident E4M3 weights with
  their original 128-by-128 UE8M0 scale blocks for `M<=8`. It accumulates in
  FP32 and emits BF16; no activation-QDQ operator is called.
- `fp8_source_expand_bf16` creates the one-layer BF16 weight transient used for
  `M>8`. The transient feeds the existing owned grouped-BF16 CUTLASS bridge and
  is released immediately. DSV4 BMM uses one bridge problem per group rather
  than a Python or framework matmul loop.
- The extension has a strict two-symbol ABI and is required during weight load.
  It is a separate digest-keyed JIT family so its source/toolchain identity
  cannot alias the CB main module or direct-MXFP8 module.

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
- **`cb_gemv_v2`** is the smem-resident-dictionary decode schedule selected
  by `PRISMAQUANT_CB_GEMV` (default `auto` since 0.8.9, engaged only where
  the compiled occupancy predicate says it wins; `inherited` is the kill
  switch). Every FP4-v2 serve loads and prepares this extension regardless,
  because it needs `cb_expand_v2` for quality prefill. On DSV4 K=2048/4096, and only with all
  `PRISMAQUANT_CB_W2_*` overrides absent, its virtual-warp specialization is
  bit-exact to the inherited default for the k12/k16/k18 release surface;
  other widths keep rowpack reduction order and may reassociate.

**`gridbook/csrc/cb_fused_gemm.cu`** (CUTLASS, separate JIT ext) — prefill:

- **`cb_fused_prefill_mm`** — decode-in-prologue fused GEMM (CUTLASS sm120
  collective: decode each B superblock into smem, then the FP8/FP4 tensor-core
  MMA — INV-1 **and** INV-2, bit-exact vs the passthrough `fork64`). Wins the
  measured mid-M points (1.04–1.45× at M=32/64/128). Production FP8-CB dispatch
  considers it for **M=9–128** when its rung/layout/device predicates hold; at
  large M every M-tile CTA re-decodes B, so transient-expand is preferred. Its
  extension availability and selector mode are resolved during model load, not
  on the first eligible forward; changing the relevant environment value after
  load raises instead of silently changing residency or dispatch. Unset means
  auto: only a cc 12.0/12.1 loading device resolves this specialization on.
  SM89 never queries its Blackwell-only JIT and uses CUDA expansion + native
  W8A8 CUTLASS above M=8; explicitly setting the selector to `1` there refuses
  the model load rather than substituting that route behind the request.

**`gridbook/csrc/cb_bf16_grouped_gemm.cu`** (CUTLASS, separate JIT ext) — the
quality-preserving BF16 bridge:

- **`cb_bf16_grouped_mm`** — consumes ragged expert segments and transiently
  expanded BF16 weights in one owned CUTLASS grouped GEMM. MoE prefill uses
  `E>1`; dense FP4-CB and dense source block-FP8 use the same binding with `E=1`
  for every `M>8`. Grouped DSV4 source block-FP8 uses `E=G` with equal `M`-row
  segments. This
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
| `PRISMAQUANT_CB_EXT_DIR` | `~/.cache/prismaquant-cb-ext` | Root of the JIT build cache. Point it at a persistent, writable directory in containers so a cold start does not rebuild every module it reaches. Each of the nine modules owns a SUBDIRECTORY of it — `main`, `v2`, and the identity-keyed `bf16_grouped/<digest>`, `fused/<digest>`, `fused_fp4/<digest>`, `fused_fp4v2/<digest>`, `moe_persistent_b/<digest>`, `mxfp8_dense/<digest>`, `fp8_source_w8a16/<digest>` — so no two ninja workspaces share artefacts and a changed source, header, lane macro, target or toolchain ABI lands in a new directory instead of serving a stale kernel. Upgrading Gridbook therefore costs ONE rebuild per affected module; the old directories are inert and can be deleted. |
| `PRISMAQUANT_CUTLASS_INCLUDE` | unset (vLLM's bundled copy) | The `include` directory — the one holding `cutlass/cutlass.h` — that all **four** CUTLASS-compiling modules build against: grouped-BF16, fused FP8-CB, fused NVFP4-CB and the fused FP4-CB v2 mid-M lane. Set it to build in a venv with no vLLM wheel, or against a CUTLASS newer than the bundled tree. Unset, the path is discovered under `vllm/third_party/` *without importing vLLM* (importing it merely to locate files would eagerly initialize optional compiler backends, Triton among them). A set-but-wrong value **fails** with the missing header named; it never falls back silently, because compiling against a different CUTLASS than the one asked for is the surprise this override exists to prevent. |
| `PRISMAQUANT_CB_GEMV` | `auto` (0.8.9) | Which grouped FP4-CB **decode GEMV** serves a layer: `auto` \| `v2` \| `inherited`. **Unset means `auto` since 0.8.9**: the smem-resident v2 decode GEMV is used only where the compiled occupancy predicate (`cb_gemv_v2_prefers_inherited`, the arithmetic of the binary that launches) says it wins, and an unavailable extension degrades LOUDLY to the inherited kernel — the inherited path is exactly the pre-0.8.9 default, so degrading is correct, only slower. On DSV4 K=2048/4096 with every `PRISMAQUANT_CB_W2_*` override absent, the v2 selection is bit-exact to the inherited default for k12/k16/k18; other widths retain rowpack reduction order and may reassociate. `inherited` is the kill switch and A/B control: it never probes or builds the v2 module and reproduces the pre-0.8.9 dispatch exactly. All FP4-v2 quality paths build/load `cb_gemv_v2.cu` regardless of this selector because that module also owns the required exact expander; FP8-only serves do not need it. An unknown spelling raises; changing it mid-process raises. |
| `PRISMAQUANT_CB_FP8_GEMV_V2` | `auto` (0.8.9) | Routed FP8-CB whole-row decode GEMV sibling: `auto` \| `1`/`require` \| `0`/`off`. **Unset means `auto` since 0.8.9**: exactly the qualified cell — `k=28/n_sub=4/type_size=112` at K=2048/4096, the cell the bit-exact quality gate and the served final-binary decode run measured — takes the whole-row sibling, and every other routed FP8-CB stack keeps the inherited grouped FP8 kernel with the reason logged per stack. `1` keeps its pre-0.8.9 meaning exactly: an A/B-arm REQUIREMENT under which any off-cell routed FP8-CB stack (or per-expert mixed FP8-CB group) fails the model load, so a candidate arm can never silently mix kernels. Value is process-stable; unknown spellings and mid-process changes raise. The main extension's strict ABI rejects a stale `.so` without `cb_moe_gemv_fp8_v2` even when this selector is off. Evidence: exact eager quality is zero-delta over 240 scored positions (both decode contracts); the final-binary served rerun (B-v3, 2026-08-14, cudagraph FULL_DECODE_ONLY, route census, host quiesced) measured **+8.62 % decode throughput** at DSV4 shapes with content/acceptance integrity clean — the earlier held ~7.2 % signal's cross-arm integrity failure was traced to host interference, not the kernel. Soak and high-concurrency gates have not run as named protocols; the sibling serves only the routed `M ≤ 16` GEMV band, so long-prefill scheduling is outside its reach. |
| `PRISMAQUANT_CB_FUSED_FP4` | off | Dense fp4-CB native-FP4 prefill opt-in. `1`/`midm` use a fully attested artifact scalar for all prefill shapes / `16 < M <= 128`. `static_lsq`/`static_lsq_midm` keep that exact `G` and the native E2M1/SFA payload, but fit the existing per-row EVT residual by least squares; they add no model metadata, weight copy, decoder, or GEMM. `rowwise`/`rowwise_midm` instead derive an independent full-range scalar per runtime row and are the only fused choices accepted for legacy artifacts. All dense modes use one occupancy selector: TileM256 is chosen only for `M >= 256` and `ceil(M/256) * ceil(N/128) >= ceil(2*SM_count/3)`; otherwise TileM128 runs. The `*_midm` modes therefore always remain TileM128. All values are experimental and default-off; unknown spellings and mid-process changes fail. **Tops the dense precedence chain** — it changes the served activation contract, so it outranks `PRISMAQUANT_CB_FP4_FUSED_MIDM` and `PRISMAQUANT_CB_BF16_SM120` (see [Lane precedence](#lane-precedence--which-flag-wins-when-several-are-set)). A selected mode that the **load-time** gate finds ineligible **fails the load**; a mode that passes that gate and then declines a concrete **call** (in practice the rowwise / static-LSQ quantizers' half-precision guard) **raises** naming the activation dtype and shape. Neither falls through to the exact BF16 route: that route's activation bucket is the fp32-emulated group QDQ rather than the format's native ue4m3 scale factors, so serving it would silently substitute the contract the flag exists to make explicit. The K24 short exact gate passed, but long-context evidence is mixed and no >=4B/MoE served validation exists; see the [dated audit](audits/fused_nvfp4_enablement_2026-07-31.md). |
| `PRISMAQUANT_CB_FUSED_FP4_MOE` | off | Grouped-MoE fp4-CB native-FP4 prefill opt-in. Static `1`/`128` and `256` select TileM 128 and 256 and require both attested stage scalars. `static_lsq`/`static_lsq128` and `static_lsq256` select those same tiles while reusing the shared fixed-`G` LSQ quantizer. `rowwise`/`rowwise128` and `rowwise256` use independent runtime row scales and may serve legacy artifacts. **Tops the routed precedence chain** — it changes the served activation contract, so it outranks `PRISMAQUANT_CB_MOE_PERSISTENT_B` and `PRISMAQUANT_CB_BF16_SM120` (see [Lane precedence](#lane-precedence--which-flag-wins-when-several-are-set)). An ineligible selection **fails the load**, naming the cached eligibility reason; a mode that passes that gate and then misses at **call** time **raises** (`became unavailable after model load`). Neither returns to the exact native BF16 quality bridge — that bridge serves a different activation contract, and Gridbook does not substitute one silently. Unknown spellings and mid-process changes fail. Keep this off pending the dated audit's routed-quality, workload, and routing-policy gates. |
| `PRISMAQUANT_CB_BF16_SM120` | off | `1` routes the quality-preserving BF16 grouped bridge (every default NVFP4-CB prefill — dense `E=1` and routed MoE — plus the FP8-CB fallback) to the **sm12x-native** CUTLASS 3.x collective instead of the default SM80-schedule `DefaultGemmGrouped`. Compiled only for cc 12.0/12.1; resolved at model load and **fails the load** if unavailable rather than silently serving the other lane. Same operands, same single bf16 round, different FP32 reduction order — bit-gated against the torch reference, served protocol NOT run. The lane's collective has two A-source modes (bit-identical to each other, gated `torch.equal`): the row-padded copy, and an **in-mainloop A-row gather** that never materializes the padded activation; with the gather mode and the swizzle-group-aligned expert order, measured (GB10, pingpong 64×128×64): 1.13–1.37× the default bridge, and vs segmented BF16 matmuls **1.03–1.05× at T=128 and 1.10–1.15× at T=512** (the padded-copy mode's 0.83–0.92× T=512 deficit is closed at the construction level). The routed path still pays one host read of the per-expert block offsets per layer. **The swizzle-group packing is coupled to the expert-chunk size and is off unless ONE chunk covers the layer** (`chunk >= E`): the decoded BF16 transient is chunked over experts by `PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK`, or by `PRISMAQUANT_CB_PREFILL_CHUNK_BYTES` (1 GiB) divided by one expert's `w13` BF16 bytes (`2·inter·hidden·2`), and a narrower chunk indexes blocks as `block_off[c0]..block_off[c1]`, which assumes expert-major contiguity. So a layer whose experts do not fit one chunk — the `E=128` cells in the benchmark run at `chunks=2` — takes the gather but NOT the packing, and lowering `..._CHUNK_BYTES` disables it the same way. The measured order win is an `E=32` result. Packing within a chunk is queued in [ROADMAP](../ROADMAP.md#p1--close-the-remaining-native-parity-gaps). **Lowest lane in both precedence chains**: a fused NVFP4 mode or, on the routed path, persistent-B (default `auto` since 0.8.9 — so on routed CB layers this lane serves only where persistent-B is off or cannot attest) will serve instead — this lane is still attested at load either way (see [Lane precedence](#lane-precedence--which-flag-wins-when-several-are-set)). Unknown spellings and mid-process changes raise. See [KERNELS](KERNELS.md#sm12x-native-grouped-bf16-opt-in-prismaquant_cb_bf16_sm120) and the [benchmark table](BENCHMARKS.md#2026-08-02-sm12x-grouped-bf16-lane-in-mainloop-a-row-gather--swizzle-aligned-tile-order-proposal-data). |
| `PRISMAQUANT_CB_FP4_FUSED_MIDM` | off | `1` routes **dense FP4-CB v2 quality prefill at 9 ≤ M ≤ 128** to the fused decode-in-prologue lane (`csrc/cb_fused_fp4v2_gemm.cu`) instead of `expand_fp4_v2_to_weight` + the owned CUTLASS bridge. CONTRACT-PRESERVING: the decoded weights are bit-identical to `cb_expand_v2` and the activation is the same group-16 QDQ output, so only the FP32 GEMM reduction order changes. Compiled only for cc 12.0/12.1; resolved at model load and **fails the load** if unavailable rather than silently serving the bridge. Since 2026-08-02 the load-time gate is per LAYER, not merely "is the extension present": an uncompiled rung, `K % 256 ≠ 0`, `N % 8 ≠ 0`, a non-v2/non-product format, or a fused module whose roles use different interned codebooks all **fail the load** with the reason named — those are properties of the layer, so with the flag on they would otherwise have served the bridge for every request while the load reported success. The **M band is the one call-time fall-through** and is deliberate: `9 ≤ M ≤ 128` is a property of the REQUEST (and `M ≤ 128` is a HARD gate the kernel itself enforces, since decode-in-prologue re-decodes B per M-tile), so out-of-band shapes take today's exact path unchanged — that is what makes this a mid-M lane. Measured 1.06–4.37× the bridge at M ∈ {9,16,32,64,128}, bit-checked against a same-config oracle; served protocol NOT run. **Outranked by `PRISMAQUANT_CB_FUSED_FP4`** and above `PRISMAQUANT_CB_BF16_SM120` in the dense chain; still attested at load when overridden (see [Lane precedence](#lane-precedence--which-flag-wins-when-several-are-set)). Unknown spellings and mid-process changes raise. See [KERNELS](KERNELS.md#fp4-cb-v2-fused-mid-m-opt-in-prismaquant_cb_fp4_fused_midm) and the [benchmark table](BENCHMARKS.md#2026-08-02-fp4-cb-v2-fused-mid-m-lane-microbenchmark-proposal-data). |
| `PRISMAQUANT_CB_MOE_PERSISTENT_B` | `auto` (0.8.9) | The persistent-B decode-in-mainloop lane for the **routed CB MoE quality prefill** above the `M<=16` GEMV band, BOTH payload families: FP4-CB two-tier v2 (`csrc/cb_moe_persistent_b.cu`, ROADMAP K1.1) and stock FP8-CB (K1.1's second payload family). A CTA owns one (expert, N-tile), decodes that weight tile from packed CB bytes into shared memory ONCE, and streams the expert's routed rows through it, so the `[E,N,K]` BF16 transient never exists and unrouted experts cost nothing. Same activation payload, FP32 accumulate, one bf16 round; weight decode is bit-identical to the expanders (`torch.equal`), so only the FP32 reduction order differs from the bridge (reassociation-class). **Unset means `auto` since 0.8.9**: each routed CB layer engages its family's arm where the load-time predicate and the extension attest, and keeps the expand+bridge route where they do not (per-role FP8-CB split books, an ineligible tile config, a missing extension), with a per-layer fallback line naming why — the bridge is exactly the pre-0.8.9 default. `1`/`require` keeps the pre-0.8.9 semantics exactly: every routed CB layer must take the lane, and a layer no arm can serve **fails the load** by name — the A/B-integrity contract. `0`/`off` is the kill switch. FP8 eligibility: cfg1 k≤33, cfg4 k≤31, cfg2/3 to k=48 (the 2-CTAs/SM occupancy floor). It takes precedence over `PRISMAQUANT_CB_BF16_SM120` for these layers, because it replaces the pair of operations that lane is one half of — and is itself outranked by `PRISMAQUANT_CB_FUSED_FP4_MOE`, which changes the activation contract (see [Lane precedence](#lane-precedence--which-flag-wins-when-several-are-set)); when overridden it is announced rather than discovered. Unknown spellings and mid-process changes raise. Served evidence: the FP4 arm's same-session served A/B on the DSv4 92 GB body (kl_mean −0.051 %, PPL −0.30 % — arithmetic noise) and the 0.8.9 default-state served KL/PPL leg on the shipped clean 87 GB body, on top of each arm's bitwise decode-identity CUDA suite and the 15.8–18.4× whole-routed-operator microbenchmark at DSv4 shapes. See [KERNELS](KERNELS.md#persistent-b-decode-in-mainloop-default-auto-prismaquant_cb_moe_persistent_b) and the [benchmark table](BENCHMARKS.md#2026-08-02-persistent-b-grouped-moe-decode-in-mainloop-microbenchmark-proposal-data). |
| `PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG` | `0` | Tile override for the lane above; `0` lets the kernel choose from the SHAPES (mean routed rows per expert), which is the production setting. A non-zero value is a 1-based index into `cb_moe_persistent_b_configs()` and is **validated at model load against what this build compiled**, so a stale or mistyped index fails the load instead of aborting the first request that carries routed rows. Measurement knob only. |
| `GRIDBOOK_MXFP8_DENSE` | off | `1` opts in Gridbook's **direct MXFP8 dense W8A8 lane** (`csrc/mxfp8_dense_gemm.cu`) for `mxfp8_e4m3_e8m0_g32`: E4M3 weights with one UE8M0 scale per 32 K-elements, served on the stock sm120/sm121 block-scaled CollectiveBuilder collective (`kind::mxf8f6f4`), with activations quantized dynamically per 32. Correctness was audited on sm_121, but the NATIVE-PARITY served timing bench remains pending. With the flag unset, a direct-MXFP8 passthrough unit refuses at model load with this flag named; compiled only for cc 12.0/12.1 and attested at load like every lane (`lane_select.require_lane`). The source `fp8_e4m3_ue8m0_block128` wire is **not controlled by this flag** and never enters this W8A8 lane: it uses the separately attested W8A16 source route with unchanged BF16 activations. Unknown spellings and mid-process changes raise. |
| `PRISMAQUANT_CB_FUSED_MIDM` | `auto` | Resolved during model load against the loading device. Unset enables the CUTLASS decode-in-prologue specialization only on cc 12.0/12.1; on SM89 it does **not** query/build the Blackwell-only fused extension and M>8 uses native CUDA expansion + native W8A8 CUTLASS. `0` disables the specialization everywhere. `1` requires it and therefore fails model load on SM89, an unidentified device, any other unsupported capability, or a supported device whose extension does not load; Gridbook never substitutes the expansion route behind that explicit requirement. Only `''` (unset), `0` and `1` are accepted — before 2026-08-02 this was compared `!= "0"`, so `false` and `off` silently enabled the lane they read as disabling. Process-stable: changing the value after dispatch is fixed raises rather than taking effect. |
| `PRISMAQUANT_CB_DECODE_CONTRACT` | `v1` | `v2` selects the scale-epilogue-hoist decode contract. Measured **null** on the served 27B; kept for reproducibility. |
| `PRISMAQUANT_DEBUG_PREFIXES` | off | `1` prints, per Linear, whether it resolved to a CB scheme or to a config-declared non-CB group — the first tool to reach for when memory use is higher than expected. |
| `PRISMAQUANT_PRELOAD_FUSED` | off | `1` independently attempts to build/preload **every** native extension family at registration — decode GEMV, GEMV-v2, grouped BF16, both fused FP8-CB/NVFP4-CB modules, fused FP4-v2, persistent-B MoE, direct MXFP8 dense, and source block-FP8 W8A16 — so both arms of a served A/B can carry identical extension residency. (Before 2026-08-02 it warmed only the two fused modules, which left the other five free to differ between arms; the name is kept because it is the published one.) Each family is attempted independently and fail-soft: one that will not build on this box leaves the others warmed. Only `''` (unset), `0` and `1` are accepted: this was compared `== "1"` until 2026-08-02, so `true` and a stray-space `" 1 "` warmed nothing while the operator believed both arms were residency-matched — a failure invisible in the results. A family that does not warm is now named on stderr rather than silently skipped. Registration treats this as a capability probe; a serving caller still requires its selected native operation and fails closed (see the measurement side-effect in [`KERNELS.md`](KERNELS.md#a-measurement-side-effect-worth-knowing)). |

### Lane precedence — which flag wins when several are set

Setting two lane flags is legal and does not raise. Exactly one route serves,
and **which one is decided by what each flag CHANGES, not by the order the
flags were added**: a flag that changes the served *activation contract*
outranks any flag that only moves the GEMM schedule behind the contract the
artifact already declares.

| FP4-CB path | Precedence, highest first |
|---|---|
| **Routed** (MoE) | `PRISMAQUANT_CB_FUSED_FP4_MOE` **>** `PRISMAQUANT_CB_MOE_PERSISTENT_B` (default `auto` since 0.8.9 — engaged wherever its family arm attests) **>** `PRISMAQUANT_CB_BF16_SM120` **>** the expand + grouped bridge (the auto fallback) |
| **Dense** (Linear, prefill above the `M ≤ 8` decode-GEMV band) | `PRISMAQUANT_CB_FUSED_FP4` **>** `PRISMAQUANT_CB_FP4_FUSED_MIDM` **>** `PRISMAQUANT_CB_BF16_SM120` **>** the default expand + bridge |

The fused NVFP4 modes sit at the top of both chains for the same reason: they
change the served activation contract. Everything below them is a schedule
change under an unchanged contract. Within the routed chain, persistent-B
outranks the sm12x bridge lane because it *replaces the pair of operations that
lane is one half of* — expansion plus grouped GEMM become one
decode-in-mainloop launch, so there is no bridge left for the lane to schedule.

Two consequences worth knowing:

- **A losing flag is still attested at model load.** Every selected lane is
  resolved and attested during `process_weights_after_loading`, including the
  ones a higher-precedence flag will override, so an unserveable explicit
  selection **fails the load** rather than being silently ignored. You cannot
  hide a broken lane behind a winning one.
- **The model-load dispatch line names the route that will actually serve**,
  and names the flag it outranks. Before 2026-08-02 a run with both routed
  flags set logged persistent-B at load and then served the fused kernel for
  every request — a dispatch log that names the wrong kernel is worse than no
  log, because it is the artifact an A/B is read from.

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

The grep also finds one name that runs the other way. `moe_routing.py`'s
TileM commentary offers `PRISMAQUANT_CB_GROUPED_TILE_M` as a measurement
override, but **no code reads it** — the name occurs once in the tree, inside
that comment. There is no operator TileM override today; the K0.4 selector
chooses the tile from host-known integers and reports the choice through
dispatch telemetry. (`PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA` in that same grep
output is not a variable at all: it is a Python constant in `bench_serve.py`
holding a schema string.)

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

| Variable | Recognized value(s) | Default (unset) | Effect |
|---|---|---|---|
| `PRISMAQUANT_CB_FP8_SCHED` | `legacy` | double-buffer | fp8-CB dense decode GEMV schedule. `legacy` selects the original single-buffer path; the two are **bit-identical** (same partial-sum order). |
| `PRISMAQUANT_CB_FP4V2_SCHED` | `db` | single-buffer | fp4-CB two-tier dense decode GEMV schedule. `db` selects the prefetch double buffer, which measured the loss; the two are **bit-identical**. |
| `PRISMAQUANT_CB_W2_SCHED` | `legacy`, `rowpack` | the round-2 warp schedule (~3 superblocks/warp below `n_sb = 8`) | `w2` grouped-MoE schedule. `legacy` is the original 8/4-warp heuristic and is the numerics-preserving baseline — the default reassociates the `w2` fp32 partial-sum order, which is why it is gated. `rowpack` measured negative and is kept as a recorded result. |
| `PRISMAQUANT_CB_W2_WARPS` | integer `1`–`8` | `0` (override disabled) | Warp-count override, applied on the default schedule only — `legacy` ignores it. Values outside 1–8 leave the computed count alone. |
| `PRISMAQUANT_CB_W2_ROWS` | `4`, `8`, `16` | `8` | Rows (= warps) per block, read **only** under `PRISMAQUANT_CB_W2_SCHED=rowpack`. Anything else is coerced to 8, and a shape whose smem exceeds 48 KiB falls through to the default warp schedule regardless. |

**None of these five validates its input.** They are `strcmp`/`atoi` reads in the
launcher (`csrc/cb_gemv.cu`, `pq_env_is` / `pq_env_int`), so an unrecognized
value is silently the default rather than an error: `..._W2_SCHED=rowpak` runs
the default schedule, and a non-numeric `..._W2_WARPS` reads as `0` and disables
the override. Unlike the lane flags in the operator table above, which fail the
model load on an unknown spelling, these never raise — read back what you set.

**MoE transient sizing and grouped-launch overrides.** These change memory and
launch shape, not kernel family. Production dispatch is fixed: grouped CUDA GEMV
at M≤16; above 16, eligible FP8-CB fused CUTLASS or exact expansion + owned
CUTLASS grouped GEMM.

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK` | unset (byte-budget derived) | Explicit positive expert count per transient chunk. It overrides the byte budget; reduce it only when measured serve slack requires it. |
| `PRISMAQUANT_CB_PREFILL_CHUNK_BYTES` | `1073741824` (1 GiB) | Maximum BF16 weight transient used to derive the expert chunk. Activations, routing buffers, and allocator overhead are additional; if one expert exceeds the budget, chunk 1 is used with a warning. |
| `PRISMAQUANT_CB_GROUPED_TRIM` | `1` (on) | Spends ONE `.item()` on the real block total and slices the padded grouped collective down to it, so no wasted tile is ever launched. `0` keeps the full static capacity — up to E wasted tiles per stage — in exchange for a path with **no host read of device data at all**. Any value other than `1` reads as off. |

**Correctness gate you can switch off** — there is exactly one, and it is
debug-only.

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_SKIP_CB_CAST_CHECK` | unset (gate ON) | `1` **downgrades a load-time correctness failure to a warning.** The gate proves the codebook sidecar survives its cast to the serving dtype bit-for-bit; the cast is exact only for target-grid values, so a learned table emitted off-grid otherwise means every CB weight in that layer decodes against a silently-rounded table. Setting this serves those wrong values and prints why. It is for diagnosing a bad artifact, never for running one. |

**Custom-op boundary**

The `prismaquant::cb_linear_forward` and `prismaquant::cb_moe_forward` opaque
ops are the sole production entry points. The switch that once exposed their
host-side branches to tracing no longer exists, and **there is no environment
variable here** — this section documents an invariant, not a knob.

**No Gridbook op carries `torch.Tag.cudagraph_unsafe`, and none should.** Since
the M-branch hoist, dynamo sees only those two whole-dispatch ops; the kernel
ops execute inside their eager implementations and are never graph nodes at
all, so a tag on them was metadata nothing read. Tagging the two whole-dispatch
ops instead is not the fix — it would make **every** CB layer an eager
partition boundary, which is the 2026-07-21 corruption configuration at worse
granularity. The two ops are capture-safe by construction: the M-branch
resolves host-side at capture time, and the arms that would host-sync are
unreachable at captured decode sizes.

The tag only ever did anything under `use_inductor_graph_partition=True`
combined with PIECEWISE cudagraphs, where it forced each tagged op into an
inductor graph-**partition** boundary. FULL capture ignores partitioning
entirely. Reproducing the historical corruption today would require reverting
the M-branch hoist *and* enabling `use_inductor_graph_partition=True` with
piecewise cudagraphs on a torch predating
[pytorch#165815](https://github.com/pytorch/pytorch/pull/165815) — three
conditions, none of which this repo can reach on its own.

## Tests

`tests/test_cb_kernels.py` uses an independent PyTorch decode on the **real
exported** 0.6B tensors and (a) matches `nvfp4_cb_reconstruct @ x` to ≤1e-2 rel,
(b) checks codeword extraction bit-exactness vs `nvfp4_cb_unpack`.
`tests/test_cuda_gemv.py` gates the `cb_gemv.cu` kernels (dense + grouped-MoE fp8
and fp4-v2, QDQ bit-exactness, the expander) against independent PyTorch/FP64
references. The grouped-BF16 bridge is gated against segmented BF16 matmul
references, while `tests/test_fused_prefill.py` gates the specialized prefill
kernels and `tests/test_persistent_tc.py` gates the research-only persistent-N
source behind its opt-in. `tests/test_fp8_source_w8a16_cuda.py` independently
gates the source GEMV and BF16 expander; the integration suite separately pins
resident raw storage, dense/grouped dispatch, no activation QDQ, and fail-closed
load behavior.
