# Serving kernels

This document describes how a `gridbook` artifact is served fast. It is a design and
status document, not an API reference; the normative decode semantics are in
[`SPEC.md`](SPEC.md). The single live implementation checklist is the
[`ROADMAP.md` kernel TODO](../ROADMAP.md#kernel-todo-canonical); status prose
here is evidence and design context, not a second backlog.

Terminology used below: **GEMM** = matrix-matrix multiply (prefill / large batch);
**GEMV** = matrix-vector multiply (decode / batch-1 or small batch); **M** = the
number of flattened token rows in a dispatch (for grouped source-FP8 BMM this
does not include the separate `G` dimension); **MMA** = a tensor-core
matrix-multiply-accumulate instruction; **QDQ** = quantize-then-dequantize the
activations; **smem** = GPU shared memory; **TTFT** = time to first token
(prefill latency).

## The three invariants everything is built around

- **INV-1 — no resident expansion.** A CB layer keeps its packed `cb_qweight`
  (indices + scale plane) plus the small shared codebook; a source block-FP8
  layer keeps its raw E4M3 weight plus UE8M0 scale blocks. The dense BF16 weight
  is **never resident**. Decoding happens in registers/smem per tile, or into a
  per-layer scratch buffer that is freed after the matmul.
  This is what makes a smaller-on-disk artifact also smaller in memory — the
  reason the format fits large models on one box. A resident-footprint assertion
  is a load-time gate, not a nicety.
- **INV-2 — native tensor-core matrix kernels for prefill.** FP8-CB decodes into
  native E4M3 codes and feeds CUTLASS W8A8. FP4-CB's quality-preserving lane
  decodes to a bounded BF16 transient and feeds an owned CUTLASS grouped BF16
  GEMM; it does not claim W4A4 arithmetic. The separate native-W4A4 lane remains
  gated by served quality, not just kernel speed. A decode-to-BF16-then-plain
  `torch` matmul implementation is an offline reference only.
- **INV-3 — native-only, fail-closed serving.** Gridbook defines, compiles, and
  dispatches no Triton operators. Decode/expansion/QDQ/routing support work is
  native CUDA; GEMM and grouped GEMM are native CUTLASS. If the native operation
  required by a format or shape is unavailable, serving raises a diagnostic
  error instead of changing kernel family or silently accepting lower
  throughput. A vLLM installation may itself contain Triton for unrelated
  runtime components; that is outside Gridbook's operator lane.

The same rule covers activation glue. Gridbook does not call vLLM's
`apply_moe_activation` helper because its SWIGLUSTEP branch imports Triton.
Supported gated activations resolve at model load to direct registered
`torch.ops._C.{silu,gelu,gelu_tanh,swigluoai}_and_mul` operators; any other
activation fails before serving (`gridbook/native_cutlass.py`).

## Target hardware

The reference target is the **NVIDIA GB10 / DGX Spark**: Blackwell **`sm_121`**
(the `sm_120` consumer family — **not** the `sm_100a` datacenter family, so the
`tcgen05` MMA instructions do not exist here; the FP4 path is the `sm_120/121`
block-scaled `mma` family). 128 GB unified memory (~121 GB usable), ~273 GB/s.
Toolchain: `nvcc` 13.0. Opt-in smem is **99 KB per block** — a flat FP4 codebook
is `2^k × 4` bytes, so a flat table is comfortable to `k ≤ 13`, marginal at
`k = 14`, and impossible at `k ≥ 15` (those rungs need a structured/computed
codebook — see [`SPEC.md`](SPEC.md) §8).

---

## Decode path (small M): bandwidth-bound fused GEMV

Decode is where a codebook format *should* win: fewer bytes per weight means less
HBM traffic, and at batch-1 the tensor cores are idle anyway, so there is no MMA
disadvantage. The decode kernel streams the packed indices, expands each group in
registers, multiplies by the group scale, and accumulates against the few
activation rows — **never** materializing the full weight (INV-1).

- **FP8-CB decode** uses a CUDA dequant-GEMV with a **fused activation QDQ**, an
  E4M3-byte scale lookup, and a warp-per-superblock decomposition. On the 27B
  artifact this took decode from 4.2 tok/s (the initial Triton path) to **~10.3
  tok/s — at/above the native NVFP4/FP8 baseline** (10.26). At matched body bytes
  that parity *is* the ceiling: decode is bandwidth-bound at ~250-355 GB/s
  effective, and both artifacts move the same bytes.
- **FP4-CB two-tier (v2) decode** reads the **9-byte** scale plane instead of 16
  and **composes the E4M3 scale in-register**: a nibble indexes a 16-entry
  multiplier table (in registers/constant memory), then `× 2^(E-127)` via an
  exponent add (FP32-exact; `exp2`/`ldexp`-style bit math). That is ~3-5 extra ALU
  ops per 16 weights on a kernel whose cost is memory, so the smaller scale plane
  (−44% scale bytes, −8.75% total stream at k=16) makes v2 decode neutral-to-faster
  than v1. The E4M3 plane is never reconstructed resident (INV-1 for v2).

**Dense dispatch.** Every arm is native and the row boundaries are part of the
serving contract:

| Format / M | Production kernel |
|---|---|
| FP8-CB or FP4-CB, `M ≤ 8` | Native CUDA GEMV |
| FP8-CB, `9 ≤ M ≤ 128` | Fused CUTLASS decode-in-prologue when the rung/layout/device predicates hold; otherwise native CUDA FP8 expansion + CUTLASS W8A8 GEMM |
| FP8-CB, `M > 128` | Native CUDA FP8 expansion + CUTLASS W8A8 GEMM |
| FP4-CB, `M > 8` | Native CUDA BF16 expansion + Gridbook-owned CUTLASS grouped BF16 GEMM with `E=1` |
| Source block-FP8, `M <= 8` | Native CUDA raw-E4M3/UE8M0 W8A16 GEMV; BF16 activations are not quantized |
| Source block-FP8, `M > 8` | Native CUDA BF16 expansion + Gridbook-owned CUTLASS grouped BF16 GEMM; `E=1` for dense and one problem per DSV4 `wo_a` group |

A missing native extension is an error, not another dispatch arm. See the
CUDA-graph section for why this host-side branch matters. FP4-v2 model load
requires the v2 extension even when the decode selector remains `inherited`:
`cb_expand_v2` owns the exact quality expansion, and its device prepare
currently admits only CUDA cc 12.0/12.1. The grouped-BF16 GEMM is SM80-capable
in isolation, but does not lower that complete FP4 serving floor.
The dense table applies only after the 0.5 load gate: public CB dense Linears
must be biasless, and FP4 must be unsigned product-v2. A non-`None` bias,
signed S-rung, or FP4-v1 dense layer is format-valid where applicable but has
no complete owned every-M native operation. The public method rejects bias;
model load rejects the unsupported FP4 families.

### Source block-FP8 W8A16 (`fp8_e4m3_ue8m0_block128`)

This source-passthrough wire is a **weight-storage contract**, not permission to
change the activation format. Gridbook retains the source E4M3 bytes and one
UE8M0 exponent per 128-by-128 weight block. BF16 activations cross the complete
route unchanged; neither arm calls an activation quantizer or QDQ operation.

- `fp8_source_gemv` owns `M <= 8`, where `M` is the product of the leading
  token dimensions after `[...,G,K]` is viewed as `[M,G,K]`; it is not `G*M`.
  It reads the raw E4M3 weight byte and the matching 128-by-128 UE8M0 scale in
  the GEMV, accumulates in FP32, and rounds the output to BF16. It does not
  materialize a weight tile.
- `fp8_source_expand_bf16` owns the large-M weight conversion. It expands one
  layer to a contiguous BF16 scratch tensor, which is consumed by
  `cb_bf16_grouped_mm` and then released. Dense Linear uses the bridge with
  `E=1`. For grouped DSV4 `attn.wo_a`, `[M,G,K]` activations are made
  group-major, the expanded weight is viewed as `[G,N,K]`, and cumulative
  endpoints `[M,2M,...,G*M]` describe exactly one existing CUTLASS problem per
  group; the result is restored to `[M,G,N]`. No new framework BMM fallback is
  involved.
- Load requires both the source extension and the owned grouped-BF16 bridge.
  Missing source, strict symbol, supported device, shape, or bridge capability
  is a load-time refusal. Grouped BMM is qualified only for DSV4's exact
  per-group geometry `G=8, N=1024, K=4096` at `tp=1`; any near miss is refused
  before extension resolution or adapter-marker installation. Dense source-FP8
  Linears are a separate contract: they retain the alignment and scale-shape
  gates above and are not restricted to the grouped `wo_a` geometry; this
  release still admits them only at `tp=1`.
- Whole-layer dispatch stays behind an opaque custom op. `M` is selected when
  the operation executes. The unit gate covers dense CUDA-graph replay at actual
  flattened decode-token sizes `M in {1,2,4,8}`; `G` does not turn a grouped
  `M=1` decode into `M=8`. Large-M expansion remains outside those graphs, and
  exact-artifact grouped graph evidence remains a pre-tag gate.

The direct `mxfp8_e4m3_e8m0_g32` wire is intentionally separate. It retains the
existing opt-in W8A8 `mxfp8_dense_gemm.cu` route and dynamically quantized MXFP8
activations. The W8A16 source route does not reuse its scale-plane conversion or
its activation quantizer. Kernel unit gates, exact-artifact served parity, and
served performance for W8A16 remain pre-tag requirements; no result is claimed
here before those gates are recorded.

---

## Prefill / non-GEMV path (M > 8): native transient expansion

The default prefill **expands one layer's weight tile transiently** into a scratch
buffer, runs a GEMM, and frees the buffer. Memory stays bounded (INV-1). FP8-CB
expands **directly to FP8** (the codebook values are already on the E4M3 grid)
and calls the registered native CUDA per-token quantizer plus CUTLASS scaled
GEMM directly (INV-2). Gridbook deliberately bypasses
`vllm._custom_ops`: that Python convenience layer can select a Triton
implementation for unsupported shapes. Model load attests the registered
`torch.ops._C.dynamic_per_token_scaled_fp8_quant` and
`torch.ops._C.cutlass_scaled_mm` operators, and the call site enforces the
CUTLASS shape contract before launch. A missing ABI or incompatible shape is an
error, not an implementation switch. The quality-preserving
FP4-CB v2 path expands with native CUDA to a bounded BF16 transient and consumes
it with Gridbook's native CUTLASS bridge. It preserves the established
activation-QDQ contract but does not claim native W4A4 arithmetic. The separate
native-FP4 decode-in-prologue kernel is opt-in because it changes the served
activation bucket. The current `static_lsq` policy preserves the
producer-attested global scale and packed payload while fixing the original
dense accuracy and timing defects on the short exact screen, but its
long-prompt evidence remains statistically unresolved and it has no >=4B or
MoE served validation. It therefore remains opt-in; see the
[fused-NVFP4 enablement audit](audits/fused_nvfp4_enablement_2026-07-31.md).

The honest limitation of transient-expand is **memory traffic**: the tile is
written to HBM and then read back by the GEMM, so prefill moves roughly 2× the
bytes of a resident-weight GEMM. Tuning the expander (the native CUDA expander
measured at ~2× the now-retired Triton prototype, with FP8-direct output) narrowed
but cannot remove this — on the 27B
artifact prefill went 1.62 s → 1.08 s TTFT versus the native baseline's 0.75 s, a
residual ~1.44× set by the doubled traffic. Only a kernel that **never
materializes the tile** closes it fully.

### NVFP4-CB fused native-FP4 prefill (opt-in)

`csrc/cb_fused_fp4_gemm.cu` and the vendored
`sm120_cb_fused_fp4_mma.hpp` decode FP4-CB weights directly into a Blackwell
block-scaled MMA prologue. The output is bit-exact with the native NVFP4
reference when both consume the same E2M1 activation bytes, UE4M3 scale-factor
bytes, and row residuals. That isolates and qualifies the packed-weight decoder
and GEMM; it does not make native activation QDQ equivalent to the default
FP32-emulated activation bucket.

The preferred experimental activation policy is `static_lsq`. It keeps the
producer-attested global scale `G`, and its E2M1/SFA payload is byte-equal to
vLLM's fixed-`G` quantizer. For each row it replaces only the existing EVT
residual with `(x·q)/(q·q)`, falling back to `1/G` when `q·q` is zero. This is
the exact least-squares fit for the already-fixed native payload. It is one
policy in the shared activation quantizer and feeds the existing fused operator:
there is no extra serialized metadata, resident weight, weight decoder, or
GEMM. `static_lsq_midm` applies the same policy only for `16 < M <= 128`.

Two performance fixes are also shared by every dense activation policy:

- The layout-v2 scale decoder now assigns two lanes per row and composes four
  adjacent factors per lane, eliminating repeated exponent/code loads. It kept
  output bits unchanged and improved all 12 interleaved production-shape raw
  cells by 7.9–23.4%.
- TileM is selected by occupancy rather than a blanket size rule. TileM256 runs
  only for `M >= 256` when
  `ceil(M/256) * ceil(N/128) >= ceil(2*SM_count/3)`; otherwise TileM128 runs.
  Device SM count is cached, probe failure selects 128, and both routes call the
  already-qualified concrete GEMM runners. Selected raw cells gained
  22.57–50.32%; narrow cells where TileM256 lost 19.81–21.69% remain on 128.

### Grouped (MoE) TileM selection (K0.4)

The routed grouped lanes now select TileM the same way, from
`moe_routing.cb_grouped_tile_m`. Before this the FP8 grouped path resolved
`tile_m=None` to the kernel's compiled default — so serving never reached
TileM=256 no matter what the shared-memory table admitted — and the FP4 path
read its tile off the *suffix* of an activation-policy env string
(`static_lsq256`), putting a performance knob on a numerics selector.

**Rule.** With `P = tokens × top_k` routed pairs over `E` experts and
`ρ = P/E`, TileM=256 is selected only when `ρ > 512` **and** the grid lower
bound `ceil(P/256) × ceil(min(2·inter, hidden)/TileN)` reaches the same
`ceil(2·SM_count/3)` occupancy floor the dense selector uses. TileN is 64 for
the FP8 grouped tile and 128 for the FP4 one. A build that did not compile 256
for this rung, a failed SM probe, or the zero-margin `(256, k32)` cell all
select 128.

**Why ρ.** A CTA decodes `TileN×K` weights once and then issues `t·TileN·K`
MACs, so `T(t) ∝ B(t)·(d + t·m)` over `B(t) = Σ_e ceil(c_e/t)` M-tiles. The
decode:MMA ratio is `1:t` independent of `N` and `K`, so **both projection
stages give the same condition** even though their shapes differ. The exact
padding lemma — `pad₂₅₆(c) − pad₁₂₈(c) = 128` iff `c mod 256 ∈ [1,128]`, else
0 — turns that into `ρ > 128·(1 + 256/x)` for `x = d/m`, and inverting the
dense TileM A/B above (22.57–50.32% at fixed occupancy ⇒ `2(x+128)/(x+256) ∈
[1.226, 1.503]`) bounds `x ∈ [75, 259]`, i.e. a threshold in `[254, 565]`. The
shipped 512 is the pessimistic end and is **proposal data for the grouped lanes
until a routed sweep pins it**; the mainloop is warp-specialized, so under
perfect decode/MMA overlap the crossover moves and the additive model is the
less conservative of the two in exactly the band `x` lands in.

### Grouped destination slice (`out` / `n_offset`, 0.8.3)

`cb_fused_moe_grouped` optionally writes its `N` output columns into a column
slice of a wider caller-owned `[Mp, ldd]` bf16 buffer instead of allocating a
fresh `[Mp, N]` one. Only the D leading dimension changes — the problem shape
stays `{Mp, N, K}` and the epilogue's `b_scales` node keeps indexing its own
`[E, N]` extent — so a bit difference between the two forms is a real defect,
and `test_destination_slice_is_bit_identical_to_a_fresh_buffer` asserts against
exactly that.

**Why it exists.** A routed stack whose `gate` and `up` halves carry different
codebooks is two GEMMs, but `native_moe_activation` consumes one fused
`[Mp, 2·inter]` buffer. Concatenating afterwards would move ~0.6 GB per layer
per forward at DeepSeek-V4-Flash shapes (~24 GB across 43 layers). Splitting
`N` is free by construction: the tile count `ceil(Mp/TM) × ceil(N/TN)` is
preserved exactly when `N` is split at a `TN` multiple, so two launches at
`N=inter` issue the same total tiles as one at `N=2·inter`, plus one launch.

The tile selector runs **once, and both role launches share its `tile_m`**. It
has to: `tile_m` fixes the shape of every routing tensor (`a_pad`, `expert_ids`,
`dest`), and the two launches share one routing build, so re-selecting per
launch would mean re-padding the activations per role and paying the routing
cost twice. Splitting `N` halves each launch's CTA count against the fused `N1`,
which is a consideration for what `N` the selector is fed, not a licence to
select twice.

**Graph safety.** The selector reads *only* host-known integers —
`topk_ids.shape`, layer constants, the extension's own compiled tile list, and
the cached (non-synchronizing) SM count. It never touches the routed histogram,
which is device data. That is not a convenience: `tile_m` fixes both the kernel
symbol (`run_moe_grouped<TM,KB>` is a distinct `__global__` per TileM) and every
routing tensor's shape (`cap_blocks = P//tile_m + E`), so it must be decidable
at graph-record time. A histogram-reading selector would also have to run
strictly *upstream* of `cb_grouped_pad_routing` (which takes `tile_m` as an
input and produces the trim read as an output), making its read a new, earlier
sync rather than one foldable into the existing one. The price is that a
histogram-free rule must hold for every histogram consistent with `(P, E)`, so
it widens later than an oracle would; both thresholds are above ordinary
chunked-prefill batch sizes.

Related fix: `cb_grouped_pad_routing` documented "NO HOST READS" while calling
`torch.bincount`, which sizes its CUDA output from `.max().item()` and
therefore host-syncs — the same trap the persistent-B lane hit. Every padded
grouped lane was uncapturable as a result. The counts now come from the
`scatter_add_` form that lane already proved (identical integers, static shape).

Both tiles compute the same values; the suite gates that by **equality**
pre-combine in stable-argsorted pair order, not by tolerance.

### Dispatch telemetry (K0.4)

Every fused call — dense and routed — records its latest route on the layer as
plain Python scalars via `nvfp4_activation_contract.emit_route`: requested
activation `policy`, the `symbol` actually invoked, `tile_m`, problem `shape`,
the activation `contract` that ran, fallback `state`
(`served`/`fallback`/`error`), the exact `reason`, and selector provenance
(`tile_rho`, `tile_candidate_ctas`, `tile_sm_count`, `tile_compiled`). This
extends the 0.4.2 dense mechanism above rather than adding a parallel one — the
same tensor-free, sync-free, last-write-wins attributes the A/B harness already
reads back — and the three original dense attributes keep their names. Writes
are two-phase (`error` before the launch, `served` after), so "raised
mid-launch" is distinguishable from "never launched". `tile_rho` is what makes
a tile choice auditable offline: a report reader can re-derive the verdict
without the GPU.

The current dense Qwen3-0.6B K24 same-process result is deliberately reported
at two levels. Exact 6×128 quality passed every predeclared gate and offline
one-token wall time improved `1.478x`; exact 2×512 with real chunking improved
`1.741x` but failed the 0.5% PPL gate; a 32×512 measurement-only screen had a
passing `+0.2635%` PPL point estimate whose prompt-cluster interval still crossed
the limit. Raw operator speed has not reached native-microkernel parity, while
offline one-token wall time is not streaming TTFT or p95 served-SLO evidence.
Keep the dense flag off by default until broader exact/model/task validation;
grouped MoE can reuse the same `static_lsq` quantizer and grouped GEMM, but it
is not quality-qualified. The available partial-LFM artifact predates the
serialized activation contract, so its LSQ attempt correctly fails closed
before kernel dispatch rather than inventing runtime scales.

### FP8-CB fused decode-in-prologue (the 1× fix)

#### Rung coverage: what this lane can and cannot serve (K1.2)

The FP8-CB product ladder is **every integer `k` in [28, 48]** (3.5–6.0 bpw in
0.125 steps; `runtime_contract.json` carries all 21, and Gridbook serves all 21).
The fused mid-M lane backs **the multiples of 4 — `k ∈ {28, 32, 36, 40, 44, 48}`
— and no others.** That is a property of the format and of TMA, not a missing
template instantiation:

1. **TMA box.** Packed B is fetched with a box of `(TileN, type_size)` *bytes*,
   where `type_size = 4k` is one 256-weight superblock. TMA requires the box's
   contiguous extent to be a 16-byte multiple, so `4k % 16 == 0` ⟺ `k % 4 == 0`.
2. **Uniform sub-table width.** The fused mainloop decodes a codeword with a
   *single* width `CbSubW = k/4` and indexes the flat codebook at
   `(s << CbSubW) + idx`. The format splits `k` over `n_sub = 4` **raggedly**
   (`csrc/cb_gemv.cu` `SubSplit`: widths are `k//4 + (i < k%4)`), so at `k = 37`
   the true widths are `(10, 9, 9, 9)` with non-uniform offsets. A uniform decode
   would be *wrong*, not merely unaligned.

Both conditions coincide, so one law expresses them, and the six compiled rungs
are already the maximum this collective admits. Supporting the other 15 would
need a different packed-B TMA schedule (a box spanning 4 superblocks is the
smallest 16-byte-aligned one for odd `k`, and its shared-memory cost is far over
budget) *and* a ragged-width decode — i.e. a new kernel and a producer-layout
conversation, not an instantiation.

Consequence for dispatch, and what ROADMAP K1.2 actually delivered: its first
arm ("instantiate every product rung") is closed by the laws above, so the
second is the live one — **encode the concrete route so an allocator cannot
price an unbacked fast path.** The compiled set is therefore *queryable*
(`cb_fused_kbits()`), Python gates on the derived law
(`codec.FP8_FUSED_KBITS`) and then confirms against the module rather than
carrying a duplicated literal, every switch in the kernel is generated from one
rung list, and an off-law rung is refused with a message naming the law and
pointing at the routes that do serve it. Rungs off the law are not unsupported —
they take the decode GEMV and the expand + CUTLASS quality bridge, exactly as
before.

Measured shared memory, `GemmKernel::SharedStorageSize` at TileN=64 / TileK=128
/ Stages=2 against the 101,376 B `sm_120` ceiling (regenerated 2026-08-02 from
`csrc/tools/smem_probe_tilem.cu`; the previously published table quoted the
pre-R6 base and was stale by up to 16,384 B once the LUT stage landed):

| TileM | k28 | k32 | k36 | k40 | k44 | k48 |
|---|---|---|---|---|---|---|
| 128 | 67,584 | 70,656 | 74,752 | 80,896 | 91,136 | 93,184 |
| 256 | 100,352 | **101,376** | 103,424 | 105,472 | 107,520 | 109,568 |

TileM=128 fits at every rung; **TileM=256 fits only at k28 and k32**, and the
k32 cell lands on *exactly* the ceiling (zero margin), so the tile selector will
not choose it until it is launch-verified. TileM must be a multiple of the
TiledMma's 128-row M, so 128/256/384 are the only candidates at all and 384 is
infeasible everywhere (`smem_A` alone is 98,304 B). The kernel encodes this as a
closed form that is `static_assert`ed cell-by-cell against those twelve measured
numbers, so a storage-policy change becomes a compile error rather than a stale
table.

Build cost: instantiating the six rungs × (dense unscaled, dense scaled, grouped
128) plus grouped 256 × {k28, k32} is **20 kernel instantiations, ~76 s cold-cache
JIT** in the GB10 container. The K1.2 work changed no instantiation, but it is
not free: measured both ways on the same box, the cold build went **71.4 s at
the merge base to 76.0 s / 75.7 s, i.e. +4.6 s (~6%)**. That delta is
compile-time *evaluation* — the rung-law predicates and the twelve-cell
`static_assert` table above — not code generation. The measurement and its
method are in [CONTAINER](CONTAINER.md).

#### Measured status

A fused collective mainloop that **decodes CB indices inside the GEMM's
global→shared prologue** — never writing the expanded tile to HBM — exists and is
**bit-exact** against the transient path (a forked `sm_120` block-scaled MMA
collective, packed-B TMA load + consumer-side smem decode). Its honest status:

- It **wins at medium M** — roughly 1.04× / 1.26× / 1.45× at M = 32 / 64 / 128.
- It **loses at large M** (≈0.22× at M≈1400). This is *structural, not a bug*:
  every M-tile CTA re-decodes the same weight (B) tiles, so decode work scales with
  `ceil(M / tile)` while the transient path expands each tile exactly once.
- Large-M parity therefore requires some **weight-stationary/no-HBM-
  materialization design** that amortizes weight decode across M. The first
  persistent-N implementation landed, passed parity, and measured 2–5.7×
  slower than transient expand at 27B shapes; it is rejected. Any replacement
  starts from the fresh roofline in the canonical TODO, while the serial
  transient path remains the large-M default.

A **baseline-parity gate** preceded all fork work: a plain `sm_120` block-scaled
GEMM built from vendored CUTLASS headers matched the runtime's native
`cutlass_scaled_mm` to within 0.91-0.99×, proving the toolchain and the tile-layout
understanding before the mainloop was touched. That gate has served its purpose
and its source (`csrc/sm120_fp8_gemm.cu`) was deleted on 2026-08-01 — it had zero
references and its binding validated per-token/per-channel scales it then ignored
(the epilogue was hardcoded to `{1.0f, 0.0f}`), which is a misleading thing to
keep around. The forks it de-risked are the live artifacts:
`sm120_fp8_mm_fork`/`fork64` in `csrc/cb_fused_gemm.cu`, both test-exercised.
See [`audits/ultraplan_perf_2026-08-01.md`](audits/ultraplan_perf_2026-08-01.md)
§4. Note the fork uses a **fixed-config**
GEMM: the runtime's `cutlass_scaled_mm` reconfigures on narrow N and is not
bit-exact across configs, so an N-chunked expand+GEMM overlap was tried and
**rejected** (0.46× and not bit-exact).

### FP4-CB v2 fused mid-M (opt-in, `PRISMAQUANT_CB_FP4_FUSED_MIDM`)

`csrc/cb_fused_fp4v2_gemm.cu` is the **contract-preserving** twin of the FP8-CB
mid-M lane above, for the *quality* FP4-CB path. It closes the 2026-08-01
performance audit's §3 P2a item and its structural cause (c): FP8-CB owned
M = 9–128 with a fused kernel; FP4 had **no mid-M lane at all** — M = 9–16
always took the BF16 bridge, and so did everything above it, because the only
fused FP4 kernel in the tree (`cb_fused_fp4_gemm.cu`) serves a *different*
activation contract behind its own six promotion gates.

**What it does.** Packed FP4-v2 CB rows are decoded to BF16 inside the CUTLASS
producer/consumer stage, so the `[N, K]` BF16 transient never reaches HBM. The
decoded values are **bit-identical to `cb_expand_v2`** and the activations are
the same BF16 group-16 QDQ output the bridge already consumes, so the served
weight values and the served activation bucket are untouched — the only thing
that moves is the **FP32 GEMM reduction order**, the same requalification class
the promoted FP8 mid-M kernel cleared.

**Design.** The collective is hand-assembled from the 16-bit forms of the four
choices upstream's `CollectiveBuilder` makes for F8F6F4 (it refuses 16-bit
input; see the sm12x BF16 lane below for the full reasoning) — the same
selection, on the **cooperative** kernel layer. The B side departs from the FP8
fork because fp4-v2 rows carry an **odd `type_size = 4k+9`**: the packed row
stride is never a 16-byte multiple, so TMA is structurally unusable for B. The
producer instead publishes a 16-byte per-stage descriptor (the CTA's first
output row) into smem under the stage's mbarrier, and the consumer threads
gather the packed bytes straight from gmem with aligned-u32 windows — the
construction `sm120_cb_fused_fp4_mma.hpp` already ships for this payload.
Because the packed stream never touches a TMA descriptor or a k-sized smem
layout, **`k_bits` is a runtime parameter** and the whole K12–K24 product
ladder is served by four compiled kernels.

**Tile and smem** (measured by `csrc/tools/smem_probe_fp4v2_bf16.cu`,
host-only, no launch). Shipped: **`128×64×64`, 2 stages**. `TileM=128` is the
cooperative layer's floor; narrow N is the fp8 mid-M lane's proven pattern and
is what buys the codebook its headroom; `TileK=64` keeps the 128-byte K-major
swizzle atom (64 bf16 = 128 B) and divides the 256-weight CB superblock exactly
four ways.

| TileN×TileK / stages | Lut 0 | Lut 4 KiB | Lut 16 KiB | Lut 32 KiB | Lut 48 KiB |
|---|---|---|---|---|---|
| **64×64 / 2 (shipped)** | 52,224 | 56,320 | 68,608 | 84,992 | 101,376 (0 margin) |
| 64×64 / 3 | 68,608 | 72,704 | 84,992 | 101,376 (0 margin) | OVER |
| 64×128 / 2 | 93,184 | 97,280 | OVER | OVER | OVER |
| 128×64 / 2 | 60,416 | 64,512 | 76,800 | 93,184 | OVER |

Budget is 101,376 B. The 48 KiB stage lands on **exactly** the ceiling with
zero margin and is deliberately not compiled — the fp8 lane's `TileM=256/k32`
entry is the precedent for treating a zero-margin config as untrusted. Every
shipped configuration is 1 CTA/SM, which is also what the fp8 mid-M twin runs
at while winning; at M ≤ 128 there is exactly ONE M-tile, so the grid is
`N/64` CTAs and per-SM occupancy is not the limiter the P1 grouped lane found
it to be.

**Codebook residency ladder.** The fp4-v2 product dictionary is BF16 *values* —
`(8 << ceil(k/2)) + (8 << floor(k/2))` bytes, 1 KiB at k12 rising to 64 KiB at
k24 — so it cannot be staged whole at the top of the ladder. The smem stage
holds a **prefix** of the flat `[sub0 | sub1]` codebook and the mainloop selects
its two gather pointers from the staged length **once per decode**, so a
partially staged table costs a pointer select and never a per-gather branch.
Full staging holds to k22; k23/k24 stage `sub0` only and read `sub1` from
global. The (256,16) fp32 compose table stays in global with `__ldg`, exactly as
`cb_expand_v2` and the decode GEMV do.

**Mid-M only, and the gate is in the kernel.** Decode-in-prologue re-decodes B
once per M-tile, so beyond one tile the redundancy dominates (the fp8 twin
measured 0.22× at M ≈ 1400). The binding itself `TORCH_CHECK`s
`1 ≤ M ≤ cb_fused_fp4v2_max_m()` (= 128), so a drifted python gate cannot
quietly serve a slow schedule; python reads the ceiling back from the kernel.

**Evidence.**

- **Decode bit-exactness — DONE.** `tests/test_fp4v2_fused_prefill.py` proves it
  three ways. The primary gate feeds a one-hot `A` (`a[m, k0+m] = 1`), which
  makes the GEMM a *read-out* of the decoded tile (`y[m,n] = W[n, k0+m]`
  exactly), and sweeps `k0` so the **entire** decoded `[64, 512]` tile is
  compared to `cb_expand_v2` **bit-for-bit at all 13 rungs** — a tolerance is
  never used. It is repeated with the codebook forced entirely to global
  (`force_lut_bytes=0`) at the three smem-tightest rungs, so the residency
  choice is proven not to move a bit. A `debug_mode` coordinate-write probe
  pins the decode write view against the MMA read view (it is what found a real
  swizzle-view bug: slicing the pipe mode before
  `as_position_independent_swizzle_tensor` produced a different physical
  mapping than the reader's — columns XOR-8 on odd rows, rows 8/9 transposed).
  Finally `sm120_fp4v2_bf16_mm_fork` — the same tile/TiledMma/epilogue with
  plain BF16 B, hence the identical FP32 reduction order — must reproduce the
  fused output bit-for-bit on the `cb_expand_v2` tile.
- **Served NATIVE-PARITY — PENDING.** The reduction-order change is gated only
  by unit tests (relative-L2 vs fp32, capped at 2e-3 and required to be no worse
  than a BF16 `F.linear` on the same operands, at M ∈ {9,16,32,64,128}). That is
  why the lane is OPT-IN.

**Promotion checklist**: decode bit-gate ✅ · smem table measured ✅ ·
end-to-end tolerance band ✅ · **served NATIVE-PARITY protocol ❌ (pending)** ·
the M ≤ 12 latency cliff below explained ❌.

**Measured** (GB10, cc 12.1, k16, warm medians, PROPOSAL DATA — see
[BENCHMARKS](BENCHMARKS.md#2026-08-02-fp4-cb-v2-fused-mid-m-lane-microbenchmark-proposal-data)):
**1.06×–4.37×** the shipping expand + bridge route across 27B/DSV4-class dense
shapes at M ∈ {9,16,32,64,128}, every cell bit-equal to the oracle. The band is
far above the fp8 twin's 1.04–1.45× for a structural reason: the fp4 quality
expand writes **BF16** (2 bytes/weight, 4× the fp8-CB expand's transient
traffic), so deleting it is worth more. One **unexplained cliff**: the fused
lane costs ~0.37 ms at M ≤ 12 and ~0.20 ms at M ≥ 13 on the 27B qkv shape
(reproducible, order-independent, and absent from the bridge) — profile it
before promotion.

**Related dispatch note.** The audit's §3 P3.5 item — extend the *native*-NVFP4
fused gate from `M > 16` to `M ≥ 9` so opt-in users do not hit the bridge at
M = 9–16 — is **moot for the quality path** now that this lane exists. That gate
is untouched here; it still matters only if P3 promotes first.

---

## MoE path: grouped (token, expert) GEMV

A fused Mixture-of-Experts layer packs many experts into stacked 3-D weights
(`(E, out, in)`; see [`SPEC.md`](SPEC.md) §4). The naive serving loop decodes and
matmuls **per expert**, which on a 256-expert model means a launch/sync storm
(~10k host operations per token) and cripples decode.

The production decode kernel is a **grouped GEMV**: **one launch per projection**
covers *all* routed `(token, expert)` pairs for that projection. On the 35B MoE
artifact this took decode from 3.5 tok/s (per-expert loop) to **~33 tok/s** (9.4×)
— **faster than BF16** (28.4) and within 8% of the native baseline (35.9), at 3×
smaller and −43% ALL-KL. For the FP4 two-tier grid there is a dedicated grouped
GEMV that composes the two-tier scale in-register per the decode rules above.

The correctness-first per-expert loop remains useful as a test/reference
calculation, but it is not a serving fallback. Production grouped decode requires
the native grouped GEMV and fails closed when that kernel is unavailable.

**Smem-resident dictionary (opt-in, `PRISMAQUANT_CB_GEMV=v2`).** The shipped
grouped GEMV gathers sub-codebook entries from global/L2 per lane per
superblock. `csrc/cb_gemv_v2.cu` instead stages the whole product dictionary in
shared memory once per block and bursts each output row's packed bytes into
smem before decoding it — the same insight the fused **prefill** mainloop
already uses (`csrc/cutlass_fork/sm120_cb_fused_mma.hpp`, where it removed the
k48 L1 cliff), applied to the MoE M≤16 grouped-decode regime. It opts in to the sm_121a
99 KB dynamic-smem budget, which the shipped decode GEMV does not (it caps its
rowpack request at 48 KB), so the k20/k24 staged configurations are only
expressible in the new kernel. The loader admits only the native Blackwell
target capabilities (12.0/12.1) with at least 99 KiB opt-in shared memory; this
PR's execution battery covers cc 12.1, not cc 12.0. On the DSV4 release widths
K=2048/4096 (8/16 superblocks), compile-time virtual-warp specializations
reproduce the inherited default kernel's eight accumulator chains, lane
reductions, serial warp sum, and final BF16 round bit-for-bit. This exactness
contract requires every `PRISMAQUANT_CB_W2_*` override to be absent. Other
widths retain the ascending rowpack reduction and may reassociate against the
inherited default. The alternate kernel is **not** the global default: unset
means `inherited`. Explicit `auto` or `v2` reaches it only where the hardware
gate and compiled occupancy predicate both pass;
`PRISMAQUANT_CB_GEMV=inherited` is the kill switch. The
selection resolves once per process and is fixed per (layer, stack) at load, so
the call-site branch is a trace-time constant and FULL-decode cudagraphs are
unaffected.

The final-source cc 12.1 operator gate for source SHA
`d72b15ecaad14e7af07f8af555259f5d1423cee2dacce160c13b3caf7b8bc92b`
passed 30/30, including eager and CUDA-graph exactness for k12/k16/k18 at both
release widths, both decode contracts, and every dictionary residency. Its
captured v1 direct-operator benchmark was bit-exact and measured
**1.8175–1.9977x** over inherited across the six cells. The same-process DSV4
model gate was valid and passed with exact zero full-vocabulary KL, NLL, PPL,
and target-logprob
deltas over 240 scored positions. Evidence:
`/home/rob/dq-runs/dsv4-flash-0731/mtp-throughput-research/gemv-v2-bitexact-operator-v4/{run.log,benchmark.json}`
and
`/home/rob/dq-runs/dsv4-flash-0731/mtp-throughput-research/gemv-v2-bitexact-quality-ab-v2/report.json`.
These are source-tree qualification results, not the final clean-wheel served
graph/throughput gate; that gate remains open.

**Routed FP8-CB whole-row sibling (default `auto` since 0.8.9,
`PRISMAQUANT_CB_FP8_GEMV_V2`).** This is independent of the FP4 selector
above and lives in the main `csrc/cb_gemv.cu` extension. The inherited FP8
grouped kernel assigns one CTA to one `(routed pair, output row)` and one
physical warp to each of its eight accumulator chains. The sibling stages the
complete packed row in warp-private shared memory, stages the exact 1 KiB K28
E4M3 LUT once per block, and lets each physical warp serve two output rows
while reproducing those same eight chains as named virtual accumulators. It
preserves both decode contracts: v1 still rounds `E4M3 * row_scale` to BF16
before each FMA, and v2 still applies the row scale once after the inherited
lane reductions and serial eight-chain sum.

The release surface is intentionally closed to
`k=28/n_sub=4/type_size=112` and K in `{2048,4096}` — the qualified cell the
evidence below measured. **Unset means `auto` since 0.8.9**: exactly that
cell takes the sibling and every other routed FP8-CB stack keeps the
inherited kernel with the reason logged per stack; `1`/`require` keeps the
pre-0.8.9 A/B-arm contract, under which any off-cell uniform FP8-CB stack
(or per-expert mixed FP8-CB group, whose family-local dispatch has not
implemented this sibling) fails the load instead of creating a partly
inherited candidate arm; `0`/`off` disables it. The selector resolves once
per process and each `w13`/`w2` decision is fixed at model load. FP4-CB
layers and source-passthrough groups are outside its scope. The main
extension's strict symbol set includes `cb_moe_gemv_fp8_v2`, so a cached
pre-sibling module is rejected before a forward can reach it.

For `cb_gemv.cu` SHA256
`3bf545f381acdbf503a9d78fb1bb9665b647ff0318fa27e7b24a5f96bbc26894`,
the final-source cc 12.1 operator run passed all 17 eager/registered-op,
CUDA-graph replay, fullgraph, and rejection tests for both decode contracts.
The exact dsv4flash0731 eager same-engine report at
`/home/rob/dq-runs/dsv4-flash-0731/mtp-throughput-research/routed-fp8-v2-quality-ab-v1/report.json`
has SHA256
`013ecf0efda1a707ead44fa9f57a94a017595aff2b65dc18cf142b97e8642314`
and records exact zero full-vocabulary KL/NLL/PPL/target-logprob,
generation-digest, and router-route delta over 240 scored positions. Its main
extension SHA256 is
`8f1f287e906562f152d9935deace4149dee3f0eb555d781a6400fe56e29d1104`.
An earlier served A/B/A2 using a different main binary showed an approximately
7.2% cycle-throughput signal but failed cross-arm content/acceptance
integrity, so it was held. **The final-binary served rerun (B-v3,
2026-08-14) closed it**: on a quiesced host with `cudagraph_mode=
FULL_DECODE_ONLY` and a route census confirming 16 stacks on `whole-row-v2`,
decode measured **18.442 vs 16.979 tok/s (+8.62 %)** with acceptance-rate
parity, and the earlier arm's memory cliff was attributed to host
interference (B-v3 bottomed at 5,228 MiB free over 3,532 samples). On that
record the selector defaults to `auto` on the qualified cell since 0.8.9.
Soak and high-concurrency protocols have not run as named gates; the sibling
serves only the routed `M ≤ 16` GEMV band, so long-prefill scheduling is
outside its reach.

MoE **prefill** used to be that per-expert loop, whose launch storm dominated
TTFT. The production quality lane now expands bounded expert chunks with native
CUDA and submits their ragged segments to a Gridbook-owned CUTLASS grouped BF16
GEMM. FP8-CB can additionally use its native E4M3/CUTLASS paths where their
contract applies. Missing expansion or grouped-GEMM support is fatal; it never
selects a Python loop or interpreted kernel.

The grouped-BF16 bridge is a generic SM80-compatible
`DefaultGemmGrouped`, not a Blackwell-optimized collective. On one GB10 it was
**6–17% slower on warm GPU time** than segmented BF16 matmuls across the recorded
synthetic DSV4 shapes. It removes an unowned serving dispatch and preserves the
quality contract, but it is not yet a prefill-speed result; the next optimization
target is a measured CUTLASS 3.x SM100/SM121 grouped collective — which now
exists as the opt-in lane described next.

### sm12x-native grouped BF16 (opt-in, `PRISMAQUANT_CB_BF16_SM120`)

`csrc/cb_bf16_grouped_gemm.cu` carries a **second lane** for the same bridge: a
CUTLASS 3.x collective on `arch::Sm120` with a TMA warp-specialized mainloop,
stages carved out of the 101,376-byte sm120 shared-memory budget, and the
plain `alpha=1/beta=0` epilogue the bridge has always had. It is the
2026-08-01 performance audit's §3 P1 item, and it exists because *every*
default NVFP4-CB prefill — dense `E=1` and routed MoE — plus the FP8-CB
fallback flows through this file.

**Design.** Upstream CUTLASS 4.3.4 has no sm120 ptr-array/grouped collective,
and its sm120 dense `CollectiveBuilder` refuses 16-bit input outright
(`static_assert(... "SM120 TmaWarpSpecialized builder currently only supports
F8F6F4 MMA")`, and `rr_op_selector_sm120` returns an 8-bit atom
unconditionally). The mainloop itself is type-generic, so the collective is
assembled by hand from the 16-bit forms of the four choices that builder makes:
the `SM80_16x8x16_F32BF16BF16F32_TN` bf16 tensor-core atom (register-sourced,
which is what this mainloop requires), a 2×2×1 **pingpong** warp layout,
`Tile<64,32,16>` so one `ldmatrix.x4` fills a thread's B fragment, and
CUTLASS's own K-major `rs_smem_selector` for a swizzled tile that TMA writes
and LDSM reads. Stages come from the builder's own `StageCountAutoCarveout`.

**Grouping** is the row-padded, TILE-INDEXED construction the two fused
kernels already use, extracted into `csrc/cb_grouped_common.hpp`: each
expert's rows span whole `TileM` blocks, B carries a batch mode of per-expert
stride `N*K` (exactly a contiguous `[E,N,K]` stack), and each M-tile reads
`expert_ids[m_tile]` as its B `l`-coordinate. The expert selection and the
A-side row sourcing live in a thin fork of the standard sm120 TMA mainloop
(`csrc/cutlass_fork/sm120_bf16_expert_mma.hpp`, four marked additions). No
packed-B or LUT machinery: B is plain BF16.

**Two A-source modes, one collective.** The ONE compiled kernel reads its A
tiles either way at runtime:

* *padded-copy mode* (`cb_bf16_grouped_mm_sm120[_out]`) — the caller
  materializes the row-padded `[Mp, K]` activation and the producer TMA-reads
  it, exactly the original construction;
* *in-mainloop gather mode* (`cb_bf16_grouped_mm_sm120_gather[_out]`) — the
  producer warp reads each padded row `m` from row `row_src[m]` of the
  COMPACT activation with predicated, zero-filling 16-byte `cp.async` (ids
  outside `[0, S)` are the padding rows), so the padded copy never exists.
  The pipeline accounting is upstream's own
  `sm120_mma_tma_blockwise_scaling.hpp` producer idiom (33 producer events:
  the TMA leader's `arrive_and_expect_tx` for the B-only transaction bytes
  plus one `cp.async` `noinc` arrival per lane).

The two modes load byte-identical smem tiles — padding rows are zeros either
way, and the gather predicate reproduces TMA's out-of-bounds K-residue
zero-fill — so their outputs are **bit-identical**, asserted with
`torch.equal` in `tests/test_bf16_grouped_cutlass.py`. The gather mode is
therefore not a requalification event: the lane's numerics class is pinned by
the padded mode's existing gate.

**Tile-order policy (swizzle-group-aligned expert packing).** The persistent
scheduler sweeps N with groups of `swizzle` M-tiles, so an expert whose tiles
straddle a group boundary has its B slice fetched from DRAM once per group
touched rather than once. `bf16_grouped_lane.pack_expert_blocks` orders
experts first-fit-decreasing on padded block counts so group boundaries
coincide with expert boundaries wherever the histogram allows — deterministic
host math on the routing histogram, telemetered as
`(groups_touched, groups_minimum)`. Tile order is scheduler order (the same
thing the swizzle argument already permutes): the bit gate asserts a packed
order is a pure block permutation of the natural one.

**Measured configuration** (GB10, cc 12.1, CUTLASS 4.3.4): **pingpong
`64×128×64`, 3 mainloop stages, 83,968 B** of the 101,376-byte budget, plus a
tile-scheduler swizzle of 1 below 64 padded M-tiles and 8 at or above (a
runtime tile-ORDER argument; it cannot move a bit). `TileM=64` is the whole
point of the rung and is why the **pingpong** kernel layer is used at all: the
cooperative layer static_asserts a 256-thread TiledMma whose 4×2×1 warp layout
floors `TileM` at 128, and `TileM` is the granularity each expert's rows are
padded up to. `256×128×64`, `128×256×64` and `128×64×128` each carve down to
ONE stage and are infeasible; every other feasible tile/stage/layer combination
was compiled and timed and none was better
([the sweep table](BENCHMARKS.md#what-was-swept)). The lane needs the `sm_12Xa`
arch-conditional target even though its MMA is architecture-generic: the
sm90-family *kernel layer* compiles its body only under
`__CUDA_ARCH_FEAT_SM12x_ALL` and otherwise aborts every launch. Built as plain
`sm_121` it compiles and loads, then faults; the loader comment records both
measurements.

**Requalification surface: FP32 reduction order, and nothing else.** Both lanes
accumulate in FP32 with no scales and round once to BF16 from the same
operands. What differs is the order of the accumulation (tile shape,
K-iteration, warp partitioning) — the same class of change the promoted FP8
mid-M fused kernel cleared. Measured across the shapes gated in
`tests/test_bf16_grouped_cutlass.py` plus DSV4/Laguna projections, both lanes
and a per-segment BF16 `F.linear` land on the *same* relative L2 against an
FP32 reference (1.612e-3 – 1.663e-3, ratio 1.0000), and the two lanes' BF16
outputs were bit-identical on every one of them. That is not a promise the
kernels make — bf16×bf16 products are exact in fp32, so the lanes differ only
in the ~2⁻²⁴ rounding of their partial sums, an order of magnitude below the
2⁻⁸ quantum of the bf16 result — so the tests gate a bound, not equality.
Within the sm12x lane, the gather mode and the tile-order policy sit BELOW
this surface entirely: gather-vs-padded is gated bit-equal, and a block
permutation of tile order cannot change any row's accumulation, also gated
bit-equal.

**Cost the construction adds, and how it was closed.** The construction
re-reads an expert's B slice once per padded M-tile, and at these shapes the
operator is bound by that traffic — so the 2026-08-01 measurements identified
two structural taxes: the rounding of each expert's rows up to `TileM`
(isolated: padding-free synthetic routing ran 1.08–1.13× segmented while real
ragged `T=512` routing ran 0.88×), and the padded activation gather. Both are
now closed at the construction level (2026-08-02): the **gather mode**
deletes the padded copy — its A stream reads the compact activation, which is
L2-resident at serving sizes, and padding rows load nothing (zero-fill) — and
the **swizzle-group-aligned expert order** removes the B-slice group-straddle
excess, measured 13.9–16.9% of GEMM time at `T=512` (39 groups touched
against the 32-group minimum with the natural order; packing reaches 32/32)
and neutral at `T=128`, where the grid sits below the swizzle threshold. A
`TileM` ladder was evaluated against the sweep record and is measured-dead
for these cells: every 128-row tile landed at ≤ 0.97× segmented because L2
already recovers same-expert B reuse while the coarser rounding is real work.
The routed path still pays one host read of the per-expert block offsets
(the decoded weight transient is chunked over experts), which is also where
the packing order gets its histogram for free.

Both are **wired through the routed dispatch**, not merely available: stage
one of `moe._apply_prefill_native_bf16_sm120` calls the gather entry point
with the routing's own `dest` vector as `row_src` (real rows name their
token, padding rows name the throwaway row `T`, and the kernel reads zeros for
any id outside `[0, T)`), so the `[Mp, K]` copy is not built at all. Stage two
deliberately stays in padded mode — its A operand *is* the padded intermediate,
so there is no compact form to gather from. The packed expert order is applied
only when one expert chunk covers every expert; a narrower chunk indexes blocks
as `block_offsets[c0]..block_offsets[c1]` and therefore assumes expert-major
contiguity, which a permutation would break.

**Measured against its target: met on every cell at both token counts.** The
P1 target was "≥ segmented-BF16 parity warm". With the gather mode and the
packed order (GB10, whole operator, warm medians): **1.032–1.051× segmented
at `T=128` and 1.102–1.151× at `T=512`**, beating the SM80 bridge it
replaces by 1.133–1.221× and 1.177–1.370× respectively — the `T=512` cells
that previously reached only 0.83–0.92× now clear parity by 10–15%. Full
tables, the isolated tile-order effect, and the padded-copy mode's
historical numbers:
[BENCHMARKS](BENCHMARKS.md#2026-08-02-sm12x-grouped-bf16-lane-in-mainloop-a-row-gather--swizzle-aligned-tile-order-proposal-data).

**Status: OPT-IN** behind `PRISMAQUANT_CB_BF16_SM120=1`, resolved at model load
(never at first forward) and failing the load if the flag is on where the lane
cannot serve. With the flag unset the dispatch is byte-for-byte what it was.
**Promotion checklist:** bit-level unit gate — **done**, including the
gather-vs-padded bit-identity gate and the tile-order permutation gate (the
file above); kernel-level speed target — **met on all cells at `T=128` and
`T=512`**, measured (above); whole-routed-operator
[NATIVE-PARITY](NATIVE-PARITY.md) protocol — **not run**.
`scripts/bench_bf16_grouped_sm120.py` produces proposal data only.

### Persistent-B decode-in-mainloop (default auto, `PRISMAQUANT_CB_MOE_PERSISTENT_B`)

ROADMAP **K1.1**, audit §3 **P2b**. Both lanes above still *materialize* the
decoded weights: `cb_expand_fp4_v2` writes an `[E,N,K]` BF16 tile to HBM and
the grouped GEMM reads it back. That expansion runs over **every** expert,
routed or not, at 2 bytes/weight — 4× FP8-CB's direct-to-E4M3 transient, and
~35% of MoE layer time at Laguna scale. This kernel
(`csrc/cb_moe_persistent_b.cu`) deletes it: the packed CB bytes (~0.3
B/weight) are read once, decoded to BF16 **in shared memory inside the
mainloop**, and consumed in place.

**Design — persistent-B along M.** A CTA owns one `(expert, N-tile)` work
unit. It reads that expert's exact routed segment from `expert_ends` and loops
the segment in `TM`-row M-tiles **inside the kernel**, so the per-(expert,
N-tile) setup is hoisted out of the M loop and each decoded weight value feeds
`TM` activation rows before it is discarded. This is the inverse of the two
fused lanes, whose CTAs are *M-tile-indexed*: there a second M-tile of the
same expert is a different CTA that re-decodes B from scratch, which is why
FP8-CB's fused lane is gated at `M≤128` and measures 0.22× at `M≈1400`.

**It is not the retired persistent-N kernel.** That one owned a bare N-slice
per CTA and walked M with CUDA-core/SM89-class atoms on the **dense** path.
This is MoE-only (every binding takes `expert_ids`/`expert_ends`; there is no
dense entry point in the translation unit, and `cb_moe_persistent_b_is_moe_only()`
is gated by a test), it uses `mma.sync.m16n8k16` tensor cores, and its win
comes from amortizing a **decode** the dense kernel never had. Dense large-M
stays behind [`K1.3`](../ROADMAP.md#kernel-todo-canonical).

**Grouping: exact segments, not padded tiles.** Unlike every CUTLASS lane in
this tree, the mainloop is hand-assembled from `mma.sync` / `ldmatrix` /
`cp.async`, so it is not bound by a uniform tile that cannot straddle two
experts. It therefore consumes the **exact** `expert_ends` segments the
default quality path already builds — same stable argsort, same
`cumsum(bincount(...))`, same `index_select`/`index_add_` combine. Consequences:
zero padded rows (the padded layout can inflate a 4,096-row `E=256` prefill by
up to `256×(TileM−1)` rows), and **zero host reads** — the padded lanes each
spend one `.item()` on the real block total, this path spends none, because
the launch geometry is `E × ceil(N/TN)` CTAs, a function of the layer shape
alone. Empty experts cost two int32 loads and a CTA return. Skew costs a
longer CTA, never a serialized grid.

**Measured configuration** (GB10, cc 12.1; reproduce with
`csrc/tools/persistent_b_probe.cu`). `TK` is fixed at 64 BF16 columns = 128 B
per row = the 8 sixteen-byte chunks an XOR swizzle needs to make `ldmatrix`
conflict-free over 8 rows, and it divides the 256-column CB superblock evenly.
Bytes quoted at `k=24`, the widest packed superblock:

| cfg | TM | TN | warps | A | B | packed | smem | CTAs/SM | accum regs/thread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 64 | 8 | 32,768 | 8,192 | 7,424 | 48,384 | 2 | 32 |
| 2 | 64 | 64 | 4 | 16,384 | 8,192 | 7,424 | 32,000 | 3 | 32 |
| 3 | 128 | 32 | 4 | 32,768 | 4,096 | 3,712 | 40,576 | 2 | 32 |
| 4 | 64 | 128 | 8 | 16,384 | 16,384 | 14,848 | 47,616 | 2 | 32 |

Every compiled config holds **≥2 CTAs/SM including the ~1 KiB the hardware
reserves per CTA**, and the binding `TORCH_CHECK`s exactly that — the lesson
W4 paid for when its first collective wanted 75,776 B and got one CTA per SM.
Two wider tiles were compiled, measured and **dropped**: `128×128` (64,000 B)
and `256×64` (81,152 B) both fall to 1 CTA/SM and neither won a single sweep
cell — `256×64` halves the decode repetition at large rows-per-expert, exactly
the regime it should own, and still lost there (18.6 vs 7.8 ms, DSV4 `w2`,
`T=2048`). Occupancy dominates decode amortization on this device. A resident
codebook in shared memory (`cb_gemv_v2.cu`'s DS=2 idea) was measured and
**rejected** for the same reason: the 4 KiB `k=16` table pushed the default
config from 46,336 to 50,432 B, which is under half the 102,400 B SM budget
but not once the per-CTA reservation is counted, costing 1.9× (4.09 → 7.76 ms)
while buying ~1% where it did fit. The gathers already hit L1 at the rungs
where the table is small.

**Tile selection is a function of SHAPES, never of a routing value.** `cfg=0`
(the production setting) picks from `P/E`, the mean routed rows per expert,
where `P` is `a.size(0)` and `E` is `qw.size(0)` — both known to the host
without touching device memory, so the choice stays a trace-time constant and
the launch is capturable as-is. Measured crossover: cfg 4 below ~64 mean rows,
cfg 1 above.

**The FP8-CB arm (K1.1's second payload family).** The same schedule serves
stock FP8-CB routed layers, behind the same flag: the decode stage gathers the
`n_sub=4` ragged codewords through the exact FP32 table torch converts from
the E4M3 book at load (`e4m3->f32` is definitionally torch's own conversion),
applies the per-(expert, output-row) FP32 scale, and rounds once to BF16 —
the exact value chain of the bridge expander's FP8 branch, so here too only
the FP32 reduction order changes. Eligibility at this device's shared-memory
budget: cfg 1 to k≤33, cfg 4 to k≤31, cfgs 2/3 to k=48 (every compiled config
must hold the 2-CTAs/SM floor at the layer's `type_size`); layers whose `w13`
splits gate/up into **two per-role books** are outside the arm — the decode
consumes one stacked stock book per projection — and under `auto` such layers
keep the bridge with the reason announced (the shipped DSv4 87 GB body's 11
FP8-CB expert layers are exactly this case), while an explicit `require`
fails the load by name. Bitwise decode identity against `cb_expand_fp8` is
tested across the rung set, and the whole-routed-operator microbenchmark at
DSv4 shapes (E=256, K28, topk 6) measures **15.8–18.4×** the expand+bridge
operator with rel-L2 ≤ 6.2e-04 (reduction order only). There is no FP8 D2R
variant.

**Requalification surface.** Activations are untouched — the same exact
group-16 RTN QDQ payload, before FC1 and between FC1 and FC2. The decode is
**bit-identical to `cb_expand_v2`**, and that is a tested fact rather than an
asserted one: `cb_moe_persistent_b_decode` exposes the mainloop's own decode
stage and the suite compares it to the expander with `torch.equal` across the
`k=12…24` rungs, on both full and windowed row ranges. Accumulation is FP32
with one BF16 round. What changes is the FP32 **reduction order** —
reassociation-class, the same surface the sm12x lane and the promoted FP8
mid-M fused kernel cleared.

**Status: DEFAULT (`auto`) since 0.8.9**, both payload families, resolved at
model load (never at first forward). Auto engages each routed CB layer's
family arm where the load-time predicate and the extension attest, and keeps
the expand+bridge route where they do not, with a per-layer fallback line
naming why — the bridge is exactly the pre-0.8.9 default, so degrading is
correct, only slower. `1`/`require` keeps the pre-0.8.9 explicit semantics:
a layer no arm can serve fails the load by name (the A/B-integrity
contract), including the tile override `PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG`,
which is validated against what this build actually compiled. `0`/`off` is
the kill switch, byte-for-byte the pre-lane dispatch. **Promotion record:**
decode bit-exactness gate — **done**; whole-operator numerics under the
reduction-order discipline — **done**; routing breadth (empty / single /
one-row / skewed) — **done**; graph capture-replay and non-default-stream —
**done**; whole-routed-operator microbenchmark — **done, wins every cell**:
1.05–3.36× over the default bridge and 1.04–3.02× over the pingpong bridge
across nine whole-operator cells, with the eliminated expansion measuring
20.9–46.7% of the default operator
([table](BENCHMARKS.md#2026-08-02-persistent-b-grouped-moe-decode-in-mainloop-microbenchmark-proposal-data));
served [NATIVE-PARITY](NATIVE-PARITY.md) on the whole routed operator —
**run**: the FP4 arm's same-session served A/B on the DSv4 92 GB body
(kl_mean −0.051 %, direct PPL −0.30 % — arithmetic noise), and the 0.8.9
default-state served KL/PPL leg on the shipped clean 87 GB body against its
gold record (32 FP4-CB layers on the lane, 11 per-role FP8-CB layers on the
announced bridge fallback: kl_mean +0.17 %, kl_p99 −0.03 %, direct PPL
−0.06 % — inside the ±0.7 % cross-session KL envelope). `scripts/bench_moe_persistent_b.py` produces
proposal data only.

The ratio narrows as mean routed rows per expert grows — the kernel decodes a
weight tile once per `TM`-row M-tile, so at `P/E = 512` it decodes four times
where the expansion decodes once and the costs nearly cancel (1.05×). The
production-shaped `E=128` cells hold 1.98–3.11×. Raising `TM` past 128 without
losing the second CTA per SM is the identified next step, and it is a change of
tile, not of schedule.

MoE dispatch is separate from the dense M boundary: `M≤16` uses the owned
grouped CUDA GEMV. Above 16, each routed CB layer runs the decode-in-mainloop
schedule above wherever its family arm attests (the 0.8.9 `auto` default,
which replaces both halves of the expand+bridge pair); a layer the lane
cannot serve falls back to exact BF16 expansion plus the owned CUTLASS
grouped bridge — for FP8-CB via its quality-green fused CUTLASS path first
where that contract applies. Only expert chunk size / transient-byte-budget
overrides remain—there is no stock/loop/ batched/L2 production selector.

The predecessor native chunk-expander path measured **293 → 1,821 tok/s at 8k**
and **207 → 1,822 tok/s at 63k** on Laguna-S-2.1 (117B MoE). Those numbers are
preserved as historical evidence and are not yet a benchmark of the new owned
CUTLASS grouped bridge. Chunked prefill re-expands per microbatch, so
`--max-num-batched-tokens 16384` remains workload-relevant.

The native-W4A4 grouped experiment remains excluded from the production quality
lane: switching from Gridbook's established activation QDQ to native UE4M3
activation factors materially changed the served distribution. It must not be
promoted on kernel parity or raw speed alone; see the
[fused-NVFP4 enablement audit](audits/fused_nvfp4_enablement_2026-07-31.md).

The activation-contract-preserving decode-in-mainloop schedule that target
called for is implemented and, since 0.8.9, the default (`auto`) route for
routed CB quality prefill (the section above); the served promotion gate that
kept [`K1.1`](../ROADMAP.md#kernel-todo-canonical) open has run.

---

## Native-kernel availability and failure policy

Every production Gridbook operation has a native CUDA/CUTLASS implementation.
Capability probes may return "unavailable" while inspecting an installation,
but a serving call that needs that operation raises immediately with the source,
toolchain, cache, or capability problem. There is no interpreted-kernel fallback,
and Gridbook does not depend on Triton.

The retired Triton prototypes remain relevant only to dated measurements: for
example, the initial dense decode prototype measured 4.20 tok/s before the CUDA
GEMV reached 10.28 tok/s. Those results explain the production policy; they do
not describe a backend that current Gridbook can select. CPU/PyTorch references
in tests continue to validate format semantics without entering serving.

---

## CUDA-graph safety rules

CUDA-graph capture removes per-kernel launch overhead, but it is fragile against
data-dependent control flow. The kernels follow these rules:

1. **Every kernel is a registered custom op** with a `fake`/meta implementation
   (`direct_register_custom_op` + `fake_impl`). This makes each kernel opaque to
   `torch.compile`/Dynamo (no graph break inside it) and lets compiled and eager
   serving produce identical output.
2. **No host-side branching on tensor values inside a captured region**, and fixed
   shapes. Bit-exactness with capture **off** must always hold (capture-on ==
   capture-off logits). The #47 corollary: a data-dependent host *read* is wrong
   under capture even where the runtime would permit it, because a replay
   recomputes the tensor but keeps the captured host value — the padded grouped
   routing's optional trim count is exactly such a read, so under capture
   `_padded_route` launches the data-independent static-capacity layout instead,
   and the opt-in sm12x expert-chunked BF16 bridge (`PRISMAQUANT_CB_BF16_SM120`,
   whose launch bounds are irreducibly routing-dependent host values) refuses
   capture outright. The default expand + grouped bridge and the persistent-B
   lane never take a host read and capture as-is.
3. **All device-side constants and per-device kernel setup happen once, at model
   load.** A real bug this caught: an activation-QDQ kernel built its FP4/E2M1
   grid on the CPU and H2D-copied it *every call* — a hidden sync in eager mode
   and a hard error under capture. The grid is now cached per device. The same
   rule covers the **99 KiB dynamic-shared-memory opt-in**, which is a
   `cudaFuncSetAttribute` call and therefore not stream-ordered work: a lazy
   first-launch setup leaves it to whichever call happens to be first, which for
   a prefill path can be the inside of a capture. `cb_gemv_v2_prepare` and
   `cb_moe_persistent_b_prepare` are separate entry points that the loaders call
   during `process_weights_after_loading`, so every compiled configuration is
   prepared before any forward or capture runs; the launchers re-call them
   behind a per-device atomic, which in steady state is one load.
4. **Hoist the whole M-gated dispatch behind one opaque op.** The retired
   pre-hardening host branch was capture-hostile: a prefill-sized trace could
   bake the expand arm into decode. Every current Linear/MoE/source-BMM call
   permanently crosses one opaque `cb_*_forward` op whose eager implementation
   resolves M at capture time. A `FULL_DECODE_ONLY` capture records the GEMV arm
   at each fixed actual flattened token count `M <= 8`; a source-BMM group count
   is not folded into M. Prefill stays outside that graph and resolves the
   large-M arm independently. There is no environment switch back to the
   retired branch.
5. **Use full-decode graphs without compilation over the plugin.** The validated
   shape is `mode=0`, `cudagraph_mode=FULL_DECODE_ONLY`, capture sizes
   `[1,2,4,8]`. No Gridbook op carries `torch.Tag.cudagraph_unsafe`, so nothing
   partitions the captured region — Gridbook's whole-dispatch ops are opaque
   graph nodes and are capture-safe by construction. On a close-rate
   0.6B canary, changed inputs at capture sizes 1 and 4 matched eager text,
   tokens, and per-token logprobs exactly; 32+256 latency improved 20.1% and was
   5.9% behind native. This is directional A8-vs-W4A4 execution-contract
   evidence, not an exact-byte 27B claim. The FP4-v2 295B path separately
   measured +24% decode. Capture only the batch sizes the deployment needs:
   the 0.6B validation reported ~30 s startup and 2.64 GiB for four sizes.
   Those served measurements predate source-FP8 W8A16 and must not be cited as
   its exact-artifact graph evidence.

## A measurement side-effect worth knowing

Loading *any* additional CUDA extension into the serving process shifts the
allocator's addresses, which changes alignment-sensitive kernel dispatch elsewhere
in the model and perturbs floating-point reduction order. On the 27B artifact this
produced a **±17%** swing in the confident-KL *evaluation* (both readings still far
better than the baseline). It was not a bug in the historical FP8-CB operators:
the expanded and fused prefill arms compared there were bit-identical offline.
That statement does **not** apply to native-fused FP4, which changes the
activation-scale contract. The allocator effect is nevertheless the concrete
mechanism behind cross-session KL drift. When running an A/B, **match extension
residency across arms** or the comparison is confounded. This matters more for
benchmarking than serving; it is documented in [`BENCHMARKS.md`](BENCHMARKS.md)
too.

## Status summary

| Path | Status |
|---|---|
| Source block-FP8 W8A16, dense + DSV4 grouped `wo_a` | **Implemented in 0.8.5; target-only DSV4 candidate load/generation passed on the exact 0.8.6 runtime; final clean-commit replay pending.** Raw E4M3/UE8M0 stays resident; BF16 activations are unchanged; actual flattened decode-token `M<=8` uses native CUDA GEMV and larger M uses bounded native BF16 expansion plus the owned CUTLASS grouped bridge. Grouped BMM is admitted only at `(G=8,N=1024,K=4096,tp=1)`; dense geometry is gated separately and also remains `tp=1` only. Direct MXFP8 g32 remains the separate opt-in W8A8 lane. DSpark is experimental and is not the release default; do not claim its served parity, graph, or 128k performance until the separate gates in RELEASING are recorded |
| FP8-CB decode (dense) | **Shipped**, at/above native parity |
| FP4-CB v2 decode (dense) | **Shipped**: bit-matched CUDA GEMV (13/13 parity against the historical Triton result plus the independent expansion reference). The decode chain is compute-bound at GEMV shapes (ncu SM 71%/mem 44%) under the bit-exact contract — the measured ceiling, not a staging problem |
| MoE grouped decode GEMV | **Shipped**: fp8 66–95% of peak; fp4-v2 w2 schedule redesigned (+50%, 37–47% of peak; reassociation served-gated with an env-switched legacy path). A rowpack variant measured NEGATIVE and stays opt-in-off as a documented result |
| MoE grouped decode GEMV, smem-resident dictionary | **Default `auto` since 0.8.9** (`PRISMAQUANT_CB_GEMV`; `inherited` is the kill switch and never probes or builds): v2 engages only where the compiled occupancy predicate says it wins, and an unavailable extension degrades loudly to inherited. On the K=2048/4096 shapes (8/16 superblocks) the compile-time virtual-warp specialization is bit-exact to the inherited default for every rung — with all `PRISMAQUANT_CB_W2_*` overrides absent — and measured 1.8175–1.9977x in the final-source captured v1 direct-op benchmark; the same-process DSV4 quality gate produced zero full-vocabulary KL/NLL/PPL/target-logprob delta over 240 positions. Other widths retain rowpack order and may reassociate against inherited — on those the predicate is the only gate and no served A/B exists. The compiled predicate still routes k24 at K≥2048 to inherited. Historical Laguna validation measured +5.97% |
| MoE grouped decode GEMV, routed FP8 whole-row sibling | **Default `auto` since 0.8.9** (`PRISMAQUANT_CB_FP8_GEMV_V2`) for exactly K28/4/112 at K=2048/4096 — off-cell stacks keep the inherited kernel with the reason logged; `1`/`require` is the strict A/B-arm spelling under which unsupported uniform FP8 cells and per-expert mixed FP8-CB groups fail closed. The final-source operator gate passed 17/17; the exact-artifact eager same-engine gate produced zero full-vocabulary and router-route delta over 240 positions; the final-binary served rerun (B-v3, 2026-08-14, FULL_DECODE_ONLY graphs, route census, quiesced host) measured **+8.62% decode** with acceptance parity, attributing the earlier held signal's integrity failure to host interference. Soak and high-concurrency protocols have not run as named gates |
| Transient-expand prefill (dense) | **Shipped**; ~1.44× native at large M (traffic-bound) |
| FP8-CB fused decode-in-prologue prefill | **Bit-exact; dispatch-eligible at M=9–128 and measured wins at M=32/64/128**; native transient expansion + CUTLASS serves ineligible and large-M shapes. **Rung surface: `k ∈ {28,32,36,40,44,48}` — the multiples of 4 in the product range, which is the complete set the packed-B TMA box (`type_size = 4k` must be 16-byte aligned) and the mainloop's uniform `CbSubW = k/4` sub-table width admit.** The other 15 integer rungs are served (decode GEMV + expand/CUTLASS bridge), just not by this lane; the compiled set is reported by `cb_fused_kbits()` and dispatch derives eligibility from it rather than from a literal ladder |
| FP4-CB v2 fused mid-M prefill (dense) | **Opt-in** (`PRISMAQUANT_CB_FP4_FUSED_MIDM`), contract-preserving. The in-prologue decode is proven **bit-identical to `cb_expand_v2` for the whole decoded tile at all 13 K12–K24 rungs** (one-hot read-out, no tolerance), so only the FP32 reduction order moves. Measured 1.06–4.37× the shipping expand + bridge route at M ∈ {9,16,32,64,128} on 27B/DSV4-class shapes, every cell bit-equal to the same-config oracle — a wider band than the fp8 twin because the fp4 quality expand writes BF16 (4× the fp8 expand's transient bytes). Served NATIVE-PARITY protocol **not run**; an unexplained M ≤ 12 latency cliff is open. Ineligible shapes (M ≤ 8, M > 128, multi-dictionary fused modules, uncompiled rungs) fall through to today's exact path unchanged |
| NVFP4-CB fused native-FP4 prefill (dense and MoE) | **Explicit opt-in**: shared `static_lsq` activation policy, optimized v2 scale decode, and occupancy-aware TileM routing. Short exact K24 quality/performance passed; long-context evidence is mixed; raw native parity, >=4B, task, MoE routed-quality, and p95 served gates remain open |
| MoE persistent-B decode-in-mainloop prefill (FP4-CB v2 + stock FP8-CB) | **Default `auto` since 0.8.9** (`PRISMAQUANT_CB_MOE_PERSISTENT_B`; `1`/`require` keeps the fail-load A/B-integrity semantics, `0`/`off` the kill switch), ROADMAP K1.1 both payload families. Decodes each expert weight tile once and streams that expert's exact routed segment through it, eliminating the `[E,N,K]` BF16 transient and all work for unrouted experts. Weight decode bit-identical to the expanders (`torch.equal`); reduction-order-class output difference vs the bridge. Whole-routed-operator microbenchmark wins every cell (FP4 1.05–3.36×; FP8 15.8–18.4× at DSv4 shapes); served: FP4 same-session A/B kl_mean −0.051% / PPL −0.30%, plus the 0.8.9 default-state served KL/PPL leg on the shipped clean 87 GB body. Per-role FP8-CB split books are outside the FP8 arm and keep the bridge under auto (announced). **Staging vectorized (2026-08-21)**: byte-granular packed-superblock copies → u32 interior + funnel-shift edges (odd `4k+9` phases) + whole-slot zeroing, byte-neutral per the staging theorem; helpers `__noinline__` because inlining cost ~26 registers/thread and measured a ~23% whole-operator regression |
| Persistent-N large-M dense prefill | **RETIRED FROM SERVING; MEASURED NEGATIVE**: parity-green, but 2–5.7× *slower* than expand-then-GEMM at 27B shapes — the CUDA expander had already cut the dense expand tax to ~10%, removing the opportunity. The serving selector, custom op, package loader, and switch are deleted. The `.cu` remains accessible only to the explicit direct research test. The equivalent idea for **MoE** is still open and is the roadmap's next kernel |
| MoE prefill | **Native-only production lane**: grouped CUDA GEMV at M≤16; above 16, the persistent-B decode-in-mainloop lane wherever its family arm attests (the 0.8.9 `auto` default), else eligible quality-green FP8 fused CUTLASS or exact CUDA expansion + owned CUTLASS grouped GEMM. The current generic SM80-compatible grouped bridge is not Blackwell-optimized and measured 6–17% slower than segmented BF16 matmuls on warm synthetic DSV4 shapes; an sm12x-native CUTLASS 3.x collective for the same bridge exists **opt-in** (`PRISMAQUANT_CB_BF16_SM120`, pingpong 64×128×64, in-mainloop A-row gather + swizzle-group-aligned expert order): measured 1.13–1.37× that bridge and **1.03–1.05× segmented matmuls at T=128, 1.10–1.15× at T=512** on the DSV4/Laguna cells — the ragged row-padding tax the tile-indexed construction previously paid (0.83–0.92× segmented at T=512) is closed at the construction level; bit-gated (the gather mode is bit-identical to the padded-copy mode), served protocol not run. **Per-chunk packing landed 2026-08-21 (ROADMAP K1.5)**: multi-chunk layers now apply swizzle-group ordering within each chunk's own expert range — outputs bit-identical (chunk boundaries split only the expert dimension), isolated stage-one gather −10.3…−11.0% on straddling segments, inert on uniform-router control. Historical Laguna-S-2.1 measurements for the predecessor native chunk-expander path were 293 → 1,821 tok/s @8k and 207 → 1,822 @63k; they must not be relabelled as measurements of the new bridge. The v0.4.2 fused native-NVFP4 route remains an explicit A/B opt-in without MoE quality qualification |
| Missing required native kernel | **Fails closed** with an operation-specific diagnostic; no Triton dependency or serving fallback |
