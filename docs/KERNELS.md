# Serving kernels

This document describes how a `gridbook` artifact is served fast. It is a design and
status document, not an API reference; the normative decode semantics are in
[`SPEC.md`](SPEC.md). The single live implementation checklist is the
[`ROADMAP.md` kernel TODO](../ROADMAP.md#kernel-todo-canonical); status prose
here is evidence and design context, not a second backlog.

Terminology used below: **GEMM** = matrix-matrix multiply (prefill / large batch);
**GEMV** = matrix-vector multiply (decode / batch-1 or small batch); **M** = the
number of activation rows (tokens) in a matmul; **MMA** = a tensor-core
matrix-multiply-accumulate instruction; **QDQ** = quantize-then-dequantize the
activations; **smem** = GPU shared memory; **TTFT** = time to first token
(prefill latency).

## The three invariants everything is built around

- **INV-1 — no resident expansion.** The resident weight is always the packed
  `cb_qweight` (indices + scale plane) plus the small shared codebook. The dense
  weight is **never** materialized in memory. Decoding happens in registers/smem
  per tile, or into a per-layer scratch buffer that is freed after the matmul.
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
PR's execution battery covers cc 12.1, not cc 12.0. It
is **not** bit-exact against the default
grouped schedule — a native schedule-reassociation class — and it is
**not** a default: unset means `inherited`. Explicit `auto` or `v2` reaches it
only where the hardware gate and compiled occupancy predicate both pass;
`PRISMAQUANT_CB_GEMV=inherited` is the kill switch. The
selection resolves once per process and is fixed per (layer, stack) at load, so
the call-site branch is a trace-time constant and FULL-decode cudagraphs are
unaffected.

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
kernels already use, extracted into `csrc/cb_grouped_common.hpp`: the caller
pre-gathers and PADS activation rows so each expert's segment spans whole
`TileM` blocks, B carries a batch mode of per-expert stride `N*K` (exactly a
contiguous `[E,N,K]` stack), and each M-tile reads `expert_ids[m_tile]` as its
B `l`-coordinate. That one selection lives in a thin fork of the standard
sm120 TMA mainloop (`csrc/cutlass_fork/sm120_bf16_expert_mma.hpp`, three marked
additions). No packed-B or LUT machinery: B is plain BF16.

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

**Cost the lane adds, and what bounds it.** The construction re-reads an
expert's B slice once per padded M-tile, and at these shapes the operator is
bound by that traffic — so the rounding of each expert's rows up to `TileM` is
the dominant cost, not the schedule. Isolated by measurement: with the padding
removed (a synthetic routing whose expert counts are exact tile multiples) this
collective runs **1.08–1.13× faster than segmented cuBLAS**; with real ragged
routing at `T=512` (1.25× padded rows) it runs 0.88×. The lane also pays a
padded activation gather the exact-segment lane does not (an activation-side
transient of the same class the promoted routed fused path already allocates —
never a second weight copy) and one host read of the per-expert block offsets,
because the decoded weight transient is chunked over experts and each chunk
launches over its own contiguous block range.

**Measured against its target: met at `T=128`, missed at `T=512`.** The P1
target was "≥ segmented-BF16 parity warm". At `T=128` — the token count of the
published DSV4 bridge measurement — the lane clears it on all four
DSV4/Laguna cells (1.019–1.051× segmented) while beating the SM80 bridge it
would replace by 1.18–1.27×. At `T=512` it beats that bridge on three of four
cells but reaches only 0.83–0.92× of segmented, and at `T=2048` only the
short-`K` cell reaches parity. The deficit is the ragged padding tax quantified
above, which no tile/stage/layer/raster/swizzle combination removes at a fixed
`TileM`. Full tables, the packed-vs-ragged isolation and the complete sweep:
[BENCHMARKS](BENCHMARKS.md#2026-08-01-sm12x-native-grouped-bf16-lane-microbenchmark-proposal-data).

**Status: OPT-IN** behind `PRISMAQUANT_CB_BF16_SM120=1`, resolved at model load
(never at first forward) and failing the load if the flag is on where the lane
cannot serve. With the flag unset the dispatch is byte-for-byte what it was.
**Promotion checklist:** bit-level unit gate — **done** (the file above);
kernel-level speed target — **met at `T=128`, missed at `T=512`**, measured
(above); whole-routed-operator [NATIVE-PARITY](NATIVE-PARITY.md) protocol —
**not run**. The identified next step is a change of CONSTRUCTION rather than
schedule: a `TileM` ladder selected by measured rows-per-expert (the fused
lanes already have one, and the `tile_m` binding plus the dispatch helper
already parameterise it), or an A-side row gather inside the mainloop that
removes the padded copy entirely. `scripts/bench_bf16_grouped_sm120.py`
produces proposal data only.

### Persistent-B decode-in-mainloop (opt-in, `PRISMAQUANT_CB_MOE_PERSISTENT_B`)

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

**Requalification surface.** Activations are untouched — the same exact
group-16 RTN QDQ payload, before FC1 and between FC1 and FC2. The decode is
**bit-identical to `cb_expand_v2`**, and that is a tested fact rather than an
asserted one: `cb_moe_persistent_b_decode` exposes the mainloop's own decode
stage and the suite compares it to the expander with `torch.equal` across the
`k=12…24` rungs, on both full and windowed row ranges. Accumulation is FP32
with one BF16 round. What changes is the FP32 **reduction order** —
reassociation-class, the same surface the sm12x lane and the promoted FP8
mid-M fused kernel cleared.

**Status: OPT-IN** behind `PRISMAQUANT_CB_MOE_PERSISTENT_B=1`, resolved at
model load (never at first forward), FP4-CB-v2 layers only, and failing the
load if the flag is on where the lane cannot serve — including the tile
override `PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG`, which is validated against
what this build actually compiled. With the flag unset the dispatch is
byte-for-byte what it was. **Promotion checklist:** decode bit-exactness gate
— **done**; whole-operator numerics under the reduction-order discipline —
**done**; routing breadth (empty / single / one-row / skewed) — **done**;
graph capture-replay and non-default-stream — **done**; whole-routed-operator
microbenchmark — **done, wins every cell**: 1.05–3.36× over the default bridge
and 1.04–3.02× over the pingpong bridge across nine whole-operator cells, with
the eliminated expansion measuring 20.9–46.7% of the default operator
([table](BENCHMARKS.md#2026-08-02-persistent-b-grouped-moe-decode-in-mainloop-microbenchmark-proposal-data));
whole-routed-operator served [NATIVE-PARITY](NATIVE-PARITY.md) protocol —
**not run**, and that is what stands between this lane and a default.
`scripts/bench_moe_persistent_b.py` produces proposal data only.

The ratio narrows as mean routed rows per expert grows — the kernel decodes a
weight tile once per `TM`-row M-tile, so at `P/E = 512` it decodes four times
where the expansion decodes once and the costs nearly cancel (1.05×). The
production-shaped `E=128` cells hold 1.98–3.11×. Raising `TM` past 128 without
losing the second CTA per SM is the identified next step, and it is a change of
tile, not of schedule.

MoE dispatch is separate from the dense M boundary: `M≤16` uses the owned
grouped CUDA GEMV. Above 16, FP8-CB first attempts its quality-green fused
CUTLASS path and otherwise uses exact BF16 expansion plus the owned CUTLASS
grouped bridge; FP4-CB uses exact BF16 expansion plus that bridge, or — behind
`PRISMAQUANT_CB_MOE_PERSISTENT_B=1` — the decode-in-mainloop schedule above,
which replaces both halves of that pair. Only expert chunk size /
transient-byte-budget overrides remain—there is no stock/loop/ batched/L2
production selector.

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
called for is now implemented behind `PRISMAQUANT_CB_MOE_PERSISTENT_B=1` (the
section above); what remains open on [`K1.1`](../ROADMAP.md#kernel-todo-canonical)
is the served promotion gate, not the kernel.

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
   capture-off logits).
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
   bake the expand arm into decode. Every current Linear/MoE call permanently
   crosses one opaque `cb_*_forward` op whose eager implementation resolves M at
   capture time. A `FULL_DECODE_ONLY` capture records the GEMV arm at each fixed
   decode size; prefill stays outside that graph and resolves the large-M arm
   independently. There is no environment switch back to the retired branch.
5. **Use full-decode graphs without compilation over the plugin.** The validated
   shape is `mode=0`, `cudagraph_mode=FULL_DECODE_ONLY`, capture sizes
   `[1,2,4,8]`, and `PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE` unset. On a close-rate
   0.6B canary, changed inputs at capture sizes 1 and 4 matched eager text,
   tokens, and per-token logprobs exactly; 32+256 latency improved 20.1% and was
   5.9% behind native. This is directional A8-vs-W4A4 execution-contract
   evidence, not an exact-byte 27B claim. The FP4-v2 295B path separately
   measured +24% decode. Capture only the batch sizes the deployment needs:
   the 0.6B validation reported ~30 s startup and 2.64 GiB for four sizes.

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
| FP8-CB decode (dense) | **Shipped**, at/above native parity |
| FP4-CB v2 decode (dense) | **Shipped**: bit-matched CUDA GEMV (13/13 parity against the historical Triton result plus the independent expansion reference). The decode chain is compute-bound at GEMV shapes (ncu SM 71%/mem 44%) under the bit-exact contract — the measured ceiling, not a staging problem |
| MoE grouped decode GEMV | **Shipped**: fp8 66–95% of peak; fp4-v2 w2 schedule redesigned (+50%, 37–47% of peak; reassociation served-gated with an env-switched legacy path). A rowpack variant measured NEGATIVE and stays opt-in-off as a documented result |
| MoE grouped decode GEMV, smem-resident dictionary | **Opt-in** (`PRISMAQUANT_CB_GEMV=v2`), never a default. Wins 1.13–1.58× on k13/k16/k20 in a 16-cell GB10 sweep; loses on k24 at K≥2048 (occupancy wall), where a compiled predicate routes the cell back to the shipped kernel. Reassociation-class output difference vs the default schedule (9/204 synthetic cells, worst `max_rel` 5.88e-03) — **not** bit-exact. Live GB10 validation on Jason Wong's 117B Laguna release dispatched all 94 expert stacks to v2 with no fallback and measured 24.993 vs 23.585 tok/s (+5.97%); long-prefill, concurrency, and soak requests completed without Gridbook errors |
| Transient-expand prefill (dense) | **Shipped**; ~1.44× native at large M (traffic-bound) |
| FP8-CB fused decode-in-prologue prefill | **Bit-exact; dispatch-eligible at M=9–128 and measured wins at M=32/64/128**; native transient expansion + CUTLASS serves ineligible and large-M shapes |
| NVFP4-CB fused native-FP4 prefill (dense and MoE) | **Explicit opt-in**: shared `static_lsq` activation policy, optimized v2 scale decode, and occupancy-aware TileM routing. Short exact K24 quality/performance passed; long-context evidence is mixed; raw native parity, >=4B, task, MoE routed-quality, and p95 served gates remain open |
| MoE persistent-B decode-in-mainloop prefill (FP4-CB v2) | **Opt-in** (`PRISMAQUANT_CB_MOE_PERSISTENT_B=1`), ROADMAP K1.1. Decodes each expert weight tile once and streams that expert's exact routed segment through it, eliminating the `[E,N,K]` BF16 transient and all work for unrouted experts. Weight decode bit-identical to `cb_expand_v2` (`torch.equal`); reduction-order-class output difference vs the bridge, gated at parity with per-segment `F.linear`. Whole-routed-operator microbenchmark wins; served protocol **not run** |
| Persistent-N large-M dense prefill | **RETIRED FROM SERVING; MEASURED NEGATIVE**: parity-green, but 2–5.7× *slower* than expand-then-GEMM at 27B shapes — the CUDA expander had already cut the dense expand tax to ~10%, removing the opportunity. The serving selector, custom op, package loader, and switch are deleted. The `.cu` remains accessible only to the explicit direct research test. The equivalent idea for **MoE** is still open and is the roadmap's next kernel |
| MoE prefill | **Native-only production lane**: grouped CUDA GEMV at M≤16; above 16, eligible quality-green FP8 fused CUTLASS or exact CUDA expansion + owned CUTLASS grouped GEMM. The current generic SM80-compatible grouped bridge is not Blackwell-optimized and measured 6–17% slower than segmented BF16 matmuls on warm synthetic DSV4 shapes; an sm12x-native CUTLASS 3.x collective for the same bridge exists **opt-in** (`PRISMAQUANT_CB_BF16_SM120`, pingpong 64×128×64): measured 1.18–1.27× that bridge and 1.02–1.05× segmented matmuls at T=128, but only 0.83–0.92× segmented at T=512, where the ragged row-padding tax of the tile-indexed construction binds — bit-gated, served protocol not run. Historical Laguna-S-2.1 measurements for the predecessor native chunk-expander path were 293 → 1,821 tok/s @8k and 207 → 1,822 @63k; they must not be relabelled as measurements of the new bridge. The v0.4.2 fused native-NVFP4 route remains an explicit A/B opt-in without MoE quality qualification |
| Missing required native kernel | **Fails closed** with an operation-specific diagnostic; no Triton dependency or serving fallback |
