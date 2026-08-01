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

A **baseline-parity gate** precedes all fork work: a plain `sm_120` block-scaled
GEMM built from vendored CUTLASS headers matches the runtime's native
`cutlass_scaled_mm` to within 0.91-0.99×, proving the toolchain and the tile-layout
understanding before touching the mainloop. Note the fork uses a **fixed-config**
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
target is a measured CUTLASS 3.x SM100/SM121 grouped collective.

MoE dispatch is separate from the dense M boundary: `M≤16` uses the owned
grouped CUDA GEMV. Above 16, FP8-CB first attempts its quality-green fused
CUTLASS path and otherwise uses exact BF16 expansion plus the owned CUTLASS
grouped bridge; FP4-CB uses exact BF16 expansion plus that bridge. Only expert
chunk size / transient-byte-budget overrides remain—there is no stock/loop/
batched/L2 production selector.

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

The remaining MoE prefill target is an activation-contract-preserving
persistent/grouped decode-in-mainloop schedule (the expand is ~35% of MoE
layer time at Laguna scale) — see
[`K1.1` in the kernel TODO](../ROADMAP.md#kernel-todo-canonical).

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
3. **All device-side constants are precomputed once per device.** A real bug this
   caught: an activation-QDQ kernel built its FP4/E2M1 grid on the CPU and
   H2D-copied it *every call* — a hidden sync in eager mode and a hard error under
   capture. The grid is now cached per device.
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
| Persistent-N large-M dense prefill | **RETIRED FROM SERVING; MEASURED NEGATIVE**: parity-green, but 2–5.7× *slower* than expand-then-GEMM at 27B shapes — the CUDA expander had already cut the dense expand tax to ~10%, removing the opportunity. The serving selector, custom op, package loader, and switch are deleted. The `.cu` remains accessible only to the explicit direct research test. The equivalent idea for **MoE** is still open and is the roadmap's next kernel |
| MoE prefill | **Native-only production lane**: grouped CUDA GEMV at M≤16; above 16, eligible quality-green FP8 fused CUTLASS or exact CUDA expansion + owned CUTLASS grouped GEMM. The current generic SM80-compatible grouped bridge is not Blackwell-optimized and measured 6–17% slower than segmented BF16 matmuls on warm synthetic DSV4 shapes. Historical Laguna-S-2.1 measurements for the predecessor native chunk-expander path were 293 → 1,821 tok/s @8k and 207 → 1,822 @63k; they must not be relabelled as measurements of the new bridge. The v0.4.2 fused native-NVFP4 route remains an explicit A/B opt-in without MoE quality qualification |
| Missing required native kernel | **Fails closed** with an operation-specific diagnostic; no Triton dependency or serving fallback |
