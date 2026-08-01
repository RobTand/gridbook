# Serving kernels

This document describes how a `gridbook` artifact is served fast. It is a design and
status document, not an API reference; the normative decode semantics are in
[`SPEC.md`](SPEC.md).

Terminology used below: **GEMM** = matrix-matrix multiply (prefill / large batch);
**GEMV** = matrix-vector multiply (decode / batch-1 or small batch); **M** = the
number of activation rows (tokens) in a matmul; **MMA** = a tensor-core
matrix-multiply-accumulate instruction; **QDQ** = quantize-then-dequantize the
activations; **smem** = GPU shared memory; **TTFT** = time to first token
(prefill latency).

## The two invariants everything is built around

- **INV-1 — no resident expansion.** The resident weight is always the packed
  `cb_qweight` (indices + scale plane) plus the small shared codebook. The dense
  weight is **never** materialized in memory. Decoding happens in registers/smem
  per tile, or into a per-layer scratch buffer that is freed after the matmul.
  This is what makes a smaller-on-disk artifact also smaller in memory — the
  reason the format fits large models on one box. A resident-footprint assertion
  is a load-time gate, not a nicety.
- **INV-2 — native tensor cores for prefill.** The production prefill decodes CB
  indices into native FP4/FP8 codes and feeds the hardware tensor-core GEMM — the
  same path a plain NVFP4/FP8 layer uses. A decode-to-BF16-then-`torch`-matmul
  kernel is a correctness/fallback tool only; it does not meet the prefill goal.

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

**Dispatch.** The path is M-gated: a CUDA GEMV for the smallest M, a Triton GEMV
for a middle band, and the transient-expand prefill path (below) above that. See
the CUDA-graph section for why this host-side branch matters.

---

## Prefill path (large M): transient expansion

The default prefill **expands one layer's weight tile transiently** into a scratch
buffer, runs a GEMM, and frees the buffer. Memory stays bounded (INV-1). FP8-CB
expands **directly to FP8** (the codebook values are already on the E4M3 grid)
and calls the stock native tensor-core GEMM (INV-2). FP4-CB v2 currently expands
to BF16 and calls cuBLAS; that is a correctness-first fallback which does not yet
meet INV-2. The separate native-FP4 decode-in-prologue kernel is opt-in
because it changes the served activation bucket. The current `static_lsq`
policy fixes the original dense accuracy and timing defects on the short exact
screen, but its long-prompt evidence remains statistically unresolved and it
has no >=4B or MoE served validation. It therefore remains opt-in; see the
[fused-NVFP4 enablement audit](audits/fused_nvfp4_enablement_2026-07-31.md).

The honest limitation of transient-expand is **memory traffic**: the tile is
written to HBM and then read back by the GEMM, so prefill moves roughly 2× the
bytes of a resident-weight GEMM. Tuning the expander (a CUDA expander at ~2× the
Triton one, FP8-direct output) narrowed but cannot remove this — on the 27B
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
`static_lsq` is not yet wired or quality-qualified for grouped MoE.

### FP8-CB fused decode-in-prologue (the 1× fix)

A fused collective mainloop that **decodes CB indices inside the GEMM's
global→shared prologue** — never writing the expanded tile to HBM — exists and is
**bit-exact** against the transient path (a forked `sm_120` block-scaled MMA
collective, packed-B TMA load + consumer-side smem decode). Its honest status:

- It **wins at medium M** — roughly 1.04× / 1.26× / 1.45× at M = 32 / 64 / 128.
- It **loses at large M** (≈0.22× at M≈1400). This is *structural, not a bug*:
  every M-tile CTA re-decodes the same weight (B) tiles, so decode work scales with
  `ceil(M / tile)` while the transient path expands each tile exactly once.
- Large-M parity therefore requires a **weight-stationary / persistent-N schedule**
  (decode each weight tile once, loop M inside the CTA) — a kernel-layer
  restructure beyond the collective fork. Until that lands, the serial transient
  path is the default for large-M prefill.

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

The **correctness-first per-expert loop is retained** as a fallback path, and its
numerics are pinned bit-identical to the grouped kernel by a regression test.

**Smem-resident dictionary (opt-in, `PRISMAQUANT_CB_GEMV=v2`).** The shipped
grouped GEMV gathers sub-codebook entries from global/L2 per lane per
superblock. `csrc/cb_gemv_v2.cu` instead stages the whole product dictionary in
shared memory once per block and bursts each output row's packed bytes into
smem before decoding it — the same insight the fused **prefill** mainloop
already uses (`csrc/cutlass_fork/sm120_cb_fused_mma.hpp`, where it removed the
k48 L1 cliff), applied to the M≤16 decode regime. It opts in to the sm_121a
99 KB dynamic-smem budget, which the shipped decode GEMV does not (it caps its
rowpack request at 48 KB), so the k20/k24 staged configurations are only
expressible in the new kernel. The loader admits only the native Blackwell
target capabilities (12.0/12.1) with at least 99 KiB opt-in shared memory; this
PR's execution battery covers cc 12.1, not cc 12.0. It
is **not** bit-exact against the default
grouped schedule — reassociation class, same as CUDA-vs-Triton — and it is
**not** a default: unset means `inherited`. Explicit `auto` or `v2` reaches it
only where the hardware gate and compiled occupancy predicate both pass;
`PRISMAQUANT_CB_GEMV=inherited` is the kill switch. The
selection resolves once per process and is fixed per (layer, stack) at load, so
the call-site branch is a trace-time constant and FULL-decode cudagraphs are
unaffected.

MoE **prefill** used to be that per-expert loop, whose launch storm dominated
TTFT. It no longer is. A **CUDA chunk-expander** now expands expert chunks
directly into vLLM's own fused-MoE grouped kernel (raw unpadded views,
byte-identical to the Triton expand on every rung), and the fp8-CB default is
`auto`: a first-prefill measured selection, per layer, over the candidate paths,
cached for the process. Measured on Laguna-S-2.1 (117B MoE): **293 → 1,821 tok/s
at 8k** and **207 → 1,822 tok/s at 63k**. Chunked prefill re-expands per
microbatch, so `--max-num-batched-tokens 16384` matters for this path.

**fp4**-CB MoE prefill can explicitly ride the same chunked `stock` path. The
unset policy is the conservative per-expert `loop`. The native-FP4 grouped path
is enabled with `PRISMAQUANT_CB_FUSED_FP4_MOE=1` (or `128`/`256` to select its
tile), while an explicit `PRISMAQUANT_CB_PREFILL` remains authoritative and is a
reliable bisection control. This is deliberately an A/B opt-in: its grouped
routing and native-MMA arithmetic are qualified, but switching from Gridbook's
fp32-emulated activation QDQ to native UE4M3 activation factors materially
changes the served distribution. It must not be promoted on kernel parity or
raw speed alone; see the
[fused-NVFP4 enablement audit](audits/fused_nvfp4_enablement_2026-07-31.md).
Stock's transient is a bf16 expand rather than
the fp8-direct CUDA one, at 2 B/elt, so its chunk is sized from a **byte budget**
(`PRISMAQUANT_CB_PREFILL_CHUNK_BYTES`, default 1 GiB) instead of the fp8 lane's
flat 256: on a 192-expert Hy3-class band that is 1,184 MiB of measured
transient rather than 4,736 MiB, at no measured time cost. The fp4 branch also
drops the `codec.pad_qweight` copy — `type_size = 4k+9` leaves the expanders'
8-byte codeword window inside the superblock for every `k`, so the padding
bytes were never read (bit-identical output, verified with
`PRISMAQUANT_CB_EXPAND=pad`).

The remaining MoE prefill target is an activation-contract-preserving
persistent/grouped decode-in-mainloop schedule (the expand is ~35% of MoE
layer time at Laguna scale) — see [`ROADMAP.md`](../ROADMAP.md).

---

## Triton fallbacks

Every path has a **Triton fallback** used for correctness and CI, and where a CUDA
kernel is not available for a particular grid/mode combination. Triton **cannot
emit the Blackwell block-scaled FP4 MMA**, so a Triton kernel can do the smem
codebook lookup but only reaches BF16 MMA — it violates INV-2 and is therefore a
correctness/decode tool, **not** a production prefill target. The package installs
and its numerics verify on machines without the CUDA extension via these
fallbacks; do not mistake a working Triton prefill for a production-eligible one.

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
4. **Hoist the whole M-gated dispatch behind one opaque op.** The historical
   inline branch was capture-hostile: a prefill-sized trace could bake the
   expand arm into decode. With the default `PRISMAQUANT_CB_DISPATCH=op`, each
   Linear/MoE call is one opaque `cb_*_forward` op whose eager implementation
   resolves M at capture time. A `FULL_DECODE_ONLY` capture records the GEMV arm
   at each fixed decode size; prefill stays outside that graph and resolves the
   large-M arm independently. `PRISMAQUANT_CB_DISPATCH=inline` is an A/B escape,
   not the graph configuration.
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
| FP4-CB v2 decode (dense) | **Shipped**: bit-matched CUDA GEMV (13/13 parity vs Triton + expand reference). The decode chain is compute-bound at GEMV shapes (ncu SM 71%/mem 44%) under the bit-exact contract — the measured ceiling, not a staging problem |
| MoE grouped decode GEMV | **Shipped**: fp8 66–95% of peak; fp4-v2 w2 schedule redesigned (+50%, 37–47% of peak; reassociation served-gated with an env-switched legacy path). A rowpack variant measured NEGATIVE and stays opt-in-off as a documented result |
| MoE grouped decode GEMV, smem-resident dictionary | **Opt-in** (`PRISMAQUANT_CB_GEMV=v2`), never a default. Wins 1.13–1.58× on k13/k16/k20 in a 16-cell GB10 sweep; loses on k24 at K≥2048 (occupancy wall), where a compiled predicate routes the cell back to the shipped kernel. Reassociation-class output difference vs the default schedule (9/204 synthetic cells, worst `max_rel` 5.88e-03) — **not** bit-exact. Live GB10 validation on Jason Wong's 117B Laguna release dispatched all 94 expert stacks to v2 with no fallback and measured 24.993 vs 23.585 tok/s (+5.97%); long-prefill, concurrency, and soak requests completed without Gridbook errors |
| Transient-expand prefill (dense) | **Shipped**; ~1.44× native at large M (traffic-bound) |
| FP8-CB fused decode-in-prologue prefill | **Bit-exact, wins M∈(16,128], loses large M** — persistent-N is the answer |
| NVFP4-CB fused native-FP4 prefill (dense) | **Explicit opt-in**: shared `static_lsq` activation policy, optimized v2 scale decode, and occupancy-aware TileM routing. Short exact K24 quality/performance passed; long-context evidence is mixed; raw native parity, >=4B, task, p95 served, and MoE gates remain open |
| Persistent-N large-M dense prefill | **Built and MEASURED NEGATIVE**: parity-green, but 2–5.7× *slower* than expand-then-GEMM at 27B shapes — the CUDA expander had already cut the dense expand tax to ~10%, removing the opportunity. Quarantined behind `PRISMAQUANT_ENABLE_PTC=1` as a schedule reference; do not enable it. The equivalent idea for **MoE** is still open and is the roadmap's next kernel |
| MoE prefill | **Shipped**: CUDA chunk-expander into vLLM's fused-MoE grouped kernel; fp8-CB default is `auto` (measured per-layer path selection). Laguna-S-2.1 117B: 293 → 1,821 tok/s @8k, 207 → 1,822 @63k. fp4-CB defaults to conservative `loop`; native fused-FP4 remains an explicit A/B opt-in, and dense `static_lsq` supplies no MoE qualification. Explicit `stock` uses a byte-budgeted expert chunk (1,184 MiB vs 4,736 MiB measured transient on a 192-expert band) and no pad copy |
| Triton fallbacks | **Shipped** for every path (correctness/CI; not INV-2-eligible) |
