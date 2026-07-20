# Motivation — why codebook formats, and why they can serve fast

## The problem: the 2-4 bit regime

Above ~4 bits per weight, simple per-group scalar quantization (round each weight
to a small grid with a shared scale) is close enough to lossless that the format
choice barely matters. Below ~4 bits it stops working: there are too few grid
points, the per-weight error grows, and single-format methods crack — quality
falls off faster than the bit savings justify.

This is exactly the regime that matters for fitting large models onto a single
box. A 295B-parameter model is ~570 GB in BF16 and ~150 GB even at 4 bits; getting
it onto one 128 GB machine means living **between 2 and 3 bits per weight**, where
scalar formats are worst.

## Codebooks buy the quality back — cheaply

The classical answer is **vector quantization**: don't round each weight
independently: round *groups* of weights jointly to the nearest entry in a learned
(or structured) codebook. A codebook of `2^k` vectors over `d` weights spends
`k/d` bits per weight on the index and captures correlations a per-scalar grid
cannot. This is what the GGUF "IQ" formats, AQLM, QuIP#, and QTIP all do in
different ways.

`cbq`'s codewords are **8-dimensional vectors of FP4 (or FP8) grid values** — a
*product* vector quantizer: the `k`-bit index is split into a few sub-indices,
each selecting from a smaller sub-codebook, and the sub-codewords are concatenated
into the 8-dim vector. This keeps codebooks small (encode-searchable and
smem-resident) while spanning a large effective codeword set.

An internal rate-distortion study (numerical, held-out, MSE — a coding-theoretic
distortion proxy, **not** a served metric) measured the two questions that decide
whether this can work:

1. **How much does forcing codewords onto the FP4 grid cost?** Small. The FP4-grid
   "tax" versus an unconstrained codebook of the same size averaged **~+4.5%** MSE
   (full/direction+magnitude mode) and ~+10% (explicit-sign magnitude mode), across
   synthetic and real-weight sources. The grid constraint is **not** the ceiling.
2. **Is the FP4 codebook fundamentally worse than IQ's codebook at matched size?**
   No. At matched codebook size, an FP4-grid codebook is roughly on par with IQ's
   fixed grid, and a *learned* magnitude codebook slightly beats it. IQ's advantage
   in head-to-head tests came mostly from a **larger bit budget**, not a better
   codebook design.

So the codebook and the grid are cheap. The one real structural cost is elsewhere.

## The one structural cost, and how it is removed

An FP4-CB tile must carry a **group-16 E4M3 scale plane** — because the whole
point is that a decoded tile is a *native NVFP4 tile* the tensor core can consume
directly, and that is the scale representation NVFP4's tensor-core path expects.
That plane costs **0.5 bits/weight**. IQ formats amortize their scales with a
two-tier scheme (a per-superblock super-scale plus cheap per-group sub-scales) for
roughly **0.31 bits/weight**. The honest gap is that **~0.19 bits/weight of
scales** — the reason a naive FP4-CB trails IQ at matched bytes.

`cbq` closes it with a **two-tier scale coding of its own** (format layout v2,
specified in [`SPEC.md`](SPEC.md) §1.2): a 1-byte E8M0 super-scale per 256 weights
plus a 4-bit sub-code per group-16, indexing a fixed table of E4M3-exact
multipliers. It composes back to a bona-fide E4M3 plane **by construction** (no
rounding), so the tensor core still sees exactly the plane it wants — but on disk
and in memory the scales cost **0.28 bits/weight**, now *cheaper* than IQ's 0.31.
Measured on real weights, this coding was **error-neutral-to-better** than the
direct E4M3 plane while saving ~0.22 bits/weight (the direct plane turned out to
be granularity-starved on the subnormal-heavy scales real LLM weights produce; the
two-tier super-scale restores per-block renormalization).

Net: with the FP4-grid tax small, the codebook competitive, and the scale cost
now at or below IQ's, an FP4-CB format can plausibly **match IQ-class quality at
matched bytes** — while decoding to native FP4.

> These are emulation/MSE results. MSE is a distortion proxy, not the served KL
> that ultimately decides quality. The served evidence is in
> [`BENCHMARKS.md`](BENCHMARKS.md); it is what actually promotes or rejects the
> format at each scale.

## The historic blocker: serving

If codebooks are such a good deal in the 2-4 bit regime, why isn't everything a
codebook? **Because they have been slow to serve.** A vector-quantized weight has
to be *decoded* before it can be multiplied, and that decode has traditionally
meant one of two bad options:

- **A forked runtime / custom kernels.** AQLM, QuIP#, QTIP, and friends each ship
  their own inference path. That is a maintenance burden and a deployment barrier:
  you cannot just point a stock server at the checkpoint.
- **Decode to a wide dtype, then a software matmul.** Expand the codebook to
  BF16/FP16 and run a general-purpose matmul. This leaves the tensor cores idle on
  prefill and is memory-heavy.

A related trap is **load-time expansion**: decode the compressed weight into a
dense low-bit tensor once, at load, to reuse an existing kernel. On a unified-memory
box this saves *disk* but not *memory* — the resident footprint returns to the
expanded size and can OOM the very box the compression was meant to fit. (An
earlier custom-format attempt on this hardware did exactly this: a 93 GB artifact
expanded to 116 GB resident and OOM-killed. That failure is why INV-1 in the spec
forbids resident expansion outright.)

## The design goal: IQ-class quality at native-kernel speed, on a stock server

`cbq`'s formats are built so the decode is **not** a software detour:

- **A decoded CB tile is a native hardware tile** (bit-compatible NVFP4 / FP8).
  The codebook lookup replaces only the *load* of weight bytes; the scales and the
  matmul are the runtime's own native path, unchanged.
- **Serving is an out-of-tree plugin** that registers a quantization method and
  delegates non-CB layers to the stock `compressed-tensors` path — **zero core
  patches, zero forked runtime.** A standard server auto-detects the checkpoint and
  serves it.
- **No resident expansion (INV-1):** weights are expanded transiently, one tile at
  a time, so a smaller-on-disk artifact is also smaller-in-memory — which is the
  whole point on a single 128 GB box.

The kernel design that delivers this is in [`KERNELS.md`](KERNELS.md).

## Honest comparison to GGUF (k-quant and IQ)

GGUF / `llama.cpp` is today's most widely used low-bit serving path, and the
comparison is the fair one to make. It is not a strawman:

- **k-quant** (Q4_K, Q5_K, Q6_K, …) has a genuinely good fused CUDA matmul path
  (MMQ). It serves fast. Its quality rungs, though, live around 3-6 bits — it does
  not have strong sub-3-bit options.
- **IQ** (IQ2_XXS … IQ3_S) reaches much better quality *per bit* in the sub-3-bit
  range, because it is codebook-based. But IQ decode falls to a **slower
  general-CUDA dequant path** rather than the fast MMQ path — so you pay a
  throughput tax, most visibly on **prefill** (processing the prompt), where the
  matmul is large and would otherwise be tensor-core-bound.

`cbq` aims squarely at the gap: **IQ-class quality (codebook, sub-3-bit-capable)
with native-kernel speed (decoded tiles hit the tensor cores).** The clearest
single data point is on a 295B model at matched bytes: the CB build's prefill runs
at **~2.1× the throughput** of the equivalent GGUF IQ build, while landing on the
**same** tool-use benchmark score. The prefill tax that IQ pays is exactly what
the native-tile design removes. Details, numbers, and caveats in
[`BENCHMARKS.md`](BENCHMARKS.md).

What `cbq` does **not** claim over GGUF: it is Blackwell-specific (the tensor-core
path is `sm_120/121`); it does not interoperate with GGUF files; and at matched
bytes the *quality* is base-model- and bit-budget-dominated — the win over GGUF is
**speed at parity quality**, not a quality leap over IQ. The win over uniform
NVFP4/FP8 at matched bytes *is* a quality win (see the 27B/35B KL numbers), because
there the comparison is codebook-vs-scalar at the same byte budget.
