# Roadmap

What is done, what is open, and what was tried and rejected. Nothing here is a
commitment to a date. Kernel statuses below track the internal kernel/format
standard; a path is **DEFAULT** (on, no flag), **OPT-IN** (behind an environment
switch), or **MEASURED NEGATIVE** (built, measured, and kept off).

---

## Done

These are listed because older documents — and older versions of this file —
still describe some of them as future work.

- **Dense decode.** CUDA decode GEMV for FP8-CB and FP4-CB (two-tier v2),
  DEFAULT. Measured at/above native-format parity on the 27B (10.3 vs 10.26
  tok/s).
- **MoE decode.** Grouped `(token, expert)` GEMV plus a deterministic combine,
  DEFAULT. Took 35B-MoE decode from 3.52 → ~33 tok/s (faster than BF16's 28.4).
- **MoE prefill.** A CUDA chunk-expander feeding vLLM's own fused-MoE grouped
  kernel. Measured on Laguna-S-2.1 (117B MoE): 293 → **1,821 tok/s** at 8k and
  207 → **1,822 tok/s** at 63k (commit `8829c16`). The current default is
  `auto`: a measured per-layer selection over the candidate paths, promoted after
  a two-model gate showed it at or above the best fixed path on both models.
- **Mid-M fused prefill** (16 < M ≤ 128): a CUTLASS decode-in-prologue GEMM with
  an fp32 epilogue, promoted to DEFAULT after measuring **1.40× in its niche**
  at the promotion gate (1.04–1.45× across M = 32/64/128 on GB10) with the
  quality gate preserved. `sm_120`-family only; fails soft elsewhere.
- **Quantized MTP draft head.** The 295B artifact ships an FP8-CB K44 draft
  block; speculative decode at k=1 measured 14.6 → **16.1 tok/s** on prose. (The
  remaining upside needs vLLM to capture drafter CUDA graphs — upstream work, see
  below.)
- **Every integer rung** across both ladders (NVFP4-CB K12–K24, FP8-CB K28–K48)
  with ceil-first uneven index splits, encoder-anchored and frozen.
- **Packaging.** The CUDA sources ship inside the Python package, so a
  non-editable `pip install` produces a working CUDA path. Previously any
  non-editable install silently degraded to the Triton fallback.

---

## Open

### Reference encoder

This repository defines and **serves** the format; it does not yet **produce**
artifacts. They come from the authors' offline research pipeline
([PrismaQuant](https://github.com/RobTand/prismaquant)), which is not part of
this repo. A standalone, documented reference encoder is planned — codebook
construction, the weighted product-VQ search, the two-tier scale encoder and the
safetensors exporter — so that anyone can turn a BF16 model into a gridbook
artifact using only this repository and [`docs/SPEC.md`](docs/SPEC.md). Until it
lands, the spec is deliberately complete enough to implement an encoder
independently.

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

- **A capability guard on the dense FP8-CB prefill.** It calls a CUTLASS fp8 GEMM
  that needs `sm_89+` with no guard and no fallback, while the declared
  capability floor is 8.0 — so an A100 is told it is supported and then expected
  to fail on the first real prompt. The Triton path is already a correct
  fallback; it just is not selected.
- **An architecture precheck before the fused CUTLASS build**, so a non-Blackwell
  GPU skips a doomed multi-minute compile inside the user's first request instead
  of failing soft after it.

### vLLM compatibility preflight

The plugin imports vLLM internals (fused-MoE classes, `_custom_ops`, the
quantization registry) that carry no stability promise, and vLLM
logs-and-continues when a plugin fails to load — so drift surfaces as an
unrelated "invalid quantization method" at model load. A symbol canary that fails
with one actionable sentence naming the missing symbol and the tested vLLM
version is the intended fix.

### Large-M fused MoE prefill

The remaining kernel target. On a Laguna-class MoE the transient expand is ~35%
of MoE layer time, so a persistent/grouped decode-in-mainloop schedule — decode
each expert weight tile once and stream M through it — is where the next real
prefill win is. This is the MoE analog of the dense experiment below, which
failed for a reason that does not apply here.

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

- **Tensor parallel.** No `tp > 1` support exists and none is planned. Open an
  issue if you need it.
- **A vLLM fork or core patches.** Running on stock vLLM is the point.

---

## Measured and rejected

Kept here because a rejected experiment with a number attached is more useful
than silence.

| Item | Verdict |
|---|---|
| **Persistent-N large-M dense prefill** | Built, parity-green, and **2–5.7× slower** than expand-then-GEMM at 27B shapes: the CUDA expander had already shrunk the dense expand tax to ~10%, removing the opportunity that motivated it. The kernel is retained, quarantined behind `PRISMAQUANT_ENABLE_PTC=1`, as a schedule reference. Do not enable it. |
| **`grouped_fused` MoE prefill as default** | Wins on small-expert MoE (35B class), loses on large-expert (Laguna class). Reverted to OPT-IN; the measured per-layer `auto` selection is the end state. |
| **w2 rowpack decode schedule** | Measured negative; stays behind an environment switch as a recorded result. |
| **Decode contract v2** (scale-epilogue hoist) | Measured **null** on the served 27B (10.10 vs 10.13 tok/s, quality-neutral) — decode is bandwidth-bound at per-byte parity, so there was nothing for the hoist to recover. Default stays v1; v2 remains available. |
| **L2-pinned per-expert scratch pipeline** | Wedged live serving three times, including the serial variant. DIAGNOSTIC-ONLY, excluded from the `auto` candidate set; the underlying L2-residency hypothesis is still unmeasured. |
| **Signed "S-rung" formats** | Serving correctness proven bit-exact end to end, but in a matched-rate head-to-head over 776 per-(Linear, rung) comparisons the unsigned rungs won 79% of the time and the allocator placed 6 signed units against 147 unsigned. Closed as research-only; the spec keeps them for exotic weight geometries. |
| **Naive inline CUDA-graph capture of the decode path** | Measured *worse*: a prefill-sized trace baked the expand arm into decode. This specific design remains rejected. The later opaque whole-dispatch op fixed the mechanism; mode-0 `FULL_DECODE_ONLY` is now a validated candidate (20.1% faster on the close-rate 0.6B canary), pending the 27B streaming gate. |
