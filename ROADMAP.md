# Roadmap

This repository currently ships the **format specification** and the **serving
plugin** (decode + prefill kernels for vLLM). The items below are planned; nothing
here is a commitment to a date, and priorities may change.

## Reference encoder

Today, `gridbook` artifacts are produced by the authors' offline research quantization
pipeline, which is **not** part of this repository — this repo defines and *serves*
the format, it does not yet produce artifacts. A **standalone, documented reference
encoder** is planned, so that anyone can turn a BF16 model into a `gridbook` artifact
using only this repository and [`docs/SPEC.md`](docs/SPEC.md): the codebook
construction (fixed lattice and shared-per-role learned), the weighted product-VQ
search, the two-tier scale encoder, and the safetensors exporter. Until it lands,
the spec is complete enough to implement an encoder independently — that is a
stated goal of publishing the spec openly.

## Quantized multi-token-prediction (MTP) head for speculative decode

Several target models ship a multi-token-prediction (MTP) draft head usable for
speculative decoding. On a 128 GB box, a BF16 MTP head (e.g. ~7.5 GB on the 295B
artifact) does not fit alongside ~110 GB of weights — attempting it OOMs the box.
Quantizing the MTP head to a CB format (~2 GB) would let speculative decode fit on
a single box, a direct decode-throughput win. The artifact already carries the MTP
head for larger boxes; the work is to quantize and serve it.

## Persistent-N large-M prefill kernel

The fused decode-in-prologue collective is bit-exact and wins at medium batch
(M∈(16,128]) but loses at large M because every M-tile re-decodes the same weight
tiles (see [`docs/KERNELS.md`](docs/KERNELS.md)). Large-M prefill parity requires a
**weight-stationary / persistent-N schedule** — decode each weight tile once and
loop the M dimension inside the CTA. This is the kernel restructure that would close
the remaining dense-prefill gap to the native NVFP4/FP8 GEMM.

## Batched-expert MoE prefill

MoE decode is solved by the grouped `(token, expert)` GEMV, but MoE **prefill**
still runs the correctness-path per-expert loop, whose launch storm dominates TTFT.
A batched-expert transient expand + grouped GEMM (the prefill analog of the grouped
decode kernel) is the remaining MoE serving piece.

## Package / repository naming decision

RESOLVED (2026-07-20): the project name is **gridbook** (Robert). The PyPI
package name is `gridbook`; the Python import name stays `gridbook`
until the pre-1.0 import rename (with a compatibility shim); the vLLM registry
key `"prismaquant"` is permanent (baked into published artifact configs).

**Independent of that decision, the vLLM quantization-method registry key stays
`"prismaquant"`.** It is the string baked into every already-published artifact's
`config.json` (`"quant_method": "prismaquant"`), and keeping it is what lets a
renamed package still load existing checkpoints. The package name and the registry
key are decoupled on purpose.
