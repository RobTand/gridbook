# cbq — codebook-quantized weights that serve at native tensor-core speed

> **Working title.** `cbq` is a placeholder package/repository name pending the
> maintainer's final naming decision. The one name that is **fixed** is the vLLM
> quantization-method key the artifacts carry on disk: `"prismaquant"` (see
> [Naming](#naming)). Everything else in this README that says `cbq` may be
> renamed before the first tagged release.

`cbq` defines a small family of **product vector-quantized (product-VQ) weight
formats** — **NVFP4-CB** (4-bit grid) and **FP8-CB** (8-bit grid) — and ships an
**out-of-tree vLLM plugin** that serves them. The formats are designed around a
single property:

> **A decoded codebook tile is bit-compatible with a standard hardware format**
> (NVFP4 / FP8). So a codebook weight is never "decoded to a learned vector and
> then multiplied in software." It is expanded — transiently, one layer tile at a
> time, never resident — into ordinary NVFP4/FP8 codes and fed to the **same
> Blackwell tensor-core GEMM** vLLM already runs for those formats. No forked
> runtime, no changes to vLLM core, no custom serving stack.

The result is a route to **sub-4-bit-class model sizes at native-format serving
speed**, mixed per-Linear with plain NVFP4, FP8, and BF16 inside one standard
`safetensors` checkpoint.

---

## Why this exists

Below ~4 bits per weight, single-format quantization loses quality fast, and the
methods that recover it (learned codebooks / vector quantization) have
historically been **slow to serve**: a VQ or IQ-style weight has to be
*decoded* before it can be multiplied, and that decode usually runs on general
CUDA cores, not tensor cores.

Concretely, in the `llama.cpp` / GGUF ecosystem — today's most common low-bit
serving path:

- **k-quant** formats have a good fused CUDA matmul path (MMQ), so they serve
  reasonably fast, but their quality rungs stop around 3-4 bits.
- **IQ** formats (the sub-3-bit "I-quant" rungs) reach much better quality, but
  their decode falls to a **slower CUDA-core dequant path** — a *prefill tax* you
  pay on every prompt token.

`cbq` targets **IQ-class quality at native-kernel speed**. Because a decoded CB
tile *is* an NVFP4/FP8 tile, the serving kernel:

1. **Decode (batch-1 / small M):** a bandwidth-bound fused GEMV streams the packed
   indices, expands per group in registers, and multiplies — fewer bytes per
   weight means *less* memory traffic than a wider format, so decode is at or
   above native-format speed.
2. **Prefill (large M):** the layer's weight tile is expanded **transiently**
   into a scratch buffer (bounded memory — never a resident dense weight), then
   run through the stock native GEMM.

See [`docs/MOTIVATION.md`](docs/MOTIVATION.md) for the full argument and
[`docs/KERNELS.md`](docs/KERNELS.md) for the kernel design.

---

## Headline results

All quality numbers are **at matched bits-per-weight** against a baseline
quantized to stock per-Linear NVFP4/FP8 (served natively by vLLM's
`compressed-tensors` path). Measured on a single **NVIDIA GB10 / DGX Spark**
(Blackwell `sm_121`, 128 GB unified memory, ~273 GB/s). KL is exact full-vocab
KL-vs-BF16 on held-out WikiText. Single box, single calibration seed — read the
caveats in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Model | Size | bpp | Quality vs matched-bpp NVFP4/FP8 baseline | Serving speed |
|---|---|---|---|---|
| Qwen3-class hybrid | 27B | 5.5 | **ALL-KL −58%**, confident-KL −53%, PPL gap to BF16 3× smaller | decode **at native parity** (10.3 tok/s); prefill 1.44× the baseline |
| Qwen3-MoE-class | 35B (A3B), 256 experts | 4.75 | **confident-KL −53%**, ALL-KL −43%, top-1 agreement higher on both slices | decode **faster than BF16** (33 tok/s); MoE prefill still WIP |
| Tencent-Hunyuan-class MoE | 295B (A21B), 192 experts | 2.9 | *no KL claim at this scale* (see below) | **serves on ONE DGX Spark**; prefill **2.1× the GGUF IQ artifact** (89 vs 42 tok/s); ToolEvalBench 87 = parity with the GGUF IQ build at matched bytes |

The 295B result carries **no quality-vs-teacher claim**: a 295B BF16 reference
cannot be run on a single box to compute KL against it. What is validated is that
it loads, serves, and generates coherent, arithmetically-correct output on one
128 GB box, at 2.1× the prefill throughput of the matched-bytes GGUF IQ build.

---

## Install

```bash
pip install cbq          # working-title name; see the banner above
```

Requirements:

- **NVIDIA Blackwell GPU**, compute capability `sm_120`/`sm_121` (the GB10 / DGX
  Spark is the reference target). The fused tensor-core prefill path is
  Blackwell-specific.
- **CUDA toolkit with `nvcc`** on `PATH`. The plugin builds a CUDA extension at
  install time (a `torch.utils.cpp_extension` / `CUDAExtension` build). `nvcc`
  13.0 is the tested toolchain.
- **PyTorch** built for a matching CUDA version.
- **vLLM**. Pin a tested vLLM version per artifact (the plugin surface is
  intentionally small — a quantization config, a linear/MoE method, and a few
  registered custom ops — but vLLM's internal APIs drift; see
  [`docs/KERNELS.md`](docs/KERNELS.md)).

If the CUDA extension cannot be built (no `nvcc`, non-Blackwell GPU), the plugin
still imports and runs through **Triton fallback kernels** for correctness and
CI. Those fallbacks do **not** reach the FP4 tensor cores and are not a
production serving target — they exist so the package installs and the numerics
can be verified anywhere.

The plugin registers itself through vLLM's `vllm.general_plugins` entry point: on
`import vllm` it calls `register_quantization_config("prismaquant")`, after which
vLLM **auto-detects** a `cbq` checkpoint from its `config.json` and needs no extra
flags. There are **zero vLLM-core patches or monkeypatches** on the load path.

---

## Quickstart — serve an artifact

A `cbq` artifact is a standard Hugging Face directory: `config.json` +
`model.safetensors` (or shards) + a codebook sidecar `cb_codebooks.pqcb` +
tokenizer files. `config.json` already contains the full quantization config, so
vLLM auto-detects the method.

```bash
# The plugin is discovered via its entry point; no --quantization flag needed.
vllm serve /path/to/cbq-artifact \
  --enforce-eager \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8000
```

Notes:

- `--enforce-eager` is currently recommended: the decode path uses an
  M-gated kernel dispatch that is not yet CUDA-graph-friendly for every model
  (details and the exceptions in [`docs/KERNELS.md`](docs/KERNELS.md)).
- Set `--max-model-len` to fit your KV budget. For the 295B artifact on a single
  128 GB box, 8192-16384 is the practical range (the weights alone are ~110 GB).

Then hit the standard OpenAI-compatible endpoint on port 8000.

---

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/SPEC.md`](docs/SPEC.md) | **Normative format specification** — byte layout, product-VQ, v1/v2 scale coding, codebook sidecar, `config.json` vocabulary, and the extensibility contract. Implementation-independent; MUST/SHOULD language. |
| [`docs/MOTIVATION.md`](docs/MOTIVATION.md) | Why the 2-4 bpp regime needs codebooks, why serving has been the blocker, and an honest comparison to GGUF k-quant/IQ. |
| [`docs/KERNELS.md`](docs/KERNELS.md) | The serving kernels: transient-expand tensor-core prefill, fused act-QDQ decode GEMV, two-tier in-register scale compose, grouped MoE GEMV, Triton fallbacks, and CUDA-graph safety rules. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | The measured results with full hardware/protocol context and caveats. |
| [`ROADMAP.md`](ROADMAP.md) | What is planned: a reference encoder, quantized-MTP spec decode, large-M prefill kernels, and the naming decision. |

---

## Naming

The vLLM registry key is **`"prismaquant"`** — this is the string in every
published artifact's `config.json` (`"quant_method": "prismaquant"`) and it
**stays that way** for backward compatibility, regardless of what the Python
package and repository are ultimately called. It is a plain identifier string,
not a code dependency on any other project. The `cbq` working title applies only
to the package/repo/CLI names and may change before the first release.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). The format specification in
[`docs/SPEC.md`](docs/SPEC.md) is published under the same license: the format is
free, open, and intended to be implemented by anyone, in any runtime, without
permission.

## Attribution

Robert Tand — <robert.tand@icloud.com>
