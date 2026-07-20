# vllm-prismaquant

Out-of-tree vLLM quantization plugin for PrismaQuant's **NVFP4-CB / FP8-CB**
codebook formats (docs/nvfp4-cb-plan/). This is **prototype (i)** of the serving
plan (`docs/nvfp4-cb-plan/serving-kernel.md`): a **correctness-first, Triton**
serving path used to obtain the first served KL-vs-BF16 and a first speed
reading. It is **not production-eligible**.

## Invariants

- **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) scales. The dense `[N,K]` weight is
  never materialized in HBM — each superblock's weight tile is expanded inside
  the kernel, in registers, then consumed by the matmul.
- **INV-2 (WAIVED):** we decode FP4/FP8 codes to bf16 and run `tl.dot` (bf16
  MMA). Triton cannot emit the Blackwell sm_121 block-scaled FP4 MMA, so this
  path will fail the prefill perf gate by construction. The production prefill is
  prototype (iii) (CUTLASS/CuTe fused-expand).

## Scope

- Even-split **product** mode only (NVFP4_CB_K16 → (8,8); FP8_CB_K44 →
  (11,11,11,11)). Uneven splits and signed/full modes: out of scope here.
- Uniform CB artifacts (all target Linears one format). Mixed containers would
  delegate plain NVFP4/FP8 to stock compressed-tensors — stubbed with a clear
  `NotImplementedError`.
- Single-GPU (tp=1); `--enforce-eager`.

## Layout / registration

- `register_quantization_config("prismaquant")` — vLLM auto-detects from
  `config.json["quantization_config"]["quant_method"] == "prismaquant"`.
- The exporter inlines the full quant config (config_groups + ignore) into
  `config.json`, so `from_config` gets everything; the shared `cb_codebook.*`
  tensors live in a sidecar `cb_codebooks.safetensors`, loaded **once** by the
  config via `get_current_vllm_config()` at weight-load time. **Zero vLLM-core
  monkeypatching.**

## Tests

`tests/test_cb_kernels.py` runs the Triton path on the **real exported** 0.6B
tensors and (a) matches `nvfp4_cb_reconstruct @ x` to ≤1e-2 rel, (b) checks the
kernel's codeword extraction is bit-exact vs `nvfp4_cb_unpack`.
