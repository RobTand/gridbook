# Fused NVFP4 enablement decision — 2026-07-31

Last updated: 2026-08-01.

Status: **keep both fused NVFP4 paths explicit opt-ins**. The dense path remains behind `PRISMAQUANT_CB_FUSED_FP4`; the grouped-MoE path remains behind `PRISMAQUANT_CB_FUSED_FP4_MOE`. The CUDA operator gate is green after the fixes below, but the same-process model gate is red. Neither path should become the default.

## Decision basis

The fused kernel is bit-exact with the native NVFP4 reference when both consume the same packed E2M1 activations and UE4M3 scale factors. That does not make it numerically equivalent to Gridbook's current served baseline:

- baseline: fp32-emulated group-scale FP4 activation QDQ;
- fused: native NVFP4 global fp32 scale plus per-group UE4M3 factors.

The A/B therefore measures the complete activation-and-kernel execution contract, not a weight-only kernel substitution. A model-level distribution change can coexist with a correct fused kernel. That is what the evidence shows.

The configured promotion screen required mean KL at most `1e-4`, mean NLL regression at most `ln(1.005) = 0.0049875`, PPL regression at most 0.5%, and an offline timing speedup of at least 1.10x. Older rows use a paired top-1024-plus-tail approximation. The final LFM smoke uses cardinality-checked exact full-vocabulary KL and is `measurement_only`: it can reject enablement, but its one short prompt is not a model-quality characterization.

| Screen | Sample | Verified dispatch | Mean ΔNLL; PPL ratio | Mean KL (mode), baseline→fused / reverse | Offline one-token wall time, baseline→fused | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Dense Qwen3-0.6B, NVFP4-CB K16 v2 | 4 × 128 tokens; 508 scored positions | 896 fused successes, 0 errors; 896 ineligible merged-projection fallbacks (50% success) | `+0.054099`; `1.055589` | `0.170481` / `0.179538` | `11.861 ms` → `13.754 ms` (`0.862x`) | Fails equivalence and timing screens |
| Jason Laguna, TileM128, corrected tokenizer | 4 × 512 tokens; 2,044 scored positions | 376/376 fused successes, zero fallbacks/errors; baseline used 376/376 loop calls | `-0.109419`; `0.896355` | `0.734445` / `0.761243` | `3339.246 ms` → `1066.132 ms` (`3.132x`) | Fails distribution-equivalence gate |
| Jason Laguna, TileM256, exploratory only | 1 × 128 tokens; 127 scored positions | 94/94 fused successes, zero fallbacks/errors | `-0.383291`; `0.681615` | `1.102434` / `1.195316` | `2440.166 ms` → `1441.300 ms` (`1.693x`) | Not a formal gate; see caveat below |
| LFM2.5 8B-A1B partial K16 v2, TileM128, chunked prefill | 1 × 17 tokens; 16 scored positions | baseline loop 4/4; fused 4/4; zero fallbacks/errors; all nine integrity gates passed | `+0.054964`; `1.056503` | exact full vocabulary: `0.247178` / `0.288945` | not measured | Rejects enablement; teacher→candidate mean KL worsened by `+0.131101` |

**The wall times above are offline `generate(..., max_tokens=1)` request times, including scheduler, prefill, logits, and one-token output processing. They are not streaming TTFT and are not served SLO evidence.**

The corrected TileM128 short-prompt corroboration (2 × 128 tokens, 254 positions) also failed the KL gate: `0.921599` baseline→fused and `1.118585` reverse, despite a `2.940x` offline wall-time speedup. Its ΔNLL was `-0.151407`. The apparent NLL/PPL improvements do not authorize promotion: Jason's artifact has no BF16-teacher A/B here, no task validation was run, and the fused probability distribution moved substantially.

An earlier dense 8 × 512 BF16-teacher screen was mixed rather than evidence of a simple quality regression: baseline→fused coarse KL was `0.161079`, fused target NLL was `0.001084` worse, and dense wall time regressed from `19.664 ms` to `22.020 ms` (`0.893x`), while coarse teacher→candidate KL moved from `2.029680` to `2.013154` (slightly closer to the teacher). That schema-v3 result lacks the current artifact/tokenizer/dtype and full-vocabulary attestations. It is rejection/screening evidence only and cannot greenlight the path.

The LFM smoke closes the missing teacher-backed integration loop without overstating its sample. The corrected partial artifact and the unquantized BF16 teacher have matching config/tokenizer identity; the teacher has only BF16 parameters and no quantization config; prompt-logprob coverage is exact over all 128,000 output IDs. Chunked prefill was enabled because that is LFM's supported vLLM contract. The fused path still moved the distribution by orders of magnitude more than the `1e-4` promotion limit and worsened both target PPL and teacher KL. That is sufficient stop-gate evidence: the path cannot be promoted on the present record, so no additional GPU time was spent trying to greenlight it.

TileM256 is operator-qualified but not model-qualified. Its only model screen used the artifact's legacy tokenizer-regex behavior, one prompt, and one timing repeat. It remains useful as exploratory paired evidence, but it is superseded for promotion decisions by the corrected-tokenizer TileM128 runs and must not be treated as a TileM256 quality result.

## Operator gate and fixes

The final installed-wheel fail-closed CUDA module run passed **68/68 tests with
zero skips, failures, or errors in 127.64 seconds** on GB10, including its cold
JIT build during collection. The wheel was force-installed outside either
source checkout; the tests and producer fixture were separately mounted
read-only. Coverage includes:

- per-symbol SASS attestation that each concrete TileM128 and TileM256 fused kernel contains `OMMA.SF.16864`, with no `QMMA` instruction in the module;
- every fused-eligible product rung K12–K24 and signed rung S12–S20 under both
  layout v1 and two-tier v2, plus public-binding rejection of the first
  oversized signed LUT at S21;
- dense M from 32 through 2,048, ragged N/K shapes, and padded row-stride views;
- grouped TileM128 and TileM256 parity against the dense fused reference;
- Jason's exact expert-stage shapes `(N,K)=(2048,3072)` and `(3072,1024)`, non-monotonic expert IDs, a non-default CUDA stream, and TileM256 graph capture plus two bitwise-equal replays;
- malformed dtype, device, contiguity, shape, capacity, and layout rejection, plus a device-side fail-closed trap for grouped expert IDs outside `[-1, E-1]`;
- a full two-stage routed-MoE oracle for TileM128/256 at `T=tile-1`, `tile`, and `tile+1`, covering empty-expert, hotspot, non-monotonic, and balanced top-2 routes. The real grouped path is bitwise BF16-equal to an independently constructed per-expert dense-native reference through activation, route weighting, and combine;
- vLLM activation packing and scale layout. The apparent byte mismatch was only E2M1 `+0` versus `-0`; decoded values are elementwise identical, while any non-zero-code disagreement remains a hard failure.

The stagewise oracle localizes the model shift. A→B changes only legacy fp32-scale activation QDQ to native E2M1/UE4M3 activation factors; B→C changes only BF16-emulated weight multiplication to the native block-scaled MMA/accumulation; C→D changes only independent dense-native expert calls to the production grouped routing/addressing path:

| TileM | Stage | A→B relative L2 | B→C relative L2 | C→D relative L2 / BF16 bit mismatches |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | `7.317546%` | `0.265724%` | `0` / `0` |
| 128 | 2 | `6.148441%` | `0.263228%` | `0` / `0` |
| 256 | 1 | `7.386171%` | `0.179576%` | `0` / `0` |
| 256 | 2 | `6.192255%` | `0.282756%` | `0` / `0` |

The activation representation dominates by more than an order of magnitude. Grouped expert addressing, padding, and routing add exactly zero difference on the tested matrix.

The review fixed these concrete defects and test gaps:

1. Replaced the templated grouped host runner with concrete TileM128 and TileM256 runners. nvcc 13 had emitted duplicate unsigned-layout kernel identities with an unregistered host stub, causing `cudaFuncSetAttribute`/CUTLASS initialization to fail with `invalid resource handle`.
2. Passed the current CUDA stream through CUTLASS initialization and execution for dense, reference, and grouped calls, and surfaced CUTLASS/CUDA status on failure.
3. Enforced complete tensor device, dtype, contiguity, shape, storage-capacity, scale-plane, LUT, and same-device contracts; propagated `num_experts` and trapped invalid expert IDs before any packed-weight gather.
4. Removed the unchecked public `debug_out` pointer from the production binding.
5. Made SASS attestation locate a compatible CUDA 13.1 `nvdisasm` and attribute instructions to both concrete fused symbols rather than accepting a module-wide false positive.
6. Added the exact Jason stage-shape, full-routing, stagewise-decomposition, non-default-stream, and CUDA-graph tests described above.
7. Corrected `PRISMAQUANT_PRELOAD_FUSED=1` to attempt both independent fused extensions. The previous code loaded only the FP8 extension, so NVFP4 A/B residency was not matched; validation separately fails closed unless its required FP4 module loaded.
8. Registered LFM2-MoE's top-level expert loader so selected stacked CB tensors reach Gridbook's fill guard.
9. Hardened the same-process A/B harness (schema v5): it now attests the actually imported Gridbook Python/CUDA sources and extension; hashes local candidate/teacher weights, config, tokenizer, quant config, and codebook; requires exact config/tokenizer identity and an unquantized all-BF16 teacher; rejects duplicated/misaligned prompts and logprobs; distinguishes explicit measurement-only, coarse screening, and promotion states; and offers cardinality-checked exact full-vocabulary KL. Hub candidates and every Gridbook sidecar/header read are pinned to one immutable model commit.
10. Scoped, reset, attested, and exception-safely restored both process-stable
    fused selectors inside the explicit same-process A/B harness. Ordinary
    serving still rejects a mid-process activation-contract change.
11. Added visible-row and backing-storage-capacity validation to the dense
    public binding, including hostile narrow-view and post-creation storage
    shrink tests.
12. Centralized fused FP4 LUT sizing across dense and MoE dispatch. Product
    K24 and signed S20 are the exact 16-KiB boundary; signed S21 and above now
    fail closed before dispatch, while the CUDA binding independently rejects
    an oversized direct call.
13. Keyed the fused JIT module and build directory by source/header digests,
    concrete GPU architecture, Python/Torch/CUDA/compiler ABI, and CUTLASS
    sentinel headers; validate the loaded module identity and include all
    required sources in wheel/sdist/install checks.

The first partial LFM artifact failed before any kernel ran because untouched BF16 expert banks had been rewritten into an aggregate representation its vLLM loader could not consume. That failure exposed the Gridbook loader gap fixed here and a producer-side partial-export defect fixed in PrismaQuant PR #40. A corrected 14,537,263,589-byte artifact then passed exact recursive inventory and codebook verification: four selected layers produced eight packed U8 CB tensors replacing exactly 384 selected per-expert tensors, while all 1,918 common untouched tensors retained their dtype and shape. vLLM loaded and executed that artifact in the LFM row above. The original failed artifact remains an integration regression, not model evidence.

These fixes make the arithmetic and dispatch evidence credible. They do not erase the activation-contract change or supply the missing served-quality and served-performance evidence.

## Reconsideration gates

Reconsider a narrowly shape-gated default only after all of the following pass:

1. Teacher-backed full-vocabulary KL/PPL and representative task A/Bs in the same serving session, including a dense model of at least 4B and representative routed-expert models. Apparent self-PPL improvement without a teacher or task result is insufficient.
2. A served workload matrix covering prompt-length distribution, concurrency, chunked prefill, batched/speculative decode at shipped M and acceptance, and MoE routed-token histograms. Report p95 TTFT separately from p95 ITL/p05 TPS.
3. At least a 10% representative p95 TTFT improvement with uncertainty excluding parity, with no workload cell regressing p95 TTFT, p99 ITL, or throughput by more than 5%.
4. Routing- and padding-aware TileM128/TileM256 gates. One Laguna geometry and one selected TileM cannot justify a blanket grouped default.
5. Production dispatch telemetry that proves the fused path ran and reports shape-specific fallbacks. Dense merged projections currently make only half of the measured calls eligible.
6. Revalidation on every supported deployment GPU/runtime/build identity. The present evidence attests only the pinned GB10 environment below.

Until those gates pass, the performance result is promising for grouped prefill research, but the safe product decision is unchanged: explicit opt-in only.

## Reproducibility record

- GPU: NVIDIA GB10, compute capability 12.1, 130,595,930,112 bytes device memory.
- Operator-test and model-A/B image: `vllm-node@sha256:d0840ff0e0ba1899a51bf4cb473f43d0c765288b8de708080ad9d95768615141`.
- Model A/B image: `vllm-node@sha256:d0840ff0e0ba1899a51bf4cb473f43d0c765288b8de708080ad9d95768615141`.
- Model A/B runtime: CUDA 13.0, PyTorch `2.11.0+cu130`, vLLM `0.23.1rc1.dev764+g54b16d8a9.d20260703`, TP=1, BF16, eager mode, and prefix caching off. Dense/Laguna rows used chunked prefill off; the authoritative LFM row used chunked prefill on with `max_num_batched_tokens=33`.
- Gridbook release-source worktree: `fix/fused-nvfp4-validation`, based on
  `777dcc46f99c4914c4d2c50dd084c5df7fc31b05`. CUDA source SHA-256
  `b76cc186965f2ae232d54d72053c217fc6058479bc6948ad82fc8d59b9cc8c08`;
  collective-header SHA-256
  `7670117d8eb16310da9e2568a2641effef5dee6d232a29c04e33a60b54879347`;
  exhaustive test-matrix SHA-256
  `ba129ea94813f2a1dcf3cd3d0df5d24a41cb3a02213a2706050ab4accfee448f`.
- Final release-source JIT identity:
  `ea1619b01b363cdb916806e6203b3abd0519ed326e0d6aebfd59954775352604`;
  module ABI schema `1`; extension SHA-256
  `623439250ec30daca85c6a731c159273847f9a24870bf5335ee947fa2c7621c3`.
- The LFM model A/B attested the immediately preceding CUDA source hash
  `1a8178a9e8a40b3cc8632d2b0890fd0b010f10419778e3e545f8cd93139fe57b`
  and extension hash
  `6646bee7f74be73cd28a7112a76f50639a63fd3fe64998d69b891310d5692239`.
  The subsequent CUDA-source change added only host-side packed-storage
  validation; the device mainloop/header and numerical execution contract did
  not change. The exhaustive 68-test gate above rebuilt and requalified the
  release source.
- Dense model: local Qwen3-0.6B `NVFP4_CB_K16`, two-tier layout v2 (`type_size=73`), `/home/rob/dq-runs/nvfp4-cb-phase0/serve/nvfp4cb_k16_v2`.
- Grouped model: `JasonW2025/Laguna-S-2.1-PrismaQuant-gridbook-2.6bit-vllm`, Hugging Face revision `1ae37c430f905b02da3367e5ee309c684196e513`, local snapshot `/home/rob/dq-runs/jason-laguna-2.6bit`.
- Teacher-backed grouped model: LFM2.5 8B-A1B BF16 teacher plus a four-layer `NVFP4_CB_K16` v2 partial artifact (`model.layers.{2,8,14,23}` gate/up and down expert stacks), exact export size 14,537,263,589 bytes.
- Evidence root: `/home/rob/dq-runs/evidence/gridbook-0.4.0-fused-nvfp4-gb10`.

Evidence SHA-256 values:

```text
d44cc164e2aa140372428d0fa5cd037d5cfbcd73ac06754d5bed9ec72d7ef3a7  cuda-operator-tests-installed-wheel-v0.4.1.xml (final 68-case release gate)
c60f6fa59c56660686075f35ae3a66fb428376716af68dd3e91092711ac38bf7  cuda-operator-tests-stagewise-final.log (prior 33-case stagewise run)
211d1cd7fb5705d172353d8efb6348c02be32142ed9555fdd4c77c343ddb5896  cuda-operator-tests-stagewise-final.xml (prior 33-case stagewise run)
4207be397ab6a26caed1dbbaa941e91506215719d3f8677acfeb2ac5cece5516  dense-k16-v2-m128-ab.json
230570f6e05188af08cf85914d0dfdadbe7069077fa99a83c7a40d8113545bcd  jason-laguna-moe128-t128-fixed-tokenizer-ab.json
a076e798d49c52cde30dc640787b71595641056472b04d25f57b43a656627180  jason-laguna-moe128-t512-fixed-tokenizer-ab.json
54202c30905113bf64e0e5bb91c3b279904d455b02b3b4a8a594d1ac11a6feaa  jason-laguna-moe256-t128-ab.json
051a82082bf8b0af480dfc08a98736525c6899d6c395667f453d1f792cf9d02e  lfm25-moe128-t17-chunked-teacher-full-vocab-smoke.json
58606f4b93f0c0bf2c98949cbdc70649c3e98ea1bae2a91aba7ef1e36772c05e  lfm25-moe128-t17-chunked-teacher-full-vocab-smoke.log
5df2d99ad65df6bb741c4dee4fd9c0770231c55fd7b4b9a82a86dc46a618e6ee  lfm25-k16-v2-l4-smoke evidence manifest
cfed087bf1c6c6ba37234ddf6c7b676402f00883a71c72082495ae6542b9ae6a  corrected LFM cb_codebooks.pqcb
6646bee7f74be73cd28a7112a76f50639a63fd3fe64998d69b891310d5692239  LFM-run fused FP4 extension
```
