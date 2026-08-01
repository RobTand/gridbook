# Fused NVFP4 enablement decision — 2026-07-31

Last updated: 2026-08-01.

Status: **keep both fused NVFP4 paths explicit opt-ins**. The dense path remains
behind `PRISMAQUANT_CB_FUSED_FP4`; the grouped-MoE path remains behind
`PRISMAQUANT_CB_FUSED_FP4_MOE`. The CUDA operator gate is green and the new
dense `static_lsq` policy is both materially faster and much more accurate than
the original fixed-residual policy. Its current model evidence is nevertheless
mixed across prompt lengths, covers only Qwen3-0.6B, and does not qualify MoE.
Neither path should become the default yet.

## Decision basis

The fused kernel is bit-exact with the native NVFP4 reference when both consume the same packed E2M1 activations and UE4M3 scale factors. That does not make it numerically equivalent to Gridbook's current served baseline:

- baseline: fp32-emulated group-scale FP4 activation QDQ;
- fused: native NVFP4 global fp32 scale plus per-group UE4M3 factors.

The A/B therefore measures the complete activation-and-kernel execution contract, not a weight-only kernel substitution. A model-level distribution change can coexist with a correct fused kernel. That is what the evidence shows.

### Current dense `static_lsq` evidence

`static_lsq` keeps the producer-attested global scale `G` and emits the same
packed E2M1 activation values and UE4M3 scale-factor bytes as vLLM's native
fixed-`G` quantizer. It changes only the existing per-row EVT residual to the
closed-form least-squares optimum for those fixed bytes. It serializes no new
metadata and creates no second weight, decoder, or GEMM: one policy-templated
activation quantizer feeds the existing `cb_fused_fp4_prefill_mm_scaled`
operator and the existing packed weights/codebooks. `static_lsq_midm` is the
same contract limited to `16 < M <= 128`.

The current predeclared dense screen uses the intentional activation-contract
change as its premise rather than pretending it should be numerically
identical. Its limits are pair KL `<= 0.25`, mean NLL regression
`<= ln(1.005) = 0.0049875`, PPL regression `<= 0.5%`, exact teacher-to-fused
mean KL `<= 0.501`, teacher-KL regression `<= 0.01`, and offline timing speedup
`>= 1.10x`. The exact teacher threshold is useful only on its predeclared
sample; a baseline that itself exceeds it cannot turn the candidate-specific
relative gate into a failure.

| Qwen3-0.6B `NVFP4_CB_K24` v3-MSE screen | Sample and KL mode | Verified dense dispatch | Mean ΔNLL; PPL ratio | Pair / teacher KL | Offline one-token wall time | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Short, window seed 42 | 6 × 128; 762 scored; exact full vocabulary | 4,704/4,704 fused, zero fallback/error; TileM128 only | `-0.008817`; `0.991222` | pair `0.240804`; teacher baseline `0.490901` → fused `0.487344` | `13.274 ms` → `8.981 ms` (`1.478x`) | Passes every predeclared gate |
| Long, window seed 43, real chunking | 2 × 512; 1,022 scored; exact full vocabulary; `max_num_batched_tokens=256` | 2,240/2,240 fused, zero fallback/error; 1,120 TileM128 + 1,120 TileM256 | `+0.016062`; `1.016192` | pair `0.208840`; teacher baseline `0.511352` → fused `0.515249` (`+0.003898`) | `31.643 ms` → `18.180 ms` (`1.741x`) | Fails NLL and PPL gates; relative teacher gate passes |
| Long screen, window seed 43, real chunking | 32 × 512; 16,352 scored; top-64 plus tail; `max_num_batched_tokens=256` | 7,168/7,168 fused, zero fallback/error; 3,584 TileM128 + 3,584 TileM256 | `+0.002631`; `1.002635` | coarse pair `0.161632` / reverse `0.162403`; no teacher | not measured | PPL point estimate passes, but measurement-only and not promotion-eligible |

In the top-64 row, target NLL and PPL are still exact for every submitted next
token: the harness preserves the forced target entry before truncating the rest
of the distribution. Only pair KL uses the top-K-plus-tail approximation. The
retained-mass means were `0.876861` baseline and `0.877327` fused, but their
minima were `0.159888` and `0.016670`; that is why its coarse KL can reject a
candidate but cannot greenlight one. The preserved short top-1024 screen makes
the limitation concrete: its coarse teacher KL was baseline `0.437847` → fused
`0.442252` (fused worse by `0.004405`), while exact full-vocabulary KL on the
same windows was `0.490901` → `0.487344` (fused better by `0.003557`). The
coarse direction flip cannot arbitrate promotion.

The 32-window result explains why the two-window failure cannot be treated as a
stable long-context estimate: 16 prompt deltas were positive and 16 negative,
and a prompt-cluster 95% t interval for mean ΔNLL was
`[-0.012012, 0.017275]` (implied PPL ratio `[0.988060, 1.017425]`). That
interval crosses both parity and the 0.5% regression limit. The honest decision
is therefore not that long-context quality passed; it is that the point estimate
is encouraging and the current sample does not establish the gate. The one-sided
95% upper bound is ΔNLL `0.014805` (PPL `+1.4915%`), and the non-inferiority
t-test against the 0.5% limit gives `p=0.3725`. A larger exact teacher-backed
run, a >=4B dense model, tasks, and routed-MoE coverage are still required.
The preserved eight-window long screen also moved from a `+0.6872%` PPL point
estimate to `+0.2635%` when expanded to 32 windows, another reason to retain all
intermediate reports and avoid treating either small point estimate as final.

The wall times in this section and below are offline
`generate(..., max_tokens=1)` request times, including scheduler, prefill,
logits, and one-token output processing. They are useful end-to-end integration
evidence, but are not streaming TTFT and are not p95 served-SLO evidence. Raw
kernel timings are a still narrower layer: the optimized fused v2 decoder and
TileM routing improve operator time, but the raw fused operator has not reached
native-microkernel parity. Neither layer of timing may be relabelled as served
parity.

### Earlier fixed-residual and rowwise stop evidence

The original configured promotion screen required mean KL at most `1e-4`, mean NLL regression at most `ln(1.005) = 0.0049875`, PPL regression at most 0.5%, and an offline timing speedup of at least 1.10x. That strict equivalence screen correctly rejected the original fixed-residual path, but it is not silently reused as the quality objective for an intentional activation-contract change. Older rows use a paired top-1024-plus-tail approximation. The LFM smoke uses cardinality-checked exact full-vocabulary KL and is `measurement_only`: it can reject enablement, but its one short prompt is not a model-quality characterization.

| Screen | Sample | Verified dispatch | Mean ΔNLL; PPL ratio | Mean KL (mode), baseline→fused / reverse | Offline one-token wall time, baseline→fused | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Dense Qwen3-0.6B, NVFP4-CB K16 v2 | 4 × 128 tokens; 508 scored positions | 896 fused successes, 0 errors; 896 ineligible merged-projection fallbacks (50% success) | `+0.054099`; `1.055589` | `0.170481` / `0.179538` | `11.861 ms` → `13.754 ms` (`0.862x`) | Fails equivalence and timing screens |
| Jason Laguna, TileM128, corrected tokenizer | 4 × 512 tokens; 2,044 scored positions | 376/376 fused successes, zero fallbacks/errors; baseline used 376/376 loop calls | `-0.109419`; `0.896355` | `0.734445` / `0.761243` | `3339.246 ms` → `1066.132 ms` (`3.132x`) | Fails distribution-equivalence gate |
| Jason Laguna, TileM256, exploratory only | 1 × 128 tokens; 127 scored positions | 94/94 fused successes, zero fallbacks/errors | `-0.383291`; `0.681615` | `1.102434` / `1.195316` | `2440.166 ms` → `1441.300 ms` (`1.693x`) | Not a formal gate; see caveat below |
| LFM2.5 8B-A1B partial K16 v2, TileM128, chunked prefill | 1 × 17 tokens; 16 scored positions | baseline loop 4/4; fused 4/4; zero fallbacks/errors; all nine integrity gates passed | `+0.054964`; `1.056503` | exact full vocabulary: `0.247178` / `0.288945` | not measured | Rejects enablement; teacher→candidate mean KL worsened by `+0.131101` |

The corrected TileM128 short-prompt corroboration (2 × 128 tokens, 254 positions) also failed the KL gate: `0.921599` baseline→fused and `1.118585` reverse, despite a `2.940x` offline wall-time speedup. Its ΔNLL was `-0.151407`. The apparent NLL/PPL improvements do not authorize promotion: Jason's artifact has no BF16-teacher A/B here, no task validation was run, and the fused probability distribution moved substantially.

An earlier dense 8 × 512 BF16-teacher screen was mixed rather than evidence of a simple quality regression: baseline→fused coarse KL was `0.161079`, fused target NLL was `0.001084` worse, and dense wall time regressed from `19.664 ms` to `22.020 ms` (`0.893x`), while coarse teacher→candidate KL moved from `2.029680` to `2.013154` (slightly closer to the teacher). That schema-v3 result lacks the current artifact/tokenizer/dtype and full-vocabulary attestations. It is rejection/screening evidence only and cannot greenlight the path.

The LFM smoke closes the earlier missing teacher-backed integration loop without overstating its sample. The corrected partial artifact and the unquantized BF16 teacher have matching config/tokenizer identity; the teacher has only BF16 parameters and no quantization config; prompt-logprob coverage is exact over all 128,000 output IDs. Chunked prefill was enabled because that is LFM's supported vLLM contract. The original fixed-residual fused path moved the distribution by orders of magnitude more than the `1e-4` equivalence limit and worsened both target PPL and teacher KL. That remains stop-gate evidence for that activation policy; the later dense `static_lsq` result does not retroactively qualify it or supply the missing MoE `static_lsq` validation.

The MoE serving selector now reuses that same `static_lsq` quantizer for both
stages and feeds its output to the unchanged grouped kernel. A current-source
4×128, chunk-64 attempt on the corrected partial LFM artifact was intentionally
rejected as invalid evidence: all 128 candidate attempts failed closed to the
loop. The artifact predates the attested activation contract and contains zero
serialized `input_global_scale` tensors, so Gridbook had no lawful fixed `G`
for either stage. Its identical outputs and `0.9995x` timing are fallback
telemetry, not LSQ quality/performance evidence. A producer re-export with
stage-specific serialized scales is required before the MoE A/B can run.

The manually forced grouped TileM256 path is operator-qualified but not
model-qualified. Its only grouped-model screen used the artifact's legacy
tokenizer-regex behavior, one prompt, and one timing repeat. Dense automatic
routing now exercises both TileM128 and TileM256 in the chunked K24 runs above,
with bit-exact cross-tile operator tests and explicit route telemetry. That does
not qualify grouped-MoE TileM256 or transfer dense quality evidence to MoE.

## Operator gate and fixes

The original installed-wheel fail-closed CUDA module run passed **68/68 tests
with zero skips, failures, or errors in 127.64 seconds** on GB10, including its
cold JIT build during collection. At current HEAD, the combined pinned-container
gate passed **99/99 tests** in 125.12 seconds across fused prefill, static-LSQ,
rowwise, and activation-reference suites. The installed wheel was
force-installed outside either source checkout for the release gate; its tests
and producer fixture were separately mounted read-only. Coverage includes:

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
- fixed-`G` LSQ payload equality with vLLM for BF16 and FP16, exact midpoint and
  underflow boundaries, independently checked row-optimal residuals, zero-row
  fallback, non-default streams, CUDA graphs, malformed inputs, and bit-exact
  stock-native versus CB-fused output from the same activation payload;
- dense occupancy routing, including bit-exact TileM128/256 results, ragged and
  multi-LUT shapes, route telemetry integrity, and fail-closed v5/v6 chunked-
  prefill contract attestation.

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
14. Added fixed-`G` least-squares residual fitting as one policy in the shared
    native-NVFP4 activation quantizer. Its E2M1/SFA payload remains byte-equal to
    vLLM; only the existing row residual changes, and the policy feeds the sole
    existing weight decoder/GEMM. This removes the accuracy defect without
    duplicating weights, serialized metadata, or a matmul implementation.
15. Reworked the two-tier v2 scale decoder so two lanes own a row and each lane
    composes four adjacent scales. It removes redundant exponent/code loads
    without changing the compose table or stores. Interleaved raw measurements
    improved all 12 production cells by 7.9–23.4%; this is operator evidence,
    not a claim of native or served parity.
16. Added an occupancy-aware dense TileM selector. For `M >= 256`, it chooses
    the already-qualified TileM256 runner only when
    `ceil(M/256) * ceil(N/128) >= ceil(2*SM_count/3)`; otherwise it retains
    TileM128. The SM count is cached per device and probe failure conservatively
    selects 128, with no tensor read or synchronization. Interleaved raw tests
    gained 22.57–33.12% on selected wide cells and 50.32% on selected narrow
    M=1024, while narrow cells where TileM256 lost 19.81–21.69% stayed on
    TileM128. Both choices call existing concrete runners; no GEMM body was
    copied.
17. Extended dense dispatch telemetry to attest TileM, shape, candidate CTA
    count, and cached SM count on every fused success, and made the A/B harness
    require exactly one legal route record per success.
18. Reused the same fixed-`G` LSQ quantizer in grouped MoE for explicit
    TileM128/256 selectors. Both stages call the unchanged grouped GEMM and
    packed weights. Eligibility now caches and reports the exact fail-closed
    reason, and the A/B report aggregates those reasons rather than making a
    zero-difference fallback look like validation.

The first partial LFM artifact failed before any kernel ran because untouched BF16 expert banks had been rewritten into an aggregate representation its vLLM loader could not consume. That failure exposed the Gridbook loader gap fixed here and a producer-side partial-export defect fixed in PrismaQuant PR #40. A corrected 14,537,263,589-byte artifact then passed exact recursive inventory and codebook verification: four selected layers produced eight packed U8 CB tensors replacing exactly 384 selected per-expert tensors, while all 1,918 common untouched tensors retained their dtype and shape. vLLM loaded and executed that artifact in the LFM row above. The original failed artifact remains an integration regression, not model evidence.

These fixes make the arithmetic, dispatch, and dense short-prompt evidence
credible. They do not erase the activation-contract change, turn raw timing
into served timing, or supply broader dense/MoE served-quality evidence.

## Reconsideration gates

Reconsider a narrowly shape-gated default only after all of the following pass:

1. Teacher-backed full-vocabulary KL/PPL and representative task A/Bs in the same serving session, including a dense model of at least 4B and representative routed-expert models. Apparent self-PPL improvement without a teacher or task result is insufficient.
2. A served workload matrix covering prompt-length distribution, concurrency, chunked prefill, batched/speculative decode at shipped M and acceptance, and MoE routed-token histograms. Report p95 TTFT separately from p95 ITL/p05 TPS.
3. At least a 10% representative p95 TTFT improvement with uncertainty excluding parity, with no workload cell regressing p95 TTFT, p99 ITL, or throughput by more than 5%.
4. Extend the implemented dense occupancy gate to a routing- and padding-aware
   grouped-MoE policy, then validate both concrete tiles over shipped routed-token
   histograms. A dense selector and one Laguna geometry cannot justify a blanket
   grouped default.
5. Production dispatch telemetry that proves the fused path ran and reports
   shape-specific fallbacks over every representative artifact, not only the
   fully eligible K24 dense canary.
6. Revalidation on every supported deployment GPU/runtime/build identity. The present evidence attests only the pinned GB10 environment below.

Until those gates pass, `static_lsq` is a promising dense experiment and the
grouped path remains research-only. The safe product decision is unchanged:
explicit opt-in only.

## Reproducibility record

Current dense `static_lsq` record:

- Gridbook source: `fix/fused-nvfp4-accuracy-perf` at
  `e5eac95180dc73d989663fa544e6f4cfdf44c7f0`; fused CUDA source SHA-256
  `0a713fd4e8e6b48cb21a05577df73129b0c001bacc3004afa4bd63884c79e067`;
  collective-header SHA-256
  `2a96954881bff0b36b1d2f778a7a21266200303c8c5bbdbb00c9a8ab83d578ac`;
  schema-v5 harness SHA-256
  `7d5ed9827188d30dc0fd390e9937d02854bd7272ea97b97baa79041132aba0e9`.
- Fused extension module:
  `pq_cb_fused_fp4_d9614dc7d9b56fbf69d8216dd705ef564c50a622c0988c0be4f1652655564c54`;
  extension SHA-256
  `2797a22bee319bb8caca0532e509b2468a76b44f094f331872d9c890e7f98bc1`.
- Candidate: local Qwen3-0.6B `NVFP4_CB_K24` v3-MSE,
  `/home/rob/dq-runs/nvfp4-cb-phase0/serve/nvfp4cb_k24_v3_mse`;
  teacher `/home/rob/models/Qwen3-0.6B`; WikiText input SHA-256
  `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`.
- Evidence root:
  `/home/rob/dq-runs/evidence/gridbook-k24-static-lsq-20260801`.
- Runtime image and GPU are the pinned GB10 environment below. Chunked prefill
  was explicitly requested, resolved enabled, and compatible with the model's
  official contract in all current runs; the 512-token cells forced real
  256-token chunks.

Earlier fixed-residual/rowwise record:

- GPU: NVIDIA GB10, compute capability 12.1, 130,595,930,112 bytes device memory.
- Operator-test and model-A/B image: `vllm-node@sha256:d0840ff0e0ba1899a51bf4cb473f43d0c765288b8de708080ad9d95768615141`.
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
51969c54bbb179da2a2babad84892e89cee4d6c30a5da20e38dbf1e2f14c75b4  qwen06-k24-static-lsq-v5-s128-seed42-coarse-screen.json (superseded screen; preserved)
4cd72570e9b0c52c5462dd1b60fc284c5166c9978db2a5241e191171e8b45da6  qwen06-k24-static-lsq-v5-s128-seed42-exact.json
25bb5c80759958b3e3d17cfdace93ba0c1ff529a23e1a48afa8411957bcf4330  qwen06-k24-static-lsq-v5-s512-chunk256-seed43-exact.json
929ba22455ca0d3d81617787437d24258d71785d399a88d7aa125893e947b247  qwen06-k24-static-lsq-v5-s512-chunk256-seed43-n8-screen.json (intermediate screen; preserved)
114fac9e4a98ad13210b95c134824984c78d9f142aa579cc0415563edc36dd17  qwen06-k24-static-lsq-v5-s512-chunk256-seed43-n32-screen.json
602411da3d1286187b1e38352ef3d36ba7b2062090380416166595d9c482ac77  lfm25-k16-lsq-moe128-s128-chunk64-n4.json (invalid fallback evidence; preserved)
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
