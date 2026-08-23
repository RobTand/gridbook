# Troubleshooting

Every entry below is keyed to something you will actually see — a log line, an
exception, or a number that is wrong. If your symptom is not here, please open an
issue; [CONTRIBUTING.md](../CONTRIBUTING.md) lists what to include.

**First, verify the native kernel set.** Current Gridbook has no slow serving
lane: if a required CUDA/CUTLASS extension did not load, the operation fails
closed. The check in
[INSTALL.md](INSTALL.md#verify-the-install) answers that directly without
serving a model (a cold JIT build still takes roughly 30 seconds per extension).

**A note on grepping.** The plugin's log lines and environment variables use the
project's older `prismaquant` prefix, not `gridbook` — searching your vLLM log
for "gridbook" will find nothing. Search for **`[prismaquant-cb]`**.

- [The native extension did not load](#the-native-extension-did-not-load)
- [Broken install: CUDA sources missing from the package](#broken-install-cuda-sources-missing-from-the-package)
- [Incompatible JIT extension: the module loaded but has the wrong API](#incompatible-jit-extension-the-module-loaded-but-has-the-wrong-api)
- [Fused prefill extension unavailable](#fused-prefill-extension-unavailable)
- [Invalid quantization method: gridbook](#invalid-quantization-method-gridbook)
- [Missing codebook sidecar or quant config](#missing-codebook-sidecar-or-quant-config)
- [Codebook provenance mismatch](#codebook-provenance-mismatch)
- [Out of memory on a 32 GB card](#out-of-memory-on-a-32-gb-card)
- [Out of memory / box hang on a DGX Spark](#out-of-memory--box-hang-on-a-dgx-spark)
- [Serving is far slower than the published numbers](#serving-is-far-slower-than-the-published-numbers)
- [Do I really need `--enforce-eager`?](#do-i-really-need---enforce-eager)
- [Non-Blackwell GPU: what breaks](#non-blackwell-gpu-what-breaks)
- [Tensor parallel (`tp > 1`)](#tensor-parallel-tp--1)
- [Expert parallel (`-tp N --enable-expert-parallel`)](#expert-parallel--tp-n---enable-expert-parallel)
- [The model loads but generates garbage](#the-model-loads-but-generates-garbage)
- [Other exceptions you may hit](#other-exceptions-you-may-hit)
- [The first model load stalls for kernel compilation](#the-first-model-load-stalls-for-kernel-compilation)
- [Benchmark numbers move between runs](#benchmark-numbers-move-between-runs)

---

<a id="the-cuda-extension-did-not-load-triton-fallback"></a>

## The native extension did not load

**Symptom** — on stderr, during model load:

```
[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV extension could not be
built (<ErrorType>: <message>); native Gridbook execution is unavailable and
serving will fail closed. To enable the native path: install a CUDA toolchain
matching your torch build ...
```

(Older plugin builds may instead say that they are falling back to Triton. That
message identifies the retired fail-soft release, not current behavior.)

**What it means** — a capability probe could not build/load native code. The
first serving call that requires it raises `NativeKernelUnavailableError`; it
does not run a different kernel or continue at prototype speed. Treat this as a
broken serving environment even if model construction progressed far enough to
emit more logs.

For historical scale, the retired Triton dense decode prototype measured **4.20
tok/s** on the 27B against **10.28** on the CUDA GEMV. MoE lost more by an
artifact-dependent amount. Those before/after measurements explain why the
fallback was removed; there is no current Triton arm to reproduce from a serving
switch. [BENCHMARKS.md](BENCHMARKS.md#retired-triton-path-historical-cost)
states which parts were measured and which were inferred.

**Causes and fixes**, in order of likelihood:

| The `<ErrorType>` you see | Cause | Fix |
|---|---|---|
| `RuntimeError` mentioning `nvcc`, or `FileNotFoundError: nvcc` | No CUDA compiler in the *serving* process. Many vLLM images ship the runtime but not the toolkit. | Install a CUDA toolkit matching your torch build (distro `cuda-toolkit`, or the `nvidia-cuda-nvcc-*` wheel) and make sure `nvcc` is on `PATH` or `CUDA_HOME` points at it. Verify with `nvcc --version` **inside the container**. |
| `PermissionError` / `OSError` on the build directory | The build cache path is not writable by the serving user. | Set `PRISMAQUANT_CB_EXT_DIR` to a writable directory. |
| A compiler error from `nvcc` | Toolchain/torch mismatch, or an `nvcc` too old for your GPU's architecture. | `nvcc` 13.0 is the tested toolchain. Match the CUDA major version your torch was built against. |
| Anything, but only inside a container that used to work | The ephemeral build cache is being rebuilt and failing. | See [persisting the cache](INSTALL.md#persisting-the-jit-build-cache). |

No current environment switch enables Triton. `PRISMAQUANT_CB_DECODE` and
`PRISMAQUANT_CB_EXPAND` have no reader left in the code at all, so an inherited
`=triton` setting is inert rather than dangerous — but delete it from old
scripts/model-card commands anyway, because a variable that looks like a
bisection lever and silently does nothing is worse than no lever.

---

## Broken install: CUDA sources missing from the package

**Symptom** — on stderr:

```
[prismaquant-cb] ERROR: broken gridbook install — gridbook is installed without
its CUDA sources: ['cb_gemv.cu'] not found under <path>. This is a packaging
defect, not a missing CUDA toolchain — reinstall gridbook ...
```

Older installs produce a bare `FileNotFoundError` naming a path with `/../csrc/`
in it, e.g. `FileNotFoundError: .../site-packages/gridbook/../csrc/cb_gemv.cu`.

**Cause** — the `.cu` sources are not present in the installed package. This was
a real defect in early builds: `csrc/` lived at the repo root and only the Python
package was installed, so any non-editable `pip install` produced a
working-but-slow server in that retired release (reported in
[issue #1](https://github.com/RobTand/gridbook/issues/1)). The sources now ship
*inside* the package at `gridbook/csrc/`.

**Fix**

```bash
pip install --force-reinstall git+https://github.com/RobTand/gridbook
```

Then re-run the [install check](INSTALL.md#verify-the-install); `cb_gemv.cu
found` must print `True`. Do **not** work around this by copying `csrc/` into
`site-packages` — if it recurs on a current version, it is a bug worth an issue.

---

## Incompatible JIT extension: the module loaded but has the wrong API

**Symptom** — on stderr:

```
[prismaquant-cb] ERROR: incompatible CUDA decode-GEMV extension — the module
loaded for cb_gemv.cu from '<module path>' does not satisfy the current call
contract: missing ['cb_gemv_fp4_v2']; every required symbol is [...]. requested
build directory '/opt/gridbook/ext-cache' has mode ..., owner uid:gid ..., and
is ... by this process. Clear this extension's build directory ...
```

The fused loaders report the same module path, missing contract, and
build-directory diagnostics. An optional optimized extension may
return `None` only where the caller has a separately qualified **native** CUDA +
CUTLASS route. A required serving operation raises; none selects Triton.

**Cause** — this symptom proves only that a module loaded successfully but does
not export the API the current Python code will call. A stale or corrupt build
artifact is possible, especially when one cache is carried across upgrades or
image versions; an unexpected module supplied by the environment is another
possibility. The error reports both the loaded module's `__file__` and the
requested build directory so those cases can be distinguished.

The build cache is *meant* to persist across restarts
([persisting the cache](INSTALL.md#persisting-the-jit-build-cache); the
reference image pins `PRISMAQUANT_CB_EXT_DIR=/opt/gridbook/ext-cache`), but use
one cache directory per gridbook version when mounting it from the host.

An **unwritable cache does not by itself explain a successfully loaded old
module**. PyTorch normally needs to acquire a lock and write build metadata or
outputs before loading; insufficient permissions ordinarily raise during that
lock/build phase and land in the separate *"extension could not be built"*
warning. Check ownership when that warning contains `PermissionError`, rather
than inferring root ownership from a missing-symbol error.

**Fix** — delete the directory (or point `PRISMAQUANT_CB_EXT_DIR` at a fresh
one) and restart; you pay one ~30 s rebuild. Compare the module path in the
error with the requested cache path. If a build error reports permissions,
inspect the directory from *inside* the serving container:

```bash
docker exec <container> sh -lc 'ls -ld "$PRISMAQUANT_CB_EXT_DIR"; id'
```

**Why it fails closed** — until the loaders checked, the
module was returned unexamined. A missing symbol then surfaced either as
`AttributeError: module 'prismaquant_cb_ext' has no attribute '<name>'` raised
mid-forward from inside a custom op, or — for optional bindings — as no error at
all and an unproven dispatch. Current required symbols are validated before use.

---

## Fused prefill extension unavailable

**Symptom**

```
[prismaquant-cb] WARNING: fused prefill extension unavailable (<...>); mid-M
stays on the transient expand path (the shipping default — this is expected on
non-sm_120 GPUs and without nvcc).
```

**This can be fine when the native transient route is qualified.** It affects
eligible FP8-CB calls in the M=9–128 dispatch band; the measured M=32/64/128
points were ~1.04–1.45× faster on the fused
CUTLASS `sm_120`-family kernel
([KERNELS.md](KERNELS.md)). Dispatch then uses native CUDA transient expansion
plus CUTLASS GEMM, which is the general shipping path—not a Triton fallback.

It is *expected* on any non-Blackwell GPU (the kernel is `sm_120`-family only)
and anywhere `nvcc` is missing. It can also mean vLLM's bundled CUTLASS headers
were not found — the fused build takes them from your vLLM install, and
`PRISMAQUANT_CUTLASS_INCLUDE` points it elsewhere
(see [INSTALL.md](INSTALL.md#cutlass-comes-from-your-vllm-install)).

To skip the build attempt entirely: `PRISMAQUANT_CB_FUSED_MIDM=0`.

---

## Invalid quantization method: gridbook

**Symptom** — vLLM fails at model load with something like:

```
ValueError: Invalid quantization method: gridbook
```

possibly preceded, much earlier in the log, by:

```
ERROR ... Failed to load plugin gridbook
Traceback (most recent call last): ...
```

**Cause** — the plugin did not register. vLLM catches and *logs* plugin-load
exceptions and then keeps booting, so the real error is upstream of the one you
noticed. Three common reasons:

1. **gridbook is installed in a different environment than vLLM.** Check:
   `python -c "import vllm, gridbook; print(vllm.__file__, gridbook.__file__)"`.
2. **The entry point is missing** (an incomplete or manually-copied install):

   ```bash
   python -c "
   from importlib.metadata import entry_points
   print([e for e in entry_points(group='vllm.general_plugins')])"
   ```

   You should see a `gridbook` entry.
3. **vLLM API drift.** gridbook imports vLLM internals that carry no stability
   promise. To see the true error, force registration yourself:

   ```bash
   python -c "import gridbook; gridbook.register()"
   ```

   An `ImportError` / `AttributeError` naming a `vllm...` symbol means your vLLM
   version has moved something. Pin a version from
   [the tested stack](INSTALL.md#tested-software-stack), and please file an issue
   with the traceback and `vllm.__version__` — that is the report that lets the
   plugin be widened.

Legacy artifacts carrying `"quant_method": "prismaquant"` are also accepted; both
keys are registered.

---

## Missing codebook sidecar or quant config

**Symptom** — a `FileNotFoundError` (local model) or a Hub `EntryNotFoundError`
naming one of:

- `cb_codebooks.pqcb` — the shared codebook tables
- `quant_config.json` — the full per-Linear format assignment

**Cause** — a gridbook artifact is *not* just weights. `config.json` carries only
a small pointer stub; the real configuration lives in `quant_config.json`, and
the codebook values live in the `.pqcb` sidecar. A partial download, a
`--include "*.safetensors"` fetch, or a hand-assembled directory will be missing
them.

**Fix** — download the whole repository:

```bash
huggingface-cli download rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm \
  --local-dir ./qwen27b-gridbook
```

Serving by Hub id (`vllm serve rdtand/...`) also works — the sidecars are fetched
from the Hub automatically in that case.

---

## Codebook provenance mismatch

**Symptom** — model load aborts with:

```
ValueError: [prismaquant-cb] ERROR: codebook provenance mismatch for
<path>/cb_codebooks.pqcb (... expected <digest>, computed <digest> ...)
```

**What it means** — the per-table hashes in
`quant_config.json["provenance"]["codebook_sha256"]` do not match the codebook
sidecar ([SPEC.md §4.1](SPEC.md)). The usual causes are a hand-assembled model
directory that picked up a `.pqcb` from a *different* artifact, a stale sidecar
left behind when the model was re-encoded, and length-preserving data corruption
(a bad disk or an interrupted in-place sync).

**Why this is an error and not a warning** — wrong codebook *values* cannot be
caught any later. A `k`-bit codeword indexes a `2^k`-row table, so every index
is in range by construction and a wrong table decodes to a correctly-shaped
tensor of structured garbage: the server starts, the weights "load", and
generation is quietly wrong. See
[the model loads but generates garbage](#the-model-loads-but-generates-garbage)
for the other member of that family. (Structural damage was never the problem —
a byte-truncated `.pqcb` already fails to deserialize with
`SafetensorError: MetadataIncompleteBuffer`.)

**Fix** — re-download the complete artifact, including both `quant_config.json`
and `cb_codebooks.pqcb` (see
[missing sidecar](#missing-codebook-sidecar-or-quant-config)). To verify a local
artifact without starting a server (no GPU or vLLM needed):

```bash
python scripts/verify_codebooks.py /path/to/model-directory
```

There is no serve-anyway switch: a declared mismatch would otherwise silently
decode every affected weight with wrong values. Legacy artifacts whose config
does not declare `codebook_sha256` remain supported and load without the check.

---

## Out of memory on a 32 GB card

**Symptom** — the 27B artifact (23.0 GB on disk) OOMs during load on a 32 GB
RTX 5090, reporting ~30.5 GiB of weights, leaving no room for KV cache.

**Cause** — tracked in
[issue #1](https://github.com/RobTand/gridbook/issues/1), reported by an external
user on exactly that card. Two contributing defects were identified: fused
attention modules on hybrid (linear-attention) models silently falling back to
BF16 instead of the codebook format, and both the original and the derived
padded copy of each quantized weight staying resident after load.

**Status** — **both defects are fixed on `master`** (commit `9b6cb2f`): the
fused-module scheme lookup now probes the mapper-transformed namespace through
one shared resolver, and only one padded copy of each dense CB weight stays
resident. On the maintainer's box that took the 27B from **35.86 → 20.82 GiB**
of model weights. The reporter, running their own patches, measured **20.54
GiB** of weights and **6.48 GiB** of KV at 32k context on the 5090 itself.

**The issue is still open** and no maintainer-side 32 GB card exists, so a
32 GB fit is *strongly expected* but not independently confirmed on that
hardware. Install from `master` (not from a release older than that commit), and
check the issue.

**Mitigations if it still does not fit**, in decreasing order of effect:

```bash
--max-model-len 8192            # shrink the KV budget
--kv-cache-dtype fp8            # roughly halve KV bytes
--language-model-only           # VL models: skip the vision tower
--gpu-memory-utilization 0.95   # discrete GPU only, not on unified memory
```

If you hit an OOM whose numbers do not match the above, that is a *new* report —
please include the model, the card, the vLLM version and the loader traceback.

---

## Out of memory / box hang on a DGX Spark

**Symptom** — the whole machine becomes unresponsive, or the server is killed
with exit code 137, on a GB10 / DGX Spark.

**Cause** — unified memory. The GPU and the host share one physical pool, so
`--gpu-memory-utilization` is not carving up a private VRAM budget; a high value
plus activation spikes starves the host. "Move it to CPU" is a no-op there.

**Fix** — start at `--gpu-memory-utilization 0.85`, bring the server up, and
check that several GiB of `MemAvailable` remain (`grep MemAvailable
/proc/meminfo`) before raising it. The measured 295B configuration on that box
serves at util 0.90 with ~6.5 GiB of pool left unclaimed as a spike buffer; 0.94
and 0.95 both died under long-prefill spikes. Speculative decoding and
`torch.compile` both raise the floor — start those configurations lower, not
higher.

---

## Serving is far slower than the published numbers

Work down this list:

1. **Are all required native extensions loaded?** Run the
   [install check](INSTALL.md#verify-the-install) and inspect every
   `[prismaquant-cb]` diagnostic. Current Gridbook fails when a required kernel
   is missing; if a server merely becomes slow, first confirm you are not
   running an older fail-soft Gridbook release.
2. **Is `--enforce-eager` set?** Without it, decode measured *worse* — see
   [the next section](#do-i-really-need---enforce-eager).
3. **Are any `PRISMAQUANT_*` variables set?** Old model cards and scripts may
   carry retired Triton or MoE prefill-mode selectors. Remove retired selectors
   and compare with the documented native defaults. Check with
   `env | grep PRISMAQUANT`.
4. **MoE prefill on a large batch?** Current dispatch is fixed: grouped CUDA
   GEMV at M≤16, then quality-green fused CUTLASS for eligible FP8-CB calls or
   exact BF16 expansion + the owned CUTLASS grouped bridge. The predecessor CUDA
   chunk-expander path took Laguna-class prefill from 293 → 1,821 tok/s at 8k and
   207 → 1,822 at 63k (commit `8829c16`), but those are historical—not a
   throughput promise for the new bridge. Verify the artifact and plugin commit.
   `--max-num-batched-tokens 16384` matters here: chunked prefill re-expands per
   microbatch.
5. **Large-M dense prefill is a known gap**, not a misconfiguration: ~1.44× the
   native NVFP4/FP8 GEMM's TTFT on the 27B. Decode, not prefill, is where the
   format is at parity.

---

## Do I really need `--enforce-eager`?

It remains the conservative configuration used for the published 27B result,
but it is no longer the only capture-safe choice.

The old inline dispatch let a prefill-sized trace bake the wrong arm into a
decode graph. That is a historical failure of the retired pre-hardening path.
Current Gridbook permanently wraps each complete Linear/MoE dispatch in one
opaque custom op. A decode-size capture therefore records the GEMV arm, while
prefill executes eagerly and chooses its own arm; there is no switch back to
the old host branch.

The validated candidate is:

```text
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

On the dated close-rate 0.6B canary, the opaque arm improved Gridbook's 32+256
whole-request latency by 20.1%, to within 5.9% of native; changed batch-1 and
batch-4 prompts matched eager text, tokens, and token logprobs exactly. That
run predates the current permanent boundary and direct registered-operator FP8
binding, so it remains
historical capture evidence rather than a benchmark of today's full serving
stack. Graph capture also adds startup time and memory, so choose capture sizes
that match the concurrency you actually serve. The 295B FP4-v2 path separately
measured +24% decode. The published 27B FP8-CB artifact still needs the full
streaming gate before its quickstart changes.

The rules and evidence limits are in
[KERNELS.md](KERNELS.md#cuda-graph-safety-rules) and
[BENCHMARKS.md](BENCHMARKS.md#2026-07-31-cuda-graph-canary--close-rate-not-formal-parity).

---

## Non-Blackwell GPU: what breaks

The grouped-BF16 CUTLASS kernel used by FP4-CB can compile for compute
capability 8.0, but that is not the full FP4 serving floor. Every FP4-v2 quality
path also needs the exact v2 expander, whose device prepare currently accepts
only cc 12.0/12.1. FP8-CB native prefill requires cc 8.9 and is checked during
model construction. See the [hardware matrix](INSTALL.md#hardware-matrix).

- **`sm_89` / `sm_90` (RTX 4090, L40S, H100)** — FP8-only decode and dense
  prefill are expected to work; when the optional Blackwell fused kernel is
  ineligible, FP8 uses native CUDA expansion + CUTLASS. FP4-CB is rejected at
  weight load because v2 expander prepare rejects these devices. The FP8 claim
  is inferred from code and untested by the author.
- **`sm_80` (A100)** — an FP8-CB artifact is rejected early with a clear
  `sm_89+` error. The former behavior loaded successfully and then failed at
  the FP8 quantizer/scaled-matmul boundary on the first prompt longer than 16
  tokens. Current Gridbook attests the registered native `torch.ops._C`
  operators directly and never enters vLLM's fallback-capable Python wrapper.
  FP4-CB is also rejected: the grouped-BF16 kernel can target SM80, but the
  required v2 expander cannot prepare on this card.
- Per-artifact groups matter too. The 27B's vision tower is stock NVFP4 W4A16 and
  needs a vLLM NVFP4 backend independently of gridbook.

If you get one of these paths working on a card not in the table, an issue with
your versions and numbers is genuinely useful — that table is how it gets wider.

---

## Tensor parallel (`tp > 1`)

Scoped since the 2026-08-23 shard-aware loading wave. **Dense CB Linears**
load correctly above one rank with no change to any exported byte: packed
rows are independent streams (column split), and row splits land on
superblock boundaries only — a degree whose K-shard or N-shard would break a
group boundary is refused at weight construction with a structured
`ShardGroupAlignmentError` naming the target, axis, group size and degree.

**Routed CB MoE expert stacks are the exception, and they need a different
flag.** A CB expert stack's last dimension is superblock bytes, not input
columns, so a tensor-parallel intermediate split would cut a packed
superblock. Serve them with `-tp N --enable-expert-parallel`, which shards the
expert axis instead — see [Expert parallel](#expert-parallel-tp--1---enable-expert-parallel).
`-tp N` alone refuses at construction and says so.

Everything else still refuses at construction, naming itself: delegated
compressed-tensors groups, source-passthrough units, quantized embedding
units and mixed-format fused projections. An artifact mixing those surfaces
with dense CB therefore fails on its first unsupported layer. Ignored
(BF16) Linears keep vLLM's own sharding.

Two honest caveats: no two-node serve has been measured yet (the reference
hardware is single-GPU DGX Spark; cross-node TP over 10 GbE without RDMA is
expected to lose at batch-1 decode), so dense TP>1 is a correctness feature
for models that do not fit one box, not a measured speedup. And per-token
dynamic FP8 activation scales are computed over each rank's local K window
on row-parallel layers at TP>1, exactly as stock W8A8 schemes behave — so
served logits are not bit-identical to TP=1; quality comparison belongs to
the standing same-session KL gate.

---

## Expert parallel (`-tp N --enable-expert-parallel`)

The multi-rank mode for **routed CB MoE expert stacks**. Tensor parallelism
splits a unit's rows and columns, which a CB expert stack cannot survive — its
last dimension is `(in/256)·type_size` superblock bytes, and there is no
partial-superblock decode. Expert parallelism splits the expert axis, which is
the axis a stack is already indexed on, so each rank holds whole experts and
whole superblocks. A rank's expert bytes are byte-identical to the
corresponding slice of a single-rank stack.

**Symptom** — `Gridbook serves CB MoE expert stacks ... above one rank only
under expert parallelism; the live vLLM worker reports TP=2 and expert
parallelism is off for this MoE layer.`

**Fix** — add `--enable-expert-parallel`. `-tp N` alone tensor-parallelizes the
expert stacks, which is the case being refused. Dense Linears in the same model
stay tensor-parallel at the full world size; that is vLLM's own split, not a
setting you choose separately.

On a successful load each admitted layer announces itself:

```
[prismaquant-cb] moe_admission model.layers.3.mlp.experts -> expert_parallel(ep_size=2); this rank holds 128 of 256 experts
```

Other refusals on this path, and what each means:

| Refusal names | Why |
|---|---|
| `moe tp_size=N` | Expert parallelism is on but the MoE layer is still tensor-parallel; EP must own the whole MoE axis. |
| `all2all expert-parallel topology (dp_size=… pcp_size=… sp_size=…)` | Data-, pipeline-context- or sequence-parallel EP. vLLM switches those to all2all dispatch/combine kernels, which expect a MoE method that exchanges tokens between ranks; Gridbook computes only its own experts. Set those sizes to 1. |
| `expert-load-balancing (EPLB) is enabled` | Gridbook holds whole expert stacks resident and cannot follow a live re-placement. |
| `skip_final_all_reduce is set` | Gridbook returns this rank's partial output and relies on vLLM's stock final all-reduce to sum the ranks. With the reduce skipped, nothing sums them. |
| `expert_map is not a bijection` / `is not monotone` | The placement this rank was given does not name each of its local slots exactly once in ascending order. Both stock placement strategies (`linear`, `round_robin`) satisfy this; a custom one may not. |
| `per-expert format groups` | A mixed-format expert stack. Its format partition is declared over *global* expert ids and each per-format sub-stack is sized from that partition, so a rank owning an arbitrary subset can neither size nor fill them. Export the layer as a single CB format. |

**The honest caveat**: no two-node serve has been measured. Everything above is
established on one box by simulating the split in-process — byte-identical
per-rank stacks, bitwise-inert remote pairs through the real kernels, and
per-rank partials that sum to the whole-layer answer
(`tests/test_moe_ep_exactness.py`). Treat expert-parallel serving as a
correctness feature for models that do not fit one box, not a measured
speedup, until that gate runs.

---

## The model loads but generates garbage

**Symptom** — the server starts, weights load without error, and generation is
incoherent (not merely low-quality).

**Most likely cause on a Mixture-of-Experts model**: the architecture's expert
weights were never routed through the codebook loader. Some MoE architectures
load experts at the top-level model rather than through vLLM's per-layer fused
MoE loader; those need an explicit module entry in the packaged
`gridbook/runtime_contract.json`. That file is the sole authoritative list;
the plugin derives its loader registry from it. An unregistered architecture
is rejected by the post-load fill guard before it can serve wrong weights.

**What to do** — open an issue naming the model architecture class (the
`architectures` field in the model's `config.json`). Adding one is a contract
change with a loader/fill regression test.

Other checks: confirm you are not mixing a checkpoint with a plugin version that
predates its architecture support, and confirm the artifact is complete (see
[missing sidecar](#missing-codebook-sidecar-or-quant-config)).

---

## Other exceptions you may hit

| Message | Meaning |
|---|---|
| `... effective load_weights method does not expose Gridbook mixed-fused loader ABI 1 ...` | The fused roles are valid, but the selected model class has not installed the transaction router needed to address their private carrier parameters. Add/audit that exact class's top-level wrapper; do not re-export the roles to one format. |
| `fused module ... has only partial explicit role ownership; missing [...]` | At least one physical sibling of a known vLLM fusion has a CB/source declaration and another does not. Gridbook refuses to let the first role's scheme claim the whole module; declare every role or a deliberately supported whole-module representation. |
| `<prefix>: mixed fused role <n> (...) is missing checkpoint planes [...]` | vLLM merged several independently encoded Linears, but the checkpoint stream did not supply every plane registered for one role. Gridbook commits a composite fused module only as a complete transaction; check artifact completeness and the architecture's checkpoint-name mapper. |
| `incomplete mixed fused checkpoint transactions; ...` | At least one composite fused module reached end-of-load with missing role tensors. The listed nested parameter names identify the exact absent packed weight/scale plane; this is an artifact or name-mapping defect, never a request to force the siblings to one format or scalar. |
| `MoE stack ... is declared both as source-native ... and as ... CB ...` | One exact routed-expert stack is claimed by source MXFP4 and Gridbook CB metadata, which is illegal because both methods would own the same resident weights. Different modules in the same decoder layer (for example source-MXFP4 routed experts plus CB attention/shared experts) remain valid. |
| `<prefix>: fp4 MoE experts require two-tier v2 scale coding` | The artifact uses the legacy fp4 scale plane for MoE experts, which has no expert transient path. Legacy-format artifact; report the model id. |
| `<prefix>: in_features <K> not a multiple of ...` | The Linear's input width is not a multiple of the format's superblock — an artifact/export defect. |
| `prismaquant shared-CB: no unique scheme for '<name>'` | The top-level MoE expert loader could not resolve a shared-expert Linear to exactly one format entry. Report with the model id and the full name in the message. |
| `prismaquant CB tensor '<name>' resolved to plain bf16 parameter '<target>'` | The architecture's shared-expert prefix did not resolve to a native CB Linear. Current Gridbook aliases HunYuan-V3 shared and MTP-nested forms; reaching this error means the architecture/prefix contract drifted. Model load stops because decoding CB into an upstream plain BF16 Linear could select a non-Gridbook kernel. Report both names and the model/vLLM versions. |

---

## The first model load stalls for kernel compilation

Expected once per build cache outside a prewarmed image: required CUDA/CUTLASS
extensions JIT-compile while the model is loading. Production Gridbook resolves
every reachable module before the model becomes serve-ready; compilation is not
deferred to the first forward.
Measured cold in the reference container (`vllm-node:latest`, compile-only, no
GPU): **28.7 s** and **32.3 s** at `TORCH_CUDA_ARCH_LIST` 12.1 and 12.0, matching
the 29.4 s / 29.7 s recorded in `gridbook/cuda_ext.py`. Call it ~30 s; it varies
with the arch list and the host. The plugin deliberately warms it at *weight
load* so no served request pays — but in a container with an
ephemeral home directory you pay it on every start.

Fix: [persist the build cache](INSTALL.md#persisting-the-jit-build-cache) with
`PRISMAQUANT_CB_EXT_DIR` or a volume mount.

---

## Benchmark numbers move between runs

If you are A/B-ing quantizations, know this before you draw a conclusion:
loading *any* additional CUDA extension into the serving process shifts the
allocator's addresses, which changes alignment-sensitive kernel dispatch
elsewhere in the model and perturbs floating-point reduction order. On the 27B
artifact that produced a **±17%** swing in a measured KL evaluation with
byte-identical served weights.

Match extension residency across arms, or the comparison is confounded.

Fix: run **both** arms with `PRISMAQUANT_PRELOAD_FUSED=1`. At plugin registration
that now builds and loads *every* native extension family — the decode GEMV,
GEMV-v2, grouped BF16, both fused FP8/NVFP4 modules, fused FP4-v2 and persistent-B
MoE — rather than only the two fused ones it warmed before, so an arm cannot drift
by loading a module the other never touched. Individual loaders stay fail-soft: one
that will not build on this box leaves the others warmed. Details:
[KERNELS.md](KERNELS.md#a-measurement-side-effect-worth-knowing) and
[BENCHMARKS.md](BENCHMARKS.md#caveats--read-these).
