# Troubleshooting

Every entry below is keyed to something you will actually see — a log line, an
exception, or a number that is wrong. If your symptom is not here, please open an
issue; [CONTRIBUTING.md](../CONTRIBUTING.md) lists what to include.

**First, find out which path you are on.** Most "gridbook is slow" reports are
really "the CUDA extension did not load". The check in
[INSTALL.md](INSTALL.md#verify-the-install) answers that in ten seconds without
serving a model.

**A note on grepping.** The plugin's log lines and environment variables use the
project's older `prismaquant` prefix, not `gridbook` — searching your vLLM log
for "gridbook" will find nothing. Search for **`[prismaquant-cb]`**.

- [The CUDA extension did not load (Triton fallback)](#the-cuda-extension-did-not-load-triton-fallback)
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
- [The model loads but generates garbage](#the-model-loads-but-generates-garbage)
- [Other exceptions you may hit](#other-exceptions-you-may-hit)
- [The first request stalls for ~30 seconds](#the-first-request-stalls-for-30-seconds)
- [Benchmark numbers move between runs](#benchmark-numbers-move-between-runs)

---

## The CUDA extension did not load (Triton fallback)

**Symptom** — on stderr, during model load:

```
[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV extension could not be
built (<ErrorType>: <message>); falling back to the Triton decode path (slow
prototype). To get the CUDA path: install a CUDA toolchain matching your torch
build ...
```

(Older plugin builds print `WARNING: CUDA decode-GEMV extension unavailable`.
Same meaning.)

**What it means** — the plugin is serving **correct** output on the pre-CUDA
reference kernels, which are several times slower. None of the published
performance numbers are reachable on this path. It is not a correctness problem.

For scale, on dense decode — where the fallback is unchanged code — the 27B
measured **4.20 tok/s** on the Triton decode-GEMM against **10.28** on the CUDA
GEMV. MoE loses more, and by an artifact-dependent amount. Those are the
before/after numbers from when the CUDA kernels landed, **not** a fresh benchmark
of today's fallback; [BENCHMARKS.md](BENCHMARKS.md#what-the-fallback-costs)
states which parts were measured and which were not.

**Causes and fixes**, in order of likelihood:

| The `<ErrorType>` you see | Cause | Fix |
|---|---|---|
| `RuntimeError` mentioning `nvcc`, or `FileNotFoundError: nvcc` | No CUDA compiler in the *serving* process. Many vLLM images ship the runtime but not the toolkit. | Install a CUDA toolkit matching your torch build (distro `cuda-toolkit`, or the `nvidia-cuda-nvcc-*` wheel) and make sure `nvcc` is on `PATH` or `CUDA_HOME` points at it. Verify with `nvcc --version` **inside the container**. |
| `PermissionError` / `OSError` on the build directory | The build cache path is not writable by the serving user. | Set `PRISMAQUANT_CB_EXT_DIR` to a writable directory. |
| A compiler error from `nvcc` | Toolchain/torch mismatch, or an `nvcc` too old for your GPU's architecture. | `nvcc` 13.0 is the tested toolchain. Match the CUDA major version your torch was built against. |
| Anything, but only inside a container that used to work | The ephemeral build cache is being rebuilt and failing. | See [persisting the cache](INSTALL.md#persisting-the-jit-build-cache). |

`PRISMAQUANT_CB_DECODE=triton` also forces this path deliberately. If you (or a
model card) set it, unset it.

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
working-but-slow server (reported in
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

The persistent-TC and fused loaders report the same module path, missing
contract, and build-directory diagnostics. They remain fail-soft and return
`None`, so their callers take the existing correct fallback.

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

**Why it is reported rather than tolerated** — until the loaders checked, the
module was returned unexamined. A missing symbol then surfaced either as
`AttributeError: module 'prismaquant_cb_ext' has no attribute '<name>'` raised
mid-forward from inside a custom op, or — for the optional bindings, which read
an absent symbol as "an older build" — as no error at all and a quietly slower
server.

---

## Fused prefill extension unavailable

**Symptom**

```
[prismaquant-cb] WARNING: fused prefill extension unavailable (<...>); mid-M
stays on the transient expand path (the shipping default — this is expected on
non-sm_120 GPUs and without nvcc).
```

**This is usually fine.** It affects only prefills of 17–128 tokens, where a
CUTLASS `sm_120`-family kernel would have been ~1.04–1.45× faster
([KERNELS.md](KERNELS.md)). Everything falls back to the transient-expand path,
which is the general shipping default.

It is *expected* on any non-Blackwell GPU (the kernel is `sm_120`-family only)
and anywhere `nvcc` is missing. It can also mean vLLM's bundled CUTLASS headers
were not found — the fused build takes them from your vLLM install
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

1. **Are you on the Triton fallback?** By far the most common cause — see
   [above](#the-cuda-extension-did-not-load-triton-fallback). Run the
   [install check](INSTALL.md#verify-the-install).
2. **Is `--enforce-eager` set?** Without it, decode measured *worse* — see
   [the next section](#do-i-really-need---enforce-eager).
3. **Are any `PRISMAQUANT_*` variables set?** Several model cards and scripts set
   escape hatches such as `PRISMAQUANT_CB_DECODE=triton` (forces the slow decode
   path) or `PRISMAQUANT_CB_PREFILL=loop` (forces the per-expert MoE prefill).
   Check with `env | grep PRISMAQUANT`.
4. **MoE prefill on a large batch?** The default for fp8-CB MoE prefill is
   `auto`, a measured per-layer path selection. On a Laguna-class 117B MoE the
   CUDA chunk-expander took prefill from 293 → 1,821 tok/s at 8k and 207 → 1,822
   at 63k (commit `8829c16`); if you are far below that, you are probably on an
   older plugin, or on the Triton path.
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
decode graph. The default `PRISMAQUANT_CB_DISPATCH=op` now wraps each complete
Linear/MoE dispatch in one opaque custom op. A decode-size capture therefore
records the GEMV arm, while prefill executes eagerly and chooses its own arm.

The validated candidate is:

```text
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

Leave `PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE` unset and do not select
`PRISMAQUANT_CB_DISPATCH=inline`. On the close-rate 0.6B canary this improved
Gridbook's 32+256 whole-request latency by 20.1%, to within 5.9% of native;
changed batch-1 and batch-4 prompts matched eager text, tokens, and token
logprobs exactly. It also adds graph-capture startup time and memory, so choose
capture sizes that match the concurrency you actually serve. The 295B FP4-v2
path separately measured +24% decode. The published 27B FP8-CB artifact still
needs the full streaming gate before its quickstart changes.

The rules and evidence limits are in
[KERNELS.md](KERNELS.md#cuda-graph-safety-rules) and
[BENCHMARKS.md](BENCHMARKS.md#2026-07-31-cuda-graph-canary--close-rate-not-formal-parity).

---

## Non-Blackwell GPU: what breaks

The base FP4-CB path declares compute capability **8.0**, while FP8-CB's native
prefill path requires **8.9** and is now checked during model construction. See
the [hardware matrix](INSTALL.md#hardware-matrix) for the per-path breakdown.

- **`sm_89` / `sm_90` (RTX 4090, L40S, H100)** — decode and dense prefill are
  *expected* to work (the decode kernel has no architecture guards); the mid-M
  fused kernel fails soft with the warning above. **Inferred from code, untested
  by the author.**
- **`sm_80` (A100)** — an FP8-CB artifact is rejected early with a clear
  `sm_89+` error. The former behavior loaded successfully and then failed at
  `cutlass_scaled_mm` / `scaled_fp8_quant` on the first prompt longer than 16
  tokens. FP4-CB retains its BF16 fallback, but is untested on this card.
- Per-artifact groups matter too. The 27B's vision tower is stock NVFP4 W4A16 and
  needs a vLLM NVFP4 backend independently of gridbook.

If you get one of these paths working on a card not in the table, an issue with
your versions and numbers is genuinely useful — that table is how it gets wider.

---

## Tensor parallel (`tp > 1`)

Unsupported and rejected during model construction. The plugin contains no
tensor-parallel handling for the packed index stream, per-role codebook offsets
or scale planes. Serve with `tp=1`. Multi-GPU support is not currently on the
roadmap; if you need it, say so on an issue.

---

## The model loads but generates garbage

**Symptom** — the server starts, weights load without error, and generation is
incoherent (not merely low-quality).

**Most likely cause on a Mixture-of-Experts model**: the architecture's expert
weights were never routed through the codebook loader. Some MoE architectures
load experts at the top-level model rather than through vLLM's per-layer fused
MoE loader; those need an explicit per-architecture registration in the plugin
(`gridbook/plugin.py`). Registered today: HunYuan-V3 (and its MTP drafter),
Laguna, and Qwen3.5-MoE (and its MTP). An unregistered architecture fails
*silently* into wrong weights rather than raising.

**What to do** — open an issue naming the model architecture class (the
`architectures` field in the model's `config.json`). Adding one is a guarded
one-liner.

Other checks: confirm you are not mixing a checkpoint with a plugin version that
predates its architecture support, and confirm the artifact is complete (see
[missing sidecar](#missing-codebook-sidecar-or-quant-config)).

---

## Other exceptions you may hit

| Message | Meaning |
|---|---|
| `fused module <name> maps to mixed CB decode formats — export union-find should prevent this` | The artifact assigns different formats to Linears that vLLM fuses into one module (q/k/v, gate/up). That is an artifact defect, not a serving one — please report it with the model id. |
| `<prefix>: fp4 MoE experts require two-tier v2 scale coding` | The artifact uses the legacy fp4 scale plane for MoE experts, which has no expert transient path. Legacy-format artifact; report the model id. |
| `<prefix>: in_features <K> not a multiple of ...` | The Linear's input width is not a multiple of the format's superblock — an artifact/export defect. |
| `prismaquant shared-CB: no unique scheme for '<name>'` | The top-level MoE expert loader could not resolve a shared-expert Linear to exactly one format entry. Report with the model id and the full name in the message. |
| `persistent-TC ext not enabled (PRISMAQUANT_ENABLE_PTC=1)` | You enabled a code path that requires the quarantined persistent-N kernel. **Do not set `PRISMAQUANT_ENABLE_PTC=1`** — that kernel is off by default because it measured *slower* than the shipping path and is under a stability quarantine. |

---

## The first request stalls for ~30 seconds

Expected once per build cache: the CUDA extension JIT-compiles on first use.
Measured cold in the reference container (`vllm-node:latest`, compile-only, no
GPU): **28.7 s** and **32.3 s** at `TORCH_CUDA_ARCH_LIST` 12.1 and 12.0, matching
the 29.4 s / 29.7 s recorded in `gridbook/cuda_ext.py`. Call it ~30 s; it varies
with the arch list and the host. The plugin deliberately warms it at *weight
load* so the first request is not the one that pays — but in a container with an
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

Match extension residency across arms, or the comparison is confounded. Details:
[KERNELS.md](KERNELS.md#a-measurement-side-effect-worth-knowing) and
[BENCHMARKS.md](BENCHMARKS.md#caveats--read-these).
