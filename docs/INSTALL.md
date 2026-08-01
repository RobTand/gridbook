# Installing gridbook

gridbook is a **plugin for vLLM**, not a runtime. Install it *into an environment
that already has vLLM and a compatible PyTorch*. It deliberately declares no
version pins for torch or vLLM, and vLLM is not a hard dependency. In a fresh
environment pip may resolve generic torch/triton builds, so use the established
serving environment and `--no-deps` when its stack is already managed (see
[Why nothing is pinned](#why-nothing-is-pinned)).

- [Requirements](#requirements)
- [Tested software stack](#tested-software-stack)
- [Hardware matrix](#hardware-matrix)
- [Install routes](#install-routes)
- [What happens on the first model load](#what-happens-on-the-first-model-load)
- [Persisting the JIT build cache](#persisting-the-jit-build-cache)
- [Verify the install](#verify-the-install)
- [Per-artifact requirements](#per-artifact-requirements)
- [Known limits](#known-limits)

---

## Requirements

| | Requirement | Notes |
|---|---|---|
| **OS** | Linux | Only Linux is tested; the package declares `Operating System :: POSIX :: Linux`. |
| **Python** | ≥ 3.10 | Installed-wheel CPU CI covers 3.10–3.13; GPU serving is measured on 3.12.3. |
| **GPU** | NVIDIA Blackwell `sm_120` / `sm_121` for the native path | Other NVIDIA cards: see the [hardware matrix](#hardware-matrix). Non-NVIDIA is unsupported. |
| **CUDA toolchain** | `nvcc` on `PATH` (or `CUDA_HOME` set) **in the serving process** | The kernels are JIT-compiled at runtime, not at install time. Missing `nvcc` is not an install error — it is a silent downgrade to the Triton path. |
| **PyTorch** | the build your vLLM uses | Measured: `2.11.0+cu130`. Installing gridbook into a fresh environment can pull a *generic* PyPI torch that does not match your CUDA — install into the vLLM environment instead. |
| **vLLM** | already installed | Measured against `0.23.1rc1.dev764+g54b16d8a9`. |
| **Parallelism** | `tp=1` | The plugin has no tensor-parallel handling. |

---

## Tested software stack

Read this as a compatibility statement, not a support promise. gridbook depends
on a handful of vLLM internals (fused-MoE classes, the quantization registry,
`vllm._custom_ops`) and on CUTLASS headers that ship *inside the vLLM install*,
so vLLM version drift is the most likely source of breakage.

| Component | Version | Evidence |
|---|---|---|
| vLLM | `0.23.1rc1.dev764+g54b16d8a9` | **MEASURED** — the image every published benchmark ran in, re-verified 2026-07-28. |
| torch | `2.11.0+cu130` | **MEASURED**, same image. |
| triton | `3.6.0` | **MEASURED**, same image. |
| transformers | `5.13.0` | **MEASURED**, same image. |
| CUDA / `nvcc` | `13.0.88` | **MEASURED**, same image. |
| CUTLASS | `4.3.4` (bundled by vLLM at `vllm/third_party/fmha_sm100/cutlass`) | **MEASURED**, same image. |
| Python | `3.12.3` | **MEASURED**, same image. |
| CPU architecture | **arm64** | **MEASURED** — the reference box is a DGX Spark. x86 is untested by the author. |
| vLLM `0.25.1`, `0.23.1rc1.dev1060` | x86 + RTX 5090 | **USER-REPORTED** working in [issue #1](https://github.com/RobTand/gridbook/issues/1), with that issue's local patches applied. |

Anything not listed is untested. A vLLM whose internals have moved typically
fails as an *unhelpful downstream error* rather than a clean one, because vLLM
logs-and-continues when a plugin fails to load — see
[TROUBLESHOOTING](TROUBLESHOOTING.md#invalid-quantization-method-gridbook).

### CUTLASS comes from your vLLM install

The mid-M fused prefill extension compiles against CUTLASS headers discovered
under `vllm/third_party/`, not a vendored copy. That is deliberate — it
guarantees ABI agreement with vLLM's own kernels — but it means the fused path's
availability is a property of *your* vLLM build. If those headers are absent or
in a different layout, the fused extension fails soft and mid-M prefill uses the
transient-expand path instead (correct, slightly slower).

---

## Hardware matrix

The decode kernel (`gridbook/csrc/cb_gemv.cu`) contains no architecture guards,
no inline PTX and no `-arch` flag — torch derives the target from the GPU present
— so it is *expected* to compile and run anywhere from `sm_80` up. Only the
mid-M fused prefill kernel is genuinely Blackwell-family (`sm_120`) bound, and
it already fails soft.

| GPU | Decode | Dense FP8-CB prefill | Mid-M fused prefill | Overall |
|---|---|---|---|---|
| **GB10 / DGX Spark `sm_121`** | ✅ CUDA | ✅ | ✅ | **MEASURED** — the reference target for every published number. |
| **RTX 5090 `sm_120`** | ✅ CUDA | ✅ | ✅ (expected) | **USER-REPORTED** working on the 27B artifact. Note the arch flag differs (`sm_120` vs the measured `sm_121`); no speed numbers exist for this card. |
| **H100 `sm_90`** | expected ✅ | expected ✅ | ❌ fails soft → expand path | **INFERRED, UNTESTED.** |
| **RTX 4090 / L40S `sm_89`** | expected ✅ | expected ✅ | ❌ fails soft → expand path | **INFERRED, UNTESTED.** |
| **A100 `sm_80`** | expected ✅ for FP4-CB | ❌ FP8-CB requires `sm_89+` | n/a | **FAILS EARLY for FP8-CB.** Gridbook rejects the first FP8-CB target during model construction instead of loading a serve that will fail above 16 tokens. FP4-CB retains its BF16 fallback; untested on this card. |
| **Supported NVIDIA GPU, no `nvcc`** | Triton fallback | Triton/stock fallback | n/a | Correct numerics, prototype speed. Not a serving target. |
| **Non-NVIDIA** | unsupported | unsupported | unsupported | **UNSUPPORTED / UNQUALIFIED.** No ROCm backend or dispatch hook ships in this release. |

Per-artifact caveat: an artifact may contain non-CB groups served by stock
`compressed-tensors` (the 27B's vision tower is NVFP4 W4A16). Those groups carry
**their own** hardware requirements, independent of gridbook's kernels. For a
delegated NVFP4 **MoE** group those requirements are sharper than the table
above — see [**`DELEGATED-NVFP4-MOE.md`**](DELEGATED-NVFP4-MOE.md) for the
version-scoped GB10 backend matrix. In particular, Marlin logs a generic
weight-only fallback but does not say that a W4A4 group's activation scales
were discarded.

---

## Install routes

### 1. From source — available today

```bash
pip install git+https://github.com/RobTand/gridbook
```

Run this in the environment that already has vLLM. `--no-deps` is a reasonable
addition if you want to be certain pip touches nothing else:

```bash
pip install --no-deps git+https://github.com/RobTand/gridbook
```

### 2. From PyPI — recommended

```bash
pip install gridbook
```

For an already-pinned serving environment, this avoids asking pip to resolve
torch, triton, or other runtime packages again:

```bash
pip install --no-deps gridbook
```

### 3. Container

The repo-root [`Dockerfile`](../Dockerfile) layers gridbook onto a pinned vLLM
image and compiles the kernels at image-build time. Full instructions —
build, serve, volumes, and what is pinned — are in
[**`CONTAINER.md`**](CONTAINER.md). If you build your own image instead, mount a
host directory over the JIT build cache so it stays warm across container
restarts: see [persisting the build cache](#persisting-the-jit-build-cache).

### 4. From a checkout (for development)

```bash
git clone https://github.com/RobTand/gridbook
pip install -e ./gridbook --no-deps
```

Editable and non-editable installs resolve the CUDA sources identically — they
ship inside the package at `gridbook/csrc/` and are located through
`importlib.resources`. (Historically they lived at the repo root and a
non-editable install silently lost them, producing a working-but-slow server.
That is fixed; the [verification snippet](#verify-the-install) proves it for your
install.)

### Why nothing is pinned

`torch` and `triton` are declared as bare requirements with no version bounds.
This is deliberate: vLLM environments run local-version wheels
(`2.11.0+cu130`) that no PyPI pin can satisfy, and the reference vLLM build is a
source build that is not on PyPI at all. vLLM itself is an **optional** extra
(`pip install gridbook[serve]`), never a hard dependency, so `import gridbook`
works on a machine with no GPU and no vLLM — which is what lets the format and
codec tests run in CI.

The consequence you must handle yourself: **do not create a fresh virtualenv and
`pip install gridbook` into it**, or pip may pull a generic PyPI torch that does
not match your CUDA. Install into the vLLM environment.

---

## What happens on the first model load

Nothing is compiled at `pip install` time. The extensions are built lazily by
`torch.utils.cpp_extension.load()` the first time each is needed:

| Extension | Sources | Built when | Cost |
|---|---|---|---|
| `prismaquant_cb_ext` (decode GEMV, MoE grouped GEMV, prefill expander) | `gridbook/csrc/cb_gemv.cu` | **at weight load**, deliberately warmed there so it does not poison the first request | **~30 s** once — see below |
| `prismaquant_cb_v2_ext` (experimental smem-resident-dictionary MoE decode GEMV) | `gridbook/csrc/cb_gemv_v2.cu` | at weight load only for an **fp4**-CB two-tier MoE layer with `PRISMAQUANT_CB_GEMV=auto` or `v2`; unset stays on the inherited kernel and does not build this extension | comparable to `prismaquant_cb_ext`; unsupported devices and failed builds degrade to the shipped kernel with one stderr warning |
| `pq_cb_fused` (mid-M CUTLASS fused prefill) | `gridbook/csrc/cb_fused_gemm.cu` + `csrc/cutlass_fork/*.hpp` | first prefill with 16 < tokens ≤ 128 | longer (a CUTLASS collective); not separately timed |
| `pq_cb_ptc` (persistent-N prefill) | `gridbook/csrc/cb_persistent_tc.cu` | only when `PRISMAQUANT_ENABLE_PTC=1` | **quarantined, off by default** — measured negative for performance and under an open stability quarantine. Do not enable it. |

**Where "~30 s" comes from.** Cold `get_ext()` in the reference container
(`vllm-node:latest`, compile-only, **no** `--gpus`, `PRISMAQUANT_CB_EXT_DIR`
empty): **28.7 s** at `TORCH_CUDA_ARCH_LIST=12.1` and **32.3 s** at `12.0`
(2026-07-28), against **29.4 s / 29.7 s** recorded for the same check in
`gridbook/cuda_ext.py`. It is a compile time, so it moves with your arch list,
your `nvcc` and your host — treat ~30 s as the order of magnitude, not a
specification.

Both of the first two **fail soft**: if the build fails, the plugin prints a
warning and continues on a slower path, rather than refusing to serve. That is
why an install problem shows up as *low throughput*, not as an error — and why
the verification step below matters.

---

## Persisting the JIT build cache

The build output is cached at:

```
$PRISMAQUANT_CB_EXT_DIR   # if set
~/.cache/prismaquant-cb-ext   # default
```

Inside a container that default is ephemeral, so you pay the ~30 s build on
**every container start**. Mount a host directory over it:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v /host/cache/gridbook-ext:/root/.cache/prismaquant-cb-ext \
  ...
```

or point the variable somewhere persistent:

```bash
-e PRISMAQUANT_CB_EXT_DIR=/mnt/persistent/gridbook-ext
```

The directory must be writable by the serving process. It is never `/tmp`.

---

## Verify the install

Run this in the environment where you will serve. It needs **no model and no
vLLM server** — it resolves the packaged sources, performs the same JIT build the
plugin does, and reports the environment switches that can route around the
result.

**What it can and cannot tell you.** It proves the two things that fail
*silently*: that the packaged `.cu` sources resolve from your install, and that
they compile here. It does **not** prove a kernel ever runs on your GPU — the
build needs `nvcc`, not a device, so it prints `LOADED` on a machine with no GPU
at all. The last block below covers the third silent failure (dispatch gated off
by environment), but the only end-to-end proof is serving a model and seeing no
`[prismaquant-cb] WARNING` line.

```bash
python - <<'PY'
import os
import gridbook
from gridbook import cuda_ext
import torch

print("gridbook        :", gridbook.__version__)
print("torch           :", torch.__version__)
print("cuda available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print("gpu             :", torch.cuda.get_device_name(0), f"(sm_{major}{minor})")

src = cuda_ext.csrc_dir()
print("packaged csrc   :", src)
print("cb_gemv.cu found:", os.path.isfile(os.path.join(src, "cb_gemv.cu")))

ext = cuda_ext.get_ext()       # JIT-compiles on first run (~30 s), then cached
if ext is None:
    print("BUILD           : CUDA extension UNAVAILABLE -> fallback path (slow).")
    print("                  See docs/TROUBLESHOOTING.md")
else:
    print("BUILD           : CUDA extension LOADED ->", getattr(ext, "__file__", "?"))
    have = [s for s in ("cb_gemv_fp8", "cb_gemv_fp4_v2", "cb_expand_fp8_into",
                        "cb_moe_gemv_fp8") if hasattr(ext, s)]
    print("                  symbols:", have)

# A built extension is still not a used extension: these route around it.
decode = os.environ.get("PRISMAQUANT_CB_DECODE", "cuda")
print("CB_DECODE       :", decode,
      "(default 'cuda')" if decode == "cuda"
      else "*** NOT 'cuda' -> CUDA decode is DISABLED, you are on Triton ***")
env = {k: v for k, v in os.environ.items() if k.startswith("PRISMAQUANT_")}
print("other switches  :", {k: v for k, v in env.items()
                            if k != "PRISMAQUANT_CB_DECODE"} or "none set")
print("RESULT          :", "CUDA decode path"
      if (ext is not None and decode == "cuda") else "FALLBACK path (slow)")
PY
```

A healthy install prints `BUILD : CUDA extension LOADED`, four symbols,
`CB_DECODE : cuda`, and `RESULT : CUDA decode path`.

The `symbols:` line above is informational. The loader itself asserts the
symbols its callers dereference unconditionally, so `BUILD : CUDA extension
LOADED` already means those are present; a build that lacks one reports
*"incompatible CUDA decode-GEMV extension"* and returns `UNAVAILABLE` instead of
loading. Symbols listed there but **not** required — `cb_expand_fp8_into` is
one — are optional bindings that older extensions may not have; their absence
costs a fast path, not correctness.

Four distinct failures are reported differently, on purpose:

- **`cb_gemv.cu found: False`** plus an *"installed without its CUDA sources"*
  error — a **packaging** problem with your install (reinstall gridbook).
- **`BUILD : CUDA extension UNAVAILABLE`** preceded by a compiler error — a
  **toolchain** problem on your machine (usually no `nvcc`).
- **`BUILD : CUDA extension UNAVAILABLE`** preceded by an *"incompatible ...
  extension"* error naming missing symbols — the loaded binary does not match
  the current Python call contract. Use the named cache diagnostics and see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md#incompatible-jit-extension-the-module-loaded-but-has-the-wrong-api).
- **`BUILD : ... LOADED` but `RESULT : FALLBACK path`** — nothing is broken; an
  environment variable is forcing the slow path. `PRISMAQUANT_CB_DECODE=triton`
  is a bisection switch that some scripts and model cards set. Unset it.

All four are covered in [TROUBLESHOOTING](TROUBLESHOOTING.md).

---

## Per-artifact requirements

Weights must fit *alongside* the KV cache and activation headroom. All figures
below are measured or user-reported, and named as such.

| Artifact | Weights on disk | Measured serving configuration |
|---|---|---|
| [27B, 5.5 bpp](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) | 23.0 GB (21.4 GiB) | **MEASURED** on GB10. On a 32 GB RTX 5090: **USER-REPORTED** 20.54 GiB resident weights + 6.48 GiB KV at 32k context with `--language-model-only`, using the local patches from [issue #1](https://github.com/RobTand/gridbook/issues/1). Those two fixes are now on `master` (`9b6cb2f`; 35.86 → 20.82 GiB of weights measured on the maintainer's box), but the issue is still open and no 32 GB card has been retested by a maintainer — [details](TROUBLESHOOTING.md#out-of-memory-on-a-32-gb-card). |
| [Laguna-S-2.1, 6.0 bpp](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) | 89.4 GB (83.2 GiB) | **MEASURED** on one 128 GB DGX Spark: `--max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --enforce-eager --max-num-batched-tokens 16384`. Only 12 of 48 layers are full-attention, so a full 256k request costs ~6 GiB of fp8 KV. |
| [Hy3-295B-A21B, 2.9 bpp](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) | 105.7 GB (98.5 GiB) | **MEASURED** on one 128 GB DGX Spark: `--max-model-len 8192`–`16384`, fp8 KV (12.7 GiB at 12288). Native 262k context does not fit — true of any ~105 GB weight class on this box. |

**On unified-memory boxes (DGX Spark), `--gpu-memory-utilization` is not the
whole story.** GPU and host share one physical pool, so a high utilization value
plus framework/activation spikes can OOM-kill the *box*, not just the server.
The measured discipline on that hardware is: start at **0.85**, verify several
GiB of `MemAvailable` remain after the server is up, and only then raise it. On a
discrete GPU with its own VRAM, ordinary vLLM utilization guidance applies.

---

## Known limits

- **`tp=1` only.** No tensor-parallel support; `--tensor-parallel-size > 1` is
  untested and unimplemented for CB weights.
- **`--enforce-eager` is the published-model configuration.** The default
  opaque dispatch has since made mode-0 `FULL_DECODE_ONLY` capture-correct and
  it improved a close-rate 0.6B canary by 20.1%. Keep eager for published
  recipes until that graph setup clears their own streaming gate; see
  [`KERNELS.md`](KERNELS.md#cuda-graph-safety-rules).
- **This repo serves the format; it does not produce artifacts.** A reference
  encoder is a [roadmap](../ROADMAP.md) item.
- **vLLM internals coupling.** gridbook imports fused-MoE and quantization
  internals that are not part of any stability contract. Pin a vLLM version you
  have tested.
