# Installing gridbook

gridbook is a **plugin for vLLM**, not a runtime. You install it *into an
environment that already has vLLM and a working PyTorch*. It never installs or
upgrades torch/vLLM itself, and it deliberately declares no version pins on them
(see [Why nothing is pinned](#why-nothing-is-pinned)).

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
| **Python** | ≥ 3.10 | Measured on 3.12.3. 3.10/3.11 satisfy the code (no newer-syntax constructs) but have not been executed. |
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
| **A100 `sm_80`** | expected ✅ | ❌ **expected to error** — the fp8 GEMM it calls needs `sm_89+` and has no capability guard or fallback | n/a | **NOT RECOMMENDED.** The plugin declares a capability floor of 8.0 and will start, but any prompt longer than 16 tokens is expected to fail. INFERRED from code. |
| **Older / no `nvcc` / non-NVIDIA** | Triton fallback | Triton fallback | n/a | Correct numerics, prototype speed. Not a serving target. |

Per-artifact caveat: an artifact may contain non-CB groups served by stock
`compressed-tensors` (the 27B's vision tower is NVFP4 W4A16). Those groups carry
**their own** hardware requirements, independent of gridbook's kernels.

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

### 2. From PyPI — planned, not yet published

```bash
pip install gridbook          # not on PyPI yet — see ROADMAP.md
```

Until that lands, route 1 is the equivalent.

### 3. Container

The repo-root [`Dockerfile`](../Dockerfile) layers gridbook onto a pinned vLLM
image. Mount a host directory over the JIT build cache so it stays warm across
container restarts — see
[persisting the build cache](#persisting-the-jit-build-cache).

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

Run this in the environment where you will serve. It needs **no model, no vLLM
server and no GPU-heavy work** — it resolves the packaged sources and then
performs the same JIT build the plugin does, so it answers the only question
that matters: *am I on the CUDA path or the Triton fallback?*

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
    print("RESULT          : CUDA extension UNAVAILABLE -> Triton fallback (slow).")
    print("                  See docs/TROUBLESHOOTING.md")
else:
    print("RESULT          : CUDA extension LOADED ->", getattr(ext, "__file__", "?"))
    have = [s for s in ("cb_gemv_fp8", "cb_gemv_fp4_v2", "cb_expand_fp8_into",
                        "cb_moe_gemv_fp8") if hasattr(ext, s)]
    print("                  symbols:", have)
PY
```

A healthy install prints `RESULT : CUDA extension LOADED` and four symbols.

Two distinct failures are reported differently, on purpose:

- **`cb_gemv.cu found: False`** plus an *"installed without its CUDA sources"*
  error — a **packaging** problem with your install (reinstall gridbook).
- **`RESULT : CUDA extension UNAVAILABLE`** preceded by a compiler error — a
  **toolchain** problem on your machine (usually no `nvcc`).

Both are covered in [TROUBLESHOOTING](TROUBLESHOOTING.md).

---

## Per-artifact requirements

Weights must fit *alongside* the KV cache and activation headroom. All figures
below are measured or user-reported, and named as such.

| Artifact | Weights on disk | Measured serving configuration |
|---|---|---|
| [27B, 5.5 bpp](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) | 23.0 GB (21.4 GiB) | **MEASURED** on GB10. **USER-REPORTED** on a 32 GB RTX 5090: 20.54 GiB resident weights + 6.48 GiB KV at 32k context, with `--language-model-only` and the patches from [issue #1](https://github.com/RobTand/gridbook/issues/1). Check that issue's status before counting on a 32 GB fit. |
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
- **`--enforce-eager` is the tested configuration.** The decode dispatch branches
  on token count on the host. Naive CUDA-graph capture measured *worse*, because
  vLLM pads captured decode batches past the prefill threshold. Some paths are
  capture-clean (see [`KERNELS.md`](KERNELS.md#cuda-graph-safety-rules)), but
  eager is what the published numbers used.
- **This repo serves the format; it does not produce artifacts.** A reference
  encoder is a [roadmap](../ROADMAP.md) item.
- **vLLM internals coupling.** gridbook imports fused-MoE and quantization
  internals that are not part of any stability contract. Pin a vLLM version you
  have tested.
