# Installing gridbook

gridbook is a **plugin for vLLM**, not a runtime. Install it *into an environment
that already has vLLM and a compatible PyTorch*. It deliberately declares no
version pins for torch or vLLM, and vLLM is not a hard dependency. In a fresh
environment pip may resolve a generic torch build, so use the established
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
| **CUDA toolchain** | `nvcc` on `PATH` (or `CUDA_HOME` set) **in the serving process** | The kernels are JIT-compiled at runtime, not at install time. Missing `nvcc` may not fail package installation, but a required native serving operation fails closed when its extension cannot build. |
| **PyTorch** | the build your vLLM uses | Measured: `2.11.0+cu130`. Installing gridbook into a fresh environment can pull a *generic* PyPI torch that does not match your CUDA — install into the vLLM environment instead. |
| **vLLM** | already installed | Measured against `0.23.1rc1.dev764+g54b16d8a9`. |
| **Parallelism** | `tp=1` for everything except dense CB Linears | Dense CB Linears shard at load time (superblock-aligned, structured refusal on illegal boundaries). MoE, delegated groups and passthrough units refuse above one. See [known limits](#known-limits). |

---

## Tested software stack

Read this as a compatibility statement, not a support promise. gridbook depends
on a handful of vLLM internals (fused-MoE classes, the quantization registry,
the registered `vllm._C` CUDA operator ABI) and on CUTLASS headers that ship
*inside the vLLM install*, so vLLM version drift is the most likely source of
breakage. Gridbook calls the registered native FP8 quantizer and CUTLASS
scaled-matmul operators directly; it deliberately does not use the
fallback-capable `vllm._custom_ops` convenience wrappers.

| Component | Version | Evidence |
|---|---|---|
| vLLM | `0.23.1rc1.dev764+g54b16d8a9` | **MEASURED** — the image every published benchmark ran in, re-verified 2026-07-28. |
| torch | `2.11.0+cu130` | **MEASURED**, same image. |
| Triton in the surrounding vLLM image | `3.6.0` | **MEASURED**, same image, but not a Gridbook dependency or Gridbook serving backend. |
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

**Four** extensions compile against CUTLASS headers discovered under
`vllm/third_party/`, not a vendored copy: the required grouped-BF16 quality
bridge, the mid-M fused FP8-CB prefill lane, the fused NVFP4-CB lane, and the
fused FP4-CB v2 mid-M lane. That is deliberate — it guarantees ABI agreement
with vLLM's own kernels — but it means their availability is a property of
*your* vLLM build. If those headers are absent or in a different layout,
Gridbook may use the separately qualified native CUDA transient-expand +
CUTLASS route for that shape. If no qualified native route is available,
serving fails closed — and note that the grouped-BF16 bridge is **required**,
not optional, so a build that cannot find CUTLASS at all cannot serve FP4-CB.

`PRISMAQUANT_CUTLASS_INCLUDE` overrides the discovery for all four. Point it at
the `include` directory of a CUTLASS checkout — the one holding
`cutlass/cutlass.h`:

```bash
export PRISMAQUANT_CUTLASS_INCLUDE=/path/to/cutlass/include
```

Two situations need it: a virtualenv with no vLLM wheel to discover headers
from, and building against a CUTLASS newer than the bundled copy. A
set-but-wrong value **fails** with the missing header named rather than falling
back to the bundled tree, because silently compiling against a different CUTLASS
than the one you asked for is precisely the surprise the override exists to
prevent.

---

## Hardware matrix

The main decode kernel (`gridbook/csrc/cb_gemv.cu`) contains no architecture
guards, inline PTX, or fixed `-arch` flag, so that translation unit is expected
to compile from `sm_80` up. That fact is **not** an FP4-CB support claim. Every
FP4-v2 quality path also needs `cb_expand_v2` from `cb_gemv_v2.cu`; its
device-prepare contract currently admits only CUDA compute capability 12.0 or
12.1. Gridbook attests that floor during weight load and rejects the FP4-CB
layer if it is not met. The owned grouped-BF16 CUTLASS GEMM used after expansion
is independently SM80-compatible, but it cannot make an FP4 serve viable
without the Blackwell-only expander.

| GPU | Dense `M ≤ 8` | FP8-CB `M=9–128` | Native general / FP4 `M>8` | Overall |
|---|---|---|---|---|
| **GB10 / DGX Spark `sm_121`** | ✅ native CUDA GEMV | fused CUTLASS when eligible, otherwise CUDA expand + CUTLASS | FP8 CUDA expand + CUTLASS; FP4 native BF16 expand + owned CUTLASS grouped (`E=1`) | **MEASURED target.** Published artifact results remain tied to their recorded pre-change commits until the new dispatch is rebenchmarked. |
| **RTX 5090 `sm_120`** | expected same | expected same | expected same | **USER-REPORTED** working on the 27B artifact before the final native-only dispatch. The arch flag differs (`sm_120` vs measured `sm_121`); no new speed numbers exist. |
| **H100 `sm_90`** | FP8-CB decode expected; FP4-CB is rejected at weight load | Blackwell fused kernel ineligible; native FP8 expand + CUTLASS expected | FP8 native expansion + CUTLASS expected; FP4 unavailable because v2 expander prepare rejects this device | **FP8-ONLY IS INFERRED / UNTESTED. FP4-CB IS UNSUPPORTED IN 0.5.** |
| **RTX 4090 / L40S `sm_89`** | FP8-CB decode expected; FP4-CB is rejected at weight load | Blackwell fused kernel ineligible; native FP8 expand + CUTLASS expected | FP8 native expansion + CUTLASS expected; FP4 unavailable because v2 expander prepare rejects this device | **FP8-ONLY IS INFERRED / UNTESTED. FP4-CB IS UNSUPPORTED IN 0.5.** |
| **A100 `sm_80`** | no production artifact lane: FP8 prefill needs `sm_89+`, and FP4 load requires cc 12.0/12.1 | ❌ FP8-CB requires `sm_89+` | grouped BF16 GEMM can compile, but the required FP4-v2 expander rejects this device | **UNSUPPORTED FOR PRODUCTION CB SERVING IN 0.5.** No slow fallback is selected. |
| **Supported NVIDIA GPU, no `nvcc`** | unavailable unless compatible extensions were prebuilt | unavailable unless compatible extensions were prebuilt | unavailable | **FAILS CLOSED.** There is no Triton serving fallback. |
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
torch or other runtime packages again:

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

`torch` is declared as a bare requirement with no version bound. Triton is not a
Gridbook dependency. This is deliberate: vLLM environments run local-version wheels
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
| `prismaquant_cb_ext` (decode GEMV, grouped-MoE decode GEMV, FP8 prefill expander, activation QDQ/combine) | `gridbook/csrc/cb_gemv.cu` | **at weight load** for every CB artifact | **Required.** About 30 s once for the recorded main-extension build — see below. |
| `prismaquant_cb_v2_ext` (FP4-v2 quality expander plus optional alternate FP4 decode GEMV) | `gridbook/csrc/cb_gemv_v2.cu` | **at weight load for every FP4-v2 layer**, regardless of `PRISMAQUANT_CB_GEMV`; device prepare immediately attests cc 12.0/12.1. The selector controls only which decode GEMV runs. | **Required for every FP4-v2 quality path; fail-closed.** Compilation itself needs no GPU, but the device prepare does. |
| `pq_cb_bf16_grouped` (quality-preserving grouped/dense bridge) | `gridbook/csrc/cb_bf16_grouped_gemm.cu` | **at weight load** for routed quality prefill and dense FP4-CB quality prefill; dense uses grouped count `E=1` | **Required by those paths; fail-closed.** The generic SM80-compatible schedule is not Blackwell-optimized and measured 6–17% slower than segmented BF16 matmuls on warm synthetic DSV4 shapes. |
| `pq_cb_fused` (FP8-CB mid-M fused CUTLASS prefill) | `gridbook/csrc/cb_fused_gemm.cu` + `csrc/cutlass_fork/*.hpp` | **at weight load** when `PRISMAQUANT_CB_FUSED_MIDM` is enabled; availability and mode are fixed before serving, and later environment mutation raises | **Optional specialization.** An unavailable specialization stays on the exact native FP8 expansion/CUTLASS route; it is never first-built inside an eligible forward. |
| `pq_cb_fused_fp4_<identity>` (native-NVFP4 fused CUTLASS prefill) | `gridbook/csrc/cb_fused_fp4_gemm.cu` + `csrc/cutlass_fork/*.hpp` | only when the explicit fused-FP4 experiment is preloaded or selected | **Optional, experimental, and default-off.** An unavailable specialization stays on the exact native FP4-v2 expansion/grouped-BF16 route. |

The one retained persistent-N `.cu` file (`cb_persistent_tc.cu`) is a research
source, not a sixth serving extension: its selector, custom op, and package
loader are deleted, and its test compiles it directly behind an opt-in.

**Where "~30 s" comes from.** Cold `get_ext()` in the reference container
(`vllm-node:latest`, compile-only, **no** `--gpus`, `PRISMAQUANT_CB_EXT_DIR`
empty): **28.7 s** at `TORCH_CUDA_ARCH_LIST=12.1` and **32.3 s** at `12.0`
(2026-07-28), against **29.4 s / 29.7 s** recorded for the same check in
`gridbook/cuda_ext.py`. It is a compile time, so it moves with your arch list,
your `nvcc` and your host — treat ~30 s as the order of magnitude, not a
specification.

The main extension is always required. FP4-v2 additionally requires the v2
extension and its device prepare; routed/native quality prefill and dense FP4
quality prefill require the grouped-BF16 extension. Serving fails closed if a
required extension cannot build, load, or attest its device contract. An
optional fused specialization may be skipped only because dispatch has a
separately qualified native CUDA/CUTLASS path; it never selects Triton. The
verification step below distinguishes main-extension readiness from a package
that merely imports; artifact-specific required extensions are attested during
weight load.

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

**Every module owns a subdirectory** of that root, so no two ninja workspaces
share artefacts:

```
main                    # cb_gemv.cu           (decode GEMV / QDQ / expanders)
v2                      # cb_gemv_v2.cu        (FP4-v2 GEMV + exact expander)
bf16_grouped/<digest>   # cb_bf16_grouped_gemm.cu  (quality prefill bridge)
fused/<digest>          # cb_fused_gemm.cu     (FP8-CB fused prefill)
fused_fp4/<digest>      # cb_fused_fp4_gemm.cu (NVFP4-CB fused prefill)
```

The three CUTLASS modules key their directory (and their module name) by a
digest of their packaged sources, Gridbook headers, compiled-in lane macros,
target capability and toolchain ABI. That is what makes an edited header
impossible to serve from a stale cached kernel — and it means **upgrading
Gridbook costs one rebuild per affected module**, landing in a new digest
directory. The old directories are inert; delete them to reclaim the space.

---

## Verify the install

Run this in the environment where you will serve. It needs **no model and no
vLLM server** — it resolves the packaged sources, performs the same JIT build the
plugin does, and reports the environment switches that can route around the
result.

**What it can and cannot tell you.** It proves that the packaged `.cu` sources
resolve from your install and that they compile here. It does **not** prove a
kernel ever runs on your GPU — the
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
    print("BUILD           : REQUIRED CUDA extension UNAVAILABLE -> CANNOT SERVE.")
    print("                  Gridbook has no Triton fallback; see docs/TROUBLESHOOTING.md")
else:
    print("BUILD           : CUDA extension LOADED ->", getattr(ext, "__file__", "?"))
    have = [s for s in ("cb_gemv_fp8", "cb_gemv_fp4_v2", "cb_expand_fp8",
                        "cb_moe_gemv_fp8") if hasattr(ext, s)]
    print("                  symbols:", have)

env = {k: v for k, v in os.environ.items() if k.startswith("PRISMAQUANT_")}
retired = {k: v for k, v in env.items()
           if v.lower() == "triton" or "TRITON" in k}
print("retired switches:", retired or "none set")
print("RESULT          :", "main native extension ready"
      if ext is not None else "NOT READY TO SERVE")
PY
```

A healthy install prints `BUILD : CUDA extension LOADED`, the listed symbols,
no retired switches, and `RESULT : main native extension ready`.

The `symbols:` line above is informational. The loader itself asserts the full
native symbol family its callers dereference, so `BUILD : CUDA extension
LOADED` already means that ABI is present; a build that lacks one reports an
incompatible CUDA decode-GEMV extension and returns `UNAVAILABLE` instead of
loading. Artifact-specific paths additionally attest the direct registered FP8
CUDA/CUTLASS ABI and any grouped/fused extension they require during model load.

Four distinct failures are reported differently, on purpose:

- **`cb_gemv.cu found: False`** plus an *"installed without its CUDA sources"*
  error — a **packaging** problem with your install (reinstall gridbook).
- **`BUILD : CUDA extension UNAVAILABLE`** preceded by a compiler error — a
  **toolchain** problem on your machine (usually no `nvcc`).
- **`BUILD : CUDA extension UNAVAILABLE`** preceded by an *"incompatible ...
  extension"* error naming missing symbols — the loaded binary does not match
  the current Python call contract. Use the named cache diagnostics and see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md#incompatible-jit-extension-the-module-loaded-but-has-the-wrong-api).
- **A non-empty `retired switches:` line** — an old script or model card still
  requests a removed Triton lane. Unset it; current Gridbook has no equivalent
  serving switch.

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

- **Tensor parallel: dense CB and dense FP8-source Linears above one rank.**
  Since 2026-08-23 dense CB Linears load shard-aware at
  `--tensor-parallel-size > 1` (whole packed rows on the output axis,
  superblock-aligned byte windows on the input axis; a boundary that would
  split a group is a structured construction-time refusal), and dense
  `fp8_e4m3_ue8m0_block128` source-passthrough Linears do the same under
  their own law: each rank's extent on the sharded axis must be a whole
  multiple of the 128-element source block, per fused role on a merged plane,
  because vLLM narrows the UE8M0 scale plane over the block grid by ceil
  division. The same format's **grouped-BMM** units shard only at the
  degrees whose grouped geometry was measured (1, 2 and 4 — sharding a
  grouped plane divides the kernel's group count, so each degree is its own
  qualification); a degree outside that list is refused. Delegated
  compressed-tensors groups, other source-passthrough formats, quantized
  embedding units and mixed-format fused projections refuse at construction
  naming themselves; routed CB MoE has its own axis, below. No cross-node
  serve has been measured on this hardware; treat TP>1 as a correctness
  feature for models that do not fit one box, not a speedup.
- **Routed CB MoE needs `--enable-expert-parallel`, not `-tp` alone.** A CB
  expert stack's last dimension is superblock bytes, not input columns, so a
  tensor-parallel split would cut a packed superblock. Above one rank serve
  routed CB MoE with `-tp N --enable-expert-parallel`: each rank holds a
  disjoint subset of whole experts, byte-identical to the corresponding slice
  of a single-rank stack. `-tp N` without the flag refuses at construction and
  names the flag; so do data-/pipeline-context-/sequence-parallel EP, EPLB,
  and mixed per-expert-format stacks
  ([details](TROUBLESHOOTING.md#expert-parallel--tp-n---enable-expert-parallel)).
  The two-node gate has not been run, so this is correctness-only too.
- **Dense CB Linears are biasless.** A public dense CB call with non-`None`
  bias is rejected because the opaque native operation has no owned biased
  kernel in 0.5.
- **Dense FP4 0.5 serves unsigned product-v2 only.** Signed S-rungs and FP4-v1
  product layers remain format-valid research inputs, but model load rejects
  them until an exact every-M native quality path exists.
- **`--enforce-eager` is the published-model configuration.** The permanent
  opaque dispatch has since made mode-0 `FULL_DECODE_ONLY` capture-correct and
  it improved a close-rate 0.6B canary by 20.1%. Keep eager for published
  recipes until that graph setup clears their own streaming gate; see
  [`KERNELS.md`](KERNELS.md#cuda-graph-safety-rules).
- **This repo serves the format; it does not produce artifacts.** PrismaQuant is
  the canonical producer; Gridbook does not maintain a duplicate encoder.
- **vLLM internals coupling.** gridbook imports fused-MoE and quantization
  internals that are not part of any stability contract. Pin a vLLM version you
  have tested.
