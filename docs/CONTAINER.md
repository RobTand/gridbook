# Running gridbook in a container

The hardest part of installing gridbook is not gridbook — it is assembling a
toolchain: vLLM, a matching PyTorch, `nvcc`, and the CUDA headers those two
agree on. The [`Dockerfile`](../Dockerfile) at the repo root collapses that into
one build, layered on an official `vllm/vllm-openai` image.

The image also **compiles gridbook's CUDA kernels at image-build time**, so the
first request a user makes is not stalled behind a kernel build.

---

## Quick start

```bash
git clone https://github.com/RobTand/gridbook
cd gridbook
docker build -t gridbook:local .
```

Then serve a published artifact:

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  -v gridbook-ext:/opt/gridbook/ext-cache \
  gridbook:local \
  rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm \
  --served-model-name qwen \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager
```

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":16}'
```

Everything after the image name is passed straight to `vllm serve`, so every
upstream vLLM flag works unchanged.

---

## What the image contains

| | |
|---|---|
| Base | `vllm/vllm-openai:v0.24.0` (`linux/arm64` + `linux/amd64`) |
| vLLM | 0.24.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| CUDA / `nvcc` | 13.0 |
| CUTLASS | 4.3.4, bundled by vLLM at `vllm/third_party/fmha_sm100/cutlass` |
| gridbook | installed from the build context, non-editable |
| Kernel cache | prebuilt at `/opt/gridbook/ext-cache` |
| Endpoint | OpenAI-compatible, `0.0.0.0:8000` |

The image adds roughly **0.2 GB** on top of the base image.

### Why `v0.24.0`

gridbook's only *measured* serving stack is a vLLM **source build**
(`0.23.1rc1.dev764+g54b16d8a9`, built 2026-07-03). No published vLLM release
reproduces that build, so no tag is a perfect match. `v0.24.0` (tagged
2026-06-30) is the official release nearest in time, and it was checked to agree
with the measured stack on everything the kernels touch — torch 2.11.0+cu130,
Triton 3.6.0, `nvcc` 13.0, CUTLASS 4.3.4 at the same path — and to export every
vLLM symbol the plugin imports, including the private fused-MoE internals
(`RoutedExperts`, `FusedMoEMethodBase`, `MoEActivation`,
`dispatch_fused_moe_kernel`, `_get_config_dtype_str`, `moe_align_block_size`).

`v0.25.1` and `v0.26.0` were checked to carry the same gridbook-facing symbols
and can be selected with `--build-arg VLLM_TAG=v0.26.0`, but they sit further
from the measured stack. See [Verified vs untested](#verified-vs-untested).

### A defect in the base image this Dockerfile repairs

The stock `vllm/vllm-openai` image installs the CUDA math **runtime** libraries
but not their headers — `/usr/local/cuda/include` has no `cusparse.h`,
`cublas.h`, `cusolverDn.h` or `cufft.h`. Because torch's
`<ATen/cuda/CUDAContext.h>` includes `<cusparse.h>`, *every* `torch.utils.
cpp_extension` CUDA JIT build fails there:

```
fatal error: cusparse.h: No such file or directory
```

For gridbook that would be a silent downgrade to the Triton fallback path, not a
visible error. The matching headers are, however, already inside the image:
torch's own `nvidia-*-cu13` wheels ship a complete include tree that is
version-matched to the torch build by construction. The Dockerfile links in
**only the headers that are missing** — a computed rule, not a hardcoded list,
so nothing is ever shadowed and the step becomes a no-op if a future base image
ships them properly. On `v0.24.0` this links 59 headers and costs no extra bytes.

---

## Build options

```bash
docker build \
  --build-arg VLLM_TAG=v0.24.0 \
  --build-arg GRIDBOOK_CUDA_ARCH=12.1a \
  --build-arg GRIDBOOK_PREWARM=1 \
  -t gridbook:local .
```

| Build arg | Default | Meaning |
|---|---|---|
| `VLLM_TAG` | `v0.24.0` | Base image tag. |
| `GRIDBOOK_CUDA_ARCH` | `12.1a` | `TORCH_CUDA_ARCH_LIST` used to compile the kernels. |
| `GRIDBOOK_PREWARM` | `1` | `0` skips kernel compilation; the user pays a one-time build inside their first request instead. |

### `GRIDBOOK_CUDA_ARCH` is important — read this

The upstream vLLM image ships
`TORCH_CUDA_ARCH_LIST="8.0 8.7 8.9 9.0 10.0 11.0 12.0"`. **`12.1` is absent.**
gridbook's kernels are JIT-compiled by torch, which inherits that list, so on the
GB10 / DGX Spark reference target (`sm_121`) the stock list would never emit
matching SASS. The Dockerfile therefore sets the list explicitly, defaulting to
`12.1a`.

The consequence is that **the image's prebuilt kernel cache is
architecture-locked**. torch only reuses a cached build when the arch flags
match, and the value is baked into the image's environment so build time and run
time agree. To target other hardware, rebuild:

```bash
docker build --build-arg GRIDBOOK_CUDA_ARCH=9.0 -t gridbook:h100 .   # H100
docker build --build-arg GRIDBOOK_CUDA_ARCH=8.9 -t gridbook:ada  .   # RTX 4090
docker build --build-arg GRIDBOOK_CUDA_ARCH=12.0 -t gridbook:5090 .  # RTX 5090
```

The CUTLASS mid-M fused prefill kernel is `sm_120`-family only; on a non-Blackwell
arch the build prints a note and skips prewarming it, and at runtime that path
falls back to the transient-expand path by design. Decode is unaffected — the
decode GEMV is architecture-generic CUDA and compiles for every arch above.

You can also override the arch at *run* time (`-e TORCH_CUDA_ARCH_LIST=9.0`),
which invalidates the baked cache and triggers a one-time rebuild inside the
container. Rebuilding the image is the better path.

### Build-time gates

The build **fails** rather than producing a quietly broken image if:

- a runtime dependency (`torch`, `triton`, `safetensors`, `vllm`) does not import;
- gridbook did not install into `site-packages`;
- the packaged CUDA sources (`gridbook/csrc/*.cu`) are missing from the install —
  this is the exact defect that used to make a non-editable `pip install`
  fail-soft to the slow Triton path;
- the `vllm.general_plugins` entry point is not registered;
- `gridbook.register()` raises against the chosen vLLM (an API-drift canary —
  vLLM's plugin loader swallows load exceptions and continues, so without this
  check drift surfaces much later as a confusing "unknown quantization method");
- the decode-GEMV kernel fails to compile.

---

## Running

### Required flags

| Flag | Why |
|---|---|
| `--gpus all` | The container needs the GPU. |
| `--ipc=host` | vLLM uses shared memory; the default 64 MB `/dev/shm` is too small. |
| `-p 8000:8000` | Publish the port. The entrypoint already binds `0.0.0.0`; a loopback bind inside a container is unreachable from the host even when published. |

### Volumes

| Mount | Purpose |
|---|---|
| `-v hf-cache:/root/.cache/huggingface` | Model weights. Without it, every container restart re-downloads the artifact. |
| `-v gridbook-ext:/opt/gridbook/ext-cache` | Compiled kernels. |

**Use named volumes, not bind mounts, for the kernel cache.** Docker copies the
image's existing contents into a *named* volume on first use, so the prewarmed
kernels survive. A **bind mount shadows** the directory, hiding the prewarmed
build and forcing a recompile on first use. Verified on this image: with a named
volume the `.so` files are present; with a bind mount over the same path the
directory is empty.

If you prefer a host directory for model weights, bind-mounting the HF cache is
fine — that one starts empty anyway:

```bash
-v /path/on/host/hf:/root/.cache/huggingface
```

### Entrypoint

`ENTRYPOINT ["/usr/local/bin/gridbook-serve"]` wraps `vllm serve` and appends
`--host 0.0.0.0` and `--port 8000` **only when you have not supplied them**.
Explicit `--host`/`--port` (both `--port 9000` and `--port=9000` forms) always
win. To bypass it entirely, use `--entrypoint`:

```bash
docker run --rm --gpus all --entrypoint bash gridbook:local -lc 'python3 -c "import gridbook; print(gridbook.__file__)"'
```

---

## Published artifacts

All three currently-published gridbook artifacts, with their on-disk sizes from
the Hugging Face API:

| Artifact | Size | Notes |
|---|---|---|
| [`rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) | 23.0 GB | The best starting point. |
| [`rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm`](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) | 89.4 GB | Long-context; needs a 128 GB box. |
| [`rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) | 105.7 GB | 295B MoE on a single 128 GB box. |

Quality and speed measurements for these live in
[`BENCHMARKS.md`](BENCHMARKS.md), with their caveats. Nothing in this document
adds a performance claim.

---

## Memory guidance

Budget, roughly:

```
artifact bytes  +  KV cache  +  ~2-4 GB runtime/activation overhead
```

- **`--gpu-memory-utilization`** is a fraction of total GPU memory, and it is the
  main knob. The project's own serve scripts start at **0.80** and gate on free
  memory after load rather than trusting the fraction.
- **`--kv-cache-dtype fp8`** roughly halves KV bytes and is used by every
  published serve recipe for these artifacts.
- **`--max-model-len`** and **`--max-num-seqs`** multiply into the KV pool. The
  27B recipe above uses 32768 / 2. Reduce both first when a load OOMs.
- **`--enforce-eager`** skips CUDA-graph capture. It costs some decode
  throughput but removes a large capture-time memory spike; every measurement in
  `BENCHMARKS.md` was taken with it.

### On unified-memory boxes (GB10 / DGX Spark)

The GPU and the host share **one physical 128 GB pool**. Two consequences that
do not apply to a discrete-GPU box:

1. "Move it to CPU" frees nothing.
2. `--gpu-memory-utilization 0.9` is not a safety margin — it is carving from the
   same pool the OS and the container runtime are using, and it has taken the
   whole machine down. The project's operating discipline is to start at 0.80,
   confirm at least ~8 GiB of `MemAvailable` remains after the model loads, and
   only then raise it.

The 89.4 GB and 105.7 GB artifacts leave very little headroom on a 128 GB box;
expect to tune `--max-model-len` down substantially.

---

## Verified vs untested

Everything below was run on this repo's Dockerfile against
`vllm/vllm-openai:v0.24.0` (`linux/arm64`), **with no GPU attached** — `docker
build` never has one.

**Verified**

- The image builds end to end.
- 59 missing CUDA headers are linked; `cusparse.h` resolves afterwards.
- Install verification passes: gridbook 0.1.0, vLLM 0.24.0, torch 2.11.0+cu130,
  entry point registered, packaged `csrc` present.
- `gridbook.register()` succeeds against **released vLLM 0.24.0** — the API
  canary passes.
- Kernels compile without a GPU: decode GEMV **30.8 s**, CUTLASS mid-M fused
  prefill **79.5 s** (`torch.cuda.is_available() == False` throughout). Only
  `nvcc` and an explicit `TORCH_CUDA_ARCH_LIST` are required.
- Both extensions load from the baked cache in **0.65 s** at container start,
  versus ~110 s to compile them.
- The entrypoint's host/port defaulting behaves correctly across five argument
  shapes, including `--port=9000`.
- The build **fails** when the packaged CUDA sources are removed (the gate is
  real, not decorative).
- `--build-arg GRIDBOOK_CUDA_ARCH=9.0` builds successfully, compiles the decode
  GEMV for `sm_90`, and correctly skips the Blackwell-only fused prefill prewarm.
- Named volumes preserve the prewarmed cache; bind mounts shadow it.

**Untested — do not read these as working**

- **No GPU run of any kind.** No model was loaded from this image, no token was
  generated, and no throughput was measured from it.
- **vLLM 0.24.0 has never served a gridbook artifact end to end.** The only
  measured serving stack remains the `0.23.1rc1.dev764+g54b16d8a9` source build.
  What is established here is toolchain parity, symbol presence, install
  integrity and kernel compilation — not served correctness.
- **`linux/amd64` is unbuilt.** Only `arm64` was built. The base image is
  multi-arch, but nothing about gridbook on x86 has been exercised.
- **Execution on `sm_90` / `sm_89` / `sm_80` is untested.** `sm_90` was shown to
  *compile*; it was not run. See the
  [hardware matrix](INSTALL.md#hardware-matrix) for the per-path breakdown and
  the known `sm_89` floor on the dense FP8-CB prefill path — an A100 in
  particular is expected to fail on the first prompt longer than 16 tokens.
- **No image has been published** to any registry.

If you want the exact stack the published benchmarks were measured on, that is
the third-party arm64 image `eugr/spark-vllm@sha256:d0840ff0e0ba1899a51bf4cb473f
43d0c765288b8de708080ad9d95768615141`. It is a moving `:latest` tag from outside
this project and is not a reproducible dependency — recorded here for provenance,
not recommended as a base.

---

## Troubleshooting

Container-specific failures are below;
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) covers the plugin's own failure modes.

**`fatal error: cusparse.h: No such file or directory`**
You are building on a base image whose CUDA headers are incomplete and whose
`nvidia/cu*/include` wheels are absent. Install the dev packages
(`libcusparse-dev-13-0`, `libcublas-dev-13-0`) in a derived layer.

**`[prismaquant-cb] WARNING: ... falling back to the Triton decode path`**
The kernel build failed at runtime. The Triton path is correct but is not a
production serving target. Two usual causes: the arch flags changed between
build and run (check `TORCH_CUDA_ARCH_LIST` is still `12.1a` inside the
container), or the ext-cache directory was shadowed by a bind mount.

**Kernels rebuild on every container start**
The cache directory is not persisting, or the arch list changed. Use a *named*
volume for `/opt/gridbook/ext-cache`, and do not override
`TORCH_CUDA_ARCH_LIST` at run time.

**`unknown quantization method` at model load**
vLLM did not load the plugin. vLLM's loader logs plugin failures and continues,
so check the server log for a `gridbook` traceback near startup. The image's
build-time canary makes this unlikely unless you changed `VLLM_TAG`.

**Server starts but is unreachable from the host**
Confirm `-p 8000:8000` and that nothing overrode the bind to `127.0.0.1`.

**OOM during load, or the whole machine becomes unresponsive**
Lower `--gpu-memory-utilization` (try 0.75), then `--max-model-len`, then
`--max-num-seqs`. On a unified-memory box see the warning above — the failure
mode there is the host dying, not a clean CUDA OOM.

---

## Publishing to GHCR (not yet done)

No image has been published. When that decision is made, GHCR is the registry to
use: unlimited pulls for public images, no Docker Hub rate-limit cliff for users,
and `GITHUB_TOKEN` authenticates automatically inside Actions.

The steps, for whoever runs them:

```bash
# 1. Authenticate (classic PAT with write:packages, or GITHUB_TOKEN in Actions)
echo "$GHCR_TOKEN" | docker login ghcr.io -u RobTand --password-stdin

# 2. Build and tag. Match the tag to the base vLLM version so the coupling is
#    legible: gridbook 0.1.0 on vLLM 0.24.0.
docker build -t ghcr.io/robtand/gridbook:0.1.0-vllm0.24.0 \
             -t ghcr.io/robtand/gridbook:latest .

# 3. Push
docker push ghcr.io/robtand/gridbook:0.1.0-vllm0.24.0
docker push ghcr.io/robtand/gridbook:latest

# 4. Make the package public (GitHub → Packages → gridbook → Package settings →
#    Change visibility), otherwise pulls require authentication.
```

For a multi-arch image, build with buildx instead — note that the kernels are
arch-locked, so an `amd64`+`arm64` manifest still needs a `GRIDBOOK_CUDA_ARCH`
decision per platform, and the honest default may be to publish `arm64` only
until an x86 Blackwell box has actually been tested:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/robtand/gridbook:0.1.0-vllm0.24.0 --push .
```

Two commitments worth weighing before starting:

1. A published image is a **standing obligation** — roughly 11 GB compressed,
   rebuilt on every vLLM release users care about.
2. The closest precedent, `vllm-project/vllm-gguf-plugin`, publishes **no image
   at all** and ships only a wheel. A Dockerfile in the repo gets most of the
   reproducibility benefit at near-zero recurring cost, which is why it is the
   first step and the image is a deliberate second one.
