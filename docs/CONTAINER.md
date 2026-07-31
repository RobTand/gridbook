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

The build takes a few minutes, most of it compiling the CUDA kernels, and it
**fails loudly** rather than producing a quietly-degraded image (see
[Build-time gates](#build-time-gates)). To build *and* re-run every check
recorded in [Verified vs untested](#verified-vs-untested):

```bash
bash scripts/verify-image.sh --tag gridbook:local
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

Size added over the base image, measured two ways because they disagree and both
get quoted: **+31.8 MB** by `docker image inspect -f '{{.Size}}'`
(10 617 665 238 → 10 649 446 254 B), which sums the layer contents; **+0.2 GB**
by `docker images` disk-usage accounting (32.2 → 32.4 GB), which rounds. Of that
31.8 MB only **3.2 MB** is the prewarmed kernel cache — the same build with
`--build-arg GRIDBOOK_PREWARM=0` measures 10 646 242 057 B — and the rest is the
gridbook package plus the build context kept at `/opt/gridbook/src`, so the exact
figure tracks the repo's own size. The header repair below adds zero bytes: it is
symlinks.

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

### Two defects in the base image this Dockerfile repairs

#### 1. Missing CUDA headers

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

#### 2. `docker run --user` cannot import vLLM at all

In the stock image, running under any UID that has no `/etc/passwd` entry —
i.e. essentially every `docker run --user 1000:1000` — dies on `import vllm`:

```
File ".../torch/_inductor/runtime/cache_dir_utils.py", line 23, in default_cache_dir
  sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
File "/usr/lib/python3.12/getpass.py", line 169, in getuser
  return pwd.getpwuid(os.getuid())[0]
KeyError: 'getpwuid(): uid not found: 1000'
```

torch calls `getpass.getuser()` at **import** time to name its cache
directories, and vLLM does the same for a temp directory. Setting
`TORCHINDUCTOR_CACHE_DIR` is *not* enough — measured: the import then dies one
frame later in `torch/_inductor/codecache.py`, which calls `default_cache_dir()`
directly and so never consults that variable. `getpass.getuser()` consults
`LOGNAME` before `pwd`, so this image bakes `LOGNAME=gridbook`, which repairs
every call site at once. It affects nothing but the *names* of cache and temp
directories; it confers no identity and no permission.

This one is not gridbook-specific either: it applies to anything running
`vllm/vllm-openai:v0.24.0` under `--user`.

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
| `GRIDBOOK_PREWARM` | `1` | `0` skips kernel compilation; the user pays a one-time build inside their first request instead. Both values are build-tested. |

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

**It is a process-wide torch setting, not a gridbook one.** gridbook JIT-compiles
inside the vLLM process, so there is no way to scope this variable to the plugin.
Baking it narrows the arch list for *every* torch JIT path in the container —
vLLM's own `cpp_extension` / inductor compiles included — from the base image's
`8.0 8.7 8.9 9.0 10.0 11.0 12.0` down to this one value (confirmed with
`docker inspect`: the image exports `TORCH_CUDA_ARCH_LIST=12.1a`). That is the
intended trade: the stock list omits `12.1` entirely, so it is *wrong* for
gridbook on the reference target, and compiling all eight arches would add
minutes to every build. If something else in your container needs the wider
list, restore it at run time with `-e TORCH_CUDA_ARCH_LIST="8.0 8.7 8.9 9.0 10.0
11.0 12.0 12.1a"` and accept a one-time gridbook kernel rebuild.

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
- the decode-GEMV kernel fails to compile;
- a non-root, non-group-0 UID (1000:1000, `HOME=/` — what `docker run --user`
  actually gives you) cannot `import vllm`, register the plugin, or load the
  prewarmed kernels. All three of those failures are silent at run time: the
  first two crash inside vLLM's swallow-and-continue plugin loader, the third
  downgrades decode to the Triton prototype with a warning.

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
| `-v hf-cache:/root/.cache/huggingface` | Model weights. Without it, every container restart re-downloads the artifact. Under `--user`, use `/opt/gridbook/hf` + `HF_HOME` instead — see [Running as a non-root UID](#running-as-a-non-root-uid---user). |
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

### Running as a non-root UID (`--user`)

Supported, and gated at build time. Two of the three moving parts still need
flags from you, because they are mount points rather than image state — so use
this recipe:

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/opt/gridbook/hf \
  -v gridbook-hf:/opt/gridbook/hf \
  -v gridbook-ext:/opt/gridbook/ext-cache \
  gridbook:local <model> ...
```

Why each piece:

- **The kernel cache must be writable by your UID.**
  `torch.utils.cpp_extension` takes a `lock` file *inside* the build directory
  before it even checks whether anything needs building, so merely *loading* the
  prewarmed cache needs create permission there. The image opens
  `/opt/gridbook/ext-cache` to all UIDs for exactly this reason. If it were only
  group-0-writable (the usual recipe), `--user 1000:1000` would fall back to the
  slow Triton decode path with nothing but a warning.
- **`HF_HOME` must point somewhere your UID can write.** `HF_HOME` defaults to
  `~/.cache/huggingface`; under `--user` docker sets `HOME=/` for a UID with no
  passwd entry, and `/root` is mode 0700 and unreadable anyway. The image ships a
  world-writable `/opt/gridbook/hf` as a mount point, and docker seeds a fresh
  *named* volume from it — including the mode — so the volume is writable by any
  UID. A **bind mount** here is on you: it keeps the host directory's ownership.
- **`LOGNAME` is already baked in**, which is what makes `import vllm` work at
  all under `--user` (see [base-image defect 2](#2-docker-run---user-cannot-import-vllm-at-all)).

Read-only root filesystems work too, with a writable `/tmp` and a writable
kernel cache:

```bash
docker run --rm --gpus all --read-only --tmpfs /tmp \
  -v gridbook-ext:/opt/gridbook/ext-cache ... gridbook:local <model>
```

Without `--tmpfs /tmp` a bare `--read-only` container fail-softs to the Triton
path (`FileNotFoundError: No usable temporary directory found`) — torch and vLLM
both need a writable temp directory.

---

## Keeping this image from rotting

The image pins a vLLM release, links headers by a computed rule, and bakes a
kernel cache — all three can break silently when `VLLM_TAG` moves. Nothing in CI
covers them (no `nvcc` on hosted runners, ~32 GB for the base image), so the
coverage is a script you run:

```bash
bash scripts/verify-image.sh                       # build + 10 run-time checks
bash scripts/verify-image.sh --no-build --tag X    # re-check an existing image
```

It builds the Dockerfile from the repo root and then checks, on the built image:
root, `--user 1000:1000`, `--user 1000:0`, `--user 65534:65534`, a named volume
over the kernel cache under `--user`, `--read-only --tmpfs /tmp` under `--user`,
three entrypoint argument shapes, and that `HF_HOME=/opt/gridbook/hf` is
writable. Exit status is 0 only if all pass. It needs no GPU.

Run it: before tagging a release, after changing `VLLM_TAG`, and after any
Dockerfile change. The build-time gates (below) run inside `docker build` and so
are covered by anyone's build, but the run-time behaviours are not.

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
  throughput but removes capture startup and memory. Published-model rows in
  `BENCHMARKS.md` use it; the explicitly labelled 0.6B graph canary uses
  mode-0 `FULL_DECODE_ONLY` with capture sizes `[1,2,4,8]`.

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
`vllm/vllm-openai:v0.24.0` (`linux/arm64`, GB10 host), **with no GPU attached** —
`docker build` never has one. Last run **2026-07-28**, from a clean
`git clone`-equivalent tree, i.e. the Quick Start command exactly as written.
Re-run it yourself with `bash scripts/verify-image.sh`; the run-time half of this
list *is* that script, so the two cannot drift.

**Verified**

- The image builds end to end; `docker build --check` reports no warnings.
- 59 missing CUDA headers are linked; `cusparse.h` resolves afterwards.
- Install verification passes: gridbook 0.1.0, vLLM 0.24.0, torch 2.11.0+cu130,
  entry point registered, packaged `csrc` present.
- `gridbook.register()` succeeds against **released vLLM 0.24.0** — the API
  canary passes.
- Kernels compile without a GPU: decode GEMV **28.7–29.8 s**, CUTLASS mid-M
  fused prefill **71.0–75.9 s**, **101.4–105.3 s** for the build step as a whole
  (three uncached builds; `torch.cuda.is_available() == False` throughout). Only
  `nvcc` and an explicit `TORCH_CUDA_ARCH_LIST` are required. These are the
  current Dockerfile's own build-log numbers on the host above and move with the
  host's CPU.
- Loading the prewarmed extensions instead of compiling them, measured two ways
  because the difference is a factor of ~20 and both get quoted:
  - **inside a process that has already imported torch/vLLM** — the serving
    condition — decode **0.04–0.05 s**, fused **0.02 s** (3 runs);
  - **from a cold `python3 -c`**, where the timing also contains torch's own
    import, decode **0.59–0.66 s**, fused **0.80 s**, both **1.40–1.46 s**
    (3 runs).
- `--user` is genuinely supported, not asserted: `--user 1000:1000`,
  `--user 1000:0` and `--user 65534:65534` each import vLLM, register the plugin,
  and load **both** prewarmed extensions. A build-time gate re-checks this at
  uid/gid 1000:1000 with `HOME=/`, so the image cannot ship without it.
  (Both halves of this were broken before: `getpass.getuser()` on import, and a
  kernel cache only writable by group 0.)
- Read-only rootfs: `--read-only --tmpfs /tmp` plus a named volume for the kernel
  cache works, including combined with `--user 1000:1000`.
- Named volumes preserve the prewarmed cache; bind mounts shadow it.
- `-e HF_HOME=/opt/gridbook/hf` is writable under `--user 1000:1000`.
- The entrypoint's host/port defaulting behaves correctly across the bare,
  `--port=9000` and `--host 127.0.0.1` shapes.
- The build **fails** when the packaged CUDA sources are removed (the gate is
  real, not decorative).
- `--build-arg GRIDBOOK_CUDA_ARCH=9.0` builds successfully, compiles the decode
  GEMV for `sm_90` (28.4 s), and correctly skips the Blackwell-only fused
  prefill prewarm.
- `--build-arg GRIDBOOK_PREWARM=0` builds successfully: the prewarm step is
  skipped, the kernel-cache directory is still created world-writable so the
  first-use build works under `--user`, and the image is 3 MB smaller.
- The run-time checks are known to *fail* on an image without these fixes: run
  `scripts/verify-image.sh --no-build` against a pre-fix image and 5 of 10 fail.

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
- **Nothing builds this image automatically.** CI cannot: GitHub-hosted runners
  have no `nvcc`, and the base image alone is ~32 GB on disk. `scripts/verify-
  image.sh` is therefore a **manual** gate — run it before tagging a release,
  after bumping `VLLM_TAG`, and after any Dockerfile change. Until someone runs
  it, this section is a claim about the last time it was run, not about HEAD.

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
production serving target. The message names the exception; the usual causes are:

| In the message | Cause |
|---|---|
| `PermissionError: ... '/opt/gridbook/ext-cache/lock'` | The kernel cache is not writable by your UID — you mounted your own directory over it, or you are on an image predating this fix. |
| `FileNotFoundError: No usable temporary directory` | `--read-only` without `--tmpfs /tmp`. |
| a long `nvcc` error / a recompile every start | `TORCH_CUDA_ARCH_LIST` differs from build time, or a bind mount shadowed the prewarmed cache. |

**`KeyError: 'getpwuid(): uid not found: 1000'` at startup**
You are on a base image or an older gridbook image without the `LOGNAME` fix,
running `--user` with a UID that has no `/etc/passwd` entry. Workaround without
rebuilding: add `-e LOGNAME=anything` to the `docker run`.

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
