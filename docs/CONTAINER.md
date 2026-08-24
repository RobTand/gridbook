# Running gridbook in a container

The hardest part of installing gridbook is not gridbook — it is assembling a
toolchain: vLLM, a matching PyTorch, `nvcc`, and the CUDA headers those two
agree on. The [`Dockerfile`](../Dockerfile) at the repo root collapses that into
one build, layered on an official `vllm/vllm-openai` image.

The image also **compiles Gridbook's required CUDA/CUTLASS extensions at
image-build time**, so model load is not stalled behind kernel builds. The
default cache contains the main decode/support module, the required FP4-v2
expander module, the required grouped-BF16 quality bridge, and the optional
FP8 fused specialization on Blackwell. Experimental fused FP4 remains an
explicit, default-off build option.

**One opt-in lane is not prewarmed at all**: the fused FP4-CB v2 mid-M lane
(`PRISMAQUANT_CB_FP4_FUSED_MIDM`). The persistent-B grouped MoE lane
(`PRISMAQUANT_CB_MOE_PERSISTENT_B`) gained a soft Blackwell-gated prewarm in
0.8.9 when its default became `auto` — an image built before that, or one
whose prewarm failed, keeps the expand+bridge route per layer (announced)
until the JIT build lands in the extension cache. So there are six prewarm
targets for seven build-cache modules. Operationally: "resolved at model
load" means that if you set the remaining flag,
the **first model load in that container runs `nvcc` in-image and pays a
cold CUTLASS/CUDA build inside the load**, not at image-build time — several
minutes, on the request path, with the model already being read. It is a
one-time cost per cache, so the mitigation is the usual one: mount a persistent
volume over `PRISMAQUANT_CB_EXT_DIR` (see [Volumes](#volumes)) and do a
throwaway load with the flag set before serving traffic. Both lanes fail the
load rather than silently falling back, so a failed build is loud, not slow.

**Fused FP8 module build cost (old matrix measured; v10 matrix pending).**
The 2026-08-02 GB10 / cc 12.1 cold-cache measurement covered 20 kernel
instantiations — six `k_bits` rungs × {dense unscaled, dense scaled, grouped
TileM=128}, plus grouped TileM=256 at k28/k32. It was measured by clearing
`$PRISMAQUANT_CB_EXT_DIR/fused` and timing the loader:

| source | cold build |
|---|---|
| pre-K1.2 (merge base) | 71.4 s |
| K1.2 | 76.0 s, 75.7 s |

**+4.6 s (~6%) for that historical matrix.** v10 expands the canonical producer
law to K4..K48 step 4: 12 × {dense unscaled, dense scaled, grouped TileM=128}
plus grouped TileM=256 through K32, or 44 instantiations. Its cold build time has
not been measured. Do not reuse the ~76 s result for this larger source surface;
the assigned Blackwell compile gate must record a new number. Legacy irregular
K28..K48 reader rungs remain outside this collective and use the generic routes
(see [KERNELS](KERNELS.md#rung-coverage-what-this-lane-can-and-cannot-serve-k12)).

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

### Provision a release-pinned image before the first serve

Do not make the first production `docker run` clone or install Gridbook on its
startup path. Build a small derived image once, from the already-provisioned
serving image, using [`docker/Dockerfile.gridbook-pinned`](../docker/Dockerfile.gridbook-pinned):

```bash
docker build \
  -f docker/Dockerfile.gridbook-pinned \
  --build-arg BASE_IMAGE=gridbook:local \
  --build-arg GRIDBOOK_REF=v0.9.0 \
  -t gridbook:v0.9.0-pinned .
```

`GRIDBOOK_REF` is parameterized so the same recipe works for a release tag or,
during a pre-tag rehearsal, the exact release commit SHA. For a release image,
use the immutable `v0.8.6` tag. The direct-VCS requirement causes pip to write
the requested ref and resolved 40-character commit to the installed
distribution's PEP 610 `direct_url.json`; the Docker build reads that record
back and fails if the requested revision or resolved commit is absent. This is
the attestation-clean path: provisioning happens in an auditable image layer,
not as an unrecorded mutation when the server starts.

This generic recipe retains the repository's vLLM 0.24 base by default and is
not the DeepSeek-V4 DSpark-qualified image. DSV4-Flash must use the exact EUGR,
vLLM, torch, and FlashInfer tuple plus the `(64, 256)` startup canary recorded
in [`PLUGIN.md`](PLUGIN.md) and [`RELEASING.md`](RELEASING.md).

Then serve from the derived image with the same arguments shown above. On the
0.6B smoke, the one-time image build took **383 s**, while an actual serve start
from that completed image took **18 s**. The distinction is operationally
important: budget the 383-second provision once before traffic, so the first
serve sees the 18-second startup rather than paying installation and build work
on its critical path.

---

## What the image contains

| | |
|---|---|
| Base | `vllm/vllm-openai:v0.24.0` (`linux/arm64` + `linux/amd64`) |
| vLLM | 0.24.0 |
| PyTorch | 2.11.0+cu130 |
| Triton (ambient vLLM component; not used by Gridbook) | 3.6.0 |
| CUDA / `nvcc` | 13.0 |
| CUTLASS | 4.3.4, bundled by vLLM at `vllm/third_party/fmha_sm100/cutlass` |
| gridbook | installed from the build context, non-editable |
| Kernel cache | prebuilt at `/opt/gridbook/ext-cache` |
| Endpoint | OpenAI-compatible, `0.0.0.0:8000` |

Image size depends on the selected architecture and whether optional fused FP4
is prewarmed. Inspect it with `docker image inspect -f '{{.Size}}'
gridbook:local`; do not reuse the retired two-extension image's size numbers for
0.5. The header repair below adds no copied payload because it uses symlinks.

### Why `v0.24.0`

gridbook's only *measured* serving stack is a vLLM **source build**
(`0.23.1rc1.dev764+g54b16d8a9`, built 2026-07-03). No published vLLM release
reproduces that build, so no tag is a perfect match. `v0.24.0` (tagged
2026-06-30) is the official release nearest in time, and it was checked to agree
with the measured stack on everything the kernels touch — torch 2.11.0+cu130,
`nvcc` 13.0, CUTLASS 4.3.4 at the same path — and to export every
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

For Gridbook that makes required native extensions unavailable and serving fails
closed. The matching headers are, however, already inside the image:
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
  --build-arg GRIDBOOK_PREWARM_FUSED_FP4=0 \
  -t gridbook:local .
```

| Build arg | Default | Meaning |
|---|---|---|
| `VLLM_TAG` | `v0.24.0` | Base image tag. |
| `GRIDBOOK_CUDA_ARCH` | `12.1a` | `TORCH_CUDA_ARCH_LIST` used to compile the kernels. |
| `GRIDBOOK_PREWARM` | `1` | `1` compiles the three required modules (main, FP4-v2, grouped BF16) and the optional FP8 fused module when eligible. `0` skips all prewarming; model load then pays the builds. |
| `GRIDBOOK_PREWARM_FUSED_FP4` | `0` | `1` additionally compiles the experimental fused-FP4 module for a Blackwell target and treats failure as a broken explicitly requested image. It does not enable the runtime experiment. |

### `GRIDBOOK_CUDA_ARCH` is important — read this

The upstream vLLM image ships
`TORCH_CUDA_ARCH_LIST="8.0 8.7 8.9 9.0 10.0 11.0 12.0"`. **`12.1` is absent.**
That list used to decide gridbook's targets too, which made the reference target
(GB10 / DGX Spark, `sm_121`) run from PTX JIT or mismatched SASS outside this
image. Since 2026-08-01 **gridbook derives its own `-gencode` from the live
device** and no longer reads the list at all; `GRIDBOOK_CUDA_ARCH` now decides
what the image PREWARMS (the build host has no GPU, so the prewarm pins that
capability while it compiles) and what every OTHER torch JIT path in the
container inherits.

The consequence is that **the image's prebuilt kernel cache is
architecture-locked** — it holds binaries for the prewarmed capability, and a
container run on different hardware simply rebuilds for what it finds. To
prewarm for other hardware, rebuild:

```bash
docker build --build-arg GRIDBOOK_CUDA_ARCH=9.0 -t gridbook:h100 .   # H100, FP8-only
docker build --build-arg GRIDBOOK_CUDA_ARCH=8.9 -t gridbook:ada  .   # RTX 4090, FP8-only
docker build --build-arg GRIDBOOK_CUDA_ARCH=12.0 -t gridbook:5090 .  # RTX 5090
```

The optional CUTLASS mid-M FP8 fused kernel is `sm_120`-family only. On a
non-Blackwell arch the build skips that optional module and FP8 uses its exact
native transient-expand + CUTLASS path. The main decode extension remains
architecture-generic. FP4-CB is different: every FP4-v2 quality path needs the
v2 exact expander, whose **device prepare** accepts only cc 12.0/12.1. The image
can compile that module without a GPU for any requested arch, but an FP4 model
load on H100/Ada/A100 fails its device attestation; those images are FP8-only.

**`TORCH_CUDA_ARCH_LIST` no longer changes Gridbook's targets at run time.**
Since 2026-08-01 every Gridbook module derives its `-gencode` from the LIVE
device instead of inheriting that list (the stock list omits `12.1`, which is
the reference target — see the audit's §3 P0.1). Setting it at run time
therefore steers torch's *other* JIT paths, not Gridbook's: to target other
hardware, run on that hardware, or rebuild the image so the prewarm compiles for
it. A build host with no visible GPU pins the capability for the duration of the
build (the Dockerfile's `load_for_build`).

**It is a process-wide torch setting, not a gridbook one.** gridbook JIT-compiles
inside the vLLM process, so there is no way to scope this variable to the plugin.
Baking it narrows the arch list for *every* torch JIT path in the container —
vLLM's own `cpp_extension` / inductor compiles included — from the base image's
`8.0 8.7 8.9 9.0 10.0 11.0 12.0` down to this one value (confirmed with
`docker inspect`: the image exports `TORCH_CUDA_ARCH_LIST=12.1a`). That is the
intended trade: compiling all eight arches would add minutes to every build. If
something else in your container needs the wider list, restore it at run time
with `-e TORCH_CUDA_ARCH_LIST="8.0 8.7 8.9 9.0 10.0 11.0 12.0 12.1a"` — that is
now free for gridbook, whose targets come from the device rather than from this
variable.

### Build-time gates

The build **fails** rather than producing a quietly broken image if:

- a runtime dependency (`torch`, `safetensors`, `vllm`) does not import;
- gridbook did not install into `site-packages`;
- the packaged CUDA sources (`gridbook/csrc/*.cu`) are missing from the install —
  this is the exact defect that made a non-editable `pip install` fail-soft to
  the slow Triton path in retired releases;
- the `vllm.general_plugins` entry point is not registered;
- `gridbook.register()` raises against the chosen vLLM (an API-drift canary —
  vLLM's plugin loader swallows load exceptions and continues, so without this
  check drift surfaces much later as a confusing "unknown quantization method");
- any required module fails to compile or load: main decode/support, FP4-v2
  exact expansion, or the owned grouped-BF16 quality bridge. Docker build has
  no GPU, so it deliberately does not call the v2 device prepare; FP4 model
  load performs that cc 12.0/12.1 attestation on the serving device;
- `GRIDBOOK_PREWARM_FUSED_FP4=1` was requested but the optional fused-FP4
  module cannot be built for the selected Blackwell target;
- a non-root, non-group-0 UID (1000:1000, `HOME=/` — what `docker run --user`
  actually gives you) cannot `import vllm`, register the plugin, or load the
  prewarmed kernels. The first two can still be obscured by vLLM's
  swallow-and-continue plugin loader; the third now makes native execution
  unavailable and Gridbook fails closed.

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
  group-0-writable (the usual recipe), `--user 1000:1000` would be unable to load
  the required native extension and serving would fail closed.
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

Without `--tmpfs /tmp` a bare `--read-only` container cannot initialize the
native build/cache path (`FileNotFoundError: No usable temporary directory
found`) and serving fails closed — torch and vLLM both need a writable temp
directory.

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
writable. Every kernel-cache probe loads the three required compiled modules;
if either optional fused module was prewarmed, it must load too. Exit status is
0 only if all pass. It needs no GPU and therefore does **not** call
`cb_gemv_v2_prepare`; FP4 model load owns that device-specific check.

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

The last fully recorded container run was **2026-07-28** on
`vllm/vllm-openai:v0.24.0` (`linux/arm64`, GB10 host) with no GPU attached. It
verified the base-image repairs, plugin registration, arbitrary-UID cache
permissions, read-only-rootfs recipe, named-volume behavior, entrypoint, and
the then-current main/fused compile cache. Those exact compile times and image
sizes are intentionally not carried forward: 0.5 expands the required cache to
main + FP4-v2 + grouped BF16, so the retired two-module measurements do not
describe this Dockerfile.

The 0.5 manual gate is:

```bash
bash scripts/verify-image.sh --tag gridbook:local --log /tmp/gridbook-image.log
```

It must pass before a container is called release-verified. The build hard-gates
all three required extension loads; the runtime matrix repeats those loads as
root, arbitrary UIDs, named-volume, and read-only-rootfs cases. A prewarmed
optional FP8 or FP4 fused module is checked only when its cache directory is
present. This is compile/load evidence, not device evidence: no-GPU image checks
never call the FP4-v2 device prepare.

**Still untested unless a dated release record says otherwise**

- A GPU model load or generated token from the `v0.24.0`-based image. The only
  measured serving stack remains the
  `0.23.1rc1.dev764+g54b16d8a9` source build.
- `linux/amd64`; the prior container run was arm64.
- FP8 execution on `sm_90` / `sm_89`. FP4 execution there is not merely
  untested: 0.5 rejects it because the required v2 expander prepare admits only
  cc 12.0/12.1. A100 has no complete production CB lane in 0.5.
- A published registry image. This repository currently publishes the wheel,
  not a container image.

Nothing builds this image automatically: hosted runners lack `nvcc`, and the
base image alone is about 32 GB. `scripts/verify-image.sh` is therefore a manual
release gate after Dockerfile or `VLLM_TAG` changes.

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

**`[prismaquant-cb] WARNING: ... native Gridbook execution is unavailable and serving will fail closed`**
The native kernel build failed at runtime. There is no Gridbook Triton fallback.
The message names the exception; the usual causes are:

| In the message | Cause |
|---|---|
| `PermissionError: ... '/opt/gridbook/ext-cache/lock'` | The kernel cache is not writable by your UID — you mounted your own directory over it, or you are on an image predating this fix. |
| `FileNotFoundError: No usable temporary directory` | `--read-only` without `--tmpfs /tmp`. |
| a long `nvcc` error / a recompile every start | A bind mount shadowed the prewarmed cache, or the live device differs from the one the image prewarmed for (Gridbook compiles for the live device, not for `TORCH_CUDA_ARCH_LIST`). |

**`KeyError: 'getpwuid(): uid not found: 1000'` at startup**
You are on a base image or an older gridbook image without the `LOGNAME` fix,
running `--user` with a UID that has no `/etc/passwd` entry. Workaround without
rebuilding: add `-e LOGNAME=anything` to the `docker run`.

**Kernels rebuild on every container start**
The cache directory is not persisting, or this image predates a change to a
module's build inputs. Use a *named* volume for `/opt/gridbook/ext-cache`. The
identity-keyed modules (`fused/<digest>`, `fused_fp4/<digest>`,
`bf16_grouped/<digest>`) rebuild ONCE after any change to their packaged
sources, headers, compiled-in lane macros, target or toolchain ABI — that is the
mechanism refusing to serve a stale kernel, not a cache miss. Overriding
`TORCH_CUDA_ARCH_LIST` no longer affects Gridbook's targets (see above).

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
