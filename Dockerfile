# syntax=docker/dockerfile:1.7
# =============================================================================
# gridbook — out-of-tree vLLM quantization plugin for NVFP4-CB / FP8-CB
#
# Build:  docker build -t gridbook:local .
# Serve:  docker run --gpus all --ipc=host -p 8000:8000 gridbook:local <model>
#
# See docs/CONTAINER.md for the full run recipe, volume mounts and VRAM guidance.
# =============================================================================

# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
# WHY v0.24.0: gridbook's only *measured* serving stack is a vLLM source build
# (0.23.1rc1.dev764+g54b16d8a9, built 2026-07-03). No published vLLM release
# corresponds to that build exactly, so there is no tag that reproduces it.
# v0.24.0 (tagged 2026-06-30) is the official release nearest in time, and it
# was verified to match the measured stack on every dimension the kernels touch:
#
#     torch 2.11.0+cu130 · triton 3.6.0 · nvcc 13.0 · CUTLASS 4.3.4 bundled at
#     vllm/third_party/fmha_sm100/cutlass · linux/arm64 + linux/amd64
#
# and to export every vLLM symbol the plugin imports (including the private
# fused-MoE internals: RoutedExperts, FusedMoEMethodBase, MoEActivation,
# dispatch_fused_moe_kernel, _get_config_dtype_str, moe_align_block_size).
#
# HONEST LIMIT: v0.24.0 itself has NOT been served end-to-end with gridbook.
# What was verified for the base stack is toolchain parity, symbol presence,
# install, and native extension compilation. The 0.5 Dockerfile hard-gates its
# expanded required extension set below. v0.25.1 and v0.26.0 were checked
# to carry the same gridbook-facing symbols and can be selected with
# --build-arg VLLM_TAG=..., but they are further from the measured stack.
ARG VLLM_TAG=v0.24.0
FROM vllm/vllm-openai:${VLLM_TAG}

ARG VLLM_TAG

# -----------------------------------------------------------------------------
# CUDA architecture
# -----------------------------------------------------------------------------
# The upstream vLLM image ships TORCH_CUDA_ARCH_LIST="8.0 8.7 8.9 9.0 10.0 11.0
# 12.0" — note that 12.1 is ABSENT. gridbook's kernels are JIT-compiled by
# torch.utils.cpp_extension, which inherits that list, so on the GB10 / DGX Spark
# reference target (sm_121) the stock list would never emit matching SASS.
# We therefore set the list explicitly.
#
# TWO CONSEQUENCES — READ THIS:
#
# 1. The image's prebuilt kernel cache becomes architecture-locked. torch only
#    reuses a cached build when the arch flags match, so this value MUST be
#    identical at build time and run time or every kernel is silently
#    recompiled during model load. To target other hardware, rebuild with e.g.
#    --build-arg GRIDBOOK_CUDA_ARCH=9.0 (H100) or 8.9 (RTX 4090).
#
# 2. TORCH_CUDA_ARCH_LIST is a PROCESS-WIDE torch setting, not a gridbook one.
#    gridbook JIT-compiles inside the vLLM process, so there is no way to scope
#    it to this plugin from the environment. Baking it narrows the arch list for
#    *every* torch JIT path in the container — vLLM's own cpp_extension /
#    inductor compiles included — from the base image's
#    "8.0 8.7 8.9 9.0 10.0 11.0 12.0" to this single value. That is the intended
#    trade (the stock list omits 12.1 entirely, so on the GB10 reference target
#    it is wrong for gridbook and compiling all eight arches is minutes of build
#    time), but if something else in your container needs the broader list,
#    restore it at run time with -e TORCH_CUDA_ARCH_LIST=... — which invalidates
#    the prewarmed gridbook cache and triggers a one-time rebuild.
#
# See docs/CONTAINER.md.
ARG GRIDBOOK_CUDA_ARCH=12.1a

# Set to 0 to skip compiling the kernels into the image (smaller/faster build,
# but the user pays the one-time builds during model load).
ARG GRIDBOOK_PREWARM=1

# The experimental native-NVFP4 fused specialization is runtime-default-off,
# so its substantially larger CUTLASS build is not baked by default. Set this
# to 1 only for an image intended to run the explicit fused-FP4 experiment.
# This does not enable the runtime selector; it only makes that optional module
# resident in the image cache.
ARG GRIDBOOK_PREWARM_FUSED_FP4=0

LABEL org.opencontainers.image.title="gridbook" \
      org.opencontainers.image.description="Out-of-tree vLLM quantization plugin serving NVFP4-CB / FP8-CB product-codebook weight formats on Blackwell tensor cores" \
      org.opencontainers.image.source="https://github.com/RobTand/gridbook" \
      org.opencontainers.image.documentation="https://github.com/RobTand/gridbook/blob/master/docs/CONTAINER.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.authors="Robert Tand <robert.tand@icloud.com>" \
      org.opencontainers.image.base.name="vllm/vllm-openai:${VLLM_TAG}"

# TORCH_CUDA_ARCH_LIST must persist into the runtime environment so the
# prewarmed cache below is reused (a sub-second dlopen) instead of rebuilt (a
# ~30 s nvcc compile for the decode GEMV, minutes for the CUTLASS prefill).
# Exact measured timings live in ONE place — docs/CONTAINER.md, "Verified vs
# untested" — so the two files cannot drift apart. Do not restate digits here.
#
# PRISMAQUANT_CB_EXT_DIR is a fixed absolute path rather than ~/.cache, because
# the ninja build files record absolute paths and $HOME changes under
# `docker run --user`. A stable path keeps the cache valid for any UID.
#
# LOGNAME repairs a second defect in the base image, and it is not cosmetic:
# under `docker run --user 1000:1000` the UID has no /etc/passwd entry, so
# `getpass.getuser()` raises KeyError('getpwuid(): uid not found: 1000') — and
# torch calls it at IMPORT time (torch/_inductor/runtime/cache_dir_utils.py,
# reached from torch/_dynamo/package.py), so `import vllm` itself dies before
# anything of gridbook's runs. Setting TORCHINDUCTOR_CACHE_DIR is NOT sufficient
# (measured: the import then dies in torch/_inductor/codecache.py, which calls
# default_cache_dir() directly and ignores it). getpass.getuser() reads LOGNAME
# first, so one variable fixes every call site at once — in torch and in vLLM
# (vllm/config/vllm.py builds a tmp dir from it). gridbook itself never calls it;
# its own cache path is PRISMAQUANT_CB_EXT_DIR above. Affects only the naming of
# cache/tmp directories; it changes no identity and no permission.
# This is a base-image bug: `docker run --user` cannot import vLLM at all in
# stock vllm/vllm-openai:v0.24.0.
ENV TORCH_CUDA_ARCH_LIST=${GRIDBOOK_CUDA_ARCH} \
    PRISMAQUANT_CB_EXT_DIR=/opt/gridbook/ext-cache \
    LOGNAME=gridbook

# -----------------------------------------------------------------------------
# Complete the CUDA header set
# -----------------------------------------------------------------------------
# The upstream vLLM image installs the CUDA math *runtime* libraries but not
# their headers: /usr/local/cuda/include has no cusparse.h, cublas.h,
# cusolverDn.h or cufft.h. torch's <ATen/cuda/CUDAContext.h> includes
# <cusparse.h>, so *every* torch cpp_extension CUDA JIT build fails in the
# stock image with "fatal error: cusparse.h: No such file or directory".
# Older Gridbook releases silently downgraded to a slow Triton fallback after
# that build failure; the native-only runtime now fails the affected CB
# operator closed. Either outcome makes the image unusable, so this remains a
# build-time gate.
#
# The matching headers are already inside the image: torch's own nvidia-*-cu13
# wheels ship a complete include tree under site-packages/nvidia/cu13/include,
# version-matched to the torch build by construction. So rather than apt-install
# hundreds of MB of -dev packages (libcublas-dev alone carries huge static
# archives), link in exactly the headers that are MISSING.
#
# The rule is computed, not a hardcoded list: link a wheel header only when no
# entry of that name already exists in the toolkit include dir. Nothing is ever
# shadowed, and the step self-heals if a future base image ships them properly.
RUN <<'EOF'
set -eu
inc="$(python3 -c "
import glob
p = sorted(glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cu*/include'))
print(p[0] if p else '')
")"
dst=/usr/local/cuda/include
if [ -z "$inc" ] || [ ! -d "$inc" ]; then
  echo "[gridbook] NOTE: no bundled nvidia/cu*/include found; leaving the CUDA"
  echo "           header set as-is. If headers are missing the kernel prewarm"
  echo "           below will fail the build with the exact missing header."
  exit 0
fi
linked=0
for src in "$inc"/*; do
  name="$(basename "$src")"
  if [ ! -e "$dst/$name" ]; then
    ln -s "$src" "$dst/$name"
    linked=$((linked + 1))
  fi
done
echo "[gridbook] linked $linked missing CUDA headers from $inc into $dst"
test -e "$dst/cusparse.h" || { echo "[gridbook] FATAL: cusparse.h still unresolved" >&2; exit 1; }
EOF

# -----------------------------------------------------------------------------
# Install gridbook
# -----------------------------------------------------------------------------
# --no-deps is deliberate: torch, safetensors, huggingface-hub and vLLM are
# already installed in the base image, and letting pip resolve `torch` here
# could pull a different torch wheel over the one vLLM was compiled against.
# Triton may exist as a vLLM-owned dependency, but Gridbook neither depends on
# nor dispatches it for CB operators. The verification step below asserts each
# Gridbook runtime dependency actually imports, so nothing is assumed.
COPY . /opt/gridbook/src

RUN pip install --no-deps --no-cache-dir /opt/gridbook/src \
    && rm -rf /root/.cache/pip

# -----------------------------------------------------------------------------
# Verify the install (build fails loudly rather than shipping a broken image)
# -----------------------------------------------------------------------------
WORKDIR /
RUN python3 <<'PY'
import os, sys
from importlib.resources import files
from importlib.metadata import entry_points, version

problems = []

# 1. Runtime dependencies. Installed with --no-deps, so prove they are present
#    instead of trusting the base image.
for mod in ("torch", "safetensors", "huggingface_hub", "vllm"):
    try:
        __import__(mod)
    except Exception as exc:
        problems.append(f"runtime dependency {mod!r} does not import: {exc}")

# 2. gridbook must resolve to a real installed package, not a stray source dir.
import gridbook
loc = os.path.dirname(os.path.abspath(gridbook.__file__))
if "packages" not in loc:
    problems.append(f"gridbook resolved to {loc}, which is not a site/dist-packages install")

# 3. The packaged CUDA sources must be present. This is the invariant that a
#    non-editable `pip install` used to break: csrc lived at the repo root, so
#    only `gridbook/` landed in site-packages and every extension build failed.
#    Older releases then selected a slow Triton fallback; the native-only tree
#    fails closed. Gating the complete serving floor here makes either defect a
#    build failure. Retained research-only sources such as cb_persistent_tc.cu
#    are intentionally not part of this runtime-required list.
REQUIRED = (
    "cb_gemv.cu",
    "cb_gemv_v2.cu",
    "cb_bf16_grouped_gemm.cu",
    "cb_fused_gemm.cu",
    "cb_fused_fp4_gemm.cu",
    "cutlass_fork/sm120_cb_mma_tma.hpp",
    "cutlass_fork/sm120_cb_fused_mma.hpp",
    "cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    "cutlass_fork/sm120_expert_row_broadcast.hpp",
)
try:
    csrc = files("gridbook") / "csrc"
    missing = [n for n in REQUIRED if not (csrc / n).is_file()]
    if missing:
        problems.append(
            f"packaged CUDA sources missing from the installed package: {missing} "
            f"(looked in {csrc}). The wheel must ship gridbook/csrc/*.cu.")
except Exception as exc:
    problems.append(f"could not locate the packaged csrc directory: {exc}")

# 4. The vLLM plugin entry point must be registered, or vLLM will never load us.
eps = [e for e in entry_points(group="vllm.general_plugins") if e.name == "gridbook"]
if not eps:
    problems.append("no 'gridbook' entry point in group 'vllm.general_plugins'")

if problems:
    print("\n[gridbook] IMAGE VERIFICATION FAILED:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)

import vllm, torch
print(f"[gridbook] verified: gridbook {version('gridbook')} | "
      f"vllm {vllm.__version__} | torch {torch.__version__} | "
      f"entry point OK | packaged csrc OK")
PY

# -----------------------------------------------------------------------------
# vLLM API canary
# -----------------------------------------------------------------------------
# vLLM's plugin loader wraps plugin.load() in `except Exception: logger.
# exception(...)` — it logs and CONTINUES. So if any vLLM internal gridbook
# imports has drifted, registration silently does not happen and the user's
# only symptom is an unrelated "unknown quantization method" much later, at
# model load. Running register() here surfaces that at build time instead.
RUN python3 <<'PY'
import sys, traceback
try:
    import gridbook
    gridbook.register()
except Exception:
    traceback.print_exc()
    print("\n[gridbook] vLLM API CANARY FAILED: gridbook.register() raised "
          "against this vLLM build. The plugin's imports have drifted from "
          "this vLLM version — pick a different --build-arg VLLM_TAG, or "
          "update the plugin. Refusing to ship an image whose plugin would "
          "silently not register at serve time.", file=sys.stderr)
    sys.exit(1)
print("[gridbook] vLLM API canary OK: register() succeeded")
PY

# -----------------------------------------------------------------------------
# Pre-warm the JIT kernel cache
# -----------------------------------------------------------------------------
# Measured: torch.utils.cpp_extension.load() needs nvcc and an explicit
# TORCH_CUDA_ARCH_LIST, but NOT a GPU — so the kernels compile during
# `docker build`, which never has a GPU attached (torch.cuda.is_available() is
# False throughout; it only warns). The main, FP4-v2, and grouped-BF16
# extensions are the required serving floor and all compile here. Crucially,
# this build does NOT call cb_gemv_v2_prepare(): that is a device attestation
# and runs during FP4 model load on the actual serving GPU.
#
# The per-kernel wall-clock numbers this step prints are recorded in
# docs/CONTAINER.md, "Verified vs untested". They are deliberately NOT restated
# in this file: two copies of a measurement drift, and one of them is then a
# false claim in a public file.
RUN python3 <<'PY'
import os, sys, time

if os.environ.get("GRIDBOOK_PREWARM", "1") != "1":
    print("[gridbook] prewarm disabled (GRIDBOOK_PREWARM != 1); kernels will "
          "build during the first model load.")
    raise SystemExit(0)

from gridbook import cuda_ext

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")


def build_capability():
    """Capability encoded by the single-arch image build argument."""
    import re
    match = re.search(r"(?:^|\s)(\d+)\.(\d+)", arch)
    if match is None:
        print(f"\n[gridbook] FATAL: cannot derive a CUDA capability from "
              f"TORCH_CUDA_ARCH_LIST={arch!r}.", file=sys.stderr)
        raise SystemExit(1)
    return int(match.group(1)), int(match.group(2))


def load_for_build(load):
    """Let a compile-only loader derive its target with no build-time GPU."""
    import torch
    original = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args, **kwargs: build_capability()
    try:
        return load()
    finally:
        torch.cuda.get_device_capability = original

# Decode GEMV — the production decode path. Hard gate: the native-only runtime
# fails closed when this extension is unavailable, so an image that cannot
# build it cannot serve CB operators.
t = time.time()
if cuda_ext.get_ext() is None:
    print("\n[gridbook] FATAL: the CUDA decode-GEMV extension failed to build "
          "at image build time (reason printed above). This is the production "
          "decode path — refusing to ship an image whose native CB execution "
          "would fail closed.", file=sys.stderr)
    sys.exit(1)
print(f"[gridbook] prewarmed decode-GEMV extension in {time.time() - t:.1f}s")

# FP4-v2 exact expander. Hard compile/load gate because every FP4 quality path
# needs this module even when PRISMAQUANT_CB_GEMV=inherited. Do not call its
# device prepare here: docker build has no GPU, and model load owns that check.
t = time.time()
if cuda_ext.get_ext_v2() is None:
    print("\n[gridbook] FATAL: the required FP4-v2 expansion extension failed "
          "to build at image build time (reason printed above). Refusing to "
          "ship an image missing a required quality-path module.",
          file=sys.stderr)
    sys.exit(1)
print(f"[gridbook] prewarmed FP4-v2 expansion extension in "
      f"{time.time() - t:.1f}s (device prepare deferred to model load)")

# Owned grouped-BF16 CUTLASS quality bridge. Its loader normally derives the
# target from the live GPU; docker build has none, so derive the same tuple from
# the image's explicit single-arch build argument for this compile/load only.
t = time.time()
if load_for_build(cuda_ext.get_bf16_grouped_ext) is None:
    print("\n[gridbook] FATAL: the required CUTLASS grouped-BF16 extension "
          "failed to build at image build time (reason printed above). "
          "Refusing to ship an image missing the native quality bridge.",
          file=sys.stderr)
    sys.exit(1)
print(f"[gridbook] prewarmed grouped-BF16 quality extension in "
      f"{time.time() - t:.1f}s")

# Mid-M fused prefill (CUTLASS). Genuinely sm_120-family only, and default-ON at
# runtime — without this prewarm a non-prewarmed image burns a multi-minute
# CUTLASS compile during model load. Soft: the runtime already
# falls back to the transient-expand path by design.
if "12.0" in arch or "12.1" in arch:
    t = time.time()
    if load_for_build(cuda_ext.get_fused_ext) is None:
        print("[gridbook] NOTE: mid-M fused prefill extension did not prewarm; "
              "mid-M prefill will use the transient-expand path (a supported "
              "fallback, not an error).")
    else:
        print(f"[gridbook] prewarmed mid-M fused prefill extension in "
              f"{time.time() - t:.1f}s")
else:
    print(f"[gridbook] NOTE: TORCH_CUDA_ARCH_LIST={arch!r} is not Blackwell "
          f"sm_120/sm_121 — skipping the CUTLASS mid-M fused prefill prewarm, "
          f"which is sm_120-family only. The exact native FP8 expansion path "
          f"is unaffected; FP4 model load remains gated by v2 device prepare.")

# Native-NVFP4 fused prefill is experimental and runtime-default-off. Build it
# only when explicitly requested; unlike the soft FP8 specialization, an
# explicit prewarm request is a build contract and therefore fails closed.
if os.environ.get("GRIDBOOK_PREWARM_FUSED_FP4", "0") == "1":
    if "12.0" not in arch and "12.1" not in arch:
        print(f"\n[gridbook] FATAL: GRIDBOOK_PREWARM_FUSED_FP4=1 requires a "
              f"Blackwell sm_120/sm_121 target, got {arch!r}.",
              file=sys.stderr)
        sys.exit(1)
    t = time.time()
    if load_for_build(cuda_ext.get_fused_fp4_ext) is None:
        print("\n[gridbook] FATAL: explicitly requested fused-FP4 extension "
              "did not prewarm (reason printed above).", file=sys.stderr)
        sys.exit(1)
    print(f"[gridbook] prewarmed optional fused-FP4 extension in "
          f"{time.time() - t:.1f}s")
elif os.environ.get("GRIDBOOK_PREWARM_FUSED_FP4", "0") != "0":
    print("\n[gridbook] FATAL: GRIDBOOK_PREWARM_FUSED_FP4 must be 0 or 1.",
          file=sys.stderr)
    sys.exit(1)
else:
    print("[gridbook] optional fused-FP4 prewarm disabled; runtime remains on "
          "the exact FP4-v2 expansion + grouped-BF16 quality path unless its "
          "explicit experimental selector is enabled.")

# The retained persistent-TC source is research-only: its serving selector,
# custom op and package loader were deleted after it measured negative. It is
# therefore neither a runtime requirement nor a prewarm target.
PY

# -----------------------------------------------------------------------------
# Make the prewarmed cache usable under `docker run --user`
# -----------------------------------------------------------------------------
# torch.utils.cpp_extension._jit_compile() takes a FileBaton on
# <build_directory>/lock BEFORE it decides whether anything needs building:
#
#     baton = FileBaton(os.path.join(build_directory, 'lock'))
#     if baton.try_acquire():   # os.open(..., O_CREAT | O_EXCL)
#
# so *loading* a fully prewarmed cache still requires CREATE permission in the
# cache directory. Read access is not enough. That makes the cache directory's
# write permission a correctness property of this image, not a convenience:
# without it `docker run --user 1000:1000` cannot load the required native
# decode path and serving fails closed.
#
# chgrp 0 + g+rwX (the usual OpenShift-style recipe) only covers `--user <uid>:0`.
# `--user 1000:1000` is neither root nor in group 0, so the cache directory is
# opened world-writable. This is a single-application container whose cache
# contains only compiled kernel objects; if you run untrusted code beside the
# server, mount your own cache directory over /opt/gridbook/ext-cache instead.
RUN <<'EOF'
set -eu
# mkdir: with GRIDBOOK_PREWARM=0 nothing has created the cache dir yet, and the
# runtime build must still be able to write into it under `--user`.
mkdir -p /opt/gridbook/ext-cache
# A Hugging Face cache location any UID can write. HF_HOME defaults to
# ~/.cache/huggingface, and under `--user` $HOME is `/` (docker's value for a UID
# with no passwd entry) — unwritable — while /root is mode 0700 and unreadable.
# Docker seeds a fresh NAMED volume from the image directory including its mode,
# so `-e HF_HOME=/opt/gridbook/hf -v gridbook-hf:/opt/gridbook/hf` is writable
# for any UID. See docs/CONTAINER.md, "Running as a non-root UID".
mkdir -p /opt/gridbook/hf
chgrp -R 0 /opt/gridbook
chmod -R g+rwX /opt/gridbook
chmod -R a+rwX /opt/gridbook/ext-cache /opt/gridbook/hf
EOF

# Gate it. This invariant was asserted before it was tested, and it was wrong;
# assert it now the same way `docker run --user 1000:1000` does: non-root UID,
# non-zero GID, no supplementary groups, and HOME=/ (what docker sets for a UID
# with no passwd entry — a build RUN leaves HOME unset, which is *easier* than
# the real case and would hide bugs). Cheap: cache hit, not a compile.
RUN <<'EOF'
set -eu
env HOME=/ setpriv --reuid 1000 --regid 1000 --clear-groups python3 - <<'PY'
import os, sys, time

# 1. The import barrier: getpass.getuser() has no passwd entry for this UID.
#    Stock vllm/vllm-openai dies here; the baked LOGNAME is what fixes it.
import getpass
try:
    user = getpass.getuser()
except Exception as exc:
    print(f"\n[gridbook] --user GATE FAILED: getpass.getuser() raised "
          f"{type(exc).__name__}: {exc} for uid 1000. torch calls this at "
          f"import time, so `import vllm` would die under `docker run --user`.",
          file=sys.stderr)
    sys.exit(1)

import vllm            # noqa: F401  — the actual import that used to die
import gridbook
gridbook.register()    # and the actual registration the plugin loader performs

if os.environ.get("GRIDBOOK_PREWARM", "1") != "1":
    print(f"[gridbook] --user gate OK (getpass={user}, import vllm, register); "
          f"kernel-cache load not checked because prewarm is disabled. The cache "
          f"directory is world-writable, so the model-load build works too.")
    raise SystemExit(0)

from gridbook import cuda_ext
import re
import torch

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
match = re.search(r"(?:^|\s)(\d+)\.(\d+)", arch)
if match is None:
    print(f"\n[gridbook] --user GATE FAILED: cannot derive a CUDA capability "
          f"from TORCH_CUDA_ARCH_LIST={arch!r}.", file=sys.stderr)
    sys.exit(1)
capability = int(match.group(1)), int(match.group(2))
original_get_device_capability = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda *args, **kwargs: capability

t = time.time()
ok_decode = cuda_ext.get_ext() is not None
t_decode = time.time() - t

t = time.time()
ok_v2 = cuda_ext.get_ext_v2() is not None
t_v2 = time.time() - t

t = time.time()
ok_grouped = cuda_ext.get_bf16_grouped_ext() is not None
t_grouped = time.time() - t

if not (ok_decode and ok_v2 and ok_grouped):
    print(f"\n[gridbook] --user GATE FAILED: uid/gid 1000:1000 could not load the "
          f"prewarmed required extensions (main={ok_decode}, v2={ok_v2}, "
          f"grouped_bf16={ok_grouped}; reason above). Under `docker run "
          f"--user` native CB execution would fail closed. Refusing to ship "
          f"the image.", file=sys.stderr)
    sys.exit(1)

msg = (f"[gridbook] --user gate OK: uid/gid 1000:1000 HOME=/ getpass={user}, "
       f"import vllm + register OK, required extensions loaded "
       f"(main={t_decode:.2f}s, v2={t_v2:.2f}s, "
       f"grouped_bf16={t_grouped:.2f}s)")

# The fused prefill extension is only prewarmed on the sm_120 family; check it
# only when it was actually built.
if os.path.isdir(os.path.join(os.environ["PRISMAQUANT_CB_EXT_DIR"], "fused")):
    t = time.time()
    if cuda_ext.get_fused_ext() is None:
        print("\n[gridbook] --user GATE FAILED: uid/gid 1000:1000 could not load "
              "the prewarmed mid-M fused prefill extension.", file=sys.stderr)
        sys.exit(1)
    msg += f", fused in {time.time() - t:.2f}s"

# Fused FP4 is optional and default-off; require cache-load integrity only when
# an explicit image build actually populated that module family.
fused_fp4_root = os.path.join(
    os.environ["PRISMAQUANT_CB_EXT_DIR"], "fused_fp4")
if os.path.isdir(fused_fp4_root):
    t = time.time()
    if cuda_ext.get_fused_fp4_ext() is None:
        print("\n[gridbook] --user GATE FAILED: uid/gid 1000:1000 could not "
              "load the explicitly prewarmed fused-FP4 extension.",
              file=sys.stderr)
        sys.exit(1)
    msg += f", fused_fp4 in {time.time() - t:.2f}s"
torch.cuda.get_device_capability = original_get_device_capability
print(msg)
PY
EOF

# The gate ran as uid 1000 and only ever created/removed the baton lock file,
# but re-assert the permissions so nothing it touched can narrow them.
RUN chmod -R a+rwX /opt/gridbook/ext-cache

# -----------------------------------------------------------------------------
# Entrypoint — OpenAI-compatible server on 0.0.0.0:8000
# -----------------------------------------------------------------------------
# Same shape as upstream's ENTRYPOINT ["vllm", "serve"], so every upstream flag
# and manifest still works, but defaults the bind to 0.0.0.0:8000 when the user
# has not specified it. Binding to loopback inside a container makes the server
# unreachable from the host even with -p published, which is a routine and
# confusing first-run failure.
RUN <<'EOF'
set -eu
cat > /usr/local/bin/gridbook-serve <<'SH'
#!/usr/bin/env bash
# gridbook entrypoint: `vllm serve` with a container-reachable default bind.
set -euo pipefail

have_host=0
have_port=0
for arg in "$@"; do
  case "$arg" in
    --host|--host=*) have_host=1 ;;
    --port|--port=*) have_port=1 ;;
  esac
done

if [ "$have_host" -eq 0 ]; then
  set -- "$@" --host 0.0.0.0
fi
if [ "$have_port" -eq 0 ]; then
  set -- "$@" --port 8000
fi

exec vllm serve "$@"
SH
chmod 0755 /usr/local/bin/gridbook-serve
EOF

WORKDIR /vllm-workspace
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/gridbook-serve"]
