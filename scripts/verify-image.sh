#!/usr/bin/env bash
# Build the gridbook image and check the run-time properties the Dockerfile
# claims. No GPU is used or required: every check here is an install/permission/
# import property, and `docker build` never has a GPU anyway.
#
# WHY THIS EXISTS: CI (.github/workflows/ci.yml) cannot build this image —
# GitHub-hosted runners have no nvcc, and the base image alone is ~32 GB on
# disk. So the image has no automatic coverage and will rot silently across vLLM
# releases. This script is the manual gate: run it before tagging a release,
# after bumping VLLM_TAG, and after any Dockerfile change.
#
#   bash scripts/verify-image.sh                  # build + check
#   bash scripts/verify-image.sh --no-build       # check an existing --tag
#   bash scripts/verify-image.sh --tag gridbook:x --context /path/to/tree
#
# Exit status is 0 only if every check passed.
set -uo pipefail

TAG=gridbook:verify
CONTEXT=.
BUILD=1
LOG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PASS=0
FAIL=0
declare -a FAILED

check() {  # check <name> <expected-substring> <docker-run-args...>
  local name="$1" want="$2"; shift 2
  local out rc
  out="$(timeout 600 docker run --rm "$@" 2>&1)"; rc=$?
  if [ $rc -eq 0 ] && printf '%s' "$out" | grep -qF -- "$want"; then
    PASS=$((PASS + 1))
    printf 'PASS  %-42s %s\n' "$name" "$(printf '%s' "$out" | grep -F -- "$want" | tail -1)"
  else
    FAIL=$((FAIL + 1)); FAILED+=("$name")
    printf 'FAIL  %-42s rc=%s\n' "$name" "$rc"
    printf '%s\n' "$out" | tail -12 | sed 's/^/        | /'
  fi
}

# The probe is passed on stdin so it needs no bind mount (a bind mount would
# itself be a permission variable in a test about permissions).
PROBE='
import getpass, os, sys, time
import vllm, gridbook
gridbook.register()
from gridbook import cuda_ext
t = time.time(); d = cuda_ext.get_ext(); td = time.time() - t
t = time.time(); f = cuda_ext.get_fused_ext(); tf = time.time() - t
print("PROBE uid=%d gid=%d user=%s HOME=%s vllm=%s register=OK decode=%s (%.2fs) "
      "fused=%s (%.2fs)" % (os.getuid(), os.getgid(), getpass.getuser(),
      os.environ.get("HOME"), vllm.__version__, d is not None, td, f is not None, tf))
sys.exit(0 if d is not None else 1)
'

if [ "$BUILD" = 1 ]; then
  echo "== docker build --check =="
  docker build --check "$CONTEXT" || { echo "build --check reported problems" >&2; exit 1; }
  echo "== docker build -t $TAG $CONTEXT =="
  if [ -n "$LOG" ]; then
    docker build --progress=plain -t "$TAG" "$CONTEXT" 2>&1 | tee "$LOG" | tail -3
    rc=${PIPESTATUS[0]}
  else
    docker build -t "$TAG" "$CONTEXT"; rc=$?
  fi
  [ "$rc" -eq 0 ] || { echo "BUILD FAILED (rc=$rc)" >&2; exit 1; }
fi

echo
echo "== run-time checks against $TAG =="

# 1. root: the ordinary path.
check "root" "decode=True" --entrypoint python3 "$TAG" -c "$PROBE"

# 2. `docker run --user` with a non-zero GID. This is the case the image used to
#    fail two ways: getpass.getuser() (no passwd entry) and a non-writable
#    kernel-cache directory. Both are silent — the second one downgrades decode
#    to the slow Triton path with only a warning.
check "--user 1000:1000" "decode=True" --user 1000:1000 --entrypoint python3 "$TAG" -c "$PROBE"
check "--user 1000:0" "decode=True" --user 1000:0 --entrypoint python3 "$TAG" -c "$PROBE"
check "--user 65534:65534 (nobody)" "decode=True" --user 65534:65534 --entrypoint python3 "$TAG" -c "$PROBE"

# 3. Named volume over the kernel cache: docker seeds it from the image, so the
#    prewarmed build must survive, under --user too.
VOL="gbverify-$$"
docker volume create "$VOL" >/dev/null
check "named volume + --user 1000:1000" "decode=True" \
  --user 1000:1000 -v "$VOL":/opt/gridbook/ext-cache --entrypoint python3 "$TAG" -c "$PROBE"
docker volume rm "$VOL" >/dev/null

# 4. Read-only rootfs. Needs a writable /tmp and a writable kernel cache; the
#    documented recipe is exactly this pair of flags.
VOL2="gbverify-ro-$$"
docker volume create "$VOL2" >/dev/null
check "--read-only + tmpfs + volume + --user" "decode=True" \
  --read-only --tmpfs /tmp --user 1000:1000 -v "$VOL2":/opt/gridbook/ext-cache \
  --entrypoint python3 "$TAG" -c "$PROBE"
docker volume rm "$VOL2" >/dev/null

# 5. The entrypoint's host/port defaulting. `vllm serve` needs a GPU, so run the
#    wrapper with a stub `vllm` on PATH that just echoes its arguments.
STUB='mkdir -p /tmp/stub && printf "#!/bin/sh\necho ARGS: \$*\n" > /tmp/stub/vllm && chmod +x /tmp/stub/vllm && PATH=/tmp/stub:$PATH /usr/local/bin/gridbook-serve'
check "entrypoint: bare" "ARGS: serve m --host 0.0.0.0 --port 8000" \
  --entrypoint bash "$TAG" -c "$STUB m"
check "entrypoint: --port=9000" "ARGS: serve m --port=9000 --host 0.0.0.0" \
  --entrypoint bash "$TAG" -c "$STUB m --port=9000"
check "entrypoint: --host 127.0.0.1" "ARGS: serve m --host 127.0.0.1 --port 8000" \
  --entrypoint bash "$TAG" -c "$STUB m --host 127.0.0.1"

# 6. The Hugging Face cache mount point must be writable by an arbitrary UID
#    (this is what makes `-e HF_HOME=/opt/gridbook/hf` work under --user).
check "HF_HOME writable under --user" "HF_WRITABLE" --user 1000:1000 --entrypoint bash "$TAG" \
  -c 'touch /opt/gridbook/hf/probe && echo HF_WRITABLE'

echo
echo "== $PASS passed, $FAIL failed =="
if [ "$FAIL" -ne 0 ]; then
  printf 'failed: %s\n' "${FAILED[*]}"
  exit 1
fi
