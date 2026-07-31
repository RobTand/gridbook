#!/usr/bin/env bash
# Run the GPU-free part of the gridbook test suite, one file per pytest process.
#
#   usage: .github/scripts/run_cpu_tests.sh <tests-dir>
#
# WHY ONE PROCESS PER FILE
# ------------------------
# tests/test_target_namespace_compat.py deliberately injects stub `vllm.*`
# modules into sys.modules so it can exercise the config resolver without the
# serving stack.  Those stubs persist for the rest of the session, so a later
# file's `pytest.importorskip("vllm")` guard *passes* and the test then dies on
# `import vllm._custom_ops`.  Measured 2026-07-28 in a clean container
# (python:3.10-slim, installed wheel only, torch 2.13.0+cpu, no GPU / nvcc /
# vLLM / prismaquant):
#   pytest test_transient_fp8.py                     -> 1 passed, 7 skipped
#   pytest test_target_namespace_compat.py test_transient_fp8.py
#           (one process)                            -> 3 failed, 31 passed,
#                                                       4 skipped
# The 3 failures are spurious: ModuleNotFoundError on `vllm._custom_ops` in
# tests whose importorskip("vllm") guard was satisfied by the stubs.  Per-file
# isolation removes the coupling without touching the shared test sources.
# (The durable fix is a conftest that restores sys.modules; that lives in the
# plugin source tree, not here.)
#
# WHY NO PYTEST MARKERS
# ---------------------
# The suite already selects itself at runtime: every CUDA/vLLM/artifact-backed
# test is guarded by pytest.skip / importorskip / skipif(not cuda_ok).  Run
# per-file on a machine with no GPU, no nvcc, no vLLM and no artifacts, each
# file is all-pass-or-skip.  A marker layer on top would duplicate that logic
# across the test modules for no added signal, so it
# was not added -- see docs/RELEASING.md.
set -uo pipefail

TESTS="${1:?usage: run_cpu_tests.sh <tests-dir>}"

# Every optional monorepo/GPU dependency is guarded with importorskip/skip, so
# every test module can be collected in the released-package environment.
EXCLUDE=()

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [ ! -d "$TESTS" ]; then
  echo "no such tests dir: $TESTS" >&2
  exit 2
fi
if [ -e "$PWD/gridbook/__init__.py" ]; then
  echo "refusing to run from a directory containing ./gridbook -- the checkout" >&2
  echo "would shadow the installed wheel and the run would prove nothing." >&2
  exit 2
fi

fail=0
ran=0
skipped_files=()

for f in "$TESTS"/test_*.py; do
  base="$(basename "$f")"
  for x in "${EXCLUDE[@]}"; do
    if [ "$base" = "$x" ]; then
      skipped_files+=("$base")
      continue 2
    fi
  done
  echo "::group::$base"
  "$PYTHON_BIN" -m pytest "$f" -q --no-header -rs -p no:cacheprovider
  rc=$?
  echo "::endgroup::"
  # 0 = passed (or everything skipped), 5 = no tests collected.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    echo "::error file=tests/$base::pytest exited $rc"
    fail=1
  fi
  ran=$((ran + 1))
done

echo
echo "ran $ran test file(s)"
if [ "${#skipped_files[@]}" -gt 0 ]; then
  echo "excluded: ${skipped_files[*]}"
fi
if [ "$fail" -ne 0 ]; then
  echo "CPU test suite FAILED"
  exit 1
fi
echo "CPU test suite OK"
