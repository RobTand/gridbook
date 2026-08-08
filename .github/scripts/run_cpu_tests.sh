#!/usr/bin/env bash
# Run the GPU-free part of the gridbook test suite, one file per pytest process.
#
#   usage: .github/scripts/run_cpu_tests.sh <tests-dir>
#
# WHY ONE PROCESS PER FILE
# ------------------------
# The vLLM-stubbing tests now use tests/conftest.py to snapshot and restore the
# exact affected import graph; a combined collection/run is order-independent.
# Keep per-file execution here because this is the installed-wheel release gate:
# it gives each optional dependency/CUDA probe a clean interpreter and reports
# the precise distribution test file that failed without sacrificing coverage.
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

# Some CPU tests exercise repository-only validation utilities under scripts/.
# The tests themselves are copied away from the checkout so local gridbook/
# cannot shadow the installed wheel, and the utilities deliberately remain out
# of that runtime wheel. GitHub happens to provide GITHUB_WORKSPACE, but the
# release gate is also a documented local command and must not depend on that
# ambient CI variable. Resolve the canonical source root from this script and
# export it only as a data-file locator; it is never added to PYTHONPATH.
if [ -z "${GRIDBOOK_SOURCE_ROOT:-}" ]; then
  GATE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  GRIDBOOK_SOURCE_ROOT="$(CDPATH= cd -- "${GATE_DIR}/../.." && pwd)"
fi
export GRIDBOOK_SOURCE_ROOT

if [ ! -f "${GRIDBOOK_SOURCE_ROOT}/gridbook/__init__.py" ]; then
  echo "GRIDBOOK_SOURCE_ROOT is not a Gridbook source tree: ${GRIDBOOK_SOURCE_ROOT}" >&2
  exit 2
fi

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
