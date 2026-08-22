#!/bin/bash
# Race for an idle-GPU window against sibling measurement containers, then run
# the tile-m sweep holding BOTH canonical GPU locks exclusively for the whole
# docker job. Retries while the sweep's own guard refuses (exit 2), capped at
# 3 attempts; exit 0 on success and rc==3 (structural blocker) propagates
# immediately. Args: LOG, then any args are passed through to the bench.
#
# PRISMAQUANT_SWEEP_MOUNT overrides the host directory bind-mounted at
# /tmp/ext-rho (default /tmp/ext-rho). The cell checkpoint is passed as
# --json /tmp/ext-rho/rho_sweep_cells.json with PRISMAQUANT_SWEEP_STATE set
# to match, so cells survive the --rm container and --resume works across
# restarts.
set -u
LOG=$1; shift
MOUNT=${PRISMAQUANT_SWEEP_MOUNT:-/tmp/ext-rho}
mkdir -p "$MOUNT"

while true; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$apps" = "0" ]; then
    sleep 20
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [ "$apps" = "0" ]; then break; fi
  fi
  sleep 45
done
echo "=== window found $(date) ==="

attempt=0
while true; do
  attempt=$((attempt + 1))
  # Both locks in this fixed order, held until the docker job returns.
  (
    flock 9
    flock 8
    docker run --rm --gpus all \
      -v /home/rob/gridbook:/src \
      -v /home/rob/prismaquant:/prismaquant:ro \
      -v "$MOUNT:/tmp/ext-rho" \
      -w /src -e PRISMAQUANT_CB_EXT_DIR=/tmp/ext-rho \
      -e PRISMAQUANT_SWEEP_STATE=/tmp/ext-rho/rho_sweep_cells.json \
      --entrypoint /bin/bash gridbook:0.8.11-clean-187c721 -lc '
pip uninstall -y gridbook -q 2>/dev/null
pip install pytest -q 2>/dev/null 1>&2
export PYTHONPATH=/src:/prismaquant
python3 scripts/bench_grouped_tile_m_sweep.py --json /tmp/ext-rho/rho_sweep_cells.json "$@"
' _ "$@" >> "$LOG" 2>&1
  ) 9>/home/rob/dq-runs/gpu.lock 8</tmp/claude-1000/gpu-bench.lock
  rc=$?
  echo "=== attempt $attempt exit $rc at $(date) ===" >> "$LOG"
  [ "$rc" = "0" ] && exit 0
  if [ "$rc" = "3" ]; then exit 3; fi   # structural blocker, not contention
  if [ "$attempt" -ge 3 ]; then
    msg="race_run_guarded: giving up after $attempt attempts (last rc=$rc)"
    echo "$msg at $(date)" >> "$LOG"
    echo "$msg" >&2
    exit "$rc"
  fi
  sleep 120
done
