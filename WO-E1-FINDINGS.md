# WO-E1 Findings — Route telemetry for the Gridbook trellis lanes

## Summary

Implemented `emit_route`/`read_route` telemetry in both trellis lanes
(`gridbook/trellis_e2m1_lane.py:apply`, `gridbook/trellis_e4m3_lane.py:apply`)
at their dispatch sites, using the existing `gridbook/nvfp4_activation_contract.py:150`
mechanism. Every forward emits exactly one `state: served` record with
`kind: dense`, `policy: <FAMILY>:<MODE>`, `symbol: torch._scaled_mm`,
`shape: M:N:K`, and `contract` equal to `layer.gridbook_activation_contract`
(which IS `ACTIVATION_CONTRACT`). No record is emitted before dispatch.
Extended `tests/test_trellis_e2m1_lane.py` and `tests/test_trellis_e4m3_lane.py`
with four green-on-GPU checks per lane; updated
`docs/TRELLIS-R256-RESEARCH.md` with the new telemetry section.

## Contradictions and limitations (not quiet guesses)

### 1. Regime (decode vs batch) — the contract's two cells vs the lane's one kernel

`gridbook/runtime_contract.json:216` publishes **two** `lane_eligibility`
cells per trellis family — `regime: decode` and `regime: batch` — each naming
the same `activation_contract` and `rungs_q256`. The lane, however, has a
single code path: `torch._scaled_mm` with the same operands for every `M`
(`gridbook/trellis_e4m3_lane.py:287`, `gridbook/trellis_e2m1_lane.py:307`).
There is no `M` threshold that changes the kernel or the activation contract,
unlike CB dense (`MOE_PREFILL_M_THRESHOLD = 16`) or MoE persistent-B.

**Finding:** the record **does not** carry a separate `decode`/`batch` label.
`shape` carries `M`, so a verifier can infer a regime if it has a threshold,
but the lane itself does not distinguish them and the record does not pretend
to. A consumer must not read one code path as two attestations. This is
documented in the lane comments and in `docs/TRELLIS-R256-RESEARCH.md:route telemetry`
and is the honest answer to WO-E1's "if the lane can distinguish them at
dispatch — if it cannot, say so".

Guessing a threshold (e.g. `M <= 8` decode) would be a second source of truth
for a distinction the hardware/KERNEL does not make, exactly the heuristic WO-E1
warns against.

### 2. The no-fallback argument — sound, but not substituted

Both lanes have no Triton or CB-symbol fallback: the only failure mode is
`NativeKernelUnavailableError` ("There is no Triton or CB-symbol fallback" is
documented in `trellis_e4m3_lane`/`e2m1_lane` module docstrings and the
`config.py:_build_trellis_method` refusal sites, and `apply` contains no
fallback branch). One could argue lane construction alone implies the native
route.

**Finding:** the no-fallback property **is** machine-checkable in-package
(inspect `apply` — no alternative kernel, and `create_weights` refuses TP>1) and
is recorded here, but it is **not** substituted for the records, per WO-E1's
rule. Served evidence is the `read_route` record; construction is the gate
that made it reachable.

### 3. Activation-contract sourcing — no second literal

- E2M1: `ACTIVATION_CONTRACT = EXECUTION_CONTRACT` where `EXECUTION_CONTRACT`
  is imported from `nvfp4_activation_contract` (`trellis_e2m1_lane.py:97`);
  the lane stamps `layer.gridbook_activation_contract` from that same constant
  and the telemetry reads `layer.gridbook_activation_contract`.
- E4M3: `ACTIVATION_CONTRACT = "fp8_per_token_dynamic"`
  (`trellis_e4m3_lane.py:106`); the telemetry also reads
  `layer.gridbook_activation_contract`.

Both equal the strings the packaged `runtime_contract.json` publishes
(`runtime_contract.py:TRELLIS_ACTIVATION_CONTRACTS`, validated in
`tests/test_runtime_contract_trellis.py:252`). No second literal is typed at
the telemetry site; the test
`test_route_contract_matches_packaged_runtime_contract` reads the packaged file
via `load_runtime_contract` and fails closed on a mismatch (a fixture with a
different string fails the `== expected` assertion).

### 4. Residency mode — declared, not heuristic

Residency is a declared mode (`GRIDBOOK_TRELLIS_E*_MODE` latched via
`lane_select.latched_bool`, no default — `trellis_e2m1_lane.py:111`,
`trellis_e4m3_lane.py:115`). `create_weights` stamps `layer.trellis_mode` and
`layer.trellis_family`; `apply` publishes `policy=f"{FAMILY}:{mode}"`. The
record therefore survives the `resident`/`streamed` distinction and a streamed
forward is never recorded as resident (gated in tests).

### 5. What the record does NOT attest

- **Quality, speed, or TP>1** — the record states which route executed, never
  how good or fast it is, and no `lane_eligibility` cell or `formats` row is
  added here (per WO-E1 docs rule).
- **Encoding quality** — the smoke checkpoint remains self-consistent, random
  weights.
- **Routed MoE** — trellis lanes are `LinearBase`-only (`config.py:1431`); no
  routed `structure` is reachable.

## Verification

- Trellis lane tests: `tests/test_trellis_e2m1_lane.py` and
  `tests/test_trellis_e4m3_lane.py` each now carry the four WO-E1 checks
  (`test_route_record_emitted_per_dispatch`,
  `test_route_contract_matches_packaged_runtime_contract`,
  `test_route_residency_survives`, `test_no_route_when_not_exercised`),
  skipped cleanly with `needs a CUDA device` when CUDA is unavailable, matching
  the existing `requires_cuda` style.
- Full suite and GPU run are reported in the commit message.

## Scratch use

Temporary files, if any, were placed under
`/home/rob/gb-e1-trellis-emit-route/scratch/` per worktree rules; `/tmp` was
not used.
