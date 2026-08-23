# Math review — 2026-08-21

Date: 2026-08-21
Branch: `perf/kernel-eval-2026-08-21` (at `master` @ `ac39366`; prismaquant companion at `audit/math-reunderwrite-2026-08-21`)
Status: COMPLETE (byte-law audit, lemma verifications, new proofs; kernel landings B1/B7, with B2 proposed and REJECTED on measurement)
Scope: the mathematical content of this repository — byte laws and their cross-repo agreement, decode-window arithmetic, dispatch-threshold derivations, staging correctness preconditions — plus a performance re-evaluation of the decode/prefill kernels.

Method: independent numeric verification via scratch scripts outside the repo (written under the volatile `/tmp/opencode/{matheval,reports}/`; the 2026-08-21 22:30 snapshot lives at `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/{matheval,reports}/`), first-principles derivations with named numerical anchors, and the repo's own bit-exactness suites run in `gridbook:0.8.11-clean-187c721` on the GB10.

---

## §1 Byte-law triple agreement — VERIFIED

Field-by-field comparison across (a) producer `prismaquant/cb_layout.py`, (b) packaged `gridbook/runtime_contract.json`, (c) materialized copy `gridbook_runtime_contract.0.8.11.json`:

- Packing: VEC_DIM=8, SUPERBLOCK=256, CODEWORDS_PER_SUPERBLOCK=32, INDEX_BYTES_PER_K=4, `lsb_first`, `ceil_first` — identical in all three; contracts md5-identical to each other.
- Layout: current v2 / supported [1,2]; scale planes fp4-v1:16B, fp4-two_tier:9B, fp8:0B.
- type_size law per 256-weight superblock: `(fp4,v1)→4k+16`, `(fp4,v2,two_tier)→4k+9`, `(fp8,·)→4k` — holds on every rung through both implementations of the law.
- Formats: NVFP4_CB_K{k} k∈[12,24] n_sub=2 product; FP8_CB_K{k} k∈[28,48] n_sub=4 product; signed family contract-only (producer deleted it 2026-08-17).
- Compose table `(E,c) ↦ T[c]·2^(E−127)` bitwise-equal across struct-level fp32 recompute, gridbook codec, and producer tables on the legal mask (252/4096 pairs).

## §2 Decode-window arithmetic — VERIFIED + CHARACTERIZATION THEOREMS

Codeword v occupies bits `[v·k, (v+1)·k)` LSB-first; the decoder reads an aligned u32 word pair at the containing boundary (`rem = (v·k) mod 32 ∈ [0,32)`):

- Containment always holds for k∈[12,24]: max rem+k = 54 < 64 ⇒ two words suffice universally; the third (spill) word is needed exactly when `rem + k > 64`.
- Spill predicate exactness: enumeration over all 21 rungs × 32 codewords × 8 phases matches the predicate bit-exactly (5376 cases); closed forms locate the only live shipped cell: `k > 32 + gcd(k,32)` (u32 windows) / `k > 32 + gcd(k,8)` (u64 bursts) ⇒ persistent-B k=44 is the sole rung whose hot loop must read the third word — which is precisely why the grouped kernel's predicated load is bit-neutral and profitable elsewhere.
- Independent bit-by-bit decoder ≡ `cb_torch_reference.extract_codewords` at k∈{13,17,20,24}; ceil-first sub-index consumption (sub0 at offset 0) agrees across producer packer, oracle, and CUDA SubSplit.

## §3 ρ-crossover threshold — TIGHTNESS PROVEN (upgrades the PROPOSAL-DATA status)

The wide-tile selector fires at ρ = P/E > 512 (`moe_routing.py:238-263, GROUPED_WIDE_TILE_MIN_ROWS_PER_EXPERT`). This review establishes:

- Padding lemma and tile identity `B(128) = 2·B(256) − q` verified exactly on 20k random histograms, with the premise correction that **q counts experts whose residue c_e mod 256 lies in [1,128]** (not c_e ≡ 0 mod 256; counterexamples recorded).
- Residue-only reduction: the TileM-256 advantage is `Δ = x(Σa_e + n_hi) − 128q` — histogram-blind in P, dependent only on residues.
- The analytic sufficient family `ρ > 128(1+256/x)` is **tight**, not conservative: it is the exact price of histogram-blindness (Thms 4–5). Under measured x∈[75,259] the band is [254.5, 564.9]; the shipped 512 certifies the win only for x̂ ≥ 85.3.
- **Non-monotonicity finding**: the advantage profile is not monotone in ρ — at x=75, uniform routing LOSES again over ρ∈(~261,~373) after first winning near ~133. Consequence for calibration: any routed sweep validating the constant must sweep past first crossings; a three-item acceptance test is specified (Thm 7).
- Convention gap noted: the comment's `(B128−q)/q > 256/x` form is undefined at q=0 where the direct model gives TileM=256 an unconditional win — no effect on thresholds.

Status change recommended for ROADMAP K0.4's residual: from "proposal data until a routed sweep pins it" to "tight bound derived; sweep must clear Thm 7's acceptance test."

## §4 Empty-expert cost — VERIFIED NEGLIGIBLE

Work unit (expert, N-tile) pays two int32 loads before early-out (:667-684). Exact waste: `2·E_empty·⌈N/TN⌉` loads plus short CTA slots; at DSv4-class shapes that is ~161 B ≈ 0.59 ns against ms-scale cells ⇒ shape-pure (graph-capturable) grids pay ≤ ~0.1% worst-case for routing-obliviousness; unmeasurable for ρ ≳ 12. The design trade (no host reads, no device-side grid sizing) is thereby priced.

## §5 Staging vectorization precondition theorem (feeds proposed B2)

Theorem (staging byte-neutrality): uchar4/u32 vectorized copies are byte-identical to the byte-granular loop for type_size ≡ 0 (mod 4), and for odd type_size = 4k+9 provided (i) misaligned source phases are absorbed by align-down + funnel-shift on the LOAD side while dst positions stay absolute, and (ii) the seam zero-fill of [type_size, ts_pad) is preserved — negative test shows drift appears exactly when ts mod 4 ≠ 0 without the seam discipline. Proposed implementation B2 states all five preconditions in-source (P1–P5) and its decode-probe torch.equal gates ran green, but it was REJECTED on performance (+7…+11% whole-operator at k=12, independent A/B); the theorem stands independently of that implementation.

## §6 Kernel performance re-evaluation — landings and evidence

All changes bit-gated; benchmarks in-container on GB10 (median-of-N CUDA-event timing).

### B1 — dense FP4-v2 GEMV round-2 backport (`PRISMAQUANT_CB_FP4V2_DENSE_R2`, **default ON since 0.8.13**)

Ports from the grouped kernel: predicated spill-word read, uint2 packed codebook gathers (per-element bf16→f32 chains unchanged — packed conversion stays unported per its recorded regression), aligned-down u64 burst staging with last-superblock fallback and a slot-bound guard. Bit-exact both modes (92 legacy gates + 15 new dual-mode tests). Benchmark: R2 wins all 32 measured points; gains grow with k (−0.9%…−13.6% @k16, −7.2%…−20.4% @k20), consistent with the compute-bound ncu profile. Promotion path: **flipped to default ON in 0.8.13 (2026-08-23)** on real-shape evidence, NOT on the originally-recorded NATIVE-PARITY served protocol, which was never run (no wrapper exists). Re-measured on the Qwen3.8-27B CB gold artifact's four real fp4 rungs at true (N,K) across M∈{1,2,4,8,16}: all 40 (shape × M) points bit-identical and faster, worst per-M aggregate −3.44%, best −10.17%. Bit-identity removes the quality axis, so the flip is perf-only and no artifact needs re-validation. See CHANGELOG 0.8.13 for the full deviation statement.

### B2 — persistent-B packed-staging vectorization (proposed; REJECTED)

Byte-granular staging → u32 interior with funnel-shift edges, whole-slot zeroing helper. Bit-neutrality per §5; decode-probe suites green. REJECTED on independent A/B measurement: +7…+11% whole-operator at the shipped DSv4-dominant k=12, and the `__noinline__` variant slower still (+14%). The implementation's recorded rationale — `__noinline__` scoping protects register allocation because inlining measured +26 registers/thread ⇒ ~23% whole-operator regression — is contradicted by `cuobjdump --dump-resource-usage` of the built binaries: the inlined build allocates FEWER registers than the byte-loop baseline on every fp4 instantiation, including the `<128,64,8>` tile that actually runs at DSV4 shapes (112 → 80; `<128,32,4>` 128 → 120, `<64,64,4>` 126 → 72), the noinline build more (128/125), and both inline choices lose to baseline. The mechanism (SASS, consensus review): mainloop unchanged; the byte loop's compiler-unrolled 8-wide independent `LDG.U8` burst became five `unroll 1` loops per row with `VOTE.ANY`/`WARPSYNC`/`SHFL.DOWN` gadgets — a dependent load→store chain per word. The staging-share isolation cell remains unmeasured; the staging theorem (§5) is unaffected.

### B7 — per-chunk swizzle-group packing (= ROADMAP K1.5, landed unconditional)

Tile-order win restored for multi-chunk layers by applying pack_expert_blocks within each chunk's own expert range. Output equality argued and verified: chunk boundaries split the expert dimension only; per-expert operands and FP32 reduction order are launch-membership-independent; end-to-end torch.equal held at top_k=1, atomicAdd-combine envelope respected at top_k>1 (measured ≤4.7e-3 vs documented 2.1e-2 bound). Graph capture preserved (packing derives from the one pre-existing block_offsets read; capture refusal asserted for the new spelling). Isolated stage-one gather: packed/natural = 0.890/0.897 (−10.3..11.0%) on straddling segments; inert (1.001×) on uniform-router control — the win appears exactly when segments straddle, as designed.

### K1.3 — large-M dense prefill (analysis; no code)

Roofline memo conclusions adopted: decode repetition follows TM (not CTA ownership), so the MoE persistent-B lesson transfers as cheap-decoder + occupancy floor rather than decode-once; transient path's honest payoff band for a perfect weight-read-once kernel is TTFT ~1.02×, realistic candidate band 1.05–1.17× vs today's 1.44×; conditional GO requires TM=256/TN=32 (~40 KB, e4m3 activation contract) clearing break-even at M≈1400; three sub-day gates (G0 traffic profile, G1 decode-pass microbench, G2 occupancy, G3 L2/A-restream) must pass before any `.cu`. Persistent-N remains quarantined; overlap schedules remain dead.

## §7 Findings and actions

| # | Finding | Severity | Action |
|---|---|---|---|
| G1 | ρ-threshold tightness + non-monotone profile | INFO/CALIBRATION | Documented here; ROADMAP note updated; sweep acceptance test specified |
| G2 | q-count premise correction (residues [1,128]) | INFO | Prover record; comment-level fix available if desired |
| G3 | Spill-word live set = {persistent-B k44} | INFO | Justifies predicated-load optimizations; documented |
| G4 | Empty-expert overhead negligible | INFO | Design trade priced; no action |
| G5 | Dense-fp4v2 structural lag vs grouped kernel | PERF | FIXED by B1; **default ON in 0.8.13** (2026-08-23 real-shape M-sweep) |
| G6 | Multi-chunk layers missing tile-order win | PERF | FIXED by B7 (unconditional, bit-identical outputs) |
| G7 | Byte-granular staging instruction waste | PERF | PROPOSED fix B2 REJECTED on measurement (k=12 +7…11%); item open |

Verification artifacts: in-repo: `docs/audits/math_review_2026-08-21_proofs.md` (full proofs incl. ρ-tightness Thms 4–7, window lemma, staging theorem); working reports snapshotted at `/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/reports/` (the `/tmp/opencode/reports/` originals are volatile).
