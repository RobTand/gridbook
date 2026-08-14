# MXFP4-CB Research Prototype (`research/mxfp4_cb/`)

**Status: research prototype, not production Gridbook.** CPU-only, pure Python+torch, deliberately small and falsifiable. Nothing here is mainlined; it does not touch `gridbook/`, runtime registration, CUDA sources, or existing tests.

## What this prototypes

One plausible OCP-MXFP4-targeted codebook representation that mirrors Gridbook's superblock codec but targets the OCP MXFP4 physical grid:

* **Element grid**: E2M1 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` (identical to NVFP4; bias 1).
* **Scale grid**: E8M0 / UE8M0 per **32** weights (MX block), power-of-two `2^(E-127)`, bias 127, byte `0xFF` reserved as NaN. One scale per MX block, 8 scales per 256-weight superblock.
* **Superblock**: 256 weights = 32 codewords × 8 dims. `in_features % 256 == 0` required (same as Gridbook SPEC §0).
* **Index stream**: `4*k` bytes per SB, 32 `k`-bit codewords LSB-first (SPEC §1.1). `bit_split(k, n_sub)` larger parts first (ceil-first).
* **Scale plane**: 8 bytes of E8M0, immediately after the index stream.
* **Wire**: per SB `type_size = 4*k + 8` bytes, `effective_bpw = (type_size*8)/256 = k/8 + 0.25`. Codebook amortisation negligible.

This fills the gap left by the shipped Gridbook formats (`NVFP4_CB_K*` with group-16 E4M3 / two-tier; `FP8_CB_K*` with per-channel fp32) which have no MXFP4 block-32 / E8M0 variant.

## Wire / BPW accounting

| k | index B | scale B | type_size B | bpw | saving vs direct MXFP4 (4.25 bpw) |
|---|---|---|---|---|---|
|12|48|8|56|1.750|2.43×|
|13|52|8|60|1.875|2.27×|
|16|64|8|72|2.250|1.89×|
|20|80|8|88|2.750|1.55×|
|24|96|8|104|3.250|1.31×|

Direct MXFP4 = 4 bits/weight + 8 bits per 32 = 0.25 bpw → 4.25 bpw total. Codebook at `k=16` is 2.25 bpw, `k=12` is 1.75 bpw.

For comparison: NVFP4-CB v2 is `k/8 + 0.28125`; v1 is `k/8 + 0.50`. MXFP4-CB trades a slightly leaner scale plane (8 B vs 9/16 B) for power-of-two-only scales, which costs fine-grained range control.

## Modules

* `format.py` — `Mxfp4CbFormat` dataclass/schema, `type_size`, `bpw`, `bit_split`, validation.
* `e2m1.py` — reference E2M1 nibble encode/decode (RTN ties-to-even) + E8M0 per-32 encode/decode via `frexp` (exact, no `log2` rounding defect), packed-nibble helpers, direct MXFP4 `quantize/dequant` baseline.
* `codec.py` — Gridbook-style superblock `pack_indices`/`unpack_indices` (LSB-first) + `encode_mxfp4_cb` / `decode_mxfp4_cb` reference paths with shape/dtype/invariant validation.
* `baseline.py` — direct MXFP4 (no codebook) encode/decode and `reconstruction_metrics`.
* `cross_platform.py` — the stronger cross-platform hypothesis experiment: one canonical FP32 codebook → deterministic projection to NVFP4 (E2M1+E4M3) and MXFP4 (E2M1+E8M0), keep same indices, report assignment stability + reconstruction/output divergence, propose minimum metadata.
* `tests/` — CPU pytest suite (see below).

## Assumptions (what the prototype *asserts* without proof)

1. **E2M1 grid is exactly NVFP4's** — OCP MXFP4 E2M1 uses the same 8 magnitudes and bias 1. The spec is explicit; if an implementation uses a different E2M1 variant, the projection diverges.
2. **E8M0 semantics**: byte 0 = `2^-127`, `0xFF` = NaN (OCP), encoder clamps overflow to `0xFE`. Zero groups map to byte 0. This matches `gridbook/mxfp8.py`'s UE8M0 convention; OCP's definition is assumed identical.
3. **Scale rule**: smallest power-of-two `scale` with `amax/scale ≤ 6` per 32-block, `frexp`-computed. Zero-block → min scale. This is the natural MX adaptation of the MXFP8 rule; a production encoder could choose `floor` or `ceil` differently and would shift bytes.
4. **Codebook is E2M1-valued** — codebook entries themselves are on the E2M1 grid (so the MXFP4 decode is `e2m1_value * 2^exp`). A learned unconstrained FP32 codebook would need an extra quant step.
5. **Superblock & vec_dim identical to Gridbook** — reusing 256 / 8 / 32 lets us keep the same bitstream and TMA-friendly byte boundaries (`32*k` bits = `4*k` bytes). Different choices would change `type_size` arithmetic.
6. **Scale plane is flat 8 bytes**, not nibble-packed or compressed. Two-tier-style sub-table compression is not applied; it could be but is out of scope.
7. **No fused-kernel constraints modeled** — shared-memory LUT budgets, TMA alignment, smem swizzles that bound `k` in production are not modeled beyond the `k≤24` sanity cap.

## What would need a producer trainer

* **Learned E2M1 codebooks**: product `n_sub=2` tables `(2^ceil(k/2),4)` + `(2^floor(k/2),4)` per role, trained under the E8M0-per-32 scale law (not per-16 E4M3). The current lattice synthesis in `cross_platform.py` / tests is random; a real producer would k-means / VQ-train on scaled weight blocks with outlier-aware grouping and tie-breaking.
* **Scale-aware assignment**: per-vector assignment must be done in the scaled domain (`vec / scale`), not raw weight, and must be deterministic across replicas (argmin first-occurrence). The trainer must also decide the zero-block and no-clip policies.
* **Calibration vs direct export**: NVFP4-CB and MXFP4-CB need separate calibration passes if indices are platform-specific (see cross-platform result); a single calibration cannot serve both scale grids optimally.
* **Per-expert / per-layer rung selection**: which layers get which `k` under a global `bpw` budget (Gridbook's `config_groups` vocabulary would need a new family like `MXFP4_CB_K*` and `scale_coding: "e8m0_per32"`).
* **Activation contract**: this prototype is weight-only; a W4A4 contract for MXFP4-CB would need a separate static/dynamic activation quantizer (not modeled).
* **Digest / provenance**: `codebook_sha256` binding for the new family (same construction as Gridbook SPEC §4.1).

## Next HIP kernel boundary (exact)

The prototype stops **before** any HIP/CUDA kernel; the next boundary is precisely:

**Function:** `hip_mxfp4_cb_gemv_composed` (or `cb_expand_mx`) — the analogue of `gridbook/csrc/cb_gemv.cu` / `cb_gemv_v2.cu` and `gridbook/expand.py:expand_fp4_v2_to_weight`.

**Inputs (host-visible):**
```
qweight: uint8 [rows, (K/256)*type_size]   // raw (not padded) packed plane
cb_flat: bfloat16 [(2^w0*4 + 2^w1*4)]    // flat product table, BF16 exact
cb_row_offset: int32 [rows]                // per-row table base (for fused roles)
compose_or_scales: —                       // NONE: E8M0 scales are inline in qweight, no separate compose table
N, K, k_bits, n_sub, type_size             // ints; type_size == 4*k+8
```

**Behavior:** for each `(row, sb, vec)` gather the `k`-bit codeword from the `4*k` index bytes (LSB-first 8-byte window, same as `extract_codewords` in `tests/cb_torch_reference.py`), split into sub-indices via `bit_split(k,n_sub)` (`sub0 = code & ((1<<w0)-1)`, `sub1 = (code>>w0) & ...`), gather two 4-vectors from `cb_flat` (or 8-signs + magnitude for `signed` mode), compose the 8-dim codeword, multiply each element `j` by `E8M0_scale[ (vec*8+j)/32 ]` where the 8 scales are the 8 bytes after the index stream of that SB (`2^(byte-127)`, byte `0xFF` → fault), accumulate or expand to BF16 transient `[N,K]` (or directly to a GEMV output).

**Required checks (mirroring `expand.py`):** `K%256==0`, `type_size==4*k+8`, `cb_flat` contiguous BF16, `qweight` uint8 stride 1, `cb_row_offset` int32, all colocated, `k` in supported set. No Triton lane — fail-closed if native kernel unavailable.

**What this prototype already proves for that kernel:** exact `pack/unpack` bitstream, `e2m1`/`e8m0` byte semantics, per-32 scale replication, and deterministic assignment — all exercised on CPU against the `cb_torch_reference.py` oracle pattern.

## Tests and how to run

```bash
python -m pytest research/mxfp4_cb/tests -v
# with output:
python -m pytest research/mxfp4_cb/tests -v -s
# cross-platform only:
python -m pytest research/mxfp4_cb/tests/test_cross_platform.py -v -s
```

Coverage:

* **Exact wire roundtrip** — `pack_indices` ↔ `unpack_indices` for multiple `k`, byte-boundary exact.
* **Special/edge values** — all 16 E2M1 codes round-trip, ±0, NaN/Inf rejection, E8M0 zero-block and NaN-reserved `0xFF`, power-of-two thresholds, `log2` defect via `frexp`.
* **Malformed input** — bad `K`, rank, dtype, `qweight` short row, codebook shape, format invariants.
* **Scale grouping** — per-32 grouping covers superblock, uniform SB single byte, successive scales ratio 2, zero-block isolation.
* **Deterministic reconstruction** — `encode`→`decode` stable, re-encode identical bytes, deterministic across seeds.
* **Baseline comparison** — synthetic expert-like tensors (heavy-tail + outliers), `k=12/16/20`: report `rel_l2`, `max_abs`, `mse` for codebook vs direct MXFP4, plus `X·W` output error; assert compression ratio and that low-`k` codebook is lossier than direct (honest).
* **Cross-platform hypothesis** — canonical FP32 codebook → `snap_to_e2m1` projection to both NVFP4 (per-16 E4M3) and MXFP4 (per-32 E8M0), keep same indices, measure stability (<95%), cross weight `rel_l2>0.1`, output `rel_l2>0.05` — strong hypothesis **fails**.

## Honest assessment

The prototype **does not show that MXFP4-CB is a win on quality** at low `k` — it shows it is a win on **compression**:

* At `k=12` (1.75 bpw vs direct 4.25 bpw, 2.43× smaller) the synthetic up-projection error is materially higher than direct MXFP4, as expected. At `k=20` (2.75 bpw) the gap narrows. No training was done; a learned codebook would close some distance but cannot recover the information discarded by power-of-two-only scales on outlier channels — the E8M0 grid is coarser than E4M3 in the critical 0.5–6 range for large-amplitude blocks.

* The cross-platform experiment **falsifies the strong hypothesis**: sharing identical `k`-bit indices between NVFP4 (16-element E4M3 scales) and MXFP4 (32-element E8M0 power-of-two scales) is incoherent. Measured on `k=16, 4×512` synthetic expert tensors:
  * stability ~60–85% (depends on seed/k), instability >2% always,
  * reusing one platform's indices under the other's scales gives `weight rel_l2 >0.1` and `output rel_l2 >0.05`,
  * `optimal-vs-optimal` weight divergence `>0.05` even when each uses its own optimal indices.

  The root cause is scale grouping (`32` vs `16`) and scale quantizer (power-of-two vs E4M3): the same logical vector falls into different scaled nearest-neighbour decisions.

* **Minimum metadata to restore coherence:** per-platform index streams (and scale planes). A single per-superblock flag/bit is insufficient — you need the full `4*k`-byte index stream **plus** its native scale plane per target. Practically: ship per-platform `cb_qweight` shards (dual streams `4k+16` for NVFP4 v1 + `4k+8` for MXFP4) or ship per-platform artifacts. Storing only a delta/correction on top of a shared stream would be larger than the second stream for the measured instability rate.

* **HIP kernel path**: the prototype's wire is ready for a HIP kernel that reads the same `4*k`-byte LSB-first stream the NVFP4 kernels read; the kernel boundary above is minimal and matches `gridbook/csrc/cb_gemv*` conventions. Performance will be gating: `k>16` needs product/signed tricks (not flat tables) to stay within ~100 KB smem, same note as Gridbook SPEC §8.

If the design is pursued, the next step is a producer trainer that **trains separate codebooks and assignments per scale family** and emits per-platform `quant_config.json` + `cb_codebooks.pqcb` + provenance, not a universal canonical artifact.

