# New mathematics for the gridbook kernels

**Branch:** `perf/kernel-eval-2026-08-21` (working tree; the B2 staging
vectorization and its gate test are uncommitted at the time of writing).
**Scope:** research + proofs only — no repository file is modified by this
document. **Verification:** pure python/numpy integer simulation under
`/tmp/opencode/matheval/gridbook_newmath/` (volatile; snapshot at
`/home/rob/dq-runs/review-watch-2026-08-21/opencode-evidence/matheval/gridbook_newmath/`) (`part1_rho.py`,
`part1_profile.py`, `part2_window.py`, `part3_empty.py`,
`part4_staging.py`); every lemma below that admits an exhaustive or
Monte-Carlo check was checked, and the check is named where it is used.
Nothing here replaces a GPU measurement; where a claim is measurement-shaped
it is marked **PROPOSAL DATA** in the house style.

Two premises of the tasking needed correction before the proofs could be
stated honestly; both corrections are load-bearing:

1. **What `q` counts.** In `B(128) = 2·B(256) − q` (gridbook/moe_routing.py:243),
   `q` is **not** "the number of experts with `c_e ≡ 0 (mod 256)`". It is the
   number of experts whose routed count has *residue in [1, 128]*,
   `q = #{e : c_e mod 256 ∈ [1,128]}`. Lemma 2 proves the identity for this
   `q`; §1.1.3 gives explicit counterexamples (`c = 100`, `c = 128`,
   `c = 300`) showing any other reading breaks it.
2. **Which window the fp4-v2 decoder uses.** The extraction window is a
   **u32-aligned pair of words** (`rem = bitpos mod 32 ∈ [0,32)`), not an
   8-byte-aligned u64 window (`rem ∈ [0,64)`). The aligned-u64 machinery in
   `cb_gemv.cu:1354-1367` belongs to the gmem→smem **staging burst**, not to
   the decode window. This distinction is exactly what makes the third word
   dead for every shipped rung (§2).

---

## Part 1 — the TileM=256 crossover: exact padding algebra, worst-case tightness, and what the simulation actually measured

Setting: a grouped MoE layer routes `P = tokens × top_k` pairs over `E`
experts with per-expert counts `(c_1, …, c_E)`, `Σ c_e = P`. A TileM=`t`
launch pads each expert's segment up to a multiple of `t` rows and launches
one CTA per `(M-tile, N-tile)`.

### 1.1 The cost model, reconstructed exactly

From moe_routing.py:238-249 and docs/KERNELS.md ("Why ρ"), the model is:

- Per CTA at tile width `t`: the packed weight tile `TileN × K` is **decoded
  once**, then the mainloop issues `t · TileN · K` MACs against `t` padded
  activation rows.
- `d` = per-M-tile cost of one decode of a `TileN × K` packed tile
  (codebook gathers, scale compose, per-tile fixed overhead);
  `m` = per-padded-row cost of the MMA mainloop plus the row's A-tile
  traffic, i.e. the cost of `TileN · K` MAC-equivalents.
  Both are per `(M-tile, N-tile)` CTA quantities, so a launch with
  `N/TileN` N-tiles costs

      T(t) ∝ B(t) · (d + t·m),     B(t) = Σ_e ceil(c_e / t),

  the M-tile count being `B(t)` because padding makes expert `e` occupy
  `ceil(c_e/t)` M-tiles. The `N/TileN` factor is common to both arms of any
  comparison and cancels.
- `x := d/m` is the **decode:MMA ratio expressed in rows**: decoding a
  `TileN×K` tile once costs as much as `x` rows of mainloop. Because both
  the decode work and the per-row MMA work scale with `TileN·K`, `x`
  depends on neither `N` nor `K` — which is why the w13 and w2 projections
  give the *same* crossover condition even though their shapes differ.
- The selector reads only host-known integers (`P = tokens·top_k` from
  tensor metadata, `E`, tile lists, SM count), never the histogram — that
  is what makes it capture-safe (docs/KERNELS.md, "Graph safety"). Every
  rule below must therefore be a function of `(P, E, x)` alone; §1.2 shows
  precisely what that constraint costs.

#### 1.1.1 Lemma 1 (exact padding lemma)

*For every integer `c ≥ 0`, with `r = c mod 256`:*

    pad₂₅₆(c) − pad₁₂₈(c) = 128 · 𝟙[r ∈ [1, 128]],

*where `pad_t(c) = t·ceil(c/t)`.*

*Proof.* Write `c = 256a + r`. If `r = 0`: both paddings are exact,
difference 0. If `1 ≤ r ≤ 128`: `pad₁₂₈(c) = 256a + 128·ceil(r/128) =
256a + 128` while `pad₂₅₆(c) = 256(a+1)`; difference `128`. If
`129 ≤ r ≤ 255`: `pad₁₂₈(c) = 256a + 128·ceil(r/128) = 256a + 256` and
`pad₂₅₆(c) = 256(a+1)`; difference 0. ∎

*Check:* exhaustive over `c ∈ [0, 20000)` (`part1_rho.py::check_padding_lemma`).

#### 1.1.2 Lemma 2 (the tile identity, and what q counts)

*For every histogram,* `B(128) = 2·B(256) − q` *where*
`q = #{e : c_e mod 256 ∈ [1,128]}`. *The identity is exact — equality, not
an inequality — for every histogram, provided `q` counts that residue
class. Under the misreading* `q₀ = #{e : c_e ≡ 0 (mod 256)}` *it fails in
general (counterexamples: single-expert histograms `c = 100`, `c = 128`,
`c = 300`).*

*Proof.* Divide Lemma 1 by 256 after summing over experts:
`Σ_e pad₂₅₆(c_e)/256 − Σ_e pad₁₂₈(c_e)/256 = (128/256)·q`, i.e.
`B(256) − B(128)/2 = q/2`. ∎

*Check:* 20 000 random, all-residues-128, multiples-of-256, and short
histograms (`part1_rho.py::check_identity`); the `q₀` counterexamples are
asserted there too.

#### 1.1.3 Proposition 3 (the exact crossover condition is a condition on residues only)

*Define* `Δ := cost(128) − cost(256) = B(128)(d + 128m) − B(256)(d + 256m)`.
*Then, with* `a_e = ⌊c_e/256⌋`, `r_e = c_e mod 256`,
`R = Σ_e r_e`, `n_hi = #{e : r_e ∈ [129,255]}`, `q = #{e : r_e ∈ [1,128]}`:

    Δ = x · (Σ_e a_e + n_hi) − 128·q = [ x·(P − R + 256·n_hi) − 32768·q ] / 256,

*(using `m` as the unit and `Σ_e a_e = (P − R)/256`). TileM=256 wins iff
`Δ ≥ 0`. So the verdict depends on the histogram only through the residue
statistics `(R, n_hi, q)` — how the whole-256 multiples are distributed
across experts is irrelevant.*

*Proof.* Substitute Lemma 2 into the definition and expand:
`Δ = (2B256 − q)(x + 128) − B256(x + 256) = x·B256 − q(x + 128)`. Since
`B256 = Σ(a_e + 𝟙[r_e ≥ 1])` and `q = Σ𝟙[r_e ∈ [1,128]]`, we get
`B256 − q = Σ a_e + n_hi`. ∎

*Check:* 20 000 histograms against direct cost evaluation, both forms
(`part1_rho.py::check_residue_condition`).

This recovers the shipped comment's phrasing: `Δ ≥ 0` rearranges to
`(B(128) − q)/q > 256/x` when strict (moe_routing.py:244), since
`B(128) − q = 2(B256 − q)`.

### 1.2 The host-knowable sufficient condition is valid AND tight

#### 1.2.1 Theorem 4 (sufficiency)

*Fix `x > 0`. If* `cost(256) > cost(128)` *for some histogram with mean*
`ρ = P/E`, *then* `ρ < 128·(1 + 256/x)`. *Contrapositive:*
`ρ ≥ 128(1 + 256/x)` ⟹ `cost(256) ≤ cost(128)` *for every histogram with
that mean.*

*Proof.* Assume 256 loses: `Δ < 0`, i.e. `x·B256 < q(x + 128)`.

(i) Since `q ≤ E`: `B256 < E(x+128)/x`.

(ii) Padding `= 256·B256 − P = Σ_{r_e ≥ 1}(256 − r_e) ≥ Σ_{q-experts}(256 −
r_e) ≥ 128q` (each `[1,128]`-residue expert pads by at least 128), so
`P = 256·B256 − padding ≤ 256·B256 − 128·xB256/(x+128) =
128·B256·(x+256)/(x+128)`.

(iii) Multiply (i) by the coefficient from (ii):
`P < 128·E(x+128)/x · (x+256)/(x+128) = 128E + 32768E/x`, i.e.
`ρ < 128 + 32768/x = 128(1 + 256/x)`. ∎

#### 1.2.2 Theorem 5 (tightness — the worst case is exactly `c_e ≡ 128 mod 256`)

*No smaller threshold works as a function of `(ρ, x)` alone: for every
`x > 0` and the adversarial family "every expert at residue 128"
(`c_e = 128 + 256·a_e`, quotients summing to `A = Σa_e`), 256 loses iff*

    x·A < 128·E   ⟺   ρ = 128 + 256·A/E < 128 + 32768/x.

*Hence histograms with `cost(256) > cost(128)` exist at every
`ρ < 128(1 + 256/x)` (within `256/E` discretization), and the supremum of
losing means equals the bound without attaining it.*

*Proof.* For the family: `q = E` (every residue is 128),
`B256 = E + A`, so `Δ = x(E + A) − E(x + 128) = xA − 128E`. ∎

*Check:* rational-arithmetic evaluation on both sides of the sup for
`x ∈ {75, 150, 259}` — losing up to `ρ = {564.875, 346.438, 254.500}`
against bounds `{564.907, 346.453, 254.517}`, gaps `≤ 256/E`
(`part1_rho.py::check_theorems_4_5`); plus 4 000-sample multinomial searches
at and above each bound find no losing histogram.

**Consequence (this is the sentence KERNELS.md is missing).** The bound is
not merely "valid"; it is the *exact price of histogram-blindness*. At
`ρ = 130` with `x = 259` there genuinely exists a capturable-consistent
histogram (all experts at `c ≡ 128`, quotient mass spread as in Theorem 5)
on which TileM=256 loses — uniform routing just never produces it. Any
capture-safe selector that may read only `ρ` must wait until
`ρ > 128(1 + 256/x)`; with the audited band `x ∈ [74.6, 259.3]` (Part 5)
that is `ρ* ∈ [254.4, 567.1]`, quoted as `[254, 565]` in moe_routing.py:252
after rounding `x` to `[75, 259]`. The shipped 512 lies inside the interval,
toward its conservative side — but note precisely (Lemma 17): a constant
certifies the whole audited band only if it reaches `ρ*(x_lo) = 567.1`, so
512 is formally certified only down to `x̂ ≥ 85.3`; the bottom sliver of the
band (`x̂ ∈ [74.6, 85.3)`) is covered empirically (§1.3.2) and by the
selector's occupancy floor, not by Theorem 4 alone. That residual gap is
exactly what the PROPOSAL DATA caveat and the routed sweep are for.

### 1.3 Why simulation lands at ~124–134 while the bound says 254–565

The task premise says simulated crossovers landed 124–134 for
`x ∈ {75, 150, 259}`. Reproducing that requires modeling what "typical"
means, and the reproduction sharpens the story in two ways.

#### 1.3.1 Proposition 6 (typical first crossing: the single-tile phase boundary)

*Under uniform routing (`P` balls into `E` bins uniformly), for `ρ` in the
single-tile regime — `1 ≤ c_e ≤ 255` for all practical `e`, which holds
overwhelmingly when `σ(ρ) ≪ ρ` and `ρ + 3σ < 256` — we have `Σa_e = 0` and
residues equal the counts, so Prop 3 collapses to*

    Δ = x·#{e : 129 ≤ c_e} − 128·#{e : c_e ≤ 128},

*i.e. 256 wins iff `#{c_e ≤ 128}/E < x/(x + 128)`. With `c_e → Poisson(ρ)`
(`E → ∞` limit of multinomial routing), the crossing `ρ*(x)` solves*

    Pr[Poisson(ρ*) ≤ 128] = x/(x + 128).

*Solving: `ρ* = {132.5, 127.5, 123.8}` for `x = {75, 150, 259}`.*
*Direct simulation of the exact integer cost model on sampled multinomial
histograms (`E = 256`, median-Δ bisection) gives first crossings
`{132.5, 127.4, 123.7}` — the reported 124–134 band.*

Why nearly `x`-independent: across `x = 75 → 259` the right-hand side moves
from 0.369 to 0.669, but the Poisson CDF at 128 has slope ~0.036 per unit
`ρ` there, so `ρ*` moves by under ten rows. The binding structure is the
phase boundary at `ρ ≈ 128` (counts sliding across the 128 row-line), not
the decode amortization ratio; that is why three different `x` values land
in a band of width ~9 while the analytic family spans 310 rows.

Note the uniform-*residue* heuristic (`ρ* ≈ R/E − 256·n_hi/E + 32768(q/E)/x
≈ 16384/x − 0.5`, i.e. `{218, 109, 63}`) is **self-inconsistent in this
band**: it assumes counts wrap the 256 boundary often enough to flatten
residues, which cannot happen when `ρ* < 256`. It becomes valid only well
above ~512. The honest typical model is the count distribution itself
(Prop 6), not uniform residues.

#### 1.3.2 New finding: the advantage profile is NOT monotone — re-crossings above the first crossing

Sweeping the full range instead of stopping at the first sign change
(`part1_profile.py`, `E = 256`, median Δ over sampled histograms):

| x | crossings of Δ (uniform routing, exact cost model) |
|---|---|
| 75 | 256 wins at ρ ≈ **133**; loses again at ρ ≈ **261**; wins for good at ρ ≈ **373** |
| 150 | 256 wins at ρ ≈ **127**, no further loss up to ρ = 700 |
| 259 | 256 wins at ρ ≈ **123**, no further loss up to ρ = 700 |

*Mechanism.* When the concentrated count distribution slides across a wrap
boundary (`ρ ≈ 256 + 128k` regions), the wrapped residues land en masse in
the cheap-pad class `[1,128]`: `q` jumps from ~0 toward 1 while `Σa_e`
grows only stepwise, and for small `x` (decode expensive) that flips Δ back
negative. Hand-check at `x = 75, ρ = 320`: essentially all counts wrapped
once, residues concentrated near 64 ⇒ `q ≈ E`, `n_hi ≈ 0`, `Σa ≈ E`, giving
`Δ/E ≈ 75 − 128 < 0`. As `σ(ρ)` grows the oscillation damps; for larger `x`
it never re-crosses at all.

**Consequences.**

1. The "~4× conservatism gap" compares the bound to *first* crossings. The
   honest comparison for `x = 75` is bound 564.9 vs last crossing ≈ 373:
   a factor of **1.51**, not 4.2. For `x = 150` and `x = 259` the factors
   are 2.72 and 2.06. The remaining gap is precisely the value of the
   histogram information the selector refuses to read (Thm 5).
2. A calibration sweep that measures only the first crossing would
   misread the `x = 75` cell as safely servable at `ρ = 200`; it is not —
   uniform routing itself prefers 128 again by `ρ = 261`. Any sweep protocol
   derived from this document must sweep `ρ` past the analytic threshold,
   not stop at the first win.
3. The shipped constant survives the discovery empirically: every observed
   last-crossing (~373 at worst) sits far below 512. Formally, Theorem 4
   certifies 512 against *all* histograms only for `x̂ ≥ 85.3` (Lemma 17);
   at the very bottom of the audited `x` band the all-histogram threshold
   is up to ~55 rows higher, which is precisely the sliver the routed sweep
   must pin (Thm 7).

#### 1.3.3 Theorem 7 (what a routed-sweep calibration must test)

*A routed sweep pins the constant iff it confirms, on production routed
histograms `h_j` with per-layer `ρ_j` and an independently identified `x`
(Part 5):*

1. **Model validity (per-histogram).** The sign of
   `Δ(h_j) = x(Σ_e⌊c_e/256⌋ + n_hi(h_j)) − 128·q(h_j)` predicts the sign of
   the measured `cost(128) − cost(256)` on every recorded cell. A single
   sign mismatch falsifies the additive model `T(t) ∝ B(t)(d + tm)` (e.g.
   warp-specialized overlap breaking additivity), not merely the constant.
2. **Bound validity (histogram-uniform).** No observed histogram with
   `ρ ≥ 128(1 + 256/x̂)` shows 256 losing. This must hold for *every*
   histogram; it is the empirical content of Thm 4.
3. **Constant adequacy.** Measured `cost(256) ≤ cost(128)` at `ρ = 512` on
   every cell, including cells swept past their first crossing (§1.3.2),
   and the measured last-crossing envelope stays below 512 for the audited
   `x̂` band.

*Prediction if the theory is right: first crossings reproduce the
`123–134` band at the audited `x` values (Prop 6); re-crossings appear only
near the bottom of the `x` band and terminate below 512; no violation of 2
ever appears.*

Status: items 1–3 are **PROPOSAL DATA until run**; the simulation evidence
above is CPU-side integer arithmetic, not operator timing.

---

## Part 2 — the window non-aliasing lemma for the fp4-v2 codeword decoder

Three call sites share one extraction scheme:

- **Dense GEMV** `fp4v2_decode_fma` (cb_gemv.cu:745-781) fed by
  cb_gemv.cu:837-839 / :864-866: the slot holds exactly one superblock
  (`type_size = 4k+9` bytes, slot byte 0 = superblock byte 0);
  `w0,w1,w2` are read **unconditionally** as three consecutive u32 smem words.
- **Grouped MoE GEMV** `cb_moe_gemv_fp4_v2_kernel` (cb_gemv.cu:1354-1394):
  interior superblocks are staged with ONE aligned u64 burst from the
  aligned-down base (`off8 = gsrc & 7`, `nv = ceil((off8 + type_size)/8)`
  words); extraction is identical to dense but with
  `bitpos = off8·8 + v·k`, and the third word is **predicated**:
  `w2 = (rem + k_bits > 64) ? s32[widx+2] : 0` (cb_gemv.cu:1383). The last
  superblock of a row takes the byte path with `off8 = 0`
  (cb_gemv.cu:1363-1367) because a u64 there could read past the tensor.
- **Persistent-B decode probe/mainloop** `cb_extract_code`
  (cb_moe_persistent_b.cu:313-328): same as dense with the third load
  inside the same predicate.

The common arithmetic, for a codeword index `v ∈ [0,32)` at bit position
`bitpos` within the staged slot image:

    b0 = bitpos >> 3;   rem = ((b0 & 3) << 3) + (bitpos & 7) = bitpos mod 32;
    widx = b0 >> 2 = bitpos div 32;
    lo = (w1 << 32) | w0;                       // window = slot bits [32·widx, 32·widx+64)
    code = lo >> rem;  if (rem + k > 64) code |= w2 << (64 − rem);
    code &= (1 << k) − 1;

**The window base is `4·widx` slot bytes — u32-aligned-down from the
codeword's first bit, not 8-byte-aligned.** The task phrasing "aligned u64
window at byte floor(8vk/64)" would put `rem ∈ [0,64)` and genuinely
require spilling (e.g. k=24, bitpos ≡ 56 mod 64 ⇒ rem+k = 80). The kernels
do not do that; they assemble the window from two u32 words, so
`rem ∈ [0,32)` and the slack after the codeword start is always ≥ 32 bits.
This is what the predicated-load optimization rests on.

### 2.1 Lemma 10 (containment)

*For every `k ∈ [12,24]`, every `v ∈ [0,32)`, and every staging head offset
`off8 ∈ [0,7]`: all `k` bits of codeword `v` lie inside the two-word window,
i.e. `rem + k ≤ 64`; in fact `rem + k ≤ 55`. Hence `code` equals the ground
truth stream bits `[v·k, (v+1)·k)` and the third word never contributes.*

*Proof.* `rem = bitpos mod 32 ≤ 31` by construction (it is literally
`(b0&3)<<3 | (bitpos&7)`), so `rem + k ≤ 31 + 24 = 55 < 64`. The masked
field keeps exactly field bits `[rem, rem+k)` = stream bits
`[32·widx + rem, 32·widx + rem + k) = [bitpos, bitpos + k)`, which are the
LSB-first codeword bits by the packing definition. ∎

*Check:* 21 rungs × 32 codewords × 8 phases = 5376 cases, random payloads,
extraction vs an independent bit-slice oracle: exact everywhere; max
observed `rem + k = 54`; spill count 0 (`part2_window.py`). The grouped
staging coverage claim (`nv` u64 words cover both the payload and the whole
two-word window for every rung/phase) is asserted in the same script.

### 2.2 Lemma 11 (the predicate exactly characterizes spill)

*Let `code₃` denote the masked three-word assembly and `code₂` the masked
two-word assembly. Then for every `(k, v, off8)`: `code₂ == code₃ ==
ground truth` **for every payload** iff `rem + k ≤ 64`; if `rem + k > 64`
the bits `[64 − rem, rem + k − 64)` of the codeword live only in `w2`, so
`code₂ ≠ ground truth` for some payloads (they can still coincide on a
payload whose spilled bits happen to be zero). Consequently zeroing `w2`
when the predicate is false (or not loading it at all) is bit-neutral in ALL
cases, and required in none.*

*Proof.* If `rem + k ≤ 64`, every kept bit lies in `lo`, and the spilled
term `w2 << (64−rem)` contributes only at bit positions `≥ 64 − rem ≥ k`,
all of which the mask removes; so `code₃ = code₂ = truth`. If
`rem + k > 64`, codeword bits `64 − rem … k − 1` (count `rem + k − 64 > 0`)
sit at field positions `≥ 64`, absent from `lo`; `code₃` recovers them from
`w2`'s low bits (shift `64 − rem ∈ (0,64)`), `code₂` cannot. ∎

*Check:* verified jointly with Lemma 10 on the same 5376 cases plus
synthetic high-`k` streams where the predicate fires
(`part2_window.py::two_word_ok`).

### 2.3 Corollary 12 (enumeration; where the third load is ever required)

With `off8 = 0` (dense kernel; persistent-B; grouped last superblock),
`rem = (v·k) mod 32` ranges over the subgroup `gcd(k,32)·Z/32Z` whose
largest element is `32 − gcd(k,32)`, so **some codeword spills iff
`k > 32 + gcd(k,32)`**; with `off8` free over `[0,7]` the phases add
multiples of 8, giving **`k > 32 + gcd(k,8)`** for the grouped interior
path. Both closed forms match exhaustive enumeration for all
`k ∈ [12, 200)` (`part2_window.py` tail; standalone check).

Enumerated consequences:

| rung range | third word required? |
|---|---|
| fp4-v2 product ladder `k ∈ [12,24]`, any `v`, any `off8` | **never** (empty set; first possible `k` is 35 grouped, 35 ungrouped) |
| FP8-CB persistent-B arm `k ∈ {28…48}`, `off8 = 0` | only **k = 44**: `v ≡ 2, 5 (mod 8)`, i.e. 8 of 32 codewords per superblock |
| `k = 34, 36, 40, 48` | provably never (`k ≤ 32 + gcd(k,32)` fails at equality) |

So in everything shipped today the predicated third load is a pure
load-count optimization with exactly one live cell (persistent-B cfgs 2/3
at `k=44`), where it is *correctness-relevant bookkeeping* rather than dead
code: the branch fires, reads `s32[widx+2]`, and the mask still trims the
result to payload bits. Two safety notes proven alongside
(`part2_window.py`): (i) even the *unconditional* three-word read stays
inside the padded slot for every rung of either family
(`4(widx_max + 2) + 4 ≤ ts_pad`, e.g. bytes 196 ≤ 200 at k=48), so
predication changes no addressable-range behavior; (ii) when the spill
branch does fire at k=44 its extra source words land in payload, not slack,
and the mask makes slack contents irrelevant either way — the
zero-fill-consumes-zeros remark in the B2 contract (P2,
cb_moe_persistent_b.cu:600-604) is about definedness of the staged image,
not about bits reaching `code`.

---

## Part 3 — empty experts in persistent-B: the exact wasted-work bound

The persistent-B grid is `E × ceil(N/TN)` work units, a function of layer
shape alone (docs/KERNELS.md, "Grouping: exact segments"). Each CTA opens
its work unit by loading the segment endpoints
(cb_moe_persistent_b.cu:779-786):

    raw_lo = (e == 0) ? 0 : expert_ends[e-1];   // int32 load (skipped for e=0)
    raw_hi = expert_ends[e];                    // int32 load
    …clamp to [0, P]…
    if (m_hi <= m_lo) continue;                 // empty expert: early out

### 3.1 Lemma 13 (wasted work per empty expert)

*An empty expert contributes exactly* `ceil(N/TN)` *CTAs that each perform
at most two int32 global loads, one compare-and-branch, and an exit — and
nothing else. Total across `E_empty` empty experts:*
`2·E_empty·ceil(N/TN)` *loads and* `E_empty·ceil(N/TN)` *short-lived CTA
slots. The launch geometry itself (`E × ceil(N/TN)` CTAs) is
routing-independent and is paid even when no expert is empty, so it is not
a cost of routing-obliviousness.*

*A DRAM-time bound:* the loads touch `expert_ends[e−1..e]`, adjacent int32s
in a `4E = 1 KiB` array that is L2-resident after the first wave; the
pessimistic uncached model charges `8 B` per CTA (one 32-byte sector in
practice), giving

    τ_waste ≤ 8 B · E_empty · ceil(N/TN) / BW      (BW ≈ 273 GB/s on GB10).

### 3.2 DSv4 numbers (E=256, topk=6, N=4096 for both w13 and w2 at h4096/i2048)

Under uniform routing `P(empty) = (1 − 1/256)^{6T} ≈ e^{−ρ}`:

| T | ρ | cfg/TN | E[empty] | empty CTAs | wasted bytes | τ @273 GB/s |
|---|---|---|---|---|---|---|
| 32 | 0.75 | cfg4/128 | 120.7 | 3 864 | 30.9 KB | 113 ns |
| 128 | 3 | cfg4/128 | 12.7 | 406 | 3.2 KB | 11.9 ns |
| 256 | 6 | cfg4/128 | **0.63** | 20 | 161 B | **0.59 ns** |
| 512 | 12 | cfg4/128 | 1.5e−3 | 0.05 | 0.4 B | ~0 |
| 2048 | 48 | cfg4/128 | ~e^{−48} | 0 | 0 | 0 |

At the task's named operating point (`T = 256`, `topk = 6`, `ρ = 6`):
`P(empty) = (1 − 1/256)^{1536} = 0.002450 ≈ e^{−6} = 0.002479`; expected
empty experts `0.63`; expected waste `≈ 161 B ≈ 0.59 ns`. Persistent-B
whole-operator cells at these shapes are milliseconds (measured anchors:
the 18.6-vs-7.8 ms DSV4 `w2` sweep cell and the 4.09→7.76 ms smem-resident
rejection in docs/KERNELS.md). The worst in-regime case (T=32, the smallest
batch above the grouped-GEMV handoff) is 113 ns, or ≤ 906 ns under the
fully pessimistic sector model with no L2 reuse — against even a 1 ms cell
that is ≤ 0.09%, against the measured production cells 0.006–0.001%.

**Corollary 14 (price of shape-pure grids).** *Routing-obliviousness in
the persistent-B lane costs at most ~0.1% of operator time at any batch the
lane serves, and becomes unmeasurable (`< 10⁻⁵%`) once `ρ ≳ 12`; the bound
is monotone-decreasing in T under uniform routing because useful work grows
like `P` while empties decay like `e^{−ρ}`.* The honest scope note: this
lemma prices only the early-out path. What shape-purity *also* gives up is
histogram-aware block ordering (the swizzle-group-aligned packing of the
sm12x lane), which is a different, larger, and already-measured lever — do
not cite Corollary 14 against it.

## Part 4 — staging vectorization is byte-neutral, given the tail invariant

Context: the B2 task replaces the persistent-B packed-superblock staging
loop (byte-granular `dst[b] = __ldg(src+b)` over `[0, type_size)`, then
zero-fill of `[type_size, ts_pad)`) with u32 vector copies
(`pb_stage_row` / `pb_stage_row_zero`, working tree,
cb_moe_persistent_b.cu:629-691; call sites :840-855 baseline mainloop,
:898-917 D2R variant, decode probes :1078/:1120/:1168). **Status: B2 was
subsequently REJECTED on performance (independent A/B: +7…+11% whole-
operator at k=12); this part concerns byte-neutrality only and does not
depend on that implementation landing.** The bit-neutrality
contract is stated as P1–P5 at cb_moe_persistent_b.cu:588-625 and gated by
`tests/test_persistent_b_stage_vectorization.py` (torch.equal A/B across
byte phases). Here is the theorem that contract restates, with the proof
the gate deserves.

### 4.1 Theorem 15 (byte-image identity)

*Let `ts = type_size`, `ts_pad = ((ts+3)/4)·4 + 8` (the `ts_padded()`
definition), and let the reference image be: payload bytes `[0, ts)` copied
verbatim from `src`, bytes `[ts, ts_pad)` zero. Any copy scheme that
(i) writes every byte of `[0, ts)` with its payload value exactly once,
(ii) writes every byte of `[ts, ts_pad)` with zero exactly once, produces a
byte-identical slot image — regardless of load width, alignment, or the
instructions used to move payload bytes. In particular:*

- *(a)* *for every `ts ≡ 0 (mod 4)` (all FP8-CB rungs, `ts = 4k`,
  `k ∈ [28,48]`) the u32/uchar4 scheme "aligned body words + sub-word
  scalar tail + whole-word zero fill" satisfies (i)+(ii);*
- *(b)* *for every odd `ts = 4k+9` (FP4-CB v2 ladder, `k ∈ [12,24]`) the
  same scheme satisfies (i)+(ii) **iff the seam bytes**
  `[ts, 4·ceil(ts/4))` *— the high half of the last partial payload word —
  are zero-filled byte-wise while that word's low bytes carry payload;
  dropping exactly that seam write breaks identity precisely when
  `ts mod 4 ≠ 0`.*

*Proof.* (ii) is by construction of the zero region: the seam loop writes
bytes `[ts, zw<<2)` with `zw = ceil(ts/4)`, which are disjoint from every
payload write (all payload writes stop at or below `zw<<2 − (4 − ts mod 4)`
… precisely: the tail payload loop covers `[tail_begin, ts)`), and the
whole-word loop `[zw, ts_pad/4)` covers the remainder; `ts_pad ≡ 0 (mod 4)`
makes the word view well-defined. For (i): dst is 4-byte aligned
(`extern __shared__ __align__(16)` base, u16-pointer arithmetic to `sPk`
lands on a multiple of 16 bytes, slot stride `ts_pad ≡ 0 mod 4`), so each
aligned-down dst word `j ∈ [1, jhi)` must equal little-endian assembly of
stream bytes `[4j − m_src, 4j − m_src + 4)` where `m_src = src_addr mod 4`;
`__funnelshift_r(wsrc[j], wsrc[j+1], 8·m_src)` computes exactly
`(wsrc[j+1]:wsrc[j]) >> 8·m_src` truncated to 32 bits — the little-endian
byte rotation — and the uchar4 twin computes the identical four bytes by
permutation. The bounds `jlo = 1` and `jhi = (ts + m_src − 4) >> 2` keep
every vector read inside `[src − m_src, src + ts)`, i.e. never before the
tensor front nor past the row end (the tight checkpoint plane has no slack);
word 0 and the sub-word tail take the scalar paths. Disjointness of the
four regions (scalar head `[0, head_end)`, body words, scalar tail,
seam + slack zeros) then makes the image well-defined and equal to the
reference byte-for-byte. ∎

*Check:* simulation of reference vs the u32-funnel scheme vs the uchar4
twin over all four source phases, all FP8 rungs (`ts = 4k, k = 28…48`) and
FP4 rungs (`ts = 4k+9, k = 12…24`), varied superblock offsets, with garbage
-initialized slots: **0 mismatches**; the negative test (drop only the seam
write) breaks identity **exactly** on the `ts mod 4 ≠ 0` rungs and never on
`ts ≡ 0` rungs — 0 unexpected outcomes (`part4_staging.py`). The negative
half matters: it shows the zero-fill is load-bearing, not decorative.

### 4.2 Where the zero-fill guarantee lives, and whether the u32 path preserves it

- **Definition:** `ts_padded()` (cb_moe_persistent_b.cu:1197-1199 on this
  tree) rounds `type_size` up to a multiple of 4 and adds 8 — the padding
  is therefore ≥ 8 bytes, a multiple of 4, and large enough that the
  decoder's three-word window at the last codeword stays in-slot (Part 2).
- **Enforcement:** the seam + whole-word zero loops at the end of
  `pb_stage_row` (:681-691 area), all-of-slot zeroing in `pb_stage_row_zero`
  (:629-634) for out-of-range rows, and the same two loops inline at the
  pre-B2 call sites.
- **Consumers:** the decode stage reads the slot through u32 windows
  anchored at codeword starts (`cb_extract_code`), legitimately reaching
  into the slack (P2, :600-604); FP8 additionally stages row scales after
  the packed plane, addressed off `TN·ts_pad`, which requires `ts_pad ≡ 0
  (mod 4)` for the float view to stay aligned (:743-748).
- **Verdict:** the u32 path as written **preserves** the guarantee (it
  re-establishes the full zero tail explicitly), and the gate test pins it
  from outside. Line numbers on an in-flight branch drift; the durable
  anchors are the function names above plus the P1–P5 block.

### 4.3 The precondition, stated precisely

Any future rewrite of the staging (uchar4, cp.async bulk, TMA-style) is
bit-neutral iff it maintains, for every staged row:

1. **Absolute positions:** payload byte `b` lands at slot byte `b`
   (readers index absolute offsets — P1);
2. **Full coverage, single writer:** every byte of `[0, ts_pad)` written
   exactly once per staging generation — payload verbatim in `[0, ts)`;
   seam bytes `[ts, 4⌈ts/4⌉)` zeroed byte-wise whenever `ts mod 4 ≠ 0`;
   whole slack words zeroed thereafter (P2 + Theorem 15b);
3. **Alignment invariants:** slot base 4-byte aligned (16-byte claimed and
   used by the u64/u32 views) and `ts_pad ≡ 0 (mod 4)` (float scale view)
   — P3;
4. **Bounded reads:** no vector read crosses `[tensor_start − 0, row_end]`,
   absorbing source misalignment on the LOAD side only (P4/P5 funnel-shift
   discipline).

A violation of 2 is invisible to value tests on aligned FP8 rungs and
appears only as garbage-dependent decode at odd `type_size` — exactly the
failure mode the negative test in `part4_staging.py` demonstrates and the
phase-sweep gate test exists to catch on hardware.

---

## Part 5 — identifying x from timings, and propagating the interval (lemmas)

### 5.1 Lemma 16 (two-point identification)

*(a) Same-shape tiles.* At fixed padded row count `M` (both arms fully
tiled), `T(128)/T(256) = 2(x + 128)/(x + 256)`. Hence a measured ratio
`r = T(128)/T(256) ∈ (1, 2)` identifies

    x = 256(r − 1)/(2 − r),

*a strictly increasing bijection `(1,2) → (0, ∞)`. Applied to the dense
TileM A/B gains (22.57% → r = 1.2257; 50.32% → r = 1.5032): `x = {74.6,
259.3}` — the audited band `[75, 259]` of moe_routing.py:251.*

*(b) Fused vs expand.* With the additive model, one fused-lane timing pair
at two token counts `M₁ ≠ M₂` and fixed tile `t` solves the two unknowns:
`T(Mᵢ) = ceil(Mᵢ/t)(d + t·m)` gives `m = (T₂ − T₁)/((b₂ − b₁)t)` and
`d = T₁/b₁ − t·m` (`bᵢ = ceil(Mᵢ/t)`), hence `x = d/m` directly; the
expand-only operator supplies the independent check that its own time is
`(decode-once) + M·(per-row traffic + MMA)` with the same slope `m`.

### 5.2 Lemma 17 (interval propagation through the threshold family)

*`ρ*(x) = 128(1 + 256/x)` is strictly decreasing on `(0, ∞)`. Therefore an
identified band `x ∈ [x_lo, x_hi]` propagates to the threshold interval
`[ρ*(x_hi), ρ*(x_lo)]` — endpoints swap — and any constant `C` is certified
for the whole band iff `C ≥ ρ*(x_lo)`. For `[74.6, 259.3]`: thresholds
`[254.4, 567.1]`. The shipped 512 therefore does **not** reach the band's
pessimistic end: it is certified only over the sub-band where
`ρ*(x̂) ≤ 512`, i.e. `x̂ ≥ ≈85.3`; below that, Theorem 4 alone cannot
defend the constant against adversarial histograms.*

*Proof:* monotonicity of `1/x`; endpoint evaluation. For the shipped
constant the certificate edges are `ρ*(x̂) = 512 ⟺ x̂ = 256/(512/128 − 1) =
256/3 ≈ 85.3` (an identified `x̂ < 85.3` invalidates a 512 constant) and,
illustratively, `x̂ ≥ 153.9` would license dropping the constant to `≤ 341`.
All numeric claims of this lemma are checked in part3_empty.py.*

The practical reading for the routed sweep: measuring `x̂` tightly matters
most between `x̂ = 74.6` and `x̂ = 85.3`, where the shipped certificate
survives or dies; near the top of the audited band the constant is loose by
a factor of 2 and effort is better spent on histogram structure (Part 1.3).

---

## Where each piece belongs

**docs/KERNELS.md** (design-record voice, same-commit rule per AGENTS.md §10):
- §Grouped TileM selection ("Why ρ"): add one sentence citing Lemma 2's `q`
  definition explicitly (residue class `[1,128]`, *not* multiples of 256),
  Theorem 5's tightness ("no smaller host-knowable threshold exists"), and
  Prop 6's typical-crossing formula with the 123–134 reproduction — plus
  the §1.3.2 re-crossing caveat and its sweep-protocol consequence.
- §Decode path (fp4-v2): one paragraph stating Lemma 10/11 (containment +
  predicate characterization) and Corollary 12's table outcome ("third word
  dead on every shipped rung; live only at k=44 in persistent-B").
- §Persistent-B ("Grouping: exact segments"): quantified empty-expert bound
  (Lemma 13/Corollary 14) next to the existing "two int32 loads" sentence.
- Persistent-B B2 contract (P1–P5): reference Theorem 15 and the four-point
  precondition list instead of restating them.

**ROADMAP.md** (kernel TODO): the routed-sweep calibration as Theorem 7's
three-item acceptance test (including the sweep-past-first-crossing
requirement); the predicated third load and the B2 vectorization as items
whose review anchor is Lemmas 10–11 and Theorem 15 respectively; optionally
a line item for pinning `x̂` per Lemma 16(b) since the certificate margin at
512 depends on it.

**Future paper appendix**: Parts 1.2–1.3 in full (the residue-only
reduction of Prop 3, the tightness argument of Thms 4–5, and especially the
non-monotone advantage profile — that oscillation seems to be new to the
grouped-GEMM tile-selection literature); Part 2 complete with the
subgroup closed forms (`k > 32 + gcd(k,32)` / `k > 32 + gcd(k,8)`);
Part 4 as a short lemma on byte-exact vectorization contracts. Parts 3 and
5 are engineering notes; they belong in docs, not a paper.

## Verification index

| claim | script | result |
|---|---|---|
| Lemma 1, exhaustive c ∈ [0,20000) | part1_rho.py | pass |
| Lemma 2 (+ q₀ counterexamples) | part1_rho.py | pass, 20k histograms |
| Prop 3 residue form | part1_rho.py | pass, 20k histograms |
| Thm 4 sampling search above bound | part1_rho.py | no losing histogram |
| Thm 5 adversarial tightness (rational) | part1_rho.py | gaps ≤ 256/E |
| Prop 6 first crossings | part1_rho.py + part1_profile.py | 132.5 / 127.4 / 123.7 |
| §1.3.2 re-crossing profile | part1_profile.py | x=75: 133 / 261 / 373 |
| Lemmas 10–11, 5376 cases + spill characterization | part2_window.py | pass |
| Corollary 12 closed forms, k ∈ [12,200) | part2_window.py tail | pass |
| Lemma 13/Cor 14 table | part3_empty.py | table above |
| Lemma 16/17 numerics | part3_empty.py | x ∈ [74.6, 259.3] → [254.4, 567.1] |
| Thm 15 identity + negative test | part4_staging.py | 0 mismatches; 0 unexpected |


