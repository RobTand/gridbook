# gridbook codebook formats — Format Specification

**Status: normative.** This document specifies the on-disk byte layout, tensor
naming, configuration vocabulary, and decode semantics of the NVFP4-CB and
FP8-CB codebook quantization formats. It is **implementation-independent**: any
encoder that emits these bytes and any runtime that decodes them per this
document interoperate. The specification is published under Apache License 2.0
and may be implemented by anyone, in any language or runtime, without permission.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHOULD**,
**SHOULD NOT**, **MAY**, and **OPTIONAL** are to be interpreted as described in
RFC 2119.

---

## 0. Format family at a glance

A **codeword** encodes a **`d = 8`**-dimensional vector of grid values. A
**`k`-bit index per 8 weights** selects a codeword. There are two value grids and
two index-encoding modes.

| Family | Grid | Codeword values | Activation | Per-group scale | Body bits/weight |
|---|---|---|---|---|---|
| `NVFP4_CB_K{k}` | FP4 / E2M1 | `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` | W4A4 | group-16 E4M3, **in the weight bytes** | `k/8 + 0.5` (v1) |
| `FP8_CB_K{k}` | FP8 / E4M3 | E4M3 grid, `‖·‖ ≤ 448` | W8A8 | **none in weight bytes** — per-output-channel FP32, separate tensor | `k/8` |

**NVFP4 product rung domain.** `NVFP4_CB_K{k}` is publicly defined for every
integer `12 ≤ k ≤ 24`, with `n_sub = 2` and the ceil-first split above. The
packing primitive mechanically extends to direct-kernel research K1..K32: at
K1 the split is `(1, 0)`, so the second sub-codebook has one four-value entry
and its index is the empty/zero bit string; at K32 the split is `(16, 16)`.
E2M1 has 16 wire
encodings but 15 distinct numeric values because its two zero encodings
coincide, so a four-value subvector has `15^4 = 50,625` distinct numeric
choices: more than `2^15`, but no more than `2^16`. Each K32 half can therefore
contain the complete numeric lattice (with deterministic duplicate rows to
fill its 65,536-entry table); a K33 half cannot add a new numeric lattice value.
That mathematical ceiling motivates the direct CUDA research instantiations,
but does not enlarge the format: reader, producer, format-chooser, and
producer-facing lane domains are all exactly K12..K24. K1..K11 and K25..K32
are not valid public artifact rungs. The briefly developed K1..K25 public
expansion was retracted before the planned 0.9.1 release.

> **Removed family.** `NVFP4_CB_S{k}` (sign-magnitude half-grid, `mode:
> "signed"`) was deleted from the runtime on 2026-08-23 — the producer had
> stopped emitting it on 2026-08-17 because an `n_sub=1` codebook can never
> satisfy the native-FP4 predicate. No published artifact encodes it. A
> conforming implementation **MUST refuse** a `"mode": "signed"` scheme
> instead of decoding it.

> **Naming.** The dtype in a family name (`NVFP4`, `FP8`) names the **grid the
> codebook's values live on — not the storage width**. Storage is the k-bit
> vector index: `FP8_CB_K32` stores 4.0 bits per weight, and what those 4 bits
> reconstruct is an exact `float8_e4m3fn` value. The grid constraint is what
> makes decoded tiles **native-kernel-compatible**: an expanded FP8-CB weight is
> a standard per-channel fp8 tensor (stock fp8 tensor-core GEMMs consume it
> directly), and a decoded NVFP4-CB tile is bit-compatible NVFP4.

A decoded FP4 tile is **bit-compatible NVFP4**: E2M1 codes plus an NVFP4 group-16
E4M3 scale plane. A conforming FP4-CB decode therefore produces exactly what a
standard NVFP4 tensor-core GEMM consumes. This property is the reason the family
exists and is a normative goal of the FP4 grid (§8).

**Superblock constraint.** The quantization unit is a **256-weight superblock**
along the input dimension. `in_features % 256 == 0` is **REQUIRED** for any CB
target Linear. A Linear whose input dimension is not a multiple of 256 **MUST NOT**
be encoded in a CB format; producers **MUST** fall back to a non-CB format (e.g.
BF16) for it and list it in the `ignore` set (§5).

---

## 1. Superblock byte layout (the weight stream)

Per output channel (row) there are `in_features / 256` superblocks laid out
contiguously. The packed weight tensor is 2-D `uint8` of shape
`(rows, (in_features / 256) * type_size)`. Each superblock occupies exactly
`type_size` bytes.

A superblock is laid out as an **index stream** followed by a **scale plane**:

```
┌───────────────────────────────┬──────────────────────────────┐
│ INDEX STREAM  (4k bytes)       │ SCALE PLANE                  │
│ 32 codewords of k bits,        │ fp4 v1: 16 bytes             │   type_size bytes
│ LSB-first bit packing          │ fp4 v2:  9 bytes             │
│                                │ fp8:     0 bytes (absent)    │
└───────────────────────────────┴──────────────────────────────┘
```

- The index stream is **`4k` bytes** long: 32 codewords × `k` bits = `32k` bits =
  `4k` bytes (integer for every `k`). "32 codewords" = 256 weights ÷ 8 (the
  vector dimension `d`).
- The scale plane (fp4 only) holds 16 group scales: 256 weights ÷ 16 (the group
  size). Its byte length depends on the scale coding (§2).
- Because `32k = 4k × 8`, **superblock boundaries fall on byte boundaries**; a
  whole row's index region MAY be unpacked as one contiguous bitstream.

> **Producer/recap note.** The index stream is `4k` bytes containing 32
> codewords of `k` bits each — it is *not* "32 bytes." (This corrects a common
> shorthand.)

### 1.1 Index stream — bit packing (LSB-first)

The 32 codewords of a superblock are concatenated into one bitstream, **least
significant bit first**, then emitted 8 bits per byte:

```
stream bit index:   0            k           2k                    32k-1
                    ┌── cw0 ──┐  ┌── cw1 ──┐  ┌── cw2 ──┐   ...    ┌ cw31 ┐
byte b  =  Σ over j in [0,8)  of  stream_bit[8*b + j] << j            (4k bytes)
```

Codeword `c` occupies stream bits `[c*k, c*k + k)`, its own **LSB first**. A
decoder **MUST** reconstruct codeword `c` by reading those `k` bits LSB-first into
an integer.

Each `k`-bit codeword encodes one 8-dimensional vector. Its internal structure
depends on the **mode**:

**`full` mode** — the codeword *is* the codebook index (`0 ≤ idx < 2^k`):
```
 bit:  k-1 ................. 0
       [        idx         ]
```

**`product` mode** — the index is split into `n_sub` sub-indices packed
contiguously, **sub-index 0 in the low bits**. Sub-index widths are the
`bit_split(k, n_sub)` partition: as even as possible, **larger parts first**.
Examples: `bit_split(13, 2) = (7, 6)`; `bit_split(40, 4) = (10, 10, 10, 10)`;
`bit_split(36, 4) = (9, 9, 9, 9)`.
```
 fp4  (n_sub = 2, widths b0 >= b1, b0 + b1 = k):
 bit:  k-1 ........ b0 | b0-1 ...... 0
       [    sub1     ] | [   sub0    ]

 fp8  (n_sub = 4, widths b0..b3):   high ─────────────────► low
       [ sub3 ][ sub2 ][ sub1 ][ sub0 ]
```
Sub-index `i` decodes `8 / n_sub` coordinates via sub-codebook `i`. The 8-dim
codeword is the concatenation `[sub0 | sub1 | ...]`. `n_sub` **MUST** be 2 for the
FP4 grid and 4 for the FP8 grid in this version.

**`signed` mode — REMOVED (2026-08-23).** The historical encoding was 8
explicit sign bits (low byte) then a `(k-8)`-bit magnitude index into one
non-negative half-grid table. It is no longer part of the format: implementations
**MUST refuse** `"mode": "signed"` rather than decode it.

### 1.2 Scale coding — FP4 grid

The scale plane immediately follows the `4k` index bytes. Two codings are
defined. **Absence of a `scale_coding` key in the scheme (§5) means v1.**

**v1 — E4M3-direct (16 bytes, default).** 16 `E4M3` bytes, one group-16 block
scale per 16 consecutive weights. Group `g` covers weights `[16g, 16g+16)` of the
superblock. Each byte is the `float8_e4m3fn` value reinterpreted as `uint8`
(`scale.to(float8_e4m3fn).view(uint8)`). This plane is **byte-identical to
NVFP4's block-scale plane**. Reconstruction:
```
weight[i] = codeword_value[i] * e4m3_scale[ group(i) ]      where group(i) = (i mod 256) // 16
```

**v2 — two-tier (9 bytes, `layout_version: 2`).** The plane is:
```
[ SUPER : 1 byte (E8M0, bias 127) | SUB : 8 bytes (16 × 4-bit codes) ]
```
- `SUPER` is a `uint8` exponent `E`; the superblock's power-of-two super-scale is
  `2^(E - 127)` (MX / E8M0 convention, bias 127).
- `SUB` is 16 4-bit codes `c_g`, group `g` stored in byte `g // 2`, with
  **even `g` in the low nibble** (LSB-first, consistent with the index stream).
- Each `c_g` indexes a fixed **16-entry multiplier table `T`** carried in the
  scheme (`scale_coding.table`).

Reconstruction:
```
scale_g = T[c_g] * 2^(E - 127)
```

The composed value **MUST** be an exact `float8_e4m3fn` value in `(0, 448]`. This
holds by construction: every table entry has the form `(8+j)/8 × 2^i` (an E4M3
significand `8+j ∈ [8,15]` times a power of two), so the product is an E4M3 value
whenever the composed exponent is in range, and a conforming encoder **MUST** emit
only `(E, c_g)` pairs whose composition round-trips `float8_e4m3fn` bit-exactly and
lies in `(0, 448]`. A decoder therefore reconstructs a bona-fide E4M3 plane with a
plain FP32 multiply — no cast, no rounding.

> An implementation **MAY** precompute the full product as a `256 × 16` table
> indexed by `(E, c_g)` and read `scale_g` directly; this is an optimization of the
> normative `T[c_g] * 2^(E - 127)` formula and **MUST** produce identical bytes.

The default table is **`T4_2oct8m`** = `{1.0, 1.125, 1.25, 1.375, 1.5, 1.625,
1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75}` — all 8 E4M3 mantissa
steps across 2 octaves. A 5-bit-sub variant (`type_size = 4k + 11`, 32-entry
table) is reserved but not required. The table ships in the config (§5), so it is
self-describing and per-artifact tunable without a layout change; a producer
**MUST** assert every entry is E4M3-exact at pack time.

**Zero and out-of-range rules (v2).** `T` contains no zero, so a composed scale is
never 0. An all-zero group takes the first legal candidate (deterministic). An
all-zero superblock stores the smallest legal `E` and all-zero sub codes. A group
whose ideal scale sits below the superblock's reachable set snaps **up** to the
smallest reachable value (the no-clip direction). Encoders **MUST** be
deterministic on the scale path (identical input bytes → identical output).

### 1.3 Scale coding — FP8 grid

The FP8 grid has **no per-superblock scale plane**. `type_size = 4k`. Its scales
are **per-output-channel FP32**, shipped in a separate tensor (§4).

The Gridbook runtime profile distinguishes a wire **reader domain** from a
canonical **producer menu**. FP8-CB readers accept every integer
`k ∈ [28,48]`, preserving historical artifacts. New producers emit exactly
`k ∈ {40,44,48}`. These sets do not change the packing rule:
every accepted FP8 row still has exactly `4k` index bytes per 256 weights and
uses the ceil-first four-way product split above. A runtime may support a
broader direct-kernel research `k` (the generic implementation retains aligned
K4..K24), but it MUST NOT claim that value as part of Gridbook's v11 reader or
producer profile. The pre-release K4..K48/4 producer expansion was retracted
before the planned 0.9.1 release.
Reconstruction:
```
weight[i] = codeword_value[i] * weight_scale[row]
```

### 1.4 `type_size` and effective bits (normative; asserted by producers)

Rows outside the producer menus above are arithmetic examples for historical
readers or direct-kernel research; their presence in this table does not make
them legal producer outputs.

| Grid | k | `type_size` v1 | `type_size` v2 (4-bit sub) | index bytes (4k) | scale bytes v1 / v2 |
|---|---|---|---|---|---|
| fp4 | 1  | 20  | 13  | 4   | 16 / 9 |
| fp4 | 12 | 64  | 57  | 48  | 16 / 9 |
| fp4 | 13 | 68  | 61  | 52  | 16 / 9 |
| fp4 | 14 | 72  | 65  | 56  | 16 / 9 |
| fp4 | 16 | 80  | 73  | 64  | 16 / 9 |
| fp4 | 18 | 88  | 81  | 72  | 16 / 9 |
| fp4 | 20 | 96  | 89  | 80  | 16 / 9 |
| fp4 | 24 | 112 | 105 | 96  | 16 / 9 |
| fp4 | 25 | 116 | 109 | 100 | 16 / 9 |
| fp4 | 26 | 120 | 113 | 104 | 16 / 9 |
| fp4 | 32 | 144 | 137 | 128 | 16 / 9 |
| fp8 | 36 | 144 | —   | 144 | 0 |
| fp8 | 40 | 160 | —   | 160 | 0 |
| fp8 | 44 | 176 | —   | 176 | 0 |
| fp8 | 48 | 192 | —   | 192 | 0 |

- `effective_bits(fp4, v1) = (4k + 16) * 8 / 256 = k/8 + 0.5`
- `effective_bits(fp4, v2) = (4k + 9) * 8 / 256 = k/8 + 0.28125`
- `effective_bits(fp8 body) = 4k * 8 / 256 = k/8` (plus the per-channel FP32
  plane, negligible amortized over `in_features`)

`effective_bits` is **version-keyed**: a v2 artifact accounts scales at
`0.28125 bpw`, a v1 artifact at `0.5 bpw`. Two-tier (v2) applies to the FP4 grid
only (the FP8 grid has no per-superblock scale plane). A producer **MUST** assert
`type_size` matches `(grid, k, layout_version)`, so a mislabeled artifact fails at
load rather than decoding silently wrong.

---

## 2. Worked example

`NVFP4_CB_K12`, `product` mode, one row, `in_features = 256` (one superblock),
`n_sub = 2`, `bit_split(12, 2) = (6, 6)`. Each codeword = `sub0 (6 bits) |
(sub1 (6 bits) << 6)`.

Vector 0 picks `(sub0, sub1) = (5, 3)` → codeword `c0 = 5 | (3 << 6) = 197`
(binary `0b0011000101`, 12 bits). The stream's first 12 bits are the LSB-first
bits of 197. Codewords are **not** byte-aligned (only the `4k`-byte index region
is): byte 1's low 4 bits finish `c0` and its high 4 bits begin `c1`.

Index region = `4 * 12 = 48` bytes. Scale plane (v1) = 16 bytes →
`type_size = 64` → `cb_qweight` shape `(1, 64)`.

To decode vector 0: read its 12 stream bits → `197`; `sub0 = 197 & 63 = 5`,
`sub1 = (197 >> 6) & 63 = 3`; codeword = `[sub_cb0[5] (4 coords) | sub_cb1[3]
(4 coords)]`; multiply each coordinate by its group-16 scale.

---

## 3. Decode reference recipe

```
for each row, each superblock s:
  base       = s * type_size
  idx_bytes  = qweight[row, base : base + 4k]
  codewords  = unpack LSB-first -> 32 integers of k bits each
  for v in 0..31:
    code = codewords[v]
    if mode == full:     cw = codebook[code]                          # 8 coords
    if mode == product:  cw = concat( sub_cb[i][ (code >> off_i) & ((1<<b_i)-1) ]  for i in 0..n_sub-1 )
    for coord j in 0..7:
      w_idx = s*256 + v*8 + j
      if grid == fp4 and layout_version == 1:  scale = e4m3( qweight[row, base + 4k + group16(v,j)] )
      if grid == fp4 and layout_version == 2:  scale = T[sub_code(v,j)] * 2^(super - 127)
      if grid == fp8:                          scale = weight_scale[row]
      weight[row, w_idx] = cw[j] * scale
```

where `group16(v,j) = (v*8 + j) // 16` and `super`, `sub_code` are read from the
9-byte v2 plane. A conforming decode **MUST** reproduce the encoder's rendered
weights bit-for-bit.

---

## 4. Tensor names (safetensors)

For each CB target Linear `<q>` (e.g. `model.layers.0.mlp.gate_proj`):

| Tensor | dtype / shape | Families | Meaning |
|---|---|---|---|
| `<q>.cb_qweight` | `uint8` `(rows, (in/256)*type_size)` | all | §1 superblock byte stream |
| `<q>.weight_scale` | `fp32` `(rows,)` | **fp8 only** | per-output-channel scale (fp4 scales live inside `cb_qweight`) |

**Stacked packed experts** (a 3-D source weight `(E, out, in)` — e.g. a fused
MoE `experts.gate_up_proj` / `experts.down_proj`): the expert axis stays explicit.

| Tensor | dtype / shape |
|---|---|
| `<q>.cb_qweight` | `uint8` `(E, out, (in/256)*type_size)` — expert `e` is `cb_qweight[e]`, laid out exactly as the 2-D case |
| `<q>.weight_scale` | `fp32` `(E, out)` — fp8 only |

Legacy packed-MoE stacks without a `per_expert_format_groups` declaration
**MUST** share one format and one codebook; the versioned split-stack schema is
the explicit partitioned exception. Dense siblings which vLLM presents as one
merged output (q/k/v, gate/up, and architecture-specific merges) remain
independent Linears and **MAY use different Gridbook or declared source-native
formats**. The runtime loads each role through its own method and concatenates
the output in vLLM's declared shard order when their representation or physical
activation metadata differs. A compatible same-representation merge MAY keep
one method, but only after contracted role scalars are proven equal. A producer
MUST preserve the individual role tensors and metadata; it MUST NOT coerce
their allocation to a single format or scalar merely because vLLM constructs
one `MergedColumnParallelLinear`. A runtime without a loader wired for private
role carriers MUST reject the model as an unsupported loader integration; it
MUST NOT describe the independent formats themselves as invalid.

**Codebook sidecar.** Codebooks are shipped **once per `(ref, format)`**, never
per tensor, in a sidecar file `cb_codebooks.pqcb`. The `.pqcb` file **IS a
safetensors file** (safetensors container, `.pqcb` extension) — it **MUST NOT** be
loaded via pickle/`torch.load`.

| Tensor | dtype / shape | Meaning |
|---|---|---|
| `cb_codebook.<ref>.<fmt>` | `fp16` `(2^K, 8)` | `full` codebook (`K = k`) |
| `cb_codebook.<ref>.<fmt>.sub{i}` | `fp16` `(2^b_i, 8/n_sub)` | `product` sub-codebook `i` |

- `<ref>` is `lattice` (a deterministic fixed lattice — no per-tensor sidecar
  entries beyond the shared table) or a role name (a shared per-role learned
  codebook, e.g. `gate_proj`, pooled across layers and, for MoE, across experts).
- `<fmt>` is the rung name (`NVFP4_CB_K16`, `FP8_CB_K44`, …).
- Codebook values are grid-valued and **exact in fp16** for both grids. A runtime
  **MAY** re-pack them to 4-bit (fp4) / 8-bit (fp8) codes at load — this is a tiny
  table transform, not a resident weight expansion.

All **non-target tensors** (norms, embeddings, `lm_head`, BF16-assigned Linears)
are copied verbatim (BF16 passthrough) and their module names **MUST** appear in
the config `ignore` set (§5).

### 4.1 External codebook provenance (`codebook_sha256`)

Wrong codebook **values** are not detectable at decode time: a `k`-bit codeword
indexes a `2^k`-row table (§1.1), so every index is in range by construction and
a sidecar with the right names and shapes but the wrong numbers — a stale
sidecar, one from a different checkpoint at the same rung, or length-preserving
data corruption — decodes to a correctly-shaped tensor of structured garbage
rather than raising. (Structural damage is already caught: a byte-truncated
safetensors file fails to deserialize, and a table missing by name fails at
lookup.)

The full quant config therefore **SHOULD** carry an artifact-external digest for
every table in `cb_codebooks.pqcb`:

```json
"provenance": {
  "codebook_sha256": {
    "cb_codebook.lattice.NVFP4_CB_K16.sub0": "<64 lowercase hex characters>",
    "cb_codebook.lattice.NVFP4_CB_K16.sub1": "<64 lowercase hex characters>"
  }
}
```

The expected values live in `quant_config.json`, not in safetensors metadata
inside the file they describe. A self-declared digest cannot distinguish the
intended sidecar from a different intact sidecar carrying its own matching
declaration.

**Construction (normative, per table).** Convert the table to IEEE-754
binary16 (`fp16`), contiguous row-major (C) order, serialize its raw
little-endian bytes with no framing, and compute SHA-256 over those bytes. This
is the construction used by the PrismaQuant exporters:

```python
import hashlib, torch
def codebook_sha256(table: torch.Tensor) -> str:
    raw = table.to(torch.float16).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()
```

The tensor name is the key in the mapping and is not absorbed into its digest.
Shape and dtype are validated elsewhere by the format contract; all conforming
sidecar tables are already `fp16` (§4). The explicit conversion is retained to
match existing exporter output byte-for-byte.

If `provenance.codebook_sha256` is present, a runtime **MUST** require it to
cover the sidecar's tensor names exactly, verify every digest before use, and
refuse a malformed, incomplete, or mismatched mapping. The field remains
**OPTIONAL** for backward compatibility: a config with no mapping **MUST** load
without this integrity check. An empty mapping is not the same as absence.

This binding detects accidental corruption and a sidecar swapped independently
of its config. It is not a signature: replacing both files with a mutually
consistent pair is outside its scope.

---

## 5. Configuration vocabulary (`config.json`)

The quantization config is compressed-tensors-**style** but uses a distinct
`quant_method` because compressed-tensors' scheme vocabulary cannot express
codebooks. A producer **MUST** write the canonical method name `"gridbook"`.

The primary published layout is a pointer stub embedded in
`config.json["quantization_config"]`, so vLLM can auto-detect Gridbook without
inlining the full assignment:

```jsonc
{
  "quant_method": "gridbook",             // REQUIRED for new artifacts
  "format": "nvfp4_cb",
  "config_file": "quant_config.json",      // default when omitted
  "codebook_file": "cb_codebooks.pqcb"     // default when omitted
}
```

`config_file` names the sidecar containing the full configuration. The default
is `quant_config.json`. A consumer **MUST** also accept the compatibility form
where that full configuration is inlined directly in
`config.json["quantization_config"]` (identified by the presence of
`config_groups`). The full configuration, whether referenced or inline, uses
the following vocabulary:

```jsonc
{
  "quant_method": "gridbook",             // canonical producer value (§6)
  "format": "nvfp4_cb",
  "layout_version": 2,                     // top-level; present only for v2 (fp4 two-tier). Absent => v1.
  "config_groups": {
    "group_0": {
      "targets": ["model.layers.0.mlp.gate_proj", "..."],   // module-name prefixes
      "format": "NVFP4_CB_K16",
      "scheme": {
        "grid": "fp4",                     // "fp4" | "fp8"
        "mode": "product",                 // "full" | "product" ("signed" removed 2026-08-23; MUST be refused)
        "k": 16,
        "superblock": 256,
        "group_size": 16,                  // fp4 group-16 scale; 0 for fp8
        "vec_dim": 8,
        "n_sub": 2,                        // product sub-count; 1 for full
        "type_size": 73,                   // bytes per 256-weight superblock (MUST match grid,k,version)
        "act_bits": 4,                     // 4 (fp4, W4A4) | 8 (fp8, W8A8)
        "codebook_source": "lattice",      // "lattice" | "learned"
        "codebook_ref": ["cb_codebook.lattice.NVFP4_CB_K16.sub0",
                         "cb_codebook.lattice.NVFP4_CB_K16.sub1"],
        "codebook_group": null,            // role name for learned; null for lattice
        // present for FP4 v2 groups ONLY (absence => v1 scale coding):
        "scale_coding": {
          "kind": "two_tier",
          "sub_bits": 4,
          "super_bias": 127,
          "table": [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
        }
      }
    }
  },
  // REQUIRED only when DSpark construction targets carry contracted FP4-CB
  // activation scalars under physical mtp.* checkpoint names:
  "dspark_target_bridge": {
    "schema": "gridbook.dspark-target-bridge.v1",
    "num_hidden_layers": 43,
    "n_mtp_layers": 3,
    "construction_to_physical": {
      "model.layers.43.ffn.experts.gate_up_proj":
        "mtp.0.ffn.experts.gate_up_proj"
    }
  },
  "ignore": ["model.norm", "lm_head", "..."],   // non-CB modules -> unquantized
  "provenance": {                                // RECOMMENDED, not required to decode
    "codebook_sha256": { "cb_codebook.lattice.NVFP4_CB_K16.sub0": "...", "...": "..." },
    "codebook_source": "lattice",
    "tensor_formats": { "model.layers.0.mlp.gate_proj": "NVFP4_CB_K16", "...": "..." }
  }
}
```

- `codebook_ref` is a single tensor name (`full`) or an ordered list
  `sub0..sub{n_sub-1}` (`product`).
- Targets sharing one `(codebook_ref, format)` **SHOULD** be grouped into one
  config group.
- `dspark_target_bridge` is an explicit namespace attestation, not a name
  heuristic. Its construction keys **MUST** be declared targets in
  `config_groups` and cover the complete activation contract (custom FP4-CB
  plus any delegated stock-NVFP4 activation target); its physical values
  **MUST** be exactly
  `execution_contracts.nvfp4_w4a4.target_names`. Every mapping is one-to-one,
  preserves the complete target tail, uses `model.layers.(L+s)` -> `mtp.s`,
  and satisfies `s < n_mtp_layers`. A consumer **MUST** reject an unknown
  schema, topology mismatch, unused/missing target, changed tail, or duplicate
  physical target. Before copying weights it **MUST** also compare stamped `L`
  and `n_mtp_layers` with the instantiated draft model. Targets outside that
  activation contract do not appear in this map; DSpark `main_proj` remains a
  delegated source-format special case and is excluded.

### 5.1 Per-role codebooks on routed expert stacks (since 0.8.3)

A learned codebook is fit per (layer, projection), so the three logical roles
of a routed expert stack — `gate`, `up`, `down` — may each want their own book
at the same rung. An artifact expresses that by naming the roles as separate
targets with different `codebook_ref`s:

```jsonc
"targets": ["model.layers.1.mlp.experts.gate_proj"],  // one group per book
"targets": ["model.layers.1.mlp.experts.up_proj"],
"targets": ["model.layers.1.mlp.experts.down_proj"],
```

Rules a consumer **MUST** apply when resolving one routed expert prefix:

- All targets under the prefix **MUST** agree on `grid`, `mode`, `k`, `n_sub`,
  `type_size` and `activation_contract`. Only `codebook_ref` may differ. (An
  exporter's serving-unit promotion already forces one format per routed
  stack, so this is a statement of what a valid artifact looks like, not a
  restriction on allocation.)
- A fused `gate_up_proj` target claims **both** the `gate` and `up` roles with
  one book. `down_proj` claims `down` alone; w2 is a one-role stack and never
  splits.
- If two targets claim the same role with different books, or a role is left
  without a book, the consumer **MUST refuse to load**. It **MUST NOT** fall
  back to another role's codebook.
- When every target names the same book the stack is *uniform*, and the
  consumer **MUST** behave exactly as it did before this section existed.

**Fail-open warning for consumers written against ≤0.8.2:** Gridbook 0.8.2
compared only the format tuple above — which excludes `codebook_ref` — and
then adopted the first target's scheme by sorted name, i.e. `down_proj`'s. A
per-role artifact loaded silently and decoded every stack with the wrong book.
Any consumer that resolves a routed stack from one arbitrary target has the
same defect.

Per-role books and per-expert format groups (§ `per_expert_format_groups`) are
**independent features that Gridbook does not compose**: a target of the form
`…experts.gate_proj.format_group_0` is refused at load. No known allocator
emits one, since routed layers take a single rung.
- A group is FP4-v2 iff it carries a `scale_coding` object; the top-level
  `layout_version: 2` signals that v2 groups exist. **Absence of `scale_coding`
  MUST be interpreted as v1.** Old (v1-only) artifacts therefore parse unchanged,
  permanently.

Provenance fields (git commit, calibration/codebook hashes, per-tensor format
map) are **RECOMMENDED** for reproducibility auditing but are not needed to
decode; a consumer **MUST NOT** require them.

---

## 6. Runtime registration and dispatch

**Registry keys.** The canonical vLLM quantization-method key is the exact string
`"gridbook"`. New artifacts **MUST** write this value in
`config.json["quantization_config"]["quant_method"]`. A conforming runtime
**MUST** register `"gridbook"` and **MUST** also accept `"prismaquant"` as a
read-only legacy alias so artifacts exported before the rename continue to load.
The alias selects the same configuration and dispatch implementation; producers
**MUST NOT** write it for new artifacts. Neither identifier implies a runtime
dependency on the producing project.

A conforming runtime dispatches per module prefix:

- prefix matches a group's `targets` → the CB method (decode per that group's
  `scheme`);
- prefix in `ignore` → the runtime's unquantized linear method;
- prefix is a plain NVFP4 / FP8 layer of a **mixed container** → **delegate to the
  runtime's stock `compressed-tensors` handling**. The CB config owns the CB
  groups and forwards everything else, so FP8/NVFP4 layers hit the runtime's own
  maintained kernels and the CB implementation never reimplements them.

Fused siblings and packed MoE experts are guaranteed uniform per group by the
producer (§4), so a runtime **MUST NOT** need per-shard scheme mixing within one
config group.

---

## 7. Serving invariants

Two invariants are normative for any *production-speed* serving implementation
(a correctness-only reference decoder MAY relax INV-2 but MUST honor INV-1):

- **INV-1 — no resident expansion.** The resident weight state **MUST** be the
  packed `cb_qweight` (index stream + scale plane) plus the shared codebook. A
  runtime **MUST NOT** materialize the dense `(rows, in_features)` weight (or a
  resident per-superblock FP32 scale plane) in memory. Decoding to native tiles
  **MUST** happen transiently (per-tile scratch in registers/shared memory or a
  per-layer buffer that is freed after the matmul). This is the property that
  makes a smaller-on-disk artifact also smaller-in-memory.
- **INV-2 — native tensor cores for prefill.** The production prefill path
  **SHOULD** feed decoded FP4/FP8 codes to the hardware tensor-core GEMM (the same
  path a plain NVFP4/FP8 layer uses). A decode-to-BF16-then-software-matmul path is
  a correctness fallback only and does not meet the prefill performance goal.

For a v2 (two-tier) artifact, a runtime **MUST** keep resident scale state at the
packed 9 bytes/superblock — reconstructing the 16-byte E4M3 plane resident (or an
FP32 plane) violates INV-1. Composition to E4M3 **MUST** occur inside the
transient tile expansion or the decode kernel's registers.

---

## 8. Extensibility

The format is designed to grow without breaking deployed artifacts. The
extension points, and what an implementation MUST honor, are:

**Adding a rate rung (new `k`).** A new `k` is fully described by
`(grid, mode, k, n_sub, type_size)` in the scheme; `type_size` follows the §1.4
formula. No layout change. A decoder that implements the §3 recipe already handles
any `k` for which it can hold/compute the codebook. Producers **SHOULD** pick `k`
so the codebook is servable (see the smem note below).

**Adding a value grid.** New grids (beyond FP4/E2M1 and FP8/E4M3) MAY be defined
by specifying the codeword value set, the activation width (`act_bits`), the
per-group scale representation, and `n_sub`/`d`. A new grid **MUST** publish its
`type_size` formula and its reconstruction rule, and **SHOULD** preserve the
"decoded tile is a native hardware tile" property that gives the family its
serving speed; a grid that requires software-only matmul is permitted but forfeits
the native-speed goal.

**Adding a scale coding.** New scale codings are declared with a new
`scale_coding.kind` and a bumped `layout_version`. The v1/v2 rule generalizes: a
consumer that does not recognize a `scale_coding.kind` **MUST** refuse the artifact
rather than guess. Absence of `scale_coding` **MUST** always mean v1 (E4M3-direct),
forever — this is the permanent backward-compatibility anchor. A new coding
**SHOULD** reconstruct to a native per-group scale representation (as v2 composes
to exact E4M3) so no downstream kernel surgery is needed.

**Adding an index mode.** `full` and `product` are the defined modes; `signed`
was removed on 2026-08-23 and MUST be refused (see §0). A new mode MAY be added
by specifying its `k`-bit codeword decode into 8
coordinates; it **MUST** keep the LSB-first superblock packing (§1.1) and the
256-weight superblock.

**What every conforming implementation MUST honor (the stable core):**

1. the 256-weight superblock and `in_features % 256 == 0` requirement;
2. LSB-first index packing (§1.1) and the `full`/`product` decode
   semantics (§1), plus the refusal of the removed `signed` mode;
3. `type_size = f(grid, k, layout_version)` and the version-keyed `effective_bits`
   (§1.4), asserted at pack/load;
4. v1 = E4M3-direct 16-byte plane; v2 = two-tier 9-byte plane composing to exact
   E4M3 (§1.2); FP8 = no plane + per-channel FP32 (§1.3);
5. the `cb_qweight` / `weight_scale` tensor names and the `cb_codebooks.pqcb`
   safetensors sidecar (§4);
6. the `config.json` vocabulary and the "absence of `scale_coding` ⇒ v1" /
   "unknown `scale_coding.kind` ⇒ refuse" rules (§5);
7. the canonical `"gridbook"` registry key and legacy `"prismaquant"` read alias
   (§6);
8. INV-1 for any serving implementation (§7).

**Implementation notes (non-normative but load-bearing for servability).** On the
reference GB10 hardware, a flat codebook table in shared memory is `2^k × 4` bytes
(FP4). Against the measured 99 KB opt-in shared memory per block, flat tables are
comfortable to `k ≤ 13` and marginal at `k = 14`; higher `k` **SHOULD** use a
structured/computed codebook (a small stored generator plus sign/permutation
decomposition, as the `product` mode provides) rather than a stored flat
table, or the fused prefill kernel is infeasible. This is a hardware constraint on
codebook design, not a format-layout constraint — the byte layout is identical
regardless.
