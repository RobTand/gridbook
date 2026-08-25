# TCQ R256 research format

Status: **research-only**. The IDs below have no entry in
`runtime_contract.json`, no public producer or chooser, and no release or
device-qualification authority. The executable wire contract is
`gridbook.trellis.wire.v1` in `gridbook/trellis.py`; the isolated CUDA ABI is
schema 2 in `gridbook/csrc/trellis_r256.cu`.

## Rungs

The suffix is exact body bits per 256 weights, before scale, alphabet,
schedule, offset, header, and row-padding bytes. Artifact comparisons must use
`gridbook.trellis.account`, not the suffix alone.

| Family | Candidate rungs | Research-only boundary | Native terminal |
|---|---|---|---|
| `TCQ_E2M1_R256` | `R384`, `R512`, `R640`, `R768`, `R896` (1.5–3.5 body bpw; approximately 2.0–4.0 including the group-16 scale plane) | every integer `R256`–`R1016`; shaped ceiling `R768`, then mixed bypass | NVFP4 4-bit scalar, not TCQ |
| `TCQ_E4M3_R256` | `R1152` (4.5 body bpw; row-scale overhead is shape-dependent) | every integer `R256`–`R2040`; shaped ceiling `R1792`, then mixed bypass | E4M3 8-bit scalar, not TCQ |

The candidate list is the measured quality decision, not the mathematical
surface and not producer authority. Stage-5 full-model KL plus weighted RD
keeps E2M1 through the approximately 4-bpw shoulder: E4M3 lost the matched
weighted objective at 3.0, 3.5, and 4.0 body bpw, and its small KL advantage
at 3.5 came with worse PPL on both slices. E4M3 owns only the measured 4.5
shoulder, where it beat FP8-CB in both KL and PPL. Its 5.0 and 6.0 screens lost
to FP8-CB and are not candidates. Every integral q256 value inside the stated
research bounds remains legal for experiments so future per-Linear allocation
does not inherit artificial half-bit holes. E2M1 rates above 3 and E4M3 rates
above 7 use native-code bypass; they test mixed schedules but do not pretend
that a coded bit enlarges the hardware alphabet.

## Wire

- 256-state tail-biting convolutional code, octal generators `(561, 753)`,
  memory order 8. Every physical block must contain at least eight coded
  positions, including a short final block.
- One tensor-shared schedule with one nibble per input column. `tight_offsets`
  permits a global RWF budget and serializes one block-offset table.
  `fixed_quota_per_256` permits a different importance-directed arrangement in
  each block, requires the same exact bit quota in every complete block, and
  derives fixed-stride block offsets rather than serializing them.
- Tight LSB-first body packing. Blocks may start at non-byte-aligned bit
  offsets. `block_offset_bits` is tensor-shared. Rows alone are padded to a
  16-byte stride; padding bits and bytes must be zero.
- Per-rate alphabets are sorted native code bytes/nibbles and are
  Ungerboeck-partitioned by striding positions `j, j+4, ...`. Their canonical
  serialized bytes are SHA-256 bound in the header. E4M3FN bytes `0x7f` and
  `0xff` are refused everywhere; the full rate-7 alphabet replaces those two
  NaN slots with duplicate signed-zero slots.
- The header digest binds only the alphabet section. It is a deterministic
  asset-identity check, not authentication or a checksum for the schedule,
  scales, or packed body; a containing artifact must bind the complete file.
- E2M1 carries one E4M3FN scale byte per row/group16 plus a positive FP32
  `global_scale_real`. E4M3 carries one positive FP32 scale per row.
- `account()` reports body bits, physical body bytes, row padding, every side
  section, total bytes, and side-inclusive exact bpw.

## Kernel and serving boundary

`trellis_r256_decode_codes` reconstructs exact native E2M1 nibbles or finite
E4M3 bytes. `trellis_r256_expand` is an explicit transient correctness tool.
`trellis_r256_dequant_gemv` decodes in-register and never constructs an
expanded resident weight matrix. Mixed rates, bypass, `T'=8`, tails, and
non-byte-aligned block boundaries share the same ABI.

`prepare_wire_cuda` clones and validates a wire once, then owns its tensors
and scalar contract privately. Its graph-capturable `_out` operation writes a
preallocated native-code tile without a per-call host synchronization: two
E2M1 codes per byte, or one finite E4M3 code per byte. The serialized scale
plane stays separate. This is only a code-tile handoff prototype; it does not
expose a scale-aware tensor-core consumer.

These operations establish decode correctness, not a production lane:

- **INV-1:** serialized/model-resident state remains packed. A test expansion
  must be discarded and cannot become a cached parameter or HBM weight plane.
- **INV-2:** production prefill must decode into native FP4/FP8 codes and scales
  consumed by the appropriate tensor-core mainloop. The current FP32 research
  GEMV and transient expander do not satisfy that gate. Long-prefill,
  medium-prefill, graph-capture, bandwidth, and real-device qualification are
  mandatory before any runtime-contract entry.

The fixed-quota layout is included now because it removes per-row schedule
branching and gives future prefill tiles stable rate counts. It is not evidence
that the missing native-MMA mainloop exists or performs acceptably.

The one permitted performance redesign was negative on GB10. At a
`1024 x 4096` shape, E4M3 `R1152` native-code expansion took 0.2262 ms versus
0.01872 ms for matched `FP8_CB_K36`, about 12.1 times slower. E2M1 `R512`
took 0.5706 ms versus 0.06877 ms for `NVFP4_CB_K18`, but those outputs are not
semantically comparable because the existing CB bridge emits scaled BF16.
These are synthetic kernel diagnostics, not HBM or serving qualification.
The preserved evidence is `stage_kernel_bench.json` in the campaign run root;
the servability and bandwidth gates remain failed.
