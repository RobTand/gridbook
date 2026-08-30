# QTIP-derived online transforms around native NVFP4

Status: **research-only, opt-in, and not runtime-contract eligible**. This is
an execution ABI for experiments, not a production format or a serving claim.
The method lineage is QTIP. EXL3 is not a dependency or selectable runtime.

The implementation extends the existing `TCQ_E2M1_R256` W4A4 lane. It does
not add a weight carrier, cache, decoder, or GEMM: the trellis wire still
decodes through its existing resident/streamed path into native E2M1 codes and
E4M3 group-16 scales, and the multiplication remains Blackwell
`torch._scaled_mm`. The only new work is an input transform before native FP4
activation quantization and an inverse output transform after the native GEMM.

## Algebra

For column-vector convention `y = W x`, each side uses a block-diagonal
normalized Sylvester Hadamard `H` and a diagonal Rademacher matrix `D`:

```
R_in  = H_in  D_in
R_out = H_out D_out
Q ~= R_out W R_in^T
y_hat = R_out^T Q R_in x
```

vLLM batches vectors as rows. The runtime therefore executes `x D_in H_in`
before quantization and `z H_out D_out` after `_scaled_mm`. The bias is added
only after the inverse output transform, in the original output basis.

Hadamards are block-diagonal because many model dimensions are not powers of
two. Every block size is a power of two and must divide its dimension exactly;
schema v1 permits no padding. This is still an orthogonal transform, but it is
not asserted to have the same quality as QTIP's full implementation.

## Sidecar ABI

An E2M1 trellis `scheme` may carry this additional object:

```jsonc
"online_transform": {
  "schema": "gridbook.qtip-online-hadamard.v1",
  "algorithm": "block_walsh_hadamard",
  "normalization": "orthonormal",
  "padding": "none",
  "transform_sha256": "<64 lowercase hex characters>",
  "input": {
    "dimension": 4096,
    "block_size": 256,
    "seed": 11,
    "sign_generator": "sha256_counter_rademacher",
    "sign_sha256": "<64 lowercase hex characters>"
  },
  "output": {
    "dimension": 4096,
    "block_size": 256,
    "seed": 29,
    "sign_generator": "sha256_counter_rademacher",
    "sign_sha256": "<64 lowercase hex characters>"
  }
}
```

The complete field set is closed: missing and unknown fields are errors. Input
dimension must equal wire columns; output dimension must equal wire rows.
Seeds are unsigned 64-bit integers. The family must be `TCQ_E2M1_R256`.

`transform_sha256` binds the canonical JSON (sorted keys, compact separators,
ASCII, no non-finite numbers) of every root field except itself. It therefore
binds the schema, algorithm, normalization, padding, both dimensions and block
sizes, seeds, generators, and sign digests. The containing artifact must still
bind its config; this self-carried digest detects metadata drift, not a swap of
a complete mutually-consistent artifact. The producer helper is
`gridbook.qtip_hadamard.online_transform_digest(contract)`.

`gridbook.qtip_hadamard.seeded_sign_digest(role, dimension, seed)` is the
producer helper. Its language-independent construction is:

1. Set `domain = ASCII("gridbook.qtip-online-hadamard.v1/signs\\0")`.
2. For counter `j = 0, 1, ...`, append
   `SHA256(domain || ASCII(role) || 0x00 || N_le64 || seed_le64 || j_le64)`.
3. Consume bits bytewise, least-significant bit first; bit 0 is `+1`, bit 1 is
   `-1`. Truncate to `N` bits and clear unused high bits in the final byte.
4. `sign_sha256` is SHA-256 of that packed byte vector.

The pinned conformance vector is:

```
role=input, dimension=19, seed=0x0123456789abcdef
c9850b2a7c2d365cdb23964ff33c6e934fd3dde47bedb07b35eb9ba8823f6368
```

Gridbook regenerates and verifies both digests while parsing the sidecar. It
also compares the sidecar geometry to the loaded wire. A changed seed,
dimension, block size, normalization, algorithm, padding rule, digest, wire
shape, or family refuses before the first forward.

## Runtime boundary

The base lane still requires both `GRIDBOOK_TRELLIS_E2M1=1` and an explicit
`GRIDBOOK_TRELLIS_E2M1_MODE={resident,streamed}`. A transformed artifact also
requires:

```
GRIDBOOK_QTIP_ONLINE_HADAMARD_RESEARCH=1
```

The flag is latched. If metadata declares a transform and the flag is absent,
method construction fails rather than multiplying a transformed weight in the
wrong basis. Setting the flag on an artifact without `online_transform` does
not change the old lane: no sign buffers are created and neither transform is
called.

Signs are small deterministic buffers created at load. They do not replace or
duplicate the trellis decode residency. The current FHT is a transparent torch
reference: it computes stages in FP32, rounds the transformed activation to
BF16 before the native FP4 quantizer, and rounds the inverse-transformed result
to BF16 before bias. There is no BF16 GEMM fallback and no PrismaQuant import.

## Evidence and promotion boundary

`tests/test_qtip_hadamard.py` pins the sign generator, fail-closed schema, row
orientation, and end-to-end orthogonal algebra against explicit matrices.
`tests/test_trellis_dispatch.py` gates config dispatch and opt-in refusal.
`tests/test_trellis_e2m1_lane.py` drives both resident and streamed native W4A4
paths, proving the quantizer receives the transformed activation and the
post-GEMM result receives the inverse output transform.

The torch FHT allocates intermediates and launches one elementwise stage per
`log2(block_size)`. It is not an opaque custom op, performant FHT, or
CUDA-graph-qualified implementation. Consequently this ABI has no
`runtime_contract.json` format row or lane-eligibility cell and must not be
described as production serving support.

`scripts/bench_qtip_hadamard.py` records CUDA-event samples and a
`torch.profiler` trace for the reference implementation. Promotion requires a
native graph-safe FHT/fusion, before/after in-process and box telemetry,
work-per-joule, exact-artifact load and graph replay, matched quality, and
prefill/decode throughput against the untransformed W4A4 lane.
