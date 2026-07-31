"""Self-contained (no `prismaquant` import at serve time) CB codec helpers:

* load-time preprocessing that turns the shipped layout (LAYOUT.md) into the
  small resident tensors the Triton kernel consumes — the flat codebook, the
  pre-decoded fp4 scale plane, and an 8-byte-padded index stream. None of these
  is a dense [N,K] weight (INV-1 holds);
* activation QDQ that reproduces the emulation gate's served-activation buckets
  (fp4 group-16 RTN / fp8 dynamic per-token) so served KL is comparable to the
  emulated prediction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VEC_DIM = 8
SUPERBLOCK = 256
FP4_GROUP = 16
_E4M3 = torch.float8_e4m3fn
NVFP4_GRID_MAX = 6.0
FP8_ELEMENT_MAX = 448.0

# E2M1 element grid (sorted ascending), for the fp4 activation RTN.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Two-tier v2 scale coding (docs/lanes/nvfp4-cb/two-tier-scale-spec.md §1). Kept
# in sync with prismaquant.nvfp4_cb_formats (the reference the kernel matches).
SCALE_CODING_TWO_TIER = "two_tier"
TWO_TIER_SUPER_BIAS = 127
TWO_TIER_SUB_TABLE = (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)


def type_size(k: int, is_fp4: bool) -> int:
    return 4 * int(k) + (16 if is_fp4 else 0)


def build_flat_codebook(sub_tables: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate product sub-tables (each (2^w, sub_dim)) into the flat layout
    the kernel gathers from: block ``s`` = ``sub_tables[s].reshape(-1)`` (row
    major, so entry (idx, local) sits at idx*sub_dim + local)."""
    return torch.cat([t.reshape(-1).to(torch.bfloat16).contiguous()
                      for t in sub_tables]).contiguous()


def build_compose_table(sub_table) -> torch.Tensor:
    """Two-tier v2 (docs/lanes/nvfp4-cb/two-tier-scale-spec.md §1): the (256,16)
    compose table ``T[c]·2^(E-127)`` flattened to (4096,) fp32, bit-exact to
    ``nvfp4_cb_formats._two_tier_tables`` (float64 product -> fp32). The kernel
    gathers ``compose[super_e*16 + code]`` per group — no resident fp32 plane
    (spec §4/G4), just this 16 KiB constant table."""
    T = torch.tensor(list(sub_table), dtype=torch.float64)
    exps = torch.arange(256, dtype=torch.float64)
    compose = (T[None, :] * torch.pow(2.0, exps[:, None] - 127.0)).to(
        torch.float32)                                # (256, 16)
    return compose.reshape(-1).contiguous()           # (4096,)


def decode_fp4_scale_plane(qw: torch.Tensor, k: int) -> torch.Tensor:
    """(N, n_sb*type_size) uint8 -> (N, n_sb*16) fp32 group-16 scales, decoded
    from the E4M3 scale plane that follows each superblock's 4k index bytes."""
    n, row_bytes = qw.shape
    ts = type_size(k, is_fp4=True)
    n_sb = row_bytes // ts
    blk = qw.reshape(n, n_sb, ts)
    plane = blk[:, :, 4 * k:4 * k + FP4_GROUP].contiguous()      # (N, n_sb, 16)
    return plane.view(_E4M3).to(torch.float32).reshape(n, n_sb * FP4_GROUP)


PAD_BYTES = 16


def pad_qweight(qw: torch.Tensor) -> torch.Tensor:
    """Right-pad each row by ``PAD_BYTES`` so the padded buffer satisfies BOTH
    invariants every consumer of a padded row depends on:

    1. **>= 8 bytes of read slack per row.** The decode/expand kernels extract a
       codeword with an 8-byte window anchored at the codeword's first byte, so
       the last codeword of the last superblock of the last row reads up to 7
       bytes past the packed data. Without the slack that is an out-of-bounds
       global read (illegal-memory-access, or silent garbage in the final
       output rows).
    2. **The padded row stride stays a 16-byte multiple** whenever the UNPADDED
       one was. The fp8 CUTLASS entries (``cb_fused_prefill_mm_scaled``, the
       persistent-TC prefill) take the row stride explicitly and TORCH_CHECK
       ``stride(0) % 16 == 0`` — TMA needs 16-byte-aligned row starts. Every fp8
       rung has ``type_size = 4k`` in {112,128,144,160,176,192}, so
       ``row_bytes`` is 16-aligned and ``row_bytes + 16`` still is; the old
       ``+ 8`` pad was NOT, which is why the padded buffer could not be shared
       with the registered ``cb_qweight`` parameter and both had to stay
       resident (see ``linear.process_weights_after_loading``). Pad width is
       therefore load-bearing, not a spare-bytes choice: dropping it back to 8
       re-breaks the fp8 prefill kernels' alignment check.

    (fp4 rungs carry an odd ``type_size`` — ``4k+16`` v1, ``4k+9`` v2 — so their
    row stride is not 16-aligned either way; they only ever hit kernels that
    take the stride explicitly and require ``stride(1) == 1``.)

    Consumers must read the row stride from the tensor (``.stride(0)``), never
    derive it as ``size(1) - PAD_BYTES``.
    """
    return F.pad(qw.contiguous(), (0, PAD_BYTES), value=0).contiguous()


# Signed E2M1 grid, cached per device: building it per call allocated a CPU
# tensor and H2D-copied it on EVERY fp4 activation QDQ — a hidden sync in the
# eager decode hot path, and a hard error under CUDA-graph capture (unpinned
# CPU->CUDA copy). Warmup forwards populate the cache before any capture.
_FP4_QDQ_GRID: dict = {}


def _fp4_qdq_grid(device: torch.device) -> torch.Tensor:
    grid = _FP4_QDQ_GRID.get(device)
    if grid is None:
        grid = torch.tensor(sorted({v for a in _E2M1 for v in (a, -a)}),
                            dtype=torch.float32, device=device)
        _FP4_QDQ_GRID[device] = grid
    return grid


def fp4_group16_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """W4A4 activation bucket: RTN to E2M1 at group-16 amax/6 scale (mirrors
    format_registry `_make_rtn('fp4_e2m1', 16)`)."""
    in_f = x.shape[-1]
    grid = _fp4_qdq_grid(x.device)
    w = x.reshape(-1, in_f).float().reshape(-1, in_f // FP4_GROUP, FP4_GROUP)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / NVFP4_GRID_MAX
    xg = w / scale
    idx = torch.bucketize(xg.contiguous(), grid)
    lo = grid[(idx - 1).clamp_min(0)]
    hi = grid[idx.clamp_max(grid.numel() - 1)]
    q = torch.where((hi - xg).abs() < (xg - lo).abs(), hi, lo)
    return (q * scale).reshape(x.shape).to(x.dtype)


# ---------------------------------------------------------------------------
# fp4-MMA fused-prefill helpers (docs/lanes/nvfp4-cb/fp4-fused-prefill.md).
# The fused kernel consumes the CB codebook as E2M1 NIBBLE CODES (the fp4
# codebook is e2m1-grid-valued by construction, nvfp4_cb_formats._snap_to_grid,
# so this re-encoding is lossless) and the two-tier compose table as E4M3
# BYTES (exact by construction, two-tier-scale-spec.md §1.2).
# ---------------------------------------------------------------------------

def fp4_e2m1_codes(t: torch.Tensor) -> torch.Tensor:
    """E2M1 nibble code (0..15, bit3 = sign) of each element of an
    e2m1-grid-valued tensor. Raises if any value is off-grid."""
    mag = torch.tensor(_E2M1, dtype=torch.float32, device=t.device)
    x = t.to(torch.float32)
    idx = (x.abs().unsqueeze(-1) == mag).float().argmax(dim=-1)
    if not bool((mag[idx] == x.abs()).all()):
        raise ValueError("fp4_e2m1_codes: value off the E2M1 grid")
    return (idx + torch.where(x < 0, 8, 0)).to(torch.uint8)


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """(..., K) uint8 nibble codes -> (..., K/2) packed bytes (even element =
    low nibble — the cute little-endian sub-byte convention)."""
    lo = codes[..., 0::2].to(torch.int16)
    hi = codes[..., 1::2].to(torch.int16)
    return (lo | (hi << 4)).to(torch.uint8)


def build_fp4_value_lut(cb_flat: torch.Tensor, k_bits: int,
                        n_sub: int) -> torch.Tensor:
    """The fused kernel's smem value LUT, as a flat uint8 tensor.

    product (n_sub=2): two ceil-first sub-tables of 4-dim vectors; each entry
    becomes a u16 of 4 nibble codes (LSB-first) -> bytes = (2^w0 + 2^w1) * 2.
    signed (n_sub=1): 2^(k-8) positive 8-dim magnitude vectors; each entry a
    u32 of 8 nibble codes -> bytes = 2^(k-8) * 4. Max 16 KiB (k24 product).
    """
    if n_sub == 2:
        w0 = k_bits - k_bits // 2
        w1 = k_bits // 2
        t0 = cb_flat[:(1 << w0) * 4].reshape(-1, 4)
        t1 = cb_flat[(1 << w0) * 4:(1 << w0) * 4 + (1 << w1) * 4].reshape(-1, 4)
        out = []
        for t in (t0, t1):
            c = fp4_e2m1_codes(t).to(torch.int32)
            u16 = c[:, 0] | (c[:, 1] << 4) | (c[:, 2] << 8) | (c[:, 3] << 12)
            out.append(u16.to(torch.int16).view(torch.uint8).reshape(-1))
        return torch.cat(out).contiguous()
    if n_sub == 1:
        m = cb_flat[:(1 << (k_bits - 8)) * 8].reshape(-1, 8)
        c = fp4_e2m1_codes(m).to(torch.int64)
        u32 = torch.zeros(c.shape[0], dtype=torch.int64, device=c.device)
        for j in range(8):
            u32 |= c[:, j] << (4 * j)
        return (u32.to(torch.int32).view(torch.uint8).reshape(-1).contiguous())
    raise ValueError(f"fp4 value LUT: unsupported n_sub={n_sub}")


def build_compose_u8(sub_table=TWO_TIER_SUB_TABLE) -> torch.Tensor:
    """The (256*16) two-tier compose table as E4M3 BYTES. Legal (E, c) pairs
    are e4m3-exact by construction (spec §1.2); illegal pairs are never
    emitted by the encoder, so their bytes are clamped placeholders."""
    comp = build_compose_table(sub_table).clamp(0.0, FP8_ELEMENT_MAX)
    return comp.to(_E4M3).view(torch.uint8).contiguous()


def sf_swizzle_offsets(rows: int, groups: int,
                       device: torch.device) -> torch.Tensor:
    """(rows, groups) -> flat offsets of the CUTLASS Sm1xx 128x4 scale-factor
    atom layout (tile_atom_to_shape_SF*, Step<_2,_1,_3>): K-blocks fastest,
    then row-blocks. rows/groups are padded to 128/4 by the caller."""
    r = torch.arange(rows, device=device).unsqueeze(1)
    g = torch.arange(groups, device=device).unsqueeze(0)
    gpad = (groups + 3) // 4 * 4
    return ((r % 32) * 16 + ((r // 32) % 4) * 4 + (g % 4)
            + (g // 4) * 512 + (r // 128) * (512 * (gpad // 4)))


def swizzle_sf_plane(sf_bytes: torch.Tensor) -> torch.Tensor:
    """(rows, groups) uint8 e4m3 scale bytes -> the flat swizzled ue4m3 plane
    the block-scaled kernels consume (padded to 128 rows x 4 groups)."""
    rows, groups = sf_bytes.shape
    rpad = (rows + 127) // 128 * 128
    gpad = (groups + 3) // 4 * 4
    out = torch.zeros(rpad * gpad, dtype=torch.uint8, device=sf_bytes.device)
    off = sf_swizzle_offsets(rows, groups, sf_bytes.device)
    out[off.reshape(-1)] = sf_bytes.reshape(-1)
    return out


def nvfp4_act_quant_ref(x2: torch.Tensor, global_scale: torch.Tensor):
    """Torch reference of native NVFP4 activation quantization (the fused
    kernel's activation bucket): per-tensor fp32 global scale x per-group-16
    e4m3 SF, e2m1 RTN data. Returns (packed [M, K/2] uint8, sf_bytes
    [M, K/16] uint8, recip fp32 scalar tensor). NOTE: this is deliberately
    NOT codec.fp4_group16_act_qdq — the hardware SF operand is ue4m3, so the
    Triton path's fp32 group scale is unrepresentable (see the lane doc)."""
    M, K = x2.shape
    xs = x2.float() * global_scale
    g = xs.reshape(M, K // FP4_GROUP, FP4_GROUP)
    sf = (g.abs().amax(dim=-1) / NVFP4_GRID_MAX).clamp(
        2.0 ** -9, FP8_ELEMENT_MAX).to(_E4M3)
    sf_f = sf.to(torch.float32)
    grid = _fp4_qdq_grid(x2.device)
    xg = (g / sf_f.unsqueeze(-1)).clamp(-NVFP4_GRID_MAX, NVFP4_GRID_MAX)
    idx = torch.bucketize(xg.contiguous(), grid)
    lo = grid[(idx - 1).clamp_min(0)]
    hi = grid[idx.clamp_max(grid.numel() - 1)]
    q = torch.where((hi - xg).abs() <= (xg - lo).abs(), hi, lo)
    codes = fp4_e2m1_codes(q.reshape(M, K))
    return (pack_e2m1_codes(codes), sf.view(torch.uint8),
            (1.0 / global_scale).to(torch.float32))


def fp8_dynamic_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """W8A8 activation bucket: vLLM dynamic per-token E4M3 (mirrors
    fp8_dynamic.fp8_dynamic_activation_qdq_vllm)."""
    rows = x.reshape(-1, x.shape[-1]).float()
    min_scale = 1.0 / (FP8_ELEMENT_MAX * 512.0)
    scale = (rows.abs().amax(dim=-1, keepdim=True) / FP8_ELEMENT_MAX
             ).clamp_min(min_scale)
    q = (rows / scale).clamp(-FP8_ELEMENT_MAX, FP8_ELEMENT_MAX).to(_E4M3)
    deq = q.to(torch.float32) * scale
    return deq.reshape(x.shape).to(x.dtype)
