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

# Two-tier v2 scale coding (docs/nvfp4-cb-plan/two-tier-scale-spec.md §1). Kept
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
    """Two-tier v2 (docs/nvfp4-cb-plan/two-tier-scale-spec.md §1): the (256,16)
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
