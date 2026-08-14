"""MXFP4-CB format dataclass and byte/bpw accounting.

This is the small research prototype for one plausible OCP-MXFP4-targeted
codebook representation.  It is deliberately NOT production Gridbook; it
lives under research/mxfp4_cb/ and is CPU-only.

Wire idea (mirrors gridbook SPEC §1 but targets MXFP4 element/scale grid):

* Superblock = 256 weights along K, vec_dim = 8 => 32 codewords per SB.
* Each codeword is k bits => index stream = 4*k bytes per SB (identical to
  Gridbook; LSB-first packing).
* Scale plane = 8 × E8M0 (UE8M0, bias 127) — one power-of-two scale per 32
  weights (MX block size).  No E4M3, no two-tier.
* type_size = 4*k + 8  bytes per SB.
* effective bits/weight = (type_size*8)/256 = k/8 + 0.25  (+ negligible
  codebook amortisation).
* Codeword values live on the E2M1 grid {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  exactly as NVFP4; reconstruction is codeword_value * E8M0_scale(block).

References:
  OCP MX spec v1.0 — MXFP4 E2M1 elements, E8M0 shared scale, block 32.
  Gridbook docs/SPEC.md §§1–1.4 for packing/type_size derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

SUPERBLOCK = 256
VEC_DIM = 8
CODEWORDS_PER_SB = SUPERBLOCK // VEC_DIM  # 32
MX_BLOCK = 32
SCALES_PER_SB = SUPERBLOCK // MX_BLOCK  # 8
E8M0_BIAS = 127
E2M1_GRID_MAX = 6.0

# E2M1 magnitude grid (sorted ascending)
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Plausible rung set — same k range as NVFP4_CB_K12..K24 for easy comparison,
# but any k with a servable codebook is structurally legal.
SUPPORTED_K = tuple(range(12, 25))  # 12..24 inclusive

# Modes match Gridbook SPEC §1.1; this prototype implements product (n_sub=2)
# and the 8-sign-bit signed layout.  Full (n_sub irrelevant) is also legal.
SUPPORTED_MODES = ("product", "signed", "full")
N_SUB_FOR_MODE = {"product": 2, "signed": 1, "full": 1}
# For the 'full' mode n_sub is conceptually 1 but the flat table is 2^k × 8.


def bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    """Even split of k bits into n_sub parts, larger first (SPEC 1.1)."""
    if n_sub <= 0:
        raise ValueError(f"n_sub must be positive, got {n_sub}")
    base, extra = divmod(k, n_sub)
    return tuple(base + (1 if i < extra else 0) for i in range(n_sub))


def type_size_for_k(k: int) -> int:
    """Bytes per 256-weight superblock for MXFP4-CB."""
    return 4 * int(k) + SCALES_PER_SB  # 4k index + 8 scale


def effective_bpw(k: int) -> float:
    """Bits per weight including MX scale plane."""
    return type_size_for_k(k) * 8 / SUPERBLOCK


@dataclass(frozen=True)
class Mxfp4CbFormat:
    """Explicit schema for the MXFP4-CB hypothesis.

    Attributes are the minimal wire description; ``type_size`` and ``effective_bpw``
    are derived, not stored, but validated.  The dataclass is intentionally
    strict so malformed configs fail at construction, not at decode.
    """

    k: int
    mode: Literal["product", "signed", "full"] = "product"
    n_sub: int | None = None
    superblock: int = SUPERBLOCK
    vec_dim: int = VEC_DIM
    group_size: int = MX_BLOCK  # MX block = 32
    scale_coding: str = "e8m0_per32"
    grid: str = "e2m1"
    scale_grid: str = "e8m0"
    codebook_source: str = "lattice"  # or "learned"

    def __post_init__(self):
        if self.k not in SUPPORTED_K:
            # Not a hard error for research, but flag narrowly: allow any 1..24
            # for shakedown while warning on published rung set.
            if not (1 <= self.k <= 24):
                raise ValueError(f"k must be in 1..24 for prototype, got {self.k}")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {self.mode!r}")
        expected_n_sub = N_SUB_FOR_MODE[self.mode]
        # For product, n_sub forced to 2; for signed/full, 1.
        # Allow caller to omit n_sub (None) and infer; reject mismatches.
        if self.n_sub is None:
            object.__setattr__(self, "n_sub", expected_n_sub)
        elif self.n_sub != expected_n_sub:
            raise ValueError(
                f"mode {self.mode!r} requires n_sub={expected_n_sub}, got {self.n_sub}"
            )
        if self.superblock != SUPERBLOCK:
            raise ValueError(f"superblock must be {SUPERBLOCK}, got {self.superblock}")
        if self.vec_dim != VEC_DIM:
            raise ValueError(f"vec_dim must be {VEC_DIM}, got {self.vec_dim}")
        if self.group_size != MX_BLOCK:
            raise ValueError(f"group_size must be {MX_BLOCK} for MXFP4, got {self.group_size}")
        if self.scale_coding != "e8m0_per32":
            raise ValueError(f"scale_coding must be 'e8m0_per32', got {self.scale_coding!r}")
        if self.grid != "e2m1":
            raise ValueError(f"grid must be 'e2m1', got {self.grid!r}")
        if self.scale_grid != "e8m0":
            raise ValueError(f"scale_grid must be 'e8m0', got {self.scale_grid!r}")

    # --- derived wire properties -----------------------------------------
    @property
    def index_bytes(self) -> int:
        return 4 * self.k

    @property
    def scale_bytes(self) -> int:
        return SCALES_PER_SB

    @property
    def type_size(self) -> int:
        return type_size_for_k(self.k)

    @property
    def bpw(self) -> float:
        return effective_bpw(self.k)

    @property
    def sub_widths(self) -> tuple[int, ...]:
        if self.mode == "product":
            return bit_split(self.k, 2)
        if self.mode == "signed":
            return (self.k - 8,)
        # full
        return (self.k,)

    def validate_weights_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 2:
            raise ValueError(f"MXFP4-CB weight must be 2-D [rows, K], got shape {shape}")
        rows, k_dim = shape
        if rows <= 0 or k_dim <= 0:
            raise ValueError(f"rows and K must be positive, got {shape}")
        if k_dim % self.superblock != 0:
            raise ValueError(
                f"K={k_dim} must be multiple of superblock {self.superblock}"
            )
        if k_dim % self.group_size != 0:
            raise ValueError(f"K={k_dim} must be multiple of MX block {self.group_size}")

    def packed_row_bytes(self, K: int) -> int:
        self.validate_weights_shape((1, K))
        return (K // self.superblock) * self.type_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type_size"] = self.type_size
        d["effective_bpw"] = self.bpw
        d["index_bytes"] = self.index_bytes
        d["scale_bytes"] = self.scale_bytes
        d["sub_widths"] = self.sub_widths
        return d

    def bpw_accounting(self) -> str:
        return (
            f"k={self.k} ({self.mode}, n_sub={self.n_sub})  "
            f"type_size={self.type_size} B/SB  "
            f"[{self.index_bytes}B index + {self.scale_bytes}B E8M0]  "
            f"bpw={self.bpw:.5f}  "
            f"(index {self.k/8:.3f} + scale {self.scale_bytes*8/SUPERBLOCK:.3f})"
        )


__all__ = [
    "SUPERBLOCK",
    "VEC_DIM",
    "CODEWORDS_PER_SB",
    "MX_BLOCK",
    "SCALES_PER_SB",
    "E8M0_BIAS",
    "E2M1_GRID_MAX",
    "E2M1_MAGNITUDES",
    "SUPPORTED_K",
    "Mxfp4CbFormat",
    "bit_split",
    "type_size_for_k",
    "effective_bpw",
]
