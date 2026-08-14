"""CPU/static gates for the default-off native OMMA direct-fragment probe.

Nothing here imports CUDA or invokes nvcc.  The tests pin the numerical proof,
the CUTLASS atom-derived lane mapping, the isolated loader contract, and the
source-level refusal to masquerade a BF16 HMMA kernel as native FP4.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import struct

import pytest

cuda_ext = pytest.importorskip(
    "gridbook.cuda_ext", reason="gridbook not importable")

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "gridbook/csrc/cutlass_fork/sm120_cb_fused_fp4_mma.hpp"
SOURCE = ROOT / "gridbook/csrc/cb_fp4_direct_fragment_probe.cu"
FLAG = "PRISMAQUANT_CB_FP4_D2F_PROBE"


def _ue4m3_positive_values() -> set[Fraction]:
    """Exact positive E4M3 lattice serialized by the native NVFP4 packer."""
    values = {Fraction(0)}
    # Positive E4M3/UE4M3 bytes 0x00..0x7e.  exp=15,mant=7 is NaN;
    # exp=0 is denormal with quantum 2^-9; normal bias is seven.
    for byte in range(1, 0x7F):
        exp = (byte >> 3) & 0xF
        mant = byte & 0x7
        if exp == 0:
            values.add(Fraction(mant, 1 << 9))
        else:
            power = exp - 10  # (8+mant)/8 * 2**(exp-7)
            value = Fraction(8 + mant)
            value = value * (1 << power) if power >= 0 else value / (1 << -power)
            values.add(value)
    return values


def test_arbitrary_fp32_group16_scales_are_not_exactly_representable():
    """One FP32 row residual cannot repair arbitrary per-group UE4M3 loss.

    For two groups the residual cancels, so exactness requires the ratio of
    their quality scales to equal a ratio of two UE4M3 values.  BF16 maxima
    1 and the next BF16 value 129/128 yield the scale ratio 129/128 after the
    shared `/6`; that ratio is absent from the complete UE4M3 ratio lattice.
    """
    lattice = _ue4m3_positive_values() - {Fraction(0)}
    native_ratios = {left / right for left in lattice for right in lattice}
    assert Fraction(129, 128) not in native_ratios

    # Pin the source's actual float32 `/ 6`, not only the ideal rationals.
    # The rounded scales are 11184811/2^26 and 43/256; their ratio is still
    # outside the complete hardware-scale ratio set.
    as_f32 = lambda value: struct.unpack(  # noqa: E731 - local exact oracle
        "<f", struct.pack("<f", float(value)))[0]
    scale0 = Fraction.from_float(as_f32(Fraction(1, 6)))
    scale1 = Fraction.from_float(as_f32(Fraction(129, 128 * 6)))
    assert scale0 == Fraction(11184811, 67108864)
    assert scale1 == Fraction(43, 256)
    assert scale1 / scale0 not in native_ratios

    # The single-group example also shows the direct narrowing loss: the
    # shipping scale for amax=1 is 1/6, between adjacent UE4M3 values.
    assert Fraction(1, 6) not in lattice
    assert Fraction(5, 32) in lattice
    assert Fraction(11, 64) in lattice
    assert Fraction(5, 32) < Fraction(1, 6) < Fraction(11, 64)


def test_sm120_atom_lane_mapping_yields_aligned_codewords():
    """Derive BLayout/SFBLayout coordinates, do not trust hand-written lore.

    CUTLASS declares BLayout Shape((4,8),(8,2)),
    Stride((64,1),(8,256)). CUTE's first mode is fastest and the codomain is
    flattened (N8,K64), hence n=lane//4 and k=8*(lane%4) (+32 for reg1).
    Every B register is exactly one serialized eight-value CB codeword.
    """
    seen = set()
    for lane in range(32):
        t0, t1 = lane % 4, lane // 4
        thread_offset = 64 * t0 + t1
        n = thread_offset % 8
        k_thread = thread_offset // 8
        assert n == lane // 4
        assert k_thread == 8 * (lane % 4)
        for reg, value_offset in enumerate((0, 256)):
            flat = thread_offset + value_offset
            n_reg, k_reg = flat % 8, flat // 8
            assert n_reg == n
            assert k_reg == k_thread + 32 * reg
            assert k_reg % 8 == 0
            seen.add((n_reg, k_reg))
    assert seen == {(n, k) for n in range(8)
                    for k in (0, 8, 16, 24, 32, 40, 48, 56)}

    # SFBLayout Shape((4,8),64), Stride((0,1),8): groups of four lanes
    # share N and each 32-bit VS=16 register carries K groups 0,16,32,48.
    for lane in range(32):
        n = (0 * (lane % 4) + lane // 4) % 8
        assert n == lane // 4
        assert (0, 16, 32, 48) == tuple(range(0, 64, 16))


def test_probe_is_native_omma_and_populates_register_fragments():
    header = HEADER.read_text()
    source = SOURCE.read_text()
    assert "SM120_16x8x64_TN_VS" in header
    assert "float_ue4m3_t" in source
    assert "true, ProbeKBits" in source
    assert "decode_direct_fragment_k18" in header
    direct = header.split("decode_direct_fragment_k18(", 1)[1].split(
        "// --- the decode:", 1)[0]
    assert "recast<uint32_t>(tCrB_k)" in direct
    assert "fSFB(i) = sf" in direct
    assert "smem_B.begin" not in direct
    assert "smem_SFB.begin" not in direct

    combined = header + source
    assert "mma.sync.m16n8k16" not in combined
    assert "HMMA" not in combined
    assert "quality_bucket_native_w4a4_only" in source
    assert '"arbitrary_fp32_group_scale_exact", 0' in source


def test_k18_geometry_scale_encoding_and_budget_are_explicit():
    source = SOURCE.read_text()
    assert "ProbeKBits = 18" in source
    assert "ProbeTypeSize = 4 * ProbeKBits + 9" in source
    assert "ProbeLutBytes = 2048" in source  # 2 * 2^9 u16 tables
    assert "ProbeComposeBytes = 4096" in source
    assert "C64 + A16 + SFA4 + B32 + SFB16 = 132" in source
    assert '"operand_accumulator_register_floor", 132' in source


def test_fail_closed_cells_and_graph_safe_out_abi_are_authored():
    source = SOURCE.read_text()
    assert "M == 128 || M == 4096" in source
    assert "N == 4096" in source
    assert "K == 2048 || K == 4096" in source
    assert "cb_fp4_direct_fragment_k18_out" in source
    assert "out must be preallocated" in source
    assert "workspace must be preallocated" in source
    assert "ptr_lut_tile_ids == nullptr" in HEADER.read_text()
    assert "args.ptr_expert_ids == nullptr" in HEADER.read_text()


def test_loader_is_default_off_before_any_build_work(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(cuda_ext, "_fp4_d2f_probe", None)
    monkeypatch.setattr(cuda_ext, "_fp4_d2f_probe_tried", False)

    def forbidden():
        raise AssertionError("default-off probe reached build work")

    monkeypatch.setattr(cuda_ext, "_load_fp4_d2f_probe_ext_locked", forbidden)
    assert cuda_ext.get_fp4_direct_fragment_probe_ext() is None
    assert cuda_ext._fp4_d2f_probe_tried is False
    assert ("fp4_d2f_probe", "get_fp4_direct_fragment_probe_ext") not in (
        cuda_ext._PRELOAD_FAMILIES)


@pytest.mark.parametrize("bad", ["true", "yes", "on", "2", "probe"])
def test_loader_refuses_opt_in_typos(monkeypatch, bad):
    monkeypatch.setenv(FLAG, bad)
    with pytest.raises(ValueError, match=FLAG):
        cuda_ext.get_fp4_direct_fragment_probe_ext()


def test_probe_build_identity_covers_gridbook_include_closure():
    source = SOURCE.read_text()
    local_includes = {
        name for name in cuda_ext._FP4_D2F_PROBE_BUILD_INPUTS
        if name != "cb_fp4_direct_fragment_probe.cu"
    }
    assert "cutlass_fork/sm120_cb_fused_fp4_mma.hpp" in source
    assert "cb_grouped_common.hpp" in source
    assert local_includes == {
        "cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
        "cb_grouped_common.hpp",
        "cutlass_fork/sm120_expert_row_broadcast.hpp",
    }


def test_probe_source_is_wheel_package_data():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"csrc/*.cu"' in pyproject
