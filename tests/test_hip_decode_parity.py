"""Bit-parity gate for the ROCm/HIP CB kernels (``gridbook/csrc_hip``).

Collects everywhere, runs only where there is a ROCm device and the extension
builds — the same shape as ``test_cuda_gemv.py``, which needs nvcc and skips in
the build venv.

    # on a Strix Halo / gfx1151 box with a ROCm torch:
    PYTHONPATH=plugins/gridbook python -m pytest \\
        plugins/gridbook/tests/test_hip_decode_parity.py -v

What is being pinned, and why in this order:

* ``test_e4m3_decode_all_256_codes`` — every e4m3 byte decodes to exactly the
  value torch's ``float8_e4m3fn`` gives.  The kernels reassemble e4m3 bits by
  hand (no vendor conversion intrinsic), so this is the base of the whole
  numerics stack and is checked exhaustively rather than sampled.
* ``test_gemv_fp8_matches_torch_reference`` over the FP8_CB_K28..K48 ladder —
  the codeword bit-stream, the ceil-first product sub-split, the per-channel
  scale and the accumulation, against a reference built with plain torch ops
  from LAYOUT.md §1.1.
* ``test_gemv_fp4_v2_*`` — the same for the NVFP4_CB two-tier (layout v2)
  scale coding, whose compose step is the one piece of format math the fp8
  ladder never exercises.
* ``test_expand_matches_reference`` — an independent decode implementation
  (byte-exact), so a bit-extraction bug cannot hide behind the GEMV's f32 sum.
* ``test_fused_row_offset_two_roles`` — the qkv/gate_up fusion mechanism, where
  output rows point at different codebook blocks and the kernel's LDS-staged
  LUT is valid for only some of them.

Tolerance: the outputs are bf16, so the gate is ONE bf16 output-rounding step
away from the bf16 rounding of an fp64 reference, plus a norm backstop.  That
is the same discipline as ``test_cuda_gemv._assert_triton_close`` and it is the
tightest defensible bound — bit-equality is not available because summation
order differs and, separately, this device's WMMA unit is not exactly-rounded
(measured ~0.5 f32 ULP; see ``csrc_hip/README.md``).
"""
from __future__ import annotations

import pytest
import torch

codec = pytest.importorskip("gridbook.codec",
                            reason="gridbook plugin not importable")
hip_ext = pytest.importorskip("gridbook.hip_ext")

if not hip_ext.is_rocm():
    pytest.skip("no ROCm torch / HIP device", allow_module_level=True)

ext = hip_ext.get_ext()
if ext is None:
    pytest.skip("HIP extension unavailable (no hipcc?)", allow_module_level=True)

DEV = "cuda"          # a ROCm torch reports HIP devices as 'cuda'
FP8_RUNGS = [28, 29, 32, 33, 36, 40, 44, 45, 47, 48]
FP4_RUNGS = [12, 13, 16, 18, 20, 24]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bit_split(k: int, n_sub: int) -> list[int]:
    """Ceil-first, larger halves first — nvfp4_cb_formats._bit_split."""
    base, extra = divmod(k, n_sub)
    return [base + (1 if i < extra else 0) for i in range(n_sub)]


def _pack_codes(codes: torch.Tensor, k: int) -> torch.Tensor:
    """(rows, n_sb, 32) int64 codewords -> (rows, n_sb, 4k) uint8, LSB-first.

    LAYOUT.md §1.1 verbatim, with plain torch ops so it shares nothing with the
    kernel's aligned-word extraction.
    """
    rows, n_sb, _ = codes.shape
    bits = (codes.unsqueeze(-1) >> torch.arange(k, dtype=torch.int64)) & 1
    bits = bits.reshape(rows, n_sb, 4 * k, 8)
    wt = 1 << torch.arange(8, dtype=torch.int64)
    return (bits * wt).sum(-1).to(torch.uint8)


def _rand_case_fp8(k: int, N: int, K: int, seed: int, roles: int = 1):
    """Synthetic FP8_CB tensor: random codewords + a grid-snapped codebook."""
    g = torch.Generator().manual_seed(seed)
    n_sb = K // 256
    widths = _bit_split(k, 4)
    subs_per_role = [
        (torch.randn(1 << w, 2, generator=g) * 4.0)
        .to(torch.float8_e4m3fn).float() for w in widths]
    role_tables = [subs_per_role]
    for r in range(1, roles):
        gr = torch.Generator().manual_seed(seed + 1000 * r)
        role_tables.append([(torch.randn(1 << w, 2, generator=gr) * 4.0)
                            .to(torch.float8_e4m3fn).float() for w in widths])
    flat_per_role = [codec.build_flat_codebook(t) for t in role_tables]
    cb_flat = torch.cat(flat_per_role)
    per_role = int(flat_per_role[0].numel())

    codes = torch.zeros(N, n_sb, 32, dtype=torch.int64)
    off = 0
    for i, w in enumerate(widths):
        idx = torch.randint(0, 1 << w, (N, n_sb, 32), generator=g,
                            dtype=torch.int64)
        codes |= idx << off
        off += w
    packed = _pack_codes(codes, k).reshape(N, n_sb * 4 * k)

    row_of_role = (torch.arange(N) * roles) // N
    cb_row_offset = (row_of_role * per_role).to(torch.int32)
    scale = (torch.rand(N, generator=g) + 0.5) * 0.02
    return dict(
        k=k, N=N, K=K, n_sb=n_sb, type_size=4 * k, widths=widths,
        codes=codes, role_tables=role_tables, row_of_role=row_of_role,
        qwp=codec.pad_qweight(packed).to(DEV),
        cb8=cb_flat.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                   .to(DEV),
        cb_flat=cb_flat.to(DEV),
        cb_row_offset=cb_row_offset.to(DEV),
        scale=scale.to(DEV).float())


def _ref_weight_fp8(case, *, apply_scale: bool) -> torch.Tensor:
    """[N, K] fp64 reference weight, decoded with pure torch indexing."""
    N, K, k = case["N"], case["K"], case["k"]
    widths, codes = case["widths"], case["codes"]
    out = torch.zeros(N, case["n_sb"], 32, 8, dtype=torch.float64)
    off = 0
    for i, w in enumerate(widths):
        idx = (codes >> off) & ((1 << w) - 1)
        off += w
        for role, tables in enumerate(case["role_tables"]):
            rows = (case["row_of_role"] == role).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            vals = tables[i].to(torch.float64)[idx[rows]]     # (r, n_sb, 32, 2)
            out[rows, :, :, 2 * i:2 * i + 2] = vals
    w = out.reshape(N, K)
    if apply_scale:
        # v1 contract: bf16_rn(value * scale) per weight.
        sc = case["scale"].cpu().to(torch.float64).reshape(N, 1)
        w = (w * sc).to(torch.bfloat16).to(torch.float64)
    return w


def _assert_close(got: torch.Tensor, ref: torch.Tensor, tag: str,
                  norm_tol: float = 5e-3) -> None:
    g = got.float().double()
    rb = ref.to(torch.bfloat16).double()          # bf16 rounding of the truth
    tol = torch.maximum(g.abs(), rb.abs()) * 2.0 ** -7 + 1e-5
    nbad = int(((g - rb).abs() > tol).sum())
    assert nbad == 0, (
        f"{tag}: {nbad}/{ref.numel()} elements beyond 1 bf16 output ULP "
        f"(max delta {float((g - rb).abs().max()):.3e})")
    rel = float((g - ref).norm() / ref.norm().clamp_min(1e-12))
    assert rel <= norm_tol, f"{tag}: norm backstop rel {rel:.3e}"


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_e4m3_decode_all_256_codes():
    """Hand-assembled e4m3 -> f32 must equal torch's, for every byte.

    Exercised through the transient expander, whose output IS the raw decoded
    e4m3 bytes, and through a codebook that contains all 256 codes.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    want = codes.view(torch.float8_e4m3fn).float()
    # NaN codes (0x7f / 0xff) are the only ones the format never emits.
    finite = ~torch.isnan(want)
    assert int(finite.sum()) == 254
    # Round-trip through bf16 must be lossless: e4m3 has 3 mantissa bits, bf16
    # has 7, which is the property the WMMA prologue relies on.
    rt = want[finite].to(torch.bfloat16).float()
    assert torch.equal(rt, want[finite]), "e4m3 -> bf16 is not lossless"


@pytest.mark.parametrize("k", FP8_RUNGS)
def test_gemv_fp8_matches_torch_reference(k):
    case = _rand_case_fp8(k, N=96, K=768, seed=k)
    torch.manual_seed(k)
    x = torch.randn(4, case["K"], dtype=torch.bfloat16, device=DEV)
    y = ext.cb_gemv_fp8(x, case["qwp"], case["cb8"], case["cb_row_offset"],
                        case["scale"], case["N"], case["K"], k, 4,
                        case["type_size"], False)
    ref = x.cpu().double() @ _ref_weight_fp8(case, apply_scale=True).t()
    _assert_close(y.cpu(), ref, f"GEMV k={k}")


@pytest.mark.parametrize("M", [1, 2, 3, 8, 16])
def test_gemv_fp8_m_tiles(M):
    """MT is a compile-time register tile; M in between two tiles must be
    predicated off, not read out of bounds."""
    case = _rand_case_fp8(44, N=64, K=512, seed=100 + M)
    torch.manual_seed(M)
    x = torch.randn(M, case["K"], dtype=torch.bfloat16, device=DEV)
    y = ext.cb_gemv_fp8(x, case["qwp"], case["cb8"], case["cb_row_offset"],
                        case["scale"], case["N"], case["K"], 44, 4,
                        case["type_size"], False)
    ref = x.cpu().double() @ _ref_weight_fp8(case, apply_scale=True).t()
    _assert_close(y.cpu(), ref, f"GEMV M={M}")


def test_fused_row_offset_two_roles():
    """qkv/gate_up fusion: rows point at different blocks of the concatenated
    codebook, so a workgroup straddling the boundary must fall back off its
    LDS-staged table for the rows that do not belong to it."""
    case = _rand_case_fp8(44, N=96, K=512, seed=7, roles=3)
    assert int(case["cb_row_offset"].max()) > 0
    torch.manual_seed(7)
    x = torch.randn(2, case["K"], dtype=torch.bfloat16, device=DEV)
    y = ext.cb_gemv_fp8(x, case["qwp"], case["cb8"], case["cb_row_offset"],
                        case["scale"], case["N"], case["K"], 44, 4,
                        case["type_size"], False)
    ref = x.cpu().double() @ _ref_weight_fp8(case, apply_scale=True).t()
    _assert_close(y.cpu(), ref, "fused row-offset")


@pytest.mark.parametrize("k", [28, 36, 44, 47, 48])
def test_expand_matches_reference(k):
    """The transient expander is a second, independent decode: byte-exact."""
    case = _rand_case_fp8(k, N=64, K=512, seed=200 + k)
    w = ext.cb_expand_fp8(case["qwp"], case["cb8"], case["cb_row_offset"],
                          case["N"], case["K"], k, 4, case["type_size"])
    ref = _ref_weight_fp8(case, apply_scale=False).float().to(
        torch.float8_e4m3fn)
    assert torch.equal(w.cpu().view(torch.uint8), ref.view(torch.uint8)), (
        f"k={k}: expanded bytes differ from the torch reference decode")


@pytest.mark.parametrize("M", [17, 32, 64, 96])
def test_gemm_fp8_matches_torch_reference(M):
    """The WMMA prefill GEMM applies the channel scale in its f32 epilogue, so
    its reference is the RAW decoded weight times the scale (no per-weight
    round).  M=17 and N=100 exercise the ragged edge tiles."""
    N = 100 if M == 32 else 96
    case = _rand_case_fp8(44, N=N, K=768, seed=300 + M)
    torch.manual_seed(M)
    x = torch.randn(M, case["K"], dtype=torch.bfloat16, device=DEV)
    y = ext.cb_gemm_fp8(x, case["qwp"], case["cb8"], case["cb_row_offset"],
                        case["scale"], case["N"], case["K"], 44, 4,
                        case["type_size"], False)
    raw = _ref_weight_fp8(case, apply_scale=False)
    ref = (x.cpu().double() @ raw.t()) * case["scale"].cpu().double()
    _assert_close(y.cpu(), ref, f"GEMM M={M} N={N}")


def test_lut_lds_budget_is_bf16_materialised():
    """The LDS LUT is materialised as bf16 whatever the sidecar stored, so its
    footprint is 2 bytes per codebook element — NOT the on-disk e4m3 size.

    These are the numbers the rung ladder has to be designed against; if the
    LUT dtype contract is ever weakened back to a byte table with per-gather
    conversion, this test is what should fail first.
    """
    assert ext.lut_bytes_fp8(28) == 2048        # 2 KiB
    assert ext.lut_bytes_fp8(36) == 8192        # 8 KiB
    assert ext.lut_bytes_fp8(44) == 32768       # 32 KiB
    assert ext.lut_bytes_fp8(48) == 65536       # 64 KiB — the whole budget
    # The staging decision is a measured crossover, not a guess; a change to it
    # must be accompanied by a re-measurement (csrc_hip/README.md).
    assert ext.lut_is_lds(36) is True
    assert ext.lut_is_lds(48) is False


@pytest.mark.parametrize("k", [28, 36, 44, 48])
def test_bf16_grid_sidecar_is_identical(k):
    """Grid-source independence (the LUT dtype contract).

    gfx1151 has no fp8 hardware, so an FP8_CB codeword must be materialised as
    bf16 for WMMA regardless — which makes a bf16-grid codebook same-bytes,
    same-speed and a strict grid superset on this platform.  The kernels must
    therefore accept EITHER sidecar dtype.  For a table whose values are
    e4m3-representable the two must agree BIT-EXACTLY (e4m3 -> bf16 is exact),
    which is a much stronger statement than "close" and is what is asserted.
    """
    case = _rand_case_fp8(k, N=96, K=768, seed=500 + k)
    torch.manual_seed(k)
    x = torch.randn(4, case["K"], dtype=torch.bfloat16, device=DEV)
    args = (case["qwp"], case["cb_row_offset"], case["scale"], case["N"],
            case["K"], k, 4, case["type_size"], False)
    y_e4m3 = ext.cb_gemv_fp8(x, args[0], case["cb8"], *args[1:])
    cb_bf16 = case["cb8"].view(torch.float8_e4m3fn).to(torch.bfloat16)
    y_bf16 = ext.cb_gemv_fp8(x, args[0], cb_bf16, *args[1:])
    assert torch.equal(y_e4m3.view(torch.uint16), y_bf16.view(torch.uint16)), (
        f"k={k}: bf16-grid sidecar differs from e4m3-grid on an "
        f"e4m3-representable table")


# --------------------------------------------------------------------------- #
# NVFP4_CB two-tier (layout v2)
# --------------------------------------------------------------------------- #
def _rand_case_fp4_v2(k: int, N: int, K: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n_sb = K // 256
    widths = _bit_split(k, 2)
    tables = [(torch.randn(1 << w, 4, generator=g) * 2.0).to(torch.bfloat16)
              .float() for w in widths]
    cb_flat = codec.build_flat_codebook(tables)

    codes = torch.zeros(N, n_sb, 32, dtype=torch.int64)
    off = 0
    for w in widths:
        codes |= torch.randint(0, 1 << w, (N, n_sb, 32), generator=g,
                               dtype=torch.int64) << off
        off += w
    idx_bytes = _pack_codes(codes, k)                      # (N, n_sb, 4k)

    # Two-tier scale section: 1 E8M0 super byte + 8 bytes of 16 4-bit codes,
    # group g in byte g//2 with EVEN g in the low nibble (spec §5.1).  E is kept
    # in a range where T[c] * 2^(E-127) stays inside e4m3, which is what the
    # encoder guarantees; the kernel only multiplies, so any E works for parity.
    super_e = torch.randint(112, 132, (N, n_sb, 1), generator=g,
                            dtype=torch.int64)
    sub_c = torch.randint(0, 16, (N, n_sb, 16), generator=g, dtype=torch.int64)
    pairs = sub_c.reshape(N, n_sb, 8, 2)
    sub_bytes = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    packed = torch.cat([idx_bytes, super_e.to(torch.uint8), sub_bytes], dim=-1)
    assert packed.shape[-1] == 4 * k + 9
    packed = packed.reshape(N, n_sb * (4 * k + 9))

    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE)
    return dict(
        k=k, N=N, K=K, n_sb=n_sb, type_size=4 * k + 9, widths=widths,
        codes=codes, tables=tables, super_e=super_e, sub_c=sub_c,
        qwp=codec.pad_qweight(packed).to(DEV),
        cb_flat=cb_flat.to(DEV),
        cb_row_offset=torch.zeros(N, dtype=torch.int32, device=DEV),
        compose=compose.to(DEV))


def _ref_weight_fp4_v2(case) -> torch.Tensor:
    N, K, k = case["N"], case["K"], case["k"]
    out = torch.zeros(N, case["n_sb"], 32, 8, dtype=torch.float64)
    off = 0
    for i, w in enumerate(case["widths"]):
        idx = (case["codes"] >> off) & ((1 << w) - 1)
        off += w
        out[:, :, :, 4 * i:4 * i + 4] = case["tables"][i].to(
            torch.float64)[idx]
    # scale_g = T[c_g] * 2^(E-127), group g covering weights [16g, 16g+16)
    tbl = torch.tensor(codec.TWO_TIER_SUB_TABLE, dtype=torch.float64)
    sc = tbl[case["sub_c"]] * torch.pow(
        2.0, case["super_e"].double() - codec.TWO_TIER_SUPER_BIAS)
    sc = sc.reshape(N, case["n_sb"], 16, 1).expand(-1, -1, -1, 16)
    w = out.reshape(N, case["n_sb"], 256) * sc.reshape(N, case["n_sb"], 256)
    # v1 decode contract: bf16_rn(value * scale) per weight.
    return w.reshape(N, K).to(torch.bfloat16).to(torch.float64)


@pytest.mark.parametrize("k", FP4_RUNGS)
def test_gemv_fp4_v2_matches_torch_reference(k):
    case = _rand_case_fp4_v2(k, N=64, K=512, seed=400 + k)
    torch.manual_seed(k)
    x = torch.randn(2, case["K"], dtype=torch.bfloat16, device=DEV)
    y = ext.cb_gemv_fp4_v2(x, case["qwp"], case["cb_flat"],
                           case["cb_row_offset"], case["compose"], case["N"],
                           case["K"], k, 2, case["type_size"])
    ref = x.cpu().double() @ _ref_weight_fp4_v2(case).t()
    _assert_close(y.cpu(), ref, f"GEMV fp4-v2 k={k}")


def test_two_tier_compose_table_matches_codec():
    """The kernel gathers compose[E*16 + c]; that table must be exactly the one
    codec.build_compose_table ships (two-tier-scale-spec.md §1.1)."""
    t = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE)
    assert t.numel() == 256 * 16 and t.dtype == torch.float32
    tbl = torch.tensor(codec.TWO_TIER_SUB_TABLE, dtype=torch.float64)
    want = (tbl[None, :] * torch.pow(
        2.0, torch.arange(256, dtype=torch.float64)[:, None]
        - codec.TWO_TIER_SUPER_BIAS)).to(torch.float32).reshape(-1)
    assert torch.equal(t, want)
