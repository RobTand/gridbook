"""Correctness gates for the FP8-CB arm of the persistent-B MoE lane (K1.2).

``csrc/cb_moe_persistent_b.cu`` serves a second payload family through the
SAME schedule the FP4-CB v2 gates in ``test_cb_moe_persistent_b.py`` already
qualify: stock FP8-CB (product mode n_sub=4, ``type_size == 4*k``), decoded as
``bf16_rn(f32(e4m3) * row_scale)`` — the exact value chain of the default
bridge (``moe.py::_expand_native_bf16_slice``, FP8 branch).

WHAT IS PROVEN HERE, AND WITH WHICH INSTRUMENT

* The DECODE is an identity, gated with ``torch.equal`` on an integer view
  against BOTH the pure-Torch decoder (``cb_torch_reference``, no Gridbook
  kernel — the gate cannot pass on a shared bug) and the SHIPPING chain
  (``cb_expand_fp8`` + the bridge's ``float() * scale -> bf16``).  The
  expander needs ``codec.pad_qweight`` for its raw-plane byte window; the
  persistent-B kernel does NOT (it stages superblocks into its own padded
  shared slot), so the padding is applied to the REFERENCE side only —
  padding the kernel side instead would manufacture agreement on a layout the
  serving path never passes it.
* A dedicated ALL-256-BYTE-VALUES gate drives every e4m3 byte pattern —
  NaN (0x7f/0xff) and -0 (0x80) included — through the decode.  Random
  fixtures cannot be relied on to hit those, and the serving books are
  NaN-free by the load-time lossless-cast gate, so this is the one place the
  full byte alphabet is exercised.
* The whole OPERATOR is not an identity (fp32-accumulate GEMM, own reduction
  order); it is gated exactly as the FP4 file gates it: relative L2 against a
  true-fp32 reference built from the SAME BF16 operands, no worse than a
  per-segment BF16 ``F.linear``.
* FP8 adds one surface FP4 does not have: the 2-CTAs/SM occupancy floor is
  BINDING above k=33/k=31 for the wide tiles.  The eligibility predicate the
  lane consults at model load (``cb_moe_persistent_b_fp8_cfg_eligible``) is
  gated AGAINST REALITY here: every config it blesses must launch, every
  config it refuses must be refused by the kernel too, and auto must serve
  every rung up to k=48.

The FP8-CB plane has NO packed scale section (``type_size == 4*k`` is pure
codeword bits), so any random byte plane is legal — there is no two-tier
legality mask to respect and no producer arm is needed for coverage here.

Container-only, like the rest of the native-kernel suite: skips cleanly
without CUDA or off the cc 12.0/12.1 devices the lane is compiled for.
"""
from __future__ import annotations

import contextlib
import functools

import pytest
import torch
import torch.nn.functional as F

codec = pytest.importorskip("gridbook.codec")

from cb_torch_reference import reconstruct_cb_weight  # noqa: E402
from gridbook.cuda_ext import (get_ext,  # noqa: E402
                               get_moe_persistent_b_ext)

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

ext = get_moe_persistent_b_ext()
if ext is None:
    pytest.skip("the persistent-B grouped MoE extension is unavailable "
                "(needs cc 12.0/12.1 + nvcc)", allow_module_level=True)

DEV = torch.device("cuda")
CONFIGS = [list(map(int, row)) for row in ext.cb_moe_persistent_b_configs()]
MIN_TN = min(cfg[1] for cfg in CONFIGS)
MAX_TN = max(cfg[1] for cfg in CONFIGS)


@contextlib.contextmanager
def _true_fp32_matmul():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_float32_matmul_precision(precision)


# ===========================================================================
# Operand construction.
# ===========================================================================
def _fp8_widths(k: int) -> list[int]:
    base, extra = divmod(k, 4)
    return [base + (1 if i < extra else 0) for i in range(4)]


def _fp8_lut_elems(k: int) -> int:
    return sum(2 << w for w in _fp8_widths(k))


@functools.lru_cache(maxsize=None)
def _pack_fp8(k: int, K: int, E: int, N: int, seed: int):
    """``(qw [E,N,row_bytes], lut_u8, lut_f32, scale [E,N], type_size)``.

    The codebook is drawn on the e4m3 grid the way the serving book is held
    (E4M3 bytes); ``lut_f32`` is torch's own conversion of those bytes, which
    is exactly the table the moe.py load gate hands the kernel.  The plane is
    arbitrary bytes — every FP8-CB codeword bit pattern is legal.
    """
    type_size = 4 * k
    g = torch.Generator(device="cpu").manual_seed(seed)
    lut_u8 = ((torch.randn(_fp8_lut_elems(k), generator=g) * 0.5)
              .to(torch.float8_e4m3fn).view(torch.uint8).to(DEV))
    lut_f32 = lut_u8.view(torch.float8_e4m3fn).float().contiguous()
    row_bytes = (K // codec.SUPERBLOCK) * type_size
    qw = torch.randint(0, 256, (E, N, row_bytes), dtype=torch.uint8,
                       generator=g).to(DEV).contiguous()
    scale = ((torch.rand(E, N, generator=g) * 1.5 + 0.25) * 0.01) \
        .float().to(DEV).contiguous()
    return qw, lut_u8, lut_f32, scale, type_size


def _torch_reference_decode(qw, lut_f32, scale, K, k, type_size):
    """The pure-Torch serving chain: values(f32) * row_scale -> bf16."""
    E, N, _ = qw.shape
    rows = E * N
    return reconstruct_cb_weight(
        qw.reshape(rows, -1), lut_f32,
        torch.zeros(rows, dtype=torch.int32, device=DEV),
        scale.reshape(rows), torch.zeros(1, device=DEV),
        N=rows, K=K, k_bits=k, n_sub=4, type_size=type_size,
        is_fp4=False)


def _expander_reference_decode(qw, lut_u8, scale, K, k, type_size):
    """The SHIPPING chain: cb_expand_fp8 (padded plane) + float()*scale->bf16.

    ``pad_qweight`` on the reference input only — the expander's aligned
    8-byte codeword window reads past the tight plane's final row, which the
    kernel under test never does (advisorial note in the module docstring).
    """
    from gridbook import ops as pq_ops

    E, N, _ = qw.shape
    rows = E * N
    raw = codec.pad_qweight(qw.reshape(rows, -1))
    row0 = torch.zeros(rows, dtype=torch.int32, device=DEV)
    value = pq_ops.cb_expand_fp8(raw, lut_u8, row0, rows, K, k, 4, type_size)
    return (value.float()
            * scale.reshape(rows).to(torch.float32)[:, None]
            ).to(torch.bfloat16)


def _probe_decode(qw, lut_f32, scale, K, k, type_size, row0=0, nrows=None):
    E, N, _ = qw.shape
    rows = E * N
    if nrows is None:
        nrows = rows
    return ext.cb_moe_persistent_b_decode_fp8(
        qw.reshape(-1), lut_f32, scale.reshape(-1), row0, nrows, K, k,
        type_size)


def _decoded_weights(qw, lut_f32, scale, k, type_size, K):
    E, N, _ = qw.shape
    return _probe_decode(qw, lut_f32, scale, K, k, type_size).view(E, N, K)


def _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=0, out=None):
    if out is None:
        out = torch.empty((a.shape[0], qw.shape[1]),
                          dtype=torch.bfloat16, device=DEV)
    ext.cb_moe_persistent_b_prefill_fp8(
        out, a, qw, lut_f32, scale, ends, k, type_size, cfg)
    return out


def _ends(counts) -> torch.Tensor:
    return torch.tensor(counts, dtype=torch.int32,
                        device=DEV).cumsum(0, dtype=torch.int32).contiguous()


def _per_segment_references(a, weights, expert_ends):
    rows = int(expert_ends[-1].item()) if expert_ends.numel() else 0
    n = weights.shape[1]
    y_bf16 = torch.zeros((rows, n), dtype=torch.bfloat16, device=DEV)
    y_fp32 = torch.zeros((rows, n), dtype=torch.float32, device=DEV)
    with _true_fp32_matmul():
        start = 0
        for expert, stop in enumerate(expert_ends.cpu().tolist()):
            if stop > start:
                y_bf16[start:stop] = F.linear(a[start:stop], weights[expert])
                y_fp32[start:stop] = (
                    a[start:stop].float() @ weights[expert].float().t())
            start = stop
    return y_bf16, y_fp32


def _rel_l2(y, reference):
    return ((y.float() - reference).norm()
            / reference.norm().clamp_min(1e-12))


def _assert_reassociation_only(y, bf16_linear, fp32, label):
    kernel_rel = _rel_l2(y, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)
    assert torch.isfinite(y).all(), f"{label}: non-finite output"
    assert kernel_rel <= 2e-3, (
        f"{label}: persistent-B FP8 relative L2 {kernel_rel:.6e} exceeds the "
        f"BF16 output-rounding backstop")
    assert kernel_rel <= torch.maximum(1.25 * linear_rel, linear_rel + 2e-5), (
        f"{label}: persistent-B FP8 {kernel_rel:.6e} vs per-segment BF16 "
        f"F.linear {linear_rel:.6e}: the reduction-order difference is larger "
        f"than reassociation explains")


# The rungs: the ceil-first 4-way split changes shape at every k mod 4, so all
# four residues appear; k=28 is DSv4's shipping rung (the built 92 GB body's
# 11 FP8-CB layers), k=48 is the format ceiling (type_size=192, the widest
# staged superblock and the rung where cfg eligibility binds hardest), and
# k=4/12 cover the small-book floor.  N=17/33 are not multiples of 8 or 32,
# so the flat plane's rows line up with no codeword or gather boundary.
_RUNGS = [
    (4, 512, 3, 17),
    (12, 256, 4, 8),
    (28, 1024, 3, 33),
    (29, 512, 3, 17),
    (33, 512, 2, 40),
    (34, 768, 2, 33),
    (40, 512, 2, 24),
    (47, 256, 2, 17),
    (48, 512, 2, 24),
]
_RUNG_IDS = [f"k{k}-K{K}-E{E}-N{N}" for k, K, E, N in _RUNGS]


# ===========================================================================
# 1. The decode stage is an IDENTITY, gated bitwise.
# ===========================================================================
@pytest.mark.parametrize("k,K,E,N", _RUNGS, ids=_RUNG_IDS)
def test_decode_probe_is_bit_identical_to_the_torch_reference(k, K, E, N):
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(
        k, K, E, N, seed=k * 1009 + K)
    rows = E * N

    got = _probe_decode(qw, lut_f32, scale, K, k, type_size)
    want = _torch_reference_decode(qw, lut_f32, scale, K, k, type_size)

    assert got.shape == (rows, K)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("k,K,E,N", _RUNGS, ids=_RUNG_IDS)
def test_decode_probe_is_bit_identical_to_the_shipping_expander_chain(
        k, K, E, N):
    """...and it is the SERVED bridge chain, bit for bit, on the same bytes."""
    if get_ext() is None:
        pytest.skip("the main CB extension (cb_expand_fp8) is unavailable")
    qw, lut_u8, lut_f32, scale, type_size = _pack_fp8(
        k, K, E, N, seed=k * 1009 + K)
    rows = E * N

    got = _probe_decode(qw, lut_f32, scale, K, k, type_size)
    want = _expander_reference_decode(qw, lut_u8, scale, K, k, type_size)

    assert got.shape == (rows, K)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_decode_all_256_e4m3_byte_values_including_nan_and_negative_zero():
    """Every e4m3 byte pattern flows through the decode, deterministically.

    k=32 makes each sub-book exactly 256 two-byte entries, the books are laid
    out so entry j carries bytes (2j)%256 and (2j+1)%256, and one K=2048 row
    holds codewords 0..255 replicated across all four sub-fields — so entry c
    of every book is gathered and all 256 byte values (NaN 0x7f/0xff and -0
    0x80 included) reach the multiply.  Serving books are NaN-free by the
    load-time lossless-cast gate; this gate is where the full alphabet is
    still proven, against the pure-Torch serving chain.
    """
    k, K = 32, 2048
    type_size = 4 * k
    assert _fp8_widths(k) == [8, 8, 8, 8]
    lut_u8 = (torch.arange(_fp8_lut_elems(k), dtype=torch.int64) % 256) \
        .to(torch.uint8).to(DEV)
    lut_f32 = lut_u8.view(torch.float8_e4m3fn).float().contiguous()

    cw = torch.arange(256, dtype=torch.int64) * 0x01010101
    plane_bytes = torch.zeros(256, 4, dtype=torch.uint8)
    for byte in range(4):
        plane_bytes[:, byte] = ((cw >> (8 * byte)) & 0xFF).to(torch.uint8)
    row = plane_bytes.reshape(-1).to(DEV)
    assert row.numel() == (K // codec.SUPERBLOCK) * type_size
    qw = torch.stack([row, row]).reshape(1, 2, -1).contiguous()
    scale = torch.tensor([[1.0, 0.37]], device=DEV).contiguous()

    got = _probe_decode(qw, lut_f32, scale, K, k, type_size)
    want = _torch_reference_decode(qw, lut_f32, scale, K, k, type_size)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # The gate really covered the alphabet: NaN entries decoded to NaN.
    assert got.isnan().any(), "the all-bytes fixture never produced a NaN"


@pytest.mark.parametrize("k,K,E,N", [(28, 1024, 3, 33), (48, 512, 2, 24)],
                         ids=["k28", "k48"])
def test_decode_probe_window_at_row0_is_bit_identical(k, K, E, N):
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(
        k, K, E, N, seed=k * 7 + K)
    rows = E * N
    row0 = max(1, rows // 3)
    nrows = max(1, rows - row0 - 1)

    got = _probe_decode(qw, lut_f32, scale, K, k, type_size, row0=row0,
                        nrows=nrows)
    full = _probe_decode(qw, lut_f32, scale, K, k, type_size)
    assert got.shape == (nrows, K)
    assert torch.equal(got.view(torch.int16),
                       full[row0:row0 + nrows].view(torch.int16))


def test_decode_probe_zero_rows_is_a_well_formed_empty_result():
    k, K, E, N = 28, 512, 2, 16
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=5)
    got = _probe_decode(qw, lut_f32, scale, K, k, type_size, row0=3, nrows=0)
    assert got.shape == (0, K)
    assert got.dtype is torch.bfloat16


# ===========================================================================
# 2. Whole-operator numerics: reassociation, and nothing else.
# ===========================================================================
@pytest.mark.parametrize("counts,k,K,N", [
    ([10, 0, 130, 3], 28, 1024, 256),
    ([200, 1, 0], 29, 512, 128),
    ([1, 1], 48, 512, 64),
    ([0, 0, 5], 40, 768, 40),
    ([64, 64, 64, 64, 64], 28, 2048, 512),
    ([3, 0, 4], 28, 256, MIN_TN),
], ids=["empty-middle-longtail", "skewed-trailing-empty", "one-row-each-k48",
        "leading-empty-ragged-n", "exact-tile-multiples", "narrowest-tile-n"])
def test_whole_operator_error_matches_a_per_segment_bf16_reference(
        counts, k, K, N):
    """Same discipline as the FP4 file: both sides consume the SAME BF16
    operands (the caller's activations, the kernel's own bit-exact decode), so
    the only admissible difference is the fp32 summation order."""
    torch.manual_seed(20260817)
    E, P = len(counts), sum(counts)
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(
        k, K, E, N, seed=k * 31 + N)
    weights = _decoded_weights(qw, lut_f32, scale, k, type_size, K)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)

    y = _prefill(a, qw, lut_f32, scale, ends, k, type_size)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(y, bf16_linear, fp32, f"counts={counts}")


def test_dsv4_shape_smoke_k28():
    """The shipping rung at the built body's exact layer geometry (w2 stage:
    K=inter=2048, N=hidden=4096, row_bytes=896; E kept small for time — the
    schedule is per-(expert, N-tile) and E only multiplies work units)."""
    counts = [40, 0, 300, 7, 1, 0, 64, 100]
    k, K, N = 28, 2048, 4096
    E, P = len(counts), sum(counts)
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=92)
    torch.manual_seed(93)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    weights = _decoded_weights(qw, lut_f32, scale, k, type_size, K)

    y = _prefill(a, qw, lut_f32, scale, ends, k, type_size)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(y, bf16_linear, fp32, "dsv4-w2-shape")


# ===========================================================================
# 3. Routing breadth, and the promise that nothing outside a segment is
#    written.
# ===========================================================================
_ROUTING = [
    ([0, 7, 5], "leading-empty"),
    ([0, 0, 33, 0, 0], "all-empty-but-one"),
    ([0, 0, 0, 0], "all-empty"),
    ([1, 1, 1, 1, 1, 1], "one-row-per-expert"),
    ([700, 3, 0, 1, 0, 2], "skewed-past-tile-m"),
]


@pytest.mark.parametrize("counts", [c for c, _ in _ROUTING],
                         ids=[i for _, i in _ROUTING])
def test_routing_breadth_is_correct_and_writes_only_routed_rows(counts):
    torch.manual_seed(8242 + len(counts))
    k, K, N = 28, 512, 64
    spare = 5
    E, total = len(counts), sum(counts)
    P = total + spare
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=877)
    weights = _decoded_weights(qw, lut_f32, scale, k, type_size, K)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)

    out = torch.full((P, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    _prefill(a, qw, lut_f32, scale, ends, k, type_size, out=out)
    torch.cuda.synchronize()

    tail = out[total:]
    sentinel = torch.full_like(tail, float("nan"))
    assert torch.equal(tail.view(torch.int16), sentinel.view(torch.int16)), (
        f"counts={counts}: the kernel wrote {int((~tail.isnan()).sum())} "
        f"values past the last routed row")

    routed = out[:total]
    if total:
        bf16_linear, fp32 = _per_segment_references(a, weights, ends)
        _assert_reassociation_only(routed, bf16_linear, fp32,
                                   f"counts={counts}")
    else:
        assert routed.numel() == 0


# ===========================================================================
# 4. Config selection and the occupancy-floor eligibility contract.
# ===========================================================================
def test_all_configs_eligible_at_the_dsv4_rung_and_agree_with_auto():
    """k=28 (the built body's FP8 rung) holds every compiled tile; each must
    compute the same operator as auto within reassociation."""
    counts = [0, 5, 200, 1, 0, 70]
    k, K, N = 28, 1024, 200
    E, P = len(counts), sum(counts)
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=1234)
    torch.manual_seed(9190)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    weights = _decoded_weights(qw, lut_f32, scale, k, type_size, K)

    auto = _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=0)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(auto, bf16_linear, fp32, "cfg=0")
    for cfg in range(1, len(CONFIGS) + 1):
        assert bool(ext.cb_moe_persistent_b_fp8_cfg_eligible(cfg, type_size)), (
            f"cfg {cfg} must be eligible at the DSv4 rung (type_size="
            f"{type_size})")
        chosen = _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=cfg)
        _assert_reassociation_only(chosen, bf16_linear, fp32, f"cfg={cfg}")
        disagreement = _rel_l2(chosen.float(), auto.float())
        assert disagreement <= 4e-3, (
            f"cfg={cfg} disagrees with auto by {disagreement:.6e}")


def test_eligibility_predicate_matches_kernel_refusals_at_k48():
    """ATTESTATION AGAINST REALITY at the format ceiling: every tile the
    predicate blesses must launch, every tile it refuses must be refused by
    the kernel's own occupancy TORCH_CHECK — and at least one of each must
    exist at k=48, or the gate is vacuous."""
    counts = [7, 0, 30]
    k, K, N = 48, 512, 128
    E, P = len(counts), sum(counts)
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=4488)
    torch.manual_seed(4489)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    weights = _decoded_weights(qw, lut_f32, scale, k, type_size, K)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)

    eligible = [cfg for cfg in range(1, len(CONFIGS) + 1)
                if bool(ext.cb_moe_persistent_b_fp8_cfg_eligible(
                    cfg, type_size))]
    ineligible = [cfg for cfg in range(1, len(CONFIGS) + 1)
                  if cfg not in eligible]
    assert eligible, "k=48 must keep at least one eligible tile"
    assert ineligible, ("k=48 must exclude at least one tile, or the "
                        "eligibility surface is untested")

    for cfg in eligible:
        y = _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=cfg)
        _assert_reassociation_only(y, bf16_linear, fp32, f"k48-cfg={cfg}")
    for cfg in ineligible:
        with pytest.raises(RuntimeError, match="two CTAs per SM"):
            _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=cfg)

    # And auto must serve the ceiling rung by filtering on the same predicate.
    y = _prefill(a, qw, lut_f32, scale, ends, k, type_size, cfg=0)
    _assert_reassociation_only(y, bf16_linear, fp32, "k48-auto")


# ===========================================================================
# 5. CUDA-graph capture.
# ===========================================================================
def test_cuda_graph_capture_and_replay_match_eager():
    """Same geometry claim as the FP4 arm: the grid is a pure function of
    (E, N) and the tile, so the call captures with no host sync; replay is
    bitwise."""
    counts = [0, 5, 130, 1, 0, 12]
    k, K, N = 28, 512, 128
    E, P0 = len(counts), sum(counts)
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=1911)
    torch.manual_seed(1912)
    P = P0 + 3
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    out = torch.full((P, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    poison = torch.full_like(out, float("nan"))

    ext.cb_moe_persistent_b_prefill_fp8(
        out, a, qw, lut_f32, scale, ends, k, type_size, 0)
    torch.cuda.synchronize()
    eager = out.clone()
    assert not torch.equal(eager.view(torch.int16), poison.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    out.copy_(poison)
    with torch.cuda.graph(graph):
        ext.cb_moe_persistent_b_prefill_fp8(
            out, a, qw, lut_f32, scale, ends, k, type_size, 0)

    for replay in range(3):
        out.copy_(poison)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.clone().view(torch.int16),
                           eager.view(torch.int16)), (
            f"replay {replay} diverged from the eager result")


# ===========================================================================
# 6. Contract rejections.
# ===========================================================================
def test_rejects_the_fp4_type_size_law():
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(28, 512, 2, 16, seed=1)
    a = torch.randn(4, 512, device=DEV, dtype=torch.bfloat16)
    ends = _ends([2, 2])
    out = torch.empty(4, 16, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="type_size == 4\\*k"):
        ext.cb_moe_persistent_b_prefill_fp8(
            out, a, qw, lut_f32, scale, ends, 28, 4 * 28 + 9, 0)


def test_rejects_a_wrong_size_codebook():
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(28, 512, 2, 16, seed=2)
    a = torch.randn(4, 512, device=DEV, dtype=torch.bfloat16)
    ends = _ends([2, 2])
    out = torch.empty(4, 16, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="flat codebook"):
        ext.cb_moe_persistent_b_prefill_fp8(
            out, a, qw, lut_f32[:-2], scale, ends, 28, type_size, 0)


def test_rejects_a_wrong_shape_scale_table():
    qw, _lut_u8, lut_f32, scale, type_size = _pack_fp8(28, 512, 2, 16, seed=3)
    a = torch.randn(4, 512, device=DEV, dtype=torch.bfloat16)
    ends = _ends([2, 2])
    out = torch.empty(4, 16, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="scale must be \\[E,N\\]"):
        ext.cb_moe_persistent_b_prefill_fp8(
            out, a, qw, lut_f32, scale[:, :8].contiguous(), ends, 28,
            type_size, 0)


def test_rejects_a_non_fp32_codebook():
    qw, lut_u8, _lut_f32, scale, type_size = _pack_fp8(28, 512, 2, 16, seed=4)
    a = torch.randn(4, 512, device=DEV, dtype=torch.bfloat16)
    ends = _ends([2, 2])
    out = torch.empty(4, 16, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="FP32 table"):
        ext.cb_moe_persistent_b_prefill_fp8(
            out, a, qw, lut_u8, scale, ends, 28, type_size, 0)
