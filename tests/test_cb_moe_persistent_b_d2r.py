"""CUDA gates for persistent-B's experimental BF16 direct-to-register B.

These tests intentionally exercise the exact cooperative helper, every
existing tile cfg, all format rungs, both DSV4 production GEMM shapes, ragged
and empty expert segments, and eager plus mutated CUDA-graph replay.  The
candidate is compared to the established persistent-B operator on identical
packed weights; no served-performance claim is made here.
"""
from __future__ import annotations

import pytest
import torch

from cb_torch_reference import reconstruct_cb_weight
from gridbook import ops
from gridbook.cuda_ext import get_moe_persistent_b_ext
from test_cb_moe_persistent_b import _ends, _pack

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

ext = get_moe_persistent_b_ext()
if ext is None:
    pytest.fail("CUDA is available but the persistent-B extension did not "
                "build/load; the D2R gate must not false-green by skipping")

_D2R_SYMBOLS = (
    "cb_moe_persistent_b_prefill_d2r",
    "cb_moe_persistent_b_d2r_decode_pairs",
    "cb_moe_persistent_b_d2r_prepare",
    "cb_moe_persistent_b_d2r_configs",
)
missing = [name for name in _D2R_SYMBOLS if not hasattr(ext, name)]
if missing:
    pytest.fail(f"persistent-B module is missing D2R symbols {missing}")
ext.cb_moe_persistent_b_d2r_prepare()

DEV = torch.device("cuda")
BASE_CONFIGS = [list(map(int, row))
                for row in ext.cb_moe_persistent_b_configs()]
D2R_CONFIGS = [list(map(int, row))
               for row in ext.cb_moe_persistent_b_d2r_configs()]


def _run(symbol, a, qw, lut, compose, ends, k, type_size, cfg):
    out = torch.full((a.shape[0], qw.shape[1]), float("nan"),
                     dtype=torch.bfloat16, device=DEV)
    getattr(ext, symbol)(out, a, qw, lut, compose, ends, k, type_size, cfg)
    return out


def test_d2r_cfg_attestation_matches_every_existing_tile_and_drops_only_b_smem():
    assert len(D2R_CONFIGS) == len(BASE_CONFIGS) == 4
    for base, direct in zip(BASE_CONFIGS, D2R_CONFIGS):
        tm, tn, warps, threads, base_smem, capacity = base
        (dtm, dtn, dwarps, dthreads, direct_smem, dcapacity,
         wn, wm, matom, natom) = direct
        assert (dtm, dtn, dwarps, dthreads, dcapacity) == \
            (tm, tn, warps, threads, capacity)
        assert (wn, wm) == (warps, 1)
        assert 4 * matom * natom == 32
        assert natom in (1, 2)
        assert base_smem - direct_smem == tn * 64 * 2


@pytest.mark.parametrize("k", range(1, 25), ids=lambda k: f"k{k}")
def test_cooperative_pair_probe_is_bit_exact_on_every_format_rung(k):
    K, E, N = 512, 2, 5
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=91000 + k, source="synth", super_span=None)
    rows = E * N
    flat = qw.reshape(-1)
    got = ext.cb_moe_persistent_b_d2r_decode_pairs(
        flat, lut, compose, 0, rows, K, k, type_size)
    established = ext.cb_moe_persistent_b_decode(
        flat, lut, compose, 0, rows, K, k, type_size)
    independent = reconstruct_cb_weight(
        qw.reshape(rows, -1), lut,
        torch.zeros(rows, dtype=torch.int32, device=DEV),
        torch.zeros(1, device=DEV), compose, N=rows, K=K, k_bits=k,
        n_sub=2, type_size=type_size, is_fp4=True, is_v2=True)
    assert torch.equal(got.view(torch.int16), established.view(torch.int16))
    assert torch.equal(got.view(torch.int16), independent.view(torch.int16))


@pytest.mark.parametrize("cfg", range(1, len(BASE_CONFIGS) + 1),
                         ids=lambda cfg: f"cfg{cfg}")
def test_identity_activation_proves_every_fragment_coordinate_and_m_loop(cfg):
    """P=K=256 and A=I makes Y exactly decoded_W.T, bit for bit.

    This covers all 32 codewords, both B registers, every lane pair, every
    kk/stage, every N tile, and 2 or 4 in-kernel M loops depending on cfg.
    """
    k, K, E, N = 18, 256, 1, 256
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=92018, source="synth", super_span=None)
    a = torch.eye(K, dtype=torch.bfloat16, device=DEV)
    ends = _ends([K])
    decoded = ext.cb_moe_persistent_b_d2r_decode_pairs(
        qw.reshape(-1), lut, compose, 0, N, K, k, type_size)
    got = _run("cb_moe_persistent_b_prefill_d2r", a, qw, lut, compose,
               ends, k, type_size, cfg)
    assert torch.equal(got.view(torch.int16),
                       decoded.t().contiguous().view(torch.int16))


_PRODUCTION_CELLS = [
    (k, K, 4096)
    for k in (12, 16, 18)
    for K in (4096, 2048)
]


@pytest.mark.parametrize(
    "k,K,N", _PRODUCTION_CELLS,
    ids=[f"k{k}-K{K}-N{N}" for k, K, N in _PRODUCTION_CELLS])
@pytest.mark.parametrize("cfg", range(1, len(BASE_CONFIGS) + 1),
                         ids=lambda cfg: f"cfg{cfg}")
def test_production_cells_match_eager_and_mutated_graph(k, K, N, cfg):
    """Every K12/K16/K18 DSV4 w13/w2 shape, every compiled cfg.

    ``counts`` includes leading, middle and trailing empty experts plus ragged
    1/3/17-row segments.  Graph replay mutates A in place, so equality proves
    the graph runs the candidate rather than replaying a captured constant.
    """
    counts = [0, 1, 17, 0, 3, 0]
    E, P = len(counts), sum(counts)
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=93000 + 10 * k + K, source="synth")
    ends = _ends(counts)
    generator = torch.Generator(device="cuda").manual_seed(94000 + k + K)
    a = torch.randn(P, K, dtype=torch.bfloat16, device=DEV,
                    generator=generator) * 0.125

    baseline = _run("cb_moe_persistent_b_prefill", a, qw, lut, compose,
                    ends, k, type_size, cfg)
    candidate = _run("cb_moe_persistent_b_prefill_d2r", a, qw, lut, compose,
                     ends, k, type_size, cfg)
    torch.cuda.synchronize()
    assert torch.equal(candidate.view(torch.int16), baseline.view(torch.int16))

    graph_out = torch.full_like(candidate, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.cb_moe_persistent_b_prefill_d2r(
            graph_out, a, qw, lut, compose, ends, k, type_size, cfg)

    mutated = torch.randn(P, K, dtype=torch.bfloat16, device=DEV,
                          generator=generator) * 0.25
    a.copy_(mutated)
    # Same P, different expert ownership including newly empty/non-empty
    # segments. The captured kernel must reread device-resident route ends.
    ends.copy_(_ends([2, 0, 0, 9, 0, 10]))
    mutated_baseline = _run(
        "cb_moe_persistent_b_prefill", a, qw, lut, compose, ends, k,
        type_size, cfg)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out.view(torch.int16),
                       mutated_baseline.view(torch.int16))


def test_tail_n_and_zero_routed_rows_are_safe():
    k, K, E, N = 16, 512, 4, 40
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=95016, source="synth")
    ends = _ends([0, 5, 0, 2])
    a = torch.randn(7, K, dtype=torch.bfloat16, device=DEV)
    for cfg in range(1, len(BASE_CONFIGS) + 1):
        baseline = _run("cb_moe_persistent_b_prefill", a, qw, lut, compose,
                        ends, k, type_size, cfg)
        direct = _run("cb_moe_persistent_b_prefill_d2r", a, qw, lut, compose,
                      ends, k, type_size, cfg)
        assert torch.equal(direct.view(torch.int16), baseline.view(torch.int16))

    empty_a = torch.empty(0, K, dtype=torch.bfloat16, device=DEV)
    empty_ends = torch.zeros(E, dtype=torch.int32, device=DEV)
    empty = _run("cb_moe_persistent_b_prefill_d2r", empty_a, qw, lut,
                 compose, empty_ends, k, type_size, 0)
    assert empty.shape == (0, N)


def test_every_cfg_handles_independent_expert_boundaries():
    counts = [0, 1, 63, 64, 65, 127, 128, 129]
    k, K, E, N = 16, 256, len(counts), 40
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=95516, source="synth")
    ends = _ends(counts)
    a = torch.randn(sum(counts), K, dtype=torch.bfloat16, device=DEV)
    for cfg in range(1, len(BASE_CONFIGS) + 1):
        baseline = _run("cb_moe_persistent_b_prefill", a, qw, lut, compose,
                        ends, k, type_size, cfg)
        direct = _run("cb_moe_persistent_b_prefill_d2r", a, qw, lut, compose,
                      ends, k, type_size, cfg)
        assert torch.equal(direct.view(torch.int16), baseline.view(torch.int16))


@pytest.mark.parametrize("field,value,match", [
    ("k", 25, r"k_bits in \[1,24\]"),
    ("cfg", len(BASE_CONFIGS) + 1, "cfg must be 0"),
], ids=["unsupported-rung", "unsupported-cfg"])
def test_candidate_binding_fails_closed(field, value, match):
    k, K, E, N = 16, 512, 2, 64
    qw, lut, compose, type_size = _pack(
        k, K, E, N, seed=96016, source="synth")
    a = torch.randn(4, K, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(4, N, dtype=torch.bfloat16, device=DEV)
    args = dict(k=k, cfg=0)
    args[field] = value
    with pytest.raises(RuntimeError, match=match):
        ext.cb_moe_persistent_b_prefill_d2r(
            out, a, qw, lut, compose, _ends([1, 3]), args["k"],
            type_size, args["cfg"])
