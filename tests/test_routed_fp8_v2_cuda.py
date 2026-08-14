"""Bit-exact and graph-replay gates for routed K28 whole-row FP8-CB GEMV."""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from gridbook.cuda_ext import get_ext  # noqa: E402
from gridbook import ops as pq_ops  # noqa: E402

EXT = get_ext()
if EXT is None or not hasattr(EXT, "cb_moe_gemv_fp8_v2"):
    pytest.fail(
        "CUDA is available but the routed FP8-v2 extension/symbol is not; "
        "refusing a vacuous module skip on the target host",
        pytrace=False,
    )


def _case(k: int, n: int, p: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    experts = 4
    xrows = 8
    row_bytes = (k // 256) * 112
    return [
        torch.randn((xrows, k), generator=gen, device="cuda",
                    dtype=torch.bfloat16),
        torch.randint(0, 256, (experts, n, row_bytes), generator=gen,
                      device="cuda", dtype=torch.uint8),
        (torch.randn((1024,), generator=gen, device="cuda") * 80.0)
        .clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        .contiguous(),
        (torch.rand((experts, n), generator=gen, device="cuda")
         * (0.004 - 0.00005) + 0.00005).contiguous(),
        torch.randint(0, experts, (p,), generator=gen, device="cuda",
                      dtype=torch.int32),
        torch.randint(0, xrows, (p,), generator=gen, device="cuda",
                      dtype=torch.int32),
    ]


def _inherited(args):
    return EXT.cb_moe_gemv_fp8(*args, 28, 4, 112)


def _whole_row(args):
    return EXT.cb_moe_gemv_fp8_v2(*args, 28, 4, 112)


def _whole_row_registered(args):
    return pq_ops.cb_moe_gemv_fp8_v2(*args, 28, 4, 112)


def _assert_storage_equal(reference, candidate):
    assert torch.equal(reference.view(torch.uint16),
                       candidate.view(torch.uint16))


@pytest.mark.parametrize("contract", ["v1", "v2"])
@pytest.mark.parametrize(
    ("k", "n", "p"),
    [(2048, 65, 5), (4096, 79, 7),
     (2048, 4096, 30), (4096, 2048, 36)],
)
def test_routed_fp8_v2_matches_inherited_storage_bits(
        monkeypatch, contract, k, n, p):
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE_CONTRACT", contract)
    args = _case(k, n, p, seed=k + n + p)
    _assert_storage_equal(_inherited(args), _whole_row(args))
    _assert_storage_equal(_inherited(args), _whole_row_registered(args))


@pytest.mark.parametrize("contract", ["v1", "v2"])
@pytest.mark.parametrize(
    ("k", "n", "p"), [(2048, 4096, 30), (4096, 2048, 36)])
def test_routed_fp8_v2_live_graph_replay(monkeypatch, contract, k, n, p):
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE_CONTRACT", contract)
    args = _case(k, n, p, seed=2 * k + n + p)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _inherited(args)
            _whole_row_registered(args)
    torch.cuda.current_stream().wait_stream(side)

    inherited_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(inherited_graph):
        reference = _inherited(args)
    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        candidate = _whole_row_registered(args)

    gen = torch.Generator(device="cuda").manual_seed(7 * k + n)
    for _ in range(128):
        args[0].copy_(torch.randn(args[0].shape, generator=gen, device="cuda",
                                  dtype=torch.bfloat16))
        args[3].copy_(torch.rand(args[3].shape, generator=gen, device="cuda")
                      * 0.08 + 0.002)
        args[4].copy_(torch.randint(0, args[1].shape[0], args[4].shape,
                                    generator=gen, device="cuda",
                                    dtype=torch.int32))
        args[5].copy_(torch.randint(0, args[0].shape[0], args[5].shape,
                                    generator=gen, device="cuda",
                                    dtype=torch.int32))
        inherited_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize()
        _assert_storage_equal(reference, candidate)


def test_routed_fp8_v2_registered_op_fullgraph_contract(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE_CONTRACT", "v1")
    args = _case(4096, 2048, 36, seed=8192)
    x, qw, cb, scale, pair_expert, pair_xrow = args

    def run(x_arg):
        return pq_ops.cb_moe_gemv_fp8_v2(
            x_arg, qw, cb, scale, pair_expert, pair_xrow, 28, 4, 112)

    eager = run(x)
    compiled = torch.compile(run, backend="eager", fullgraph=True)(x)
    _assert_storage_equal(eager, compiled)


@pytest.mark.parametrize(
    ("k_bits", "n_sub", "type_size", "message"),
    [(27, 4, 108, "K28"), (28, 2, 112, "K28"),
     (28, 4, 111, "K28")],
)
def test_routed_fp8_v2_rejects_other_layouts(
        k_bits, n_sub, type_size, message):
    args = _case(2048, 64, 6, seed=99)
    with pytest.raises(RuntimeError, match=message):
        EXT.cb_moe_gemv_fp8_v2(
            *args, k_bits, n_sub, type_size)


def test_routed_fp8_v2_rejects_other_width():
    args = _case(1024, 64, 6, seed=100)
    with pytest.raises(RuntimeError, match="K=2048 or K=4096"):
        EXT.cb_moe_gemv_fp8_v2(*args, 28, 4, 112)
