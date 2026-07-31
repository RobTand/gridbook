"""GB10 correctness and execution-contract tests for CB-GEMV-v2.

These tests use PrismaQuant's real two-tier encoder rather than fabricating a
packed scale plane. They skip cleanly without CUDA/nvcc or outside the narrow
GB10 device family on which the experimental kernel is admitted.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
from gridbook import ops as pq_ops  # noqa: E402
from gridbook.cuda_ext import get_ext, get_ext_v2  # noqa: E402
from gridbook.moe_gemv_select import cb_gemv_v2_device_support  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)
supported, reason, _device_index = cb_gemv_v2_device_support("cuda:0")
if not supported:
    pytest.skip(reason, allow_module_level=True)
inherited = get_ext()
v2ext = get_ext_v2()
if inherited is None or v2ext is None:
    pytest.skip("required CUDA extensions could not be built",
                allow_module_level=True)
v2ext.cb_gemv_v2_prepare()


@contextmanager
def _env(**values):
    old = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _case(k: int, K: int, *, seed: int = 1, E: int = 3, N: int = 17):
    device = torch.device("cuda")
    cb = pq._resolve_codebook(k, "fp4", "product", None, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn(E, N, K, generator=generator) * 0.02).to(device)
    fields = pq.nvfp4_cb_fields(
        weight, k, grid="fp4", mode="product", codebook=cb,
        scale_coding="two_tier", encode_tier="fast")
    packed = pq.nvfp4_cb_assemble_bytes(
        fields, k, grid="fp4", mode="product")
    type_size = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    qw = packed.reshape(E, N, (K // 256) * type_size).contiguous().to(device)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    cb_flat = codec.build_flat_codebook(subs)
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(device)
    x = torch.randn(4, K, generator=generator,
                    dtype=torch.bfloat16).to(device)
    pair_expert = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32,
                               device=device)
    pair_xrow = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.int32,
                             device=device)
    return x, qw, cb_flat, compose, pair_expert, pair_xrow, type_size


def _run_v2(case, k, *, dict_mode=0):
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    return v2ext.cb_gemv_v2(
        x, qw, cb, compose, pair_expert, pair_xrow,
        k, type_size, 0, dict_mode)


def _run_inherited_rowpack(case, k):
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    return inherited.cb_moe_gemv_fp4_v2(
        x, qw, cb, compose, pair_expert, pair_xrow, k, 2, type_size)


@pytest.mark.parametrize("k,K", [
    (13, 512),
    (16, 1536),
    (20, 2048),
    (24, 4096),
])
@pytest.mark.parametrize("contract", [None, "v2"])
def test_v2_is_bit_exact_to_inherited_rowpack(k, K, contract):
    case = _case(k, K, seed=k + K)
    with _env(PRISMAQUANT_CB_W2_SCHED="rowpack",
              PRISMAQUANT_CB_DECODE_CONTRACT=contract):
        want = _run_inherited_rowpack(case, k)
        got = _run_v2(case, k)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_compiled_dispatch_predicate_covers_measured_wall_and_invalid_inputs():
    assert v2ext.cb_gemv_v2_prefers_inherited(24, 105, 2048) is True
    assert v2ext.cb_gemv_v2_prefers_inherited(24, 105, 1536) is False
    assert v2ext.cb_gemv_v2_prefers_inherited(20, 89, 4096) is False
    assert v2ext.cb_gemv_v2_prefers_inherited(0, 9, 4096) is True
    assert v2ext.cb_gemv_v2_prefers_inherited(16, 73, 0) is True


def test_cuda_graph_replay_matches_eager():
    k, K = 16, 512
    case = _case(k, K, seed=77)
    with _env(PRISMAQUANT_CB_DECODE_CONTRACT="v2"):
        eager = _run_v2(case, k)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _run_v2(case, k)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


def test_custom_op_torch_compile_contract():
    k, K = 16, 512
    case = _case(k, K, seed=88)
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case

    def run(x_arg):
        return pq_ops.cb_moe_gemv_v2(
            x_arg, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 0, 0)

    eager = run(x)
    compiled = torch.compile(run, backend="eager", fullgraph=True)(x)
    assert torch.equal(compiled.view(torch.int16), eager.view(torch.int16))


def test_debug_bindings_reject_malformed_inputs():
    k, K = 16, 512
    case = _case(k, K, seed=99)
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    with pytest.raises(RuntimeError, match="compose must"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose.cpu(), pair_expert, pair_xrow,
            k, type_size, 0, 0)
    with pytest.raises(RuntimeError, match="dict_mode"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 0, 4)
    with pytest.raises(RuntimeError, match="rpb"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 1025, 0)
    with pytest.raises(RuntimeError, match="same length"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow[:-1],
            k, type_size, 0, 0)
    flat = qw.reshape(-1)
    with pytest.raises(RuntimeError, match="positive.*multiple of 256"):
        v2ext.cb_expand_v2(flat, cb, compose, 0, 1, 511, k, type_size)
    rows_total = flat.numel() // ((K // 256) * type_size)
    with pytest.raises(RuntimeError, match="outside"):
        v2ext.cb_expand_v2(
            flat, cb, compose, rows_total, 1, K, k, type_size)
