"""Correctness gates for the owned CUTLASS BF16 grouped bridge.

The bridge consumes already-quantized BF16 activations and already-expanded
BF16 weights. Those two tensors are the quality contract; the grouped kernel
may differ from a per-expert GEMM only by FP32 summation order. Tests therefore
compare both implementations to an FP32 reference made from the *same* BF16
operands, and separately ratchet the chunked in-place serving topology.

CUDA-only: the module skips in build environments without the serving
toolchain, matching the native-kernel suite convention.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

ops = pytest.importorskip("gridbook.ops")
from gridbook.cuda_ext import get_bf16_grouped_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

ext = get_bf16_grouped_ext()
if ext is None:
    pytest.skip("CUTLASS grouped BF16 extension unavailable",
                allow_module_level=True)

DEV = "cuda"


def _routing_counts(experts: int, active: dict[int, int]):
    counts = torch.zeros(experts, dtype=torch.int32, device=DEV)
    for expert, rows in active.items():
        counts[expert] = rows
    return counts, counts.cumsum(0, dtype=torch.int32).contiguous()


def _per_expert_references(a, weights, expert_ends):
    """Return BF16 F.linear and highest-precision FP32 references."""
    y_bf16 = torch.empty(
        (a.shape[0], weights.shape[1]), dtype=torch.bfloat16, device=DEV)
    y_fp32 = torch.empty(
        (a.shape[0], weights.shape[1]), dtype=torch.float32, device=DEV)
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


def test_repeated_endpoints_and_chunked_output_match_single_launch():
    """Sixty-one empty experts and chunk boundaries cannot drop/shift rows."""
    torch.manual_seed(7)
    experts, k, n = 64, 512, 256
    _, ends = _routing_counts(experts, {0: 3, 31: 5, 63: 4})
    pairs = 12
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(
        experts, n, k, device=DEV, dtype=torch.bfloat16)

    single = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    chunked = torch.empty_like(single)
    for start in range(0, experts, 16):
        stop = min(experts, start + 16)
        ops.cb_bf16_grouped_mm_out(
            chunked, a, weights[start:stop].contiguous(), ends, start)

    assert torch.equal(chunked, single)
    assert torch.isfinite(single).all()


def test_fp32_accumulation_error_matches_bf16_linear_reference():
    """CUTLASS cannot consume materially more error than per-expert GEMM.

    K=4096 makes accumulation order visible while the sparse E=32 routing
    keeps the test bounded. Twenty-nine repeated endpoints cover the empty-
    expert scheduler case in the same numerical gate.
    """
    torch.manual_seed(20260801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    experts, k, n = 32, 4096, 512
    active = {0: 7, 13: 19, 31: 11}
    _, ends = _routing_counts(experts, active)
    pairs = sum(active.values())
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(
        experts, n, k, device=DEV, dtype=torch.bfloat16)

    cutlass = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    bf16_linear, fp32 = _per_expert_references(a, weights, ends)
    cutlass_rel = _rel_l2(cutlass, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)

    # Absolute backstop for BF16 output rounding plus a relative comparison to
    # the prior per-expert BF16 GEMM. Live GB10 attestation at E=64/N=1024 gave
    # 1.673277e-3 vs 1.673275e-3 (ratio 1.0000015).
    assert cutlass_rel <= 2e-3
    assert cutlass_rel <= torch.maximum(
        1.25 * linear_rel, linear_rel + 2e-5)


def test_all_empty_problem_has_a_well_formed_empty_output():
    experts, k, n = 16, 256, 128
    ends = torch.zeros(experts, dtype=torch.int32, device=DEV)
    a = torch.empty(0, k, dtype=torch.bfloat16, device=DEV)
    weights = torch.randn(
        experts, n, k, dtype=torch.bfloat16, device=DEV)
    out = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    assert out.shape == (0, n)
    assert out.dtype is torch.bfloat16
