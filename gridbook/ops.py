"""Custom-op registration for the CB decode-GEMM.

Registered through ``torch.library.custom_op`` with a ``register_fake`` so it is
opaque to Dynamo and safe under ``torch.compile`` / CUDA-graph capture (the GGUF
`ops.py` pattern: a fake impl that returns a correctly-shaped empty tensor). No
vLLM import here, so the op is usable from the standalone correctness tests too.
"""
from __future__ import annotations

import torch

import os

from .kernels import cb_decode_linear

# CUDA-graph capture safety (root-caused 2026-07-21).
#
# These decode/expand ops are pure, fixed-shape-per-graph, and do NO host sync
# in their captured kernel launches (the M-branch and env reads are host-side and
# resolve at capture time), so they ARE cuda-graph-capturable — proven by the
# working ``cudagraph_mode=FULL`` path, which captures them whole.
#
# Tagging them ``cudagraph_unsafe`` is actively HARMFUL under
# ``use_inductor_graph_partition=True`` + PIECEWISE cudagraphs: the tag forces
# each op to become an inductor graph-PARTITION BOUNDARY, and this torch/vLLM
# build mishandles the hand-off of an eager boundary op's output into the
# following cuda-graph-captured region (stale/aliased buffer) -> DETERMINISTIC
# output corruption ("408 408", CJK intrusions), while ``FULL_DECODE_ONLY`` (one
# graph, no internal boundary) and the pure-Triton path (no custom op) stay
# correct. Keeping the ops INSIDE the captured partition (no tag) fixes it.
#
# Default: NOT unsafe (stay inside the partition). Set
# ``PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE=1`` to restore the old boundary behaviour
# for an A/B (reproduces the corruption on the compile+piecewise path).
_PQ_UNSAFE = ((torch.Tag.cudagraph_unsafe,)
              if os.environ.get("PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE") == "1"
              else ())


@torch.library.custom_op("prismaquant::cb_gemm", mutates_args=(), tags=_PQ_UNSAFE)
def cb_gemm(x: torch.Tensor, qw_padded: torch.Tensor, cb_flat: torch.Tensor,
            cb_row_offset: torch.Tensor, scale: torch.Tensor,
            compose: torch.Tensor, N: int, K: int,
            k_bits: int, n_sub: int, type_size: int,
            is_fp4: bool, is_v2: bool) -> torch.Tensor:
    return cb_decode_linear(x, qw_padded, cb_flat, cb_row_offset, scale,
                            compose, N=N, K=K, k_bits=k_bits, n_sub=n_sub,
                            type_size=type_size, is_fp4=is_fp4, is_v2=is_v2)


@cb_gemm.register_fake
def _cb_gemm_fake(x, qw_padded, cb_flat, cb_row_offset, scale, compose, N, K,
                  k_bits, n_sub, type_size, is_fp4, is_v2):
    return torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)


@torch.library.custom_op("prismaquant::cb_gemv_fp8", mutates_args=(), tags=_PQ_UNSAFE)
def cb_gemv_fp8(x: torch.Tensor, qw_padded: torch.Tensor,
                cb_flat: torch.Tensor, cb_row_offset: torch.Tensor,
                scale: torch.Tensor, N: int, K: int, k_bits: int, n_sub: int,
                type_size: int) -> torch.Tensor:
    """CUDA decode-GEMV for FP8_CB (prototype ii): takes RAW bf16 activations
    and fuses the per-token fp8 dynamic QDQ + the bandwidth-bound dequant-GEMV
    into one op (two kernel launches vs the Triton path's ~7). Caller must
    check ``cuda_ext.get_ext()`` first."""
    from .cuda_ext import get_ext
    return get_ext().cb_gemv_fp8(x, qw_padded, cb_flat, cb_row_offset, scale,
                                 N, K, k_bits, n_sub, type_size, True)


@cb_gemv_fp8.register_fake
def _cb_gemv_fp8_fake(x, qw_padded, cb_flat, cb_row_offset, scale, N, K,
                      k_bits, n_sub, type_size):
    return torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)


@torch.library.custom_op("prismaquant::cb_gemv_fp4_v2", mutates_args=(), tags=_PQ_UNSAFE)
def cb_gemv_fp4_v2(xq: torch.Tensor, qw_padded: torch.Tensor,
                   cb_flat: torch.Tensor, cb_row_offset: torch.Tensor,
                   compose: torch.Tensor, N: int, K: int, k_bits: int,
                   n_sub: int, type_size: int) -> torch.Tensor:
    """Dense CUDA decode-GEMV for fp4-CB two-tier (v2): takes ALREADY act-QDQ'd
    bf16 activations (fp4 group-16 RTN runs in ``codec``, OUTSIDE the kernel —
    same as the Triton fp4 path — so CUDA-vs-Triton numerics stay aligned) and
    runs the bandwidth-bound dequant-GEMV, composing the two-tier weight scale
    in-register from the packed 9-byte plane. Caller must check
    ``cuda_ext.get_ext()`` first."""
    from .cuda_ext import get_ext
    return get_ext().cb_gemv_fp4_v2(xq, qw_padded, cb_flat, cb_row_offset,
                                    compose, N, K, k_bits, n_sub, type_size)


@cb_gemv_fp4_v2.register_fake
def _cb_gemv_fp4_v2_fake(xq, qw_padded, cb_flat, cb_row_offset, compose, N, K,
                         k_bits, n_sub, type_size):
    return torch.empty((*xq.shape[:-1], N), dtype=xq.dtype, device=xq.device)


@torch.library.custom_op("prismaquant::cb_expand_fp8", mutates_args=(), tags=_PQ_UNSAFE)
def cb_expand_fp8(qw_padded: torch.Tensor, cb_flat_fp8: torch.Tensor,
                  cb_row_offset: torch.Tensor, N: int, K: int, k_bits: int,
                  n_sub: int, type_size: int) -> torch.Tensor:
    """CUDA fp8-direct transient expand (prefill): packed rows -> a native
    [N, K] e4m3 tile for the stock per-channel fp8 GEMM. Registered as a
    custom op so vLLM's fullgraph compile (VLLM_COMPILE piecewise — the
    drafter-capture prerequisite) can trace the prefill path; raw pybind
    calls are dynamo-unsupported. Caller must check ``cuda_ext.get_ext()``."""
    from .cuda_ext import get_ext
    return get_ext().cb_expand_fp8(qw_padded, cb_flat_fp8, cb_row_offset,
                                   N, K, k_bits, n_sub, type_size)


@cb_expand_fp8.register_fake
def _cb_expand_fp8_fake(qw_padded, cb_flat_fp8, cb_row_offset, N, K, k_bits,
                        n_sub, type_size):
    return torch.empty((N, K), dtype=torch.float8_e4m3fn,
                       device=qw_padded.device)


@torch.library.custom_op("prismaquant::fp8_act_qdq", mutates_args=(), tags=_PQ_UNSAFE)
def fp8_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)
    as a custom op for the compile path."""
    from .cuda_ext import get_ext
    return get_ext().fp8_act_qdq(x)


@fp8_act_qdq.register_fake
def _fp8_act_qdq_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp4_v2", mutates_args=(), tags=_PQ_UNSAFE)
def cb_moe_gemv_fp4_v2(xq: torch.Tensor, qw: torch.Tensor,
                       cb_flat: torch.Tensor, compose: torch.Tensor,
                       pair_expert: torch.Tensor, pair_xrow: torch.Tensor,
                       k_bits: int, n_sub: int, type_size: int) -> torch.Tensor:
    """Grouped MoE decode GEMV, fp4-CB two-tier v2 (act-QDQ outside)."""
    from .cuda_ext import get_ext
    return get_ext().cb_moe_gemv_fp4_v2(xq, qw, cb_flat, compose, pair_expert,
                                        pair_xrow, k_bits, n_sub, type_size)


@cb_moe_gemv_fp4_v2.register_fake
def _cb_moe_gemv_fp4_v2_fake(xq, qw, cb_flat, compose, pair_expert, pair_xrow,
                             k_bits, n_sub, type_size):
    return torch.empty((pair_expert.shape[0], qw.shape[1]), dtype=xq.dtype,
                       device=xq.device)


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp8", mutates_args=(), tags=_PQ_UNSAFE)
def cb_moe_gemv_fp8(xq: torch.Tensor, qw: torch.Tensor,
                    cb_flat_fp8: torch.Tensor, scale: torch.Tensor,
                    pair_expert: torch.Tensor, pair_xrow: torch.Tensor,
                    k_bits: int, n_sub: int, type_size: int) -> torch.Tensor:
    """Grouped MoE decode GEMV, fp8-CB v1 (act already fp8-QDQ'd)."""
    from .cuda_ext import get_ext
    return get_ext().cb_moe_gemv_fp8(xq, qw, cb_flat_fp8, scale, pair_expert,
                                     pair_xrow, k_bits, n_sub, type_size)


@cb_moe_gemv_fp8.register_fake
def _cb_moe_gemv_fp8_fake(xq, qw, cb_flat_fp8, scale, pair_expert, pair_xrow,
                          k_bits, n_sub, type_size):
    return torch.empty((pair_expert.shape[0], qw.shape[1]), dtype=xq.dtype,
                       device=xq.device)


@torch.library.custom_op("prismaquant::cb_moe_combine", mutates_args=(), tags=_PQ_UNSAFE)
def cb_moe_combine(y: torch.Tensor, pair_w: torch.Tensor,
                   tok_start: torch.Tensor, T: int) -> torch.Tensor:
    """Router-weighted per-token combine of grouped GEMV outputs."""
    from .cuda_ext import get_ext
    return get_ext().cb_moe_combine(y, pair_w, tok_start, T)


@cb_moe_combine.register_fake
def _cb_moe_combine_fake(y, pair_w, tok_start, T):
    return torch.empty((T, y.shape[1]), dtype=y.dtype, device=y.device)


# ---------------------------------------------------------------------------
# M-branch hoist (compile-lane fix, 2026-07-21). torch.compile traces the
# model ONCE at a prefill-sized example, so an in-graph `M <= threshold`
# branch bakes the PREFILL path into the single dynamic graph — compiled
# decode then runs the 2x-heavier transient-expand path (inductor dump:
# cb_expand_fp8 x3248, cb_gemv_fp8 x0) and the large expand transient sits on
# the fragile partition boundary that produced the capture corruption. The
# fix: the ENTIRE per-layer dispatch becomes ONE opaque custom op whose EAGER
# implementation does the M-branch at call time. Piecewise/full capture at a
# decode size records the GEMV kernels (the branch resolves correctly at
# capture time, per captured size); prefill executes the same op eagerly and
# takes the expand path. The MoE prefill loop's host syncs become invisible
# to dynamo (they run inside the op), so compile no longer requires the
# stock prefill path.
#
# Layers register themselves here at process_weights_after_loading; the op
# carries only (tensors..., layer_id) and the impl/fake consult the registry.
# The id is a python int baked per call site — each layer module passes its
# own — so the traced graph binds each site to its layer statically.
_LAYER_REGISTRY: dict = {}
_DISPATCH_VIA_OP: list = []


def dispatch_via_op() -> bool:
    if not _DISPATCH_VIA_OP:
        _DISPATCH_VIA_OP.append(
            os.environ.get("PRISMAQUANT_CB_DISPATCH", "op") != "inline")
    return _DISPATCH_VIA_OP[0]


def register_cb_layer(method, layer) -> int:
    layer_id = len(_LAYER_REGISTRY)
    _LAYER_REGISTRY[layer_id] = (method, layer)
    return layer_id


@torch.library.custom_op("prismaquant::cb_linear_forward", mutates_args=())
def cb_linear_forward(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    method, layer = _LAYER_REGISTRY[layer_id]
    return method._apply_inline(layer, x)


@cb_linear_forward.register_fake
def _cb_linear_forward_fake(x, layer_id):
    _method, layer = _LAYER_REGISTRY[layer_id]
    return torch.empty((*x.shape[:-1], layer._cb_N), dtype=x.dtype,
                       device=x.device)


@torch.library.custom_op("prismaquant::cb_moe_forward", mutates_args=())
def cb_moe_forward(x: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
    method, layer = _LAYER_REGISTRY[layer_id]
    return method._apply_inline(layer, x, topk_weights, topk_ids)


@cb_moe_forward.register_fake
def _cb_moe_forward_fake(x, topk_weights, topk_ids, layer_id):
    return torch.empty_like(x)
