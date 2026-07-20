"""Custom-op registration for the CB decode-GEMM.

Registered through ``torch.library.custom_op`` with a ``register_fake`` so it is
opaque to Dynamo and safe under ``torch.compile`` / CUDA-graph capture (the GGUF
`ops.py` pattern: a fake impl that returns a correctly-shaped empty tensor). No
vLLM import here, so the op is usable from the standalone correctness tests too.
"""
from __future__ import annotations

import torch

from .kernels import cb_decode_linear


@torch.library.custom_op("prismaquant::cb_gemm", mutates_args=())
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


@torch.library.custom_op("prismaquant::cb_gemv_fp8", mutates_args=())
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


@torch.library.custom_op("prismaquant::cb_gemv_fp4_v2", mutates_args=())
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
