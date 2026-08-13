"""Custom-op registration for Gridbook's native CUDA/CUTLASS kernels.

Registered through ``torch.library.custom_op`` with a ``register_fake`` so it is
opaque to Dynamo and safe under ``torch.compile`` / CUDA-graph capture (the GGUF
``ops.py`` pattern: a fake impl that returns a correctly-shaped empty tensor).
No vLLM import and no interpreted-kernel dependency lives here.
"""
from __future__ import annotations

import itertools
import weakref

import torch

# CUDA-graph capture safety (root-caused 2026-07-21; re-derived 2026-08-02).
#
# NO GRIDBOOK OP CARRIES ``torch.Tag.cudagraph_unsafe``, and none should.
#
# WHAT THE TAG DOES. Its only consumer is inductor's ``should_partition``,
# which is itself reachable only under ``use_inductor_graph_partition=True``
# combined with PIECEWISE cudagraphs. FULL capture ignores graph partitioning
# entirely, and vLLM leaves the partition wrapper off at every optimization
# level. So the tag is inert in every shipped configuration.
#
# WHY IT IS ALSO IRRELEVANT HERE. Since the M-branch hoist (5b4e5e7) dynamo
# sees ONLY ``cb_linear_forward`` / ``cb_moe_forward``. Every op below runs
# inside those two ops' eager implementations and is never a graph NODE, so
# its tags are metadata nothing reads. The published
# ``PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE`` knob had therefore been a silent no-op
# since the hoist landed, and is retired rather than left as a switch that
# looks like it does something.
#
# WHY NOT TAG THE TWO WHOLE-DISPATCH OPS INSTEAD. That is the one change that
# WOULD take effect, and it recreates the 2026-07-21 defect at worse
# granularity: every CB layer would become an eager partition boundary, and
# this torch/vLLM build mishandles the hand-off of an eager boundary op's
# output into the following cuda-graph-captured region (stale/aliased buffer)
# -> DETERMINISTIC output corruption ("408 408", CJK intrusions), while
# ``FULL_DECODE_ONLY`` (one graph, no internal boundary) and the historical
# inline prototype stayed correct. That root-cause record is kept here
# deliberately: it is the reason the tag must stay off, not a reason it was
# once needed.
#
# WHY THE TWO DISPATCH OPS ARE CAPTURE-SAFE BY CONSTRUCTION. Their M-branch
# and env reads resolve HOST-side at capture time, and the arms that would
# host-sync are the prefill arms, which are unreachable at captured decode
# sizes. Reproducing the historical corruption would now require reverting the
# hoist AND ``use_inductor_graph_partition=True`` with piecewise cudagraphs on
# a torch predating pytorch#165815.


@torch.library.custom_op("prismaquant::cb_gemv_fp8", mutates_args=())
def cb_gemv_fp8(x: torch.Tensor, qw_padded: torch.Tensor,
                cb_flat: torch.Tensor, cb_row_offset: torch.Tensor,
                scale: torch.Tensor, N: int, K: int, k_bits: int, n_sub: int,
                type_size: int) -> torch.Tensor:
    """CUDA decode-GEMV for FP8_CB (prototype ii): takes RAW bf16 activations
    and fuses the per-token fp8 dynamic QDQ + the bandwidth-bound dequant-GEMV
    into one op (two kernel launches vs the Triton path's ~7). The op fails
    closed when the native extension is unavailable."""
    from .cuda_ext import require_ext
    return require_ext("FP8-CB decode GEMV").cb_gemv_fp8(
        x, qw_padded, cb_flat, cb_row_offset, scale,
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
    in-register from the packed 9-byte plane. The op fails closed when the
    native extension is unavailable."""
    from .cuda_ext import require_ext
    return require_ext("FP4-CB v2 decode GEMV").cb_gemv_fp4_v2(
        xq, qw_padded, cb_flat, cb_row_offset, compose,
        N, K, k_bits, n_sub, type_size)


@cb_gemv_fp4_v2.register_fake
def _cb_gemv_fp4_v2_fake(xq, qw_padded, cb_flat, cb_row_offset, compose, N, K,
                         k_bits, n_sub, type_size):
    return torch.empty((*xq.shape[:-1], N), dtype=xq.dtype, device=xq.device)


@torch.library.custom_op("prismaquant::cb_expand_fp8", mutates_args=())
def cb_expand_fp8(qw_padded: torch.Tensor, cb_flat_fp8: torch.Tensor,
                  cb_row_offset: torch.Tensor, N: int, K: int, k_bits: int,
                  n_sub: int, type_size: int) -> torch.Tensor:
    """CUDA fp8-direct transient expand (prefill): packed rows -> a native
    [N, K] e4m3 tile for the direct per-channel CUTLASS GEMM. Registered as a
    custom op so vLLM's fullgraph compile (VLLM_COMPILE piecewise — the
    drafter-capture prerequisite) can trace the prefill path; raw pybind
    calls are dynamo-unsupported. The op fails closed when native code is not
    available."""
    from .cuda_ext import require_ext
    return require_ext("FP8-CB transient expansion").cb_expand_fp8(
        qw_padded, cb_flat_fp8, cb_row_offset,
        N, K, k_bits, n_sub, type_size)


@cb_expand_fp8.register_fake
def _cb_expand_fp8_fake(qw_padded, cb_flat_fp8, cb_row_offset, N, K, k_bits,
                        n_sub, type_size):
    return torch.empty((N, K), dtype=torch.float8_e4m3fn,
                       device=qw_padded.device)


@torch.library.custom_op("prismaquant::cb_expand_fp4_v2",
                         mutates_args=())
def cb_expand_fp4_v2(qw_flat: torch.Tensor, cb_flat: torch.Tensor,
                     compose: torch.Tensor, row0: int, nrows: int, K: int,
                     k_bits: int, type_size: int) -> torch.Tensor:
    """Native FP4-v2 product-codebook expansion into a BF16 transient."""
    from .cuda_ext import require_ext_v2
    return require_ext_v2("FP4-CB v2 transient expansion").cb_expand_v2(
        qw_flat, cb_flat, compose, row0, nrows, K, k_bits, type_size)


@cb_expand_fp4_v2.register_fake
def _cb_expand_fp4_v2_fake(qw_flat, cb_flat, compose, row0, nrows, K,
                           k_bits, type_size):
    return torch.empty((nrows, K), dtype=torch.bfloat16,
                       device=qw_flat.device)


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm",
                         mutates_args=())
def cb_bf16_grouped_mm(a: torch.Tensor, weights: torch.Tensor,
                       expert_ends: torch.Tensor,
                       expert_start: int = 0) -> torch.Tensor:
    """Native CUTLASS grouped BF16 GEMM.

    ``a`` contains expert-sorted activation rows, ``weights`` is a contiguous
    expert slice ``[E_local, N, K]``, and ``expert_ends`` contains cumulative
    row ends for the full expert stack. The allocating form zeroes rows outside
    the local expert slice; routed serving normally uses the in-place form
    below so every chunk writes directly into one caller-owned output.
    """
    from .cuda_ext import require_bf16_grouped_ext
    return require_bf16_grouped_ext(
        "CUTLASS grouped BF16 GEMM").cb_bf16_grouped_mm(
            a, weights, expert_ends, expert_start)


@cb_bf16_grouped_mm.register_fake
def _cb_bf16_grouped_mm_fake(a, weights, expert_ends, expert_start=0):
    return torch.empty((a.shape[0], weights.shape[1]), dtype=torch.bfloat16,
                       device=a.device)


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm_out",
                         mutates_args=("out",))
def cb_bf16_grouped_mm_out(out: torch.Tensor, a: torch.Tensor,
                           weights: torch.Tensor,
                           expert_ends: torch.Tensor,
                           expert_start: int = 0) -> None:
    """Write one expert slice through the native CUTLASS grouped BF16 op."""
    from .cuda_ext import require_bf16_grouped_ext
    require_bf16_grouped_ext(
        "CUTLASS grouped BF16 GEMM").cb_bf16_grouped_mm_out(
            out, a, weights, expert_ends, expert_start)


@cb_bf16_grouped_mm_out.register_fake
def _cb_bf16_grouped_mm_out_fake(out, a, weights, expert_ends,
                                 expert_start=0):
    return None


@torch.library.custom_op("prismaquant::fp8_source_gemv", mutates_args=())
def fp8_source_gemv(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor,
                     groups: int) -> torch.Tensor:
    """Raw-resident block128 source-FP8 GEMV for BF16 activations."""
    from .cuda_ext import require_fp8_source_w8a16_ext

    return require_fp8_source_w8a16_ext(
        "source-FP8 W8A16 decode GEMV", device=x.device
    ).fp8_source_gemv(x, q, scales, groups)


@fp8_source_gemv.register_fake
def _fp8_source_gemv_fake(x, q, scales, groups):
    del scales, groups
    return torch.empty((x.shape[0], q.shape[0]), dtype=torch.bfloat16,
                       device=x.device)


@torch.library.custom_op("prismaquant::fp8_source_expand_bf16",
                         mutates_args=())
def fp8_source_expand_bf16(q: torch.Tensor,
                           scales: torch.Tensor) -> torch.Tensor:
    """Expand one caller-scoped BF16 source-weight tile for native prefill."""
    from .cuda_ext import require_fp8_source_w8a16_ext

    return require_fp8_source_w8a16_ext(
        "source-FP8 W8A16 transient expansion", device=q.device
    ).fp8_source_expand_bf16(q, scales)


@fp8_source_expand_bf16.register_fake
def _fp8_source_expand_bf16_fake(q, scales):
    del scales
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm_sm120",
                         mutates_args=())
def cb_bf16_grouped_mm_sm120(a: torch.Tensor, weights: torch.Tensor,
                             expert_ids: torch.Tensor,
                             tile_m: int) -> torch.Tensor:
    """sm12x-native CUTLASS grouped BF16 GEMM (OPT-IN lane).

    ``a`` is the ROW-PADDED activation ``[Mp, K]`` whose every ``tile_m``-row
    block belongs to one expert, ``weights`` is the contiguous stack
    ``[E, N, K]``, and ``expert_ids[m_tile]`` names each block's expert (``-1``
    for a padding block). Same numerics class as the default SM80 lane — fp32
    accumulate, one bf16 round — with a different FP32 reduction order.
    """
    from .cuda_ext import require_bf16_grouped_ext
    return require_bf16_grouped_ext(
        "sm12x-native CUTLASS grouped BF16 GEMM").cb_bf16_grouped_mm_sm120(
            a, weights, expert_ids, tile_m)


@cb_bf16_grouped_mm_sm120.register_fake
def _cb_bf16_grouped_mm_sm120_fake(a, weights, expert_ids, tile_m):
    return torch.empty((a.shape[0], weights.shape[1]), dtype=torch.bfloat16,
                       device=a.device)


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm_sm120_out",
                         mutates_args=("out",))
def cb_bf16_grouped_mm_sm120_out(out: torch.Tensor, a: torch.Tensor,
                                 weights: torch.Tensor,
                                 expert_ids: torch.Tensor,
                                 tile_m: int) -> None:
    """Write the sm12x-native grouped BF16 result into a caller-owned tensor."""
    from .cuda_ext import require_bf16_grouped_ext
    require_bf16_grouped_ext(
        "sm12x-native CUTLASS grouped BF16 GEMM").cb_bf16_grouped_mm_sm120_out(
            out, a, weights, expert_ids, tile_m)


@cb_bf16_grouped_mm_sm120_out.register_fake
def _cb_bf16_grouped_mm_sm120_out_fake(out, a, weights, expert_ids, tile_m):
    return None


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm_sm120_gather",
                         mutates_args=())
def cb_bf16_grouped_mm_sm120_gather(a: torch.Tensor, row_src: torch.Tensor,
                                    weights: torch.Tensor,
                                    expert_ids: torch.Tensor,
                                    tile_m: int) -> torch.Tensor:
    """sm12x grouped BF16 GEMM, in-mainloop A-row gather mode (OPT-IN lane).

    ``a`` is the COMPACT activation ``[S, K]``; ``row_src`` is int32 ``[Mp]``
    naming the source row of every padded row (ids outside ``[0, S)`` are the
    padding rows and read zeros), so the row-padded activation copy of the
    plain mode is never materialized. Bit-identical output to running the
    padded-copy mode on the materialized gather of the same ``row_src``.
    """
    from .cuda_ext import require_bf16_grouped_ext
    return require_bf16_grouped_ext(
        "sm12x-native CUTLASS grouped BF16 GEMM"
    ).cb_bf16_grouped_mm_sm120_gather(a, row_src, weights, expert_ids, tile_m)


@cb_bf16_grouped_mm_sm120_gather.register_fake
def _cb_bf16_grouped_mm_sm120_gather_fake(a, row_src, weights, expert_ids,
                                          tile_m):
    return torch.empty((row_src.shape[0], weights.shape[1]),
                       dtype=torch.bfloat16, device=a.device)


@torch.library.custom_op("prismaquant::cb_bf16_grouped_mm_sm120_gather_out",
                         mutates_args=("out",))
def cb_bf16_grouped_mm_sm120_gather_out(out: torch.Tensor, a: torch.Tensor,
                                        row_src: torch.Tensor,
                                        weights: torch.Tensor,
                                        expert_ids: torch.Tensor,
                                        tile_m: int) -> None:
    """Write the gather-mode sm12x grouped BF16 result into a caller-owned
    ``[Mp, N]`` tensor."""
    from .cuda_ext import require_bf16_grouped_ext
    require_bf16_grouped_ext(
        "sm12x-native CUTLASS grouped BF16 GEMM"
    ).cb_bf16_grouped_mm_sm120_gather_out(out, a, row_src, weights,
                                          expert_ids, tile_m)


@cb_bf16_grouped_mm_sm120_gather_out.register_fake
def _cb_bf16_grouped_mm_sm120_gather_out_fake(out, a, row_src, weights,
                                              expert_ids, tile_m):
    return None


@torch.library.custom_op("prismaquant::cb_moe_persistent_b_prefill",
                         mutates_args=("out",))
def cb_moe_persistent_b_prefill(out: torch.Tensor, a: torch.Tensor,
                                qw: torch.Tensor, lut: torch.Tensor,
                                compose: torch.Tensor,
                                expert_ends: torch.Tensor, k_bits: int,
                                type_size: int, cfg: int) -> None:
    """Persistent-B grouped MoE decode-in-mainloop, into a caller-owned [P,N].

    ONE launch per projection stage over the exact routed segments. The packed
    FP4-CB-v2 expert bytes are decoded inside the mainloop, so no expanded
    ``[E,N,K]`` BF16 transient is allocated or written.
    """
    from .cuda_ext import require_moe_persistent_b_ext
    require_moe_persistent_b_ext(
        "persistent-B grouped MoE prefill").cb_moe_persistent_b_prefill(
            out, a, qw, lut, compose, expert_ends, k_bits, type_size, cfg)


@cb_moe_persistent_b_prefill.register_fake
def _cb_moe_persistent_b_prefill_fake(out, a, qw, lut, compose, expert_ends,
                                      k_bits, type_size, cfg):
    return None


@torch.library.custom_op("prismaquant::cb_moe_persistent_b_decode",
                         mutates_args=())
def cb_moe_persistent_b_decode(qw_flat: torch.Tensor, lut: torch.Tensor,
                               compose: torch.Tensor, row0: int, nrows: int,
                               K: int, k_bits: int,
                               type_size: int) -> torch.Tensor:
    """The persistent-B mainloop's decode stage, standalone (bit-exactness
    oracle; not a serving path)."""
    from .cuda_ext import require_moe_persistent_b_ext
    return require_moe_persistent_b_ext(
        "persistent-B MoE decode probe").cb_moe_persistent_b_decode(
            qw_flat, lut, compose, row0, nrows, K, k_bits, type_size)


@cb_moe_persistent_b_decode.register_fake
def _cb_moe_persistent_b_decode_fake(qw_flat, lut, compose, row0, nrows, K,
                                     k_bits, type_size):
    return qw_flat.new_empty((nrows, K), dtype=torch.bfloat16)


# NOTE (2026-08-01): `prismaquant::cb_expand_fp8_into` (the out-variant of
# cb_expand_fp8) and its `cb_expand_fp8_into_available` probe were registered
# here with zero serving call sites — residue of the L2-pinned per-expert
# scratch pipeline, which only ever needed an out-variant because a persisting
# access-policy window pins a fixed ADDRESS RANGE. That pipeline wedged live
# serving three times and was removed; these were removed with their kernel and
# binding per docs/audits/ultraplan_perf_2026-08-01.md §4. Prefill expansion is
# the allocating `cb_expand_fp8` above.


@torch.library.custom_op("prismaquant::fp8_act_qdq", mutates_args=())
def fp8_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)
    as a custom op for the compile path."""
    from .cuda_ext import require_ext
    return require_ext("FP8 activation QDQ").fp8_act_qdq(x)


@fp8_act_qdq.register_fake
def _fp8_act_qdq_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("prismaquant::fp4_act_qdq", mutates_args=())
def fp4_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused group-16 fp4 (E2M1) activation QDQ (bit-exact to
    codec.fp4_group16_act_qdq) as a custom op for the compile path."""
    from .cuda_ext import require_ext
    return require_ext("FP4 activation QDQ").fp4_act_qdq(x)


@fp4_act_qdq.register_fake
def _fp4_act_qdq_fake(x):
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        raise RuntimeError("fp4_act_qdq wants a CUDA bf16 tensor")
    if x.dim() < 1:
        raise RuntimeError("fp4_act_qdq needs at least one dimension")
    torch._check(x.shape[-1] > 0,
                 lambda: "fp4_act_qdq needs a positive last dimension")
    torch._check(
        x.shape[-1] % 16 == 0,
        lambda: "fp4_act_qdq needs a last dim that is a multiple of the fp4 "
                "group (16)")
    # Native calls x.contiguous() before allocating its output. FakeTensor
    # stride metadata must describe that same contiguous result.
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def fp4_act_qdq_or_codec(x: torch.Tensor) -> torch.Tensor:
    """Exact group-16 FP4 activation QDQ, native and fail-closed on CUDA.

    The eager codec remains a CPU numerical oracle. CUDA serving never falls
    back to it: a missing native symbol raises through :func:`fp4_act_qdq` so
    throughput measurements cannot silently exercise a host-dispatch path.
    """
    if x.is_cuda:
        if x.dtype is not torch.bfloat16:
            raise TypeError("native FP4 activation QDQ requires CUDA BF16")
        return fp4_act_qdq(x)
    from . import codec
    return codec.fp4_group16_act_qdq(x).to(torch.bfloat16)


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp4_v2", mutates_args=())
def cb_moe_gemv_fp4_v2(xq: torch.Tensor, qw: torch.Tensor,
                       cb_flat: torch.Tensor, compose: torch.Tensor,
                       pair_expert: torch.Tensor, pair_xrow: torch.Tensor,
                       k_bits: int, n_sub: int, type_size: int) -> torch.Tensor:
    """Grouped MoE decode GEMV, fp4-CB two-tier v2 (act-QDQ outside)."""
    from .cuda_ext import require_ext
    return require_ext("grouped FP4-CB v2 decode GEMV").cb_moe_gemv_fp4_v2(
        xq, qw, cb_flat, compose, pair_expert, pair_xrow,
        k_bits, n_sub, type_size)


@cb_moe_gemv_fp4_v2.register_fake
def _cb_moe_gemv_fp4_v2_fake(xq, qw, cb_flat, compose, pair_expert, pair_xrow,
                             k_bits, n_sub, type_size):
    return torch.empty((pair_expert.shape[0], qw.shape[1]), dtype=xq.dtype,
                       device=xq.device)


# CB-GEMV-v2. Same job, same inputs and same output contract as
# ``cb_moe_gemv_fp4_v2`` above — it is the smem-resident-dictionary
# reimplementation, and ``moe_gemv_select.cb_gemv_choice`` picks between the
# two per (layer, stack). Differences in the SIGNATURE only: no ``n_sub`` (v2 is
# product-mode only), plus ``rpb`` / ``dict_mode`` (rpb<=0 and dict_mode==0
# select the kernel's measured auto policies). The C++ launcher reads the
# decode contract from the environment per call, like the inherited kernel.
#
# Like every op above it carries NO ``cudagraph_unsafe`` tag — see the module
# header. The invariant used to be stated the other way round ("it MUST carry
# the same tagging as every op above"), which was true of the mechanism and
# wrong about the direction: tagging these ops makes each a graph-PARTITION
# BOUNDARY under ``use_inductor_graph_partition=True`` + PIECEWISE cudagraphs,
# and this build mishandles the hand-off, giving DETERMINISTIC output
# corruption. What must hold is that no op disagrees with its neighbours, and
# the agreed value is UNTAGGED.
#
# Separate JIT module (``get_ext_v2`` -> ``prismaquant_cb_v2_ext``), not a
# second source of the inherited one: both .cu files define
# ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)`` and would collide at link.
@torch.library.custom_op("prismaquant::cb_moe_gemv_v2", mutates_args=())
def cb_moe_gemv_v2(xq: torch.Tensor, qw: torch.Tensor,
                   cb_flat: torch.Tensor, compose: torch.Tensor,
                   pair_expert: torch.Tensor, pair_xrow: torch.Tensor,
                   k_bits: int, type_size: int, rpb: int,
                   dict_mode: int) -> torch.Tensor:
    """Grouped MoE decode GEMV, fp4-CB two-tier v2, smem-resident-dictionary
    kernel (act-QDQ outside)."""
    from .cuda_ext import require_ext_v2
    return require_ext_v2("grouped FP4-CB v2 decode GEMV").cb_gemv_v2(
        xq, qw, cb_flat, compose, pair_expert, pair_xrow,
        k_bits, type_size, rpb, dict_mode)


@cb_moe_gemv_v2.register_fake
def _cb_moe_gemv_v2_fake(xq, qw, cb_flat, compose, pair_expert, pair_xrow,
                         k_bits, type_size, rpb, dict_mode):
    return torch.empty((pair_expert.shape[0], qw.shape[1]), dtype=xq.dtype,
                       device=xq.device)


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp8", mutates_args=())
def cb_moe_gemv_fp8(xq: torch.Tensor, qw: torch.Tensor,
                    cb_flat_fp8: torch.Tensor, scale: torch.Tensor,
                    pair_expert: torch.Tensor, pair_xrow: torch.Tensor,
                    k_bits: int, n_sub: int, type_size: int) -> torch.Tensor:
    """Grouped MoE decode GEMV, fp8-CB v1 (act already fp8-QDQ'd)."""
    from .cuda_ext import require_ext
    return require_ext("grouped FP8-CB decode GEMV").cb_moe_gemv_fp8(
        xq, qw, cb_flat_fp8, scale, pair_expert, pair_xrow,
        k_bits, n_sub, type_size)


@cb_moe_gemv_fp8.register_fake
def _cb_moe_gemv_fp8_fake(xq, qw, cb_flat_fp8, scale, pair_expert, pair_xrow,
                          k_bits, n_sub, type_size):
    return torch.empty((pair_expert.shape[0], qw.shape[1]), dtype=xq.dtype,
                       device=xq.device)


@torch.library.custom_op("prismaquant::cb_moe_combine", mutates_args=())
def cb_moe_combine(y: torch.Tensor, pair_w: torch.Tensor,
                   tok_start: torch.Tensor, T: int) -> torch.Tensor:
    """Router-weighted per-token combine of grouped GEMV outputs."""
    from .cuda_ext import require_ext
    return require_ext("grouped CB router combine").cb_moe_combine(
        y, pair_w, tok_start, T)


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
# historical external prefill path.
#
# Layers register themselves here at process_weights_after_loading; the op
# carries only (tensors..., layer_id) and the impl/fake consult the registry.
# The id is a python int baked per call site — each layer module passes its
# own — so the traced graph binds each site to its layer statically.
_LAYER_REGISTRY: dict[int, tuple[weakref.ReferenceType,
                                 weakref.ReferenceType]] = {}
_LAYER_IDS = itertools.count()


# Warned once per process, not per layer: the condition is a property of the
# engine's compilation config, so N identical lines per model would bury it.
_CAPTURE_GATE_WARNED: list[bool] = []

# The token counts below which each dispatch resolves to a decode kernel. Above
# them the PREFILL arm runs, and the prefill arms take host reads (the chunked
# bridge reads per-expert block offsets), which cannot be captured. Kept here
# rather than imported from linear/moe so this check does not pull vLLM-bound
# modules into ops.py's import graph; the ratchet below pins them together.
_DENSE_DECODE_MAX = 8
_MOE_DECODE_MAX = 16


def warn_if_capture_sizes_exceed_the_decode_gates() -> None:
    """Report capture sizes that would record a PREFILL arm inside a graph.

    Gridbook's capture-safety argument (see this module's header) is
    SIZE-CONDITIONAL and, until now, unenforced: the two whole-dispatch ops are
    capture-safe because at captured decode sizes the host-syncing prefill arms
    are unreachable. Nothing checked that the engine's configured capture sizes
    actually stay under the gates that make that true.

    WARNS RATHER THAN RAISES, deliberately. The shape of vLLM's compilation
    config has moved across releases, so a strict read here would fail loads
    for a config-schema reason rather than a correctness one — the opposite of
    fail-closed's intent. Every step degrades to silence, and the message names
    the offending sizes and the gates so the operator can act.
    """
    if _CAPTURE_GATE_WARNED:
        return
    _CAPTURE_GATE_WARNED.append(True)
    try:
        from vllm.config import get_current_vllm_config

        compilation = get_current_vllm_config().compilation_config
        sizes = [int(s) for s in
                 (getattr(compilation, "cudagraph_capture_sizes", None) or ())]
    except Exception:  # noqa: BLE001 — advisory only; never break a load
        return
    gate = min(_DENSE_DECODE_MAX, _MOE_DECODE_MAX)
    over = sorted(s for s in sizes if s > gate)
    if not over:
        return
    import sys

    print(
        "[prismaquant-cb] WARNING: cudagraph_capture_sizes includes "
        f"{over}, above Gridbook's decode gates (dense M<={_DENSE_DECODE_MAX}, "
        f"routed tokens<={_MOE_DECODE_MAX}). At those sizes the captured graph "
        "records the PREFILL arm, which performs host reads (the chunked BF16 "
        "bridge reads per-expert block offsets) and is not capturable. Use "
        "cudagraph_mode=FULL_DECODE_ONLY, or keep the capture sizes at or "
        f"below {gate}.", file=sys.stderr, flush=True)


def register_cb_layer(method, layer) -> int:
    """Register one compiled-dispatch target without owning its model.

    vLLM can load and unload several engines in one Python process.  Strong
    references here used to pin every method, layer, parameter, and associated
    GPU allocation until process exit.  The module/quant-method ownership graph
    is authoritative; this registry is only an integer indirection for custom
    ops, so both references must be weak.

    IDs are monotonic rather than derived from ``len(_LAYER_REGISTRY)``.  A
    captured graph containing an expired ID can therefore never alias a newly
    loaded layer after the weakref callback removes its old entry.
    """
    # Every CB layer, dense and routed, funnels through here at model load, so
    # this is the one place the size-conditional capture-safety argument can be
    # checked against the engine's actual configuration.
    warn_if_capture_sizes_exceed_the_decode_gates()
    layer_id = next(_LAYER_IDS)

    def expire(_ref, *, expected_id=layer_id):
        _LAYER_REGISTRY.pop(expected_id, None)

    method_ref = weakref.ref(method, expire)
    layer_ref = weakref.ref(layer, expire)
    _LAYER_REGISTRY[layer_id] = (method_ref, layer_ref)
    return layer_id


def _lookup_cb_layer(layer_id: int):
    entry = _LAYER_REGISTRY.get(layer_id)
    if entry is None:
        raise RuntimeError(
            f"CB dispatch layer id {layer_id} is stale or unknown; the model "
            "may have been unloaded")
    method = entry[0]()
    layer = entry[1]()
    if method is None or layer is None:
        # A callback normally removes the entry first.  Keep this explicit for
        # finalizer ordering and interpreter-shutdown edge cases.
        _LAYER_REGISTRY.pop(layer_id, None)
        raise RuntimeError(
            f"CB dispatch layer id {layer_id} expired after model unload")
    return method, layer


@torch.library.custom_op("prismaquant::cb_linear_forward", mutates_args=())
def cb_linear_forward(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    method, layer = _lookup_cb_layer(layer_id)
    return method._apply_inline(layer, x)


@cb_linear_forward.register_fake
def _cb_linear_forward_fake(x, layer_id):
    _method, layer = _lookup_cb_layer(layer_id)
    return torch.empty((*x.shape[:-1], layer._cb_N), dtype=x.dtype,
                       device=x.device)


@torch.library.custom_op("prismaquant::fp8_source_linear_forward",
                         mutates_args=())
def fp8_source_linear_forward(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Opaque whole-dispatch op for raw-resident source-FP8 W8A16."""
    method, layer = _lookup_cb_layer(layer_id)
    return method._apply_inline(layer, x)


@fp8_source_linear_forward.register_fake
def _fp8_source_linear_forward_fake(x, layer_id):
    _method, layer = _lookup_cb_layer(layer_id)
    groups = int(layer._fp8_source_groups)
    rows = int(layer._fp8_source_rows)
    if bool(getattr(layer, "is_bmm", False)):
        return torch.empty((*x.shape[:-2], groups, rows), dtype=x.dtype,
                           device=x.device)
    return torch.empty((*x.shape[:-1], int(layer._fp8_source_N)),
                       dtype=x.dtype, device=x.device)


def _neutralize_moe_padding_sentinel(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make vLLM's ``-1`` padded routes inert without a device-to-host read.

    With ``VLLM_MOE_SKIP_PADDING``, vLLM preserves the static token shape and
    marks every routed slot of a padded token with expert id ``-1``.  Gridbook
    owns several routed implementations, all behind :func:`cb_moe_forward`;
    normalizing at that opaque boundary keeps every implementation safe and
    keeps the pointwise work out of Dynamo's graph.

    Only the documented ``-1`` sentinel is rewritten.  Every other id remains
    unchanged; validation of generally corrupt router output is a separate
    contract and is not implied here.  The corresponding weight is set to
    exact zero (rather than multiplied by a mask, which would preserve NaN),
    making the padded routed contribution inert.  This does not touch ``x``
    or the separately scheduled shared-expert path.
    """
    padding = topk_ids == -1
    return (topk_weights.masked_fill(padding, 0),
            topk_ids.masked_fill(padding, 0))


@torch.library.custom_op("prismaquant::cb_moe_forward", mutates_args=())
def cb_moe_forward(x: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
    method, layer = _lookup_cb_layer(layer_id)
    topk_weights, topk_ids = _neutralize_moe_padding_sentinel(
        topk_weights, topk_ids)
    return method._apply_inline(layer, x, topk_weights, topk_ids)


@cb_moe_forward.register_fake
def _cb_moe_forward_fake(x, topk_weights, topk_ids, layer_id):
    return torch.empty_like(x)
