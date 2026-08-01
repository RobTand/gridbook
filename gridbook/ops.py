"""Custom-op registration for Gridbook's native CUDA/CUTLASS kernels.

Registered through ``torch.library.custom_op`` with a ``register_fake`` so it is
opaque to Dynamo and safe under ``torch.compile`` / CUDA-graph capture (the GGUF
``ops.py`` pattern: a fake impl that returns a correctly-shaped empty tensor).
No vLLM import and no interpreted-kernel dependency lives here.
"""
from __future__ import annotations

import itertools
import os
import weakref

import torch

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
# graph, no internal boundary) and the historical inline prototype stayed
# correct. Keeping the ops INSIDE the captured partition (no tag) fixes it.
#
# Default: NOT unsafe (stay inside the partition). Set
# ``PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE=1`` to restore the old boundary behaviour
# for an A/B (reproduces the corruption on the compile+piecewise path).
_PQ_UNSAFE = ((torch.Tag.cudagraph_unsafe,)
              if os.environ.get("PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE") == "1"
              else ())


@torch.library.custom_op("prismaquant::cb_gemv_fp8", mutates_args=(), tags=_PQ_UNSAFE)
def cb_gemv_fp8(x: torch.Tensor, qw_padded: torch.Tensor,
                cb_flat: torch.Tensor, cb_row_offset: torch.Tensor,
                scale: torch.Tensor, N: int, K: int, k_bits: int, n_sub: int,
                type_size: int) -> torch.Tensor:
    """CUDA decode-GEMV for FP8_CB (prototype ii): takes RAW bf16 activations
    and fuses the per-token fp8 dynamic QDQ + the bandwidth-bound dequant-GEMV
    into one op (two kernel launches vs the Triton path's ~7). Caller must
    fails closed when the native extension is unavailable."""
    from .cuda_ext import require_ext
    return require_ext("FP8-CB decode GEMV").cb_gemv_fp8(
        x, qw_padded, cb_flat, cb_row_offset, scale,
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
    the native extension is unavailable."""
    from .cuda_ext import require_ext
    return require_ext("FP4-CB v2 decode GEMV").cb_gemv_fp4_v2(
        xq, qw_padded, cb_flat, cb_row_offset, compose,
        N, K, k_bits, n_sub, type_size)


@cb_gemv_fp4_v2.register_fake
def _cb_gemv_fp4_v2_fake(xq, qw_padded, cb_flat, cb_row_offset, compose, N, K,
                         k_bits, n_sub, type_size):
    return torch.empty((*xq.shape[:-1], N), dtype=xq.dtype, device=xq.device)


@torch.library.custom_op("prismaquant::cb_expand_fp8", mutates_args=(), tags=_PQ_UNSAFE)
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
                         mutates_args=(), tags=_PQ_UNSAFE)
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
                         mutates_args=(), tags=_PQ_UNSAFE)
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
                         mutates_args=("out",), tags=_PQ_UNSAFE)
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


@torch.library.custom_op("prismaquant::cb_expand_fp8_into",
                         mutates_args=("out",), tags=_PQ_UNSAFE)
def cb_expand_fp8_into(out: torch.Tensor, qw_padded: torch.Tensor,
                       cb_flat_fp8: torch.Tensor, cb_row_offset: torch.Tensor,
                       N: int, K: int, k_bits: int, n_sub: int,
                       type_size: int) -> None:
    """``cb_expand_fp8`` writing into a CALLER-OWNED buffer.

    The allocating form is unusable for the L2 pipeline: the persisting-access
    window pins a fixed ADDRESS RANGE, so a fresh allocation per expert would
    land outside the pinned range and the lever would silently do nothing. ``out``
    may be LARGER than ``N*K`` (it is one half of a rotating arena sized for the
    largest expert group), and only the first ``N*K`` bytes are written.

    ``mutates_args=("out",)`` is load-bearing: an empty annotation would let
    compile/capture reorder or elide the write against the GEMM that reads it —
    a silent wrong answer, not a crash. Missing native code is fatal.
    """
    from .cuda_ext import require_ext
    require_ext("FP8-CB in-place transient expansion").cb_expand_fp8_into(
        out, qw_padded, cb_flat_fp8, cb_row_offset,
        N, K, k_bits, n_sub, type_size)


@cb_expand_fp8_into.register_fake
def _cb_expand_fp8_into_fake(out, qw_padded, cb_flat_fp8, cb_row_offset, N, K,
                             k_bits, n_sub, type_size):
    return None


def cb_expand_fp8_into_available() -> bool:
    """Report whether this extension build exposes the research utility."""
    from .cuda_ext import get_ext
    ext = get_ext()
    return ext is not None and hasattr(ext, "cb_expand_fp8_into")


@torch.library.custom_op("prismaquant::fp8_act_qdq", mutates_args=(), tags=_PQ_UNSAFE)
def fp8_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)
    as a custom op for the compile path."""
    from .cuda_ext import require_ext
    return require_ext("FP8 activation QDQ").fp8_act_qdq(x)


@fp8_act_qdq.register_fake
def _fp8_act_qdq_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("prismaquant::fp4_act_qdq", mutates_args=(), tags=_PQ_UNSAFE)
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


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp4_v2", mutates_args=(), tags=_PQ_UNSAFE)
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
# It MUST carry the same ``_PQ_UNSAFE`` tagging as every op above — see the
# module header: tagging these ops ``cudagraph_unsafe`` under
# ``use_inductor_graph_partition=True`` + PIECEWISE cudagraphs makes each a
# graph-PARTITION BOUNDARY and this build mishandles the hand-off, giving
# DETERMINISTIC output corruption. An op that disagreed with its neighbours on
# this tag would partition the decode path in exactly the wrong place.
#
# Separate JIT module (``get_ext_v2`` -> ``prismaquant_cb_v2_ext``), not a
# second source of the inherited one: both .cu files define
# ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)`` and would collide at link.
@torch.library.custom_op("prismaquant::cb_moe_gemv_v2", mutates_args=(), tags=_PQ_UNSAFE)
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


@torch.library.custom_op("prismaquant::cb_moe_gemv_fp8", mutates_args=(), tags=_PQ_UNSAFE)
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


@torch.library.custom_op("prismaquant::cb_moe_combine", mutates_args=(), tags=_PQ_UNSAFE)
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


@torch.library.custom_op("prismaquant::cb_moe_forward", mutates_args=())
def cb_moe_forward(x: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
    method, layer = _lookup_cb_layer(layer_id)
    return method._apply_inline(layer, x, topk_weights, topk_ids)


@cb_moe_forward.register_fake
def _cb_moe_forward_fake(x, topk_weights, topk_ids, layer_id):
    return torch.empty_like(x)
