"""Custom-op registration for the CB decode-GEMM.

Registered through ``torch.library.custom_op`` with a ``register_fake`` so it is
opaque to Dynamo and safe under ``torch.compile`` / CUDA-graph capture (the GGUF
`ops.py` pattern: a fake impl that returns a correctly-shaped empty tensor). No
vLLM import here, so the op is usable from the standalone correctness tests too.
"""
from __future__ import annotations

import itertools
import os
import sys
import weakref

import torch

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
    a silent wrong answer, not a crash. Caller must check ``cuda_ext.get_ext()``.
    """
    from .cuda_ext import get_ext
    get_ext().cb_expand_fp8_into(out, qw_padded, cb_flat_fp8, cb_row_offset,
                                 N, K, k_bits, n_sub, type_size)


@cb_expand_fp8_into.register_fake
def _cb_expand_fp8_into_fake(out, qw_padded, cb_flat_fp8, cb_row_offset, N, K,
                             k_bits, n_sub, type_size):
    return None


def cb_expand_fp8_into_available() -> bool:
    """Whether THIS extension build ships the into-buffer expander. An older
    build must degrade to the stock path, never crash a serve."""
    from .cuda_ext import get_ext
    ext = get_ext()
    return ext is not None and hasattr(ext, "cb_expand_fp8_into")


@torch.library.custom_op("prismaquant::cb_prefill_persistent_tc",
                         mutates_args=(), tags=_PQ_UNSAFE)
def cb_prefill_persistent_tc(a_fp8: torch.Tensor, packed_u8: torch.Tensor,
                             cb_flat_fp8_u8: torch.Tensor, N: int, K: int,
                             k_bits: int, type_size: int,
                             variant: int = 1) -> torch.Tensor:
    """Persistent-N tensor-core FP8_CB prefill (#4b): [M,K] e4m3 activations x
    packed CB rows -> UNSCALED bf16 [M, N]. The caller MUST apply the
    per-token activation scale and the per-output-channel weight scale.
    Quarantined behind ``PRISMAQUANT_ENABLE_PTC=1`` (cuda_ext)."""
    from .cuda_ext import get_persistent_ext
    ext = get_persistent_ext()
    if ext is None:
        raise RuntimeError(
            "persistent-TC ext not enabled (PRISMAQUANT_ENABLE_PTC=1)")
    return ext.cb_prefill_persistent_tc(a_fp8, packed_u8, cb_flat_fp8_u8,
                                        N, K, k_bits, type_size, variant)


@cb_prefill_persistent_tc.register_fake
def _cb_prefill_persistent_tc_fake(a_fp8, packed_u8, cb_flat_fp8_u8, N, K,
                                   k_bits, type_size, variant=1):
    return torch.empty((a_fp8.shape[0], N), dtype=torch.bfloat16,
                       device=a_fp8.device)


@torch.library.custom_op("prismaquant::fp8_act_qdq", mutates_args=(), tags=_PQ_UNSAFE)
def fp8_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)
    as a custom op for the compile path."""
    from .cuda_ext import get_ext
    return get_ext().fp8_act_qdq(x)


@fp8_act_qdq.register_fake
def _fp8_act_qdq_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("prismaquant::fp4_act_qdq", mutates_args=(), tags=_PQ_UNSAFE)
def fp4_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Fused group-16 fp4 (E2M1) activation QDQ (bit-exact to
    codec.fp4_group16_act_qdq) as a custom op for the compile path."""
    from .cuda_ext import get_ext
    return get_ext().fp4_act_qdq(x)


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


# One-shot resolution of the fp4 act-QDQ implementation.
#
# The fp8 CB lane has always fused its activation QDQ into ONE CUDA launch
# (fp8_act_qdq above); the fp4 lane ran codec.fp4_group16_act_qdq, roughly two
# dozen eager torch dispatches plus the matching allocator round-trips per
# module call. The grouped MoE decode path pays that twice per layer per token
# (moe.py, the module input and the intermediate), which is milliseconds of pure
# host dispatch on a many-layer MoE.
#
# The fallback is bit-identical, only slow, so correctness never depends on the
# ext being present. But a SILENT fallback would make a decode benchmark measure
# the wrong throughput class while looking valid. So the resolution prints its
# verdict once, loudly, on BOTH branches -- the same discipline as
# cb_expand_fp8_into_available()'s degrade-never-crash contract, with the
# addition that the degraded branch says so out loud.
_FP4_ACT_QDQ_OK = None


def fp4_act_qdq_ok() -> bool:
    """True when the CUDA fp4 act-QDQ is available (resolved once, cached)."""
    global _FP4_ACT_QDQ_OK
    if _FP4_ACT_QDQ_OK is None:
        from .cuda_ext import get_ext
        ext = get_ext()
        _FP4_ACT_QDQ_OK = ext is not None and hasattr(ext, "fp4_act_qdq")
        if _FP4_ACT_QDQ_OK:
            print("[prismaquant-cb] act-qdq fp4=cuda "
                  "(prismaquant::fp4_act_qdq, 1 launch/call)", flush=True)
        else:
            print("[prismaquant-cb] WARNING: act-qdq fp4=EAGER-CODEC -- the "
                  "CUDA fp4_act_qdq symbol is missing from the ext, so every "
                  "fp4-CB projection pays ~two dozen torch dispatches instead "
                  "of one kernel launch. Numerics are unchanged; DECODE "
                  "THROUGHPUT IS NOT REPRESENTATIVE -- do not bench.",
                  file=sys.stderr, flush=True)
    return _FP4_ACT_QDQ_OK


def fp4_act_qdq_or_codec(x: torch.Tensor) -> torch.Tensor:
    """fp4 group-16 activation QDQ: CUDA op when available, eager codec else.

    Bit-identical either way (tests/test_fp4_act_qdq.py asserts torch.equal,
    never a tolerance) -- linear.py and moe.py both document that this QDQ runs
    OUTSIDE the kernel precisely so CUDA-vs-Triton numerics stay aligned, so a
    tolerance here would silently break that contract.
    """
    if x.is_cuda and x.dtype is torch.bfloat16 and fp4_act_qdq_ok():
        return fp4_act_qdq(x)
    from . import codec
    return codec.fp4_group16_act_qdq(x).to(torch.bfloat16)


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
    from .cuda_ext import get_ext_v2
    return get_ext_v2().cb_gemv_v2(xq, qw, cb_flat, compose, pair_expert,
                                   pair_xrow, k_bits, type_size, rpb,
                                   dict_mode)


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
_LAYER_REGISTRY: dict[int, tuple[weakref.ReferenceType,
                                 weakref.ReferenceType]] = {}
_LAYER_IDS = itertools.count()
_DISPATCH_VIA_OP: list = []


def dispatch_via_op() -> bool:
    if not _DISPATCH_VIA_OP:
        _DISPATCH_VIA_OP.append(
            os.environ.get("PRISMAQUANT_CB_DISPATCH", "op") != "inline")
    return _DISPATCH_VIA_OP[0]


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
