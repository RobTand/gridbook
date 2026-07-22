"""``PrismaQuantCBLinearMethod`` — weight loading + apply for CB Linears.

Load-time (LAYOUT.md §3): a byte-shaped uint8 ``cb_qweight`` per Linear, an
fp8-only per-output-channel ``weight_scale``, and the model-level shared
``cb_codebook.*`` sidecars (loaded once by the config). Apply: emulate the
served W4A4/W8A8 activation bucket, then the Triton decode-GEMM custom op.

Fused vLLM modules (qkv_proj, gate_up_proj) hold several roles' output rows in
one weight; per-role shared codebooks are concatenated and addressed by a
per-output-row offset (``cb_row_offset``) so fusion stays correct.
"""
from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)

from . import codec
from .expand import expand_cb_to_fp8, expand_fp4_v2_to_weight
from .ops import cb_gemm, cb_gemv_fp4_v2, cb_gemv_fp8, dispatch_via_op

# Fallback fused mapping if the config's packed_modules_mapping is unset.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# M-gate for the CB dispatch (GGUF's mmvq_safe pattern, quantization/linear.py
# :34-57): M<=threshold is the decode regime -> keep the bf16-MMA Triton
# decode-GEMM; M>threshold is prefill -> transiently expand FP8_CB to a native
# fp8 tile and hit vLLM's stock W8A8 fp8 GEMM (native tensor cores). NVFP4_CB
# stays on the Triton path either way (transient FP4 needs FP4-MMA, out of
# scope). 16 mirrors the decode/prefill split the decode kernel already tiles at.
# Env-overridable so the prefill A/B (old Triton path vs transient native GEMM)
# is a serve-flag toggle: set PRISMAQUANT_PREFILL_M_THRESHOLD huge to force the
# Triton decode path at prefill (isolates the transient-expansion lever).
PREFILL_M_THRESHOLD = int(os.environ.get("PRISMAQUANT_PREFILL_M_THRESHOLD", "16"))

# Within the decode regime, the CUDA GEMV handles M<=this and the Triton
# decode-GEMM the rest. 8 is measured on GB10 (bench_cuda_gemv.py): the
# weight-stationary GEMV re-reads x per row-block, so it wins 3.2x at M=1-2,
# 2.7x at M=4, 1.2x at M=8, and LOSES (0.66x) at M=16 where Triton's tl.dot
# amortizes x across the row tile.
CUDA_GEMV_M_MAX = int(os.environ.get("PRISMAQUANT_CB_CUDA_M_MAX", "8"))

# NOTE on a rejected variant (2026-07-18): N-chunking the transient expand +
# GEMM with side-stream overlap measured 0.46x (small-N GEMMs lose more than
# the overlap hides) AND is not bit-exact — cutlass_scaled_mm picks different
# configs per shape (split-K on narrow N), which breaks the served-KL
# contract. Prefill overlap needs cross-layer prefetch or the fused
# decode-in-prologue kernel, not N-chunking.


@register_weight_loader_v2_supported_method
class PrismaQuantCBLinearMethod(LinearMethodBase):
    def __init__(self, quant_config, scheme: dict, prefix: str) -> None:
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        # Two-tier v2 scale coding (fp4 only) — absence of scale_coding ⇒ v1.
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        elif isinstance(sc, str):
            self.is_v2 = sc == codec.SCALE_CODING_TWO_TIER
            self._sub_table = codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = False
            self._sub_table = None
        if self.is_v2 and not self.is_fp4:
            raise ValueError(f"{prefix}: two-tier scale coding is fp4-only")

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        K = input_size_per_partition
        if K % codec.SUPERBLOCK != 0:
            raise ValueError(f"{self.prefix}: in_features {K} not a multiple of "
                             f"{codec.SUPERBLOCK}")
        rows = sum(output_partition_sizes)
        row_bytes = (K // codec.SUPERBLOCK) * self.type_size
        layer.logical_widths = list(output_partition_sizes)
        layer._cb_input_size = K

        cb_qweight = ModelWeightParameter(
            data=torch.empty(rows, row_bytes, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("cb_qweight", cb_qweight)

        if not self.is_fp4:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(rows, dtype=torch.float32),
                output_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight_scale", weight_scale)

    # -- shard-role resolution for a (possibly fused) layer -----------------
    def _shard_roles(self):
        leaf = self.prefix.split(".")[-1]
        pmm = getattr(self.quant_config, "packed_modules_mapping", {}) or {}
        shard_leaves = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf) or [leaf]
        prefixes = [self.prefix[: -len(leaf)] + sl for sl in shard_leaves]
        # Keep only shards that are actual CB targets (all, for uniform arts).
        return [p for p in prefixes if p in self.quant_config.target_scheme]

    def _ckpt_cb_rows(self) -> dict[str, int]:
        """Row count of every ``*.cb_qweight`` tensor in the on-disk checkpoint
        (header-only read, cached once on the shared quant_config). Used to
        recover the per-ROLE row split of a vLLM-merged linear whose roles have
        separate codebooks (e.g. GDN ``in_proj_qkvz`` = ``in_proj_qkv`` +
        ``in_proj_z``); ``logical_widths`` does not expose that boundary."""
        qc = self.quant_config
        cache = getattr(qc, "_ckpt_cb_row_cache", None)
        if cache is not None:
            return cache
        import glob
        import json
        import struct
        from vllm.config import get_current_vllm_config
        model_dir = get_current_vllm_config().model_config.model
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        cache = {}
        for st in files:
            with open(st, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name != "__metadata__" and name.endswith(".cb_qweight"):
                    cache[name] = int(meta["shape"][0])
        qc._ckpt_cb_row_cache = cache
        return cache

    def _lookup_ckpt_rows(self, shard_prefix: str, ckpt_rows: dict) -> int:
        """Rows of ``<shard_prefix>.cb_qweight`` in the checkpoint. shard_prefix
        is a vLLM-mapped name; the on-disk tensor keeps the export name, which
        differs only by the model-nesting prefix (Qwen3-VL: ``model.language_
        model.`` vs ``language_model.model.``). Match on the module-local tail
        (from ``.layers.`` on, which is nesting-invariant), requiring a unique
        hit so a mis-mapped name fails loudly rather than silently truncating."""
        want = shard_prefix + ".cb_qweight"
        if want in ckpt_rows:
            return ckpt_rows[want]
        cut = shard_prefix.rfind(".layers.")
        tail = (shard_prefix[cut:] if cut >= 0
                else "." + shard_prefix.split(".")[-1]) + ".cb_qweight"
        hits = [r for name, r in ckpt_rows.items() if name.endswith(tail)]
        assert len(hits) == 1, (
            f"{shard_prefix}: expected exactly one checkpoint cb_qweight ending "
            f"'{tail}', found {len(hits)} — cannot resolve merged-role row split")
        return hits[0]

    def process_weights_after_loading(self, layer):
        dev = layer.cb_qweight.device
        codebooks = self.quant_config.get_codebooks()

        # Build the concatenated flat codebook + per-row base offset. The
        # per-row offset MUST cover every output row (its length == N), else the
        # decode/expand kernels read cb_row_offset out of bounds.
        shard_prefixes = self._shard_roles()
        widths = list(layer.logical_widths)
        if len(shard_prefixes) == len(widths):
            # One CB role per logical shard: a genuinely fused module (qkv_proj /
            # gate_up_proj) with a per-role codebook, or a plain single Linear.
            # logical_widths is the authoritative per-shard row count.
            shard_widths = widths
        else:
            # vLLM merged MORE logical shards than we have CB roles: e.g. a
            # Gated-DeltaNet ``in_proj_qkvz`` (logical_widths=[q,k,v,z], 4 chunks)
            # that the export packed as TWO targets ``in_proj_qkv``(=q+k+v) +
            # ``in_proj_z``(=z), each with its own codebook. logical_widths does
            # NOT expose the per-ROLE boundary, so derive each role's row count
            # from the checkpoint's separate ``<role>.cb_qweight`` tensor (vLLM
            # merges them on load, but they are distinct on disk). The old code
            # used widths[0] for the single role -> cb_row_offset was short ->
            # illegal memory access in the decode/expand kernels.
            ckpt_rows = self._ckpt_cb_rows()
            shard_widths = [self._lookup_ckpt_rows(sp, ckpt_rows)
                            for sp in shard_prefixes]
            assert sum(shard_widths) == sum(widths), (
                f"{self.prefix}: checkpoint role rows {shard_widths} "
                f"(sum {sum(shard_widths)}) != logical width sum {sum(widths)}")
        blocks, row_offsets = [], []
        cb_cumoffset = 0
        for i, sp in enumerate(shard_prefixes):
            ref = self.quant_config.target_scheme[sp]["codebook_ref"]
            names = ref if isinstance(ref, list) else [ref]
            subs = [codebooks[n].to(dev) for n in names]
            flat = codec.build_flat_codebook(subs)
            blocks.append(flat)
            w = shard_widths[i]
            # cumulative base (not i*cb_total): correct even if per-shard
            # codebooks differ in size. All rows of shard i point at that
            # shard's block within the concatenated cb_flat.
            row_offsets.append(torch.full((w,), cb_cumoffset,
                                          dtype=torch.int32, device=dev))
            cb_cumoffset += flat.numel()
        cb_flat = torch.cat(blocks).contiguous()
        cb_row_offset = torch.cat(row_offsets).contiguous()

        qw = layer.cb_qweight.data
        assert cb_row_offset.numel() == qw.shape[0], (
            f"{self.prefix}: cb_row_offset has {cb_row_offset.numel()} rows but "
            f"the packed weight has {qw.shape[0]} — per-row offset must cover "
            "every output row (kernels index it by row).")
        layer._cb_qw_padded = codec.pad_qweight(qw)
        layer._cb_flat = cb_flat
        layer._cb_row_offset = cb_row_offset
        dummy = torch.zeros(1, dtype=torch.float32, device=dev)
        if self.is_fp4 and self.is_v2:
            # v2: NO resident fp32 plane (spec §4/G4). The kernel composes the
            # E4M3 scales in-register from the packed 9 bytes via this (256,16)
            # table; the 9-byte plane stays inside cb_qweight.
            layer._cb_compose = codec.build_compose_table(self._sub_table).to(dev)
            layer._cb_scale = dummy
            # Warm the CUDA-GEMV JIT build at LOAD time — otherwise the ~30 s
            # in-container build fires on the first decode step and poisons the
            # first request's latency (same reason as the fp8 branch below).
            self._cuda_gemv_ok()
        elif self.is_fp4:
            layer._cb_scale = codec.decode_fp4_scale_plane(qw, self.k).to(dev)
            layer._cb_compose = dummy
        else:
            layer._cb_scale = layer.weight_scale.data.reshape(-1).to(
                torch.float32)
            layer._cb_compose = dummy
            # E4M3-byte codebook for the fp8-direct transient expand and the
            # CUDA GEMV (exact: every codebook value is on the e4m3 grid, so
            # bf16 -> fp8 is a lossless re-encoding of the same table).
            layer._cb_flat_fp8 = cb_flat.to(torch.float8_e4m3fn).view(
                torch.uint8).contiguous()
            # Warm the CUDA-GEMV JIT build at LOAD time — otherwise the ~30 s
            # in-container build fires on the first decode step and poisons
            # the first request's latency (seen live: 1.89 tok/s rep 1).
            self._cuda_gemv_ok()
        layer._cb_N = qw.shape[0]
        layer._cb_K = layer._cb_input_size
        from .ops import register_cb_layer
        layer._cb_layer_id = register_cb_layer(self, layer)

    def _cuda_gemv_ok(self) -> bool:
        """CUDA decode-GEMV eligibility (fp8 n_sub=4 rungs, or fp4 two-tier v2
        n_sub=2 rungs; env-gated; ext built). Cached — the answer never changes
        within a process."""
        ok = getattr(self, "_cuda_gemv_cached", None)
        if ok is None:
            gate = os.environ.get("PRISMAQUANT_CB_DECODE", "cuda") == "cuda"
            fp8_ok = not self.is_fp4 and self.n_sub == 4
            fp4v2_ok = self.is_fp4 and self.is_v2 and self.n_sub == 2
            ok = gate and (fp8_ok or fp4v2_ok)
            if ok:
                from .cuda_ext import get_ext
                ok = get_ext() is not None
            self._cuda_gemv_cached = ok
        return ok

    def apply(self, layer, x, bias=None):
        # M-branch hoist (compile lane): dispatch through ONE opaque custom op
        # so torch.compile never traces the M-branch (which otherwise bakes
        # the prefill expand path into the decode graph — see ops.py). Eager
        # serving behavior is identical: the op impl calls _apply_inline.
        # PRISMAQUANT_CB_DISPATCH=inline restores in-graph branching (A/B).
        # bias falls back to inline (no served model carries one; the cutlass
        # path fuses bias and we keep that numerics contract untouched).
        lid = getattr(layer, "_cb_layer_id", None)
        if bias is None and lid is not None and dispatch_via_op():
            from .ops import cb_linear_forward
            return cb_linear_forward(x, lid)
        return self._apply_inline(layer, x, bias)

    def _apply_inline(self, layer, x, bias=None):
        N, K = layer._cb_N, layer._cb_K
        M = x.reshape(-1, K).shape[0]
        # Decode regime (M small), plus fp4-v1 which has no transient path yet
        # (its v1 e4m3 plane is not composed during expansion) — Triton decode.
        if M <= PREFILL_M_THRESHOLD or (self.is_fp4 and not self.is_v2):
            if M <= CUDA_GEMV_M_MAX and self._cuda_gemv_ok():
                if self.is_fp4:
                    # fp4-v2 CUDA GEMV: act-QDQ (fp4 group-16 RTN) runs OUTSIDE
                    # the kernel via codec — exactly as the Triton fp4 path — so
                    # CUDA-vs-Triton numerics stay aligned. The kernel gathers
                    # the bf16 codebook and composes the two-tier scale
                    # in-register from the packed 9-byte plane.
                    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
                    y = cb_gemv_fp4_v2(xq, layer._cb_qw_padded, layer._cb_flat,
                                       layer._cb_row_offset, layer._cb_compose,
                                       N, K, self.k, self.n_sub, self.type_size)
                else:
                    # CUDA bandwidth-bound GEMV, act-QDQ fused (raw x in);
                    # gathers the E4M3-byte codebook (same values as _cb_flat,
                    # 4x smaller).
                    y = cb_gemv_fp8(x, layer._cb_qw_padded, layer._cb_flat_fp8,
                                    layer._cb_row_offset, layer._cb_scale,
                                    N, K, self.k, self.n_sub, self.type_size)
                if bias is not None:
                    y = y + bias
                return y
            xq = (codec.fp4_group16_act_qdq(x) if self.is_fp4
                  else codec.fp8_dynamic_act_qdq(x))
            y = cb_gemm(xq, layer._cb_qw_padded, layer._cb_flat,
                        layer._cb_row_offset, layer._cb_scale,
                        layer._cb_compose, N, K, self.k, self.n_sub,
                        self.type_size, self.is_fp4, self.is_v2)
            if bias is not None:
                y = y + bias
            return y

        if self.is_fp4:
            # fp4 v2 prefill: transiently expand to a bf16 weight (value ×
            # composed E4M3 v2 scale) and run one cuBLAS GEMM, amortising the
            # decode over M — the fp4 counterpart of the fp8 transient. INV-1:
            # the [N,K] tile is bounded to one layer, freed per forward. (bf16
            # MMA — INV-2 waived; the FP4-MMA CUTLASS prefill is prototype iii.)
            import torch.nn.functional as F
            xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
            W = expand_fp4_v2_to_weight(
                layer._cb_qw_padded, layer._cb_flat, layer._cb_row_offset,
                layer._cb_compose, N, K, self.k, self.n_sub, self.type_size)
            y = F.linear(xq, W)
            del W
            if bias is not None:
                y = y + bias
            return y

        # FP8_CB prefill (M large): transiently expand THIS layer's packed
        # weight into a native fp8 tile and call vLLM's stock per-channel W8A8
        # fp8 GEMM (native tensor cores), then free the tile. An expanded
        # FP8_CB weight IS a standard per-channel fp8 checkpoint (codebook
        # values on the e4m3 grid; layer.weight_scale per output channel).
        # The expand writes the fp8 bytes directly (no bf16 intermediate, no
        # cast pass) — byte-identical to the old bf16-expand + cast, at a third
        # of the expand-side HBM traffic.
        # INV-1: the [N,K] tile is bounded to one layer (expand -> GEMM ->
        # free), never resident/model-wide (the NVINT2 OOM trap). `ops` is
        # imported lazily so the module still imports without vLLM (venv tests).
        import vllm._custom_ops as ops
        x2 = x.reshape(-1, K)
        xq, sa = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        # scale_b is the per-output-channel weight scale as [N, 1] (matches
        # vLLM's stock per-channel fp8 scheme; verified against a fp32 dequant
        # reference in tests/test_transient_fp8.py::test_transient_gemm_*).
        ws = layer._cb_scale.reshape(N, 1)

        if self._cuda_gemv_ok():
            # CUDA expander (stream-bandwidth-bound; the Triton byte-gather
            # ran at 61-86 GB/s and serialized ~half the prefill).
            from .ops import cb_expand_fp8
            W_e4m3 = cb_expand_fp8(
                layer._cb_qw_padded, layer._cb_flat_fp8,
                layer._cb_row_offset, N, K, self.k, self.n_sub,
                self.type_size)
        else:
            W_e4m3 = expand_cb_to_fp8(
                layer._cb_qw_padded, layer._cb_flat_fp8, layer._cb_row_offset,
                N, K, self.k, self.n_sub, self.type_size)  # [N,K] e4m3
        out = ops.cutlass_scaled_mm(xq, W_e4m3.t(), sa, ws, torch.bfloat16, bias)
        del W_e4m3
        return out.reshape(*x.shape[:-1], N)
