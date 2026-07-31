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
from .ops import (cb_gemm, cb_gemv_fp4_v2, cb_gemv_fp8, dispatch_via_op,
                  fp4_act_qdq_or_codec)

# NOTE: the fused-sibling fallback map (qkv_proj, gate_up_proj, in_proj_qkvz,
# in_proj_ba) used to be duplicated here. It — and the namespace handling around
# it — now live once, in ``config`` (``_FUSED_FALLBACK`` +
# ``PrismaQuantConfig.shard_target_keys``), because this module's copy and the
# config's copy answered "which shards does this fused module have?" separately
# and drifted (issue #1).

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

# fp4-MMA fused prefill gate (OPT-IN, default OFF): "" = off (the shipping
# fp4 dispatch is byte-identical), "1" = fused block-scaled prefill for all
# M > PREFILL_M_THRESHOLD, "midm" = only 16 < M <= 128 (the fp8 kernel's
# measured niche). Promotion to a default requires a served A/B per
# docs/lanes/nvfp4-cb/STANDARDS.md — the activation bucket changes.
_FP4_FUSED_MODE: list = []


def _fp4_fused_mode() -> str:
    # Explicit opt-in: what this changes for a served artifact is the fp4
    # activation bucket moving
    # from the Triton path's fp32 emulation scales to the format's native
    # ue4m3 scale factors — measured ~7.5e-2 relative against Triton. The fused
    # kernel is bit-exact against the stock NVF4 collective, so this is
    # arguably the *more* faithful rendering of what NVFP4 hardware serving
    # does; but it is a change in served numerics that has NOT been validated
    # on the serving metric (no KL/PPL A/B). See
    # docs/KERNELS.md ("Fused decode-in-prologue").
    if not _FP4_FUSED_MODE:
        _FP4_FUSED_MODE.append(
            os.environ.get("PRISMAQUANT_CB_FUSED_FP4", "").strip())
    return _FP4_FUSED_MODE[0]


# Within the decode regime, the CUDA GEMV handles M<=this and the Triton
# decode-GEMM the rest. 8 is measured on GB10 (bench_cuda_gemv.py): the
# weight-stationary GEMV re-reads x per row-block, so it wins 3.2x at M=1-2,
# 2.7x at M=4, 1.2x at M=8, and LOSES (0.66x) at M=16 where Triton's tl.dot
# amortizes x across the row tile.
CUDA_GEMV_M_MAX = int(os.environ.get("PRISMAQUANT_CB_CUDA_M_MAX", "8"))

# ROCm/HIP kernels (gridbook/csrc_hip). Imported only when torch is a ROCm
# build, so a CUDA install neither imports nor builds anything HIP; `_HIP` stays
# None and `_apply_inline` keeps its previous behaviour exactly.
#
# RELEASE HOLD, and why this file is still synced verbatim. The HIP lane is
# held out of the public gridbook project until it has a serving metric
# (scripts/sync_gridbook.py HELD_PATHS: csrc_hip/, hip_ext.py, linear_hip.py),
# so in the RELEASE tree `linear_hip` does not exist and this import raises
# ImportError -> the except arm below sets `_HIP = None` -> the dispatch is
# byte-identical to the pre-ROCm one. That degradation is what lets the hold be
# on PATHS only. Holding these lines too would fork linear.py -- the package's
# most-edited file, the dispatch core -- between the two trees, and the drift
# gate would then have to carry a *content* exception it cannot check the way
# it checks a path exception. Path hold: policy. Content fork: band-aid.
try:                                                # pragma: no cover - env
    if getattr(torch.version, "hip", None):
        from . import linear_hip as _HIP
    else:
        _HIP = None
except Exception:                                   # pragma: no cover - env
    _HIP = None

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
        """The ``target_scheme`` keys of this (possibly fused) module's CB
        roles, in shard order — the per-role codebooks
        ``process_weights_after_loading`` concatenates.

        Delegated to the config so this and ``_scheme_for_prefix`` can never
        disagree about which shards a fused module has, or about which
        namespace vintage the keys are in. ``unfused_fallback=True`` keeps this
        call site's semantics: a plain Linear is its own single role (the
        scheme lookup deliberately has no such rung). Serving prefixes may
        carry a class wrapper (``language_model.model.*``) OR the mapper's own
        namespace while the stored keys sit in another — both the 0.8B hybrid
        (resolved ZERO roles, died on the width assert) and issue #1 were that
        one question answered locally and wrongly."""
        return self.quant_config.shard_target_keys(self.prefix,
                                                   unfused_fallback=True)

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
        if not os.path.isdir(model_dir):
            # Hub repo id: vLLM already downloaded the weights; reuse its
            # snapshot dir instead of a raw path join (serve-by-id fix).
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(model_dir,
                                          allow_patterns=["*.safetensors"])
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
        # Fused projections commonly reuse the exact same shared codebook for
        # every role.  Keep one physical block per exact reference tuple and
        # point all matching roles at it.  Comparing references (rather than
        # tensor values) is deliberate: differently named codebooks remain
        # distinct even when their current contents happen to match.
        block_offsets: dict[tuple[str, ...], int] = {}
        cb_cumoffset = 0
        for i, sp in enumerate(shard_prefixes):
            ref = self.quant_config.target_scheme[sp]["codebook_ref"]
            names = tuple(ref) if isinstance(ref, (list, tuple)) else (ref,)
            if names not in block_offsets:
                subs = [codebooks[n].to(dev) for n in names]
                flat = codec.build_flat_codebook(
                    subs, self.prefix, "fp4" if self.is_fp4 else "fp8")
                block_offsets[names] = cb_cumoffset
                blocks.append(flat)
                cb_cumoffset += flat.numel()
            w = shard_widths[i]
            # Exact-reference duplicates reuse their first block; distinct
            # references retain a cumulative base (rather than i*cb_total),
            # which remains correct when blocks differ in size.
            row_offsets.append(torch.full((w,), block_offsets[names],
                                          dtype=torch.int32, device=dev))
        cb_flat = torch.cat(blocks).contiguous()
        cb_row_offset = torch.cat(row_offsets).contiguous()

        qw = layer.cb_qweight.data
        rows, row_bytes = int(qw.shape[0]), int(qw.shape[1])
        assert cb_row_offset.numel() == rows, (
            f"{self.prefix}: cb_row_offset has {cb_row_offset.numel()} rows but "
            f"the packed weight has {rows} — per-row offset must cover "
            "every output row (kernels index it by row).")
        layer._cb_flat = cb_flat
        layer._cb_row_offset = cb_row_offset
        # The fp8 mid-M fused kernel has no row-offset input: every row uses
        # LUT base zero.  Derive and cache its safety fact without a CUDA->host
        # sync.  One interned block means every role necessarily points at the
        # first (zero-based) block; two or more blocks must use the offset-aware
        # fallback paths.
        layer._cb_fp8_fused_lut_ok = len(blocks) == 1
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
            layer._cb_flat_fp8 = codec.flat_codebook_fp8(
                cb_flat, self.prefix)
            # Warm the CUDA-GEMV JIT build at LOAD time — otherwise the ~30 s
            # in-container build fires on the first decode step and poisons
            # the first request's latency (seen live: 1.89 tok/s rep 1).
            self._cuda_gemv_ok()

        # ONE resident copy of the packed weight (issue #1, 2026-07-25).
        #
        # ``pad_qweight`` allocates a padded COPY, and the registered
        # ``cb_qweight`` parameter used to stay live alongside it for the whole
        # serve — every dense CB weight was resident TWICE. Measured on the
        # shipped Qwen3.6-27B gridbook artifact (safetensors header): 15.07 GiB
        # of dense ``cb_qweight`` out of 21.38 GiB total, i.e. ~36.5 GiB of
        # weights served instead of ~21.4 GiB. It went unnoticed because the
        # reference box is a 128 GB unified-memory DGX Spark, where the slack
        # simply absorbed it; it only bites on 24/32 GB cards (reported from a
        # 32 GB RTX 5090). MoE stacks were never affected — ``moe.py`` pads only
        # bounded per-forward transients, never the resident expert stack.
        #
        # The fix is a narrow VIEW, not a ``del``: ``cb_qweight.data`` is still
        # read by the fp8 mid-M fused prefill entry and the persistent-TC path,
        # both of which take the row stride explicitly and only require
        # ``stride(1) == 1`` and a 16-byte-multiple ``stride(0)`` — which the
        # 16-byte pad preserves (see ``codec.pad_qweight``). Dropping the local
        # ``qw`` then releases the original storage.
        layer._cb_qw_padded = codec.pad_qweight(qw)
        layer.cb_qweight.data = layer._cb_qw_padded.narrow(1, 0, row_bytes)
        del qw
        layer._cb_N = rows
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
            # n_sub==2 product, n_sub==1 signed (S-rungs) — both CUDA-served
            fp4v2_ok = self.is_fp4 and self.is_v2 and self.n_sub in (1, 2)
            ok = gate and (fp8_ok or fp4v2_ok)
            if ok:
                from .cuda_ext import get_ext
                ok = get_ext() is not None
            self._cuda_gemv_cached = ok
        return ok

    def _ptc_ok(self, layer, K: int) -> bool:
        """Persistent-TC prefill eligibility for THIS layer (cached; the answer
        never changes after load). Mirrors the kernel's own TORCH_CHECKs so a
        miss is a silent fall-through, never a crash."""
        ok = getattr(layer, "_cb_ptc_ok", None)
        if ok is not None:
            return ok
        ok = (not self.is_fp4                      # fp8-CB only
              and self.n_sub == 4                  # kernel LUT = 4 sub-tables
              and K % 256 == 0 and K % 64 == 0
              and self.k % 4 == 0
              and self.type_size == 4 * self.k and self.type_size <= 192
              and 16 * K + 16384 <= 99 * 1024      # smem plan
              and getattr(layer, "_cb_flat_fp8", None) is not None
              and layer.cb_qweight.data.dim() == 2
              and layer.cb_qweight.data.stride(1) == 1)
        if ok:
            # Single uniform rung: one codebook block for every output row (the
            # kernel has no per-row codebook offset). One host sync, once.
            ro = layer._cb_row_offset
            ok = bool((ro.max() == ro.min()).item()) and int(ro[0]) == 0
        if ok:
            from .cuda_ext import get_persistent_ext
            ok = get_persistent_ext() is not None
        if not ok and os.environ.get("PRISMAQUANT_DEBUG_PREFIXES") == "1":
            print(f"[cb] persistent-TC ineligible: {self.prefix} "
                  f"(K={K} k={self.k} ts={self.type_size} n_sub={self.n_sub})")
        layer._cb_ptc_ok = ok
        return ok

    def _fused_fp8_lut_ok(self, layer) -> bool:
        """Whether the offset-free fp8 fused kernel can decode every row.

        ``cb_fused_prefill_mm_scaled`` receives the flat LUT but no per-row
        offsets, so it always addresses LUT base zero.  A uniform non-zero
        base is therefore not sufficient unless the caller also slices the
        LUT (which this path intentionally does not do).  Production layers
        cache this fact while interning blocks at load time; the tensor check
        is a defensive/testing fallback for manually constructed layers.  The
        format checks pin the four-subtable fp8 layout hard-coded by the CUDA
        prologue rather than relying on the shipped scheme menu implicitly.
        """
        ok = getattr(layer, "_cb_fp8_fused_lut_ok", None)
        if ok is None:
            ro = getattr(layer, "_cb_row_offset", None)
            ok = (ro is not None and ro.numel() == layer._cb_N
                  and ro.numel() > 0
                  and bool((ro == 0).all().item()))
            layer._cb_fp8_fused_lut_ok = ok
        return (not self.is_fp4
                and self.n_sub == 4
                and self.type_size == 4 * self.k
                and bool(ok))

    # -- fp4-MMA fused prefill (opt-in; see the dispatch site below) ---------
    def _fused_fp4_ok(self, layer, K: int) -> bool:
        """Eligibility for the fused fp4 block-scaled prefill (cached per
        layer). Mirrors the kernel's TORCH_CHECKs so a miss is a silent
        fall-through, never a crash."""
        ok = getattr(layer, "_cb_fp4_fused_ok", None)
        if ok is not None:
            return ok
        ok = (self.is_fp4
              and 12 <= self.k <= 24
              and self.n_sub in (1, 2)
              and K % 256 == 0
              and layer._cb_N % 8 == 0
              and self.type_size == 4 * self.k + (9 if self.is_v2 else 16))
        if ok and self.n_sub == 2:
            w0 = self.k - self.k // 2
            ok = ((1 << w0) + (1 << (self.k // 2))) * 2 <= 16384
        if ok:
            # Single uniform codebook block (the kernel has no per-row LUT
            # offset — same constraint as the fp8 fused entries).
            ro = layer._cb_row_offset
            ok = bool((ro.max() == ro.min()).item()) and int(ro[0]) == 0
        if ok:
            try:
                import vllm._custom_ops as vops
                ok = hasattr(vops, "scaled_fp4_quant")
            except Exception:  # noqa: BLE001 - venv without vllm
                ok = False
        if ok:
            from .cuda_ext import get_fused_fp4_ext
            fext = get_fused_fp4_ext()
            ok = (fext is not None
                  and hasattr(fext, "cb_fused_fp4_prefill_mm_scaled"))
        layer._cb_fp4_fused_ok = ok
        return ok

    def _try_fused_fp4(self, layer, x, N: int, K: int, M: int):
        """One fused NVF4 block-scaled GEMM over this layer's packed rows, or
        None if ineligible (the caller then runs the shipping path)."""
        if not self._fused_fp4_ok(layer, K):
            return None
        import vllm._custom_ops as vops
        from .cuda_ext import get_fused_fp4_ext
        fext = get_fused_fp4_ext()
        lut = getattr(layer, "_cb_fp4_lut", None)
        if lut is None:
            dev = layer._cb_flat.device
            lut = codec.build_fp4_value_lut(
                layer._cb_flat, self.k, self.n_sub).to(dev)
            layer._cb_fp4_lut = lut
            layer._cb_fp4_compose_u8 = (
                codec.build_compose_u8(self._sub_table).to(dev) if self.is_v2
                else torch.zeros(1, dtype=torch.uint8, device=dev))
            layer._cb_fp4_ones = torch.ones(N, dtype=torch.float32, device=dev)
        x2 = x.reshape(-1, K)
        amax = x2.float().abs().amax()
        gs = ((448.0 * 6.0) / amax.clamp_min(1e-12)).to(torch.float32)
        aq, sfa = vops.scaled_fp4_quant(x2, gs)
        a_scales = (1.0 / gs).reshape(1).expand(M).contiguous()
        y = fext.cb_fused_fp4_prefill_mm_scaled(
            aq, sfa.view(torch.uint8).reshape(-1), layer._cb_qw_padded,
            layer._cb_fp4_lut, layer._cb_fp4_compose_u8, a_scales,
            layer._cb_fp4_ones, N, K, self.k, self.n_sub, self.type_size,
            self.is_v2)
        return y.reshape(*x.shape[:-1], N)

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
        # ROCm/HIP: one guarded delegation, additive by construction. `_HIP` is
        # None on a CUDA box (and whenever the HIP extension does not build), so
        # the path below is byte-identical to what it was before ROCm support
        # existed. See linear_hip.maybe_apply — it returns None for every rung
        # or shape the HIP kernels do not cover, and the CUDA/Triton dispatch
        # then runs unchanged.
        if _HIP is not None:
            _hip_out = _HIP.maybe_apply(self, layer, x, bias)
            if _hip_out is not None:
                return _hip_out

        N, K = layer._cb_N, layer._cb_K
        M = x.reshape(-1, K).shape[0]

        # fp4-MMA fused prefill (OPT-IN, default OFF — the shipping path below
        # is byte-identical until PRISMAQUANT_CB_FUSED_FP4 is set). Decodes the
        # packed CB rows inside the CUTLASS block-scaled prologue and runs the
        # native NVF4 MMA (OMMA.SF.16864, k=64). NOTE the activation bucket
        # CHANGES on this path: native NVFP4 quantization (per-tensor fp32
        # global x per-group-16 ue4m3 SF) instead of the Triton/transient
        # paths' fp32-group-scale QDQ — the hardware SF operand is ue4m3, an
        # fp32 group scale is unrepresentable. Promotion therefore requires a
        # served KL A/B (docs/KERNELS.md), which is
        # why this lands opt-in. "1" = all prefill M; "midm" = only the fp8
        # kernel's proven 16<M<=128 niche.
        if (self.is_fp4 and bias is None and M > PREFILL_M_THRESHOLD
                and _fp4_fused_mode() in ("1", "midm")
                and not (_fp4_fused_mode() == "midm" and M > 128)):
            y = self._try_fused_fp4(layer, x, N, K, M)
            if y is not None:
                return y

        # Decode regime (M small), plus fp4-v1 which has no transient path yet
        # (its v1 e4m3 plane is not composed during expansion) — Triton decode.
        if M <= PREFILL_M_THRESHOLD or (self.is_fp4 and not self.is_v2):
            if M <= CUDA_GEMV_M_MAX and self._cuda_gemv_ok():
                if self.is_fp4:
                    # fp4-v2 CUDA GEMV: act-QDQ (fp4 group-16 RTN) runs OUTSIDE
                    # the kernel — exactly as the Triton fp4 path — so
                    # CUDA-vs-Triton numerics stay aligned. The resolver picks
                    # the fused CUDA op when the ext has it and the eager codec
                    # otherwise; the two are bit-identical (tests/
                    # test_fp4_act_qdq.py asserts torch.equal). The kernel
                    # gathers the bf16 codebook and composes the two-tier scale
                    # in-register from the packed 9-byte plane.
                    xq = fp4_act_qdq_or_codec(x)
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
            xq = fp4_act_qdq_or_codec(x)
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

        # Mid-M fused decode-in-prologue (task 7's measured WIN niche): at
        # M in (16, 128] ONE M-tile covers the batch, so decoding B inside
        # the CUTLASS prologue has no redundancy and beats expand+GEMM by
        # 1.04-1.45x (M=32/64/128, GB10). OPT-IN (PRISMAQUANT_CB_FUSED_MIDM=1).
        # Numerics: the _scaled entry applies BOTH the per-token activation
        # scale and the per-channel weight scale inside its fp32 EVT epilogue
        # and rounds once to bf16 — the same rounding ORDER as
        # cutlass_scaled_mm (the older unscaled entry rounded first and scaled
        # in python, which moved served prompt logprobs by up to 0.86 nats).
        # Step-4 rungs only (the kernel's KBits template dispatch).
        if (bias is None and 16 < x2.shape[0] <= 128
                and self.k in (28, 32, 36, 40, 44, 48)
                and os.environ.get("PRISMAQUANT_CB_FUSED_MIDM", "1") != "0"
                and self._fused_fp8_lut_ok(layer)):
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            if fext is not None and hasattr(fext, "cb_fused_prefill_mm_scaled"):
                y = fext.cb_fused_prefill_mm_scaled(
                    xq, layer.cb_qweight.data, layer._cb_flat_fp8,
                    sa.reshape(-1).to(torch.float32).contiguous(),
                    layer._cb_scale.reshape(-1).to(torch.float32).contiguous(),
                    N, K, self.k)
                return y.reshape(*x.shape[:-1], N)

        # Persistent-N tensor-core prefill (#4b, OPT-IN and quarantined):
        # PRISMAQUANT_CB_PREFILL_DENSE=persistent routes large-M fp8-CB prefill
        # into the fused decode+TC-GEMM kernel, skipping the [N,K] e4m3
        # transient entirely. Any constraint miss falls through SILENTLY to the
        # shipping expand+cutlass path below (no behaviour change when off).
        if (os.environ.get("PRISMAQUANT_CB_PREFILL_DENSE") == "persistent"
                and x2.shape[0] > 128
                and self._ptc_ok(layer, K)):
            from .ops import cb_prefill_persistent_tc
            d = cb_prefill_persistent_tc(
                xq, layer.cb_qweight.data, layer._cb_flat_fp8, N, K,
                self.k, self.type_size,
                int(os.environ.get("PRISMAQUANT_PTC_VARIANT", "1")))
            # Scale convention: the kernel returns the UNSCALED accumulation,
            # so the per-token activation scale `sa` [M,1] and the
            # per-output-channel weight scale _cb_scale [N] are applied
            # outside. NOTE: this is the OLD convention — the mid-M fused path
            # above has since moved to an in-epilogue fp32 scale
            # (cb_fused_prefill_mm_scaled) to match cutlass_scaled_mm's
            # rounding order; this route still rounds to bf16 first, which is
            # a rounding-ORDER difference vs the shipping path (hence the
            # opt-in gate). Behaviour deliberately unchanged here.
            y = (d.float() * sa.reshape(-1, 1)
                 * layer._cb_scale.reshape(1, -1).float()).to(x.dtype)
            if bias is not None:
                y = y + bias
            return y.reshape(*x.shape[:-1], N)

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
