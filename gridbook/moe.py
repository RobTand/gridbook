"""``PrismaQuantCBMoEMethod`` — FusedMoE serving for stacked CB expert weights
(docs/nvfp4-cb-plan/moe_cb_design.md §4, LAYOUT.md §3 stacked layout).

Each expert stack ships as ONE tensor per role: ``<q>.cb_qweight`` uint8
``(E, out, (in/256)·type_size)`` (+ fp8 ``<q>.weight_scale`` ``(E, out)``), where
``cb_qweight[e]`` is exactly the dense §1 superblock layout. All experts of a
stack share one format + one codebook (per-layer uniformity, union-find at
export; asserted here). Serving mirrors ``GGUFMoEMethod``: register w13/w2 expert
buffers, then a per-expert **transient** decode (one expert's ``[out, in]`` bf16
tile live at a time — INV-1, the dense transient pattern extended to experts).

  w13 = fused gate_up_proj : (E, 2·inter, hidden)  -> cb_qweight (E, 2·inter, bytes)
  w2  = down_proj          : (E, hidden, inter)    -> cb_qweight (E, hidden, bytes)

NOTE (post-27B GPU/vLLM validation): the FusedMoE weight-loader wiring and the
routed forward are exercised by the synthetic-MoE serve smoke, deferred to the
first idle GPU window (resource-discipline hold). The decode math per expert is
the dense path (bit-exact CPU/triton-tested in test_cb_kernels / test_two_tier_v2);
this file adds the expert-stack loop + buffer mapping. CPU unit tests below pin
the buffer shapes, w13/w2 split, and per-layer uniformity.
"""
from __future__ import annotations

import os
import sys

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.utils import set_weight_attrs

from . import codec
from .expand import (
    expand_cb_to_fp8,
    expand_cb_to_value,
    expand_fp4_v2_to_weight,
)
from .moe_autotune import STOCK as _AUTO_STOCK, cb_prefill_auto
from .moe_l2 import (
    L2_PIPELINE,
    cb_l2_cap_bytes,
    cb_l2_live_groups,
    cb_l2_min_m,
    cb_l2_pin_action,
    cb_l2_plan,
)
from .moe_routing import cb_grouped_pad_routing
from .ops import dispatch_via_op


def _row_bytes(in_features: int, type_size: int) -> int:
    return (in_features // codec.SUPERBLOCK) * type_size


# torch._grouped_mm — the single-launch ragged grouped GEMM (2d×3d: mat_a
# [P,K] × mat_b [G,K,N] with cumulative-end offsets [G] -> [P,N]) that collapses
# the batched-prefill per-expert-segment GEMMs into ONE kernel. Opt-in
# (PRISMAQUANT_CB_PREFILL_GROUPED_MM=1) because its ragged-offset / B-layout
# constraints on sm_121 are unproven in this tree; the segmented fallback is the
# correctness-first default. Availability + first-use viability are cached so a
# reject degrades to segmented once, not per forward.
_GROUPED_MM_OK: bool | None = None


def _grouped_mm_available() -> bool:
    global _GROUPED_MM_OK
    if _GROUPED_MM_OK is None:
        _GROUPED_MM_OK = hasattr(torch, "_grouped_mm")
    return _GROUPED_MM_OK


def _disable_grouped_mm(exc) -> None:
    global _GROUPED_MM_OK
    if _GROUPED_MM_OK:
        import sys
        print(f"[prismaquant-cb] torch._grouped_mm unusable for CB prefill "
              f"({type(exc).__name__}: {exc}); using the segmented GEMM.",
              file=sys.stderr, flush=True)
    _GROUPED_MM_OK = False


class PrismaQuantCBMoEMethod(FusedMoEMethodBase):
    """CB decode for RoutedExperts (FusedMoE) — one uniform CB format per layer."""

    def __init__(self, quant_config, moe: FusedMoEConfig, scheme: dict,
                 prefix: str) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = (sc == codec.SCALE_CODING_TWO_TIER)
            self._sub_table = codec.TWO_TIER_SUB_TABLE if self.is_v2 else None
        if self.is_fp4 and not self.is_v2:
            # fp4-v1 expert transient is a follow-up (no compose-during-expand);
            # export MoE experts as fp8-CB or fp4 two-tier v2.
            raise NotImplementedError(
                f"{prefix}: fp4 MoE experts require two-tier v2 scale coding "
                "(fp4-v1 expert transient not yet implemented)")

    # -- weight buffers (stacked experts) ------------------------------------
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        del params_dtype
        E = num_experts
        inter = intermediate_size_per_partition
        layer._cb_hidden = hidden_size
        layer._cb_inter = inter
        # extra_weight_attrs already carries the weight_loader; ONE
        # set_weight_attrs per param (a second call trips vLLM's
        # "Overwriting existing tensor attribute" assert — 35B first serve).
        attrs = dict(extra_weight_attrs)
        # w13 = gate_up: out=2*inter, in=hidden.  w2 = down: out=hidden, in=inter.
        w13 = torch.nn.Parameter(torch.empty(
            E, 2 * inter, _row_bytes(hidden_size, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w13, {**attrs, "is_transposed": False})
        layer.register_parameter("w13_cb_qweight", w13)

        w2 = torch.nn.Parameter(torch.empty(
            E, hidden_size, _row_bytes(inter, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w2, {**attrs, "is_transposed": False})
        layer.register_parameter("w2_cb_qweight", w2)

        if not self.is_fp4:                       # fp8: per-(expert, out) scale
            w13s = torch.nn.Parameter(
                torch.empty(E, 2 * inter, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w13s, dict(attrs))
            layer.register_parameter("w13_weight_scale", w13s)
            w2s = torch.nn.Parameter(
                torch.empty(E, hidden_size, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w2s, dict(attrs))
            layer.register_parameter("w2_weight_scale", w2s)

        # Instance-level load hook (GGUF-plugin pattern, zero core patches):
        # vLLM's RoutedExperts.load_weights maps checkpoint names by
        # substring-replacing the projection name, which (a) derives DOTTED
        # attribute names for our `<proj>.cb_qweight` suffix (getattr fails)
        # and (b) applies a bf16-orientation transpose heuristic that would
        # corrupt byte tensors (last dim = row_bytes, never hidden). CB
        # tensors therefore load DIRECTLY into our stacked params; every
        # other tensor delegates to the original loader untouched.
        orig_load = getattr(layer, "load_weights", None)
        if not getattr(layer, "_cb_load_wrapped", False) and orig_load is not None:
            prefix = self.prefix
            cb_map = {
                "gate_up_proj.cb_qweight": "w13_cb_qweight",
                "down_proj.cb_qweight": "w2_cb_qweight",
                "gate_up_proj.weight_scale": "w13_weight_scale",
                "down_proj.weight_scale": "w2_weight_scale",
            }

            def _cb_load_weights(weights):
                deferred = []
                for name, w in weights:
                    pname = cb_map.get(name)
                    if pname is not None and hasattr(layer, pname):
                        p = getattr(layer, pname)
                        if tuple(p.shape) != tuple(w.shape):
                            raise ValueError(
                                f"{prefix}.{name}: checkpoint shape "
                                f"{tuple(w.shape)} != param {tuple(p.shape)}"
                                f" — stacked (E, out, bytes) contract violated")
                        p.data.copy_(w.to(p.dtype))
                        yield pname
                    else:
                        deferred.append((name, w))
                if deferred:
                    yield from orig_load(deferred)

            layer.load_weights = _cb_load_weights
            layer._cb_load_wrapped = True

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        return None

    # -- per-stack codebook / compose + uniformity assert --------------------
    def process_weights_after_loading(self, layer: torch.nn.Module):
        dev = layer.w13_cb_qweight.device
        E = layer.w13_cb_qweight.shape[0]
        codebooks = self.quant_config.get_codebooks()
        ref = self.scheme["codebook_ref"]
        names = ref if isinstance(ref, list) else [ref]
        subs = [codebooks[n].to(dev) for n in names]
        layer._cb_flat = codec.build_flat_codebook(subs)
        layer._cb_row0 = torch.zeros(1, dtype=torch.int32, device=dev)
        if self.is_v2:
            layer._cb_compose = codec.build_compose_table(
                self._sub_table).to(dev)
        else:
            layer._cb_compose = torch.zeros(1, dtype=torch.float32, device=dev)
        # Per-layer uniformity: one format for all experts (union-find at
        # export). The stacked buffer is single-format by construction; assert
        # the byte width matches the scheme so a mis-exported stack fails loudly.
        exp_w13 = _row_bytes(layer._cb_hidden, self.type_size)
        exp_w2 = _row_bytes(layer._cb_inter, self.type_size)
        assert layer.w13_cb_qweight.shape[2] == exp_w13, (
            f"{self.prefix}: w13 byte width {layer.w13_cb_qweight.shape[2]} != "
            f"{exp_w13} (type_size/uniformity mismatch)")
        assert layer.w2_cb_qweight.shape[2] == exp_w2
        layer._cb_E = E
        from .ops import register_cb_layer
        layer._cb_layer_id = register_cb_layer(self, layer)

    # -- per-expert decode to a bounded transient [out, in] bf16 -------------
    def _decode_expert(self, layer, which: str, e: int) -> torch.Tensor:
        """Decode ONE expert's CB weight to a bf16 ``[out, in]`` transient
        (INV-1: one expert live at a time). fp8: value × per-channel scale;
        fp4 v2: value × composed group scale."""
        qw = getattr(layer, f"{which}_cb_qweight")[e]          # (out, bytes)
        out = qw.shape[0]
        # w13 in=hidden (gate_up), w2 in=inter (down).
        in_f = layer._cb_hidden if which == "w13" else layer._cb_inter
        qwp = codec.pad_qweight(qw.contiguous())
        row0 = torch.zeros(out, dtype=torch.int32, device=qw.device)
        if self.is_fp4:                                        # fp4 v2
            W = expand_fp4_v2_to_weight(
                qwp, layer._cb_flat, row0, layer._cb_compose,
                out, in_f, self.k, self.n_sub, self.type_size)
        else:                                                  # fp8
            val = expand_cb_to_value(qwp, layer._cb_flat, row0,
                                     out, in_f, self.k, self.n_sub,
                                     self.type_size, is_fp4=False)
            ws = getattr(layer, f"{which}_weight_scale")[e].to(torch.float32)
            W = (val.float() * ws[:, None]).to(torch.bfloat16)
        return W                                               # (out, in) bf16

    def apply(self, layer: RoutedExperts, x: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts, shared_experts_input) -> torch.Tensor:
        # Shared expert (e.g. hy_v3): apply() correctly returns ROUTED-ONLY.
        # vLLM's MoERunner._apply_quant_method runs the SharedExperts wrapper
        # SEPARATELY (`_maybe_apply_shared_experts` -> `SharedExperts._layer(
        # shared_experts_input)`), producing shared_output, and _maybe_combine
        # adds it to our routed output — verified against the vLLM v0.23 runner
        # (moe_runner.py). The wrapper's `_layer` IS the shared_mlp module,
        # whose CB weights we decoded to bf16 at load (moe_toplevel_loader), so
        # the shared contribution is computed and included; a routed-only return
        # here is the contract, NOT a dropped component. (The `_unpack` path also
        # accepts a `(shared, routed)` tuple for methods that fuse the shared
        # expert into their kernel — we don't, so single-tensor is correct.)
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input unsupported for CB MoE")
        # M-branch hoist (see ops.py/linear.py): the token-count branch AND
        # the prefill loop's host syncs live inside ONE opaque custom op, so
        # compile/capture at decode sizes record the grouped decode kernels
        # and dynamo never traces the loop (mode-3 no longer needs the stock
        # prefill path for graph safety). PRISMAQUANT_CB_DISPATCH=inline
        # restores in-graph dispatch for A/B bisection.
        # Unregistered layers (synthetic test fixtures that skip
        # process_weights_after_loading) fall back to inline dispatch.
        lid = getattr(layer, "_cb_layer_id", None)
        if lid is not None and dispatch_via_op():
            from .ops import cb_moe_forward
            return cb_moe_forward(x, topk_weights, topk_ids, lid)
        return self._apply_inline(layer, x, topk_weights, topk_ids)

    def _apply_inline(self, layer, x, topk_weights, topk_ids):
        act = MoEActivation.from_str(layer.activation.value)
        num_tokens = x.shape[0]

        # Decode regime: the grouped CUDA path — ONE kernel launch per
        # projection covers every routed (token, expert) pair (the per-expert
        # loop below costs ~10k host syncs/launches per token: 3.52 tok/s
        # served vs BF16's 28.4 on the 35B A3B). Covers fp8-CB v1 and fp4-CB
        # two-tier v2; _cuda_moe_ok gates the format (fp4-v1 has no grouped
        # path yet and falls through to the loop below).
        if num_tokens <= 16 and self._cuda_moe_ok(layer):
            return self._apply_grouped_decode(
                layer, x, topk_weights, topk_ids, act)

        # Prefill / fallback (num_tokens > 16, or no grouped-CUDA decode path).
        # Two implementations, env-selected via PRISMAQUANT_CB_PREFILL:
        #
        #   'batched' (DEFAULT) — _apply_prefill_batched. Round-1 cost model of
        #     the loop it replaces: on Hy3 (192 experts, ~all hit at prefill) the
        #     per-expert loop pays, PER HIT EXPERT, one act-QDQ + TWO Triton
        #     transient expands (w13/w2, ~192×2 launches/layer) + two F.linear +
        #     an index_add_ — the Triton-expand and QDQ launches are the dominant
        #     remaining prefill structure cost (~112 tok/s). The batched path
        #     issues ONE act-QDQ over all tokens and ONE expand per projection
        #     per expert-CHUNK (the dense expander over a reshaped [(C*out),
        #     bytes] view), collapsing those E expands + E QDQ passes to
        #     E/chunk launches; the GEMM stays grouped/segmented bf16.
        #
        #   'loop' — _apply_prefill_loop, kept verbatim for A/B bisection. Its
        #     hit-expert list already costs ONE host sync (the pre-fix per-expert
        #     `sel.any()` cost E syncs — the dominant share of the 3.5 s TTFT);
        #     the batched path preserves that one-sync property.
        #
        # Numerics: the two paths use BIT-IDENTICAL weights (same expander) and
        # BIT-IDENTICAL per-token QDQ (QDQ is a per-row op — see
        # _apply_prefill_batched); only the GEMM accumulation and the
        # cross-expert combine reassociate (REASSOCIATION-CLASS), held to the
        # tolerance contract by tests/test_moe_batched_prefill.py and gated by
        # the served logprob A/B before adoption. Prefill is eager/uncaptured
        # (FULL_DECODE_ONLY), so this path needs no CUDA-graph capture-safety.
        # Default LOOP until the batched path passes its at-scale served gate:
        # the first 1.4k-token prefill on Hy3 crashed the serve (transient
        # chunk tiles ~1.6 GB vs the loop's ~56 MB against thin post-KV
        # slack, 2026-07-20). Batched stays opt-in: PRISMAQUANT_CB_PREFILL=batched.
        #
        #   'stock' — _apply_prefill_stock. The CAPTURE-SAFE successor to
        #     'batched' (task 15): transiently expand each expert-CHUNK into a
        #     HARDWARE format (fp8-CB -> fp8 bytes, fp4-CB-v2 -> bf16) and hand it
        #     to vLLM's OWN fused-MoE grouped Triton kernel with DEVICE-SIDE
        #     routing (moe_align_block_size). No host reads of device data
        #     anywhere (no .tolist/.item/.cpu/nonzero-to-python) and a FIXED
        #     python trip count (ceil(E/chunk) over ALL experts, empty experts a
        #     masked pass) — the prerequisite for cudagraph_mode=FULL, which the
        #     'batched' path's two .tolist() syncs + tensor-derived python bounds
        #     preclude. Our activation QDQ is preserved: fp8 uses vLLM's per-token
        #     fp8 dynamic between the two projections (bit-equivalent to
        #     codec.fp8_dynamic_act_qdq — test_stock_fp8_quant_matches_codec),
        #     fp4-v2 runs codec.fp4_group16_act_qdq explicitly on the module input
        #     AND the intermediate. Opt-in: PRISMAQUANT_CB_PREFILL=stock.
        # Default: fp8-CB rides 'stock' (vLLM's own fused-MoE grouped kernel
        # over CUDA-expanded e4m3 chunks — measured 5.5x the loop on
        # Laguna-256E, 2026-07-23, with the slack-gate discipline bounding
        # the ~1.6 GB chunk transient). fp4-CB keeps 'loop' until its stock
        # variant (bf16 expand) gets the same at-scale measurement.
        #
        #   'grouped_fused' — _apply_prefill_grouped_fused (round 1 of the MoE
        #     fused campaign, OPT-IN). Drops the stock path's HBM e4m3 expand
        #     round-trip (write N*K bytes, read them back) by decoding each
        #     expert's packed CB rows INSIDE the CUTLASS prologue
        #     (cb_fused_prefill_mm_scaled, the dense mid-M default). Round 1 is
        #     a host-side loop over experts with ONE device->host sync per
        #     layer (the E+1 segment boundaries); round 2 replaces the loop
        #     with a true array-of-problems grouped CUTLASS over the same
        #     collective. Any constraint miss falls through to 'stock'.
        #
        #     ROUND 2 (_apply_prefill_grouped_fused_v2) is what
        #     'grouped_fused' selects when the grouped binding is present: ONE
        #     launch per projection stage over a TileM-padded, expert-sorted row
        #     collective, retiring R1's 2*E launches (~5-10 ms/layer of pure
        #     launch overhead once R1 had removed the expand round-trip).
        #     'grouped_fused_r1' forces R1 — the bisection reference R2 is
        #     validated against.
        # fp8-CB default: stock. grouped_fused won on the 35B (+9%, KL gate
        # passed) but REGRESSED on Laguna-class (1,503 vs 1,821 tok/s @8k,
        # 2026-07-26): R2 re-decodes each expert's B per M-tile
        # (ceil(m_e/TileM) x) and pads to tile multiples — decode redundancy
        # scales with expert SIZE, and Laguna's experts are ~6x the 35B's.
        # Promotion reverted per the two-model ladder rule; grouped_fused
        # stays opt-in.
        #
        #   'l2_pipeline' — _apply_prefill_l2_pipeline (round 4). Decodes into
        #     an L2-PINNED rotating scratch pair instead of a fresh HBM tile,
        #     attacking the decoded-weight round-trip the earlier rounds left
        #     (~17 of ~42 ms/layer on a large-expert MoE). Falls through to
        #     'stock' when the build/format/window cap misses. It is also a
        #     candidate of 'auto' — no promotion is decided in code.
        #
        #   'auto' — _apply_prefill_auto (OPT-IN until a two-model gate clears
        #     it). The principled end state: MEASURE stock and grouped_fused (at
        #     each compiled TileM) once per layer on the layer's own real inputs
        #     and cache the argmin. No shape heuristic, no model table — the
        #     regression above is exactly the kind of model-dependent crossover
        #     that a static default cannot express.
        # fp8-CB default: 'auto' — measured per-layer selection over stock +
        # grouped_fused at every rung-feasible TileM (cuda-event timing on
        # the first qualifying prefill, cached; deterministic stock output
        # on the tuning call). Two-model gate 2026-07-26: 35B 4,405 vs best
        # fixed 4,285; Laguna-class 2,063 vs best fixed 1,821 — auto >= best
        # fixed mode on both. Composed paths are individually KL-gated.
        # NOTE: l2_pipeline is DIAGNOSTIC-ONLY (2026-07-27): wedged the live
        # serve three times (overlapped: stream/capture deadlock; serial:
        # still wedges despite non-default-stream test battery green). The
        # L2-residency hypothesis remains unmeasured; do not add it to auto
        # candidates until a live serve survives a full prefill battery.
        mode = os.environ.get("PRISMAQUANT_CB_PREFILL") or (
            "auto" if not self.is_fp4 else "loop")
        if mode == "auto":
            return self._apply_prefill_auto(
                layer, x, topk_weights, topk_ids, act)
        if mode in ("grouped_fused", "grouped_fused_r1"):
            out = None
            if mode == "grouped_fused":
                out = self._apply_prefill_grouped_fused_v2(
                    layer, x, topk_weights, topk_ids, act)
            if out is None:
                out = self._apply_prefill_grouped_fused(
                    layer, x, topk_weights, topk_ids, act)
            if out is not None:
                return out
            return self._apply_prefill_stock(
                layer, x, topk_weights, topk_ids, act)
        if mode == "l2_pipeline":
            out = self._apply_prefill_l2_pipeline(
                layer, x, topk_weights, topk_ids, act)
            if out is not None:
                return out
            return self._apply_prefill_stock(
                layer, x, topk_weights, topk_ids, act)
        if mode == "loop":
            return self._apply_prefill_loop(
                layer, x, topk_weights, topk_ids, act)
        if mode == "stock":
            return self._apply_prefill_stock(
                layer, x, topk_weights, topk_ids, act)
        return self._apply_prefill_batched(
            layer, x, topk_weights, topk_ids, act)

    # -- prefill: grouped FUSED (decode-in-prologue, round 1) ---------------
    def _gf_ok(self, layer) -> bool:
        """Eligibility for the grouped-fused prefill (cached per layer; the
        answer never changes after load). Mirrors the kernel's own
        TORCH_CHECKs so a miss is a silent fall-through, never a crash:

          * fp8-CB only (the fused kernel's LUT is the 4-sub-table e4m3
            codebook; fp4 two-tier composes a scale the prologue can't);
          * k in {28,32,36,40,44,48} — the KBits template dispatch, uniform
            per layer by the export union-find;
          * both projection K's are multiples of 256 (SUPERBLOCK, and the
            kernel's K%256 check);
          * packed row stride == row_bytes == (K/256)*4*k, which satisfies
            BOTH `stride(0) % 16 == 0` (4*k in {144,160,176,192}) and
            `stride(0) >= (K/256)*4*k_bits` with equality (UNPADDED rows).
        """
        ok = getattr(layer, "_cb_gf_ok", None)
        if ok is not None:
            return ok
        ok = (not self.is_fp4
              and self.n_sub == 4
              and self.k in (28, 32, 36, 40, 44, 48)
              and self.type_size == 4 * self.k
              and layer._cb_hidden % codec.SUPERBLOCK == 0
              and layer._cb_inter % codec.SUPERBLOCK == 0
              and hasattr(layer, "w13_weight_scale"))
        if ok:
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            ok = fext is not None and hasattr(fext, "cb_fused_prefill_mm_scaled")
        if ok:
            # 3-D stacked buffers: [e] must give a 2-D CONTIGUOUS [out, bytes]
            # view (stride(1)==1, stride(0)==row_bytes) — true for a slice of a
            # contiguous 3-D tensor, asserted rather than assumed.
            for which, in_f in (("w13", layer._cb_hidden),
                                ("w2", layer._cb_inter)):
                qw = getattr(layer, f"{which}_cb_qweight")
                rb = _row_bytes(in_f, self.type_size)
                # stride(0) matters only to R2, which hands the WHOLE stack to
                # the kernel and TORCH_CHECKs the expert stride; checking it
                # here keeps an exotic layout a silent fall-through rather than
                # a crash inside the extension.
                ok = ok and (qw.dim() == 3 and qw.stride(2) == 1
                             and qw.stride(1) == rb and qw.shape[2] == rb
                             and qw.stride(0) == qw.shape[1] * rb
                             and rb % 16 == 0)
        layer._cb_gf_ok = bool(ok)
        return layer._cb_gf_ok

    def _apply_prefill_grouped_fused(self, layer, x, topk_weights, topk_ids,
                                     act):
        """ROUND 1 of the MoE grouped fused prefill
        (``PRISMAQUANT_CB_PREFILL=grouped_fused``, OPT-IN). Returns ``None`` on
        any constraint miss so the caller falls through to 'stock'.

        WHAT IT REMOVES. The stock path CUDA-expands each expert chunk's packed
        CB rows into an HBM e4m3 tile and then runs vLLM's Triton fused-MoE
        grouped GEMM over that tile: N*K bytes written, then read straight back
        — a pure round-trip tax (~17 ms/layer expand vs ~25 ms/layer GEMM on a
        256-expert reference). ``cb_fused_prefill_mm_scaled`` decodes the packed
        rows INSIDE the CUTLASS prologue, so the tile never exists.

        ROUTING / SYNC DESIGN. Per (token, expert) PAIR p = t*top_k + j:
        ``pair_expert = topk_ids.reshape(-1)``. A STABLE argsort by expert makes
        each expert's rows a contiguous, token-major segment (the loop path's
        ``torch.where(topk_ids == e)`` order, so per-segment GEMMs bit-match the
        loop and only the combine reassociates). Segment boundaries are
        ``cumsum(bincount(pair_expert, minlength=E))`` — an [E+1] tensor built
        entirely on device. ONE ``.tolist()`` fetches those E+1 offsets; that is
        the path's ONLY device->host sync per layer (the 'batched' path already
        tolerates two, the loop one). Python then slices the sorted row-index
        tensor per expert with no further reads, and zero-row experts are
        skipped on the fetched offsets alone. NOT capture-safe (host-read
        control flow) — same class as 'batched'; prefill is eager
        (FULL_DECODE_ONLY), so that costs nothing today.

        NUMERICS. Activations: the same per-token fp8-dynamic QDQ the stock path
        uses (``moe_kernel_quantize_input``, bit-equivalent to
        codec.fp8_dynamic_act_qdq — test_stock_fp8_quant_matches_codec) on the
        module input AND the intermediate, gathered per expert (QDQ is a
        per-row op, so gather-after-quant == quant-after-gather bit-exactly).
        Weights: the fused prologue's decode is bit-exact vs the expander.
        Scales: the kernel's fp32 EVT epilogue applies per-token a_scales and
        per-channel b_scales and rounds ONCE to bf16 — the same rounding order
        as cutlass_scaled_mm and as the stock W8A8 kernel. Only the GEMM
        accumulation and the cross-expert combine reassociate
        (REASSOCIATION-CLASS, the suite's 2e-2 contract).

        MIN-M. ``check_fused_inputs`` imposes NO minimum M: M is a runtime GEMM
        extent and CUTLASS covers a short expert with one partial 128-row tile.
        No padding is needed (and none is done).
        """
        if not self._gf_ok(layer):
            return None
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden                        # w13 in / w2 out
        inter = layer._cb_inter
        N1 = 2 * inter                               # w13 out (gate_up)
        d = N1 // 2
        lut = self._stock_cb_flat_fp8(layer)
        kb = self.k
        fp8_dtype = current_platform.fp8_dtype()

        out = torch.zeros_like(x)

        # ---- routing (all device-side up to the single sync) ---------------
        pair_expert = topk_ids.reshape(-1).to(torch.long)          # [P]
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)                              # [P]
        order = torch.argsort(pair_expert, stable=True)
        ptok_sorted = pair_token[order]                            # [P]
        pw_sorted = topk_weights.reshape(-1)[order]                # [P]
        counts = torch.bincount(pair_expert, minlength=E)          # [E]
        bounds_t = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        bounds = bounds_t.tolist()                                 # THE one sync

        # ---- input activation QDQ, ONCE over all token rows ----------------
        a1, a1s = moe_kernel_quantize_input(
            x, None, fp8_dtype, per_act_token_quant=True)
        a1s = a1s.reshape(-1).to(torch.float32)

        w13q = layer.w13_cb_qweight.data
        w2q = layer.w2_cb_qweight.data
        w13s = layer.w13_weight_scale
        w2s = layer.w2_weight_scale

        for e in range(E):
            p0, p1 = bounds[e], bounds[e + 1]
            if p1 == p0:                              # zero-row expert: skip
                continue
            rows = ptok_sorted[p0:p1]
            ae = a1.index_select(0, rows).contiguous()             # [m_e, Kh]
            ase = a1s.index_select(0, rows).contiguous()           # [m_e]

            # stage 1: gate_up = ae @ W13[e]^T (decode in prologue)
            gate_up = fext.cb_fused_prefill_mm_scaled(
                ae, w13q[e], lut, ase,
                w13s[e].reshape(-1).to(torch.float32).contiguous(),
                N1, Kh, kb)                                        # [m_e, N1]
            del ae, ase
            a = torch.empty((gate_up.shape[0], d), dtype=gate_up.dtype,
                            device=dev)
            apply_moe_activation(act, a, gate_up)                  # silu(g)*u
            del gate_up

            # intermediate QDQ — the stock path's exact function
            a2, a2s = moe_kernel_quantize_input(
                a, None, fp8_dtype, per_act_token_quant=True)
            del a

            # stage 2: y = a2 @ W2[e]^T
            y = fext.cb_fused_prefill_mm_scaled(
                a2.contiguous(), w2q[e], lut,
                a2s.reshape(-1).to(torch.float32).contiguous(),
                w2s[e].reshape(-1).to(torch.float32).contiguous(),
                Kh, inter, kb)                                     # [m_e, Kh]
            del a2, a2s
            y = y * pw_sorted[p0:p1, None].to(y.dtype)             # router w
            out.index_add_(0, rows, y.to(out.dtype))
            del y
        return out

    # -- prefill: grouped FUSED round 2 (one launch per stage) --------------
    def _gf2_ok(self, layer) -> bool:
        """Eligibility for ROUND 2. Everything R1 requires (``_gf_ok`` — fp8-CB,
        the KBits rung, K%256, unpadded 16B-aligned rows) PLUS the grouped
        binding itself, which ships independently of ``cb_fused_prefill_mm_scaled``:
        an older extension build must fall back to R1, not crash."""
        ok = getattr(layer, "_cb_gf2_ok", None)
        if ok is not None:
            return ok
        ok = self._gf_ok(layer)
        if ok:
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            ok = (fext is not None
                  and hasattr(fext, "cb_fused_moe_grouped")
                  and hasattr(fext, "cb_fused_moe_tile_m"))
        layer._cb_gf2_ok = bool(ok)
        return layer._cb_gf2_ok

    def _gf2_tile_sizes(self, layer) -> list[int]:
        """The TileM values the grouped kernel is actually COMPILED for on this
        build, ascending. Always enumerated from the extension — a (TileM,
        k_bits) pair can be uncompilable (shared-memory limit), so a hardcoded
        list would hand the kernel a tile it cannot serve.

        Precedence: the per-rung query (the only one that knows this layer's
        k_bits), then the general one, then the back-compat single default. An
        extension build predating the newer symbols therefore degrades to
        exactly the one tile it has always supported. Cached per layer (the
        answer is a property of the build + the layer's rung)."""
        sizes = getattr(layer, "_cb_gf2_tiles", None)
        if sizes is not None:
            return sizes
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()
        sizes = []
        if fext is not None:
            for getter, args in (
                    ("cb_fused_moe_tile_sizes_for_kbits", (self.k,)),
                    ("cb_fused_moe_tile_sizes", ()),
                    ("cb_fused_moe_tile_m", ())):
                fn = getattr(fext, getter, None)
                if fn is None:
                    continue
                try:
                    got = fn(*args)
                except Exception:  # noqa: BLE001 — treat as "not offered"
                    continue
                got = [got] if isinstance(got, int) else list(got)
                sizes = sorted({int(s) for s in got if int(s) > 0})
                if sizes:
                    break
        layer._cb_gf2_tiles = sizes
        return sizes

    def _grouped_call(self, fext, args, tile_m: int):
        """``cb_fused_moe_grouped`` with an explicit TileM, tolerating an
        extension build whose binding predates the ``tile_m`` parameter: that
        build only ever has ONE tile, so dropping the argument is correct when
        it equals the default and a fall-through (``None``) otherwise."""
        if getattr(self, "_cb_grouped_tile_arg", True):
            try:
                return fext.cb_fused_moe_grouped(*args, tile_m)
            except TypeError:
                self._cb_grouped_tile_arg = False
        if tile_m != int(fext.cb_fused_moe_tile_m()):
            return None
        return fext.cb_fused_moe_grouped(*args)

    def _apply_prefill_grouped_fused_v2(self, layer, x, topk_weights, topk_ids,
                                        act, *, tile_m=None):
        """ROUND 2 of the MoE grouped fused prefill (selected by
        ``PRISMAQUANT_CB_PREFILL=grouped_fused`` when the grouped binding
        exists). Returns ``None`` on any constraint miss so the caller falls
        back to R1 and then to 'stock'.

        WHAT IT REMOVES. R1 already decodes CB rows inside the CUTLASS prologue,
        so the only structural cost it left is the HOST LOOP: 2*E dense kernel
        launches per layer (~5-10 ms/layer at E=256, comparable to the GEMM work
        itself). R2 issues ONE ``cb_fused_moe_grouped`` per projection stage over
        the whole routed collective; the kernel picks each M-tile's B operand
        from a per-tile expert id.

        PADDED ROUTING. The kernel's tile is uniform, so each expert's rows must
        start on a TileM boundary — hence a PADDED gather rather than R1's exact
        segments. ``cb_grouped_pad_routing`` (see its docstring for the
        capacity-bound proof and the no-host-read construction) returns per-tile
        expert ids and, per padded row, the index into the STABLE-argsorted pair
        array. Stable sort keeps each expert's rows token-major, so a padded
        segment's GEMM sees the same row order as ``_apply_prefill_loop``'s
        ``torch.where(topk_ids == e)`` and only the combine reassociates.
        ``expert_ids`` is built once and REUSED across both stages: the row
        layout is a property of the routing, not of the projection.

        PADDING IS INERT BY CONSTRUCTION, not by cancellation. Padding rows
        gather activation zeros (scale 1.0 — any finite value works since the
        row is zero, but 1.0 avoids a degenerate 0*0 scale in the intermediate
        per-token QDQ), and the combine scatters them into a THROWAWAY row T of
        a ``[T+1, hidden]`` accumulator that is then sliced off. That is
        stronger than zeroing their router weight: it holds even if a padding
        row's output were non-finite, so the guarantee does not depend on the
        kernel's behaviour on an all-zero tile.

        TRIM. ``PRISMAQUANT_CB_GROUPED_TRIM`` (default "1") spends ONE
        ``.item()`` on the real block total and slices the padded collective
        down to it, so no wasted tile is ever launched. R1 already spends one
        sync per layer, so the default is strictly no worse. Setting it to "0"
        keeps the full static capacity — up to E wasted tiles per stage, but the
        path then contains NO host read of device data at all, which is the
        prerequisite for future CUDA-graph capture of prefill (today prefill is
        eager/FULL_DECODE_ONLY, so the trim is the better default).

        NUMERICS. Identical contract to R1: the stock path's own
        ``moe_kernel_quantize_input`` per-token fp8 QDQ on the module input AND
        the intermediate (a per-row op, so gather-after-quant ==
        quant-after-gather bit-exactly), bit-exact prologue weight decode, and
        the same fp32 EVT ``bf16_rn(b_scale * (a_scale * acc))`` rounding order.

        TILE M. ``tile_m`` selects among the tile sizes the kernel was compiled
        for (``_gf2_tile_sizes``); ``None`` means the kernel's own default. It is
        a PERFORMANCE knob with a real tradeoff — a larger tile amortises the
        per-tile B decode over more rows but wastes more padded rows on short
        experts — which is why 'auto' measures it rather than fixing it. Nothing
        else in this method is tile-specific: the capacity bound ``P//tile_m + E``
        is proved for arbitrary tile_m (see ``cb_grouped_pad_routing``), and the
        combine is indexed by ``dest``/``row_src``, whose length is
        ``cap_blocks*tile_m`` by construction — so both generalise unchanged.
        """
        if not self._gf2_ok(layer):
            return None
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden                        # w13 in / w2 out
        inter = layer._cb_inter
        N1 = 2 * inter                               # w13 out (gate_up)
        d = N1 // 2
        lut = self._stock_cb_flat_fp8(layer)
        kb = self.k
        fp8_dtype = current_platform.fp8_dtype()
        # Never hardcode a tile: the kernel owns which TileM it compiled.
        tile_m = int(fext.cb_fused_moe_tile_m() if tile_m is None else tile_m)
        if tile_m <= 0:
            return None

        # ---- routing (device-side; static shapes) --------------------------
        pair_expert = topk_ids.reshape(-1).to(torch.long)          # [P]
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)                              # [P]
        order = torch.argsort(pair_expert, stable=True)            # STABLE
        ptok_sorted = pair_token[order]
        pw_sorted = topk_weights.reshape(-1)[order].to(torch.float32)

        expert_ids, row_src, is_pad, n_blocks = cb_grouped_pad_routing(
            topk_ids, E, tile_m)

        if os.environ.get("PRISMAQUANT_CB_GROUPED_TRIM", "1") == "1":
            nb = int(n_blocks.item())                # THE one sync (optional)
            expert_ids = expert_ids[:nb].contiguous()
            row_src = row_src[:nb * tile_m]
            is_pad = is_pad[:nb * tile_m]

        # ---- input activation QDQ, ONCE over all token rows ----------------
        a1, a1s = moe_kernel_quantize_input(
            x, None, fp8_dtype, per_act_token_quant=True)
        a1s = a1s.reshape(-1).to(torch.float32)

        # ONE index vector does both the gather and the scatter: real rows map
        # to their token, padding rows to the appended zero row / throwaway
        # output row T. Gathering the zeros (rather than masking an fp8 tensor
        # after the fact) keeps every op on this path a plain index_select.
        rows = ptok_sorted.index_select(0, row_src)                 # [Mp]
        dest = torch.where(is_pad, torch.full_like(rows, T), rows)
        a1x = torch.cat([a1, a1.new_zeros((1, Kh))])
        # Padding scale 1.0 (see docstring): the row is zero either way, but a
        # zero scale would make the intermediate QDQ's per-row amax degenerate.
        a1sx = torch.cat([a1s, a1s.new_ones(1)])
        a_pad = a1x.index_select(0, dest).contiguous()
        as_pad = a1sx.index_select(0, dest).contiguous()

        w13s = layer.w13_weight_scale.reshape(E, N1).to(
            torch.float32).contiguous()
        w2s = layer.w2_weight_scale.reshape(E, Kh).to(
            torch.float32).contiguous()

        # stage 1: gate_up = a_pad @ W13[expert_of_tile]^T
        gate_up = self._grouped_call(
            fext, (a_pad, layer.w13_cb_qweight.data, lut, as_pad, w13s,
                   expert_ids, N1, Kh, kb), tile_m)                # [Mp, N1]
        del a_pad, as_pad
        if gate_up is None:                          # binding has no TileM knob
            return None
        a = torch.empty((gate_up.shape[0], d), dtype=gate_up.dtype, device=dev)
        apply_moe_activation(act, a, gate_up)                      # silu(g)*u
        del gate_up

        a2, a2s = moe_kernel_quantize_input(
            a, None, fp8_dtype, per_act_token_quant=True)
        del a

        # stage 2: y = a2 @ W2[expert_of_tile]^T — SAME expert_ids, the row
        # layout is a property of the routing, not of the projection.
        y = self._grouped_call(
            fext, (a2.contiguous(), layer.w2_cb_qweight.data, lut,
                   a2s.reshape(-1).to(torch.float32).contiguous(), w2s,
                   expert_ids, Kh, inter, kb), tile_m)             # [Mp, Kh]
        del a2, a2s
        if y is None:
            return None

        # Padding rows carry pair 0's router weight; harmless, because they are
        # scattered into throwaway row T, which is sliced off.
        pw_pad = pw_sorted.index_select(0, row_src)
        y = y * pw_pad[:, None].to(y.dtype)
        out = torch.zeros((T + 1, Kh), dtype=x.dtype, device=dev)
        out.index_add_(0, dest, y.to(out.dtype))
        return out[:T]

    # -- prefill: L2-resident rotating scratch (round 4) --------------------
    def _l2_ok(self, layer) -> bool:
        """Eligibility for the L2 pipeline (cached per layer; the answer never
        changes after load). Requirements, each a silent fall-through:

          * fp8-CB — the path decodes to native e4m3 bytes and feeds
            ``cutlass_scaled_mm``, the same W8A8 GEMM class the dense path
            trusts; fp4-CB has no faithful packed-NVFP4 transient (see
            ``_apply_prefill_stock``).
          * per-(expert,out) weight scales present (the GEMM's b_scales).
          * K % SUPERBLOCK on both projections (the expander's own constraint).
          * 3-D stacked qweights whose ``[c0:c1]`` slice is a CONTIGUOUS view,
            so the decode reads a raw slice with no pad and no copy — the same
            property ``_expand_stack_slice`` relies on.
          * an extension build carrying ``cb_expand_fp8_into``: the allocating
            expander cannot be used here (a fresh allocation lands outside the
            pinned address range), and an older build must degrade to stock.
        """
        ok = getattr(layer, "_cb_l2_ok", None)
        if ok is not None:
            return ok
        ok = (not self.is_fp4
              and self.n_sub == 4
              and hasattr(layer, "w13_weight_scale")
              and layer._cb_hidden % codec.SUPERBLOCK == 0
              and layer._cb_inter % codec.SUPERBLOCK == 0)
        if ok:
            for which, in_f in (("w13", layer._cb_hidden),
                                ("w2", layer._cb_inter)):
                qw = getattr(layer, f"{which}_cb_qweight")
                rb = _row_bytes(in_f, self.type_size)
                ok = ok and (qw.dim() == 3 and qw.stride(2) == 1
                             and qw.stride(1) == rb and qw.shape[2] == rb
                             and qw.stride(0) == qw.shape[1] * rb)
        if ok:
            from . import ops as pq_ops
            ok = pq_ops.cb_expand_fp8_into_available()
        if ok:
            try:
                from vllm import _custom_ops as vllm_ops  # noqa: F401
                ok = hasattr(vllm_ops, "cutlass_scaled_mm")
            except Exception:  # noqa: BLE001 — treat as "not offered"
                ok = False
        layer._cb_l2_ok = bool(ok)
        return layer._cb_l2_ok

    @staticmethod
    def _l2_ext_call(name, *args, default=None):
        """Call an OPTIONAL L2 binding. The window API ships independently of
        the expander, so every one of these is getattr+try guarded: an older
        extension must mean 'no L2 lever', never a crashed serve."""
        try:
            from .cuda_ext import get_ext
            ext = get_ext()
            fn = getattr(ext, name, None) if ext is not None else None
            if fn is None:
                return default
            return fn(*args)
        except Exception:  # noqa: BLE001 — an absent/failing knob is not fatal
            return default

    def _l2_plan(self, layer):
        """The rotating-pair plan for this layer, cached on it.

        The cap is DERIVED (``moe_l2.cb_l2_cap_bytes``) from what the device
        reports for persisting L2, halved because the pinned window must cover
        BOTH halves of the arena; ``torch.cuda``'s L2 size is the fallback when
        the extension predates the query. ``PRISMAQUANT_CB_L2_WINDOW_MB``
        overrides the per-half cap for bisection only. The expert tile is sized
        on the LARGER projection (w13's ``2*inter*hidden``), because one arena
        serves both stages.
        """
        plan = getattr(layer, "_cb_l2_plan", "unset")
        if plan != "unset":
            return plan
        persist = self._l2_ext_call("l2_persisting_max_bytes", default=None)
        if not persist:
            try:
                props = torch.cuda.get_device_properties(
                    layer.w13_cb_qweight.device)
                persist = int(getattr(props, "L2_cache_size", 0) or 0)
            except Exception:  # noqa: BLE001
                persist = 0
        cap = cb_l2_cap_bytes(
            persist,
            env_mb=os.environ.get("PRISMAQUANT_CB_L2_WINDOW_MB"),
            max_window_bytes=self._l2_ext_call("l2_max_window_bytes",
                                               default=None))
        force = os.environ.get("PRISMAQUANT_CB_L2_GROUP")
        expert_bytes = max(2 * layer._cb_inter * layer._cb_hidden,
                           layer._cb_hidden * layer._cb_inter)   # e4m3 = 1 B
        plan = cb_l2_plan(layer._cb_E, expert_bytes, cap,
                          force_group=int(force) if force else None)
        layer._cb_l2_plan = plan
        return plan

    def _l2_arena(self, layer, plan):
        """Allocate the arena ONCE per layer and cache it on the layer.

        ONE contiguous RAW-BYTE (uint8) block — the expander's ``out`` contract
        — reinterpreted as e4m3 per unit; the two halves are slices of it, so a
        SINGLE
        pinning window covers both (a persisting window is one address range per
        stream — two separate allocations could not both be covered). It is
        never freed: re-allocating per forward would move the address out from
        under the pinned window and would also churn the caching allocator
        across streams.
        """
        buf = getattr(layer, "_cb_l2_scratch", None)
        if buf is None or buf.numel() < plan.arena_bytes:
            buf = torch.empty(plan.arena_bytes, dtype=torch.uint8,
                              device=layer.w13_cb_qweight.device)
            layer._cb_l2_scratch = buf
        return buf

    def _l2_stream(self, main):
        """A side stream for the DIAGNOSTIC overlapped variant, cached per
        ``(device, current stream)``.

        The round-4 original cached ONE ``torch.cuda.Stream()`` on ``self`` for
        the life of the process. That handle then got reused under whatever
        stream happened to be current at the next call — and vLLM serves on its
        own non-default stream, not the default stream every synthetic test ran
        on. Keying the cache on the CURRENT stream makes a handle usable only in
        the exact context it was created for, so a stream captured under one
        serving context can never be replayed against another. (Creating one per
        forward would also be correct, but a CUDA stream create/destroy per
        layer per forward is itself a heavyweight driver call; the keyed cache
        keeps the steady state free.)
        """
        cache = getattr(self, "_l2_side_streams", None)
        if cache is None:
            cache = self._l2_side_streams = {}
        key = (main.device, main.cuda_stream)
        s = cache.get(key)
        if s is None:
            s = cache[key] = torch.cuda.Stream(device=main.device)
        return s

    def _l2_pin(self, arena, nbytes, streams) -> bool:
        """Claim the persisting-access window on every stream that touches the
        arena. The window is a per-STREAM attribute, so a stream whose accesses
        are not covered streams its traffic past L2 — exactly the traffic this
        round exists to keep resident.

        NOT called per forward — see ``_l2_pin_window`` for the lifecycle and
        ``moe_l2.cb_l2_pin_action`` for why (these calls are device-wide and
        implicitly synchronizing)."""
        ok = True
        for s in streams:
            with torch.cuda.stream(s):
                ok = bool(self._l2_ext_call(
                    "l2_pin_region", arena, int(nbytes), default=False)) and ok
        return ok

    def _l2_unpin(self, streams) -> None:
        """Release the window on the given streams."""
        for s in streams:
            with torch.cuda.stream(s):
                self._l2_ext_call("l2_unpin")

    def _l2_pin_window(self, layer, arena, nbytes, streams) -> bool:
        """Idempotent pin: do the device-wide work only when the
        ``(arena address, streams)`` key actually changes.

        The decision itself is ``moe_l2.cb_l2_pin_action`` (pure, CPU-tested);
        this method is only its CUDA effect plus the per-layer memo.

        LIFETIME SPLIT (the correction to round 5's over-fix). The DEVICE-WIDE
        carve-out reservation is grow-only and effectively once per process —
        re-issuing it per layer per forward is synchronizing and drove a live
        serve's throughput to zero. The PER-STREAM window is a different animal:
        cheap to set, but it must be cleared before we hand the stream back
        (``_l2_reset_window``, called from the caller's ``finally``). Round 5
        removed BOTH, which left our window attached to vLLM's serving stream
        for the life of the process — every later kernel on that stream then ran
        pointed at a foreign address range, which is the leading suspect for the
        third live wedge.
        """
        key = (int(arena.data_ptr()), int(nbytes),
               tuple(int(s.cuda_stream) for s in streams))
        cur = getattr(layer, "_cb_l2_pinned_key", None)
        unpin_first, pin = cb_l2_pin_action(cur, key)
        if unpin_first:
            # The old key's streams are the ones recorded with it; releasing on
            # the CURRENT streams would leave a stale window on a stream we no
            # longer touch, so the streams travel with the memo.
            self._l2_unpin(getattr(layer, "_cb_l2_pinned_streams", streams))
            layer._cb_l2_pinned_key = None
        if not pin:
            return cur is not None
        ok = self._l2_pin(arena, nbytes, streams)
        if ok:
            layer._cb_l2_pinned_key = key
            layer._cb_l2_pinned_streams = tuple(streams)
        return ok

    def _l2_reset_window(self, streams) -> None:
        """Clear the per-stream access-policy window on every stream we set it
        on. Cheap (a stream attribute, no device-wide call), so it belongs in a
        per-forward ``finally``: the invariant is that vLLM's serving stream
        never carries our window outside our own forward. The device-wide
        carve-out reservation deliberately survives — see _l2_pin_window."""
        for s in streams:
            with torch.cuda.stream(s):
                self._l2_ext_call("l2_reset_window")

    def _apply_prefill_l2_pipeline(self, layer, x, topk_weights, topk_ids,
                                   act):
        """ROUND 4: decode into an L2-PINNED rotating scratch pair
        (``PRISMAQUANT_CB_PREFILL=l2_pipeline``). Returns ``None`` on any
        constraint miss so the caller falls through to 'stock'.

        WHAT IT REMOVES. Rounds 1-3 attacked launch count and tile redundancy.
        What is left on a large-expert MoE is the decoded-weight ROUND TRIP: the
        stock path writes each expert's ``N_e x K`` e4m3 tile to HBM and the very
        next kernel reads it back (~17 ms of ~42 ms/layer). Here the decode
        writes into one half of a small, address-stable arena held in L2 by a
        persisting-access window, and the GEMM reads that half back — ideally
        out of L2. The bytes are the SAME bytes (same expander), so this is a
        placement change, not a numerics change.

        UNIT SEQUENCE. The work is a flat list of DECODE UNITS, two per live
        expert group: ``[(g0,w13), (g0,w2), (g1,w13), (g1,w2), ...]``. Unit ``u``
        decodes into ``arena[u % 2]``, so the alternation is uniform across the
        stage boundary. Groups no token routed to are dropped from the list using
        only the E+1 offsets the routing already fetched
        (``cb_l2_live_groups``), so an empty group costs neither a launch nor
        window residency.

        SERIAL IS THE DEFAULT (and the only path a serve takes). Decode unit
        ``u`` into its half, then GEMM from that half — all on the CURRENT
        stream, with no side stream, no events and no ``wait_stream``. Two
        reasons, one structural and one measured:

          * STRUCTURAL. vLLM serves on its own non-default stream and may run a
            small batch inside a CUDA-graph capture. Cross-stream event waits
            against a stream that is not part of the capture are illegal there
            and HANG rather than error — which is exactly what the overlapped
            variant did on its first qualifying prefill (throughput -> 0, request
            stuck Running, no OOM, no watchdog line). Every synthetic test ran on
            the DEFAULT stream, which is why the suite was green.
          * MEASURED. There is no overlap win to give up. ``gridbook/csrc/cb_fused_gemm.cu``
            records chunked-expand + GEMM overlap at 0.74-0.79x of SERIAL speed
            on this part: unified memory at ~273 GB/s leaves the expander no
            spare bandwidth for the GEMM to hide behind, so the two streams
            contend instead of overlapping.

        The rotating pair is KEPT in the serial path: it is what the arena
        sizing and the single pinned window are built around, and one stream
        already orders decode-before-GEMM and GEMM-before-overwrite, so no
        cross-buffer synchronization is needed at all. Buffer reuse is safe by
        program order.

        The overlapped variant survives behind ``PRISMAQUANT_CB_L2_OVERLAP=1``
        (``_l2_units_overlapped``) for diagnosis only.

        NUMERICS. Weight bytes are bit-identical to stock (same expander kernel,
        same packed rows, only the destination differs). Activations use the
        stock path's own per-token fp8 dynamic QDQ on the input AND the
        intermediate (a per-row op, so gather-after-quant == quant-after-gather
        bit-exactly). ``cutlass_scaled_mm`` applies per-token a_scales and
        per-channel b_scales in its fp32 EVT epilogue and rounds ONCE to bf16 —
        the promoted rounding order, the same one the dense fp8 path ships. Only
        the GEMM accumulation and the cross-expert combine reassociate
        (REASSOCIATION-CLASS, the suite's 2e-2 contract).

        FALL-THROUGH. ``None`` (-> stock) when the format/build gate misses, or
        when the LARGEST expert tile exceeds the derived per-half cap: no
        rotation of that pair could keep a tile resident, so paying the
        pipeline's bookkeeping would be dishonest. If ``l2_pin_region`` reports
        the window unavailable we still RUN (rather than fall through): the
        rotating pair is a real structural change on its own — a bounded,
        reused transient in place of stock's per-chunk allocation — and the R3
        tuner is what decides whether it wins. Falling through there would hide
        a candidate the tuner exists to judge. (It is NOT overlap that carries
        this path: the default is serial by construction, and overlap measured
        as a LOSS on this part — see the serial-default note above.)
        """
        if x.shape[0] < cb_l2_min_m(
                os.environ.get("PRISMAQUANT_CB_L2_MIN_M")):
            # TINY-M FLOOR. Below the mid-M boundary the per-expert pipeline is
            # pure bookkeeping: 2 launches per live group plus a pinned window,
            # to move a handful of rows. The R3 tuner would never propose R4
            # here (its own guard is M>=1024), but the DIRECT env mode bypasses
            # the tuner, so the floor has to live in the path.
            return None
        if not self._l2_ok(layer):
            return None
        plan = self._l2_plan(layer)
        if plan is None:                     # tile > window cap, or no window
            return None

        from vllm import _custom_ops as vllm_ops
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )
        from . import ops as pq_ops

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden
        inter = layer._cb_inter
        N1 = 2 * inter
        d = N1 // 2
        lut = self._stock_cb_flat_fp8(layer)
        fp8_dtype = current_platform.fp8_dtype()

        # ---- routing: R1's construction verbatim (ONE host sync) -----------
        pair_expert = topk_ids.reshape(-1).to(torch.long)
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)
        order = torch.argsort(pair_expert, stable=True)          # STABLE
        ptok_sorted = pair_token[order]
        pw_sorted = topk_weights.reshape(-1)[order]
        counts = torch.bincount(pair_expert, minlength=E)
        bounds = torch.cat([counts.new_zeros(1),
                            torch.cumsum(counts, 0)]).tolist()   # THE one sync

        live = cb_l2_live_groups(plan.groups, bounds)
        out = torch.zeros_like(x)
        if not live:
            return out

        # ---- input activation QDQ, ONCE over all token rows ----------------
        a1, a1s = moe_kernel_quantize_input(
            x, None, fp8_dtype, per_act_token_quant=True)
        a1s = a1s.reshape(-1, 1).to(torch.float32)

        arena = self._l2_arena(layer, plan)
        halves = [arena[:plan.buffer_bytes],
                  arena[plan.buffer_bytes:2 * plan.buffer_bytes]]
        main = torch.cuda.current_stream()
        overlap = os.environ.get("PRISMAQUANT_CB_L2_OVERLAP") == "1"
        if overlap and torch.cuda.is_current_stream_capturing():
            # Cross-stream waits against a non-captured stream are not legal
            # inside a graph capture; the driver hangs instead of erroring.
            # Refuse rather than risk it — stock is capture-clean.
            return None
        side = self._l2_stream(main) if overlap else None
        streams = (main, side) if overlap else (main,)
        # Row-offset vector for the expander: zeros, one per decoded row, sized
        # once for the worst unit and sliced — allocating it per unit would put
        # a fresh H2D-shaped allocation on the hot path for a constant.
        row0 = getattr(layer, "_cb_l2_row0", None)
        max_rows = plan.group_size * max(N1, Kh)
        if row0 is None or row0.numel() < max_rows:
            row0 = torch.zeros(max_rows, dtype=torch.int32, device=dev)
            layer._cb_l2_row0 = row0

        units = []
        for e0, e1, p0, p1 in live:
            units.append(("w13", e0, e1, p0, p1))
            units.append(("w2", e0, e1, p0, p1))

        def _decode(unit, buf_idx):
            which, e0, e1, _p0, _p1 = unit
            n_e = e1 - e0
            out_f = N1 if which == "w13" else Kh
            in_f = Kh if which == "w13" else inter
            rows = n_e * out_f
            packed = getattr(layer, f"{which}_cb_qweight")[e0:e1]
            pq_ops.cb_expand_fp8_into(
                halves[buf_idx], packed.reshape(rows, -1), lut,
                row0[:rows], rows, in_f, self.k, self.n_sub, self.type_size)

        def _weights(unit, buf_idx):
            which, e0, e1, _p0, _p1 = unit
            n_e = e1 - e0
            out_f = N1 if which == "w13" else Kh
            in_f = Kh if which == "w13" else inter
            # Raw bytes ARE the e4m3 tile (the expander writes e4m3 bit
            # patterns); reinterpret, never convert.
            return (halves[buf_idx][:n_e * out_f * in_f]
                    .view(torch.float8_e4m3fn).view(n_e, out_f, in_f))

        # The intermediate activations of the w13 unit, consumed by the w2 unit
        # of the SAME group. A one-slot holder rather than a closure-local, so
        # both drivers below share one definition of the GEMM work.
        carry = [None]

        def _gemm_unit(unit, buf_idx):
            which, e0, e1, _p0, _p1 = unit
            W = _weights(unit, buf_idx)
            if which == "w13":
                group_a2 = []
                for e in range(e0, e1):
                    q0, q1 = bounds[e], bounds[e + 1]
                    if q1 == q0:
                        group_a2.append(None)
                        continue
                    rows = ptok_sorted[q0:q1]
                    ae = a1.index_select(0, rows).contiguous()
                    ase = a1s.index_select(0, rows).contiguous()
                    gate_up = vllm_ops.cutlass_scaled_mm(
                        ae, W[e - e0].t(), ase,
                        layer.w13_weight_scale[e].reshape(1, N1).to(
                            torch.float32),
                        torch.bfloat16)
                    a = torch.empty((gate_up.shape[0], d),
                                    dtype=gate_up.dtype, device=dev)
                    apply_moe_activation(act, a, gate_up)
                    del gate_up
                    a2, a2s = moe_kernel_quantize_input(
                        a, None, fp8_dtype, per_act_token_quant=True)
                    group_a2.append((a2.contiguous(),
                                     a2s.reshape(-1, 1).to(torch.float32)))
                carry[0] = group_a2
            else:
                for e in range(e0, e1):
                    q0, q1 = bounds[e], bounds[e + 1]
                    got = carry[0][e - e0]
                    if got is None:
                        continue
                    a2, a2s = got
                    y = vllm_ops.cutlass_scaled_mm(
                        a2, W[e - e0].t(), a2s,
                        layer.w2_weight_scale[e].reshape(1, Kh).to(
                            torch.float32),
                        torch.bfloat16)
                    rows = ptok_sorted[q0:q1]
                    y = y * pw_sorted[q0:q1, None].to(y.dtype)
                    out.index_add_(0, rows, y.to(out.dtype))
                carry[0] = None

        # PIN. Idempotent and off the per-forward path: the reservation/reset
        # pair is device-wide and implicitly synchronizing, so re-issuing it per
        # layer per forward is a throughput sink by itself. Skipped entirely
        # under graph capture, where a device-limit call is not legal; the rest
        # of the serial path is capture-clean (one stream, no events, no host
        # sync beyond the routing sync the caller already performs).
        if not torch.cuda.is_current_stream_capturing():
            self._l2_pin_window(layer, arena, plan.arena_bytes, streams)

        try:
            if overlap:
                self._l2_units_overlapped(units, _decode, _gemm_unit, main, side)
            else:
                # SERIAL (default). One stream orders decode-before-GEMM and the
                # next decode after the GEMMs that read that half, so the
                # rotation needs no events at all. The rotation is kept because
                # the arena sizing and the single pinned window are defined over
                # the pair.
                for u, unit in enumerate(units):
                    i = u % 2
                    _decode(unit, i)
                    _gemm_unit(unit, i)
        finally:
            # Hand the stream back clean. Cheap (stream attribute only); the
            # device-wide reservation is NOT touched here.
            self._l2_reset_window(streams)
        return out

    @staticmethod
    def _l2_units_overlapped(units, decode, gemm_unit, main, side):
        """DIAGNOSTIC-ONLY overlapped driver (``PRISMAQUANT_CB_L2_OVERLAP=1``).

        KNOWN TO HANG A LIVE SERVE. It is retained unchanged because its hazard
        logic is correct and worth keeping for later analysis; what is not safe
        is the CONTEXT — vLLM's non-default current stream and possible graph
        capture, where cross-stream waits against a non-captured stream hang.
        Do not put this on a serving path.

        EVENT STRUCTURE (a missed dependency is a silent WRONG ANSWER, not a
        crash, so it is spelled out). Two event pairs, indexed by buffer:

          * ``dec_done[i]`` recorded on the SIDE stream after the decode that
            fills buffer ``i``; the MAIN stream waits on it before any GEMM that
            reads buffer ``i`` — RAW (produce-then-consume).
          * ``gemm_done[i]`` recorded on the MAIN stream after the LAST GEMM
            that reads buffer ``i``; the SIDE stream waits on it before the next
            decode that overwrites buffer ``i`` — WAR (the reuse hazard a naive
            double-buffer forgets).

        The loop issues the decode of unit ``u+1`` (buffer ``j=(u+1)%2``) BEFORE
        the GEMMs of unit ``u`` (buffer ``i=u%2``) — that is the whole overlap.
        Buffer ``j`` was last read by unit ``u-1``, so the side stream first
        waits ``gemm_done[j]`` (skipped at ``u==0``). ``side.wait_stream(main)``
        at entry orders the first decode after the prologue and after any prior
        use of the arena; ``main.wait_stream(side)`` at exit leaves nothing
        outstanding.
        """
        dec_done = [torch.cuda.Event() for _ in range(2)]
        gemm_done = [torch.cuda.Event() for _ in range(2)]
        side.wait_stream(main)                   # arena free; prologue visible

        def _decode_on(u, b):
            with torch.cuda.stream(side):
                decode(units[u], b)
            dec_done[b].record(side)

        _decode_on(0, 0)
        for u, unit in enumerate(units):
            i = u % 2
            main.wait_event(dec_done[i])         # RAW: decode -> GEMM
            if u + 1 < len(units):
                j = (u + 1) % 2
                if u >= 1:
                    # WAR: buffer j was last READ by unit u-1.
                    side.wait_event(gemm_done[j])
                _decode_on(u + 1, j)
            gemm_unit(unit, i)
            gemm_done[i].record(main)            # WAR release for buffer i
        main.wait_stream(side)                   # nothing outstanding on side

    # -- prefill: MEASURED per-layer path selection -------------------------
    def _prefill_candidates(self, layer, x, topk_weights, topk_ids, act):
        """The prefill paths worth measuring for THIS layer, as
        ``[(name, thunk)]`` with 'stock' first. grouped_fused appears once per
        TileM the build actually compiled for this layer's rung; if the grouped
        binding or the rung constraints are unmet the list is just 'stock', and
        auto degenerates to the default with one wasted timing call."""
        cands = [(_AUTO_STOCK, lambda: self._apply_prefill_stock(
            layer, x, topk_weights, topk_ids, act))]
        if self._gf2_ok(layer):
            for tm in self._gf2_tile_sizes(layer):
                cands.append((
                    f"grouped_fused:tile_m={tm}",
                    # Bind tm per iteration — a bare closure would capture the
                    # loop variable and time the last tile N times.
                    (lambda t: lambda: self._apply_prefill_grouped_fused_v2(
                        layer, x, topk_weights, topk_ids, act, tile_m=t))(tm)))
        # Round 4 is just another candidate: it returns None (-> disqualified,
        # like every other gate here) when the build/format/window cap misses,
        # so no promotion decision is taken in code — the tuner MEASURES it
        # against stock on this layer's own inputs.
        #
        # ...but it is OPT-IN inside auto (PRISMAQUANT_CB_L2_AUTOTUNE=1) until
        # its GPU gates have actually RUN. 'auto' is now the shipping default,
        # so anything in this list is reachable by a default serve; l2_pipeline
        # is the only candidate here whose parity, ragged/zero-row and
        # buffer-rotation RACE tests have never executed (they are skip-only
        # while the box is under the wedge quarantine). An unmeasured path that
        # the default can silently select is exactly the promotion-without-
        # evidence this repo forbids. PRISMAQUANT_CB_PREFILL=l2_pipeline still
        # selects it directly, which is how the GPU session measures it; drop
        # this gate once those tests are green on real hardware.
        if (os.environ.get("PRISMAQUANT_CB_L2_AUTOTUNE") == "1"
                and self._l2_ok(layer) and self._l2_plan(layer) is not None):
            cands.append((L2_PIPELINE, lambda: self._apply_prefill_l2_pipeline(
                layer, x, topk_weights, topk_ids, act)))
        return cands

    def _apply_prefill_auto(self, layer, x, topk_weights, topk_ids, act):
        """MEASURED per-layer prefill selection (``PRISMAQUANT_CB_PREFILL=auto``
        — the fp8-CB DEFAULT since the two-model gate cleared it in 3062fbf:
        35B 4,405 tok/s vs stock 3,932, Laguna-class 2,063 vs stock 1,821).
        Thin adapter: the policy —
        threshold, per-layer caching, determinism, forcing — lives in
        ``moe_autotune.cb_prefill_auto``, which is torch-only so it is testable
        without vLLM/CUDA. This method only supplies the candidates."""
        return cb_prefill_auto(
            layer, x.shape[0],
            lambda: self._prefill_candidates(
                layer, x, topk_weights, topk_ids, act),
            lambda: self._apply_prefill_stock(
                layer, x, topk_weights, topk_ids, act),
            min_m=int(os.environ.get("PRISMAQUANT_CB_AUTOTUNE_MIN_M")
                      or "1024"),
            forced=os.environ.get("PRISMAQUANT_CB_PREFILL_AUTO_FORCE") or None,
            log=self._log_prefill_choice)

    def _log_prefill_choice(self, best, timings, forced=False):
        """One line per layer, to stderr like every other gate on this path.
        This is the evidence trail: a serving run must be able to show WHY it
        picked what it picked, with the measurements that decided it."""
        detail = " ".join(f"{n}={timings[n]:.3f}ms" for n in sorted(timings))
        print(f"[prismaquant-cb] prefill auto {self.prefix}: "
              f"{'forced' if forced else 'chose'} {best}"
              + (f" | {detail}" if detail else ""),
              file=sys.stderr, flush=True)

    # -- prefill: per-expert loop (bisection reference) ---------------------
    def _apply_prefill_loop(self, layer, x, topk_weights, topk_ids, act):
        """Original per-expert prefill loop (PRISMAQUANT_CB_PREFILL=loop): the
        validated-numerics reference the batched path must match. Per hit expert,
        QDQ that expert's rows, decode its w13/w2 to a bounded bf16 transient
        (_decode_expert, INV-1 one expert live), two F.linear (bf16 MMA), and a
        router-weighted index_add_ combine in expert-ascending order."""
        out = torch.zeros_like(x)
        # ONE host sync for the hit-expert list (the pre-fix per-expert
        # `bool(sel.any())` cost E syncs per layer per forward).
        hit = torch.bincount(
            topk_ids.reshape(-1), minlength=layer._cb_E) > 0
        hit_experts = hit.nonzero(as_tuple=True)[0].tolist()   # one sync
        for e in hit_experts:
            sel = (topk_ids == e)
            tok_idx, slot = torch.where(sel)                   # tokens -> expert e
            xe = codec.fp4_group16_act_qdq(x[tok_idx]) if self.is_fp4 \
                else codec.fp8_dynamic_act_qdq(x[tok_idx])
            xe = xe.to(torch.bfloat16)
            W13 = self._decode_expert(layer, "w13", e)         # (2*inter, hidden)
            gate_up = torch.nn.functional.linear(xe, W13)      # (n_e, 2*inter)
            del W13
            d = gate_up.shape[-1] // 2
            a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                            device=gate_up.device)
            apply_moe_activation(act, a, gate_up)              # silu(gate)*up
            aq = (codec.fp4_group16_act_qdq(a) if self.is_fp4
                  else codec.fp8_dynamic_act_qdq(a)).to(torch.bfloat16)
            W2 = self._decode_expert(layer, "w2", e)           # (hidden, inter)
            oe = torch.nn.functional.linear(aq, W2)            # (n_e, hidden)
            del W2
            oe = oe * topk_weights[tok_idx, slot][:, None].to(oe.dtype)
            out.index_add_(0, tok_idx, oe.to(out.dtype))
        return out

    # -- prefill: batched-expert path (default) -----------------------------
    def _apply_prefill_batched(self, layer, x, topk_weights, topk_ids, act, *,
                               use_grouped_mm=None, chunk=None):
        """Batched prefill (PRISMAQUANT_CB_PREFILL=batched). Replaces the
        per-expert loop's E Triton expands + E QDQ passes with ONE act-QDQ over
        all tokens and ONE expand per projection per expert-CHUNK, then a
        grouped/segmented bf16 GEMM over the per-expert token groups.

        INV-1: the transient weight tile is bounded to ONE chunk of experts
        (PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK, default 64 — ~1.6 GB for Hy3's
        3072×4096 w13, vs ~7.2 GB for all 192 at once), expanded right before
        its GEMM and freed right after; w13 is freed before w2 is expanded, so
        peak is one chunk's LARGER projection, not their sum.

        Numerics vs _apply_prefill_loop: weights are bit-identical (same
        expander, same per-(expert,out) scale) and the per-token QDQ is
        bit-identical because BOTH fp8-dynamic and fp4-group16 activation QDQ are
        PER-TOKEN-ROW operations — codec.fp8_dynamic_act_qdq /
        fp4_group16_act_qdq reshape to rows and quantise each row from its own
        amax with no cross-row coupling, so QDQ(x)[sel] == QDQ(x[sel])
        row-for-row (test_qdq_once_equals_per_selection). One act-QDQ over all
        tokens therefore reproduces the loop's per-expert QDQ(x[tok_idx]) on the
        same rows. Only the GEMM accumulation and cross-expert combine
        reassociate (REASSOCIATION-CLASS)."""
        if use_grouped_mm is None:
            use_grouped_mm = os.environ.get(
                "PRISMAQUANT_CB_PREFILL_GROUPED_MM", "0") == "1"
        if chunk is None:
            chunk = int(os.environ.get(
                "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK") or "64")
        chunk = max(1, chunk)
        T = x.shape[0]
        topk = topk_ids.shape[-1]
        dev = x.device
        out = torch.zeros_like(x)

        # ONE activation QDQ over ALL token rows (per-row op — see docstring).
        xq = (codec.fp4_group16_act_qdq(x) if self.is_fp4
              else codec.fp8_dynamic_act_qdq(x)).to(torch.bfloat16)

        # (token, expert, weight) pairs from the full topk grid, sorted
        # expert-ascending and STABLE so within-expert row order stays
        # token-major — exactly the loop's torch.where(topk_ids==e) order (the
        # per-segment GEMM then bit-matches the loop; only the combine
        # reassociates). pair p = t*topk + j -> expert topk_ids[t,j].
        pair_expert = topk_ids.reshape(-1)                     # [P] = T*topk
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(topk)                            # [P]
        pair_weight = topk_weights.reshape(-1)                 # [P]
        order = torch.argsort(pair_expert, stable=True)
        ptok_sorted = pair_token[order]
        pw_sorted = pair_weight[order]

        counts = torch.bincount(pair_expert, minlength=layer._cb_E)   # [E]
        hit_experts_t = counts.nonzero(as_tuple=True)[0]              # ascending
        hit_offsets_t = torch.cat([
            counts.new_zeros(1),
            torch.cumsum(counts[hit_experts_t], 0)])                 # [n_hit+1]
        # Two O(1) host syncs (NOT O(E)): the hit-expert ids and their
        # sorted-array segment boundaries. Everything above stays GPU-resident.
        hit_experts = hit_experts_t.tolist()
        hit_offsets = hit_offsets_t.tolist()
        n_hit = len(hit_experts)

        for c0 in range(0, n_hit, chunk):
            c1 = min(n_hit, c0 + chunk)
            these = hit_experts[c0:c1]                         # python ids
            p0, p1 = hit_offsets[c0], hit_offsets[c1]
            if p1 == p0:
                continue
            # local per-expert segment bounds within this chunk's [p0, p1).
            bounds = [hit_offsets[c0 + i] - p0 for i in range(c1 - c0 + 1)]
            xrows = xq[ptok_sorted[p0:p1]]                     # [Pc, hidden]

            # stage 1: gate_up = xrows @ W13[e]^T (grouped).
            W13 = self._expand_expert_stack(layer, "w13", these)  # [C,2i,hidden]
            gate_up = self._grouped_gemm(xrows, W13, bounds, use_grouped_mm)
            del W13, xrows
            d = gate_up.shape[-1] // 2
            a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                            device=dev)
            apply_moe_activation(act, a, gate_up)             # silu(gate)*up
            del gate_up
            aq = (codec.fp4_group16_act_qdq(a) if self.is_fp4
                  else codec.fp8_dynamic_act_qdq(a)).to(torch.bfloat16)
            del a

            # stage 2: y = aq @ W2[e]^T (grouped).
            W2 = self._expand_expert_stack(layer, "w2", these)    # [C,hidden,i]
            y = self._grouped_gemm(aq, W2, bounds, use_grouped_mm)  # [Pc,hidden]
            del W2, aq

            y = y * pw_sorted[p0:p1, None].to(y.dtype)        # router weight
            out.index_add_(0, ptok_sorted[p0:p1], y.to(out.dtype))
            del y
        return out

    def _expand_expert_stack(self, layer, which: str, experts) -> torch.Tensor:
        """Decode a CHUNK of experts' CB weights to a stacked bf16 tile
        ``[C, out, in]`` in ONE Triton launch: the dense expander runs over the
        reshaped ``[(C*out), row_bytes]`` view. Every expert-row is an
        independent decode keyed only by its own packed bytes + the shared
        per-layer codebook at offset 0 — exactly what ``_decode_expert`` does per
        expert — so the stacked tile is BIT-IDENTICAL to stacking
        ``_decode_expert`` over ``experts`` (test_batched_expand_matches_decode).
        INV-1: bounded to one chunk; the caller frees it after the GEMM."""
        packed = getattr(layer, f"{which}_cb_qweight")[experts]   # [C,out,bytes]
        C, out_f = packed.shape[0], packed.shape[1]
        in_f = layer._cb_hidden if which == "w13" else layer._cb_inter
        qwp = codec.pad_qweight(packed.reshape(C * out_f, -1).contiguous())
        row0 = torch.zeros(C * out_f, dtype=torch.int32, device=packed.device)
        if self.is_fp4:                                        # fp4 two-tier v2
            W = expand_fp4_v2_to_weight(
                qwp, layer._cb_flat, row0, layer._cb_compose,
                C * out_f, in_f, self.k, self.n_sub, self.type_size)
        else:                                                  # fp8
            val = expand_cb_to_value(qwp, layer._cb_flat, row0, C * out_f, in_f,
                                     self.k, self.n_sub, self.type_size,
                                     is_fp4=False)
            ws = getattr(layer, f"{which}_weight_scale")[experts].reshape(
                C * out_f).to(torch.float32)
            W = (val.float() * ws[:, None]).to(torch.bfloat16)
        return W.view(C, out_f, in_f)

    def _grouped_gemm(self, x, w_stack, bounds, use_grouped_mm):
        """Grouped bf16 GEMM: ``out[bounds[g]:bounds[g+1]] = x[seg] @ w_stack[g]^T``.

        Default = segmented — one ``F.linear`` per group over the pre-expanded
        chunk weights, BIT-IDENTICAL per segment to the loop's per-expert
        ``F.linear(xe, W)`` (same cuBLAS call, same stable row order). Opt-in
        ``PRISMAQUANT_CB_PREFILL_GROUPED_MM=1`` collapses the groups into ONE
        ``torch._grouped_mm`` launch (reassociation-class vs the loop), degrading
        to segmented if _grouped_mm is absent or rejects the ragged offsets /
        transposed-B layout on this build."""
        P = x.shape[0]
        out_f = w_stack.shape[1]
        G = w_stack.shape[0]
        if use_grouped_mm and _grouped_mm_available():
            try:
                offs = torch.tensor(bounds[1:], dtype=torch.int32,
                                    device=x.device)
                # 2d×3d: mat_a [P,K] × mat_b [G,K,N] (w_stack^T) -> [P,N]. Cast
                # to x.dtype so gate_up/y stay bf16 like the loop's F.linear
                # (grouped_mm may accumulate out to fp32).
                return torch._grouped_mm(
                    x, w_stack.transpose(1, 2), offs=offs).to(x.dtype)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash serve
                _disable_grouped_mm(exc)
        y = torch.empty((P, out_f), dtype=x.dtype, device=x.device)
        for g in range(G):
            s0, s1 = bounds[g], bounds[g + 1]
            if s1 > s0:                                        # skip 0-token seg
                y[s0:s1] = torch.nn.functional.linear(x[s0:s1], w_stack[g])
        return y

    # -- prefill: capture-safe stock-kernel path (task 15) ------------------
    def _apply_prefill_stock(self, layer, x, topk_weights, topk_ids, act, *,
                             chunk=None):
        """Capture-safe batched prefill (PRISMAQUANT_CB_PREFILL=stock): transient
        expansion of each expert-CHUNK into a HARDWARE format consumed by vLLM's
        OWN fused-MoE grouped Triton kernel with DEVICE-SIDE routing. The
        successor to _apply_prefill_batched, which cannot be CUDA-graph captured
        (its hit-expert list + segment bounds are read to host via two .tolist()
        syncs and drive python control flow). This path is the prerequisite for
        cudagraph_mode=FULL (drafter capture + captured decode target).

        CAPTURE-SAFETY CONTRACT (the hard requirement beyond correctness):
          * NO host reads of device data — no .tolist / .item / .cpu / .nonzero
            -> python anywhere in the path. Routing is resolved entirely on-device
            by vLLM's moe_align_block_size (a C++/Triton op vLLM itself captures in
            its EP decode graphs); num_tokens_post_padded stays a device scalar
            consumed as a kernel arg, never read back.
          * FIXED python trip count — the chunk loop runs ceil(E/chunk) times over
            ALL E experts (never a hit-filtered subset); an empty chunk costs one
            masked pass. Every loop bound comes from a shape or config int, never
            from a tensor value, so the unrolled capture is identical every replay.
          * Per-chunk EP shard — a chunk of ``chunk`` experts is selected by an
            expert_map (global->local, -1 outside; built from python ints), and
            moe_align_block_size(..., ignore_invalid_experts=True) produces LOCAL
            expert_ids for exactly that shard (out-of-chunk pairs excluded). This
            is vLLM's own expert-parallel mechanism, reused to bound the transient
            weight tile to ONE chunk (INV-1) instead of all E experts at once.

        TWO-STAGE STRUCTURE (this build exposes only the MONOLITHIC fused_experts,
        which does its own intermediate quant with no seam for ours — so we call
        the underlying per-projection kernel dispatch_fused_moe_kernel TWICE with
        OUR activation QDQ between, mirroring fused_experts_impl exactly): the same
        moe_align routing tensors serve both projections (stage 1: A=[M,hidden],
        top_k=T; stage 2: A=[M*T,inter], top_k=1 — the kernel indexes A by
        offs_token//top_k, so one routing covers both).

        ACTIVATION QDQ — part of the validated numerics, never dropped:
          * fp8-CB: the stock W8A8 kernel's built-in per-token fp8-dynamic activation
            quant (moe_kernel_quantize_input) is bit-equivalent to
            codec.fp8_dynamic_act_qdq (test_stock_fp8_quant_matches_codec) and IS
            the transient the dense linear.py fp8 prefill already trusts, so it runs
            the input AND intermediate quant. Weights expand DIRECTLY to fp8 bytes
            (expand_cb_to_fp8, the dense fp8-direct transient over the [C*out,bytes]
            stack view) + the per-(expert,out) weight_scale — a plain per-channel
            fp8 checkpoint, native tensor cores.
          * fp4-CB-v2: no faithful packed-NVFP4 exists (a CB codebook value is not
            on the E2M1 grid, so re-quantising to NVFP4 would double-quantise), and
            this build has no split W4A4 grouped MoE we can wedge our group-16 QDQ
            into — so we ship the stated FALLBACK: expand to bf16 (exact CB
            reconstruct, expand_fp4_v2_to_weight) + the stock BF16 grouped kernel,
            with codec.fp4_group16_act_qdq run EXPLICITLY on the input and the
            intermediate (still capture-safe, still native Triton).

        Numerics vs _apply_prefill_loop: weights bit-identical (same expanders,
        fp8-byte gather == the loop's val.to(e4m3)); activation QDQ per-token
        identical (fp8-dynamic proven; fp4-group16 is ours verbatim). Only the GEMM
        accumulation (fp8 fp32-accum vs the loop's bf16 F.linear) and the
        cross-chunk combine reassociate (REASSOCIATION-CLASS), held to the suite's
        2e-2 tolerance by tests/test_moe_stock_prefill.py and gated by the served
        logprob A/B before adoption.

        Memory (INV-1): one projection's chunk tile is live at a time (w13 freed
        before w2 expands); Hy3 (E=192, 2i=3072, hidden=4096) at chunk=16 =>
        16*3072*4096 = 192 MiB fp8 w13 (~2x for the bf16 fp4 tile), vs the crashed
        batched path's ~1.6 GB. The chunk env is PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK
        (default 16)."""
        # vLLM fused-MoE internals imported lazily: the stable public surface is
        # at module top; these are version-fragile, and a build lacking them must
        # fail only this opt-in path, never the module import (CPU codec tests).
        import triton.language as tl
        from vllm.platforms import current_platform
        from vllm.model_executor.layers.fused_moe.config import (
            _get_config_dtype_str,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            dispatch_fused_moe_kernel,
            try_get_optimal_moe_config,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )

        if chunk is None:
            chunk = int(os.environ.get(
                "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK") or "256")
        chunk = max(1, chunk)
        is_fp8 = not self.is_fp4
        E = layer._cb_E
        M = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden                         # hidden (w13 in, w2 out)
        inter = layer._cb_inter                       # moe intermediate
        N1 = 2 * inter                                # w13 out (gate_up)
        d = N1 // 2                                    # silu halves gate_up
        fp8_dtype = current_platform.fp8_dtype()
        compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
                        torch.float32: tl.float32}[x.dtype]

        # Kernel tile config (block sizes) — computed ONCE from shapes/config so
        # BLOCK_SIZE_M is identical across chunks (moe_align + both dispatches
        # must agree). E=chunk here only tunes group_m; correctness-independent.
        cfg_dtype = _get_config_dtype_str(x.dtype, use_fp8_w8a8=is_fp8)
        config = try_get_optimal_moe_config(
            (chunk, N1, Kh), (chunk, Kh, inter), top_k, cfg_dtype, M)
        block_m = config["BLOCK_SIZE_M"]

        topk_ids_i = topk_ids.to(torch.int32)
        topk_weights = topk_weights.contiguous()
        out = torch.zeros_like(x)

        # Input activation quant is x-only -> compute ONCE (only the intermediate
        # a2 is per-chunk). fp8: vLLM per-token fp8 dynamic (== codec, proven);
        # fp4-v2: our group-16 RTN, run explicitly.
        _timing = os.environ.get("PRISMAQUANT_CB_PREFILL_TIMING") == "1"
        if _timing:
            torch.cuda.synchronize()
            _t = {"qdq": 0.0, "align": 0.0, "expand": 0.0, "gemm": 0.0,
                  "act": 0.0, "combine": 0.0}
            import time as _time
            _tw0 = _time.time()
            _t0 = _time.time()
        if is_fp8:
            a1, a1s = moe_kernel_quantize_input(
                x, None, fp8_dtype, per_act_token_quant=True)
        else:
            a1 = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
            a1s = None
        if _timing:
            torch.cuda.synchronize(); _t["qdq"] += _time.time() - _t0

        for c0 in range(0, E, chunk):                 # FIXED trip = ceil(E/chunk)
            c1 = min(E, c0 + chunk)
            expert_map = self._stock_chunk_expert_map(c0, c1, E, dev)
            # Device-side routing for THIS expert shard (local ids, out-of-chunk
            # pairs excluded). Shared verbatim by both projections.
            if _timing: _t0 = _time.time()
            sorted_ids, expert_ids, num_pad = moe_align_block_size(
                topk_ids_i, block_m, E, expert_map, ignore_invalid_experts=True)
            if _timing:
                torch.cuda.synchronize(); _t["align"] += _time.time() - _t0

            # ---- stage 1: gate_up = x @ W13[e]^T -------------------------------
            if _timing: _t0 = _time.time()
            W13 = self._expand_stack_slice(layer, "w13", c0, c1, to_fp8=is_fp8)
            # W2's expand has no dependency on stage 1 — run it on a side
            # stream so it hides under the stage-1 GEMM + activation. The
            # peak transient grows by |W2| (~0.8 GB on Laguna); the serve
            # slack gate is the sizing authority. Prefill is eager
            # (FULL_DECODE_ONLY graphs), so side-stream use is capture-safe.
            # PRISMAQUANT_CB_PREFILL_OVERLAP=0 restores serial (bisection).
            # Measured NULL on 35B-A3B (17 ms/layer, both arms identical,
            # 2026-07-26); stays opt-in until a positive exists at any scale.
            _ovl = (is_fp8 and os.environ.get(
                "PRISMAQUANT_CB_PREFILL_OVERLAP") == "1")
            W2 = None
            if _ovl:
                if not hasattr(self, "_ovl_stream"):
                    self._ovl_stream = torch.cuda.Stream()
                self._ovl_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._ovl_stream):
                    W2 = self._expand_stack_slice(
                        layer, "w2", c0, c1, to_fp8=is_fp8)
            if _timing:
                torch.cuda.synchronize(); _t["expand"] += _time.time() - _t0
            ic1 = torch.empty((M, top_k, N1), dtype=x.dtype, device=dev)
            b1s = layer.w13_weight_scale[c0:c1] if is_fp8 else None    # [nE, 2i]
            if _timing: _t0 = _time.time()
            dispatch_fused_moe_kernel(
                a1, W13, ic1, a1s, b1s, None, topk_weights, sorted_ids,
                expert_ids, num_pad, False, top_k, config,
                compute_type=compute_type, use_fp8_w8a8=is_fp8,
                use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=is_fp8, block_shape=None, B_bias=None)
            if _timing:
                torch.cuda.synchronize(); _t["gemm"] += _time.time() - _t0
            del W13

            # silu(gate)*up over the routed rows (garbage in unrouted rows is
            # never read by stage 2 — same masking as fused_experts_impl).
            ic2 = torch.empty((M * top_k, d), dtype=x.dtype, device=dev)
            if _timing: _t0 = _time.time()
            apply_moe_activation(act, ic2, ic1.view(-1, N1))
            if _timing:
                torch.cuda.synchronize(); _t["act"] += _time.time() - _t0
            del ic1

            # ---- intermediate activation QDQ (part of the numerics) -----------
            if _timing: _t0 = _time.time()
            if is_fp8:
                a2, a2s = moe_kernel_quantize_input(
                    ic2, None, fp8_dtype, per_act_token_quant=True)
                b2s = layer.w2_weight_scale[c0:c1]            # [nE, hidden]
            else:
                a2 = codec.fp4_group16_act_qdq(ic2).to(torch.bfloat16)
                a2s = b2s = None
            if _timing:
                torch.cuda.synchronize(); _t["qdq"] += _time.time() - _t0
            del ic2

            # ---- stage 2: y = a @ W2[e]^T, router-weighted --------------------
            if _timing: _t0 = _time.time()
            if W2 is None:
                W2 = self._expand_stack_slice(
                    layer, "w2", c0, c1, to_fp8=is_fp8)
            else:
                # W2 was expanded on the side stream under stage 1; order
                # stage 2 after it and pin its lifetime to this stream.
                torch.cuda.current_stream().wait_stream(self._ovl_stream)
                W2.record_stream(torch.cuda.current_stream())
            if _timing:
                torch.cuda.synchronize(); _t["expand"] += _time.time() - _t0
            ic3 = torch.empty((M, top_k, Kh), dtype=x.dtype, device=dev)
            ic3.zero_()                                       # out-of-chunk slots
            if _timing: _t0 = _time.time()
            dispatch_fused_moe_kernel(
                a2, W2, ic3, a2s, b2s, None, topk_weights, sorted_ids,
                expert_ids, num_pad, True, 1, config,
                compute_type=compute_type, use_fp8_w8a8=is_fp8,
                use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=is_fp8, block_shape=None, B_bias=None)
            if _timing:
                torch.cuda.synchronize(); _t["gemm"] += _time.time() - _t0
            del W2
            # Combine this shard's top_k contributions and accumulate (the loop's
            # index_add_ over experts, one expert-partition per chunk).
            if _timing: _t0 = _time.time()
            out += ic3.sum(dim=1).to(out.dtype)
            del ic3
            if _timing:
                torch.cuda.synchronize(); _t["combine"] += _time.time() - _t0
        if _timing:
            _tot = _time.time() - _tw0
            _acc = sum(_t.values())
            print(f"[cb-prefill-timing] M={M} total={_tot*1000:.0f}ms "
                  + " ".join(f"{k}={v*1000:.0f}ms" for k, v in _t.items())
                  + f" other={(_tot-_acc)*1000:.0f}ms", flush=True)
        return out

    @staticmethod
    def _stock_chunk_expert_map(c0: int, c1: int, E: int,
                                dev) -> torch.Tensor:
        """global->local expert_map for one chunk shard: experts [c0, c1) map to
        [0, c1-c0), all others -1. Built from python ints only (no tensor read),
        so it is a constant of the captured graph."""
        m = torch.full((E,), -1, dtype=torch.int32, device=dev)
        m[c0:c1] = torch.arange(c1 - c0, dtype=torch.int32, device=dev)
        return m

    def _expand_stack_slice(self, layer, which: str, c0: int, c1: int, *,
                            to_fp8: bool) -> torch.Tensor:
        """Decode a CONTIGUOUS chunk of experts [c0, c1) to a stacked transient
        ``[nE, out, in]`` in ONE expander launch over the reshaped
        ``[(nE*out), row_bytes]`` view — identical to stacking _decode_expert over
        the chunk (test_stock_expand_matches_decode), but slice-indexed (a view,
        no advanced-index copy) so it never issues an index-tensor H2D sync under
        capture. ``to_fp8`` picks the fp8-direct byte transient (expand_cb_to_fp8,
        for the W8A8 kernel); else the bf16 transient (fp4-v2 value x composed
        scale, or fp8 value x per-channel scale)."""
        packed = getattr(layer, f"{which}_cb_qweight")[c0:c1]      # view [nE,o,b]
        nE = c1 - c0
        out_f = packed.shape[1]
        in_f = layer._cb_hidden if which == "w13" else layer._cb_inter
        row0 = torch.zeros(nE * out_f, dtype=torch.int32, device=packed.device)
        if to_fp8:                                                # fp8 bytes
            # CUDA expander (2.1x the Triton one, byte-identical) on the RAW
            # slice view: the kernel stages exactly type_size bytes per
            # superblock into smem, so it needs neither the +8 row pad nor
            # the .contiguous() copy the Triton path required — that pair
            # cost ~26 ms/layer of pure memcpy on Laguna-256E prefill.
            # PRISMAQUANT_CB_EXPAND=triton restores the old path (bisection).
            if os.environ.get("PRISMAQUANT_CB_EXPAND") == "triton":
                qwp = codec.pad_qweight(
                    packed.reshape(nE * out_f, -1).contiguous())
                W = expand_cb_to_fp8(
                    qwp, self._stock_cb_flat_fp8(layer), row0, nE * out_f,
                    in_f, self.k, self.n_sub, self.type_size)
            else:
                from . import ops as pq_ops
                W = pq_ops.cb_expand_fp8(
                    packed.reshape(nE * out_f, -1),
                    self._stock_cb_flat_fp8(layer), row0, nE * out_f, in_f,
                    self.k, self.n_sub, self.type_size)
        elif self.is_fp4:                                         # fp4 two-tier v2
            qwp = codec.pad_qweight(
                packed.reshape(nE * out_f, -1).contiguous())
            W = expand_fp4_v2_to_weight(
                qwp, layer._cb_flat, row0, layer._cb_compose, nE * out_f, in_f,
                self.k, self.n_sub, self.type_size)
        else:                                                     # fp8 -> bf16
            qwp = codec.pad_qweight(
                packed.reshape(nE * out_f, -1).contiguous())
            val = expand_cb_to_value(qwp, layer._cb_flat, row0, nE * out_f, in_f,
                                     self.k, self.n_sub, self.type_size,
                                     is_fp4=False)
            ws = getattr(layer, f"{which}_weight_scale")[c0:c1].reshape(
                nE * out_f).to(torch.float32)
            W = (val.float() * ws[:, None]).to(torch.bfloat16)
        return W.view(nE, out_f, in_f)

    @staticmethod
    def _stock_cb_flat_fp8(layer) -> torch.Tensor:
        """The per-layer codebook re-encoded to E4M3 bytes for the fp8-direct
        expand (every CB value is on the e4m3 grid — lossless). Cached on the
        layer (also built by the CUDA-decode gate); built once, before capture."""
        cb = getattr(layer, "_cb_flat_fp8", None)
        if cb is None:
            cb = layer._cb_flat.to(torch.float8_e4m3fn).view(
                torch.uint8).contiguous()
            layer._cb_flat_fp8 = cb
        return cb

    # -- grouped CUDA decode path -------------------------------------------
    def _cuda_moe_ok(self, layer) -> bool:
        ok = getattr(layer, "_cb_moe_cuda_ok", None)
        if ok is None:
            # Grouped CUDA GEMV support: fp8-CB v1 (n_sub=4, per-(expert,out)
            # fp32 scale) and fp4-CB two-tier v2 (n_sub=2, scale composed
            # in-kernel from the packed 9-byte section). fp4-v1 (bare e4m3
            # plane) has no grouped path yet.
            fmt_ok = ((self.n_sub == 4 and not self.is_fp4)
                      or (self.n_sub in (1, 2) and self.is_fp4
                          and self.is_v2))
            ok = (fmt_ok and os.environ.get(
                "PRISMAQUANT_CB_DECODE", "cuda") == "cuda")
            if ok:
                from .cuda_ext import get_ext
                ok = get_ext() is not None
            if ok and not self.is_fp4 and not hasattr(layer, "_cb_flat_fp8"):
                layer._cb_flat_fp8 = layer._cb_flat.to(
                    torch.float8_e4m3fn).view(torch.uint8).contiguous()
            layer._cb_moe_cuda_ok = ok
        return ok

    def _apply_grouped_decode(self, layer, x, topk_weights, topk_ids, act):
        """One grouped GEMV launch per projection over all routed
        (token, expert) pairs, numerics-matched to the per-expert loop:
        per-token activation QDQ on the module input AND the intermediate
        (fp8 dynamic for fp8-CB v1, fp4 group-16 RTN for fp4-CB v2), weights
        bf16(val*scale), fp32-accum GEMVs, per-add bf16 combine in the loop's
        expert-ascending order."""
        from . import ops as pq_ops
        T, K_hidden = x.shape
        topk = topk_ids.shape[-1]
        # Pairs sorted (token-major, expert-ascending) — the loop's
        # index_add_ order per token. All GPU, no host syncs.
        ids_sorted, order = torch.sort(topk_ids.to(torch.int32), dim=-1)
        w_sorted = torch.gather(topk_weights, -1, order.to(torch.int64))
        pair_expert = ids_sorted.reshape(-1).contiguous()
        pair_xrow = (torch.arange(T, device=x.device, dtype=torch.int32)
                     .repeat_interleave(topk))
        # Match the loop's bf16-rounded router weight before the multiply.
        pair_w = (w_sorted.reshape(-1).to(torch.bfloat16)
                  .to(torch.float32).contiguous())
        tok_start = (torch.arange(T + 1, device=x.device, dtype=torch.int32)
                     * topk)
        pair_self = torch.arange(pair_expert.numel(), device=x.device,
                                 dtype=torch.int32)

        if self.is_fp4:                                  # fp4-CB two-tier v2
            # Activation QDQ (fp4 group-16 RTN) stays OUTSIDE the kernel,
            # bit-identical to the loop's codec.fp4_group16_act_qdq; the kernel
            # composes the two-tier weight scale in-register from the packed
            # 9-byte section + the resident (256,16) compose table.
            xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
            gate_up = pq_ops.cb_moe_gemv_fp4_v2(
                xq, layer.w13_cb_qweight.data, layer._cb_flat,
                layer._cb_compose, pair_expert, pair_xrow,
                self.k, self.n_sub, self.type_size)      # (P, 2*inter)
        else:                                            # fp8-CB v1
            xq = pq_ops.fp8_act_qdq(x.to(torch.bfloat16))
            gate_up = pq_ops.cb_moe_gemv_fp8(
                xq, layer.w13_cb_qweight.data, layer._cb_flat_fp8,
                layer.w13_weight_scale.data, pair_expert, pair_xrow,
                self.k, self.n_sub, self.type_size)      # (P, 2*inter)

        d = gate_up.shape[-1] // 2
        a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                        device=gate_up.device)
        apply_moe_activation(act, a, gate_up)

        if self.is_fp4:
            aq = codec.fp4_group16_act_qdq(a).to(torch.bfloat16)
            y_down = pq_ops.cb_moe_gemv_fp4_v2(
                aq, layer.w2_cb_qweight.data, layer._cb_flat,
                layer._cb_compose, pair_expert, pair_self,
                self.k, self.n_sub, self.type_size)      # (P, hidden)
        else:
            aq = pq_ops.fp8_act_qdq(a)
            y_down = pq_ops.cb_moe_gemv_fp8(
                aq, layer.w2_cb_qweight.data, layer._cb_flat_fp8,
                layer.w2_weight_scale.data, pair_expert, pair_self,
                self.k, self.n_sub, self.type_size)      # (P, hidden)
        return pq_ops.cb_moe_combine(y_down, pair_w, tok_start, T)
