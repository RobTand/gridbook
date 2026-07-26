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
        mode = os.environ.get("PRISMAQUANT_CB_PREFILL") or (
            "stock" if not self.is_fp4 else "loop")
        if mode == "loop":
            return self._apply_prefill_loop(
                layer, x, topk_weights, topk_ids, act)
        if mode == "stock":
            return self._apply_prefill_stock(
                layer, x, topk_weights, topk_ids, act)
        return self._apply_prefill_batched(
            layer, x, topk_weights, topk_ids, act)

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
