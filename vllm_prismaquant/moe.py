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
from .expand import expand_cb_to_value, expand_fp4_v2_to_weight


def _row_bytes(in_features: int, type_size: int) -> int:
    return (in_features // codec.SUPERBLOCK) * type_size


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
        if not getattr(layer, "_cb_load_wrapped", False):
            orig_load = layer.load_weights
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

        out = torch.zeros_like(x)
        # Prefill / fallback: grouped by expert — decode each routed expert
        # once (transient), matmul its tokens, combine with the router
        # weight. bf16 MMA. The hit-expert list is computed with ONE host
        # sync (the old per-expert `bool(sel.any())` cost E syncs per layer
        # per forward — the dominant share of the 3.5 s TTFT).
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

    # -- grouped CUDA decode path -------------------------------------------
    def _cuda_moe_ok(self, layer) -> bool:
        ok = getattr(layer, "_cb_moe_cuda_ok", None)
        if ok is None:
            # Grouped CUDA GEMV support: fp8-CB v1 (n_sub=4, per-(expert,out)
            # fp32 scale) and fp4-CB two-tier v2 (n_sub=2, scale composed
            # in-kernel from the packed 9-byte section). fp4-v1 (bare e4m3
            # plane) has no grouped path yet.
            fmt_ok = ((self.n_sub == 4 and not self.is_fp4)
                      or (self.n_sub == 2 and self.is_fp4 and self.is_v2))
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
        from .cuda_ext import get_ext
        ext = get_ext()
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
            gate_up = ext.cb_moe_gemv_fp4_v2(
                xq, layer.w13_cb_qweight.data, layer._cb_flat,
                layer._cb_compose, pair_expert, pair_xrow,
                self.k, self.n_sub, self.type_size)      # (P, 2*inter)
        else:                                            # fp8-CB v1
            xq = ext.fp8_act_qdq(x.to(torch.bfloat16))
            gate_up = ext.cb_moe_gemv_fp8(
                xq, layer.w13_cb_qweight.data, layer._cb_flat_fp8,
                layer.w13_weight_scale.data, pair_expert, pair_xrow,
                self.k, self.n_sub, self.type_size)      # (P, 2*inter)

        d = gate_up.shape[-1] // 2
        a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                        device=gate_up.device)
        apply_moe_activation(act, a, gate_up)

        if self.is_fp4:
            aq = codec.fp4_group16_act_qdq(a).to(torch.bfloat16)
            y_down = ext.cb_moe_gemv_fp4_v2(
                aq, layer.w2_cb_qweight.data, layer._cb_flat,
                layer._cb_compose, pair_expert, pair_self,
                self.k, self.n_sub, self.type_size)      # (P, hidden)
        else:
            aq = ext.fp8_act_qdq(a)
            y_down = ext.cb_moe_gemv_fp8(
                aq, layer.w2_cb_qweight.data, layer._cb_flat_fp8,
                layer.w2_weight_scale.data, pair_expert, pair_self,
                self.k, self.n_sub, self.type_size)      # (P, hidden)
        return ext.cb_moe_combine(y_down, pair_w, tok_start, T)
