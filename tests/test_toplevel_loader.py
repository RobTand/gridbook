"""Top-level stacked-CB expert-loader wrapper (moe_toplevel_loader.py).

Regression cover for the HYV3 (Hy3) serving bug: archs that load MoE experts at
the top-level model via a per-expert ``expert_params_mapping`` (and ``continue``
past ``mlp.experts`` in their stacked-params loop) NEVER call the per-layer
``FusedMoE.load_weights``, so our instance-level CB hook is dead code for them
and the stock loader ``KeyError``s on our stacked ``…experts.gate_up_proj.
cb_qweight``. The wrapper here intercepts exactly those, copies them into the
registered fused params, and delegates everything else unchanged.

torch-only (no vLLM): the wrapper module imports just torch, so this runs in any
venv with torch. The full 194-tensor artifact mapping proof is the offline
validate_hy3_cb_loader.py oracle (create_weights + real quant_config schemes).
"""
import pytest
import torch

import types

from gridbook import moe_toplevel_loader
from gridbook.moe_toplevel_loader import (
    _build_reverse_fusion,
    _load_shared_cb,
    _spec_layer_rename,
    install_toplevel_cb_expert_loader,
    map_cb_expert_name,
    resolve_shared_cb_target,
)


def test_map_cb_expert_name_positive():
    P = "model.layers.7.mlp.experts."
    assert map_cb_expert_name(P + "gate_up_proj.cb_qweight") == P + "w13_cb_qweight"
    assert map_cb_expert_name(P + "down_proj.cb_qweight") == P + "w2_cb_qweight"
    assert map_cb_expert_name(P + "gate_up_proj.weight_scale") == P + "w13_weight_scale"
    assert map_cb_expert_name(P + "down_proj.weight_scale") == P + "w2_weight_scale"
    # prefix-agnostic (pure suffix rewrite): works with or without model. prefix
    assert map_cb_expert_name("layers.7.mlp.experts.down_proj.cb_qweight") == \
        "layers.7.mlp.experts.w2_cb_qweight"


def test_map_cb_expert_name_excludes_non_experts():
    # The .experts. anchor (not just the leaf name) excludes shared_mlp / dense
    # MLP / router / attention — those must reach the ORIGINAL loader.
    for n in [
        "model.layers.72.mlp.shared_mlp.gate_proj.cb_qweight",   # shared expert
        "model.layers.72.mlp.shared_mlp.up_proj.cb_qweight",
        "model.layers.5.mlp.shared_mlp.gate_up_proj.cb_qweight",  # synth fused shared
        "model.layers.0.mlp.gate_proj.cb_qweight",               # dense L0
        "model.layers.0.mlp.down_proj.cb_qweight",               # dense L0 down
        "model.layers.1.mlp.router.gate.weight",                 # router
        "model.layers.1.self_attn.qkv_proj.cb_qweight",          # dense attn
        "model.layers.1.mlp.gate.expert_bias",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]:
        assert map_cb_expert_name(n) is None, n


def test_wrapper_routes_experts_and_delegates_rest():
    E, HID, INTER, BYTES = 4, 16, 8, 3

    class _FakeCausalLM:                      # fresh class -> fresh sentinel
        def __init__(self):
            self._params = {
                "model.layers.1.mlp.experts.w13_cb_qweight":
                    torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8),
                "model.layers.1.mlp.experts.w2_cb_qweight":
                    torch.zeros(E, HID, BYTES, dtype=torch.uint8),
                "model.layers.1.mlp.experts.w13_weight_scale":
                    torch.zeros(E, 2 * INTER, dtype=torch.float32),
                "model.layers.1.mlp.experts.w2_weight_scale":
                    torch.zeros(E, HID, dtype=torch.float32),
                "model.layers.0.mlp.down_proj.cb_qweight":       # dense, delegated
                    torch.zeros(HID, BYTES, dtype=torch.uint8),
                "model.embed_tokens.weight": torch.zeros(10, HID),
            }
            self.delegated = []

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):     # ORIGINAL (stub stock loader)
            loaded = set()
            for name, w in weights:
                self.delegated.append(name)
                if name in self._params:
                    self._params[name].copy_(w)
                    loaded.add(name)
            return loaded

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    ckpt = [
        ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        ("model.layers.1.mlp.experts.down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        ("model.layers.1.mlp.experts.gate_up_proj.weight_scale",
         torch.full((E, 2 * INTER), 2.0)),
        ("model.layers.1.mlp.experts.down_proj.weight_scale",
         torch.full((E, HID), 3.0)),
        ("model.layers.0.mlp.down_proj.cb_qweight",
         torch.full((HID, BYTES), 5, dtype=torch.uint8)),
        ("model.embed_tokens.weight", torch.full((10, HID), 1.0)),
    ]
    loaded = m.load_weights(iter(ckpt))      # generator input (streaming semantics)

    # expert stacks copied into the registered fused params
    assert torch.all(m._params["model.layers.1.mlp.experts.w13_cb_qweight"] == 7)
    assert torch.all(m._params["model.layers.1.mlp.experts.w2_cb_qweight"] == 9)
    assert torch.allclose(m._params["model.layers.1.mlp.experts.w13_weight_scale"],
                          torch.tensor(2.0))
    assert torch.allclose(m._params["model.layers.1.mlp.experts.w2_weight_scale"],
                          torch.tensor(3.0))
    # dense / embedding delegated to the original and copied there
    assert torch.all(m._params["model.layers.0.mlp.down_proj.cb_qweight"] == 5)
    assert torch.allclose(m._params["model.embed_tokens.weight"], torch.tensor(1.0))
    # NO expert tensor leaked to the original loader (no double-load)
    assert not any(".experts." in n for n in m.delegated), m.delegated
    # loaded set = 4 mapped expert params (model. prefix) + 2 delegated
    assert loaded == {
        "model.layers.1.mlp.experts.w13_cb_qweight",
        "model.layers.1.mlp.experts.w2_cb_qweight",
        "model.layers.1.mlp.experts.w13_weight_scale",
        "model.layers.1.mlp.experts.w2_weight_scale",
        "model.layers.0.mlp.down_proj.cb_qweight",
        "model.embed_tokens.weight",
    }


def test_wrapper_defers_unmappable_expert_name():
    # A suffix match whose target param is absent (PP/EP-missing, or an MTP/spec
    # layer the original filters) must DEFER to the original, never hard-fail.
    class _FakeCausalLM:
        def __init__(self):
            self._params = {}            # no expert params registered here
            self.delegated = []

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):
            self.delegated = [n for n, _ in weights]
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    m.load_weights(iter([
        ("model.layers.80.mlp.experts.gate_up_proj.cb_qweight",
         torch.zeros(2, 2, 1, dtype=torch.uint8)),
    ]))
    assert m.delegated == ["model.layers.80.mlp.experts.gate_up_proj.cb_qweight"]


def test_wrapper_shape_mismatch_raises():
    class _FakeCausalLM:
        def __init__(self):
            self._params = {
                "model.layers.1.mlp.experts.w13_cb_qweight":
                    torch.zeros(4, 6, 3, dtype=torch.uint8),
            }

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):
            for _ in weights:
                pass
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    with pytest.raises(ValueError, match="contract violated"):
        m.load_weights(iter([
            ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
             torch.zeros(4, 6, 99, dtype=torch.uint8)),   # wrong byte width
        ]))


def test_install_is_idempotent():
    class _FakeCausalLM:
        def load_weights(self, weights):
            return set()

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    wrapped = _FakeCausalLM.load_weights
    install_toplevel_cb_expert_loader(_FakeCausalLM)     # second call: no-op
    assert _FakeCausalLM.load_weights is wrapped
    assert _FakeCausalLM._pq_cb_wrapped is True


# ---------------------------------------------------------------------------
# Shared-expert (shared_mlp) CB interception. The Hy3 shared MLP is built as
# PLAIN bf16 Linears (fused gate_up_proj + down_proj), so the checkpoint's
# …shared_mlp.{gate_proj,up_proj,down_proj}.cb_qweight have no cb_qweight param
# to load into. The wrapper decodes them to bf16 and injects into the .weight.
# ---------------------------------------------------------------------------

_REV = _build_reverse_fusion({"gate_up_proj": ["gate_proj", "up_proj"],
                              "qkv_proj": ["q_proj", "k_proj", "v_proj"]})


def test_resolve_shared_cb_fused_gate_up():
    # vLLM built a merged bf16 gate_up_proj.weight (no cb_qweight) -> intercept,
    # routing gate to shard 0 and up to shard 1.
    P = "model.layers.1.mlp.shared_mlp"
    params = {P + ".gate_up_proj.weight", P + ".down_proj.weight"}
    assert resolve_shared_cb_target(P + ".gate_proj.cb_qweight", params, _REV) \
        == (P + ".gate_up_proj.weight", 0)
    assert resolve_shared_cb_target(P + ".up_proj.cb_qweight", params, _REV) \
        == (P + ".gate_up_proj.weight", 1)
    # fp8 weight_scale routes exactly like its cb_qweight sibling.
    assert resolve_shared_cb_target(P + ".gate_proj.weight_scale", params, _REV) \
        == (P + ".gate_up_proj.weight", 0)


def test_resolve_shared_cb_direct_down():
    P = "model.layers.1.mlp.shared_mlp"
    params = {P + ".gate_up_proj.weight", P + ".down_proj.weight"}
    assert resolve_shared_cb_target(P + ".down_proj.cb_qweight", params, _REV) \
        == (P + ".down_proj.weight", 0)
    assert resolve_shared_cb_target(P + ".down_proj.weight_scale", params, _REV) \
        == (P + ".down_proj.weight", 0)


def test_resolve_shared_cb_defers_dense_and_fused_cb():
    # A genuine dense-CB Linear and a fused-attention shard both have a
    # REGISTERED cb_qweight param -> defer to the original loader (return None).
    params = {
        "model.layers.1.self_attn.qkv_proj.cb_qweight",   # fused attn (q/k/v)
        "model.layers.1.self_attn.o_proj.cb_qweight",      # dense o_proj
        "model.layers.0.mlp.down_proj.cb_qweight",         # dense L0 MLP (CB)
    }
    for n in [
        "model.layers.1.self_attn.q_proj.cb_qweight",       # -> qkv_proj (CB)
        "model.layers.1.self_attn.v_proj.weight_scale",
        "model.layers.1.self_attn.o_proj.cb_qweight",
        "model.layers.1.self_attn.o_proj.weight_scale",
        "model.layers.0.mlp.down_proj.cb_qweight",
    ]:
        assert resolve_shared_cb_target(n, params, _REV) is None, n


def test_resolve_shared_cb_defers_absent_and_non_cb():
    params = {"model.layers.1.mlp.shared_mlp.gate_up_proj.weight"}
    # target entirely absent on this rank -> defer
    assert resolve_shared_cb_target(
        "model.layers.9.mlp.shared_mlp.down_proj.cb_qweight", params, _REV) is None
    # not a CB tensor at all -> not ours
    assert resolve_shared_cb_target("model.embed_tokens.weight", params, _REV) is None
    assert resolve_shared_cb_target(
        "model.layers.1.mlp.gate.expert_bias", params, _REV) is None


def _stub_decode(scheme, cb_qweight, weight_scale, codebooks, dev):
    """CPU stand-in for the CUDA expander: returns an [out, in] bf16 tile filled
    with the source's marker byte so tests can trace placement/fusion order.
    in_features is recovered exactly as the real decode does."""
    out = int(cb_qweight.shape[0])
    in_f = (int(cb_qweight.shape[1]) // int(scheme["type_size"])) * 256
    marker = float(cb_qweight.reshape(-1)[0].item())
    return torch.full((out, in_f), marker, dtype=torch.bfloat16)


class _StubQuantConfig:
    def __init__(self, target_scheme):
        self.target_scheme = target_scheme

    def get_codebooks(self):
        return {}


def test_load_shared_cb_fuses_and_injects(monkeypatch):
    monkeypatch.setattr(moe_toplevel_loader, "_decode_cb_linear_to_bf16",
                        _stub_decode)
    HID, INTER, TS = 512, 256, 128            # hidden, shared inter, fp8 type_size
    P = "model.layers.1.mlp.shared_mlp"
    # gate/up cb_qweight: (inter, (hidden/256)*TS) -> in_f == hidden after decode.
    gu_bytes = (HID // 256) * TS
    dn_bytes = (INTER // 256) * TS
    params_dict = {
        P + ".gate_up_proj.weight": torch.zeros(2 * INTER, HID, dtype=torch.bfloat16),
        P + ".down_proj.weight": torch.zeros(HID, INTER, dtype=torch.bfloat16),
    }
    scheme = {"grid": "fp8", "k": 32, "n_sub": 4, "type_size": TS,
              "codebook_ref": ["cb"]}
    quant_config = _StubQuantConfig({
        P + ".gate_proj": scheme, P + ".up_proj": scheme, P + ".down_proj": scheme,
    })
    buf = {
        P + ".gate_proj.cb_qweight": torch.full((INTER, gu_bytes), 1, dtype=torch.uint8),
        P + ".gate_proj.weight_scale": torch.ones(INTER, dtype=torch.float32),
        P + ".up_proj.cb_qweight": torch.full((INTER, gu_bytes), 2, dtype=torch.uint8),
        P + ".up_proj.weight_scale": torch.ones(INTER, dtype=torch.float32),
        P + ".down_proj.cb_qweight": torch.full((HID, dn_bytes), 3, dtype=torch.uint8),
        P + ".down_proj.weight_scale": torch.ones(HID, dtype=torch.float32),
    }
    loaded = _load_shared_cb(None, buf, params_dict, _REV, quant_config)
    assert loaded == {P + ".gate_up_proj.weight", P + ".down_proj.weight"}
    gu = params_dict[P + ".gate_up_proj.weight"]
    assert torch.all(gu[:INTER] == 1), "gate rows (shard 0) go first"
    assert torch.all(gu[INTER:] == 2), "up rows (shard 1) go second"
    assert tuple(gu.shape) == (2 * INTER, HID)
    assert torch.all(params_dict[P + ".down_proj.weight"] == 3)


def test_wrapper_end_to_end_shared_mlp(monkeypatch):
    monkeypatch.setattr(moe_toplevel_loader, "_decode_cb_linear_to_bf16",
                        _stub_decode)
    HID, INTER, TS = 512, 256, 128
    P = "model.layers.1.mlp.shared_mlp"
    gu_bytes = (HID // 256) * TS
    scheme = {"grid": "fp8", "k": 32, "n_sub": 4, "type_size": TS,
              "codebook_ref": ["cb"]}

    class _FakeCausalLM:
        packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        def __init__(self):
            self._params = {
                P + ".gate_up_proj.weight": torch.zeros(2 * INTER, HID, dtype=torch.bfloat16),
                P + ".down_proj.weight": torch.zeros(HID, INTER, dtype=torch.bfloat16),
                "model.embed_tokens.weight": torch.zeros(4, HID),
            }
            # a CB quant_method somewhere on the model so _find_prismaquant_config
            # locates the config (mirrors a real dense-CB Linear).
            qc = _StubQuantConfig({P + ".gate_proj": scheme, P + ".up_proj": scheme,
                                   P + ".down_proj": scheme})

            class _M:
                class quant_method:
                    quant_config = qc
            self._marker_mod = _M()
            self.delegated = []

        def modules(self):
            return [self._marker_mod]

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):
            loaded = set()
            for name, w in weights:
                self.delegated.append(name)
                if name in self._params:
                    self._params[name].copy_(w)
                    loaded.add(name)
            return loaded

    install_toplevel_cb_expert_loader(_FakeCausalLM)
    m = _FakeCausalLM()
    dn_bytes = (INTER // 256) * TS
    ckpt = [
        (P + ".gate_proj.cb_qweight", torch.full((INTER, gu_bytes), 1, dtype=torch.uint8)),
        (P + ".gate_proj.weight_scale", torch.ones(INTER)),
        (P + ".up_proj.cb_qweight", torch.full((INTER, gu_bytes), 2, dtype=torch.uint8)),
        (P + ".up_proj.weight_scale", torch.ones(INTER)),
        (P + ".down_proj.cb_qweight", torch.full((HID, dn_bytes), 3, dtype=torch.uint8)),
        (P + ".down_proj.weight_scale", torch.ones(HID)),
        ("model.embed_tokens.weight", torch.full((4, HID), 5.0)),
    ]
    loaded = m.load_weights(iter(ckpt))
    # shared_mlp CB tensors were intercepted (never delegated to the original)
    assert not any("shared_mlp" in n for n in m.delegated), m.delegated
    # decoded + injected
    assert torch.all(m._params[P + ".gate_up_proj.weight"][:INTER] == 1)
    assert torch.all(m._params[P + ".gate_up_proj.weight"][INTER:] == 2)
    assert torch.all(m._params[P + ".down_proj.weight"] == 3)
    # embedding still delegated + loaded
    assert "model.embed_tokens.weight" in m.delegated
    assert torch.allclose(m._params["model.embed_tokens.weight"], torch.tensor(5.0))
    assert P + ".gate_up_proj.weight" in loaded
    assert P + ".down_proj.weight" in loaded
    assert "model.embed_tokens.weight" in loaded


# ---------------------------------------------------------------------------
# Spec-layer (MTP drafter) support. vLLM's HYV3MTP is fed the WHOLE checkpoint
# stream but keeps only its spec layer (model.layers.{num_hidden_layers}.*),
# renaming that layer's block tensors into a `.mtp_block.` nesting via
# `_rewrite_spec_layer_name`. Our stacked-CB expert tensors arrive WITHOUT the
# `.mtp_block.` infix while the registered params carry it, so the wrapper must
# apply the same rename before resolving. torch-only fakes mirror that rename +
# the routed_experts param nesting; no vLLM.
# ---------------------------------------------------------------------------

_MTP_SPEC_LAYER = 80


def _mtp_rewrite(spec_layer: int, name: str) -> str:
    """Faithful copy of HYV3MTP._rewrite_spec_layer_name (hy_v3_mtp.py)."""
    if f"model.layers.{spec_layer}.embed_tokens" in name:
        return "__skip__"
    if f"model.layers.{spec_layer}.shared_head" in name:
        return "__skip__"
    if not any(w in name for w in
               ("enorm", "hnorm", "eh_proj", "final_layernorm")):
        name = name.replace(f"model.layers.{spec_layer}.",
                            f"model.layers.{spec_layer}.mtp_block.")
    return name


def _make_fake_mtp_cls():
    """A minimal HYV3MTP-shaped drafter class: exposes the spec rename +
    a config, and an ORIGINAL load_weights that (like the real one) filters to
    the spec layer, applies the rename, and drops `__skip__`."""

    class _FakeMTP:
        def __init__(self, params):
            self._params = dict(params)
            self.delegated = []
            self.config = types.SimpleNamespace(
                num_hidden_layers=_MTP_SPEC_LAYER, num_nextn_predict_layers=1)

        def _rewrite_spec_layer_name(self, spec_layer, name):
            return _mtp_rewrite(spec_layer, name)

        def named_parameters(self):
            return list(self._params.items())

        def modules(self):
            return []

        def load_weights(self, weights):     # ORIGINAL (stub stock loader)
            loaded = set()
            for name, w in weights:
                if name.startswith(f"model.layers.{_MTP_SPEC_LAYER}."):
                    name = self._rewrite_spec_layer_name(_MTP_SPEC_LAYER, name)
                    if name == "__skip__":
                        continue
                elif name.startswith("model.layers."):
                    # A body (non-spec) layer the drafter never instantiates:
                    # the real loader's `spec_layer is None` filter drops it.
                    continue
                self.delegated.append(name)
                if name in self._params:
                    self._params[name].copy_(w)
                    loaded.add(name)
            return loaded

    return _FakeMTP


def test_spec_layer_rename_helper():
    class _M:
        config = types.SimpleNamespace(num_hidden_layers=80,
                                       num_nextn_predict_layers=1)

        def _rewrite_spec_layer_name(self, s, name):
            return _mtp_rewrite(s, name)

    r = _spec_layer_rename(_M())
    # spec-layer block tensor -> .mtp_block. nesting
    assert r("model.layers.80.mlp.experts.gate_up_proj.cb_qweight") == \
        "model.layers.80.mtp_block.mlp.experts.gate_up_proj.cb_qweight"
    assert r("model.layers.80.self_attn.q_proj.cb_qweight") == \
        "model.layers.80.mtp_block.self_attn.q_proj.cb_qweight"
    # glue stays at the layer level (no .mtp_block.)
    assert r("model.layers.80.enorm.weight") == "model.layers.80.enorm.weight"
    assert r("model.layers.80.eh_proj.weight") == "model.layers.80.eh_proj.weight"
    # embed/shared_head -> skip (main embed/lm_head reused)
    assert r("model.layers.80.embed_tokens.weight") == "__skip__"
    assert r("model.layers.80.shared_head.head.weight") == "__skip__"
    # body layer name unchanged (defers)
    assert r("model.layers.5.mlp.experts.gate_up_proj.cb_qweight") == \
        "model.layers.5.mlp.experts.gate_up_proj.cb_qweight"


def test_spec_layer_rename_none_for_body_model():
    # A model without _rewrite_spec_layer_name / config -> identity (None).
    assert _spec_layer_rename(object()) is None

    class _NoSpec:
        config = types.SimpleNamespace(num_hidden_layers=80,
                                       num_nextn_predict_layers=0)

        def _rewrite_spec_layer_name(self, s, name):  # present but n_spec == 0
            return name

    assert _spec_layer_rename(_NoSpec()) is None


def test_spec_layer_routes_mtp_experts_and_defers_body():
    E, HID, INTER, BYTES = 4, 16, 8, 3
    RE = f"model.layers.{_MTP_SPEC_LAYER}.mtp_block.mlp.experts.routed_experts"
    params = {
        RE + ".w13_cb_qweight": torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8),
        RE + ".w2_cb_qweight": torch.zeros(E, HID, BYTES, dtype=torch.uint8),
        f"model.layers.{_MTP_SPEC_LAYER}.enorm.weight": torch.zeros(HID),
    }
    cls = _make_fake_mtp_cls()
    install_toplevel_cb_expert_loader(cls)
    m = cls(params)
    ckpt = [
        (f"model.layers.{_MTP_SPEC_LAYER}.mlp.experts.gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        (f"model.layers.{_MTP_SPEC_LAYER}.mlp.experts.down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        (f"model.layers.{_MTP_SPEC_LAYER}.enorm.weight",
         torch.full((HID,), 2.0)),
        # a BODY expert tensor: the drafter has no param for it -> defer + drop
        ("model.layers.5.mlp.experts.gate_up_proj.cb_qweight",
         torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8)),
    ]
    loaded = m.load_weights(iter(ckpt))

    # spec-layer expert stacks copied into the .mtp_block. routed_experts params
    assert torch.all(m._params[RE + ".w13_cb_qweight"] == 7)
    assert torch.all(m._params[RE + ".w2_cb_qweight"] == 9)
    # glue (enorm) stays at the layer level and is delegated + loaded
    assert torch.allclose(m._params[f"model.layers.{_MTP_SPEC_LAYER}.enorm.weight"],
                          torch.tensor(2.0))
    # NO expert tensor leaked to the original loader (no double-load)
    assert not any(".experts." in n for n in m.delegated), m.delegated
    assert loaded == {RE + ".w13_cb_qweight", RE + ".w2_cb_qweight",
                      f"model.layers.{_MTP_SPEC_LAYER}.enorm.weight"}


def test_spec_layer_skips_embed_and_shared_head():
    cls = _make_fake_mtp_cls()
    install_toplevel_cb_expert_loader(cls)
    m = cls({f"model.layers.{_MTP_SPEC_LAYER}.enorm.weight": torch.zeros(4)})
    loaded = m.load_weights(iter([
        (f"model.layers.{_MTP_SPEC_LAYER}.embed_tokens.weight",
         torch.zeros(4, 4)),
        (f"model.layers.{_MTP_SPEC_LAYER}.shared_head.head.weight",
         torch.zeros(4, 4)),
    ]))
    # both -> "__skip__" -> delegated, the original drops them (never loaded)
    assert loaded == set()
    assert not any("embed_tokens" in n or "shared_head" in n
                   for n in m.delegated), m.delegated


def test_load_shared_cb_with_spec_rename(monkeypatch):
    # Decode-at-load path under a spec rename: buffer keyed on the ORIGINAL
    # (no-mtp_block) name so the scheme lookup matches the config target, while
    # the bf16 target PARAM carries the .mtp_block. infix.
    monkeypatch.setattr(moe_toplevel_loader, "_decode_cb_linear_to_bf16",
                        _stub_decode)
    HID, INTER, TS = 512, 256, 128
    cfg = f"model.layers.{_MTP_SPEC_LAYER}.mlp.shared_mlp"          # config key
    par = f"model.layers.{_MTP_SPEC_LAYER}.mtp_block.mlp.shared_mlp"  # param key
    scheme = {"grid": "fp8", "k": 32, "n_sub": 4, "type_size": TS,
              "codebook_ref": ["cb"]}
    quant_config = _StubQuantConfig({
        cfg + ".gate_proj": scheme, cfg + ".up_proj": scheme,
        cfg + ".down_proj": scheme})
    params_dict = {
        par + ".gate_up_proj.weight": torch.zeros(2 * INTER, HID, dtype=torch.bfloat16),
        par + ".down_proj.weight": torch.zeros(HID, INTER, dtype=torch.bfloat16),
    }

    def rename(n):     # spec rename: model.layers.80. -> ...mtp_block.
        return n.replace(f"model.layers.{_MTP_SPEC_LAYER}.",
                         f"model.layers.{_MTP_SPEC_LAYER}.mtp_block.")

    gu_bytes = (HID // 256) * TS
    dn_bytes = (INTER // 256) * TS
    buf = {
        cfg + ".gate_proj.cb_qweight": torch.full((INTER, gu_bytes), 1, dtype=torch.uint8),
        cfg + ".gate_proj.weight_scale": torch.ones(INTER, dtype=torch.float32),
        cfg + ".up_proj.cb_qweight": torch.full((INTER, gu_bytes), 2, dtype=torch.uint8),
        cfg + ".up_proj.weight_scale": torch.ones(INTER, dtype=torch.float32),
        cfg + ".down_proj.cb_qweight": torch.full((HID, dn_bytes), 3, dtype=torch.uint8),
        cfg + ".down_proj.weight_scale": torch.ones(HID, dtype=torch.float32),
    }
    loaded = _load_shared_cb(None, buf, params_dict, _REV, quant_config,
                             rename=rename)
    assert loaded == {par + ".gate_up_proj.weight", par + ".down_proj.weight"}
    gu = params_dict[par + ".gate_up_proj.weight"]
    assert torch.all(gu[:INTER] == 1), "gate rows (shard 0) first"
    assert torch.all(gu[INTER:] == 2), "up rows (shard 1) second"
    assert torch.all(params_dict[par + ".down_proj.weight"] == 3)


# ---------------------------------------------------------------------------
# Multimodal prefix mapping (Qwen3.5-MoE): the checkpoint says
# ``model.language_model.layers.N.…`` while named_parameters() says
# ``language_model.model.layers.N.…``. The original loader applies the arch's
# WeightsMapper only INSIDE itself, so the wrapper must apply the model's own
# ``hf_to_vllm_mapper`` before resolving.
# ---------------------------------------------------------------------------

class _FakeWeightsMapper:
    """Minimal stand-in for vLLM's WeightsMapper (prefix rewrites only)."""

    def __init__(self, orig_to_new_prefix):
        self._prefixes = dict(orig_to_new_prefix)

    def _map_name(self, key):
        for orig, new in self._prefixes.items():
            if key.startswith(orig):
                if new is None:
                    return None
                return new + key[len(orig):]
        return key


_QWEN35_MAPPER = _FakeWeightsMapper({
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
})


def test_hf_mapper_rename_maps_multimodal_prefix():
    from gridbook.moe_toplevel_loader import _hf_mapper_rename

    class _M:
        hf_to_vllm_mapper = _QWEN35_MAPPER

    rename = _hf_mapper_rename(_M())
    assert rename("model.language_model.layers.3.mlp.experts.down_proj.cb_qweight") \
        == "language_model.model.layers.3.mlp.experts.down_proj.cb_qweight"
    # unmapped names pass through unchanged
    assert rename("model.layers.3.mlp.experts.down_proj.cb_qweight") == \
        "model.layers.3.mlp.experts.down_proj.cb_qweight"


def test_hf_mapper_rename_absent_is_none():
    from gridbook.moe_toplevel_loader import _hf_mapper_rename

    class _M:
        pass

    assert _hf_mapper_rename(_M()) is None


def test_compose_renames():
    from gridbook.moe_toplevel_loader import _compose_renames

    assert _compose_renames(None, None) is None
    f = _compose_renames(lambda n: n + "|a", None)
    assert f("x") == "x|a"
    g = _compose_renames(lambda n: n + "|a", lambda n: n + "|b")
    assert g("x") == "x|a|b"
    # a "__skip__" produced by an earlier stage short-circuits
    h = _compose_renames(lambda n: "__skip__", lambda n: n + "|b")
    assert h("x") == "__skip__"


def test_wrapper_routes_experts_under_language_model_prefix():
    """The 35B CB bug: stacked-CB expert tensors under the multimodal
    ``model.language_model.`` checkpoint prefix must reach the FusedMoE params
    registered under ``language_model.model.``."""
    E, HID, INTER, BYTES = 2, 8, 4, 3
    P = "language_model.model.layers.1.mlp.experts."

    class _FakeCondGen:
        hf_to_vllm_mapper = _QWEN35_MAPPER

        def __init__(self):
            self._params = {
                P + "w13_cb_qweight": torch.zeros(E, 2 * INTER, BYTES,
                                                  dtype=torch.uint8),
                P + "w2_cb_qweight": torch.zeros(E, HID, BYTES,
                                                 dtype=torch.uint8),
                P + "w13_weight_scale": torch.zeros(E, 2 * INTER,
                                                    dtype=torch.float32),
                P + "w2_weight_scale": torch.zeros(E, HID, dtype=torch.float32),
                # shared expert: vLLM built a FUSED CB Linear -> must DEFER
                "language_model.model.layers.1.mlp.shared_expert."
                "gate_up_proj.cb_qweight": torch.zeros(2 * INTER, BYTES,
                                                       dtype=torch.uint8),
                "language_model.model.layers.1.mlp.gate.weight":
                    torch.zeros(E, HID),
            }
            self.delegated = []

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):     # ORIGINAL (stub stock loader)
            loaded = set()
            for name, w in weights:
                self.delegated.append(name)
            return loaded

    install_toplevel_cb_expert_loader(_FakeCondGen)
    m = _FakeCondGen()
    C = "model.language_model.layers.1.mlp."
    loaded = m.load_weights(iter([
        (C + "experts.gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        (C + "experts.down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        (C + "experts.gate_up_proj.weight_scale", torch.full((E, 2 * INTER), 2.0)),
        (C + "experts.down_proj.weight_scale", torch.full((E, HID), 3.0)),
        # shared-expert CB shard: vLLM built a fused CB param -> delegate
        (C + "shared_expert.gate_proj.cb_qweight",
         torch.full((INTER, BYTES), 5, dtype=torch.uint8)),
        (C + "gate.weight", torch.zeros(E, HID)),
    ]))

    assert torch.all(m._params[P + "w13_cb_qweight"] == 7)
    assert torch.all(m._params[P + "w2_cb_qweight"] == 9)
    assert torch.allclose(m._params[P + "w13_weight_scale"], torch.tensor(2.0))
    assert torch.allclose(m._params[P + "w2_weight_scale"], torch.tensor(3.0))
    assert loaded == {P + "w13_cb_qweight", P + "w2_cb_qweight",
                      P + "w13_weight_scale", P + "w2_weight_scale"}
    # routed-expert tensors never leak; shared-expert + router DO delegate
    assert not any(".experts." in n for n in m.delegated), m.delegated
    assert m.delegated == [C + "shared_expert.gate_proj.cb_qweight",
                           C + "gate.weight"]


def test_install_wraps_subclass_with_own_load_weights():
    """A subclass defining its OWN load_weights must be wrapped even though it
    inherits the base's ``_pq_cb_wrapped`` sentinel; a subclass that merely
    INHERITS the wrapped function must not be double-wrapped."""
    class _Base:
        def load_weights(self, weights):
            return {n for n, _ in weights}

    class _OwnLoader(_Base):
        def load_weights(self, weights):     # distinct function object
            return {n for n, _ in weights}

    class _Inherits(_Base):
        pass

    install_toplevel_cb_expert_loader(_Base)
    base_fn = _Base.__dict__["load_weights"]
    install_toplevel_cb_expert_loader(_OwnLoader)
    install_toplevel_cb_expert_loader(_Inherits)

    assert getattr(base_fn, "_pq_cb_wrapper", False)
    own_fn = _OwnLoader.__dict__["load_weights"]
    assert getattr(own_fn, "_pq_cb_wrapper", False) and own_fn is not base_fn
    # inherited: no own load_weights installed, sentinel recorded
    assert "load_weights" not in _Inherits.__dict__
    assert _Inherits.__dict__.get("_pq_cb_wrapped") is True
    # idempotent
    install_toplevel_cb_expert_loader(_Base)
    assert _Base.__dict__["load_weights"] is base_fn
