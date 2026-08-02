"""Mixed-container dispatch: CB groups (with a "scheme" key) stay ours; stock
compressed-tensors groups (no "scheme" key) are delegated to a real
CompressedTensorsConfig (serving-kernel.md §2). vLLM-only, so skip-guarded —
run in the vllm-node container:

  docker run --rm --gpus all -v /home/rob/gridbook:/gridbook --entrypoint bash \
    vllm-node:latest -c 'pip install -e /gridbook --no-deps -q; \
    cd /; python3 -m pytest /gridbook/tests/test_delegation.py -q'
"""
import pytest

pytest.importorskip("vllm")
from gridbook.config import PrismaQuantConfig  # noqa: E402
from gridbook.delegated_preflight import DelegatedBackendError  # noqa: E402


def _mixed_config():
    nvfp4_w = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
               "group_size": 16, "symmetric": True, "dynamic": False,
               "scale_dtype": "torch.float8_e4m3fn"}
    nvfp4_a = {"num_bits": 4, "type": "float", "strategy": "tensor_group",
               "group_size": 16, "symmetric": True, "dynamic": "local",
               "scale_dtype": "torch.float8_e4m3fn"}
    return {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_cb": {
                "format": "FP8_CB_K44",
                "targets": ["model.layers.0.mlp.down_proj"],
                "scheme": {"grid": "fp8", "mode": "product", "k": 44,
                           "n_sub": 4, "type_size": 176, "group_size": 0,
                           "vec_dim": 8, "codebook_group": "down_proj",
                           "codebook_source": "learned",
                           "codebook_ref": [
                               "cb_codebook.down_proj.FP8_CB_K44.sub0"]}},
            "group_nvfp4": {
                "format": "nvfp4-pack-quantized",
                "weights": nvfp4_w, "input_activations": nvfp4_a,
                "targets": ["re:.*mlp.gate_proj$", "re:.*mlp.up_proj$",
                            "re:.*self_attn.*_proj$"]},
        },
        "ignore": ["lm_head", "model.embed_tokens"],
    }


def test_cb_stock_split_and_ct_built():
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    assert c.ct_config is not None, "stock CT config not built for mixed artifact"
    assert len(c.target_scheme) == 1                     # 1 CB target
    assert "model.layers.0.mlp.down_proj" in c._cb_targets


def test_cb_prefix_stays_ours():
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    # CB target resolves to a CB scheme (fused-aware).
    assert c._scheme_for_prefix("model.layers.0.mlp.down_proj") is not None
    # stock prefix does NOT CB-resolve (falls through to CT delegation).
    assert c._scheme_for_prefix("model.layers.0.mlp.gate_proj") is None


def test_ct_owns_stock_and_ignores_cb():
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
    )
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    # CB modules are in CT's ignore so CT never tries to own them.
    assert "model.layers.0.mlp.down_proj" in c.ct_config.ignore
    # CT recognises a stock prefix as one of its targets.
    tgt = find_matched_target(
        "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj",
        list(c.ct_config.target_scheme_map.keys()),
        c.ct_config.packed_modules_mapping)
    assert tgt is not None


def test_uniform_cb_has_no_ct():
    """A pure-CB (uniform) artifact must not build a CT config (no stock groups)."""
    cfg = _mixed_config()
    del cfg["config_groups"]["group_nvfp4"]
    c = PrismaQuantConfig.from_config(cfg)
    c._ensure_resolved()
    assert c.ct_config is None


# --- D0.2: the delegated backend a mixed artifact actually resolves to -------
#
# The policy itself is unit-tested vLLM-free in ``test_delegated_preflight.py``.
# What is pinned HERE is the wiring: that ``PrismaQuantConfig`` reads the right
# declared group for a serving prefix using compressed-tensors' own matcher,
# and that its delegation choke point runs the policy on whatever vLLM handed
# back. Stub backend identities stand in for the vLLM ladder so the check does
# not need a GPU, a real FusedMoE, or a particular vLLM build.


def _stub_backend(qualname, module):
    """A class whose IDENTITY (module + qualname) is what the policy reads."""
    cls = type("StubBackend", (), {})
    cls.__name__ = qualname
    cls.__qualname__ = qualname
    cls.__module__ = module
    return cls


_EMULATION_EXPERTS = _stub_backend(
    "Nvfp4QuantizationEmulationTritonExperts",
    "vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe")
_MARLIN_EXPERTS = _stub_backend(
    "MarlinExperts", "vllm.model_executor.layers.fused_moe.experts.marlin_moe")
_CUTLASS_EXPERTS = _stub_backend(
    "CutlassExpertsFp4",
    "vllm.model_executor.layers.fused_moe.experts.cutlass_moe")


class _StubMethod:
    """The post-``__init__`` shape of ``CompressedTensorsW4A4Nvfp4MoEMethod``."""

    def __init__(self, experts_cls):
        self.experts_cls = experts_cls


def _delegating_config(method):
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    c.ct_config.get_quant_method = lambda layer, prefix: method
    return c


class _Layer:
    pass


def test_stock_group_resolves_for_a_delegated_prefix():
    c = PrismaQuantConfig.from_config(_mixed_config())
    c._ensure_resolved()
    name, group = c._stock_group_for_prefix(
        _Layer(), "model.layers.0.mlp.gate_proj")
    assert name == "group_nvfp4"
    assert group["input_activations"]["num_bits"] == 4      # declared W4A4
    # A CB target is never a delegated group.
    assert c._stock_group_for_prefix(
        _Layer(), "model.layers.0.mlp.down_proj")[0] is None


def test_delegated_emulation_backend_is_rejected_at_load():
    c = _delegating_config(_StubMethod(_EMULATION_EXPERTS))
    with pytest.raises(DelegatedBackendError) as excinfo:
        c._delegate(_Layer(), "model.layers.0.mlp.gate_proj")
    message = str(excinfo.value)
    assert "Nvfp4QuantizationEmulationTritonExperts" in message
    assert "Triton-backed" in message
    assert "group_nvfp4" in message and "W4A4" in message


def test_delegated_marlin_backend_is_rejected_at_load():
    c = _delegating_config(_StubMethod(_MARLIN_EXPERTS))
    with pytest.raises(DelegatedBackendError, match="activation scales"):
        c._delegate(_Layer(), "model.layers.0.mlp.gate_proj")


def test_delegated_native_backend_is_accepted_at_load():
    method = _StubMethod(_CUTLASS_EXPERTS)
    c = _delegating_config(method)
    assert c._delegate(_Layer(), "model.layers.0.mlp.gate_proj") is method


def _shared_mlp_config():
    """hy_v3-shaped: CB shared_mlp targets (layer 1) + an ignored (BF16)
    shared_mlp Linear (layer 2) + a genuine dense-MLP CB target (layer 0)."""
    cfg = _mixed_config()
    del cfg["config_groups"]["group_nvfp4"]
    cb = cfg["config_groups"]["group_cb"]
    cb["targets"] = [
        "model.layers.0.mlp.down_proj",              # dense layer, real key
        "model.layers.1.mlp.shared_mlp.gate_proj",
        "model.layers.1.mlp.shared_mlp.up_proj",
        "model.layers.1.mlp.shared_mlp.down_proj",
    ]
    cfg["ignore"].append("model.layers.2.mlp.shared_mlp.down_proj")
    return cfg


def test_shared_mlp_collapsed_dispatch_aliases():
    """HunYuan V3 builds the shared MLP with the PARENT ``…mlp`` prefix, so
    get_quant_method sees ``…mlp.gate_up_proj``/``…mlp.down_proj`` — the CB
    schemes must resolve under those collapsed names (and collapsed ignores
    must match), while the original ``.shared_mlp.`` keys survive for the
    checkpoint-name-keyed loading paths."""
    c = PrismaQuantConfig.from_config(_shared_mlp_config())
    c.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"],
                                "qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    c._ensure_resolved()
    # Collapsed direct + fused dispatch (what hy_v3 hands get_quant_method).
    assert c._scheme_for_prefix("model.layers.1.mlp.down_proj") is not None
    assert c._scheme_for_prefix("model.layers.1.mlp.gate_up_proj") is not None
    # HYV3 MTP nests the same block under .mtp_block. and still passes the
    # parent MLP prefix when constructing the shared projections. They must be
    # CB-owned too; a plain-bf16 compatibility fallback is forbidden.
    assert c._scheme_for_prefix(
        "model.layers.1.mtp_block.mlp.down_proj") is not None
    assert c._scheme_for_prefix(
        "model.layers.1.mtp_block.mlp.gate_up_proj") is not None
    # Also cover an architecture that preserves the explicit shared_mlp prefix.
    assert c._scheme_for_prefix(
        "model.layers.1.mtp_block.mlp.shared_mlp.down_proj") is not None
    # BF16 shared layer: the collapsed ignore entry matches its serve prefix.
    assert c._is_ignored("model.layers.2.mlp.down_proj")
    assert c._is_ignored("model.layers.2.mtp_block.mlp.down_proj")
    # Original keys survive (loader/scale paths are checkpoint-name-keyed) and
    # the genuine dense-layer key is untouched by aliasing.
    assert "model.layers.1.mlp.shared_mlp.gate_proj" in c.target_scheme
    assert c._scheme_for_prefix("model.layers.0.mlp.down_proj") is not None
    # A layer with no shared_mlp gained no phantom fused resolution.
    assert c._scheme_for_prefix("model.layers.2.mlp.gate_up_proj") is None
