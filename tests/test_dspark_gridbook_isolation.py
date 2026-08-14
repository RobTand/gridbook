"""DSpark structural and behavioral contract — three namespaces, fail-closed, mutation-strong.

Covers required corrections A-E. Uses E=2 synthetic target-like DSpark model with actual wrapper.
"""
import json
import pytest
import torch

from gridbook.moe_toplevel_loader import (
    install_toplevel_cb_expert_loader,
    _remap_dspark_physical_to_registered,
    _remap_dspark_physical_to_construction,
    _DSPARK_REGISTERED_STAGES,
    _DSPARK_CONSTRUCTION_OFFSET,
)
from gridbook.config import PrismaQuantConfig
from gridbook.runtime_contract import load_runtime_contract


# --- Synthetic DSpark model (E=2) ---
E, HID, INTER, BYTES = 2, 16, 8, 4

def _dparams():
    # Routed experts at registered layers 0,1,2
    d = {}
    for stage in (0,1,2):
        prefix = f"model.layers.{stage}.ffn.experts.routed_experts"
        d[prefix + ".w13_cb_qweight"] = torch.zeros(E, 2*INTER, BYTES, dtype=torch.uint8)
        d[prefix + ".w13_weight_scale"] = torch.zeros(E, 2*INTER, dtype=torch.float32)
        d[prefix + ".w2_cb_qweight"] = torch.zeros(E, HID, BYTES, dtype=torch.uint8)
        d[prefix + ".w2_weight_scale"] = torch.zeros(E, HID, dtype=torch.float32)
        # shared experts
        se = f"model.layers.{stage}.ffn.shared_experts"
        d[se + ".gate_up_proj.weight"] = torch.zeros(2*INTER, HID)
        d[se + ".down_proj.weight"] = torch.zeros(HID, INTER)
        # attention fused
        att = f"model.layers.{stage}.attn"
        d[att + ".fused_wqa_wkv.weight"] = torch.zeros(2*HID, HID)
        d[att + ".wq_b.weight"] = torch.zeros(HID, HID)
        d[att + ".wo_b.weight"] = torch.zeros(HID, HID)
    # heads
    d["model.main_proj.weight"] = torch.zeros(4096, 12288, dtype=torch.float8_e4m3fn)
    d["model.main_proj.scale"] = torch.zeros(32, 96, dtype=torch.float8_e8m0fnu)
    d["model.main_norm.weight"] = torch.zeros(HID)
    d["model.norm.weight"] = torch.zeros(HID)
    d["model.hc_head_fn"] = torch.zeros(4, 4*HID)
    d["model.hc_head_base"] = torch.zeros(4)
    d["model.hc_head_scale"] = torch.zeros(1)
    d["model.markov_head.weight"] = torch.zeros(HID, HID)
    return d

def _make_fake_dspark_cls(params):
    class _FakeDSpark:
        def __init__(self):
            self._params = dict(params)
            # Provide minimal quant_config for expected derivation
            # Use artifact topology 43/3 so loader derives allowed stages (0,1,2)
            class _QC:
                _dspark_topology = (43, 3)
                pass
            # Build target_scheme properly
            qc = _QC()
            qc.target_scheme = {}
            for s in (0,1,2):
                qc.target_scheme[f"model.layers.{43+s}.ffn.experts.gate_up_proj"] = {"grid":"fp4","mode":"product"}
                qc.target_scheme[f"model.layers.{43+s}.ffn.experts.down_proj"] = {"grid":"fp4","mode":"product"}
                qc.target_scheme[f"model.layers.{43+s}.attn.fused_wqa_wkv"] = {"grid":"fp4","mode":"product"}
            self.quant_config = qc
            self.hf_to_vllm_mapper = None
            self.packed_modules_mapping = None
        def named_parameters(self):
            return list(self._params.items())
        def load_weights(self, weights):
            loaded = set()
            for name, w in weights:
                # Simulate target _remap_dspark_name for delegated mtp.* -> model.*
                mapped = name
                if name.startswith("mtp.") or name.startswith("model.mtp."):
                    norm = name
                    if name.startswith("model.mtp."):
                        norm = "mtp." + name[len("model.mtp."):]
                    # simple remap: mtp.{stage}.{rest} -> model.layers.{stage}.{rest} or model.{head}
                    try:
                        rest_stage = norm[len("mtp."):]
                        dot = rest_stage.find(".")
                        stage_str = rest_stage if dot==-1 else rest_stage[:dot]
                        stage = int(stage_str)
                        rest = rest_stage[len(stage_str):]
                        inner = rest.lstrip(".")
                        if inner.startswith("confidence_head."):
                            continue
                        heads = ("main_proj.", "main_norm.", "norm.", "hc_head_fn", "hc_head_base", "hc_head_scale", "markov_head.")
                        is_head = any(inner.startswith(h) for h in heads)
                        if is_head:
                            mapped = f"model.{inner}"
                        else:
                            mapped = f"model.layers.{stage}{rest}"
                    except ValueError:
                        mapped = name
                if mapped in self._params:
                    if tuple(self._params[mapped].shape) != tuple(w.shape):
                        raise ValueError(f"shape mismatch {mapped}")
                    self._params[mapped].copy_(w.to(self._params[mapped].dtype))
                    loaded.add(mapped)
                elif ".confidence_head." in mapped:
                    continue
                else:
                    if mapped.startswith("model."):
                        continue
            return loaded
    # Make class recognizable as DSpark for exact check
    _FakeDSpark.__module__ = "vllm.models.deepseek_v4.nvidia.dspark"
    _FakeDSpark.__name__ = "DSparkDeepseekV4ForCausalLM"
    # Need to create new type with correct module/name for isinstance check uses type(model).__module__
    # So we return a dynamically created type
    attrs = dict(_FakeDSpark.__dict__)
    NewCls = type("DSparkDeepseekV4ForCausalLM", (_FakeDSpark,), {})
    NewCls.__module__ = "vllm.models.deepseek_v4.nvidia.dspark"
    return NewCls

# Helper to build config with physical mtp targets
def _dspark_config_with_topology():
    # Build a miniature quant_config that declares DSpark CB at physical mtp.0..2
    # Use construction mapping to 43..45 via sidecar dspark_topology
    return {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 43, "n_mtp_layers": 3},
        "dspark_block_size": 5,
        "config_groups": {
            "dspark_experts": {
                "targets": [
                    "mtp.0.ffn.experts.gate_up_proj",
                    "mtp.0.ffn.experts.down_proj",
                    "mtp.1.ffn.experts.gate_up_proj",
                    "mtp.1.ffn.experts.down_proj",
                    "mtp.2.ffn.experts.gate_up_proj",
                    "mtp.2.ffn.experts.down_proj",
                    "mtp.0.attn.fused_wqa_wkv",
                    "mtp.0.ffn.shared_experts.gate_up_proj",
                ],
                "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["cb_codebook.experts.NVFP4_CB_K15.sub0","cb_codebook.experts.NVFP4_CB_K15.sub1"]}
            },
            "heads": {
                "targets": ["model.norm"],
                "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["cb_codebook.experts.NVFP4_CB_K15.sub0","cb_codebook.experts.NVFP4_CB_K15.sub1"]}
            }
        },
        "ignore": [],
        "provenance": {}
    }

# --------------------------------------------------------------------------
# 1. Three namespaces remain explicit
# --------------------------------------------------------------------------
def test_physical_to_construction_vs_registered_are_distinct():
    for stage in (0,1,2):
        phys = f"mtp.{stage}.ffn.experts.gate_up_proj.cb_qweight"
        reg = _remap_dspark_physical_to_registered(phys)
        cons = _remap_dspark_physical_to_construction(phys, 43)
        assert reg == f"model.layers.{stage}.ffn.experts.gate_up_proj.cb_qweight"
        assert cons == f"model.layers.{43+stage}.ffn.experts.gate_up_proj.cb_qweight"
        assert reg != cons
    # heads same in both
    for h in ["mtp.1.norm.weight", "mtp.0.main_proj.weight"]:
        r = _remap_dspark_physical_to_registered(h)
        c = _remap_dspark_physical_to_construction(h, 43)
        assert r == c == h.replace("mtp.1.", "model.").replace("mtp.0.", "model.") if "mtp." in h else None

def test_construction_prefix_is_43_44_45():
    # Target constructor uses num_hidden_layers + i
    import pathlib
    target_path = pathlib.Path("/target-vllm/vllm/models/deepseek_v4/nvidia/dspark.py")
    text = target_path.read_text()
    assert "f\"layers.{self.num_hidden_layers + i}\"" in text or "layers.{self.num_hidden_layers + i}" in text
    # n_mtp_layers or 3
    assert 'getattr(config, "n_mtp_layers", None) or 3' in text

def test_registered_remap_uses_stage_i():
    import pathlib
    text = pathlib.Path("/target-vllm/vllm/models/deepseek_v4/nvidia/dspark.py").read_text()
    assert "model.layers.{stage}" in text
    assert "_remap_dspark_name" in text

def test_get_draft_quant_config_call_exists():
    import pathlib
    text = pathlib.Path("/target-vllm/vllm/model_executor/models/utils.py").read_text()
    assert "get_draft_quant_config" in text
    assert "VllmConfig.get_quantization_config(draft_model_config" in text

# --------------------------------------------------------------------------
# 2. Config body/draft isolation and construction vs registered
# --------------------------------------------------------------------------
def test_draft_physical_targets_resolve_at_construction_43_45(monkeypatch):
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = _dspark_config_with_topology()
    c = PrismaQuantConfig.from_config(cfg)
    c._ensure_resolved()
    assert "model.layers.43.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.44.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.45.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.0.ffn.experts.gate_up_proj" not in c.target_scheme
    # MoE scheme uses _moe_scheme_for_prefix for experts
    assert c._moe_scheme_for_prefix("model.layers.43.ffn.experts") is not None
    assert c._scheme_for_prefix("model.layers.43.attn.fused_wqa_wkv") is not None
    assert c._scheme_for_prefix("model.layers.3.ffn.experts") is None

def test_body_and_draft_independent_without_collision():
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 43, "n_mtp_layers": 3},
        "config_groups": {
            "body": {
                "targets": [f"model.layers.{i}.ffn.experts.gate_up_proj" for i in range(3)] + [f"model.layers.{i}.attn.wq_a" for i in range(3,5)],
                "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["cb0","cb1"]}
            },
            "draft": {
                "targets": ["mtp.0.ffn.experts.gate_up_proj","mtp.1.ffn.experts.gate_up_proj","mtp.2.ffn.experts.gate_up_proj"],
                "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["cb0","cb1"]}
            }
        },
        "ignore": [],
        "provenance": {}
    }
    c = PrismaQuantConfig.from_config(cfg)
    c._ensure_resolved()
    assert "model.layers.0.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.43.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.44.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.45.ffn.experts.gate_up_proj" in c.target_scheme
    assert len([k for k in c.target_scheme if "gate_up_proj" in k]) == 6

# --------------------------------------------------------------------------
# 3. Stage validation
# --------------------------------------------------------------------------
def test_unknown_stage_mtp3_fails():
    cfg = _dspark_config_with_topology()
    cfg["config_groups"]["dspark_experts"]["targets"].append("mtp.3.ffn.experts.gate_up_proj")
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(ValueError, match="unknown stage"):
        c._ensure_resolved()

def test_non_numeric_stage_fails():
    cfg = _dspark_config_with_topology()
    cfg["config_groups"]["dspark_experts"]["targets"].append("mtp.foo.ffn.experts.gate_up_proj")
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(ValueError, match="non-numeric"):
        c._ensure_resolved()

def test_gapped_stage_fails():
    cfg = _dspark_config_with_topology()
    # Remove mtp.1 to create gap 0,2
    cfg["config_groups"]["dspark_experts"]["targets"] = [t for t in cfg["config_groups"]["dspark_experts"]["targets"] if not t.startswith("mtp.1.")]
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(ValueError, match="expected contiguous"):
        c._ensure_resolved()

def test_main_proj_cb_declaration_rejected():
    cfg = _dspark_config_with_topology()
    cfg["config_groups"]["dspark_experts"]["targets"].append("mtp.0.main_proj")
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(ValueError, match="main_proj"):
        c._ensure_resolved()

def test_topology_missing_fails_closed():
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        # No dspark_topology and no vllm config
        "config_groups": {
            "g": {"targets":["mtp.0.ffn.experts.gate_up_proj"], "scheme":{"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}}
        },
        "ignore": []
    }
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(RuntimeError, match="cannot derive topology"):
        c._ensure_resolved()

# --------------------------------------------------------------------------
# 4. Mutation-strong synthetic DSpark loader
# --------------------------------------------------------------------------
def test_synthetic_dspark_loader_exact_bytes_and_heads():
    from gridbook.cb_fill_guard import CB_FILLED_ATTR
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    tensors = []
    for stage in (0,1,2):
        tensors.append((f"mtp.{stage}.ffn.experts.gate_up_proj.cb_qweight", torch.full((E,2*INTER,BYTES), stage+10, dtype=torch.uint8)))
        tensors.append((f"mtp.{stage}.ffn.experts.gate_up_proj.weight_scale", torch.full((E,2*INTER), float(stage+1))))
        tensors.append((f"mtp.{stage}.ffn.experts.down_proj.cb_qweight", torch.full((E,HID,BYTES), stage+20, dtype=torch.uint8)))
        tensors.append((f"mtp.{stage}.ffn.experts.down_proj.weight_scale", torch.full((E,HID), float(stage+2))))
        tensors.append((f"mtp.{stage}.attn.wq_a.weight", torch.ones(HID,HID)))
    tensors.append(("mtp.0.main_proj.weight", torch.full((4096,12288), 1, dtype=torch.float8_e4m3fn)))
    tensors.append(("mtp.0.main_proj.scale", torch.full((32,96), 2, dtype=torch.float8_e8m0fnu)))
    tensors.append(("mtp.1.main_norm.weight", torch.ones(HID)))
    tensors.append(("mtp.2.norm.weight", torch.ones(HID)))
    tensors.append(("mtp.2.hc_head_fn", torch.ones(4,4*HID)))
    tensors.append(("mtp.2.markov_head.weight", torch.ones(HID,HID)))
    tensors.append(("mtp.1.confidence_head.weight", torch.ones(10)))
    loaded = m.load_weights(iter(tensors))
    assert torch.all(m._params["model.layers.0.ffn.experts.routed_experts.w13_cb_qweight"] == 10)
    assert torch.all(m._params["model.layers.1.ffn.experts.routed_experts.w13_cb_qweight"] == 11)
    assert torch.all(m._params["model.layers.2.ffn.experts.routed_experts.w2_cb_qweight"] == 22)
    assert torch.all(m._params["model.main_proj.weight"] == 1)
    assert "model.main_proj.weight" in loaded
    assert not any("confidence" in k for k in loaded)
    assert len(loaded) >= 10

def test_duplicate_before_second_mutation_fails():
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    t = torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)
    with pytest.raises(ValueError, match="duplicate"):
        m.load_weights(iter([
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", t),
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", t),
        ]))

def test_missing_field_fails_exact():
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    with pytest.raises(ValueError, match="missing"):
        m.load_weights(iter([
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)),
            ("mtp.0.ffn.experts.gate_up_proj.weight_scale", torch.zeros(E,2*INTER)),
            ("mtp.1.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)),
            ("mtp.1.ffn.experts.gate_up_proj.weight_scale", torch.zeros(E,2*INTER)),
            ("mtp.1.ffn.experts.down_proj.cb_qweight", torch.zeros(E,HID,BYTES, dtype=torch.uint8)),
            ("mtp.1.ffn.experts.down_proj.weight_scale", torch.zeros(E,HID)),
            ("mtp.2.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)),
            ("mtp.2.ffn.experts.gate_up_proj.weight_scale", torch.zeros(E,2*INTER)),
            ("mtp.2.ffn.experts.down_proj.cb_qweight", torch.zeros(E,HID,BYTES, dtype=torch.uint8)),
            ("mtp.2.ffn.experts.down_proj.weight_scale", torch.zeros(E,HID)),
        ]))

def test_extra_wrong_field_fails():
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    with pytest.raises(ValueError, match="unknown/extra/wrong-leaf"):
        m.load_weights(iter([
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)),
            ("mtp.0.ffn.experts.gate_up_proj.weight_scale", torch.zeros(E,2*INTER)),
            ("mtp.0.ffn.experts.down_proj.cb_qweight", torch.zeros(E,HID,BYTES, dtype=torch.uint8)),
            ("mtp.0.ffn.experts.down_proj.weight_scale", torch.zeros(E,HID)),
            ("mtp.0.ffn.experts.gate_up_proj.fake_leaf.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)),
        ]))

def test_shape_mismatch_fails():
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    with pytest.raises(ValueError, match="shape"):
        m.load_weights(iter([
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(1,2*INTER,BYTES, dtype=torch.uint8)),
        ]))

def test_wrapper_absent_should_fail_if_dspark_cb_declared(monkeypatch):
    from gridbook import moe_toplevel_loader as mtl
    cfg = _dspark_config_with_topology()
    # Ensure dspark module is considered installed for normal, then clear
    # First ensure it is installed (our earlier runtime should have it)
    # Patch to empty
    monkeypatch.setattr(mtl, "installed_module_paths", lambda: set())
    # Need to create new config that hasn't been resolved yet; force re-resolve
    c = PrismaQuantConfig.from_config(cfg)
    # Clear any cached installed check by ensuring fresh
    with pytest.raises(RuntimeError, match="not installed"):
        c._ensure_resolved()

def test_body_layer_3_not_misclassified_with_draft():
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = _dspark_config_with_topology()
    c = PrismaQuantConfig.from_config(cfg)
    c._ensure_resolved()
    assert c._scheme_for_prefix("model.layers.3.attn.wq_a") is None
    with pytest.raises(ValueError, match="unknown stage"):
        c._scheme_for_prefix("model.layers.46.ffn.experts.gate_up_proj")

def test_draft_quant_config_none_fails(monkeypatch):
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = _dspark_config_with_topology()
    # Mock vllm_config with speculative_config and None draft quant_config
    import types
    from unittest import mock
    mock_vcfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            draft_model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(num_hidden_layers=43, n_mtp_layers=3, dspark_block_size=5),
                quantization="fp8"
            ),
            num_speculative_tokens=5
        ),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(num_hidden_layers=43)
        ),
        quant_config=None
    )
    # Patch get_current_vllm_config_or_none to return mock, and get_draft_quant_config to return None
    import gridbook.config as gcfg
    orig_topology_fn = gcfg._dspark_topology_from_vllm_config
    # Ensure topology derived from sidecar still works, but draft quant_config check should fail
    with mock.patch("vllm.config.get_current_vllm_config_or_none", return_value=mock_vcfg):
        with mock.patch("vllm.model_executor.models.utils.get_draft_quant_config", return_value=None):
            c = PrismaQuantConfig.from_config(cfg)
            with pytest.raises(RuntimeError, match="draft quant_config is None"):
                c._ensure_resolved()

def test_parent_draft_mismatch_fails(monkeypatch):
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = _dspark_config_with_topology()
    import types
    from unittest import mock
    mock_vcfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            draft_model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(num_hidden_layers=42, n_mtp_layers=3, dspark_block_size=5)
            ),
            num_speculative_tokens=5
        ),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(num_hidden_layers=43)
        ),
        quant_config=None
    )
    # Make get_draft_quant_config return a valid gridbook config to pass that check, but parent mismatch should still fail
    fake_qc = types.SimpleNamespace(get_name=lambda: "gridbook")
    with mock.patch("vllm.config.get_current_vllm_config_or_none", return_value=mock_vcfg):
        with mock.patch("vllm.model_executor.models.utils.get_draft_quant_config", return_value=fake_qc):
            c = PrismaQuantConfig.from_config(cfg)
            with pytest.raises(RuntimeError, match="mismatch"):
                c._ensure_resolved()

def test_dtype_mismatch_fails():
    Cls = _make_fake_dspark_cls(_dparams())
    from gridbook.moe_toplevel_loader import install_toplevel_cb_expert_loader
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    # Send wrong dtype (float32 instead of uint8)
    with pytest.raises(ValueError, match="dtype|shape"):
        m.load_weights(iter([
            ("mtp.0.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.float32)),
        ]))

# --- Additional mutation-strong miniature topology and real-path tests ---

def test_miniature_topology_non_43_non_3():
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 5, "n_mtp_layers": 2, "dspark_block_size": 5},
        "config_groups": {
            "draft": {
                "targets": ["mtp.0.ffn.experts.gate_up_proj", "mtp.1.ffn.experts.gate_up_proj", "mtp.0.attn.fused_wqa_wkv"],
                "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["cb0","cb1"]}
            }
        },
        "ignore": [],
        "provenance": {}
    }
    c = PrismaQuantConfig.from_config(cfg)
    # Mock vllm config to satisfy topology cross-check
    import types
    from unittest import mock
    mock_vcfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            draft_model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5, n_mtp_layers=2, dspark_block_size=5, dspark_target_layer_ids=[0])),
            num_speculative_tokens=5
        ),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5)),
        quant_config=None
    )
    fake_qc = types.SimpleNamespace(get_name=lambda: "gridbook")
    with mock.patch("vllm.config.get_current_vllm_config_or_none", return_value=mock_vcfg):
        with mock.patch("vllm.model_executor.models.utils.get_draft_quant_config", return_value=fake_qc):
            c._ensure_resolved()
    # Construction stages should be 5,6 not 43/44/45
    assert "model.layers.5.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.6.ffn.experts.gate_up_proj" in c.target_scheme
    assert "model.layers.43.ffn.experts.gate_up_proj" not in c.target_scheme
    # Registered stages are 0,1 ; 43 should not appear
    assert c._scheme_for_prefix("model.layers.5.attn.fused_wqa_wkv") is not None
    assert c._scheme_for_prefix("model.layers.3.ffn.experts.gate_up_proj") is None
    # Unknown stage far beyond old +5 window must fail (e.g., 10 when nh=5, n_mtp=2, allowed 5,6)
    with pytest.raises(ValueError, match="unknown stage"):
        c._scheme_for_prefix("model.layers.10.ffn.experts.gate_up_proj")
    with pytest.raises(ValueError, match="unknown stage"):
        c._moe_scheme_for_prefix("model.layers.12.ffn.experts")

def test_real_target_remapper_once_and_no_double_remap():
    """Delegated tensors must reach target via raw physical name, not pre-remapped."""
    import pathlib
    text = pathlib.Path("/target-vllm/vllm/models/deepseek_v4/nvidia/dspark.py").read_text()
    assert "mtp.(\\d+)\\.(.*)" in text or 'mtp\\.' in text
    assert "_remap_dspark_name" in text
    assert "model.layers.{stage}" in text
    # Also test via wrapper delegation: feeding mtp.0.main_proj.weight results in model.main_proj.weight being loaded via raw delegation
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    t_proj = torch.full((4096,12288), 7, dtype=torch.float8_e4m3fn)
    t_scale = torch.full((32,96), 9, dtype=torch.float8_e8m0fnu)
    tensors = []
    for stage in (0,1,2):
        tensors.append((f"mtp.{stage}.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8)))
        tensors.append((f"mtp.{stage}.ffn.experts.gate_up_proj.weight_scale", torch.zeros(E,2*INTER, dtype=torch.float32)))
        tensors.append((f"mtp.{stage}.ffn.experts.down_proj.cb_qweight", torch.zeros(E,HID,BYTES, dtype=torch.uint8)))
        tensors.append((f"mtp.{stage}.ffn.experts.down_proj.weight_scale", torch.zeros(E,HID, dtype=torch.float32)))
    tensors.append(("mtp.0.main_proj.weight", t_proj))
    tensors.append(("mtp.0.main_proj.scale", t_scale))
    tensors.append(("mtp.2.norm.weight", torch.ones(HID)))
    tensors.append(("mtp.2.hc_head_scale", torch.ones(1)))
    loaded = m.load_weights(iter(tensors))
    assert "model.main_proj.weight" in loaded
    assert torch.all(m._params["model.main_proj.weight"] == 7)
    from gridbook.config import _remap_dspark_physical_to_registered as reg_map, _remap_dspark_physical_to_construction as cons_map
    assert reg_map("mtp.0.ffn.experts.gate_up_proj", (0,1,2)) == "model.layers.0.ffn.experts.gate_up_proj"
    assert cons_map("mtp.0.ffn.experts.gate_up_proj", 5, (0,1)) == "model.layers.5.ffn.experts.gate_up_proj"

def test_head_stage_pinning_rejects_alias():
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    # Config with mtp.1.main_proj should be rejected (only stage 0 allowed) but need contiguous stages 0,1 to pass first check
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 5, "n_mtp_layers": 2},
        "config_groups": {
            "g": {"targets": ["mtp.0.ffn.experts.gate_up_proj", "mtp.1.main_proj", "mtp.1.ffn.experts.gate_up_proj"], "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}}
        },
        "ignore": []
    }
    c = PrismaQuantConfig.from_config(cfg)
    with pytest.raises(ValueError, match="main_proj"):
        c._ensure_resolved()
    # Loader should also reject mtp.1.main_proj.weight at load time
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    with pytest.raises(ValueError, match="main_proj"):
        m.load_weights(iter([("mtp.1.main_proj.weight", torch.zeros(10))]))

def test_dense_and_shared_and_router_and_heads_all_families():
    # Ensure every leaf family under config can be resolved without hardcode
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 5, "n_mtp_layers": 2},
        "config_groups": {
            "experts": {"targets": ["mtp.0.ffn.experts.gate_up_proj","mtp.0.ffn.experts.down_proj","mtp.1.ffn.experts.gate_up_proj","mtp.1.ffn.experts.down_proj"], "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}},
            "dense": {"targets": ["mtp.0.attn.fused_wqa_wkv","mtp.0.attn.wq_b","mtp.1.attn.fused_wqa_wkv"], "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}},
            "shared": {"targets": ["mtp.0.ffn.shared_experts.gate_up_proj","mtp.0.ffn.shared_experts.down_proj"], "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}},
        },
        "ignore": []
    }
    c = PrismaQuantConfig.from_config(cfg)
    import types
    from unittest import mock
    mock_vcfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(draft_model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5, n_mtp_layers=2, dspark_block_size=5)), num_speculative_tokens=5),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5)),
        quant_config=None
    )
    fake_qc = types.SimpleNamespace(get_name=lambda: "gridbook")
    with mock.patch("vllm.config.get_current_vllm_config_or_none", return_value=mock_vcfg):
        with mock.patch("vllm.model_executor.models.utils.get_draft_quant_config", return_value=fake_qc):
            c._ensure_resolved()
    assert c._scheme_for_prefix("model.layers.5.attn.fused_wqa_wkv") is not None
    assert c._scheme_for_prefix("model.layers.6.attn.fused_wqa_wkv") is not None
    assert c._moe_scheme_for_prefix("model.layers.5.ffn.experts") is not None
    assert c._moe_scheme_for_prefix("model.layers.6.ffn.experts") is not None
    # Shared expert is dense linear, check via _scheme_for_prefix (shared_experts is alias but after alias collapsed)
    assert c._scheme_for_prefix("model.layers.5.ffn.shared_experts.gate_up_proj") is not None

def test_unknown_stage_far_beyond_window_fails_closed():
    from gridbook.moe_toplevel_loader import _INSTALLED_MODULE_PATHS
    _INSTALLED_MODULE_PATHS.add("vllm.models.deepseek_v4.nvidia.dspark")
    cfg = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "dspark_topology": {"num_hidden_layers": 5, "n_mtp_layers": 2},
        "config_groups": {
            "g": {"targets": ["mtp.0.ffn.experts.gate_up_proj","mtp.1.ffn.experts.gate_up_proj"], "scheme": {"grid":"fp4","mode":"product","k":15,"n_sub":2,"type_size":69,"group_size":16,"vec_dim":8,"scale_coding":{"kind":"two_tier"},"codebook_ref":["a","b"]}}
        },
        "ignore": []
    }
    c = PrismaQuantConfig.from_config(cfg)
    import types
    from unittest import mock
    mock_vcfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(draft_model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5, n_mtp_layers=2, dspark_block_size=5)), num_speculative_tokens=5),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=5)),
        quant_config=None
    )
    fake_qc = types.SimpleNamespace(get_name=lambda: "gridbook")
    with mock.patch("vllm.config.get_current_vllm_config_or_none", return_value=mock_vcfg):
        with mock.patch("vllm.model_executor.models.utils.get_draft_quant_config", return_value=fake_qc):
            c._ensure_resolved()
    # Stage 5 is allowed, 6 allowed, 100 is far beyond old +5 window (5+2+5=12) but now must still fail
    with pytest.raises(ValueError, match="unknown stage"):
        c._scheme_for_prefix("model.layers.100.ffn.experts.gate_up_proj")
    # Loader should also fail for mtp.10 physical far beyond
    Cls = _make_fake_dspark_cls(_dparams())
    install_toplevel_cb_expert_loader(Cls)
    m = Cls()
    with pytest.raises(ValueError, match="unknown stage"):
        m.load_weights(iter([("mtp.10.ffn.experts.gate_up_proj.cb_qweight", torch.zeros(E,2*INTER,BYTES, dtype=torch.uint8))]))
