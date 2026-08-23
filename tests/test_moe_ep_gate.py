"""The expert-parallel admission gate for CB MoE expert stacks.

Companion to ``test_tp_dense_shard.py`` section 3 (the refusal lattice), on
the axis that admits rather than refuses. Above one rank a stacked CB MoE
layer is served ONLY under ``--enable-expert-parallel`` with every other
parallel axis at 1; every other topology refuses at method construction,
before a buffer exists, naming itself and the flag that would fix it.

The matrix here is over vLLM's ``FusedMoEParallelConfig``, constructed
directly rather than through an engine: there is one GPU in this box and no
second rank. What this proves is the DISPATCH decision — which topologies
Gridbook admits and which it refuses, and what the refusal says. It does not
prove a distributed run; no engine, no collectives, no second device.

Run: ``python -m pytest tests/test_moe_ep_gate.py -q``.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("vllm")

from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: E402

from gridbook import config as config_mod  # noqa: E402
from gridbook.config import PrismaQuantConfig  # noqa: E402


_L0 = "model.layers.0"
_SCHEME = {
    "grid": "fp8",
    "k": 28,
    "n_sub": 4,
    "type_size": 112,
    "codebook_ref": "cb0",
}


def _world_size(monkeypatch, size):
    monkeypatch.setattr(
        config_mod, "_initialized_tensor_parallel_world_size",
        lambda: size)


def _parallel(*, use_ep=True, tp_size=1, dp_size=1, pcp_size=1, sp_size=1,
              ep_size=2, enable_eplb=False):
    """A FusedMoEParallelConfig with exactly the axes this gate reads."""
    return types.SimpleNamespace(
        use_ep=use_ep,
        tp_size=tp_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        sp_size=sp_size,
        ep_size=ep_size,
        ep_rank=0,
        enable_eplb=enable_eplb,
        # vLLM's own derivation, restated so a topology change here cannot
        # silently disagree with the property the gate reads.
        use_all2all_kernels=bool(
            use_ep and (dp_size > 1 or pcp_size > 1 or sp_size > 1)),
    )


def _experts(parallel=None, *, num_experts=256, num_local=128,
             skip_final_all_reduce=False):
    layer = object.__new__(RoutedExperts)
    layer.moe_config = types.SimpleNamespace(
        moe_parallel_config=parallel,
        num_experts=num_experts,
        num_local_experts=num_local if parallel is not None else num_experts,
        skip_final_all_reduce=skip_final_all_reduce,
    )
    return layer


def _artifact_config():
    return {
        "quant_method": "gridbook",
        "format": "nvfp4-cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "experts": {
                "format": "FP8_CB_K28",
                "scheme": dict(_SCHEME),
                "targets": [f"{_L0}.mlp.experts.gate_up_proj",
                            f"{_L0}.mlp.experts.down_proj"],
            },
        },
        "ignore": ["lm_head"],
    }


def _resolved_config():
    cfg = PrismaQuantConfig.from_config(_artifact_config())
    cfg._ensure_resolved()
    return cfg


def _gate(monkeypatch, world, parallel, **kwargs):
    """Run just the topology predicate, returning the admitted mode."""
    _world_size(monkeypatch, world)
    cfg = _resolved_config()
    return cfg._require_ep_moe_serving(
        "CB MoE expert stacks (stacked whole-tensor loader)",
        f"{_L0}.mlp.experts", _experts(parallel, **kwargs))


# --- admitted ----------------------------------------------------------------


def test_single_rank_is_admitted_and_unchanged(monkeypatch):
    assert _gate(monkeypatch, 1, None) == "single_rank"


def test_uninitialized_parallelism_behaves_as_single_rank(monkeypatch):
    monkeypatch.setattr(
        config_mod, "_initialized_tensor_parallel_world_size", lambda: None)
    cfg = _resolved_config()
    assert cfg._require_ep_moe_serving(
        "s", f"{_L0}.mlp.experts", _experts(None)) == "single_rank"


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_pure_expert_parallelism_is_admitted(monkeypatch, ep_size):
    mode = _gate(monkeypatch, ep_size, _parallel(ep_size=ep_size))
    assert mode == f"expert_parallel(ep_size={ep_size})"


def test_admission_survives_a_non_contiguous_placement(monkeypatch):
    """Placement is a LOADER law, not a gate input; the gate is topology."""
    assert _gate(monkeypatch, 2, _parallel()).startswith("expert_parallel")


# --- refused -----------------------------------------------------------------


def test_tp_without_expert_parallel_refuses_and_names_the_flag(monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        _gate(monkeypatch, 2, _parallel(use_ep=False, tp_size=2))
    message = str(excinfo.value)
    assert "--enable-expert-parallel" in message
    assert "expert stacks" in message and "TP=2" in message
    assert f"{_L0}.mlp.experts" in message
    # It must say WHY tensor parallelism cannot work here, not just no.
    assert "superblock" in message


def test_expert_parallel_with_a_still_tensor_parallel_moe_refuses(monkeypatch):
    with pytest.raises(ValueError, match="moe tp_size=2"):
        _gate(monkeypatch, 4, _parallel(tp_size=2))


@pytest.mark.parametrize("axis", ["dp_size", "pcp_size", "sp_size"])
def test_all2all_expert_parallel_topologies_refuse(monkeypatch, axis):
    with pytest.raises(ValueError) as excinfo:
        _gate(monkeypatch, 4, _parallel(**{axis: 2}))
    message = str(excinfo.value)
    assert "all2all" in message
    assert f"{axis}=2" in message
    assert "dispatch/combine" in message


def test_eplb_refuses(monkeypatch):
    with pytest.raises(ValueError, match="EPLB"):
        _gate(monkeypatch, 2, _parallel(enable_eplb=True))


def test_skip_final_all_reduce_refuses(monkeypatch):
    """Gridbook returns this rank's PARTIAL; something must sum the ranks."""
    with pytest.raises(ValueError, match="skip_final_all_reduce"):
        _gate(monkeypatch, 2, _parallel(), skip_final_all_reduce=True)


def test_an_unreadable_parallel_config_refuses(monkeypatch):
    with pytest.raises(ValueError, match="unreadable"):
        _gate(monkeypatch, 2, None)


# --- T6: mixed per-expert-format stacks stay refused --------------------------


def test_mixed_format_expert_stacks_refuse_under_expert_parallelism(
        monkeypatch):
    """A format partition declared over GLOBAL expert ids cannot be split.

    The refusal must also say what DOES serve above one rank, or an operator
    reads it as "no multi-rank MoE at all" and stops.
    """
    _world_size(monkeypatch, 2)
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "gridbook",
        "format": "nvfp4-cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "experts": {
                "format": "FP8_CB_K28",
                "scheme": dict(_SCHEME),
                "targets": [
                    f"{_L0}.mlp.experts.gate_up_proj.format_group_0",
                    f"{_L0}.mlp.experts.down_proj.format_group_0",
                ],
            },
            "experts_fp4": {
                "format": "NVFP4_CB_K16",
                "scheme": {"grid": "fp4", "k": 16, "n_sub": 2,
                           "type_size": 73, "codebook_ref": "cb1"},
                "targets": [
                    f"{_L0}.mlp.experts.gate_up_proj.format_group_1",
                    f"{_L0}.mlp.experts.down_proj.format_group_1",
                ],
            },
        },
        "per_expert_format_groups": {
            "version": 1,
            "layers": {
                "0": {
                    "w13": [
                        {"format_wire_id": "FP8_CB_K28",
                         "expert_ids": [0, 1],
                         "tensor_prefix": f"{_L0}.mlp.experts.gate_up_proj.format_group_0"},
                        {"format_wire_id": "NVFP4_CB_K16",
                         "expert_ids": [2, 3],
                         "tensor_prefix": f"{_L0}.mlp.experts.gate_up_proj.format_group_1"},
                    ],
                    "w2": [
                        {"format_wire_id": "FP8_CB_K28",
                         "expert_ids": [0, 1],
                         "tensor_prefix": f"{_L0}.mlp.experts.down_proj.format_group_0"},
                        {"format_wire_id": "NVFP4_CB_K16",
                         "expert_ids": [2, 3],
                         "tensor_prefix": f"{_L0}.mlp.experts.down_proj.format_group_1"},
                    ],
                },
            },
        },
    })
    cfg._ensure_resolved()
    layer = _experts(_parallel(), num_experts=4, num_local=2)
    layer.moe_config.num_experts = 4
    with pytest.raises(ValueError) as excinfo:
        cfg.get_quant_method(layer, f"{_L0}.mlp.experts")
    message = str(excinfo.value)
    assert "per-expert format groups" in message
    assert "--enable-expert-parallel" in message
    assert "GLOBAL expert ids" in message
