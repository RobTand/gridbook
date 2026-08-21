"""Capture-safety gates for ``_padded_route`` (RobTand/gridbook#47).

THE CONTRACT PROVEN HERE. ``_padded_route`` has exactly two host reads — the
optional trim count and the optional per-expert ``block_offsets`` — and both
are ROUTING-DEPENDENT, so neither can execute inside CUDA-graph capture: the
runtime forbids the copy, and even a permitted read would bake one
capture-time routing's value into every replay. Under capture the function
must therefore (a) force ``trim=False`` and return the static-capacity layout
whose shape is known from shapes alone, byte-identical to the eager
``trim=False`` arm, and (b) refuse ``block_offsets`` with an error that names
the remediation. Eager behaviour must be unchanged.

INSTRUMENTS. The CPU tier drives the ``_capturing_now`` seam directly and
poisons ``torch.Tensor.tolist`` / ``torch.Tensor.item`` so any surviving host
read fails the test rather than passing silently — with a control assert that
the poison actually fires. The GPU tier captures ``_padded_route`` inside a
real ``torch.cuda.graph``, replays it against mutated routing, and keeps a
negative control proving the trim read really does abort capture on this
torch (skipping, not failing, if a future torch permits it — at which point
the guard's justification should be revisited).

vLLM is stubbed (``gridbook.moe`` imports its FusedMoE surface at module
top); the import of ``gridbook.moe`` itself is REAL and unguarded, so a
broken module import fails this file loudly instead of skipping it green —
the 0.8.9 lesson.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

CUDA_OK = torch.cuda.is_available()


def _install_vllm_stubs():
    """Install only the native MoE class surface imported by Gridbook."""

    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value

    module("vllm")
    module("vllm.model_executor")
    utils = module("vllm.model_executor.utils")
    utils.set_weight_attrs = lambda param, attrs: [
        setattr(param, name, value) for name, value in attrs.items()
    ]
    module("vllm.model_executor.layers")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    fused = module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = type("RoutedExperts", (), {})
    config = module("vllm.model_executor.layers.fused_moe.config")
    config.FusedMoEConfig = type("FusedMoEConfig", (), {})
    config.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
    base = module(
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
    )
    base.FusedMoEMethodBase = type(
        "FusedMoEMethodBase",
        (),
        {"__init__": lambda self, moe_config=None: None},
    )
    parameter = module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **_kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **_kwargs):
            del data

    parameter.ModelWeightParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    """Import ``gridbook.moe`` against a private stubbed vLLM graph."""
    del isolated_gridbook_runtime_imports
    _install_vllm_stubs()

    from gridbook import moe as moe_module

    globals()["moe"] = moe_module
    yield


def _skewed_routing(device="cpu"):
    """A routing where trimming matters: empty experts, uneven counts.

    E=8, T=24, top_k=2 -> P=48 pairs; tile_m=8 gives cap_blocks = 48//8 + 8
    = 14 while the real total is far smaller, so the trimmed and untrimmed
    layouts differ in shape and the pad tail is non-empty.
    """
    E, T, top_k, tile_m = 8, 24, 2, 8
    torch.manual_seed(4247)
    # Concentrate on experts {0, 3, 5}; experts 1, 2, 4, 6, 7 get nothing.
    choices = torch.tensor([0, 3, 5], dtype=torch.int64)
    topk_ids = choices[torch.randint(0, 3, (T, top_k))].to(device)
    topk_weights = torch.rand(T, top_k, device=device)
    return topk_ids, topk_weights, E, tile_m


class _Poisoned(AssertionError):
    pass


def _poison_host_reads(monkeypatch):
    """Make any ``.tolist()`` / ``.item()`` raise, and prove the poison bites."""

    def _raise(self, *a, **k):
        raise _Poisoned("host read (tolist/item) executed under capture")

    monkeypatch.setattr(torch.Tensor, "tolist", _raise)
    monkeypatch.setattr(torch.Tensor, "item", _raise)
    with pytest.raises(_Poisoned):
        torch.zeros(1).item()
    with pytest.raises(_Poisoned):
        torch.zeros(1).tolist()


def _routes_equal(a, b):
    assert torch.equal(a.expert_ids, b.expert_ids)
    assert torch.equal(a.row_src, b.row_src)
    assert torch.equal(a.is_pad, b.is_pad)
    assert torch.equal(a.dest, b.dest)
    assert torch.equal(a.pw_sorted, b.pw_sorted)


# ===========================================================================
# CPU tier: the seam, both arms.
# ===========================================================================
def test_capture_forces_the_static_capacity_layout(monkeypatch):
    """Under capture, ``trim=True`` degrades to the static-capacity layout
    with no host read, bit-identical to the eager ``trim=False`` arm."""
    topk_ids, topk_weights, E, tile_m = _skewed_routing()
    reference = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=False)

    monkeypatch.setattr(moe, "_capturing_now", lambda t: True)
    _poison_host_reads(monkeypatch)
    captured = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=True)

    cap_blocks = topk_ids.numel() // tile_m + E
    assert captured.expert_ids.numel() == cap_blocks
    assert captured.block_offsets is None
    _routes_equal(captured, reference)


def test_capture_refuses_the_bridge_offsets(monkeypatch):
    """``block_offsets`` under capture is refused with lane + remediation."""
    topk_ids, topk_weights, E, tile_m = _skewed_routing()
    monkeypatch.setattr(moe, "_capturing_now", lambda t: True)
    _poison_host_reads(monkeypatch)
    with pytest.raises(RuntimeError, match="gridbook#47") as err:
        moe._padded_route(
            topk_ids, topk_weights, E, tile_m,
            trim=True, block_offsets=True, pack_group=8)
    message = str(err.value)
    assert "BF16 grouped bridge" in message
    assert "FULL_DECODE_ONLY" in message
    assert "--enforce-eager" in message
    assert str(moe.MOE_PREFILL_M_THRESHOLD) in message


def test_eager_trim_still_trims():
    """Outside capture nothing changes: ``trim=True`` really slices to the
    real block count, which for the skewed routing is smaller than capacity."""
    topk_ids, topk_weights, E, tile_m = _skewed_routing()
    from gridbook.moe_routing import cb_grouped_block_offsets

    real_blocks = int(cb_grouped_block_offsets(topk_ids, E, tile_m)[E])
    cap_blocks = topk_ids.numel() // tile_m + E
    assert real_blocks < cap_blocks, "fixture must make trimming observable"

    trimmed = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=True)
    assert trimmed.expert_ids.numel() == real_blocks

    bridge = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=True, block_offsets=True)
    assert bridge.block_offsets is not None
    assert bridge.block_offsets[E] == real_blocks


def test_static_capacity_is_the_trimmed_layout_plus_pad_tail():
    """The property that makes forcing ``trim=False`` bit-neutral for real
    outputs: the untrimmed layout is the trimmed layout followed by pure
    padding (expert_ids -1, is_pad True, dest = throwaway row T)."""
    topk_ids, topk_weights, E, tile_m = _skewed_routing()
    T = int(topk_ids.shape[0])
    trimmed = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=True)
    full = moe._padded_route(
        topk_ids, topk_weights, E, tile_m, trim=False)

    nb = trimmed.expert_ids.numel()
    assert torch.equal(full.expert_ids[:nb], trimmed.expert_ids)
    assert torch.equal(full.row_src[:nb * tile_m], trimmed.row_src)
    assert torch.equal(full.is_pad[:nb * tile_m], trimmed.is_pad)
    assert torch.equal(full.dest[:nb * tile_m], trimmed.dest)
    assert (full.expert_ids[nb:] == -1).all()
    assert bool(full.is_pad[nb * tile_m:].all())
    assert (full.dest[nb * tile_m:] == T).all()


def test_the_seam_is_false_on_cpu_tensors():
    """``_capturing_now`` must not touch the CUDA runtime for CPU tensors."""
    assert moe._capturing_now(torch.zeros(1)) is False


# ===========================================================================
# GPU tier: a real capture, a real replay, and the negative control.
# ===========================================================================
@pytest.mark.skipif(not CUDA_OK, reason="needs CUDA")
def test_padded_route_captures_and_replays_with_new_routing():
    """The #47 repro, fixed: capture ``_padded_route(trim=True)`` inside a
    real graph (previously: "operation not permitted when stream is
    capturing"), then rewrite the routing in place and replay. The replayed
    outputs must equal an eager static-capacity recompute of the NEW routing
    — the property a baked-in trim count would violate."""
    dev = torch.device("cuda")
    topk_ids, topk_weights, E, tile_m = _skewed_routing(device=dev)

    ids_buf = topk_ids.clone()
    w_buf = topk_weights.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route = moe._padded_route(ids_buf, w_buf, E, tile_m, trim=True)
    assert route.block_offsets is None
    cap_blocks = ids_buf.numel() // tile_m + E
    assert route.expert_ids.numel() == cap_blocks

    # New routing, same shapes: shift every expert id to a different live set.
    remap = {0: 5, 3: 0, 5: 3}
    new_ids = topk_ids.cpu().apply_(lambda e: remap[e]).to(dev)
    new_w = torch.rand_like(topk_weights)
    ids_buf.copy_(new_ids)
    w_buf.copy_(new_w)
    graph.replay()
    torch.cuda.synchronize()

    eager = moe._padded_route(new_ids, new_w, E, tile_m, trim=False)
    _routes_equal(route, eager)


@pytest.mark.skipif(not CUDA_OK, reason="needs CUDA")
def test_the_trim_read_is_the_thing_capture_forbids():
    """Negative control: prove the read the guard avoids really does abort
    capture on this torch. Skips rather than fails if a future torch permits
    it — at which point ``_padded_route``'s capture guard should be
    revisited (the CORRECTNESS argument — a baked-in routing-dependent count
    — still stands even then)."""
    n_blocks = torch.tensor([3], device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            int(n_blocks.item())
    except RuntimeError:   # torch.AcceleratorError subclasses RuntimeError
        return
    pytest.skip("this torch permits .item() during capture; the #47 guard "
                "is now justified by replay correctness alone")
