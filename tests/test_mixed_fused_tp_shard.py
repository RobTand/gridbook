"""Mixed-format fused projections above one tensor-parallel rank.

DeepSeek-V4's shipped 92 GB artifact has EIGHT shared-expert
``gate_up_proj`` modules whose two roles do not share one method: four fuse a
CB ``gate_proj`` with a source-passthrough block-FP8 ``up_proj`` (layers 0,
19, 21, 22), and four fuse two CB roles at different rungs, K48 with K44
(layers 12, 13, 18, 32).  vLLM presents each as ONE
``MergedColumnParallelLinear``; Gridbook gives each role its own carrier and
its own existing method.  Above one rank the composer therefore has exactly
two jobs, and this file pins both:

1. **Construct at the role's own law.**  Each carrier is built with the
   ROLE's whole-tensor output size, so the role's existing shard gate — the
   CB ``ShardGroupAlignmentError`` and the source lane's
   ``ShardAlignmentError`` — decides legality, before any parameter exists.
   No alignment quantum is restated here.
2. **Narrow like the role's standalone loader.**  Gridbook's top-level
   router owns these loads (the checkpoint planes are whole and are addressed
   by role name, not by fused name), so it must place the same bytes vLLM's
   own ``load_column_parallel_weight`` would place for a standalone Linear of
   that role.  Every assertion below drives vLLM's real parameter classes as
   the reference rather than restating their arithmetic.

**Scope — what these tests do NOT establish.**  One process, one device, no
engine and no collectives: the ranks are simulated by pinning the global
rank/world-size accessors and running construction twice, exactly the idiom
``test_tp_dense_shard.py`` and ``test_fp8_source_w8a16_tp_shard.py`` already
use.  The numeric identity below is measured on the DECODED weight planes and
fp32 reference products, which isolates the shard law from the serving
kernel; it is not a served two-rank measurement.
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm.model_executor.parameter as _vllm_parameter  # noqa: E402

from cb_torch_reference import (  # noqa: E402
    E2M1_MAGNITUDES,
    reconstruct_cb_weight,
)

from gridbook.fp8_source_w8a16 import (  # noqa: E402
    ShardAlignmentError,
    WIRE_FP8_BLOCK128,
    build_fp8_source_w8a16_method,
)
from gridbook.linear import (  # noqa: E402
    PrismaQuantCBLinearMethod,
    ShardGroupAlignmentError,
)
from gridbook.mixed_linear import (  # noqa: E402
    MixedFusedLinearMethod,
    MixedFusedShardError,
)
from gridbook.moe_toplevel_loader import _MixedFusedTransactions  # noqa: E402


# --- the shipped DSv4 geometry ----------------------------------------------
#
# model.layers.{0,19,21,22}.mlp.shared_experts.gate_up_proj: a 2048-wide CB
# gate role (config group_0, FP8_CB_K48) fused with a 2048-wide block-FP8
# source-passthrough up role, over K = 4096.

K_FULL = 4096
ROLE_N = 2048
CB_K_BITS, CB_N_SUB, CB_TYPE_SIZE = 48, 4, 192
SUPERBLOCK = 256
DS_BLOCK = 128
GATE = "model.layers.0.ffn.shared_experts.w1"
UP = "model.layers.0.ffn.shared_experts.w3"
FUSED = "model.layers.0.ffn.shared_experts.gate_up_proj"

CB_SCHEME = {
    "grid": "fp8", "mode": "product", "k": CB_K_BITS, "n_sub": CB_N_SUB,
    "type_size": CB_TYPE_SIZE, "group_size": 0, "vec_dim": 8,
    "codebook_ref": [f"cb.fp8.k{CB_K_BITS}.sub{i}" for i in range(CB_N_SUB)],
}
CB_ROW_BYTES = (K_FULL // SUPERBLOCK) * CB_TYPE_SIZE


def _cb_method(prefix: str = GATE) -> PrismaQuantCBLinearMethod:
    return PrismaQuantCBLinearMethod(types.SimpleNamespace(), CB_SCHEME,
                                     prefix)


def _source_method():
    return build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)


def _roles():
    """The shipped role order: CB gate first, source-passthrough up second."""
    return [(GATE, _cb_method(GATE)), (UP, _source_method())]


@pytest.fixture(autouse=True)
def _pin_rank_zero(monkeypatch):
    """Parameters read the process-global TP group at ``__init__``.

    Pinned to rank 0 / world 1 by default so construction works outside an
    engine; every test that needs another rank re-pins it explicitly through
    ``_as_rank``.
    """
    monkeypatch.setattr(_vllm_parameter, "get_tensor_model_parallel_rank",
                        lambda: 0)
    monkeypatch.setattr(
        _vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1)


def _as_rank(monkeypatch, rank: int, world: int) -> None:
    """Present this process as rank *rank* of *world* to BOTH accessors.

    vLLM's parameter classes read ``vllm.model_executor.parameter``'s binding
    at construction; the mixed composer reads ``vllm.distributed``'s.  A test
    that pinned only one of them would prove the composer agrees with itself.
    """
    import vllm.distributed as _vllm_distributed

    monkeypatch.setattr(_vllm_parameter, "get_tensor_model_parallel_rank",
                        lambda: rank)
    monkeypatch.setattr(
        _vllm_parameter, "get_tensor_model_parallel_world_size",
        lambda: world)
    monkeypatch.setattr(_vllm_distributed,
                        "get_tensor_model_parallel_rank", lambda: rank)


# --- checkpoint fixtures -----------------------------------------------------
#
# Whole-tensor planes, exactly as the artifact stores them: the router is
# handed these and owns the shard itself.


def _cb_planes(seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    qweight = torch.randint(0, 256, (ROLE_N, CB_ROW_BYTES), dtype=torch.uint8,
                            generator=generator)
    scale = torch.rand(ROLE_N, generator=generator) * 3 + 0.25
    return qweight, scale


def _cb_codebook(seed: int = 6):
    """A format-legal product codebook for k=48, n_sub=4 (four 12-bit subs)."""
    generator = torch.Generator().manual_seed(seed)
    grid = torch.tensor(
        sorted({v for m in E2M1_MAGNITUDES for v in (m, -m)}),
        dtype=torch.float32)
    cols = 8 // CB_N_SUB
    tables = []
    for _ in range(CB_N_SUB):
        pick = torch.randint(0, grid.numel(), ((1 << 12) * cols,),
                             generator=generator)
        tables.append(grid[pick].reshape(1 << 12, cols))
    return torch.cat([t.reshape(-1) for t in tables]).contiguous()


def _cb_decode(qweight, scale, cb_flat):
    rows = int(qweight.shape[0])
    return reconstruct_cb_weight(
        qweight, cb_flat, torch.zeros(rows, dtype=torch.int32), scale,
        torch.zeros(1), N=rows, K=K_FULL, k_bits=CB_K_BITS, n_sub=CB_N_SUB,
        type_size=CB_TYPE_SIZE, is_fp4=False, is_v2=False)


def _source_planes(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(ROLE_N, K_FULL, generator=generator) * 0.05
    q = values.to(torch.float8_e4m3fn)
    raw = torch.randint(120, 134,
                        (ROLE_N // DS_BLOCK, K_FULL // DS_BLOCK),
                        generator=generator, dtype=torch.uint8)
    return q, raw.view(torch.float8_e8m0fnu)


def _source_decode(q, scales):
    """E4M3 values times their UE8M0 block-128 exponents, in fp32.

    Restates the wire format rather than importing the lane's own decoder: a
    shard law proven against the lane's decoder would prove self-consistency.
    """
    n, k = q.shape
    exponents = scales.view(torch.uint8).to(torch.int32) - 127
    factors = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32),
                          exponents)
    full = factors.repeat_interleave(DS_BLOCK, dim=0).repeat_interleave(
        DS_BLOCK, dim=1)
    return q.to(torch.float32) * full[:n, :k]


def _checkpoint(cb, source):
    qweight, scale = cb
    q, scales = source
    return {
        GATE + ".cb_qweight": qweight,
        GATE + ".weight_scale": scale,
        UP + ".weight": q,
        UP + ".weight_scale_inv": scales,
    }


# --- construction + load -----------------------------------------------------


def _build_mixed(monkeypatch, *, rank: int, degree: int,
                 role_widths=(ROLE_N, ROLE_N), k_full: int = K_FULL,
                 roles=None):
    """Construct the fused module as rank *rank* of a *degree*-way split."""

    _as_rank(monkeypatch, rank, degree)
    method = MixedFusedLinearMethod(FUSED, _roles() if roles is None
                                    else roles)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        k_full,
        [width // degree for width in role_widths],
        k_full,
        sum(role_widths),
        torch.bfloat16,
    )
    return method, layer


def _load_mixed(layer, checkpoint):
    params = dict(layer.named_parameters())
    transactions = _MixedFusedTransactions(params)
    for name, plane in checkpoint.items():
        transactions.stage(name, plane)
    transactions.finish()
    return {name.rsplit(".", 1)[-1]: param for name, param in params.items()
            for _ in (0,)}


def _carrier_planes(layer, index: int):
    carrier = list(layer._gridbook_mixed_roles)[index]
    return dict(carrier.named_parameters(recurse=False))


def _standalone(monkeypatch, method, *, rank: int, degree: int,
                width: int = ROLE_N, k_full: int = K_FULL):
    """The role built as its OWN column-parallel Linear on rank *rank*."""

    _as_rank(monkeypatch, rank, degree)
    layer = torch.nn.Module()
    layer.prefix = "standalone"
    method.create_weights(layer, k_full, [width // degree], k_full, width,
                          torch.bfloat16,
                          weight_loader=lambda *a, **k: None)
    return layer


# ===========================================================================
# 1. Rank-slice byte identity, against vLLM's own loader.
# ===========================================================================

@pytest.mark.parametrize("degree", [2, 4])
def test_every_carrier_plane_equals_the_standalone_vllm_narrow(
        monkeypatch, degree):
    """Acceptance 1: per plane, per role, per rank — the same bytes.

    The reference is not arithmetic restated here: it is vLLM's real
    ``load_column_parallel_weight`` running on a parameter the ROLE's own
    method built for a standalone Linear of that role on the same rank.
    """

    cb = _cb_planes()
    source = _source_planes()
    checkpoint = _checkpoint(cb, source)

    for rank in range(degree):
        _method, layer = _build_mixed(monkeypatch, rank=rank, degree=degree)
        _load_mixed(layer, checkpoint)

        references = [
            _standalone(monkeypatch, _cb_method(GATE), rank=rank,
                        degree=degree),
            _standalone(monkeypatch, _source_method(), rank=rank,
                        degree=degree),
        ]
        for plane_name, whole in (
                ("cb_qweight", cb[0]), ("weight_scale", cb[1])):
            getattr(references[0], plane_name).load_column_parallel_weight(
                whole)
        for plane_name, whole in (
                ("weight", source[0]), ("weight_scale_inv", source[1])):
            getattr(references[1], plane_name).load_column_parallel_weight(
                whole)

        for index, reference in enumerate(references):
            got = _carrier_planes(layer, index)
            want = dict(reference.named_parameters(recurse=False))
            assert set(got) == set(want), (
                f"role {index} plane set differs at rank {rank}")
            for plane_name, param in got.items():
                assert torch.equal(
                    param.data.view(torch.uint8),
                    want[plane_name].data.view(torch.uint8)), (
                    f"role {index} plane {plane_name!r} differs from the "
                    f"standalone vLLM narrow on rank {rank} of {degree}")


@pytest.mark.parametrize("degree", [2, 4])
def test_rank_planes_partition_the_checkpoint_without_overlap(
        monkeypatch, degree):
    """The ranks' slices tile the whole plane exactly once, in rank order.

    Byte identity against a reference on each rank does not by itself say the
    ranks COVER the tensor: a loader that gave every rank rank 0's slice would
    satisfy that test if the reference had the same bug.  Concatenating the
    ranks in order and comparing to the checkpoint closes it.
    """

    cb = _cb_planes()
    source = _source_planes()
    checkpoint = _checkpoint(cb, source)
    collected: dict[str, list[torch.Tensor]] = {}

    for rank in range(degree):
        _method, layer = _build_mixed(monkeypatch, rank=rank, degree=degree)
        _load_mixed(layer, checkpoint)
        for index in range(2):
            for plane_name, param in _carrier_planes(layer, index).items():
                collected.setdefault(plane_name, []).append(
                    param.data.clone())

    for plane_name, whole in (("cb_qweight", cb[0]), ("weight_scale", cb[1]),
                              ("weight", source[0]),
                              ("weight_scale_inv", source[1])):
        joined = torch.cat(collected[plane_name], dim=0)
        assert torch.equal(joined.view(torch.uint8), whole.view(torch.uint8)), (
            f"the {degree} ranks' {plane_name!r} slices do not tile the "
            "checkpoint plane")


def test_degree_one_load_is_the_whole_plane_unchanged(monkeypatch):
    """Acceptance 3: at degree 1 nothing narrows and nothing else moves."""

    cb = _cb_planes()
    source = _source_planes()
    _method, layer = _build_mixed(monkeypatch, rank=0, degree=1)
    _load_mixed(layer, _checkpoint(cb, source))
    for index, planes in ((0, {"cb_qweight": cb[0], "weight_scale": cb[1]}),
                          (1, {"weight": source[0],
                               "weight_scale_inv": source[1]})):
        got = _carrier_planes(layer, index)
        assert set(got) == set(planes)
        for plane_name, whole in planes.items():
            assert torch.equal(got[plane_name].data.view(torch.uint8),
                               whole.view(torch.uint8))


def test_degree_one_construction_never_reads_the_distributed_state(
        monkeypatch):
    """Degree 1 must stay byte-identical to the pre-TP path.

    vLLM stamps a live world size onto replicated and ``disable_tp`` merged
    planes too, so a composer that consulted the global rank at degree 1 would
    change behaviour on a TP=2 serve for a module that is not sharded at all.
    The accessor is made to raise: construction must still succeed.
    """

    import vllm.distributed as _vllm_distributed

    def _explode():
        raise AssertionError(
            "the mixed composer read the tensor-parallel rank at degree 1")

    monkeypatch.setattr(_vllm_distributed, "get_tensor_model_parallel_rank",
                        _explode)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    method.create_weights(layer, K_FULL, [ROLE_N, ROLE_N], K_FULL,
                          2 * ROLE_N, torch.bfloat16)
    assert [tuple(param.shape)
            for param in _carrier_planes(layer, 0).values()] == [
        (ROLE_N, CB_ROW_BYTES), (ROLE_N,)]


# ===========================================================================
# 2. Numeric identity of the rank-local roles.
# ===========================================================================

@pytest.mark.parametrize("degree", [2, 4])
def test_rank_roles_reconstruct_the_tp1_product_bitwise(monkeypatch, degree):
    """Acceptance 2: un-interleaving the ranks reproduces the TP=1 output.

    Measured on the decoded weight planes and an fp32 reference product, not
    on the serving kernel: a column split gives every rank a disjoint set of
    output columns and changes no reduction order, so the claim under test is
    that the identity is BITWISE.  Both roles are checked; the result is
    reported per role rather than assumed.
    """

    cb = _cb_planes()
    source = _source_planes()
    checkpoint = _checkpoint(cb, source)
    cb_flat = _cb_codebook()
    x = torch.randn(4, K_FULL, generator=torch.Generator().manual_seed(9))

    whole = [
        _cb_decode(cb[0], cb[1], cb_flat),
        _source_decode(source[0], source[1]),
    ]
    def _product(plane):
        # The CB reference decoder returns bfloat16; the product itself is
        # taken in fp32 so the comparison is about the SHARD, not rounding.
        return x @ plane.to(torch.float32).t()

    tp1 = torch.cat([_product(plane) for plane in whole], dim=-1)

    per_rank = []
    for rank in range(degree):
        _method, layer = _build_mixed(monkeypatch, rank=rank, degree=degree)
        _load_mixed(layer, checkpoint)
        gate = _carrier_planes(layer, 0)
        up = _carrier_planes(layer, 1)
        local = [
            _cb_decode(gate["cb_qweight"].data, gate["weight_scale"].data,
                       cb_flat),
            _source_decode(up["weight"].data, up["weight_scale_inv"].data),
        ]
        for index, plane in enumerate(local):
            reference = whole[index].narrow(
                0, rank * (ROLE_N // degree), ROLE_N // degree)
            assert torch.equal(plane, reference), (
                f"role {index} decoded plane on rank {rank} is not the "
                "corresponding slice of the whole-tensor decode")
        # What the fused module returns on this rank: its roles, concatenated
        # in vLLM's declared shard order.
        per_rank.append(torch.cat([_product(plane) for plane in local],
                                  dim=-1))

    # vLLM's all-gather of a merged column-parallel output re-interleaves the
    # ranks role by role; undo that to recover the TP=1 layout.
    local_width = ROLE_N // degree
    rebuilt = torch.cat([
        torch.cat([chunk.narrow(-1, index * local_width, local_width)
                   for chunk in per_rank], dim=-1)
        for index in range(2)
    ], dim=-1)
    assert torch.equal(rebuilt, tp1)


def test_apply_concatenates_the_rank_local_roles_in_shard_order(monkeypatch):
    """The composer's own contribution at rank r: the concatenation.

    Deterministic role methods isolate ``apply`` from either format's kernel;
    what is under test is that a sharded module returns this rank's roles, in
    vLLM's declared order, at the rank-local widths.
    """

    class _Constant:
        def __init__(self, value):
            self.value = value

        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, *args, **kwargs):
            layer.register_parameter("weight", torch.nn.Parameter(
                torch.full((output_partition_sizes[0],
                            input_size_per_partition), self.value),
                requires_grad=False))

        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            width = layer.weight.shape[0]
            return torch.full((*x.shape[:-1], width), self.value,
                              dtype=x.dtype, device=x.device)

    _as_rank(monkeypatch, 1, 2)
    method = MixedFusedLinearMethod(FUSED, [(GATE, _Constant(1.0)),
                                            (UP, _Constant(2.0))])
    layer = torch.nn.Module()
    method.create_weights(layer, 256, [8, 4], 256, 24, torch.bfloat16)
    out = method.apply(layer, torch.zeros(3, 256))
    assert out.shape == (3, 12)
    assert torch.equal(out, torch.cat([torch.ones(3, 8),
                                       torch.full((3, 4), 2.0)], dim=-1))


# ===========================================================================
# 3. Refusals, before any byte moves.
# ===========================================================================

def test_cb_role_width_below_its_own_quantum_refuses_with_the_cb_error(
        monkeypatch):
    """The ROLE's law fires, carrying the role's own structured fields.

    An fp8 CB role's rank-local width must stay a multiple of 16 (the native
    kernel's row quantum).  24 rows at degree 2 gives 12, which is not; the
    refusal must be the CB error, not one invented by the composer.
    """

    _as_rank(monkeypatch, 0, 2)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    with pytest.raises(ShardGroupAlignmentError) as caught:
        method.create_weights(layer, K_FULL, [12, 1024], K_FULL, 24 + 2048,
                              torch.bfloat16)
    assert caught.value.axis == "output"
    assert caught.value.group_size == 16
    assert caught.value.tp_degree == 2
    assert caught.value.shard_size == 12
    assert not list(layer.named_parameters())


def test_source_role_width_below_its_block_refuses_with_the_source_error(
        monkeypatch):
    """The passthrough role's 128-element block law, likewise its own error."""

    _as_rank(monkeypatch, 0, 2)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    with pytest.raises(ShardAlignmentError, match="column-parallel shard"):
        method.create_weights(layer, K_FULL, [1024, 32], K_FULL, 2048 + 64,
                              torch.bfloat16)
    assert not list(layer.named_parameters())


def test_row_parallel_split_of_a_merged_plane_refuses(monkeypatch):
    """A merged projection is column-parallel; the composer owns this one."""

    _as_rank(monkeypatch, 0, 2)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    with pytest.raises(MixedFusedShardError, match="INPUT axis"):
        method.create_weights(layer, K_FULL // 2, [ROLE_N, ROLE_N], K_FULL,
                              2 * ROLE_N, torch.bfloat16)
    assert not list(layer.named_parameters())


def test_uneven_column_partition_refuses_before_construction(monkeypatch):
    _as_rank(monkeypatch, 0, 2)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    with pytest.raises(MixedFusedShardError, match="uneven output"):
        method.create_weights(layer, K_FULL, [1024, 1024], K_FULL, 3072,
                              torch.bfloat16)


def test_rank_outside_the_structural_degree_refuses(monkeypatch):
    _as_rank(monkeypatch, 3, 2)
    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    with pytest.raises(MixedFusedShardError, match="outside the column "
                                                   "degree"):
        method.create_weights(layer, K_FULL, [1024, 1024], K_FULL,
                              2 * ROLE_N, torch.bfloat16)


@pytest.mark.parametrize("plane", ["cb_qweight", "weight"])
def test_an_already_rank_local_plane_refuses_instead_of_replicating(
        monkeypatch, plane):
    """A whole plane of the wrong multiple refuses AT STAGE.

    This is the silent failure the narrowing exists to prevent: a plane that
    already has this rank's extent would otherwise pass a shape-equality gate
    and be copied identically onto every rank.
    """

    cb = _cb_planes()
    source = _source_planes()
    checkpoint = dict(_checkpoint(cb, source))
    name = (GATE if plane == "cb_qweight" else UP) + "." + plane
    checkpoint[name] = checkpoint[name].narrow(0, 0, ROLE_N // 2)

    _method, layer = _build_mixed(monkeypatch, rank=0, degree=2)
    params = dict(layer.named_parameters())
    transactions = _MixedFusedTransactions(params)
    with pytest.raises(ValueError, match="needs a WHOLE checkpoint plane"):
        for tensor_name, tensor in checkpoint.items():
            transactions.stage(tensor_name, tensor)
    assert all(not bool(getattr(param, "_gridbook_mixed_fused_filled", False))
               for param in params.values())


def test_an_unstamped_carrier_refuses_rather_than_assuming_degree_one():
    """The shard stamp is part of the carrier ABI, not an optimization."""

    method = MixedFusedLinearMethod(FUSED, _roles())
    layer = torch.nn.Module()
    method.create_weights(layer, K_FULL, [ROLE_N, ROLE_N], K_FULL,
                          2 * ROLE_N, torch.bfloat16)
    params = dict(layer.named_parameters())
    for param in params.values():
        delattr(param, "_gridbook_mixed_fused_shard")
    transactions = _MixedFusedTransactions(params)
    with pytest.raises(ValueError, match="carries no shard stamp"):
        transactions.stage(GATE + ".cb_qweight", _cb_planes()[0])


# ===========================================================================
# 4. The shipped DSv4 modules.
# ===========================================================================

@pytest.mark.parametrize("layer_index", [12, 13, 18, 32])
def test_shipped_dsv4_two_cb_rung_modules_construct_and_load_at_tp2(
        monkeypatch, layer_index):
    """The OTHER four: two CB roles at different rungs, K48 fused with K44.

    A mixed-format fusion does not require a non-CB role — two CB roles whose
    schemes differ have no honest single merged method either, and layers 12,
    13, 18 and 32 of the shipped artifact are exactly that. Their planes have
    different row widths, which is what makes the per-carrier narrowing
    (rather than one merged narrow) the thing being checked here.
    """

    gate = f"model.layers.{layer_index}.ffn.shared_experts.w1"
    up = f"model.layers.{layer_index}.ffn.shared_experts.w3"
    fused = f"model.layers.{layer_index}.ffn.shared_experts.gate_up_proj"
    k44 = {**CB_SCHEME, "k": 44, "type_size": 176,
           "codebook_ref": [f"cb.fp8.k44.sub{i}" for i in range(CB_N_SUB)]}
    k44_row_bytes = (K_FULL // SUPERBLOCK) * 176

    generator = torch.Generator().manual_seed(31)
    checkpoint = {
        gate + ".cb_qweight": torch.randint(
            0, 256, (ROLE_N, CB_ROW_BYTES), dtype=torch.uint8,
            generator=generator),
        gate + ".weight_scale": torch.rand(ROLE_N, generator=generator),
        up + ".cb_qweight": torch.randint(
            0, 256, (ROLE_N, k44_row_bytes), dtype=torch.uint8,
            generator=generator),
        up + ".weight_scale": torch.rand(ROLE_N, generator=generator),
    }

    for rank in range(2):
        _as_rank(monkeypatch, rank, 2)
        method = MixedFusedLinearMethod(fused, [
            (gate, _cb_method(gate)),
            (up, PrismaQuantCBLinearMethod(types.SimpleNamespace(), k44, up)),
        ])
        layer = torch.nn.Module()
        method.create_weights(layer, K_FULL, [ROLE_N // 2, ROLE_N // 2],
                              K_FULL, 2 * ROLE_N, torch.bfloat16)
        _load_mixed(layer, checkpoint)
        for index, (name, row_bytes) in enumerate(
                ((gate, CB_ROW_BYTES), (up, k44_row_bytes))):
            planes = _carrier_planes(layer, index)
            assert tuple(planes["cb_qweight"].shape) == (ROLE_N // 2,
                                                         row_bytes)
            assert torch.equal(
                planes["cb_qweight"].data,
                checkpoint[name + ".cb_qweight"].narrow(
                    0, rank * (ROLE_N // 2), ROLE_N // 2))
            assert torch.equal(
                planes["weight_scale"].data,
                checkpoint[name + ".weight_scale"].narrow(
                    0, rank * (ROLE_N // 2), ROLE_N // 2))


@pytest.mark.parametrize("layer_index", [0, 19, 21, 22])
def test_shipped_dsv4_mixed_modules_construct_and_load_at_tp2(
        monkeypatch, layer_index):
    """The four CB-gate / passthrough-up modules, at their real shapes."""

    gate = f"model.layers.{layer_index}.ffn.shared_experts.w1"
    up = f"model.layers.{layer_index}.ffn.shared_experts.w3"
    fused = f"model.layers.{layer_index}.ffn.shared_experts.gate_up_proj"
    cb = _cb_planes()
    source = _source_planes()
    checkpoint = {gate + ".cb_qweight": cb[0], gate + ".weight_scale": cb[1],
                  up + ".weight": source[0],
                  up + ".weight_scale_inv": source[1]}

    for rank in range(2):
        _as_rank(monkeypatch, rank, 2)
        method = MixedFusedLinearMethod(
            fused, [(gate, _cb_method(gate)), (up, _source_method())])
        layer = torch.nn.Module()
        method.create_weights(layer, K_FULL, [ROLE_N // 2, ROLE_N // 2],
                              K_FULL, 2 * ROLE_N, torch.bfloat16)
        _load_mixed(layer, checkpoint)
        assert [tuple(param.shape)
                for param in _carrier_planes(layer, 0).values()] == [
            (ROLE_N // 2, CB_ROW_BYTES), (ROLE_N // 2,)]
        assert [tuple(param.shape)
                for param in _carrier_planes(layer, 1).values()] == [
            (ROLE_N // 2, K_FULL),
            (ROLE_N // (2 * DS_BLOCK), K_FULL // DS_BLOCK)]
