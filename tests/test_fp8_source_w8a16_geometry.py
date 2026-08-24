"""CPU-only admission gates for the source-FP8 W8A16 lane.

Two families live here: the grouped-BMM geometry qualification, and the
tensor-parallel shard laws.  Both refuse before any native resolution, and
the shard laws refuse before any parameter exists at all -- a misaligned
block-scale narrow is silently wrong rather than loud, so construction time
is the only safe place to catch it.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import fp8_source_w8a16 as _lane  # noqa: E402


def _module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _method(monkeypatch):
    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
    linear = _module(monkeypatch, "vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    parameter = _module(monkeypatch, "vllm.model_executor.parameter")
    parameter.BlockQuantScaleParameter = type(
        "BlockQuantScaleParameter", (), {})
    parameter.ModelWeightParameter = type("ModelWeightParameter", (), {})

    from gridbook import fp8_source_w8a16 as lane

    return lane, lane.build_fp8_source_w8a16_method(
        lane.WIRE_FP8_BLOCK128)


def _layer(*, groups, rows, k, tp_size, is_bmm=True, plan=None):
    layer = torch.nn.Module()
    total_rows = groups * rows
    q = torch.empty(
        total_rows, k, dtype=torch.float8_e4m3fn, device="meta")
    scales = torch.empty(
        (total_rows + 127) // 128,
        (k + 127) // 128,
        dtype=torch.float8_e8m0fnu,
        device="meta",
    )
    layer.register_parameter(
        "weight", torch.nn.Parameter(q, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    layer.tp_size = tp_size
    if plan is not None:
        setattr(layer, _lane._SHARD_ATTR, plan)
    if is_bmm:
        layer.is_bmm = True
        layer.bmm_batch_size = groups
    return layer


def _patch_load_edges(monkeypatch, lane):
    from gridbook import cuda_ext, dsv4_woa, ops

    calls = []
    monkeypatch.setattr(lane, "_require_source_cuda", lambda tensor: None)
    monkeypatch.setattr(
        cuda_ext,
        "require_fp8_source_w8a16_ext",
        lambda operation="", device=None: calls.append("source_ext"),
    )
    monkeypatch.setattr(
        cuda_ext,
        "require_bf16_grouped_ext",
        lambda operation="": calls.append("grouped_ext"),
    )
    monkeypatch.setattr(
        ops, "register_cb_layer", lambda method, layer: 17)
    monkeypatch.setattr(
        dsv4_woa,
        "install_dsv4_woa_adapter",
        lambda: calls.append("adapter"),
    )
    return dsv4_woa, calls


@pytest.mark.parametrize(
    ("groups", "rows", "k", "degree", "got"),
    (
        (7, 1024, 4096, 1, "G=7, N=1024, K=4096, shard degree=1"),
        (8, 896, 4096, 1, "G=8, N=896, K=4096, shard degree=1"),
        (8, 1024, 3968, 1, "G=8, N=1024, K=3968, shard degree=1"),
        # Degrees 2 and 4 are measured, so their group counts pass; the near
        # miss is a degree nobody ran.  At TP=8 a DSv4 wo_a rank would hold
        # one group -- perfectly aligned, and still refused.
        (1, 1024, 4096, 8, "G=1, N=1024, K=4096, shard degree=8"),
        # The right group count at the WRONG degree: G=4 is qualified at
        # degree 2 only, because the plane it came from is what was measured.
        (4, 1024, 4096, 4, "G=4, N=1024, K=4096, shard degree=4"),
    ),
)
def test_grouped_geometry_near_miss_refuses_before_native_resolution_or_marker(
        monkeypatch, groups, rows, k, degree, got):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=groups, rows=rows, k=k, tp_size=degree,
        plan=lane.ShardPlan(row_degree=1, col_degree=degree))

    with pytest.raises(ValueError, match="qualified only") as exc:
        method.process_weights_after_loading(layer)

    message = str(exc.value)
    # The refusal enumerates the whole qualified set, so it stays true as
    # degrees are qualified instead of naming a stale single geometry.
    assert "G=8, N=1024, K=4096 at shard degree 1" in message
    assert "G=4, N=1024, K=4096 at shard degree 2" in message
    assert "G=2, N=1024, K=4096 at shard degree 4" in message
    assert got in message
    assert calls == []
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)
    assert not hasattr(layer, lane._READY_ATTR)


def test_exact_grouped_geometry_resolves_both_arms_then_installs_marker(
        monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(groups=8, rows=1024, k=4096, tp_size=1)

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext", "adapter"]
    assert layer._fp8_source_groups == 8
    assert layer._fp8_source_rows == 1024
    assert layer._fp8_source_K == 4096
    assert getattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR) == \
        dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI


def test_dense_geometry_remains_separate_from_grouped_dsv4_gate(monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=136, k=256, tp_size=1, is_bmm=False)

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext"]
    assert layer._fp8_source_groups == 1
    assert layer._fp8_source_rows == 136
    assert layer._fp8_source_K == 256
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)


def test_dense_shard_without_a_plan_refuses_above_one_rank(monkeypatch):
    """No structural plan and a sharded worker: refuse, never guess.

    ``tp_size`` alone cannot say whether THIS plane is sharded -- vLLM stamps
    the world size onto replicated layers too -- so a layer whose parameters
    this lane did not construct is refused above one rank.
    """
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(groups=1, rows=256, k=256, tp_size=2, is_bmm=False)

    with pytest.raises(lane.ShardAlignmentError,
                       match="no shard plan.*TP=2"):
        method.process_weights_after_loading(layer)

    assert calls == []
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)
    assert not hasattr(layer, lane._READY_ATTR)


def test_dense_column_shard_finalizes_when_the_local_plane_is_aligned(
        monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=256, k=512, tp_size=2, is_bmm=False,
        plan=lane.ShardPlan(row_degree=1, col_degree=2))

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext"]
    assert layer._fp8_source_shard_degree == 2
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)


def test_dense_row_shard_finalizes_when_the_local_plane_is_aligned(
        monkeypatch):
    lane, method = _method(monkeypatch)
    _dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=256, k=512, tp_size=2, is_bmm=False,
        plan=lane.ShardPlan(row_degree=2, col_degree=1))

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext"]
    assert layer._fp8_source_shard_degree == 2


@pytest.mark.parametrize(
    ("plan_kwargs", "rows", "k", "message"),
    (
        ({"row_degree": 1, "col_degree": 2}, 136, 256, "column shard.*N=136"),
        ({"row_degree": 2, "col_degree": 1}, 256, 136, "row shard.*K=136"),
    ),
)
def test_dense_shard_with_a_misaligned_resident_plane_refuses(
        monkeypatch, plan_kwargs, rows, k, message):
    """Defense in depth: the resident local shape is re-checked at load end.

    Construction already refused every misaligned shard, so reaching here
    means a loader produced an unexpected local extent.  Refuse rather than
    serve a plane whose scale narrow was misindexed.
    """
    lane, method = _method(monkeypatch)
    _dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=rows, k=k, tp_size=2, is_bmm=False,
        plan=lane.ShardPlan(**plan_kwargs))

    with pytest.raises(lane.ShardAlignmentError, match=message):
        method.process_weights_after_loading(layer)

    assert calls == []
    assert not hasattr(layer, lane._READY_ATTR)


# --- the construction-time shard laws ----------------------------------------
#
# These exercise the lane's own module-level helpers, which is where the laws
# live and where ``create_weights`` calls them (pinned by AST in
# tests/test_runtime_contract_tp.py).  No vLLM parameter object is needed:
# the point is that the refusal happens BEFORE one is built.


def _plan(*, input_size, input_size_per_partition, output_size,
          output_partition_sizes, prefix="model.layers.0.mlp.down_proj"):
    return _lane._resolve_shard_plan(
        input_size=input_size,
        input_size_per_partition=input_size_per_partition,
        output_size=output_size,
        output_partition_sizes=output_partition_sizes,
        prefix=prefix,
    )


def _gate(plan, *, input_size_per_partition, output_partition_sizes,
          prefix="model.layers.0.mlp.down_proj"):
    _lane._require_shard_alignment(
        plan,
        input_size_per_partition=input_size_per_partition,
        output_partition_sizes=output_partition_sizes,
        prefix=prefix,
    )


def test_replicated_planes_derive_degree_one_at_any_world_size():
    """DSv4's fused wqa_wkv and indexer wq_b are replicated but tp_size=N.

    vLLM hands a replicated plane its FULL sizes on both axes, so the
    structural degrees are 1 and no law applies -- which is the whole reason
    the lane must not read ``tp_size``.
    """
    plan = _plan(input_size=4096, input_size_per_partition=4096,
                 output_size=1536, output_partition_sizes=[1024, 512])
    assert plan == _lane.ShardPlan(row_degree=1, col_degree=1)
    assert plan.degree == 1
    # 512 is a multiple of 128 but the roles are irrelevant at degree 1;
    # the gate must not refuse a replicated plane for any width.
    _gate(plan, input_size_per_partition=4096,
          output_partition_sizes=[1024, 500])


def test_dsv4_tp2_shapes_are_all_admitted():
    """Every sharded DSv4 passthrough plane at TP=2, from the model dims."""
    cases = (
        # (name, input_size, in_local, output_size, local role widths)
        ("attn.wq_b", 1024, 1024, 32768, [16384]),
        ("attn.wo_b", 8192, 4096, 4096, [4096]),
        ("shared_experts.gate_up_proj", 4096, 4096, 4096, [1024, 1024]),
        ("shared_experts.down_proj", 2048, 1024, 4096, [4096]),
        ("attn.wo_a", 4096, 4096, 8192, [4096]),
    )
    for name, in_full, in_local, out_full, widths in cases:
        plan = _plan(input_size=in_full, input_size_per_partition=in_local,
                     output_size=out_full, output_partition_sizes=widths,
                     prefix=name)
        assert plan.degree == 2, name
        _gate(plan, input_size_per_partition=in_local,
              output_partition_sizes=widths, prefix=name)


def test_row_shard_that_cuts_a_source_block_refuses():
    plan = _plan(input_size=260, input_size_per_partition=130,
                 output_size=256, output_partition_sizes=[256])
    assert plan == _lane.ShardPlan(row_degree=2, col_degree=1)
    with pytest.raises(_lane.ShardAlignmentError) as exc:
        _gate(plan, input_size_per_partition=130,
              output_partition_sizes=[256])
    message = str(exc.value)
    assert "row-parallel shard of degree 2" in message
    assert "130" in message and "128" in message
    assert "model.layers.0.mlp.down_proj" in message


def test_column_shard_that_cuts_a_source_block_refuses():
    plan = _plan(input_size=256, input_size_per_partition=256,
                 output_size=272, output_partition_sizes=[136])
    assert plan == _lane.ShardPlan(row_degree=1, col_degree=2)
    with pytest.raises(_lane.ShardAlignmentError) as exc:
        _gate(plan, input_size_per_partition=256,
              output_partition_sizes=[136])
    message = str(exc.value)
    assert "column-parallel shard of degree 2" in message
    assert "[136]" in message


def test_merged_plane_refuses_when_only_one_role_is_misaligned():
    """A merged plane's scale offsets are converted role by role.

    The TOTAL local width can be a clean multiple of 128 while an individual
    fused role is not; that is still a misindexed narrow, so the law is
    per role.
    """
    widths = [64, 192]
    assert sum(widths) % 128 == 0
    plan = _plan(input_size=256, input_size_per_partition=256,
                 output_size=512, output_partition_sizes=widths)
    assert plan.degree == 2
    with pytest.raises(_lane.ShardAlignmentError) as exc:
        _gate(plan, input_size_per_partition=256,
              output_partition_sizes=widths)
    assert "EVERY fused role" in str(exc.value)


def test_uneven_partition_refuses_with_both_extents_named():
    with pytest.raises(_lane.ShardAlignmentError) as exc:
        _plan(input_size=384, input_size_per_partition=256,
              output_size=256, output_partition_sizes=[256])
    message = str(exc.value)
    assert "uneven input (row-parallel) partition" in message
    assert "384" in message and "256" in message


def test_two_dimensional_sharding_refuses():
    plan = _lane.ShardPlan(row_degree=2, col_degree=2)
    with pytest.raises(_lane.ShardAlignmentError, match="BOTH axes"):
        _gate(plan, input_size_per_partition=256,
              output_partition_sizes=[256])


def test_shard_alignment_error_is_a_value_error():
    assert issubclass(_lane.ShardAlignmentError, ValueError)


def test_contract_tp_rows_match_this_gate(monkeypatch):
    """The contract's rows for ``fp8_e4m3_ue8m0_block128`` restate THIS gate.

    The packaged table admits the DENSE arm above one rank under whole-128
    shard laws, and admits the grouped-BMM arm at the shard degrees that were
    measured -- and only those.  Feed the lane the CONTRACT'S OWN numbers on
    every half: an aligned dense shard at the published law must finalize, a
    misaligned one must refuse, the BMM arm must finalize at each published
    degree, and it must refuse at a degree the table does not list.  If the
    gate and the table ever drift apart, one half of this test fails.
    """
    import json
    from importlib.resources import files as resource_files

    contract = json.loads(resource_files("gridbook").joinpath(
        "runtime_contract.json").read_text(encoding="utf-8"))
    row = next(unit for unit in contract["tensor_parallel"]["units"]
               if unit["unit"] == "fp8_e4m3_ue8m0_block128")
    assert "max_world_size" not in row
    arms = {arm["arm"]: arm for arm in row["arms"]}
    law = arms["dense"]["shard_admission"]
    assert "max_world_size" not in arms["dense"]
    assert "max_world_size" not in arms["bmm"]
    bmm_law = arms["bmm"]["shard_admission"]
    degrees = bmm_law["qualified_shard_degrees"]
    assert degrees == sorted(set(degrees)), "degrees are a sorted, unique set"
    geometry = arms["bmm"]["requires_geometry"]
    assert geometry == {"bmm_groups": 8, "rows_per_group": 1024, "k": 4096}

    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)

    # Dense, at exactly the published law: admitted above one rank.
    group = law["input_axis_group"]
    quantum = law["output_axis_quantum"]
    dense = _layer(
        groups=1, rows=2 * quantum, k=2 * group, tp_size=2, is_bmm=False,
        plan=lane.ShardPlan(row_degree=1, col_degree=2))
    method.process_weights_after_loading(dense)
    assert dense._fp8_source_shard_degree == 2

    # One block below the law: refused.
    calls.clear()
    misaligned = _layer(
        groups=1, rows=quantum + 8, k=2 * group, tp_size=2, is_bmm=False,
        plan=lane.ShardPlan(row_degree=1, col_degree=2))
    with pytest.raises(lane.ShardAlignmentError):
        method.process_weights_after_loading(misaligned)
    assert calls == []

    # BMM, at exactly the published geometry, at every published degree:
    # admitted, with the group count the shard actually leaves the kernel.
    for degree in degrees:
        calls.clear()
        bmm = _layer(
            groups=geometry["bmm_groups"] // degree,
            rows=geometry["rows_per_group"],
            k=geometry["k"],
            tp_size=degree,
            plan=lane.ShardPlan(row_degree=1, col_degree=degree),
        )
        method.process_weights_after_loading(bmm)
        assert calls == ["source_ext", "grouped_ext", "adapter"]
        assert bmm._fp8_source_groups == geometry["bmm_groups"] // degree
        assert bmm._fp8_source_shard_degree == degree
        assert getattr(bmm, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR) == \
            dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI

    # One degree past the published list: refused, however aligned it is.
    unmeasured = max(degrees) * 2
    calls.clear()
    beyond = _layer(
        groups=max(1, geometry["bmm_groups"] // unmeasured),
        rows=geometry["rows_per_group"],
        k=geometry["k"],
        tp_size=unmeasured,
        plan=lane.ShardPlan(row_degree=1, col_degree=unmeasured),
    )
    with pytest.raises(ValueError,
                       match=rf"qualified only.*shard degree={unmeasured}"):
        method.process_weights_after_loading(beyond)

    assert calls == []
    assert not hasattr(beyond, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)
    assert not hasattr(beyond, lane._READY_ATTR)
