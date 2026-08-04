"""Wire contract and CPU routing helpers for split-format MoE expert stacks.

The schema is producer-owned by PrismaQuant's tier-2 exporter.  This module is
deliberately torch/vLLM-free at import time so an artifact can be refused for a
bad declaration before any device work or method construction begins.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence


SCHEMA_KEY = "per_expert_format_groups"
SUPPORTED_SCHEMA_VERSIONS = (1,)
FAMILIES = ("w13", "w2")
MXFP4_SOURCE = "mxfp4_e2m1_ue8m0_g32"


class PerExpertFormatError(ValueError):
    """A split-stack declaration Gridbook refuses to serve."""


@dataclass(frozen=True)
class ExpertFormatGroup:
    family: str
    format_wire_id: str
    expert_ids: tuple[int, ...]
    tensor_prefix: str

    @property
    def is_passthrough(self) -> bool:
        return self.format_wire_id == MXFP4_SOURCE


@dataclass(frozen=True)
class LayerFormatGroups:
    layer_id: str
    w13: tuple[ExpertFormatGroup, ...]
    w2: tuple[ExpertFormatGroup, ...]
    num_experts: int
    # family -> expert id -> (group index, position in the packed sub-stack)
    index_maps: Mapping[str, tuple[tuple[int, int], ...]]

    def groups(self, family: str) -> tuple[ExpertFormatGroup, ...]:
        if family not in FAMILIES:
            raise KeyError(family)
        return getattr(self, family)


_LAYER_ID_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")


def layer_id_for_prefix(prefix: str) -> str | None:
    match = _LAYER_ID_RE.search(prefix)
    return match.group(1) if match is not None else None


def _known_cb_wire_ids(runtime_contract: Mapping[str, Any]) -> set[str]:
    known: set[str] = set()
    for declaration in runtime_contract.get("formats", ()):
        if not isinstance(declaration, Mapping):
            continue
        pattern = declaration.get("name_pattern")
        if not isinstance(pattern, str) or "{k}" not in pattern:
            continue
        for rung in declaration.get("rungs", ()):
            if isinstance(rung, int) and not isinstance(rung, bool):
                known.add(pattern.format(k=rung))
    return known


def parse_declaration(
    config: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any],
    cb_schemes: Mapping[str, Mapping[str, Any]] | None = None,
    canonicalize: Callable[[str], str] | None = None,
) -> dict[str, LayerFormatGroups]:
    """Parse producer v1, or return ``{}`` when the key is absent.

    ``cb_schemes`` is the canonical target-prefix -> scheme map built from
    config groups.  Supplying it additionally proves that every CB declaration
    names a real tensor stack and that the scheme's grid/rung agrees with the
    wire id.  Passthrough prefixes intentionally have no CB scheme.
    """

    raw = config.get(SCHEMA_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PerExpertFormatError(
            f"{SCHEMA_KEY!r} must be an object with 'version' and 'layers', "
            f"got {type(raw).__name__}"
        )
    if set(raw) != {"version", "layers"}:
        raise PerExpertFormatError(
            f"{SCHEMA_KEY!r} must contain exactly 'version' and 'layers', "
            f"got {sorted(map(str, raw))}"
        )
    version = raw["version"]
    if (not isinstance(version, int) or isinstance(version, bool)
            or version not in SUPPORTED_SCHEMA_VERSIONS):
        raise PerExpertFormatError(
            f"{SCHEMA_KEY!r} declares schema version {version!r}; this build "
            f"supports only {SUPPORTED_SCHEMA_VERSIONS}. Gridbook does not "
            "guess at newer split-stack semantics."
        )
    layers = raw["layers"]
    if not isinstance(layers, Mapping) or not layers:
        raise PerExpertFormatError(
            f"{SCHEMA_KEY!r}.layers must be a non-empty object"
        )

    known_cb = _known_cb_wire_ids(runtime_contract)
    known = known_cb | {MXFP4_SOURCE}
    resolved: dict[str, LayerFormatGroups] = {}
    for raw_layer_id, raw_families in layers.items():
        if (not isinstance(raw_layer_id, str) or not raw_layer_id.isdigit()
                or str(int(raw_layer_id)) != raw_layer_id):
            raise PerExpertFormatError(
                f"{SCHEMA_KEY!r}.layers key must be a canonical numeric "
                f"string, got {raw_layer_id!r}"
            )
        if not isinstance(raw_families, Mapping) or set(raw_families) != set(FAMILIES):
            got = (
                sorted(map(str, raw_families))
                if isinstance(raw_families, Mapping)
                else type(raw_families).__name__
            )
            raise PerExpertFormatError(
                f"layer {raw_layer_id}: expected exactly w13 and w2 families, "
                f"got {got}"
            )

        parsed: dict[str, tuple[ExpertFormatGroup, ...]] = {}
        maps: dict[str, tuple[tuple[int, int], ...]] = {}
        family_sets: dict[str, set[int]] = {}
        for family in FAMILIES:
            entries = raw_families[family]
            if not isinstance(entries, list) or not entries:
                raise PerExpertFormatError(
                    f"layer {raw_layer_id}/{family}: format groups must be a "
                    "non-empty list"
                )
            groups: list[ExpertFormatGroup] = []
            owners: dict[int, str] = {}
            seen_formats: set[str] = set()
            for group_index, entry in enumerate(entries):
                if not isinstance(entry, Mapping) or set(entry) != {
                    "format_wire_id", "expert_ids", "tensor_prefix"
                }:
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family} group {group_index}: "
                        "expected exact keys format_wire_id/expert_ids/"
                        "tensor_prefix"
                    )
                wire_id = entry["format_wire_id"]
                if not isinstance(wire_id, str) or wire_id not in known:
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}: unknown format wire "
                        f"id {wire_id!r}"
                    )
                if wire_id in seen_formats:
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}: format {wire_id!r} is "
                        "declared more than once"
                    )
                seen_formats.add(wire_id)
                tensor_prefix = entry["tensor_prefix"]
                if not isinstance(tensor_prefix, str) or not tensor_prefix:
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}/{wire_id}: "
                        "tensor_prefix must be a non-empty string"
                    )
                prefix_layer = layer_id_for_prefix(tensor_prefix)
                if prefix_layer != raw_layer_id:
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}/{wire_id}: "
                        f"tensor_prefix {tensor_prefix!r} belongs to layer "
                        f"{prefix_layer!r}"
                    )
                if wire_id in known_cb:
                    projection = (
                        "gate_up_proj" if family == "w13" else "down_proj"
                    )
                    if f".{projection}.format_group_" not in tensor_prefix:
                        raise PerExpertFormatError(
                            f"layer {raw_layer_id}/{family}/{wire_id}: CB "
                            f"tensor_prefix {tensor_prefix!r} does not name "
                            f"the {projection} family sub-stack"
                        )
                raw_ids = entry["expert_ids"]
                if (not isinstance(raw_ids, list) or not raw_ids
                        or any(not isinstance(value, int)
                               or isinstance(value, bool) or value < 0
                               for value in raw_ids)):
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}/{wire_id}: expert_ids "
                        "must be a non-empty list of non-negative integers"
                    )
                expert_ids = tuple(raw_ids)
                if expert_ids != tuple(sorted(expert_ids)):
                    raise PerExpertFormatError(
                        f"layer {raw_layer_id}/{family}/{wire_id}: expert_ids "
                        f"must be sorted, got {list(expert_ids)}"
                    )
                for expert_id in expert_ids:
                    if expert_id in owners:
                        raise PerExpertFormatError(
                            f"layer {raw_layer_id}/{family}: expert "
                            f"{expert_id} is double-claimed by "
                            f"{owners[expert_id]!r} and {wire_id!r}"
                        )
                    owners[expert_id] = wire_id

                if wire_id in known_cb and cb_schemes is not None:
                    lookup_prefix = (
                        canonicalize(tensor_prefix)
                        if canonicalize is not None else tensor_prefix
                    )
                    scheme = cb_schemes.get(lookup_prefix)
                    if scheme is None:
                        raise PerExpertFormatError(
                            f"layer {raw_layer_id}/{family}/{wire_id}: CB "
                            f"tensor_prefix {tensor_prefix!r} has no config-"
                            "group scheme"
                        )
                    expected_grid = "fp4" if wire_id.startswith("NVFP4_CB_") else "fp8"
                    try:
                        expected_k = int(wire_id.rsplit("K", 1)[1])
                    except (IndexError, ValueError):  # registry drift
                        raise PerExpertFormatError(
                            f"invalid known CB wire id {wire_id!r}"
                        ) from None
                    if (scheme.get("grid") != expected_grid
                            or int(scheme.get("k", -1)) != expected_k):
                        raise PerExpertFormatError(
                            f"layer {raw_layer_id}/{family}/{wire_id}: scheme "
                            f"at {tensor_prefix!r} declares grid="
                            f"{scheme.get('grid')!r}, k={scheme.get('k')!r}"
                        )
                groups.append(ExpertFormatGroup(
                    family, wire_id, expert_ids, tensor_prefix
                ))
            family_sets[family] = set(owners)
            parsed[family] = tuple(groups)

        union = family_sets["w13"] | family_sets["w2"]
        num_experts = max(union, default=-1) + 1
        expected = set(range(num_experts))
        for family in FAMILIES:
            missing = sorted(expected - family_sets[family])
            extra = sorted(family_sets[family] - expected)
            if missing or extra:
                raise PerExpertFormatError(
                    f"layer {raw_layer_id}/{family}: bad expert partition; "
                    f"missing expert ids {missing}, unexpected expert ids {extra}"
                )
            positions: list[tuple[int, int] | None] = [None] * num_experts
            for group_index, group in enumerate(parsed[family]):
                for local_position, expert_id in enumerate(group.expert_ids):
                    positions[expert_id] = (group_index, local_position)
            assert all(position is not None for position in positions)
            maps[family] = tuple(
                position for position in positions if position is not None
            )

        resolved[raw_layer_id] = LayerFormatGroups(
            layer_id=raw_layer_id,
            w13=parsed["w13"],
            w2=parsed["w2"],
            num_experts=num_experts,
            index_maps=maps,
        )
    return resolved


def dispatch_grouped_pairs(
    hidden_states: Any,
    topk_weights: Any,
    topk_ids: Any,
    groups: Sequence[ExpertFormatGroup],
    run_group: Callable[[ExpertFormatGroup, Any, Any, Any], Any],
    *,
    require_complete: bool = True,
):
    """Correctness-first subgroup dispatch used by CPU oracles and delegates.

    Each selected ``(token, expert)`` becomes a one-expert routed row.  The
    group's runner therefore applies its router weight exactly once; outputs
    are scattered back to tokens and summed.  This intentionally favors a
    transparent oracle over launch minimization.
    """

    # Kept as a local import: declaration/refusal tests need no torch install.
    import torch

    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have identical shape")
    tokens = int(topk_ids.shape[0])
    output = None
    claimed = torch.zeros_like(topk_ids, dtype=torch.bool)
    for group in groups:
        expert_ids = torch.tensor(
            group.expert_ids, dtype=topk_ids.dtype, device=topk_ids.device
        )
        mask = (topk_ids[..., None] == expert_ids).any(dim=-1)
        claimed |= mask
        pair = mask.nonzero(as_tuple=False)
        if pair.numel() == 0:
            continue
        token_index = pair[:, 0]
        selected_global = topk_ids[pair[:, 0], pair[:, 1]]
        local_map = torch.full(
            (max(group.expert_ids) + 1,), -1,
            dtype=torch.long, device=topk_ids.device,
        )
        local_map[expert_ids.long()] = torch.arange(
            len(group.expert_ids), device=topk_ids.device
        )
        local_ids = local_map[selected_global.long()].reshape(-1, 1)
        local_weights = topk_weights[pair[:, 0], pair[:, 1]].reshape(-1, 1)
        pair_output = run_group(
            group,
            hidden_states.index_select(0, token_index),
            local_weights,
            local_ids,
        )
        if output is None:
            output = torch.zeros(
                (tokens, pair_output.shape[-1]), dtype=pair_output.dtype,
                device=pair_output.device,
            )
        output.index_add_(0, token_index, pair_output)
    if require_complete and not bool(torch.all(claimed)):
        missing = topk_ids[~claimed].detach().cpu().tolist()
        raise RuntimeError(f"routing selected undeclared expert ids {missing}")
    if output is None:
        return hidden_states.new_zeros(hidden_states.shape)
    return output


def dispatch_family_stages(
    hidden_states: Any,
    topk_weights: Any,
    topk_ids: Any,
    layer: LayerFormatGroups,
    run_stage: Callable[[str, ExpertFormatGroup, Any, Any], Any],
    activate: Callable[[Any], Any],
):
    """Family-independent mixed dispatch, and the fused-kernel oracle.

    ``run_stage`` receives pair rows and group-local expert ids.  w13 and w2
    partitions may differ: the intermediate stays in original routed-pair
    order between the two family loops.  Router weights are deliberately absent
    from both stages and applied once, after w2, immediately before the token
    combine.
    """

    import torch

    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError("routing tensors must be equal-shape rank-2 tensors")
    tokens, topk = topk_ids.shape
    pair_expert = topk_ids.reshape(-1).long()
    if bool(torch.any(pair_expert < 0)) or bool(
        torch.any(pair_expert >= layer.num_experts)
    ):
        raise RuntimeError("routing selected an expert outside the declaration")
    pair_token = torch.arange(
        tokens, device=topk_ids.device, dtype=torch.long
    ).repeat_interleave(topk)
    family_maps = {}
    for family in FAMILIES:
        family_maps[family] = (
            torch.tensor(
                [item[0] for item in layer.index_maps[family]],
                device=topk_ids.device, dtype=torch.long,
            ),
            torch.tensor(
                [item[1] for item in layer.index_maps[family]],
                device=topk_ids.device, dtype=torch.long,
            ),
        )

    intermediate = None
    for group_index, group in enumerate(layer.w13):
        group_map, position_map = family_maps["w13"]
        selected = (group_map.index_select(0, pair_expert) == group_index) \
            .nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        global_ids = pair_expert.index_select(0, selected)
        local = position_map.index_select(0, global_ids)
        values = run_stage(
            "w13", group,
            hidden_states.index_select(0, pair_token.index_select(0, selected)),
            local,
        )
        if intermediate is None:
            intermediate = values.new_empty((pair_expert.numel(), values.shape[-1]))
        intermediate.index_copy_(0, selected, values)
    if intermediate is None:
        raise RuntimeError("w13 declaration routed no expert pairs")
    intermediate = activate(intermediate)

    pair_output = None
    for group_index, group in enumerate(layer.w2):
        group_map, position_map = family_maps["w2"]
        selected = (group_map.index_select(0, pair_expert) == group_index) \
            .nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        global_ids = pair_expert.index_select(0, selected)
        local = position_map.index_select(0, global_ids)
        values = run_stage(
            "w2", group, intermediate.index_select(0, selected), local
        )
        if pair_output is None:
            pair_output = values.new_empty((pair_expert.numel(), values.shape[-1]))
        pair_output.index_copy_(0, selected, values)
    if pair_output is None:
        raise RuntimeError("w2 declaration routed no expert pairs")

    weighted = pair_output * topk_weights.reshape(-1, 1).to(pair_output.dtype)
    output = pair_output.new_zeros((tokens, pair_output.shape[-1]))
    output.index_add_(0, pair_token, weighted)
    return output
