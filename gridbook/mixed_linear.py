"""Composite dense method for heterogeneous vLLM fused projections.

vLLM represents several checkpoint siblings (q/k/v, gate/up and model-specific
variants) as one ``MergedColumnParallelLinear``.  That is an output-layout
choice, not a quantization constraint: each sibling remains an independently
encoded Linear.  This method gives every role a private carrier populated by
its existing Gridbook/source-native method, runs those methods unchanged, and
concatenates their outputs in vLLM's declared shard order.

No common resident weight, requantization, or fallback kernel is introduced.
The only extra resident state is the ordinary child parameters the two methods
would have owned separately; the concatenated activation is the tensor vLLM's
merged module already promises to return.

**Tensor parallelism.**  A merged plane above one rank hands this method its
RANK-LOCAL role widths together with the WHOLE tensor's output size.  Both
numbers come from ``create_weights``' own arguments, so the column degree is
structural and is well defined offline — it is never read from ``layer.tp_size``
(vLLM stamps the world size onto replicated and ``disable_tp`` planes too).
Each role carrier is then constructed with its own whole-tensor output size, so
the ROLE's existing shard law — not a law invented here — decides whether the
geometry is legal, and it decides before any parameter exists.  The narrowing
itself happens where the bytes arrive, in Gridbook's top-level transaction
router, against the ``MixedFusedShard`` stamped on every carrier parameter here.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)


_GROUP_ATTR = "_gridbook_mixed_fused_group"
_SOURCE_ATTR = "_gridbook_mixed_fused_source"
_PLANE_ATTR = "_gridbook_mixed_fused_plane"
_FILLED_ATTR = "_gridbook_mixed_fused_filled"
_SHARD_ATTR = "_gridbook_mixed_fused_shard"


class MixedFusedShardError(ValueError):
    """A tensor-parallel geometry the mixed fused composer itself refuses.

    Only geometries no ROLE law can see are refused here — the composer owns
    the merged plane's structure, not any role's alignment quanta.  A role
    width that violates its own format's law raises that role's own error
    (``ShardGroupAlignmentError``/``ShardAlignmentError``) from inside the
    role's ``create_weights``, which this method calls before any parameter
    for that role exists.
    """


class MixedFusedShard(NamedTuple):
    """The narrowing law for one carrier parameter, stamped at construction.

    ``col_degree`` and ``tp_rank`` are the merged plane's, not the plane's own:
    every role of a merged column-parallel Linear is narrowed at the SAME rank
    coordinates, which is exactly why vLLM's merged loader can address roles by
    offset at all.  ``output_dim`` is the parameter's own declared column axis
    (``None`` for a per-tensor scalar, which replicates), and the packed fields
    are recorded so a refusal can name the units the extents are counted in.
    """

    col_degree: int
    tp_rank: int
    output_dim: int | None
    packed_dim: int | None
    packed_factor: int | None


def _axis_degree(full: int, local: int, *, axis: str, prefix: str) -> int:
    """Whole-number shard degree along one axis, or a structured refusal."""

    full = int(full)
    local = int(local)
    if full <= 0 or local <= 0:
        raise MixedFusedShardError(
            f"{prefix}: a mixed fused plane needs a positive {axis} extent; "
            f"got full={full}, per-rank={local}")
    if full % local != 0:
        raise MixedFusedShardError(
            f"{prefix}: refusing an uneven {axis} partition of a mixed fused "
            f"plane — the whole extent {full} is not a whole multiple of this "
            f"rank's extent {local}. Gridbook composes mixed-format roles only "
            "where every rank holds the same shape; serve at a "
            "tensor-parallel size that divides the layer evenly.")
    return full // local


class _RoleCarrier(torch.nn.Module):
    """Parameter owner presented to one already-existing linear method."""


@register_weight_loader_v2_supported_method
class MixedFusedLinearMethod(LinearMethodBase):
    """Compose ordered per-role methods behind one vLLM fused Linear."""

    def __init__(self, prefix: str,
                 roles: Sequence[tuple[str, LinearMethodBase]]) -> None:
        if len(roles) < 2:
            raise ValueError(
                f"{prefix}: a mixed fused method needs at least two roles")
        self.prefix = prefix
        self.roles = tuple(roles)

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        widths = [int(width) for width in output_partition_sizes]
        if len(widths) != len(self.roles):
            raise ValueError(
                f"{self.prefix}: vLLM declares {len(widths)} logical output "
                f"shards {widths}, but Gridbook resolved {len(self.roles)} "
                "format owners")
        if any(width <= 0 for width in widths):
            raise ValueError(
                f"{self.prefix}: fused role widths must be positive, got "
                f"{widths}")

        # The serving degrees, from vLLM's own constructor arguments.
        row_degree = _axis_degree(
            input_size, input_size_per_partition,
            axis="input (row-parallel)", prefix=self.prefix)
        col_degree = _axis_degree(
            output_size, sum(widths),
            axis="output (column-parallel)", prefix=self.prefix)
        if row_degree != 1:
            raise MixedFusedShardError(
                f"{self.prefix}: refusing a mixed fused plane sharded on the "
                f"INPUT axis (row degree {row_degree}, column degree "
                f"{col_degree}). A merged projection is column-parallel by "
                "construction; a row-parallel split of it would give every "
                "rank a partial sum of every role, which no role's format law "
                "was qualified against. Serve this module column-parallel, or "
                "at tensor-parallel size 1.")
        tp_rank = 0
        if col_degree > 1:
            # Read ONCE, here, where the parameters are built — never inside
            # the loader, which must be able to run offline and must place
            # every plane of every role at the same rank coordinates.
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = int(get_tensor_model_parallel_rank())
            if not 0 <= tp_rank < col_degree:
                raise MixedFusedShardError(
                    f"{self.prefix}: vLLM reports tensor-parallel rank "
                    f"{tp_rank}, which is outside the column degree "
                    f"{col_degree} implied by this layer's own shapes "
                    f"(whole output {int(output_size)}, per-rank "
                    f"{sum(widths)}). Refusing rather than narrowing at an "
                    "out-of-range offset.")

        carriers = torch.nn.ModuleList()
        child_attrs = dict(extra_weight_attrs)
        # The architecture loader cannot address nested role parameters by its
        # ordinary fused name. Gridbook's top-level transaction router owns
        # these loads, so inheriting the parent's merged weight loader would be
        # both unused and dangerously ambiguous.
        child_attrs["weight_loader"] = None
        for index, ((source, method), width) in enumerate(
                zip(self.roles, widths)):
            carrier = _RoleCarrier()
            # Role methods that report a shard refusal name the layer they
            # were handed; give them the role's identity rather than "".
            carrier.prefix = f"{self.prefix} role {index} ({source})"
            # The role's OWN whole-tensor output size.  This is the single
            # line that makes the role's existing shard law fire: it now sees
            # a per-rank width against a whole extent, exactly as it would if
            # this role were a standalone column-parallel Linear.
            method.create_weights(
                carrier,
                input_size_per_partition,
                [width],
                input_size,
                width * col_degree,
                params_dtype,
                **child_attrs,
            )
            direct_params = list(carrier.named_parameters(recurse=False))
            if not direct_params:
                raise RuntimeError(
                    f"{self.prefix}: role {index} ({source}) method "
                    f"{type(method).__name__} registered no parameters")
            for plane, param in direct_params:
                setattr(param, _GROUP_ATTR, self.prefix)
                setattr(param, _SOURCE_ATTR, source)
                setattr(param, _PLANE_ATTR, plane)
                setattr(param, _FILLED_ATTR, False)
                output_dim = getattr(param, "output_dim", None)
                setattr(param, _SHARD_ATTR, MixedFusedShard(
                    col_degree=col_degree,
                    tp_rank=tp_rank,
                    output_dim=(None if output_dim is None
                                else int(output_dim)),
                    packed_dim=getattr(param, "packed_dim", None),
                    packed_factor=getattr(param, "packed_factor", None),
                ))
                # A vLLM parameter records the rank it was built on. If the
                # two disagree, one of them is reading a different distributed
                # state than the other, and a narrow at either coordinate
                # would be a guess.  Only meaningful where this plane is
                # actually sharded: vLLM stamps the live rank onto REPLICATED
                # planes too, so at degree 1 a non-zero param rank is normal
                # and narrowing does not happen at all.
                param_rank = (None if col_degree == 1
                              else getattr(param, "tp_rank", None))
                if param_rank is not None and int(param_rank) != tp_rank:
                    raise MixedFusedShardError(
                        f"{self.prefix}: role {index} ({source}) plane "
                        f"{plane!r} was built on tensor-parallel rank "
                        f"{int(param_rank)} while the merged plane resolved "
                        f"rank {tp_rank} at column degree {col_degree}; "
                        "refusing rather than narrowing at a rank the "
                        "parameter does not agree with.")
            carriers.append(carrier)

        layer.add_module("_gridbook_mixed_roles", carriers)
        layer.logical_widths = widths
        layer._gridbook_mixed_methods = tuple(
            method for _, method in self.roles)

    def process_weights_after_loading(self, layer) -> None:
        carriers = tuple(layer._gridbook_mixed_roles)
        methods = tuple(layer._gridbook_mixed_methods)
        if len(carriers) != len(methods) or len(carriers) != len(self.roles):
            raise RuntimeError(
                f"{self.prefix}: mixed fused carrier/method topology changed "
                "between construction and load finalization")
        for index, (carrier, method) in enumerate(zip(carriers, methods)):
            missing = [
                plane for plane, param in carrier.named_parameters(
                    recurse=False)
                if not bool(getattr(param, _FILLED_ATTR, False))
            ]
            if missing:
                raise RuntimeError(
                    f"{self.prefix}: mixed fused role {index} "
                    f"({self.roles[index][0]}) is missing checkpoint planes "
                    f"{missing}; refusing uninitialized serving state")
            method.process_weights_after_loading(carrier)

    def apply(self, layer, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        outputs = []
        for index, (carrier, method, width) in enumerate(zip(
                layer._gridbook_mixed_roles,
                layer._gridbook_mixed_methods,
                layer.logical_widths)):
            output = method.apply(carrier, x, None)
            if output.shape[:-1] != x.shape[:-1] \
                    or int(output.shape[-1]) != int(width):
                raise RuntimeError(
                    f"{self.prefix}: mixed fused role {index} returned "
                    f"shape {tuple(output.shape)}, expected "
                    f"{tuple(x.shape[:-1]) + (int(width),)}")
            outputs.append(output)
        dtype = outputs[0].dtype
        device = outputs[0].device
        if any(output.dtype != dtype or output.device != device
               for output in outputs[1:]):
            raise RuntimeError(
                f"{self.prefix}: mixed fused role methods returned different "
                "dtype/device contracts")
        merged = torch.cat(outputs, dim=-1)
        return merged if bias is None else merged + bias


__all__ = [
    "MixedFusedLinearMethod",
    "MixedFusedShard",
    "MixedFusedShardError",
]
