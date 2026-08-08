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
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)


_GROUP_ATTR = "_gridbook_mixed_fused_group"
_SOURCE_ATTR = "_gridbook_mixed_fused_source"
_PLANE_ATTR = "_gridbook_mixed_fused_plane"
_FILLED_ATTR = "_gridbook_mixed_fused_filled"


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
        del output_size
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
            method.create_weights(
                carrier,
                input_size_per_partition,
                [width],
                input_size,
                width,
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


__all__ = ["MixedFusedLinearMethod"]
