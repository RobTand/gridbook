"""DeepSeek-V4 ``wo_a`` adapter for Gridbook-owned BMM linears.

The qualified Spark vLLM baseline does not call ``self.wo_a`` from NVIDIA
DSV4 attention.  Its specialized output projection instead reads the weight
and scale parameters directly and feeds them to DeepGEMM ``fp8_einsum``.
That bypass is invalid for either Gridbook-owned route: direct-g32 MXFP8 W8A8
repackages its scale plane for CuTe, while block128 source-FP8 W8A16 retains
the raw checkpoint value and scale planes for Gridbook's own dispatch.

This module installs one narrowly guarded class wrapper.  It activates only
when the ``wo_a`` module carries one of Gridbook's versioned BMM ownership
markers: direct MXFP8 W8A8 or block128 source-FP8 W8A16.  Every other DSV4
attention instance calls the original vLLM method byte-for-byte.  The guarded
path uses vLLM's own PyTorch-native inverse RoPE, calls the BMM-aware Gridbook
linear normally, then calls ``wo_b``.  No Gridbook Triton fallback or stock
DeepGEMM scale-layout guess is involved.
"""
from __future__ import annotations

import functools
import sys
from typing import Any

import torch

__all__ = [
    "DSV4_MXFP8_BMM_ATTR",
    "DSV4_MXFP8_BMM_ABI",
    "DSV4_FP8_SOURCE_W8A16_BMM_ATTR",
    "DSV4_FP8_SOURCE_W8A16_BMM_ABI",
    "dsv4_gridbook_o_proj",
    "dsv4_mxfp8_o_proj",
    "install_dsv4_woa_adapter",
]

DSV4_MXFP8_BMM_ATTR = "_gridbook_mxfp8_bmm"
DSV4_MXFP8_BMM_ABI = 1
DSV4_FP8_SOURCE_W8A16_BMM_ATTR = "_gridbook_fp8_source_w8a16_bmm"
DSV4_FP8_SOURCE_W8A16_BMM_ABI = 1
_ADAPTER_ABI = 2
_WRAPPED_ABI_ATTR = "_gridbook_dsv4_woa_adapter_abi"
_LOGGED = False

_NVIDIA_ATTN_CLASSES = {
    "vllm.models.deepseek_v4.nvidia.flashinfer_sparse": (
        "DeepseekV4FlashInferMLAAttention",
        "DeepseekV4FlashInferSM120Attention",
    ),
    "vllm.models.deepseek_v4.nvidia.flashmla": (
        "DeepseekV4FlashMLAAttention",
    ),
}


def _owns_gridbook_bmm(wo_a: Any) -> bool:
    """Whether ``wo_a`` carries one exact Gridbook BMM contract marker."""

    if wo_a is None:
        return False
    return (
        getattr(wo_a, DSV4_MXFP8_BMM_ATTR, None) == DSV4_MXFP8_BMM_ABI or
        getattr(wo_a, DSV4_FP8_SOURCE_W8A16_BMM_ATTR, None) ==
        DSV4_FP8_SOURCE_W8A16_BMM_ABI
    )


def dsv4_gridbook_o_proj(
    attention: Any, o: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Run DSV4 output projection through the owning Gridbook BMM method."""

    wo_a = getattr(attention, "wo_a", None)
    if not _owns_gridbook_bmm(wo_a):
        raise RuntimeError(
            "Gridbook DSV4 wo_a adapter was called for a layer that does not "
            "own a supported Gridbook BMM contract"
        )
    if o.ndim != 3:
        raise RuntimeError(
            f"Gridbook DSV4 wo_a expects [tokens, heads, head_dim], got "
            f"{tuple(o.shape)}"
        )
    groups = int(attention.n_local_groups)
    if groups <= 0 or int(o.shape[1]) % groups != 0:
        raise RuntimeError(
            f"Gridbook DSV4 wo_a cannot partition {o.shape[1]} heads into "
            f"{groups} local groups"
        )

    rotary = attention.rotary_emb
    forward_native = getattr(rotary, "forward_native", None)
    if not callable(forward_native):
        raise RuntimeError(
            "Gridbook DSV4 wo_a needs vLLM rotary_emb.forward_native with "
            "inverse=True; the pinned DSV4 runtime ABI is unavailable"
        )
    o_ref, key_ref = forward_native(
        positions, o, None, inverse=True
    )
    if key_ref is not None:
        raise RuntimeError("inverse RoPE unexpectedly returned a key tensor")
    grouped = o_ref.reshape(int(o.shape[0]), groups, -1)
    z = wo_a(grouped)
    if isinstance(z, tuple):
        z = z[0]
    expected = (int(o.shape[0]), groups, int(attention.o_lora_rank))
    if tuple(z.shape) != expected:
        raise RuntimeError(
            f"Gridbook DSV4 wo_a returned {tuple(z.shape)}, expected "
            f"{expected}"
        )
    result = attention.wo_b(z.flatten(1))
    if isinstance(result, tuple):
        result = result[0]
    return result


# Compatibility spelling used by the existing direct-MXFP8 lane and callers.
dsv4_mxfp8_o_proj = dsv4_gridbook_o_proj


def _wrap_attention_class(cls: type) -> bool:
    current = cls._o_proj
    abi = getattr(current, _WRAPPED_ABI_ATTR, None)
    if abi == _ADAPTER_ABI:
        return False
    if abi is not None:
        raise RuntimeError(
            f"{cls.__module__}.{cls.__name__} already carries Gridbook DSV4 "
            f"wo_a adapter ABI {abi!r}, expected {_ADAPTER_ABI}"
        )

    @functools.wraps(current)
    def _o_proj(self, o, positions, _original=current):
        if _owns_gridbook_bmm(getattr(self, "wo_a", None)):
            return dsv4_gridbook_o_proj(self, o, positions)
        return _original(self, o, positions)

    setattr(_o_proj, _WRAPPED_ABI_ATTR, _ADAPTER_ABI)
    setattr(_o_proj, "_gridbook_original_o_proj", current)
    cls._o_proj = _o_proj
    return True


def install_dsv4_woa_adapter() -> None:
    """Wrap already-loaded NVIDIA DSV4 attention classes, idempotently.

    Gridbook is asked for ``wo_a``'s quant method while DSV4's NVIDIA model
    module is constructing, so the selected attention module is already in
    ``sys.modules``.  Refusing when no known class is loaded keeps a future
    incompatible vLLM layout from silently taking an unquantized path.
    """

    global _LOGGED
    seen = 0
    changed = 0
    for module_name, class_names in _NVIDIA_ATTN_CLASSES.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            seen += 1
            changed += int(_wrap_attention_class(cls))
    if seen == 0:
        raise RuntimeError(
            "Gridbook BMM encountered DSV4 wo_a, but no admitted vLLM NVIDIA "
            "DSV4 attention class is loaded"
        )
    if not _LOGGED:
        print(
            "[prismaquant-cb] dsv4_wo_a=gridbook-bmm "
            f"adapter_abi={_ADAPTER_ABI} classes={seen} changed={changed}",
            flush=True,
        )
        _LOGGED = True
