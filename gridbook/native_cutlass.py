"""Fail-closed bindings to vLLM's compiled NVIDIA quantization/CUTLASS ops.

Gridbook deliberately bypasses ``vllm._custom_ops.cutlass_scaled_mm``: that
Python convenience wrapper contains a Triton fallback for incompatible matrix
shapes. The functions here validate the exact native contract and invoke the
registered ``torch.ops._C`` operators directly. A missing ABI or unsupported
shape is a model-load/runtime error, never an implementation switch.
"""
from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import threading

import torch

from .cuda_ext import NativeKernelUnavailableError


_REQUIRED_OPS = (
    "dynamic_per_token_scaled_fp8_quant",
    "cutlass_scaled_mm",
)

_NATIVE_MOE_ACTIVATION_OPS = {
    "silu": "silu_and_mul",
    "gelu": "gelu_and_mul",
    "gelu_tanh": "gelu_tanh_and_mul",
    "gelu_pytorch_tanh": "gelu_tanh_and_mul",
    "swigluoai": "swigluoai_and_mul",
}

_NATIVE_EXTENSION_MODULE = "vllm._C_stable_libtorch"
_NATIVE_EXTENSION_LOCK = threading.Lock()


def _import_native_ops(context: str) -> None:
    # When vLLM already loaded the stable extension, its registered operator is
    # sufficient proof and avoids touching the Python package at all.
    if callable(getattr(torch.ops._C, "cutlass_scaled_mm", None)):
        return
    with _NATIVE_EXTENSION_LOCK:
        if callable(getattr(torch.ops._C, "cutlass_scaled_mm", None)):
            return
        try:
            # Importing ``vllm.platforms`` or even the ``vllm`` package loads
            # optional compiler backends, including Triton in the pinned image.
            # Locate the distribution without importing it, then load the
            # compiled extension by filename under its ABI-bearing module name.
            distribution = importlib.metadata.distribution("vllm")
            package_dir = Path(distribution.locate_file("vllm"))
            suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
            candidates = sorted(
                path for path in package_dir.glob("_C_stable_libtorch*")
                if path.is_file() and path.name.endswith(suffixes)
            )
            if len(candidates) != 1:
                raise ImportError(
                    f"expected one vLLM stable extension under {package_dir}, "
                    f"found {[path.name for path in candidates]}")
            spec = importlib.util.spec_from_file_location(
                _NATIVE_EXTENSION_MODULE, candidates[0])
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"cannot create an extension spec for {candidates[0]}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_NATIVE_EXTENSION_MODULE] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                if sys.modules.get(_NATIVE_EXTENSION_MODULE) is module:
                    sys.modules.pop(_NATIVE_EXTENSION_MODULE, None)
                raise
        except Exception as exc:  # noqa: BLE001 - contract error has context
            raise NativeKernelUnavailableError(
                f"{context}: cannot load vLLM's compiled CUDA operators "
                "without importing a compiler backend") from exc


def _activation_value(activation) -> str:
    value = getattr(activation, "value", activation)
    return str(value).strip().lower()


def require_native_fp8_cutlass(context: str) -> None:
    """Load and attest the pinned vLLM native operator ABI."""
    _import_native_ops(context)
    missing = [name for name in _REQUIRED_OPS
               if not callable(getattr(torch.ops._C, name, None))]
    if missing:
        raise NativeKernelUnavailableError(
            f"{context}: pinned vLLM ABI is missing native operators: "
            f"{', '.join(missing)}")


def require_native_fp4_quant(context: str) -> None:
    """Load and attest vLLM's directly registered CUDA NVFP4 quantizer."""
    _import_native_ops(context)
    if not callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
        raise NativeKernelUnavailableError(
            f"{context}: pinned vLLM ABI is missing native operator "
            "scaled_fp4_quant")


def require_native_moe_activation(activation, context: str) -> str:
    """Resolve one direct compiled activation op or fail during model load.

    vLLM's ``apply_moe_activation`` convenience helper contains a reachable
    Triton SWIGLUSTEP branch. Gridbook does not call that helper: it accepts
    only activation kinds backed by a registered native ``torch.ops._C`` op.
    """
    _import_native_ops(context)
    value = _activation_value(activation)
    op_name = _NATIVE_MOE_ACTIVATION_OPS.get(value)
    if op_name is None:
        raise NativeKernelUnavailableError(
            f"{context}: activation {value!r} has no direct native Gridbook "
            "operator")
    if not callable(getattr(torch.ops._C, op_name, None)):
        raise NativeKernelUnavailableError(
            f"{context}: pinned vLLM ABI is missing native activation "
            f"operator {op_name}")
    return value


def native_moe_activation(
    activation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply a gated MoE activation through a direct compiled CUDA op."""
    if input.device.type != "cuda" or output.device.type != "cuda":
        raise NativeKernelUnavailableError(
            "native MoE activation requires CUDA tensors")
    if input.dim() != 2 or output.dim() != 2:
        raise ValueError("native MoE activation requires 2-D tensors")
    if output.shape[0] != input.shape[0] \
            or output.shape[1] * 2 != input.shape[1]:
        raise ValueError(
            f"invalid gated activation shapes input={tuple(input.shape)} "
            f"output={tuple(output.shape)}")
    value = _activation_value(activation)
    op_name = _NATIVE_MOE_ACTIVATION_OPS.get(value)
    if op_name is None:
        raise NativeKernelUnavailableError(
            f"activation {value!r} has no direct native Gridbook operator")
    op = getattr(torch.ops._C, op_name, None)
    if not callable(op):
        raise NativeKernelUnavailableError(
            f"native activation operator {op_name} is unavailable")
    op(output, input)
    return output


def native_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic E4M3 quantization through vLLM's native CUDA op."""
    if x.device.type != "cuda" or x.dim() != 2:
        raise NativeKernelUnavailableError(
            "native FP8 quantization requires a 2-D CUDA tensor")
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32,
                        device=x.device)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        out, x, scale, None)
    return out, scale


def native_fp4_quant(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-global-scale NVFP4 quantization via the compiled CUDA op.

    This reproduces the native 128x4 scale-factor layout expected by the
    Gridbook decode-in-prologue CUTLASS kernel without importing vLLM's Python
    convenience wrapper.
    """
    if x.device.type != "cuda" or x.dim() != 2:
        raise NativeKernelUnavailableError(
            "native FP4 quantization requires a 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"native FP4 quantization requires BF16/FP16 input, got {x.dtype}")
    if x.shape[1] % 16:
        raise ValueError(
            "native FP4 quantization requires K divisible by 16; "
            f"got {x.shape[1]}")
    if input_global_scale.device != x.device \
            or input_global_scale.numel() != 1:
        raise ValueError(
            "native FP4 quantization requires one global-scale value on the "
            "input device")
    if input_global_scale.dtype != torch.float32:
        raise TypeError(
            "native FP4 quantization requires a float32 global scale, got "
            f"{input_global_scale.dtype}")
    require_native_fp4_quant("native FP4 quantization")
    packed, scale_factors = torch.ops._C.scaled_fp4_quant(
        x, input_global_scale, True)
    return packed, scale_factors.view(torch.float8_e4m3fn)


def native_cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Invoke native CUTLASS scaled-MM with no Triton-compatible fallback."""
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise NativeKernelUnavailableError(
            "native CUTLASS scaled-MM requires CUDA tensors")
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(
            f"invalid scaled-MM shapes a={tuple(a.shape)} b={tuple(b.shape)}")
    # This is the exact guard in vLLM's wrapper before it chooses CUTLASS.
    # Enforce it here so a future odd partition cannot change kernel family.
    if b.shape[0] % 16 or b.shape[1] % 16:
        raise NativeKernelUnavailableError(
            f"CUTLASS scaled-MM requires B dimensions divisible by 16; "
            f"got {tuple(b.shape)}")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported CUTLASS output dtype {out_dtype}")
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype,
                      device=a.device)
    torch.ops._C.cutlass_scaled_mm(
        out, a, b, scale_a, scale_b, None)
    return out
