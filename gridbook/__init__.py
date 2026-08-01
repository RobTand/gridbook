"""vLLM out-of-tree plugin for the NVFP4-CB / FP8-CB product-codebook formats.

Registers a ``QuantizationConfig`` (``"gridbook"``, with ``"prismaquant"`` kept
as a legacy alias) plus the linear and fused-MoE methods that serve
codebook-quantized weights. The CUDA kernels ship as sources under
``gridbook/csrc`` and are JIT-compiled on first use; without nvcc the plugin
falls back to a correct-but-slow Triton path.

``register`` is lazy so ``import gridbook.codec`` / ``import gridbook.kernels``
(the format and correctness tests) work without vLLM installed.
"""

# Single source of truth for package and release metadata. Gridbook is the sole
# owner of its runtime; producer repositories consume a released version or an
# immutable commit and never mirror this package tree.
__version__ = "0.4.2"


def register() -> None:
    from .plugin import register as _register
    _register()
