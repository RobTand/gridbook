"""vLLM out-of-tree plugin for the NVFP4-CB / FP8-CB product-codebook formats.

Registers a ``QuantizationConfig`` (``"gridbook"``, with ``"prismaquant"`` kept
as a legacy alias) plus the linear and fused-MoE methods that serve
codebook-quantized weights. Native CUDA/CUTLASS kernel sources ship under
``gridbook/csrc`` and are JIT-compiled during production model load (or lazily
for direct low-level callers). Serving fails closed when a required native
kernel is unavailable; Gridbook has no Triton runtime lane.

``register`` is lazy so format and native-kernel contract tests can import the
package without vLLM installed.
"""

# Single source of truth for package and release metadata. Gridbook is the sole
# owner of its runtime; producer repositories consume a released version or an
# immutable commit and never mirror this package tree.
__version__ = "0.8.2"


def register() -> None:
    from .plugin import register as _register
    _register()
