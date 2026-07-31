"""vLLM out-of-tree plugin for the NVFP4-CB / FP8-CB product-codebook formats.

Registers a ``QuantizationConfig`` (``"gridbook"``, with ``"prismaquant"`` kept
as a legacy alias) plus the linear and fused-MoE methods that serve
codebook-quantized weights. The CUDA kernels ship as sources under
``gridbook/csrc`` and are JIT-compiled on first use; without nvcc the plugin
falls back to a correct-but-slow Triton path.

``register`` is lazy so ``import gridbook.codec`` / ``import gridbook.kernels``
(the format and correctness tests) work without vLLM installed.
"""

# Development head, ahead of the last PyPI release (0.1.1), hence the dev suffix.
# This file is identical in both trees that hold this package: prismaquant's
# plugins/gridbook (where the work lands) and the RobTand/gridbook release repo
# (which builds and publishes it). prismaquant's scripts/sync_gridbook.py is the
# one-way path between them; tests/test_gridbook_sync.py is the drift gate.
# Releasing — the tag and the PyPI upload — happens only in the release repo.
__version__ = "0.2.0.dev0"


def register() -> None:
    from .plugin import register as _register
    _register()
