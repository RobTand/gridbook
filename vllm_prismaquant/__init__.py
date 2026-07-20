"""vLLM out-of-tree plugin for PrismaQuant NVFP4-CB / FP8-CB codebook formats.

Prototype (i) of docs/nvfp4-cb-plan/serving-kernel.md: a correctness-first,
Triton-based serving path (INV-1 honored, INV-2 waived). Not production-eligible.

``register`` is lazy so ``import vllm_prismaquant.kernels`` (the correctness
tests) works without vLLM installed.
"""

__version__ = "0.0.1"


def register() -> None:
    from .plugin import register as _register
    _register()
