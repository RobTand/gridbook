"""Get ourselves consulted for the embedding on architectures that never ask.

vLLM builds a quantized module by calling ``quant_config.get_quant_method(layer,
prefix)`` from inside the layer's own constructor.  A model class that builds

    self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)

with neither ``quant_config`` nor ``prefix`` therefore can **never** serve a
quantized lookup table, no matter what the checkpoint declares -- the dispatch
in ``config.get_quant_method`` and the method in ``embedding.py`` are both
present and correct, and neither is ever reached.  The failure is silent at
construction and surfaces much later as vLLM refusing an unclaimed parameter:

    ValueError: There is no module or parameter named
    'embed_tokens.weight_global_scale' in Qwen3_5Model.  The available
    parameters belonging to embed_tokens (VocabParallelEmbedding) are:
    {'embed_tokens.weight'}

Most vLLM model classes pass both arguments (``deepseek_v2``, ``arctic``,
``dbrx``, ...).  ``qwen3_5`` does not, and Qwen3.8-27B's NVFP4 embedding is
1.83 GB of a 12.98 GB artifact -- serving that table in BF16 instead pushes the
artifact from 12.09 GiB to 13.79 GiB and leaves roughly 2 GiB on a 16 GiB card
for CUDA context, KV and activations.  So this is the difference between the
artifact fitting its target card and not.

WHAT THIS INSTALLS.  A wrap on the inner model class that supplies the two
arguments that class's own ``__init__`` already holds (it takes ``vllm_config``
and ``prefix``, and even stores ``self.quant_config``).  It is deliberately
inert unless ALL of:

  * the ambient quant config is ours (duck-typed on ``_embedding_format``),
  * that config declares an embedding unit for this exact prefix, and
  * the embedding is being constructed without a ``quant_config``/``prefix``,

so nothing changes for another checkpoint, another architecture, a model class
that already does this correctly, or a future vLLM that starts passing the
arguments itself.  ``ParallelLMHead`` is excluded: it subclasses
``VocabParallelEmbedding`` but is an output projection served through
``config_groups`` (``embedding.parse_declaration`` refuses it on the producer
side too).

WHY SUBSTITUTE DURING CONSTRUCTION rather than replace the module afterwards.
Rebuilding ``embed_tokens`` post-hoc would first allocate the full BF16 table --
2.5 GB for this artifact -- and free it, and the artifact this exists for is
precisely the one being served under a deliberately tight memory budget.  The
argument substitution costs nothing.
"""
from __future__ import annotations

import contextlib
import functools
import inspect

__all__ = [
    "embedding_prefix",
    "install_quantized_embedding_construction",
]

_INSTALLED_FLAG = "_gridbook_embedding_construction_installed"


def embedding_prefix(model_prefix: str) -> str:
    """The vLLM module name of the lookup table under *model_prefix*."""
    model_prefix = str(model_prefix or "")
    return f"{model_prefix}.embed_tokens" if model_prefix else "embed_tokens"


@contextlib.contextmanager
def _supplying(quant_config, prefix: str):
    """Make an argument-less ``VocabParallelEmbedding`` build ask *quant_config*.

    Scoped to one model ``__init__`` so the only construction it can reach is
    that model's own lookup table.  The signature is bound rather than pattern
    matched on ``kwargs`` so a caller passing ``quant_config`` positionally is
    still recognised as having supplied it.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    original = VocabParallelEmbedding.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        try:
            bound = signature.bind_partial(self, *args, **kwargs)
        except TypeError:
            # A signature we cannot read is one we must not rewrite.
            return original(self, *args, **kwargs)
        already_declared = (
            bound.arguments.get("quant_config") is not None
            or bool(bound.arguments.get("prefix"))
        )
        if not already_declared and not isinstance(self, ParallelLMHead):
            kwargs["quant_config"] = quant_config
            kwargs["prefix"] = prefix
        return original(self, *args, **kwargs)

    VocabParallelEmbedding.__init__ = patched
    try:
        yield
    finally:
        VocabParallelEmbedding.__init__ = original


def _declared_embedding(args, kwargs) -> tuple[object | None, str]:
    """``(quant_config, prefix)`` when THIS model declares a quantized table.

    Returns ``(None, "")`` for every case that must stay untouched.  Resolution
    errors are deliberately NOT swallowed: this runs before the first Linear is
    built, so a config that cannot resolve here would fail moments later in
    ``get_quant_method`` anyway, and hiding it would turn a loud failure into an
    embedding that is silently served unquantized -- the exact class of bug this
    module exists to close.
    """
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is None and args:
        vllm_config = args[0]
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return None, ""
    lookup = getattr(quant_config, "_embedding_format", None)
    if not callable(lookup):
        return None, ""          # somebody else's quantization
    resolve = getattr(quant_config, "_ensure_resolved", None)
    if callable(resolve):
        # A pointer config parses its declaration lazily, and the embedding is
        # the FIRST module built -- before any get_quant_method call has forced
        # resolution.  Without this the units are still empty and the lookup
        # below reports "not declared" for an artifact that declares it.
        resolve()
    prefix = embedding_prefix(kwargs.get("prefix", ""))
    if lookup(prefix) is None:
        return None, ""
    return quant_config, prefix


def install_quantized_embedding_construction(model_cls: type) -> None:
    """Wrap *model_cls* so its lookup table is offered to our dispatch.

    Idempotent across repeated plugin loads, and a no-op on a class whose
    ``__init__`` we cannot wrap.
    """
    if getattr(model_cls, _INSTALLED_FLAG, False):
        return
    original_init = model_cls.__init__
    if not callable(original_init):
        return

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        quant_config, prefix = _declared_embedding(args, kwargs)
        if quant_config is None:
            return original_init(self, *args, **kwargs)
        with _supplying(quant_config, prefix):
            return original_init(self, *args, **kwargs)

    model_cls.__init__ = __init__
    setattr(model_cls, _INSTALLED_FLAG, True)
