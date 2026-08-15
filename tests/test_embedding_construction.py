"""A model class that never asks must still get a quantized embedding.

vLLM dispatches a quantized module from inside the layer constructor, so a
model that builds ``VocabParallelEmbedding(vocab, hidden)`` with no
``quant_config`` and no ``prefix`` can never reach our embedding method.  That
is not hypothetical: ``vllm/model_executor/models/qwen3_5.py`` does exactly
this, and the Qwen3.8-27B artifact's declared NVFP4 embedding -- 1.83 GB of a
12.98 GB artifact -- died at load with

    ValueError: There is no module or parameter named
    'embed_tokens.weight_global_scale' in Qwen3_5Model

The bar here is both directions.  The wrap must fire for a declared unit, and
must be *provably inert* for everything else, because it runs on every arch we
install on.
"""
from __future__ import annotations

import inspect

import pytest

from gridbook.embedding_construction import (
    embedding_prefix,
    install_quantized_embedding_construction,
)


class _FakeVocabParallelEmbedding:
    """Stands in for vLLM's class: records what it was constructed with."""

    def __init__(self, num_embeddings, embedding_dim, *,
                 quant_config=None, prefix=""):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_config = quant_config
        self.prefix = prefix


class _FakeParallelLMHead(_FakeVocabParallelEmbedding):
    """Subclasses the embedding in vLLM too -- and must never be claimed."""


class _FakeQuantConfig:
    """Duck-types the parts of PrismaQuantConfig the wrap touches."""

    def __init__(self, units, *, resolve_populates=None):
        self._units = dict(units)
        self._resolve_populates = resolve_populates
        self.resolved = 0

    def _ensure_resolved(self):
        self.resolved += 1
        if self._resolve_populates is not None:
            self._units = dict(self._resolve_populates)

    def _embedding_format(self, prefix):
        return self._units.get(prefix)


class _FakeVllmConfig:
    def __init__(self, quant_config):
        self.quant_config = quant_config


@pytest.fixture(autouse=True)
def _patch_vllm_embedding(monkeypatch):
    """Route the wrap's vLLM import at our fakes (no vLLM needed to test)."""
    module = type(inspect)("vllm.model_executor.layers.vocab_parallel_embedding")
    module.VocabParallelEmbedding = _FakeVocabParallelEmbedding
    module.ParallelLMHead = _FakeParallelLMHead
    import sys

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        module,
    )
    yield


def _model_class():
    """A fresh class shaped like Qwen3_5Model: holds both arguments, passes
    neither."""

    class _Model:
        def __init__(self, *, vllm_config, prefix=""):
            self.quant_config = vllm_config.quant_config
            self.embed_tokens = _FakeVocabParallelEmbedding(248320, 5120)

    return _Model


def test_a_declared_embedding_reaches_our_dispatch():
    cls = _model_class()
    install_quantized_embedding_construction(cls)
    qc = _FakeQuantConfig({"model.embed_tokens": "nvfp4"})

    model = cls(vllm_config=_FakeVllmConfig(qc), prefix="model")

    assert model.embed_tokens.quant_config is qc
    assert model.embed_tokens.prefix == "model.embed_tokens"


def test_the_declaration_is_parsed_before_the_first_lookup():
    """The embedding is the FIRST module built, so nothing has forced the
    pointer config to resolve yet.  Without `_ensure_resolved` the units are
    still empty and a declared table is silently served unquantized."""
    cls = _model_class()
    install_quantized_embedding_construction(cls)
    qc = _FakeQuantConfig({}, resolve_populates={"model.embed_tokens": "nvfp4"})

    model = cls(vllm_config=_FakeVllmConfig(qc), prefix="model")

    assert qc.resolved == 1
    assert model.embed_tokens.quant_config is qc


def test_an_undeclared_embedding_is_left_alone():
    cls = _model_class()
    install_quantized_embedding_construction(cls)
    qc = _FakeQuantConfig({"model.language_model.embed_tokens": "nvfp4"})

    model = cls(vllm_config=_FakeVllmConfig(qc), prefix="model")

    assert model.embed_tokens.quant_config is None
    assert model.embed_tokens.prefix == ""


def test_somebody_elses_quantization_is_left_alone():
    class _OtherConfig:
        pass

    cls = _model_class()
    install_quantized_embedding_construction(cls)

    model = cls(vllm_config=_FakeVllmConfig(_OtherConfig()), prefix="model")

    assert model.embed_tokens.quant_config is None


def test_an_unquantized_model_is_left_alone():
    cls = _model_class()
    install_quantized_embedding_construction(cls)

    model = cls(vllm_config=_FakeVllmConfig(None), prefix="model")

    assert model.embed_tokens.quant_config is None


def test_a_model_that_already_passes_its_own_arguments_is_not_second_guessed():
    """A future vLLM that fixes this upstream must not have its own values
    overwritten by ours."""

    class _CorrectModel:
        def __init__(self, *, vllm_config, prefix=""):
            self.embed_tokens = _FakeVocabParallelEmbedding(
                248320, 5120,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )

    install_quantized_embedding_construction(_CorrectModel)
    qc = _FakeQuantConfig({"model.embed_tokens": "nvfp4"})

    model = _CorrectModel(vllm_config=_FakeVllmConfig(qc), prefix="model")

    assert model.embed_tokens.quant_config is qc
    assert model.embed_tokens.prefix == "model.embed_tokens"


def test_the_output_projection_is_never_claimed():
    """ParallelLMHead subclasses VocabParallelEmbedding but is a GEMM served
    through config_groups; claiming it here would take the head off that path."""

    class _ModelWithHead:
        def __init__(self, *, vllm_config, prefix=""):
            self.lm_head = _FakeParallelLMHead(248320, 5120)
            self.embed_tokens = _FakeVocabParallelEmbedding(248320, 5120)

    install_quantized_embedding_construction(_ModelWithHead)
    qc = _FakeQuantConfig({"model.embed_tokens": "nvfp4"})

    model = _ModelWithHead(vllm_config=_FakeVllmConfig(qc), prefix="model")

    assert model.lm_head.quant_config is None
    assert model.embed_tokens.quant_config is qc


def test_the_patch_is_reverted_after_construction():
    """The substitution is scoped to one __init__; a later build elsewhere must
    see the untouched constructor."""
    cls = _model_class()
    install_quantized_embedding_construction(cls)
    qc = _FakeQuantConfig({"model.embed_tokens": "nvfp4"})
    cls(vllm_config=_FakeVllmConfig(qc), prefix="model")

    loose = _FakeVocabParallelEmbedding(248320, 5120)
    assert loose.quant_config is None


def test_install_is_idempotent():
    cls = _model_class()
    install_quantized_embedding_construction(cls)
    first = cls.__init__
    install_quantized_embedding_construction(cls)
    assert cls.__init__ is first


def test_a_construction_error_does_not_leave_the_constructor_patched():
    class _Exploding:
        def __init__(self, *, vllm_config, prefix=""):
            raise RuntimeError("boom")

    install_quantized_embedding_construction(_Exploding)
    qc = _FakeQuantConfig({"model.embed_tokens": "nvfp4"})
    with pytest.raises(RuntimeError):
        _Exploding(vllm_config=_FakeVllmConfig(qc), prefix="model")

    loose = _FakeVocabParallelEmbedding(248320, 5120)
    assert loose.quant_config is None


@pytest.mark.parametrize("model_prefix,expected", [
    ("model", "model.embed_tokens"),
    ("language_model.model", "language_model.model.embed_tokens"),
    ("", "embed_tokens"),
])
def test_embedding_prefix(model_prefix, expected):
    assert embedding_prefix(model_prefix) == expected
