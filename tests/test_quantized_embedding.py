"""Quantized-embedding declaration, decode, and the dispatch hazard.

The parity of GridBook's decode against PrismaQuant's exporter is proved
elsewhere, on the real 248320x5120 tensor, because that is a cross-package
question and the two packages do not import each other
(`dq-runs/qwen38-27b/embed_nvfp4_parity.py`: bit-identical on all three
checks).  What is tested HERE is what lives in this package:

  * the declaration schema fails closed on every axis a producer can get wrong;
  * the decode reproduces an independently-written implementation of the
    documented on-disk contract -- written from the spec in the module header,
    not transcribed from the implementation, so agreement is evidence;
  * gather is not a second approximation of full decode;
  * ``ParallelLMHead`` is NOT claimed by the embedding path.  It subclasses
    ``VocabParallelEmbedding``, so an `isinstance` written the obvious way
    silently takes the output projection off vLLM's GEMM path and runs it as a
    lookup.  That is the single most dangerous mistake available in this file
    and it deserves a test that fails loudly rather than a comment.
"""
from __future__ import annotations

import pytest
import torch

from gridbook.embedding import (
    EmbeddingFormatError,
    FORMATS,
    GridbookNVFP4EmbeddingMethod,
    SCHEMA_KEY,
    nvfp4_gather_dequant,
    parse_declaration,
)


def _identity(name: str) -> str:
    return name


# --------------------------------------------------------------------------
# Declaration schema
# --------------------------------------------------------------------------

def test_absence_is_a_no_op():
    """Every published artifact predates this schema."""
    assert parse_declaration({}, canonicalize=_identity) == {}


def test_parses_a_valid_declaration():
    units = parse_declaration(
        {SCHEMA_KEY: {"version": 1, "units": {"model.embed_tokens": "nvfp4"}}},
        canonicalize=_identity)
    assert units == {"model.embed_tokens": FORMATS["nvfp4"]}


def test_unknown_schema_version_refuses():
    with pytest.raises(EmbeddingFormatError, match="schema version"):
        parse_declaration(
            {SCHEMA_KEY: {"version": 2, "units": {"e": "nvfp4"}}},
            canonicalize=_identity)


def test_unknown_format_refuses():
    with pytest.raises(EmbeddingFormatError, match="has not audited"):
        parse_declaration(
            {SCHEMA_KEY: {"version": 1, "units": {"e": "int3_secret"}}},
            canonicalize=_identity)


def test_unit_claimed_by_both_vocabularies_refuses():
    """Two dispatches owning one unit's resident weights is unresolvable."""
    with pytest.raises(EmbeddingFormatError, match="claimed by both"):
        parse_declaration(
            {SCHEMA_KEY: {"version": 1,
                          "units": {"model.embed_tokens": "nvfp4"}}},
            canonicalize=_identity,
            cb_targets=frozenset({"model.embed_tokens"}))


def test_empty_units_refuses_rather_than_silently_serving_bf16():
    with pytest.raises(EmbeddingFormatError, match="declares no units"):
        parse_declaration({SCHEMA_KEY: {"version": 1, "units": {}}},
                          canonicalize=_identity)


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

def _reference_decode(packed_u8, scale_fp8, global_real, group=16):
    """The on-disk contract, written from the spec rather than the code.

        weight_packed  two E2M1 nibbles per byte, EVEN element in the LOW nibble
        nibble         bit 3 = sign, bits 0..2 index [0,.5,1,1.5,2,3,4,6]
        value          sign * magnitude * scale.float() * global_real
    """
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    rows, half = packed_u8.shape
    cols = half * 2
    out = torch.empty(rows, cols, dtype=torch.float32,
                      device=packed_u8.device)
    scale = scale_fp8.to(torch.float32) * float(global_real)
    pk = packed_u8.to(torch.int64)
    for c in range(cols):
        byte = pk[:, c // 2]
        nib = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
        mag = torch.tensor(mags, device=pk.device)[nib & 0x7]
        sign = torch.where((nib >> 3) == 1, -1.0, 1.0)
        out[:, c] = sign * mag * scale[:, c // group]
    return out


@pytest.fixture
def packed_case():
    torch.manual_seed(7)
    rows, cols = 6, 64
    codes = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.rand(rows, cols // 16) * 4 + 0.25).to(torch.float8_e4m3fn)
    return packed, scale, torch.tensor([0.0037], dtype=torch.float32)


def test_decode_matches_an_independent_spec_implementation(packed_case):
    packed, scale, g = packed_case
    rows = torch.arange(packed.shape[0])
    got = nvfp4_gather_dequant(packed, scale.view(torch.uint8), g, rows,
                               out_dtype=torch.float32)
    want = _reference_decode(packed, scale, float(g))
    assert torch.equal(got, want)


def test_gather_is_not_a_second_approximation(packed_case):
    packed, scale, g = packed_case
    full = nvfp4_gather_dequant(packed, scale.view(torch.uint8), g,
                                torch.arange(packed.shape[0]),
                                out_dtype=torch.float32)
    pick = torch.tensor([4, 0, 4, 2])
    sub = nvfp4_gather_dequant(packed, scale.view(torch.uint8), g, pick,
                               out_dtype=torch.float32)
    assert torch.equal(sub, full.index_select(0, pick))


def test_token_id_shape_is_preserved(packed_case):
    packed, scale, g = packed_case
    ids = torch.randint(0, packed.shape[0], (3, 5))
    out = nvfp4_gather_dequant(packed, scale.view(torch.uint8), g, ids,
                               out_dtype=torch.bfloat16)
    assert out.shape == (3, 5, packed.shape[1] * 2)
    assert out.dtype == torch.bfloat16


def test_sign_bit_is_not_inverted(packed_case):
    """A sign flip is invisible in an aggregate error metric on random codes."""
    packed = torch.tensor([[0x02]], dtype=torch.uint8)   # code 2 (+1.0), code 0
    scale = torch.tensor([[1.0]]).to(torch.float8_e4m3fn)
    g = torch.tensor([1.0])
    out = nvfp4_gather_dequant(packed, scale.view(torch.uint8), g,
                               torch.tensor([0]), group_size=2,
                               out_dtype=torch.float32)
    assert out[0, 0].item() == pytest.approx(1.0)        # low nibble = even
    packed_neg = torch.tensor([[0x0A]], dtype=torch.uint8)  # 0b1010 -> -1.0
    out_neg = nvfp4_gather_dequant(packed_neg, scale.view(torch.uint8), g,
                                   torch.tensor([0]), group_size=2,
                                   out_dtype=torch.float32)
    assert out_neg[0, 0].item() == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# The dispatch hazard
# --------------------------------------------------------------------------

def test_parallel_lm_head_is_refused_by_the_embedding_method():
    """ParallelLMHead subclasses VocabParallelEmbedding. It is not a lookup.

    Dispatch in ``config.get_quant_method`` excludes it; this is the method's
    own second line of defence, so a future refactor of that seam fails here
    rather than in a served artifact's logits.
    """
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    method = GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "lm_head")
    head = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(head)
    with pytest.raises(EmbeddingFormatError, match="ParallelLMHead"):
        method.create_weights(head, 128, [64], 128, 64, torch.bfloat16)


def test_create_weights_makes_it_an_instance_of_QuantizeMethodBase():
    """The bug this exists for shipped once and cost a load smoke to find.

    vLLM's post-load sweep picks the modules to finalize with
    ``isinstance(quant_method, QuantizeMethodBase)`` -- NOMINAL, not
    structural.  Implementing the full surface without being a subclass meant
    ``process_weights_after_loading`` was never called: weights loaded, the
    dispatch was right, the artifact inspected clean, and the model died on its
    first forward with a missing derived attribute.

    Stubbed rather than skipped.  The obvious spelling of this test is
    ``importorskip('vllm')``, but vLLM is not installed in the build venv where
    this suite runs, so that spelling would SKIP here and guard nothing -- the
    precise condition under which the original bug survived to a container.  A
    local ABC reproduces the only thing the sweep relies on.
    """
    import abc
    import sys
    import types

    base_mod = "vllm.model_executor.layers.quantization.base_config"
    created = []
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.quantization", base_mod):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            created.append(name)

    class _Base(abc.ABC):
        @abc.abstractmethod
        def create_weights(self): ...
        @abc.abstractmethod
        def apply(self): ...

    mod = sys.modules[base_mod]
    had = getattr(mod, "QuantizeMethodBase", None)
    mod.QuantizeMethodBase = _Base
    try:
        from gridbook.embedding import _bind_to_quantize_method_base

        assert not isinstance(
            GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "e"), _Base), (
            "precondition: not a subclass before binding")
        _bind_to_quantize_method_base()
        assert isinstance(
            GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "e"), _Base), (
            "vLLM's post-load sweep would skip this method, and the model "
            "would fail on its first forward rather than at load")
        _bind_to_quantize_method_base()      # idempotent

        # The mechanism working is not the same as it being INVOKED. Binding
        # has to happen before the sweep, and create_weights is the only hook
        # that always runs first; exercising create_weights itself needs a
        # real vLLM, so the call site is asserted directly. Crude, but it
        # fails if someone deletes the one line that makes this work.
        import inspect
        src = inspect.getsource(
            GridbookNVFP4EmbeddingMethod.create_weights)
        assert "_bind_to_quantize_method_base()" in src, (
            "create_weights no longer binds the base class; the post-load "
            "sweep will skip this method again")
    finally:
        if had is not None:
            mod.QuantizeMethodBase = had
        for name in created:
            sys.modules.pop(name, None)


def test_apply_refuses_rather_than_faking_a_gemm():
    method = GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "model.embed_tokens")
    with pytest.raises(EmbeddingFormatError, match="no GEMM path"):
        method.apply(torch.nn.Module(), torch.zeros(1))


def test_tied_weights_are_refused():
    """A tied lm_head would inherit rounding the allocator never priced."""
    method = GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "model.embed_tokens")
    with pytest.raises(EmbeddingFormatError, match="ties its"):
        method.tie_weights(torch.nn.Module(), torch.nn.Module())


def test_degenerate_global_scale_is_refused():
    """Inverting a nonpositive divisor would poison every token silently."""
    method = GridbookNVFP4EmbeddingMethod(FORMATS["nvfp4"], "model.embed_tokens")
    layer = torch.nn.Module()
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([0.0]), requires_grad=False)
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros(2, 8, dtype=torch.uint8), requires_grad=False)
    with pytest.raises(EmbeddingFormatError, match="nonpositive"):
        method.process_weights_after_loading(layer)
