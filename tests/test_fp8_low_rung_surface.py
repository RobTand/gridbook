"""CPU/static guards for the v10 FP8-CB K4..K48 producer surface.

Nothing in this module imports or builds a CUDA extension.  It pins the source
laws that a later cross-compile and physical serve must exercise, including the
K4 last-codeword shared-memory boundary that motivated the guarded word load.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

import pytest

from gridbook import codec
from gridbook.runtime_contract import load_runtime_contract


def _source_root(test_file: Path | None = None) -> Path:
    """Locate source files in-tree or from the installed-wheel test gate."""
    location = Path(__file__) if test_file is None else Path(test_file)
    roots = [location.resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        if (root / "gridbook" / "__init__.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "no Gridbook source checkout: run in-tree or set GRIDBOOK_SOURCE_ROOT")


_ROOT = _source_root()
_PRODUCER_RUNGS = tuple(range(4, 49, 4))
_READER_RUNGS = (4, 8, 12, 16, 20, 24, *range(28, 49))


def _source(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "variable", ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE")
)
def test_source_root_honors_installed_wheel_locator(
    monkeypatch, tmp_path, variable,
):
    source_root = tmp_path / "checkout"
    package = source_root / "gridbook"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("GRIDBOOK_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("GITHUB_WORKSPACE", raising=False)
    monkeypatch.setenv(variable, str(source_root))

    staged_test = tmp_path / "wheel-tests" / "tests" / "test_surface.py"
    assert _source_root(staged_test) == source_root.resolve()


def test_contract_codec_and_fused_translation_unit_share_one_rung_law():
    fp8 = next(
        row for row in load_runtime_contract()["formats"]
        if row["family"] == "FP8_CB_K"
    )
    assert tuple(fp8["rungs"]) == _READER_RUNGS
    assert tuple(fp8["producer_rungs"]) == _PRODUCER_RUNGS
    assert codec.FP8_FUSED_KBITS == _PRODUCER_RUNGS

    cuda = _source("gridbook/csrc/cb_fused_gemm.cu")
    macro = re.search(
        r"#define PQ_FUSED_RUNGS\(X\)(.*?)(?:\n\n)", cuda, re.DOTALL
    )
    assert macro is not None
    compiled = tuple(int(value) for value in re.findall(r"X\((\d+)\)",
                                                        macro.group(1)))
    assert compiled == _PRODUCER_RUNGS
    assert "constexpr int64_t kFusedKbLo = 4;" in cuda
    assert "constexpr int64_t kFusedKbHi = 48;" in cuda
    assert "constexpr int64_t kFusedKbStep = 4;" in cuda


def test_generic_fp8_word_loader_guards_every_shared_tail_read():
    cuda = _source("gridbook/csrc/cb_gemv.cu")
    reader_law = cuda[cuda.index("inline bool fp8_reader_kbits_supported("):
                      cuda.index("\n}\n", cuda.index(
                          "inline bool fp8_reader_kbits_supported(")) + 2]
    assert "k_bits >= 4 && k_bits <= 24 && k_bits % 4 == 0" in reader_law
    assert "k_bits >= 28 && k_bits <= 48" in reader_law
    assert cuda.count("fp8_reader_kbits_supported(k_bits)") == 3

    begin = cuda.index("DEVINL void fp8_load_codeword_words(")
    end = cuda.index("\n}\n", begin) + 2
    helper = cuda[begin:end]
    assert "w0 = s32[widx];" in helper
    assert "(rem + k_bits > 32) ? s32[widx + 1] : 0u" in helper
    assert "(rem + k_bits > 64) ? s32[widx + 2] : 0u" in helper
    # Definition plus dense double-buffer x2, dense single-buffer, grouped,
    # and expander call sites.  A new inline FP8 read must use this helper too.
    assert cuda.count("fp8_load_codeword_words(") == 6

    for k_bits in _READER_RUNGS:
        body_words = k_bits  # 4*k bytes / sizeof(uint32_t)
        for vector in range(32):
            bitpos = vector * k_bits
            byte0 = bitpos >> 3
            rem = ((byte0 & 3) << 3) + (bitpos & 7)
            word = byte0 >> 2
            assert word < body_words
            if rem + k_bits > 32:
                assert word + 1 < body_words
            if rem + k_bits > 64:
                assert word + 2 < body_words

    # The exact old over-read: K4/vector31 ends at bit 128, so its first word
    # is resident and widx+1 equals the one-past-body word.  The strict `> 32`
    # predicate suppresses that read.
    k_bits, vector = 4, 31
    bitpos = vector * k_bits
    byte0 = bitpos >> 3
    rem = ((byte0 & 3) << 3) + (bitpos & 7)
    word = byte0 >> 2
    assert (word, rem, word + 1, k_bits) == (3, 28, 4, 4)
    assert rem + k_bits == 32


def test_blackwell_fused_decoder_guards_tail_and_reserves_aligned_lut():
    header = _source(
        "gridbook/csrc/cutlass_fork/sm120_cb_fused_mma.hpp"
    )
    assert "KBits >= 4 && KBits <= 48 && KBits % 4 == 0" in header
    assert "CbLutLogicalBytes" in header
    assert "(CbLutLogicalBytes + 1023) / 1024" in header
    assert "(rem + CbKBits > 32) ? row32[widx + 1] : 0u" in header
    assert "(rem + CbKBits > 64) ? row32[widx + 2] : 0u" in header

    cuda = _source("gridbook/csrc/cb_fused_gemm.cu")
    asserted = {
        (int(tile), int(k_bits)): int(size)
        for tile, k_bits, size in re.findall(
            r"PQ_ASSERT_SMEM\((\d+),\s*(\d+),\s*(\d+)\)", cuda
        )
    }
    for k_bits in _PRODUCER_RUNGS:
        sub_bytes = (1 << (k_bits // 4)) * 2
        resident_128 = 4 if 4 * sub_bytes <= 16384 else 2
        logical_128 = resident_128 * sub_bytes
        lut_128 = ((logical_128 + 1023) // 1024) * 1024
        expected_128 = 256 * 128 + 512 * k_bits + 19456 + lut_128
        assert asserted[(128, k_bits)] == expected_128

        resident_256 = 4 if 4 * sub_bytes <= 1024 else 0
        logical_256 = resident_256 * sub_bytes
        lut_256 = (0 if logical_256 == 0
                   else ((logical_256 + 1023) // 1024) * 1024)
        expected_256 = 256 * 256 + 512 * k_bits + 19456 + lut_256
        assert asserted[(256, k_bits)] == expected_256


def test_dense_fp8_graph_boundary_remains_opaque_with_fake_shapes():
    ops = _source("gridbook/ops.py")
    for op in ("cb_gemv_fp8", "cb_expand_fp8", "cb_linear_forward"):
        assert (f'@torch.library.custom_op("prismaquant::{op}"' in ops)
        assert f"@{op}.register_fake" in ops
    linear = _source("gridbook/linear.py")
    assert "return cb_linear_forward(x, lid)" in linear
    # Route telemetry must remain scalar-only; a tensor read here would insert
    # a synchronization or a Dynamo-visible data dependency.
    telemetry = _source("gridbook/nvfp4_activation_contract.py")
    begin = telemetry.index("def emit_route(")
    end = telemetry.index("\n\ndef read_route", begin)
    body = telemetry[begin:end]
    assert ".item(" not in body
    assert ".cpu(" not in body
