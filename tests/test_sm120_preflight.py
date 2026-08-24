"""CPU laws for the RTX 50 NVFP4 compile-only preflight."""
from __future__ import annotations

from pathlib import Path
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import cuda_ext, sm120_preflight


def _module(symbols, path):
    return types.SimpleNamespace(
        __file__=str(path),
        **{name: (lambda: None) for name in symbols},
    )


def test_sm120_preflight_uses_exact_generic_and_accelerated_targets(
    tmp_path, monkeypatch
):
    calls = []

    all_symbols = {
        "main": cuda_ext._EXT_SYMBOLS,
        "v2": cuda_ext._V2_SYMBOLS,
        "persistent_b": (*cuda_ext._MOE_PERSISTENT_B_SYMBOLS,
                         *cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS),
        "bf16_bridge": (*cuda_ext._BF16_GROUPED_SYMBOLS,
                        *cuda_ext._BF16_GROUPED_SM120_SYMBOLS),
    }

    def fake_load(**kwargs):
        key = Path(kwargs["build_directory"]).name
        calls.append((key, kwargs))
        return _module(all_symbols[key], tmp_path / f"{key}.so")

    import torch.utils.cpp_extension as cpp_extension
    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(
        cuda_ext, "_find_cutlass_include",
        lambda: str(tmp_path / "cutlass" / "include"),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability",
        lambda *_a, **_k: pytest.fail("compile preflight queried a device"),
    )
    monkeypatch.setattr(
        sm120_preflight, "_sass_targets",
        lambda path: ["sm_120a" if Path(path).stem in {
                          "persistent_b", "bf16_bridge"}
                      else "sm_120"],
    )

    receipt = sm120_preflight.compile_nvfp4_sm120_preflight(tmp_path / "build")
    by_key = dict(calls)
    assert by_key["main"]["extra_cuda_cflags"][-1] == (
        "-gencode=arch=compute_120,code=sm_120")
    assert by_key["v2"]["extra_cuda_cflags"][-1] == (
        "-gencode=arch=compute_120,code=sm_120")
    assert by_key["persistent_b"]["extra_cuda_cflags"][-1] == (
        "-gencode=arch=compute_120a,code=sm_120a")
    assert by_key["bf16_bridge"]["extra_cuda_cflags"][-1] == (
        "-gencode=arch=compute_120a,code=sm_120a")
    assert receipt["reader_rungs"] == list(range(1, 26))
    assert receipt["producer_rungs"] == list(range(1, 26))
    assert receipt["direct_kernel_research_rungs"] == list(range(1, 33))
    assert "artifact_compatibility_k26_k32" in receipt["claims_excluded"]
    assert receipt["qualification_ceiling"] == "compile_only"
    assert receipt["device_executed"] is False
    assert set(receipt["modules"]) == {
        "main", "v2", "persistent_b", "bf16_bridge"}


def test_sm120_source_guard_pins_full_range_and_global_expander():
    src = Path(cuda_ext.csrc_dir())
    sm120_preflight._validate_sources({
        "main": src / "cb_gemv.cu",
        "v2": src / "cb_gemv_v2.cu",
        "persistent_b": src / "cb_moe_persistent_b.cu",
        "bf16_bridge": src / "cb_bf16_grouped_gemm.cu",
    })


def test_sm120_preflight_refuses_wrong_sass_target(tmp_path, monkeypatch):
    def fake_load(**kwargs):
        key = Path(kwargs["build_directory"]).name
        symbols = {
            "main": cuda_ext._EXT_SYMBOLS,
            "v2": cuda_ext._V2_SYMBOLS,
            "persistent_b": (*cuda_ext._MOE_PERSISTENT_B_SYMBOLS,
                             *cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS),
            "bf16_bridge": (*cuda_ext._BF16_GROUPED_SYMBOLS,
                            *cuda_ext._BF16_GROUPED_SM120_SYMBOLS),
        }[key]
        return _module(symbols, tmp_path / f"{key}.so")

    import torch.utils.cpp_extension as cpp_extension
    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(
        cuda_ext, "_find_cutlass_include",
        lambda: str(tmp_path / "cutlass" / "include"),
    )
    monkeypatch.setattr(sm120_preflight, "_sass_targets", lambda _path: ["sm_121"])
    with pytest.raises(RuntimeError, match="SASS targets"):
        sm120_preflight.compile_nvfp4_sm120_preflight(tmp_path / "bad")


def test_sm120_cli_writes_exact_printed_receipt(tmp_path, monkeypatch, capsys):
    receipt = {
        "schema": "gridbook.sm120-nvfp4-compile-preflight.v1",
        "qualification_ceiling": "compile_only",
        "device_executed": False,
    }
    monkeypatch.setattr(
        sm120_preflight, "compile_nvfp4_sm120_preflight",
        lambda *_a, **_k: receipt)
    path = tmp_path / "receipt.json"
    assert sm120_preflight.main([
        "--build-directory", str(tmp_path / "build"),
        "--receipt", str(path),
    ]) == 0
    assert path.read_text(encoding="utf-8") == capsys.readouterr().out
