"""CPU-only tests for the callable Ada compile preflight."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gridbook import cuda_ext
from gridbook import sm89_preflight


def _module_with_production_symbols():
    return types.SimpleNamespace(
        **{name: (lambda: None) for name in cuda_ext._EXT_SYMBOLS}
    )


def test_sm89_preflight_uses_explicit_arch_and_never_queries_a_device(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return _module_with_production_symbols()

    import torch.utils.cpp_extension as cpp_extension
    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability",
        lambda *_a, **_k: pytest.fail("compile preflight queried a live GPU"),
    )
    abi = []
    monkeypatch.setattr(
        sm89_preflight, "require_native_fp8_cutlass", abi.append
    )

    receipt = sm89_preflight.compile_dense_fp8_sm89_preflight(
        tmp_path / "sm89-build"
    )
    assert seen["extra_cuda_cflags"] == [
        "-O3", "-gencode=arch=compute_89,code=sm_89"
    ]
    assert seen["build_directory"] == str((tmp_path / "sm89-build").resolve())
    assert seen["sources"] == [
        str(Path(cuda_ext.csrc_dir()) / "cb_gemv.cu")
    ]
    assert abi == ["dense FP8-CB SM89 compile-only preflight"]
    assert receipt["capability"] == [8, 9]
    assert receipt["producer_rungs"] == list(range(4, 49, 4))
    assert receipt["qualification_ceiling"] == "compile_only"
    assert receipt["device_executed"] is False
    assert receipt["vllm_native_abi"]["status"] == "present_not_executed"
    assert {"device_correctness", "device_performance", "torch_compile",
            "vllm_cudagraph"} == set(receipt["claims_excluded"])


def test_sm89_preflight_refuses_an_incomplete_production_module(
    tmp_path, monkeypatch
):
    module = _module_with_production_symbols()
    delattr(module, "cb_expand_fp8")
    import torch.utils.cpp_extension as cpp_extension
    monkeypatch.setattr(cpp_extension, "load", lambda **_kwargs: module)
    monkeypatch.setattr(
        sm89_preflight, "require_native_fp8_cutlass",
        lambda _context: pytest.fail("external ABI checked after symbol failure"),
    )

    with pytest.raises(cuda_ext.StaleExtensionError, match="cb_expand_fp8"):
        sm89_preflight.compile_dense_fp8_sm89_preflight(tmp_path / "bad")


def test_sm89_preflight_source_guard_covers_low_rungs_and_dense_routes():
    source = Path(cuda_ext.csrc_dir()) / "cb_gemv.cu"
    sm89_preflight._validate_dense_fp8_source(source)


def test_sm89_preflight_cli_persists_the_printed_receipt(
    tmp_path, monkeypatch, capsys
):
    receipt = {
        "schema": "gridbook.sm89-compile-preflight.v1",
        "qualification_ceiling": "compile_only",
        "device_executed": False,
    }
    monkeypatch.setattr(
        sm89_preflight,
        "compile_dense_fp8_sm89_preflight",
        lambda *_args, **_kwargs: receipt,
    )
    path = tmp_path / "receipts" / "sm89.json"
    assert sm89_preflight.main([
        "--build-directory", str(tmp_path / "build"),
        "--receipt", str(path),
    ]) == 0
    assert path.read_text(encoding="utf-8") == capsys.readouterr().out
    assert path.read_text(encoding="utf-8").endswith("\n")
