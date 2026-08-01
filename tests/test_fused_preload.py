"""Residency-matched preload contract for fused FP8 and NVFP4 modules."""

from gridbook import cuda_ext


def test_preload_attempts_fp8_and_fp4(monkeypatch):
    calls = []

    def fp8():
        calls.append("fp8")
        raise RuntimeError("one failed loader must not suppress the other")

    def fp4():
        calls.append("fp4")

    monkeypatch.setattr(cuda_ext, "get_fused_ext", fp8)
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", fp4)
    status = cuda_ext.preload_fused_extensions()

    assert calls == ["fp8", "fp4"]
    assert status == {"fp8": False, "fp4": False}

    calls.clear()
    try:
        cuda_ext.preload_fused_extensions(strict=True)
    except RuntimeError as exc:
        assert "fp8=" in str(exc)
        assert "fp4=unavailable" in str(exc)
    else:  # pragma: no cover - strict mode must fail closed
        raise AssertionError("strict preload accepted unavailable extensions")
    assert calls == ["fp8", "fp4"]
