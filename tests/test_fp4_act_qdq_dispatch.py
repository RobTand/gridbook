"""CPU-safe dispatch and FakeTensor contracts for fp4 activation QDQ."""
import types

import pytest
import torch

from gridbook import codec, ops
from gridbook import cuda_ext


@pytest.fixture(autouse=True)
def _reset_capability(monkeypatch):
    monkeypatch.setattr(ops, "_FP4_ACT_QDQ_OK", None)


def test_missing_symbol_is_a_supported_eager_fallback(monkeypatch, capsys):
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: object())
    assert ops.fp4_act_qdq_ok() is False
    assert "symbol is missing" in capsys.readouterr().err


def test_capability_probe_accepts_the_required_symbol(monkeypatch, capsys):
    ext = types.SimpleNamespace(fp4_act_qdq=object())
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: ext)
    assert ops.fp4_act_qdq_ok() is True
    assert "act-qdq fp4=cuda" in capsys.readouterr().out


def test_cpu_bf16_never_reaches_the_cuda_only_op(monkeypatch):
    monkeypatch.setattr(
        ops, "fp4_act_qdq_ok",
        lambda: pytest.fail("CPU input should short-circuit before the probe"))
    monkeypatch.setattr(
        ops, "fp4_act_qdq",
        lambda x: pytest.fail("CPU input reached the CUDA-only op"))
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    expected = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    actual = ops.fp4_act_qdq_or_codec(x)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))


def test_fake_tensor_matches_native_contiguous_stride():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 48, 3, device="cuda",
                        dtype=torch.bfloat16).transpose(1, 2)
        assert not x.is_contiguous()
        out = ops.fp4_act_qdq(x)
        assert out.shape == (2, 3, 48)
        assert out.is_contiguous()


def test_fake_tensor_rejects_invalid_width():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 15, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="multiple of the fp4 group"):
            ops.fp4_act_qdq(x)
