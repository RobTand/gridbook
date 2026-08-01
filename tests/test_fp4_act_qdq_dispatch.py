"""CPU-oracle and FakeTensor contracts for native FP4 activation QDQ."""

import pytest
import torch

from gridbook import codec, ops
def test_cpu_bf16_never_reaches_the_cuda_only_op(monkeypatch):
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
