"""Native checks for the research-only QTIP BF16/H128 CUDA primitive."""
from __future__ import annotations

import os
import shutil

import pytest

torch = pytest.importorskip("torch")

from gridbook import cuda_ext  # noqa: E402
from gridbook import qtip_hadamard as qtip  # noqa: E402


def _native_available() -> bool:
    if not torch.cuda.is_available() or shutil.which("nvcc") is None:
        return False
    # Keep JIT products on the persistent workspace volume, never /tmp.
    return os.environ.get("PRISMAQUANT_CB_EXT_DIR", "").startswith(
        "/home/rob/")


pytestmark = pytest.mark.skipif(
    not _native_available(), reason="needs CUDA, nvcc, and persistent JIT cache")


@pytest.fixture(scope="module")
def extension():
    qtip.prepare_qtip_hadamard_cuda()
    ext = cuda_ext.require_qtip_hadamard_warp128_ext("QTIP CUDA test")
    assert ext.qtip_hadamard_abi_schema() == 1
    assert ext.__gridbook_jit_abi_schema__ == 1
    assert len(ext.__gridbook_jit_identity__) == 64
    assert ext.__gridbook_jit_capability__ == \
        tuple(torch.cuda.get_device_capability())
    return ext


def _input_reference(value: torch.Tensor,
                     signs: torch.Tensor) -> torch.Tensor:
    return qtip._normalized_block_hadamard_rows(
        value * signs.to(dtype=value.dtype), 128).to(torch.bfloat16)


def _output_reference(value: torch.Tensor,
                      signs: torch.Tensor) -> torch.Tensor:
    transformed = qtip._normalized_block_hadamard_rows(value, 128)
    return (transformed * signs.float()).to(torch.bfloat16)


@pytest.mark.parametrize("rows,dimension", [(1, 128), (7, 256), (33, 1024)])
def test_native_input_and_output_match_torch_semantics_exactly(
        extension, rows, dimension):
    generator = torch.Generator(device="cuda").manual_seed(
        20260830 + rows + dimension)
    value = torch.randn(
        rows, dimension, dtype=torch.bfloat16, device="cuda",
        generator=generator)
    input_signs = qtip.seeded_signs(
        "input", dimension, 11, device="cuda", dtype=torch.bfloat16)
    output_signs = qtip.seeded_signs(
        "output", dimension, 29, device="cuda", dtype=torch.bfloat16)

    got_input = qtip.apply_input_transform(value, input_signs, 128)
    got_output = qtip.apply_inverse_output_transform(
        value, output_signs, 128)
    assert torch.equal(got_input, _input_reference(value, input_signs))
    assert torch.equal(got_output, _output_reference(value, output_signs))
    assert got_input.dtype == got_output.dtype == torch.bfloat16


def test_public_dispatch_uses_native_only_for_cuda_bf16_128(
        extension, monkeypatch):
    value = torch.ones(2, 128, dtype=torch.bfloat16, device="cuda")
    signs = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    sentinel = torch.full_like(value, 17)
    calls = []

    def spy(arg_value, arg_signs, sign_before):
        calls.append((arg_value, arg_signs, sign_before))
        return sentinel

    monkeypatch.setattr(qtip, "_qtip_hadamard_warp128", spy)
    assert qtip.apply_input_transform(value, signs, 128) is sentinel
    assert calls[-1][2] is True
    assert qtip.apply_inverse_output_transform(value, signs, 128) is sentinel
    assert calls[-1][2] is False

    calls.clear()
    fp32 = value.float()
    got_fp32 = qtip.apply_input_transform(fp32, signs.float(), 128)
    assert calls == []
    assert torch.equal(got_fp32, _input_reference(fp32, signs.float()))
    got_block64 = qtip.apply_input_transform(value, signs, 64)
    assert calls == []
    expected64 = qtip._normalized_block_hadamard_rows(
        value * signs, 64).to(torch.bfloat16)
    assert torch.equal(got_block64, expected64)


def test_matching_native_cell_fails_closed_without_extension(
        extension, monkeypatch):
    value = torch.ones(1, 128, dtype=torch.bfloat16, device="cuda")
    signs = torch.ones(128, dtype=torch.bfloat16, device="cuda")

    def unavailable(_operation):
        raise cuda_ext.NativeKernelUnavailableError("deliberately unavailable")

    monkeypatch.setattr(
        cuda_ext, "require_qtip_hadamard_warp128_ext", unavailable)
    with pytest.raises(
            cuda_ext.NativeKernelUnavailableError,
            match="deliberately unavailable"):
        qtip.apply_input_transform(value, signs, 128)


@pytest.mark.parametrize("case,match", [
    ("input_dtype", "input must be bfloat16"),
    ("sign_dtype", "signs must be bfloat16"),
    ("input_rank", "input must be a contiguous 2-D"),
    ("sign_rank", "signs must be a contiguous 1-D"),
    ("dimension", "positive multiple of 128"),
    ("sign_length", "signs length must equal"),
    ("input_contiguous", "input must be contiguous"),
])
def test_raw_native_abi_rejects_malformed_inputs(extension, case, match):
    value = torch.ones(2, 128, dtype=torch.bfloat16, device="cuda")
    signs = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    if case == "input_dtype":
        value = value.float()
    elif case == "sign_dtype":
        signs = signs.float()
    elif case == "input_rank":
        value = value.reshape(2, 2, 64)
    elif case == "sign_rank":
        signs = signs.reshape(2, 64)
    elif case == "dimension":
        value = torch.ones(2, 192, dtype=torch.bfloat16, device="cuda")
        signs = torch.ones(192, dtype=torch.bfloat16, device="cuda")
    elif case == "sign_length":
        signs = signs[:-1]
    elif case == "input_contiguous":
        value = torch.ones(
            2, 256, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(RuntimeError, match=match):
        extension.qtip_hadamard_warp128(value, signs, True)


@pytest.mark.parametrize("sign_before", [True, False])
def test_valid_unaligned_contiguous_views_take_safe_scalar_loads(
        extension, sign_before):
    value_storage = torch.randn(
        257, dtype=torch.bfloat16, device="cuda")
    sign_storage = torch.ones(
        129, dtype=torch.bfloat16, device="cuda")
    value = value_storage[1:].view(2, 128)
    signs = sign_storage[1:]
    assert value.is_contiguous() and signs.is_contiguous()
    assert value.data_ptr() % 8 != 0 and signs.data_ptr() % 8 != 0
    got = extension.qtip_hadamard_warp128(value, signs, sign_before)
    expected = (_input_reference(value, signs) if sign_before
                else _output_reference(value, signs))
    assert torch.equal(got, expected)


@pytest.mark.parametrize("sign_before", [True, False])
def test_zero_row_raw_abi_returns_empty_bf16(extension, sign_before):
    value = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
    signs = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    got = extension.qtip_hadamard_warp128(value, signs, sign_before)
    assert got.shape == (0, 256)
    assert got.dtype == torch.bfloat16 and got.device.type == "cuda"


@pytest.mark.parametrize("sign_before", [True, False])
def test_native_op_uses_current_stream_and_replays_mutated_graph_operand(
        extension, sign_before):
    generator = torch.Generator(device="cuda").manual_seed(551)
    static_value = torch.randn(
        5, 256, dtype=torch.bfloat16, device="cuda", generator=generator)
    role = "input" if sign_before else "output"
    signs = qtip.seeded_signs(
        role, 256, 91, device="cuda", dtype=torch.bfloat16)
    replacement = torch.randn(
        5, 256, dtype=torch.bfloat16, device="cuda", generator=generator)
    transform = (qtip.apply_input_transform if sign_before
                 else qtip.apply_inverse_output_transform)
    reference = _input_reference if sign_before else _output_reference
    worker = torch.cuda.Stream()
    worker.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(worker):
        warm = transform(static_value, signs, 128)
    worker.synchronize()
    assert torch.equal(warm, reference(static_value, signs))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=worker):
        captured = transform(static_value, signs, 128)
    static_value.copy_(replacement)
    graph.replay()
    worker.synchronize()
    assert torch.equal(captured, reference(replacement, signs))


def test_custom_op_fake_contract_supports_fullgraph_compile(extension):
    value = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
    signs = qtip.seeded_signs(
        "output", 256, 7, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(
        lambda a, d: qtip.apply_inverse_output_transform(a, d, 128),
        backend="eager", fullgraph=True)
    got = compiled(value, signs)
    assert torch.equal(got, _output_reference(value, signs))
