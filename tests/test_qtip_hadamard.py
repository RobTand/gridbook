"""CPU/reference gates for the research QTIP online-transform ABI."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gridbook import qtip_hadamard as qtip_module  # noqa: E402
from gridbook.qtip_hadamard import (  # noqa: E402
    SIGN_GENERATOR,
    TRANSFORM_ALGORITHM,
    TRANSFORM_NORMALIZATION,
    TRANSFORM_PADDING,
    TRANSFORM_SCHEMA,
    apply_input_transform,
    apply_inverse_output_transform,
    online_transform_digest,
    seeded_sign_digest,
    seeded_signs,
    validate_online_transform,
)


def _side(role: str, dimension: int, block_size: int, seed: int) -> dict:
    return {
        "dimension": dimension,
        "block_size": block_size,
        "seed": seed,
        "sign_generator": SIGN_GENERATOR,
        "sign_sha256": seeded_sign_digest(role, dimension, seed),
    }


def _contract(rows=8, columns=16) -> dict:
    contract = {
        "schema": TRANSFORM_SCHEMA,
        "algorithm": TRANSFORM_ALGORITHM,
        "normalization": TRANSFORM_NORMALIZATION,
        "padding": TRANSFORM_PADDING,
        "input": _side("input", columns, 4, 0x1234),
        "output": _side("output", rows, 4, 0x5678),
    }
    contract["transform_sha256"] = online_transform_digest(contract)
    return contract


def _block_hadamard(dimension: int, block_size: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < block_size:
        h = torch.cat((torch.cat((h, h), dim=1),
                       torch.cat((h, -h), dim=1)), dim=0)
    h = h * (block_size ** -0.5)
    out = torch.zeros(dimension, dimension, dtype=torch.float32)
    for start in range(0, dimension, block_size):
        out[start:start + block_size, start:start + block_size] = h
    return out


def test_seeded_sign_construction_has_pinned_cross_runtime_digest():
    # This is an ABI vector, not merely a self-consistency check. A change in
    # RNG, bit order, role domain, or tail masking must move this value.
    assert seeded_sign_digest("input", 19, 0x0123456789ABCDEF) == \
        "c9850b2a7c2d365cdb23964ff33c6e934fd3dde47bedb07b35eb9ba8823f6368"
    signs = seeded_signs("input", 19, 0x0123456789ABCDEF)
    assert signs.shape == (19,)
    assert set(signs.tolist()) == {-1.0, 1.0}


def test_validator_binds_every_dimension_seed_and_sign_digest():
    contract = _contract()
    got = validate_online_transform(
        contract, rows=8, columns=16, target="layer")
    assert got == contract

    bad_dimension = copy.deepcopy(contract)
    bad_dimension["input"]["dimension"] = 8
    with pytest.raises(ValueError, match="does not match"):
        validate_online_transform(
            bad_dimension, rows=8, columns=16, target="layer")

    bad_digest = copy.deepcopy(contract)
    bad_digest["output"]["seed"] += 1
    with pytest.raises(ValueError, match="does not bind"):
        validate_online_transform(
            bad_digest, rows=8, columns=16, target="layer")


@pytest.mark.parametrize("mutation,match", [
    (("normalization", "none"), "normalization"),
    (("padding", "zero"), "padding"),
    (("algorithm", "fft"), "algorithm"),
])
def test_validator_refuses_semantic_drift(mutation, match):
    contract = _contract()
    contract[mutation[0]] = mutation[1]
    with pytest.raises(ValueError, match=match):
        validate_online_transform(
            contract, rows=8, columns=16, target="layer")


def test_validator_refuses_unknown_fields_and_nondividing_blocks():
    contract = _contract()
    contract["future_order"] = "guess-me"
    with pytest.raises(ValueError, match="unknown"):
        validate_online_transform(
            contract, rows=8, columns=16, target="layer")

    contract = _contract()
    contract["input"]["block_size"] = 8
    contract["input"]["dimension"] = 12
    contract["input"]["sign_sha256"] = seeded_sign_digest(
        "input", 12, contract["input"]["seed"])
    with pytest.raises(ValueError, match="must divide"):
        validate_online_transform(
            contract, rows=8, columns=12, target="layer")


def test_root_digest_binds_block_geometry_and_fixed_semantics():
    contract = _contract()
    contract["input"]["block_size"] = 8
    with pytest.raises(ValueError, match="transform_sha256.*does not bind"):
        validate_online_transform(
            contract, rows=8, columns=16, target="layer")


def test_row_kernels_match_the_explicit_h_d_algebra_exactly():
    x = torch.arange(-32, 32, dtype=torch.float32).reshape(4, 16) / 8
    z = torch.arange(-16, 16, dtype=torch.float32).reshape(4, 8) / 4
    d_in = seeded_signs("input", 16, 0x1234)
    d_out = seeded_signs("output", 8, 0x5678)
    h_in = _block_hadamard(16, 4)
    h_out = _block_hadamard(8, 4)

    got_in = apply_input_transform(x, d_in, 4)
    want_in = ((x * d_in) @ h_in).to(torch.bfloat16)
    assert torch.equal(got_in, want_in)  # row x D H = (H D x_col).T

    got_out = apply_inverse_output_transform(z, d_out, 4)
    want_out = ((z @ h_out) * d_out).to(torch.bfloat16)
    assert torch.equal(got_out, want_out)  # row y H D = (D H y_col).T


def test_cpu_bf16_block128_stays_on_transparent_reference(monkeypatch):
    def native_must_not_run(*_args, **_kwargs):
        raise AssertionError("CPU reference dispatched to the CUDA op")

    monkeypatch.setattr(
        qtip_module, "_qtip_hadamard_warp128", native_must_not_run)
    x = torch.arange(256, dtype=torch.float32).reshape(2, 128) \
             .sub(127).div(31).to(torch.bfloat16)
    signs = seeded_signs("input", 128, 3, dtype=torch.bfloat16)
    got = apply_input_transform(x, signs, 128)
    want = qtip_module._normalized_block_hadamard_rows(
        x * signs, 128).to(torch.bfloat16)
    assert torch.equal(got, want)


def test_native_dispatch_cell_is_exactly_cuda_bf16_block128():
    cuda_bf16 = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    cuda_fp32 = SimpleNamespace(is_cuda=True, dtype=torch.float32)
    cpu_bf16 = SimpleNamespace(is_cuda=False, dtype=torch.bfloat16)
    assert qtip_module._use_native_warp128(cuda_bf16, 128)
    assert not qtip_module._use_native_warp128(cuda_bf16, 256)
    assert not qtip_module._use_native_warp128(cuda_fp32, 128)
    assert not qtip_module._use_native_warp128(cpu_bf16, 128)


@pytest.mark.parametrize("input_block,output_block,expected_calls", [
    (128, 128, 1),
    (128, 256, 1),
    (256, 128, 1),
    (256, 256, 0),
    (4096, 256, 0),
])
def test_model_load_prepares_iff_artifact_explicitly_declares_h128(
        monkeypatch, input_block, output_block, expected_calls):
    calls = []
    monkeypatch.setattr(
        qtip_module, "prepare_qtip_hadamard_cuda", lambda: calls.append(1))
    declared = (input_block, output_block)
    result = qtip_module.prepare_qtip_hadamard_model_load(
        input_block, output_block)
    assert len(calls) == expected_calls
    assert result is None and declared == (input_block, output_block)


@pytest.mark.parametrize("bad_block", [0, -2, 3, True, 1.5])
def test_application_refuses_malformed_block_sizes(bad_block):
    x = torch.ones(2, 128)
    signs = torch.ones(128)
    with pytest.raises(ValueError, match="positive power of two"):
        apply_input_transform(x, signs, bad_block)


def test_application_refuses_malformed_shapes_and_nondividing_blocks():
    with pytest.raises(ValueError, match="must be 2-D"):
        apply_input_transform(torch.ones(128), torch.ones(128), 128)
    with pytest.raises(ValueError, match="does not bind"):
        apply_input_transform(torch.ones(2, 128), torch.ones(127), 128)
    with pytest.raises(ValueError, match="does not divide"):
        apply_inverse_output_transform(
            torch.ones(2, 192), torch.ones(192), 128)


def test_end_to_end_transformed_linear_recovers_original_linear():
    # Use block_size=4: 1/sqrt(4)=1/2 is exactly representable, so this checks
    # the algebra without a tolerance or an irrational normalization factor.
    generator = torch.Generator().manual_seed(7)
    w = torch.randn(8, 16, generator=generator)
    x = torch.randn(3, 16, generator=generator)
    d_in = seeded_signs("input", 16, 0x1234)
    d_out = seeded_signs("output", 8, 0x5678)
    h_in = _block_hadamard(16, 4)
    h_out = _block_hadamard(8, 4)
    r_in = h_in @ torch.diag(d_in)
    r_out = h_out @ torch.diag(d_out)
    q = r_out @ w @ r_in.T

    x_rot = (x * d_in) @ h_in
    z = x_rot @ q.T
    got = (z @ h_out) * d_out
    want = x @ w.T
    assert torch.allclose(got, want, rtol=2e-6, atol=2e-6)
