"""Native-vs-reference checks for the isolated research TCQ R256 module."""
from __future__ import annotations

import os
import random
import shutil
import struct

import pytest

torch = pytest.importorskip("torch")

from gridbook import trellis


def _alphabet(family, rate):
    full = trellis.canonical_full_alphabet(family)
    count = 1 << (rate + 1)
    if count == len(full):
        return full
    return tuple(full[i * (len(full) - 1) // (count - 1)]
                 for i in range(count))


def _wire(family, q256, columns, *, rows=4):
    template = trellis.build_q256_schedule(family, q256, 256)
    expanded = tuple(template[i % 256] for i in range(columns))
    terminal = trellis.native_bits(family)
    alphabets = {rate: _alphabet(family, rate)
                 for rate in sorted({r for r in expanded if r < terminal})}
    rng = random.Random(q256 * 1009 + columns)
    u = [[rng.getrandbits(1) for _ in range(columns)] for _ in range(rows)]
    points = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    finite = [c for c in range(1 << terminal)
              if family != trellis.TCQ_E4M3_R256 or c not in (0x7f, 0xff)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                bypass[row][column] = rng.choice(finite)
            else:
                points[row][column] = rng.randrange(1 << (rate - 1))
    if family == trellis.TCQ_E2M1_R256:
        scale_blob = bytes([0x38] * (rows * ((columns + 15) // 16)))
        global_real = 0.5
    else:
        scale_blob = struct.pack(
            f"<{rows}f", *[0.5 * (2 ** (row % 4)) for row in range(rows)])
        global_real = 1.0
    return trellis.pack_planes(
        family=family, body_rate_q256=q256, schedule=expanded,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
        u_bits=u, point_indices=points, bypass_codes=bypass,
        alphabets=alphabets, scale_blob=scale_blob,
        global_scale_real=global_real)


def _native_available():
    if not torch.cuda.is_available() or shutil.which("nvcc") is None:
        return False
    # The suite and extension cache must stay under /home/rob.  Refuse rather
    # than silently letting torch's JIT choose /tmp.
    root = os.environ.get("PRISMAQUANT_CB_EXT_DIR", "")
    return root.startswith("/home/rob/")


def _raw_inputs(family, q256=512, columns=256, *, rows=4):
    from gridbook import cuda_ext
    from gridbook.trellis_ops import wire_cuda_tensors

    wire = _wire(family, q256, columns, rows=rows)
    tensors = wire_cuda_tensors(wire)
    return cuda_ext.require_trellis_r256_ext("hostile raw-ABI test"), wire, tensors


def _raw_block_plan(wire, values):
    """Compact plan for the v2 ABI, honouring a hostile ``schedule`` swap."""
    from gridbook.trellis_ops import wire_cuda_block_plan
    return wire_cuda_block_plan(wire)


def _raw_decode(extension, wire, tensors, **replacements):
    names = ("payload", "schedule", "column_offsets", "previous_u_offsets",
             "alphabet_lut", "scales", "family")
    values = dict(zip(names, tensors))
    values.update(replacements)
    return extension.trellis_r256_decode_codes(
        values["payload"], values["schedule"], values["column_offsets"],
        values["previous_u_offsets"], values["alphabet_lut"], wire.rows,
        wire.columns, wire.row_stride_bytes, values["family"])


def _raw_expand(extension, wire, tensors, **replacements):
    names = ("payload", "schedule", "column_offsets", "previous_u_offsets",
             "alphabet_lut", "scales", "family")
    values = dict(zip(names, tensors))
    values.update(replacements)
    rate, offset, ordinal, meta = _raw_block_plan(wire, values)
    return extension.trellis_r256_expand(
        values["payload"], values["schedule"], values["column_offsets"],
        values["previous_u_offsets"], rate, offset, ordinal, meta,
        values["alphabet_lut"], values["scales"],
        wire.rows, wire.columns, wire.row_stride_bytes, values["family"])


def _raw_gemv(extension, wire, tensors, **replacements):
    names = ("payload", "schedule", "column_offsets", "previous_u_offsets",
             "alphabet_lut", "scales", "family")
    values = dict(zip(names, tensors))
    values.update(replacements)
    x = torch.ones((1, wire.columns), dtype=torch.float32, device="cuda")
    return extension.trellis_r256_dequant_gemv(
        x, values["payload"], values["schedule"], values["column_offsets"],
        values["previous_u_offsets"], values["alphabet_lut"], values["scales"],
        wire.rows, wire.columns, wire.row_stride_bytes, values["family"])


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="CUDA+nvcc and a /home/rob PRISMAQUANT_CB_EXT_DIR are required")


@pytest.mark.parametrize("family,q256,columns", [
    (trellis.TCQ_E2M1_R256, 385, 265),       # odd block boundary + tail
    (trellis.TCQ_E2M1_R256, 896, 256),       # mixed TCQ/bypass
    (trellis.TCQ_E2M1_R256, 1016, 256),      # T'=8
    (trellis.TCQ_E4M3_R256, 1920, 256),      # mixed rate-7/bypass
    (trellis.TCQ_E4M3_R256, 2040, 256),      # T'=8
])
def test_cuda_codes_expand_and_gemv_match_reference(
        family, q256, columns):
    from gridbook.trellis_ops import (
        decode_wire_codes_cuda, expand_wire_cuda, gemv_wire_cuda,
        prepare_wire_cuda)

    wire = _wire(family, q256, columns)
    expected_codes = torch.tensor(
        trellis.decode_codes(wire), dtype=torch.uint8, device="cuda")
    expected_values = torch.tensor(
        trellis.decode_values(wire), dtype=torch.float32, device="cuda")

    actual_codes = decode_wire_codes_cuda(wire)
    actual_values = expand_wire_cuda(wire)
    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(actual_values, expected_values)

    prepared = prepare_wire_cuda(wire)
    actual_native_packed = prepared.decode_native_packed()
    if family == trellis.TCQ_E2M1_R256:
        expected_native_packed = torch.zeros(
            (wire.rows, (wire.columns + 1) // 2), dtype=torch.uint8,
            device="cuda")
        expected_native_packed[:, :wire.columns // 2] = (
            expected_codes[:, 0:wire.columns - 1:2]
            | (expected_codes[:, 1:wire.columns:2] << 4)
        )
        if wire.columns & 1:
            expected_native_packed[:, -1] = expected_codes[:, -1]
            assert torch.count_nonzero(actual_native_packed[:, -1] & 0xf0) == 0
    else:
        expected_native_packed = expected_codes
    assert torch.equal(actual_native_packed, expected_native_packed)

    x = torch.linspace(-1.0, 1.0, 2 * columns, device="cuda",
                       dtype=torch.float32).reshape(2, columns)
    actual_gemv = gemv_wire_cuda(x, wire)
    expected_gemv = x @ expected_values.t()
    torch.testing.assert_close(actual_gemv, expected_gemv,
                               rtol=2e-5, atol=2e-5)


def test_cuda_short_tail_block_tprime_eight_matches_reference():
    from gridbook.trellis_ops import decode_wire_codes_cuda

    family = trellis.TCQ_E2M1_R256
    schedule = list(trellis.build_q256_schedule(family, 385, 256))
    schedule.extend([3] * 8)
    columns = len(schedule)
    alphabets = {rate: _alphabet(family, rate)
                 for rate in sorted(set(schedule))}
    wire = trellis.pack_planes(
        family=family,
        body_rate_q256=(sum(schedule) * 256 + columns // 2) // columns,
        schedule=schedule, layout=trellis.LAYOUT_TIGHT_OFFSETS,
        u_bits=[[column & 1 for column in range(columns)] for _ in range(2)],
        point_indices=[[0] * columns for _ in range(2)],
        bypass_codes=[[0] * columns for _ in range(2)], alphabets=alphabets,
        scale_blob=bytes([0x38] * (2 * ((columns + 15) // 16))))
    expected = torch.tensor(trellis.decode_codes(wire), dtype=torch.uint8,
                            device="cuda")
    assert torch.equal(decode_wire_codes_cuda(wire), expected)


def test_research_loader_identity_and_isolation():
    from gridbook import cuda_ext

    ext = cuda_ext.require_trellis_r256_ext("test")
    assert ext.trellis_r256_wire_schema() == trellis.SCHEMA
    assert ext.trellis_r256_abi_schema() == 3
    assert ext.__gridbook_jit_abi_schema__ == 3
    assert len(ext.__gridbook_jit_identity__) == 64
    assert all("trellis" not in family for family, _loader
               in cuda_ext._PRELOAD_FAMILIES)


def test_prevalidated_packed_decode_rejects_owner_mutation_and_bad_output():
    from gridbook.trellis_ops import prepare_wire_cuda

    wire = _wire(trellis.TCQ_E2M1_R256, 512, 265, rows=2)
    prepared = prepare_wire_cuda(wire)
    wrong = torch.empty(
        (wire.rows, prepared.output_columns + 1), dtype=torch.uint8,
        device="cuda")
    with pytest.raises(ValueError, match="native packed output"):
        prepared.decode_native_packed_out(wrong)

    with pytest.raises(AttributeError):
        prepared.row_body_bits = 1

    # Deliberately pierce the private owner for the hostile mutation test.  A
    # normal caller has no tensor getter and therefore no alias to mutate.
    private = prepared._PreparedTrellisWireCuda__tensors
    private[0][0, 0] ^= 1
    with pytest.raises(RuntimeError, match="contract changed after validation"):
        prepared.decode_native_packed()


def test_prevalidated_packed_decode_rejects_hostile_scalar_mutation():
    from gridbook.trellis_ops import prepare_wire_cuda

    wire = _wire(trellis.TCQ_E2M1_R256, 512, 265, rows=2)
    prepared = prepare_wire_cuda(wire)
    prepared._PreparedTrellisWireCuda__row_body_bits = 1
    with pytest.raises(RuntimeError, match="contract changed after validation"):
        prepared.decode_native_packed()


def test_prevalidated_packed_decode_is_cuda_graph_capturable():
    from gridbook.trellis_ops import prepare_wire_cuda

    wire = _wire(trellis.TCQ_E4M3_R256, 1152, 256, rows=2)
    prepared = prepare_wire_cuda(wire)
    output = prepared.empty_native_packed()
    prepared.decode_native_packed_out(output)
    torch.cuda.synchronize()
    expected = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.decode_native_packed_out(output)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected)


@pytest.mark.parametrize("invalid_rate", [0, 5])
def test_raw_cuda_abi_rejects_schedule_before_rate_controlled_shifts(
        invalid_rate):
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    schedule = tensors[1].clone()
    schedule[0] = invalid_rate
    with pytest.raises(RuntimeError, match="schedule rate is outside"):
        _raw_decode(extension, wire, tensors, schedule=schedule)


def test_raw_cuda_abi_rejects_noncanonical_and_oob_column_offsets():
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    offsets = tensors[2].clone()
    offsets[0] = wire.row_stride_bytes * 8
    with pytest.raises(RuntimeError, match="column offset.*row body"):
        _raw_decode(extension, wire, tensors, column_offsets=offsets)


def test_raw_cuda_abi_rejects_noncanonical_and_oob_previous_offsets():
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    previous = tensors[3].clone()
    coded_column = next(
        i for i, rate in enumerate(wire.expanded_schedule) if rate < 4)
    previous[coded_column, 0] = wire.row_stride_bytes * 8
    with pytest.raises(RuntimeError, match="previous-u offset plan"):
        _raw_decode(extension, wire, tensors, previous_u_offsets=previous)


@pytest.mark.parametrize("name,index,reshape,match", [
    ("payload", 0, lambda tensor: tensor.reshape(-1),
     r"payload must have shape \[rows,row_stride\]"),
    ("schedule", 1, lambda tensor: tensor.reshape(1, -1),
     r"schedule must have shape \[columns\]"),
    ("column_offsets", 2, lambda tensor: tensor.reshape(1, -1),
     r"column_offsets must have shape \[columns\]"),
    ("previous_u_offsets", 3, lambda tensor: tensor.reshape(-1),
     r"previous_u_offsets must have shape \[columns,8\]"),
    ("alphabet_lut", 4, lambda tensor: tensor.reshape(-1),
     r"alphabet_lut must have exact shape \[native_bits,256\]"),
])
def test_raw_cuda_abi_rejects_same_numel_wrong_rank(
        name, index, reshape, match):
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    malformed = reshape(tensors[index])
    with pytest.raises(RuntimeError, match=match):
        _raw_decode(extension, wire, tensors, **{name: malformed})


def test_raw_cuda_abi_rejects_oversized_alphabet_lut():
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    lut = torch.cat((tensors[4], tensors[4][:1]), dim=0)
    with pytest.raises(RuntimeError, match="alphabet_lut must have exact shape"):
        _raw_decode(extension, wire, tensors, alphabet_lut=lut)


@pytest.mark.parametrize("family,q256", [
    (trellis.TCQ_E2M1_R256, 512),
    (trellis.TCQ_E4M3_R256, 1024),
])
@pytest.mark.parametrize("operation", [_raw_expand, _raw_gemv])
def test_raw_cuda_value_ops_reject_same_numel_wrong_scale_shape(
        family, q256, operation):
    extension, wire, tensors = _raw_inputs(family, q256)
    scales = tensors[5].reshape(-1)
    with pytest.raises(RuntimeError, match="decoded scales must have exact shape"):
        operation(extension, wire, tensors, scales=scales)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", [_raw_expand, _raw_gemv])
def test_raw_cuda_value_ops_reject_nonpositive_or_nonfinite_scales(
        bad_scale, operation):
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    scales = tensors[5].clone()
    scales[0, 0] = bad_scale
    with pytest.raises(RuntimeError, match="scales must be finite and positive"):
        operation(extension, wire, tensors, scales=scales)


@pytest.mark.parametrize("padding", ["high_bit", "row_byte"])
def test_raw_cuda_abi_rejects_noncanonical_payload_padding(padding):
    extension, wire, tensors = _raw_inputs(
        trellis.TCQ_E2M1_R256, q256=385, columns=256)
    assert wire.row_body_bits & 7
    payload = tensors[0].clone()
    used_bytes = (wire.row_body_bits + 7) // 8
    if padding == "high_bit":
        payload[0, used_bytes - 1] |= 0x80
    else:
        payload[0, used_bytes] = 1
    with pytest.raises(RuntimeError, match="padding must be canonical zero"):
        _raw_decode(extension, wire, tensors, payload=payload)


@pytest.mark.parametrize("family,q256,bad_code", [
    (trellis.TCQ_E2M1_R256, 512, 0x10),
    (trellis.TCQ_E4M3_R256, 1024, 0x7f),
    (trellis.TCQ_E4M3_R256, 1024, 0xff),
])
def test_raw_cuda_abi_rejects_invalid_alphabet_native_codes(
        family, q256, bad_code):
    extension, wire, tensors = _raw_inputs(family, q256)
    alphabet_lut = tensors[4].clone()
    alphabet_lut.fill_(bad_code)
    with pytest.raises(RuntimeError, match="alphabet LUT.*invalid native code"):
        _raw_decode(extension, wire, tensors, alphabet_lut=alphabet_lut)


@pytest.mark.parametrize("nan_code", [0x7f, 0xff])
def test_raw_cuda_abi_rejects_e4m3_nan_bypass_payload(nan_code):
    extension, wire, tensors = _raw_inputs(
        trellis.TCQ_E4M3_R256, 2040)
    terminal_column = next(
        i for i, rate in enumerate(wire.expanded_schedule) if rate == 8)
    column_offsets, _previous = trellis.derived_decode_plan(wire)
    bit_offset = column_offsets[terminal_column]

    payload_bytes = bytearray(tensors[0].cpu().reshape(-1).tolist())
    for bit in range(8):
        absolute = bit_offset + bit
        byte = absolute >> 3
        mask = 1 << (absolute & 7)
        if (nan_code >> bit) & 1:
            payload_bytes[byte] |= mask
        else:
            payload_bytes[byte] &= ~mask
    payload = torch.tensor(
        payload_bytes, dtype=torch.uint8, device="cuda").reshape_as(tensors[0])

    with pytest.raises(RuntimeError, match="payload decoded.*invalid native code"):
        _raw_decode(extension, wire, tensors, payload=payload)


def test_raw_cuda_gemv_rejects_zero_batch_before_grid_launch():
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    payload, schedule, offsets, previous, lut, scales, family = tensors
    x = torch.empty((0, wire.columns), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="batch must be positive"):
        extension.trellis_r256_dequant_gemv(
            x, payload, schedule, offsets, previous, lut, scales, wire.rows,
            wire.columns, wire.row_stride_bytes, family)


def test_raw_cuda_abi_rejects_int64_to_int_narrowing():
    extension, wire, tensors = _raw_inputs(trellis.TCQ_E2M1_R256)
    payload, schedule, offsets, previous, lut, _scales, family = tensors
    with pytest.raises(RuntimeError, match="rows must fit int32"):
        extension.trellis_r256_decode_codes(
            payload, schedule, offsets, previous, lut, 1 << 31,
            wire.columns, wire.row_stride_bytes, family)
