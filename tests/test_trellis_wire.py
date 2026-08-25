"""CPU contract tests for the research-only TCQ R256 wire."""
from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import replace

import pytest

from gridbook import trellis


def _alphabet(family, rate):
    full = trellis.canonical_full_alphabet(family)
    count = 1 << (rate + 1)
    if count == len(full):
        return full
    # Evenly cover the native grid, retaining deterministic sorted order.
    indices = [i * (len(full) - 1) // (count - 1) for i in range(count)]
    return tuple(full[i] for i in indices)


def _unpack_lsb_bits(value: str) -> list[int]:
    packed = bytes.fromhex(value)
    return [
        (packed[index // 8] >> (index & 7)) & 1
        for index in range(len(packed) * 8)
    ]


def _wire(family, q256, columns, layout, seed=0, rows=3):
    expanded = trellis.build_q256_schedule(family, q256, columns)
    schedule = expanded
    terminal = trellis.native_bits(family)
    used = sorted({rate for rate in expanded if rate < terminal})
    alphabets = {rate: _alphabet(family, rate) for rate in used}
    rng = random.Random(seed)
    u = [[0] * columns for _ in range(rows)]
    points = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    finite = [c for c in range(1 << terminal)
              if family != trellis.TCQ_E4M3_R256 or c not in (0x7f, 0xff)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                bypass[row][column] = rng.choice(finite)
            else:
                u[row][column] = rng.getrandbits(1)
                points[row][column] = rng.randrange(1 << (rate - 1))
    if family == trellis.TCQ_E2M1_R256:
        scales = bytes([0x38] * (rows * ((columns + 15) // 16)))
        global_real = 0.75
    else:
        scales = struct.pack(f"<{rows}f", *[0.5 + row for row in range(rows)])
        global_real = 1.0
    return trellis.pack_planes(
        family=family, body_rate_q256=q256, schedule=schedule, layout=layout,
        u_bits=u,
        point_indices=points, bypass_codes=bypass, alphabets=alphabets,
        scale_blob=scales, global_scale_real=global_real)


_RANDOM_CASES = [
    # E2M1: low endpoint, odd body-bit boundaries, candidate rungs, ceiling,
    # bypass, and exactly T'=8.
    (trellis.TCQ_E2M1_R256, 256, 256),
    (trellis.TCQ_E2M1_R256, 257, 265),
    (trellis.TCQ_E2M1_R256, 383, 511),
    (trellis.TCQ_E2M1_R256, 384, 256),
    (trellis.TCQ_E2M1_R256, 511, 265),
    (trellis.TCQ_E2M1_R256, 512, 511),
    (trellis.TCQ_E2M1_R256, 640, 256),
    (trellis.TCQ_E2M1_R256, 767, 265),
    (trellis.TCQ_E2M1_R256, 768, 511),
    (trellis.TCQ_E2M1_R256, 769, 256),
    (trellis.TCQ_E2M1_R256, 896, 256),
    (trellis.TCQ_E2M1_R256, 1016, 256),
    # E4M3: bottom, candidate band, full finite trellis alphabet, one bypass,
    # and exactly T'=8.
    (trellis.TCQ_E4M3_R256, 256, 256),
    (trellis.TCQ_E4M3_R256, 513, 265),
    (trellis.TCQ_E4M3_R256, 767, 511),
    (trellis.TCQ_E4M3_R256, 768, 256),
    (trellis.TCQ_E4M3_R256, 896, 265),
    (trellis.TCQ_E4M3_R256, 1025, 511),
    (trellis.TCQ_E4M3_R256, 1152, 256),
    (trellis.TCQ_E4M3_R256, 1280, 265),
    (trellis.TCQ_E4M3_R256, 1536, 511),
    (trellis.TCQ_E4M3_R256, 1792, 256),
    (trellis.TCQ_E4M3_R256, 1793, 256),
    (trellis.TCQ_E4M3_R256, 2040, 256),
]


@pytest.mark.parametrize("family,q256,columns", _RANDOM_CASES)
@pytest.mark.parametrize("layout", [
    trellis.LAYOUT_TIGHT_OFFSETS,
    trellis.LAYOUT_FIXED_QUOTA_PER_256,
])
def test_random_wire_roundtrip_is_bit_exact_and_exactly_accounted(
        family, q256, columns, layout):
    seed = (q256 * 1000003 + columns * 101 +
            (17 if family == trellis.TCQ_E4M3_R256 else 0) +
            (31 if layout == trellis.LAYOUT_FIXED_QUOTA_PER_256 else 0))
    wire = _wire(family, q256, columns, layout, seed=seed)
    encoded = wire.to_bytes()
    parsed = trellis.TrellisWire.from_bytes(encoded)

    assert parsed.to_bytes() == encoded
    assert trellis.decode_codes(parsed) == trellis.decode_codes(wire)
    assert trellis.decode_values(parsed) == trellis.decode_values(wire)
    accounting = trellis.account(parsed)
    assert accounting.total_bytes == len(encoded)
    assert accounting.body_bits == wire.rows * wire.row_body_bits
    assert accounting.row_padding_bytes >= 0
    assert wire.row_stride_bytes % 16 == 0
    assert wire.block_offsets_bits[-1] == wire.row_body_bits
    assert wire.alphabet_digest == hashlib.sha256(
        b"".join(struct.pack("<BH", rate, len(codes)) + bytes(codes)
                 for rate, codes in sorted(wire.alphabets.items()))).hexdigest()


@pytest.mark.parametrize("family,q256", [
    (trellis.TCQ_E2M1_R256, 1016),
    (trellis.TCQ_E4M3_R256, 2040),
])
def test_short_tail_biting_endpoint_is_exactly_eight_steps(family, q256):
    wire = _wire(family, q256, 256,
                 trellis.LAYOUT_FIXED_QUOTA_PER_256)
    terminal = trellis.native_bits(family)
    assert sum(rate < terminal for rate in wire.expanded_schedule) == 8
    _ = trellis.decode_codes(wire)


def test_stage5_viterbi_golden_vector_closes_and_decodes():
    """Freeze one real Stage-5 eager-Viterbi result at seed 20260825."""
    u = _unpack_lsb_bits(
        "8070d48f68c8eb1d131b04c20f1f418f"
        "a1280d304d965f9d3bd4e9fd70fb91d6"
    )
    points = _unpack_lsb_bits(
        "e165bed867bba5cf94f2d1f78f1bb4a0"
        "c31f7de7fb9a54f7a7fba64dbd73c665"
    )
    # This is the exact sorted E2M1 rate-2 alphabet supplied to the encoder.
    codes = (15, 13, 11, 9, 8, 2, 4, 7)
    wire = trellis.pack_planes(
        family=trellis.TCQ_E2M1_R256, body_rate_q256=512,
        schedule=[2] * 256,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
        u_bits=[u], point_indices=[points], bypass_codes=[[0] * 256],
        alphabets={2: codes}, scale_blob=bytes([0x38] * 16))
    decoded = bytes(trellis.decode_codes(wire)[0])
    assert hashlib.sha256(decoded).hexdigest() == (
        "289f28f80580fcd7565401b9b1f0d9c"
        "79451a31b7f46e76d98dbc4b1c5b78e61"
    )

    start_state = sum(u[-offset] << (8 - offset)
                      for offset in range(1, 9))
    state = start_state
    for bit in u:
        state = ((bit << 7) | (state >> 1)) & 0xFF
    assert start_state == state == 214


def test_non_byte_aligned_block_boundary_and_short_tail():
    wire = _wire(trellis.TCQ_E2M1_R256, 385, 265,
                 trellis.LAYOUT_FIXED_QUOTA_PER_256)
    assert wire.block_offsets_bits[1] == 385
    assert wire.block_offsets_bits[1] % 8 == 1
    assert len([r for r in wire.expanded_schedule[256:] if r < 4]) == 9
    assert trellis.TrellisWire.from_bytes(wire.to_bytes()) == wire


def test_fixed_quota_keeps_full_importance_schedule_and_omits_offsets():
    family = trellis.TCQ_E2M1_R256
    schedule = [1, 3] * 128 + [3, 1] * 128
    columns = len(schedule)
    wire = trellis.pack_planes(
        family=family, body_rate_q256=512, schedule=schedule,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
        u_bits=[[column & 1 for column in range(columns)]],
        point_indices=[[0] * columns], bypass_codes=[[0] * columns],
        alphabets={1: _alphabet(family, 1), 3: _alphabet(family, 3)},
        scale_blob=bytes([0x38] * ((columns + 15) // 16)))
    parsed = trellis.TrellisWire.from_bytes(wire.to_bytes())
    assert parsed.schedule == tuple(schedule)
    assert parsed.block_offsets_bits == (0, 512, 1024)
    assert trellis.account(parsed).block_offset_bytes == 0
    assert trellis.account(parsed).schedule_bytes == columns // 2

    bad = schedule.copy()
    bad[0], bad[256] = 2, 2
    with pytest.raises(ValueError, match="fixed-quota block 0"):
        trellis.pack_planes(
            family=family, body_rate_q256=512, schedule=bad,
            layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
            u_bits=[[0] * columns], point_indices=[[0] * columns],
            bypass_codes=[[0] * columns],
            alphabets={1: _alphabet(family, 1), 2: _alphabet(family, 2),
                       3: _alphabet(family, 3)},
            scale_blob=bytes([0x38] * ((columns + 15) // 16)))


def test_tail_block_with_exactly_eight_coded_steps_roundtrips():
    family = trellis.TCQ_E2M1_R256
    schedule = list(trellis.build_q256_schedule(family, 385, 256))
    schedule.extend([3] * 8)
    columns = len(schedule)
    alphabet = {rate: _alphabet(family, rate)
                for rate in sorted(set(schedule))}
    wire = trellis.pack_planes(
        family=family,
        body_rate_q256=(sum(schedule) * 256 + columns // 2) // columns,
        schedule=schedule, layout=trellis.LAYOUT_TIGHT_OFFSETS,
        u_bits=[[column & 1 for column in range(columns)]],
        point_indices=[[0] * columns], bypass_codes=[[0] * columns],
        alphabets=alphabet,
        scale_blob=bytes([0x38] * ((columns + 15) // 16)))
    assert wire.block_offsets_bits == (0, 385, 409)
    assert len([rate for rate in wire.expanded_schedule[256:] if rate < 4]) == 8
    assert trellis.decode_codes(trellis.TrellisWire.from_bytes(wire.to_bytes()))


def test_schedule_refuses_a_block_with_fewer_than_eight_coded_positions():
    schedule = [4] * 256
    schedule[:7] = [3] * 7
    with pytest.raises(ValueError, match="T'=7"):
        trellis.validate_schedule(
            trellis.TCQ_E2M1_R256, schedule, 256,
            trellis.LAYOUT_FIXED_QUOTA_PER_256)


def test_schedule_refuses_an_all_bypass_block():
    with pytest.raises(ValueError, match="T'=0"):
        trellis.validate_schedule(
            trellis.TCQ_E4M3_R256, [8] * 256, 256,
            trellis.LAYOUT_FIXED_QUOTA_PER_256)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data[:20], "truncated trellis header"),
        (lambda data: data[:9] + b"\xff" + data[10:],
         "unknown trellis family/layout code"),
        (lambda data: data[:10] + b"\xff" + data[11:],
         "unknown trellis family/layout code"),
        (lambda data: data[:11] + b"\x04" + data[12:],
         "invalid block offset width"),
        (lambda data: data + b"\x00", "payload has"),
        (lambda data: data[:-1], "payload has"),
    ],
)
def test_malformed_wire_envelope_is_refused(mutation, match):
    wire = _wire(trellis.TCQ_E2M1_R256, 384, 256,
                 trellis.LAYOUT_FIXED_QUOTA_PER_256)
    with pytest.raises(ValueError, match=match):
        trellis.TrellisWire.from_bytes(mutation(wire.to_bytes()))


def test_nonzero_schedule_padding_nibble_is_refused():
    wire = _wire(trellis.TCQ_E2M1_R256, 384, 265,
                 trellis.LAYOUT_TIGHT_OFFSETS)
    data = bytearray(wire.to_bytes())
    # Header is 88 bytes; the high nibble of the last schedule byte is padding.
    data[88 + len(wire.schedule) // 2] |= 0xF0
    with pytest.raises(ValueError, match="padding nibble"):
        trellis.TrellisWire.from_bytes(bytes(data))


@pytest.mark.parametrize("mutation,match", [
    (lambda wire: replace(
        wire, block_offsets_bits=(0,) * len(wire.block_offsets_bits)),
     "block_offset_bits"),
    (lambda wire: replace(
        wire, payload=wire.payload[:-1] + b"\x01"),
     "row padding"),
])
def test_to_bytes_revalidates_directly_constructed_wire(mutation, match):
    wire = _wire(trellis.TCQ_E2M1_R256, 385, 265,
                 trellis.LAYOUT_TIGHT_OFFSETS)
    invalid = mutation(wire)
    with pytest.raises(ValueError, match=match):
        invalid.to_bytes()


@pytest.mark.parametrize("where", ["alphabet", "bypass", "scale"])
def test_e4m3_nan_codes_are_refused_everywhere(where):
    family = trellis.TCQ_E4M3_R256
    schedule = [1] * 256
    alphabet = list(_alphabet(family, 1))
    bypass = [[0] * 256]
    scale = struct.pack("<f", 1.0)
    if where == "alphabet":
        alphabet[-1] = 0x7f
        alphabet.sort(key=lambda c: (
            float("inf") if c == 0x7f else trellis.e4m3fn_value(c), c))
    elif where == "bypass":
        schedule = [8] * 256
        schedule[:8] = [1] * 8
        bypass[0][8] = 0xff
    else:
        # E2M1's group scale is itself E4M3FN, so exercise that path.
        family = trellis.TCQ_E2M1_R256
        schedule = [1] * 256
        alphabet = list(_alphabet(family, 1))
        scale = bytes([0x7f] * 16)
    with pytest.raises(ValueError, match="NaN|positive"):
        trellis.pack_planes(
            family=family, body_rate_q256=256, schedule=schedule,
            layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
            u_bits=[[0] * 256], point_indices=[[0] * 256],
            bypass_codes=bypass, alphabets={1: alphabet}, scale_blob=scale)


def test_alphabet_digest_corruption_is_refused():
    wire = _wire(trellis.TCQ_E2M1_R256, 384, 256,
                 trellis.LAYOUT_FIXED_QUOTA_PER_256)
    data = bytearray(wire.to_bytes())
    # Header is 88 bytes and the fixed-quota schedule is 128 bytes. Offsets
    # are derived for this layout; mutate the first code after its 3B entry.
    alphabet_code = 88 + 128 + 3
    data[alphabet_code] ^= 1
    with pytest.raises(ValueError, match="alphabet digest mismatch"):
        trellis.TrellisWire.from_bytes(bytes(data))


@pytest.mark.parametrize("format_id", [
    trellis.TCQ_E2M1_R256,
    trellis.TCQ_E4M3_R256,
    "TCQ_E2M1_R384",
    "TCQ_E4M3_R1536",
])
def test_research_formats_refuse_public_artifact_authority(format_id):
    with pytest.raises(RuntimeError, match="research-only"):
        trellis.refuse_public_artifact_authority(format_id)


def test_research_ids_are_absent_from_public_runtime_contract():
    from pathlib import Path

    contract = (Path(trellis.__file__).with_name("runtime_contract.json")
                .read_text())
    assert "TCQ_E2M1" not in contract
    assert "TCQ_E4M3" not in contract


def test_accounting_refuses_a_directly_constructed_invalid_wire():
    wire = _wire(
        trellis.TCQ_E2M1_R256, 385, 256,
        trellis.LAYOUT_FIXED_QUOTA_PER_256, seed=19)
    invalid = trellis.TrellisWire(
        family=wire.family, layout=wire.layout, rows=wire.rows,
        columns=wire.columns, body_rate_q256=wire.body_rate_q256,
        schedule=wire.schedule, block_offsets_bits=wire.block_offsets_bits,
        alphabets=wire.alphabets, scale_blob=wire.scale_blob,
        global_scale_real=wire.global_scale_real,
        row_body_bits=wire.row_body_bits,
        row_stride_bytes=wire.row_stride_bytes,
        payload=wire.payload[:-1] + b"\x01")
    with pytest.raises(ValueError, match="padding"):
        trellis.account(invalid)


def test_rung_names_use_exact_q256_suffixes():
    assert trellis.rung_id(trellis.TCQ_E2M1_R256, 384) == "TCQ_E2M1_R384"
    assert trellis.rung_id(trellis.TCQ_E4M3_R256, 1536) == "TCQ_E4M3_R1536"
    assert trellis.RUNG_POLICIES[trellis.TCQ_E2M1_R256].candidate_q256 == (
        384, 512, 640, 768, 896)
    assert trellis.RUNG_POLICIES[trellis.TCQ_E4M3_R256].candidate_q256 == (
        1152,)
    assert trellis.rung_id(trellis.TCQ_E2M1_R256, 1016) == "TCQ_E2M1_R1016"
    assert trellis.rung_id(trellis.TCQ_E4M3_R256, 2040) == "TCQ_E4M3_R2040"
    with pytest.raises(ValueError):
        trellis.rung_id(trellis.TCQ_E2M1_R256, 1024)


def test_canonical_e4m3_alphabet_has_256_finite_slots():
    alphabet = trellis.canonical_full_alphabet(trellis.TCQ_E4M3_R256)
    assert len(alphabet) == 256
    assert 0x7f not in alphabet and 0xff not in alphabet
    assert alphabet.count(0x00) == 2
    assert alphabet.count(0x80) == 2
    assert all(trellis.e4m3fn_value(code) == trellis.e4m3fn_value(code)
               for code in alphabet)


def test_dedicated_cuda_abi_exports_only_trellis_symbols():
    from gridbook import cuda_ext

    source = ((trellis.__file__.rsplit("/", 1)[0]) +
              "/csrc/trellis_r256.cu")
    text = open(source, encoding="utf-8").read()
    for symbol in cuda_ext._TRELLIS_R256_SYMBOLS:
        assert f'"{symbol}"' in text
    assert not any(symbol.startswith("cb_")
                   for symbol in cuda_ext._TRELLIS_R256_SYMBOLS)
    assert ("trellis_r256", "get_trellis_r256_ext") not in (
        cuda_ext._PRELOAD_FAMILIES)
