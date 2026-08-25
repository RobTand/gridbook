"""Research-only q256 tail-biting trellis wire and CPU decoder.

This module deliberately does not register a Gridbook quantization format.
It is the executable wire contract used to evaluate the two candidate
families ``TCQ_E2M1_R256`` and ``TCQ_E4M3_R256`` before a producer, chooser,
or serving lane is allowed to claim them.  In particular, none of these IDs
appears in ``runtime_contract.json``.

The body is tight and LSB-first.  Each non-bypass weight stores one coded bit
followed by ``rate - 1`` point bits.  A bypass weight stores its native FP
code directly (four bits for E2M1 or eight for E4M3).  Blocks are not byte
aligned; only the end of a row is padded, to a 16-byte stride.  The
``tight_offsets`` layout serializes one tensor-wide ``block_offset_bits``
vector.  ``fixed_quota_per_256`` gives every complete block the same bit quota
and derives those offsets instead.  Both layouts serialize one tensor-shared
per-input-column schedule, never one schedule per row.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Iterable, Mapping, Sequence


SCHEMA = "gridbook.trellis.wire.v1"
TCQ_E2M1_R256 = "TCQ_E2M1_R256"
TCQ_E4M3_R256 = "TCQ_E4M3_R256"
FAMILIES = (TCQ_E2M1_R256, TCQ_E4M3_R256)

LAYOUT_TIGHT_OFFSETS = "tight_offsets"
LAYOUT_FIXED_QUOTA_PER_256 = "fixed_quota_per_256"
LAYOUTS = (LAYOUT_TIGHT_OFFSETS, LAYOUT_FIXED_QUOTA_PER_256)

SUPERBLOCK = 256
STATES = 256
MEMORY_ORDER = 8
GENERATOR_0 = 0o561
GENERATOR_1 = 0o753
ROW_ALIGNMENT = 16
MIN_TRELLIS_STEPS = MEMORY_ORDER

_MAGIC = b"GBTCQ1\0\0"
_VERSION = 1
_FAMILY_CODE = {TCQ_E2M1_R256: 1, TCQ_E4M3_R256: 2}
_CODE_FAMILY = {value: key for key, value in _FAMILY_CODE.items()}
_LAYOUT_CODE = {LAYOUT_TIGHT_OFFSETS: 1,
                LAYOUT_FIXED_QUOTA_PER_256: 2}
_CODE_LAYOUT = {value: key for key, value in _LAYOUT_CODE.items()}
# 88 bytes, fixed for schema v1.  The third uint32 after rows/columns is the
# nominal body-rate quota in bits per 256 weights; block count is derived from
# columns.  The section lengths make parsing unique.
_HEADER = struct.Struct("<8sBBBBIIIIIIQQf32s")

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_NAN_CODES = frozenset((0x7F, 0xFF))


@dataclass(frozen=True)
class RungPolicy:
    """Research rung surface; all rates are body bits per 256 weights."""

    family: str
    candidate_q256: tuple[int, ...]
    research_floor_q256: int
    research_ceiling_q256: int
    native_terminal_q256: int


RUNG_POLICIES = {
    TCQ_E2M1_R256: RungPolicy(
        TCQ_E2M1_R256, (384, 512, 640, 768, 896), 256, 1016, 1024),
    TCQ_E4M3_R256: RungPolicy(
        TCQ_E4M3_R256, (1152,), 256, 2040, 2048),
}


def rung_id(family: str, body_rate_q256: int) -> str:
    """Return the collision-free experimental rung ID.

    The integer suffix is intentionally not a rounded decimal rate.  For
    example, ``R384`` means exactly 384 body bits per 256 weights (1.5 bpw).
    Side planes and row padding are accounted separately by :func:`account`.
    """
    policy = _policy(family)
    q = _as_int(body_rate_q256, "body_rate_q256")
    if not policy.research_floor_q256 <= q <= policy.research_ceiling_q256:
        raise ValueError(
            f"{family} q256 body rate {q} outside research range "
            f"[{policy.research_floor_q256}, {policy.research_ceiling_q256}]")
    return f"{family.removesuffix('_R256')}_R{q}"


def _policy(family: str) -> RungPolicy:
    try:
        return RUNG_POLICIES[family]
    except KeyError as exc:
        raise ValueError(f"unknown trellis family {family!r}") from exc


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def native_bits(family: str) -> int:
    return 4 if _policy(family).family == TCQ_E2M1_R256 else 8


def max_trellis_rate(family: str) -> int:
    return native_bits(family) - 1


def e2m1_value(code: int) -> float:
    """Exact numeric value of one hardware E2M1 nibble."""
    code = _as_int(code, "E2M1 code")
    if not 0 <= code <= 15:
        raise ValueError(f"E2M1 code outside [0,15]: {code}")
    value = _E2M1_MAGNITUDES[code & 7]
    return -value if code & 8 else value


def e4m3fn_value(code: int) -> float:
    """Decode one finite NVIDIA E4M3FN byte without importing torch."""
    code = _as_int(code, "E4M3 code")
    if not 0 <= code <= 255:
        raise ValueError(f"E4M3 code outside [0,255]: {code}")
    if code in _E4M3_NAN_CODES:
        raise ValueError(f"E4M3 NaN code 0x{code:02x} is forbidden")
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0xF
    mantissa = code & 7
    if exp == 0:
        value = math.ldexp(mantissa / 8.0, 1 - 7)
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exp - 7)
    return sign * value


def code_value(family: str, code: int) -> float:
    return (e2m1_value(code) if family == TCQ_E2M1_R256
            else e4m3fn_value(code))


def canonical_full_alphabet(family: str) -> tuple[int, ...]:
    """Full native-code alphabet, sorted for Ungerboeck partitioning.

    E4M3FN has 254 finite byte patterns.  Its two NaN encodings are replaced
    by duplicate signed-zero slots, yielding the 256 slots required at rate 7
    without ever materializing or serializing a NaN code.
    """
    if family == TCQ_E2M1_R256:
        codes = range(16)
    elif family == TCQ_E4M3_R256:
        codes = [c for c in range(256) if c not in _E4M3_NAN_CODES]
        codes.extend((0x00, 0x80))
    else:
        _policy(family)
        raise AssertionError("unreachable")
    return tuple(sorted(codes, key=lambda c: (code_value(family, c), c)))


def validate_alphabets(
    family: str, alphabets: Mapping[int, Sequence[int]],
    *, used_rates: Iterable[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Validate and canonicalize per-rate native-code alphabets."""
    limit = max_trellis_rate(family)
    selected = set(alphabets) if used_rates is None else set(used_rates)
    selected.discard(native_bits(family))
    out: dict[int, tuple[int, ...]] = {}
    for rate in sorted(selected):
        rate = _as_int(rate, "alphabet rate")
        if not 1 <= rate <= limit:
            raise ValueError(f"invalid {family} trellis alphabet rate {rate}")
        if rate not in alphabets:
            raise ValueError(f"missing alphabet for used rate {rate}")
        values = tuple(_as_int(c, f"alphabet[{rate}] code")
                       for c in alphabets[rate])
        expected = 1 << (rate + 1)
        if len(values) != expected:
            raise ValueError(
                f"alphabet[{rate}] has {len(values)} codes, expected {expected}")
        maximum = (1 << native_bits(family)) - 1
        if any(c < 0 or c > maximum for c in values):
            raise ValueError(f"alphabet[{rate}] contains an out-of-range code")
        if family == TCQ_E4M3_R256:
            bad = sorted(set(values) & _E4M3_NAN_CODES)
            if bad:
                raise ValueError(
                    f"alphabet[{rate}] contains forbidden E4M3 NaN codes {bad}")
        ordered = tuple(sorted(values, key=lambda c: (code_value(family, c), c)))
        if values != ordered:
            raise ValueError(
                f"alphabet[{rate}] must be sorted by decoded value then code")
        duplicates = len(set(values)) != len(values)
        canonical_r7_duplicates = (
            family == TCQ_E4M3_R256
            and rate == 7
            and values.count(0x00) == 2
            and values.count(0x80) == 2
            and all(
                values.count(code) == 1
                for code in range(256)
                if code not in _E4M3_NAN_CODES | {0x00, 0x80}
            )
        )
        if duplicates and not canonical_r7_duplicates:
            raise ValueError(
                f"alphabet[{rate}] contains duplicate native codes")
        out[rate] = values
    return out


def build_q256_schedule(family: str, body_rate_q256: int,
                        columns: int = SUPERBLOCK) -> tuple[int, ...]:
    """Bresenham schedule with an exact integer body-bit quota per q256.

    Complete 256-column blocks all have exactly ``body_rate_q256`` bits.  A
    tail is the deterministic prefix of the same template.  The returned
    schedule is useful as a default for either wire layout; an importance-
    placed fixed-quota schedule may use a different arrangement in each block
    as long as every complete block retains the exact quota.
    """
    bits = native_bits(family)
    q = _as_int(body_rate_q256, "body_rate_q256")
    columns = _as_int(columns, "columns")
    if columns <= 0:
        raise ValueError("columns must be positive")
    maximum_q = bits * SUPERBLOCK - MIN_TRELLIS_STEPS
    if not SUPERBLOCK <= q <= maximum_q:
        raise ValueError(
            f"{family} q256 rate {q} outside [{SUPERBLOCK}, "
            f"{maximum_q}]")
    base, promoted = divmod(q, SUPERBLOCK)
    if promoted and base == bits:
        raise RuntimeError("internal schedule construction crossed terminal rate")
    template = tuple(
        base + int((i + 1) * promoted // SUPERBLOCK >
                   i * promoted // SUPERBLOCK)
        for i in range(SUPERBLOCK))
    if sum(template) != q:
        raise RuntimeError("internal q256 schedule construction lost its quota")
    return tuple(template[i % SUPERBLOCK] for i in range(columns))


def _expanded_schedule(schedule: Sequence[int], columns: int,
                       layout: str) -> tuple[int, ...]:
    raw = tuple(_as_int(rate, "schedule rate") for rate in schedule)
    if layout not in LAYOUTS:
        raise ValueError(f"unknown trellis layout {layout!r}")
    if len(raw) != columns:
        raise ValueError(
            f"{layout} schedule has {len(raw)} entries, expected {columns}")
    return raw


def validate_schedule(family: str, schedule: Sequence[int], columns: int,
                      layout: str) -> tuple[int, ...]:
    expanded = _expanded_schedule(schedule, columns, layout)
    maximum = native_bits(family)
    bad = sorted({rate for rate in expanded if not 1 <= rate <= maximum})
    if bad:
        raise ValueError(f"{family} schedule contains invalid rates {bad}")
    for block, start in enumerate(range(0, columns, SUPERBLOCK)):
        block_schedule = expanded[start:start + SUPERBLOCK]
        trellis_steps = sum(rate < maximum for rate in block_schedule)
        if trellis_steps < MIN_TRELLIS_STEPS:
            raise ValueError(
                f"block {block} has T'={trellis_steps}; tail-biting "
                f"{STATES}-state decode requires T'>={MIN_TRELLIS_STEPS}")
    return expanded


def block_offset_bits(expanded_schedule: Sequence[int]) -> tuple[int, ...]:
    """Tensor-shared bit offsets, with a terminal sentinel."""
    out = [0]
    total = 0
    for start in range(0, len(expanded_schedule), SUPERBLOCK):
        total += sum(expanded_schedule[start:start + SUPERBLOCK])
        out.append(total)
    return tuple(out)


def _pack_nibbles(values: Sequence[int]) -> bytes:
    out = bytearray((len(values) + 1) // 2)
    for i, value in enumerate(values):
        if not 0 <= value <= 15:
            raise ValueError(f"nibble outside [0,15]: {value}")
        out[i // 2] |= value << (4 * (i & 1))
    return bytes(out)


def _unpack_nibbles(data: bytes, count: int) -> tuple[int, ...]:
    return tuple((data[i // 2] >> (4 * (i & 1))) & 15
                 for i in range(count))


def _alphabet_blob(alphabets: Mapping[int, Sequence[int]]) -> bytes:
    out = bytearray()
    for rate, codes in sorted(alphabets.items()):
        out += struct.pack("<BH", rate, len(codes))
        out += bytes(codes)
    return bytes(out)


def _parse_alphabet_blob(blob: bytes) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    offset = 0
    while offset < len(blob):
        if len(blob) - offset < 3:
            raise ValueError("truncated alphabet directory")
        rate, count = struct.unpack_from("<BH", blob, offset)
        offset += 3
        if len(blob) - offset < count:
            raise ValueError("truncated alphabet payload")
        if rate in out:
            raise ValueError(f"duplicate alphabet rate {rate}")
        out[rate] = tuple(blob[offset:offset + count])
        offset += count
    return out


class _BitWriter:
    def __init__(self, size: int):
        self.data = bytearray(size)

    def put(self, bit_offset: int, value: int, width: int) -> None:
        if not 0 <= value < (1 << width):
            raise ValueError(f"value {value} does not fit {width} bits")
        for bit in range(width):
            if (value >> bit) & 1:
                absolute = bit_offset + bit
                self.data[absolute >> 3] |= 1 << (absolute & 7)


def _get_bits(data: bytes, bit_offset: int, width: int) -> int:
    value = 0
    for bit in range(width):
        absolute = bit_offset + bit
        value |= ((data[absolute >> 3] >> (absolute & 7)) & 1) << bit
    return value


@dataclass(frozen=True)
class TrellisAccounting:
    body_bits: int
    body_storage_bytes: int
    row_padding_bytes: int
    header_bytes: int
    schedule_bytes: int
    block_offset_bytes: int
    alphabet_bytes: int
    scale_bytes: int
    side_bytes: int
    total_bytes: int
    exact_bpw: float


@dataclass(frozen=True)
class TrellisWire:
    """Parsed research wire; all byte fields are immutable."""

    family: str
    layout: str
    rows: int
    columns: int
    body_rate_q256: int
    schedule: tuple[int, ...]
    block_offsets_bits: tuple[int, ...]
    alphabets: Mapping[int, tuple[int, ...]]
    scale_blob: bytes
    global_scale_real: float
    row_body_bits: int
    row_stride_bytes: int
    payload: bytes

    @property
    def rung(self) -> str:
        return rung_id(self.family, self.body_rate_q256)

    @property
    def expanded_schedule(self) -> tuple[int, ...]:
        return _expanded_schedule(self.schedule, self.columns, self.layout)

    @property
    def alphabet_digest(self) -> str:
        return hashlib.sha256(_alphabet_blob(self.alphabets)).hexdigest()

    @property
    def alphabet_digests(self) -> Mapping[int, str]:
        """Per-rate digests for receipts, in addition to the bound aggregate."""
        return {
            rate: hashlib.sha256(
                struct.pack("<BH", rate, len(codes)) + bytes(codes)
            ).hexdigest()
            for rate, codes in sorted(self.alphabets.items())
        }

    def row_payload(self, row: int) -> bytes:
        if not 0 <= row < self.rows:
            raise IndexError(row)
        start = row * self.row_stride_bytes
        return self.payload[start:start + self.row_stride_bytes]

    def to_bytes(self) -> bytes:
        # A frozen dataclass can still be constructed directly (and its
        # Mapping-valued alphabet member can be mutated by a caller).  Never
        # serialize such an object without re-establishing every wire
        # invariant, including canonical zero body/row padding.
        self.validate()
        schedule_blob = _pack_nibbles(self.schedule)
        alphabet_blob = _alphabet_blob(self.alphabets)
        if self.layout == LAYOUT_TIGHT_OFFSETS:
            offset_width = 4 if self.row_body_bits <= 0xFFFFFFFF else 8
            offset_code = "I" if offset_width == 4 else "Q"
            offsets_blob = struct.pack(
                f"<{len(self.block_offsets_bits)}{offset_code}",
                *self.block_offsets_bits)
        else:
            offset_width = 0
            offsets_blob = b""
        digest = hashlib.sha256(alphabet_blob).digest()
        header = _HEADER.pack(
            _MAGIC, _VERSION, _FAMILY_CODE[self.family],
            _LAYOUT_CODE[self.layout], offset_width,
            self.rows, self.columns, self.body_rate_q256,
            len(self.schedule), len(alphabet_blob), len(self.scale_blob),
            self.row_body_bits, self.row_stride_bytes,
            float(self.global_scale_real), digest)
        return (header + schedule_blob + offsets_blob + alphabet_blob +
                self.scale_blob + self.payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> "TrellisWire":
        if len(data) < _HEADER.size:
            raise ValueError("truncated trellis header")
        (magic, version, family_code, layout_code, offset_width,
         rows, columns, body_rate_q256, schedule_count, alphabet_size, scale_size,
         row_body_bits, row_stride, global_scale, digest) = _HEADER.unpack_from(data)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("not a gridbook.trellis.wire.v1 payload")
        try:
            family = _CODE_FAMILY[family_code]
            layout = _CODE_LAYOUT[layout_code]
        except KeyError as exc:
            raise ValueError("unknown trellis family/layout code") from exc
        expected_offset_widths = (
            (4, 8) if layout == LAYOUT_TIGHT_OFFSETS else (0,)
        )
        if offset_width not in expected_offset_widths:
            raise ValueError(
                f"invalid block offset width {offset_width} for {layout}")
        cursor = _HEADER.size
        schedule_size = (schedule_count + 1) // 2
        end = cursor + schedule_size
        if end > len(data):
            raise ValueError("truncated schedule")
        schedule = _unpack_nibbles(data[cursor:end], schedule_count)
        if schedule_count & 1 and data[end - 1] & 0xF0:
            raise ValueError("nonzero high padding nibble in schedule")
        cursor = end
        blocks = (columns + SUPERBLOCK - 1) // SUPERBLOCK
        offset_count = blocks + 1 if offset_width else 0
        offsets_size = offset_count * offset_width
        end = cursor + offsets_size
        if end > len(data):
            raise ValueError("truncated block offsets")
        if offset_width:
            offset_code = "I" if offset_width == 4 else "Q"
            offsets = struct.unpack(
                f"<{offset_count}{offset_code}", data[cursor:end])
        else:
            offsets = block_offset_bits(
                _expanded_schedule(schedule, columns, layout))
        cursor = end
        end = cursor + alphabet_size
        if end > len(data):
            raise ValueError("truncated alphabets")
        alphabet_blob = data[cursor:end]
        if hashlib.sha256(alphabet_blob).digest() != digest:
            raise ValueError("alphabet digest mismatch")
        alphabets = _parse_alphabet_blob(alphabet_blob)
        cursor = end
        end = cursor + scale_size
        if end > len(data):
            raise ValueError("truncated scale plane")
        scale_blob = data[cursor:end]
        cursor = end
        payload_size = rows * row_stride
        if len(data) - cursor != payload_size:
            raise ValueError(
                f"payload has {len(data)-cursor} bytes, expected {payload_size}")
        wire = cls(
            family, layout, rows, columns, body_rate_q256, schedule,
            tuple(offsets), alphabets, scale_blob, global_scale,
            row_body_bits, row_stride, data[cursor:])
        wire.validate()
        return wire

    def validate(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive")
        expanded = validate_schedule(
            self.family, self.schedule, self.columns, self.layout)
        rung_id(self.family, self.body_rate_q256)
        if self.layout == LAYOUT_FIXED_QUOTA_PER_256:
            for block, start in enumerate(
                    range(0, self.columns - SUPERBLOCK + 1, SUPERBLOCK)):
                actual = sum(expanded[start:start + SUPERBLOCK])
                if actual != self.body_rate_q256:
                    raise ValueError(
                        f"fixed-quota block {block} has {actual} body bits; "
                        f"expected {self.body_rate_q256}")
        else:
            rate_residual = abs(
                sum(expanded) * SUPERBLOCK
                - self.body_rate_q256 * self.columns
            )
            if rate_residual >= SUPERBLOCK:
                raise ValueError(
                    "tight-offset schedule differs from its declared q256 "
                    "target by at least one physical body bit")
        expected_offsets = block_offset_bits(expanded)
        if tuple(self.block_offsets_bits) != expected_offsets:
            raise ValueError("block_offset_bits does not match the schedule")
        if self.row_body_bits != expected_offsets[-1]:
            raise ValueError("row_body_bits does not match block offsets")
        minimum_stride = (self.row_body_bits + 7) // 8
        expected_stride = ((minimum_stride + ROW_ALIGNMENT - 1) //
                           ROW_ALIGNMENT * ROW_ALIGNMENT)
        if self.row_stride_bytes != expected_stride:
            raise ValueError("row stride is not the exact 16-byte padded size")
        if len(self.payload) != self.rows * self.row_stride_bytes:
            raise ValueError("payload length does not match rows * row stride")
        used = {r for r in expanded if r < native_bits(self.family)}
        validated = validate_alphabets(self.family, self.alphabets,
                                       used_rates=used)
        if set(validated) != set(self.alphabets):
            raise ValueError("wire contains an unused alphabet")
        if self.family == TCQ_E2M1_R256:
            expected_scales = self.rows * ((self.columns + 15) // 16)
            if len(self.scale_blob) != expected_scales:
                raise ValueError(
                    f"E2M1 group16 scale plane is {len(self.scale_blob)} bytes, "
                    f"expected {expected_scales}")
            if not math.isfinite(self.global_scale_real) or self.global_scale_real <= 0:
                raise ValueError("E2M1 global_scale_real must be finite and positive")
            if any(e4m3fn_value(code) <= 0 for code in self.scale_blob):
                raise ValueError(
                    "E2M1 group16 scale bytes must decode finite and positive")
        else:
            if self.global_scale_real != 1.0:
                raise ValueError("E4M3 global_scale_real is fixed at 1.0")
            if len(self.scale_blob) != self.rows * 4:
                raise ValueError("E4M3 requires one fp32 scale per row")
            scales = struct.unpack(f"<{self.rows}f", self.scale_blob)
            if any(not math.isfinite(v) or v <= 0 for v in scales):
                raise ValueError("E4M3 row scales must be finite and positive")
            bypass_offsets: list[int] = []
            for block, start in enumerate(range(0, self.columns, SUPERBLOCK)):
                bit_offset = self.block_offsets_bits[block]
                for column in range(start, min(start + SUPERBLOCK, self.columns)):
                    rate = expanded[column]
                    if rate == 8:
                        bypass_offsets.append(bit_offset)
                    bit_offset += rate
            for row in range(self.rows):
                row_data = self.row_payload(row)
                if any(
                    _get_bits(row_data, offset, 8) in _E4M3_NAN_CODES
                    for offset in bypass_offsets
                ):
                    raise ValueError("wire contains an E4M3 NaN bypass code")
        # Padding bits and bytes have one canonical representation.
        for row in range(self.rows):
            row_data = self.row_payload(row)
            used_bytes = (self.row_body_bits + 7) // 8
            if self.row_body_bits & 7:
                mask = (1 << (self.row_body_bits & 7)) - 1
                if row_data[used_bytes - 1] & ~mask:
                    raise ValueError("nonzero high padding bits in final body byte")
            if any(row_data[used_bytes:]):
                raise ValueError("nonzero row padding bytes")


def pack_planes(
    *, family: str, body_rate_q256: int, schedule: Sequence[int], layout: str,
    u_bits: Sequence[Sequence[int]], point_indices: Sequence[Sequence[int]],
    bypass_codes: Sequence[Sequence[int]],
    alphabets: Mapping[int, Sequence[int]], scale_blob: bytes,
    global_scale_real: float = 1.0,
) -> TrellisWire:
    """Pack already-encoded tail-biting planes into the research wire.

    The offline Viterbi encoder owns the tail-biting proof.  This packer owns
    range, layout, side-plane, and NaN validation; it intentionally cannot
    turn arbitrary values into a purportedly valid constrained path.
    """
    rows = len(u_bits)
    if rows <= 0:
        raise ValueError("at least one row is required")
    columns = len(u_bits[0])
    if columns <= 0:
        raise ValueError("at least one column is required")
    for name, plane in (("u_bits", u_bits),
                        ("point_indices", point_indices),
                        ("bypass_codes", bypass_codes)):
        if len(plane) != rows or any(len(row) != columns for row in plane):
            raise ValueError(f"{name} must have rectangular shape [{rows},{columns}]")
    expanded = validate_schedule(family, schedule, columns, layout)
    used = {rate for rate in expanded if rate < native_bits(family)}
    checked_alphabets = validate_alphabets(
        family, alphabets, used_rates=used)
    if set(checked_alphabets) != set(alphabets):
        raise ValueError("alphabets contains a rate not used by the schedule")
    offsets = block_offset_bits(expanded)
    row_bits = offsets[-1]
    body_bytes = (row_bits + 7) // 8
    row_stride = ((body_bytes + ROW_ALIGNMENT - 1) //
                  ROW_ALIGNMENT * ROW_ALIGNMENT)
    payload = bytearray(rows * row_stride)
    terminal = native_bits(family)
    for row in range(rows):
        writer = _BitWriter(row_stride)
        bit_offset = 0
        for column, rate in enumerate(expanded):
            if rate == terminal:
                code = _as_int(bypass_codes[row][column], "bypass code")
                if family == TCQ_E4M3_R256 and code in _E4M3_NAN_CODES:
                    raise ValueError(
                        f"row {row} column {column}: E4M3 NaN bypass code forbidden")
                writer.put(bit_offset, code, terminal)
            else:
                u = _as_int(u_bits[row][column], "coded bit")
                point = _as_int(point_indices[row][column], "point index")
                writer.put(bit_offset, u, 1)
                writer.put(bit_offset + 1, point, rate - 1)
            bit_offset += rate
        start = row * row_stride
        payload[start:start + row_stride] = writer.data
    canonical_global_scale = struct.unpack(
        "<f", struct.pack("<f", float(global_scale_real)))[0]
    wire = TrellisWire(
        family=family, layout=layout, rows=rows, columns=columns,
        body_rate_q256=_as_int(body_rate_q256, "body_rate_q256"),
        schedule=tuple(schedule), block_offsets_bits=offsets,
        alphabets=checked_alphabets, scale_blob=bytes(scale_blob),
        global_scale_real=canonical_global_scale, row_body_bits=row_bits,
        row_stride_bytes=row_stride, payload=bytes(payload))
    wire.validate()
    return wire


def decode_codes(wire: TrellisWire) -> list[list[int]]:
    """Scan-free 256-state decoder returning native FP code bytes/nibbles."""
    wire.validate()
    schedule = wire.expanded_schedule
    terminal = native_bits(wire.family)
    out = [[0] * wire.columns for _ in range(wire.rows)]
    for block, start in enumerate(range(0, wire.columns, SUPERBLOCK)):
        stop = min(start + SUPERBLOCK, wire.columns)
        trellis_columns = [c for c in range(start, stop)
                           if schedule[c] < terminal]
        ordinal = {column: i for i, column in enumerate(trellis_columns)}
        for row in range(wire.rows):
            data = wire.row_payload(row)
            bit_offset = wire.block_offsets_bits[block]
            column_offsets: dict[int, int] = {}
            for column in range(start, stop):
                column_offsets[column] = bit_offset
                bit_offset += schedule[column]
            for column in range(start, stop):
                rate = schedule[column]
                offset = column_offsets[column]
                if rate == terminal:
                    code = _get_bits(data, offset, terminal)
                    if wire.family == TCQ_E4M3_R256 and code in _E4M3_NAN_CODES:
                        raise ValueError("wire contains an E4M3 NaN bypass code")
                    out[row][column] = code
                    continue
                u = _get_bits(data, offset, 1)
                at = ordinal[column]
                state = 0
                count = len(trellis_columns)
                for i in range(1, MEMORY_ORDER + 1):
                    previous = trellis_columns[(at - i) % count]
                    previous_u = _get_bits(data, column_offsets[previous], 1)
                    state |= previous_u << (MEMORY_ORDER - i)
                register = (u << MEMORY_ORDER) | state
                subset = (2 * ((register & GENERATOR_0).bit_count() & 1) +
                          ((register & GENERATOR_1).bit_count() & 1))
                point = _get_bits(data, offset + 1, rate - 1)
                out[row][column] = wire.alphabets[rate][subset + 4 * point]
    return out


def decoded_scales(wire: TrellisWire) -> list[list[float]]:
    """Numeric scale plane consumed by the reference/CUDA value decoder."""
    if wire.family == TCQ_E2M1_R256:
        groups = (wire.columns + 15) // 16
        return [[e4m3fn_value(wire.scale_blob[row * groups + group]) *
                 wire.global_scale_real for group in range(groups)]
                for row in range(wire.rows)]
    values = struct.unpack(f"<{wire.rows}f", wire.scale_blob)
    return [[float(value)] for value in values]


def decode_values(wire: TrellisWire) -> list[list[float]]:
    """Decode normalized FP codes and apply the contracted scale plane."""
    codes = decode_codes(wire)
    scales = decoded_scales(wire)
    if wire.family == TCQ_E2M1_R256:
        return [[e2m1_value(code) * scales[row][column // 16]
                 for column, code in enumerate(codes[row])]
                for row in range(wire.rows)]
    return [[e4m3fn_value(code) * scales[row][0] for code in codes[row]]
            for row in range(wire.rows)]


def decode_codes_torch(wire: TrellisWire, *, device="cpu"):
    """Pure-reference native codes as a torch tensor, imported lazily."""
    import torch

    return torch.tensor(decode_codes(wire), dtype=torch.uint8, device=device)


def decode_values_torch(wire: TrellisWire, *, device="cpu"):
    """Pure-reference scaled values as a float32 torch tensor."""
    import torch

    return torch.tensor(decode_values(wire), dtype=torch.float32,
                        device=device)


def derived_decode_plan(wire: TrellisWire) -> tuple[list[int], list[list[int]]]:
    """Return absolute column and previous-coded-bit offsets.

    This tensor-shared plan is derived workspace, not artifact side info.  The
    native decoder uses it to do eight independent bit gathers rather than a
    state scan.  Bypass columns carry eight ``-1`` entries.
    """
    schedule = wire.expanded_schedule
    terminal = native_bits(wire.family)
    column_offsets = [0] * wire.columns
    previous_offsets = [[-1] * MEMORY_ORDER for _ in range(wire.columns)]
    for block, start in enumerate(range(0, wire.columns, SUPERBLOCK)):
        stop = min(start + SUPERBLOCK, wire.columns)
        bit_offset = wire.block_offsets_bits[block]
        for column in range(start, stop):
            column_offsets[column] = bit_offset
            bit_offset += schedule[column]
        coded = [column for column in range(start, stop)
                 if schedule[column] < terminal]
        for ordinal, column in enumerate(coded):
            previous_offsets[column] = [
                column_offsets[coded[(ordinal - i) % len(coded)]]
                for i in range(1, MEMORY_ORDER + 1)]
    return column_offsets, previous_offsets


def alphabet_lut(wire: TrellisWire) -> list[list[int]]:
    """Dense rate x 256 native-code table used by the CUDA ABI."""
    maximum = max_trellis_rate(wire.family)
    table = [[0] * 256 for _ in range(maximum + 1)]
    for rate, codes in wire.alphabets.items():
        points = 1 << (rate - 1)
        for subset in range(4):
            for point in range(points):
                table[rate][subset * points + point] = codes[subset + 4 * point]
    return table


def account(wire: TrellisWire) -> TrellisAccounting:
    """Exact body, side-plane, row-padding and total byte accounting."""
    wire.validate()
    schedule_bytes = (len(wire.schedule) + 1) // 2
    offset_width = (
        4 if wire.row_body_bits <= 0xFFFFFFFF else 8
    ) if wire.layout == LAYOUT_TIGHT_OFFSETS else 0
    offset_bytes = len(wire.block_offsets_bits) * offset_width
    alphabet_bytes = len(_alphabet_blob(wire.alphabets))
    body_storage = wire.rows * ((wire.row_body_bits + 7) // 8)
    padding = len(wire.payload) - body_storage
    side = (_HEADER.size + schedule_bytes + offset_bytes + alphabet_bytes +
            len(wire.scale_blob))
    total = side + len(wire.payload)
    return TrellisAccounting(
        body_bits=wire.rows * wire.row_body_bits,
        body_storage_bytes=body_storage, row_padding_bytes=padding,
        header_bytes=_HEADER.size, schedule_bytes=schedule_bytes,
        block_offset_bytes=offset_bytes, alphabet_bytes=alphabet_bytes,
        scale_bytes=len(wire.scale_blob), side_bytes=side,
        total_bytes=total, exact_bpw=8.0 * total / (wire.rows * wire.columns))


def refuse_public_artifact_authority(format_id: str) -> None:
    """Fail closed if research TCQ is offered as a public artifact format."""
    if format_id in FAMILIES or any(
            format_id.startswith(family.removesuffix("_R256") + "_R")
            for family in FAMILIES):
        raise RuntimeError(
            f"{format_id} is research-only under {SCHEMA}; Gridbook has no "
            "public producer, chooser, artifact authority, or qualified "
            "prefill lane for this trellis family")


__all__ = [
    "SCHEMA", "TCQ_E2M1_R256", "TCQ_E4M3_R256", "FAMILIES",
    "LAYOUT_TIGHT_OFFSETS", "LAYOUT_FIXED_QUOTA_PER_256", "RUNG_POLICIES",
    "SUPERBLOCK", "STATES", "GENERATOR_0", "GENERATOR_1",
    "TrellisWire", "TrellisAccounting", "account", "alphabet_lut",
    "block_offset_bits", "build_q256_schedule", "canonical_full_alphabet",
    "code_value", "decode_codes", "decode_codes_torch", "decode_values",
    "decode_values_torch", "decoded_scales",
    "derived_decode_plan", "e2m1_value", "e4m3fn_value", "pack_planes",
    "refuse_public_artifact_authority", "rung_id", "validate_alphabets",
    "validate_schedule",
]
