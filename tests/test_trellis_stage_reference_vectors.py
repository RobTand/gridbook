"""Frozen Stage-3/Stage-5 decode vectors for the research TCQ wire.

These expected digests were generated on 2026-08-25 from the campaign's
independent measurement references, not from :mod:`gridbook.trellis`:

* ``stage3_mixed_rate.py``
  ``b137b282ca7d67828a26aa24470d5ef219eaca146255ea2043d7c0ee5fb85795``
* ``tcq_pilot.py``
  ``13bd902641ec7385cf84d96f4c0d8192acdbf3c72f0976c42359c3b4b6faeb2d``
* ``stage5_e4m3_codec.py``
  ``eaf3221cf04049b3bf57e555ad2e0fb24ff81ae8eecf71e42558a25395eebb22``
* ``stage5_encoder.py``
  ``16559405911eb6a397636cde4def0875b54830a47fcb883b07920732cb4240af``

The compact fixture regenerates only its input planes using fixed integer
arithmetic.  The expected native-code and native-packed digests below are
static outputs of the external references.  No campaign path is read at test
time.  E2M1 cases include complete mixed-rate blocks and short tail-biting
blocks; E4M3 cases cover shaped rates 3--6 plus native rate-8 bypass.  Every
physical block has at least eight coded positions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import shutil
import struct

import pytest

from gridbook import trellis


_ROWS = 2

# Canonical native-code alphabets frozen by the external Stage-5 codec.  Hex is
# row-major sorted-code order; the trellis partition is positions j, j+4, ... .
_E2_ALPHABET_HEX = {
    1: "0f0a0207",
    2: "0f0d0b0908020407",
    3: "0f0e0d0c0b0a09000801020304050607",
}
_E4_ALPHABET_HEX = {
    3: "feeddccbbaa998870718293a4b5c6d7e",
    4: (
        "fef6eee6ded5cdc5bdb4aca49c948b83020a131b232b333c444c545d656d757e"
    ),
    5: (
        "fefaf6f2eeeae6e2dedad6d2cecac6c2bebab6b2aea9a5a19d9995918d898581"
        "8004080c1014181c2024292d3135393d4145494d5155595d6165696d7175797e"
    ),
    6: (
        "fefcfaf8f6f4f2f0eeeceae8e6e4e2e0dedcdad8d6d4d2d0cecccac8c6c4c2c0"
        "bebcbab8b6b4b2b0aeacaaa8a6a4a2a09e9c9a98969492908e8c8a88868482"
        "008001030507090b0d0f11131517191b1d1f21232527292b2d2f31333537393b"
        "3d3f41434547494b4d4f51535557595b5d5f61636567696b6d6f71737577797b7e"
    ),
}


@dataclass(frozen=True)
class _GoldenCase:
    family: str
    seed: int
    columns: int
    body_rate_q256: int
    layout: str
    native_codes_sha256: str
    native_packed_sha256: str

    @property
    def id(self) -> str:
        return f"{self.family}-s{self.seed}-c{self.columns}"


_CASES = tuple(
    _GoldenCase(*values)
    for values in (
        ("e2", 101, 256, 654, "fixed", "e8c913d55eb7e20dd1714ba577e5fdfc1d2b69babd159f3e57d3ad6a718dbade", "762391abf82573db2ee9b550dd8e851cfcd675bb5bdb7b46b1801a32ddb526a5"),
        ("e2", 102, 256, 616, "fixed", "be4eeb99c14d79ebb80b162691d772ef196e3865b62f332e1cf85e994f46a15e", "90e70d1326d6435f4fdbca3acdcde29d08eb1b676c67a36122b54b11428806dd"),
        ("e2", 103, 256, 626, "fixed", "3859c1744db1e1fd331f63ee5429fa22c7e4399f23847e334e7b21c03a730e4c", "35944de1a4bf67a29f8392021f331a6ef6557da457a7a172836f72f4c598c19d"),
        ("e2", 104, 256, 658, "fixed", "cd2a9295c1fcbf9aec8d608f56373411f6febe3fee47c7ef5031e40cf65400bd", "83d6e872775f1c7a57437f192fa6486061fb37929bb9950fd8f79034428ea33f"),
        ("e2", 105, 256, 628, "fixed", "8158dc4bca33fcaaafcbdcb860ec400aef90ab75685d118b666a1cb4735a027a", "e952645732c66c7cb63889b9dd1d166b81e9ca8c6d25a622cdd97e876556df12"),
        ("e2", 106, 256, 623, "fixed", "a4bae7bb29b0b36a960b658fd7e341c62e4aa727bbb47cd6d691617e06e0751f", "ea8c6f2010122fc91670d91cf75f72a904f5fe4b43fc8f79c77a9008af0236e1"),
        ("e2", 107, 8, 512, "tight", "44f9d01b8a90e195541e5858daf5b31c56688f6fc563ecc787ca60ab93ada6b7", "5e82cece2cbb4bbcfc92bef3b9400654f10fefba387894b6a220039fb66f11e9"),
        ("e2", 108, 17, 587, "tight", "7fd4dd2ba3d79944018f45e15e9d26eed7ff2300e8fc640d88c1294b205c84fa", "54924e72f2db9545e3362767260b26f633592d67e6887b7c3bfed379592e9a7b"),
        ("e2", 109, 63, 601, "tight", "1548aef7e338c822ba3a8171604e14b1221601983a0d2dd3c6b5b4a508411018", "76358642f2673db8972b4620e7950e80e09476cc240bd6059a641b32755b6149"),
        ("e2", 110, 129, 645, "tight", "ffc82c0bc753865027fdf98690f235c5d9343dc273c8527f4f356fcd31982918", "7ad01001a3bd2af9a7722203482f50d4a69b1358028d842e58198abe267cdfd2"),
        ("e2", 111, 255, 656, "tight", "b241820442fb63c5b76e399821529f8cfbc1116e128b3b3bbe3ccec6b9cc4f14", "379d6256d9f46d19e8b50cc78bfbe7a638f7c1e99cf1873793315aaca3fff99f"),
        ("e2", 112, 265, 646, "fixed", "d381b25ef79b37acf6c4aa877b8ab53c0ef24068cb95ad7a6900fac8d4fdf50c", "894afefc8c0dc33c4b3e6925e567f2738a77f6670b1dc0dcd4ad2e38f3ba28a3"),
        ("e4", 201, 256, 1356, "fixed", "53f2bf39ac7d897042d9a4d83c35789d465f1d48764158ac3b5c8ecdd01248fa", "53f2bf39ac7d897042d9a4d83c35789d465f1d48764158ac3b5c8ecdd01248fa"),
        ("e4", 202, 256, 1330, "fixed", "b273ee03f5ec0a95722de90410ee76f91fff07764e417b141a5188dc98640627", "b273ee03f5ec0a95722de90410ee76f91fff07764e417b141a5188dc98640627"),
        ("e4", 203, 256, 1381, "fixed", "f0d60e91f939edfc3d06139eed82410bb537b2a90ac8a024c0dc09e6a859b369", "f0d60e91f939edfc3d06139eed82410bb537b2a90ac8a024c0dc09e6a859b369"),
        ("e4", 204, 256, 1321, "fixed", "2cba21f84eb22c09cb795f01ee800dfba677d55bacd7f5f7daaca47e208ed782", "2cba21f84eb22c09cb795f01ee800dfba677d55bacd7f5f7daaca47e208ed782"),
        ("e4", 205, 256, 1355, "fixed", "ff5c9454633e3997b16c56be434d4875b08ec70862628512265cab99f0fa781b", "ff5c9454633e3997b16c56be434d4875b08ec70862628512265cab99f0fa781b"),
        ("e4", 206, 256, 1359, "fixed", "b85b5d4711f9f1d2e6ebd3326d28e839fb280edae539c4cf116fcb11c8f9b236", "b85b5d4711f9f1d2e6ebd3326d28e839fb280edae539c4cf116fcb11c8f9b236"),
        ("e4", 207, 8, 1152, "tight", "ec2ed5483d6b31ae822eac16537a229f7de9f2c2c0302c57ccbad6daf699ae3c", "ec2ed5483d6b31ae822eac16537a229f7de9f2c2c0302c57ccbad6daf699ae3c"),
        ("e4", 208, 31, 1354, "tight", "af548f09099b5eda46708be2eac01649ec9cd6351429baaa98e276f217931528", "af548f09099b5eda46708be2eac01649ec9cd6351429baaa98e276f217931528"),
        ("e4", 209, 127, 1302, "tight", "dc205bc411fed3fbb1230f3183722d3ba08fdd6b0a5eb18f5428094f408de93d", "dc205bc411fed3fbb1230f3183722d3ba08fdd6b0a5eb18f5428094f408de93d"),
        ("e4", 210, 264, 1274, "fixed", "6fb5de094d91314a99ed69debbcae618326e873c63f6843b419d0a83ae4a5527", "6fb5de094d91314a99ed69debbcae618326e873c63f6843b419d0a83ae4a5527"),
        ("e4", 211, 300, 1362, "fixed", "fd3c1cd5098604d59285542684fd02f52fb827351fe4ec2a2235b52e81b29506", "fd3c1cd5098604d59285542684fd02f52fb827351fe4ec2a2235b52e81b29506"),
        ("e4", 212, 511, 1361, "fixed", "7fe5f4953493b717465fd690041c749bb3a3db64fe2c5eeb58ca693bf6775526", "7fe5f4953493b717465fd690041c749bb3a3db64fe2c5eeb58ca693bf6775526"),
    )
)


def _mix(seed: int, row: int, column: int, plane: int = 0) -> int:
    value = (
        seed * 0x9E3779B1
        + row * 0x85EBCA6B
        + column * 0xC2B2AE35
        + plane * 0x27D4EB2F
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def _schedule(case: _GoldenCase) -> tuple[int, ...]:
    choices = (1, 2, 3, 4) if case.family == "e2" else (3, 4, 5, 6, 8)
    values = []
    for column in range(case.columns):
        local = column % trellis.SUPERBLOCK
        if local < 8:
            # Preserve the nu=8 tail-biting floor even for an eight-column tail.
            rate = (1 + (local + case.seed) % 3 if case.family == "e2"
                    else 3 + (local + case.seed) % 4)
        else:
            rate = choices[_mix(case.seed, 0, local, 7) % len(choices)]
        values.append(rate)
    return tuple(values)


def _alphabets(case: _GoldenCase, schedule: tuple[int, ...]):
    terminal = 4 if case.family == "e2" else 8
    source = _E2_ALPHABET_HEX if case.family == "e2" else _E4_ALPHABET_HEX
    return {
        rate: tuple(bytes.fromhex(source[rate]))
        for rate in sorted(set(schedule))
        if rate < terminal
    }


def _planes(case: _GoldenCase, schedule: tuple[int, ...]):
    terminal = 4 if case.family == "e2" else 8
    finite = tuple(
        code for code in range(1 << terminal)
        if case.family != "e4" or code not in (0x7F, 0xFF)
    )
    u_bits, point_indices, bypass_codes = [], [], []
    for row in range(_ROWS):
        u_row, point_row, bypass_row = [], [], []
        for column, rate in enumerate(schedule):
            u_row.append(_mix(case.seed, row, column, 1) & 1)
            point_row.append(
                _mix(case.seed, row, column, 2) % (1 << (rate - 1))
                if rate < terminal else 0
            )
            bypass_row.append(
                finite[_mix(case.seed, row, column, 3) % len(finite)]
                if rate == terminal else 0
            )
        u_bits.append(u_row)
        point_indices.append(point_row)
        bypass_codes.append(bypass_row)
    return u_bits, point_indices, bypass_codes


def _wire(case: _GoldenCase) -> trellis.TrellisWire:
    schedule = _schedule(case)
    u_bits, point_indices, bypass_codes = _planes(case, schedule)
    family = (
        trellis.TCQ_E2M1_R256 if case.family == "e2"
        else trellis.TCQ_E4M3_R256
    )
    layout = (
        trellis.LAYOUT_FIXED_QUOTA_PER_256 if case.layout == "fixed"
        else trellis.LAYOUT_TIGHT_OFFSETS
    )
    if case.family == "e2":
        scale_blob = bytes([0x38] * (_ROWS * ((case.columns + 15) // 16)))
    else:
        scale_blob = struct.pack(f"<{_ROWS}f", *[1.0, 2.0])
    return trellis.pack_planes(
        family=family,
        body_rate_q256=case.body_rate_q256,
        schedule=schedule,
        layout=layout,
        u_bits=u_bits,
        point_indices=point_indices,
        bypass_codes=bypass_codes,
        alphabets=_alphabets(case, schedule),
        scale_blob=scale_blob,
    )


def _native_code_bytes(wire: trellis.TrellisWire) -> bytes:
    return bytes(code for row in trellis.decode_codes(wire) for code in row)


def _native_packed_bytes(wire: trellis.TrellisWire) -> bytes:
    rows = trellis.decode_codes(wire)
    if wire.family == trellis.TCQ_E4M3_R256:
        return bytes(code for row in rows for code in row)
    packed = bytearray()
    for row in rows:
        for column in range(0, wire.columns, 2):
            high = row[column + 1] << 4 if column + 1 < wire.columns else 0
            packed.append(row[column] | high)
    return bytes(packed)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_golden_matrix_covers_required_families_rates_and_tails():
    assert len(_CASES) == 24
    assert any(case.family == "e2" and case.columns == 256 for case in _CASES)
    assert any(case.family == "e2" and case.columns % 256 for case in _CASES)
    e4_rates = set()
    for case in _CASES:
        if case.family == "e4":
            e4_rates.update(_schedule(case))
    assert {3, 4, 5, 6, 8} <= e4_rates


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_cpu_wire_matches_external_stage_reference(case):
    wire = _wire(case)
    parsed = trellis.TrellisWire.from_bytes(wire.to_bytes())
    assert _sha256(_native_code_bytes(parsed)) == case.native_codes_sha256
    assert _sha256(_native_packed_bytes(parsed)) == case.native_packed_sha256


def _native_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return (
        torch.cuda.is_available()
        and shutil.which("nvcc") is not None
        and os.environ.get("PRISMAQUANT_CB_EXT_DIR", "").startswith("/home/rob/")
    )


@pytest.mark.skipif(
    not _native_cuda_available(),
    reason="CUDA+nvcc and a /home/rob extension cache are required",
)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_checked_and_prevalidated_cuda_match_external_stage_reference(case):
    from gridbook.trellis_ops import decode_wire_codes_cuda, prepare_wire_cuda

    wire = _wire(case)
    checked = decode_wire_codes_cuda(wire)
    checked_bytes = bytes(checked.detach().cpu().reshape(-1).tolist())
    assert _sha256(checked_bytes) == case.native_codes_sha256

    packed = prepare_wire_cuda(wire).decode_native_packed()
    packed_bytes = bytes(packed.detach().cpu().reshape(-1).tolist())
    assert _sha256(packed_bytes) == case.native_packed_sha256
