"""The checkpoint vocabulary that routes a TCQ wire to a Gridbook trellis lane.

WHAT THIS CLOSES.  Both trellis lanes existed as ``LinearMethod`` classes that
nothing could construct: their factories wanted ``rows/columns/
row_stride_bytes/row_body_bits`` from somewhere, and their finalize hooks
required a caller to have already bound ``gridbook_trellis_prepared``.  No
config parsed a trellis target, no dispatch arm reached them, and no loader
built a wire.  This module is the missing declaration half; ``config.py``'s
dispatch arm and the lanes' own finalize hooks are the other two.

THE WIRE IS SELF-DESCRIBING, AND THAT DECIDES THE CARRIAGE.  A ``TrellisWire``
serializes to one blob (``to_bytes``) carrying the header, the per-column rate
schedule, the tight block offsets, the per-rate alphabets, the scale plane and
the padded row bodies.  Only the body is a rectangle; **the schedule, the
alphabets and the scale plane exist nowhere else**.  So a checkpoint cannot
carry "the payload" as a ``[rows, row_stride]`` tensor and rebuild a wire from
it -- the earlier lane drafts declared exactly that parameter, and it is not a
sufficient carrier.  One opaque ``wire_bytes`` blob per Linear is, and it is
what these schemes describe.

EVERY SCALE IS DERIVED FROM THE BLOB, NEVER LOADED BESIDE IT.  ``decoded_scales
(wire)`` and ``wire.scale_blob`` already are the operands the two lanes need
for ``scale_b``, and ``global_scale_real`` is a header field.  A checkpoint
that also carried them as separate tensors could disagree with its own wire,
and nothing would notice; deriving them makes that state unrepresentable.  The
one exception is genuinely not a wire fact: E2M1's ``input_global_scale`` is
the **A-side** static scale the forward quantizes activations with, so it is a
real loaded parameter (as it is for stock NVFP4).

WHY VALIDATION LOOKS AT ``trellis.py`` AND NOT AT ``runtime_contract.json``.
Principle 14 governs claims about *another* runtime -- what vLLM executes,
which kernel a route lands on.  The set of wires this package's own reader
accepts is not such a claim: ``FAMILIES`` and ``RUNG_POLICIES`` are the reader
domain, in-package and authoritative.  Adding a ``formats`` row or a
``lane_eligibility`` cell would instead be asserting a *serving* fact, and no
attestation for these lanes exists yet, so no cell is added: absence already
resolves ``unattested``, which is the honest status for a lane that has never
been loaded by vLLM.
"""
from __future__ import annotations

from typing import Any, Mapping

from .trellis import FAMILIES, RUNG_POLICIES, TCQ_E2M1_R256, TrellisWire
from .qtip_hadamard import validate_online_transform

__all__ = [
    "TRELLIS_SCHEME_KEY",
    "is_trellis_scheme",
    "validate_trellis_scheme",
    "parse_wire_for_scheme",
]

#: The discriminator. A CB scheme carries ``grid``; a trellis scheme carries
#: ``family``. ``config.py`` must test this BEFORE handing a scheme dict to
#: ``_validate_cb_format_scheme``, which refuses any grid outside {fp4, fp8}
#: and would therefore reject every trellis target as malformed CB.
TRELLIS_SCHEME_KEY = "family"

_REQUIRED = ("family", "body_rate_q256", "rows", "columns", "wire_bytes")


def is_trellis_scheme(scheme: Any) -> bool:
    """Whether this ``config_groups`` scheme names a trellis family."""
    return (isinstance(scheme, Mapping)
            and scheme.get(TRELLIS_SCHEME_KEY) in FAMILIES)


def _as_int(scheme: Mapping, field: str, target: str) -> int:
    value = scheme.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"trellis target {target!r}: {field} must be an integer, got "
            f"{value!r}")
    if value <= 0:
        raise ValueError(
            f"trellis target {target!r}: {field} must be positive, got {value}")
    return value


def validate_trellis_scheme(scheme: Mapping, target: str) -> dict:
    """Resolve a declared trellis scheme against this package's reader domain.

    Returns the normalized scheme. Raises ``ValueError`` on anything the
    reader cannot serve, at sidecar-parse time -- before a parameter is sized,
    which is the same fail-closed moment ``_validate_cb_format_scheme`` owns
    for CB.
    """
    family = scheme.get("family")
    if family not in FAMILIES:
        raise ValueError(
            f"trellis target {target!r}: family must be one of {FAMILIES}, "
            f"got {family!r}")
    missing = [f for f in _REQUIRED if f not in scheme]
    if missing:
        raise ValueError(
            f"trellis target {target!r}: scheme is missing {missing}; a "
            "trellis scheme must declare its family, rate and geometry so the "
            "sidecar can be gated without parsing the blob")
    policy = RUNG_POLICIES[family]
    rate = _as_int(scheme, "body_rate_q256", target)
    if not (policy.research_floor_q256 <= rate <= policy.research_ceiling_q256):
        raise ValueError(
            f"trellis target {target!r}: {family} body_rate_q256={rate} is "
            f"outside the reader domain [{policy.research_floor_q256}, "
            f"{policy.research_ceiling_q256}]")
    rows = _as_int(scheme, "rows", target)
    columns = _as_int(scheme, "columns", target)
    if family == TCQ_E2M1_R256 and columns % 16:
        raise ValueError(
            f"trellis target {target!r}: {family} serves through a group-16 "
            f"block-scaled fp4 mainloop, so K must be a multiple of 16, got "
            f"{columns}")
    wire_bytes = _as_int(scheme, "wire_bytes", target)
    normalized = {
        "family": family,
        "body_rate_q256": rate,
        "rows": rows,
        "columns": columns,
        "wire_bytes": wire_bytes,
    }
    online_transform = scheme.get("online_transform")
    if online_transform is not None:
        if family != TCQ_E2M1_R256:
            raise ValueError(
                f"trellis target {target!r}: online_transform is research-"
                f"implemented only for {TCQ_E2M1_R256}, got {family}")
        normalized["online_transform"] = validate_online_transform(
            online_transform, rows=rows, columns=columns, target=target)
    return normalized


def parse_wire_for_scheme(blob: bytes, scheme: Mapping,
                          target: str) -> TrellisWire:
    """Parse a loaded blob and refuse it unless it IS what the scheme declared.

    This is the link that makes the sidecar a gate input rather than prose.
    The scheme is what a chooser, a shipcard and a byte-budget read; the blob
    is what the kernel executes. Nothing else in the load path compares them,
    so a mismatch here would otherwise serve one artifact while every receipt
    described another.
    """
    declared = validate_trellis_scheme(scheme, target)
    if len(blob) != declared["wire_bytes"]:
        raise ValueError(
            f"trellis target {target!r}: scheme declares wire_bytes="
            f"{declared['wire_bytes']} but the loaded blob is {len(blob)} "
            "bytes")
    wire = TrellisWire.from_bytes(bytes(blob))
    actual = {
        "family": wire.family,
        "body_rate_q256": wire.body_rate_q256,
        "rows": wire.rows,
        "columns": wire.columns,
    }
    expected = {k: declared[k] for k in actual}
    if actual != expected:
        raise ValueError(
            f"trellis target {target!r}: the loaded wire is {actual} but the "
            f"sidecar scheme declares {expected}; refusing rather than serving "
            "bytes no receipt describes")
    return wire
