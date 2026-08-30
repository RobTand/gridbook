"""Research-only QTIP-style online transforms for the E2M1 trellis lane.

The native W4A4 GEMM stays unchanged.  For a stored transformed weight

    Q ~= R_out W R_in.T,       R = H D,

the activation side applies ``R_in`` before native FP4 quantization and the
output side applies ``R_out.T`` after ``_scaled_mm``.  With row-major batches
those operations are, respectively, ``x D H`` and ``y H D``.

CUDA BF16 tensors with ``block_size=128`` use an isolated opaque CUDA op: one
warp owns each 128-value block, performs H4 in registers, and completes H128
with warp shuffles.  The op is graph-safe after its source-identity-keyed JIT
module has been prepared at model load.  CPU, non-BF16 and non-128 calls retain
the transparent torch reference.  A matching CUDA BF16/128 call fails closed
when the native module is unavailable rather than silently changing its
execution contract.  Both paths remain research-only; the native primitive is
not a claim of fusion with activation quantization or the GEMM epilogue.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping

import torch

from .lane_select import latched_bool

__all__ = [
    "ONLINE_HADAMARD_FLAG",
    "TRANSFORM_SCHEMA",
    "TRANSFORM_ALGORITHM",
    "TRANSFORM_NORMALIZATION",
    "TRANSFORM_PADDING",
    "SIGN_GENERATOR",
    "online_hadamard_enabled",
    "seeded_sign_digest",
    "seeded_signs",
    "online_transform_digest",
    "validate_online_transform",
    "prepare_qtip_hadamard_cuda",
    "prepare_qtip_hadamard_model_load",
    "apply_input_transform",
    "apply_inverse_output_transform",
]

ONLINE_HADAMARD_FLAG = "GRIDBOOK_QTIP_ONLINE_HADAMARD_RESEARCH"
TRANSFORM_SCHEMA = "gridbook.qtip-online-hadamard.v1"
TRANSFORM_ALGORITHM = "block_walsh_hadamard"
TRANSFORM_NORMALIZATION = "orthonormal"
TRANSFORM_PADDING = "none"
SIGN_GENERATOR = "sha256_counter_rademacher"

_ROOT_PAYLOAD_FIELDS = frozenset({
    "schema", "algorithm", "normalization", "padding", "input", "output",
})
_ROOT_FIELDS = _ROOT_PAYLOAD_FIELDS | {"transform_sha256"}
_SIDE_FIELDS = frozenset({
    "dimension", "block_size", "seed", "sign_generator", "sign_sha256",
})
_SIGN_DOMAIN = (TRANSFORM_SCHEMA + "/signs\0").encode("ascii")
_NATIVE_BLOCK_SIZE = 128


@torch.library.custom_op(
    "prismaquant::qtip_hadamard_warp128", mutates_args=())
def _qtip_hadamard_warp128(
    value: torch.Tensor, signs: torch.Tensor, sign_before: bool,
) -> torch.Tensor:
    """Opaque research CUDA primitive; public wrappers own dispatch policy."""
    from .cuda_ext import require_qtip_hadamard_warp128_ext

    return require_qtip_hadamard_warp128_ext(
        "QTIP online block-128 sign/Hadamard transform",
    ).qtip_hadamard_warp128(value, signs, sign_before)


@_qtip_hadamard_warp128.register_fake
def _qtip_hadamard_warp128_fake(value, signs, sign_before):
    return torch.empty_like(value, dtype=torch.bfloat16)


def prepare_qtip_hadamard_cuda() -> None:
    """Build/validate the research native module outside first forward.

    The E2M1 lane calls this during model load for every transformed artifact
    whose input or output block size is 128. Direct research callers that plan
    to capture a CUDA graph must call it before entering the capture region.
    """
    from .cuda_ext import require_qtip_hadamard_warp128_ext

    require_qtip_hadamard_warp128_ext(
        "QTIP online block-128 model-load preparation")


def prepare_qtip_hadamard_model_load(
        input_block_size: int, output_block_size: int) -> None:
    """Prepare native code iff the artifact explicitly declares H128.

    This is a dispatch decision only: it never rewrites producer block
    geometry. H256 and every other valid size continue to the torch reference.
    """
    if input_block_size == _NATIVE_BLOCK_SIZE \
            or output_block_size == _NATIVE_BLOCK_SIZE:
        prepare_qtip_hadamard_cuda()


def online_hadamard_enabled() -> bool:
    """Return the latched, explicit research opt-in for online transforms."""
    return latched_bool(
        ONLINE_HADAMARD_FLAG,
        default=False,
        meaning="the research QTIP online sign/Hadamard transform",
    )


def _require_plain_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def _sign_bit_bytes(role: str, dimension: int, seed: int) -> bytes:
    """Generate packed Rademacher bits (LSB-first; one means negative).

    The construction is intentionally independent of Python, torch, NumPy and
    their RNG versions.  Block ``j`` is

    ``SHA256(domain || role || N_le64 || seed_le64 || j_le64)``.

    Its 256 bits are consumed bytewise, least-significant bit first.  The last
    byte is masked above ``dimension`` before the sign-vector digest is taken.
    """
    if role not in ("input", "output"):
        raise ValueError(f"role must be 'input' or 'output', got {role!r}")
    dimension = _require_plain_int(dimension, "dimension")
    seed = _require_plain_int(seed, "seed")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if not 0 <= seed < (1 << 64):
        raise ValueError(f"seed must fit uint64, got {seed}")
    prefix = (
        _SIGN_DOMAIN
        + role.encode("ascii") + b"\0"
        + dimension.to_bytes(8, "little")
        + seed.to_bytes(8, "little")
    )
    needed = (dimension + 7) // 8
    out = bytearray()
    counter = 0
    while len(out) < needed:
        out.extend(hashlib.sha256(
            prefix + counter.to_bytes(8, "little")
        ).digest())
        counter += 1
    del out[needed:]
    if dimension % 8:
        out[-1] &= (1 << (dimension % 8)) - 1
    return bytes(out)


def seeded_sign_digest(role: str, dimension: int, seed: int) -> str:
    """SHA-256 of the canonical packed sign-bit vector."""
    return hashlib.sha256(_sign_bit_bytes(role, dimension, seed)).hexdigest()


def seeded_signs(role: str, dimension: int, seed: int, *,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Materialize the canonical sign vector as exact ``{-1,+1}`` values."""
    packed = _sign_bit_bytes(role, dimension, seed)
    values = [
        -1.0 if ((packed[i // 8] >> (i % 8)) & 1) else 1.0
        for i in range(dimension)
    ]
    return torch.tensor(values, dtype=dtype, device=device)


def online_transform_digest(value: Mapping[str, Any]) -> str:
    """Digest the canonical transform contract, excluding its own digest.

    This binds the block geometry and fixed transform semantics as well as the
    two sign-vector digests.  The containing artifact must still bind its
    config; like every self-carried checksum this detects drift, not a swap of
    a complete mutually-consistent artifact.
    """
    payload = {field: value[field] for field in _ROOT_PAYLOAD_FIELDS}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_side(value: Any, *, role: str, dimension: int,
                   target: str) -> dict[str, Any]:
    where = f"trellis target {target!r} online_transform.{role}"
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(_SIDE_FIELDS - set(value))
    unknown = sorted(set(value) - _SIDE_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"{where} has missing={missing}, unknown={unknown}; schema "
            f"{TRANSFORM_SCHEMA} is fail-closed")
    declared_dimension = _require_plain_int(
        value["dimension"], f"{where}.dimension")
    if declared_dimension != dimension:
        raise ValueError(
            f"{where}.dimension={declared_dimension} does not match the "
            f"trellis {role} dimension {dimension}")
    block_size = _require_plain_int(value["block_size"],
                                    f"{where}.block_size")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(
            f"{where}.block_size must be a positive power of two, got "
            f"{block_size}")
    if dimension % block_size:
        raise ValueError(
            f"{where}.block_size={block_size} must divide dimension="
            f"{dimension}; schema v1 has padding='none'")
    seed = _require_plain_int(value["seed"], f"{where}.seed")
    if not 0 <= seed < (1 << 64):
        raise ValueError(f"{where}.seed must fit uint64, got {seed}")
    if value["sign_generator"] != SIGN_GENERATOR:
        raise ValueError(
            f"{where}.sign_generator must be {SIGN_GENERATOR!r}, got "
            f"{value['sign_generator']!r}")
    digest = value["sign_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or digest != digest.lower()
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(
            f"{where}.sign_sha256 must be 64 lowercase hexadecimal chars")
    actual = seeded_sign_digest(role, dimension, seed)
    if not hmac.compare_digest(digest, actual):
        raise ValueError(
            f"{where}.sign_sha256={digest} does not bind the signs derived "
            f"from seed={seed}; expected {actual}")
    return {
        "dimension": dimension,
        "block_size": block_size,
        "seed": seed,
        "sign_generator": SIGN_GENERATOR,
        "sign_sha256": digest,
    }


def validate_online_transform(value: Any, *, rows: int, columns: int,
                              target: str) -> dict[str, Any]:
    """Validate and normalize the complete v1 transform ABI.

    Unknown fields are rejected.  This matters because accepting a new order,
    padding law, or sign generator while executing v1 arithmetic would serve a
    different matrix without necessarily changing any tensor shape.
    """
    where = f"trellis target {target!r} online_transform"
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(_ROOT_FIELDS - set(value))
    unknown = sorted(set(value) - _ROOT_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"{where} has missing={missing}, unknown={unknown}; schema "
            f"{TRANSFORM_SCHEMA} is fail-closed")
    expected = {
        "schema": TRANSFORM_SCHEMA,
        "algorithm": TRANSFORM_ALGORITHM,
        "normalization": TRANSFORM_NORMALIZATION,
        "padding": TRANSFORM_PADDING,
    }
    for field, wanted in expected.items():
        if value[field] != wanted:
            raise ValueError(
                f"{where}.{field} must be {wanted!r}, got {value[field]!r}")
    normalized = {
        **expected,
        "input": _validate_side(value["input"], role="input",
                                dimension=columns, target=target),
        "output": _validate_side(value["output"], role="output",
                                 dimension=rows, target=target),
    }
    digest = value["transform_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or digest != digest.lower()
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(
            f"{where}.transform_sha256 must be 64 lowercase hexadecimal "
            "chars")
    actual = online_transform_digest(normalized)
    if not hmac.compare_digest(digest, actual):
        raise ValueError(
            f"{where}.transform_sha256={digest} does not bind the declared "
            f"algorithm/normalization/padding/geometry/signs; expected "
            f"{actual}")
    normalized["transform_sha256"] = digest
    return normalized


def _normalized_block_hadamard_rows(x: torch.Tensor,
                                    block_size: int) -> torch.Tensor:
    """Apply block-diagonal, orthonormal Sylvester Hadamards to row vectors."""
    if x.dim() != 2:
        raise ValueError(f"Hadamard input must be 2-D, got {tuple(x.shape)}")
    if (isinstance(block_size, bool) or not isinstance(block_size, int)
            or block_size <= 0 or block_size & (block_size - 1)):
        raise ValueError(
            f"block_size must be a positive power of two, got {block_size!r}")
    dimension = x.shape[1]
    if dimension % block_size:
        raise ValueError(
            f"block_size={block_size} does not divide dimension={dimension}")
    work = x.float().reshape(x.shape[0], dimension // block_size, block_size)
    stride = 1
    while stride < block_size:
        paired = work.reshape(*work.shape[:-1], -1, 2, stride)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        work = torch.stack((left + right, left - right), dim=-2) \
                    .reshape(*work.shape)
        stride *= 2
    return work.mul_(block_size ** -0.5).reshape(x.shape)


def _validate_application(value: torch.Tensor, signs: torch.Tensor, *,
                          block_size: int, role: str,
                          value_name: str) -> None:
    if value.dim() != 2:
        raise ValueError(
            f"{role} Hadamard input must be 2-D, got {tuple(value.shape)}")
    if signs.dim() != 1 or signs.numel() != value.shape[1]:
        raise ValueError(
            f"{role} signs shape {tuple(signs.shape)} does not bind "
            f"{value_name} shape {tuple(value.shape)}")
    if (isinstance(block_size, bool) or not isinstance(block_size, int)
            or block_size <= 0 or block_size & (block_size - 1)):
        raise ValueError(
            f"block_size must be a positive power of two, got {block_size!r}")
    if value.shape[1] % block_size:
        raise ValueError(
            f"block_size={block_size} does not divide "
            f"dimension={value.shape[1]}")
    if value.shape[1] == 0:
        raise ValueError(f"{role} dimension must be positive")


def _use_native_warp128(value: torch.Tensor, block_size: int) -> bool:
    """The exact, deliberately narrow research dispatch cell."""
    return (value.is_cuda and value.dtype == torch.bfloat16
            and block_size == _NATIVE_BLOCK_SIZE)


def apply_input_transform(x: torch.Tensor, signs: torch.Tensor,
                          block_size: int) -> torch.Tensor:
    """Apply ``R_in`` to column activations (row form: ``x D H``)."""
    _validate_application(
        x, signs, block_size=block_size, role="input", value_name="x")
    prepared_signs = signs.to(device=x.device, dtype=x.dtype)
    if _use_native_warp128(x, block_size):
        return _qtip_hadamard_warp128(
            x.contiguous(), prepared_signs.contiguous(), True)
    return _normalized_block_hadamard_rows(
        x * prepared_signs, block_size
    ).to(torch.bfloat16)


def apply_inverse_output_transform(y: torch.Tensor, signs: torch.Tensor,
                                   block_size: int) -> torch.Tensor:
    """Apply ``R_out.T`` to column outputs (row form: ``y H D``)."""
    _validate_application(
        y, signs, block_size=block_size, role="output", value_name="y")
    prepared_signs = signs.to(device=y.device, dtype=y.dtype)
    if _use_native_warp128(y, block_size):
        return _qtip_hadamard_warp128(
            y.contiguous(), prepared_signs.contiguous(), False)
    transformed = _normalized_block_hadamard_rows(y, block_size)
    return (transformed * signs.to(
        device=y.device, dtype=transformed.dtype)).to(torch.bfloat16)
