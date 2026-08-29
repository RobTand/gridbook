"""Explicit low-level ops for the research-only TCQ R256 wire.

Importing this module registers the isolated research CUDA ops.  It does not register a
quantization config, producer, chooser, or runtime-contract format.  Expansion
is an explicit transient test operation and must not be retained as model
state (INV-1).  The direct GEMV is a decode rung, not an INV-2-compliant
prefill implementation.  The prepared native-code path validates once, owns
private immutable tensors, and writes nibble-packed E2M1 or byte-packed E4M3
into a caller-provided buffer without a per-call host synchronization.
"""
from __future__ import annotations

import torch

from .trellis import (
    TCQ_E2M1_R256,
    TrellisWire,
    alphabet_lut,
    decoded_scales,
    derived_block_plan,
    derived_decode_plan,
)


def wire_cuda_tensors(wire: TrellisWire, device=None):
    """Materialize the small derived decode plan and packed wire tensors.

    ``column_offsets`` and ``previous_u_offsets`` are deterministic workspace,
    shared by all rows.  They are not serialized artifact side information.
    """
    wire.validate()
    device = torch.device("cuda" if device is None else device)
    column_offsets, previous_offsets = derived_decode_plan(wire)
    schedule = torch.tensor(
        wire.expanded_schedule, dtype=torch.uint8, device=device)
    payload = torch.tensor(
        list(wire.payload), dtype=torch.uint8, device=device).reshape(
            wire.rows, wire.row_stride_bytes)
    columns = torch.tensor(column_offsets, dtype=torch.int64, device=device)
    previous = torch.tensor(
        previous_offsets, dtype=torch.int64, device=device)
    lut = torch.tensor(
        alphabet_lut(wire), dtype=torch.uint8, device=device)
    scales = torch.tensor(
        decoded_scales(wire), dtype=torch.float32, device=device)
    family = 1 if wire.family == TCQ_E2M1_R256 else 2
    return payload, schedule, columns, previous, lut, scales, family


def wire_cuda_block_plan(wire: TrellisWire, device=None):
    """Materialize the compact per-column / per-block plan the v2 kernels stage.

    Like ``wire_cuda_tensors`` this is deterministic derived workspace rebuilt
    from the wire, shared by all rows, and never artifact side information.
    """
    device = torch.device("cuda" if device is None else device)
    rate, offset, ordinal, meta = derived_block_plan(wire)
    return (
        torch.tensor(rate, dtype=torch.uint8, device=device),
        torch.tensor(offset, dtype=torch.int32, device=device),
        torch.tensor(ordinal, dtype=torch.int16, device=device),
        torch.tensor(meta, dtype=torch.int32, device=device),
    )


@torch.library.custom_op("prismaquant::trellis_r256_validate_wire",
                         mutates_args=())
def _trellis_r256_validate_wire(
    payload: torch.Tensor, schedule: torch.Tensor,
    column_offsets: torch.Tensor, previous_u_offsets: torch.Tensor,
    col_rate: torch.Tensor, col_bit_offset: torch.Tensor,
    col_ordinal: torch.Tensor, block_meta: torch.Tensor,
    alphabet_lut_tensor: torch.Tensor, scales: torch.Tensor,
    rows: int, columns: int, row_stride: int, family: int,
) -> torch.Tensor:
    """Synchronizing load-time validation; never call from a hot path.

    Validates the scan-free tables against the payload and then the compact
    block plan against those tables, so the v2 decode path can never reach a
    bit address the scan-free validation has not already proven.
    """
    from .cuda_ext import require_trellis_r256_ext
    return require_trellis_r256_ext(
        "TCQ R256 load-time wire validation").trellis_r256_validate_wire(
            payload, schedule, column_offsets, previous_u_offsets,
            col_rate, col_bit_offset, col_ordinal, block_meta,
            alphabet_lut_tensor, scales, rows, columns, row_stride, family)


@_trellis_r256_validate_wire.register_fake
def _trellis_r256_validate_wire_fake(
    payload, schedule, column_offsets, previous_u_offsets,
    col_rate, col_bit_offset, col_ordinal, block_meta,
    alphabet_lut_tensor, scales, rows, columns, row_stride, family,
):
    return torch.empty((1,), dtype=torch.int64, device=payload.device)


@torch.library.custom_op(
    "prismaquant::trellis_r256_decode_native_packed_prevalidated_out",
    mutates_args=("output",),
)
def _trellis_r256_decode_native_packed_prevalidated_out(
    payload: torch.Tensor, col_rate: torch.Tensor,
    col_bit_offset: torch.Tensor, col_ordinal: torch.Tensor,
    block_meta: torch.Tensor, alphabet_lut_tensor: torch.Tensor,
    output: torch.Tensor,
    rows: int, columns: int, row_stride: int, family: int,
) -> None:
    from .cuda_ext import require_trellis_r256_ext
    require_trellis_r256_ext(
        "TCQ R256 prevalidated native packed-code decode",
    ).trellis_r256_decode_native_packed_prevalidated_out(
        payload, col_rate, col_bit_offset, col_ordinal, block_meta,
        alphabet_lut_tensor, output, rows, columns, row_stride, family)


@_trellis_r256_decode_native_packed_prevalidated_out.register_fake
def _trellis_r256_decode_native_packed_prevalidated_out_fake(
    payload, col_rate, col_bit_offset, col_ordinal, block_meta,
    alphabet_lut_tensor, output, rows, columns, row_stride, family,
):
    return None


def _tensor_fingerprint(tensor: torch.Tensor):
    return (
        tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, tensor.device.type, tensor.device.index, tensor._version,
    )


class PreparedTrellisWireCuda:
    """Private, once-validated CUDA owner for the no-sync packed-code path.

    The owner clones every input before device validation and exposes no tensor
    getters.  Its per-call identity/version check rejects mutation or storage
    replacement before the unchecked hot-path symbol is reached.  The raw
    pybind symbol remains an internal research ABI, not artifact authority.
    """

    __slots__ = (
        "__tensors", "__fingerprints", "__contract_fingerprint",
        "__rows", "__columns", "__row_stride", "__row_body_bits",
        "__family", "__output_columns", "__device",
    )

    def __init__(
        self, tensors, *, rows, columns, row_stride, row_body_bits, family,
    ):
        self.__tensors = tuple(tensors)
        self.__fingerprints = tuple(
            _tensor_fingerprint(tensor) for tensor in self.__tensors)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__row_stride = int(row_stride)
        self.__row_body_bits = int(row_body_bits)
        self.__family = int(family)
        self.__output_columns = (
            (self.__columns + 1) // 2
            if self.__family == 1 else self.__columns)
        self.__device = self.__tensors[0].device
        self.__contract_fingerprint = self._contract_values()

    def _contract_values(self):
        return (
            self.__rows, self.__columns, self.__row_stride,
            self.__row_body_bits, self.__family, self.__output_columns,
            self.__device,
        )

    @property
    def rows(self):
        return self.__rows

    @property
    def columns(self):
        return self.__columns

    @property
    def row_stride(self):
        return self.__row_stride

    @property
    def row_body_bits(self):
        return self.__row_body_bits

    @property
    def family(self):
        return self.__family

    @property
    def output_columns(self):
        return self.__output_columns

    @property
    def device(self):
        return self.__device

    def _require_unchanged(self):
        observed = tuple(
            _tensor_fingerprint(tensor) for tensor in self.__tensors)
        if (
            observed != self.__fingerprints
            or self._contract_values() != self.__contract_fingerprint
        ):
            raise RuntimeError(
                "prepared TCQ CUDA wire contract changed after validation")

    def empty_native_packed(self) -> torch.Tensor:
        return torch.empty(
            (self.rows, self.output_columns), dtype=torch.uint8,
            device=self.device)

    def decode_native_packed_out(self, output: torch.Tensor) -> None:
        self._require_unchanged()
        if (
            output.dtype != torch.uint8
            or output.device != self.device
            or not output.is_contiguous()
            or tuple(output.shape) != (self.rows, self.output_columns)
        ):
            raise ValueError(
                "native packed output must be contiguous uint8 with shape "
                f"[{self.rows},{self.output_columns}] on {self.device}")
        (payload, _schedule, _columns, _previous, rate, offset, ordinal,
         meta, lut, _scales) = self.__tensors
        _trellis_r256_decode_native_packed_prevalidated_out(
            payload, rate, offset, ordinal, meta, lut, output, self.rows,
            self.columns, self.row_stride, self.family)

    def decode_native_packed(self) -> torch.Tensor:
        output = self.empty_native_packed()
        self.decode_native_packed_out(output)
        return output


def prepare_wire_cuda(
    wire: TrellisWire, device=None,
) -> PreparedTrellisWireCuda:
    """Clone and device-validate a wire once for the no-sync hot path."""
    materialized = wire_cuda_tensors(wire, device)
    payload, schedule, columns, previous, lut, scales, family = materialized
    rate, offset, ordinal, meta = wire_cuda_block_plan(wire, device)
    private = tuple(
        tensor.clone() for tensor in
        (payload, schedule, columns, previous, rate, offset, ordinal, meta,
         lut, scales)
    )
    observed_body_bits = _trellis_r256_validate_wire(
        *private, wire.rows, wire.columns, wire.row_stride_bytes, family)
    body_bits = int(observed_body_bits.item())
    if body_bits != wire.row_body_bits:
        raise RuntimeError(
            "device wire validation returned a different row-body length: "
            f"{body_bits} != {wire.row_body_bits}")
    return PreparedTrellisWireCuda(
        private, rows=wire.rows, columns=wire.columns,
        row_stride=wire.row_stride_bytes, row_body_bits=body_bits,
        family=family)


@torch.library.custom_op("prismaquant::trellis_r256_decode_codes",
                         mutates_args=())
def trellis_r256_decode_codes(
    payload: torch.Tensor, schedule: torch.Tensor,
    column_offsets: torch.Tensor, previous_u_offsets: torch.Tensor,
    alphabet_lut_tensor: torch.Tensor, rows: int, columns: int,
    row_stride: int, family: int,
) -> torch.Tensor:
    from .cuda_ext import require_trellis_r256_ext
    return require_trellis_r256_ext(
        "TCQ R256 native-code decode").trellis_r256_decode_codes(
            payload, schedule, column_offsets, previous_u_offsets,
            alphabet_lut_tensor, rows, columns, row_stride, family)


@trellis_r256_decode_codes.register_fake
def _trellis_r256_decode_codes_fake(
    payload, schedule, column_offsets, previous_u_offsets,
    alphabet_lut_tensor, rows, columns, row_stride, family,
):
    return torch.empty((rows, columns), dtype=torch.uint8,
                       device=payload.device)


@torch.library.custom_op("prismaquant::trellis_r256_expand", mutates_args=())
def trellis_r256_expand(
    payload: torch.Tensor, schedule: torch.Tensor,
    column_offsets: torch.Tensor, previous_u_offsets: torch.Tensor,
    col_rate: torch.Tensor, col_bit_offset: torch.Tensor,
    col_ordinal: torch.Tensor, block_meta: torch.Tensor,
    alphabet_lut_tensor: torch.Tensor, scales: torch.Tensor,
    rows: int, columns: int, row_stride: int, family: int,
) -> torch.Tensor:
    """Explicit transient correctness expansion; never resident model state."""
    from .cuda_ext import require_trellis_r256_ext
    return require_trellis_r256_ext(
        "TCQ R256 transient expansion").trellis_r256_expand(
            payload, schedule, column_offsets, previous_u_offsets,
            col_rate, col_bit_offset, col_ordinal, block_meta,
            alphabet_lut_tensor, scales, rows, columns, row_stride, family)


@trellis_r256_expand.register_fake
def _trellis_r256_expand_fake(
    payload, schedule, column_offsets, previous_u_offsets,
    col_rate, col_bit_offset, col_ordinal, block_meta,
    alphabet_lut_tensor, scales, rows, columns, row_stride, family,
):
    return torch.empty((rows, columns), dtype=torch.float32,
                       device=payload.device)


@torch.library.custom_op("prismaquant::trellis_r256_dequant_gemv",
                         mutates_args=())
def trellis_r256_dequant_gemv(
    x: torch.Tensor, payload: torch.Tensor, schedule: torch.Tensor,
    column_offsets: torch.Tensor, previous_u_offsets: torch.Tensor,
    alphabet_lut_tensor: torch.Tensor, scales: torch.Tensor,
    rows: int, columns: int, row_stride: int, family: int,
) -> torch.Tensor:
    """Direct research decode/dequant GEMV with no expanded weight matrix."""
    from .cuda_ext import require_trellis_r256_ext
    return require_trellis_r256_ext(
        "TCQ R256 dequant GEMV").trellis_r256_dequant_gemv(
            x, payload, schedule, column_offsets, previous_u_offsets,
            alphabet_lut_tensor, scales, rows, columns, row_stride, family)


@trellis_r256_dequant_gemv.register_fake
def _trellis_r256_dequant_gemv_fake(
    x, payload, schedule, column_offsets, previous_u_offsets,
    alphabet_lut_tensor, scales, rows, columns, row_stride, family,
):
    return torch.empty((x.shape[0], rows), dtype=torch.float32,
                       device=x.device)


def decode_wire_codes_cuda(wire: TrellisWire, device=None) -> torch.Tensor:
    values = wire_cuda_tensors(wire, device)
    payload, schedule, columns, previous, lut, _scales, family = values
    return trellis_r256_decode_codes(
        payload, schedule, columns, previous, lut, wire.rows, wire.columns,
        wire.row_stride_bytes, family)


def expand_wire_cuda(wire: TrellisWire, device=None) -> torch.Tensor:
    values = wire_cuda_tensors(wire, device)
    payload, schedule, columns, previous, lut, scales, family = values
    rate, offset, ordinal, meta = wire_cuda_block_plan(wire, device)
    return trellis_r256_expand(
        payload, schedule, columns, previous, rate, offset, ordinal, meta,
        lut, scales, wire.rows, wire.columns, wire.row_stride_bytes, family)


def gemv_wire_cuda(x: torch.Tensor, wire: TrellisWire) -> torch.Tensor:
    values = wire_cuda_tensors(wire, x.device)
    payload, schedule, columns, previous, lut, scales, family = values
    return trellis_r256_dequant_gemv(
        x.contiguous().to(torch.float32), payload, schedule, columns, previous,
        lut, scales, wire.rows, wire.columns, wire.row_stride_bytes, family)


__all__ = [
    "PreparedTrellisWireCuda", "decode_wire_codes_cuda", "expand_wire_cuda",
    "gemv_wire_cuda", "prepare_wire_cuda",
    "trellis_r256_decode_codes", "trellis_r256_dequant_gemv",
    "trellis_r256_expand", "wire_cuda_tensors",
]
