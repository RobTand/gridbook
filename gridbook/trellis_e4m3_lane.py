"""The E4M3 trellis W8A8 dense serving lane.

WHAT THIS CLOSES.  Gridbook could decode the TCQ wire but could not *serve* it:
``trellis`` appeared in three modules and none was a ``LinearMethod``, so a
trellis artifact had no route into vLLM at all.  This is that route for the
E4M3 family, and it is deliberately the E4M3 one first: it is a *consumer
swap*, not a new mainloop.

WHY IT NEEDS NO BRIDGE (measured, not asserted --
``dq-runs/trellis-kernel-20260829/e4m3_scaled_mm_identity.py``).  The E4M3 wire
contracts ``W[r][c] = e4m3fn_value(code[r][c]) * row_scale[r]`` with
``global_scale_real`` fixed at 1.0 and one fp32 scale per row.  ``_scaled_mm``
computes ``(A @ B.T) * scale_a[:,None] * scale_b[None,:]``.  The trellis row
scale IS ``scale_b``, the decoded uint8 code plane viewed as ``float8_e4m3fn``
IS ``B``, and ``codes.to(fp32) * row_scale`` equals the wire's contracted
weight under ``torch.equal`` on every shape tested.  So there is no repack, no
re-derivation, and no scale bridge on this lane.

(An earlier revision of this paragraph added "unlike E2M1, whose group-16
block-scale plane still needs one".  That was FALSE and is retracted: the E2M1
wire's group-16 ue4m3 plane is exactly the operand the block-scaled fp4
mainloop wants, and ``trellis_e2m1_lane`` serves it with no bridge either --
it needs only the cuBLAS 128x4 blocking of that plane.  See that module.)

THE ACTIVATION SIDE IS NEW COST AND IT IS NOT PRICED.  This lane is **W8A8**:
``x`` is quantized per token to E4M3 because that is the only shape
``torch._scaled_mm`` will dispatch on sm120/sm121 (bf16xfp8 is refused
outright).  Every trellis quality number that exists -- the 4-bit and 8-bit
ladders, the 24-tensor sweep, the four-family menu -- is weight-only corpus
SSE, which prices W*A16.  **Serving through this lane therefore executes an
activation contract nothing has measured**, and that is exactly the
2026-08-17 NVFP4_CB defect (rendering identity without execution identity).
The lane stamps ``fp8_per_token_dynamic`` so the served contract is legible to
the route probe, and it stays OPT-IN until a served A/B prices the A-side.

RESIDENCY IS AN EXPLICIT DECLARED MODE, NOT A HEURISTIC.  Decode costs a flat
~186 us per 4096x4096 tile on GB10 -- flat in ``M``, because it is per weight,
not per token.  So the choice is real and neither branch dominates
(``e4m3_lane_bench.py`` / ``_q10_aside_cost.py``, M=512 / N=K=4096):

  * ``resident``  decode once at load.  Its GEMM is 94.2 us vs 264.6 us bf16
                  (2.81x) -- but the forward is NOT GEMM-only, and an earlier
                  revision of this paragraph said it was.  The per-token A-side
                  quantize costs 147.3 us, 1.56x the GEMM it feeds and 54% of
                  the arm, so the whole forward is 274.2 us = **0.96x bf16**
                  (``_q10_aside_cost.py``, 9 paired reps).  Resident weight is
                  full fp8 -- 8 bpw, so the wire's compression is a download
                  saving only.
  * ``streamed``  decode every forward.  458 us vs 281 us bf16 = **0.61x** --
                  not the "parity" an A-excluded reading gives -- holding the
                  wire's own bpw per layer (8x smaller at 2.0 bpw) plus ONE
                  decode tile shared across every layer on the device.  That
                  per-layer figure is true only because the lane drops the
                  loaded parameter and pools the scratch; see
                  ``trellis_decode_pool`` for what it was before.

CAVEAT ON THAT 147 us, AND IT CUTS THE OTHER WAY.  The A-side quantizer
measured there is a naive multi-kernel torch reference (amax, divide, clamp,
cast).  vLLM's fused per-token fp8 quant is one kernel over 4 MB and should
cost far less, so 147 us is an UPPER bound on the A-side and 0.96x is a LOWER
bound on the lane.  The lane's real standing is somewhere in **[0.96x, 2.81x]**
and only the container run can place it.  What is settled is that the
GEMM-only ratio is not the lane's speedup.

``streamed`` is the mode the low-bpp mandate needs and ``resident`` is the fast
one; picking between them by peeking at a shape would be principle 1's banned
heuristic, so the operator declares it and it is stamped.  Neither is default:
the flag must be set at all.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from . import trellis_decode_pool as decode_pool
from .lane_select import latched_bool
from .nvfp4_activation_contract import emit_route
from .trellis import TCQ_E4M3_R256, decoded_scales
from .trellis_ops import prepare_wire_cuda
from .trellis_scheme import parse_wire_for_scheme, validate_trellis_scheme

__all__ = [
    "TRELLIS_E4M3_FLAG",
    "TRELLIS_E4M3_MODE_ENV",
    "MODE_RESIDENT",
    "MODE_STREAMED",
    "ACTIVATION_CONTRACT",
    "trellis_e4m3_enabled",
    "trellis_e4m3_mode",
    "build_trellis_e4m3_method",
]

TRELLIS_E4M3_FLAG = "GRIDBOOK_TRELLIS_E4M3"
TRELLIS_E4M3_MODE_ENV = "GRIDBOOK_TRELLIS_E4M3_MODE"

MODE_RESIDENT = "resident"
MODE_STREAMED = "streamed"
#: Said in one place so the mode error and the docs cannot drift apart.
_STREAMED_FOOTPRINT = ("wire bpw per layer plus ONE decode tile shared across all layers on the device")
_MODES = (MODE_RESIDENT, MODE_STREAMED)

#: What this lane actually executes on the A side, in the vocabulary
#: ``nvfp4_activation_contract.ROUTE_CONTRACTS`` already defines. Stamped so a
#: served route can be compared against a priced one instead of assumed equal.
ACTIVATION_CONTRACT = "fp8_per_token_dynamic"


def trellis_e4m3_enabled() -> bool:
    """The latched OPT-IN flag for this lane."""
    return latched_bool(TRELLIS_E4M3_FLAG, default=False,
                        meaning="the E4M3 trellis W8A8 dense lane")


def trellis_e4m3_mode() -> str:
    """The declared residency mode. No default: an unset flag is an error.

    Defaulting would silently pick an artifact's footprint for the operator --
    ``resident`` reports the wire's bpw on disk while occupying 8 bpw in
    memory. That is exactly the kind of invisible discrepancy that makes a
    footprint claim unfalsifiable, so the mode is required and stamped.
    """
    raw = (os.environ.get(TRELLIS_E4M3_MODE_ENV) or "").strip()
    if raw not in _MODES:
        raise ValueError(
            f"{TRELLIS_E4M3_MODE_ENV} must be one of {_MODES}, got {raw!r}. "
            "This lane will not choose a residency mode for you: 'resident' "
            "decodes at load (fast forward, 8 bpw in memory) and 'streamed' "
            "decodes every forward (" + _STREAMED_FOOTPRINT + ", "
            "~bf16 throughput). "
            "The mode changes the footprint the artifact actually occupies.")
    return raw


def build_trellis_e4m3_method(scheme, prefix: str = "<trellis>",
                              mode: str | None = None):
    """Construct the vLLM linear method serving an E4M3 TCQ wire.

    ``scheme`` is the checkpoint's declaration for this target (see
    ``trellis_scheme``), validated here so an unserveable geometry is refused
    at method construction rather than at the first forward.

    Lazy vLLM import, exactly as ``build_mxfp8_dense_method`` does it: policy
    modules and CPU tests import this file with no vLLM present. The returned
    instance's CLASS NAME is the backend label a route probe reads.
    """
    if not trellis_e4m3_enabled():
        raise RuntimeError(
            f"the E4M3 trellis W8A8 lane is opt-in and {TRELLIS_E4M3_FLAG} is "
            "not set to 1; refusing the unit (fail-closed) rather than "
            "silently serving it through a bf16 expand, which would be a "
            "different kernel and a different activation contract than the "
            "one the operator selected")
    resolved = mode or trellis_e4m3_mode()
    if resolved not in _MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_trellis_scheme(scheme, prefix)
    if declared["family"] != TCQ_E4M3_R256:
        raise ValueError(
            f"{prefix}: the E4M3 lane serves {TCQ_E4M3_R256}, not "
            f"{declared['family']}")
    rows = declared["rows"]
    columns = declared["columns"]
    wire_bytes = declared["wire_bytes"]

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    # Imported as a MODULE, not a symbol: binding ``native_fp8_quant`` into
    # this closure would freeze the A-side quantizer at build time, so a test
    # could not substitute a reference implementation and a probe could not
    # see which one ran. The A side is the unpriced half of this lane; it must
    # stay inspectable.
    from . import native_cutlass

    class TrellisE4M3LinearMethod(LinearMethodBase):
        """W8A8 E4M3 trellis linear (Gridbook-owned lane)."""

        def __init__(self, mode: str) -> None:
            self._mode = mode

        # -- load -------------------------------------------------------
        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            if out_size != rows or in_size != columns:
                raise ValueError(
                    f"trellis wire is {rows}x{columns} but the layer wants "
                    f"{out_size}x{in_size}; a wire is bound to its shape")
            weight_loader = extra_weight_attrs.get("weight_loader")
            # THE WHOLE WIRE, as one opaque blob. Not a [rows, row_stride]
            # body: the per-column rate schedule, the tight block offsets, the
            # per-rate alphabets and the scale plane live in the header and
            # exist nowhere else, so a body-shaped parameter cannot reconstruct
            # a wire. ``BasevLLMParameter`` carries no shard dimensions, which
            # is the honest declaration -- a blob has no output axis to split,
            # and dispatch has already refused TP>1 for this lane. A future
            # sharded form needs per-rank wires, not a byte slice.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8),
                weight_loader=weight_loader))
            layer.trellis_rows = rows
            layer.trellis_columns = columns
            layer.trellis_mode = self._mode
            layer.trellis_family = TCQ_E4M3_R256
            layer.gridbook_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the blob, prepare it on device, derive every scale.

            This is the load-time seam the lane previously did not have: an
            earlier revision required a caller to have bound
            ``gridbook_trellis_prepared`` already, and no caller existed. The
            method owns it now, because the method is what declared the
            parameter the wire arrives in.

            Doing it here rather than at first forward is ``docs/KERNELS.md``
            CUDA-graph safety rule 3: no host->device setup inside a capture.
            """
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            wire = parse_wire_for_scheme(
                blob.contiguous().numpy().tobytes(), scheme, prefix)
            # NO gsr CHECK HERE, deliberately. The E4M3 contract pins
            # global_scale_real to 1.0 and puts the whole value in the per-row
            # fp32 plane, which matters because ``decoded_scales`` reads that
            # plane and does NOT apply gsr -- a wire carrying any other value
            # would be served as if it were 1.0. But ``TrellisWire.validate``
            # already refuses that wire, and ``from_bytes`` validates, so a
            # check here could never fire. An unreachable guard that reads
            # like a gate is worse than none: it invites the belief that the
            # lane is what enforces this. The wire format is.
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            prepared = prepare_wire_cuda(wire, device=device)
            layer.gridbook_trellis_prepared = prepared
            layer.trellis_row_body_bits = wire.row_body_bits
            # DERIVED from the wire, never loaded beside it: a separately
            # carried scale plane could disagree with the bytes it scales and
            # nothing would notice.
            scale = torch.tensor(
                [row[0] for row in decoded_scales(wire)],
                dtype=torch.float32, device=device)
            layer.register_buffer("scale_b", scale.view(1, rows).contiguous(),
                                  persistent=False)
            if self._mode == MODE_RESIDENT:
                codes = prepared.decode_native_packed()
                layer.register_buffer(
                    "weight_fp8", codes.view(torch.float8_e4m3fn),
                    persistent=False)
                # Drop the wire: keeping both would occupy MORE than bf16 and
                # make the footprint claim false in the other direction.
                del layer.wire_bytes
                layer.gridbook_trellis_prepared = None
            else:
                # The prepared wire is a private device clone, so the loaded
                # parameter is a second copy of the same bytes. Keeping it
                # would make streamed hold 2x wire + a tile -- more than
                # resident, inverting the mode's whole point.
                del layer.wire_bytes
                # One reusable decode target per device, not per layer (see
                # trellis_decode_pool), reserved at load so the forward never
                # allocates inside a graph capture.
                decode_pool.reserve(layer, prepared.rows,
                                    prepared.output_columns, prepared.device)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # A side: per-token dynamic E4M3. Not a choice -- bf16 x fp8 is
            # refused by _scaled_mm on this hardware, so W8A8 is the only
            # native shape. See the module docstring on why this is unpriced.
            a_q, a_scale = native_cutlass.native_fp8_quant(x2)
            if layer.trellis_mode == MODE_RESIDENT:
                b = layer.weight_fp8
            else:
                prepared = layer.gridbook_trellis_prepared
                prepared.decode_native_packed_out(layer.decode_buf)
                b = layer.decode_buf.view(torch.float8_e4m3fn)
            y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale,
                                 scale_b=layer.scale_b,
                                 out_dtype=torch.bfloat16)
            # -- K0.4 route telemetry: the record PrismaQuant's
            # -- validate_native_export reads via read_route ----------
            # Every dispatch publishes ONE route record through the
            # existing nvfp4_activation_contract.emit_route surface.
            # No second telemetry channel, no new counter namespace,
            # and read_route's shape is untouched: the consumer is
            # pinned against it. The record must let a priced artifact
            # be compared against what SERVED instead of assumed.
            #
            # WHAT A CONSUMER MAY CONCLUDE. Family (TCQ_E4M3_R256),
            # the executed activation contract
            # (layer.gridbook_activation_contract, which IS
            # ACTIVATION_CONTRACT), the residency mode actually taken,
            # and the problem shape (M:N:K) are each first-class.
            #
            # WHAT IT MAY NOT. Regime (decode vs batch) is NOT
            # distinguished by this lane: one torch._scaled_mm kernel
            # serves every M, so the shape's M is the only M-bearing
            # field. The contract publishes separate decode/batch cells;
            # this lane's single kernel satisfies both. See
            # WO-E1-FINDINGS.md and docs/TRELLIS-R256-RESEARCH.md.
            # Telemetry never breaks the request.
            try:
                _m = int(x2.shape[0])
                _rows = int(layer.trellis_rows)
                _cols = int(layer.trellis_columns)
                emit_route(
                    layer,
                    kind="dense",
                    policy=f"{TCQ_E4M3_R256}:{layer.trellis_mode}",
                    symbol="torch._scaled_mm",
                    tile_m=0,
                    shape=f"M{_m}:N{_rows}:K{_cols}",
                    contract=layer.gridbook_activation_contract,
                    state="served",
                    reason=None,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], y.shape[-1])

    return TrellisE4M3LinearMethod(resolved)
