"""The E2M1 trellis W4A4 dense serving lane.

WHAT THIS CLOSES.  ``trellis_e4m3_lane`` gave the 8-bit family a route into
vLLM.  This is the 4-bit half, and it is the one the low-bpp mandate actually
needs: the E2M1 wire is where the trellis ladder lives below 3 bpw.

IT NEEDS NO BRIDGE EITHER, AND THAT CORRECTS A RECORDED BLOCKER.  The
``route_probe.py`` note (and the docs and memory that inherited it) said the
E2M1 tile "carries the right CODES but the wrong SCALE PLANE", describing a
per-output-row fp32 plane.  **That is the E4M3 family's contract, not E2M1's.**
The E2M1 wire contracts

    W[r][c] = e2m1_value(code[r][c])
              * e4m3fn_value(scale_blob[r*groups + c//16])
              * global_scale_real

-- a group-16 ue4m3 block-scale plane, which is precisely the operand the
Blackwell ``block_scaled_ue4m3xe2m1`` mainloop demands.  Measured, not
asserted (``dq-runs/trellis-kernel-20260829/_q6_swizzle.py``): with varied
scales on BOTH operands, at aligned and unaligned ``N`` and arbitrary ``M``,
decode -> ``_scaled_mm`` reproduces the wire's contracted product with
**max absolute error exactly 0** on 8/8 shapes.  ``decode_native_packed``
already emits two nibbles per byte in the order the fp4 operand wants (low
nibble = even column, checked), so the payload is a pure ``view``.

TWO THINGS THIS OP ACCEPTS AND SILENTLY GETS WRONG.  Both were caught by
comparing numbers, and neither raises:

  1.  An unswizzled scale plane is accepted and silently miscomputes.  The
      plane must be in the cuBLAS 128x4 blocked layout, and that layout is
      NEVER row-major -- the 128x4 tile is swizzled internally, so no shape
      makes the two coincide.  A row-major plane is 67-70% wrong at ALIGNED
      shapes too (``_q9_rowmajor_at_aligned.py``); zero-padding one to a
      legal shape only stops it raising.  ``blocked_scales`` below is the
      real layout and agrees byte-for-byte with torch's own ``to_blocked``.
  2.  ``scale_result=`` is accepted and NOT APPLIED (relative error 1.0).  So
      ``global_scale_real`` is folded into the epilogue multiply instead,
      together with the A-side ``1/input_global_scale`` correction that
      ``nvfp4_activation_contract.reciprocal_vector`` applies on the CB lane.

THE ACTIVATION SIDE IS NEW COST AND IT IS NOT PRICED -- same caveat as the
E4M3 lane, and it bites harder here.  This lane is **W4A4**: ``x`` is
quantized to E2M1 group-16 with a STATIC global scale, the
``e2m1_group16_ue4m3_static`` contract.  Every trellis quality number that
exists is weight-only corpus SSE, which prices W*A16.  A 4-bit activation is a
far larger perturbation than the fp8 one, so **nothing in the menu predicts
this lane's quality.**  It stays opt-in until a served A/B prices it.

PORTABILITY DOES NOT TRANSFER FROM THE E4M3 LANE.  fp4 x fp4 is Blackwell-only.
The AMD / pre-Blackwell portability argument belongs to E4M3 W8A8 (fp8 x fp8 is
what Ada, Hopper and hipBLASLt expose); this lane is sm120/sm121 only.

RESIDENCY IS AN EXPLICIT DECLARED MODE, NOT A HEURISTIC -- as on the E4M3
lane, and measured here rather than inherited (fp4 decode is 204.9 us per
4096x4096 tile on GB10, flat in ``M``; the E4M3 lane's 188 us is a different
family at a different rate).  The two branches differ less than on E4M3
because a resident fp4 tile is only 4.5 bpw, not 8.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from . import trellis_decode_pool as decode_pool
from .lane_select import latched_bool
from .trellis import TCQ_E2M1_R256
from .trellis_ops import prepare_wire_cuda
from .trellis_scheme import parse_wire_for_scheme, validate_trellis_scheme
from .nvfp4_activation_contract import EXECUTION_CONTRACT, GROUP_SIZE, emit_route

__all__ = [
    "TRELLIS_E2M1_FLAG",
    "TRELLIS_E2M1_MODE_ENV",
    "MODE_RESIDENT",
    "MODE_STREAMED",
    "ACTIVATION_CONTRACT",
    "trellis_e2m1_enabled",
    "trellis_e2m1_mode",
    "blocked_scales",
    "build_trellis_e2m1_method",
]

TRELLIS_E2M1_FLAG = "GRIDBOOK_TRELLIS_E2M1"
TRELLIS_E2M1_MODE_ENV = "GRIDBOOK_TRELLIS_E2M1_MODE"

MODE_RESIDENT = "resident"
MODE_STREAMED = "streamed"
#: Said in one place so the mode error and the docs cannot drift apart.
_STREAMED_FOOTPRINT = ("wire bpw per layer plus ONE decode tile shared across all layers on the device")
_MODES = (MODE_RESIDENT, MODE_STREAMED)

#: What this lane executes on the A side, in the vocabulary
#: ``nvfp4_activation_contract`` already owns. Stamped so a served route can be
#: compared against a priced one instead of assumed equal.
ACTIVATION_CONTRACT = EXECUTION_CONTRACT

#: cuBLAS block-scaling tile. Not tunable -- it is the hardware's layout.
_SF_ROW_TILE = 128
_SF_COL_TILE = 4


def trellis_e2m1_enabled() -> bool:
    """The latched OPT-IN flag for this lane."""
    return latched_bool(TRELLIS_E2M1_FLAG, default=False,
                        meaning="the E2M1 trellis W4A4 dense lane")


def trellis_e2m1_mode() -> str:
    """The declared residency mode. No default: an unset flag is an error."""
    raw = (os.environ.get(TRELLIS_E2M1_MODE_ENV) or "").strip()
    if raw not in _MODES:
        raise ValueError(
            f"{TRELLIS_E2M1_MODE_ENV} must be one of {_MODES}, got {raw!r}. "
            "This lane will not choose a residency mode for you: 'resident' "
            "decodes at load (fast forward, ~4.5 bpw in memory) and 'streamed' "
            "decodes every forward (" + _STREAMED_FOOTPRINT + "). The mode "
            "changes the "
            "footprint the artifact actually occupies.")
    return raw


def blocked_scales(plane: torch.Tensor) -> torch.Tensor:
    """Rearrange a ``[rows, groups]`` scale plane into the cuBLAS 128x4 layout.

    This is the layout documented at cuBLAS 3.1.4.3.2 and implemented by
    ``torch.testing._internal.common_quantized.to_blocked`` (not importable
    here -- that module pulls in ``expecttest``), and byte-for-byte equal to it
    on every shape checked. It is NOT optional and NOT a padding: an unswizzled
    plane is accepted by ``_scaled_mm`` and silently miscomputes by 67-70%, at
    aligned shapes as well as unaligned ones. The test suite pins this function
    against the wire's contracted product, which is ground truth rather than
    agreement with another implementation.
    """
    if plane.dim() != 2:
        raise ValueError(f"scale plane must be 2-D, got {tuple(plane.shape)}")
    rows, cols = plane.shape
    n_row_blocks = (rows + _SF_ROW_TILE - 1) // _SF_ROW_TILE
    n_col_blocks = (cols + _SF_COL_TILE - 1) // _SF_COL_TILE
    padded_rows = n_row_blocks * _SF_ROW_TILE
    padded_cols = n_col_blocks * _SF_COL_TILE
    padded = plane
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=plane.device,
                             dtype=plane.dtype)
        padded[:rows, :cols] = plane
    blocks = padded.view(n_row_blocks, _SF_ROW_TILE,
                         n_col_blocks, _SF_COL_TILE).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16) \
                 .flatten()


def build_trellis_e2m1_method(scheme, prefix: str = "<trellis>",
                              mode: str | None = None):
    """Construct the vLLM linear method serving an E2M1 TCQ wire.

    ``scheme`` is the checkpoint's declaration for this target (see
    ``trellis_scheme``), validated here so an unserveable geometry is refused
    at method construction rather than at the first forward.
    """
    if not trellis_e2m1_enabled():
        raise RuntimeError(
            f"the E2M1 trellis W4A4 lane is opt-in and {TRELLIS_E2M1_FLAG} is "
            "not set to 1; refusing the unit (fail-closed) rather than "
            "silently serving it through a bf16 expand, which would be a "
            "different kernel and a different activation contract than the "
            "one the operator selected")
    resolved = mode or trellis_e2m1_mode()
    if resolved not in _MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_trellis_scheme(scheme, prefix)
    if declared["family"] != TCQ_E2M1_R256:
        raise ValueError(
            f"{prefix}: the E2M1 lane serves {TCQ_E2M1_R256}, not "
            f"{declared['family']}")
    rows = declared["rows"]
    columns = declared["columns"]
    wire_bytes = declared["wire_bytes"]
    # ``validate_trellis_scheme`` already refuses a K the group-16 mainloop
    # cannot take; assert the invariant here so the local arithmetic below
    # cannot outlive that check if the schemes module is ever relaxed.
    assert columns % GROUP_SIZE == 0

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    # Imported as MODULES, not symbols: binding the A-side quantizer into this
    # closure would freeze it at build time, so a test could not substitute a
    # reference implementation and a probe could not see which one ran. The A
    # side is the unpriced half of this lane; it must stay inspectable.
    from . import native_cutlass

    groups = columns // GROUP_SIZE

    class TrellisE2M1LinearMethod(LinearMethodBase):
        """W4A4 E2M1 trellis linear (Gridbook-owned lane)."""

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
            # THE WHOLE WIRE, as one opaque blob -- see the E4M3 lane's note:
            # the schedule, the block offsets, the alphabets and the group
            # scale plane live in the header, so a body-shaped parameter
            # cannot reconstruct a wire.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8),
                weight_loader=weight_loader))
            # The A-SIDE static scale. This one is NOT derived, because it is
            # not a wire fact: it is the activation quantizer's global scale,
            # exactly as stock NVFP4 carries ``input_global_scale``. It must be
            # in the checkpoint or the lane cannot quantize x.
            layer.register_parameter("trellis_input_global_scale",
                                     BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32),
                weight_loader=weight_loader))
            layer.trellis_rows = rows
            layer.trellis_columns = columns
            layer.trellis_groups = groups
            layer.trellis_mode = self._mode
            layer.trellis_family = TCQ_E2M1_R256
            layer.gridbook_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the blob, prepare it, block the scale plane, derive gsr.

            This is the load-time seam the lane previously did not have: an
            earlier revision required a caller to have bound
            ``gridbook_trellis_prepared``, ``trellis_global_scale_real`` and
            ``trellis_input_global_scale`` already, and no caller existed. Two
            of those three are wire facts and are now derived here; only the
            A-side scale is loaded, because only it is not in the wire.

            Doing it here rather than at first forward is ``docs/KERNELS.md``
            CUDA-graph safety rule 3.
            """
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            wire = parse_wire_for_scheme(
                blob.contiguous().numpy().tobytes(), scheme, prefix)
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            prepared = prepare_wire_cuda(wire, device=device)
            layer.gridbook_trellis_prepared = prepared
            layer.trellis_row_body_bits = wire.row_body_bits
            gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
            if not gs > 0.0:
                raise ValueError(
                    f"{prefix}: trellis_input_global_scale must be a positive "
                    f"scalar (it divides the activations), got {gs!r}")
            # Derived, never accepted from the loader: the epilogue is a pure
            # function of the wire's own global scale and the A-side scale, so
            # a loader cannot bind one that disagrees with them.
            layer.trellis_epilogue_scale = float(wire.global_scale_real) / gs
            # DERIVED from the wire: the group-16 ue4m3 plane IS scale_b.
            plane = torch.frombuffer(
                bytearray(wire.scale_blob), dtype=torch.uint8).to(
                    device).view(rows, groups).view(torch.float8_e4m3fn)
            layer.register_buffer("scale_b", blocked_scales(plane),
                                  persistent=False)
            if self._mode == MODE_RESIDENT:
                layer.register_buffer(
                    "weight_fp4",
                    prepared.decode_native_packed().view(
                        torch.float4_e2m1fn_x2),
                    persistent=False)
                # Drop the wire: keeping both would make the footprint claim
                # false in the other direction.
                del layer.wire_bytes
                layer.gridbook_trellis_prepared = None
            else:
                # The prepared wire is a private device clone, so the loaded
                # parameter is a second copy of the same bytes. Keeping it
                # would make streamed hold 2x wire + a tile -- more than
                # resident, inverting the mode's whole point.
                del layer.wire_bytes
                # One decode target per device, not per layer: see
                # trellis_decode_pool. Reserved at load so the address is
                # fixed before any capture.
                decode_pool.reserve(layer, prepared.rows,
                                    prepared.output_columns, prepared.device)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # A side: static-global-scale NVFP4. Not a choice -- bf16 x fp4 is
            # refused by _scaled_mm on this hardware, so W4A4 is the only
            # native shape. native_fp4_quant already emits the 128x4 blocked
            # scale-factor layout, so no blocking is needed on this side.
            gs = layer.trellis_input_global_scale.data.reshape(())
            a_q, a_scale = native_cutlass.native_fp4_quant(x2.contiguous(), gs)
            if a_q.dtype == torch.uint8:
                # vLLM's production op returns the packed nibbles as PLAIN
                # uint8; ``_scaled_mm``'s block-scaled fp4 path requires BOTH
                # operands to be ``float4_e2m1fn_x2`` and rejects the pair
                # outright ("Invalid scaling configuration") when only B is.
                # Found by the first container run, 2026-08-29 -- the test
                # stub returned the viewed dtype, so the whole CPU gate was
                # blind to it. A reinterpret, never a copy.
                a_q = a_q.view(torch.float4_e2m1fn_x2)
            if layer.trellis_mode == MODE_RESIDENT:
                b = layer.weight_fp4
            else:
                layer.gridbook_trellis_prepared.decode_native_packed_out(
                    layer.decode_buf)
                b = layer.decode_buf.view(torch.float4_e2m1fn_x2)
            y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale,
                                 scale_b=layer.scale_b,
                                 out_dtype=torch.bfloat16)
            # The two scalars _scaled_mm does not carry, in one multiply:
            # the wire's global_scale_real, and the reciprocal of the A-side
            # static global scale (the same correction the CB lane applies via
            # nvfp4_activation_contract.reciprocal_vector). ``scale_result=``
            # is NOT used: this op accepts it and does not apply it.
            y = y * layer.trellis_epilogue_scale
            # -- K0.4 route telemetry: the record PrismaQuant's
            # -- validate_native_export reads via read_route ----------
            # Every dispatch publishes ONE route record through the
            # existing nvfp4_activation_contract.emit_route surface.
            # No second telemetry channel, no new counter namespace,
            # and read_route's shape is untouched: the consumer is
            # pinned against it. The record must let a priced artifact
            # be compared against what SERVED instead of assumed.
            #
            # WHAT A CONSUMER MAY CONCLUDE. Family (TCQ_E2M1_R256),
            # the executed activation contract
            # (layer.gridbook_activation_contract, which IS
            # ACTIVATION_CONTRACT and therefore EXECUTION_CONTRACT),
            # the residency mode actually taken, and the problem shape
            # (M:N:K) are each first-class.
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
                    policy=f"{TCQ_E2M1_R256}:{layer.trellis_mode}",
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
            return y.reshape(*orig[:-1], rows)

    return TrellisE2M1LinearMethod(resolved)
