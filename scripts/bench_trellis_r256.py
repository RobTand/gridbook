#!/usr/bin/env python3
"""Synthetic bandwidth probe for the research TCQ R256 CUDA ABI.

No output artifact is written.  Set ``PRISMAQUANT_CB_EXT_DIR`` and ``TMPDIR``
to persistent directories under ``/home/rob`` before invoking this script.
The numbers are kernel-prototype diagnostics, never serving qualification.

The prepared TCQ native-code expander is compared on the same
``[rows, columns]`` shape to the existing Gridbook codebook expand lane at the
closest currently declared reader-compatible rung by side-inclusive stored
tensor bytes:

* ``TCQ_E2M1_R256`` versus NVFP4-CB v2 ``expand_fp4_v2_to_weight``;
* ``TCQ_E4M3_R256`` versus FP8-CB ``cb_expand_fp8``.

``stored_decimal_gb_per_s`` is a normalization by serialized tensor payload;
it is not a claim about physical HBM traffic. E2M1 TCQ emits packed nibbles
while the available NVFP4-CB comparison emits BF16, so no E2 speedup is
reported. E4M3 TCQ and FP8-CB both emit raw code bytes with separate row
scales, but the result is still a kernel prototype rather than serving parity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import time

import torch

from gridbook import codec, trellis
from gridbook.expand import expand_fp4_v2_to_weight
from gridbook.ops import cb_expand_fp8
from gridbook.runtime_contract import load_runtime_contract
from gridbook.trellis_ops import (
    prepare_wire_cuda,
    trellis_r256_decode_codes,
    trellis_r256_dequant_gemv,
    wire_cuda_tensors,
)


DEFAULT_Q256 = {
    trellis.TCQ_E2M1_R256: 512,
    trellis.TCQ_E4M3_R256: 1152,
}
CB_FAMILY = {
    trellis.TCQ_E2M1_R256: "NVFP4_CB_K",
    trellis.TCQ_E4M3_R256: "FP8_CB_K",
}


def cb_reader_rungs(family):
    """Return reader-compatible CB rungs for a historical baseline."""
    cb_family = CB_FAMILY[family]
    row = next(
        item for item in load_runtime_contract()["formats"]
        if item["family"] == cb_family
    )
    return tuple(int(k) for k in row["rungs"])


def cb_storage_accounting(family, k_bits, rows, columns):
    """Exact serialized tensor-payload accounting for one CB layer.

    Container framing and JSON metadata are excluded on both sides.  One
    complete fp16 product codebook is conservatively charged to this layer;
    real artifacts can share that table across many layers.  Runtime-only row
    padding, derived row offsets, repacked LUTs, and the NVFP4 compose table are
    reported separately by the GPU fixture and are not serialized bytes.
    """
    rows, columns, k_bits = int(rows), int(columns), int(k_bits)
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    if columns <= 0 or columns % codec.SUPERBLOCK:
        raise ValueError(
            "the existing CB expand ABI requires columns to be a positive "
            f"multiple of {codec.SUPERBLOCK}, got {columns}")
    if k_bits not in cb_reader_rungs(family):
        raise ValueError(
            f"K{k_bits} is not a reader-compatible {CB_FAMILY[family]} rung")

    n_sub = 2 if family == trellis.TCQ_E2M1_R256 else 4
    scale_plane_bytes = 9 if family == trellis.TCQ_E2M1_R256 else 0
    type_size = 4 * k_bits + scale_plane_bytes
    superblocks = columns // codec.SUPERBLOCK
    qweight_bytes = rows * superblocks * type_size
    fp8_row_scale_bytes = (
        rows * 4 if family == trellis.TCQ_E4M3_R256 else 0)
    codebook_elements = sum(
        entries * width
        for entries, width in codec.product_subtable_shapes(k_bits, n_sub))
    codebook_bytes = codebook_elements * 2  # normative fp16 sidecar
    total_bytes = qweight_bytes + fp8_row_scale_bytes + codebook_bytes
    weights = rows * columns
    return {
        "scope": (
            "tensor payloads only; one complete shared fp16 product codebook "
            "is fully charged; safetensors/config framing excluded"
        ),
        "family": CB_FAMILY[family],
        "rung": f"{CB_FAMILY[family]}{k_bits}",
        "k_bits_per_vec8": k_bits,
        "n_sub": n_sub,
        "type_size_bytes_per_256": type_size,
        "superblocks_per_row": superblocks,
        "qweight_bytes": qweight_bytes,
        "fp8_row_scale_bytes": fp8_row_scale_bytes,
        "shared_codebook_fp16_bytes_full_charge": codebook_bytes,
        "total_bytes": total_bytes,
        "total_stored_bits": total_bytes * 8,
        "exact_bpw": total_bytes * 8.0 / weights,
    }


def select_matched_cb_rung(family, target_total_bytes, rows, columns):
    """Nearest reader-compatible rung by exact side-inclusive stored bytes."""
    candidates = [
        cb_storage_accounting(family, k, rows, columns)
        for k in cb_reader_rungs(family)
    ]
    return min(
        candidates,
        key=lambda item: (
            abs(item["total_bytes"] - int(target_total_bytes)),
            item["k_bits_per_vec8"],
        ),
    )


def stored_decimal_gb_per_s(stored_bytes, elapsed_ms):
    """Normalized serialized-byte rate using decimal GB, not GiB."""
    if elapsed_ms <= 0:
        raise ValueError(f"elapsed_ms must be positive, got {elapsed_ms}")
    return float(stored_bytes) / (float(elapsed_ms) / 1000.0) / 1.0e9


def _alphabet(family, rate):
    full = trellis.canonical_full_alphabet(family)
    count = 1 << (rate + 1)
    if count == len(full):
        return full
    return tuple(full[i * (len(full) - 1) // (count - 1)]
                 for i in range(count))


def synthetic_importance_schedule(family, q256, columns, seed):
    """Deterministic mixed-rate fixed-quota schedule for the bandwidth probe."""
    terminal = trellis.native_bits(family)
    rng = random.Random(seed)
    schedule = []
    for block_start in range(0, columns, 256):
        width = min(256, columns - block_start)
        if width != 256:
            raise ValueError("the matched-CB probe requires complete blocks")
        base, remainder = divmod(q256, 256)
        if remainder:
            low, high, high_count = base, base + 1, remainder
        elif 1 < base < terminal:
            low, high, high_count = base - 1, base + 1, 128
        else:
            low, high, high_count = base, base, 0
        if not 1 <= low <= high <= terminal:
            raise ValueError("q256 cannot form a legal mixed-rate schedule")
        importance = [(rng.random(), column) for column in range(width)]
        importance.sort()
        block = [low] * width
        for _weight, column in importance[width - high_count:]:
            block[column] = high
        if sum(block) != q256:
            raise AssertionError((sum(block), q256))
        schedule.extend(block)
    return tuple(schedule)


def synthetic_wire(family, q256, rows, columns, seed):
    expanded = synthetic_importance_schedule(
        family, q256, columns, seed + 17)
    terminal = trellis.native_bits(family)
    alphabets = {rate: _alphabet(family, rate)
                 for rate in sorted({r for r in expanded if r < terminal})}
    rng = random.Random(seed)
    u = [[rng.getrandbits(1) for _ in range(columns)] for _ in range(rows)]
    point = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    finite = [code for code in range(1 << terminal)
              if family != trellis.TCQ_E4M3_R256 or code not in (0x7f, 0xff)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                bypass[row][column] = rng.choice(finite)
            else:
                point[row][column] = rng.randrange(1 << (rate - 1))
    if family == trellis.TCQ_E2M1_R256:
        scales = bytes([0x38] * (rows * ((columns + 15) // 16)))
    else:
        scales = struct.pack(f"<{rows}f", *([1.0] * rows))
    return trellis.pack_planes(
        family=family, body_rate_q256=q256, schedule=expanded,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
        u_bits=u, point_indices=point, bypass_codes=bypass,
        alphabets=alphabets, scale_blob=scales)


def _tensor_bytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def synthetic_cb_expand(family, accounting, rows, columns, seed):
    """Build the existing CB lane's legal synthetic expand inputs.

    Returns ``(callable, runtime_accounting, semantic_contract)``.  The
    callable deliberately uses the same public wrapper/custom op as serving's
    transient bridge, not a private CUDA binding or a Python decoder.
    """
    k_bits = int(accounting["k_bits_per_vec8"])
    type_size = int(accounting["type_size_bytes_per_256"])
    superblocks = columns // codec.SUPERBLOCK
    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_offsets = torch.zeros(rows, dtype=torch.int32, device="cuda")

    if family == trellis.TCQ_E2M1_R256:
        # Legal v2 scale plane: E8M0 super exponent 127 and sixteen packed
        # sub-codes of zero.  Randomizing all nine bytes would generate scale
        # combinations a conforming producer never emits.
        index = torch.randint(
            0, 256, (rows, superblocks, 4 * k_bits),
            dtype=torch.uint8, generator=generator)
        scale = torch.zeros(
            (rows, superblocks, 9), dtype=torch.uint8)
        scale[:, :, 0] = 127
        packed = torch.cat((index, scale), dim=2).reshape(
            rows, superblocks * type_size).to("cuda").contiguous()
        padded = codec.pad_qweight(packed)

        e2m1 = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            dtype=torch.bfloat16, device="cuda")
        tables = []
        for entries, width in codec.product_subtable_shapes(k_bits, 2):
            count = entries * width
            indices = torch.arange(count, device="cuda") % e2m1.numel()
            tables.append(e2m1[indices].reshape(entries, width).contiguous())
        cb_flat = codec.build_flat_product_codebook(
            tables, k_bits, 2, prefix="TCQ matched NVFP4-CB baseline",
            grid="fp4")
        compose = codec.build_compose_table(
            codec.TWO_TIER_SUB_TABLE).to("cuda")

        def expand():
            return expand_fp4_v2_to_weight(
                padded, cb_flat, row_offsets, compose, rows, columns,
                k_bits, 2, type_size)

        runtime = {
            "padded_qweight_bytes": _tensor_bytes(padded),
            "per_call_compacted_qweight_bytes": accounting["qweight_bytes"],
            "row_offsets_derived_bytes": _tensor_bytes(row_offsets),
            "repacked_codebook_bytes": _tensor_bytes(cb_flat),
            "compose_table_derived_bytes": _tensor_bytes(compose),
            "separate_row_scale_bytes": 0,
        }
        semantic = {
            "api": "gridbook.expand.expand_fp4_v2_to_weight",
            "output_dtype": "torch.bfloat16",
            "output_bytes_per_weight": 2,
            "scale_application": "two-tier v2 scales composed in expand",
        }
        return expand, runtime, semantic

    packed = torch.randint(
        0, 256, (rows, superblocks * type_size),
        dtype=torch.uint8, generator=generator).to("cuda").contiguous()
    padded = codec.pad_qweight(packed)
    # 0x38 is a finite E4M3FN code.  Values do not affect index-decode traffic;
    # using one legal code keeps the synthetic fixture out of NaN space.
    runtime_lut_bytes = accounting[
        "shared_codebook_fp16_bytes_full_charge"] // 2
    cb_flat_fp8 = torch.full(
        (runtime_lut_bytes,), 0x38, dtype=torch.uint8, device="cuda")

    def expand():
        return cb_expand_fp8(
            padded, cb_flat_fp8, row_offsets, rows, columns, k_bits, 4,
            type_size)

    runtime = {
        "padded_qweight_bytes": _tensor_bytes(padded),
        "per_call_compacted_qweight_bytes": 0,
        "row_offsets_derived_bytes": _tensor_bytes(row_offsets),
        "repacked_codebook_bytes": _tensor_bytes(cb_flat_fp8),
        "compose_table_derived_bytes": 0,
        # This scale is serialized and charged above, but raw cb_expand_fp8
        # does not consume it.  The following CUTLASS scaled GEMM does.
        "separate_row_scale_bytes": rows * 4,
    }
    semantic = {
        "api": "gridbook.ops.cb_expand_fp8",
        "output_dtype": "torch.float8_e4m3fn",
        "output_bytes_per_weight": 1,
        "scale_application": (
            "raw E4M3 codes only; separate FP32 row scale is applied by the "
            "following scaled GEMM, outside this expand timing"
        ),
    }
    return expand, runtime, semantic


def timed_ms(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    samples.sort()
    return samples[len(samples) // 2]


def main():
    process_start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("e2m1", "e4m3"), default="e2m1")
    parser.add_argument(
        "--q256", type=int, default=None,
        help="TCQ body bits per 256 weights (default: E2M1=512, E4M3=1152)")
    parser.add_argument(
        "--cb-k", type=int, default=None,
        help="explicit CB comparator rung; default chooses the current "
             "reader-compatible rung nearest in exact stored bytes")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if not os.environ.get("TMPDIR", "").startswith("/home/rob/"):
        raise SystemExit("TMPDIR must be pinned below /home/rob; /tmp is forbidden")
    if not os.environ.get("PRISMAQUANT_CB_EXT_DIR", "").startswith("/home/rob/"):
        raise SystemExit("PRISMAQUANT_CB_EXT_DIR must be below /home/rob")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    family = (trellis.TCQ_E2M1_R256 if args.family == "e2m1"
              else trellis.TCQ_E4M3_R256)
    q256 = DEFAULT_Q256[family] if args.q256 is None else args.q256
    if args.columns <= 0 or args.columns % codec.SUPERBLOCK:
        raise SystemExit(
            "matched-CB measurement requires --columns to be a positive "
            f"multiple of {codec.SUPERBLOCK}; the existing CB expand ABI "
            "cannot run a genuine same-shape short-tail baseline")
    build_start = time.monotonic()
    wire = synthetic_wire(
        family, q256, args.rows, args.columns, args.seed)
    trellis_accounting = trellis.account(wire).__dict__
    if args.cb_k is None:
        cb_accounting = select_matched_cb_rung(
            family, trellis_accounting["total_bytes"], args.rows,
            args.columns)
    else:
        cb_accounting = cb_storage_accounting(
            family, args.cb_k, args.rows, args.columns)
    cb_expand, cb_runtime, cb_semantic = synthetic_cb_expand(
        family, cb_accounting, args.rows, args.columns, args.seed + 1)
    validation_start = time.monotonic()
    prepared = prepare_wire_cuda(wire)
    load_time_validation_ms = (
        time.monotonic() - validation_start) * 1000.0
    native_packed_output = prepared.empty_native_packed()
    payload, schedule, columns, previous, lut, scales, family_code = (
        wire_cuda_tensors(wire))
    x = torch.randn((args.batch, args.columns), dtype=torch.float32,
                    device="cuda")
    common = (payload, schedule, columns, previous, lut)
    trellis_runtime = {
        "payload_bytes": _tensor_bytes(payload),
        "expanded_schedule_derived_bytes": _tensor_bytes(schedule),
        "column_offsets_derived_bytes": _tensor_bytes(columns),
        "previous_u_offsets_derived_bytes": _tensor_bytes(previous),
        "alphabet_lut_bytes": _tensor_bytes(lut),
        "decoded_scales_bytes": _tensor_bytes(scales),
        "native_output_bytes_per_weight": (
            0.5 if family == trellis.TCQ_E2M1_R256 else 1),
    }
    safe_debug_decode_ms = timed_ms(
        lambda: trellis_r256_decode_codes(
            *common, args.rows, args.columns, wire.row_stride_bytes,
            family_code), args.warmup, args.iterations)
    native_packed_ms = timed_ms(
        lambda: prepared.decode_native_packed_out(native_packed_output),
        args.warmup, args.iterations)
    cb_expand_ms = timed_ms(
        cb_expand, args.warmup, args.iterations)
    gemv_ms = timed_ms(
        lambda: trellis_r256_dequant_gemv(
            x, *common, scales, args.rows, args.columns,
            wire.row_stride_bytes, family_code),
        args.warmup, args.iterations)
    packed_bytes = len(wire.payload)
    weights = args.rows * args.columns
    setup_seconds = time.monotonic() - build_start
    source_root = Path(__file__).resolve().parents[1]
    source_paths = (
        source_root / "gridbook" / "trellis.py",
        source_root / "gridbook" / "trellis_ops.py",
        source_root / "gridbook" / "csrc" / "trellis_r256.cu",
        Path(__file__).resolve(),
    )
    source_sha256 = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    schedule_histogram = {
        str(rate): wire.expanded_schedule.count(rate)
        for rate in sorted(set(wire.expanded_schedule))
    }
    comparable = family == trellis.TCQ_E4M3_R256
    result = {
        "schema": "gridbook.trellis-r256-kernel-bench.v3",
        "status": "research_only_not_qualified",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution": {
            "physical_host": os.environ.get("STAGE8_PHYSICAL_HOST"),
            "systemd_invocation_id": os.environ.get("INVOCATION_ID"),
            "container_image": os.environ.get("STAGE8_CONTAINER_IMAGE"),
            "source_manifest_sha256": os.environ.get(
                "STAGE8_SOURCE_MANIFEST_SHA256"),
            "source_sha256": source_sha256,
        },
        "device": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "parameters": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "wire_schema": trellis.SCHEMA,
        "family": family,
        "rung": wire.rung,
        "shape": [args.rows, args.columns],
        "batch": args.batch,
        "schedule": {
            "placement": "deterministic synthetic importance ordering",
            "mixed_rate": len(schedule_histogram) > 1,
            "histogram": schedule_histogram,
            "fixed_quota_per_complete_256": True,
        },
        "accounting": trellis_accounting,
        "load_time_validation_ms": load_time_validation_ms,
        "safe_debug_decode": {
            "median_ms": safe_debug_decode_ms,
            "million_weights_per_s": weights / safe_debug_decode_ms / 1000.0,
            "packed_payload_gib_per_s": packed_bytes / (safe_debug_decode_ms / 1000.0) /
                                          (1 << 30),
            "includes_per_call_device_validation_and_host_sync": True,
        },
        "native_code_expand": {
            "semantics": {
                "api": "PreparedTrellisWireCuda.decode_native_packed_out",
                "output_dtype": "torch.uint8",
                "output_layout": (
                    "two E2M1 codes per byte, even column in low nibble"
                    if family == trellis.TCQ_E2M1_R256
                    else "one finite E4M3 code per byte"
                ),
                "scale_application": "native scale plane remains separate",
                "preallocated_output": True,
                "per_call_host_synchronization": False,
            },
            "runtime_inputs": trellis_runtime,
            "native_output_bytes": _tensor_bytes(native_packed_output),
            "median_ms": native_packed_ms,
            "million_weights_per_s": weights / native_packed_ms / 1000.0,
            "packed_payload_gib_per_s": packed_bytes / (native_packed_ms / 1000.0) /
                                          (1 << 30),
            "stored_decimal_gb_per_s": stored_decimal_gb_per_s(
                trellis_accounting["total_bytes"], native_packed_ms),
        },
        "matched_codebook_expand": {
            "selection": (
                "explicit --cb-k" if args.cb_k is not None
                else "nearest reader-compatible rung by exact total stored "
                     "tensor-payload bytes"
            ),
            "same_shape": True,
            "accounting": cb_accounting,
            "runtime_inputs": cb_runtime,
            "semantics": cb_semantic,
            "median_ms": cb_expand_ms,
            "million_weights_per_s": weights / cb_expand_ms / 1000.0,
            "stored_decimal_gb_per_s": stored_decimal_gb_per_s(
                cb_accounting["total_bytes"], cb_expand_ms),
        },
        "expand_comparison": {
            "output_semantics_comparable": comparable,
            "trellis_speedup_over_codebook": (
                cb_expand_ms / native_packed_ms if comparable else None),
            "diagnostic_codebook_ms_over_trellis_ms": (
                cb_expand_ms / native_packed_ms),
            "trellis_minus_codebook_stored_bytes": (
                trellis_accounting["total_bytes"]
                - cb_accounting["total_bytes"]
            ),
            "trellis_minus_codebook_bpw": (
                trellis_accounting["exact_bpw"]
                - cb_accounting["exact_bpw"]
            ),
            "absolute_bpw_mismatch": abs(
                trellis_accounting["exact_bpw"]
                - cb_accounting["exact_bpw"]
            ),
            "interpretation": (
                "E4M3 compares raw native-code bytes with separate row scales; "
                "E2M1 has no speedup claim because the existing CB wrapper "
                "emits scaled BF16 rather than packed native nibbles. Neither "
                "case is prefill or serving parity."
            ),
        },
        "dequant_gemv": {
            "median_ms": gemv_ms,
            "million_weights_per_s": weights * args.batch / gemv_ms / 1000.0,
            "packed_payload_gib_per_s":
                packed_bytes * args.batch / (gemv_ms / 1000.0) / (1 << 30),
        },
        "setup_and_all_timed_loops_seconds": setup_seconds,
        "total_process_seconds": time.monotonic() - process_start,
        "limitations": [
            "synthetic packed-wire traffic, not end-to-end serving",
            "FP32 correctness GEMV, not a tensor-core prefill mainloop",
            "stored GB/s is a serialized-byte normalization, not measured HBM traffic",
            "E2M1 TCQ/CB output semantics differ; no E2 speedup is claimed",
            "native packed output is transient staging, never resident model state",
            "the fast op is valid only through its immutable prepared owner",
            "CB short-tail shapes are unsupported; matched measurement requires columns%256==0",
            "one shared CB codebook is fully charged to this one synthetic layer",
            "no device qualification or release authority",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
