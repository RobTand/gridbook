"""Compile-only preflight for Gridbook's dense FP8-CB Ada routes.

This module deliberately never asks for a live CUDA device and never launches
a kernel.  It cross-compiles the production ``cb_gemv.cu`` module with an
explicit ``sm_89`` SASS target, validates the production symbol surface, and
loads vLLM's direct native FP8 quantization/CUTLASS ABI without invoking it.

Passing this preflight can support only ``qualification=compile_only`` in the
runtime contract.  It is not a substitute for a physical RTX 4090 correctness,
performance, torch.compile, or CUDA-graph receipt.

Run explicitly on an assigned compile host::

    python -m gridbook.sm89_preflight --build-directory /path/to/build \
        --receipt /path/to/receipt.json

Gridbook's extension loaders use an explicit ``-gencode`` flag rather than
``TORCH_CUDA_ARCH_LIST``; this preflight uses that same mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import cuda_ext
from .native_cutlass import _REQUIRED_OPS, require_native_fp8_cutlass
from .runtime_contract import load_runtime_contract


SM89_CAPABILITY = (8, 9)
SM89_GENCODE = cuda_ext._gencode_flag(SM89_CAPABILITY, accelerated=False)
_FP8_PRODUCER_RUNGS = tuple(range(4, 49, 4))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_fp8_format_row() -> dict[str, Any]:
    contract = load_runtime_contract()
    rows = [row for row in contract["formats"]
            if row["family"] == "FP8_CB_K"]
    if len(rows) != 1:
        raise RuntimeError("runtime contract must contain one FP8_CB_K row")
    row = rows[0]
    if tuple(row["producer_rungs"]) != _FP8_PRODUCER_RUNGS:
        raise RuntimeError(
            "SM89 preflight producer law drift: expected "
            f"{list(_FP8_PRODUCER_RUNGS)}, got {row['producer_rungs']}"
        )
    return row


def _validate_dense_fp8_source(source: Path) -> None:
    """Refuse a source tree whose generic Ada surface predates low rungs."""

    text = source.read_text(encoding="utf-8")
    required = (
        "fp8_reader_kbits_supported(k_bits)",
        '"fp8 k_bits is outside the v10 accepted reader domain"',
        "type_size == 4 * k_bits",
        "type_size <= 192",
        "fp8_load_codeword_words(",
        'm.def("cb_gemv_fp8"',
        'm.def("cb_expand_fp8"',
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise RuntimeError(
            f"{source} does not expose the v10 dense FP8-CB generic surface; "
            f"missing source guards/bindings {missing}"
        )


def compile_dense_fp8_sm89_preflight(
    build_directory: str | os.PathLike[str], *, verbose: bool = False
) -> dict[str, Any]:
    """Cross-compile and ABI-check dense FP8-CB routes without kernel launch.

    ``build_directory`` is intentionally explicit so a preflight cannot alter
    the production extension cache.  The returned object is suitable for an
    external compile receipt, but its qualification ceiling is permanently
    ``compile_only``.
    """

    _dense_fp8_format_row()
    source_dir = Path(cuda_ext._require_csrc("cb_gemv.cu"))
    source = source_dir / "cb_gemv.cu"
    _validate_dense_fp8_source(source)
    source_sha256 = _sha256(source)

    build = Path(build_directory).expanduser().resolve()
    if build.exists() and not build.is_dir():
        raise NotADirectoryError(f"preflight build path is not a directory: {build}")
    build.mkdir(parents=True, exist_ok=True)

    # Importing and loading the extension registers its bindings but does not
    # invoke them.  The explicit arch flag is the same authority as production
    # cuda_ext.get_ext and makes a live-device capability probe unnecessary.
    import torch  # noqa: F401 -- cpp_extension requires torch imported first
    from torch.utils.cpp_extension import load

    module_name = f"gridbook_fp8_sm89_preflight_{source_sha256[:16]}"
    module = load(
        name=module_name,
        sources=[os.fspath(source)],
        build_directory=os.fspath(build),
        extra_cuda_cflags=["-O3", SM89_GENCODE],
        verbose=bool(verbose),
    )
    cuda_ext._require_symbols(
        module,
        cuda_ext._EXT_SYMBOLS,
        build_dir=os.fspath(build),
        source="cb_gemv.cu (sm89 compile-only preflight)",
    )

    # The large-M route composes Gridbook's native expander with vLLM's direct
    # per-token FP8 quantizer and CUTLASS W8A8 op.  This checks that exact ABI,
    # bypassing the convenience wrapper that can choose Triton.  No tensor is
    # allocated and neither operator is called.
    require_native_fp8_cutlass("dense FP8-CB SM89 compile-only preflight")

    return {
        "schema": "gridbook.sm89-compile-preflight.v1",
        "capability": list(SM89_CAPABILITY),
        "qualification_ceiling": "compile_only",
        "device_executed": False,
        "producer_rungs": list(_FP8_PRODUCER_RUNGS),
        "gridbook_extension": {
            "source": "gridbook/csrc/cb_gemv.cu",
            "source_sha256": source_sha256,
            "module_name": module_name,
            "gencode": SM89_GENCODE,
            "required_symbols": list(cuda_ext._EXT_SYMBOLS),
        },
        "vllm_native_abi": {
            "required_ops": list(_REQUIRED_OPS),
            "status": "present_not_executed",
        },
        "claims_excluded": [
            "device_correctness",
            "device_performance",
            "torch_compile",
            "vllm_cudagraph",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-directory", required=True,
        help="dedicated build directory; the production extension cache is not used",
    )
    parser.add_argument(
        "--receipt",
        help="optional path for the same compile-only JSON printed to stdout",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    receipt = compile_dense_fp8_sm89_preflight(
        args.build_directory, verbose=args.verbose
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        receipt_path = Path(args.receipt).expanduser().resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "SM89_CAPABILITY",
    "SM89_GENCODE",
    "compile_dense_fp8_sm89_preflight",
    "main",
]
