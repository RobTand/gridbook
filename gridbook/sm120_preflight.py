"""No-launch compile preflight for NVFP4-CB on RTX 50 (compute 12.0).

The physical qualification host is a GB10 (compute 12.1), so it cannot prove
RTX 50 device correctness or speed.  This callable instead cross-compiles the
four CUDA modules with explicit ``sm_120`` / ``sm_120a`` SASS, checks their
complete binding contracts, and verifies the emitted cubin targets.  Public
NVFP4 format support is K1..K25.  The bindings intentionally retain K26..K32
templates as a direct research surface; compiling those templates does not
make them reader-, producer-, lane-, or artifact-supported.  The preflight
never queries a live device and never invokes an operator.

A passing receipt has a permanent ``compile_only`` ceiling.  Run it in the
pinned CUDA serving toolchain with an explicit, non-production build directory::

    python -m gridbook.sm120_preflight --build-directory /path/to/build \
        --receipt /path/to/receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from . import cuda_ext
from .runtime_contract import load_runtime_contract


SM120_CAPABILITY = (12, 0)
SM120_GENERIC_GENCODE = cuda_ext._gencode_flag(
    SM120_CAPABILITY, accelerated=False)
SM120_ACCELERATED_GENCODE = cuda_ext._gencode_flag(
    SM120_CAPABILITY, accelerated=True)
_NVFP4_PUBLIC_RUNGS = tuple(range(1, 26))
_NVFP4_DIRECT_KERNEL_RUNGS = tuple(range(1, 33))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvfp4_format_row() -> dict[str, Any]:
    rows = [row for row in load_runtime_contract()["formats"]
            if row["family"] == "NVFP4_CB_K"]
    if len(rows) != 1:
        raise RuntimeError("runtime contract must contain one NVFP4_CB_K row")
    row = rows[0]
    expected = {
        "rungs": _NVFP4_PUBLIC_RUNGS,
        "producer_rungs": _NVFP4_PUBLIC_RUNGS,
    }
    for field, law in expected.items():
        if tuple(row[field]) != law:
            raise RuntimeError(
                f"SM120 preflight {field} drift: expected "
                f"{list(law)}, got {row[field]}")
    return row


def _validate_sources(sources: dict[str, Path]) -> None:
    required = {
        "main": (
            '"fp4-v2 GEMV: k_bits must be in [1,32]',
            '"fp4-v2 MoE GEMV: k_bits must be in [1,32]',
            'm.def("cb_gemv_fp4_v2"',
            'm.def("cb_moe_gemv_fp4_v2"',
        ),
        "v2": (
            '"k_bits must be in [1,32]',
            "cb_expand_v2_kernel<WARPS, false>",
            'm.def("cb_expand_v2_stages_dictionary"',
        ),
        "persistent_b": (
            '"cb_moe_persistent_b: FP4-CB v2 supports k_bits in [1,32]',
            "k<=32 makes each half-index <=16 bits",
            'm.def("cb_moe_persistent_b_prefill"',
        ),
        "bf16_bridge": (
            'm.def("cb_bf16_grouped_mm"',
            "PRISMAQUANT_CB_BF16_SM120",
            'm.def("cb_bf16_grouped_mm_sm120"',
        ),
    }
    for key, fragments in required.items():
        text = sources[key].read_text(encoding="utf-8")
        missing = [fragment for fragment in fragments if fragment not in text]
        if missing:
            raise RuntimeError(
                f"{sources[key]} predates the K1..K32 direct NVFP4 kernel "
                "research surface; "
                f"missing {missing}")


def _sass_targets(shared_object: str | os.PathLike[str]) -> list[str]:
    """Return SASS targets reported by cuobjdump, refusing PTX-only output."""

    tool = shutil.which("cuobjdump")
    if tool is None:
        raise RuntimeError("cuobjdump is required to attest emitted SASS")
    result = subprocess.run(
        [tool, "--list-elf", os.fspath(shared_object)], check=False,
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"cuobjdump --list-elf failed for {shared_object}: "
            f"{(result.stderr or result.stdout).strip()}")
    targets = sorted(set(re.findall(r"sm_\d+a?", result.stdout)))
    if not targets:
        raise RuntimeError(f"{shared_object} contains no reported SASS cubin")
    return targets


def compile_nvfp4_sm120_preflight(
    build_directory: str | os.PathLike[str], *, verbose: bool = False,
) -> dict[str, Any]:
    """Cross-compile public K1..K25 plus research-only K26..K32, no launch."""

    _nvfp4_format_row()
    build_inputs = (
        "cb_gemv.cu", "cb_gemv_v2.cu", "cb_moe_persistent_b.cu",
        *cuda_ext._BF16_GROUPED_BUILD_INPUTS,
    )
    src_dir = Path(cuda_ext._require_csrc(*build_inputs))
    sources = {
        "main": src_dir / "cb_gemv.cu",
        "v2": src_dir / "cb_gemv_v2.cu",
        "persistent_b": src_dir / "cb_moe_persistent_b.cu",
        "bf16_bridge": src_dir / "cb_bf16_grouped_gemm.cu",
    }
    _validate_sources(sources)

    build = Path(build_directory).expanduser().resolve()
    if build.exists() and not build.is_dir():
        raise NotADirectoryError(f"preflight build path is not a directory: {build}")
    build.mkdir(parents=True, exist_ok=True)

    import torch  # noqa: F401 -- cpp_extension requires torch first
    from torch.utils.cpp_extension import load

    specs = {
        "main": (cuda_ext._EXT_SYMBOLS, SM120_GENERIC_GENCODE, False),
        "v2": (cuda_ext._V2_SYMBOLS, SM120_GENERIC_GENCODE, False),
        "persistent_b": (
            (*cuda_ext._MOE_PERSISTENT_B_SYMBOLS,
             *cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS),
            SM120_ACCELERATED_GENCODE,
            True,
        ),
        "bf16_bridge": (
            (*cuda_ext._BF16_GROUPED_SYMBOLS,
             *cuda_ext._BF16_GROUPED_SM120_SYMBOLS),
            SM120_ACCELERATED_GENCODE,
            True,
        ),
    }
    source_inputs = {
        "main": ("cb_gemv.cu",),
        "v2": ("cb_gemv_v2.cu",),
        "persistent_b": ("cb_moe_persistent_b.cu",),
        "bf16_bridge": cuda_ext._BF16_GROUPED_BUILD_INPUTS,
    }
    cutlass_include = cuda_ext._find_cutlass_include()
    cutlass_util_include = os.path.join(
        os.path.dirname(cutlass_include), "tools", "util", "include")
    modules: dict[str, dict[str, Any]] = {}
    for key, (symbols, gencode, accelerated) in specs.items():
        source = sources[key]
        source_sha256 = _sha256(source)
        module_name = f"gridbook_nvfp4_sm120_{key}_{source_sha256[:16]}"
        module_build = build / key
        module_build.mkdir(parents=True, exist_ok=True)
        flags = ["-O3"]
        if accelerated:
            flags.append("--expt-relaxed-constexpr")
        if key == "bf16_bridge":
            flags.append(f"-D{cuda_ext._BF16_GROUPED_SM120_DEFINE}=1")
        flags.append(gencode)
        include_paths = [os.fspath(src_dir)]
        if key == "bf16_bridge":
            include_paths = [cutlass_include, cutlass_util_include,
                             os.fspath(src_dir)]
        module = load(
            name=module_name,
            sources=[os.fspath(source)],
            extra_include_paths=include_paths,
            extra_cuda_cflags=flags,
            build_directory=os.fspath(module_build),
            verbose=bool(verbose),
        )
        cuda_ext._require_symbols(
            module, symbols, build_dir=os.fspath(module_build),
            source=f"{source.name} (sm120 compile-only preflight)")
        targets = _sass_targets(module.__file__)
        expected_target = "sm_120a" if accelerated else "sm_120"
        if targets != [expected_target]:
            raise RuntimeError(
                f"{source.name} SASS targets {targets} != [{expected_target!r}]")
        modules[key] = {
            "source": f"gridbook/csrc/{source.name}",
            "source_sha256": source_sha256,
            "build_input_sha256": {
                name: _sha256(src_dir / name) for name in source_inputs[key]
            },
            "module_name": module_name,
            "gencode": gencode,
            "required_symbols": list(symbols),
            "sass_targets": targets,
        }

    return {
        "schema": "gridbook.sm120-nvfp4-compile-preflight.v1",
        "capability": list(SM120_CAPABILITY),
        "qualification_ceiling": "compile_only",
        "device_executed": False,
        "reader_rungs": list(_NVFP4_PUBLIC_RUNGS),
        "producer_rungs": list(_NVFP4_PUBLIC_RUNGS),
        "direct_kernel_research_rungs": list(_NVFP4_DIRECT_KERNEL_RUNGS),
        "modules": modules,
        "claims_excluded": [
            "device_correctness",
            "device_performance",
            "torch_compile",
            "vllm_cudagraph",
            "artifact_compatibility_k26_k32",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    receipt = compile_nvfp4_sm120_preflight(
        args.build_directory, verbose=args.verbose)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        path = Path(args.receipt).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SM120_ACCELERATED_GENCODE",
    "SM120_CAPABILITY",
    "SM120_GENERIC_GENCODE",
    "compile_nvfp4_sm120_preflight",
    "main",
]
