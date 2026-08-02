"""SASS attestation: the hot decode modules are compiled for THIS device.

``get_ext()`` and ``get_ext_v2()`` used to pass only ``-O3`` and inherit
``TORCH_CUDA_ARCH_LIST``. The stock vLLM base image ships
``"8.0 8.7 8.9 9.0 10.0 11.0 12.0"``, which OMITS 12.1 — so outside Gridbook's
own Dockerfile a GB10 ran the production decode GEMV from PTX JIT or against a
mismatched SASS target (2026-08-01 performance audit, §3 P0.1). Both loaders
now derive ``-gencode`` from the live device, and the only way to check that
claim is to disassemble what was actually built.

GPU lane only: this builds (or loads) the real extensions and shells out to
``cuobjdump``. Every missing prerequisite skips with its reason, never passes
vacuously.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
cuda_ext = pytest.importorskip("gridbook.cuda_ext")


def _cuobjdump() -> str | None:
    """An executable cuobjdump, preferring PATH then the toolkit default."""
    for candidate in ("cuobjdump", "/usr/local/cuda/bin/cuobjdump"):
        resolved = shutil.which(candidate)
        if resolved is None and os.path.isfile(candidate):
            resolved = candidate
        if resolved is None:
            continue
        try:
            subprocess.run([resolved, "--version"], capture_output=True,
                           check=True)
        except Exception:  # noqa: BLE001 — a broken candidate is not a result
            continue
        return resolved
    return None


def _elf_targets(binary: str) -> set[str]:
    """The ``sm_XY`` device images a built ``.so`` actually contains.

    ``--list-elf`` names one cubin per compiled architecture and needs no
    nvdisasm, which makes it the cheap and portable form of this question. A
    module that carries only PTX for the live device has no ELF for it — which
    is exactly the failure this file exists to catch — so the absence of an
    entry is a real answer, not a tooling gap.
    """
    tool = _cuobjdump()
    if tool is None:
        pytest.skip("cuobjdump is unavailable; cannot attest the SASS target")
    listed = subprocess.run([tool, "--list-elf", binary],
                            capture_output=True, text=True)
    output = listed.stdout
    if listed.returncode != 0 or not output.strip():
        # Older toolkits spell it differently; -sass answers the same question
        # (it prints a "code for sm_XY" banner per image) at the cost of
        # needing nvdisasm.
        sass = subprocess.run([tool, "-sass", binary],
                              capture_output=True, text=True)
        if sass.returncode != 0:
            pytest.skip(
                f"cuobjdump could not read {binary}: "
                f"--list-elf: {listed.stderr.strip()}; "
                f"-sass: {sass.stderr.strip()}")
        output = sass.stdout
    return set(re.findall(r"sm_(\d+)", output))


@pytest.mark.parametrize("loader_name,source", [
    ("get_ext", "cb_gemv.cu"),
    ("get_ext_v2", "cb_gemv_v2.cu"),
])
def test_hot_decode_module_carries_this_devices_sass(loader_name, source):
    if not torch.cuda.is_available():
        pytest.skip("a CUDA device is required to attest its own SASS target")
    major, minor = torch.cuda.get_device_capability()
    module = getattr(cuda_ext, loader_name)()
    if module is None:
        pytest.skip(f"the {source} extension could not be built here "
                    f"(the loader printed the reason)")
    binary = getattr(module, "__file__", None)
    if not binary:
        pytest.skip(f"the loaded {source} module has no file to disassemble")

    targets = _elf_targets(binary)
    expected = f"{major}{minor}"
    assert expected in targets, (
        f"{source} was built without SASS for the live device: expected an "
        f"sm_{expected} device image in {binary}, found "
        f"{sorted('sm_' + t for t in targets) or 'none'}. The decode GEMV "
        f"would run from PTX JIT or a mismatched target — this is the "
        f"TORCH_CUDA_ARCH_LIST inheritance bug (audit §3 P0.1) reappearing.")


def test_grouped_bf16_bridge_carries_this_devices_sass():
    """The quality-path CUTLASS bridge has always derived its own target.

    Kept alongside the two fixed modules so the three architecture-generic
    loaders are attested by one mechanism: if the shared ``_gencode_flag``
    helper ever regressed, this fails with them rather than silently keeping
    a passing baseline.
    """
    if not torch.cuda.is_available():
        pytest.skip("a CUDA device is required to attest its own SASS target")
    major, minor = torch.cuda.get_device_capability()
    module = cuda_ext.get_bf16_grouped_ext()
    if module is None:
        pytest.skip("the grouped-BF16 CUTLASS bridge could not be built here")
    binary = getattr(module, "__file__", None)
    if not binary:
        pytest.skip("the loaded grouped-BF16 module has no file to inspect")
    targets = _elf_targets(binary)
    assert f"{major}{minor}" in targets, (
        f"cb_bf16_grouped_gemm.cu carries {sorted(targets)}, not "
        f"sm_{major}{minor}")


def test_generic_modules_are_not_pinned_to_an_arch_conditional_target():
    """``compute_XYa``/``sm_XYa`` is requested, never inherited.

    The two hot decode modules are sm_80+ portable and must stay generic: an
    arch-conditional binary refuses to load on any other capability at all.
    The two fused modules require the conditional target for their tensor-core
    instructions, and since 2026-08-01 the grouped BF16 bridge requests it too
    — but only where it compiles its sm12x lane (cc 12.x), because the
    sm90-family kernel layer that lane goes through compiles its body only
    under the architecture feature macro. Which callers ask for which
    is asserted in ``tests/test_ext_build_identity.py``; this case pins the
    helper both of them use.
    """
    flag = cuda_ext._gencode_flag((12, 1), accelerated=False)
    assert flag == "-gencode=arch=compute_121,code=sm_121"
    accelerated = cuda_ext._gencode_flag((12, 1), accelerated=True)
    assert accelerated == "-gencode=arch=compute_121a,code=sm_121a"
