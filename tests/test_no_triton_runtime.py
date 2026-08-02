"""Static ratchet: Gridbook's packaged serving path is native-only.

Three layers, in increasing strength:

1. **The packaged runtime** (``gridbook/*.py``) is scanned with every rule.
2. **The rest of the tree** (``scripts/``, ``tests/``) is scanned with the rules
   that make a module *reach* Triton. A helper script or a test that grows a
   ``import triton`` is how the dependency comes back through the side door,
   and neither is covered by scanning the package alone. The dispatch-policy
   rules (external GEMM fallbacks) stay package-only on purpose: a test that
   compares a CUDA kernel against ``F.linear`` is using the reference oracle
   the kernel is supposed to match, which is the opposite of a fallback.
3. **The dynamic edge.** ``plugin.py`` calls ``importlib.import_module`` over
   the model-module list in ``runtime_contract.json``, which no AST scan can
   see. That list is pinned here against an explicit allow-list, so a new
   dynamic import target cannot land without a reviewed edit to this file.

Plus a GPU-lane *runtime* assertion (skipped without CUDA): executing a
Gridbook-owned op must not pull a single new ``triton`` module into
``sys.modules``. vLLM imports Triton unconditionally, so the claim that can
honestly be made — and is made here — is a **delta** against what a stock
compressed-tensors serving surface already loaded.
"""
from __future__ import annotations

import ast
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


TEST_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
SOURCE_PACKAGE = TEST_ROOT / "gridbook"
if SOURCE_PACKAGE.is_dir():
    ROOT = TEST_ROOT
    PACKAGE = SOURCE_PACKAGE
else:
    # Release CI stages tests outside the checkout specifically to exercise the
    # non-editable wheel. Scan the package that Python actually imported rather
    # than passing vacuously because ``<staged-tests>/../gridbook`` is absent.
    import gridbook

    PACKAGE = Path(gridbook.__file__).resolve().parent
    ROOT = PACKAGE.parent


# ---------------------------------------------------------------------------
# Violation kinds. ``reach`` is what actually loads Triton into the process;
# ``mention`` only names it; ``dispatch`` is the serving-path policy about
# leaving Gridbook's own kernels for someone else's GEMM.
_REACH = "reach"
_MENTION = "mention"
_DISPATCH = "dispatch"

# Files exempt from the ``mention`` rules ONLY, each with its reason. The
# ``reach`` rules apply to every file with no exceptions: an entry here buys
# the right to NAME the banned lane, never to import or call it — which is
# asserted separately below, so an exemption cannot quietly widen.
_MENTION_EXEMPT: dict[str, str] = {
    "gridbook/delegated_preflight.py":
        "Defines the fail-closed D0.2 policy that REFUSES a delegated backend "
        "resolving to Triton. A rejection table has to spell the name it "
        "rejects, and the token test that catches an unnamed future backend "
        "has to hold the token.",
    "tests/test_no_triton_runtime.py":
        "This ratchet. It is the scanner, so it necessarily names what it "
        "bans, in rules, messages, and its own allow-lists.",
    "tests/test_delegated_preflight.py":
        "Proves the D0.2 policy REJECTS a Triton-backed delegated backend. It "
        "builds stub backend classes to be refused and asserts the refusal "
        "names them; a test that could not spell the name could not check it.",
}


def _display(path: Path) -> str:
    for base in (ROOT, TEST_ROOT, TESTS_DIR.parent):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return path.name


def _is_triton_module(name: str | None) -> bool:
    return bool(name) and (name == "triton" or name.startswith("triton."))


def _call_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _runtime_violations(path: Path, *,
                        kinds: frozenset[str]) -> list[str]:
    """Rule hits in *path* restricted to *kinds*, formatted for an assert."""

    exemption = _MENTION_EXEMPT.get(_display(path).replace(os.sep, "/"))
    if exemption is not None:
        kinds = kinds - {_MENTION}
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: set[tuple[int, str, str]] = set()
    docstrings = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and owner.body:
            first = owner.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))

    def hit(kind: str, node: ast.AST, message: str) -> None:
        if kind in kinds:
            violations.add((node.lineno, kind, message))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_triton_module(alias.name):
                    hit(_REACH, node, f"import {alias.name}")
                if alias.name == "gridbook.kernels":
                    hit(_REACH, node, "import of retired kernels module")
                if alias.name == "vllm._custom_ops":
                    hit(_REACH, node,
                        "import of vLLM helper with Triton fallback")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_triton_module(module):
                hit(_REACH, node, f"from {module} import ...")
            if (node.level and module == "kernels") \
                    or module == "gridbook.kernels":
                hit(_REACH, node, "import of retired kernels module")
            if module == "gridbook" and any(
                    alias.name == "kernels" for alias in node.names):
                hit(_REACH, node, "import of retired kernels module")
            if module == "vllm._custom_ops":
                hit(_REACH, node, "import of vLLM helper with Triton fallback")
            if module == "vllm.platforms" or module.startswith(
                    "vllm.platforms."):
                hit(_REACH, node,
                    "import of vLLM platform package initializes compiler "
                    "backends")
            if module.endswith("fused_moe.utils") and any(
                    alias.name == "moe_kernel_quantize_input"
                    for alias in node.names):
                hit(_REACH, node, "import of external MoE quantizer helper")
            if module.endswith("fused_moe.activation") and any(
                    alias.name == "apply_moe_activation"
                    for alias in node.names):
                hit(_REACH, node,
                    "import of activation helper with Triton branch")
        elif isinstance(node, ast.Name) and "triton" in node.id.lower():
            hit(_MENTION, node, f"executable reference `{node.id}`")
        elif isinstance(node, ast.Attribute) and "triton" in node.attr.lower():
            hit(_MENTION, node, f"executable attribute `.{node.attr}`")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)) \
                and "triton" in node.name.lower():
            hit(_MENTION, node, f"executable definition `{node.name}`")
        elif isinstance(node, ast.Constant) and id(node) not in docstrings \
                and isinstance(node.value, str) \
                and _is_triton_module(node.value.strip().lower()):
            hit(_MENTION, node, f"executable literal {node.value!r}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "triton" or name.startswith("triton."):
                hit(_REACH, node, f"call to {name}")
            if name in {"torch._grouped_mm", "F.linear",
                        "torch.nn.functional.linear"}:
                hit(_DISPATCH, node, f"external GEMM fallback call to {name}")
            if name in {"__import__", "importlib.import_module"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and (_is_triton_module(arg.value)
                             or arg.value == "gridbook.kernels"):
                    hit(_REACH, node, f"dynamic import of {arg.value}")
    return [f"{_display(path)}:{line}: [{kind}] {message}"
            for line, kind, message in sorted(violations)]


_ALL_KINDS = frozenset({_REACH, _MENTION, _DISPATCH})
_REACH_ONLY = frozenset({_REACH, _MENTION})


def test_packaged_runtime_contains_no_triton_code():
    retired = [
        PACKAGE / "kernels.py",
        PACKAGE / "moe_autotune.py",
        PACKAGE / "moe_l2.py",
    ]
    present = [str(path.relative_to(ROOT)) for path in retired if path.exists()]
    assert not present, (
        "retired interpreted/selectable serving modules remain: "
        + ", ".join(present)
    )

    violations = [
        violation
        for path in sorted(PACKAGE.rglob("*.py"))
        for violation in _runtime_violations(path, kinds=_ALL_KINDS)
    ]
    assert not violations, "interpreted serving code remains:\n" + "\n".join(
        violations)


def test_repository_scripts_and_tests_reach_no_triton():
    """No NEW Triton-reaching import sneaks in anywhere else in the tree.

    Scanning only ``gridbook/`` left two whole directories unwatched, and both
    ship in the source distribution and run in CI.
    """

    roots = [directory for directory in (TEST_ROOT / "scripts", TESTS_DIR)
             if directory.is_dir()]
    assert roots, "neither scripts/ nor tests/ is present to scan"
    violations = [
        violation
        for directory in roots
        for path in sorted(directory.rglob("*.py"))
        for violation in _runtime_violations(path, kinds=_REACH_ONLY)
    ]
    assert not violations, (
        "Triton-reaching code outside the package:\n" + "\n".join(violations))


def test_mention_exemptions_never_hide_a_reaching_import():
    """An exemption buys the right to name Triton, never to reach it."""

    for relative, reason in _MENTION_EXEMPT.items():
        assert reason.strip(), f"{relative} is exempt with no written reason"
        path = ROOT / relative
        if not path.is_file():
            path = TEST_ROOT / relative
        if not path.is_file():
            continue          # staged-tests layout; the package scan covers it
        reaching = _runtime_violations(path, kinds=frozenset({_REACH}))
        assert not reaching, (
            f"{relative} is exempt from the name rules only, but it reaches "
            "Triton:\n" + "\n".join(reaching))


# ---------------------------------------------------------------------------
# The dynamic edge: plugin.py imports these by name out of the runtime
# contract, so the AST scan above cannot see them. Every entry is a vLLM
# architecture module whose model classes load MoE experts at the top level
# (see plugin.py). Adding one is a decision to import a new third-party module
# into the serving process, which is exactly the kind of change this file
# exists to make visible — so it must be made here as well as in the contract.
_CONTRACT_MODEL_MODULES = {
    "vllm.model_executor.models.hy_v3",
    "vllm.model_executor.models.hy_v3_mtp",
    "vllm.model_executor.models.laguna",
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_5_mtp",
    "vllm.model_executor.models.lfm2_moe",
}


def test_runtime_contract_model_modules_are_exactly_allow_listed():
    from gridbook.runtime_contract import load_runtime_contract

    declared = set(
        load_runtime_contract()["producer_profiles"]["top_level_loader_modules"]
    )
    assert declared == _CONTRACT_MODEL_MODULES, (
        "runtime_contract.json's dynamic import targets drifted from the "
        "reviewed allow-list.\n"
        f"  contract only: {sorted(declared - _CONTRACT_MODEL_MODULES)}\n"
        f"  allow-list only: {sorted(_CONTRACT_MODEL_MODULES - declared)}"
    )


# ---------------------------------------------------------------------------
# GPU lane: the runtime delta assertion.

#: What a vLLM process loads to serve a *stock* compressed-tensors checkpoint —
#: i.e. the alternative to using Gridbook at all. Triton that appears here is
#: vLLM's, not Gridbook's, and subtracting it is what makes the claim honest.
_BASELINE_SCRIPT = """
import json, sys
import vllm
import vllm.model_executor.models
import vllm.model_executor.layers.fused_moe
import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors
print("@@" + json.dumps({"modules": sorted(
    n for n in sys.modules if n == "triton" or n.startswith("triton."))}))
"""

#: Bare vLLM, then Gridbook's own import graph, then an actual Gridbook op.
#: Two snapshots so an import-time regression and an execution-time regression
#: are distinguishable in the failure message.
_GRIDBOOK_SCRIPT = """
import json, sys


def triton_modules():
    return sorted(n for n in sys.modules if n == "triton" or n.startswith("triton."))


def emit(payload):
    print("@@" + json.dumps(payload))
    raise SystemExit(0)


import vllm  # noqa: F401  -- bare vLLM is the snapshot point
after_vllm = triton_modules()

import torch
if not torch.cuda.is_available():
    emit({"skip": "no CUDA device"})

import gridbook.plugin
from gridbook.cuda_ext import get_ext

gridbook.plugin.register()
ext = get_ext()
if ext is None or not hasattr(ext, "fp4_act_qdq"):
    emit({"skip": "Gridbook CUDA extension unavailable (no nvcc?)"})
after_gridbook_import = triton_modules()

x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
y = torch.ops.prismaquant.fp4_act_qdq(x)
torch.cuda.synchronize()
assert y.shape == x.shape
after_execution = triton_modules()

emit({"after_vllm": after_vllm,
      "after_gridbook_import": after_gridbook_import,
      "after_execution": after_execution})
"""


def _run_probe(script: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=3600,
        cwd=str(TEST_ROOT), env=dict(os.environ),
    )
    marker = [line for line in completed.stdout.splitlines()
              if line.startswith("@@")]
    assert marker, (
        "Triton-delta probe produced no result.\n"
        f"exit={completed.returncode}\nstdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-4000:]}"
    )
    return json.loads(marker[-1][2:])


def test_gridbook_execution_imports_no_additional_triton_modules():
    """Gridbook's execution adds no Triton module beyond vLLM's own.

    Deliberately a *delta*, not an absolute: vLLM imports Triton
    unconditionally, so "no triton in ``sys.modules``" is not a claim this
    plugin can make from inside a vLLM process, and pretending otherwise would
    be the kind of overclaim the docs are being cleaned up to remove.
    """

    if util.find_spec("vllm") is None:
        import pytest

        pytest.skip("vLLM not installed; the delta needs a real vLLM baseline")
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; the GPU lane cannot execute an op")

    result = _run_probe(_GRIDBOOK_SCRIPT)
    if "skip" in result:
        pytest.skip(result["skip"])
    baseline = set(_run_probe(_BASELINE_SCRIPT)["modules"])

    after_vllm = set(result["after_vllm"])
    after_import = set(result["after_gridbook_import"])
    after_execution = set(result["after_execution"])

    execution_delta = sorted(after_execution - after_import)
    assert not execution_delta, (
        "executing a Gridbook-owned op imported Triton modules: "
        + ", ".join(execution_delta))

    import_delta = sorted(after_import - after_vllm - baseline)
    assert not import_delta, (
        "importing Gridbook's runtime imported Triton modules that a stock "
        "compressed-tensors vLLM process does not: " + ", ".join(import_delta))


def test_distribution_does_not_depend_on_triton():
    # Python 3.10 is supported and has no stdlib tomllib. Inspect every TOML
    # requirement array while ignoring comments, which is sufficient for this
    # static package-policy ratchet and keeps it runnable at the declared floor.
    source_roots = [TEST_ROOT]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            source_roots.append(Path(value).expanduser())
    pyproject = next(
        (root / "pyproject.toml" for root in source_roots
         if (root / "pyproject.toml").is_file()),
        None,
    )
    if pyproject is not None:
        text = pyproject.read_text()
        arrays = re.findall(
            r"(?ms)^(?:requires|dependencies|[A-Za-z0-9_-]+)\s*=\s*\[(.*?)\]",
            text,
        )
        assert arrays, "pyproject.toml has no dependency arrays"
        requirements = re.findall(
            r"[\"']\s*([^\"']+)\s*[\"']", "\n".join(arrays)
        )
    else:
        # A wheel intentionally does not ship pyproject.toml. Its installed
        # Core Metadata is the release artifact's authoritative dependency set.
        requirements = metadata.requires("gridbook") or []

    def package_name(requirement: str) -> str:
        return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0] \
            .lower().replace("_", "-")

    bad = [requirement for requirement in requirements
           if package_name(requirement) == "triton"]
    assert not bad, f"dependency still present: {bad}"
