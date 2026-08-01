"""Static ratchet: Gridbook's packaged serving path is native-only."""
from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "gridbook"


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


def _runtime_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: set[tuple[int, str]] = set()
    docstrings = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and owner.body:
            first = owner.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_triton_module(alias.name):
                    violations.add((node.lineno, f"import {alias.name}"))
                if alias.name == "gridbook.kernels":
                    violations.add(
                        (node.lineno, "import of retired kernels module"))
                if alias.name == "vllm._custom_ops":
                    violations.add((node.lineno,
                                    "import of vLLM helper with Triton fallback"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_triton_module(module):
                violations.add((node.lineno, f"from {module} import ..."))
            if (node.level and module == "kernels") \
                    or module == "gridbook.kernels":
                violations.add((node.lineno, "import of retired kernels module"))
            if module == "gridbook" and any(
                    alias.name == "kernels" for alias in node.names):
                violations.add((node.lineno,
                                "import of retired kernels module"))
            if module == "vllm._custom_ops":
                violations.add((node.lineno,
                                "import of vLLM helper with Triton fallback"))
            if module == "vllm.platforms" or module.startswith(
                    "vllm.platforms."):
                violations.add((node.lineno,
                                "import of vLLM platform package initializes "
                                "compiler backends"))
            if module.endswith("fused_moe.utils") and any(
                    alias.name == "moe_kernel_quantize_input"
                    for alias in node.names):
                violations.add((node.lineno,
                                "import of external MoE quantizer helper"))
            if module.endswith("fused_moe.activation") and any(
                    alias.name == "apply_moe_activation"
                    for alias in node.names):
                violations.add((
                    node.lineno,
                    "import of activation helper with Triton branch"))
        elif isinstance(node, ast.Name) and "triton" in node.id.lower():
            violations.add(
                (node.lineno, f"executable reference `{node.id}`"))
        elif isinstance(node, ast.Attribute) and "triton" in node.attr.lower():
            violations.add(
                (node.lineno, f"executable attribute `.{node.attr}`"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)) \
                and "triton" in node.name.lower():
            violations.add((node.lineno,
                            f"executable definition `{node.name}`"))
        elif isinstance(node, ast.Constant) and id(node) not in docstrings \
                and isinstance(node.value, str) \
                and _is_triton_module(node.value.strip().lower()):
            violations.add(
                (node.lineno, f"executable literal {node.value!r}"))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "triton" or name.startswith("triton."):
                violations.add((node.lineno, f"call to {name}"))
            if name in {"torch._grouped_mm", "F.linear",
                        "torch.nn.functional.linear"}:
                violations.add((node.lineno,
                                f"external GEMM fallback call to {name}"))
            if name in {"__import__", "importlib.import_module"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and (_is_triton_module(arg.value)
                             or arg.value == "gridbook.kernels"):
                    violations.add((node.lineno,
                                    f"dynamic import of {arg.value}"))
    return [f"{path.relative_to(ROOT)}:{line}: {message}"
            for line, message in sorted(violations)]


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
        for violation in _runtime_violations(path)
    ]
    assert not violations, "interpreted serving code remains:\n" + "\n".join(
        violations)


def test_distribution_does_not_depend_on_triton():
    # Python 3.10 is supported and has no stdlib tomllib. Inspect every TOML
    # requirement array while ignoring comments, which is sufficient for this
    # static package-policy ratchet and keeps it runnable at the declared floor.
    text = (ROOT / "pyproject.toml").read_text()
    arrays = re.findall(
        r"(?ms)^(?:requires|dependencies|[A-Za-z0-9_-]+)\s*=\s*\[(.*?)\]",
        text,
    )
    assert arrays, "pyproject.toml has no dependency arrays"
    requirements = re.findall(
        r"[\"']\s*([^\"']+)\s*[\"']", "\n".join(arrays)
    )

    def package_name(requirement: str) -> str:
        return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0] \
            .lower().replace("_", "-")

    bad = [requirement for requirement in requirements
           if package_name(requirement) == "triton"]
    assert not bad, f"dependency still present: {bad}"
