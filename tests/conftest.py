"""Shared pytest infrastructure for tests that temporarily stub vLLM."""

from __future__ import annotations

import sys

import pytest


_VLLM_BOUND_GRIDBOOK_MODULES = {
    "gridbook.config",
    "gridbook.linear",
    "gridbook.moe",
}


def _is_isolated_runtime_module(name: str) -> bool:
    return (
        name == "vllm"
        or name.startswith("vllm.")
        or name in _VLLM_BOUND_GRIDBOOK_MODULES
    )


def _clear_gridbook_package_attrs() -> None:
    package = sys.modules.get("gridbook")
    if package is None:
        return
    for module_name in _VLLM_BOUND_GRIDBOOK_MODULES:
        vars(package).pop(module_name.rsplit(".", 1)[1], None)


@pytest.fixture(scope="module")
def isolated_gridbook_runtime_imports():
    """Give one test module a private Gridbook/vLLM import graph.

    Several CPU-only tests replace vLLM with deliberately minimal modules.
    Pytest imports every selected test file before running module fixtures, so
    collection-time stubs otherwise leak into later files and make results
    depend on the file order.  Snapshot complete module objects, start the
    requesting module clean, and restore the exact prior graph afterward.

    Only Gridbook's three vLLM-bound modules are replaced.  In particular,
    ``gridbook.ops`` remains process-global: Torch custom-op registrations hold
    references to that module's registry and cannot safely be re-imported.
    """

    before = {
        name: module
        for name, module in sys.modules.items()
        if _is_isolated_runtime_module(name)
    }
    package = sys.modules.get("gridbook")
    missing = object()
    package_attrs = {
        module_name.rsplit(".", 1)[1]: getattr(
            package, module_name.rsplit(".", 1)[1], missing
        )
        for module_name in _VLLM_BOUND_GRIDBOOK_MODULES
    } if package is not None else {}
    for name in list(sys.modules):
        if _is_isolated_runtime_module(name):
            sys.modules.pop(name, None)
    _clear_gridbook_package_attrs()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_isolated_runtime_module(name):
                sys.modules.pop(name, None)
        _clear_gridbook_package_attrs()
        sys.modules.update(before)
        package = sys.modules.get("gridbook")
        if package is not None:
            for name, value in package_attrs.items():
                if value is not missing:
                    setattr(package, name, value)
