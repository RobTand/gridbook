"""Shared pytest infrastructure for tests that temporarily stub vLLM."""

from __future__ import annotations

import os
import re
import sys

import pytest


# Gridbook-owned includes are quoted and either sit beside the ``.cu`` (``cb_``)
# or in the vendored fork tree; everything else is CUTLASS/CuTe, whose complete
# graph ninja's generated depfile owns inside a digest-keyed build directory.
_OWNED_INCLUDE = re.compile(r'#include\s+"((?:cutlass_fork/|cb_)[^"]+)"')


def gridbook_include_closure(csrc_dir: str, root: str) -> set[str]:
    """Every Gridbook-owned header reachable from ``root``, TRANSITIVELY.

    A build identity that hashes only the includes written in the ``.cu`` is
    one shared header away from being wrong: ``cb_grouped_common.hpp`` includes
    ``cutlass_fork/sm120_expert_row_broadcast.hpp``, so an edit to the latter
    changes four modules' binaries while naming none of them. The grouped-BF16
    module carried exactly that hole until 2026-08-02 (its declared inputs
    covered its direct includes and stopped there), which is why the five
    "declared inputs cover the includes" tests walk this closure instead.

    ``root`` is package-relative, as the declared build-input tuples are.
    Returns the reachable set EXCLUDING ``root`` itself. A named include that
    does not exist on disk is reported by the caller's set comparison rather
    than silently skipped, so a typo in a source cannot hide here.
    """
    seen: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        path = os.path.join(csrc_dir, current)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for name in _OWNED_INCLUDE.findall(handle.read()):
                if name not in seen:
                    seen.add(name)
                    pending.append(name)
    seen.discard(root)
    return seen


@pytest.fixture(autouse=True)
def _fresh_env_latches():
    """Give each test a process that has never read a dispatch flag.

    Gridbook latches every dispatch-selecting environment variable to the first
    value it observes, so a later change raises instead of mixing two kernel
    behaviours inside one run. That is exactly right in a serve and exactly
    wrong across a test session, where one file's ``monkeypatch.setenv`` would
    otherwise pin the value every subsequent test sees. Each test starts from
    the unread state a fresh process would have.
    """
    try:
        from gridbook import lane_select
    except Exception:  # noqa: BLE001 — gridbook may not be importable here
        yield
        return
    lane_select.reset_for_tests()
    yield
    lane_select.reset_for_tests()


_VLLM_BOUND_GRIDBOOK_MODULES = {
    "gridbook.config",
    "gridbook.linear",
    "gridbook.mixed_linear",
    "gridbook.moe",
}


def _is_isolated_runtime_module(name: str) -> bool:
    """Modules whose binding is snapshotted and restored around one test file."""
    return (
        name == "vllm"
        or name.startswith("vllm.")
        or name in _VLLM_BOUND_GRIDBOOK_MODULES
    )


def _may_remove_from_sys_modules(name: str) -> bool:
    """Whether this module may be REMOVED, as opposed to merely restored.

    Snapshot-and-restore is always safe: rebinding an existing module object
    mutates nothing outside ``sys.modules``.  REMOVING one is only safe if the
    module can be imported again, and an installed vLLM cannot: importing it
    registers opaque types with Torch (``vllm.utils.torch_utils.LayerName`` and
    friends) in global C++ state that no ``sys.modules`` bookkeeping can undo.
    A second import therefore dies with

        RuntimeError: Type 'vllm.utils.torch_utils.LayerName' is already
        registered as an opaque type

    and every later test needing real vLLM fails -- 290 of them in a full GPU
    run, each passing when its file runs alone.

    This is the same exemption ``gridbook.ops`` already carries, for the same
    reason: Torch registrations outlive the import graph.  A STUB vLLM has no
    ``__file__`` and no such side effect, so removing it is both safe and the
    entire point of this fixture -- collection-time stubs must not leak.
    """
    if name in _VLLM_BOUND_GRIDBOOK_MODULES:
        return True
    return getattr(sys.modules.get(name), "__file__", None) is None


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
        if _is_isolated_runtime_module(name) and _may_remove_from_sys_modules(name):
            sys.modules.pop(name, None)
    _clear_gridbook_package_attrs()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_isolated_runtime_module(name) and \
                    _may_remove_from_sys_modules(name):
                sys.modules.pop(name, None)
        _clear_gridbook_package_attrs()
        # `before` still holds every isolated name, removable or not, so a stub
        # a test bound over a real module is undone here even though the real
        # module was never removed.
        sys.modules.update(before)
        package = sys.modules.get("gridbook")
        if package is not None:
            for name, value in package_attrs.items():
                if value is not missing:
                    setattr(package, name, value)
