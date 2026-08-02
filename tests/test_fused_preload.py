"""Residency-matched preload contract for Gridbook's native CUDA extensions.

``PRISMAQUANT_PRELOAD_FUSED=1`` has to warm the FULL extension inventory, not a
subset: an arm that loads only the two fused modules is still residency-
MISmatched against an arm that later touches the GEMV, v2, grouped-BF16 or
persistent-B modules, and that mismatch is the ±17% measurement-arithmetic
confound (2026-08-01 performance audit §3 P4, "preload completeness").
"""

import pytest

from gridbook import cuda_ext

# family -> loader attribute on ``cuda_ext``. Spelled out here rather than read
# back from the implementation on purpose: this file is an INDEPENDENT
# statement of the inventory, so dropping a loader from the warm-up has to fail
# a test instead of silently agreeing with itself.
LOADERS = {
    "gemv": "get_ext",                                # cb_gemv.cu
    "gemv_v2": "get_ext_v2",                          # cb_gemv_v2.cu
    "bf16_grouped": "get_bf16_grouped_ext",           # cb_bf16_grouped_gemm.cu
    "fp8": "get_fused_ext",                           # cb_fused_gemm.cu
    "fp4": "get_fused_fp4_ext",                       # cb_fused_fp4_gemm.cu
    "fp4v2": "get_fused_fp4v2_ext",                   # cb_fused_fp4v2_gemm.cu
    "moe_persistent_b": "get_moe_persistent_b_ext",   # cb_moe_persistent_b.cu
}


def _patch_all(monkeypatch, calls, *, resident=(), raises=()):
    """Replace EVERY loader with a recording stub.

    All seven must be stubbed in every test: the warm-up really does call all
    of them now, so an unpatched loader would start a real multi-minute nvcc
    build inside the test run. Any family the implementation has grown past
    this file is stubbed too, so inventory drift is reported by the coverage
    assertions instead of compiling CUDA in the middle of a unit test.
    """
    inventory = dict(LOADERS)
    inventory.update(dict(cuda_ext._PRELOAD_FAMILIES))
    for family, attr in inventory.items():
        def stub(family=family):
            calls.append(family)
            if family in raises:
                raise RuntimeError(
                    "one failed loader must not suppress the others")
            return object() if family in resident else None
        monkeypatch.setattr(cuda_ext, attr, stub)


def test_registry_names_exactly_the_seven_loader_families():
    """The warm-up inventory is all seven modules — no more, no fewer."""
    assert dict(cuda_ext._PRELOAD_FAMILIES) == LOADERS
    assert len(cuda_ext._PRELOAD_FAMILIES) == len(LOADERS)  # no duplicate keys
    for attr in LOADERS.values():
        assert callable(getattr(cuda_ext, attr))


def test_preload_attempts_every_loader_family(monkeypatch):
    """All seven independent JIT modules are warmed, each exactly once."""
    calls = []
    _patch_all(monkeypatch, calls, resident=set(LOADERS))

    status = cuda_ext.preload_native_extensions()

    assert sorted(calls) == sorted(LOADERS)
    assert status == {family: True for family in LOADERS}


def test_one_raising_loader_does_not_suppress_the_others(monkeypatch):
    calls = []
    _patch_all(monkeypatch, calls, resident={"gemv"}, raises={"fp8"})

    status = cuda_ext.preload_native_extensions()

    # Every family attempted despite the raise, an earlier success kept, and a
    # loader that returned None reported as not resident.
    assert sorted(calls) == sorted(LOADERS)
    assert status["gemv"] is True
    assert status["fp8"] is False
    assert status["fp4"] is False
    assert set(status) == set(LOADERS)


def test_strict_reports_every_failure_after_every_attempt(monkeypatch):
    calls = []
    _patch_all(monkeypatch, calls, raises={"fp8"})

    with pytest.raises(RuntimeError) as excinfo:
        cuda_ext.preload_native_extensions(strict=True)

    message = str(excinfo.value)
    assert "fp8=" in message                    # the raised error, verbatim
    assert "fp4=unavailable" in message         # loaded to None, no exception
    for family in LOADERS:
        assert f"{family}=" in message
    # Raised only AFTER every attempt ran — strict mode must not short-circuit
    # the warm-up it exists to verify.
    assert sorted(calls) == sorted(LOADERS)


def test_strict_accepts_a_fully_resident_process(monkeypatch):
    calls = []
    _patch_all(monkeypatch, calls, resident=set(LOADERS))

    status = cuda_ext.preload_native_extensions(strict=True)

    assert status == {family: True for family in LOADERS}
    assert sorted(calls) == sorted(LOADERS)


def test_published_fused_name_still_warms_everything(monkeypatch):
    """``preload_fused_extensions`` is documented; it must keep working."""
    calls = []
    _patch_all(monkeypatch, calls, raises={"fp8"})

    status = cuda_ext.preload_fused_extensions()

    assert sorted(calls) == sorted(LOADERS)
    assert status == {family: False for family in LOADERS}

    calls.clear()
    try:
        cuda_ext.preload_fused_extensions(strict=True)
    except RuntimeError as exc:
        assert "fp8=" in str(exc)
        assert "fp4=unavailable" in str(exc)
    else:  # pragma: no cover - strict mode must fail closed
        raise AssertionError("strict preload accepted unavailable extensions")
    assert sorted(calls) == sorted(LOADERS)


def test_every_registered_family_name_resolves():
    """A registry entry naming a loader this module lacks is a DEFECT.

    ``_PRELOAD_FAMILIES`` holds loaders BY NAME so a lane can register from
    inside its own additive block. The cost of that indirection is that a
    typo or a rename produces a whole module silently absent from every
    residency match, reported identically to a Blackwell-only lane on an Ada
    box — so it is asserted here rather than discovered in a bench report.
    """
    unresolved = [f"{family} -> {name}"
                  for family, name in cuda_ext._PRELOAD_FAMILIES
                  if not callable(getattr(cuda_ext, name, None))]
    assert not unresolved, (
        f"the preload registry names loaders cuda_ext does not define: "
        f"{unresolved}")


def test_a_registry_defect_is_reported_as_a_defect_not_as_unavailable(
        monkeypatch, capsys):
    """The two failure modes must be distinguishable in one glance."""
    calls = []
    _patch_all(monkeypatch, calls, resident=tuple(LOADERS))
    monkeypatch.setattr(
        cuda_ext, "_PRELOAD_FAMILIES",
        list(cuda_ext._PRELOAD_FAMILIES) + [("ghost", "get_ghost_ext")])

    status = cuda_ext.preload_native_extensions()

    assert status["ghost"] is False
    error = capsys.readouterr().err
    assert "preload registry names loaders" in error
    assert "ghost -> get_ghost_ext" in error
    assert "This is a Gridbook defect, not a property of this machine" in error
    # And every real family was still warmed: one broken entry must not skip
    # the rest, or the residency match is quietly partial again.
    assert sorted(calls) == sorted(LOADERS)


def test_a_partial_warm_up_is_never_silent(monkeypatch, capsys):
    """plugin.register discards the status, so the loader must say so itself.

    A half-warmed process is precisely the ±17% measurement-arithmetic
    confound PRISMAQUANT_PRELOAD_FUSED exists to remove, and it is invisible
    in the results — it shows up only as a residency mismatch between arms.
    """
    calls = []
    _patch_all(monkeypatch, calls, resident=("gemv", "gemv_v2", "fp8"))

    status = cuda_ext.preload_native_extensions()

    assert status["fp4"] is False and status["gemv"] is True
    error = capsys.readouterr().err
    assert "did not warm every family" in error
    assert "fp4=unavailable" in error
    assert "only comparable if BOTH warm the same set" in error
