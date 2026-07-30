"""Contract tests for the CB-GEMV-v2 kernel selector (``PRISMAQUANT_CB_GEMV``).

GPU-free and vLLM-free by construction: ``gridbook.moe_gemv_select`` imports
nothing but the standard library, which is the reason the policy lives there
and not in ``moe.py`` (see that module's docstring). Everything asserted here
is the part that decides WHICH kernel a served layer runs — a wrong answer is
silent (both kernels produce correct output; only speed and reassociation
differ), so it has to be pinned by test rather than by observation.
"""
import importlib.util
import pathlib

import pytest

sel = pytest.importorskip("gridbook.moe_gemv_select",
                          reason="gridbook not importable")

ENV = "PRISMAQUANT_CB_GEMV"
CONTRACT_ENV = "PRISMAQUANT_CB_DECODE_CONTRACT"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts from an unresolved selector and a clean env.

    The selector caches into module globals ON PURPOSE (resolve-once is the
    contract), so the fixture — not the test — owns the reset.
    """
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(CONTRACT_ENV, raising=False)
    monkeypatch.setattr(sel, "_CB_GEMV", None, raising=False)
    monkeypatch.setattr(sel, "_CB_GEMV_V2_OK", None, raising=False)
    yield


# --- the selector -----------------------------------------------------------

def test_mode_defaults_to_auto_when_unset():
    assert sel.cb_gemv_mode() == "auto"


@pytest.mark.parametrize("val", ["auto", "v2", "inherited"])
def test_mode_accepts_every_documented_spelling(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert sel.cb_gemv_mode() == val


@pytest.mark.parametrize("val", ["  V2 ", "AUTO", "Inherited\n"])
def test_mode_normalises_case_and_whitespace(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert sel.cb_gemv_mode() == val.strip().lower()


@pytest.mark.parametrize("val", ["v3", "true", "1", "", "smem", "v2,auto"])
def test_mode_rejects_unknown_spelling(monkeypatch, val):
    """A typo must NOT degrade silently to the inherited kernel: that would be
    an A/B whose two arms are the same kernel, reported as 'v2 buys nothing'."""
    monkeypatch.setenv(ENV, val)
    with pytest.raises(ValueError) as e:
        sel.cb_gemv_mode()
    assert ENV in str(e.value)


def test_mode_is_resolved_once_and_is_stable(monkeypatch):
    monkeypatch.setenv(ENV, "v2")
    assert sel.cb_gemv_mode() == "v2"
    assert sel.cb_gemv_mode() == "v2"          # idempotent, no raise


def test_mode_rejects_mid_process_change(monkeypatch):
    """One model can serve BOTH kernels, so a selector that moved mid-serve
    would split the dispatch and make every number from that process
    unattributable."""
    monkeypatch.setenv(ENV, "v2")
    assert sel.cb_gemv_mode() == "v2"
    monkeypatch.setenv(ENV, "inherited")
    with pytest.raises(RuntimeError) as e:
        sel.cb_gemv_mode()
    assert "mid-process" in str(e.value)


def test_mode_rejects_unset_after_resolution(monkeypatch):
    """Deleting the env is a mid-process change too (it re-defaults to auto)."""
    monkeypatch.setenv(ENV, "inherited")
    assert sel.cb_gemv_mode() == "inherited"
    monkeypatch.delenv(ENV)
    with pytest.raises(RuntimeError):
        sel.cb_gemv_mode()


# --- the decode-contract argument -------------------------------------------
# The inherited kernel resolves this itself, per launch, with a bare strcmp
# (`pq_env_is(...)` at csrc/cb_gemv.cu:471/760/1357/1437). The v2 kernel takes
# it as an argument. Under a mixed dispatch the two MUST agree, so the python
# side is a bare `== "v2"` — normalising case or whitespace here would make
# " v2" mean v2 to one kernel and v1 to the other, per weight, silently.

@pytest.mark.parametrize("raw,expect", [
    (None, 0),
    ("v1", 0),
    ("v2", 1),
    ("V2", 0),
    (" v2", 0),
    ("v2 ", 0),
    ("", 0),
])
def test_decode_contract_arg_matches_pq_env_is_exactly(monkeypatch, raw,
                                                       expect):
    if raw is None:
        monkeypatch.delenv(CONTRACT_ENV, raising=False)
    else:
        monkeypatch.setenv(CONTRACT_ENV, raw)
    assert sel.decode_contract_v2_arg() == expect


# --- the per-(layer, stack) choice ------------------------------------------

class _FakeExt:
    """Stands in for the compiled ``prismaquant_cb_v2_ext``."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def cb_gemv_v2_prefers_inherited(self, k_bits, type_size, in_features):
        self.calls.append((k_bits, type_size, in_features))
        return self.verdict


def _install_ext(monkeypatch, verdict):
    ext = _FakeExt(verdict)
    monkeypatch.setattr(sel, "cb_gemv_v2_available", lambda: True)
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")
    monkeypatch.setattr(cuda_ext, "get_ext_v2", lambda: ext)
    return ext


def _explode():
    raise AssertionError("availability must not be probed on this path")


def test_kill_switch_never_probes_the_extension(monkeypatch):
    """``inherited`` is the A/B control AND the kill switch: it must reproduce
    today's dispatch without so much as attempting the v2 JIT build."""
    monkeypatch.setenv(ENV, "inherited")
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, 2, 73, 4096)
    assert use_v2 is False
    assert "kill switch" in why


@pytest.mark.parametrize("n_sub", [1, 4])
def test_non_product_mode_rejected_before_the_probe(monkeypatch, n_sub):
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, n_sub, 4 * 16 + 9, 4096)
    assert use_v2 is False
    assert f"n_sub={n_sub}" in why


def test_wrong_type_size_rejected_before_the_probe(monkeypatch):
    """type_size != 4k+9 is not the fp4 two-tier v2 plane the kernel decodes."""
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, 2, 64, 4096)
    assert use_v2 is False
    assert "4k+9" in why


def test_unavailable_extension_falls_back(monkeypatch):
    monkeypatch.setattr(sel, "cb_gemv_v2_available", lambda: False)
    use_v2, why = sel.cb_gemv_choice(16, 2, 73, 4096)
    assert use_v2 is False
    assert "unavailable" in why


def test_verdict_is_delegated_to_the_compiled_predicate(monkeypatch):
    """The occupancy rule has exactly ONE implementation — the one compiled
    into the binary that launches. Assert we call it and honour its answer,
    with the arguments it expects, rather than re-deriving it in python."""
    monkeypatch.setenv(ENV, "auto")
    ext = _install_ext(monkeypatch, verdict=False)
    use_v2, why = sel.cb_gemv_choice(16, 2, 73, 4096)
    assert use_v2 is True and why == "mode=auto"
    assert ext.calls == [(16, 73, 4096)]


def test_predicate_veto_is_honoured(monkeypatch):
    """The k24 long-K occupancy wall is a SPEED decision — it must fall back,
    not raise."""
    monkeypatch.setenv(ENV, "v2")
    ext = _install_ext(monkeypatch, verdict=True)
    use_v2, why = sel.cb_gemv_choice(24, 2, 105, 4096)
    assert use_v2 is False
    assert "occupancy wall" in why
    assert ext.calls == [(24, 105, 4096)]


def test_explicit_v2_still_obeys_the_predicate(monkeypatch):
    """``v2`` is an A/B label, not an override: it must not force a cell onto
    the kernel the measurement says loses there."""
    monkeypatch.setenv(ENV, "v2")
    _install_ext(monkeypatch, verdict=True)
    assert sel.cb_gemv_choice(24, 2, 105, 4096)[0] is False


# --- packaging --------------------------------------------------------------

def test_kernel_source_ships_inside_the_package():
    """The source is JIT-compiled at first use, so it must be resolvable from
    an installed package, not from a repo-root path."""
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")
    d = pathlib.Path(cuda_ext.csrc_dir())
    assert (d / "cb_gemv_v2.cu").is_file(), sorted(p.name for p in d.iterdir())


def test_check_dist_requires_the_new_kernel():
    """``check_dist.py``'s REQUIRED list is the only literal floor that would
    catch this file missing from a wheel."""
    script = (pathlib.Path(__file__).resolve().parents[1]
              / ".github" / "scripts" / "check_dist.py")
    if not script.is_file():
        pytest.skip("packaging gate not present (installed-package run)")
    spec = importlib.util.spec_from_file_location("_check_dist", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "gridbook/csrc/cb_gemv_v2.cu" in mod.REQUIRED
