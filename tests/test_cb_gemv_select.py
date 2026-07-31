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
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

sel = pytest.importorskip("gridbook.moe_gemv_select",
                          reason="gridbook not importable")

ENV = "PRISMAQUANT_CB_GEMV"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts from an unresolved selector and a clean env.

    The selector caches into module globals ON PURPOSE (resolve-once is the
    contract), so the fixture — not the test — owns the reset.
    """
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(sel, "_CB_GEMV", None, raising=False)
    monkeypatch.setattr(sel, "_CB_GEMV_V2_OK", {}, raising=False)
    monkeypatch.setattr(sel, "_CB_GEMV_V2_WARNED", set(), raising=False)
    yield


# --- the selector -----------------------------------------------------------

def test_mode_defaults_to_inherited_when_unset():
    assert sel.cb_gemv_mode() == "inherited"


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
    """Deleting an explicit opt-in reverts to inherited and must be rejected."""
    monkeypatch.setenv(ENV, "v2")
    assert sel.cb_gemv_mode() == "v2"
    monkeypatch.delenv(ENV)
    with pytest.raises(RuntimeError):
        sel.cb_gemv_mode()


# --- device support / preparation ------------------------------------------

def _fake_cuda_device(monkeypatch, capability=(12, 1), optin=101376):
    import torch
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda index: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(
                            shared_memory_per_block_optin=optin))


def test_device_support_accepts_validated_gb10(monkeypatch):
    _fake_cuda_device(monkeypatch)
    ok, why, index = sel.cb_gemv_v2_device_support("cuda:0")
    assert ok is True and index == 0
    assert "(12, 1)" in why and "101376" in why


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (12, 2)])
def test_device_support_rejects_unvalidated_capability(monkeypatch, capability):
    _fake_cuda_device(monkeypatch, capability=capability)
    ok, why, index = sel.cb_gemv_v2_device_support("cuda:0")
    assert ok is False and index == 0
    assert "outside the supported" in why


def test_device_support_rejects_insufficient_optin_smem(monkeypatch):
    _fake_cuda_device(monkeypatch, optin=96 * 1024)
    ok, why, index = sel.cb_gemv_v2_device_support("cuda:0")
    assert ok is False and index == 0
    assert "below the required" in why


def test_unsupported_device_never_builds(monkeypatch):
    monkeypatch.setattr(sel, "cb_gemv_v2_device_support",
                        lambda device=None: (False, "unsupported test GPU", 3))
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")
    monkeypatch.setattr(cuda_ext, "get_ext_v2", _explode)
    assert sel.cb_gemv_v2_available("cuda:3") is False


def test_availability_prepares_once_per_device_thread_safely(monkeypatch):
    import torch
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")

    class PreparedExt:
        def __init__(self):
            self.prepares = 0

        def cb_gemv_v2_prepare(self):
            self.prepares += 1

    ext = PreparedExt()
    monkeypatch.setattr(
        sel, "cb_gemv_v2_device_support",
        lambda device=None: (True, "test GB10", int(str(device).split(":")[-1])))
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(cuda_ext, "get_ext_v2", lambda: ext)

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda _: sel.cb_gemv_v2_available("cuda:0"),
                            range(16)))
    assert all(got)
    assert ext.prepares == 1
    assert sel.cb_gemv_v2_available("cuda:1") is True
    assert ext.prepares == 2


def test_v2_loader_contract_requires_prepare_symbol():
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")
    mod = SimpleNamespace(
        cb_gemv_v2=object(), cb_gemv_v2_prefers_inherited=object(),
        cb_expand_v2=object(), __file__="/tmp/stale-v2.so")
    with pytest.raises(cuda_ext.StaleExtensionError) as exc:
        cuda_ext._require_symbols(
            mod, cuda_ext._V2_SYMBOLS, build_dir="/tmp/v2-cache",
            source="cb_gemv_v2.cu")
    assert "cb_gemv_v2_prepare" in str(exc.value)
    assert "/tmp/v2-cache" in str(exc.value)


def test_decode_contract_is_resolved_in_cpp_per_call():
    """Do not regress to a Python/load-time cached contract while inherited
    resolves the same env switch in its C++ launcher on every call."""
    # The CPU matrix runs a copied test directory against the installed wheel,
    # not against a checkout. Resolve the shipped source through the same
    # package-resource API the JIT loader uses so this pins both the contract
    # and wheel completeness without assuming a repo-relative path.
    cuda_ext = pytest.importorskip("gridbook.cuda_ext")
    source = (pathlib.Path(cuda_ext.csrc_dir()) /
              "cb_gemv_v2.cu").read_text()
    assert ('pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2")'
            in source)
    assert "decode_contract_v2_arg" not in source


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
    monkeypatch.setattr(sel, "cb_gemv_v2_available", lambda device=None: True)
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


def test_unset_default_never_probes_or_builds(monkeypatch):
    """The regression guard for the submitted rollout: an absent variable is
    the inherited path, not an implicit experimental ``auto`` opt-in."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, 2, 73, 4096, "cuda:0")
    assert use_v2 is False
    assert "kill switch" in why


@pytest.mark.parametrize("n_sub", [1, 4])
def test_non_product_mode_rejected_before_the_probe(monkeypatch, n_sub):
    monkeypatch.setenv(ENV, "v2")
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, n_sub, 4 * 16 + 9, 4096)
    assert use_v2 is False
    assert f"n_sub={n_sub}" in why


def test_wrong_type_size_rejected_before_the_probe(monkeypatch):
    """type_size != 4k+9 is not the fp4 two-tier v2 plane the kernel decodes."""
    monkeypatch.setenv(ENV, "v2")
    monkeypatch.setattr(sel, "cb_gemv_v2_available", _explode)
    use_v2, why = sel.cb_gemv_choice(16, 2, 64, 4096)
    assert use_v2 is False
    assert "4k+9" in why


def test_unavailable_extension_falls_back(monkeypatch):
    monkeypatch.setenv(ENV, "v2")
    monkeypatch.setattr(sel, "cb_gemv_v2_available",
                        lambda device=None: False)
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


def test_choice_passes_layer_device_to_support_probe(monkeypatch):
    monkeypatch.setenv(ENV, "v2")
    seen = []
    monkeypatch.setattr(
        sel, "cb_gemv_v2_available",
        lambda device=None: seen.append(device) is None and False)
    assert sel.cb_gemv_choice(16, 2, 73, 4096, "cuda:7")[0] is False
    assert seen == ["cuda:7"]


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
