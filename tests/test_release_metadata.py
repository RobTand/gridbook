"""Release metadata has one version identity across human-facing files.

This module also gates the *packaging split*: which native sources ship in the
wheel and which are kept in the repository and the sdist only (developer tools
under ``csrc/tools/``, the pristine ``cutlass_fork/*_orig.hpp`` diff baselines).
See ``docs/audits/ultraplan_perf_2026-08-01.md`` §4.

The two halves run in different environments *by construction*, which is the
point: ``.github/scripts/check_dist.py`` inspects built artifacts before they
are published, while these tests run in both places the suite runs --

* in a source checkout, the declaration test asserts the three files that must
  agree (``pyproject.toml``, ``MANIFEST.in``, ``check_dist.py``) actually do,
  because a split declared in only two of them silently half-works;
* against an INSTALLED wheel -- which is exactly how CI's ``cpu-tests`` job
  runs the suite, from outside the checkout -- the contents test asserts the
  serving sources arrived and the sdist-only ones did not.
"""

from pathlib import Path
import re

import gridbook
import pytest


ROOT = Path(__file__).resolve().parents[1]

# Kept in the repo and the sdist, never in the wheel. Mirrors
# check_dist.py's SDIST_ONLY_GLOBS; the declaration test below is what keeps
# the mirror from rotting.
SDIST_ONLY_GLOBS = ("gridbook/csrc/tools/*",
                    "gridbook/csrc/cutlass_fork/*_orig.hpp")

# The wheel must carry every source cuda_ext.py JIT-compiles. Same floor as
# check_dist.REQUIRED / check_installed.REQUIRED_SOURCES, expressed against the
# installed package.
WHEEL_REQUIRED = (
    "cb_gemv.cu",
    "cb_gemv_v2.cu",
    "cb_bf16_grouped_gemm.cu",
    "cb_fused_gemm.cu",
    "cb_fused_fp4_gemm.cu",
    "cb_fused_fp4v2_gemm.cu",
    "cb_moe_persistent_b.cu",
    # Gridbook-owned headers. Each is a declared ``_*_BUILD_INPUTS`` entry, so
    # it is #included by a JIT-compiled translation unit AND hashed into that
    # module's build identity; a wheel without it cannot build the lane.
    # cb_grouped_common.hpp is shared by all four fused/grouped loaders.
    "cb_grouped_common.hpp",
    "cutlass_fork/sm120_cb_mma_tma.hpp",
    "cutlass_fork/sm120_cb_fused_mma.hpp",
    "cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    "cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp",
    "cutlass_fork/sm120_bf16_expert_mma.hpp",
    "cutlass_fork/sm120_expert_row_broadcast.hpp",
)


def _source_checkout() -> bool:
    """True when the suite runs from the repository rather than a wheel."""
    return (ROOT / "CITATION.cff").is_file()


def test_citation_and_changelog_match_package_version():
    if not _source_checkout():
        pytest.skip("source-tree release metadata is not shipped in the wheel")
    version = gridbook.__version__
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*["\']([^"\']+)["\']\s*$', citation,
                      flags=re.MULTILINE)
    assert match is not None, "CITATION.cff has no parseable version field"
    assert match.group(1) == version

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog,
                     flags=re.MULTILINE), (
        f"CHANGELOG.md has no release heading for {version}")


def test_sdist_only_split_is_declared_in_all_three_places():
    """A one-sided exclusion is the failure mode this catches.

    Dropping a path from the wheel takes three coordinated declarations, and
    any one alone produces a plausible-looking tree that is wrong:

    * ``pyproject.toml`` ``exclude-package-data`` -- takes it out of the wheel.
      Both the package key and the namespace-package key are required, because
      ``packages.find`` defaults to ``namespaces = true`` and setuptools then
      attributes MANIFEST.in files to the *sub*package.
    * ``MANIFEST.in`` -- puts it back in the sdist, which the exclusion above
      would otherwise strip as well.
    * ``check_dist.py`` ``SDIST_ONLY_GLOBS`` -- gates both directions on the
      built artifacts, so the split cannot silently reverse.
    """
    if not _source_checkout():
        pytest.skip("packaging declarations are not shipped in the wheel")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    gate = (ROOT / ".github/scripts/check_dist.py").read_text(encoding="utf-8")

    for glob in SDIST_ONLY_GLOBS:
        in_package = glob.split("/", 1)[1]          # csrc/tools/*
        assert f'"{in_package}"' in pyproject, (
            f"{glob} is not excluded from the wheel: add {in_package!r} to "
            f"[tool.setuptools.exclude-package-data]")
        directory = glob.rsplit("/", 1)[0]          # gridbook/csrc/tools
        assert directory in manifest, (
            f"{glob} is excluded from the wheel but no MANIFEST.in rule puts "
            f"it in the sdist -- it would vanish from the release entirely")
        assert glob.replace("gridbook", "{PKG}") in gate, (
            f"{glob} is not in check_dist.py's SDIST_ONLY_GLOBS, so nothing "
            f"gates the split on the built artifacts")

    # The mirror only helps if the two lists are the same length.
    declared = re.search(r"SDIST_ONLY_GLOBS = \[(.*?)\]", gate, flags=re.S)
    assert declared is not None, "check_dist.py has no SDIST_ONLY_GLOBS list"
    assert declared.group(1).count("f\"{PKG}/") == len(SDIST_ONLY_GLOBS)


def test_installed_wheel_carries_serving_sources_and_no_sdist_only_ones():
    """The release gate, observed from inside an installed wheel.

    ``check_dist.py`` proves this about the artifact; this proves it about the
    thing that actually got installed, which is the failure mode that shipped
    once already (sources resolved from a path the wheel never populated).
    """
    if _source_checkout():
        pytest.skip("a checkout holds the full source tree by design")
    cuda_ext = pytest.importorskip("gridbook.cuda_ext",
                                   reason="gridbook not importable")
    csrc = Path(cuda_ext.csrc_dir())

    missing = [name for name in WHEEL_REQUIRED if not (csrc / name).is_file()]
    assert not missing, (
        f"installed wheel is missing runtime-required native sources: "
        f"{missing} (looked under {csrc})")

    assert not (csrc / "tools").exists(), (
        f"{csrc}/tools is sdist-only and must not be installed; a package-data "
        f"glob widened or a stale build/ was reused")
    orig = sorted(p.name for p in (csrc / "cutlass_fork").glob("*_orig.hpp"))
    assert not orig, (
        f"pristine CUTLASS diff baselines {orig} are sdist-only and must not "
        f"be installed")
