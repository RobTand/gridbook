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

import ast
import os
from pathlib import Path
import re
import runpy
import subprocess

import gridbook
import pytest


ROOT = Path(__file__).resolve().parents[1]

# Kept in the repo and the sdist, never in the wheel. Mirrors
# check_dist.py's SDIST_ONLY_GLOBS; the declaration test below is what keeps
# the mirror from rotting.
SDIST_ONLY_GLOBS = ("gridbook/csrc/tools/*",
                    "gridbook/csrc/cutlass_fork/*_orig.hpp")

# Source-only validation entry points. ``check_dist.py`` applies this exact
# floor to the built sdist and rejects any wheel leak; this source-level mirror
# prevents a new utility from being named in only one packaging declaration.
SDIST_REQUIRED_UTILITIES = (
    "scripts/prepare_lfm_fused_validation.py",
    "scripts/validate_fused_nvfp4_ab.py",
    "scripts/validate_fused_nvfp4_three_arm.py",
    "scripts/validate_moe_persistent_b_ab.py",
    "scripts/validate_moe_gemv_v2_ab.py",
    "scripts/validate_moe_fp8_gemv_v2_ab.py",
)

# The wheel must carry every source cuda_ext.py JIT-compiles. Same floor as
# check_dist.REQUIRED / check_installed.REQUIRED_SOURCES, expressed against the
# installed package.
WHEEL_REQUIRED = (
    "cb_gemv.cu",
    "cb_gemv_v2.cu",
    "fp8_source_w8a16.cu",
    "mxfp8_dense_gemm.cu",
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


def test_release_pinned_image_default_matches_package_version():
    """The convenience image must not silently reinstall the prior release."""

    if not _source_checkout():
        pytest.skip("container recipe is not shipped in the wheel")
    expected = f"v{gridbook.__version__}"
    dockerfile = (
        ROOT / "docker" / "Dockerfile.gridbook-pinned"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"^ARG GRIDBOOK_REF=([^\s]+)\s*$", dockerfile, flags=re.MULTILINE
    )
    assert match is not None, "pinned-image Dockerfile has no GRIDBOOK_REF"
    assert match.group(1) == expected

    container_doc = (ROOT / "docs" / "CONTAINER.md").read_text(
        encoding="utf-8"
    )
    assert f"--build-arg GRIDBOOK_REF={expected}" in container_doc
    assert f"gridbook:{expected}-pinned" in container_doc


def test_release_workflow_rejects_tags_off_master():
    """A tag alone must not authorize an off-release-branch upload."""

    if not _source_checkout():
        pytest.skip("release workflow is not shipped in the wheel")
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "fetch-depth: 0" in workflow
    assert "Tag commit must be reachable from master" in workflow
    assert 'git rev-parse "${GITHUB_REF}^{commit}"' in workflow
    assert "refs/remotes/origin/master" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert workflow.index("Tag commit must be reachable from master") < (
        workflow.index("Install build tooling")
    )
    assert "wheel_sha256: ${{ steps.meta.outputs.wheel_sha256 }}" in workflow
    assert 'hashlib.file_digest(handle, "sha256")' in workflow
    assert workflow.count("needs.build.outputs.wheel_sha256") == 3
    assert "Downloaded wheel must match the build receipt" in workflow
    assert "Downloaded wheel must match the verified build" in workflow
    assert "Release wheel must match the published build" in workflow


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


def test_native_serving_floor_mirrors_are_exactly_equal():
    """The three literal source floors must advance in the same change.

    A package-data glob catches files that exist in a checkout, but it cannot
    keep an installed-wheel probe or a future checkout's non-vacuous floor in
    sync. Normalizing and comparing the mirrors makes the documented lockstep
    relationship executable.
    """
    if not _source_checkout():
        pytest.skip("release scripts are not shipped in the wheel")

    dist = runpy.run_path(str(ROOT / ".github/scripts/check_dist.py"))
    installed = runpy.run_path(
        str(ROOT / ".github/scripts/check_installed.py")
    )
    dist_floor = {
        path.removeprefix("gridbook/") for path in dist["REQUIRED"]
    }
    installed_floor = set(installed["REQUIRED_SOURCES"])
    metadata_floor = {f"csrc/{path}" for path in WHEEL_REQUIRED}
    assert dist_floor == installed_floor == metadata_floor


def test_sdist_validation_utility_floor_matches_manifest_and_checkout():
    """Every source-only harness must be present in the built-sdist gate.

    ``check_dist.py::check_validation_utilities`` performs the artifact-level
    membership test after build. This declaration test prevents CI's checkout
    locator from masking an omitted MANIFEST entry before that build occurs.
    """

    if not _source_checkout():
        pytest.skip("release packaging declarations are not shipped in the wheel")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    dist = runpy.run_path(str(ROOT / ".github/scripts/check_dist.py"))
    assert tuple(dist["SDIST_REQUIRED_UTILITIES"]) == SDIST_REQUIRED_UTILITIES
    for utility in SDIST_REQUIRED_UTILITIES:
        assert (ROOT / utility).is_file(), f"missing source utility {utility}"
        assert f"include {utility}" in manifest, (
            f"{utility} is gated for the built sdist but MANIFEST.in omits it"
        )


def test_cpu_gate_locates_source_utilities_without_ci_environment(tmp_path):
    """The local installed-wheel gate must not rely on GITHUB_WORKSPACE.

    The real gate invokes one pytest process per staged test file. A tiny fake
    interpreter is enough to attest that its child processes receive the
    source-root locator while the gate itself runs away from the checkout.
    """
    if not _source_checkout():
        pytest.skip("the release harness is not shipped in the wheel")

    staged = tmp_path / "gbtests"
    staged.mkdir()
    (staged / "test_probe.py").write_text("# gate probe\n", encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'test -f "${GRIDBOOK_SOURCE_ROOT}/scripts/'
        'validate_fused_nvfp4_three_arm.py"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("GITHUB_WORKSPACE", None)
    environment.pop("GRIDBOOK_SOURCE_ROOT", None)
    environment["PYTHON_BIN"] = str(fake_python)
    completed = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/run_cpu_tests.sh"), str(staged)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CPU test suite OK" in completed.stdout


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


def test_suite_needs_nothing_the_wheel_does_not_declare():
    """No test or script may write safetensors: that path imports NumPy.

    ``cpu-tests`` and ``release.yml``'s ``verify`` run this suite against the
    installed wheel, in the wheel's own dependency closure. NumPy is not in it
    — deliberately (``gridbook/cb_digest.py``) — but it is on every developer
    host, so a fixture built with ``safetensors.torch.save_file`` passes
    locally and fails all four CI legs plus the release gate. That is what
    happened, and a text-level ban is not enough: the F16 fixture writer in
    ``test_codebook_digest.py`` documents the rule in prose and must stay
    legal, so the ban is on the *call*, not the word.
    """

    roots = [directory
             for directory in (Path(__file__).resolve().parent,
                               ROOT / "scripts")
             if directory.is_dir()]
    assert roots, "no suite directory to scan"

    banned: list[str] = []
    for directory in roots:
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"),
                             filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) \
                        and (node.module or "").startswith("safetensors") \
                        and any(a.name == "save_file" for a in node.names):
                    banned.append(f"{path.name}:{node.lineno}: imports it")
                elif isinstance(node, ast.Call):
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else \
                        getattr(func, "id", "")
                    if name == "save_file":
                        banned.append(f"{path.name}:{node.lineno}: calls it")
    assert not banned, (
        "safetensors' write path imports NumPy, which the wheel does not "
        "depend on; serialize the fixture directly instead (see "
        "test_codebook_digest.py::_write):\n  " + "\n  ".join(banned))
