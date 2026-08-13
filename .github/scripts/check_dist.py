#!/usr/bin/env python3
"""Packaging gate: assert the built sdist/wheel actually carry the CUDA sources.

Run locally exactly as CI runs it::

    python -m build            # writes dist/*.whl and dist/*.tar.gz
    python .github/scripts/check_dist.py dist .

Why this exists
---------------
`gridbook` ships pure-Python wheels and JIT-compiles its CUDA extensions on
first use via ``torch.utils.cpp_extension.load()``.  That only works if the
``.cu``/``.hpp`` sources are *inside the installed package*.  Older Gridbook
releases could turn an omitted source into a warning plus a slow Triton
fallback (3.52 tok/s versus ~33 tok/s for one measured 35B MoE artifact; see
docs/TROUBLESHOOTING.md and docs/BENCHMARKS.md).  The native-only serving tree
now fails closed instead: an artifact missing a serving source installs and
imports, but cannot execute the affected CB operator.  Both failure modes make
the artifact unshippable, so every check here is fatal.

Checks
------
1. Exactly one wheel and one sdist in the dist dir.
2. A hard floor of runtime-required sources is present in both.  (This floor
   cannot be satisfied vacuously -- it is a literal list, not a diff.)
3. No drift: every ``.cu``/``.cuh``/``.hpp`` that exists under
   ``<repo>/gridbook/csrc`` in the checkout is present in the sdist, and every
   one of them *except the declared sdist-only set* is present in the wheel.
   This catches a new kernel file that nobody added to the package-data globs.
   The sdist-only set (developer tools, pristine diff baselines) is asserted
   in both directions -- present in the sdist, ABSENT from the wheel -- so the
   exclusion cannot silently rot back into every user's site-packages.
4. No stray top-level ``csrc/`` **in the checkout**.  A tree that holds the
   pre-fix repo-root ``csrc/`` as well as ``gridbook/csrc`` has two copies of
   every kernel, and the root one then rots silently.  This has to be
   checked against the checkout, not the artifacts: MANIFEST.in never grafts a
   root ``csrc/``, so setuptools packages neither -- an artifact-only scan
   passes vacuously and proves nothing.  (The artifact scan is kept as well, as
   a cheap guard against a future MANIFEST.in rule that *does* graft it, but it
   is check 4b, not the check that catches a duplicate checkout source tree.)
5. Core distribution metadata is present (fatal: Name/Version/Requires-Python).
   Cosmetic-but-recommended metadata (Author-email, Project-URL, Classifier,
   Summary) is reported as a warning, not a failure -- missing PyPI polish is
   not a reason to block a merge.
6. The wheel is *publishable*, not just installable: the long description is
   the canonical public README, it carries
   none of the retired prototype/INV-2 framing, and a declared
   ``License-Expression`` is backed by an actual license file.  ``twine check``
   passes on a wheel that fails all three -- it checks that the metadata
   renders, not that it is the right metadata.
7. The repository-only validation entry points exercised by the CPU suite are
   present in the sdist and remain absent from the runtime wheel.
"""
from __future__ import annotations

import email
import fnmatch
import pathlib
import re
import sys
import tarfile
import zipfile

PKG = "gridbook"

# Runtime-required native sources, reached from gridbook/*.py:
#   cb_gemv.cu          -> cuda_ext.get_ext()            (decode GEMV; the hot one)
#   cb_gemv_v2.cu       -> cuda_ext.get_ext_v2()         (smem-resident-dict
#                                                         decode GEMV, opt-in)
#   cb_bf16_grouped_gemm.cu -> cuda_ext.get_bf16_grouped_ext()
#                            (exact native quality bridge for dense/MoE)
#   cb_fused_gemm.cu    -> cuda_ext.get_fused_ext()      (fused prefill)
#   cb_fused_fp4_gemm.cu -> cuda_ext.get_fused_fp4_ext() (fused NVFP4 prefill)
#   cb_fused_fp4v2_gemm.cu -> cuda_ext.get_fused_fp4v2_ext()
#                            (contract-preserving fused FP4-v2 quality lane)
#   cb_moe_persistent_b.cu -> cuda_ext.get_moe_persistent_b_ext()
#                            (persistent-B grouped MoE decode-in-mainloop)
# cb_fused_gemm.cu #includes the three cutlass_fork headers listed below.
# cb_fused_fp4_gemm.cu #includes sm120_cb_fused_fp4_mma.hpp; both are also
# JIT-identity inputs and therefore belong on this non-vacuous floor.
# The Gridbook-owned headers are runtime-required for the same reason: every
# one is a declared ``_*_BUILD_INPUTS`` entry, so it is both #included by a
# JIT-compiled translation unit and hashed into that module's build identity.
#   cb_grouped_common.hpp  shared grouping glue (EVT trees, smem gate, host
#                          validation) -- pulled in by ALL FOUR of
#                          get_fused_ext / get_fused_fp4_ext /
#                          get_fused_fp4v2_ext / get_bf16_grouped_ext, so its
#                          absence breaks every grouped and fused lane at once.
#   sm120_cb_fp4v2_bf16_mma.hpp  the FP4-v2 CB->BF16 decode-in-prologue
#                                mainloop fork (get_fused_fp4v2_ext).
#   sm120_bf16_expert_mma.hpp    the expert-indexed BF16 mainloop fork
#                                (get_bf16_grouped_ext, get_fused_fp4v2_ext).
# NOTE: cb_grouped_common.hpp sits at csrc/ top level with a ``.hpp`` suffix,
# which the pyproject package-data globs (csrc/*.cu, *.cuh, *.h and
# csrc/cutlass_fork/*.hpp) do not match. Listing it here makes that a named
# runtime-floor failure rather than a generic drift report.
# A serving-reachable opt-in specialization still belongs on this floor: check
# 3 (drift) would also notice it missing, but only while the file exists in the
# checkout the CI job happens to be run against. Source-only research kernels
# do not belong on the runtime floor. They are nevertheless packaged while
# present because check 3 covers every native checkout source; in particular,
# retained ``cb_persistent_tc.cu`` is not reachable from the package runtime.
REQUIRED = [
    f"{PKG}/csrc/cb_gemv.cu",
    f"{PKG}/csrc/cb_gemv_v2.cu",
    f"{PKG}/csrc/fp8_source_w8a16.cu",
    f"{PKG}/csrc/mxfp8_dense_gemm.cu",
    f"{PKG}/csrc/cb_bf16_grouped_gemm.cu",
    f"{PKG}/csrc/cb_fused_gemm.cu",
    f"{PKG}/csrc/cb_fused_fp4_gemm.cu",
    f"{PKG}/csrc/cb_fused_fp4v2_gemm.cu",
    f"{PKG}/csrc/cb_moe_persistent_b.cu",
    f"{PKG}/csrc/cb_grouped_common.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_cb_mma_tma.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_cb_fused_mma.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_cb_fused_fp4_mma.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_bf16_expert_mma.hpp",
    f"{PKG}/csrc/cutlass_fork/sm120_expert_row_broadcast.hpp",
]

# A deleted serving module can survive in setuptools' reusable ``build/lib``
# and be copied into a later wheel even though it no longer exists in the
# checkout.  This happened to the retired Triton implementation, so absence is
# a first-class artifact invariant rather than an assumption about a clean
# staging directory.
FORBIDDEN = [
    f"{PKG}/kernels.py",
    f"{PKG}/moe_autotune.py",
    f"{PKG}/moe_l2.py",
]

NATIVE_SUFFIXES = (".cu", ".cuh", ".hpp", ".h")

# Native sources that ship in the SDIST but must NOT ship in the WHEEL.
# Kept in the repo and the sdist for auditability and reproducibility; kept out
# of every user's site-packages because no runtime path loads them.
#
#   csrc/tools/     standalone main() developer binaries (smem-budget probe,
#                   toolchain probe). Their OUTPUT ships -- the smem table
#                   baked into cb_fused_gemm.cu -- the binaries do not.
#   *_orig.hpp      pristine CUTLASS copies the forks are diffed against
#                   (~67 KB). Nothing #includes them; they exist so a reviewer
#                   can see exactly what each fork changed.
#
# See docs/audits/ultraplan_perf_2026-08-01.md §4. The mechanism is the
# single-level package-data globs in pyproject.toml plus the MANIFEST.in rules
# that re-add these paths to the sdist; this list is what makes the split a
# gate rather than an accident of glob syntax.
SDIST_ONLY_GLOBS = [
    f"{PKG}/csrc/tools/*",
    f"{PKG}/csrc/cutlass_fork/*_orig.hpp",
]

# These validation entry points are source utilities rather than importable
# runtime modules, so they belong in the auditable sdist and not the wheel.
# Their CPU tests are still part of the installed-wheel gate: run_cpu_tests.sh
# locates them through GRIDBOOK_SOURCE_ROOT without putting the checkout on
# PYTHONPATH. Keep this literal floor so a newly added harness cannot ship a
# test in the sdist while silently omitting the utility that test exercises.
SDIST_REQUIRED_UTILITIES = [
    "scripts/prepare_lfm_fused_validation.py",
    "scripts/validate_fused_nvfp4_ab.py",
    "scripts/validate_fused_nvfp4_three_arm.py",
    "scripts/validate_moe_persistent_b_ab.py",
    "scripts/validate_moe_gemv_v2_ab.py",
]

_errors: list[str] = []
_warnings: list[str] = []


def err(msg: str) -> None:
    _errors.append(msg)
    print(f"FAIL  {msg}")


def warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"WARN  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def wheel_members(path: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if not n.endswith("/")]


def sdist_members(path: pathlib.Path) -> list[str]:
    """Members with the leading ``<name>-<version>/`` component stripped."""
    out = []
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/", 1)
            out.append(parts[1] if len(parts) == 2 else parts[0])
    return out


def wheel_metadata(path: pathlib.Path) -> email.message.Message:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist()
                    if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(zf.read(name))


def is_sdist_only(path: str) -> bool:
    """True for a native source that belongs in the sdist but not the wheel."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SDIST_ONLY_GLOBS)


def check_artifact(label: str, members: list[str], expected: set[str],
                   excluded: set[str] = frozenset()) -> None:
    present = set(members)

    forbidden = [path for path in FORBIDDEN if path in present]
    if forbidden:
        err(f"{label}: retired runtime modules present: {forbidden}. Remove "
            "build/ and rebuild; these modules must never ship.")
    else:
        ok(f"{label}: no retired runtime modules")

    missing = [r for r in REQUIRED if r not in present]
    if missing:
        err(f"{label}: runtime-required sources missing: {missing}")
    else:
        ok(f"{label}: all {len(REQUIRED)} runtime-required native sources present")

    drift = sorted(expected - present)
    if drift:
        err(f"{label}: native sources in the checkout but not in the artifact "
            f"(package-data globs are out of date): {drift}")
    elif expected:
        ok(f"{label}: no drift vs the checkout ({len(expected)} native files)")

    # The other direction: a source declared sdist-only must not have leaked
    # into this artifact. Without this, widening a package-data glob (or a
    # stale build/lib) silently re-ships developer tools and diff baselines.
    leaked = sorted(path for path in excluded if path in present)
    if leaked:
        err(f"{label}: sdist-only native sources packaged into the artifact "
            f"({leaked}). These are kept in the repo and the sdist for "
            f"auditability and must stay out of the wheel; check the "
            f"package-data globs in pyproject.toml (they must not recurse) "
            f"and remove a stale build/ directory.")
    elif excluded:
        ok(f"{label}: none of the {len(excluded)} sdist-only native sources "
           f"leaked in")

    # Check 4b -- see check_checkout_layout() for the one that actually catches
    # a stale repo-root csrc/. This only fires if MANIFEST.in is ever changed to
    # graft the root copy into the distribution.
    stray = sorted(n for n in present
                   if n.startswith("csrc/") and n.endswith(NATIVE_SUFFIXES))
    if stray:
        err(f"{label}: stray top-level csrc/ packaged into the artifact "
            f"({len(stray)} files, e.g. {stray[0]}). The runtime resolves "
            f"sources from {PKG}/csrc only; drop the MANIFEST.in rule that "
            f"grafts the repo-root copy.")
    else:
        ok(f"{label}: no top-level csrc/ packaged")


def check_checkout_layout(root: pathlib.Path) -> None:
    """Check 4: no stale repo-root ``csrc/`` in the checkout itself.

    This is deliberately a *checkout* check. Before the packaging fix the CUDA
    sources lived at the repo root; they now live inside the package. A tree
    holding both is a rot hazard: edits land in one copy and the build silently
    ships the other. Nothing in MANIFEST.in grafts a root ``csrc/``, so this
    condition is invisible in the built artifacts -- it can only be seen here.
    """
    stale = root / "csrc"
    if not stale.is_dir():
        ok("checkout: no stale repo-root csrc/")
        return
    files = sorted(p for p in stale.rglob("*")
                   if p.is_file() and p.suffix in NATIVE_SUFFIXES)
    if not files:
        ok(f"checkout: {stale} exists but holds no native sources")
        return
    err(f"checkout: stale repo-root csrc/ holds {len(files)} native source(s) "
        f"(e.g. {files[0].relative_to(root)}). The canonical and only runtime "
        f"location is {PKG}/csrc/. Remove the root copy: `git rm -r csrc`.")


def check_validation_utilities(wheel: pathlib.Path, sdist: pathlib.Path) -> None:
    """Gate the intentional sdist-only validation-script split."""
    wheel_present = set(wheel_members(wheel))
    sdist_present = set(sdist_members(sdist))
    missing = sorted(set(SDIST_REQUIRED_UTILITIES) - sdist_present)
    if missing:
        err(f"sdist: validation utilities missing: {missing}")
    else:
        ok(f"sdist: all {len(SDIST_REQUIRED_UTILITIES)} validation utilities "
           "present")

    leaked = sorted(set(SDIST_REQUIRED_UTILITIES) & wheel_present)
    if leaked:
        err(f"wheel: source-only validation utilities leaked in: {leaked}")
    else:
        ok("wheel: source-only validation utilities remain excluded")


def check_publishable(wheel: pathlib.Path) -> None:
    """Check 6: the wheel is fit to be *uploaded*, not merely fit to install.

    Two defects got as far as a staged, ``twine check``-clean artifact and were
    caught only by a human reading the METADATA by hand:

    * **The long description was the wrong README.** Releases must be built
      from this canonical repository, whose README is the public one. This
      check enforces that the built metadata contains that document rather
      than an unrelated or stale checkout's README.
    * **No license file.** ``License-Expression: Apache-2.0`` is only a
      *declaration*; the Apache-2.0 text itself is carried by the
      ``license-files`` mechanism, and a tree without a LICENSE at the root
      still builds and still passes ``twine check``.

    ``twine check`` validates that the metadata *parses and renders*. It says
    nothing about whether the rendered text is the right text. Hence this.
    """
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        md = email.message_from_bytes(
            zf.read(next(n for n in names
                         if n.endswith(".dist-info/METADATA"))))
    body = md.get_payload() or ""

    # -- the long description is the public README ---------------------------
    first = next((ln for ln in body.strip().splitlines() if ln.strip()), "")
    if not body.strip():
        err("wheel METADATA has an empty long description (no readme reached "
            "the build)")
    elif first.strip() != f"# {PKG}":
        err(f"wheel long description does not start with '# {PKG}' (got "
            f"{first.strip()!r}). Build releases only from the canonical "
            f"Gridbook repository.")
    else:
        ok(f"long description is the public README (starts '# {PKG}')")

    # Retired framing that must never reach the PyPI page. These are the exact
    # strings from the pre-rename internal README.
    stale = [s for s in ("vllm-prismaquant", "prototype (i)", "INV-2")
             if s in body]
    if stale:
        err(f"wheel long description contains retired framing {stale} -- the "
            f"Triton prototype/INV-2 language was superseded by the "
            f"native-only CUDA/CUTLASS serving contract and must not ship as "
            f"the PyPI page.")
    else:
        ok("long description carries no retired prototype/INV-2 framing")

    # -- the license text is actually in the wheel ---------------------------
    licenses = [n for n in names if ".dist-info/licenses/" in n]
    if md.get("License-Expression") and not licenses:
        err(f"wheel declares License-Expression "
            f"{md.get('License-Expression')!r} but carries no license file "
            f"(no .dist-info/licenses/). Add LICENSE at the repo root -- "
            f"setuptools' default license-files globs pick it up.")
    elif licenses:
        ok(f"license text shipped: {licenses} "
           f"(License-File: {md.get_all('License-File')})")


def check_release_metadata(root: pathlib.Path, version: str) -> None:
    """Static citation/changelog identity must match the built distribution."""
    citation = root / "CITATION.cff"
    changelog = root / "CHANGELOG.md"
    if not citation.is_file():
        err("checkout has no CITATION.cff")
    else:
        match = re.search(
            r'^version:\s*["\']([^"\']+)["\']\s*$',
            citation.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match is None:
            err("CITATION.cff has no parseable version field")
        elif match.group(1) != version:
            err(f"CITATION.cff version {match.group(1)!r} != built version "
                f"{version!r}")
        else:
            ok(f"CITATION.cff version matches built version {version}")
    if not changelog.is_file():
        err("checkout has no CHANGELOG.md")
    elif not re.search(
        rf"^## {re.escape(version)}(?:\s|$)",
        changelog.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    ):
        err(f"CHANGELOG.md has no release heading for {version}")
    else:
        ok(f"CHANGELOG.md has release heading for {version}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: check_dist.py <dist-dir> <repo-root>", file=sys.stderr)
        return 2
    dist = pathlib.Path(sys.argv[1])
    root = pathlib.Path(sys.argv[2])

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1:
        err(f"expected exactly 1 wheel in {dist}, found {[w.name for w in wheels]}")
    if len(sdists) != 1:
        err(f"expected exactly 1 sdist in {dist}, found {[s.name for s in sdists]}")
    if _errors:
        return 1
    wheel, sdist = wheels[0], sdists[0]
    ok(f"artifacts: {wheel.name} + {sdist.name}")

    # Expected native-file set, derived from the checkout so it cannot drift,
    # then split: the sdist carries the whole tree, the wheel carries the
    # runtime subset (see SDIST_ONLY_GLOBS).
    csrc = root / PKG / "csrc"
    expected: set[str] = set()
    if csrc.is_dir():
        for p in sorted(csrc.rglob("*")):
            if p.is_file() and p.suffix in NATIVE_SUFFIXES:
                expected.add(str(p.relative_to(root)))
    else:
        err(f"{csrc} does not exist. The CUDA sources must live inside the "
            f"package -- a repo-root csrc/ cannot be resolved from "
            f"site-packages under a non-editable install.")
    sdist_only = {path for path in expected if is_sdist_only(path)}
    if any(path in REQUIRED for path in sdist_only):
        err(f"a runtime-required source is declared sdist-only: "
            f"{sorted(set(REQUIRED) & sdist_only)}. The wheel must carry every "
            f"source the runtime JIT-compiles.")

    check_checkout_layout(root)
    check_artifact("wheel", wheel_members(wheel), expected - sdist_only,
                   excluded=sdist_only)
    check_artifact("sdist", sdist_members(sdist), expected)
    check_validation_utilities(wheel, sdist)

    md = wheel_metadata(wheel)
    for field in ("Name", "Version", "Requires-Python"):
        if not md.get(field):
            err(f"wheel METADATA has no {field}")
    if not _errors:
        ok(f"metadata: {md.get('Name')} {md.get('Version')} "
           f"(requires-python {md.get('Requires-Python')})")
    if md.get("Version"):
        check_release_metadata(root, md["Version"])
    # Recommended, not load-bearing -- warn only.
    for field in ("Summary", "Author-email", "Project-URL", "Classifier"):
        if not md.get_all(field):
            warn(f"wheel METADATA has no {field} (PyPI page will be bare)")

    check_publishable(wheel)

    print()
    if _errors:
        print(f"{len(_errors)} error(s), {len(_warnings)} warning(s)")
        return 1
    print(f"PASS ({len(_warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
