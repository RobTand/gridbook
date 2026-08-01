# Releasing gridbook

How to cut a release, what the automation does, and the one-time PyPI setup that
only the maintainer can do.

Companion docs: [`DISTRIBUTION.md`](DISTRIBUTION.md) holds the *why* (channel
ranking, naming decision, the full pre-flight checklist). This file is the *how*.

---

## 0. The irreversible parts, first

- **A PyPI version can never be reused.** Not after yanking, not after deleting
  the release, not after deleting the project. If `0.1.0` goes up broken, the
  next release is `0.1.1` forever.
- **Deleting a PyPI *project* frees the name for anyone else to take.** If a
  release is bad, **yank** it (Manage → release → Options → Yank). Yanking hides
  it from resolvers while leaving explicit pins working.
- **A "pending" Trusted Publisher does not reserve the name.** The name is only
  claimed by a successful upload.

Rehearse on TestPyPI. `release.yml` routes any PEP 440 pre-release version
(`v0.1.0rc1`, `v0.1.0a1`, …) to TestPyPI automatically, precisely so the first
real version number is not spent on a rehearsal.

---

## 1. One-time setup (maintainer only)

Do this **once**, and only after the pre-flight checklist in
[`DISTRIBUTION.md` §3](DISTRIBUTION.md) is green.

### 1.1 GitHub environments

`https://github.com/RobTand/gridbook/settings/environments` → **New environment**.
Create exactly two. **No secrets go in either** — with Trusted Publishing there
is nothing to store.

| Environment name | Used by |
|---|---|
| `pypi` | final releases |
| `testpypi` | pre-release tags |

Recommended: add yourself as a **required reviewer** on `pypi`. That puts a
human approval click in front of the single irreversible step in the whole
pipeline, and it costs nothing.

### 1.2 PyPI pending publisher

<https://pypi.org/manage/account/publishing/> → *Add a new pending publisher* →
**GitHub** tab. Type these **exactly**:

| Form field | Value |
|---|---|
| PyPI Project Name | `gridbook` |
| Owner | `RobTand` |
| Repository name | `gridbook` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

### 1.3 TestPyPI pending publisher

<https://test.pypi.org/manage/account/publishing/> — identical values, except:

| Form field | Value |
|---|---|
| Environment name | `testpypi` |

TestPyPI is a separate instance with a separate account.

> A wrong value in either form does **not** fail loudly. It fails as a rejected
> OIDC claim at publish time, which is a confusing place to find a typo. The
> "Workflow name" field wants the release workflow's **filename**, which is
> `release.yml`.

After the first successful upload the *pending* publisher becomes a normal
publisher on the project. Nothing further to do.

---

## 2. Cutting a release

### 2.0 Source ownership — release this repository directly

This repository is the **only** source tree for the Gridbook Python runtime,
CUDA sources, runtime tests, package metadata, and releases. PrismaQuant owns
the producer/exporter and validates compatibility against an immutable
Gridbook commit; it does not contain or synchronize a copy of this package.

That boundary is deliberate. A runtime change is reviewed, tested, packaged,
and tagged here exactly once. Cross-project compatibility is expressed through
the packaged runtime contract and exact external pins, never by copying files.

Before release, require a clean checkout and confirm the distribution/layout
gates from this repository itself:

```bash
git status --short
python -m pytest tests/ -q
python -m build
python .github/scripts/check_dist.py dist .
```

### 2.1 Bump the version — one place

The version has a **single source of truth**: `__version__` in
`gridbook/__init__.py`. `pyproject.toml` declares `dynamic = ["version"]` and
reads it from there, so there is nothing else to edit.

Bump it **in this repository**, add the corresponding `CHANGELOG.md` entry, and
update static release metadata such as `CITATION.cff` in the same PR. A
PrismaQuant compatibility pin is updated only after this PR merges, using the
resulting immutable commit.

### 2.2 Pre-tag gate that CI cannot run

CI has no GPU and no CUDA toolkit, so **it cannot prove the extension compiles**.
The packaging checks in CI pass happily on a wheel whose source resolution is
still wrong — only an actual compile from the *installed* package closes that
loop. Run this by hand before tagging, on a supported Blackwell GPU with `nvcc`.
The loader queries the concrete device capability and the fused gate executes
GPU operators, so GPU passthrough is mandatory:

```bash
python -m build
docker run --rm --gpus all -v "$PWD/dist:/dist" \
  --entrypoint bash <your-vllm-image> -c '
  pip install --no-deps -q /dist/gridbook-*.whl
  cd /                      # never run this from the repo: it would shadow the install
  TORCH_CUDA_ARCH_LIST=12.1 python -c "
from gridbook.cuda_ext import (
    csrc_dir, get_bf16_grouped_ext, get_ext, get_ext_v2,
    get_fused_ext, get_fused_fp4_ext,
)
from gridbook.native_cutlass import (
    require_native_fp8_cutlass, require_native_moe_activation,
)
print(\"csrc:\", csrc_dir())
m = get_ext()
assert m is not None, \"main extension failed to build\"
assert hasattr(m, \"cb_gemv_fp8\"), \"built module is missing cb_gemv_fp8\"
v2 = get_ext_v2()
assert v2 is not None, \"FP4-v2 extension failed to build\"
for symbol in (\"cb_gemv_v2\", \"cb_expand_v2\"):
    assert hasattr(v2, symbol), f\"built v2 module is missing {symbol}\"
grouped = get_bf16_grouped_ext()
assert grouped is not None, \"grouped BF16 CUTLASS extension failed to build\"
for symbol in (\"cb_bf16_grouped_mm\", \"cb_bf16_grouped_mm_out\"):
    assert hasattr(grouped, symbol), f\"built grouped module is missing {symbol}\"
fp8_fused = get_fused_ext()
assert fp8_fused is not None, \"fused FP8-CB extension failed to build\"
for symbol in (\"cb_fused_prefill_mm_scaled\", \"cb_fused_moe_grouped\"):
    assert hasattr(fp8_fused, symbol), f\"built FP8 fused module is missing {symbol}\"
fp4_fused = get_fused_fp4_ext()
assert fp4_fused is not None, \"fused NVFP4-CB extension failed to build\"
for symbol in (\"cb_fused_fp4_prefill_mm_scaled\",
               \"cb_fused_fp4_moe_grouped\",
               \"cb_fused_fp4_moe_tile_sizes\"):
    assert hasattr(fp4_fused, symbol), f\"built FP4 fused module is missing {symbol}\"
assert list(fp4_fused.cb_fused_fp4_moe_tile_sizes()) == [128, 256]
require_native_fp8_cutlass(\"release native-op gate\")
require_native_moe_activation(\"silu\", \"release native-op gate\")
print(\"ok:\", m, v2, grouped, fp8_fused, fp4_fused)"'
```

This command compiles every serving-reachable packaged extension: the main and
v2 decode/expansion modules, the required exact grouped BF16 CUTLASS quality
bridge, and both optional fused specializations. It also attests the direct
compiled vLLM FP8/CUTLASS and activation ABI that Gridbook invokes without a
Python fallback helper. The retained `cb_persistent_tc.cu` source is not part
of this gate: its serving selector, custom op, and package loader were deleted,
and it remains available only to the explicitly opted-in research test.

The fused FP4 extension also requires the GPU operator/SASS suite from an
installed wheel. Stage `tests/test_fused_fp4_prefill.py` outside the checkout,
install the matching PrismaQuant producer only as a test fixture, and run it on
the release GPU image. The release gate is 0 failures, 0 skips,
`OMMA.SF.16864` present in both fused symbols, and no `QMMA` in either symbol.
Both fused runtime flags remain default-off unless the served quality gate
separately promotes them.

If the command prints all five extension modules and passes the native-op
attestation, the installed wheel's native serving floor is genuinely sound.
Two ways it can fail, and they mean different things:

- `[prismaquant-cb] ERROR: broken gridbook install — …` → the packaged sources
  are missing. **Do not tag.** This is a packaging defect.
- `[prismaquant-cb] WARNING: … extension unavailable …` → no usable `nvcc` in
  that image. Fix the image and re-run; this gate has not been passed.

For a required module, either failure means a release cut cannot serve the
corresponding Gridbook format: the call fails closed with a native-extension
diagnostic. Gridbook has no Triton dependency or Triton fallback for CB
operators; vLLM may still ship or use Triton for components Gridbook does not
own. Do not tag.

The full serve smoke (real artifact, decode throughput vs `docs/BENCHMARKS.md`)
is the other GPU-only gate — see [`DISTRIBUTION.md` §3.4](DISTRIBUTION.md).

### 2.3 Tag and push

Tags are `v` + the PEP 440 version. The workflow refuses to publish if the tag
and the built version disagree.

```bash
git commit -am "release 0.1.0rc1"
git push
# wait for CI to go green on master, then:
git tag v0.1.0rc1
git push origin v0.1.0rc1     # <- this, and only this, starts a release
```

**Push the tag only. Do not create the GitHub Release by hand** — `release.yml`
creates it, and a manually-created release of the same tag makes the workflow's
`gh release create` step fail.

Order of operations for a first launch:

1. `v0.1.0rc1` → TestPyPI. Verify from a clean venv on a machine with nvcc:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ gridbook
   ```
   then re-run the compile check from §2.2 against the *downloaded* wheel.
2. `v0.1.0` → PyPI. The name is claimed at that moment.

### 2.4 After the release

- PyPI project page: summary, author, links and classifiers render (they come
  from `pyproject.toml`); the README long-description renders without broken
  relative links.
- `pip install gridbook` in a clean venv on a serving box, then the §2.2 compile
  check against the installed package.
- The GitHub Release carries both `dist/*.whl` and `dist/*.tar.gz`.
- `README.md`'s install instructions are now true — if they were hedged
  ("working-title name"), unhedge them.

---

## 3. What the automation does

### `ci.yml` — every push to `master`, every PR, manual dispatch

| Job | What it proves |
|---|---|
| `build` | `python -m build` (sdist, then wheel **from the sdist**), `twine check --strict`, and `.github/scripts/check_dist.py` |
| `install` (3.10 / 3.11 / 3.12 / 3.13) | the wheel installed **`--no-deps`** still imports, exposes `__version__` matching the dist metadata, exposes a loadable `vllm.general_plugins` entry point, and resolves `gridbook/csrc/*` from site-packages |
| `cpu-tests` (3.10 / 3.11 / 3.12 / 3.13) | the GPU-free part of the suite, run against the **installed wheel** from outside the checkout |

`check_dist.py` is the packaging gate. It asserts a literal floor of current
serving-reachable sources is in *both* artifacts: main and v2 decode/expansion,
the grouped BF16 CUTLASS bridge, both fused specializations, and their owned
headers. The retired persistent-TC source is deliberately outside that runtime
floor. The gate separately asserts that **nothing** under `gridbook/csrc` in
the checkout is missing from either artifact (so retained research sources and
any new kernel still cannot disappear because a `package-data` glob drifted),
and that the **checkout** has no stale repo-root `csrc/` left over from before
the packaging fix. That last one is checked against the checkout and not the
artifacts on purpose: `MANIFEST.in` never grafts a root `csrc/`, so a two-copy
tree is invisible in the built wheel/sdist and an artifact-only scan would pass
vacuously.

`check_installed.py` is the install gate. It refuses to run from a directory
that shadows site-packages, asserts `import gridbook` pulls in **no** torch,
vLLM, or Triton module, and proves every source/header in the literal native
serving floor resolves from `cuda_ext.csrc_dir()` inside site-packages.

**Test selection.** The suite selects itself at runtime — every CUDA, vLLM or
artifact-backed test is guarded by `pytest.skip` / `importorskip` /
`skipif(not cuda_ok)` — so no marker scheme was added; markers would duplicate
that logic for no extra signal. One mechanical fact drives
`run_cpu_tests.sh`:

- **One pytest process per file.** `test_target_namespace_compat.py` injects stub
  `vllm.*` modules into `sys.modules` so it can exercise the config resolver
  without the serving stack. Those stubs survive into later files, whose
  `importorskip("vllm")` guards then pass spuriously. Measured 2026-07-28 in the
  clean container described below: `test_transient_fp8.py` alone → 1 passed,
  7 skipped; the same file preceded by `test_target_namespace_compat.py` in one
  process → 3 failed, 31 passed, 4 skipped. The 3 failures are spurious —
  `ModuleNotFoundError: No module named 'vllm._custom_ops'` in tests the stub
  let past their `importorskip`.
Every optional monorepo dependency in `test_cb_kernels.py` is now guarded, so
the installed-wheel matrix collects every test module. Current CI runs the
suite on Python 3.10–3.13 from outside the checkout, with one pytest process per
file.

### `release.yml` — tag pushes matching `v*` only

```
build ──► verify ──► publish (pypi | testpypi) ──► github-release
```

- **`build`** — same build + `twine check` + `check_dist.py` as CI, then asserts
  the tag matches the version baked into the wheel, and decides PyPI vs TestPyPI
  from whether that version is a PEP 440 pre-release.
- **`verify`** — installs the exact wheel about to be published and re-runs the
  install gate and the CPU tests against it.
- **`publish`** — `pypa/gh-action-pypi-publish` with `id-token: write` and no
  other permission. Trusted Publishing (OIDC): no API token, no repository
  secret. `skip-existing` is deliberately **not** set, so a duplicate version is
  a loud failure rather than a silent no-op.
- **`github-release`** — `gh release create` with `--generate-notes`, attaching
  both artifacts, marked `--prerelease` for pre-release versions.

There is no `push: branches:` trigger and no `release:` trigger. Merging this
workflow, or any later commit to `master`, can never attempt an upload — the
first publish requires a human to push a tag. It is therefore safe to merge
**before** the PyPI project exists.

---

## 4. Maintaining the action pins

Every action is pinned to an exact patch tag (`actions/checkout@v7.0.1`, …),
checked 2026-07-28 — not to a floating major like `@v7` and not to a branch.
`.github/dependabot.yml` opens one grouped PR a week to bump them, and CI on that
PR is a real test of the new versions because the packaging gates run there.

Git tags are mutable in principle, so full-SHA pinning is stricter. If you want
it, replace each `@vX.Y.Z` with the tag's commit SHA and keep the version in a
trailing comment; Dependabot understands and maintains that form too.

---

## Trusted publishing: bound and proven (0.1.1)

`v0.1.1` published to PyPI with **no token and no repository secret** — all four
jobs green, PEP 740 attestations included. From here a release is:

```bash
# bump __version__ in gridbook/__init__.py, commit, then:
git tag -a vX.Y.Z -m "gridbook X.Y.Z" && git push origin vX.Y.Z
```

Nothing else. The workflow refuses to publish if the tag disagrees with the
built version, or if the built wheel fails a non-editable install with the
CUDA sources re-resolved from site-packages.

### The trap that cost us 0.1.0 — do not repeat it

`v0.1.0`'s run failed at `publish` with:

```
* `invalid-publisher`: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
```

The claims it presented were correct. The cause was sequencing: **a *pending*
publisher only becomes a real one when the project is created through it.**
`gridbook` was created by a token upload of `0.1.0rc1` — a rehearsal — which
orphaned the pending publisher and left the project with none attached. So
`0.1.0` had to be published with an API token from a workstation, and its
GitHub Release created by hand.

**If trusted publishing is the intent, the FIRST upload must come from the
workflow.** Do not token-upload a release candidate first; there is nothing to
rehearse that the `verify installed wheel` job does not already check.

Once a project exists, its publisher is added from the *project* settings, not
the account-level pending form:
<https://pypi.org/manage/project/gridbook/settings/publishing/> → owner
`RobTand`, repository `gridbook`, workflow `release.yml`, environment `pypi`.
