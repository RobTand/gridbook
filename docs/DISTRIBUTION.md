# Distribution — decision record and one-time action checklist

**Status:** historical decision record, written 2026-07-28. The original
two-tree synchronization design was retired on 2026-07-31: this repository is
now the sole owner of Gridbook runtime code, CUDA sources, tests, packaging,
and releases. PrismaQuant consumes a pinned external Gridbook contract and no
longer vendors the package. [`RELEASING.md`](RELEASING.md) is authoritative for
current release steps; two-tree passages below are retained only to explain
the failure mode that motivated single ownership. The current runtime contract
also supersedes every fail-soft/Triton passage below: Gridbook has no Triton
dependency or serving lane, required CUDA/CUTLASS kernels fail closed, and
[`RELEASING.md`](RELEASING.md) carries the active gate.

**Scope.** `gridbook` is technically finished enough to be used by strangers and
is not being used by strangers. This document records *why* each distribution
channel was chosen or rejected, and reduces the irreversible parts to a checklist.

**How to read it.** Section 1 is the ledger (what to do, in what order, and what
each thing buys). Section 2 is the part only Robert can do, with exact values to
type. Section 3 is the gate that must pass before the first PyPI upload, because
that upload is permanent. Section 4 is the honest list of things that look like
progress and are actually a maintenance obligation. Section 5 is the factual kit
for an announcement — not the announcement. Section 6 is what remains unverified,
and Section 7 is the correction log.

**House rule, applied here too:** every external claim carries a URL, every
performance number carries a source-file citation, and anything not established
is written as `unverified` rather than filled in.

### Two facts about this document itself — read before trusting a checkbox

**1. `github.com/RobTand/gridbook` is the only Gridbook source tree.**

The former PrismaQuant mirror created precisely the ambiguity this document
warned about: one tree could contain a packaging fix while the other produced
a source-less wheel. The fix is ownership, not a more elaborate sync. Runtime
changes now land here; producer integration installs an immutable commit and
validates the packaged runtime contract.

**2. Citations here quote strings, not line numbers.** Both trees change several
times an hour while this workstream is in flight; a first draft of this document
cited `BENCHMARKS.md` line numbers that were stale within four minutes. A quoted
string can be `grep`-ed and either still exists or provably does not. Line
numbers are given only alongside the string they point at, as a convenience.

**Publication status of this file — already public, decide whether to keep it
that way.** This is an internal decision record that lives in a public repo. It
contains internal machine paths, internal task numbers, and a launch plan that has
not been executed. Two separate surfaces, only one of which is still open:

- **PyPI sdist — closed.** Excluded by name (`MANIFEST.in`:
  `exclude docs/DISTRIBUTION.md`, `prune docs/hf-cards`), because a published PyPI
  file is permanent and cannot be trimmed later.
- **GitHub — already open.** This file was committed and pushed to
  `origin/master` in `9b6cb2f` on 2026-07-28, so it is public now. That is
  revisable: `git rm` it (optionally moving it to the private monorepo) if a
  strategy document that says things like "do not donate the repo yet" and "do not
  post to r/LocalLLaMA yet" is not something the project wants readable by anyone
  evaluating it. **This is a judgment call for Robert, not a defect** — recorded
  because the first draft of this document never considered that it publishes
  itself.

---

## 0. Decisions already made

### 0.1 The distribution name is `gridbook`

As of 2026-07-28, `gridbook`, `vllm-gridbook`, `gridbook-vllm` and `prismaquant`
are **all unclaimed** on PyPI (`https://pypi.org/pypi/<name>/json` → 404 for each;
snapshots in `/home/rob/dq-runs/scratch/pypi_*.json`).

Ship **one** distribution, named `gridbook`. Reasoning:

- It is already the import name, the repo name, the docs name, and — load-bearing —
  it is baked into the published HuggingFace artifact names
  (`*-prismaquant-gridbook-*`), which carry whatever traction exists.
- The cost is discovery: PyPI's search ranks *prefix* matches ahead of
  substring-only matches, so a bare `gridbook` never appears for the query `vllm`
  (behaviour documented at <https://github.com/posit-dev/positron/issues/14086>,
  2026-06-08). Buy that back through `summary`, `keywords` and classifiers — not
  through a second distribution.
- An alias package (`vllm-gridbook` pointing at `gridbook`) is a band-aid with a
  permanent maintenance cost and two version lines to keep in sync. Rejected.
- Squatting risk is not a real concern: PEP 541 permits name reassignment only
  for abandoned or squatted names, and an actively released project is neither
  (<https://peps.python.org/pep-0541/>,
  <https://docs.pypi.org/project-management/name-retention/>).

Precedent for the alternative, for the record: the vllm-project org names its
community plugins `vllm-gguf-plugin` / `vllm-bnb-plugin`, and the closest
solo-researcher analogue publishes as `turboquant-vllm`
(<https://pypi.org/project/turboquant-vllm/>). If gridbook is ever donated to the
vllm-project org (§1, LATER), an org-convention rename to `vllm-gridbook-plugin`
is a likely condition — plan for it as a *rename with an alias shim at that time*,
not as a reason to claim two names now.

### 0.2 Version `0.0.1` is spent — the real line starts at `0.1.0`

`0.0.1` was the prototype version. PyPI version numbers can never be reused, even
after deletion (<https://pypi.org/help/>). Treat `0.0.1` as spent and never upload
it.

`0.1.0` landed in tree **(A)** on 2026-07-28, with `gridbook.__version__` as the
single source of truth (`dynamic = ["version"]` + `[tool.setuptools.dynamic]`) so
`pyproject.toml` and the package cannot drift apart. It reached tree **(B)** by
rsync at 14:02 EDT the same day; before that moment this repo still read
`version = "0.0.1"`, and a build here would have consumed the wrong number.
**That is a per-build fact, not a settled one** — verify it in (B) at build time
via §3.0, do not read it here.

The release workflow makes the version half of this mechanical: `release.yml`'s
build job parses the wheel filename and exits non-zero unless it equals the
pushed tag (`"tag {tag} means version {want}, but the build produced {built}"`).
A `v0.1.0` tag on a tree still reading `0.0.1` fails there, before any upload.

### 0.3 Sequencing is forced by two irreversible facts

1. **A PyPI filename/version can never be reused** — "Deletion of a project,
   release or file on PyPI is permanent and irreversible, without exception" and
   "PyPI does not allow for a filename to be reused, even once a project has been
   deleted and recreated" (<https://pypi.org/help/>). Yanking
   (<https://docs.pypi.org/project-management/yanking/>) only hides a release from
   resolvers; it is not deletion. **So a broken `0.1.0` is permanent.**
2. **A "pending" Trusted Publisher does not reserve the name.** PyPI's own docs:
   configuring a publisher for a not-yet-existing project "does not reserve the
   project name until first use ... if another user registers the project name
   before you actually publish to it, your pending publisher will be invalidated"
   (<https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>).

Therefore: **fix packaging → verify a real non-editable install → TestPyPI dry
run → then and only then claim the name with a real `0.1.0` upload.**

---

## 1. The channel ledger

Ranked by reach ÷ effort. "Effort" is calendar-honest for one person: *trivial*
< 15 min, *small* < 2 h, *medium* ≈ a day, *large* ≈ a week.

| # | Channel | What it buys (reach) | Setup | Ongoing | Irreversible? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **GitHub About + topics** | The repo currently has `description: null` and `topics: []` (`gh api repos/RobTand/gridbook`, 2026-07-28), so it is absent from GitHub topic browse and weak in GitHub search. `topic:dgx-spark` has 255 repos, `topic:blackwell` 154, `topic:fp4` 17 — page-one visibility on exactly the right browse pages. | trivial | none | No (editable any time) | **RECOMMENDED** — highest reach per minute in the whole set, currently at literal zero. |
| 2 | **`pyproject.toml` metadata + summary rewrite** | The summary is the one line PyPI prints under the title and indexes for search. It currently reads *"prototype (i): correctness-first Triton serving, INV-2 waived"* — it tells a stranger the project is unfinished. No authors, no URLs, no classifiers, no keywords. | trivial | none | Yes, once uploaded (metadata is baked per release; fixable only by a new version) | **RECOMMENDED** — must land before the first upload. |
| 3 | **Fix `csrc/` packaging** (task #39) | This is the gate on everything else. `cuda_ext.py` used to resolve CUDA sources repo-root-relative (`os.path.join(dirname(__file__), os.pardir, "csrc", ...)`), while `[tool.setuptools.packages.find] include = ["gridbook*"]` shipped only the Python package — so any non-editable install had no `csrc/` and silently fell back to the Triton path. **Landed 2026-07-28 in BOTH trees** — the in-tree plugin (source of truth) and this public repo, which is the tree releases are built from; `csrc/` is now `gridbook/csrc/` in both (see §3.1). | medium | low | No | **DONE (was blocking).** §3.2 verified on a wheel and an sdist built from *this* tree: `check_dist.py` PASS, `check_installed.py` PASS in a torch-less venv, and a non-editable wheel install in `vllm-node:latest` JIT-built the extension in 30.2 s with all four entry symbols present. |
| 4 | **PyPI `gridbook` + Trusted Publishing** | The only channel that removes gridbook's actual gate. There is no vLLM plugin registry to be listed in; `pip install gridbook` *is* the distribution mechanism. | small (after #3) | per-release | **YES — permanently** (see §0.3) | **RECOMMENDED** — but strictly gated on §3. |
| 5 | **Tag + GitHub Release + CHANGELOG** | The repo has **0 releases and 0 tags**; version has read `0.0.1` since the prototype. Model cards currently pin the plugin *by date* — one of them says "plugin ≥2026-07-28", for which no public commit exists (newest public commit is `b4694e3`, pushed 2026-07-27T01:35Z). There is no way to tell a user which checkout has a fix. | small | per-release | Tags are effectively permanent once public | **RECOMMENDED** — also the trigger for the release workflow. |
| 6 | **HF card standardization + Collection + `library_name`/`base_model`** | HF is the surface gridbook's audience already uses. The three gridbook repos carry no `library_name` and no `pipeline_tag`, so they appear in no task filter and get no snippet. HF's own release checklist calls out `base_model` + `base_model_relation: quantized`, Collections and copy-and-run snippets as the discoverability levers (<https://huggingface.co/docs/hub/en/model-release-checklist>). | small | per-artifact | No (cards are editable) | **RECOMMENDED** — see §2.C. |
| 7 | **`CITATION.cff`** | Renders GitHub's "Cite this repository" sidebar with copyable BibTeX, and is what Zenodo reads when minting a release DOI (<https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files>). Cheap credibility, consistent with the DOI already held on the flagship PrismaQuant artifact. | trivial | none | No | **RECOMMENDED** |
| 8 | **`Dockerfile` in the repo** (build-it-yourself, no published image) | Makes the vLLM-version coupling explicit and reproducible. This is vLLM's own documented extension pattern (`FROM vllm/vllm-openai:<tag>`, <https://docs.vllm.ai/en/stable/deployment/docker/>). Critically, the only *known-good* stack today is a third-party community image on a moving `:latest` tag — a Dockerfile the user builds replaces that with something reproducible. | small | pin bump per vLLM release | No | **RECOMMENDED** — ship the Dockerfile, **not** an image. |
| 9 | **vLLM docs PR** — add gridbook under `## Out-of-Tree Quantization Plugins` | `vllm-project/vllm/docs/features/quantization/README.md` has that section and it names **zero** third-party implementations. gridbook would be the first. Marginal reviewer cost is one link. | small | ~none | No (a PR can be closed) | **RECOMMENDED — after PyPI**, so the PR points at something installable. |
| 10 | **`NVIDIA/dgx-spark-playbooks` PR** | 1,185 stars, 264 forks, 47 playbooks, Apache-2.0, pushed 2026-07-27, community PRs visibly open (<https://github.com/NVIDIA/dgx-spark-playbooks>). Its audience is definitionally people holding gridbook's reference hardware. Best audience-fit-per-effort of any external channel found. | medium | low | No | **LATER** — after the wheel exists. Note: the repo publishes no written third-party contribution policy, so the PR format has to be inferred from merged examples. |
| 11 | **HF library registration** (`huggingface.js` `model-libraries.ts`) | Puts a repo link, a docs link and a generated "how to load this model" snippet on **every** gridbook model page — the install instruction rendered at the exact point of friction. **239** libraries are registered (`grep -c 'prettyLabel:'` = `grep -c 'repoName:'` = 239 on the fetched `model-libraries.ts`; 238 distinct top-level keys) and **none** is vLLM-side — case-insensitive counts of `vllm`, `sglang` and `compressed-tensors` in that file are each **0**. Documented precondition: at least one model must already carry `library_name: gridbook`. | small | ~none | No | **LATER** — after PyPI and after the cards carry `library_name`. Acceptance is not guaranteed (the guide is framed around libraries that load from the Hub), so treat it as a cheap attempt, not a plan dependency. |
| 12 | **arXiv → HF Papers linkage** | Papers with Code was shut down by Meta in July 2025 and redirects to HuggingFace; HF Papers is the successor, and HF auto-extracts an `arxiv:<ID>` tag when a card links an arXiv abstract (<https://huggingface.co/docs/hub/en/model-cards#linking-a-paper>). This is now the *only* surviving paper→code linkage. | trivial | none | No | **LATER** — blocked on the paper having an arXiv ID. |
| 13 | **Reddit / HN / social launch post** | Real reach, entirely human judgment. | — | — | Effectively yes (a bad first post is not retractable) | **LATER — Robert only.** r/LocalLLaMA's rules could **not** be verified from this environment (reddit.com and old.reddit.com are both fetch-blocked). Third-party summaries describe a ~1-in-10 self-promotion ratio and required affiliation disclosure — **treat that as `unverified` and read the actual sidebar before posting.** |
| 14 | **Donate the repo to the `vllm-project` org** (as `vllm-gridbook-plugin`) | Highest reach available. The org already hosts community quantization plugins as first-class repos — `vllm-gguf-plugin` (26 stars) and `vllm-bnb-plugin` (2 stars) — so the bar is clearly technical fit, not popularity. | medium (conversation) | **high** — org review norms, likely a rename, shared ownership of release cadence | **Yes, socially** | **LATER** — a real option, worth raising *after* the wheel is live and the docs are accurate, not before. |
| 15 | **Published container image (GHCR)** | Removes the toolchain requirement entirely for users. GHCR is the right registry (unlimited public pulls, free bandwidth in Actions, no Docker Hub pull-rate cliff). | medium | **high** — the `vllm/vllm-openai` base is ~10.4 GB compressed and moves on every vLLM release; a published image is a standing rebuild obligation | Images can be deleted, but users pin them | **NO (for now)** — see §4. `vllm-gguf-plugin`, the closest precedent, publishes no image at all. |
| 16 | **Prebuilt binary wheels for multiple CUDA variants** | Removes the nvcc requirement. | large | **high** — a build matrix per CUDA × arch, forever | Yes (uploaded wheels are permanent) | **NO (for now)** — see §4. |
| 17 | **`BudEcosystem/Awesome-vLLM-plugins`** | The only curated vLLM plugin list found. 8 stars, last pushed 2025-12-12 (<https://github.com/BudEcosystem/Awesome-vLLM-plugins>). gridbook meets its stated bar (entry points under `vllm.*_plugins`). | trivial | none | No | **NO** — effectively dead traffic. Add it only if a PR happens to be free; do not spend attention on it. |
| 18 | **Papers with Code** | — | — | — | — | **NO** — shut down July 2025, redirects to HuggingFace. Recorded here so nobody re-derives it. |
| 19 | **conda-forge** | Another package channel. | medium | ongoing feedstock maintenance | No | **NO** — the audience installs vLLM from pip/source; a conda feedstock for a CUDA-JIT plugin is pure overhead. |
| 20 | **Alias PyPI package (`vllm-gridbook`)** | Would win the `vllm` prefix query on PyPI search. | small | permanent two-name maintenance | Yes | **NO** — §0.1. Buy the same discovery with `keywords` and `summary` instead. |

### 1.1 Download evidence — consistent with the install gate being *a* bottleneck, not proof it is *the* bottleneck

Recorded because it is the motivating observation for this document. It is
observational, it has one arm that cuts against it, and the heading above is
deliberately weaker than the first draft's ("the install gate — not the format —
is the bottleneck"), which the data below does not support.

Pulled from the HF API 2026-07-28 18:03 UTC
(`https://huggingface.co/api/models?author=rdtand&expand[]=downloadsAllTime&expand[]=createdAt`;
per-day figures use fractional days from `createdAt` to that instant):

- `rdtand/` totals **913,796** all-time downloads across 18 repos.
- The three gridbook artifacts hold **1,807** of that (27B 1,464 / Laguna 311 /
  Hy3 32).
- The cleanest available comparison holds base model and bit class fixed:
  `Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` (created 2026-07-11, 5,694
  downloads ≈ **334/day**) vs `Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm`
  (created 2026-07-21, 32 downloads ≈ **4.5/day**) — a **~74×** gap. The GGUF
  artifact serves through `vllm-gguf-plugin`, one `pip install` away; the gridbook
  artifact requires a git clone plus nvcc JIT.
- A third arm rules out "nobody wants a 295B":
  `Hy3-295B-A21B-PrismaQuant-5.3bit-2xSpark-vllm` needs **two** Sparks and still
  did ≈**88**/day.

**The arm that cuts against the conclusion, stated first because it is the one a
skeptic will find.** `rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm`
(created 2026-07-22, 1,464 downloads ≈ **241/day**) sits behind the *identical*
install gate as the Hy3 gridbook artifact and runs **~53×** its rate — within
**1.4×** of the GGUF comparator that supposedly demonstrates the gate. So the gate
cannot be the whole story. The most likely additional factor is addressable
audience: a 23 GB 27B fits any 32 GB card, while a 110 GB artifact fits
essentially one machine that exists in small numbers. Read the Hy3 pair as *at
most* an upper bound on the gate's cost, and read the 27B as evidence that a
gridbook artifact people can actually run does move.

**Further caveats, stated plainly.** n=1 model pair; observational, not an
experiment; confounded by README quality and by the gridbook repos' thinner
metadata. HF counts a download per HTTP request to `config.json` when no library
is registered (<https://huggingface.co/docs/hub/en/models-download-stats>), so
these are request counts, not users. The per-day normalization is if anything
*conservative* against the conclusion, since download curves spike at publication
and decay, which flatters the newer (gridbook) repos.

**What the section is actually sufficient to justify:** that fixing the install
gate is cheap, blocks every other channel in the ledger, and has no downside — not
that it will produce a 74× lift.

### 1.2 The packaging blueprint already exists and is Apache-2.0

`vllm-project/vllm-gguf-plugin` is an out-of-tree vLLM *quantization* plugin that
has already solved every packaging problem gridbook has:

- CUDA sources live **inside** the Python package (`vllm_gguf_plugin/csrc/...`) —
  the direct fix for the resolution bug.
- AOT build via `torch.utils.cpp_extension.CUDAExtension` with
  `py_limited_api=True` plus `options={"bdist_wheel": {"py_limited_api": "cp310"}}`,
  producing **one** `cp310-abi3` wheel per platform (~6.4 MB) instead of a
  per-Python matrix.
- Wheels for `manylinux_2_28_x86_64` **and** `manylinux_2_28_aarch64` — aarch64 is
  the DGX Spark architecture — built on free GitHub-hosted `ubuntu-24.04-arm`
  runners.
- Release publishes via `pypa/gh-action-pypi-publish@release/v1` with
  `permissions: id-token: write` and `environment: pypi` — tokenless OIDC, no
  secret on the box. Prereleases route to TestPyPI.
- The non-default CUDA variant carries a local version label (`+cu129`) and is
  attached to the **GitHub Release only**, because PyPI rejects local version
  labels — the concrete working answer to "how do you ship two CUDA variants".

Sources: <https://github.com/vllm-project/vllm-gguf-plugin> —
`setup.py`, `pyproject.toml`, `.github/workflows/release.yml`,
`scripts/build_release_wheel.sh`.

**What gridbook adopted, and what it deliberately did not.** The *packaging*
half of the blueprint is in (sources inside the package, resolved with
`importlib.resources`, declared as `package-data`, gated in CI by
`.github/scripts/check_dist.py`). The **AOT compilation half was not adopted for
v0.1.0**: gridbook ships a pure-Python wheel and keeps JIT-compiling its kernels
on first use.

That is the right call for the first release — it avoids a build matrix, avoids
pinning a CUDA major into a permanent artifact, and sidesteps the CUTLASS
coupling below — but it is a real tradeoff to record: **it keeps `nvcc` a hard
runtime requirement in the serving container**, which is the single biggest
remaining friction for a stranger. Revisit AOT once the wheel is proven and there
is evidence that the toolchain requirement is what people bounce off.

**If AOT is ever adopted, one deviation from the blueprint is mandatory:** that
project's `TORCH_CUDA_ARCH_LIST` is `"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"`
for CUDA ≥ 13 — it omits **12.1**, which is gridbook's own reference target (GB10
/ DGX Spark, `sm_121`). Checked locally against the build venv: torch
2.11.0+cu130's `_get_cuda_arch_flags` lists `'12.0','12.0a','12.1','12.1a'` as
supported, so 12.1 is available and must be requested explicitly. Copying the arch
list verbatim would ship a wheel that misses the hardware every published
benchmark was measured on.

**The CUTLASS coupling is a hard AOT blocker, not a detail.** `cuda_ext.py`
discovers CUTLASS headers by globbing the *installed vLLM's* directory tree. Under
JIT that is a soft coupling that fails over to Triton. Under AOT it becomes a
build-time dependency on a vLLM internal layout that carries no compatibility
promise (vLLM is at `v0.26.0` with **101** releases —
`gh api repos/vllm-project/vllm/releases --paginate --jq '.[].tag_name' | wc -l`,
2026-07-28 — and its plugin docs put
version-compat responsibility on the plugin author). Resolve that before AOT, not
during.

---

## 2. One-time human actions

These are the parts an agent must not do. Each is written so it can be executed
without further research.

### 2.A — PyPI: claim `gridbook` and configure Trusted Publishing

> **[`RELEASING.md`](RELEASING.md) is authoritative for the procedure.** When
> this section was written, neither it nor `.github/workflows/release.yml`
> existed, so the steps below were a *prediction* of a workflow. Both exist now
> (`ls .github/workflows/` → `ci.yml`, `release.yml`), the values agree, and the
> steps below are retained only because they carry the decision context. **If the
> two ever disagree, `RELEASING.md` and the workflow file win** — a second copy of
> a procedure is a second thing to drift.

**Do NOT do this until §3 passes.** A pending publisher does not reserve the name
(§0.3), so there is no first-mover reason to rush it.

**Step 1 — accounts.** Create/confirm accounts on both, with 2FA enabled (PyPI
requires 2FA for uploads):

- <https://pypi.org/account/register/>
- <https://test.pypi.org/account/register/> (separate account; TestPyPI is a
  distinct instance)

**Step 2 — GitHub environments.** In
`https://github.com/RobTand/gridbook/settings/environments` → **New environment**,
create two, exactly:

| Environment name | Purpose |
|---|---|
| `pypi` | real releases |
| `testpypi` | prereleases / dry runs |

No secrets go in either. (With Trusted Publishing there is nothing to store —
that is the point.) Optionally add yourself as a required reviewer on `pypi` so a
release cannot fire unattended.

**Step 3 — PyPI pending publisher.** Go to
<https://pypi.org/manage/account/publishing/> → *"Add a new pending publisher"* →
**GitHub** tab. Fill in **exactly**:

| Form field | Value to type |
|---|---|
| PyPI Project Name | `gridbook` |
| Owner | `RobTand` |
| Repository name | `gridbook` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

> ⚠ **The "Workflow name" value must be the release workflow's exact filename.**
> It is `release.yml` (`.github/workflows/release.yml`). A mismatch here does not
> fail loudly — it fails as a rejected OIDC claim at publish time, which is a
> confusing place to discover a typo.

**Step 4 — TestPyPI pending publisher.** Repeat at
<https://test.pypi.org/manage/account/publishing/> with the identical values
except:

| Form field | Value to type |
|---|---|
| Environment name | `testpypi` |

**Step 5 — the dry run.** Push a prerelease tag (e.g. `v0.1.0rc1`) — **the tag
and nothing else**. `release.yml` detects the PEP 440 pre-release, routes the
upload to TestPyPI, and creates the GitHub pre-release itself. Then verify from a
clean venv on a machine with nvcc:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ gridbook
```

**Step 6 — the real claim.** Only after §3 is fully green: push tag `v0.1.0`,
again **the tag alone**. The workflow publishes to PyPI, the name is claimed, and
the workflow creates the GitHub Release.

**Notes.**
- **Never create the GitHub Release by hand** in Step 5 or Step 6. `release.yml`'s
  `github-release` job runs `gh release create "$GITHUB_REF_NAME"`, which fails if
  a release already exists for that tag — so a hand-cut release turns a successful
  publish into a red workflow. Pushing the tag is the whole action. See
  [`RELEASING.md` §2.3](RELEASING.md).
- Do **not** upload `0.0.1`. See §0.2.
- **What the automation catches, and the one thing it does not.** `publish`
  declares `needs: [build, verify]`. `build` runs `check_dist.py` (fatal) and the
  tag-vs-version assertion; `verify` installs the built wheel **non-editable**
  into a clean interpreter and re-resolves the packaged CUDA sources from
  `site-packages`. So a tag pushed against a tree carrying the packaging defect
  *cannot* publish — it dies in `build`, before any upload. The bypass that
  remains is a manual `python -m build && twine upload` from a workstation, which
  touches none of it. **Never release that way**: it is the only remaining path by
  which a permanently broken `0.1.0` reaches PyPI.
- After the first successful publish, the *pending* publisher becomes a normal
  publisher on the project. Nothing further to do.
- If a release turns out broken: **yank it** (project page → Manage → the release
  → Options → Yank), do not delete it. Deleting a project releases the name to
  anyone; yanking hides it from resolvers while leaving explicit pins working.

### 2.B — GitHub repo polish

All of this is at <https://github.com/RobTand/gridbook> and takes ~10 minutes.

**About / description.** Click the gear next to "About". Paste:

> Out-of-tree vLLM plugin and open format spec for NVFP4-CB / FP8-CB product-codebook weights — sub-4-bit-class model sizes served on native Blackwell tensor cores.

**Website field.** Leave blank until the HF Collection exists (§2.C), then set it
to the Collection URL. (Second choice: the PyPI project page.)

**Topics.** In the same dialog, add these 14 (GitHub allows up to 20; lowercase,
hyphenated, ≤50 chars —
<https://docs.github.com/articles/classifying-your-repository-with-topics>):

```
vllm
quantization
llm-inference
llm-serving
cuda-kernels
triton
blackwell
dgx-spark
nvfp4
fp4
fp8
vector-quantization
codebook-quantization
moe
```

Measured browse-surface sizes, 2026-07-28 18:05 UTC
(`gh api "search/repositories?q=topic:<t>&per_page=1" --jq .total_count`):
`llm-inference` 2,610 repos · `quantization` 2,111 · `vllm` 1,801 ·
`cuda-kernels` 366 · `dgx-spark` 255 · `blackwell` 154 · `fp4` 17. The last three
are small enough for page-one visibility. These counts drift by a few repos a day;
they are recorded to size the surfaces, not as fixed figures.

**Enable Issues** — already on (`has_issues: true`). **Issue #1 has been
acknowledged but not answered.** Opened 2026-07-25 by an RTX 5090 owner who
diagnosed three bugs. State as of 2026-07-28 18:00 UTC
(`gh api repos/RobTand/gridbook/issues/1 --jq .comments` → `2`): a maintainer
acknowledgement at 17:20 UTC (11 characters, "Im looking"), then at 17:59 UTC a
substantive follow-up **from the reporter**, who re-tested against `da86eef` and
reports bugs 1–2 fixed by the parse-time target canonicalization — while raising a
**new, unfiled** problem: with the mid-M fused prefill path enabled by default,
`cb_qweight` and `_cb_qw_padded` are both live at runtime, which he measures as
~35 GB on a 32 GB card. So the technical answer is still owed, the issue has grown
rather than shrunk, and it now contains a bug report that exists in no task.
Answering it is worth more than any channel in §1; see
<https://github.com/RobTand/gridbook/issues/1> and tasks #45/#46 (#46 is the
double-residency item, and the reporter's note is evidence it is user-visible on
consumer cards, not just a footprint nicety).

**Enable Discussions** — currently off (`has_discussions: false`). Settings →
Features → check **Discussions**. Reasoning: it gives "how do I run this on X"
somewhere to go that is not the issue tracker, which matters once installs work.
Counter-argument, honestly: it is another inbox. If the answer is "I will not read
it", leave it off — an unanswered Discussions tab is worse than none.

**First Release.** Nothing to do here by hand. Do **not** visit
`/releases/new` — `release.yml` creates the GitHub Release itself, and a
hand-made release on the same tag makes its `gh release create` step fail (§2.A;
`RELEASING.md` §2.3). After §3 passes, the entire action is `git push origin
v0.1.0`; the workflow publishes to PyPI, then cuts the Release with
`--generate-notes` and attaches `dist/*`. If a hand-written body is wanted
instead of generated notes, edit the Release **after** the workflow has created
it.

**`CITATION.cff`** (staged separately, task-tracked): author Robert Tand,
`robert.tand@icloud.com`, Apache-2.0, `repository-code:
https://github.com/RobTand/gridbook`, with `preferred-citation` left out until the
paper has an arXiv ID.

### 2.C — HuggingFace: cards, metadata, Collection

**The staged card edits now live in the public repo at
[`docs/hf-cards/`](hf-cards/)** — a `_TEMPLATE.md`, one rendered file per
artifact (front matter + body, each in a copyable fenced block), plus
`_METADATA.md` (which card metadata actually filters on the Hub) and
`_COLLECTION.md` (exact Collection title, description, members and steps). That
directory carries its own APPLY GATE; read it before pasting anything. Raw
sources and the fetched `config_*.json` / `quantcfg_*.json` used to verify the
card claims against the shipped bytes are in
`/home/rob/dq-runs/scratch/gridbook-hf/`.

The substance, summarized here so the decision is recorded in one place:

**C1 — YAML front-matter.** Each gridbook card currently has `license`,
`base_model` and free-form `tags`, but **no `library_name` and no
`pipeline_tag`** — so the repos appear in no task filter and get no widget or
generated snippet. Add to each:

```yaml
library_name: gridbook
pipeline_tag: text-generation      # image-text-to-text for the 27B (vision tower)
base_model_relation: quantized
tags:
  - prismaquant
  - gridbook
  - codebook-quantization
  - vllm
  - blackwell
  - dgx-spark
  - mixed-precision
  - conversational
```

Reference for the shape: `RedHatAI/Qwen3-32B-NVFP4` runs
`pipeline_tag: text-generation` with tags `fp4, vllm, compressed-tensors, 8-bit`.
HF's release checklist recommends `base_model` + `base_model_relation: quantized`
for quantized derivatives
(<https://huggingface.co/docs/hub/en/model-release-checklist>).

**C2 — one identical serve block at the very top of all three cards.** Today the
three cards give **three different install commands**, two of which silently
produce a degraded (Triton-path) install. The rendered blocks in `docs/hf-cards/`
already standardize this — but they say `pip install gridbook`, so **applying
them is gated on §2.A step 6.** Until the wheel is live, either hold the cards or
render them against the `git+https://` route.

**C3 — remove the vendored plugin copy from the Hy3 repo.** It ships
`serving/gridbook/**`, a snapshot that is already missing 7 files present on
master, plus `serving/gridbook/vllm_prismaquant.egg-info/` — build junk under the
pre-rename package name, published publicly. It is currently the *only* install
path that works (by accident of `pip install -e`), so it must be replaced by the
working `pip install gridbook`, not just deleted.

**C4 — backfill `base_model`** on `rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm` and
`...-5.5bit-vllm`, which carry none at all and are therefore invisible in Gemma4's
model tree. (Not gridbook artifacts, but the same one-line fix and the same visit.)

**C5 — create the Collection.** <https://huggingface.co/rdtand> → **New
Collection** → name `gridbook — codebook-quantized artifacts` → add the three
gridbook repos. Then set the GitHub repo's Website field to the Collection URL
(§2.B).

**C6 — funnel from the high-traffic repos.** `PrismaSCOUT 27B` (~227k downloads),
the 35B (~98k), `PrismaAURA 27B` (~20k) currently make no mention of the newer
gridbook artifact of the same base model. One line at the top of each pointing at
the successor is the cheapest reach in the entire HF surface.

### 2.D — vLLM upstream: the docs PR

**Target:** `vllm-project/vllm`, file
`docs/features/quantization/README.md`, section `## Out-of-Tree Quantization
Plugins` (heading at line 76 as of 2026-07-28,
<https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/README.md>).

That section documents how to register a quantization config via
`@register_quantization_config`, how to implement linear and MoE methods, and how
to use the plugin — but it names **zero** third-party implementations. gridbook
would be the first.

**Sequence it after PyPI**, so the PR points at something a reader can install.

**Draft PR body** (to be edited by Robert before sending):

> **Title:** docs: link a third-party out-of-tree quantization plugin example
>
> The `Out-of-Tree Quantization Plugins` section explains the mechanism but does
> not point at any working third-party implementation. This adds one, so readers
> have a complete example to read.
>
> [gridbook](https://github.com/RobTand/gridbook) (Apache-2.0, `pip install
> gridbook`) is an out-of-tree quantization plugin implementing product-codebook
> weight formats (NVFP4-CB / FP8-CB). It registers through
> `vllm.general_plugins` + `register_quantization_config`, implements
> `QuantizeMethodBase` for both `LinearBase` and fused MoE, and registers custom
> ops for its decode/prefill kernels — i.e. it exercises the documented plugin
> surface end to end. The weight format is specified normatively and openly at
> [docs/SPEC.md](https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md).
>
> Happy to drop this if the maintainers prefer the section stay
> implementation-neutral, or to reword it as a neutral "examples in the wild" list
> that others can add to.

**Also worth knowing** (do not put in the PR): the vllm-project org hosts
community quantization plugins as first-class repos — `vllm-gguf-plugin` (26
stars) and `vllm-bnb-plugin` (2 stars, pushed 2026-07-28) — under the naming
convention `vllm-<x>-plugin`. At 2 stars the bar is technical fit, not popularity.
That is a separate, higher-commitment conversation (ledger row 14), not this PR.

### 2.E — the `NVIDIA/dgx-spark-playbooks` PR (LATER)

<https://github.com/NVIDIA/dgx-spark-playbooks> — 1,185 stars, 264 forks, 47
playbooks, Apache-2.0, pushed 2026-07-27, community PRs visibly open. Its audience
is people holding gridbook's exact reference hardware, and "a 295B MoE serves on
one Spark" is precisely the kind of claim the repo exists to document.

**Before writing anything:** read `CONTRIBUTING` and an existing merged playbook.
The README publishes **no written third-party contribution policy** — the format
must be inferred from merged examples and the open PR queue, not assumed.

Gate this on the wheel being live, so the playbook's first step is
`pip install gridbook`.

---

## 3. Pre-flight checklist — must be green before the first PyPI upload

A broken `0.1.0` is permanent (§0.3). Every box below is a hard gate.

**Every box in §3 is a claim about tree (B) — this repo — because that is the
tree release artifacts are built from** (`MANIFEST.in`: *"Release artifacts are
built from THIS tree (the public repo)"*). A box ticked on the strength of tree
(A) is not ticked. §3.0 exists to make that impossible to forget.

### 3.0 Canonical-tree gate — **run this first, every time, before any build**

Build only from a clean checkout of this repository. There is no upstream
runtime mirror to compare or synchronize. The package tree, tests, metadata,
and release workflow reviewed in the PR are exactly the tree being released.

- [ ] **The checkout is clean and on the reviewed release commit.**

      ```bash
      test -z "$(git status --porcelain)"
      git log -1 --show-signature --oneline
      ```

      PrismaQuant updates its immutable compatibility pin only after this
      commit merges; that downstream pin is not an input to this build.
- [ ] **The packaging invariant holds in (B), checked in (B).** Check the
      *behaviour*, not the source text — a `grep` for `os.pardir` matches the
      docstring that warns against reintroducing it, which makes it useless as a
      gate:

      ```bash
      test -d gridbook/csrc && ! test -e csrc   # sources inside the package
      python - <<'PY'
      import os
      from gridbook.cuda_ext import csrc_dir
      import gridbook
      d = csrc_dir()
      assert os.path.basename(d) == "csrc", d
      assert os.path.basename(os.path.dirname(d)) == "gridbook", d
      cu = ("cb_gemv.cu", "cb_fused_gemm.cu", "cb_persistent_prefill.cu",
            "cb_persistent_tc.cu", "sm120_fp8_gemm.cu", "smem_probe_tilem.cu",
            "toolchain_probe.cu")
      assert not [f for f in cu if not os.path.exists(os.path.join(d, f))]
      assert len([f for f in os.listdir(os.path.join(d, "cutlass_fork"))
                  if f.endswith(".hpp")]) == 5
      print("ok", gridbook.__version__, d)
      PY
      ```

      Measured in (B) 2026-07-28 18:10 UTC: `csrc_dir()` →
      `<repo>/gridbook/csrc`, 0 missing `.cu`, 5 `cutlass_fork/*.hpp`. Run the
      same block again *after* a non-editable install from a directory that is not
      the repo (§3.2) — that is the case the release actually ships.
- [ ] **The build under test was produced from (B).** Not from (A), not from a
      copy of (A). If a build command in §3.2 was run somewhere else, the boxes it
      ticked are void.

**Measured record of why this gate exists** (2026-07-28, both builds by
`python -m build --wheel --sdist`, both gated with the repo's own
`.github/scripts/check_dist.py`):

| Built from | Version | `csrc` entries in wheel | `check_dist.py` |
|---|---|---|---|
| (B) at 17:5x UTC — pre-rsync | `0.0.1` | **0** | exit **1**: *"pub/gridbook/csrc does not exist"*, *"wheel: runtime-required sources missing: ['gridbook/csrc/cb_gemv.cu', …]"*, 3 errors |
| (A) at 18:03 UTC | `0.1.0` | **12** | exit **0** — `PASS` |
| (B) at 18:05 UTC — post-rsync | `0.1.0` | **12** | exit **0** — `PASS` |

The middle row is the one that makes the point: (A) was green the entire time. Had
a tag been pushed in that window on the strength of (A)'s state, the artifact
built would have been the top row.

**The mechanical backstop, and its limit.** `release.yml` runs `check_dist.py`
and the tag-vs-version assertion in `build`, which `publish` depends on, so a tag
pushed in that window fails CI rather than publishing (§2.A). This gate is
belt-and-braces for the case CI cannot see: a manual `twine upload`.

### 3.1 Packaging mechanism — landed in (A) 2026-07-28; **confirm in (B)**

Marked `[~]` where the mechanism is *known landed in (A)* but must still be
observed in (B) at build time via §3.0 — the state below was true of (A) and
false of (B) for several hours on 2026-07-28.

- [~] `csrc/` resolves **inside the installed package**, not via `os.pardir`.
      Sources moved to `gridbook/csrc/` (the repo-root `csrc/` is gone) and
      `cuda_ext.py` now resolves them through `importlib.resources`
      (`csrc_dir()`), so an in-repo checkout, `pip install -e` and a wheel install
      all resolve identically. *Verified in (B) 2026-07-28 18:05 UTC:
      `ls gridbook/csrc` lists 8 entries, `ls csrc` → "No such file or directory".*
- [~] `[tool.setuptools.package-data]` declares
      `gridbook = ["csrc/*.cu", "csrc/*.cuh", "csrc/*.h", "csrc/cutlass_fork/*.hpp"]`,
      which also lands them in the sdist. *Verified in (B): 12 `csrc` members in
      both wheel and sdist.*
- [ ] Confirm on the **built artifacts** (not the source tree) that every
      runtime-required native source is present: `cb_gemv.cu`,
      `cb_fused_gemm.cu`, `cb_persistent_prefill.cu`, `cb_persistent_tc.cu`,
      `sm120_fp8_gemm.cu`, `smem_probe_tilem.cu`, `toolchain_probe.cu`, and the 5
      `csrc/cutlass_fork/*.hpp` headers. A `package-data` glob that silently
      matches nothing is the exact failure mode this gate exists for.
- [ ] The CUTLASS question is **decided and written down**: `cuda_ext.py`
      currently globs CUTLASS headers out of the *installed vLLM's* directory
      (`_find_cutlass_include()`), and in the tested image two different CUTLASS
      versions match that glob (4.3.4 under `fmha_sm100/`, 4.2.1 under
      `deep_gemm/`) with unspecified ordering. Pick one of: vendor the headers,
      make discovery deterministic + add an env override, or keep only the
      CUTLASS-dependent kernels on the JIT path. Do not ship an
      order-nondeterministic build input.

### 3.2 Wheel-install verification (no GPU required)

Run inside a docker container **without `--gpus`** — compile-only is the validated
pattern on this box:

**§3.0 must be green first, and every command below runs with tree (B) as the
working directory** — `cd /home/rob/gridbook`. `check_dist.py`'s second argument
is the *checkout to compare the artifacts against*; passing `.` from anywhere
else silently changes what is being gated.

- [ ] `pip install .` (**non-editable**) into a *clean* venv succeeds.
- [ ] After install, `python -c "import gridbook, pathlib; ..."` confirms every
      source path `cuda_ext.py` will request actually exists under
      `site-packages/`.
- [ ] `python -m build` produces both sdist and wheel; `twine check dist/*` passes.
- [ ] **`python .github/scripts/check_dist.py dist .` passes, run from (B).** This
      is the packaging gate that already exists in the repo: it asserts a literal
      floor of runtime-required `.cu`/`.hpp` sources is present in *both*
      artifacts, and that nothing under `gridbook/csrc` in the checkout is missing
      from either. It is written to be fatal on purpose — a wheel that omits the
      sources still installs, still imports, still registers with vLLM, and still
      serves *correct* output on the slow path, with one stderr warning as the only
      clue. That is the exact defect class that must not reach a permanent version
      number. It is demonstrably fatal, not decorative: see the §3.0 table, where
      the same script exits 1 on the pre-rsync tree and 0 on the synced one.
- [ ] The install works from a directory that is **not** the repo (`cd /` first) —
      this is what catches accidental relative-path dependence.
- [ ] **The sdist contains no internal working material.** `MANIFEST.in` grafts
      all of `docs/`, so anything dropped there ships inside a permanent artifact.
      `docs/DISTRIBUTION.md` (this file — internal paths, internal task numbers, an
      unexecuted launch plan) and `docs/hf-cards/` (un-applied card drafts, each
      with its own APPLY GATE) are excluded by name. Confirm after any new doc
      lands:

      ```bash
      python - <<'PY'
      import glob, tarfile
      t = tarfile.open(glob.glob('dist/*.tar.gz')[0])
      print(sorted(n for n in t.getnames() if '/docs/' in n))
      PY
      ```

      Expected 2026-07-28: `BENCHMARKS, CONTAINER, INSTALL, KERNELS, MOTIVATION,
      PLUGIN, RELEASING, SPEC, TROUBLESHOOTING` — and neither `DISTRIBUTION.md`
      nor `hf-cards/`.

### 3.3 Entry-point check

- [ ] `python -c "from importlib.metadata import entry_points;
      print(list(entry_points(group='vllm.general_plugins')))"` lists
      `gridbook = gridbook:register`.
- [ ] `import gridbook` succeeds **without vLLM installed** (the `register`
      indirection in `gridbook/__init__.py` exists for exactly this; keep it).
- [ ] A canary that imports every vLLM symbol the plugin touches exists and
      passes against the tested vLLM. **Why this matters:** vLLM's plugin loader
      wraps `plugin.load()` in `try/except Exception: logger.exception(...)` — it
      logs and *continues*. So any drift in gridbook's top-level imports does not
      surface as a plugin error; it surfaces later as an unrelated "unknown
      quantization method" at model load.

### 3.4 Real serve smoke — **orchestrator/GPU only**

- [ ] Install the *built wheel* (not `-e`, not the repo) into the serving
      container, serve a published artifact by Hub id, and confirm from the log
      that the **CUDA path is active** — i.e. that
      `[prismaquant-cb] WARNING: CUDA decode-GEMV extension unavailable` does
      **not** appear.
- [ ] Confirm decode throughput matches the measured figure for that artifact in
      `docs/BENCHMARKS.md` (e.g. 27B: ~10.3 tok/s). Missing required Gridbook
      kernels now fail closed; this speed check catches a stale runtime or an
      unintended native dispatch regression that symbol/load gates cannot.
- [ ] Confirm on the wheel-installed path that both artifact registry keys
      resolve to the same implementation: canonical `"gridbook"` and the
      read-only legacy alias `"prismaquant"`. Published artifacts use the
      canonical key and the pointer-sidecar layout. `docs/SPEC.md` now describes
      that shipped contract, including the fully inlined compatibility form.

### 3.5 Docs accuracy — no upload with a false README

The README rewrite (task #42) landed 2026-07-28 and closed most of this list.
What it closed, recorded so it is not re-litigated:

- [x] The install line is honest — it offers `pip install
      git+https://github.com/RobTand/gridbook` as the route that works *today* and
      labels the PyPI line "planned; not yet published". **This must flip to a
      plain `pip install gridbook` as part of the release**, not before.
- [x] The false "builds a CUDA extension **at install time**" claim is replaced
      with the truth: JIT-compiled on first model load, in the process that
      serves, which is where `nvcc` is actually needed.
- [x] The "zero monkeypatches on the load path" overstatement is gone.
- [x] The flat "Blackwell only" requirement is replaced by a per-GPU-class
      compatibility table that separates MEASURED (GB10 `sm_121`) from
      USER-REPORTED (RTX 5090, issue #1) from INFERRED-UNTESTED (H100 / 4090 /
      A100), and states the A100 prefill failure explicitly.
- [x] `tp=1` is stated in the README's requirements table, not buried in an
      unlinked doc.
- [x] The published artifacts are listed with links and real sizes; the
      quickstart names a real repo id and ends in a `curl`; and there is an
      explicit "how to tell it is actually working" section quoting the fail-soft
      warning string verbatim.

**Closed since this list was first written** (re-measured in (B), 2026-07-28
18:04 UTC — each with the command that closed it, because these boxes were open
for about twenty minutes and the evidence for closing them must be reproducible):

- [x] **The 295B prefill ratio now agrees across the repo at `~2.6×`.**
      `grep -n '2\.1' docs/MOTIVATION.md` → no output;
      `grep -rn '2\.1×' --include='*.md' .` matches nothing outside
      `docs/hf-cards/` (see the caveat below). `BENCHMARKS.md` now reads
      *"Prefill ~109 tok/s vs the GGUF IQ build's 42 — ~2.6× faster"* and
      *"~2.6× prefill"*, matching `README.md`'s
      *"~109 tok/s vs the matched-byte GGUF IQ build's 42 = ~2.6×"*.
      **Caveat:** `docs/hf-cards/Hy3-295B-…-2.9bit-vllm.md` still carries a note
      describing the *old* three-way contradiction ("calls it 2.1×… says ~2.1×
      prefill… `docs/MOTIVATION.md` says ~2.1×"). That note is now itself stale
      and should be deleted when the cards are applied (§2.C), or it will
      reintroduce the confusion it was written to flag.
- [x] **TEB is stated once, consistently.** `BENCHMARKS.md`'s table row and its
      bullet both read `88 (130/148)`; the `87 (129/148)` that looked like a
      self-contradiction is the **GGUF IQ comparator's** score in the same table
      (*"against the GGUF IQ build's 87 (129/148) and k-quant's 86 (128/148)"*),
      not a second value for the same artifact. The honest "read it as parity,
      not a win" framing is in both places.
- [x] **The 35B artifact's unavailability is stated plainly**, in both documents
      that cite it: `BENCHMARKS.md` — *"the 35B MoE artifact is not published"*;
      `README.md` — *"**35B MoE**, 256 experts (**not published**)"*.
- [x] **The docs the README links now exist.** `ls docs/` → `BENCHMARKS.md`,
      `CONTAINER.md`, `DISTRIBUTION.md`, `INSTALL.md`, `KERNELS.md`,
      `MOTIVATION.md`, `PLUGIN.md`, `RELEASING.md`, `SPEC.md`,
      `TROUBLESHOOTING.md`, `hf-cards/`.
- [x] `docs/PLUGIN.md` is retitled (`# Plugin reference`, no `vllm-prismaquant`),
      linked from the README's Documentation table, and its sidecar filename reads
      `cb_codebooks.pqcb`.

Still open, and still blocking:

- [ ] The 27B KL numbers in `docs/BENCHMARKS.md` (`0.0134` ALL-KL, `−58.3%`) and
      in the HF card (`0.0049` ALL-KL, `−77%`) are either reconciled or explicitly
      cross-referenced with the protocol difference named. The documented ±17%
      session-arithmetic drift cannot span a 2.7× gap, so silence here reads as
      one of them being inflated. **This is the last live numeric contradiction in
      the repo** — the staged card at
      `docs/hf-cards/Qwen3.6-27B-…-5.5bit-vllm.md` already flags it against
      itself, and `docs/hf-cards/_COLLECTION.md` still quotes the unreconciled
      `−77%` in the Collection blurb, so applying the cards without settling this
      propagates it to a fourth place.
- [ ] `docs/SPEC.md`'s registry key is corrected to `"gridbook"` (§3.4). This is
      the **only** document that needs the change; `ROADMAP.md` already says so.
- [ ] `docs/PLUGIN.md` is the sole public home of 5 of the 26 `PRISMAQUANT_*` env
      knobs — the other 21 are documented nowhere, including one a model card
      instructs users to set. Not release-blocking; recorded so it is not lost.

> **Correction to an earlier revision of this list.** It carried a box asking
> that `ROADMAP.md` "no longer list as 'planned' two things that shipped
> (Persistent-N large-M prefill; batched-expert MoE prefill)". Both halves were
> wrong. `ROADMAP.md` already records Persistent-N as *"Built, parity-green, and
> **2–5.7× slower** than expand-then-GEMM at 27B shapes … The serving selector,
> custom op, loader, and switch are deleted; only direct research source remains"* — an honest
> negative result, not a plan. And `grep -i batched ROADMAP.md` returns nothing
> (rc=1), so there was no second item to fix. The box would have read as a live
> blocker that no one could action.

### 3.6 Metadata and hygiene — landed in (A) 2026-07-28; **confirm in (B)**

- [~] Version is `0.1.0`, with `gridbook.__version__` as the single source of
      truth (`dynamic = ["version"]` + `[tool.setuptools.dynamic]`), so
      `pyproject.toml` and the package cannot drift. **Confirm in (B), not (A)** —
      this repo read `version = "0.0.1"` until the 14:02 EDT rsync on 2026-07-28,
      and an earlier revision of this box was ticked anyway. The tag-vs-version
      guard now exists: `release.yml`'s build job parses the wheel filename and
      exits with *"tag {tag} means version {want}, but the build produced
      {built}"* on a mismatch.
- [~] `description` no longer says *"prototype (i): correctness-first Triton
      serving, INV-2 waived"*. Same caveat: that stale string was live in (B) for
      hours after it was fixed in (A). It is also *still moving* — the summary was
      edited in (A) again at 14:02 EDT. Read the built `METADATA`, not this file:
      a released summary cannot be edited.
- [x] The prototype framing is gone from `gridbook/__init__.py`'s module
      docstring. **Still open:** the in-tree `plugins/gridbook/README.md` is a
      different and older document than the public one and still opens
      `# vllm-prismaquant` with the same dead framing.
- [x] `authors = [{ name = "Robert Tand", email = "robert.tand@icloud.com" }]`.
- [x] `[project.urls]` — Homepage / Repository / Issues / Documentation.
- [x] `keywords` and `classifiers` present. Note the PEP 639 interaction that was
      resolved here and should not be undone: `license = "Apache-2.0"` is an SPDX
      expression requiring `setuptools>=77`, and a `License ::` *classifier* must
      **not** be present alongside it — setuptools rejects both together.
- [ ] Classifiers cover the Python versions actually supported. Currently 3.10–3.12;
      add `3.13` only if it is tested.
- [x] **The dev/system email address does not appear anywhere in the public tree,
      its git history, or the built metadata.** The public attribution address is
      `robert.tand@icloud.com`, always — and the dev address is not to be written
      into public files even inside a warning about not writing it — **including
      inside this checklist item**, which is why the command below reads the
      local-part out of the environment instead of spelling it out:

      ```bash
      # DEV_LOCALPART: the local-part of the dev/system address. Set it in the
      # shell, do not commit it.
      grep -rniI -- "$DEV_LOCALPART" --exclude-dir=.git . ; echo "want rc=1"
      git log --format='%ae|%ce' | sort -u        # want: only the public identity
      git config user.email
      python -c "import zipfile,glob,sys; z=zipfile.ZipFile(glob.glob('dist/*.whl')[0]); \
        print([l for l in z.read([n for n in z.namelist() if n.endswith('METADATA')][0]) \
        .decode().splitlines() if l.startswith('Author')])"
      ```

      Measured in (B) 2026-07-28 18:04 UTC: the grep returned **rc=1** (no match);
      `git log` returned a single identity, `rob@sparky.lan`, for both author and
      committer; the built wheel's `METADATA` carries
      `Author-email: Robert Tand <robert.tand@icloud.com>`. **Re-run immediately
      before the tag** — one earlier hit (`docs/hf-cards/README.md`, inside a
      warning *about* the address) appeared and was removed within a single
      afternoon, so this is a per-commit property, not a settled one.
- [ ] `LICENSE` is present in the sdist.
- [ ] Sanity-check the dependency decision before it becomes permanent metadata:
      `dependencies = ["torch", "triton", "safetensors", "huggingface-hub"]` are
      deliberately unpinned (the reference serving builds use local-version wheels
      like `2.11.0+cu130` that no PyPI pin can satisfy), and vLLM is an
      **optional** extra (`[serve]`) rather than a hard dependency. Both are the
      right calls — confirm they are what you intend to ship, because a released
      version's metadata cannot be edited.

### 3.7 The name decision is recorded

- [ ] §0.1 still reflects the intent: one distribution, named `gridbook`. If that
      changed, this document changes *before* the upload, not after.

---

## 4. What NOT to do yet — and why

These all look like progress and are actually standing obligations. A solo
researcher's scarcest resource is the recurring cost, not the setup cost.

**1. Do not publish a container image.** The `vllm/vllm-openai` base is ~10.4 GB
compressed and republishes on every vLLM release
(<https://hub.docker.com/r/vllm/vllm-openai>). Publishing a gridbook image means
rebuilding an ~11 GB artifact on someone else's release cadence, forever, and
users will pin the tag you stop rebuilding. `vllm-gguf-plugin` — the closest
precedent, inside the vLLM org — publishes no image at all. **Ship the
`Dockerfile` instead** (ledger row 8): the user builds it, the vLLM pin is
explicit, the recurring cost is one line bump. GHCR is the right registry *if*
that day comes (unlimited public pulls, free bandwidth in Actions, one-month
notice before policy change), never Docker Hub with its pull-rate cliff.

**2. Do not ship a wide prebuilt-wheel matrix.** One `cp310-abi3` wheel per
platform (x86_64 + aarch64), against one CUDA major, is the correct scope for
v0.1.0. A CUDA×arch×torch matrix is a permanent CI burden, and every wheel
uploaded is permanent. Note the specific trap: PyPI **rejects local version
labels**, so a second CUDA variant cannot live on PyPI at all — it has to be a
GitHub Release asset, which is how `vllm-gguf-plugin` handles it.

**3. Do not donate the repo to the `vllm-project` org yet.** It is a real option
and probably the highest-reach one — but it means org review norms, a likely
rename to `vllm-gridbook-plugin`, and shared ownership of the release cadence.
Have that conversation from a position where the wheel is live and the docs are
accurate, not from one where the README's install command is false.

**4. Do not open the HF library-registration PR yet.** Its documented
precondition is that a model already carries `library_name: gridbook` and is
visible at `huggingface.co/models?other=gridbook`, and the generated snippet it
installs on every model page must be a *true* command. Both are downstream of
§3.

**5. Do not post to r/LocalLLaMA, HN, or X yet.** Beyond §3: the subreddit's
actual rules **could not be verified from this environment** (reddit.com and
old.reddit.com are both fetch-blocked). Read the sidebar first. A launch post is
effectively single-shot, and the first thing a capable reader does is try the
install command.

**6. Do not spend attention on the awesome-list channel.**
`BudEcosystem/Awesome-vLLM-plugins` is 8 stars and last pushed 2025-12-12. It
costs a PR and returns approximately nothing. `pprp/Awesome-LLM-Quantization` has
436 stars but is a *papers* list, not a tools list.

**7. The `docs/SPEC.md` registry/config-layout blocker is closed.** Sections 5
and 6 now specify the shipped pointer-sidecar layout, canonical `"gridbook"`
producer key, legacy `"prismaquant"` read alias, and accepted fully inlined
compatibility form. The remaining §3.4 wheel-installed serve smoke is an
implementation/release gate, not a specification contradiction.

**8. Do not claim `vllm-gridbook` "just in case".** §0.1. Two names is two
release lines and a permanent explanation.

---

## 5. Announcement support kit

**This is not a launch post.** Robert writes that. This section is the factual
substrate: the accurate description, the numbers that are safe to quote and where
each came from, the numbers that are **not** safe to quote yet, the correct links,
and the attribution line.

### 5.1 The one-paragraph description (accurate as of 2026-07-28)

> **gridbook** is an open weight-format family and an out-of-tree vLLM plugin.
> **NVFP4-CB** and **FP8-CB** store weights as product vector-quantized codebook
> indices whose codebook entries lie exactly on a hardware numeric grid (E2M1 /
> e4m3) — so a decoded codebook tile is *bit-identical to a standard NVFP4 or FP8
> tile*, and needs no dequantization step before a tensor-core GEMM can consume
> it. That is what the FP8-CB prefill path does today: expand straight to e4m3
> bytes and hand them to the stock CUTLASS W8A8 kernel. The dtype in each name is
> the grid the codebook values live on, not the storage width: storage is a k-bit
> index per 8-weight vector. Tiles are expanded transiently, one layer at a time,
> never resident. The plugin loads through vLLM's documented
> `vllm.general_plugins` entry point on a stock vLLM install — no forked runtime
> and no vLLM-core patches (it does wrap `load_weights` on the specific model
> classes it supports). The format specification is Apache-2.0 and is published to
> be implemented by anyone, in any runtime, without permission.

One-liner, if a shorter form is needed:

> Codebook-quantized weights that decode onto the hardware numeric grid, so they
> serve on native tensor-core kernels instead of paying a dequantization tax — an
> open format spec plus an out-of-tree vLLM plugin.

> ⚠ **Do not broaden the serving-path sentence back out.** An earlier revision
> read *"feeds the same Blackwell tensor-core GEMM vLLM already runs"*, full stop.
> That is true of exactly one of the four live paths. Per the kernel standard
> (`prismaquant/docs/nvfp4-cb-plan/STANDARDS.md`, "Kernel standard (the serving
> surface)"):
>
> | Path | What actually runs |
> |---|---|
> | Prefill dense **fp8-CB** | `cb_expand_fp8` (direct e4m3 bytes) → **stock CUTLASS W8A8** ✔ the claim |
> | Prefill dense **fp4-CB** | Triton v2 expand to **bf16** (composed scales) → **cuBLAS** — not the NVFP4 GEMM |
> | Decode (dense and MoE) | **gridbook's own CUDA GEMV**, not a vLLM GEMM at all |
> | Mid-M 17–128 fp8-CB | gridbook's CUTLASS fused decode-in-prologue |
>
> The bit-identity claim is exact and is the real point; the "same GEMM" claim is
> a generalization of the fp8-CB prefill case. The phrasing is inherited from
> `README.md` and `docs/SPEC.md`, so correcting it there is a separate, worthwhile
> edit — but a launch post is the worst place for the first reader to find it.

### 5.2 Numbers that are safe to quote, with citations

All from `docs/BENCHMARKS.md` in this repo. All measured on **one** NVIDIA GB10 /
DGX Spark (Blackwell `sm_121`, 128 GB unified memory, ~273 GB/s), served through
vLLM with `--enforce-eager`. All quality comparisons are **at matched
bits-per-weight** against the same model quantized to stock per-Linear NVFP4/FP8
and served natively by vLLM's `compressed-tensors` path.

| Claim | Value | Source |
|---|---|---|
| 27B @ 5.5 bpp — ALL-KL vs matched-bpp NVFP4/FP8 baseline | **−58.3%** (0.0134 vs 0.0321) | `docs/BENCHMARKS.md` § "27B (Qwen3-class hybrid)" |
| 27B — confident-KL | **−52.9%** (0.01134 vs 0.02407) | same |
| 27B — matched-bytes denominator | CB body 16.713 GB vs baseline 16.707 GB (**0.04%** apart) | same |
| 27B — PPL gap to BF16 | ~3× smaller (CB +0.043 vs baseline +0.128) | same |
| 27B — decode | **10.27–10.30 tok/s vs the native baseline's 10.26** — parity, which is the ceiling for a bandwidth-bound decode at matched bytes | same |
| 27B — prefill | **1.44× the native baseline** (1.075 s vs 0.746 s TTFT@1400) — an honest deficit, not a win | same |
| 35B MoE @ 4.75 bpp — confident-KL | **−53%** (0.01706 vs 0.03625) | `docs/BENCHMARKS.md` § "35B MoE" |
| 35B MoE — ALL-KL | **−43%** (0.0278 vs 0.0492) | same |
| 35B MoE — decode | **32.6–33.3 tok/s, faster than BF16's 28.43**, within 8% of the native baseline | same |
| 295B @ 2.9 bpp — single-box fit | `model.safetensors` = **110.3 GB**; loads, fits and serves on **one** 128 GB DGX Spark; load 77 s; 44,272 tokens of KV at 4k context | `docs/BENCHMARKS.md` § "295B MoE" |
| 295B — validation bar met | coherent, arithmetically correct generation (17×24=408; 60 mi / 1.5 h = 40 mph; correct recursive Fibonacci) | same |
| 295B — allocator result at a joint menu | offered vanilla NVFP4 **and** FP8 alongside codebook rungs, the measured allocator gave 36 Linears to vanilla FP8 and **zero** to vanilla NVFP4 | same |

**Mandatory caveats to carry with any of the above** (they are in
`docs/BENCHMARKS.md` § "Caveats — read these", and dropping them makes the claim
non-rigorous):

- Single box, single calibration seed. Single-seed KL at this calibration size has
  a ~±10–40% noise band. The headline −58% / −53% are far outside it and are
  corroborated by PPL and top-1 agreement; smaller deltas in those tables should
  not be over-read.
- **±17% measurement-arithmetic sensitivity.** Loading extra CUDA extensions
  shifts allocator addresses and perturbs FP reduction order, moving the
  *measured* KL even when the served bytes are identical. A/B arms must match
  extension residency.
- **The 295B carries no quality-vs-teacher claim.** A 295B BF16 reference cannot
  be run on one box, so there is no KL or PPL-vs-BF16 number at that scale.
- Prefill is honestly uneven: at/above native parity on decode, **not** at parity
  on large-M prefill.
- Reported bpp is over *quantizable* parameters; the excluded BF16 floor is still
  resident on disk, so `bpp × params` understates artifact size.

### 5.3 Numbers that are **BLOCKED** — do not quote until reconciled

**Two entries cleared while this document was being written.** Recorded rather
than deleted, because the reason they were blocked is the reason to re-check
before quoting rather than trusting this table:

| Claim | Status |
|---|---|
| 295B prefill throughput and its ratio to the GGUF IQ build | **CLEARED 2026-07-28 18:04 UTC.** The repo now agrees on **~109 tok/s vs 42 = ~2.6×**: `README.md` — *"~109 tok/s vs the matched-byte GGUF IQ build's 42 = ~2.6×"*; `docs/BENCHMARKS.md` — *"Prefill ~109 tok/s vs the GGUF IQ build's 42 — ~2.6× faster"*; `grep -n '2\.1' docs/MOTIVATION.md` → no output. This is the single best headline in the set — it is the format's whole thesis. **Two conditions on quoting it:** (a) carry the protocol caveat that already sits in BENCHMARKS — the comparison is *"against a different serving stack (llama.cpp CUDA-core IQ dequant)"*, not against another vLLM path; (b) the stale note in `docs/hf-cards/Hy3-295B-…-2.9bit-vllm.md` that still describes the old 2.1×/2.6× split must be deleted before those cards are applied. |
| 295B ToolEvalBench | **CLEARED — it was never a self-contradiction.** `README.md` and both places in `docs/BENCHMARKS.md` say **88 (130/148)** for the gridbook artifact; the `87 (129/148)` is the **GGUF IQ comparator's** score in the same table, and `86 (128/148)` is k-quant's. Quote it exactly as BENCHMARKS already frames it: *"Read it as parity, not a win"* — across serving configs the same bytes measured 85–87 and the GGUF family's own band is 86–87, so +1 sits inside the churn band. |

**Still blocked** (one further entry cleared 2026-07-28 and is struck through
below, same rule: recorded, not deleted):

| Claim | Why it is blocked |
|---|---|
| ~~27B "−77% KL"~~ | **RECONCILED 2026-07-28 — quotable with the cross-reference now in the docs.** The two readouts are different *builds* measured in different *sessions*: the −58.3% is the 2026-07-18 A/B whose CB arm used a 4-rung menu (`K36/K40/K44/K48`), while the Hub artifact is the later 8-rung ladder (`K36`–`K47`, per its shipped `quant_config.json`) measured 2026-07-22. Absolute KL does not survive the session change either — the *same unchanged* NVFP4/FP8 baseline artifact reads confident-KL **0.02407** in the first session and **0.01302** in the second (1.85×), so cross-session absolute comparison is invalid by construction. Recomputing all four arms of the 2026-07-22 session from its stored top-20 dumps with `kl_tool.py compare` reproduces the card exactly: CB **0.0049 / 0.00295**, NVFP4/FP8 baseline **0.0211 / 0.01302**, PrismaSCOUT-5.31 **0.0344 / 0.02491**. Both sessions return the same verdict; **quote the conservative −58.3%**, and link the reconciliation in [`BENCHMARKS.md` § two sessions, two builds](BENCHMARKS.md#two-sessions-two-builds-why-the-model-card-says-77-and-this-page-says-583) whenever both numbers can be seen at once. `README.md` and the model card are now cross-referenced. |
| Anything about the 35B MoE artifact being available | The 35B gridbook artifact is **not published**. The result is real and is in `docs/BENCHMARKS.md`; the model is not downloadable. Say so if the number is used. |

### 5.4 Links

| What | URL |
|---|---|
| Repository | <https://github.com/RobTand/gridbook> |
| Format specification | <https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md> |
| Why it exists | <https://github.com/RobTand/gridbook/blob/master/docs/MOTIVATION.md> |
| Kernel design | <https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md> |
| Measured results + caveats | <https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md> |
| PyPI | `https://pypi.org/project/gridbook/` — **not live yet**; do not publish this link until §2.A step 6 completes |
| HF — 27B (23.0 GB) | <https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm> |
| HF — Laguna-S-2.1 (84 GB) | <https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm> |
| HF — Hy3-295B (110.3 GB) | <https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm> |
| HF Collection | `unverified` — does not exist yet (§2.C5) |
| The quantization pipeline that produces the artifacts | <https://github.com/RobTand/prismaquant> |
| Paper (arXiv) | `unverified` — no arXiv ID established at the time of writing |

Note the default branch is **`master`**, not `main` — permalinks must use
`/blob/master/`.

### 5.5 Attribution

```
Robert Tand — robert.tand@icloud.com
```

`robert.tand@icloud.com` is the **only** address that appears in any public file,
commit, package metadata, model card, PR, or post. The dev/system address used on
this box is never published — check `git config user.email` on any machine that
authors a public commit, and grep the built sdist and wheel (§3.6).

License line: **Apache-2.0**, both for the code and for the format specification.

---

## 6. Open items and unverified facts

Recorded so nobody re-derives them, and so nothing here is mistaken for
established.

- **r/LocalLLaMA rules: `unverified`.** reddit.com and old.reddit.com are both
  fetch-blocked from this environment. Third-party summaries describe a ~1-in-10
  self-promotion ratio, required affiliation disclosure, and removal of low-effort
  or primarily LLM-generated posts. Read the actual sidebar before posting.
- **`NVIDIA/dgx-spark-playbooks` third-party contribution policy: `unverified`.**
  No written policy in the README; the process is inferred from the open PR queue.
- **HF library-registration acceptance: not guaranteed.** The guide is framed
  around libraries that themselves load from the Hub. Treat as a cheap attempt,
  not a plan dependency.
- **The only known-good serving stack is not a released anything.** Every
  validated run used a community arm64 image (`eugr/spark-vllm`, digest
  `sha256:d0840ff0…`, created 2026-07-03) carrying an *unreleased* vLLM source
  build `0.23.1rc1.dev764+g54b16d8a9`, with the plugin installed via
  `pip install -e --no-deps`. No released vLLM, no x86 stack, and no non-editable
  install has ever been exercised end to end. Any compatibility table must label
  every other cell inferred. This is also *why* the packaging bug survived: the
  only install mode ever tested is the one mode that hides it.
- **PyPI availability was checked 2026-07-28** and could change. Re-check
  immediately before the real-claim tag.
- **The `csrc` fix, the CI/release workflow, the Dockerfile, the README/INSTALL
  rewrite, the HF card staging, and the two issue-#1 fixes are tracked separately**
  (tasks #39–#43, #45, #46). This document is the decision record; it is not the
  implementation.
- **A bug report exists that no task covers.** The reporter's 2026-07-28 17:59 UTC
  follow-up on issue #1 says that with the mid-M fused prefill path enabled by
  default, `cb_qweight` and `_cb_qw_padded` are *both* live at runtime and he
  measures ~35 GB on a 32 GB card. Task #46 is the closest match but was scoped as
  a footprint item; this is a hard OOM on the most common consumer Blackwell card.
  Someone should decide whether it is #46 or a new one. **It is also the first
  independent report of anyone trying to run gridbook** — which is worth more than
  the download table in §1.1.
- **Everything in `docs/hf-cards/` is staged, not applied.** Two of its notes are
  now stale in the direction that matters: the Hy3 card's "2.1× vs 2.6×"
  contradiction note (resolved — §5.3), and `_COLLECTION.md`'s use of the
  unreconciled `−77%` 27B figure (still blocked — §5.3). Applying the directory
  as-is publishes both.

---

## 7. Revision note — what changed on 2026-07-28 and why

An adversarial verification pass found this document asserting state that was true
of tree (A) and false of tree (B), while (B) is the tree releases are built from.
The corrections, so the reasoning is not re-litigated:

| Was | Now |
|---|---|
| §0.2 *"Done: the tree is now at 0.1.0"* | Scoped to (A), with the rsync time and the CI guard that makes it mechanical |
| §3.1 / §3.6 `[x]` on packaging and version | `[~]`, meaning *landed in (A), confirm in (B)*, each with the command that confirmed it |
| No gate on the rsync at all | **§3.0**, first and blocking, with the measured three-build table showing what a build in the un-synced window produces |
| §3.2 built "wherever you are" | Explicitly from (B); `check_dist.py`'s second argument explained |
| §3.4 *"`ROADMAP.md` calls that permanent"* | Removed — fabricated; `ROADMAP.md` already agrees with the box |
| §3.5 ROADMAP TODO (Persistent-N, batched MoE) | Removed — one item was already recorded as a measured negative, the other does not exist in `ROADMAP.md` |
| §3.5 BENCHMARKS/MOTIVATION contradictions, missing docs | Closed with quoted evidence; only the 27B KL gap remains |
| §2.B *"issue #1 has zero replies"* | Two comments; acknowledged but unanswered, and it has grown a new bug |
| §1 *"237 libraries"*, §1.2 *"93 releases"* | 239 and 101, both re-measured |
| §1.1 *"the install gate — not the format — is the bottleneck"* | Hedged, with the 27B counter-observation (~241/day behind the same gate) stated first |
| §5.1 *"feeds the same tensor-core GEMM vLLM already runs"* | Narrowed to what `STANDARDS.md` actually supports, with the four-path table |
| Line-number citations into a moving tree | Quoted strings throughout (see the header) |

Two mechanism changes, not just wording: **§3.0** exists so no future revision can
tick a box on the wrong tree, and `MANIFEST.in` now excludes this file and
`docs/hf-cards/` from the sdist so internal working material cannot reach a
permanent PyPI artifact.

---

*Written 2026-07-28; corrected the same day. If this document and the code
disagree, the code wins and this document is stale — fix it here rather than
propagating it.*
