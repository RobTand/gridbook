# Staged Hugging Face card blocks

**Nothing in this directory has been uploaded.** These are *staged* model-card
edits for the three published `gridbook` artifacts on the Hub. A human applies
them; no agent or CI in this repo pushes to Hugging Face.

## What is here

| File | What it is |
|---|---|
| [`_TEMPLATE.md`](_TEMPLATE.md) | The standard "Serve this model" block, with substitution variables. Edit this one when the install story changes, then re-render the three below. |
| [`Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm.md`](Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm.md) | Rendered block + front matter for the 27B VL artifact |
| [`Laguna-S-2.1-prismaquant-gridbook-6bit-vllm.md`](Laguna-S-2.1-prismaquant-gridbook-6bit-vllm.md) | Rendered block + front matter for the Laguna-S-2.1 117B MoE artifact |
| [`Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm.md`](Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm.md) | Rendered block + front matter for the Hy3 295B MoE artifact |
| [`_METADATA.md`](_METADATA.md) | Which card metadata actually filters on the Hub, measured; `library_name` / `base_model` / `pipeline_tag` recommendations |
| [`_COLLECTION.md`](_COLLECTION.md) | Proposed HF Collection: exact title, description, members, and creation steps |

Each per-model file has two fenced regions to copy — a YAML **front matter**
block and a markdown **body** block — plus an un-fenced "notes" section that is
*not* part of the card (rationale, open gaps, what is measured vs. unverified).

---

## APPLY GATE — read before pasting anything

The rendered blocks tell a reader to run `pip install gridbook`. **That command
is not yet true.** Three preconditions, **none of which is satisfied by this
repository as published today**. Check each one against the *published* repo —
not against a local development tree — because that is what a reader installs.

**1. Packaging — fixed upstream, NOT YET IN THIS REPO.**

`gridbook/cuda_ext.py` resolves the CUDA sources repo-root-relative
(`.../gridbook/../csrc/...`) while `pyproject.toml` ships only the `gridbook*`
package. Under any **non-editable** install, `<site-packages>/csrc` does not
exist, every extension build raises `FileNotFoundError`, and the plugin
fail-softs to the slow Triton path. Confirmed in the wild: the "Additional note"
in [issue #1](https://github.com/RobTand/gridbook/issues/1).

The fix — sources moved *inside* the package as `gridbook/csrc/`, resolved
through `importlib.resources`, plus a distinct `IncompleteInstallError` so a
packaging defect never again reads as "no nvcc" — exists in the upstream
development tree and **has not been synced here**. Measured on `master` at
`b4694e3` (2026-07-28):

```bash
ls gridbook/csrc                                    # -> No such file or directory
grep -n 'os.pardir' gridbook/cuda_ext.py            # -> 35, 96, 146
grep -c 'importlib.resources' gridbook/cuda_ext.py  # -> 0
```

Gate: all three must invert (directory exists, no `os.pardir` hits, at least one
`importlib.resources` hit) before any card claiming `pip install gridbook` is
applied.

**2. Public repo synced — REQUIRED, and it is not only about packaging.**

Every card's "How to tell the fast path is active" section tells a reader to
grep the `[prismaquant-cb]` stderr tag. That tag is stable across both trees;
the *sentences* around it are not, and the sample lines quoted in the cards are
the upstream wording, which this repo does not yet emit:

```bash
grep -n 'could not be built' gridbook/cuda_ext.py   # -> no matches at b4694e3
```

The blocks are written to survive that (they say "grep the tag, not the
sentence", and hedge the samples with "read roughly"), but at apply time the
sample lines should still be re-read off the *released* plugin so they match
what a user actually sees.

**3. PyPI — STILL OPEN.** `https://pypi.org/pypi/gridbook/json` → **404**
(re-checked 2026-07-28). The name is unclaimed; the first upload is a human
action and is irreversible in ways worth reading
[`../DISTRIBUTION.md`](../DISTRIBUTION.md) about first.

So:

> **Do not apply these card *bodies* until (1)+(2) this repo is synced and
> carries the packaging fix, (3) `pip install gridbook` resolves on PyPI, and —
> the real test — a clean-venv install of that wheel builds the CUDA extension
> on a Blackwell box.**

That last one is what actually matters; (1)–(3) are inputs to it. A clean-venv
install is the only check that exercises the exact code path issue #1 fell into
— an editable install cannot, because `-e` leaves the repo-root `csrc/` on disk
and hides the bug. It is a GPU + nvcc test, so it is not run from here.

**What is *not* gated:** the front-matter-only edit — `library_name`,
`pipeline_tag`, `base_model_relation`, `tags` — makes no install claim, so it can
land today, independent of all three preconditions. See Route B (b) below and
[`_METADATA.md`](_METADATA.md). The same is true of the two Gemma4 `base_model`
backfills, which touch no gridbook repo at all.

If the PyPI publish is deferred, re-render with the `GIT_TAG` install variant in
[`_TEMPLATE.md`](_TEMPLATE.md) — but only after a tag exists, **and only after
(a)**, since an untagged/unfixed `git+` install is exactly the broken path the
live Laguna card already ships. There are **0 releases and 0 tags** on
`RobTand/gridbook` as of 2026-07-28, and two of the three live cards already
cite plugin versions *by date*, one of which — "plugin ≥2026-07-28" — has no
corresponding public commit.

---

## How to apply (documentation — these commands are not run here)

Cards live at the repo root as `README.md`. Two supported routes — but both
start from an assembled card file, so build that first.

### Step 0 — assemble the card file (both routes need it)

Nothing in this directory is a ready-to-upload card. Each per-model file holds a
fenced **front matter** block and a fenced **body** block that have to be joined
to the parts of the live card being kept:

```bash
REPO=rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm
OUT=$PWD/$(basename $REPO).README.md   # the assembled card. NOTE: the file next
                                       # to this README is the *staging* file
                                       # ($(basename $REPO).md) — it is not
                                       # uploadable as-is.
```

Into `$OUT`, in order:

1. `---`, the staged front matter, `---` — replacing the live card's YAML block
   entirely.
2. The live card's H1 and its one-paragraph description, unchanged.
3. The staged body block — which **replaces** the live `## Serving` / `## Serve`
   section, i.e. exactly where the stale install commands live.
4. The rest of the live card from its first `## Measured …` / `## Format …`
   heading onward, unchanged.

### Route A — git (recommended: reviewable diff, atomic)

```bash
# One-time: a token with write scope on the rdtand namespace
hf auth login                      # (older CLI: huggingface-cli login)

# GIT_LFS_SKIP_SMUDGE=1 fetches LFS pointers, not the 23-106 GB of shards.
# Never clone one of these repos without it; a card edit does not need weights.
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/$REPO ./$(basename $REPO)
cd ./$(basename $REPO)

cp "$OUT" README.md                # the assembled card from Step 0
git add README.md
git commit -m "card: standard gridbook serve block + metadata"
git push
```

### Route B — `hf upload` / `HfApi` (card-only, no weights, no clone)

```bash
# CLI (huggingface_hub >= 0.34 ships `hf`; `huggingface-cli` is the older alias)
hf upload $REPO $OUT README.md \
  --repo-type model \
  --commit-message "card: standard gridbook serve block + metadata"
```

```python
# Python — same thing, plus a metadata-only variant that leaves the prose alone
from huggingface_hub import HfApi, ModelCard
api = HfApi()

REPO = "rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm"
OUT  = "./Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm.README.md"

# (a) whole card
api.upload_file(
    path_or_fileobj=OUT,
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
    commit_message="card: standard gridbook serve block + metadata",
)

# (b) front matter only — edit metadata, keep the existing body verbatim.
#     Useful to land the discovery metadata before the APPLY GATE clears,
#     since it makes no install claim.
card = ModelCard.load(REPO)
card.data.library_name = "gridbook"
card.data.pipeline_tag = "image-text-to-text"
card.data.base_model_relation = "quantized"
card.data.tags = [...]          # see _METADATA.md
card.push_to_hub(REPO)
```

### Preview before pushing

```bash
python -c "
from huggingface_hub import ModelCard
c = ModelCard(open('$OUT').read())
c.validate()          # raises on malformed front matter
print(c.data)
"
```

---

## Why this matters (the measured case)

Pulled from the HF API on 2026-07-28:

- `rdtand/` totals **913,796** all-time downloads across 18 repos.
- The three **gridbook** artifacts hold **1,807** of that: 27B 1,464 · Laguna 311 · Hy3 32.
- The sharpest available comparison holds base model *and* bit class fixed:
  `Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` (created 2026-07-11, 5,694 downloads
  ≈ 335/day) vs `Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm` (created
  2026-07-21, 32 downloads ≈ 4.6/day). The GGUF artifact's serving path is one
  `pip install vllm-gguf-plugin`; the gridbook artifact's is a git clone plus
  nvcc JIT.

Caveats, stated plainly: n=1 model pair, observational, confounded by card
quality and by the gridbook repos' thinner metadata. HF counts a download per
HTTP request to `config.json` when no library override exists, so these are
request counts, not users. This is a **lead**, not a proof — but it points at the
install gate, and the install gate is fixable.

---

## Hard constraints these files obey

- Public attribution is **robert.tand@icloud.com** only, on every card. No other
  address appears in any card content.
- Every number in the staged blocks is copied from an existing card, from
  `docs/BENCHMARKS.md`, or from the HF API, and is cited in the per-file notes.
  Nothing is inferred, rounded, or re-derived.
- Where a fact is *not* known (e.g. the vLLM version used for the 27B and Laguna
  measurements), the block says so instead of guessing.
