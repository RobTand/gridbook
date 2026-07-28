# Proposed HF Collection

`rdtand` currently has **zero** collections (`GET /api/collections?owner=rdtand`
→ `[]`, 2026-07-28). Collections render on the profile page, are shareable as a
single link, can be pinned, and give the three gridbook artifacts one URL to
point at from the repo README, a PyPI project page, a Spark playbook, or a post.

**Nothing below has been created.** Creating a collection is a mutating API call;
that is a human action.

---

## Collection 1 — the gridbook family (create this one)

**Title** (max 100 chars on the Hub; this is 62):

```
gridbook — codebook-quantized models served by stock vLLM
```

**Description:**

```markdown
Product-VQ codebook weight formats (NVFP4-CB / FP8-CB) whose codebook values sit
exactly on hardware grids, so a decoded tile is a bit-standard NVFP4/FP8 tensor
and runs on the same Blackwell tensor-core GEMM vLLM already uses. Served through
an out-of-tree plugin — stock vLLM, no fork, no core patches.

Formats, kernels and the Apache-2.0 spec: https://github.com/RobTand/gridbook
Allocation is measured per-Linear by PrismaQuant: https://github.com/RobTand/prismaquant

Requires a Blackwell GPU (sm_120 / sm_121) and nvcc in the serving container.
Read each card's "Serve this model" block before downloading — these are large
artifacts with specific serve flags.
```

**Members, in this order** (smallest / most accessible first — a collection is
read top-down, and the 23 GB artifact is the only one a non-Spark owner can even
attempt):

| # | Repo | Item note (Collections support a per-item note) |
|---|---|---|
| 1 | `rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm` | 23 GB · vision-language + MTP · the **quality-validated** one: −77% held-out KL vs the matched-size NVFP4+FP8 artifact, ToolEvalBench 87 |
| 2 | `rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm` | 89 GB · 117B sparse MoE coding model · full **256k context on one DGX Spark**, 14.9 tok/s decode · no quality claims published |
| 3 | `rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm` | 106 GB · 295B-A21B MoE at 2.9 bpp on **one** 128 GB Spark, MTP drafter included · no quality claims published |

Every note above is copied from the artifact's own card; nothing is re-derived.
The "no quality claims" flags are kept deliberately — the cards say it, so the
collection says it.

### Optionally: add the two baseline arms

Adding the artifacts the gridbook claims are measured *against* makes the
collection self-contained for anyone who wants to reproduce the comparison:

| Repo | Item note |
|---|---|
| `rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm` | **baseline arm** — same model, same 5.5 bpp, conventional NVFP4+FP8. The 27B KL comparison is against this. |
| `rdtand/Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` | **baseline arm** — same model, same bit class, GGUF IQ. The 295B prefill comparison is against this. |

Tradeoff: it strengthens the honesty story and gives a reader something to A/B,
but it dilutes "this collection is the gridbook family" and puts two non-gridbook
artifacts under a gridbook title. **Recommendation: create the collection with
the three gridbook artifacts only, and link the baselines from each card instead**
— the cards already name them, and the staged blocks add the cross-links.

---

## Collection 2 — the whole PrismaQuant family (optional, later)

A second collection grouping all 18 `rdtand/` artifacts would surface ~914k
downloads' worth of repos under one shareable link, and give the older,
high-traffic artifacts a visible path to the newer formats. Suggested title:

```
PrismaQuant — measured per-Linear format allocation, vLLM-servable
```

Do this *after* Collection 1 exists and the cards are updated; a collection of
cards that still contradict each other is not worth advertising.

---

## How to create it (documentation — not run here)

### Route A — the Hub UI (recommended for the first one)

1. Go to `https://huggingface.co/rdtand` → **Collections** tab → **New collection**.
2. Paste the title and description above. Set visibility **Public**.
3. Add each repo: **Add item** → paste the full repo id → select the model.
4. Drag into the order above, then use the item **⋮ → Add note** to paste each
   item note.
5. Optionally pin the collection to the profile.

### Route B — `huggingface_hub`

```python
from huggingface_hub import HfApi
api = HfApi()   # requires a token with write scope on the rdtand namespace

col = api.create_collection(
    title="gridbook — codebook-quantized models served by stock vLLM",
    namespace="rdtand",
    description=(
        "Product-VQ codebook weight formats (NVFP4-CB / FP8-CB) whose codebook "
        "values sit exactly on hardware grids, so a decoded tile is a "
        "bit-standard NVFP4/FP8 tensor and runs on the same Blackwell "
        "tensor-core GEMM vLLM already uses. Served through an out-of-tree "
        "plugin — stock vLLM, no fork, no core patches.\n\n"
        "Formats, kernels and the Apache-2.0 spec: "
        "https://github.com/RobTand/gridbook\n"
        "Allocation is measured per-Linear by PrismaQuant: "
        "https://github.com/RobTand/prismaquant\n\n"
        "Requires a Blackwell GPU (sm_120 / sm_121) and nvcc in the serving "
        "container. Read each card's \"Serve this model\" block before "
        "downloading — these are large artifacts with specific serve flags."
    ),
    private=False,
    exists_ok=True,
)

items = [
    ("rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm",
     "23 GB · vision-language + MTP · the quality-validated one: -77% held-out "
     "KL vs the matched-size NVFP4+FP8 artifact, ToolEvalBench 87"),
    ("rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm",
     "89 GB · 117B sparse MoE coding model · full 256k context on one DGX "
     "Spark, 14.9 tok/s decode · no quality claims published"),
    ("rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm",
     "106 GB · 295B-A21B MoE at 2.9 bpp on one 128 GB Spark, MTP drafter "
     "included · no quality claims published"),
]

for repo_id, note in items:
    api.add_collection_item(
        collection_slug=col.slug,
        item_id=repo_id,
        item_type="model",
        note=note,
        exists_ok=True,
    )

print(f"https://huggingface.co/collections/{col.slug}")
```

### After it exists

Link the collection URL from:

- each of the three model cards (one line at the top of the "Serve this model"
  block, replacing the per-card "Other gridbook artifacts:" list once the
  collection is a stable URL);
- the repo `README.md` — its "Published artifacts" table links the three repos
  individually; a collection link belongs next to that table as the one URL to
  hand out. (At published `master` `b4694e3` the README linked **no** artifact at
  all; the table is part of the concurrent docs rewrite and may not be pushed
  yet — check before describing the README's state anywhere public.)
- `[project.urls]` in `pyproject.toml` as a `Models` entry;
- the GitHub repo's **Homepage** field, which is currently empty.
