# Card metadata that actually drives Hugging Face discovery

Measured against the live Hub API on **2026-07-28**. Every count below is a real
query, not an estimate. Method:

```bash
# number of models carrying a free-form tag
curl -s "https://huggingface.co/api/models?filter=<tag>&limit=1000" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))'
```

`1000` means "capped at my `limit=1000`", i.e. **≥1000** — not an exact count.

---

## 1. What the three gridbook repos have today

| Field | 27B | Laguna | Hy3 |
|---|---|---|---|
| `library_name` | — | — | — |
| `pipeline_tag` | — | — | — |
| `base_model` | `Qwen/Qwen3.6-27B` ✅ | `poolside/Laguna-S-2.1` ✅ | `tencent/Hy3` ✅ |
| `base_model_relation` | — (inferred) | — (inferred) | — (inferred) |
| `license` | apache-2.0 | openmdw-1.1 | apache-2.0 |
| tags | 4 | 8 | 7 |

**Five** of the 18 `rdtand/` repos have neither `library_name` nor
`pipeline_tag`, and these three are three of them — so they appear in **no task
filter**, get no widget or code snippet, and are reachable essentially only by
direct link. The other two are `Hy3-295B-A21B-PrismaQuant-2.8bit-gguf-vllm` and
`Hy3-295B-A21B-PrismaQuant-5.3bit-2xSpark-vllm`; the GGUF one is the arm this
repo's README uses as its headline prefill comparison, so it is worth fixing in
the same pass even though it is not a gridbook artifact.

Measured by iterating `GET /api/models?author=rdtand&limit=100` and reading
`cardData.library_name` / `cardData.pipeline_tag` on each of the 18.

For contrast, the two high-traffic siblings. Two tag counts exist for any repo —
the **author-written** `cardData.tags`, and the **derived** `tags` the Hub
returns (author tags plus `base_model:…`, `license:…`, `region:us`, the
safetensors dtype tag, the model-type tag, and, where set, `pipeline_tag` and
`language`). Both are quoted here because only the first is editable:

| Repo | downloads | `library_name` | `pipeline_tag` | author tags | derived tags |
|---|---|---|---|---|---|
| `Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` | 389,620 | `vllm` | — | **9** | **16** |
| `Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` | 297,483 | `vllm` | `image-text-to-text` | **14** | **25** |

PrismaSCOUT's 9 author tags include `compressed-tensors`, `nvfp4`,
`mixed-precision`, `blackwell`; the 35B additionally carries
`base_model_relation: quantized` and `language`, which is most of why its derived
list is 9 longer than its author list.

---

## 2. Which tags actually filter — measured

| Tag | Models carrying it | Verdict |
|---|---|---|
| `codebook-quantization` | **1** (only the 27B) | keep — it is the precise term, and being the sole occupant is fine |
| `gridbook` | **4** (our 3 + one third-party fork-ish repo) | keep — this is the brand pool |
| `codebook` | **14** | keep — natural search word, tiny pool, page-one visibility |
| `vector-quantization` | **47** | **add** — this is what researchers call the technique; currently on none of our repos |
| `single-gpu` | 24 | keep where already present (Laguna, Hy3) |
| `dgx-spark` | **218** | **add to all three** — literally the reference hardware; small enough for page-one browse |
| `blackwell` | **453** | **add to all three** — the required GPU class |
| `fp4` | 471 | optional; `nvfp4` is the more precise term and is what the family already uses |
| `mixed-precision` | 562 | **add** — accurate (per-Linear format allocation is the whole thesis) and mid-size pool |
| `nvfp4` | ≥1000 | keep/add |
| `fp8` | ≥1000 | **add** — every gridbook artifact has FP8-CB and/or vanilla FP8 content, and none of the three carries this tag today |
| `vllm`, `quantized`, `moe`, `conversational`, `long-context`, `mtp`, `speculative-decoding`, `code`, `multimodal`, `vision-language` | ≥1000 each | generic — near-zero discovery value on their own, but they are the vocabulary the Hub's facets and third-party crawlers key on, and they cost nothing |

**Reference points for what a quantized repo conventionally tags** (fetched the
same day):

- `RedHatAI/Qwen3-32B-NVFP4`: `pipeline_tag: text-generation`, `library_name`
  **unset**, cardData tags `["fp4", "vllm"]`. The Hub *auto-derives* the rest —
  `compressed-tensors`, `8-bit`, `conversational`, `base_model:quantized:…`,
  `region:us` — so its visible tag list is much longer than what the author wrote.
- `Qwen/Qwen3-32B-AWQ`: `library_name: transformers`, `pipeline_tag:
  text-generation`; `awq` and `4-bit` are auto-derived.
- `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16`: author tags include
  `int4`, `vllm`, `compressed-tensors`, `gptq`, `llmcompressor`, `neuralmagic` —
  i.e. the *toolchain* is tagged, not just the format. `gridbook` and
  `prismaquant` are our equivalents.

**Auto-derived, do not hand-write:** `8-bit` / `4-bit` (from the safetensors
dtype index), `conversational` (from a chat template + a text `pipeline_tag`),
`base_model:quantized:<id>` (from `base_model`), `region:us`, `custom_code` (from
`auto_map`), and the model-type tag (`qwen3_5`, `laguna`, `hy_v3`).

---

## 3. `library_name` — recommend `gridbook`

**Decision: set `library_name: gridbook` on all three gridbook repos**, and keep
`vllm` as a plain tag.

Verified facts behind that:

1. **`vllm` is not a registered HF library.** Downloaded
   `huggingface.js/packages/tasks/src/model-libraries.ts` (1,802 lines) and
   brace-parsed the `MODEL_LIBRARIES_UI_ELEMENTS` object to enumerate its
   top-level keys: **238** entries, of which `vllm`, `sglang`,
   `compressed-tensors` and `gguf` are **absent**, while `transformers`, `mlx`,
   `peft`, `timm` and `diffusers` are present. (There is no `llama.cpp` key
   either — the nearest is `llama-cpp-python`; the GGUF ecosystem is not
   represented by a library entry of its own.) So the `library_name: vllm` on the
   flagship repos is already an *unregistered* value — the Hub accepts arbitrary
   strings here.
2. **An unregistered value does not zero the download counter.** Per HF's
   [download-stats docs](https://huggingface.co/docs/hub/en/models-download-stats),
   a library only changes counting if it declares a `countDownloads` override in
   that file; with no entry, the default query files (`config.json`, …) apply —
   the same as leaving the field unset. Empirical check: the `vllm`-labelled
   flagship has 389,620 all-time downloads, so unregistered labels plainly count.
3. **It is the documented precondition for registering the library upstream.**
   HF's [adding-libraries guide](https://huggingface.co/docs/hub/en/models-adding-libraries#register-your-library)
   asks that at least one model already carry `library_name: <x>` and be visible
   at `huggingface.co/models?other=<x>` before you open the PR. Registration
   would buy a pretty label, a link to this repo and its docs **on every model
   page**, and a generated "how to run this" snippet — i.e. the `pip install
   gridbook` + `vllm serve` block rendered natively at the exact point of
   friction. Acceptance is not guaranteed (the guide is framed around libraries
   that load *from* the Hub), so treat it as a cheap attempt, not a dependency.
4. **Nothing is lost.** These artifacts are not `transformers`-loadable and have
   no inference widget either way; the `vllm` discovery value is preserved by
   keeping `vllm` in `tags`.

**Do not** set `library_name: transformers` — it would advertise a load path that
does not exist for these files and invite `AutoModel` bug reports.

---

## 4. `base_model` — already working here, broken on two other repos

All three gridbook repos have `base_model` set and the Hub has already derived
`base_model:quantized:<id>` on each, which is what creates the **backlink from
the original model's page** (the "Quantizations" entry in the base model's model
tree). That is a genuine discovery channel: someone browsing `Qwen/Qwen3.6-27B`
or `tencent/Hy3` sees the artifact without ever having heard of gridbook.

Adding `base_model_relation: quantized` explicitly is still worth doing — HF's
[model-release checklist](https://huggingface.co/docs/hub/en/model-release-checklist)
recommends it for quantized derivatives, and it removes the dependence on
inference.

**Two non-gridbook repos are missing the backlink entirely** and should be
backfilled (cheap, and they are the same family):

- `rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm` (1,594 downloads)
- `rdtand/Gemma4-31B-IT-PrismaQuant-5.5bit-vllm` (3,186 downloads)

Both declare `base_model: ["google/gemma-4-31b-it"]` in cardData, yet **no
`base_model:` tag was derived** — they are invisible in Gemma 4's model tree.

Root cause, verified: the canonical repo id is **`google/gemma-4-31B-it`** with a
capital `B`. `…/api/models/google/gemma-4-31b-it` 307-redirects to the capital-B
id, and tag derivation does not follow the redirect. Measured:

```
filter=base_model:quantized:google/gemma-4-31B-it  ->  5+ repos
   (unsloth/gemma-4-31B-it-GGUF, nvidia/Gemma-4-31B-IT-NVFP4,
    RedHatAI/gemma-4-31B-it-FP8-block, unsloth/gemma-4-31B-it-NVFP4, …)
filter=base_model:quantized:google/gemma-4-31b-it  ->  0 repos
```

Fix = correct the **case** of the id, plus `base_model_relation: quantized`:

```yaml
base_model: google/gemma-4-31B-it
base_model_relation: quantized
```

**The list form is not the problem — do not "fix" that too.**
`nvidia/Gemma-4-31B-IT-NVFP4` declares `base_model: ['google/gemma-4-31B-it']`
(a list) and *does* get `['base_model:google/gemma-4-31B-it',
'base_model:quantized:google/gemma-4-31B-it']` derived. The lowercase `b` is the
whole cause; a string is used above only because a single base model reads better
as one. Measured:

```
nvidia/Gemma-4-31B-IT-NVFP4        base_model=['google/gemma-4-31B-it']  -> 2 derived base_model tags
rdtand/Gemma4-31B-IT-PrismaQuant-6bit-vllm    base_model=['google/gemma-4-31b-it']  -> []
rdtand/Gemma4-31B-IT-PrismaQuant-5.5bit-vllm  base_model=['google/gemma-4-31b-it']  -> []
```

---

## 5. `pipeline_tag`

| Repo | Recommended | Why |
|---|---|---|
| 27B | `image-text-to-text` | `config.json` is `Qwen3_5ForConditionalGeneration` with `vision_config` + `image_token_id`; the card documents a quantized ViT verified on image inputs. Matches the sibling repos of the same base model. |
| Laguna | `text-generation` | `LagunaForCausalLM`, text-only coding model |
| Hy3 | `text-generation` | `HYV3ForCausalLM`, text-only |

Setting it is what puts the repo in a task facet at all, and it is the trigger
for the auto-derived `conversational` tag (all three ship a `chat_template.jinja`).

---

## 6. Optional levers, deliberately left out of the staged blocks

- **`language`.** The sibling `Qwen3.6-27B-PrismaQuant-5.5bit-vllm` declares
  `["en","zh"]` and the 35B declares `["en","multilingual"]`. Defensible to carry
  over for the 27B (same base model), but it is a claim about the *base* model
  that none of the gridbook cards currently makes, so it is not in the staged
  front matter. Add it if you want the language facets.
- **`arxiv:<ID>` linkage.** The Hub auto-extracts an `arxiv:` tag when a card
  links an arXiv abstract, which cross-links the model to its HF Paper page.
  Since Papers-with-Code shut down (July 2025), this is the surviving
  paper→code linkage. Worth adding to every card **once the AURA paper has an
  arXiv ID** — it is not a card edit that can be staged today.
- **`inference: false`.** Present on some sibling repos; harmless, mildly
  clarifying for artifacts that no HF inference provider can run.

---

## 7. Recommended order of operations

0. **Ungated, do any time:** the front-matter-only edits on the three gridbook
   repos (`library_name`, `pipeline_tag`, `base_model_relation`, `tags`) and the
   two Gemma4 `base_model` case fixes. None of these makes an install claim, so
   none of them waits on the APPLY GATE. Route B (b) in
   [`README.md`](README.md) is the metadata-only call.
1. **Sync the packaging fix into this repo, then publish the wheel** — otherwise
   the cards' *body* install line is still false and the metadata just drives
   more traffic into a broken door. See the APPLY GATE; the fix is upstream but
   not in published `master` `b4694e3`.
2. Apply the three staged card **bodies** (the front matter will already be
   there from step 0).
3. Consider the same `library_name` / `pipeline_tag` treatment for the other two
   repos that carry neither — the GGUF Hy3 arm and the 2×Spark Hy3 arm (§1).
4. Create the Collection — see [`_COLLECTION.md`](_COLLECTION.md).
5. Only then open the `huggingface.js` library-registration PR, since its snippet
   must be a true command.
