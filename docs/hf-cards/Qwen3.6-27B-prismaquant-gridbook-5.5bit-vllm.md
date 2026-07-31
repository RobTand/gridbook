# `rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm` — staged card block

Staged, **not uploaded**. Read the APPLY GATE in [`README.md`](README.md) first.

---

## 1. Front matter (replaces the existing YAML block)

```yaml
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
base_model_relation: quantized
library_name: gridbook
pipeline_tag: image-text-to-text
tags:
  - gridbook
  - codebook-quantization
  - codebook
  - vector-quantization
  - prismaquant
  - vllm
  - quantized
  - mixed-precision
  - nvfp4
  - fp8
  - blackwell
  - dgx-spark
  - multimodal
  - vision-language
  - mtp
  - speculative-decoding
  - conversational
```

## 2. Body block (insert after the intro paragraph; replaces `## Serving`)

````markdown
## Serve this model

| | |
|---|---|
| **Plugin** | [`gridbook`](https://github.com/RobTand/gridbook) — an out-of-tree vLLM quantization plugin. Stock vLLM, no fork, no core patches. |
| **GPU** | NVIDIA Blackwell, compute capability `sm_120` / `sm_121`. Measured on GB10 / DGX Spark (`sm_121`). On older GPUs the plugin still loads but runs its **Triton fallback** kernels — correct, not fast, and not a production serving target. |
| **Memory** | **20.8 GiB weights** + KV cache, measured on one 128 GB unified-memory GB10 (`Model loading took 20.82 GiB`). Before gridbook 0.1.0 the same artifact loaded at **35.86 GiB** — dense codebook weights were resident twice ([issue #1](https://github.com/RobTand/gridbook/issues/1), fixed in [0.1.0](https://github.com/RobTand/gridbook/releases/tag/v0.1.0)); **use 0.1.0 or newer**. A 32 GB consumer Blackwell (RTX 5090, `sm_120`) should now fit weights plus a useful KV budget, and the issue reporter measured 20.54 GiB with 32k context using equivalent patches — but **we have not verified 0.1.0 on a 32 GB card ourselves**. |
| **vLLM** | Served here on the `vllm-node` image used for the 0.1.0 verification; also exercised on 0.25.1 and 0.23.1rc1.dev1060 by an external reporter ([issue #1](https://github.com/RobTand/gridbook/issues/1)). The plugin surface is small but vLLM internals drift — pin a version you have tested. |
| **Toolchain** | CUDA toolkit with `nvcc` on `PATH` **in the serving container** — the plugin JIT-builds its kernels on first model load (~30 s, cached). `nvcc` 13.0 is the tested toolchain. |
| **Parallelism** | Single GPU (`tp=1`). Tensor parallelism is not implemented in the plugin. |

```bash
pip install gridbook

vllm serve rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Check it answers:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
```

**How to tell the fast path is active.** The plugin *fail-softs*: if the CUDA
extension cannot be built it still serves, through Triton fallback kernels, at a
fraction of the speed. That is not a crash — it is a line on stderr, tagged
`[prismaquant-cb]`:

```bash
vllm serve ... 2>&1 | grep '\[prismaquant-cb\]'
```

Any `WARNING` or `ERROR` on that tag means the CUDA decode path did **not**
load, and the numbers on this card are **not** reachable on your box. (The tag
is also used for a few harmless informational lines; it is the `WARNING` /
`ERROR` ones that matter.) The exact wording differs between plugin versions —
**grep the tag, not the sentence.** The two that matter read roughly:

```
[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV extension could not be built (<ExcType>: …); falling back to the Triton decode path (slow prototype). To get the CUDA path: …
[prismaquant-cb] ERROR: broken gridbook install — gridbook is installed without its CUDA sources: …
```

The first is an environment problem (usually no `nvcc` in the serving
container, or a torch/CUDA version mismatch); the second is a defect in the
install itself. Both are diagnosed in
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md#the-cuda-extension-did-not-load-triton-fallback).
Set `PRISMAQUANT_CB_EXT_DIR` to a writable, persistent directory to keep the
one-time JIT build across restarts (important in containers).

Notes for this model:

- **Serve with `--enforce-eager`.** This artifact's body is FP8-CB, which uses an
  M-gated decode dispatch — a host-side branch that CUDA-graph capture cannot
  see. With capture on, the server pads decode batches above the prefill-M
  threshold and every graphed step takes the slower expand path. See
  [`docs/KERNELS.md` §CUDA-graph safety rules](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md#cuda-graph-safety-rules).
- **Vision tower** is NVFP4 weight-only (W4A16) and verified on image inputs; it
  needs a Blackwell-class or Marlin-supported path. Not tested on Ada.
  "Blackwell-class" is not one family — `sm_100` is capability family 10 and
  `sm_120`/`sm_121` is family 12, and vLLM's gates test the family, not the
  vendor name:
  [the GB10 capability-family rule](https://github.com/RobTand/gridbook/blob/master/docs/DELEGATED-NVFP4-MOE.md#why-gb10-is-family-120).
- **MTP draft block ships in BF16** but no speculative-decoding invocation is
  published for this artifact — the command above serves drafter-free.

**More:** [install + hardware matrix](https://github.com/RobTand/gridbook/blob/master/docs/INSTALL.md) ·
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md) ·
[format spec](https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md) ·
[kernels](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md) ·
[benchmarks + protocol](https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md) ·
[issues](https://github.com/RobTand/gridbook/issues)

**Other gridbook artifacts:**
[Laguna-S-2.1 6-bit](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) (89 GB, 117B MoE, 256k ctx) ·
[Hy3-295B-A21B 2.9-bit](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) (106 GB, 295B MoE + MTP)
````

---

## Notes — NOT part of the card

**What changed vs. the live card (fetched 2026-07-28).**

| Live card | Staged | Why |
|---|---|---|
| `pip install gridbook   # JIT-builds kernels for your GPU (capability >= 8.0 floor)` | `pip install gridbook`, with the capability story moved into the GPU row | The `>= 8.0` figure is real (`config.py: get_min_capability() -> 80`) but it is the *gate to load*, not the gate to be fast. Advertising it invites A100/H100/4090 users to install, pass the gate, silently get the Triton path, and measure numbers far below this card. |
| No `--enforce-eager` | `--enforce-eager` added | **The one derived flag in this staging set — it is not from this artifact's own validated invocation.** The live card's serve block is `--max-model-len 32768 --gpu-memory-utilization 0.90` and nothing else; `--enforce-eager` is added because `docs/KERNELS.md` rule 4 ("Keep `--enforce-eager` for the M-gated path") applies to this artifact's FP8-CB body, and because the live *Laguna* card — same FP8-CB body class — already carries it. See the verification item below; `_TEMPLATE.md` rule 3 permits a derived flag only when it is labelled like this. |
| No `--host/--port` | added | loopback bind is unreachable from another host. |
| "RTX 5090 (32 GB, sm_120) is the natural consumer target — 23 GB weights + ~7 GB KV" | "intended target but **not currently verified**", with issue #1 linked | Issue #1 reports OOM at 30.5 GiB on exactly that card. The root cause — dense CB weights double-resident, `linear.process_weights_after_loading` keeping both `cb_qweight` and `_cb_qw_padded` — is **unfixed at published `master` `b4694e3`**. A fix (16-byte pad so the padded buffer can be shared, `cb_qweight.data` re-pointed at a `narrow` view, original storage released) exists in the development tree but is **uncommitted and unpushed**, and no 32 GB box has been measured either way. Publishing the positive claim is not defensible under either state. **Revert this row to a positive claim only when the fix is public *and* a 32 GB verification exists** — the fix landing is not by itself sufficient. |
| No verification step | `curl` + a `[prismaquant-cb]` stderr check | Silent degradation to the Triton path is the single most likely thing to go wrong for a first-time user, and no *card* mentions it. (The repo's `docs/TROUBLESHOOTING.md` does, and uses the same grep-the-tag advice — but that page postdates published `master` `b4694e3`, and a reader who never leaves the Hub never sees it.) |
| No cross-links | links to the other two gridbook artifacts | Every gridbook card is currently a dead end. |

**⚠ Verify before applying: `--enforce-eager` on this artifact is derived, not
measured.** Every other flag in every staged block was copied from that
artifact's own validated invocation; this one was reasoned from
`docs/KERNELS.md` rule 4. The reasoning is sound and matches what the Laguna
card already ships for the same body format, but this artifact has a stock
NVFP4 W4A16 vision tower alongside its FP8-CB body, so its capture profile is
not identical to Laguna's. Either (a) run the 27B once with and once without the
flag and keep whichever wins, or (b) drop it and render the live card's flags
verbatim. Do not apply this card while the flag is neither measured nor removed.

**Front-matter changes:** added `library_name`, `pipeline_tag`,
`base_model_relation`, and 13 tags. Rationale and measured filter counts in
[`_METADATA.md`](_METADATA.md). `pipeline_tag: image-text-to-text` matches the
sibling repos of the same base model (`Qwen3.6-27B-PrismaQuant-5.5bit-vllm`) and
the artifact's own `config.json` (`Qwen3_5ForConditionalGeneration`, with
`vision_config` and `image_token_id`).

**Left untouched.** Everything from `## Measured quality` down — the KL table,
the format breakdown, the honest-limitations section, and the
`robert.tand@icloud.com` contact line. Two known issues in that region are
**out of scope for this staging** and belong to the docs-consistency pass:

- ~~This card reports KL(all) 0.0049 / KL(conf) 0.00295 and "−77% KL", while
  `docs/BENCHMARKS.md` reports 0.0134 / 0.01134 and "−58.3% ALL-KL"…~~
  **RECONCILED 2026-07-28.** They are different builds measured in different
  sessions — this card's numbers are the shipped 8-rung ladder (2026-07-22),
  BENCHMARKS' are the 4-rung A/B build (2026-07-18) — and absolute KL is not
  comparable across those sessions (the *same* NVFP4/FP8 baseline artifact reads
  0.02407 vs 0.01302 confident-KL). `BENCHMARKS.md` and `README.md` now carry
  the cross-reference; **this card needs one too** — add a line pointing at
  [`BENCHMARKS.md` § two sessions, two builds](https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md#two-sessions-two-builds-why-the-model-card-says-77-and-this-page-says-583)
  when the card is applied, so the two figures cannot be read as a
  contradiction.
- The card's ToolEvalBench line cites PrismaAURA-5.5 at 91; `docs/BENCHMARKS.md`
  and the repo README should be checked to agree.

**Sources.** Card text and sizes: live card + HF API (`22.98 GB`, 16 files,
created 2026-07-22, modified 2026-07-23, 1,464 all-time downloads, fetched
2026-07-28). Serve flags: live card. `--enforce-eager`:
`docs/KERNELS.md` §CUDA-graph safety rules. Warning string:
`gridbook/cuda_ext.py`, `get_ext()` — the symbol, not a line number (the
message text and its line number have each moved once already). Min capability:
`gridbook/config.py`, `get_min_capability()`.
Serve-by-Hub-id: public repo commit `7f2b502` (the in-tree development hash
`002471f` does not resolve in `RobTand/gridbook`). 5090 status:
[issue #1](https://github.com/RobTand/gridbook/issues/1) (open, 2026-07-25).
