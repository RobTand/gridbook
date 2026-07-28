# Standard gridbook card block — template

One consistent "Serve this model" section for every gridbook artifact. Render it
per model by substituting the variables below; the three rendered files in this
directory are the current renderings.

**Placement in the card:** immediately after the H1 and the one-paragraph
description, **above** the measurement/allocation sections, and it **replaces**
the existing `## Serving` / `## Serve` section (that is where the stale install
commands live today — three different ones across three cards, two of which
silently produce a degraded install).

---

## Variables

| Variable | Meaning | Source of truth |
|---|---|---|
| `{{REPO_ID}}` | Full HF repo id | HF API listing |
| `{{ARTIFACT_GB}}` | On-disk size of the artifact | the artifact's own card / HF API blob sizes |
| `{{MEM_REQ}}` | Free GPU / unified memory needed to serve at the command's settings | the artifact's card, measured |
| `{{GPU_LINE}}` | Which GPUs this was measured on, and the honest status of anything smaller | `docs/BENCHMARKS.md` + the artifact's card + open issues |
| `{{VLLM_LINE}}` | vLLM version requirement, or an explicit "not recorded" | the artifact's card / issue reports |
| `{{SERVE_FLAGS}}` | The validated `vllm serve` flags for this model | the artifact's card / its `serve.sh` |
| `{{EXTRA_NOTES}}` | Per-model serving caveats (spec decode, KV dtype, context) | the artifact's card |
| `{{INSTALL_LINE}}` | See "Install variants" below | — |

## Install variants

Pick **one**. Both are false today — see the APPLY GATE in
[`README.md`](README.md).

```
# PYPI  (default once `gridbook` is published and a clean-venv install builds the ext)
pip install gridbook

# GIT_TAG  (fallback: packaging fix landed, PyPI publish deferred — requires a real tag)
pip install "gridbook @ git+https://github.com/RobTand/gridbook@vX.Y.Z"
```

Do **not** render `pip install git+https://github.com/RobTand/gridbook` without a
tag: it is unpinned (the card would silently change meaning on every push) and,
until the packaging fix lands, it installs a plugin whose CUDA extension cannot
build.

---

## The block

````markdown
## Serve this model

| | |
|---|---|
| **Plugin** | [`gridbook`](https://github.com/RobTand/gridbook) — an out-of-tree vLLM quantization plugin. Stock vLLM, no fork, no core patches. |
| **GPU** | {{GPU_LINE}} |
| **Memory** | {{MEM_REQ}} |
| **vLLM** | {{VLLM_LINE}} |
| **Toolchain** | CUDA toolkit with `nvcc` on `PATH` **in the serving container** — the plugin JIT-builds its kernels on first model load (~30 s, cached). `nvcc` 13.0 is the tested toolchain. |
| **Parallelism** | Single GPU (`tp=1`). Tensor parallelism is not implemented in the plugin. |

```bash
{{INSTALL_LINE}}

vllm serve {{REPO_ID}} \
  --host 0.0.0.0 --port 8000 \
{{SERVE_FLAGS}}
```

Check it answers:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"{{REPO_ID}}","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
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

{{EXTRA_NOTES}}

**More:** [install + hardware matrix](https://github.com/RobTand/gridbook/blob/master/docs/INSTALL.md) ·
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md) ·
[format spec](https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md) ·
[kernels](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md) ·
[benchmarks + protocol](https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md) ·
[issues](https://github.com/RobTand/gridbook/issues)

**Other gridbook artifacts:**
[Qwen3.6-27B 5.5-bit](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) (23 GB, VL + MTP) ·
[Laguna-S-2.1 6-bit](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) (89 GB, 117B MoE, 256k ctx) ·
[Hy3-295B-A21B 2.9-bit](https://huggingface.co/rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm) (106 GB, 295B MoE + MTP)
````

---

## Rules the rendering follows

1. **`--host 0.0.0.0 --port 8000` in every command.** A loopback bind is
   unreachable from another machine on the LAN, which is how these get tested.
2. **Serve-by-Hub-id, not a placeholder path.** `vllm serve <repo-id>` works —
   the plugin resolves its `quant_config.json` / `cb_codebooks.pqcb` sidecars via
   `hf_hub_download` (public repo commit `7f2b502`, 2026-07-22). Rendering
   `/path/to/artifact` (as the repo README still does) makes the block
   non-copy-pasteable. *Cite public hashes only.* The in-tree development commit
   for the same change is `002471f`, which does not resolve in
   `RobTand/gridbook`; a reader who is handed it cannot look it up.
3. **Measured flags by default; a derived flag must be labelled.** Each
   rendering's `{{SERVE_FLAGS}}` comes from that artifact's own validated
   invocation. Flags are not carried between models — the three artifacts
   genuinely need different ones (`--enforce-eager` vs a `FULL_DECODE_ONLY`
   capture config; see `docs/KERNELS.md` §CUDA-graph safety rules). A flag that
   is *derived* from a documented kernel rule rather than copied from that
   artifact's own invocation is permitted, but it must be named as derived in
   the rendering's notes and listed as a pre-apply verification item. Exactly
   one flag is in that state today: `--enforce-eager` on the 27B.
4. **A verification step.** A user must be able to tell success from
   silent-degradation without reading the repo. Grep the `[prismaquant-cb]`
   stderr tag, never a verbatim sentence — the message wording is not a stable
   interface and has already changed once between the in-tree source and the
   published plugin.
5. **No unverified hardware claims.** If a GPU class has not been shown to work,
   the block says what is known and links the open issue.
6. **Sizes are measured from the HF API, in the unit they are labelled with.**
   `GET /api/models/<id>?blobs=true`, summed over `siblings`. A GiB value
   printed with a `GB` label is a bug, not a rounding choice — it is how the
   live Laguna card came to say "84 GB" for an 89.4 GB artifact.
