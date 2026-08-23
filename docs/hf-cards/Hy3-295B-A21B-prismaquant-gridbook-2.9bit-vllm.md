# `rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm` — staged card block

Staged, **not uploaded**. Read the APPLY GATE in [`README.md`](README.md) first.

---

## 1. Front matter (replaces the existing YAML block)

```yaml
license: apache-2.0
base_model: tencent/Hy3
base_model_relation: quantized
library_name: gridbook
pipeline_tag: text-generation
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
  - moe
  - single-gpu
  - mtp
  - speculative-decoding
  - conversational
```

## 2. Body block (insert after the intro bullets; replaces `## Serving`)

````markdown
## Serve this model

| | |
|---|---|
| **Plugin** | [`gridbook`](https://github.com/RobTand/gridbook) — an out-of-tree vLLM quantization plugin. Stock vLLM, no fork, no core patches. |
| **GPU** | NVIDIA Blackwell, compute capability `sm_120` / `sm_121`. Measured on GB10 / DGX Spark (`sm_121`). There is **no Triton fallback**: on a GPU that cannot run the native CUDA/CUTLASS kernel an artifact requires, serving **fails closed** with the missing operation instead of continuing on a slower one. Older GPU classes are unqualified — see the [compatibility table](https://github.com/RobTand/gridbook/blob/master/README.md#compatibility). |
| **Memory** | ~106 GB resident weights (including the CB-quantized MTP drafter) on a ~121 GB usable unified pool — one 128 GB DGX Spark. `--gpu-memory-utilization 0.90` is the **validated** setting; higher values OOM'd under long-prefill activation spikes. |
| **vLLM** | **≥ 0.23** with `HYV3ForCausalLM` (stock — no fork). Speed numbers on this card were measured on vLLM 0.23. |
| **Toolchain** | CUDA toolkit with `nvcc` on `PATH` **in the serving container** — the plugin JIT-builds its kernels on first model load (~30 s, cached). `nvcc` 13.0 is the tested toolchain. |
| **Parallelism** | Single GPU (`tp=1`) for MoE and delegated groups; dense CB Linears shard at load time above one rank (group-boundary violations are refused). No cross-node speedup is claimed. |

```bash
pip install gridbook

vllm serve rdtand/Hy3-295B-A21B-prismaquant-gridbook-2.9bit-vllm \
  --host 0.0.0.0 --port 8000 \
  --served-model-name hy3 \
  --max-model-len 12288 --max-num-seqs 2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --attention-backend TRITON_ATTN \
  --speculative-config '{"method":"hy_v3_mtp","num_speculative_tokens":1,"attention_backend":"TRITON_ATTN"}'
```

Check it answers:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
```

**How to verify native readiness.** Gridbook has no Triton dependency or serving
fallback. If a required CUDA/CUTLASS extension cannot be built, the relevant
load/forward fails closed after a line on stderr tagged `[prismaquant-cb]`:

```bash
vllm serve ... 2>&1 | grep '\[prismaquant-cb\]'
```

A `WARNING` or `ERROR` that explicitly says a **required** native operation is
unavailable means the model cannot be served through Gridbook on that
environment. A shape-specialized optimization may be unavailable only when the
same diagnostic names a separately qualified native CUDA/CUTLASS route. The tag
also carries harmless informational lines, so read the message after grepping
the stable tag. The two fatal forms read roughly:

```
[prismaquant-cb] WARNING: gridbook's CUDA decode-GEMV extension could not be built (<ExcType>: …); native Gridbook execution is unavailable and serving will fail closed. To enable the native path: …
[prismaquant-cb] ERROR: broken gridbook install — gridbook is installed without its CUDA sources: …
```

The first is an environment problem (usually no `nvcc` in the serving
container, or a torch/CUDA version mismatch); the second is a defect in the
install itself. Both are diagnosed in
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md#the-native-extension-did-not-load).
Set `PRISMAQUANT_CB_EXT_DIR` to a writable, persistent directory to keep the
one-time JIT build across restarts (important in containers).

Notes for this model:

- **MTP speculative decoding is on by default above** (`k=1`), and is what the
  card's 16.1 tok/s prose-decode figure was measured with. To serve without it,
  drop the last two flags (`--attention-backend` and `--speculative-config`).
- **The attention backend is outside Gridbook's operator lane.** Gridbook's
  no-Triton guarantee covers *its own* operators — decode GEMV, codebook
  expansion, activation QDQ, routing/combine, and the CUTLASS GEMMs. It does
  not cover vLLM's attention kernels, which are vLLM's choice and are
  unaffected by this plugin. The `TRITON_ATTN` pair above is therefore
  **what was measured at publication, not a recommendation**: with spec decode
  the drafter and target must agree on a backend, and on the vLLM of that date
  FlashInfer captured only single-token-decode graphs, so a disagreeing pair
  made vLLM silently disable *all* CUDA graphs. `k=1` was the optimum on that
  vLLM — the drafter runs uncaptured and its host cost scales with `k`.
- **Serving without a Triton attention backend.** Dropping the last two flags
  serves this artifact drafter-free on vLLM's default attention backend, with
  no `TRITON_ATTN` anywhere; the 16.1 tok/s prose-decode figure was measured
  *with* spec decode and does not carry over to that configuration. Keeping
  spec decode on a non-Triton backend — `--attention-backend FLASHINFER` plus
  the matching `"attention_backend"` inside `--speculative-config` — is the
  option to try, but it is **UNTESTED for this artifact**: re-qualifying it
  needs a served 295B run, and none has been made. Do not read it as qualified,
  and do not carry this card's spec-decode numbers across a backend change.
- **This artifact uses a `FULL_DECODE_ONLY` graph capture, not `--enforce-eager`**
  — unlike the other two gridbook artifacts. Its hot decode paths are the FP4-v2
  grouped MoE kernels, which have no host-side branching, so capture is safe and
  measured **+24% decode** here. See
  [`docs/KERNELS.md` §CUDA-graph safety rules](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md#cuda-graph-safety-rules).
- **Context 12288 is the validated setting**; larger fits at reduced batch. fp8
  KV cache was used for every validation run.

**More:** [install + hardware matrix](https://github.com/RobTand/gridbook/blob/master/docs/INSTALL.md) ·
[troubleshooting](https://github.com/RobTand/gridbook/blob/master/docs/TROUBLESHOOTING.md) ·
[format spec](https://github.com/RobTand/gridbook/blob/master/docs/SPEC.md) ·
[kernels](https://github.com/RobTand/gridbook/blob/master/docs/KERNELS.md) ·
[benchmarks + protocol](https://github.com/RobTand/gridbook/blob/master/docs/BENCHMARKS.md) ·
[issues](https://github.com/RobTand/gridbook/issues)

**Other gridbook artifacts:**
[Qwen3.6-27B 5.5-bit](https://huggingface.co/rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm) (23 GB, VL + MTP) ·
[Laguna-S-2.1 6-bit](https://huggingface.co/rdtand/Laguna-S-2.1-prismaquant-gridbook-6bit-vllm) (89 GB, 117B MoE, 256k ctx)
````

---

## Notes — NOT part of the card

**What changed vs. the live card (fetched 2026-07-28).**

| Live card | Staged | Why |
|---|---|---|
| `pip install -e ./serving/gridbook --no-deps` + `MODEL_DIR=. ./serving/serve.sh` | `pip install gridbook` + an explicit `vllm serve` line | The live command is the only one across the three cards that currently *works* — and only by accident: `-e` preserves the `../csrc` relative path that a normal install breaks. It requires the user to have cloned a 106 GB repo before they can install the plugin, and it pins them to a vendored plugin copy that is already stale. |
| serve config hidden inside `serving/serve.sh` | flags inlined | A card reader should not have to open a shell script to learn the serving contract. All flags are copied verbatim from `serve.sh`, including `--served-model-name hy3` (hence `"model":"hy3"` in the `curl`). |
| Requirements in prose | requirements table | Same shape as the other two cards. |
| No verification step | `curl` + a `[prismaquant-cb]` stderr check | Same reasoning as the other cards. |
| No cross-links | links to the other two gridbook artifacts | Every gridbook card is currently a dead end. |

**⚠ Serve-by-Hub-id is UNVERIFIED for this artifact.** The staged command uses
`vllm serve rdtand/Hy3-...` rather than a local directory. The plugin does
resolve its sidecars (`quant_config.json`, `cb_codebooks.pqcb`) through
`hf_hub_download` when the model is given as a repo id (`config.py:_resolve_model_file`,
landed in public repo commit `7f2b502`, 2026-07-22 — the in-tree development
hash `002471f` does not resolve in `RobTand/gridbook`) and the other two cards already
serve this way — but this artifact's published validation used
`MODEL_DIR=. ./serving/serve.sh` against a local clone, and this specific path
has not been exercised for a 5-shard 106 GB repo. **Verify a Hub-id serve before
applying this card**, or render the local-directory form instead.

**⚠ The vendored plugin copy should be removed from this repo.** It ships
`serving/gridbook/**` — a snapshot of the plugin that is already missing files
present on master (`gridbook/csrc/cb_persistent_tc.cu`,
`gridbook/csrc/smem_probe_tilem.cu`,
`gridbook/csrc/cutlass_fork/sm120_cb_persistent_mma.hpp`,
`gridbook/csrc/cutlass_fork/sm120_expert_row_broadcast.hpp` and
`gridbook/moe_routing.py`) — plus
`serving/gridbook/vllm_prismaquant.egg-info/{PKG-INFO,SOURCES.txt,entry_points.txt,…}`:
build artifacts published under the **pre-rename** package name. HF repos are not
rsynced from this tree, so this copy drifts permanently, and because it is the
only install path that currently works, users are pinned to a stale fork by
default. Once `pip install gridbook` works, delete `serving/gridbook/` (keep
`serving/serve.sh`, or fold it into the card) — that is a separate commit to the
HF repo, not a card edit.

**Left untouched.** The format-allocation table (the exact on-disk accounting),
the "No quality claims" section with its validation ledger, the naming note about
what "FP8"/"NVFP4" mean inside a CB format name, and the attribution line.

**Known cross-doc inconsistency, out of scope here.** This card's prefill figure
is "~109 tok/s" vs 42 for the GGUF arm. The repo README says "89 vs 42 tok/s" and
calls it "2.1×"; `docs/BENCHMARKS.md` says "~109 … ~2.6× faster" on line 170 and
"~2.1× prefill" on line 191; `docs/MOTIVATION.md` says "~2.1×". The TEB score is
88 in the `BENCHMARKS.md` table and 87 in the bullet 28 lines below it. Pick one
measured value with one stated protocol across all five documents — that belongs
to the docs-consistency pass, not to this staging.

**Sources.** Card text, size, allocation table, speed table: live card
(verbatim). Serve flags, memory guidance, spec-decode/TRITON_ATTN constraint,
and the 12288 context figure: this repo's own `serving/serve.sh` (fetched from
the HF repo 2026-07-28), verbatim. `FULL_DECODE_ONLY` +24%:
`docs/KERNELS.md` §CUDA-graph safety rules. Warning string:
`gridbook/cuda_ext.py`, `get_ext()` — the symbol, not a line number.
HF API: 60 files, 105.74 GB, created 2026-07-21,
modified 2026-07-23, 32 all-time downloads.
