# Contributing to gridbook

Thanks for looking. This project is small and maintained by one person on one
machine, which shapes what is most useful to contribute.

---

## The single most useful contribution: a hardware report

Every published performance number comes from **one** GB10 / DGX Spark
(`sm_121`, arm64). The
[compatibility tables](docs/INSTALL.md#hardware-matrix) mark everything else as
*inferred from the code and untested* — and that honesty is exactly what keeps
people from trying it.

If you run gridbook on anything else — RTX 5090, 4090, L40S, H100, x86, a
different vLLM version — **open an issue with what happened**, working or not.
That converts an inferred cell into a measured one, which is worth more here than
most code changes.

## Reporting a bug

Please include:

1. **GPU** and compute capability (`torch.cuda.get_device_capability()`).
2. **Versions**: `vllm.__version__`, `torch.__version__`, `gridbook.__version__`,
   `nvcc --version`, Python, CPU architecture.
3. **Install route**: `pip install git+...`, editable checkout, container, etc.
4. **The output of the [install check](docs/INSTALL.md#verify-the-install)** —
   this immediately separates "the CUDA path is broken" from "the CUDA path is
   fine and something else is wrong", and saves a whole round trip.
5. **Every `[prismaquant-cb]` line from your log** (the runtime prefix is
   `prismaquant`, not `gridbook` — grep for that).
6. The **model repo id** and the exact `vllm serve` command.
7. Any `PRISMAQUANT_*` variables you have set (`env | grep PRISMAQUANT`).

[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) covers the known failure modes; if
yours is there and the fix does not work, say so — that is a documentation bug
worth knowing about.

## Scope

| In scope here | Belongs elsewhere |
|---|---|
| The vLLM serving plugin (`gridbook/`) and its CUDA/Triton kernels (`gridbook/csrc/`) | The quantization pipeline that *produces* artifacts — that is [PrismaQuant](https://github.com/RobTand/prismaquant) |
| The normative format spec (`docs/SPEC.md`) and independent implementations of it | Model-specific quality tuning / calibration |
| New model-architecture support (usually a guarded one-liner in `plugin.py`) | Forks of vLLM core — a hard non-goal; gridbook must work on stock vLLM |
| Documentation, install/packaging, CI | |

Support for a new MoE architecture is often just a registration in
`_install_toplevel_cb_expert_loaders()` — if a model loads but generates
garbage, that is usually what is missing, and an issue naming the
`architectures` entry from its `config.json` is enough to get started.

## Running the tests

The suite selects itself at runtime — every CUDA, vLLM or artifact-backed test is
guarded by `pytest.skip` / `importorskip` / a CUDA check — so on a machine with
no GPU it is all-pass-or-skip rather than a wall of errors.

```bash
pip install -e . --no-deps
pip install pytest
.github/scripts/run_cpu_tests.sh tests      # the GPU-free tier, as CI runs it
```

That script runs **one pytest process per file** on purpose:
`tests/test_target_namespace_compat.py` injects stub `vllm.*` modules into
`sys.modules`, which leak into later files in the same session and turn their
skip guards into spurious failures.

| Tier | What it needs | Runs in CI? |
|---|---|---|
| **Anywhere** — no GPU, no vLLM | torch only | ✅ on every push and PR |
| **GPU** — kernel parity, decode/prefill numerics | a CUDA GPU and `nvcc` | ❌ no GPU runners; run locally |
| **GPU + artifacts** — end-to-end paths | an exported CB artifact on disk; a few still point at paths that only exist on the author's box | ❌ |

CI additionally gates the packaging surface: that the wheel and sdist really
contain `gridbook/csrc/*.cu`, that a **non-editable** install resolves them from
`site-packages`, that `import gridbook` needs no torch/triton/vLLM, and that the
`vllm.general_plugins` entry point is discoverable. It cannot compile the CUDA
extension — free runners have no `nvcc` — so that stays a manual pre-tag gate.

**Wanted, in rough order of value:**

1. A `conftest.py` that restores `sys.modules` after the stub-injecting test, so
   the suite runs in one process.
2. Replacing the remaining hardcoded `/home/rob/...` artifact paths with an
   environment variable plus `pytest.skip` when unset —
   `tests/test_cuda_gemv.py` already does this correctly; copy that pattern.
3. Anything that lets a GPU test run against a small, publicly downloadable
   artifact instead of a private one.

## Ground rules for changes

1. **Measured claims only.** Any performance or quality number added to code
   comments, docs or a commit message must come from a measurement, with the
   hardware and protocol named. If it is inferred from code rather than measured,
   label it as inferred. No estimates presented as results — this is the
   project's central discipline, and the reason the docs mark untested cells
   instead of filling them in.
2. **Numerics changes need a parity argument.** The kernels hold a bit-exactness
   contract against the Triton reference path (≤ 1 bf16 output ULP, plus a norm
   backstop). A change that reassociates a reduction is allowed, but it must say
   so and be gated on a served quality check — silent numerics drift is how a
   quantization project loses its ability to A/B anything.
3. **Fail soft, but never silently wrong.** Missing toolchain → warn and fall
   back. Missing *data* → raise. The distinction matters: a slow server is
   recoverable, a wrong one is not.
4. **No new vLLM-core patches.** The plugin's value is that it runs on stock
   vLLM. Wrapping a specific model class's own `load_weights` is the existing,
   documented exception; extending it to `vllm/` internals is not.
5. **Keep `import gridbook` free of vLLM and GPU requirements.** The codec and
   format tests must run on a laptop.

## Pull requests

- Small and focused beats large and sweeping; this repo is mirrored from an
  internal monorepo, so a PR is applied on the source side and synced back. Your
  change may therefore land as a differently-shaped commit — authorship is
  preserved in the commit message, and the issue/PR is linked.
- Please describe how you tested it, on what hardware.
- By contributing you agree your contribution is licensed under Apache-2.0, the
  same license as the project. There is no CLA.

## Security

For anything you would rather not file publicly, email
<robert.tand@icloud.com>.
