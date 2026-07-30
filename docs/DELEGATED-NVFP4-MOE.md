# Delegated NVFP4 MoE groups on GB10 (`sm_121`)

A gridbook artifact is a **mixed container**. Config groups that carry a
`"scheme"` key are CB groups and are served by this plugin; groups without one
use the stock `compressed-tensors` vocabulary and are **delegated** to a real
`CompressedTensorsConfig` that the plugin constructs — see
[`SPEC.md` §6](SPEC.md) ("prefix is a plain NVFP4 / FP8 layer of a mixed
container → delegate to the runtime's stock `compressed-tensors` handling",
`docs/SPEC.md:397-400`) and [`PLUGIN.md`](PLUGIN.md) (`docs/PLUGIN.md:43-50`).

The consequence is already stated in two places and then dropped:

- `docs/PLUGIN.md:49-50` — "an artifact's hardware requirements are the union of
  gridbook's and those of its delegated groups."
- `docs/INSTALL.md:86-88` — "Those groups carry **their own** hardware
  requirements, independent of gridbook's kernels."

This page is the missing half of that sentence for the case that actually bites
on the reference target: **a delegated NVFP4 MoE group on GB10.** The failure it
documents is not a crash — it is a *silent* numerics change.

- [Scope: does this apply to me?](#scope-does-this-apply-to-me)
- [1. The capability-family formula](#1-the-capability-family-formula)
- [2. Which NVFP4 MoE backends run on GB10](#2-which-nvfp4-moe-backends-run-on-gb10)
- [3. The silent one: MARLIN drops the activation scale](#3-the-silent-one-marlin-drops-the-activation-scale)
- [4. Why relaxing the CUTEDSL / TRTLLM gate does not help](#4-why-relaxing-the-cutedsl--trtllm-gate-does-not-help)
- [5. What `auto` picks, and the exact `--kernel-config` spellings](#5-what-auto-picks-and-the-exact---kernel-config-spellings)
- [6. A gate you can run before you serve](#6-a-gate-you-can-run-before-you-serve)
- [7. Provenance: what is code-read and what is measured](#7-provenance-what-is-code-read-and-what-is-measured)

---

## Scope: does this apply to me?

**Only if a delegated (non-CB) group in your artifact is an NVFP4 *MoE* group.**

| Situation | This page applies? |
|---|---|
| CB expert band served by the gridbook plugin | **No.** CB groups do not go through vLLM's NVFP4 MoE oracle at all, and gridbook artifacts do not use `--kernel-config` for them. Use `--quantization gridbook`. |
| Delegated NVFP4 **MoE** group (`w13`/`w2` stacked experts) | **Yes** — all of it. |
| Delegated NVFP4 **dense Linear** group (e.g. the 27B's vision tower) | **Partly.** §1 is what "Blackwell-class" means precisely; §2–§6 are MoE-path only. |
| Delegated FP8 Linears (e.g. the 295B's 36 vanilla-FP8 Linears) | **No.** |

**None of the three currently published artifacts contains a delegated NVFP4 MoE
group** — the 295B's non-CB groups are FP8 Linears and the 27B's is an NVFP4
W4A16 vision tower (`docs/PLUGIN.md:47-49`). This page is therefore forward-looking:
it is for anyone producing a mixed container that leaves an MoE band to stock
`compressed-tensors`, and for anyone reading `docs/INSTALL.md:86-88` and wondering
what "their own hardware requirements" costs on this box.

---

## 1. The capability-family formula

GB10 is `sm_121`. vLLM's family predicate is integer division by ten:

```python
return (current_capability.to_int() // 10) == (capability // 10)
```

— vLLM `vllm/platforms/interface.py:363-375` (`is_device_capability_family`).

`sm_121` → `to_int()` = `121` → `121 // 10` = **12**. So GB10 matches
`is_device_capability_family(120)` and **fails** `is_device_capability_family(100)`.

This is the single fact that decides everything below, and it is easy to get
wrong in the other direction: "Blackwell" is not one family here. `sm_100`
(GB100/GB200) is family 10; `sm_120`/`sm_121` (RTX 50-series, GB10) is family 12.
A backend that gates on `family(100)` and calls itself "Blackwell-only" is
**not** available on a DGX Spark. The gate is correct; only its prose is
misleading.

### Two predicates, both spelled "100", opposite answers on GB10

vLLM has a *second* capability predicate and the two are easy to confuse when
skim-reading a gate, because both take the literal `100`:

| Call | Meaning | On `sm_121` |
|---|---|---|
| `is_device_capability_family(100)` | **exact family**: `121 // 10 == 100 // 10` → `12 == 10` | **False** |
| `has_device_capability(100)` | **floor**: `to_int() >= 100` → `121 >= 100` — `vllm/platforms/interface.py:315-336` | **True** |

`FlashInferExperts` uses *both*: `family(120)` in its device gate and
`has_device_capability(100)` — the floor — to admit the NVFP4 scheme pair
(`experts/flashinfer_cutlass_moe.py:177-184`). That is why it is available on
GB10 while CUTEDSL and TRTLLM, which use the family form, are not. Read the
predicate name, not the number.

---

## 2. Which NVFP4 MoE backends run on GB10

Every gate below is `_supports_current_device()` on the expert class, consumed by
`is_supported_config()` (`modular_kernel.py:536-547`).
Line numbers throughout this page are **vLLM v0.23.0**
(see [§7](#7-provenance-what-is-code-read-and-what-is-measured)). A path that
starts with `vllm/`, `csrc/` or `CMakeLists.txt` is relative to the vLLM source
root; every other path (`experts/…`, `oracle/…`, `modular_kernel.py`,
`config.py`) is relative to **`vllm/model_executor/layers/fused_moe/`**.

| `moe_backend` | Device gate | On GB10 | Accepts a W4A16 checkpoint? |
|---|---|---|---|
| `flashinfer_cutlass` | `is_device_capability(90) or family(100) or family(120)`, plus `has_flashinfer_cutlass_fused_moe()` — `experts/flashinfer_cutlass_moe.py:131-143` | ✅ (admits the NVFP4 pair via the `has_device_capability(100)` **floor**, `experts/flashinfer_cutlass_moe.py:177-184`) | ❌ needs `(kNvfp4Static, kNvfp4Dynamic)` — `experts/flashinfer_cutlass_moe.py:149-185` |
| `cutlass` (logs as `VLLM_CUTLASS`) | `family(100) or family(110) or family(120)` — `experts/cutlass_moe.py:692-699` | ✅ | ❌ `_supports_quant_scheme` is `== (kNvfp4Static, kNvfp4Dynamic)` — `experts/cutlass_moe.py:706-710` |
| `flashinfer_cutedsl` | `family(100)` only — `experts/flashinfer_cutedsl_moe.py:68-75` | ❌ | ❌ |
| `flashinfer_trtllm` | `family(100)` only — `experts/trtllm_nvfp4_moe.py:112-120` | ❌ | ❌ |
| *(`FLASHINFER_CUTEDSL_BATCHED`, reached only via the batched activation format)* | `family(100)` only — `experts/flashinfer_cutedsl_batched_moe.py:61-68` | ❌ | ❌ |
| `flashinfer_b12x` | `family(120)`, plus `has_flashinfer_b12x_moe()` — `experts/flashinfer_b12x_moe.py:133-140` | ✅ device-wise, but **excluded from `auto`** — `oracle/nvfp4.py:170-182` | ✅ accepts `(kNvfp4Static, None)` — `experts/flashinfer_b12x_moe.py:147-157` |
| `marlin` | no family gate: `has_device_capability((7, 5))` — `experts/marlin_moe.py:590-593` | ✅ | ✅ — **and this is the trap, see [§3](#3-the-silent-one-marlin-drops-the-activation-scale)** |
| `emulation` | no capability gate — inherits `TritonExperts`: `is_cuda_alike() or is_xpu()` (`experts/triton_moe.py:71-73`) | ✅ correctness only | ❌ raises if `a13_scale`/`a2_scale` are `None` — `oracle/nvfp4.py:430-434` |
| `humming` | accepted by the config validator (`vllm/config/kernel.py:122-137`) but **absent from the NVFP4 oracle's `map_nvfp4_backend`** (`oracle/nvfp4.py:141-157`). The Humming expert classes are wired into the **MXFP4** oracle only (`oracle/mxfp4.py:160-165`) | ❌ raises `ValueError` at startup on v0.23.0 | n/a |

**The two fused NVFP4 MoE backends that actually work on GB10 are
`FLASHINFER_CUTLASS` and `VLLM_CUTLASS`.** Both require a W4A4 checkpoint.

### `VLLM_CUTLASS` has a build-time precondition and a dtype restriction

Passing the device gate is necessary, not sufficient. The grouped GEMM behind
`CutlassExpertsFp4` dispatches to an SM120 code path that only exists if the
vLLM wheel was compiled for a 12.x arch:

- `CMakeLists.txt:927-932` picks `FP4_ARCHS` (`12.0f` under CUDA ≥ 13.0, else
  `12.0a;12.1a`), and `CMakeLists.txt:948-950` then defines `ENABLE_NVFP4_SM120=1`
  / `-DENABLE_CUTLASS_MOE_SM120=1`.
- `csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu:612-618`
  routes `version_num >= 120 && version_num < 130` to
  `run_fp4_blockwise_scaled_group_mm_sm120` (`ArchTag = cutlass::arch::Sm120`,
  `OpClassBlockScaledTensorOp` — same file `:439-440`).
- **bf16 output only.** A non-bf16 output dtype on a 12.x device hits
  `STD_TORCH_CHECK_NOT_IMPLEMENTED(false, "SM120 NVFP4 MOE only supports bfloat16
  output, ...")` — same file `:699-707`. Serve `--dtype bfloat16`.

A wheel built without those archs does not fail the *Python* gate; it fails
inside the op with "No compiled `cutlass_fp4_group_mm` kernel for CUDA device
capability: 121" (same file `:628-631`).

---

## 3. The silent one: MARLIN drops the activation scale

This is the reason this page exists. **`marlin` is the only backend in the table
that will load a W4A4 NVFP4 MoE group on GB10 and serve it with different
numerics than the checkpoint declares — with no warning line.**

Three steps in vLLM's own source, in order:

**(a) MARLIN's scheme check ignores the activation half entirely.** Where the
CUTLASS backends compare the *pair*, MARLIN compares only the weight key:

```python
    SUPPORTED_W = [ ... kNvfp4Static, ... ]
    return weight_key in SUPPORTED_W
```

— `experts/marlin_moe.py:600-619`. `activation_key` is a parameter and is never
read. So a `(kNvfp4Static, kNvfp4Dynamic)` W4A4 group is "supported".

**(b) The activation scales are then set to `None`.**

```python
    elif nvfp4_backend == NvFp4MoeBackend.MARLIN:
        a13_scale = None
        a2_scale = None
```

— `oracle/nvfp4.py:405-407`, in `convert_to_nvfp4_moe_kernel_format`. The
`prepare_nvfp4_moe_layer_for_marlin` call immediately below it takes no
activation-scale argument at all.

**(c) The quant config built for the kernel is the weight-only one.**

```python
    if backend == NvFp4MoeBackend.MARLIN:
        return nvfp4_w4a16_moe_quant_config(...)
```

— `oracle/nvfp4.py:478-484`. The function name is the proof, and its signature
(`config.py:855-860`) has **no** `a1_gscale`/`a2_gscale` parameter —
there is nowhere for a calibrated activation scale to go.

Contrast the fused path, same file, same function:

```python
    return nvfp4_moe_quant_config(
        g1_alphas=w13_scale_2,
        g2_alphas=w2_scale_2,
        a1_gscale=(1.0 / a13_scale),
        a2_gscale=(1.0 / a2_scale),
        ...
```

— `oracle/nvfp4.py:500-505`. The reciprocal is the checkpoint's calibrated
per-expert input global scale being handed to the kernel.

**Net effect.** Weights load, the server starts, every request succeeds, and
every token is computed with **W4A16** numerics from a **W4A4** checkpoint. The
only externally visible symptom is a quality change. There is no `WARNING`,
because from vLLM's point of view nothing went wrong — MARLIN is a weight-only
kernel and it did exactly what it does.

> **If you benchmark a W4A4 delegated group under `marlin`, you are not
> benchmarking W4A4.** Do not quote such a run as an activation-quantization
> comparison.

The mirror-image case is loud and therefore harmless: a genuinely weight-only
(W4A16) group whose `QuantKey` ends `xNone` is *rejected* by every fused backend
at startup, because their `_supports_quant_scheme` requires the pair
(`experts/cutlass_moe.py:706-710`, `experts/flashinfer_cutedsl_moe.py:82-89`,
`experts/trtllm_nvfp4_moe.py:128-136`). You get a `ValueError` naming the
scheme, not a silent degrade.

---

## 4. Why relaxing the CUTEDSL / TRTLLM gate does not help

The obvious "fix" — patch `family(100)` to also accept `family(120)` in
`experts/flashinfer_cutedsl_moe.py:68-75` and `experts/trtllm_nvfp4_moe.py:112-120`
— does not produce a working backend on GB10. The device gate is not the thing
protecting you.

Each of those gates is an `and` of the family check with an availability probe:
`has_flashinfer_cutedsl_moe_nvfp4()` (`experts/flashinfer_cutedsl_moe.py:74`)
and `has_flashinfer_trtllm_fused_moe()` (`experts/trtllm_nvfp4_moe.py:119`). On
GB10 **both probes return `True`** — the FlashInfer package is importable and
reports the feature present. Removing the family check therefore lets selection
succeed and moves the failure downstream, into code that has no fallback:

- **CUTEDSL** dies in FlashInfer's JIT at `get_nvcc_flags_list(
  supported_major_versions=[10])` — the compile flags for this kernel are
  enumerated for major version 10 only.
- **TRTLLM** dies inside a prebuilt cubin whose name ends `_sm100f`.

`FLASHINFER_CUDA_ARCH_LIST` only relocates the failure again; it does not
conjure an sm_12x cubin. **Leave those gates alone.** The two CUTLASS backends
in §2 are the supported answer.

*(FlashInfer is a separate package, not part of the vLLM tree, so those last two
symbols are quoted from an observed GB10 run rather than from a vLLM file:line —
see [§7](#7-provenance-what-is-code-read-and-what-is-measured).)*

---

## 5. What `auto` picks, and the exact `--kernel-config` spellings

### The `auto` walk order

`moe_backend="auto"` iterates this list and takes the first backend whose
`is_supported_config()` returns `True` (`oracle/nvfp4.py:174-182`, loop at
`oracle/nvfp4.py:315-329`):

```
FLASHINFER_TRTLLM → FLASHINFER_CUTEDSL → FLASHINFER_CUTEDSL_BATCHED
                  → FLASHINFER_CUTLASS → VLLM_CUTLASS → MARLIN → EMULATION
```

`FLASHINFER_B12X` is deliberately **not** in that list — "intentionally excluded
from auto-selection until the upstream CUTLASS SM121 MMA op guard is resolved"
(`oracle/nvfp4.py:170-173`). It is opt-in only.

Read that order against §2 and §3. On GB10 the first three are eliminated by the
family gate, so a **W4A4** group lands on `FLASHINFER_CUTLASS` — the right
answer. But a group that fails the fused backends' scheme check for any reason
keeps walking, and the next stop that accepts it is **`MARLIN`**. That is the
mechanism by which the silent W4A16 downgrade in §3 gets selected *by default*,
without anyone typing `marlin`.

**Pin the backend explicitly for any run you intend to quote.** `auto`'s answer
is a property of your image and its device probes, not of your intent.

### Canonical spellings

`KernelConfig._normalize_moe_backend` lowercases and maps `-` → `_`
(`vllm/config/kernel.py:211-216`), so case and dashes do not matter. These are
the canonical forms:

```bash
# Fused W4A4 on GB10 — the recommended path for a delegated NVFP4 MoE group
--kernel-config '{"moe_backend":"flashinfer_cutlass"}'

# vLLM's own CUTLASS; also fused, also W4A4-only. Needs a wheel built with
# ENABLE_NVFP4_SM120 and --dtype bfloat16 (see §2).
--kernel-config '{"moe_backend":"cutlass"}'

# Weight-only. Correct for a genuine W4A16 group; SILENTLY W4A16 for a W4A4 one.
--kernel-config '{"moe_backend":"marlin"}'

# Bisection only — proves a scheme is genuinely unsupported rather than
# silently degraded, because it raises instead of dropping scales.
--kernel-config '{"moe_backend":"emulation"}'
```

### Confirm it at startup

vLLM logs the choice once, from `_make_log_backend` (`oracle/nvfp4.py:200-205`):

```
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: [...]
```

For a W4A4 delegated group on GB10 that name must be `FLASHINFER_CUTLASS` or
`VLLM_CUTLASS`. If it says `MARLIN`, you are on §3.

**This does not apply to gridbook's own CB groups.** The CB expert band is
claimed by the plugin, not by a vLLM NVFP4 MoE backend, and `--kernel-config` has
no effect on it. See [`PLUGIN.md`](PLUGIN.md).

---

## 6. A gate you can run before you serve

A note nobody reads is not a gate. The constructive form of §3 is a check that
**fails the artifact** rather than describing the hazard:

> Given (a) the artifact's declared activation scheme — does it ship
> `input_scale` / `input_global_scale` tensors for its MoE band? — and (b) the
> `moe_backend` you will actually serve with, refuse the combination when the
> kernel would not honour the scheme the checkpoint declares.
>
> - `W4A4 checkpoint` + a **weight-only** backend — on v0.23.0 that is
>   `marlin`, and in general any backend whose `_supports_quant_scheme`
>   ignores `activation_key` → **FAIL**: activation scales dropped, server
>   starts, numerics silently differ from `config.json`.
> - `W4A16 checkpoint` + a **fused** backend (`flashinfer_cutlass`, `cutlass`,
>   `flashinfer_cutedsl`, `flashinfer_trtllm`, `emulation`) → **FAIL**: startup
>   raise (loud, but still worth catching before a queue slot is spent).
> - anything else → pass.
>
> Keep that first list keyed to the *predicate*, not to a fixed backend name:
> a future backend that ignores `activation_key` inherits the same hazard, and
> `humming` is only excluded here because v0.23.0's NVFP4 oracle has no mapping
> for it at all ([§2](#2-which-nvfp4-moe-backends-run-on-gb10)).
>
> The backend must be **supplied**, not inferred: with `auto` the tool cannot
> know what the device probes will decide, so the honest result is
> "not checked", not "pass".

Closing this gap does not recover the numerics — under MARLIN they are
unrecoverable by construction, the kernel is weight-only. It makes the mismatch
**non-silent**, which is the entire failure mode.

The natural home for this in gridbook is artifact-production time (see
[`SPEC.md`](SPEC.md) on what a producer guarantees per group): if you delegate an
MoE band to stock `compressed-tensors`, record in the artifact's card which
`moe_backend` it was validated under, so a reader is not left to discover the
downgrade from a quality delta.

---

## 7. Provenance: what is code-read and what is measured

**vLLM line numbers.** Every `vllm/...`, `csrc/...` and `CMakeLists.txt`
reference on this page was read from a **vLLM v0.23.0** source tree, and each
cited file was checked to be byte-identical to that release (no local patch on
any file quoted here). vLLM moves fast and this surface is
not a stability contract — re-grep for the symbol name (`is_device_capability_family`,
`_supports_current_device`, `nvfp4_w4a16_moe_quant_config`) rather than trusting
a line number against a different build. `INSTALL.md`'s reference stack is
`0.23.1rc1.dev764+g54b16d8a9`, which is a *different* build from the v0.23.0
tree these numbers came from; the symbols were identical in both as of writing,
the numbers were checked only against v0.23.0.

**Code-read, not run:** the whole of §1–§3 and §5 is a source reading. It
requires no GPU and you can reproduce it with `grep` on your own vLLM.

**Observed on GB10 hardware:** that `FLASHINFER_CUTLASS` and `VLLM_CUTLASS` both
serve an NVFP4 W4A4 MoE checkpoint on `sm_121`; and the two downstream failure
sites in §4 (`get_nvcc_flags_list(supported_major_versions=[10])`, the `_sm100f`
cubin) after locally relaxing the family gate. Those observations were made
serving a **standalone NVFP4 checkpoint**, not a gridbook mixed container — the
delegated-group path composes the same vLLM code, but no published gridbook
artifact exercises it yet (see [Scope](#scope-does-this-apply-to-me)).

**Not measured, deliberately absent:** any throughput or quality number for a
delegated NVFP4 MoE group inside a gridbook artifact, and any claim about
`flashinfer_b12x`, which is listed in §2 from its gate only and has not been run
here.

---

**See also:** [`INSTALL.md` §Hardware matrix](INSTALL.md#hardware-matrix) ·
[`PLUGIN.md`](PLUGIN.md) · [`SPEC.md`](SPEC.md) ·
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
