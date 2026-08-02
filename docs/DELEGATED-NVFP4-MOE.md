# Delegated NVFP4 MoE groups on GB10 (`sm_121`)

A gridbook artifact can mix CB groups, which this plugin serves, with ordinary
`compressed-tensors` groups, which gridbook delegates back to vLLM. This page
covers one narrow case: a delegated NVFP4 **MoE** group on GB10. It does not
apply to gridbook's own CB expert stacks or to delegated dense Linears.

No currently published gridbook artifact has such a group. The page is an
operator check for future mixed artifacts, not a claim about the three current
model releases.

## Version scope

The NVFP4 MoE oracle is an internal vLLM interface and changes quickly. The
claims below were source-checked against both:

| Build | Role |
|---|---|
| `0.23.1rc1.dev764+g54b16d8a9.d20260703` | gridbook's measured reference image |
| `0.24.0` | current review image |

The capability predicate, compressed-tensors scheme dispatch, automatic backend
order, Marlin conversion, and Marlin warning described here are the same in
both builds. Backend activation support differs in places, so treat the table
as scoped to these builds and re-check it when changing vLLM or FlashInfer.

## Identify the declared scheme

Read the resolved `compressed-tensors` config group that targets all three MoE
projections (gate, up, and down):

- NVFP4 `weights` plus NVFP4 `input_activations` declares **W4A4**.
- NVFP4 `weights` plus `input_activations: null` (or no activation entry)
  declares **W4A16**.

That is also how vLLM decides: `CompressedTensorsMoEMethod.get_moe_method` reads
the resolved group's `weights` and `input_activations`, then constructs the
NVFP4 method with `use_a16=(input_quant is None)`.

Do not infer the scheme from whether a safetensors key such as
`input_global_scale` happens to exist. Runtime parameter creation and conversion
can allocate or synthesize scale tensors. The config declaration, after target
and ignore resolution, is the contract.

Groups with a gridbook `"scheme"` key are CB groups and never enter this vLLM
oracle. See [the format specification](SPEC.md#6-runtime-registration-and-dispatch).

## Why GB10 is family 120

GB10 reports CUDA capability `sm_121`. In both audited builds,
`is_device_capability_family` compares:

```python
(current_capability.to_int() // 10) == (capability // 10)
```

Therefore `121 // 10 == 120 // 10` and GB10 is in family 120, not family
100. This differs from `has_device_capability(100)`, which is a minimum check
and is true because `121 >= 100`.

This distinction explains why a backend gated by `family(100)` does not run on
GB10 even though a separate `has_device_capability(100)` check passes.

## Backend behavior on GB10

The table describes source eligibility, not a promise that a particular wheel,
shape, activation, or parallel configuration will pass every runtime check.

| Backend | GB10 device gate | Declared NVFP4 schemes | Important behavior |
|---|---|---|---|
| `flashinfer_trtllm` | family 100 | W4A4 | Rejected by the device-family gate. |
| `flashinfer_cutedsl` | family 100 | W4A4 | Rejected by the device-family gate. The batched variant has the same gate. |
| `flashinfer_cutlass` | includes family 120 and requires its FlashInfer feature | W4A4 | First native W4A4 candidate in `auto` on GB10. |
| `cutlass` (logged as `VLLM_CUTLASS`) | includes family 120 | W4A4 | Second native W4A4 candidate. The wheel must contain the SM12x op; the audited reference source restricts this path to BF16 output on SM12x. |
| `marlin` | CUDA capability 7.5 or newer | W4A4 and W4A16 pass its selector | Always builds the NVFP4 **W4A16** quant config. For W4A4 input, activation scales are discarded. |
| `emulation` | CUDA/Triton support | W4A4 | Slow correctness path; preserves activation QDQ. |
| `flashinfer_b12x` | family 120 and its FlashInfer feature | W4A4 and W4A16 | Explicit opt-in only; excluded from `auto`. It dynamically quantizes BF16 activations to FP4 in-kernel, including for a W4A16 declaration. Do not describe that case as W4A16 compute. |

`flashinfer_b12x` is especially version-sensitive. In the audited builds it is
excluded from automatic selection pending an SM121 MMA guard, and its accepted
activations differ between the reference nightly and 0.24. It has not been
validated by a published gridbook model. Pinning it is an explicit experiment,
not a general W4A16 recommendation.

Although `humming` is accepted by the general kernel-config validator in these
builds, the NVFP4 oracle does not map it. An explicit NVFP4 request therefore
fails at startup rather than selecting a Humming implementation.

## The Marlin mismatch is underdiagnosed, not log-silent

Marlin's `_supports_quant_scheme` checks the NVFP4 weight key but does not use
the activation key. A W4A4 pair consequently passes selection. The NVFP4
conversion then:

1. sets both activation-scale arguments to `None`;
2. prepares the weights for Marlin; and
3. builds `nvfp4_w4a16_moe_quant_config`, whose interface has no activation
   scale.

The result is weight-only W4A16 execution even when the resolved config group
declared W4A4.

This is visible in logs, but not diagnosed precisely. Both audited builds emit:

- an INFO line selecting the `MARLIN` NVFP4 MoE backend; and
- a WARNING saying weight-only FP4 compression will use Marlin.

The warning does **not** say that the selected group declared W4A4, name the
activation scales being dropped, or distinguish intentional W4A16 from a W4A4
fallback. That missing connection is the operational hazard. A Marlin run of a
W4A4 group must not be reported as a W4A4 activation-quantization result.

## What `auto` can select

For NVFP4 MoE, both audited builds try:

```text
FLASHINFER_TRTLLM
→ FLASHINFER_CUTEDSL
→ FLASHINFER_CUTEDSL_BATCHED
→ FLASHINFER_CUTLASS
→ VLLM_CUTLASS
→ MARLIN
→ EMULATION
```

`FLASHINFER_B12X` is not in that list.

On GB10, the family check eliminates the first three. For a W4A4 declaration,
`auto` next tries the two native W4A4 candidates, but either may still fail an
extension, activation, shape, dtype, or deployment-config check. Marlin is the
next candidate and accepts the NVFP4 weight key. Never assume that `auto` chose
a native W4A4 path.

Pin the backend for a run whose numerics you intend to compare:

```bash
# Delegated W4A4; startup fails if the requested native path is unavailable.
--kernel-config '{"moe_backend":"flashinfer_cutlass"}'

# vLLM's native CUTLASS alternative. Use BF16 on the audited SM12x build.
--dtype bfloat16 \
--kernel-config '{"moe_backend":"cutlass"}'

# Intentional weight-only W4A16.
--kernel-config '{"moe_backend":"marlin"}'

# Slow W4A4 correctness check.
--kernel-config '{"moe_backend":"emulation"}'
```

At startup, inspect the `Using '<BACKEND>' NvFp4 MoE backend` INFO line. If a
W4A4 group says `MARLIN`, the runtime is using weight-only numerics regardless
of the generic warning's wording.

## Preflight policy

**Shipped.** Gridbook enforces this table at model load in
`gridbook/delegated_preflight.py`, called from the single delegation choke point
in `config.py` (`_delegate`). A delegated group is refused when its resolved
backend is Triton-backed, when the backend is documented to discard declared
activation scales, or when an NVFP4 W4A4 declaration resolves to a backend that
is not in the audited table — UNKNOWN fails closed rather than becoming a false
pass. The error names the resolved backend class, the group, and the declared
contract. There is no environment-variable bypass. Extending the tables is the
supported way to admit a newly audited backend.

A useful preflight has three outcomes:

| Declared scheme | Requested backend | Result |
|---|---|---|
| W4A4 | `flashinfer_cutlass`, `cutlass`, or `emulation` | **PASS** for scheme compatibility; startup must still prove device/config support. |
| W4A4 | `marlin` | **FAIL**: known conversion to W4A16. |
| W4A4 | `flashinfer_b12x` | **UNKNOWN** for project validation: the source accepts it, but it is version-sensitive, opt-in, and untested by a published gridbook artifact. |
| W4A16 | `marlin` | **PASS** for scheme compatibility. |
| W4A16 | the audited W4A4-only backends | **FAIL**: unsupported declaration. |
| W4A16 | `flashinfer_b12x` | **FAIL by default** if preserving W4A16 activation semantics is required; this opt-in backend dynamically quantizes activations. |
| Either | `auto` | **UNKNOWN** until the serving environment runs every device and config probe. |
| Either | unknown backend or unaudited vLLM version | **UNKNOWN**; update the versioned allowlist before serving. |

Here, PASS means only that the declared activation semantics and requested
backend agree. It does not prove the kernel is compiled or supports the model's
shape and activation.

Implement this as an explicit, versioned backend table. Do not try to future-
proof it by inspecting whether `_supports_quant_scheme` reads
`activation_key`: that method is only a selection predicate, not a runtime
semantics contract. A future backend can accept a pair and still transform its
scales, as the opt-in B12x path demonstrates.

The preflight must resolve the target's declared config group, not scan tensor
names. It must also return UNKNOWN for `auto` rather than turning missing
environment information into a false pass.

## Provenance and limits

The review checked these vLLM source contracts in both versioned images:

- `vllm/platforms/interface.py`:
  `has_device_capability` and `is_device_capability_family`;
- `compressed_tensors_moe.py`:
  `CompressedTensorsMoEMethod.get_moe_method`;
- `oracle/nvfp4.py`:
  `select_nvfp4_moe_backend`,
  `convert_to_nvfp4_moe_kernel_format`, and
  `make_nvfp4_moe_quant_config`;
- the backend classes' `_supports_current_device` and
  `_supports_quant_scheme` methods; and
- `marlin_utils_fp4.py`:
  `prepare_nvfp4_moe_layer_for_marlin` and its warning.

No delegated NVFP4 MoE gridbook artifact was served as part of this docs review,
and this page intentionally publishes no throughput or quality number. Runtime
support remains a property of the exact vLLM wheel, FlashInfer package, model
configuration, and GPU.

See also [installation and hardware](INSTALL.md#hardware-matrix),
[plugin dispatch](PLUGIN.md), and [the format specification](SPEC.md).
