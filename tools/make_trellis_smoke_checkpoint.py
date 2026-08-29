"""Emit a tiny, self-consistent trellis checkpoint -- the exporter's embryo.

WHY THIS EXISTS.  Nothing produces a trellis checkpoint, so "Gridbook can serve
a trellis wire" could only ever be tested against hand-built layers. This
writes the real thing: a model directory vLLM can open, whose ``o_proj`` and
``down_proj`` Linears carry TCQ wires declared through the ``config_groups``
vocabulary in ``trellis_scheme``.

SELF-CONSISTENT, NOT ENCODED.  There is no weight->wire encoder in this
package (the Viterbi encoder lives in the research trees), so this tool builds
a VALID wire first and takes the model's reference weight to be that wire's own
decoded value. That is the right scope for a serving smoke and the wrong scope
for anything else: it exercises load, dispatch, residency and the kernel, and
it says NOTHING about encoding quality, which is measured separately. A real
exporter substitutes an encoder here and changes nothing else.

FUSED MODULES ARE LEFT ALONE ON PURPOSE.  vLLM merges q/k/v and gate/up, and
per-role wires cannot be concatenated (each carries its own alphabets, rate
schedule and row padding), so those go in ``ignore`` as bf16. ``o_proj`` and
``down_proj`` are unfused on this architecture and are the trellis units.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import struct
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gridbook import trellis                                    # noqa: E402

GROUP = 16


def _wire(family: str, rows: int, columns: int, q256: int, seed: int):
    """A valid wire of the requested family and geometry."""
    template = trellis.build_q256_schedule(family, q256, 256)
    expanded = tuple(template[i % 256] for i in range(columns))
    terminal = trellis.native_bits(family)
    alphabets = {
        rate: trellis.canonical_full_alphabet(family)[:1 << (rate + 1)]
        for rate in sorted({r for r in expanded if r < terminal})
    }
    rng = random.Random(seed)
    u = [[rng.getrandbits(1) for _ in range(columns)] for _ in range(rows)]
    points = [[0] * columns for _ in range(rows)]
    bypass = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        for column, rate in enumerate(expanded):
            if rate == terminal:
                code = rng.randrange(1 << terminal)
                if family == trellis.TCQ_E4M3_R256:
                    # Never emit a NaN/Inf e4m3 code: the wire refuses them,
                    # and a served NaN would look like a kernel bug.
                    while code in (0x7F, 0xFF, 0x7E, 0xFE):
                        code = rng.randrange(1 << terminal)
                bypass[row][column] = code
            else:
                points[row][column] = rng.randrange(1 << (rate - 1))
    if family == trellis.TCQ_E2M1_R256:
        groups = (columns + GROUP - 1) // GROUP
        scale_blob = bytes(0x30 + ((r * groups + g) % 0x14)
                           for r in range(rows) for g in range(groups))
    else:
        # One fp32 scale per row; global_scale_real stays 1.0 by contract.
        scale_blob = struct.pack(
            f"<{rows}f", *[0.02 + 0.001 * r for r in range(rows)])
    return trellis.pack_planes(
        family=family, body_rate_q256=q256, schedule=expanded,
        layout=trellis.LAYOUT_FIXED_QUOTA_PER_256,
        u_bits=u, point_indices=points, bypass_codes=bypass,
        alphabets=alphabets, scale_blob=scale_blob)


def _scheme(wire, blob):
    return {"family": wire.family, "body_rate_q256": wire.body_rate_q256,
            "rows": wire.rows, "columns": wire.columns,
            "wire_bytes": len(blob)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--family", default=trellis.TCQ_E4M3_R256,
                    choices=list(trellis.FAMILIES))
    ap.add_argument("--rate", type=int, default=None,
                    help="body_rate_q256; default = the family's first rung")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--reference", action="store_true",
                    help="also write each wire's decoded weight (host checks)")
    args = ap.parse_args()

    rate = args.rate
    if rate is None:
        rate = (1152 if args.family == trellis.TCQ_E4M3_R256 else 512)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tensors: dict[str, torch.Tensor] = {}
    reference: dict[str, torch.Tensor] = {}
    groups: dict[str, dict] = {}
    ignore: list[str] = []
    H, I, L, V = args.hidden, args.intermediate, args.layers, args.vocab

    tensors["model.embed_tokens.weight"] = torch.randn(V, H) * 0.02
    tensors["model.norm.weight"] = torch.ones(H)
    tensors["lm_head.weight"] = torch.randn(V, H) * 0.02
    ignore += ["lm_head"]

    for layer in range(L):
        stem = f"model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{stem}.{name}.weight"] = torch.ones(H)
        # Fused on the vLLM side -> bf16, and named in ignore.
        for name, (o, i) in (("self_attn.q_proj", (H, H)),
                             ("self_attn.k_proj", (H, H)),
                             ("self_attn.v_proj", (H, H)),
                             ("mlp.gate_proj", (I, H)),
                             ("mlp.up_proj", (I, H))):
            tensors[f"{stem}.{name}.weight"] = torch.randn(o, i) * 0.02
            ignore.append(f"{stem}.{name}")
        # The trellis units: unfused, one wire each.
        for name, (rows, columns) in (("self_attn.o_proj", (H, H)),
                                      ("mlp.down_proj", (H, I))):
            target = f"{stem}.{name}"
            wire = _wire(args.family, rows, columns, rate,
                         seed=1009 * layer + len(name))
            blob = wire.to_bytes()
            tensors[f"{target}.wire_bytes"] = torch.frombuffer(
                bytearray(blob), dtype=torch.uint8).clone()
            if args.family == trellis.TCQ_E2M1_R256:
                # The A-side static scale: the one quantity that is NOT a wire
                # fact, so the one a checkpoint must carry.
                tensors[f"{target}.trellis_input_global_scale"] = torch.tensor(
                    [4.0], dtype=torch.float32)
            groups[f"trellis_{layer}_{name.replace('.', '_')}"] = {
                "format": "TRELLIS", "targets": [target],
                "scheme": _scheme(wire, blob)}
            if args.reference:
                # The reference weight IS the wire's decoded value (docstring).
                reference[target] = trellis.decode_values_torch(
                    wire, device="cpu").to(torch.float32)

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(out / "model.safetensors"))
    if reference:
        # OUTSIDE the model directory, and not named *.safetensors within it:
        # vLLM globs every shard in the model dir, so a reference file placed
        # there is loaded as model weights and fails on the first key that
        # names no parameter. (It did, on the first container run.)
        save_file({k: v.contiguous() for k, v in reference.items()},
                  str(out.parent / f"{out.name}-reference.safetensors"))

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": H, "intermediate_size": I,
        "num_hidden_layers": L, "num_attention_heads": args.heads,
        "num_key_value_heads": args.heads, "vocab_size": V,
        "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "gridbook", "format": "mixed-precision",
            "config_groups": groups, "ignore": sorted(set(ignore)),
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"wrote {out} -- {len(groups)} trellis units, "
          f"{args.family} R{rate}, {len(tensors)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
