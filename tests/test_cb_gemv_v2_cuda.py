"""GB10 correctness and execution-contract tests for CB-GEMV-v2.

These tests use PrismaQuant's real two-tier encoder rather than fabricating a
packed scale plane. They skip cleanly without CUDA/nvcc or outside the narrow
GB10 device family on which the experimental kernel is admitted.
"""
from __future__ import annotations

import json
import os
import struct
from contextlib import contextmanager

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
from gridbook import ops as pq_ops  # noqa: E402
from gridbook.cuda_ext import get_ext, get_ext_v2  # noqa: E402
from gridbook.moe_gemv_select import cb_gemv_v2_device_support  # noqa: E402
from cb_torch_reference import (  # noqa: E402
    cb_linear_reference,
    reconstruct_cb_weight,
    synth_product_codebook,
    synth_two_tier_v2_plane,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)
supported, reason, _device_index = cb_gemv_v2_device_support("cuda:0")
if not supported:
    pytest.skip(reason, allow_module_level=True)
inherited = get_ext()
v2ext = get_ext_v2()
if inherited is None or v2ext is None:
    pytest.skip("required CUDA extensions could not be built",
                allow_module_level=True)
v2ext.cb_gemv_v2_prepare()


@contextmanager
def _env(**values):
    old = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_bf16_safetensors(path, tensors):
    """Write the tiny producer fixture without safetensors' NumPy writer.

    Gridbook's installed-wheel test closure deliberately excludes NumPy.  The
    exporter-consumer seam only needs a legal source checkpoint, so serialize
    its single BF16 tensor directly using the documented safetensors layout.
    """

    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        data = tensor.detach().to(device="cpu", dtype=torch.bfloat16)
        data = data.contiguous()
        raw = bytes(data.view(torch.uint8).reshape(-1).tolist())
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(data.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _case(k: int, K: int, *, seed: int = 1, E: int = 3, N: int = 17):
    device = torch.device("cuda")
    cb = pq._resolve_codebook(k, "fp4", "product", None, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn(E, N, K, generator=generator) * 0.02).to(device)
    fields = pq.nvfp4_cb_fields(
        weight, k, grid="fp4", mode="product", codebook=cb,
        scale_coding="two_tier", encode_tier="fast")
    packed = pq.nvfp4_cb_assemble_bytes(
        fields, k, grid="fp4", mode="product")
    type_size = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    qw = packed.reshape(E, N, (K // 256) * type_size).contiguous().to(device)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    cb_flat = codec.build_flat_codebook(subs)
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(device)
    x = torch.randn(4, K, generator=generator,
                    dtype=torch.bfloat16).to(device)
    pair_expert = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32,
                               device=device)
    pair_xrow = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.int32,
                             device=device)
    return x, qw, cb_flat, compose, pair_expert, pair_xrow, type_size


def _synthetic_case(k: int, K: int = 256, *, seed: int = 1,
                    E: int = 2, N: int = 9):
    """Legal v2 bytes over the full ABI without depending on lattice assets."""

    device = torch.device("cuda")
    tables = [table.to(torch.bfloat16).to(device)
              for table in synth_product_codebook(k, seed=seed)]
    cb_flat = codec.build_flat_codebook(tables)
    packed = synth_two_tier_v2_plane(
        E * N, K, k, sub_table=codec.TWO_TIER_SUB_TABLE,
        seed=seed + 1, device=device, super_span=1)
    type_size = 4 * k + 9
    qw = packed.reshape(E, N, (K // 256) * type_size).contiguous()
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    x = torch.randn(3, K, generator=generator,
                    dtype=torch.bfloat16).to(device)
    pair_expert = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    pair_xrow = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    return x, qw, cb_flat, compose, pair_expert, pair_xrow, type_size


def _run_v2(case, k, *, dict_mode=0):
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    return v2ext.cb_gemv_v2(
        x, qw, cb, compose, pair_expert, pair_xrow,
        k, type_size, 0, dict_mode)


def _run_inherited(case, k):
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    return inherited.cb_moe_gemv_fp4_v2(
        x, qw, cb, compose, pair_expert, pair_xrow, k, 2, type_size)


@pytest.mark.parametrize("k", range(1, 33))
def test_direct_nvfp4_research_range_inherited_dense_grouped_and_expand(k):
    """Low-level CUDA coverage: public K1..K25 plus research-only K26..K32."""

    case = _synthetic_case(k, seed=30_000 + k)
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    E, N, row_bytes = qw.shape
    K = (row_bytes // type_size) * 256
    flat_rows = qw.reshape(E * N, row_bytes)
    row_offsets = torch.zeros(E * N, dtype=torch.int32, device=qw.device)
    reference_w = reconstruct_cb_weight(
        flat_rows, cb, row_offsets, torch.zeros(1, device=qw.device), compose,
        N=E * N, K=K, k_bits=k, n_sub=2, type_size=type_size,
        is_fp4=True, is_v2=True)

    expanded = v2ext.cb_expand_v2(
        flat_rows.reshape(-1), cb, compose, 0, E * N, K, k, type_size)
    assert torch.equal(expanded.view(torch.uint16),
                       reference_w.view(torch.uint16))

    padded = codec.pad_qweight(flat_rows)
    dense = inherited.cb_gemv_fp4_v2(
        x[:1], padded, cb, row_offsets, compose, E * N, K, k, 2, type_size)
    dense_ref = cb_linear_reference(
        x[:1], padded, cb, row_offsets, torch.zeros(1, device=qw.device),
        compose, N=E * N, K=K, k_bits=k, n_sub=2,
        type_size=type_size, is_fp4=True, is_v2=True)
    assert torch.allclose(dense.float(), dense_ref.float(), rtol=8e-3,
                          atol=2e-2)

    grouped = _run_inherited(case, k)
    grouped_ref = torch.stack([
        (x[int(xrow)].float()
         @ reference_w[int(expert) * N:(int(expert) + 1) * N].float().T
         ).to(torch.bfloat16)
        for expert, xrow in zip(pair_expert.cpu(), pair_xrow.cpu())
    ])
    assert torch.allclose(grouped.float(), grouped_ref.float(), rtol=8e-3,
                          atol=2e-2)


@pytest.mark.parametrize("k", (1, 25), ids=lambda k: f"k{k}")
@pytest.mark.parametrize("exporter_name", ("resident", "streaming"))
def test_exported_public_endpoint_bytes_execute_in_cuda_consumer(
    k, exporter_name, tmp_path,
):
    """Close the producer/export/runtime seam at both public endpoints.

    The all-rung kernel gate above deliberately synthesizes legal bytes so it
    can retain K26..K32 as a research surface.  This companion test runs the
    real PrismaQuant resident and streaming exporters, reloads their
    safetensors and sidecars, validates the exact K1/K25 product-table geometry
    with Gridbook, and feeds those exported bytes to both the shipping expander
    and dense CUDA GEMV.
    """

    from safetensors.torch import load_file
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming

    qname = "model.layers.0.self_attn.q_proj"
    device = torch.device("cuda")
    source = tmp_path / f"source-{exporter_name}-k{k}"
    output = tmp_path / f"output-{exporter_name}-k{k}"
    source.mkdir()
    generator = torch.Generator(device="cpu").manual_seed(60_000 + k)
    weight = (torch.randn(8, 256, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    _write_bf16_safetensors(
        source / "model.safetensors", {f"{qname}.weight": weight}
    )
    (source / "config.json").write_text(json.dumps({"hidden_size": 256}))
    assignment = tmp_path / f"assignment-{exporter_name}-k{k}.json"
    assignment.write_text(json.dumps({qname: f"NVFP4_CB_K{k}"}))
    exporter = (
        export_nvfp4_cb
        if exporter_name == "resident"
        else export_nvfp4_cb_streaming
    )

    with _env(PRISMAQUANT_CB_ENCODE_COMPILE="0",
              PRISMAQUANT_CB_LDLQ="0"):
        exporter(
            source,
            assignment,
            output,
            {qname: torch.linspace(0.5, 1.5, 256)},
            shared_codebook_spec={"source": "lattice"},
            device="cpu",
            allow_unstamped_research=True,
        )

    quant_config = json.loads((output / "quant_config.json").read_text())
    group = next(
        item for item in quant_config["config_groups"].values()
        if qname in item["targets"]
    )
    scheme = group["scheme"]
    assert (scheme["k"], scheme["n_sub"], scheme["type_size"]) == (
        k, 2, 4 * k + 9
    )
    tensors = load_file(str(output / "model.safetensors"))
    sidecar = load_file(str(output / quant_config["codebook_file"]))
    refs = scheme["codebook_ref"]
    names = list(refs) if isinstance(refs, list) else [refs]
    tables = [sidecar[name].to(device) for name in names]
    cb_flat = codec.build_flat_product_codebook(
        tables, k, 2, prefix=f"exported NVFP4_CB_K{k}", grid="fp4"
    )
    packed = tensors[f"{qname}.cb_qweight"].to(device).contiguous()
    rows, row_bytes = packed.shape
    type_size = int(scheme["type_size"])
    K = (row_bytes // type_size) * 256
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(device)
    row_offsets = torch.zeros(rows, dtype=torch.int32, device=device)

    expanded = v2ext.cb_expand_v2(
        packed.reshape(-1), cb_flat, compose, 0, rows, K, k, type_size
    )
    padded = codec.pad_qweight(packed)
    reference_w = reconstruct_cb_weight(
        padded, cb_flat, row_offsets, torch.zeros(1, device=device), compose,
        N=rows, K=K, k_bits=k, n_sub=2, type_size=type_size,
        is_fp4=True, is_v2=True,
    )
    assert torch.equal(expanded.view(torch.uint16),
                       reference_w.view(torch.uint16))

    x = torch.randn(3, K, generator=generator, dtype=torch.bfloat16).to(device)
    got = inherited.cb_gemv_fp4_v2(
        x, padded, cb_flat, row_offsets, compose,
        rows, K, k, 2, type_size,
    )
    expected = cb_linear_reference(
        x, padded, cb_flat, row_offsets, torch.zeros(1, device=device), compose,
        N=rows, K=K, k_bits=k, n_sub=2, type_size=type_size,
        is_fp4=True, is_v2=True,
    )
    assert torch.allclose(got.float(), expected.float(), rtol=8e-3,
                          atol=2e-2)


@pytest.mark.parametrize("k,staged", [(1, True), (25, True),
                                       (26, False), (32, False)])
def test_expander_staging_boundary_and_v2_global_parity(k, staged):
    case = _synthetic_case(k, K=512, seed=40_000 + k)
    _x, qw, cb, compose, _pair_expert, _pair_xrow, type_size = case
    E, N, row_bytes = qw.shape
    K = (row_bytes // type_size) * 256
    assert bool(v2ext.cb_expand_v2_stages_dictionary(k)) is staged
    flat_rows = qw.reshape(E * N, row_bytes)
    offsets = torch.zeros(E * N, dtype=torch.int32, device=qw.device)
    expanded = v2ext.cb_expand_v2(
        flat_rows.reshape(-1), cb, compose, 0, E * N, K, k, type_size)
    expanded_ref = reconstruct_cb_weight(
        flat_rows, cb, offsets, torch.zeros(1, device=qw.device), compose,
        N=E * N, K=K, k_bits=k, n_sub=2, type_size=type_size,
        is_fp4=True, is_v2=True)
    assert torch.equal(expanded.view(torch.uint16),
                       expanded_ref.view(torch.uint16))
    with _env(PRISMAQUANT_CB_W2_SCHED="rowpack"):
        want = _run_inherited(case, k)
        # GLOBAL is valid for every rung and is the mandatory high-rung route.
        got = _run_v2(case, k, dict_mode=1)
        assert torch.equal(got.view(torch.uint16), want.view(torch.uint16))
        if staged:
            # FULL exercises K1's 24-byte tail copy and K25's 98,304-byte
            # near-budget dictionary.
            got_full = _run_v2(case, k, dict_mode=3)
            assert torch.equal(got_full.view(torch.uint16),
                               want.view(torch.uint16))


@pytest.mark.parametrize("k", [1, 25, 26, 32])
def test_expander_shoulder_cuda_graph_replay_reloads_packed_bytes(k):
    """Replay crosses both sides of the full-LUT residency shoulder.

    The graph owns a stable output allocation but must reread the caller-owned
    packed bytes on every replay.  K25 is the 98,304-byte staged launch; K26 is
    the first zero-dynamic-smem/global-LUT launch.  K1 and K32 pin the packed
    codeword endpoints at the same time.
    """

    case = _synthetic_case(k, seed=50_000 + k)
    _x, qw, cb, compose, _pair_expert, _pair_xrow, type_size = case
    rows, K = qw.shape[0] * qw.shape[1], 256
    live = qw.reshape(-1).clone()
    changed_qw = qw.clone()
    # Only mutate the code plane. Every bit pattern remains a valid packed
    # product index, while the two-tier scale bytes stay numerically tame.
    changed_qw[..., :4 * k].bitwise_xor_(0xA5)
    changed = changed_qw.reshape(-1).contiguous()

    def run(packed):
        return pq_ops.cb_expand_fp4_v2(
            packed, cb, compose, 0, rows, K, k, type_size)

    eager_before = run(live)
    eager_after = run(changed)
    assert not torch.equal(eager_before.view(torch.uint16),
                           eager_after.view(torch.uint16))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(live)

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.uint16),
                       eager_before.view(torch.uint16))
    live.copy_(changed)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.uint16),
                       eager_after.view(torch.uint16))


@pytest.mark.parametrize("k", [1, 25, 26, 32])
def test_expander_shoulder_torch_compile_fullgraph(k):
    """The registered expander remains opaque and shape-correct to Dynamo."""

    case = _synthetic_case(k, seed=60_000 + k)
    _x, qw, cb, compose, _pair_expert, _pair_xrow, type_size = case
    rows, K = qw.shape[0] * qw.shape[1], 256
    packed = qw.reshape(-1)

    def run(packed_arg):
        return pq_ops.cb_expand_fp4_v2(
            packed_arg, cb, compose, 0, rows, K, k, type_size)

    eager = run(packed)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    got = compiled(packed)
    assert got.shape == (rows, K)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.uint16), eager.view(torch.uint16))


@pytest.mark.parametrize("k,K", [
    (13, 512),
    (16, 1536),
    (20, 2304),
    (24, 3840),
])
@pytest.mark.parametrize("contract", [None, "v2"])
def test_v2_is_bit_exact_to_inherited_rowpack(k, K, contract):
    case = _case(k, K, seed=k + K)
    with _env(PRISMAQUANT_CB_W2_SCHED="rowpack",
              PRISMAQUANT_CB_DECODE_CONTRACT=contract):
        want = _run_inherited(case, k)
        got = _run_v2(case, k)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("k,K", [
    (12, 2048),
    (12, 4096),
    (16, 2048),
    (16, 4096),
    (18, 2048),
    (18, 4096),
])
def test_v2_release_shapes_are_bit_exact_to_inherited_default(k, K):
    """Pin the actual DSV4 target/draft release baseline, not rowpack.

    The inherited default selects eight warps for both production widths
    (n_sb=8/16). Exercise both decode contracts and every relevant dictionary
    residency: auto, forced-global, forced-half and forced-full must all
    reproduce its exact BF16 output. This is the coverage the original
    rowpack-only parity test did not provide.
    """
    case = _case(k, K, seed=10_000 + 100 * k + K)
    for contract in (None, "v2"):
        with _env(PRISMAQUANT_CB_W2_SCHED=None,
                  PRISMAQUANT_CB_W2_ROWS=None,
                  PRISMAQUANT_CB_W2_WARPS=None,
                  PRISMAQUANT_CB_DECODE_CONTRACT=contract):
            want = _run_inherited(case, k)
            for dict_mode in (0, 1, 2, 3):  # auto, GLOBAL, HALF, FULL
                got = _run_v2(case, k, dict_mode=dict_mode)
                assert torch.equal(got.view(torch.int16),
                                   want.view(torch.int16)), (
                    f"k={k} K={K} contract={contract or 'v1'} "
                    f"dict_mode={dict_mode}: v2 != inherited default")


def test_compiled_dispatch_predicate_covers_measured_wall_and_invalid_inputs():
    assert v2ext.cb_gemv_v2_prefers_inherited(24, 105, 2048) is True
    assert v2ext.cb_gemv_v2_prefers_inherited(24, 105, 1536) is False
    assert v2ext.cb_gemv_v2_prefers_inherited(20, 89, 4096) is False
    assert v2ext.cb_gemv_v2_prefers_inherited(0, 9, 4096) is True
    assert v2ext.cb_gemv_v2_prefers_inherited(16, 73, 0) is True


def test_rowpack_cuda_graph_replay_matches_eager():
    k, K = 16, 512
    case = _case(k, K, seed=77)
    with _env(PRISMAQUANT_CB_DECODE_CONTRACT="v2"):
        eager = _run_v2(case, k)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _run_v2(case, k)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("k,K", [
    (12, 2048),
    (12, 4096),
    (16, 2048),
    (16, 4096),
    (18, 2048),
    (18, 4096),
])
@pytest.mark.parametrize("contract", [None, "v2"])
def test_release_shape_cuda_graph_replay_matches_inherited_default(
        k, K, contract):
    case = _case(k, K, seed=20_000 + 100 * k + K)
    with _env(PRISMAQUANT_CB_W2_SCHED=None,
              PRISMAQUANT_CB_W2_ROWS=None,
              PRISMAQUANT_CB_W2_WARPS=None,
              PRISMAQUANT_CB_DECODE_CONTRACT=contract):
        want = _run_inherited(case, k)
        for dict_mode in (0, 1, 2, 3):
            eager = _run_v2(case, k, dict_mode=dict_mode)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _run_v2(case, k, dict_mode=dict_mode)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(eager.view(torch.int16),
                               want.view(torch.int16))
            assert torch.equal(captured.view(torch.int16),
                               want.view(torch.int16))


def test_custom_op_torch_compile_contract():
    k, K = 16, 512
    case = _case(k, K, seed=88)
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case

    def run(x_arg):
        return pq_ops.cb_moe_gemv_v2(
            x_arg, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 0, 0)

    eager = run(x)
    compiled = torch.compile(run, backend="eager", fullgraph=True)(x)
    assert torch.equal(compiled.view(torch.int16), eager.view(torch.int16))


def test_debug_bindings_reject_malformed_inputs():
    k, K = 16, 512
    case = _case(k, K, seed=99)
    x, qw, cb, compose, pair_expert, pair_xrow, type_size = case
    with pytest.raises(RuntimeError, match="compose must"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose.cpu(), pair_expert, pair_xrow,
            k, type_size, 0, 0)
    with pytest.raises(RuntimeError, match="dict_mode"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 0, 4)
    with pytest.raises(RuntimeError, match="rpb"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow,
            k, type_size, 1025, 0)
    with pytest.raises(RuntimeError, match="same length"):
        v2ext.cb_gemv_v2(
            x, qw, cb, compose, pair_expert, pair_xrow[:-1],
            k, type_size, 0, 0)
    flat = qw.reshape(-1)
    with pytest.raises(RuntimeError, match="positive.*multiple of 256"):
        v2ext.cb_expand_v2(flat, cb, compose, 0, 1, 511, k, type_size)
    rows_total = flat.numel() // ((K // 256) * type_size)
    with pytest.raises(RuntimeError, match="outside"):
        v2ext.cb_expand_v2(
            flat, cb, compose, rows_total, 1, K, k, type_size)
