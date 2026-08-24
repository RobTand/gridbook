"""Shard-aware loading for dense CB Linears at TP>1 (campaign 2026-08-23).

Dense CB Linears shard purely at LOAD time; no exported byte changes. This
file pins the three facts that make that true instead of assumed:

1. **The slicing law** (validation.md T1/T2): a row-parallel K-shard cut on
   a superblock boundary reconstructs BITWISE to the whole-tensor decode,
   including the fp4 two-tier scale plane whose E8M0 exponent rides INSIDE
   each superblock; column-parallel row slices are independent streams. The
   oracle is ``tests/cb_torch_reference`` — the independent decoder that
   deliberately restates format constants rather than importing them.
2. **The structured refusal**: a shard boundary that WOULD split a group
   raises ``ShardGroupAlignmentError`` at weight construction, carrying
   qname / axis / group_size / tp_degree / shard_size as fields. Gates read
   the fields; ``str(exc)`` is rendered prose, never the contract.
3. **The refusal lattice**: dense CB Linears construct at TP=2, as do
   qualified source-passthrough units and the mixed-format fused projections
   composed from them (each role's own law gates its own carrier — see
   ``test_fp8_source_w8a16_tp_shard.py`` and
   ``test_mixed_fused_tp_shard.py``); MoE expert stacks, delegated
   compressed-tensors groups, unqualified source-passthrough formats and
   quantized embeddings refuse AT CONSTRUCTION naming themselves;
   unquantized (ignored) targets stay on vLLM-native BF16 sharding; TP=1 and
   uninitialized-parallelism behave exactly as before.

Section 2 simulates the loader contract IN-PROCESS: there is one GPU in this
box and no second rank, so vLLM's attested v2 narrowing arithmetic
(``load_column_parallel_weight`` / ``load_row_parallel_weight``, extracted
from the pinned serving image; see dq-runs tp-support-2026-08-22 reports
``vllmcontract.md`` and ``cbshard.md``) is applied by hand to the checkpoint
tensors before Gridbook's own load finalization runs. This proves byte
placement and reconstruction identity; it does NOT prove a distributed run —
no engine, no collectives, no second device.

CPU-only where possible; against the REAL vLLM parameter classes when vLLM
is installed (same idiom as ``test_weight_residency``).
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from cb_torch_reference import (
    E2M1_MAGNITUDES,
    extract_codewords,
    reconstruct_cb_weight,
    synth_product_codebook,
    synth_two_tier_v2_plane,
    two_tier_compose_legality,
)


# ---------------------------------------------------------------------------
# vLLM surface: real when installed, minimal stubs otherwise.
# ---------------------------------------------------------------------------

def _install_vllm_stubs():
    def _mod(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    _mod("vllm")
    _mod("vllm.model_executor")
    _mod("vllm.model_executor.layers")
    _mod("vllm.model_executor.layers.quantization")
    linear = _mod("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    base = _mod("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def __init__(self):
            pass

    base.QuantizationConfig = QuantizationConfig
    base.QuantizeMethodBase = object
    embedding = _mod("vllm.model_executor.layers.vocab_parallel_embedding")
    embedding.UnquantizedEmbeddingMethod = type("UEM", (), {})
    embedding.VocabParallelEmbedding = type("VPE", (), {})
    # Subclass, as in vLLM -- see test_target_namespace_compat.
    embedding.ParallelLMHead = type(
        "ParallelLMHead", (embedding.VocabParallelEmbedding,), {})
    fused_moe = _mod("vllm.model_executor.layers.fused_moe")
    fused_moe.RoutedExperts = type("RoutedExperts", (), {})
    parameter = _mod("vllm.model_executor.parameter")

    class _StubParam(torch.nn.Parameter):
        def __new__(cls, data, **kw):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kw):
            pass

    parameter.ModelWeightParameter = _StubParam
    parameter.ChannelQuantScaleParameter = _StubParam
    parameter.PerTensorScaleParameter = _StubParam


def _require_real_vllm():
    """Cases that reach ``PrismaQuantConfig._is_ignored`` need the real thing:
    it delegates to compressed-tensors' ``should_ignore_layer`` (deliberately —
    a local near-copy once made one ignore list mean two things), which the
    minimal stubs above do not recreate."""
    if globals().get("VLLM_IS_STUBBED"):
        pytest.skip("reaches config._is_ignored, which delegates to "
                    "compressed-tensors' should_ignore_layer; needs real vLLM")


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    del isolated_gridbook_runtime_imports
    stubbed = False
    try:
        from vllm.model_executor.parameter import ModelWeightParameter  # noqa: F401
        from vllm.model_executor.layers.linear import LinearMethodBase  # noqa: F401
    except Exception:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                sys.modules.pop(name, None)
        _install_vllm_stubs()
        stubbed = True
    globals()["VLLM_IS_STUBBED"] = stubbed

    from gridbook import codec as codec_module
    from gridbook.config import PrismaQuantConfig as config_class
    from gridbook import config as config_module
    from gridbook import linear as linear_module
    from gridbook.linear import (
        PrismaQuantCBLinearMethod as method_class,
        ShardGroupAlignmentError as alignment_error,
    )
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    globals()["codec"] = codec_module
    globals()["PrismaQuantConfig"] = config_class
    globals()["config_mod"] = config_module
    globals()["cb_linear"] = linear_module
    globals()["PrismaQuantCBLinearMethod"] = method_class
    globals()["ShardGroupAlignmentError"] = alignment_error
    globals()["LinearBase"] = LinearBase
    globals()["UnquantizedLinearMethod"] = UnquantizedLinearMethod
    globals()["VocabParallelEmbedding"] = VocabParallelEmbedding
    globals()["ParallelLMHead"] = ParallelLMHead
    globals()["RoutedExperts"] = RoutedExperts
    yield


@pytest.fixture(autouse=True)
def _reset_dispatch_caches():
    """Keep process-global dispatch policy independent between tests."""
    cb_linear._FP4_FUSED_MODE.clear()
    cb_linear._FP4_DENSE_SM_COUNTS.clear()
    yield
    cb_linear._FP4_FUSED_MODE.clear()
    cb_linear._FP4_DENSE_SM_COUNTS.clear()


@pytest.fixture(autouse=True)
def _pin_tp_group(monkeypatch):
    """Real vLLM parameters read the process-global TP group at __init__.

    Pin rank 0 / world 1 so parameters can be CONSTRUCTED outside an engine;
    every TP>1 fact in this file arrives through constructor arguments and
    explicit rank arithmetic, never through that global state.
    """
    par = sys.modules.get("vllm.model_executor.parameter")
    if par is None or not hasattr(par, "get_tensor_model_parallel_rank"):
        return
    monkeypatch.setattr(par, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(par, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def _stub_native_attestation(monkeypatch):
    """Load-time native-kernel attestation is CUDA work; stub the edges.

    The layout/finalize code under test is host arithmetic. Fail-closed
    attestation behaviour itself has dedicated tests elsewhere
    (test_fail_loud.py); here it must simply succeed without a GPU.
    """
    from gridbook import cuda_ext, ops

    native = types.SimpleNamespace()
    monkeypatch.setattr(cuda_ext, "require_ext", lambda operation: native)
    monkeypatch.setattr(cuda_ext, "require_ext_v2", lambda operation: native)
    monkeypatch.setattr(
        cuda_ext, "require_fp4_v2_expander",
        lambda operation, **kwargs: native)
    monkeypatch.setattr(
        cuda_ext, "require_bf16_grouped_ext", lambda operation: native)
    monkeypatch.setattr(cuda_ext, "get_ext", lambda: native)
    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: None)
    monkeypatch.setattr(ops, "register_cb_layer", lambda method, layer: 4101)
    monkeypatch.setattr(
        cb_linear, "require_native_fp8_cutlass", lambda operation: None)
    monkeypatch.setattr(
        cb_linear, "require_native_fp4_quant", lambda operation: None)


class _Layer(torch.nn.Module):
    """Stands in for vLLM's LinearBase: parameter registration only."""


def _linear_layer():
    """A LinearBase INSTANCE for dispatch, bypassing engine __init__ args."""
    return object.__new__(LinearBase)


def _world_size(monkeypatch, size):
    """Pin the LIVE vLLM world size the config gate reads."""
    monkeypatch.setattr(
        config_mod, "_initialized_tensor_parallel_world_size",
        lambda: size)


# ---------------------------------------------------------------------------
# Synthetic FP8-CB / FP4-CB-v2 fixtures (format constants restated by the
# independent reference module; see cbshard.md §"What the exporter ships").
# ---------------------------------------------------------------------------

K_FULL = 1024          # 4 superblocks
N_FULL = 64            # halves stay % 16 (fp8) and % 8 (fp4) legal at TP=2
FP8_K_BITS, FP8_N_SUB, FP8_TS = 32, 4, 128       # type_size = 4k
FP4_K_BITS, FP4_N_SUB, FP4_TS = 12, 2, 57        # type_size = 4k + 9


def _signed_grid_table(rows: int, cols: int, generator) -> torch.Tensor:
    grid = torch.tensor(
        sorted({v for m in E2M1_MAGNITUDES for v in (m, -m)}),
        dtype=torch.float32)
    pick = torch.randint(0, grid.numel(), (rows * cols,),
                         generator=generator)
    return grid[pick].reshape(rows, cols)


def _fp8_fixture(seed: int = 11):
    """A format-legal FP8-CB payload: any index bytes decode; scales per row.

    Returns (qw [N, row_bytes] uint8, sub_tables, scale [N] fp32, cb_flat,
    row_offsets [N] int32).
    """
    generator = torch.Generator().manual_seed(seed)
    row_bytes = (K_FULL // 256) * FP8_TS
    qw = torch.randint(0, 256, (N_FULL, row_bytes), dtype=torch.uint8,
                       generator=generator)
    widths = [8, 8, 8, 8]                       # split(32, 4), ceil-first
    subs = [_signed_grid_table(1 << w, 8 // FP8_N_SUB, generator)
            for w in widths]
    cb_flat = torch.cat([t.reshape(-1) for t in subs]).contiguous()
    scale = torch.rand(N_FULL, generator=generator) * 3 + 0.25
    offsets = torch.zeros(N_FULL, dtype=torch.int32)
    return qw, subs, scale, cb_flat, offsets


def _fp4_v2_fixture(seed: int = 23):
    """A format-legal FP4-CB layout-v2 payload via the reference synthesizer.

    The 9-byte two-tier plane lives INSIDE each superblock, so this fixture
    is the one that catches a K-shard stranding a super exponent. The
    sub-table literal mirrors ``gridbook.codec.TWO_TIER_SUB_TABLE``; it is
    FIXTURE INPUT only — every decode below runs through the independent
    reference module, which recomputes the compose table from these values.
    """
    subs = synth_product_codebook(FP4_K_BITS, seed=seed)
    sub_table = (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
    plane = synth_two_tier_v2_plane(
        N_FULL, K_FULL, FP4_K_BITS, sub_table=sub_table, seed=seed + 1)
    compose, _legal = two_tier_compose_legality(sub_table)
    flat = torch.cat([t.reshape(-1) for t in subs]).contiguous()
    offsets = torch.zeros(N_FULL, dtype=torch.int32)
    return plane, subs, compose.reshape(-1).contiguous(), flat, offsets


def _decode(qw, cb_flat, offsets, scale_or_compose, *, N, K, is_fp4):
    return reconstruct_cb_weight(
        qw, cb_flat, offsets,
        torch.zeros(1) if is_fp4 else scale_or_compose,
        scale_or_compose if is_fp4 else torch.zeros(1),
        N=N, K=K,
        k_bits=FP4_K_BITS if is_fp4 else FP8_K_BITS,
        n_sub=FP4_N_SUB if is_fp4 else FP8_N_SUB,
        type_size=FP4_TS if is_fp4 else FP8_TS,
        is_fp4=is_fp4, is_v2=is_fp4)


# ---------------------------------------------------------------------------
# 1. The slicing law: aligned shards reconstruct bitwise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [2, 4])
def test_row_parallel_superblock_shards_reconstruct_bitwise_fp8(tp):
    """T1 oracle: byte windows taken exactly as the row-parallel loader takes
    them concatenate to the full plane, and their decodes concatenate to the
    whole-tensor decode — bit for bit."""
    qw, _subs, scale, cb_flat, offsets = _fp8_fixture()
    full = _decode(qw, cb_flat, offsets, scale, N=N_FULL, K=K_FULL,
                   is_fp4=False)

    k_local = K_FULL // tp
    sb_bytes = (k_local // 256) * FP8_TS
    shards, decoded = [], []
    for rank in range(tp):
        window = qw.narrow(1, rank * sb_bytes, sb_bytes)
        assert (rank * sb_bytes) % FP8_TS == 0
        shards.append(window)
        decoded.append(_decode(window, cb_flat, offsets, scale,
                               N=N_FULL, K=k_local, is_fp4=False))
    # Raw-byte identity catches off-by-one superblock offsets even where a
    # decode would accidentally tolerate them.
    assert torch.equal(torch.cat(shards, dim=1), qw)
    assert torch.equal(torch.cat(decoded, dim=1), full)


def test_row_parallel_shard_keeps_the_two_tier_scale_plane_whole():
    """The fp4-v2 scale plane (E8M0 super byte + sub nibbles) sits inside its
    superblock; a superblock-aligned cut must carry it intact."""
    plane, _subs, compose, cb_flat, offsets = _fp4_v2_fixture()
    full = _decode(plane, cb_flat, offsets, compose, N=N_FULL, K=K_FULL,
                   is_fp4=True)

    sb_bytes = (512 // 256) * FP4_TS
    halves = [plane.narrow(1, rank * sb_bytes, sb_bytes) for rank in range(2)]
    assert torch.equal(torch.cat(halves, dim=1), plane)
    decoded = [
        _decode(half, cb_flat, offsets, compose, N=N_FULL, K=512, is_fp4=True)
        for half in halves]
    assert torch.equal(torch.cat(decoded, dim=1), full)


@pytest.mark.parametrize("tp", [2, 4])
def test_column_parallel_row_slices_are_independent_streams(tp):
    """T2 oracle: rows travel whole; per-row reconstruction equals the whole-
    tensor reconstruction of those rows; the per-row scale follows them."""
    qw, _subs, scale, cb_flat, offsets = _fp8_fixture()
    full = _decode(qw, cb_flat, offsets, scale, N=N_FULL, K=K_FULL,
                   is_fp4=False)

    rows_local = N_FULL // tp
    slices, decoded, scales = [], [], []
    for rank in range(tp):
        rows = slice(rank * rows_local, (rank + 1) * rows_local)
        shard_qw = qw.narrow(0, rows.start, rows_local)
        shard_offsets = offsets.narrow(0, rows.start, rows_local)
        slices.append(shard_qw)
        scales.append(scale[rows])
        decoded.append(_decode(shard_qw, cb_flat, shard_offsets, scale[rows],
                               N=rows_local, K=K_FULL, is_fp4=False))
    assert torch.equal(torch.cat(decoded, dim=0), full)
    assert torch.equal(torch.cat(scales, dim=0), scale)
    # Rows are byte-independent: concatenating the packed rows restores the
    # plane exactly (there is no cross-row packing anywhere in the format).
    assert torch.equal(torch.cat(slices, dim=0), qw)


def test_misaligned_window_is_unrepresentable_as_a_legal_shard():
    """Negative branch of the slicing law: the smallest non-superblock cut
    does not map onto whole codeword/scale windows — the corruption the
    construction gate exists to make unrepresentable."""
    ts = FP8_TS
    k_bad = 384                              # 1.5 superblocks
    assert k_bad % 256 != 0
    byte_window = (k_bad // 256) * ts + ts // 2   # what a naive narrow gives
    assert byte_window % (2 * ts) != 0       # lands mid-superblock
    # The independent decoder REFUSES such a geometry outright...
    with pytest.raises(ValueError, match="multiple of 256"):
        extract_codewords(
            torch.zeros(N_FULL, byte_window, dtype=torch.uint8),
            N=N_FULL, K=k_bad, k_bits=FP8_K_BITS, type_size=FP8_TS)
    # ...and decoding the misranged window as if it were legal yields bytes
    # that differ from the true next-superblock content (garbage, not data).
    qw, _subs, _scale, _flat, _offsets = _fp8_fixture()
    assert not torch.equal(qw[:, :byte_window], qw[:, ts:ts + byte_window])


# ---------------------------------------------------------------------------
# 2. Loader-contract simulation at TP=2, in process (NO second rank: the vLLM
#    narrowing arithmetic applied here is attested from the pinned image —
#    parameter.py load_column_parallel_weight / load_row_parallel_weight —
#    not invented for this test; see vllmcontract.md).
# ---------------------------------------------------------------------------

_DENSE_SCHEME = {"grid": "fp8", "mode": "product", "k": FP8_K_BITS,
                 "n_sub": FP8_N_SUB, "type_size": FP8_TS, "group_size": 0,
                 "vec_dim": 8, "codebook_group": "mlp",
                 "codebook_source": "learned",
                 "codebook_ref": [f"cb.sub{i}" for i in range(FP8_N_SUB)]}

_DOWN_PROJ = "model.layers.0.mlp.down_proj"     # row-parallel
_GATE_PROJ = "model.layers.0.mlp.gate_proj"     # column-parallel


def _dense_config(targets):
    return {
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {"g": {"format": "FP8_CB_K32",
                                "targets": list(targets),
                                "scheme": dict(_DENSE_SCHEME)}},
        "ignore": ["lm_head"],
    }


def _rank_layer(prefix, *, rank, tp, row_parallel, qw_full, scale_full,
                codebooks):
    """create_weights -> simulated stock v2 narrowing -> load finalization.

    The narrowing mirrors the attested v2 formulas: the destination param is
    allocated at PER-RANK shape; the checkpoint tensor is narrowed at
    ``tp_rank * shard_size`` where shard_size is the PARAM's extent along the
    declared dim. ChannelQuantScaleParameter carries only output_dim, so on a
    row-parallel layer its base-class load copies WHOLE — the replicated
    per-output-row scale.
    """
    k_local = K_FULL // tp if row_parallel else K_FULL
    partitions = [N_FULL] if row_parallel else [N_FULL // tp]
    cfg = PrismaQuantConfig.from_config(_dense_config([prefix]))
    cfg._ensure_resolved()
    cfg.get_codebooks = lambda: dict(codebooks)
    method = PrismaQuantCBLinearMethod(cfg, dict(_DENSE_SCHEME), prefix)
    layer = _Layer()
    method.create_weights(layer, k_local, partitions, K_FULL, N_FULL,
                          torch.bfloat16, weight_loader=None)

    qw_param = layer.cb_qweight
    if row_parallel:
        source = qw_full.narrow(1, rank * qw_param.shape[1],
                                qw_param.shape[1])
    else:
        source = qw_full.narrow(0, rank * qw_param.shape[0],
                                qw_param.shape[0])
    qw_param.data.copy_(source)

    if row_parallel:
        layer.weight_scale.data.copy_(scale_full)
    else:
        layer.weight_scale.data.copy_(
            scale_full.narrow(0, rank * partitions[0], partitions[0]))

    method.process_weights_after_loading(layer)
    return method, layer


@pytest.mark.parametrize(("row_parallel", "prefix"), [
    (True, _DOWN_PROJ), (False, _GATE_PROJ)])
def test_loader_simulation_per_rank_bytes_and_decode_at_tp2(
        row_parallel, prefix):
    """Each rank's resident packed plane equals EXACTLY its attested window
    of the checkpoint, and decoding the rank-local layers reproduces exactly
    the whole-tensor reconstruction once concatenated along the sharded axis.
    """
    qw, subs, scale, cb_flat, offsets = _fp8_fixture()
    codebooks = {name: subs[i]
                 for i, name in enumerate(_DENSE_SCHEME["codebook_ref"])}
    full = _decode(qw, cb_flat, offsets, scale, N=N_FULL, K=K_FULL,
                   is_fp4=False)

    k_local = K_FULL // 2 if row_parallel else K_FULL
    n_local = N_FULL if row_parallel else N_FULL // 2
    rank_decodes = []
    for rank in range(2):
        _method, layer = _rank_layer(
            prefix, rank=rank, tp=2, row_parallel=row_parallel,
            qw_full=qw, scale_full=scale, codebooks=codebooks)
        # Placement facts recorded on the layer.
        if row_parallel:
            assert layer._cb_input_tp_degree == 2
            assert layer._cb_output_tp_degree == 1
        else:
            assert layer._cb_input_tp_degree == 1
            assert layer._cb_output_tp_degree == 2
        assert layer._cb_N == n_local and layer._cb_K == k_local
        assert layer.cb_qweight.shape == (n_local, (k_local // 256) * FP8_TS)
        # The replicated sidecar produced the SAME flat book on this rank...
        assert torch.equal(layer._cb_flat, cb_flat.to(torch.bfloat16))
        # ...and the per-rank offset table covers exactly this rank's rows.
        assert layer._cb_row_offset.numel() == n_local
        rank_decodes.append(_decode(
            layer.cb_qweight.data, layer._cb_flat, layer._cb_row_offset,
            layer._cb_scale, N=n_local, K=k_local, is_fp4=False))
    sharded_axis = 1 if row_parallel else 0
    assert torch.equal(torch.cat(rank_decodes, dim=sharded_axis), full)


def test_loader_simulation_row_parallel_replicates_channel_scales():
    """FP8-CB row-parallel keeps ALL N output rows per rank, so the
    per-output-row scale plane must arrive replicated (base-class copy),
    never narrowed."""
    qw, subs, scale, _flat, _offsets = _fp8_fixture()
    codebooks = {f"cb.sub{i}": subs[i] for i in range(len(subs))}
    for rank in range(2):
        _method, layer = _rank_layer(
            _DOWN_PROJ, rank=rank, tp=2, row_parallel=True,
            qw_full=qw, scale_full=scale, codebooks=codebooks)
        assert torch.equal(layer.weight_scale.data, scale)
        assert torch.equal(layer._cb_scale, scale.to(torch.float32))


def test_column_parallel_narrows_channel_scales_with_rows():
    """ChannelQuantScaleParameter(output_dim=0) follows the row split: rank r
    holds exactly scale[r*n_pp : (r+1)*n_pp]."""
    qw, subs, scale, _flat, _offsets = _fp8_fixture()
    codebooks = {f"cb.sub{i}": subs[i] for i in range(len(subs))}
    n_pp = N_FULL // 2
    for rank in range(2):
        _method, layer = _rank_layer(
            _GATE_PROJ, rank=rank, tp=2, row_parallel=False,
            qw_full=qw, scale_full=scale, codebooks=codebooks)
        assert layer.weight_scale.shape == (n_pp,)
        assert torch.equal(layer.weight_scale.data,
                           scale.narrow(0, rank * n_pp, n_pp))


# ---------------------------------------------------------------------------
# 3. Structured refusal at weight construction.
# ---------------------------------------------------------------------------

def _bare_method(is_fp4: bool, prefix: str = "model.layers.0.mlp.down_proj"):
    method = object.__new__(PrismaQuantCBLinearMethod)
    method.prefix = prefix
    method.is_fp4 = is_fp4
    method.k = FP4_K_BITS if is_fp4 else FP8_K_BITS
    method.n_sub = FP4_N_SUB if is_fp4 else FP8_N_SUB
    method.type_size = FP4_TS if is_fp4 else FP8_TS
    method.is_v2 = is_fp4
    method.has_static_fp4_activation = False
    return method


def test_row_parallel_split_superblock_refuses_with_structured_fields():
    """TP=2 down_proj with K=768: per-rank K=384 cuts superblock 1 in half.
    The refusal must carry qname, axis, group size and degree as FIELDS."""
    method = _bare_method(is_fp4=False)
    layer = _Layer()
    with pytest.raises(ShardGroupAlignmentError) as exc:
        method.create_weights(layer, 384, [N_FULL], 768, N_FULL,
                              torch.bfloat16, weight_loader=None)
    err = exc.value
    assert err.qname == method.prefix
    assert err.axis == "input"
    assert err.group_size == 256
    assert err.tp_degree == 2
    assert err.shard_size == 384
    # And no parameter was registered: the refusal precedes allocation.
    assert list(layer.named_parameters()) == []


def test_column_parallel_kernel_alignment_refuses_with_structured_fields():
    """TP=2 fp4 qkv-style merge whose second logical shard breaks the
    8-wide kernel row quantum."""
    method = _bare_method(
        is_fp4=True, prefix="model.layers.0.self_attn.qkv_proj")
    layer = _Layer()
    with pytest.raises(ShardGroupAlignmentError) as exc:
        method.create_weights(layer, 1024, [2048, 512, 500], 1024, 4096,
                              torch.bfloat16, weight_loader=None)
    err = exc.value
    assert err.qname == method.prefix
    assert err.axis == "output"
    assert err.group_size == 8
    assert err.tp_degree == 2
    assert err.shard_size == 500


def test_fp8_column_quantum_is_16_not_8():
    method = _bare_method(is_fp4=False, prefix=_GATE_PROJ)
    layer = _Layer()
    with pytest.raises(ShardGroupAlignmentError) as exc:
        method.create_weights(layer, 1024, [2048, 12], 1024, 4096,
                              torch.bfloat16, weight_loader=None)
    assert exc.value.axis == "output"
    assert exc.value.group_size == 16
    assert exc.value.tp_degree == 2


def test_tp1_artifact_error_keeps_its_exact_previous_form():
    """Byte-identical TP=1 behaviour: an out-of-SPEC in_features raises the
    PLAIN ValueError with the historical message, not the shard error."""
    method = _bare_method(is_fp4=False)
    layer = _Layer()
    with pytest.raises(ValueError) as exc:
        method.create_weights(layer, 100, [N_FULL], 100, N_FULL,
                              torch.bfloat16, weight_loader=None)
    assert type(exc.value) is ValueError
    assert str(exc.value) == (
        f"{method.prefix}: in_features 100 not a multiple of 256")


def test_tp1_misaligned_rows_still_refuse_only_at_load_finalization():
    """At TP=1 the N-alignment law stays enforced where it always lived — the
    post-load native attestation — so construction must NOT raise for an
    unsharded layer whose row count breaks the quantum."""
    method = _bare_method(is_fp4=True)
    layer = _Layer()
    method.create_weights(layer, 512, [12], 512, 12, torch.bfloat16,
                          weight_loader=None)
    assert layer.cb_qweight.shape == (12, (512 // 256) * FP4_TS)


def test_construction_records_both_axes_degrees():
    method_row = _bare_method(is_fp4=False)
    layer_row = _Layer()
    method_row.create_weights(layer_row, 512, [N_FULL], 1024, N_FULL,
                              torch.bfloat16, weight_loader=None)
    assert layer_row._cb_input_tp_degree == 2
    assert layer_row._cb_output_tp_degree == 1

    method_col = _bare_method(is_fp4=False, prefix=_GATE_PROJ)
    layer_col = _Layer()
    method_col.create_weights(layer_col, 1024, [N_FULL // 2], 1024, N_FULL,
                              torch.bfloat16, weight_loader=None)
    assert layer_col._cb_input_tp_degree == 1
    assert layer_col._cb_output_tp_degree == 2


# ---------------------------------------------------------------------------
# 4. Rank-local merged-role geometry (the GDN in_proj_qkvz class).
# ---------------------------------------------------------------------------

_ROLE_STEM = "model.layers.0.linear_attn"


def _role_method():
    method = object.__new__(PrismaQuantCBLinearMethod)
    method.prefix = f"{_ROLE_STEM}.in_proj_qkvz"
    return method


def test_merged_role_widths_scale_to_rank_local_at_tp2():
    method = _role_method()
    layer = types.SimpleNamespace(_cb_output_tp_degree=2)
    ckpt_rows = {
        f"{_ROLE_STEM}.in_proj_qkv.cb_qweight": 12288,
        f"{_ROLE_STEM}.in_proj_z.cb_qweight": 6144,
    }
    # logical_widths may be FINER than the roles (q,k,v,z chunks); only the
    # sums must agree after scaling.
    widths = [4096, 2048, 3072]
    local = method._rank_local_role_widths(
        layer, [f"{_ROLE_STEM}.in_proj_qkv", f"{_ROLE_STEM}.in_proj_z"],
        widths, ckpt_rows)
    assert local == [6144, 3072]


def test_merged_role_widths_are_identity_at_degree_one():
    method = _role_method()
    layer = types.SimpleNamespace(_cb_output_tp_degree=1)
    ckpt_rows = {f"{_ROLE_STEM}.in_proj_qkv.cb_qweight": 12288,
                 f"{_ROLE_STEM}.in_proj_z.cb_qweight": 6144}
    local = method._rank_local_role_widths(
        layer, [f"{_ROLE_STEM}.in_proj_qkv", f"{_ROLE_STEM}.in_proj_z"],
        [12288, 6144], ckpt_rows)
    assert local == [12288, 6144]


def test_merged_role_uneven_across_ranks_refuses_structured():
    method = _role_method()
    layer = types.SimpleNamespace(_cb_output_tp_degree=2)
    ckpt_rows = {f"{_ROLE_STEM}.in_proj_qkv.cb_qweight": 12289,
                 f"{_ROLE_STEM}.in_proj_z.cb_qweight": 6144}
    with pytest.raises(ShardGroupAlignmentError) as exc:
        method._rank_local_role_widths(
            layer,
            [f"{_ROLE_STEM}.in_proj_qkv", f"{_ROLE_STEM}.in_proj_z"],
            [6145, 3072], ckpt_rows)
    err = exc.value
    assert err.axis == "output"
    assert err.group_size == 2
    assert err.tp_degree == 2
    assert err.shard_size == 12289
    assert "in_proj_qkv" in err.qname


def test_merged_role_width_sum_mismatch_still_asserts_loudly():
    method = _role_method()
    layer = types.SimpleNamespace(_cb_output_tp_degree=2)
    ckpt_rows = {f"{_ROLE_STEM}.in_proj_qkv.cb_qweight": 100,
                 f"{_ROLE_STEM}.in_proj_z.cb_qweight": 50}
    with pytest.raises(AssertionError, match="logical width sum"):
        method._rank_local_role_widths(
            layer,
            [f"{_ROLE_STEM}.in_proj_qkv", f"{_ROLE_STEM}.in_proj_z"],
            [80], ckpt_rows)


# ---------------------------------------------------------------------------
# 5. The refusal lattice at get_quant_method.
# ---------------------------------------------------------------------------

_L0 = "model.layers.0"


def _artifact_config(extra_groups=None, extra=None, targets=None):
    cfg = {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "g": {"format": "FP8_CB_K32",
                  "targets": list(targets or [
                      f"{_L0}.self_attn.q_proj",
                      f"{_L0}.self_attn.o_proj",
                      f"{_L0}.mlp.gate_proj",
                      f"{_L0}.mlp.up_proj",
                      f"{_L0}.mlp.down_proj"]),
                  "scheme": dict(_DENSE_SCHEME)},
        },
        "ignore": ["lm_head"],
    }
    cfg["config_groups"].update(extra_groups or {})
    cfg.update(extra or {})
    return cfg


def _resolved_config(**kwargs):
    cfg = PrismaQuantConfig.from_config(_artifact_config(**kwargs))
    cfg._ensure_resolved()
    return cfg


def test_dense_cb_constructs_at_live_tp2(monkeypatch):
    _world_size(monkeypatch, 2)
    cfg = _resolved_config()
    method = cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.down_proj")
    assert isinstance(method, PrismaQuantCBLinearMethod)


def test_dense_cb_fused_single_method_constructs_at_live_tp2(monkeypatch):
    _world_size(monkeypatch, 2)
    cfg = _resolved_config()
    method = cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.gate_up_proj")
    assert isinstance(method, PrismaQuantCBLinearMethod)


def test_delegated_stock_group_refuses_at_live_tp2(monkeypatch):
    _require_real_vllm()
    _world_size(monkeypatch, 2)
    cfg = _resolved_config()
    sentinel = object()

    class _CT:
        packed_modules_mapping = {}
        get_quant_method = staticmethod(lambda layer, prefix: sentinel)

    cfg.ct_config = _CT()
    with pytest.raises(ValueError, match="compressed-tensors.*TP=2"):
        cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.sometarget")


def test_unsharded_source_passthrough_unit_refuses_at_live_tp2(monkeypatch):
    """A passthrough format with NO sharded audit still refuses by name.

    The MXFP8 dense lane has no shard laws of its own, so the config gate is
    what keeps it at one rank.
    """
    _world_size(monkeypatch, 2)
    cfg = _resolved_config(extra={
        "source_passthrough": {
            "version": 1,
            "units": {f"{_L0}.attn.wq_a": "mxfp8_e4m3_e8m0_g32"},
        }})
    with pytest.raises(ValueError, match="source-passthrough.*mxfp8"):
        cfg.get_quant_method(_linear_layer(), f"{_L0}.attn.wq_a")


def test_fp8_source_passthrough_unit_dispatches_at_live_tp2(monkeypatch):
    """The lifted lane: dispatch no longer refuses; its own laws decide.

    The FP8-source W8A16 lane enforces structural shard laws inside
    ``create_weights``, so the config gate hands the unit through and the
    per-rank legality is decided where the shapes are known.  The downstream
    device/backend attestations are stubbed here: what is under test is the
    DISPATCH decision, not the audit that follows it.
    """
    _world_size(monkeypatch, 2)
    sentinel = object()
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda: (12, 1))
    monkeypatch.setattr(config_mod, "_build_passthrough_method",
                        lambda fmt, layer, prefix: sentinel)
    monkeypatch.setattr(config_mod, "require_native_passthrough_backend",
                        lambda **kwargs: None)
    cfg = _resolved_config(extra={
        "source_passthrough": {
            "version": 1,
            "units": {f"{_L0}.attn.wq_a": "fp8_e4m3_ue8m0_block128"},
        }})

    assert cfg.get_quant_method(
        _linear_layer(), f"{_L0}.attn.wq_a") is sentinel


def test_moe_expert_stack_refuses_at_live_tp2_even_in_dense_artifact(
        monkeypatch):
    _world_size(monkeypatch, 2)
    cfg = _resolved_config(extra_groups={
        "experts": {"format": "FP8_CB_K28", "scheme": dict(_DENSE_SCHEME),
                    "targets": [f"{_L0}.ffn.experts.gate_up_proj",
                                f"{_L0}.ffn.experts.down_proj"]}})
    experts = object.__new__(RoutedExperts)
    with pytest.raises(ValueError, match="expert stacks.*TP=2"):
        cfg.get_quant_method(experts, f"{_L0}.ffn.experts")


def test_quantized_embedding_unit_refuses_at_live_tp2(monkeypatch):
    _world_size(monkeypatch, 2)
    cfg = _resolved_config(extra={
        "quantized_embedding": {
            "version": 1, "units": {"model.embed_tokens": "nvfp4"}}})
    embed = object.__new__(VocabParallelEmbedding)
    with pytest.raises(ValueError, match="embedding units.*TP=2"):
        cfg.get_quant_method(embed, "model.embed_tokens")


def test_mixed_format_fused_projection_dispatches_at_live_tp2(monkeypatch):
    """The composite arm is admitted; each ROLE's own law gates it.

    It used to refuse by name here.  It no longer owns a law of its own: the
    composer derives the column degree from vLLM's constructor arguments and
    builds each carrier at that role's whole-tensor output size, so the CB and
    source-passthrough shard gates fire per role at ``create_weights``.  See
    ``tests/test_mixed_fused_tp_shard.py`` for the byte placement.
    """

    from gridbook.mixed_linear import MixedFusedLinearMethod

    _world_size(monkeypatch, 2)
    # As in the passthrough dispatch test above: the downstream device and
    # backend attestations are stubbed, because what is under test here is
    # the DISPATCH decision, not the audit that follows it.
    sentinel = object()
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda: (12, 1))
    monkeypatch.setattr(config_mod, "_build_passthrough_method",
                        lambda fmt, layer, prefix: sentinel)
    monkeypatch.setattr(config_mod, "require_native_passthrough_backend",
                        lambda **kwargs: None)
    # up_proj must belong to exactly ONE vocabulary: it is the passthrough
    # role here, so leave it out of the CB group.
    cfg = _resolved_config(
        targets=[f"{_L0}.self_attn.q_proj", f"{_L0}.self_attn.o_proj",
                 f"{_L0}.mlp.gate_proj", f"{_L0}.mlp.down_proj"],
        extra={
            "source_passthrough": {
                "version": 1,
                "units": {f"{_L0}.mlp.up_proj": "fp8_e4m3_ue8m0_block128"},
            }})
    cfg._has_mixed_fused_loader = types.MethodType(lambda self: True, cfg)
    # gate_proj stays CB, up_proj passthrough -> the composite arm.
    method = cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.gate_up_proj")
    assert isinstance(method, MixedFusedLinearMethod)

    # Without the top-level router the composite still refuses, because its
    # loads cannot be addressed by the fused name; that refusal is about the
    # loader ABI, not about the world size.
    cfg._has_mixed_fused_loader = types.MethodType(lambda self: False, cfg)
    with pytest.raises(RuntimeError, match="mixed-fused loader ABI 1"):
        cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.gate_up_proj")


def test_ignored_target_stays_on_native_bf16_sharding_at_live_tp2(
        monkeypatch):
    """Ignored Linears are vLLM-native surface; they must NOT refuse — every
    shipped artifact carries some (sub-quantum GDN scalers)."""
    _require_real_vllm()
    _world_size(monkeypatch, 2)
    cfg = PrismaQuantConfig.from_config(_artifact_config(extra={
        "ignore": ["lm_head", f"{_L0}.linear_attn.in_proj_a"]}))
    cfg._ensure_resolved()
    method = cfg.get_quant_method(_linear_layer(),
                                  f"{_L0}.linear_attn.in_proj_a")
    assert isinstance(method, UnquantizedLinearMethod)


def test_delegation_and_dense_construction_survive_at_live_tp1(monkeypatch):
    """TP=1 behaviour is unchanged: delegation still delegates, dense still
    constructs, and the world-size read is LATCHED after the first 1."""
    _require_real_vllm()
    calls = []

    def _live():
        calls.append(1)
        return 1

    monkeypatch.setattr(
        config_mod, "_initialized_tensor_parallel_world_size", _live)
    cfg = _resolved_config()
    sentinel = object()

    class _CT:
        packed_modules_mapping = {}
        get_quant_method = staticmethod(lambda layer, prefix: sentinel)

    cfg.ct_config = _CT()
    # An undeclared prefix reaches the delegated CT arm...
    assert cfg.get_quant_method(
        _linear_layer(), f"{_L0}.mlp.sometarget") is sentinel
    # ...while a declared CB target still constructs the dense method.
    assert isinstance(cfg.get_quant_method(
        _linear_layer(), f"{_L0}.self_attn.o_proj"),
        PrismaQuantCBLinearMethod)
    assert len(calls) >= 1
    before = len(calls)
    cfg.get_quant_method(_linear_layer(), f"{_L0}.mlp.up_proj")
    assert len(calls) == before           # latched: no re-read after TP=1


def test_uninitialized_parallelism_behaves_as_tp1(monkeypatch):
    """Offline/unit construction without a distributed group defers, exactly
    as the old gate documented."""
    _world_size(monkeypatch, None)
    cfg = _resolved_config()
    assert cfg._tensor_parallel_world_size() is None
    assert isinstance(cfg.get_quant_method(
        _linear_layer(), f"{_L0}.mlp.down_proj"), PrismaQuantCBLinearMethod)


def test_refusal_message_carries_surface_name_prefix_and_degree(monkeypatch):
    _world_size(monkeypatch, 3)
    cfg = _resolved_config(extra={
        "source_passthrough": {
            "version": 1,
            "units": {f"{_L0}.attn.wq_a": "mxfp8_e4m3_e8m0_g32"},
        }})
    with pytest.raises(ValueError) as exc:
        cfg.get_quant_method(_linear_layer(), f"{_L0}.attn.wq_a")
    message = str(exc.value)
    assert "source-passthrough unit format 'mxfp8_e4m3_e8m0_g32'" \
        in message
    assert f"'{_L0}.attn.wq_a'" in message
    assert "TP=3" in message
