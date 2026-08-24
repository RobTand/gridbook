"""Dense CB weights must be resident exactly ONCE (issue #1, 2026-07-25).

``process_weights_after_loading`` derives a right-padded copy of the packed
weight for the decode/expand kernels. Before the fix the registered
``cb_qweight`` parameter stayed live alongside it, so every dense CB Linear was
resident twice — 15.07 GiB of duplicated ``cb_qweight`` on the shipped
Qwen3.6-27B gridbook artifact (21.38 GiB of weights served as ~36.5 GiB).
Invisible on the 128 GB unified-memory reference box; fatal on a 32 GB card.

The parameter is now a narrow VIEW of the padded buffer, which is only legal
because the pad is 16 bytes wide: the fp8 CUTLASS prefill entries still read
``layer.cb_qweight.data`` and TORCH_CHECK ``stride(0) % 16 == 0``. Both halves
of that contract are pinned here.

CPU-only; vLLM symbols are stubbed when unavailable (same idiom as
``test_target_namespace_compat``). Required native-kernel attestation is stubbed
explicitly so no CUDA build is touched.
"""
import json
import struct
import sys
import types

import pytest

torch = pytest.importorskip("torch")

def _install_vllm_stubs():
    """Install the vLLM surface imported by Gridbook's dense CB runtime."""
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
        """Minimal vLLM parameter with the real ``.data`` behaviour."""

        def __new__(cls, data, **kw):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kw):
            pass

    parameter.ModelWeightParameter = _StubParam
    parameter.ChannelQuantScaleParameter = _StubParam
    parameter.PerTensorScaleParameter = _StubParam


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    """Import Gridbook against a private real-or-stubbed vLLM graph."""
    del isolated_gridbook_runtime_imports
    env = pytest.MonkeyPatch()
    try:
        try:
            from vllm.model_executor.parameter import ModelWeightParameter  # noqa: F401
            from vllm.model_executor.layers.linear import LinearMethodBase  # noqa: F401
        except Exception:
            for name in list(sys.modules):
                if name == "vllm" or name.startswith("vllm."):
                    sys.modules.pop(name, None)
            _install_vllm_stubs()

        from gridbook import codec as codec_module
        from gridbook.config import PrismaQuantConfig as config_class
        from gridbook import linear as linear_module
        from gridbook.linear import PrismaQuantCBLinearMethod as method_class

        globals()["codec"] = codec_module
        globals()["PrismaQuantConfig"] = config_class
        globals()["cb_linear"] = linear_module
        globals()["PrismaQuantCBLinearMethod"] = method_class
        yield
    finally:
        env.undo()


@pytest.fixture(autouse=True)
def _reset_fp4_fused_mode_cache():
    """Keep process-global dispatch policy independent between tests."""
    cb_linear._FP4_FUSED_MODE.clear()
    cb_linear._FP4_DENSE_SM_COUNTS.clear()
    yield
    cb_linear._FP4_FUSED_MODE.clear()
    cb_linear._FP4_DENSE_SM_COUNTS.clear()


def _write_checkpoint_header(directory, *, rows=7):
    directory.mkdir(parents=True, exist_ok=True)
    header = json.dumps({
        "model.layers.0.mlp.down_proj.cb_qweight": {
            "dtype": "U8",
            "shape": [rows, 16],
            "data_offsets": [0, rows * 16],
        },
    }, separators=(",", ":")).encode("utf-8")
    path = directory / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


@pytest.mark.parametrize("k_bits", [4, 8, 12, 16, 20, 24, *range(28, 49)])
def test_fp8_scheme_loader_accepts_v10_reader_domain(k_bits):
    from gridbook.runtime_contract import load_runtime_contract

    PrismaQuantConfig._validate_cb_format_scheme(
        {
            "grid": "fp8", "mode": "product", "k": k_bits,
            "n_sub": 4, "type_size": 4 * k_bits,
        },
        "model.layers.0.self_attn.q_proj",
        load_runtime_contract(),
    )


@pytest.mark.parametrize("k_bits", [3, 5, 25, 27, 49])
def test_fp8_scheme_loader_rejects_values_outside_v10_reader_domain(k_bits):
    from gridbook.runtime_contract import load_runtime_contract

    with pytest.raises(ValueError, match="outside the packaged reader domain"):
        PrismaQuantConfig._validate_cb_format_scheme(
            {
                "grid": "fp8", "mode": "product", "k": k_bits,
                "n_sub": 4, "type_size": 4 * k_bits,
            },
            "model.layers.0.self_attn.q_proj",
            load_runtime_contract(),
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("k", True, "k must be an integer"),
        ("n_sub", 2, "requires n_sub=4"),
        ("type_size", 17, r"requires type_size=4\*k=16"),
    ],
)
def test_fp8_scheme_loader_rejects_incoherent_physical_fields(
    field, value, message
):
    from gridbook.runtime_contract import load_runtime_contract

    scheme = {
        "grid": "fp8", "mode": "product", "k": 4,
        "n_sub": 4, "type_size": 16,
    }
    scheme[field] = value
    with pytest.raises(ValueError, match=message):
        PrismaQuantConfig._validate_cb_format_scheme(
            scheme, "model.layers.0.self_attn.q_proj", load_runtime_contract()
        )


@pytest.mark.parametrize("k_bits", range(1, 26))
def test_nvfp4_scheme_loader_accepts_complete_v11_public_domain(k_bits):
    from gridbook.runtime_contract import load_runtime_contract

    PrismaQuantConfig._validate_cb_format_scheme(
        {
            "grid": "fp4", "mode": "product", "k": k_bits,
            "n_sub": 2, "type_size": 4 * k_bits + 9,
            "scale_coding": {"kind": "two_tier"},
        },
        "model.layers.0.mlp.down_proj",
        load_runtime_contract(),
    )


@pytest.mark.parametrize("k_bits", [0, 26, 32, 33, 48])
def test_nvfp4_scheme_loader_rejects_outside_v11_reader_domain(k_bits):
    from gridbook.runtime_contract import load_runtime_contract

    with pytest.raises(ValueError, match="outside the packaged reader domain"):
        PrismaQuantConfig._validate_cb_format_scheme(
            {
                "grid": "fp4", "mode": "product", "k": k_bits,
                "n_sub": 2, "type_size": 4 * k_bits + 9,
                "scale_coding": "two_tier",
            },
            "model.layers.0.mlp.down_proj",
            load_runtime_contract(),
        )


@pytest.mark.parametrize(
    "scheme,message",
    [
        ({"grid": "fp4", "mode": "product", "k": 1, "n_sub": 1,
          "type_size": 13, "scale_coding": "two_tier"},
         "requires n_sub=2"),
        ({"grid": "fp4", "mode": "product", "k": 25, "n_sub": 2,
          "type_size": 108, "scale_coding": "two_tier"},
         r"requires type_size=4\*k\+9=109"),
        ({"grid": "fp4", "mode": "product", "k": 16, "n_sub": 2,
          "type_size": 80, "scale_coding": "unknown"},
         "scale coding must be 'v1' or 'two_tier'"),
    ],
)
def test_nvfp4_scheme_loader_rejects_incoherent_physical_fields(
    scheme, message
):
    from gridbook.runtime_contract import load_runtime_contract

    with pytest.raises(ValueError, match=message):
        PrismaQuantConfig._validate_cb_format_scheme(
            scheme, "model.layers.0.mlp.down_proj", load_runtime_contract()
        )


def _checkpoint_header_method(source):
    quant_config = types.SimpleNamespace(_get_sidecar_source=lambda: source)
    method = object.__new__(PrismaQuantCBLinearMethod)
    method.quant_config = quant_config
    return method, quant_config


def test_checkpoint_header_snapshot_uses_cached_artifact_revision(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    _write_checkpoint_header(snapshot, rows=11)
    revision = "c" * 40
    method, _quant_config = _checkpoint_header_method(
        ("owner/model", revision)
    )

    import huggingface_hub
    calls = []

    def fake_snapshot_download(repo_id, *, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download
    )
    expected = {"model.layers.0.mlp.down_proj.cb_qweight": 11}
    assert method._ckpt_cb_rows() == expected
    assert method._ckpt_cb_rows() == expected
    assert calls == [("owner/model", revision, ["*.safetensors"])]


def test_checkpoint_header_local_source_never_calls_hub(tmp_path, monkeypatch):
    model_dir = tmp_path / "local-model"
    _write_checkpoint_header(model_dir, rows=13)
    method, _quant_config = _checkpoint_header_method(
        (str(model_dir), None)
    )

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *_args, **_kwargs: pytest.fail(
            "local checkpoint headers must not use the Hub"
        ),
    )
    assert method._ckpt_cb_rows() == {
        "model.layers.0.mlp.down_proj.cb_qweight": 13
    }


def test_dense_fp4_fused_prefill_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4", raising=False)
    assert cb_linear._fp4_fused_mode() == ""


@pytest.mark.parametrize(
    "value", [
        "1", "midm", "static_lsq", "static_lsq_midm",
        "rowwise", "rowwise_midm",
    ]
)
def test_dense_fp4_fused_prefill_explicit_modes(monkeypatch, value):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", value)
    assert cb_linear._fp4_fused_mode() == value


@pytest.mark.parametrize("value", ["yes", "MIDM", "128"])
def test_dense_fp4_fused_prefill_rejects_unknown_modes(monkeypatch, value):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", value)
    with pytest.raises(ValueError, match="invalid PRISMAQUANT_CB_FUSED_FP4"):
        cb_linear._fp4_fused_mode()


def test_dense_fp4_fused_prefill_cannot_change_mid_process(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", "1")
    assert cb_linear._fp4_fused_mode() == "1"
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", "midm")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        cb_linear._fp4_fused_mode()


@pytest.mark.parametrize(
    "M,N,sm_count,expected",
    [
        (255, 6144, 48, 128),       # TileM=256 is never used below M=256.
        (256, 3968, 48, 128),       # 31 candidate CTAs: below 2/3 occupancy.
        (256, 3976, 48, 256),       # 32 candidate CTAs: exact legal crossover.
        (768, 1024, 48, 128),       # measured narrow long-K loss (24 CTAs).
        (769, 1024, 48, 256),       # ceil(M/256)=4 -> 32 CTAs.
        (256, 6144, 0, 128),        # unavailable device metadata fails closed.
    ],
)
def test_dense_fp4_tile_selector_shape_boundaries(M, N, sm_count, expected):
    assert cb_linear._fp4_dense_tile_m(M, N, sm_count) == expected


def test_dense_fp4_sm_count_is_cached_without_synchronizing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: calls.append(index) or types.SimpleNamespace(
            multi_processor_count=48),
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda *_args, **_kwargs: pytest.fail(
            "SM-count resolution must not synchronize CUDA"),
    )
    device = torch.device("cuda:3")
    assert cb_linear._fp4_dense_sm_count(device) == 48
    assert cb_linear._fp4_dense_sm_count(device) == 48
    assert calls == [3]


@pytest.mark.parametrize(
    "M,N,expected_tile",
    [(256, 6144, 256), (768, 1024, 128), (1024, 1024, 256)],
)
def test_dense_fp4_tile_route_reaches_binding_and_telemetry(
        monkeypatch, M, N, expected_tile):
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method._sub_table = [1.0] * 16
    method._fused_fp4_ok = lambda *_args, **_kwargs: True
    K = 256
    layer = types.SimpleNamespace(
        _cb_fp4_lut=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(N),
        _cb_qw_padded=torch.ones(N, 73, dtype=torch.uint8),
    )
    observed = []

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, _multiplier):
            return (torch.zeros(M, K // 2, dtype=torch.uint8),
                    torch.zeros(M * (K // 16), dtype=torch.uint8),
                    torch.ones(M, dtype=torch.float32))

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(*args):
            observed.append((args[-2], args[-1]))
            return torch.zeros(M, N, dtype=torch.bfloat16)

    from gridbook import cuda_ext
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    monkeypatch.setattr(cb_linear, "_fp4_dense_sm_count", lambda _device: 48)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    out = method._try_fused_fp4(layer, x, N, K, M, rowwise=True)
    assert out.shape == (M, N)
    assert observed == [(None, expected_tile)]
    assert layer._cb_fp4_fused_tile_m == expected_tile
    assert layer._cb_fp4_fused_sm_count == 48
    assert layer._cb_fp4_fused_tile_candidate_ctas == (
        ((M + 255) // 256) * ((N + 127) // 128)
    )


@pytest.mark.parametrize("value", ["rowwise", "rowwise_midm"])
def test_dense_rowwise_modes_reach_only_the_rowwise_activation_family(
    monkeypatch, value
):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", value)
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    observed = []

    def attempt(layer, x, n, k, m, *, rowwise=False):
        observed.append(rowwise)
        return torch.zeros(m, n, dtype=x.dtype)

    method._try_fused_fp4 = attempt
    layer = types.SimpleNamespace(_cb_N=8, _cb_K=256)
    out = method._apply_inline(
        layer, torch.zeros(32, 256, dtype=torch.bfloat16)
    )
    assert out.shape == (32, 8)
    assert observed == [True]


@pytest.mark.parametrize("value", ["static_lsq", "static_lsq_midm"])
def test_dense_static_lsq_modes_reach_only_the_lsq_activation_family(
    monkeypatch, value
):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", value)
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    observed = []

    def attempt(
        layer, x, n, k, m, *, rowwise=False, static_lsq=False,
    ):
        observed.append((rowwise, static_lsq))
        return torch.zeros(m, n, dtype=x.dtype)

    method._try_fused_fp4 = attempt
    layer = types.SimpleNamespace(_cb_N=8, _cb_K=256)
    out = method._apply_inline(
        layer, torch.zeros(32, 256, dtype=torch.bfloat16)
    )
    assert out.shape == (32, 8)
    assert observed == [(False, True)]


@pytest.mark.parametrize(
    "k,n_sub,expected",
    [(24, 2, True), (20, 1, False), (21, 1, False)],
    ids=["product-k24-16k", "signed-s20-family-removed",
         "signed-s21-family-removed"],
)
def test_dense_fused_fp4_selector_enforces_lut_smem_limit(
    monkeypatch, k, n_sub, expected
):
    """The Python selector must reject a non-product n_sub before JIT.

    Product K24 sits exactly on the 16-KiB boundary and stays eligible; any
    n_sub other than 2 is ineligible outright now that the signed family is
    removed (its S21 over-carve scenario died with it). This drives the real
    dense eligibility method on CPU.
    """

    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    method.is_v2 = True
    method.has_static_fp4_activation = True
    method.k = k
    method.n_sub = n_sub
    method.type_size = 4 * k + 9
    layer = types.SimpleNamespace(
        _cb_N=8,
        _cb_row_offset=torch.zeros(8, dtype=torch.int32),
        _cb_fp4_input_global_scale=torch.ones(1, dtype=torch.float32),
    )

    from gridbook import cuda_ext

    class FusedExt:
        cb_fused_fp4_prefill_mm_scaled = object()

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: FusedExt())
    assert method._fused_fp4_ok(layer, 256) is expected


def test_dense_static_and_rowwise_eligibility_are_isolated(monkeypatch):
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    method.is_v2 = True
    method.has_static_fp4_activation = False
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    layer = types.SimpleNamespace(
        _cb_N=8,
        _cb_row_offset=torch.zeros(8, dtype=torch.int32),
    )

    from gridbook import cuda_ext

    class FusedExt:
        cb_fused_fp4_prefill_mm_scaled = object()
        cb_nvfp4_quantize_rows = object()
        cb_nvfp4_quantize_static_lsq = object()

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: FusedExt())
    # A legacy artifact is rejected by the unchanged static family even when
    # every kernel symbol exists, but the explicit rowwise family may run.
    assert method._fused_fp4_ok(layer, 256) is False
    assert method._fused_fp4_ok(layer, 256, rowwise=True) is True
    assert method._fused_fp4_ok(layer, 256, static_lsq=True) is False


def test_dense_rowwise_eligibility_requires_quantizer_and_gemm(monkeypatch):
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    method.is_v2 = True
    method.has_static_fp4_activation = True
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    layer = types.SimpleNamespace(
        _cb_N=8,
        _cb_row_offset=torch.zeros(8, dtype=torch.int32),
        _cb_fp4_input_global_scale=torch.tensor(2.0),
        _cb_fp4_input_global_scale_f32=2.0,
    )

    from gridbook import cuda_ext

    class StaticOnlyExt:
        cb_fused_fp4_prefill_mm_scaled = object()

    monkeypatch.setattr(
        cuda_ext, "get_fused_fp4_ext", lambda: StaticOnlyExt()
    )
    assert method._fused_fp4_ok(layer, 256) is True
    assert method._fused_fp4_ok(layer, 256, rowwise=True) is False
    assert method._fused_fp4_ok(layer, 256, static_lsq=True) is False


def test_dense_static_lsq_eligibility_requires_contract_and_matching_symbol(
    monkeypatch,
):
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.prefix = "test.dense"
    method.is_fp4 = True
    method.is_v2 = True
    method.has_static_fp4_activation = True
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    layer = types.SimpleNamespace(
        _cb_N=8,
        _cb_row_offset=torch.zeros(8, dtype=torch.int32),
        _cb_fp4_input_global_scale=torch.tensor(2.0),
        _cb_fp4_input_global_scale_f32=2.0,
    )

    from gridbook import cuda_ext

    class LsqExt:
        cb_fused_fp4_prefill_mm_scaled = object()
        cb_nvfp4_quantize_static_lsq = object()

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: LsqExt())
    assert method._fused_fp4_ok(layer, 256, static_lsq=True) is True

    unstamped_layer = types.SimpleNamespace(
        _cb_N=8,
        _cb_row_offset=torch.zeros(8, dtype=torch.int32),
        _cb_fp4_input_global_scale=torch.tensor(2.0),
    )
    assert method._fused_fp4_ok(
        unstamped_layer, 256, static_lsq=True
    ) is False

    legacy = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    legacy.prefix = "test.legacy_dense"
    legacy.is_fp4 = True
    legacy.is_v2 = True
    legacy.has_static_fp4_activation = False
    legacy.k = 16
    legacy.n_sub = 2
    legacy.type_size = 73
    legacy_layer = types.SimpleNamespace(
        _cb_N=8, _cb_row_offset=torch.zeros(8, dtype=torch.int32)
    )
    assert legacy._fused_fp4_ok(
        legacy_layer, 256, static_lsq=True
    ) is False


@pytest.fixture(autouse=True)
def _single_process_tp(monkeypatch):
    """Pin tensor-parallel rank/size to a single process for the duration of
    each test.

    Real vLLM's ``BasevLLMParameter.__init__`` reads the TP rank and world size
    from the process-global TP group, which only exists inside a launched
    engine — so constructing a ``ModelWeightParameter`` outside one dies with
    "tensor model parallel group is not initialized".

    This is what lets the file run against the REAL parameter class wherever
    vLLM is installed (the GPU container), instead of only against the stub
    above. That distinction is the whole point: the contract under test is a
    property of ``ModelWeightParameter.data`` — rebinding it to a narrow view
    must release the original storage — so verifying it only against a
    ``torch.nn.Parameter`` stub would verify nothing about the code that ships.
    No-op when vLLM is absent and the stub is in force.
    """
    par = sys.modules.get("vllm.model_executor.parameter")
    if par is None or not hasattr(par, "get_tensor_model_parallel_rank"):
        return
    monkeypatch.setattr(par, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(par, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def _attest_native_loader_without_building_cuda(monkeypatch):
    """Unit tests exercise load-time layout on CPU, not the JIT itself.

    Production now attests the required main extension during model load. Stub
    only that attestation here; fail-closed loader behavior has dedicated tests
    in ``test_fail_loud.py`` and CUDA extension tests.
    """
    from gridbook import cuda_ext
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
    monkeypatch.setattr(
        cb_linear, "require_native_fp8_cutlass", lambda operation: None)
    monkeypatch.setattr(
        cb_linear, "require_native_fp4_quant", lambda operation: None)


# The real shipped fp8 rungs. type_size == 4*k, all 16-byte multiples — the
# property the padded-stride invariant rests on.
_FP8_RUNGS = [(28, 112), (32, 128), (36, 144), (40, 160), (44, 176), (48, 192)]


# ---------------------------------------------------------------------------
# 1. pad invariant: >= 8 bytes of read slack AND a 16-byte-multiple row stride
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,type_size", _FP8_RUNGS)
@pytest.mark.parametrize("K", [256, 1024, 2048, 5120])
def test_pad_qweight_keeps_stride_16_aligned_and_gives_slack(k, type_size, K):
    assert type_size == 4 * k, "fp8 rung table drifted from type_size = 4k"
    row_bytes = (K // codec.SUPERBLOCK) * type_size
    assert row_bytes % 16 == 0, "unpadded fp8 row stride must already be 16B"
    qw = torch.zeros(8, row_bytes, dtype=torch.uint8)
    padded = codec.pad_qweight(qw)
    assert padded.shape == (8, row_bytes + codec.PAD_BYTES)
    assert padded.stride(1) == 1
    assert padded.stride(0) % 16 == 0, (
        "the fp8 CUTLASS prefill entries TORCH_CHECK stride(0) % 16 == 0")
    assert padded.stride(0) - row_bytes >= 8, (
        "the 8-byte codeword window reads past the last packed byte")
    assert padded.stride(0) >= (K // 256) * 4 * k       # the kernel's own check
    assert torch.equal(padded[:, :row_bytes], qw)
    assert int(padded[:, row_bytes:].max()) == 0        # pad is zeroed


def test_pad_width_is_a_16_multiple():
    """A pad that is not itself a 16-byte multiple cannot preserve a
    16-byte-aligned row stride — that was exactly the old +8's failure."""
    assert codec.PAD_BYTES % 16 == 0 and codec.PAD_BYTES >= 8


# ---------------------------------------------------------------------------
# 2. one storage: cb_qweight is a view of _cb_qw_padded, not a second copy
# ---------------------------------------------------------------------------

_K, _N = 512, 64
_SCHEME = {"grid": "fp8", "mode": "product", "k": 44, "n_sub": 4,
           "type_size": 176, "group_size": 0, "vec_dim": 8,
           "codebook_group": "mlp", "codebook_source": "learned",
           "codebook_ref": ["cb.a", "cb.b", "cb.c", "cb.d"]}
_TARGET = "model.layers.0.mlp.down_proj"


class _Layer(torch.nn.Module):
    """Stands in for vLLM's ``LinearBase``: only parameter registration and
    attribute assignment are exercised by process_weights_after_loading."""


def _loaded_layer():
    """A CB Linear taken through create_weights -> (weight load) ->
    process_weights_after_loading, entirely on CPU."""
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {"g": {"format": "FP8_CB_K44",
                                "targets": [_TARGET],
                                "scheme": dict(_SCHEME)}},
        "ignore": ["lm_head"],
    })
    cfg._ensure_resolved()
    # k=44, n_sub=4 -> ceil-first widths [11,11,11,11], sub_dim = 8//4 = 2.
    cfg.get_codebooks = lambda: {
        n: torch.zeros(2 ** 11, 2, dtype=torch.bfloat16)
        for n in _SCHEME["codebook_ref"]}

    method = PrismaQuantCBLinearMethod(cfg, dict(_SCHEME), _TARGET)
    layer = _Layer()
    method.create_weights(layer, _K, [_N], _K, _N, torch.bfloat16,
                          weight_loader=None)
    # "Load" the checkpoint tensors.
    layer.cb_qweight.data.copy_(torch.randint(
        0, 256, layer.cb_qweight.shape, dtype=torch.uint8))
    layer.weight_scale.data.copy_(torch.rand(_N))
    original = layer.cb_qweight.data
    method.process_weights_after_loading(layer)
    return layer, original


def test_cb_qweight_and_padded_share_one_storage():
    layer, _original = _loaded_layer()
    qw, padded = layer.cb_qweight.data, layer._cb_qw_padded
    assert qw.data_ptr() == padded.data_ptr()
    assert (qw.untyped_storage().data_ptr()
            == padded.untyped_storage().data_ptr())
    assert qw.untyped_storage().nbytes() == padded.untyped_storage().nbytes()


def test_layer_no_longer_references_the_original_storage():
    """The layer must hold NO handle on the pre-pad allocation — that handle is
    what kept the second 15.07 GiB alive for the whole serve. (The only other
    handle was a local in ``process_weights_after_loading``, dropped on return.)
    Compared by storage identity, not by value."""
    layer, original = _loaded_layer()
    assert (layer.cb_qweight.data.untyped_storage().data_ptr()
            != original.untyped_storage().data_ptr())
    assert (layer._cb_qw_padded.untyped_storage().data_ptr()
            != original.untyped_storage().data_ptr())


def test_repointed_cb_qweight_still_satisfies_the_fp8_kernel_checks():
    """``layer.cb_qweight.data`` is still passed to cb_fused_prefill_mm_scaled
    (mid-M on sm12x auto/require). Mirror its TORCH_CHECKs."""
    layer, _ = _loaded_layer()
    qw = layer.cb_qweight.data
    row_bytes = (_K // codec.SUPERBLOCK) * _SCHEME["type_size"]
    assert qw.dim() == 2 and qw.shape == (_N, row_bytes)
    assert qw.stride(1) == 1
    assert qw.stride(0) % 16 == 0
    assert qw.stride(0) >= (_K // 256) * 4 * _SCHEME["k"]
    assert qw.stride(0) == row_bytes + codec.PAD_BYTES     # padded, not copied


def test_repointed_cb_qweight_holds_the_loaded_bytes():
    """A view must not change what the kernels read."""
    layer, original = _loaded_layer()
    assert torch.equal(layer.cb_qweight.data, original)


def test_weight_scale_is_not_duplicated():
    """``_cb_scale`` aliases ``weight_scale.data`` (already fp32 and 1-D, so
    reshape(-1).to(float32) is a no-op view) — there is no second copy to free,
    which is why ``weight_scale`` is deliberately left registered."""
    layer, _ = _loaded_layer()
    assert (layer._cb_scale.untyped_storage().data_ptr()
            == layer.weight_scale.data.untyped_storage().data_ptr())


def test_public_apply_requires_opaque_registered_dispatch():
    """Production cannot expose the Python body to Inductor/Triton."""
    from gridbook.cuda_ext import NativeKernelUnavailableError

    layer, _ = _loaded_layer()
    method = PrismaQuantCBLinearMethod(
        types.SimpleNamespace(), dict(_SCHEME), _TARGET)
    del layer._cb_layer_id
    with pytest.raises(NativeKernelUnavailableError, match="not registered"):
        method.apply(layer, torch.zeros(1, _K, dtype=torch.bfloat16))


def test_public_apply_rejects_bias_without_native_operator():
    from gridbook.cuda_ext import NativeKernelUnavailableError

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    with pytest.raises(NativeKernelUnavailableError, match="biased CB Linear"):
        method.apply(
            layer, torch.zeros(1, _K, dtype=torch.bfloat16),
            torch.zeros(layer._cb_N, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# 3. fused roles: exact codebook references share one LUT block safely
# ---------------------------------------------------------------------------

_FUSED_K = 28
_FUSED_TYPE_SIZE = 4 * _FUSED_K
_FUSED_SCHEME = {
    "grid": "fp8", "mode": "product", "k": _FUSED_K, "n_sub": 4,
    "type_size": _FUSED_TYPE_SIZE, "group_size": 0, "vec_dim": 8,
    "codebook_group": "attn", "codebook_source": "learned",
}
_FUSED_PREFIX = "model.layers.0.self_attn.qkv_proj"
_FUSED_ROLES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
]
_REF_A = tuple(f"cb.a.sub{i}" for i in range(4))
_REF_B = tuple(f"cb.b.sub{i}" for i in range(4))
# Preserve deliberately uneven role boundaries while keeping the fused output
# dimension inside the native FP8 CUTLASS contract (N % 16 == 0).
_FUSED_WIDTHS = (3, 5, 8)
_FUSED_BLOCK_VALUES = 4 * (2 ** (_FUSED_K // 4)) * 2


def _fused_loaded_layer(role_refs):
    """Load a three-role fused dense layer without touching CUDA or vLLM TP."""
    target_scheme = {
        role: {**_FUSED_SCHEME, "codebook_ref": list(ref)}
        for role, ref in zip(_FUSED_ROLES, role_refs)
    }
    all_names = {name for ref in role_refs for name in ref}
    codebooks = {
        name: torch.full(
            (2 ** (_FUSED_K // 4), 2),
            # Deliberately identical contents across differently named refs:
            # deduplication is by provenance identity, never by current value.
            1.0,
            dtype=torch.bfloat16,
        )
        for name in all_names
    }

    class _FusedConfig:
        def __init__(self):
            self.target_scheme = target_scheme

        def shard_target_keys(self, prefix, *, unfused_fallback=False):
            assert prefix == _FUSED_PREFIX
            assert unfused_fallback
            return list(_FUSED_ROLES)

        def get_codebooks(self):
            return codebooks

    method = PrismaQuantCBLinearMethod(
        _FusedConfig(),
        {**_FUSED_SCHEME, "codebook_ref": list(role_refs[0])},
        _FUSED_PREFIX,
    )
    layer = _Layer()
    rows = sum(_FUSED_WIDTHS)
    row_bytes = (_K // codec.SUPERBLOCK) * _FUSED_TYPE_SIZE
    layer.cb_qweight = torch.nn.Parameter(
        torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.linspace(0.5, 1.5, rows), requires_grad=False)
    layer.logical_widths = list(_FUSED_WIDTHS)
    layer._cb_input_size = _K
    method.process_weights_after_loading(layer)
    return method, layer, codebooks


def _flat_for_ref(ref, codebooks):
    return codec.build_flat_codebook(
        [codebooks[name] for name in ref], _FUSED_PREFIX, "fp8")


def test_identical_fused_refs_deduplicate_block_and_offsets():
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    assert layer._cb_flat.numel() == _FUSED_BLOCK_VALUES
    assert torch.equal(
        layer._cb_row_offset,
        torch.zeros(sum(_FUSED_WIDTHS), dtype=torch.int32),
    )
    assert layer._cb_fp8_fused_lut_ok is True
    assert method._fused_fp8_lut_ok(layer) is True


def test_distinct_fused_refs_keep_distinct_blocks_and_offsets():
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_B, _REF_A])
    expected_offsets = torch.tensor(
        [0] * _FUSED_WIDTHS[0]
        + [_FUSED_BLOCK_VALUES] * _FUSED_WIDTHS[1]
        + [0] * _FUSED_WIDTHS[2],
        dtype=torch.int32,
    )
    assert layer._cb_flat.numel() == 2 * _FUSED_BLOCK_VALUES
    assert torch.equal(layer._cb_row_offset, expected_offsets)
    assert layer._cb_fp8_fused_lut_ok is False
    assert method._fused_fp8_lut_ok(layer) is False


def test_codebook_ref_order_is_part_of_dedup_identity():
    reversed_ref = tuple(reversed(_REF_A))
    method, layer, _ = _fused_loaded_layer([_REF_A, reversed_ref, _REF_A])
    assert layer._cb_flat.numel() == 2 * _FUSED_BLOCK_VALUES
    assert torch.equal(
        layer._cb_row_offset,
        torch.tensor(
            [0] * _FUSED_WIDTHS[0]
            + [_FUSED_BLOCK_VALUES] * _FUSED_WIDTHS[1]
            + [0] * _FUSED_WIDTHS[2],
            dtype=torch.int32,
        ),
    )
    assert method._fused_fp8_lut_ok(layer) is False


@pytest.mark.parametrize("role_refs", [
    (_REF_A, _REF_A, _REF_A),
    (_REF_A, _REF_B, _REF_A),
])
def test_dedup_preserves_every_rows_addressed_lut(role_refs):
    """The compact layout addresses the same LUT as the old concatenation."""
    _method, layer, codebooks = _fused_loaded_layer(role_refs)
    legacy_blocks = [_flat_for_ref(ref, codebooks) for ref in role_refs]
    legacy_flat = torch.cat(legacy_blocks)
    legacy_base = 0
    row = 0
    for width, block in zip(_FUSED_WIDTHS, legacy_blocks):
        new_bases = layer._cb_row_offset[row:row + width]
        assert bool((new_bases == new_bases[0]).all())
        new_base = int(new_bases[0])
        assert torch.equal(
            layer._cb_flat[new_base:new_base + block.numel()],
            legacy_flat[legacy_base:legacy_base + block.numel()],
        )
        legacy_base += block.numel()
        row += width


def _mock_fp8_ops(monkeypatch, N):
    monkeypatch.setattr(
        cb_linear, "native_fp8_quant",
        lambda x: (x, torch.ones(x.shape[0], 1, dtype=torch.float32)))
    monkeypatch.setattr(
        cb_linear, "native_cutlass_scaled_mm",
        lambda xq, wt, sa, ws, dtype: torch.full(
            (xq.shape[0], N), 7.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("M", [9, 16, 32])
def test_same_ref_fused_roles_enter_native_fused_kernel(monkeypatch, M):
    """FP8-CB M=9..128 reaches fused CUTLASS, never legacy cb_gemm."""
    from gridbook import cuda_ext
    from gridbook import ops as cb_ops

    # Set BEFORE the load: the flag is latched process-stable now, so a value
    # first observed at load cannot be changed afterwards — which is the point
    # of the latch, and is what this test was implicitly relying on not being
    # true.
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    from gridbook import lane_select
    monkeypatch.setattr(
        lane_select, "device_capability", lambda _device=None: (12, 1))
    # Explicit mode is a true requirement at load. The full stand-in below is
    # installed for forward; load only needs a non-None attestation object.
    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: object())
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    _mock_fp8_ops(monkeypatch, N)
    calls = []

    class _FusedExt:
        # A faithful stand-in must answer the RUNG query (K1.2): dispatch now
        # asks the module which rungs it compiled instead of carrying a
        # literal ladder, and `cb_fused_kbits` is in the fused module's STRICT
        # symbol contract — a real module that loads always has it.
        @staticmethod
        def cb_fused_kbits():
            return list(codec.FP8_FUSED_KBITS)

        @staticmethod
        def cb_fused_prefill_mm_scaled(xq, qw, cb, sa, ws, n, k, k_bits):
            calls.append((qw, cb, n, k, k_bits))
            return torch.full((M, N), 11.0, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: _FusedExt())
    monkeypatch.setattr(
        cb_ops, "cb_expand_fp8",
        lambda *a, **kw: pytest.fail("eligible shared LUT fell back"),
    )
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert len(calls) == 1
    assert calls[0][1] is layer._cb_flat_fp8
    assert torch.equal(out, torch.full_like(out, 11.0))


@pytest.mark.parametrize("M", [9, 16])
def test_distinct_ref_roles_use_native_expand_before_fused_ext(monkeypatch, M):
    """An offset-bearing LUT is fused-ineligible and stays native-only."""
    from gridbook import cuda_ext
    from gridbook import ops as cb_ops

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_B, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    _mock_fp8_ops(monkeypatch, N)
    monkeypatch.setattr(method, "_cuda_gemv_ok", lambda: True)
    monkeypatch.setattr(
        cuda_ext, "get_fused_ext",
        lambda: pytest.fail("offset-unsafe fused extension was queried"),
    )
    fallback_calls = []

    def _expand(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return torch.zeros(N, K, dtype=torch.float32)

    monkeypatch.setattr(cb_ops, "cb_expand_fp8", _expand)
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert len(fallback_calls) == 1
    assert torch.equal(out, torch.full_like(out, 7.0))
    from gridbook.nvfp4_activation_contract import read_route
    route = read_route(layer)
    assert route["policy"] == "fp8_cb_expand_cutlass_w8a8"
    assert route["symbol"] == "cb_expand_fp8+_C.cutlass_scaled_mm"
    assert route["shape"] == f"FP8_CB_K{method.k}:M{M}:N{N}:K{K}"
    assert route["contract"] == "fp8_per_token_dynamic"
    assert route["state"] == "served"
    assert route["reason"] is None


@pytest.mark.parametrize("M", [9, 16])
def test_missing_fused_ext_falls_back_to_native_expand(monkeypatch, M):
    """The optional fused module may miss; the required CUDA expander may not."""
    from gridbook import cuda_ext
    from gridbook import ops as cb_ops

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    _mock_fp8_ops(monkeypatch, N)
    monkeypatch.setattr(method, "_cuda_gemv_ok", lambda: True)
    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: None)
    calls = []

    def _expand(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(N, K, dtype=torch.float32)

    monkeypatch.setattr(cb_ops, "cb_expand_fp8", _expand)
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert len(calls) == 1
    assert torch.equal(out, torch.full_like(out, 7.0))


@pytest.mark.parametrize("M", [9, 16])
def test_missing_required_cuda_ext_fails_closed(monkeypatch, M):
    """No fused kernel and no CUDA expander is fatal, never a Triton fallback."""
    from gridbook import cuda_ext
    from gridbook import ops as cb_ops

    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    _mock_fp8_ops(monkeypatch, N)
    monkeypatch.setattr(method, "_cuda_gemv_ok", lambda: False)
    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: None)
    monkeypatch.setattr(
        cb_ops, "cb_expand_fp8",
        lambda *a, **kw: pytest.fail("unavailable CUDA expander was called"),
    )
    with pytest.raises(RuntimeError, match="alternate fallback is forbidden"):
        method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    from gridbook.nvfp4_activation_contract import read_route
    route = read_route(layer)
    assert route["policy"] == "fp8_cb_expand_cutlass_w8a8"
    assert route["state"] == "error"
    assert route["reason"] == "native composition did not return"


@pytest.mark.parametrize("M", [1, 8])
def test_fp8_decode_uses_native_cuda_gemv(monkeypatch, M):
    """The lower FP8-CB boundary remains the existing native CUDA GEMV."""
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    calls = []
    monkeypatch.setattr(method, "_cuda_gemv_ok", lambda: True)

    def _gemv(x, *args):
        calls.append(x.shape)
        return torch.full((M, N), 13.0, dtype=torch.bfloat16)

    monkeypatch.setattr(sys.modules["gridbook.linear"], "cb_gemv_fp8", _gemv)
    out = method._apply_inline(layer, torch.zeros(M, K, dtype=torch.bfloat16))
    assert calls == [torch.Size([M, K])]
    assert torch.equal(out, torch.full_like(out, 13.0))
    from gridbook.nvfp4_activation_contract import read_route
    route = read_route(layer)
    assert route["policy"] == "fp8_cb_cuda_gemv"
    assert route["symbol"] == "cb_gemv_fp8"
    assert route["shape"] == f"FP8_CB_K{method.k}:M{M}:N{N}:K{K}"
    assert route["contract"] == "fp8_per_token_dynamic"
    assert route["state"] == "served"
    assert route["reason"] is None


def test_fp8_decode_route_retains_error_when_native_launch_raises(monkeypatch):
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    N, K = layer._cb_N, layer._cb_K
    monkeypatch.setattr(method, "_cuda_gemv_ok", lambda: True)
    monkeypatch.setattr(
        sys.modules["gridbook.linear"], "cb_gemv_fp8",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        method._apply_inline(layer, torch.zeros(1, K, dtype=torch.bfloat16))

    from gridbook.nvfp4_activation_contract import read_route
    route = read_route(layer)
    assert route["policy"] == "fp8_cb_cuda_gemv"
    assert route["symbol"] == "cb_gemv_fp8"
    assert route["shape"] == f"FP8_CB_K{method.k}:M1:N{N}:K{K}"
    assert route["state"] == "error"
    assert route["reason"] == "launch did not return"


def test_fused_lut_guard_rejects_uniform_nonzero_base():
    """The kernel starts at cb[0]; merely uniform offsets are insufficient."""
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    del layer._cb_fp8_fused_lut_ok
    layer._cb_row_offset.fill_(_FUSED_BLOCK_VALUES)
    assert method._fused_fp8_lut_ok(layer) is False


@pytest.mark.parametrize("attr,value", [
    ("n_sub", 2),
    ("type_size", _FUSED_TYPE_SIZE + 16),
])
def test_fused_lut_guard_rejects_incompatible_fp8_layout(attr, value):
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    setattr(method, attr, value)
    assert method._fused_fp8_lut_ok(layer) is False


# ---------------------------------------------------------------------------
# 4. native-FP4 fused roles: one unambiguous LUT identity per 128-row N tile
# ---------------------------------------------------------------------------

_FP4_K = 16
_FP4_TYPE_SIZE = 4 * _FP4_K + 9
_FP4_SCHEME = {
    "grid": "fp4", "mode": "product", "k": _FP4_K, "n_sub": 2,
    "type_size": _FP4_TYPE_SIZE, "group_size": 16, "vec_dim": 8,
    "codebook_group": "attn", "codebook_source": "learned",
    "scale_coding": {"kind": "two_tier"},
}
_FP4_REF_A = ("cb.fp4.a0", "cb.fp4.a1")
_FP4_REF_B = ("cb.fp4.b0", "cb.fp4.b1")
_FP4_BLOCK_VALUES = 2 * (2 ** (_FP4_K // 2)) * 4


def _fp4_fused_loaded_layer(role_refs, widths):
    target_scheme = {
        role: {**_FP4_SCHEME, "codebook_ref": list(ref)}
        for role, ref in zip(_FUSED_ROLES, role_refs)
    }
    codebooks = {}
    for ref in role_refs:
        # Differently named blocks deliberately decode to different E2M1 values
        # so the dispatch test below can prove both physical LUTs were built.
        value = 0.0 if ref == _FP4_REF_A else 1.0
        for name in ref:
            codebooks[name] = torch.full(
                (2 ** (_FP4_K // 2), 4), value, dtype=torch.bfloat16)

    class _FusedFp4Config:
        def __init__(self):
            self.target_scheme = target_scheme

        @staticmethod
        def shard_target_keys(prefix, *, unfused_fallback=False):
            assert prefix == _FUSED_PREFIX and unfused_fallback
            return list(_FUSED_ROLES)

        @staticmethod
        def get_codebooks():
            return codebooks

    method = PrismaQuantCBLinearMethod(
        _FusedFp4Config(),
        {**_FP4_SCHEME, "codebook_ref": list(role_refs[0])},
        _FUSED_PREFIX,
    )
    layer = _Layer()
    rows = sum(widths)
    row_bytes = (_K // codec.SUPERBLOCK) * _FP4_TYPE_SIZE
    layer.cb_qweight = torch.nn.Parameter(
        torch.randint(0, 256, (rows, row_bytes), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.logical_widths = list(widths)
    layer._cb_input_size = _K
    method.process_weights_after_loading(layer)
    # This helper isolates post-load multi-LUT routing. Production artifacts
    # reach it only after the activation-contract loader has attested and
    # installed the static native-FP4 scale; model that completed gate here.
    method.has_static_fp4_activation = True
    layer._cb_fp4_input_global_scale = torch.tensor([3.0])
    return method, layer


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("1", (False, False)),
        ("static_lsq", (False, True)),
        ("rowwise", (True, False)),
    ],
)
def test_explicit_dense_fp4_mode_is_attested_and_cached_at_model_load(
    monkeypatch, mode, expected,
):
    calls = []
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", mode)

    def eligible(self, layer, K, *, rowwise=False, static_lsq=False):
        calls.append((K, rowwise, static_lsq))
        return True

    monkeypatch.setattr(
        PrismaQuantCBLinearMethod, "_fused_fp4_ok", eligible
    )
    _method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_A, _FP4_REF_A), (128, 128, 128)
    )
    assert layer._cb_fused_fp4_mode == mode
    assert calls == [(_K, *expected)]


def test_explicit_dense_fp4_mode_fails_at_model_load_when_unavailable(
    monkeypatch,
):
    from gridbook.cuda_ext import NativeKernelUnavailableError

    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4", "rowwise")
    monkeypatch.setattr(
        PrismaQuantCBLinearMethod,
        "_fused_fp4_ok",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(
        NativeKernelUnavailableError,
        match="requested native fused dense FP4 mode 'rowwise'",
    ):
        _fp4_fused_loaded_layer(
            (_FP4_REF_A, _FP4_REF_A, _FP4_REF_A), (128, 128, 128)
        )


def test_default_dense_fp4_mode_does_not_resolve_optional_fused_extension(
    monkeypatch,
):
    from gridbook import cuda_ext

    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4", raising=False)
    monkeypatch.setattr(
        cuda_ext,
        "get_fused_fp4_ext",
        lambda: pytest.fail("default FP4 mode built optional fused extension"),
    )
    _method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_A, _FP4_REF_A), (128, 128, 128)
    )
    assert layer._cb_fused_fp4_mode == ""


def test_fp8_fused_midm_optout_skips_model_load_jit(monkeypatch):
    from gridbook import cuda_ext

    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "0")
    monkeypatch.setattr(
        cuda_ext,
        "get_fused_ext",
        lambda: pytest.fail("FP8 fused opt-out built optional extension"),
    )
    _loaded_layer()


def test_fp8_fused_midm_auto_on_sm89_never_loads_blackwell_extension(
        monkeypatch):
    """Ada AUTO reaches expand+CUTLASS; it never probes the sm12x JIT."""
    from gridbook import cuda_ext, lane_select
    from gridbook import ops as cb_ops

    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_MIDM", raising=False)
    monkeypatch.setattr(
        lane_select, "device_capability", lambda _device=None: (8, 9))
    monkeypatch.setattr(
        cuda_ext,
        "get_fused_ext",
        lambda: pytest.fail("SM89 AUTO queried the Blackwell-only extension"),
    )
    method, layer, _ = _fused_loaded_layer([_REF_A, _REF_A, _REF_A])
    assert layer._cb_fp8_fused_midm is False

    N, K = layer._cb_N, layer._cb_K
    _mock_fp8_ops(monkeypatch, N)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: pytest.fail(
            "Ada route selection/telemetry synchronized CUDA"
        ),
    )
    expands = []

    def _expand(*args, **kwargs):
        expands.append((args, kwargs))
        return torch.zeros(N, K, dtype=torch.float32)

    monkeypatch.setattr(cb_ops, "cb_expand_fp8", _expand)
    out = method._apply_inline(
        layer, torch.zeros(16, K, dtype=torch.bfloat16)
    )
    assert len(expands) == 1
    assert torch.equal(out, torch.full_like(out, 7.0))
    from gridbook.nvfp4_activation_contract import read_route
    route = read_route(layer)
    assert route["kind"] == "dense"
    assert route["policy"] == "fp8_cb_expand_cutlass_w8a8"
    assert route["symbol"] == "cb_expand_fp8+_C.cutlass_scaled_mm"
    assert route["tile_m"] == 0
    assert route["shape"] == f"FP8_CB_K{method.k}:M16:N{N}:K{K}"
    assert route["contract"] == "fp8_per_token_dynamic"
    assert route["state"] == "served"
    assert route["reason"] is None


def test_fp8_fused_midm_explicit_on_sm89_fails_before_jit(monkeypatch):
    """An explicit Blackwell-only request on Ada fails at model load."""
    from gridbook import cuda_ext, lane_select
    from gridbook.cuda_ext import NativeKernelUnavailableError

    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    monkeypatch.setattr(
        lane_select, "device_capability", lambda _device=None: (8, 9))
    monkeypatch.setattr(
        cuda_ext,
        "get_fused_ext",
        lambda: pytest.fail("explicit SM89 refusal reached the sm12x JIT"),
    )
    with pytest.raises(
        NativeKernelUnavailableError,
        match=(r"PRISMAQUANT_CB_FUSED_MIDM=1 requires.*"
               r"compute capability 12\.0 or 12\.1.*reports 8\.9"),
    ):
        _fused_loaded_layer([_REF_A, _REF_A, _REF_A])


def test_fp8_fused_midm_explicit_on_blackwell_fails_if_extension_misses(
        monkeypatch):
    """`1` is a requirement, not auto with a different spelling."""
    from gridbook import cuda_ext, lane_select
    from gridbook.cuda_ext import NativeKernelUnavailableError

    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    monkeypatch.setattr(
        lane_select, "device_capability", lambda _device=None: (12, 1))
    monkeypatch.setattr(cuda_ext, "get_fused_ext", lambda: None)
    with pytest.raises(
        NativeKernelUnavailableError,
        match=r"PRISMAQUANT_CB_FUSED_MIDM=1 requires.*did not load",
    ):
        _fused_loaded_layer([_REF_A, _REF_A, _REF_A])


def test_fp8_fused_midm_optout_cannot_enable_jit_after_model_load(
        monkeypatch):
    from gridbook import cuda_ext

    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "0")
    method, layer, _ = _fused_loaded_layer(
        [_REF_A, _REF_A, _REF_A])
    _mock_fp8_ops(monkeypatch, layer._cb_N)
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_MIDM", "1")
    monkeypatch.setattr(
        cuda_ext,
        "get_fused_ext",
        lambda: pytest.fail("mid-serve opt-in triggered a fused JIT"),
    )
    with pytest.raises(
        RuntimeError,
        match="PRISMAQUANT_CB_FUSED_MIDM changed after Gridbook dispatch"
    ):
        method._apply_inline(
            layer, torch.zeros(16, layer._cb_K, dtype=torch.bfloat16))


def test_fp8_native_quality_shape_rejects_unaligned_fused_output_at_load(
        monkeypatch):
    from gridbook.cuda_ext import NativeKernelUnavailableError

    monkeypatch.setattr(
        sys.modules[__name__], "_FUSED_WIDTHS", (3, 5, 4))
    with pytest.raises(
        NativeKernelUnavailableError,
        match="FP8 CUTLASS quality prefill requires N divisible by 16",
    ):
        _fused_loaded_layer([_REF_A, _REF_A, _REF_A])


def test_distinct_fp4_refs_emit_tile_identity_map_without_role_lut_copies():
    method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_B, _FP4_REF_A), (128, 256, 128))
    assert method.is_fp4
    assert layer._cb_fp4_fused_lut_ok is True
    assert layer._cb_fp4_lut_ranges == (
        (0, _FP4_BLOCK_VALUES),
        (_FP4_BLOCK_VALUES, _FP4_BLOCK_VALUES),
    )
    assert torch.equal(
        layer._cb_fp4_lut_tile_ids,
        torch.tensor([0, 1, 1, 0], dtype=torch.int32),
    )
    assert layer._cb_flat.numel() == 2 * _FP4_BLOCK_VALUES


def test_fp4_quality_expansion_routes_each_role_segment_to_its_lut(
        monkeypatch):
    method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_B, _FP4_REF_A), (128, 256, 128))
    calls = []

    def expand_segment(qw, cb, offsets, compose, nrows, K, *format_args):
        del offsets, compose, format_args
        calls.append((nrows, cb.data_ptr(), qw.shape[0]))
        return torch.full(
            (nrows, K), float(len(calls)), dtype=torch.bfloat16)

    monkeypatch.setattr(
        cb_linear, "expand_fp4_v2_to_weight", expand_segment)
    weight = method._expand_fp4_quality_weight(
        layer, layer._cb_N, layer._cb_K)

    a_start, _ = layer._cb_fp4_lut_ranges[0]
    b_start, _ = layer._cb_fp4_lut_ranges[1]
    a_ptr = layer._cb_flat.narrow(0, a_start, 1).data_ptr()
    b_ptr = layer._cb_flat.narrow(0, b_start, 1).data_ptr()
    assert layer._cb_fp4_quality_segments == (
        (0, 128, 0), (128, 256, 1), (384, 128, 0))
    assert calls == [(128, a_ptr, 128), (256, b_ptr, 256),
                     (128, a_ptr, 128)]
    assert weight.shape == (512, _K)
    assert torch.all(weight[:128] == 1)
    assert torch.all(weight[128:384] == 2)
    assert torch.all(weight[384:] == 3)


def test_fp4_expander_device_attestation_failure_is_a_load_error(monkeypatch):
    from gridbook import cuda_ext
    from gridbook.cuda_ext import NativeKernelUnavailableError

    def reject(_operation, **_kwargs):
        raise NativeKernelUnavailableError("unsupported FP4 expander device")

    monkeypatch.setattr(cuda_ext, "require_fp4_v2_expander", reject)
    with pytest.raises(
        NativeKernelUnavailableError, match="unsupported FP4 expander device"
    ):
        _fp4_fused_loaded_layer(
            (_FP4_REF_A, _FP4_REF_A, _FP4_REF_A), (128, 128, 128))


def test_fp4_tile_map_fails_closed_when_distinct_role_boundary_is_unaligned(
        monkeypatch):
    method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_B, _FP4_REF_A), (64, 128, 64))
    assert layer._cb_fp4_fused_lut_ok is False
    assert layer._cb_fp4_lut_tile_ids is None

    from gridbook import cuda_ext
    monkeypatch.setattr(
        cuda_ext, "get_fused_fp4_ext",
        lambda: pytest.fail("ambiguous tile map queried fused extension"))
    assert method._fused_fp4_ok(layer, _K) is False


def test_fp4_multilut_dispatch_builds_unique_blocks_and_passes_tile_ids(
        monkeypatch):
    method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_B, _FP4_REF_A), (128, 256, 128))
    M, N, K = 32, layer._cb_N, layer._cb_K
    def _quant(x, _global_scale):
        return (torch.zeros(x.shape[0], x.shape[1] // 2, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.uint8))

    monkeypatch.setattr(cb_linear, "native_fp4_quant", _quant)
    calls = []

    class _FusedExt:
        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(
                aq, sfa, packed, lut, compose, a_scales, b_scales,
                n, k, k_bits, n_sub, type_size, is_v2, lut_tile_ids,
                tile_m):
            calls.append((lut.clone(), lut_tile_ids.clone(), n, k, k_bits,
                          n_sub, type_size, is_v2, tile_m))
            return torch.full((aq.shape[0], n), 13.0, dtype=torch.bfloat16)

    from gridbook import cuda_ext
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: _FusedExt())
    out = method._try_fused_fp4(
        layer, torch.ones(M, K, dtype=torch.bfloat16), N, K, M)
    assert torch.equal(out, torch.full_like(out, 13.0))
    assert len(calls) == 1
    lut, tile_ids, *_ = calls[0]
    block_bytes = codec.fp4_value_lut_nbytes(_FP4_K, 2)
    assert lut.numel() == 2 * block_bytes
    assert not torch.equal(lut[:block_bytes], lut[block_bytes:])
    assert torch.equal(tile_ids, torch.tensor([0, 1, 1, 0], dtype=torch.int32))


def test_fp4_single_lut_dispatch_keeps_offset_free_binding(monkeypatch):
    method, layer = _fp4_fused_loaded_layer(
        (_FP4_REF_A, _FP4_REF_A, _FP4_REF_A), (128, 128, 128))
    M, N, K = 32, layer._cb_N, layer._cb_K
    monkeypatch.setattr(
        cb_linear,
        "native_fp4_quant",
        lambda x, _scale: (
            torch.zeros(x.shape[0], x.shape[1] // 2, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.uint8),
        ),
    )
    tile_maps = []

    class _FusedExt:
        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(*args):
            tile_maps.append(args[-2])
            assert args[-1] == 128
            return torch.zeros(M, N, dtype=torch.bfloat16)

    from gridbook import cuda_ext
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: _FusedExt())
    x = torch.ones(M, K, dtype=torch.bfloat16)
    assert method._try_fused_fp4(layer, x, N, K, M) is not None
    # Exercise the cached-LUT call as well: ``ranges`` must remain available and
    # the optional map must stay absent so the kernel preserves TMA/LUT overlap.
    assert method._try_fused_fp4(layer, x, N, K, M) is not None
    assert tile_maps == [None, None]
    assert layer._cb_fp4_lut.numel() == codec.fp4_value_lut_nbytes(_FP4_K, 2)
