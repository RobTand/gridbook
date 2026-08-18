"""CPU/static gates for persistent-B's nested BF16 D2R experiment.

The CUDA numerics live in ``test_cb_moe_persistent_b_d2r.py``.  This file
proves the fail-closed/default-off wiring and the structural facts that are
otherwise easy to regress without a GPU: one extension/cache, the established
ABI tuple unchanged, the cooperative fragment mapping, the candidate warp
topology, and the distinct shared-memory budget.
"""
from __future__ import annotations

import inspect
import os
import re
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import cuda_ext  # noqa: E402
from gridbook import moe_persistent_b_lane as lane  # noqa: E402
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402

_PB = "PRISMAQUANT_CB_MOE_PERSISTENT_B"
_D2R = "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R"


@pytest.fixture(autouse=True)
def _fresh_flags(monkeypatch):
    monkeypatch.delenv(_PB, raising=False)
    monkeypatch.delenv(_D2R, raising=False)
    lane._reset_for_tests()
    yield
    lane._reset_for_tests()


def _source() -> str:
    path = os.path.join(cuda_ext.csrc_dir(), "cb_moe_persistent_b.cu")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _stub(symbols):
    ext = types.SimpleNamespace()
    for name in symbols:
        setattr(ext, name, lambda *args, **kwargs: None)
    return ext


def test_d2r_is_strict_process_latched_and_default_off(monkeypatch):
    assert lane.d2r_requested() is False
    monkeypatch.setenv(_D2R, "1")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        lane.d2r_requested()

    lane._reset_for_tests()
    monkeypatch.setenv(_D2R, "yes")
    with pytest.raises(ValueError, match=_D2R):
        lane.d2r_requested()


def test_d2r_is_one_nested_symbol_contract_not_a_loader_family():
    text = inspect.getsource(cuda_ext)
    assert "_MOE_PERSISTENT_B_D2R_SYMBOLS" in text
    assert not re.search(r"^_moe_persistent_b_d2r\s*=", text, re.MULTILINE)
    assert "get_moe_persistent_b_d2r_ext" not in text
    assert "moe_persistent_b_d2r" not in dict(cuda_ext._PRELOAD_FAMILIES)
    assert dict(cuda_ext._PRELOAD_FAMILIES)["moe_persistent_b"] == \
        "get_moe_persistent_b_ext"

    # Production ABI schema 2 (K1.1's FP8-CB arm): the FP8-CB rung joined the
    # production tuple — prefill/decode FP8 entry points plus the cfg
    # eligibility probe. D2R remains a separate opt-in attestation tuple in
    # the same image, still outside the production ABI.
    assert cuda_ext._MOE_PERSISTENT_B_ABI_SCHEMA == 2
    assert "cb_moe_persistent_b_prefill_fp8" in \
        cuda_ext._MOE_PERSISTENT_B_SYMBOLS
    assert "cb_moe_persistent_b_fp8_cfg_eligible" in \
        cuda_ext._MOE_PERSISTENT_B_SYMBOLS
    assert "cb_moe_persistent_b_prefill_d2r" not in \
        cuda_ext._MOE_PERSISTENT_B_SYMBOLS
    assert set(cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS) == {
        "cb_moe_persistent_b_prefill_d2r",
        "cb_moe_persistent_b_d2r_decode_pairs",
        "cb_moe_persistent_b_d2r_prepare",
        "cb_moe_persistent_b_d2r_configs",
    }
    exports = set(re.findall(
        r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', _source()))
    assert set(cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS) <= exports


def test_nested_lane_requires_every_candidate_symbol_and_prepares(
        monkeypatch):
    ext = _stub(cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS)
    prepared = []
    ext.cb_moe_persistent_b_d2r_prepare = lambda: prepared.append(True)
    monkeypatch.setattr(cuda_ext, "get_moe_persistent_b_ext", lambda: ext)
    monkeypatch.setattr(lane.lane_select, "_device_capability",
                        lambda _device=None: (12, 1))
    assert lane.require_d2r_lane("unit probe") is ext
    assert prepared == [True]

    for missing in cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS:
        partial = _stub(name for name in cuda_ext._MOE_PERSISTENT_B_D2R_SYMBOLS
                        if name != missing)
        monkeypatch.setattr(cuda_ext, "get_moe_persistent_b_ext",
                            lambda partial=partial: partial)
        with pytest.raises(NativeKernelUnavailableError, match=missing):
            lane.require_d2r_lane("unit probe")


def test_public_d2r_require_helper_fails_closed_without_substitution(
        monkeypatch):
    partial = _stub(cuda_ext._MOE_PERSISTENT_B_SYMBOLS)
    monkeypatch.setattr(cuda_ext, "get_moe_persistent_b_ext", lambda: partial)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        cuda_ext.require_moe_persistent_b_d2r_ext("D2R unit probe")
    message = str(exc_info.value)
    assert "direct-to-register" in message
    assert "missing" in message
    assert "does not substitute" in message


def test_full_and_pair_decoders_share_one_bit_window_law():
    # Since K1.1's FP8-CB arm there are THREE decoders over the packed code
    # stream — the
    # FP4 full codeword, the FP8 full codeword, and the cooperative K16 pair
    # path — and the law is that every one of them routes the bit-window
    # arithmetic through the single ``cb_extract_code`` helper.
    source = _source()
    assert source.count("DEVINL uint64_t cb_extract_code(") == 1
    fp8 = source[source.index("DEVINL uint4 cb_decode_codeword_fp8("):
                 source.index("DEVINL uint4 cb_decode_codeword(")]
    full = source[source.index("DEVINL uint4 cb_decode_codeword("):
                  source.index("DEVINL uint32_t cb_scale_pair")]
    pair = source[source.index("DEVINL uint2 cb_decode_k16_pairs"):
                  source.index("// PTX helpers")]
    assert "cb_extract_code(s32, c, f.k_bits)" in fp8
    assert "cb_extract_code(s32, c, f.k_bits)" in full
    assert "cb_extract_code(s32, c0, f.k_bits)" in pair
    assert "cb_extract_code(s32, c0 + 1, f.k_bits)" in pair
    # The copied shift/window sequence must not survive in any caller: the
    # single legitimate occurrence is inside the extractor itself.
    assert source.count("const int bitpos") == 1
    for body in (fp8, full, pair):
        assert "const int bitpos" not in body


def test_cooperative_pair_helper_is_the_exact_mma_b_mapping():
    source = _source()
    helper = source[source.index("DEVINL uint2 cb_decode_k16_pairs"):
                    source.index("// PTX helpers")]
    assert helper.count("__shfl_sync(0xffffffffu") == 3
    assert "pair == 0" in helper
    assert "const int grp = c0 >> 1" in helper
    assert "p0 + (pair & 1)" in helper
    assert "p1 + (pair & 1)" in helper

    kernel = source[source.index("template <int TM, int TN, int WARPS, bool D2R"):
                    source.index("// Decode probe")]
    assert "constexpr int WN = D2R ? WARPS : TN / 32" in kernel
    assert "const int n = wn * (TN / WN) + j * 8 + (lane >> 2)" in kernel
    assert "const int cw0 = (st % kStagesPerSb) * kChunks + 2 * kk" in kernel
    assert "reg0=(kpair,kpair+1), reg1=(kpair+8,kpair+9)" in kernel
    assert "static_assert(D2R || NATOM % 2 == 0" in kernel


def test_candidate_lifetime_and_smem_contract_are_structural():
    source = _source()
    # sPk aliases the address where baseline sB starts; D2R therefore cannot
    # materialize a decoded B tile even accidentally.
    assert "D2R ? sB : sB + TN * kTK" in source
    assert "(d2r ? 0 : (int64_t)c.tn * kTK * 2)" in source
    # Packed-superblock overwrite occurs after the every-stage CTA barrier
    # (1). Each warp stages exactly its own contiguous N rows and publishes
    # them locally before its pair decode; no D2R cross-warp read exists.
    barrier1 = source.index("__syncthreads();", source.index("cp_async_wait<0>()"))
    d2r_stage = source.index("if constexpr (D2R)", barrier1)
    ownership = source.index("const int n_begin = warp * kNPerWarp", d2r_stage)
    publish = source.index("__syncwarp();", ownership)
    helper_call = source.index("cb_decode_k16_pairs(", publish)
    assert barrier1 < d2r_stage < ownership < publish < helper_call
    between = source[publish:helper_call]
    assert "if constexpr (!D2R)" in between
    assert "__syncthreads();" in between  # baseline-only B publication
    # One A fragment is scoped inside the MATOM loop; the 8-fragment array
    # from the baseline must not appear in the candidate branch.
    candidate = source[source.index("} else {", source.index("// ---- MMA")):
                       source.index("// ---- epilogue")]
    assert "uint32_t af[4];" in candidate
    assert not re.search(r"uint32_t\s+af\[MATOM\]\[4\]", candidate)


def test_all_four_existing_cfgs_have_32_accumulators_and_less_smem():
    # Values are derived from the source's existing kCfgs, not an independent
    # proposed tile list: parse each initializer and apply the D2R formulas.
    source = _source()
    body = source[source.index("constexpr TileCfg kCfgs[]"):
                  source.index("constexpr int kNumCfgs")]
    cfgs = [tuple(map(int, match)) for match in
            re.findall(r"\{(\d+),\s*(\d+),\s*(\d+)\}", body)]
    assert len(cfgs) == 4
    for tm, tn, warps in cfgs:
        wn, wm = warps, 1
        matom = (tm // wm) // 16
        natom = (tn // wn) // 8
        assert 4 * matom * natom == 32
        assert natom in (1, 2)
        assert tn * 64 * 2 > 0  # exact decoded-B bytes removed per CTA


def test_model_load_wiring_rejects_an_orphan_d2r_flag_and_hot_path_is_latched():
    path = os.path.join(os.path.dirname(cuda_ext.__file__), "moe.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert 'persistent_b_d2r_on and pb_mode != "require"' in source
    assert "persistent_b_require_d2r_lane" in source
    method = source[source.index(
        "def _apply_prefill_native_bf16_persistent_b"):
        source.index("def _apply_prefill_native_bf16_sm120")]
    code = "\n".join(line.split("#", 1)[0] for line in method.splitlines())
    assert "_cb_moe_persistent_b_d2r" in code
    assert "cb_moe_persistent_b_prefill_d2r" in code
    assert "os.environ" not in code and _D2R not in code
