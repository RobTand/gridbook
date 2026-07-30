"""The ``.pqcb`` sidecar is bound to its own declared ``cb_tables_sha256``.

Wrong codebook *values* are invisible at decode time — a ``k``-bit codeword
indexes a ``2^k``-row table, so every index is in range by construction and a
sidecar with the right shapes but the wrong numbers yields a correctly-shaped
tensor of structured garbage rather than an error. These tests pin the three
behaviours that close that hole:

  * declared digest matches            -> loads
  * declared digest does NOT match     -> ValueError (refuse to serve)
  * no digest declared at all          -> loads (backward compatibility)

CPU-only, and deliberately vLLM-free: ``gridbook.cb_digest`` imports no serving
symbols, so this file injects no ``vllm`` stubs into ``sys.modules`` and cannot
perturb any other test file sharing the process.
"""
import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from gridbook.cb_digest import (  # noqa: E402
    CB_DIGEST_META_KEY,
    SKIP_DIGEST_ENV,
    codebook_digest,
    load_codebooks,
    read_declared_digest,
)

_SUB0 = "cb_codebook.lattice.NVFP4_CB_K16.sub0"
_SUB1 = "cb_codebook.lattice.NVFP4_CB_K16.sub1"


def _tables():
    """Two tables of different dtype, deterministic."""
    g = torch.Generator().manual_seed(0)
    return {
        _SUB0: torch.randn(16, 8, generator=g),
        _SUB1: torch.randn(16, 8, generator=g).to(torch.float16),
    }


def _write(tmp_path, tables, digest):
    path = str(tmp_path / "cb_codebooks.pqcb")
    meta = {} if digest is None else {CB_DIGEST_META_KEY: digest}
    st.save_file(tables, path, metadata=meta)
    return path


# -- the digest construction (docs/SPEC.md 4.1) -----------------------------

def test_digest_is_insertion_order_independent():
    t = _tables()
    reversed_order = {k: t[k] for k in reversed(list(t))}
    assert list(reversed_order) != list(t)
    assert codebook_digest(reversed_order) == codebook_digest(t)


def test_digest_changes_on_value_dtype_shape_and_membership():
    base = codebook_digest(_tables())

    t = _tables()
    t[_SUB0] = t[_SUB0].clone()
    t[_SUB0][0, 0] += 1.0
    assert codebook_digest(t) != base, "value change must change the digest"

    t = _tables()
    t[_SUB0] = t[_SUB0].to(torch.float64)
    assert codebook_digest(t) != base, "dtype change must change the digest"

    t = _tables()
    t[_SUB0] = t[_SUB0].reshape(8, 16)
    assert codebook_digest(t) != base, "reshape must change the digest"

    t = _tables()
    t["cb_codebook.lattice.NVFP4_CB_K16.sub2"] = t[_SUB0].clone()
    assert codebook_digest(t) != base, "extra table must change the digest"

    t = _tables()
    t[f"{_SUB0}.renamed"] = t.pop(_SUB0)
    assert codebook_digest(t) != base, "rename must change the digest"


def test_digest_hashes_bfloat16_tables():
    """``Tensor.numpy()`` raises on bfloat16; the uint8 view must not."""
    assert len(codebook_digest({"cb": torch.randn(16, 8).to(torch.bfloat16)})) == 64


def test_digest_matches_raw_numpy_bytes_for_numpy_dtypes():
    """The uint8 view is byte-identical to ``Tensor.numpy().tobytes()``, so
    this cannot invalidate a digest an existing encoder already wrote."""
    import hashlib
    tables = _tables()
    h = hashlib.sha256()
    for n in sorted(tables):
        t = tables[n].cpu().contiguous()
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    assert h.hexdigest() == codebook_digest(tables)


# -- the binding at load ----------------------------------------------------

def test_matching_digest_loads(tmp_path):
    tables = _tables()
    declared = codebook_digest(tables)
    path = _write(tmp_path, tables, declared)
    assert read_declared_digest(path) == declared
    got = load_codebooks(path)
    assert sorted(got) == sorted(tables)
    for k in tables:
        assert torch.equal(got[k], tables[k])


def test_absent_digest_still_loads(tmp_path, capsys):
    """Backward compatibility: every artifact published before the binding
    existed carries no metadata and MUST keep loading."""
    tables = _tables()
    path = _write(tmp_path, tables, None)
    assert read_declared_digest(path) is None
    got = load_codebooks(path)
    assert sorted(got) == sorted(tables)
    assert CB_DIGEST_META_KEY in capsys.readouterr().err


def test_wrong_digest_is_refused(tmp_path):
    tables = _tables()
    path = _write(tmp_path, tables, "0" * 64)
    with pytest.raises(ValueError) as exc:
        load_codebooks(path)
    msg = str(exc.value)
    assert CB_DIGEST_META_KEY in msg
    assert codebook_digest(tables) in msg
    assert "0" * 64 in msg


def test_wrong_values_are_refused(tmp_path):
    """THE case that motivates all of this: names, shapes and dtypes are all
    correct, the declaration is honest, only the numbers are wrong. Nothing
    downstream of the load can tell — every codeword still indexes in range."""
    tables = _tables()
    declared = codebook_digest(tables)
    wrong = dict(tables)
    wrong[_SUB0] = wrong[_SUB0].clone()
    wrong[_SUB0][3, 3] = 42.0
    path = _write(tmp_path, wrong, declared)
    from safetensors.torch import load_file
    raw = load_file(path)                       # a plain load sees nothing amiss
    assert sorted(raw) == sorted(tables)
    assert all(raw[k].shape == tables[k].shape for k in tables)
    assert all(raw[k].dtype == tables[k].dtype for k in tables)
    with pytest.raises(ValueError):
        load_codebooks(path)


def test_missing_table_is_refused(tmp_path):
    tables = _tables()
    declared = codebook_digest(tables)
    with pytest.raises(ValueError):
        load_codebooks(_write(tmp_path, {_SUB0: tables[_SUB0]}, declared))


def test_skip_env_downgrades_to_warning(tmp_path, monkeypatch, capsys):
    tables = _tables()
    path = _write(tmp_path, tables, "0" * 64)
    monkeypatch.setenv(SKIP_DIGEST_ENV, "1")
    got = load_codebooks(path)
    assert sorted(got) == sorted(tables)
    err = capsys.readouterr().err
    assert SKIP_DIGEST_ENV in err and "WARNING" in err


def test_skip_env_only_honours_exactly_one(tmp_path, monkeypatch):
    path = _write(tmp_path, _tables(), "0" * 64)
    for val in ("0", "", "true", "yes"):
        monkeypatch.setenv(SKIP_DIGEST_ENV, val)
        with pytest.raises(ValueError):
            load_codebooks(path)


def test_module_needs_no_vllm():
    """Guards the property that lets this file stay stub-free."""
    import sys
    assert "gridbook.cb_digest" in sys.modules
    assert not any(m == "vllm" or m.startswith("vllm.") for m in sys.modules)
