"""``F.embedding_bag`` correctness — guards the CPU-kernel int64-index + max-init fix.

The oracle is the definition written out in numpy: split the indices into
bags, drop ``padding_idx``, and reduce the gathered rows.  It used to be the
reference framework, which a fast-tier install does not have, so the whole
module skipped there and the fix it guards went unchecked.
"""

import numpy as np
import pytest

import lucid
from lucid.nn.functional import embedding_bag


def _bags(idx: np.ndarray, off: np.ndarray | None) -> list[np.ndarray]:
    """A 2-D input is one bag per row; a 1-D one is cut at ``off``."""
    if idx.ndim == 2:
        return list(idx)
    assert off is not None
    ends = list(off[1:]) + [len(idx)]
    return [idx[start:end] for start, end in zip(off, ends)]


def _oracle(
    idx: np.ndarray,
    off: np.ndarray | None,
    W: np.ndarray,
    mode: str,
    padding_idx: int | None = None,
) -> np.ndarray:
    rows = []
    for bag in _bags(idx, off):
        kept = [int(i) for i in bag if i != padding_idx]
        if not kept:
            rows.append(np.zeros(W.shape[1], dtype=W.dtype))
            continue
        gathered = W[kept]
        reduce = {"sum": np.sum, "mean": np.mean, "max": np.max}[mode]
        rows.append(reduce(gathered, axis=0))
    return np.stack(rows)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("idx_dtype", [np.int64, np.int32])
def test_1d_offsets_matches_oracle(mode: str, idx_dtype: type) -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((12, 6)).astype(np.float32)
    idx = np.array([1, 2, 4, 5, 4, 3, 2, 0], dtype=idx_dtype)
    off = np.array([0, 3, 5], dtype=idx_dtype)
    got = embedding_bag(
        lucid.from_numpy(idx), lucid.from_numpy(W), offsets=lucid.from_numpy(off), mode=mode
    ).numpy()
    want = _oracle(idx, off, W, mode)
    assert np.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_2d_matches_oracle(mode: str) -> None:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((12, 6)).astype(np.float32)
    idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    got = embedding_bag(lucid.from_numpy(idx), lucid.from_numpy(W), mode=mode).numpy()
    want = _oracle(idx, None, W, mode)
    assert np.allclose(got, want, atol=1e-5)


def test_max_all_negative_bag() -> None:
    # The max-init bug: an all-negative bag was masked by the 0-seed → 0.
    W = -np.abs(np.random.default_rng(2).standard_normal((6, 4))).astype(np.float32)
    idx = np.array([0, 1, 2], dtype=np.int64)
    off = np.array([0], dtype=np.int64)
    got = embedding_bag(
        lucid.from_numpy(idx), lucid.from_numpy(W), offsets=lucid.from_numpy(off), mode="max"
    ).numpy()
    want = _oracle(idx, off, W, "max")
    assert (want < 0).all()
    assert np.allclose(got, want, atol=1e-5)


def test_padding_idx() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((10, 5)).astype(np.float32)
    idx = np.array([1, 2, 2, 3, 4], dtype=np.int64)
    off = np.array([0, 2], dtype=np.int64)
    got = embedding_bag(
        lucid.from_numpy(idx),
        lucid.from_numpy(W),
        offsets=lucid.from_numpy(off),
        mode="sum",
        padding_idx=2,
    ).numpy()
    want = _oracle(idx, off, W, "sum", padding_idx=2)
    assert np.allclose(got, want, atol=1e-5)


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_metal_matches_oracle(mode: str) -> None:
    # GPU path uses a segment one-hot (matmul / masked-max), not the CPU loop.
    rng = np.random.default_rng(4)
    W = rng.standard_normal((12, 6)).astype(np.float32)
    idx = np.array([1, 2, 4, 5, 4, 3, 2, 0], dtype=np.int64)
    off = np.array([0, 3, 5], dtype=np.int64)
    got = embedding_bag(
        lucid.from_numpy(idx).to("metal"),
        lucid.from_numpy(W).to("metal"),
        offsets=lucid.from_numpy(off).to("metal"),
        mode=mode,
    ).numpy()
    want = _oracle(idx, off, W, mode)
    assert np.allclose(got, want, atol=1e-4)
