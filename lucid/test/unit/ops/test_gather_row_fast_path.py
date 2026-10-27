"""CPU ``gather`` / ``index_select`` take a row-aligned fast path; it must agree.

When every non-gathered axis of the output is as wide as the input's,
the CPU kernel now reads ``input[o, index, i]`` directly, typed, rather
than walking coordinates with a memcpy per element (CHA-5).  The path
dispatches on the element width and the index dtype, and wraps negative
indices.  The general walk still handles narrower outputs and 16-byte
elements.  Every case is held to numpy's ``take`` / ``take_along_axis``.
"""

import numpy as np
import pytest

import lucid

BASE = np.random.default_rng(0).standard_normal((3, 5, 4))
DTYPES = {
    "float32": BASE.astype(np.float32),
    "float64": BASE,
    "float16": BASE.astype(np.float16),
    "int8": (BASE * 3).astype(np.int8),
    "int64": (BASE * 3).astype(np.int64),
    "bool": BASE > 0,
    "complex64": (BASE + 1j * BASE[::-1]).astype(np.complex64),
}


@pytest.mark.parametrize("index_dtype", [np.int64, np.int32])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("name", list(DTYPES))
def test_index_select_matches_take(name: str, dim: int, index_dtype: type) -> None:
    a = DTYPES[name]
    index = np.array([0, -1, 1, -1], dtype=index_dtype)
    got = lucid.index_select(lucid.tensor(a), dim, lucid.tensor(index)).numpy()
    np.testing.assert_array_equal(got, np.take(a, index, axis=dim))


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("name", list(DTYPES))
def test_gather_matches_take_along_axis(name: str, dim: int) -> None:
    a = DTYPES[name]
    rng = np.random.default_rng(dim)
    index = rng.integers(-a.shape[dim], a.shape[dim], size=a.shape)
    got = lucid.gather(lucid.tensor(a), dim, lucid.tensor(index)).numpy()
    np.testing.assert_array_equal(got, np.take_along_axis(a, index % a.shape[dim], axis=dim))


def test_a_narrower_output_still_takes_the_general_walk() -> None:
    index = np.random.default_rng(3).integers(0, 5, size=(2, 3, 4))
    got = lucid.gather(lucid.tensor(BASE), 1, lucid.tensor(index)).numpy()
    np.testing.assert_array_equal(got, np.take_along_axis(BASE[:2], index, axis=1))


@pytest.mark.parametrize("bad", [5, -6])
def test_an_index_out_of_range_is_refused(bad: int) -> None:
    with pytest.raises(Exception, match="out of range"):
        lucid.index_select(lucid.tensor(BASE), 1, lucid.tensor([0, bad]))


def test_repeated_rows_accumulate_their_gradient() -> None:
    x = lucid.tensor(BASE, requires_grad=True)
    index = [1, 1, 3, -1]
    weight = np.random.default_rng(4).standard_normal((3, 4, 4))
    (lucid.index_select(x, 1, lucid.tensor(index)) * lucid.tensor(weight)).sum().backward()
    want = np.zeros_like(BASE)
    np.add.at(want, (slice(None), np.array(index) % 5), weight)
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-12)
