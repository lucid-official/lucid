"""Sorting, splitting and max / min do what their documentation says.

Each of these was documented one way and behaved another:

* ``sort`` / ``argsort`` / ``Tensor.topk`` advertised ``descending=`` and
  ``largest=``; the engine took neither, so every call with them raised.
* The CPU sort was ``std::sort``, not stable, although both docstrings
  promised stability (the Metal path was).  Past a few dozen elements,
  equal keys came back in arbitrary order.
* ``split`` and ``chunk`` refused a size that does not divide the
  dimension — ``split(x, 2)`` on a length of 3 raised instead of
  returning pieces of 2 and 1.
* ``max`` / ``min`` gave every position tied for the extremum the full
  gradient, so the shares summed to the number of ties.  The header said
  "ties share equally"; now they do, as the reference framework's
  ``amax`` does.
"""

import numpy as np
import pytest

import lucid
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu"] + (["metal"] if metal_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [7, 5000])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_sorting_is_stable_in_both_directions(device: str, n: int, dtype) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(n)
    a = rng.integers(0, 5, (3, n)).astype(dtype)  # many ties
    x = lucid.tensor(a, device=device)
    for dim in (1, -1):
        np.testing.assert_array_equal(
            lucid.argsort(x, dim=dim).numpy(), np.argsort(a, dim, kind="stable")
        )
        np.testing.assert_array_equal(
            lucid.argsort(x, dim=dim, descending=True).numpy(),
            np.argsort(-a, dim, kind="stable"),
        )
        np.testing.assert_array_equal(
            lucid.sort(x, dim=dim, descending=True).values.numpy(), -np.sort(-a, dim)
        )
        np.testing.assert_array_equal(
            lucid.sort(x, dim=dim, descending=True).indices.numpy(),
            np.argsort(-a, dim, kind="stable"),
        )
        np.testing.assert_array_equal(
            x.sort(dim=dim, descending=True).values.numpy(), -np.sort(-a, dim)
        )
        np.testing.assert_array_equal(
            x.sort(dim=dim, descending=True).indices.numpy(),
            np.argsort(-a, dim, kind="stable"),
        )


def test_nan_sorts_as_the_largest_value() -> None:
    x = lucid.tensor([1.0, float("nan"), 3.0, 2.0])
    assert lucid.argsort(x).tolist() == [0, 3, 2, 1]
    assert lucid.argsort(x, descending=True).tolist() == [1, 2, 3, 0]


def test_a_descending_sort_differentiates() -> None:
    x = lucid.tensor([3.0, 1.0, 2.0], requires_grad=True)
    y, idx = lucid.sort(x, dim=0, descending=True)
    assert idx.tolist() == [0, 2, 1]
    (y * lucid.tensor([1.0, 10.0, 100.0])).sum().backward()
    assert x.grad.tolist() == [1.0, 100.0, 10.0]


@pytest.mark.parametrize("device", DEVICES)
def test_topk_returns_the_smallest_on_request(device: str) -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, (4, 9)).astype(np.float32)
    x = lucid.tensor(a, device=device)
    want = np.argsort(a, axis=1, kind="stable")[:, :3]
    for values, indices in (
        lucid.topk(x, 3, dim=1, largest=False),
        x.topk(3, dim=1, largest=False),
    ):
        np.testing.assert_array_equal(indices.numpy(), want)
        np.testing.assert_array_equal(
            values.numpy(), np.take_along_axis(a, want, axis=1)
        )
    largest_values, _ = lucid.topk(x, 3, dim=1)
    assert np.all(np.diff(largest_values.numpy(), axis=1) <= 0)


@pytest.mark.parametrize("n", [0, 1, 3, 6, 7, 10])
def test_split_and_chunk_follow_the_reference_rule(n: int) -> None:
    a = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    x = lucid.tensor(a)
    for size in (1, 2, 3, 4):
        want = [len(a[i : i + size]) for i in range(0, n, size)] or [0]
        assert [t.shape[0] for t in lucid.split(x, size, 0)] == want
        assert [t.shape[0] for t in x.split(size, dim=0)] == want
    for count in (1, 2, 3, 4, 5):
        step = -(-n // count)
        want = [min(step, n - i) for i in range(0, n, step)] if n else [0] * count
        assert [t.shape[0] for t in lucid.chunk(x, count, dim=0)] == want
        assert [t.shape[0] for t in x.chunk(count, dim=0)] == want
        if n:
            joined = np.concatenate([t.numpy() for t in lucid.chunk(x, count, 0)])
            np.testing.assert_array_equal(joined, a)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["max", "min"])
def test_ties_share_the_gradient(device: str, op: str) -> None:
    extreme = 5.0 if op == "max" else -5.0
    a = np.array(
        [[extreme, extreme, 0.0, extreme], [1.0, extreme, 2.0, 0.5]], dtype=np.float32
    )
    reduce = getattr(lucid, op)

    x = lucid.tensor(a, device=device, requires_grad=True)
    reduce(x, dim=1).values.sum().backward()
    np.testing.assert_allclose(
        x.grad.numpy(), [[1 / 3, 1 / 3, 0, 1 / 3], [0, 1, 0, 0]], rtol=1e-6
    )

    x = lucid.tensor(a, device=device, requires_grad=True)
    reduce(x).backward()
    assert float(x.grad.sum().item()) == pytest.approx(1.0)

    # The graph-mode backward takes the same shares.
    x = lucid.tensor(a, device=device, requires_grad=True)
    (d,) = grad(reduce(x, dim=1).values.sum(), [x], create_graph=True)
    np.testing.assert_allclose(
        d.numpy(), [[1 / 3, 1 / 3, 0, 1 / 3], [0, 1, 0, 0]], rtol=1e-6
    )


def test_a_nan_slice_gets_no_gradient_rather_than_nan() -> None:
    x = lucid.tensor([[1.0, float("nan")], [2.0, 3.0]], requires_grad=True)
    lucid.max(x, dim=1).values.sum().backward()
    assert x.grad.tolist() == [[0.0, 0.0], [0.0, 1.0]]


def test_radix_sort_stability_signed_zero_nan_and_short_rows() -> None:
    """LCD-360: LSD radix sort preserves -0.0/+0.0 and NaN stability alongside parallel short rows."""
    a = np.zeros(300, dtype=np.float32)
    a[0] = -0.0
    a[1] = 0.0
    a[2] = float("nan")
    a[3] = float("inf")
    a[4] = float("-inf")
    a[5] = -0.0
    t = lucid.tensor(a)
    idx_asc = lucid.argsort(t).tolist()
    assert idx_asc[0] == 4  # -inf first
    assert idx_asc[-2:] == [3, 2]  # +inf then nan
    # All zeros keep their input relative order: 0, 1, 5, 6, 7, ...
    zero_positions = [i for i in idx_asc if i not in (2, 3, 4)]
    assert zero_positions == sorted(zero_positions)
    sorted_vals = lucid.sort(t).values.numpy()
    assert np.copysign(1.0, sorted_vals[1]) == -1.0
    assert np.copysign(1.0, sorted_vals[2]) == 1.0
    assert np.copysign(1.0, sorted_vals[3]) == -1.0

    idx_desc = lucid.argsort(t, descending=True).tolist()
    assert idx_desc[:2] == [2, 3]  # nan then +inf
    assert idx_desc[-1] == 4  # -inf last

    rng = np.random.default_rng(360)
    rows_i64 = rng.integers(-10, 10, size=(256, 64), dtype=np.int64)
    np.testing.assert_array_equal(
        lucid.argsort(lucid.tensor(rows_i64), dim=1).numpy(),
        np.argsort(rows_i64, axis=1, kind="stable"),
    )
