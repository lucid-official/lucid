"""``embedding_bag`` had no backward at all.

``embedding_op`` wires one; ``embedding_bag_op`` built its output and
returned it, with no ``Backward`` class and no ``wire_autograd``.  So the
forward was right, the output carried no ``grad_fn``, and
``weight.grad`` stayed ``None``:

    m = nn.EmbeddingBag(8, 4)
    m(idx).sum().backward()
    m.weight.grad          # None

No error, no NaN — an ``nn.EmbeddingBag`` layer simply never trained, and
a loss containing one would go down on everything else while that layer
sat still.  Found by the module axis of the audit once it could construct
the class at all.

The gradient is a scatter back onto the rows each bag gathered: ``sum``
sends the bag's gradient to every row it touched, ``mean`` divides by the
bag size first, and ``max`` sends it only to the row that won each
column.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

WEIGHT = np.random.default_rng(0).standard_normal((8, 4))
INDICES = np.array([[0, 1, 2], [3, 4, 1]])
MODES = ["sum", "mean", "max"]


def _grad(mode: str, **kwargs) -> np.ndarray:
    w = lucid.tensor(WEIGHT.copy(), requires_grad=True)
    idx = lucid.tensor(INDICES, dtype=lucid.int32)
    F.embedding_bag(idx, w, mode=mode, **kwargs).sum().backward()
    assert w.grad is not None, "no gradient reached the weight"
    return np.asarray(w.grad.numpy())


# ── the gradient exists at all ────────────────────────────────────────────────


@pytest.mark.parametrize("mode", MODES)
def test_the_weight_receives_a_gradient(mode) -> None:
    assert np.abs(_grad(mode)).sum() > 0.0


def test_the_module_trains() -> None:
    """The shape the defect actually took: a layer that never learns."""
    module = nn.EmbeddingBag(num_embeddings=8, embedding_dim=4)
    module(lucid.tensor(INDICES, dtype=lucid.int32)).sum().backward()
    assert module.weight.grad is not None
    assert np.abs(np.asarray(module.weight.grad.numpy())).sum() > 0.0


def test_the_output_carries_a_grad_fn() -> None:
    w = lucid.tensor(WEIGHT.copy(), requires_grad=True)
    out = F.embedding_bag(lucid.tensor(INDICES, dtype=lucid.int32), w)
    assert out.requires_grad
    assert out._impl.grad_fn is not None


# ── and it is the right one ───────────────────────────────────────────────────


def test_sum_sends_the_bag_gradient_to_every_row_it_touched() -> None:
    """With ``loss = out.sum()`` each bag's gradient is all ones, so a row
    accumulates exactly once per appearance."""
    got = _grad("sum")
    counts = np.bincount(INDICES.ravel(), minlength=8)
    assert np.allclose(got, counts[:, None] * np.ones((8, 4)))


def test_mean_divides_by_the_bag_size() -> None:
    got = _grad("mean")
    expected = np.zeros((8, 4))
    for bag in INDICES:
        for index in bag:
            expected[index] += 1.0 / len(bag)
    assert np.allclose(got, expected)


def test_max_sends_it_only_to_the_winning_row() -> None:
    """One row per column per bag, so the total equals the number of
    columns times the number of bags."""
    got = _grad("max")
    assert np.isclose(got.sum(), INDICES.shape[0] * WEIGHT.shape[1])
    expected = np.zeros((8, 4))
    for bag in INDICES:
        for d in range(WEIGHT.shape[1]):
            winner = bag[int(np.argmax(WEIGHT[bag, d]))]
            expected[winner, d] += 1.0
    assert np.allclose(got, expected)


def test_padding_idx_receives_nothing() -> None:
    got = _grad("sum", padding_idx=1)
    assert np.abs(got[1]).sum() == 0.0
    assert np.abs(got).sum() > 0.0


@pytest.mark.parametrize("mode", MODES)
def test_one_dimensional_indices_with_offsets(mode) -> None:
    ids = np.array([1, 2, 4, 5, 4, 3, 2, 7])
    offsets = np.array([0, 4])
    w = lucid.tensor(WEIGHT.copy(), requires_grad=True)
    F.embedding_bag(
        lucid.tensor(ids, dtype=lucid.int32),
        w,
        offsets=lucid.tensor(offsets, dtype=lucid.int32),
        mode=mode,
    ).sum().backward()
    assert w.grad is not None
    assert np.abs(np.asarray(w.grad.numpy())).sum() > 0.0


def test_rows_no_bag_touched_stay_at_zero() -> None:
    got = _grad("sum")
    untouched = sorted(set(range(8)) - set(INDICES.ravel().tolist()))
    assert untouched, "the fixture must leave some rows out"
    for row in untouched:
        assert np.abs(got[row]).sum() == 0.0


# ── the forward is unchanged ──────────────────────────────────────────────────


@pytest.mark.parametrize("mode", MODES)
def test_the_forward_still_reduces_the_way_it_did(mode) -> None:
    out = np.asarray(
        F.embedding_bag(
            lucid.tensor(INDICES, dtype=lucid.int32), lucid.tensor(WEIGHT.copy()), mode=mode
        ).numpy()
    )
    reduce = {"sum": np.sum, "mean": np.mean, "max": np.max}[mode]
    expected = np.stack([reduce(WEIGHT[bag], axis=0) for bag in INDICES])
    assert np.allclose(out, expected)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("mode", MODES)
def test_the_two_devices_agree(mode) -> None:
    weight = WEIGHT.astype(np.float32)

    def grad_on(device: str) -> np.ndarray:
        w = lucid.tensor(weight.copy(), device=device, requires_grad=True)
        idx = lucid.tensor(INDICES, dtype=lucid.int32, device=device)
        F.embedding_bag(idx, w, mode=mode).sum().backward()
        return np.asarray(w.grad.numpy())

    assert np.allclose(grad_on("cpu"), grad_on("metal"), atol=1e-5)
