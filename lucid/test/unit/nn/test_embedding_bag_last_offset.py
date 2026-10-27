"""``embedding_bag``'s sentinel offset and its per-sample weights (#86).

Two ways a bag came back wrong without an error:

* Under ``include_last_offset=True`` the final offset is a sentinel — the
  end of the last bag — so there is one bag fewer than offsets.  The engine
  counted it as a bag (an extra, empty output row) and ended every other
  bag at the end of the index buffer, so ``offsets=[0, 2, 5]`` summed all
  five rows into the first bag.  CPU, Metal and both backwards shared it.
* ``per_sample_weights`` was accepted and dropped: a weighted bag was a
  plain sum.  The reference supports it for ``mode="sum"`` and refuses it
  otherwise.

The oracle is the definition in numpy.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu"] + (["metal"] if metal_available() else [])
INDICES = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
TABLE = np.random.default_rng(0).standard_normal((10, 3))


def _oracle(
    off: list[int], mode: str, weights: np.ndarray | None = None, padding_idx: int | None = None
) -> np.ndarray:
    """Bag b is ``INDICES[off[b]:off[b + 1]]``; the last offset only ends a bag."""
    rows = []
    for start, end in zip(off[:-1], off[1:]):
        keep = [k for k in range(start, end) if INDICES[k] != padding_idx]
        vecs = TABLE[INDICES[keep]]
        if weights is not None:
            vecs = vecs * weights[keep, None]
        if not keep:
            rows.append(np.zeros(TABLE.shape[1]))
        elif mode == "sum":
            rows.append(vecs.sum(0))
        elif mode == "mean":
            rows.append(vecs.mean(0))
        else:
            rows.append(vecs.max(0))
    return np.stack(rows)


OFFSETS = [
    [0, 2, 8],  # the sentinel is the index count, as documented
    [0, 3, 3, 8],  # an empty bag in the middle
    [0, 2, 5],  # a sentinel short of the end: indices past it are in no bag
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("off", OFFSETS, ids=["at end", "empty bag", "short"])
def test_the_last_offset_ends_the_last_bag(off: list[int], mode: str, device: str) -> None:
    out = F.embedding_bag(
        lucid.tensor(INDICES).to(device),
        lucid.tensor(TABLE.astype(np.float32)).to(device),
        lucid.tensor(off).to(device),
        mode=mode,
        include_last_offset=True,
    )
    assert tuple(out.shape) == (len(off) - 1, TABLE.shape[1])
    np.testing.assert_allclose(out.numpy(), _oracle(off, mode), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_the_gradient_reaches_only_the_rows_a_bag_holds(device: str) -> None:
    off = [0, 2, 5]
    w = lucid.tensor(TABLE.astype(np.float32)).to(device).requires_grad_()
    F.embedding_bag(
        lucid.tensor(INDICES).to(device),
        w,
        lucid.tensor(off).to(device),
        mode="sum",
        include_last_offset=True,
    ).sum().backward()
    want = np.zeros_like(TABLE)
    for k in range(off[-1]):
        want[INDICES[k]] += 1.0
    np.testing.assert_allclose(w.grad.numpy(), want, atol=1e-6)


def test_the_last_offset_needs_an_offset() -> None:
    with pytest.raises(Exception, match="include_last_offset"):
        F.embedding_bag(
            lucid.tensor(INDICES),
            lucid.tensor(TABLE),
            lucid.tensor(np.zeros(0, dtype=np.int64)),
            mode="sum",
            include_last_offset=True,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("last", [False, True], ids=["offsets", "last offset"])
@pytest.mark.parametrize("padding_idx", [None, 1])
def test_per_sample_weights_scale_each_row(
    last: bool, padding_idx: int | None, device: str
) -> None:
    psw = np.random.default_rng(1).standard_normal(len(INDICES))
    off = [0, 3, 3, 8] if last else [0, 3, 3]
    out = F.embedding_bag(
        lucid.tensor(INDICES).to(device),
        lucid.tensor(TABLE.astype(np.float32)).to(device),
        lucid.tensor(off).to(device),
        mode="sum",
        per_sample_weights=lucid.tensor(psw.astype(np.float32)).to(device),
        include_last_offset=last,
        padding_idx=padding_idx,
    )
    want = _oracle(off if last else off + [len(INDICES)], "sum", psw, padding_idx)
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-5, atol=1e-6)


def test_per_sample_weights_differentiate_in_both_inputs() -> None:
    psw = np.random.default_rng(2).standard_normal(len(INDICES))
    off = [0, 3, 8]
    w = lucid.tensor(TABLE, requires_grad=True)
    p = lucid.tensor(psw, requires_grad=True)
    out = F.embedding_bag(
        lucid.tensor(INDICES),
        w,
        lucid.tensor(off),
        mode="sum",
        per_sample_weights=p,
        include_last_offset=True,
    )
    (out * out).sum().backward()
    bag_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
    g_out = 2 * out.detach().numpy()
    want_p = (TABLE[INDICES] * g_out[bag_of]).sum(1)
    want_w = np.zeros_like(TABLE)
    np.add.at(want_w, INDICES, psw[:, None] * g_out[bag_of])
    np.testing.assert_allclose(p.grad.numpy(), want_p, rtol=1e-12)
    np.testing.assert_allclose(w.grad.numpy(), want_w, rtol=1e-12)


def test_per_sample_weights_on_a_two_dimensional_input() -> None:
    x = INDICES.reshape(2, 4)
    psw = np.random.default_rng(3).standard_normal(x.shape)
    out = F.embedding_bag(
        lucid.tensor(x), lucid.tensor(TABLE), mode="sum", per_sample_weights=lucid.tensor(psw)
    )
    np.testing.assert_allclose(out.numpy(), (TABLE[x] * psw[..., None]).sum(1), rtol=1e-12)


def test_per_sample_weights_are_refused_outside_sum() -> None:
    psw = lucid.ones(len(INDICES))
    with pytest.raises(NotImplementedError, match="mode='sum'"):
        F.embedding_bag(
            lucid.tensor(INDICES),
            lucid.tensor(TABLE),
            lucid.tensor([0, 4]),
            mode="mean",
            per_sample_weights=psw,
        )
    with pytest.raises(ValueError, match="shape of the indices"):
        F.embedding_bag(
            lucid.tensor(INDICES),
            lucid.tensor(TABLE),
            lucid.tensor([0, 4]),
            mode="sum",
            per_sample_weights=lucid.ones(3),
        )
