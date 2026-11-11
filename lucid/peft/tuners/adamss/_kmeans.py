r"""K-means of the rows of AdaMSS's left singular factor, in Lucid ops, drawn as the reference library draws it.

AdaMSS splits the rows of the weight into subspaces by clustering the rows of :math:`U_r` (the
:math:`r` leading left singular vectors).  The clustering is not in an adapter file, so it has to be
rebuilt from the weight, every time, to the same answer: the reference library runs scikit-learn's
``KMeans(init="random", n_init=1, max_iter=10, random_state=123456789)``, and a split that differs
from it reads a reference-written file as another function.  This module is that algorithm, with
no library behind it:

* the seeds are ``RandomState(123456789).choice(n, k, replace=False)`` of numpy's legacy generator —
  a Mersenne Twister seeded by ``init_genrand`` and read as :math:`53`-bit doubles — which the
  standard library's own twister reproduces once it is handed the same state (``_Stream``);
* the iterations are Lloyd's: assign each row to the nearest center by :math:`\|c\|^2 - 2\,x\cdot c`,
  average, relocate the centers of empty clusters onto the rows farthest from theirs, stop when the
  labels repeat or the centers move less than ``1e-4`` of the mean variance, at most ten times, and
  label once more against the last centers if it stopped on the tolerance.

Everything runs on the CPU in float32 whatever device the weight is on: the labels depend on
comparisons of distances, and the CPU is the one place they are the same on every machine and every
run.  Rows are centred first, as the library does, so the distances are those of the same floats.
"""

import bisect
import random
import struct
from typing import Final

import lucid
from lucid._tensor.tensor import Tensor

#: The reference's fixed settings: its generator's seed, Lloyd's iteration cap and its tolerance factor.
_SEED: Final = 123456789
_ITERATIONS: Final = 10
_TOLERANCE: Final = 1e-4

_WORDS: Final = 624


class _Stream:
    """numpy's legacy ``RandomState(seed)``, as far as ``choice`` reads it: a Mersenne Twister of 32-bit words.

    ``RandomState`` seeds the twister's 624 words with Knuth's ``init_genrand`` recurrence; the
    standard library's ``random.Random`` is the same twister, so setting its state to those words
    gives numpy's words.  A double takes two of them (the top 27 bits of the first, the top 26 of
    the second).
    """

    def __init__(self, seed: int) -> None:
        words = [seed & 0xFFFFFFFF]
        for position in range(1, _WORDS):
            last = words[-1]
            words.append((1812433253 * (last ^ (last >> 30)) + position) & 0xFFFFFFFF)
        self._twister = random.Random()
        self._twister.setstate((3, (*words, _WORDS), None))

    def _double(self) -> float:
        high = self._twister.getrandbits(32) >> 5
        low = self._twister.getrandbits(32) >> 6
        return (high * 67108864.0 + low) / 9007199254740992.0

    def distinct(self, population: int, count: int) -> list[int]:
        """``count`` different positions of ``range(population)``, drawn as ``RandomState.choice(..., replace=False, p=uniform)`` draws them.

        The library hands ``choice`` the weights ``1 / population`` rounded to float32; a position
        already taken has its weight zeroed, and a round of draws that lands twice on one position
        keeps the first and draws again for the rest.
        """
        weight = struct.unpack("f", struct.pack("f", 1.0 / population))[0]
        weights = [weight] * population
        found: list[int] = []
        while len(found) < count:
            draws = [self._double() for _ in range(count - len(found))]
            for position in found:
                weights[position] = 0.0
            total = 0.0
            cdf: list[float] = []
            for value in weights:
                total += value
                cdf.append(total)
            cdf = [value / total for value in cdf]
            fresh: list[int] = []
            for draw in draws:
                position = bisect.bisect_right(cdf, draw)
                if position not in fresh:
                    fresh.append(position)
            found.extend(fresh)
        return found


def _as_ints(tensor: Tensor) -> list[int]:
    """The elements of a one-dimensional integer tensor, as Python ints."""
    listed = tensor.tolist()
    assert isinstance(listed, list)
    return [value for value in listed if isinstance(value, int)]


def _nearest(points: Tensor, centers: Tensor) -> Tensor:
    """The index of the center each row is nearest to (the first, on a tie): the argmin of ``|c|^2 - 2 x.c``."""
    squared = (centers * centers).sum(dim=1)
    nearest: Tensor = (squared - 2 * (points @ centers.mT)).argmin(dim=1)
    return nearest


def _relocated(
    points: Tensor, centers: Tensor, labels: Tensor, sums: Tensor, counts: Tensor
) -> None:
    """Give each empty cluster the row farthest from its own center, in place: the library's repair of a cluster that lost every row.

    Nothing happens if every row sits on its center (more clusters than distinct rows).
    """
    empty = _as_ints((counts == 0).nonzero().reshape(-1))
    if not empty:
        return
    distances = ((points - centers.index_select(0, labels)) ** 2).sum(dim=1)
    if float(distances.max().item()) == 0.0:
        return
    farthest = _as_ints(distances.argsort(descending=True)[: len(empty)])
    owners = _as_ints(labels)
    for cluster, row in zip(empty, farthest, strict=True):
        sums[owners[row]] = sums[owners[row]] - points[row]
        sums[cluster] = points[row]
        counts[cluster] = 1.0
        counts[owners[row]] = counts[owners[row]] - 1.0


def _kmeans_labels(points: Tensor, clusters: int) -> Tensor:
    """The cluster of every row of ``points`` (``(n, d)``, float32, on the CPU), among ``clusters`` of them.

    Cluster ``j`` is the one grown from the ``j``-th seed row, so the labels are the library's own,
    not a permutation of them.

    Parameters
    ----------
    points : Tensor
        The rows to cluster.
    clusters : int
        How many clusters; at most ``n``.

    Returns
    -------
    Tensor
        ``(n,)`` int64 labels in ``[0, clusters)``.
    """
    device = points.device
    tolerance = _TOLERANCE * float(points.var(dim=0, unbiased=False).mean().item())
    centered = points - points.mean(dim=0)
    seeds = lucid.tensor(_Stream(_SEED).distinct(points.shape[0], clusters), device=device)
    centers = centered.index_select(0, seeds)
    members = lucid.arange(clusters, device=device).reshape(1, -1)
    previous = lucid.full((points.shape[0],), -1, dtype=lucid.int64, device=device)
    labels = previous
    exact = False
    for _ in range(_ITERATIONS):
        labels = _nearest(centered, centers)
        onehot = (labels.reshape(-1, 1) == members).to(centered.dtype)
        sums, counts = onehot.mT @ centered, onehot.sum(dim=0)
        _relocated(centered, centers, labels, sums, counts)
        updated = sums / lucid.where(counts > 0, counts, lucid.ones_like(counts)).reshape(-1, 1)
        shift = float(((updated - centers) ** 2).sum().item())
        centers = updated
        if bool((labels == previous).all().item()):
            exact = True
            break
        if shift <= tolerance:
            break
        previous = labels
    return labels if exact else _nearest(centered, centers)
