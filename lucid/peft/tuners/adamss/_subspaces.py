r"""The frozen split of an AdaMSS layer: the weight's leading directions, and the subspaces its rows fall into.

AdaMSS (Zheng et al., 2025, NeurIPS, section 3) reads the weight's rows, as the paper reads its
columns, as points of several low-dimensional subspaces.  With the weight :math:`W` joined to its
bias as one more column, :math:`\tilde W = [W \mid b]` of shape ``out x (in + 1)``, the split is

1. the truncated SVD :math:`\tilde W \approx U_r \Sigma_r V_r^\top` (Algorithm 1, steps 1-3), the
   reference's randomised one, drawn from ``random_seed``;
2. a clustering of the ``out`` rows of :math:`U_r` into :math:`K` subspaces (step 4; the reference
   fixes :math:`K` rather than estimating it, as the paper's Appendix C allows);
3. for each subspace, the :math:`r_k` leading right singular directions of its rows of :math:`U_r`
   (steps 6-8), the orthonormal start of that subspace's trained factor.

The adapter then trains, per subspace, :math:`A_k \in \mathbb R^{r_k \times r}` (starting at those
directions) and :math:`B_k \in \mathbb R^{n_k \times r_k}` (starting at zero), and the layer's update
is :math:`\Delta\tilde W = \Pi\,[B_1 A_1; \dots; B_K A_K]\,V_r^\top`, the blocks stacked in cluster
order and :math:`\Pi` putting each block's rows back where the weight had them.

What this module decides is a function of the weight and the config alone and is not in an adapter
file, so it must come out the same each time: it runs on the CPU in float32 whatever device the
weight is on.  The reference's decomposition is randomised, so on a weight whose rank exceeds ``r``
it only approximates the leading subspace, by a draw this library cannot reproduce; where ``r`` is
at least the rank (small layers, low-rank weights) the subspace, and so the split, is the same.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._spectral import _svd_lowrank
from lucid.peft.tuners.adamss._config import AdamssConfig
from lucid.peft.tuners.adamss._kmeans import _kmeans_labels

#: Power iterations of the randomised SVD: the reference's.
_NITER = 2


class _Split(NamedTuple):
    """What the SVD and the clustering give a layer: its projection, its subspaces and where they start."""

    #: ``V_r^T``, ``(r, in + 1)`` (or ``(r, in)`` without a bias): the frozen projection every subspace reads.
    projection: Tensor
    #: For each output row, its position among the subspaces' stacked rows, ``(out,)`` int64.
    gather: Tensor
    #: The rows of each subspace, in the weight's order.
    members: tuple[Tensor, ...]
    #: The start of each subspace's ``A_k``, ``(r_k, r)``, orthonormal rows.
    starts: tuple[Tensor, ...]


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Draw from the default generator started at ``seed``, and leave it where it was found."""
    saved = lucid.get_rng_state()
    lucid.manual_seed(seed)
    try:
        yield
    finally:
        lucid.set_rng_state(saved)


def _subspace_rank(sizes: Tensor, rows: int, config: AdamssConfig) -> int:
    """The rank ``r_k`` of a subspace of ``rows`` rows whose singular values are ``sizes``.

    The configured ``subspace_rank`` capped by the rows, or, with ``use_dynamic_rank``, by how many
    singular values of the rows' Gram matrix (the squares of ``sizes``) exceed ``svd_threshold`` of
    the largest; at least one, and no more than the rank ``r`` the projection has.
    """
    rank = min(rows, config.subspace_rank)
    if config.use_dynamic_rank:
        gram = sizes * sizes
        rank = min(int((gram > config.svd_threshold * gram[0]).sum().item()), rank)
    return max(1, min(rank, int(sizes.shape[0])))


def _split(joined: Tensor, config: AdamssConfig) -> _Split:
    """Split the rows of ``joined`` (the weight with its bias column, float32, on the CPU) into subspaces.

    Parameters
    ----------
    joined : Tensor
        ``(out, columns)``.
    config : AdamssConfig
        ``r``, ``num_subspaces``, ``subspace_rank``, ``use_dynamic_rank``, ``svd_threshold`` and
        ``random_seed``.

    Returns
    -------
    _Split
        The projection, the rows of each subspace, the permutation that restores the weight's row
        order, and each subspace's start.

    Raises
    ------
    ValueError
        The rows are too alike to fall into ``num_subspaces`` distinct clusters (a cluster is empty).
    """
    rows = joined.shape[0]
    rank = min(config.r, *joined.shape)
    with _seeded(config.random_seed):
        left, _, right = _svd_lowrank(joined, rank, _NITER)
    left, projection = left[:, :rank], right.mT[:rank].contiguous()
    clusters = min(config.num_subspaces, rows)
    labels = _kmeans_labels(left, clusters)
    members = tuple((labels == cluster).nonzero().reshape(-1) for cluster in range(clusters))
    if any(int(m.shape[0]) == 0 for m in members):
        raise ValueError(
            f"the {rows} rows of this weight fall into fewer than num_subspaces={clusters} "
            "distinct clusters (duplicate rows); lower `num_subspaces`"
        )
    starts: list[Tensor] = []
    for rows_k in members:
        _, sizes, directions = linalg.svd(left.index_select(0, rows_k), full_matrices=False)
        starts.append(directions[: _subspace_rank(sizes, int(rows_k.shape[0]), config)])
    stacked = lucid.cat(list(members))
    gather = stacked.argsort()
    return _Split(projection, gather, members, tuple(starts))
