"""nn.functional's distance functions: ``cosine_similarity`` and ``pairwise_distance``.

Re-exported by :mod:`lucid.nn.functional.activations`.
"""

import lucid as _l
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry, _autocast_suspended
from lucid._ops._reduce import _require_floating
from lucid._tensor.tensor import Tensor
from lucid.nn.functional.normalization import normalize


def cosine_similarity(
    x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8
) -> Tensor:
    r"""Cosine similarity between two tensors along a dimension.

    Measures the angle between two vectors irrespective of their
    magnitudes — the natural similarity metric for unit-norm embedding
    spaces (contrastive learning, retrieval, face verification).

    Parameters
    ----------
    x1, x2 : Tensor
        Tensors of the same shape.
    dim : int, optional
        Dimension along which the dot product / norms are taken.
        Default ``1``.
    eps : float, optional
        Lower bound on the denominator to avoid division by zero.
        Default ``1e-8``.

    Returns
    -------
    Tensor
        Similarity tensor whose shape matches the inputs with ``dim``
        reduced, values in :math:`[-1, 1]`.

    Notes
    -----
    .. math::

        \text{cosine\_sim}(x_1, x_2) =
        \frac{x_1 \cdot x_2}{\max(\|x_1\|_2 \, \|x_2\|_2,\, \epsilon)}

    For pairwise similarity matrices, use ``unsqueeze`` to broadcast the
    inputs into rank-3 tensors before calling this function.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import cosine_similarity
    >>> a = lucid.tensor([[1.0, 0.0]])
    >>> b = lucid.tensor([[1.0, 1.0]])
    >>> cosine_similarity(a, b, dim=1)
    tensor([0.7071])
    """
    for operand in (x1, x2):
        _require_floating(_unwrap(operand), "cosine_similarity")
    io = [_unwrap(t.to(_l.result_type(x1, x2))) for t in (x1, x2)]
    with _autocast_entry("cosine_similarity", *io) as io:
        x1n, x2n = (_unwrap(normalize(_wrap(t), p=2.0, dim=dim, eps=eps)) for t in io)
        return _wrap(_C_engine.sum(_C_engine.mul(x1n, x2n), [dim]))


def pairwise_distance(
    x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
) -> Tensor:
    r"""Element-wise :math:`L_p` distance between two equally-shaped tensors.

    Computes the per-row :math:`L_p` norm of the difference — the standard
    metric in metric-learning triplet and contrastive losses.

    Parameters
    ----------
    x1, x2 : Tensor
        Tensors of the same shape.  Distance is reduced over the **last**
        dimension.
    p : float, optional
        Order of the norm.  ``2`` for Euclidean, ``1`` for Manhattan.
        Default ``2.0``.
    eps : float, optional
        Small constant added to ``x1 - x2`` before the norm, as the
        reference does.  Default ``1e-6``.
    keepdim : bool, optional
        If ``True``, retain the reduced dimension with size ``1``.
        Default ``False``.

    Returns
    -------
    Tensor
        Distance tensor.

    Notes
    -----
    .. math::

        d_p(x_1, x_2) = \| x_1 - x_2 + \epsilon \|_p
        = \Big(\sum_i |x_{1,i} - x_{2,i} + \epsilon|^p\Big)^{1/p}

    The ``eps`` shift keeps coincident points off the norm's kink.  With
    ``eps=0`` they sit on it, and their gradient is 0 — the subgradient
    the reference picks — not ``nan``.  The shift used to be added to
    ``|x1 - x2|`` instead, which differs by up to ``2 * eps`` per entry
    where ``x1 < x2``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import pairwise_distance
    >>> a = lucid.tensor([[1.0, 2.0]])
    >>> b = lucid.tensor([[4.0, 6.0]])
    >>> pairwise_distance(a, b, p=2.0)
    tensor([5.])
    """
    # The shared vector norm, so a zero distance (``eps=0`` at coincident
    # points) has the 0 subgradient rather than a nan one, half inputs
    # reduce in float32, and ``p=0`` counts instead of dividing by zero.
    # No row: the reference's norm has none, where vector_norm's is FP32 on
    # Metal, so autocast is off for it.
    with _autocast_suspended(_unwrap(x1).device):
        return _l.linalg.vector_norm(x1 - x2 + eps, p, -1, keepdim)
