"""nn.functional's softmax family: the activations that normalise along a dim.

``softmax``, ``log_softmax`` and ``softmin``, and the Gumbel-softmax
relaxation with the straight-through estimator its ``hard`` mode uses.
Re-exported by :mod:`lucid.nn.functional.activations`.
"""

import operator
import warnings

import lucid as _l
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._reduce import _log_softmax_adapter, _softmax_adapter, _softmin_adapter
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike


def _get_softmax_dim(name: str, ndim: int, dim: int | None, stacklevel: int) -> int:
    if dim is not None:
        return operator.index(dim)
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return 0 if ndim in (0, 1, 3) else 1


def softmax(
    input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DTypeLike | None = None
) -> Tensor:
    r"""Apply the softmax function along a dimension.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape (the "logits").
    dim : int or None, optional
        Dimension along which softmax is computed.
    _stacklevel : int, optional
        Internal stack level for the implicit-dimension warning.
    dtype : DTypeLike or None, optional
        Target floating-point dtype to cast ``input`` to before softmax.

    Returns
    -------
    Tensor
        Same-shape tensor whose entries along ``dim`` sum to 1.

    Notes
    -----
    .. math::

        \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    The engine uses the log-sum-exp shift :math:`x_i \mapsto x_i - \max_j x_j`.
    For loss computation, prefer :func:`log_softmax` followed by :func:`nll_loss`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> p = softmax(logits, dim=1)
    >>> p
    tensor([[0.09003, 0.2447, 0.6652]])
    >>> p.sum(dim=1)
    tensor([1.])
    """
    return _wrap(
        _softmax_adapter(
            _unwrap(input), _get_softmax_dim("softmax", input.ndim, dim, _stacklevel), dtype
        )
    )


def log_softmax(
    input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DTypeLike | None = None
) -> Tensor:
    r"""Numerically stable log-softmax along a dimension.

    Parameters
    ----------
    input : Tensor
        Input logits of any shape.
    dim : int or None, optional
        Dimension along which the log-softmax is computed.
    _stacklevel : int, optional
        Internal stack level for the implicit-dimension warning.
    dtype : DTypeLike or None, optional
        Target floating-point dtype to cast ``input`` to before log-softmax.

    Returns
    -------
    Tensor
        Log-probabilities of the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{log\_softmax}(x)_i = x_i - \log\!\sum_j e^{x_j}
        = (x_i - m) - \log\!\sum_j e^{x_j - m}, \quad m = \max_j x_j

    Pairing ``log_softmax`` with :func:`nll_loss` is numerically equivalent
    to — and more stable than — ``softmax`` followed by :func:`cross_entropy`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import log_softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> log_softmax(logits, dim=1)
    tensor([[-2.408, -1.408, -0.4076]])
    """
    return _wrap(
        _log_softmax_adapter(
            _unwrap(input), _get_softmax_dim("log_softmax", input.ndim, dim, _stacklevel), dtype
        )
    )


def softmin(
    input: Tensor, dim: int | None = None, _stacklevel: int = 3, dtype: DTypeLike | None = None
) -> Tensor:
    r"""Softmin — softmax applied to the negation of the input.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape.
    dim : int or None, optional
        Dimension along which softmin is computed.
    _stacklevel : int, optional
        Internal stack level for the implicit-dimension warning.
    dtype : DTypeLike or None, optional
        Target floating-point dtype to cast ``input`` to before softmin.

    Returns
    -------
    Tensor
        Tensor of the same shape with values summing to ``1`` along ``dim``.

    Notes
    -----
    .. math::

        \text{softmin}(x)_i = \text{softmax}(-x)_i = \frac{e^{-x_i}}{\sum_j e^{-x_j}}

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softmin
    >>> x = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> softmin(x, dim=1)
    tensor([[0.6652, 0.2447, 0.09003]])
    """
    axis = _get_softmax_dim("softmin", input.ndim, dim, _stacklevel)
    return _wrap(_softmin_adapter(_unwrap(input), axis, dtype=dtype))


def straight_through(hard: Tensor, soft: Tensor) -> Tensor:
    r"""Straight-through estimator — ``hard`` forward, ``soft`` backward.

    Returns a tensor numerically equal to ``hard`` whose gradient is the
    gradient of ``soft``:

    .. math::

        \mathrm{ST}(h, s) = s + \mathrm{sg}\!\big[h - s\big],

    where :math:`\mathrm{sg}` is the stop-gradient operator.  The forward
    value is exactly :math:`h`, and :math:`\partial \mathrm{ST} /
    \partial s` is the identity, so a discrete or otherwise
    non-differentiable ``hard`` can sit in the middle of a network while
    the surrounding layers still train.

    Parameters
    ----------
    hard : Tensor
        The value the forward pass should carry — typically the output of
        a non-differentiable step such as a one-hot ``argmax`` or a
        nearest-codebook lookup.
    soft : Tensor
        The differentiable surrogate whose gradient is borrowed.  Must be
        broadcast-compatible with ``hard``; in practice the two have the
        same shape.

    Returns
    -------
    Tensor
        Equal to ``hard`` in value, differentiable with respect to
        whatever ``soft`` depends on.

    Notes
    -----
    Introduced by Bengio, Léonard, and Courville, *"Estimating or
    Propagating Gradients Through Stochastic Neurons for Conditional
    Computation"* (arXiv:1308.3432).  It is the mechanism behind
    :func:`gumbel_softmax` with ``hard=True`` and behind
    :class:`lucid.nn.VectorQuantizer`, which is why it lives here rather
    than being rewritten at each call site.

    The estimator is biased — the returned gradient is not the gradient of
    the discrete function, which has none — but it is low-variance and
    empirically trains where the true gradient does not exist.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> soft = lucid.tensor([0.3, 0.7], requires_grad=True)
    >>> hard = lucid.tensor([0.0, 1.0])
    >>> out = F.straight_through(hard, soft)
    >>> out.tolist()               # the forward value is ``hard``
    [0.0, 1.0]
    >>> out.sum().backward()
    >>> soft.grad.tolist()         # gradient passed through untouched
    [1.0, 1.0]
    """
    return soft + (hard - soft).detach()


def gumbel_softmax(
    logits: Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1
) -> Tensor:
    r"""Gumbel-Softmax — differentiable relaxation of categorical sampling.

    Draws Gumbel noise, adds it to the logits, and softmaxes by
    temperature :math:`\tau`.  Lets gradients flow through what would
    otherwise be a non-differentiable categorical sample — central to
    differentiable discrete-action RL, VQ-VAE codebook training, and
    Concrete distributions (Jang et al. / Maddison et al. 2017).

    Parameters
    ----------
    logits : Tensor
        Unnormalised log-probabilities of any shape; the last (or
        ``dim``-th) axis indexes the categorical alternatives.
    tau : float, optional
        Temperature :math:`\tau > 0`.  Smaller :math:`\tau` makes the
        relaxation closer to a one-hot, at the cost of larger gradient
        variance.  Default ``1.0``.
    hard : bool, optional
        If ``True`` the forward pass returns a true one-hot vector while
        the backward pass uses the soft gradient (straight-through
        estimator).  Default ``False``.
    eps : float, optional
        Deprecated legacy parameter kept for signature parity.  Default ``1e-10``.
    dim : int, optional
        Dimension over which to take the softmax.  Default ``-1``.

    Returns
    -------
    Tensor
        Sample with the same shape as ``logits``.  Rows sum to ``1``
        along ``dim``; in hard mode each row is one-hot.

    Notes
    -----
    .. math::

        g_i \sim \text{Gumbel}(0, 1), \qquad
        y_i = \frac{\exp((\log p_i + g_i)/\tau)}{\sum_j \exp((\log p_j + g_j)/\tau)}

    Gumbel samples are drawn as :math:`-\log(-\log U)` with
    :math:`U \sim \mathrm{Uniform}(0,1)`, clamped away from the boundaries
    to avoid :math:`\log 0`.  In *hard* mode the result is
    :math:`y_{\text{hard}} - y_{\text{soft}}.\text{detach}() + y_{\text{soft}}`,
    which evaluates to :math:`y_{\text{hard}}` in the forward and to
    :math:`y_{\text{soft}}` in the backward.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import gumbel_softmax
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> y = gumbel_softmax(logits, tau=0.5, hard=True)
    >>> y.sum().item()
    1.0
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.", stacklevel=2)
    # Sample Gumbel noise: −log(−log U) with U ∈ (0, 1).  Clamp away
    # from the boundaries to avoid log(0).
    u: Tensor = _l.rand_like(logits).clip(1e-7, 1.0 - 1e-7)
    gumbel: Tensor = -(-(u.log())).log()
    y_soft: Tensor = softmax((logits + gumbel) / tau, dim=dim)
    if not hard:
        return y_soft
    # Build a one-hot at argmax, then re-attach ``y_soft``'s gradient.
    idx: Tensor = y_soft.argmax(dim=dim, keepdim=True)
    y_hard: Tensor = _l.zeros_like(y_soft).scatter(dim, idx, _l.ones_like(idx, dtype=y_soft.dtype))
    return straight_through(y_hard, y_soft)
