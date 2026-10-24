"""nn.functional's shrinkage operators: ``hardshrink``, ``softshrink``, ``tanhshrink``.

Re-exported by :mod:`lucid.nn.functional.activations`.
"""

import lucid as _l
from lucid._C import engine as _C_engine
from lucid._dispatch import _float_operand, _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid.nn.functional._inplace import _dead_zone_zero, _keep_in_graph


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    r"""Hard shrinkage operator.

    Sets elements with magnitude below :math:`\lambda` to zero, leaves
    larger-magnitude elements untouched.  Standard tool for sparse coding
    and the proximal operator of the :math:`\ell_0` "pseudonorm" on a
    coefficient-by-coefficient basis.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    lambd : float, optional
        Threshold :math:`\lambda \ge 0`.  Default ``0.5``.

    Returns
    -------
    Tensor
        ``input`` with every entry in :math:`[-\lambda, \lambda]` set to zero.

    Notes
    -----
    .. math::

        \text{HardShrink}(x) = \begin{cases}
            x & |x| > \lambda \\
            0 & \text{otherwise}
        \end{cases}

    Discontinuous at :math:`x = \pm\lambda` — for an everywhere-continuous
    alternative use :func:`softshrink` (the proximal operator of the
    :math:`\ell_1` norm).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import hardshrink
    >>> x = lucid.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
    >>> hardshrink(x, lambd=0.5)
    tensor([-1., 0., 0., 0., 1.])
    """
    outside = _unwrap((input > lambd) | (input < -lambd))
    return _keep_in_graph(
        _wrap(_C_engine.where(outside, _unwrap(input), _dead_zone_zero(input))), input
    )


def tanhshrink(input: Tensor) -> Tensor:
    r"""Tanh shrinkage activation.

    Behaves like a smooth high-pass filter: subtracts the tanh-saturated
    component of the input, leaving small linear residuals near zero and
    near-linear behaviour for large magnitudes.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape; activation is element-wise.

    Returns
    -------
    Tensor
        Activated tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{TanhShrink}(x) = x - \tanh(x)

    For :math:`|x| \ll 1`, :math:`\tanh(x) \approx x - x^3/3`, so
    :math:`\text{TanhShrink}(x) \approx x^3/3` — small signals are
    cubically suppressed.  For :math:`|x| \gg 1`, :math:`\tanh` saturates
    to :math:`\pm 1` so the output behaves like :math:`x \mp 1`.
    :math:`C^\infty`-smooth and monotonic.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import tanhshrink
    >>> x = lucid.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    >>> tanhshrink(x)
    tensor([-1.036, -0.03788, 0., 0.03788, 1.036])
    """
    if input.dtype == _l.bool_:
        raise TypeError("tanhshrink: bool input is not supported")
    xi = _float_operand(_unwrap(input))
    return _wrap(_C_engine.sub(xi, _C_engine.tanh(xi)))


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    r"""Soft shrinkage operator — proximal operator of :math:`\lambda \|\cdot\|_1`.

    Translates the input toward zero by :math:`\lambda` and zeros out
    everything in the dead zone :math:`[-\lambda, \lambda]`.  Used in
    iterative-shrinkage solvers (ISTA / FISTA) and in sparse autoencoder
    decoders.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    lambd : float, optional
        Shrinkage threshold :math:`\lambda \ge 0`.  Default ``0.5``.

    Returns
    -------
    Tensor
        Shrunk tensor with the same shape as ``input``.

    Notes
    -----
    .. math::

        \text{SoftShrink}(x) = \begin{cases}
            x - \lambda & x > \lambda \\
            x + \lambda & x < -\lambda \\
            0 & \text{otherwise}
        \end{cases}

    Continuous everywhere (unlike :func:`hardshrink`), with derivative
    :math:`1` outside the dead zone and :math:`0` inside it.  Coincides
    with the closed-form solution of
    :math:`\arg\min_y \tfrac12 (y - x)^2 + \lambda |y|`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import softshrink
    >>> x = lucid.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])
    >>> softshrink(x, lambd=0.5)
    tensor([-0.5, 0., 0., 0., 0.5])
    """
    if lambd < 0.0:
        raise _C_engine.InvalidArgument(
            f"softshrink: lambd must be no less than 0.0, but got {lambd}"
        )
    xi = _unwrap(input)
    lam = _C_engine.full(xi.shape, lambd, xi.dtype, xi.device)
    neg_lam = _C_engine.full(xi.shape, -lambd, xi.dtype, xi.device)
    out = _C_engine.where(
        _C_engine.greater(xi, lam), _C_engine.sub(xi, lam), _dead_zone_zero(input)
    )
    return _keep_in_graph(
        _wrap(
            _C_engine.where(_C_engine.less(xi, neg_lam), _C_engine.add(xi, lam), out)
        ),
        input,
    )
