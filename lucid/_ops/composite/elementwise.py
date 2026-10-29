"""Element-wise composite ops layered on the engine.

Three subgroups:

1. **Aliases** for engine ops the reference framework exposes under multiple names
   (``absolute`` ↔ ``abs``, ``subtract`` ↔ ``sub``, etc.).
2. **Inverse-hyperbolic** functions composed from ``log`` and ``sqrt``.
3. **Specials** — ``expm1``, ``sinc``, ``heaviside``, ``xlogy``, ``logit``,
   ``signbit``, ``float_power``, ``fmax`` / ``fmin`` — each implemented as
   a small expression over engine primitives, so autograd works without
   any extra wiring.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import lucid
from lucid import _promotion
from lucid._dispatch import _float_input
from lucid._ops.composite import _bessel
from lucid._ops.composite._gamma import _lgamma, _polygamma
from lucid._ops.composite._opmath import _in_float32, _promote_pair, _real
from lucid._ops.composite._pow2 import _copysign, _frexp, _ldexp, _signbit
from lucid._ops.composite._shared import _is_tensor
from lucid._types import Scalar, TensorOrScalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Aliases ────────────────────────────────────────────────────────────────


def absolute(input: Tensor) -> Tensor:
    r"""Element-wise absolute value.

    Verbose alias of :func:`lucid.abs`, exposed for parity with reference
    frameworks that publish both ``abs`` and ``absolute`` spellings of the
    same operation.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any numeric dtype.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``input`` containing
        :math:`|x_i|` element-wise.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = |x_i|

    The gradient is :math:`\operatorname{sign}(x_i)`; the sub-gradient at
    :math:`x_i = 0` is conventionally taken to be ``0``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.5, 0.0, 2.0])
    >>> lucid.absolute(x)
    tensor([1.5, 0., 2.])
    """
    return lucid.abs(input)


def negative(input: Tensor) -> Tensor:
    r"""Element-wise negation.

    Verbose alias of :func:`lucid.neg`. Equivalent to the unary ``-`` operator
    on a tensor.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any numeric dtype.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``input`` whose entries are
        :math:`-x_i`.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = -x_i

    The gradient with respect to ``input`` is :math:`-1` everywhere.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, -2.0, 3.5])
    >>> lucid.negative(x)
    tensor([-1., 2., -3.5])
    """
    return lucid.neg(input)


def positive(input: Tensor) -> Tensor:
    r"""Element-wise identity — returns the input unchanged.

    Provided as the unary-``+`` counterpart of :func:`negative`, matching
    the convention that ``+x`` is a no-op on numeric data.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        ``input`` itself, returned without modification (no copy, no
        autograd op inserted).

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = +x_i = x_i

    Useful only for symmetry with frameworks that treat ``+x`` as a unary
    operator.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, -2.0])
    >>> lucid.positive(x) is x
    True
    """
    return input


def subtract(input: Tensor, other: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    r"""Element-wise subtraction with an optional scalar multiplier.

    Computes :math:`a - \alpha \cdot b` element-wise, broadcasting ``input``
    and ``other`` to a common shape and following the standard dtype-promotion
    rules.

    Parameters
    ----------
    input : Tensor
        Minuend.
    other : Tensor | Scalar
        Subtrahend. Python scalars are broadcast against ``input``.
    alpha : float, optional
        Scalar multiplier applied to ``other`` before subtraction.
        Defaults to ``1.0`` (plain subtraction).

    Returns
    -------
    Tensor
        Element-wise difference of the broadcast shape of ``input`` and ``other``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = a_i - \alpha \cdot b_i

    Dtype follows standard promotion: e.g. ``int64 - float32 → float32``.
    Gradients flow as :math:`\partial / \partial a = 1` and
    :math:`\partial / \partial b = -\alpha`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([4.0, 5.0])
    >>> b = lucid.tensor([1.0, 2.0])
    >>> lucid.subtract(a, b, alpha=2.0)
    tensor([2., 1.])
    """
    alpha = _promotion.as_number(alpha)
    return (
        input - other if alpha == 1.0 and not isinstance(alpha, bool) else input - (other * alpha)
    )


def multiply(input: Tensor, other: Tensor | Scalar) -> Tensor:
    r"""Element-wise product.

    Verbose alias of the ``*`` operator. Broadcasts ``input`` and ``other`` and
    applies the standard dtype-promotion rules.

    Parameters
    ----------
    input : Tensor
        First operand.
    other : Tensor | Scalar
        Second operand.

    Returns
    -------
    Tensor
        Element-wise product of the broadcast shape of ``input`` and ``other``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = a_i \cdot b_i

    Gradients: :math:`\partial / \partial a = b`,
    :math:`\partial / \partial b = a`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.multiply(a, 2.0)
    tensor([2., 4., 6.])
    """
    return input * other


def divide(input: Tensor, other: Tensor | Scalar, *, rounding_mode: str | None = None) -> Tensor:
    r"""Element-wise division — :func:`lucid.div` under its longer name.

    ``rounding_mode=None`` is the ``/`` operator: integer operands are
    promoted to a floating dtype.  ``"trunc"`` rounds the quotient toward
    zero and ``"floor"`` toward negative infinity, keeping an integer dtype.

    Parameters
    ----------
    input : Tensor
        Numerator.
    other : Tensor | Scalar
        Denominator.
    rounding_mode : {None, "trunc", "floor"}, optional
        How the quotient is rounded.  Default ``None``: not at all.

    Returns
    -------
    Tensor
        Element-wise quotient of the broadcast shape of ``input`` and ``other``.

    Notes
    -----
    .. math::

        \text{out}_i = \frac{a_i}{b_i}

    Division by zero gives :math:`\pm\infty` (NaN for ``0/0``).  The gradients
    are :math:`1/b` and :math:`-a / b^{2}`, and zero for a rounded quotient.

    Examples
    --------
    >>> import lucid
    >>> lucid.divide(lucid.tensor([6.0, 9.0]), 3.0)
    tensor([2., 3.])
    >>> lucid.divide(lucid.tensor([7, -7]), 2, rounding_mode="trunc")
    tensor([3, -3], dtype=lucid.int64)
    """
    return lucid.div(input, other, rounding_mode=rounding_mode)


def true_divide(input: Tensor, other: Tensor | Scalar) -> Tensor:
    r"""Element-wise true (floating-point) division.

    Equivalent to :func:`divide`; named ``true_divide`` for parity with
    NumPy / reference-framework spellings that distinguish it from
    integer-flavored ``floor_divide``.

    Parameters
    ----------
    input : Tensor
        Numerator.
    other : Tensor | Scalar
        Denominator.

    Returns
    -------
    Tensor
        Element-wise quotient in a floating-point dtype.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \frac{a_i}{b_i}

    Even for integer inputs, the output dtype is a floating type — this is
    the defining property that separates ``true_divide`` from
    ``floor_divide``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([7, 8], dtype=lucid.int32)
    >>> lucid.true_divide(a, 2)
    tensor([3.5, 4.])
    """
    return input / other


def rsub(input: Tensor, other: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    r"""Reverse subtraction: :math:`b - \alpha \cdot a`.

    Useful when ``other`` is a Python scalar that cannot left-bind the ``-``
    operator against a tensor in a particular calling convention.

    Parameters
    ----------
    input : Tensor
        Right operand (subtrahend, multiplied by ``alpha``).
    other : Tensor | Scalar
        Left operand (minuend).
    alpha : float, optional
        Scalar multiplier applied to ``input``. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Element-wise difference of the broadcast shape of ``input`` and ``other``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = b_i - \alpha \cdot a_i

    Equivalent to :func:`subtract` with the operands reversed (and ``alpha``
    re-attached to the new second argument). Gradient w.r.t. ``input`` is
    :math:`-\alpha`; w.r.t. ``other`` is :math:`1`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0])
    >>> lucid.rsub(a, 10.0)
    tensor([9., 8.])
    """
    alpha = _promotion.as_number(alpha)
    return (
        other - input if alpha == 1.0 and not isinstance(alpha, bool) else other - (input * alpha)
    )


def arctan2(input: Tensor, other: Tensor) -> Tensor:
    r"""Quadrant-correct two-argument arctangent.

    Verbose alias of :func:`lucid.atan2`.  Returns the polar angle
    :math:`\theta` of the point :math:`(x, y)` in the Cartesian plane,
    resolving the correct quadrant from the signs of both arguments.

    Parameters
    ----------
    input : Tensor
        Ordinate (imaginary / y-coordinate). Must broadcast with ``other``.
    other : Tensor
        Abscissa (real / x-coordinate). Must broadcast with ``input``.

    Returns
    -------
    Tensor
        Element-wise angle in radians on :math:`[-\pi, \pi]`, with the
        broadcast shape of ``input`` and ``other``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \arctan2(y, x) =
        \begin{cases}
            \arctan(y/x),          & x > 0, \\
            \arctan(y/x) + \pi,    & x < 0,\ y \geq 0, \\
            \arctan(y/x) - \pi,    & x < 0,\ y < 0, \\
            +\pi/2,                & x = 0,\ y > 0, \\
            -\pi/2,                & x = 0,\ y < 0, \\
            0,                     & x = 0,\ y = 0.
        \end{cases}

    The naive :math:`\arctan(y / x)` collapses both half-planes into
    :math:`(-\pi/2, \pi/2)`; ``arctan2`` instead inspects the sign of
    each argument to recover the full :math:`[-\pi, \pi]` range, which
    is essential for converting Cartesian to polar coordinates.

    Examples
    --------
    >>> import lucid
    >>> y = lucid.tensor([1.0,  1.0, -1.0, -1.0])
    >>> x = lucid.tensor([1.0, -1.0, -1.0,  1.0])
    >>> lucid.arctan2(y, x)
    tensor([0.7854, 2.356, -2.356, -0.7854])
    """
    return lucid.atan2(input, other)


# ── Inverse hyperbolic (composed) ──────────────────────────────────────────


def arccosh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic cosine.

    Verbose alias of :func:`acosh`.  Computes the principal branch
    :math:`\operatorname{arccosh}(x) = \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr)`,
    defined for real :math:`x \geq 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor; element-wise values must satisfy :math:`x \geq 1`
        for a real result.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic cosine, same shape as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arccosh}(x) =
        \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr), \qquad x \geq 1.

    Inputs with :math:`x < 1` produce NaN through the real ``sqrt``
    branch.  The derivative is :math:`1 / \sqrt{x^{2} - 1}`, singular
    at :math:`x = 1`.  Both ``arccosh`` and :func:`acosh` refer to the
    same composite; either spelling may be used for parity with NumPy
    or reference-framework code.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 10.0])
    >>> lucid.arccosh(x)
    tensor([0., 1.317, 2.993])
    """
    input = _float_input(input)
    return lucid.log(input + lucid.sqrt(input * input - 1.0))


def acosh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic cosine.

    Short-name alias of :func:`arccosh`. Computes the principal branch of
    :math:`\operatorname{arccosh}(x)` for real inputs.

    Parameters
    ----------
    input : Tensor
        Input tensor with values in the domain :math:`x \geq 1`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic cosine.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{acosh}(x) = \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr),
        \qquad x \geq 1.

    Inputs with :math:`x < 1` produce NaN. The derivative is
    :math:`1 / \sqrt{x^{2} - 1}`, which is singular at :math:`x = 1`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 10.0])
    >>> lucid.acosh(x)
    tensor([0., 1.317, 2.993])
    """
    return arccosh(input)


def arcsinh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic sine.

    Verbose alias of :func:`asinh`.  Computes
    :math:`\operatorname{arcsinh}(x) = \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr)`,
    defined for all real ``input``.

    Parameters
    ----------
    input : Tensor
        Input tensor; domain is all of :math:`\mathbb{R}`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic sine, same shape as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arcsinh}(x) =
        \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr).

    Odd function: :math:`\operatorname{arcsinh}(-x) = -\operatorname{arcsinh}(x)`.
    Derivative is :math:`1 / \sqrt{x^{2} + 1}`.  Both ``arcsinh`` and
    :func:`asinh` refer to the same composite.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.arcsinh(x)
    tensor([-0.8814, 0., 0.8814])
    """
    input = _float_input(input)
    return lucid.log(input + lucid.sqrt(input * input + 1.0))


def asinh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic sine.

    Short-name alias of :func:`arcsinh`. Defined for all real inputs.

    Parameters
    ----------
    input : Tensor
        Input tensor. Domain is all of :math:`\mathbb{R}`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic sine.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{asinh}(x) = \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr).

    The function is odd: :math:`\operatorname{asinh}(-x) = -\operatorname{asinh}(x)`.
    Derivative is :math:`1 / \sqrt{x^{2} + 1}`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.asinh(x)
    tensor([-0.8814, 0., 0.8814])
    """
    return arcsinh(input)


def arctanh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic tangent.

    Verbose alias of :func:`atanh`.  Computes
    :math:`\operatorname{arctanh}(x) =
    \tfrac{1}{2}\log\!\bigl((1+x)/(1-x)\bigr)`, defined for
    :math:`|x| < 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor; element-wise values must satisfy :math:`|x| < 1`
        for a finite real result.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic tangent, same shape as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arctanh}(x) =
        \tfrac{1}{2}\log\!\Bigl(\tfrac{1 + x}{1 - x}\Bigr),
        \qquad |x| < 1.

    Boundary values :math:`x = \pm 1` produce :math:`\pm\infty`;
    :math:`|x| > 1` produces NaN.  Derivative is :math:`1 / (1 - x^{2})`.
    Both ``arctanh`` and :func:`atanh` refer to the same composite.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-0.5, 0.0, 0.5])
    >>> lucid.arctanh(x)
    tensor([-0.5493, 0., 0.5493])
    """
    input = _float_input(input)
    return lucid.log((1.0 + input) / (1.0 - input)) * 0.5


def atanh(input: Tensor) -> Tensor:
    r"""Inverse hyperbolic tangent.

    Short-name alias of :func:`arctanh`. Domain is the open interval
    :math:`(-1, 1)`; values at the boundary diverge to :math:`\pm\infty`.

    Parameters
    ----------
    input : Tensor
        Input tensor with values in :math:`|x| < 1`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic tangent.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{atanh}(x) = \tfrac{1}{2}\,
        \log\!\Bigl(\tfrac{1 + x}{1 - x}\Bigr),
        \qquad |x| < 1.

    Values outside :math:`(-1, 1)` produce NaN; :math:`x = \pm 1` produces
    :math:`\pm\infty`. The derivative is :math:`1 / (1 - x^{2})`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-0.5, 0.0, 0.5])
    >>> lucid.atanh(x)
    tensor([-0.5493, 0., 0.5493])
    """
    return arctanh(input)


# ── Specials ───────────────────────────────────────────────────────────────


def expm1(input: Tensor) -> Tensor:
    r"""Element-wise :math:`e^{x} - 1`.

    Computes the exponential-minus-one function to full relative precision,
    including for :math:`|x| \ll 1`, where ``exp(x) - 1`` cancels: in
    float32 that naive form returned ``1.19e-7`` for ``x = 1e-7`` (19% off)
    and ``0`` below it.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any floating-point dtype.

    Returns
    -------
    Tensor
        Element-wise :math:`e^{x} - 1`.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = e^{x_i} - 1.

    The function is useful in probability and finance where :math:`x` is
    a small log-difference, since ``log(1 + y) = expm1⁻¹(y)`` and the pair
    avoids subtractive cancellation when chained.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> lucid.expm1(x)
    tensor([0., 1.718, 6.389])
    """
    # Kahan's form: with u = exp(x), (u - 1) * x / log(u) cancels the rounding
    # of u against the same rounding in log(u), so it keeps full relative
    # precision near 0.  The guards take u == 1 (x underflowed), u == 0 and
    # u == inf, where the ratio is 0/0 or inf/inf.
    with lucid.no_grad():
        u = lucid.exp(input)
        accurate = lucid.where(u == 1.0, input, (u - 1.0) * input / lucid.log(u))
        # Kahan's second guard: once u - 1 rounds to -1, -1 is the answer.  The
        # ratio would take log(u) of a subnormal u, whose rounding is no longer
        # relative — expm1(-100) in float32 came out -1.000169.
        accurate = lucid.where(u - 1.0 == -1.0, u - 1.0, accurate)
        accurate = lucid.where((u == 0.0) | ~lucid.isfinite(u), u - 1.0, accurate)
    # The value is the accurate one; the gradient is exp(x)'s, carried by the
    # naive form under it, so the guards' 0/0 branches never reach backward.
    naive = lucid.exp(input) - 1.0
    with lucid.no_grad():
        correction = lucid.where(
            lucid.isfinite(naive), accurate - naive, lucid.zeros_like(accurate)
        )
    # Where exp(x) rounds to 1, x + 0.5 * c * (c - c.detach()) with clamped c
    # equals x bit-for-bit (preserving expm1(-0) = -0 and keeping ±inf finite
    # in MulBackward) while carrying both the unit slope and unit curvature of
    # exp(x) at the origin for higher-order autograd.
    c = lucid.clamp(input, -1.0, 1.0)
    small_branch = input + 0.5 * c * (c - c.detach())
    return lucid.where(u == 1.0, small_branch, naive + correction)


def sinc(input: Tensor) -> Tensor:
    r"""Normalised sinc function with the standard removable singularity at 0.

    The "normalised" sinc carries a factor of :math:`\pi` inside the sine,
    so its zeros coincide with the non-zero integers — the convention used
    in signal processing.

    Parameters
    ----------
    input : Tensor
        Input tensor. Any floating-point dtype.

    Returns
    -------
    Tensor
        Element-wise normalised sinc with ``sinc(0) = 1``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{sinc}(x) =
        \begin{cases}
            \dfrac{\sin(\pi x)}{\pi x}, & x \neq 0, \\
            1,                          & x = 0.
        \end{cases}

    The composite uses :func:`lucid.where` to substitute the limit value
    at ``x = 0`` so the gradient flows cleanly through the removable
    singularity.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 0.5, 1.5])
    >>> lucid.sinc(x)
    tensor([1., 0.6366, -0.2122])
    """
    px = input * math.pi
    is_zero = input == 0.0
    safe_px = lucid.where(is_zero, lucid.full_like(input, 1.0), px)
    val = lucid.sin(safe_px) / safe_px
    # Use the 2nd-order Taylor expansion 1 - (pi * x)^2 / 6 at x = 0 so
    # both the value (1), slope (0), and curvature (-pi^2 / 3) are exact.
    return lucid.where(is_zero, 1.0 - (px * px) / 6.0, val)


def heaviside(input: Tensor, values: Tensor | Scalar) -> Tensor:
    r"""Heaviside step function with a user-selectable value at :math:`x = 0`.

    Produces ``0`` for negative inputs and ``1`` for positive inputs; the
    value at exactly zero is taken from ``values`` (typically ``0``,
    ``0.5``, or ``1`` depending on convention).

    Parameters
    ----------
    input : Tensor
        Input tensor.
    values : Tensor | Scalar
        Value(s) used wherever ``x == 0``. If a scalar, it is broadcast to
        the shape of ``input``; if a tensor, it must broadcast against ``input``.

    Returns
    -------
    Tensor
        Element-wise Heaviside step.

    Notes
    -----
    Mathematical definition:

    .. math::

        H(x_i) =
        \begin{cases}
            0,            & x_i < 0, \\
            \text{values}_i, & x_i = 0, \\
            1,            & x_i > 0.
        \end{cases}

    The gradient is zero almost everywhere — the derivative is a Dirac
    delta in the distributional sense, which autograd does not propagate.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.heaviside(x, 0.5)
    tensor([0., 0.5, 1.])
    """
    if not _is_tensor(values):
        values = lucid.full_like(input, float(_promotion.as_number(cast(float, values))))
    return lucid.where(
        input == 0.0,
        values,
        lucid.where(input > 0.0, lucid.ones_like(input), lucid.zeros_like(input)),
    )


def _x_times_log(
    x: Tensor | Scalar,
    y: Tensor | Scalar,
    log: Callable[[Tensor], Tensor],
    pole: float,
    root: float,
) -> Tensor:
    r""":math:`x \cdot \log(y)` for a logarithm ``log``, taken as ``0`` at ``x == 0``.

    The single owner of the ``0 · log`` convention: :func:`xlogy`,
    :func:`lucid.special.xlog1py` and :func:`lucid.special.entr` all reach
    it, and so do the log-densities of ``lucid.distributions``.

    The guard is on the *operand*, not on the product.  Where ``x == 0``
    and ``log(y)`` is not finite — ``y`` at or below the pole, or
    infinite — ``y`` is replaced by ``root`` (where ``log`` is exactly 0)
    before the logarithm, so the product is an exact 0 and neither
    factor's backward sees an infinity.  Masking the product afterwards
    is not enough: ``where`` hands the unselected branch a zero gradient,
    and ``mul``'s backward still forms ``0 · log(0) = 0 · (-inf) = NaN``
    for ``x`` and ``0 / 0`` for ``y``.  Everywhere else — including
    ``x == 0`` with a finite ``log(y)`` — the product is left alone, so
    ``d/dx = log(y)`` there and a NaN ``y`` still propagates.

    Parameters
    ----------
    x, y : Tensor | Scalar
        Broadcast-compatible operands.  A Python scalar follows the other
        operand's device and is weak in dtype; integer and bool operands
        are taken in the default float dtype.
    log : Callable[[Tensor], Tensor]
        The logarithm, ``lucid.log`` or ``lucid.log1p``.
    pole : float
        Where ``log`` diverges to ``-inf``: ``0`` for ``log``, ``-1`` for
        ``log1p``.
    root : float
        Where ``log`` is exactly 0: ``1`` for ``log``, ``0`` for ``log1p``.

    Returns
    -------
    Tensor
        ``x * log(y)`` with the ``x == 0`` convention.
    """
    x, y = _promote_pair(x, y, float_valued=True)
    degenerate = (x == 0.0) & ((y <= pole) | lucid.isinf(y))
    return x * log(lucid.where(degenerate, root, y))


def xlogy(input: Tensor | Scalar, other: Tensor | Scalar) -> Tensor:
    r"""Compute :math:`x \log y` with the convention :math:`0 \cdot \log y = 0`.

    Frequently used to compute cross-entropy losses and log-densities
    where the limit :math:`\lim_{x \to 0^{+}} x \log x = 0` must be
    honoured to avoid NaN contamination from probability values that
    are exactly zero.

    Parameters
    ----------
    input : Tensor | Scalar
        Left operand (multiplier).
    other : Tensor | Scalar
        Argument of the logarithm. Must be non-negative for a real result.

    Returns
    -------
    Tensor
        Element-wise :math:`x \log y` with the ``x == 0`` convention
        applied.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{xlogy}(x, y) =
        \begin{cases}
            \mathrm{NaN},  & y = \mathrm{NaN}, \\
            0,             & x = 0, \\
            x \cdot \log y, & x \neq 0.
        \end{cases}

    The convention covers only ``x == 0``: with ``x != 0`` the plain
    definition applies, so ``xlogy(2, 0)`` is ``-inf`` and a negative
    ``other`` gives NaN.

    Gradients are :math:`\partial/\partial x = \log y` and
    :math:`\partial/\partial y = x / y`, except where ``x == 0`` and
    :math:`\log y` is not finite (``y <= 0`` or ``other`` infinite): there
    both are ``0``, so the backward never forms ``0 \cdot \infty``.  At
    ``x == 0`` with ``0 < y < \infty`` the ``input`` gradient stays
    :math:`\log y`.  Two cells differ from the reference framework on
    purpose: at ``(0, 0)`` its ``other`` gradient is ``0 / 0 = NaN`` and
    Lucid's is ``0``, and at ``(0, +\infty)`` its ``input`` gradient is
    :math:`+\infty` and Lucid's is ``0`` (a composite cannot give a
    value of 0 a gradient of :math:`+\infty` without forming
    ``0 \cdot \infty`` in the forward pass).

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([0.0, 2.0, 3.0])
    >>> lucid.xlogy(x, y)
    tensor([0., 0.6931, 2.197])
    >>> lucid.xlogy(2.0, 0.0)
    tensor(-inf)
    """
    return _x_times_log(input, other, lucid.log, pole=0.0, root=1.0)


def logit(input: Tensor, eps: float | None = None) -> Tensor:
    r"""Logit function — inverse of the logistic sigmoid.

    Maps a probability in :math:`(0, 1)` to a log-odds value in
    :math:`\mathbb{R}`. Optionally clamps the input to
    :math:`[\epsilon, 1 - \epsilon]` first to avoid divergence at the
    boundaries.

    Parameters
    ----------
    input : Tensor
        Input tensor with values in :math:`(0, 1)`.
    eps : float | None, optional
        If provided, ``input`` is clamped to ``[eps, 1 - eps]`` before the
        log-ratio is taken, preventing infinities. Defaults to ``None``
        (no clamping).

    Returns
    -------
    Tensor
        Element-wise logit value.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{logit}(x) = \log\!\Bigl(\tfrac{x}{1 - x}\Bigr).

    Inverse of the sigmoid:
    :math:`\operatorname{logit}(\sigma(z)) = z`. Without ``eps``, inputs
    of exactly ``0`` or ``1`` produce :math:`-\infty` and :math:`+\infty`
    respectively.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.1, 0.5, 0.9])
    >>> lucid.logit(x)
    tensor([-2.197, 0., 2.197])
    """
    input = _float_input(input)
    if eps is not None:
        input = lucid.clamp(input, eps, 1.0 - eps)
    return lucid.log(input / (1.0 - input))


def signbit(input: Tensor) -> Tensor:
    r"""Element-wise IEEE-754 sign-bit predicate.

    Returns a boolean tensor that is ``True`` wherever the sign bit of
    ``input`` is set (including ``-0.0``, ``-inf``, and negative ``NaN``
    for floating-point inputs, and strictly negative values for integers).

    Parameters
    ----------
    input : Tensor
        Input tensor. Any real numeric dtype.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``input``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{signbit}(x_i) =
        \begin{cases}
            \text{True},  & \text{sign bit of } x_i \text{ is } 1, \\
            \text{False}, & \text{otherwise.}
        \end{cases}

    Non-differentiable: the gradient is zero almost everywhere and the
    output dtype is boolean.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, -0.0, 0.0, 1.0])
    >>> lucid.signbit(x)
    tensor([True, True, False, False], dtype=lucid.bool)
    """
    return _signbit(input)


def float_power(input: Tensor | Scalar, exponent: Tensor | Scalar) -> Tensor:
    r"""Element-wise power that always promotes operands to ``float64``.

    Unlike the regular :func:`lucid.pow`, ``float_power`` upcasts both
    operands to ``float64`` before exponentiation, so that negative bases
    raised to non-integer powers and very large exponents behave
    consistently without integer overflow or domain errors.

    Parameters
    ----------
    input : Tensor | Scalar
        Base.
    exponent : Tensor | Scalar
        Exponent.

    Returns
    -------
    Tensor
        Element-wise ``x ** y`` evaluated in ``float64``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = x_i^{y_i}.

    Promotion is mandatory — even integer inputs are cast to ``float64``.
    The result is always ``float64`` regardless of input dtype.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([2, 3], dtype=lucid.int32)
    >>> lucid.float_power(x, 0.5).dtype
    lucid.float64
    """
    if _is_tensor(input):
        input = input.to(dtype=lucid.float64)
    if _is_tensor(exponent):
        exponent = exponent.to(dtype=lucid.float64)
    if not _is_tensor(input) and _is_tensor(exponent):
        input = exponent.new_full((), float(_promotion.as_number(input)))
    if not _is_tensor(exponent) and _is_tensor(input):
        exponent = input.new_full((), float(_promotion.as_number(exponent)))
    # ``cast`` evaluates its first argument at run time, and ``Tensor``
    # is imported here only under ``TYPE_CHECKING`` — so the bare name
    # raised ``NameError`` on **every** call and float_power had never
    # worked.  The string form is what the rest of the codebase uses.
    return lucid.pow(cast("Tensor", input), cast("Tensor", exponent))


def fmax(input: TensorOrScalar, other: TensorOrScalar) -> Tensor:
    r"""Element-wise NaN-quiet maximum.

    Like :func:`lucid.maximum`, but treats NaN as missing data: if exactly
    one operand is NaN at a given position, the non-NaN value is returned.
    Only when both are NaN does the result remain NaN.

    Parameters
    ----------
    input : Tensor
        First operand.
    other : Tensor
        Second operand. Must broadcast with ``input``.

    Returns
    -------
    Tensor
        Element-wise NaN-quiet maximum.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{fmax}(a_i, b_i) =
        \begin{cases}
            b_i,           & a_i = \text{NaN}, \\
            a_i,           & b_i = \text{NaN}, \\
            \max(a_i, b_i),& \text{otherwise.}
        \end{cases}

    Contrast with :func:`lucid.maximum`, which propagates NaN per IEEE 754.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> a = lucid.tensor([1.0, math.nan, 3.0])
    >>> b = lucid.tensor([2.0, 2.0, math.nan])
    >>> lucid.fmax(a, b)
    tensor([2., 2., 3.])
    """
    input, other = _promote_pair(input, other)
    return lucid.where(lucid.logical_or(input >= other, lucid.isnan(other)), input, other)


def fmin(input: TensorOrScalar, other: TensorOrScalar) -> Tensor:
    r"""Element-wise NaN-quiet minimum.

    Like :func:`lucid.minimum`, but treats NaN as missing data: if exactly
    one operand is NaN at a given position, the non-NaN value is returned.
    Only when both are NaN does the result remain NaN.

    Parameters
    ----------
    input : Tensor
        First operand.
    other : Tensor
        Second operand. Must broadcast with ``input``.

    Returns
    -------
    Tensor
        Element-wise NaN-quiet minimum.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{fmin}(a_i, b_i) =
        \begin{cases}
            b_i,           & a_i = \text{NaN}, \\
            a_i,           & b_i = \text{NaN}, \\
            \min(a_i, b_i),& \text{otherwise.}
        \end{cases}

    Contrast with :func:`lucid.minimum`, which propagates NaN per IEEE 754.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> a = lucid.tensor([1.0, math.nan, 3.0])
    >>> b = lucid.tensor([2.0, 2.0, math.nan])
    >>> lucid.fmin(a, b)
    tensor([1., 2., 3.])
    """
    input, other = _promote_pair(input, other)
    return lucid.where(lucid.logical_or(input <= other, lucid.isnan(other)), input, other)


# ── Special math functions ─────────────────────────────────────────────────


def erfc(input: Tensor) -> Tensor:
    r"""Complementary error function :math:`\text{erfc}(x) = 1 - \text{erf}(x)`.

    Computed directly, not as ``1 - erf(x)``: past ``x ≈ 4`` that
    difference is a few units of float32's spacing below one, so
    ``erfc(4)`` came back 0 where the answer is 1.54e-8 — and every tail
    probability built on it inherited the zero.  The form used is the
    Chebyshev fit from Numerical Recipes (``erfcc``), whose relative error
    stays below 1.2e-7 for every ``input``, carried to negative ``input`` by
    :math:`\text{erfc}(-x) = 2 - \text{erfc}(x)`.

    Parameters
    ----------
    input : Tensor
        Real-valued input; a bool or integer one is taken at the default float.

    Returns
    -------
    Tensor
        Same shape as ``input``, values in :math:`[0, 2]`.

    Examples
    --------
    >>> import lucid
    >>> abs(lucid.erfc(lucid.tensor(4.0)).item() - 1.5417e-08) < 1e-12
    True
    """
    input = _real(input)
    if input.dtype == lucid.float64:
        return _erfc_double(input)
    with lucid.no_grad():
        z = lucid.where(input >= 0.0, input, -input)
        t = 1.0 / (1.0 + 0.5 * z)
        poly = lucid.full_like(t, 0.17087277)
        for c in (
            -0.82215223,
            1.48851587,
            -1.13520398,
            0.27886807,
            -0.18628806,
            0.09678418,
            0.37409196,
            1.00002368,
            -1.26551223,
        ):
            poly = c + t * poly
        tail = t * lucid.exp(poly - z * z)
        accurate = lucid.where(input >= 0.0, tail, 2.0 - tail)
    if not input.requires_grad:
        return accurate
    erf_val = lucid.erf(input)
    return accurate + (erf_val.detach() - erf_val)


def _erfc_double(x: Tensor) -> Tensor:
    """erfc in float64, where the Chebyshev fit's 1.2e-7 is float32 grade.

    ``1 - erf(x)`` holds to float64's spacing until erfc itself shrinks
    toward it, so it serves below x = 3; from there the continued fraction
    for erfcx does, at 60 terms, without ever subtracting.  Both branches
    are evaluated — ``where`` selects — so each is clamped onto its own
    side first, keeping the losing one finite for the gradient.
    """
    z = lucid.where(x >= 0.0, x, -x)
    head = 1.0 - lucid.erf(lucid.clamp(z, max=3.0))
    far = lucid.clamp(z, min=3.0)
    frac = lucid.zeros_like(far)
    for k in range(60, 0, -1):
        frac = (0.5 * k) / (far + frac)
    tail = lucid.exp(-far * far) / (math.sqrt(math.pi) * (far + frac))
    value = lucid.where(z < 3.0, head, tail)
    return lucid.where(x >= 0.0, value, 2.0 - value)


def copysign(input: Tensor, other: TensorOrScalar) -> Tensor:
    """Element-wise sign transplant: magnitudes from ``input``, signs from ``other``.

    Useful for sign-preserving operations that need to preserve a sign
    bit independent of an arbitrary magnitude computation.

    Parameters
    ----------
    input : Tensor
        Magnitude source; result has ``|x|`` element-wise.
    other : Tensor or Scalar
        Sign source; result takes the IEEE-754 sign bit of ``other``
        (broadcastable to the shape of ``input``).

    Returns
    -------
    Tensor
        Broadcast shape of ``input`` and ``other`` at their common floating dtype.
    """
    x, y = _promote_pair(input, other, float_valued=True)
    return _copysign(x, y)


def ldexp(input: Tensor, other: Tensor | Scalar) -> Tensor:
    r"""Element-wise :math:`\text{input} \cdot 2^{\text{other}}`.

    For an integral exponent the product is rounded once, so it is exact
    wherever it is representable.  That holds on every device and at every
    magnitude, subnormal operands and answers included, and also where
    :math:`2^{\text{other}}` alone is out of range (``ldexp(1e30, -200)``).
    A fractional exponent costs an ulp or so.  Differentiable in both:
    :math:`\partial / \partial\,\text{input} = 2^{\text{other}}` (exact
    for an integral exponent) and :math:`\partial / \partial\,\text{other}
    = \text{input} \cdot 2^{\text{other}} \ln 2` (within a few ulp).

    Parameters
    ----------
    input : Tensor
        Mantissa.
    other : Tensor or Scalar
        Power-of-two exponent; broadcastable to the shape of ``input``.

    Returns
    -------
    Tensor
        Broadcast shape of ``input`` and ``other``.

    See Also
    --------
    frexp : inverse decomposition into mantissa / exponent.

    Notes
    -----
    On Metal a subnormal gradient flushes to zero, as in every backward
    there; under ``lucid.compile`` float32 subnormal values flush too.
    """
    orig = (input := _float_input(input)).dtype
    work_dtype = orig if orig in (lucid.float32, lucid.float64) else lucid.float32
    work = input if orig == work_dtype else input.to(dtype=work_dtype)
    out = _ldexp(work, other)
    return out.to(dtype=orig) if orig != work_dtype else out


def frexp(input: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose ``input`` into mantissa ``m`` and exponent ``e`` such that
    ``input = m * 2**e`` with ``|m|`` in ``[0.5, 1)`` (or ``m == 0``).

    Returns a ``(mantissa, exponent)`` tuple.  Exponent dtype is int32.
    Non-finite inputs and signed zero propagate verbatim:
    ``NaN -> (NaN, 0)``, ``±inf -> (±inf, 0)``, ``±0.0 -> (±0.0, 0)``.
    Exact on every device, subnormal inputs included.  The mantissa is
    differentiable, ``dm / dinput = 2**-e`` (a subnormal one flushes on
    Metal); the exponent is not.
    """
    orig_dtype = input.dtype
    work_dtype = orig_dtype if orig_dtype in (lucid.float32, lucid.float64) else lucid.float32
    work = input if orig_dtype == work_dtype else input.to(dtype=work_dtype)
    m, e = _frexp(work)
    return (m.to(dtype=orig_dtype) if orig_dtype != work_dtype else m), e


# ── Log-gamma and Digamma ─────────────────────────────────────────────────


def lgamma(input: Tensor) -> Tensor:
    r"""Natural log of the gamma function :math:`\ln \Gamma(x)`.

    Implemented as a pure composite over engine primitives — uses the
    Lanczos approximation (Numerical Recipes, :math:`g = 7`, 9 series
    coefficients).  Relative error stays below ``1.5e-15`` for real
    ``x > 0`` in float64; ``x < 1/2`` goes through Euler's reflection.
    Autograd flows through automatically because every step is an engine
    op.

    Parameters
    ----------
    input : Tensor
        Real input.  The non-positive integers and ``±inf`` give ``+inf``.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    See Also
    --------
    digamma : derivative :math:`\psi(x) = \frac{d}{dx} \ln \Gamma(x)`.
    """
    return _lgamma(input)


def digamma(input: Tensor) -> Tensor:
    r"""Digamma function :math:`\psi(x) = \frac{d}{dx} \ln \Gamma(x)`.

    Implemented as a composite of engine primitives so autograd flows
    through naturally, and its gradient is the trigamma function.  For
    ``x >= 0`` the argument is shifted by 8 via
    :math:`\psi(x) = \psi(x + 8) - \sum_{k=0}^{7} 1/(x + k)` and the
    asymptotic Bernoulli series is evaluated there; for ``x < 0`` the
    reflection formula :math:`\psi(x) = \psi(1 - x) - \pi \cot(\pi x)`
    maps the argument onto the right half.

    Parameters
    ----------
    input : Tensor
        Real input.  The negative integers and ``-inf`` are poles and give
        NaN; ``psi(+0)`` is ``-inf`` and ``psi(-0)`` is ``+inf``.

    Returns
    -------
    Tensor
        Same shape as ``input``.

    See Also
    --------
    lgamma : :math:`\ln \Gamma(x)` itself.
    """
    return _polygamma(0, input)


# ── Modified Bessel function I₀ ───────────────────────────────────────────


def i0(input: Tensor) -> Tensor:
    r"""Modified Bessel function of the first kind, order 0: :math:`I_0(x)`.

    Composite implementation via the Abramowitz & Stegun polynomial
    approximation (Table 9.8.1):

    * ``|x| <= 3.75`` — polynomial series in :math:`(x / 3.75)^2`.
    * ``|x| >  3.75`` — polynomial series in :math:`3.75 / |x|`, multiplied
      by :math:`e^{|x|} / \sqrt{|x|}` to recover the exponential growth.

    Each branch runs on ``input`` clamped into its own interval, so the one
    ``where`` does not pick contributes no ``nan`` to the gradient.

    A ``float64`` input takes the Cephes double-precision series instead.

    Parameters
    ----------
    input : Tensor
        Real input.

    Returns
    -------
    Tensor
        :math:`I_0(x)` element-wise; always positive.
    """
    x = _real(input)
    if x.dtype == lucid.float64:
        return _bessel._i0(x)
    return _in_float32(_bessel._i0_as_fit, x)


__all__ = [
    "absolute",
    "negative",
    "positive",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "rsub",
    "arctan2",
    "arccosh",
    "acosh",
    "arcsinh",
    "asinh",
    "arctanh",
    "atanh",
    "expm1",
    "sinc",
    "heaviside",
    "xlogy",
    "logit",
    "signbit",
    "float_power",
    "fmax",
    "fmin",
    "erfc",
    "copysign",
    "ldexp",
    "frexp",
    "lgamma",
    "digamma",
    "i0",
]


# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its module globals when something asks for
# them, so a name that exists only under ``TYPE_CHECKING`` makes
# ``inspect.signature`` raise NameError — which breaks help(), IDEs and
# the docs build. Imported at the bottom because the module is part of
# the cycle that ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
