"""The orthogonal polynomials of :mod:`lucid.special`.

All seven families share the recurrence shape ``P_{k+1} = α(x, k)·P_k +
β(x, k)·P_{k-1}``, unfolded into a Python loop of ``n + 1`` terms — fine for
the small degrees orthogonal polynomials are used at, and differentiable in
``x`` throughout.

The degree is read as the reference reads it: a Python number for every
element, or a tensor holding one per element, which broadcasts against
``x`` and joins it in dtype promotion.  Either ``x`` or ``n`` may be the
Python number, not both.  A degree is truncated toward zero, and a negative
one is the zero polynomial.  ``n`` was a Python int alone, and a negative
one raised ``ValueError``.
"""

import math
from collections.abc import Callable, Iterator

import lucid
from lucid._ops.composite._opmath import _real, _real_pair
from lucid._tensor.tensor import Tensor
from lucid._types import TensorOrScalar


def _terms(
    x: Tensor,
    top: int,
    first: Callable[[Tensor], Tensor],
    advance: Callable[[Tensor, Tensor, Tensor, int], Tensor],
) -> Iterator[Tensor]:
    """``P_0, P_1, ..., P_top`` at ``x``: ``P_0`` is 1, ``first(x)`` is ``P_1``,
    and ``advance(x, P_k, P_{k-1}, k)`` is ``P_{k+1}``."""
    if top < 0:
        return
    p_prev = lucid.ones_like(x)
    yield p_prev
    if top < 1:
        return
    p_curr = first(x)
    yield p_curr
    for k in range(1, top):
        p_prev, p_curr = p_curr, advance(x, p_curr, p_prev, k)
        yield p_curr


def _top_degree(op: str, n: float) -> int:
    """The highest degree a call needs, ``n`` truncated toward zero.

    A NaN or infinite degree has no integer to truncate to; the reference
    casts it anyway and answers whatever the cast gives.

    Raises
    ------
    ValueError
        If ``n`` is not finite.
    """
    if not math.isfinite(n):
        raise ValueError(f"{op}: expected a finite degree n, got {n}")
    return int(n)


def _ortho_poly(
    op: str,
    x: TensorOrScalar,
    n: TensorOrScalar,
    first: Callable[[Tensor], Tensor],
    advance: Callable[[Tensor, Tensor, Tensor, int], Tensor],
    *,
    shift: bool = False,
) -> Tensor:
    """``P_n(x)`` of the family ``first`` / ``advance`` (see :func:`_terms`).

    ``shift`` evaluates it at ``2x - 1``, the family's shifted form on
    ``[0, 1]``.  A tensor degree runs the recurrence to its largest entry
    and keeps each element's own term.
    """
    degree: Tensor | None = None
    if isinstance(n, Tensor):
        x, n = _real_pair(op, x, n)
        degree = lucid.trunc(n)
        top = _top_degree(op, degree.max().item() if degree.numel() else -1)
    elif not isinstance(x, Tensor):
        raise TypeError(
            f"{op}: expected a Tensor for x or n, got "
            f"{type(x).__name__} and {type(n).__name__}"
        )
    elif isinstance(n, bool | int | float):
        x, top = _real(x), _top_degree(op, n)
    else:
        raise TypeError(
            f"{op}: expected the degree n to be a Tensor or a real number, "
            f"got {type(n).__name__}"
        )
    if shift:
        x = 2 * x - 1
    out = lucid.zeros_like(x)
    if degree is not None:
        shape = lucid.broadcast_shapes(tuple(x.shape), tuple(degree.shape))
        out = lucid.zeros(shape, dtype=x.dtype, device=x.device)
    for k, term in enumerate(_terms(x, top, first, advance)):
        out = term if degree is None else lucid.where(degree == float(k), term, out)
    return out


def _chebyshev_step(v: Tensor, p: Tensor, q: Tensor, _k: int) -> Tensor:
    """The step all four Chebyshev kinds share; they differ in ``P_1``."""
    return 2 * v * p - q


def chebyshev_polynomial_t(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Chebyshev polynomial of the first kind, :math:`T_n(x)`.

    The Chebyshev T polynomials are orthogonal on :math:`[-1, 1]` with
    weight :math:`1/\sqrt{1 - x^2}` and underlie spectral methods,
    minimax polynomial approximation, and Clenshaw-Curtis quadrature.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`T_n` is evaluated, of any real dtype.  Customarily on
        :math:`[-1, 1]` but defined for all real ``x``.
    n : Tensor | Scalar
        Degree of :math:`T_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`T_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Defined by :math:`T_n(\cos\theta) = \cos(n\theta)` and built by the
    three-term recurrence

    .. math::

        T_0(x) = 1, \quad T_1(x) = x, \quad
        T_{k+1}(x) = 2x\, T_k(x) - T_{k-1}(x).

    On :math:`[-1, 1]`, :math:`|T_n(x)| \le 1`; the extrema lie at the
    Chebyshev nodes :math:`\cos(j\pi/n)` for :math:`j = 0,\ldots,n`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_t
    >>> chebyshev_polynomial_t(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    tensor([-1., -0., -1., 1.])
    """
    return _ortho_poly("chebyshev_polynomial_t", x, n, lambda v: v, _chebyshev_step)


def chebyshev_polynomial_u(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Chebyshev polynomial of the second kind, :math:`U_n(x)`.

    The U polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{1 - x^2}`.  They appear naturally as the derivative of
    the T polynomials (up to a normalisation) and in Gauss-Chebyshev
    quadrature of the second kind.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`U_n` is evaluated, of any real dtype.
    n : Tensor | Scalar
        Degree of :math:`U_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`U_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Defined by :math:`U_n(\cos\theta) = \sin((n+1)\theta)/\sin\theta`
    and obeying

    .. math::

        U_0(x) = 1, \quad U_1(x) = 2x, \quad
        U_{k+1}(x) = 2x\, U_k(x) - U_{k-1}(x).

    Linked to T by :math:`T_n'(x) = n\, U_{n-1}(x)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_u
    >>> chebyshev_polynomial_u(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    tensor([-4., -0., -1., 4.])
    """
    return _ortho_poly("chebyshev_polynomial_u", x, n, lambda v: 2 * v, _chebyshev_step)


def chebyshev_polynomial_v(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Chebyshev polynomial of the third kind, :math:`V_n(x)`.

    The V polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{(1 + x)/(1 - x)}` and arise in problems with one
    Dirichlet and one Neumann boundary condition along the interval.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`V_n` is evaluated, of any real dtype.
    n : Tensor | Scalar
        Degree of :math:`V_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`V_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Half-angle definition

    .. math::

        V_n(\cos\theta) = \frac{\cos((n + \tfrac{1}{2})\theta)}
                                {\cos(\theta/2)},

    with the recurrence

    .. math::

        V_0(x) = 1, \quad V_1(x) = 2x - 1, \quad
        V_{k+1}(x) = 2x\, V_k(x) - V_{k-1}(x).

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_v
    >>> chebyshev_polynomial_v(lucid.tensor([-1.0, 0.0, 1.0]), n=2)
    tensor([5., -1., 1.])
    """
    return _ortho_poly(
        "chebyshev_polynomial_v", x, n, lambda v: 2 * v - 1, _chebyshev_step
    )


def chebyshev_polynomial_w(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Chebyshev polynomial of the fourth kind, :math:`W_n(x)`.

    The W polynomials are orthogonal on :math:`[-1, 1]` with weight
    :math:`\sqrt{(1 - x)/(1 + x)}` — the mirror image of the V family —
    and partner with V under the substitution :math:`x \mapsto -x`.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`W_n` is evaluated, of any real dtype.
    n : Tensor | Scalar
        Degree of :math:`W_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`W_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Half-angle definition

    .. math::

        W_n(\cos\theta) = \frac{\sin((n + \tfrac{1}{2})\theta)}
                                {\sin(\theta/2)},

    with the recurrence

    .. math::

        W_0(x) = 1, \quad W_1(x) = 2x + 1, \quad
        W_{k+1}(x) = 2x\, W_k(x) - W_{k-1}(x).

    Symmetry relation: :math:`W_n(-x) = (-1)^n V_n(x)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import chebyshev_polynomial_w
    >>> chebyshev_polynomial_w(lucid.tensor([-1.0, 0.0, 1.0]), n=2)
    tensor([1., -1., 5.])
    """
    return _ortho_poly(
        "chebyshev_polynomial_w", x, n, lambda v: 2 * v + 1, _chebyshev_step
    )


def shifted_chebyshev_polynomial_t(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Shifted Chebyshev T polynomial :math:`T^*_n(x)` on :math:`[0, 1]`.

    Maps the standard Chebyshev T family from :math:`[-1, 1]` onto the
    unit interval :math:`[0, 1]` via the linear substitution
    :math:`x \mapsto 2x - 1`.  Useful when the natural domain of an
    approximation problem is :math:`[0, 1]`.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`T^*_n` is evaluated, of any real dtype.  Customarily on the
        shifted domain :math:`[0, 1]`.
    n : Tensor | Scalar
        Degree of :math:`T^*_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`T^*_n(x) = T_n(2x - 1)`, element-wise, broadcast over ``x`` and
        ``n``.

    Notes
    -----
    Definition: :math:`T^*_n(x) = T_n(2x - 1)`.  Inherits orthogonality
    on :math:`[0, 1]` with weight :math:`1/\sqrt{x - x^2}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_t
    >>> shifted_chebyshev_polynomial_t(lucid.tensor([0.0, 0.5, 1.0]), n=2)
    tensor([1., -1., 1.])
    """
    op = "shifted_chebyshev_polynomial_t"
    return _ortho_poly(op, x, n, lambda v: v, _chebyshev_step, shift=True)


def shifted_chebyshev_polynomial_u(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Shifted Chebyshev U polynomial :math:`U^*_n(x)` on :math:`[0, 1]`.

    Standard Chebyshev U polynomials shifted from :math:`[-1, 1]` onto
    the unit interval by :math:`x \mapsto 2x - 1`.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`U^*_n` is evaluated, of any real dtype.  Customarily on
        :math:`[0, 1]`.
    n : Tensor | Scalar
        Degree of :math:`U^*_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`U^*_n(x) = U_n(2x - 1)`, element-wise, broadcast over ``x`` and
        ``n``.

    Notes
    -----
    :math:`U^*_n(x) = U_n(2x - 1)`.  Orthogonal on :math:`[0, 1]` with
    weight :math:`\sqrt{x - x^2}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_u
    >>> shifted_chebyshev_polynomial_u(lucid.tensor([0.0, 0.5, 1.0]), n=2)
    tensor([3., -1., 3.])
    """
    op = "shifted_chebyshev_polynomial_u"
    return _ortho_poly(op, x, n, lambda v: 2 * v, _chebyshev_step, shift=True)


def shifted_chebyshev_polynomial_v(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Shifted Chebyshev V polynomial :math:`V^*_n(x)` on :math:`[0, 1]`.

    Chebyshev V polynomials of the third kind composed with
    :math:`x \mapsto 2x - 1`, yielding a basis orthogonal on
    :math:`[0, 1]`.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`V^*_n` is evaluated, of any real dtype.  Customarily on
        :math:`[0, 1]`.
    n : Tensor | Scalar
        Degree of :math:`V^*_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`V^*_n(x) = V_n(2x - 1)`, element-wise, broadcast over ``x`` and
        ``n``.

    Notes
    -----
    :math:`V^*_n(x) = V_n(2x - 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_v
    >>> shifted_chebyshev_polynomial_v(lucid.tensor([0.0, 1.0]), n=1)
    tensor([-3., 1.])
    """
    op = "shifted_chebyshev_polynomial_v"
    return _ortho_poly(op, x, n, lambda v: 2 * v - 1, _chebyshev_step, shift=True)


def shifted_chebyshev_polynomial_w(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Shifted Chebyshev W polynomial :math:`W^*_n(x)` on :math:`[0, 1]`.

    Chebyshev W polynomials of the fourth kind composed with
    :math:`x \mapsto 2x - 1`.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`W^*_n` is evaluated, of any real dtype.  Customarily on
        :math:`[0, 1]`.
    n : Tensor | Scalar
        Degree of :math:`W^*_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`W^*_n(x) = W_n(2x - 1)`, element-wise, broadcast over ``x`` and
        ``n``.

    Notes
    -----
    :math:`W^*_n(x) = W_n(2x - 1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import shifted_chebyshev_polynomial_w
    >>> shifted_chebyshev_polynomial_w(lucid.tensor([0.0, 1.0]), n=1)
    tensor([-1., 3.])
    """
    op = "shifted_chebyshev_polynomial_w"
    return _ortho_poly(op, x, n, lambda v: 2 * v + 1, _chebyshev_step, shift=True)


def hermite_polynomial_h(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Physicist's Hermite polynomial :math:`H_n(x)`.

    The physicists' Hermite polynomials are orthogonal on
    :math:`(-\infty, \infty)` with weight :math:`e^{-x^2}` and form the
    eigenfunctions of the quantum harmonic oscillator (in conjunction
    with the Gaussian envelope).

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`H_n` is evaluated, of any real dtype.
    n : Tensor | Scalar
        Degree of :math:`H_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`H_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Rodrigues formula and recurrence:

    .. math::

        H_n(x) = (-1)^n e^{x^2}
                 \frac{d^n}{dx^n} e^{-x^2}, \qquad
        H_{k+1}(x) = 2x\, H_k(x) - 2k\, H_{k-1}(x).

    Starting values: :math:`H_0(x) = 1`, :math:`H_1(x) = 2x`.  Linked to
    the probabilists' Hermite polynomials by :math:`H_n(x) =
    2^{n/2} He_n(\sqrt{2}\, x)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import hermite_polynomial_h
    >>> hermite_polynomial_h(lucid.tensor([-1.0, 0.0, 1.0]), n=3)
    tensor([4., -0., -4.])
    """
    return _ortho_poly(
        "hermite_polynomial_h",
        x,
        n,
        lambda v: 2 * v,
        lambda v, p, q, k: 2 * v * p - 2 * float(k) * q,
    )


def hermite_polynomial_he(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Probabilist's Hermite polynomial :math:`\mathit{He}_n(x)`.

    Probabilists' Hermite polynomials are orthogonal on
    :math:`(-\infty, \infty)` with weight :math:`e^{-x^2/2} /
    \sqrt{2\pi}` — the standard normal density — and are the natural
    basis for Hermite-expansions of functions against a Gaussian
    measure (Wiener chaos, Gauss-Hermite quadrature with weight 1).

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`\mathit{He}_n` is evaluated, of any real dtype.
    n : Tensor | Scalar
        Degree of :math:`\mathit{He}_n`, for every element or one per element;
        truncated toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`\mathit{He}_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Recurrence:

    .. math::

        \mathit{He}_0(x) = 1, \quad \mathit{He}_1(x) = x, \quad
        \mathit{He}_{k+1}(x) = x\, \mathit{He}_k(x)
                             - k\, \mathit{He}_{k-1}(x).

    Connection to physicists' Hermite: :math:`\mathit{He}_n(x) =
    2^{-n/2} H_n(x/\sqrt 2)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import hermite_polynomial_he
    >>> hermite_polynomial_he(lucid.tensor([-1.0, 0.0, 1.0]), n=3)
    tensor([2., -0., -2.])
    """
    return _ortho_poly(
        "hermite_polynomial_he",
        x,
        n,
        lambda v: v,
        lambda v, p, q, k: v * p - float(k) * q,
    )


def legendre_polynomial_p(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Legendre polynomial :math:`P_n(x)`.

    The Legendre polynomials are orthogonal on :math:`[-1, 1]` with
    uniform weight, and serve as the angular eigenfunctions of the
    Laplacian on the sphere (spherical harmonics with :math:`m = 0`)
    and as a basis for Gauss-Legendre quadrature.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`P_n` is evaluated, of any real dtype.  Customarily on
        :math:`[-1, 1]`.
    n : Tensor | Scalar
        Degree of :math:`P_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`P_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Bonnet recurrence:

    .. math::

        P_0(x) = 1, \quad P_1(x) = x, \quad
        (k+1)\, P_{k+1}(x) = (2k + 1)\, x\, P_k(x) - k\, P_{k-1}(x).

    On :math:`[-1, 1]`, :math:`|P_n(x)| \le 1` with boundary values
    :math:`P_n(1) = 1`, :math:`P_n(-1) = (-1)^n`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import legendre_polynomial_p
    >>> legendre_polynomial_p(lucid.tensor([-1.0, 0.0, 0.5, 1.0]), n=3)
    tensor([-1., -0., -0.4375, 1.])
    """
    return _ortho_poly(
        "legendre_polynomial_p",
        x,
        n,
        lambda v: v,
        lambda v, p, q, k: ((2 * k + 1) * v * p - float(k) * q) / float(k + 1),
    )


def laguerre_polynomial_l(x: TensorOrScalar, n: TensorOrScalar) -> Tensor:
    r"""Laguerre polynomial :math:`L_n(x)`.

    The (simple) Laguerre polynomials are orthogonal on
    :math:`[0, \infty)` with weight :math:`e^{-x}` and appear as the
    radial eigenfunctions of the hydrogen atom, as the basis of
    Gauss-Laguerre quadrature, and in survival-analysis kernels.

    Parameters
    ----------
    x : Tensor | Scalar
        Where :math:`L_n` is evaluated, of any real dtype.  Customarily on
        :math:`[0, \infty)`.
    n : Tensor | Scalar
        Degree of :math:`L_n`, for every element or one per element; truncated
        toward zero, and a negative degree gives 0.

    Returns
    -------
    Tensor
        :math:`L_n(x)` element-wise, broadcast over ``x`` and ``n``.

    Notes
    -----
    Recurrence:

    .. math::

        L_0(x) = 1, \quad L_1(x) = 1 - x, \quad
        (k+1)\, L_{k+1}(x) = (2k + 1 - x)\, L_k(x)
                           - k\, L_{k-1}(x).

    The leading coefficient is :math:`(-x)^n / n!`, and
    :math:`L_n(0) = 1` for every ``n``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.special import laguerre_polynomial_l
    >>> laguerre_polynomial_l(lucid.tensor([0.0, 1.0, 2.0]), n=2)
    tensor([1., -0.5, -1.])
    """
    return _ortho_poly(
        "laguerre_polynomial_l",
        x,
        n,
        lambda v: 1 - v,
        lambda v, p, q, k: ((2 * k + 1 - v) * p - float(k) * q) / float(k + 1),
    )


# Documented and listed under ``lucid.special``, the one path they are
# reached by (H8), like the functions defined there.
for _fn in (
    chebyshev_polynomial_t,
    chebyshev_polynomial_u,
    chebyshev_polynomial_v,
    chebyshev_polynomial_w,
    shifted_chebyshev_polynomial_t,
    shifted_chebyshev_polynomial_u,
    shifted_chebyshev_polynomial_v,
    shifted_chebyshev_polynomial_w,
    hermite_polynomial_h,
    hermite_polynomial_he,
    legendre_polynomial_p,
    laguerre_polynomial_l,
):
    _fn.__module__ = "lucid.special"
del _fn
