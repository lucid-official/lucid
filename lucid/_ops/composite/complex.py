"""Complex-number composite ops: ``angle``, ``polar``, ``view_as_real``,
``view_as_complex``.

The primitives ``real`` / ``imag`` / ``view_as_real`` / ``view_as_complex`` /
``complex`` / ``conj`` are engine ops.  ``real``, ``imag``, ``view_as_real``,
and ``view_as_complex`` return zero-copy storage views on both CPU and Metal.

* ``angle(c) = atan2(imag(c), real(c))`` — always in ``[-π, π]``.
* ``polar(abs, angle) = complex(abs * cos(angle), abs * sin(angle))``.
* ``view_as_real(c) -> real(..., 2)`` — zero-copy view exposing ``[real, imag]``
  pairs along a trailing size-2 axis.
* ``view_as_complex(real(..., 2)) -> complex(...)`` — zero-copy inverse view.
"""

import math
from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _float_input, _unwrap, _wrap
from lucid._ops.composite._opmath import _promote_pair
from lucid._types import TensorOrScalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def angle(input: Tensor) -> Tensor:
    r"""Phase angle (argument) of each element.

    Computes the polar-coordinate angle of each complex element:
    :math:`\arg(z) = \arctan2(\Im(z), \Re(z))`, in radians on the
    interval :math:`[-\pi, \pi]`.  A real element is a complex number on
    the real axis: its angle is :math:`\pi` when it is negative and 0
    otherwise, and NaN stays NaN.

    Parameters
    ----------
    input : Tensor
        Complex or real tensor.  A bool or integer one is taken at the
        default floating dtype.

    Returns
    -------
    Tensor
        Real tensor with the same shape as ``input``: ``float32`` for
        ``complex64``, ``float64`` for ``complex128``, a real input's own
        floating dtype.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \arg(z_i) = \arctan2(\Im(z_i), \Re(z_i))

    Uses quadrant-correct :func:`lucid.atan2` internally — unlike a naive
    ``arctan(imag / real)``, this returns the proper four-quadrant angle.
    Zero complex inputs yield 0.0 (not NaN).

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+0j, 1+1j, 0+1j, -1+0j])
    >>> lucid.angle(z)
    tensor([0., 0.7854, 1.571, 3.142])
    >>> lucid.angle(lucid.tensor([-2, 3]))
    tensor([3.142, 0.])
    """
    x = _float_input(input)
    if lucid.is_complex(x):
        return lucid.atan2(lucid.imag(x), lucid.real(x))
    half_turn = lucid.where(x < 0.0, lucid.full_like(x, math.pi), lucid.zeros_like(x))
    return lucid.where(lucid.isnan(x), x, half_turn)


def polar(abs: TensorOrScalar, angle: TensorOrScalar) -> Tensor:
    r"""Build a complex tensor from polar coordinates (magnitude, phase).

    Constructs each complex output entry as
    :math:`z = r e^{i\theta} = r\cos\theta + i\,r\sin\theta`, where ``r``
    comes from ``abs`` and :math:`\theta` from ``angle``.  Both inputs are
    real-valued and follow standard broadcasting rules.

    Parameters
    ----------
    abs : Tensor
        Real F32 tensor of magnitudes (non-negative values are typical
        but not enforced — a negative magnitude simply rotates the phase
        by :math:`\pi`).
    angle : Tensor
        Real F32 tensor of phase angles, in radians. Must broadcast with
        ``abs``.

    Returns
    -------
    Tensor
        ``complex64`` tensor whose shape is the broadcast shape of the two
        inputs.

    Notes
    -----
    Mathematical definition:

    .. math::

        z = r\,e^{i\theta} = r\cos\theta + i\,r\sin\theta.

    Inverse of the pair :func:`lucid.abs` / :func:`angle` for complex
    tensors: ``polar(abs(z), angle(z))`` reconstructs ``z`` up to
    floating-point round-off.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> r = lucid.tensor([1.0, 2.0])
    >>> th = lucid.tensor([0.0, math.pi / 3])
    >>> lucid.polar(r, th)
    tensor([(1.+0.j), (1.+1.732j)], dtype=lucid.complex64)
    """
    abs, angle = _promote_pair(abs, angle)
    return lucid.complex(abs * lucid.cos(angle), abs * lucid.sin(angle))


def view_as_real(input: Tensor) -> Tensor:
    r"""Reinterpret a complex tensor as a real tensor with a trailing pair axis.

    Given an input of shape :math:`(\dots)` and dtype ``complex64`` or
    ``complex128``, returns a zero-copy ``float32`` or ``float64`` view of
    shape :math:`(\dots, 2)` where the trailing axis carries ``[real, imag]``
    pairs.

    Parameters
    ----------
    input : Tensor
        Complex-valued tensor (dtype ``complex64`` or ``complex128``).

    Returns
    -------
    Tensor
        Real view of shape ``input.shape + (2,)`` sharing storage with
        ``input``.

    Notes
    -----
    Conceptually this is the map

    .. math::

        z_i = a_i + i\,b_i \;\mapsto\; (a_i, b_i).

    Returns a zero-copy storage view on both CPU and Metal: in-place writes
    into the returned tensor reach ``input`` and vice versa, with stride
    ``(*(2 * s for s in input.stride()), 1)`` and ``2 * input.storage_offset()``.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3+4j])
    >>> lucid.view_as_real(z)
    tensor([[1., 2.], [3., 4.]])
    """
    return _wrap(_C_engine.view_as_real(_unwrap(input)))


def view_as_complex(input: Tensor) -> Tensor:
    r"""Reinterpret a real tensor with a trailing pair axis as a complex tensor.

    Inverse of :func:`view_as_real`.  Given a real input of shape
    :math:`(\dots, 2)`, returns a zero-copy ``complex64`` or ``complex128``
    view of shape :math:`(\dots)` whose last axis is consumed.

    Parameters
    ----------
    input : Tensor
        Real tensor whose final dimension has size 2 with stride 1 and even
        strides / storage offset along all preceding dimensions; the two
        components are interpreted as ``(real, imag)``.

    Returns
    -------
    Tensor
        Complex view of shape ``input.shape[:-1]`` sharing storage with
        ``input``.

    Raises
    ------
    ValueError
        If ``input.ndim < 1``, the last axis does not have size 2 or unit
        stride, or any non-singleton preceding dimension stride or the
        storage offset is odd.

    Notes
    -----
    Conceptually:

    .. math::

        (a_i, b_i) \;\mapsto\; z_i = a_i + i\,b_i.

    Returns a zero-copy storage view on both CPU and Metal: in-place writes
    into the returned tensor reach ``input`` and vice versa, with stride
    ``tuple(s // 2 for s in input.stride()[:-1])`` and
    ``input.storage_offset() // 2``.

    Examples
    --------
    >>> import lucid
    >>> r = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> lucid.view_as_complex(r)
    tensor([(1.+2.j), (3.+4.j)], dtype=lucid.complex64)
    """
    return _wrap(_C_engine.view_as_complex(_unwrap(input)))


__all__ = ["angle", "polar", "view_as_real", "view_as_complex"]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
