"""``Tensor.__pos__``, which is :func:`lucid.positive`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def __pos__(self: Tensor) -> Tensor:
    r"""Unary plus: ``+self``, which is :func:`lucid.positive`.

    Returns
    -------
    Tensor
        ``self``, not a copy.

    Raises
    ------
    RuntimeError
        If ``self`` is a boolean tensor.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, -2.0, 3.0])
    >>> +a is a
    True
    """
    from lucid._ops.composite.elementwise import positive

    return positive(self)
