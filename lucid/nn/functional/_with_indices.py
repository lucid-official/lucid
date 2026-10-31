"""The ``*_with_indices`` pooling functions, generated from the pools they wrap.

The reference exposes each max pool twice: ``max_pool2d(..., return_indices=True)``
and ``max_pool2d_with_indices(...)``, the latter always returning
``(output, indices)`` and accepting ``return_indices`` only so the two share a
signature.  The second spelling is the first with the flag forced on, so it is
generated here from the pool rather than written out eight times; each family
(windowed, adaptive, fractional) has one signature and one factory.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from lucid.nn.functional.pooling import (
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
    max_pool1d,
    max_pool2d,
    max_pool3d,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class _WindowedWithIndices(Protocol):
    """``max_pool{1,2,3}d_with_indices``."""

    def __call__(
        self,
        input: Tensor,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> tuple[Tensor, Tensor]: ...


class _AdaptiveWithIndices(Protocol):
    """``adaptive_max_pool{1,2,3}d_with_indices``."""

    def __call__(
        self, input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
    ) -> tuple[Tensor, Tensor]: ...


class _FractionalWithIndices(Protocol):
    """``fractional_max_pool{2,3}d_with_indices``."""

    def __call__(
        self,
        input: Tensor,
        kernel_size: int | tuple[int, ...],
        output_size: int | tuple[int, ...] | None = None,
        output_ratio: float | tuple[float, ...] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


def _doc(pool_name: str) -> str:
    """The shared docstring of a generated variant of ``pool_name``."""
    return (
        f"Run :func:`{pool_name}` and return ``(output, indices)``.\n\n"
        f":func:`{pool_name}` with ``return_indices=True``; see it for the other\n"
        "parameters, the shape of the result and the meaning of the indices.\n\n"
        "Parameters\n"
        "----------\n"
        "return_indices : bool, optional\n"
        f"    Accepted so the signature matches ``{pool_name}``'s, and ignored:\n"
        "    the indices are always returned.\n\n"
        "Returns\n"
        "-------\n"
        "tuple of Tensor\n"
        "    The pooled values and the position of each one's source.\n"
    )


def _named(fn: Callable[..., object], pool_name: str) -> None:
    """Give a generated function the name, qualified name and docstring of its public spelling."""
    name = f"{pool_name}_with_indices"
    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = _doc(pool_name)


def _windowed(pool: Callable[..., tuple[Tensor, Tensor]], pool_name: str) -> _WindowedWithIndices:
    """``pool`` with ``return_indices`` forced on."""

    def variant(
        input: Tensor,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del return_indices  # signature parity: the indices are the point
        return pool(
            input, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode, return_indices=True
        )

    _named(variant, pool_name)
    return variant


def _adaptive(pool: Callable[..., tuple[Tensor, Tensor]], pool_name: str) -> _AdaptiveWithIndices:
    """``pool`` with ``return_indices`` forced on."""

    def variant(
        input: Tensor, output_size: int | tuple[int | None, ...], return_indices: bool = False
    ) -> tuple[Tensor, Tensor]:
        del return_indices  # signature parity: the indices are the point
        return pool(input, output_size, return_indices=True)

    _named(variant, pool_name)
    return variant


def _fractional(
    pool: Callable[..., tuple[Tensor, Tensor]], pool_name: str
) -> _FractionalWithIndices:
    """``pool`` with ``return_indices`` forced on."""

    def variant(
        input: Tensor,
        kernel_size: int | tuple[int, ...],
        output_size: int | tuple[int, ...] | None = None,
        output_ratio: float | tuple[float, ...] | None = None,
        return_indices: bool = False,
        _random_samples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del return_indices  # signature parity: the indices are the point
        return pool(
            input,
            kernel_size,
            output_size,
            output_ratio,
            return_indices=True,
            _random_samples=_random_samples,
        )

    _named(variant, pool_name)
    return variant


max_pool1d_with_indices = _windowed(max_pool1d, "max_pool1d")
max_pool2d_with_indices = _windowed(max_pool2d, "max_pool2d")
max_pool3d_with_indices = _windowed(max_pool3d, "max_pool3d")

adaptive_max_pool1d_with_indices = _adaptive(adaptive_max_pool1d, "adaptive_max_pool1d")
adaptive_max_pool2d_with_indices = _adaptive(adaptive_max_pool2d, "adaptive_max_pool2d")
adaptive_max_pool3d_with_indices = _adaptive(adaptive_max_pool3d, "adaptive_max_pool3d")

fractional_max_pool2d_with_indices = _fractional(fractional_max_pool2d, "fractional_max_pool2d")
fractional_max_pool3d_with_indices = _fractional(fractional_max_pool3d, "fractional_max_pool3d")
