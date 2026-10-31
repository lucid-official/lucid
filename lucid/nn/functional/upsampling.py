"""
nn.functional's deprecated spellings of :func:`interpolate`.

The reference keeps ``upsample``, ``upsample_nearest`` and ``upsample_bilinear``
for old code, each a warning plus one call to ``interpolate``.  They are here
for the same code, with the reference's signatures; new code calls
:func:`~lucid.nn.functional.interpolate`.
"""

import warnings
from typing import TYPE_CHECKING

from lucid.nn.functional.sampling import interpolate

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


type _Size = int | tuple[int, ...] | None
type _Scale = float | tuple[float, ...] | None


def _warn_use_interpolate(name: str) -> None:
    """Say, as the reference does, that ``name`` is only an alias of ``interpolate``."""
    warnings.warn(
        f"`nn.functional.{name}` is deprecated. Use `nn.functional.interpolate` instead.",
        UserWarning,
        stacklevel=3,
    )


def upsample(
    input: Tensor,
    size: _Size = None,
    scale_factor: _Scale = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    r"""Resample ``input`` to ``size`` or by ``scale_factor`` (deprecated).

    Deprecated in favour of :func:`interpolate`, which it calls; it warns on
    every call, as the reference's does.

    Parameters
    ----------
    input : Tensor
        Input of shape ``(N, C, *spatial)``.
    size : int or tuple of int, optional
        Target spatial size.
    scale_factor : float or tuple of float, optional
        Multiplier for the spatial size.
    mode : str, optional
        The resampling kernel: ``'nearest'``, ``'linear'``, ``'bilinear'``,
        ``'bicubic'``, ``'trilinear'`` ...  Default: ``'nearest'``.
    align_corners : bool, optional
        Whether corner samples line up; see :func:`interpolate`.

    Returns
    -------
    Tensor
        The resampled tensor.

    Examples
    --------
    >>> import warnings
    >>> import lucid
    >>> from lucid.nn.functional import upsample
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = upsample(lucid.randn(1, 3, 4, 4), scale_factor=2)
    >>> y.shape
    (1, 3, 8, 8)
    """
    _warn_use_interpolate("upsample")
    return interpolate(input, size, scale_factor, mode, align_corners)


def upsample_nearest(input: Tensor, size: _Size = None, scale_factor: _Scale = None) -> Tensor:
    r"""Resample ``input`` by nearest neighbours (deprecated).

    ``interpolate(input, size, scale_factor, mode='nearest')``, which it
    calls; it warns on every call, as the reference's does.

    Parameters
    ----------
    input : Tensor
        A batch of 1-, 2- or 3-D signals, ``(N, C, *spatial)``.
    size : int or tuple of int, optional
        The output's spatial extent; give this or ``scale_factor``.
    scale_factor : float or tuple of float, optional
        How many times larger the output is along each spatial axis.

    Returns
    -------
    Tensor
        ``input`` with every pixel repeated to fill the new extent.

    Examples
    --------
    >>> import warnings
    >>> import lucid
    >>> from lucid.nn.functional import upsample_nearest
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = upsample_nearest(lucid.randn(1, 3, 4, 4), scale_factor=2)
    >>> y.shape
    (1, 3, 8, 8)
    """
    _warn_use_interpolate("upsample_nearest")
    return interpolate(input, size, scale_factor, mode="nearest")


def upsample_bilinear(input: Tensor, size: _Size = None, scale_factor: _Scale = None) -> Tensor:
    r"""Resample a 4-D ``input`` bilinearly, with aligned corners (deprecated).

    ``interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)``,
    which it calls; it warns on every call, as the reference's does.

    Parameters
    ----------
    input : Tensor
        Images, ``(N, C, H, W)``.
    size : int or tuple of int, optional
        Height and width of the output.
    scale_factor : float or tuple of float, optional
        Scale of height and width.

    Returns
    -------
    Tensor
        ``(N, C, H_out, W_out)``, with the corner pixels preserved.

    Examples
    --------
    >>> import warnings
    >>> import lucid
    >>> from lucid.nn.functional import upsample_bilinear
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = upsample_bilinear(lucid.randn(1, 3, 4, 4), scale_factor=2)
    >>> y.shape
    (1, 3, 8, 8)
    """
    _warn_use_interpolate("upsample_bilinear")
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)
