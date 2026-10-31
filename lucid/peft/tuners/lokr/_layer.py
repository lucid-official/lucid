r"""The LoKr layers: a frozen base module with a Kronecker-product update beside it.

``LoKrLayer`` is the family root.  An adapter's update is

.. math:: \Delta W = s\,W_1 \otimes W_2, \qquad s = \alpha / r,

each of :math:`W_1`, :math:`W_2` either a full matrix (``lokr_w1``,
``lokr_w2``) or a low-rank product (``lokr_w1_a @ lokr_w1_b``,
``lokr_w2_a @ lokr_w2_b``); a convolution's :math:`W_2` may instead be a
Tucker contraction of a spatial core ``lokr_t2`` with two channel matrices.  The
output and input dimensions are each factored into two near-equal factors
(``factorization``), the first going to :math:`W_1` and the second to
:math:`W_2`.  Factors are bare parameters in ``ParameterDict`` tables, so an
adapter file reads ``....lokr_w1`` and so on, the reference's keys.
"""

import math
from typing import ClassVar, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._kinds import _Geometry, _tucker
from lucid.peft.tuners._lycoris import _LycorisConfig, _LycorisLayer
from lucid.peft.tuners.lokr._config import LoKrConfig


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Split ``dimension`` into two integers whose product it is, as close to equal (or to ``factor``) as possible.

    Parameters
    ----------
    dimension : int
        The number to factor.
    factor : int, default=-1
        If positive and it divides ``dimension``, the first number is ``factor``.
        Otherwise the search is for a divisor near the square root (``-1``) or
        no larger than ``factor``.

    Returns
    -------
    tuple of int
        ``(m, n)`` with ``m * n == dimension`` and ``m <= n``, unless ``factor`` divides ``dimension``
        (then ``(factor, dimension // factor)`` as given).

    Examples
    --------
    >>> from lucid.peft.tuners.lokr import factorization
    >>> factorization(256), factorization(128), factorization(127), factorization(128, factor=4)
    ((16, 16), (8, 16), (1, 127), (4, 32))
    """
    if factor > 0 and dimension % factor == 0:
        return factor, dimension // factor
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        m, n = n, m
    return m, n


def _kron(left: Tensor, right: Tensor) -> Tensor:
    """``kron(left, right)`` for a matrix ``left`` and a ``right`` with trailing spatial axes: ``left`` gets matching 1s."""
    return lucid.kron(left.reshape(*left.shape, *([1] * (right.ndim - left.ndim))), right)


class LoKrLayer(_LycorisLayer, family=True):
    r"""Adapter bookkeeping and the update shared by the LoKr layers.

    Per adapter, :math:`W_1` is ``lokr_w1`` (``(a, c)``) or ``lokr_w1_a`` /
    ``lokr_w1_b`` (``(a, r)``, ``(r, c)``); :math:`W_2` is ``lokr_w2``
    (``(b, d)`` plus the kernel for a convolution), ``lokr_w2_a`` /
    ``lokr_w2_b`` (``(b, r)``, ``(r, d·kernel)``), or with a spatial core
    ``lokr_t2`` (``(r, r, *kernel)``) and ``(r, b)``, ``(r, d)``.  Here
    :math:`d_{out} = ab` and :math:`d_{in} = cd` by ``factorization``.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in every table.
    config : LoKrConfig
        The adapter's config (dropouts, ``init_weights``, ``decompose_both``,
        ``decompose_factor``, ``use_effective_conv2d``).
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a positive rank.
    alpha : float, default=0.0
        Numerator of the scale.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    )
    _repr_prefix = "lokr."
    lokr_w1: nn.ParameterDict
    lokr_w1_a: nn.ParameterDict
    lokr_w1_b: nn.ParameterDict
    lokr_w2: nn.ParameterDict
    lokr_w2_a: nn.ParameterDict
    lokr_w2_b: nn.ParameterDict
    lokr_t2: nn.ParameterDict

    @staticmethod
    @override
    def _rank_dropout_scale(config: _LycorisConfig) -> bool:
        assert isinstance(config, LoKrConfig)
        return config.rank_dropout_scale

    @override
    def _factor_shapes(
        self, r: int, geometry: _Geometry, config: _LycorisConfig, effective: bool
    ) -> dict[str, tuple[int, ...]]:
        assert isinstance(config, LoKrConfig)
        out_l, out_k = factorization(geometry.out, config.decompose_factor)
        in_m, in_n = factorization(geometry.inn, config.decompose_factor)
        # A factor is a full matrix once the rank is at least half its larger side.
        full_first = not (config.decompose_both and r < max(out_l, in_m) / 2)
        full_second = r >= max(out_k, in_n) / 2
        shapes: dict[str, tuple[int, ...]]
        if full_first:
            shapes = {"lokr_w1": (out_l, in_m)}
        else:
            shapes = {"lokr_w1_a": (out_l, r), "lokr_w1_b": (r, in_m)}
        if full_second:
            return shapes | {"lokr_w2": (out_k, in_n, *geometry.kernel)}
        if effective:
            # A 1-d kernel rides a trailing axis of 1, the layout the reference's keys have.
            core = (r, r, *geometry.kernel, *([1] if len(geometry.kernel) == 1 else []))
            return shapes | {"lokr_t2": core, "lokr_w2_a": (r, out_k), "lokr_w2_b": (r, in_n)}
        return shapes | {
            "lokr_w2_a": (out_k, r),
            "lokr_w2_b": (r, in_n * math.prod(geometry.kernel)),
        }

    @override
    def _initialise(self, adapter: str, init_weights: object) -> None:
        # Which half is zero decides what the adapter does at the start; the
        # other is Kaiming-uniform, as is the spatial core.
        first = (self.lokr_w1, self.lokr_w1_a, self.lokr_w1_b)
        second = (self.lokr_w2, self.lokr_w2_a, self.lokr_w2_b)
        self._kaiming(adapter, self.lokr_t2)
        if init_weights is True:
            # Zero the first factor (for a low-rank one, its left matrix), the rest random.
            self._zero(adapter, self.lokr_w1, self.lokr_w1_a)
            self._kaiming(adapter, self.lokr_w1_b, *second)
        elif init_weights == "lycoris":
            self._kaiming(adapter, *first, self.lokr_w2_a)
            self._zero(adapter, self.lokr_w2, self.lokr_w2_b)
        else:
            self._kaiming(adapter, *first, *second)

    def _left(self, adapter: str) -> Tensor:
        """:math:`W_1`, whole or as the product of its two factors."""
        if adapter in self.lokr_w1:
            return _parameter_at(self.lokr_w1, adapter)
        return _parameter_at(self.lokr_w1_a, adapter) @ _parameter_at(self.lokr_w1_b, adapter)

    def _right(self, adapter: str) -> Tensor:
        """:math:`W_2`, whole, as a product, or from its spatial core."""
        if adapter in self.lokr_w2:
            return _parameter_at(self.lokr_w2, adapter)
        rows, columns = _parameter_at(self.lokr_w2_a, adapter), _parameter_at(
            self.lokr_w2_b, adapter
        )
        if adapter in self.lokr_t2:
            return _tucker(_parameter_at(self.lokr_t2, adapter), rows, columns)
        return rows @ columns

    @override
    def _scaled_delta(self, adapter: str) -> Tensor:
        return _kron(self._left(adapter), self._right(adapter)) * self._specs[adapter].scaling


class Linear(LoKrLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a LoKr update: :math:`y = W x + b + s\,(W_1 \otimes W_2)\,x`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`LoKrLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoKrConfig
    >>> from lucid.peft.tuners.lokr import Linear
    >>> base = nn.Linear(8, 6)
    >>> layer = Linear(base, "default", LoKrConfig(target_modules=["x"]), r=2, alpha=4)
    >>> tuple(layer.lokr_w1["default"].shape), tuple(layer.lokr_w2["default"].shape)
    ((2, 2), (3, 4))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # lokr_w1 starts at zero
    True
    """

    base_layer: nn.Linear


class Conv1d(LoKrLayer, base_type=nn.Conv1d):
    """An ``nn.Conv1d`` with a LoKr update: the Kronecker kernel is convolved with the base's stride, padding and groups.

    Parameters
    ----------
    base_layer : nn.Conv1d
        The layer to adapt; the other parameters are those of :class:`LoKrLayer`.
    """

    base_layer: nn.Conv1d


class Conv2d(LoKrLayer, base_type=nn.Conv2d):
    """An ``nn.Conv2d`` with a LoKr update: the Kronecker kernel is convolved with the base's stride, padding and groups.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`LoKrLayer`.
    """

    base_layer: nn.Conv2d
