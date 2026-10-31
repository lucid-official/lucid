r"""The LoHa layers: a frozen base module with a Hadamard product of low-rank products beside it.

``LoHaLayer`` is the family root.  An adapter's update is

.. math:: \Delta W = s\,(W_{1a} W_{1b}) \odot (W_{2a} W_{2b}), \qquad s = \alpha / r,

reshaped to the base weight; a convolution may instead carry spatial cores
(``hada_t1`` / ``hada_t2``, ``use_effective_conv2d``), each product becoming a
Tucker contraction of its core with two channel matrices.  The factors are
bare parameters in ``ParameterDict`` tables, so a state dict reads
``....hada_w1_a.default`` and an adapter file ``....hada_w1_a``, the reference's keys.
"""

from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._kinds import _Geometry, _tucker
from lucid.peft.tuners._lycoris import _LycorisConfig, _LycorisLayer


class LoHaLayer(_LycorisLayer, family=True):
    r"""Adapter bookkeeping and the update shared by the LoHa layers.

    Per adapter: ``hada_w1_a`` / ``hada_w1_b`` / ``hada_w2_a`` / ``hada_w2_b``
    — for a linear layer (or a flattened kernel) ``(out, r)``, ``(r, in)``,
    ``(out, r)``, ``(r, in)`` — and, with a spatial core, ``hada_t1`` /
    ``hada_t2`` of ``(r, r, *kernel)`` with the matrices ``(r, out)`` and
    ``(r, in)``.  Rank dropout always rescales the rows it keeps (the
    reference's LoHa does), so ``rank_dropout_scale`` reads ``True``.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in every table.
    config : LoHaConfig
        The adapter's config (dropouts, ``init_weights``, ``use_effective_conv2d``).
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a positive rank.
    alpha : float, default=0.0
        Numerator of the scale.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
    )
    _repr_prefix = "loha."
    hada_w1_a: nn.ParameterDict
    hada_w1_b: nn.ParameterDict
    hada_w2_a: nn.ParameterDict
    hada_w2_b: nn.ParameterDict
    hada_t1: nn.ParameterDict
    hada_t2: nn.ParameterDict

    @override
    def _factor_shapes(
        self, r: int, geometry: _Geometry, config: _LycorisConfig, effective: bool
    ) -> dict[str, tuple[int, ...]]:
        del config  # LoHa reads nothing more of it
        if effective:
            # A 1-d kernel rides a trailing axis of 1, the layout the reference's keys have.
            core = (r, r, *geometry.kernel, *([1] if len(geometry.kernel) == 1 else []))
            channels = {"w1_a": (r, geometry.out), "w1_b": (r, geometry.inn)}
            channels |= {"w2_a": (r, geometry.out), "w2_b": (r, geometry.inn)}
            shapes = {f"hada_{name}": shape for name, shape in channels.items()}
            return shapes | {"hada_t1": core, "hada_t2": core}
        return {
            "hada_w1_a": (geometry.out, r),
            "hada_w1_b": (r, geometry.flat_in),
            "hada_w2_a": (geometry.out, r),
            "hada_w2_b": (r, geometry.flat_in),
        }

    @override
    def _initialise(self, adapter: str, init_weights: object) -> None:
        # The product is zero as soon as one factor of the second half is, so
        # that factor alone starts at zero (and the adapter does nothing).
        self._kaiming(
            adapter, self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_t1, self.hada_t2
        )
        (self._zero if init_weights else self._kaiming)(adapter, self.hada_w2_b)

    @override
    def _scaled_delta(self, adapter: str) -> Tensor:
        w1a, w1b = (_parameter_at(t, adapter) for t in (self.hada_w1_a, self.hada_w1_b))
        w2a, w2b = (_parameter_at(t, adapter) for t in (self.hada_w2_a, self.hada_w2_b))
        if adapter in self.hada_t1:
            first = _tucker(_parameter_at(self.hada_t1, adapter), w1a, w1b)
            second = _tucker(_parameter_at(self.hada_t2, adapter), w2a, w2b)
        else:
            first, second = w1a @ w1b, w2a @ w2b
        return first * second * self._specs[adapter].scaling


class Linear(LoHaLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a LoHa update: :math:`y = W x + b + s\,((W_{1a}W_{1b}) \odot (W_{2a}W_{2b}))\,x`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`LoHaLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoHaConfig
    >>> from lucid.peft.tuners.loha import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", LoHaConfig(target_modules=["x"]), r=2, alpha=4)
    >>> layer.scaling["default"], tuple(layer.hada_w1_a["default"].shape)
    (2.0, (4, 2))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # hada_w2_b starts at zero
    True
    """

    base_layer: nn.Linear


class Conv1d(LoHaLayer, base_type=nn.Conv1d):
    """An ``nn.Conv1d`` with a LoHa update: the factored kernel is convolved with the base's stride, padding and groups.

    Parameters
    ----------
    base_layer : nn.Conv1d
        The layer to adapt; the other parameters are those of :class:`LoHaLayer`.
    """

    base_layer: nn.Conv1d


class Conv2d(LoHaLayer, base_type=nn.Conv2d):
    """An ``nn.Conv2d`` with a LoHa update: the factored kernel is convolved with the base's stride, padding and groups.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`LoHaLayer`.
    """

    base_layer: nn.Conv2d
