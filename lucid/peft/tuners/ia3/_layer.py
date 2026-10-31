r"""The (IA)^3 layers: a frozen base module whose activations a learned vector rescales.

``IA3Layer`` is the family root.  A layer holds one vector per adapter in
``ia3_l`` (``ParameterDict``, so a state dict reads ``....ia3_l.default``
and an adapter file ``....ia3_l``, the reference's key).  A layer is
*attention-style* (the vector scales the base **output**,
:math:`y = \ell \odot f(x)`) or *feed-forward* (it scales the **input**,
:math:`y = f(\ell \odot x)`); that is the layer's ``is_feedforward``, fixed when
the layer is made, and the vector's shape follows it — ``(out, 1)`` or
``(1, in)`` for a linear layer, ``(1, channels, 1, ...)`` for a convolution.
Several active adapters multiply.

Because the update is multiplicative it folds into the weight (and, for an
output-side vector, the bias) by a product, and ``unmerge`` divides it back
out — which is exact only up to rounding, as the reference warns.
"""

import warnings
from abc import abstractmethod
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _BaseTunerLayer, _check_adapter_name, _parameter_at
from lucid.peft.tuners.ia3._config import IA3Config

_WEIGHT = "base_layer.weight"
_BIAS = "base_layer.bias"
#: Added to a vector before dividing a merged weight by it, so a zero entry does not divide by zero.
_UNMERGE_EPS = 1e-8


class IA3Layer(_BaseTunerLayer):
    r"""Adapter bookkeeping shared by the (IA)^3 layers.

    Holds the wrapped layer as ``base_layer`` — the very module, so its
    parameters stay the same ``Parameter`` objects — and, per adapter name, a
    vector in ``ia3_l``.  A layer holds any number of adapters, and the
    forward pass applies the product of the ones named in ``active_adapter``
    that it holds.

    Parameters
    ----------
    base_layer : nn.Module
        The layer to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    config : IA3Config
        The config; ``init_ia3_weights`` is read.
    is_feedforward : bool, default=False
        Scale the layer's input (a feed-forward module) instead of its output.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.fan_in_fan_out`` is set (the layer's weight is stored ``(out, in)``).
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("ia3_l",)

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: IA3Config,
        is_feedforward: bool = False,
    ) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super().__init__(base_layer)
        self.is_feedforward = is_feedforward
        self.ia3_l: nn.ParameterDict = nn.ParameterDict()
        self.update_layer(adapter_name, config)

    # ── what a layer type supplies ────────────────────────────────────────

    @abstractmethod
    def _features(self) -> tuple[int, int]:
        """``(in, out)`` — the sizes the vector can be over."""

    @abstractmethod
    def _vector_shape(self, size: int) -> tuple[int, ...]:
        """The shape of a vector over ``size`` features."""

    @abstractmethod
    def _activation_scale(self, vector: Tensor) -> Tensor:
        """``vector`` shaped to multiply an activation (the layer's input or output)."""

    @abstractmethod
    def _weight_scale(self, vector: Tensor) -> Tensor:
        """``vector`` shaped to multiply the base weight, with the same effect as on the activation."""

    # ── the layer ─────────────────────────────────────────────────────────

    @property
    def in_features(self) -> int:
        """Input features (channels) of the base layer."""
        return self._features()[0]

    @property
    def out_features(self) -> int:
        """Output features (channels) of the base layer."""
        return self._features()[1]

    def _vector(self, adapter: str) -> Parameter:
        return _parameter_at(self.ia3_l, adapter)

    @override
    def __repr__(self) -> str:
        return "ia3." + super().__repr__()

    @override
    def extra_repr(self) -> str:
        return (
            f"adapter={self.active_adapter!r}, is_feedforward={self.is_feedforward}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    def update_layer(self, adapter_name: str, config: IA3Config) -> None:
        """Create adapter ``adapter_name``: one vector, ones unless ``config.init_ia3_weights`` is off.

        The vector lives on the base weight's device and has its dtype.  A layer that holds an
        adapter already keeps applying the adapters it applied: the new one is inactive until
        ``set_adapter`` names it.  A layer with none starts applying this one.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in ``ia3_l``.
        config : IA3Config
            Read for ``init_ia3_weights``.

        Raises
        ------
        ValueError
            The layer already holds an adapter of this name.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.ia3_l:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        size = self.in_features if self.is_feedforward else self.out_features
        anchor = self.weight
        vector = Parameter(
            lucid.randn(*self._vector_shape(size), device=anchor.device, dtype=anchor.dtype)
        )
        if config.init_ia3_weights:
            init.ones_(vector)
        self.ia3_l[adapter_name] = vector
        if not self.active_adapters:
            self.active_adapter = adapter_name

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.disable_adapters or self.merged:
            result: Tensor = self.base_layer(x)
            return result
        scale: Tensor | None = None
        for adapter in self.active_adapters:
            if adapter in self.ia3_l:
                vector = self._activation_scale(self._vector(adapter))
                scale = vector if scale is None else scale * vector
        if scale is None:
            result = self.base_layer(x)
        elif self.is_feedforward:
            scaled = x * scale
            result = self.base_layer(scaled if scaled.dtype == x.dtype else scaled.to(x.dtype))
        else:
            result = self.base_layer(x)
            scaled = result * scale
            result = scaled if scaled.dtype == result.dtype else scaled.to(result.dtype)
        return result

    # ── merging ───────────────────────────────────────────────────────────

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`(\ell' - 1) \odot W`, the change merging ``adapter`` makes to the weight.

        The update is multiplicative, so this is what the weight gains rather than a term to add
        to it in the forward pass.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight.
        """
        weight = self.weight
        return weight * self._weight_scale(self._vector(adapter)) - weight

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        targets = {_WEIGHT: self.weight}
        bias = self.bias
        if bias is not None and not self.is_feedforward:
            targets[_BIAS] = bias
        return targets

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        vector = self._vector(adapter)
        values = {_WEIGHT: self.weight * self._weight_scale(vector)}
        bias = self.bias
        if bias is not None and not self.is_feedforward:
            values[_BIAS] = bias * vector.reshape(*bias.shape)
        return values

    @override
    def _fold(self, adapter: str) -> None:
        targets = self._merge_targets()
        for key, value in self._merged_values(adapter).items():
            targets[key].copy_(value)

    @override
    def _unfold(self) -> None:
        """Divide every merged vector back out of the base weight (and bias).

        Exact only up to rounding: a vector entry near zero amplifies the error, and the division
        adds a tolerance of ``1e-8`` so a zero entry does not divide by zero.  Warns, as the
        reference does.
        """
        warnings.warn("Unmerge result can be inaccurate for (IA)^3.", stacklevel=4)
        targets = self._merge_targets()
        for adapter in reversed(self.merged_adapters):
            vector = self._vector(adapter)
            weight = targets[_WEIGHT]
            weight.copy_(weight / (self._weight_scale(vector) + _UNMERGE_EPS))
            if _BIAS in targets:
                bias = targets[_BIAS]
                bias.copy_(bias / (vector.reshape(*bias.shape) + _UNMERGE_EPS))


class Linear(IA3Layer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with an (IA)^3 vector: :math:`y = \ell \odot (W x + b)`, or :math:`y = W(\ell \odot x) + b`.

    The vector is ``(out_features, 1)`` for an output-side layer and ``(1, in_features)`` for a
    feed-forward one, as in the reference; merging multiplies it into the weight's rows or
    columns.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``ia3_l``.
    config : IA3Config
        The adapter's config.
    is_feedforward : bool, default=False
        Scale the input instead of the output.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import IA3Config
    >>> from lucid.peft.tuners.ia3 import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", IA3Config(target_modules=["x"]))
    >>> tuple(layer.ia3_l["default"].shape)
    (4, 1)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the vector starts at ones
    True
    """

    base_layer: nn.Linear

    @override
    def _features(self) -> tuple[int, int]:
        return self.base_layer.in_features, self.base_layer.out_features

    @override
    def _vector_shape(self, size: int) -> tuple[int, ...]:
        return (1, size) if self.is_feedforward else (size, 1)

    @override
    def _activation_scale(self, vector: Tensor) -> Tensor:
        return vector.flatten()

    @override
    def _weight_scale(self, vector: Tensor) -> Tensor:
        return vector  # (out, 1) scales rows and (1, in) columns of the (out, in) weight


class _ConvNd(IA3Layer):
    """What the (IA)^3 convolutions share: a ``(1, channels, 1, ...)`` vector over the channel axis."""

    base_layer: nn.Conv2d | nn.Conv3d

    @override
    def _features(self) -> tuple[int, int]:
        return self.base_layer.in_channels, self.base_layer.out_channels

    @override
    def _vector_shape(self, size: int) -> tuple[int, ...]:
        return (1, size, *([1] * (self.base_layer.weight.ndim - 2)))

    @override
    def _activation_scale(self, vector: Tensor) -> Tensor:
        return vector  # channels are axis 1 of an activation

    @override
    def _weight_scale(self, vector: Tensor) -> Tensor:
        if not self.is_feedforward:
            return vector.transpose(0, 1)  # (out, 1, ...) scales the filters
        groups = self.base_layer.groups
        if groups == 1:
            return vector  # (1, in, ...) scales the input channels of every filter
        # A grouped kernel has ``in / groups`` input channels per filter, each group's own slice.
        per_group_in = self.base_layer.in_channels // groups
        per_group_out = self.base_layer.out_channels // groups
        tail = tuple(vector.shape[2:])
        sliced = vector.reshape(groups, 1, per_group_in, *tail)
        spread = sliced.expand(groups, per_group_out, per_group_in, *tail)
        return spread.reshape(self.base_layer.out_channels, per_group_in, *tail)


class Conv2d(_ConvNd, base_type=nn.Conv2d):
    r"""An ``nn.Conv2d`` with an (IA)^3 vector over its channels: ``(1, channels, 1, 1)``.

    Output-side layers scale the output channels (:math:`\ell` is ``(1, out, 1, 1)``);
    feed-forward ones scale the input channels (``(1, in, 1, 1)``).  Merging multiplies the
    filters, or each filter's input channels (per group for a grouped convolution).

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`Linear`.
    """

    base_layer: nn.Conv2d


class Conv3d(_ConvNd, base_type=nn.Conv3d):
    r"""An ``nn.Conv3d`` with an (IA)^3 vector over its channels: ``(1, channels, 1, 1, 1)``.

    Parameters
    ----------
    base_layer : nn.Conv3d
        The layer to adapt; the other parameters are those of :class:`Linear`.
    """

    base_layer: nn.Conv3d
