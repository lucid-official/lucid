r"""The RoAd layers: a frozen linear module whose output a learned 2-D rotation turns.

RoAd (Liao and Monz, 2024, arXiv:2409.00119, section 3.2) adapts :math:`z = R\,h`, with
:math:`h = W^\top x + b` the base layer's output and :math:`R` block diagonal with a
:math:`2 \times 2` rotation-and-scale block per pair of output features (Eq. 3).  Folded into
the layer, :math:`W' = R W` and :math:`b' = R b`, so a merged model costs what the base does.
The scales :math:`\alpha` start at one and the angles :math:`\theta` at zero (the identity),
and the blocks are not orthogonal once :math:`\alpha` moves, so unmerging inverts each block
(:meth:`~lucid.peft.tuners.road._rotation._Rotation.inverse`) rather than transposing it.

``RoadLayer`` is the family root.  An adapter holds ``road_theta`` and ``road_alpha`` (``out/2``
numbers each for ``road_1``, ``out`` for ``road_2``, ``2 out`` for ``road_4``), the reference's
keys, and the read-only mappings ``variant`` and ``group_size``.  With several active adapters
the rotations are applied in the order they are listed, :math:`R_2 R_1 h`, which is also the
order merging folds them in.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._layer import _dotted, _parameter_at
from lucid.peft.tuners._support import _adapter_dtype, _ConfigLayer, _LinearSized, _SpecField
from lucid.peft.tuners.road._config import RoadConfig
from lucid.peft.tuners.road._rotation import _Rotation

#: The standard deviations of the ``init_weights=False`` start (angles about zero, scales about one), the reference's.
_RANDOM_STD = 0.5
_BASE = "base_layer"


class _RoadSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the variant and
    the group size shape the graph.
    """

    variant: str
    group_size: int


class RoadLayer(_ConfigLayer[RoadConfig], _LinearSized, family=True):
    r"""Adapter bookkeeping and the rotation shared by the RoAd layers.

    Per adapter: ``road_theta`` and ``road_alpha`` and the read-only mappings ``variant`` and
    ``group_size``.  ``get_delta_weight`` is the dense matrix :math:`R` the adapter multiplies
    the output by; the layer itself never forms it.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose output is turned; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : RoadConfig
        ``variant``, ``group_size`` and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is some other module type.
    ValueError
        ``base_layer`` has parameters that only its first input will create, or its output width
        is not a multiple of ``group_size``.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("road_theta", "road_alpha")
    _repr_prefix = "road."
    road_theta: nn.ParameterDict
    road_alpha: nn.ParameterDict

    variant = _SpecField[str]("Variant of each adapter, by name (read-only).")
    group_size = _SpecField[int]("Group size of each adapter, by name (read-only).")

    @override
    def _enter(self, adapter_name: str, config: RoadConfig) -> None:
        if self.out_features % config.group_size != 0:
            raise ValueError(
                f"The out_features of the base layer must be divisible by group_size "
                f"({config.group_size}) when using RoadLayer."
            )
        size = {"road_1": 1, "road_2": 2, "road_4": 4}[config.variant] * self.out_features // 2
        anchor = self.weight
        device, dtype = anchor.device, _adapter_dtype(anchor)
        theta = Parameter(lucid.zeros(size, device=device, dtype=dtype))
        alpha = Parameter(lucid.ones(size, device=device, dtype=dtype))
        if not config.init_weights:
            init.normal_(theta, mean=0.0, std=_RANDOM_STD)
            init.normal_(alpha, mean=1.0, std=_RANDOM_STD)
        self._specs[adapter_name] = _RoadSpec(config.variant, config.group_size)
        self.road_theta[adapter_name] = theta
        self.road_alpha[adapter_name] = alpha

    # ── the rotation ──────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _RoadSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _RoadSpec)
        return spec

    def _rotation(self, adapter: str) -> _Rotation:
        """The coefficients of ``adapter``'s matrix :math:`R`, differentiable in its parameters."""
        spec = self._spec(adapter)
        return _Rotation.of(
            spec.variant,
            spec.group_size,
            _parameter_at(self.road_theta, adapter),
            _parameter_at(self.road_alpha, adapter),
        )

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the dense :math:`R`, which multiplies the layer's output (and so its weight and bias) from the left.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out, out)``, in the adapter's dtype.
        """
        return self._rotation(adapter).matrix()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """The base layer's output, turned by every active adapter in the order listed.

        The base output alone when adapters are disabled or merged.  The rotation runs in the
        adapters' dtype (float32 beside a half-precision base) and the result returns to the base
        output's.
        """
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        out = result
        held = self._adapter_names()
        for adapter in self.active_adapters:
            if adapter in held:  # an adapter aimed at other layers: this one runs its base
                rotation = self._rotation(adapter)
                out = rotation.apply(out.to(rotation.first.dtype))
        return out if out.dtype == result.dtype else out.to(result.dtype)

    # ── merging ───────────────────────────────────────────────────────────

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        targets = {_dotted(_BASE, "weight"): self.weight}
        bias = self.bias
        if isinstance(bias, Parameter):
            targets[_dotted(_BASE, "bias")] = bias
        return targets

    @staticmethod
    def _turned(rotation: _Rotation, key: str, value: Tensor) -> Tensor:
        """``R`` times a merge target: the rows of a weight (output features first), or a bias."""
        wide = value.to(rotation.first.dtype)
        turned = rotation.apply(wide.mT).mT if key.endswith("weight") else rotation.apply(wide)
        return turned.to(value.dtype)

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        rotation = self._rotation(adapter)
        return {
            key: self._turned(rotation, key, param) for key, param in self._merge_targets().items()
        }

    @override
    def _fold(self, adapter: str) -> None:
        values = self._merged_values(adapter)
        for key, param in self._merge_targets().items():
            param.copy_(values[key])

    @override
    def _unfold(self) -> None:
        targets = self._merge_targets()
        for adapter in reversed(self.merged_adapters):
            inverse = self._rotation(adapter).inverse()
            values = {key: self._turned(inverse, key, param) for key, param in targets.items()}
            for key, param in targets.items():
                param.copy_(values[key])


class Linear(RoadLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` whose output a RoAd rotation turns: :math:`y = R\,(W x + b)`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`RoadLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import RoadConfig
    >>> from lucid.peft.tuners.road import Linear
    >>> base = nn.Linear(4, 8)
    >>> layer = Linear(base, "default", RoadConfig(group_size=4, target_modules=["x"]))
    >>> tuple(layer.road_theta["default"].shape), tuple(layer.road_alpha["default"].shape)
    ((4,), (4,))
    >>> x = lucid.randn(3, 4)
    >>> bool((layer(x) == base(x)).all().item())   # angle 0 and scale 1 start at the identity
    True
    """

    base_layer: nn.Linear
