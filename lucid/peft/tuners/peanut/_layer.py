r"""The PEANuT layers: a frozen linear module whose weight update is a small network applied to the weight.

PEANuT (Zhong et al., 2024, arXiv:2410.01870, section 3.3) models the update as a function of the frozen
weight, :math:`y = (W + f(W;\theta))\,x`, with :math:`f` a bottleneck network (a LoRA's product
:math:`AB` is the case with no non-linearity and no dependence on :math:`W`).  Here, for
:math:`W \in \mathbb{R}^{d_{out} \times d_{in}}`,

.. math:: h_0 = \sigma(W^\top A^\top), \qquad
          h \leftarrow \sigma(E_i h)\ (i = 1..L), \qquad
          h \leftarrow \sigma(D_i h) + h_{L-i}\ (i = 1..L), \qquad
          f(W) = s\,(h\,B^\top)^\top ,

with the encoders :math:`E_i`, the decoders :math:`D_i` and :math:`L` the ``depth`` (appendix F, where
the residual connections are introduced).  :math:`B` starts at zero, and so does the update.  The
update is a function of :math:`W`, so merging stashes it (``_PristineMerge``) and unmerging gives it
back bit for bit.

``PeanutLayer`` is the family root.  An adapter holds four tables of small linear modules,
``peanut_A``, ``peanut_B`` and the lists ``peanut_encoders`` / ``peanut_decoders``, the reference's keys.
"""

from collections.abc import Mapping
from typing import ClassVar, NamedTuple, override

import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._stash import _WEIGHT, _PristineMerge
from lucid.peft.tuners._support import _LinearSized, _SpecField
from lucid.peft.tuners.peanut._activations import _lookup
from lucid.peft.tuners.peanut._config import PeanutConfig


class _PeanutSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the depth, the
    scale and the activation are baked into the graph.
    """

    r: int
    depth: int
    scaling: float
    act_fn: str


def _linear_at(table: nn.ModuleDict[nn.Module], adapter: str) -> nn.Linear:
    """``table[adapter]``, which a ``ModuleDict`` types as any module: it is one of the network's linear layers."""
    layer = table[adapter]
    assert isinstance(layer, nn.Linear)
    return layer


class PeanutLayer(_DeltaLayer[PeanutConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the weight-reading network shared by the PEANuT layers.

    Per adapter: the modules of the network in four tables and the read-only mappings ``r``,
    ``depth``, ``scaling`` and ``act_fn``.  ``get_delta_weight`` is :math:`f(W)` for the weight as
    it was before any merge; the forward pass applies :math:`x\,f(W)^\top` to the input.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight the network reads; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : PeanutConfig
        ``r``, ``depth``, ``act_fn``, ``scaling`` and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is some other module type.
    ValueError
        ``base_layer`` has parameters that only its first input will create.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "peanut_A",
        "peanut_B",
        "peanut_encoders",
        "peanut_decoders",
    )
    _repr_prefix = "peanut."
    peanut_A: nn.ModuleDict[nn.Module]
    peanut_B: nn.ModuleDict[nn.Module]
    peanut_encoders: nn.ModuleDict[nn.Module]
    peanut_decoders: nn.ModuleDict[nn.Module]

    r = _SpecField[int]("Hidden width of each adapter's network, by name (read-only).")
    depth = _SpecField[int]("Encoder and decoder layers of each adapter, by name (read-only).")
    scaling = _SpecField[float]("Scale of each adapter's update, by name (read-only).")
    act_fn = _SpecField[str]("Name of each adapter's non-linearity, by name (read-only).")

    @override
    def _new_table(self) -> nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _enter(self, adapter_name: str, config: PeanutConfig) -> None:
        home, r = self._home(), config.r
        down = home.linear(self.out_features, r)
        up = home.linear(r, self.out_features)
        encoders = nn.ModuleList([home.linear(r, r) for _ in range(config.depth)])
        decoders = nn.ModuleList([home.linear(r, r) for _ in range(config.depth)])
        for module in (down, *encoders, *decoders):
            init.kaiming_uniform_(module.weight, a=5**0.5)
        if config.init_weights:
            init.zeros_(up.weight)
        else:
            init.kaiming_uniform_(up.weight, a=5**0.5)
        self._specs[adapter_name] = _PeanutSpec(r, config.depth, config.scaling, config.act_fn)
        self.peanut_A[adapter_name] = down
        self.peanut_B[adapter_name] = up
        self.peanut_encoders[adapter_name] = encoders
        self.peanut_decoders[adapter_name] = decoders

    def _spec(self, adapter: str) -> _PeanutSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _PeanutSpec)
        return spec

    def _network(self, adapter: str, weight: Tensor) -> Tensor:
        r""":math:`s\,h\,B^\top`, shaped ``(in, out)``: the update the network makes of ``weight``, transposed."""
        spec = self._spec(adapter)
        act = _lookup(spec.act_fn)
        down, up = _linear_at(self.peanut_A, adapter), _linear_at(self.peanut_B, adapter)
        encoders, decoders = self.peanut_encoders[adapter], self.peanut_decoders[adapter]
        assert isinstance(encoders, nn.ModuleList) and isinstance(decoders, nn.ModuleList)
        hidden: Tensor = act(weight.to(down.weight.dtype).mT @ down.weight.mT)
        shortcuts = []
        for encoder in encoders:
            shortcuts.append(hidden)
            hidden = act(encoder(hidden))
        for decoder, shortcut in zip(decoders, reversed(shortcuts), strict=True):
            hidden = act(decoder(hidden)) + shortcut
        update: Tensor = up(hidden) * spec.scaling
        return update

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {_WEIGHT: self._network(adapter, base[_WEIGHT]).mT}

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        update = self._network(adapter, self.weight)
        out: Tensor = x.to(update.dtype) @ update
        return out


class Linear(PeanutLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a PEANuT update: :math:`y = (W + f(W;\theta))\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`PeanutLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PeanutConfig
    >>> from lucid.peft.tuners.peanut import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", PeanutConfig(r=2, depth=1, target_modules=["x"]))
    >>> sorted(layer.peanut_A["default"].state_dict()), tuple(layer.peanut_B["default"].weight.shape)
    (['weight'], (4, 2))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear
