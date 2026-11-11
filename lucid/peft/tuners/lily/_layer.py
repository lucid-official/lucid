r"""The Lily layers: a frozen linear layer beside a low-rank down-projection and a router that mixes shared up-projections.

Lily (Zhong et al., 2024, arXiv:2407.09946, Sec. 3.1-3.2) breaks LoRA's pairing of a down-projection
to one layer.  A layer projects its input down, :math:`h = x A` (Eq. 1); a router :math:`R` turns
:math:`h` into softmax weights :math:`S` over the ``num_B`` up-projections :math:`B_e` the model
shares (Eq. 2); and the layer adds :math:`s\,h \sum_e S_e B_e` (Eq. 3-4).  :math:`A` is shared by a
run of ``stride_A`` neighbouring layers, the :math:`B_e` by every layer of a kind, and each layer
has a router of its own.  In the paper's implementation (Fig. 7) the weights :math:`S` are the
router's softmax *averaged over every row of the batch*, so one combined :math:`B` serves a whole
call.

``LilyLayer`` is the family root.  A layer holds, per adapter name, three ``ModuleDict``\ s:
``lily_A`` (an ``nn.Linear(in, r)``), ``lily_B`` (an ``nn.Linear(out, num_B * r)``, whose weight
``(num_B * r, out)`` is the ``num_B`` up-projections end to end) and ``lily_router`` (an
``nn.Linear(r, num_B)``) — all bias-free, so a state dict reads ``....lily_A.default.weight`` and an
adapter file ``....lily_A.weight``.  The tuner makes the entries of ``lily_A`` and ``lily_B`` of
layers that share them the same module.  The update depends on the batch, so no weight holds it:
the layer cannot merge.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._support import _LinearSized, _SpecField, _Unmergeable
from lucid.peft.tuners.lily._config import LilyConfig


class _LilySpec(NamedTuple):
    """What a layer remembers of one adapter besides its modules.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the scale and the
    expert count are baked into the graph.
    """

    r: int
    scaling: float
    stride_A: int
    num_B: int


class LilyLayer(_DeltaLayer[LilyConfig], _LinearSized, _Unmergeable, family=True):
    r"""Adapter bookkeeping and the routed update shared by the Lily layers.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : LilyConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear`` as the layer class requires.
    ValueError
        ``base_layer`` is lazy and has not been run.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("lily_A", "lily_B", "lily_router")
    _repr_prefix = "lily."
    _unmergeable_because = (
        "a Lily layer, which mixes its up-projections by the batch's router average"
    )
    lily_A: nn.ModuleDict[nn.Module]
    lily_B: nn.ModuleDict[nn.Module]
    lily_router: nn.ModuleDict[nn.Module]

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Inner dimension of each adapter's projections, by name (read-only).")
    scaling = _SpecField[float]("Multiplier of each adapter's update, by name (read-only).")
    stride_A = _SpecField[int](
        "Layers that share each adapter's down-projection, by name (read-only)."
    )
    num_B = _SpecField[int]("Up-projections of each adapter, by name (read-only).")

    @override
    def _new_table(self) -> nn.ParameterDict | nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _enter(self, adapter_name: str, config: LilyConfig) -> None:
        home = self._home()
        lily_a = home.linear(self.in_features, config.r)
        lily_b = home.linear(self.out_features, config.num_B * config.r)
        router = home.linear(config.r, config.num_B)
        # Each starts as a linear layer's weight does; the up-projections at zero unless asked not to.
        init.kaiming_uniform_(lily_a.weight, a=math.sqrt(5))
        init.kaiming_uniform_(router.weight, a=math.sqrt(5))
        if config.init_weights:
            init.zeros_(lily_b.weight)
        else:
            init.kaiming_uniform_(lily_b.weight, a=math.sqrt(5))
        self._specs[adapter_name] = _LilySpec(
            config.r, config.scaling, config.stride_A, config.num_B
        )
        self.lily_A[adapter_name] = lily_a
        self.lily_B[adapter_name] = lily_b
        self.lily_router[adapter_name] = router

    def _spec(self, adapter: str) -> _LilySpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _LilySpec)  # ``_enter`` is the only writer
        return spec

    def _projections(self, adapter: str) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """``(down, up, router)`` of ``adapter``, which are ``nn.Linear`` by construction."""
        held = (self.lily_A[adapter], self.lily_B[adapter], self.lily_router[adapter])
        down, up, router = held
        assert isinstance(down, nn.Linear) and isinstance(up, nn.Linear)
        assert isinstance(router, nn.Linear)
        return down, up, router

    def _update(self, adapter: str, x: Tensor) -> Tensor:
        r"""The adapter's addition to the base output for ``x``: :math:`s\,h \sum_e S_e B_e` with :math:`S` the router's mean over all rows."""
        spec = self._spec(adapter)
        down, up, router = self._projections(adapter)
        hidden: Tensor = down(x.to(down.weight.dtype))
        weights = F.softmax(router(hidden), dim=-1).reshape(-1, spec.num_B).mean(dim=0)
        experts = up.weight.reshape(spec.num_B, -1, int(up.weight.shape[1]))  # (num_B, r, out)
        combined = (weights.reshape(spec.num_B, 1, 1) * experts).sum(dim=0)  # (r, out)
        return (hidden @ combined) * spec.scaling

    @override
    def forward(self, x: Tensor) -> Tensor:
        """The base output plus every active adapter's routed update; the base alone when adapters are disabled."""
        result: Tensor = self.base_layer(x)
        if self.disable_adapters:
            return result
        out = result
        for adapter in self.active_adapters:
            if adapter in self._specs:  # an adapter aimed at other layers: this one runs its base
                out = out + self._update(adapter, x).to(result.dtype)
        return out


class Linear(LilyLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a Lily update: :math:`y = W x + b + s\,(x A) \sum_e S_e B_e`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`LilyLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LilyConfig
    >>> from lucid.peft.tuners.lily import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", LilyConfig(r=3, num_B=2, target_modules=["x"]))
    >>> tuple(layer.lily_A["default"].weight.shape), tuple(layer.lily_B["default"].weight.shape)
    ((3, 8), (6, 4))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the up-projections start at zero
    True
    """

    base_layer: nn.Linear
