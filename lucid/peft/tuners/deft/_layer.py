r"""The DEFT layers: a frozen linear module whose weight loses a learned subspace and gains an injection into it.

DEFT (Kumar et al., 2025, arXiv:2509.22793, section 3.1) writes the adapted weight as

.. math:: W' = (I - P_{proj})\,W + Q_P R \qquad (\text{Eq. 1, 2}),

removing from the columns of :math:`W` the subspace :math:`P_{proj}` projects onto and writing the
rank-``r`` injection :math:`R` into the span of :math:`Q_P`.  The direction matrix
:math:`P \in \mathbb{R}^{d_{out} \times r}` and :math:`R \in \mathbb{R}^{r \times d_{in}}` are the
parameters.  The projector comes from :math:`P` by one of the decompositions of section 3.1.1:
``"relu"`` takes :math:`Q_P = P` and :math:`P_{proj} = P\,\mathrm{relu}(P)^\top` (not orthogonal),
``"qr"`` takes :math:`Q_P` from the QR decomposition of :math:`P` and :math:`P_{proj} = Q_P Q_P^\top`.
With ``para=True`` there is no :math:`R` and the update is pure removal (PaRa), which cannot start as
the identity.  The injection is scaled by ``alpha / r`` when ``alpha`` is given.

With :math:`M = \mathrm{right}^\top W` (``right`` is :math:`\mathrm{relu}(P)` or :math:`Q_P`) the
update is :math:`\Delta W = Q_P\,(s R - M)`, a rank-``r`` matrix of the weight, so the forward pass
never forms an ``out x out`` projector.  The update reads :math:`W`, so merging stashes it
(``_PristineMerge``) and unmerging gives it back bit for bit.  The default start chooses
:math:`R = M / s`, which makes :math:`\Delta W` zero.

``DeftLayer`` is the family root; an adapter holds ``deft_P`` (``(out, r)``) and, unless ``para``,
``deft_R`` (``(r, in)``), the reference's keys.  ``r`` is capped at the output size.
"""

from collections.abc import Mapping
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.linalg import qr
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._stash import _WEIGHT, _PristineMerge
from lucid.peft.tuners._support import _LinearSized, _SpecField
from lucid.peft.tuners.deft._config import DeftConfig


class _DeftSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the scale and
    the decomposition are baked into the graph.
    """

    deft_r: int
    deft_decomposition: str
    deft_init_scale: float
    deft_para: bool
    deft_scaling: float


class DeftLayer(_DeltaLayer[DeftConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the subspace edit shared by the DEFT layers.

    Per adapter: ``deft_P``, ``deft_R`` (absent with ``para``), a dropout of the injection's input
    and the read-only mappings ``deft_r``, ``deft_decomposition``, ``deft_init_scale``,
    ``deft_para`` and ``deft_scaling``.  ``get_delta_weight`` is :math:`\Delta W = Q_P\,(s R - M)` for
    the weight as it was before any merge.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight is edited; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : DeftConfig
        ``r``, ``decomposition_method``, ``init_scale``, ``alpha``, ``para``, ``deft_dropout`` and
        ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    ValueError
        ``base_layer`` creates its parameters on its first input, or ``config.fan_in_fan_out`` is set
        (the layer's weight is stored ``(out, in)``).
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("deft_P", "deft_R")
    _dropout_table = "deft_dropout"
    _repr_prefix = "deft."
    deft_P: nn.ParameterDict
    deft_R: nn.ParameterDict
    deft_dropout: nn.ModuleDict[nn.Module]

    deft_r = _SpecField[int](
        "Rank of each adapter, capped at the output size, by name (read-only)."
    )
    deft_decomposition = _SpecField[str]("Decomposition of each adapter, by name (read-only).")
    deft_init_scale = _SpecField[float]("Scale of each adapter's random start of R (read-only).")
    deft_para = _SpecField[bool]("Whether each adapter is pure removal, by name (read-only).")
    deft_scaling = _SpecField[float]("Scale of each adapter's injection, by name (read-only).")

    @override
    def _prepare(self, config: DeftConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)

    def _spec(self, adapter: str) -> _DeftSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _DeftSpec)
        return spec

    @staticmethod
    def _decomposed(direction: Tensor, method: str) -> tuple[Tensor, Tensor]:
        r"""``(Q_P, right)`` of the direction matrix: the basis the injection goes into and the factor of the projector :math:`Q_P\,\mathrm{right}^\top`."""
        if method == "relu":
            return direction, F.relu(direction)
        basis, _ = qr(direction)
        return basis, basis

    @override
    def _enter(self, adapter_name: str, config: DeftConfig) -> None:
        home = self._home()
        r = min(config.r, self.out_features)
        scaling = config.alpha / r if config.alpha else 1.0
        direction = home.zeros(self.out_features, r)
        init.normal_(direction, mean=0.0, std=0.02)
        injection: Tensor | None = None
        if not config.para and config.init_weights:
            # The identity start: R = right^T W / s makes the update (s R - right^T W) zero.
            with lucid.no_grad():
                _, right = self._decomposed(direction, config.decomposition_method)
                injection = right.mT @ self.weight.detach().to(home.dtype) / scaling
        elif not config.para:
            injection = home.zeros(r, self.in_features)
            init.normal_(injection, mean=0.0, std=0.02 * config.init_scale)
        self._specs[adapter_name] = _DeftSpec(
            r, config.decomposition_method, config.init_scale, config.para, scaling
        )
        self.deft_P[adapter_name] = Parameter(direction)
        if injection is not None:
            self.deft_R[adapter_name] = Parameter(injection)
        self._enter_dropout(adapter_name, config.deft_dropout)

    def _basis(self, adapter: str) -> tuple[Tensor, Tensor]:
        """``(Q_P, right)`` of the adapter's current ``P``."""
        direction = _parameter_at(self.deft_P, adapter)
        return self._decomposed(direction, self._spec(adapter).deft_decomposition)

    def _injection(self, adapter: str) -> Tensor | None:
        """:math:`s R`, or ``None`` for pure removal."""
        if adapter not in self.deft_R:
            return None
        return _parameter_at(self.deft_R, adapter) * self._spec(adapter).deft_scaling

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        basis, right = self._basis(adapter)
        removed = right.mT @ base[_WEIGHT].to(basis.dtype)  # M = right^T W
        injection = self._injection(adapter)
        return {_WEIGHT: basis @ (-removed if injection is None else injection - removed)}

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        # x (Q_P (s R - M))^T, with the removal read from the input before the injection's dropout.
        basis, right = self._basis(adapter)
        x = x.to(basis.dtype)
        removal = (x @ (self.weight.to(basis.dtype).mT @ right)) @ basis.mT
        injection = self._injection(adapter)
        if injection is None:
            return -removal
        injected: Tensor = (self._dropped(adapter, x) @ injection.mT) @ basis.mT
        return injected - removal


class Linear(DeftLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a DEFT update: :math:`y = ((I - P_{proj})\,W + Q_P R)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`DeftLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import DeftConfig
    >>> from lucid.peft.tuners.deft import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", DeftConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.deft_P["default"].shape), tuple(layer.deft_R["default"].shape)
    ((4, 2), (2, 8))
    >>> x = lucid.randn(3, 8)
    >>> bool(lucid.allclose(layer(x), base(x), atol=1e-5))   # R starts where the update is zero
    True
    """

    base_layer: nn.Linear
