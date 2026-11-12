r"""The GLoRA layers: a frozen linear module whose weight and bias five support tensors correct.

GLoRA (Chavan et al., 2023, arXiv:2306.07967, section 2.2) writes the adapted layer as

.. math:: f(x) = (W_0 + W_0 A + B)\,x + C W_0 + D \odot b_0 + E + b_0 \qquad (\text{Eq. } 10),

and folds it into one weight and one bias (Eq. 13, 14):

.. math:: W = W_0 + W_0 \odot A + B, \qquad b = b_0 + b_0 \odot D + E + W_0 C .

Each of :math:`A, B, C, D, E` is a low-rank product, a vector, a scalar or absent (Eq. 11; the
``config_*`` fields choose).  The update reads :math:`W_0` and :math:`b_0`, so merging stashes them
(``_PristineMerge``): the merged weight and bias are the sums above, taken over every merged
adapter, and an unmerge gives :math:`W_0` and :math:`b_0` back bit for bit.  A module without a bias has
no :math:`C`, :math:`D` or :math:`E`; their slots hold absent tensors.

``GloraLayer`` is the family root.  An adapter holds the five support tensors, each a small module
(``glora_A`` ... ``glora_E``, see ``_paths``) whose parameters are the reference's ``Xd``, ``Xu``
and ``X``, and the buffer ``zeros`` of an absent tensor.
"""

from collections.abc import Mapping
from typing import ClassVar, NamedTuple, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._stash import _PristineMerge
from lucid.peft.tuners._support import _LinearSized, _SpecField
from lucid.peft.tuners.glora._config import GloraConfig
from lucid.peft.tuners.glora._paths import _Path

_BASE_WEIGHT = "base_layer.weight"
_BASE_BIAS = "base_layer.bias"


class _GloraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int


class GloraLayer(_DeltaLayer[GloraConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the corrections shared by the GLoRA layers.

    Per adapter: the five support tensors, each in its own table (``glora_A`` ... ``glora_E``), and
    the read-only mapping ``r``.  ``get_delta_weight`` is :math:`W_0 \odot A + B` for the weight
    as it was before any merge; the bias correction is the same computation on the bias.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight and bias are corrected; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : GloraConfig
        ``r``, ``config_A_B``, ``config_C``, ``config_D_E`` and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    ValueError
        ``base_layer`` creates its parameters on its first input, which has not come.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "glora_A",
        "glora_B",
        "glora_C",
        "glora_D",
        "glora_E",
    )
    _merged_names: ClassVar[tuple[str, ...]] = ("weight", "bias")
    _repr_prefix = "glora."
    glora_A: nn.ModuleDict[nn.Module]
    glora_B: nn.ModuleDict[nn.Module]
    glora_C: nn.ModuleDict[nn.Module]
    glora_D: nn.ModuleDict[nn.Module]
    glora_E: nn.ModuleDict[nn.Module]

    r = _SpecField[int]("Rank of the low-rank supports of each adapter, by name (read-only).")

    @override
    def _new_table(self) -> nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _enter(self, adapter_name: str, config: GloraConfig) -> None:
        home = self._home()
        out, inn, has_bias = self.out_features, self.in_features, self.bias is not None
        start = config.init_weights

        def support(kind: str, shape: tuple[int, ...], column: bool) -> _Path:
            return _Path.build(
                kind, shape, column=column, r=config.r, home=home, init_weights=start
            )

        # C and the bias terms are absent from a layer with no bias, whatever the config asks.
        supports = {
            "glora_A": support(config.config_A_B, (out, inn), True),
            "glora_B": support(config.config_A_B, (out, inn), True),
            "glora_C": support(config.config_C if has_bias else "none", (inn, 1), True),
            "glora_D": support(config.config_D_E if has_bias else "none", (out,), False),
            "glora_E": support(config.config_D_E if has_bias else "none", (out,), False),
        }
        self._specs[adapter_name] = _GloraSpec(config.r)
        for name, path in supports.items():
            getattr(self, name)[adapter_name] = path

    def _support(self, table: str, adapter: str) -> Tensor:
        path = getattr(self, table)[adapter]
        assert isinstance(path, _Path)
        value: Tensor = path()
        return value

    def _corrections(
        self, adapter: str, weight: Tensor, bias: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        r"""The change of the weight, :math:`W A + B`, and of the bias, :math:`b D + E + W C`, from ``weight`` and ``bias``."""
        a, b = self._support("glora_A", adapter), self._support("glora_B", adapter)
        weight = weight.to(a.dtype)
        correction = weight * a + b
        if bias is None:
            return correction, None
        c = self._support("glora_C", adapter)
        d, e = self._support("glora_D", adapter), self._support("glora_E", adapter)
        return correction, bias.to(d.dtype) * d + e + (weight @ c).squeeze(-1)

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        weight, bias = self._corrections(adapter, base[_BASE_WEIGHT], base.get(_BASE_BIAS))
        changes = {_BASE_WEIGHT: weight}
        if bias is not None:
            changes[_BASE_BIAS] = bias
        return changes

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        base = self._pristine()  # the base's own values, which a merge has changed in place
        weight, bias = self._corrections(adapter, base[_BASE_WEIGHT], base.get(_BASE_BIAS))
        out: Tensor = x.to(weight.dtype) @ weight.mT
        return out if bias is None else out + bias


class Linear(GloraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a GLoRA update: :math:`y = (W + W A + B)\,x + b + b D + E + W C`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`GloraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import GloraConfig
    >>> from lucid.peft.tuners.glora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", GloraConfig(r=2, target_modules=["x"]))
    >>> sorted(layer.glora_A["default"].state_dict()), tuple(layer.glora_C["default"].Xu.shape)
    (['Xd', 'Xu'], (2, 1))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the supports start at zero
    True
    """

    base_layer: nn.Linear
