r"""The BEFT layers: a frozen linear module and a trained bias added to its output.

BEFT (Huang et al., 2025, arXiv:2509.15974) fine-tunes nothing but a bias :math:`\beta`, a vector of
the layer's output width that starts at zero; the adapted layer is :math:`y = W x + b + \beta`.  The
paper puts it on the value projections of attention.  Where the base layer has a bias already, the
adapter folds into it, :math:`b' = b + \beta`; where it has none the adapter still runs in the
forward pass but cannot be merged (the reference warns of this when it is built and refuses the
merge).  A merge keeps the bias it was folded into, so unmerging gives it back bit for bit.

``BeftLayer`` is the family root.  An adapter holds ``beft_bias``, a ``(1, out)`` parameter, the
reference's key and shape; with several active adapters the biases add.  Nothing in the layer
depends on the input, so ``get_delta_weight`` is zero and the change is ``get_delta_bias``.
"""

import warnings
from collections.abc import Mapping
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._stash import _PristineMerge
from lucid.peft.tuners._support import _LinearSized
from lucid.peft.tuners.beft._config import BeftConfig

#: The one parameter of the base BEFT changes, by its name under the module at the bottom.
_BIAS: Final = "bias"


class BeftLayer(_DeltaLayer[BeftConfig], _LinearSized, _PristineMerge, family=True):
    r"""Adapter bookkeeping and the added bias shared by the BEFT layers.

    Per adapter: ``beft_bias``.  The forward pass adds each active adapter's bias to the base
    output; merging adds it to the base bias.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose output the bias shifts; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : BeftConfig
        ``init_weights`` chooses a zero start (the default) or a standard normal draw.

    Raises
    ------
    TypeError
        The module is not an ``nn.Linear``, so there is no output width to give a bias.
    ValueError
        The module still has lazy parameters, whose output width is not known yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("beft_bias",)
    _merged_names: ClassVar[tuple[str, ...]] = (_BIAS,)
    beft_bias: nn.ParameterDict

    @override
    def _enter(self, adapter_name: str, config: BeftConfig) -> None:
        if self.bias is None:
            warnings.warn(
                "Detected that the base layer has no bias term. "
                "Note you cannot merge the BEFT adapter into the base layer.",
                stacklevel=3,
            )
        home = self._home()
        shape = (1, self.out_features)
        start = (
            home.zeros(*shape)
            if config.init_weights
            else lucid.randn(*shape, device=home.device, dtype=home.dtype)
        )
        self.beft_bias[adapter_name] = nn.Parameter(start)

    def get_delta_bias(self, adapter: str) -> Tensor:
        """The change ``adapter`` makes to the base bias: its ``out`` numbers, in the adapter's dtype.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out,)``.
        """
        return _parameter_at(self.beft_bias, adapter).reshape(-1)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        """Zero: BEFT changes the bias alone, which ``get_delta_bias`` returns.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Zeros shaped like the base weight, in the adapter's dtype.
        """
        dtype = self.get_delta_bias(adapter).dtype
        return lucid.zeros(self.weight.shape, device=self.weight.device, dtype=dtype)

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        del base  # the added bias does not depend on the bias it is added to
        return {self._base_key(_BIAS): self.get_delta_bias(adapter)}

    @override
    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        pending = super()._pending_merges(adapter_names)
        if pending and self.bias is None:
            raise RuntimeError(
                "Base layer has no bias, cannot merge a BEFT adapter into it: its bias has "
                "nothing to be added to"
            )
        return pending

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        del x  # the bias does not read the input
        return self.get_delta_bias(adapter)


class Linear(BeftLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a BEFT bias: :math:`y = W x + b + \beta`.

    Stackable: the bias reads nothing of the weight, so the layer may stand on another method's
    layer (and one may stand on it).

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`BeftLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import BeftConfig
    >>> from lucid.peft.tuners.beft import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", BeftConfig(target_modules=["x"]))
    >>> tuple(layer.beft_bias["default"].shape)
    (1, 4)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the bias starts at zero
    True
    """

    _stackable = True
