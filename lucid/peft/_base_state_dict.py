"""The base model's own tensors, under the names the model had before it was adapted.

An adapted model's state dict names a base weight ``q.base_layer.weight`` where
the model had ``q.weight``, a classification head ``head.original_module.weight``,
and carries the adapters' tensors beside them.  ``get_base_model_state_dict`` reads
the base weights out under their original names and ``set_base_model_state_dict``
loads a state dict of original names back in — for weights restored after a
distributed wrap, or a base checkpoint loaded into a model that is already adapted.
"""

from typing import Final

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn._incompatible_keys import _IncompatibleKeys
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._peft_model import PeftModel
from lucid.peft.tuners._layer import _BaseTunerLayer, _join

#: The class of module that stands in for a base module, and the attribute it keeps the base in.
_STAND_INS: Final[tuple[tuple[type[nn.Module], str], ...]] = (
    (_BaseTunerLayer, "base_layer"),
    (ModulesToSaveWrapper, "original_module"),
)


class _BaseWeights:
    """Which tensors of an adapted model are the base model's, and what each was called before.

    The map is read off the module tree, not worked out from the key strings: a module
    whose own name is ``base_layer`` is a module of the base model, not an infix to strip.
    Everything a stand-in module owns apart from the module it stands in for — the
    adapter's tables, a ``modules_to_save`` copy — belongs to the adapter.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        #: Modules that hold base tensors: adapted module path -> original module path.
        self._owners: dict[str, str] = {}
        self._collect(model, "", "")

    def _collect(self, module: nn.Module, adapted: str, original: str) -> None:
        for kind, attribute in _STAND_INS:
            if isinstance(module, kind):
                inner = getattr(module, attribute)
                self._collect(inner, _join(adapted, attribute), original)
                return
        self._owners[adapted] = original
        for name, child in module.named_children():
            self._collect(child, _join(adapted, name), _join(original, name))

    def original_keys(self) -> dict[str, str]:
        """Each base tensor's key in the adapted model's state dict -> its key before adapting."""
        keys: dict[str, str] = {}
        for key in self._model.state_dict():
            owner, _, leaf = key.rpartition(".")
            if owner in self._owners:
                keys[key] = _join(self._owners[owner], leaf)
        return keys


def _adapted_model(model: PeftModel, where: str) -> nn.Module:
    if not isinstance(model, PeftModel):
        raise TypeError(f"{where} takes a PeftModel, got {type(model).__name__}")
    return model.get_base_model()


def get_base_model_state_dict(model: PeftModel) -> dict[str, Tensor]:
    r"""Return the base model's state dict, under the key names it had before it was adapted.

    The adapters' tensors are left out, and the wrappers adapting put in the tree are
    undone in the names: ``q.base_layer.weight`` comes back as ``q.weight``, and
    ``head.original_module.weight`` (a module saved in full) as ``head.weight``.  The
    tensors are the model's own, not copies.

    Parameters
    ----------
    model : PeftModel
        An adapted model, as :func:`~lucid.peft.get_peft_model` returns it.

    Returns
    -------
    dict of str to Tensor
        The base model's state dict.

    Raises
    ------
    TypeError
        ``model`` is not a ``PeftModel``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_base_model_state_dict, get_peft_model
    >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    >>> peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
    >>> sorted(get_base_model_state_dict(peft))
    ['0.bias', '0.weight', '1.bias', '1.weight']
    """
    base = _adapted_model(model, "get_base_model_state_dict")
    state = base.state_dict()
    return {original: state[key] for key, original in _BaseWeights(base).original_keys().items()}


def set_base_model_state_dict(
    model: PeftModel, state_dict: dict[str, Tensor], strict: bool = True
) -> _IncompatibleKeys:
    r"""Load a state dict of the base model's original key names into an adapted model.

    The counterpart of :func:`~lucid.peft.get_base_model_state_dict`: the keys are the ones the
    model had before it was adapted (``q.weight``), and each is loaded into the place the
    adapted model keeps that tensor (``q.base_layer.weight``).  The adapters' tensors
    are not touched.

    Parameters
    ----------
    model : PeftModel
        An adapted model.
    state_dict : dict of str to Tensor
        The base model's tensors by their original names.
    strict : bool, default=True
        Raise when a base tensor is missing from ``state_dict`` or ``state_dict`` holds a
        key the base model does not.

    Returns
    -------
    _IncompatibleKeys
        The base keys ``state_dict`` lacks (``missing_keys``) and the keys of ``state_dict``
        that match no base tensor (``unexpected_keys``).

    Raises
    ------
    TypeError
        ``model`` is not a ``PeftModel``.
    RuntimeError
        ``strict`` is true and the keys do not match; nothing is loaded.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import (
    ...     LoraConfig, get_base_model_state_dict, get_peft_model, set_base_model_state_dict
    ... )
    >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
    >>> weights = {k: lucid.zeros(*v.shape) for k, v in get_base_model_state_dict(peft).items()}
    >>> set_base_model_state_dict(peft, weights)
    _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    >>> float(peft.get_base_model()[0].base_layer.weight.abs().sum())
    0.0
    """
    base = _adapted_model(model, "set_base_model_state_dict")
    adapted_of = {original: key for key, original in _BaseWeights(base).original_keys().items()}
    missing = [original for original in adapted_of if original not in state_dict]
    unexpected = [key for key in state_dict if key not in adapted_of]
    if strict and (missing or unexpected):
        problems = []
        if missing:
            problems.append(f"Missing key(s) in state_dict: {missing}")
        if unexpected:
            problems.append(f"Unexpected key(s) in state_dict: {unexpected}")
        raise RuntimeError("Error(s) in loading state_dict:\n\t" + "\n\t".join(problems))
    base.load_state_dict(
        {adapted_of[key]: value for key, value in state_dict.items() if key in adapted_of},
        strict=False,
    )
    return _IncompatibleKeys(missing_keys=missing, unexpected_keys=unexpected)
