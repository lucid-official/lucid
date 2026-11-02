"""How a param group's hyper-parameters reach its engine optimizer.

A group key is not always an engine argument.  ``decoupled_weight_decay``
is never one: it picks which engine rule runs (Adam against AdamW) or, for
an optimizer that decays in Python, whether the engine's own ``weight_decay``
is switched off.  A few more keys (NAdam's ``momentum_decay``) the engine
takes only at construction, so changing one needs a new engine.  These
helpers hold that translation in one place, for the constructor, the
hand-over of an edit and the rules check alike.
"""

from typing import ClassVar, Protocol

DECOUPLED = "decoupled_weight_decay"


class _ArgRules(Protocol):
    """What an optimizer class says about its group keys (see ``Optimizer``)."""

    _REBUILD_KEYS: ClassVar[tuple[str, ...]]
    _PYTHON_DECOUPLED_WD: ClassVar[bool]


def engine_args(
    rules: _ArgRules, values: dict[str, object], *, constructing: bool
) -> dict[str, object]:
    """Return a group's hyper-parameters as the arguments the engine takes.

    ``decoupled_weight_decay`` is dropped, and ``weight_decay`` is zeroed when
    the optimizer decays in Python and the flag is on, so the engine does not
    decay a second time.

    Parameters
    ----------
    rules : _ArgRules
        The optimizer (or its class): ``_REBUILD_KEYS`` and
        ``_PYTHON_DECOUPLED_WD`` say which keys the engine does not take.
    values : dict
        The group's ``_HYPERPARAMS`` entries, by name.
    constructing : bool
        ``True`` for the engine's constructor, which takes the keys in
        ``_REBUILD_KEYS``; ``False`` for ``set_hyperparams``, which does not.

    Returns
    -------
    dict
        The entries to hand to the engine.
    """
    out = {k: v for k, v in values.items() if k != DECOUPLED}
    if rules._PYTHON_DECOUPLED_WD and values.get(DECOUPLED, False):
        out["weight_decay"] = 0.0
    if not constructing:
        for key in rules._REBUILD_KEYS:
            out.pop(key, None)
    return out


def rule_args[V](rebuild_keys: tuple[str, ...], values: dict[str, V]) -> dict[str, V]:
    """Return a group's hyper-parameters as the engine's rules check them.

    Parameters
    ----------
    rebuild_keys : tuple of str
        The keys the engine's ``check_hyperparams`` does not know.
    values : dict
        The group's ``_HYPERPARAMS`` entries, by name.

    Returns
    -------
    dict
        ``values`` without ``decoupled_weight_decay`` and ``rebuild_keys``.
    """
    return {k: v for k, v in values.items() if k != DECOUPLED and k not in rebuild_keys}


def check_decoupled_flag(owner: str, values: dict[str, object]) -> None:
    """Raise ``TypeError`` when ``decoupled_weight_decay`` is present and no bool.

    Parameters
    ----------
    owner : str
        The optimizer's name, for the message.
    values : dict
        The group's ``_HYPERPARAMS`` entries, by name.
    """
    flag = values.get(DECOUPLED, False)
    if not isinstance(flag, bool):
        raise TypeError(f"{owner}: {DECOUPLED} must be a bool, got {type(flag).__name__}")
