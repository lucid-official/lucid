"""Hyper-parameter reading for the compiled optimizers.

The compiled optimizers read every hyper-parameter from the live
``param_groups`` on each step (see :mod:`lucid.compile._optim.compiler`);
this module holds the one helper they share for it.
"""

from typing import Mapping


def _hp(g: Mapping[str, object], key: str, default: float) -> float:
    """Extract a float hyperparameter with the named default."""
    v = g.get(key, default)
    if v is None:
        return default
    if not isinstance(v, (int, float, bool)):
        raise TypeError(f"hyperparameter {key!r} must be numeric, got {type(v).__name__}")
    return float(v)
