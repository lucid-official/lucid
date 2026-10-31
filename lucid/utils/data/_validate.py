"""
Argument checks shared by the samplers.

Every sampler takes a handful of plain-Python scalars (a count, a flag)
that, left unchecked, fail far from the call that passed them — a string
``replacement`` is truthy, ``batch_size=True`` is the integer 1.  The
checks live here so each constructor refuses the value at its own
boundary with the same exception class and wording.
"""


def _check_bool(name: str, value: object) -> None:
    """Refuse a non-``bool`` flag; truthy stand-ins (``"yes"``, ``1``) are errors."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}: {value!r}.")


def _check_positive_int(name: str, value: object) -> None:
    """Refuse anything but a positive ``int`` — ``bool`` is not a count."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
