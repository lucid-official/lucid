"""The field checks of ``AdaLoraConfig`` that are AdaLoRA's own: step counts, moving-average weights, rank patterns."""

import re
from collections.abc import Mapping
from typing import Final

from lucid.peft._config import _check_number

#: ``<module path>.lora_E`` with or without the adapter's name after it: how a rank pattern names a layer.
_TRIPLET_KEY: Final = re.compile(r"(?P<layer>.*\.lora_E|lora_E)(?:\.[^.]+)?")


def _check_steps(name: str, value: object) -> int:
    """A count of training steps: a non-negative ``int`` (``bool`` excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer, got {value}")
    return value


def _check_total_step(name: str, value: object) -> int | None:
    """``None`` (refused, with the reference's message, once every field is read) or a step count."""
    return None if value is None else _check_steps(name, value)


def _check_ema(name: str, value: object) -> float:
    beta = _check_number(name, value)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"`{name}` must lie strictly between 0 and 1, got {beta}")
    return beta


def _check_non_negative(name: str, value: object) -> float:
    number = _check_number(name, value)
    if number < 0:
        raise ValueError(
            f"`{name}` should be greater than or equal to 0, but the value passed is {number}"
        )
    return number


def _check_rank_pattern(name: str, value: object) -> dict[str, list[bool]] | None:
    """``None``, or ``{"<module>.lora_E": [kept?, ...]}`` — a key may carry the adapter's name, which is dropped."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a dict of str to a list of bool, or None, got {value!r}")
    pattern: dict[str, list[bool]] = {}
    for key, kept in value.items():
        found = _TRIPLET_KEY.fullmatch(key) if isinstance(key, str) else None
        if found is None:
            raise ValueError(f"`{name}` key {key!r} is not of the form '<module>.lora_E'")
        if not isinstance(kept, (list, tuple)) or not all(isinstance(k, bool) for k in kept):
            raise TypeError(f"`{name}[{key!r}]` must be a list of bool, got {kept!r}")
        pattern[found["layer"]] = list(kept)
    return pattern
