"""
Where a warm-restarts schedule is in its cycle.
"""

import math

#: Keys a checkpoint written before the cycle was public carries.
_LEGACY_KEYS = {"_T_cur": "T_cur", "_T_i": "T_i"}


def _check_cycle_lengths(T_0: object, T_mult: object, eta_min: object) -> None:
    """Refuse a cycle length the schedule cannot be placed in, before any state is set.

    Checked on what was passed, not on the annotation: a ``T_0`` of ``0`` or
    ``3.5`` made every step a division by zero or a cycle that never ends.
    """
    if not (isinstance(T_0, int) and T_0 > 0):
        raise ValueError(f"Expected positive integer T_0, but got {T_0}")
    if not (isinstance(T_mult, int) and T_mult >= 1):
        raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
    if not isinstance(eta_min, float | int):
        raise TypeError(f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}")


def _current_cycle_keys(state_dict: dict[str, object]) -> dict[str, object]:
    """``state_dict`` with an older Lucid's private cycle keys under their public names."""
    current = dict(state_dict)
    for old, new in _LEGACY_KEYS.items():
        if old in current:
            current.setdefault(new, current.pop(old))
    return current


class _WarmRestartCycle:
    """The position in the cycle, ``T_cur`` epochs into one ``T_i`` long."""

    T_0: int
    T_mult: int
    T_cur: float
    T_i: int
    last_epoch: int

    def _advance_cycle(self) -> None:
        """Move a step through the cycle — to its start, before step 0 — as the reference."""
        if self.last_epoch < 0:
            self.T_cur, self.T_i = 0, self.T_0
            return
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur %= self.T_i
            self.T_i *= self.T_mult

    def _place_in_cycle(self, epoch: float) -> None:
        """Put the cycle at ``epoch``, which may be fractional, from the start."""
        if epoch < self.T_0:
            self.T_i, self.T_cur = self.T_0, epoch
        elif self.T_mult == 1:
            self.T_cur = epoch % self.T_0
        else:
            n = int(math.log(epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult))
            self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
            self.T_i = self.T_0 * self.T_mult**n
