"""What ``vmap(randomness=)`` asks of the random draws of a batch.

``vmap`` runs ``func`` once per batch element, and each element's random ops
draw from the one default generator.  Whether those draws are the elements'
own or one draw they all share is the policy:

* ``"error"`` — a random op inside ``func`` is an error (the check lives in
  :func:`lucid._ctx.check_random_allowed`);
* ``"different"`` — every element draws on its own, one after the other;
* ``"same"`` — every element sees the draw the first one saw.

A policy is a :class:`Randomness` subclass, entered in the registry by being
defined; the one thing a run needs from it that differs is a :class:`Draws`,
the scope the elements are run in.
"""

from collections.abc import Callable
from types import TracebackType
from typing import ClassVar, Self, final, override

from lucid._ctx import check_randomness
from lucid._factories.random import get_rng_state, set_rng_state
from lucid._tensor.tensor import Tensor

__all__ = [
    "DifferentRandomness",
    "Draws",
    "ErrorRandomness",
    "Randomness",
    "SameRandomness",
    "SharedDraws",
]


class Draws:
    """The scope a batch's elements run in: each draws from where the
    one before it left the generator."""

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None

    def element[R](self, call: Callable[[], R]) -> R:
        """The result of ``call``, one batch element's (or one chunk's) run."""
        return call()


@final
class SharedDraws(Draws):
    """The scope in which every element starts from the generator state the
    first one started from, so all of them draw the same numbers.

    Left, the generator stands where one element's run leaves it — the
    batch as a whole used up what a single call would, which is what keeps
    the draws after a ``"same"`` ``vmap`` the ones after a plain call.
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: Tensor | None = None
        self._end: Tensor | None = None

    @override
    def __enter__(self) -> Self:
        self._start = get_rng_state()
        self._end = None
        return self

    @override
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._end is not None:
            set_rng_state(self._end)

    @override
    def element[R](self, call: Callable[[], R]) -> R:
        assert self._start is not None, "elements run inside the scope"
        set_rng_state(self._start)
        result = call()
        if self._end is None:
            self._end = get_rng_state()
        return result


class Randomness:
    """One value of ``randomness=``.

    A subclass names its mode — ``class Same(Randomness, mode="same")`` — and
    is entered in the registry when the class body ends.

    Attributes
    ----------
    mode : str
        The string the caller passes.
    draws : type of Draws
        The scope the elements run in.
    whole_batch : bool
        Whether one call of ``func`` on the whole batch honours the policy.
        It does not for ``"different"``: a random op in that call draws once
        and the draw is shared by the batch.
    """

    mode: ClassVar[str]
    draws: ClassVar[type[Draws]] = Draws
    whole_batch: ClassVar[bool] = True
    _registry: ClassVar[dict[str, Randomness]] = {}

    @override
    def __init_subclass__(cls, *, mode: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if mode is not None:
            cls.mode = mode
            Randomness._registry[mode] = cls()

    @staticmethod
    def named(mode: str) -> Randomness:
        """The policy called ``mode``; ``RuntimeError`` for any other value."""
        return Randomness._registry[check_randomness(mode)]


@final
class ErrorRandomness(Randomness, mode="error"):
    pass


@final
class DifferentRandomness(Randomness, mode="different"):
    whole_batch = False


@final
class SameRandomness(Randomness, mode="same"):
    draws = SharedDraws
