r"""Reading a call that has more than one valid spelling.

The reference's random factories are overloaded: ``randint(high, size)`` and
``randint(low, high, size)``, ``rand(2, 3)`` and ``rand(size=(2, 3))``,
``normal`` in four shapes by what ``mean`` and ``std`` are.  No one Python
signature says all of them, and ``def f(low=0, high=None)`` binds a call the
reference refuses.  So a factory declares each spelling it accepts, as a
:class:`Spelling` of typed :class:`Slot`\ s, and :class:`Overloads` reads a
call against them in order, lowers the first one that fits onto the factory's
one canonical function, and refuses a call that fits none the way the
reference does.

The canonical function keeps the signature the factory shipped with; only
the reading of a call moved here, so every spelling of a draw reaches the
same code and a seed fixes the same values whichever way it was written.
"""

import functools
import inspect
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import SupportsIndex

from lucid import _promotion


class SlotKind(Enum):
    """How a :class:`Slot` is passed in the call: by position or name, as the rest of
    the positional arguments, or by name only."""

    #: By position or by its name.
    POSITIONAL = auto()
    #: The rest of the positional arguments, as ``*size`` collects them.
    VARIADIC = auto()
    #: By its name only.
    KEYWORD = auto()


@dataclass(frozen=True, eq=False)
class Slot:
    """One argument of a spelling: its name, what it accepts, how it is passed.

    ``splat`` hands a sequence value to the factory as positional arguments
    (``rand(size=(2, 3))`` reaches ``rand(*size)`` as ``rand(2, 3)``); a
    :attr:`SlotKind.VARIADIC` slot always does.  ``optional`` lets the call
    leave the slot out, in which case the factory's own default stands.
    """

    name: str
    accepts: Callable[[object], bool]
    label: str
    kind: SlotKind = SlotKind.POSITIONAL
    optional: bool = False
    splat: bool = False


def is_int(value: object) -> bool:
    """An integer — anything that indexes, but a bool is a flag, not a count.

    The check behind an integer :class:`Slot` such as ``randint``'s ``high``.
    """
    return isinstance(value, SupportsIndex) and not isinstance(value, bool)


def is_extent(value: object) -> bool:
    """A list or tuple of integers: the shape a factory draws.

    The check behind a shape :class:`Slot` such as ``randint``'s ``size``.
    """
    return isinstance(value, (list, tuple)) and all(is_int(v) for v in value)


def is_number(value: object) -> bool:
    """A real Python or NumPy scalar, but not a tensor.

    The check behind the float :class:`Slot` of ``normal``, which tells its float
    overload from its tensor overloads.
    """
    return _promotion.try_unwrap(value) is None and isinstance(
        _promotion.as_number(value), (int, float)
    )


def is_tensor(value: object) -> bool:
    """A tensor (or an engine tensor) — the ``Tensor`` overloads' operand.

    The check behind the tensor :class:`Slot` of ``normal``'s tensor overloads.
    """
    return _promotion.try_unwrap(value) is not None


def anything(value: object) -> bool:
    """Accepts every value; the factory itself checks it and says why.

    For a :class:`Slot` the spelling does not choose by, such as the ``input`` of ``randint_like``.
    """
    del value
    return True


@dataclass(frozen=True, eq=False)
class Spelling[R]:
    """One way the reference lets a call be written.

    ``fixed`` are the keyword arguments the spelling implies without being
    passed (``randint(high, size)`` is ``low=0``).  ``route`` replaces the
    factory when this spelling is a different overload, not another way to
    write the same one (``normal`` of tensors).
    """

    slots: tuple[Slot, ...]
    fixed: Mapping[str, object] = field(default_factory=dict)
    route: Callable[..., R] | None = None

    @property
    def names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.slots)

    def match(
        self, args: tuple[object, ...], kwargs: Mapping[str, object], reserved: frozenset[str]
    ) -> tuple[tuple[object, ...], dict[str, object]] | None:
        """Read a call as this spelling, or say it is not one.

        Parameters
        ----------
        args : tuple
            The call's positional arguments.
        kwargs : Mapping
            The call's keyword arguments.
        reserved : frozenset
            Every name some spelling takes: a keyword in it that this
            spelling does not consume is another spelling's argument, so the
            call is not this one.  Any other keyword is an option (``dtype``,
            ``device``) and goes through to the target.

        Returns
        -------
        tuple or None
            ``(positional, keyword)`` for the target, or ``None`` if the call
            is not this spelling.
        """
        taken = self._bind(args, kwargs)
        if taken is None:
            return None
        bound, consumed = taken
        if not all(slot.accepts(bound[slot.name]) for slot in self.slots if slot.name in bound):
            return None
        options = {k: v for k, v in kwargs.items() if k not in consumed}
        if any(k in reserved for k in options):
            return None
        return self._lower(bound, options)

    def _of(self, kind: SlotKind) -> list[Slot]:
        return [s for s in self.slots if s.kind is kind]

    def _bind(
        self, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> tuple[dict[str, object], set[str]] | None:
        """Each slot's value and the keywords that supplied them, or ``None`` on a misfit."""
        positional = self._of(SlotKind.POSITIONAL)
        variadic = self._of(SlotKind.VARIADIC)
        if len(args) > len(positional) and not variadic:
            return None
        bound: dict[str, object] = {
            slot.name: value for slot, value in zip(positional, args, strict=False)
        }
        for slot in variadic:
            bound[slot.name] = _collect(args[len(positional) :])
        consumed: set[str] = set()
        for slot in positional + self._of(SlotKind.KEYWORD):
            if slot.name in kwargs:
                if slot.name in bound:
                    return None
                bound[slot.name] = kwargs[slot.name]
                consumed.add(slot.name)
            elif slot.name not in bound and not slot.optional:
                return None
        return bound, consumed

    def _lower(
        self, bound: Mapping[str, object], options: Mapping[str, object]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        splat: tuple[object, ...] = ()
        named: dict[str, object] = dict(self.fixed)
        for slot in self.slots:
            if slot.name not in bound:
                continue
            value = bound[slot.name]
            if (slot.splat or slot.kind is SlotKind.VARIADIC) and isinstance(value, (list, tuple)):
                splat = tuple(value)
            else:
                named[slot.name] = value
        return splat, {**named, **options}

    def signature(self, target: Callable[..., object]) -> inspect.Signature:
        """This spelling as a Python signature, with ``target``'s keyword options."""
        kinds = inspect.Parameter
        params = [
            *(
                kinds(s.name, kinds.POSITIONAL_OR_KEYWORD, default=_default(s))
                for s in self._of(SlotKind.POSITIONAL)
            ),
            *(kinds(s.name, kinds.VAR_POSITIONAL) for s in self._of(SlotKind.VARIADIC)),
            *(
                kinds(s.name, kinds.KEYWORD_ONLY, default=_default(s))
                for s in self._of(SlotKind.KEYWORD)
            ),
        ]
        taken = {p.name for p in params}
        params.extend(
            p
            for p in inspect.signature(target).parameters.values()
            if p.kind is kinds.KEYWORD_ONLY and p.name not in taken
        )
        return inspect.Signature(params)

    def describe(self) -> str:
        """The spelling as a call reads, for the refusal message."""
        parts: list[str] = []
        keyword: list[str] = []
        for slot in self.slots:
            if slot.kind is SlotKind.VARIADIC:
                parts.append(f"*{slot.name}: {slot.label}")
            elif slot.kind is SlotKind.KEYWORD:
                keyword.append(f"{slot.name}: {slot.label}")
            else:
                parts.append(f"{slot.name}: {slot.label}")
        if keyword:
            parts.append("*")
            parts.extend(keyword)
        return f"({', '.join(parts)})"


def _default(slot: Slot) -> object:
    """The default a signature shows for ``slot``: none if the call must give it."""
    return ... if slot.optional else inspect.Parameter.empty


def _collect(rest: tuple[object, ...]) -> tuple[object, ...]:
    """``rand(2, 3)`` and ``rand((2, 3))`` both collect ``(2, 3)``."""
    if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
        return tuple(rest[0])
    return rest


#: Public factory -> the spellings it was declared with.
_DECLARED: weakref.WeakKeyDictionary[Callable[..., object], tuple[Spelling[object], ...]] = (
    weakref.WeakKeyDictionary()
)


def spellings_of(factory: Callable[..., object]) -> tuple[Spelling[object], ...]:
    """The spellings ``factory`` was declared with, or ``()`` if it was not.

    What the stub-versus-runtime test reads to know the placements a factory's overloads may use.
    """
    return _DECLARED.get(factory, ())


class Overloads[R]:
    """Declares the spellings a factory accepts; used as the factory's decorator.

    The decorated function is the canonical one every spelling lowers onto,
    and it keeps its signature, so ``inspect.signature`` and the released
    surface still see the one callers have always had.  Spellings are tried
    in order and the first that fits wins.
    """

    def __init__(self, *spellings: Spelling[R]) -> None:
        self._spellings = spellings
        self._reserved = frozenset(n for s in spellings for n in s.names)

    def __call__[**P](self, factory: Callable[P, R]) -> Callable[P, R]:
        name = factory.__name__

        @functools.wraps(factory)
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            for spelling in self._spellings:
                lowered = spelling.match(args, kwargs, self._reserved)
                if lowered is not None:
                    positional, keyword = lowered
                    target: Callable[..., R] = spelling.route or factory
                    return target(*positional, **keyword)
            raise TypeError(self._refusal(name, args, kwargs))

        _DECLARED[call] = self._spellings
        return call

    def _refusal(self, name: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> str:
        got = [type(a).__name__ for a in args]
        got.extend(f"{k}={type(v).__name__}" for k, v in kwargs.items())
        expected = "\n".join(f" * {name}{s.describe()}" for s in self._spellings)
        return (
            f"{name}() received an invalid combination of arguments - got "
            f"({', '.join(got)}), but expected one of:\n{expected}"
        )
