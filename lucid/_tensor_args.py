"""Tensor-argument validation at the public boundary.

A parameter that stands for a tensor must be refused with ``TypeError`` before
anything reads it.  Left to travel, ``None`` reached a pybind11 ``TensorImpl``
parameter — which accepts it as a null pointer, the engine then dereferenced —
or a composite's first ``x.ndim``, so the same mistake answered with a signal,
an engine ``LucidError`` or an ``AttributeError`` depending on which function
met it.  The reference raises ``TypeError`` for every one of them.

This module owns that rule so no op restates it:

* :func:`require_tensor` and :func:`require_tensors` are the one wording of the
  refusal; the registry's free functions and Tensor methods call them where a
  tensor slot holds something else.
* :func:`guard_functions` wraps a module's public Python functions.  Which
  parameters are tensors is read from the function's own annotations (once, on
  its first call): a bare ``Tensor`` must be one, ``Tensor | None`` may also be
  ``None``, a ``Sequence[Tensor]`` is a tensor in every element, and a
  ``*args: Tensor`` is a tensor in every one.  A union that also admits a number
  (``Tensor | float``, ``TensorOrScalar``) is the function's to judge.
"""

import annotationlib
import functools
import inspect
import re
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import override

from lucid import _promotion

_MISSING = object()

#: The file whose frames a ``warnings.warn(..., skip_file_prefixes=GUARD_FRAMES)``
#: does not count: a guarded function runs one frame deeper than the function
#: itself, which would move a warning's ``stacklevel`` onto the guard.
GUARD_FRAMES: tuple[str, ...] = (__file__,)

#: Attribute a guarded function carries, so a second pass leaves it alone.
_MARK = "__tensor_args_guarded__"

#: ``Tensor`` once :func:`_tensor_class` has loaded it; ``lucid._tensor`` sits
#: above this module, so it cannot be imported while this one is.
_TENSOR: type | None = None


def _tensor_class() -> type:
    global _TENSOR
    if _TENSOR is None:
        from lucid._tensor.tensor import Tensor

        _TENSOR = Tensor
    return _TENSOR


def _is_operand(value: object) -> bool:
    """Whether ``value`` is a tensor (or the engine's own handle for one)."""
    return isinstance(value, _tensor_class()) or _promotion.try_unwrap(value) is not None


def require_tensor(
    op: str, name: str, value: object, position: int | None = None, *, number: bool = False
) -> None:
    """Refuse ``value``, the argument ``name`` of ``op``, unless it is a tensor.

    Parameters
    ----------
    op : str
        Name of the function, as the message spells it.
    name : str
        Parameter the value arrived as.
    value : object
        The argument.
    position : int, optional
        Zero-based place of the argument, when it was given positionally.
    number : bool, default=False
        Whether a Python number is as good as a tensor here.

    Raises
    ------
    TypeError
        If ``value`` is not a ``Tensor`` (nor, with ``number``, a number).
    """
    if _is_operand(value) or (
        number and isinstance(_promotion.as_number(value), _promotion.NUMBER)
    ):
        return
    where = f" (position {position + 1})" if position is not None else ""
    wanted = "Tensor or Number" if number else "Tensor"
    raise TypeError(
        f"{op}(): argument '{name}'{where} must be {wanted}, not {type(value).__name__}"
    )


def slot_name(names: Sequence[str], index: int, default: str) -> str:
    """The advertised name of operand ``index``, or ``default`` past the named ones.

    Parameters
    ----------
    names : sequence of str
        The names the op's signature gives its leading tensor operands.
    index : int
        Zero-based place of the operand.
    default : str
        The name for an operand ``names`` does not reach.

    Returns
    -------
    str
        The name a refusal calls the operand.
    """
    return names[index] if index < len(names) else default


def require_tensors(op: str, name: str, values: object, position: int | None = None) -> None:
    """Refuse the sequence ``values``, the argument ``name`` of ``op``, unless
    every element is a tensor.

    Parameters
    ----------
    op : str
        Name of the function, as the message spells it.
    name : str
        Parameter the sequence arrived as.
    values : object
        The argument.
    position : int, optional
        Zero-based place of the argument, when it was given positionally.

    Raises
    ------
    TypeError
        If ``values`` is not a sequence, or an element is not a ``Tensor``.
    """
    where = f"argument {position}" if position is not None else f"argument '{name}'"
    if not isinstance(values, (list, tuple)) and (
        isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
    ):
        raise TypeError(
            f"{op}(): argument '{name}' must be a sequence of Tensors, not {type(values).__name__}"
        )
    tensor = _tensor_class()
    for index, element in enumerate(values):
        if not isinstance(element, tensor) and _promotion.try_unwrap(element) is None:
            raise TypeError(
                f"{op}(): expected Tensor as element {index} in {where}, "
                f"but got {type(element).__name__}"
            )


class _Slot(ABC):
    """One tensor-bearing parameter of a function, and how to check it.

    ``position`` is where a positional call puts the argument — ``None`` for a
    keyword-only parameter, which only a keyword reaches.
    """

    __slots__ = ("name", "position", "tensor")

    def __init__(self, name: str, position: int | None) -> None:
        self.name = name
        self.position = position
        self.tensor = _tensor_class()

    def argument(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        """What the call bound to this parameter, or ``_MISSING`` if it left it to a default."""
        if self.position is not None and self.position < len(args):
            return args[self.position]
        return kwargs.get(self.name, _MISSING)

    def given_at(self, args: tuple[object, ...]) -> int | None:
        """The place the call gave the argument, or ``None`` if it named it."""
        if self.position is not None and self.position < len(args):
            return self.position
        return None

    @abstractmethod
    def check(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        """Raise ``TypeError`` if the call put a non-tensor in this parameter."""


class _One(_Slot):
    """``Tensor``; ``| None`` when ``optional``; ``| Scalar`` when ``number``."""

    __slots__ = ("number", "optional")

    def __init__(self, name: str, position: int | None, *, optional: bool, number: bool) -> None:
        super().__init__(name, position)
        self.optional = optional
        self.number = number

    @override
    def check(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        position = self.position
        if position is not None and position < len(args):
            value = args[position]
        else:
            value = kwargs.get(self.name, _MISSING)
            position = None
            if value is _MISSING:
                return
        if isinstance(value, self.tensor) or (value is None and self.optional):
            return
        require_tensor(op, self.name, value, position, number=self.number)


class _Like(_Slot):
    """``Tensor | TensorLike``: anything a tensor can be built from, but not ``None``.

    The engine's converter takes whatever else the caller hands over and fails
    with its own error; ``None`` is the one value that builds a tensor of
    nothing, so it is the one refused here.
    """

    __slots__ = ()

    @override
    def check(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        if self.argument(args, kwargs) is None:
            where = self.given_at(args)
            at = f" (position {where + 1})" if where is not None else ""
            raise TypeError(
                f"{op}(): argument '{self.name}'{at} must be Tensor or array-like, not NoneType"
            )


class _Many(_Slot):
    """``Sequence[Tensor]``, or that ``| None`` when ``optional``."""

    __slots__ = ("optional",)

    def __init__(self, name: str, position: int | None, optional: bool) -> None:
        super().__init__(name, position)
        self.optional = optional

    @override
    def check(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        value = self.argument(args, kwargs)
        if value is _MISSING or (value is None and self.optional):
            return
        require_tensors(op, self.name, value, self.given_at(args))


class _Rest(_Slot):
    """``*name: Tensor`` — every positional argument from ``position`` on."""

    __slots__ = ()

    @override
    def check(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        for index, value in enumerate(args[self.position or 0 :]):
            if not _is_operand(value):
                raise TypeError(
                    f"{op}(): expected Tensor as element {index} in argument "
                    f"'{self.name}', but got {type(value).__name__}"
                )


_SEQUENCE_OF_TENSORS = re.compile(
    r"(?:Sequence|list|tuple|Iterable|collections\.abc\.Sequence)\[\s*Tensor\s*(?:,\s*\.\.\.\s*)?\]"
)

#: The annotation names that stand for "a Python number" beside ``Tensor``.
_NUMBER_NAMES = frozenset({"Scalar", "int", "float", "bool", "complex"})


def _classify(text: str) -> tuple[str, bool] | None:
    """``("one" | "number" | "like" | "many", optional)`` for the annotation source ``text``, or ``None``.

    Only an annotation that says nothing but "a tensor" (``"one"``), "a tensor
    or a number" (``"number"``) or "a sequence of tensors" (``"many"``)
    qualifies; one that also names a string, a dtype or another type leaves the
    argument to the function.
    """
    parts = [p.strip() for p in text.split("|")]
    optional = "None" in parts
    parts = [p for p in parts if p != "None"]
    if parts == ["Tensor"]:
        return "one", optional
    if parts == ["TensorOrScalar"] or (
        "Tensor" in parts and set(parts) - {"Tensor"} <= _NUMBER_NAMES
    ):
        return "number", optional
    if set(parts) == {"Tensor", "TensorLike"}:
        return "like", optional
    if len(parts) == 1 and _SEQUENCE_OF_TENSORS.fullmatch(parts[0]):
        return "many", optional
    return None


def _slot_for(param: inspect.Parameter, position: int) -> _Slot | None:
    """The check ``param`` (the ``position``-th of its function) calls for, or ``None``."""
    text = param.annotation if isinstance(param.annotation, str) else ""
    if param.kind is param.VAR_POSITIONAL:
        return _Rest(param.name, position) if text.strip() == "Tensor" else None
    kind = _classify(text) if text else None
    if kind is None:
        return None
    shape, optional = kind
    positional = param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    place = position if positional else None
    if shape == "many":
        return _Many(param.name, place, optional)
    if shape == "like":
        return None if optional else _Like(param.name, place)
    return _One(param.name, place, optional=optional, number=shape == "number")


def _slots_of(fn: Callable[..., object]) -> tuple[_Slot, ...]:
    """The tensor-bearing parameters of ``fn``, read from its annotations."""
    try:
        parameters = inspect.signature(fn, annotation_format=annotationlib.Format.STRING).parameters
    except (TypeError, ValueError):
        return ()
    slots: list[_Slot] = []
    for position, param in enumerate(parameters.values()):
        if param.name != "self" and (slot := _slot_for(param, position)) is not None:
            slots.append(slot)
        if param.kind is param.VAR_POSITIONAL:
            break
    return tuple(slots)


def _guarded[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """``fn`` behind a check of its tensor parameters (read on the first call)."""
    slots: tuple[_Slot, ...] | None = None
    op = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal slots
        if slots is None:
            slots = _slots_of(fn)
        for slot in slots:
            slot.check(op, args, kwargs)
        return fn(*args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


#: The guard of each function, so one function has one guard however many
#: namespaces re-export it: ``pickle`` finds a function by ``__module__`` and
#: ``__qualname__`` and refuses it unless that lookup returns the very object.
_GUARDS: dict[types.FunctionType, Callable[..., object]] = {}


def _owner_slot(fn: types.FunctionType) -> tuple[object, str] | None:
    """The object and attribute name under which ``fn`` is found by ``__module__`` and ``__qualname__``."""
    if "<locals>" in fn.__qualname__:
        return None
    *path, last = fn.__qualname__.split(".")
    owner: object = sys.modules.get(fn.__module__)
    for part in path:
        owner = getattr(owner, part, None)
    return (owner, last) if owner is not None and vars(owner).get(last) is fn else None


def guard_functions(namespace: types.ModuleType | type, names: Sequence[str]) -> None:
    """Put the tensor-argument check in front of the functions ``names`` of ``namespace``.

    The guard is installed where the function is defined as well as in
    ``namespace``, and one function has one guard: ``module.name`` is the
    function whichever door it is reached by, a Tensor method importing its
    composite, a package re-exporting it, or ``pickle`` looking it up by name.
    A name that is not a plain Python function (a class, a constant, an engine
    builtin) is left as it is, and a function already guarded is not wrapped
    twice.

    Parameters
    ----------
    namespace : module or type
        Where the functions are found, and where the guarded ones are put.
    names : sequence of str
        The names to guard.
    """
    for name in names:
        fn = vars(namespace).get(name)
        if not isinstance(fn, types.FunctionType) or getattr(fn, _MARK, False):
            continue
        guard = _GUARDS.get(fn)
        if guard is None:
            guard = _GUARDS[fn] = _guarded(fn)
            if (home := _owner_slot(fn)) is not None:
                owner, attr = home
                setattr(owner, attr, guard)
        setattr(namespace, name, guard)


def guard_namespace(name: str, names: Sequence[str]) -> None:
    """:func:`guard_functions` for the already-imported module called ``name``.

    For a module that guards itself as its last statement:
    ``guard_namespace(__name__, __all__)``.

    Parameters
    ----------
    name : str
        Name of the module, as ``sys.modules`` knows it.
    names : sequence of str
        The names in it to guard.
    """
    guard_functions(sys.modules[name], names)


def guard_methods(cls: type) -> None:
    """Put the tensor-argument check in front of the public methods ``cls`` defines.

    The methods the op registry generates are left out — they bind their own
    operands (:func:`require_tensor`) — and so is anything that is not a plain
    function: a property, a static or class method, a dunder.  ``self`` is not
    checked.
    """
    guard_functions(
        cls,
        [
            name
            for name, fn in vars(cls).items()
            if not name.startswith("_")
            and isinstance(fn, types.FunctionType)
            and "<locals>" not in fn.__qualname__
        ],
    )
