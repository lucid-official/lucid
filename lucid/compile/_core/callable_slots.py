"""
lucid.compile._core.callable_slots — what a function reads that no argument hands it.

A function's closure cells and the globals its code names are state a trace
bakes in.  :func:`callable_slots` lists them with the keys
:func:`read_slot` re-reads them by; the compile key (``signature``) and the
hook contract (``hook_contract``) both follow them.
"""

import weakref
from functools import partial
from types import CodeType

__all__ = ["callable_slots", "cell_value", "global_names", "read_slot"]

# By the code object's address: hashing a code object hashes its bytecode and
# constants, ~2 µs a call for a large function.  An entry leaves with its code.
_GLOBAL_NAMES: dict[int, tuple[weakref.ref[CodeType], tuple[str, ...]]] = {}


def global_names(code: CodeType) -> tuple[str, ...]:
    """The names ``code`` and the code nested in it may read as globals, sorted.

    ``co_names`` holds attribute names too, so this over-approximates: a
    global that only shares a name with an attribute is watched for nothing
    worse than an extra trace.
    """
    key = id(code)
    hit = _GLOBAL_NAMES.get(key)
    if hit is not None:
        return hit[1]
    found = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, CodeType):
            found.update(global_names(const))
    names = tuple(sorted(found))
    gone = partial(_GLOBAL_NAMES.pop, key, None)
    _GLOBAL_NAMES[key] = (weakref.ref(code, lambda _: gone()), names)
    return names


def cell_value(cell: object) -> object:
    """A closure cell's contents; ``None`` while its variable is unbound.

    Parameters
    ----------
    cell : object
        A cell from a function's ``__closure__``.

    Returns
    -------
    object
        What the cell holds, or ``None`` for an empty cell (a variable the
        enclosing function assigns later).
    """
    try:
        return getattr(cell, "cell_contents", None)
    except ValueError:  # an empty cell: the variable is assigned later
        return None


def read_slot(fn: object, key: tuple[str, str]) -> object:
    """What ``fn``'s closure cell or global ``key`` holds now.

    Parameters
    ----------
    fn : object
        The function the slot belongs to.
    key : tuple of (str, str)
        ``("cell", index)`` or ``("global", name)``, as :func:`callable_slots` lists them.

    Returns
    -------
    object
        The slot's current value, or ``None`` if the cell or global is gone.
    """
    kind, name = key
    if kind == "cell":
        cells = getattr(fn, "__closure__", None) or ()
        return cell_value(cells[int(name)]) if int(name) < len(cells) else None
    space = getattr(fn, "__globals__", None)
    return space.get(name) if isinstance(space, dict) else None


def callable_slots(fn: object) -> list[tuple[tuple[str, str], object]]:
    """What ``fn`` reads that no argument hands it: its closure cells and globals.

    A global counts when ``fn``'s code names it (LCD-599): a module-level
    model is as much the function's state as one it closes over.  What a
    function ``fn`` calls reads in its own globals is not followed.

    Parameters
    ----------
    fn : object
        A function, or anything without ``__closure__`` / ``__code__`` (then empty).

    Returns
    -------
    list of ((str, str), object)
        ``(("cell", index), value)`` for each closure cell, then
        ``(("global", name), value)`` for each global the code names, which
        :func:`read_slot` re-reads by the same key.
    """
    out: list[tuple[tuple[str, str], object]] = [
        (("cell", str(i)), cell_value(cell))
        for i, cell in enumerate(getattr(fn, "__closure__", None) or ())
    ]
    code = getattr(fn, "__code__", None)
    space = getattr(fn, "__globals__", None)
    if isinstance(code, CodeType) and isinstance(space, dict):
        out.extend((("global", name), space[name]) for name in global_names(code) if name in space)
    return out
