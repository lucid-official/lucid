"""Axis semantics of :func:`rearrange`, :func:`reduce` and :func:`repeat`.

The three entry points share one grammar and one plan, as in the reference
einops: a pattern is parsed into elementary axes, every axis length is
resolved against the input shape and the keyword lengths, and the call is
lowered to at most six engine steps — reshape, reduce, permute, insert and
broadcast new axes, reshape.

The engine kernels this replaces parsed patterns themselves and owned none
of the semantics a pattern carries.  A keyword length was overwritten by
the input's dimension rather than checked against it, ``...`` was a syntax
error, a literal or a group of groups reached a ``std::get`` on the wrong
alternative (``bad_variant_access``), and a malformed pattern surfaced as
whichever reshape or permute it broke.  Here the whole contract is checked
once, at the public boundary, and every refusal names the axis and the
pattern it came from.

Parsing is pure and depends only on the pattern, the operation, the
keyword names and the input rank, so it is cached; resolving the lengths
depends on the shape as well and is cached under that key, so a call that
repeats a pattern on a repeated shape costs a dictionary lookup.
"""

import functools
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import SupportsIndex, override

from lucid._C import engine as _C_engine

# Stands in for ``...`` while parsing: one character, so it tokenises as a
# name.  A pattern that contains it is refused before parsing
# (``_split_pattern``), or a hand-written one would pass for an ellipsis.
_ELLIPSIS = "…"

_REDUCTIONS = ("min", "max", "sum", "mean", "prod", "any", "all")

# Elementary-axis length placeholders, resolved per shape.
_UNKNOWN = -1  # inferred from the input dimension it is part of
_KEYWORD = -2  # given by the caller as a keyword length


class _Refusal(Exception):
    """A pattern or shape the operation cannot take; ``kind`` is its class."""

    def __init__(self, kind: type[Exception], message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _bad_pattern(message: str) -> _Refusal:
    return _Refusal(_C_engine.InvalidArgument, message)


def _bad_shape(message: str) -> _Refusal:
    return _Refusal(_C_engine.ShapeMismatch, message)


class _Anonymous:
    """An axis written as a number greater than 1.

    Every occurrence is a distinct axis — ``(h 2) (w 2)`` has two — so
    identity, not value, is what tells two apart.
    """

    __slots__ = ("length",)

    def __init__(self, length: int) -> None:
        self.length = length

    @override
    def __repr__(self) -> str:
        return str(self.length)


_Axis = str | _Anonymous


@dataclass
class _Side:
    """One side of a pattern.

    ``composition`` holds one entry per tensor dimension: a list of the
    elementary axes merged into it (empty for a unit axis written ``1`` or
    ``()``), or the bare ellipsis marker.  Inside a group the marker is
    one of the group's members.
    """

    composition: list[list[_Axis] | str]
    identifiers: list[_Axis]
    has_ellipsis: bool = False
    ellipsis_grouped: bool = False
    has_anonymous: bool = False


def _axis_name_problem(name: str) -> str | None:
    if not name.isidentifier():
        return "not a valid identifier"
    if name.startswith("_") or name.endswith("_"):
        return "an axis name may not start or end with an underscore"
    return None


def _plain(axis: _Axis) -> str:
    return str(axis) if isinstance(axis, _Anonymous) else axis.replace(_ELLIPSIS, "...")


def _spelled(axis: _Axis) -> str:
    return _plain(axis) if isinstance(axis, _Anonymous) else repr(_plain(axis))


def _describe(axes: Sequence[_Axis]) -> str:
    return ", ".join(_spelled(a) for a in axes)


def _axes_appear(axes: Sequence[_Axis]) -> str:
    """``axis 'a' appears`` or ``axes 'a', 'b' appear``, as the count asks."""
    if len(axes) == 1:
        return f"axis {_describe(axes)} appears"
    return f"axes {_describe(axes)} appear"


class _SideParser:
    """Tokeniser and grammar for one side of a pattern."""

    def __init__(self, text: str, label: str) -> None:
        self.label = label
        self.side = _Side(composition=[], identifiers=[])
        self.group: list[_Axis] | None = None
        self.text = self._replace_ellipsis(text)

    def _replace_ellipsis(self, text: str) -> str:
        if "." not in text:
            return text
        if "..." not in text or text.count(".") != 3:
            raise _bad_pattern(f"the {self.label} side may contain '.' only as one '...'")
        self.side.has_ellipsis = True
        return text.replace("...", _ELLIPSIS)

    def parse(self) -> _Side:
        token = ""
        for char in self.text:
            if char in "() " or char.isspace():
                if token:
                    self._add(token)
                token = ""
                self._bracket(char)
            elif char.isalnum() or char in ("_", _ELLIPSIS):
                token += char
            else:
                raise _bad_pattern(f"unexpected character {char!r} on the {self.label} side")
        if self.group is not None:
            raise _bad_pattern(f"unbalanced parentheses on the {self.label} side")
        if token:
            self._add(token)
        return self.side

    def _bracket(self, char: str) -> None:
        if char == "(":
            if self.group is not None:
                raise _bad_pattern(
                    f"nested parentheses on the {self.label} side: axis "
                    "composition is one level deep, write ((a b) c) as (a b c)"
                )
            self.group = []
        elif char == ")":
            if self.group is None:
                raise _bad_pattern(f"unbalanced parentheses on the {self.label} side")
            self.side.composition.append(self.group)
            self.group = None

    def _add(self, token: str) -> None:
        if token == _ELLIPSIS:
            self._add_ellipsis()
        elif token.isdecimal():
            self._add_number(int(token))
        else:
            problem = _axis_name_problem(token)
            if problem is not None:
                raise _bad_pattern(f"{_spelled(token)} is not a valid axis name: {problem}")
            if token in self.side.identifiers:
                raise _bad_pattern(
                    f"axis {token!r} appears more than once on the {self.label} side"
                )
            self._place(token)

    def _add_ellipsis(self) -> None:
        self.side.identifiers.append(_ELLIPSIS)
        if self.group is None:
            self.side.composition.append(_ELLIPSIS)
        else:
            self.group.append(_ELLIPSIS)
            self.side.ellipsis_grouped = True

    def _add_number(self, value: int) -> None:
        if value == 1:
            # A unit axis: a dimension of its own outside a group, and
            # nothing inside one (it does not change the group's product).
            if self.group is None:
                self.side.composition.append([])
            return
        if value < 1:
            raise _bad_pattern(
                f"an anonymous axis must have length 1 or more, got {value} "
                f"on the {self.label} side"
            )
        self.side.has_anonymous = True
        self._place(_Anonymous(value))

    def _place(self, axis: _Axis) -> None:
        self.side.identifiers.append(axis)
        if self.group is None:
            self.side.composition.append([axis])
        else:
            self.group.append(axis)


def _parse(text: str, label: str) -> _Side:
    return _SideParser(text, label).parse()


@dataclass(frozen=True, slots=True)
class _Recipe:
    """A pattern resolved to elementary axes, independent of the shape.

    Elementary axes are numbered in the order they appear on the left,
    followed by the axes only the right side introduces.  ``lengths`` holds
    each one's length where the pattern fixes it, or a placeholder.
    """

    names: tuple[_Axis, ...]
    lengths: tuple[int, ...]
    keyword_slots: tuple[tuple[str, int], ...]
    # Per input dimension: its elementary axes as written, then the known
    # and the unknown ones among them.
    inputs: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
    n_left: int
    # Positions, among the left's elementary axes, of the ones reduced away.
    reduced: tuple[int, ...]
    # The surviving axes, in left order, rearranged into right order.
    permutation: tuple[int, ...]
    # (position in the flat output, elementary axis) for each new axis.
    added: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[int, ...], ...]


def _split_pattern(pattern: str) -> tuple[str, str]:
    if _ELLIPSIS in pattern:
        # The marker the parser stands in for ``...``: written by hand it
        # would read as an ellipsis that no '...' announced, unchecked.
        raise _bad_pattern(
            "'…' (U+2026) is not part of the grammar; write the ellipsis as " "three dots, '...'"
        )
    if pattern.count("->") != 1:
        raise _bad_pattern("a pattern needs exactly one '->'")
    left, right = pattern.split("->")
    return left, right


def _missing(axes: list[_Axis], side: _Side) -> list[_Axis]:
    return [a for a in axes if a not in side.identifiers]


def _check_rearrange(left: _Side, right: _Side) -> None:
    if left.has_anonymous or right.has_anonymous:
        raise _bad_pattern(
            "rearrange takes no anonymous axes longer than 1; use repeat to "
            "add an axis or reduce to remove one"
        )
    only = _missing(left.identifiers, right) + _missing(right.identifiers, left)
    if only:
        raise _bad_pattern(
            f"{_axes_appear(only)} on only one side; rearrange needs every axis on both"
        )


def _check_repeat(left: _Side, right: _Side, keywords: set[str]) -> None:
    dropped = _missing(left.identifiers, right)
    if dropped:
        raise _bad_pattern(
            f"{_axes_appear(dropped)} only on the left side; repeat cannot remove axes"
        )
    unsized = [
        a for a in _missing(right.identifiers, left) if isinstance(a, str) and a not in keywords
    ]
    if unsized:
        raise _bad_pattern(f"new axes {_describe(unsized)} need lengths; pass them as keywords")


def _check_axes_agree(left: _Side, right: _Side, operation: str, keywords: set[str]) -> None:
    """The reference's rules for which axes may appear on which side."""
    if right.has_ellipsis and not left.has_ellipsis:
        raise _bad_pattern("'...' appears on the right side but not on the left")
    if left.ellipsis_grouped:
        raise _bad_pattern("'...' may not stand inside parentheses on the left side")
    if operation == "rearrange":
        _check_rearrange(left, right)
    elif operation == "repeat":
        _check_repeat(left, right, keywords)
    elif operation not in _REDUCTIONS:
        raise _bad_pattern(
            f"unknown reduction {operation!r}; expected one of {', '.join(_REDUCTIONS)}"
        )
    elif created := _missing(right.identifiers, left):
        raise _bad_pattern(
            f"{_axes_appear(created)} only on the right side; reduce cannot create axes"
        )


def _spell_out(side: _Side, covered: list[_Axis]) -> list[list[_Axis]]:
    """``side``'s dimensions as groups, with ``...`` replaced by ``covered``."""
    out: list[list[_Axis]] = []
    for entry in side.composition:
        if entry == _ELLIPSIS:
            out.extend([axis] for axis in covered)
        elif isinstance(entry, list):
            group: list[_Axis] = []
            for axis in entry:
                group.extend(covered if axis == _ELLIPSIS else [axis])
            out.append(group)
    return out


def _expand_ellipsis(
    left: _Side, right: _Side, ndim: int
) -> tuple[list[list[_Axis]], list[list[_Axis]]]:
    """Spell ``...`` out as the elementary axes it covers in this input."""
    fixed = len(left.composition) - (1 if left.has_ellipsis else 0)
    if not left.has_ellipsis and ndim != fixed:
        raise _bad_shape(f"the pattern names {fixed} dimensions, the input has {ndim}")
    if ndim < fixed:
        raise _bad_shape(f"the pattern needs at least {fixed} dimensions, the input has {ndim}")
    covered: list[_Axis] = [f"{_ELLIPSIS}{i}" for i in range(ndim - fixed)]
    for side in (left, right):
        if side.has_ellipsis:
            side.identifiers.remove(_ELLIPSIS)
            side.identifiers.extend(covered)
    return _spell_out(left, covered), _spell_out(right, covered)


def _declare(left_groups: list[list[_Axis]], right: _Side) -> tuple[list[_Axis], list[int], int]:
    """Number the elementary axes: the left's in order, then the right's new ones."""
    names: list[_Axis] = []
    lengths: list[int] = []
    for axis in [a for group in left_groups for a in group] + right.identifiers:
        if axis not in names:
            names.append(axis)
            lengths.append(axis.length if isinstance(axis, _Anonymous) else _UNKNOWN)
    return names, lengths, sum(len(group) for group in left_groups)


def _keyword_slots(
    keywords: tuple[str, ...], names: list[_Axis], lengths: list[int]
) -> tuple[tuple[str, int], ...]:
    """Where each keyword length goes; marks those axes as given."""
    slots: list[tuple[str, int]] = []
    for name in keywords:
        problem = _axis_name_problem(name)
        if problem is not None:
            raise _bad_pattern(f"{name!r} is not a valid axis name: {problem}")
        if name not in names:
            raise _bad_pattern(f"axis {name!r} is given a length but not used in the pattern")
        slots.append((name, names.index(name)))
        lengths[names.index(name)] = _KEYWORD
    return tuple(slots)


def _input_dims(
    left_groups: list[list[_Axis]], names: list[_Axis], lengths: list[int]
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    """Per input dimension, its members and which of them must be inferred."""
    dims = []
    for group in left_groups:
        members = tuple(names.index(a) for a in group)
        known = tuple(i for i in members if lengths[i] != _UNKNOWN)
        unknown = tuple(i for i in members if lengths[i] == _UNKNOWN)
        if len(unknown) > 1:
            raise _bad_pattern(
                f"cannot infer the lengths of {_describe([names[i] for i in unknown])} "
                "from one dimension; give all but one of them as keywords"
            )
        dims.append((members, known, unknown))
    return tuple(dims)


@functools.lru_cache(maxsize=512)
def _recipe(pattern: str, operation: str, keywords: tuple[str, ...], ndim: int) -> _Recipe:
    left_text, right_text = _split_pattern(pattern)
    left = _parse(left_text, "left")
    right = _parse(right_text, "right")
    _check_axes_agree(left, right, operation, set(keywords))
    left_groups, right_groups = _expand_ellipsis(left, right, ndim)
    names, lengths, n_left = _declare(left_groups, right)
    slots = _keyword_slots(keywords, names, lengths)
    inputs = _input_dims(left_groups, names, lengths)

    # Reducing before permuting keeps a reduction over the input's own
    # layout: a permuted view reduced along its last axis made the CPU
    # reduction walk a strided copy, eight times slower for "b n d -> b d".
    flat_left = [a for group in left_groups for a in group]
    flat_right = [a for group in right_groups for a in group]
    survivors = [a for a in flat_left if a in right.identifiers]
    return _Recipe(
        names=tuple(names),
        lengths=tuple(lengths),
        keyword_slots=slots,
        inputs=inputs,
        n_left=n_left,
        reduced=tuple(i for i, a in enumerate(flat_left) if a not in right.identifiers),
        permutation=tuple(survivors.index(a) for a in flat_right if a in survivors),
        added=tuple(
            (i, names.index(a)) for i, a in enumerate(flat_right) if a not in left.identifiers
        ),
        outputs=tuple(tuple(names.index(a) for a in group) for group in right_groups),
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    """The engine steps for one pattern on one shape; ``None`` skips a step."""

    split: tuple[int, ...] | None
    reduced: tuple[int, ...]
    permutation: tuple[int, ...] | None
    inserted: tuple[int, ...]
    broadcast: tuple[int, ...] | None
    merge: tuple[int, ...] | None


def _product(values: list[int]) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def _resolve_lengths(
    recipe: _Recipe, shape: tuple[int, ...], keywords: tuple[tuple[str, int], ...]
) -> tuple[list[int], bool]:
    """Every elementary axis length, and whether the input must be split."""
    lengths = list(recipe.lengths)
    given = dict(keywords)
    for name, slot in recipe.keyword_slots:
        lengths[slot] = given[name]
    must_split = False
    for dim, (members, known, unknown) in enumerate(recipe.inputs):
        length = shape[dim]
        if len(members) != 1:
            must_split = True
        if not known and len(unknown) == 1:
            lengths[unknown[0]] = length
            continue
        known_product = _product([lengths[i] for i in known])
        if not unknown and length != known_product:
            raise _bad_shape(_mismatch(recipe, dim, length, members, lengths))
        if unknown:
            if known_product == 0 or length % known_product != 0:
                written = " ".join(_plain(recipe.names[i]) for i in members)
                raise _bad_shape(
                    f"input dimension {dim} of length {length} cannot be split "
                    f"into ({written}): the known part has length {known_product}"
                )
            lengths[unknown[0]] = length // known_product
    return lengths, must_split


def _mismatch(
    recipe: _Recipe, dim: int, length: int, members: tuple[int, ...], lengths: list[int]
) -> str:
    """What to say when dimension ``dim`` is not the length the pattern fixes."""
    found = f"input dimension {dim} has length {length}"
    if not members:
        return f"{found}, but the pattern has a unit axis there"
    axis = recipe.names[members[0]] if len(members) == 1 else None
    if isinstance(axis, str):
        given = lengths[members[0]]
        return f"axis {axis!r} has length {length} in the input, but {axis}={given} was given"
    written = " ".join(f"{_plain(recipe.names[i])}={lengths[i]}" for i in members)
    product = _product([lengths[i] for i in members])
    return f"{found}, but ({written}) multiply to {product}"


@functools.lru_cache(maxsize=1024)
def _plan(
    pattern: str, operation: str, shape: tuple[int, ...], keywords: tuple[tuple[str, int], ...]
) -> _Plan:
    recipe = _recipe(pattern, operation, tuple(name for name, _ in keywords), len(shape))
    lengths, must_split = _resolve_lengths(recipe, shape, keywords)
    permutation = recipe.permutation
    return _Plan(
        split=tuple(lengths[: recipe.n_left]) if must_split else None,
        reduced=recipe.reduced,
        permutation=(None if permutation == tuple(range(len(permutation))) else permutation),
        inserted=tuple(pos for pos, _ in recipe.added),
        broadcast=(
            tuple(lengths[i] for group in recipe.outputs for i in group) if recipe.added else None
        ),
        merge=(
            tuple(_product([lengths[i] for i in group]) for group in recipe.outputs)
            if any(len(group) != 1 for group in recipe.outputs)
            else None
        ),
    )


def _length(name: str, value: SupportsIndex) -> int:
    try:
        length = operator.index(value)
    except TypeError:
        raise TypeError(
            f"the length of axis {name!r} must be an integer, got {type(value).__name__}"
        ) from None
    if length < 0:
        raise _bad_pattern(f"the length of axis {name!r} must be non-negative, got {length}")
    return length


def plan(
    pattern: str, operation: str, shape: tuple[int, ...], axes_lengths: Mapping[str, SupportsIndex]
) -> _Plan:
    """The engine steps for ``operation`` with ``pattern`` on ``shape``.

    Parameters
    ----------
    pattern : str
        The pattern as the caller wrote it.
    operation : str
        ``"rearrange"``, ``"repeat"``, or the name of a reduction.
    shape : tuple of int
        The input's shape.
    axes_lengths : Mapping of str to int
        The keyword lengths, validated here.

    Returns
    -------
    _Plan
        The steps to run, each ``None`` (or empty) when it is not needed.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The pattern is malformed, or names axes the operation cannot take.
    lucid._C.engine.ShapeMismatch
        The input's shape does not fit the pattern or the keyword lengths.
    TypeError
        A keyword length is not an integer.
    """
    try:
        keywords = tuple((name, _length(name, v)) for name, v in axes_lengths.items())
        return _plan(pattern, operation, shape, keywords)
    except _Refusal as refusal:
        context = f"einops.{_entry_point(operation)}({pattern!r}) on shape {shape}"
        if axes_lengths:
            context += " with " + ", ".join(f"{n}={v!r}" for n, v in axes_lengths.items())
        raise refusal.kind(f"{context}: {refusal}") from None


def _entry_point(operation: str) -> str:
    return operation if operation in ("rearrange", "repeat") else "reduce"


__all__ = ["plan"]
