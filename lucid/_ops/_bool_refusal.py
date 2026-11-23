"""The dtypes the reference's CPU turns away, and who refuses them (LCD-1013, -1020, -1026).

The reference's CPU backend turns a bool or an integer operand away from a long
list of ops -- a kernel with no instantiation for the dtype raises
``NotImplementedError``, a guard on the input dtype raises ``RuntimeError`` -- and
Lucid computed them instead, promoting the operand to a float behind the caller's
back (``mask.mean()``, ``ids.std()``, ``softmax(ids)``).  A bool is a truth value
and an index is a position, not a count or a magnitude, so the program that meant
a number casts at the call site (``mask.float().mean()``) and the one that did not
learns of it where it happened.

The rule has one owner per layer, and an op has one of them (two exceptions, below):

* An op with an engine schema (``abs``, ``relu``, ``softmax``, ``mean``, the pools
  and norms, ...) is refused by the engine: its ``OpSchema`` declares the
  refusal and ``SchemaGuard`` raises it for every caller -- the free function,
  the ``Tensor`` method, the in-place twin, ``nn.functional`` and a direct engine
  call alike.  :func:`engine_refusals` reads those declarations back out of the
  engine.
* An op with no schema of its own -- a composite built from other ops, a loss, an
  adapter whose answer depends on the form of the call (``pow`` takes a bool base
  to a number, ``clamp`` a single tensor bound) -- is refused by its Python entry,
  through :data:`ROWS`.  Each entry calls :func:`refuse` (or :func:`refusal`, for a
  check that decides for itself when to refuse) with the op's name, so the error
  class and the message never diverge between a free function, its ``Tensor``
  method and the in-place twin.

A row is keyed ``(op, role)``.  The role is ``""`` for the op's primary operand, and
names another operand when the reference answers it differently (``grid`` of
``grid_sample`` is a plain dtype ``RuntimeError`` where the image is a kernel's
``NotImplementedError``, ``mask`` of ``masked_select`` takes a bool and refuses an
integer).  A role with no row of its own takes the op's primary one.  A row says,
for each of bool, the signed integers and the signed integers narrower than int64
(the reference's ``avg_pool`` and ``one_hot`` take int64 and refuse the rest), the
class the reference raises and the cast to name -- or nothing, where the reference
answers.

The owner is one per ``(op, dtype)``: the engine's for the dtypes its schema declares and
Python's for the rest (``rms_norm`` and ``embedding_bag`` are the engine's for integers and
the entry's for bool).  An entry that reaches the engine op only on some of its paths asks the
engine's declaration back with :func:`refuse_declared` instead of declaring it again
(``grid_sample``, ``embedding_bag``, ``rms_norm``).  An op's *role* may be Python's where its
primary operand is the engine's (the ``attn_mask`` of attention, whose queries and keys the
engine checks).  The bypass is the caller's: a pipeline that resamples a mask casts it itself
(``lucid.utils.transforms.functional._sampled``).

Neither list is a census of the ops that refuse.  The refusal sweeps
(``lucid/test/unit/ops/test_bool_refusal_sweep.py`` and its integer twin) derive
their candidates from the audit's op census and compare Lucid's answer with the
reference's recorded one, and ``lucid/test/unit/ops/test_bool_refusal_table.py`` and
``test_integer_refusal_table.py`` pin every row of both lists against the reference.

Cost.  The check runs on every call of an op that carries it, floating input
included, so :func:`refuse` reads the operand's dtype code once and returns at that
point for anything that is not bool or an integer -- no name is built and no table is
looked up on the path every float tensor takes.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dtype import _ENGINE_TO_DTYPE

#: The reference's ``NotImplementedError`` -- a kernel with no instantiation for the
#: dtype.  Lucid's class is also a ``RuntimeError``, as the reference's.
_KERNEL = _C_engine.NotImplementedError

#: The reference's plain ``RuntimeError`` -- a guard on the input dtype.  The
#: engine's dtype-refusal class is a ``RuntimeError`` (and a ``TypeError``),
#: and what ``mean(dtype=...)`` already raises for the dtype it refuses.
_GUARD = _C_engine.DtypeMismatch

#: The reference's ``ValueError`` -- a check Python itself makes on an argument
#: (``affine_grid``'s ``theta``, ``one_hot``'s class tensor).  The engine's argument
#: class is a ``ValueError`` and a ``RuntimeError``.
_INVALID = _C_engine.InvalidArgument

#: ``Dtype.Bool``'s integer code: ``impl.dtype.value == BOOL_CODE`` is 150 ns
#: cheaper than comparing the enum.
BOOL_CODE: int = _C_engine.Bool.value

# The integral dtypes in code order (bool, then the signed integers by width).
_INTEGRAL = sorted(
    (d for d in _ENGINE_TO_DTYPE if _promotion.is_integral(d)), key=lambda d: d.value
)

#: The largest integral dtype code.  Every dtype with a larger one is a float or complex,
#: which no row refuses, so ``code > LAST_INTEGRAL_CODE`` is the whole answer for them.
LAST_INTEGRAL_CODE: int = _INTEGRAL[-1].value

#: How a dtype is spelled in a message, by its code: the engine's ``dtype_name``
#: (``int32``), not the enum's (``I32``), so the two layers word one refusal alike.
_NAMES: tuple[str, ...] = tuple(
    name for _, name in sorted((d.value, dtype._name) for d, dtype in _ENGINE_TO_DTYPE.items())
)
_ARTICLES: tuple[str, ...] = tuple(
    f"{'an' if name.startswith('int') else 'a'} {name}" for name in _NAMES
)

# The cast hints.  The engine words the first three in ``OpSchema.h``
# (``cast_hint_text``); a test holds the two spellings equal.
_FLOATING = "a floating dtype, e.g. x.float()"
_INTEGER = "an integer dtype, e.g. x.long()"
_NUMBER = "an integer or floating dtype, e.g. x.long() or x.float()"
_INT64 = "int64, e.g. x.long()"
_BOOL_MASK = "a bool dtype, e.g. mask.bool()"

#: What a role is called in a message where it is not an operand's own name.
_ROLE_PHRASE: Mapping[str, str] = MappingProxyType(
    {"mixed": "operand beside another dtype", "global": "input of a global average"}
)


class Rule(NamedTuple):
    """What the reference does with a dtype an op turns away.

    A named pair, so a rule compares equal to the ``(error class, cast hint)`` tuple
    :data:`BOOL_REFUSED` has always held.
    """

    #: The class the reference raises, as Lucid names it.
    error: type[RuntimeError]
    #: The cast the message offers in place of the refused dtype.
    hint: str


KERNEL_FLOATING = Rule(_KERNEL, _FLOATING)
KERNEL_NUMBER = Rule(_KERNEL, _NUMBER)
KERNEL_INTEGER = Rule(_KERNEL, _INTEGER)
GUARD_FLOATING = Rule(_GUARD, _FLOATING)
GUARD_NUMBER = Rule(_GUARD, _NUMBER)
GUARD_INTEGER = Rule(_GUARD, _INTEGER)
GUARD_BOOL = Rule(_GUARD, _BOOL_MASK)
INVALID_FLOATING = Rule(_INVALID, _FLOATING)
INVALID_INT64 = Rule(_INVALID, _INT64)


@dataclass(frozen=True, slots=True)
class Row:
    """One operand of one op: which of its dtypes are refused, and how.

    A slot left ``None`` is a dtype the reference answers.  ``on_ints`` and
    ``on_narrow`` are two spellings of one question -- the widths the rule covers --
    so a row sets at most one of them.
    """

    op: str
    role: str = ""
    #: The rule for a bool operand.
    on_bool: Rule | None = None
    #: The rule for every signed integer (int8, int16, int32, int64).
    on_ints: Rule | None = None
    #: The rule for the signed integers narrower than int64 only.
    on_narrow: Rule | None = None
    #: The rule for a float operand.  Only an operand that must be a bool and nothing else
    #: (a mask) sets it, and only :func:`refuse_unless_bool` reads it: every other check
    #: returns at once for a float.
    on_float: Rule | None = None
    #: The rule for a complex operand, read by :func:`refuse_complex` and
    #: :func:`refuse_unless_bool`.
    on_complex: Rule | None = None

    def __post_init__(self) -> None:
        """Refuse a row that sets both integer slots."""
        if self.on_ints is not None and self.on_narrow is not None:
            raise ValueError(f"{self.op}: a row sets on_ints or on_narrow, not both")

    @property
    def key(self) -> tuple[str, str]:
        """``(op, role)``, the row's place in :data:`ROWS`."""
        return (self.op, self.role)

    def by_code(self) -> tuple[Rule | None, ...]:
        """The rule for each integral dtype, indexed by its engine code."""
        slots: list[Rule | None] = [None] * (LAST_INTEGRAL_CODE + 1)
        for dtype in _INTEGRAL:
            if dtype.value == BOOL_CODE:
                slots[dtype.value] = self.on_bool
            elif dtype.value == LAST_INTEGRAL_CODE:
                slots[dtype.value] = self.on_ints
            else:
                slots[dtype.value] = self.on_ints or self.on_narrow
        return tuple(slots)


def _row(
    op: str,
    role: str = "",
    *,
    bool: Rule | None = None,
    ints: Rule | None = None,
    narrow: Rule | None = None,
    floats: Rule | None = None,
    complexes: Rule | None = None,
) -> Row:
    """A :class:`Row`, spelled the way the table below reads."""
    return Row(
        op,
        role,
        on_bool=bool,
        on_ints=ints,
        on_narrow=narrow,
        on_float=floats,
        on_complex=complexes,
    )


#: The spatial ranks the pooling ops come in (``max_unpool1d`` .. ``3d``).
_RANKS = (1, 2, 3)

#: Every refusal this layer owns, one line per op and operand.  In-place twins
#: (``pow_``) take the row of their out-of-place op, and an alias (``clip``,
#: ``upsample``) the row of the op it forwards to.
_DECLARED: tuple[Row, ...] = (
    # Order statistics and counting over a sorted sequence.  The reference answers an
    # integer to the searches, ``isin`` and the sorts.
    _row("quantile", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("nanquantile", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("searchsorted", bool=KERNEL_INTEGER),
    _row("bucketize", bool=KERNEL_INTEGER),
    _row("isin", bool=GUARD_INTEGER),
    # Moments.  ``mean``, ``var`` and ``std`` are the engine's.
    _row("std_mean", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("var_mean", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("nanmean", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("cov", bool=GUARD_NUMBER),
    _row("corrcoef", bool=GUARD_NUMBER),
    # The reference wants the input's dtype of ``bins`` and ``weight``: any other is a
    # dtype mismatch, not the kernel's refusal of the input.
    _row("histogram", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING, complexes=KERNEL_FLOATING),
    _row("histogramdd", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING, complexes=KERNEL_FLOATING),
    _row("histogram", "bins", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("histogram", "weight", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("gradient", bool=GUARD_NUMBER),
    _row("poisson", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("multinomial", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    # Elementwise.  ``pow`` refuses a bool base only beside a tensor exponent
    # (``mask ** True`` is a number), ``clamp`` a bool result of two bounds.
    _row("clamp", bool=KERNEL_INTEGER),
    _row("pow", bool=KERNEL_NUMBER),
    _row("frexp", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("polar", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    # Special functions and linear algebra.  ``gammainc`` takes an integer beside a float
    # (they promote) and refuses an integral pair.
    _row("gammainc", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("gammaincc", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("multigammaln", bool=GUARD_NUMBER),
    _row("vecdot", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("vander", bool=GUARD_NUMBER),
    _row("matrix_exp", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("ldl_solve", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    # Activations, dropout and normalisation composed from other ops.  ``rms_norm``'s
    # integers are the engine's (the schema declares them); only its bool is here.
    _row("celu", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("hardshrink", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("softshrink", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("tanhshrink", bool=GUARD_NUMBER),
    _row("hardtanh", bool=GUARD_NUMBER),
    _row("threshold", bool=KERNEL_NUMBER),
    _row("rrelu", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("prelu", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    # An input beside a weight of another dtype is a type promotion the reference refuses.
    _row("prelu", "mixed", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("glu", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("gumbel_softmax", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("feature_alpha_dropout", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("rms_norm", bool=KERNEL_FLOATING),
    _row("instance_norm", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    # The reference's ``local_response_norm`` takes int64 (its pooled sum is an average
    # pool, which has an int64 kernel) and refuses the rest.
    _row("local_response_norm", bool=KERNEL_FLOATING, narrow=KERNEL_FLOATING),
    # Pooling.  The average pool the reference's adaptive form is built on is a kernel
    # for every window and a dtype guard for a global (one-element) output.
    *(_row(f"max_unpool{n}d", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING) for n in _RANKS),
    *(_row(f"adaptive_avg_pool{n}d", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING) for n in _RANKS),
    *(
        _row(f"adaptive_avg_pool{n}d", "global", bool=GUARD_FLOATING, ints=GUARD_FLOATING)
        for n in _RANKS
    ),
    *(_row(f"adaptive_max_pool{n}d", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING) for n in _RANKS),
    # Initialisers and samplers drawn through a float tensor.
    _row("orthogonal", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("sparse", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("cauchy", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("exponential", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    # Indices and masks: the reference wants int64 class indices and a bool mask.
    _row("one_hot", bool=INVALID_INT64, narrow=INVALID_INT64),
    _row("masked_select", "mask", ints=GUARD_BOOL, floats=GUARD_BOOL, complexes=GUARD_BOOL),
    _row("scaled_dot_product_attention", "attn_mask", ints=GUARD_BOOL),
    # The engine op declares an embedding table's integers (``refuse_declared``); its bool
    # is the Python entry's.
    _row("embedding_bag", bool=GUARD_FLOATING),
    # Losses (:mod:`lucid.nn.functional._loss_checks` raises these).  The class is the
    # reference's for a bool or integer *input*; it words one beside a floating operand as
    # a dtype mismatch for some of them, which the row does not tell apart.  A loss the
    # reference answers an integer to (``poisson_nll_loss``, the margin and triplet
    # losses) refuses a bool only.
    _row("binary_cross_entropy", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("binary_cross_entropy_with_logits", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("cross_entropy", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("ctc_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("gaussian_nll_loss", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("hinge_embedding_loss", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("huber_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("kl_div", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("l1_loss", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("margin_ranking_loss", bool=GUARD_FLOATING),
    _row("mse_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("multi_margin_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("multilabel_margin_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("multilabel_soft_margin_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("nll_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("poisson_nll_loss", bool=GUARD_FLOATING),
    _row("smooth_l1_loss", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    _row("soft_margin_loss", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("triplet_margin_loss", bool=GUARD_FLOATING),
    _row("triplet_margin_with_distance_loss", bool=GUARD_FLOATING),
    # Resampling.  ``affine_grid`` checks ``theta`` in Python (a ``ValueError``).
    _row("interpolate", bool=KERNEL_FLOATING, ints=KERNEL_FLOATING),
    # The image is the engine's (``refuse_declared``); the grid is a dtype guard of the entry's.
    _row("grid_sample", "grid", bool=GUARD_FLOATING, ints=GUARD_FLOATING),
    _row("affine_grid", bool=INVALID_FLOATING, ints=INVALID_FLOATING),
)


def _table(rows: tuple[Row, ...]) -> Mapping[tuple[str, str], Row]:
    """``rows`` by key, refusing a key declared twice."""
    table: dict[tuple[str, str], Row] = {}
    for row in rows:
        if row.key in table:
            raise ValueError(f"{row.op} ({row.role or 'primary'}) is declared twice")
        table[row.key] = row
    return MappingProxyType(table)


#: ``(op, role)`` -> the row, for every refusal the Python entry points own.
ROWS: Mapping[tuple[str, str], Row] = _table(_DECLARED)

#: The same rows compiled for the call: ``(op, role)`` -> the rule for each integral dtype,
#: indexed by the dtype's engine code, then the rule for a float and for a complex.
_BY_CODE: Mapping[tuple[str, str], tuple[Rule | None, ...]] = MappingProxyType(
    {key: (*row.by_code(), row.on_float, row.on_complex) for key, row in ROWS.items()}
)

#: Which slot of a compiled row a dtype code reads: its own for bool and the integers, then
#: the float slot, then the complex one.
_COMPLEX_CODES = frozenset({_C_engine.C64.value, _C_engine.C128.value})
_SLOT: tuple[int, ...] = tuple(
    code
    if code <= LAST_INTEGRAL_CODE
    else LAST_INTEGRAL_CODE + (2 if code in _COMPLEX_CODES else 1)
    for code in range(max(d.value for d in _ENGINE_TO_DTYPE) + 1)
)

#: op -> ``(error class the reference raises for a bool operand, what to cast to)`` for
#: the ops this layer refuses a bool in -- the ones with no engine schema to do it.  The
#: bool view of :data:`ROWS`, primary operands only.
BOOL_REFUSED: Mapping[str, Rule] = MappingProxyType(
    {row.op: row.on_bool for row in ROWS.values() if row.role == "" and row.on_bool is not None}
)

_KIND_CLASS: Mapping[str, type[RuntimeError]] = MappingProxyType(
    {"kernel": _KERNEL, "guard": _GUARD}
)


@functools.cache
def engine_refusals() -> Mapping[str, tuple[Rule | None, ...]]:
    """The dtypes each engine schema refuses, in the shape :data:`ROWS` compiles to.

    The engine owns these refusals: ``OpSchema`` declares one and ``SchemaGuard``
    raises it for every caller, so no Python entry repeats it.  This reads the
    declarations back out of the registry, for the tests that hold the two layers to
    one rule.  An op the engine refuses without registering a schema (``var``,
    ``frac``) is not in the registry and so not in this mapping.

    Returns
    -------
    Mapping[str, tuple[Rule | None, ...]]
        ``op -> rules``: the rule for each integral dtype, indexed by its engine code
        (``None`` where the engine answers).  A rule's class is the engine's
        ``NotImplementedError`` for a ``"kernel"`` refusal, ``DtypeMismatch`` for a
        ``"guard"`` one.
    """
    refusals: dict[str, tuple[Rule | None, ...]] = {}
    for schema in _C_engine.op_registry_all():
        slots: list[Rule | None] = [None] * (LAST_INTEGRAL_CODE + 1)
        for dtype in _INTEGRAL:
            declared = schema.refusal_for(dtype)
            if declared is not None:
                kind, hint = declared
                slots[dtype.value] = Rule(_KIND_CLASS[kind], hint)
        if any(slot is not None for slot in slots):
            refusals[schema.name] = tuple(slots)
    return MappingProxyType(refusals)


@functools.cache
def engine_bool_refused() -> Mapping[str, Rule]:
    """The ops whose engine schema refuses a bool tensor, in :data:`BOOL_REFUSED`'s shape.

    Returns
    -------
    Mapping[str, Rule]
        ``op -> (error class, cast hint)``, the bool view of :func:`engine_refusals`.
    """
    refused: dict[str, Rule] = {}
    for op, rules in engine_refusals().items():
        rule = rules[BOOL_CODE]
        if rule is not None:
            refused[op] = rule
    return MappingProxyType(refused)


def _rule(op: str, role: str, code: int) -> Rule | None:
    """The rule ``op``'s ``role`` row holds for the integral dtype ``code``.

    A role with no row of its own reads the op's primary one, and an in-place twin
    (``pow_``) the row of its out-of-place op.
    """
    base = op.removesuffix("_")
    rules = _BY_CODE.get((base, role)) or _BY_CODE.get((base, ""))
    if rules is None:
        raise KeyError(f"{op} has no refusal row; add one to lucid._ops._bool_refusal")
    return rules[_SLOT[code]]


def _error(
    op: str, code: int, role: str, rule: Rule, *, index: bool, mismatch: bool
) -> RuntimeError:
    """The exception ``rule`` names, worded for ``op``'s operand of dtype ``code``."""
    error, cast = rule
    if mismatch:
        error = _GUARD
    if index:
        cast = _INTEGER
    what = _NAMES[code]
    if role:
        what = f"{_ARTICLES[code]} {_ROLE_PHRASE.get(role, role)}"
    return error(f"{op} does not accept {what} — cast to {cast}")


def _dtype_code(operand: object) -> int:
    """``operand``'s engine dtype code, or ``-1`` for anything that is not a tensor."""
    impl = _promotion.try_unwrap(operand)
    return -1 if impl is None else impl.dtype.value


def _check(op: str, code: int, role: str = "") -> None:
    """Raise ``op``'s refusal of the dtype ``code`` if its ``role`` row has one."""
    rule = _rule(op, role, code)
    if rule is not None:
        raise _error(op, code, role, rule, index=False, mismatch=False)


def refusal(
    op: str, code: int, role: str = "", *, index: bool = False, mismatch: bool = False
) -> RuntimeError:
    """The error ``op`` raises for an operand of the integral dtype ``code``, to ``raise``.

    Parameters
    ----------
    op : str
        The name the caller used.  A trailing ``_`` (the in-place twin) reads the row
        of the out-of-place op, and is kept in the message.
    code : int
        The operand's engine dtype code (``impl.dtype.value``): bool or a signed
        integer.
    role : str, optional
        Names the operand when it is not the first one (``"bins"``, ``"weight"``).
    index : bool, optional
        The operand is an index -- a class label -- so the cast to name is an
        integer one, whatever dtype the op computes in.
    mismatch : bool, optional
        The operand sits beside a floating one the op computes in (a loss's
        target): the reference refuses that as a dtype mismatch, whatever
        kernel class the op's row names, so the guard class is raised.

    Returns
    -------
    RuntimeError
        The reference's class for ``op`` (the engine's ``NotImplementedError``,
        ``DtypeMismatch`` or ``InvalidArgument``, all ``RuntimeError`` subclasses), with
        a message that names the op, the dtype and the cast to make.

    Raises
    ------
    KeyError
        ``op`` has no row, or its row does not refuse the dtype: a caller asking for
        a refusal the table does not declare.
    """
    rule = _rule(op, role, code)
    if rule is None:
        raise KeyError(f"{op} does not refuse {_NAMES[code]}; no refusal to raise")
    return _error(op, code, role, rule, index=index, mismatch=mismatch)


def bool_refusal(
    op: str, role: str = "", *, index: bool = False, mismatch: bool = False
) -> RuntimeError:
    """The error ``op`` raises for a bool operand, for a call site to ``raise``.

    Parameters
    ----------
    op : str
        The name the caller used (see :func:`refusal`).
    role : str, optional
        Names the operand when it is not the first one.
    index : bool, optional
        The operand is an index, so the cast to name is an integer one.
    mismatch : bool, optional
        The operand sits beside a floating one: the guard class is raised.

    Returns
    -------
    RuntimeError
        The reference's class for ``op`` and a message naming the op and the cast.
    """
    return refusal(op, BOOL_CODE, role, index=index, mismatch=mismatch)


def is_bool(x: object) -> bool:
    """Whether ``x`` is a bool tensor, as a ``Tensor`` or as the engine's impl.

    Parameters
    ----------
    x : object
        Any operand.  A Python ``bool`` is a number, not a bool tensor, and so
        are ``None`` and a list.

    Returns
    -------
    bool
        ``True`` only for a tensor whose dtype is bool.
    """
    impl = _promotion.try_unwrap(x)
    return impl is not None and impl.dtype.value == BOOL_CODE


def refuse(op: str, operand: object, role: str = "") -> None:
    """Raise ``op``'s refusal when ``operand`` is a bool or integer tensor its row turns away.

    A float or complex tensor, a number, ``None`` and a non-tensor pass at once: the
    only dtype read is the tensor's, and no row is looked up for any of them.

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS`.
    operand : object
        The operand to look at.
    role : str, optional
        The operand's role when it is not the first one, and the row that holds it.

    Raises
    ------
    RuntimeError
        The class :func:`refusal` builds for ``op``.
    """
    impl = _promotion.try_unwrap(operand)
    if impl is None:
        return
    code = impl.dtype.value
    if code <= LAST_INTEGRAL_CODE:
        _check(op, code, role)


def refuse_mixed(op: str, first: object, second: object) -> None:
    """Raise ``op``'s refusal for a bool or integer operand of a pair, by how the pair meets.

    For the ops whose two operands the reference refuses to promote (``prelu``'s input and
    weight): a dtype the primary row turns away is refused as that row says when the
    operands share a dtype, and as the ``"mixed"`` row says -- a dtype guard -- when they
    do not.  A pair of floats, or an operand that is not a tensor, passes.

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS` that also has a
        ``"mixed"`` row.
    first, second : object
        The two operands.

    Raises
    ------
    RuntimeError
        The class :func:`refusal` builds for ``op``.
    """
    impl_first, impl_second = _promotion.try_unwrap(first), _promotion.try_unwrap(second)
    if impl_first is None or impl_second is None:
        return
    codes = (impl_first.dtype.value, impl_second.dtype.value)
    if codes[0] > LAST_INTEGRAL_CODE and codes[1] > LAST_INTEGRAL_CODE:
        return
    role = "" if codes[0] == codes[1] else "mixed"
    for code in codes:
        if code <= LAST_INTEGRAL_CODE:
            _check(op, code, role)


def refuse_declared(op: str, operand: object) -> None:
    """Raise the refusal the engine's schema for ``op`` declares for ``operand``'s dtype.

    For an entry that reaches the engine op on some of its paths only (a weighted or
    gradient-scaled one is composed from other ops), and so would let through what the
    schema turns away.  The engine owns the rule: this reads it back
    (:func:`engine_refusals`) instead of declaring it a second time.  A float or complex
    tensor, a non-tensor and an op with no schema pass.

    Parameters
    ----------
    op : str
        The op whose engine schema to ask.
    operand : object
        The operand to look at.

    Raises
    ------
    RuntimeError
        The class the schema names, worded as the engine words it.
    """
    impl = _promotion.try_unwrap(operand)
    if impl is None:
        return
    code = impl.dtype.value
    if code > LAST_INTEGRAL_CODE:
        return
    rules = engine_refusals().get(op)
    rule = None if rules is None else rules[code]
    if rule is not None:
        raise _error(op, code, "", rule, index=False, mismatch=False)


def refuse_complex(op: str, operand: object, role: str = "") -> None:
    """Raise ``op``'s refusal when ``operand`` is a complex tensor its row turns away.

    For an op whose kernel is real-valued and which would otherwise cast the operand to
    float32 and drop its imaginary part (``histogram``).

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS`.
    operand : object
        The operand to look at.  A real tensor and a non-tensor pass.
    role : str, optional
        The operand's role, and the row that holds it.

    Raises
    ------
    RuntimeError
        The class :func:`refusal` builds for ``op``.
    """
    code = _dtype_code(operand)
    if code in _COMPLEX_CODES:
        _check(op, code, role)


def refuse_unless_bool(op: str, operand: object, role: str = "") -> None:
    """Raise ``op``'s refusal when ``operand`` is a tensor of any dtype but bool.

    For an operand that is a truth value and nothing else (a mask): the reference refuses
    an integer and a float one alike, and answers a bool.  The row's ``on_ints`` rule is
    the integer's, its ``on_float`` and ``on_complex`` theirs.

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS`.
    operand : object
        The operand to look at.  A non-tensor passes.
    role : str, optional
        The operand's role, and the row that holds it.

    Raises
    ------
    RuntimeError
        The class :func:`refusal` builds for ``op``.
    """
    code = _dtype_code(operand)
    if code not in (-1, BOOL_CODE):
        _check(op, code, role)


def refuse_bool(op: str, *operands: object, role: str = "") -> None:
    """Raise ``op``'s refusal when any of ``operands`` is a bool tensor.

    For the entries whose op answers an integer (``cov``, ``isin``, ``clamp``): only a
    bool is looked at, so an integer operand costs nothing here.  It stays beside
    :func:`refuse` for that reason -- the fast path for the ops the reference answers for
    integers, which would otherwise look their row up for every integer they are given.

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS`.
    *operands : object
        The operands to look at.  Numbers, ``None`` and non-tensors pass: only
        a tensor's dtype can be bool.
    role : str, optional
        Names the operand in the message when it is not the first one.

    Raises
    ------
    RuntimeError
        The class :func:`bool_refusal` builds for ``op``.
    """
    for operand in operands:
        impl = _promotion.try_unwrap(operand)
        if impl is not None and impl.dtype.value == BOOL_CODE:
            raise bool_refusal(op, role)


def refuse_common(op: str, *operands: object) -> None:
    """Raise ``op``'s refusal when the operands meet at a bool or integer dtype it turns away.

    For the ops whose operands meet at a common dtype (``searchsorted``, ``gammainc``):
    the reference has no kernel for an integral pair and answers an integer beside a
    float by promoting, so a float (or a non-tensor) among the operands passes, and
    otherwise the widest operand's dtype -- the one they promote to -- is the one
    judged.

    Parameters
    ----------
    op : str
        The refusing op's name, the ``op`` of a row of :data:`ROWS`.
    *operands : object
        The operands that meet at one dtype.

    Raises
    ------
    RuntimeError
        The class :func:`refusal` builds for ``op``.
    """
    common = -1
    for operand in operands:
        impl = _promotion.try_unwrap(operand)
        if impl is None:
            return
        code = impl.dtype.value
        if code > LAST_INTEGRAL_CODE:
            return
        common = max(common, code)
    if common >= 0:
        _check(op, common)


__all__ = [
    "BOOL_CODE",
    "BOOL_REFUSED",
    "LAST_INTEGRAL_CODE",
    "ROWS",
    "Row",
    "Rule",
    "bool_refusal",
    "engine_bool_refused",
    "engine_refusals",
    "is_bool",
    "refusal",
    "refuse",
    "refuse_bool",
    "refuse_common",
    "refuse_complex",
    "refuse_declared",
    "refuse_mixed",
    "refuse_unless_bool",
]
