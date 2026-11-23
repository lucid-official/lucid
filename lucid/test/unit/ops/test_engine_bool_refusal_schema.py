"""An op's schema refuses the dtypes the reference refuses (LCD-1020-E1, LCD-1026-E1).

``OpSchema`` carries ``refusal`` (``DtypeRefusal``), ``refused`` (a ``DtypeMask``, one bit per
dtype code) and ``cast`` (``CastHint``), and the ``SchemaGuard`` constructor, the first thing every
engine entry builds, turns a refused dtype away with the class the schema names: ``kernel`` raises
``NotImplementedError`` (the reference has no kernel for the dtype), ``guard`` raises
``DtypeMismatch`` (a check on the input dtype).  An op has one class; what varies with the width is
whether it is refused (``avg_pool2d`` answers int64 and refuses the narrower integers).  The engine
is the one owner of these refusals -- ``lucid.abs``, ``Tensor.abs``, ``abs()`` and
``nn.functional`` reach the same check -- so this file calls it directly, and
``test_bool_refusal_table.py`` pins every Python spelling.

What is held:

* the declared set is exactly :data:`_DECLARED` -- an op that starts or stops refusing a dtype
  without this file changing is a mistake, and an op the reference refuses cannot be forgotten
  silently;
* each declared op, called on every dtype it refuses on CPU and Metal, raises its declared class
  with a message that names the op, the dtype and the cast;
* the operands the guard does not see (the ``b`` of ``solve``, the keys of attention, the weight
  of ``embedding_bag``) are held to the same schema through ``check_schema_dtype``;
* the casts the message offers are answered, and so is every width the schema leaves alone
  (``avg_pool`` on int64) and every op in the schema table that was not declared.
"""

from collections.abc import Callable
from typing import Final

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap

_D: Final = _C_engine.Dtype

_NUMBER: Final = "an integer or floating dtype, e.g. x.long() or x.float()"
_FLOATING: Final = "a floating dtype, e.g. x.float()"

_KERNEL: Final = "kernel"
_GUARD: Final = "guard"

_CLASS: Final = {_KERNEL: _C_engine.NotImplementedError, _GUARD: _C_engine.DtypeMismatch}

_BOOL: Final = frozenset({_D.Bool})
_INTS: Final = frozenset({_D.I8, _D.I16, _D.I32, _D.I64})
_NARROW: Final = frozenset({_D.I8, _D.I16, _D.I32})
_COMPLEX: Final = frozenset({_D.C64, _D.C128})

#: How each refused dtype is spelled in the message (``dtype_name``) and how it is built here.
_NAME: Final = {
    _D.Bool: "bool",
    _D.I8: "int8",
    _D.I16: "int16",
    _D.I32: "int32",
    _D.I64: "int64",
    _D.C64: "complex64",
    _D.C128: "complex128",
}
_LUCID: Final = {
    _D.Bool: lucid.bool_,
    _D.I8: lucid.int8,
    _D.I16: lucid.int16,
    _D.I32: lucid.int32,
    _D.I64: lucid.int64,
    _D.C64: lucid.complex64,
    _D.C128: lucid.complex128,
}

_Rule = tuple[str, str, frozenset[_C_engine.Dtype]]

#: ``op -> (kind, cast hint, refused dtypes)`` for every registered schema that refuses a dtype.
_DECLARED: Final[dict[str, _Rule]] = {
    # Real |x| and the step functions: no bool kernel, every integer answers.
    "abs": (_KERNEL, _NUMBER, _BOOL),
    "round": (_KERNEL, _NUMBER, _BOOL),
    "floor": (_KERNEL, _NUMBER, _BOOL),
    "ceil": (_KERNEL, _NUMBER, _BOOL),
    "trunc": (_KERNEL, _NUMBER, _BOOL),
    # relu and relu6 are a dtype guard in the reference, and answer integers.
    "relu": (_GUARD, _NUMBER, _BOOL),
    "relu6": (_GUARD, _NUMBER, _BOOL),
    # The rest of the activations: no bool or integer kernel, floats only.
    "selu": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "silu": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "gelu": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "gelu_exact": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "mish": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "hard_sigmoid": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "hard_swish": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "leaky_relu": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "elu": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "softplus": (_KERNEL, _FLOATING, _INTS),
    "softmax": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "log_softmax": (_KERNEL, _FLOATING, _BOOL | _INTS),
    # A mean is not a count.
    "mean": (_GUARD, _FLOATING, _BOOL | _INTS),
    # Pools: max_pool answers integers in two and three dimensions only; avg_pool has an int64
    # kernel and none for the narrower widths.
    "max_pool1d": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "max_pool2d": (_KERNEL, _NUMBER, _BOOL),
    "max_pool3d": (_KERNEL, _NUMBER, _BOOL),
    "avg_pool1d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    "avg_pool2d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    "avg_pool3d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    # Normalisation and sampling: floats only.
    "layer_norm": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "group_norm": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "rms_norm": (_KERNEL, _FLOATING, _INTS),
    "batch_norm1d": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "batch_norm": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "batch_norm3d": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "batch_norm_eval": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "grid_sample": (_KERNEL, _FLOATING, _BOOL | _INTS),
    "unfold": (_KERNEL, _FLOATING, _INTS),
    "fold": (_KERNEL, _FLOATING, _INTS),
    # The GEMM and convolution kernels (LCD-1025): no bool kernel in the reference.  It answers
    # integers where Lucid does not (the access side, LCD-1059), so the cast offered is the one
    # Lucid answers: a float.  Metal used to leak MLX's own error for these.  conv_transpose is the
    # exception: the reference answers int64 only, so the narrower integers are refused too.
    "linear": (_KERNEL, _FLOATING, _BOOL),
    "bilinear_layer": (_KERNEL, _FLOATING, _BOOL),
    "conv1d": (_KERNEL, _FLOATING, _BOOL),
    "conv2d": (_KERNEL, _FLOATING, _BOOL),
    "conv3d": (_KERNEL, _FLOATING, _BOOL),
    "conv_transpose1d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    "conv_transpose2d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    "conv_transpose3d": (_KERNEL, _FLOATING, _BOOL | _NARROW),
    # Dropout draws a ``1 / (1 - p)`` mask in the input's dtype; the reference refuses a bool or an
    # integer at its dtype guard (LCD-1025), where the mask sampler's own error was a
    # ``NotImplementedError``.
    "dropout": (_GUARD, _FLOATING, _BOOL | _INTS),
    "dropoutnd": (_GUARD, _FLOATING, _BOOL | _INTS),
    "alpha_dropout": (_GUARD, _FLOATING, _BOOL | _INTS),
    # Guards on the dtype of a table / operand, integers only: a bool table is still answered.
    "embedding_bag": (_GUARD, _FLOATING, _INTS),
    "scaled_dot_product_attention": (_GUARD, _FLOATING, _INTS),
    "det": (_GUARD, _FLOATING, _INTS),
    "inv": (_GUARD, _FLOATING, _INTS),
    "solve": (_GUARD, _FLOATING, _INTS),
    "lu_solve": (_GUARD, _FLOATING, _INTS),
    # The reference's nextafter has a kernel for floats only.
    "nextafter": (_KERNEL, _FLOATING, _BOOL | _INTS | _COMPLEX),
}

#: Ops that refuse and that no ``LUCID_REGISTER_OP`` publishes, so the registry does not list
#: them: ``var`` builds its own forward and is reached through ``std`` and ``var`` only, ``frac``
#: is composed and asks ``check_schema_dtype`` of a schema of its own that is never registered.
_UNREGISTERED: Final[dict[str, _Rule]] = {
    "var": (_GUARD, _FLOATING, _BOOL | _INTS),
    "frac": (_KERNEL, _FLOATING, _BOOL | _INTS),
}


class _Builder:
    """``build(*shape)``: an engine impl of ones in ``dtype`` on ``device``.

    ``dtype=`` builds another dtype on the same device (an index, a pivot vector), ``eye`` a
    non-singular matrix, which a solver answers.
    """

    def __init__(self, dtype: lucid.dtype, device: str) -> None:
        self._dtype = dtype
        self._device = device

    def __call__(self, *shape: int, dtype: lucid.dtype | None = None) -> object:
        return lucid.ones(*shape, dtype=dtype or self._dtype, device=self._device)._impl

    def eye(self, n: int) -> object:
        return lucid.eye(n, dtype=self._dtype, device=self._device)._impl

    def pivots(self, n: int) -> object:
        """The pivot vector of the identity factorisation: ``1, 2, ..., n``."""
        return lucid.arange(1, n + 1, dtype=lucid.int32, device=self._device)._impl


#: A call into the engine: ``x`` builds the primary operand in the dtype under test, ``real``
#: builds a float32 one (a weight, a grid) that is never the dtype that is refused.
_Call = Callable[[_Builder, _Builder], object]

_E: Final = _C_engine

#: The GEMM and convolution ops (LCD-1025) with their operands' shapes, input first and bias last
#: (``bilinear_layer`` has a second input before the weight).
_LAYER_OPS: Final[dict[str, tuple[Callable[..., object], tuple[tuple[int, ...], ...]]]] = {
    "linear": (_E.nn.linear, ((2, 3), (4, 3), (4,))),
    "bilinear_layer": (_E.nn.bilinear_layer, ((2, 3), (2, 4), (5, 3, 4), (5,))),
    "conv1d": (_E.nn.conv1d, ((1, 1, 4), (1, 1, 2), (1,))),
    "conv2d": (_E.nn.conv2d, ((1, 1, 4, 4), (1, 1, 2, 2), (1,))),
    "conv3d": (_E.nn.conv3d, ((1, 1, 3, 3, 3), (1, 1, 2, 2, 2), (1,))),
    "conv_transpose1d": (_E.nn.conv_transpose1d, ((1, 1, 4), (1, 1, 2), (1,))),
    "conv_transpose2d": (_E.nn.conv_transpose2d, ((1, 1, 4, 4), (1, 1, 2, 2), (1,))),
    "conv_transpose3d": (_E.nn.conv_transpose3d, ((1, 1, 3, 3, 3), (1, 1, 2, 2, 2), (1,))),
}


def _layer_call(name: str) -> _Call:
    """``name`` called with every operand built in the dtype under test."""
    fn, shapes = _LAYER_OPS[name]
    return lambda x, real: fn(*(x(*shape) for shape in shapes))


def _layer_call_with_one(name: str, position: int) -> _Call:
    """``name`` called with the operand at ``position`` in the dtype under test, the rest float32."""
    fn, shapes = _LAYER_OPS[name]
    return lambda x, real: fn(
        *((x if at == position else real)(*shape) for at, shape in enumerate(shapes))
    )


_CALLS: Final[dict[str, _Call]] = {
    "abs": lambda x, real: _E.abs(x(3)),
    "round": lambda x, real: _E.round(x(3)),
    "floor": lambda x, real: _E.floor(x(3)),
    "ceil": lambda x, real: _E.ceil(x(3)),
    "trunc": lambda x, real: _E.trunc(x(3)),
    "relu": lambda x, real: _E.relu(x(3)),
    "relu6": lambda x, real: _E.relu6(x(3)),
    "selu": lambda x, real: _E.selu(x(3)),
    "silu": lambda x, real: _E.silu(x(3)),
    "gelu": lambda x, real: _E.gelu(x(3)),
    "gelu_exact": lambda x, real: _E.gelu_exact(x(3)),
    "mish": lambda x, real: _E.mish(x(3)),
    "hard_sigmoid": lambda x, real: _E.hard_sigmoid(x(3)),
    "hard_swish": lambda x, real: _E.hard_swish(x(3)),
    "leaky_relu": lambda x, real: _E.leaky_relu(x(3), 0.1),
    "elu": lambda x, real: _E.elu(x(3), 1.0),
    "softplus": lambda x, real: _E.softplus(x(3)),
    "softmax": lambda x, real: _E.softmax(x(2, 3), -1),
    "log_softmax": lambda x, real: _E.log_softmax(x(2, 3), -1),
    "mean": lambda x, real: _E.mean(x(2, 3), [1], False),
    "var": lambda x, real: _E.var(x(2, 3), [1], False),
    "frac": lambda x, real: _E.frac(x(3)),
    "max_pool1d": lambda x, real: _E.nn.max_pool1d(x(1, 1, 4), 2),
    "max_pool2d": lambda x, real: _E.nn.max_pool2d(x(1, 1, 4, 4), 2, 2),
    "max_pool3d": lambda x, real: _E.nn.max_pool3d(x(1, 1, 4, 4, 4), 2, 2, 2),
    "avg_pool1d": lambda x, real: _E.nn.avg_pool1d(x(1, 1, 4), 2),
    "avg_pool2d": lambda x, real: _E.nn.avg_pool2d(x(1, 1, 4, 4), 2, 2),
    "avg_pool3d": lambda x, real: _E.nn.avg_pool3d(x(1, 1, 4, 4, 4), 2, 2, 2),
    "layer_norm": lambda x, real: _E.nn.layer_norm(x(2, 4), real(4), real(4), 1e-5),
    "group_norm": lambda x, real: _E.nn.group_norm(x(2, 4, 3), real(4), real(4), 2, 1e-5),
    "rms_norm": lambda x, real: _E.nn.rms_norm(x(2, 4), real(4), 1e-5),
    "batch_norm1d": lambda x, real: _E.nn.batch_norm1d(x(2, 3, 4), real(3), real(3), 1e-5),
    "batch_norm": lambda x, real: _E.nn.batch_norm(x(2, 3, 4, 4), real(3), real(3), 1e-5),
    "batch_norm3d": lambda x, real: _E.nn.batch_norm3d(x(2, 3, 2, 2, 2), real(3), real(3), 1e-5),
    "batch_norm_eval": lambda x, real: _E.nn.batch_norm_eval(
        x(2, 3, 4, 4), real(3), real(3), real(3), real(3), 1e-5
    ),
    "grid_sample": lambda x, real: _E.nn.grid_sample(x(1, 1, 4, 4), real(1, 2, 2, 2), 0, 0, False),
    "unfold": lambda x, real: _E.nn.unfold(x(1, 1, 4, 4), [2, 2], [1, 1], [0, 0], [1, 1]),
    "fold": lambda x, real: _E.nn.fold(x(1, 4, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]),
    # The layer ops: every operand is built in the dtype under test (``_LAYER_OPS``).
    **{name: _layer_call(name) for name in _LAYER_OPS},
    # A seeded generator: the draw needs one under ``set_deterministic(True)``, which a sweep in the
    # same process can leave on.  A refused dtype is turned away before the draw (and before that
    # check).
    "dropout": lambda x, real: _E.nn.dropout(x(2, 3), 0.5, True, _E.Generator(0)),
    "dropoutnd": lambda x, real: _E.nn.dropoutnd(x(2, 3, 4), 0.5, True, _E.Generator(0)),
    "alpha_dropout": lambda x, real: _E.nn.alpha_dropout(x(2, 3), 0.5, True, _E.Generator(0)),
    # The table is the dtype under test; the indices are integers by contract.
    "embedding_bag": lambda x, real: _E.nn.embedding_bag(
        x(5, 3), real(4, dtype=lucid.int64), real(2, dtype=lucid.int64)
    ),
    "scaled_dot_product_attention": lambda x, real: _E.nn.scaled_dot_product_attention(
        x(1, 2, 3, 4), x(1, 2, 3, 4), x(1, 2, 3, 4), None, 0.5, False
    ),
    "det": lambda x, real: _E.linalg.det(x.eye(2)),
    "inv": lambda x, real: _E.linalg.inv(x.eye(2)),
    "solve": lambda x, real: _E.linalg.solve(x.eye(2), x.eye(2)),
    "lu_solve": lambda x, real: _E.linalg.lu_solve(x.eye(2), x.pivots(2), x.eye(2)),
    "nextafter": lambda x, real: _E.nextafter(x(3), x(3)),
}

#: The solvers with no schema (cholesky, eig, ldl_factor, pinv, qr, svd) take their integer
#: refusal from one helper; an integer is the reference's dtype guard, ``DtypeMismatch``.
_SOLVERS: Final[dict[str, Callable[[object], object]]] = {
    "cholesky": lambda a: _E.linalg.cholesky(a, False),
    "eig": _E.linalg.eig,
    "ldl_factor": _E.linalg.ldl_factor,
    "pinv": _E.linalg.pinv,
    "qr": _E.linalg.qr,
    "svd": lambda a: _E.linalg.svd(a, True),
}

#: The dropout ops with nothing to drop (``training=False``, or ``p == 0``): the input comes back as
#: it is, so a bool or an integer is answered, as the reference answers it.
_PASS_THROUGH: Final[dict[str, Callable[[object], object]]] = {
    "dropout-eval": lambda a: _E.nn.dropout(a, 0.5, False),
    "dropout-p0": lambda a: _E.nn.dropout(a, 0.0, True),
    "dropoutnd-eval": lambda a: _E.nn.dropoutnd(a, 0.5, False),
    "dropoutnd-p0": lambda a: _E.nn.dropoutnd(a, 0.0, True),
    "alpha_dropout-eval": lambda a: _E.nn.alpha_dropout(a, 0.5, False),
    "alpha_dropout-p0": lambda a: _E.nn.alpha_dropout(a, 0.0, True),
}

#: The widths the schema leaves alone, which must still answer: the integer kernels the reference
#: has.  ``avg_pool`` has int64; the ops that refuse only a bool answer every integer width.
_ACCEPTED: Final[dict[str, frozenset[_C_engine.Dtype]]] = {
    "avg_pool1d": frozenset({_D.I64}),
    "avg_pool2d": frozenset({_D.I64}),
    "avg_pool3d": frozenset({_D.I64}),
    **{
        name: _INTS
        for name in (
            "abs",
            "round",
            "floor",
            "ceil",
            "trunc",
            "relu",
            "relu6",
            "max_pool2d",
            "max_pool3d",
        )
    },
}

_ALL: Final = {**_DECLARED, **_UNREGISTERED}
_BY_NAME: Final = sorted(_ALL)

#: Every (op, refused dtype) cell, the unit the refusal tests run on.
_CELLS: Final = [(name, dt) for name in _BY_NAME for dt in sorted(_ALL[name][2], key=int)]
_CELL_IDS: Final = [f"{name}-{_NAME[dt]}" for name, dt in _CELLS]


def _builder(dtype: lucid.dtype, device: str) -> _Builder:
    return _Builder(dtype, device)


def _refusal(name: str, dtype: _C_engine.Dtype, device: str) -> BaseException:
    """What ``name`` raises from the engine when its primary operand is ``dtype``."""
    if device == "metal" and dtype == _D.C128:
        pytest.skip("complex128 cannot be built on Metal")
    with pytest.raises(RuntimeError) as caught:
        _CALLS[name](_builder(_LUCID[dtype], device), _builder(lucid.float32, device))
    return caught.value


# ── the declaration ──────────────────────────────────────────────────────────


def test_every_declared_op_has_a_call() -> None:
    assert set(_CALLS) == set(_ALL)


def test_the_registry_declares_exactly_the_expected_ops() -> None:
    # Read from the registry, not from the call table, so a schema that gains or loses a refusal
    # (or an op that is registered after this file was written) fails here, dtype by dtype.
    declared: dict[str, _Rule] = {}
    for schema in _C_engine.op_registry_all():
        refused = schema.refused_dtypes
        if not refused:
            continue
        rules = {schema.refusal_for(dt) for dt in refused}
        assert len(rules) == 1, (schema.name, rules)  # one class and one hint per op
        rule = rules.pop()
        assert rule is not None, schema.name
        declared[schema.name] = (rule[0], rule[1], frozenset(refused))
    assert declared == _DECLARED


def test_the_dtype_views_agree_with_each_other() -> None:
    # ``bool_refusal`` is the bool column of ``refusal_for``; ``refused_dtypes`` lists exactly the
    # dtypes ``refusal_for`` answers; a dtype outside the mask is answered ``None``.
    for schema in _C_engine.op_registry_all():
        refused = schema.refused_dtypes
        assert refused == sorted(refused, key=int), schema.name
        assert schema.bool_refusal == schema.refusal_for(_D.Bool), schema.name
        for dt in _D.__members__.values():
            answered = schema.refusal_for(dt) is not None
            assert answered == (dt in refused), (schema.name, dt)


def test_an_op_that_accepts_bool_reports_no_refusal() -> None:
    # The ops the reference answers for a bool (a sum counts, max and min order, a bool
    # raised to a power is a number) and the ones it promotes (sigmoid, tanh) stay as they are.
    for name in (
        "sum",
        "prod",
        "max",
        "min",
        "pow",
        "maximum",
        "minimum",
        "add",
        "sigmoid",
        "tanh",
    ):
        schema = _C_engine.op_lookup(name)
        assert schema is not None, name
        assert schema.refused_dtypes == [], name
        assert schema.bool_refusal is None, name


def test_the_kinds_are_the_two_the_reference_has() -> None:
    assert {kind for kind, _, _ in _ALL.values()} == {_KERNEL, _GUARD}


def test_a_float_is_never_in_a_declared_mask() -> None:
    # The refusal is of bools, integers and (for nextafter) complex; a float reaches the kernel.
    floats = {_D.F16, _D.BF16, _D.F32, _D.F64}
    assert all(not (dts & floats) for _, _, dts in _ALL.values())


# ── the refusal ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("name", "dtype"), _CELLS, ids=_CELL_IDS)
def test_a_refused_dtype_is_refused_with_the_declared_class(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    kind, hint, _ = _ALL[name]
    error = _refusal(name, dtype, device)
    # Exact: ``DtypeMismatch`` and ``NotImplementedError`` are both ``RuntimeError``, and the
    # reference's callers catch one without the other.
    assert type(error) is _CLASS[kind], (name, type(error).__mro__)
    message = str(error)
    assert f"{name} does not accept {_NAME[dtype]}" in message, message
    assert f"cast to {hint}" in message, message


@pytest.mark.parametrize(("name", "dtype"), _CELLS, ids=_CELL_IDS)
def test_the_kernel_refusal_is_not_a_dtype_error_and_the_reverse(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    kind, _, _ = _ALL[name]
    error = _refusal(name, dtype, device)
    other = _CLASS[_GUARD if kind == _KERNEL else _KERNEL]
    assert not isinstance(error, other), (name, type(error))


@pytest.mark.parametrize(
    ("name", "dtype"),
    [(n, dt) for n, dt in _CELLS if _ALL[n][0] == _GUARD],
    ids=[f"{n}-{_NAME[dt]}" for n, dt in _CELLS if _ALL[n][0] == _GUARD],
)
def test_a_guard_says_what_it_expected(name: str, dtype: _C_engine.Dtype, device: str) -> None:
    # ``DtypeMismatch("a floating dtype", "int32", ...)``: a bool is told "a non-bool dtype"
    # because an op that refuses only a bool takes integers, so the integer guard names a float.
    message = str(_refusal(name, dtype, device))
    expected = "a non-bool dtype" if dtype == _D.Bool else "a floating dtype"
    assert f"expected {expected}, got {_NAME[dtype]}" in message, message


#: Ops that take a tensor of any shape, called on one operand.
_ANY_SHAPE: Final[dict[str, Callable[[object], object]]] = {
    "abs": _E.abs,
    "relu": _E.relu,
    "silu": _E.silu,
    "mean": lambda a: _E.mean(a, [], False),
    "softmax": lambda a: _E.softmax(a, -1),
}


@pytest.mark.parametrize("name", sorted(_ANY_SHAPE))
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0), (1, 1)], ids=["0d", "empty", "empty2d", "unit"]
)
def test_the_refusal_does_not_depend_on_the_shape(
    name: str, shape: tuple[int, ...], device: str
) -> None:
    # The check is the first thing the constructor does: it never looks at the data, so a
    # 0-d tensor (softmax's scalar path) and an empty one are refused like any other.  An integer
    # is refused where the op refuses one (abs and relu answer integers).
    kind, _, dtypes = _ALL[name]
    for dtype in sorted(dtypes, key=int):
        impl = lucid.ones(*shape, dtype=_LUCID[dtype], device=device)._impl
        with pytest.raises(RuntimeError) as caught:
            _ANY_SHAPE[name](impl)
        assert type(caught.value) is _CLASS[kind], (name, dtype)


def test_a_transposed_bool_is_refused(device: str) -> None:
    mask = lucid.ones(3, 4, dtype=lucid.bool_, device=device).T
    assert not mask.is_contiguous()
    with pytest.raises(_C_engine.NotImplementedError):
        _E.abs(mask._impl)
    with pytest.raises(_C_engine.DtypeMismatch):
        _E.relu(mask._impl)


def test_a_transposed_integer_is_refused_like_a_contiguous_one(device: str) -> None:
    ids = lucid.ones(3, 4, dtype=lucid.int32, device=device).T
    assert not ids.is_contiguous()
    with pytest.raises(_C_engine.NotImplementedError, match="silu does not accept int32"):
        _E.silu(ids._impl)
    with pytest.raises(_C_engine.DtypeMismatch, match="mean does not accept int32"):
        _E.mean(ids._impl, [], False)


def _layer_positions() -> list[tuple[str, int]]:
    return [
        (name, at) for name, (_, shapes) in sorted(_LAYER_OPS.items()) for at in range(len(shapes))
    ]


@pytest.mark.parametrize(("name", "position"), _layer_positions())
def test_a_bool_operand_in_any_position_of_a_layer_op_is_refused(
    name: str, position: int, device: str
) -> None:
    # Every operand of a layer op goes through the schema's guard (``promote_for_schema``), so a
    # bool weight or bias is refused as a bool input is, before the operands' dtypes are compared.
    # The reference reports a dtype mismatch for a bool beside a float (a plain ``RuntimeError``);
    # the same departure ``grid_sample`` has for a bool grid.  The cast the message offers repairs it.
    call = _layer_call_with_one(name, position)
    with pytest.raises(RuntimeError) as caught:
        call(_builder(lucid.bool_, device), _builder(lucid.float32, device))
    assert type(caught.value) is _C_engine.NotImplementedError, (name, type(caught.value).__mro__)
    assert f"{name} does not accept bool" in str(caught.value)


def _transpose_positions() -> list[tuple[str, int, _C_engine.Dtype]]:
    return [
        (name, at, dt)
        for name in ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d")
        for at in range(len(_LAYER_OPS[name][1]))
        for dt in sorted(_NARROW, key=int)
    ]


@pytest.mark.parametrize(
    ("name", "position", "dtype"),
    _transpose_positions(),
    ids=[f"{n}-{at}-{_NAME[dt]}" for n, at, dt in _transpose_positions()],
)
def test_a_narrow_integer_operand_of_conv_transpose_is_refused_in_any_position(
    name: str, position: int, dtype: _C_engine.Dtype, device: str
) -> None:
    # The reference answers conv_transpose for int64 only, so Lucid's schema refuses the narrower
    # widths on both devices; on Metal the same call used to leak MLX's ``InvalidArgument``.
    call = _layer_call_with_one(name, position)
    with pytest.raises(_C_engine.NotImplementedError) as caught:
        call(_builder(_LUCID[dtype], device), _builder(lucid.float32, device))
    assert f"{name} does not accept {_NAME[dtype]}" in str(caught.value)


def _bool(device: str, *shape: int) -> object:
    return lucid.ones(*shape, dtype=lucid.bool_, device=device)._impl


def test_the_layer_refusal_does_not_depend_on_the_layout(device: str) -> None:
    # The guard reads the dtype and nothing else: an empty batch, a 1-d input and a transposed
    # (non-contiguous) one are refused like any other.  The reference answers the empty batch with an
    # empty result; the refusal here is the one that was there before, now with the declared class.
    weight, bias = _bool(device, 4, 3), _bool(device, 4)
    transposed = lucid.ones(3, 2, dtype=lucid.bool_, device=device).T
    assert not transposed.is_contiguous()
    for x in (_bool(device, 0, 3), _bool(device, 3), transposed._impl):
        with pytest.raises(_C_engine.NotImplementedError, match="linear does not accept bool"):
            _E.nn.linear(x, weight, bias)


@pytest.mark.parametrize("dtype", [_D.Bool, *sorted(_INTS, key=int)], ids=lambda d: _NAME[d])
def test_the_dropout_refusal_does_not_depend_on_the_layout(
    dtype: _C_engine.Dtype, device: str
) -> None:
    # The same for the draw: an empty tensor and a transposed one are refused, with the guard class.
    transposed = lucid.ones(4, 2, dtype=_LUCID[dtype], device=device).T
    assert not transposed.is_contiguous()
    empty = lucid.ones(0, 4, dtype=_LUCID[dtype], device=device)
    for x in (empty._impl, transposed._impl):
        with pytest.raises(
            _C_engine.DtypeMismatch, match=f"dropout does not accept {_NAME[dtype]}"
        ):
            _E.nn.dropout(x, 0.5, True, _E.Generator(0))


@pytest.mark.parametrize("name", sorted(_PASS_THROUGH))
@pytest.mark.parametrize("dtype", [_D.Bool, *sorted(_INTS, key=int)], ids=lambda d: _NAME[d])
def test_a_dropout_with_nothing_to_drop_answers_a_refused_dtype(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    # The refusal is the draw's: ``training=False`` and ``p == 0`` return the input untouched
    # (the reference does too), and a guard built at the entry would refuse them as well.  The
    # conditional path calls ``check_schema_dtype`` only where it draws.
    mask = lucid.tensor([[1, 0, 1], [0, 1, 1]], dtype=_LUCID[dtype], device=device)
    out = _wrap(_PASS_THROUGH[name](mask._impl))
    assert out.dtype == _LUCID[dtype]
    assert out.tolist() == mask.tolist()


# ── the operands the guard does not see ──────────────────────────────────────


def _second_operand_calls() -> dict[str, Callable[[_Builder, _Builder], object]]:
    # ``x`` builds the integer under test, ``real`` a float32: the primary operand is the float.
    return {
        "solve": lambda x, real: _E.linalg.solve(real.eye(2), x.eye(2)),
        "lu_solve": lambda x, real: _E.linalg.lu_solve(real.eye(2), real.pivots(2), x.eye(2)),
        "scaled_dot_product_attention-key": lambda x, real: _E.nn.scaled_dot_product_attention(
            real(1, 2, 3, 4), x(1, 2, 3, 4), real(1, 2, 3, 4), None, 0.5, False
        ),
        "scaled_dot_product_attention-value": lambda x, real: _E.nn.scaled_dot_product_attention(
            real(1, 2, 3, 4), real(1, 2, 3, 4), x(1, 2, 3, 4), None, 0.5, False
        ),
        "lu_solve-factor": lambda x, real: _E.linalg.lu_solve(
            x.eye(2), real.pivots(2), real.eye(2)
        ),
    }


@pytest.mark.parametrize("case", sorted(_second_operand_calls()))
@pytest.mark.parametrize("dtype", sorted(_INTS, key=int), ids=lambda d: _NAME[d])
def test_every_operand_is_held_to_the_schema(
    case: str, dtype: _C_engine.Dtype, device: str
) -> None:
    # The guard sees the first operand; the others call ``check_schema_dtype``, so an integer ``b``
    # beside a float ``a`` is refused as the integer it is (a ``DtypeMismatch`` that names the cast,
    # not the mixed-dtype check that would have followed).
    op = case.split("-")[0]
    with pytest.raises(_C_engine.DtypeMismatch) as caught:
        _second_operand_calls()[case](
            _builder(_LUCID[dtype], device), _builder(lucid.float32, device)
        )
    assert f"{op} does not accept {_NAME[dtype]}" in str(caught.value)


def test_a_pivot_vector_is_an_integer_by_contract(device: str) -> None:
    # lu_solve's pivots are int32 and are not held to the float schema: the factorisation answers.
    real = _builder(lucid.float32, device)
    assert _E.linalg.lu_solve(real.eye(2), real.pivots(2), real.eye(2)) is not None


def test_the_indices_of_an_embedding_bag_are_integers_by_contract(device: str) -> None:
    real = _builder(lucid.float32, device)
    out = _E.nn.embedding_bag(real(5, 3), real(4, dtype=lucid.int64), real(2, dtype=lucid.int64))
    assert out is not None


# ── the schema-less solvers and complex ops ──────────────────────────────────


@pytest.mark.parametrize("name", sorted(_SOLVERS))
@pytest.mark.parametrize("dtype", sorted(_INTS, key=int), ids=lambda d: _NAME[d])
def test_a_solver_with_no_schema_refuses_an_integer_as_a_dtype_guard(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    # One helper (``linalg_detail::require_solver_dtype``): an integer is the reference's dtype
    # guard.  What else the kernels do not take keeps its not-implemented error.
    a = _builder(_LUCID[dtype], device).eye(3)
    with pytest.raises(_C_engine.DtypeMismatch) as caught:
        _SOLVERS[name](a)
    assert f"{name} does not accept {_NAME[dtype]}" in str(caught.value)


@pytest.mark.parametrize("dtype", sorted(_INTS, key=int), ids=lambda d: _NAME[d])
def test_the_complex_constructors_refuse_an_integer_as_a_dtype_guard(
    dtype: _C_engine.Dtype, device: str
) -> None:
    ints = _builder(_LUCID[dtype], device)
    with pytest.raises(_C_engine.DtypeMismatch, match=f"does not accept {_NAME[dtype]}"):
        _E.complex(ints(3), ints(3))
    with pytest.raises(_C_engine.DtypeMismatch, match=f"does not accept {_NAME[dtype]}"):
        _E.view_as_complex(ints(3, 2))


# ── what still answers ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _BY_NAME)
def test_the_cast_the_message_offers_answers(name: str, device: str) -> None:
    _, hint, _ = _ALL[name]
    dtype = lucid.int64 if hint == _NUMBER else lucid.float32
    out = _CALLS[name](_builder(dtype, device), _builder(lucid.float32, device))
    assert out is not None


@pytest.mark.parametrize("name", _BY_NAME)
def test_a_float_input_answers_on_every_declared_op(name: str, device: str) -> None:
    # The refusal reads the dtype and nothing else: the float path is the one every model runs.
    out = _CALLS[name](_builder(lucid.float32, device), _builder(lucid.float32, device))
    assert out is not None


@pytest.mark.parametrize(
    ("name", "dtype"),
    [(name, dt) for name in sorted(_ACCEPTED) for dt in sorted(_ACCEPTED[name], key=int)],
    ids=[
        f"{name}-{_NAME[dt]}"
        for name in sorted(_ACCEPTED)
        for dt in sorted(_ACCEPTED[name], key=int)
    ],
)
def test_a_width_the_schema_leaves_alone_is_answered(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    # The reference has an int64 kernel for avg_pool and a kernel for every integer in the step
    # functions: refusing them would be the over-refusal this table exists to avoid.
    assert not _ALL[name][2] & {dtype}, (name, dtype)
    out = _CALLS[name](_builder(_LUCID[dtype], device), _builder(lucid.float32, device))
    assert out is not None


def test_the_mask_cast_to_float_is_averaged_and_activated(device: str) -> None:
    mask = lucid.tensor([True, False, True, True], device=device)
    assert mask.float().mean().item() == pytest.approx(0.75)
    assert _E.silu(mask.float()._impl).dtype == _C_engine.F32
    assert _E.mean(mask.float()._impl, [], False).dtype == _C_engine.F32


# ── the ops the schema leaves alone ──────────────────────────────────────────


@pytest.mark.parametrize("name", ["sum", "prod", "max", "min"])
def test_the_reductions_the_reference_answers_still_take_a_bool(name: str, device: str) -> None:
    mask = lucid.tensor([[True, False], [True, True]], device=device)._impl
    out = getattr(_C_engine, name)(mask, [], False)
    assert out is not None


@pytest.mark.parametrize("name", ["sum", "prod", "max", "min"])
@pytest.mark.parametrize("dtype", sorted(_INTS, key=int), ids=lambda d: _NAME[d])
def test_the_reductions_the_reference_answers_still_take_an_integer(
    name: str, dtype: _C_engine.Dtype, device: str
) -> None:
    # An integer sum or product is exact and the reference answers it: not part of this batch.
    values = lucid.ones(2, 2, dtype=_LUCID[dtype], device=device)._impl
    out = getattr(_C_engine, name)(values, [], False)
    assert out is not None
