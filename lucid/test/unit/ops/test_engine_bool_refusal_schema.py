"""An op's schema refuses a bool where the reference does (LCD-1020-E1).

``OpSchema`` carries ``bool_refusal`` (``BoolRefusal`` + ``BoolCast``) and the
``SchemaGuard`` constructor, the first thing every engine entry builds, turns a
bool input away with the class the schema names: ``kernel`` raises
``NotImplementedError`` (the reference has no kernel for bool), ``guard`` raises
``DtypeMismatch`` (a check on the input dtype).  The Python wrappers
(``BOOL_REFUSED``) refuse earlier with the same classes, so a call through
``lucid.abs`` never reaches the engine's check; this file calls the engine
directly, as the entries that bypass the wrappers do.

Three things are held:

* the declared set is exactly :data:`_DECLARED` -- an op that starts or stops
  refusing without this file changing is a mistake, and an op the reference
  refuses cannot be forgotten silently;
* each declared op, called on a bool on CPU and Metal, raises its declared class
  with a message that names the op and the cast;
* the casts the message offers are answered (``mean(mask.float())``), and so is
  everything else in the schema table that was not declared.
"""

from collections.abc import Callable
from typing import Final

import pytest

import lucid
from lucid._C import engine as _C_engine

_NUMBER: Final = "an integer or floating dtype, e.g. x.long() or x.float()"
_FLOATING: Final = "a floating dtype, e.g. x.float()"

_KERNEL: Final = "kernel"
_GUARD: Final = "guard"

_CLASS: Final = {_KERNEL: _C_engine.NotImplementedError, _GUARD: _C_engine.DtypeMismatch}

#: ``op -> (kind, cast hint)`` for every registered schema that refuses bool.
_DECLARED: Final[dict[str, tuple[str, str]]] = {
    # Real |x| and the step functions: no bool kernel, every integer answers.
    "abs": (_KERNEL, _NUMBER),
    "round": (_KERNEL, _NUMBER),
    "floor": (_KERNEL, _NUMBER),
    "ceil": (_KERNEL, _NUMBER),
    "trunc": (_KERNEL, _NUMBER),
    # relu and relu6 are a dtype guard in the reference, and answer integers.
    "relu": (_GUARD, _NUMBER),
    "relu6": (_GUARD, _NUMBER),
    # The rest of the activations: no bool kernel, floats only.
    "selu": (_KERNEL, _FLOATING),
    "silu": (_KERNEL, _FLOATING),
    "gelu": (_KERNEL, _FLOATING),
    "gelu_exact": (_KERNEL, _FLOATING),
    "mish": (_KERNEL, _FLOATING),
    "hard_sigmoid": (_KERNEL, _FLOATING),
    "hard_swish": (_KERNEL, _FLOATING),
    "leaky_relu": (_KERNEL, _FLOATING),
    "elu": (_KERNEL, _FLOATING),
    "softmax": (_KERNEL, _FLOATING),
    "log_softmax": (_KERNEL, _FLOATING),
    # A mean is not a count.
    "mean": (_GUARD, _FLOATING),
    # Pools: max_pool answers integers in two and three dimensions only.
    "max_pool1d": (_KERNEL, _FLOATING),
    "max_pool2d": (_KERNEL, _NUMBER),
    "max_pool3d": (_KERNEL, _NUMBER),
    "avg_pool1d": (_KERNEL, _FLOATING),
    "avg_pool2d": (_KERNEL, _FLOATING),
    "avg_pool3d": (_KERNEL, _FLOATING),
    # Normalisation and sampling: floats only.
    "layer_norm": (_KERNEL, _FLOATING),
    "group_norm": (_KERNEL, _FLOATING),
    "batch_norm1d": (_KERNEL, _FLOATING),
    "batch_norm": (_KERNEL, _FLOATING),
    "batch_norm3d": (_KERNEL, _FLOATING),
    "batch_norm_eval": (_KERNEL, _FLOATING),
    "grid_sample": (_KERNEL, _FLOATING),
}

#: Declared ops that no ``LUCID_REGISTER_OP`` publishes: ``var`` builds its own forward and is
#: reached through ``std`` and ``var`` only, so the registry does not list it.
_UNREGISTERED: Final[dict[str, tuple[str, str]]] = {"var": (_GUARD, _FLOATING)}

#: An operand builder: ``make(dtype)(*shape)`` is an engine impl on the device under test.
_Make = Callable[..., object]
#: A call into the engine: ``x`` builds the primary operand in the dtype under test, ``real``
#: builds a float32 one (a weight, a grid) that is never the dtype that is refused.
_Call = Callable[[_Make, _Make], object]

_E: Final = _C_engine

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
    "softmax": lambda x, real: _E.softmax(x(2, 3), -1),
    "log_softmax": lambda x, real: _E.log_softmax(x(2, 3), -1),
    "mean": lambda x, real: _E.mean(x(2, 3), [1], False),
    "var": lambda x, real: _E.var(x(2, 3), [1], False),
    "max_pool1d": lambda x, real: _E.nn.max_pool1d(x(1, 1, 4), 2),
    "max_pool2d": lambda x, real: _E.nn.max_pool2d(x(1, 1, 4, 4), 2, 2),
    "max_pool3d": lambda x, real: _E.nn.max_pool3d(x(1, 1, 4, 4, 4), 2, 2, 2),
    "avg_pool1d": lambda x, real: _E.nn.avg_pool1d(x(1, 1, 4), 2),
    "avg_pool2d": lambda x, real: _E.nn.avg_pool2d(x(1, 1, 4, 4), 2, 2),
    "avg_pool3d": lambda x, real: _E.nn.avg_pool3d(x(1, 1, 4, 4, 4), 2, 2, 2),
    "layer_norm": lambda x, real: _E.nn.layer_norm(x(2, 4), real(4), real(4), 1e-5),
    "group_norm": lambda x, real: _E.nn.group_norm(x(2, 4, 3), real(4), real(4), 2, 1e-5),
    "batch_norm1d": lambda x, real: _E.nn.batch_norm1d(x(2, 3, 4), real(3), real(3), 1e-5),
    "batch_norm": lambda x, real: _E.nn.batch_norm(x(2, 3, 4, 4), real(3), real(3), 1e-5),
    "batch_norm3d": lambda x, real: _E.nn.batch_norm3d(x(2, 3, 2, 2, 2), real(3), real(3), 1e-5),
    "batch_norm_eval": lambda x, real: _E.nn.batch_norm_eval(
        x(2, 3, 4, 4), real(3), real(3), real(3), real(3), 1e-5
    ),
    "grid_sample": lambda x, real: _E.nn.grid_sample(x(1, 1, 4, 4), real(1, 2, 2, 2), 0, 0, False),
}

_ALL: Final = {**_DECLARED, **_UNREGISTERED}
_BY_NAME: Final = sorted(_ALL)


def _builder(dtype: lucid.dtype, device: str) -> _Make:
    """``build(*shape)``: an engine impl of ones in ``dtype`` on ``device``."""

    def build(*shape: int) -> object:
        return lucid.ones(*shape, dtype=dtype, device=device)._impl

    return build


def _refusal(name: str, device: str) -> BaseException:
    """What ``name`` raises from the engine when its primary operand is bool."""
    with pytest.raises(RuntimeError) as caught:
        _CALLS[name](_builder(lucid.bool_, device), _builder(lucid.float32, device))
    return caught.value


# ── the declaration ──────────────────────────────────────────────────────────


def test_every_declared_op_has_a_call() -> None:
    assert set(_CALLS) == set(_ALL)


def test_the_registry_declares_exactly_the_expected_ops() -> None:
    # Read from the registry, not from the call table, so a schema that gains or loses
    # ``refusing_bool`` (or an op that is registered after this file was written) fails here.
    declared = {
        schema.name: schema.bool_refusal
        for schema in _C_engine.op_registry_all()
        if schema.bool_refusal is not None
    }
    assert declared == _DECLARED


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
        assert schema.bool_refusal is None, name


def test_the_kinds_are_the_two_the_reference_has() -> None:
    assert {kind for kind, _ in _ALL.values()} == {_KERNEL, _GUARD}


# ── the refusal ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _BY_NAME)
def test_a_bool_is_refused_with_the_declared_class(name: str, device: str) -> None:
    kind, hint = _ALL[name]
    error = _refusal(name, device)
    # Exact: ``DtypeMismatch`` and ``NotImplementedError`` are both ``RuntimeError``, and the
    # reference's callers catch one without the other.
    assert type(error) is _CLASS[kind], (name, type(error).__mro__)
    message = str(error)
    assert f"{name} does not accept bool" in message, message
    assert f"cast to {hint}" in message, message


@pytest.mark.parametrize("name", _BY_NAME)
def test_the_kernel_refusal_is_not_a_dtype_error_and_the_reverse(name: str, device: str) -> None:
    kind, _ = _ALL[name]
    error = _refusal(name, device)
    other = _CLASS[_GUARD if kind == _KERNEL else _KERNEL]
    assert not isinstance(error, other), (name, type(error))


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
    # 0-d tensor (softmax's scalar path) and an empty one are refused like any other.
    kind, _ = _ALL[name]
    impl = lucid.ones(*shape, dtype=lucid.bool_, device=device)._impl
    with pytest.raises(RuntimeError) as caught:
        _ANY_SHAPE[name](impl)
    assert type(caught.value) is _CLASS[kind]


def test_a_transposed_bool_is_refused(device: str) -> None:
    mask = lucid.ones(3, 4, dtype=lucid.bool_, device=device).T
    assert not mask.is_contiguous()
    with pytest.raises(_C_engine.NotImplementedError):
        _E.abs(mask._impl)
    with pytest.raises(_C_engine.DtypeMismatch):
        _E.relu(mask._impl)


# ── what still answers ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _BY_NAME)
def test_the_cast_the_message_offers_answers(name: str, device: str) -> None:
    _, hint = _ALL[name]
    dtype = lucid.int64 if hint == _NUMBER else lucid.float32
    out = _CALLS[name](_builder(dtype, device), _builder(lucid.float32, device))
    assert out is not None


@pytest.mark.parametrize("name", _BY_NAME)
def test_a_float_input_answers_on_every_declared_op(name: str, device: str) -> None:
    # The refusal reads the dtype and nothing else: the float path is the one every model runs.
    out = _CALLS[name](_builder(lucid.float32, device), _builder(lucid.float32, device))
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
