"""An op's schema refuses a bool where the reference does (LCD-1020-E1).

``OpSchema`` carries ``bool_refusal`` (``BoolRefusal`` + ``BoolCast``) and the
``SchemaGuard`` constructor, the first thing every engine entry builds, turns a
bool input away with the class the schema names: ``kernel`` raises
``NotImplementedError`` (the reference has no kernel for bool), ``guard`` raises
``DtypeMismatch`` (a check on the input dtype).  The engine is the one owner of
these refusals -- ``lucid.abs``, ``Tensor.abs``, ``abs()`` and ``nn.functional`` reach
the same check -- so this file calls it directly, and
``test_bool_refusal_table.py`` pins every Python spelling.

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
from lucid._dispatch import _wrap

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
    # The GEMM and convolution kernels (LCD-1025): no bool kernel in the reference.  It answers
    # integers where Lucid does not, so the cast offered is the one Lucid answers: a float.
    # Metal used to leak MLX's own error for these.
    "linear": (_KERNEL, _FLOATING),
    "bilinear_layer": (_KERNEL, _FLOATING),
    "conv1d": (_KERNEL, _FLOATING),
    "conv2d": (_KERNEL, _FLOATING),
    "conv3d": (_KERNEL, _FLOATING),
    "conv_transpose1d": (_KERNEL, _FLOATING),
    "conv_transpose2d": (_KERNEL, _FLOATING),
    "conv_transpose3d": (_KERNEL, _FLOATING),
    # Dropout draws a ``1 / (1 - p)`` mask in the input's dtype; the reference refuses a bool at its
    # dtype guard (LCD-1025), where the mask sampler's own error was a ``NotImplementedError``.
    "dropout": (_GUARD, _FLOATING),
    "dropoutnd": (_GUARD, _FLOATING),
    "alpha_dropout": (_GUARD, _FLOATING),
}

#: Ops that refuse bool and that no ``LUCID_REGISTER_OP`` publishes, so the registry does not list
#: them: ``var`` builds its own forward and is reached through ``std`` and ``var`` only, ``frac``
#: is composed and raises through ``throw_bool_refusal``.
_UNREGISTERED: Final[dict[str, tuple[str, str]]] = {
    "var": (_GUARD, _FLOATING),
    "frac": (_KERNEL, _FLOATING),
}

#: An operand builder: ``make(dtype)(*shape)`` is an engine impl on the device under test.
_Make = Callable[..., object]
#: A call into the engine: ``x`` builds the primary operand in the dtype under test, ``real``
#: builds a float32 one (a weight, a grid) that is never the dtype that is refused.
_Call = Callable[[_Make, _Make], object]

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
    "batch_norm1d": lambda x, real: _E.nn.batch_norm1d(x(2, 3, 4), real(3), real(3), 1e-5),
    "batch_norm": lambda x, real: _E.nn.batch_norm(x(2, 3, 4, 4), real(3), real(3), 1e-5),
    "batch_norm3d": lambda x, real: _E.nn.batch_norm3d(x(2, 3, 2, 2, 2), real(3), real(3), 1e-5),
    "batch_norm_eval": lambda x, real: _E.nn.batch_norm_eval(
        x(2, 3, 4, 4), real(3), real(3), real(3), real(3), 1e-5
    ),
    "grid_sample": lambda x, real: _E.nn.grid_sample(x(1, 1, 4, 4), real(1, 2, 2, 2), 0, 0, False),
    # The layer ops: every operand is built in the dtype under test (``_LAYER_OPS``).
    **{name: _layer_call(name) for name in _LAYER_OPS},
    # A seeded generator: the draw needs one under ``set_deterministic(True)``, which a sweep in the
    # same process can leave on.  A bool is refused before the draw (and before that check).
    "dropout": lambda x, real: _E.nn.dropout(x(2, 3), 0.5, True, _E.Generator(0)),
    "dropoutnd": lambda x, real: _E.nn.dropoutnd(x(2, 3, 4), 0.5, True, _E.Generator(0)),
    "alpha_dropout": lambda x, real: _E.nn.alpha_dropout(x(2, 3), 0.5, True, _E.Generator(0)),
}

#: The dropout ops with nothing to drop (``training=False``, or ``p == 0``): the input comes back as
#: it is, so a bool is answered, as the reference answers it.
_PASS_THROUGH: Final[dict[str, Callable[[object], object]]] = {
    "dropout-eval": lambda a: _E.nn.dropout(a, 0.5, False),
    "dropout-p0": lambda a: _E.nn.dropout(a, 0.0, True),
    "dropoutnd-eval": lambda a: _E.nn.dropoutnd(a, 0.5, False),
    "dropoutnd-p0": lambda a: _E.nn.dropoutnd(a, 0.0, True),
    "alpha_dropout-eval": lambda a: _E.nn.alpha_dropout(a, 0.5, False),
    "alpha_dropout-p0": lambda a: _E.nn.alpha_dropout(a, 0.0, True),
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


def test_the_dropout_refusal_does_not_depend_on_the_layout(device: str) -> None:
    # The same for the draw: an empty tensor and a transposed one are refused, with the guard class.
    transposed = lucid.ones(4, 2, dtype=lucid.bool_, device=device).T
    assert not transposed.is_contiguous()
    for x in (_bool(device, 0, 4), transposed._impl):
        with pytest.raises(_C_engine.DtypeMismatch, match="dropout does not accept bool"):
            _E.nn.dropout(x, 0.5, True, _E.Generator(0))


@pytest.mark.parametrize("name", sorted(_PASS_THROUGH))
def test_a_dropout_with_nothing_to_drop_answers_a_bool(name: str, device: str) -> None:
    # The refusal is the draw's: ``training=False`` and ``p == 0`` return the input untouched
    # (the reference does too), and a guard built at the entry would refuse them as well.
    mask = lucid.tensor([[True, False, True], [False, True, True]], device=device)
    out = _wrap(_PASS_THROUGH[name](mask._impl))
    assert out.dtype == lucid.bool_
    assert out.tolist() == mask.tolist()


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
