"""``custom_fwd`` / ``custom_bwd`` / ``is_autocast_available`` (LCD-740).

A custom ``Function`` whose ``forward`` does ``a @ b`` and is applied under
autocast returns a half-precision result, so the gradient that reaches its
``backward`` is half precision while the tensors it saved are float32, and
``grad @ b.T`` failed with ``DtypeMismatch``.  The reference framework gives
``custom_fwd`` / ``custom_bwd`` for exactly this; Lucid had neither.

``custom_fwd`` records the autocast state of its device on the context (and,
with ``cast_inputs``, casts the floating inputs and runs ``forward`` with
autocast off); ``custom_bwd`` runs ``backward`` in the recorded state.
"""

import inspect
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.amp import autocast, custom_bwd, custom_fwd, is_autocast_available
from lucid.autograd import Function
from lucid.autograd.function import FunctionCtx
from lucid.test._fixtures.devices import metal_available

#: (device, autocast dtype) -> what the forward's ``a @ b`` is under it.
CELLS = [("cpu", lucid.bfloat16, lucid.bfloat16)] + (
    [("metal", lucid.float16, lucid.float16), ("metal", lucid.bfloat16, lucid.bfloat16)]
    if metal_available()
    else []
)

State = tuple[bool, lucid.dtype | None]


@pytest.fixture(autouse=True)
def _amp_off_around_each_test() -> object:
    assert not autocast.is_autocast_enabled(), "a previous test left autocast on"
    yield
    assert not autocast.is_autocast_enabled(), "this test left autocast on"


def _matmul(device: str, *, decorated: bool, cast_inputs: lucid.dtype | None = None) -> type:
    """``Function`` computing ``a @ b``, with or without the decorators."""
    fwd: Callable[..., Callable[..., object]] = (
        custom_fwd(device_type=device, cast_inputs=cast_inputs) if decorated else (lambda f: f)
    )
    bwd: Callable[..., Callable[..., object]] = (
        custom_bwd(device_type=device) if decorated else (lambda f: f)
    )

    class MatMul(Function):
        @staticmethod
        @fwd
        def forward(ctx: FunctionCtx, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
            ctx.save_for_backward(a, b)
            return a @ b

        @staticmethod
        @bwd
        def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
            a, b = ctx.saved_tensors
            return grad @ b.mT, a.mT @ grad

    return MatMul


def _operands(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(0)
    a = lucid.randn(3, 4, device=device).requires_grad_(True)
    b = lucid.randn(4, 2, device=device).requires_grad_(True)
    return a, b


# ── the reported repro ───────────────────────────────────────────────────────


@pytest.mark.parametrize("device, dtype, computed", CELLS, ids=lambda v: str(v))
def test_without_the_decorators_backward_meets_mixed_dtypes(
    device: str, dtype: lucid.dtype, computed: lucid.dtype
) -> None:
    a, b = _operands(device)
    with autocast(device, dtype=dtype):
        y = _matmul(device, decorated=False).apply(a, b)
    assert y.dtype == computed
    with pytest.raises(TypeError, match="DtypeMismatch"):
        y.float().sum().backward()


@pytest.mark.parametrize("device, dtype, computed", CELLS, ids=lambda v: str(v))
def test_decorated_forward_is_half_precision_and_gradients_are_float32(
    device: str, dtype: lucid.dtype, computed: lucid.dtype
) -> None:
    a, b = _operands(device)
    with autocast(device, dtype=dtype):
        y = _matmul(device, decorated=True).apply(a, b)
    assert y.dtype == computed
    y.float().sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.dtype == b.grad.dtype == lucid.float32
    # The same product as an ordinary graph: the gradients agree to the
    # half-precision rounding the decorated backward computed them in.
    a2, b2 = _operands(device)
    (a2 @ b2).sum().backward()
    assert a2.grad is not None and b2.grad is not None
    for got, want in ((a.grad, a2.grad), (b.grad, b2.grad)):
        assert bool(lucid.allclose(got, want, rtol=5e-2, atol=5e-2))


@pytest.mark.parametrize("device", ["cpu"] + (["metal"] if metal_available() else []))
def test_outside_autocast_the_decorators_change_nothing(device: str) -> None:
    a, b = _operands(device)
    y = _matmul(device, decorated=True).apply(a, b)
    assert y.dtype == lucid.float32
    y.sum().backward()
    a2, b2 = _operands(device)
    (a2 @ b2).sum().backward()
    assert a.grad is not None and a2.grad is not None
    assert a.grad.tolist() == a2.grad.tolist()


def test_the_decorated_function_matches_the_reference_on_the_cpu(ref: ModuleType) -> None:
    class RefMatMul(ref.autograd.Function):
        @staticmethod
        @ref.amp.custom_fwd(device_type="cpu")
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a @ b

        @staticmethod
        @ref.amp.custom_bwd(device_type="cpu")
        def backward(ctx, grad):
            a, b = ctx.saved_tensors
            return grad @ b.mT, a.mT @ grad

    values = [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 1.0], [-2.0, 0.125, 1.0, 3.0]]
    others = [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.5], [0.75, 1.25]]
    ra, rb = ref.tensor(values, requires_grad=True), ref.tensor(others, requires_grad=True)
    with ref.autocast("cpu", dtype=ref.bfloat16):
        ry = RefMatMul.apply(ra, rb)
    ry.float().sum().backward()

    la = lucid.tensor(values, requires_grad=True)
    lb = lucid.tensor(others, requires_grad=True)
    with autocast("cpu", dtype=lucid.bfloat16):
        ly = _matmul("cpu", decorated=True).apply(la, lb)
    ly.float().sum().backward()

    assert str(ly.dtype).split(".")[-1] == str(ry.dtype).split(".")[-1]
    assert la.grad is not None and lb.grad is not None
    assert str(la.grad.dtype).split(".")[-1] == str(ra.grad.dtype).split(".")[-1]
    for got, want in ((ly.float(), ry.float()), (la.grad, ra.grad), (lb.grad, rb.grad)):
        assert bool(lucid.allclose(got, lucid.tensor(want.tolist()), rtol=2e-2, atol=2e-2))


# ── backward runs in the forward's state ─────────────────────────────────────


def _recording(device: str, log: list[State]) -> type:
    class Probe(Function):
        @staticmethod
        @custom_fwd(device_type=device)
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            log.append((autocast.is_autocast_enabled(device), autocast.get_autocast_dtype(device)))
            return x * 2.0

        @staticmethod
        @custom_bwd(device_type=device)
        def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> lucid.Tensor:
            log.append((autocast.is_autocast_enabled(device), autocast.get_autocast_dtype(device)))
            return grad * 2.0

    return Probe


# (forward scope dtype or None for no scope, scope .backward() is called in)
SCOPES = [
    (lucid.bfloat16, None),
    (lucid.bfloat16, lucid.float16),
    (None, lucid.bfloat16),
    (None, None),
]


@pytest.mark.parametrize("forward, backward", SCOPES, ids=lambda v: str(v))
def test_backward_sees_the_forwards_state_whatever_scope_it_is_called_from(
    device: str, forward: lucid.dtype | None, backward: lucid.dtype | None
) -> None:
    log: list[State] = []
    x = lucid.ones(3, device=device).requires_grad_(True)
    with autocast(device, dtype=forward, enabled=forward is not None):
        y = _recording(device, log).apply(x)
    with autocast(device, dtype=backward, enabled=backward is not None):
        y.sum().backward()
    assert len(log) == 2
    assert log[1] == log[0]


# ── cast_inputs ──────────────────────────────────────────────────────────────


def _echo(device: str, cast_inputs: lucid.dtype | None) -> Callable[..., object]:
    """A decorated ``forward`` returning what it was handed and the state it saw."""

    @custom_fwd(device_type=device, cast_inputs=cast_inputs)
    def forward(ctx: FunctionCtx, *args: object, **kwargs: object) -> tuple[object, ...]:
        return args, kwargs, autocast.is_autocast_enabled(device)

    return forward


def test_cast_inputs_casts_floats_nested_and_runs_forward_with_autocast_off(device: str) -> None:
    half = lucid.float16 if device == "metal" else lucid.bfloat16
    f32 = lucid.ones(2, device=device)
    f64 = lucid.ones(2, dtype=lucid.float64) if device == "cpu" else None
    ints = lucid.arange(3, device=device)
    ctx = FunctionCtx()
    with autocast(device, dtype=half):
        args, kwargs, enabled = _echo(device, half)(
            ctx, f32, [f32, 7], (ints, {"k": f32}), "s", w=f32
        )
    assert enabled is False  # forward ran with autocast off
    assert args[0].dtype == half
    assert [t.dtype if isinstance(t, lucid.Tensor) else t for t in args[1]] == [half, 7]
    assert args[2][0].dtype == ints.dtype  # integers are not floating
    assert args[2][1]["k"].dtype == half
    assert args[3] == "s"
    assert kwargs["w"].dtype == half
    if f64 is not None:
        with autocast(device, dtype=half):
            (cast64,), _, _ = _echo(device, half)(ctx, f64)
        assert cast64.dtype == lucid.float64  # autocast never narrows float64
    assert ctx._amp_forward_state.enabled is False  # so backward runs with it off too


def test_cast_inputs_does_nothing_outside_an_autocast_scope(device: str) -> None:
    half = lucid.float16 if device == "metal" else lucid.bfloat16
    x = lucid.ones(2, device=device)
    ctx = FunctionCtx()
    (got,), _, enabled = _echo(device, half)(ctx, x)
    assert got.dtype == lucid.float32 and enabled is False


def test_cast_inputs_leaves_tensors_of_another_device_alone() -> None:
    # A cpu scope casts cpu tensors only; a metal-resident one is not its business.
    if not metal_available():
        pytest.skip("Metal is not available")
    ctx = FunctionCtx()
    on_metal = lucid.ones(2, device="metal")
    on_cpu = lucid.ones(2)
    with autocast("cpu", dtype=lucid.bfloat16):
        (m, c), _, _ = _echo("cpu", lucid.bfloat16)(ctx, on_metal, on_cpu)
    assert m.dtype == lucid.float32 and c.dtype == lucid.bfloat16


def test_cast_inputs_passes_other_containers_through_whole() -> None:
    from collections import namedtuple

    Pair = namedtuple("Pair", "x y")
    pair = Pair(lucid.ones(2), 3)
    with autocast("cpu", dtype=lucid.bfloat16):
        (got,), _, _ = _echo("cpu", lucid.bfloat16)(FunctionCtx(), pair)
    assert got is pair


# ── decoration and errors ────────────────────────────────────────────────────


def test_both_call_forms_decorate_and_keep_the_function_identity() -> None:
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        """doc"""
        return x

    bare = custom_fwd(forward, device_type="cpu")
    factory = custom_fwd(device_type="cpu")(forward)
    for wrapped in (bare, factory):
        assert wrapped is not forward
        assert wrapped.__name__ == "forward" and wrapped.__doc__ == "doc"
        assert wrapped.__wrapped__ is forward
    assert custom_bwd(forward, device_type="cpu").__wrapped__ is forward


@pytest.mark.parametrize("decorator", [custom_fwd, custom_bwd])
def test_an_unknown_device_type_is_refused_at_decoration(decorator: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match="unknown device_type 'tpu'"):
        decorator(device_type="tpu")


@pytest.mark.parametrize("decorator", [custom_fwd, custom_bwd])
def test_device_type_is_required_and_keyword_only(decorator: Callable[..., object]) -> None:
    with pytest.raises(TypeError):
        decorator()
    with pytest.raises(TypeError):
        decorator(None, "cpu")


def test_backward_without_custom_fwd_says_what_to_decorate() -> None:
    @custom_bwd(device_type="cpu")
    def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> lucid.Tensor:
        return grad

    with pytest.raises(RuntimeError, match="decorate the Function's forward with custom_fwd"):
        backward(FunctionCtx(), lucid.ones(1))


def test_backward_and_forward_must_name_the_same_device() -> None:
    ctx = FunctionCtx()
    custom_fwd(lambda c, x: x, device_type="cpu")(ctx, lucid.ones(1))

    @custom_bwd(device_type="metal")
    def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> lucid.Tensor:
        return grad

    with pytest.raises(RuntimeError, match="same device_type"):
        backward(ctx, lucid.ones(1))


# ── is_autocast_available ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, expected", [("cpu", True), ("metal", True), ("tpu", False), ("", False), ("CPU", False)]
)
def test_is_autocast_available(name: str, expected: bool) -> None:
    assert is_autocast_available(name) is expected


def test_is_autocast_available_agrees_with_the_reference_on_the_cpu(ref: ModuleType) -> None:
    assert is_autocast_available("cpu") is ref.amp.is_autocast_available("cpu") is True


def test_is_autocast_available_names_what_autocast_accepts() -> None:
    for name in ("cpu", "metal"):
        autocast(name)  # does not raise
    with pytest.raises(ValueError):
        autocast("tpu")


# ── signatures ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["custom_fwd", "custom_bwd", "is_autocast_available"])
def test_signatures_match_the_reference(ref: ModuleType, name: str) -> None:
    def shape(fn: Callable[..., object]) -> list[tuple[str, object, object]]:
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert shape(getattr(lucid.amp, name)) == shape(getattr(ref.amp, name))
