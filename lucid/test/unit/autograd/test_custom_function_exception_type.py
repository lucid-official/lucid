"""An exception a custom ``Function``'s ``backward`` raises reaches the caller as itself.

The engine's bridge into a Python ``backward`` caught every Python exception
and rethrew it as a C++ ``std::runtime_error``, so the caller of
``backward()`` got a ``RuntimeError`` whatever ``backward`` had raised:

    ValueError("bad grad")  ->  RuntimeError("PythonBackward raised: ValueError: bad grad")

The type survived only as text in the message, so ``except ValueError``
around ``backward()`` never matched, a user's own exception class could not
be caught by name, and the ``VersionMismatch`` that ``ctx.saved_tensors``
raises for a write into a saved tensor arrived as a plain ``RuntimeError``.
The reference hands the exception through as it was raised, cause and
traceback included.  The engine now does too, on every path that calls into
``backward``: ``Tensor.backward``, ``autograd.grad``, a ``Function`` with
several outputs (the engine's barrier path), and ``create_graph=True``.
"""

import traceback
from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.autograd import Function


class _UserError(Exception):
    """An exception class of the caller's own, unrelated to ``RuntimeError``."""


# What ``backward`` raises in each test, and the message it carries.
RAISES: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "TypeError": TypeError,
    "user": _UserError,
}
MESSAGE = "backward refused: grad has shape (3,)"


def _raising(exc: type[Exception]) -> type[Function]:
    """A ``Function`` doubling its input whose ``backward`` raises ``exc``."""

    class Raise(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            raise exc(MESSAGE)

    return Raise


def _leaf(device: str) -> lucid.Tensor:
    return lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)


# Every way into the engine that runs a single-output ``backward``.  The two
# ``create_graph`` paths call it through ``apply_for_graph``, the others
# through ``apply``.
PATHS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], object]] = {
    "backward": lambda out, x: out.backward(),
    "grad": lambda out, x: lucid.autograd.grad(out, [x]),
    "backward-create_graph": lambda out, x: out.backward(create_graph=True),
    "grad-create_graph": lambda out, x: lucid.autograd.grad(out, [x], create_graph=True),
}


@pytest.mark.parametrize("path", list(PATHS), ids=list(PATHS))
@pytest.mark.parametrize("exc", list(RAISES), ids=list(RAISES))
def test_backward_raises_its_own_type(exc: str, path: str, device: str) -> None:
    x = _leaf(device)
    out = _raising(RAISES[exc]).apply(x).sum()
    with pytest.raises(RAISES[exc]) as info:
        PATHS[path](out, x)
    # Exactly the class raised — not a ``RuntimeError`` carrying its name.
    assert type(info.value) is RAISES[exc]
    assert not isinstance(info.value, RuntimeError)
    assert str(info.value) == MESSAGE


class _TwoOut(Function):
    """Two outputs, so the engine collects both gradients before ``backward``."""

    @staticmethod
    def forward(ctx, x):
        return x * 2.0, x * 3.0

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        raise ValueError(MESSAGE)


@pytest.mark.parametrize("path", ["backward", "grad"])
def test_a_function_with_several_outputs_raises_its_own_type(path: str, device: str) -> None:
    x = _leaf(device)
    a, b = _TwoOut.apply(x)
    out = a.sum() + b.sum()
    with pytest.raises(ValueError) as info:
        PATHS[path](out, x)
    assert type(info.value) is ValueError
    assert str(info.value) == MESSAGE


def test_an_engine_error_inside_backward_keeps_its_type(device: str) -> None:
    # A Lucid op inside ``backward`` that refuses raises the engine's own
    # typed error; it used to come out of ``backward()`` as a RuntimeError
    # whose message began "PythonBackward raised: ShapeMismatch".
    class Mismatch(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            return grad + lucid.ones(5, device=device)

    x = _leaf(device)
    with pytest.raises(_C_engine.ShapeMismatch):
        Mismatch.apply(x).sum().backward()


def test_the_cause_and_the_traceback_come_with_it(device: str) -> None:
    class Chained(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            try:
                {}["missing"]
            except KeyError as e:
                raise ValueError(MESSAGE) from e

    x = _leaf(device)
    with pytest.raises(ValueError) as info:
        Chained.apply(x).sum().backward()
    assert isinstance(info.value.__cause__, KeyError)
    # The traceback runs down into the ``backward`` that raised.
    frames = traceback.extract_tb(info.value.__traceback__)
    assert frames[-1].name == "backward"


def test_backward_after_a_raising_backward_still_runs(device: str) -> None:
    x = _leaf(device)
    with pytest.raises(_UserError):
        _raising(_UserError).apply(x).sum().backward()

    y = _leaf(device)
    (y * 3.0).sum().backward()
    assert y.grad is not None
    assert y.grad.tolist() == [3.0, 3.0, 3.0]


# ── against the reference ────────────────────────────────────────────────────


def _ref_raising(ref, exc: type[Exception]):
    class RefRaise(ref.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, grad):
            raise exc(MESSAGE)

    return RefRaise


def _raised(run: Callable[[], object]) -> BaseException:
    try:
        run()
    except Exception as e:
        return e
    raise AssertionError("backward did not raise")


@pytest.mark.parity
@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create_graph"])
@pytest.mark.parametrize("exc", list(RAISES), ids=list(RAISES))
def test_the_type_is_the_one_the_reference_raises(
    exc: str, create_graph: bool, device: str, ref
) -> None:
    x = _leaf(device)
    out = _raising(RAISES[exc]).apply(x).sum()
    got = _raised(lambda: lucid.autograd.grad(out, [x], create_graph=create_graph))

    rx = ref.tensor([1.0, 2.0, 3.0], requires_grad=True)
    rout = _ref_raising(ref, RAISES[exc]).apply(rx).sum()
    want = _raised(lambda: ref.autograd.grad(rout, [rx], create_graph=create_graph))

    assert type(got) is type(want)
    assert str(got) == str(want)
