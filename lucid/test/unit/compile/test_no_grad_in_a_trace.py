"""What runs under ``no_grad`` in a traced forward is a constant to the backward (LCD-901).

A recording kept the op and not the grad mode it ran in, so a compiled backward
differentiated through every ``no_grad`` block on the way to the loss.
``y + (y.mean(0) * 0.1)`` with the mean under ``no_grad`` gave the first
parameter a gradient off by up to 1.03 on every call, with the loss exactly
right and no word of it.  ``lucid.erfc`` and ``nn.utils.spectral_norm``, which
hold a ``no_grad`` of their own, were wrong the same way.  ``y.detach()`` was
right all along: only a block failed.

Each case runs three calls with different inputs against a twin running eagerly
and compares the loss, the gradient (the parameters, for ``fused_step``) and the
input.  Where the recording can stand for the eager call, a case also asserts that
the entry stayed compiled: refusing every ``no_grad`` would pass the comparison.

A value the recording gives one id cannot both change and carry a gradient, so a
write autograd does not track (under ``no_grad``, through ``.data``, ``.detach()``)
to a tensor a gradient flows through refuses and runs eagerly.
"""

from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.autograd._grad_mode import _grad_mode_reads_refused
from lucid.autograd.checkpoint import checkpoint
from lucid.autograd.function import Function, FunctionCtx
from lucid.compile._core import fallback
from lucid.compile._core.trace_context import recording
from lucid.compile._entry.function import compiled_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile.test_write_on_a_tensor_the_call_made import (
    WIDTH,
    _loss,
    _metal_ok,
    _run,
    _sequential,
)

pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning"),
]

Tensor = lucid.Tensor

#: A segmented step runs eagerly when a segment reads no gradient of its input,
#: which a block that reads its input does; ``compiled_step`` has no eager path
#: (see the refusals below).  The rest stay compiled.
_STAY_COMPILED = ["make_step", "fused_step", "module_no_grad"]
_COMPARED = ["make_step", "segmented", "compiled_step", "fused_step", "module_no_grad"]


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


class _Holds(nn.Module):
    """A layer with a parameter and a buffer that ``body`` may read or write."""

    def __init__(self, body: Body) -> None:
        super().__init__()
        self.body = body
        self.p = nn.Parameter(lucid.ones(WIDTH) * 0.5)
        self.register_buffer("b", lucid.zeros(WIDTH))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(self, x)


Body = Callable[[_Holds, Tensor], Tensor]


def _under_no_grad(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = y.mean(0) * 0.1
    return y + c


def _under_flag(s: _Holds, y: Tensor) -> Tensor:
    before = lucid.is_grad_enabled()
    lucid.set_grad_enabled(False)
    c = y.mean(0) * 0.1
    lucid.set_grad_enabled(before)
    return y + c


def _under_inference(s: _Holds, y: Tensor) -> Tensor:
    with lucid.inference_mode():
        c = y.mean(0) * 0.1
    return y + c


def _nested_off(s: _Holds, y: Tensor) -> Tensor:
    with lucid.enable_grad(), lucid.no_grad():
        c = y.mean(0) * 0.1
    return y + c


def _nested_back_on(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad(), lucid.enable_grad():
        c = y.mean(0) * 0.1
    return y + c


def _view_under_no_grad(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = y[:, :1]
    return y + c


def _reshape_under_no_grad(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = y.reshape(-1, WIDTH)
    return y * c


def _clone_under_no_grad(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = y.clone()
    return y * c


def _detached(s: _Holds, y: Tensor) -> Tensor:
    return y + (y.mean(0) * 0.1).detach()


def _detached_first(s: _Holds, y: Tensor) -> Tensor:
    return y + y.detach().mean(0) * 0.1


def _data(s: _Holds, y: Tensor) -> Tensor:
    return y + y.data.mean(0) * 0.1


def _parameter_too(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = s.p * 2.0
    return y * c + s.p


def _buffer_copy(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        s.b.copy_(y.mean(0))
    return y + s.b


def _buffer_average(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        s.b.mul_(0.9).add_(y.mean(0) * 0.1)
    return y + s.b


def _buffer_slice(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        s.b[:2] = y.mean(0)[:2]
    return y + s.b


def _buffer_copy_of_a_value_from_before(s: _Holds, y: Tensor) -> Tensor:
    first = y[0]
    with lucid.no_grad():
        s.b.copy_(first)
    return y + s.b


def _reentrant_inside(s: _Holds, y: Tensor) -> Tensor:
    return checkpoint(_under_no_grad_of, y, use_reentrant=True)


def _nonreentrant_inside(s: _Holds, y: Tensor) -> Tensor:
    return checkpoint(_under_no_grad_of, y, use_reentrant=False)


def _under_no_grad_of(a: Tensor) -> Tensor:
    with lucid.no_grad():
        c = a.mean(0) * 0.1
    return a + c


def _reentrant_around(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = checkpoint(_mean_of, y, use_reentrant=True)
    return y + c


def _nonreentrant_around(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = checkpoint(_mean_of, y, use_reentrant=False)
    return y + c


def _mean_of(a: Tensor) -> Tensor:
    return a.mean(0) * 0.1


class _NoGradInForward(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        with lucid.no_grad():
            c = a.mean(0) * 0.1
        return a * 2.0 + c

    @staticmethod
    def backward(ctx: FunctionCtx, g: Tensor) -> Tensor:
        return g * 3.0


class _NoGradInBackward(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return a * 2.0

    @staticmethod
    def backward(ctx: FunctionCtx, g: Tensor) -> Tensor:
        (a,) = ctx.saved_tensors
        with lucid.no_grad():
            c = a.mean() * 0.1
        return g * c


def _function_with_it_in_forward(s: _Holds, y: Tensor) -> Tensor:
    return _NoGradInForward.apply(y)


def _function_with_it_in_backward(s: _Holds, y: Tensor) -> Tensor:
    return _NoGradInBackward.apply(y)


def _function_under_it(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = _NoGradInForward.apply(y)
    return y + c


def _function_after_it(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = y.mean(0) * 0.1
    return _NoGradInForward.apply(y + c)


class _Doubled(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx: FunctionCtx, a: Tensor) -> Tensor:
        return a * 2.0

    @staticmethod
    def backward(ctx: FunctionCtx, g: Tensor) -> Tensor:
        return g * 2.0


def _function_before_it(s: _Holds, y: Tensor) -> Tensor:
    # The call records its forward with the block's marking off; the block after
    # it is marked again only if the call puts the marking back.
    z = _Doubled.apply(y * s.p)
    with lucid.no_grad():
        c = z.mean(0) * 0.1
    return z + c


def _erfc(s: _Holds, y: Tensor) -> Tensor:
    return lucid.erfc(y * 0.1)


def _expm1(s: _Holds, y: Tensor) -> Tensor:
    return lucid.expm1(y * 0.1)


def _log1p(s: _Holds, y: Tensor) -> Tensor:
    return lucid.log1p(y.abs() * 0.1)


def _arctan2(s: _Holds, y: Tensor) -> Tensor:
    return lucid.atan2(y, y * 0.5 + 1.0)


#: The recording stands for these: a block is a constant to the backward, and the
#: entries in ``_STAY_COMPILED`` compile them.
CONSTANTS: dict[str, Body] = {
    "no_grad": _under_no_grad,
    "set_grad_enabled": _under_flag,
    "inference_mode": _under_inference,
    "enable_grad_then_no_grad": _nested_off,
    "no_grad_then_enable_grad": _nested_back_on,
    "view": _view_under_no_grad,
    "reshape": _reshape_under_no_grad,
    "clone": _clone_under_no_grad,
    "detach_after": _detached,
    "detach_before": _detached_first,
    "data": _data,
    "parameter_too": _parameter_too,
    "buffer_copy": _buffer_copy,
    "buffer_average": _buffer_average,
    "buffer_slice": _buffer_slice,
    "buffer_copy_of_a_value_from_before": _buffer_copy_of_a_value_from_before,
    "checkpoint_reentrant_inside": _reentrant_inside,
    "checkpoint_nonreentrant_inside": _nonreentrant_inside,
    "checkpoint_reentrant_around": _reentrant_around,
    "checkpoint_nonreentrant_around": _nonreentrant_around,
    "function_in_forward": _function_with_it_in_forward,
    "function_in_backward": _function_with_it_in_backward,
    "function_under_it": _function_under_it,
    "function_after_it": _function_after_it,
    "function_before_it": _function_before_it,
    "erfc": _erfc,
    "expm1": _expm1,
    "log1p": _log1p,
    "arctan2": _arctan2,
}

#: ``compiled_step`` has no way to carry a write on a buffer back, and says so.
_WRITES_A_BUFFER = {
    "buffer_copy",
    "buffer_average",
    "buffer_slice",
    "buffer_copy_of_a_value_from_before",
}


@pytest.mark.parametrize("entry", _COMPARED)
@pytest.mark.parametrize("body", list(CONSTANTS))
def test_a_constant_to_the_backward(body: str, entry: str) -> None:
    if entry == "compiled_step" and body in _WRITES_A_BUFFER:
        pytest.skip("compiled_step refuses a write on a buffer, no_grad or not")
    ran = _run(lambda: _Holds(CONSTANTS[body]), entry)
    if entry in _STAY_COMPILED:
        assert ran.stayed_compiled(), "refusing the block would pass the comparison"


def _body_on_a_parameter(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        c = s.p * 2.0
    return y * c


def _parameter_detached(s: _Holds, y: Tensor) -> Tensor:
    return y * s.p.detach()


#: A parameter reached only through a sink gets ``None`` in eager.
_ONLY_THROUGH_IT: dict[str, Body] = {"no_grad": _body_on_a_parameter, "detach": _parameter_detached}
_ZEROS_IN_A_SEGMENT = pytest.mark.xfail(
    strict=True,
    reason="LCD-899: segments=N hands a parameter off the path zeros, where eager has None",
)


@pytest.mark.parametrize(
    "entry",
    [
        "make_step",
        pytest.param("segmented", marks=_ZEROS_IN_A_SEGMENT),
        "compiled_step",
        "fused_step",
    ],
)
@pytest.mark.parametrize("body", list(_ONLY_THROUGH_IT))
def test_a_parameter_only_a_constant_reads_gets_no_gradient(body: str, entry: str) -> None:
    ran = _run(lambda: _Holds(_ONLY_THROUGH_IT[body]), entry)
    assert ran.stayed_compiled() in (True, None)


def _no_grad_add(s: _Holds, y: Tensor) -> Tensor:
    z = y * 1.0
    with lucid.no_grad():
        z.add_(1.0)
    return z * z


def _no_grad_copy_of_itself(s: _Holds, y: Tensor) -> Tensor:
    z = y * 2.0
    with lucid.no_grad():
        z.copy_(z * 3.0)
    return z * z


def _no_grad_on_parameter(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        s.p.mul_(0.9)
    return y * s.p


def _no_grad_on_parameter_view(s: _Holds, y: Tensor) -> Tensor:
    with lucid.no_grad():
        s.p[:2].mul_(0.9)
    return y * s.p


def _detach_add(s: _Holds, y: Tensor) -> Tensor:
    z = y * 2.0
    z.detach().add_(1.0)
    return z * z


def _data_add(s: _Holds, y: Tensor) -> Tensor:
    z = y * 2.0
    z.data.add_(1.0)
    return z * z


def _data_on_parameter(s: _Holds, y: Tensor) -> Tensor:
    s.p.data.mul_(0.9)
    return y * s.p


def _data_copy_of_a_value_with_a_gradient(s: _Holds, y: Tensor) -> Tensor:
    s.b.data.copy_(y[0])
    return y + s.b


#: Eager keeps the tensor's place in the graph and takes the new values as a
#: constant.  A recorded value has one id, so these refuse.
UNTRACKED_WRITES: dict[str, Body] = {
    "no_grad_add": _no_grad_add,
    "no_grad_copy_of_itself": _no_grad_copy_of_itself,
    "no_grad_on_parameter": _no_grad_on_parameter,
    "no_grad_on_parameter_view": _no_grad_on_parameter_view,
    "detach_add": _detach_add,
    "data_add": _data_add,
    "data_on_parameter": _data_on_parameter,
    "data_copy_of_a_value_with_a_gradient": _data_copy_of_a_value_with_a_gradient,
}


@pytest.mark.parametrize("entry", ["make_step", "segmented", "fused_step"])
@pytest.mark.parametrize("body", list(UNTRACKED_WRITES))
def test_a_write_autograd_does_not_track_runs_eagerly(body: str, entry: str) -> None:
    ran = _run(lambda: _Holds(UNTRACKED_WRITES[body]), entry)
    assert ran.stayed_compiled() is False


@pytest.mark.parametrize("body", ["no_grad_add", "detach_add", "no_grad_on_parameter"])
def test_compiled_step_refuses_a_write_autograd_does_not_track(body: str) -> None:
    model = _sequential(lambda: _Holds(UNTRACKED_WRITES[body]))
    x = lucid.randn(6, WIDTH, device=COMPILE_DEVICE)
    with pytest.raises(RuntimeError, match="does not track"):
        compiled_step(model, x, _loss)


def _spectral_norm() -> nn.Module:
    return nn.utils.spectral_norm(nn.Linear(WIDTH, WIDTH))


def _spectral_parametrization() -> nn.Module:
    return nn.utils.parametrizations.spectral_norm(nn.Linear(WIDTH, WIDTH))


class _Table(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(6, WIDTH, max_norm=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.table(lucid.arange(x.shape[0], device=x.device))


@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
@pytest.mark.parametrize("layer", [_spectral_norm, _spectral_parametrization])
def test_a_layer_holding_a_no_grad_stays_compiled(
    layer: Callable[[], nn.Module], entry: str
) -> None:
    assert _run(layer, entry).stayed_compiled()


@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
def test_a_max_norm_table_writes_its_weight_under_no_grad(entry: str) -> None:
    assert _run(_Table, entry).stayed_compiled() is False


@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
def test_mpsgraph_autodiff_does_not_stand_in_for_the_walker(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It differentiates whatever it is handed, so a block rules it out."""
    monkeypatch.setenv("LUCID_MANUAL_VJP", "0")
    ran = _run(lambda: _Holds(_under_no_grad), entry)
    assert ran.stayed_compiled() is False


def _plain() -> nn.Module:
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH))


def test_a_fused_update_is_no_block_to_the_autodiff_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The update ``fused_step`` records after the loss runs under ``no_grad``.

    It comes after the backward, so its ops are not marked: marked, they ruled
    out MPSGraph's autodiff for every model.
    """
    monkeypatch.setenv("LUCID_MANUAL_VJP", "0")
    assert _run(_plain, "fused_step").stayed_compiled()


def _branches_on_grad_mode(a: Tensor) -> Tensor:
    return a * 3.0 if lucid.is_grad_enabled() else a * 1.0


def _branch_reentrant(s: _Holds, y: Tensor) -> Tensor:
    return checkpoint(_branches_on_grad_mode, y, use_reentrant=True)


def _branch_nonreentrant(s: _Holds, y: Tensor) -> Tensor:
    return checkpoint(_branches_on_grad_mode, y, use_reentrant=False)


@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
@pytest.mark.parametrize(
    "body", [_branch_reentrant, _branch_nonreentrant], ids=["reentrant", "nonreentrant"]
)
def test_a_checkpoint_reading_the_grad_mode_runs_eagerly(body: Body, entry: str) -> None:
    """Eager runs it twice, under each grad mode: no one recording stands for both."""
    assert _run(lambda: _Holds(body), entry).stayed_compiled() is False


def _ops_marked(fn: Callable[[Tensor], Tensor], grad_mode: bool) -> list[str]:
    x = lucid.ones(2, WIDTH, device=COMPILE_DEVICE)
    with recording(watch=(x,), grad_mode=grad_mode) as tracer:
        fn(x)
    return [node.name for node in tracer.graph.ops if "grad_off" in node.attrs]


def _one_of_each(x: Tensor) -> Tensor:
    y = x * 2.0
    with lucid.no_grad():
        return y.exp() + F.relu(y)


def test_an_inference_recording_marks_nothing() -> None:
    """A recording wholly under ``no_grad`` answers no question about gradients.

    Its graph is the one Core ML and a compiled forward read, op for op.
    """
    assert _ops_marked(_one_of_each, grad_mode=False) == []


def test_a_differentiated_recording_marks_what_ran_under_no_grad() -> None:
    assert "exp" in _ops_marked(_one_of_each, grad_mode=True)
    assert "mul" not in _ops_marked(_one_of_each, grad_mode=True)


def _scaled_with_a_block(s: _Holds, y: Tensor) -> Tensor:
    w = y * s.p
    with lucid.no_grad():
        c = w.mean(0) * 0.1
    return w + c


def _scaled_without_a_block(s: _Holds, y: Tensor) -> Tensor:
    w = y * s.p
    c = w.mean(0) * 0.1
    return w + c


def test_the_cache_tells_a_block_from_the_same_ops_without_it() -> None:
    """Same ops, same inputs: a shared executable would answer the first one's gradient."""
    x = lucid.randn(6, WIDTH, device=COMPILE_DEVICE)
    for body in (_scaled_with_a_block, _scaled_without_a_block):
        model = _Holds(body).to(COMPILE_DEVICE)
        _loss(model(x)).backward()
        want = model.p.grad.detach().clone()
        model.zero_grad()
        lucid.compile.make_step(model, _loss)(x).backward()
        assert float((model.p.grad - want).abs().max().item()) < 3e-4


def _reads_inside(scopes: int) -> tuple[bool, str]:
    """Open ``scopes`` nested refusals around one read, in a recording."""
    x = lucid.ones(2, WIDTH, device=COMPILE_DEVICE)
    with recording(watch=(x,)) as tracer, ExitStack() as stack:
        for i in range(scopes):
            stack.enter_context(_grad_mode_reads_refused(f"scope {i}"))
        lucid.is_grad_enabled()
    return bool(tracer.graph.unsupported), str(tracer.graph.unsupported)


def test_a_read_of_the_grad_mode_marks_a_recording_only_inside_a_refusal() -> None:
    assert _reads_inside(0)[0] is False
    refused, why = _reads_inside(1)
    assert refused and "scope 0" in why


def test_the_innermost_refusal_gives_the_reason() -> None:
    assert "scope 1" in _reads_inside(2)[1]


def test_a_refusal_is_undone_when_its_body_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        with _grad_mode_reads_refused("scope"):
            1 / 0
    assert _reads_inside(0)[0] is False


def test_a_read_outside_any_recording_is_unchanged_by_a_refusal() -> None:
    with _grad_mode_reads_refused("scope"):
        assert lucid.is_grad_enabled() is True
        with lucid.no_grad():
            assert lucid.is_grad_enabled() is False
