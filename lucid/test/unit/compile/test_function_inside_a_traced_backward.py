"""A compiled step gives eager's gradient when a traced ``backward`` applies a ``Function``.

A ``_traced_backward`` class's ``backward`` is recorded at ``apply``, and its ops
are then moved out of ``graph.ops`` into a :class:`VjpRegion`.  A ``Function``
applied inside that ``backward`` was treated like one on the forward path: it
committed its own region, indexed by where its ``forward`` ops stood right then.
The move took those ops away, so the region's range came to cover whatever the
trace recorded next.  The backward walker jumped over those ops, and the regions
of a ``backward`` recorded and then discarded did the same.  A parameter whose
gradient ran through the loss ops lost part of it, or all of it, without a word:
the wrong value or ``None``, where eager has a gradient.

A recorded ``backward`` is emitted, never differentiated, so a ``Function`` inside
it only ever runs its ``forward`` — what eager does under its first-order
``no_grad``.  Such a step stays compiled and matches eager.  Second derivatives
are refused by the compiled step's node, as they are for any compiled step.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.autograd import Function, checkpoint
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.trace_context import recording
from lucid.compile._entry.function import compiled_step

_LR = 1e-4


class _Scale(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 3.0

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 3.0


class _Plain(Function):
    """Not traced, with a ``backward`` no first-order pass may run."""

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 3.0

    @staticmethod
    def backward(ctx, g):
        del ctx, g
        raise AssertionError("a first-order backward ran the inner Function's backward")


class _Raises(Function):
    """Traced, with a ``backward`` no first-order pass may run."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 3.0

    @staticmethod
    def backward(ctx, g):
        del ctx, g
        raise AssertionError("a first-order backward ran the inner Function's backward")


class _Depth2(Function):
    """A traced ``Function`` whose ``forward`` applies another."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return _Scale.apply(a) * 1.0

    @staticmethod
    def backward(ctx, g):
        del ctx
        return _Scale.apply(g)


class _Passes(Function):
    """Hands its input back, so it records no ``forward`` op."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g


class _Multi(Function):
    """Two inner outputs in a ``backward``, for a region with two cotangents."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 3.0, a * 4.0

    @staticmethod
    def backward(ctx, g0, g1):
        del ctx
        return g0 * 3.0 + g1 * 4.0


def _two_outputs(name: str, inner: Callable[[lucid.Tensor], lucid.Tensor]) -> type[Function]:
    """A materializing two-output class whose ``backward`` runs ``inner`` on each cotangent."""

    def forward(ctx, a):
        del ctx
        return a * 2.0, a * 5.0

    def backward(ctx, g0, g1):
        del ctx
        return inner(g0) * 2.0 + inner(g1) * 5.0

    return type(
        name,
        (Function,),
        {
            "_traced_backward": True,
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )


class _NoMaterialize(Function):
    """``None`` for an output no gradient reached, in each demand's own variant."""

    _traced_backward = True
    _materialize_grads = False

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 2.0, a * 5.0

    @staticmethod
    def backward(ctx, g0, g1):
        del ctx
        total = None
        if g0 is not None:
            total = _Scale.apply(g0) * 2.0
        if g1 is not None:
            term = _Scale.apply(g1) * 5.0
            total = term if total is None else total + term
        return total


class _OnSaved(Function):
    """An inner call on a forward tensor ``backward`` saved, not on a cotangent."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * a

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return _Scale.apply(a) * g * (2.0 / 3.0)


class _SvdValsInBackward(Function):
    """A traced Lucid ``Function`` inside a user's traced ``backward``."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return g * 2.0 * lucid.linalg.svdvals(a).sum()


class _PlainPasses(Function):
    """Not traced, handing back a tensor the compiled loss descends from."""

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g


class _PlainOnSaved(Function):
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * a

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return _PlainPasses.apply(a) * g * 2.0


class _AppliesItself(Function):
    """Its ``backward`` applies the class being recorded."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 2.0

    @staticmethod
    def backward(ctx, g):
        del ctx
        return _AppliesItself.apply(g)


class _SubsetTraceFails(Function):
    """The ``{0}`` variant's recording fails after it applied a traced ``Function``.

    That recording's ops are discarded, and a region it left would cover later ops.
    """

    _traced_backward = True
    _materialize_grads = False

    @staticmethod
    def forward(ctx, a):
        del ctx
        return a * 2.0, a * 5.0

    @staticmethod
    def backward(ctx, g0, g1):
        del ctx
        term = None if g0 is None else _Scale.apply(g0) * 2.0
        assert g1 is not None
        return term * 1.0 + g1 * 5.0 if term is not None else g1 * 5.0


def _off_the_gradient_path(h: lucid.Tensor) -> lucid.Tensor:
    """The loss reads none of the call's outputs, only ops recorded after it."""
    _two_outputs("_OffPath", _traced_inner).apply(h)
    return h * 2.0


class _CheckpointBody(nn.Module):
    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        return _Scale.apply(h * 2.0)


def _checkpoint_around_traced(h: lucid.Tensor) -> lucid.Tensor:
    return checkpoint(_CheckpointBody(), h, use_reentrant=False)


class _NonDerivativeOuter(Function):
    """Wraps a traced call in ``forward``; its own ``backward`` is the one eager runs."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return _Scale.apply(a * 2.0) * 1.5

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 7.0


class _InnerAtTheEnd(Function):
    """The inner call's ops end where the outer's do, and begin later."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return _Scale.apply(a * 2.0)

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 7.0


class _SameOps(Function):
    """The inner call is the outer's whole ``forward``: one range, two regions."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        return _Scale.apply(a)

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 7.0


class _DemandInForward(Function):
    """A non-materializing call nested in ``forward``; only the outer's ``backward`` counts."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        del ctx
        o0, o1 = _NoMaterialize.apply(a)
        return o0 + o1

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 7.0


class _DemandAroundDemand(Function):
    """Both calls commit a demand group over the same ops."""

    _traced_backward = True
    _materialize_grads = False

    @staticmethod
    def forward(ctx, a):
        del ctx
        return _NoMaterialize.apply(a)

    @staticmethod
    def backward(ctx, g0, g1):
        del ctx
        if g0 is None:
            return g1 * 11.0
        return g0 * 7.0 if g1 is None else g0 * 7.0 + g1 * 11.0


_Call = Callable[[lucid.Tensor], object]


@dataclass(frozen=True)
class _Case:
    call: _Call
    # The outputs the loss ops after the call read.
    read: tuple[int, ...] = (0,)
    # Applied twice, so a stale range would reach the second call's ops as well.
    twice: bool = False


def _in_backward(inner: Callable[[lucid.Tensor], lucid.Tensor]) -> _Call:
    return _two_outputs("_Outer", inner).apply


def _traced_inner(g: lucid.Tensor) -> lucid.Tensor:
    return _Scale.apply(g)


_INNER_IN_BACKWARD: dict[str, _Case] = {
    "both-outputs": _Case(_in_backward(_traced_inner), (0, 1)),
    "first-output": _Case(_in_backward(_traced_inner), (0,)),
    "second-output": _Case(_in_backward(_traced_inner), (1,)),
    "two-calls-in-a-row": _Case(_in_backward(_traced_inner), (0, 1), twice=True),
    "no-materialize-both": _Case(_NoMaterialize.apply, (0, 1)),
    "no-materialize-first": _Case(_NoMaterialize.apply, (0,)),
    "no-materialize-second": _Case(_NoMaterialize.apply, (1,)),
    "no-materialize-two-calls": _Case(_NoMaterialize.apply, (0, 1), twice=True),
    "depth-2": _Case(_in_backward(lambda g: _Depth2.apply(g) * 1.0), (0, 1)),
    "depth-2-backward": _Case(_Depth2.apply),
    "inner-untraced": _Case(_in_backward(_Plain.apply), (0, 1)),
    "inner-backward-raises": _Case(_in_backward(_Raises.apply), (0, 1)),
    "inner-hands-back-input": _Case(_in_backward(_Passes.apply), (0, 1)),
    "inner-two-outputs": _Case(_in_backward(lambda g: sum(_Multi.apply(g))), (0, 1)),
    "inner-on-saved-tensor": _Case(_OnSaved.apply),
    "lucid-linalg": _Case(_SvdValsInBackward.apply),
    "inner-untraced-on-a-forward-tensor": _Case(_PlainOnSaved.apply),
    "applies-its-own-class": _Case(_AppliesItself.apply),
    "a-failed-subset-trace-both": _Case(_SubsetTraceFails.apply, (0, 1)),
    "a-failed-subset-trace-second": _Case(_SubsetTraceFails.apply, (1,)),
    "off-the-gradient-path": _Case(_off_the_gradient_path),
}

_INNER_IN_FORWARD: dict[str, _Case] = {
    "wrapped": _Case(_NonDerivativeOuter.apply),
    "inner-at-the-end": _Case(_InnerAtTheEnd.apply),
    "same-ops": _Case(_SameOps.apply),
    "demand-group-inside": _Case(_DemandInForward.apply),
    "demand-group-around-first": _Case(_DemandAroundDemand.apply, (0,)),
    "demand-group-around-second": _Case(_DemandAroundDemand.apply, (1,)),
    "demand-group-around-both": _Case(_DemandAroundDemand.apply, (0, 1)),
    "checkpoint-around-traced": _Case(_checkpoint_around_traced),
}

_COMPILES = {**_INNER_IN_BACKWARD, **_INNER_IN_FORWARD}


class _Weighted(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.p = nn.Parameter(lucid.tensor([[1.0, -0.5, 0.5]], device=device))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.p * x


class _ThroughCall(nn.Module):
    """The call, then loss ops — what a stale region's range lands on."""

    def __init__(self, case: _Case) -> None:
        super().__init__()
        self.case = case

    def _once(self, h: lucid.Tensor) -> lucid.Tensor:
        out = self.case.call(h)
        got = out if isinstance(out, tuple) else (out,)
        return sum(((got[j] * 0.25).exp() * 0.5).sum().reshape(1) for j in self.case.read)

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        total = self._once(h)
        return total + self._once(h * 0.5) if self.case.twice else total


class _Scaled(nn.Module):
    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        return h * 1.5


def _model(case: _Case, device: str) -> nn.Module:
    return nn.Sequential(_Weighted(device), _ThroughCall(case), _Scaled())


def _data(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    return lucid.tensor([[1.0, 0.5, 2.0]], device=device), lucid.ones(1, device=device)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _weight(model: nn.Module) -> lucid.Tensor:
    return next(iter(model.parameters()))


def _eager_gradient(case: _Case, device: str) -> list[float]:
    model = _model(case, device)
    x, t = _data(device)
    _loss(model(x), t).backward()
    grad = _weight(model).grad
    assert grad is not None
    assert bool(lucid.isfinite(grad).all().item())
    # A row that may as well read as zero proves nothing about a missing gradient.
    assert all(abs(v) > 1e-3 for v in grad.reshape(-1).tolist())
    return grad.reshape(-1).tolist()


def _assert_same(got: list[float], want: list[float], what: str) -> None:
    assert lucid.allclose(
        lucid.tensor(got), lucid.tensor(want), rtol=1e-3, atol=1e-5
    ), f"{what}: compiled {got} != eager {want}"


def _gradient_of(model: nn.Module, what: str) -> list[float]:
    grad = _weight(model).grad
    assert grad is not None, f"{what}: the compiled step left the gradient None"
    return grad.reshape(-1).tolist()


def _by_make_step(case: _Case, device: str) -> list[list[float]]:
    model = _model(case, device)
    step = make_step(model, _loss)
    x, t = _data(device)
    got = []
    for _ in range(2):  # The trace, then the cached executable.
        _weight(model).grad = None
        step(x, t).backward()
        got.append(_gradient_of(model, "make_step"))
    assert not step.eager_only
    assert step.cache
    return got


def _by_module_step(case: _Case, device: str) -> list[list[float]]:
    model = _model(case, device)
    compiled = lucid.compile(model)
    x, t = _data(device)
    got = []
    for _ in range(2):
        _weight(model).grad = None
        compiled.step(x, t, loss_fn=_loss).backward()
        got.append(_gradient_of(model, "CompiledModule.step"))
    return got


def _by_segments(case: _Case, device: str) -> list[list[float]]:
    model = _model(case, device)
    step = make_step(model, _loss, segments=3)
    x, t = _data(device)
    got = []
    for _ in range(2):
        _weight(model).grad = None
        step(x, t).backward()
        got.append(_gradient_of(model, "make_step(segments=3)"))
    assert not step.eager_only
    return got


def _by_compiled_step(case: _Case, device: str) -> list[list[float]]:
    model = _model(case, device)
    x, t = _data(device)
    compiled_step(model, x, lambda y: _loss(y, t))
    return [_gradient_of(model, "compiled_step")]


def _by_fused_step(case: _Case, device: str) -> list[list[float]]:
    """The gradient the update implies: a plain SGD step is ``-lr * grad``."""
    model = _model(case, device)
    before = _weight(model).detach().clone()
    step = fused_step(model, _loss, optim.SGD(model.parameters(), lr=_LR))
    x, t = _data(device)
    step(x, t)
    assert not step.cache_info()["eager_only"]
    implied = (before - _weight(model).detach()) / _LR
    return [implied.reshape(-1).tolist()]


_ENTRIES: dict[str, Callable[[_Case, str], list[list[float]]]] = {
    "make_step": _by_make_step,
    "module_step": _by_module_step,
    "segments": _by_segments,
    "compiled_step": _by_compiled_step,
    "fused_step": _by_fused_step,
}


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
@pytest.mark.parametrize("name", sorted(_COMPILES))
def test_the_step_compiles_and_matches_eager(name: str, entry: str, device_gpu_only: str) -> None:
    case = _COMPILES[name]
    want = _eager_gradient(case, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = _ENTRIES[entry](case, device_gpu_only)
    for k, grad in enumerate(got):
        # The fused step's implied gradient is a difference of float32 weights.
        if entry == "fused_step":
            assert lucid.allclose(
                lucid.tensor(grad), lucid.tensor(want), rtol=5e-2, atol=1e-2
            ), f"fused_step: compiled {grad} != eager {want}"
        else:
            _assert_same(grad, want, f"{entry} run {k}")


@pytest.mark.parametrize("name", sorted(_INNER_IN_BACKWARD))
def test_every_region_stays_inside_the_ops_of_its_call(name: str, device_gpu_only: str) -> None:
    """A region covers forward ops that stay in ``graph.ops``, nothing recorded later."""
    case = _INNER_IN_BACKWARD[name]
    h = lucid.tensor([[1.0, -2.0, 0.5]], device=device_gpu_only, requires_grad=True)
    with recording(watch=(h,)) as tracer:
        out = case.call(h)
        ops_after_call = int(tracer.op_count())
        for tensor in out if isinstance(out, tuple) else (out,):
            (tensor * 1.5).exp().sum()
    regions = tracer.graph.vjp_regions
    assert regions, "the outer call must have been traced"
    assert all(int(r.fwd_end) <= ops_after_call for r in regions)


@pytest.mark.parametrize(
    "name", ["both-outputs", "no-materialize-both", "depth-2", "inner-on-saved-tensor"]
)
def test_create_graph_on_a_compiled_step_is_refused_not_answered_without_the_graph(
    name: str, device_gpu_only: str
) -> None:
    """A second derivative runs the inner ``backward``; a compiled step has none to give."""
    case = _COMPILES[name]
    x, t = _data(device_gpu_only)

    # Eager has one, so the refusal is not a thing every engine would answer.
    eager = _model(case, device_gpu_only)
    _loss(eager(x), t).backward(create_graph=True)
    grad = _weight(eager).grad
    assert grad is not None
    assert grad.requires_grad
    (grad * grad).sum().backward()

    model = _model(case, device_gpu_only)
    step = make_step(model, _loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        loss = step(x, t)
    with pytest.raises(NotImplementedError, match="once_differentiable"):
        loss.backward(create_graph=True)
    assert not step.eager_only
    assert _weight(model).grad is None


class _OffMetalInner(Function):
    @staticmethod
    def forward(ctx, a):
        del ctx
        return (a.cpu() * 3.0).to(a.device)

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 3.0


class _HostReadInner(Function):
    """An inner call whose ops a compiled backward cannot emit: the outer cannot be traced."""

    @staticmethod
    def forward(ctx, a):
        del ctx
        return lucid.tensor(a.tolist(), device=a.device) * 3.0

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 3.0


_REFUSED: dict[str, tuple[_Call, str]] = {
    "off-metal": (_in_backward(_OffMetalInner.apply), "backward could not be traced"),
    "host-read": (_in_backward(_HostReadInner.apply), "backward could not be traced"),
    "off-metal-after-a-traced-call": (
        _in_backward(lambda g: _OffMetalInner.apply(_Scale.apply(g))),
        "backward could not be traced",
    ),
}


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_an_inner_call_that_cannot_be_recorded_sends_the_step_eager_loudly(
    name: str, device_gpu_only: str
) -> None:
    call, why = _REFUSED[name]
    case = _Case(call, (0, 1))
    want = _eager_gradient(case, device_gpu_only)
    model = _model(case, device_gpu_only)
    step = make_step(model, _loss)
    x, t = _data(device_gpu_only)
    with pytest.warns(CompileFallbackWarning, match=why):
        step(x, t).backward()
    assert len(step.eager_only) == 1
    _assert_same(_gradient_of(model, "fallback"), want, "fallback")


class _TwoParams(nn.Module):
    """Each parameter feeds a different input; the loss reads one output."""

    def __init__(self, device: str, call: Callable[..., tuple[lucid.Tensor, lucid.Tensor]]) -> None:
        super().__init__()
        self.p = nn.Parameter(lucid.tensor([1.0, -2.0], device=device))
        self.q = nn.Parameter(lucid.tensor([0.5, 0.25], device=device))
        self.call = call

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out, _ = self.call(self.p * x, self.q * x)
        return (out * 1.5).exp().reshape(-1)[:1] * 0.01


class _NoneForTheUnread(Function):
    """A parameter behind an output nothing reads gets ``None`` from eager, not zero."""

    _traced_backward = True
    _materialize_grads = False

    @staticmethod
    def forward(ctx, a, b):
        del ctx
        return a * 2.0, b * 5.0

    @staticmethod
    def backward(ctx, g0, g1):
        del ctx
        return (
            None if g0 is None else _Scale.apply(g0) * 2.0,
            None if g1 is None else _Scale.apply(g1) * 5.0,
        )


def test_a_parameter_eager_leaves_none_is_none_not_zero(device_gpu_only: str) -> None:
    x = lucid.ones(2, device=device_gpu_only)
    t = lucid.ones(1, device=device_gpu_only)

    eager = _TwoParams(device_gpu_only, _NoneForTheUnread.apply)
    _loss(eager(x), t).backward()
    assert eager.p.grad is not None
    assert eager.q.grad is None

    model = _TwoParams(device_gpu_only, _NoneForTheUnread.apply)
    step = make_step(model, _loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    assert not step.eager_only
    assert model.q.grad is None
    assert model.p.grad is not None
    _assert_same(model.p.grad.tolist(), eager.p.grad.tolist(), "the reached parameter")
