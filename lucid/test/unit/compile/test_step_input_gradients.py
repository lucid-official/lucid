"""A compiled training step adds every gradient eager would, inputs too (LCD-617).

``make_step``'s backward returned ``None`` for every argument, so a compiled
head fed by an eager trunk (``loss = step(trunk(inp), t)``) left the trunk with
no gradient and trained it not at all, without a word.  That was on Metal,
where the step compiles; on CPU the step runs eagerly and was right.  The
segmented step's first segment did the same.  ``fused_step`` runs its backward
inside the executable and returns a loss with no ``grad_fn``, so it has no way
to return an input gradient.  It dropped the gradient when it compiled and
computed it when it fell back to eager; it now refuses such an input on every
path.

``compiled_step`` dropped the input's gradient too, and assigned ``p.grad``
where ``backward`` adds: a second step without ``zero_grad`` left 1x the
gradient where eager has 2x, a frozen parameter got one, and a held
``g = p.grad`` saw nothing.  ``segments="auto"`` timed its candidates on the
call's own tensors, then set every ``.grad`` to None — the gradient already
there was lost, and with input gradients flowing, the timing runs would have
backpropagated into the caller's trunk.

Each case is held to eager on an identical model, on CPU and Metal.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.compile._entry.segmented_step as segmented_mod
import lucid.compile._entry.step as step_mod
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.autograd.function import Function, FunctionCtx
from lucid.compile import compiled_step, fused_step, make_step

pytestmark = pytest.mark.filterwarnings(
    "ignore::lucid.compile._core.fallback.CompileFallbackWarning"
)

_TOL = 1e-5

_Step = Callable[..., lucid.Tensor]
_LossFn = Callable[..., lucid.Tensor]

_STEPS: dict[str, Callable[[nn.Module, _LossFn], _Step]] = {
    "monolithic": lambda model, loss_fn: make_step(model, loss_fn),
    "segmented": lambda model, loss_fn: make_step(model, loss_fn, segments=2),
}


def _head(device: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)


def _trunk(device: str) -> nn.Sequential:
    lucid.manual_seed(1)
    return nn.Sequential(nn.Linear(10, 16), nn.Tanh()).to(device)


def _data(device: str) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(2)
    inp = lucid.randn(8, 10, device=device)
    x = lucid.randn(8, 16, device=device)
    t = lucid.randint(0, 4, (8,), device=device)
    return inp, x, t


def _leaf(t: lucid.Tensor) -> lucid.Tensor:
    return t.detach().clone().requires_grad_(True)


def _close(got: lucid.Tensor | None, want: lucid.Tensor | None) -> bool:
    assert got is not None and want is not None, (got, want)
    scale = max(1.0, float(want.abs().max().item()))
    return float((got - want).abs().max().item()) <= _TOL * scale


def _ran_compiled(step: _Step, kind: str, device: str) -> None:
    """On Metal the step must have compiled, not fallen back to eager."""
    if device != "metal":
        return
    if kind == "monolithic":
        assert not step.eager_only and step.cache
    else:
        assert not step.eager_only["flag"]


@pytest.mark.parametrize("kind", sorted(_STEPS))
def test_a_leaf_input_gets_the_eager_gradient(kind: str, device: str) -> None:
    _, x0, t = _data(device)
    eager, compiled = _head(device), _head(device)
    xe, xc = _leaf(x0), _leaf(x0)
    (F.cross_entropy(eager(xe), t) * 2.0).backward()

    step = _STEPS[kind](compiled, F.cross_entropy)
    (step(xc, t) * 2.0).backward()

    _ran_compiled(step, kind, device)
    assert _close(xc.grad, xe.grad)
    for pc, pe in zip(compiled.parameters(), eager.parameters(), strict=True):
        assert _close(pc.grad, pe.grad)


def _train_once(
    trunk: nn.Module,
    head: nn.Module,
    loss_of: _LossFn,
    inp: lucid.Tensor,
    t: lucid.Tensor,
) -> float:
    opt = optim.SGD([*trunk.parameters(), *head.parameters()], lr=0.5)
    opt.zero_grad()
    loss = loss_of(trunk(inp), t)
    loss.backward()
    opt.step()
    return float(loss.item())


@pytest.mark.parametrize("kind", sorted(_STEPS))
def test_an_eager_trunk_trains_under_a_compiled_head(kind: str, device: str) -> None:
    inp, _, t = _data(device)
    trunk_e, head_e = _trunk(device), _head(device)
    trunk_c, head_c = _trunk(device), _head(device)
    before = [p.detach().clone() for p in trunk_c.parameters()]

    want = _train_once(
        trunk_e, head_e, lambda x, tt: F.cross_entropy(head_e(x), tt), inp, t
    )
    step = _STEPS[kind](head_c, F.cross_entropy)
    got = _train_once(trunk_c, head_c, step, inp, t)

    _ran_compiled(step, kind, device)
    assert abs(got - want) <= _TOL * max(1.0, abs(want))
    for p, b in zip(trunk_c.parameters(), before, strict=True):
        assert float((p - b).abs().max().item()) > 1e-4, "the trunk did not move"
    pairs = zip(
        [*trunk_c.parameters(), *head_c.parameters()],
        [*trunk_e.parameters(), *head_e.parameters()],
        strict=True,
    )
    for pc, pe in pairs:
        assert _close(pc, pe)


def _sq_err(y: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    d = y - target
    return (d * d).mean()


@pytest.mark.parametrize("which", ["input", "target", "both", "same-tensor"])
def test_each_tensor_that_requires_grad_gets_its_gradient(
    which: str, device: str
) -> None:
    lucid.manual_seed(3)
    x0 = lucid.randn(4, 6, device=device)
    y0 = lucid.randn(4, 6, device=device)

    def args() -> tuple[lucid.Tensor, lucid.Tensor]:
        x = _leaf(x0) if which in ("input", "both", "same-tensor") else x0
        y = _leaf(y0) if which in ("target", "both") else y0
        return (x, x) if which == "same-tensor" else (x, y)

    lucid.manual_seed(4)
    eager = nn.Linear(6, 6).to(device)
    lucid.manual_seed(4)
    compiled = nn.Linear(6, 6).to(device)
    xe, ye = args()
    _sq_err(eager(xe), ye).backward()
    xc, yc = args()
    step = make_step(compiled, _sq_err)
    step(xc, yc).backward()

    _ran_compiled(step, "monolithic", device)
    for got, want in ((xc, xe), (yc, ye)):
        if want.requires_grad:
            assert _close(got.grad, want.grad)
        else:
            assert got.grad is None


def _reads_aux_off_the_loss_path(y: lucid.Tensor, aux: lucid.Tensor) -> lucid.Tensor:
    _metric = aux * 3.0  # traced, discarded: the loss does not depend on aux
    return (y * y).mean()


def test_an_input_the_loss_does_not_depend_on_gets_none(device: str) -> None:
    _, x, _ = _data(device)
    aux_e, aux_c = _leaf(x), _leaf(x)
    eager, compiled = _head(device), _head(device)
    _reads_aux_off_the_loss_path(eager(x), aux_e).backward()
    step = make_step(compiled, _reads_aux_off_the_loss_path)
    step(x, aux_c).backward()

    _ran_compiled(step, "monolithic", device)
    assert aux_e.grad is None and aux_c.grad is None
    for pc, pe in zip(compiled.parameters(), eager.parameters(), strict=True):
        assert _close(pc.grad, pe.grad)


class _Pair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(5, 3)

    def forward(self, xs: list[lucid.Tensor]) -> lucid.Tensor:
        return self.a(xs[0]) * self.b(xs[1])


def test_a_nested_input_gets_its_gradient(device: str) -> None:
    lucid.manual_seed(5)
    a0, b0 = lucid.randn(2, 4, device=device), lucid.randn(2, 5, device=device)
    lucid.manual_seed(6)
    eager = _Pair().to(device)
    lucid.manual_seed(6)
    compiled = _Pair().to(device)
    ae, ac = _leaf(a0), _leaf(a0)
    _sq_err(eager([ae, b0]), lucid.zeros(2, 3, device=device)).backward()
    step = make_step(compiled, _sq_err)
    step([ac, b0], lucid.zeros(2, 3, device=device)).backward()

    _ran_compiled(step, "monolithic", device)
    assert _close(ac.grad, ae.grad)


class _ScaleBy5(Function):
    """Identity forward, five-fold backward: not the derivative of its forward."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        return x * 1.0

    @staticmethod
    def backward(ctx: FunctionCtx, grad: lucid.Tensor) -> lucid.Tensor:
        return grad * 5.0


class _ScaledInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(_ScaleBy5.apply(x))


def test_a_custom_backward_on_the_input_path_runs_eagerly(device: str) -> None:
    """LCD-565 still holds: the input's gradient passes through the custom node."""
    _, x0, t = _data(device)
    lucid.manual_seed(7)
    eager = _ScaledInput().to(device)
    lucid.manual_seed(7)
    compiled = _ScaledInput().to(device)
    xe, xc = _leaf(x0), _leaf(x0)
    F.cross_entropy(eager(xe), t).backward()
    step = make_step(compiled, F.cross_entropy)
    step(xc, t).backward()

    assert step.eager_only
    assert _close(xc.grad, xe.grad)


def test_a_step_whose_inputs_need_no_gradient_computes_none(
    monkeypatch: pytest.MonkeyPatch, device_gpu_only: str
) -> None:
    """One executable per pattern of inputs that require grad, built once each."""
    device = device_gpu_only
    traces: list[int] = []
    trace = step_mod._trace_step

    def counted(*args: object, **kwargs: object) -> object:
        traces.append(1)
        return trace(*args, **kwargs)

    monkeypatch.setattr(step_mod, "_trace_step", counted)
    _, x, t = _data(device)
    step = make_step(_head(device), F.cross_entropy)
    for _ in range(3):
        step(x, t).backward()
    assert len(traces) == 1 and len(step.cache) == 1
    (entry,) = step.cache.values()
    assert entry.input_grads == ()

    xg = _leaf(x)
    step(xg, t).backward()
    assert len(traces) == 2 and len(step.cache) == 2
    step(x, t).backward()
    step(xg, t).backward()
    assert len(traces) == 2


def test_a_segmented_step_keys_its_first_segment_on_the_grad_need(
    monkeypatch: pytest.MonkeyPatch, device_gpu_only: str
) -> None:
    device = device_gpu_only
    compiles: list[int] = []
    compile_segment = segmented_mod._compile_segment

    def counted(*args: object) -> object:
        compiles.append(1)
        return compile_segment(*args)

    monkeypatch.setattr(segmented_mod, "_compile_segment", counted)
    _, x, t = _data(device)
    step = make_step(_head(device), F.cross_entropy, segments=2)
    for _ in range(2):
        step(x, t).backward()
    assert len(compiles) == 2  # one per segment
    xg = _leaf(x)
    step(xg, t).backward()
    step(x, t).backward()
    step(xg, t).backward()
    # Only the first segment's input changed its need; the second's always
    # requires grad (the head's parameters do).
    assert len(compiles) == 3
    assert xg.grad is not None


@pytest.mark.parametrize("source", ["leaf", "trunk"])
def test_fused_step_refuses_an_input_that_requires_grad(
    source: str, device: str
) -> None:
    inp, x0, t = _data(device)
    head = _head(device)
    step = fused_step(head, F.cross_entropy, optim.SGD(head.parameters(), lr=0.5))
    x = _leaf(x0) if source == "leaf" else _trunk(device)(inp)
    before = [p.detach().clone() for p in head.parameters()]

    with pytest.raises(RuntimeError, match="input 0 requires grad"):
        step(x, t)
    for p, b in zip(head.parameters(), before, strict=True):
        assert float((p - b).abs().max().item()) == 0.0
    assert step.cache_info()["entries"] == 0
    if device != "metal":
        return  # the compiled optimizer fused_step drives is Metal-only

    step(x.detach(), t)  # a detached input trains the head as before
    moved = [
        float((p - b).abs().max().item())
        for p, b in zip(head.parameters(), before, strict=True)
    ]
    assert max(moved) > 1e-4


# ── compiled_step, and every entry: gradients add, as ``backward`` adds ──────


def _same_grads(got: nn.Module, want: nn.Module) -> None:
    for pc, pe in zip(got.parameters(), want.parameters(), strict=True):
        if pe.grad is None:
            assert pc.grad is None
        else:
            assert _close(pc.grad, pe.grad)


@pytest.mark.parametrize("source", ["leaf", "trunk"])
def test_compiled_step_hands_the_input_its_gradient(
    source: str, device_gpu_only: str
) -> None:
    """``compiled_step`` has no eager path: on a CPU trace it raises."""
    device = device_gpu_only
    inp, x0, t = _data(device)
    eager, compiled = _head(device), _head(device)
    trunk_e, trunk_c = _trunk(device), _trunk(device)
    for model in (eager, compiled):
        model[0].bias.requires_grad_(False)  # eager gives a frozen one nothing
    xe, xc = (
        (_leaf(x0), _leaf(x0)) if source == "leaf" else (trunk_e(inp), trunk_c(inp))
    )

    want = F.cross_entropy(eager(xe), t)
    want.backward()
    got = compiled_step(compiled, xc, lambda y: F.cross_entropy(y, t))

    assert abs(float(got.item()) - float(want.item())) <= _TOL
    assert compiled[0].bias.grad is None
    _same_grads(compiled, eager)
    if source == "leaf":
        assert _close(xc.grad, xe.grad)
    else:
        _same_grads(trunk_c, trunk_e)


_KINDS = ["compiled_step", "monolithic", "segmented", "autoseg"]


def _runner(
    kind: str, model: nn.Module
) -> tuple[Callable[[lucid.Tensor, lucid.Tensor], None], _Step | None]:
    """One training step of ``kind`` — its backward run — and the step object."""
    if kind == "compiled_step":

        def run_compiled(x: lucid.Tensor, t: lucid.Tensor) -> None:
            compiled_step(model, x, lambda y: F.cross_entropy(y, t))

        return run_compiled, None
    segments: int | str = {"monolithic": 1, "segmented": 2, "autoseg": "auto"}[kind]
    step = make_step(model, F.cross_entropy, segments=segments)

    def run(x: lucid.Tensor, t: lucid.Tensor) -> None:
        step(x, t).backward()

    return run, step


@pytest.mark.parametrize("kind", _KINDS)
def test_a_step_adds_into_the_gradient_already_there(kind: str, device: str) -> None:
    """Two steps after an eager one, no ``zero_grad``: 3x, into the held tensor."""
    if kind == "compiled_step" and device != "metal":
        pytest.skip("compiled_step has no eager path: a CPU trace raises")
    _, x, t = _data(device)
    eager, compiled = _head(device), _head(device)
    for model in (eager, compiled):
        F.cross_entropy(model(x), t).backward()
    held = [p.grad for p in compiled.parameters()]

    run, step = _runner(kind, compiled)
    for _ in range(2):
        F.cross_entropy(eager(x), t).backward()
        run(x, t)

    if step is not None and kind != "autoseg":
        _ran_compiled(step, kind, device)
    _same_grads(compiled, eager)
    for h, pe in zip(held, eager.parameters(), strict=True):
        assert _close(h, pe.grad), "the held gradient did not take the sum"


def test_auto_segments_time_their_candidates_off_the_callers_graph(
    device: str,
) -> None:
    """The timing runs neither reach the trunk nor survive on the head."""
    inp, _, t = _data(device)
    trunk_e, head_e = _trunk(device), _head(device)
    trunk_c, head_c = _trunk(device), _head(device)
    F.cross_entropy(head_e(trunk_e(inp)), t).backward()

    step = make_step(head_c, F.cross_entropy, segments="auto")
    step(trunk_c(inp), t).backward()

    assert step.chosen_segments() is not None
    _same_grads(trunk_c, trunk_e)
    _same_grads(head_c, head_e)
