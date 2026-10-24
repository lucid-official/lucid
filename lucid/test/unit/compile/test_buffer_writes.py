"""A compiled call's in-place writes to its model's buffers.

Every form below used to happen once — on the call that traced — and then
never again: the executable computed the new value, nothing stored it, and
every later compiled call left the buffer where the trace had put it.  Two
engine faults made it so.  An in-place op splices an out-of-place result into
its target, and the tracer was never told — so ``a.mul_(b).add_(c)`` even
recorded the add reading ``a`` from before the mul — and a Python-level
rebinding (``t[i] = v``, ``module.buf = new``) swapped the tensor outside the
trace's sight.  Now the tracer follows the splice, a trace writes into the
tensor instead of rebinding it, and the compiled call carries each written
buffer's new value back (``lucid/compile/_core/buffer_writes.py``).

Any tensor the call holds from outside is carried back the same way, buffer or
not.  A write to one of the call's own arguments is not — the next call brings
other tensors — so that call runs eager: slower, not wrong, and not applied
twice on the call that traced.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

KINDS = ("chain", "copy", "count", "setitem", "fill", "reassign")


class _Writer(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.lin = nn.Linear(4, 1)
        self.register_buffer("ema", lucid.zeros(()))
        self.register_buffer("count", lucid.zeros((), dtype=lucid.int64))
        self.register_buffer("vec", lucid.zeros(3))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = self.lin(x)
        with lucid.no_grad():
            m = y.mean().detach()
            if self.kind == "chain":
                self.ema.mul_(0.5).add_(m * 0.5)
            elif self.kind == "copy":
                self.ema.copy_(self.ema * 0.5 + m * 0.5)
            elif self.kind == "count":
                self.count.add_(1)
            elif self.kind == "setitem":
                self.vec[1] = self.vec[1] + m
            elif self.kind == "fill":
                self.ema.fill_(3.0)
            elif self.kind == "reassign":
                self.ema = self.ema * 0.5 + m * 0.5
        return y


def _history(kind: str, mode: str) -> list[tuple[float, int, list[float]]]:
    lucid.manual_seed(0)
    model = _Writer(kind).to(COMPILE_DEVICE)
    step = lucid.compile.make_step(model, lambda out: (out * out).mean())
    compiled = lucid.compile(model)
    seen: list[tuple[float, int, list[float]]] = []
    for i in range(3):
        x = lucid.full((3, 4), float(i + 1), device=COMPILE_DEVICE)
        if mode == "step":
            step(x).backward()
        elif mode == "module":
            with lucid.no_grad():
                compiled(x)
        else:
            model(x)
        seen.append(
            (
                round(float(model.ema.item()), 5),
                int(model.count.item()),
                [round(v, 5) for v in model.vec.tolist()],
            )
        )
    if mode == "step":
        assert len(step.cache) == 1 and not step.eager_only.snapshot()
    if mode == "module":
        info = compiled.cache_info()
        assert info["entries"] == 1 and not info["eager_only"]
    return seen


@pytest.mark.parametrize("mode", ["step", "module"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_compiled_call_writes_its_buffers_every_call(kind: str, mode: str) -> None:
    assert _history(kind, mode) == _history(kind, "eager")


def test_chained_in_place_ops_trace_in_order() -> None:
    from lucid._dispatch import _unwrap
    from lucid.compile import _tracing

    ema = lucid.zeros((), device=COMPILE_DEVICE)
    x = lucid.ones(3, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing() as tracer:
        ema.mul_(0.5).add_(x.mean() * 0.5)
    ops = list(tracer.graph.ops)
    first_mul = next(op for op in ops if op.name == "mul")
    add = next(op for op in ops if op.name == "add")
    # The add reads the mul's result, not ``ema`` from before it.
    assert add.inputs[0] == first_mul.outputs[0].id
    assert tracer.lookup_id(_unwrap(ema)) == add.outputs[0].id


@pytest.mark.usefixtures("inference")
def test_a_tensor_held_from_outside_is_carried_back() -> None:
    tally = lucid.zeros((), device=COMPILE_DEVICE)

    class _Counts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            with lucid.no_grad():
                tally.add_(1.0)
            return self.lin(x)

    compiled = lucid.compile(_Counts().to(COMPILE_DEVICE))
    for _ in range(3):
        compiled(lucid.ones(2, 4, device=COMPILE_DEVICE))
    assert float(tally.item()) == 3.0
    assert not compiled.cache_info()["eager_only"]


@pytest.mark.usefixtures("inference")
def test_a_write_to_an_argument_runs_eager_once_per_call() -> None:
    class _Bumps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            with lucid.no_grad():
                x.add_(1.0)
            return self.lin(x)

    compiled = lucid.compile(_Bumps().to(COMPILE_DEVICE))
    x = lucid.zeros(2, 4, device=COMPILE_DEVICE)
    for _ in range(3):
        compiled(x)
    # Once per call — including the call that traced.
    assert float(x[0, 0].item()) == 3.0
    assert compiled.cache_info()["eager_only"]
