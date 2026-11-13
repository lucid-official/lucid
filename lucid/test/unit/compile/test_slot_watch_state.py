"""What a :class:`SlotWatch` says of registered buffers and gradients, on the CPU (LCD-992).

The watch is the one owner of what a traced call leaves in Python that a compiled run
cannot reproduce.  Two kinds were not its to answer until LCD-992:

* a registered buffer a call replaces by a tensor of another shape.  A write into a buffer is
  carried, and a replacement of the same shape is made a write by ``Module.__setattr__``, but a
  history grown by ``cat`` is neither: the executable would be fed the history it read at trace
  time and the module would keep the first call's.  :meth:`SlotWatch.rebound` gives it the
  verdict an attribute has (the call read the old tensor, or the buffer held none and the new
  one is no constant); ``test_refused_trace_state.py`` runs the whole path on Metal;
* a gradient.  ``.backward()`` accumulates into ``.grad`` of every leaf it reaches, which is the
  tensor's state and no slot of a module: :meth:`SlotWatch.restore` puts it back.

Neither needs an executable, so both run where there is no Metal.
"""

import lucid
import lucid.nn as nn
from lucid.compile import _tracing
from lucid.compile._core.rebound_slots import SlotWatch

WIDTH = 4


class _History(nn.Module):
    def __init__(self, shape: tuple[int, ...] = (1, WIDTH)) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(WIDTH, WIDTH))
        self.register_buffer("hist", lucid.zeros(*shape), persistent=False)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        raise NotImplementedError


class _Grows(_History):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.hist = lucid.cat([self.hist, x.mean(0, keepdim=True)])
        return x @ self.w + self.hist.sum()


class _ReplacesWithoutReading(_History):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.hist = x.mean(0, keepdim=True).reshape(WIDTH)
        return x @ self.w


class _BuiltByTheFirstCall(_History):
    """A buffer that starts empty and is built from the first batch, as a per-channel observer's
    ``min_val`` is: what the executable would be fed of it is not read."""

    def __init__(self) -> None:
        super().__init__((0, WIDTH))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self.hist.shape[0] == 0:
            self.hist = x.mean(0, keepdim=True)
        return x @ self.w


class _StoresWhereNoneWas(_History):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("seen", None, persistent=False)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.seen = x.mean(0, keepdim=True)
        return x @ self.w


class _SameShape(_History):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.hist = self.hist * 0.5 + x.mean(0, keepdim=True)
        return x @ self.w


class _TableOnce(_History):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("table", None, persistent=False)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self.table is None:
            self.table = lucid.arange(WIDTH).float() * 0.1
        return x @ self.w + self.table


def _verdict(model: nn.Module) -> str | None:
    watch = SlotWatch([model], None)
    with lucid.no_grad(), _tracing(module=model) as tracer:
        model(lucid.ones(6, WIDTH))
    found = watch.rebound(tracer)
    return None if found is None else found.why


def test_a_buffer_the_call_reads_and_replaces_by_a_longer_one_has_a_verdict() -> None:
    why = _verdict(_Grows())
    assert why is not None
    assert "registered buffer 'hist' of _Grows, which the call reads" in why


def test_a_buffer_the_call_stores_where_it_held_none_has_a_verdict() -> None:
    why = _verdict(_StoresWhereNoneWas())
    assert why is not None
    assert "registered buffer 'seen' of _StoresWhereNoneWas, where it held no tensor" in why


def test_a_buffer_replaced_without_being_read_has_no_verdict() -> None:
    """As an attribute: nothing is fed the old tensor, and the one the module keeps is for whoever
    reads it outside the call."""
    assert _verdict(_ReplacesWithoutReading()) is None


def test_an_empty_buffer_the_first_call_builds_has_no_verdict() -> None:
    """The one that made the QAT observers fall back to eager when it had one."""
    assert _verdict(_BuiltByTheFirstCall()) is None


def test_a_replacement_of_the_same_shape_is_a_write_and_has_no_verdict() -> None:
    assert _verdict(_SameShape()) is None


def test_a_table_built_once_from_constants_has_no_verdict() -> None:
    assert _verdict(_TableOnce()) is None


def test_a_watch_restores_a_buffer_the_call_replaced() -> None:
    model = _Grows()
    kept = model.hist
    watch = SlotWatch([model], None)
    model(lucid.ones(6, WIDTH))
    assert model.hist.shape[0] == 2
    watch.restore()
    assert model.hist is kept


class _Backward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(WIDTH))
        self.register_buffer("tracked", lucid.ones(WIDTH).requires_grad_(), persistent=False)
        self.free = lucid.ones(WIDTH).requires_grad_()

    def forward(self) -> None:
        ((self.w + self.tracked + self.free) ** 2).sum().backward()


def _grads(model: _Backward) -> list[list[float] | None]:
    held = (model.w, model.tracked, model.free)
    return [None if t.grad is None else [float(v) for v in t.grad.tolist()] for t in held]


def test_a_watch_clears_a_gradient_the_call_made() -> None:
    """Every tensor the module holds, parameter, buffer or attribute, had none before."""
    model = _Backward()
    watch = SlotWatch([model], None)
    model()
    assert all(g is not None for g in _grads(model))
    watch.restore()
    assert _grads(model) == [None, None, None]


def test_a_watch_puts_back_the_values_of_a_gradient_the_call_added_to() -> None:
    model = _Backward()
    for t in (model.w, model.tracked, model.free):
        t.grad = lucid.full((WIDTH,), 5.0)
    watch = SlotWatch([model], None)
    model()
    assert _grads(model)[0] != [5.0] * WIDTH
    watch.restore()
    assert _grads(model) == [[5.0] * WIDTH] * 3
    watch.restore()  # again: the copy it holds is not the tensor it gave back
    model()
    watch.restore()
    assert _grads(model) == [[5.0] * WIDTH] * 3


def test_a_watch_puts_back_the_gradient_of_a_tensor_the_call_was_given() -> None:
    model = _Backward()
    x = lucid.ones(WIDTH).requires_grad_()
    watch = SlotWatch([model], None, [x])
    (x * x).sum().backward()
    assert x.grad is not None
    watch.restore()
    assert x.grad is None
