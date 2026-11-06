"""A view taken before an in-place write to its base reads the write (LCD-837).

Eager keeps every view of one buffer in a family that reads the buffer, so
``v = y[1]; y.mul_(2.0); return v`` answers the doubled row.  The recording
only moved the written tensor to a new id: ``v`` kept the id of its value
from before the write, and the compiled call returned that — a plausible
number, off from eager, with no error and no fallback.  (``y.mT`` was off by
exactly the write.)

Now a write hands the tracer every view it touches as stale, and reading
one sends the call to eager.  A read is an op's input, the value another
write copies in (``z[:] = v``), a result the call returns, a loss.  Nothing
is refused until a stale view is read: a view taken after the write, a view
nobody reads again, a view read only before the write, and a view the write
does not reach all keep compiling.

The property is eager agreement on two inputs that eager tells apart.  The
matrices cover every view kind against every write kind, read three ways
(returned, consumed by an op, copied into another tensor); the rest pin the
other shapes the bug takes — a base the trace made itself, a read before the
write, overlapping buffers the call writes, a view of batch norm's running
statistics, and a Core ML export.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.coreml import _build as cml_build
from lucid.compile._optim.compiler import _CompiledSGD
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except RuntimeError:
        return False


needs_metal = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

SHAPE = (3, 4)

Fn = Callable[[lucid.Tensor], lucid.Tensor]
View = Callable[[lucid.Tensor], lucid.Tensor]
Write = Callable[[lucid.Tensor, lucid.Tensor], None]
Read = Callable[[lucid.Tensor], lucid.Tensor]

VIEWS: dict[str, View] = {
    "index": lambda y: y[1],
    "slice": lambda y: y[:2],
    "transpose": lambda y: y.mT,
    "reshape": lambda y: y.reshape(4, 3),
    "unsqueeze": lambda y: y.unsqueeze(0),
    "diagonal": lambda y: lucid.diagonal(y),
    # ``y`` is contiguous, so this is ``y`` itself; and it stays apart from ``y``
    # when it is a copy, which is what a transpose makes it.
    "contiguous": lambda y: y.contiguous(),
    "contiguous-free": lambda y: lucid.contiguous(y),
    "contiguous-copy": lambda y: y.mT.contiguous(),
    # A dropout with nothing to drop is its input, as ``contiguous`` is.
    "dropout-eval": lambda y: F.dropout(y, 0.5, False),
    "dropout-p0": lambda y: F.dropout(y, 0.0, True),
    "dropout-p1-eval": lambda y: F.dropout(y, 1.0, False),
    "dropout-eval-inplace": lambda y: F.dropout(y, 0.5, False, inplace=True),
    "dropout1d-eval": lambda y: F.dropout1d(y, 0.5, False),
    "dropout1d-p0": lambda y: F.dropout1d(y, 0.0, True),
    "dropout2d-eval": lambda y: F.dropout2d(y.unsqueeze(0), 0.5, False)[0],
    "dropout2d-p0": lambda y: F.dropout2d(y.unsqueeze(0), 0.0, True)[0],
    "dropout3d-eval": lambda y: F.dropout3d(y.unsqueeze(0).unsqueeze(0), 0.5, False)[0][0],
    "dropout3d-p0": lambda y: F.dropout3d(y.unsqueeze(0).unsqueeze(0), 0.0, True)[0][0],
    "alpha-dropout-eval": lambda y: F.alpha_dropout(y, 0.5, False),
    "alpha-dropout-p0": lambda y: F.alpha_dropout(y, 0.0, True),
    "feature-alpha-dropout-eval": lambda y: F.feature_alpha_dropout(y, 0.5, False),
    "feature-alpha-dropout-p0": lambda y: F.feature_alpha_dropout(y, 0.0, True),
}


def _slice_assign(y: lucid.Tensor, x: lucid.Tensor) -> None:
    y[1:] = x[1:] * 3.0


def _list_index(y: lucid.Tensor, x: lucid.Tensor) -> None:
    y[[0, 2]] = x[[1, 2]]


def _through_sibling(y: lucid.Tensor, x: lucid.Tensor) -> None:
    sibling = y[1:3]
    sibling.mul_(3.0)


WRITES: dict[str, Write] = {
    "mul_": lambda y, x: y.mul_(2.0),
    "add_": lambda y, x: y.add_(1.0),
    "copy_": lambda y, x: y.copy_(x * 3.0 + 1.0),
    "slice-assign": _slice_assign,
    "list-index": _list_index,
    "sibling": _through_sibling,
}

READS: dict[str, Read] = {
    # The value the call returns, straight from the view.
    "returned": lambda v: v,
    "consumed": lambda v: v * 1.0,
}


def _case(view: View, write: Write, read: Read) -> Fn:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2
        v = view(y)
        write(y, x)
        return read(v)

    return fn


def _inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(0)
    first = lucid.randn(*SHAPE, device=COMPILE_DEVICE)
    second = lucid.randn(*SHAPE, device=COMPILE_DEVICE) * 7 + 3
    return first, second


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _agrees_with_eager(fn: Fn) -> lucid.compile.CompiledModule:
    """Compile ``fn``, require eager's answers on two inputs, return the wrapper."""
    first, second = _inputs()
    with lucid.no_grad():
        want_first, want_second = fn(first), fn(second)
        assert _gap(want_first, want_second) > 1e-3, "the probe is blind: eager repeats itself"
        compiled = lucid.compile(fn)
        got_first, got_second = compiled(first), compiled(second)
    scale = 1.0 + float(want_second.abs().max().item())
    assert _gap(got_first, want_first) <= 1e-5 * scale
    assert _gap(got_second, want_second) <= 1e-5 * scale
    assert _gap(got_first, got_second) > 1e-3, "the compiled call repeats its first answer"
    return compiled


@needs_metal
@pytest.mark.parametrize("read", READS)
@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("view", VIEWS)
def test_a_view_read_after_a_write_answers_what_eager_answers(
    view: str, write: str, read: str
) -> None:
    _agrees_with_eager(_case(VIEWS[view], WRITES[write], READS[read]))


def _taken_after(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    y.mul_(2.0)
    return y[1]


def _write_through_then_read_the_base(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    sibling = y[1:3]
    sibling.mul_(3.0)
    return y


def _temporary_view(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    # ``y[1]`` is a view the write reaches, and nothing reads it afterwards.
    row = y[1] * 2.0
    y.mul_(2.0)
    return y * row.sum()


def _disjoint_sibling(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    kept = y[:2]
    other = y[2:]
    other.mul_(3.0)
    return kept


def _accumulate(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    for _ in range(3):
        y.add_(1.0)
    return y


def _setitem_with_a_temporary_read(y: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    y[1] = y[1] + x[0]
    return y


CONTROLS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "view-taken-after": _taken_after,
    "write-through-a-view-read-the-base": _write_through_then_read_the_base,
    "temporary-view": _temporary_view,
    "disjoint-sibling": _disjoint_sibling,
    "add-in-a-loop": _accumulate,
    "setitem-reading-a-temporary": _setitem_with_a_temporary_read,
}


@needs_metal
@pytest.mark.parametrize("name", CONTROLS)
def test_a_write_nothing_stale_reads_keeps_compiling(name: str) -> None:
    body = CONTROLS[name]

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        return body(x * 2, x)

    compiled = _agrees_with_eager(fn)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], f"{name} no longer compiles"


@needs_metal
@pytest.mark.parametrize("view", ["index", "slice", "transpose"])
def test_reading_a_stale_view_runs_the_call_eager(view: str) -> None:
    """The three the issue was filed with: the view is not the whole buffer, so
    the recording cannot follow the write and the signature goes to eager."""
    compiled = _agrees_with_eager(_case(VIEWS[view], WRITES["mul_"], READS["returned"]))
    assert compiled.cache_info()["eager_only"]


@needs_metal
def test_reading_a_returned_stale_view_marks_the_trace() -> None:
    """The harness reads the output ids after the trace; a stale view must not
    come back as an id (or as None, which it takes for a value that flows
    out untouched).  Binding an argument to its id is ``lookup_id``, which
    only answers."""
    first, _second = _inputs()
    fn = _case(VIEWS["index"], WRITES["mul_"], READS["returned"])
    with lucid.no_grad(), lucid.compile._tracing(watch=[first]) as tracer:
        out = fn(first)
    assert not tracer.graph.unsupported
    assert tracer.lookup_id(out._impl) is not None
    assert not tracer.graph.unsupported, "naming an id is not a read"
    tracer.read_id(out._impl)
    assert "after an in-place write" in tracer.graph.unsupported


class _Apply(nn.Module):
    def __init__(self, fn: Fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


@pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)
@pytest.mark.parametrize("read", READS)
@pytest.mark.parametrize("view", ["index", "slice", "transpose", "reshape"])
def test_export_refuses_a_stale_view_it_would_have_frozen(
    view: str, read: str, tmp_path: object
) -> None:
    """A CPU trace has the same defect through the same hook: no view of a
    CPU family is re-derived, and the package held the value from before the
    write.  Export raises instead; nothing is written, so nothing is loaded."""
    model = _Apply(_case(VIEWS[view], WRITES["mul_"], READS[read])).eval()
    lucid.manual_seed(0)
    with pytest.raises(ValueError, match="after an in-place write"):
        cml.export(model, lucid.randn(*SHAPE), f"{tmp_path}/stale.mlpackage")


def _put_slice(z: lucid.Tensor, v: lucid.Tensor) -> None:
    z[:] = v


def _put_ellipsis(z: lucid.Tensor, v: lucid.Tensor) -> None:
    z[...] = v


def _put_copy(z: lucid.Tensor, v: lucid.Tensor) -> None:
    z.copy_(v)


PUTS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], None]] = {
    "slice": _put_slice,
    "ellipsis": _put_ellipsis,
    "copy_": _put_copy,
}


@needs_metal
@pytest.mark.parametrize("put", PUTS)
@pytest.mark.parametrize("view", VIEWS)
def test_a_stale_view_copied_into_another_tensor_is_a_read(view: str, put: str) -> None:
    """``z[:] = v`` reads ``v`` without any op: the write records ``v``'s id as
    ``z``'s value, which was the value from before the write."""
    make, write = VIEWS[view], PUTS[put]

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2
        v = make(y)
        y.mul_(2.0)
        z = make(x) * 0.0
        write(z, v)
        return z

    _agrees_with_eager(fn)


def _zeros_on_device() -> lucid.Tensor:
    return lucid.zeros(*SHAPE).to(COMPILE_DEVICE)


def _read_sum(x: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
    return x[0] * v.sum()


def _read_slice(x: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
    return v[1:3] + 0.0 * x[0][1:3]


def _read_abs(x: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
    return v.abs()


def _read_clone(x: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
    return v.clone()


MADE_READS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "sum": _read_sum,
    "slice": _read_slice,
    "abs": _read_abs,
    "clone": _read_clone,
}


@needs_metal
@pytest.mark.parametrize("read", MADE_READS)
def test_a_view_of_a_constant_the_trace_made_follows_a_write_that_reads_the_input(
    read: str,
) -> None:
    """The base is built inside the call, so no op has produced it and the
    tracer folds what reads it.  A write that depends on the input makes the
    view's later values depend on it too; folding it read the trace-time ones."""
    body = MADE_READS[read]

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = _zeros_on_device()
        v = y[1]
        y.add_(x)
        return body(x, v)

    _agrees_with_eager(fn)


@needs_metal
def test_a_constant_write_to_a_constant_keeps_compiling() -> None:
    """The write puts another constant into the base: the view reads a constant
    either way, and the call is still one graph."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = _zeros_on_device()
        v = y[1]
        y.add_(lucid.ones(*SHAPE).to(COMPILE_DEVICE))
        return x * v.sum()

    compiled = _agrees_with_eager(fn)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]


@needs_metal
@pytest.mark.parametrize("constant", [False, True])
def test_a_view_of_a_constant_read_before_the_write_keeps_that_value(constant: bool) -> None:
    """``a`` reads the view while the base is still zeros; the write changes
    the bytes the view's feed points at, so the feed keeps a copy."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = _zeros_on_device()
        v = y[1]
        a = v + x[0]
        y.add_(lucid.ones(*SHAPE).to(COMPILE_DEVICE) if constant else x)
        return a

    _agrees_with_eager(fn)


class _OverlappingViews(nn.Module):
    """A buffer, and two views of it the call holds and writes through."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("vec", lucid.zeros(4))
        self.held: tuple[lucid.Tensor, ...] = ()

    def hold(self) -> None:
        self.held = (self.vec[:2], self.vec[1:3])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        head, mid = self.held
        head.add_(1.0)
        mid.add_(1.0)
        return x * 1.0


@needs_metal
def test_overlapping_buffers_written_in_one_call_end_as_eager_ends() -> None:
    """The call writes ``head`` (``vec[:2]``) and then ``mid`` (``vec[1:3]``),
    which overlap: ``mid``'s copy from before its write already holds
    ``head``'s, so putting the buffers back for the next run has to start
    from the last one."""
    eager, compiled_model = (
        _OverlappingViews().to(COMPILE_DEVICE),
        _OverlappingViews().to(COMPILE_DEVICE),
    )
    eager.hold()
    compiled_model.hold()
    compiled = lucid.compile(compiled_model)
    x = lucid.ones(2, device=COMPILE_DEVICE)
    with lucid.no_grad():
        for _ in range(3):
            eager(x)
            compiled(x)
            assert compiled_model.vec.tolist() == eager.vec.tolist()


class _RunningStatsView(nn.Module):
    """A view of batch norm's running mean, read after the layer updates it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)
        self.held: tuple[lucid.Tensor, ...] = ()

    def hold(self) -> None:
        self.held = (self.bn.running_mean[:2],)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.bn(self.lin(x)) + self.held[0].sum()


@needs_metal
def test_a_view_of_batch_norm_running_stats_follows_the_update() -> None:
    """Batch norm's kernel leaves the buffer's bytes alone and hands the new
    values out, so no write hook sees it: the view read the previous step's
    mean, one step behind eager."""

    def build() -> _RunningStatsView:
        lucid.manual_seed(0)
        model = _RunningStatsView().to(COMPILE_DEVICE)
        model.hold()
        return model

    eager, stepped = build(), build()
    step = lucid.compile.make_step(stepped, lambda out: (out * out).mean())
    lucid.manual_seed(1)
    for _ in range(3):
        x = lucid.randn(8, 4, device=COMPILE_DEVICE)
        want = float(((eager(x) ** 2).mean()).item())
        got = float(step(x).item())
        assert abs(got - want) <= 1e-4 * (1.0 + abs(want))


def _writes_a_view_of_its_input(x: lucid.Tensor) -> lucid.Tensor:
    a = x * 2
    x[:1].add_(1.0)
    return a


@pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)
def test_binding_an_input_the_forward_wrote_through_a_view_is_not_a_read() -> None:
    """``x`` is an input; writing ``x[:1]`` leaves ``x`` itself stale, and
    naming its id for the program's input is not a read of its value.  It
    traced before and must still."""
    model = _Apply(_writes_a_view_of_its_input).eval()
    lucid.manual_seed(0)
    _graph, _feeds, inputs, outputs, _values = cml_build.trace(model, lucid.randn(*SHAPE))
    assert len(inputs) == 1 and len(outputs) == 1


@pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)
@pytest.mark.parametrize("put", PUTS)
def test_export_refuses_a_stale_view_copied_into_the_result(put: str, tmp_path: object) -> None:
    write = PUTS[put]

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2
        v = y[1]
        y.mul_(2.0)
        z = x[1] * 0.0
        write(z, v)
        return z

    lucid.manual_seed(0)
    with pytest.raises(ValueError, match="after an in-place write"):
        cml.export(_Apply(fn).eval(), lucid.randn(*SHAPE), f"{tmp_path}/stale.mlpackage")


@needs_metal
def test_a_batch_norm_update_is_never_a_constant_write() -> None:
    """The layer is built inside the call, so its running mean is a constant the
    trace made; an earlier constant write (``k.add_(ones)``) must not leave the
    update of that mean looking like one.  The update's values come from the
    input."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        k = lucid.zeros(4).to(COMPILE_DEVICE)
        k.add_(lucid.ones(4).to(COMPILE_DEVICE))
        bn = nn.BatchNorm1d(4).to(COMPILE_DEVICE)
        view = bn.running_mean[:2]
        y = bn(x)
        return y * view.sum()

    _agrees_with_eager(fn)


# ---------------------------------------------------------------------------
# Every entry point that names a value the call returns or keeps reads it with
# ``read_id``.  Each test hands one of them a stale view where it looks —
# the loss, the result, a cache buffer, an optimizer's new parameter — so that
# putting ``lookup_id`` back at that site fails here.  The entry points that
# fall back to eager must answer what eager does; the ones that have no eager
# path (``compiled_step``, ``compile_optimizer``) must refuse.
# ---------------------------------------------------------------------------


def _stale_loss(out: lucid.Tensor, *rest: lucid.Tensor) -> lucid.Tensor:
    """A loss that is a view of ``y``, read after ``y`` is doubled in place."""
    y = (out * out).reshape(-1)
    loss = y[0]
    y.mul_(2.0)
    return loss


def _twin_models(build: Callable[[], nn.Module]) -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    first = build().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    return first, build().to(COMPILE_DEVICE)


def _linear() -> nn.Module:
    return nn.Linear(4, 2)


def _batch() -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randn(3, 4, device=COMPILE_DEVICE)


@needs_metal
def test_compiled_step_refuses_a_loss_that_is_a_stale_view() -> None:
    """It has no eager path: a loss named by its id from before the write
    would train on half the loss eager computes."""
    with pytest.raises(RuntimeError, match="after an in-place write"):
        lucid.compile.compiled_step(_linear().to(COMPILE_DEVICE), _batch(), _stale_loss)


@needs_metal
def test_make_step_follows_eager_when_the_loss_is_a_stale_view() -> None:
    model, ref = _twin_models(_linear)
    step = lucid.compile.make_step(model, _stale_loss)
    x = _batch()
    for _ in range(2):
        loss = step(x)
        loss.backward()
        want = _stale_loss(ref(x))
        want.backward()
        assert abs(float(loss.item()) - float(want.item())) <= 1e-5 * (
            1.0 + abs(float(want.item()))
        )
        assert _gap(model.weight.grad, ref.weight.grad) <= 1e-5


@needs_metal
def test_fused_step_follows_eager_when_the_loss_is_a_stale_view() -> None:
    model, ref = _twin_models(_linear)
    step = lucid.compile.fused_step(model, _stale_loss, optim.SGD(model.parameters(), lr=0.1))
    ref_opt = optim.SGD(ref.parameters(), lr=0.1)
    x = _batch()
    for _ in range(2):
        loss = step(x)
        ref_opt.zero_grad()
        want = _stale_loss(ref(x))
        want.backward()
        ref_opt.step()
        assert abs(float(loss.item()) - float(want.item())) <= 1e-5 * (
            1.0 + abs(float(want.item()))
        )
        assert _gap(model.weight, ref.weight) <= 1e-5


class _StaleOut(nn.Module):
    """A segment whose output is a view of a tensor it doubles afterwards."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(4))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = x * self.w
        out = y[:, :2]
        y.mul_(2.0)
        return out


def _two_segments() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), _StaleOut())


@needs_metal
def test_a_segmented_step_follows_eager_when_a_segment_returns_a_stale_view() -> None:
    model, ref = _twin_models(_two_segments)
    step = lucid.compile.make_step(model, lambda out: (out * out).mean(), segments=2)
    x = _batch()
    for _ in range(2):
        loss = step(x)
        loss.backward()
        want = (ref(x) ** 2).mean()
        want.backward()
        assert abs(float(loss.item()) - float(want.item())) <= 1e-5 * (
            1.0 + abs(float(want.item()))
        )
        assert _gap(model[0].weight.grad, ref[0].weight.grad) <= 1e-5


def _decode_forward(
    cache: StaticCache, stale: str
) -> Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]:
    """A one-layer decode step whose result (``"output"``) or key buffer
    (``"cache"``) is a view of a tensor written in place afterwards.  The
    buffer is square in its last two axes, so ``y.mT`` has its shape and is
    still not the whole of ``y``."""

    def forward(hidden: lucid.Tensor, cache_position: lucid.Tensor) -> lucid.Tensor:
        key = (hidden * 1.0).reshape(1, 1, 1, 4)
        keys, _ = cache.update(key, key, 0, {"cache_position": cache_position})
        if stale == "output":
            y = keys * 2.0
            out = y[:, :, :2]
            y.mul_(2.0)
            return out
        y = cache.key_cache[0] * 1.0
        view = y.mT
        y.mul_(2.0)
        cache.key_cache[0] = view
        return hidden * 1.0

    return forward


@needs_metal
@pytest.mark.parametrize("stale", ["output", "cache"])
def test_a_compiled_decode_follows_eager_when_it_returns_a_stale_view(stale: str) -> None:
    cache, shadow = StaticCache(4), StaticCache(4)
    step = lucid.compile.compiled_decode_step(_decode_forward(cache, stale), cache)
    eager = _decode_forward(shadow, stale)
    with lucid.no_grad():
        for position in range(3):
            hidden = lucid.ones(1, 1, 4, device=COMPILE_DEVICE) * (position + 1)
            at = lucid.tensor([position], device=COMPILE_DEVICE).long()
            assert step(hidden, at).tolist() == eager(hidden, at).tolist()
            assert cache.key_cache[0].tolist() == shadow.key_cache[0].tolist()


@needs_metal
def test_a_compiled_optimizer_refuses_an_update_that_is_a_stale_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled optimizer has no eager path either.  Its update is a hook a
    subclass writes; one that hands back a view of what it wrote in place
    afterwards would have the parameter take the value from before."""

    def update(
        self: _CompiledSGD,
        p: lucid.Tensor,
        g: lucid.Tensor,
        state: dict[str, lucid.Tensor],
        flags: object,
        gs: Callable[[str], lucid.Tensor],
        ps: Callable[[str], lucid.Tensor],
    ) -> tuple[lucid.Tensor, dict[str, lucid.Tensor]]:
        y = p - gs("lr") * g
        new = y[0:1]
        y.mul_(2.0)
        return new, {}

    monkeypatch.setattr(_CompiledSGD, "_update", update)
    model = _linear().to(COMPILE_DEVICE)
    compiled = lucid.compile.compile_optimizer(optim.SGD(model.parameters(), lr=0.1))
    for p in model.parameters():
        p.grad = lucid.ones_like(p)
    with pytest.raises(RuntimeError, match="after an in-place write"):
        compiled.step()


# ---------------------------------------------------------------------------
# The same family of bug, found while reviewing this fix.  The class is covered
# in ``test_where_a_write_lands.py``; these are the cases it was filed with.
# ---------------------------------------------------------------------------


@needs_metal
def test_a_tensor_built_from_numbers_and_written_in_place_starts_over_each_call() -> None:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = lucid.tensor([[1.0, 2.0, 3.0, 4.0]] * 3, device=COMPILE_DEVICE)
        y.add_(x)
        return y * 2.0

    _agrees_with_eager(fn)


class _ViewBuffer(nn.Module):
    """A buffer that is itself a view, written in place."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("vec", lucid.zeros(5, device=COMPILE_DEVICE)[1:])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.vec[0:2].add_(1.0)
        return x * 1.0


class _ComplexPartBuffer(nn.Module):
    """A complex buffer whose real part the call writes."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("z", lucid.zeros(4, dtype=lucid.complex64, device=COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.z.real.add_(1.0)
        return x * 1.0


def _buffers_after_three_calls(
    model: nn.Module, name: str, *, compiled: bool
) -> list[list[complex]]:
    run = lucid.compile(model) if compiled else model
    x = lucid.ones(2, device=COMPILE_DEVICE)
    seen: list[list[complex]] = []
    with lucid.no_grad():
        for _ in range(3):
            run(x)
            seen.append(getattr(model, name).tolist())
    return seen


@needs_metal
@pytest.mark.parametrize("kind", [_ViewBuffer, _ComplexPartBuffer], ids=["view", "real-part"])
def test_a_write_to_a_buffer_that_is_a_view_is_carried_back(kind: type[nn.Module]) -> None:
    name = "vec" if kind is _ViewBuffer else "z"
    want = _buffers_after_three_calls(kind().to(COMPILE_DEVICE), name, compiled=False)
    got = _buffers_after_three_calls(kind().to(COMPILE_DEVICE), name, compiled=True)
    assert got == want


@pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)
def test_export_leaves_the_example_as_it_found_it_when_the_forward_writes_a_view() -> None:
    lucid.manual_seed(0)
    example = lucid.randn(*SHAPE)
    before = example.tolist()
    cml_build.trace(_Apply(_writes_a_view_of_its_input).eval(), example)
    assert example.tolist() == before


class _SharedAlias(nn.Module):
    """Two aliases of one Metal shared buffer; the call writes one, returns the other."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("m", lucid.metal.shared_tensor(SHAPE).to(COMPILE_DEVICE))
        self.alias = self.m.to("cpu").to(COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        self.m.add_(x.mean())
        return self.alias * 1.0


@needs_metal
def test_an_alias_of_a_shared_buffer_reads_what_the_call_wrote() -> None:
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)

    def outputs(compiled: bool) -> list[float]:
        model = _SharedAlias()
        run = lucid.compile(model) if compiled else model
        with lucid.no_grad():
            return [float(run(x).max().item()) for _ in range(3)]

    assert outputs(True) == outputs(False)
