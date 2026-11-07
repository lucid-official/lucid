"""A trace aliases through ``contiguous`` as eager does, for each call it stands for.

``contiguous`` answers its own input when that is contiguous, and a copy when it
is not, so whether a later write reaches the result depends on where the input's
elements sit.  A trace used to copy every time: ``v = y.contiguous(); y.mul_(2)``
read ``v`` from before the write, a plausible number with no fallback (LCD-890).
``test_view_before_inplace_write`` covers that row against every write.

Answering as eager does makes the answer depend on the trace's input, and two
calls can share a key and lay their input out differently:

* a strided input and a contiguous one of the same shape;
* two strided inputs, whose views are contiguous in different places;
* under ``dynamic=True``, a batch of one and any other, since a size-one axis
  makes a view contiguous that is not at another batch.

So the key tells them apart, and these tests call each pair in both orders and
compare the result and what became of the input with eager.  The controls hold
the other side: a call with no such difference keeps sharing one executable.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.coreml import _build as cml_build
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
        return True
    except RuntimeError:
        return False


needs_metal = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Fn = Callable[[lucid.Tensor], object]


def _grid(rows: int, cols: int) -> lucid.Tensor:
    return (lucid.arange(float(rows * cols)) + 1.0).reshape(rows, cols).to(COMPILE_DEVICE)


# Every one is (3, 4): one key but for their layout.
LAYOUTS: dict[str, Callable[[], lucid.Tensor]] = {
    "whole": lambda: _grid(3, 4),
    "sliced": lambda: _grid(5, 4)[2:],
    "every-other-row": lambda: _grid(6, 4)[::2],
    "every-other-column": lambda: _grid(3, 8)[:, ::2],
    "transposed": lambda: _grid(4, 3).mT,
}


def _write_the_result(x: lucid.Tensor) -> lucid.Tensor:
    v = x.contiguous()
    v.mul_(2.0)
    return x * 1.0


def _write_a_view_of_the_result(x: lucid.Tensor) -> lucid.Tensor:
    v = x.contiguous()
    v[0].mul_(2.0)
    return x * 1.0


def _write_the_input(x: lucid.Tensor) -> lucid.Tensor:
    v = x.contiguous()
    x.mul_(2.0)
    return v * 1.0


def _write_a_contiguous_row(x: lucid.Tensor) -> lucid.Tensor:
    # A row is contiguous for ``every-other-row`` and not for ``every-other-column``.
    v = x[0].contiguous()
    v.mul_(2.0)
    return x * 1.0


def _write_a_contiguous_block(x: lucid.Tensor) -> lucid.Tensor:
    y = x[:2]
    v = y.contiguous()
    v.mul_(2.0)
    return y * 1.0


def _no_write(x: lucid.Tensor) -> lucid.Tensor:
    return x.contiguous().sum() + x[0].contiguous().sum()


def _branch_on_layout(x: lucid.Tensor) -> lucid.Tensor:
    return x * (2.0 if x.is_contiguous() else 3.0)


BODIES: dict[str, Fn] = {
    "write-the-result": _write_the_result,
    "write-a-view-of-the-result": _write_a_view_of_the_result,
    "write-the-input": _write_the_input,
    "write-a-contiguous-row": _write_a_contiguous_row,
    "write-a-contiguous-block": _write_a_contiguous_block,
    "no-write": _no_write,
    "branch-on-layout": _branch_on_layout,
}


def _ran(fn: Fn, x: lucid.Tensor) -> tuple[object, object]:
    out = fn(x)
    assert isinstance(out, lucid.Tensor)
    return out.tolist(), x.tolist()


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("second", LAYOUTS)
@pytest.mark.parametrize("first", LAYOUTS)
@pytest.mark.parametrize("body", BODIES)
def test_a_later_call_with_another_layout_answers_as_eager(
    body: str, first: str, second: str
) -> None:
    fn = BODIES[body]
    want = [_ran(fn, LAYOUTS[name]()) for name in (first, second, first)]
    compiled = lucid.compile(fn)
    got = [_ran(compiled, LAYOUTS[name]()) for name in (first, second, first)]
    assert got == want


@needs_metal
@pytest.mark.usefixtures("inference")
def test_calls_that_lay_out_their_input_alike_share_one_executable() -> None:
    """A slice begins part-way into its buffer and lays out its views as the whole does."""
    compiled = lucid.compile(_no_write)
    for make in (LAYOUTS["whole"], LAYOUTS["sliced"], LAYOUTS["whole"]):
        assert compiled(make()).item() == _no_write(make()).item()
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]


@needs_metal
@pytest.mark.usefixtures("inference")
def test_each_strided_layout_has_its_own_entry() -> None:
    compiled = lucid.compile(_no_write)
    for make in (LAYOUTS["every-other-row"], LAYOUTS["every-other-column"]):
        assert compiled(make()).item() == _no_write(make()).item()
    assert compiled.cache_info()["entries"] == 2


def _read_before_the_write(x: lucid.Tensor) -> lucid.Tensor:
    y = x * 2.0
    a = y.contiguous() + x
    y.mul_(3.0)
    return a + y


CONTROLS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "of-the-input": lambda x: x.contiguous() * 2.0,
    "of-a-result": lambda x: (x * 2.0).contiguous() + x,
    "of-a-slice": lambda x: x[1:].contiguous().sum() * x,
    "of-a-strided-view": lambda x: x[:, ::2].contiguous().sum() * x,
    "of-a-transpose": lambda x: x.mT.contiguous().sum() * x,
    "free-function": lambda x: lucid.contiguous(x * 2.0) + x,
    "read-before-the-write": _read_before_the_write,
    "eval-dropout": lambda x: nn.functional.dropout(x * 2.0, 0.5, False) + x,
    "no-dropout": lambda x: nn.functional.dropout(x * 2.0, 0.0, True) + x,
    "eval-channel-dropout": lambda x: nn.functional.dropout1d(x * 2.0, 0.5, False) + x,
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", CONTROLS)
def test_a_contiguous_nothing_writes_keeps_compiling(name: str) -> None:
    fn = CONTROLS[name]
    compiled = lucid.compile(fn)
    for scale in (1.0, 7.0):
        x = _grid(3, 4) * scale
        assert compiled(x).tolist() == fn(x).tolist()
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], f"{name} no longer compiles"


@needs_metal
@pytest.mark.usefixtures("inference")
def test_the_input_handed_back_among_the_results_is_the_input() -> None:
    def fn(x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        return x.contiguous(), x * 2.0

    compiled = lucid.compile(fn)
    x = _grid(3, 4)
    kept, doubled = compiled(x)
    assert kept.tolist() == x.tolist()
    assert doubled.tolist() == (x * 2.0).tolist()


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_call_that_is_only_contiguous_answers_as_eager() -> None:
    """Its trace records no op, as ``lambda x: x`` does, so the signature runs eager."""
    compiled = lucid.compile(lambda x: x.contiguous())
    x = _grid(3, 4)
    assert compiled(x).tolist() == x.tolist()
    assert compiled.cache_info()["eager_only"]


class _WritesItsBuffer(nn.Module):
    """A write through ``contiguous`` reaches the buffer only if that is contiguous."""

    def __init__(self, state: lucid.Tensor) -> None:
        super().__init__()
        self.register_buffer("state", state)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        v = self.state.contiguous()
        v.add_(1.0)
        return x + self.state


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize(
    "state",
    [
        lambda: lucid.zeros(3, 4).to(COMPILE_DEVICE),
        lambda: lucid.zeros(3, 8).to(COMPILE_DEVICE)[:, ::2],
    ],
    ids=["contiguous", "strided"],
)
def test_a_write_through_contiguous_of_a_buffer_ends_as_eager_ends(
    state: Callable[[], lucid.Tensor],
) -> None:
    eager, traced = _WritesItsBuffer(state()), _WritesItsBuffer(state())
    compiled = lucid.compile(traced)
    x = _grid(3, 4)
    for _ in range(3):
        assert compiled(x).tolist() == eager(x).tolist()
        assert traced.state.tolist() == eager.state.tolist()


# ---------------------------------------------------------------------------
# ``dynamic=True``: a size-one batch axis makes a view contiguous, and a batch of
# one is its own key.
# ---------------------------------------------------------------------------


def _read_a_block_after_the_write(x: lucid.Tensor) -> lucid.Tensor:
    y = x * 2.0
    v = y[:, :2].contiguous()
    y.mul_(2.0)
    return v * 1.0


def _write_a_block_read_its_base(x: lucid.Tensor) -> lucid.Tensor:
    y = x * 2.0
    v = y[:, :2].contiguous()
    v.mul_(2.0)
    return y * 1.0


def _squeeze_every_axis(x: lucid.Tensor) -> lucid.Tensor:
    return (x * 2.0).squeeze() * 1.0


BATCH_BODIES: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "read-after-the-write": _read_a_block_after_the_write,
    "write-read-the-base": _write_a_block_read_its_base,
    "squeeze": _squeeze_every_axis,
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("order", [(3, 1, 3, 5), (1, 3, 1, 5), (3, 5, 1)], ids=str)
@pytest.mark.parametrize("body", BATCH_BODIES)
def test_a_batch_of_one_answers_as_eager_under_a_dynamic_compile(
    body: str, order: tuple[int, ...]
) -> None:
    fn = BATCH_BODIES[body]
    compiled = lucid.compile(fn, dynamic=True)
    for batch in order:
        x = _grid(batch, 4)
        assert compiled(x).tolist() == fn(x).tolist(), f"batch {batch}"


@needs_metal
@pytest.mark.usefixtures("inference")
def test_the_other_batches_still_share_one_dynamic_executable() -> None:
    fn = CONTROLS["of-the-input"]
    compiled = lucid.compile(fn, dynamic=True)
    for batch in (3, 5, 8):
        x = _grid(batch, 4)
        assert compiled(x).tolist() == fn(x).tolist()
    assert compiled.cache_info()["entries"] == 1


# ---------------------------------------------------------------------------
# The other entry points key through the same signature and trace the same way.
# ---------------------------------------------------------------------------


class _Writes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = self.lin(x)
        v = y.contiguous()
        y.mul_(2.0)
        return v + y


class _Plain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.lin(x.contiguous()).contiguous()


def _twins(kind: type[nn.Module]) -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    first = kind().to(COMPILE_DEVICE)
    lucid.manual_seed(0)
    return first, kind().to(COMPILE_DEVICE)


def _mean_square(out: lucid.Tensor) -> lucid.Tensor:
    return (out * out).mean()


@needs_metal
@pytest.mark.parametrize(("kind", "steps"), [(_Writes, 1), (_Plain, 2)])
def test_a_step_trains_through_contiguous_as_eager_does(kind: type[nn.Module], steps: int) -> None:
    model, ref = _twins(kind)
    step = lucid.compile.make_step(model, _mean_square)
    x = _grid(3, 4) / 10.0
    for _ in range(steps):
        for p in (*model.parameters(), *ref.parameters()):
            p.grad = None
        loss = step(x)
        loss.backward()
        want = _mean_square(ref(x))
        want.backward()
        assert abs(float(loss.item()) - float(want.item())) <= 1e-4 * (
            1.0 + abs(float(want.item()))
        )
        assert float((model.lin.weight.grad - ref.lin.weight.grad).abs().max().item()) <= 1e-3


@needs_metal
def test_a_fused_step_trains_through_contiguous_as_eager_does() -> None:
    model, ref = _twins(_Plain)
    step = lucid.compile.fused_step(model, _mean_square, optim.SGD(model.parameters(), lr=0.01))
    ref_opt = optim.SGD(ref.parameters(), lr=0.01)
    x = _grid(3, 4) / 10.0
    for _ in range(2):
        loss = step(x)
        ref_opt.zero_grad()
        want = _mean_square(ref(x))
        want.backward()
        ref_opt.step()
        assert abs(float(loss.item()) - float(want.item())) <= 1e-4 * (
            1.0 + abs(float(want.item()))
        )


class _Apply(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


def _read_the_alias_after_the_write(x: lucid.Tensor) -> lucid.Tensor:
    y = x * 2.0
    v = y.contiguous()
    y.mul_(2.0)
    return v


coreml_writer = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


@coreml_writer
def test_an_export_reads_the_alias_after_the_write() -> None:
    """The result is the tensor written, so its value is the one after the write,
    which is not the stale view :mod:`.test_view_before_inplace_write` refuses."""
    model = _Apply(_read_the_alias_after_the_write).eval()
    graph, _feeds, _inputs, outputs, _values = cml_build.trace(model, _cpu_grid())
    assert not graph.unsupported
    _name, output_id, _tensor = outputs[0]
    assert graph.ops[-1].outputs[0].id == output_id
    assert graph.ops[-1].name == "mul"


@coreml_writer
def test_an_export_of_a_copy_keeps_the_value_from_before_the_write() -> None:
    """A transpose is not contiguous: the copy is its own value, not refused as stale."""

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2.0
        v = y.mT.contiguous()
        y.mul_(2.0)
        return v

    graph, _feeds, _inputs, outputs, _values = cml_build.trace(_Apply(fn).eval(), _cpu_grid())
    assert not graph.unsupported
    _name, output_id, _tensor = outputs[0]
    made = next(op for op in graph.ops if op.outputs[0].id == output_id)
    assert made.name == "contiguous"


def _cpu_grid() -> lucid.Tensor:
    return (lucid.arange(12.0) + 1.0).reshape(3, 4)
