"""Graph-mode backward reads a tensor as the forward saw it, not as a later in-place write left it.

Eager backward reads what a node saved by value.  Graph-mode backward —
``backward(create_graph=True)``, ``autograd.grad(..., create_graph=True)`` —
reads through the tensors the node was handed, values and graph position
both.  Two ways those had changed by the time it read them:

* The in-place op's own node was handed its destination itself when only
  the *other* operand required grad.  The write then rebound that tensor,
  so the node read the result for the operand, on both devices:

      buf.mul_(v); grad(buf.sum(), v, create_graph=True)  ->  buf * v, not buf

  The node also became the grad_fn of the tensor it held — a reference
  cycle that kept both alive after the last use, two buffers per call.

* A node that saved a tensor before an in-place write into it.  A write
  that records a graph gave the tensor a new grad_fn and left its version
  alone, so nothing refused the read.  The CPU refuses such a write
  outright (the node holds its buffer); on metal it went through:

      out = (w * buf).sum(); buf.add_(v)
      grad(out, w, create_graph=True)  ->  buf + v, not buf

  Such a write moves the version now, and that backward raises
  ``VersionMismatch``, as the reference's does — which is also what lets a
  custom ``Function`` see a write into a tensor it saved.

The in-place op's own node keeps a snapshot with a version count of its
own, so it never refuses itself, and graph mode computes what eager
backward computes.  Checks against the reference live in
``lucid/test/parity/autograd/test_create_graph_inplace_parity.py``.
"""

import gc
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.autograd
from lucid._C import engine as _C_engine
from lucid.autograd.graph import allow_mutation_on_saved_tensors

B = np.array([1.0, 2.0, 3.0])
V = np.array([2.0, 3.0, 4.0])

# The first and second derivative of ``op(b, v)`` with respect to ``v``,
# at the ``b`` the forward saw.
DERIVATIVES: dict[str, tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]] = {
    "mul_": (lambda b, v: b, lambda b, v: np.zeros_like(b)),
    "add_": (lambda b, v: np.ones_like(b), lambda b, v: np.zeros_like(b)),
    "sub_": (lambda b, v: -np.ones_like(b), lambda b, v: np.zeros_like(b)),
    "div_": (lambda b, v: -b / v**2, lambda b, v: 2.0 * b / v**3),
}
OPS = list(DERIVATIVES)


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.detach().numpy(), dtype=np.float64)


def _second(g: lucid.Tensor, wrt: lucid.Tensor) -> np.ndarray:
    """``d(sum g)/d wrt``, zero when ``g`` carries no dependence on it."""
    if not g.requires_grad:
        return np.zeros(wrt.shape)
    (h,) = lucid.autograd.grad(g.sum(), wrt, allow_unused=True)
    return np.zeros(wrt.shape) if h is None else _np(h)


def _written(op: str, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    """A destination without grad, written with an operand that has it."""
    buf = lucid.tensor(B.tolist(), device=device)
    v = lucid.tensor(V.tolist(), requires_grad=True, device=device)
    getattr(buf, op)(v)
    return buf, v


# ── the in-place op's own node ────────────────────────────────────────────────


@pytest.mark.parametrize("op", OPS)
def test_graph_mode_reads_the_destination_as_it_was(op: str, device: str) -> None:
    first, second = DERIVATIVES[op]
    buf, v = _written(op, device)
    (g,) = lucid.autograd.grad(buf.sum(), v, create_graph=True)
    np.testing.assert_allclose(_np(g), first(B, V), rtol=1e-5)
    np.testing.assert_allclose(_second(g, v), second(B, V), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("op", OPS)
def test_graph_mode_agrees_with_eager_backward(op: str, device: str) -> None:
    buf, v = _written(op, device)
    (g,) = lucid.autograd.grad(buf.sum(), v, create_graph=True)
    buf, v = _written(op, device)
    buf.sum().backward()
    assert v.grad is not None
    np.testing.assert_allclose(_np(g), _np(v.grad), rtol=1e-5)


def test_backward_with_create_graph_reads_the_destination_as_it_was(device: str) -> None:
    # The other graph-mode entry point: ``backward`` rather than ``grad``.
    buf, v = _written("mul_", device)
    buf.sum().backward(create_graph=True)
    assert v.grad is not None
    np.testing.assert_allclose(_np(v.grad), B, rtol=1e-5)


def test_two_writes_with_the_same_operand(device: str) -> None:
    # ``buf * v * v``: each node reads the destination as its own write
    # found it, so d/dv = 2 b v and d2/dv2 = 2 b.  Reading it as the last
    # write left it gave 30 and 136 for 6 and 16.
    buf = lucid.tensor(B.tolist(), device=device)
    v = lucid.tensor(V.tolist(), requires_grad=True, device=device)
    buf.mul_(v)
    buf.mul_(v)
    (g,) = lucid.autograd.grad(buf.sum(), v, create_graph=True)
    np.testing.assert_allclose(_np(g), 2.0 * B * V, rtol=1e-5)
    np.testing.assert_allclose(_second(g, v), 2.0 * B, rtol=1e-5)


def test_a_broadcast_operand(device: str) -> None:
    buf = lucid.ones(2, 3, device=device)
    v = lucid.tensor(V.tolist(), requires_grad=True, device=device)
    buf.mul_(v)
    buf.mul_(v)
    (g,) = lucid.autograd.grad(buf.sum(), v, create_graph=True)
    np.testing.assert_allclose(_np(g), 4.0 * V, rtol=1e-5)
    np.testing.assert_allclose(_second(g, v), np.full(3, 4.0), rtol=1e-5)


def test_a_zero_dimensional_destination(device: str) -> None:
    buf = lucid.tensor(2.0, device=device)
    v = lucid.tensor(3.0, requires_grad=True, device=device)
    buf.div_(v)
    (g,) = lucid.autograd.grad(buf, v, create_graph=True)
    (h,) = lucid.autograd.grad(g, v)
    assert g.item() == pytest.approx(-2.0 / 9.0)
    assert h.item() == pytest.approx(4.0 / 27.0)


def test_an_empty_destination(device: str) -> None:
    buf = lucid.zeros(0, device=device)
    v = lucid.zeros(0, requires_grad=True, device=device)
    buf.mul_(v)
    (g,) = lucid.autograd.grad(buf.sum(), v, create_graph=True)
    assert tuple(g.shape) == (0,)


def test_the_operand_is_the_destination(device: str) -> None:
    # ``y.mul_(y)`` saves ``y`` twice; the second slot read ``y * y``.
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    y = x * 1.0
    y.mul_(y)
    (g,) = lucid.autograd.grad(y.sum(), x, create_graph=True)
    np.testing.assert_allclose(_np(g), [2.0, 4.0], rtol=1e-5)
    np.testing.assert_allclose(_second(g, x), [2.0, 2.0], rtol=1e-5)


def test_a_retained_graph_reads_the_same_values_every_time(device: str) -> None:
    buf, v = _written("mul_", device)
    out = buf.sum()
    (g1,) = lucid.autograd.grad(out, v, create_graph=True)
    (g2,) = lucid.autograd.grad(out, v, create_graph=True)
    out.backward()
    assert v.grad is not None
    for got in (g1, g2, v.grad):
        np.testing.assert_allclose(_np(got), B, rtol=1e-5)


def test_a_unary_write_still_reads_its_input(device: str) -> None:
    x = lucid.tensor([0.5, 1.0], requires_grad=True, device=device)
    y = x * 1.0
    y.sin_()
    (g,) = lucid.autograd.grad(y.sum(), x, create_graph=True)
    np.testing.assert_allclose(_np(g), np.cos([0.5, 1.0]), rtol=1e-5)
    np.testing.assert_allclose(_second(g, x), -np.sin([0.5, 1.0]), rtol=1e-5)


def test_no_grad_write_is_untouched(device: str) -> None:
    buf = lucid.tensor(B.tolist(), device=device)
    v = lucid.tensor(V.tolist(), requires_grad=True, device=device)
    with lucid.no_grad():
        buf.mul_(v)
    np.testing.assert_allclose(_np(buf), B * V)
    assert not buf.requires_grad
    assert buf.grad_fn is None


def test_a_write_into_a_tensor_with_a_view(device_cpu_only: str) -> None:
    # The write lands in the buffer the view reads, so the node gets a copy
    # of the old values rather than a handle on that buffer.
    base = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device_cpu_only)
    view = base[:2]
    v = lucid.tensor([2.0, 2.0, 2.0, 2.0], requires_grad=True, device=device_cpu_only)
    base.mul_(v)
    np.testing.assert_allclose(_np(view), [2.0, 4.0])
    (g,) = lucid.autograd.grad(base.sum(), v, create_graph=True)
    np.testing.assert_allclose(_np(g), [1.0, 2.0, 3.0, 4.0])


_N = 1 << 14  # 64 KiB of float32: a lost buffer is far above the noise


def _destination_without_grad(device: str) -> lucid.Tensor:
    buf = lucid.ones(_N, device=device)
    buf.mul_(lucid.ones(_N, requires_grad=True, device=device))
    return buf


def _operand_is_destination(device: str) -> lucid.Tensor:
    y = lucid.ones(_N, requires_grad=True, device=device) * 1.0
    y.mul_(y)
    return y


@pytest.mark.parametrize(
    "write",
    [_destination_without_grad, _operand_is_destination],
    ids=["destination-without-grad", "operand-is-destination"],
)
def test_the_node_does_not_keep_the_tensor_it_writes_alive(
    write: Callable[[str], lucid.Tensor], device: str
) -> None:
    # Handed the destination itself, the node became the destination's
    # grad_fn while holding it: a cycle, and two buffers lost per call.
    dev = _C_engine.CPU if device == "cpu" else _C_engine.GPU
    lucid.eval(write(device))
    gc.collect()
    before = _C_engine.memory_stats(dev).current_bytes
    for _ in range(8):
        lucid.eval(write(device))
    gc.collect()
    assert _C_engine.memory_stats(dev).current_bytes - before < _N * 4


# ── a node that saved the tensor before the write ─────────────────────────────
#
# A write that records a graph moves the destination's version, as any write
# does, so a node that saved the destination refuses at backward — eager and
# graph mode alike, as the reference does.  The CPU refuses most of these at
# the write already: the node holds the buffer the write would land in.


def _recorded_writes() -> dict[str, Callable[[lucid.Tensor], object]]:
    return {
        "mul_": lambda t: t.mul_(lucid.full(t.shape, 3.0, requires_grad=True, device=t.device)),
        "add_": lambda t: t.add_(lucid.ones(t.shape, requires_grad=True, device=t.device)),
        "exp_": lambda t: t.exp_(),
        "relu_": lambda t: t.relu_(),
    }


@pytest.mark.parametrize("write", list(_recorded_writes()))
def test_a_recorded_write_moves_the_version(write: str, device: str) -> None:
    # What a node that saved the tensor — an engine node, or a custom
    # ``Function``'s ``save_for_backward`` — compares at backward.
    x = lucid.tensor([0.5, 1.0], requires_grad=True, device=device)
    y = x * 1.0
    before = y._impl.version
    _recorded_writes()[write](y)
    assert y.grad_fn is not None
    assert y._impl.version > before


def test_consecutive_writes_still_differentiate(device: str) -> None:
    # The version moves on every write, and no write's own node refuses:
    # each holds a snapshot with a count of its own.
    xs = np.array([0.5, 1.0])
    x = lucid.tensor(xs.tolist(), requires_grad=True, device=device)
    y = x * 1.0
    y.mul_(2.0)
    y.add_(1.0)
    y.exp_()
    (g,) = lucid.autograd.grad(y.sum(), x, create_graph=True)
    np.testing.assert_allclose(_np(g), 2.0 * np.exp(2.0 * xs + 1.0), rtol=1e-5)
    np.testing.assert_allclose(_second(g, x), 4.0 * np.exp(2.0 * xs + 1.0), rtol=1e-5)
    x.grad = None
    y2 = x * 1.0
    y2.mul_(2.0)
    y2.add_(1.0)
    y2.exp_()
    y2.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(_np(x.grad), 2.0 * np.exp(2.0 * xs + 1.0), rtol=1e-5)


def test_a_saved_tensor_without_grad_written_by_a_recorded_op(device: str) -> None:
    def setup() -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        w = lucid.tensor([1.0, 1.0, 1.0], requires_grad=True, device=device)
        buf = lucid.tensor(B.tolist(), device=device)
        v = lucid.tensor(V.tolist(), requires_grad=True, device=device)
        return w, buf, v, (w * buf).sum()

    w, buf, v, out = setup()
    if device == "cpu":
        with pytest.raises(_C_engine.LucidError, match="shares storage"):
            buf.add_(v)
        return
    # Metal took the write and read ``buf + v`` for ``buf`` under
    # create_graph, while eager backward read the saved ``buf``.
    buf.add_(v)
    with pytest.raises(_C_engine.VersionMismatch):
        lucid.autograd.grad(out, w, create_graph=True)
    w, buf, v, out = setup()
    buf.add_(v)
    with pytest.raises(_C_engine.VersionMismatch):
        out.backward()


def test_a_saved_tensor_with_grad_written_by_a_recorded_op(device: str) -> None:
    def setup() -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
        y = x * 2.0
        return x, y, (y * y).sum()

    x, y, z = setup()
    if device == "cpu":
        with pytest.raises(_C_engine.LucidError, match="shares storage"):
            y.mul_(3.0)
        return
    # Under create_graph metal read the written ``y`` (24x for 8x) and
    # differentiated through the write.
    y.mul_(3.0)
    with pytest.raises(_C_engine.VersionMismatch):
        lucid.autograd.grad(z, x, create_graph=True)
    x, y, z = setup()
    y.mul_(3.0)
    with pytest.raises(_C_engine.VersionMismatch):
        z.backward()


def test_a_waived_check_reads_the_values_from_before_the_write(device_gpu_only: str) -> None:
    # ``allow_mutation_on_saved_tensors`` promises the values from before
    # the write; graph mode rebuilds them from what the node saved.
    dev = device_gpu_only
    for create_graph in (True, False):
        w = lucid.tensor([1.0, 1.0, 1.0], requires_grad=True, device=dev)
        buf = lucid.tensor(B.tolist(), device=dev)
        v = lucid.tensor(V.tolist(), requires_grad=True, device=dev)
        out = (w * buf).sum()
        with allow_mutation_on_saved_tensors():
            buf.add_(v)
            (g,) = lucid.autograd.grad(out, w, create_graph=create_graph)
        np.testing.assert_allclose(_np(g), B, rtol=1e-5)


def test_a_saved_output_written_by_a_recorded_op(device: str) -> None:
    # ``sigmoid`` saves its output and checks its version at backward, so an
    # unwaived in-place write into ``y`` raises ``VersionMismatch`` on both
    # devices; under ``allow_mutation_on_saved_tensors()`` graph mode rebuilds
    # ``sigmoid``'s output from the saved pre-write storage.
    xs = np.array([1.0, 2.0])
    s = 1.0 / (1.0 + np.exp(-xs))
    x = lucid.tensor(xs.tolist(), requires_grad=True, device=device)
    y = x.sigmoid()
    y.mul_(2.0)
    with pytest.raises(_C_engine.VersionMismatch, match="sigmoid output 0"):
        lucid.autograd.grad(y.sum(), x, create_graph=True)

    x2 = lucid.tensor(xs.tolist(), requires_grad=True, device=device)
    y2 = x2.sigmoid()
    with allow_mutation_on_saved_tensors():
        y2.mul_(2.0)
        (g,) = lucid.autograd.grad(y2.sum(), x2, create_graph=True)
        h = _second(g, x2)
    np.testing.assert_allclose(_np(g), 2.0 * s * (1.0 - s), rtol=1e-5)
    np.testing.assert_allclose(h, 2.0 * s * (1.0 - s) * (1.0 - 2.0 * s), rtol=1e-4)


def test_a_write_under_no_grad_into_a_saved_output(device: str) -> None:
    # ``sigmoid`` versions its saved output, so a ``no_grad`` write into ``y``
    # raises ``VersionMismatch`` in both eager and graph mode unless waived by
    # ``allow_mutation_on_saved_tensors()``, where both modes differentiate at
    # the pre-write saved output.
    xs = np.array([1.0, 2.0])
    s = 1.0 / (1.0 + np.exp(-xs))
    for create_graph in (True, False):
        x = lucid.tensor(xs.tolist(), requires_grad=True, device=device)
        y = x.sigmoid()
        with lucid.no_grad():
            y.exp_()
        with pytest.raises(_C_engine.VersionMismatch, match="sigmoid output 0"):
            lucid.autograd.grad(y.sum(), x, create_graph=create_graph)

        x2 = lucid.tensor(xs.tolist(), requires_grad=True, device=device)
        y2 = x2.sigmoid()
        with allow_mutation_on_saved_tensors():
            with lucid.no_grad():
                y2.exp_()
            (g,) = lucid.autograd.grad(y2.sum(), x2, create_graph=create_graph)
        np.testing.assert_allclose(_np(g), s * (1.0 - s), rtol=1e-5)


# ── a write into a gradient ───────────────────────────────────────────────────
#
# A tensor read from ``.grad`` is written into its gradient's buffer.  With an
# operand that requires grad the op records a node, and the node's snapshot
# of the destination must not be one more reader of that buffer — a reader
# has the write refused.  It is a copy, as for a tensor with live views.

_GRAD_WRITTEN = {"mul_": 2.0 * V, "add_": 2.0 + V, "sub_": 2.0 - V, "div_": 2.0 / V}


@pytest.mark.parametrize("op", OPS)
def test_a_write_into_a_gradient_with_an_operand_that_requires_grad(op: str, device: str) -> None:
    first, _ = DERIVATIVES[op]
    p = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    (p * 2.0).sum().backward()
    w = lucid.tensor(V.tolist(), requires_grad=True, device=device)
    g = p.grad
    assert g is not None
    getattr(g, op)(w)
    assert p.grad is not None
    np.testing.assert_allclose(_np(p.grad), _GRAD_WRITTEN[op], rtol=1e-5)
    assert g.requires_grad
    (h,) = lucid.autograd.grad(g.sum(), w)
    np.testing.assert_allclose(_np(h), first(np.full(3, 2.0), V), rtol=1e-5)


# ── no over-refusal (CHA-62, LCD-320) ─────────────────────────────────────────
#
# validate_versions checks the inputs a node saved by value, not every input it
# was given.  A write that records a graph moves the version, and a node that
# took the tensor without saving it — ``add``, ``sum`` and ``concatenate`` read
# none of its values — let it through, as the reference does.


def _through_add(y: lucid.Tensor) -> lucid.Tensor:
    z = y + 1.0
    y.mul_(3.0)
    return z.sum()


def _through_sum(y: lucid.Tensor) -> lucid.Tensor:
    s = y.sum()
    y.add_(1.0)
    return s + y.sum()


def _through_concatenate(y: lucid.Tensor) -> lucid.Tensor:
    c = lucid.cat([y, y])
    y.add_(1.0)
    return c.sum()


# Each builds a loss from ``y = 2x`` with a write into ``y`` after ``y`` was
# taken, beside the reference's ``dloss/dx``.
_UNSAVED: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], list[float]]] = {
    "add": (_through_add, [2.0, 2.0]),
    "sum": (_through_sum, [4.0, 4.0]),
    "concatenate": (_through_concatenate, [4.0, 4.0]),
}


@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("consumer", list(_UNSAVED))
def test_a_write_into_an_input_the_node_did_not_save(
    consumer: str, create_graph: bool, device: str
) -> None:
    build, want = _UNSAVED[consumer]
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    # Held here: a node that keeps only a weak handle on ``y`` (concatenate)
    # skips the check once ``y`` is gone.
    y = x * 2.0
    loss = build(y)
    if create_graph:
        (g,) = lucid.autograd.grad(loss, x, create_graph=True)
    else:
        loss.backward()
        g = x.grad
    assert g is not None
    np.testing.assert_allclose(_np(g), want)
