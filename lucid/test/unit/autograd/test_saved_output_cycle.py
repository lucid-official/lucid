"""A backward node never holds its own output strongly (LCD-578).

The output owns the node that differentiates it, through ``grad_fn``.  A node
that held the output back closed a cycle only a backward broke: a forward
that ran none kept the output, the node and the graph behind them alive, one
copy per call.  ``scatter_reduce``'s amax / amin / prod nodes held the result
to read its values and guard its version.  They hold it weakly now, read it
while it lives — its version checked — and recompute it from the operands
they guard once it is gone.

Every node whose backward reads its op's output is listed in
:data:`_SAVES_OUTPUT`, and each must leave nothing alive after forwards that
run no backward.
"""

import gc
from collections.abc import Callable
from types import ModuleType


import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.autograd import Function, FunctionCtx
from lucid.test._fixtures.memory import live_bytes

_ELEMS = 1 << 14  # 64 KiB of float32 per output: far above the noise
_SIDE = 64
_CALLS = 8


def _retained(run: Callable[[], lucid.Tensor], device: str) -> int:
    """Bytes on ``device`` that ``_CALLS`` calls of ``run`` leave alive when
    each result is dropped without a backward."""
    lucid.eval(run())  # the first call settles caches and lazy constants
    gc.collect()
    before = live_bytes(device)
    for _ in range(_CALLS):
        out = run()
        lucid.eval(out)
        del out
    gc.collect()
    return live_bytes(device) - before


# ── one leak test per reduce mode ─────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["amax", "amin", "prod", "mean", "sum"])
def test_a_scatter_reduce_that_runs_no_backward_frees_its_graph(
    mode: str, device: str
) -> None:
    base = lucid.randn(_ELEMS, requires_grad=True, device=device)
    src = lucid.randn(_ELEMS, requires_grad=True, device=device)
    index = lucid.arange(_ELEMS, device=device)
    grown = _retained(lambda: base.scatter_reduce(0, index, src, reduce=mode), device)
    assert grown < _ELEMS * 4, f"{_CALLS} forwards left {grown} bytes alive"


# ── every node that reads its op's output ─────────────────────────────────────


class _SavesItsOutput(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2.0
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
        return g * 2.0


class _SavesBothOutputs(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        a, b = x * 2.0, x * 3.0
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    def backward(ctx: FunctionCtx, ga: lucid.Tensor, gb: lucid.Tensor) -> lucid.Tensor:
        return ga * 2.0 + gb * 3.0


def _vector(device: str) -> lucid.Tensor:
    return lucid.randn(_ELEMS, requires_grad=True, device=device)


def _positive(device: str) -> lucid.Tensor:
    return (lucid.rand(_ELEMS, device=device) + 0.5).requires_grad_()


def _open_unit(device: str) -> lucid.Tensor:
    return (lucid.rand(_ELEMS, device=device) * 1.8 - 0.9).requires_grad_()


def _rows(device: str) -> lucid.Tensor:
    return lucid.randn(_SIDE, _ELEMS // _SIDE, requires_grad=True, device=device)


def _spd(device: str) -> lucid.Tensor:
    a = lucid.randn(_SIDE, _SIDE, device=device)
    return (a @ a.mT + _SIDE * lucid.eye(_SIDE, device=device)).requires_grad_()


def _rhs(like: lucid.Tensor) -> lucid.Tensor:
    return lucid.ones(_SIDE, 4, device=like.device)


def _engine(
    fn: Callable[[_C_engine.TensorImpl], _C_engine.TensorImpl],
) -> Callable[[lucid.Tensor], lucid.Tensor]:
    return lambda x: _wrap(fn(x.impl))


def _scatter(mode: str) -> Callable[[lucid.Tensor], lucid.Tensor]:
    def run(x: lucid.Tensor) -> lucid.Tensor:
        src = x.detach() * 0.5
        index = lucid.arange(_ELEMS, device=x.device)
        return x.scatter_reduce(0, index, src.requires_grad_(), reduce=mode)

    return run


#: The node whose backward reads its op's output -> (input, forward).
_SAVES_OUTPUT: dict[
    str, tuple[Callable[[str], lucid.Tensor], Callable[[lucid.Tensor], lucid.Tensor]]
] = {
    # Unary kernels that save their output (kSavesOutput).
    "TanhBackward": (_vector, lucid.tanh),
    "SigmoidBackward": (_vector, lucid.sigmoid),
    "ExpBackward": (_vector, lucid.exp),
    "SqrtBackward": (_positive, lucid.sqrt),
    "RsqrtBackward": (_positive, lucid.rsqrt),
    "ErfinvBackward": (_open_unit, lucid.erfinv),
    "CubeRootBackward": (_vector, _engine(_C_engine.cube_root)),
    "RPowScalarBackward": (_vector, _engine(lambda a: _C_engine.rpow_scalar(2.0, a))),
    # Reductions that save their output.
    "MaxBackward": (_rows, lambda x: lucid.amax(x, dim=1)),
    "MinBackward": (_rows, lambda x: lucid.amin(x, dim=1)),
    # Nodes that keep the output's values themselves.
    "SoftmaxBackward": (_rows, lambda x: F.softmax(x, dim=-1)),
    "LogSoftmaxBackward": (_rows, lambda x: F.log_softmax(x, dim=-1)),
    "CumprodBackward": (_vector, lambda x: lucid.cumprod(x, 0)),
    "CummaxBackward": (_vector, lambda x: lucid.cummax(x, 0)[0]),
    "CumminBackward": (_vector, lambda x: lucid.cummin(x, 0)[0]),
    "DetBackward": (_spd, lucid.linalg.det),
    "InvBackward": (_spd, lucid.linalg.inv),
    "SolveBackward": (_spd, lambda A: lucid.linalg.solve(A, _rhs(A))),
    "SolveTriangularBackward": (
        _spd,
        lambda A: lucid.linalg.solve_triangular(A, _rhs(A), upper=False),
    ),
    "LUSolveBackward": (
        _spd,
        lambda A: lucid.linalg.lu_solve(*lucid.linalg.lu_factor(A), _rhs(A)),
    ),
    "NormBackward": (
        _spd,
        _engine(lambda a: _C_engine.linalg.norm(a, 2.0, [], False)),
    ),
    # The scatter_reduce nodes, which held it strongly.
    "ScatterExtremumNode[amax]": (_vector, _scatter("amax")),
    "ScatterExtremumNode[amin]": (_vector, _scatter("amin")),
    "ScatterProdNode": (_vector, _scatter("prod")),
    # Python Functions whose ctx saves the output.
    "custom Function saving its output": (_vector, _SavesItsOutput.apply),
    "cholesky (_CholeskyAutograd)": (_spd, lucid.linalg.cholesky),
    "lu_factor (_LUFactorGrad)": (_spd, lambda A: lucid.linalg.lu_factor(A)[0]),
    "ldl_factor (_LDLFactorGrad)": (_spd, lambda A: lucid.linalg.ldl_factor(A)[0]),
    # A node two or more outputs share: freed by refcounting through the engine
    # context's SavedVariable entries (LCD-588 / LCD-602).
    "custom Function saving both of its outputs": (
        _vector,
        lambda x: _SavesBothOutputs.apply(x)[0],
    ),
    "eig (_Eig)": (_spd, lambda A: lucid.linalg.eig(A)[1]),
    "eigh (_Eigh)": (_spd, lambda A: lucid.linalg.eigh(A)[1]),
    "svd (_SVD)": (_spd, lambda A: lucid.linalg.svd(A, full_matrices=False)[0]),
    "qr (_QR)": (_spd, lambda A: lucid.linalg.qr(A)[0]),
}


@pytest.mark.parametrize("node", list(_SAVES_OUTPUT))
def test_a_node_that_reads_its_output_frees_it_without_a_backward(
    node: str, device: str
) -> None:
    make, forward = _SAVES_OUTPUT[node]
    x = make(device)
    out = forward(x)
    assert out.grad_fn is not None, f"{node}: the case records no graph"
    one_output = out.numel() * out.element_size()
    del out
    grown = _retained(lambda: forward(x), device)
    assert grown < one_output, f"{node}: {_CALLS} forwards left {grown} bytes alive"


# ── the result is read live, or recomputed once it is gone ──────────────────


# Ties at every position: base's element and two of src's share the extremum
# at 0 and 2, src's two at 3 for amin, and src beats base alone at 1.
_BASE = [3.0, 0.0, 5.0, -1.0]
_SRC = [3.0, 3.0, 1.0, 5.0, 5.0, -1.0, 2.0]
_INDEX = [0, 0, 1, 2, 2, 3, 3]


class _Weigh(Function):
    """``x * [1, 2, ..., n]`` through a node that keeps nothing (a Python
    Function's node holds no tensor of its own), so the scatter result is
    freed as soon as nothing outside the graph holds it — even before
    ``backward()`` runs.  (A built-in consumer such as ``y.sum()`` also takes
    the recompute path once ``y`` is dropped outside the graph, because the
    engine calls ``release_saved`` on the consumer before running the scatter
    node.)"""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        return x * lucid.arange(1.0, x.shape[0] + 1.0, device=x.device)

    @staticmethod
    def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
        return g * lucid.arange(1.0, g.shape[0] + 1.0, device=g.device)


_CONSUMERS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "weigh": lambda y: _Weigh.apply(y).sum(),
    "sum": lambda y: y.sum(),
}


def _scatter_grads(
    mode: str, device: str, *, drop: bool, consumer: str = "weigh", passes: int = 1
) -> list[np.ndarray]:
    base = lucid.tensor(_BASE, device=device, requires_grad=True)
    src = lucid.tensor(_SRC, device=device, requires_grad=True)
    index = lucid.tensor(_INDEX, device=device)
    y = base.scatter_reduce(0, index, src, reduce=mode)
    loss = _CONSUMERS[consumer](y)
    if drop:
        del y
    grads: list[np.ndarray] = []
    for k in range(passes):
        base.grad = None
        src.grad = None
        loss.backward(retain_graph=k + 1 < passes)
        assert base.grad is not None and src.grad is not None
        grads += [base.grad.numpy(), src.grad.numpy()]
    return grads


def test_a_result_only_the_caller_held_is_gone_at_backward(device: str) -> None:
    # What puts the dropped "weigh" cases below on the recompute path.
    base = lucid.randn(_ELEMS, requires_grad=True, device=device)
    src = lucid.randn(_ELEMS, requires_grad=True, device=device)
    index = lucid.arange(_ELEMS, device=device)
    y = base.scatter_reduce(0, index, src, reduce="amax")
    loss = _Weigh.apply(y).sum()
    lucid.eval(loss)
    gc.collect()
    held = live_bytes(device)
    del y
    gc.collect()
    freed = held - live_bytes(device)
    assert freed >= _ELEMS * 4, f"dropping the result freed {freed} bytes"
    loss.backward()
    assert base.grad is not None and src.grad is not None


@pytest.mark.parametrize("consumer", list(_CONSUMERS))
@pytest.mark.parametrize("mode", ["amax", "amin"])
def test_a_dropped_result_gives_the_kept_results_gradient(
    mode: str, consumer: str, device: str
) -> None:
    kept = _scatter_grads(mode, device, drop=False, consumer=consumer)
    dropped = _scatter_grads(mode, device, drop=True, consumer=consumer)
    for a, b in zip(kept, dropped, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["amax", "amin"])
def test_the_tie_split_matches_the_reference(
    mode: str, device: str, ref: ModuleType
) -> None:
    base = ref.tensor(_BASE, requires_grad=True)
    src = ref.tensor(_SRC, requires_grad=True)
    y = base.scatter_reduce(0, ref.tensor(_INDEX), src, reduce=mode)
    (y * ref.arange(1.0, len(_BASE) + 1.0)).sum().backward()
    want = [base.grad.numpy(), src.grad.numpy()]
    got = _scatter_grads(mode, device, drop=True)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", ["amax", "amin"])
@pytest.mark.parametrize("drop", [False, True], ids=["live", "recomputed"])
def test_a_retained_graph_answers_twice(mode: str, drop: bool, device: str) -> None:
    first_base, first_src, second_base, second_src = _scatter_grads(
        mode, device, drop=drop, passes=2
    )
    np.testing.assert_array_equal(first_base, second_base)
    np.testing.assert_array_equal(first_src, second_src)


@pytest.mark.parametrize("mode", ["amax", "amin", "prod"])
@pytest.mark.parametrize("drop", [False, True], ids=["live", "recomputed"])
def test_create_graph_is_refused_either_way(mode: str, drop: bool, device: str) -> None:
    base = lucid.tensor(_BASE, device=device, requires_grad=True)
    src = lucid.tensor(_SRC, device=device, requires_grad=True)
    y = base.scatter_reduce(0, lucid.tensor(_INDEX, device=device), src, reduce=mode)
    loss = _Weigh.apply(y).sum()
    if drop:
        del y
    with pytest.raises(RuntimeError, match="create_graph=True is not yet supported"):
        lucid.autograd.grad(loss, [base, src], create_graph=True)


@pytest.mark.parametrize("mode", ["amax", "amin", "prod"])
@pytest.mark.parametrize("drop", [False, True], ids=["live", "recomputed"])
def test_allow_mutation_on_saved_tensors_uses_pre_write_operands(
    mode: str, drop: bool, device: str
) -> None:
    def run(mutate: bool) -> list[np.ndarray]:
        b0 = lucid.tensor([3.0, 0.0, 5.0, -1.0, 7.0], device=device, requires_grad=True)
        s0 = lucid.tensor(_SRC, device=device, requires_grad=True)
        base = b0 * 1.0
        src = s0 * 1.0
        index = lucid.tensor(_INDEX, device=device)
        y = base.scatter_reduce(0, index, src, reduce=mode)
        loss = _Weigh.apply(y).sum()
        if drop:
            del y
        if mutate:
            with lucid.autograd.graph.allow_mutation_on_saved_tensors():
                with lucid.no_grad():
                    base.add_(10.0)
                    src.add_(-20.0)
                loss.backward()
        else:
            loss.backward()
        assert b0.grad is not None and s0.grad is not None
        return [b0.grad.numpy(), s0.grad.numpy()]

    want_b, want_s = run(mutate=False)
    got_b, got_s = run(mutate=True)
    np.testing.assert_allclose(got_b, want_b, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got_s, want_s, rtol=0, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("mode", ["amax", "amin"])
@pytest.mark.parametrize("drop", [False, True], ids=["live", "recomputed"])
def test_extremum_nan_winner_gradient_matches_reference(
    mode: str, drop: bool, device: str, ref: ModuleType
) -> None:
    base_vals = [3.0, 0.0, 5.0, -1.0, 7.0]
    src_vals = [3.0, float("nan"), 1.0, 5.0, 5.0, -1.0, 2.0]

    r_base = ref.tensor(base_vals, requires_grad=True)
    r_src = ref.tensor(src_vals, requires_grad=True)
    r_y = r_base.scatter_reduce(0, ref.tensor(_INDEX), r_src, reduce=mode)
    (r_y * ref.arange(1.0, len(base_vals) + 1.0)).sum().backward()

    base = lucid.tensor(base_vals, device=device, requires_grad=True)
    src = lucid.tensor(src_vals, device=device, requires_grad=True)
    y = base.scatter_reduce(0, lucid.tensor(_INDEX, device=device), src, reduce=mode)
    loss = _Weigh.apply(y).sum()
    if drop:
        del y
    loss.backward()
    assert base.grad is not None and src.grad is not None
    np.testing.assert_allclose(
        base.grad.numpy(), r_base.grad.numpy(), rtol=0, atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        src.grad.numpy(), r_src.grad.numpy(), rtol=0, atol=1e-6, equal_nan=True
    )
    assert np.isnan(base.grad.numpy()[0])
    assert np.all(np.isnan(src.grad.numpy()[:2]))


@pytest.mark.parametrize("mode", ["amax", "amin", "prod"])
def test_a_write_into_the_result_is_still_refused(mode: str, device: str) -> None:
    base = lucid.tensor(_BASE, device=device, requires_grad=True)
    src = lucid.tensor(_SRC, device=device, requires_grad=True)
    out = base.scatter_reduce(0, lucid.tensor(_INDEX, device=device), src, reduce=mode)
    out.mul_(2.0)
    with pytest.raises(_C_engine.VersionMismatch, match=f"scatter_{mode} output 0"):
        out.sum().backward()


def _replace_by_setitem(y: lucid.Tensor) -> lucid.Tensor:
    # ``y`` is rebound to a new impl; the result the node saw is dropped.
    y[0] = 5.0
    return y.sum()


def _write_under_no_grad_then_drop(y: lucid.Tensor) -> lucid.Tensor:
    loss = y.sum()
    with lucid.no_grad():
        y.add_(0.0)
    return loss


@pytest.mark.xfail(
    strict=True,
    reason="LCD-590: a saved output's version does not outlive the tensor, so a "
    "write after which nothing holds the result is not refused",
)
@pytest.mark.parametrize(
    "write",
    [_replace_by_setitem, _write_under_no_grad_then_drop],
    ids=["setitem", "no_grad"],
)
@pytest.mark.parametrize("mode", ["amax", "amin", "prod"])
def test_a_write_and_then_dropping_the_result_is_refused(
    mode: str, write: Callable[[lucid.Tensor], lucid.Tensor], device: str
) -> None:
    base = lucid.tensor(_BASE, device=device, requires_grad=True)
    src = lucid.tensor(_SRC, device=device, requires_grad=True)
    y = base.scatter_reduce(0, lucid.tensor(_INDEX, device=device), src, reduce=mode)
    loss = write(y)
    del y
    with pytest.raises(_C_engine.VersionMismatch, match=f"scatter_{mode} output 0"):
        loss.backward()
