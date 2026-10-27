"""What ``Function.apply`` promises about ``forward``, its outputs, and what it saved.

Three defects with one owner, ``lucid/autograd/function.py``:

* LCD-588 / LCD-602 — a ``forward`` that saved its own outputs and ran no
  backward was never freed when two or more outputs were saved, or had to wait
  for the cyclic GC when one was saved.  The saved tensors live in the node's
  engine context (LCD-588), and each saved output is stored as a detached
  storage view plus a weak pointer to the node and its ``output_nr``
  (LCD-602), so dropping all outputs frees the node, its context, and the
  saved buffers immediately by plain reference counting.
* LCD-555 — ``forward`` ran with grad mode on.  Each call built a graph that
  registration then threw away, and an untracked call kept it, sending a
  gradient to a tensor ``forward`` closed over.  ``forward`` now runs under
  ``no_grad``, as the reference runs it.
* LCD-566 — a bool or integer output came back with ``requires_grad=True``
  and a ``grad_fn``.  Such an output is non-differentiable, as if marked.
  A marked output is handed back as a new tensor, and the one ``forward``
  returned is no longer detached in place.
"""

import gc
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.autograd import Function, FunctionCtx
from lucid.test._fixtures.memory import live_bytes

_ELEMS = 1 << 14  # 64 KiB of float32 per output: far above the noise
_SIDE = 64
_CALLS = 8


def _first(out: object) -> lucid.Tensor:
    first = out[0] if isinstance(out, tuple) else out
    assert isinstance(first, lucid.Tensor)
    return first


def _retained(run: Callable[[], object], device: str) -> int:
    """Bytes on ``device`` that ``_CALLS`` calls of ``run`` leave alive when
    each result is dropped without a backward."""
    lucid.eval(_first(run()))  # the first call settles caches and lazy constants
    gc.collect()
    before = live_bytes(device)
    for _ in range(_CALLS):
        out = run()
        lucid.eval(_first(out))
        del out
    gc.collect()
    return live_bytes(device) - before


# ── LCD-588: a call that runs no backward is freed ────────────────────────────


_SAVES = ("inputs", "one output", "all outputs")


def _scaled(n_out: int, saves: str) -> type[Function]:
    """``x * 2, x * 3, …`` — ``n_out`` outputs — saving what ``saves`` names."""
    scales = [float(k + 2) for k in range(n_out)]

    class Scaled(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> object:
            outs = tuple(x * s for s in scales)
            picked = {"inputs": (x,), "one output": outs[:1], "all outputs": outs}
            ctx.save_for_backward(*picked[saves])
            return outs if n_out > 1 else outs[0]

        @staticmethod
        def backward(ctx: FunctionCtx, *grads: lucid.Tensor) -> lucid.Tensor:
            _ = ctx.saved_tensors  # read, as a backward that uses them does
            total = grads[0] * scales[0]
            for g, s in zip(grads[1:], scales[1:], strict=True):
                total = total + g * s
            return total

    return Scaled


@pytest.mark.parametrize("saves", _SAVES)
@pytest.mark.parametrize("n_out", [1, 2, 3])
def test_a_call_that_runs_no_backward_is_freed(n_out: int, saves: str, device: str) -> None:
    fn = _scaled(n_out, saves)
    x = lucid.randn(_ELEMS, requires_grad=True, device=device)
    assert _first(fn.apply(x)).grad_fn is not None, "the case records no graph"
    grown = _retained(lambda: fn.apply(x), device)
    assert grown < _ELEMS * 4, f"{_CALLS} forwards left {grown} bytes alive"


@pytest.mark.parametrize("saves", ("one output", "all outputs"))
@pytest.mark.parametrize("n_out", [1, 2, 3])
def test_a_call_that_runs_no_backward_is_freed_by_refcount(
    n_out: int, saves: str, device: str
) -> None:
    fn = _scaled(n_out, saves)
    x = lucid.randn(_ELEMS, requires_grad=True, device=device)
    lucid.eval(_first(fn.apply(x)))
    gc.collect()
    gc.disable()
    try:
        before = live_bytes(device)
        for _ in range(4):
            out = fn.apply(x)
            lucid.eval(_first(out))
            del out
        grown = live_bytes(device) - before
    finally:
        gc.enable()
        gc.collect()
    assert grown < _ELEMS * 4, f"4 forwards left {grown} bytes alive without a GC"


def _square(device: str) -> lucid.Tensor:
    a = lucid.randn(_SIDE, _SIDE, device=device)
    return (a @ a.mT + _SIDE * lucid.eye(_SIDE, device=device)).requires_grad_()


#: Lucid's own ``Function``s, the multi-output ones (``eig``, ``eigh``, ``svd``,
#: ``qr``) among them.
_IN_TREE: dict[str, Callable[[lucid.Tensor], object]] = {
    "eig": lucid.linalg.eig,
    "eigh": lucid.linalg.eigh,
    "svd": lambda A: lucid.linalg.svd(A, full_matrices=False),
    "qr": lucid.linalg.qr,
    "cholesky": lucid.linalg.cholesky,
    "lu_factor": lucid.linalg.lu_factor,
    "ldl_factor": lucid.linalg.ldl_factor,
    "slogdet": lambda A: lucid.linalg.slogdet(A)[1],
    "pinv": lucid.linalg.pinv,
}


@pytest.mark.parametrize("name", list(_IN_TREE))
def test_an_in_tree_function_is_freed_without_a_backward(name: str, device: str) -> None:
    A = _square(device)
    run = _IN_TREE[name]
    assert _first(run(A)).grad_fn is not None, "the case records no graph"
    grown = _retained(lambda: run(A), device)
    assert grown < _SIDE * _SIDE * 4, f"{name}: {_CALLS} calls left {grown} bytes"


@pytest.mark.parametrize("name", list(_IN_TREE))
def test_an_in_tree_function_is_freed_by_refcount(name: str, device: str) -> None:
    A = _square(device)
    run = _IN_TREE[name]
    lucid.eval(_first(run(A)))
    gc.collect()
    gc.disable()
    try:
        before = live_bytes(device)
        for _ in range(4):
            out = run(A)
            lucid.eval(_first(out))
            del out
        grown = live_bytes(device) - before
    finally:
        gc.enable()
        gc.collect()
    assert grown < _SIDE * _SIDE * 4, f"{name}: 4 forwards left {grown} bytes alive without a GC"


class _ReadsBoth(Function):
    """``(2x, 3x)``, whose backward reads both saved outputs."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        a, b = x * 2.0, x * 3.0
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    def backward(ctx: FunctionCtx, ga: lucid.Tensor, gb: lucid.Tensor) -> lucid.Tensor:
        a, b = ctx.saved_tensors
        return ga * (b - a) + gb * a


def test_one_kept_output_keeps_what_backward_reads(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    a, b = _ReadsBoth.apply(x)
    del b
    gc.collect()
    a.sum().backward()  # gb arrives as zeros: the gradient is b - a = x
    assert x.grad is not None
    assert x.grad.tolist() == [1.0, 2.0, 3.0]


class _Exp(Function):
    """``exp(x)``, its backward reading the saved output."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
        (y,) = ctx.saved_tensors
        return g * y


class _ExpPair(Function):
    """``(exp(x), exp(2x))``, its backward reading both saved outputs."""

    @staticmethod
    def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        y, z = x.exp(), (x * 2.0).exp()
        ctx.save_for_backward(y, z)
        return y, z

    @staticmethod
    def backward(ctx: FunctionCtx, gy: lucid.Tensor, gz: lucid.Tensor) -> lucid.Tensor:
        y, z = ctx.saved_tensors
        return gy * y + gz * 2.0 * z


def _exp_loss(x: lucid.Tensor) -> lucid.Tensor:
    return (_Exp.apply(x) ** 2).sum()  # e^{2x}: f'' = 4 e^{2x}


def _pair_loss(x: lucid.Tensor) -> lucid.Tensor:
    y, z = _ExpPair.apply(x)
    return (y * y).sum() + z.sum()  # 2 e^{2x}: f'' = 8 e^{2x}


@pytest.mark.parametrize("loss,scale", [(_exp_loss, 4.0), (_pair_loss, 8.0)], ids=["one", "two"])
def test_a_second_derivative_runs_through_saved_outputs(
    loss: Callable[[lucid.Tensor], lucid.Tensor], scale: float, device: str
) -> None:
    # The outputs' Python tensors are gone before backward — only the
    # consumers' nodes hold them — so what ``backward`` reads must still
    # carry the node, or the second derivative loses its terms through them.
    x = lucid.tensor([0.1, 0.5], requires_grad=True, device=device)
    (g,) = lucid.autograd.grad(loss(x), [x], create_graph=True)
    (second,) = lucid.autograd.grad(g.sum(), [x])
    want = (x.detach() * 2.0).exp() * scale
    assert lucid.allclose(second, want, rtol=1e-5)


@pytest.mark.parametrize("grad_mode", [False, True], ids=["no_grad", "grad"])
@pytest.mark.parametrize("n_out", [1, 2])
def test_a_write_into_a_saved_output_is_refused(n_out: int, grad_mode: bool, device: str) -> None:
    fn = _scaled(n_out, "all outputs")
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    outs = fn.apply(x)
    roots = list(outs) if isinstance(outs, tuple) else [outs]
    if grad_mode:
        roots[-1].mul_(2.0)
    else:
        with lucid.no_grad():
            roots[-1].mul_(2.0)
    # Seeded at the outputs themselves: an op consuming one records its
    # version too and would refuse first, before the node's own check ran.
    with pytest.raises(_C_engine.VersionMismatch, match=rf"\(Scaled output {n_out - 1}\)"):
        lucid.autograd.backward(roots, [lucid.ones_like(r) for r in roots])


# ── LCD-555: ``forward`` records nothing ─────────────────────────────────────


def _probe(seen: dict[str, object]) -> type[Function]:
    class Probe(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            seen["enabled"] = lucid.is_grad_enabled()
            y = x * 2.0
            seen["inner"] = (y.requires_grad, y.grad_fn)
            return y

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
            return g * 2.0

    return Probe


@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
def test_forward_runs_with_grad_mode_off(tracked: bool, device: str) -> None:
    seen: dict[str, object] = {}
    x = lucid.randn(3, requires_grad=tracked, device=device)
    with lucid.enable_grad():
        y = _probe(seen).apply(x)
        assert lucid.is_grad_enabled(), "grad mode was not restored"
    assert seen == {"enabled": False, "inner": (False, None)}
    assert y.requires_grad is tracked


def test_grad_mode_comes_back_when_forward_raises(device: str) -> None:
    class Raises(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            raise ValueError("refused")

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
            return g

    with pytest.raises(ValueError, match="refused"):
        Raises.apply(lucid.randn(3, requires_grad=True, device=device))
    assert lucid.is_grad_enabled()


def test_a_tensor_forward_closes_over_gets_no_gradient(device: str) -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)

    class Scale(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            return x * w

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
            return g * w.detach()

    # No input needs a gradient, so nothing is recorded — the reference
    # sends none to ``w`` either.
    untracked = Scale.apply(lucid.ones(3, device=device))
    assert not untracked.requires_grad and untracked.grad_fn is None

    x = lucid.ones(3, requires_grad=True, device=device)
    Scale.apply(x).sum().backward()
    assert w.grad is None
    assert x.grad is not None and x.grad.tolist() == [1.0, 2.0, 3.0]


def test_forward_allocates_no_graph_node(device: str) -> None:
    class Chain(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            return ((x * 2.0).exp() + 1.0).sin()

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
            return g

    x = lucid.randn(4, requires_grad=True, device=device) * 1.0  # has its node
    probe = (x * 1.0).grad_fn
    assert probe is not None
    y = Chain.apply(x)
    assert y.grad_fn is not None
    # Every node takes the next sequence number: the call's own is the only one.
    assert y.grad_fn.sequence_nr == probe.sequence_nr + 1


# ── LCD-566: a bool or integer output is not differentiable ──────────────────


def _index_of(x: lucid.Tensor, dtype: lucid.dtype) -> lucid.Tensor:
    return (x > 0.0) if dtype == lucid.bool_ else lucid.arange(3, device=x.device).to(dtype)


@pytest.mark.parametrize("marked", [False, True], ids=["unmarked", "marked"])
@pytest.mark.parametrize("dtype", [lucid.bool_, lucid.int32, lucid.int64], ids=str)
def test_an_integer_output_is_not_differentiable(
    dtype: lucid.dtype, marked: bool, device: str
) -> None:
    grads: list[lucid.Tensor] = []

    class WithIndex(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
            idx = _index_of(x, dtype)
            if marked:
                ctx.mark_non_differentiable(idx)
            return x * 2.0, idx

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor, g_idx: lucid.Tensor) -> lucid.Tensor:
            grads.append(g_idx)
            return g * 2.0

    x = lucid.tensor([1.0, -2.0, 3.0], requires_grad=True, device=device)
    y, idx = WithIndex.apply(x)
    assert (idx.requires_grad, idx.grad_fn, idx.is_leaf) == (False, None, True)
    assert y.requires_grad and y.grad_fn is not None
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0, 2.0]
    # ``backward`` still has one gradient per output — zeros for the index.
    assert [g.tolist() for g in grads] == [[0, 0, 0]]


def test_a_function_returning_only_an_index_records_nothing(device: str) -> None:
    class ArgMax(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            return x.argmax()

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> None:
            return None

    out = ArgMax.apply(lucid.randn(4, requires_grad=True, device=device))
    assert (out.requires_grad, out.grad_fn) == (False, None)


def test_a_complex_output_stays_differentiable(device_cpu_only: str) -> None:
    class ToComplex(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> lucid.Tensor:
            return x.to(lucid.complex64)

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor) -> lucid.Tensor:
            return lucid.real(g)

    x = lucid.randn(3, requires_grad=True, device=device_cpu_only)
    out = ToComplex.apply(x)
    assert out.requires_grad and out.grad_fn is not None


def test_an_integer_input_passed_through_is_left_alone(device: str) -> None:
    class PassIndex(Function):
        @staticmethod
        def forward(
            ctx: FunctionCtx, x: lucid.Tensor, idx: lucid.Tensor
        ) -> tuple[lucid.Tensor, lucid.Tensor]:
            return x * 2.0, idx

        @staticmethod
        def backward(
            ctx: FunctionCtx, g: lucid.Tensor, g_idx: lucid.Tensor
        ) -> tuple[lucid.Tensor, None]:
            return g * 2.0, None

    idx = lucid.arange(3, device=device)
    y, out = PassIndex.apply(lucid.randn(3, requires_grad=True, device=device), idx)
    assert (idx.requires_grad, idx.grad_fn, idx.is_leaf) == (False, None, True)
    assert out is not idx and not out.requires_grad
    assert out.tolist() == [0, 1, 2]


@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
def test_a_marked_tensor_forward_did_not_make_is_left_alone(tracked: bool, device: str) -> None:
    # A module buffer, say: marking it used to detach it in place, rebinding
    # the object the module still holds.
    buf = lucid.ones(3, device=device)
    impl = buf._impl

    class WithBuffer(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
            ctx.mark_non_differentiable(buf)
            return x * 2.0, buf

        @staticmethod
        def backward(ctx: FunctionCtx, g: lucid.Tensor, g_buf: lucid.Tensor) -> lucid.Tensor:
            return g * 2.0

    x = lucid.randn(3, requires_grad=tracked, device=device)
    _, out = WithBuffer.apply(x)
    assert buf._impl is impl
    assert (buf.requires_grad, buf.grad_fn, buf.is_leaf) == (False, None, True)
    assert out is not buf and (out.requires_grad, out.grad_fn) == (False, None)
    assert out.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parity
@pytest.mark.parametrize("marked", [False, True], ids=["unmarked", "marked"])
@pytest.mark.parametrize("dtype", ["float32", "complex64", "int64", "bool"])
def test_which_outputs_are_differentiable_matches_the_reference(
    dtype: str, marked: bool, ref: ModuleType
) -> None:
    def outcome(lib: ModuleType, x: object) -> tuple[bool, bool, bool]:
        class Cast(lib.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = (x * 2.0).to(getattr(lib, dtype))
                if marked:
                    ctx.mark_non_differentiable(out)
                return x * 3.0, out

            @staticmethod
            def backward(ctx, g, g_out):
                return g * 3.0

        y, out = Cast.apply(x)
        return bool(y.requires_grad), bool(out.requires_grad), out.grad_fn is None

    got = outcome(lucid, lucid.tensor([1.0, 2.0], requires_grad=True))
    want = outcome(ref, ref.tensor([1.0, 2.0], requires_grad=True))
    assert got == want


def test_marked_outputs_add_no_stand_in_detach_to_grad_mode_trace(
    device: str,
) -> None:
    """Non-differentiable output stand-in aliases do not emit extra detach ops (LCD-612)."""
    from lucid.compile._core.trace_context import recording

    class TwoMarked(Function):
        @staticmethod
        def forward(
            ctx: FunctionCtx, x: lucid.Tensor
        ) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
            m1 = x + 1.0
            m2 = (x > 0.0).long()
            ctx.mark_non_differentiable(m1)
            return x * 2.0, m1, m2

        @staticmethod
        def backward(
            ctx: FunctionCtx,
            gy: lucid.Tensor,
            gm1: lucid.Tensor,
            gm2: lucid.Tensor,
        ) -> lucid.Tensor:
            del ctx, gm1, gm2
            return gy * 2.0

    x = lucid.randn(4, device=device, requires_grad=True)
    with recording(watch=(x,), grad_mode=False) as tr_no_grad:
        TwoMarked.apply(x)
    with recording(watch=(x,), grad_mode=True) as tr_grad:
        TwoMarked.apply(x)

    ops_no_grad = [op.name for op in tr_no_grad.graph.ops]
    ops_grad = [op.name for op in tr_grad.graph.ops]
    assert ops_grad == ops_no_grad
    assert ops_grad.count("detach") == 2
