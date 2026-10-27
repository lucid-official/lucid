"""``func.jvp`` / ``jacfwd`` / ``linearize``: attachment, fallback and cost (LCD-463).

Three things broke together in one refactor and are pinned here:

* **Attachment.**  The tangent was built with ``create_graph`` whenever grad
  mode was on, so a plain call came back ``requires_grad=True`` and held its
  graph; the reference attaches a result only when it depends on something
  that requires grad — a primal, a tangent, or a parameter ``func``
  captures.  Compared flag by flag, and gradient by gradient, against the
  reference.
* **Fallback.**  The double-backward route differentiates a backward, which
  an op with no graph-mode backward (``LSTM``, a padded ``avg_pool``)
  refuses; the per-element route that answered instead had been deleted.
* **Cost.**  Learning whether captured parameters attach the primal output
  takes one evaluation of ``func`` at the primals themselves — so ``func``
  runs twice in exactly one case, with the same random draws.  The counts
  are pinned so they cannot change silently.
"""

import annotationlib
import contextlib
import inspect
import math
import warnings
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest

import lucid
import lucid.func as func
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

_CASES = ["plain", "primal_rg", "tangent_rg", "captured", "no_grad"]


def _flat(t: Tensor) -> list[float]:
    return [float(v) for v in t.detach().cpu().reshape(-1).tolist()]


def _assert_values(got: Tensor, want: Tensor, tol: float = 1e-10) -> None:
    g, w = _flat(got), _flat(want)
    assert len(g) == len(w)
    assert all(math.isclose(a, b, rel_tol=tol, abs_tol=tol) for a, b in zip(g, w, strict=True)), (
        g,
        w,
    )


def _vec(m: ModuleType, vals: list[float], device: str = "cpu") -> Tensor:
    t: Tensor = m.tensor(vals, dtype=m.float64, device=device)
    return t


def _scenario(
    m: ModuleType, case: str
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor, Tensor]:
    """``(f, x, t, w)`` with ``f(a) = w * a * a`` — the tangent ``2 w a t``
    depends on the captured ``w``, the primal, and the tangent alike."""
    x = _vec(m, [0.5, -1.0, 2.0])
    t = _vec(m, [1.0, 0.25, -0.5])
    w = _vec(m, [1.5, -2.0, 0.75])
    if case == "primal_rg":
        x.requires_grad_(True)
    if case == "tangent_rg":
        t.requires_grad_(True)
    if case in ("captured", "no_grad"):
        w.requires_grad_(True)

    def f(a: Tensor) -> Tensor:
        return w * a * a

    return f, x, t, w


def _mode(m: ModuleType, case: str) -> contextlib.AbstractContextManager[None]:
    return m.no_grad() if case == "no_grad" else contextlib.nullcontext()


# ── attachment, against the reference ───────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("case", _CASES)
def test_jvp_attaches_where_the_reference_does(ref: ModuleType, case: str) -> None:
    got = {}
    for m in (lucid, ref):
        f, x, t, _ = _scenario(m, case)
        with _mode(m, case):
            got[m] = m.func.jvp(f, (x,), (t,))
    (y_l, dy_l), (y_r, dy_r) = got[lucid], got[ref]
    assert (y_l.requires_grad, dy_l.requires_grad) == (
        y_r.requires_grad,
        dy_r.requires_grad,
    )
    _assert_values(y_l, y_r)
    _assert_values(dy_l, dy_r)


@pytest.mark.parity
@pytest.mark.parametrize("case", ["primal_rg", "tangent_rg", "captured"])
def test_an_attached_tangent_carries_the_reference_gradient(ref: ModuleType, case: str) -> None:
    """Attached is not enough: a loss on the tangent and the primal output
    must send the reference's gradient to whatever requires grad."""
    grads = {}
    for m in (lucid, ref):
        f, x, t, w = _scenario(m, case)
        y, dy = m.func.jvp(f, (x,), (t,))
        ((y * y).sum() + (dy * dy).sum()).backward()
        leaf = {"primal_rg": x, "tangent_rg": t, "captured": w}[case]
        grads[m] = leaf.grad
    assert grads[lucid] is not None and grads[ref] is not None
    _assert_values(grads[lucid], grads[ref])


@pytest.mark.parity
@pytest.mark.parametrize("case", ["plain", "primal_rg", "captured", "no_grad"])
def test_jacfwd_attaches_where_the_reference_does(ref: ModuleType, case: str) -> None:
    got = {}
    for m in (lucid, ref):
        f, x, _, _ = _scenario(m, case)
        with _mode(m, case):
            got[m] = m.func.jacfwd(f)(x)
    assert got[lucid].requires_grad == got[ref].requires_grad
    _assert_values(got[lucid], got[ref])


@pytest.mark.parity
@pytest.mark.parametrize("case", ["plain", "primal_rg", "captured"])
def test_linearize_attaches_where_the_reference_does(ref: ModuleType, case: str) -> None:
    got = {}
    for m in (lucid, ref):
        f, x, t, _ = _scenario(m, case)
        y, lin = m.func.linearize(f, x)
        got[m] = (y, lin(t))
    (y_l, z_l), (y_r, z_r) = got[lucid], got[ref]
    assert (y_l.requires_grad, z_l.requires_grad) == (
        y_r.requires_grad,
        z_r.requires_grad,
    )
    _assert_values(y_l, y_r)
    _assert_values(z_l, z_r)


# ── the per-element fallback ─────────────────────────────────────────────────


def _image(m: ModuleType, shape: tuple[int, ...], offset: float) -> Tensor:
    n = math.prod(shape)
    vals = [math.sin(offset + 0.37 * i) for i in range(n)]
    return _vec(m, vals).reshape(shape)


def _padded_pool(a: Tensor) -> Tensor:
    pooled: Tensor = F.avg_pool2d(a, 3, stride=1, padding=1)
    return pooled


@pytest.mark.parity
def test_jvp_through_a_padded_avg_pool_matches_the_reference(ref: ModuleType) -> None:
    got = {}
    for m in (lucid, ref):
        x, t = _image(m, (1, 2, 5, 5), 0.1), _image(m, (1, 2, 5, 5), 1.3)
        pool = m.nn.functional.avg_pool2d
        got[m] = m.func.jvp(lambda a: pool(a, 3, stride=1, padding=1), (x,), (t,))
    (y_l, dy_l), (y_r, dy_r) = got[lucid], got[ref]
    assert (y_l.requires_grad, dy_l.requires_grad) == (False, False)
    assert (y_r.requires_grad, dy_r.requires_grad) == (False, False)
    _assert_values(dy_l, dy_r)


@pytest.mark.parity
def test_jacfwd_and_linearize_through_a_padded_avg_pool(ref: ModuleType) -> None:
    jac, lin = {}, {}
    for m in (lucid, ref):
        x, t = _image(m, (1, 1, 3, 3), 0.4), _image(m, (1, 1, 3, 3), 2.2)
        pool = m.nn.functional.avg_pool2d

        def f(a: Tensor, pool: Callable[..., Tensor] = pool) -> Tensor:
            return pool(a, 3, stride=1, padding=1)

        jac[m] = m.func.jacfwd(f)(x)
        lin[m] = m.func.linearize(f, x)[1](t)
    _assert_values(jac[lucid], jac[ref])
    _assert_values(lin[lucid], lin[ref])
    assert not jac[lucid].requires_grad and not lin[lucid].requires_grad


@pytest.mark.parity
def test_jvp_through_an_lstm_matches_the_reference(ref: ModuleType) -> None:
    """The LSTM's own parameters require grad, so the reference attaches the
    tangent to them.  Its backward has no graph mode here: the tangent is
    exact but comes back detached, with a warning, and the primal output
    still trains the parameters."""
    lucid.manual_seed(0)
    lstm_l = nn.LSTM(3, 4, batch_first=True).double()
    lstm_r = ref.nn.LSTM(3, 4, batch_first=True).double()
    with ref.no_grad():
        for (name, p_l), (_, p_r) in zip(
            lstm_l.named_parameters(), lstm_r.named_parameters(), strict=True
        ):
            p_r.copy_(ref.tensor(p_l.tolist(), dtype=ref.float64))
            assert name in dict(lstm_r.named_parameters())
    got = {}
    for m, lstm in ((lucid, lstm_l), (ref, lstm_r)):
        x, t = _image(m, (1, 5, 3), 0.2), _image(m, (1, 5, 3), 0.9)
        got[m] = m.func.jvp(lambda a, lstm=lstm: lstm(a)[0], (x,), (t,))
    (y_l, dy_l), (y_r, dy_r) = got[lucid], got[ref]
    _assert_values(y_l, y_r, tol=1e-9)
    _assert_values(dy_l, dy_r, tol=1e-9)
    assert y_l.requires_grad and y_r.requires_grad
    assert dy_r.requires_grad and not dy_l.requires_grad


def test_lstm_jvp_warns_that_its_tangent_is_detached() -> None:
    lstm = nn.LSTM(3, 4, batch_first=True)
    x, t = lucid.randn(1, 5, 3), lucid.randn(1, 5, 3)
    with pytest.warns(UserWarning, match="no graph-mode backward"):
        func.jvp(lambda a: lstm(a)[0], (x,), (t,))


def test_autograd_jvp_falls_back_one_element_at_a_time(device: str) -> None:
    """Exact against the Jacobian itself, assembled by plain reverse passes."""
    x = lucid.randn(1, 2, 4, 4).to(device)
    t = lucid.randn(1, 2, 4, 4).to(device)
    _, dy = lucid.autograd.jvp(_padded_pool, x, t)
    jac = lucid.autograd.jacobian(_padded_pool, x)
    want = (jac.reshape(32, 32) @ t.reshape(32, 1)).reshape(1, 2, 4, 4)
    assert isinstance(dy, Tensor)
    assert dy.device == x.device
    assert float((dy - want).abs().max().item()) < 1e-5


def test_create_graph_through_such_an_op_is_refused() -> None:
    """A tangent that must stay differentiable cannot be built through it —
    refused, not handed back silently detached."""
    x, t = lucid.randn(1, 1, 4, 4), lucid.randn(1, 1, 4, 4)
    with pytest.raises(NotImplementedError):
        lucid.autograd.jvp(_padded_pool, x, t, create_graph=True)
    with pytest.raises(NotImplementedError):
        func.jvp(_padded_pool, (x.requires_grad_(True),), (t,))


def test_strict_refuses_an_unreached_input_on_the_fallback_too() -> None:
    x, t = lucid.randn(1, 1, 4, 4), lucid.randn(1, 1, 4, 4)
    with pytest.raises(RuntimeError, match="independent of input 1"):
        lucid.autograd.jvp(lambda a, b: _padded_pool(a), (x, x), (t, t), strict=True)


def test_the_fallback_checks_v_as_the_fast_route_does() -> None:
    x = lucid.randn(1, 1, 4, 4)
    with pytest.raises(_C_engine.ShapeMismatch):
        lucid.autograd.jvp(_padded_pool, x, lucid.randn(4))


# ── how often func runs ─────────────────────────────────────────────────────


class _Counted:
    """``f(a) = w * a * a`` that counts its evaluations."""

    def __init__(self, w: Tensor) -> None:
        self.w = w
        self.calls = 0

    def __call__(self, a: Tensor) -> Tensor:
        self.calls += 1
        out: Tensor = self.w * a * a
        return out


@contextlib.contextmanager
def _grad_mode(on: bool) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        if not on:
            stack.enter_context(lucid.no_grad())
        yield


@pytest.mark.parametrize(
    ("grad_on", "x_rg", "w_rg", "calls"),
    [
        pytest.param(False, False, True, 1, id="grad_off"),
        pytest.param(True, True, False, 1, id="primal_requires_grad"),
        pytest.param(True, False, False, 2, id="nothing_requires_grad"),
        pytest.param(True, False, True, 2, id="captured_parameter"),
    ],
)
def test_how_many_times_jvp_runs_func(grad_on: bool, x_rg: bool, w_rg: bool, calls: int) -> None:
    f = _Counted(lucid.tensor([1.5, -2.0, 0.75]).requires_grad_(w_rg))
    x = lucid.tensor([0.5, -1.0, 2.0]).requires_grad_(x_rg)
    with _grad_mode(grad_on):
        func.jvp(f, (x,), (lucid.ones(3),))
    assert f.calls == calls


@pytest.mark.parametrize("w_rg", [False, True], ids=["plain", "captured"])
def test_jacfwd_and_linearize_decide_once(w_rg: bool) -> None:
    """One evaluation at the arguments, then one per column or per call —
    never the second evaluation :func:`jvp` needs, once per column."""
    f = _Counted(lucid.tensor([1.5, -2.0, 0.75]).requires_grad_(w_rg))
    x = lucid.tensor([0.5, -1.0, 2.0])
    func.jacfwd(f)(x)
    assert f.calls == 1 + x.numel()

    f.calls = 0
    _, lin = func.linearize(f, x)
    assert f.calls == 1
    lin(lucid.ones(3))
    lin(lucid.ones(3))
    assert f.calls == 3


def test_the_degraded_tangent_costs_one_more_run() -> None:
    """Through an op with no graph-mode backward, behind captured parameters:
    the primal run, the refused attached attempt, the first-order rerun."""
    calls = 0
    lstm = nn.LSTM(3, 4, batch_first=True)

    def f(a: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        out: Tensor = lstm(a)[0]
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        func.jvp(f, (lucid.randn(1, 5, 3),), (lucid.randn(1, 5, 3),))
    assert calls == 3


# ── the two runs draw the same randomness ────────────────────────────────────


def test_dropout_draws_the_same_mask_in_both_runs() -> None:
    """``func`` runs twice when nothing passed in requires grad; a mask
    drawn afresh the second time would pair one run's primal output with
    another run's tangent."""
    lucid.manual_seed(3)
    x = lucid.ones(64)
    y, dy = func.jvp(lambda a: F.dropout(a, 0.5, training=True), (x,), (x,))
    # f(a) = m * a / (1 - p), so with a = t = 1 the two are equal exactly
    # when the masks are.
    assert _flat(y) == _flat(dy)
    assert 0.0 in _flat(y) and 2.0 in _flat(y)


def test_the_stream_moves_as_far_as_one_run() -> None:
    def f(a: Tensor) -> Tensor:
        out: Tensor = F.dropout(a, 0.5, training=True)
        return out

    x = lucid.ones(16)
    lucid.manual_seed(5)
    f(x)
    after_one = _flat(lucid.rand(8))
    lucid.manual_seed(5)
    func.jvp(f, (x,), (x,))
    assert _flat(lucid.rand(8)) == after_one


# ── the public signatures say what they return ───────────────────────────────


@pytest.mark.parametrize("name", ["grad", "grad_and_value", "vjp", "jacrev", "jacfwd", "hessian"])
def test_transform_signatures_are_not_widened_to_any(name: str) -> None:
    sig = inspect.signature(getattr(func, name), annotation_format=annotationlib.Format.STRING)
    assert "Any" not in str(sig), str(sig)
