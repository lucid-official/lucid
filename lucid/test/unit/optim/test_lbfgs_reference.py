"""LBFGS runs the reference's algorithm (LCD-222).

One ``step(closure)`` used to take a single quasi-Newton direction and an
Armijo backtracking search, so ``max_iter`` was ignored: on a 4-d SPD
quadratic, ``LBFGS(max_iter=20)`` made 4 closure calls and stopped 0.4 from
the optimum, where the reference converges inside that one step.

The defect class is an optimizer whose signature matches the reference while
its algorithm does not.  These tests pin the algorithm by its observable
trajectory — final parameters *and* the number of closure evaluations — for
both line-search modes, the stopping rules, the history window and the
checkpointed state.
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine

_A4 = np.array([[3, 1, 0, 0], [1, 4, 1, 0], [0, 1, 5, 1], [0, 0, 1, 6]], dtype=np.float32)
_B4 = np.array([1, 2, 3, 4], dtype=np.float32)

# Condition number 1e3: enough curvature pairs are needed that a short
# history visibly changes the path.
_A6 = np.diag(np.logspace(0, 3, 6)).astype(np.float32)
_B6 = np.ones(6, dtype=np.float32)

_ROSEN_2 = np.array([-1.2, 1.0], dtype=np.float32)
_ROSEN_4 = np.array([-1.2, 1.0, -0.5, 0.8], dtype=np.float32)


class _Lib:
    """Builds the same problem in Lucid (on ``device``) or in the reference."""

    def __init__(self, lib: Any, device: str | None) -> None:
        self.lib = lib
        self.device = device

    def param(self, a: np.ndarray) -> Any:
        if self.device is None:
            return self.lib.nn.Parameter(self.lib.tensor(a))
        return nn.Parameter(lucid.tensor(a, device=self.device))

    def const(self, a: np.ndarray) -> Any:
        if self.device is None:
            return self.lib.tensor(a)
        return lucid.tensor(a, device=self.device)


def _quadratic(a: np.ndarray, b: np.ndarray) -> Callable[[_Lib, Any], Any]:
    def f(lib: _Lib, x: Any) -> Any:
        return 0.5 * (x * (lib.const(a) @ x)).sum() - (lib.const(b) * x).sum()

    return f


def _rosenbrock(lib: _Lib, x: Any) -> Any:
    del lib  # needs no constants
    return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()


def _run(
    lib: _Lib, objective: Callable[[_Lib, Any], Any], x0: np.ndarray, steps: int = 1, **kwargs: Any
) -> tuple[np.ndarray, int, float, Any]:
    """Return the trajectory, closure calls, first ``step`` return, optimizer."""
    x = lib.param(x0)
    opt = lib.lib.optim.LBFGS([x], **kwargs)
    calls = [0]

    def closure() -> Any:
        calls[0] += 1
        opt.zero_grad()
        loss = objective(lib, x)
        loss.backward()
        return loss

    trajectory = []
    first = None
    for _ in range(steps):
        out = opt.step(closure)
        first = float(out.detach()) if first is None else first
        trajectory.append(x.detach().numpy().astype(np.float64).copy())
    return np.stack(trajectory), calls[0], float(first), opt


def _both(
    ref: Any,
    device: str,
    objective: Callable[[_Lib, Any], Any],
    x0: np.ndarray,
    steps: int = 1,
    **kwargs: Any,
) -> tuple[tuple[np.ndarray, int, float, Any], tuple[np.ndarray, int, float, Any]]:
    got = _run(_Lib(lucid, device), objective, x0, steps, **kwargs)
    want = _run(_Lib(ref, None), objective, x0, steps, **kwargs)
    return got, want


_LINE_SEARCHES = [None, "strong_wolfe"]


# ── the issue's repro ────────────────────────────────────────────────────────


@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
def test_one_step_converges_on_the_spd_quadratic(device: str, line_search_fn: str | None) -> None:
    traj, calls, _, _ = _run(
        _Lib(lucid, device),
        _quadratic(_A4, _B4),
        np.zeros(4, dtype=np.float32),
        lr=1,
        max_iter=20,
        history_size=10,
        line_search_fn=line_search_fn,
    )
    x_star = np.linalg.solve(_A4.astype(np.float64), _B4.astype(np.float64))
    assert np.abs(traj[-1] - x_star).max() < 1e-5
    assert 4 < calls <= 25  # max_eval defaults to max_iter * 5 // 4


# ── trajectories against the reference ───────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
@pytest.mark.parametrize("max_iter", [1, 5, 20])
def test_spd_quadratic_step_matches_the_reference(
    device: str, ref: Any, max_iter: int, line_search_fn: str | None
) -> None:
    got, want = _both(
        ref,
        device,
        _quadratic(_A4, _B4),
        np.zeros(4, dtype=np.float32),
        lr=1,
        max_iter=max_iter,
        history_size=10,
        line_search_fn=line_search_fn,
    )
    assert got[1] == want[1], "closure calls"
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-5)
    # ``step`` returns the loss of its first evaluation.
    assert got[2] == pytest.approx(want[2], abs=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
@pytest.mark.parametrize("x0", [_ROSEN_2, _ROSEN_4], ids=["2d", "4d"])
def test_rosenbrock_trajectory_matches_the_reference(
    device: str, ref: Any, x0: np.ndarray, line_search_fn: str | None
) -> None:
    # lr 0.5 and five iterations per step keep both runs well inside the
    # region where float32 rounding does not change the path.
    got, want = _both(
        ref, device, _rosenbrock, x0, steps=3, lr=0.5, max_iter=5, line_search_fn=line_search_fn
    )
    assert got[1] == want[1], "closure calls"
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-4)


@pytest.mark.parity
def test_several_parameters_are_one_flat_vector(device: str, ref: Any) -> None:
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((1, 3)).astype(np.float32)
    b0 = rng.standard_normal(1).astype(np.float32)
    xs = rng.standard_normal((8, 3)).astype(np.float32)
    ys = rng.standard_normal((8, 1)).astype(np.float32)

    def run(lib: _Lib) -> tuple[np.ndarray, int]:
        w, b = lib.param(w0), lib.param(b0)
        x, y = lib.const(xs), lib.const(ys)
        opt = lib.lib.optim.LBFGS([w, b], line_search_fn="strong_wolfe")
        calls = [0]

        def closure() -> Any:
            calls[0] += 1
            opt.zero_grad()
            loss = ((x @ w.T + b - y) ** 2).mean()
            loss.backward()
            return loss

        for _ in range(2):
            opt.step(closure)
        flat = np.concatenate([w.detach().numpy().ravel(), b.detach().numpy()])
        return flat, calls[0]

    got, want = run(_Lib(lucid, device)), run(_Lib(ref, None))
    assert got[1] == want[1]
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-5)


# ── stopping rules ───────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
@pytest.mark.parametrize(
    "tolerance",
    [{"tolerance_grad": 1e-2}, {"tolerance_change": 1e-3}],
    ids=["tolerance_grad", "tolerance_change"],
)
def test_a_tolerance_ends_the_step_where_the_reference_does(
    device: str, ref: Any, tolerance: dict[str, float], line_search_fn: str | None
) -> None:
    kwargs: dict[str, Any] = dict(lr=1, max_iter=20, line_search_fn=line_search_fn, **tolerance)
    got, want = _both(ref, device, _quadratic(_A4, _B4), np.zeros(4, dtype=np.float32), **kwargs)
    unstopped = _run(
        _Lib(lucid, device),
        _quadratic(_A4, _B4),
        np.zeros(4, dtype=np.float32),
        lr=1,
        max_iter=20,
        line_search_fn=line_search_fn,
    )
    assert got[1] == want[1], "closure calls"
    assert got[1] < unstopped[1], "the tolerance did not end the step early"
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-5)


def test_a_met_tolerance_grad_returns_after_one_evaluation(device: str) -> None:
    x0 = np.zeros(4, dtype=np.float32)
    traj, calls, first, opt = _run(
        _Lib(lucid, device), _quadratic(_A4, _B4), x0, tolerance_grad=10.0
    )
    assert calls == 1
    np.testing.assert_array_equal(traj[-1], x0)
    assert first == 0.0
    assert opt.state_dict()["state"][0]["n_iter"] == 0


@pytest.mark.parity
@pytest.mark.parametrize("max_iter", [1, 4, 20])
def test_max_eval_defaults_to_five_quarters_of_max_iter(ref: Any, max_iter: int) -> None:
    p = nn.Parameter(lucid.zeros(2))
    q = ref.nn.Parameter(ref.zeros(2))
    got = optim.LBFGS([p], max_iter=max_iter).param_groups[0]["max_eval"]
    want = ref.optim.LBFGS([q], max_iter=max_iter).param_groups[0]["max_eval"]
    assert got == want == max_iter * 5 // 4


@pytest.mark.parity
def test_max_eval_caps_the_line_search(device: str, ref: Any) -> None:
    got, want = _both(
        ref,
        device,
        _rosenbrock,
        _ROSEN_4,
        lr=1,
        max_iter=20,
        max_eval=7,
        line_search_fn="strong_wolfe",
    )
    assert got[1] == want[1] <= 7
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-5)


# ── the history window ───────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
def test_history_size_truncates_like_the_reference(
    device: str, ref: Any, line_search_fn: str | None
) -> None:
    kwargs: dict[str, Any] = dict(lr=1, max_iter=12, history_size=2, line_search_fn=line_search_fn)
    got, want = _both(ref, device, _quadratic(_A6, _B6), np.zeros(6, dtype=np.float32), **kwargs)
    assert got[1] == want[1], "closure calls"
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-4)
    held = got[3].state_dict()["state"][0]
    assert len(held["old_dirs"]) == len(held["old_stps"]) == len(held["ro"]) == 2
    assert len(want[3].state_dict()["state"][0]["old_dirs"]) == 2


# ── state ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
def test_a_reloaded_state_continues_identically(device: str, line_search_fn: str | None) -> None:
    lib = _Lib(lucid, device)
    kwargs: dict[str, Any] = dict(lr=0.5, max_iter=4, line_search_fn=line_search_fn)

    def make(x0: np.ndarray) -> tuple[Any, optim.LBFGS, Callable[[], Any], list[int]]:
        x = lib.param(x0)
        opt = optim.LBFGS([x], **kwargs)
        calls = [0]

        def closure() -> Any:
            calls[0] += 1
            opt.zero_grad()
            loss = _rosenbrock(lib, x)
            loss.backward()
            return loss

        return x, opt, closure, calls

    x, opt, closure, calls = make(_ROSEN_4)
    opt.step(closure)
    saved = copy.deepcopy(opt.state_dict())
    y, fresh, fresh_closure, fresh_calls = make(x.detach().numpy())
    fresh.load_state_dict(saved)
    calls[0] = 0
    for _ in range(2):
        opt.step(closure)
        fresh.step(fresh_closure)
        np.testing.assert_array_equal(x.detach().numpy(), y.detach().numpy())
    assert calls[0] == fresh_calls[0]
    assert fresh.state_dict()["state"][0]["n_iter"] == opt.state_dict()["state"][0]["n_iter"]


@pytest.mark.parity
@pytest.mark.parametrize("line_search_fn", _LINE_SEARCHES)
def test_state_keys_are_the_references(ref: Any, line_search_fn: str | None) -> None:
    got, want = _both(
        ref, "cpu", _rosenbrock, _ROSEN_4, lr=0.5, max_iter=4, line_search_fn=line_search_fn
    )
    mine = got[3].state_dict()["state"]
    theirs = want[3].state_dict()["state"]
    assert list(mine) == list(theirs) == [0]
    # ``al`` is the reference's scratch buffer for the two-loop recursion.
    assert set(mine[0]) == set(theirs[0]) - {"al"}
    assert mine[0]["n_iter"] == theirs[0]["n_iter"]
    assert mine[0]["func_evals"] == theirs[0]["func_evals"]


def test_a_fresh_optimizer_has_no_state() -> None:
    opt = optim.LBFGS([nn.Parameter(lucid.zeros(2))])
    assert opt.state_dict()["state"] == {}


# ── what the optimizer refuses ───────────────────────────────────────────────


@pytest.mark.parity
def test_more_than_one_param_group_is_refused(ref: Any) -> None:
    a, b = nn.Parameter(lucid.zeros(2)), nn.Parameter(lucid.zeros(2))
    with pytest.raises(ValueError, match="parameter groups"):
        optim.LBFGS([{"params": [a]}, {"params": [b]}])
    c, d = ref.nn.Parameter(ref.zeros(2)), ref.nn.Parameter(ref.zeros(2))
    with pytest.raises(ValueError, match="parameter groups"):
        ref.optim.LBFGS([{"params": [c]}, {"params": [d]}])


def test_a_second_group_cannot_be_added_later() -> None:
    opt = optim.LBFGS([nn.Parameter(lucid.zeros(2))])
    with pytest.raises(ValueError, match="parameter groups"):
        opt.add_param_group({"params": [nn.Parameter(lucid.zeros(2))]})
    assert len(opt.param_groups) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"line_search_fn": "armijo"}, "line_search_fn"), ({"history_size": 0}, "history_size")],
)
def test_an_option_the_step_cannot_run_is_refused_up_front(
    kwargs: dict[str, Any], message: str
) -> None:
    p = nn.Parameter(lucid.zeros(2))
    with pytest.raises(_C_engine.InvalidArgument, match=message):
        optim.LBFGS([p], **kwargs)
    opt = optim.LBFGS([p])
    opt.param_groups[0].update(kwargs)
    with pytest.raises(_C_engine.InvalidArgument, match=message):
        opt.step(lambda: (p**2).sum())


def test_a_parameter_stays_the_same_leaf_buffer(device: str) -> None:
    """The step writes into each parameter instead of rebinding it."""
    from lucid._dispatch import _unwrap

    x = nn.Parameter(lucid.tensor(_ROSEN_4, device=device))
    impl = x._impl
    opt = optim.LBFGS([x], lr=0.5, max_iter=4, line_search_fn="strong_wolfe")

    def closure() -> Any:
        opt.zero_grad()
        loss = _rosenbrock(_Lib(lucid, device), x)
        loss.backward()
        return loss

    opt.step(closure)
    assert x._impl is impl
    assert _unwrap(x).is_leaf and x.requires_grad
    assert not np.array_equal(x.detach().numpy(), _ROSEN_4)
