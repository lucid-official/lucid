"""An optimizer keeps training a parameter after the first ``step()``,
whatever happens to that parameter's flag or buffer in between.

Engine optimizers bind ``TensorImpl`` pointers.  Two kinds of call used to
give a parameter a new impl under the same ``Parameter`` object after the
optimizer had bound the old one, and the parameter silently stopped
moving:

* ``requires_grad`` changes — the setter, ``requires_grad_`` (even a
  redundant ``requires_grad_(True)``) and ``Module.requires_grad_``.  These
  now flip the flag on the impl itself, so nothing is replaced.
* conversions through ``Module._apply`` — ``.to(device)``, ``.double()``,
  ``.half()``.  These really do replace the buffer; the optimizer now
  notices, rebuilds the group's engine and carries its state across.

"Keeps training" is not enough on its own: an engine rebuilt with empty
state also moves the parameter, along a different path.  So every test
here compares the whole trajectory with an uninterrupted run, which is
what the reference framework produces, since its ``Parameter`` object
never changes.  The ``parity`` tests check that against the reference
directly.
"""

import warnings
from typing import Any, Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

_W0 = np.array([[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]], dtype=np.float32)
_B0 = np.array([0.1, -0.2], dtype=np.float32)
_X = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]], dtype=np.float32)
_STEPS = 4

_OPTIMIZERS: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("Adam", {"lr": 0.05}),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.01}),
]
_OPT_IDS = [name for name, _ in _OPTIMIZERS]


def _other(device: str) -> str:
    return "metal" if device == "cpu" else "cpu"


class _Run:
    """One model, one optimizer and the device/dtype its input must use."""

    def __init__(self, name: str, kwargs: dict[str, Any], device: str) -> None:
        self.model = nn.Linear(3, 2)
        self.model.weight = nn.Parameter(lucid.tensor(_W0.copy()))
        self.model.bias = nn.Parameter(lucid.tensor(_B0.copy()))
        self.model.to(device)
        self.opt = getattr(optim, name)(self.model.parameters(), **kwargs)
        self.device = device
        self.dtype = lucid.float32

    def step(self) -> np.ndarray:
        x = lucid.tensor(_X, device=self.device, dtype=self.dtype)
        self.opt.zero_grad()
        (self.model(x) ** 2).sum().backward()
        self.opt.step()
        return self.model.weight.detach().to("cpu").to(lucid.float64).numpy().copy()


# Each path takes the run between steps and returns the tolerance its
# trajectory is held to: exact when only a flag changed, float32 rounding
# when the buffer was converted.
def _rg_method_redundant(r: _Run) -> float:
    r.model.weight.requires_grad_(True)
    r.model.bias.requires_grad_(True)
    return 0.0


def _rg_setter_redundant(r: _Run) -> float:
    r.model.weight.requires_grad = True
    r.model.bias.requires_grad = True
    return 0.0


def _rg_method_toggle(r: _Run) -> float:
    for p in r.model.parameters():
        p.requires_grad_(False)
        p.requires_grad_(True)
    return 0.0


def _rg_setter_toggle(r: _Run) -> float:
    for p in r.model.parameters():
        p.requires_grad = False
        p.requires_grad = True
    return 0.0


def _module_requires_grad(r: _Run) -> float:
    r.model.requires_grad_()
    return 0.0


def _module_move(r: _Run) -> float:
    r.device = _other(r.device)
    r.model.to(r.device)
    return 1e-6


def _module_cast(r: _Run) -> float:
    # float64 does not exist on Metal; the cast there is to float16.
    if r.device == "cpu":
        r.model.double()
        r.dtype = lucid.float64
        return 1e-6
    r.model.half()
    r.dtype = lucid.float16
    return 5e-2


_PATHS: dict[str, Callable[[_Run], float]] = {
    "requires_grad_-redundant": _rg_method_redundant,
    "requires_grad=-redundant": _rg_setter_redundant,
    "requires_grad_-toggle": _rg_method_toggle,
    "requires_grad=-toggle": _rg_setter_toggle,
    "module.requires_grad_": _module_requires_grad,
    "module.to-device": _module_move,
    "module.cast": _module_cast,
}


def _trajectory(
    name: str, kwargs: dict[str, Any], device: str, path: Callable[[_Run], float]
) -> tuple[np.ndarray, float]:
    """Weights after each step, with ``path`` applied after steps 1 and 2."""
    run = _Run(name, kwargs, device)
    tol = 0.0
    out = []
    for i in range(_STEPS):
        out.append(run.step())
        if i in (0, 1):
            tol = max(tol, path(run))
    return np.stack(out), tol


def _uninterrupted(name: str, kwargs: dict[str, Any], device: str) -> np.ndarray:
    run = _Run(name, kwargs, device)
    return np.stack([run.step() for _ in range(_STEPS)])


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("path", list(_PATHS), ids=list(_PATHS))
@pytest.mark.parametrize(("name", "kwargs"), _OPTIMIZERS, ids=_OPT_IDS)
def test_training_continues_with_state(
    name: str, kwargs: dict[str, Any], path: str, device: str
) -> None:
    want = _uninterrupted(name, kwargs, device)
    with warnings.catch_warnings():
        # These optimizers hand their state over, so nothing may be reset.
        warnings.simplefilter("error", RuntimeWarning)
        got, tol = _trajectory(name, kwargs, device, _PATHS[path])
    # Every step moved the weights — the bug left steps 2 onwards frozen.
    assert np.all(np.abs(np.diff(got, axis=0)).max(axis=(1, 2)) > 0)
    # And along the uninterrupted path, so momentum / moments survived.
    np.testing.assert_allclose(got, want, rtol=0, atol=max(tol, 1e-7))


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_flag_change_keeps_impl_and_grad(device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    impl = p._impl
    (p * 3.0).sum().backward()
    p.requires_grad_(False)
    p.requires_grad = True
    p.requires_grad_(True)
    assert p._impl is impl
    np.testing.assert_allclose(p.grad.numpy(), [3.0, 3.0, 3.0])
    (p * 2.0).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), [5.0, 5.0, 5.0])


def test_frozen_parameter_takes_no_gradient() -> None:
    w = nn.Parameter(lucid.ones(2))
    x = nn.Parameter(lucid.ones(2))
    w.requires_grad_(False)
    (w * x).sum().backward()
    assert w.grad is None
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0])


def test_computed_tensor_flag_follows_its_graph() -> None:
    x = lucid.ones(2, requires_grad=True)
    y = x * 2.0
    # Already requires grad: a no-op, and the graph back to x is intact.
    assert y.requires_grad_(True) is y
    assert not y.is_leaf
    y.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0])
    with pytest.raises(RuntimeError, match="leaf"):
        (x * 2.0).requires_grad_(False)
    for v in (True, False):
        z = x * 2.0
        with pytest.raises(RuntimeError, match="leaf"):
            z.requires_grad = v


def test_conversion_carries_state_into_new_dtype() -> None:
    run = _Run("Adam", {"lr": 0.05}, "cpu")
    run.step()
    run.model.double()
    run.dtype = lucid.float64
    run.step()
    state = run.opt.state_dict()["state"][0]
    assert state["exp_avg"].dtype == np.float64
    # The per-parameter count stays an integer and keeps counting.
    assert state["step"].dtype == np.int64
    assert int(state["step"]) == 2


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_uneven_step_counts_survive_a_conversion(device: str) -> None:
    """Each parameter keeps its own count across the rebuild."""
    run = _Run("Adam", {"lr": 0.05}, device)
    run.model.weight.requires_grad_(False)
    run.step()  # only the bias moves: steps (weight, bias) = (0, 1)
    run.model.weight.requires_grad_(True)
    run.step()  # (1, 2)
    _module_move(run)
    run.step()  # (2, 3)
    state = run.opt.state_dict()["state"]
    assert [int(state[i]["step"]) for i in (0, 1)] == [2, 3]


_ALL_ENGINES: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.05, "momentum": 0.9}),
    ("Adam", {"lr": 0.05}),
    ("AdamW", {"lr": 0.05}),
    ("RMSprop", {"lr": 0.05}),
    ("Adagrad", {"lr": 0.05}),
    ("Adadelta", {"lr": 1.0}),
    ("Adamax", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("NAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05}),
    ("Rprop", {"lr": 0.05}),
]


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize(("name", "kwargs"), _ALL_ENGINES, ids=[n for n, _ in _ALL_ENGINES])
def test_every_engine_carries_its_state_across_a_move(
    name: str, kwargs: dict[str, Any], device: str
) -> None:
    want = _uninterrupted(name, kwargs, device)
    run = _Run(name, kwargs, device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = [run.step()]
        _module_move(run)
        got += [run.step() for _ in range(_STEPS - 1)]
    np.testing.assert_allclose(np.stack(got), want, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("SGD", {"lr": 0.05}), *_OPTIMIZERS],
    ids=["SGD-plain", *_OPT_IDS],
)
def test_rebuild_with_nothing_to_carry_does_not_warn(name: str, kwargs: dict[str, Any]) -> None:
    """No state yet — a step() before any backward — is nothing lost."""
    run = _Run(name, kwargs, "cpu")
    run.opt.step()
    run.model.double()
    run.dtype = lucid.float64
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run.step()
        run.step()


class _HooklessEngine:
    """An engine optimizer that keeps state but cannot export it."""

    def __init__(self, params: list[Any]) -> None:
        self.params = params

    def step(self) -> None:
        """Nothing to do: the test only rebuilds it."""

    def state_buffers(self) -> list[tuple[str, list[Any]]]:
        return []


class _HooklessOptimizer(optim.Optimizer):
    def __init__(self, params: list[nn.Parameter]) -> None:
        super().__init__(params, {"lr": 0.1})

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        params: list[nn.Parameter] = group["params"]  # type: ignore[assignment]
        self._engines.append(_HooklessEngine([p._impl for p in params]))

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        for engine in self._engine_optims:
            engine.step()  # type: ignore[attr-defined]


def test_engine_without_state_hooks_warns_once() -> None:
    model = nn.Linear(2, 1)
    opt = _HooklessOptimizer(list(model.parameters()))
    opt.step()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.double()
        opt.step()
        model.float()
        opt.step()
    resets = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(resets) == 1
    assert "state restarted" in str(resets[0].message)


def test_engine_free_optimizer_is_untouched() -> None:
    """LBFGS steps the parameters itself and rebinds them every step."""
    p = nn.Parameter(lucid.tensor([3.0, -2.0]))
    opt = optim.LBFGS([p], lr=1.0, max_iter=20)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    np.testing.assert_allclose(p.detach().numpy(), [0.0, 0.0], atol=1e-4)


# ── against the reference ────────────────────────────────────────────────────

_FLAG_PATHS = [
    "requires_grad_-redundant",
    "requires_grad=-redundant",
    "requires_grad_-toggle",
    "requires_grad=-toggle",
    "module.requires_grad_",
]


def _ref_trajectory(
    ref: Any, name: str, kwargs: dict[str, Any], toggle: Callable[[Any], None]
) -> np.ndarray:
    model = ref.nn.Linear(3, 2)
    with ref.no_grad():
        model.weight.copy_(ref.tensor(_W0))
        model.bias.copy_(ref.tensor(_B0))
    opt = getattr(ref.optim, name)(model.parameters(), **kwargs)
    x = ref.tensor(_X)
    out = []
    for i in range(_STEPS):
        opt.zero_grad()
        (model(x) ** 2).sum().backward()
        opt.step()
        out.append(model.weight.detach().double().numpy().copy())
        if i in (0, 1):
            toggle(model)
    return np.stack(out)


@pytest.mark.parity
@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("path", _FLAG_PATHS, ids=_FLAG_PATHS)
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [*_OPTIMIZERS, ("RMSprop", {"lr": 0.05})],
    ids=[*_OPT_IDS, "RMSprop"],
)
def test_flag_changes_follow_the_reference(
    name: str, kwargs: dict[str, Any], path: str, device: str, ref: Any
) -> None:
    def toggle(model: Any) -> None:
        for p in model.parameters():
            p.requires_grad_(False)
            p.requires_grad_(True)

    want = _ref_trajectory(ref, name, kwargs, toggle)
    got, _ = _trajectory(name, kwargs, device, _PATHS[path])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_wgan_critic_toggle_follows_the_reference(device: str, ref: Any) -> None:
    """The WGAN loop: an RMSprop critic frozen for every generator step.

    Freezing used to swap the critic's impls, so after the first iteration
    neither its own optimizer nor anything else moved it.
    """
    rng = np.random.default_rng(0)
    g0 = rng.standard_normal((4, 2)).astype(np.float32) * 0.5
    d0 = rng.standard_normal((1, 4)).astype(np.float32) * 0.5
    real = rng.standard_normal((8, 4)).astype(np.float32)
    noise = rng.standard_normal((8, 2)).astype(np.float32)

    def run(lib: Any, tensor: Callable[[np.ndarray], Any]) -> np.ndarray:
        gen = lib.nn.Linear(2, 4, bias=False)
        critic = lib.nn.Linear(4, 1, bias=False)
        gen.weight = lib.nn.Parameter(tensor(g0))
        critic.weight = lib.nn.Parameter(tensor(d0))
        opt_g = lib.optim.RMSprop(gen.parameters(), lr=0.01)
        opt_d = lib.optim.RMSprop(critic.parameters(), lr=0.01)
        x_real, z = tensor(real), tensor(noise)
        trace = []
        for _ in range(4):
            for p in critic.parameters():
                p.requires_grad = True
            opt_d.zero_grad()
            fake = gen(z).detach()
            (critic(fake).mean() - critic(x_real).mean()).backward()
            opt_d.step()
            for p in critic.parameters():
                p.requires_grad = False
            opt_g.zero_grad()
            (-critic(gen(z)).mean()).backward()
            opt_g.step()
            trace.append(np.concatenate([_host(critic.weight).ravel(), _host(gen.weight).ravel()]))
        return np.stack(trace)

    def _host(t: Any) -> np.ndarray:
        if isinstance(t, lucid.Tensor):
            return t.detach().to("cpu").numpy().astype(np.float64)
        return t.detach().double().numpy()

    want = run(ref, lambda a: ref.tensor(a))
    got = run(lucid, lambda a: lucid.tensor(a, device=device))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)
