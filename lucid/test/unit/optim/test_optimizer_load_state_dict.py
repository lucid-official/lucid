"""``Optimizer.load_state_dict`` is a replacement, and a checked one (LCD-722).

The loader used to *merge*: state the checkpoint did not mention survived, so
a checkpoint taken before the first step could not reset an optimizer, and a
parameter the checkpoint had no entry for kept the state of the run it was
loaded over.  It compared the number of groups and nothing else, so a
checkpoint whose groups held other parameters loaded its hyper-parameters
onto the wrong ones; it refused a buffer saved in another precision; and a
refused checkpoint had already overwritten the learning rate.

These pin the contract, for every optimizer, against the reference framework:

* **replacement** — a checkpoint taken before any step resets a stepped
  optimizer, and a parameter absent from a partial checkpoint starts afresh;
* **group sizes** — a group of another size is refused with the reference's
  message, however the parameters are redistributed;
* **dtype** — floating-point state is converted to its parameter's dtype;
* **atomicity** — a refused checkpoint leaves the groups, the state and the
  next step exactly as they were.
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
from lucid.test._fixtures.devices import device_dtype_params

# A matrix and a vector: two parameters, so a checkpoint can cover one of them.
_SHAPES = ((3, 4), (4,))
_TOL = 1e-5

# Every engine optimizer, with the options that decide which state it keeps.
_CASES: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
    ("Adam", {"lr": 0.05}),
    ("Adam", {"lr": 0.05, "amsgrad": True}),
    ("AdamW", {"lr": 0.05}),
    ("RMSprop", {"lr": 0.01}),
    ("RMSprop", {"lr": 0.01, "momentum": 0.9, "centered": True}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.05}),
    ("Adadelta", {"lr": 0.5}),
    ("Adamax", {"lr": 0.05}),
    ("NAdam", {"lr": 0.05}),
    ("RAdam", {"lr": 0.05}),
    ("ASGD", {"lr": 0.05, "t0": 2.0}),
    ("Rprop", {"lr": 0.05}),
]


def _case_id(case: tuple[str, dict[str, Any]]) -> str:
    name, kw = case
    return name + "".join(f"-{k}={v}" for k, v in kw.items())


_CASE_PARAMS = [pytest.param(c, id=_case_id(c)) for c in _CASES]


def _arrays(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=np.float32) for s in _SHAPES]


def _as_array(v: object) -> np.ndarray:
    numpy = getattr(v, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else v)


def _lucid_params(arrays: list[np.ndarray], device: str = "cpu") -> list[nn.Parameter]:
    return [nn.Parameter(lucid.tensor(a.copy()).to(device)) for a in arrays]


def _feed(
    params: list[nn.Parameter], grads: list[np.ndarray], device: str, only: tuple[int, ...] = ()
) -> None:
    """Give the parameters their gradients for one step; ``only`` restricts which."""
    for i, (p, g) in enumerate(zip(params, grads, strict=True)):
        p.grad = lucid.tensor(g.copy()).to(device) if not only or i in only else None


def _rewind(params: list[nn.Parameter], arrays: list[np.ndarray], device: str) -> None:
    """Put the parameters back to ``arrays`` in place, as a restart from a checkpoint does."""
    with lucid.no_grad():
        for p, a in zip(params, arrays, strict=True):
            p.copy_(lucid.tensor(a.copy()).to(device))


class _Pair:
    """The same optimizer in Lucid and in the reference framework, stepped together."""

    def __init__(self, case: tuple[str, dict[str, Any]], ref: Any, device: str = "cpu") -> None:
        name, kw = case
        self.ref = ref
        self.device = device
        self.init = _arrays(0)
        self.params = _lucid_params(self.init, device)
        self.opt = getattr(optim, name)(self.params, **kw)
        self.ref_params = [ref.nn.Parameter(ref.from_numpy(a.copy())) for a in self.init]
        self.ref_opt = getattr(ref.optim, name)(self.ref_params, **kw)

    def step(self, seed: int, only: tuple[int, ...] = ()) -> None:
        grads = _arrays(100 + seed)
        _feed(self.params, grads, self.device, only)
        self.opt.step()
        for i, (rp, g) in enumerate(zip(self.ref_params, grads, strict=True)):
            rp.grad = self.ref.from_numpy(g.copy()) if not only or i in only else None
        self.ref_opt.step()

    def rewind(self, arrays: list[np.ndarray]) -> None:
        _rewind(self.params, arrays, self.device)
        with self.ref.no_grad():
            for rp, a in zip(self.ref_params, arrays, strict=True):
                rp.copy_(self.ref.from_numpy(a.copy()))

    def load(self, saved: tuple[dict[str, Any], dict[str, Any]]) -> None:
        self.opt.load_state_dict(copy.deepcopy(saved[0]))
        self.ref_opt.load_state_dict(copy.deepcopy(saved[1]))

    def save(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return copy.deepcopy(self.opt.state_dict()), copy.deepcopy(self.ref_opt.state_dict())

    def assert_agree(self, what: str) -> None:
        """Parameters and optimizer state are the reference's."""
        for i, (p, rp) in enumerate(zip(self.params, self.ref_params, strict=True)):
            np.testing.assert_allclose(
                p.numpy(), rp.detach().numpy(), rtol=_TOL, atol=_TOL, err_msg=f"{what}: param {i}"
            )
        self.assert_state_agrees(what)

    def assert_state_agrees(self, what: str) -> None:
        """Every state entry the reference holds is Lucid's, to the same values."""
        state, ref_state = self.opt.state_dict()["state"], self.ref_opt.state_dict()["state"]
        assert sorted(state) == sorted(ref_state), f"{what}: state keys"
        for idx, ref_entry in ref_state.items():
            assert set(state[idx]) == set(ref_entry), f"{what}: entry {idx}"
            for key, want in ref_entry.items():
                got = _as_array(state[idx][key])
                np.testing.assert_allclose(
                    got.astype(np.float64),
                    want.detach().numpy().astype(np.float64),
                    rtol=_TOL,
                    atol=_TOL,
                    err_msg=f"{what}: state {idx}/{key}",
                )


# ── replacement ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", _CASE_PARAMS)
def test_a_checkpoint_before_the_first_step_resets_the_optimizer(
    case: tuple[str, dict[str, Any]], device: str, ref: Any
) -> None:
    pair = _Pair(case, ref, device)
    fresh = pair.save()
    for k in range(3):
        pair.step(k)
    pair.load(fresh)
    pair.rewind(pair.init)
    for k in range(3, 5):
        pair.step(k)
    pair.assert_agree("after two steps from the reset")


@pytest.mark.parametrize("case", _CASE_PARAMS)
def test_a_parameter_the_checkpoint_has_no_entry_for_starts_afresh(
    case: tuple[str, dict[str, Any]], device: str, ref: Any
) -> None:
    pair = _Pair(case, ref, device)
    for k in range(3):
        pair.step(k, only=(0,))
    partial = pair.save()
    assert set(partial[0]["state"]) <= {0}
    for k in range(3, 5):
        pair.step(k)
    pair.load(partial)
    assert set(pair.opt.state_dict()["state"]) <= {0}, "the second parameter kept its state"
    pair.rewind(pair.init)
    for k in range(5, 7):
        pair.step(k)
    pair.assert_agree("after two steps from the partial checkpoint")


@pytest.mark.parametrize("case", _CASE_PARAMS)
def test_loading_into_an_optimizer_that_never_stepped_continues_the_run(
    case: tuple[str, dict[str, Any]], device: str, ref: Any
) -> None:
    pair = _Pair(case, ref, device)
    for k in range(3):
        pair.step(k)
    saved = pair.save()
    # A twin built over the same values has not stepped, and has no engines yet.
    name, kw = case
    values = [p.numpy() for p in pair.params]
    twin_params = _lucid_params(values, device)
    twin = getattr(optim, name)(twin_params, **kw)
    twin.load_state_dict(copy.deepcopy(saved[0]))
    pair.step(3)
    _feed(twin_params, _arrays(103), device)
    twin.step()
    for a, b in zip(pair.params, twin_params, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_loading_replaces_the_state_of_a_run_that_resumed_from_a_pickle() -> None:
    # An optimizer restored from a pickle holds its state in ``state`` until its
    # first step builds the engines; a checkpoint loaded over it must win.
    params = _lucid_params(_arrays(0))
    opt = optim.Adam(params, lr=0.05)
    for k in range(3):
        _feed(params, _arrays(100 + k), "cpu")
        opt.step()
    stepped = copy.deepcopy(opt)
    fresh = optim.Adam(_lucid_params(_arrays(0)), lr=0.05).state_dict()
    stepped.load_state_dict(fresh)
    assert stepped.state_dict()["state"] == {}


# ── group sizes ────────────────────────────────────────────────────────────

_GROUPINGS = [
    pytest.param([[0, 1], [2, 3]], [[0], [1, 2, 3]], id="[2,2]->[1,3]"),
    pytest.param([[0, 1, 2, 3]], [[0, 1]], id="4->2"),
    pytest.param([[0, 1]], [[0, 1, 2, 3]], id="2->4"),
    pytest.param([[0, 1], [2, 3]], [[0, 1, 2, 3]], id="two groups->one"),
]


def _four_params() -> list[nn.Parameter]:
    rng = np.random.default_rng(7)
    return [nn.Parameter(lucid.tensor(rng.standard_normal(3).astype(np.float32))) for _ in range(4)]


def _grouped(name: str, kw: dict[str, Any], layout: list[list[int]]) -> optim.Optimizer:
    params = _four_params()
    groups = [{"params": [params[i] for i in g], "lr": 0.1 * (n + 1)} for n, g in enumerate(layout)]
    return getattr(optim, name)(groups, **{k: v for k, v in kw.items() if k != "lr"})


def _ref_grouped(ref: Any, name: str, kw: dict[str, Any], layout: list[list[int]]) -> Any:
    params = [ref.nn.Parameter(ref.zeros(3)) for _ in range(4)]
    groups = [{"params": [params[i] for i in g], "lr": 0.1 * (n + 1)} for n, g in enumerate(layout)]
    return getattr(ref.optim, name)(groups, **{k: v for k, v in kw.items() if k != "lr"})


@pytest.mark.parametrize("case", _CASE_PARAMS)
@pytest.mark.parametrize(("saved_layout", "live_layout"), _GROUPINGS)
def test_a_group_of_another_size_is_refused_with_the_references_message(
    case: tuple[str, dict[str, Any]],
    saved_layout: list[list[int]],
    live_layout: list[list[int]],
    ref: Any,
) -> None:
    name, kw = case
    saved = _grouped(name, kw, saved_layout).state_dict()
    live = _grouped(name, kw, live_layout)
    ref_live = _ref_grouped(ref, name, kw, live_layout)
    ref_saved = _ref_grouped(ref, name, kw, saved_layout).state_dict()
    with pytest.raises(ValueError) as ref_err:
        ref_live.load_state_dict(ref_saved)
    before = [g["lr"] for g in live.param_groups]
    with pytest.raises(ValueError) as err:
        live.load_state_dict(saved)
    assert str(ref_err.value) in str(err.value)
    assert [g["lr"] for g in live.param_groups] == before, "a refused load changed a group"


# ── dtype ──────────────────────────────────────────────────────────────────

_NP = {lucid.float16: np.float16, lucid.float32: np.float32, lucid.float64: np.float64}
_CROSSINGS = [
    p
    for p in device_dtype_params(
        [lucid.float32, lucid.float64, lucid.float16], [lucid.float32, lucid.float64, lucid.float16]
    )
    if p.values[1] != p.values[2]
]


@pytest.mark.parametrize("case", _CASE_PARAMS)
@pytest.mark.parametrize("device, saved_dtype, live_dtype", _CROSSINGS)
def test_state_saved_in_another_precision_follows_its_parameter(
    case: tuple[str, dict[str, Any]], device: str, saved_dtype: Any, live_dtype: Any, ref: Any
) -> None:
    name, kw = case
    init = _arrays(0)
    src = nn.Parameter(lucid.tensor(init[0], dtype=saved_dtype).to(device))
    ref_src = ref.nn.Parameter(ref.from_numpy(init[0].astype(_NP[saved_dtype])))
    opt, ref_opt = getattr(optim, name)([src], **kw), getattr(ref.optim, name)([ref_src], **kw)
    for k in range(3):
        g = _arrays(100 + k)[0].astype(_NP[saved_dtype])
        src.grad = lucid.tensor(g.copy(), dtype=saved_dtype).to(device)
        opt.step()
        ref_src.grad = ref.from_numpy(g.copy())
        ref_opt.step()

    dst = nn.Parameter(lucid.tensor(init[0], dtype=live_dtype).to(device))
    ref_dst = ref.nn.Parameter(ref.from_numpy(init[0].astype(_NP[live_dtype])))
    live, ref_live = getattr(optim, name)([dst], **kw), getattr(ref.optim, name)([ref_dst], **kw)
    live.load_state_dict(copy.deepcopy(opt.state_dict()))
    ref_live.load_state_dict(copy.deepcopy(ref_opt.state_dict()))

    state, ref_state = live.state_dict()["state"], ref_live.state_dict()["state"]
    assert sorted(state) == sorted(ref_state)
    tol = 1e-2 if lucid.float16 in (saved_dtype, live_dtype) else 1e-5
    for idx, ref_entry in ref_state.items():
        assert set(state[idx]) == set(ref_entry)
        for key, want in ref_entry.items():
            got = _as_array(state[idx][key])
            if key != "step":
                assert got.dtype == _NP[live_dtype] or got.ndim == 0, (key, got.dtype)
            np.testing.assert_allclose(
                got.astype(np.float64),
                want.detach().numpy().astype(np.float64),
                rtol=tol,
                atol=tol,
                err_msg=f"{key}",
            )

    g = _arrays(200)[0].astype(_NP[live_dtype])
    dst.grad = lucid.tensor(g.copy(), dtype=live_dtype).to(device)
    live.step()
    ref_dst.grad = ref.from_numpy(g.copy())
    ref_live.step()
    np.testing.assert_allclose(
        dst.numpy().astype(np.float64),
        ref_dst.detach().numpy().astype(np.float64),
        rtol=tol * 10,
        atol=tol * 10,
    )


# ── atomicity ──────────────────────────────────────────────────────────────


def _stepped(case: tuple[str, dict[str, Any]], steps: int = 2) -> tuple[Any, list[nn.Parameter]]:
    """An optimizer over two groups, one parameter each, stepped ``steps`` times."""
    name, kw = case
    params = _lucid_params(_arrays(0))
    groups = [{"params": [params[0]], "lr": 0.1}, {"params": [params[1]], "lr": 0.2}]
    opt = getattr(optim, name)(groups, **{k: v for k, v in kw.items() if k != "lr"})
    for k in range(steps):
        _feed(params, _arrays(100 + k), "cpu")
        opt.step()
    return opt, params


def _hyperparams(opt: optim.Optimizer) -> list[dict[str, object]]:
    return [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]


def _state_arrays(opt: optim.Optimizer) -> dict[int, dict[str, np.ndarray]]:
    return {
        i: {k: _as_array(v).copy() for k, v in entry.items()}
        for i, entry in opt.state_dict()["state"].items()
    }


def _break_hyperparam(sd: dict[str, Any]) -> type[Exception]:
    sd["param_groups"][1]["lr"] = -1.0
    return ValueError


def _break_group_size(sd: dict[str, Any]) -> type[Exception]:
    sd["param_groups"][1]["params"] = [1, 2]
    return ValueError


def _break_group_count(sd: dict[str, Any]) -> type[Exception]:
    del sd["param_groups"][1]
    return ValueError


def _break_buffer_shape(sd: dict[str, Any]) -> type[Exception]:
    entry = sd["state"][1]
    key = next(k for k in entry if k not in ("step", "eta", "mu", "mu_product"))
    entry[key] = np.zeros((99,), dtype=np.float32)
    return _C_engine.ShapeMismatch


_BREAKS = [
    pytest.param(_break_hyperparam, id="hyper-parameter"),
    pytest.param(_break_group_size, id="group-size"),
    pytest.param(_break_group_count, id="group-count"),
    pytest.param(_break_buffer_shape, id="buffer-shape"),
]
# SGD without momentum keeps no buffer to break.
_STATEFUL = [p for p in _CASE_PARAMS if p.values[0] != ("SGD", {"lr": 0.1})]


@pytest.mark.parametrize("case", _STATEFUL)
@pytest.mark.parametrize("break_it", _BREAKS)
def test_a_refused_checkpoint_leaves_the_optimizer_as_it_was(
    case: tuple[str, dict[str, Any]], break_it: Callable[[dict[str, Any]], type[Exception]]
) -> None:
    opt, params = _stepped(case)
    twin, twin_params = _stepped(case)
    saved = copy.deepcopy(opt.state_dict())
    # Every other group and buffer of the checkpoint is valid and different, so a
    # load that applied any of it before failing would show.
    bad = copy.deepcopy(saved)
    bad["param_groups"][0]["lr"] = 0.5
    for entry in bad["state"].values():
        for key, value in entry.items():
            if key != "step":
                entry[key] = np.asarray(np.asarray(value) + 3.0)
    error = break_it(bad)

    hyper, state = _hyperparams(opt), _state_arrays(opt)
    with pytest.raises(error):
        opt.load_state_dict(bad)

    assert _hyperparams(opt) == hyper
    after = _state_arrays(opt)
    assert after.keys() == state.keys()
    for idx, entry in state.items():
        for key, want in entry.items():
            np.testing.assert_array_equal(after[idx][key], want, err_msg=f"{idx}/{key}")
    _feed(params, _arrays(300), "cpu")
    opt.step()
    _feed(twin_params, _arrays(300), "cpu")
    twin.step()
    for a, b in zip(params, twin_params, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("case", _STATEFUL)
@pytest.mark.parametrize("break_it", _BREAKS)
def test_a_refused_checkpoint_leaves_an_optimizer_that_never_stepped_untouched(
    case: tuple[str, dict[str, Any]], break_it: Callable[[dict[str, Any]], type[Exception]]
) -> None:
    source, _ = _stepped(case)
    bad = copy.deepcopy(source.state_dict())
    bad["param_groups"][0]["lr"] = 0.5
    error = break_it(bad)
    opt, _ = _stepped(case, steps=0)
    hyper = _hyperparams(opt)
    with pytest.raises(error):
        opt.load_state_dict(bad)
    assert _hyperparams(opt) == hyper
    assert opt.state_dict()["state"] == {}


# ── the checkpoint's hyper-parameters apply ────────────────────────────────

# (the optimizer that wrote the checkpoint, the one that reads it): the reader
# is built with other options, and takes the writer's.
_HYPER = [
    pytest.param(
        ("Adam", {"lr": 0.01, "betas": (0.8, 0.9)}), ("Adam", {"lr": 0.05}), id="Adam-betas"
    ),
    pytest.param(
        ("NAdam", {"lr": 0.01, "momentum_decay": 1e-2}),
        ("NAdam", {"lr": 0.05}),
        id="NAdam-momentum_decay",
    ),
    pytest.param(
        ("SGD", {"lr": 0.02, "momentum": 0.5, "nesterov": True}),
        ("SGD", {"lr": 0.1, "momentum": 0.9}),
        id="SGD-momentum",
    ),
    pytest.param(
        ("RMSprop", {"lr": 0.005, "alpha": 0.8, "momentum": 0.5}),
        ("RMSprop", {"lr": 0.01, "momentum": 0.9}),
        id="RMSprop-alpha",
    ),
]


@pytest.mark.parametrize(("written_by", "read_by"), _HYPER)
def test_the_checkpoints_hyperparameters_replace_the_readers(
    written_by: tuple[str, dict[str, Any]], read_by: tuple[str, dict[str, Any]], ref: Any
) -> None:
    writer, reader = _Pair(written_by, ref), _Pair(read_by, ref)
    for k in range(3):
        writer.step(k)
    reader.load(writer.save())
    reader.rewind([p.numpy() for p in writer.params])
    for k in range(3, 5):
        reader.step(k)
    reader.assert_agree("after two steps from the checkpoint")


# ── optimizers that keep their state in Python ─────────────────────────────

_PYTHON_SIDE = [
    pytest.param("SparseAdam", {"lr": 0.1}, id="SparseAdam"),
    pytest.param("LBFGS", {"lr": 1.0}, id="LBFGS"),
]


def _flat(name: str, kw: dict[str, Any], n: int) -> optim.Optimizer:
    return getattr(optim, name)([nn.Parameter(lucid.zeros(4, 3)) for _ in range(n)], **kw)


@pytest.mark.parametrize(("name", "kw"), _PYTHON_SIDE)
def test_a_group_of_another_size_is_refused_by_python_side_optimizers(
    name: str, kw: dict[str, Any]
) -> None:
    saved = _flat(name, kw, 3).state_dict()
    live = _flat(name, kw, 2)
    with pytest.raises(ValueError, match="doesn't match the size of optimizer's group"):
        live.load_state_dict(saved)


@pytest.mark.parametrize(("name", "kw"), _PYTHON_SIDE)
def test_a_rejected_hyperparameter_leaves_python_side_optimizers_untouched(
    name: str, kw: dict[str, Any]
) -> None:
    saved = _flat(name, kw, 2).state_dict()
    saved["param_groups"][0]["lr"] = -1.0
    live = _flat(name, kw, 2)
    hyper = _hyperparams(live)
    with pytest.raises(ValueError):
        live.load_state_dict(saved)
    assert _hyperparams(live) == hyper


def test_lbfgs_checkpoint_before_the_first_step_resets_its_history() -> None:
    p = nn.Parameter(lucid.tensor([1.0, 2.0]))
    opt = optim.LBFGS([p], lr=1.0, max_iter=3)
    fresh = copy.deepcopy(opt.state_dict())

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.state_dict()["state"] != {}
    opt.load_state_dict(fresh)
    assert opt.state_dict()["state"] == {}
