"""Guard tests for pickle and copy.deepcopy on stepped optimizers (LCD-245)."""

import copy
import pickle
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._tensor.tensor import Tensor

type _OptimizerFactory = Callable[[list[nn.Parameter]], optim.Optimizer]

_SHAPES: tuple[tuple[int, ...], ...] = ((3, 4), (4, 2))
_N_BEFORE: int = 3
_N_AFTER: int = 2

_OPTIMIZER_SPECS: list[tuple[str, _OptimizerFactory]] = [
    ("SGD_plain", lambda ps: optim.SGD(ps, lr=0.05)),
    (
        "SGD_momentum",
        lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9, weight_decay=1e-4),
    ),
    (
        "SGD_nesterov",
        lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True),
    ),
    ("ASGD", lambda ps: optim.ASGD(ps, lr=0.05, lambd=1e-3, alpha=0.75, t0=2.0)),
    ("Adam", lambda ps: optim.Adam(ps, lr=0.01, weight_decay=1e-4)),
    ("Adam_amsgrad", lambda ps: optim.Adam(ps, lr=0.01, amsgrad=True)),
    ("AdamW", lambda ps: optim.AdamW(ps, lr=0.01, weight_decay=0.02)),
    (
        "AdamW_amsgrad",
        lambda ps: optim.AdamW(ps, lr=0.01, weight_decay=0.02, amsgrad=True),
    ),
    ("Adamax", lambda ps: optim.Adamax(ps, lr=0.01, weight_decay=1e-4)),
    ("NAdam", lambda ps: optim.NAdam(ps, lr=0.01, weight_decay=1e-4)),
    ("RAdam", lambda ps: optim.RAdam(ps, lr=0.01, weight_decay=1e-4)),
    (
        "Adagrad",
        lambda ps: optim.Adagrad(ps, lr=0.05, lr_decay=1e-3, weight_decay=1e-4),
    ),
    ("Adadelta", lambda ps: optim.Adadelta(ps, lr=0.5, rho=0.9, weight_decay=1e-4)),
    ("RMSprop", lambda ps: optim.RMSprop(ps, lr=0.01, momentum=0.9, centered=True)),
    ("Rprop", lambda ps: optim.Rprop(ps, lr=0.01, etas=(0.5, 1.2))),
    ("SparseAdam", lambda ps: optim.SparseAdam(ps, lr=0.01)),
]


def _make_arrays(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.asarray(rng.standard_normal(s), dtype=np.float32) for s in _SHAPES]


def _make_sparse_grads(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for s in _SHAPES:
        arr = np.zeros(s, dtype=np.float32)
        touched = seed % s[0]
        arr[touched] = np.asarray(rng.standard_normal(s[1:]), dtype=np.float32)
        out.append(arr)
    return out


def _make_params(arrays: list[np.ndarray], device: str) -> list[nn.Parameter]:
    return [nn.Parameter(lucid.tensor(a.copy()).to(device)) for a in arrays]


def _feed_grads(
    params: list[Tensor] | list[nn.Parameter],
    grads: list[np.ndarray],
    device: str,
) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = lucid.tensor(g.copy()).to(device)


def _extract_params(opt: optim.Optimizer) -> list[Tensor]:
    out: list[Tensor] = []
    for group in opt.param_groups:
        params = group["params"]
        assert isinstance(params, list)
        out.extend(params)
    return out


def _as_array(v: object) -> np.ndarray:
    if isinstance(v, Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_state_dict_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    act_state = actual["state"]
    exp_state = expected["state"]
    assert isinstance(act_state, dict)
    assert isinstance(exp_state, dict)
    assert act_state.keys() == exp_state.keys()
    for idx, exp_slot in exp_state.items():
        act_slot = act_state[idx]
        assert isinstance(act_slot, dict)
        assert isinstance(exp_slot, dict)
        assert act_slot.keys() == exp_slot.keys()
        for key, exp_val in exp_slot.items():
            np.testing.assert_array_equal(
                _as_array(act_slot[key]),
                _as_array(exp_val),
                err_msg=f"state[{idx}][{key!r}] mismatch",
            )


def _assert_params_equal(actual: list[Tensor], expected: list[Tensor] | list[nn.Parameter]) -> None:
    for i, (pa, pe) in enumerate(zip(actual, expected, strict=True)):
        assert pa.device == pe.device, f"param {i} device mismatch"
        assert pa.dtype == pe.dtype, f"param {i} dtype mismatch"
        np.testing.assert_array_equal(pa.numpy(), pe.numpy(), err_msg=f"param {i} value mismatch")


@pytest.mark.parametrize(
    ("name", "factory"),
    _OPTIMIZER_SPECS,
    ids=[spec[0] for spec in _OPTIMIZER_SPECS],
)
class TestOptimizerPickleAndDeepcopy:
    def test_pickle_roundtrip_after_step_matches_uninterrupted_trajectory(
        self, name: str, factory: _OptimizerFactory, device: str
    ) -> None:
        grad_fn = _make_sparse_grads if name == "SparseAdam" else _make_arrays
        init_arrays = _make_arrays(42)
        grads = [grad_fn(100 + k) for k in range(_N_BEFORE + _N_AFTER)]

        params_orig = _make_params(init_arrays, device)
        opt_orig = factory(params_orig)
        for g in grads[:_N_BEFORE]:
            _feed_grads(params_orig, g, device)
            opt_orig.step()
        opt_orig.zero_grad()

        blob = pickle.dumps(opt_orig)
        opt_restored = pickle.loads(blob)
        params_restored = _extract_params(opt_restored)

        _assert_params_equal(params_restored, params_orig)
        _assert_state_dict_equal(opt_restored.state_dict(), opt_orig.state_dict())

        for g in grads[_N_BEFORE:]:
            _feed_grads(params_orig, g, device)
            opt_orig.step()

            _feed_grads(params_restored, g, device)
            opt_restored.step()

            _assert_params_equal(params_restored, params_orig)
            _assert_state_dict_equal(opt_restored.state_dict(), opt_orig.state_dict())

    def test_deepcopy_after_step_matches_trajectory_and_isolates_original(
        self, name: str, factory: _OptimizerFactory, device: str
    ) -> None:
        grad_fn = _make_sparse_grads if name == "SparseAdam" else _make_arrays
        init_arrays = _make_arrays(77)
        grads = [grad_fn(200 + k) for k in range(_N_BEFORE + _N_AFTER)]

        params_orig = _make_params(init_arrays, device)
        opt_orig = factory(params_orig)
        for g in grads[:_N_BEFORE]:
            _feed_grads(params_orig, g, device)
            opt_orig.step()
        opt_orig.zero_grad()

        snap_params_before = [p.numpy().copy() for p in params_orig]
        snap_sd_before = copy.deepcopy(opt_orig.state_dict())

        # Verify bare copy.deepcopy(opt_orig) succeeds without raising TypeError.
        opt_bare_copy = copy.deepcopy(opt_orig)
        _assert_state_dict_equal(opt_bare_copy.state_dict(), snap_sd_before)

        # Clone parameter storage explicitly in the deepcopy memo because
        # Parameter(param) currently shares underlying buffer storage (see FOUND).
        memo: dict[int, object] = {
            id(p): nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
            for p in params_orig
        }
        opt_copied = copy.deepcopy(opt_orig, memo)
        params_copied = _extract_params(opt_copied)

        for p_orig, p_copy in zip(params_orig, params_copied, strict=True):
            assert p_orig is not p_copy
            assert isinstance(p_copy, nn.Parameter)

        _assert_params_equal(params_copied, params_orig)
        _assert_state_dict_equal(opt_copied.state_dict(), snap_sd_before)

        # Stepping the copy alone must not mutate the original optimizer's params or state.
        _feed_grads(params_copied, grads[_N_BEFORE], device)
        opt_copied.step()
        for p_orig, arr_before in zip(params_orig, snap_params_before, strict=True):
            np.testing.assert_array_equal(p_orig.numpy(), arr_before)
        _assert_state_dict_equal(opt_orig.state_dict(), snap_sd_before)

        # Now step the original with the same gradient and verify they match again.
        _feed_grads(params_orig, grads[_N_BEFORE], device)
        opt_orig.step()
        _assert_params_equal(params_copied, params_orig)
        _assert_state_dict_equal(opt_copied.state_dict(), opt_orig.state_dict())

        for g in grads[_N_BEFORE + 1 :]:
            _feed_grads(params_orig, g, device)
            opt_orig.step()

            _feed_grads(params_copied, g, device)
            opt_copied.step()

            _assert_params_equal(params_copied, params_orig)
            _assert_state_dict_equal(opt_copied.state_dict(), opt_orig.state_dict())
