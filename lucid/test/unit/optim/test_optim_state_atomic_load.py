"""Atomic validation and declarative slot round-trip tests for C++ optimizers (AE-6 / LCD-239)."""

import copy
from typing import Callable

import numpy as np
import pytest

import lucid
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._tensor import Tensor

OptimizerFactory = Callable[[list[Tensor]], optim.Optimizer]


CPP_OPTIMIZER_SPECS: list[tuple[str, OptimizerFactory, list[str], list[str]]] = [
    (
        "SGD_momentum",
        lambda params: optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=1e-4),
        [],
        ["momentum_buffer"],
    ),
    (
        "ASGD",
        lambda params: optim.ASGD(params, lr=0.05, lambd=1e-3, alpha=0.75, t0=10.0),
        ["step", "eta", "mu"],
        ["ax"],
    ),
    ("Adam", lambda params: optim.Adam(params, lr=0.01), ["step"], ["exp_avg", "exp_avg_sq"]),
    (
        "Adam_amsgrad",
        lambda params: optim.Adam(params, lr=0.01, amsgrad=True),
        ["step"],
        ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
    ),
    (
        "AdamW_amsgrad",
        lambda params: optim.AdamW(params, lr=0.01, weight_decay=0.02, amsgrad=True),
        ["step"],
        ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
    ),
    (
        "NAdam",
        lambda params: optim.NAdam(params, lr=0.01),
        ["step", "mu_product"],
        ["exp_avg", "exp_avg_sq"],
    ),
    ("RAdam", lambda params: optim.RAdam(params, lr=0.01), ["step"], ["exp_avg", "exp_avg_sq"]),
    ("Adamax", lambda params: optim.Adamax(params, lr=0.01), ["step"], ["exp_avg", "exp_inf"]),
    ("Adagrad", lambda params: optim.Adagrad(params, lr=0.01), ["step"], ["sum"]),
    (
        "Adadelta",
        lambda params: optim.Adadelta(params, lr=0.5),
        ["step"],
        ["square_avg", "acc_delta"],
    ),
    (
        "RMSprop_full",
        lambda params: optim.RMSprop(params, lr=0.01, momentum=0.9, centered=True),
        ["step"],
        ["square_avg", "momentum_buffer", "grad_avg"],
    ),
    ("Rprop", lambda params: optim.Rprop(params, lr=0.01), ["step"], ["prev", "step_size"]),
]


def _make_params() -> list[Tensor]:
    p0 = Tensor(
        np.array([[1.25, -0.5, 0.75], [0.25, -1.5, 2.0]], dtype=np.float32), requires_grad=True
    )
    p1 = Tensor(np.array([-0.8, 0.4, 1.1, -0.3], dtype=np.float32), requires_grad=True)
    return [p0, p1]


def _step_once(opt: optim.Optimizer, params: list[Tensor], scale: float = 1.0) -> None:
    opt.zero_grad()
    loss = (params[0] * (1.5 * scale)).sum() + (params[1] * (-0.7 * scale)).sum()
    loss.backward()
    opt.step()


def _assert_state_dict_equal(actual: dict, expected: dict) -> None:
    assert actual["state"].keys() == expected["state"].keys()
    for idx in expected["state"]:
        act_slot = actual["state"][idx]
        exp_slot = expected["state"][idx]
        assert act_slot.keys() == exp_slot.keys()
        for key, exp_val in exp_slot.items():
            act_val = act_slot[key]
            if isinstance(exp_val, Tensor):
                assert isinstance(act_val, Tensor)
                np.testing.assert_array_equal(act_val.numpy(), exp_val.numpy())
            elif isinstance(exp_val, np.ndarray):
                np.testing.assert_array_equal(np.asarray(act_val), exp_val)
            else:
                assert act_val == exp_val


def _PerturbedStateDict(sd: dict) -> dict:
    """Return a deep copy of state_dict where every tensor/scalar has been noticeably altered."""
    mutated = copy.deepcopy(sd)
    for slot in mutated["state"].values():
        for k, v in list(slot.items()):
            if isinstance(v, Tensor):
                slot[k] = Tensor(v.numpy() + 42.0, dtype=v.dtype, device=v.device)
            elif isinstance(v, np.ndarray):
                if np.issubdtype(v.dtype, np.integer):
                    slot[k] = np.asarray(v + 7, dtype=v.dtype)
                else:
                    slot[k] = np.asarray(v + 3.5, dtype=v.dtype)
            elif isinstance(v, (int, float)):
                slot[k] = v + 7
    return mutated


@pytest.mark.parametrize(
    ("name", "factory", "scalar_keys", "buffer_keys"),
    CPP_OPTIMIZER_SPECS,
    ids=[spec[0] for spec in CPP_OPTIMIZER_SPECS],
)
class TestOptimizerDeclarativeAtomicState:
    def test_state_dict_roundtrip_and_trajectory_parity(
        self, name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
    ) -> None:
        params_a = _make_params()
        opt_a = factory(params_a)
        _step_once(opt_a, params_a, scale=1.0)
        _step_once(opt_a, params_a, scale=1.3)

        saved = copy.deepcopy(opt_a.state_dict())
        for slot in saved["state"].values():
            for sk in scalar_keys:
                assert sk in slot, f"{name} missing scalar slot {sk!r}"
            for bk in buffer_keys:
                assert bk in slot, f"{name} missing buffer slot {bk!r}"

        params_b = [Tensor(p.numpy().copy(), requires_grad=True) for p in params_a]
        opt_b = factory(params_b)
        opt_b.load_state_dict(saved)

        _assert_state_dict_equal(opt_b.state_dict(), opt_a.state_dict())

        _step_once(opt_a, params_a, scale=0.8)
        _step_once(opt_b, params_b, scale=0.8)

        for pa, pb in zip(params_a, params_b):
            np.testing.assert_array_equal(pa.numpy(), pb.numpy())
        _assert_state_dict_equal(opt_b.state_dict(), opt_a.state_dict())

    def test_corrupted_later_buffer_shape_leaves_earlier_slots_and_params_untouched(
        self, name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
    ) -> None:
        params = _make_params()
        opt = factory(params)
        _step_once(opt, params, scale=1.0)
        before = copy.deepcopy(opt.state_dict())

        # Perturb all earlier slots and param 0 so any partial write before failure would be detected.
        bad_sd = _PerturbedStateDict(before)
        target_buf_key = buffer_keys[-1]
        bad_sd["state"][1][target_buf_key] = Tensor(np.zeros((99,), dtype=np.float32))

        with pytest.raises(_C_engine.ShapeMismatch):
            opt.load_state_dict(bad_sd)

        _assert_state_dict_equal(opt.state_dict(), before)

        # Verify subsequent step matches an optimizer that never saw the failed load.
        ref_params = [Tensor(p.numpy().copy(), requires_grad=True) for p in params]
        ref_opt = factory(ref_params)
        ref_opt.load_state_dict(before)

        _step_once(opt, params, scale=1.2)
        _step_once(ref_opt, ref_params, scale=1.2)
        for p, rp in zip(params, ref_params):
            np.testing.assert_array_equal(p.numpy(), rp.numpy())

    def test_corrupted_later_buffer_dtype_leaves_earlier_slots_and_params_untouched(
        self, name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
    ) -> None:
        params = _make_params()
        opt = factory(params)
        _step_once(opt, params, scale=1.0)
        before = copy.deepcopy(opt.state_dict())

        bad_sd = _PerturbedStateDict(before)
        target_buf_key = buffer_keys[-1]
        orig_arr = np.asarray(before["state"][1][target_buf_key])
        bad_sd["state"][1][target_buf_key] = orig_arr.astype(np.float64)

        with pytest.raises(_C_engine.DtypeMismatch):
            opt.load_state_dict(bad_sd)

        _assert_state_dict_equal(opt.state_dict(), before)

    def test_corrupted_scalar_slot_leaves_all_state_untouched(
        self, name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
    ) -> None:
        if not scalar_keys:
            pytest.skip(f"{name} has no scalar state slots")

        params = _make_params()
        opt = factory(params)
        _step_once(opt, params, scale=1.0)
        before = copy.deepcopy(opt.state_dict())

        bad_sd = _PerturbedStateDict(before)
        target_scalar_key = scalar_keys[-1]
        # Replace scalar in param 1 with a 1-D array of length 2 (not a 1-element scalar).
        bad_sd["state"][1][target_scalar_key] = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(_C_engine.LucidError):
            opt.load_state_dict(bad_sd)

        _assert_state_dict_equal(opt.state_dict(), before)

    def test_failed_load_on_unstepped_optimizer_does_not_initialize_slots(
        self, name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
    ) -> None:
        src_params = _make_params()
        src_opt = factory(src_params)
        _step_once(src_opt, src_params, scale=1.0)
        bad_sd = _PerturbedStateDict(src_opt.state_dict())
        bad_sd["state"][1][buffer_keys[-1]] = Tensor(np.zeros((17,), dtype=np.float32))

        fresh_params = _make_params()
        fresh_opt = factory(fresh_params)
        assert fresh_opt.state_dict()["state"] == {}

        with pytest.raises(_C_engine.ShapeMismatch):
            fresh_opt.load_state_dict(bad_sd)

        assert fresh_opt.state_dict()["state"] == {}


@pytest.mark.parametrize(
    ("name", "factory", "scalar_keys", "buffer_keys"),
    CPP_OPTIMIZER_SPECS,
    ids=[spec[0] for spec in CPP_OPTIMIZER_SPECS],
)
def test_cpp_engine_load_state_buffers_device_mismatch_is_atomic(
    name: str, factory: OptimizerFactory, scalar_keys: list[str], buffer_keys: list[str]
) -> None:
    params = _make_params()
    opt = factory(params)
    _step_once(opt, params, scale=1.0)
    before = copy.deepcopy(opt.state_dict())

    eng = opt._engine_optims[0]
    raw_bufs = eng.state_buffers()

    # Build a mutated NamedBuffers list where earlier buffers are modified validly on CPU,
    # and the very last buffer slot for param 1 is on GPU (causing DeviceMismatch).
    target_buf_key = buffer_keys[-1]
    corrupted_bufs: list[tuple[str, list[_C_engine.TensorImpl | None]]] = []
    for slot_name, tensors in raw_bufs:
        new_tensors = list(tensors)
        if slot_name == target_buf_key:
            # Perturb param 0 on CPU so we can detect if param 0 was written before param 1 failed.
            t0_arr = np.asarray(new_tensors[0].data_as_python()) + 99.0
            new_tensors[0] = Tensor(t0_arr, dtype=lucid.float32, device="cpu")._impl
            # Put param 1 on GPU while parameter lives on CPU.
            t1_arr = np.asarray(new_tensors[1].data_as_python())
            new_tensors[1] = Tensor(t1_arr, dtype=lucid.float32, device="metal")._impl
        corrupted_bufs.append((slot_name, new_tensors))

    with pytest.raises(_C_engine.DeviceMismatch):
        eng.load_state_buffers(corrupted_bufs)

    _assert_state_dict_equal(opt.state_dict(), before)
