"""``lucid.optim.swa_utils``: weight averaging, the SWA learning rate and BatchNorm refresh.

``AveragedModel`` holds a deep copy of a model and folds the model's weights into
it (SWA by default, EMA or any rule you pass); ``SWALR`` anneals the learning rate
to the SWA rate; ``update_bn`` re-estimates BatchNorm statistics for an averaged
model.  The parity tests drive Lucid and the reference through the same inputs
and compare every averaged tensor, every learning rate and every statistic.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.optim import swa_utils
from lucid.test._fixtures.devices import device_dtype_params

_DEVICE_DTYPES = device_dtype_params([lucid.float32, lucid.float64])
_TOL = {lucid.float32: 1e-5, lucid.float64: 1e-12}

SURFACE = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]


def _lucid_net(device: str, dtype: lucid.dtype) -> nn.Module:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    return net.double().to(device) if dtype == lucid.float64 else net.to(device)


def _ref_net(ref: Any, source: nn.Module, dtype: lucid.dtype) -> Any:
    net = ref.nn.Sequential(ref.nn.Linear(3, 4), ref.nn.BatchNorm1d(4), ref.nn.Linear(4, 2))
    if dtype == lucid.float64:
        net = net.double()
    net.load_state_dict(
        {k: ref.from_numpy(v.numpy().copy()) for k, v in source.state_dict().items()}
    )
    return net


def _arrays(state: dict[str, lucid.Tensor], seed: int) -> dict[str, np.ndarray]:
    """A new value for every entry of ``state``; counters get a distinct integer."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, value in state.items():
        shape = tuple(value.shape)
        if "num_batches_tracked" in key:
            out[key] = np.asarray(7 * seed + 3, dtype=np.int64)
        else:
            out[key] = np.asarray(rng.standard_normal(shape), dtype=value.numpy().dtype)
    return out


def _put(net: nn.Module, arrays: dict[str, np.ndarray], device: str) -> None:
    net.load_state_dict({k: lucid.tensor(a.copy()).to(device) for k, a in arrays.items()})


def _put_ref(ref: Any, net: Any, arrays: dict[str, np.ndarray]) -> None:
    net.load_state_dict({k: ref.from_numpy(np.array(a)) for k, a in arrays.items()})


def _values(averaged: nn.Module) -> dict[str, np.ndarray]:
    return {k: np.array(v.numpy(), copy=True) for k, v in averaged.state_dict().items()}


def _same(ours: dict[str, np.ndarray], theirs: Any, tol: float, where: str) -> None:
    ref_state = {k: v.detach().numpy() for k, v in theirs.state_dict().items()}
    assert list(ours) == list(ref_state), where
    for key, value in ours.items():
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(value, ref_state[key], err_msg=f"{where} {key}")
        else:
            np.testing.assert_allclose(
                value, ref_state[key], rtol=tol, atol=tol, err_msg=f"{where} {key}"
            )


# -- AveragedModel ---------------------------------------------------------------


def _swa_rule(a: lucid.Tensor, c: lucid.Tensor, n: Any) -> lucid.Tensor:
    return a + (c - a) / (n + 1)


def _half_way(a: lucid.Tensor, c: lucid.Tensor, n: Any) -> lucid.Tensor:
    return a + 0.5 * (c - a)


# name -> (kwargs for Lucid, kwargs for the reference given the reference module)
_RULES: dict[str, Callable[[Any, Any], tuple[dict[str, Any], dict[str, Any]]]] = {
    "default": lambda s, r: ({}, {}),
    "swa_fn": lambda s, r: (
        {"avg_fn": s.get_swa_avg_fn()},
        {"avg_fn": r.optim.swa_utils.get_swa_avg_fn()},
    ),
    # The reference's explicit SWA list rule cannot average an integer buffer
    # (it divides integers with addcdiv); the default rule is the same average.
    "swa_multi": lambda s, r: ({"multi_avg_fn": s.get_swa_multi_avg_fn()}, {}),
    "ema_fn": lambda s, r: (
        {"avg_fn": s.get_ema_avg_fn(0.9)},
        {"avg_fn": r.optim.swa_utils.get_ema_avg_fn(0.9)},
    ),
    "ema_multi": lambda s, r: (
        {"multi_avg_fn": s.get_ema_multi_avg_fn(0.9)},
        {"multi_avg_fn": r.optim.swa_utils.get_ema_multi_avg_fn(0.9)},
    ),
    "custom_fn": lambda s, r: ({"avg_fn": _half_way}, {"avg_fn": _half_way}),
}


@pytest.mark.parity
@pytest.mark.parametrize("use_buffers", [False, True])
@pytest.mark.parametrize("rule", list(_RULES))
@pytest.mark.parametrize("device, dtype", _DEVICE_DTYPES)
def test_averages_match_the_reference(
    device: str, dtype: lucid.dtype, rule: str, use_buffers: bool, ref: Any
) -> None:
    tol = _TOL[dtype]
    ours_kw, ref_kw = _RULES[rule](swa_utils, ref)
    model = _lucid_net(device, dtype)
    ref_model = _ref_net(ref, model, dtype)
    averaged = swa_utils.AveragedModel(model, use_buffers=use_buffers, **ours_kw)
    ref_averaged = ref.optim.swa_utils.AveragedModel(ref_model, use_buffers=use_buffers, **ref_kw)
    for k in range(1, 6):
        arrays = _arrays(model.state_dict(), k)
        _put(model, arrays, device)
        _put_ref(ref, ref_model, arrays)
        averaged.update_parameters(model)
        ref_averaged.update_parameters(ref_model)
        _same(_values(averaged), ref_averaged, tol, f"{rule} update {k}")
    assert int(averaged.n_averaged.item()) == 5


@pytest.mark.parity
@pytest.mark.parametrize("device, dtype", _DEVICE_DTYPES)
def test_the_average_is_independent_of_the_source_and_serves_forward(
    device: str, dtype: lucid.dtype, ref: Any
) -> None:
    model = _lucid_net(device, dtype)
    averaged = swa_utils.AveragedModel(model)
    before = _values(averaged)
    _put(model, _arrays(model.state_dict(), 9), device)
    for key, value in _values(averaged).items():
        if key != "n_averaged":
            np.testing.assert_array_equal(value, before[key], err_msg=key)

    averaged.update_parameters(model)
    x = lucid.randn(5, 3).to(device)
    x = x.double() if dtype == lucid.float64 else x
    model.eval()
    averaged.eval()
    np.testing.assert_allclose(averaged(x).numpy(), model(x).numpy(), rtol=_TOL[dtype], atol=1e-6)


@pytest.mark.parity
def test_state_dict_keys_match_the_reference_and_round_trip(ref: Any) -> None:
    model = _lucid_net("cpu", lucid.float32)
    averaged = swa_utils.AveragedModel(model, use_buffers=True)
    ref_averaged = ref.optim.swa_utils.AveragedModel(_ref_net(ref, model, lucid.float32))
    assert list(averaged.state_dict()) == list(ref_averaged.state_dict())

    for k in (1, 2, 3):
        _put(model, _arrays(model.state_dict(), k), "cpu")
        averaged.update_parameters(model)
    clone = swa_utils.AveragedModel(_lucid_net("cpu", lucid.float32), use_buffers=True)
    clone.load_state_dict(averaged.state_dict())
    assert int(clone.n_averaged.item()) == 3
    for key, value in _values(averaged).items():
        np.testing.assert_array_equal(_values(clone)[key], value, err_msg=key)
    # The next update continues the running mean rather than starting over.
    arrays = _arrays(model.state_dict(), 4)
    _put(model, arrays, "cpu")
    averaged.update_parameters(model)
    clone.update_parameters(model)
    for key, value in _values(averaged).items():
        np.testing.assert_allclose(_values(clone)[key], value, rtol=1e-6, err_msg=key)


def test_a_rule_receives_the_count_as_an_integer_tensor_on_the_tensors_device(device: str) -> None:
    model = _lucid_net(device, lucid.float32)
    seen: list[tuple[int, lucid.dtype, object]] = []

    def rule(a: lucid.Tensor, c: lucid.Tensor, n: lucid.Tensor) -> lucid.Tensor:
        seen.append((int(n.item()), n.dtype, n.device))
        return a

    averaged = swa_utils.AveragedModel(model, avg_fn=rule)
    averaged.update_parameters(model)
    assert not seen  # the first update copies
    averaged.update_parameters(model)
    averaged.update_parameters(model)
    assert seen
    assert {count for count, _, _ in seen} == {1, 2}
    assert all(dt == lucid.int64 and dev == model[0].weight.device for _, dt, dev in seen)


def test_a_list_rule_gets_tensors_grouped_by_device_and_dtype(device: str) -> None:
    model = _lucid_net(device, lucid.float32)
    calls: list[list[lucid.Tensor]] = []

    def rule(avg: list[lucid.Tensor], cur: list[lucid.Tensor], n: lucid.Tensor) -> None:
        calls.append(list(avg))
        assert len(avg) == len(cur)

    averaged = swa_utils.AveragedModel(model, multi_avg_fn=rule, use_buffers=True)
    averaged.update_parameters(model)
    averaged.update_parameters(model)
    assert len(calls) == 2  # float32 parameters and buffers, then the int64 counter
    for group in calls:
        assert len({t.dtype for t in group}) == 1
    assert sorted(len(g) for g in calls) == [1, 8]


def test_both_rules_together_are_refused() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Only one of avg_fn and multi_avg_fn"):
        swa_utils.AveragedModel(
            model, avg_fn=_swa_rule, multi_avg_fn=swa_utils.get_swa_multi_avg_fn()
        )


@pytest.mark.parametrize("decay", [-0.1, 1.5])
@pytest.mark.parametrize("factory", ["get_ema_avg_fn", "get_ema_multi_avg_fn"])
def test_decay_outside_the_unit_interval_is_refused(factory: str, decay: float) -> None:
    with pytest.raises(ValueError, match="Invalid decay"):
        getattr(swa_utils, factory)(decay)


def test_a_model_of_another_architecture_is_refused_and_changes_nothing() -> None:
    averaged = swa_utils.AveragedModel(_lucid_net("cpu", lucid.float32))
    averaged.update_parameters(_lucid_net("cpu", lucid.float32))
    before = _values(averaged)
    with pytest.raises(ValueError, match="same architecture"):
        averaged.update_parameters(nn.Linear(3, 4))
    for key, value in _values(averaged).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)


def test_the_two_forms_of_a_rule_agree() -> None:
    for single, multi in (
        (swa_utils.get_swa_avg_fn(), swa_utils.get_swa_multi_avg_fn()),
        (swa_utils.get_ema_avg_fn(0.7), swa_utils.get_ema_multi_avg_fn(0.7)),
    ):
        a, c = lucid.randn(4), lucid.randn(4)
        expected = single(a, c, lucid.tensor(2, dtype=lucid.int64)).numpy()
        pair = [a.clone()]
        multi(pair, [c], lucid.tensor(2, dtype=lucid.int64))
        np.testing.assert_allclose(pair[0].numpy(), expected, rtol=1e-6)


@pytest.mark.parity
def test_the_public_names_are_the_references(ref: Any) -> None:
    assert sorted(swa_utils.__all__) == sorted(ref.optim.swa_utils.__all__) == sorted(SURFACE)
    assert optim.swa_utils is swa_utils


# -- SWALR -----------------------------------------------------------------------


def _drive(opt: Any, sched: Any, steps: int) -> list[list[float]]:
    out = []
    for _ in range(steps):
        opt.step()
        sched.step()
        out.append([float(g["lr"]) for g in opt.param_groups])
    return out


def _sgd(module: Any, groups: int, lrs: list[float]) -> Any:
    params = [p for p in module.parameters()]
    split = [{"params": params[i::groups], "lr": lrs[i]} for i in range(groups)]
    return module, split


@pytest.mark.parity
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("anneal_epochs", [0, 1, 3, 10])
@pytest.mark.parametrize("strategy", ["cos", "linear"])
def test_swalr_trajectory_matches_the_reference(
    strategy: str, anneal_epochs: int, groups: int, ref: Any
) -> None:
    lrs = [0.1, 0.02][:groups]
    swa = 0.05 if groups == 1 else [0.05, 0.3]
    _, ours_groups = _sgd(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), groups, lrs)
    _, ref_groups = _sgd(ref.nn.Sequential(ref.nn.Linear(2, 2), ref.nn.Linear(2, 2)), groups, lrs)
    opt = optim.SGD(ours_groups, lr=0.1)
    ref_opt = ref.optim.SGD(ref_groups, lr=0.1)
    sched = swa_utils.SWALR(opt, swa, anneal_epochs=anneal_epochs, anneal_strategy=strategy)
    ref_sched = ref.optim.swa_utils.SWALR(
        ref_opt, swa, anneal_epochs=anneal_epochs, anneal_strategy=strategy
    )
    assert [g["lr"] for g in opt.param_groups] == pytest.approx(
        [g["lr"] for g in ref_opt.param_groups], rel=1e-12
    )
    ours = _drive(opt, sched, 15)
    theirs = _drive(ref_opt, ref_sched, 15)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)
    assert [g["swa_lr"] for g in opt.param_groups] == [g["swa_lr"] for g in ref_opt.param_groups]
    assert sched.get_last_lr() == pytest.approx(ref_sched.get_last_lr(), rel=1e-12)


@pytest.mark.parity
def test_swalr_takes_over_from_another_schedule(ref: Any) -> None:
    ours_net, ref_net = nn.Linear(2, 2), ref.nn.Linear(2, 2)
    opt = optim.SGD(ours_net.parameters(), lr=0.1)
    ref_opt = ref.optim.SGD(ref_net.parameters(), lr=0.1)
    decay = optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
    ref_decay = ref.optim.lr_scheduler.ExponentialLR(ref_opt, gamma=0.8)
    np.testing.assert_allclose(_drive(opt, decay, 4), _drive(ref_opt, ref_decay, 4), rtol=1e-12)
    # The scheduler built afterwards starts from the rate the first left behind.
    sched = swa_utils.SWALR(opt, swa_lr=0.01, anneal_epochs=4)
    ref_sched = ref.optim.swa_utils.SWALR(ref_opt, swa_lr=0.01, anneal_epochs=4)
    np.testing.assert_allclose(
        _drive(opt, sched, 8), _drive(ref_opt, ref_sched, 8), rtol=1e-12, atol=1e-12
    )


def test_swalr_holds_the_swa_rate_after_the_annealing() -> None:
    opt = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    sched = swa_utils.SWALR(opt, swa_lr=0.03, anneal_epochs=3, anneal_strategy="linear")
    rates = [r[0] for r in _drive(opt, sched, 8)]
    assert rates[:3] == pytest.approx([0.1 + (0.03 - 0.1) * t / 3 for t in (1, 2, 3)])
    assert rates[3:] == pytest.approx([0.03] * 5)


@pytest.mark.parametrize("strategy", ["cos", "linear"])
def test_swalr_resumes_from_a_checkpoint(strategy: str) -> None:
    def build() -> tuple[Any, Any]:
        o = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
        return o, swa_utils.SWALR(o, swa_lr=0.02, anneal_epochs=6, anneal_strategy=strategy)

    opt_a, sched_a = build()
    straight = _drive(opt_a, sched_a, 10)

    opt_b, sched_b = build()
    _drive(opt_b, sched_b, 4)
    saved_opt, saved_sched = opt_b.state_dict(), sched_b.state_dict()
    assert "anneal_func" not in saved_sched and "optimizer" not in saved_sched

    opt_c, sched_c = build()
    opt_c.load_state_dict(saved_opt)
    sched_c.load_state_dict(saved_sched)
    np.testing.assert_allclose(_drive(opt_c, sched_c, 6), straight[4:], rtol=1e-12)


def test_swalr_validates_its_arguments_before_touching_the_optimizer() -> None:
    opt = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="'cos' or 'linear'"):
        swa_utils.SWALR(opt, swa_lr=0.05, anneal_strategy="square")  # type: ignore[arg-type]
    for bad in (-1, 2.5):
        with pytest.raises(ValueError, match="anneal_epochs"):
            swa_utils.SWALR(opt, swa_lr=0.05, anneal_epochs=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 1 values for swa_lr"):
        swa_utils.SWALR(opt, swa_lr=[0.1, 0.2])
    assert "swa_lr" not in opt.param_groups[0]


# -- update_bn -------------------------------------------------------------------


def _bn_batches(seed: int, n: int, device: str) -> list[lucid.Tensor]:
    rng = np.random.default_rng(seed)
    return [
        lucid.tensor(rng.standard_normal((6, 3)).astype(np.float32)).to(device) for _ in range(n)
    ]


@pytest.mark.parity
@pytest.mark.parametrize("as_tuples", [False, True])
@pytest.mark.parametrize("was_training", [False, True])
def test_update_bn_matches_the_reference(
    device: str, was_training: bool, as_tuples: bool, ref: Any
) -> None:
    model = _lucid_net(device, lucid.float32)
    ref_model = _ref_net(ref, model, lucid.float32)
    model.train(was_training)
    ref_model.train(was_training)
    model[1].momentum = 0.3
    ref_model[1].momentum = 0.3
    batches = _bn_batches(3, 5, device)
    feed = [(b, 0) for b in batches] if as_tuples else batches
    ref_feed = (
        [(ref.from_numpy(b.numpy().copy()), 0) for b in batches]
        if as_tuples
        else [ref.from_numpy(b.numpy().copy()) for b in batches]
    )

    swa_utils.update_bn(feed, model)
    ref.optim.swa_utils.update_bn(ref_feed, ref_model)

    bn, ref_bn = model[1], ref_model[1]
    np.testing.assert_allclose(
        bn.running_mean.numpy(), ref_bn.running_mean.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        bn.running_var.numpy(), ref_bn.running_var.numpy(), rtol=1e-5, atol=1e-6
    )
    assert int(bn.num_batches_tracked.item()) == int(ref_bn.num_batches_tracked.item()) == 5
    assert bn.momentum == 0.3
    assert model.training is was_training and bn.training is was_training


def test_update_bn_leaves_a_model_without_batch_norm_alone() -> None:
    model = nn.Linear(3, 2)
    model.eval()
    swa_utils.update_bn([lucid.randn(4, 3)], model)
    assert model.training is False


def test_update_bn_restores_the_model_when_the_loader_fails() -> None:
    model = _lucid_net("cpu", lucid.float32)
    model.eval()

    def loader() -> Any:
        yield lucid.randn(4, 3)
        raise RuntimeError("loader broke")

    with pytest.raises(RuntimeError, match="loader broke"):
        swa_utils.update_bn(loader(), model)
    assert model.training is False
    assert model[1].momentum == 0.1


def test_update_bn_moves_batches_to_the_requested_device(device: str) -> None:
    model = _lucid_net(device, lucid.float32)
    swa_utils.update_bn(_bn_batches(1, 2, "cpu"), model, device=device)
    assert int(model[1].num_batches_tracked.item()) == 2


def test_update_bn_refreshes_an_averaged_model(device: str) -> None:
    model = _lucid_net(device, lucid.float32)
    averaged = swa_utils.AveragedModel(model)
    averaged.update_parameters(model)
    swa_utils.update_bn(_bn_batches(2, 3, device), averaged)
    bn = averaged.module[1]
    assert int(bn.num_batches_tracked.item()) == 3
    assert not math.isclose(float(bn.running_var.sum().item()), 4.0)
