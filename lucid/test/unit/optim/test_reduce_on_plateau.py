"""``ReduceLROnPlateau`` keeps numbers, validates its arguments and exposes the reference's names.

It stored the metric it was handed, so the loss tensor -- with its autograd
graph -- lived in the scheduler between epochs, an in-place change to the
tensor moved the stored best, and a checkpoint carried a tensor.  A per-group
``min_lr`` list crashed the first reduction, an unknown mode or a factor of
1 or more was accepted, and ``best`` / ``num_bad_epochs`` /
``cooldown_counter`` / ``last_epoch`` / ``get_last_lr`` did not exist
(LCD-788).
"""

import gc
import weakref
from types import ModuleType
from typing import Any

import pytest

import lucid

_METRICS = [1.0, 0.9, 0.95, 0.95, 0.97, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6]


def _optimizer(lib: ModuleType, groups: int = 2) -> Any:
    params = [lib.nn.Parameter(lib.ones(2)) for _ in range(groups)]
    return lib.optim.SGD(
        [{"params": [p], "lr": 0.1 / (i + 1)} for i, p in enumerate(params)], lr=0.1
    )


def _run(lib: ModuleType, steps: int = len(_METRICS), **kwargs: Any) -> tuple[Any, Any]:
    opt = _optimizer(lib)
    sched = lib.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, **kwargs)
    for metric in _METRICS[:steps]:
        opt.step()
        sched.step(metric)
    return opt, sched


def test_the_graph_of_the_loss_is_released_after_the_step() -> None:
    weight = lucid.nn.Parameter(lucid.ones(3))
    opt = lucid.optim.SGD([weight], lr=0.1)
    sched = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0)
    loss = (weight * weight).sum()
    assert loss.requires_grad
    alive = weakref.ref(loss)
    sched.step(loss)
    del loss
    gc.collect()
    assert alive() is None


def test_the_best_is_a_float_and_a_checkpoint_holds_no_tensor() -> None:
    opt = _optimizer(lucid)
    sched = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt)
    metric = lucid.tensor(2.0)
    sched.step(metric)
    assert type(sched.best) is float
    assert not any(isinstance(v, lucid.Tensor) for v in sched.state_dict().values())
    metric.fill_(100.0)
    assert sched.best == 2.0


def test_a_metric_with_more_than_one_element_is_refused_and_changes_nothing() -> None:
    opt = _optimizer(lucid)
    sched = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt)
    before = sched.state_dict()
    with pytest.raises(ValueError, match="scalar metric"):
        sched.step(lucid.tensor([1.0, 2.0]))
    assert sched.state_dict() == before


@pytest.mark.parity
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(mode="max"),
        dict(threshold_mode="abs", threshold=0.01),
        dict(cooldown=2, factor=0.5),
        dict(min_lr=0.03),
        dict(min_lr=[0.03, 0.002]),
        dict(min_lr=(0.03, 0.002)),
    ],
    ids=["default", "max", "abs", "cooldown", "min_lr", "min_lr-list", "min_lr-tuple"],
)
def test_the_rates_and_the_public_state_are_the_references(kwargs: dict[str, Any], ref: Any) -> None:
    _, got = _run(lucid, **kwargs)
    _, want = _run(ref, **kwargs)
    assert got.get_last_lr() == pytest.approx(want.get_last_lr())
    for name in ("best", "num_bad_epochs", "cooldown_counter", "last_epoch", "min_lrs"):
        assert getattr(got, name) == getattr(want, name), name
    assert sorted(got.state_dict()) == sorted(want.state_dict())


@pytest.mark.parity
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="foo"),
        dict(threshold_mode="foo"),
        dict(factor=1.0),
        dict(factor=2.0),
        dict(min_lr=[0.1]),
        dict(min_lr=[0.1, 0.1, 0.1]),
    ],
    ids=["mode", "threshold_mode", "factor-1", "factor-2", "min_lr-short", "min_lr-long"],
)
def test_a_bad_argument_is_refused_as_the_reference_refuses_it(
    kwargs: dict[str, Any], ref: Any
) -> None:
    want: Exception | None = None
    try:
        ref.optim.lr_scheduler.ReduceLROnPlateau(_optimizer(ref), **kwargs)
    except Exception as exc:
        want = exc
    assert want is not None
    with pytest.raises(type(want)) as got:
        lucid.optim.lr_scheduler.ReduceLROnPlateau(_optimizer(lucid), **kwargs)
    assert str(got.value) == str(want)


def test_something_that_is_not_an_optimizer_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="not an Optimizer"):
        lucid.optim.lr_scheduler.ReduceLROnPlateau(object())  # type: ignore[arg-type]


@pytest.mark.parity
def test_the_epoch_argument_warns_and_sets_the_epoch(ref: Any) -> None:
    for lib in (lucid, ref):
        sched = lib.optim.lr_scheduler.ReduceLROnPlateau(_optimizer(lib))
        with pytest.warns(UserWarning, match="epoch parameter"):
            sched.step(1.0, 7)
        assert sched.last_epoch == 7


def test_a_resumed_scheduler_continues_where_the_checkpoint_left_off() -> None:
    whole_opt, whole = _run(lucid)
    half_opt, half = _run(lucid, steps=7)
    saved, saved_lr = half.state_dict(), [float(g["lr"]) for g in half_opt.param_groups]
    opt = _optimizer(lucid)
    for group, lr in zip(opt.param_groups, saved_lr, strict=True):
        group["lr"] = lr
    resumed = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1)
    resumed.load_state_dict(saved)
    for metric in _METRICS[7:]:
        opt.step()
        resumed.step(metric)
    assert [g["lr"] for g in opt.param_groups] == [g["lr"] for g in whole_opt.param_groups]
    assert resumed.state_dict() == whole.state_dict()


def test_a_checkpoint_from_before_the_names_were_public_still_loads() -> None:
    opt = _optimizer(lucid)
    sched = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
    sched.load_state_dict(
        {
            "mode": "min",
            "factor": 0.1,
            "patience": 2,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0.01,
            "eps": 1e-8,
            "verbose": False,
            "_best": lucid.tensor(0.5),
            "_num_bad_epochs": 2,
            "_cooldown_counter": 0,
        }
    )
    assert (sched.best, sched.num_bad_epochs, sched.min_lrs) == (0.5, 2, [0.01, 0.01])
    assert type(sched.best) is float
    opt.step()
    sched.step(0.6)  # the third bad epoch in a row: past patience
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 0.01])


def test_a_group_added_later_follows_a_scalar_min_lr_and_a_list_one_refuses_it() -> None:
    for min_lr, refuses in ((0.001, False), ([0.001, 0.001], True)):
        opt = _optimizer(lucid)
        sched = lucid.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, min_lr=min_lr)
        opt.add_param_group({"params": [lucid.nn.Parameter(lucid.ones(2))], "lr": 0.1})
        sched.step(1.0)
        before = (sched.state_dict(), [g["lr"] for g in opt.param_groups])
        if refuses:
            with pytest.raises(RuntimeError, match="differs from when"):
                sched.step(1.0)
            assert (sched.state_dict(), [g["lr"] for g in opt.param_groups]) == before
        else:
            sched.step(1.0)
            assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 0.005, 0.01])
