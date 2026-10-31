"""The schedulers take the arguments the reference's do, and read them in one place.

``MultiplicativeLR`` accepted one callable only where the reference takes a
list of them too; ``step(epoch)`` -- the documented per-batch idiom of
``CosineAnnealingWarmRestarts`` and the deprecated-but-working jump of every
other schedule -- did not exist; ``T_0`` and ``T_mult`` were not validated;
the base class was only reachable as a private name; and every scheduler
carried a ``verbose`` argument (and ``PolynomialLR`` an ``eta_min``) the
reference does not have (LCD-789).
"""

import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.optim.lr_scheduler as lucid_schedulers

type Build = Callable[[Any, Any], Any]

_LRS = (0.1, 0.05)


def _optimizer(lib: ModuleType) -> Any:
    params = [lib.nn.Parameter(lib.zeros(2)) for _ in _LRS]
    return lib.optim.SGD(
        [{"params": [p], "lr": lr} for p, lr in zip(params, _LRS, strict=True)], lr=0.1
    )


def _rates(opt: Any) -> list[float]:
    return [float(g["lr"]) for g in opt.param_groups]


def _flat(rows: list[list[float]]) -> list[float]:
    return [lr for row in rows for lr in row]


def _error_of(call: Callable[[], object]) -> Exception:
    try:
        call()
    except Exception as exc:
        return exc
    raise AssertionError("no error raised")


# ── MultiplicativeLR / LambdaLR: one function or one per group ───────────────


@pytest.mark.parity
def test_multiplicative_takes_a_list_of_lambdas_as_the_reference_does(ref: Any) -> None:
    rows = []
    for lib in (lucid, ref):
        opt = _optimizer(lib)
        sched = lib.optim.lr_scheduler.MultiplicativeLR(opt, [lambda e: 0.9, lambda e: 0.5])
        row = []
        for _ in range(8):
            opt.step()
            sched.step()
            row.append(_rates(opt))
        rows.append(row)
    assert _flat(rows[0]) == pytest.approx(_flat(rows[1]))
    assert len(sched.lr_lambdas) == 2


@pytest.mark.parity
@pytest.mark.parametrize("name", ["LambdaLR", "MultiplicativeLR"])
@pytest.mark.parametrize(
    "lambdas",
    [[lambda e: 1.0], [lambda e: 1.0] * 3, 3, [lambda e: 1.0, 3]],
    ids=["too-few", "too-many", "not-callable", "one-not-callable"],
)
def test_a_bad_lambda_argument_is_refused_as_the_reference_refuses_it(
    name: str, lambdas: Any, ref: Any
) -> None:
    want = _error_of(lambda: getattr(ref.optim.lr_scheduler, name)(_optimizer(ref), lambdas))
    got = _error_of(lambda: getattr(lucid.optim.lr_scheduler, name)(_optimizer(lucid), lambdas))
    assert type(got) is type(want) or (type(got), type(want)) == (TypeError, TypeError)
    if isinstance(want, ValueError):
        assert str(got) == str(want)


def test_a_tuple_of_lambdas_is_one_per_group() -> None:
    opt = _optimizer(lucid)
    sched = lucid.optim.lr_scheduler.MultiplicativeLR(opt, (lambda e: 0.5, lambda e: 0.25))
    opt.step()
    sched.step()
    assert _rates(opt) == pytest.approx([0.05, 0.0125])


# ── step(epoch) ──────────────────────────────────────────────────────────────

_EPOCH_SCHEDULES: dict[str, Build] = {
    "StepLR": lambda S, o: S.StepLR(o, 3, 0.5),
    "MultiStepLR": lambda S, o: S.MultiStepLR(o, [2, 5, 5], 0.5),
    "ExponentialLR": lambda S, o: S.ExponentialLR(o, 0.9),
    "CosineAnnealingLR": lambda S, o: S.CosineAnnealingLR(o, 7, 0.001),
    "LambdaLR": lambda S, o: S.LambdaLR(o, lambda e: 0.9**e),
    "MultiplicativeLR": lambda S, o: S.MultiplicativeLR(o, lambda e: 0.9),
    "LinearLR": lambda S, o: S.LinearLR(o, 0.25, 1.0, 6),
    "ConstantLR": lambda S, o: S.ConstantLR(o, 0.5, 4),
    "PolynomialLR": lambda S, o: S.PolynomialLR(o, 9, 2.0),
    "CyclicLR": lambda S, o: S.CyclicLR(o, 0.01, 0.2, 3, 4),
    "OneCycleLR": lambda S, o: S.OneCycleLR(o, [0.2, 0.1], total_steps=30),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_EPOCH_SCHEDULES))
def test_step_with_an_epoch_warns_and_jumps_as_the_reference_does(name: str, ref: Any) -> None:
    rows = []
    for lib in (lucid, ref):
        opt = _optimizer(lib)
        sched = _EPOCH_SCHEDULES[name](lib.optim.lr_scheduler, opt)
        row = []
        for epoch in (0, 3, 7, 12, 5, 20, 2):
            opt.step()
            with pytest.warns(UserWarning, match="epoch parameter"):
                sched.step(epoch)
            assert sched.last_epoch == epoch
            row.append(_rates(opt))
        rows.append(row)
    assert _flat(rows[0]) == pytest.approx(_flat(rows[1]), rel=1e-9, abs=1e-12)


# ── CosineAnnealingWarmRestarts: step(epoch) is the documented idiom ─────────


def _restarts(**kwargs: Any) -> Build:
    return lambda S, o: S.CosineAnnealingWarmRestarts(o, **kwargs)


@pytest.mark.parity
@pytest.mark.parametrize("kwargs", [dict(T_0=4), dict(T_0=3, T_mult=2, eta_min=0.01)])
def test_the_per_batch_idiom_follows_the_reference(kwargs: dict[str, Any], ref: Any) -> None:
    batches = 5
    rows, epochs = [], []
    for lib in (lucid, ref):
        opt = _optimizer(lib)
        sched = _restarts(**kwargs)(lib.optim.lr_scheduler, opt)
        row = []
        for epoch in range(8):
            for i in range(batches):
                opt.step()
                sched.step(epoch + i / batches)
                row.append(_rates(opt))
        # Interleaved with plain steps, which continue from the last epoch.
        sched.step(26)
        sched.step()
        row.append(_rates(opt))
        rows.append(row)
        epochs.append(sched.last_epoch)
    assert _flat(rows[0]) == pytest.approx(_flat(rows[1]), rel=1e-9, abs=1e-12)
    assert epochs[0] == epochs[1] == 27


@pytest.mark.parity
def test_a_negative_epoch_is_refused_and_changes_nothing(ref: Any) -> None:
    errors = []
    for lib in (lucid, ref):
        opt = _optimizer(lib)
        sched = lib.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 4)
        opt.step()
        sched.step()
        before = (_rates(opt), sched.last_epoch)
        errors.append(_error_of(lambda: sched.step(-1)))
        assert (_rates(opt), sched.last_epoch) == before
    assert type(errors[0]) is type(errors[1]) is ValueError
    assert str(errors[0]) == str(errors[1])


@pytest.mark.parity
@pytest.mark.parametrize(
    "kwargs",
    [dict(T_0=0), dict(T_0=-3), dict(T_0=3.5), dict(T_0=3, T_mult=0), dict(T_0=3, T_mult=1.5)],
    ids=["T_0-0", "T_0-neg", "T_0-float", "T_mult-0", "T_mult-float"],
)
def test_restart_arguments_are_validated_as_the_reference_validates_them(
    kwargs: dict[str, Any], ref: Any
) -> None:
    want = _error_of(
        lambda: ref.optim.lr_scheduler.CosineAnnealingWarmRestarts(_optimizer(ref), **kwargs)
    )
    got = _error_of(
        lambda: lucid.optim.lr_scheduler.CosineAnnealingWarmRestarts(_optimizer(lucid), **kwargs)
    )
    assert type(got) is type(want) is ValueError
    assert str(got) == str(want)


def test_an_eta_min_that_is_not_a_number_is_a_type_error() -> None:
    junk: Any = "x"
    with pytest.raises(TypeError, match="eta_min"):
        lucid.optim.lr_scheduler.CosineAnnealingWarmRestarts(_optimizer(lucid), 3, eta_min=junk)


# ── the base class and the signatures ────────────────────────────────────────

_SCHEDULERS = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
]


def test_the_public_base_class_and_its_old_name_are_one_class() -> None:
    assert lucid_schedulers.LRScheduler is lucid_schedulers._LRScheduler
    for name in ("StepLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "NoamScheduler"):
        assert issubclass(getattr(lucid_schedulers, name), lucid_schedulers.LRScheduler)


def test_the_epoch_warning_points_at_this_projects_tracker() -> None:
    text = lucid_schedulers.EPOCH_DEPRECATION_WARNING
    assert "epoch" in text and "github.com/lucid-official/lucid/issues" in text


@pytest.mark.parity
@pytest.mark.parametrize("name", _SCHEDULERS)
def test_a_scheduler_has_the_parameters_of_the_reference(name: str, ref: Any) -> None:
    def shape(cls: type) -> list[tuple[str, object]]:
        return [
            (p.name, p.default)
            for p in inspect.signature(cls.__init__).parameters.values()
            if p.name != "self"
        ]

    got = getattr(lucid.optim.lr_scheduler, name)
    want = getattr(ref.optim.lr_scheduler, name)
    assert shape(got) == shape(want)
    assert not hasattr(got, "print_lr")
    if name not in ("SequentialLR", "ChainedScheduler", "ReduceLROnPlateau"):
        assert list(inspect.signature(got.step).parameters) == ["self", "epoch"]


def test_no_scheduler_has_a_verbose_argument_or_an_extra_polynomial_floor() -> None:
    for name in _SCHEDULERS + ["NoamScheduler"]:
        params = inspect.signature(getattr(lucid.optim.lr_scheduler, name).__init__).parameters
        assert "verbose" not in params, name
    assert "eta_min" not in inspect.signature(lucid.optim.PolynomialLR.__init__).parameters


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "param"),
    [
        ("CyclicLR", "mode"),
        ("CyclicLR", "scale_mode"),
        ("OneCycleLR", "anneal_strategy"),
        ("ReduceLROnPlateau", "mode"),
        ("ReduceLROnPlateau", "threshold_mode"),
    ],
)
def test_a_mode_is_annotated_with_the_values_it_accepts(name: str, param: str, ref: Any) -> None:
    def literal(lib: ModuleType) -> str:
        cls = getattr(lib.optim.lr_scheduler, name)
        # The reference keeps its annotations as source text.
        return str(inspect.signature(cls.__init__).parameters[param].annotation).replace(
            "typing.", ""
        )

    assert literal(lucid).startswith("Literal[")
    assert literal(lucid) == literal(ref)


# ── resuming ─────────────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("name", ["StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"])
def test_resuming_over_groups_that_never_had_a_schedule_is_a_key_error(name: str, ref: Any) -> None:
    def build(lib: ModuleType) -> Exception:
        cls = getattr(lib.optim.lr_scheduler, name)
        args = (3,) if name != "CosineAnnealingLR" else (5,)
        return _error_of(lambda: cls(_optimizer(lib), *args, last_epoch=4))

    want, got = build(ref), build(lucid)
    assert type(got) is type(want) is KeyError
    assert str(got) == str(want)


def test_resuming_over_groups_that_have_a_schedule_continues_it() -> None:
    first = _optimizer(lucid)
    S = lucid.optim.lr_scheduler
    S.StepLR(first, 3, 0.5)
    for group in first.param_groups:
        assert group["initial_lr"] == group["lr"]
    sched = S.StepLR(first, 3, 0.5, last_epoch=2)
    assert sched.base_lrs == list(_LRS)
    assert sched.last_epoch == 3
