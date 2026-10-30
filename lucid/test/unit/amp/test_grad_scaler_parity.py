"""``GradScaler`` takes the reference framework's signature and accessors (LCD-740).

``GradScaler(init_scale, ...)`` began with ``init_scale`` where the reference
begins with ``device``, so ``GradScaler("cpu")`` raised "could not convert
string to float" and ``GradScaler(device="cpu")`` raised ``TypeError``.  It
lacked the reference's ``is_enabled`` and the growth / backoff / interval
accessors, answered ``65536.0`` from ``get_scale()`` when disabled (the
reference says ``1.0``), and wrote the growth counter to ``state_dict`` under
``growth_tracker`` where the reference writes ``_growth_tracker``.

Now ``device`` is the first parameter (``'metal'`` or ``'cpu'``), the
accessors exist, a disabled scaler reads as scale ``1.0`` with an empty state,
and ``state_dict`` carries the reference's key while ``load_state_dict`` still
accepts the old one, so checkpoints written before this change load.
"""

import inspect
from types import ModuleType

import pytest

import lucid
from lucid import optim
from lucid.amp import GradScaler

INF = float("inf")

#: ``state_dict`` of a scaler in its middle: every key, the old tracker name.
OLD_STATE = {
    "scale": 512.0,
    "growth_factor": 2.0,
    "backoff_factor": 0.5,
    "growth_interval": 4,
    "growth_tracker": 3,
}
NEW_STATE = {k if k != "growth_tracker" else "_growth_tracker": v for k, v in OLD_STATE.items()}


def _param(value: float, device: str) -> lucid.nn.Parameter:
    return lucid.nn.Parameter(lucid.tensor([value], device=device))


def _clean_step(scaler: GradScaler, device: str) -> None:
    p = _param(1.0, device)
    opt = optim.SGD([p], lr=0.0)
    scaler.scale((p * 1.0).sum()).backward()
    scaler.step(opt)
    scaler.update()


def _overflowing_step(scaler: GradScaler, device: str) -> None:
    p = _param(1.0, device)
    opt = optim.SGD([p], lr=0.0)
    p.grad = lucid.tensor([INF], device=device)
    scaler.step(opt)
    scaler.update()


# ── the signature ────────────────────────────────────────────────────────────


def test_the_constructor_binds_as_the_reference_does(ref: ModuleType) -> None:
    ours = inspect.signature(GradScaler).parameters
    theirs = inspect.signature(ref.amp.GradScaler).parameters
    assert list(ours) == list(theirs)
    for name in ours:
        assert ours[name].kind == theirs[name].kind, name
        if name != "device":  # the default names the device each framework runs on
            assert ours[name].default == theirs[name].default, name


def test_device_is_first_and_defaults_to_metal() -> None:
    assert next(iter(inspect.signature(GradScaler).parameters)) == "device"
    assert inspect.signature(GradScaler).parameters["device"].default == "metal"


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_device_is_accepted_by_position_and_by_keyword(device: str) -> None:
    assert GradScaler(device).get_scale() == 2.0**16
    assert GradScaler(device=device, init_scale=8.0).get_scale() == 8.0
    # The remaining parameters follow it, in the reference's order.
    scaler = GradScaler(device, 8.0, 4.0, 0.25, 7, True)
    assert (scaler.get_scale(), scaler.get_growth_factor(), scaler.get_backoff_factor()) == (
        8.0,
        4.0,
        0.25,
    )
    assert scaler.get_growth_interval() == 7 and scaler.is_enabled()


@pytest.mark.parametrize("device", ["tpu", "gpu", "", "CPU", None])
def test_an_unknown_device_is_a_value_error(device: str | None) -> None:
    with pytest.raises(ValueError, match="GradScaler: unknown device"):
        GradScaler(device)
    with pytest.raises(ValueError, match="GradScaler: unknown device"):
        GradScaler(device, enabled=False)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"growth_factor": 1.0}, "growth_factor must be > 1.0"),
        ({"growth_factor": 0.5}, "growth_factor must be > 1.0"),
        ({"backoff_factor": 1.0}, "backoff_factor must be < 1.0"),
        ({"backoff_factor": 2.0}, "backoff_factor must be < 1.0"),
    ],
)
def test_an_enabled_scaler_refuses_factors_that_cannot_grow_or_shrink(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GradScaler("cpu", **kwargs)
    GradScaler("cpu", enabled=False, **kwargs)  # a disabled one never scales


# ── accessors ────────────────────────────────────────────────────────────────


def test_the_accessors_read_what_the_constructor_was_given() -> None:
    scaler = GradScaler("cpu", 8.0, 3.0, 0.25, 5)
    assert scaler.get_growth_factor() == 3.0
    assert scaler.get_backoff_factor() == 0.25
    assert scaler.get_growth_interval() == 5
    assert scaler.is_enabled() is True
    assert GradScaler("cpu", enabled=False).is_enabled() is False


def test_the_setters_drive_the_next_update() -> None:
    scaler = GradScaler("cpu", 8.0, 2.0, 0.5, 100)
    scaler.set_growth_interval(2)
    scaler.set_growth_factor(4.0)
    scaler.set_backoff_factor(0.25)
    assert (scaler.get_growth_interval(), scaler.get_growth_factor()) == (2, 4.0)
    assert scaler.get_backoff_factor() == 0.25
    _clean_step(scaler, "cpu")
    assert scaler.get_scale() == 8.0  # one clean update of two
    _clean_step(scaler, "cpu")
    assert scaler.get_scale() == 32.0  # grew by the new factor, at the new interval
    _overflowing_step(scaler, "cpu")
    assert scaler.get_scale() == 8.0  # backed off by the new factor


def test_the_accessors_follow_the_reference(ref: ModuleType) -> None:
    ours, theirs = GradScaler("cpu", 8.0, 3.0, 0.25, 5), ref.amp.GradScaler(
        "cpu", 8.0, 3.0, 0.25, 5
    )
    for scaler in (ours, theirs):
        scaler.set_growth_factor(5.0)
        scaler.set_backoff_factor(0.125)
        scaler.set_growth_interval(9)
    for getter in ("get_scale", "get_growth_factor", "get_backoff_factor", "get_growth_interval"):
        assert getattr(ours, getter)() == getattr(theirs, getter)(), getter
    assert ours.is_enabled() == theirs.is_enabled()


# ── a disabled scaler ────────────────────────────────────────────────────────


def test_a_disabled_scaler_reads_as_scale_one_with_no_state(ref: ModuleType) -> None:
    ours, theirs = GradScaler("cpu", enabled=False), ref.amp.GradScaler("cpu", enabled=False)
    assert ours.get_scale() == theirs.get_scale() == 1.0
    assert ours.state_dict() == theirs.state_dict() == {}
    ours.load_state_dict({})  # nothing to load into: a no-op, as for the reference
    ours.load_state_dict(NEW_STATE)
    assert ours.get_scale() == 1.0 and ours.state_dict() == {}


def test_a_disabled_scaler_passes_everything_through() -> None:
    scaler = GradScaler("cpu", enabled=False)
    x = lucid.ones(2)
    assert scaler.scale(x) is x
    scaler.update(new_scale=4.0)
    assert scaler.get_scale() == 1.0
    p = _param(1.0, "cpu")
    p.grad = lucid.tensor([0.5])
    scaler.step(optim.SGD([p], lr=1.0))
    assert p.tolist() == [0.5]  # stepped, gradient untouched


# ── state_dict ───────────────────────────────────────────────────────────────


def test_state_dict_has_the_reference_keys_and_values(ref: ModuleType) -> None:
    ours = GradScaler("cpu", init_scale=1024.0, growth_interval=7).state_dict()
    theirs = ref.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=7).state_dict()
    assert ours == theirs
    assert set(ours) == {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }


def test_state_dict_round_trips_after_training_steps(device: str) -> None:
    scaler = GradScaler(device, init_scale=2.0**8, growth_interval=3)
    for _ in range(2):
        _clean_step(scaler, device)
    state = scaler.state_dict()
    assert state["_growth_tracker"] == 2
    fresh = GradScaler(device)
    fresh.load_state_dict(state)
    assert fresh.state_dict() == state
    _clean_step(scaler, device)
    _clean_step(fresh, device)
    assert fresh.state_dict() == scaler.state_dict()
    assert fresh.get_scale() == 2.0**9  # the third clean step grew it, in both


def test_the_old_tracker_key_still_loads() -> None:
    scaler = GradScaler("cpu")
    scaler.load_state_dict(OLD_STATE)
    assert scaler.state_dict() == NEW_STATE  # and is written back under the new one


def test_the_new_key_wins_when_both_are_present() -> None:
    scaler = GradScaler("cpu")
    scaler.load_state_dict({**NEW_STATE, "growth_tracker": 99})
    assert scaler.state_dict()["_growth_tracker"] == 3


@pytest.mark.parametrize("missing", list(NEW_STATE))
def test_a_state_missing_a_key_names_it_and_leaves_the_scaler_unchanged(missing: str) -> None:
    scaler = GradScaler("cpu", init_scale=2.0**4, growth_interval=9)
    before = scaler.state_dict()
    state = {k: v for k, v in NEW_STATE.items() if k != missing}
    with pytest.raises(KeyError, match=missing):
        scaler.load_state_dict(state)
    assert scaler.state_dict() == before  # no half-loaded state


def test_an_empty_state_is_refused_by_an_enabled_scaler(ref: ModuleType) -> None:
    with pytest.raises(RuntimeError, match="empty"):
        GradScaler("cpu").load_state_dict({})
    with pytest.raises(RuntimeError, match="empty"):
        ref.amp.GradScaler("cpu").load_state_dict({})


def test_a_reference_state_loads_into_a_scaler(ref: ModuleType) -> None:
    state = ref.amp.GradScaler("cpu", init_scale=64.0, growth_interval=11).state_dict()
    scaler = GradScaler("cpu")
    scaler.load_state_dict(state)
    assert scaler.state_dict() == state
