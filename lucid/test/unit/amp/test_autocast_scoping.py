"""An ``autocast`` scope ends where its ``with`` block ends (CHA-47).

``__exit__`` used to leave the engine guard on the object and install a
second one, so a scope held in a variable stayed in force after the block —
``(a @ a)`` was still float16 until the object was collected.  The guard is
now released in ``__exit__``; its destructor restores the state recorded on
entry.  The scope is re-usable, re-entrant, nests, unwinds on an exception,
and the decorator form behaves the same.

``device_type`` is kept and checked, and the engine tracks autocast state
per device on each thread so a scope only affects ops on its own device.
"""

import threading
from collections.abc import Callable

import pytest

import lucid

autocast = lucid.amp.autocast

OFF = (False, None)


def _state() -> tuple[bool, lucid.dtype | None]:
    return autocast.is_autocast_enabled(), autocast.get_autocast_dtype()


@pytest.fixture(autouse=True)
def _amp_off_around_each_test() -> object:
    assert _state() == OFF, "a previous test left autocast on"
    yield
    assert _state() == OFF, "this test left autocast on"


@pytest.fixture
def a(device_gpu_only: str) -> lucid.Tensor:
    return lucid.randn(4, 4, device=device_gpu_only)


# ── one object, used more than once ─────────────────────────────────────────


def test_exit_restores_while_the_object_is_alive(a: lucid.Tensor) -> None:
    ac = autocast(device_type="metal", dtype=lucid.float16)
    with ac:
        assert (a @ a).dtype == lucid.float16
    assert (a @ a).dtype == lucid.float32  # was float16 until ``del ac``
    assert _state() == OFF


def test_an_object_can_be_entered_again(a: lucid.Tensor) -> None:
    ac = autocast(device_type="metal", dtype=lucid.bfloat16)
    for _ in range(3):
        with ac:
            assert (a @ a).dtype == lucid.bfloat16
        assert (a @ a).dtype == lucid.float32


def test_an_active_object_can_be_entered_again(a: lucid.Tensor) -> None:
    # The reference framework loses the outer state here and stays on.
    ac = autocast(device_type="metal", dtype=lucid.float16)
    with ac:
        with ac:
            assert (a @ a).dtype == lucid.float16
        assert (a @ a).dtype == lucid.float16
    assert _state() == OFF


# ── nesting ─────────────────────────────────────────────────────────────────


def test_nested_scopes_restore_the_outer_dtype(a: lucid.Tensor) -> None:
    outer = autocast(device_type="metal", dtype=lucid.float16)
    inner = autocast(device_type="metal", dtype=lucid.bfloat16)
    with outer:
        with inner:
            assert (a @ a).dtype == lucid.bfloat16
        assert _state() == (True, lucid.float16)
        assert (a @ a).dtype == lucid.float16
    assert _state() == OFF


def test_a_disabled_scope_leaves_autocast_off(a: lucid.Tensor) -> None:
    with autocast(device_type="metal", dtype=lucid.float16, enabled=False):
        assert (a @ a).dtype == lucid.float32
    assert _state() == OFF


# ── leaving by an exception ─────────────────────────────────────────────────


def test_an_exception_unwinds_the_scope(a: lucid.Tensor) -> None:
    ac = autocast(device_type="metal", dtype=lucid.float16)
    with pytest.raises(ValueError, match="boom"):
        with ac:
            raise ValueError("boom")
    assert (a @ a).dtype == lucid.float32


def test_an_exception_unwinds_every_nested_scope(a: lucid.Tensor) -> None:
    outer = autocast(device_type="metal", dtype=lucid.float16)
    inner = autocast(device_type="metal", dtype=lucid.bfloat16)
    with pytest.raises(ValueError):
        with outer:
            with inner:
                raise ValueError
    assert _state() == OFF


# ── decorator ───────────────────────────────────────────────────────────────


def test_the_decorator_scopes_each_call(a: lucid.Tensor) -> None:
    @autocast(device_type="metal", dtype=lucid.bfloat16)
    def mm() -> lucid.dtype:
        return (a @ a).dtype

    assert mm() == lucid.bfloat16
    assert _state() == OFF
    with autocast(device_type="metal", dtype=lucid.float16):
        assert mm() == lucid.bfloat16
        assert (a @ a).dtype == lucid.float16
    assert _state() == OFF


def test_the_decorator_unwinds_on_an_exception(a: lucid.Tensor) -> None:
    @autocast(device_type="metal", dtype=lucid.float16)
    def boom() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError):
        boom()
    assert (a @ a).dtype == lucid.float32


def test_a_decorated_function_can_recurse(a: lucid.Tensor) -> None:
    @autocast(device_type="metal", dtype=lucid.float16)
    def depth(n: int) -> int:
        assert (a @ a).dtype == lucid.float16
        return 0 if n == 0 else 1 + depth(n - 1)

    assert depth(3) == 3
    assert _state() == OFF


# ── threads ─────────────────────────────────────────────────────────────────


def test_one_object_shared_by_two_threads() -> None:
    # Thread A sits in a bfloat16 scope and enters the shared float16 one;
    # thread B enters it too; A leaves first.  A must land back on bfloat16
    # — releasing B's guard instead would restore B's (empty) state on A.
    shared = autocast(device_type="metal", dtype=lucid.float16)
    a_inside, b_inside, a_left = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, object] = {}

    def thread_a() -> None:
        with autocast(device_type="metal", dtype=lucid.bfloat16):
            with shared:
                a_inside.set()
                b_inside.wait(10)
            seen["a"] = _state()
        seen["a_after"] = _state()
        a_left.set()

    def thread_b() -> None:
        a_inside.wait(10)
        with shared:
            b_inside.set()
            a_left.wait(10)
            seen["b"] = _state()
        seen["b_after"] = _state()

    workers = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(20)
    assert seen == {
        "a": (True, lucid.bfloat16),
        "a_after": OFF,
        "b": (True, lucid.float16),
        "b_after": OFF,
    }


# ── device_type ─────────────────────────────────────────────────────────────


def test_an_unknown_device_type_is_refused() -> None:
    with pytest.raises(ValueError, match="device_type"):
        autocast(device_type="gpu")


def test_a_cpu_scope_leaves_metal_ops_alone(a: lucid.Tensor) -> None:
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        assert (a @ a).dtype == lucid.float32


def test_a_metal_scope_leaves_cpu_ops_alone() -> None:
    c = lucid.randn(4, 4)
    with autocast(device_type="metal", dtype=lucid.bfloat16):
        assert (c @ c).dtype == lucid.float32


def test_the_decorator_keeps_its_device_type(a: lucid.Tensor) -> None:
    @autocast(device_type="cpu", dtype=lucid.bfloat16)
    def mm() -> lucid.dtype:
        return (a @ a).dtype

    assert mm() == lucid.float32


# ── against the reference ───────────────────────────────────────────────────


def _dtype_name(d: object) -> str:
    return str(d).rsplit(".", 1)[-1]


def _scenarios(make: Callable[..., object], mm: Callable[[], str]) -> dict[str, list[str]]:
    """The matmul dtype seen at each point of each scenario."""
    out: dict[str, list[str]] = {}

    ac = make()
    log = [mm()]
    with ac:
        log.append(mm())
    log.append(mm())
    with ac:
        log.append(mm())
    log.append(mm())
    out["reuse"] = log

    log = []
    try:
        with ac:
            log.append(mm())
            raise ValueError
    except ValueError:
        pass
    log.append(mm())
    out["exception"] = log

    dec = make()(mm)
    log = [dec(), mm()]
    with make():
        log.append(dec())
        log.append(mm())
    log.append(mm())
    out["decorator"] = log

    log = []
    with make(enabled=False):
        log.append(mm())
    log.append(mm())
    out["disabled"] = log
    return out


@pytest.mark.parity
def test_scoping_matches_the_reference_on_cpu(ref) -> None:
    c = lucid.randn(4, 4)
    got = _scenarios(
        lambda enabled=True: autocast(device_type="cpu", dtype=lucid.bfloat16, enabled=enabled),
        lambda: _dtype_name((c @ c).dtype),
    )
    r = ref.randn(4, 4)
    want = _scenarios(
        lambda enabled=True: ref.autocast(device_type="cpu", dtype=ref.bfloat16, enabled=enabled),
        lambda: _dtype_name((r @ r).dtype),
    )
    assert got == want


def test_engine_scope_untraced_and_running_stats_dtype() -> None:
    """LCD-242 / LCD-500-C5: `untraced()` restores thread-local engine state and norm running stats stay float32 under autocast."""
    from lucid._C import engine as _C_engine
    from lucid._engine_scope import untraced
    import lucid.nn as nn
    import lucid.nn.functional as F

    # Outside trace: untraced() is a no-op.
    assert _C_engine.compile.current_tracer() is None
    with untraced():
        assert _C_engine.compile.current_tracer() is None

    # Inside trace: untraced() detaches the tracer and restores it on exit.
    with lucid.compile._tracing() as tracer:
        assert _C_engine.compile.current_tracer() is tracer
        with untraced():
            assert _C_engine.compile.current_tracer() is None
        assert _C_engine.compile.current_tracer() is tracer
        _ = F.dropout(lucid.ones(4), p=0.5, training=True)

    # Inside autocast: running stats stay float32 without an explicit autocast_dtype() guard.
    x = lucid.randn(2, 3, 4)
    bn = nn.BatchNorm1d(3)
    inorm = nn.InstanceNorm1d(3, track_running_stats=True)
    rm = lucid.zeros(3)
    rv = lucid.ones(3)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        assert _C_engine.amp_active_dtype(_C_engine.Device.CPU) == _C_engine.Dtype.BF16
        _ = bn(x)
        _ = inorm(x)
        _ = F.instance_norm(x, running_mean=rm, running_var=rv, use_input_stats=True)
        assert bn.running_mean.dtype == lucid.float32  # type: ignore[union-attr]
        assert inorm.running_mean.dtype == lucid.float32  # type: ignore[union-attr]
        assert rm.dtype == lucid.float32
