"""``shadow_alloc`` belongs to the thread that opens it.

The block exists so the docs pipeline can build a model without
allocating it, and it gets there by swapping entry points every thread
shares.  What keeps that from reaching anyone else is pinned here: a
thread outside the block — one started inside it included — computes
real values, two blocks open in two threads each last until their own
exit, and once the last block closes nothing in the process is left
bound to a stand-in, not even a module first imported inside a block.

The threads meet at events, never at sleeps, so every interleaving
below is the one the test names.
"""

import sys
import threading
import types
from collections.abc import Callable

import pytest

import lucid
import lucid._dispatch as _dispatch
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._ops._registry import _REGISTRY
from lucid.nn._shadow import is_active, is_shadow, shadow_alloc
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

# Seconds.  Only a guard against a deadlocked test, never a pacing delay.
_TIMEOUT = 30.0


class _Worker(threading.Thread):
    """A thread that keeps what its body raised, for ``finish`` to re-raise."""

    def __init__(self, body: Callable[[], None]) -> None:
        super().__init__(daemon=True)
        self._body = body
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._body()
        except BaseException as exc:  # re-raised on the test's thread by finish()
            self.error = exc

    def finish(self) -> None:
        self.join(_TIMEOUT)
        assert not self.is_alive(), "the worker thread never finished"
        if self.error is not None:
            raise self.error


def _wait(event: threading.Event) -> None:
    assert event.wait(_TIMEOUT), "the other thread never reached its checkpoint"


# ── other threads compute for real ────────────────────────────────────────────


@pytest.mark.parametrize("device", _DEVICES)
def test_another_thread_computes_for_real_while_a_block_is_open(device: str) -> None:
    entered, release = threading.Event(), threading.Event()
    holder_saw: dict[str, bool] = {}

    def hold() -> None:
        with shadow_alloc():
            holder_saw["shadow"] = is_shadow(lucid.zeros(3, device=device))
            entered.set()
            _wait(release)

    holder = _Worker(hold)
    holder.start()
    try:
        _wait(entered)
        x = lucid.zeros(3, device=device)
        assert not is_active()
        assert not is_shadow(x)
        assert x.tolist() == [0.0, 0.0, 0.0]
        assert (x + 1).tolist() == [1.0, 1.0, 1.0]
        layer = nn.Linear(2, 2).to(device)
        assert not is_shadow(layer.weight)
        out = layer(lucid.ones(1, 2, device=device))
        assert not is_shadow(out)
        assert out.shape == (1, 2)
    finally:
        release.set()
        holder.finish()
    assert holder_saw == {"shadow": True}


def test_a_thread_started_inside_a_block_computes_for_real() -> None:
    child_saw: dict[str, object] = {}

    def child() -> None:
        x = lucid.zeros(3)
        child_saw.update(active=is_active(), shadow=is_shadow(x), value=x.tolist())

    with shadow_alloc():
        worker = _Worker(child)
        worker.start()
        worker.finish()
        assert is_shadow(lucid.zeros(3))
    assert child_saw == {"active": False, "shadow": False, "value": [0.0, 0.0, 0.0]}


def _overlap(inside_second: Callable[[], None]) -> None:
    """A enters, B enters, A exits, then ``inside_second`` runs in B's block."""
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def first() -> None:
        with shadow_alloc():
            a_in.set()
            _wait(b_in)
        a_out.set()

    def second() -> None:
        _wait(a_in)
        with shadow_alloc():
            b_in.set()
            _wait(a_out)
            inside_second()

    a, b = _Worker(first), _Worker(second)
    a.start()
    b.start()
    a.finish()
    b.finish()


def test_overlapping_blocks_each_last_until_their_own_exit() -> None:
    b_saw: dict[str, bool] = {}

    def look() -> None:
        b_saw["active"] = is_active()
        b_saw["shadow"] = is_shadow(lucid.zeros(4))

    _overlap(look)
    assert b_saw == {"active": True, "shadow": True}
    after = lucid.zeros(4)
    assert not is_shadow(after)
    assert after.tolist() == [0.0, 0.0, 0.0, 0.0]


# ── nothing is left behind once the last block closes ─────────────────────────

_LATE_MODULE = "lucid_test_shadow_late_import"
# Read at collection, before any block has opened.
_ORIGINAL_UNWRAPS = {"_unwrap": _dispatch._unwrap, "_unwrap_or_scalar": _dispatch._unwrap_or_scalar}


def _single() -> None:
    with shadow_alloc():
        lucid.zeros(2)


def _nested() -> None:
    with shadow_alloc():
        with shadow_alloc():
            lucid.zeros(2)
        lucid.zeros(2)


def _raising() -> None:
    with pytest.raises(RuntimeError, match="boom"), shadow_alloc():
        raise RuntimeError("boom")


def _overlapping() -> None:
    _overlap(lambda: None)


def _late_import() -> None:
    """Import a module for the first time inside the block, as a lazy import would."""
    with shadow_alloc():
        mod = types.ModuleType(_LATE_MODULE)
        sys.modules[_LATE_MODULE] = mod
        exec(
            "from lucid._dispatch import _unwrap, _unwrap_or_scalar\n"
            "from lucid._C.engine import zeros\n",
            mod.__dict__,
        )


def _foreign_unwraps() -> list[str]:
    """Every module binding ``_unwrap`` / ``_unwrap_or_scalar`` to a foreign callable."""
    leaks: list[str] = []
    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue
        for name in ("_unwrap", "_unwrap_or_scalar"):
            value = vars(mod).get(name)
            if value is None or value is _ORIGINAL_UNWRAPS[name]:
                continue
            # A module's own helper that happens to share the name.
            if getattr(value, "__module__", None) == mod_name:
                continue
            leaks.append(f"{mod_name}.{name} -> {value!r}")
    return leaks


@pytest.mark.parametrize(
    "scenario",
    [_single, _nested, _raising, _overlapping, _late_import],
    ids=["single", "nested", "raising", "overlapping", "late-import"],
)
def test_the_last_exit_leaves_no_stand_in_bound(
    scenario: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_before = {
        name: getattr(_C_engine, name) for name in dir(_C_engine) if not name.startswith("_")
    }
    registry_before = [entry.engine_fn for entry in _REGISTRY]
    assert not _foreign_unwraps()
    # Registered first so the late-import module is removed after the test.
    monkeypatch.setitem(sys.modules, _LATE_MODULE, types.ModuleType(_LATE_MODULE))

    scenario()

    assert not is_active()
    assert _C_engine.zeros is engine_before["zeros"]
    changed = [
        name for name, value in engine_before.items() if getattr(_C_engine, name) is not value
    ]
    assert not changed
    assert all(
        entry.engine_fn is before for entry, before in zip(_REGISTRY, registry_before, strict=True)
    )
    assert not _foreign_unwraps()
    late = sys.modules[_LATE_MODULE]
    if scenario is _late_import:
        assert late._unwrap is _ORIGINAL_UNWRAPS["_unwrap"]
        assert late._unwrap_or_scalar is _ORIGINAL_UNWRAPS["_unwrap_or_scalar"]
        assert late.zeros is engine_before["zeros"]
    real = lucid.zeros(3)
    assert not is_shadow(real)
    assert real.tolist() == [0.0, 0.0, 0.0]
