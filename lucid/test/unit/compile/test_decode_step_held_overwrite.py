"""A held tensor overwritten at positions the decode step reads off its input (LCD-934).

``compiled_decode_step`` is the one entry whose trace-time call does not stand
as the call and whose executable then answers every call, the first included.
A tensor the forward makes and Python holds (a lazily built table, a closure's
cell) is built afresh inside that executable on every run; the Python-held one
stays as the trace-time call left it.  So an overwrite of it is owed to the
eager call only when its answer or its next value would differ: an overwrite at
constant positions lands the same values however often it runs, and the table
may stay compiled (``test_decode_step_outside_writes``); one at positions read
off the input lands elsewhere each call, and the table eager keeps is every
call's positions together, not the last call's.

The walk that decides it (``Tracer::reads_its_own_value``) keeps the base of
``masked_fill`` / ``scatter_set`` as a value that is merely passed along only
when the positions are constants (``Tracer::keeps_operand``): otherwise the held
tensor's earlier value is read, the trace is refused, and the step runs eagerly.
Without that clause the trace compiles and the second call answers from a table
holding the second call's overwrite alone.

Every case runs against an eager twin and compares the answer and the held
tensor after each call.
"""

import functools
from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

Tensor = lucid.Tensor

_CALLS = 5
_LEN = 8
_ROWS, _WIDTH = 3, 4


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


@pytest.fixture(autouse=True)
def _fresh_session() -> None:
    _C_engine.compile.session_cache_clear()


def _made() -> Tensor:
    return lucid.ones(_ROWS, _WIDTH, device=COMPILE_DEVICE)


def _constant_mask() -> Tensor:
    return lucid.arange(_ROWS * _WIDTH * 1.0, device=COMPILE_DEVICE).reshape(_ROWS, _WIDTH) > 6.0


Write = Callable[[Tensor, Tensor, Tensor], None]


def _masked_fill(t: Tensor, hidden: Tensor, _cp: Tensor) -> None:
    t.masked_fill_(hidden > 0.0, 7.0)


def _scatter(t: Tensor, hidden: Tensor, _cp: Tensor) -> None:
    t.scatter_(
        1, (hidden > 0.0).long() * 3, lucid.full((_ROWS, _WIDTH), 9.0, device=COMPILE_DEVICE)
    )


def _index_put(t: Tensor, hidden: Tensor, _cp: Tensor) -> None:
    rows = (hidden[:, 0] > 0.0).long() * 2
    t.index_put_((rows,), lucid.full((_WIDTH,), 3.0, device=COMPILE_DEVICE))


def _masked_fill_at_the_position(t: Tensor, _hidden: Tensor, cp: Tensor) -> None:
    t.masked_fill_(lucid.arange(_ROWS * 1.0, device=COMPILE_DEVICE).reshape(_ROWS, 1) == cp, 5.0)


def _masked_fill_at_constants(t: Tensor, _hidden: Tensor, _cp: Tensor) -> None:
    t.masked_fill_(_constant_mask(), 7.0)


#: Positions read off the call's arguments: each call overwrites elsewhere.
FROM_THE_INPUT: dict[str, Write] = {
    "masked_fill_": _masked_fill,
    "scatter_": _scatter,
    "index_put_": _index_put,
    "masked_fill_-at-the-cache-position": _masked_fill_at_the_position,
}


def _inputs(pos: int) -> tuple[Tensor, Tensor]:
    base = lucid.arange(_ROWS * _WIDTH * 1.0, device=COMPILE_DEVICE).reshape(_ROWS, _WIDTH)
    hidden = (base * 0.9 + pos * 2.1).sin()
    return hidden, lucid.tensor([pos % _ROWS], device=COMPILE_DEVICE).long()


class _Step:
    """A decode forward that overwrites a tensor it made and keeps in a table, and
    answers from it.  The table is an ``lru_cache``: a holder the entry's key
    does not see, so the executable traced on the call that built it answers
    every call after."""

    def __init__(self, write: Write) -> None:
        self.cache = StaticCache(_LEN)
        self.write = write
        self.table = functools.lru_cache(maxsize=1)(_made)

    def forward(self, hidden: Tensor, cache_position: Tensor) -> Tensor:
        # The step runs the call that creates the cache eagerly and traces the
        # next: the table is first built, and first written, inside that trace.
        built = len(self.cache.key_cache) > 0
        held = self.table() if built else _made()
        if built:
            self.write(held, hidden, cache_position)
        key = ((hidden * held).mean(0) * 1.0).reshape(1, 1, 1, _WIDTH)
        keys, _ = self.cache.update(key, key, 0, {"cache_position": cache_position})
        return keys * 1.0

    def kept(self) -> list[float]:
        if self.table.cache_info().currsize == 0:
            return []
        return [float(v) for v in self.table().reshape(-1).tolist()]


def _run(write: Write, *, compiled: bool) -> tuple[list[list[float]], list[list[float]], int]:
    rig = _Step(write)
    step = compiled_decode_step(rig.forward, rig.cache) if compiled else rig.forward
    answers: list[list[float]] = []
    kept: list[list[float]] = []
    for pos in range(_CALLS):
        out = step(*_inputs(pos))
        answers.append([float(v) for v in out.reshape(-1).tolist()])
        kept.append(rig.kept())
    return answers, kept, len(step.cache) if compiled else 0  # type: ignore[attr-defined]


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("write", FROM_THE_INPUT)
def test_a_held_table_overwritten_at_positions_from_the_input_runs_eagerly(write: str) -> None:
    """The call that builds the table runs eagerly, so that the next finds it as
    eager leaves it.  An executable traced on that call would answer the next
    one too, from a table built afresh with that call's overwrite alone."""
    got_answers, got_kept, _ = _run(FROM_THE_INPUT[write], compiled=True)
    want_answers, want_kept, _ = _run(FROM_THE_INPUT[write], compiled=False)
    for call in range(_CALLS):
        assert _close(got_answers[call], want_answers[call]), f"answer {call}"
        assert _close(got_kept[call], want_kept[call]), f"held tensor after call {call}"


def test_a_held_table_overwritten_at_constant_positions_stays_compiled() -> None:
    """The other side: the same values however often it runs, so nobody is owed
    the write and the step keeps its executable."""
    got_answers, got_kept, entries = _run(_masked_fill_at_constants, compiled=True)
    want_answers, want_kept, _ = _run(_masked_fill_at_constants, compiled=False)
    assert entries == 1
    for call in range(_CALLS):
        assert _close(got_answers[call], want_answers[call]), f"answer {call}"
        assert _close(got_kept[call], want_kept[call]), f"held tensor after call {call}"
