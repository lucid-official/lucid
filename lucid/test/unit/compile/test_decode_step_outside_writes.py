"""A compiled decode step carries, or declines, a write to a tensor from outside (LCD-860).

``compiled_decode_step`` wrote back the cache's own buffers and nothing else.  A
decode forward that also updated a tensor it did not create — a running tally, a
view of a buffer, a batch norm's running statistics — recorded the write like
any other op, but nobody asked the executable for the value: every compiled
step left the tensor where the trace-time call had put it, so the update
happened once.  After three steps the tally read ``[2.0]`` against eager's
``[3.0]``, with no error and no fallback.

The step now asks for what ``lucid.compile`` asks for (:class:`CarriedWrites`):
a write landing in an outside tensor is put back after the trace and carried on
every run, a batch norm's statistics included.  A write with nowhere to go (one
that lands in the call's own arguments) runs eagerly, as does the rest.  The
cache's own buffers are not such a write — they are rebound by the step itself —
so the decode stays compiled.

Every case runs one decode against a twin that runs eagerly, and compares the
answer and every tensor the step could have touched after each call.  A later
call is what tells these bugs from a correct answer: the first compiled one
agrees either way.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

Tensor = lucid.Tensor

_CALLS = 6
_LEN = 8
_WIDTH = 4


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
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()


@dataclass
class _Rig:
    """One decode step's model: its state, and the body that writes to it."""

    cache: StaticCache
    tally: Tensor
    vec: Tensor
    mean: Tensor
    var: Tensor
    bn: nn.BatchNorm1d
    cumulative: nn.BatchNorm1d | None
    body: Callable[[_Rig, Tensor, Tensor], Tensor]
    stash: list[Tensor] = field(default_factory=list)

    def forward(self, hidden: Tensor, cache_position: Tensor) -> Tensor:
        """The cache write every decode step makes, with the case's body around it."""
        key = (self.body(self, hidden, cache_position).mean(0) * 1.0).reshape(1, 1, 1, _WIDTH)
        keys, _ = self.cache.update(key, key, 0, {"cache_position": cache_position})
        return keys * 1.0

    def state(self) -> dict[str, list[float]]:
        """Everything the step could have written, as plain numbers."""
        found = {
            "tally": self.tally,
            "vec": self.vec,
            "mean": self.mean,
            "var": self.var,
            "running_mean": self.bn.running_mean,
            "running_var": self.bn.running_var,
            "counter": self.bn.num_batches_tracked,
        }
        if self.cumulative is not None:
            found["cumulative_mean"] = self.cumulative.running_mean
            found["cumulative_counter"] = self.cumulative.num_batches_tracked
        for i, buf in enumerate(self.cache.key_cache):
            found[f"key{i}"] = buf
        return {name: [float(v) for v in t.reshape(-1).tolist()] for name, t in found.items()}


def _rig(body: Callable[[_Rig, Tensor, Tensor], Tensor], *, cumulative: bool = False) -> _Rig:
    lucid.manual_seed(0)
    return _Rig(
        StaticCache(_LEN),
        lucid.zeros(1, device=COMPILE_DEVICE),
        lucid.zeros(_WIDTH, device=COMPILE_DEVICE),
        lucid.zeros(_WIDTH, device=COMPILE_DEVICE),
        lucid.ones(_WIDTH, device=COMPILE_DEVICE),
        nn.BatchNorm1d(_WIDTH).to(COMPILE_DEVICE).train(),
        nn.BatchNorm1d(_WIDTH, momentum=None).to(COMPILE_DEVICE).train() if cumulative else None,
        body,
    )


def _pass(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    return hidden * 1.0


def _outside_add(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    r.tally.add_(1.0)
    return hidden * 1.0


def _outside_reads_own_value(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    r.tally.mul_(2.0).add_(hidden.sum() * 0.01)
    return hidden * 1.0


def _view_write(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    r.vec[:2].add_(1.0)
    return hidden * 1.0


def _view_taken_then_written(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    window = r.vec.reshape(2, 2)
    window[1].add_(hidden.sum() * 0.1)
    return hidden * 1.0


def _write_read_back(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    r.vec.add_(hidden.mean(0))
    return hidden * 1.0 + r.vec


def _batch_norm_module(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    return r.bn(hidden)


def _batch_norm_functional(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    return F.batch_norm(hidden, r.mean, r.var, None, None, training=True, momentum=0.3)


def _batch_norm_cumulative(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    assert r.cumulative is not None
    return r.cumulative(hidden)


def _write_with_the_batch_norm(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    r.tally.add_(2.0)
    return r.bn(hidden) + r.vec.add_(1.0)


def _argument_write(r: _Rig, hidden: Tensor, cp: Tensor) -> Tensor:
    cp_before = cp * 1
    r.tally.add_(cp_before.sum().to(dtype=r.tally.dtype))
    hidden.add_(0.5)
    return hidden * 1.0


class _Case(NamedTuple):
    body: Callable[[_Rig, Tensor, Tensor], Tensor]
    #: Whether the decode still compiles: a write that lands in one of the
    #: call's own arguments cannot be carried to the next call's, nor can a
    #: cumulative average.
    compiled: bool
    cumulative: bool = False


CASES: dict[str, _Case] = {
    "cache only": _Case(_pass, True),
    "outside add_": _Case(_outside_add, True),
    "outside reads its own value": _Case(_outside_reads_own_value, True),
    "write through a view": _Case(_view_write, True),
    "view taken, then written": _Case(_view_taken_then_written, True),
    "written, then read back": _Case(_write_read_back, True),
    "batch norm module": _Case(_batch_norm_module, True),
    "batch norm functional": _Case(_batch_norm_functional, True),
    "writes and batch norm": _Case(_write_with_the_batch_norm, True),
    "batch norm, cumulative average": _Case(_batch_norm_cumulative, False, cumulative=True),
    "write to an argument": _Case(_argument_write, False),
}


def _inputs(pos: int) -> tuple[Tensor, Tensor]:
    base = lucid.arange(3 * _WIDTH, device=COMPILE_DEVICE).reshape(3, _WIDTH) * 0.1
    hidden = base * (pos + 1.0) + 0.05 * pos
    return hidden, lucid.tensor([pos], device=COMPILE_DEVICE).long()


class _Run(NamedTuple):
    answers: list[list[float]]
    states: list[dict[str, list[float]]]
    entries: int


def _run(case: _Case, compiled: bool) -> _Run:
    rig = _rig(case.body, cumulative=case.cumulative)

    def forward(hidden: Tensor, cache_position: Tensor) -> Tensor:
        return rig.forward(hidden, cache_position)

    step = compiled_decode_step(forward, rig.cache) if compiled else forward
    answers: list[list[float]] = []
    states: list[dict[str, list[float]]] = []
    for pos in range(_CALLS):
        hidden, cp = _inputs(pos)
        out = step(hidden, cp)
        answers.append([float(v) for v in out.reshape(-1).tolist()])
        states.append(rig.state())
    entries = len(step.cache) if compiled else 0  # type: ignore[attr-defined]
    return _Run(answers, states, entries)


def _assert_close(got: list[float], want: list[float], what: str) -> None:
    assert len(got) == len(want), what
    worst = max((abs(g - w) for g, w in zip(got, want, strict=True)), default=0.0)
    assert worst < 1e-4, f"{what}: compiled {got} against eager {want}"


@pytest.mark.parametrize("name", CASES)
def test_a_decode_step_ends_each_call_where_eager_does(name: str) -> None:
    """The answer and every tensor the step wrote match an eager twin, call by call."""
    case = CASES[name]
    eager = _run(case, compiled=False)
    comp = _run(case, compiled=True)
    for call in range(_CALLS):
        _assert_close(comp.answers[call], eager.answers[call], f"{name}: answer {call}")
        assert comp.states[call].keys() == eager.states[call].keys()
        for tensor, want in eager.states[call].items():
            _assert_close(comp.states[call][tensor], want, f"{name}: {tensor} after call {call}")


@pytest.mark.parametrize("name", CASES)
def test_a_decode_step_stays_compiled_unless_a_write_has_nowhere_to_go(name: str) -> None:
    """The cache's own buffer writes never make a step eager; one into an argument does."""
    case = CASES[name]
    entries = _run(case, compiled=True).entries
    assert entries == (1 if case.compiled else 0), f"{name}: {entries} executables"


def test_a_write_into_an_argument_says_why_it_runs_eagerly() -> None:
    """The refusal is a notice, not a silent difference."""
    rig = _rig(_argument_write)

    def forward(hidden: Tensor, cache_position: Tensor) -> Tensor:
        return rig.forward(hidden, cache_position)

    step = compiled_decode_step(forward, rig.cache)
    with pytest.warns(fallback.CompileFallbackWarning, match="its own arguments"):
        for pos in range(3):
            step(*_inputs(pos))


@pytest.mark.parametrize("compiled", [False, True])
def test_a_cache_written_in_place_is_the_cache_not_an_outside_write(compiled: bool) -> None:
    """A step that writes its cache's buffers in place stays compiled and agrees with eager.

    The buffers are fed and written like any outside tensor, yet they are the
    cache, which the step rolls forward itself: carrying them as well must not
    refuse the trace or write them twice.
    """
    cache = StaticCache(_LEN)

    def forward(hidden: Tensor, cache_position: Tensor) -> Tensor:
        key = (hidden.mean(0) * 1.0).reshape(1, 1, 1, _WIDTH)
        if len(cache.key_cache) == 0:
            cache.update(key, key, 0, {"cache_position": cache_position})
        else:
            cache.key_cache[0].index_copy_(2, cache_position, key)
            cache.value_cache[0].index_copy_(2, cache_position, key)
            cache._cumulative_length[0] += 1
            cache._seen_tokens += 1
        return cache.key_cache[0] * 1.0

    step = compiled_decode_step(forward, cache) if compiled else forward
    for pos in range(_CALLS):
        out = step(*_inputs(pos))
    if compiled:
        assert len(step.cache) == 1  # type: ignore[attr-defined]
    twin = StaticCache(_LEN)
    for pos in range(_CALLS):
        hidden, cp = _inputs(pos)
        key = hidden.mean(0).reshape(1, 1, 1, _WIDTH)
        twin.update(key, key, 0, {"cache_position": cp})
    want = [float(v) for v in twin.key_cache[0].reshape(-1).tolist()]
    _assert_close([float(v) for v in cache.key_cache[0].reshape(-1).tolist()], want, "cache")
    _assert_close([float(v) for v in out.reshape(-1).tolist()], want, "answer")
