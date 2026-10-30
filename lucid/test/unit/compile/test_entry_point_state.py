"""Every compile entry point keys on, and writes back, the state its trace reads.

One defect class, found entry point by entry point.  An executable is the same
only for calls made in the state it was traced in — each module's ``training``
flag and the attributes a forward branches on, and the autocast region — and a
trace of a training BatchNorm leaves its statistics and counter to be written
back after every run.

* ``fused_step`` and the segmented ``make_step`` keyed on their inputs alone:
  after ``model.eval()``, a dropout's new ``p`` or a BatchNorm's new ``eps``
  they replayed the graph of the state the model had left (LCD-720).
* Every key saw public scalar attributes only: a forward branching on a
  private flag or on a list of names replayed the stale branch (LCD-704).
* The segmented ``make_step`` never wrote its BatchNorm statistics back: they
  stayed at their initial values and the counter at 0 (LCD-719).
* No key saw the autocast region: a model compiled outside it answered
  float32 inside a float16 region.

So every entry point is held to eager after each kind of change made once it
has traced, and over three training steps of a BatchNorm and the evaluation
that follows them.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import override

import pytest

import lucid
import lucid.nn as nn
from lucid.amp import autocast
from lucid.compile import compiled_step, fused_step, make_step
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.utils.cache import StaticCache

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


class _Gate(nn.Module):
    """Branches on a private flag and on a list of names, as a LoRA layer does."""

    def __init__(self) -> None:
        super().__init__()
        self._off = False
        self.modes: list[str] = []

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self._off:
            x = x * 0.5
        if "shift" in self.modes:
            x = x + 1.0
        return x


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.0),
        _Gate(),
        nn.Linear(16, 4),
    )
    return model.to(COMPILE_DEVICE)


def _batch(seed: int = 1) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(6, 8, device=COMPILE_DEVICE)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).mean()


_Run = Callable[[lucid.Tensor], lucid.Tensor]


@dataclass(frozen=True)
class _EntryPoint:
    """How one entry point is compiled for a model, and the eager call it stands for."""

    build: Callable[[nn.Sequential], _Run]
    eager: Callable[[nn.Sequential, lucid.Tensor], lucid.Tensor]


def _forward(model: nn.Sequential) -> _Run:
    compiled = lucid.compile(model)

    def run(x: lucid.Tensor) -> lucid.Tensor:
        with lucid.no_grad():
            out = compiled(x)
        assert isinstance(out, lucid.Tensor)
        return out

    return run


def _eager_forward(model: nn.Sequential, x: lucid.Tensor) -> lucid.Tensor:
    with lucid.no_grad():
        out = model(x)
    assert isinstance(out, lucid.Tensor)
    return out


def _stepping(
    make: Callable[[nn.Sequential], Callable[..., lucid.Tensor]],
) -> Callable[[nn.Sequential], _Run]:
    def build(model: nn.Sequential) -> _Run:
        step = make(model)
        return lambda x: step(x).detach()

    return build


def _fused(model: nn.Sequential) -> Callable[..., lucid.Tensor]:
    # lr=0: the parameters stay where the eager reference's are.
    return fused_step(model, _loss, lucid.optim.SGD(model.parameters(), lr=0.0))


def _direct(model: nn.Sequential) -> _Run:
    return lambda x: compiled_step(model, x, _loss).detach()


def _eager_loss(model: nn.Sequential, x: lucid.Tensor) -> lucid.Tensor:
    out = model(x)
    assert isinstance(out, lucid.Tensor)
    return _loss(out).detach()


ENTRIES: dict[str, _EntryPoint] = {
    "compile": _EntryPoint(_forward, _eager_forward),
    "make_step": _EntryPoint(_stepping(lambda m: make_step(m, _loss)), _eager_loss),
    "segmented": _EntryPoint(_stepping(lambda m: make_step(m, _loss, segments=2)), _eager_loss),
    "fused_step": _EntryPoint(_stepping(_fused), _eager_loss),
    "compiled_step": _EntryPoint(_direct, _eager_loss),
}


def _nothing(model: nn.Sequential) -> None:
    del model


def _set(index: int, name: str, value: object) -> Callable[[nn.Sequential], None]:
    return lambda model: setattr(model[index], name, value)


def _shift(model: nn.Sequential) -> None:
    gate = model[4]
    assert isinstance(gate, _Gate)
    gate.modes.append("shift")


@dataclass(frozen=True)
class _Change:
    """What changes once the entry point has traced, the modes it is made in,
    and the region the next call is made in.

    In eval mode every entry point compiles; in training mode the forward-only
    ``lucid.compile`` runs a BatchNorm that tracks statistics eagerly.
    """

    mutate: Callable[[nn.Sequential], object]
    modes: tuple[str, ...] = ("train", "eval")
    setup: Callable[[nn.Sequential], object] = _nothing
    region: Callable[[], AbstractContextManager[object]] = nullcontext
    tol: float = 1e-4


CHANGES: dict[str, _Change] = {
    "eval": _Change(lambda m: m.eval(), modes=("train",)),
    "train": _Change(lambda m: m.train(), modes=("eval",)),
    "dropout p": _Change(_set(3, "p", 0.0), modes=("train",), setup=_set(3, "p", 0.5)),
    "bn eps": _Change(_set(1, "eps", 0.5)),
    "bn momentum": _Change(_set(1, "momentum", 0.9), modes=("train",)),
    "bn track_running_stats": _Change(_set(1, "track_running_stats", False)),
    "public scalar": _Change(_set(2, "negative_slope", 0.7)),
    "private flag": _Change(_set(4, "_off", True)),
    "list state": _Change(_shift),
    "autocast": _Change(
        _nothing, region=lambda: autocast(COMPILE_DEVICE, dtype=lucid.float16), tol=5e-3
    ),
}

_CASES = [
    (entry, change, mode)
    for entry in ENTRIES
    for change, how in CHANGES.items()
    for mode in how.modes
]


def _assert_close(got: lucid.Tensor, want: lucid.Tensor, tol: float, what: str) -> None:
    assert got.dtype == want.dtype, f"{what}: dtype {got.dtype} vs eager {want.dtype}"
    a, b = got.to(lucid.float32).numpy(), want.to(lucid.float32).numpy()
    worst = float(abs(a - b).max()) if a.size else 0.0
    scale = max(float(abs(b).max()) if b.size else 0.0, 1.0)
    assert worst <= tol * scale, f"{what}: off eager by {worst:.3e}"


def _assert_same_buffers(model: nn.Module, ref: nn.Module, what: str) -> None:
    mine, theirs = dict(model.named_buffers()), dict(ref.named_buffers())
    assert mine.keys() == theirs.keys()
    for name, buf in mine.items():
        _assert_close(buf, theirs[name], 1e-4, f"{what}, buffer {name}")


@pytest.mark.parametrize("entry, change, mode", _CASES)
def test_a_change_after_the_first_call_is_followed(entry: str, change: str, mode: str) -> None:
    point, how = ENTRIES[entry], CHANGES[change]
    model, ref = _net(), _net()
    for m in (model, ref):
        m.train(mode == "train")
        how.setup(m)
    run = point.build(model)
    x = _batch()
    run(x)
    point.eager(ref, x)
    for m in (model, ref):
        how.mutate(m)
    with how.region():
        got = run(x)
        want = point.eager(ref, x)
    what = f"{entry} in {mode} mode after {change}"
    _assert_close(got, want, how.tol, what)
    _assert_same_buffers(model, ref, what)


# ``segments="auto"`` times a few training steps of each candidate split on the
# first call; those runs must leave the statistics as they found them.  With a
# segment per child, the batch norm's segment runs before the gate's — an
# identity, nothing to compile — sends the step eager: the statistics the
# first wrote must not count again.
_TRAINING_ENTRIES: dict[str, _EntryPoint] = {
    **ENTRIES,
    "segmented, a later segment eager": _EntryPoint(
        _stepping(lambda m: make_step(m, _loss, segments=6)), _eager_loss
    ),
    "segmented auto": _EntryPoint(
        _stepping(lambda m: make_step(m, _loss, segments="auto")), _eager_loss
    ),
}


@pytest.mark.parametrize("entry", list(_TRAINING_ENTRIES))
def test_batch_norm_statistics_follow_eager_through_training(entry: str) -> None:
    point = _TRAINING_ENTRIES[entry]
    model, ref = _net(), _net()
    run = point.build(model)
    for seed in (1, 2, 3):
        x = _batch(seed)
        run(x)
        point.eager(ref, x)
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm1d)
    assert int(bn.num_batches_tracked.item()) == 3, f"{entry}: the counter did not count"
    _assert_same_buffers(model, ref, f"{entry} after three steps")
    model.eval()
    ref.eval()
    x = _batch(4)
    _assert_close(run(x), point.eager(ref, x), 1e-4, f"{entry} evaluated after training")


# ── the decode step, which keys on the modules its function reaches ───────


class _Decoder(nn.Module):
    """Writes one key/value slot of a StaticCache, then gates what it read."""

    def __init__(self) -> None:
        super().__init__()
        self.kv = nn.Linear(4, 4)
        self.gate = _Gate()

    @override
    def forward(
        self, h: lucid.Tensor, cache: StaticCache, cache_position: lucid.Tensor
    ) -> lucid.Tensor:
        k = self.kv(h).reshape(1, 1, 1, 4)
        keys, _ = cache.update(k, k, 0, {"cache_position": cache_position})
        return self.gate(keys.sum(dim=2))


_DECODE_CHANGES: dict[str, Callable[[_Decoder], object]] = {
    "private flag": lambda d: setattr(d.gate, "_off", True),
    "list state": lambda d: d.gate.modes.append("shift"),
}


def _decode(model: _Decoder, compiled: bool, change: Callable[[_Decoder], object]) -> lucid.Tensor:
    """Three decode steps — the first fills the cache eagerly, the second
    traces — with ``change`` made before the third; its output."""
    cache = StaticCache(4)

    def forward(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
        return model(h, cache, cp)

    step = compiled_decode_step(forward, cache) if compiled else forward

    def at(pos: int) -> lucid.Tensor:
        h = lucid.ones(1, 1, 4, device=COMPILE_DEVICE) * (pos + 1.0)
        return step(h, lucid.tensor([pos], device=COMPILE_DEVICE).long())

    with lucid.no_grad():
        at(0)
        at(1)
        change(model)
        return at(2)


@pytest.mark.parametrize("change", list(_DECODE_CHANGES))
def test_a_decode_step_follows_the_state_of_the_modules_it_reaches(change: str) -> None:
    mutate = _DECODE_CHANGES[change]
    lucid.manual_seed(0)
    model = _Decoder().to(COMPILE_DEVICE).eval()
    want = _decode(model, compiled=False, change=mutate)
    lucid.manual_seed(0)
    model = _Decoder().to(COMPILE_DEVICE).eval()
    got = _decode(model, compiled=True, change=mutate)
    _assert_close(got, want, 1e-4, f"decode after {change}")
