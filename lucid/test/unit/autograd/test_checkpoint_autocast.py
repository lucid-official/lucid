"""``checkpoint`` recomputes its segment under the autocast state the forward ran in (LCD-739).

The segment runs twice: under ``no_grad`` in the forward, then again during
backward to rebuild its graph.  ``checkpoint`` replayed the random stream
for the second run but nothing about autocast, so under AMP the recompute
ran in float32 where the forward had run in bfloat16 / float16 and the
gradients belonged to a different network: a weight gradient of
``[-2.5515, -0.1658, -1.2060]`` where the direct run gave
``[-2.5469, -0.1680, -1.2031]``, in both ``use_reentrant`` modes.

The forward's state is now captured once per device in a ``_ForwardState``
beside the RNG state and entered around the recompute — on at the same
dtype, or off.  The checkpointed run's gradients equal the direct run's
exactly, and the recompute sees the forward's state whatever scope
``backward()`` is called from.
"""

import itertools
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.amp import autocast
from lucid.autograd import checkpoint
from lucid.test._fixtures.devices import metal_available
from lucid.utils.checkpoint import checkpoint as utils_checkpoint

#: (device, autocast dtype) cells the table runs.
CELLS = [("cpu", lucid.bfloat16), ("cpu", lucid.float16)] + (
    [("metal", lucid.float16), ("metal", lucid.bfloat16)] if metal_available() else []
)


def _one(out: lucid.Tensor | tuple[lucid.Tensor, ...]) -> lucid.Tensor:
    assert isinstance(out, lucid.Tensor)
    return out


#: The checkpoint entry points; ``utils`` routes through the autograd one.
ENTRIES: dict[str, Callable[..., lucid.Tensor]] = {
    "reentrant": lambda fn, x: _one(checkpoint(fn, x, use_reentrant=True)),
    "non_reentrant": lambda fn, x: _one(checkpoint(fn, x, use_reentrant=False)),
    "utils": lambda fn, x: _one(utils_checkpoint(fn, x)),
}

State = tuple[bool, lucid.dtype | None]


class _Net(nn.Module):
    """Two linears with an activation between, so rounding in the first reaches the second."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.a = nn.Linear(3, 5).to(device)
        self.b = nn.Linear(5, 3).to(device)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.b(F.gelu(self.a(x)))


def _grads(net: _Net, x: lucid.Tensor) -> list[list[float]]:
    parts = [net.a.weight.grad, net.a.bias.grad, net.b.weight.grad, net.b.bias.grad, x.grad]
    return [p.tolist() for p in parts if p is not None]


def _run(
    device: str, dtype: lucid.dtype, entry: str | None
) -> tuple[lucid.Tensor, list[list[float]]]:
    """One forward + backward under autocast; ``entry`` None is the direct run."""
    lucid.manual_seed(0)
    net = _Net(device)
    x = lucid.randn(4, 3, device=device).requires_grad_(True)
    with autocast(device, dtype=dtype):
        y = net(x) if entry is None else ENTRIES[entry](net, x)
    y.float().sum().backward()
    return y, _grads(net, x)


@pytest.fixture(autouse=True)
def _amp_off_around_each_test() -> object:
    assert not autocast.is_autocast_enabled(), "a previous test left autocast on"
    yield
    assert not autocast.is_autocast_enabled(), "this test left autocast on"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("device, dtype", CELLS, ids=lambda v: str(v))
def test_checkpointed_gradients_equal_the_direct_runs(
    device: str, dtype: lucid.dtype, entry: str
) -> None:
    direct_y, direct = _run(device, dtype, None)
    kept_y, kept = _run(device, dtype, entry)
    assert kept_y.dtype == direct_y.dtype
    assert kept_y.tolist() == direct_y.tolist()
    assert kept == direct


# ── what the recompute sees ──────────────────────────────────────────────────


def _watching(log: list[tuple[State, State]]) -> Callable[[lucid.Tensor], lucid.Tensor]:
    """A segment that logs both devices' autocast state each time it runs."""

    def segment(x: lucid.Tensor) -> lucid.Tensor:
        log.append(
            tuple(
                (autocast.is_autocast_enabled(d), autocast.get_autocast_dtype(d))
                for d in ("cpu", "metal")
            )
        )
        return F.gelu(x @ x.mT)

    return segment


def _scope(device: str, dtype: lucid.dtype | None) -> autocast:
    """``dtype`` None is autocast forced off."""
    return autocast(device, dtype=dtype, enabled=dtype is not None)


# (forward scope, scope backward() is called from): the recompute must show
# the forward's state, not the backward's.
SCOPES = [
    (lucid.bfloat16, None),
    (lucid.bfloat16, lucid.float16),
    (lucid.bfloat16, lucid.bfloat16),
    (None, lucid.bfloat16),
    (None, None),
]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("forward, backward", SCOPES, ids=lambda v: str(v))
def test_the_recompute_sees_the_forwards_state_not_the_backwards(
    device: str, entry: str, forward: lucid.dtype | None, backward: lucid.dtype | None
) -> None:
    log: list[tuple[State, State]] = []
    segment = _watching(log)
    x = lucid.randn(4, 3, device=device).requires_grad_(True)
    with _scope(device, forward):
        y = ENTRIES[entry](segment, x)
    with _scope(device, backward):
        y.sum().backward()
    assert len(log) == 2, "the segment runs once forward and once to recompute"
    assert log[1] == log[0]


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_off_scope_inside_an_on_scope_is_recomputed_off(device: str, entry: str) -> None:
    log: list[tuple[State, State]] = []
    segment = _watching(log)
    x = lucid.randn(4, 3, device=device).requires_grad_(True)
    with autocast(device, dtype=lucid.bfloat16):
        with autocast(device, enabled=False):
            y = ENTRIES[entry](segment, x)
    with autocast(device, dtype=lucid.bfloat16):
        y.sum().backward()
    assert log[0] == log[1]
    assert all(not enabled for enabled, _ in log[1])


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_devices_state_is_replayed_independently(entry: str) -> None:
    # Autocast on one device only: the other device stays off in the recompute.
    log: list[tuple[State, State]] = []
    segment = _watching(log)
    x = lucid.randn(4, 3, requires_grad=True)
    with autocast("cpu", dtype=lucid.bfloat16):
        y = ENTRIES[entry](segment, x)
    y.sum().backward()
    cpu_first, metal_first = log[0]
    assert cpu_first == (True, lucid.bfloat16) and metal_first == (False, None)
    assert log[1] == log[0]


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_scope_the_recompute_enters_is_left_again(device: str, entry: str) -> None:
    x = lucid.randn(4, 3, device=device).requires_grad_(True)
    with autocast(device, dtype=lucid.bfloat16):
        y = ENTRIES[entry](lambda t: t @ t.mT, x)
    y.sum().backward()
    assert not autocast.is_autocast_enabled()
    assert autocast.get_autocast_dtype() is None


def test_a_failing_segment_still_unwinds_the_replay(device: str) -> None:
    calls = itertools.count()

    def segment(t: lucid.Tensor) -> lucid.Tensor:
        if next(calls) == 1:  # the recompute
            raise RuntimeError("boom")
        return t @ t.mT

    x = lucid.randn(4, 3, device=device).requires_grad_(True)
    before = lucid.get_rng_state()
    with autocast(device, dtype=lucid.bfloat16):
        y = checkpoint(segment, x, use_reentrant=True)
    with pytest.raises(RuntimeError, match="boom"):
        y.sum().backward()
    assert not autocast.is_autocast_enabled()
    assert lucid.get_rng_state().tolist() == before.tolist()


def test_the_rng_state_is_still_replayed_beside_autocast(device: str) -> None:
    # A dropout inside an autocast segment masks the same units twice.
    lin = nn.Linear(6, 6).to(device)

    def segment(t: lucid.Tensor) -> lucid.Tensor:
        return F.dropout(lin(t), p=0.5, training=True)

    def run(entry: str | None) -> list[list[float]]:
        lucid.manual_seed(7)
        lin.weight.grad = None
        x = lucid.randn(8, 6, device=device).requires_grad_(True)
        with autocast(device, dtype=lucid.bfloat16):
            y = segment(x) if entry is None else ENTRIES[entry](segment, x)
        y.float().sum().backward()
        assert lin.weight.grad is not None and x.grad is not None
        return [lin.weight.grad.tolist(), x.grad.tolist()]

    direct = run(None)
    for entry in ENTRIES:
        assert run(entry) == direct, entry


def test_the_reference_holds_the_same_invariant(ref: ModuleType) -> None:
    # Documents the contract this follows: with autocast on, checkpointed
    # gradients equal the direct run's in both modes.
    ref_checkpoint = ref.utils.checkpoint.checkpoint

    def grads(reentrant: bool | None) -> list[float]:
        ref.manual_seed(0)
        lin = ref.nn.Linear(3, 3)
        x = ref.randn(2, 3, requires_grad=True)
        with ref.autocast("cpu", dtype=ref.bfloat16):
            y = lin(x) if reentrant is None else ref_checkpoint(lin, x, use_reentrant=reentrant)
        y.float().sum().backward()
        return [float(v) for v in lin.weight.grad.flatten().tolist()]

    direct = grads(None)
    assert grads(True) == direct
    assert grads(False) == direct
