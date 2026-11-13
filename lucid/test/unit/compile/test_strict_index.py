"""``LUCID_STRICT_INDEX=1`` and compiled programs (LCD-991).

Strict mode makes Metal raise the CPU's ``IndexError`` for an index outside its axis
(LCD-892).  The check reads the index back when the op is dispatched; a compiled replay
dispatches nothing, and its graph isolates a bad index in place (policy B), so a replay used
to answer what strict mode exists to refuse.  Under strict mode a program that holds an index
op is now not compiled: its signature runs eagerly, where the gates refuse.

``test_index_bounds_contract.py`` compiles every index op the contract knows and checks the
CPU's text; this file checks each entry point, the cache key, the notice, and that the list
of ops (:data:`INDEX_OPS`) is the one the engine's emitters hold.
"""

import re
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import compiled_step, fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.index_strict import INDEX_OPS
from lucid.compile._core.signature import signature_of
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.optim import SGD
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")
quiet = pytest.mark.filterwarnings("ignore:lucid.compile:UserWarning")

_V, _D, _B, _S = 16, 4, 2, 3
_SWITCH = "LUCID_STRICT_INDEX"


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(_V, _D)
        self.lin = nn.Linear(_D, _D)

    def forward(self, ids: lucid.Tensor) -> lucid.Tensor:
        return self.lin(self.emb(ids))


def _ids(bad: bool = False) -> lucid.Tensor:
    values = [[(i * _S + j) % _V for j in range(_S)] for i in range(_B)]
    if bad:
        values[1][1] = _V
    return lucid.tensor(values, dtype=lucid.int64).to(COMPILE_DEVICE)


def _target() -> lucid.Tensor:
    return lucid.ones(_B, _S, _D).to(COMPILE_DEVICE)


def _net() -> _Net:
    lucid.manual_seed(0)
    return _Net().to(COMPILE_DEVICE)


def _mse(y: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(y, t)


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SWITCH, "1")


@pytest.fixture
def default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_SWITCH, raising=False)


# ── the cache key ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "strict_on"),
    [("1", True), ("true", True), ("yes", True), ("0", False), ("", False), ("0abc", False)],
)
def test_the_switch_is_part_of_the_key(
    monkeypatch: pytest.MonkeyPatch, value: str, strict_on: bool
) -> None:
    """Read at every call, by the same rule as the guards: a signature traced with the switch
    on is another key than one traced with it off."""
    model, x = nn.Linear(4, 2), lucid.ones(3, 4)
    monkeypatch.delenv(_SWITCH, raising=False)
    base = signature_of(model, (x,), {}, dynamic=False)
    assert not base.policy[-1]
    monkeypatch.setenv(_SWITCH, value)
    key = signature_of(model, (x,), {}, dynamic=False)
    assert bool(key.policy[-1]) is strict_on
    assert (key != base) is strict_on
    monkeypatch.delenv(_SWITCH)
    assert signature_of(model, (x,), {}, dynamic=False) == base


# ── the list of ops ─────────────────────────────────────────────────────────────────

_EMITTERS = Path(__file__).resolve().parents[3] / "_C" / "compile"
#: The emitters that put a caller's index through a gate of ``OpEmitters/_IndexBounds.h``.
_GATE_USERS = {
    "OpEmitters/index/Gather.mm",
    "OpEmitters/index/Scatter.mm",
    "OpEmitters/nn/Loss.mm",
    "VjpEmitters/nn/Embedding.mm",
    "VjpEmitters/nn/Loss.mm",
    "VjpEmitters/shape/Reorder.mm",
}
_GATES = re.compile(r"\b(graph_axis_index|graph_class_target)\(")


def test_every_emitter_that_gates_an_index_is_one_this_module_knows() -> None:
    """A new emitter that calls an index gate holds an index op: add its op to ``INDEX_OPS``
    (and the file here), or a strict-mode replay of it answers policy B."""
    sources = [p for pattern in ("*.mm", "*.h", "*.cpp") for p in _EMITTERS.rglob(pattern)]
    users = {
        str(path.relative_to(_EMITTERS))
        for path in sources
        if path.name != "_IndexBounds.h" and _GATES.search(path.read_text())
    }
    assert users == _GATE_USERS


@pytest.mark.parametrize("source", ["OpEmitters/index/Gather.mm", "OpEmitters/index/Scatter.mm"])
def test_the_ops_of_the_gather_and_scatter_emitters_are_listed(source: str) -> None:
    text = (_EMITTERS / source).read_text()
    names = set(re.findall(r'op_name\(\) const override \{ return "(\w+)"', text))
    names |= set(re.findall(r'make_unique<\w+(?:<\d+>)?>\("(\w+)"\)', text))
    assert names, f"no op names found in {source}: this test's patterns need an update"
    assert names <= INDEX_OPS, names - INDEX_OPS


# ── every entry point ───────────────────────────────────────────────────────────────


@needs_metal
@quiet
@pytest.mark.usefixtures("inference", "strict")
def test_a_compiled_forward_refuses_a_bad_index_on_every_call() -> None:
    compiled = lucid.compile(_net())
    compiled(_ids())
    compiled(_ids())
    for _ in range(2):
        with pytest.raises(IndexError, match="index 16 is out of range"):
            compiled(_ids(bad=True))
    info = compiled.cache_info()
    assert info["entries"] == 0
    assert [key.policy[-1] for key in info["eager_only"]] == [True]


@needs_metal
@quiet
@pytest.mark.usefixtures("inference", "strict")
def test_the_trace_call_refuses_too_and_remembers_nothing() -> None:
    compiled = lucid.compile(_net())
    with pytest.raises(IndexError, match="index 16 is out of range"):
        compiled(_ids(bad=True))
    info = compiled.cache_info()
    assert info["entries"] == 0
    assert info["eager_only"] == ()


@needs_metal
@quiet
@pytest.mark.usefixtures("inference", "default")
def test_the_default_still_compiles_and_answers_policy_b() -> None:
    net = _net()
    compiled = lucid.compile(net)
    compiled(_ids())
    got = compiled(_ids(bad=True))
    want = net(_ids(bad=True))
    assert compiled.cache_info()["entries"] == 1
    assert [bool(v != v) for v in got.reshape(-1, _D)[:, 0].tolist()] == [
        bool(v != v) for v in want.reshape(-1, _D)[:, 0].tolist()
    ]
    assert any(v != v for v in got.reshape(-1).tolist())


@needs_metal
@quiet
@pytest.mark.usefixtures("inference")
def test_a_verdict_under_strict_neither_poisons_nor_outlives_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declined with the switch on, compiled with it off, and the two do not share a verdict."""
    compiled = lucid.compile(_net())
    monkeypatch.setenv(_SWITCH, "1")
    compiled(_ids())
    assert compiled.cache_info()["entries"] == 0
    monkeypatch.delenv(_SWITCH)
    compiled(_ids())
    assert compiled.cache_info()["entries"] == 1
    monkeypatch.setenv(_SWITCH, "1")
    with pytest.raises(IndexError):
        compiled(_ids(bad=True))
    monkeypatch.delenv(_SWITCH)
    assert any(v != v for v in compiled(_ids(bad=True)).reshape(-1).tolist())
    assert compiled.cache_info()["entries"] == 1


@needs_metal
@pytest.mark.usefixtures("inference", "strict")
def test_a_program_without_an_index_op_still_compiles() -> None:
    compiled = lucid.compile(nn.Linear(_D, _D).to(COMPILE_DEVICE))
    x = lucid.ones(_B, _D).to(COMPILE_DEVICE)
    compiled(x)
    compiled(x)
    assert compiled.cache_info()["entries"] == 1


@needs_metal
@pytest.mark.usefixtures("inference", "strict")
def test_the_notice_names_the_switch_and_the_ops_once() -> None:
    compiled = lucid.compile(_net())
    with pytest.warns(CompileFallbackWarning, match=r"LUCID_STRICT_INDEX=1.*embedding"):
        compiled(_ids())
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        compiled(_ids())
        compiled(_ids())
        lucid.compile(_net())(_ids())  # another signature of the same kind: still once
    assert not [w for w in recorded if issubclass(w.category, CompileFallbackWarning)]


def _train_step_refuses(
    make: Callable[[_Net], Callable[[lucid.Tensor, lucid.Tensor], object]],
) -> None:
    """A training step: good calls work, a bad index raises, and the parameters stay put."""
    net = _net()
    step = make(net)
    step(_ids(), _target())
    step(_ids(), _target())
    before = [p.tolist() for p in net.parameters()]
    with pytest.raises(IndexError, match="index 16 is out of range"):
        step(_ids(bad=True), _target())
    assert [p.tolist() for p in net.parameters()] == before


@needs_metal
@quiet
@pytest.mark.usefixtures("strict")
def test_make_step_refuses_a_bad_index() -> None:
    _train_step_refuses(lambda net: make_step(net, _mse))


@needs_metal
@quiet
@pytest.mark.usefixtures("strict")
def test_fused_step_refuses_a_bad_index() -> None:
    _train_step_refuses(lambda net: fused_step(net, _mse, SGD(net.parameters(), lr=0.1)))


@needs_metal
@quiet
@pytest.mark.usefixtures("strict")
def test_a_segmented_make_step_refuses_a_bad_index() -> None:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Embedding(_V, _D), nn.Linear(_D, _D), nn.Linear(_D, _D))
    step = make_step(net.to(COMPILE_DEVICE), _mse, segments=2)
    step(_ids(), _target())
    step(_ids(), _target())
    with pytest.raises(IndexError, match="index 16 is out of range"):
        step(_ids(bad=True), _target())


@needs_metal
@quiet
@pytest.mark.usefixtures("strict")
def test_compiled_step_traces_every_call_and_so_refuses_on_the_trace() -> None:
    net, target = _net(), _target()
    compiled_step(net, _ids(), lambda y: _mse(y, target))
    with pytest.raises(IndexError, match="index 16 is out of range"):
        compiled_step(net, _ids(bad=True), lambda y: _mse(y, target))


@needs_metal
@quiet
@pytest.mark.usefixtures("default")
@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
def test_a_training_step_without_the_switch_is_unchanged(entry: str) -> None:
    """Off: the step compiles, and a bad index gives the NaN loss eager Metal gives."""
    net = _net()
    twin = _net()
    if entry == "make_step":
        step = make_step(net, _mse)
        cache = vars(step)["cache"]
    else:
        fused = fused_step(net, _mse, SGD(net.parameters(), lr=0.1))
        step, cache = fused, fused._plans
    step(_ids(), _target())
    assert len(cache) == 1
    loss = step(_ids(bad=True), _target()).item()
    eager = _mse(twin(_ids(bad=True)), _target()).item()
    assert loss != loss and eager != eager


# ── the decode step ─────────────────────────────────────────────────────────────────

_L = 8


def _decode_steps(positions: list[int]) -> tuple[int, list[lucid.Tensor]]:
    lucid.manual_seed(0)
    cache = StaticCache(_L)
    proj = nn.Linear(_D, _D).to(COMPILE_DEVICE)

    def forward(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
        k = proj(h).reshape(1, 1, 1, _D)
        keys, _ = cache.update(k, k, 0, {"cache_position": cp})
        return keys.sum(dim=(1, 2, 3)) + h.sum()

    step = compiled_decode_step(forward, cache)
    outs = []
    for pos in positions:
        h = lucid.ones(1, 1, _D).to(COMPILE_DEVICE) * 0.1
        outs.append(step(h, lucid.tensor([pos], dtype=lucid.int64).to(COMPILE_DEVICE)))
    return len(vars(step)["cache"]), outs


@needs_metal
@quiet
@pytest.mark.usefixtures("inference", "default")
def test_a_decode_step_without_the_switch_still_compiles() -> None:
    compiled, _ = _decode_steps([0, 1, 2, 3])
    assert compiled == 1


@needs_metal
@quiet
@pytest.mark.usefixtures("inference", "strict")
def test_a_decode_step_refuses_a_position_past_the_cache() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _decode_steps([0, 1, 2, 3, _L])
