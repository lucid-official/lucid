"""Compiled gathers and scatters move every bit of a wide integer (LCD-435).

MPSGraph's gather family carries 32- and 64-bit integers through float32,
so a compiled ``gather`` turned 2^40 + 3 into 2^40, 2^62 + 5 into 2^62 and
int32 2^24 + 1 into 2^24; its scatter family wrote int64 in 32 bits and bool
as false.  Every compiled forward gather and scatter now goes through
``OpEmitters/_DataMovement.h``, which carries those dtypes as int16 lanes
(bool as int8) and sums int64 by half-words.  This matrix holds each op that
lowers to a gather or a scatter — and the movement ops next to them — to the
eager answer bit for bit, compared as Python ints, and checks that the call
really compiled.  Float graphs are untouched; the last tests train a float
model through the same emitters against eager.
"""

import json
import random
import re
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile import fused_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")

# Near ±2^62, ±2^40 and 2^24 ± 1: float32 keeps none of their low bits.
WIDE = {
    "int64": [
        2**40 + 1,
        2**40 + 3,
        -(2**40) - 7,
        2**62 + 5,
        -(2**62) - 3,
        2**24 + 1,
        2**24 - 1,
        -(2**24 + 1),
    ],
    "int32": [
        2**31 - 1,
        -(2**31),
        2**24 + 1,
        2**24 - 1,
        -(2**24 + 1),
        2**30 + 3,
        -(2**30) - 5,
        123456789,
    ],
}
DTYPES = {"int64": lucid.int64, "int32": lucid.int32}


def _x(dtype: str) -> lucid.Tensor:
    return lucid.tensor(WIDE[dtype], dtype=DTYPES[dtype]).to(COMPILE_DEVICE)


def _index(values: list[int] | list[list[int]]) -> lucid.Tensor:
    return lucid.tensor(values).to(COMPILE_DEVICE)


def _ints(out: object) -> list[object]:
    values: list[object] = []
    for part in out if isinstance(out, tuple) else (out,):
        assert isinstance(part, lucid.Tensor), type(part)
        values.append(part.to("cpu").tolist())
    return values


def _same(fn: Callable[..., object], *args: lucid.Tensor, compiled: bool = True) -> None:
    _C_engine.compile.session_cache_clear()
    c = lucid.compile(fn)
    got = c(*args)
    info = c.cache_info()
    if compiled:
        assert info["entries"] == 1 and not info["eager_only"], info
    assert _ints(got) == _ints(fn(*args))


PERM = [3, 7, 1, 0, 6, 2, 5, 4]
MASK = [True, False, True, True, False, False, True, False]


@dataclass(frozen=True)
class _Idx:
    """Index operands, made on the device before the trace starts (a tensor
    built inside the traced function would be a host constant)."""

    perm: lucid.Tensor
    rows: lucid.Tensor
    outer: lucid.Tensor
    negative: lucid.Tensor
    pair: lucid.Tensor
    slot0: lucid.Tensor
    slot2: lucid.Tensor
    mask: lucid.Tensor


def _idx() -> _Idx:
    return _Idx(
        perm=_index(PERM),
        rows=_index([[1, 0, 3, 2], [0, 0, 1, 3]]),
        outer=_index([[1, 0, 1, 0]]),
        negative=_index([-1, -8, 2]),
        pair=_index([5, 1]),
        slot0=_index([0] * 8),
        slot2=_index([2] * 8),
        mask=_index(MASK),
    )


OPS: dict[str, Callable[[lucid.Tensor, _Idx], object]] = {
    "gather": lambda t, i: lucid.gather(t, 0, i.perm) + 0,
    "gather_inner_axis": lambda t, i: lucid.gather(t.reshape(2, 4), 1, i.rows) + 0,
    "gather_outer_axis": lambda t, i: lucid.gather(t.reshape(2, 4), 0, i.outer) + 0,
    "index_select": lambda t, i: lucid.index_select(t, 0, i.perm) + 0,
    "take_along_dim": lambda t, i: lucid.take_along_dim(t.reshape(2, 4), i.rows, 1) + 0,
    "getitem_index": lambda t, i: t[i.perm] + 0,
    "getitem_negative_index": lambda t, i: t[i.negative] + 0,
    "getitem_int": lambda t, i: t.reshape(2, 4)[1] + 0,
    "getitem_slice": lambda t, i: t[1:7:2] + 0,
    "scatter": lambda t, i: lucid.zeros_like(t).scatter(0, i.perm, t) + 0,
    "scatter_into_wide": lambda t, i: t.scatter(0, i.pair, t[:2]) + 0,
    "scatter_add": lambda t, i: t.scatter_add(0, i.perm, t) + 0,
    "scatter_add_one_slot": lambda t, i: t.scatter_add(0, i.slot0, t) + 0,
    "index_put": lambda t, i: lucid.zeros_like(t).index_put((i.perm,), t) + 0,
    "index_put_accumulate": lambda t, i: t.index_put((i.slot2,), t, accumulate=True) + 0,
    "topk": lambda t, i: tuple(o + 0 for o in lucid.topk(t, 5)),
    "sort": lambda t, i: lucid.sort(t, descending=True)[0] + 0,
    "cat": lambda t, i: lucid.cat([t, lucid.flip(t, [0])]) + 0,
    "stack": lambda t, i: lucid.stack([t, t]) + 0,
    "flip": lambda t, i: lucid.flip(t, [0]) + 0,
    "where": lambda t, i: lucid.where(i.mask, t, lucid.flip(t, [0])) + 0,
    "embedding": lambda t, i: F.embedding(i.perm, t.reshape(8, 1)) + 0,
    "interpolate_nearest": lambda t, i: (
        F.interpolate(t.reshape(1, 1, 8), size=13, mode="nearest") + 0
    ),
}


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_wide_integers_come_back_whole(dtype: str, op: str) -> None:
    i = _idx()
    _same(lambda t: OPS[op](t, i), _x(dtype))


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_boolean_mask_selection_is_whole(dtype: str) -> None:
    """A mask's size depends on its values, so this one runs eagerly
    (``nonzero``); the answer still has to be eager's."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        mask = _index(MASK)
        _same(lambda t: t[mask] + 0, _x(dtype), compiled=False)


def test_bool_scatter_writes_true() -> None:
    """MPSGraph's scatter wrote bool as false; it is carried as int8."""
    src, perm = _index(MASK), _index(PERM)
    _same(lambda s: lucid.zeros_like(s).scatter(0, perm, s), src)
    _same(lambda s: lucid.zeros_like(s).index_put((perm,), s), src)


def test_bool_sort_values() -> None:
    """``sort`` gathers bool values straight, no longer through an int32 key."""
    _same(lambda s: lucid.sort(s, descending=True)[0], _index(MASK))


def test_int64_sum_carries_across_half_words() -> None:
    """Every carry the half-word sum makes: 0xFFFF in both low half-words of
    many updates landing on one slot, and high words that wrap int32."""
    vals = [2**32 - 1, 2**48 - 1, -1, 2**63 - 1, -(2**63), 0xFFFF_FFFF_FFFF, 2**62 + 5, -7] * 8
    x = lucid.tensor(vals, dtype=lucid.int64).to(COMPILE_DEVICE)
    idx = _index([i % 3 for i in range(len(vals))])
    _same(lambda t: lucid.zeros_like(t[:3]).scatter_add(0, idx, t) + 0, x)


def test_int64_sum_over_a_long_update_axis_runs_eagerly() -> None:
    """More than 2^15 updates may meet in one slot, past what an int32
    half-word sum holds exactly, so the emitter declines and eager sums."""
    n = (1 << 15) + 1
    x = (lucid.arange(n, dtype=lucid.int64) * (2**40 + 3)).to(COMPILE_DEVICE)
    idx = lucid.zeros(n, dtype=lucid.int64).to(COMPILE_DEVICE)

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return lucid.zeros_like(t[:1]).scatter_add(0, idx, t) + 0

    _C_engine.compile.session_cache_clear()
    c = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        got = c(x)
    info = c.cache_info()
    assert info["entries"] == 0 and info["eager_only"], info
    assert _ints(got) == _ints(fn(x))


def _random_int64(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(64) - 2**63 for _ in range(n)]


def _assign(t: lucid.Tensor, i: lucid.Tensor) -> lucid.Tensor:
    out = lucid.zeros_like(t[:3])
    out[i] = t
    return out + 0


COLLIDING: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "scatter": lambda t, i: lucid.zeros_like(t[:3]).scatter(0, i, t) + 0,
    "index_put": lambda t, i: lucid.zeros_like(t[:3]).index_put((i,), t) + 0,
    "setitem": _assign,
}


@pytest.mark.parametrize("n", [64, 4096, 65536])
@pytest.mark.parametrize("spread", ["one_slot", "three_slots"])
@pytest.mark.parametrize("op", list(COLLIDING))
def test_colliding_int64_overwrites_keep_one_whole_update(op: str, spread: str, n: int) -> None:
    """Where several updates aim at one slot the winner is unspecified, but it
    is one of them, whole.  Scattered as int16 lanes, each lane picked its own
    winner and the slot held a mix of two updates (0xcb4a0d781cc5675 from
    updates 0 and 32); the winner is now chosen once, on its position."""
    slots = [0] * n if spread == "one_slot" else [k % 3 for k in range(n)]
    idx = _index(slots)
    fn = COLLIDING[op]
    _C_engine.compile.session_cache_clear()
    c = lucid.compile(fn)
    for run in range(4):
        vals = _random_int64(n, seed=1000 * n + run)
        got = c(lucid.tensor(vals, dtype=lucid.int64).to(COMPILE_DEVICE), idx).to("cpu").tolist()
        for slot, value in enumerate(got):
            landed = {v for v, k in zip(vals, slots, strict=True) if k == slot}
            assert value in (landed or {0}), (slot, hex(value))
    info = c.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def test_int64_scatter_drops_an_out_of_range_update() -> None:
    """Policy B (``_IndexBounds.h``) holds on the carried path: the graph is
    built on in-range indices, and a later call's stray one is dropped."""
    x = _x("int64")
    _C_engine.compile.session_cache_clear()

    def fn(t: lucid.Tensor, i: lucid.Tensor) -> lucid.Tensor:
        return t.scatter(0, i, t) + 0

    c = lucid.compile(fn)
    c(x, _index(PERM))
    stray = _index([3, 7, 1, 0, 6, 2, 5, 40])
    want = fn(x.to("cpu"), lucid.tensor([3, 7, 1, 0, 6, 2, 5, 4]))
    got = c(x, stray).to("cpu").tolist()
    # Positions 0..7 except slot 4 take their update; slot 4 keeps x[4].
    expect = want.tolist()
    expect[4] = WIDE["int64"][4]
    assert got == expect


_DYNAMIC_PROGRAM = textwrap.dedent("""
    import json
    import lucid
    from lucid._C import engine as _C_engine

    V = [2**40 + 1, 2**40 + 3, -(2**40) - 7, 2**62 + 5]
    fns = {
        "scatter": lambda t, i: t.scatter(1, i, t) + 0,
        "scatter_add": lambda t, i: t.scatter_add(1, i, t) + 0,
        "gather": lambda t, i: lucid.gather(t, 1, i) + 0,
        "topk": lambda t, i: lucid.topk(t, 2, dim=1)[0] + 0,
    }
    for dtype in ("int64", "float32"):
        for name, fn in fns.items():
            _C_engine.compile.session_cache_clear()
            c = lucid.compile(fn, dynamic=True)
            for b in (2, 3, 5):
                x = lucid.tensor([V] * b, dtype=getattr(lucid, dtype)).to("metal")
                i = lucid.tensor([[3, 2, 1, 0]] * b).to("metal")
                same = c(x, i).to("cpu").tolist() == fn(x, i).to("cpu").tolist()
                info = c.cache_info()
                print(json.dumps({"case": f"{dtype}/{name}/{b}", "ok": same,
                                  "entries": info["entries"],
                                  "eager": len(info["eager_only"])}), flush=True)
""")


def test_a_dynamic_batch_moves_wide_integers_without_aborting() -> None:
    """One executable serves every batch.  A dynamic-batch overwrite scatter
    used to abort the process inside MPSGraph — its sink slab was a constant
    of the base's shape, which a symbolic batch does not have — so this runs
    in a subprocess."""
    run = subprocess.run(
        [sys.executable, "-c", _DYNAMIC_PROGRAM],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 24, run.stdout
    for row in rows:
        assert row["ok"] and row["entries"] == 1 and row["eager"] == 0, row


class _GatherNet(nn.Module):
    """Float data through each emitter the helper wraps: embedding, a
    take_along_dim gather, topk values and a cross-entropy class gather."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.fc = nn.Linear(8, 5)

    def forward(self, idx: lucid.Tensor) -> lucid.Tensor:
        h = self.emb(idx)
        picked = lucid.take_along_dim(h, lucid.zeros_like(idx).unsqueeze(-1).expand(*h.shape), 1)
        top = lucid.topk(h + picked, 2, dim=1)[0].sum(dim=1)
        return self.fc(top)


def test_float_training_through_the_wrapped_emitters_matches_eager() -> None:
    lucid.manual_seed(0)
    idx = lucid.randint(0, 16, size=(4, 6)).to(COMPILE_DEVICE)
    tgt = lucid.randint(0, 5, size=(4,)).to(COMPILE_DEVICE)
    eager_model = _GatherNet().to(COMPILE_DEVICE)
    comp_model = _GatherNet().to(COMPILE_DEVICE)
    comp_model.load_state_dict(eager_model.state_dict())

    eager_opt = optim.SGD(list(eager_model.parameters()), lr=0.1)
    step = fused_step(comp_model, F.cross_entropy, optim.SGD(list(comp_model.parameters()), lr=0.1))
    for _ in range(3):
        eager_opt.zero_grad()
        eager_loss = F.cross_entropy(eager_model(idx), tgt)
        eager_loss.backward()
        eager_opt.step()
        comp_loss = step(idx, tgt)
        assert abs(float(eager_loss.item()) - float(comp_loss.item())) < 1e-5
    for (name, pe), (_, pc) in zip(
        eager_model.named_parameters(), comp_model.named_parameters(), strict=True
    ):
        diff = float((pe.detach() - pc.detach()).abs().max().item())
        assert diff < 1e-5, name


_EMITTERS = Path(lucid.__file__).parent / "_C" / "compile" / "OpEmitters"
_RAW_MOVE = re.compile(
    r"\b(gatherAlongAxis(?:Tensor)?|gatherWithUpdatesTensor|gatherNDWithUpdatesTensor"
    r"|scatterAlongAxis(?:Tensor)?|scatterWithDataTensor|scatterWithUpdatesTensor"
    r"|scatterNDWithDataTensor|scatterNDWithUpdatesTensor)\s*:"
)
# ``@selector(scatterAlongAxis:...)`` asks whether the method exists; it moves nothing.
_SELECTOR = re.compile(r"@selector\([^)]*\)")


@pytest.mark.skipif(not _EMITTERS.is_dir(), reason="engine sources not in this install")
def test_forward_emitters_move_data_only_through_the_helper() -> None:
    """A forward emitter sees every dtype, so it must not reach an MPSGraph
    gather or scatter except through ``_DataMovement.h``.  The one other
    caller is ``_Resample.h``'s backward scatter, which only ever sums a
    floating gradient.  (VJP emitters live elsewhere and see gradients only.)"""
    allowed = {"_DataMovement.h": None, "_Resample.h": {"scatterWithUpdatesTensor"}}
    offenders = []
    for path in sorted(_EMITTERS.rglob("*")):
        if path.suffix not in {".h", ".mm"}:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = _SELECTOR.sub("", line.split("//", 1)[0])
            for m in _RAW_MOVE.finditer(code):
                ok = allowed.get(path.name, set())
                if ok is None or m.group(1) in ok:
                    continue
                offenders.append(f"{path.relative_to(_EMITTERS)}:{lineno}: {m.group(1)}")
    assert not offenders, "\n".join(offenders)
