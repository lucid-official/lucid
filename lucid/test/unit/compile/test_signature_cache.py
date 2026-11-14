"""A wrapper's own cache of compiled signatures is bounded, least recently used out (LCD-1008).

An ``MPSGraphExecutable`` keeps the intermediate buffers MPSGraph planned for it
until it is released.  The session cache's byte cap (LCD-1000) drops only the
references the session cache holds; a live ``CompiledModule`` kept every signature's
executable in a plain dict, and so did ``make_step``, ``fused_step`` and the decode
step.  They now keep a :class:`SignatureCache`: an entry cap and a byte cap, oldest
use out, the entry just stored spared.

* the cache itself (no device needed): order, the newest spared, what counts as a use,
  ``0`` = no cap, the environment knobs;
* the wrappers (Metal): what is kept, that a dropped signature recompiles to the same
  answer, that the session cache's eviction and a module's are independent.

The footprint of one live module over many shapes is in ``test_cache_memory.py``;
that a module outlives the session cache's eviction is in
``test_executable_cache_eviction.py``.
"""

import copy
import pickle
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core import signature_cache as sc
from lucid.compile._core.signature_cache import SignatureCache, signature_limits
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

_MB = 1024 * 1024
_METAL = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


@dataclass
class _Entry:
    """What a wrapper stores: something that says how many bytes it pins."""

    name: str
    retained: int = 0


def _cache(
    max_entries: int = 0, max_bytes: int = 0, **sizes: int
) -> tuple[SignatureCache[str, _Entry], Callable[[str], _Entry]]:
    cache: SignatureCache[str, _Entry] = SignatureCache(max_entries, max_bytes)

    def put(name: str) -> _Entry:
        entry = _Entry(name, sizes.get(name, 0))
        cache[name] = entry
        return entry

    return cache, put


# ── the cache ───────────────────────────────────────────────────────────────


def test_the_least_recently_used_entry_goes_first_over_the_entry_cap() -> None:
    cache, put = _cache(max_entries=3)
    for name in "abc":
        put(name)
    assert cache.lookup("a") is not None  # a is now newer than b
    put("d")
    assert sorted(cache) == ["a", "c", "d"]
    put("e")
    assert sorted(cache) == ["a", "d", "e"]


def test_the_byte_cap_drops_the_oldest_until_what_is_left_fits() -> None:
    cache, put = _cache(max_bytes=10, a=4, b=4, c=4, d=4)
    for name in "abc":
        put(name)
    assert sorted(cache) == ["b", "c"] and cache.retained_bytes == 8
    put("d")
    assert sorted(cache) == ["c", "d"] and cache.retained_bytes == 8


def test_the_entry_just_stored_stays_however_large() -> None:
    """It is the one about to run: a cap smaller than it must not drop it."""
    cache, put = _cache(max_bytes=1, a=100, b=200)
    put("a")
    assert list(cache) == ["a"]
    put("b")
    assert list(cache) == ["b"] and cache.retained_bytes == 200


def test_the_entry_cap_never_empties_the_cache() -> None:
    cache, put = _cache(max_entries=1)
    put("a")
    put("b")
    assert list(cache) == ["b"]


def test_zero_means_no_cap() -> None:
    cache, put = _cache(max_entries=0, max_bytes=0, **{str(i): 10**12 for i in range(300)})
    for i in range(300):
        put(str(i))
    assert len(cache) == 300


def test_only_lookup_counts_as_a_use() -> None:
    """Introspection (``get``, ``[]``, ``items``, ``in``) must not keep an entry alive."""
    cache, put = _cache(max_entries=2)
    put("a")
    put("b")
    assert cache.get("a") is not None
    assert cache["a"] is not None
    assert "a" in cache
    assert {name for name, _ in cache.items()} == {"a", "b"}
    put("c")  # a was the oldest use all along
    assert sorted(cache) == ["b", "c"]


def test_a_missing_key_is_a_miss_not_a_use() -> None:
    cache, put = _cache(max_entries=2)
    put("a")
    assert cache.lookup("nope") is None
    assert cache.get("nope") is None
    with pytest.raises(KeyError):
        cache["nope"]


def test_storing_under_a_present_key_replaces_it_and_its_bytes() -> None:
    cache, put = _cache(max_bytes=100, a=60, b=30)
    put("a")
    put("b")
    cache["a"] = _Entry("a2", 10)
    assert len(cache) == 2 and cache["a"].name == "a2" and cache.retained_bytes == 40
    # ``a`` is the newest now: the oldest is ``b``.
    cache["c"] = _Entry("c", 80)
    assert sorted(cache) == ["a", "c"] and cache.retained_bytes == 90


def test_the_mapping_surface_a_dict_user_expects() -> None:
    cache, put = _cache()
    a, b = put("a"), put("b")
    assert len(cache) == 2 and bool(cache) and "a" in cache and "z" not in cache
    assert cache.pop("a") is a
    assert cache.pop("a", None) is None
    del cache["b"]
    assert not cache
    put("c")
    cache.clear()
    assert len(cache) == 0 and cache.retained_bytes == 0
    assert b.name == "b"


def test_a_cache_copies_and_pickles_like_the_dict_it_replaced() -> None:
    """A module that was never called is copied and pickled (``deepcopy(model)``, a save)."""
    cache, put = _cache(max_entries=3, a=5)
    put("a")
    for clone in (copy.deepcopy(cache), pickle.loads(pickle.dumps(cache))):
        assert list(clone) == ["a"] and clone.retained_bytes == 5 and clone.max_entries == 3
        clone["b"] = _Entry("b", 1)  # the lock is back
        assert len(clone) == 2 and len(cache) == 1


def test_iteration_survives_a_store_made_during_it() -> None:
    cache, put = _cache(max_entries=2)
    put("a")
    put("b")
    for name in cache:
        put(name + "!")  # evicts while iterating; a plain dict would raise here
    assert len(cache) == 2


def test_concurrent_use_neither_corrupts_nor_overshoots() -> None:
    cache, put = _cache(max_entries=8, **{str(i): 1 for i in range(64)})

    def worker(offset: int) -> None:
        for i in range(400):
            put(str((i + offset) % 64))
            cache.lookup(str((i * 7 + offset) % 64))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for done in [pool.submit(worker, k * 5) for k in range(4)]:
            done.result()  # re-raises what a worker raised
    assert len(cache) <= 8


# ── the environment ────────────────────────────────────────────────────────


@pytest.fixture
def quiet_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Neither knob set, and nothing reported yet: a reported variable is said once per process."""
    monkeypatch.delenv(sc.ENTRIES_ENV, raising=False)
    monkeypatch.delenv(sc.BYTES_ENV, raising=False)
    monkeypatch.setattr(sc, "_REPORTED", set())
    yield monkeypatch


def test_the_default_is_a_handful_of_hundreds_and_an_eighth_of_memory(
    quiet_environment: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    got = signature_limits()
    assert got.entries == sc.DEFAULT_MAX_ENTRIES == 256
    assert got.bytes == _C_engine.compile.default_cache_bytes() > 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("text", "want"), [("0", 0), ("1", 1), ("37", 37), ("100000", 100000)])
def test_the_entry_cap_takes_a_whole_number(
    quiet_environment: pytest.MonkeyPatch, text: str, want: int
) -> None:
    quiet_environment.setenv(sc.ENTRIES_ENV, text)
    assert signature_limits().entries == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("0", 0),
        ("123456", 123456),
        ("64k", 64 * 1024),
        ("64K", 64 * 1024),
        ("64m", 64 * _MB),
        ("2G", 2 * 1024**3),
    ],
)
def test_the_byte_cap_takes_a_number_with_an_optional_suffix(
    quiet_environment: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str, want: int
) -> None:
    quiet_environment.setenv(sc.BYTES_ENV, text)
    assert signature_limits().bytes == want
    assert capsys.readouterr().err == ""


_UNUSABLE = ["", "-5", "1x", "abc", "1.5g", " 12", "12 ", "99999999999999999999", "99999999999g"]


@pytest.mark.parametrize("text", _UNUSABLE)
def test_an_unusable_byte_cap_falls_back_and_says_so_once(
    quiet_environment: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    """``1g`` read as one byte would evict everything but the newest entry: reject, do not guess."""
    default = _C_engine.compile.default_cache_bytes()
    quiet_environment.setenv(sc.BYTES_ENV, text)
    assert signature_limits().bytes == default
    said = capsys.readouterr().err
    assert f"ignoring {sc.BYTES_ENV}" in said and f"using {default}" in said
    assert signature_limits().bytes == default
    assert capsys.readouterr().err == ""  # once


@pytest.mark.parametrize("text", [*_UNUSABLE, "64k", "1g"])
def test_an_unusable_entry_cap_falls_back_and_says_so_once(
    quiet_environment: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    """The entry cap is a plain count: a size suffix is as wrong as a sign."""
    quiet_environment.setenv(sc.ENTRIES_ENV, text)
    assert signature_limits().entries == sc.DEFAULT_MAX_ENTRIES
    said = capsys.readouterr().err
    assert f"ignoring {sc.ENTRIES_ENV}" in said
    assert signature_limits().entries == sc.DEFAULT_MAX_ENTRIES
    assert capsys.readouterr().err == ""


def test_a_cache_reads_the_environment_when_it_is_made(
    quiet_environment: pytest.MonkeyPatch,
) -> None:
    quiet_environment.setenv(sc.ENTRIES_ENV, "5")
    quiet_environment.setenv(sc.BYTES_ENV, "7k")
    made: SignatureCache[str, _Entry] = SignatureCache()
    assert (made.max_entries, made.max_bytes) == (5, 7 * 1024)
    quiet_environment.setenv(sc.ENTRIES_ENV, "9")
    assert made.max_entries == 5  # fixed for this cache
    explicit: SignatureCache[str, _Entry] = SignatureCache(max_entries=2, max_bytes=3)
    assert (explicit.max_entries, explicit.max_bytes) == (2, 3)


# ── the wrappers ───────────────────────────────────────────────────────────


def _mlp() -> nn.Module:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.fc1(x).relu())

    lucid.manual_seed(0)
    return _M().to(COMPILE_DEVICE)


def _entries(cm: object) -> list[object]:
    return list(cm._cache.values())  # type: ignore[attr-defined]


def _alive(entry: object, cm: object) -> bool:
    return any(e is entry for e in _entries(cm))


@_METAL
@pytest.mark.usefixtures("inference")
def test_a_module_keeps_its_newest_signatures_and_recompiles_a_dropped_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(sc.ENTRIES_ENV, "2")
    model = _mlp()
    cm = lucid.compile(model)
    xs = {b: metal_tensor(b, 8) for b in (2, 3, 4)}
    first = {b: cm(x).numpy() for b, x in xs.items()}
    assert cm.cache_info()["entries"] == 2
    kept = _entries(cm)  # the 3 and the 4

    again = cm(xs[2])  # 2 was dropped: traced and compiled once more
    assert cm.cache_info()["entries"] == 2
    assert not _alive(kept[0], cm), "the oldest of the two (3) should have gone"
    assert _alive(kept[1], cm)
    # The answer is the one it gave before it was dropped, and eager's.
    np.testing.assert_array_equal(again.numpy(), first[2])
    np.testing.assert_allclose(again.numpy(), model(xs[2]).numpy(), atol=1e-5)
    # ...and the entry that came back is a fresh one, run once.
    fresh = [e for e in _entries(cm) if e is not kept[1]]
    assert len(fresh) == 1 and fresh[0].n_hits == 1  # type: ignore[attr-defined]


@_METAL
def test_a_compiled_module_that_was_never_called_still_copies_and_pickles() -> None:
    cm = lucid.compile(nn.Sequential(nn.Linear(8, 4)).to(COMPILE_DEVICE))
    for clone in (copy.deepcopy(cm), pickle.loads(pickle.dumps(cm))):
        assert clone.cache_info()["entries"] == 0
        assert clone._cache.max_entries == cm._cache.max_entries  # type: ignore[attr-defined]


@_METAL
@pytest.mark.usefixtures("inference")
def test_a_hit_keeps_a_signature_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sc.ENTRIES_ENV, "2")
    cm = lucid.compile(_mlp())
    a, b, c = (metal_tensor(n, 8) for n in (2, 3, 4))
    cm(a)
    cm(b)
    ea, eb = _entries(cm)
    cm(a)  # a hit: a is now the newer of the two
    cm(c)
    assert _alive(ea, cm) and not _alive(eb, cm)


@_METAL
@pytest.mark.usefixtures("inference")
def test_cache_introspection_is_not_a_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sc.ENTRIES_ENV, "2")
    cm = lucid.compile(_mlp())
    cm(metal_tensor(2, 8))
    cm(metal_tensor(3, 8))
    ea, eb = _entries(cm)
    first_key = next(iter(cm.cache_info()["keys"]))
    cm.timing()
    cm.graph_dump()
    cm.graph_dump(first_key)
    cm(metal_tensor(4, 8))
    assert not _alive(ea, cm) and _alive(eb, cm)


@_METAL
@pytest.mark.usefixtures("inference")
def test_the_newest_signature_runs_under_a_byte_cap_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sc.BYTES_ENV, "1")
    model = _mlp()
    cm = lucid.compile(model)
    for b in (2, 3):
        x = metal_tensor(b, 8)
        np.testing.assert_allclose(cm(x).numpy(), model(x).numpy(), atol=1e-5)
        assert cm.cache_info()["entries"] == 1


@_METAL
@pytest.mark.usefixtures("inference")
def test_a_module_over_the_default_caps_keeps_every_signature_it_was_fed() -> None:
    """A loop's handful of shapes, and a few dozen: nothing is dropped by default."""
    cm = lucid.compile(_mlp())
    for b in range(1, 41):
        cm(metal_tensor(b, 8))
    assert cm.cache_info()["entries"] == 40


@_METAL
@pytest.mark.usefixtures("inference")
def test_the_entry_estimate_is_the_one_the_session_cache_counts() -> None:
    """One definition of "pinned bytes": the engine's, for a graph, either layer."""
    previous = _C_engine.compile.session_cache_set_max_bytes(0)
    try:
        _C_engine.compile.session_cache_clear()
        cm = lucid.compile(_mlp())
        cm(metal_tensor(16, 8))
        (entry,) = _entries(cm)
        assert entry.retained > 0  # type: ignore[attr-defined]
        assert entry.retained == _C_engine.compile.session_cache_bytes()  # type: ignore[attr-defined]
        assert cm._cache.retained_bytes == entry.retained  # type: ignore[attr-defined]
    finally:
        _C_engine.compile.session_cache_set_max_bytes(previous)
        _C_engine.compile.session_cache_clear()


@_METAL
@pytest.mark.usefixtures("inference")
def test_clearing_the_session_cache_does_not_touch_a_modules_signatures() -> None:
    """Independent layers: the session cache's eviction is not the module's."""
    cm = lucid.compile(_mlp())
    for b in (2, 3, 4):
        cm(metal_tensor(b, 8))
    _C_engine.compile.session_cache_clear()
    assert cm.cache_info()["entries"] == 3
    entries = _entries(cm)
    cm(metal_tensor(3, 8))
    assert all(_alive(e, cm) for e in entries)


@_METAL
def test_a_training_step_keeps_its_newest_signatures_and_trains_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_step`` drops the oldest signature; its recompile gives eager's loss and grads."""
    monkeypatch.setenv(sc.ENTRIES_ENV, "2")

    def build() -> nn.Linear:
        lucid.manual_seed(0)
        return nn.Linear(4, 3).to(COMPILE_DEVICE)

    def loss_fn(y: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        return ((y - t) ** 2).mean()

    compiled_model, eager_model = build(), build()
    step = lucid.compile.make_step(compiled_model, loss_fn)
    batches = [(metal_tensor(b, 4), metal_tensor(b, 3)) for b in (2, 3, 4)]

    def both(x: lucid.Tensor, t: lucid.Tensor) -> None:
        compiled_model.zero_grad()
        eager_model.zero_grad()
        got = step(x, t)
        got.backward()
        want = loss_fn(eager_model(x), t)
        want.backward()
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=1e-6)
        for pc, pe in zip(compiled_model.parameters(), eager_model.parameters(), strict=True):
            np.testing.assert_allclose(pc.grad.numpy(), pe.grad.numpy(), rtol=1e-5, atol=1e-6)

    for x, t in batches:
        both(x, t)
    assert len(step.cache) == 2  # type: ignore[attr-defined]
    kept = list(step.cache.values())  # type: ignore[attr-defined]
    both(*batches[0])  # dropped: compiled again
    after = list(step.cache.values())  # type: ignore[attr-defined]
    assert len(after) == 2
    assert not any(e is kept[0] for e in after) and any(e is kept[1] for e in after)
    assert not step.eager_only  # type: ignore[attr-defined]


@_METAL
def test_a_fused_step_keeps_its_newest_plans_and_trains_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fused_step`` is bounded by plans (least recently used out), not first in first out."""
    import lucid.nn.functional as F
    import lucid.optim as optim
    from lucid.compile._entry import fused_step as fused_module

    monkeypatch.setattr(fused_module, "_MAX_PLANS", 2)

    def build() -> tuple[nn.Linear, optim.SGD]:
        lucid.manual_seed(0)
        model = nn.Linear(4, 3).to(COMPILE_DEVICE)
        return model, optim.SGD(model.parameters(), lr=0.05)

    (fused_model, fused_opt), (eager_model, eager_opt) = build(), build()
    step = lucid.compile.fused_step(fused_model, F.mse_loss, fused_opt)
    batches = {b: (metal_tensor(b, 4), metal_tensor(b, 3)) for b in (2, 3, 4)}

    def both(b: int) -> None:
        x, t = batches[b]
        got = step(x, t)
        eager_opt.zero_grad()
        want = F.mse_loss(eager_model(x), t)
        want.backward()
        eager_opt.step()
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-5)

    for b in (2, 3, 2, 4):  # 2 is used again before 4 arrives: 3 is the one to go
        both(b)
    assert step.cache_info()["entries"] == 2
    both(3)  # dropped, so built again, and the run continues the same trajectory
    assert step.cache_info()["entries"] == 2
    for pf, pe in zip(fused_model.parameters(), eager_model.parameters(), strict=True):
        np.testing.assert_allclose(pf.numpy(), pe.numpy(), rtol=1e-4, atol=1e-5)


@_METAL
@pytest.mark.usefixtures("inference")
def test_the_decode_step_keeps_its_executables_in_a_signature_cache() -> None:
    """The decode step's one executable is counted like a module's, so its prefill lengths are bounded."""
    from lucid.compile._entry.decode_step import compiled_decode_step
    from lucid.test.unit.compile.test_compiled_decode import _B, _D, _L, _MaskedDecoder
    from lucid.utils.cache import StaticCache

    lucid.manual_seed(0)
    model = _MaskedDecoder().to(COMPILE_DEVICE).eval()
    cache = StaticCache(_L)
    step = compiled_decode_step(lambda h, cp: model(h, cache, cp), cache)
    for pos in range(3):
        h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE) * ((pos + 1) * 0.1)
        step(h, lucid.tensor([pos], device=COMPILE_DEVICE).long()).eval()
    held = step.cache  # type: ignore[attr-defined]
    assert isinstance(held, SignatureCache) and len(held) == 1
    assert next(iter(held.values())).retained > 0
