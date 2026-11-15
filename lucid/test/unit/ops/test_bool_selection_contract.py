"""One bool contract for the selection family (LCD-1011).

``topk(bool, largest=True)`` raised while ``topk(bool, largest=False)``
answered: the engine's ``topk`` refuses bool, but the Python entry builds the
smallest-k from ``argsort`` / ``gather``, which take bool, so one op said two
things depending on a keyword.  The table below is the contract for every
entry of the family: ``"refuse"`` raises on every device, ``"accept"``
returns a result on every device, whichever keyword, dim or spelling
(free function or ``Tensor`` method) reached it.

The table is held to the reference twice over: the parity test replays each
row there, so a row that drifts from it fails.  The reference's own Metal
backend diverges from its CPU one for ``median`` (it accepts there), so the
CPU answer is the one the table follows.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Literal

import pytest

import lucid

Verdict = Literal["refuse", "accept"]

# ``fw`` is the framework module under test, ``t`` a bool vector and ``m`` a
# bool matrix; both frameworks spell every one of these calls the same way.
_Call = Callable[[ModuleType, object, object], object]


def _row(verdict: Verdict, call: _Call) -> tuple[Verdict, _Call]:
    return verdict, call


_TABLE: dict[str, tuple[Verdict, _Call]] = {
    # topk: both ends, every spelling, 0-d and the unsorted flag.
    "topk": _row("refuse", lambda fw, t, m: fw.topk(t, 2)),
    "topk-largest": _row("refuse", lambda fw, t, m: fw.topk(t, 2, largest=True)),
    "topk-smallest": _row("refuse", lambda fw, t, m: fw.topk(t, 2, largest=False)),
    "topk-unsorted": _row("refuse", lambda fw, t, m: fw.topk(t, 2, largest=False, sorted=False)),
    "topk-dim0": _row("refuse", lambda fw, t, m: fw.topk(m, 1, dim=0, largest=False)),
    "topk-method": _row("refuse", lambda fw, t, m: t.topk(2)),
    "topk-method-smallest": _row("refuse", lambda fw, t, m: t.topk(2, largest=False)),
    "topk-0d": _row("refuse", lambda fw, t, m: fw.topk(t[0], 1)),
    "topk-0d-smallest": _row("refuse", lambda fw, t, m: fw.topk(t[0], 1, largest=False)),
    "topk-k0": _row("refuse", lambda fw, t, m: fw.topk(t, 0, largest=False)),
    # The rest of the order statistics.
    "kthvalue": _row("refuse", lambda fw, t, m: fw.kthvalue(t, 2)),
    "kthvalue-dim": _row("refuse", lambda fw, t, m: fw.kthvalue(m, 2, dim=1, keepdim=True)),
    "kthvalue-method": _row("refuse", lambda fw, t, m: t.kthvalue(2)),
    "median": _row("refuse", lambda fw, t, m: fw.median(t)),
    "median-dim": _row("refuse", lambda fw, t, m: fw.median(m, dim=1)),
    "median-method": _row("refuse", lambda fw, t, m: t.median()),
    "nanmedian": _row("refuse", lambda fw, t, m: fw.nanmedian(t)),
    "nanmedian-dim": _row("refuse", lambda fw, t, m: fw.nanmedian(m, dim=1)),
    "argmax": _row("refuse", lambda fw, t, m: fw.argmax(t)),
    "argmax-dim": _row("refuse", lambda fw, t, m: fw.argmax(m, dim=1)),
    "argmin": _row("refuse", lambda fw, t, m: fw.argmin(t)),
    "argmin-dim": _row("refuse", lambda fw, t, m: fw.argmin(m, dim=1)),
    # Sorting only moves values, so bool is welcome at both ends.
    "sort": _row("accept", lambda fw, t, m: fw.sort(t)),
    "sort-descending": _row("accept", lambda fw, t, m: fw.sort(t, descending=True)),
    "sort-dim0": _row("accept", lambda fw, t, m: fw.sort(m, dim=0, descending=True)),
    "sort-method": _row("accept", lambda fw, t, m: t.sort(descending=True)),
    "argsort": _row("accept", lambda fw, t, m: fw.argsort(t)),
    "argsort-descending": _row("accept", lambda fw, t, m: fw.argsort(t, descending=True)),
    "argsort-method": _row("accept", lambda fw, t, m: t.argsort(descending=True)),
    "msort": _row("accept", lambda fw, t, m: fw.msort(m)),
    "mode": _row("accept", lambda fw, t, m: fw.mode(m, dim=1)),
    "max": _row("accept", lambda fw, t, m: fw.max(t)),
    "max-dim": _row("accept", lambda fw, t, m: fw.max(m, dim=1)),
    "max-method-dim": _row("accept", lambda fw, t, m: m.max(dim=0, keepdim=True)),
    "min": _row("accept", lambda fw, t, m: fw.min(t)),
    "min-dim": _row("accept", lambda fw, t, m: fw.min(m, dim=1)),
    "amax-dim": _row("accept", lambda fw, t, m: fw.amax(m, dim=1)),
    "amin-dim": _row("accept", lambda fw, t, m: fw.amin(m, dim=1)),
    "cummax": _row("accept", lambda fw, t, m: fw.cummax(t, 0)),
    "cummin": _row("accept", lambda fw, t, m: fw.cummin(t, 0)),
}

_VECTOR = [True, False, True, True]
_MATRIX = [[True, False, True], [False, False, True]]


def _as_lists(result: object) -> object:
    """``result`` as plain Python values, whichever framework made it."""
    if isinstance(result, tuple):
        return tuple(_as_lists(part) for part in result)
    tolist = getattr(result, "tolist", None)
    return tolist() if callable(tolist) else result


@pytest.mark.parametrize("name", _TABLE)
def test_bool_verdict_is_the_same_on_every_device(device: str, name: str) -> None:
    verdict, call = _TABLE[name]
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    if verdict == "refuse":
        # The class is both ``NotImplementedError`` and, as the reference's
        # refusals are, a ``RuntimeError``.
        with pytest.raises(NotImplementedError) as caught:
            call(lucid, t, m)
        assert isinstance(caught.value, RuntimeError)
        return
    assert call(lucid, t, m) is not None


@pytest.mark.parametrize(
    "call",
    [
        lambda t: lucid.topk(t, 2),
        lambda t: lucid.topk(t, 2, largest=True),
        lambda t: lucid.topk(t, 2, largest=False),
        lambda t: lucid.topk(t, 2, largest=False, sorted=False),
        lambda t: t.topk(2),
        lambda t: t.topk(2, largest=False),
    ],
    ids=["free", "largest", "smallest", "unsorted", "method", "method-smallest"],
)
def test_every_topk_entry_refuses_bool_in_the_same_words(
    device: str, call: Callable[[lucid.Tensor], object]
) -> None:
    with pytest.raises(NotImplementedError) as caught:
        call(lucid.tensor(_VECTOR, device=device))
    assert str(caught.value) == "topk does not accept bool — cast to an integer dtype"


@pytest.mark.parity
@pytest.mark.parametrize("name", _TABLE)
def test_table_row_is_what_the_reference_does_on_cpu(ref: ModuleType, name: str) -> None:
    verdict, call = _TABLE[name]
    t = ref.tensor(_VECTOR)
    m = ref.tensor(_MATRIX)
    if verdict == "refuse":
        with pytest.raises((RuntimeError, NotImplementedError)):
            call(ref, t, m)
        return
    call(ref, t, m)


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, (verdict, _) in _TABLE.items() if verdict == "accept"])
def test_accepted_rows_return_the_reference_values(ref: ModuleType, device: str, name: str) -> None:
    _, call = _TABLE[name]
    got = call(lucid, lucid.tensor(_VECTOR, device=device), lucid.tensor(_MATRIX, device=device))
    want = call(ref, ref.tensor(_VECTOR), ref.tensor(_MATRIX))
    assert _as_lists(got) == _as_lists(want)


@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
@pytest.mark.parametrize("dtype", ["int8", "int64", "float32"])
def test_non_bool_topk_still_answers_at_both_ends(device: str, dtype: str, largest: bool) -> None:
    x = lucid.tensor([3, 1, 2, 5], dtype=getattr(lucid, dtype), device=device)
    values, indices = lucid.topk(x, 2, largest=largest)
    assert values.tolist() == ([5, 3] if largest else [1, 2])
    assert indices.tolist() == ([3, 0] if largest else [1, 2])


@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
def test_a_compiled_bool_topk_refuses_as_eager_does(
    device: str, largest: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``largest=False`` used to trace into argsort + gather, which the
    # rank-canonical rewrite at its lowest threshold then lowered for a bool
    # feed.  It is refused while the function is traced, before any graph.
    monkeypatch.setenv("LUCID_COMPILE_RANK_CANON_MIN_LEAD", "0")
    x = (lucid.randint(0, 2, (4, 6)) == 1).to(device)
    compiled = lucid.compile(lambda t: lucid.topk(t, 2, dim=-1, largest=largest)[0])
    with pytest.raises(NotImplementedError, match="topk does not accept bool"):
        compiled(x)
