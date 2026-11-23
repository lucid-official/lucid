"""Lucid refuses an integer tensor exactly where the reference CPU does (LCD-1026).

The integer refusal policy used to be a list somebody kept: each batch hunted the ops the
reference turns an integer away from, one by one, and an op nobody thought of stayed open.
LCD-1026-E1 moved the primitives' share of it into the engine schema and LCD-1026-A1 the
Python share into the entry checks; nothing proved *either* of them complete.  This is the
class-wide guard, the integer twin of ``test_bool_refusal_sweep.py``: it derives the
candidates instead of listing them.

* The candidates are every symbol of ``dtype_contract.json``.  Its ``int_refusals`` section
  names, for each symbol the reference refuses an integer, the class it raised, and
  ``int_widths`` the widths when only some are refused (``avg_pool2d`` takes ``int64``) -- so
  the sweep needs no reference to run (about two seconds, both devices).
  ``test_contract_integer_columns_are_current`` re-measures the columns against the reference
  (``-m parity``) so the checked-in table cannot drift.  There is no ``uint8`` column: Lucid
  has no such dtype.
* The question asked of each (symbol, width) is the audit's own, shared with the bool sweep
  (``_dtype_sweep.py``): the first invocation that works, its primary operand rebuilt at the
  width, and the answer compared with the reference's -- ``over-accept``, ``class``,
  ``non-runtime`` or ``over-refuse``.

The baseline only shrinks, and it is split by who shrinks it, so the engine and the Python
owners can each delete their own rows in parallel without editing one file:

``integer_refusal_baseline.engine.json``
    a primitive with a schema: the fix is a schema row (``OpSchema.refusal_for``).
``integer_refusal_baseline.python.json``
    a composite, an adapter or a loss: the fix is the Python entry check.
``integer_refusal_baseline.deferred.json``
    not a refusal to add: the access side (Lucid refuses what the reference answers -- a
    feature to build, not a check to delete) and the probe's own artifacts.

A (symbol, width) deviation that is in none of them fails, and so does a recorded one that no
longer deviates or deviates another way; a symbol in two files fails too.
"""

import json
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Final, NamedTuple

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._helpers.dtype_refusal import engine_refusal
from lucid.test.audit.tools.dtype_contract import INTEGERS
from lucid.test.unit.ops import _dtype_sweep as sweep
from lucid.test.unit.ops._dtype_sweep import Deviation, Probe

# Probing calls every symbol the audit surveys, deprecated aliases included; a warning
# from one is not what is being asked.
pytestmark = pytest.mark.filterwarnings("ignore")

_OWNERS: Final = ("engine", "python", "deferred")
_ENTRY_KEYS: Final = frozenset({"reason", "op", *sweep.DEVICES})


class Cell(NamedTuple):
    """One question: a symbol asked at an integer width."""

    symbol: str
    dtype: str


@dataclass(frozen=True)
class Entry:
    """One symbol of a baseline file: the cells it deviates in, per device, and why.

    ``op`` names the engine primitive, for the engine file alone.
    """

    owner: str
    reason: str
    op: str | None
    cells: dict[str, dict[str, str]]


@dataclass(frozen=True)
class Baseline:
    """The recorded deviations: the three owner files, read side by side."""

    by_owner: dict[str, dict[str, Entry]]

    @classmethod
    def load(cls) -> Baseline:
        return cls({owner: _read(owner) for owner in _OWNERS})

    def entries(self) -> dict[str, Entry]:
        return {name: entry for owner in _OWNERS for name, entry in self.by_owner[owner].items()}

    def shared(self) -> set[str]:
        """The symbols more than one file claims."""
        seen: set[str] = set()
        shared: set[str] = set()
        for owner in _OWNERS:
            shared |= seen & set(self.by_owner[owner])
            seen |= set(self.by_owner[owner])
        return shared

    def cells(self, device: str) -> dict[Cell, str]:
        """``cell -> the kind of deviation`` recorded for ``device``."""
        return {
            Cell(name, dtype): kind
            for name, entry in self.entries().items()
            for dtype, kind in entry.cells.get(device, {}).items()
        }


@dataclass(frozen=True)
class Verdict:
    """How a measurement stands against the baseline, cell by cell."""

    new: dict[Cell, Deviation]
    stale: dict[Cell, str]
    rekinded: dict[Cell, tuple[str, str]]

    @property
    def clean(self) -> bool:
        return not (self.new or self.stale or self.rekinded)


def compare(found: dict[Cell, Deviation], recorded: dict[Cell, str]) -> Verdict:
    """What ``found`` adds to, and no longer matches in, ``recorded``."""
    return Verdict(
        new={cell: dev for cell, dev in found.items() if cell not in recorded},
        stale={cell: kind for cell, kind in recorded.items() if cell not in found},
        rekinded={
            cell: (recorded[cell], dev.kind)
            for cell, dev in found.items()
            if cell in recorded and recorded[cell] != dev.kind
        },
    )


def _path(owner: str) -> pathlib.Path:
    return sweep.AUDIT_DIR / f"integer_refusal_baseline.{owner}.json"


def _raw(owner: str) -> dict[str, dict[str, str | dict[str, str]]]:
    """One baseline file's ``deviations``, as written."""
    raw: dict[str, dict[str, str | dict[str, str]]] = json.loads(_path(owner).read_text())[
        "deviations"
    ]
    return raw


def _read(owner: str) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    for name, entry in _raw(owner).items():
        op = entry.get("op")
        entries[name] = Entry(
            owner=owner,
            reason=str(entry["reason"]),
            op=op if isinstance(op, str) else None,
            cells={d: cells for d in sweep.DEVICES if isinstance(cells := entry.get(d), dict)},
        )
    return entries


def _found(device: str) -> dict[Cell, Deviation]:
    """Every (symbol, width) of the contract Lucid answers differently from the reference."""
    return {
        Cell(name, dtype): deviation
        for name, cells in sweep.deviations(device, INTEGERS).items()
        for dtype, deviation in cells.items()
    }


def _show(cells: Iterable[Cell], found: dict[Cell, Deviation]) -> str:
    """One line per symbol: the widths, and how the first of them deviates."""
    widths: dict[str, list[Cell]] = defaultdict(list)
    for cell in sorted(cells):
        widths[cell.symbol].append(cell)
    return "\n".join(
        f"  {name}: {', '.join(c.dtype for c in group)} "
        f"[{found[group[0]].kind}] {found[group[0]].detail}"
        for name, group in widths.items()
    )


# ── the contract is well-formed ──────────────────────────────────────────────


def test_integer_columns_agree_with_the_accepted_lists() -> None:
    contract = sweep.Contract.load()
    for name, accepted in contract.symbols.items():
        refused = [width for width in INTEGERS if width not in accepted]
        assert contract.refused_integers(name) == refused, (name, accepted)
    assert set(contract.int_refusals) <= set(contract.symbols)
    # int_widths lists the exceptions only: a symbol that refuses every width is not in it.
    assert all(0 < len(widths) < len(INTEGERS) for widths in contract.int_widths.values())
    assert set(contract.int_widths) <= set(contract.int_refusals)


def test_contract_names_only_dtypes_lucid_has() -> None:
    contract = sweep.Contract.load()
    named = {dtype for accepted in contract.symbols.values() for dtype in accepted}
    named |= {width for widths in contract.int_widths.values() for width in widths}
    assert all(hasattr(lucid, dtype) for dtype in named), sorted(
        dtype for dtype in named if not hasattr(lucid, dtype)
    )
    assert "uint8" not in named | set(INTEGERS)


def test_every_contract_symbol_has_a_working_invocation() -> None:
    # The sweep skips a symbol it cannot call, so one the audit can no longer invoke
    # would drop out of the guard unnoticed.
    missing = sweep.unprobed()
    assert not missing, f"no invocation of the audit runs these any more: {missing}"


# ── the baseline is well-formed ──────────────────────────────────────────────


def test_there_is_one_baseline_file_per_owner() -> None:
    present = {
        path.name.split(".")[-2] for path in sweep.AUDIT_DIR.glob("integer_refusal_baseline.*.json")
    }
    assert present == set(_OWNERS), present


def test_no_symbol_is_in_two_baselines() -> None:
    # One owner per symbol: two files claiming it is two owners shrinking the same row.
    shared = Baseline.load().shared()
    assert not shared, f"in more than one owner file: {sorted(shared)}"


def test_baseline_entries_are_well_formed() -> None:
    contract = sweep.Contract.load()
    for owner in _OWNERS:
        assert json.loads(_path(owner).read_text())["owner"] == owner, owner
        for name, entry in Baseline.load().by_owner[owner].items():
            raw = _raw(owner)[name]
            assert set(raw) <= _ENTRY_KEYS, (name, sorted(raw))
            assert name in contract.symbols, f"{name} is not a symbol of the contract"
            assert entry.reason.strip(), f"{name} has no reason"
            assert entry.cells, f"{name} records no deviating cell"
            assert (entry.op is not None) == (owner == "engine"), (name, entry.op)
            for device, cells in entry.cells.items():
                assert cells, f"{name} lists an empty {device}"
                for dtype, kind in cells.items():
                    assert dtype in INTEGERS, (name, device, dtype)
                    assert kind in sweep.KINDS, (name, device, dtype, kind)
                    # A refusal-side cell names a width the reference refuses; an over-refusal
                    # one a width it answers.
                    answers = contract.reference_class(name, dtype) is None
                    assert (kind == "over-refuse") == answers, (name, device, dtype, kind)


def test_engine_rows_are_ones_the_schema_has_not_yet_declared() -> None:
    # An engine row is the schema's to fix.  Where the schema already declares the
    # refusal the answer comes from a layer above it, and the row belongs in the python file.
    for name, entry in Baseline.load().by_owner["engine"].items():
        assert entry.op is not None
        for cells in entry.cells.values():
            for dtype in cells:
                assert engine_refusal(entry.op, getattr(lucid, dtype)) is None, (name, dtype)


# ── the ratchet ──────────────────────────────────────────────────────────────


def test_no_new_deviation(device: str) -> None:
    # The reference refuses an integer and Lucid computes it (or refuses with another class),
    # or the reference answers and Lucid refuses: either way the baseline does not know it.
    found = _found(device)
    verdict = compare(found, Baseline.load().cells(device))
    if verdict.new:
        pytest.fail(
            "Lucid departs from the reference on an integer operand where the baseline records "
            "no deviation.  Fix the op (the engine schema row / the Python entry check) -- do "
            "not baseline it:\n" + _show(verdict.new, found)
        )


def test_baseline_has_no_stale_entry(device: str) -> None:
    # The other direction: a cell the sweep no longer reproduces is a fix that landed, and
    # the baseline only shrinks.
    found = _found(device)
    verdict = compare(found, Baseline.load().cells(device))
    if verdict.stale or verdict.rekinded:
        pytest.fail(
            "baseline cells that no longer deviate (or deviate another way) -- delete or "
            "re-kind them in the owner's file:\n"
            + "\n".join(
                [f"  {c.symbol} {c.dtype}: now agrees with the reference" for c in verdict.stale]
                + [
                    f"  {c.symbol} {c.dtype}: baseline [{was}], now [{now}]"
                    for c, (was, now) in verdict.rekinded.items()
                ]
            )
        )


# ── the checker itself ───────────────────────────────────────────────────────

_ACCEPTS: Final = Deviation("over-accept", "answers")


def test_a_deviation_the_baseline_lacks_fails() -> None:
    cell = Cell("lucid.sin", "int8")
    verdict = compare({cell: _ACCEPTS}, {})
    assert set(verdict.new) == {cell} and not verdict.clean


def test_an_entry_that_no_longer_deviates_fails() -> None:
    cell = Cell("lucid.sin", "int8")
    verdict = compare({}, {cell: "over-accept"})
    assert set(verdict.stale) == {cell} and not verdict.clean


def test_an_entry_that_deviates_another_way_fails() -> None:
    cell = Cell("lucid.sin", "int8")
    verdict = compare({cell: _ACCEPTS}, {cell: "class"})
    assert verdict.rekinded == {cell: ("class", "over-accept")} and not verdict.clean


def test_a_baseline_that_matches_is_clean() -> None:
    cell = Cell("lucid.sin", "int8")
    assert compare({cell: _ACCEPTS}, {cell: "over-accept"}).clean


@pytest.mark.parametrize("width", INTEGERS)
def test_the_primary_operand_is_rebuilt_at_the_width(device: str, width: str) -> None:
    # The sweep's one move: were the operand left at float, every cell would pass for the
    # wrong reason.
    seen: list[lucid.dtype] = []

    def behaviour(x: lucid.Tensor) -> object:
        seen.append(x.dtype)
        return x + 1

    found = sweep.deviation_of(Probe.stub(behaviour), "NotImplementedError", device, width)
    assert seen == [getattr(lucid, width)]
    assert found is not None and found.kind == "over-accept"


@pytest.mark.parametrize("width", INTEGERS)
def test_the_checker_agrees_with_a_refusal_of_the_same_class(device: str, width: str) -> None:
    def refuse(x: lucid.Tensor) -> object:
        raise _C_engine.NotImplementedError("no kernel")

    assert sweep.deviation_of(Probe.stub(refuse), "NotImplementedError", device, width) is None
    found = sweep.deviation_of(Probe.stub(refuse), "RuntimeError", device, width)
    assert found is not None and found.kind == "class"


# ── the reference ────────────────────────────────────────────────────────────


@pytest.mark.parity
def test_contract_integer_columns_are_current(ref: ModuleType) -> None:
    # The sweep trusts ``dtype_contract.json``; this is what keeps it honest.  The reference
    # is asked again, and the integer columns -- who accepts which width, and who raises what
    # -- must be the ones checked in.
    now = sweep.measured(ref)
    contract = sweep.Contract.load()
    hint = "regenerate with `python -m lucid.test.audit.tools.dtype_contract`"

    def integers(accepted: dict[str, list[str]]) -> dict[str, list[str]]:
        return {name: [d for d in dtypes if d in INTEGERS] for name, dtypes in accepted.items()}

    assert set(now.accepted) == set(contract.symbols), (
        f"symbols measured now but not recorded: {sorted(set(now.accepted) - set(contract.symbols))}; "
        f"recorded but not measured: {sorted(set(contract.symbols) - set(now.accepted))}"
    )
    got, want = integers(now.accepted), integers(contract.symbols)
    assert got == want, (
        f"the accepted integer widths of dtype_contract.json are stale; {hint}\n"
        + "\n".join(
            f"  {name}: {want[name]} -> {got[name]}" for name in want if got[name] != want[name]
        )
    )
    assert now.int_refusals() == contract.int_refusals, hint
    assert now.int_widths() == contract.int_widths, hint
