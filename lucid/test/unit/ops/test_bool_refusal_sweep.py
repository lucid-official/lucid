"""Lucid refuses a bool tensor exactly where the reference CPU does (LCD-1020).

``BOOL_REFUSED`` (``lucid/_ops/_bool_refusal.py``) lists the ops LCD-1013 taught to
refuse a bool operand, and ``test_bool_refusal_table.py`` proves every row of it is
pinned.  Nothing proved the reverse -- that no op the reference refuses is *missing* --
so each batch found the stragglers by hand.  This is the class-wide guard: it derives
the candidates instead of listing them.

* The candidates are every symbol of ``dtype_contract.json``, the audit's measurement of
  what the reference accepts.  Its ``bool_refusals`` section names, for each symbol whose
  accepted list lacks ``bool``, the class the reference raised -- so the sweep needs no
  reference to run (about a second).  ``test_contract_bool_column_is_current`` re-measures
  the column against the reference (``-m parity``) so the checked-in table cannot drift.
* The question asked of each symbol, and the four kinds of deviation it can answer with, are
  the audit's own and are shared with the integer sweep: see ``_dtype_sweep.py``.

``bool_refusal_baseline.json`` records today's deviations and only shrinks: a deviation
that is not in it fails, and so does an entry that no longer deviates.  Each entry names
the card that fixes it (``owner``): the engine schema rows (``LCD-1020-E1``), the Python
composites (``LCD-1020-A1``), or why it is not a defect (``probe-artifact``,
``call-dependent``) or not in this batch (``out-of-batch``).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Final

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test.audit import _surface
from lucid.test.unit.ops import _dtype_sweep as sweep
from lucid.test.unit.ops._dtype_sweep import GUARD, KERNEL, Deviation, Kind, Probe

# Probing calls every symbol the audit surveys, deprecated aliases included; a warning
# from one is not what is being asked.
pytestmark = pytest.mark.filterwarnings("ignore")

_BASELINE: Final = sweep.AUDIT_DIR / "bool_refusal_baseline.json"

_OWNERS: Final = ("LCD-1020-E1", "LCD-1020-A1", "out-of-batch", "probe-artifact", "call-dependent")
_BOOL: Final = ("bool",)


@dataclass(frozen=True)
class Recorded:
    """One entry of the baseline: a deviation that is known and owned.

    ``devices`` is both unless the deviation is one device's alone (Metal raises the
    backend's own error for a bool matmul where the CPU already refuses as the reference
    does).
    """

    kind: str
    owner: str
    reason: str
    devices: tuple[str, ...]


def _accepts_bool(accepted: dict[str, list[str]]) -> set[str]:
    return {name for name, dtypes in accepted.items() if "bool" in dtypes}


def _refusals() -> dict[str, str]:
    """``symbol -> the class the reference raised for a bool operand``."""
    return sweep.Contract.load().bool_refusals


def _symbols() -> dict[str, list[str]]:
    """``symbol -> the dtypes the reference accepts``."""
    return sweep.Contract.load().symbols


def _baseline(device: str | None = None) -> dict[str, Recorded]:
    """The recorded deviations, the ones that hold on ``device`` when it is given."""
    raw: dict[str, dict[str, str | list[str]]] = json.loads(_BASELINE.read_text())["deviations"]
    entries = {
        name: Recorded(
            kind=str(entry["kind"]),
            owner=str(entry["owner"]),
            reason=str(entry["reason"]),
            devices=tuple(entry.get("devices", sweep.DEVICES)),
        )
        for name, entry in raw.items()
    }
    if device is None:
        return entries
    return {name: entry for name, entry in entries.items() if device in entry.devices}


def _deviations(device: str) -> dict[str, Deviation]:
    """Every symbol of the contract Lucid answers differently from the reference."""
    return {name: cells["bool"] for name, cells in sweep.deviations(device, _BOOL).items()}


# ── the contract and the baseline are well-formed ────────────────────────────


def test_every_symbol_without_bool_has_a_refusal_class() -> None:
    lacking = set(_symbols()) - _accepts_bool(_symbols())
    assert lacking == set(_refusals()), (
        "bool_refusals must name exactly the symbols whose accepted list lacks bool; "
        "regenerate with `python -m lucid.test.audit.tools.dtype_contract`"
    )


def test_every_contract_symbol_is_still_surveyed() -> None:
    surveyed = {symbol.qualname for symbol in _surface.enumerate_surface()}
    gone = sorted(set(_symbols()) - surveyed)
    assert not gone, f"in the contract but no longer surveyed (regenerate it): {gone}"


def test_every_contract_symbol_has_a_working_invocation() -> None:
    missing = sweep.unprobed()
    assert not missing, f"no invocation of the audit runs these any more: {missing}"


def test_baseline_entries_are_well_formed() -> None:
    raw = json.loads(_BASELINE.read_text())["deviations"]
    contract = set(_symbols())
    for name, entry in _baseline().items():
        assert name in contract, f"{name} is not a symbol of the contract"
        assert set(raw[name]) <= {"kind", "owner", "reason", "devices"}, (name, sorted(raw[name]))
        assert entry.kind in sweep.KINDS, (name, entry.kind)
        assert entry.owner in _OWNERS, (name, entry.owner)
        assert entry.reason.strip(), f"{name} has no reason"
        assert entry.devices and set(entry.devices) <= set(sweep.DEVICES), (name, entry.devices)
    # A refusal-side entry names the class the reference raises; one that is not a
    # refusal in the contract is an over-refusal, and the other way round.
    answers = _accepts_bool(_symbols())
    for name, entry in _baseline().items():
        assert (entry.kind == "over-refuse") == (name in answers), (name, entry.kind)


# ── the ratchet ──────────────────────────────────────────────────────────────


def test_no_new_deviation(device: str) -> None:
    # The reference refuses a bool and Lucid computes it, or refuses with another class.
    found = _deviations(device)
    recorded = _baseline(device)
    new = {name for name, found_ in found.items() if found_.kind != "over-refuse"} - set(recorded)
    assert not new, (
        "Lucid departs from the reference on a bool operand where the baseline records "
        "no deviation.  Fix the op (BOOL_REFUSED / the engine schema), do not baseline it:\n"
        + sweep.describe(found, new)
    )


def test_over_refusal(device: str) -> None:
    # A bool the reference answers, Lucid must answer: refusing more than the reference is
    # the same fault the other way round, and breaks the program the reference runs.
    found = _deviations(device)
    over = {name for name, found_ in found.items() if found_.kind == "over-refuse"}
    unrecorded = over - set(_baseline(device))
    assert not unrecorded, "Lucid refuses a bool the reference answers:\n" + sweep.describe(
        found, unrecorded
    )


def test_baseline_has_no_stale_entry(device: str) -> None:
    # The other direction: an entry the sweep no longer reproduces is a fix that landed,
    # and the baseline only shrinks.
    found = _deviations(device)
    recorded = _baseline(device)
    stale = sorted(
        name
        for name, entry in recorded.items()
        if name not in found or found[name].kind != entry.kind
    )
    assert not stale, (
        "baseline entries that no longer deviate (or deviate another way) -- delete or "
        "re-kind them:\n"
        + "\n".join(
            f"  {name}: baseline [{recorded[name].kind}], now "
            + (f"[{found[name].kind}]" if name in found else "agrees with the reference")
            for name in stale
        )
    )


# ── the checker itself ───────────────────────────────────────────────────────


def _raise(error: Exception) -> Callable[[lucid.Tensor], object]:
    def behaviour(x: lucid.Tensor) -> object:
        raise error

    return behaviour


def test_the_checker_reports_a_callable_that_accepts_bool(device: str) -> None:
    # The negative control: were the sweep blind, it would pass for the wrong reason.
    accepting = Probe.stub(lambda x: x + 1)
    for reference in (KERNEL, GUARD):
        found = sweep.deviation_of(accepting, reference, device, "bool")
        assert found is not None and found.kind == "over-accept"
    assert sweep.deviation_of(accepting, None, device, "bool") is None


@pytest.mark.parametrize(
    ("raised", "reference", "kind"),
    [
        (_C_engine.NotImplementedError("x"), KERNEL, None),
        (_C_engine.DtypeMismatch("x"), GUARD, None),
        (_C_engine.NotImplementedError("x"), GUARD, "class"),
        (_C_engine.DtypeMismatch("x"), KERNEL, "class"),
        (TypeError("x"), KERNEL, "non-runtime"),
        (TypeError("x"), GUARD, "non-runtime"),
        (ValueError("x"), "ValueError", None),
        (TypeError("x"), "ValueError", "class"),
    ],
    ids=[
        "kernel-is-kernel",
        "guard-is-guard",
        "kernel-for-guard",
        "guard-for-kernel",
        "plain-type-error-for-kernel",
        "plain-type-error-for-guard",
        "same-value-error",
        "other-for-value-error",
    ],
)
def test_the_checker_compares_the_class(
    device: str, raised: Exception, reference: str, kind: Kind | None
) -> None:
    found = sweep.deviation_of(Probe.stub(_raise(raised)), reference, device, "bool")
    assert (found.kind if found else None) == kind


def test_the_checker_reports_a_refusal_the_reference_does_not_make(device: str) -> None:
    found = sweep.deviation_of(Probe.stub(_raise(RuntimeError("no"))), None, device, "bool")
    assert found is not None and found.kind == "over-refuse"


# ── the reference ────────────────────────────────────────────────────────────


@pytest.mark.parity
def test_contract_bool_column_is_current(ref: ModuleType) -> None:
    # The sweep trusts ``dtype_contract.json``; this is what keeps it honest.  The
    # reference is asked again, and the bool column -- who accepts, and who raises what --
    # must be the one checked in.
    now = sweep.measured(ref)
    refused = now.bool_refusals()
    recorded = _symbols()
    want = _accepts_bool(recorded)
    got = _accepts_bool(now.accepted)
    assert got == want, (
        "the bool column of dtype_contract.json is stale; regenerate it with "
        "`python -m lucid.test.audit.tools.dtype_contract`\n"
        f"  now accept bool: {sorted(got - want)}\n"
        f"  no longer accept: {sorted(want - got)}"
    )
    assert set(now.accepted) == set(recorded), (
        f"symbols measured now but not recorded: {sorted(set(now.accepted) - set(recorded))}; "
        f"recorded but not measured: {sorted(set(recorded) - set(now.accepted))}"
    )
    assert refused == _refusals(), {
        name: (_refusals().get(name), refused.get(name))
        for name in set(refused) | set(_refusals())
        if refused.get(name) != _refusals().get(name)
    }
