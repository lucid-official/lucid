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
* The question asked of each symbol is the audit's own: the first invocation that works
  (``_specs.invocations``), its primary operand rebuilt as bool (``_probe.dtype_args``),
  exactly as ``dtype_contract.measure`` rebuilt it for the reference.
* A deviation is one of ``over-accept`` (the reference refuses, Lucid answers), ``class``
  (both refuse, one with ``NotImplementedError`` -- a kernel with no bool instantiation --
  and the other with a plain ``RuntimeError`` -- a dtype guard), ``non-runtime`` (Lucid's
  refusal is not a ``RuntimeError`` at all: a plain ``TypeError``), ``over-refuse`` (the
  reference answers, Lucid refuses).

``bool_refusal_baseline.json`` records today's deviations and only shrinks: a deviation
that is not in it fails, and so does an entry that no longer deviates.  Each entry names
the card that fixes it (``owner``): the engine schema rows (``LCD-1020-E1``), the Python
composites (``LCD-1020-A1``), or why it is not a defect (``probe-artifact``,
``call-dependent``) or not in this batch (``out-of-batch``).
"""

import functools
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context, _produces_float64
from lucid.test.audit.tools import dtype_contract

# Probing calls every symbol the audit surveys, deprecated aliases included; a warning
# from one is not what is being asked.
pytestmark = pytest.mark.filterwarnings("ignore")

Kind = Literal["over-accept", "class", "non-runtime", "over-refuse"]

_AUDIT: Final = pathlib.Path(_surface.__file__).parent
_CONTRACT: Final = _AUDIT / "dtype_contract.json"
_BASELINE: Final = _AUDIT / "bool_refusal_baseline.json"

_KINDS: Final = ("over-accept", "class", "non-runtime", "over-refuse")
_OWNERS: Final = ("LCD-1020-E1", "LCD-1020-A1", "out-of-batch", "probe-artifact", "call-dependent")
_DEVICES: Final = ("cpu", "metal")

#: The two classes the reference refuses a bool with, and the rest are named as they come.
_KERNEL: Final = "NotImplementedError"
_GUARD: Final = "RuntimeError"


@dataclass(frozen=True)
class Deviation:
    """Where Lucid's answer to a bool operand departs from the reference's."""

    kind: Kind
    detail: str


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


@dataclass(frozen=True)
class _Probe:
    """One symbol as the audit invokes it."""

    fn: Callable[..., object]
    call: _specs.Call


def _accepts_bool(accepted: dict[str, list[str]]) -> set[str]:
    return {name for name, dtypes in accepted.items() if "bool" in dtypes}


def _refusals() -> dict[str, str]:
    """``symbol -> the class the reference raised for a bool operand``."""
    refusals: dict[str, str] = json.loads(_CONTRACT.read_text())["bool_refusals"]
    return refusals


def _symbols() -> dict[str, list[str]]:
    """``symbol -> the dtypes the reference accepts``."""
    symbols: dict[str, list[str]] = json.loads(_CONTRACT.read_text())["symbols"]
    return symbols


def _baseline(device: str | None = None) -> dict[str, Recorded]:
    """The recorded deviations, the ones that hold on ``device`` when it is given."""
    raw: dict[str, dict[str, str | list[str]]] = json.loads(_BASELINE.read_text())["deviations"]
    entries = {
        name: Recorded(
            kind=str(entry["kind"]),
            owner=str(entry["owner"]),
            reason=str(entry["reason"]),
            devices=tuple(entry.get("devices", _DEVICES)),
        )
        for name, entry in raw.items()
    }
    if device is None:
        return entries
    return {name: entry for name, entry in entries.items() if device in entry.devices}


@functools.lru_cache(maxsize=1)
def _probes() -> dict[str, _Probe]:
    """The audit's working invocation of every symbol the contract measured.

    Chosen the way ``dtype_contract.measure`` chooses it -- the first candidate that runs
    on Lucid and answers -- so the question put to Lucid is the one put to the reference.
    """
    wanted = set(_symbols())
    ctx = Context(metal=False)
    found: dict[str, _Probe] = {}
    for symbol in _surface.enumerate_surface():
        if symbol.qualname not in wanted:
            continue
        fn = _surface.resolve(symbol)
        if fn is None:
            continue
        with _probe.preserved_globals():
            for domain in ctx.domains:
                for candidate in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                    try:
                        out = fn(*candidate.args, **candidate.kwargs)
                    except Exception:  # noqa: BLE001 - surveying: this candidate is not the call
                        continue
                    if _probe.to_numpy(out) is not None:
                        found[symbol.qualname] = _Probe(fn, candidate)
                        break
                if symbol.qualname in found:
                    break
    return found


def _moved(value: object, device: str) -> object:
    """A keyword argument on ``device``: tensors move, everything else is itself."""
    if isinstance(value, lucid.Tensor):
        return value.to(device)
    if isinstance(value, list | tuple):
        return type(value)(_moved(item, device) for item in value)
    return value


def refusal_of(probe: _Probe, device: str) -> Exception | None:
    """What Lucid does with the symbol's primary operand rebuilt as bool.

    Returns the error it raised, or ``None`` when it answered.  A refusal of the *values*
    (a zero divisor) is an answer: the op dispatched on the dtype and took it.
    """
    bool_dtype = _probe.dtype_of("bool")

    def build(array: np.ndarray, follow: bool) -> lucid.Tensor:
        if not follow:
            return _probe.tensor_of(array, device=device)
        return _probe.tensor_of(array.astype(np.bool_), dtype=bool_dtype, device=device)

    with _probe.preserved_globals():
        try:
            args = _probe.dtype_args(probe.call, "bool", build)
            kwargs = {key: _moved(value, device) for key, value in probe.call.kwargs.items()}
            probe.fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a refusal is the measurement
            return None if _probe.refuses_a_value(exc) else exc
    return None


def _names(error: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(error).__mro__}


def _same_class(error: BaseException, reference: str) -> bool:
    """Whether ``error`` is the reference's class.

    ``NotImplementedError`` is a ``RuntimeError`` too, so the plain ``RuntimeError`` of a
    dtype guard is the one that is *not* also a ``NotImplementedError``: the two are
    different answers to whoever catches them.
    """
    names = _names(error)
    if reference == _GUARD:
        return _GUARD in names and _KERNEL not in names
    return reference in names


def deviation_of(probe: _Probe, reference: str | None, device: str = "cpu") -> Deviation | None:
    """How Lucid departs from the reference on ``probe``, or ``None``.

    Parameters
    ----------
    probe : _Probe
        The symbol and its working invocation.
    reference : str or None
        The class the reference raised for a bool operand, or ``None`` when it answers.
    device : str
        ``"cpu"`` or ``"metal"``.
    """
    refused = refusal_of(probe, device)
    if refused is None:
        if reference is None:
            return None
        return Deviation("over-accept", f"answers; the reference raises {reference}")
    shown = f"{type(refused).__name__}: {str(refused)[:70]}"
    if reference is None:
        return Deviation("over-refuse", f"refuses ({shown}); the reference answers")
    if _same_class(refused, reference):
        return None
    if reference in (_KERNEL, _GUARD) and not isinstance(refused, RuntimeError):
        return Deviation("non-runtime", f"raises {shown}; the reference raises {reference}")
    return Deviation("class", f"raises {shown}; the reference raises {reference}")


@functools.lru_cache(maxsize=2)
def _deviations(device: str) -> dict[str, Deviation]:
    """Every symbol of the contract Lucid answers differently from the reference."""
    refusals = _refusals()
    out: dict[str, Deviation] = {}
    for name, probe in _probes().items():
        # An op whose answer is float64 by definition has no Metal dtype to answer in; the
        # audit's dtype axis calls it not applicable there, and so does this.
        if device == "metal" and _produces_float64(probe.fn, probe.call):
            continue
        found = deviation_of(probe, refusals.get(name), device)
        if found is not None:
            out[name] = found
    return out


def _describe(found: dict[str, Deviation], names: set[str]) -> str:
    return "\n".join(
        f"  {name}: [{found[name].kind}] {found[name].detail}" for name in sorted(names)
    )


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
    # The sweep skips a symbol it cannot call, so one the audit can no longer invoke
    # would drop out of the guard unnoticed.
    missing = sorted(set(_symbols()) - set(_probes()))
    assert not missing, f"no invocation of the audit runs these any more: {missing}"


def test_baseline_entries_are_well_formed() -> None:
    raw = json.loads(_BASELINE.read_text())["deviations"]
    contract = set(_symbols())
    for name, entry in _baseline().items():
        assert name in contract, f"{name} is not a symbol of the contract"
        assert set(raw[name]) <= {"kind", "owner", "reason", "devices"}, (name, sorted(raw[name]))
        assert entry.kind in _KINDS, (name, entry.kind)
        assert entry.owner in _OWNERS, (name, entry.owner)
        assert entry.reason.strip(), f"{name} has no reason"
        assert entry.devices and set(entry.devices) <= set(_DEVICES), (name, entry.devices)
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
        + _describe(found, new)
    )


def test_over_refusal(device: str) -> None:
    # A bool the reference answers, Lucid must answer: refusing more than the reference is
    # the same fault the other way round, and breaks the program the reference runs.
    found = _deviations(device)
    over = {name for name, found_ in found.items() if found_.kind == "over-refuse"}
    unrecorded = over - set(_baseline(device))
    assert not unrecorded, "Lucid refuses a bool the reference answers:\n" + _describe(
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


def _stub(behaviour: Callable[[lucid.Tensor], object]) -> _Probe:
    """A symbol made of ``behaviour``, called the way the audit calls an op."""
    return _Probe(behaviour, _specs.Call([lucid.tensor([[1.0, 2.0], [3.0, 4.0]])], {}, 0, "stub"))


def _raise(error: Exception) -> Callable[[lucid.Tensor], object]:
    def behaviour(x: lucid.Tensor) -> object:
        raise error

    return behaviour


def test_the_checker_reports_a_callable_that_accepts_bool(device: str) -> None:
    # The negative control: were the sweep blind, it would pass for the wrong reason.
    accepting = _stub(lambda x: x + 1)
    for reference in (_KERNEL, _GUARD):
        found = deviation_of(accepting, reference, device)
        assert found is not None and found.kind == "over-accept"
    assert deviation_of(accepting, None, device) is None


@pytest.mark.parametrize(
    ("raised", "reference", "kind"),
    [
        (_C_engine.NotImplementedError("x"), _KERNEL, None),
        (_C_engine.DtypeMismatch("x"), _GUARD, None),
        (_C_engine.NotImplementedError("x"), _GUARD, "class"),
        (_C_engine.DtypeMismatch("x"), _KERNEL, "class"),
        (TypeError("x"), _KERNEL, "non-runtime"),
        (TypeError("x"), _GUARD, "non-runtime"),
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
    found = deviation_of(_stub(_raise(raised)), reference, device)
    assert (found.kind if found else None) == kind


def test_the_checker_reports_a_refusal_the_reference_does_not_make(device: str) -> None:
    found = deviation_of(_stub(_raise(RuntimeError("no"))), None, device)
    assert found is not None and found.kind == "over-refuse"


# ── the reference ────────────────────────────────────────────────────────────


@pytest.mark.parity
def test_contract_bool_column_is_current(ref: ModuleType) -> None:
    # The sweep trusts ``dtype_contract.json``; this is what keeps it honest.  The
    # reference is asked again, and the bool column -- who accepts, and who raises what --
    # must be the one checked in.
    accepted, _, refused = dtype_contract.measure(ref)
    recorded = _symbols()
    want = _accepts_bool(recorded)
    got = _accepts_bool(accepted)
    assert got == want, (
        "the bool column of dtype_contract.json is stale; regenerate it with "
        "`python -m lucid.test.audit.tools.dtype_contract`\n"
        f"  now accept bool: {sorted(got - want)}\n"
        f"  no longer accept: {sorted(want - got)}"
    )
    assert set(accepted) == set(recorded), (
        f"symbols measured now but not recorded: {sorted(set(accepted) - set(recorded))}; "
        f"recorded but not measured: {sorted(set(recorded) - set(accepted))}"
    )
    assert refused == _refusals(), {
        name: (_refusals().get(name), refused.get(name))
        for name in set(refused) | set(_refusals())
        if refused.get(name) != _refusals().get(name)
    }
