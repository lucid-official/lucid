"""What the dtype refusal sweeps share: the contract, the probe, and the comparison.

The bool sweep (``test_bool_refusal_sweep.py``, LCD-1020) and the integer sweep
(``test_integer_refusal_sweep.py``, LCD-1026) ask one question of every symbol of
``dtype_contract.json`` -- *given this dtype, does Lucid answer or refuse the way the reference
does, and with the same class of error?* -- and differ only in the dtypes and in the baseline
that records today's answers.  This module is the question.

* :class:`Contract` reads the checked-in measurement of the reference
  (``lucid/test/audit/tools/dtype_contract.py`` writes it), so the sweep needs no reference to
  run.
* :func:`probes` finds, for each symbol of it, the audit's own working invocation, the call the
  generator replayed on the reference.
* :func:`deviations` puts that call to Lucid with the primary operand rebuilt at the dtype
  (``_probe.dtype_args``, exactly as the generator rebuilt it) and names how the answer departs
  from the reference's: ``over-accept`` (the reference refuses, Lucid answers), ``class`` (both
  refuse, one with ``NotImplementedError`` -- a kernel with no instantiation -- and the other
  with a plain ``RuntimeError`` -- a dtype guard), ``non-runtime`` (Lucid's refusal is not a
  ``RuntimeError`` at all), ``over-refuse`` (the reference answers, Lucid refuses).
* :func:`measured` asks the reference again (``-m parity``), so the contract cannot drift.
"""

import functools
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal

import numpy as np

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context, _produces_float64
from lucid.test.audit.tools import dtype_contract

Kind = Literal["over-accept", "class", "non-runtime", "over-refuse"]

AUDIT_DIR: Final = pathlib.Path(_surface.__file__).parent
CONTRACT_PATH: Final = AUDIT_DIR / "dtype_contract.json"

KINDS: Final = ("over-accept", "class", "non-runtime", "over-refuse")
DEVICES: Final = ("cpu", "metal")

#: The two classes the reference refuses a dtype with, and the rest are named as they come.
KERNEL: Final = "NotImplementedError"
GUARD: Final = "RuntimeError"


@dataclass(frozen=True)
class Deviation:
    """Where Lucid's answer to a dtype departs from the reference's."""

    kind: Kind
    detail: str


@dataclass(frozen=True)
class Probe:
    """One symbol as the audit invokes it."""

    fn: Callable[..., object]
    call: _specs.Call

    @classmethod
    def stub(cls, behaviour: Callable[[lucid.Tensor], object]) -> Probe:
        """A symbol made of ``behaviour``, called the way the audit calls an op."""
        operand = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        return cls(behaviour, _specs.Call([operand], {}, 0, "stub"))


@dataclass(frozen=True)
class Contract:
    """The reference's answers, as ``dtype_contract.json`` recorded them.

    ``symbols`` is every dtype the reference accepts, per symbol.  A refused dtype carries the
    class of the error it raised: ``bool_refusals`` for bool, ``int_refusals`` for the integer
    widths -- ``int_widths`` naming the refused ones when only some are (``avg_pool2d`` takes
    ``int64``) and every width otherwise.
    """

    symbols: dict[str, list[str]]
    bool_refusals: dict[str, str]
    int_refusals: dict[str, str]
    int_widths: dict[str, list[str]]

    @classmethod
    @functools.cache
    def load(cls) -> Contract:
        raw = json.loads(CONTRACT_PATH.read_text())
        return cls(raw["symbols"], raw["bool_refusals"], raw["int_refusals"], raw["int_widths"])

    def refused_integers(self, symbol: str) -> list[str]:
        """The integer widths the reference refuses ``symbol``."""
        if symbol not in self.int_refusals:
            return []
        return self.int_widths.get(symbol, list(dtype_contract.INTEGERS))

    def reference_class(self, symbol: str, dtype: str) -> str | None:
        """The class the reference raised for ``symbol`` at ``dtype``, ``None`` if it answers."""
        if dtype == "bool":
            return self.bool_refusals.get(symbol)
        if dtype in self.refused_integers(symbol):
            return self.int_refusals[symbol]
        return None


def _invocation(symbol: _surface.Symbol, fn: Callable[..., object], ctx: Context) -> Probe | None:
    """The first call of ``symbol`` that runs on Lucid and answers.

    Chosen the way ``dtype_contract.measure`` chooses it, so the question put to Lucid is the
    one put to the reference.
    """
    with _probe.preserved_globals():
        for domain in ctx.domains:
            for candidate in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*candidate.args, **candidate.kwargs)
                except Exception:  # noqa: BLE001 - surveying: this candidate is not the call
                    continue
                if _probe.to_numpy(out) is not None:
                    return Probe(fn, candidate)
    return None


@functools.cache
def probes() -> dict[str, Probe]:
    """The audit's working invocation of every symbol the contract measured."""
    wanted = set(Contract.load().symbols)
    ctx = Context(metal=False)
    found: dict[str, Probe] = {}
    for symbol in _surface.enumerate_surface():
        fn = _surface.resolve(symbol) if symbol.qualname in wanted else None
        if fn is None:
            continue
        probe = _invocation(symbol, fn, ctx)
        if probe is not None:
            found[symbol.qualname] = probe
    return found


def unprobed() -> list[str]:
    """The contract's symbols no invocation of the audit runs any more.

    The sweeps skip a symbol they cannot call, so one the audit can no longer invoke would drop
    out of the guard unnoticed.
    """
    return sorted(set(Contract.load().symbols) - set(probes()))


def _moved(value: object, device: str) -> object:
    """A keyword argument on ``device``: tensors move, everything else is itself."""
    if isinstance(value, lucid.Tensor):
        return value.to(device)
    if isinstance(value, list | tuple):
        return type(value)(_moved(item, device) for item in value)
    return value


def refusal_of(probe: Probe, dtype: str, device: str) -> Exception | None:
    """What Lucid does with the symbol's primary operand rebuilt at ``dtype``.

    Returns the error it raised, or ``None`` when it answered.  A refusal of the *values*
    (a zero divisor) is an answer: the op dispatched on the dtype and took it.
    """
    target = _probe.dtype_of(dtype)
    host = _probe.numpy_of(dtype)

    def build(array: np.ndarray, follow: bool) -> lucid.Tensor:
        if not follow:
            return _probe.tensor_of(array, device=device)
        return _probe.tensor_of(array.astype(host), dtype=target, device=device)

    with _probe.preserved_globals():
        try:
            args = _probe.dtype_args(probe.call, dtype, build)
            kwargs = {key: _moved(value, device) for key, value in probe.call.kwargs.items()}
            probe.fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a refusal is the measurement
            return None if _probe.refuses_a_value(exc) else exc
    return None


def _same_class(error: BaseException, reference: str) -> bool:
    """Whether ``error`` is the reference's class.

    ``NotImplementedError`` is a ``RuntimeError`` too, so the plain ``RuntimeError`` of a
    dtype guard is the one that is *not* also a ``NotImplementedError``: the two are
    different answers to whoever catches them.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    if reference == GUARD:
        return GUARD in names and KERNEL not in names
    return reference in names


def deviation_of(probe: Probe, reference: str | None, device: str, dtype: str) -> Deviation | None:
    """How Lucid departs from the reference on ``probe`` at ``dtype``, or ``None``.

    Parameters
    ----------
    probe : Probe
        The symbol and its working invocation.
    reference : str or None
        The class the reference raised for ``dtype``, or ``None`` when it answers.
    device : str
        ``"cpu"`` or ``"metal"``.
    dtype : str
        The dtype the primary operand is rebuilt at.
    """
    refused = refusal_of(probe, dtype, device)
    if refused is None:
        if reference is None:
            return None
        return Deviation("over-accept", f"answers; the reference raises {reference}")
    shown = f"{type(refused).__name__}: {str(refused)[:70]}"
    if reference is None:
        return Deviation("over-refuse", f"refuses ({shown}); the reference answers")
    if _same_class(refused, reference):
        return None
    if reference in (KERNEL, GUARD) and not isinstance(refused, RuntimeError):
        return Deviation("non-runtime", f"raises {shown}; the reference raises {reference}")
    return Deviation("class", f"raises {shown}; the reference raises {reference}")


@functools.cache
def deviations(device: str, dtypes: tuple[str, ...]) -> dict[str, dict[str, Deviation]]:
    """Every symbol of the contract Lucid answers differently from the reference, per dtype."""
    contract = Contract.load()
    out: dict[str, dict[str, Deviation]] = {}
    for name, probe in probes().items():
        # An op whose answer is float64 by definition has no Metal dtype to answer in; the
        # audit's dtype axis calls it not applicable there, and so does this.
        if device == "metal" and _produces_float64(probe.fn, probe.call):
            continue
        for dtype in dtypes:
            found = deviation_of(probe, contract.reference_class(name, dtype), device, dtype)
            if found is not None:
                out.setdefault(name, {})[dtype] = found
    return out


def describe(found: dict[str, Deviation], names: set[str]) -> str:
    """One line per symbol of ``names``: how it deviates."""
    return "\n".join(
        f"  {name}: [{found[name].kind}] {found[name].detail}" for name in sorted(names)
    )


@functools.cache
def measured(ref: ModuleType) -> dtype_contract.Measurement:
    """The reference asked again, once per process (every parity test of the sweeps reads it)."""
    return dtype_contract.measure(ref)
