"""Measure which dtypes the reference framework accepts, per symbol.

The dtype axis compares cpu against metal.  That answers *whether* the
two devices agree and never *which one is right* — and deciding that by
eye was going badly.  ``avg_pool2d`` looks obviously float-only and takes
int64; ``sigmoid`` looks obviously float-only and accepts every integer,
promoting on the way out; ``argsort`` takes bool and ``argmax`` does not.
None of that is guessable from the name, and a wrong guess widens a gate
that should have been narrowed — which is worse than the asymmetry it
was fixing, because a wrong answer is now returned on both devices.

So the third opinion is measured.  For each symbol the audit surveys,
this replays *the audit's own invocation* against the reference at each
dtype and records what survives.  Replaying the same call is the point:
if the two built their arguments differently the comparison would be
between two different questions and the table would be evidence for
nothing.  Both sides go through :func:`_probe.dtype_args`.

Run::

    python -m lucid.test.audit.tools.dtype_contract

For every symbol whose accepted list lacks ``bool`` the table also records
the class of the error the reference raised (``bool_refusals``), and the same
for the integer widths (``int_refusals``; ``int_widths`` names the refused
ones when only some are).  The sweeps
(``lucid/test/unit/ops/test_{bool,integer}_refusal_sweep.py``) hold Lucid to
them: refuse where the reference refuses, with the same class.

The result is checked in as ``dtype_contract.json`` so the audit can read
it without the reference installed.  Symbols the reference has no
counterpart for — Lucid-only ops, and anything whose invocation cannot be
translated — are listed under ``unmapped`` rather than silently omitted,
so a missing verdict is visibly a gap and not a claim.
"""

import argparse
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

import lucid
from lucid.test._fixtures.ref_framework import require_ref
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context

#: Names that differ between the two frameworks.  Kept small and
#: explicit — a fuzzy match would map ``lucid.trace`` onto something
#: unrelated and the table would look measured while being invented.
ALIASES: dict[str, str] = {
    "hard_sigmoid": "hardsigmoid",
    "hard_swish": "hardswish",
    "hard_tanh": "hardtanh",
    "swish": "silu",
    "gelu_exact": "gelu",
    "arctan": "atan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctanh": "atanh",
    "arcsinh": "asinh",
    "arccosh": "acosh",
}

#: What a refusal column records for a symbol the reference neither answered
#: nor raised for at that dtype: the probe returned ``None``.
_NO_RESULT = "NoResult"

#: The integer widths the framework has, in the order the probe tries them.  No unsigned
#: width: ``uint8`` is not a Lucid dtype, so a column for it could only ever be a claim about
#: something Lucid cannot be asked.
INTEGERS: Final[tuple[str, ...]] = tuple(name for name in _probe.DTYPES if name.startswith("int"))

#: Where each Lucid namespace lives in the reference.  ``None`` means the
#: symbol is a tensor method.
NAMESPACES: dict[str, str | None] = {
    "lucid": "",
    "Tensor": None,
    "F": "nn.functional",
    "lucid.linalg": "linalg",
    "lucid.special": "special",
    "lucid.fft": "fft",
    "lucid.nn.init": "nn.init",
}


def _reference_callable(ref: Any, qualname: str) -> Any:
    """The reference counterpart of a Lucid symbol, or ``None``."""
    head, _, short = qualname.rpartition(".")
    where = NAMESPACES.get(head or "lucid", "missing")
    if where == "missing":
        return None
    name = ALIASES.get(short, short)
    if where is None:
        return name  # a method: resolved per-tensor, once one exists
    target = ref
    for part in filter(None, where.split(".")):
        target = getattr(target, part, None)
        if target is None:
            return None
    found = getattr(target, name, None)
    if found is None and not name.endswith("_"):
        # ``nn.init`` spells every initialiser in place.
        found = getattr(target, name + "_", None)
    return found


def _as_reference(ref: Any, array: np.ndarray, follow: bool, name: str) -> Any:
    """One reference tensor, mirroring what the axis builds for Lucid."""
    if follow:
        array = np.ascontiguousarray(array.astype(_probe.numpy_of(name)))
    else:
        array = np.ascontiguousarray(array)
    return ref.from_numpy(array)


@dataclass(frozen=True)
class Measurement:
    """What the reference did with every surveyed symbol, and the columns read off it.

    ``refused`` holds, for every dtype a symbol was probed at and not accepted, the class of
    the error the reference raised.  The class is part of the contract: a kernel with no
    instantiation for the dtype raises ``NotImplementedError`` and a guard on the input
    dtype raises ``RuntimeError``, and a framework that refuses with the other one answers a
    different question for whoever catches it.
    """

    accepted: dict[str, list[str]]
    unmapped: list[str]
    refused: dict[str, dict[str, str]]

    def bool_refusals(self) -> dict[str, str]:
        """``symbol -> class`` for every measured symbol whose accepted list lacks ``bool``."""
        return {
            name: self.refused.get(name, {}).get("bool", _NO_RESULT)
            for name, works in sorted(self.accepted.items())
            if "bool" not in works
        }

    def int_widths(self) -> dict[str, list[str]]:
        """``symbol -> the integer widths refused``, for the symbols that refuse only some.

        A symbol that refuses every width is the common case and is not listed: ``avg_pool2d``
        takes ``int64`` alone, and that is the exception worth writing down.
        """
        refused = self._refused_widths()
        return {name: widths for name, widths in refused.items() if len(widths) < len(INTEGERS)}

    def int_refusals(self) -> dict[str, str]:
        """``symbol -> class`` for every measured symbol the reference refuses an integer.

        One class per symbol, because that is what was measured: no symbol's class depends on
        the width.  One that did would make the column say something false, so it stops the
        regeneration instead (``ValueError``) and the column grows a width key when it does.
        """
        out: dict[str, str] = {}
        for name, widths in self._refused_widths().items():
            classes = {self.refused.get(name, {}).get(width, _NO_RESULT) for width in widths}
            if len(classes) > 1:
                raise ValueError(
                    f"{name}: the reference refuses its integer widths with different classes "
                    f"({sorted(classes)}); int_refusals holds one class per symbol"
                )
            out[name] = classes.pop()
        return out

    def _refused_widths(self) -> dict[str, list[str]]:
        found = {
            name: [width for width in INTEGERS if width not in works]
            for name, works in sorted(self.accepted.items())
        }
        return {name: widths for name, widths in found.items() if widths}


def _working_call(symbol: _surface.Symbol, fn: Any, ctx: Context) -> Any:
    """The first invocation of ``symbol`` that runs on Lucid and answers, or ``None``.

    The audit's own choice, so the question put to the reference is the one the audit asks.
    """
    with _probe.preserved_globals():
        for domain in ctx.domains:
            for candidate in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*candidate.args, **candidate.kwargs)
                except Exception:  # noqa: BLE001 - surveying, not asserting
                    continue
                if _probe.to_numpy(out) is not None:
                    return candidate
    return None


def _ask_reference(ref: Any, call: Any, target: Any) -> tuple[list[str], dict[str, str]]:
    """Replay ``call`` on the reference at every dtype: those it answers, and why it did not."""
    works: list[str] = []
    classes: dict[str, str] = {}
    for name in _probe.DTYPES:
        try:
            args = _probe.dtype_args(call, name, lambda a, f, n=name: _as_reference(ref, a, f, n))
            fn_ref = target
            if isinstance(target, str):
                # A method.  The primary argument becomes the receiver
                # and drops out of the argument list.
                if call.primary >= len(args):
                    break
                fn_ref = getattr(args[call.primary], target, None)
                if fn_ref is None:
                    break
                args = args[: call.primary] + args[call.primary + 1 :]
            out = fn_ref(*args, **_reference_kwargs(ref, call.kwargs))
            if out is not None:
                works.append(name)
            else:
                classes[name] = _NO_RESULT
        except Exception as exc:  # noqa: BLE001 - a refusal is the measurement
            if _probe.refuses_a_value(exc):
                works.append(name)
            else:
                classes[name] = type(exc).__name__
    return works, {name: cls for name, cls in classes.items() if name not in works}


def measure(ref: Any, verbose: bool = False) -> Measurement:
    """Probe every surveyed symbol at every dtype."""
    ctx = Context(metal=False)
    accepted: dict[str, list[str]] = {}
    unmapped: list[str] = []
    refused: dict[str, dict[str, str]] = {}

    for symbol in _surface.enumerate_surface():
        fn = _surface.resolve(symbol)
        if fn is None:
            continue
        target = _reference_callable(ref, symbol.qualname)
        if target is None:
            unmapped.append(symbol.qualname)
            continue

        call = _working_call(symbol, fn, ctx)
        if call is None:
            continue
        try:
            call.base
        except TypeError:
            continue

        works, why = _ask_reference(ref, call, target)
        if works:
            accepted[symbol.qualname] = works
            refused[symbol.qualname] = why
        if verbose:
            print(f"{symbol.qualname:52s} {works}")

    return Measurement(accepted, sorted(unmapped), refused)


def _reference_kwargs(ref: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments with any Lucid tensor carried across."""
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, lucid.Tensor):
            array = _probe.to_numpy(value)
            out[key] = value if array is None else ref.from_numpy(array)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ref = require_ref()
    measured = measure(ref, verbose=args.verbose)

    path = args.output or pathlib.Path(__file__).parent.parent / "dtype_contract.json"
    previous = {}
    if path.exists():
        previous = json.loads(path.read_text())

    payload = {
        "_comment": [
            "Which dtypes each op accepts in the reference framework,",
            "measured rather than assumed.  The audit's dtype axis compares",
            "cpu against metal, which says the two disagree but not which",
            "one is right; this table decides that.",
            "",
            "Generated by replaying the audit's own invocation against the",
            "reference — see lucid/test/audit/tools/dtype_contract.py.  Both",
            "sides build their arguments through _probe.dtype_args so the",
            "two are answering the same question.",
            "",
            "bool_refusals / int_refusals: the class of the error the reference",
            "raised for a bool / an integer operand, for every symbol whose accepted",
            "list lacks it.  int_widths: the refused integer widths, for the symbols",
            "that refuse only some (avg_pool2d takes int64).  No uint8 -- Lucid has none.",
            "",
            "Regenerate with:",
            "    python -m lucid.test.audit.tools.dtype_contract",
        ],
        "measured_against": f"reference framework {getattr(ref, '__version__', '?')}",
        "measured_on": previous.get("measured_on", ""),
        "symbols": dict(sorted(measured.accepted.items())),
        "bool_refusals": measured.bool_refusals(),
        "int_refusals": measured.int_refusals(),
        "int_widths": measured.int_widths(),
        "unmapped": measured.unmapped,
    }
    # The date moves with the measurement, not with the run: an unchanged contract stays
    # byte-identical, so regenerating to check it leaves no diff.
    if {k: v for k, v in payload.items() if k != "measured_on"} != {
        k: v for k, v in previous.items() if k != "measured_on"
    }:
        payload["measured_on"] = datetime.date.today().isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(measured.accepted)} symbols measured, {len(measured.unmapped)} unmapped -> {path}")


if __name__ == "__main__":
    main()
