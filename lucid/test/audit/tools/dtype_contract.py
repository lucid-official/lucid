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

The result is checked in as ``dtype_contract.json`` so the audit can read
it without the reference installed.  Symbols the reference has no
counterpart for — Lucid-only ops, and anything whose invocation cannot be
translated — are listed under ``unmapped`` rather than silently omitted,
so a missing verdict is visibly a gap and not a claim.
"""

import argparse
import json
import pathlib
from typing import Any

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


def measure(ref: Any, verbose: bool = False) -> tuple[dict[str, list[str]], list[str]]:
    """Probe every surveyed symbol at every dtype.

    Returns the accepted-dtype table and the list of symbols with no
    reference counterpart.
    """
    ctx = Context(metal=False)
    accepted: dict[str, list[str]] = {}
    unmapped: list[str] = []

    for symbol in _surface.enumerate_surface():
        fn = _surface.resolve(symbol)
        if fn is None:
            continue
        target = _reference_callable(ref, symbol.qualname)
        if target is None:
            unmapped.append(symbol.qualname)
            continue

        call = None
        with _probe.preserved_globals():
            for domain in ctx.domains:
                for candidate in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                    try:
                        out = fn(*candidate.args, **candidate.kwargs)
                    except Exception:  # noqa: BLE001 - surveying, not asserting
                        continue
                    if _probe.to_numpy(out) is not None:
                        call = candidate
                        break
                if call is not None:
                    break
        if call is None:
            continue
        try:
            call.base
        except TypeError:
            continue

        works: list[str] = []
        for name in _probe.DTYPES:
            try:
                args = _probe.dtype_args(
                    call, name, lambda a, f, n=name: _as_reference(ref, a, f, n)
                )
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
            except Exception as exc:  # noqa: BLE001 - a refusal is the measurement
                if _probe.refuses_a_value(exc):
                    works.append(name)
        if works:
            accepted[symbol.qualname] = works
        if verbose:
            print(f"{symbol.qualname:52s} {works}")

    return accepted, sorted(unmapped)


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
    accepted, unmapped = measure(ref, verbose=args.verbose)

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
            "Regenerate with:",
            "    python -m lucid.test.audit.tools.dtype_contract",
        ],
        "measured_against": f"reference framework {getattr(ref, '__version__', '?')}",
        "measured_on": previous.get("measured_on", ""),
        "symbols": dict(sorted(accepted.items())),
        "unmapped": unmapped,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(accepted)} symbols measured, {len(unmapped)} unmapped -> {path}")


if __name__ == "__main__":
    main()
