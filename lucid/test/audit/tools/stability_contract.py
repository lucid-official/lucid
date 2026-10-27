"""Measure which scales the reference framework answers finitely at.

The stability axis feeds each op an input scaled across fourteen orders
of magnitude and reports a NaN as a defect.  Its premise — "a finite,
in-domain input must not produce a NaN" — carries the whole check in the
words *in domain*, and the axis had no way to know where that was.

So it reported ``acos`` for answering NaN at 1e+6.  ``acos`` is defined
on [-1, 1]; NaN is the only answer it *can* give there, and the same is
true of ``asin``, ``atanh``, ``logit``, ``erfinv`` and ``acosh``.  Ten of
the twenty-six findings were the op being right.

The domain cannot be derived from the op — it is a mathematical fact
about the function, not a property of the code — and hand-listing it is
how the dtype table went wrong before it was measured.  So it is
measured here the same way: the audit's own probe, at the audit's own
scales, replayed against the reference.  A NaN Lucid produces where the
reference produces a number is a defect; a NaN they both produce is the
function's domain.

Run::

    python -m lucid.test.audit.tools.stability_contract

The result is checked in as ``stability_contract.json`` so the audit
reads it without the reference installed.
"""

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from lucid.test._fixtures.ref_framework import require_ref
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context
from lucid.test.audit._axes_stability import StabilityAxis
from lucid.test.audit.tools.dtype_contract import _reference_callable, _reference_kwargs


def measure(ref: Any, verbose: bool = False) -> dict[str, list[str]]:
    """For each symbol, the scales at which the reference is finite."""
    ctx = Context(metal=False)
    finite: dict[str, list[str]] = {}

    for symbol in _surface.enumerate_surface():
        fn = _surface.resolve(symbol)
        if fn is None:
            continue
        target = _reference_callable(ref, symbol.qualname)
        if target is None:
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
            base = np.abs(call.base) + 0.25
        except TypeError:
            continue
        if base.ndim == 0 or base.size == 1:
            continue

        ok: list[str] = []
        evaluated = 0
        for scale in StabilityAxis._SCALES:
            probe = base * scale
            if not np.isfinite(probe).all():
                continue
            try:
                args = [
                    (
                        ref.from_numpy(np.ascontiguousarray(probe))
                        if index == call.primary
                        else _carry(ref, value)
                    )
                    for index, value in enumerate(call.args)
                ]
                fn_ref = target
                if isinstance(target, str):
                    if call.primary >= len(args):
                        break
                    fn_ref = getattr(args[call.primary], target, None)
                    if fn_ref is None:
                        break
                    args = args[: call.primary] + args[call.primary + 1 :]
                out = fn_ref(*args, **_reference_kwargs(ref, call.kwargs))
            except Exception:  # noqa: BLE001 - a refusal is not a finite answer
                continue
            array = _probe.to_numpy(out)
            if array is None or array.dtype.kind not in "fc":
                continue
            evaluated += 1
            if not np.isnan(array).any():
                ok.append(f"{scale:g}")
        # Recorded even when the list comes back empty, as long as the
        # replay actually ran.  Dropping those lost the most useful thing
        # the table can say: ``zeta`` and ``xlogy`` are NaN in the
        # reference at *every* scale on this probe — their companion
        # argument carries negatives from the domain — so a NaN from
        # Lucid is the function's domain rather than a defect.  Absent and
        # "finite nowhere" are different facts and were being written the
        # same way.
        if evaluated:
            finite[symbol.qualname] = ok
        if verbose:
            print(f"{symbol.qualname:52s} {ok}")

    return finite


def _carry(ref: Any, value: Any) -> Any:
    """A companion argument, moved across unchanged."""
    array = _probe.to_numpy(value)
    if array is None or not hasattr(value, "shape"):
        return value
    return ref.from_numpy(np.ascontiguousarray(array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ref = require_ref()
    finite = measure(ref, verbose=args.verbose)

    path = args.output or pathlib.Path(__file__).parent.parent / "stability_contract.json"
    payload = {
        "_comment": [
            "Which input scales the reference framework answers finitely at,",
            "measured rather than assumed.  The stability axis reports a NaN",
            "from a finite input as a defect; whether that is right depends",
            "on the op's domain, which is a fact about the function and not",
            "something the code can be asked.",
            "",
            "A NaN Lucid produces where the reference produces a number is a",
            "defect.  A NaN they both produce is the domain — acos of 1e6 has",
            "no other answer.",
            "",
            "Regenerate with:",
            "    python -m lucid.test.audit.tools.stability_contract",
        ],
        "measured_against": f"reference framework {getattr(ref, '__version__', '?')}",
        "scales": [f"{s:g}" for s in StabilityAxis._SCALES],
        "finite_at": dict(sorted(finite.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(finite)} symbols measured -> {path}")


if __name__ == "__main__":
    main()
