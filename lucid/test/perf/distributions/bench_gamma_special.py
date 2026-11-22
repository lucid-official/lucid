"""Cost of ``lgamma`` / ``digamma`` / ``polygamma`` and of their positive-domain entries (LCD-1058).

Every gamma-family density and entropy is a handful of ``lgamma`` / ``digamma`` calls (about 90 %
of the plain Beta density's engine events were three ``lgamma``), so their own cost is the floor of
``bench_gamma_family``.  ``lucid/_ops/composite/_gamma.py`` has two entries for each: the public one,
which covers the whole real line (a reflection, the poles, an infinity guard), and a private
positive-domain one for a caller that guarantees ``x > 0``.  This script times both, forward plus
backward of the sum, on inputs in ``(0.1, 50)``.

Run it as a module (it is not a pytest module; ``test_bench_smoke.py`` only checks it still runs)::

    python3 -m lucid.test.perf.distributions.bench_gamma_special                  # 1e6, cpu + metal
    python3 -m lucid.test.perf.distributions.bench_gamma_special --n 100000 --repeats 5   # smoke
    python3 -m lucid.test.perf.distributions.bench_gamma_special --devices cpu --dtype float64
    python3 -m lucid.test.perf.distributions.bench_gamma_special --cases lgamma --forward-only

The reported time is the **minimum** of ``--repeats`` runs after warm-up, with the outputs
materialised and the device synchronised before the clock stops (an MLX graph is lazy); ``ratio``
is ``positive / public``.  ``--events`` adds the engine-event count of one step, which does not
depend on the machine (the count a structural guard can pin).
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import lucid
from lucid._ops.composite._gamma import _digamma_positive, _lgamma_positive, _polygamma_positive
from tools._bench_timing import cold_ms

Unary = Callable[[lucid.Tensor], lucid.Tensor]


@dataclass(frozen=True)
class Case:
    """One function: the public entry and the positive-domain entry of the same value."""

    name: str
    public: Unary
    positive: Unary


CASES: tuple[Case, ...] = (
    Case("lgamma", lucid.lgamma, _lgamma_positive),
    Case("digamma", lucid.digamma, _digamma_positive),
    Case("trigamma", lambda x: lucid.special.polygamma(1, x), lambda x: _polygamma_positive(1, x)),
)
CASE_NAMES = tuple(c.name for c in CASES)


def _inputs(n: int, device: str, dtype: lucid.dtype) -> lucid.Tensor:
    lucid.manual_seed(0)
    x = 0.1 + 49.9 * lucid.rand(n, device=device, dtype=dtype)
    return x.detach().requires_grad_(True)


def _step(fn: Unary, x: lucid.Tensor, forward_only: bool) -> Callable[[], lucid.Tensor]:
    """One forward and, unless ``forward_only``, the backward of its sum."""

    def step() -> lucid.Tensor:
        x.grad = None
        out = fn(x)
        if forward_only:
            return out
        out.sum().backward()
        assert x.grad is not None
        return x.grad

    return step


def time_ms(fn: Unary, x: lucid.Tensor, repeats: int, forward_only: bool, warmup: int = 2) -> float:
    """Minimum wall-clock milliseconds of ``repeats`` materialised steps."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"time_ms: need repeats >= 1 and warmup >= 0, got {repeats}, {warmup}")
    step = _step(fn, x, forward_only)
    for _ in range(warmup):
        cold_ms(step)
    return min(cold_ms(step) for _ in range(repeats))


def event_count(fn: Unary, x: lucid.Tensor, forward_only: bool) -> int:
    """Engine events of one step."""
    step = _step(fn, x, forward_only)
    step()
    with lucid.profiler.profile() as prof:
        step()
    return len(list(prof.events()))


def run(
    n: int,
    devices: Sequence[str],
    dtype: lucid.dtype,
    repeats: int,
    cases: Sequence[str],
    forward_only: bool = False,
    events: bool = False,
) -> list[dict[str, object]]:
    """Time every selected case on every device; one row per (case, device)."""
    unknown = sorted(set(cases) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; choose from {list(CASE_NAMES)}")
    rows: list[dict[str, object]] = []
    for device in devices:
        x = _inputs(n, device, dtype)
        for case in (c for c in CASES if c.name in cases):
            public = time_ms(case.public, x, repeats, forward_only)
            positive = time_ms(case.positive, x, repeats, forward_only)
            row: dict[str, object] = {
                "case": case.name,
                "device": device,
                "dtype": str(dtype),
                "n": n,
                "forward_only": forward_only,
                "public_ms": public,
                "positive_ms": positive,
                "ratio": positive / public,
            }
            if events:
                row["public_events"] = event_count(case.public, x, forward_only)
                row["positive_events"] = event_count(case.positive, x, forward_only)
            rows.append(row)
    return rows


def _print(rows: Sequence[dict[str, object]]) -> None:
    has_events = any("public_events" in r for r in rows)
    head = f"{'case':<10} {'device':<6} {'public ms':>10} {'positive ms':>12} {'ratio':>7}"
    print(head + (f" {'events (pub/pos)':>17}" if has_events else ""))
    for r in rows:
        line = (
            f"{r['case']!s:<10} {r['device']!s:<6} {float(str(r['public_ms'])):>10.2f} "
            f"{float(str(r['positive_ms'])):>12.2f} {float(str(r['ratio'])):>6.2f}x"
        )
        if has_events:
            line += f" {r['public_events']!s:>8}/{r['positive_events']!s:<8}"
        print(line)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="elements per input (default 1e6)")
    parser.add_argument("--repeats", type=int, default=9, help="timed runs; the minimum is kept")
    parser.add_argument(
        "--devices", default="cpu,metal", help="comma separated (default cpu,metal)"
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--cases", default=",".join(CASE_NAMES), help="comma separated subset")
    parser.add_argument("--forward-only", action="store_true", help="time the forward pass alone")
    parser.add_argument("--events", action="store_true", help="also count engine events per step")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    dtype = lucid.float64 if args.dtype == "float64" else lucid.float32
    rows = run(
        args.n,
        args.devices.split(","),
        dtype,
        args.repeats,
        args.cases.split(","),
        forward_only=args.forward_only,
        events=args.events,
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print(rows)


if __name__ == "__main__":
    main()
