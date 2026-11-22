"""Cost of the range-safe Gamma / Beta / Dirichlet / Poisson densities and entropies (LCD-1050).

``lucid/distributions/_stirling_math.py`` evaluates the plain and the centred (Stirling) branch on
every lane and selects with ``where``: a data-dependent branch would sync a device and break
tracing.  This script times what that costs against the closed form the distributions used before
(LCD-1041), forward plus backward, so a restructure can be judged by a ratio and not a feeling.

Run it as a module (it is not a pytest module; ``test_bench_smoke.py`` only checks it still runs)::

    python3 -m lucid.test.perf.distributions.bench_gamma_family                 # 1e6, cpu + metal
    python3 -m lucid.test.perf.distributions.bench_gamma_family --n 10000 --repeats 3   # smoke
    python3 -m lucid.test.perf.distributions.bench_gamma_family --devices cpu --dtype float64
    python3 -m lucid.test.perf.distributions.bench_gamma_family --cases beta_log_prob,gamma_entropy

What is measured
----------------
Per case, two variants on the same inputs:

* ``plain`` -- the pre-LCD-1041 closed form written with ``lgamma`` / ``digamma`` / ``xlogy``;
* ``current`` -- the public distribution method (``validate_args=False``), i.e. the owner's code.

Each call is one forward and the backward of its sum to every differentiable input, with the
outputs materialised and the device synchronised before the clock stops (an MLX graph is lazy:
timing the forward alone measures graph construction).  The reported time is the **minimum** of
``--repeats`` runs after warm-up; ``ratio`` is ``current / plain``.  The inputs sit in the centred
lane (large parameters, values near the mean), but both branches run on every lane, so the cost does
not depend on the values.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import lucid
import lucid.distributions as D
from lucid.special import xlog1py
from tools._bench_timing import cold_ms

# Each case builds its differentiable inputs for ``n`` elements and returns the two variants.
Variant = Callable[[Sequence[lucid.Tensor]], lucid.Tensor]
Inputs = list[lucid.Tensor]


@dataclass(frozen=True)
class Case:
    """One density or entropy: how to draw its inputs, and the plain and the current evaluation."""

    name: str
    build: Callable[[int, str, lucid.dtype], Inputs]
    plain: Variant
    current: Variant


def _uniform(n: int, low: float, high: float, device: str, dtype: lucid.dtype) -> lucid.Tensor:
    return low + (high - low) * lucid.rand(n, device=device, dtype=dtype)


def _grad(t: lucid.Tensor) -> lucid.Tensor:
    return t.detach().requires_grad_(True)


def _beta_inputs(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    a = _uniform(n, 30.0, 300.0, device, dtype)
    b = _uniform(n, 30.0, 300.0, device, dtype)
    x = (a / (a + b)) * (1.0 + 0.02 * lucid.randn(n, device=device, dtype=dtype))
    return [_grad(a), _grad(b), _grad(lucid.clamp(x, min=1e-3, max=1.0 - 1e-3))]


def _beta_params(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    return _beta_inputs(n, device, dtype)[:2]


def _beta_lp_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    a, b, x = v
    log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(a + b)
    return lucid.xlogy(a - 1.0, x) + xlog1py(b - 1.0, -x) - log_b


def _beta_ent_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    a, b = v
    s = a + b
    log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(s)
    return (
        log_b
        - (a - 1.0) * lucid.digamma(a)
        - (b - 1.0) * lucid.digamma(b)
        + (s - 2.0) * lucid.digamma(s)
    )


def _gamma_inputs(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    alpha = _uniform(n, 20.0, 200.0, device, dtype)
    rate = _uniform(n, 0.5, 2.0, device, dtype)
    x = (alpha / rate) * (1.0 + 0.05 * lucid.randn(n, device=device, dtype=dtype))
    return [_grad(alpha), _grad(rate), _grad(lucid.clamp(x, min=1e-3))]


def _gamma_params(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    return [_grad(_uniform(n, 1.0, 200.0, device, dtype))]


def _gamma_lp_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    alpha, rate, x = v
    return alpha * lucid.log(rate) + lucid.xlogy(alpha - 1.0, x) - rate * x - lucid.lgamma(alpha)


def _gamma_ent_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    (alpha,) = v
    return alpha + lucid.lgamma(alpha) + (1.0 - alpha) * lucid.digamma(alpha)


_K = 3


def _dirichlet_inputs(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    rows = max(n // _K, 1)
    a = _uniform(rows * _K, 20.0, 200.0, device, dtype).reshape(rows, _K)
    x = a / a.sum(dim=-1, keepdim=True)
    return [_grad(a), _grad(x)]


def _dirichlet_params(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    return _dirichlet_inputs(n, device, dtype)[:1]


def _dirichlet_lp_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    a, x = v
    log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(a.sum(dim=-1))
    return lucid.xlogy(a - 1.0, x).sum(dim=-1) - log_b


def _dirichlet_ent_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    (a,) = v
    s = a.sum(dim=-1)
    log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(s)
    return log_b + (s - _K) * lucid.digamma(s) - ((a - 1.0) * lucid.digamma(a)).sum(dim=-1)


def _poisson_inputs(n: int, device: str, dtype: lucid.dtype) -> Inputs:
    rate = _uniform(n, 20.0, 200.0, device, dtype)
    k = lucid.floor(rate + rate.sqrt() * lucid.randn(n, device=device, dtype=dtype))
    return [_grad(rate), lucid.clamp(k, min=0.0).detach()]


def _poisson_lp_plain(v: Sequence[lucid.Tensor]) -> lucid.Tensor:
    rate, k = v
    return lucid.xlogy(k, rate) - rate - lucid.lgamma(k + 1.0)


CASES: tuple[Case, ...] = (
    Case(
        "beta_log_prob",
        _beta_inputs,
        _beta_lp_plain,
        lambda v: D.Beta(v[0], v[1], validate_args=False).log_prob(v[2]),
    ),
    Case(
        "beta_entropy",
        _beta_params,
        _beta_ent_plain,
        lambda v: D.Beta(v[0], v[1], validate_args=False).entropy(),
    ),
    Case(
        "gamma_log_prob",
        _gamma_inputs,
        _gamma_lp_plain,
        lambda v: D.Gamma(v[0], v[1], validate_args=False).log_prob(v[2]),
    ),
    Case(
        "gamma_entropy",
        _gamma_params,
        _gamma_ent_plain,
        lambda v: D.Gamma(v[0], lucid.ones_like(v[0]), validate_args=False).entropy(),
    ),
    Case(
        "dirichlet_log_prob",
        _dirichlet_inputs,
        _dirichlet_lp_plain,
        lambda v: D.Dirichlet(v[0], validate_args=False).log_prob(v[1]),
    ),
    Case(
        "dirichlet_entropy",
        _dirichlet_params,
        _dirichlet_ent_plain,
        lambda v: D.Dirichlet(v[0], validate_args=False).entropy(),
    ),
    Case(
        "poisson_log_prob",
        _poisson_inputs,
        _poisson_lp_plain,
        lambda v: D.Poisson(v[0], validate_args=False).log_prob(v[1]),
    ),
)
CASE_NAMES = tuple(c.name for c in CASES)


def _step(variant: Variant, inputs: Inputs) -> Callable[[], tuple[lucid.Tensor, ...]]:
    """One forward and the backward of its sum to every input that takes a gradient."""
    wanted = [t for t in inputs if t.requires_grad]

    def step() -> tuple[lucid.Tensor, ...]:
        for t in wanted:
            t.grad = None
        out = variant(inputs)
        out.sum().backward()
        return (out, *(t.grad for t in wanted if t.grad is not None))

    return step


def time_ms(variant: Variant, inputs: Inputs, repeats: int, warmup: int = 2) -> float:
    """Minimum wall-clock milliseconds of ``repeats`` materialised forward + backward steps."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"time_ms: need repeats >= 1 and warmup >= 0, got {repeats}, {warmup}")
    step = _step(variant, inputs)
    for _ in range(warmup):
        cold_ms(step)
    return min(cold_ms(step) for _ in range(repeats))


def run(
    n: int, devices: Sequence[str], dtype: lucid.dtype, repeats: int, cases: Sequence[str]
) -> list[dict[str, object]]:
    """Time every selected case on every device; one row per (case, device)."""
    unknown = sorted(set(cases) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; choose from {list(CASE_NAMES)}")
    rows: list[dict[str, object]] = []
    for device in devices:
        for case in (c for c in CASES if c.name in cases):
            lucid.manual_seed(0)
            inputs = case.build(n, device, dtype)
            plain = time_ms(case.plain, inputs, repeats)
            current = time_ms(case.current, inputs, repeats)
            rows.append(
                {
                    "case": case.name,
                    "device": device,
                    "dtype": str(dtype),
                    "n": n,
                    "plain_ms": plain,
                    "current_ms": current,
                    "ratio": current / plain,
                }
            )
    return rows


def _print(rows: Sequence[dict[str, object]]) -> None:
    print(f"{'case':<20} {'device':<6} {'plain ms':>10} {'current ms':>11} {'ratio':>7}")
    for r in rows:
        print(
            f"{r['case']!s:<20} {r['device']!s:<6} {float(str(r['plain_ms'])):>10.2f} "
            f"{float(str(r['current_ms'])):>11.2f} {float(str(r['ratio'])):>6.2f}x"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000, help="elements per input (default 1e6)")
    parser.add_argument("--repeats", type=int, default=9, help="timed runs; the minimum is kept")
    parser.add_argument(
        "--devices", default="cpu,metal", help="comma separated (default cpu,metal)"
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--cases", default=",".join(CASE_NAMES), help="comma separated subset")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    dtype = lucid.float64 if args.dtype == "float64" else lucid.float32
    rows = run(args.n, args.devices.split(","), dtype, args.repeats, args.cases.split(","))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print(rows)


if __name__ == "__main__":
    main()
