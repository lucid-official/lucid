"""The discrete samplers are exact, per element, and not a Python loop (LCD-681).

``Binomial.sample`` used to switch to a Normal approximation for the whole batch
once any ``total_count`` passed 25 (``Binomial(100, 0.01)`` had variance 0.84
against 0.99, and the ``n = 10`` members of a mixed batch were approximated
too), and ``lucid.poisson`` approximated every rate from 30 up and walked the
elements one by one on the host (0.5 s for 1e5 draws of ``Poisson(5)``).

The guard is the total-variation distance to the exact pmf.  An exact sampler
sits at the sampling noise of its own histogram, which is computed from the pmf
itself (``0.5 * sum sqrt(2 p / (pi N))``) and so scales with the regime; the
old approximations sat 5-30 times above it.
"""

import math
import time
from collections import Counter
from collections.abc import Callable

import pytest

import lucid
import lucid.distributions as D

N = 200_000


def _poisson_pmf(lam: float) -> Callable[[int], float]:
    def pmf(k: int) -> float:
        if lam == 0.0:
            return float(k == 0)
        return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

    return pmf


def _binomial_pmf(n: int, p: float) -> Callable[[int], float]:
    def pmf(k: int) -> float:
        if k < 0 or k > n:
            return 0.0
        if p <= 0.0 or p >= 1.0:
            return float(k == (n if p >= 1.0 else 0))
        return math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log1p(-p)
        )

    return pmf


def _tv_and_noise(samples: list[float], pmf: Callable[[int], float]) -> tuple[float, float]:
    """Total-variation distance of the histogram to ``pmf``, and the distance an exact sampler shows."""
    counts = Counter(int(x) for x in samples)
    lo, hi = max(0, min(counts) - 60), max(counts) + 60
    total = len(samples)
    tv = noise = mass = 0.0
    for k in range(lo, hi + 1):
        q = pmf(k)
        mass += q
        tv += abs(counts.get(k, 0) / total - q)
        noise += math.sqrt(2.0 * q / (math.pi * total))
    tv += max(0.0, 1.0 - mass)
    return 0.5 * tv, 0.5 * noise


def _assert_exact(samples: list[float], pmf: Callable[[int], float], what: str) -> None:
    tv, noise = _tv_and_noise(samples, pmf)
    assert tv <= 1.6 * noise + 0.0015, f"{what}: TV {tv:.4f} against sampling noise {noise:.4f}"


@pytest.mark.parametrize("rate", [0.3, 5.0, 9.5, 10.0, 12.0, 30.0, 50.0, 1000.0])
def test_poisson_matches_the_exact_pmf(rate: float, device: str) -> None:
    lucid.manual_seed(11)
    draws = lucid.poisson(lucid.full((N,), rate, device=device))
    assert draws.dtype == lucid.float32 and draws.device.type == device
    _assert_exact(draws.tolist(), _poisson_pmf(rate), f"Poisson({rate})")


@pytest.mark.parametrize(
    ("n", "p"),
    [
        (10, 0.3),
        (25, 0.1),
        (26, 0.1),
        (100, 0.01),
        (1000, 0.001),
        (100, 0.5),
        (100, 0.9),
        (1000, 0.3),
        (5, 0.99),
    ],
)
def test_binomial_matches_the_exact_pmf(n: int, p: float, device: str) -> None:
    lucid.manual_seed(12)
    dist = D.Binomial(n, lucid.tensor(p, device=device))
    draws = dist.sample((N,))
    assert draws.device.type == device
    _assert_exact(draws.tolist(), _binomial_pmf(n, p), f"Binomial({n}, {p})")


def test_a_mixed_batch_is_exact_per_element(device: str) -> None:
    # One large ``total_count`` used to push the whole batch onto the Normal
    # approximation, so the n = 10 members were wrong as well.
    lucid.manual_seed(13)
    dist = D.Binomial(
        lucid.tensor([10.0, 10.0, 100.0, 1000.0], device=device),
        probs=lucid.tensor([0.1, 0.5, 0.01, 0.3], device=device),
    )
    columns = dist.sample((N,)).transpose(0, 1).tolist()
    specs = [(10, 0.1), (10, 0.5), (100, 0.01), (1000, 0.3)]
    for column, (n, p) in zip(columns, specs, strict=True):
        _assert_exact(column, _binomial_pmf(n, p), f"Binomial({n}, {p}) in a mixed batch")


def test_a_mixed_rate_batch_is_exact_per_element(device: str) -> None:
    lucid.manual_seed(14)
    rates = lucid.tensor([0.5, 9.0, 11.0, 80.0], device=device)
    columns = lucid.poisson(rates.expand(N, 4)).transpose(0, 1).tolist()
    for column, rate in zip(columns, [0.5, 9.0, 11.0, 80.0], strict=True):
        _assert_exact(column, _poisson_pmf(rate), f"Poisson({rate}) in a mixed batch")


@pytest.mark.parametrize(
    ("n", "p"), [(100, 0.01), (1000, 0.001), (60, 0.5), (40, 0.95), (2000, 0.2)]
)
def test_binomial_mean_and_variance(n: int, p: float, device: str) -> None:
    lucid.manual_seed(15)
    x = D.Binomial(n, lucid.tensor(p, device=device)).sample((N,)).float()
    mean, var = n * p, n * p * (1.0 - p)
    assert abs(x.mean().item() - mean) <= 5.0 * math.sqrt(var / N)
    assert abs(x.var().item() / var - 1.0) <= 0.03


@pytest.mark.parametrize("rate", [0.5, 5.0, 40.0, 1000.0])
def test_poisson_mean_and_variance(rate: float, device: str) -> None:
    lucid.manual_seed(16)
    x = lucid.poisson(lucid.full((N,), rate, device=device)).float()
    assert abs(x.mean().item() - rate) <= 5.0 * math.sqrt(rate / N)
    assert abs(x.var().item() / rate - 1.0) <= 0.03


def test_negative_binomial_follows_the_vectorised_poisson(device: str) -> None:
    lucid.manual_seed(17)
    dist = D.NegativeBinomial(
        lucid.tensor(5.0, device=device), probs=lucid.tensor(0.3, device=device)
    )
    x = dist.sample((N,)).float()
    assert abs(x.mean().item() / dist.mean.item() - 1.0) <= 0.02
    assert abs(x.var().item() / dist.variance.item() - 1.0) <= 0.04


def test_degenerate_parameters(device: str) -> None:
    dist = D.Binomial(
        lucid.tensor([0.0, 5.0, 5.0, 7.0], device=device),
        probs=lucid.tensor([0.3, 0.0, 1.0, 0.5], device=device),
    )
    draws = dist.sample((50,)).tolist()
    assert all(row[0] == 0 and row[1] == 0 and row[2] == 5 and 0 <= row[3] <= 7 for row in draws)
    zero = lucid.poisson(lucid.zeros(4, device=device))
    assert zero.tolist() == [0, 0, 0, 0]
    assert lucid.poisson(lucid.zeros(0, device=device)).shape == (0,)
    assert lucid.poisson(lucid.tensor(3.0, device=device)).shape == ()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_poisson_rejects_a_rate_it_cannot_draw(bad: float) -> None:
    with pytest.raises(ValueError, match="rate must be finite and non-negative"):
        lucid.poisson(lucid.tensor([1.0, bad]))


def test_the_same_seed_gives_the_same_samples(device: str) -> None:
    def draw() -> list[list[float]]:
        lucid.manual_seed(21)
        binom = D.Binomial(100, lucid.tensor(0.3, device=device)).sample((500,)).tolist()
        pois = lucid.poisson(lucid.full((500,), 30.0, device=device)).tolist()
        neg = D.NegativeBinomial(
            lucid.tensor(5.0, device=device), probs=lucid.tensor(0.3, device=device)
        ).sample((500,))
        return [binom, pois, neg.tolist()]

    assert draw() == draw()


def test_an_explicit_generator_replays_poisson(device: str) -> None:
    rate = lucid.full((300,), 25.0, device=device)
    first = lucid.poisson(rate, generator=lucid.Generator(3)).tolist()
    lucid.manual_seed(99)  # the global stream must not matter
    assert lucid.poisson(rate, generator=lucid.Generator(3)).tolist() == first


def test_poisson_is_not_a_per_element_python_loop() -> None:
    lucid.poisson(lucid.full((64,), 5.0))  # engine warm-up
    start = time.perf_counter()
    lucid.poisson(lucid.full((100_000,), 5.0))
    elapsed = time.perf_counter() - start
    # ~0.03 s measured; the loop took 0.54 s.  A generous bound, not a benchmark.
    assert elapsed < 0.25, f"1e5 Poisson(5) draws took {elapsed:.2f} s"
