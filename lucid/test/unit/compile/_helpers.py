"""Shared helpers for the ``lucid.compile`` unit test surface.

The compile-side tests assert two properties against the eager path:

  * **Bit-exact parity** — every cached executable produces the same
    tensors the eager path would.  ``assert_compile_parity`` runs both
    paths and compares with ``rtol=0`` / ``atol=0`` by default; loosen
    on a case-by-case basis (we expect 0.0 for inference paths).

  * **Cache state** — ``compile_cached`` builds a :class:`CompiledModule`
    and primes it with one warm-up call so subsequent invocations are
    guaranteed-cache-hit, which keeps the parity assertions free of
    first-call recompile noise.

These helpers are deliberately small: every test should still read top
to bottom.  They live in this folder (not ``lucid/test/_helpers``)
because they only make sense for the compile suite.
"""

import contextlib
import os
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from unittest import mock

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.test._fixtures.compile_notices import VERBOSE_ENV

# Compile path is Metal-only — MPSGraph is the only emit backend in
# 3.5.  Tests must put both model and inputs on the ``metal`` device
# before calling ``lucid.compile``; CPU traces are auto-routed to
# eager-fallback (the cache stays empty and ``eager_only`` grows).
COMPILE_DEVICE = "metal"


@contextlib.contextmanager
def verbose_compile_notices() -> Iterator[None]:
    """``LUCID_COMPILE_VERBOSE=1`` for the ``with`` block, the old environment after it."""
    with mock.patch.dict(os.environ, {VERBOSE_ENV: "1"}):
        yield


def to_metal(model: nn.Module) -> nn.Module:
    """Move ``model`` to the metal device.  Returns ``model`` for chaining."""
    return model.to(COMPILE_DEVICE)


def metal_tensor(*args: object, **kwargs: object) -> Tensor:
    """Construct a tensor on the metal device.  Wraps ``lucid.randn`` etc."""
    t = lucid.randn(*args, **kwargs)
    return t.to(COMPILE_DEVICE)


def unwrap(out: object) -> Tensor:
    """Extract the underlying tensor from a model output (Tensor or dataclass).

    Lucid model factories return ``BaseModelOutput`` / ``ImageClassifierOutput``
    dataclasses with a ``last_hidden_state`` or ``logits`` field; raw
    nn.Module subclasses return tensors directly.  Tests handle both.
    """
    if isinstance(out, Tensor):
        return out
    for attr in ("logits", "last_hidden_state", "prediction"):
        v = getattr(out, attr, None)
        if isinstance(v, Tensor):
            return v
    raise TypeError(f"cannot unwrap {type(out).__name__!r} to a Tensor")


def assert_compile_parity(
    model: nn.Module, *inputs: Tensor, atol: float = 1e-4, rtol: float = 1e-5
) -> tuple[Tensor, Tensor]:
    """Assert ``compile(model)(x)`` matches eager ``model(x)`` within fp32 tolerance.

    The eager path goes through MLX kernels; the compile path emits
    via MPSGraph.  Both lower to Metal but with different fusion +
    reduction ordering, so results agree to ~1e-6 absolute on a
    well-conditioned net rather than bit-for-bit.  The default
    tolerance (atol=1e-4, rtol=1e-5) catches algorithmic regressions
    (wrong gate order / weight layout / etc. would produce O(1)
    drift) without flagging legitimate fp32 reordering.

    For paths where both backends share the same underlying kernel
    (LSTM dispatches MLX inside the MPSGraph executable), bit-exact
    equality holds — override with ``atol=0, rtol=0`` for those.

    Returns
    -------
    (eager_out, compiled_out) : both unwrapped to :class:`Tensor`.
    """
    model.eval()
    to_metal(model)
    inputs = tuple(x.to(COMPILE_DEVICE) for x in inputs)
    eager = unwrap(model(*inputs))

    compiled_model = lucid.compile(model)
    # Under grad mode a model with trainable parameters runs eagerly (LCD-582).
    with lucid.no_grad():
        compiled = unwrap(compiled_model(*inputs))

    diff = float((eager - compiled).abs().max().item())
    max_eager = float(eager.abs().max().item())
    rel = diff / max(max_eager, 1e-12)

    if atol == 0.0 and rtol == 0.0:
        assert diff == 0.0, f"bit-exact parity broken: abs diff = {diff:.3e}, " f"rel = {rel:.3e}"
    else:
        threshold = atol + rtol * max_eager
        assert diff <= threshold, (
            f"parity outside tolerance: abs diff = {diff:.3e}, "
            f"rel = {rel:.3e} (atol={atol}, rtol={rtol}, "
            f"threshold={threshold:.3e})"
        )

    return eager, compiled


def assert_cache_hit(compiled_model: object, expected_entries: int = 1) -> None:
    """Assert the compiled module has exactly ``expected_entries`` cached executables.

    Tests that prime + call should see one entry; tests that vary shape
    should see N.  This catches regressions where the signature key
    accidentally captures something it shouldn't (causing spurious
    cache misses on identical inputs).
    """
    info = compiled_model.cache_info()
    n = info["entries"]
    assert n == expected_entries, (
        f"cache entries: expected {expected_entries}, got {n}. " f"keys = {info['keys']}"
    )


def assert_compiles(model: nn.Module, *inputs: Tensor, atol: float = 1e-4) -> None:
    """Assert ``model`` actually COMPILES (no eager fallback) and matches eager.

    The plain :func:`assert_compile_parity` only checks the *output* — which
    passes even when the model silently falls back to eager (the eager-vs-eager
    diff is then trivially 0).  That blind spot is exactly how single-layer
    LSTM compile stayed broken for months.  This helper additionally asserts
    the executable was built and nothing was blacklisted as ``eager_only``, so
    a regression that pushes a model back to eager is caught.
    """
    model.eval()
    to_metal(model)
    inputs = tuple(x.to(COMPILE_DEVICE) for x in inputs)
    eager = unwrap(model(*inputs))

    cm = lucid.compile(model)
    # Under grad mode a model with trainable parameters runs eagerly (LCD-582).
    with lucid.no_grad():
        compiled = unwrap(cm(*inputs))
    compiled.eval()

    info = cm.cache_info()
    assert info["entries"] >= 1 and not info["eager_only"], (
        f"{type(model).__name__} did not compile cleanly: "
        f"entries={info['entries']}, eager_only={info['eager_only']}"
    )

    diff = float((eager - compiled).abs().max().item())
    assert diff <= atol, f"{type(model).__name__} compile parity broken: {diff:.3e}"


def assert_no_eager_fallback(compiled_model: object) -> None:
    """Assert no signature was blacklisted as eager-only.

    A clean compile should never insert anything into the eager_only
    set.  If it does, the trace contained an op without an emitter or
    an emitter returned nullptr — the test should pinpoint which.
    """
    info = compiled_model.cache_info()
    eager_only = info["eager_only"]
    assert not eager_only, (
        f"expected zero eager-fallback signatures, got {len(eager_only)}: " f"{eager_only}"
    )


# ── wall-clock guards ─────────────────────────────────────────────────────────────────
#
# A timing guard compares one variant against another (rank-5 against its rank-4 twin,
# compiled against eager).  One wall-clock sample per side flakes under machine load: a
# spike that lands on one side reads as a regression.  Every such guard goes through
# ``compare_ms`` instead (``test_timing_guards_use_the_helper.py`` rejects a raw
# ``perf_counter`` in a test file): both sides are measured in alternation, so a load
# spike hits both, each side keeps its minimum over every round, and a measurement that
# still misses the bound is retried from a fresh build before it is reported.
#
# None of that removes a *persistent* difference between two graphs, and the hosted runner's
# virtual GPU has them: the same fixed graph measured 0.75x its comparison on an M1 Pro and
# 1.11x on the runner (``layer_norm_after_add``, LCD-935), and a rank-5 multiply 2.1x and 4.4x.
# A guard is therefore a bound on a *large* cost -- the defects they were written for ran 2x to
# 100x their twin -- and ``compare_ms`` refuses a ratio bound it cannot resolve.  A cost below
# that is asserted on the graph (``_canon`` in the ``test_rank_canonical_*`` files), not on a
# clock.

Calls = Mapping[str, Callable[[], object]]

#: The smallest ratio bound a guard may state (``1.0`` itself is allowed: it says "no ratio,
#: only ``slack_ms``").  The bounds that held on the runner run after run are 1.5x and up; the
#: ones that failed there were 1.1x and a 3x that a graph at 2.1x here measured 4.4x on it.
MIN_RATIO_BOUND = 1.5
#: What a guard that states no slack of its own gets: the launch and sync jitter of a
#: millisecond-scale graph, which a ratio of a ~1 ms side does not cover.
MIN_SLACK_MS = 0.25


@dataclass(frozen=True)
class Timing:
    """The best comparison ``compare_ms`` saw: the first to hold, else the lowest ratio."""

    ratio: float
    ms: Mapping[str, float]
    attempts: int
    within: bool
    report: str


def read_sum(out: Tensor) -> float:
    """Force ``out`` with a full reduction (the cost lands on every side alike)."""
    return float(out.sum().item())


def read_first(out: Tensor) -> float:
    """Force ``out`` through one element, so a rank-5 output's copy-out is measured."""
    return float(out.reshape(-1)[0].item())


def compiled_call(
    fn: Callable[..., Tensor], *args: Tensor, read: Callable[[Tensor], float] = read_sum
) -> Callable[[], float]:
    """A self-synchronising call of ``lucid.compile(fn)`` on ``args``, for ``compare_ms``."""
    cm = lucid.compile(fn)

    def call() -> float:
        with lucid.no_grad():
            return read(cm(*args))

    return call


def step_call(step: Callable[..., Tensor], *args: Tensor) -> Callable[[], float]:
    """A self-synchronising forward + backward of a ``make_step`` step, for ``compare_ms``."""

    def call() -> float:
        loss = step(*args)
        loss.backward()
        return float(loss.item())

    return call


def with_env[T](call: Callable[[], T], name: str, value: str | None) -> Callable[[], T]:
    """``call`` with ``name`` set to ``value`` (unset when ``None``) while it runs.

    A compile setting is read when the cache key is made, on *every* call
    (``compile_policy``), so a side measured under a setting has to run each call under it:
    setting the variable once around the warm-up would time a recompile under the other.
    """

    def wrapped() -> T:
        before = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        try:
            return call()
        finally:
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before

    return wrapped


def _measure(calls: Calls, rounds: int, reps: int, warmup: int) -> dict[str, float]:
    for call in calls.values():
        for _ in range(warmup):
            call()
    best = dict.fromkeys(calls, float("inf"))
    names = list(calls)
    for rnd in range(rounds):
        for name in names if rnd % 2 == 0 else reversed(names):
            for _ in range(reps):
                t0 = time.perf_counter()
                calls[name]()
                best[name] = min(best[name], time.perf_counter() - t0)
    return {name: t * 1e3 for name, t in best.items()}


def _checked_slack(bound: float, slack_ms: float | None) -> float:
    """The slack a guard runs with, after refusing a ratio bound a runner cannot resolve."""
    if 1.0 < bound < MIN_RATIO_BOUND:
        raise ValueError(
            f"a wall-clock ratio bound of {bound}x is below what a shared runner resolves "
            f"({MIN_RATIO_BOUND}x): the same fixed graph measured 0.75x here and 1.11x on the "
            "hosted runner (LCD-935).  Bound a large cost with a ratio >= "
            f"{MIN_RATIO_BOUND}, or assert a small one on the graph; bound=1.0 with slack_ms "
            "is an absolute guard."
        )
    return MIN_SLACK_MS if slack_ms is None else slack_ms


def compare_ms(
    calls: Calls | Callable[[], Calls],
    num: str,
    den: str,
    *,
    bound: float,
    slack_ms: float | None = None,
    attempts: int = 3,
    rounds: int = 4,
    reps: int = 5,
    warmup: int = 3,
) -> Timing:
    """Hold ``calls[num]`` to ``bound * calls[den] + slack_ms``, on a loaded machine too.

    ``calls`` maps a name to a callable that does its own work *and* waits for it (the
    device runs asynchronously).  Pass a zero-argument factory instead of the mapping when
    each attempt needs a fresh build (a cache clear, a recompile under an environment).
    Each attempt warms every callable, then times them in rounds that alternate the order,
    ``reps`` calls apiece, and keeps each side's minimum.  The first attempt to hold ends
    the search; otherwise the lowest-ratio attempt is returned with ``within=False``.

    ``bound`` is ``1.0`` (absolute only) or at least ``MIN_RATIO_BOUND``; ``slack_ms`` left
    out is ``MIN_SLACK_MS``, an explicit value (``0.0`` too) is taken as stated.
    """
    slack_ms = _checked_slack(bound, slack_ms)
    best: Timing | None = None
    for attempt in range(1, attempts + 1):
        ms = _measure(calls if isinstance(calls, Mapping) else calls(), rounds, reps, warmup)
        ratio = ms[num] / ms[den]
        within = ms[num] <= bound * ms[den] + slack_ms
        report = (
            f"{num} {ms[num]:.3f} ms vs {den} {ms[den]:.3f} ms = {ratio:.2f}x "
            f"(bound {bound}x + {slack_ms} ms, attempt {attempt}/{attempts})"
        )
        if best is None or within or ratio < best.ratio:
            best = Timing(ratio, ms, attempt, within, report)
        if within:
            break
    assert best is not None, "attempts must be at least 1"
    return best


def assert_ratio_within(
    calls: Calls | Callable[[], Calls],
    num: str,
    den: str,
    *,
    bound: float,
    slack_ms: float | None = None,
    attempts: int = 3,
    rounds: int = 4,
    reps: int = 5,
    warmup: int = 3,
    what: str = "",
) -> Timing:
    """``compare_ms`` that fails the test with the report when no attempt held."""
    timing = compare_ms(
        calls,
        num,
        den,
        bound=bound,
        slack_ms=slack_ms,
        attempts=attempts,
        rounds=rounds,
        reps=reps,
        warmup=warmup,
    )
    assert timing.within, f"{what + ': ' if what else ''}{timing.report}"
    return timing
