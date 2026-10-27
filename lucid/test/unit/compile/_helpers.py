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

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor

# Compile path is Metal-only — MPSGraph is the only emit backend in
# 3.5.  Tests must put both model and inputs on the ``metal`` device
# before calling ``lucid.compile``; CPU traces are auto-routed to
# eager-fallback (the cache stays empty and ``eager_only`` grows).
COMPILE_DEVICE = "metal"


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
    model: nn.Module,
    *inputs: Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-5,
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


def assert_compiles(
    model: nn.Module,
    *inputs: Tensor,
    atol: float = 1e-4,
) -> None:
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
