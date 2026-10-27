"""The audit's own probes have to be able to fail.

A check that cannot fail passes for free, and a pass that means nothing
reads as coverage.  The audit reports those as VACUOUS rather than PASS,
which is honest — but a vacuous cell is still an unaudited op, and there
were thirty-six of them.  Three causes, all in the instrument:

* every regression loss was probed as ``loss(x, x)``.  Two operands built
  the same way were the *same numbers*, so the loss sat at its exact
  minimum and its gradient was identically zero.  Thirteen losses, whole
  family, derivative unmeasured.
* the stochastic flag was a substring match, so ``normal`` matched
  ``normalize``, ``sample`` matched ``grid_sample`` and ``Upsample``, and
  ``poisson`` matched ``poisson_nll_loss``.  Nine deterministic symbols
  were asked to prove they respect a seed.
* ``rrelu`` was called with its default ``training=False``, which is the
  *expectation* of the uniform slope and deterministic by design.

These tests guard the instrument, not the framework.
"""

import subprocess
import sys

import numpy as np

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context


def _chosen_call(symbol: _surface.Symbol) -> object | None:
    """The invocation the audit would actually use for ``symbol``."""
    fn = _surface.resolve(symbol)
    if fn is None:
        return None
    ctx = Context(metal=False)
    with _probe.preserved_globals():
        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*call.args, **call.kwargs)
                except Exception:  # noqa: BLE001 - surveying
                    continue
                if _probe.to_numpy(out) is not None:
                    return call
    return None


def _by_name(qualname: str) -> _surface.Symbol | None:
    for symbol in _surface.enumerate_surface():
        if symbol.qualname == qualname:
            return symbol
    return None


# ── distinct operands ─────────────────────────────────────────────────────────


def test_no_invocation_passes_the_same_numbers_twice() -> None:
    """Sweeps the whole surface, because this was never about one op.

    In a subprocess.  The sweep calls every symbol the audit would call,
    and the audit gets away with that by owning its process — inside
    pytest the same sweep left enough global state behind to fail 958
    later tests.  Restricted to ``inert`` symbols for the same reason the
    axes are: ``STATEFUL`` exists to say which names a survey must not
    touch.

    Properties like ``Tensor.T`` have a ladder call of ``f(x, x)`` and no
    second operand for the duplicate to matter to, so the assertion is on
    the ops that take two real operands: the losses, where an identical
    pair puts the loss at its own minimum.
    """
    code = """
import numpy as np
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context

ctx = Context(metal=False)
offenders = []
for symbol in _surface.enumerate_surface():
    if symbol.kind not in ("op", "method") or not symbol.inert:
        continue
    try:
        fn = _surface.resolve(symbol)
    except Exception:
        continue
    if fn is None:
        continue
    call = None
    try:
        with _probe.preserved_globals():
            for domain in ctx.domains:
                for cand in _specs.invocations(
                    symbol.short, domain, symbol.qualname, fn
                ):
                    try:
                        out = fn(*cand.args, **cand.kwargs)
                    except Exception:
                        continue
                    if _probe.to_numpy(out) is not None:
                        call = cand
                        break
                if call is not None:
                    break
    except Exception:
        continue
    if call is None:
        continue
    arrays = []
    try:
        for index, arg in enumerate(call.args):
            value = _probe.to_numpy(arg) if hasattr(arg, "_impl") else None
            if value is not None and np.asarray(value).size > 1:
                arrays.append((index, np.asarray(value).astype(np.float64, copy=False)))
    except Exception:
        continue
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            (ia, a), (ib, b) = arrays[i], arrays[j]
            if a.shape == b.shape and np.array_equal(a, b):
                offenders.append(symbol.qualname + " args[%d]==args[%d]" % (ia, ib))
print("\\n".join(offenders))
"""
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = [line for line in done.stdout.splitlines() if line.strip()]
    losses = [o for o in offenders if "loss" in o]
    assert not losses, losses


def test_repeat_draws_of_one_shape_differ() -> None:
    """``_f`` is the single operand factory; two calls must not agree."""
    first = _probe.to_numpy(_specs._f((2, 4), "moderate"))
    second = _probe.to_numpy(_specs._f((2, 4), "moderate"))
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_the_first_draw_is_unchanged() -> None:
    """Keyed on ``(shape, domain)`` so a spec asking for one operand gets
    exactly what it got before — only repeats move."""
    _specs._DRAWN.clear()
    got = np.asarray(_probe.to_numpy(_specs._f((2, 4), "moderate")))
    assert np.allclose(got, _probe.sample("moderate", (2, 4), 0))


def test_draw_counters_reset_between_invocations() -> None:
    """Otherwise a run stops being reproducible."""

    def first_arg() -> np.ndarray:
        for call in _specs.invocations("mse_loss", "moderate", "F.mse_loss", None):
            return np.asarray(_probe.to_numpy(call.args[0]))
        raise AssertionError("no invocation")

    assert np.array_equal(first_arg(), first_arg())


def test_a_regression_loss_is_not_probed_at_its_own_minimum() -> None:
    """``mse_loss(x, x)`` is zero with a zero gradient — nothing to check."""
    symbol = _by_name("F.mse_loss")
    assert symbol is not None
    call = _chosen_call(symbol)
    assert call is not None
    a = np.asarray(_probe.to_numpy(call.args[0]))
    b = np.asarray(_probe.to_numpy(call.args[1]))
    assert not np.array_equal(a, b)


# ── the stochastic flag ───────────────────────────────────────────────────────


def test_deterministic_symbols_are_not_flagged_stochastic() -> None:
    """A resampler is not a sampler."""
    flagged = {s.qualname for s in _surface.enumerate_surface() if "stochastic" in s.flags}
    for name in (
        "F.normalize",
        "F.grid_sample",
        "F.poisson_nll_loss",
        "nn.Upsample",
        "lucid.metal.empty_cache",
        "lucid.utils.transforms.functional.normalize",
        "lucid.utils.transforms.functional.sample_field_at_points",
    ):
        assert name not in flagged, name


def test_the_samplers_are_still_flagged() -> None:
    flagged = {s.qualname for s in _surface.enumerate_surface() if "stochastic" in s.flags}
    for name in (
        "lucid.rand",
        "lucid.randn",
        "lucid.randint",
        "lucid.randperm",
        "lucid.normal",
        "lucid.bernoulli",
        "F.dropout",
        "F.dropout2d",
        "F.alpha_dropout",
        "F.gumbel_softmax",
        "F.rrelu",
        "lucid.nn.init.kaiming_normal",
        "lucid.nn.init.xavier_uniform",
    ):
        assert name in flagged, name


def test_empty_is_not_asked_to_respect_a_seed() -> None:
    """It returns uninitialised memory, not a draw."""
    flagged = {s.qualname for s in _surface.enumerate_surface() if "stochastic" in s.flags}
    for name in ("lucid.empty", "lucid.empty_like", "Tensor.new_empty"):
        assert name not in flagged, name


# ── rrelu draws only in training mode ─────────────────────────────────────────


def test_rrelu_is_probed_where_it_draws() -> None:
    # The default first, for the axes that measure the op; its
    # ``training=True`` twin next, for the determinism axis, which moves on
    # from a call that does not draw (LCD-561).
    symbol = _by_name("F.rrelu")
    assert symbol is not None
    call = _chosen_call(symbol)
    assert call is not None and "training" not in call.kwargs
    fn = _surface.resolve(symbol)
    calls = list(_specs.invocations(symbol.short, "moderate", symbol.qualname, fn))
    at = next(i for i, c in enumerate(calls) if c.note == call.note)
    assert calls[at + 1].kwargs.get("training") is True


def test_rrelu_actually_draws_under_that_flag() -> None:
    """Guard the guard — if the default ever became ``True`` this test
    would keep passing for the wrong reason, so check both."""
    import lucid.nn.functional as F

    x = lucid.tensor(np.array([[-1.0, 2.0, -3.0, 4.0]]))
    lucid.manual_seed(1)
    a = np.asarray(F.rrelu(x, training=True).numpy())
    lucid.manual_seed(2)
    b = np.asarray(F.rrelu(x, training=True).numpy())
    assert not np.array_equal(a, b)

    lucid.manual_seed(1)
    again = np.asarray(F.rrelu(x, training=True).numpy())
    assert np.array_equal(a, again)  # and reproduces

    lucid.manual_seed(1)
    c = np.asarray(F.rrelu(x).numpy())
    lucid.manual_seed(2)
    d = np.asarray(F.rrelu(x).numpy())
    assert np.array_equal(c, d)  # the default is the expectation


# ── every module class is constructible ───────────────────────────────────────


def test_every_nn_module_class_can_be_built() -> None:
    """A class the audit cannot construct is a class it cannot notice a
    regression in.

    It could not construct 114 of the 221, because the ladder tried eight
    fixed argument tuples and reported everything it missed as "no
    constructor signature worked".  Those contributed nothing to any
    axis.  In a subprocess: building every layer in the framework leaves
    global state behind.
    """
    code = """
from lucid.test.audit import _surface
from lucid.test.audit._axes import _try_construct

failed = []
total = 0
for symbol in _surface.enumerate_surface():
    if symbol.kind != "module":
        continue
    total += 1
    try:
        built = _try_construct(symbol.obj)
    except Exception:
        built = None
    if built is None:
        failed.append(symbol.qualname)
print("%d/%d" % (total - len(failed), total))
for name in failed:
    print(name)
"""
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    built, total = lines[0].split("/")
    assert built == total, f"{built}/{total} built; missing: {lines[1:]}"


def test_a_quantisation_layer_gets_the_qconfig_it_demands() -> None:
    """``qat.Linear`` declares ``qconfig=None`` and then refuses it.  A
    signature-derived constructor honours the author's default, so it
    never supplied one and the whole qat stack stayed unaudited."""
    from lucid.test.audit._axes import _try_construct

    import lucid.nn.qat as qat

    module = _try_construct(qat.Linear)
    assert module is not None
    assert getattr(module, "qconfig", None) is not None


def test_probe_factories_keep_the_callers_shape() -> None:
    """LCD-391: every ``as_*`` turned a 0-d array into shape ``(1,)`` —
    ``np.ascontiguousarray`` adds a dimension, and so does the converter
    given ``dtype=`` / ``device=`` (LCD-399)."""
    factories = (
        _probe.as_f32,
        _probe.as_f64,
        _probe.as_int,
        _probe.as_complex,
        _probe.as_complex64,
    )
    for factory in factories:
        for shape in ((), (1,), (2, 3)):
            got = factory(np.full(shape, 2.0))
            assert tuple(got.shape) == shape, (factory.__name__, shape)
    assert tuple(_probe.as_f32(np.array(2.0), device="metal").shape) == ()
    assert tuple(_probe.as_int(np.array(2), device="metal").shape) == ()
    assert tuple(_probe.tensor_of(np.array(3), dtype=lucid.int32).shape) == ()
    assert tuple(_probe.tensor_of(np.array(3.0), lucid.float32, "metal").shape) == ()
