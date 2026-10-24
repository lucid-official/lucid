"""Numerical stability, memory layout and reproducibility.

These are the axes that do not ask "is the formula right" but "does the
implementation survive the inputs a real run produces".  Each has a
precedent in this framework:

* ``log_softmax`` once computed ``log(softmax(x))`` and returned ``-inf``
  past ``|logit| ~ 90``, because the non-maximum probabilities round to
  zero before the log sees them.  Found while training a network, not by
  a unit test, because every unit test used well-conditioned logits.
* the transforms and ``pdist`` bugs were layout bugs: an index tensor
  built one way worked and another way did not.
* a sampler that is not reproducible under a fixed seed makes every
  other check in this tool unfalsifiable.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

import functools
import json
import pathlib

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Axis, Context
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from lucid.test.audit._surface import Symbol


@functools.lru_cache(maxsize=1)
def _stability_contract() -> dict[str, list[str]]:
    """Where the reference answers finitely, per symbol.

    Checked in, so the audit reads it without the reference installed.
    Regenerate with
    ``python -m lucid.test.audit.tools.stability_contract``.
    """
    path = pathlib.Path(__file__).with_name("stability_contract.json")
    try:
        data = json.loads(path.read_text())
    except OSError, ValueError:
        return {}
    table = data.get("finite_at")
    return table if isinstance(table, dict) else {}


class StabilityAxis(Axis):
    """A finite, in-domain input must not produce a NaN.

    Overflow to infinity is allowed — ``exp(1e8)`` is genuinely infinite
    — but a NaN out of finite input means an intermediate cancelled,
    divided by its own zero, or overflowed into a subtraction.  That is
    the shape of every stability bug worth finding.
    """

    name = "stability"
    summary = "finite in-domain inputs across 14 orders of magnitude"
    kinds = frozenset({"op", "method"})

    _SCALES = (1e-30, 1e-15, 1e-6, 1.0, 1e6, 1e15, 1e30)

    def applies(self, symbol: Symbol) -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        try:
            base = np.abs(call.base) + 0.25
        except TypeError:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the varied argument is not a tensor — nothing to vary",
            )
        # A scalar operand is not necessarily a *value*.  ``eye(n)`` takes
        # a size, and multiplying it by 1e15 asks for a matrix of 1e30
        # elements — the process was killed by the OOM killer, signal 9,
        # and the sweep stopped there.  Scaling only means something for an
        # operand that carries data, so a bare scalar is skipped rather
        # than magnified.
        if base.ndim == 0 or base.size == 1:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "primary is a scalar — scaling it would change a size, not a value",
            )

        # Where the reference is finite.  Absent means "not measured", and
        # an unmeasured op is checked at every scale as before — the table
        # narrows the claim, it does not gate it.
        reference_finite = _stability_contract().get(
            _surface.recorded_as(symbol.qualname)
        )

        # What the check is actually about: does *changing the magnitude*
        # introduce a NaN?  A NaN that was there before anything was scaled
        # belongs to the probe, not to the op's conditioning.
        #
        # Only the primary argument is made positive.  Every companion
        # keeps whatever the domain drew, and ``moderate`` draws from
        # (-1.05, 1.35) — so ``xlogy(x, y)`` met a negative ``y``, answered
        # NaN for it at every scale including 1.0, and was reported for
        # arithmetic that was never in question.  Five findings, all of
        # them the second operand.
        #
        # Held per element rather than per array, so an op that is NaN
        # somewhere for that reason can still be caught going NaN
        # elsewhere under scaling.
        baseline_finite = None
        try:
            at_one = _probe.to_numpy(fn(*call.with_primary(base).args, **call.kwargs))
            if at_one is not None and at_one.dtype.kind in "fc":
                baseline_finite = ~np.isnan(at_one)
        except Exception:  # noqa: BLE001 - surveying, not asserting
            baseline_finite = None

        broken: list[str] = []
        domain: list[str] = []
        ran = 0
        for scale in self._SCALES:
            probe = base * scale
            if not np.isfinite(probe).all():
                continue
            # ``in-domain`` is the load-bearing word in this check, and the
            # axis had no way to know where the domain was: it reported
            # ``acos`` for answering NaN at 1e+6, where NaN is the only
            # answer the function has.  A NaN the reference also produces
            # is the domain speaking, not the implementation.
            if reference_finite is not None and f"{scale:g}" not in reference_finite:
                domain.append(f"1e{int(np.log10(scale)):+d}")
                continue
            try:
                out = _probe.to_numpy(fn(*call.with_primary(probe).args, **call.kwargs))
            except Exception:  # noqa: BLE001 - a refusal is not instability
                continue
            if out is None or out.dtype.kind not in "fc":
                continue
            ran += 1
            appeared = np.isnan(out)
            if baseline_finite is not None and baseline_finite.shape == appeared.shape:
                appeared = appeared & baseline_finite
            if appeared.any():
                broken.append(f"scale 1e{int(np.log10(scale)):+d} -> NaN")
        if not ran:
            if domain:
                # Every scale left the function's domain.  ``acosh`` on a
                # probe below 1 is the whole check falling outside where
                # the op is defined — nothing was asked, so nothing passed.
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "every scale is outside the domain: " + ", ".join(domain[:4]),
                )
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the output is not float — a scale sweep has nothing to measure",
            )
        if broken:
            return self._finding(
                symbol,
                Status.FAIL,
                "finite positive input produced NaN: " + ", ".join(broken[:4]),
                scales=broken,
            )
        note = f"{ran}/{len(self._SCALES)} scales finite"
        if domain:
            note += f" ({len(domain)} outside the domain)"
        return self._finding(symbol, Status.PASS, note)


class ExtremeValueAxis(Axis):
    """The specific limits where a naive formula is known to break.

    Generic scaling cannot express "softmax of a thousand still sums to
    one"; these are hand-written because each one is a property with a
    known answer, and each is a bug this class of code actually has.
    """

    name = "extreme"
    summary = "known-hard limits: saturated softmax, tiny log1p, huge logsumexp"
    kinds = frozenset({"op"})

    def applies(self, symbol: Symbol) -> bool:
        return symbol.short in _EXTREME_CASES and symbol.inert

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        failures: list[str] = []
        checked = 0
        for label, build, verify in _EXTREME_CASES[symbol.short]:
            try:
                out = fn(*build())
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{label}: raised {type(exc).__name__}")
                continue
            got = _probe.to_numpy(out)
            if got is None:
                failures.append(f"{label}: no tensor")
                continue
            checked += 1
            problem = verify(np.asarray(got, dtype=np.float64))
            if problem:
                failures.append(f"{label}: {problem}")
        if failures:
            return self._finding(
                symbol, Status.FAIL, "; ".join(failures[:3]), failures=failures
            )
        if not checked:
            return self._finding(symbol, Status.SKIP, "no case ran")
        return self._finding(symbol, Status.PASS, f"{checked} extreme cases")


def _f64(values: Any) -> Any:
    return _probe.as_f64(np.asarray(values, dtype=np.float64))


def _finite(name: str) -> Any:
    def check(out: np.ndarray) -> str:
        if not np.isfinite(out).all():
            return f"{name} is not finite: {out.reshape(-1)[:4]}"
        return ""

    return check


def _close_to(want: float, tol: float = 1e-6) -> Any:
    def check(out: np.ndarray) -> str:
        got = float(np.asarray(out).reshape(-1)[0])
        if not np.isfinite(got) or abs(got - want) > tol * max(abs(want), 1.0):
            return f"expected ~{want}, got {got}"
        return ""

    return check


def _sums_to_one(out: np.ndarray) -> str:
    total = float(np.sum(out))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
        return f"probabilities sum to {total}"
    return ""


#: ``name -> [(label, build_args, verify)]``.  ``build_args`` returns the
#: positional arguments; ``verify`` returns "" for pass or a reason.
_EXTREME_CASES: dict[str, list[tuple[str, Any, Any]]] = {
    "softmax": [
        ("logits +1000", lambda: (_f64([1000.0, 999.0, 998.0]),), _sums_to_one),
        ("logits -1000", lambda: (_f64([-1000.0, -999.0, -998.0]),), _sums_to_one),
        ("wide spread", lambda: (_f64([-800.0, 0.0, 800.0]),), _sums_to_one),
    ],
    "log_softmax": [
        (
            "logits +1000",
            lambda: (_f64([1000.0, 999.0, 998.0]),),
            _finite("log-probabilities"),
        ),
        (
            "wide spread",
            lambda: (_f64([-800.0, 0.0, 800.0]),),
            _finite("log-probabilities"),
        ),
    ],
    "logsumexp": [
        (
            "all +1000",
            lambda: (_f64([1000.0, 1000.0]),),
            _close_to(1000.0 + float(np.log(2.0))),
        ),
        (
            "all -1000",
            lambda: (_f64([-1000.0, -1000.0]),),
            _close_to(-1000.0 + float(np.log(2.0))),
        ),
    ],
    "sigmoid": [
        ("+1000", lambda: (_f64([1000.0]),), _close_to(1.0)),
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(0.0)),
    ],
    "tanh": [
        ("+1000", lambda: (_f64([1000.0]),), _close_to(1.0)),
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(-1.0)),
    ],
    "softplus": [
        ("+1000 must not overflow", lambda: (_f64([1000.0]),), _close_to(1000.0, 1e-9)),
        (
            "-1000 must not underflow to -inf",
            lambda: (_f64([-1000.0]),),
            _finite("softplus"),
        ),
    ],
    "log1p": [
        (
            "1e-16 must not round to zero",
            lambda: (_f64([1e-16]),),
            _close_to(1e-16, 1e-3),
        ),
    ],
    "expm1": [
        (
            "1e-16 must not round to zero",
            lambda: (_f64([1e-16]),),
            _close_to(1e-16, 1e-3),
        ),
    ],
    "logaddexp": [
        (
            "both +1000",
            lambda: (_f64([1000.0]), _f64([1000.0])),
            _close_to(1000.0 + float(np.log(2.0))),
        ),
    ],
    "logsigmoid": [
        ("-1000", lambda: (_f64([-1000.0]),), _close_to(-1000.0, 1e-9)),
        ("+1000", lambda: (_f64([1000.0]),), _finite("logsigmoid")),
    ],
    "norm": [
        ("1e200 must not overflow", lambda: (_f64([1e200, 1e200]),), _finite("norm")),
        (
            "1e-200 must not underflow",
            lambda: (_f64([1e-200, 1e-200]),),
            _finite("norm"),
        ),
    ],
    "hypot": [
        (
            "1e200 must not overflow",
            lambda: (_f64([1e200]), _f64([1e200])),
            _finite("hypot"),
        ),
    ],
    "var": [
        (
            "large mean, small spread",
            lambda: (_f64([1e8, 1e8 + 1.0, 1e8 + 2.0]),),
            _close_to(1.0, 1e-3),
        ),
    ],
    "std": [
        (
            "large mean, small spread",
            lambda: (_f64([1e8, 1e8 + 1.0, 1e8 + 2.0]),),
            _close_to(1.0, 1e-3),
        ),
    ],
}


class ContiguityAxis(Axis):
    """A non-contiguous input must give the same answer as a packed one.

    Every layout bug this framework has had looked like this: the same
    values, reached through different strides, produced different
    results.  The probe is a strided view rather than a transpose so the
    logical shape is unchanged and only the memory walk differs.
    """

    name = "layout"
    summary = "strided (non-contiguous) inputs agree with packed ones"
    kinds = frozenset({"op", "method"})

    def applies(self, symbol: Symbol) -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    # These answer with the layout itself, or read the storage through it,
    # so a strided view answers differently by design.
    _REPORTS_LAYOUT = frozenset(
        {
            "Tensor.is_contiguous",
            "Tensor.stride",
            "Tensor.storage_offset",
            "Tensor.data_ptr",
            "Tensor.untyped_storage",
            "Tensor.as_strided",
            "lucid.as_strided",
        }
    )

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        if symbol.qualname in self._REPORTS_LAYOUT:
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "reports the layout itself"
            )
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )
        original = (
            call.args[call.primary] if 0 <= call.primary < len(call.args) else None
        )
        base = _probe.to_numpy(original)
        if base is None:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the varied argument is not a tensor — nothing to vary",
            )
        if base.ndim < 1:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "0-d input has no layout to vary",
            )

        # A tensor twice the size in the last axis, whose every second
        # column holds the probe.  Slicing it back gives the same values
        # through a different stride.
        # Numeric differentiation deliberately projects onto real float64;
        # layout must instead preserve every value and the operand's dtype.
        base = np.array(base, copy=True)
        if symbol.qualname == "lucid.view_as_complex" and base.ndim >= 2:
            padded_shape = (*base.shape[:-2], base.shape[-2] * 2 + 1, base.shape[-1])
            padded = np.full(padded_shape, 0, dtype=base.dtype)
            padded[..., 1::2, :] = base
        else:
            padded_shape = (*base.shape[:-1], base.shape[-1] * 2 + 1)
            padded = np.full(padded_shape, 0, dtype=base.dtype)
            padded[..., 1::2] = base
        try:
            if not isinstance(original, lucid.Tensor):
                original = lucid.tensor(base)
            packed_probe = lucid.tensor(
                base, dtype=original.dtype, device=original.device
            )
            big = lucid.tensor(padded, dtype=original.dtype, device=original.device)
            # A positive-step slice is a view on either device: every
            # second element of ``big``'s buffer, so an op that assumes a
            # dense buffer reads the sentinels instead of the probe.  On
            # Metal the slice is a strided member of ``big``'s view family
            # (730ef06c5), not a packed copy.
            if symbol.qualname == "lucid.view_as_complex" and base.ndim >= 2:
                view = big[..., 1:, :][..., ::2, :]
            else:
                view = big[..., 1:][..., ::2]
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.SKIP, f"could not build a view: {exc!r}"
            )

        # Before the operands are compared, the view has to *be* the probe.
        # Every second column of ``padded`` is a sentinel, so a slice
        # that materialises the wrong elements picks the sentinel up — and
        # the comparison below would then be measuring an op against
        # different numbers while calling the difference a layout effect.
        #
        # This is the half the ``layout`` mutant breaks: a slice that takes
        # the wrong stride fails here, on every device.
        seen = _probe.to_numpy(view)
        if seen is None or tuple(seen.shape) != tuple(base.shape):
            return self._finding(symbol, Status.SKIP, "the view is not readable")
        if not np.allclose(seen, base, rtol=1e-5, atol=1e-6, equal_nan=True):
            return self._finding(
                symbol,
                Status.FAIL,
                "big[..., 1:][..., ::2] does not hold the probe — a strided slice "
                f"materialised the wrong values (max diff "
                f"{np.nanmax(np.abs(np.asarray(seen) - base)):.3e})",
            )
        # ``seen`` reads ``big``'s buffer.  While it lives, an in-place op
        # through the view is refused — a NumPy array holding the storage
        # would not see the write — which says nothing about the layout.
        del seen

        try:
            args = list(call.args)
            args[call.primary] = packed_probe
            packed = _probe.to_numpy(fn(*args, **call.kwargs))
            args[call.primary] = view
            strided = _probe.to_numpy(fn(*args, **call.kwargs))
        except Exception as exc:  # noqa: BLE001
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:60]}", call)
        if packed is None or strided is None or packed.shape != strided.shape:
            return self._finding(symbol, Status.SKIP, "outputs not comparable")
        agree = (
            np.array_equal(packed, strided)
            if packed.dtype.kind in "biu"
            else np.allclose(packed, strided, rtol=1e-9, atol=1e-12, equal_nan=True)
        )
        if not agree:
            return self._finding(
                symbol,
                Status.FAIL,
                "a strided view of the same values gave a different answer "
                "(including imaginary components and exact integer values)",
            )

        # They agreed — but could they have disagreed?  Only if the op was
        # handed a strided layout.  The engine's layout says what the op
        # was handed; a slice that came back packed is read as a fresh
        # tensor, so the comparison is between an op and itself.  Since
        # Metal slices became view-family members (730ef06c5) no device
        # does that, but if one ever does again the sentinel check above is
        # the half that earned the verdict, and the detail says so.
        if bool(view._impl.is_contiguous()) and list(view._impl.stride) == list(
            big[..., : base.shape[-1]]._impl.stride
        ):
            return self._finding(
                symbol,
                Status.PASS,
                "the strided slice materialised the right values; the layout "
                "half is vacuous here — the slice came back packed",
            )
        return self._finding(symbol, Status.PASS, "strided and packed agree")


class DeterminismAxis(Axis):
    """The same seed must give the same numbers.

    Only meaningful for the ops every other axis has to skip — the ones
    that draw.  Without this they would be entirely unchecked, and a
    sampler that ignores the seed makes every seeded test in the
    repository meaningless rather than merely wrong.
    """

    name = "determinism"
    summary = "stochastic ops reproduce under a fixed seed"
    kinds = frozenset({"op", "method"})

    def applies(self, symbol: Symbol) -> bool:
        if symbol.kind not in self.kinds or not symbol.inert:
            return False
        return "stochastic" in symbol.flags

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")

        first: np.ndarray | None = None
        quiet = False
        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                draws = []
                # Only where the primary is a tensor.  ``empty_like`` takes a
                # *shape*, and ``to_numpy`` reads a tuple of ints as an array
                # happily — rebuilding it handed the op a tensor where it
                # wanted a size.
                primary = call.args[call.primary] if call.args else None
                if not hasattr(primary, "dtype") or not hasattr(primary, "shape"):
                    base = None
                else:
                    try:
                        base = call.base
                    except TypeError:
                        base = None
                for _ in range(2):
                    lucid.manual_seed(1234)
                    # Rebuilt between draws, not reused.  An in-place op
                    # writes into its own input, so the second call reads
                    # what the first one left — ``rrelu_`` was compared
                    # against itself applied twice and reported as
                    # non-reproducible for being perfectly reproducible.
                    probe = call if base is None else call.with_primary(base)
                    try:
                        got = _probe.to_numpy(fn(*probe.args, **probe.kwargs))
                    except Exception:  # noqa: BLE001
                        draws = []
                        break
                    if got is None:
                        draws = []
                        break
                    draws.append(np.asarray(got, dtype=np.float64))
                if len(draws) == 2:
                    first = draws[0]
                    if not np.allclose(draws[0], draws[1], equal_nan=True):
                        return self._finding(
                            symbol,
                            Status.FAIL,
                            "two draws under the same seed differ by "
                            f"{np.nanmax(np.abs(draws[0] - draws[1])):.3e}",
                        )
                    # Guard the instrument: an op that ignores its inputs
                    # and returns a constant would pass trivially.
                    lucid.manual_seed(4321)
                    other = _probe.to_numpy(fn(*call.args, **call.kwargs))
                    if other is not None and np.allclose(
                        first, np.asarray(other, dtype=np.float64), equal_nan=True
                    ):
                        # This call does not draw; a later one may.  The
                        # default of ``alpha_dropout`` is ``training=False``,
                        # the identity, and its spec offers ``training=True``
                        # next (``_specs.invocations``).  For every op, not
                        # only those: moving on can turn VACUOUS into PASS or
                        # FAIL, and every later call still faces the
                        # same-seed check, so it is never more lenient.
                        quiet = True
                        continue
                    return self._finding(
                        symbol, Status.PASS, "reproducible and seed-sensitive"
                    )
        if quiet:
            return self._finding(
                symbol,
                Status.VACUOUS,
                "a different seed gave identical numbers — this op may not draw",
            )
        return self._finding(symbol, Status.SKIP, "no candidate invocation ran")


class RankAxis(Axis):
    """A rank the op cannot use must be refused, not answered.

    Every crash this suite has had came from the same place, and neither
    was reported as a crash the first time it happened:

        householder_product(H(4,), tau(4,))  ->  Q(0, 0)
        lstsq(A(4,), B(4,))                  ->  1.8e19 bytes requested
        lstsq(A(6,3), B(2,2))                ->  (3, 2), six rows read out of two

    The C++ backends take ``m`` from ``shape[shape.size() - 2]``.  That
    subtraction is unsigned, so a rank-1 input wraps it, reads past the
    shape, and the value it finds becomes the size every buffer in the
    function is allocated and copied at.  When the resulting over-read
    lands on mapped memory the op **returns a plausible answer**; when it
    does not, the process dies.  Thirty-six sites index a shape that way.

    So this axis degrades the rank of a working call's primary tensor and
    asks for one of two outcomes:

    * **refused** — an exception, which is the correct answer and by far
      the most common one; or
    * **answered consistently** — a result whose element count is not
      absurd for the input it was given.

    What it will not accept is the third outcome: an answer that is
    *empty or degenerate* while the input was not.  That is the shape of
    a size computed from a wrapped index, and it is the half of this bug
    that does not announce itself.

    A crash is not caught here — it takes the process with it — but the
    gate reports a stage that died as a stage that died, so the sweep
    ending inside this axis is itself the signal.
    """

    name = "rank"
    summary = "an unusable rank is refused, not answered with a degenerate result"
    kinds = frozenset({"op", "method"})

    def applies(self, symbol: Symbol) -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the varied argument is not a tensor — no rank to reduce",
            )
        if base.ndim < 2:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"already rank {base.ndim} — nothing to take away",
            )

        # Flatten to rank 1, keeping every element.  A reduced *shape*
        # rather than a reduced *size* is the point: the values are all
        # still there, so an op that legitimately accepts rank 1 can go
        # on computing, and only the shape arithmetic changes.
        flat = base.reshape(-1)
        try:
            args = list(call.with_primary(flat).args)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.SKIP, f"could not rebuild the call: {exc!r}"
            )

        try:
            out = fn(*args, **call.kwargs)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.PASS,
                f"rank {base.ndim} -> 1 refused: {type(exc).__name__}",
            )

        got = _probe.to_numpy(out)
        if got is None:
            return self._finding(
                symbol, Status.PASS, "answered with something not readable as an array"
            )
        if got.size == 0 and flat.size != 0:
            return self._finding(
                symbol,
                Status.FAIL,
                f"rank {base.ndim} -> 1 was accepted and returned an empty "
                f"{tuple(got.shape)} from a {flat.size}-element input — the "
                f"size was computed from a shape index that wrapped",
            )
        return self._finding(
            symbol, Status.PASS, f"rank {base.ndim} -> 1 answered {tuple(got.shape)}"
        )


STABILITY_AXES: tuple[Axis, ...] = (
    StabilityAxis(),
    ExtremeValueAxis(),
    ContiguityAxis(),
    RankAxis(),
    DeterminismAxis(),
)

__all__ = [
    "STABILITY_AXES",
    "ContiguityAxis",
    "RankAxis",
    "DeterminismAxis",
    "ExtremeValueAxis",
    "StabilityAxis",
]
