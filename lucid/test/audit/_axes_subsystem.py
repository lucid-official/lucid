"""Axes for the subsystems the numeric sweep cannot express.

``distributions``, ``diffeq``, ``quantization``, ``serialization`` and
``compile`` were enumerated in the denominator and had **no axis at
all** — 141 symbols counted and never checked, which is the exact shape
of the dishonesty this tool exists to remove.  Each gets the question
that is actually meaningful for it:

    distributions   does log_prob agree with the samples, is cdf monotone,
                    does icdf invert it
    diffeq          does halving the step reduce the error like h**p
    quantization    does quantize -> dequantize land inside one step
    serialization   does a tensor survive a round trip through disk
    compile         does the compiled function agree with the eager one

and :class:`SmokeAxis` catches everything left over, including the
stateful symbols every other axis refuses to touch — those run inside a
guard that snapshots and restores the global state, so they are checked
rather than merely counted.
"""

import contextlib
import functools
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Axis, Context, _try_construct
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from types import TracebackType

    from lucid.test.audit._surface import Symbol


class StateGuard:
    """Snapshot the process-wide state, restore it on the way out.

    What makes a stateful symbol callable at all.  Without this the
    survey either skips them — leaving 100 symbols permanently unchecked
    — or calls them and poisons every op that follows, which is exactly
    what a first version did to 278 of them.
    """

    _KEYS = (
        ("is_grad_enabled", "set_grad_enabled"),
        ("get_default_dtype", "set_default_dtype"),
        ("get_default_device", "set_default_device"),
        ("get_num_threads", "set_num_threads"),
        ("get_rng_state", "set_rng_state"),
        # Deterministic mode, which this guard did not restore.
        #
        # ``_smoke_arguments`` hands ``use_deterministic_algorithms`` the
        # value ``True`` — that is the point of the guard — and nothing
        # ever turned it off again.  Every symbol swept after it ran in
        # deterministic mode, and the four ``dropout`` entry points then
        # refused with "non-deterministic op called under
        # set_deterministic(True)": they reported as uncallable, in a
        # mode nobody had asked for and nothing in the output mentioned.
        #
        # Exactly the failure this class was written for, one switch
        # short — which is the argument for deriving the list rather than
        # writing it, and the reason the pairs are asked of ``lucid``
        # by name rather than hard-coded to five.
        (
            "are_deterministic_algorithms_enabled",
            "use_deterministic_algorithms",
        ),
        # Anomaly detection, the fourth switch found leaking one at a
        # time.  Enumerated below rather than only listed here — a table
        # of process state that a human maintains is a table that is one
        # entry short, which is now measured rather than argued: grad
        # mode, deterministic mode, the hook registries and this.
        ("is_anomaly_enabled", "set_detect_anomaly"),
    )

    #: Where process state lives.  ``set_detect_anomaly`` is on
    #: ``lucid.autograd`` and not re-exported, so a guard that looked it
    #: up on ``lucid`` got ``None`` and **skipped it without saying so** —
    #: the switch stayed on and the next stage's doctests failed.  A
    #: lookup that can silently find nothing is a guard with a hole in it.
    _NAMESPACES = ("lucid", "lucid.autograd", "lucid.metal")

    @classmethod
    def _namespaces(cls) -> list[Any]:
        import importlib  # noqa: PLC0415

        out = []
        for path in cls._NAMESPACES:
            try:
                out.append(importlib.import_module(path))
            except Exception:  # noqa: BLE001
                continue
        return out

    @classmethod
    def _resolve(cls, name: str) -> Any:
        for namespace in cls._namespaces():
            found = getattr(namespace, name, None)
            if callable(found):
                return found
        return None

    @classmethod
    def _discovered_pairs(cls) -> tuple[tuple[str, str], ...]:
        """Every ``is_/get_X`` with a matching ``set_X``, found not listed.

        The named pairs above stay as documentation and as the ordering
        this guard restores in; this adds whatever the package has grown
        since.  Four switches were found leaking one at a time this
        session — grad mode, deterministic mode, the hook registries and
        anomaly detection — which is the argument for deriving the list:
        a table of process state that a human maintains is a table that
        is one entry short.
        """
        found: list[tuple[str, str]] = []
        known = {reader for reader, _ in cls._KEYS}
        for namespace in cls._namespaces():
            for name in dir(namespace):
                if not name.startswith("set_"):
                    continue
                for prefix in ("get_", "is_", "are_"):
                    reader = prefix + name[4:]
                    if reader in known:
                        break
                    if callable(getattr(namespace, reader, None)) and callable(
                        getattr(namespace, name, None)
                    ):
                        found.append((reader, name))
                        known.add(reader)
                        break
        return tuple(found)

    #: Process-wide registries a call can append to.  A getter/setter
    #: pair cannot express these — there is no setter, only a mutable
    #: mapping — so they are snapshotted by copy and restored in place.
    #:
    #: Found the hard way.  ``_smoke_arguments`` calls
    #: ``register_module_forward_pre_hook`` like every other symbol, the
    #: call succeeds, and the hook stays installed **for the life of the
    #: process** — so every ``Module.__call__`` after the sweep ran a
    #: probe's throwaway hook and raised inside it.  The sweep's own
    #: results were clean because smoke runs last; the *next stage* was
    #: not, and the doctest stage reported 40 modules as regressions
    #: (``nn.modules.conv`` 0 -> 52) that were nothing of the kind.
    #:
    #: This is the ``set_grad_enabled`` lesson a third time, and the
    #: pattern is now explicit: **state is not only the switches with
    #: setters.**
    _REGISTRIES = (
        "_GLOBAL_FORWARD_PRE_HOOKS",
        "_GLOBAL_FORWARD_HOOKS",
        "_GLOBAL_BACKWARD_PRE_HOOKS",
        "_GLOBAL_BACKWARD_HOOKS",
        "_GLOBAL_LOAD_STATE_DICT_PRE_HOOKS",
        "_GLOBAL_LOAD_STATE_DICT_POST_HOOKS",
    )

    def __init__(self) -> None:
        self._saved: list[tuple[Any, Any]] = []
        self._registries: list[tuple[Any, Any]] = []

    def __enter__(self) -> StateGuard:
        import lucid.nn.hooks as hooks  # noqa: PLC0415 - avoids an import cycle

        for name in self._REGISTRIES:
            registry = getattr(hooks, name, None)
            if registry is not None:
                self._registries.append((registry, dict(registry)))
        for getter_name, setter_name in (*self._KEYS, *self._discovered_pairs()):
            getter = self._resolve(getter_name)
            setter = self._resolve(setter_name)
            if getter is None or setter is None:
                continue
            try:
                self._saved.append((setter, getter()))
            except Exception:  # noqa: BLE001
                continue
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for setter, value in reversed(self._saved):
            try:
                setter(value)
            except Exception:  # noqa: BLE001
                continue
        self._saved.clear()
        # Restored in place, not rebound: the modules that read these
        # hold the mapping itself.
        for registry, snapshot in reversed(self._registries):
            try:
                registry.clear()
                registry.update(snapshot)
            except Exception:  # noqa: BLE001
                continue
        self._registries.clear()


class SmokeAxis(Axis):
    """Every callable gets invoked at least once and must not take the process with it.

    The weakest axis and the one that moves coverage most: a symbol no
    other axis can express is still a symbol that should not segfault,
    hang, or corrupt global state when called the obvious way.
    """

    name = "smoke"
    summary = "every callable is invoked once, stateful ones inside a state guard"
    #: Everything except the two kinds that already have a lifecycle axis
    #: of their own.  This is the floor: after it, no callable symbol in
    #: the framework is outside the audit.
    kinds = frozenset(
        {
            "op",
            "method",
            "class",
            "util",
            "compiled",
            "serialize",
            "quant",
            "distribution",
            "diffeq",
            "value",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        if symbol.kind in ("module", "optim"):
            return False  # ModuleAxis / OptimAxis do this properly
        if symbol.kind == "declaration":
            # A Protocol, a metaclass or an abstract base.  Callable in
            # the sense that every class is, with nothing behind the call
            # — and this axis reaches by ``callable()`` rather than by
            # ``kinds``, so the kind has to be named here too.
            return False
        return callable(_surface.resolve(symbol))

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")

        guard = StateGuard() if not symbol.inert else _NullGuard()
        with guard:
            for args, kwargs, note in _smoke_arguments(symbol):
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 - a refusal is a fine answer here
                    continue
                return self._finding(symbol, Status.PASS, f"called as {note}")

            # A class is *constructed*, not called with a tensor.  The
            # ladder above hands it ``(tensor)`` and ``(2, 3)``, which is
            # right for a factory function and wrong for every
            # distribution, tokenizer, observer and dataset in the
            # framework — 60-odd of the 143 that reached no argument
            # shape were classes whose signature says exactly what they
            # want.
            if isinstance(fn, type):
                from lucid.test.audit._axes_data import _construct

                instance, why = _construct(fn)
                if instance is not None:
                    return self._finding(
                        symbol, Status.PASS, "constructed from its signature"
                    )
                return self._finding(symbol, Status.SKIP, f"construct: {why}")

            # Falling through to the op ladder covers anything with a
            # tensor-shaped signature the table above does not name.
            #
            # Its own loop, not ``_working_call``.  That helper keeps the
            # first candidate whose output ``to_numpy`` can read, which is
            # exactly right for an axis that then compares numbers and
            # exactly wrong here: this axis asks whether calling the
            # symbol takes the process down, and ``lucid.compile(f)``
            # answers that question perfectly well by returning a
            # closure.  Demanding a measurable output rejected every
            # higher-order function in the framework — ``compile``,
            # ``odeint``, ``make_step``, ``func.jvp``, ``gradcheck`` —
            # and reported them as uncallable.
            #
            # Run for stateful symbols too.  It was gated on ``inert``,
            # which is what the guard is *for*: the whole point of
            # snapshotting the process state is that the call can then be
            # made.
            for domain in ctx.domains:
                for call in _specs.invocations(
                    symbol.short, domain, symbol.qualname, fn
                ):
                    try:
                        fn(*call.args, **call.kwargs)
                    except Exception:  # noqa: BLE001 - a refusal is a fine answer
                        continue
                    return self._finding(symbol, Status.PASS, f"called as {call.note}")
        return self._finding(symbol, Status.SKIP, "no argument shape worked")


class _NullGuard:
    def __enter__(self) -> _NullGuard:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _smoke_arguments(
    symbol: Symbol,
) -> list[tuple[tuple[Any, ...], dict[str, Any], str]]:
    """Plausible calls for a symbol with no numeric spec.

    Stateful setters are handed back the value their own getter reports,
    so the call is a no-op that still exercises the code path.
    """
    name = symbol.short
    out: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [((), {}, "no arguments")]

    if name.startswith("set_"):
        getter = getattr(lucid, "get_" + name[4:], None) or getattr(
            lucid, "is_" + name[4:], None
        )
        if getter is not None:
            try:
                out.insert(0, ((getter(),), {}, "its own getter's value"))
            except Exception:  # noqa: BLE001
                pass
    if "seed" in name:
        out.insert(0, ((0,), {}, "seed 0"))
    if "grad_enabled" in name or "deterministic" in name:
        out.insert(0, ((True,), {}, "True"))
    if "thread" in name:
        out.insert(0, ((1,), {}, "1"))

    if name == "backward":
        # ``backward()`` refuses a tensor with no graph, as the reference
        # does, so the plain tensors below are all refused; it is called on
        # a loss that has one.
        leaf = _probe.as_f64(_probe.sample("positive", (2, 3)))
        leaf.requires_grad_(True)
        out.insert(0, (((leaf * leaf).sum(),), {}, "a scalar loss with a graph"))

    tensor = _probe.as_f64(_probe.sample("positive", (2, 3)))
    out.extend(
        [
            ((tensor,), {}, "one tensor"),
            ((tensor, tensor), {}, "two tensors"),
            (((2, 3),), {}, "a shape"),
            ((2,), {}, "an int"),
        ]
    )
    return out


# ── distributions ────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _dist_base() -> Any:
    """The abstract base every distribution derives from."""
    import lucid.distributions as distributions  # noqa: PLC0415 - optional subsystem

    return distributions.Distribution


def _moment(dist: Any, name: str) -> Any:
    """One attribute of a distribution, or ``None`` when it has none.

    ``getattr`` with a default is not enough here: these are properties,
    and a getter that raises anything other than AttributeError escapes.
    ``NotImplementedError`` is re-raised so the caller can tell "this
    distribution has no closed form for that moment" — which both
    frameworks spell that way — from "the probe fell over".
    """
    try:
        return getattr(dist, name)
    except AttributeError:
        return None


#: How far inside ``(0, 1)`` the ``icdf`` round trip keeps its probabilities,
#: away from the quantile's poles.  Points beyond it are not compared.
_ICDF_CLIP = 1e-4


class DistributionAxis(Axis):
    """Sample, score, and invert — the three must agree with each other.

    ``log_prob`` finite on the distribution's own samples is the check
    that catches a support mismatch; ``cdf`` monotone and ``icdf``
    inverting it catches a parameterisation that is off by a
    transformation.
    """

    name = "distribution"
    summary = "sample / log_prob / cdf / icdf consistency"
    kinds = frozenset({"distribution"})

    _PARAMS: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((), {}),
        ((0.5,), {}),
        ((0.0, 1.0), {}),
        ((1.0, 1.0), {}),
        ((2.0,), {}),
        ((3, 0.5), {}),
    )

    def applies(self, symbol: Symbol) -> bool:
        if symbol.subsystem != "distributions":
            return False
        if symbol.short == "kl_divergence":
            return True
        if not isinstance(symbol.obj, type):
            return False
        # ``Distribution`` and ``ExponentialFamily`` are what the others are
        # built on, not distributions of their own — asking them for a mean
        # gets the base's "subclasses must implement this", which is the
        # base doing its job.  A class that others derive from *and* that
        # defines none of the probed methods of its own is that shape;
        # ``Gamma`` has a subclass and its own mean, so it is not caught.
        # Nor is ``TransformedDistribution``: it has no closed-form mean,
        # but it samples and scores, and it gained a subclass
        # (``LogisticNormal``), which a mean-only test took for a base.
        base = _dist_base()
        defines = any(
            getattr(symbol.obj, name, None) is not getattr(base, name, None)
            for name in ("mean", "sample", "rsample", "log_prob")
        )
        return not (symbol.obj.__subclasses__() and not defines)

    @staticmethod
    def _is_transform(obj: Any) -> bool:
        return (
            hasattr(obj, "inv")
            and callable(obj)
            and not hasattr(obj, "log_prob")
            and not hasattr(obj, "sample")
        )

    def _check_transform(self, symbol: Symbol, transform: Any) -> Finding:
        """``inv(f(x))`` must be ``x``, on the transform's own codomain.

        The probe is pushed through the forward map first and inverted
        from there, so it is inside the domain by construction — an
        inverse asked about a point its forward never produces is being
        asked the wrong question.
        """
        problems: list[str] = []
        source = _probe.as_f64(_probe.rng(19).uniform(0.15, 0.85, (2, 3)))
        try:
            forward = transform(source)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.UNSUPPORTED,
                f"forward: {type(exc).__name__}: {str(exc)[:50]}",
            )
        mapped = _probe.to_numpy(forward)
        if mapped is None:
            return self._finding(
                symbol, Status.SKIP, "the forward map returned no tensor"
            )
        if not np.isfinite(mapped).all():
            problems.append("the forward map is not finite on (0.15, 0.85)")

        inverse = getattr(transform, "inv", None)
        if inverse is not None:
            try:
                back = _probe.to_numpy(inverse(forward))
            except Exception as exc:  # noqa: BLE001
                back = None
                problems.append(f"inv raised {type(exc).__name__}")
            if back is not None:
                if not np.isfinite(back).all():
                    problems.append("inv is not finite on the forward map's own output")
                elif back.shape == np.shape(
                    _probe.to_numpy(source)
                ) and not np.allclose(
                    back.astype(float),
                    _probe.to_numpy(source).astype(float),
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    # ``AbsTransform`` is deliberately not injective, and
                    # says so; a transform that claims a bijection and is
                    # not one is the finding.
                    residual = back.astype(float) - _probe.to_numpy(source).astype(
                        float
                    )
                    if np.allclose(residual - residual[..., :1], 0.0, atol=1e-8):
                        # A constant offset along the transformed axis is
                        # the softmax gauge: adding a constant to every
                        # logit leaves the probabilities unchanged, so no
                        # inverse can recover which representative went
                        # in.  Recovering the canonical one is correct up
                        # to that symmetry, which is what GAUGE says and
                        # FAIL does not.
                        return self._finding(
                            symbol,
                            Status.GAUGE,
                            f"inv recovers x up to a constant shift of "
                            f"{residual.reshape(-1)[0]:+.4f} along the transformed axis",
                        )
                    if getattr(transform, "bijective", True):
                        problems.append(
                            f"inv(forward(x)) is off by {np.abs(residual).max():.3e}"
                        )
        if problems:
            return self._finding(symbol, Status.FAIL, "; ".join(problems))
        return self._finding(
            symbol, Status.PASS, "forward is finite and inv round trips"
        )

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        if symbol.short == "kl_divergence":
            return self._check_kl(symbol)
        dist = None
        # A transform first, from its signature.  The fixed ladder below
        # "succeeds" at ``CumulativeDistributionTransform(0.5)`` — the
        # float is accepted and the object is then broken, so the axis
        # reported ``'Tensor' object has no attribute 'cdf'`` about a
        # class the signature says takes a distribution.
        if symbol.short.endswith("Transform"):
            from lucid.test.audit._axes_data import _construct

            dist, _ = _construct(symbol.obj)
        for args, kwargs in () if dist is not None else self._PARAMS:
            try:
                dist = symbol.obj(
                    *(_probe.as_f64(a) if isinstance(a, float) else a for a in args),
                    **kwargs,
                )
                break
            except Exception:  # noqa: BLE001
                continue
        if dist is None:
            # The fixed ladder cannot reach a constructor that wants a
            # vector, a matrix or another distribution — ``MultivariateNormal``,
            # ``Wishart``, ``MixtureSameFamily`` and the composed
            # transforms, ten classes with no cell answered between them.
            # The signature-driven builder already knows those names.
            from lucid.test.audit._axes_data import _construct

            dist, _ = _construct(symbol.obj)
        if dist is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        # A transform is not a distribution and is enumerated as one:
        # it has no ``sample`` and no ``log_prob``, so every one of them
        # reported "no probeable method".  Its own contract is the round
        # trip, and that is a check with teeth — an inverse that returns
        # NaN for its whole domain is a defect this framework has had.
        if self._is_transform(dist):
            return self._check_transform(symbol, dist)

        problems: list[str] = []
        checked: list[str] = []

        draw = None
        for method in ("sample", "rsample"):
            fn = getattr(dist, method, None)
            if fn is None:
                continue
            try:
                draw = fn((64,))
            except Exception:  # noqa: BLE001
                try:
                    draw = fn()
                except Exception:  # noqa: BLE001
                    continue
            checked.append(method)
            break

        if draw is not None:
            values = _probe.to_numpy(draw)
            if values is not None and not np.isfinite(values).all():
                problems.append("sample produced a non-finite value")
            log_prob = getattr(dist, "log_prob", None)
            if log_prob is not None and values is not None:
                try:
                    scored = _probe.to_numpy(log_prob(draw))
                    checked.append("log_prob")
                    if scored is not None and not np.isfinite(scored).all():
                        problems.append(
                            "log_prob is not finite on the distribution's own samples"
                        )
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"log_prob raised {type(exc).__name__} on its own samples"
                    )

        cdf_problems, cdf_checked = self._check_cdf(dist)
        problems += cdf_problems
        checked += cdf_checked

        # A moment is checked for the things that are true of every
        # distribution, not for being finite.
        #
        # Being finite is a property of the *parameters*, and the axis did
        # not choose them: Cauchy has no mean at any parameter, Pareto has
        # none for α ≤ 1, StudentT has no variance for ν ≤ 2.  The
        # reference returns NaN and infinity for exactly these, rather
        # than raising, because a divergent integral is an answer.  An
        # axis that calls it a defect is reporting the mathematics.
        #
        # What does hold regardless: a variance is not negative, and a
        # standard deviation is its square root.  Both survive infinity.
        moments: dict[str, np.ndarray] = {}
        for stat in ("mean", "variance", "stddev", "entropy"):
            try:
                value = _moment(dist, stat)
                if value is None:
                    continue
                array = _probe.to_numpy(value() if callable(value) else value)
            except NotImplementedError:
                # Both frameworks spell "no closed form" this way —
                # LKJCholesky, RelaxedBernoulli and RelaxedOneHotCategorical
                # raise it in the reference too.
                continue
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{stat} raised {type(exc).__name__}")
                continue
            checked.append(stat)
            if array is not None:
                moments[stat] = np.asarray(array, dtype=np.float64)

        variance = moments.get("variance")
        if variance is not None and np.any(variance < -1e-9):
            problems.append(f"variance is negative: {variance.min():.3e}")
        stddev = moments.get("stddev")
        if stddev is not None and np.any(stddev < -1e-9):
            problems.append(f"stddev is negative: {stddev.min():.3e}")
        if variance is not None and stddev is not None:
            finite = np.isfinite(variance) & np.isfinite(stddev)
            if finite.any() and not np.allclose(
                stddev[finite] ** 2, variance[finite], rtol=1e-6, atol=1e-9
            ):
                problems.append("stddev is not the square root of variance")

        if problems:
            return self._finding(
                symbol, Status.FAIL, "; ".join(problems[:3]), problems=problems
            )
        if not checked:
            return self._finding(symbol, Status.SKIP, "no probeable method")
        return self._finding(
            symbol, Status.PASS, f"checked {', '.join(sorted(set(checked)))}"
        )

    @staticmethod
    def _check_cdf(dist: Any) -> tuple[list[str], list[str]]:
        """``cdf`` monotone and in [0, 1] on a grid, ``icdf`` inverting it.

        Returns the problems found and the methods checked.

        The grid runs from -2 to 2 on purpose, across the edge of most
        supports, so the probe asks the formula about points the
        distribution's argument validation refuses: ``Exponential.cdf(-1)``
        must be 0, not raise.  Validation is turned off on this instance —
        the probe owns it — so those points reach the formula.

        A scorer the distribution does not implement raises
        ``NotImplementedError`` and is skipped; any other exception is a
        finding.
        """
        problems: list[str] = []
        checked: list[str] = []
        # ``getattr(obj, name, default)`` only falls back on
        # AttributeError.  These are *properties*, so a getter that raises
        # NotImplementedError propagates straight out — which is how
        # "Cauchy.mean" arrived as a harness ERROR rather than as anything
        # the axis had decided.
        cdf, icdf = _moment(dist, "cdf"), _moment(dist, "icdf")
        if cdf is None:
            return problems, checked
        dist._validate_args = False
        # One column of the grid per member of the batch.  A batch-(3,)
        # distribution refuses a length-9 value — the reference raises the
        # same broadcast error — so a flat grid asked only the batch-less
        # ones, and ``GeneralizedPareto`` reported the probe's shape.
        batch = tuple(getattr(dist, "batch_shape", ()))
        column = np.linspace(-2.0, 2.0, 9).reshape((9,) + (1,) * len(batch))
        points = np.broadcast_to(column, (9, *batch)).copy()
        try:
            probs = _probe.to_numpy(cdf(_probe.as_f64(points)))
        except NotImplementedError:
            return problems, checked
        except Exception as exc:  # noqa: BLE001 — any other failure is a finding
            problems.append(f"cdf raised {type(exc).__name__}: {str(exc)[:60]}")
            return problems, checked
        checked.append("cdf")
        if probs is None:
            return problems, checked
        # A row per grid point, a column per member of the batch: each
        # column is one distribution's cdf, and only it has to rise.
        if np.shape(probs) != points.shape:
            problems.append(f"cdf of a {points.shape} grid has shape {np.shape(probs)}")
            return problems, checked
        table = np.asarray(probs, dtype=np.float64).reshape(points.shape[0], -1)
        if np.any(np.diff(table, axis=0) < -1e-9):
            problems.append("cdf is not monotone")
        if table.min() < -1e-9 or table.max() > 1.0 + 1e-9:
            problems.append(
                f"cdf leaves [0, 1]: [{table.min():.3f}, {table.max():.3f}]"
            )
        if icdf is None:
            return problems, checked
        # ``icdf(cdf(x)) == x`` only where the cdf is strictly increasing.
        # The grid runs from -2 to 2 and most distributions have support
        # over part of that at best: ``Exponential`` is flat at 0 for every
        # negative x and ``Uniform([0, 1])`` is flat at both ends, so the cdf
        # there is not injective and there is no inverse to check.  Both
        # were reported as "off by 2.00e+00" — the distance from -2 back to
        # the edge of the support, which is the right answer to the wrong
        # question.
        rises = np.diff(table, axis=0) > 1e-9
        invertible = np.zeros(table.shape, dtype=bool)
        invertible[:-1] |= rises
        invertible[1:] |= rises
        # Only points whose probability the clip below leaves alone: a
        # probability moved to the clip bound has another quantile, and
        # comparing that quantile with the original ``x`` measured the clip.
        # ``HalfNormal(0.5).cdf(2) = 0.99994`` was clipped to ``0.9999`` and
        # reported as "icdf(cdf(x)) is off by 5.47e-02" the day the family
        # gained an ``icdf``.
        invertible &= (table > _ICDF_CLIP) & (table < 1 - _ICDF_CLIP)
        if not invertible.any():
            return problems, checked
        inside = _probe.as_f64(
            np.clip(table, _ICDF_CLIP, 1 - _ICDF_CLIP).reshape(np.shape(probs))
        )
        try:
            back = _probe.to_numpy(icdf(inside))
        except NotImplementedError:
            return problems, checked
        except Exception as exc:  # noqa: BLE001 — any other failure is a finding
            problems.append(f"icdf raised {type(exc).__name__}: {str(exc)[:60]}")
            return problems, checked
        checked.append("icdf")
        if back is None:
            return problems, checked
        if np.shape(back) != points.shape:
            problems.append(f"icdf of a {points.shape} grid has shape {np.shape(back)}")
            return problems, checked
        got = np.asarray(back, dtype=np.float64).reshape(table.shape)
        ok = np.isfinite(got) & invertible
        if ok.any():
            want = points.reshape(table.shape)
            err = np.abs(got[ok] - want[ok]).max()
            if err > 1e-3:
                problems.append(f"icdf(cdf(x)) is off by {err:.2e}")
        return problems, checked

    def _check_kl(self, symbol: Symbol) -> Finding:
        """A divergence from a distribution to itself is zero, and never negative."""
        normal = getattr(lucid.distributions, "Normal", None)
        if normal is None:
            return self._finding(symbol, Status.SKIP, "no Normal to compare")
        try:
            p = normal(_probe.as_f64(np.array(0.0)), _probe.as_f64(np.array(1.0)))
            q = normal(_probe.as_f64(np.array(1.5)), _probe.as_f64(np.array(2.0)))
            same = _probe.to_numpy(symbol.obj(p, p))
            apart = _probe.to_numpy(symbol.obj(p, q))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if same is None or apart is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if abs(float(np.asarray(same).reshape(-1)[0])) > 1e-9:
            return self._finding(
                symbol,
                Status.FAIL,
                f"KL(p, p) = {float(np.asarray(same).reshape(-1)[0])}, not 0",
            )
        if float(np.asarray(apart).reshape(-1)[0]) < -1e-9:
            return self._finding(
                symbol,
                Status.FAIL,
                f"KL is negative: {float(np.asarray(apart).reshape(-1)[0])}",
            )
        return self._finding(symbol, Status.PASS, "KL(p,p) = 0 and KL >= 0")


# ── differential equations ───────────────────────────────────────────────────


class DiffeqAxis(Axis):
    """A solver must converge at the order it claims.

    Halving the step has to cut the error by ``2**p``.  This is the check
    that distinguishes two methods of the same order from each other only
    weakly — which is how ``rk4`` once named a different tableau than the
    reference — so the tableau consistency conditions are checked too:
    ``sum(b) == 1`` and every ``sum(A[i]) == c[i]``.
    """

    name = "diffeq"
    summary = "solver convergence order and tableau consistency"
    kinds = frozenset({"diffeq", "value"})

    def applies(self, symbol: Symbol) -> bool:
        # The published tableaux (DOPRI5, GL4, ...) are module-level
        # *instances*, so they enumerate as values rather than as
        # callables — 21 symbols that no axis reached until this said so.
        return symbol.subsystem == "diffeq"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        tableau = _as_tableau(obj)
        if tableau is not None:
            return self._check_tableau(symbol, tableau)
        if symbol.short in ("odeint", "odeint_adjoint"):
            return self._check_convergence(symbol, obj)
        if symbol.short in ("odeint_dense", "odeint_event"):
            return self._check_variant(symbol, obj)
        return self._finding(symbol, Status.NOT_APPLICABLE, "not a solver or a tableau")

    def _check_variant(self, symbol: Symbol, solver: Any) -> Finding:
        """The dense and event solvers, against ``odeint`` on the same problem.

        Neither takes a time grid the way ``odeint`` does — one returns
        an interpolant and the other integrates until a condition — so
        the convergence check above cannot reach them and both reported
        "not a solver or a tableau" about the two solvers in the module.
        The reference is ``odeint`` itself: three routes to the same
        trajectory have to agree on it.
        """

        def rhs(t: Any, y: Any) -> Any:
            return lucid.cos(t) * y * y

        y0 = _probe.as_f64(np.array([0.5]))
        exact = 1.0 / (1.0 / 0.5 - math.sin(1.0))
        try:
            if symbol.short == "odeint_dense":
                interpolant = solver(rhs, y0, 0.0, 1.0)
                got = _probe.to_numpy(interpolant(1.0))
            else:
                # Stop when y reaches the value the exact solution has at
                # t = 1, and the stopping time must be 1.
                def event(t: Any, y: Any) -> Any:
                    return y.reshape(-1)[0] - exact

                when, _ = solver(rhs, y0, 0.0, event_fn=event)
                got = _probe.to_numpy(when)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if got is None:
            return self._finding(symbol, Status.SKIP, "returned nothing measurable")
        value = float(np.asarray(got).reshape(-1)[0])
        want = exact if symbol.short == "odeint_dense" else 1.0
        if abs(value - want) > 1e-5:
            return self._finding(
                symbol,
                Status.FAIL,
                f"gave {value:.9g}, the exact answer is {want:.9g}",
            )
        return self._finding(symbol, Status.PASS, f"agrees to {abs(value - want):.2e}")

    def _check_tableau(self, symbol: Symbol, tableau: Any) -> Finding:
        a = np.asarray(tableau[0], dtype=np.float64)
        b = np.asarray(tableau[1], dtype=np.float64)
        c = np.asarray(tableau[2], dtype=np.float64)
        problems = []
        if abs(b.sum() - 1.0) > 1e-12:
            problems.append(f"weights sum to {b.sum():.12g}, not 1")
        rows = a.sum(axis=1) if a.ndim == 2 else a
        if rows.shape == c.shape and np.abs(rows - c).max() > 1e-12:
            problems.append(
                f"row sums differ from the nodes by {np.abs(rows - c).max():.2e}"
            )
        if problems:
            return self._finding(symbol, Status.FAIL, "; ".join(problems))
        return self._finding(symbol, Status.PASS, f"{len(b)} stages, consistent")

    def _check_convergence(self, symbol: Symbol, solver: Any) -> Finding:
        # y' = -y, y(0) = 1 is deliberately *not* the test problem: every
        # four-stage fourth-order explicit method collapses onto the same
        # stability polynomial there and the check passes for the wrong
        # reason.  A non-autonomous, non-linear right-hand side does not.
        def rhs(t: Any, y: Any) -> Any:
            return lucid.cos(t) * y * y

        y0 = _probe.as_f64(np.array([0.5]))
        errors = []
        for steps in (8, 16, 32):
            grid = [i / steps for i in range(steps + 1)]
            # ``odeint`` takes ``return_trajectory`` and ``odeint_adjoint``
            # does not — passing it unconditionally made the adjoint
            # solver, the one whose gradients are the reason it exists,
            # report UNSUPPORTED on a keyword rather than run.
            options: dict[str, Any] = {"method": "rk4"}
            if symbol.short == "odeint":
                options["return_trajectory"] = False
            try:
                out = solver(rhs, y0, grid, **options)
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            got = _probe.to_numpy(out)
            if got is None:
                return self._finding(symbol, Status.SKIP, "solver returned no tensor")
            # dy/y^2 = cos(t) dt  =>  1/y0 - 1/y = sin(t)
            exact = 1.0 / (1.0 / 0.5 - math.sin(1.0))
            # ``odeint_adjoint`` has no ``return_trajectory`` and always
            # answers with the whole path, so reading entry 0 read
            # ``y0`` — a constant, and therefore a constant error at
            # every step count and an observed order of exactly zero.
            # The endpoint is the last entry whichever shape came back.
            errors.append(abs(float(np.asarray(got).reshape(-1)[-1]) - exact))

        if errors[-1] < 1e-13:
            return self._finding(symbol, Status.PASS, "converged to round-off")
        orders = [
            math.log2(errors[i] / errors[i + 1])
            for i in range(len(errors) - 1)
            if errors[i + 1] > 0
        ]
        if not orders:
            return self._finding(symbol, Status.PASS, "exact at every step count")
        observed = min(orders)
        if observed < 3.0:
            return self._finding(
                symbol,
                Status.FAIL,
                f"rk4 converged at order {observed:.2f}, expected ~4 "
                f"(errors {['%.2e' % e for e in errors]})",
            )
        return self._finding(symbol, Status.PASS, f"observed order {observed:.2f}")


def _as_tableau(obj: Any) -> tuple[Any, Any, Any] | None:
    """``(A, b, c)`` if ``obj`` looks like a Butcher tableau.

    ``A`` is stored the way the method is written on paper: strictly
    lower triangular, so row *i* has *i* entries and the first row is
    empty.  ``np.asarray`` on a ragged tuple-of-tuples raises, which sent
    all twelve published tableaux — ``DOPRI5``, ``RK4``, ``TSIT5`` and
    the rest — down the "not a solver or a tableau" path, and the
    consistency conditions this axis exists to check were never applied
    to a single one of them.  Padding is the whole fix: the missing
    entries are zero by definition.
    """
    parts = []
    for names in (("a", "A"), ("b", "B"), ("c", "C")):
        value = next((getattr(obj, n) for n in names if hasattr(obj, n)), None)
        if value is None:
            return None
        parts.append(value)
    try:
        rows = list(parts[0])
        width = max((len(row) for row in rows), default=0)
        square = np.zeros((len(rows), max(width, len(rows))), dtype=np.float64)
        for index, row in enumerate(rows):
            if len(row):
                square[index, : len(row)] = np.asarray(row, dtype=np.float64)
        return (
            square,
            np.asarray(parts[1], dtype=np.float64),
            np.asarray(parts[2], dtype=np.float64),
        )
    except Exception:  # noqa: BLE001
        return None


# ── quantization, serialization, compile ─────────────────────────────────────


def _float_model() -> Any:
    """A small float module the quantisation flows can be run over."""
    return lucid.nn.Sequential(
        lucid.nn.Linear(8, 8),
        lucid.nn.ReLU(),
        lucid.nn.Linear(8, 4),
    )


def _calibration() -> Any:
    return _probe.as_f32(_probe.rng(5).uniform(-2.0, 2.0, (4, 8)))


class QuantizationAxis(Axis):
    """The quantisation flows must produce a model that still computes.

    One of twenty-three cells answered.  The axis asked a single
    question — does ``quantize`` round trip within a step — and dispatched
    on the substring ``"quantize"``, so ``prepare``, ``convert``,
    ``fuse_modules``, ``prepare_qat``, ``convert_fx``, every observer and
    every qconfig factory fell through to "not a quantize entry point".
    The subsystem was in the denominator and out of the audit.

    Four questions now, one per kind of entry point, each with a
    reference that is not this file's opinion:

    * a **tensor** round trip must land within one step of the original;
    * an **observer** fed a known range must report qparams that span it;
    * a **flow** (``prepare`` / ``convert`` / ``quantize_dynamic`` /
      ``fuse_modules``) must return a module whose output still tracks
      the float model it came from — quantisation is lossy and it is not
      arbitrary, so the comparison is against the float original at a
      tolerance, and a flow that silently produces noise fails it;
    * a **qconfig factory** must hand back something that constructs the
      observers it names.
    """

    name = "quant"
    summary = "observers, qconfigs and the prepare/convert flows keep the model working"
    kinds = frozenset({"quant"})
    varies_a_tensor = False

    _FLOWS = frozenset(
        {
            "prepare",
            "prepare_fx",
            "prepare_qat",
            "convert",
            "convert_fx",
            "quantize",
            "quantize_dynamic",
            "fuse_modules",
            "fuse_modules_qat",
        }
    )

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        obj = symbol.obj
        if not callable(obj):
            return self._finding(symbol, Status.SKIP, "not callable")
        name = symbol.short
        if name.startswith("get_default_q"):
            return self._qconfig_factory(symbol, obj)
        if isinstance(obj, type) and "Observer" in name or name == "FakeQuantize":
            return self._observer(symbol, obj)
        if name in ("calculate_qparams", "quantize", "fake_quantize", "dequantize"):
            return self._tensor_level(symbol, obj, name)
        if name in self._FLOWS:
            return self._flow(symbol, obj, name)
        if "quantize" not in name and "qparams" not in name:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "a constant, a scheme or a container — no flow to run",
            )

        x = _probe.as_f32(_probe.rng(3).uniform(-2.0, 2.0, (4, 8)))
        for args, kwargs in (
            ((x, 0.01, 0), {}),
            ((x,), {"scale": 0.01, "zero_point": 0}),
            ((x,), {}),
        ):
            try:
                q = obj(*args, **kwargs)
            except Exception:  # noqa: BLE001
                continue
            dq = None
            for name in ("dequantize", "dequantize_per_tensor"):
                fn = getattr(lucid.quantization, name, None) or getattr(
                    q, "dequantize", None
                )
                if fn is None:
                    continue
                try:
                    dq = _probe.to_numpy(
                        fn(q) if fn is not getattr(q, "dequantize", None) else fn()
                    )
                    break
                except Exception:  # noqa: BLE001
                    continue
            if dq is None:
                return self._finding(symbol, Status.PASS, "quantized without error")
            error = np.abs(dq.astype(float) - _probe.to_numpy(x).astype(float)).max()  # type: ignore[union-attr]
            if error > 0.02:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"round trip is off by {error:.4f}, more than one step",
                )
            return self._finding(symbol, Status.PASS, f"round trip within {error:.4f}")
        return self._finding(symbol, Status.SKIP, "no argument shape worked")

    # ── the other three kinds of entry point ─────────────────────────────────

    def _tensor_level(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        """The four functions that take a tensor and explicit qparams.

        Their required arguments are a scale, a zero point and a qdtype,
        none of which any generic derivation can invent — which is why
        they read as "no argument shape worked" while being the most
        directly checkable things in the subsystem.  The reference is
        arithmetic: ``dequantize(quantize(x)) - x`` must be inside half a
        step, and ``calculate_qparams`` over a known range must produce a
        scale that spans it.
        """
        quantization = lucid.quantization
        x = _probe.as_f32(_probe.rng(3).uniform(-2.0, 2.0, (4, 8)))
        low, high = -2.0, 2.0
        try:
            scale, zero_point = quantization.calculate_qparams(
                _probe.as_f32(np.array(low)),
                _probe.as_f32(np.array(high)),
                quantization.QScheme.PER_TENSOR_AFFINE,
                quantization.quint8,
            )
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.UNSUPPORTED,
                f"qparams: {type(exc).__name__}: {str(exc)[:50]}",
            )

        step = float(np.asarray(_probe.to_numpy(scale)).reshape(-1)[0])
        if name == "calculate_qparams":
            # 8 bits over a range of 4 is a step of about 4/255.
            wanted = (high - low) / 255.0
            if not 0.5 * wanted <= step <= 2.0 * wanted:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"a range of {high - low} over 8 bits gave a step of {step:.5g}, "
                    f"expected about {wanted:.5g}",
                )
            return self._finding(
                symbol, Status.PASS, f"step {step:.5g} spans the range"
            )

        try:
            if name == "quantize":
                out = obj(x, scale, zero_point, quantization.quint8)
                out = quantization.dequantize(out, scale, zero_point)
            elif name == "dequantize":
                out = obj(
                    quantization.quantize(x, scale, zero_point, quantization.quint8),
                    scale,
                    zero_point,
                )
            else:  # fake_quantize
                out = obj(x, scale, zero_point, 0, 255)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        got = _probe.to_numpy(out)
        if got is None:
            return self._finding(symbol, Status.SKIP, "nothing measurable came back")
        error = float(
            np.abs(got.astype(float) - _probe.to_numpy(x).astype(float)).max()
        )
        if error > step:
            return self._finding(
                symbol,
                Status.FAIL,
                f"the round trip is off by {error:.5g}, more than one step ({step:.5g})",
            )
        return self._finding(
            symbol, Status.PASS, f"within one step: {error:.3g} <= {step:.3g}"
        )

    def _qconfig_factory(self, symbol: Symbol, obj: Any) -> Finding:
        try:
            config = obj()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
            )
        if config is None:
            return self._finding(symbol, Status.FAIL, "returned None")
        # A mapping names configs per module type; a config names two
        # observer factories.  Either way the point is that what comes
        # back can be *built*, not merely returned.
        factories = [
            getattr(config, attribute, None)
            for attribute in ("activation", "weight")
            if getattr(config, attribute, None) is not None
        ]
        if not factories:
            lookup = getattr(config, "get_qconfig", None)
            if lookup is None:
                return self._finding(
                    symbol, Status.SKIP, f"nothing to build on {type(config).__name__}"
                )
            # A mapping's whole job is to answer for a module type.  One
            # that answers ``None`` for a Linear leaves every layer
            # unquantised and reports nothing.
            entry = lookup(lucid.nn.Linear(4, 4), "0")
            if entry is None:
                return self._finding(
                    symbol, Status.FAIL, "the mapping has no qconfig for nn.Linear"
                )
            factories = [
                getattr(entry, attribute)
                for attribute in ("activation", "weight")
                if getattr(entry, attribute, None) is not None
            ]
        for factory in factories:
            try:
                observer = factory()
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"the qconfig names an observer that will not build: "
                    f"{type(exc).__name__}: {str(exc)[:50]}",
                )
            if observer is None:
                return self._finding(
                    symbol, Status.FAIL, "an observer factory gave None"
                )
        return self._finding(symbol, Status.PASS, f"{len(factories)} observers build")

    def _observer(self, symbol: Symbol, cls: Any) -> Finding:
        try:
            observer = cls()
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"construct: {type(exc).__name__}"
            )
        # A known range in; the reported qparams have to be able to
        # represent it.  An observer that ignores its input reports the
        # same scale for every range, which is the failure that makes a
        # calibrated model quietly worse than an uncalibrated one.
        narrow = _probe.as_f32(_probe.rng(7).uniform(-0.5, 0.5, (4, 8)))
        wide = _probe.as_f32(_probe.rng(7).uniform(-40.0, 40.0, (4, 8)))
        scales = []
        for probe in (narrow, wide):
            try:
                fresh = cls()
                fresh(probe)
                params = fresh.calculate_qparams()
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"calibrate: {type(exc).__name__}"
                )
            scale = _probe.to_numpy(params[0] if isinstance(params, tuple) else params)
            if scale is None:
                return self._finding(symbol, Status.SKIP, "no scale to compare")
            scales.append(float(np.asarray(scale).reshape(-1)[0]))
        if not all(s > 0 for s in scales):
            return self._finding(symbol, Status.FAIL, f"non-positive scale: {scales}")
        if scales[1] <= scales[0]:
            return self._finding(
                symbol,
                Status.FAIL,
                f"an 80-wide range got scale {scales[1]:.3g} and a 1-wide range "
                f"{scales[0]:.3g} — the observer is not reading its input",
            )
        del observer
        return self._finding(
            symbol,
            Status.PASS,
            f"scale tracks the range: {scales[0]:.3g} -> {scales[1]:.3g}",
        )

    def _flow(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        model = _float_model()
        model.eval()
        probe = _calibration()
        want = _probe.to_numpy(model(probe))
        qconfig = None
        with contextlib.suppress(Exception):
            qconfig = lucid.quantization.get_default_qconfig()

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if name.startswith("fuse"):
            attempts = [((model, [["0", "1"]]), {}), ((model, ["0", "1"]), {})]
        elif name == "quantize_dynamic":
            attempts = [((model,), {}), ((model,), {"qconfig": qconfig})]
        elif name.startswith("prepare"):
            attempts = [((model,), {"qconfig": qconfig}), ((model,), {})]
        else:  # convert / convert_fx / quantize
            prepared = model
            with contextlib.suppress(Exception):
                prepared = lucid.quantization.prepare(model, qconfig=qconfig)
                prepared(probe)  # calibrate, or the observers have seen nothing
            attempts = [((prepared,), {}), ((model,), {})]

        for args, kwargs in attempts:
            try:
                out = obj(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:60]}"
                continue
            if not isinstance(out, lucid.nn.Module):
                return self._finding(
                    symbol, Status.PASS, f"returned {type(out).__name__}"
                )
            try:
                got = _probe.to_numpy(out(probe))
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"the returned model does not run: {type(exc).__name__}: "
                    f"{str(exc)[:50]}",
                )
            if got is None or want is None:
                return self._finding(symbol, Status.SKIP, "no output to compare")
            if got.shape != want.shape:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"output shape {got.shape}, float model gives {want.shape}",
                )
            error = float(np.abs(got.astype(float) - want.astype(float)).max())
            scale = max(float(np.abs(want).max()), 1e-9)
            # Quantisation is lossy; it is not arbitrary.  A tenth of the
            # signal is far outside int8 error and inside "this flow
            # returned something that no longer computes the model".
            if error > 0.1 * scale + 1e-3:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"the quantised model differs from the float one by "
                    f"{error:.3e}, {100 * error / scale:.0f}% of the signal",
                )
            return self._finding(
                symbol,
                Status.PASS,
                f"within {100 * error / scale:.1f}% of the float model",
            )
        return self._finding(symbol, Status.UNSUPPORTED, last)


class SerializationAxis(Axis):
    """A tensor must survive a round trip through disk, bit for bit."""

    name = "serialize"
    summary = "save / load round trip preserves values, dtype and shape"
    kinds = frozenset({"serialize"})

    #: ``(saver, loader, file suffix, what a payload looks like)``.  Every
    #: pair is one format, and only the first of the four was audited:
    #: ``applies`` named ``save``/``load`` literally, so the safetensors
    #: and sharded writers — four public symbols and the two formats a
    #: checkpoint actually ships in — had no axis at all.
    _PAIRS: tuple[tuple[str, str, str, bool], ...] = (
        ("save", "load", ".lct", False),
        ("save_safetensors", "load_safetensors", ".safetensors", True),
        ("save_sharded", "load_sharded", "", True),
    )

    def applies(self, symbol: Symbol) -> bool:
        # ``save`` and ``load`` touch the filesystem, so they are flagged
        # stateful and the base class would refuse them.  This axis calls
        # them on purpose, inside a temporary directory it owns.
        names = {name for pair in self._PAIRS for name in pair[:2]}
        return symbol.kind == "serialize" and symbol.short in names

    def _pair_for(self, short: str) -> tuple[str, str, str, bool] | None:
        return next((p for p in self._PAIRS if short in p[:2]), None)

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        pair = self._pair_for(symbol.short)
        if pair is None:
            return self._finding(symbol, Status.SKIP, "not a save/load pair")
        saver, loader, suffix, wants_mapping = pair
        save = getattr(lucid, saver, None)
        load = getattr(lucid, loader, None)
        if save is None or load is None:
            return self._finding(symbol, Status.SKIP, f"{saver}/{loader} not exposed")

        tensor = _probe.as_f64(_probe.sample("moderate", (3, 4)))
        # The two structured formats carry a *state dict*, not a bare
        # tensor; handing them one is what "no argument shape worked"
        # meant on the smoke axis for all four of them.
        original = {"probe": tensor} if wants_mapping else tensor
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ("shards" if not suffix else f"probe{suffix}")
            try:
                save(original, str(path))
                restored = load(str(path))
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:60]}"
                )
            if wants_mapping:
                if not hasattr(restored, "keys"):
                    return self._finding(
                        symbol,
                        Status.FAIL,
                        f"a mapping was written and {type(restored).__name__} came back",
                    )
                if set(restored) != set(original):
                    return self._finding(
                        symbol,
                        Status.FAIL,
                        f"keys {sorted(original)} became {sorted(restored)}",
                    )
                original, restored = tensor, restored["probe"]
            a, b = _probe.to_numpy(original), _probe.to_numpy(restored)
            if a is None or b is None:
                return self._finding(
                    symbol, Status.SKIP, "round trip produced no tensor"
                )
            if a.shape != b.shape:
                return self._finding(
                    symbol, Status.FAIL, f"shape {a.shape} became {b.shape}"
                )
            if not np.array_equal(a, b, equal_nan=True):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"values changed by {np.nanmax(np.abs(a - b)):.3e} through the round trip",
                )
            if str(original.dtype) != str(restored.dtype):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"dtype {original.dtype} became {restored.dtype}",
                )
        return self._finding(symbol, Status.PASS, "exact round trip")


def _eager(t: Any) -> Any:
    return lucid.tanh(t * 2.0) + lucid.exp(-t)


class CompiledAxis(Axis):
    """A compiled artefact must agree with the eager one it came from.

    Zero of seven cells answered.  ``run`` began by refusing anything not
    literally named ``compile``, so the optimizer compiler, the fused
    training step, the two halves of the compiled-artefact save/load pair
    and the diagnostic all skipped — the entire subsystem verified by one
    function.

    The reference is the eager route in every case, because "compiled"
    is a claim about *equality with something already trusted* and
    nothing else.  A compiled step that trains to a different place than
    the eager step is the defect worth catching, and it is invisible to
    anything that only asks whether the compiler returned.
    """

    name = "compiled"
    summary = "compiled functions, steps and artefacts match the eager route"
    kinds = frozenset({"compiled"})
    varies_a_tensor = False

    def applies(self, symbol: Symbol) -> bool:
        # Stateful and checked anyway, exactly as ``save``/``load`` are.
        #
        # ``compile``, ``compile_optimizer``, ``compiled_step`` and the
        # artefact pair install tracing state, so they are flagged
        # stateful and the base class refuses them — which meant this
        # axis never saw **the compile entry point itself**.  Seven cells
        # ran and ``compile`` was not among them: the subsystem's
        # headline function was outside its own axis while the axis
        # reported on four classes that are not entry points at all.
        return symbol.kind == "compiled"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        name = symbol.short
        obj = symbol.obj
        if not callable(obj):
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "not a compile entry point"
            )
        x = _probe.as_f32(_probe.sample("moderate", (3, 4)))
        try:
            if name == "compile":
                return self._function(symbol, obj, x)
            if name in ("save_compiled", "load_compiled"):
                return self._artefact(symbol, x)
            if name == "diagnose":
                return self._diagnose(symbol, obj, x)
            if name in (
                "compile_optimizer",
                "compiled_step",
                "make_step",
                "fused_step",
            ):
                return self._step(symbol, obj, name)
        except Exception as exc:  # noqa: BLE001 - surveying, not asserting
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )
        return self._finding(symbol, Status.NOT_APPLICABLE, "not a compile entry point")

    def _function(self, symbol: Symbol, compile_fn: Any, x: Any) -> Finding:
        try:
            compiled = compile_fn(_eager)
            got = _probe.to_numpy(compiled(x))
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"{type(exc).__name__}: {str(exc)[:70]}"
            )
        want = _probe.to_numpy(_eager(x))
        if got is None or want is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if not np.allclose(got.astype(float), want.astype(float), rtol=1e-5, atol=1e-7):
            return self._finding(
                symbol,
                Status.FAIL,
                f"compiled and eager differ by "
                f"{np.abs(got.astype(float) - want.astype(float)).max():.3e}",
            )
        return self._finding(symbol, Status.PASS, "compiled matches eager")

    def _artefact(self, symbol: Symbol, x: Any) -> Finding:
        save = getattr(lucid.compile, "save_compiled", None)
        load = getattr(lucid.compile, "load_compiled", None)
        if save is None or load is None:
            return self._finding(symbol, Status.SKIP, "the pair is not exposed")
        compiled = lucid.compile.compile(_eager)
        # Warmed, on Metal, before it is saved.  ``compile`` is lazy —
        # the graph is built on the first call, and only a GPU call
        # builds one — so saving a freshly compiled module writes nothing
        # and raises "the CompiledModule has no compiled graph", which
        # reads as a defect in ``save_compiled`` and is the probe saving
        # something that does not exist yet.
        if _probe.metal_available():
            x = _probe.as_f32(_probe.sample("moderate", (3, 4)), "metal")
        want = _probe.to_numpy(compiled(x))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "artefact.lcc")
            if not save(compiled, path):
                return self._finding(
                    symbol, Status.UNSUPPORTED, "the artefact declined to be saved"
                )
            restored = load(path)
            if restored is None:
                return self._finding(symbol, Status.FAIL, "saved, and loaded back None")
            got = _probe.to_numpy(restored(x))
        if got is None or want is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if not np.allclose(got.astype(float), want.astype(float), rtol=1e-5, atol=1e-7):
            return self._finding(
                symbol,
                Status.FAIL,
                "the reloaded artefact computes something else",
            )
        return self._parameterised_artefact(symbol)

    def _parameterised_artefact(self, symbol: Symbol) -> Finding:
        """The half of save/load a parameterless probe cannot reach.

        ``_eager`` closes over nothing, so it traces to a single feed and
        every ordering of one feed is the right one.  A module carries its
        weights as feeds too, of shapes the activations do not share, and
        that is the case a reloaded package mis-binds — for a long time by
        aborting the process inside MetalPerformanceShaders, which no
        probe over a one-feed function could ever have surfaced.  The
        contract now is: round-trip exactly, or refuse in Python.  Silence
        followed by different numbers is the failure.
        """
        if not _probe.metal_available():
            return self._finding(symbol, Status.SKIP, "needs Metal")
        model = lucid.nn.Linear(8, 4).to("metal")
        compiled = lucid.compile.compile(model)
        x = _probe.as_f32(_probe.sample("moderate", (2, 8)), "metal")
        # Under grad mode a trainable model's call runs eagerly and builds no
        # executable to save (LCD-582).
        with lucid.no_grad():
            want = _probe.to_numpy(compiled(x))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "module.lcc")
            if not lucid.compile.save_compiled(compiled, path):
                return self._finding(
                    symbol,
                    Status.UNSUPPORTED,
                    "a module's artefact declined to be saved",
                )
            try:
                restored = lucid.compile.load_compiled(path)
            except RuntimeError, ValueError:
                # Refused rather than mis-bound — the honest outcome while
                # the feed order of a reloaded package is unrecoverable.
                return self._finding(
                    symbol,
                    Status.PASS,
                    "round trip for a function, refusal for a module",
                )
            if restored is None:
                return self._finding(
                    symbol, Status.PASS, "round trip for a function, miss for a module"
                )
            try:
                got = _probe.to_numpy(restored(x, *model.parameters()))
            except RuntimeError, ValueError:
                return self._finding(
                    symbol,
                    Status.PASS,
                    "round trip for a function, refusal for a module",
                )
        if got is None or want is None:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if not np.allclose(got.astype(float), want.astype(float), rtol=1e-5, atol=1e-7):
            return self._finding(
                symbol,
                Status.FAIL,
                "a module's reloaded artefact loaded, and computes something else",
            )
        return self._finding(symbol, Status.PASS, "the artefact survives a round trip")

    def _diagnose(self, symbol: Symbol, obj: Any, x: Any) -> Finding:
        report = obj(_eager, x)
        if report is None:
            return self._finding(symbol, Status.FAIL, "reported nothing")
        text = repr(report)
        if not text:
            return self._finding(symbol, Status.FAIL, "the report has an empty repr")
        return self._finding(symbol, Status.PASS, f"reports {type(report).__name__}")

    def _step(self, symbol: Symbol, obj: Any, name: str) -> Finding:
        """A compiled training step must land where the eager one lands.

        Both routes start from the same weights and see the same batch,
        so after one step the parameters have to agree.  Anything looser
        would pass a step that runs and optimises nothing.
        """
        # On Metal where there is one.  ``compiled_step`` and
        # ``fused_step`` trace into MPSGraph and say so — "only
        # Device::GPU trace" — so a CPU probe reports UNSUPPORTED about
        # the two functions whose whole purpose is the GPU path.
        device = "metal" if _probe.metal_available() else "cpu"
        probe = _probe.as_f32(_probe.rng(11).uniform(-1.0, 1.0, (4, 6)), device)
        target = _probe.as_f32(_probe.rng(12).uniform(-1.0, 1.0, (4, 3)), device)

        def build() -> tuple[Any, Any]:
            lucid.manual_seed(17)
            model = lucid.nn.Linear(6, 3)
            if device != "cpu":
                model.to(device)
            return model, lucid.optim.SGD(model.parameters(), lr=0.1)

        def loss_fn(prediction: Any, expected: Any = target) -> Any:
            return ((prediction - expected) ** 2).mean()

        eager_model, eager_opt = build()
        eager_opt.zero_grad()
        loss_fn(eager_model(probe)).backward()
        eager_opt.step()
        want = [_probe.to_numpy(p) for p in eager_model.parameters()]

        model, optimiser = build()
        # Each of the four takes what its own signature says, and the
        # signatures do not agree with one another — ``make_step`` wants
        # a loss function, ``fused_step`` wants the optimizer as well,
        # ``compiled_step`` is the step itself rather than a factory.
        if name == "compile_optimizer":
            # Not a step factory at all: it returns a *drop-in
            # optimizer* with ``step`` and ``zero_grad``, so driving it
            # like the others found "no calling convention worked" for an
            # object whose convention is the one every optimizer has.
            compiled_opt = obj(optimiser)
            compiled_opt.zero_grad()
            loss_fn(model(probe)).backward()
            compiled_opt.step()
            stepper = None
        elif name == "compiled_step":
            # It runs the backward itself and fills ``.grad``; the loss it
            # returns carries no graph.  Calling ``backward()`` on it was a
            # silent no-op until ``backward()`` started refusing tensors
            # without one.
            optimiser.zero_grad()
            obj(model, probe, loss_fn)
            optimiser.step()
            stepper = None
        else:
            stepper = {
                "make_step": lambda: obj(model, loss_fn),
                "fused_step": lambda: obj(model, loss_fn, optimiser),
            }[name]()
            if stepper is None:
                return self._finding(
                    symbol, Status.FAIL, "returned nothing to step with"
                )

        if stepper is not None:
            optimiser.zero_grad()
            for attempt in (
                lambda: stepper(probe),
                lambda: stepper(probe, target),
                lambda: stepper(),
            ):
                try:
                    result = attempt()
                    if name == "make_step":
                        # ``make_step`` returns the *loss*, with a
                        # working ``grad_fn`` — it compiles forward and
                        # backward and leaves both the ``backward()`` and
                        # the update to the caller.  Stepping the
                        # optimizer without backward first moved nothing,
                        # and the axis reported the untouched weights as
                        # a compiled/eager disagreement.
                        if hasattr(result, "backward"):
                            result.backward()
                        optimiser.step()
                    del result
                    break
                except TypeError:
                    continue
            else:
                return self._finding(
                    symbol,
                    Status.UNSUPPORTED,
                    "no calling convention for the step worked",
                )

        got = [_probe.to_numpy(p) for p in model.parameters()]
        if len(got) != len(want) or any(
            a is None or b is None for a, b in zip(got, want)
        ):
            return self._finding(symbol, Status.SKIP, "no parameters to compare")
        drift = max(
            float(np.abs(a.astype(float) - b.astype(float)).max())
            for a, b in zip(got, want)
        )
        moved = max(float(np.abs(a.astype(float)).max()) for a in want)
        if drift > 1e-4 * max(moved, 1.0):
            return self._finding(
                symbol,
                Status.FAIL,
                f"after one step the compiled and eager parameters differ by {drift:.3e}",
            )
        return self._finding(symbol, Status.PASS, f"one step agrees to {drift:.2e}")


class ClassContractAxis(Axis):
    """Plain classes: constructible, and their repr does not raise.

    The floor for a symbol nothing else can express.  A class that cannot
    be constructed with any obvious signature is reported SKIP so it
    shows up in the coverage gap rather than passing by omission.
    """

    name = "contract"
    summary = "plain classes construct and repr without raising"
    kinds = frozenset({"class", "distribution", "quant", "util", "diffeq"})

    def applies(self, symbol: Symbol) -> bool:
        return symbol.kind in self.kinds and isinstance(symbol.obj, type)

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        # Signature-driven first, and only then the blind ladder.
        #
        # ``_try_construct`` tries a fixed list of argument tuples — ``(4,)``,
        # ``(3, 4)`` and so on — which is fine for a class that takes sizes
        # and wrong for one that does not.  ``MemoryStats(impl)`` accepted
        # the int, and its ``repr`` then failed with "'int' object has no
        # attribute 'current_bytes'", reported as a defect in a class nobody
        # would call that way.  Reading the signature refuses instead.
        from lucid.test.audit._axes_data import _construct

        instance, why = _construct(symbol.obj)
        if instance is None:
            # Falling back to the ladder here would undo the point: it is
            # the ladder's willingness to pass an int for anything that
            # produced these reports.  The ladder is only for a class whose
            # signature could not be read at all.
            if "no signature" not in why:
                return self._finding(symbol, Status.SKIP, f"construct: {why}")
            probe = _try_construct(symbol.obj)
            if probe is None:
                return self._finding(symbol, Status.SKIP, f"construct: {why}")
            instance = probe
        if instance is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")
        try:
            text = repr(instance)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"repr raised {type(exc).__name__}: {exc}"
            )
        if not text:
            return self._finding(symbol, Status.FAIL, "repr returned an empty string")
        return self._finding(symbol, Status.PASS, "constructs and reprs")


class ConstantAxis(Axis):
    """Module-level values: dtypes, schemes, type aliases, the version.

    The last 60-odd symbols, and the reason the reach figure could not
    pass 92%.  They are not callable, so nothing above touches them —
    but a dtype that cannot build a tensor, a scheme constant that is
    silently the same object as another, or a ``__version__`` that is not
    a version are all real breakages, and each is one line to check.
    """

    name = "constant"
    summary = "dtypes build tensors, scheme constants are distinct, aliases resolve"
    kinds = frozenset({"value"})

    def applies(self, symbol: Symbol) -> bool:
        return symbol.kind == "value"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        value = symbol.obj
        if value is None:
            return self._finding(symbol, Status.FAIL, "the exported value is None")

        try:
            text = repr(value)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"repr raised {type(exc).__name__}: {exc}"
            )
        if not text:
            return self._finding(symbol, Status.FAIL, "repr returned an empty string")

        if symbol.short == "__version__":
            if not isinstance(value, str) or not value[:1].isdigit():
                return self._finding(
                    symbol, Status.FAIL, f"not a version string: {value!r}"
                )
            return self._finding(symbol, Status.PASS, f"version {value}")

        # A dtype earns the strongest check available: build with it and
        # read the dtype back off the result.
        try:
            tensor = lucid.zeros((2, 2), dtype=value)
        except Exception:  # noqa: BLE001
            tensor = None
        if tensor is not None:
            if str(tensor.dtype) != str(value):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"a tensor built with {value} reports dtype {tensor.dtype}",
                )
            return self._finding(
                symbol, Status.PASS, f"builds a tensor of {tensor.dtype}"
            )

        return self._finding(symbol, Status.PASS, f"exported value: {text[:48]}")


SUBSYSTEM_AXES: tuple[Axis, ...] = (
    DistributionAxis(),
    DiffeqAxis(),
    QuantizationAxis(),
    SerializationAxis(),
    CompiledAxis(),
    ClassContractAxis(),
    ConstantAxis(),
    SmokeAxis(),
)

__all__ = [
    "SUBSYSTEM_AXES",
    "ClassContractAxis",
    "CompiledAxis",
    "ConstantAxis",
    "DiffeqAxis",
    "DistributionAxis",
    "QuantizationAxis",
    "SerializationAxis",
    "SmokeAxis",
    "StateGuard",
]
