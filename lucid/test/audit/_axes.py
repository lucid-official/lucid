"""The checks.

Each axis is one question asked of one symbol.  The questions were not
chosen abstractly — every one of them has already caught a defect in this
framework, and the note on each class says which.

Three habits are built into the base rather than left to each axis:

* **the instrument is guarded** — a check that cannot fail is reported
  VACUOUS, not PASS, because a vacuous pass reads as coverage and is not;
* **a disagreement is interrogated before it is reported** — a
  finite-difference probe near a pole disagrees for reasons that have
  nothing to do with the op, so a first FAIL is re-run at a finer step and
  reclassified TRUNCATION if it shrinks quadratically;
* **two spellings beat one hand-derivation** — where an op has another
  route to the same answer, that is the reference, because a hand-written
  expected value encodes a convention the framework is free to choose
  differently (``hardtanh`` at its own clamp boundary).
"""

import annotationlib
import contextlib
import inspect
import functools
import json
import pathlib
import re
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.autograd
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._builders import BUILDERS
from lucid.test.audit._result import Finding, Status

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lucid.test.audit._specs import Call
    from lucid.test.audit._surface import Symbol


class Context:
    """Run-wide switches an axis may consult."""

    def __init__(
        self, quick: bool = False, metal: bool = True, step: float = 1e-5, tolerance: float = 2e-5
    ) -> None:
        self.quick = quick
        self.metal = metal
        self.step = step
        self.tolerance = tolerance

    @property
    def domains(self) -> list[str]:
        if self.quick:
            return ["moderate", "positive"]
        return list(_probe.DOMAINS)


@functools.lru_cache(maxsize=2048)
def _takes_no_tensor(fn: Any) -> bool:
    """Whether the first thing ``fn`` takes is a function or a module.

    The numeric axes vary a tensor and read a tensor back.  ``func.grad``
    takes a *function*, ``nn.utils.weight_norm`` takes a *module*, and
    neither has an operand for a finite difference to perturb — so all
    eleven of them reported SKIP, 140 cells across fourteen symbols,
    which reads as "nobody checked these".

    Somebody does: :class:`~lucid.test.audit._axes_state.
    FunctionalTransformAxis` compares ``grad(f)(x)`` against
    ``backward()``, and :class:`~lucid.test.audit._axes_state.NnUtilsAxis`
    checks that a parametrisation does not change what the module
    computes.  They are checked *better* than a numeric axis could,
    because the question fits.  Leaving the numeric cells in the
    denominator made a done thing look undone.
    """
    if fn is None or isinstance(fn, type):
        return False
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError):
        return False
    first = next(
        (
            p
            for p in signature.parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ),
        None,
    )
    if first is None:
        return False
    text = str(first.annotation).replace("'", "")
    if re.search(r"\bCallable\b", text):
        return True
    # ``skip_init(module_cls: type, *args, **kwargs)`` builds a layer; it
    # is checked by the ``nnutils`` axis and has no tensor for a finite
    # difference to move.
    return bool(re.fullmatch(r"Module|nn\.Module|Optimizer|type", text.strip()))


class Axis:
    """One question, asked of every symbol it applies to."""

    name: str = ""
    summary: str = ""
    #: Which :attr:`Symbol.kind` values this axis can express.  ``method``
    #: is in the default because ``Tensor.*`` is 253 symbols and leaving
    #: it out put every one of them outside the audit.
    kinds: frozenset[str] = frozenset({"op", "method"})

    #: Whether this axis works by *perturbing a tensor operand*.  True for
    #: the numeric sweep and false for the lifecycle axes, which take a
    #: module or a function on purpose.
    #:
    #: Load-bearing, and learned the hard way: putting the "no tensor
    #: operand, so this axis has nothing to ask" rule in the base class
    #: without this flag silently withdrew ``quantization.prepare``,
    #: ``convert``, ``fuse_modules`` and six more from the *quant* and
    #: *compiled* axes — which check them properly and take a module by
    #: design.  Nine symbols went from a verdict to none, and the
    #: coverage number went *up*, because the cells left the denominator.
    #: A refinement that improves the metric by deleting the measurement
    #: is the one shape of change this file must never make.
    varies_a_tensor: bool = True

    def applies(self, symbol: Symbol) -> bool:
        if symbol.kind not in self.kinds:
            return False
        if not symbol.inert:
            return False
        if self.varies_a_tensor and _takes_no_tensor(_surface.resolve(symbol)):
            return False
        return True

    def run(self, symbol: Symbol, ctx: Context) -> Finding:  # pragma: no cover - abstract
        raise NotImplementedError

    # ── shared helpers ───────────────────────────────────────────────────────

    def _finding(
        self, symbol: Symbol, status: Status, detail: str = "", **evidence: Any
    ) -> Finding:
        return Finding(self.name, symbol.qualname, status, detail, evidence)

    def _draws_randomly(self, fn: Any, call: Call) -> bool:
        """Whether two identical calls disagree.

        The definition of stochastic, asked rather than looked up.  It was
        looked up before — a tuple of substrings, ``rand``, ``dropout``,
        ``gumbel`` — and ``nn.init.orthogonal`` and ``nn.init.sparse``
        matched none of them while drawing a fresh random matrix on every
        call.  Four separate axes reported them for it, each measuring the
        draw rather than the op: a finite difference across two draws, a
        cpu/metal comparison across two devices that sample the one stream
        differently by design, and a strided view across a third.

        Two calls, no list, nothing to keep in sync.
        """
        # Rebuilt between calls, not reused.  An op that writes into its own
        # input makes the second call read what the first one wrote, so two
        # draws look identical and the op reads as deterministic — which
        # is what happened to ``nn.init.orthogonal`` and ``nn.init.sparse``
        # the moment they stopped replacing the impl and started writing
        # through it.  One fix disabling another is the reason this is
        # worth stating: the probe has to repeat the *experiment*, not the
        # call.
        primary = call.args[call.primary] if call.args else None
        base = None
        if hasattr(primary, "dtype") and hasattr(primary, "shape"):
            try:
                base = _probe.to_numpy(primary)
            except TypeError:
                base = None

        # Two draws can agree by chance: ``dropout3d`` drops whole channels,
        # so on a probe with two channels the masks coincide about one call
        # in four, and the op then read as deterministic for one sweep and
        # failed an axis that compares two draws.  Up to six draws make a
        # chance agreement of every one negligible.
        def draw() -> Any:
            again = call if base is None else call.with_primary(base)
            return _probe.to_numpy(fn(*again.args, **again.kwargs))

        try:
            first = draw()
            if first is None or first.dtype.kind not in "fciub":
                return False
            for _ in range(5):
                other = draw()
                if other is None or other.shape != first.shape:
                    return False
                if other.dtype.kind not in "fciub":
                    return False
                if not np.array_equal(first, other, equal_nan=first.dtype.kind in "fc"):
                    return True
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return False
        return False

    @staticmethod
    def _comparable(array: np.ndarray) -> np.ndarray:
        """An array in a form ``np.allclose`` can read.

        ``astype(float)`` on a complex array **discards the imaginary
        part**, silently.  Every complex op was being compared on half of
        its answer, so a backend could have got the imaginary part
        entirely wrong and this axis would have called it agreement.
        """
        if array.dtype.kind == "c":
            return np.stack([array.real, array.imag], axis=-1).astype(float)
        return array.astype(float)

    @staticmethod
    def _same_multiset(a: np.ndarray, b: np.ndarray) -> bool:
        """Whether the two hold the same magnitudes in a different order.

        A decomposition is not unique.  ``eigvals`` has no defined order,
        and ``svd``'s singular vectors are fixed only up to the sign of
        each column — so the two devices can return different valid
        answers to the same question, and did: Metal listed the same four
        eigenvalues as the CPU starting from a different one.

        Reported separately rather than passed, because "the same numbers
        arranged differently" is a weaker statement than agreement and the
        finding should say which one was established.
        """
        if a.size < 2 or a.shape != b.shape:
            return False
        return bool(
            np.allclose(
                np.sort(np.abs(a).reshape(-1)),
                np.sort(np.abs(b).reshape(-1)),
                rtol=2e-5,
                atol=1e-6,
                equal_nan=True,
            )
        )

    def _ignores_its_values(self, fn: Any, call: Call) -> bool:
        """Whether the answer is the same for two different inputs.

        Consulted only once the two devices have already disagreed, where
        it separates "computed something different" from "reported
        something about the device".  ``Tensor.is_metal`` is False on the
        CPU and True on Metal for every input there is, which is the
        correct answer twice rather than a defect.
        """
        try:
            base = call.base
        except TypeError:
            return False
        try:
            first = _probe.to_numpy(fn(*call.with_primary(base).args, **call.kwargs))
            other = _probe.to_numpy(fn(*call.with_primary(base * 2.0 + 1.0).args, **call.kwargs))
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return False
        if first is None or other is None or first.shape != other.shape:
            return False
        return bool(np.array_equal(first, other, equal_nan=first.dtype.kind == "f"))

    #: Reasons that mean "the question does not apply here", as opposed
    #: to "the harness could not build inputs".  The distinction is the
    #: whole value of the SKIP number: it is meant to be the list of
    #: things nobody checked, and an op with no tensor argument — every
    #: ``nn.utils`` module transform, every ``func`` higher-order
    #: function — is not one of them.  They are checked, by
    #: :mod:`~lucid.test.audit._axes_state`, and filing them under SKIP
    #: put 300-odd cells in a work queue that had already been done.
    _NOT_NUMERIC = ("no tensor operand", "nothing measurable")

    #: Reasons that mean the op has nothing this axis can measure, as
    #: opposed to the op refusing.  ``_probe.contract`` raises this when
    #: the output is a bool, a shape tuple or ``None`` — 99 cells were
    #: filed as UNSUPPORTED ("the op refused, loudly and by design")
    #: about ops that answered perfectly well in a type no derivative
    #: can be taken of.
    _NO_TENSOR_OUTPUT = "op did not return a tensor"

    def _refusal(self, symbol: Symbol, detail: str, call: Call | None = None) -> Finding:
        """UNSUPPORTED, unless the op had nothing this axis could use."""
        if (
            call is not None
            and not self._primary_is_a_tensor(call)
            and ("incompatible function arguments" in detail or "AttributeError" in detail)
        ):
            # ``zeros((2, 4))`` takes a shape and ``pad_packed_sequence``
            # takes a ``PackedSequence``; ``Call.base`` reads both as
            # arrays, so the substituting axes handed them a bare tensor
            # and were told ``incompatible function arguments`` /
            # ``'Tensor' object has no attribute 'batch_sizes'``.
            #
            # Decided here rather than up front: an op whose primary is a
            # *list* of tensors has no ``dtype`` either and substitutes
            # fine, and pre-empting cost 99 cells that were passing.
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the substituted argument is not a plain tensor — this axis "
                "has nothing it can vary here",
            )
        if self._NO_TENSOR_OUTPUT in detail:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the output is not a tensor — there is no derivative to take",
            )
        return self._finding(symbol, Status.UNSUPPORTED, detail)

    @staticmethod
    def _too_coarse_for_a_difference(call: Call) -> bool:
        """Whether the operand's precision is below what the probe needs.

        The sweep's working precision is float64 and the step is 1e-5, so
        a central difference resolves a derivative to roughly 1e-10.  On
        a float32 operand the same step leaves ~1e-3 of cancellation
        error — three orders above the tolerance.

        Consulted **only when a comparison has already failed**, and only
        to choose between FAIL and TRUNCATION.  Refusing float32 operands
        up front cost 366 cells that were passing their gradient check
        perfectly well at that precision: an op is not unmeasurable just
        because it *could* be noisy, it is unmeasurable when the noise is
        what the axis actually saw.  This is the same discipline as
        :func:`~lucid.test.audit._probe.quadratic_shrink` — interrogate
        the disagreement, do not pre-empt it.
        """
        try:
            primary = call.args[call.primary]
        except (IndexError, TypeError):
            return False
        return str(getattr(primary, "dtype", "")).endswith(("float32", "float16"))

    @staticmethod
    def _primary_is_a_tensor(call: Call) -> bool:
        """Whether the argument these axes substitute into is a tensor.

        ``zeros((2, 4))`` takes a *shape*, and ``Call.base`` reads it as
        an array because a tuple of ints is one — so the substituting
        axes replaced the shape with a NaN tensor and were told
        ``zeros(): incompatible function arguments``.  Twenty-odd cells
        filed as "the op refused" about a factory being asked to build a
        tensor out of a tensor.
        """
        try:
            primary = call.args[call.primary]
        except (IndexError, TypeError):
            return False
        return hasattr(primary, "dtype") and hasattr(primary, "shape")

    def _no_call(self, symbol: Symbol, why: str) -> Finding:
        """The finding for "no invocation ran", classified honestly."""
        if "NotImplementedError" in why:
            # The op said so itself.  ``F.fused_linear_gelu`` raises
            # ``IBackend::fused_linear_gelu`` on this backend, which is a
            # refusal by design and is exactly what UNSUPPORTED is for —
            # filing it under SKIP put "the harness could not build
            # inputs" against a call the harness built perfectly well.
            return self._finding(symbol, Status.UNSUPPORTED, why)
        if any(marker in why for marker in self._NOT_NUMERIC):
            return self._finding(symbol, Status.NOT_APPLICABLE, why)
        return self._finding(symbol, Status.SKIP, why)

    def _working_call(
        self, fn: Any, symbol: Symbol, ctx: Context
    ) -> tuple[Call, str, Any] | tuple[None, None, str]:
        """The first candidate invocation that runs, and its domain.

        Returns ``(call, domain, output)`` or ``(None, None, reason)``.

        The reason reported when nothing ran is the *derived* candidate's
        failure where there is one, not the last candidate's.  Candidates
        arrive in tier order — hand spec, then signature, then the blind
        ``op(x)`` / ``op(x, y)`` ladder — so "the last one" is always the
        ladder, and the ladder fails the same uninformative way for
        everything: ``op([x, x]): TypeError: rms_norm() missing 1
        required positional argument``.  That names a parameter the
        derivation had already filled, and sends whoever reads it to fix
        something that is not broken.  The derived candidate's failure is
        the one that says what the op actually refused.
        """
        derived: str | None = None
        first: str | None = None

        def note(call: Call, detail: str) -> None:
            nonlocal derived, first
            text = f"{call.note}: {detail}"
            if first is None:
                first = text
            if derived is None and call.note.startswith("derived from signature"):
                derived = text

        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*call.args, **call.kwargs)
                except Exception as exc:  # noqa: BLE001 - surveying, not asserting
                    note(call, f"{type(exc).__name__}: {str(exc)[:70]}")
                    continue
                if _probe.to_numpy(out) is None:
                    note(call, f"returned {type(out).__name__}, nothing measurable")
                    continue
                return call, domain, out
        return None, None, derived or first or "no candidate invocation ran"


# ── numeric axes ─────────────────────────────────────────────────────────────


class _DifferenceAxis(Axis):
    """Shared by the axes that compare against a finite difference.

    A central difference needs ``f(x+h)`` and ``f(x-h)`` to be the same
    function evaluated twice.  A stochastic op draws a fresh mask each
    call, so the quotient measures the draw rather than the derivative —
    dropout and its family reported a relative error of 1.0 for working
    exactly as designed, and ``gumbel_softmax`` did the same one axis
    over.  Stated once so the two axes cannot drift apart on it.
    """

    #: Ops whose differentiated argument must satisfy a *structural*
    #: invariant, not merely lie in a numeric range.
    #:
    #: A central difference perturbs one element at a time, and one
    #: element of an LU or LDL factor is not free: the triangle and the
    #: pivot sequence describe each other, so every probe after the first
    #: hands LAPACK a "factorization" that is not one.  ``dsytrs`` then
    #: walks off the buffer — the sweep died inside ``ldl_solve`` about
    #: one run in three, always with the heap already busy, never when
    #: the symbol was run on its own.
    #:
    #: This is not a tolerance to widen.  There is no step size at which
    #: a perturbed factor becomes a factor, so the derivative these axes
    #: measure does not exist to be compared against.
    _FACTORED_ARGUMENTS = frozenset({"LU", "LD"})

    def applies(self, symbol: Symbol) -> bool:
        if "stochastic" in symbol.flags:
            return False
        return super().applies(symbol)

    def _differentiates_a_factorization(self, fn: Any) -> bool:
        try:
            signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
        except (TypeError, ValueError, NameError):
            return False
        first = next(iter(signature.parameters), "")
        return first in self._FACTORED_ARGUMENTS

    def _differentiable_call(
        self, fn: Any, symbol: Symbol, ctx: Context
    ) -> tuple[Call, str, Any] | tuple[None, None, str]:
        """The first invocation whose analytic gradient is **finite**.

        ``_working_call`` keeps the first domain on which the *forward*
        pass runs, and for a function with a vertical tangent at its
        boundary that is not the same thing.  ``acos`` accepts the
        ``moderate`` draw — it returns NaN outside [-1, 1] without
        raising, so the forward call "works" — and then its derivative
        ``-1/sqrt(1-x^2)`` is non-finite and the axis skipped.  Twenty-nine
        ops went that way, every one of them differentiable and none of
        them gradient-checked: ``sqrt``, ``rsqrt``, ``asin``, ``log1p``,
        ``erfinv``, ``xlogy``, ``zeta`` and six Bessel functions.

        Each is finite on a domain the ladder already offers — ``acos``
        on ``unit``, ``sqrt`` on ``positive`` — so the fix is to keep
        looking rather than to widen a tolerance.  The first working call
        is still returned when no domain gives a finite gradient, so the
        report stays truthful about ops that genuinely have none.
        """
        first: tuple[Call, str, Any] | None = None
        # The same two-slot reason as ``_working_call``: the derived
        # candidate's failure beats the blind ladder's, which is always
        # ``op([x, x]): AttributeError`` and names nothing.
        derived: str | None = None
        earliest: str | None = None

        def note(call: Call, detail: str) -> None:
            nonlocal derived, earliest
            text = f"{call.note}: {detail}"
            if earliest is None:
                earliest = text
            if derived is None and call.note.startswith("derived from signature"):
                derived = text

        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, fn):
                try:
                    out = fn(*call.args, **call.kwargs)
                except Exception as exc:  # noqa: BLE001 - surveying, not asserting
                    note(call, f"{type(exc).__name__}: {str(exc)[:70]}")
                    continue
                if _probe.to_numpy(out) is None:
                    note(call, f"returned {type(out).__name__}, nothing measurable")
                    continue
                if first is None:
                    first = (call, domain, out)
                if self._gradient_is_finite(fn, call, ctx.step):
                    return call, domain, out
        if first is not None:
            return first
        return None, None, derived or earliest or "no candidate invocation ran"

    @staticmethod
    def _gradient_is_finite(fn: Any, call: Call, step: float = 1e-5) -> bool:
        """Whether *both* methods can be evaluated on this domain.

        The analytic gradient being finite is half of it.  A central
        difference also evaluates ``f(x ± h)``, and for a function with a
        boundary in the probe's range that is a step outside the domain:
        ``log`` on a draw that reaches 0, ``atanh`` on one that reaches
        1.  Eleven ops picked a domain their gradient survived and their
        *difference* did not, and reported "the finite difference left
        the op's domain" — a true statement about a domain the ladder
        had a better one for two entries down.
        """
        try:
            probe = call.with_primary(call.base)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            out = fn(*probe.args, **probe.kwargs)
            _probe.contract(out, _probe.covector(64, _probe.SEED_A)).backward()
        except Exception:  # noqa: BLE001 - the caller reports the real failure
            return False
        if x.grad is None:
            return False
        try:
            if not np.isfinite(np.asarray(x.grad.numpy(), dtype=np.float64)).all():
                return False
            base = np.asarray(call.base, dtype=np.float64)
            for shifted in (base + step, base - step):
                moved = call.with_primary(shifted)
                value = _probe.to_numpy(fn(*moved.args, **moved.kwargs))
                if value is None or not np.isfinite(value.astype(float)).all():
                    return False
        except Exception:  # noqa: BLE001
            return False
        return True


#: Float-valued ops whose output rightly carries no gradient to the
#: argument the gradient axis varies, each with the reason.
#:
#: Anything else that returns a float tensor detached from an input that
#: requires grad is a defect: a loss through it trains nothing.  Until
#: 2026-10-02 this axis filed every such case under UNSUPPORTED — "no
#: gradient reached the input" — and nine differentiable ops hid among the
#: comparisons and factories (``masked_select``, ``F.ctc_loss``,
#: ``linalg.lstsq`` / ``lu`` / ``lu_factor`` / ``householder_product`` /
#: ``eigvals``, ``pinv``, ``parameters_to_vector``).  An entry here has
#: to say why the reference framework does not differentiate it either.
#: Integer, bool and non-tensor outputs need no entry.
NOT_DIFFERENTIABLE: dict[str, str] = {
    "Tensor.data": "a detached alias by definition",
    "Tensor.detach": "detached by definition",
    "lucid.detach": "detached by definition",
    "Tensor.new_empty": "a new tensor that reads only shape, dtype and device",
    "Tensor.new_full": "a new tensor that reads only shape, dtype and device",
    "Tensor.new_ones": "a new tensor that reads only shape, dtype and device",
    "Tensor.new_tensor": "copies the given data into a new leaf",
    "Tensor.new_zeros": "a new tensor that reads only shape, dtype and device",
    "Tensor.histc": "a count per bin: piecewise constant, no derivative in the reference",
    "lucid.histc": "a count per bin: piecewise constant, no derivative in the reference",
    "lucid.histogram2d": "counts per bin: piecewise constant, no derivative in the reference",
    "lucid.histogramdd": "counts per bin: piecewise constant, no derivative in the reference",
    "lucid.arange": "a factory: its arguments are endpoints and a step",
    "lucid.eye": "a factory: its arguments are sizes",
    "lucid.linspace": "a factory: its arguments are endpoints and a count",
    "lucid.logspace": "a factory: its arguments are endpoints and a count",
    "lucid.empty_like": "reads only the input's shape, dtype and device",
    "lucid.full_like": "reads only the input's shape, dtype and device",
    "lucid.ones_like": "reads only the input's shape, dtype and device",
    "lucid.zeros_like": "reads only the input's shape, dtype and device",
    "lucid.fft.fftfreq": "sample frequencies of a length, not a function of data",
    "lucid.fft.rfftfreq": "sample frequencies of a length, not a function of data",
    "lucid.from_dlpack": "constructs a new leaf from foreign memory",
    "lucid.from_numpy": "constructs a new leaf from an array",
    "lucid.tensor": "constructs a new leaf from data",
    "lucid.heaviside": "a step function the reference defines no derivative for",
    "lucid.nn.utils.clip_grad_norm": "rescales .grad in place; the norm is bookkeeping",
    "lucid.nn.utils.get_total_norm": "a norm of .grad tensors, read after backward",
    "lucid.utils.transforms.functional.posterize": "bit-level quantisation of pixel values",
    "lucid.utils.transforms.functional.rotation_matrix": (
        "built from Python floats — an angle, a centre and a scale"
    ),
    "F.accuracy": "a share of correct predictions: piecewise constant",
    "F.correct_count": "a count of correct predictions: piecewise constant",
    "F.straight_through": (
        "the gradient goes to ``soft``, the second argument; the probe varies "
        "``hard``, the first, which the estimator deliberately ignores"
    ),
}

#: Window functions take a length and shape parameters, never data.
_NOT_DIFFERENTIABLE_PREFIXES = ("lucid.signal.windows.",)


def _untracked_output(produced: Any) -> str | None:
    """How an op's output fails to carry a gradient, or None when it does.

    ``"integer"`` when every tensor it returned is integer or bool — there
    is no derivative to take — and ``"float"`` when a floating output came
    back detached although the input required grad.
    """
    outs = produced if isinstance(produced, (tuple, list)) else (produced,)
    tensors = [o for o in outs if hasattr(o, "requires_grad") and hasattr(o, "dtype")]
    if not tensors or any(t.requires_grad for t in tensors):
        return None
    if all(not (t.is_floating_point() or t.is_complex()) for t in tensors):
        return "integer"
    return "float"


def _why_not_differentiable(qualname: str) -> str | None:
    """The recorded reason ``qualname`` has no derivative, if it has one."""
    qualname = _surface.recorded_as(qualname)
    if qualname in NOT_DIFFERENTIABLE:
        return NOT_DIFFERENTIABLE[qualname]
    if qualname.startswith(_NOT_DIFFERENTIABLE_PREFIXES):
        return "a window of a length and shape parameters, not a function of data"
    return None


class GradientAxis(_DifferenceAxis):
    """Analytic gradient against central finite differences, in float64.

    Found: seven in-place activations returning the pre-activation
    gradient, because they were built on a primitive that documents it
    does not extend the graph.
    """

    name = "grad"
    summary = "d/dx vs central finite differences (float64)"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        if self._differentiates_a_factorization(fn):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the differentiated argument must be a valid factorization — "
                "perturbing one element of it leaves the op's domain entirely",
            )
        call, domain, first = self._differentiable_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, str(first))
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )

        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "nothing to differentiate — the operand is not a tensor",
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def scalar(array: np.ndarray) -> float:
            probe = call.with_primary(array)
            return float(_probe.contract(fn(*probe.args, **probe.kwargs), weights))

        # analytic
        #
        # Re-drawn if the gradient comes back identically zero, because
        # that is usually a fact about the sample rather than the op.
        # ``maximum(a, b)`` differentiates to zero wherever ``b`` wins, so
        # a companion that happens to dominate every element leaves
        # nothing for the finite difference to confirm — which is what
        # the two draws of ``moderate`` did here, all six elements of one
        # above all six of the other.  That is a 1-in-64 accident and not
        # a property of ``maximum``, so reporting it as vacuous told the
        # reader about the seed.
        #
        # An op whose gradient really is zero everywhere — ``floor``,
        # ``sign`` — stays vacuous after every retry, and then the verdict
        # is about the op.
        for attempt in range(4):
            if attempt:
                try:
                    base = _probe.sample(domain, base.shape, attempt)
                except Exception:  # noqa: BLE001 - the first draw stands
                    break
            probe = call.with_primary(base)
            x = probe.args[probe.primary]
            try:
                x.requires_grad_(True)
                produced = fn(*probe.args, **probe.kwargs)
                loss = _probe.contract(produced, weights)
                loss.backward()
            except Exception:  # noqa: BLE001 - reported by the real pass below
                break
            if x.grad is None:
                break
            probe_grad = np.asarray(x.grad.numpy(), dtype=np.float64)
            if np.abs(probe_grad).max(initial=0.0) != 0.0:
                break

        probe = call.with_primary(base)
        x = probe.args[probe.primary]
        returned_itself = False
        try:
            x.requires_grad_(True)
            produced = fn(*probe.args, **probe.kwargs)
            # Read here, not after the backward pass: contracting the
            # output against the covector puts a node on ``x`` itself when
            # the op returned ``x``, so asking later always answers "it
            # has one".
            returned_itself = produced is x and x._impl.grad_fn is None
            untracked = _untracked_output(produced)
            if untracked is None:
                loss = _probe.contract(produced, weights)
                loss.backward()
        except Exception as exc:  # noqa: BLE001
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:70]}", call)
        if untracked == "integer":
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the output is integer or bool — there is no derivative to take",
            )
        if untracked == "float":
            why = _why_not_differentiable(symbol.qualname)
            if why is not None:
                return self._finding(symbol, Status.UNSUPPORTED, why)
            return self._finding(
                symbol,
                Status.FAIL,
                "the input requires grad but the float output carries no graph "
                "— a loss through this op trains nothing",
            )
        if x.grad is None:
            return self._finding(symbol, Status.UNSUPPORTED, "no gradient reached the input")

        # An op that overwrote its input is asking two different
        # questions of the two methods.  ``nn.init.eye(x)`` writes into
        # ``x`` and leaves it a leaf, so the finite difference measures a
        # function of the *old* values — correctly zero, since the
        # answer does not depend on them — while the analytic pass
        # differentiates the tensor that is now the result.  There is no
        # change to the framework that reconciles those; they are
        # derivatives of different maps.
        #
        # Ops that mutate and *do* leave a node behind are unaffected and
        # still checked: ``exp_`` adopts one whose saved input is the
        # pre-write snapshot, so both methods see the same function.
        # Compared by identity rather than by value: the invocation search
        # has already run the op once on these very arguments, so an
        # in-place one has *already* written its answer and a second call
        # changes nothing.  ``init.eye`` looked like it left its input
        # alone because the input was an identity matrix by the time this
        # ran.
        if returned_itself:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "overwrites its input and leaves no node — the difference and "
                "the analytic pass differentiate different maps",
            )

        analytic = np.asarray(x.grad.numpy(), dtype=np.float64).reshape(-1)
        if not np.isfinite(analytic).all():
            return self._finding(symbol, Status.SKIP, f"non-finite gradient on '{domain}'")
        if np.abs(analytic).max(initial=0.0) == 0.0:
            return self._finding(
                symbol,
                Status.VACUOUS,
                f"gradient is identically zero on '{domain}' — this check could not fail",
            )

        try:
            coarse = _probe.finite_difference(scalar, base, ctx.step)
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.SKIP, f"fd failed: {type(exc).__name__}")

        # A non-finite difference quotient is the probe leaving the op's
        # domain, not a wrong derivative.  ``log`` and ``sqrt`` are only
        # defined on part of the line, and perturbing the probe by h walks
        # off it — the analytic gradient was finite and checked, the
        # numerical one came back nan, and the comparison reported a
        # defect in nineteen ops that were computing correctly.
        if not np.isfinite(coarse).all():
            return self._finding(
                symbol, Status.SKIP, f"{domain}: the finite difference left the op's domain"
            )

        rel = _probe.relative(analytic, coarse.reshape(analytic.shape))
        if rel < ctx.tolerance:
            return self._finding(symbol, Status.PASS, f"{domain}: rel {rel:.2e}", rel=rel)

        # Interrogate before reporting.  Truncation falls like h**2.
        try:
            fine = _probe.finite_difference(scalar, base, ctx.step / 10.0)
        except Exception:  # noqa: BLE001
            fine = coarse
        rel_fine = _probe.relative(analytic, fine.reshape(analytic.shape))
        if _probe.quadratic_shrink(rel, rel_fine):
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} -> {rel_fine:.2e} at h/10 — the probe, not the op",
                rel=rel,
                rel_refined=rel_fine,
            )
        # A function of a *constrained* input cannot be differenced
        # coordinate by coordinate.
        #
        # ``eigvalsh`` reads one triangle and takes the matrix to be
        # symmetric, so perturbing an entry in the other triangle changes
        # nothing at all — while its gradient is stated symmetrically and
        # puts half the sensitivity on each of the mirrored pair.  Six of a
        # 4x4's sixteen coordinates come back with a difference of exactly
        # zero against a non-zero derivative, and the rest off by the
        # factor of two that splitting implies.  Lucid's gradient agrees
        # with the reference to the last digit; it is the probe that has
        # stepped off the symmetric matrices, where the function is not
        # defined.
        untouched = (np.abs(coarse.reshape(-1)) < 1e-12) & (np.abs(analytic) > 1e-6)
        if untouched.any():
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: {int(untouched.sum())} of {analytic.size} coordinates do "
                "not move the result at all — the input is constrained and a "
                "coordinate-wise difference leaves its domain",
                rel=rel,
            )

        # A corner is not a wrong derivative.
        #
        # Where the function has a kink at the probe point, the two
        # one-sided slopes differ and the central difference reports their
        # average, which matches neither.  Nothing about the op is wrong
        # and no step size helps: ``cosine_embedding_loss`` is a hinge, and
        # its analytic gradient agrees with the reference exactly while the
        # difference straddles the corner.
        #
        # Specific, because a *wrong* derivative still has the two
        # one-sided slopes agreeing with each other — they are both
        # measuring the same function.  Only a corner separates them.
        try:
            flat = base.reshape(-1)
            at = scalar(base)
            forward = np.empty(flat.size, dtype=np.float64)
            backward = np.empty(flat.size, dtype=np.float64)
            for i in range(flat.size):
                up, down = flat.copy(), flat.copy()
                up[i] += ctx.step
                down[i] -= ctx.step
                forward[i] = (scalar(up.reshape(base.shape)) - at) / ctx.step
                backward[i] = (at - scalar(down.reshape(base.shape))) / ctx.step
            sided = _probe.relative(forward, backward)
        except Exception:  # noqa: BLE001
            sided = 0.0
        if sided > 1e-3 and np.isfinite(sided):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: the one-sided slopes differ by {sided:.2e} — a corner "
                "at the probe, where a central difference is the average of two",
                rel=rel,
                one_sided=sided,
            )

        # Refining made it worse or left it alone.  Before calling that a
        # wrong derivative, try the other direction: a step *below the
        # output's resolution* cannot be refined into agreement, only
        # coarsened.  ``Tensor.half`` casts to float16, where 1e-5 does not
        # move a value near 1 at all — ``f(x+h)`` and ``f(x-h)`` round
        # to the same number and the difference is exactly zero, against an
        # analytic gradient of 1.  Coarsening recovers it; a genuinely
        # wrong formula does not move.
        try:
            blunt = _probe.finite_difference(scalar, base, ctx.step * 1000.0)
            rel_blunt = _probe.relative(analytic, blunt.reshape(analytic.shape))
        except Exception:  # noqa: BLE001
            rel_blunt = rel
        # Judged by how fast it falls, not against the tolerance.  float16
        # is coarse enough that no step reaches 2e-5 — the point is that
        # the disagreement collapses as the step *grows*, which is the
        # opposite of truncation and the signature of a step below the
        # output's resolution.
        if _probe.quadratic_shrink(rel, rel_blunt):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"{domain}: rel {rel:.2e} at h, {rel_blunt:.2e} at 1000h — the step "
                "is below the output's precision, not the derivative wrong",
                rel=rel,
                rel_coarsened=rel_blunt,
            )
        if self._too_coarse_for_a_difference(call) and rel < 1e-1:
            # A float32 operand and a disagreement inside float32's own
            # cancellation error.  TRUNCATION is what this status is
            # for: the probe was the limit, not the op.
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} on a float32 operand — inside the "
                "cancellation error of a central difference at this step",
                rel=rel,
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e}, still {rel_fine:.2e} at h/10",
            rel=rel,
            rel_refined=rel_fine,
            analytic=analytic[:8].tolist(),
            finite_difference=coarse[:8].tolist(),
        )


def _fold_onto_read_triangle(
    analytic: np.ndarray, fd: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray | None:
    """The symmetric gradient of a one-triangle input, as that triangle sees it.

    ``None`` unless the input is square and one strict triangle's
    differences are exactly zero — the op never reads it.  Then each read
    off-diagonal coordinate carries its own and its mirror's share, the
    diagonal its own, and the unread triangle nothing.
    """
    if len(shape) < 2 or shape[-1] != shape[-2]:
        return None
    n = shape[-1]
    a = analytic.reshape(shape)
    d = fd.reshape(shape)
    diagonal = np.eye(n, dtype=bool)
    mirrored = a + np.swapaxes(a, -1, -2)
    for read in (np.tril(np.ones((n, n), bool), -1), np.triu(np.ones((n, n), bool), 1)):
        unread = ~(read | diagonal)
        if np.all(d[..., unread] == 0.0):
            return np.where(read, mirrored, np.where(diagonal, a, 0.0)).reshape(-1)
    return None


class SecondGradientAxis(_DifferenceAxis):
    """Second derivative against finite differences of the first.

    Found: ``prod`` / ``max`` / ``min`` returning the incoming seed under
    ``create_graph=True``, because the reduction base class applies sum's
    rule and only two of five reductions overrode it.
    """

    name = "grad2"
    summary = "d2/dx2 vs finite differences of the analytic gradient"

    def _unreachable_second(
        self,
        symbol: Symbol,
        directional: Callable[[np.ndarray], tuple[Any, Any, Any]],
        base: np.ndarray,
        ctx: Context,
        detail: str,
        call: Call,
    ) -> Finding:
        """Judge a first gradient with no path back to its input."""
        try:
            fd = _probe.finite_difference(lambda a: float(directional(a)[1]), base, ctx.step)
        except Exception:  # noqa: BLE001
            return self._refusal(symbol, detail, call)
        if not np.isfinite(fd).all():
            return self._finding(symbol, Status.SKIP, "the second difference left the op's domain")
        largest = float(np.abs(fd).max(initial=0.0))
        if largest == 0.0:
            return self._finding(
                symbol,
                Status.PASS,
                "second derivative is zero: the gradient does not depend on the "
                "input, and create_graph=True rightly leaves no path to it",
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"create_graph=True returned a gradient detached from the input, "
            f"but the gradient moves with it (second difference up to "
            f"{largest:.2e}) — double backward is missing",
            second_difference=fd[:8].tolist(),
        )

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        if self._differentiates_a_factorization(fn):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the differentiated argument must be a valid factorization — "
                "perturbing one element of it leaves the op's domain entirely",
            )
        call, domain, why = self._differentiable_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )

        try:
            base = call.base
        except TypeError:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "nothing to differentiate — the operand is not a tensor",
            )

        w1 = _probe.covector(64, _probe.SEED_A)

        def directional(array: np.ndarray) -> tuple[Any, Any]:
            probe = call.with_primary(array)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            loss = _probe.contract(fn(*probe.args, **probe.kwargs), w1)
            (g,) = lucid.autograd.grad(loss, [x], create_graph=True)
            n = int(g.reshape(-1).shape[0])
            # As long as the gradient: a fixed 64 was shorter than any
            # operand over 64 elements, and ``w2[:n]`` then failed the
            # multiply — 49 cells reported as the op's refusal.
            w2 = _probe.covector(n, _probe.SEED_B)
            return x, (g.reshape(-1) * _probe.as_f64(w2)).sum(), g

        try:
            x, scalar, first = directional(base)
            first_scale = float(np.abs(np.asarray(first.numpy(), dtype=np.float64)).max())
        except Exception as exc:  # noqa: BLE001
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:80]}", call)
        try:
            (second,) = lucid.autograd.grad(scalar, [x])
            analytic = np.asarray(second.numpy(), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {str(exc)[:80]}"
            if "not reachable" not in str(exc):
                return self._refusal(symbol, detail, call)
            # The first gradient came back with no path to the input.  That
            # is right exactly when it does not depend on the input — a
            # linear op, or a piecewise-constant gradient (sum, max, relu)
            # away from its kinks — and then its difference is zero to the
            # bit.  Otherwise create_graph=True returned a detached
            # gradient: a second derivative asked for with
            # ``allow_unused=True`` comes back as zero, silently.  Every
            # one of these was once filed as the op's refusal, 351 cells,
            # the right answers and the wrong ones together.
            return self._unreachable_second(symbol, directional, base, ctx, detail, call)

        try:
            fd = _probe.finite_difference(lambda a: float(directional(a)[1]), base, ctx.step)
        except Exception as exc:  # noqa: BLE001
            return self._finding(symbol, Status.SKIP, f"fd failed: {type(exc).__name__}")

        if np.abs(analytic).max(initial=0.0) == 0.0 and np.abs(fd).max(initial=0.0) == 0.0:
            return self._finding(
                symbol, Status.PASS, f"{domain}: second derivative is zero (op is linear)"
            )
        # Zero, but not to the bit: both sides are rounding residue of a
        # derivative that vanishes, and their ratio is noise against noise.
        # adjust_hue — a round trip whose pieces are each exact — came back
        # "rel 1.00" from 1e-10 against 3e-9, the first cell of its kind
        # once the covector stopped truncating larger operands.
        negligible = 1e-6 * max(1.0, first_scale)
        if max(np.abs(analytic).max(initial=0.0), np.abs(fd).max(initial=0.0)) <= negligible:
            return self._finding(
                symbol,
                Status.PASS,
                f"{domain}: second derivative is zero to the difference's resolution",
            )

        # Same guard the first-derivative axis carries: differencing the
        # gradient twice steps 2h away from the probe, and ``acos``,
        # ``asin``, ``sqrt`` and ``erfinv`` are only defined on part of the
        # line.  Both the analytic second derivative and the difference
        # come back nan, which is agreement the comparison cannot express —
        # ``nan != nan`` — and it was reported as a disagreement in
        # nineteen ops that were computing correctly.
        if not np.isfinite(fd).all() or not np.isfinite(analytic).all():
            return self._finding(
                symbol, Status.SKIP, f"{domain}: the second difference left the op's domain"
            )

        rel = _probe.relative(analytic, fd.reshape(analytic.shape))
        if rel < 1e-4:
            return self._finding(symbol, Status.PASS, f"{domain}: rel {rel:.2e}", rel=rel)
        # A symmetric input read from one triangle — ``eigh``,
        # ``eigvalsh``.  The op never sees the other triangle, so those
        # coordinates difference to exactly zero, while the gradient is
        # stated symmetrically, half the sensitivity on each of a mirrored
        # pair: rel 0.5 between two right answers.  The first-derivative
        # axis sets such inputs aside; here the analytic answer is folded
        # onto the triangle the op reads and compared there, so it is still
        # checked.
        folded = _fold_onto_read_triangle(analytic, fd, tuple(np.shape(base)))
        if folded is not None:
            rel_folded = _probe.relative(folded, fd.reshape(folded.shape))
            if rel_folded < 1e-4:
                return self._finding(
                    symbol,
                    Status.PASS,
                    f"{domain}: rel {rel_folded:.2e} on the triangle the symmetric "
                    "input is read from",
                    rel=rel_folded,
                )

        fine: np.ndarray | None = None
        try:
            fine = _probe.finite_difference(
                lambda a: float(directional(a)[1]), base, ctx.step / 10.0
            )
            rel_fine = _probe.relative(analytic, fine.reshape(analytic.shape))
        except Exception:  # noqa: BLE001
            rel_fine = rel
        if _probe.quadratic_shrink(rel, rel_fine):
            return self._finding(
                symbol, Status.TRUNCATION, f"{domain}: rel {rel:.2e} -> {rel_fine:.2e} at h/10"
            )
        if self._too_coarse_for_a_difference(call) and fine is not None:
            # Rounding, not truncation: a float32 first derivative carries
            # ~1e-7 of its own size as noise, and differencing it divides
            # that by h — so the difference *grows* as h shrinks, where a
            # real second derivative holds still.  adjust_hue: analytic
            # 1e-7 (it is zero), differences 0.09 at h and ten times that
            # at h/10.
            coarse_size = float(np.abs(fd).max(initial=0.0))
            fine_size = float(np.abs(fine).max(initial=0.0))
            if coarse_size > 0.0 and fine_size >= 3.0 * coarse_size:
                return self._finding(
                    symbol,
                    Status.TRUNCATION,
                    f"{domain}: the difference grows as h shrinks ({coarse_size:.2e} -> "
                    f"{fine_size:.2e}) — float32 rounding, not a derivative",
                    rel=rel,
                )
        if self._too_coarse_for_a_difference(call) and rel < 1e-1:
            # A float32 operand and a disagreement inside float32's own
            # cancellation error.  TRUNCATION is what this status is
            # for: the probe was the limit, not the op.
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} on a float32 operand — inside the "
                "cancellation error of a central difference at this step",
                rel=rel,
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e}, still {rel_fine:.2e} at h/10",
            rel=rel,
            analytic=analytic[:8].tolist(),
            finite_difference=fd[:8].tolist(),
            first_derivative_scale=first_scale,
        )


class CreateGraphAxis(Axis):
    """``autograd.grad(create_graph=True)`` against ``backward()``.

    The two must agree on the *first* derivative whatever the graph mode.
    This is what caught the reductions: ``backward()`` was right and
    ``grad(create_graph=True)`` returned the seed, silently.
    """

    name = "creategraph"
    summary = "autograd.grad(create_graph=True) vs backward()"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )
        try:
            values = _probe.to_numpy(call.args[call.primary])
            if values is None:
                raise TypeError("no array-valued primary operand")
            base = np.asarray(values)
        except (TypeError, IndexError, AttributeError):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "nothing to differentiate — the operand is not a tensor",
            )

        weights = _probe.covector(64, _probe.SEED_A)

        def loss_of(array: np.ndarray) -> tuple[Any, Any]:
            probe = call.with_primary(array)
            x = probe.args[probe.primary]
            x.requires_grad_(True)
            return x, _probe.contract(fn(*probe.args, **probe.kwargs), weights)

        # Re-drawn on an all-zero reference, for the same reason the grad
        # axis re-draws: a gradient that is zero everywhere leaves the
        # two routes nothing to disagree about, and for a comparison op
        # that is usually the companion having won every element rather
        # than anything about the op.  An op whose gradient really is
        # zero stays zero through every draw.
        reference = np.zeros(0)
        for attempt in range(4):
            if attempt:
                try:
                    base = _probe.sample(domain, base.shape, attempt)
                except Exception:  # noqa: BLE001 - the first draw stands
                    break
            try:
                x_ref, loss = loss_of(base)
                loss.backward()
            except Exception as exc:  # noqa: BLE001
                return self._refusal(
                    symbol, f"backward(): {type(exc).__name__}: {str(exc)[:60]}", call
                )
            # Asked, not assumed.  Reading ``.grad.numpy()`` on an op
            # that received no gradient raised ``AttributeError:
            # 'NoneType' object has no attribute 'numpy'`` — 135 cells
            # reporting a harness traceback where the finding is the
            # plain fact that nothing reached the input, which this axis
            # has a proper answer for.
            if x_ref.grad is None:
                return self._finding(
                    symbol,
                    Status.UNSUPPORTED,
                    "no gradient reached the input — nothing for the two "
                    "routes to disagree about",
                )
            reference = self._gradient_lanes(x_ref.grad)
            if np.abs(reference).max(initial=0.0) != 0.0:
                break
        if np.abs(reference).max(initial=0.0) == 0.0:
            return self._finding(symbol, Status.VACUOUS, "reference gradient is identically zero")

        try:
            x_probe, loss2 = loss_of(base)
            (got,) = lucid.autograd.grad(loss2, [x_probe], create_graph=True)
            candidate = self._gradient_lanes(got)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"grad(create_graph): {type(exc).__name__}"
            )

        # Two routes that both produce NaN agree; the comparison does not
        # know that, because ``nan != nan``.  ``acos``, ``asin``, ``sqrt``
        # and ``erfinv`` are only defined on part of the line, and where
        # the probe leaves it *both* backward() and grad(create_graph=True)
        # answer nan — correctly and identically — and the relative error
        # came out nan and was reported as a disagreement in 26 ops.
        finite = np.isfinite(reference) & np.isfinite(candidate)
        if not finite.any():
            return self._finding(
                symbol,
                Status.SKIP,
                f"{domain}: both routes are non-finite here — outside the op's domain",
            )
        if not finite.all():
            if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{domain}: the two routes disagree about which entries are finite",
                    backward=reference[:8].tolist(),
                    create_graph=candidate[:8].tolist(),
                )
            reference, candidate = reference[finite], candidate[finite]

        rel = _probe.relative(reference, candidate)
        if rel < 1e-9:
            return self._finding(symbol, Status.PASS, f"{domain}: rel {rel:.2e}")
        # Two routes through the *same* backward should agree exactly, so
        # the tolerance here is 1e-9 — which is below float32 epsilon.  On
        # a float32 operand a 1e-7 disagreement is the dtype, not the two
        # routes diverging.
        if self._too_coarse_for_a_difference(call) and rel < 1e-5:
            return self._finding(
                symbol,
                Status.TRUNCATION,
                f"{domain}: rel {rel:.2e} on a float32 operand — at the dtype's "
                "own resolution, not a disagreement between the two routes",
                rel=rel,
            )
        return self._finding(
            symbol,
            Status.FAIL,
            f"{domain}: rel {rel:.2e} between backward() and grad(create_graph=True)",
            backward=reference[:8].tolist(),
            create_graph=candidate[:8].tolist(),
        )

    @staticmethod
    def _gradient_lanes(gradient: Any) -> np.ndarray:
        values = np.asarray(gradient.numpy())
        if np.iscomplexobj(values):
            return np.ascontiguousarray(values, dtype=np.complex128).reshape(-1).view(np.float64)
        return values.astype(np.float64).reshape(-1)


def _receiver_position(free_fn: Any, method_fn: Any) -> int:
    """Which of the free function's arguments the method's receiver is.

    Almost always the first — ``x.exp()`` is ``lucid.exp(x)`` — and the
    axis assumed it always.  ``where`` is the exception:
    ``lucid.where(condition, x, y)`` is spelled ``x.where(condition, y)``,
    so the receiver is the *second* free argument.  Passing the arguments
    positionally to both asked two different questions, and the axis
    reported the two different answers as a defect.  The reference has
    the same asymmetry, so this was never Lucid's to fix.

    Derived rather than listed: the method names every free parameter it
    still takes, so the one it does not name is the one it became.
    """
    try:
        free_names = [
            name
            for name, param in inspect.signature(free_fn).parameters.items()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        method_params = inspect.signature(method_fn).parameters
        method_names = set(method_params)
    except (TypeError, ValueError, NameError):
        return 0
    method_names.discard("self")

    # A registry-injected method is ``(self, *args, **kwargs)``: it names
    # nothing, so *every* free parameter looks unmatched and the
    # derivation below reads that as "the names did not settle it".
    #
    # Until these signatures could be read at all, ``inspect.signature``
    # raised and the except above returned 0 — the right answer, reached
    # by accident. Repairing the annotations turned that accident into
    # ``-1`` for any op with three arguments or more, and 69 entry-axis
    # cells went from pass to unanswered. Say the same thing on purpose.
    if not method_names - {"args", "kwargs"}:
        return 0
    missing = [i for i, name in enumerate(free_names) if name not in method_names]
    if len(missing) == 1:
        return missing[0]
    # The names did not settle it.  With two arguments position 0 is the
    # only reading that does not have the method reversing its operands,
    # which no API does — and that assumption is what found the
    # scalar-coercion gap this axis exists for.  With three there is a
    # real choice and ``where`` makes it: ``lucid.where(cond, x, y)`` is
    # ``x.where(condition, y)``, the receiver in the middle.  Nothing here
    # can tell which, so it says so instead of guessing.
    return 0 if len(free_names) <= 2 else -1


class EntryPointAxis(Axis):
    """The same op through every spelling it has.

    Found: scalar coercion existed on the operator path only, so
    ``x ** 2`` worked and ``x.pow(2)`` raised, and ``int32 + 1.5``
    silently truncated.
    """

    name = "entry"
    summary = "lucid.f(x) vs x.f() vs F.f(x)"

    #: Re-seeded before every spelling of a stochastic op.
    _SEED = 1415

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        routes = list(_surface.counterparts(symbol))
        if len(routes) < 2:
            return self._finding(symbol, Status.NOT_APPLICABLE, "only one entry point to compare")

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        # The first route's callable is what the signature tier reads.
        # Every route is the same operation, so any of them describes it.
        probe_fn = routes[0][1]
        # Each spelling of a sampler is one draw.  Unseeded, two of them
        # are two independent samples compared as one answer, and
        # ``lucid.bernoulli`` disagreed with ``Tensor.bernoulli`` for being
        # random.  Under one seed the same op draws the same numbers.
        stochastic = "stochastic" in symbol.flags
        for domain in ctx.domains:
            for call in _specs.invocations(symbol.short, domain, symbol.qualname, probe_fn):
                results.clear()
                errors.clear()
                for label, fn in routes:
                    at = _receiver_position(probe_fn, fn) if label == "method" else 0
                    if at < 0:
                        # The two spellings name their arguments
                        # differently and there is more than one candidate
                        # for the receiver, so there is no alignment to
                        # compare — ``lucid.where(cond, x, y)`` against
                        # ``Tensor.where(self, condition, other)`` shares
                        # not one parameter name.  Guessing position 0 made
                        # the axis call ``where(cond, x, y)`` against
                        # ``cond.where(x, y)``, which are two different
                        # questions, and report the two answers.
                        return self._finding(
                            symbol,
                            Status.NOT_APPLICABLE,
                            "the spellings order their arguments differently and "
                            "share no parameter name — no alignment to compare",
                        )
                    # Operands of its own per spelling, or an in-place one
                    # hands the next spelling what it wrote.
                    own = call.fresh()
                    args = own.args if label != "method" else own.args[:at] + own.args[at + 1 :]
                    target = own.args[at] if label == "method" else None
                    if stochastic:
                        lucid.manual_seed(self._SEED)
                    try:
                        out = (
                            fn(target, *args, **own.kwargs)
                            if label == "method"
                            else fn(*own.args, **own.kwargs)
                        )
                        got = _probe.to_numpy(out)
                        if got is None:
                            errors[label] = "no tensor"
                        else:
                            results[label] = got
                    except Exception as exc:  # noqa: BLE001
                        errors[label] = type(exc).__name__
                if len(results) >= 2:
                    break
            if len(results) >= 2:
                break

        if len(results) < 2:
            if results and errors:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"reachable one way but not another: ok={sorted(results)} " f"failed={errors}",
                )
            return self._finding(symbol, Status.SKIP, f"no shared invocation: {errors}")

        labels = sorted(results)
        first = results[labels[0]]
        for label in labels[1:]:
            other = results[label]
            if first.shape != other.shape:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{labels[0]} gives {first.shape}, {label} gives {other.shape}",
                )
            if not np.allclose(first, other, rtol=1e-9, atol=1e-12, equal_nan=True):
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"{labels[0]} and {label} disagree by "
                    f"{np.abs(first.astype(float) - other.astype(float)).max():.3e}",
                )
        return self._finding(symbol, Status.PASS, f"{len(labels)} entry points agree")


class DeviceAxis(Axis):
    """CPU against Metal — including a non-finite probe.

    Found: ``relu(NaN)`` returned 0 on the CPU and NaN on Metal.  The
    existing parity sweeps missed it because they probe with
    well-conditioned uniform data, which is why this axis runs the
    non-finite vector as well.
    """

    name = "device"
    summary = "cpu vs metal, finite and non-finite inputs"

    def applies(self, symbol: Symbol) -> bool:
        return super().applies(symbol) and "stochastic" not in symbol.flags

    @staticmethod
    def _takes_device_argument(fn: Any) -> bool:
        """Whether the callable has a ``device`` parameter of its own."""
        try:
            return (
                "device"
                in inspect.signature(
                    fn, annotation_format=annotationlib.Format.FORWARDREF
                ).parameters
            )
        except Exception:  # noqa: BLE001
            return False

    def _device_kwarg_check(self, symbol: Symbol, fn: Any, call: Any) -> Finding:
        """Does a factory put its output where ``device=`` says?

        ``zeros``, ``arange``, ``eye``, ``linspace`` and the signal
        windows build a tensor from nothing, so there is no input device
        for the output to follow and the cpu-vs-metal comparison has
        nothing to compare.  They do take a ``device`` argument, though,
        and whether they honour it is a real question that was going
        unasked — twenty-odd symbols reported as skips.
        """
        try:
            parameters = inspect.signature(
                fn, annotation_format=annotationlib.Format.FORWARDREF
            ).parameters
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.NOT_APPLICABLE, "builds from no tensor input")
        if "device" not in parameters:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "builds from no tensor input and takes no device argument",
            )
        try:
            out = fn(*call.args, **{**call.kwargs, "device": "metal"})
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.UNSUPPORTED, f"device='metal': {type(exc).__name__}"
            )
        device = getattr(out, "device", None)
        if device is None:
            return self._finding(symbol, Status.SKIP, "returned no tensor")
        if "metal" not in str(device):
            return self._finding(
                symbol, Status.FAIL, f"asked for device='metal' and built the result on {device}"
            )
        return self._finding(symbol, Status.PASS, "honours its device argument")

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        if not ctx.metal:
            return self._finding(symbol, Status.SKIP, "metal unavailable")
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, domain, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        if self._draws_randomly(fn, call):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "two identical calls disagree — this measures the draw, not the op",
            )

        moved_any = False

        def on(device: str, override: np.ndarray | None = None) -> Any:
            nonlocal moved_any
            # Each device gets operands of its own: ``.to("cpu")`` of a CPU
            # tensor is that tensor, so an in-place op run on the CPU first
            # would hand Metal what it wrote.
            own = call.fresh()
            args = []
            for i, a in enumerate(own.args):
                if hasattr(a, "to"):
                    moved_any = True
                    if override is not None and i == call.primary:
                        arr = np.resize(override, _probe.to_numpy(a).shape)  # type: ignore[arg-type]
                        args.append(_probe.as_f32(arr, device=device))
                    else:
                        # float32, not the probe's native float64.  This axis
                        # asks whether the two devices agree, and Metal has
                        # no float64 at all — so every float64 operand made
                        # the comparison fail on the move rather than on the
                        # answer.  649 cells, a tenth of the sweep, reported
                        # "unsupported" for a platform fact rather than
                        # comparing anything.  Both sides get the same
                        # float32 input, which is what makes them comparable.
                        # Narrowed *before* the move, not after: ``.to(metal)``
                        # on a float64 tensor is the call that raises, so a
                        # cast afterwards never runs.  A container that moves
                        # itself (``PackedSequence``) has no ``dtype`` of its
                        # own; its tensor is ``.data``.
                        dtype = getattr(a, "dtype", None) or getattr(
                            getattr(a, "data", None), "dtype", None
                        )
                        narrowed = a.to(lucid.float32) if str(dtype).endswith("float64") else a
                        args.append(narrowed.to(device))
                else:
                    args.append(a)
            return fn(*args, **own.kwargs)

        try:
            cpu_out, metal_out = on("cpu"), on("metal")
        except Exception as exc:  # noqa: BLE001
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:70]}", call)

        # A tensor that quietly stayed on the CPU makes every comparison
        # below trivially true, so the landing device is checked first.
        # What it *means* depends on whether anything was moved.
        #
        # If the call had tensor arguments and they were all moved, an
        # output on the other device is a defect and not an unanswerable
        # question: ``linalg.matrix_rank`` and ``histc`` each took a Metal
        # matrix and returned a CPU tensor, and the next op raised
        # DeviceMismatch.  Reported as SKIP, both sat unnoticed among 45
        # cells that were mostly factories.
        #
        # A factory has no tensor argument to move, so its output follows
        # the default device and that is correct.  The question worth
        # asking there is a different one — whether it honours its own
        # ``device=`` argument — and it is asked below.
        for out, want in ((cpu_out, "cpu"), (metal_out, "metal")):
            device = getattr(out, "device", None)
            if device is None or want in str(device):
                continue
            if isinstance(out, np.ndarray):
                # A bridge out of the framework: ``Tensor.numpy`` answers
                # in host memory whatever the tensor was on, and that is
                # its contract rather than a residency defect.  Only
                # visible once ``to_numpy`` started recognising arrays;
                # before that the op skipped and the question was never
                # reached.
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "answers in host memory by contract — no device to compare",
                )
            # An op that takes a ``device`` argument decides its own
            # output device, whatever its inputs are on.  Checked before
            # ``moved_any``, because a tensor argument is not proof of a
            # transform: ``signal.windows.general_cosine(M, a)`` reads
            # ``a`` as plain coefficients — ``float(a[k])`` — so moving
            # it to Metal moves nothing that reaches the output, and
            # calling that a failure blames the op for the probe.
            if self._takes_device_argument(fn):
                return self._device_kwarg_check(symbol, fn, call)
            if moved_any:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"every tensor input was moved to {want}, and the output "
                    f"came back on {device}",
                )
            return self._finding(symbol, Status.NOT_APPLICABLE, "builds from no tensor input")

        a, b = _probe.to_numpy(cpu_out), _probe.to_numpy(metal_out)
        if a is None or b is None or a.shape != b.shape:
            return self._finding(symbol, Status.SKIP, "outputs not comparable")
        af, bf = self._comparable(a), self._comparable(b)
        if not np.allclose(af, bf, rtol=2e-5, atol=1e-6, equal_nan=True):
            if self._ignores_its_values(fn, call):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "the answer does not depend on the values — this reports "
                    "the device, and both answers are right",
                )
            if self._same_multiset(af, bf):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "same magnitudes in a different arrangement — a "
                    "decomposition has no canonical order or sign",
                )
            return self._finding(
                symbol,
                Status.FAIL,
                f"{domain}: cpu and metal differ by {np.nanmax(np.abs(af - bf)):.3e}",
            )

        # The probe the old sweeps did not carry — but only where a NaN
        # means anything.  The substitution rewrites the *primary*
        # argument, and for ``one_hot`` that argument is a tensor of class
        # indices: replacing it with NaN and casting to float asks the two
        # backends what the zeroth column of a NaN-th class is, and they
        # answered differently because the question has no answer.
        primary = call.args[call.primary] if call.args else None
        if primary is None or not str(getattr(primary, "dtype", "")).endswith(
            ("float16", "float32", "float64")
        ):
            return self._finding(
                symbol,
                Status.PASS,
                f"{domain}: finite inputs agree (primary is not a float — "
                "a non-finite probe would not mean anything)",
            )
        try:
            nan_cpu = _probe.to_numpy(on("cpu", _probe.NON_FINITE))
            nan_metal = _probe.to_numpy(on("metal", _probe.NON_FINITE))
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if nan_cpu is None or nan_metal is None or nan_cpu.shape != nan_metal.shape:
            return self._finding(symbol, Status.PASS, f"{domain}: finite inputs agree")
        if not np.allclose(
            self._comparable(nan_cpu),
            self._comparable(nan_metal),
            rtol=2e-5,
            atol=1e-6,
            equal_nan=True,
        ):
            # Printed at the *disagreement*, not at element zero.  The
            # first four entries of a NaN probe are usually NaN on both
            # sides, so the message read ``cpu [nan nan nan -inf], metal
            # [nan nan nan -inf]`` — two identical lists offered as
            # evidence of a difference, which sends the reader to check
            # the harness rather than the op.
            left = self._comparable(nan_cpu).reshape(-1)
            right = self._comparable(nan_metal).reshape(-1)
            differ = np.flatnonzero(
                ~(np.isclose(left, right, rtol=2e-5, atol=1e-6, equal_nan=True))
            )
            where = differ[:4]
            return self._finding(
                symbol,
                Status.FAIL,
                "cpu and metal disagree on a non-finite input at "
                f"{differ.size} of {left.size} positions "
                f"(index {where.tolist()}: cpu {left[where]}, metal {right[where]})",
            )
        return self._finding(symbol, Status.PASS, f"{domain}: finite and non-finite agree")


class NonFiniteAxis(Axis):
    """NaN must survive an op that has no reason to consume it.

    Found: the CPU ``relu`` turned NaN into 0, so a NaN entering a network
    stopped being traceable at the first activation.  A NaN that
    propagates can be found; one that becomes a zero cannot.
    """

    name = "nonfinite"
    summary = "NaN propagation through elementwise ops"

    #: Ops whose whole job is to consume or classify a NaN, plus the ones
    #: whose *definition* is to ignore it.  ``fmax`` and ``fmin`` are the
    #: IEEE maximumNumber / minimumNumber operations — returning the
    #: non-NaN operand is what distinguishes them from ``maximum`` and
    #: ``minimum``, which do propagate — and ``sign(nan)`` is 0 in the
    #: reference.  Verified against it rather than assumed.
    #:
    #: Out-of-place names: an in-place twin (``ge_``) is the same op
    #: writing into its receiver, so :meth:`applies` exempts it with its
    #: base.  A comparison's answer is a truth value whatever dtype holds
    #: it: ``[nan, 1, -1, 0].ge_(0)`` is ``[0, 1, 0, 1]`` in the reference,
    #: a float receiver that can carry NaN and has no reason to.
    _CONSUMERS = frozenset(
        {
            "fmax",
            "fmin",
            "sign",
            "isnan",
            "isinf",
            "isfinite",
            "isneginf",
            "isposinf",
            "isreal",
            "iscomplex",
            "nan_to_num",
            "nansum",
            "nanmean",
            "nanmedian",
            "nanquantile",
            "nanstd",
            "nanvar",
            "nanargmax",
            "nanargmin",
            "logical_not",
            "signbit",
            "argsort",
            "argmax",
            "argmin",
            "sort",
            "count_nonzero",
            "any",
            "all",
            "zeros_like",
            "ones_like",
            "full_like",
            "empty_like",
            # Writes 0 over every element by definition.  It used to multiply
            # by zeros, which carried NaN through — this axis passed that,
            # and the pass was the bug.
            "zero_",
            "isclose",
            "allclose",
            "equal",
            "eq",
            "ne",
            "not_equal",
            "greater",
            "less",
            "greater_equal",
            "less_equal",
            "gt",
            "lt",
            "ge",
            "le",
            "heaviside",
        }
    )

    def applies(self, symbol: Symbol) -> bool:
        if not super().applies(symbol):
            return False
        short = symbol.short
        if short in self._CONSUMERS or short.removesuffix("_") in self._CONSUMERS:
            return False
        return "stochastic" not in symbol.flags

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
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

        try:
            shape = _probe.to_numpy(call.args[call.primary]).shape  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the varied argument is not a tensor — nothing to vary",
            )

        # Every element, not just the first.  A single NaN at index 0 asks
        # a weaker question than it looks: a crop, a ``take``, a pooling
        # window or a ``masked_select`` can legitimately not include that
        # element, and reported a defect for reading somewhere else.  If
        # the whole input is NaN then any op that reads *any* of it has to
        # say so.
        probe = np.full(shape, np.nan, dtype=np.float64)
        try:
            out = _probe.to_numpy(fn(*call.with_primary(probe).args, **call.kwargs))
        except Exception as exc:  # noqa: BLE001
            if call.domain is not None and isinstance(exc, ValueError):
                # NaN is outside the domain the op's contract is over (a
                # probability), and the op said so with a typed refusal —
                # the NaN did not become a number, which is what is asked.
                return self._finding(
                    symbol,
                    Status.PASS,
                    f"refuses a NaN outside its domain {list(call.domain)}: "
                    f"{type(exc).__name__}",
                )
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:60]}", call)

        # An op that ignores its input cannot propagate anything through it.
        # ``eye``, ``ones_``, ``new_zeros``, ``dirac`` and the rest of the
        # factories take a tensor only for its shape or its device, and
        # demanding a NaN out of them was demanding they corrupt their own
        # output.  Detected rather than listed: feed a second, different
        # input and see whether the answer moves.
        try:
            other = _probe.to_numpy(
                fn(*call.with_primary(np.full(shape, 0.5, dtype=np.float64)).args, **call.kwargs)
            )
        except Exception:  # noqa: BLE001
            other = None
        # ``equal_nan``, not ``nan_to_num``.
        #
        # The probe above is *all NaN*, so mapping NaN to 0 before the
        # comparison threw away the only thing being compared: ``floor``
        # answers NaN for the NaN probe and 0 for the 0.5 probe, both
        # became 0, and the axis concluded that ``floor`` does not depend
        # on its input.  106 cells were excused that way — ``floor``,
        # ``trunc``, ``diff`` and everything else whose value at 0.5 is
        # zero — and each of them is an op whose NaN propagation this
        # axis exists to check.
        #
        # Treating NaN as equal to NaN and unequal to everything else is
        # what "did the answer move" actually means here.
        if (
            other is not None
            and out is not None
            and np.array_equal(
                np.asarray(out, dtype=np.float64),
                np.asarray(other, dtype=np.float64),
                equal_nan=True,
            )
        ):
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "output does not depend on the input's values"
            )
        if out is None or out.size == 0:
            return self._finding(symbol, Status.SKIP, "no comparable output")
        if out.dtype.kind not in "fc":
            return self._finding(
                symbol, Status.NOT_APPLICABLE, f"output dtype {out.dtype} cannot carry NaN"
            )
        if not np.isnan(out).any():
            return self._finding(
                symbol,
                Status.FAIL,
                f"a NaN input produced no NaN anywhere in the output "
                f"(first values {np.asarray(out).reshape(-1)[:4]})",
            )
        return self._finding(symbol, Status.PASS, "NaN propagates")


class BroadcastAxis(Axis):
    """Every broadcast direction, including the ones a guard might skip.

    Found: ``where``'s guard compared the condition against one branch and
    never the two branches against each other.
    """

    name = "broadcast"
    summary = "all broadcast directions for binary ops"

    _PAIRS = (
        ((3, 4), (1, 4)),
        ((1, 4), (3, 4)),
        ((3, 1), (1, 4)),
        ((1, 4), (3, 1)),
        ((4,), (3, 4)),
        ((3, 4), (4,)),
        ((1, 1), (3, 4)),
        ((3, 4), (1, 1)),
        ((2, 1, 4), (3, 4)),
    )

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        # Accepting a second argument is not the same as *using* it.
        # ``relu(x, inplace)`` takes two positionals and ignores the
        # second, so a shape-only test called it a broadcast failure six
        # ways.  The operand has to actually change the answer.
        a0 = _probe.as_f64(_probe.sample("positive", (3, 4)))
        b0 = _probe.as_f64(_probe.sample("positive", (3, 4)))
        c0 = _probe.as_f64(_probe.sample("positive", (3, 4)) * 3.0 + 1.0)
        try:
            with_b = _probe.to_numpy(fn(a0, b0))
            with_c = _probe.to_numpy(fn(a0, c0))
        except Exception:  # noqa: BLE001
            return self._finding(symbol, Status.NOT_APPLICABLE, "not a two-tensor op")
        if with_b is None or with_c is None:
            return self._finding(symbol, Status.NOT_APPLICABLE, "not a two-tensor op")
        if np.array_equal(with_b, with_c, equal_nan=True):
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "second argument does not affect the result"
            )
        # ...and taking two operands is not the same as being elementwise.
        # ``kron`` returns (3, 16), ``mv`` contracts, ``masked_select``
        # returns a 1-D selection, ``binary_cross_entropy`` reduces to a
        # scalar.  Demanding a broadcast shape from any of them reports a
        # defect in an op that is behaving exactly as specified.  The
        # discriminator is the equal-shape case: an elementwise op maps
        # (3, 4) x (3, 4) to (3, 4).
        if tuple(with_b.shape) != (3, 4):
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"not elementwise — equal shapes give {tuple(with_b.shape)}, not (3, 4)",
            )

        failures: list[str] = []
        follows_first = True
        follows_second = True
        answered = 0
        for sa, sb in self._PAIRS:
            a = _probe.as_f64(_probe.rng(1).uniform(0.5, 1.5, sa))
            b = _probe.as_f64(_probe.rng(2).uniform(0.5, 1.5, sb))
            want = np.broadcast_shapes(sa, sb)
            try:
                got = _probe.to_numpy(fn(a, b))
            except Exception as exc:  # noqa: BLE001
                # A refusal is recorded, but it must not clear the
                # follows-flags below.  It used to, and that punished the
                # better-behaved op: ``vector_quantize(x, codebook)``
                # consults a (K, D) table per row, so its answer has *x*'s
                # shape whatever K is — the same shape as ``isin``, which
                # this axis already exempts.  The only thing keeping it out
                # of that exemption was that it *validates* the feature
                # dimension and raises on a mismatched pair, where ``isin``
                # silently accepts anything.  Refusing an incompatible pair
                # is evidence the op is not broadcasting, not evidence that
                # it is.
                failures.append(f"{sa}x{sb}: {type(exc).__name__}")
                continue
            if got is None:
                continue
            answered += 1
            follows_first &= tuple(got.shape) == sa
            follows_second &= tuple(got.shape) == sb
            if tuple(got.shape) != want:
                failures.append(f"{sa}x{sb} -> {got.shape}, expected {want}")
                continue

            # ...and the right shape is not the right answer.
            #
            # This axis compared shapes and nothing else, so an op that
            # broadcast to the correct extent and computed the wrong
            # numbers passed all nine directions — 59 symbols reporting
            # "9 directions" about a property that was never checked.
            # Found by mutation: ``(a * b) * 2`` under broadcast only.
            #
            # The reference is the op itself on operands expanded by
            # hand, so nothing here encodes an opinion about what the op
            # should compute — only that broadcasting an argument and
            # expanding it first must come to the same thing.
            if "stochastic" in symbol.flags:
                continue  # two draws differ by design; only the shape means anything
            try:
                expanded = _probe.to_numpy(
                    fn(
                        _probe.as_f64(np.broadcast_to(_probe.to_numpy(a), want)),
                        _probe.as_f64(np.broadcast_to(_probe.to_numpy(b), want)),
                    )
                )
            except Exception:  # noqa: BLE001 - the shape check already ran
                continue
            if expanded is None or expanded.shape != got.shape:
                continue
            if not np.allclose(
                self._comparable(got),
                self._comparable(expanded),
                rtol=2e-5,
                atol=1e-8,
                equal_nan=True,
            ):
                drift = float(np.abs(self._comparable(got) - self._comparable(expanded)).max())
                failures.append(
                    f"{sa}x{sb}: broadcasting differs from expanding first by {drift:.3e}"
                )
        # ...and being elementwise on equal shapes is not the same as
        # broadcasting.  ``isin`` asks "is each element of a somewhere in
        # b", so its answer has *a*'s shape whatever b's is; ``selu_``
        # writes into its first operand and cannot change shape;
        # ``new_tensor`` builds from the second and takes that one.  Each
        # passes the equal-shape test above and none of them broadcasts.
        #
        # Read off the results rather than listed: an op whose output
        # follows one operand in *every* direction is shaped by that
        # operand, not by the pair.
        # ``answered`` guards the vacuous case: an op that refuses *every*
        # direction leaves both flags untouched at True, and exempting it
        # would hide an op that ought to broadcast and raises instead.
        if failures and answered and (follows_first or follows_second):
            which = "first" if follows_first else "second"
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                f"the output follows the {which} operand in every direction — "
                "shaped by one argument, not broadcast between them",
            )
        if failures:
            return self._finding(symbol, Status.FAIL, "; ".join(failures[:4]), failures=failures)
        return self._finding(symbol, Status.PASS, f"{len(self._PAIRS)} directions")


class DtypeAxis(Axis):
    """Every dtype the framework claims, on both devices.

    Found (earlier): the CPU backend was missing Bool / I8 / I16 / F16
    paths that Metal had, so the same call worked on one device and raised
    on the other.
    """

    name = "dtype"
    summary = "dtype coverage and cpu/metal symmetry"

    @staticmethod
    def _complex_check(fn: Any, call: Any) -> list[str]:
        """Which complex dtypes the op accepts, for the ops that need one."""
        accepted: list[str] = []
        for name in _probe.COMPLEX_DTYPES:
            dtype = _probe.dtype_of(name)
            if dtype is None:
                continue

            def build(array: Any, follow: bool, _dtype: Any = dtype) -> Any:
                return _probe.tensor_of(
                    np.asarray(array, dtype=complex if follow else None),
                    dtype=_dtype if follow else None,
                )

            try:
                args = _probe.dtype_args(call, name, build)
                kwargs = _probe.dtype_kwargs(call, build)
                if _probe.to_numpy(fn(*args, **kwargs)) is not None:
                    accepted.append(name)
            except Exception:  # noqa: BLE001 - a refusal is the answer
                continue
        return accepted

    def _dtype_kwarg_check(self, symbol: Symbol, fn: Any, call: Any) -> Finding | None:
        """A factory decides its own dtype, so ask it to.

        ``zeros``, ``ones``, ``full``, ``empty``, ``rand`` and ``randn``
        take a *shape*, not a tensor, so rebuilding "the primary at dtype
        D" rebuilds an argument that is not a tensor and every dtype was
        reported as refused.  Six factories and the only question that
        matters for them — does the result come back at the dtype that
        was asked for — went unasked.

        Returns ``None`` when the symbol takes no ``dtype`` keyword, so
        the caller can fall through to its own answer.
        """
        try:
            signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
        except (TypeError, ValueError, NameError):
            return None
        if "dtype" not in signature.parameters:
            return None

        honoured: list[str] = []
        for name in _probe.DTYPES:
            dtype = _probe.dtype_of(name)
            if dtype is None:
                continue
            try:
                out = fn(*call.args, **{**call.kwargs, "dtype": dtype})
            except Exception:  # noqa: BLE001 - a refusal is a fine answer
                continue
            got = getattr(out, "dtype", None)
            if got is None:
                continue
            if str(got) != str(dtype):
                return self._finding(symbol, Status.FAIL, f"asked for {dtype} and got {got}")
            honoured.append(name)
        if not honoured:
            return None
        return self._finding(symbol, Status.PASS, f"honours dtype= for {len(honoured)} dtypes")

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
                "the varied argument is not a tensor — nothing to vary",
            )

        del base  # only needed to prove there is one; `dtype_args` reads it

        devices = ["cpu", "metal"] if ctx.metal else ["cpu"]
        support: dict[str, set[str]] = {d: set() for d in devices}
        for name in _probe.DTYPES:
            dt = _probe.dtype_of(name)
            if dt is None:
                continue
            for device in devices:

                def build(array: Any, follow: bool, _d: str = device) -> Any:
                    if not follow:
                        return _probe.tensor_of(array, device=_d)
                    return _probe.tensor_of(
                        array.astype(_probe.numpy_of(name)), dtype=dt, device=_d
                    )

                try:
                    args = _probe.dtype_args(call, name, build)
                    kwargs = _probe.dtype_kwargs(call, build)
                    if _probe.to_numpy(fn(*args, **kwargs)) is not None:
                        support[device].add(name)
                except Exception as exc:  # noqa: BLE001 - a refusal is the answer
                    if _probe.refuses_a_value(exc):
                        support[device].add(name)

        if not any(support.values()):
            complex_support = self._complex_check(fn, call)
            if complex_support:
                return self._finding(
                    symbol,
                    Status.PASS,
                    f"defined on {', '.join(complex_support)} and no real dtype",
                )
            kwarg = self._dtype_kwarg_check(symbol, fn, call)
            if kwarg is not None:
                return kwarg
            return self._finding(symbol, Status.SKIP, "no dtype accepted")
        if len(devices) == 2:
            # An op whose *result* is float64 whatever it was handed cannot
            # exist on Metal at all — `Tensor.double` and `float_power` say
            # so in their names.  Every input dtype then reads as
            # cpu-only, which is true and is not a defect: the question the
            # axis asks does not apply to them.
            if _produces_float64(fn, call):
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "result is float64 or complex128 by definition, which Metal has no dtype for",
                )
            # float64 does not exist on Metal and the engine documents the
            # downcast, so holding it against an op would flag every one.
            only_cpu = sorted(support["cpu"] - support["metal"] - {"float64"})
            only_metal = sorted(support["metal"] - support["cpu"])
            if only_cpu or only_metal:
                return self._finding(
                    symbol,
                    Status.FAIL,
                    f"asymmetric dtype support — cpu only {only_cpu}, "
                    f"metal only {only_metal}"
                    + _contract_verdict(symbol.qualname, only_cpu, only_metal),
                    cpu=sorted(support["cpu"]),
                    metal=sorted(support["metal"]),
                )
        return self._finding(
            symbol, Status.PASS, f"{len(support[devices[0]])}/{len(_probe.DTYPES)} dtypes"
        )


class EdgeAxis(Axis):
    """Degenerate shapes: empty, 0-d, and a single element.

    An op that raises on an empty tensor is usually a real gap, but the
    line is drawn at *inventing data*: an empty input must not produce a
    non-empty output.
    """

    name = "edge"
    summary = "empty, 0-d and size-1 inputs"

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        fn = _surface.resolve(symbol)
        if fn is None:
            return self._finding(symbol, Status.SKIP, "not resolvable")
        call, _, why = self._working_call(fn, symbol, ctx)
        if call is None:
            return self._no_call(symbol, why)
        try:
            shape = call.base.shape
        except TypeError:
            return self._finding(
                symbol,
                Status.NOT_APPLICABLE,
                "the varied argument is not a tensor — nothing to vary",
            )

        notes: list[str] = []

        # "Empty in, empty out" only holds for an op that keeps its shape.
        # A reduction over an empty tensor is its identity element — the
        # sum of nothing is 0, a scalar — and a selection returns however
        # many elements matched.  The rule was applied to everything, so
        # ``logsumexp``, ``masked_select`` and ``block_diag`` were reported
        # for agreeing with the reference exactly.  Whether the op keeps
        # its shape is asked rather than listed: run it once at full size
        # and see.
        reference_out = _probe.to_numpy(fn(*call.args, **call.kwargs))
        shape_preserving = reference_out is not None and tuple(reference_out.shape) == tuple(shape)

        # The primary is resized together with every operand the call
        # declares to share its shape (``Call.resized``): a loss's target
        # follows its input, so "empty input" is asked as an empty
        # *problem*, not as an input that no longer matches its target.
        empty = np.zeros((0, *shape[1:]), dtype=np.float64)
        try:
            emptied = call.resized(empty)
            out = _probe.to_numpy(fn(*emptied.args, **emptied.kwargs))
            if (
                shape_preserving
                and out is not None
                and out.ndim > 0
                and out.shape[0] != 0
                and out.size != 0
            ):
                # Only the primary and its declared peers were emptied.
                # Where another tensor argument is still full size, it is
                # the one supplying the shape and nothing was invented:
                # ``solve_ex(A_empty, B)`` answers with B's batch, and
                # ``new_tensor`` builds from its data argument and takes
                # that.  Emptiness can only propagate from an operand the
                # answer's shape depends on.
                others = [
                    a
                    for i, a in enumerate(emptied.args)
                    if i != call.primary and hasattr(a, "shape")
                ]
                if any(int(np.prod(tuple(a.shape))) != 0 for a in others):
                    return self._finding(
                        symbol,
                        Status.NOT_APPLICABLE,
                        "another operand is still full size and supplies the shape",
                    )
                return self._finding(symbol, Status.FAIL, f"empty input produced shape {out.shape}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"empty: {type(exc).__name__}")

        single = np.full((1,) * len(shape), 0.7, dtype=np.float64)
        try:
            fn(*call.resized(single).args, **call.kwargs)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"size-1: {type(exc).__name__}")

        if len(notes) == 2:
            return self._refusal(symbol, "; ".join(notes), call)
        return self._finding(symbol, Status.PASS, "; ".join(notes) or "empty and size-1 accepted")


# ── structural axes ──────────────────────────────────────────────────────────


class ModuleAxis(Axis):
    """``nn.Module`` lifecycle: build, forward, backward, save, move.

    Found (earlier, by hand): a family whose model and its dynamics both
    held the same backbone had every weight in ``state_dict`` twice —
    invisible to ``parameters()``, which dedupes by identity, so only a
    round trip through serialisation shows it.
    """

    name = "module"
    summary = "construct, forward, backward, state_dict round trip, device move"
    kinds = frozenset({"module"})
    varies_a_tensor = False

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        cls = symbol.obj

        # The first construction that can also *forward*, not the first
        # that survives ``__init__``.  ``Upsample`` builds happily from
        # its own defaults and then cannot run, and committing to that
        # object left the class unreachable while a second, equally
        # legitimate construction was available — see
        # :func:`_module_candidates`.
        module = why = out = None
        chosen = 0
        for index, candidate in enumerate(_module_candidates(cls)):
            if module is None:
                module = candidate  # keep the first, to report against
            produced, note = _try_forward(candidate)
            if produced is not None:
                module, out, why, chosen = candidate, produced, None, index
                break
            why = note
        if module is None:
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        # A container holds layers and has no forward of its own —
        # ``ModuleList`` and ``ParameterDict`` raise NotImplementedError by
        # design, and the base ``Module`` does too.  Running the lifecycle
        # against them reported a shape probe failing to find an input for
        # something that never accepts one.
        #
        # Asked of the *outcome* rather than of the function object.  The
        # containers override ``forward`` in order to raise a better
        # message than the base class's — ``"ModuleList has no forward;
        # iterate manually."`` — so identity against ``Module.forward``
        # missed all six of them and they were reported as unreachable
        # rather than as inapplicable, which are opposite claims.
        if getattr(type(module), "forward", None) is getattr(lucid.nn.Module, "forward", None) or (
            why is not None and "NotImplementedError" in why
        ):
            return self._finding(
                symbol, Status.NOT_APPLICABLE, "a container: it holds layers and defines no forward"
            )

        # A lazy layer registers placeholder parameters at construction so
        # that anything reading ``parameters()`` early is handed the
        # objects the real weights will occupy.  A placeholder holds no
        # values, so ``state_dict`` leaves it out — writing its
        # zero-element buffer would put an entry in the checkpoint that
        # restores to shape ``(0,)``.  Counting them here made all
        # thirteen lazy classes look like they had lost their state, with
        # a message blaming a doubly-registered submodule.
        params = [
            p for p in module.parameters() if not isinstance(p, lucid.nn.UninitializedParameter)
        ]
        # Non-persistent buffers are excluded from ``state_dict`` on
        # purpose — a positional-encoding table or a rotary cache is
        # recomputed from the constructor arguments, so writing it into
        # every checkpoint is waste, and the reference framework draws the
        # same distinction.  Counting them made ``RotaryEmbedding`` and
        # ``SinusoidalEmbedding`` look like they had lost their state.
        # The set belongs to the module that registered the buffer, so a wrapper of a
        # model with such a buffer (a PEFT tuner around a GPT-2) is read module by module.
        buffers = [
            name
            for _, owner in module.named_modules()
            for name, _ in owner.named_buffers(recurse=False)
            if name not in getattr(owner, "_non_persistent_buffers", set())
        ]
        state = module.state_dict()
        if len(state) != len(params) + len(buffers):
            return self._finding(
                symbol,
                Status.FAIL,
                f"state_dict has {len(state)} entries for {len(params)} parameters "
                f"and {len(buffers)} buffers — a shared submodule is registered twice",
            )

        if out is None:
            return self._finding(symbol, Status.SKIP, f"forward: {why}")

        try:
            # The *same* construction, not merely the same class.  A
            # class has several plausible constructions and the one that
            # forwarded is not always the first — reloading into the
            # first produced a checkpoint written by one shape and read
            # by another, and ``BatchNorm3d`` was reported for losing its
            # state when nothing had been lost.
            reloaded = next(
                (
                    candidate
                    for index, candidate in enumerate(_module_candidates(cls))
                    if index == chosen
                ),
                None,
            )
            if reloaded is not None:
                reloaded.load_state_dict(module.state_dict())
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"state_dict does not round trip: {type(exc).__name__}: {exc}"
            )

        if params:
            trainable = [p for p in params if p.requires_grad]
            if trainable and not out.requires_grad and lucid.is_grad_enabled():
                # backward() refuses an output with no graph, which would
                # file this under "the op refused" — but nothing refused:
                # the module has trainable parameters and its forward
                # detached them.
                return self._finding(
                    symbol,
                    Status.FAIL,
                    "forward ran but its output carries no graph — no "
                    "parameter can receive a gradient",
                )
            try:
                loss = (out * out).mean()
                module.zero_grad()
                loss.backward()
                reached = [p for p in module.parameters() if p.grad is not None]
                if not reached:
                    return self._finding(
                        symbol, Status.FAIL, "forward ran but no parameter received a gradient"
                    )
            except Exception as exc:  # noqa: BLE001
                return self._finding(
                    symbol, Status.UNSUPPORTED, f"backward: {type(exc).__name__}: {str(exc)[:60]}"
                )
        return self._finding(symbol, Status.PASS, f"{len(params)} params, {len(state)} state keys")


class OptimAxis(Axis):
    """Optimizers: a step must move parameters and survive a round trip."""

    name = "optim"
    summary = "step, state_dict round trip, convergence on a convex problem"
    kinds = frozenset({"optim"})
    varies_a_tensor = False

    def run(self, symbol: Symbol, ctx: Context) -> Finding:
        cls = symbol.obj
        target = _probe.as_f64(np.array([1.5, -0.5, 2.0]))
        weight = lucid.nn.Parameter(_probe.as_f64(np.zeros(3)))

        optimiser = None
        for kwargs in ({"lr": 0.1}, {"lr": 0.1, "max_iter": 4}, {}):
            with contextlib.suppress(Exception):
                optimiser = cls([weight], **kwargs)
                break
        if optimiser is None:
            if getattr(cls, "__abstractmethods__", frozenset()) or cls.__name__ == ("Optimizer"):
                # The base every optimizer derives from.  It has no
                # update rule of its own, which is why it cannot be
                # constructed — every concrete subclass is audited in
                # its own right.
                return self._finding(
                    symbol,
                    Status.NOT_APPLICABLE,
                    "the base class — the update rule is the subclass's",
                )
            return self._finding(symbol, Status.SKIP, "no constructor signature worked")

        def closure() -> Any:
            optimiser.zero_grad()
            loss = ((weight - target) ** 2).sum()
            loss.backward()
            return loss

        try:
            first = float(closure())
            for _ in range(12):
                loss = closure()
                try:
                    optimiser.step(closure)  # line-search optimizers need it
                except TypeError:
                    optimiser.step()
            last = float(loss)
        except Exception as exc:  # noqa: BLE001
            return self._refusal(symbol, f"{type(exc).__name__}: {str(exc)[:70]}")

        if not np.isfinite(last):
            return self._finding(symbol, Status.FAIL, f"loss became {last}")
        if last >= first:
            return self._finding(
                symbol,
                Status.FAIL,
                f"12 steps on a convex quadratic did not reduce the loss "
                f"({first:.4f} -> {last:.4f})",
            )

        try:
            state = optimiser.state_dict()
            optimiser.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            return self._finding(
                symbol, Status.FAIL, f"state_dict does not round trip: {type(exc).__name__}"
            )
        return self._finding(symbol, Status.PASS, f"loss {first:.4f} -> {last:.4f}")


# ── construction helpers ─────────────────────────────────────────────────────


def _produces_float64(fn: Any, call: Call) -> bool:
    """Whether the op answers in float64 from a float32 input.

    The discriminator for "this op cannot exist on Metal".  Asked by
    running it rather than by matching on the name, so a future
    ``as_double`` is caught without being listed anywhere.
    """

    def build(array: np.ndarray, follow: bool) -> Any:
        return _probe.tensor_of(
            array.astype(np.float32) if follow else array, dtype=lucid.float32 if follow else None
        )

    try:
        out = fn(*_probe.dtype_args(call, "float32", build), **call.kwargs)
    except Exception:  # noqa: BLE001
        return False
    first = out[0] if isinstance(out, tuple | list) and out else out
    # complex128 is float64's pair and has no Metal dtype either: ``cdouble``.
    return str(getattr(first, "dtype", "")).endswith(("float64", "complex128"))


@functools.lru_cache(maxsize=1)
def _contract() -> dict[str, list[str]]:
    """The measured reference dtype table, or empty when absent.

    Checked in, so it is available without the reference framework
    installed.  Regenerate with
    ``python -m lucid.test.audit.tools.dtype_contract``.
    """
    path = pathlib.Path(__file__).with_name("dtype_contract.json")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    symbols = data.get("symbols")
    return symbols if isinstance(symbols, dict) else {}


def _contract_verdict(qualname: str, only_cpu: list[str], only_metal: list[str]) -> str:
    """Which device is wrong, when the reference has an opinion.

    A disagreement between the two devices says they differ, never which
    one to change — and deciding that by eye was going badly: the
    reference accepts int64 for ``avg_pool2d`` and rejects every integer
    for ``softmax``, and neither is guessable from the name.  When the
    op was measured, name the side that deviates so the finding is
    actionable rather than merely true.
    """
    accepted = _contract().get(_surface.recorded_as(qualname))
    if accepted is None:
        return ""
    reference = set(accepted)
    # only_cpu: cpu takes it, metal does not.  only_metal: the reverse.
    # Whichever side agrees with the reference is the one to keep.
    parts = []
    for device, mine, theirs in (("cpu", only_cpu, only_metal), ("metal", only_metal, only_cpu)):
        over = [d for d in mine if d not in reference]
        under = [d for d in theirs if d in reference]
        if over:
            parts.append(f"{device} wrongly accepts {over}")
        if under:
            parts.append(f"{device} wrongly rejects {under}")
    return f" — reference says: {'; '.join(parts)}" if parts else ""


#: Constructor ladders, tried in order.  A module that needs an argument
#: shape this does not guess is reported SKIP rather than FAIL.
_CTOR_ARGS: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
    ((), {}),
    ((4,), {}),
    ((4, 4), {}),
    ((3, 4), {}),
    ((3, 4, 3), {}),
    ((4,), {"eps": 1e-5}),
    ((1, 4), {}),
    ((4, 4, 3), {"padding": 1}),
)


def _ctor_args_with_qconfig() -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """The ladder again, each rung carrying a qconfig.

    ``qat.Conv1d`` / ``2d`` / ``3d`` and their fused ReLU variants
    declare ``(*args, **kwargs)``, so a signature says nothing about them
    and only the ladder can reach them — and the plain ladder is refused
    with "requires a qconfig".  ``qat.Conv2d(3, 3, 3, qconfig=...)``
    builds.
    """
    try:
        qconfig = _default_qconfig()
    except Exception:  # noqa: BLE001
        return []
    return [(args, {**kwargs, "qconfig": qconfig}) for args, kwargs in _CTOR_ARGS]


#: Constructor arguments by parameter name.
#:
#: ``nn`` layer constructors are almost entirely small integers with
#: self-describing names, which is what makes deriving them possible at
#: all.  Sizes agree with :data:`_FORWARD_SHAPES` so that a module built
#: here has a chance of accepting one of the inputs tried against it.
_CTOR_BY_NAME: dict[str, Any] = {
    # widths, all agreeing with the (2, 4) / (2, 3, 6, 6) probe inputs
    "in_features": 4,
    "out_features": 4,
    "in1_features": 4,
    "in2_features": 4,
    "in_channels": 3,
    "out_channels": 3,
    "num_features": 4,
    "num_channels": 4,
    "channels": 4,
    "features": 4,
    "hidden_size": 4,
    "input_size": 4,
    "embed_dim": 4,
    "embedding_dim": 4,
    "d_model": 4,
    "dim": 4,
    "head_dim": 2,
    "size": 4,
    "num_embeddings": 8,
    "vocab_size": 8,
    "normalized_shape": 4,
    "unflattened_size": (2, 2),
    "output_size": 2,
    "num_positions": 8,
    "max_position_embeddings": 8,
    "height": 6,
    "width": 6,
    # counts
    "num_heads": 2,
    "nhead": 2,
    "num_kv_heads": 2,
    "num_layers": 1,
    "num_classes": 4,
    "groups": 1,
    "num_groups": 1,
    "upscale_factor": 2,
    "downscale_factor": 2,
    # windowing
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "output_padding": 0,
    "dilation": 1,
    # scalars
    "eps": 1e-5,
    "p": 0.5,
    "bias": True,
    "momentum": 0.1,
    "affine": True,
}


def _default_qconfig() -> Any:
    """A QConfig, built on demand.

    The quantisation layers declare ``qconfig`` with a default of
    ``None`` and then refuse it: ``qat.Linear requires a qconfig``.  A
    signature-derived constructor honours the author's default and so
    never supplied one, which left the whole ``qat`` / ``intrinsic.qat``
    stack — fifteen classes — unconstructible and therefore unaudited.
    """
    import lucid.quantization as quantization

    return quantization.QConfig(quantization.MinMaxObserver, quantization.MinMaxObserver)


def _default_qscheme() -> Any:
    import lucid.quantization as quantization

    return quantization.QScheme.PER_TENSOR_AFFINE


def _default_qdtype() -> Any:
    import lucid.quantization as quantization

    return quantization.quint8


def _default_qparams(kind: str) -> Any:
    """A ``scale`` or ``zero_point`` for the quantised activations."""
    import numpy as _np

    if kind == "scale":
        return _probe.as_f32(_np.array(0.1))
    return _probe.as_int(_np.array(0))


def _default_submodule(name: str, rank: int = 2) -> Any:
    """A concrete layer for a parameter annotated as the ``Module`` base.

    The fused ``intrinsic`` layers take the layers they fuse, and the
    annotation names the base class — which says nothing about what
    would actually work.  The parameter name does, and the *fused
    class's own name* says the rank: ``ConvReLU3d`` fuses a
    ``Conv3d``, and handing it the 2-D default built an object whose
    forward could only fail.
    """
    import lucid.nn as nn

    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank]
    norm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[rank]
    return {
        "conv": lambda: conv(3, 3, 3, padding=1),
        "bn": lambda: norm(3),
        "norm": lambda: norm(3),
        "relu": lambda: nn.ReLU(),
        "linear": lambda: nn.Linear(4, 4),
        "module": lambda: nn.Linear(4, 4),
        # A model the Core ML export would actually compress: it leaves
        # any weight under 2048 elements alone, so ``Linear(4, 4)`` gives
        # ``CompressionAware`` nothing to train against and it refuses.
        # 4 in keeps the (2, 4) probe input; 1024 out clears the floor.
        "compressible": lambda: nn.Linear(4, 1024),
        "parametrization": lambda: nn.Identity(),
        # ``ParametrizationList(modules, original)`` (LCD-624): the chain
        # of steps the original passes through, as the installer builds it.
        "modules": lambda: [nn.Identity()],
        "original": lambda: nn.Parameter(_probe.as_f32(_probe.sample("moderate", (4, 4)))),
        "encoder_layer": lambda: nn.TransformerEncoderLayer(4, 2),
        "decoder_layer": lambda: nn.TransformerDecoderLayer(4, 2),
    }[name]()


#: Constructor arguments that have to be built rather than named, keyed
#: on the parameter name.  Kept separate from :data:`_CTOR_BY_NAME` so
#: that table stays a table of constants and nothing is constructed at
#: import time.
_CTOR_FACTORY: dict[str, Any] = {
    "qconfig": _default_qconfig,
    "module": lambda: _default_submodule("module"),
    "parametrization": lambda: _default_submodule("parametrization"),
    "modules": lambda: _default_submodule("modules"),
    "original": lambda: _default_submodule("original"),
    "scale": lambda: _default_qparams("scale"),
    "zero_point": lambda: _default_qparams("zero_point"),
    "conv": lambda: _default_submodule("conv"),
    "bn": lambda: _default_submodule("bn"),
    "relu": lambda: _default_submodule("relu"),
    "linear": lambda: _default_submodule("linear"),
    "encoder_layer": lambda: _default_submodule("encoder_layer"),
    "decoder_layer": lambda: _default_submodule("decoder_layer"),
    # ``QScheme`` is an enum and answers to the enum rule; ``QDtype`` is a
    # dataclass whose instances are module-level constants, so there is
    # nothing on the class to enumerate and the name has to say it.
    "qscheme": _default_qscheme,
    "qdtype": _default_qdtype,
}


#: Constructor arguments that mean something different in one class than
#: they do everywhere else.  ``dim`` is a *width* for ``Linear``-shaped
#: layers and the table gives it 4; for ``Unflatten`` it is an **axis**,
#: and 4 is out of range on the probe — the layer reported ``'int' object
#: is not iterable`` and went unreached.  Kept as narrow as it looks:
#: one entry per genuine collision, not a per-class table.
_CTOR_BY_CLASS: dict[str, dict[str, Any]] = {
    "Unflatten": {"dim": 1, "unflattened_size": (2, 2)},
    # ``output_size`` is a *canvas* here, and the table's 2 is smaller
    # than the 3x3 kernel it has to reassemble into.
    "Fold": {"output_size": 6, "kernel_size": 3},
}


def _default_compression() -> Any:
    """A compression spec ``CompressionAware`` accepts.

    The annotation is ``WeightPrecision | Palettize | Sparsify``, which no
    rule here reads, and the enum rule alone would pick ``FLOAT`` — the
    one value the class refuses, since it compresses nothing.  A 4-bit
    palette is the class's own documented example.
    """
    import lucid.coreml as cml  # noqa: PLC0415 - optional subsystem

    return cml.Palettize(bits=4)


#: :data:`_CTOR_BY_CLASS` for arguments that have to be built: the same
#: narrow collisions, where the value is an object rather than a constant.
#: ``model`` and ``weights`` are generic enough names that answering them
#: in :data:`_CTOR_FACTORY` would change how every other class taking one
#: is built.
_CTOR_FACTORY_BY_CLASS: dict[str, dict[str, Any]] = {
    "CompressionAware": {
        "model": lambda: _default_submodule("compressible"),
        "weights": _default_compression,
    }
}


def _ctor_value(name: str, annotation: Any, depth: int, cls_name: str = "") -> Any:
    """One constructor argument, by name first and annotation second.

    Raises ``KeyError`` when neither says anything, so the caller can
    fall back to the fixed ladder rather than pass a wrong-typed value
    and read the resulting ``TypeError`` as "this class is unbuildable".
    """
    override = _CTOR_BY_CLASS.get(cls_name, {})
    if name in override:
        return override[name]
    factories = _CTOR_FACTORY_BY_CLASS.get(cls_name, {})
    if name in factories:
        try:
            return factories[name]()
        except Exception as exc:  # noqa: BLE001
            raise KeyError(name) from exc
    if name in _CTOR_BY_NAME:
        return _CTOR_BY_NAME[name]
    if name in ("conv", "bn", "norm", "relu"):
        match = re.search(r"([123])d$", cls_name)
        return _default_submodule(name, int(match.group(1)) if match else 2)
    if name in _CTOR_FACTORY:
        try:
            return _CTOR_FACTORY[name]()
        except Exception as exc:  # noqa: BLE001
            raise KeyError(name) from exc

    if isinstance(annotation, type):
        # A sub-module, which is how the fused ``intrinsic`` layers are
        # spelled: ``ConvReLU2d(conv, relu)`` takes the two layers it
        # fuses.  Built one level deep, so a cycle cannot run away.
        if depth < 1 and _is_module_class(annotation):
            built = _construct_module(annotation, depth + 1)
            if built is None:
                raise KeyError(name)
            return built
        if issubclass(annotation, bool):
            return True
        if issubclass(annotation, int):
            return 4
        if issubclass(annotation, float):
            return 0.5
        members = list(getattr(annotation, "__members__", {}).values())
        if members:  # an Enum: the first member is as good as any
            return members[0]

    text = str(annotation)
    if "int" in text and "tuple" in text:
        return 1
    if text.endswith("int") or "_Size" in text:
        return 2
    raise KeyError(name)


def _is_module_class(obj: Any) -> bool:
    try:
        import lucid.nn as _nn

        return isinstance(obj, type) and issubclass(obj, _nn.Module)
    except Exception:  # noqa: BLE001
        return False


def _construct_module(cls: Any, depth: int = 0, fill_optional: bool = False) -> Any:
    """Build ``cls`` from its own signature, or ``None``.

    The ladder this replaces tried eight fixed argument tuples —
    ``(4,)``, ``(4, 4)``, ``(3, 4, 3)`` and so on — and reported
    everything it could not hit as "no constructor signature worked".
    That was 114 of the framework's ``nn.Module`` classes, every one of
    them then contributing nothing to any axis: not the module lifecycle,
    not the smoke pass, nothing.  A class that is never constructed is a
    class the audit cannot notice a regression in.

    Reading the signature works here because ``nn`` constructors are
    almost entirely small integers with self-describing names.  Where the
    name says nothing the annotation usually does, and where neither does
    the fixed ladder still runs underneath.
    """
    try:
        # ``FORWARDREF`` rather than the default.
        #
        # H7 requires annotations that exist only under ``TYPE_CHECKING``,
        # and PEP 649 leaves those unevaluated until something asks.
        # ``inspect.signature`` asking is what raised ``NameError: name
        # 'QConfig' is not defined`` on every quantised layer — not a
        # defect in the layer, this function reading it wrongly.
        # ``FORWARDREF`` resolves the names it can and returns the rest as
        # forward references, which the annotation fallback below reads as
        # text in any case.
        signature = inspect.signature(cls, annotation_format=annotationlib.Format.FORWARDREF)
    except Exception:  # noqa: BLE001 - an unreadable signature is not a finding
        return None

    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            # ...unless the author's default is one the class then
            # rejects.  ``qconfig`` defaults to ``None`` and every
            # quantisation layer raises "requires a qconfig" on it.
            if parameter.name == "qconfig" and parameter.default is None:
                try:
                    kwargs["qconfig"] = _default_qconfig()
                except Exception:  # noqa: BLE001
                    pass
                continue
            # ``Upsample(size=None, scale_factor=None)`` is the same
            # shape of problem without the loud message.  Both are
            # optional, exactly one has to be given, and honouring both
            # defaults builds a layer whose forward can only fail — it
            # reported "Unsupported interpolation mode: 'nearest'",
            # which names neither the missing argument nor the real
            # constraint.  Four classes were unreachable that way, and
            # ``FractionalMaxPool2d`` a fifth for the same reason.
            #
            # Tried as a *second* candidate rather than instead of the
            # first: the author's default is still the more faithful
            # construction, and a class that works with it should be
            # audited as the user gets it.
            if fill_optional and parameter.default is None:
                with contextlib.suppress(KeyError):
                    kwargs[parameter.name] = _ctor_value(
                        parameter.name, parameter.annotation, depth, getattr(cls, "__name__", "")
                    )
            continue  # a default the author chose is better than a guess
        try:
            kwargs[parameter.name] = _ctor_value(
                parameter.name, parameter.annotation, depth, getattr(cls, "__name__", "")
            )
        except KeyError:
            return None

    try:
        return cls(**kwargs)
    except Exception:  # noqa: BLE001 - the fixed ladder is the fallback
        return None


def _from_float(cls: Any) -> Any:
    """A quantised layer built the way the framework builds one.

    ``nn.quantized.Conv2d.__init__`` takes eight positional arguments and
    produces a layer whose packed int8 weight has not been written yet —
    the constructor exists for the state-dict loader.  Forwarding one
    answers ``conv: W rank mismatch``, which is what eighteen quantised
    and fused layers reported: a real object, correctly built, that
    cannot run because nothing had quantised anything into it.

    The route that works is ``from_float`` over an **observed** float
    module — one carrying a calibrated ``activation_post_process``, which
    is what ``prepare`` attaches.  Note that ``convert`` does not take
    this route for convolutions: it routes the Linear family and returns
    conv layers unchanged, so the flow a user would write leaves them
    float.  That is recorded as an observation; here it simply means the
    audit has to call ``from_float`` itself.
    """
    convert = getattr(cls, "from_float", None)
    if convert is None:
        return None
    import lucid.nn.intrinsic as intrinsic  # noqa: PLC0415 - optional subsystem
    import lucid.quantization as quantization  # noqa: PLC0415

    name = getattr(cls, "__name__", "")
    source_cls = getattr(lucid.nn, name, None) or getattr(intrinsic, name, None)
    if source_cls is None or source_cls is cls:
        return None
    source = _construct_module(source_cls)
    if source is None:
        return None
    try:
        source.eval()
        qconfig = quantization.get_default_qconfig()
        observer = qconfig.activation()
        observer(_probe.as_f32(_probe.sample("moderate", (2, 3, 6, 6))))
        source.activation_post_process = observer
        source.qconfig = qconfig
        return convert(source)
    except Exception:  # noqa: BLE001 - surveying, not asserting
        return None


def _module_candidates(cls: Any) -> Iterator[Any]:
    """Every way ``cls`` might be built, most faithful first.

    A class has more than one plausible construction and only one of them
    may be able to forward — see ``fill_optional`` in
    :func:`_construct_module`.  Yielding them lets the caller keep the
    first that survives the *whole* lifecycle instead of committing to
    the first that survives ``__init__``.

    A class registered in :data:`~lucid.test.audit._builders.BUILDERS` is
    built by its builder first: its arguments are objects that relate to
    one another, which no signature says.  A builder that raises is not
    caught — the class is one the audit was told how to build, so a
    failure is a finding, not a gap.
    """
    build = BUILDERS.get(cls)
    if build is not None:
        yield build()
    for fill in (False, True):
        built = _construct_module(cls, fill_optional=fill)
        if built is not None:
            yield built
    quantised = _from_float(cls)
    if quantised is not None:
        yield quantised
    for args, kwargs in list(_CTOR_ARGS) + _ctor_args_with_qconfig():
        try:
            yield cls(*args, **kwargs)
        except Exception:  # noqa: BLE001
            continue


def _try_construct(cls: Any) -> Any:
    return next(_module_candidates(cls), None)


#: Input shapes to try against an unknown module, coarse to fine.
_FORWARD_SHAPES: tuple[tuple[int, ...], ...] = (
    (2, 4),
    (2, 4, 6),
    (2, 3, 6, 6),
    (2, 4, 4),
    (2, 3, 4, 6, 6),
)


def _module_input_shapes(module: Any) -> list[tuple[int, ...]]:
    """Shapes this particular module might accept, its own answer first.

    A constructed module knows its input width — ``Linear(4, 4)`` wants
    a trailing 4, ``Conv2d(3, ...)`` wants 3 channels — and asking it is
    the difference between probing and guessing.  Five fixed shapes left
    84 modules stuck at ``forward`` with ShapeMismatch after they had
    already been built successfully, which is the least useful place to
    stop: the object exists and nothing is asked of it.

    Its own answer comes from the declared width where the module
    exposes one, and from the first parameter otherwise — a weight of
    rank 2 is ``(out, in)`` and one of rank 4 is ``(out, in, kh, kw)``,
    so in both the second axis is what the input must supply.
    """
    widths: list[int] = []
    for attribute in ("in_features", "in_channels", "num_features", "embedding_dim"):
        value = getattr(module, attribute, None)
        if isinstance(value, int) and 0 < value <= 64:
            widths.append(value)
    normalized = getattr(module, "normalized_shape", None)
    if isinstance(normalized, int):
        widths.append(normalized)
    elif isinstance(normalized, (tuple, list)) and normalized:
        widths.append(int(normalized[-1]))

    shapes: list[tuple[int, ...]] = []
    for width in widths:
        shapes += [(2, width), (2, width, 6), (2, width, 6, 6), (2, 6, width)]

    # A lazy layer's parameters exist before their shapes do, and a
    # placeholder's ``(0,)`` says nothing about the input — read as a
    # width it produces a zero-column probe, which is how thirteen lazy
    # classes stopped being reachable at ``forward``.  The declared
    # ``in_channels`` above is ``None`` for them too, so they fall
    # through to the fixed shapes, which is the right answer: only the
    # input can say what the input is.
    try:
        first = next(
            (p for p in module.parameters() if not isinstance(p, lucid.nn.UninitializedParameter)),
            None,
        )
    except Exception:  # noqa: BLE001
        first = None
    if first is not None and hasattr(first, "shape"):
        dims = tuple(int(d) for d in first.shape)
        if len(dims) == 2:
            shapes += [(2, dims[1]), (2, 6, dims[1])]
        elif len(dims) == 3:
            shapes.append((2, dims[1], 6))
        elif len(dims) == 4:
            shapes.append((2, dims[1], 6, 6))
        elif len(dims) == 5:
            shapes.append((2, dims[1], 4, 6, 6))
        elif len(dims) == 1:
            shapes += [(2, dims[0]), (2, dims[0], 6, 6)]

    shapes += list(_FORWARD_SHAPES)

    # A layer whose name ends in ``1d`` / ``2d`` / ``3d`` states its own
    # spatial rank, and an input below it is not a probe — it is an
    # out-of-bounds read.  ``LPPool1d`` on a ``(2, 4)`` tensor calls
    # ``unfold_dim(x, dim=2, ...)`` on a rank-2 tensor, returns a
    # *larger* tensor than it was given, and corrupts the heap on the way
    # past; the sweep then died several hundred symbols later inside
    # something unrelated.  Recorded as a defect — a layer should refuse
    # a rank it cannot use — and filtered here, because a survey that
    # dies cannot report the defect it found.
    name = type(module).__name__
    match = re.search(r"([123])d$", name)
    if match is not None:
        # ``(N, C, *spatial)``: the batch is not optional in this
        # framework's kernels.  ``N + 1`` let a rank-3 probe reach a 2-D
        # convolution, which answered ``conv: x rank mismatch`` for ten
        # classes — every lazy and every quantised convolution.
        rank = int(match.group(1))
        shapes = [sh for sh in shapes if len(sh) >= rank + 2]

    # Two layers whose input shape is a function of their own
    # construction, and which no width attribute describes.
    factor = getattr(module, "upscale_factor", None)
    if factor is not None:
        # ``PixelShuffle`` moves ``r**2`` channels into the spatial
        # dimensions, so the channel count has to be divisible by it.
        shapes.insert(0, (2, int(factor) ** 2 * 2, 6, 6))
    if name == "Fold":
        # ``(N, C * kH * kW, L)`` — the unfolded form ``Fold`` reassembles,
        # where ``L`` is the number of sliding positions the *output*
        # canvas admits.  Both halves have to be derived from the layer's
        # own arguments: a guessed ``L`` is rejected just as firmly as a
        # guessed channel count, and by a message about the other one.
        def _pair(value: Any, fallback: int) -> tuple[int, int]:
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                return int(value[0]), int(value[1])
            if isinstance(value, int):
                return value, value
            return fallback, fallback

        kh, kw = _pair(getattr(module, "kernel_size", None), 1)
        oh, ow = _pair(getattr(module, "output_size", None), 6)
        sh, sw = _pair(getattr(module, "stride", None), 1)
        positions = ((oh - kh) // max(sh, 1) + 1) * ((ow - kw) // max(sw, 1) + 1)
        if positions > 0:
            shapes.insert(0, (2, 2 * kh * kw, positions))

    seen: set[tuple[int, ...]] = set()
    return [sh for sh in shapes if not (sh in seen or seen.add(sh))]


def _forward_parameters(module: Any) -> list[inspect.Parameter] | None:
    """``forward``'s parameters after ``self``, or ``None`` if unreadable."""
    forward = getattr(type(module), "forward", None)
    if forward is None:
        return None
    try:
        signature = inspect.signature(forward, annotation_format=annotationlib.Format.STRING)
    except Exception:  # noqa: BLE001 - an unreadable signature is not a finding
        return None
    return [
        p
        for name, p in signature.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _unpool_indices(module: Any, shape: tuple[int, ...], rank: int) -> Any:
    """Valid ``MaxUnpool`` indices, built rather than measured.

    Each input element addresses one position in the unpooled output,
    flat within its own spatial plane.  Taking the top-left corner of
    each pooling window keeps every index distinct and in range, which is
    the whole contract: two inputs mapping to one output position, or one
    past the end, is what the layer must not be handed.
    """
    stride = getattr(module, "stride", None) or getattr(module, "kernel_size", 2)
    step = stride[0] if isinstance(stride, (tuple, list)) else int(stride)
    step = max(int(step), 1)

    spatial = shape[-rank:]
    out_spatial = tuple(int(s) * step for s in spatial)
    grids = np.meshgrid(*[np.arange(s) * step for s in spatial], indexing="ij")
    flat = np.zeros(spatial, dtype=np.int64)
    for axis, grid in enumerate(grids):
        flat = flat * out_spatial[axis] + grid if axis else grid.astype(np.int64)
    return _probe.as_int(np.broadcast_to(flat, shape).copy())


def _per_dim(value: Any, rank: int) -> tuple[int, ...]:
    """A pooling hyperparameter as one int per spatial dimension."""
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),) * rank


def _forward_companion(
    name: str, module: Any, primary: Any, shape: tuple[int, ...], cast: Any
) -> list[Any]:
    """Values for one forward argument that is not the input, by name.

    A list because a name can mean more than one thing and the module is
    the only authority on which: a multi-label ``target`` is an integer
    matrix the width of the logits, a single-label one is an integer per
    row, and both spell the parameter ``target``.

    ``forward``'s parameter names carry the same information the free
    functions' do, and reading them is the difference between a probe
    and a guess.  ``MaxUnpool2d.forward(x, indices)`` wants the *indices
    the pooling produced*, not a second float tensor: handed one it
    raised, and three classes went unreached because the fixed ladder
    had no form that could ever satisfy them.
    """
    if name == "indices":
        # Round-tripped through the pooling the layer pairs with, so they
        # are inside the output they address by construction.  The probe
        # input plays the *pooled* side, so the pooling runs on a tensor
        # the size of the unpooled output: pooling the probe itself hands
        # back indices smaller than the input they must match, and the
        # layer rightly refuses them.  That went unnoticed while
        # ``return_indices`` did not exist and the built indices below
        # stood in; its first working run lost all three ``MaxUnpool``
        # modules to the mismatch.
        rank = max(len(shape) - 2, 1)
        pool = {
            1: lucid.nn.functional.max_pool1d,
            2: lucid.nn.functional.max_pool2d,
            3: lucid.nn.functional.max_pool3d,
        }.get(rank)
        if pool is not None:
            with contextlib.suppress(Exception):
                kernel = _per_dim(getattr(module, "kernel_size", 2), rank)
                stride = _per_dim(getattr(module, "stride", None) or kernel, rank)
                padding = _per_dim(getattr(module, "padding", 0), rank)
                unpooled = tuple(
                    (int(n) - 1) * s - 2 * p + k
                    for n, k, s, p in zip(shape[-rank:], kernel, stride, padding)
                )
                source = cast(_probe.sample("moderate", (*shape[:-rank], *unpooled), 2))
                _, indices = pool(
                    source, kernel_size=kernel, stride=stride, padding=padding, return_indices=True
                )
                if tuple(indices.shape) == tuple(shape):
                    return [indices]
        return [_unpool_indices(module, shape, rank)]
    if name in ("input_lengths", "target_lengths"):
        batch = shape[1] if len(shape) > 1 else 1
        length = shape[0] if name == "input_lengths" else max(shape[0] // 2, 1)
        return [_probe.as_int(np.full((batch,), length))]
    if name in ("targets", "target", "labels"):
        classes = int(shape[-1]) if len(shape) >= 2 else 2
        draw = _probe.rng(_probe.SEED_B)
        matrix = draw.integers(0, max(classes, 1), shape)
        return [
            _probe.as_int(draw.integers(0, max(classes, 1), (shape[0],))),
            _probe.as_int(matrix),
            # ...and the same labels at int32.  ``multilabel_margin_loss``
            # accepts *only* int32 and refuses the int64 that
            # ``lucid.tensor([[0, 1]])`` produces — recorded as an
            # inconsistency with every other classification loss, and
            # offered here so the layer is reachable at all.
            _probe.as_int(matrix).to(lucid.int32),
            cast(_probe.sample("moderate", shape, 1)),
        ]
    if name in ("key", "value", "other", "target_seq", "memory"):
        return [cast(_probe.sample("moderate", shape, 1 if name == "key" else 2))]
    if name == "var":
        # ``GaussianNLLLoss`` refuses a negative variance, as the reference
        # does, and a draw around zero is one (LCD-216).
        return [cast(_probe.sample("positive", shape, 2))]
    return [cast(_probe.sample("moderate", shape, 1))]


def _ctc_arguments(cast: Any) -> list[list[Any]]:
    """The four tensors ``ctc_loss`` takes, agreeing with each other.

    They are not four independent arguments.  ``targets`` has to hold
    ``sum(target_lengths)`` entries, and no rule that derives one
    parameter at a time can know that — deriving them separately built a
    ``targets`` of 2 against lengths summing to 4, and the kernel read
    past the end of it and took the process with it (SIGBUS, mid-sweep,
    at ``nn.CTCLoss``).

    That the framework crashes rather than refuses is a defect in the
    framework and is recorded as one.  It is also why the probe has to
    be right here rather than only honest: a survey that dies at the
    same symbol every run reports nothing about the 300 symbols after
    it.  The same lesson as ``lu_solve``'s pivots.
    """
    time_steps, batch, classes, labels = 6, 2, 5, 2
    log_probs = cast(_probe.rng(_probe.SEED_X).standard_normal((time_steps, batch, classes)) - 2.0)
    targets = _probe.as_int(_probe.rng(_probe.SEED_B).integers(1, classes, (batch, labels)))
    return [
        [
            log_probs,
            targets,
            _probe.as_int(np.full((batch,), time_steps)),
            _probe.as_int(np.full((batch,), labels)),
        ]
    ]


#: Forwards whose arguments constrain each other, keyed on the set of
#: required parameter names.  Kept to the cases where a joint constraint
#: exists — deriving one name at a time is right for everything else and
#: is what keeps this from becoming the hand-maintained table the rest of
#: this module exists to avoid.
_JOINT_FORWARDS: dict[frozenset[str], Any] = {
    frozenset({"log_probs", "targets", "input_lengths", "target_lengths"}): _ctc_arguments
}


def _forward_from_signature(module: Any, shape: tuple[int, ...], cast: Any) -> list[list[Any]]:
    """Argument lists derived from ``forward``'s own parameter list.

    The fixed ladder below tries one, two and three tensors of the probe
    shape.  ``CTCLoss.forward`` takes four and ``RotaryEmbedding.forward``
    takes none, so neither could ever be reached by trying more of the
    same thing — the same hand-maintained-enumeration failure that held
    the surface at 73.8% and the call ladder at 446 symbols.  The
    interpreter already knows the arity; ask it.
    """
    parameters = _forward_parameters(module)
    if parameters is None:
        return []
    required = [p for p in parameters if p.default is inspect.Parameter.empty]
    joint = _JOINT_FORWARDS.get(frozenset(p.name for p in required))
    if joint is not None:
        return joint(cast)
    if not required:
        # ``RotaryEmbedding``, ``SinusoidalEmbedding`` and its 2-D
        # sibling compute a table from their constructor arguments and
        # take no input at all.  Every ladder form passed one, and all
        # three reported ``forward() takes 1 positional argument but 2
        # were given`` — an arity the ladder could not express rather
        # than a module that could not run.
        return [[]] if not parameters else [[], [cast(_probe.sample("moderate", shape))]]

    primary = cast(_probe.sample("moderate", shape))
    candidates: list[list[Any]] = [[primary]]
    # An optional parameter worth passing.  ``MaxUnpool`` once took
    # ``output_size=None`` and raised "output_size is required" on the
    # default; it now infers the size, and the explicit form keeps the
    # path that honours a caller's size reached as well.
    trailing: list[Any] = []
    if type(module).__name__.startswith("MaxUnpool") and any(
        p.name == "output_size" for p in parameters
    ):
        rank = max(len(shape) - 2, 1)
        stride = getattr(module, "stride", None) or getattr(module, "kernel_size", 2)
        step = stride[0] if isinstance(stride, (tuple, list)) else int(stride)
        trailing.append(tuple(int(s) * max(int(step), 1) for s in shape[-rank:]))
    for parameter in required[1:]:
        choices = _forward_companion(parameter.name, module, primary, shape, cast)
        if not choices:
            return []
        candidates = [[*prefix, choice] for prefix in candidates for choice in choices[:4]][:8]
    if trailing:
        candidates = [[*args, *trailing] for args in candidates] + candidates
    return candidates


def _forward_inputs(module: Any, shape: tuple[int, ...]) -> list[list[Any]]:
    """Argument lists to try for one shape.

    Three things the single float64 tensor could not reach: a module
    whose forward takes a pair (every loss: ``input`` and ``target``) or
    a triple (attention: query, key, value); one whose parameters are
    float32, where a float64 probe is a DtypeMismatch and not a defect;
    and an embedding, which indexes and wants integers.
    """
    if getattr(module, "num_embeddings", None) is not None:
        rows = int(module.num_embeddings)
        idx = _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, rows, shape[:2]))
        return [[idx]]

    dtypes = [_probe.as_f64]
    # Buffers as well as parameters.
    #
    # A quantised layer holds its weights as int8 and its scales as
    # float32, so ``parameters()`` reports nothing float and the probe
    # stayed at float64 — which the transposed convolutions then
    # rejected with ``DtypeMismatch: expected float64, got float32``,
    # naming the probe's own dtype as the expectation.
    try:
        floats = [
            t
            for t in list(module.parameters()) + [b for _, b in module.named_buffers()]
            if str(getattr(t, "dtype", "")).endswith("float32")
        ]
        if floats:
            dtypes.insert(0, _probe.as_f32)
    except Exception:  # noqa: BLE001
        pass

    out: list[list[Any]] = []
    for cast in dtypes:
        out += _forward_from_signature(module, shape, cast)
    for cast in dtypes:
        one = cast(_probe.sample("moderate", shape, 0))
        two = cast(_probe.sample("moderate", shape, 1))
        three = cast(_probe.sample("moderate", shape, 2))
        out += [[one], [one, two], [one, two, three]]

        # A classification loss wants class *indices*, not a second float
        # tensor of the same shape: ``CrossEntropyLoss``, ``NLLLoss``,
        # ``CTCLoss`` and the margin losses all read the second argument
        # as an integer label per row and raised TypeError on a float one.
        if len(shape) >= 2:
            classes = int(shape[-1])
            labels = _probe.as_int(
                _probe.rng(_probe.SEED_B).integers(0, max(classes, 1), (shape[0],))
            )
            out.append([one, labels])
            # ...and the pair losses want a +/-1 sign per row.
            signs = cast(_probe.rng(_probe.SEED_A).choice([-1.0, 1.0], (shape[0],)))
            out.append([one, two, signs])
        # A probability loss reads both operands as probabilities and
        # refuses either outside [0, 1], as the reference does — every
        # pair above is drawn around zero, so ``BCELoss`` took none of
        # them (LCD-367).
        out.append(
            [cast(_probe.sample("small_pos", shape, 0)), cast(_probe.sample("small_pos", shape, 1))]
        )
    return out


def _try_forward(module: Any) -> tuple[Any, str]:
    # The *first* failure, not the last.
    #
    # ``_module_input_shapes`` puts the module's own answer first and the
    # five fixed fallbacks last, so reporting the last one reported the
    # least likely shape every time: fourteen classes all said
    # ``forward: (2, 3, 4, 6, 6): TypeError``, which is a 5-D probe
    # against ``PixelShuffle`` and says nothing.  Keeping the first
    # failure names the shape the module asked for, and the message with
    # it — the type alone hid that ``Upsample`` wanted a ``scale_factor``
    # its constructor had never been given.
    first = ""
    for shape in _module_input_shapes(module):
        for args in _forward_inputs(module, shape):
            try:
                out = module(*args)
            except Exception as exc:  # noqa: BLE001
                if not first:
                    first = f"{shape}: {type(exc).__name__}: {str(exc)[:70]}"
                continue
            tensor = out if hasattr(out, "shape") else None
            if tensor is None and isinstance(out, (tuple, list)) and out:
                # RNNs and attention return ``(output, state)``; the
                # first element is the one the axes can measure.
                tensor = out[0] if hasattr(out[0], "shape") else None
            if tensor is None:
                for attr in ("logits", "sample", "last_hidden_state"):
                    tensor = getattr(out, attr, None)
                    if tensor is not None:
                        break
            if tensor is not None and hasattr(tensor, "shape"):
                return tensor, str(shape)
            if not first:
                first = f"{shape}: returned {type(out).__name__}, nothing measurable"
    return None, first or "no input shape worked"


#: The core numeric axes, cheapest first.
CORE_AXES: tuple[Axis, ...] = (
    EntryPointAxis(),
    BroadcastAxis(),
    NonFiniteAxis(),
    EdgeAxis(),
    DtypeAxis(),
    DeviceAxis(),
    GradientAxis(),
    CreateGraphAxis(),
    SecondGradientAxis(),
    ModuleAxis(),
    OptimAxis(),
)

# Imported at the bottom: the stability and subsystem axes subclass Axis
# and are registered here, so a single ``ALL_AXES`` stays the one place a
# run is defined.  Everything they need from this module is already bound
# by the time the import executes.
from lucid.test.audit._axes_data import DATA_AXES  # noqa: E402
from lucid.test.audit._axes_stability import STABILITY_AXES  # noqa: E402
from lucid.test.audit._axes_state import STATE_AXES  # noqa: E402
from lucid.test.audit._axes_subsystem import SUBSYSTEM_AXES  # noqa: E402

#: Every axis, in the order a full run executes them — cheapest first, so
#: a ``--fail-fast`` run surfaces the loud problems before the slow ones.
#: ``SUBSYSTEM_AXES`` goes last because it ends with the smoke axis, which
#: is the floor rather than a question.
ALL_AXES: tuple[Axis, ...] = (
    *CORE_AXES[:6],
    *STABILITY_AXES,
    *CORE_AXES[6:],
    *DATA_AXES,
    *STATE_AXES,
    *SUBSYSTEM_AXES,
)


def axis_by_name(name: str) -> Axis | None:
    return next((a for a in ALL_AXES if a.name == name), None)


def axis_names() -> list[str]:
    return [a.name for a in ALL_AXES]


__all__ = ["ALL_AXES", "CORE_AXES", "Axis", "Context", "axis_by_name", "axis_names"]
