"""Input construction: domains, probes and the numeric helpers.

Two things here are load-bearing.

**Domains.**  Ops are defined on different sets, and guessing per-op by
name is its own source of error, so a check tries a ladder of domains and
keeps the first on which the forward and backward passes are both finite.
The ladder deliberately avoids sampling near a pole: a draw that lands on
``x = 6e-4`` makes ``1/x`` and ``psi`` disagree with central differences
at 1e-4 for reasons that have nothing to do with either op.

**Seeds.**  Outputs are contracted against a *fixed random* covector,
never ``ones``.  A constant seed makes softmax's gradient identically
zero, so a finite-difference check passes while testing nothing — the
failure mode the VACUOUS status exists to name.
"""

import contextlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: Default probe shape.  Small enough that a per-element finite-difference
#: sweep is cheap, large enough to exercise broadcasting and reductions.
SHAPE: tuple[int, int] = (2, 3)

#: Where the covectors come from.  Fixed so a run is reproducible and a
#: finding can be re-derived by hand.
SEED_A = 11
SEED_B = 23
SEED_X = 7


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── domains ──────────────────────────────────────────────────────────────────
# Tried in order.  ``moderate`` first because most ops accept it; the rest
# narrow onto the sets that log / sqrt / acosh / atanh need.  None of them
# samples within 0.05 of a pole.

#: Each takes the shape and a seed.  The seed is a parameter rather than
#: a closed-over constant so a second operand can be given a *different*
#: fixed draw — see :func:`sample`.
DOMAINS: dict[str, Callable[[Sequence[int], int], np.ndarray]] = {
    "moderate": lambda s, k: rng(k).uniform(-1.2, 1.2, s) + 0.15,
    "positive": lambda s, k: rng(k).uniform(0.35, 1.75, s),
    "unit": lambda s, k: rng(k).uniform(-0.75, 0.75, s),
    "gt_one": lambda s, k: rng(k).uniform(1.3, 2.6, s),
    "small_pos": lambda s, k: rng(k).uniform(0.08, 0.48, s),
    "wide": lambda s, k: rng(k).standard_normal(s) * 2.0,
}

#: The probe that finds NaN-swallowing.  ``relu`` returned 0 for the first
#: entry on the CPU and NaN on Metal; no well-conditioned sample can show
#: that, which is why device parity needs its own vector.
NON_FINITE = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 2.0], dtype=np.float64)

#: Every dtype the framework claims, for the coverage axis.
DTYPES: tuple[str, ...] = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)

#: Tried only where the real ladder found nothing.  ``angle``, ``imag``,
#: ``real`` and ``view_as_real`` are defined *only* on these, so over the
#: real ladder each correctly refused all eight and the axis reported "no
#: dtype accepted" — four complex ops with no answered cell on the one
#: axis whose subject is which dtypes an op takes.
#:
#: Deliberately not in :data:`DTYPES`.  Complex arithmetic works on Metal
#: (MLX has it) and raises ``cpu_backend::binary: dtype not supported`` on
#: the CPU, so a global complex probe makes the device-symmetry check
#: report that single backend fact **four hundred times**, once per op.
#: One asymmetry is one finding.
COMPLEX_DTYPES: tuple[str, ...] = ("complex64", "complex128")


def dtype_of(name: str) -> Any:
    return getattr(lucid, name, None)


# ── tensor factories ─────────────────────────────────────────────────────────


def tensor_of(array: Any, dtype: Any = None, device: str = "cpu") -> Any:
    """``lucid.tensor(array, dtype=, device=)`` from a contiguous copy, shape kept.

    Every audit tensor built from an array goes through here.  Two things
    turned a 0-d probe into a 1-element one (LCD-391): ``np.ascontiguousarray``
    always returns at least one dimension, and ``lucid.tensor`` given a 0-d
    ndarray together with ``dtype=`` or ``device=`` returns shape ``(1,)`` as
    well (LCD-399, the converter's bug).  The caller's shape is restored here,
    so a finding is about the op under test, not the converter.
    """
    host = np.asarray(array)
    out = lucid.tensor(np.ascontiguousarray(host), dtype=dtype, device=device)
    return out.reshape(()) if host.ndim == 0 and out.ndim != 0 else out


def _as(array: Any, np_dtype: Any, dtype: Any, device: str) -> Any:
    return tensor_of(np.asarray(array, dtype=np_dtype), dtype, device)


def as_f64(array: Any, device: str = "cpu") -> Any:
    """A float64 tensor — the working precision for every numeric check."""
    return _as(array, np.float64, lucid.float64, device)


def as_f32(array: Any, device: str = "cpu") -> Any:
    return _as(array, np.float32, lucid.float32, device)


def as_complex64(array: Any, device: str = "cpu") -> Any:
    """A complex64 tensor — what the fft family accepts."""
    return _as(array, np.complex64, lucid.complex64, device)


def as_complex(array: Any, device: str = "cpu") -> Any:
    return _as(array, np.complex128, lucid.complex128, device)


def as_int(array: Any, device: str = "cpu") -> Any:
    return _as(array, np.int64, lucid.int64, device)


def covector(n: int, seed: int = SEED_A) -> np.ndarray:
    """The weights an output is contracted against.

    Random, never constant.  See the module docstring.
    """
    return rng(seed).standard_normal(max(n, 1))


def sample(domain: str, shape: Sequence[int] = SHAPE, variant: int = 0) -> np.ndarray:
    """A probe from ``domain``.  ``variant`` picks a different draw.

    The seed is fixed so a run is reproducible and a finding can be
    re-derived by hand — which meant two operands built the same way got
    the *same numbers*, and every binary op was probed with ``a == b``
    everywhere.  That is the one input where a comparison has no
    derivative: ``maximum(a, a)`` sits exactly on its tie, so the analytic
    gradient reports a convention (Lucid gives 1 and 0, the reference
    splits 0.5 and 0.5) and a central difference reports the average of
    the two one-sided limits.  They cannot agree there, and the
    disagreement says nothing about the op.

    Still deterministic — the variant selects a different fixed seed, not
    a fresh one.
    """
    return np.asarray(DOMAINS[domain](tuple(shape), SEED_X + variant), dtype=np.float64)


# ── numeric helpers ──────────────────────────────────────────────────────────


def to_numpy(value: Any) -> np.ndarray | None:
    """Whatever an op returned, as an array, or ``None``.

    Not only tensors.  ``x.ndim``, ``lucid.is_floating_point(x)`` and
    ``x.size()`` answer with a plain int, bool or tuple, and a survey that
    only recognises tensors reports every predicate and every shape
    accessor as uncallable — 30-odd symbols marked as gaps that were
    working perfectly.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        # An array is already the answer.
        #
        # It fell through every branch and came back ``None``, so
        # ``Tensor.numpy`` — whose entire contract is to hand one back —
        # read as "returned ndarray, nothing measurable" and skipped all
        # eleven numeric axes.  So did every bridge function that answers
        # in numpy rather than in a tensor.
        return value
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:  # noqa: BLE001 - surveying, not asserting
            return None
    if isinstance(value, bool | int | float | complex):
        return np.asarray(value)
    if isinstance(value, tuple | list):
        if value and all(isinstance(v, bool | int | float) for v in value):
            return np.asarray(value)
        for item in value:
            got = to_numpy(item)
            if got is not None:
                return got
        return None
    return None


def contract(out: Any, weights: np.ndarray) -> Any:
    """Reduce any output to a differentiable scalar against ``weights``."""
    if isinstance(out, tuple | list):
        out = next((o for o in out if hasattr(o, "shape")), None)
    if out is None or not hasattr(out, "shape"):
        raise TypeError("op did not return a tensor")
    flat = out.reshape(-1)
    n = int(flat.shape[0])
    if n == 0:
        raise ValueError("op returned an empty tensor")
    if n > weights.size:
        # More output than covector.
        #
        # Every caller asks for 64 weights, and ``weights[:n]`` is a
        # *shorter* slice when the output is bigger — so the multiply
        # raised ``ShapeMismatch (broadcast): expected [216], got [64]``
        # and the gradient axes reported it as though the op were at
        # fault.  Around fifty cells, all of them ops whose output is
        # simply larger than the probe: conv2d, batch_norm, interpolate,
        # the 3-D poolings, pad.
        #
        # Extended from the same generator rather than tiled, so the
        # weights stay independent draws and a longer covector is a
        # superset of a shorter one — a run remains reproducible and a
        # finding re-derivable by hand.
        weights = rng(SEED_A).standard_normal(n)
    if flat.is_complex():
        # A complex scalar is not a real-valued loss. Probe both lanes with
        # independent covectors instead of implicitly discarding one of them.
        imaginary_weights = rng(SEED_A + 1).standard_normal(n)
        return (
            lucid.real(flat) * as_f64(weights[:n]) + lucid.imag(flat) * as_f64(imaginary_weights)
        ).sum()
    return (flat * as_f64(weights[:n])).sum()


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-free disagreement, safe on zeros."""
    scale = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)), 1e-12)
    return float(np.abs(a - b).max(initial=0.0) / scale)


def quadratic_shrink(coarse: float, fine: float, ratio: float = 10.0) -> bool:
    """Whether refining the step shrank the disagreement like ``h**2``.

    The discriminator between a finite-difference artefact and a real
    formula error.  Truncation falls by ``ratio**2`` when ``h`` falls by
    ``ratio``; a wrong derivative does not move.  A generous band, because
    round-off starts to dominate once the disagreement approaches
    ``sqrt(eps)``.
    """
    if not math.isfinite(coarse) or not math.isfinite(fine) or coarse <= 0.0:
        return False
    if fine <= 1e-11:
        return True
    return (coarse / fine) > (ratio**1.5)


def finite_difference(
    evaluate: Callable[[np.ndarray], float],
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    """Central differences of a scalar functional, one entry at a time."""
    flat = base.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (evaluate(plus.reshape(base.shape)) - evaluate(minus.reshape(base.shape))) / (
            2.0 * step
        )
    return out


def metal_available() -> bool:
    """Whether the device axis can run at all."""
    try:
        lucid.tensor(np.zeros((1,), dtype=np.float32), device="metal")
    except Exception:  # noqa: BLE001
        return False
    return True


@contextlib.contextmanager
def preserved_globals() -> Iterator[None]:
    """Restore the process-wide switches around one probe.

    The survey calls every public symbol, and three of them write state
    the whole framework reads: ``set_default_dtype``,
    ``set_default_device`` and ``set_grad_enabled``.  A successful call
    to any of them re-tunes every check that runs afterwards — the
    survey would still report, but on a framework configured differently
    from the one it claims to be describing, and nothing in the output
    would say so.  Grad is the worst of the three: left disabled, every
    later gradient check finds no graph and reports the framework as
    non-differentiable.
    """
    dtype_before = lucid.get_default_dtype()
    device_before = lucid.get_default_device()
    grad_before = lucid.is_grad_enabled()
    try:
        yield
    finally:
        if lucid.get_default_dtype() is not dtype_before:
            lucid.set_default_dtype(dtype_before)
        if lucid.get_default_device() != device_before:
            lucid.set_default_device(device_before)
        if lucid.is_grad_enabled() != grad_before:
            lucid.set_grad_enabled(grad_before)


def numpy_of(name: str) -> Any:
    """The numpy scalar type behind one of :data:`DTYPES`."""
    return {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }[name]


def refuses_a_value(exc: BaseException) -> bool:
    """Whether a refusal is about the values handed in rather than their dtype.

    Integer division by zero is the case: ``floor_divide``, ``fmod`` and
    ``remainder`` take every integer dtype, the CPU and the reference refuse
    a zero divisor (LCD-159), and Metal answers one with a defined value.
    The probe's float companions mostly truncate to 0 at an integer dtype,
    so counting that refusal as "dtype not supported" reported the CPU as
    missing four dtypes it has.  The op dispatched on the dtype; the dtype is
    supported.  Both the dtype axis and the contract generator ask through
    here, so they keep answering the same question.
    """
    return isinstance(exc, ZeroDivisionError) or "ZeroDivisionError" in str(exc)


def dtype_args(call: Any, name: str, build: Callable[..., Any]) -> list[Any]:
    """One call's arguments, rebuilt at the dtype ``name``.

    Shared by the dtype axis and by the contract generator so the two
    ask the *same question*.  The axis compares cpu against metal, which
    says the two disagree but never which is right; the generator answers
    that by replaying this identical invocation against the reference
    framework.  If they built their arguments differently the comparison
    would be between two different calls, and the contract would be
    evidence for nothing.

    ``build(array, follow)`` makes one tensor: ``follow`` true means "at
    the dtype under test", false means "keep the array's own dtype".

    Two rules are load-bearing, both learned from mass false positives:

    - every tensor argument is *rebuilt*, never moved, so a convolution
      does not fail because its weights stayed behind;
    - only *float* companions follow the primary's dtype.  An integer
      companion is an index, a size or a mask, and casting it to the
      dtype under test makes the call invalid rather than testing it.
    """
    # The primary moves to |x| + 1 — positive and away from 0, so an integer
    # or boolean rebuild stays a valid operand — unless its values are part
    # of the op's contract (``Call.domain``: a probability); moved out of
    # it, the question became "does the op take an invalid value", which
    # the CPU refuses and Metal does not check.
    in_domain = getattr(call, "domain", None) is not None
    args: list[Any] = []
    for index, value in enumerate(call.args):
        if index == call.primary:
            args.append(build(call.base if in_domain else np.abs(call.base) + 1.0, True))
        else:
            args.append(_rebuild(value, build))
    return args


def _rebuild(value: Any, build: Callable[..., Any]) -> Any:
    """One companion argument, at the dtype under test.

    Recurses into lists and tuples.  ``vjp(f, primals, cotangents)`` takes
    its tensors *inside* sequences, so a rebuild that only looked at the
    top level left them at float64 on the CPU — and float64 does not
    exist on Metal, so the op reported as unsupported there for a reason
    that belonged to the probe.
    """
    if isinstance(value, (list, tuple)):
        rebuilt = [_rebuild(item, build) for item in value]
        return type(value)(rebuilt) if isinstance(value, tuple) else rebuilt
    if not hasattr(value, "to"):
        return value
    companion = to_numpy(value)
    if companion is None:
        return value
    return build(companion, companion.dtype.kind in "fc")


def dtype_kwargs(call: Any, build: Callable[..., Any]) -> dict[str, Any]:
    """The same rebuild, for the keyword arguments.

    They were left where they were.  A tensor passed by keyword stayed on
    the CPU while every positional one moved, so
    ``multi_head_attention_forward`` reported a device mismatch on its own
    projection weight — a fault in the probe that read as one in the op.
    """
    return {name: _rebuild(value, build) for name, value in call.kwargs.items()}


__all__ = [
    "DOMAINS",
    "DTYPES",
    "NON_FINITE",
    "SHAPE",
    "as_complex",
    "as_complex64",
    "as_f32",
    "as_f64",
    "as_int",
    "contract",
    "covector",
    "dtype_args",
    "dtype_kwargs",
    "dtype_of",
    "finite_difference",
    "metal_available",
    "numpy_of",
    "preserved_globals",
    "quadratic_shrink",
    "refuses_a_value",
    "relative",
    "rng",
    "sample",
    "to_numpy",
]
