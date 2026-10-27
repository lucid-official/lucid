"""The autocast entry of a Python composite (LCD-500).

An engine op applies its row of the autocast policy table
(``lucid/_C/core/AmpPolicyTable.def``) once, at its entry, and runs its
body with autocast off.  A composite written in Python had neither, so
every engine op inside it applied its own row: under a Metal autocast
``local_response_norm`` met its float32 ``sum`` with a float16 ``div``
operand, and ``cosine_similarity`` of two float16 inputs normalised them
in float16 and only reduced in float32.

The one Python owner of that entry.  It lives here, below ``nn``,
``linalg`` and the composites, so any of them can import it; the cast rule
itself stays in the engine (``amp_cast_inputs``).  A composite with a row
enters through :func:`_autocast_entry`; one without a row keeps its
inputs, and still needs its body run with autocast off, or its parts apply
their own rows (``interpolate``'s products ran ``matmul``'s LOWER row) — it
enters through :func:`_autocast_suspended`.  A composite whose operands are
all arguments (a loss) can take :func:`_autocast_entry` around its whole
call with the :func:`_amp_op` decorator.
"""

import annotationlib
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap


@contextmanager
def _autocast_suspended(device: _C_engine.Device) -> Iterator[None]:
    """Autocast off on ``device`` for the body, with no cast at the entry.

    The entry of a composite that has no row in the policy table — the
    reference's op has none either, so it keeps its inputs' dtype — and of
    graph plumbing that does no arithmetic of its own.

    Parameters
    ----------
    device : Device
        The device whose autocast the body runs without.
    """
    if not _C_engine.amp_is_active(device):
        yield
        return
    guard = _C_engine.AutocastGuard(_C_engine.F32, device, False)
    try:
        yield
    finally:
        del guard  # the last reference — ~AutocastGuard restores the prior state


@contextmanager
def _autocast_entry(
    name: str, *inputs: _C_engine.TensorImpl
) -> Iterator[list[_C_engine.TensorImpl]]:
    """``inputs`` cast by ``name``'s row, with autocast off on their device inside.

    Parameters
    ----------
    name : str
        The composite's row in the policy table.  It must be a row: an op
        with none enters through :func:`_autocast_suspended`, so a misspelt
        row cannot quietly become "keep".  ``lucid/test/unit/amp/
        test_composite_autocast_entry.py`` checks every name passed here.
    *inputs : TensorImpl
        Its floating operands; the first one's device selects the column.

    Yields
    ------
    list of TensorImpl
        The operands at the row's dtype — unchanged outside autocast.
    """
    device = inputs[0].device
    if not _C_engine.amp_is_active(device):
        yield list(inputs)
        return
    cast = _C_engine.amp_cast_inputs(name, list(inputs))
    with _autocast_suspended(device):
        yield cast


def _amp_op[**P, R](row: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a composite with ``row``'s entry: :func:`_autocast_entry` around the call.

    Every tensor argument, by position or by keyword, enters cast by the
    row: an optional ``weight`` as well as the input, as the reference's
    autocast casts every floating tensor an op is handed.  A non-tensor
    argument passes through, and an integer tensor keeps its dtype (the row
    casts floating tensors only).  The body then runs with autocast off on
    the arguments' device.  Outside autocast the call goes straight
    through, unbound.

    For a composite whose operands are all its arguments — a loss.  One
    that computes an operand before its row applies uses
    :func:`_autocast_entry` at that point instead.

    Parameters
    ----------
    row : str
        The composite's row in the policy table.  As for
        :func:`_autocast_entry`, it must be a row:
        ``lucid/test/unit/amp/test_composite_autocast_entry.py`` checks every
        name passed here.

    Returns
    -------
    Callable
        The decorator.  What it returns keeps the decorated function's
        signature, name, module and docstring (:func:`functools.wraps`).
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)

        @functools.wraps(fn)
        def at_entry(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _C_engine.amp_is_active():
                return fn(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                # A call that does not bind: the function's own refusal says why.
                return fn(*args, **kwargs)
            held = {
                name: impl
                for name, value in bound.arguments.items()
                if (impl := _promotion.try_unwrap(value)) is not None
            }
            if not held:
                return fn(*args, **kwargs)
            with _autocast_entry(row, *held.values()) as cast:
                for (name, impl), out in zip(held.items(), cast, strict=True):
                    if out is not impl:
                        was_impl = bound.arguments[name] is impl
                        bound.arguments[name] = out if was_impl else _wrap(out)
                return fn(*bound.args, **bound.kwargs)

        return at_entry

    return decorate
