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
enters through :func:`_autocast_suspended`.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from lucid._C import engine as _C_engine


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
