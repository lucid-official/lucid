"""Pin which of ``interpolate``'s two composite paths a test exercises.

A weighted axis (linear, cubic, antialiased) is applied either as one
dense ``(out, in)`` matrix product or as a sum of gathers, chosen by
how long the axis is against how many taps it reads.  Test-sized inputs
always pick the product, while long axes in real use pick the gathers,
so a test that means to cover both has to force each one — and check
both that the other was not quietly taken and that the forced one ran.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import lucid
from lucid.nn.functional import _interpolate

PATHS = ("dense", "gather")


@contextmanager
def forced_path(monkeypatch: pytest.MonkeyPatch, path: str) -> Iterator[None]:
    """Within the block, apply every weighted axis by ``path`` only.

    Raises ``AssertionError`` if a weighted axis takes the other path, or
    if the block never reaches ``path`` at all (an engine kernel or an
    identity axis served every call, so nothing was tested).
    """
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    real_dense = _interpolate._apply_dense
    real_gathers = _interpolate._apply_gathers
    ran = 0

    def dense(x: lucid.Tensor, axis: int, plan: _interpolate._AxisPlan) -> lucid.Tensor:
        nonlocal ran
        if path != "dense":
            raise AssertionError("the dense path ran where gathers were forced")
        ran += 1
        return real_dense(x, axis, plan)

    def gathers(x: lucid.Tensor, axis: int, taps: _interpolate._FrozenTaps) -> lucid.Tensor:
        nonlocal ran
        # Nearest has no weights and always gathers; it says nothing about
        # which path a weighted axis takes.
        if taps[0][1] is not None:
            if path != "gather":
                raise AssertionError("gathers ran where the dense path was forced")
            ran += 1
        return real_gathers(x, axis, taps)

    # The plans are cached with the path baked in.
    _interpolate._axis_plan.cache_clear()
    _interpolate._dense_matrix.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(_interpolate, "_DENSE_PER_TAP", 1 << 40 if path == "dense" else 0)
        patch.setattr(_interpolate, "_apply_dense", dense)
        patch.setattr(_interpolate, "_apply_gathers", gathers)
        try:
            yield
        finally:
            _interpolate._axis_plan.cache_clear()
            _interpolate._dense_matrix.cache_clear()
    assert ran, f"no weighted axis took the {path} path"
