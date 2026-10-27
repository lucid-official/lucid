"""The variance family on a 0-d tensor, over every way to name its axis.

A 0-d tensor's one axis answers to ``dim`` 0, -1 and ``(0,)``, as the
reference accepts them.  ``_bessel_correct`` read ``shape[0]`` of it for
the divisor and raised ``IndexError``; the axes are now normalised once, in
:func:`lucid._ops._reduce._reduction_axes`, to the whole tensor, so the
divisor counts its one element.  A correction of 1 leaves no degrees of
freedom: ``nan``, with the reference's warning.
"""

import warnings
from types import ModuleType

import pytest

import lucid

_OPS = ["var", "std", "var_mean", "std_mean"]


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("dim", [None, 0, -1, (0,)], ids=str)
@pytest.mark.parametrize("correction", [0, 1])
def test_zero_dim_variance(
    ref: ModuleType, device: str, op: str, dim: object, correction: int
) -> None:
    with warnings.catch_warnings(record=True) as theirs:
        warnings.simplefilter("always")
        want = getattr(ref, op)(ref.tensor(3.0), dim=dim, correction=correction)
    with warnings.catch_warnings(record=True) as ours:
        warnings.simplefilter("always")
        got = getattr(lucid, op)(lucid.tensor(3.0, device=device), dim=dim, correction=correction)
    pairs = zip(got, want, strict=True) if op.endswith("_mean") else [(got, want)]
    for g, w in pairs:
        assert g.shape == tuple(w.shape) == ()
        assert g.tolist() == pytest.approx(w.tolist(), nan_ok=True)

    def degrees(caught: list[warnings.WarningMessage]) -> bool:
        return any("degrees of freedom" in str(w.message) for w in caught)

    assert degrees(ours) == degrees(theirs) == (correction == 1)


def test_a_repeated_zero_dim_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="appears multiple times"):
        lucid.var(lucid.tensor(3.0), dim=(0, -1))
