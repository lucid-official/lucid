"""``nn.functional.pad`` is the one ``pad``, and it refuses as the reference does.

``lucid.pad`` and ``Tensor.pad``, since removed, advertised ``mode=`` but
padded only with a constant, while ``F.pad`` had every mode (LCD-473).  The reference has
neither of the first two, so they are gone and ``F.pad`` is the only path.

``F.pad`` read its spec loosely: an odd-length one dropped its last entry,
one longer than two per dimension wrapped round onto the leading ones, a
negative entry was ignored by the non-constant modes and read across rows
by the constant one, and its refusals were ``ValueError`` where the
reference raises ``RuntimeError`` (``NotImplementedError`` for an unknown
mode).  Each is held here to the reference, values and gradients alike.
"""

from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.parity

_MODES = ["constant", "reflect", "replicate", "circular"]

#: ``(shape, padding)`` the reference pads in every mode: one pair for a
#: 2-D or 3-D input, two for a 3-D or 4-D one, three for a 4-D or 5-D one.
_CASES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((2, 5), (2, 1)),
    ((1, 2, 5), (0, 3)),
    ((1, 2, 5), (-1, 2)),
    ((1, 2, 5), (2, -2)),
    ((2, 3, 4), (1, 2, 2, 0)),
    ((1, 2, 3, 4), (-1, 1, 2, -1)),
    ((1, 1, 3, 3, 4), (1, 1, 2, 0, 0, 2)),
]


def _values(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + 1.0


@pytest.mark.parametrize(("shape", "padding"), _CASES, ids=str)
@pytest.mark.parametrize("mode", _MODES)
def test_every_mode_matches_the_reference(
    ref: ModuleType,
    device: str,
    mode: str,
    shape: tuple[int, ...],
    padding: tuple[int, ...],
) -> None:
    values = _values(shape)
    x = lucid.tensor(values, device=device, requires_grad=True)
    xr = ref.tensor(values, requires_grad=True)
    got = F.pad(x, padding, mode=mode)
    want = ref.nn.functional.pad(xr, padding, mode=mode)
    np.testing.assert_array_equal(got.numpy(), want.detach().numpy())
    weights = _values(tuple(want.shape))
    (got * lucid.tensor(weights, device=device)).sum().backward()
    (want * ref.tensor(weights)).sum().backward()
    assert x.grad is not None
    np.testing.assert_array_equal(x.grad.numpy(), xr.grad.numpy())


def test_a_constant_pad_takes_its_value_and_crops(ref: ModuleType, device: str) -> None:
    values = _values((2, 3))
    got = F.pad(lucid.tensor(values, device=device), (-1, 2, 1, -1), value=-4.0)
    want = ref.nn.functional.pad(ref.tensor(values), (-1, 2, 1, -1), value=-4.0)
    np.testing.assert_array_equal(got.numpy(), want.numpy())


#: name -> ``(shape, padding, mode)`` the reference refuses.
_REFUSED: dict[str, tuple[tuple[int, ...], tuple[int, ...], str]] = {
    "reflect as wide as the dim": ((1, 2, 3), (3, 0), "reflect"),
    "reflect wider on the right": ((1, 2, 3), (0, 4), "reflect"),
    "reflect of a length-1 dim": ((1, 2, 1), (1, 1), "reflect"),
    "circular wrapping twice": ((1, 2, 3), (4, 0), "circular"),
    "replicate of an empty dim": ((2, 0), (1, 1), "replicate"),
    "odd padding length": ((1, 2, 3), (1,), "constant"),
    "padding longer than the rank": ((2, 3), (1, 1, 1, 1, 1, 1), "constant"),
    "a 0-d input": ((), (1, 1), "constant"),
    "a crop past the dim": ((1, 2, 3), (-2, -2), "constant"),
}


@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_spec_the_reference_refuses_is_refused(
    ref: ModuleType, device: str, case: str
) -> None:
    shape, padding, mode = _REFUSED[case]
    with pytest.raises(RuntimeError):
        ref.nn.functional.pad(ref.zeros(shape), padding, mode=mode)
    # Both a RuntimeError, as the reference's, and the ValueError it was.
    with pytest.raises(_C_engine.InvalidArgument):
        F.pad(lucid.zeros(shape, device=device), padding, mode=mode)


def test_an_unknown_mode_is_not_implemented(ref: ModuleType, device: str) -> None:
    with pytest.raises(NotImplementedError, match="mirror"):
        ref.nn.functional.pad(ref.zeros(1, 2, 3), (1, 1), mode="mirror")
    with pytest.raises(NotImplementedError, match="mirror"):
        F.pad(lucid.zeros(1, 2, 3, device=device), (1, 1), mode="mirror")


@pytest.mark.parametrize("owner", [lucid, lucid.Tensor], ids=["lucid", "Tensor"])
def test_there_is_one_pad(ref: ModuleType, owner: Any) -> None:
    """``nn.functional.pad`` only — the reference has no other."""
    reference_owner = ref if owner is lucid else ref.Tensor
    assert not hasattr(reference_owner, "pad")
    assert not hasattr(owner, "pad")


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_non_constant_mode_refuses_value(
    ref: ModuleType, device: str, mode: str
) -> None:
    """LCD-507: non-constant padding modes reject an explicit ``value``."""
    with pytest.raises(RuntimeError, match="doesn't take in value argument"):
        ref.nn.functional.pad(ref.zeros(1, 2, 3), (1, 1), mode=mode, value=1.0)
    with pytest.raises(
        _C_engine.InvalidArgument, match="doesn't take in value argument"
    ):
        F.pad(lucid.zeros(1, 2, 3, device=device), (1, 1), mode=mode, value=1.0)
    with pytest.raises(
        _C_engine.InvalidArgument, match="doesn't take in value argument"
    ):
        F.pad(lucid.zeros(1, 2, 3, device=device), (1, 1), mode=mode, value=0.0)


@pytest.mark.parametrize(
    "widths",
    [[(0, 0), (0, 0), (-1, 1)], [(0, 0), (0, 0), (1, -1)], [(-1, 0), (0, 0), (0, 0)]],
)
def test_engine_pad_refuses_negative_widths(
    device: str, widths: list[tuple[int, int]]
) -> None:
    """LCD-507: ``_C_engine.pad`` refuses negative widths instead of reading across rows."""
    x = lucid.arange(6, dtype=lucid.float32, device=device).reshape(1, 2, 3)
    with pytest.raises(_C_engine.InvalidArgument, match="non-negative"):
        _C_engine.pad(x._impl, widths, 0.0)
