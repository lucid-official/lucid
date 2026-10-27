"""The split family's preconditions are the reference's (LCD-495).

Three were not, after the size-list fix of LCD-490:

* ``vsplit`` / ``hsplit`` / ``dsplit`` took a section count that does not
  divide the axis and gave ragged pieces — ``tensor_split``'s reading —
  where the reference refuses it; they also took fewer dimensions than
  the reference (``vsplit`` of a 1-D tensor).  A section count was only a
  Python ``int``: ``np.int64(3)`` and ``tensor(3)`` were read as the one
  cut index 3, and ``True`` as 1.
* ``CatTransform(lengths=None)`` cut the axis into equal parts; the
  reference's default is one element per transform, which the axis must
  then hold.  ``constraints.cat`` did the same.
* ``split`` / ``tensor_split`` / ``chunk`` of a 0-d tensor raised the
  engine's ``IndexError``.  The reference raises ``RuntimeError``.  A 0-d
  tensor's dim 0 is in range — its notional axis, by the rule
  :mod:`.test_python_dim_rule` holds — so the refusal is the rank, and it
  is ``InvalidArgument``, as ``diff`` of a 0-d tensor is; a dim past the
  notional axis is still an ``IndexError``.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions.constraints as C
import lucid.distributions.transforms as T
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.parity

Outcome = list[tuple[int, ...]] | type[BaseException]


def _outcome(call: Callable[[], Any]) -> Outcome:
    """The piece shapes, or the class of the refusal: RuntimeError or TypeError."""
    try:
        return [tuple(p.shape) for p in call()]
    except TypeError:
        return TypeError
    except RuntimeError:
        return RuntimeError


_SHAPES = {"vsplit": (4, 6, 2), "hsplit": (4, 6, 2), "dsplit": (2, 2, 6)}


@pytest.mark.parametrize("sections", [1, 2, 3, 4, 5, 6, 0, -1, -2, -4, -5])
@pytest.mark.parametrize("op", list(_SHAPES))
def test_a_section_count_must_divide_the_axis(
    ref: ModuleType, device: str, op: str, sections: int
) -> None:
    shape = _SHAPES[op]
    want = _outcome(lambda: getattr(ref, op)(ref.zeros(shape), sections))
    got = _outcome(lambda: getattr(lucid, op)(lucid.zeros(shape, device=device), sections))
    assert got == want
    if want is RuntimeError:
        with pytest.raises(_C_engine.InvalidArgument):
            getattr(lucid, op)(lucid.zeros(shape, device=device), sections)


@pytest.mark.parametrize(
    ("op", "shape"),
    [
        ("vsplit", ()),
        ("vsplit", (4,)),
        ("hsplit", ()),
        ("hsplit", (4,)),
        ("dsplit", (4,)),
        ("dsplit", (2, 4)),
    ],
    ids=str,
)
def test_each_named_split_needs_its_rank(
    ref: ModuleType, device: str, op: str, shape: tuple[int, ...]
) -> None:
    want = _outcome(lambda: getattr(ref, op)(ref.zeros(shape), 2))
    got = _outcome(lambda: getattr(lucid, op)(lucid.zeros(shape, device=device), 2))
    assert got == want


#: A section count spelled as something other than a Python ``int``.
_COUNTS: dict[str, Callable[[ModuleType], Any]] = {
    "numpy int": lambda lib: np.int64(3),
    "0-d tensor": lambda lib: lib.tensor(3),
    "1-D tensor of indices": lambda lib: lib.tensor([1, 4]),
    "bool": lambda lib: True,
}


@pytest.mark.parametrize(
    ("op", "count"),
    # The reference's vsplit takes no tensor of indices; Lucid's reads one
    # as tensor_split does, so that pairing has no reference to agree with.
    [("tensor_split", c) for c in _COUNTS]
    + [("vsplit", c) for c in _COUNTS if c != "1-D tensor of indices"],
)
def test_any_integer_counts_sections(ref: ModuleType, device: str, op: str, count: str) -> None:
    want = _outcome(lambda: getattr(ref, op)(ref.zeros(6, 2), _COUNTS[count](ref)))
    x = lucid.zeros(6, 2, device=device)
    got = _outcome(lambda: getattr(lucid, op)(x, _COUNTS[count](lucid)))
    assert got == want


_CUTS: dict[str, Callable[[Any, Any, int], Any]] = {
    "split": lambda lib, t, d: lib.split(t, 1, dim=d),
    "split sizes": lambda lib, t, d: lib.split(t, [1], dim=d),
    "Tensor.split": lambda lib, t, d: t.split(1, dim=d),
    "tensor_split": lambda lib, t, d: lib.tensor_split(t, 1, dim=d),
    "tensor_split indices": lambda lib, t, d: lib.tensor_split(t, [0], dim=d),
    "chunk": lambda lib, t, d: lib.chunk(t, 1, dim=d),
}


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("op", list(_CUTS))
def test_a_0d_tensor_cannot_be_cut(ref: ModuleType, device: str, op: str, dim: int) -> None:
    with pytest.raises(RuntimeError, match="1-dimensional"):
        _CUTS[op](ref, ref.tensor(1.0), dim)
    with pytest.raises(_C_engine.InvalidArgument, match="1-dimensional"):
        _CUTS[op](lucid, lucid.tensor(1.0, device=device), dim)


@pytest.mark.parametrize("dim", [1, -2])
@pytest.mark.parametrize("op", list(_CUTS))
def test_a_0d_tensors_dim_past_its_axis_is_still_an_index_error(
    device: str, op: str, dim: int
) -> None:
    with pytest.raises(IndexError, match="axis out of range"):
        _CUTS[op](lucid, lucid.tensor(1.0, device=device), dim)


# ── CatTransform and constraints.cat ─────────────────────────────────────────


def test_cat_transform_defaults_to_one_element_per_transform(ref: ModuleType, device: str) -> None:
    rt = ref.distributions.transforms
    transform = T.CatTransform([T.ExpTransform(), T.AffineTransform(1.0, 2.0)])
    reference = rt.CatTransform([rt.ExpTransform(), rt.AffineTransform(1.0, 2.0)])
    assert transform.lengths == reference.lengths == [1, 1]
    got = transform(lucid.tensor([0.5, -1.0], device=device))
    want = reference(ref.tensor([0.5, -1.0]))
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-6)
    with pytest.raises(AssertionError):
        reference(ref.zeros(4))
    for call in (transform, transform.inv):
        with pytest.raises(_C_engine.InvalidArgument, match="sum exactly to 4"):
            call(lucid.zeros(4, device=device))


def test_cat_constraint_defaults_to_one_element_per_constraint(
    ref: ModuleType, device: str
) -> None:
    """Equal parts of a 4-long axis answered 4 checks; the reference, 2."""
    values = [1.0, -1.0, -5.0, -6.0]
    rc = ref.distributions.constraints
    got = C.cat([C.positive, C.real]).check(lucid.tensor(values, device=device))
    want = rc.cat([rc.positive, rc.real]).check(ref.tensor(values))
    assert got.tolist() == want.tolist() == [True, True]
