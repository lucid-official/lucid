"""A norm of an integer or bool tensor is refused, for every order (LCD-384).

``vector_norm`` of an int tensor answered by accident: ``ord`` 2 and 0 in
float32, 1 and ``inf`` in the input's own dtype, every other order a
``NotImplementedError`` from ``pow_scalar``.  The reference refuses every
order with "Expected a floating point or complex tensor".  Each norm door
— ``vector_norm``, ``matrix_norm``, ``norm``, ``cdist`` and
``F.normalize`` — now refuses it the same way, with a ``DtypeMismatch``
that is both the reference's ``RuntimeError`` and a ``TypeError``.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

_VECTOR_ORDS = [2, 0, 1, math.inf, -math.inf, 3, 0.5, -1]
_MATRIX_ORDS = ["fro", "nuc", 1, -1, math.inf, -math.inf, 2, -2]
_DISCRETE = ["int8", "int32", "int64", "bool"]
_DATA = [[1, 0], [3, 4]]


def _calls(m: ModuleType) -> list[tuple[str, Callable[[object], object]]]:
    """Every norm door, once per order, as ``(id, call)`` on module ``m``."""
    functional = m.nn.functional if m is not lucid else F
    out: list[tuple[str, Callable[[object], object]]] = []
    out += [(f"vector_norm-{o}", lambda x, o=o: m.linalg.vector_norm(x, o)) for o in _VECTOR_ORDS]
    out += [
        (
            f"vector_norm-dtype-{o}",
            lambda x, o=o: m.linalg.vector_norm(x, o, dtype=m.float32),
        )
        for o in (2, 1)
    ]
    out += [(f"matrix_norm-{o}", lambda x, o=o: m.linalg.matrix_norm(x, o)) for o in _MATRIX_ORDS]
    out += [(f"norm-{o}", lambda x, o=o: m.linalg.norm(x, o)) for o in (None, "fro", 2, 1)]
    out += [
        ("cdist", lambda x: m.cdist(x, x)),
        ("normalize", lambda x: functional.normalize(x)),
    ]
    return out


_IDS = [name for name, _ in _calls(lucid)]


@pytest.mark.parametrize("dtype", _DISCRETE)
@pytest.mark.parametrize("index", range(len(_IDS)), ids=_IDS)
def test_every_norm_refuses_a_discrete_input(
    ref: ModuleType, device: str, dtype: str, index: int
) -> None:
    _, ref_call = _calls(ref)[index]
    _, call = _calls(lucid)[index]
    with pytest.raises(RuntimeError):
        ref_call(ref.tensor(_DATA, dtype=getattr(ref, dtype)))
    x = lucid.tensor(_DATA, dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(_C_engine.DtypeMismatch, match="floating point") as err:
        call(x)
    assert isinstance(err.value, TypeError) and isinstance(err.value, RuntimeError)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("ord", _VECTOR_ORDS, ids=str)
def test_a_floating_input_keeps_its_dtype(
    ref: ModuleType, device: str, dtype: str, ord: float
) -> None:
    data = [[1.5, 0.0], [-3.0, 4.0]]
    want = ref.linalg.vector_norm(ref.tensor(data, dtype=getattr(ref, dtype)), ord)
    got = lucid.linalg.vector_norm(
        lucid.tensor(data, dtype=getattr(lucid, dtype), device=device), ord
    )
    assert str(got.dtype).rsplit(".", 1)[-1] == str(want.dtype).rsplit(".", 1)[-1]
    assert got.item() == pytest.approx(want.item(), rel=2e-3)
