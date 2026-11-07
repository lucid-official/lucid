"""The max / min NaN-gradient table (LCD-878), shared by the eager parity test and the
compiled one.

Each reduction is written once over a framework module ``m`` — ``lucid`` or the
reference — so the two sides run the same spelling.  A slice holding a NaN is
differentiated three ways, by the call:

* ``x.max()`` / ``x.min()`` with no ``dim``: the NaN elements split the gradient;
* ``amax`` / ``amin`` / ``aminmax`` / ``matrix_norm``: every element of the slice is NaN;
* ``vector_norm(x, +-inf)``: the slice gets zeros.

``max(dim)`` gathers at the index it returns.  The data put a NaN in each position of
a row, two NaNs, an all-NaN row, ties and a plain row, next to a clean second row.
"""

import math
from collections.abc import Callable
from types import ModuleType

_NAN = math.nan
_INF = math.inf

#: ``(framework, tensor) -> tensor``.
Reduction = Callable[[ModuleType, object], object]


def _linalg(m: ModuleType) -> ModuleType:
    return m.linalg


REDUCTIONS: dict[str, Reduction] = {
    "max": lambda m, t: t.max(),
    "min": lambda m, t: t.min(),
    "max-dim": lambda m, t: t.max(dim=-1)[0],
    "min-dim": lambda m, t: t.min(dim=-1)[0],
    "amax-dim": lambda m, t: m.amax(t, dim=-1),
    "amin-dim": lambda m, t: m.amin(t, dim=-1),
    "amax-dim-keepdim": lambda m, t: m.amax(t, dim=-1, keepdim=True),
    "amax-all": lambda m, t: m.amax(t),
    "amin-all": lambda m, t: m.amin(t),
    "amax-dim0": lambda m, t: m.amax(t, dim=0),
    "aminmax": lambda m, t: m.stack(list(m.aminmax(t))),
    "aminmax-dim": lambda m, t: m.stack(list(m.aminmax(t, dim=1))),
    "vector_norm-inf": lambda m, t: _linalg(m).vector_norm(t, _INF),
    "vector_norm-neg-inf": lambda m, t: _linalg(m).vector_norm(t, -_INF),
    "vector_norm-inf-dim": lambda m, t: _linalg(m).vector_norm(t, _INF, dim=1),
    "matrix_norm-inf": lambda m, t: _linalg(m).matrix_norm(t, _INF),
    "matrix_norm-1": lambda m, t: _linalg(m).matrix_norm(t, 1),
}

#: Singular values of a NaN matrix; no half-precision SVD.
SPECTRAL: dict[str, Reduction] = {"matrix_norm-2": lambda m, t: _linalg(m).matrix_norm(t, 2)}

DATA: dict[str, list[list[float]]] = {
    "nan-first": [[_NAN, 1.0, 1.0], [2.0, 2.0, 0.0]],
    "nan-mid": [[1.0, _NAN, 1.0], [2.0, 2.0, 0.0]],
    "nan-last": [[1.0, 1.0, _NAN], [2.0, 2.0, 0.0]],
    "two-nan": [[_NAN, 1.0, _NAN], [2.0, 2.0, 0.0]],
    "all-nan-row": [[_NAN, _NAN, _NAN], [2.0, 2.0, 0.0]],
    "ties": [[1.0, 1.0, 0.0], [2.0, 3.0, 1.0]],
    "ordinary": [[1.0, 3.0, 2.0], [-2.0, 0.5, 4.0]],
}
