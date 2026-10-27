"""Pure-Python composite ops layered on the engine.

Composites live here so they share one consistent layering rule:

* They are built from registered engine primitives or from other
  composites — never from raw MLX / Accelerate calls.
* Autograd is automatic — every primitive used here already has a
  registered backward, so composites inherit it.
* No C++ engine changes are required to add a composite.

Categories (one file each):

* ``elementwise`` — aliases (``absolute``, ``divide``…) and special
  elementwise functions (``expm1``, ``sinc``, ``heaviside``, ``logit``…).
* ``reductions`` — NaN-safe reductions (``nansum``, ``nanmean``…).
* ``shape`` — axis swaps, stacks, splits, ``rot90``, ``vander``…
* ``blas`` — ``addmm`` family, ``mv``, ``ger``, ``block_diag``…
* ``predicates`` — ``is_storage`` / ``isin`` / ``conj`` / etc.
* ``dtype`` — ``result_type`` / ``promote_types`` / ``can_cast``.
* ``constants`` — ``pi`` / ``e`` / ``inf`` / ``nan`` / ``newaxis``.
* ``statistics`` — ``quantile`` / ``nanquantile`` / ``cov`` / ``corrcoef`` / ``cdist`` / ``bincount`` / ``multinomial``.
* ``indexing`` — ``index_fill`` / ``index_add`` / ``index_copy`` / ``scatter_reduce`` / ``masked_scatter``.
* ``integer`` — ``gcd`` / ``lcm``, on integer tensors only.
* ``spectral`` — ``stft`` / ``istft`` and the ``*_window`` factories, which forward to ``lucid.signal.windows``.

The single source of truth for "which composite names get re-exported
to the top level" is ``COMPOSITE_NAMES`` below.  The top-level
``lucid/__init__.py`` consumes that set via its lazy-loader registry.
"""

from lucid._ops.composite import (
    constants,
    elementwise,
    reductions,
    shape,
    blas,
    predicates,
    dtype,
    statistics,
    indexing,
    integer,
    complex as complex_mod,
    spectral,
)

# Re-export everything in each submodule's ``__all__`` so callers can do
# ``from lucid._ops.composite import addmm`` without having to know which
# subfile it lives in.  The actual surface lives in the submodules — this
# is purely a flat-namespace convenience.
_SUBMODULES = (
    constants,
    elementwise,
    reductions,
    shape,
    blas,
    predicates,
    dtype,
    statistics,
    indexing,
    integer,
    complex_mod,
    spectral,
)

_seen: set[str] = set()
_names: list[str] = []
for _m in _SUBMODULES:
    for _n in getattr(_m, "__all__", ()):
        if _n in _seen:
            raise RuntimeError(f"composite name collision: {_n!r} appears in multiple submodules")
        _seen.add(_n)
        _names.append(_n)
        globals()[_n] = getattr(_m, _n)

#: Public set of names this package exports — the top-level ``lucid``
#: namespace consumes this to populate its lazy-loader registry.
COMPOSITE_NAMES: frozenset[str] = frozenset(_names)

__all__ = list(_names) + ["COMPOSITE_NAMES"]
