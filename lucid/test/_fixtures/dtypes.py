"""Dtype fixtures — drive precision sweeps.

``float_dtype`` is the workhorse: every numerical test parametrizes
over ``float32`` and ``float64``.  ``float_dtype_extended`` adds
``float16`` / ``bfloat16`` for tests that explicitly cover the narrow
floats.

Integer-only tests (bitwise ops, indexing, etc.) use ``int_dtype``.
"""

from collections.abc import Sequence

import pytest

import lucid

_FLOAT_DTYPES: Sequence[lucid.dtype] = (lucid.float32, lucid.float64)
_FLOAT_DTYPES_EXTENDED: Sequence[lucid.dtype] = (
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
)
_INT_DTYPES: Sequence[lucid.dtype] = (lucid.int8, lucid.int16, lucid.int32, lucid.int64)


@pytest.fixture(params=_FLOAT_DTYPES, ids=[str(d) for d in _FLOAT_DTYPES])
def float_dtype(request: pytest.FixtureRequest) -> lucid.dtype:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=_FLOAT_DTYPES_EXTENDED, ids=[str(d) for d in _FLOAT_DTYPES_EXTENDED])
def float_dtype_extended(request: pytest.FixtureRequest) -> lucid.dtype:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=_INT_DTYPES, ids=[str(d) for d in _INT_DTYPES])
def int_dtype(request: pytest.FixtureRequest) -> lucid.dtype:
    return request.param  # type: ignore[no-any-return]
