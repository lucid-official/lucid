"""The axis-scatter family refuses an operand of the wrong dtype (LCD-288).

The engine's scatter kernels read ``src``'s buffer as ``self``'s dtype and
the index's as integers.  Handed anything else they reinterpreted the bytes:
``zeros(2, 4).scatter_add(1, [[0, 1], [2, 3]], tensor([[1., 2], [3, 4]],
float64))`` answered 1.875 where 1.0 was meant, and a ``bool`` index
scattered to the wrong slots.  The reference refuses both — a ``src`` whose
dtype is not ``self``'s for every scatter and for ``index_copy``, an index
that is not an integer tensor — and so does the engine now, at the one door
the five ops share (``require_scatter_dtypes`` in
``lucid/_C/ops/gfunc/Gfunc.cpp``).  The refusal is a ``DtypeMismatch``:
a ``RuntimeError``, as the reference raises, and a ``TypeError``.

Lucid takes an ``int32`` index as well as ``int64``, where the reference
takes ``int64`` alone; that is deliberate and not swept here.  The Python
composites ``index_add`` / ``index_copy`` / ``scatter_reduce`` cast ``src``
to ``input``'s dtype before they reach the engine, so they are not swept
either.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

_BASE = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
_INDEX = [[0, 1], [2, 3]]
_SRC = [[1.0, 2.0], [3.0, 4.0]]
_ROWS = [0, 1]  # index_copy's index along dim 0 for the whole-row source

# engine op -> how the reference spells the same write
_OPS: dict[str, Callable[..., object]] = {
    "scatter_add": lambda x, i, s: x.scatter_add(1, i, s),
    "scatter_amax": lambda x, i, s: x.scatter_reduce(1, i, s, "amax"),
    "scatter_amin": lambda x, i, s: x.scatter_reduce(1, i, s, "amin"),
    "scatter_prod": lambda x, i, s: x.scatter_reduce(1, i, s, "prod"),
}

# (self dtype, src dtype) pairs that differ; Metal holds no float64.
_PAIRS = {
    "cpu": [
        ("float32", "float64"),
        ("float64", "float32"),
        ("float32", "float16"),
        ("float32", "int64"),
        ("int32", "int64"),
        ("int64", "int32"),
        ("float32", "bool"),
    ],
    "metal": [
        ("float32", "float16"),
        ("float16", "float32"),
        ("float32", "int32"),
        ("int32", "int64"),
        ("float32", "bool"),
    ],
}

_BAD_INDEX = ["int16", "bool", "float32"]


def _lucid(values: list[object], dtype: str, device: str) -> lucid.Tensor:
    return lucid.tensor(
        values, dtype=getattr(lucid, dtype if dtype != "bool" else "bool_")
    ).to(device)


def _ref(R: ModuleType, values: list[object], dtype: str) -> object:
    return R.tensor(values, dtype=getattr(R, dtype))


def _engine(op: str, x: lucid.Tensor, i: lucid.Tensor, s: lucid.Tensor) -> lucid.Tensor:
    return _wrap(getattr(_C_engine, op)(_unwrap(x), _unwrap(i), _unwrap(s), 1))


def _pair_params() -> list[object]:
    return [
        pytest.param(
            device, op, self_dt, src_dt, id=f"{device}-{op}-{self_dt}<-{src_dt}"
        )
        for device in _DEVICES
        for op in _OPS
        for self_dt, src_dt in _PAIRS[device]
    ]


@pytest.mark.parametrize(("device", "op", "self_dt", "src_dt"), _pair_params())
def test_src_of_another_dtype_is_refused(
    ref: ModuleType, device: str, op: str, self_dt: str, src_dt: str
) -> None:
    with pytest.raises(RuntimeError):
        _OPS[op](
            _ref(ref, _BASE, self_dt),
            _ref(ref, _INDEX, "int64"),
            _ref(ref, _SRC, src_dt),
        )
    x = _lucid(_BASE, self_dt, device)
    i = _lucid(_INDEX, "int64", device)
    s = _lucid(_SRC, src_dt, device)
    with pytest.raises(_C_engine.DtypeMismatch) as info:
        _engine(op, x, i, s)
    assert isinstance(info.value, RuntimeError)
    assert isinstance(info.value, TypeError)
    assert "src must have self's dtype" in str(info.value)


@pytest.mark.parametrize("device", _DEVICES)
def test_the_reported_repro_is_refused_through_the_public_paths(device: str) -> None:
    src_dt = "float64" if device == "cpu" else "float16"
    x = lucid.zeros(2, 4, device=device)
    i = _lucid(_INDEX, "int64", device)
    s = _lucid(_SRC, src_dt, device)
    with pytest.raises(_C_engine.DtypeMismatch):
        lucid.scatter_add(x, 1, i, s)
    with pytest.raises(_C_engine.DtypeMismatch):
        x.scatter_add(1, i, s)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("self_dt", "src_dt"), [("float32", "float16"), ("int32", "int64")]
)
def test_scatter_set_refuses_as_index_copy_does(
    ref: ModuleType, device: str, self_dt: str, src_dt: str
) -> None:
    rows = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    with pytest.raises(RuntimeError):
        _ref(ref, _BASE, self_dt).index_copy(
            0, _ref(ref, _ROWS, "int64"), _ref(ref, rows, src_dt)
        )
    x = _lucid(_BASE, self_dt, device)
    idx = _lucid([[0, 0, 0, 0], [1, 1, 1, 1]], "int64", device)
    s = _lucid(rows, src_dt, device)
    with pytest.raises(_C_engine.DtypeMismatch):
        _wrap(_C_engine.scatter_set(_unwrap(x), _unwrap(idx), _unwrap(s), 0))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", [*_OPS, "scatter_set"])
@pytest.mark.parametrize("index_dt", _BAD_INDEX)
def test_index_that_is_not_an_integer_tensor_is_refused(
    ref: ModuleType, device: str, op: str, index_dt: str
) -> None:
    if op != "scatter_set":
        with pytest.raises(RuntimeError):
            _OPS[op](
                _ref(ref, _BASE, "float32"),
                _ref(ref, _INDEX, index_dt),
                _ref(ref, _SRC, "float32"),
            )
    x = _lucid(_BASE, "float32", device)
    i = _lucid(_INDEX, index_dt, device)
    s = _lucid(_SRC, "float32", device)
    with pytest.raises((_C_engine.DtypeMismatch, _C_engine.IndexError)) as info:
        _engine(op, x, i, s)
    assert "index" in str(info.value)


def _match_params() -> list[object]:
    out = []
    for device in _DEVICES:
        dtypes = ["float32", "float64"] if device == "cpu" else ["float32", "float16"]
        for dt in dtypes:
            for op in _OPS:
                out.append(pytest.param(device, op, dt, id=f"{device}-{op}-{dt}"))
        out.append(
            pytest.param(
                device, "scatter_add", "int64", id=f"{device}-scatter_add-int64"
            )
        )
    return out


@pytest.mark.parametrize(("device", "op", "dtype"), _match_params())
@pytest.mark.parametrize("index_dt", ["int32", "int64"])
def test_matching_dtypes_still_scatter(
    ref: ModuleType, device: str, op: str, dtype: str, index_dt: str
) -> None:
    want = _OPS[op](
        _ref(ref, _BASE, dtype),
        _ref(ref, _INDEX, "int64"),
        _ref(ref, _SRC, dtype),
    )
    got = _engine(
        op,
        _lucid(_BASE, dtype, device),
        _lucid(_INDEX, index_dt, device),
        _lucid(_SRC, dtype, device),
    )
    assert got.dtype == getattr(lucid, dtype)
    assert got.tolist() == want.tolist()
