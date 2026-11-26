"""Factory count, bound, and size arguments coerce single-element integer tensors (LCD-1064).

``lucid.randperm`` and ``lucid.randint`` compared a raw narrow-integer tensor
(``int8`` / ``int16`` / ``int32``) against 64-bit integer bounds, wrapping the
bound into the narrow dtype and raising a nonsense "too large for int64" /
"out of bounds for int64" error.  Other factories let pybind11 coerce via
``Tensor.__int__``, silently accepting ``bool`` and floating tensors as counts
or extents.  All factories now share ``lucid._factories._build._as_int`` as
the single owner.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine

_INT_WIDTHS = ("int8", "int16", "int32", "int64")


def _t(mod: ModuleType, value: object, dtype_name: str, dev: str) -> object:
    return mod.tensor(value, dtype=getattr(mod, dtype_name), device=dev)


FactoryCall = Callable[[ModuleType, object, str], object]

_FACTORIES: dict[str, tuple[FactoryCall, tuple[int, ...]]] = {
    "randperm": (lambda m, n, d: m.randperm(n, device=d), (7,)),
    "randint_high": (lambda m, n, d: m.randint(n, (3,), device=d), (3,)),
    "randint_bounds": (lambda m, n, d: m.randint(0, n, (3,), device=d), (3,)),
    "randint_low": (lambda m, n, d: m.randint(n, 10, (3,), device=d), (3,)),
    "randint_size": (lambda m, n, d: m.randint(0, 5, (n,), device=d), (7,)),
    "randint_like": (lambda m, n, d: m.randint_like(m.zeros(3, device=d), n), (3,)),
    "eye_n": (lambda m, n, d: m.eye(n, device=d), (7, 7)),
    "eye_m": (lambda m, n, d: m.eye(2, n, device=d), (2, 7)),
    "zeros": (lambda m, n, d: m.zeros(n, device=d), (7,)),
    "zeros_tuple": (lambda m, n, d: m.zeros((2, n), device=d), (2, 7)),
    "ones": (lambda m, n, d: m.ones(n, device=d), (7,)),
    "empty": (lambda m, n, d: m.empty(n, device=d), (7,)),
    "full": (lambda m, n, d: m.full((n,), 1.0, device=d), (7,)),
    "rand": (lambda m, n, d: m.rand(n, device=d), (7,)),
    "randn": (lambda m, n, d: m.randn(n, device=d), (7,)),
    "normal": (lambda m, n, d: m.normal(0.0, 1.0, (n,), device=d), (7,)),
    "linspace": (lambda m, n, d: m.linspace(0.0, 1.0, n, device=d), (7,)),
    "logspace": (lambda m, n, d: m.logspace(0.0, 1.0, n, device=d), (7,)),
    "empty_strided": (lambda m, n, d: m.empty_strided((n,), (1,), device=d), (7,)),
}


@pytest.mark.parametrize("name", list(_FACTORIES))
@pytest.mark.parametrize("width", _INT_WIDTHS)
def test_factories_accept_all_integer_tensor_widths(device: str, name: str, width: str) -> None:
    call, expected_shape = _FACTORIES[name]
    arg = _t(lucid, 7, width, device)
    out = call(lucid, arg, device)
    assert out.shape == expected_shape
    assert out.device.type == device


@pytest.mark.parametrize("width", _INT_WIDTHS)
def test_randperm_and_randint_values_with_narrow_int_tensors(device: str, width: str) -> None:
    dt = getattr(lucid, width)
    perm = lucid.randperm(lucid.tensor(7, dtype=dt, device=device), device=device)
    assert perm.dtype == lucid.int64
    assert sorted(perm.tolist()) == list(range(7))

    empty_perm = lucid.randperm(lucid.tensor(0, dtype=dt, device=device), device=device)
    assert empty_perm.shape == (0,)

    # Negative narrow-int low bound used to wrap int64 min into narrow dtype.
    lo = lucid.tensor(-3, dtype=dt, device=device)
    hi = lucid.tensor(4, dtype=dt, device=device)
    drawn = lucid.randint(lo, hi, (32,), device=device)
    assert drawn.dtype == lucid.int64
    assert all(-3 <= v < 4 for v in drawn.tolist())


def test_randperm_and_randint_still_enforce_bounds_on_tensor_args(device: str) -> None:
    with pytest.raises(_C_engine.InvalidArgument, match="negative dimension -1"):
        lucid.randperm(lucid.tensor(-1, dtype=lucid.int8, device=device), device=device)
    with pytest.raises(_C_engine.InvalidArgument, match="too large"):
        lucid.randperm(
            lucid.tensor(129, dtype=lucid.int16, device=device), dtype=lucid.int8, device=device
        )
    with pytest.raises(_C_engine.InvalidArgument, match="low must be less than high"):
        lucid.randint(
            lucid.tensor(5, dtype=lucid.int8, device=device),
            lucid.tensor(5, dtype=lucid.int8, device=device),
            (3,),
            device=device,
        )
    with pytest.raises(_C_engine.InvalidArgument, match="out of bounds"):
        lucid.randint(
            lucid.tensor(0, dtype=lucid.int8, device=device),
            lucid.tensor(200, dtype=lucid.int16, device=device),
            (3,),
            dtype=lucid.int8,
            device=device,
        )


def test_factories_accept_one_element_1d_and_numpy_int_scalars(device: str) -> None:
    class _CustomIndex:
        def __index__(self) -> int:
            return 5

    t1d = lucid.tensor([5], dtype=lucid.int16, device=device)
    assert lucid.randperm(t1d, device=device).shape == (5,)
    assert lucid.randint(0, t1d, (3,), device=device).shape == (3,)
    assert lucid.eye(t1d, device=device).shape == (5, 5)
    assert lucid.zeros(t1d, device=device).shape == (5,)
    assert lucid.full((t1d,), 1.0, device=device).shape == (5,)
    assert lucid.randperm(np.int8(5), device=device).shape == (5,)
    assert lucid.randint(np.int8(5), (3,), device=device).shape == (3,)
    assert lucid.randperm(_CustomIndex(), device=device).shape == (5,)
    assert lucid.zeros(_CustomIndex(), device=device).shape == (5,)

    # Tensor.random_ also flows through _int_bounds -> _as_int.
    buf = lucid.empty(16, dtype=lucid.int64, device=device)
    buf.random_(
        lucid.tensor(-2, dtype=lucid.int8, device=device),
        lucid.tensor(3, dtype=lucid.int8, device=device),
    )
    assert all(-2 <= v < 3 for v in buf.tolist())


def test_empty_strided_refuses_non_sequence_args(device: str) -> None:
    with pytest.raises(TypeError, match="empty_strided"):
        lucid.empty_strided(4, (1,), device=device)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="empty_strided"):
        lucid.empty_strided("4", (1,), device=device)  # type: ignore[arg-type]


def test_new_factory_methods_coerce_narrow_int_tensors(device: str) -> None:
    base = lucid.zeros(1, device=device)
    n = lucid.tensor(4, dtype=lucid.int8, device=device)
    assert base.new_zeros(n).shape == (4,)
    assert base.new_ones(n).shape == (4,)
    assert base.new_empty(n).shape == (4,)
    assert base.new_full((n,), 2.0).shape == (4,)


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_factories_refuse_bool_tensor_with_runtime_error(device: str, name: str) -> None:
    call, _ = _FACTORIES[name]
    with pytest.raises(_C_engine.InvalidArgument):
        call(lucid, lucid.tensor(True, dtype=lucid.bool_, device=device), device)


@pytest.mark.parametrize("name", list(_FACTORIES))
@pytest.mark.parametrize(
    "bad_kind",
    [
        "py_bool",
        "py_float",
        "np_bool",
        "float_tensor",
        "complex_tensor",
        "empty_tensor",
        "multi_tensor",
    ],
)
def test_factories_refuse_non_integer_arguments_with_type_error(
    device: str, name: str, bad_kind: str
) -> None:
    bad_values: dict[str, object] = {
        "py_bool": True,
        "py_float": 5.0,
        "np_bool": np.bool_(True),
        "float_tensor": lucid.tensor(5.0, dtype=lucid.float32, device=device),
        "complex_tensor": lucid.tensor(5.0 + 0j, dtype=lucid.complex64, device=device),
        "empty_tensor": lucid.tensor([], dtype=lucid.int64, device=device),
        "multi_tensor": lucid.tensor([3, 4], dtype=lucid.int64, device=device),
    }
    call, _ = _FACTORIES[name]
    with pytest.raises(TypeError):
        call(lucid, bad_values[bad_kind], device)


@pytest.mark.parity
@pytest.mark.parametrize(
    "name",
    [
        "randperm",
        "randint_high",
        "randint_bounds",
        "randint_low",
        "randint_size",
        "eye_n",
        "eye_m",
        "zeros",
        "full",
        "rand",
        "randn",
        "normal",
    ],
)
@pytest.mark.parametrize("width", _INT_WIDTHS)
def test_integer_tensor_args_match_reference(
    ref: ModuleType, device: str, name: str, width: str
) -> None:
    call, _ = _FACTORIES[name]
    theirs = call(ref, _t(ref, 7, width, "cpu"), "cpu")
    ours = call(lucid, _t(lucid, 7, width, device), device)
    assert tuple(ours.shape) == tuple(theirs.shape)
    assert str(ours.dtype).rsplit(".", 1)[-1] == str(theirs.dtype).rsplit(".", 1)[-1]
