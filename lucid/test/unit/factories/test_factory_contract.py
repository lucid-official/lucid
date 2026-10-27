"""Every factory builds its value in the final dtype, then makes it a leaf.

``lucid.tensor([1.5, 2.0], dtype=lucid.bfloat16, requires_grad=True)``
came back a *non-leaf*: the value was made as a float32 leaf and narrowed
afterwards, so the caller held the narrowing's output and ``.grad`` never
filled — training a bfloat16 parameter did nothing, silently.  And the
fill went through a C double: ``full`` rounded ``2**60 + 1`` and refused
a complex value outright, and NumPy scalars lost their dtype
(``tensor(np.bool_(True))`` was float32).

The matrix below holds every factory to the contract on every dtype and
both devices; the value rows pin exactness.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]
ALL_DTYPES = [
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool,
]
_FLOATING = {
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
}
_COMPLEX = {lucid.complex64, lucid.complex128}
# MLX has neither (H3), so a Metal tensor of these is refused by design.
_METAL_ABSENT = {lucid.float64, lucid.complex128}

type Factory = Callable[[lucid.dtype, str, bool], lucid.Tensor]


def _like(fn: Callable[..., lucid.Tensor], *args: object) -> Factory:
    def make(dtype: lucid.dtype, device: str, rg: bool) -> lucid.Tensor:
        src = lucid.zeros(2, 3, device=device)
        return fn(src, *args, dtype=dtype, device=device, requires_grad=rg)

    return make


FACTORIES: dict[str, Factory] = {
    "tensor_list": lambda d, dev, rg: lucid.tensor(
        [1.5, 2.0], dtype=d, device=dev, requires_grad=rg
    ),
    "tensor_scalar": lambda d, dev, rg: lucid.tensor(1.5, dtype=d, device=dev, requires_grad=rg),
    "tensor_ndarray": lambda d, dev, rg: lucid.tensor(
        np.array([1.5, 2.0]), dtype=d, device=dev, requires_grad=rg
    ),
    "tensor_tensor": lambda d, dev, rg: lucid.tensor(
        lucid.ones(2, device=dev), dtype=d, device=dev, requires_grad=rg
    ),
    "full": lambda d, dev, rg: lucid.full((2, 3), 1, dtype=d, device=dev, requires_grad=rg),
    "zeros": lambda d, dev, rg: lucid.zeros(2, 3, dtype=d, device=dev, requires_grad=rg),
    "ones": lambda d, dev, rg: lucid.ones(2, 3, dtype=d, device=dev, requires_grad=rg),
    "empty": lambda d, dev, rg: lucid.empty(2, 3, dtype=d, device=dev, requires_grad=rg),
    "eye": lambda d, dev, rg: lucid.eye(3, dtype=d, device=dev, requires_grad=rg),
    "arange": lambda d, dev, rg: lucid.arange(0, 4, dtype=d, device=dev, requires_grad=rg),
    "linspace": lambda d, dev, rg: lucid.linspace(0, 1, 4, dtype=d, device=dev, requires_grad=rg),
    "logspace": lambda d, dev, rg: lucid.logspace(0, 1, 4, dtype=d, device=dev, requires_grad=rg),
    "rand": lambda d, dev, rg: lucid.rand(2, 3, dtype=d, device=dev, requires_grad=rg),
    "randn": lambda d, dev, rg: lucid.randn(2, 3, dtype=d, device=dev, requires_grad=rg),
    "zeros_like": _like(lucid.zeros_like),
    "ones_like": _like(lucid.ones_like),
    "empty_like": _like(lucid.empty_like),
    "full_like": _like(lucid.full_like, 1),
    "rand_like": _like(lucid.rand_like),
    "randn_like": _like(lucid.randn_like),
}

# What the reference refuses too: integer and bool draws, a bool or
# complex ``arange``, a bool ``linspace`` / ``logspace``.
_DRAWS = {"rand", "randn", "rand_like", "randn_like"}


def _reference_refuses(name: str, dtype: lucid.dtype) -> bool:
    if name in _DRAWS:
        return dtype not in _FLOATING
    if name == "arange":
        return dtype is lucid.bool or dtype in _COMPLEX
    if name in ("linspace", "logspace"):
        return dtype is lucid.bool
    return False


# Engine gaps the reference does not have.  Strict: a fix turns the
# xfail into a failure, and the entry must come out.
_ENGINE_GAPS: set[tuple[str, lucid.dtype, str]] = set()


def _cases() -> list[object]:
    out: list[object] = []
    for name in FACTORIES:
        for dtype in ALL_DTYPES:
            if _reference_refuses(name, dtype):
                continue
            for device in DEVICES:
                if device == "metal" and dtype in _METAL_ABSENT:
                    continue
                for rg in (False, True) if dtype in _FLOATING else (False,):
                    marks = []
                    if (name, dtype, device) in _ENGINE_GAPS:
                        marks.append(pytest.mark.xfail(raises=NotImplementedError, strict=True))
                    out.append(
                        pytest.param(
                            name,
                            dtype,
                            device,
                            rg,
                            marks=marks,
                            id=f"{name}-{str(dtype).removeprefix('lucid.')}-{device}-rg{int(rg)}",
                        )
                    )
    return out


def _loss(t: lucid.Tensor) -> lucid.Tensor:
    y = (t * 2).sum()
    return lucid.real(y) if t.dtype in _COMPLEX else y


@pytest.mark.parametrize(("name", "dtype", "device", "rg"), _cases())
def test_factory_returns_a_leaf_in_its_dtype(
    name: str, dtype: lucid.dtype, device: str, rg: bool
) -> None:
    t = FACTORIES[name](dtype, device, rg)
    assert t.dtype is dtype
    assert t.device.type == device
    assert t.is_leaf
    assert t.requires_grad is rg
    if rg:
        _loss(t).backward()
        assert t.grad is not None
        assert t.grad.dtype is dtype
        assert all(g == 2 for g in t.grad.reshape(-1).tolist())


@pytest.mark.parametrize("device", DEVICES)
def test_as_tensor_builds_its_dtype(device: str) -> None:
    for dtype in ALL_DTYPES:
        if device == "metal" and dtype in _METAL_ABSENT:
            continue
        t = lucid.as_tensor([1, 0], dtype=dtype, device=device)
        assert (t.dtype, t.device.type, t.is_leaf) == (dtype, device, True)


@pytest.mark.parametrize(
    "np_dtype",
    [
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ],
)
def test_from_numpy_keeps_the_array_dtype(np_dtype: str) -> None:
    t = lucid.from_numpy(np.array([1, 0], dtype=np_dtype))
    assert t.dtype is getattr(lucid, np_dtype)
    assert t.is_leaf and not t.requires_grad


# ── exact values ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [2**60 + 1, 2**53 + 1, -(2**63), 2**63 - 1])
def test_int64_fill_is_exact(device: str, value: int) -> None:
    assert lucid.full((3,), value, dtype=lucid.int64, device=device).tolist() == [value] * 3
    src = lucid.zeros(3, device=device)
    assert lucid.full_like(src, value, dtype=lucid.int64).tolist() == [value] * 3
    assert lucid.tensor([value], dtype=lucid.int64, device=device).item() == value


@pytest.mark.parametrize("device", DEVICES)
def test_python_int_literal_is_exact(device: str) -> None:
    t = lucid.tensor([2**53 + 1, 2**60 + 1], device=device)
    assert t.dtype is lucid.int64
    assert t.tolist() == [2**53 + 1, 2**60 + 1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128], ids=str)
def test_complex_fill(device: str, dtype: lucid.dtype) -> None:
    if device == "metal" and dtype in _METAL_ABSENT:
        pytest.skip("MLX has no complex128")
    t = lucid.full((2, 2), 2 + 3j, dtype=dtype, device=device, requires_grad=True)
    assert t.dtype is dtype and t.is_leaf
    assert t.tolist() == [[2 + 3j, 2 + 3j], [2 + 3j, 2 + 3j]]
    like = lucid.full_like(lucid.zeros(3, device=device), -1j, dtype=dtype)
    assert like.tolist() == [-1j] * 3


@pytest.mark.parametrize("device", DEVICES)
def test_full_like_fills_in_the_requested_dtype(device: str) -> None:
    src = lucid.zeros(2, dtype=lucid.int32, device=device)
    t = lucid.full_like(src, 2.5, dtype=lucid.float32)
    assert t.dtype is lucid.float32 and t.tolist() == [2.5, 2.5]
    assert lucid.full_like(src, 7).dtype is lucid.int32


@pytest.mark.parametrize(
    ("value", "dtype", "error"),
    [
        (2 + 3j, lucid.float32, TypeError),
        (2 + 3j, lucid.int64, TypeError),
        (300, lucid.int8, OverflowError),
        (2**63, lucid.int64, OverflowError),
        (float("nan"), lucid.int32, ValueError),
        (float("inf"), lucid.int64, ValueError),
    ],
    ids=["complex-f32", "complex-i64", "300-i8", "2^63-i64", "nan-i32", "inf-i64"],
)
def test_fill_refuses_a_value_the_dtype_cannot_hold(
    value: complex, dtype: lucid.dtype, error: type[Exception]
) -> None:
    with pytest.raises(error):
        lucid.full((2,), value, dtype=dtype)
    with pytest.raises(error):
        lucid.tensor([value], dtype=dtype)


def test_fill_truncates_a_finite_float_into_an_integer() -> None:
    assert lucid.full((2,), -2.7, dtype=lucid.int32).tolist() == [-2, -2]
    assert lucid.tensor([2.7, -2.7], dtype=lucid.int64).tolist() == [2, -2]


def test_float_overflow_rounds_to_infinity() -> None:
    t = lucid.tensor([1e6, -1e6], dtype=lucid.float16)
    assert t.float().tolist() == [float("inf"), float("-inf")]
    assert lucid.tensor([1e40], dtype=lucid.float32).item() == float("inf")


# ── dtype inference ──────────────────────────────────────────────────────

_NUMPY_SCALARS: list[tuple[object, lucid.dtype]] = [
    (np.bool_(True), lucid.bool),
    (np.int8(3), lucid.int8),
    (np.int32(3), lucid.int32),
    (np.int64(3), lucid.int64),
    (np.uint8(3), lucid.int16),
    (np.float16(1.5), lucid.float16),
    (np.float32(1.5), lucid.float32),
    (np.float64(1.5), lucid.float64),
    (np.complex64(1 + 2j), lucid.complex64),
    (np.complex128(1 + 2j), lucid.complex128),
]


@pytest.mark.parametrize(("value", "dtype"), _NUMPY_SCALARS, ids=lambda v: repr(v))
def test_numpy_scalar_keeps_its_dtype(value: object, dtype: lucid.dtype) -> None:
    t = lucid.tensor(value)
    assert t.dtype is dtype
    assert t.item() == value
    assert lucid.tensor([value, value]).dtype is dtype


def test_python_floats_follow_the_default_dtype() -> None:
    previous = lucid.get_default_dtype()
    lucid.set_default_dtype(lucid.float64)
    try:
        assert lucid.tensor([1.5]).dtype is lucid.float64
        assert lucid.tensor(1.5).dtype is lucid.float64
        assert lucid.tensor([1j]).dtype is lucid.complex128
        assert lucid.tensor([1, 2]).dtype is lucid.int64
        assert lucid.tensor([]).dtype is lucid.float64
    finally:
        lucid.set_default_dtype(previous)
    assert lucid.tensor([1.5]).dtype is lucid.float32
    assert lucid.tensor([1j]).dtype is lucid.complex64


_INFERENCE_ROWS: list[object] = [
    [np.float16(1), 2.0],
    [np.float16(1), 3],
    [np.int32(1), 2],
    [np.float64(1), 1j],
    [np.bool_(True), np.bool_(False)],
    [np.int8(1), np.int16(2)],
    [True, 2],
    [1, 2.5],
    [1, 2j],
]


@pytest.mark.parametrize("row", _INFERENCE_ROWS, ids=repr)
def test_literal_inference_matches_reference(ref: ModuleType, row: list[object]) -> None:
    want = str(ref.tensor(row).dtype).rsplit(".", 1)[-1]
    assert str(lucid.tensor(row).dtype) == f"lucid.{want}"


@pytest.mark.parametrize(
    "case", ["int64-2^60+1", "int64-2^53+1", "complex64", "complex128", "bf16-leaf"]
)
def test_exact_rows_match_reference(ref: ModuleType, case: str) -> None:
    r = ref
    if case == "int64-2^60+1":
        want = r.full((2,), 2**60 + 1, dtype=r.int64).tolist()
        assert lucid.full((2,), 2**60 + 1, dtype=lucid.int64).tolist() == want
    elif case == "int64-2^53+1":
        want = r.tensor([2**53 + 1]).tolist()
        assert lucid.tensor([2**53 + 1]).tolist() == want
    elif case in ("complex64", "complex128"):
        want = r.full((2,), 2 + 3j, dtype=getattr(r, case)).tolist()
        assert lucid.full((2,), 2 + 3j, dtype=getattr(lucid, case)).tolist() == want
    else:
        rt = r.tensor([1.5, 2.0], dtype=r.bfloat16, requires_grad=True)
        lt = lucid.tensor([1.5, 2.0], dtype=lucid.bfloat16, requires_grad=True)
        assert lt.is_leaf == rt.is_leaf


# ── LCD-319: full() dtype inference when dtype is None ───────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_full_infers_dtype_from_fill_value(device: str, ref: object) -> None:
    for fill, want in [
        (True, lucid.bool),
        (False, lucid.bool),
        (5, lucid.int64),
        (-7, lucid.int64),
        (2.5, lucid.float32),
        (1 + 2j, lucid.complex64),
        (np.int16(9), lucid.int64),
        (np.float16(1.5), lucid.float32),
    ]:
        got = lucid.full((2, 2), fill, device=device)
        ref_got = ref.full((2, 2), fill)  # type: ignore[attr-defined]
        assert got.dtype is want
        assert str(got.dtype).split(".")[-1] == str(ref_got.dtype).split(".")[-1]
    assert lucid.full((2, 2), np.bool_(True), device=device).dtype is lucid.bool
    scalar_i32 = lucid.tensor(4, dtype=lucid.int32, device=device)
    ref_i32 = ref.tensor(4, dtype=ref.int32)  # type: ignore[attr-defined]
    assert lucid.full((2, 2), scalar_i32, device=device).dtype is lucid.int64
    assert ref.full((2, 2), ref_i32).dtype is ref.int64  # type: ignore[attr-defined]
    scalar_f64 = lucid.tensor(1.5, dtype=lucid.float64, device="cpu")
    ref_f64 = ref.tensor(1.5, dtype=ref.float64)  # type: ignore[attr-defined]
    assert lucid.full((2, 2), scalar_f64, device=device).dtype is lucid.float32
    assert ref.full((2, 2), ref_f64).dtype is ref.float32  # type: ignore[attr-defined]


def test_full_follows_default_float_dtype() -> None:
    previous = lucid.get_default_dtype()
    lucid.set_default_dtype(lucid.float64)
    try:
        assert lucid.full((2,), 1.5).dtype is lucid.float64
        assert lucid.full((2,), 1 + 2j).dtype is lucid.complex128
        assert lucid.full((2,), 3).dtype is lucid.int64
        assert lucid.full((2,), True).dtype is lucid.bool
    finally:
        lucid.set_default_dtype(previous)


# ── requires_grad needs a floating or complex dtype, on every path ───────
#
# One rule — bool and integer tensors cannot require gradients — has to
# hold on every way the flag gets set.  It used to live in five copies, and
# ``Tensor(int_tensor, requires_grad=True)`` reached none of them.  The
# matrix walks every public setter, so a new one that skips the rule fails
# here rather than handing back an integer tensor whose backward falls
# over inside the backend.

_NO_GRAD_MSG = "only Tensors of floating point and complex dtype can require gradients"
_NO_GRAD_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64, lucid.bool]

type GradSetter = Callable[[lucid.dtype, str], lucid.Tensor]


def _src(dtype: lucid.dtype, device: str) -> lucid.Tensor:
    return lucid.zeros(2, dtype=dtype, device=device)


def _factory_setter(name: str) -> GradSetter:
    def make(dtype: lucid.dtype, device: str) -> lucid.Tensor:
        return FACTORIES[name](dtype, device, True)

    return make


def _property_setter(dtype: lucid.dtype, device: str) -> lucid.Tensor:
    t = _src(dtype, device)
    t.requires_grad = True
    return t


def _materialize(dtype: lucid.dtype, device: str) -> lucid.Tensor:
    p = lucid.nn.UninitializedParameter()
    p.materialize(_src(dtype, device))
    return p


def _assign_state(dtype: lucid.dtype, device: str) -> lucid.Tensor:
    m = lucid.nn.Linear(2, 1)
    m.load_state_dict(
        {
            "weight": lucid.zeros(1, 2, dtype=dtype, device=device),
            "bias": lucid.zeros(1, dtype=dtype, device=device),
        },
        assign=True,
    )
    return m.weight


_GRAD_SETTERS: dict[str, GradSetter] = {
    **{f"factory-{name}": _factory_setter(name) for name in FACTORIES},
    "Tensor-list": lambda d, dev: lucid.Tensor([1, 0], dtype=d, device=dev, requires_grad=True),
    "Tensor-tensor": lambda d, dev: lucid.Tensor(_src(d, dev), requires_grad=True),
    "Tensor-tensor-cast": lambda d, dev: lucid.Tensor(
        lucid.zeros(2, device=dev), dtype=d, requires_grad=True
    ),
    "Tensor-tensor-move": lambda d, dev: lucid.Tensor(
        lucid.zeros(2, dtype=d), device=dev, requires_grad=True
    ),
    "tensor-tensor": lambda d, dev: lucid.tensor(_src(d, dev), requires_grad=True),
    "requires_grad_": lambda d, dev: _src(d, dev).requires_grad_(),
    "requires_grad-setter": _property_setter,
    "Parameter": lambda d, dev: lucid.nn.Parameter(_src(d, dev)),
    "UninitializedParameter-materialize": _materialize,
    "load_state_dict-assign": _assign_state,
    "new_tensor": lambda d, dev: _src(d, dev).new_tensor([1, 0], requires_grad=True),
    "new_zeros": lambda d, dev: _src(d, dev).new_zeros(2, requires_grad=True),
    "new_ones": lambda d, dev: _src(d, dev).new_ones(2, requires_grad=True),
    "new_empty": lambda d, dev: _src(d, dev).new_empty(2, requires_grad=True),
    "new_full": lambda d, dev: _src(d, dev).new_full((2,), 1, requires_grad=True),
}


def _setter_cases(dtypes: list[lucid.dtype]) -> list[object]:
    out: list[object] = []
    for setter in _GRAD_SETTERS:
        factory = setter.removeprefix("factory-")
        for dtype in dtypes:
            if factory in FACTORIES and _reference_refuses(factory, dtype):
                continue
            for device in DEVICES:
                if device == "metal" and dtype in _METAL_ABSENT:
                    continue
                out.append(
                    pytest.param(
                        setter,
                        dtype,
                        device,
                        id=f"{setter}-{str(dtype).removeprefix('lucid.')}-{device}",
                    )
                )
    return out


@pytest.mark.parametrize(("setter", "dtype", "device"), _setter_cases(_NO_GRAD_DTYPES))
def test_requires_grad_is_refused_on_bool_and_integer(
    setter: str, dtype: lucid.dtype, device: str
) -> None:
    with pytest.raises(RuntimeError, match=_NO_GRAD_MSG):
        _GRAD_SETTERS[setter](dtype, device)


@pytest.mark.parametrize(("setter", "dtype", "device"), _setter_cases(sorted(_FLOATING, key=str)))
def test_requires_grad_is_accepted_on_floating_and_complex(
    setter: str, dtype: lucid.dtype, device: str
) -> None:
    t = _GRAD_SETTERS[setter](dtype, device)
    assert (t.dtype, t.device.type) == (dtype, device)
    assert t.requires_grad and t.is_leaf


@pytest.mark.parametrize("device", DEVICES)
def test_unsigned_input_widens_and_still_refuses_requires_grad(device: str) -> None:
    arr = np.array([1, 0], dtype=np.uint8)
    with pytest.raises(RuntimeError, match=_NO_GRAD_MSG):
        lucid.Tensor(arr, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match=_NO_GRAD_MSG):
        lucid.tensor(arr, device=device, requires_grad=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", _NO_GRAD_DTYPES, ids=str)
def test_a_refused_flag_leaves_the_tensor_unchanged(device: str, dtype: lucid.dtype) -> None:
    t = _src(dtype, device)
    for refused in (
        lambda: t.requires_grad_(True),
        lambda: setattr(t, "requires_grad", True),
        lambda: lucid.Tensor(t, requires_grad=True),
        lambda: lucid.nn.Parameter(t),
    ):
        with pytest.raises(RuntimeError, match=_NO_GRAD_MSG):
            refused()
        assert not t.requires_grad and t.is_leaf
    assert t.requires_grad_(False) is t
    t.requires_grad = False
    assert not t.requires_grad


@pytest.mark.parametrize("target", ["cpu", "metal"])
@pytest.mark.parametrize("dtype", _NO_GRAD_DTYPES, ids=str)
def test_a_cast_to_bool_or_integer_drops_requires_grad(dtype: lucid.dtype, target: str) -> None:
    # A cast ends the graph at a discrete result rather than refusing: the
    # source's flag must not carry over, on the same device or across one.
    x = lucid.randn(3, requires_grad=True)
    for y in (x.to(target, dtype), x.to(target).to(dtype)):
        assert (y.dtype, y.device.type) == (dtype, target)
        assert not y.requires_grad
    with lucid.no_grad():
        assert not x.to(target, dtype).requires_grad
    kept = x.to(target, lucid.float16)
    assert kept.requires_grad
