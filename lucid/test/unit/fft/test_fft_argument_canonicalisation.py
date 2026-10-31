"""Every ``lucid.fft`` function reads its arguments the same way (LCD-767).

``lucid.fft._spec`` is the one place ``n`` / ``s`` / ``dim`` / the input and
its dtype are read and checked, so each rule is held here across the whole
public surface rather than for the one function it was first seen on:

* ``-1`` in ``s`` keeps the input's size (the half-spectrum readers keep
  ``2 * (m - 1)`` on the last axis);
* an empty ``dim`` / a 0-d input is the identity for ``fftn`` / ``ifftn``
  and a refusal naming the op for the other n-D transforms;
* ``n`` / ``s`` / ``dim`` that are not integers are a ``TypeError``, never
  a truncation, and so is an input that is not a tensor;
* ``bfloat16`` and ``float16`` are promoted to ``float32`` by every entry
  point alike;
* a refusal names the public op, never the kernel behind it.
"""

import re
from types import ModuleType

import pytest

import lucid

ONE_D = ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]
TWO_D = ["fft2", "ifft2", "rfft2", "irfft2", "hfft2", "ihfft2"]
N_D = ["fftn", "ifftn", "rfftn", "irfftn", "hfftn", "ihfftn"]
TRANSFORMS = ONE_D + TWO_D + N_D
MULTI = TWO_D + N_D
#: Transforms that read a half-spectrum: a real input is read as one.
SPECTRUM = {"irfft", "hfft", "irfft2", "hfft2", "irfftn", "hfftn"}
#: Transforms that read a real signal.
REAL_ONLY = {"rfft", "ihfft", "rfft2", "ihfft2", "rfftn", "ihfftn"}
#: The two that are the identity over no axis.
IDENTITY_OVER_NO_AXIS = ["fftn", "ifftn"]
KERNELS = {"fftn", "ifftn", "rfftn", "irfftn"}

_X = [[1.0, 2.0, 0.5, -1.0, 3.0], [0.0, 3.0, -2.0, 1.5, 0.5], [4.0, -1.0, 2.0, 0.25, 1.0]]


def _fn(name: str):
    return getattr(lucid.fft, name)


def _x(name: str, device: str) -> lucid.Tensor:
    return lucid.tensor(_X, device=device)


def _flat(t: lucid.Tensor) -> list[complex]:
    return [complex(v) for v in t.detach().to("cpu").flatten().tolist()]


def _close(a: lucid.Tensor, b: lucid.Tensor, tol: float = 1e-4) -> bool:
    return a.shape == b.shape and all(
        abs(p - q) < tol for p, q in zip(_flat(a), _flat(b), strict=True)
    )


def _ref_close(got: lucid.Tensor, want: object, tol: float = 1e-4) -> bool:
    assert got.shape == tuple(want.shape)
    flat = [complex(v) for v in want.detach().flatten().tolist()]
    return all(abs(p - q) < tol for p, q in zip(_flat(got), flat, strict=True))


# ── -1 in s ──────────────────────────────────────────────────────────────────


def _kept(name: str, size: int, last: bool) -> int:
    """What ``-1`` stands for on an axis of ``size``."""
    return 2 * (size - 1) if name in SPECTRUM and last else size


@pytest.mark.parametrize("name", MULTI)
def test_minus_one_in_s_keeps_the_input_size(device: str, name: str) -> None:
    x = _x(name, device)
    got = _fn(name)(x, s=(-1, 3), dim=(-2, -1))
    want = _fn(name)(x, s=(_kept(name, int(x.shape[0]), last=False), 3), dim=(-2, -1))
    assert _close(got, want)


@pytest.mark.parametrize("name", MULTI)
def test_minus_one_in_the_last_slot_keeps_the_last_axis(device: str, name: str) -> None:
    x = _x(name, device)
    got = _fn(name)(x, s=(2, -1), dim=(-2, -1))
    want = _fn(name)(x, s=(2, _kept(name, int(x.shape[-1]), last=True)), dim=(-2, -1))
    assert _close(got, want)


@pytest.mark.parametrize("name", N_D)
def test_minus_one_alone_names_the_last_axis(device: str, name: str) -> None:
    x = _x(name, device)
    got = _fn(name)(x, s=(-1,))
    want = _fn(name)(x, s=(_kept(name, int(x.shape[-1]), last=True),))
    assert _close(got, want)


@pytest.mark.parametrize("name", ONE_D)
def test_n_has_no_minus_one_spelling(device: str, name: str) -> None:
    x = _x(name, device)
    for bad in (-1, 0, -3):
        with pytest.raises(ValueError, match="invalid number of data points"):
            _fn(name)(x, n=bad)


@pytest.mark.parametrize("name", MULTI)
def test_s_below_minus_one_or_zero_is_refused(device: str, name: str) -> None:
    x = _x(name, device)
    for bad in (-2, 0):
        with pytest.raises(ValueError, match="invalid number of data points"):
            _fn(name)(x, s=(bad, 3), dim=(-2, -1))


# ── empty dim and 0-d ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", IDENTITY_OVER_NO_AXIS)
@pytest.mark.parametrize("dim", [[], ()], ids=["list", "tuple"])
def test_an_empty_dim_is_the_identity_as_complex(device: str, name: str, dim: object) -> None:
    x = _x(name, device)
    for norm in (None, "forward", "ortho"):
        got = _fn(name)(x, dim=dim, norm=norm)
        assert got.is_complex()
        assert _close(got, lucid.complex(x, lucid.zeros_like(x)))


@pytest.mark.parametrize("name", IDENTITY_OVER_NO_AXIS)
def test_the_identity_keeps_the_gradient(device: str, name: str) -> None:
    x = _x(name, device).requires_grad_(True)
    lucid.real(_fn(name)(x, dim=[])).sum().backward()
    assert x.grad is not None and _close(x.grad, lucid.ones_like(x))


@pytest.mark.parametrize("name", IDENTITY_OVER_NO_AXIS)
def test_the_identity_of_a_complex_input_is_not_the_input(device: str, name: str) -> None:
    z = lucid.complex(_x(name, device), _x(name, device) * 0.5)
    got = _fn(name)(z, dim=[])
    assert got is not z and _close(got, z)


@pytest.mark.parametrize("name", IDENTITY_OVER_NO_AXIS)
def test_a_0d_input_is_its_own_transform(device: str, name: str) -> None:
    for scalar in (3.0, -1.5):
        got = _fn(name)(lucid.tensor(scalar, device=device))
        assert got.shape == () and got.is_complex()
        assert _flat(got) == [complex(scalar)]


@pytest.mark.parametrize("name", [n for n in N_D if n not in IDENTITY_OVER_NO_AXIS])
@pytest.mark.parametrize("empty", [None, [], ()], ids=["omitted", "list", "tuple"])
def test_the_real_valued_n_d_transforms_refuse_no_axis(
    device: str, name: str, empty: object
) -> None:
    x = lucid.tensor(3.0, device=device) if empty is None else _x(name, device)
    with pytest.raises(ValueError, match=rf"{name}\(\): must transform at least one axis"):
        _fn(name)(x, dim=empty)


@pytest.mark.parametrize("name", ONE_D + TWO_D)
def test_a_0d_input_has_no_axis_for_the_1d_and_2d_forms(device: str, name: str) -> None:
    with pytest.raises(IndexError, match=rf"{name}\(\)"):
        _fn(name)(lucid.tensor(3.0, device=device))


# ── integer arguments ────────────────────────────────────────────────────────

_NOT_INT = [2.5, "4", True, 1.0, (1,)]


@pytest.mark.parametrize("name", ONE_D)
@pytest.mark.parametrize("bad", [2.5, "4", True])
def test_n_must_be_an_int(device: str, name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): n must be an int"):
        _fn(name)(_x(name, device), n=bad)


@pytest.mark.parametrize("name", ONE_D)
@pytest.mark.parametrize("bad", ["1", 1.0, True, (0, 1)])
def test_a_1d_dim_must_be_one_int(device: str, name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): dim must be an int"):
        _fn(name)(_x(name, device), dim=bad)


@pytest.mark.parametrize("name", MULTI)
@pytest.mark.parametrize("bad", [2.5, (2.5, 3), "3", (True, 3)], ids=repr)
def test_s_must_be_ints(device: str, name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): s"):
        _fn(name)(_x(name, device), s=bad, dim=(-2, -1))


@pytest.mark.parametrize("name", MULTI)
@pytest.mark.parametrize("bad", [(0.5, 1), "1", 1.5, (True, 0)], ids=repr)
def test_dim_must_be_ints(device: str, name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): dim must be an int"):
        _fn(name)(_x(name, device), dim=bad)


@pytest.mark.parametrize("name", ["fftshift", "ifftshift"])
@pytest.mark.parametrize("bad", ["1", 0.5, (True,)], ids=repr)
def test_the_shifts_take_int_dims(device: str, name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): dim must be an int"):
        _fn(name)(_x(name, device), dim=bad)


@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
@pytest.mark.parametrize("bad", [2.5, "4", True])
def test_the_frequency_helpers_take_an_int_n(name: str, bad: object) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): n must be an int"):
        _fn(name)(bad)


@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
def test_the_frequency_helpers_take_a_numeric_d(name: str) -> None:
    for bad in ("1", True, None):
        with pytest.raises(TypeError, match=rf"{name}\(\): d must be a number"):
            _fn(name)(4, d=bad)


@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
@pytest.mark.parametrize("dtype", [lucid.int32, lucid.int64, lucid.bool_, lucid.complex64])
def test_the_frequency_helpers_refuse_a_non_floating_dtype(name: str, dtype: lucid.dtype) -> None:
    with pytest.raises(TypeError, match=rf"{name}\(\): dtype must be a floating-point dtype"):
        _fn(name)(4, dtype=dtype)


@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64])
def test_the_frequency_helpers_keep_a_floating_dtype(name: str, dtype: lucid.dtype) -> None:
    assert _fn(name)(4, dtype=dtype).dtype == dtype


# ── input type ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", TRANSFORMS + ["fftshift", "ifftshift"])
def test_the_input_must_be_a_tensor(name: str) -> None:
    for bad in ([1.0, 2.0], (1.0, 2.0), 3.0):
        with pytest.raises(TypeError, match=rf"{name}\(\): input must be a Tensor"):
            _fn(name)(bad)


@pytest.mark.parametrize("name", sorted(REAL_ONLY))
def test_a_real_signal_transform_refuses_a_complex_input(device: str, name: str) -> None:
    z = lucid.complex(_x(name, device), _x(name, device))
    with pytest.raises(TypeError, match=rf"{name}\(\): expects a real-valued input"):
        _fn(name)(z)


# ── dtype policy ─────────────────────────────────────────────────────────────

_HALF = [lucid.float16, lucid.bfloat16]


@pytest.mark.parametrize("name", TRANSFORMS)
@pytest.mark.parametrize("dtype", _HALF, ids=lambda d: d._name)
def test_half_precision_is_promoted_by_every_entry_point(
    device: str, name: str, dtype: lucid.dtype
) -> None:
    x = _x(name, device)
    half = x.to(dtype)
    got = _fn(name)(half)
    want = _fn(name)(half.to(lucid.float32))
    assert got.dtype == want.dtype
    assert got.dtype in (lucid.complex64, lucid.float32)
    assert _close(got, want)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_a_bfloat16_input_has_a_gradient_in_its_own_dtype(device: str, name: str) -> None:
    if name in SPECTRUM:
        pytest.skip("a spectrum input's gradient dtype is covered by the float32 sweeps")
    x = _x(name, device).to(lucid.bfloat16).requires_grad_(True)
    out = _fn(name)(x)
    (lucid.real(out) if out.is_complex() else out).sum().backward()
    assert x.grad is not None and x.grad.dtype == lucid.bfloat16


@pytest.mark.parametrize("name", TRANSFORMS)
def test_integer_and_bool_input_is_promoted(device: str, name: str) -> None:
    ints = lucid.arange(6, device=device).reshape(2, 3)
    flags = lucid.tensor([[True, False], [True, True]], device=device)
    for t in (ints, flags):
        got = _fn(name)(t)
        assert got.dtype in (lucid.complex64, lucid.float32)


# ── messages name the user's op ──────────────────────────────────────────────


def _bad_calls(name: str, device: str):
    x = _x(name, device)
    f = _fn(name)
    calls = [lambda: f([1.0]), lambda: f(x, dim=7), lambda: f(x, norm="bogus-norm")]
    if name in ONE_D:
        calls += [lambda: f(x, n=-1), lambda: f(x, n=2.5), lambda: f(x, dim=(0, 1))]
        calls += [lambda: f(lucid.tensor(1.0, device=device))]
    else:
        calls += [
            lambda: f(x, dim=(1, 1)),
            lambda: f(x, s=(2, 2, 2)),
            lambda: f(x, s=(-2, 2), dim=(-2, -1)),
            lambda: f(x, s=2.5),
        ]
    return calls


@pytest.mark.parametrize("name", TRANSFORMS)
def test_a_refusal_never_names_a_kernel_behind_the_op(device: str, name: str) -> None:
    others = KERNELS - {name}
    for call in _bad_calls(name, device):
        with pytest.raises((TypeError, ValueError, IndexError)) as caught:
            call()
        message = str(caught.value)
        assert "[trace=" not in message, message
        for kernel in others:
            assert not re.search(rf"\b{kernel}\b", message), (kernel, message)


# ── the reference ────────────────────────────────────────────────────────────

_MINUS_ONE = [
    ("fftn", (-1, 3)),
    ("ifftn", (3, -1)),
    ("rfftn", (-1, 3)),
    ("irfftn", (-1, 3)),
    ("irfftn", (3, -1)),
    ("irfftn", (-1, -1)),
    ("hfftn", (-1, 6)),
    ("hfftn", (-1, -1)),
    ("ihfftn", (-1, 3)),
    ("fft2", (-1, 3)),
    ("ifft2", (3, -1)),
    ("rfft2", (-1, 6)),
    ("irfft2", (-1, -1)),
    ("hfft2", (2, -1)),
    ("ihfft2", (-1, -1)),
]


def _operand(m: ModuleType, name: str) -> object:
    data = m.tensor(_X)
    return m.complex(data, data * 0.5) if name in SPECTRUM else data


def _operand_lucid(name: str, device: str) -> lucid.Tensor:
    data = lucid.tensor(_X, device=device)
    return lucid.complex(data, data * 0.5) if name in SPECTRUM else data


@pytest.mark.parity
@pytest.mark.parametrize("name, s", _MINUS_ONE, ids=[f"{n}-{s}" for n, s in _MINUS_ONE])
def test_minus_one_in_s_matches_the_reference(
    ref: ModuleType, device: str, name: str, s: tuple[int, ...]
) -> None:
    got = _fn(name)(_operand_lucid(name, device), s=s)
    want = getattr(ref.fft, name)(_operand(ref, name), s=s)
    assert _ref_close(got, want)


@pytest.mark.parity
@pytest.mark.parametrize("name", IDENTITY_OVER_NO_AXIS)
def test_the_identity_matches_the_reference(ref: ModuleType, device: str, name: str) -> None:
    for dim in ([], ()):
        got = _fn(name)(_operand_lucid(name, device), dim=dim)
        want = getattr(ref.fft, name)(_operand(ref, name), dim=dim)
        assert _ref_close(got, want)
    for scalar in (3.0, -1.5):
        got = _fn(name)(lucid.tensor(scalar, device=device), norm="ortho")
        want = getattr(ref.fft, name)(ref.tensor(scalar), norm="ortho")
        assert got.shape == tuple(want.shape) and _ref_close(got, want)


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n in N_D if n not in IDENTITY_OVER_NO_AXIS])
def test_no_axis_is_refused_as_the_reference_refuses_it(
    ref: ModuleType, device: str, name: str
) -> None:
    with pytest.raises(RuntimeError, match="at least one axis"):
        getattr(ref.fft, name)(_operand(ref, name), dim=[])
    with pytest.raises(RuntimeError, match="at least one axis"):
        _fn(name)(_operand_lucid(name, device), dim=[])


@pytest.mark.parity
@pytest.mark.parametrize("name", ONE_D)
def test_non_int_arguments_are_refused_as_the_reference_refuses_them(
    ref: ModuleType, device: str, name: str
) -> None:
    for kwargs in ({"n": 2.5}, {"n": "4"}, {"dim": "1"}, {"dim": 1.0}, {"n": True}):
        with pytest.raises(TypeError):
            getattr(ref.fft, name)(_operand(ref, name), **kwargs)
        with pytest.raises(TypeError):
            _fn(name)(_operand_lucid(name, device), **kwargs)
    with pytest.raises(TypeError):
        getattr(ref.fft, name)([1.0, 2.0])
    with pytest.raises(TypeError):
        _fn(name)([1.0, 2.0])


@pytest.mark.parity
@pytest.mark.parametrize("name", MULTI)
def test_non_int_sizes_and_dims_are_refused_as_the_reference_refuses_them(
    ref: ModuleType, device: str, name: str
) -> None:
    for kwargs in ({"s": 2.5}, {"s": (2.5, 3)}, {"dim": (0.5, 1)}, {"dim": "1"}):
        with pytest.raises(TypeError):
            getattr(ref.fft, name)(_operand(ref, name), **kwargs)
        with pytest.raises(TypeError):
            _fn(name)(_operand_lucid(name, device), **kwargs)


@pytest.mark.parity
@pytest.mark.parametrize("name", ["fftfreq", "rfftfreq"])
def test_the_frequency_helpers_refuse_what_the_reference_refuses(
    ref: ModuleType, name: str
) -> None:
    with pytest.raises(RuntimeError):
        getattr(ref.fft, name)(4, dtype=ref.int32)
    with pytest.raises(TypeError):
        getattr(ref.fft, name)(2.5)
    with pytest.raises(TypeError):
        _fn(name)(2.5)
    with pytest.raises(TypeError):
        _fn(name)(4, dtype=lucid.int32)


@pytest.mark.parity
@pytest.mark.parametrize("name", ["fftshift", "ifftshift"])
def test_the_shifts_refuse_what_the_reference_refuses(ref: ModuleType, name: str) -> None:
    with pytest.raises(TypeError):
        getattr(ref.fft, name)(ref.tensor(_X), dim="1")
    with pytest.raises(TypeError):
        _fn(name)(lucid.tensor(_X), dim="1")
