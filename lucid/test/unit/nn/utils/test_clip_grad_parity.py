"""``clip_grad_*`` / ``get_total_norm`` / ``parameters_to_vector`` against the reference (LCD-778).

Three things were wrong, each invisible to a float32, list-of-parameters,
two-norm test: half-precision gradients were squared and summed in half
precision (``|g| = 300`` squares to 90000, which is ``inf`` in ``float16``,
and the clip then zeroed the gradients); a lone ``Tensor`` was listed, so it
was iterated by rows; and the odd corners of the norm orders and clip values
(``-inf``, a negative ``clip_value``, gradients on two devices) answered
differently from the reference.  So the table is dtype x container x order.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn import utils as U
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.param(
    "metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")
)
_DTYPES = ["float32", "float16", "bfloat16"]
_ORDERS = [2.0, 1.0, 0.0, 3.0, 0.5, -0.5, -1.0, -2.0, math.inf, -math.inf]
_G1 = np.array([[30.0, -40.0], [3.0, 4.0]], dtype=np.float32)
_G2 = np.array([1.0, 0.5, 2.0], dtype=np.float32)


def _ours(arr: np.ndarray, dtype: str, device: str = "cpu") -> nn.Parameter:
    p = nn.Parameter(lucid.tensor(arr).to(getattr(lucid, dtype)).to(device))
    p.grad = lucid.tensor(arr).to(getattr(lucid, dtype)).to(device)
    return p


def _theirs(ref, arr: np.ndarray, dtype: str):
    p = ref.nn.Parameter(ref.tensor(arr).to(getattr(ref, dtype)))
    p.grad = ref.tensor(arr).to(getattr(ref, dtype))
    return p


def _f32(t: lucid.Tensor) -> np.ndarray:
    return t.detach().to(lucid.float32).to("cpu").numpy()


def _assert_close(ours: lucid.Tensor, theirs, dtype: str) -> None:
    tol = {"float32": 1e-5, "float16": 2e-3, "bfloat16": 2e-2}[dtype]
    np.testing.assert_allclose(
        _f32(ours), theirs.detach().float().numpy(), rtol=tol, atol=tol, equal_nan=True
    )


def _same_dtype(ours: lucid.Tensor, theirs) -> bool:
    return str(ours.dtype).split(".")[-1] == str(theirs.dtype).split(".")[-1]


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("norm_type", _ORDERS)
@pytest.mark.parametrize("container", ["list", "single"])
def test_clip_grad_norm_matches_reference(
    ref, dtype: str, norm_type: float, container: str
) -> None:
    arrs = [_G1] if container == "single" else [_G1, _G2]
    ours = [_ours(a, dtype) for a in arrs]
    theirs = [_theirs(ref, a, dtype) for a in arrs]
    arg_o = ours[0] if container == "single" else ours
    arg_t = theirs[0] if container == "single" else theirs
    got = U.clip_grad_norm_(arg_o, 1.0, norm_type)
    want = ref.nn.utils.clip_grad_norm_(arg_t, 1.0, norm_type)
    assert got.shape == ()
    assert _same_dtype(got, want)
    _assert_close(got, want, dtype)
    for o, t in zip(ours, theirs, strict=True):
        _assert_close(o.grad, t.grad, dtype)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("norm_type", _ORDERS)
@pytest.mark.parametrize("container", ["list", "single"])
def test_get_total_norm_matches_reference(
    ref, dtype: str, norm_type: float, container: str
) -> None:
    arrs = [_G1] if container == "single" else [_G1, _G2]
    ours = [_ours(a, dtype).grad for a in arrs]
    theirs = [_theirs(ref, a, dtype).grad for a in arrs]
    got = U.get_total_norm(ours[0] if container == "single" else ours, norm_type)
    want = ref.nn.utils.get_total_norm(theirs[0] if container == "single" else theirs, norm_type)
    assert _same_dtype(got, want)
    _assert_close(got, want, dtype)


def test_half_gradients_that_overflow_when_squared_keep_a_finite_norm() -> None:
    """The reported case: ``[300, 400]`` in float16 is norm 500, not ``inf`` and zeros."""
    p = _ours(np.array([300.0, 400.0], dtype=np.float32), "float16")
    norm = U.clip_grad_norm_([p], 1.0)
    assert float(norm.item()) == 500.0
    np.testing.assert_allclose(_f32(p.grad), [0.6, 0.8], atol=2e-3)


@pytest.mark.parity
def test_mixed_precision_gradients_promote_the_norm_dtype(ref) -> None:
    a, b = _G2, np.array([4.0, 3.0], dtype=np.float32)
    ours = U.get_total_norm([_ours(a, "float16").grad, _ours(b, "float32").grad])
    theirs = ref.nn.utils.get_total_norm(
        [_theirs(ref, a, "float16").grad, _theirs(ref, b, "float32").grad]
    )
    assert _same_dtype(ours, theirs)
    # The reference rounds the float16 group's norm to float16 before it joins
    # the float32 one; this keeps it wide, so they agree to half precision only.
    _assert_close(ours, theirs, "float16")


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("clip_value", [-2.0, -1.0, 0.0, 0.5, 1.0, 100.0])
@pytest.mark.parametrize("container", ["list", "single"])
def test_clip_grad_value_matches_reference(
    ref, dtype: str, clip_value: float, container: str
) -> None:
    arrs = [_G1] if container == "single" else [_G1, _G2]
    ours = [_ours(a, dtype) for a in arrs]
    theirs = [_theirs(ref, a, dtype) for a in arrs]
    U.clip_grad_value_(ours[0] if container == "single" else ours, clip_value)
    ref.nn.utils.clip_grad_value_(theirs[0] if container == "single" else theirs, clip_value)
    for o, t in zip(ours, theirs, strict=True):
        _assert_close(o.grad, t.grad, dtype)


@pytest.mark.parity
@pytest.mark.parametrize("norm_type", _ORDERS)
@pytest.mark.parametrize("device", [_METAL])
def test_gradients_on_two_devices_are_reduced_where_they_live(
    ref, norm_type: float, device: str
) -> None:
    ours = [_ours(_G1, "float32"), _ours(_G2, "float32", device)]
    theirs = [_theirs(ref, _G1, "float32"), _theirs(ref, _G2, "float32")]
    got = U.clip_grad_norm_(ours, 1.0, norm_type)
    want = ref.nn.utils.clip_grad_norm_(theirs, 1.0, norm_type)
    assert got.device.type == "cpu"  # the first gradient's device
    _assert_close(got, want, "float32")
    for o, t in zip(ours, theirs, strict=True):
        assert str(o.grad.device) == str(o.device)
        _assert_close(o.grad, t.grad, "float32")
    total = U.get_total_norm([o.grad for o in ours], norm_type)
    want_total = ref.nn.utils.get_total_norm([t.grad for t in theirs], norm_type)
    _assert_close(total, want_total, "float32")


@pytest.mark.parity
@pytest.mark.parametrize("fn", ["clip_grad_norm_", "get_total_norm"])
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_error_if_nonfinite_message_matches_reference(ref, fn: str, bad: float) -> None:
    arr = np.array([1.0, bad], dtype=np.float32)
    ours = _ours(arr, "float32")
    theirs = _theirs(ref, arr, "float32")
    arg_o = [ours] if fn == "clip_grad_norm_" else [ours.grad]
    arg_t = [theirs] if fn == "clip_grad_norm_" else [theirs.grad]
    args = (1.0,) if fn == "clip_grad_norm_" else ()
    with pytest.raises(RuntimeError) as want:
        getattr(ref.nn.utils, fn)(arg_t, *args, error_if_nonfinite=True)
    with pytest.raises(RuntimeError) as got:
        getattr(U, fn)(arg_o, *args, error_if_nonfinite=True)
    assert str(got.value) == str(want.value)
    # Refusing leaves the gradient alone.
    np.testing.assert_array_equal(_f32(ours.grad), arr)


@pytest.mark.parity
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_non_finite_norm_scales_like_the_reference(ref, bad: float) -> None:
    arr = np.array([1.0, bad], dtype=np.float32)
    ours, theirs = _ours(arr, "float32"), _theirs(ref, arr, "float32")
    got = U.clip_grad_norm_([ours], 1.0)
    want = ref.nn.utils.clip_grad_norm_([theirs], 1.0)
    _assert_close(got, want, "float32")
    _assert_close(ours.grad, theirs.grad, "float32")


@pytest.mark.parametrize("fn", [U.clip_grad_norm_, U.clip_grad_value_])
def test_foreach_is_accepted(fn) -> None:
    p = _ours(_G2, "float32")
    fn([p], 1.0, foreach=False)
    fn([p], 1.0, foreach=None)


def test_vector_conversions_take_a_single_tensor() -> None:
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = nn.Parameter(lucid.tensor(w))
    np.testing.assert_array_equal(_f32(U.parameters_to_vector(p)), w.reshape(-1))
    U.vector_to_parameters(lucid.tensor(w.reshape(-1) + 10.0), p)
    np.testing.assert_array_equal(_f32(p), w + 10.0)
    U.vector_to_parameters(lucid.tensor(w.reshape(-1)), [p])
    np.testing.assert_array_equal(_f32(p), w)


def test_non_tensor_entries_are_refused_by_name() -> None:
    for call in (
        lambda: U.clip_grad_norm_([1.0], 1.0),
        lambda: U.clip_grad_value_([None], 1.0),
        lambda: U.parameters_to_vector([object()]),
        lambda: U.vector_to_parameters(lucid.zeros(1), ["x"]),
        lambda: U.get_total_norm([1]),
    ):
        with pytest.raises(TypeError, match="expected a Tensor"):
            call()
    with pytest.raises(TypeError, match="vector_to_parameters"):
        U.vector_to_parameters([1.0], [])  # type: ignore[arg-type]
    assert U.get_total_norm([None, lucid.tensor([3.0, 4.0])]).item() == 5.0
