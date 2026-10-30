"""``autocast`` refuses a dtype the device cannot autocast to (LCD-738).

``autocast(dtype=...)`` took any dtype and the engine computed in it:
``with autocast("cpu", dtype=int32): x @ x`` truncated its operands to
integers and returned ``[[7, 10], [15, 22]]`` for a product whose float32
value is ``[[11, 15], [21, 29]]``; bool and float64 were accepted too, and
float64 on Metal raised from ``astype``.

The scope now checks its dtype once, at construction, against the device's
supported set (float16 and bfloat16).  Anything else warns and builds a scope
with ``enabled=False`` — the reference framework's behaviour — so the ops
inside keep their float32 inputs.  The decorator form and nested scopes are
built through the same constructor and inherit the check.
"""

import warnings
from types import ModuleType

import pytest

import lucid
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

UNSUPPORTED = [lucid.float32, lucid.float64, lucid.int32, lucid.int64, lucid.bool, lucid.complex64]

#: (device, dtype) -> the dtype ``x @ x`` has inside the scope.  CPU float16
#: computes in float32 by design: Accelerate has no half-precision BLAS.
SUPPORTED = [
    ("cpu", lucid.bfloat16, lucid.bfloat16),
    ("cpu", lucid.float16, lucid.float32),
    ("metal", lucid.float16, lucid.float16),
    ("metal", lucid.bfloat16, lucid.bfloat16),
]

EXACT = [[11.0, 15.0], [21.0, 29.0]]  # x @ x in float32 for the operand below


@pytest.fixture(autouse=True)
def _amp_off_around_each_test() -> object:
    assert not autocast.is_autocast_enabled(), "a previous test left autocast on"
    yield
    assert not autocast.is_autocast_enabled(), "this test left autocast on"


@pytest.fixture
def x(device: str) -> lucid.Tensor:
    return lucid.tensor([[1.5, 2.5], [3.5, 4.5]], device=device)


@pytest.mark.parametrize("target", UNSUPPORTED, ids=str)
def test_an_unsupported_dtype_warns_and_leaves_the_matmul_in_float32(
    device: str, target: lucid.dtype, x: lucid.Tensor
) -> None:
    with pytest.warns(UserWarning, match="target dtype .* is not supported. Disabling autocast"):
        scope = autocast(device, dtype=target)
    with scope:
        assert not autocast.is_autocast_enabled(device)
        assert autocast.get_autocast_dtype(device) is None
        y = x @ x
    assert y.dtype == lucid.float32
    assert y.tolist() == EXACT


@pytest.mark.parametrize("device, target, computed", SUPPORTED, ids=lambda v: str(v))
def test_a_supported_dtype_is_accepted_silently(
    device: str, target: lucid.dtype, computed: lucid.dtype
) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal is not available")
    x = lucid.tensor([[1.5, 2.5], [3.5, 4.5]], device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scope = autocast(device, dtype=target)
    with scope:
        assert autocast.is_autocast_enabled(device)
        assert autocast.get_autocast_dtype(device) == target
        assert (x @ x).dtype == computed


def test_the_default_dtype_is_supported_on_each_device(device: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with autocast(device):
            assert autocast.is_autocast_enabled(device)


@pytest.mark.parametrize("target", UNSUPPORTED, ids=str)
def test_a_scope_that_is_off_anyway_does_not_warn(device: str, target: lucid.dtype) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with autocast(device, dtype=target, enabled=False):
            assert not autocast.is_autocast_enabled(device)


def test_the_warning_points_at_the_caller(device: str) -> None:
    with pytest.warns(UserWarning) as caught:
        autocast(device, dtype=lucid.int32)
    assert caught[0].filename == __file__


def test_the_decorator_form_inherits_the_check(device: str, x: lucid.Tensor) -> None:
    with pytest.warns(UserWarning, match="not supported"):

        @autocast(device, dtype=lucid.int32)
        def square(a: lucid.Tensor) -> lucid.Tensor:
            return a @ a

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the warning is at decoration, not per call
        y = square(x)
    assert y.dtype == lucid.float32
    assert y.tolist() == EXACT


def test_a_refused_inner_scope_turns_the_outer_one_off_and_restores_it(device: str) -> None:
    with autocast(device, dtype=lucid.bfloat16):
        with pytest.warns(UserWarning, match="not supported"):
            inner = autocast(device, dtype=lucid.int32)
        with inner:
            # ``enabled=False`` is the force-off scope: it overrides the outer one.
            assert not autocast.is_autocast_enabled(device)
        assert autocast.is_autocast_enabled(device)
        assert autocast.get_autocast_dtype(device) == lucid.bfloat16


def test_an_unknown_device_is_still_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown device_type"):
        autocast("tpu", dtype=lucid.int32)


@pytest.mark.parametrize("name", ["bfloat16", "float16", "float32", "float64", "int32", "bool"])
def test_the_cpu_table_matches_the_reference(ref: ModuleType, name: str) -> None:
    ours: list[bool] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with autocast("cpu", dtype=getattr(lucid, name)):
            ours.append(autocast.is_autocast_enabled("cpu"))
        with ref.autocast("cpu", dtype=getattr(ref, name)):
            theirs = ref.is_autocast_enabled("cpu")
    assert ours == [theirs]
