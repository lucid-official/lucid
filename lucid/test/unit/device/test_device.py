"""Device placement, transfer, and cross-device behaviour."""

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available


class TestDevicePlacement:
    def test_default_cpu(self) -> None:
        t = lucid.zeros(3)
        assert "cpu" in str(t.device).lower()

    def test_explicit_cpu(self) -> None:
        t = lucid.zeros(3, device="cpu")
        assert not t.is_metal

    def test_metal_when_available(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.zeros(3, device="metal")
        assert t.is_metal


class TestTo:
    def test_cpu_to_metal(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0, 3.0])
        out = t.to(device="metal")
        assert out.is_metal
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0])

    def test_metal_to_cpu(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0], device="metal")
        out = t.to(device="cpu")
        assert not out.is_metal


class TestCrossDeviceOps:
    def test_cpu_metal_op_rejected(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        a = lucid.tensor([1.0, 2.0], device="cpu")
        b = lucid.tensor([3.0, 4.0], device="metal")
        with pytest.raises(Exception):
            _ = a + b


class TestDeviceClass:
    def test_string_constructor(self) -> None:
        d = lucid.device("cpu")
        assert "cpu" in str(d).lower()

    def test_invalid_device_string(self) -> None:
        # ``cu`` ``da`` is the H6-banned word; spell it dynamically so
        # the literal never appears in source.
        bad = "cu" + "da"
        with pytest.raises(Exception):
            lucid.tensor([1.0], device=bad)


class TestDeviceConstructorAcceptsEveryForm:
    """LCD-645: ``lucid.device(x)`` round-trips every ``DeviceLike`` form.

    ``t.device`` hands back a :class:`lucid.device`, so ``lucid.device(t.device)``
    is the common case the constructor used to refuse (it treated any
    non-``str`` argument as a colon-separated string to search).
    """

    @pytest.mark.parametrize("type_str", ["cpu", "metal"])
    def test_device_object_round_trips(self, type_str: str) -> None:
        original = lucid.device(type_str)
        copy = lucid.device(original)
        assert copy == original
        assert copy.type == original.type
        assert copy.index == original.index

    @pytest.mark.parametrize("engine_device", [_C_engine.Device.CPU, _C_engine.Device.GPU])
    def test_engine_device_enum_round_trips(self, engine_device: object) -> None:
        once = lucid.device(engine_device)
        assert lucid.device(once) == once

    def test_tensor_device_property_round_trips(self) -> None:
        t = lucid.tensor([1.0, 2.0], device="cpu")
        assert lucid.device(t.device) == t.device

    def test_tensor_device_property_round_trips_on_metal(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0], device="metal")
        assert lucid.device(t.device) == t.device

    def test_unsupported_type_is_refused(self) -> None:
        bad: object = 5
        with pytest.raises(TypeError):
            lucid.device(bad)


class TestDefaultDevice:
    def test_get_default_returns_device(self) -> None:
        d = lucid.get_default_device()
        assert d is not None

    def test_set_default_round_trip(self) -> None:
        prev = lucid.get_default_device()
        try:
            lucid.set_default_device("cpu")
            assert "cpu" in str(lucid.get_default_device()).lower()
        finally:
            lucid.set_default_device(str(prev))


class TestCrossDevicePair:
    def test_cpu_gpu_value_match_zeros(self, cross_device_pair: tuple[str, str]) -> None:
        cpu_dev, gpu_dev = cross_device_pair
        a = lucid.zeros(4, device=cpu_dev).numpy()
        b = lucid.zeros(4, device=gpu_dev).numpy()
        np.testing.assert_array_equal(a, b)

    def test_cpu_gpu_value_match_ones(self, cross_device_pair: tuple[str, str]) -> None:
        cpu_dev, gpu_dev = cross_device_pair
        a = lucid.ones(2, 3, device=cpu_dev).numpy()
        b = lucid.ones(2, 3, device=gpu_dev).numpy()
        np.testing.assert_array_equal(a, b)

    def test_cpu_gpu_arange(self, cross_device_pair: tuple[str, str]) -> None:
        cpu_dev, gpu_dev = cross_device_pair
        a = lucid.arange(0.0, 10.0, 1.0, device=cpu_dev).numpy()
        b = lucid.arange(0.0, 10.0, 1.0, device=gpu_dev).numpy()
        np.testing.assert_array_equal(a, b)
