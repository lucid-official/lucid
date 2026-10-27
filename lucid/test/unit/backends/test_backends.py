"""``lucid.backends`` — surface + dispatch behaviour smoke."""

import pytest

import lucid


class TestBackendsSurface:
    def test_module_present(self) -> None:
        assert hasattr(lucid, "backends")

    def test_the_backends_are_named_for_apple_silicon(self) -> None:
        """One namespace per stream, named for what runs it.

        The GPU stream is ``metal`` everywhere in Lucid, so there is no
        ``mps`` alias here to keep in step with it.
        """
        assert lucid.backends.__all__ == ["accelerate", "metal", "quantized"]
        assert not hasattr(lucid.backends, "mps")
        # Nothing else public leaks through: the ``typing.final`` the module
        # decorates with used to show up in ``dir()``.
        public = [n for n in dir(lucid.backends) if not n.startswith("_")]
        assert public == ["accelerate", "metal", "quantized"]


class TestMetalBackend:
    def test_the_metal_backend_carries_its_flags(self) -> None:
        metal = lucid.backends.metal
        prev = metal.deterministic
        try:
            metal.deterministic = not prev
            assert metal.deterministic is (not prev)
        finally:
            metal.deterministic = prev
        assert isinstance(metal.benchmark, bool)

    def test_availability_lives_on_the_device_module(self) -> None:
        """``lucid.metal.is_available()`` is the accessor, and on the only
        hardware Lucid installs on it is always true."""
        assert lucid.metal.is_available() is True


class TestQuantizedEngine:
    """``"auto"`` follows availability alone, not where the tensors live.

    The docstring used to say the MLX kernel ran only for tensors on Metal,
    while ``use_mlx`` ignored the device.  The code was the intended half:
    ``QuantizedLinearMLX`` moves a CPU activation to Metal and back, so a
    CPU model is meant to get the fast path too.
    """

    def test_auto_follows_availability(self) -> None:
        quantized = lucid.backends.quantized
        prev = quantized.engine
        try:
            quantized.engine = "auto"
            assert quantized.use_mlx() == quantized.available
        finally:
            quantized.engine = prev

    @pytest.mark.skipif(
        not lucid.backends.quantized.available, reason="engine lacks the MLX quantized ops"
    )
    def test_auto_routes_a_cpu_model_to_the_mlx_layer(self) -> None:
        import lucid.nn as nn
        import lucid.nn.quantized as nnq
        import lucid.quantization as Q

        quantized = lucid.backends.quantized
        prev = quantized.engine
        try:
            quantized.engine = "auto"
            model = nn.Sequential(nn.Linear(64, 32))
            model.eval()
            qmodel = Q.quantize_dynamic(model)
            assert isinstance(qmodel[0], nnq.QuantizedLinearMLX)
            # Device-transparent: a CPU input still answers on the CPU.
            assert not qmodel(lucid.randn(2, 64)).is_metal
        finally:
            quantized.engine = prev
