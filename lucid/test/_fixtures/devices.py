"""Device fixtures — drives CPU/Metal cross-validation.

The default ``device`` fixture parametrizes every requesting test over
the available compute streams.  Metal is detected by attempting a
zero-cost allocation and left out at parametrize time when unavailable.

Metal cannot hold every dtype: MLX has no double precision, so the
engine refuses ``float64`` and ``complex128`` on the GPU stream.  Those
pairs are never generated.  :func:`pytest_generate_tests` sweeps
``device`` together with a dtype fixture and leaves them out, and
:func:`device_dtype_params` / :func:`devices_supporting` do the same for
a test that parametrizes by hand.  What Metal does with the refused
dtypes is asserted once, in
``lucid/test/unit/device/test_metal_dtype_support.py``.
"""

import functools
import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import pytest

import lucid
from lucid.test._fixtures.dtypes import _FLOAT_DTYPES, _FLOAT_DTYPES_EXTENDED, _INT_DTYPES

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet


@functools.lru_cache(maxsize=1)
def metal_available() -> bool:
    """Return True if Apple Metal is usable — try a tiny allocation
    and let any failure bubble up as ``False``."""
    try:
        from lucid._C import engine as _C_engine

        _ = _C_engine.zeros([1], _C_engine.F32, _C_engine.GPU)
    except Exception:
        return False
    return True


def _device_params() -> list[str]:
    return ["cpu", "metal"] if metal_available() else ["cpu"]


@pytest.fixture(params=_device_params())
def device(request: pytest.FixtureRequest) -> str:
    """Yield each available compute device.  Tests using this fixture
    automatically run on CPU and (when present) Metal."""
    return str(request.param)


@pytest.fixture
def device_cpu_only() -> str:
    return "cpu"


@pytest.fixture
def device_gpu_only() -> str:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    return "metal"


@pytest.fixture
def cross_device_pair() -> Iterator[tuple[str, str]]:
    """Yield ``("cpu", "metal")`` once when both are present; otherwise
    skip.  Intended for "device drift" tests that compare CPU and GPU
    outputs of the same op."""
    if not metal_available():
        pytest.skip("CPU↔Metal cross-device tests need Metal")
    yield ("cpu", "metal")


# Dtypes the GPU stream cannot hold at all.  MLX has no double-precision
# type, so the engine's ``to_mlx_dtype`` refuses these at every way onto
# Metal: a factory, ``.to("metal")``, a cast.  bfloat16 and complex64 map
# to native MLX dtypes and are held.  ``test_metal_dtype_support.py``
# checks this set against the engine in both directions, so a dtype that
# MLX learns (or loses) fails there instead of going quietly untested.
METAL_UNSUPPORTED_DTYPES: frozenset[lucid.dtype] = frozenset({lucid.float64, lucid.complex128})


def device_supports(device: str, dtype: lucid.dtype) -> bool:
    """Whether ``device`` can hold a tensor of ``dtype``."""
    return not (device == "metal" and dtype in METAL_UNSUPPORTED_DTYPES)


def devices_supporting(dtype: lucid.dtype) -> list[str]:
    """The available devices that can hold ``dtype``.

    For a test about one specific dtype that is parametrized over
    ``device``: ``@pytest.mark.parametrize("device",
    devices_supporting(lucid.float64))``.
    """
    return [d for d in _device_params() if device_supports(d, dtype)]


def device_dtype_params(
    *dtype_sweeps: Iterable[lucid.dtype], devices: Sequence[str] | None = None
) -> list[ParameterSet]:
    """Every ``(device, dtype, ...)`` combination the device can hold.

    One dtype per sweep, crossed with each device in ``devices`` (the
    available ones by default), keeping only the combinations whose
    every dtype the device supports.  Each is a ``pytest.param`` with the
    id ``"<device>-<dtype>..."`` — the id the fixtures would have given.
    """
    sweeps = [tuple(s) for s in dtype_sweeps]
    return [
        pytest.param(dev, *dts, id="-".join([dev, *map(str, dts)]))
        for dev in (_device_params() if devices is None else devices)
        for dts in itertools.product(*sweeps)
        if all(device_supports(dev, dt) for dt in dts)
    ]


# The dtype fixtures of ``dtypes.py``, by name, with the dtypes each sweeps.
_DTYPE_FIXTURES: dict[str, Sequence[lucid.dtype]] = {
    "float_dtype": _FLOAT_DTYPES,
    "float_dtype_extended": _FLOAT_DTYPES_EXTENDED,
    "int_dtype": _INT_DTYPES,
}


def _argnames(mark: pytest.Mark) -> list[str]:
    names = mark.args[0] if mark.args else mark.kwargs["argnames"]
    if isinstance(names, str):
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Sweep ``device`` and the dtype fixtures as one axis, minus the
    combinations the device cannot hold.

    Parametrized separately, the fixtures make every cross pair, Metal ×
    float64 included, and the cells that could not run were skipped from
    inside the test.  A ``parametrize`` mark naming all of them replaces
    their own params instead: the fixture plugin defers to such a mark
    and the mark plugin applies it, so the refused pairs are never
    collected.  The mark goes first, so node ids keep the
    ``[device-dtype-...]`` order the fixtures gave them.  A test that
    parametrizes ``device`` or a dtype fixture itself is left alone.

    Re-exported by ``lucid/test/conftest.py``, which is what registers it.
    """
    requested = metafunc.fixturenames
    dtype_names = [n for n in _DTYPE_FIXTURES if n in requested]
    if "device" not in requested or not dtype_names:
        return
    argnames = ["device", *dtype_names]
    explicit = {
        name for mark in metafunc.definition.iter_markers("parametrize") for name in _argnames(mark)
    }
    if explicit.intersection(argnames):
        return
    params = device_dtype_params(*(_DTYPE_FIXTURES[n] for n in dtype_names))
    metafunc.definition.add_marker(
        pytest.mark.parametrize(argnames, params, indirect=True), append=False
    )
