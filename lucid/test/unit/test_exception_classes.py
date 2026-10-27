"""A refusal is catchable by its kind, whichever layer raised it.

The same kind of refusal used to arrive as different classes: a shape
mismatch was ``ShapeMismatch`` (a ``RuntimeError``) from the engine and
``ValueError`` from Python; a bad argument value was a bare ``LucidError``
from the engine (``groups=0``, ``dropout(p=1.5)``) and ``ValueError`` from
Python; an axis out of range was sometimes ``IndexError`` and sometimes a
bare ``LucidError``.  No single ``except`` caught a kind.

Now every engine class is a ``LucidError`` — so a ``RuntimeError``, as the
reference framework raises them — and also the builtin of its kind:

=====================  ===============================
ShapeMismatch          ValueError
DeviceMismatch         ValueError
InvalidArgument        ValueError
DtypeMismatch          TypeError
IndexError             IndexError (and LookupError)
NotImplementedError    NotImplementedError
=====================  ===============================
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.test._fixtures.devices import metal_available

_KINDS = {
    "ShapeMismatch": ValueError,
    "DeviceMismatch": ValueError,
    "InvalidArgument": ValueError,
    "DtypeMismatch": TypeError,
    "IndexError": IndexError,
    "NotImplementedError": NotImplementedError,
}


@pytest.mark.parametrize("name", list(_KINDS))
def test_every_engine_class_is_a_lucid_error_and_its_builtin(name: str) -> None:
    cls = getattr(_C_engine, name)
    assert issubclass(cls, _C_engine.LucidError)
    assert issubclass(cls, RuntimeError)
    assert issubclass(cls, _KINDS[name])
    assert cls.__module__ == _C_engine.__name__


_REFUSALS: dict[str, tuple[Callable[[], object], str, type[Exception]]] = {
    "broadcast": (lambda: lucid.ones(2, 3) + lucid.ones(4, 5), "ShapeMismatch", ValueError),
    "dtype pair": (
        lambda: _C_engine.where(
            (lucid.ones(2) > 0)._impl, lucid.ones(2).int()._impl, lucid.ones(2)._impl
        ),
        "DtypeMismatch",
        TypeError,
    ),
    # The engine kernel itself: ``F.dropout`` now refuses an out-of-range
    # ``p`` in Python before the engine sees it.
    "dropout p": (
        lambda: _C_engine.nn.dropout(_unwrap(lucid.ones(3)), 1.5, True),
        "InvalidArgument",
        ValueError,
    ),
    "histc bins": (lambda: lucid.histc(lucid.ones(3), bins=0), "InvalidArgument", ValueError),
    "conv groups": (
        lambda: F.conv2d(lucid.ones(1, 2, 4, 4), lucid.ones(2, 2, 3, 3), groups=0),
        "InvalidArgument",
        ValueError,
    ),
    "unfold step": (lambda: lucid.ones(5).unfold(0, 2, 0), "InvalidArgument", ValueError),
    "cumsum axis": (lambda: lucid.cumsum(lucid.ones(2, 3), 5), "IndexError", IndexError),
    "normalize axis": (lambda: F.normalize(lucid.ones(2, 3), dim=5), "IndexError", IndexError),
    "tensordot axis": (
        lambda: lucid.tensordot(lucid.ones(2, 3), lucid.ones(3, 2), dims=([5], [0])),
        "IndexError",
        IndexError,
    ),
    "fftn empty": (lambda: lucid.fft.fftn(lucid.zeros(0)), "InvalidArgument", ValueError),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_a_refusal_is_caught_three_ways(case: str) -> None:
    fn, name, builtin = _REFUSALS[case]
    for catch in (getattr(_C_engine, name), builtin, RuntimeError):
        with pytest.raises(catch):
            fn()


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_device_mismatch_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        lucid.ones(2).to("metal") + lucid.ones(2)
    with pytest.raises(_C_engine.DeviceMismatch):
        lucid.ones(2).to("metal") + lucid.ones(2)


_MLX_INVALID_ARGUMENT_REPROS: dict[str, Callable[[], object]] = {
    "broadcast_shapes": lambda: lucid.gather(
        lucid.zeros(2, 3, device="metal"), 0, lucid.zeros(2, 4, dtype=lucid.int64, device="metal")
    ),
    "reshape": lambda: _C_engine.nn.rotary_pos_embedding(
        _unwrap(lucid.zeros(2, 4, 8, device="metal")),
        _unwrap(lucid.zeros(3, dtype=lucid.int64, device="metal")),
        True,
    ),
    "scatter": lambda: lucid.scatter_reduce(
        lucid.zeros(3, dtype=lucid.int64, device="metal"),
        0,
        lucid.zeros(2, dtype=lucid.int64, device="metal"),
        lucid.ones(2, dtype=lucid.int64, device="metal"),
        reduce="amax",
    ).tolist(),
    "fftn": lambda: lucid.fft.fftn(lucid.zeros(0, device="metal")),
}


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("prefix", list(_MLX_INVALID_ARGUMENT_REPROS))
def test_mlx_invalid_argument_is_translated_to_lucid_invalid_argument(prefix: str) -> None:
    fn = _MLX_INVALID_ARGUMENT_REPROS[prefix]
    with pytest.raises(_C_engine.InvalidArgument, match=rf"(?:^|: )\[{prefix}\]") as exc_info:
        fn()
    assert isinstance(exc_info.value, _C_engine.LucidError)
    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, RuntimeError)


def test_pybind_builtin_exception_stays_unwrapped() -> None:
    with pytest.raises(TypeError) as exc_info:
        getattr(_C_engine, "add")(1, 2)
    assert not isinstance(exc_info.value, _C_engine.LucidError)
