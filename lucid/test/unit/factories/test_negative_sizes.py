"""A tensor's shape is concrete: every way to ask for a negative size is refused.

``lucid.zeros(-1)`` built a CPU tensor of shape ``(-1,)``.  Allocation
counted the negative extent as zero elements while ``numel`` cast it to
``2**64 - 1``, so the first kernel to walk it (``t + 1``, ``t.sum()``)
read past an empty buffer and the process died with SIGSEGV.  The same
request through other doors raised five different classes, or nothing.

One element-count rule now refuses a negative extent (``shape_numel``),
and the tensor constructor derives its stride through the other
(``contiguous_stride``), so no tensor is built with one.  Every door
raises the engine's ``InvalidArgument`` — a ``LucidError``, so a
``RuntimeError`` as the reference framework raises, and a
``ValueError`` as the Python layer used to.  Each case runs in-process:
if the rule holds, nothing here can crash.
"""

import re
import subprocess
import sys
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine

DEVICES = ["cpu", "metal"]

# The two size requests every door is asked: a lone negative, and a negative
# beside a valid extent.
SIZES = [(-1,), (2, -3)]


def _first_negative(sizes: tuple[int, ...]) -> int:
    return next(s for s in sizes if s < 0)


def _pair(sizes: tuple[int, ...]) -> tuple[int, int]:
    """A two-argument size: a lone negative goes first, beside a valid 2."""
    return (sizes[0], 2) if len(sizes) == 1 else (sizes[0], sizes[1])


def _no_keep(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Sizes for expand: ``-1`` there means keep the size, so ask for ``-2``."""
    return (-2, 3) if sizes == (-1,) else sizes


def _reshape_target(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Sizes for view / reshape: ``-1`` infers, so a lone one becomes two."""
    return (-1, -1) if sizes == (-1,) else sizes


Builder = Callable[[ModuleType, str, tuple[int, ...]], object]


def _x(m: ModuleType, dev: str) -> object:
    return m.ones(2, 3, device=dev)


ENTRY_POINTS: dict[str, Builder] = {
    "zeros": lambda m, dev, s: m.zeros(*s, device=dev),
    "ones": lambda m, dev, s: m.ones(*s, device=dev),
    "empty": lambda m, dev, s: m.empty(*s, device=dev),
    "full": lambda m, dev, s: m.full(s, 1.0, device=dev),
    "rand": lambda m, dev, s: m.rand(*s, device=dev),
    "randn": lambda m, dev, s: m.randn(*s, device=dev),
    "randint": lambda m, dev, s: m.randint(0, 5, s, device=dev),
    "randperm": lambda m, dev, s: m.randperm(_first_negative(s), device=dev),
    "eye": lambda m, dev, s: m.eye(*s, device=dev),
    "linspace": lambda m, dev, s: m.linspace(0.0, 1.0, _first_negative(s), device=dev),
    "logspace": lambda m, dev, s: m.logspace(0.0, 1.0, _first_negative(s), device=dev),
    "new_zeros": lambda m, dev, s: _x(m, dev).new_zeros(s),
    "new_ones": lambda m, dev, s: _x(m, dev).new_ones(s),
    "new_empty": lambda m, dev, s: _x(m, dev).new_empty(s),
    "new_full": lambda m, dev, s: _x(m, dev).new_full(s, 1.0),
    "repeat": lambda m, dev, s: _x(m, dev).repeat(*_pair(s)),
    "tile": lambda m, dev, s: _x(m, dev).tile(_pair(s)),
    "repeat_interleave": lambda m, dev, s: _x(m, dev).repeat_interleave(_first_negative(s), dim=0),
    "expand": lambda m, dev, s: _x(m, dev).expand(*_no_keep(s)),
    "broadcast_to": lambda m, dev, s: m.broadcast_to(_x(m, dev), _no_keep(s)),
    "view": lambda m, dev, s: _x(m, dev).view(*_reshape_target(s)),
    "reshape": lambda m, dev, s: _x(m, dev).reshape(*_reshape_target(s)),
    "unflatten": lambda m, dev, s: _x(m, dev).unflatten(0, _reshape_target(s)),
    "as_strided": lambda m, dev, s: _x(m, dev).as_strided(s, (1,) * len(s)),
    "nn.Linear": lambda m, dev, s: m.nn.Linear(*_pair(s), device=dev),
    "nn.Conv1d": lambda m, dev, s: m.nn.Conv1d(*_pair(s), 3, device=dev),
    "nn.Conv2d": lambda m, dev, s: m.nn.Conv2d(*_pair(s), 3, device=dev),
    "nn.Embedding": lambda m, dev, s: m.nn.Embedding(*_pair(s), device=dev),
}


def _cases(check: str) -> list[object]:
    del check
    params = []
    for name in ENTRY_POINTS:
        for sizes in SIZES:
            for dev in DEVICES:
                params.append(pytest.param(name, sizes, dev, id=f"{name}-{sizes}-{dev}"))
    return params


def _refusal(name: str, sizes: tuple[int, ...], dev: str) -> Exception:
    if dev == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal device not available on this host")
    # Any class: the callers assert which one, so a wrong one fails there.
    with pytest.raises(Exception) as info:
        ENTRY_POINTS[name](lucid, dev, sizes)
    return info.value


@pytest.mark.parametrize(("name", "sizes", "dev"), _cases("class"))
def test_negative_size_raises_invalid_argument(name: str, sizes: tuple[int, ...], dev: str) -> None:
    err = _refusal(name, sizes, dev)
    assert type(err) is _C_engine.InvalidArgument, f"{type(err).__name__}: {err}"
    assert isinstance(err, RuntimeError)


# The op, then the size it was asked for: the reference's wording for an
# extent, or the count that would have made one.
_NAMES_THE_SIZE = re.compile(
    r"^[a-z_]+: (Trying to create tensor with negative dimension -\d+: \[|"
    r"Repeats must be non-negative, got -\d+|"
    r"only one -1 is allowed)"
)


@pytest.mark.parametrize(("name", "sizes", "dev"), _cases("message"))
def test_refusal_names_the_op_and_the_size(name: str, sizes: tuple[int, ...], dev: str) -> None:
    msg = str(_refusal(name, sizes, dev))
    assert _NAMES_THE_SIZE.match(msg), msg


@pytest.mark.parametrize("dev", DEVICES)
def test_tile_message_is_the_reference_wording(dev: str) -> None:
    if dev == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal device not available on this host")
    with pytest.raises(
        _C_engine.InvalidArgument,
        match=re.escape("tile: Trying to create tensor with negative dimension -9: [4, -9]"),
    ):
        lucid.ones(2, 3, device=dev).tile((2, -3))


def test_factory_message_is_the_reference_wording() -> None:
    with pytest.raises(
        _C_engine.InvalidArgument,
        match=re.escape("zeros: Trying to create tensor with negative dimension -3: [2, -3]"),
    ):
        lucid.zeros(2, -3)


@pytest.mark.parity
@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_reference_refuses_with_a_runtime_error(
    ref: ModuleType, name: str, sizes: tuple[int, ...]
) -> None:
    with pytest.raises(RuntimeError):
        ENTRY_POINTS[name](ref, "cpu", sizes)


# A wildcard is not a negative size: these still resolve.
@pytest.mark.parametrize("dev", DEVICES)
def test_wildcards_still_resolve(dev: str) -> None:
    if dev == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal device not available on this host")
    x = lucid.ones(2, 3, device=dev)
    assert x.reshape(-1).shape == (6,)
    assert x.view(3, -1).shape == (3, 2)
    assert x.unflatten(1, (-1, 1)).shape == (2, 3, 1)
    assert x.unflatten(1, (3, -1)).shape == (2, 3, 1)
    assert lucid.nn.Unflatten(-1, (-1, 1))(x).shape == (2, 3, 1)
    assert x.expand(-1, -1).shape == (2, 3)
    assert x.expand(4, -1, -1).shape == (4, 2, 3)
    assert lucid.zeros(0, 3, device=dev).tile((-1, 1)).shape == (0, 3)
    assert lucid.zeros(2, 0, device=dev).unflatten(1, (-1, 3)).shape == (2, 0, 3)


@pytest.mark.parametrize("dev", DEVICES)
def test_negative_count_over_an_empty_axis_tiles_it_empty(dev: str) -> None:
    # The reference rules on the output's extents: 0 * -1 is an empty axis.
    if dev == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal device not available on this host")
    x = lucid.zeros(0, 3, device=dev, requires_grad=True)
    y = x.repeat(-1, 2)
    assert y.shape == (0, 6)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == (0, 3)


@pytest.mark.parametrize("dev", DEVICES)
def test_negative_count_over_an_empty_axis_is_refused(dev: str) -> None:
    if dev == "metal" and not lucid.metal.is_available():
        pytest.skip("Metal device not available on this host")
    with pytest.raises(_C_engine.InvalidArgument, match="Repeats must be non-negative"):
        lucid.zeros(0, 3, device=dev).repeat_interleave(-1, dim=0)


def test_negative_size_cannot_crash_the_process() -> None:
    # On the old engine ``zeros(-1)`` returned and ``+ 1`` died by SIGSEGV.
    code = (
        "import lucid\n"
        "from lucid._C import engine as _C_engine\n"
        "try:\n"
        "    lucid.zeros(-1) + 1\n"
        "except _C_engine.InvalidArgument:\n"
        "    print('refused')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"rc={proc.returncode}: {proc.stderr[-500:]}"
    assert proc.stdout.strip() == "refused"
