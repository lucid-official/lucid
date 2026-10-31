"""Every lazy layer exposes the reference's public ``initialize_parameters`` (LCD-762).

The reference's lazy mixin puts ``initialize_parameters(input)`` on every lazy
layer; Lucid had it on ``LazyLinear`` only, with the convolutions and norms
sizing themselves through private names.  The template now lives on the shared
lazy base and each family supplies the one axis it reads.
"""

import inspect

import pytest

import lucid
import lucid.nn as nn

# (class, kwargs, input shape, expected size, axis of the input that carries it)
CASES = [
    ("LazyLinear", dict(out_features=3), (4, 6), 6, -1),
    ("LazyLinear", dict(out_features=3), (2, 5, 6), 6, -1),
    ("LazyConv1d", dict(out_channels=8, kernel_size=3), (2, 3, 16), 3, 1),
    ("LazyConv2d", dict(out_channels=8, kernel_size=3), (2, 3, 8, 8), 3, 1),
    ("LazyConv3d", dict(out_channels=8, kernel_size=3), (2, 3, 4, 4, 4), 3, 1),
    ("LazyConv2d", dict(out_channels=8, kernel_size=3), (3, 8, 8), 3, 0),
    ("LazyConv2d", dict(out_channels=8, kernel_size=3, bias=False), (2, 3, 8, 8), 3, 1),
    ("LazyConvTranspose1d", dict(out_channels=8, kernel_size=3), (2, 3, 16), 3, 1),
    ("LazyConvTranspose2d", dict(out_channels=8, kernel_size=3), (2, 3, 8, 8), 3, 1),
    ("LazyConvTranspose3d", dict(out_channels=8, kernel_size=3), (2, 3, 4, 4, 4), 3, 1),
    ("LazyBatchNorm1d", {}, (4, 6, 8), 6, 1),
    ("LazyBatchNorm2d", {}, (4, 6, 5, 5), 6, 1),
    ("LazyBatchNorm3d", {}, (4, 6, 3, 3, 3), 6, 1),
    ("LazyBatchNorm1d", dict(affine=False), (4, 6, 8), 6, 1),
    ("LazyInstanceNorm1d", {}, (4, 6, 8), 6, 1),
    ("LazyInstanceNorm2d", {}, (4, 6, 5, 5), 6, 1),
    ("LazyInstanceNorm3d", {}, (4, 6, 3, 3, 3), 6, 1),
    ("LazyInstanceNorm1d", {}, (6, 8), 6, 0),  # unbatched: channels lead
    ("LazyInstanceNorm2d", {}, (6, 5, 5), 6, 0),
]
IDS = [f"{c[0]}-{'x'.join(map(str, c[2]))}" for c in CASES]
SIZE_ATTRS = ("in_features", "in_channels", "num_features")


def _state_shapes(m: nn.Module) -> dict[str, tuple[int, ...]]:
    out = {f"p:{k}": tuple(v.shape) for k, v in m.named_parameters()}
    out.update({f"b:{k}": tuple(v.shape) for k, v in m.named_buffers() if v is not None})
    return out


def _size(m: nn.Module) -> int | None:
    (attr,) = [a for a in SIZE_ATTRS if hasattr(m, a)]
    return getattr(m, attr)


@pytest.mark.filterwarnings("ignore::lucid.nn.module.UninitializedParameterWarning")
@pytest.mark.parametrize(("name", "kw", "shape", "size", "axis"), CASES, ids=IDS)
def test_initialize_parameters_materialises_what_the_first_forward_would(
    name: str, kw: dict[str, object], shape: tuple[int, ...], size: int, axis: int
) -> None:
    x = lucid.ones(*shape)
    sized = getattr(nn, name)(**kw)
    forwarded = getattr(nn, name)(**kw)
    assert _size(sized) is None

    held = dict(sized.named_parameters())
    assert sized.initialize_parameters(x) is None
    forwarded(x)

    assert _size(sized) == size == _size(forwarded)
    assert _state_shapes(sized) == _state_shapes(forwarded)
    assert not sized.has_uninitialized_parameters()
    # The objects an optimizer built early holds are the ones that got filled.
    for k, p in sized.named_parameters():
        assert p is held[k], k


@pytest.mark.parametrize(("name", "kw", "shape", "size", "axis"), CASES, ids=IDS)
def test_a_sized_layer_ignores_a_second_call(
    name: str, kw: dict[str, object], shape: tuple[int, ...], size: int, axis: int
) -> None:
    m = getattr(nn, name)(**kw)
    m.initialize_parameters(lucid.ones(*shape))
    before = _state_shapes(m)
    other = list(shape)
    other[axis] = size + 3
    m.initialize_parameters(lucid.ones(*other))
    assert _size(m) == size
    assert _state_shapes(m) == before


@pytest.mark.parametrize("name", sorted({c[0] for c in CASES}))
def test_every_lazy_class_has_the_public_method_and_no_private_alias(name: str) -> None:
    cls = getattr(nn, name)
    assert callable(cls.initialize_parameters)
    assert not hasattr(cls, "_materialize_from")
    assert list(inspect.signature(cls.initialize_parameters).parameters)[:2] == ["self", "input"]


def test_lazy_conv_accepts_and_ignores_extra_arguments_as_the_reference_does() -> None:
    m = nn.LazyConv2d(4, 3)
    m.initialize_parameters(lucid.ones(1, 3, 8, 8), 1, key="ignored")
    assert m.in_channels == 3


@pytest.mark.parametrize("name", ["LazyLinear", "LazyBatchNorm1d", "LazyInstanceNorm1d"])
def test_linear_and_norms_take_the_input_alone(name: str) -> None:
    kw = {"out_features": 2} if name == "LazyLinear" else {}
    m = getattr(nn, name)(**kw)
    with pytest.raises(TypeError):
        m.initialize_parameters(lucid.ones(2, 3), 1)
