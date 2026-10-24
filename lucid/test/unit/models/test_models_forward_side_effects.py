"""A forward leaves the model as it found it, at any input size (LCD-544).

CoAtNet's transformer stage met an input of a new resolution by *replacing*
each block's relative-position table with a resized copy.  That call
answered correctly and every later one did not: the table for the built
resolution was gone, so an 8×8 → 12×12 → 8×8 round trip moved the 8×8
output; an optimizer built beforehand held a parameter the model no longer
used; and ``state_dict`` changed shape under the checkpoint.  No suite
called one model at two sizes, so none of them saw it.

So every image family that ``test_models_train_step`` builds at unit-test
size runs at its own size, then at a larger one, then at its own size
again.  The larger size may be refused — a model with a fixed position
table is entitled to that — but refused or not, the parameters must be the
same objects holding the same values, ``state_dict`` must keep its keys and
shapes, and the third call must reproduce the first bit for bit.  Each call
is seeded alike, so a family that samples in its forward is held to the same
bar.

A refusal has a bar of its own (LCD-554): it is a ``ValueError`` raised at
the model's boundary that names the size it was given — not an engine
``ShapeMismatch`` from deep inside the forward, which names neither the size
nor the rule.  ViT and EfficientDet refused that way; MaxViT did too, but
only because its pad mask did not broadcast past batch size 1.  LeNet-5,
the VAE and TVM refused from inside the engine until LCD-625.

The networks over one fixed grid are held to the bar at a size one pixel
off as well (LCD-625).  A side one past the built one rounds down onto the
built grid through a stride-2 embedding, so DiT, MeanFlow, TVM and the VAE
answered it — silently, with an output of the built size rather than the
input's.
"""

from typing import Any

import pytest

import lucid
import lucid.models as models
from lucid._C import engine as _C_engine
from lucid.test.unit.models.test_models_eval_determinism import (
    _tensors_of,
    _values_of,
)
from lucid.test.unit.models.test_models_train_step import CASES

#: How much larger the second call is.  32 keeps every stride-32 pyramid
#: and every patch-16 transformer on a valid grid, so a refusal means the
#: model does not take another size, not that this one is malformed.
_GROWTH = 32


def _image_cases() -> list[Any]:
    """The cases whose first input is a batch of images."""
    picked = []
    for case in CASES:
        family, factory, overrides, make_inputs = _values_of(case)
        args, _ = make_inputs()
        first = args[0] if args else None
        if isinstance(first, lucid.Tensor) and first.ndim == 4:
            picked.append(
                pytest.param(
                    factory,
                    overrides,
                    tuple(first.shape),
                    id=family,
                    marks=getattr(case, "marks", ()),
                )
            )
    return picked


def _call(model: lucid.nn.Module, x: lucid.Tensor) -> list[tuple[str, lucid.Tensor]]:
    lucid.manual_seed(1)
    with lucid.no_grad():
        return [(n, t.detach().clone()) for n, t in _tensors_of(model(x))]


@pytest.mark.parametrize(("factory", "overrides", "shape"), _image_cases())
def test_another_resolution_leaves_the_model_unchanged(
    factory: str, overrides: dict[str, Any], shape: tuple[int, ...]
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).eval()
    owned = dict(model.named_parameters())
    values = {n: p.detach().clone() for n, p in owned.items()}
    layout = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    batch, channels, height, width = shape
    native = lucid.randn(batch, channels, height, width)
    larger = lucid.randn(batch, channels, height + _GROWTH, width + _GROWTH)

    first = _call(model, native)
    # Refusing an unsupported size is allowed; the checks below still hold —
    # a refusal must not leave the model half-changed either.
    refusal: ValueError | None = None
    try:
        _call(model, larger)
    except ValueError as raised:
        refusal = raised
    again = _call(model, native)

    after = dict(model.named_parameters())
    assert after.keys() == owned.keys(), f"{factory}: parameter names changed"
    for name, param in after.items():
        assert param is owned[name], f"{factory}: forward replaced {name}"
        drift = float((param.detach() - values[name]).abs().max().item())
        assert drift == 0.0, f"{factory}: forward rewrote {name} by {drift}"
    assert {
        k: tuple(v.shape) for k, v in model.state_dict().items()
    } == layout, f"{factory}: state_dict layout changed"
    assert [n for n, _ in again] == [n for n, _ in first]
    for (name, a), (_, b) in zip(first, again, strict=True):
        moved = float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0
        assert moved == 0.0, f"{factory}: {name} moved by {moved} after the round trip"

    if refusal is not None:
        _assert_refused_at_the_boundary(factory, refusal, str(height + _GROWTH))


def _assert_refused_at_the_boundary(
    factory: str, refusal: ValueError, size: str
) -> None:
    # ``ShapeMismatch`` is a ValueError too, so the class alone does not tell
    # a refusal at the boundary from a crash inside the engine.
    assert not isinstance(
        refusal, _C_engine.LucidError
    ), f"{factory}: refused from inside the engine: {refusal}"
    assert size in str(
        refusal
    ), f"{factory}: the refusal does not name the input size: {refusal}"


_GRID = {"sample_size": 8, "hidden_size": 32, "depth": 1, "num_heads": 4}


def _two(value: float) -> lucid.Tensor:
    return lucid.full((2,), value)


#: The networks built over one fixed grid: factory, config overrides,
#: ``(channels, built side)``, and how to call it on a batch of two.
_FIXED_GRID = [
    pytest.param("dit_small_2", _GRID, (4, 8), lambda m, x: m(x, _two(10.0)), id="dit"),
    pytest.param(
        "mean_flow_base_2",
        _GRID,
        (4, 8),
        lambda m, x: m(x, _two(0.2), _two(0.7)),
        id="mean_flow",
    ),
    pytest.param(
        "tvm_imagenet256",
        _GRID,
        (4, 8),
        lambda m, x: m(x, _two(0.7), _two(0.2)),
        id="tvm",
    ),
    pytest.param(
        "vae",
        {"latent_dim": 8, "down_block_channels": (8, 16, 32)},
        (3, 32),
        lambda m, x: m(x),
        id="vae",
    ),
]


@pytest.mark.parametrize(("factory", "overrides", "built", "call"), _FIXED_GRID)
def test_a_side_one_pixel_off_the_built_grid_is_refused_by_name(
    factory: str,
    overrides: dict[str, Any],
    built: tuple[int, int],
    call: Any,
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).eval()
    channels, side = built
    with lucid.no_grad():
        answer = call(model, lucid.randn(2, channels, side, side))
    assert tuple(answer.sample.shape)[-2:] == (side, side)
    for off in (side + 1, side - 1):
        with pytest.raises(ValueError) as refused, lucid.no_grad():
            call(model, lucid.randn(2, channels, off, off))
        _assert_refused_at_the_boundary(factory, refused.value, f"{off}x{off}")


@pytest.mark.parametrize(
    "factory", ["dit_small_2_gen", "mean_flow_base_2_gen", "tvm_imagenet256_gen"]
)
def test_sampling_from_noise_of_another_size_is_refused_by_name(factory: str) -> None:
    """``generate`` builds its noise at the built size; noise handed in is
    held to the same check, at the network's boundary."""
    lucid.manual_seed(0)
    model = models.create_model(factory, **_GRID, num_classes=4).eval()
    with pytest.raises(ValueError) as refused:
        model.generate(1, steps=1, noise=lucid.randn(1, 4, 9, 9))
    _assert_refused_at_the_boundary(factory, refused.value, "9x9")
