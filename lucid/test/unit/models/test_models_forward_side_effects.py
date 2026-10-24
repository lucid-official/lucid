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
only because its pad mask did not broadcast past batch size 1.
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

#: Families that still refuse the larger size from inside the engine, each
#: waiting on the card that may edit its model file.  The refusal's form is
#: judged after the round-trip checks, and it is the check expected to fail.
_ENGINE_REFUSALS: dict[str, str] = {
    "lenet": (
        "LCD-554 FOUND: LeNet-5's first Linear is sized for 32 px and refuses "
        "64 px as a linear ShapeMismatch"
    ),
    "tvm": (
        "LCD-554: tvm/_model.py is held by LCD-597; 96 px fails as a broadcast "
        "ShapeMismatch against the position table"
    ),
    "vae": (
        "LCD-554 FOUND: generative/vae/_model.py (held by LCD-597) refuses "
        "64 px as a reshape ShapeMismatch"
    ),
}


def _image_cases() -> list[Any]:
    """The cases whose first input is a batch of images."""
    picked = []
    for case in CASES:
        family, factory, overrides, make_inputs = _values_of(case)
        args, _ = make_inputs()
        first = args[0] if args else None
        if isinstance(first, lucid.Tensor) and first.ndim == 4:
            marks = list(getattr(case, "marks", ()))
            if family in _ENGINE_REFUSALS:
                reason = _ENGINE_REFUSALS[family]
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            picked.append(
                pytest.param(
                    factory,
                    overrides,
                    tuple(first.shape),
                    id=family,
                    marks=marks,
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
        # ``ShapeMismatch`` is a ValueError too, so the class alone does not
        # tell a refusal at the boundary from a crash inside the engine.
        assert not isinstance(
            refusal, _C_engine.LucidError
        ), f"{factory}: refused from inside the engine: {refusal}"
        assert str(height + _GROWTH) in str(
            refusal
        ), f"{factory}: the refusal does not name the input size: {refusal}"
