"""No zoo model may depend on ``inplace=True`` being ignored (LCD-235).

Every activation and dropout used to accept ``inplace=True`` and return a
fresh tensor.  Once the flag was honoured, Xception's middle flow — whose
block is ``rep(x) + x`` with ``rep`` opening on ``nn.ReLU(inplace=True)`` —
added ``relu(x)`` instead of ``x`` to its skip: the output moved by 2.3 and
the gradients by 1.6e3, with nothing raised.  The block had only ever been
right because the flag did nothing (timm builds that ReLU with
``inplace=False`` for this reason).

So every family that ``test_models_train_step`` builds at unit-test size
and that holds a module with ``inplace=True`` takes one training forward and
backward twice — once as built, once with every ``inplace`` flag turned off
by a module walk — and the two must agree exactly, in every output tensor
and every gradient.  The caller's inputs must come back untouched too.  A
model that needs a write to stay out of place says ``inplace=False``.

A functional call written ``F.relu(t, inplace=True)`` inside a ``forward``
is not reachable by the walk; the zoo has one (MobileNetV3's squeeze-excite,
on a tensor it has just made).
"""

from typing import Any

import pytest

import lucid
import lucid.models as models
from lucid.test.unit.models.test_models_eval_determinism import _tensors_of
from lucid.test.unit.models.test_models_train_step import CASES, _family_of


def _inplace_modules(model: lucid.nn.Module) -> list[lucid.nn.Module]:
    return [m for m in model.modules() if getattr(m, "inplace", None) is True]


def _clone(value: Any) -> Any:
    return value.clone() if isinstance(value, lucid.Tensor) else value


def _step(
    model: lucid.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[tuple[str, lucid.Tensor]], dict[str, lucid.Tensor]]:
    model.zero_grad()
    lucid.manual_seed(1)
    out = model(*args, **kwargs)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return [(n, t.detach().clone()) for n, t in _tensors_of(out)], grads


def _worst(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"),
    CASES,
    ids=[_family_of(c) for c in CASES],
)
def test_training_step_is_the_same_with_inplace_off(
    family: str, factory: str, overrides: dict[str, Any], make_inputs: Any
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides)
    flagged = _inplace_modules(model)
    if not flagged:
        pytest.skip(f"{family}: no module built with inplace=True")
    model.train()
    args, kwargs = make_inputs()
    given = [(_clone(a), a) for a in [*args, *kwargs.values()]]

    honoured = _step(model, args, kwargs)
    for kept, used in given:
        if isinstance(used, lucid.Tensor):
            assert _worst(kept, used) == 0.0, f"{factory} wrote into its input"
    for module in flagged:
        module.inplace = False
    ignored = _step(model, args, kwargs)

    (outs_on, grads_on), (outs_off, grads_off) = honoured, ignored
    assert [n for n, _ in outs_on] == [n for n, _ in outs_off]
    for (name, on), (_, off) in zip(outs_on, outs_off, strict=True):
        assert _worst(on, off) == 0.0, (
            f"{factory}.{name} depends on inplace=True being ignored — an "
            "activation writes a tensor the forward reads again; build that "
            "one with inplace=False"
        )
    assert grads_on.keys() == grads_off.keys()
    differ = [n for n in grads_on if _worst(grads_on[n], grads_off[n]) != 0.0]
    assert not differ, f"{factory}: gradients depend on inplace=True: {differ[:6]}"


_BACKBONE: dict[str, object] = {
    "num_classes": 3,
    "backbone_layers": (1, 1, 1, 1),
    "fpn_out_channels": 32,
    "rpn_post_nms_top_n": 10,
}

#: The detectors ``test_models_train_step`` leaves to ``test_models_detection``
#: (they train with targets or proposals): factory, overrides, keyword inputs.
_DETECTORS: dict[str, tuple[str, dict[str, object], dict[str, object]]] = {
    "faster_rcnn": ("faster_rcnn_resnet50_fpn", _BACKBONE, {}),
    "mask_rcnn": (
        "mask_rcnn_resnet50_fpn",
        {**_BACKBONE, "roi_representation": 64, "mask_hidden_channels": 16},
        {},
    ),
    "rcnn": ("rcnn", {}, {"proposals": [[5.0, 5.0, 40.0, 40.0]]}),
    "fast_rcnn": ("fast_rcnn", {}, {"proposals": [[5.0, 5.0, 40.0, 40.0]]}),
}


@pytest.mark.parametrize("family", list(_DETECTORS))
def test_detector_forward_is_the_same_with_inplace_off(family: str) -> None:
    factory, overrides, extra = _DETECTORS[family]
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).train()
    flagged = _inplace_modules(model)
    assert flagged, f"{family}: expected a module built with inplace=True"
    x = lucid.randn(1, 3, 64, 64)
    kwargs = {k: [lucid.tensor(v)] for k, v in extra.items()}

    def step() -> tuple[list[lucid.Tensor], list[lucid.Tensor]]:
        model.zero_grad()
        lucid.manual_seed(1)
        out = [t for _, t in _tensors_of(model(x, **kwargs)) if t.is_floating_point()]
        sum((t.sum() for t in out if t.requires_grad), lucid.tensor(0.0)).backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return [t.detach().clone() for t in out], grads

    on = step()
    for module in flagged:
        module.inplace = False
    off = step()
    for got, want in zip([*on[0], *on[1]], [*off[0], *off[1]], strict=True):
        assert _worst(got, want) == 0.0, f"{factory} depends on inplace=True"


def test_the_walk_finds_the_flag() -> None:
    """Guards the test above: a walk that finds nothing would skip every family."""
    for case in CASES:
        family, factory, overrides, _make = case.values if hasattr(case, "values") else case
        if family == "xception":
            assert _inplace_modules(models.create_model(factory, **overrides))
            return
    pytest.fail("xception is not in test_models_train_step.CASES")
