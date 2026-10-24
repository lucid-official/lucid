"""Every zoo family computes in the dtype it is handed (LCD-562).

DETR raised ``DtypeMismatch`` in float16, bfloat16 and float64: the first
input of its decoder was ``lucid.zeros(..., device=)`` — float32 whatever
the model was — and so was its sine position table.  The same defect ran
through two dozen families without raising.  Lucid widens a mixed-dtype
product rather than refusing it, so a float16 model that met one float32
tensor ran the rest of its forward in float32 and answered in float32:
MaskFormer, the R-CNNs, YOLO, every diffusion and flow family, the world
models.  A float64 one answered with float32 values where a float32 draw
or table met it first.

The rule is that a tensor a forward builds for itself — a start state, a
grid, a weight table, a noise draw, a zero fallback — takes the activations'
dtype as well as their device.  Position and time tables are computed at
float32 at least and cast once, the reference's convention.

* **eval** — every family of the compiled-training matrix (its smallest
  factory, shrunk by the matrix's own rules) runs one forward in float64 on
  the CPU and in float16 and bfloat16 on Metal; every floating tensor it
  answers with has the input's dtype.
* **train** — every family of the training-step table takes its step in
  float64 and float16 on the CPU; the loss and every gradient keep the
  model's dtype.
* DETR and MaskFormer, the reported pair, are pinned through their public
  factories; Score-SDE and Fast R-CNN, whose float64 answers carried
  float32 values, are held to the float32 model's numbers.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as models
from lucid.models._registry import _REGISTRY
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile import _zoo_matrix as zoo
from lucid.test.unit.models.test_models_train_step import CASES, _family_of

_F16, _BF16, _F64 = lucid.float16, lucid.bfloat16, lucid.float64

#: Families the eval sweep cannot drive: their forward is a rollout, not one
#: call (the matrix's own ``skip`` reasons).  Their dtype is not covered.
_ROLLOUTS = {f for f, spec in zoo.SPECS.items() if spec.skip is not None}

_PAIRS = [
    (family, task)
    for task in zoo.TASKS
    for family in zoo.families(task)
    if family not in _ROLLOUTS
]

_EVAL = [("cpu", _F64), ("metal", _F16), ("metal", _BF16)]


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _need(device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")


def _floating(value: Any, path: str = "output") -> list[tuple[str, lucid.Tensor]]:
    """Every floating tensor in a forward's answer, named for the message."""
    if isinstance(value, lucid.Tensor):
        return [(path, value)] if value.is_floating_point() else []
    if isinstance(value, (tuple, list)):
        return [t for i, v in enumerate(value) for t in _floating(v, f"{path}[{i}]")]
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in _floating(v, f"{path}.{k}")]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            t
            for f in dataclasses.fields(value)
            for t in _floating(getattr(value, f.name, None), f"{path}.{f.name}")
        ]
    return []


def _cast(value: Any, device: str, dtype: lucid.dtype) -> Any:
    """Inputs on ``device``; floating ones in ``dtype``, integer ones untouched."""
    if isinstance(value, lucid.Tensor):
        value = value.to(device)
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (tuple, list)):
        return type(value)(_cast(v, device, dtype) for v in value)
    if isinstance(value, dict):
        return {k: _cast(v, device, dtype) for k, v in value.items()}
    return value


def _wrong(answer: Any, dtype: lucid.dtype) -> list[str]:
    return [f"{n}: {_name(t.dtype)}" for n, t in _floating(answer) if t.dtype != dtype]


@pytest.mark.parametrize(
    ("device", "dtype"), _EVAL, ids=[f"{d}-{_name(t)}" for d, t in _EVAL]
)
@pytest.mark.parametrize(("family", "task"), _PAIRS, ids=[f for f, _ in _PAIRS])
def test_a_forward_answers_in_the_dtype_it_is_given(
    family: str,
    task: str,
    device: str,
    dtype: lucid.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _need(device)
    monkeypatch.setattr(zoo, "DEV", device)
    _, model, cfg, _ = zoo.build(family, task)
    model = model.to(device).to(dtype).eval()
    args, kwargs = zoo.default_inputs(model, cfg, family, 1)
    with lucid.no_grad():
        answer = model(*_cast(args, device, dtype), **_cast(kwargs, device, dtype))
    assert _floating(answer), "the forward answered with no floating tensor"
    wrong = _wrong(answer, dtype)
    assert not wrong, f"a {_name(dtype)} {family} answered with {wrong[:6]}"


_TRAIN = [_F64, _F16]


@pytest.mark.parametrize("dtype", _TRAIN, ids=[_name(t) for t in _TRAIN])
@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"),
    CASES,
    ids=[_family_of(c) for c in CASES],
)
def test_a_training_step_keeps_the_model_dtype(
    family: str,
    factory: str,
    overrides: dict[str, Any],
    make_inputs: Any,
    dtype: lucid.dtype,
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).to(dtype)
    model.train()
    args, kwargs = make_inputs()
    loss = model(*_cast(args, "cpu", dtype), **_cast(kwargs, "cpu", dtype)).loss
    assert loss.dtype == dtype, f"a {_name(dtype)} {family} has a {loss.dtype} loss"
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    wrong = [n for n, g in grads if g.dtype != dtype]
    assert not wrong, f"gradients not in {_name(dtype)}: {wrong[:6]}"


def test_every_family_is_swept_or_named_as_a_rollout() -> None:
    families = {e.family for e in _REGISTRY.values()}
    swept = {f for f, _ in _PAIRS}
    assert families - swept == _ROLLOUTS, sorted(families - swept - _ROLLOUTS)


# -- the reported pair, through the public factories --------------------------

_SMALL = {
    "detr_resnet50": {
        "num_classes": 4,
        "backbone_layers": (1, 1, 1, 1),
        "n_head": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "num_queries": 4,
    },
    "maskformer_resnet50": {
        "num_classes": 4,
        "backbone_layers": (1, 1, 1, 1),
        "n_head": 2,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "num_queries": 4,
        "fpn_out_channels": 32,
    },
}

_INFER = [
    ("cpu", _F16),
    ("cpu", _BF16),
    ("cpu", _F64),
    ("metal", _F16),
    ("metal", _BF16),
]

#: Largest error against the float32 model, relative to its largest output.
_TOL = {_F64: 1e-4, _F16: 5e-2, _BF16: 2e-1}


def _paired(factory: str, device: str, dtype: lucid.dtype) -> tuple[Any, Any]:
    lucid.manual_seed(0)
    wide = models.create_model(factory, **_SMALL[factory]).eval()
    narrow = models.create_model(factory, **_SMALL[factory]).eval()
    narrow.load_state_dict(wide.state_dict())
    return wide.to(device), narrow.to(device).to(dtype)


@pytest.mark.parametrize(
    ("device", "dtype"), _INFER, ids=[f"{d}-{_name(t)}" for d, t in _INFER]
)
@pytest.mark.parametrize("factory", sorted(_SMALL))
def test_detr_and_maskformer_infer_in_reduced_and_double_precision(
    factory: str, device: str, dtype: lucid.dtype
) -> None:
    _need(device)
    wide, narrow = _paired(factory, device, dtype)
    lucid.manual_seed(1)
    x = lucid.randn(1, 3, 64, 64).to(device)
    with lucid.no_grad():
        want = wide(x)
        got = narrow(x.to(dtype))
    assert not _wrong(got, dtype), _wrong(got, dtype)
    for (name, g), (_, w) in zip(_floating(got), _floating(want), strict=True):
        a = g.to("cpu").to(lucid.float64).numpy()
        b = w.to("cpu").to(lucid.float64).numpy()
        assert np.isfinite(a).all(), name
        scale = max(float(np.abs(b).max()), 1e-6)
        err = float(np.abs(a - b).max()) / scale
        assert err < _TOL[dtype], f"{name}: {err:.3g} off the float32 model"


# -- float64 answers that carried float32 values --------------------------------


def _shrunk(family: str, task: str) -> tuple[str, dict[str, Any]]:
    name = zoo.representative(family, task)
    assert name is not None
    overrides = zoo.shrink(_REGISTRY[name].default_config)
    overrides.update(zoo.SPECS.get(family, zoo.Spec()).overrides)
    return name, overrides


def _float64_twin(name: str, overrides: dict[str, Any]) -> tuple[Any, Any]:
    lucid.manual_seed(0)
    narrow = models.create_model(name, **overrides).eval()
    wide = models.create_model(name, **overrides).eval()
    wide.load_state_dict(narrow.state_dict())
    return narrow, wide.to(_F64)


def _close(got: lucid.Tensor, want: lucid.Tensor, tol: float, what: str) -> None:
    assert got.dtype == _F64, f"{what} is {got.dtype}"
    a, b = got.numpy(), want.to(_F64).numpy()
    assert a.shape == b.shape, what
    scale = max(float(np.abs(b).max()), 1e-12)
    assert float(np.abs(a - b).max()) / scale < tol, what


def test_score_sde_float64_draws_its_target_and_noise_scale_in_float64() -> None:
    """``target`` was a float32 draw and ``std`` a float32 schedule value."""
    narrow, wide = _float64_twin(*_shrunk("score_sde", "image-generation"))
    lucid.manual_seed(1)
    x = lucid.randn(2, 3, 32, 32)
    lucid.manual_seed(2)
    want = narrow(x)
    lucid.manual_seed(2)
    got = wide(x.to(_F64))
    # The same seed draws the same numbers in either dtype, up to float32's
    # own error in the normal's tails, so the float64 answer is the float32
    # one, widened — not float32 values handed over as float64.
    _close(got.target, want.target, 1e-4, "target")
    _close(got.std, want.std, 1e-5, "std")
    _close(got.score, want.score, 1e-3, "score")
    _close(got.loss, want.loss, 1e-3, "loss")


def test_fast_rcnn_float64_keeps_its_dtype_with_and_without_proposals() -> None:
    """With no proposals the empty stand-in was a float32 ``zeros``."""
    narrow, wide = _float64_twin(*_shrunk("fast_rcnn", "object-detection"))
    lucid.manual_seed(1)
    x = lucid.randn(1, 3, 64, 64)
    with lucid.no_grad():
        empty = wide(x.to(_F64))
        assert not _wrong(empty, _F64), _wrong(empty, _F64)
        boxes = [lucid.tensor([[4.0, 4.0, 40.0, 40.0], [10.0, 12.0, 60.0, 50.0]])]
        want = narrow(x, proposals=boxes)
        got = wide(x.to(_F64), proposals=[b.to(_F64) for b in boxes])
    _close(got.logits, want.logits, 1e-4, "logits")
    _close(got.pred_boxes, want.pred_boxes, 1e-4, "pred_boxes")
