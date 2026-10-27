"""The zoo matrix cannot call a NaN model ``ok`` (LCD-563).

``python -m lucid.test.unit.compile._zoo_matrix`` reported realnvp ``ok``
with ``loss_rel NaN, grad 0.0``: every comparison it makes goes quiet on NaN,
and its float64 tie-breaker then compared 0 with 0.  Each status now asks
first for a finite loss and finite gradients that are not all zero
(:func:`_zoo_matrix.nonfinite`); these pin that rule, its wiring into a
mode's status and the sweep's verdict, and realnvp's in-domain inputs —
without running the matrix, which takes a process per family.
"""

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile import _zoo_matrix as zoo

_GRAD = {"w": np.array([0.5, -1.0], dtype=np.float32)}


@pytest.mark.parametrize(
    ("loss", "grads", "why"),
    [
        (math.nan, _GRAD, "loss is nan"),
        (math.inf, _GRAD, "loss is inf"),
        (-math.inf, _GRAD, "loss is -inf"),
        (1.0, {"b": np.zeros(2), "w": np.array([1.0, math.nan])}, "1 gradient(s)"),
        (1.0, {"w": np.array([math.inf, 1.0])}, "1 gradient(s) not finite, first 'w'"),
        (1.0, {"w": np.zeros(3), "b": np.zeros(1)}, "every gradient is zero"),
        (1.0, {}, "no parameter received a gradient"),
    ],
)
def test_a_step_that_cannot_be_compared_says_why(
    loss: float, grads: dict[str, np.ndarray], why: str
) -> None:
    reason = zoo.nonfinite(loss, grads)
    assert reason is not None and reason.startswith(why), reason


@pytest.mark.parametrize(
    "grads",
    [_GRAD, {"w": np.zeros(3), "b": np.array([1e-30], dtype=np.float32)}],
    ids=["ordinary", "one tiny gradient"],
)
def test_a_finite_step_that_trains_something_is_comparable(grads: dict[str, np.ndarray]) -> None:
    assert zoo.nonfinite(0.0, grads) is None


class _Frozen(nn.Module):
    """Its output does not depend on its one parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(3))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x + self.w * 0.0


_Batch = tuple[tuple[lucid.Tensor, ...], dict[str, Any]]
_Grads = dict[str, np.ndarray]
_needs_metal = pytest.mark.skipif(not metal_available(), reason="the matrix runs on Metal")


def _batches(x: lucid.Tensor) -> list[_Batch]:
    return [((x.to(zoo.DEV),), {}), ((x.to(zoo.DEV),), {})]


def _linear() -> nn.Module:
    return nn.Linear(3, 2)


@_needs_metal
@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize(
    ("make", "x", "why"),
    [
        (_linear, lucid.tensor([[1.0, math.nan, 0.0]]), "loss is nan"),
        (_Frozen, lucid.ones(1, 3), "every gradient is zero"),
    ],
    ids=["nan", "zero-grad"],
)
def test_a_mode_whose_eager_step_is_unusable_is_nonfinite(
    make: Callable[[], nn.Module], x: lucid.Tensor, why: str, mode: str
) -> None:
    model = make().to(zoo.DEV)
    status, detail = zoo._check_mode("synthetic", "test", model, mode, _batches(x))
    assert status == "nonfinite"
    assert detail.startswith(f"eager: {why}"), detail


@_needs_metal
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_a_mode_whose_compiled_step_is_unusable_is_nonfinite(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compiled(
        model: nn.Module, batches: list[_Batch], seed: int, start: _Grads
    ) -> tuple[float, str]:
        zoo._zero(model)
        return math.nan, ""

    monkeypatch.setattr(zoo, "_compiled_step", compiled)
    model = _linear().to(zoo.DEV)
    batches = _batches(lucid.ones(1, 3))
    status, detail = zoo._check_mode("synthetic", "test", model, mode, batches)
    assert (status, detail) == ("nonfinite", "compiled: loss is nan")


@_needs_metal
@pytest.mark.parametrize(("reference", "status"), [("finite", "wrong"), ("nan", "nonfinite")])
def test_a_nonfinite_float64_arm_is_nonfinite_not_wrong(
    reference: str, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiled agrees on the loss and doubles every gradient, so the mode
    reaches the float64 tie-breaker; a finite one calls it wrong (the
    control), a NaN one may not, and nothing can retry it into ``ok``."""
    model = _linear().to(zoo.DEV)

    def compiled(
        module: nn.Module, batches: list[_Batch], seed: int, start: _Grads
    ) -> tuple[float, str]:
        loss = zoo._eager_step(module, batches[1], seed)
        for p in module.parameters():
            assert p.grad is not None
            p.grad = p.grad * 2.0
        return loss, ""

    def float64(
        family: str, task: str, mode: str, start: _Grads, batch: _Batch, seed: int
    ) -> tuple[float, _Grads]:
        if reference == "nan":
            return math.nan, {}
        return zoo._eager_step(model, batch, seed), zoo._grads(model)

    monkeypatch.setattr(zoo, "_compiled_step", compiled)
    monkeypatch.setattr(zoo, "_float64_reference", float64)
    got, detail = zoo._check_mode("synthetic", "test", model, "eval", _batches(lucid.ones(1, 3)))
    assert got == status, detail
    if status == "nonfinite":
        assert detail == "float64 verdict, float64 without dropout: loss is nan"
    else:
        assert detail["float64"]["ok"] is False, detail


def test_a_retry_that_is_nonfinite_makes_the_pair_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "wrong" first pair is retried; a NaN on the retry cannot pass."""
    answers = iter([("wrong", {"grad": 1.0}), ("nonfinite", "eager: loss is nan")])

    def build(family: str, task: str) -> tuple[str, Any, Any, dict[str, Any]]:
        return "synthetic_small", _linear(), SimpleNamespace(), {}

    def inputs(model: Any, cfg: Any, family: str, seed: int) -> _Batch:
        return (), {}

    def check(*args: object) -> tuple[str, Any]:
        return next(answers)

    monkeypatch.setattr(zoo, "build", build)
    monkeypatch.setattr(zoo, "default_inputs", inputs)
    monkeypatch.setattr(zoo, "_check_mode", check)
    rec = zoo.run_family("synthetic", "test")
    assert rec["status"] == "nonfinite"
    assert rec["detail"] == "eval on the retry batches, eager: loss is nan"


def test_a_nonfinite_pair_fails_the_sweep_whatever_expected_says() -> None:
    rec = {
        "family": "faster_rcnn",
        "task": "object-detection",
        "status": "nonfinite",
        "mode": "eval",
        "detail": "eval eager: loss is nan",
    }
    assert ("faster_rcnn", "object-detection") in zoo.EXPECTED
    assert zoo.verdict(rec) == "nonfinite: eval eager: loss is nan"


def test_realnvp_images_are_drawn_inside_its_data_domain() -> None:
    class _Image(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x

    cfg = SimpleNamespace(sample_size=8, in_channels=3)
    for seed in (1, 2, 3, 4):
        (x,), _ = zoo.default_inputs(_Image(), cfg, "realnvp", seed)
        values = x.numpy()
        assert values.min() >= 0.0 and values.max() < 1.0


@_needs_metal
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_realnvp_trains_on_its_domain_and_is_nan_off_it(mode: str) -> None:
    """The counterfactual is why the domain is in its Spec at all."""
    name = zoo.representative("realnvp", "image-generation")
    assert name is not None
    small = {"sample_size": 16, "num_scales": 2, "residual_blocks": 1, "base_dim": 8}
    lucid.manual_seed(0)
    model = lucid.models.create_model(name, **small).to(zoo.DEV)
    model.train(mode == "train")
    cfg = SimpleNamespace(**small, in_channels=model.config.in_channels)

    batch = zoo.default_inputs(model, cfg, "realnvp", 1)
    loss = zoo._eager_step(model, batch, 100)
    assert zoo.nonfinite(loss, zoo._grads(model)) is None

    off_domain = lucid.randn(*batch[0][0].shape).to(zoo.DEV)
    loss = zoo._eager_step(model, ((off_domain,), {}), 100)
    assert zoo.nonfinite(loss, zoo._grads(model)) == "loss is nan"
