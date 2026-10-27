"""An exported package must answer its input, not the one it was traced with.

Every other numeric test in this directory compares one prediction
against one eager run. That cannot see the worst failure this subsystem
has: a package whose answer was decided while tracing. Such a package
loads, reports no error, agrees perfectly with the model on the input it
was built from, and returns that same answer for every input afterwards.

The compile backend has the same hazard and the same test —
``test_no_frozen_results.py`` — written after ``fft.fftn`` was off by 20
on its second call. The export path reaches it two ways. A traced value
that the builder decides is a constant is frozen by definition, so the
question is only whether it really was one; and the graph is walked from
the outputs, so an operation whose input edges the tracer failed to
record looks like it depends on nothing.

The property is checked rather than the mechanism: export, call with two
inputs the eager model distinguishes, and require the answer to move.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


#: Shapes built out of the input, index arithmetic that is not, and the
#: mixtures in between — the cases where "this does not depend on the
#: input" is a judgement rather than an observation.
CASES = [
    ("arange-added", lambda t: t + lucid.arange(int(t.shape[-1])).to(lucid.float32)),
    ("zeros-added", lambda t: t + lucid.zeros(*t.shape)),
    ("eye-scaled", lambda t: t[:, :3] * lucid.eye(3)),
    ("triangle-masked", lambda t: lucid.tril(t, 0)),
    ("index-gathered", lambda t: lucid.gather(t, 1, lucid.argmax(t, dim=1, keepdim=True))),
    ("reshaped-and-back", lambda t: t.reshape(-1).reshape(*t.shape) * 2.0),
    ("relu-control", lambda t: t.relu()),
]


@pytest.mark.parametrize(("name", "fn"), CASES, ids=[c[0] for c in CASES])
def test_the_second_call_answers_the_second_input(name, fn, tmp_path):
    lucid.manual_seed(0)
    model = _Apply(fn).eval()
    first = lucid.randn(3, 5)
    second = lucid.randn(3, 5) * 7 + 3

    want_first = model(first)
    want_second = model(second)
    spread = float((want_first - want_second).abs().max().item())
    assert spread > 1e-6, "the probe itself is blind — eager gives the same answer"

    exported = cml.export(model, first, f"{tmp_path}/{name}.mlpackage")
    try:
        got_first = exported.predict(first)
        got_second = exported.predict(second)
        moved = float((got_first - got_second).abs().max().item())
        assert moved > 1e-6, (
            f"{name}: the package returned the same values for two different "
            "inputs — its answer was decided while tracing"
        )
        scale = max(float(want_second.abs().max().item()), 1e-9)
        assert float((got_second - want_second).abs().max().item()) / scale < 1e-5
    finally:
        exported.close()


#: One representative per architecture the smoke test covers with a
#: single image, at a small input. A frozen answer inside a real model
#: is what this is for: the shapes agree, the first call agrees, and
#: nothing else says anything.
MODELS = [
    ("resnet_18", (1, 3, 64, 64)),
    ("swin_tiny", (1, 3, 224, 224)),
    ("vit_base_16", (1, 3, 224, 224)),
    ("convnext_tiny", (1, 3, 64, 64)),
    ("efficientnet_b0", (1, 3, 64, 64)),
    ("unet", (1, 1, 64, 64)),
]


@pytest.mark.parametrize(("factory", "shape"), MODELS, ids=[m[0] for m in MODELS])
def test_a_real_model_answers_the_second_input(factory, shape, tmp_path):
    lucid.manual_seed(0)
    model = M.create_model(factory).eval()
    first = lucid.randn(*shape)
    second = lucid.randn(*shape) * 3 + 1

    # The probe has to be able to see: an untrained model whose head is
    # zero-initialised answers both inputs with the same near-zero
    # values, and a package that froze its answer would look identical
    # to one that did not. EfficientNet is such a model, so the
    # comparison is made relative to what eager itself distinguishes.
    eager_spread = float((_out(model(first)) - _out(model(second))).abs().max().item())
    if eager_spread <= 1e-6:
        pytest.skip("this untrained model gives both inputs the same answer")

    exported = cml.export(model, first, f"{tmp_path}/{factory}.mlpackage")
    try:
        got_first = exported.predict(first)
        got_second = exported.predict(second)
        moved = float((got_first - got_second).abs().max().item())
        assert moved > eager_spread / 100.0, (
            f"{factory}: the package moved by {moved:.2e} between two inputs the "
            f"model itself separates by {eager_spread:.2e} — part of it was "
            "frozen at trace time"
        )
    finally:
        exported.close()


def _out(result: object) -> lucid.Tensor:
    """The tensor a zoo factory answers with, whatever it wraps it in."""
    if isinstance(result, lucid.Tensor):
        return result
    for field in ("logits", "out", "output", "last_hidden_state"):
        value = getattr(result, field, None)
        if isinstance(value, lucid.Tensor):
            return value
    raise AssertionError(f"no tensor in {type(result).__name__}")
