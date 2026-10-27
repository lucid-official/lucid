"""Training a model against the compression it will be shipped with.

Fitting a palette to a finished model is the cheap thing, and measured
on a trained ResNet-50 it is not enough: four-bit palettization changes
every one of five top-1 predictions. Two export-time remedies were tried
and both failed — leaving the stem and classifier uncompressed did
nothing, and a table per channel was worse at six bits. The loss is
spread through the network, which is the shape of a problem no export
setting fixes.

What is left is to show the network the compression while it can still
learn. The forward pass uses the compressed weight; the optimizer
updates the full-precision one behind it, reached through
``w + (compress(w) - w).detach()`` because compression's own derivative
is zero almost everywhere.

Measured here on a small classifier trained to 0.98, against a control
that gets the same number of extra steps without compression — which
matters, since the first version of this measurement credited the
fine-tuning itself to the feature:

    bits   post-training   compression-aware
    2          0.982             0.995
    3          0.997             0.993
    4          0.997             0.995

It helps where the compression costs something and not otherwise, which
is what should be expected and is worth writing down: on a small model
at four bits there is nothing to recover.

The other half is that the export has to write the palette the model was
trained against. It did not. A settled weight holds exactly ``2**bits``
values per row and Lloyd's algorithm still came back 9.3e-03 away from
them — two initial centres can land inside one repeated value, leaving
another unclaimed, and an entry nothing lands on never moves again. The
fitter takes the values directly when there are few enough of them, and
the round trip is exact now.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.coreml import _build

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Net(nn.Module):
    """Two convolutions, one of which is over the export's threshold."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Conv2d(3, 32, 3, padding=1)
        self.second = nn.Conv2d(32, 32, 3, padding=1)
        self.head = nn.Linear(32, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        hidden = F.relu(self.second(F.relu(self.first(x))))
        return self.head(hidden.mean(dim=(2, 3)))


def _net() -> nn.Module:
    lucid.manual_seed(0)
    return _Net().eval()


class TestWhatItCovers:
    def test_it_matches_the_export_threshold(self) -> None:
        """Training a weight the export leaves alone is wasted work.

        ``first`` is 864 elements and ``head`` is 320, both under the
        2048 the export wants before a table pays for itself; ``second``
        is 9216 and is the only one either of them touches.
        """
        aware = cml.CompressionAware(_net(), weights=cml.Palettize(bits=4))
        assert aware.covered == ["second.weight"]

    def test_a_model_with_nothing_to_compress_is_refused(self) -> None:
        """Rather than training against a compression of nothing."""
        small = nn.Sequential(nn.Linear(4, 4)).eval()
        with pytest.raises(ValueError, match="large enough"):
            cml.CompressionAware(small, weights=cml.Palettize(bits=4))

    def test_float_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="compresses nothing"):
            cml.CompressionAware(_net(), weights=cml.WeightPrecision.FLOAT)


class TestTheForwardPassIsCompressed:
    @pytest.mark.parametrize(
        ("name", "weights"),
        [
            ("palette", cml.Palettize(bits=4)),
            ("sparse", cml.Sparsify(ratio=0.5)),
            ("int8", cml.WeightPrecision.INT8),
        ],
    )
    def test_it_answers_as_the_settled_model_will(self, name, weights) -> None:
        """Otherwise the loss being trained on is not the shipped loss."""
        model = _net()
        x = lucid.randn(2, 3, 16, 16)
        aware = cml.CompressionAware(model, weights=weights)
        during = aware(x)
        settled = aware.settle()
        assert float((settled(x) - during).abs().max().item()) == 0.0

    def test_the_uncompressed_model_is_left_alone_until_settled(self) -> None:
        """The parameters stay full precision while training runs.

        Snapping them every step would be a different algorithm — the
        optimizer would have nowhere to accumulate an update smaller than
        one palette step, and nothing would move.
        """
        model = _net()
        x = lucid.randn(2, 3, 16, 16)
        before = model(x)
        aware = cml.CompressionAware(model, weights=cml.Palettize(bits=2))
        aware(x)
        assert float((model(x) - before).abs().max().item()) == 0.0

    def test_the_gradient_reaches_the_real_parameter(self) -> None:
        """The straight-through estimator, which is what makes it train.

        Compression has a zero derivative almost everywhere. Without the
        estimator every gradient would stop at it and the parameters
        would never move.
        """
        model = _net()
        aware = cml.CompressionAware(model, weights=cml.Palettize(bits=4))
        aware(lucid.randn(2, 3, 16, 16)).sum().backward()
        assert model.second.weight.grad is not None
        assert float(model.second.weight.grad.abs().max().item()) > 0.0

    def test_the_parameters_are_still_the_model_s(self) -> None:
        """An optimizer built on either has to update the same tensors."""
        model = _net()
        aware = cml.CompressionAware(model, weights=cml.Palettize(bits=4))
        assert len(list(aware.parameters())) == len(list(model.parameters()))

    def test_a_forward_leaves_every_parameter_in_its_place(self) -> None:
        """The compressed weight only shadows the parameter during a call.

        It used to be assigned over the parameter and the parameter assigned
        back, which moved ``second.weight`` behind ``second.bias`` on every
        forward; assigning a plain tensor to a registered parameter is now
        refused outright (LCD-551).
        """
        model = _net()
        before = list(model.named_parameters())
        aware = cml.CompressionAware(model, weights=cml.Palettize(bits=4))
        aware(lucid.randn(2, 3, 16, 16))
        after = list(model.named_parameters())
        assert [name for name, _ in after] == [name for name, _ in before]
        assert all(now is then for (_, now), (_, then) in zip(after, before, strict=True))
        assert "weight" not in model.second.__dict__
        assert model.second.weight is dict(before)["second.weight"]


class TestTheExportWritesWhatWasTrained:
    """The half that was broken, and the reason the fitter changed.

    Training against one palette and shipping another is worse than not
    training: it looks like it worked.
    """

    @pytest.mark.parametrize("bits", [2, 3, 4, 6])
    def test_the_palette_survives_the_export(self, bits, tmp_path) -> None:
        aware = cml.CompressionAware(_net(), weights=cml.Palettize(bits=bits))
        settled = aware.settle()
        probe = lucid.randn(1, 3, 16, 16)

        exported = cml.export(
            settled,
            probe,
            f"{tmp_path}/aware_{bits}.mlpackage",
            weights=cml.Palettize(bits=bits),
        )
        try:
            # Exact rather than merely close: the weights are already on
            # the palette, so the only difference left is float32.
            assert exported.verify(settled, probe, relative=True) < 1e-5
        finally:
            exported.close()

    def test_the_fitter_recovers_a_palette_it_is_given(self) -> None:
        """Directly, without an export in the way.

        Lloyd's algorithm came back 9.3e-03 from a palette its own input
        was built out of. A row holding at most ``count`` values has an
        exact table and iteration is a lottery for finding it.
        """
        lucid.manual_seed(0)
        count = 16
        table = lucid.sort(lucid.randn(4, count), dim=-1).values
        keys = (lucid.arange(4 * 64) % count).reshape(4, 64).to(lucid.int64)
        rows = lucid.gather(table, 1, keys)

        fitted = _build._palettes_for(rows, count)
        assert float((fitted - table).abs().max().item()) < 1e-6

    def test_an_ordinary_weight_still_goes_through_the_iteration(self) -> None:
        """The exact path has to be as narrow as the case it serves.

        A weight with more distinct values than the table has entries is
        fitted the way it always was, and the result is a table of the
        right size that its own data lands inside.
        """
        lucid.manual_seed(0)
        rows = lucid.randn(4, 4096)
        fitted = _build._palettes_for(rows, 16)
        assert tuple(fitted.shape) == (4, 16)
        assert float(fitted.min().item()) >= float(rows.min().item())
        assert float(fitted.max().item()) <= float(rows.max().item())

    def test_sparsity_survives_the_export(self, tmp_path) -> None:
        aware = cml.CompressionAware(_net(), weights=cml.Sparsify(ratio=0.5))
        settled = aware.settle()
        probe = lucid.randn(1, 3, 16, 16)

        exported = cml.export(
            settled,
            probe,
            f"{tmp_path}/sparse.mlpackage",
            weights=cml.Sparsify(ratio=0.5),
        )
        try:
            assert exported.verify(settled, probe, relative=True) < 1e-5
        finally:
            exported.close()

    def test_int8_survives_the_export_too(self, tmp_path) -> None:
        """It did not, and the reason was worth chasing twice.

        The export derived its scale as ``max / 127.5``, and a settled
        weight's largest value is ``127 * scale``, so re-deriving gave a
        grid 0.4% finer than the one trained against — with the weights
        not on it. This test used to pin that inexactness, on the grounds
        that changing the convention would move the numbers of every int8
        package already written.

        Neither was necessary. The convention is untouched; the writer
        notices when a weight already sits on a grid and writes it on
        that one rather than fitting another. An ordinary weight has as
        many distinct values as it has elements and never takes that
        path, so nothing already written moves.
        """
        aware = cml.CompressionAware(_net(), weights=cml.WeightPrecision.INT8)
        settled = aware.settle()
        probe = lucid.randn(1, 3, 16, 16)

        exported = cml.export(
            settled,
            probe,
            f"{tmp_path}/int8.mlpackage",
            weights=cml.WeightPrecision.INT8,
        )
        try:
            assert exported.verify(settled, probe, relative=True) < 1e-5
        finally:
            exported.close()


class TestItActuallyHelps:
    """The claim, on a model small enough to train inside a test.

    Two bits, because that is where the compression costs something.
    Both arms get the same number of steps — the first version of this
    measurement did not, and credited the extra fine-tuning to the
    feature — and the same batches: the control's draws are replayed for
    the compression-aware arm, so compression is the only difference.
    Drawn afresh, the batch order alone moved the result from 14 % of the
    cost recovered to 65 %.

    The arms are compared by held-out cross-entropy, not accuracy.  All
    three sit at 0.99–1.0 on 360 images, where one image is 0.003 and is
    the whole difference between them: ``accuracy(tuned) >=
    accuracy(snapped)`` held at one seed by one image and failed on CI
    hardware by one image the other way.  The loss resolves what accuracy
    cannot — across four seeds two-bit snapping raised it 1.4–1.8x over
    the control, and training against the palette on the same batches
    took back 65–71 % of that.
    """

    def test_what_two_bits_cost_is_recovered(self) -> None:
        lucid.manual_seed(1)
        prototypes = lucid.randn(10, 3, 16, 16)

        def batch(size: int = 60) -> tuple[lucid.Tensor, lucid.Tensor]:
            labels = (lucid.arange(size) % 10).to(lucid.int64)
            return prototypes[labels] + lucid.randn(size, 3, 16, 16) * 0.6, labels

        def train(module: nn.Module, steps: int, rate: float) -> None:
            optimizer = optim.Adam(module.parameters(), lr=rate)
            for step in range(steps):
                x, y = batch()
                loss = F.cross_entropy(module(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if isinstance(module, cml.CompressionAware) and step % 150 == 149:
                    module.refit()

        def measure(module: nn.Module) -> tuple[float, float]:
            # Accuracy and cross-entropy on the same held-out draws.
            lucid.manual_seed(7)
            hits = loss = 0.0
            with lucid.no_grad():
                for _ in range(6):
                    x, y = batch()
                    out = module(x)
                    hits += float((out.argmax(dim=1) == y).to(lucid.float32).mean().item())
                    loss += float(F.cross_entropy(out, y).item())
            return hits / 6, loss / 6

        def copy_of(source: nn.Module) -> nn.Module:
            fresh = _Net().eval()
            fresh.load_state_dict(source.state_dict())
            return fresh

        lucid.manual_seed(0)
        base = _Net().eval()
        train(base, 500, 3e-3)
        assert measure(base)[0] > 0.9, "the task has to be learnable to say anything"

        control = copy_of(base)
        draws = lucid.get_rng_state()
        train(control, 400, 1e-3)
        snapped = cml.CompressionAware(copy_of(control), weights=cml.Palettize(bits=2)).settle()

        aware = cml.CompressionAware(copy_of(base), weights=cml.Palettize(bits=2))
        lucid.set_rng_state(draws)  # the 400 batches the control trained on
        train(aware, 400, 1e-3)
        tuned = aware.settle()

        control_acc, control_loss = measure(control)
        _, snapped_loss = measure(snapped)
        tuned_acc, tuned_loss = measure(tuned)
        cost = snapped_loss - control_loss
        assert cost > 0.25 * control_loss, "two bits has to cost something to recover"
        assert snapped_loss - tuned_loss > cost / 3, (
            control_loss,
            snapped_loss,
            tuned_loss,
        )
        # Six images of 360: the settled model classifies like the control.
        assert tuned_acc >= control_acc - 1 / 60, (control_acc, tuned_acc)
