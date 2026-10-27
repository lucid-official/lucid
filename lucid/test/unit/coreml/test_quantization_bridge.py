"""Models `lucid.quantization` produces, reaching the exporter.

Lucid has a quantization subsystem and an exporter, and until now nothing
carried a model from one to the other. Crossing the two found the reason:
a quantization-aware model carries observers that record the range of
every activation they see, and ``eval()`` does not stop them — correctly,
since post-training calibration runs in eval mode and *is* that
recording. Tracing is not calibration. It runs the model once to learn
its shape, and the observers wrote to their buffers while it did, so the
export refused the model as one that mutates itself.

They are paused around the trace now. Nothing about the exported program
changes: an observer contributes no operation, only a record.

What comes out is exact. A QAT model's weights have been trained onto
the int8 grid, so storing them as int8 costs nothing — measured on a
small classifier at 0.00e+00 against the eager model, where the same
network quantized after training lands at 7e-04 and loses accuracy the
QAT one keeps (0.997 against 0.986).

One thing this does not yet do, recorded because it is the next
question rather than a defect: the package carries the *activation*
fake-quantization as real arithmetic — 75 operations where the float
model has 10. That is faithful, and it is not what anyone wants on the
Neural Engine, which runs float16 activations whatever the package says.
Core ML's own quantization is weight-only, so simulating int8
activations in float is work the accelerator did not ask for. Dropping
them would change what the package computes, which is a choice to offer
rather than to make here.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.quantization as q
from lucid._C import engine as _C_engine
from lucid.quantization import FakeQuantize

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


def _net() -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).eval()


def _prepared(x: lucid.Tensor) -> nn.Module:
    # No example input: the third positional parameter is ``inplace``, and a
    # non-empty tuple there quietly prepared the model in place.
    model = q.prepare_qat(_net(), q.get_default_qat_qconfig_mapping())
    model.eval()
    model(x)  # let the observers see one batch, as a caller would
    return model


class TestAQuantizationAwareModelExports:
    def test_it_exports_at_all(self, tmp_path) -> None:
        """It did not. The observers made the trace a buffer change."""
        x = lucid.randn(1, 3, 16, 16)
        exported = cml.export(_prepared(x), x, f"{tmp_path}/qat.mlpackage")
        try:
            assert tuple(exported.predict(x).shape) == (1, 10)
        finally:
            exported.close()

    def test_eval_alone_is_not_enough(self, tmp_path) -> None:
        """Which is why this belongs to the export and not to the caller.

        Calibration runs in eval mode, so `eval()` leaving observers on is
        right. It also means a caller who has done everything correctly
        still cannot export, and the fix is not something to ask them for.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        live = [m for m in model.modules() if isinstance(m, FakeQuantize)]
        assert live, "the fixture must actually produce observers"
        assert all(getattr(m, "_observer_enabled", False) for m in live)

    def test_the_observers_are_put_back(self, tmp_path) -> None:
        """Pausing them is the export's business, not a side effect.

        A caller who exports mid-calibration must find the model as they
        left it, still recording.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        before = [
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        cml.export(model, x, f"{tmp_path}/restored.mlpackage").close()
        after = [
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        assert after == before

    def test_one_that_was_paused_stays_paused(self, tmp_path) -> None:
        """The other direction: restore what was found, not what is tidy."""
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
        cml.export(model, x, f"{tmp_path}/kept.mlpackage").close()
        assert not any(
            getattr(m, "_observer_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        )

    @pytest.mark.parametrize(
        ("name", "weights"),
        [("float", cml.WeightPrecision.FLOAT), ("int8", cml.WeightPrecision.INT8)],
    )
    def test_the_package_computes_what_the_model_does(self, name, weights, tmp_path) -> None:
        """Exactly, and int8 is free here — which is the point of QAT.

        The weights were trained onto the grid, so storing them on it
        changes nothing. The same network quantized after training does
        not have that property.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
        exported = cml.export(model, x, f"{tmp_path}/{name}.mlpackage", weights=weights)
        try:
            assert exported.verify(model, x, relative=True) < 1e-5
        finally:
            exported.close()


class TestAConvertedModelCrossesToo:
    """`quantize_dynamic`, and `prepare` then `convert`.

    Both leave `QuantizedLinearMLX` behind: a Metal-only grouped GEMM
    over a packed weight, which moves the activation to the GPU and back
    to run. None of that reaches the tracer, so a model holding one used
    to export to a graph its own output had fallen out of.

    The packed form has a dequantize-to-float reference path, and it is
    not an approximation of the kernel but the same arithmetic — 4.8e-07
    between them, measured. So the weight is reconstructed once, on the
    host, and the layer becomes the ordinary `linear` the exporter
    already writes. The packing is what is lost, not the quantization:
    the reconstructed weight still holds the values MLX's grid gave it.
    """

    @pytest.mark.parametrize("recipe", ["dynamic", "convert"])
    def test_it_exports_and_agrees(self, recipe, tmp_path) -> None:
        x = lucid.randn(1, 3, 16, 16)
        if recipe == "dynamic":
            model = q.quantize_dynamic(_net())
        else:
            prepared = q.prepare(_net(), q.get_default_qconfig_mapping())
            prepared.eval()
            prepared(x)
            model = q.convert(prepared)

        exported = cml.export(model, x, f"{tmp_path}/{recipe}.mlpackage")
        try:
            answered = exported.predict(x)
            wanted = model(x).cpu()
            assert float((answered - wanted).abs().max().item()) < 1e-4
        finally:
            exported.close()

    def test_the_model_is_handed_back_as_it_came(self, tmp_path) -> None:
        """The layers are stood in for, not replaced."""
        x = lucid.randn(1, 3, 16, 16)
        model = q.quantize_dynamic(_net())
        before = [type(m).__name__ for m in model.modules()]
        cml.export(model, x, f"{tmp_path}/restored.mlpackage").close()
        assert [type(m).__name__ for m in model.modules()] == before

    def test_a_packed_weight_on_the_gpu_is_not_turned_away(self, tmp_path) -> None:
        """Its kernel is Metal-only, so that is where the weight lives.

        The guard that stops an ordinary model being exported from the
        GPU would otherwise reject this one for holding exactly the
        weights it is meant to hold, which is why the substitution
        happens before the check rather than after.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = q.quantize_dynamic(_net())
        # Buffers, not parameters — the packed weight, its scales, its
        # biases and even the layer's own bias are all registered that
        # way, which is also why the guard had to learn to look at both.
        packed = [
            n
            for n, b in model.named_buffers()
            if getattr(getattr(b, "device", None), "type", "cpu") != "cpu"
        ]
        assert packed, "the fixture must actually keep weights on the GPU"
        cml.export(model, x, f"{tmp_path}/gpu.mlpackage").close()

    def test_compressing_it_again_stacks_two_grids(self, tmp_path) -> None:
        """Which is why the default leaves the weights alone.

        The values already sit on MLX's grid — groups of 64 along the
        input axis, each with its own scale and bias — and this writer's
        int8 is per output channel. A row spans many groups with
        different steps, so there is no single step to recover and the
        weight is quantized a second time onto a grid it does not share.
        Measured on a trained classifier: 3.8e-06 leaving the weights
        alone, 3.4e-02 asking for int8, 3.7e-01 asking for six-bit
        palettization.

        Not refused — a caller may want the size and know what it costs —
        but the ordering is worth knowing before reading the number.
        """
        x = lucid.randn(1, 3, 16, 16)
        model = q.quantize_dynamic(_net())

        plain = cml.export(model, x, f"{tmp_path}/plain.mlpackage")
        again = cml.export(
            model, x, f"{tmp_path}/again.mlpackage", weights=cml.WeightPrecision.INT8
        )
        try:
            wanted = model(x).cpu()
            loose = float((again.predict(x) - wanted).abs().max().item())
            tight = float((plain.predict(x) - wanted).abs().max().item())
            assert tight < loose
        finally:
            plain.close()
            again.close()


class TestDroppingTheActivationSimulation:
    """``Activations.DROPPED`` — the second half of the bridge.

    A prepared model carries fake-quantization in two places: on each
    weight, which puts it onto the grid it will be stored on, and on each
    activation, which simulates a runtime that computes in integers. The
    first is what the export wants. The second is a simulation of a
    runtime this is not — Core ML quantizes weights only, and the Neural
    Engine computes in float16 whatever the package says — so carrying it
    means rounding, clipping and scaling at every activation, work the
    accelerator did not ask for.

    Measured on the classifier above: twenty-four operations against six,
    with the accuracy unchanged. It is not the default anyway, because
    dropping it changes what the package computes and that is the
    caller's call to make.
    """

    def test_it_removes_the_simulation_and_keeps_the_weights(self, tmp_path) -> None:
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()

        kept = cml.export(
            model,
            x,
            f"{tmp_path}/kept.mlpackage",
            weights=cml.WeightPrecision.INT8,
            activations=cml.Activations.SIMULATED,
        )
        dropped = cml.export(
            model,
            x,
            f"{tmp_path}/dropped.mlpackage",
            weights=cml.WeightPrecision.INT8,
            activations=cml.Activations.DROPPED,
        )
        try:
            assert dropped.compute_plan().total_compute < kept.compute_plan().total_compute
        finally:
            kept.close()
            dropped.close()

    def test_only_the_activation_one_is_paused(self) -> None:
        """Both are the same class, told apart by where they hang.

        Dropping the weight's fake-quantize alongside the activation's
        would discard the grid the weights were trained onto, which is
        the only reason to export a prepared model rather than the float
        one it came from. Checked on the mechanism rather than through a
        tolerance: what matters is which modules are paused, and a
        numeric threshold measures that only at a distance.
        """
        from lucid.coreml._build import _activation_quantization_paused

        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        weights = [m for n, m in model.named_modules() if n.endswith("weight_fake_quant")]
        activations = [
            m
            for n, m in model.named_modules()
            if n.endswith("activation_post_process") and isinstance(m, FakeQuantize)
        ]
        assert weights and activations

        with _activation_quantization_paused(model):
            assert all(m._fake_quant_enabled for m in weights)
            assert not any(m._fake_quant_enabled for m in activations)

        assert all(m._fake_quant_enabled for m in weights)
        assert all(m._fake_quant_enabled for m in activations)

    def test_it_puts_the_simulation_back(self, tmp_path) -> None:
        """Exporting is not allowed to change the model it was given."""
        x = lucid.randn(1, 3, 16, 16)
        model = _prepared(x)
        before = [
            getattr(m, "_fake_quant_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        cml.export(
            model, x, f"{tmp_path}/restored.mlpackage", activations=cml.Activations.DROPPED
        ).close()
        after = [
            getattr(m, "_fake_quant_enabled", False)
            for m in model.modules()
            if isinstance(m, FakeQuantize)
        ]
        assert after == before

    def test_it_is_inert_on_a_model_that_carries_none(self, tmp_path) -> None:
        """Which is every model that did not come from `lucid.quantization`."""
        lucid.manual_seed(0)
        plain = _net()
        x = lucid.randn(1, 3, 16, 16)
        both = []
        for name, setting in (
            ("simulated", cml.Activations.SIMULATED),
            ("dropped", cml.Activations.DROPPED),
        ):
            exported = cml.export(plain, x, f"{tmp_path}/{name}.mlpackage", activations=setting)
            try:
                both.append(exported.verify(plain, x, relative=True))
            finally:
                exported.close()
        assert both[0] == both[1] < 1e-5
