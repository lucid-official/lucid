"""One package for several input shapes.

A fixed-shape package needs one file per batch size and one per
resolution. Making it flexible means the program stops fixing the axes
that vary — and finding out *which* axes those are is done by tracing the
model at each shape and comparing, not by propagating symbols. The tracer
already knows every value's shape; asking it twice cannot disagree with
itself about how an operation behaves.

The interesting case is the one that cannot be made flexible, and it is
common: an adaptive pool records a kernel derived from the input size, so
the same model traced at two resolutions is two different models. That is
refused by name rather than fixed to one shape and wrong at the others.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine
from lucid.test.unit.coreml._helpers import under_hypervisor

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class TestFlexibleBatch:
    def test_every_enumerated_batch_size_runs_and_agrees(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        sizes = (1, 2, 4)
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "batch.mlpackage"),
            shapes=[(n, 3, 224, 224) for n in sizes],
        )
        try:
            for n in sizes:
                x = lucid.randn(n, 3, 224, 224)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_a_shape_that_was_not_enumerated_is_refused(self, tmp_path: object) -> None:
        """Enumerated means enumerated; Core ML does not interpolate."""
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "gaps.mlpackage"),
            shapes=[(1, 3, 224, 224), (4, 3, 224, 224)],
        )
        try:
            with pytest.raises(RuntimeError, match="not in"):
                exported.predict(lucid.randn(3, 3, 224, 224))
        finally:
            exported.close()

    def test_flexibility_does_not_cost_the_neural_engine(self, tmp_path: object) -> None:
        """Worth asserting: a flexible package that fell back to the CPU
        would still be correct, and would quietly lose the reason this
        subsystem exists."""
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        fixed = cml.export(
            model,
            x,
            str(tmp_path / "fixed.mlpackage"),
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        flexible = cml.export(
            model,
            x,
            str(tmp_path / "flex.mlpackage"),
            precision=cml.Precision.FLOAT16,
            shapes=[(1, 3, 224, 224), (2, 3, 224, 224)],
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            one, many = fixed.compute_plan(), flexible.compute_plan()
            if one.total_compute and many.total_compute:
                assert many.ane_fraction >= one.ane_fraction - 0.05
        finally:
            fixed.close()
            flexible.close()


class TestFlexibleResolution:
    def test_a_fully_convolutional_model_takes_both(self, tmp_path: object) -> None:
        if under_hypervisor():
            # The same trap as the YOLOv3 test in test_coreml.py: inside the
            # hosted runner's virtual machine Core ML runs every package on
            # BNNS's CPU path, and BNNS dies here at prediction with SIGTRAP
            # (exit 133) and nothing printed. It passes on hardware.
            pytest.skip("Core ML traps running an enumerated-shape package in a VM")
        model = M.create_model("unet").eval()
        exported = cml.export(
            model,
            lucid.randn(1, 1, 64, 64),
            str(tmp_path / "unet.mlpackage"),
            shapes=[(1, 1, 64, 64), (1, 1, 128, 128)],
        )
        try:
            for side in (64, 128):
                x = lucid.randn(1, 1, side, side)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_an_adaptive_pool_names_itself(self, tmp_path: object) -> None:
        """ResNet's head pools to 1x1 with a kernel taken from the input.

        Traced at 224 the kernel is 7; at 256 it is 8. Fixing one and
        accepting both shapes would give a package that is wrong at the
        other, with nothing about it looking wrong.
        """
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(cml.ShapeNotFlexible) as excinfo:
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "adaptive.mlpackage"),
                shapes=[(1, 3, 224, 224), (1, 3, 256, 256)],
            )
        assert excinfo.value.op_name == "avg_pool2d"
        assert "kernel_size" in str(excinfo.value)


class TestFlexibleRefusals:
    def test_a_single_shape_is_not_flexible(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="at least one shape besides"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "one.mlpackage"),
                shapes=[(1, 3, 224, 224)],
            )

    def test_more_than_one_input_is_refused(self, tmp_path: object) -> None:
        import lucid.nn as nn

        class TwoInputs(nn.Module):
            def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
                return a + b

        with pytest.raises(ValueError, match="single-input"):
            cml.export(
                TwoInputs().eval(),
                (lucid.randn(1, 4), lucid.randn(1, 4)),
                str(tmp_path / "two.mlpackage"),
                shapes=[(1, 4), (2, 4)],
            )


class TestShapeRange:
    """A range admits everything between, not a listed few.

    Which is the point: a variable sequence length or a camera whose
    resolution changes has no short list to enumerate.
    """

    def test_sizes_that_were_never_traced_still_run(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "range.mlpackage"),
            shape_range={0: (1, 8)},
        )
        try:
            # 3 and 5 are inside the range and were never traced; an
            # enumerated export would refuse them.
            for n in (1, 3, 5, 8):
                x = lucid.randn(n, 3, 224, 224)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_a_size_outside_the_range_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 224, 224),
            str(tmp_path / "bounds.mlpackage"),
            shape_range={0: (1, 4)},
        )
        try:
            with pytest.raises(RuntimeError, match="not in allowed range"):
                exported.predict(lucid.randn(9, 3, 224, 224))
        finally:
            exported.close()

    def test_a_transposed_convolution_infers_its_own_output_size(self, tmp_path: object) -> None:
        """U-Net's decoder is the case that made this necessary.

        A transposed convolution's `output_shape` disambiguates a result
        that several inputs share. Baked from the trace, it fixes the
        decoder to one resolution while the encoder follows the input, so
        the skip connections stop lining up — which the Metal compiler
        reports as a concat of mismatched tensors and an abort, nowhere
        near the operation that caused it.
        """
        model = M.create_model("unet").eval()
        exported = cml.export(
            model,
            lucid.randn(1, 1, 64, 64),
            str(tmp_path / "decoder.mlpackage"),
            shape_range={2: (32, 128), 3: (32, 128)},
        )
        try:
            for side in (32, 64, 96, 128):
                x = lucid.randn(1, 1, side, side)
                reference = model(x).logits
                got = exported.predict(x)
                assert got.shape == reference.shape
                scale = float(reference.abs().max().item())
                assert float((got - reference).abs().max().item()) / scale < 1e-5
        finally:
            exported.close()

    def test_a_range_keeps_the_neural_engine(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        fixed = cml.export(
            model,
            x,
            str(tmp_path / "f.mlpackage"),
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        ranged = cml.export(
            model,
            x,
            str(tmp_path / "r.mlpackage"),
            precision=cml.Precision.FLOAT16,
            shape_range={0: (1, 8)},
            compute_units=cml.ComputeUnits.CPU_AND_NE,
        )
        try:
            one, many = fixed.compute_plan(), ranged.compute_plan()
            if one.total_compute and many.total_compute:
                assert many.ane_fraction >= one.ane_fraction - 0.05
        finally:
            fixed.close()
            ranged.close()


class TestShapeRangeRefusals:
    def test_naming_both_kinds_of_flexibility_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="Give one"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "both.mlpackage"),
                shapes=[(1, 3, 224, 224), (2, 3, 224, 224)],
                shape_range={0: (1, 8)},
            )

    def test_an_example_outside_its_own_range_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="outside the range"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "outside.mlpackage"),
                shape_range={0: (4, 8)},
            )

    def test_an_axis_beyond_the_rank_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="outside the input's rank"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                str(tmp_path / "rank.mlpackage"),
                shape_range={7: (1, 8)},
            )
