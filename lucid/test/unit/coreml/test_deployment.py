"""What a package has to say about itself before an app can use it.

Numbers agreeing is necessary and not sufficient. A package an app can
actually load has to declare what its input *is* — an image, not an
opaque array — and carry the description, author and licence that any
model catalogue asks for. Neither shows up in a value comparison, which
is why they get their own tests.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)

Model_pb2 = pytest.importorskip(
    "coremltools.proto.Model_pb2",
    reason="the description is read back with the reference parser",
)


def _description(package: str) -> object:
    model = Model_pb2.Model()
    with open(f"{package}/Data/com.apple.CoreML/model.mlmodel", "rb") as handle:
        model.ParseFromString(handle.read())
    return model.description


class TestImageInput:
    """An app holds a pixel buffer, not a multi-array.

    Without an image input the caller converts pixels themselves, and a
    conversion that is subtly wrong — a missed scale, the wrong channel
    order — produces a model that runs and answers badly.
    """

    def test_the_description_says_image(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        pixels = lucid.rand(1, 3, 64, 64) * 255.0
        exported = cml.export(
            model,
            pixels,
            str(tmp_path / "image.mlpackage"),
            image_input=cml.ImageInput(scale=1 / 255.0, bias=(-0.5, -0.5, -0.5)),
        )
        try:
            description = _description(str(tmp_path / "image.mlpackage"))
            feature = description.input[0]
            assert feature.type.WhichOneof("Type") == "imageType"
            assert feature.type.imageType.width == 64
            assert feature.type.imageType.height == 64
        finally:
            exported.close()

    def test_an_image_package_still_runs_and_agrees(self, tmp_path: object) -> None:
        """The normalisation moved into the package; verify follows it.

        Integral pixels on purpose: the buffer is eight bits per channel,
        so fractional input would be comparing the rounding rather than
        the network.
        """
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        pixels = (lucid.rand(1, 3, 64, 64) * 255.0).round()
        exported = cml.export(
            model,
            pixels,
            str(tmp_path / "run.mlpackage"),
            image_input=cml.ImageInput(scale=1 / 255.0, bias=(-0.5, -0.5, -0.5)),
        )
        try:
            assert exported.predict(pixels).shape == (1, 10)
            assert exported.verify(model, pixels) < 1e-5
        finally:
            exported.close()

    def test_a_channel_count_that_cannot_be_that_colour_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("unet").eval()
        with pytest.raises(ValueError, match="channel"):
            cml.export(
                model,
                lucid.rand(1, 1, 64, 64) * 255.0,
                str(tmp_path / "grey.mlpackage"),
                image_input=cml.ImageInput(color=cml.ColorSpace.RGB),
            )

    def test_more_than_one_input_is_refused(self, tmp_path: object) -> None:
        import lucid.nn as nn

        class TwoInputs(nn.Module):
            def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
                return a + b

        with pytest.raises(ValueError, match="single-input"):
            cml.export(
                TwoInputs().eval(),
                (lucid.rand(1, 3, 8, 8), lucid.rand(1, 3, 8, 8)),
                str(tmp_path / "two.mlpackage"),
                image_input=cml.ImageInput(),
            )


class TestMetadata:
    def test_what_is_stated_is_recorded_and_nothing_else(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 32, 32),
            str(tmp_path / "meta.mlpackage"),
            metadata=cml.Metadata(
                description="a demo", author="Lucid", license="MIT", version="1.0"
            ),
        )
        try:
            metadata = _description(str(tmp_path / "meta.mlpackage")).metadata
            assert metadata.shortDescription == "a demo"
            assert metadata.author == "Lucid"
            assert metadata.license == "MIT"
            assert metadata.versionString == "1.0"
        finally:
            exported.close()

    def test_an_unstated_field_is_left_out(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        exported = cml.export(
            model,
            lucid.randn(1, 3, 32, 32),
            str(tmp_path / "partial.mlpackage"),
            metadata=cml.Metadata(author="Lucid"),
        )
        try:
            metadata = _description(str(tmp_path / "partial.mlpackage")).metadata
            assert metadata.author == "Lucid"
            assert metadata.shortDescription == ""
            assert metadata.license == ""
        finally:
            exported.close()
