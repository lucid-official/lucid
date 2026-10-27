"""A package reopened must be the package that was written.

The handle ``export`` returns knows things the export decided: that an
input is a picture and not an array, that the outputs are labels and not
scores, which system the package needs. ``load`` used to know none of
it, so the same file behaved differently depending on how it was
opened — and in a deployment, reopening is the ordinary path. You export
once and load forever after.

Both failures were real. A reloaded image package refused every
prediction, because Core ML will not take a multi-array where the model
declares a picture. A reloaded classifier refused ``classify``, saying
the package returns scores — which it does not.

The package declares all of it, so it is read back from the model rather
than asked for again. One thing cannot be: the pixel normalisation is
written into the *program*, as an ordinary multiply and add at the head,
not into anything the file declares. A reloaded handle can predict
without it and cannot compare without it, and the last test here is that
the difference is stated rather than guessed at.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(16, 5)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.body(x).mean(dim=(2, 3)))


def _pixels() -> lucid.Tensor:
    """Whole numbers in [0, 255], which is what an image input takes."""
    return (lucid.rand(1, 3, 32, 32) * 255).round()


class TestAnImagePackageSurvivesReloading:
    def test_it_predicts_at_all(self, tmp_path: object) -> None:
        """It did not. Core ML refuses an array where it declared a picture."""
        lucid.manual_seed(0)
        model = _Small().eval()
        pixels = _pixels()
        path = f"{tmp_path}/image.mlpackage"

        exported = cml.export(model, pixels, path, image_input=cml.ImageInput(scale=1 / 255.0))
        wanted = exported.predict(pixels)
        exported.close()

        reopened = cml.load(path)
        try:
            got = reopened.predict(pixels)
            assert float((got - wanted).abs().max().item()) == 0.0
        finally:
            reopened.close()

    def test_comparing_refuses_rather_than_guessing(self, tmp_path: object) -> None:
        """The normalisation is in the program, not in the description.

        A reopened handle knows the input is a picture and not what was
        done to it — so a comparison against the eager model would
        measure the missing scale and bias rather than the export. It
        says so instead, and names the way out.
        """
        lucid.manual_seed(0)
        model = _Small().eval()
        pixels = _pixels()
        path = f"{tmp_path}/unknown_scale.mlpackage"

        exported = cml.export(model, pixels, path, image_input=cml.ImageInput(scale=1 / 255.0))
        # The handle that wrote it can compare, because it knows.
        assert exported.verify(model, pixels, relative=True) < 1e-5
        exported.close()

        reopened = cml.load(path)
        try:
            with pytest.raises(ValueError, match="normalisation"):
                reopened.verify(model, pixels)
        finally:
            reopened.close()


class TestAClassifierPackageSurvivesReloading:
    def test_it_answers_with_a_label(self, tmp_path: object) -> None:
        """It refused, claiming the package returns scores."""
        lucid.manual_seed(0)
        model = _Small().eval()
        x = lucid.randn(1, 3, 32, 32)
        path = f"{tmp_path}/classifier.mlpackage"

        exported = cml.export(
            model, x, path, classifier=cml.Classifier(labels=[f"c{i}" for i in range(5)])
        )
        wanted_label, wanted_scores = exported.classify(x)
        exported.close()

        reopened = cml.load(path)
        try:
            label, scores = reopened.classify(x)
            assert label == wanted_label
            assert set(scores) == set(wanted_scores)
        finally:
            reopened.close()


class TestAFailedExportLeavesTheOldPackageAlone:
    """Writing in place meant clearing first, and clearing is destructive.

    Re-exporting a deployed model and hitting an unsupported operation
    used to leave the caller with neither the package they had nor the
    one they asked for: the destination was emptied before the build
    that then failed. For a file somebody is shipping, that is the worst
    thing an export can do — worse than refusing, which is all it was
    trying to do.

    The build happens beside the destination now and is moved onto it
    only once it is whole.
    """

    def test_the_deployed_package_survives(self, tmp_path: object) -> None:
        class _Samples(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + lucid.randn(1, 3)

        lucid.manual_seed(0)
        path = f"{tmp_path}/deployed.mlpackage"
        exported = cml.export(_Small().eval(), lucid.randn(1, 3, 32, 32), path)
        wanted = exported.predict(lucid.ones(1, 3, 32, 32))
        exported.close()

        with pytest.raises(Exception):
            cml.export(_Samples().eval(), lucid.zeros(1, 3), path)

        reopened = cml.load(path)
        try:
            got = reopened.predict(lucid.ones(1, 3, 32, 32))
            assert float((got - wanted).abs().max().item()) == 0.0
        finally:
            reopened.close()

    def test_it_leaves_no_half_built_package_behind(self, tmp_path: object) -> None:
        """Not at the destination, and not beside it either.

        A leftover staging directory is a smaller problem than a
        destroyed package and still a problem: it is confusing, and it
        is the size of a model.
        """
        import os

        class _Samples(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + lucid.randn(1, 3)

        lucid.manual_seed(0)
        with pytest.raises(Exception):
            cml.export(_Samples().eval(), lucid.zeros(1, 3), f"{tmp_path}/never.mlpackage")
        assert os.listdir(str(tmp_path)) == []


class TestAnOrdinaryPackageIsUntouched:
    """The recovery reads the model, so a package with neither gets neither.

    A plain export must not come back claiming to be an image model or a
    classifier — the check is as narrow as the defect.
    """

    def test_no_image_and_no_labels(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        model = _Small().eval()
        x = lucid.randn(1, 3, 32, 32)
        path = f"{tmp_path}/plain.mlpackage"

        exported = cml.export(model, x, path)
        wanted = exported.predict(x)
        exported.close()

        reopened = cml.load(path)
        try:
            assert reopened.image_input is None
            assert reopened.classifier is None
            got = reopened.predict(x)
            assert float((got - wanted).abs().max().item()) == 0.0
        finally:
            reopened.close()
