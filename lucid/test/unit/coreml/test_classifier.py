"""Labels, and what Core ML does and does not do with them.

A package that returns a score array makes the app do its own argmax and
label lookup; Vision does not even get that far, since it reads the
package's ``predictedFeatureName`` and an unset one means it returns
nothing. Declaring a classifier is what closes that.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

LABELS = tuple(f"class_{index}" for index in range(10))


def _classifier(tmp_path: object, name: str) -> tuple[object, object, lucid.Tensor]:
    model = M.create_model("resnet_18_cls", num_classes=len(LABELS)).eval()
    x = lucid.randn(1, 3, 224, 224)
    exported = cml.export(
        model, x, f"{tmp_path}/{name}.mlpackage", classifier=cml.Classifier(labels=LABELS)
    )
    return model, exported, x


class TestClassifierOutputs:
    def test_the_label_and_the_map_replace_the_scores(self, tmp_path: object) -> None:
        model, exported, x = _classifier(tmp_path, "labels")
        try:
            assert exported.output_names == ["classLabel", "classLabel_probs"]
            label, probabilities = exported.classify(x)

            scores = model(x).logits.reshape(-1).tolist()
            best = max(range(len(LABELS)), key=lambda i: scores[i])
            assert label == LABELS[best]
            assert set(probabilities) == set(LABELS)
            for index, name in enumerate(LABELS):
                assert abs(probabilities[name] - scores[index]) < 1e-5
        finally:
            exported.close()

    def test_the_probabilities_are_not_normalised(self, tmp_path: object) -> None:
        """Core ML passes the scores through, whatever the name says.

        A network ending in a linear layer hands `classify` raw scores,
        and they arrive unchanged under a feature called probabilities.
        Asserting they do not sum to one keeps that documented behaviour
        from being mistaken for a softmax the model never had.
        """
        model, exported, x = _classifier(tmp_path, "raw")
        try:
            _label, probabilities = exported.classify(x)
            scores = model(x).logits.reshape(-1).tolist()
            assert abs(sum(probabilities.values()) - sum(scores)) < 1e-5
        finally:
            exported.close()

    def test_predict_on_a_classifier_says_to_use_classify(self, tmp_path: object) -> None:
        _model, exported, x = _classifier(tmp_path, "wrongcall")
        try:
            with pytest.raises(TypeError, match="classify"):
                exported.predict(x)
        finally:
            exported.close()

    def test_classify_on_a_plain_package_says_so(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        x = lucid.randn(1, 3, 224, 224)
        exported = cml.export(model, x, f"{tmp_path}/plain.mlpackage")
        try:
            with pytest.raises(TypeError, match="scores, not labels"):
                exported.classify(x)
        finally:
            exported.close()


class TestClassifierRefusals:
    def test_a_label_count_that_does_not_match_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("resnet_18_cls", num_classes=10).eval()
        with pytest.raises(ValueError, match="labels for"):
            cml.export(
                model,
                lucid.randn(1, 3, 224, 224),
                f"{tmp_path}/count.mlpackage",
                classifier=cml.Classifier(labels=("a", "b")),
            )

    def test_a_model_with_several_outputs_is_refused(self, tmp_path: object) -> None:
        model = M.create_model("yolo_v3").eval()
        with pytest.raises(ValueError, match="one output"):
            cml.export(
                model,
                lucid.randn(1, 3, 416, 416),
                f"{tmp_path}/many.mlpackage",
                classifier=cml.Classifier(labels=LABELS),
            )
