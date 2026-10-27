"""Unit tests for Inception v3, Inception v4, and Inception-ResNet v2."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.inception import (
    InceptionConfig,
    InceptionV3,
    InceptionV3ForImageClassification,
    inception_v3,
    inception_v3_cls,
)
from lucid.models.vision.inception_resnet import (
    InceptionResNetConfig,
    InceptionResNetV2,
    InceptionResNetV2ForImageClassification,
    inception_resnet_v2,
    inception_resnet_v2_cls,
)

# ---------------------------------------------------------------------------
# Inception v3
# ---------------------------------------------------------------------------


class TestInceptionV3Config(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = InceptionConfig()
        self.assertEqual(cfg.model_type, "inception_v3")

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = InceptionConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = InceptionConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestInceptionV3Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_v3()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 299, 299)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestInceptionV3Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_v3_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(2, 3, 299, 299)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        self.assertIsNone(self.model(x).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(2, 3, 299, 299)
        labels = lucid.tensor([0, 999])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())


class TestInceptionV3Registry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="inception")
        self.assertIn("inception_v3", names)
        self.assertIn("inception_v3_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("inception_v3")
        self.assertIsInstance(m, InceptionV3)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("inception_v3_cls")
        self.assertIsInstance(m, InceptionV3ForImageClassification)


# ---------------------------------------------------------------------------
# Inception v4
# ---------------------------------------------------------------------------


class TestInceptionResNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_resnet_v2()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)


class TestInceptionResNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_resnet_v2_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_custom_num_classes(self) -> None:
        m = InceptionResNetV2ForImageClassification(InceptionResNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 299, 299)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestInceptionResNetRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="inception_resnet")
        self.assertIn("inception_resnet_v2", names)
        self.assertIn("inception_resnet_v2_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("inception_resnet_v2")
        self.assertIsInstance(m, InceptionResNetV2)


if __name__ == "__main__":
    unittest.main()
