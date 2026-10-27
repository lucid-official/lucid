"""Unit tests for MobileNet v3 (Howard et al., 2019)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.mobilenet_v3 import (
    MobileNetV3Config,
    MobileNetV3,
    MobileNetV3ForImageClassification,
    mobilenet_v3_large,
    mobilenet_v3_large_cls,
    mobilenet_v3_small,
    mobilenet_v3_small_cls,
)


class TestMobileNetV3Config(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MobileNetV3Config()
        self.assertEqual(cfg.model_type, "mobilenet_v3")


class TestMobileNetV3Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v3_large()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_small_has_fewer_params(self) -> None:
        self.assertGreater(
            mobilenet_v3_large().num_parameters(), mobilenet_v3_small().num_parameters()
        )


class TestMobileNetV3Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v3_large_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        self.assertIsNone(self.model(x).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 999])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_small_variant(self) -> None:
        m = mobilenet_v3_small_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 1000))

    def test_custom_num_classes(self) -> None:
        m = MobileNetV3ForImageClassification(MobileNetV3Config(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMobileNetV3Registry(unittest.TestCase):

    def test_4_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="mobilenet_v3")), 4)

    def test_variants_present(self) -> None:
        names = models.list_models(family="mobilenet_v3")
        self.assertIn("mobilenet_v3_large", names)
        self.assertIn("mobilenet_v3_large_cls", names)
        self.assertIn("mobilenet_v3_small", names)
        self.assertIn("mobilenet_v3_small_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("mobilenet_v3_large")
        self.assertIsInstance(m, MobileNetV3)


if __name__ == "__main__":
    unittest.main()
