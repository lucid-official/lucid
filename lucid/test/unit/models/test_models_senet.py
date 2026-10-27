"""Unit tests for SENet / SE-ResNet (Hu et al., 2018)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.senet import (
    SENetConfig,
    SENet,
    SENetForImageClassification,
    se_resnet_18,
    se_resnet_18_cls,
    se_resnet_50,
    se_resnet_101,
    se_resnet_152,
)


class TestSENetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = SENetConfig()
        self.assertEqual(cfg.model_type, "senet")

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = SENetConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = SENetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestSENetParamCounts(unittest.TestCase):

    def test_larger_has_more_params(self) -> None:
        self.assertGreater(se_resnet_50().num_parameters(), se_resnet_18().num_parameters())
        self.assertGreater(se_resnet_101().num_parameters(), se_resnet_50().num_parameters())
        self.assertGreater(se_resnet_152().num_parameters(), se_resnet_101().num_parameters())


class TestSENetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = se_resnet_18()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestSENetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = se_resnet_18_cls()
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

    def test_custom_num_classes(self) -> None:
        m = SENetForImageClassification(SENetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestSENetRegistry(unittest.TestCase):

    def test_10_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="senet")), 10)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="senet")
        for depth in [18, 34, 50, 101, 152]:
            self.assertIn(f"se_resnet_{depth}", names)
            self.assertIn(f"se_resnet_{depth}_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("se_resnet_18")
        self.assertIsInstance(m, SENet)


class TestSENetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = se_resnet_18_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = SENetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
