"""Unit tests for SKNet / SK-ResNet (Li et al., 2019)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.sknet import (
    SKNetConfig,
    SKNet,
    SKNetForImageClassification,
    sk_resnet_50,
    sk_resnet_50_cls,
    sk_resnet_101,
    sk_resnext_50_32x4d_cls,
)


class TestSKNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = SKNetConfig()
        self.assertEqual(cfg.model_type, "sknet")

    def test_json_round_trip(self) -> None:
        import json
        import os
        import tempfile

        cfg = SKNetConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = SKNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestSKNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = sk_resnet_50()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_101_more_params_than_50(self) -> None:
        self.assertGreater(sk_resnet_101().num_parameters(), sk_resnet_50().num_parameters())


class TestSKNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = sk_resnet_50_cls()
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
        m = SKNetForImageClassification(SKNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))

    def test_sk_resnext_variant(self) -> None:
        m = sk_resnext_50_32x4d_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 1000))


class TestSKNetRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        # 5 paper-cited variants (sk_resnet_18 / _34 / _50 / _101 /
        # sk_resnext_50_32x4d — Li 2019) × 2 (raw + ``_cls`` task wrapper)
        # = 10 registered names.
        self.assertEqual(len(models.list_models(family="sknet")), 10)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="sknet")
        for v in ["sk_resnet_50", "sk_resnet_101", "sk_resnext_50_32x4d"]:
            self.assertIn(v, names)
            self.assertIn(f"{v}_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("sk_resnet_50")
        self.assertIsInstance(m, SKNet)


if __name__ == "__main__":
    unittest.main()
