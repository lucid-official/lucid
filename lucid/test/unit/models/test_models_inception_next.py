"""Unit tests for InceptionNeXt (Yu et al., 2023)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.inception_next import (
    InceptionNeXtConfig,
    InceptionNeXt,
    InceptionNeXtForImageClassification,
    inception_next_tiny,
    inception_next_tiny_cls,
)


class TestInceptionNeXtConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = InceptionNeXtConfig()
        self.assertEqual(cfg.model_type, "inception_next")
        self.assertEqual(cfg.depths, (3, 3, 9, 3))
        self.assertEqual(cfg.dims, (96, 192, 384, 768))
        self.assertEqual(cfg.band_kernel, 11)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = InceptionNeXtConfig(num_classes=10, band_kernel=7)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = InceptionNeXtConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
            self.assertEqual(cfg2.band_kernel, 7)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.dims, tuple)
        finally:
            os.unlink(path)


class TestInceptionNeXtBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_next_tiny()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape, (1, 768))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))


class TestInceptionNeXtClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_next_tiny_cls()
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
        m = InceptionNeXtForImageClassification(InceptionNeXtConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestInceptionNeXtRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="inception_next")
        self.assertIn("inception_next_tiny", names)
        self.assertIn("inception_next_tiny_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("inception_next_tiny")
        self.assertIsInstance(m, InceptionNeXt)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("inception_next_tiny")
        self.assertIsInstance(cfg, InceptionNeXtConfig)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("inception_next_tiny_cls")
        self.assertIsInstance(m, InceptionNeXtForImageClassification)


class TestInceptionNeXtSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = inception_next_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = InceptionNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = inception_next_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = InceptionNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
