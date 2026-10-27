"""Unit tests for ResNeXt (Xie et al., 2017)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.resnext import (
    ResNeXtConfig,
    ResNeXt,
    ResNeXtForImageClassification,
    resnext_50_32x4d,
    resnext_50_32x4d_cls,
    resnext_101_32x4d,
)


class TestResNeXtConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = ResNeXtConfig()
        self.assertEqual(cfg.model_type, "resnext")

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = ResNeXtConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ResNeXtConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestResNeXtBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnext_50_32x4d()
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

    def test_101_larger_than_50(self) -> None:
        p50 = resnext_50_32x4d().num_parameters()
        p101 = resnext_101_32x4d().num_parameters()
        self.assertGreater(p101, p50)


class TestResNeXtClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnext_50_32x4d_cls()
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
        m = ResNeXtForImageClassification(ResNeXtConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestResNeXtRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        # 4 topologies x 2 (raw + ``_cls``).  64x4d is the paper's headline
        # Table 4 model; 32x8d comes from the reference zoo, not the paper.
        self.assertEqual(len(models.list_models(family="resnext")), 8)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="resnext")
        for v in [
            "resnext_50_32x4d",
            "resnext_101_32x4d",
            "resnext_101_32x8d",
            "resnext_101_64x4d",
        ]:
            self.assertIn(v, names)
            self.assertIn(f"{v}_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("resnext_50_32x4d")
        self.assertIsInstance(m, ResNeXt)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("resnext_50_32x4d_cls")
        self.assertIsInstance(m, ResNeXtForImageClassification)


class TestResNeXtSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = resnext_50_32x4d_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ResNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
