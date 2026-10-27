"""Unit tests for EfficientFormer (Li et al., 2022)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig,
    EfficientFormer,
    EfficientFormerForImageClassification,
    efficientformer_l1,
    efficientformer_l1_cls,
)


class TestEfficientFormerConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = EfficientFormerConfig()
        self.assertEqual(cfg.model_type, "efficientformer")


class TestEfficientFormerBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientformer_l1()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestEfficientFormerClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientformer_l1_cls()
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
        m = EfficientFormerForImageClassification(EfficientFormerConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestEfficientFormerRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="efficientformer")
        self.assertIn("efficientformer_l1", names)
        self.assertIn("efficientformer_l1_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("efficientformer_l1")
        self.assertIsInstance(m, EfficientFormer)


_SHIPPED = (
    (
        "efficientformer_l1_cls",
        "efficientformer-l1",
        "efficientformer_l1.snap_dist_in1k",
        12_289_928,
    ),
    (
        "efficientformer_l3_cls",
        "efficientformer-l3",
        "efficientformer_l3.snap_dist_in1k",
        31_406_000,
    ),
    (
        "efficientformer_l7_cls",
        "efficientformer-l7",
        "efficientformer_l7.snap_dist_in1k",
        82_229_328,
    ),
)


class TestEfficientFormerWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.weights import (
            EfficientFormerL1Weights,
            EfficientFormerL3Weights,
            EfficientFormerL7Weights,
        )

        return (EfficientFormerL1Weights, EfficientFormerL3Weights, EfficientFormerL7Weights)

    def test_default_aliases_snap_dist_in1k(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.SNAP_DIST_IN1K)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams) in zip(self._enums(), _SHIPPED):
            e = cls.SNAP_DIST_IN1K.entry
            self.assertEqual(e.num_classes, 1000)
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/SNAP_DIST_IN1K/", e.url)
            self.assertEqual(cls.SNAP_DIST_IN1K.meta["source"], f"timm/{src}")
            self.assertEqual(cls.SNAP_DIST_IN1K.meta["license"], "apache-2.0")
            self.assertEqual(cls.SNAP_DIST_IN1K.meta["num_params"], nparams)

    def test_transforms_bicubic_224(self) -> None:
        for cls in self._enums():
            tf = cls.SNAP_DIST_IN1K.transforms()
            self.assertEqual(tf.crop_size, 224)
            self.assertEqual(tf.resize_size, 235)  # timm truncates 224 / 0.95
            self.assertEqual(tf.interpolation, "bicubic")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("SNAP_DIST_IN1K", list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestEfficientFormerPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_l1_default(self) -> None:
        m = models.efficientformer_l1_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_l3_string_tag(self) -> None:
        m = models.efficientformer_l3_cls(pretrained="SNAP_DIST_IN1K")
        self.assertIsInstance(m, EfficientFormerForImageClassification)


if __name__ == "__main__":
    unittest.main()
