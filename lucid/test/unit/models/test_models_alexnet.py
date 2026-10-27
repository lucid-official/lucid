"""Unit tests for AlexNet (Krizhevsky 2014 single-stream OWT)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.alexnet import (
    AlexNetConfig,
    AlexNetForImageClassification,
    AlexNetWeights,
    alexnet,
    alexnet_cls,
)


class TestAlexNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = AlexNetConfig()
        self.assertEqual(cfg.model_type, "alexnet")
        self.assertEqual(cfg.num_classes, 1000)
        self.assertEqual(cfg.in_channels, 3)
        self.assertAlmostEqual(cfg.dropout, 0.5)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = AlexNetConfig(num_classes=100, dropout=0.3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = AlexNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 100)
            self.assertAlmostEqual(cfg2.dropout, 0.3)
        finally:
            os.unlink(path)


class TestAlexNetParamCounts(unittest.TestCase):
    """Krizhevsky 2014 single-stream OWT counts (64/192/384/256/256)."""

    def test_backbone_params(self) -> None:
        # Conv1(23296) + Conv2(307392) + Conv3(663936) + Conv4(884992) + Conv5(590080)
        self.assertEqual(alexnet().num_parameters(), 2_469_696)

    def test_classifier_params(self) -> None:
        # backbone + FC6(37752832) + FC7(16781312) + Out(4097000) = 61100840
        self.assertEqual(alexnet_cls().num_parameters(), 61_100_840)


class TestAlexNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = alexnet()
        self.model.eval()

    def test_feature_info_5_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 5)
        self.assertEqual([f.num_channels for f in fi], [64, 192, 384, 256, 256])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 256, 6, 6))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (2, 256, 6, 6))


class TestAlexNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = alexnet_cls()
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
        m = AlexNetForImageClassification(AlexNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 10))


class TestAlexNetRegistry(unittest.TestCase):

    def test_registered(self) -> None:
        names = models.list_models()
        self.assertIn("alexnet", names)
        self.assertIn("alexnet_cls", names)

    def test_family_filter(self) -> None:
        self.assertEqual(models.list_models(family="alexnet"), ["alexnet", "alexnet_cls"])

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("alexnet")
        self.assertIsInstance(cfg, AlexNetConfig)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("alexnet_cls")
        self.assertIsInstance(m, AlexNetForImageClassification)


class TestAlexNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = alexnet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = AlexNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = alexnet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = AlexNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


class TestAlexNetWeightsEnum(unittest.TestCase):
    """Static contract of :class:`AlexNetWeights` — no network required."""

    def test_default_aliases_imagenet1k_v1(self) -> None:
        self.assertIs(AlexNetWeights.DEFAULT, AlexNetWeights.IMAGENET1K_V1)

    def test_entry_fields(self) -> None:
        e = AlexNetWeights.IMAGENET1K_V1.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)  # hex-encoded SHA-256
        self.assertTrue(e.url.endswith("/model.safetensors"))
        self.assertIn("lucid-dl/alexnet", e.url)

    def test_meta_keys(self) -> None:
        meta = AlexNetWeights.IMAGENET1K_V1.meta
        self.assertEqual(meta["source"], "reference_vision/AlexNet_Weights.IMAGENET1K_V1")
        self.assertEqual(meta["num_params"], 61_100_840)
        self.assertIn("ImageNet-1k", meta["metrics"])

    def test_transforms_imagenet_preset(self) -> None:
        tf = AlexNetWeights.IMAGENET1K_V1.transforms()
        # ImageClassification preset with 224 crop / 256 resize / ImageNet stats
        self.assertEqual(tf.crop_size, 224)
        self.assertEqual(tf.resize_size, 256)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        self.assertIn("IMAGENET1K_V1", list_pretrained("alexnet_cls"))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestAlexNetWeightsLoad(unittest.TestCase):
    """End-to-end: download from Hub, SHA-verify, load into model."""

    def test_pretrained_true_loads(self) -> None:
        m = alexnet_cls(pretrained=True)
        m.eval()
        x = lucid.rand(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_pretrained_string_tag(self) -> None:
        m = alexnet_cls(pretrained="IMAGENET1K_V1")
        self.assertIsInstance(m, AlexNetForImageClassification)

    def test_explicit_enum(self) -> None:
        m = alexnet_cls(weights=AlexNetWeights.IMAGENET1K_V1)
        self.assertIsInstance(m, AlexNetForImageClassification)


if __name__ == "__main__":
    unittest.main()
