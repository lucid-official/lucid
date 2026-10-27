"""Unit tests for GoogLeNet / Inception v1 (Szegedy et al., 2014)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.googlenet import (
    GoogLeNetConfig,
    GoogLeNetForImageClassification,
    GoogLeNetOutput,
    googlenet,
    googlenet_cls,
)


class TestGoogLeNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = GoogLeNetConfig()
        self.assertEqual(cfg.model_type, "googlenet")
        self.assertEqual(cfg.num_classes, 1000)
        self.assertTrue(cfg.aux_logits)
        self.assertAlmostEqual(cfg.dropout, 0.4)
        self.assertAlmostEqual(cfg.aux_dropout, 0.7)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = GoogLeNetConfig(num_classes=100, aux_logits=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = GoogLeNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 100)
            self.assertFalse(cfg2.aux_logits)
        finally:
            os.unlink(path)


class TestGoogLeNetParamCounts(unittest.TestCase):

    def test_backbone_params(self) -> None:
        # Batch-normalised reference topology (Conv→BN→ReLU primitives).
        self.assertEqual(googlenet().num_parameters(), 5_599_904)

    def test_classifier_params(self) -> None:
        # Reference checkpoint: 13,004,888 (backbone + head + 2 aux heads).
        self.assertEqual(googlenet_cls().num_parameters(), 13_004_888)

    def test_no_aux_fewer_params(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(aux_logits=False))
        # Without aux: backbone + head only (the reference 6.6M count).
        self.assertEqual(m.num_parameters(), 6_624_904)
        self.assertLess(m.num_parameters(), 13_004_888)


class TestGoogLeNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = googlenet()
        self.model.eval()

    def test_feature_info_3_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 3)
        self.assertEqual([f.num_channels for f in fi], [480, 832, 1024])
        self.assertEqual([f.reduction for f in fi], [8, 16, 32])

    def test_forward_features_matches_the_declared_reduction(self) -> None:
        """``feature_info``'s last entry must describe what is returned.

        The backbone used to end on ``AdaptiveAvgPool2d((1, 1))``, so it
        returned a 1x1 map while declaring reduction=32 — and the earlier
        stages it advertises were computed and discarded, unreachable by any
        pyramid consumer.
        """
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        last = self.model.feature_info[-1]
        self.assertEqual(out.shape, (1, last.num_channels, 7, 7))
        self.assertEqual(224 // int(out.shape[2]), last.reduction)

    def test_forward_returns_base_model_output(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (2, 1024, 7, 7))


class TestGoogLeNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = googlenet_cls()

    def test_eval_no_aux(self) -> None:
        self.model.eval()
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))
        self.assertIsInstance(out, ImageClassificationOutput)
        self.assertIsInstance(out, GoogLeNetOutput)
        self.assertIsNone(out.aux_logits1)
        self.assertIsNone(out.aux_logits2)
        self.assertIsNone(out.loss)

    def test_train_aux_fires(self) -> None:
        self.model.train()
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsNotNone(out.aux_logits1)
        self.assertIsNotNone(out.aux_logits2)
        self.assertEqual(out.aux_logits1.shape, (2, 1000))  # type: ignore[union-attr]
        self.assertEqual(out.aux_logits2.shape, (2, 1000))  # type: ignore[union-attr]

    def test_train_loss_includes_aux(self) -> None:
        self.model.train()
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 1])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())  # type: ignore[union-attr]

    def test_eval_loss_no_aux_penalty(self) -> None:
        self.model.eval()
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 1])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)

    def test_no_aux_logits_config(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(aux_logits=False))
        m.train()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertIsNone(out.aux_logits1)
        self.assertIsNone(out.aux_logits2)

    def test_custom_num_classes(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 10))


class TestGoogLeNetRegistry(unittest.TestCase):

    def test_registered(self) -> None:
        names = models.list_models()
        self.assertIn("googlenet", names)
        self.assertIn("googlenet_cls", names)

    def test_family_filter(self) -> None:
        g = models.list_models(family="googlenet")
        self.assertEqual(sorted(g), ["googlenet", "googlenet_cls"])

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("googlenet")
        self.assertIsInstance(cfg, GoogLeNetConfig)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("googlenet_cls")
        self.assertIsInstance(m, GoogLeNetForImageClassification)


class TestGoogLeNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = googlenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = GoogLeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = googlenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = GoogLeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


class TestGoogLeNetWeightsEnums(unittest.TestCase):
    """Static contract of the ``GoogLeNetWeights`` enum."""

    def test_default_aliases_imagenet1k_v1(self) -> None:
        from lucid.models.weights import GoogLeNetWeights

        self.assertIs(GoogLeNetWeights.DEFAULT, GoogLeNetWeights.IMAGENET1K_V1)

    def test_entry_fields(self) -> None:
        from lucid.models.weights import GoogLeNetWeights

        e = GoogLeNetWeights.IMAGENET1K_V1.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertIn("lucid-dl/googlenet", e.url)
        self.assertIn("IMAGENET1K_V1", e.url)

    def test_meta_provenance(self) -> None:
        from lucid.models.weights import GoogLeNetWeights

        meta = GoogLeNetWeights.IMAGENET1K_V1.meta
        self.assertEqual(meta["source"], "reference_vision/GoogLeNet_Weights.IMAGENET1K_V1")
        self.assertEqual(meta["num_params"], 13_004_888)
        self.assertEqual(meta["license"], "bsd-3-clause")
        self.assertAlmostEqual(meta["metrics"]["ImageNet-1k"]["acc@1"], 69.778)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        self.assertIn("IMAGENET1K_V1", list_pretrained("googlenet_cls"))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestGoogLeNetPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA verify + load + forward."""

    def test_googlenet_default(self) -> None:
        from lucid.models import googlenet_cls

        m = googlenet_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
