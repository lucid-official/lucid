"""Unit tests for DenseNet (Huang et al., 2016)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.densenet import (
    DenseNet,
    DenseNetConfig,
    DenseNetForImageClassification,
    densenet_121,
    densenet_121_cls,
    densenet_169,
    densenet_201,
    densenet_264,
)


class TestDenseNetConfig(unittest.TestCase):

    def test_default_is_121(self) -> None:
        cfg = DenseNetConfig()
        self.assertEqual(cfg.model_type, "densenet")
        self.assertEqual(cfg.block_config, (6, 12, 24, 16))
        self.assertEqual(cfg.growth_rate, 32)

    def test_tuple_coercion(self) -> None:
        import json
        import os

        cfg = DenseNetConfig(block_config=(6, 12, 32, 32))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            self.assertIsInstance(d["block_config"], list)
            cfg2 = DenseNetConfig.from_dict(d)
            self.assertIsInstance(cfg2.block_config, tuple)
            self.assertEqual(cfg2.block_config, (6, 12, 32, 32))
        finally:
            os.unlink(path)


class TestDenseNetParamCounts(unittest.TestCase):

    def test_densenet121_backbone(self) -> None:
        self.assertEqual(densenet_121().num_parameters(), 6_953_856)

    def test_densenet121_classifier(self) -> None:
        # Paper-exact: 7,978,856
        self.assertEqual(densenet_121_cls().num_parameters(), 7_978_856)

    def test_densenet169_backbone(self) -> None:
        self.assertEqual(densenet_169().num_parameters(), 12_484_480)

    def test_densenet201_backbone(self) -> None:
        self.assertEqual(densenet_201().num_parameters(), 18_092_928)

    def test_densenet264_has_more_params(self) -> None:
        self.assertGreater(densenet_264().num_parameters(), densenet_201().num_parameters())


class TestDenseNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = densenet_121()
        self.model.eval()

    def test_feature_info_4_blocks(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        # DenseNet-121 ends with 1024 channels
        self.assertEqual(out.shape, (1, 1024, 1, 1))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_densenet169_more_channels(self) -> None:
        m = densenet_169()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m.forward_features(x)
        # DenseNet-169 ends with 1664 channels
        self.assertEqual(out.shape[1], 1664)


class TestDenseNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = densenet_121_cls()
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

    def test_custom_classes_and_dropout(self) -> None:
        cfg = DenseNetConfig(block_config=(6, 12, 24, 16), num_classes=10, dropout_rate=0.2)
        m = DenseNetForImageClassification(cfg)
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 10))


class TestDenseNetRegistry(unittest.TestCase):

    def test_all_variants_registered(self) -> None:
        # 5 sizes × {backbone, classifier} = 10 (121/161/169/201/264).
        self.assertEqual(len(models.list_models(family="densenet")), 10)
        for n in ("densenet_161", "densenet_161_cls"):
            self.assertIn(n, models.list_models(family="densenet"))

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("densenet_121")
        self.assertIsInstance(cfg, DenseNetConfig)
        self.assertEqual(cfg.block_config, (6, 12, 24, 16))

    def test_create_model(self) -> None:
        m = models.create_model("densenet_121")
        self.assertIsInstance(m, DenseNet)


class TestDenseNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = densenet_121_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = DenseNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = densenet_121_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = DenseNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


class TestDenseNet161(unittest.TestCase):
    """DenseNet-161 — the wide k=48 / 96-stem variant added in the sweep."""

    def test_param_count(self) -> None:
        from lucid.models import densenet_161_cls

        n = densenet_161_cls().num_parameters() / 1e6
        self.assertLess(abs(n - 28.68) / 28.68, 0.01)

    def test_forward(self) -> None:
        from lucid.models import densenet_161_cls

        m = densenet_161_cls()
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_backbone_last_channels_2208(self) -> None:
        from lucid.models import densenet_161

        m = densenet_161()
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.last_hidden_state.shape[1], 2208)


class TestDenseNetWeightsEnums(unittest.TestCase):
    """Static contract of every per-variant ``<X>Weights`` enum."""

    def test_default_aliases_imagenet1k_v1(self) -> None:
        from lucid.models.weights import (
            DenseNet121Weights,
            DenseNet161Weights,
            DenseNet169Weights,
            DenseNet201Weights,
        )

        for cls in (
            DenseNet121Weights,
            DenseNet161Weights,
            DenseNet169Weights,
            DenseNet201Weights,
        ):
            self.assertIs(cls.DEFAULT, cls.IMAGENET1K_V1)

    def test_entry_fields(self) -> None:
        from lucid.models.weights import DenseNet161Weights

        e = DenseNet161Weights.IMAGENET1K_V1.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)
        self.assertIn("lucid-dl/densenet-161", e.url)

    def test_meta_provenance(self) -> None:
        from lucid.models.weights import DenseNet201Weights

        meta = DenseNet201Weights.IMAGENET1K_V1.meta
        self.assertEqual(meta["source"], "reference_vision/DenseNet201_Weights.IMAGENET1K_V1")
        self.assertEqual(meta["num_params"], 20_013_928)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for name in (
            "densenet_121_cls",
            "densenet_161_cls",
            "densenet_169_cls",
            "densenet_201_cls",
        ):
            self.assertIn("IMAGENET1K_V1", list_pretrained(name))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestDenseNetPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA verify + load + forward."""

    def test_densenet_121_default(self) -> None:
        from lucid.models import densenet_121_cls

        m = densenet_121_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
