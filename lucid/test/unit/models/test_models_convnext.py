"""Unit tests for ConvNeXt (Liu et al., 2022)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.convnext import (
    ConvNeXt,
    ConvNeXtConfig,
    ConvNeXtForImageClassification,
    convnext_tiny,
    convnext_tiny_cls,
    convnext_small_cls,
    convnext_base_cls,
)


class TestConvNeXtConfig(unittest.TestCase):

    def test_defaults_tiny(self) -> None:
        cfg = ConvNeXtConfig()
        self.assertEqual(cfg.model_type, "convnext")
        self.assertEqual(cfg.depths, (3, 3, 9, 3))
        self.assertEqual(cfg.dims, (96, 192, 384, 768))
        self.assertAlmostEqual(cfg.layer_scale_init, 1e-6)

    def test_tuple_coercion(self) -> None:
        import json
        import os

        cfg = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ConvNeXtConfig.from_dict(d)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.dims, tuple)
            self.assertEqual(cfg2.depths, (3, 3, 27, 3))
        finally:
            os.unlink(path)


class TestConvNeXtParamCounts(unittest.TestCase):

    def test_tiny_classifier(self) -> None:
        # Reference-exact: 28,589,128
        self.assertEqual(convnext_tiny_cls().num_parameters(), 28_589_128)

    def test_small_classifier(self) -> None:
        self.assertEqual(convnext_small_cls().num_parameters(), 50_223_688)

    def test_base_classifier(self) -> None:
        self.assertEqual(convnext_base_cls().num_parameters(), 88_591_464)

    def test_larger_has_more_params(self) -> None:
        self.assertGreater(
            convnext_small_cls().num_parameters(), convnext_tiny_cls().num_parameters()
        )
        self.assertGreater(
            convnext_base_cls().num_parameters(), convnext_small_cls().num_parameters()
        )


class TestConvNeXtBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = convnext_tiny()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        self.assertEqual(len(self.model.feature_info), 4)
        channels = [fi.num_channels for fi in self.model.feature_info]
        self.assertEqual(channels, [96, 192, 384, 768])

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        # (B, 768) after global avg pool
        self.assertEqual(feat.shape, (1, 768))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))


class TestConvNeXtClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = convnext_tiny_cls()
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
        m = ConvNeXtForImageClassification(ConvNeXtConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestConvNeXtRegistry(unittest.TestCase):

    def test_10_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="convnext")), 10)

    def test_all_sizes_present(self) -> None:
        names = models.list_models(family="convnext")
        for size in ["tiny", "small", "base", "large", "xlarge"]:
            self.assertIn(f"convnext_{size}", names)
            self.assertIn(f"convnext_{size}_cls", names)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("convnext_tiny")
        self.assertIsInstance(cfg, ConvNeXtConfig)
        self.assertEqual(cfg.depths, (3, 3, 9, 3))

    def test_create_model(self) -> None:
        m = models.create_model("convnext_tiny")
        self.assertIsInstance(m, ConvNeXt)


class TestConvNeXtSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = convnext_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ConvNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = convnext_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = ConvNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


class TestConvNeXtWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def test_default_aliases_imagenet1k_v1(self) -> None:
        from lucid.models.weights import (
            ConvNeXtBaseWeights,
            ConvNeXtLargeWeights,
            ConvNeXtSmallWeights,
            ConvNeXtTinyWeights,
        )

        for cls in (
            ConvNeXtTinyWeights,
            ConvNeXtSmallWeights,
            ConvNeXtBaseWeights,
            ConvNeXtLargeWeights,
        ):
            self.assertIs(cls.DEFAULT, cls.IMAGENET1K_V1)

    def test_xlarge_default_aliases_fb_in22k_ft_in1k(self) -> None:
        from lucid.models.weights import ConvNeXtXLargeWeights

        # XLarge uses the timm-sourced fb_in22k_ft_in1k tag (different
        # naming convention from the reference_vision IMAGENET1K_V1 four).
        self.assertIs(ConvNeXtXLargeWeights.DEFAULT, ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K)

    def test_xlarge_meta_provenance(self) -> None:
        from lucid.models.weights import ConvNeXtXLargeWeights

        meta = ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K.meta
        self.assertEqual(meta["source"], "timm/convnext_xlarge.fb_in22k_ft_in1k")
        self.assertEqual(meta["license"], "apache-2.0")
        self.assertEqual(meta["num_params"], 350_196_968)
        tf = ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K.transforms()
        # timm preset: 224 crop, 256 resize, bicubic (≠ reference_vision bilinear).
        self.assertEqual(tf.crop_size, 224)
        self.assertEqual(tf.resize_size, 256)

    def test_entry_fields(self) -> None:
        from lucid.models.weights import ConvNeXtTinyWeights

        e = ConvNeXtTinyWeights.IMAGENET1K_V1.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)
        self.assertIn("lucid-dl/convnext-tiny", e.url)

    def test_meta_provenance(self) -> None:
        from lucid.models.weights import ConvNeXtBaseWeights

        meta = ConvNeXtBaseWeights.IMAGENET1K_V1.meta
        self.assertEqual(meta["source"], "reference_vision/ConvNeXt_Base_Weights.IMAGENET1K_V1")
        self.assertEqual(meta["num_params"], 88_591_464)
        self.assertIn("ImageNet-1k", meta["metrics"])

    def test_transforms_match_reference_vision(self) -> None:
        # ConvNeXt resize sizes differ per variant (236 / 230 / 232 / 232).
        from lucid.models.weights import (
            ConvNeXtBaseWeights,
            ConvNeXtLargeWeights,
            ConvNeXtSmallWeights,
            ConvNeXtTinyWeights,
        )

        cases = [
            (ConvNeXtTinyWeights, 224, 236),
            (ConvNeXtSmallWeights, 224, 230),
            (ConvNeXtBaseWeights, 224, 232),
            (ConvNeXtLargeWeights, 224, 232),
        ]
        for cls, crop, resize in cases:
            tf = cls.IMAGENET1K_V1.transforms()
            self.assertEqual(tf.crop_size, crop)
            self.assertEqual(tf.resize_size, resize)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for name in (
            "convnext_tiny_cls",
            "convnext_small_cls",
            "convnext_base_cls",
            "convnext_large_cls",
        ):
            self.assertIn("IMAGENET1K_V1", list_pretrained(name))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestConvNeXtPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_tiny_default(self) -> None:
        from lucid.models import convnext_tiny_cls

        m = convnext_tiny_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_small_string_tag(self) -> None:
        from lucid.models import convnext_small_cls

        m = convnext_small_cls(pretrained="IMAGENET1K_V1")
        self.assertIsInstance(m, ConvNeXtForImageClassification)


if __name__ == "__main__":
    unittest.main()
