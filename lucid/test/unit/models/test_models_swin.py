"""Unit tests for Swin Transformer (Liu et al., 2021)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.swin import (
    SwinConfig,
    SwinTransformer,
    SwinTransformerForImageClassification,
    swin_tiny,
    swin_tiny_cls,
    swin_small_cls,
    swin_base_cls,
)


class TestSwinConfig(unittest.TestCase):

    def test_defaults_swin_tiny(self) -> None:
        cfg = SwinConfig()
        self.assertEqual(cfg.model_type, "swin")
        self.assertEqual(cfg.embed_dim, 96)
        self.assertEqual(cfg.depths, (2, 2, 6, 2))
        self.assertEqual(cfg.num_heads, (3, 6, 12, 24))
        self.assertEqual(cfg.window_size, 7)

    def test_tuple_coercion(self) -> None:
        import json
        import os

        cfg = SwinConfig(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            self.assertIsInstance(d["depths"], list)
            cfg2 = SwinConfig.from_dict(d)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.num_heads, tuple)
            self.assertEqual(cfg2.depths, (2, 2, 18, 2))
        finally:
            os.unlink(path)


class TestSwinParamCounts(unittest.TestCase):

    def test_swin_tiny_classifier(self) -> None:
        # Reference-exact: 28,288,354
        self.assertEqual(swin_tiny_cls().num_parameters(), 28_288_354)

    def test_swin_small_classifier(self) -> None:
        self.assertEqual(swin_small_cls().num_parameters(), 49_606_258)

    def test_swin_base_classifier(self) -> None:
        self.assertEqual(swin_base_cls().num_parameters(), 87_768_224)

    def test_larger_has_more_params(self) -> None:
        self.assertGreater(swin_small_cls().num_parameters(), swin_tiny_cls().num_parameters())
        self.assertGreater(swin_base_cls().num_parameters(), swin_small_cls().num_parameters())


class TestSwinBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = swin_tiny()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        # Swin-T final dim = 96 * 8 = 768
        self.assertEqual(feat.shape, (1, 768))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))


class TestSwinClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = swin_tiny_cls()
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
        m = SwinTransformerForImageClassification(SwinConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestSwinRegistry(unittest.TestCase):

    def test_8_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="swin")), 8)

    def test_all_sizes_present(self) -> None:
        # H10: variant suffixes are full-name (tiny / small / base / large),
        # never single-letter abbreviations.  Pre-H10 ``swin_t`` etc.
        # were renamed to match the paper's naming.
        names = models.list_models(family="swin")
        for size in ["tiny", "small", "base", "large"]:
            self.assertIn(f"swin_{size}", names)
            self.assertIn(f"swin_{size}_cls", names)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("swin_tiny")
        self.assertIsInstance(cfg, SwinConfig)
        self.assertEqual(cfg.embed_dim, 96)

    def test_create_model(self) -> None:
        m = models.create_model("swin_tiny")
        self.assertIsInstance(m, SwinTransformer)


class TestSwinSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = swin_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = SwinTransformerForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = swin_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = SwinTransformerForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


# (factory, slug, tag, source, num_params, resize_size) — the three
# reference_vision IMAGENET1K_V1 variants plus the timm MS_IN22K_FT_IN1K Large.
_SHIPPED = (
    (
        "swin_tiny_cls",
        "swin-tiny",
        "IMAGENET1K_V1",
        "reference_vision/Swin_T_Weights.IMAGENET1K_V1",
        28_288_354,
        232,
    ),
    (
        "swin_small_cls",
        "swin-small",
        "IMAGENET1K_V1",
        "reference_vision/Swin_S_Weights.IMAGENET1K_V1",
        49_606_258,
        246,
    ),
    (
        "swin_base_cls",
        "swin-base",
        "IMAGENET1K_V1",
        "reference_vision/Swin_B_Weights.IMAGENET1K_V1",
        87_768_224,
        238,
    ),
    (
        "swin_large_cls",
        "swin-large",
        "MS_IN22K_FT_IN1K",
        "timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        196_532_476,
        248,
    ),
)


class TestSwinWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.vision.swin import (
            SwinBaseWeights,
            SwinLargeWeights,
            SwinSmallWeights,
            SwinTinyWeights,
        )

        return (
            SwinTinyWeights,
            SwinSmallWeights,
            SwinBaseWeights,
            SwinLargeWeights,
        )

    def test_default_aliases(self) -> None:
        # Each enum's DEFAULT points at its single shipped tag member.
        for cls, (_fac, _slug, tag, *_rest) in zip(self._enums(), _SHIPPED):
            self.assertIs(cls.DEFAULT, cls[tag])

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, tag, src, nparams, _rs) in zip(self._enums(), _SHIPPED):
            member = cls[tag]
            e = member.entry
            self.assertEqual(e.num_classes, 1000)
            # Pre-upload the SHA is the ``__PENDING_UPLOAD__`` sentinel;
            # the conversion main loop patches in the real 64-char digest
            # once the Hub artifact lands.
            self.assertTrue(e.sha256 == "__PENDING_UPLOAD__" or len(e.sha256) == 64)
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn(f"/{tag}/", e.url)
            self.assertEqual(member.meta["tag"], tag)
            self.assertEqual(member.meta["source"], src)
            self.assertEqual(member.meta["license"], "mit")
            self.assertEqual(member.meta["num_params"], nparams)
            self.assertIn("ImageNet-1k", member.meta["metrics"])

    def test_transforms_bicubic_224(self) -> None:
        for cls, (_fac, _slug, tag, _src, _np, resize) in zip(self._enums(), _SHIPPED):
            tf = cls[tag].transforms()
            self.assertEqual(tf.crop_size, 224)
            self.assertEqual(tf.resize_size, resize)
            self.assertEqual(tf.interpolation, "bicubic")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, _slug, tag, *_rest in _SHIPPED:
            self.assertIn(tag, list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestSwinPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_tiny_default(self) -> None:
        m = models.swin_tiny_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_base_string_tag(self) -> None:
        m = models.swin_base_cls(pretrained="IMAGENET1K_V1")
        self.assertIsInstance(m, SwinTransformerForImageClassification)

    def test_large_string_tag(self) -> None:
        m = models.swin_large_cls(pretrained="MS_IN22K_FT_IN1K")
        self.assertIsInstance(m, SwinTransformerForImageClassification)


if __name__ == "__main__":
    unittest.main()
