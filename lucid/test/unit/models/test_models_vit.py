"""Unit tests for Vision Transformer (Dosovitskiy et al., 2020)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid._C import engine as _C_engine
from lucid.models.vision.vit import (
    ViT,
    ViTConfig,
    ViTForImageClassification,
    vit_base_16,
    vit_base_16_cls,
    vit_base_32,
    vit_base_32_cls,
    vit_large_16,
    vit_large_16_cls,
)


class TestViTConfig(unittest.TestCase):

    def test_defaults_b16(self) -> None:
        cfg = ViTConfig()
        self.assertEqual(cfg.model_type, "vit")
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.dim, 768)
        self.assertEqual(cfg.depth, 12)
        self.assertEqual(cfg.num_heads, 12)
        self.assertAlmostEqual(cfg.mlp_ratio, 4.0)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = ViTConfig(patch_size=32, dim=1024, depth=24, num_heads=16)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ViTConfig.from_dict(d)
            self.assertEqual(cfg2.patch_size, 32)
            self.assertEqual(cfg2.dim, 1024)
            self.assertEqual(cfg2.depth, 24)
        finally:
            os.unlink(path)


class TestViTParamCounts(unittest.TestCase):

    def test_b16_classifier(self) -> None:
        # Reference-exact: 86,567,656
        self.assertEqual(vit_base_16_cls().num_parameters(), 86_567_656)

    def test_b32_classifier(self) -> None:
        # Fewer patches → fewer pos embedding params
        self.assertEqual(vit_base_32_cls().num_parameters(), 88_224_232)

    def test_l16_classifier(self) -> None:
        self.assertEqual(vit_large_16_cls().num_parameters(), 304_326_632)

    def test_larger_variants_more_params(self) -> None:
        self.assertGreater(
            vit_large_16().num_parameters(),
            vit_base_16().num_parameters(),
        )


class TestViTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vit_base_16()
        self.model.eval()

    def test_feature_info_single_stage(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 1)
        self.assertEqual(fi[0].num_channels, 768)
        self.assertEqual(fi[0].reduction, 16)

    def test_forward_features_cls_token(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        feat = self.model.forward_features(x)
        # CLS token: (B, dim)
        self.assertEqual(feat.shape, (2, 768))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        # Unsqueezed: (B, 1, dim)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))

    def test_b32_different_sequence_length(self) -> None:
        m = vit_base_32()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        # B/32 has 7×7=49 patches vs B/16's 14×14=196 patches
        feat = m.forward_features(x)
        self.assertEqual(feat.shape, (1, 768))  # CLS dim same

    def test_l16_larger_dim(self) -> None:
        m = vit_large_16()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        feat = m.forward_features(x)
        self.assertEqual(feat.shape, (1, 1024))


class TestViTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vit_base_16_cls()
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
        cfg = ViTConfig(num_classes=10)
        m = ViTForImageClassification(cfg)
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))

    def test_custom_image_size(self) -> None:
        cfg = ViTConfig(image_size=384, patch_size=16, num_classes=10)
        m = ViTForImageClassification(cfg)
        m.eval()
        x = lucid.randn(1, 3, 384, 384)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestViTResolutionBoundary(unittest.TestCase):
    """Another patch grid is refused at the patch embedding (LCD-554).

    The position table has one row per patch of the built grid.  Another
    grid failed as a broadcast ShapeMismatch at ``x + pos_embed`` that named
    neither size — or, with the same patch count on another layout, answered
    silently with every patch reading another patch's position entry.
    """

    _SMALL: dict[str, int] = {
        "image_size": 32,
        "patch_size": 8,
        "dim": 32,
        "depth": 1,
        "num_heads": 2,
    }

    def _refusal(self, factory: str, height: int, width: int) -> str:
        lucid.manual_seed(0)
        model = models.create_model(factory, **self._SMALL).eval()
        with self.assertRaises(ValueError) as caught:
            model(lucid.randn(1, 3, height, width))
        # The engine's ShapeMismatch is a ValueError too.
        self.assertNotIsInstance(caught.exception, _C_engine.LucidError)
        return str(caught.exception)

    def test_another_grid_names_both_sizes(self) -> None:
        for factory in ("vit_base_16", "vit_base_16_cls"):
            for side in (24, 40, 64):
                with self.subTest(factory=factory, side=side):
                    message = self._refusal(factory, side, side)
                    self.assertIn("built for 32x32", message)
                    self.assertIn(f"Got {side}x{side} pixels", message)
                    self.assertIn(f"image_size={side}", message)

    def test_a_non_square_input_names_its_sides(self) -> None:
        message = self._refusal("vit_base_16_cls", 32, 64)
        self.assertIn("Got 32x64 pixels, which is 4x8 patches", message)
        self.assertNotIn("image_size=", message)

    def test_the_same_patch_count_on_another_layout_is_refused(self) -> None:
        # 2x8, 8x2 and 1x16 patches are the built grid's 16 — the table fit,
        # so these used to answer, each patch reading another's position.
        for height, width in ((16, 64), (64, 16), (8, 128)):
            with self.subTest(height=height, width=width):
                message = self._refusal("vit_base_16", height, width)
                self.assertIn(f"Got {height}x{width} pixels", message)

    def test_the_built_grid_still_answers_at_any_remainder(self) -> None:
        # The patch conv drops a remainder smaller than a patch, so 39 px is
        # still the 4x4 grid the table was built for.
        lucid.manual_seed(0)
        model = models.create_model("vit_base_16_cls", **self._SMALL).eval()
        for side in (32, 36, 39):
            with self.subTest(side=side), lucid.no_grad():
                self.assertEqual(model(lucid.randn(1, 3, side, side)).logits.shape, (1, 1000))


class TestViTRegistry(unittest.TestCase):

    def test_10_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="vit")), 10)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="vit")
        for v in [
            "vit_base_16",
            "vit_base_32",
            "vit_large_16",
            "vit_large_32",
            "vit_huge_14",
        ]:
            self.assertIn(v, names)
            self.assertIn(f"{v}_cls", names)

    def test_auto_config_b16(self) -> None:
        cfg = models.AutoConfig.from_pretrained("vit_base_16")
        self.assertIsInstance(cfg, ViTConfig)
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.dim, 768)

    def test_create_model(self) -> None:
        m = models.create_model("vit_base_16")
        self.assertIsInstance(m, ViT)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("vit_base_16_cls")
        self.assertIsInstance(m, ViTForImageClassification)


class TestViTSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = vit_base_16_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ViTForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = vit_base_16_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = ViTForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


_SHIPPED = (
    ("vit_base_16_cls", "vit-base-16", "ViT_B_16_Weights", 86_567_656, 256),
    ("vit_base_32_cls", "vit-base-32", "ViT_B_32_Weights", 88_224_232, 256),
    ("vit_large_16_cls", "vit-large-16", "ViT_L_16_Weights", 304_326_632, 242),
    ("vit_large_32_cls", "vit-large-32", "ViT_L_32_Weights", 306_535_400, 256),
)


class TestViTWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.vision.vit import (
            ViTBase16Weights,
            ViTBase32Weights,
            ViTLarge16Weights,
            ViTLarge32Weights,
        )

        return (
            ViTBase16Weights,
            ViTBase32Weights,
            ViTLarge16Weights,
            ViTLarge32Weights,
        )

    def test_default_aliases_imagenet1k_v1(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.IMAGENET1K_V1)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams, _rs) in zip(self._enums(), _SHIPPED):
            e = cls.IMAGENET1K_V1.entry
            self.assertEqual(e.num_classes, 1000)
            # sha256 is the upload placeholder until the conversion loop
            # patches the real 64-char digest post-Hub-upload.
            self.assertTrue(e.sha256 == "__PENDING_UPLOAD__" or len(e.sha256) == 64)
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/IMAGENET1K_V1/", e.url)
            self.assertEqual(
                cls.IMAGENET1K_V1.meta["source"],
                f"reference_vision/{src}.IMAGENET1K_V1",
            )
            self.assertEqual(cls.IMAGENET1K_V1.meta["license"], "bsd-3-clause")
            self.assertEqual(cls.IMAGENET1K_V1.meta["num_params"], nparams)

    def test_transforms_bilinear_224(self) -> None:
        for cls, (_fac, _slug, _src, _np, resize) in zip(self._enums(), _SHIPPED):
            tf = cls.IMAGENET1K_V1.transforms()
            self.assertEqual(tf.crop_size, 224)
            self.assertEqual(tf.resize_size, resize)
            self.assertEqual(tf.interpolation, "bilinear")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("IMAGENET1K_V1", list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestViTPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_base_16_default(self) -> None:
        m = models.vit_base_16_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_base_32_string_tag(self) -> None:
        m = models.vit_base_32_cls(pretrained="IMAGENET1K_V1")
        self.assertIsInstance(m, ViTForImageClassification)


if __name__ == "__main__":
    unittest.main()
