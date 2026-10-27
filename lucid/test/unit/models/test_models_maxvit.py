"""Unit tests for MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022)."""

import tempfile
import unittest

import lucid
import lucid.models as models
import lucid.optim as optim
from lucid.models.vision.maxvit import (
    MaxViTConfig,
    MaxViT,
    MaxViTForImageClassification,
    maxvit_tiny,
    maxvit_tiny_cls,
)


class TestMaxViTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MaxViTConfig()
        self.assertEqual(cfg.model_type, "maxvit")
        self.assertIsInstance(cfg.window_size, int)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = MaxViTConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = MaxViTConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.dims, tuple)
        finally:
            os.unlink(path)


class TestMaxViTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = maxvit_tiny()
        self.model.eval()

    def test_feature_info(self) -> None:
        fi = self.model.feature_info
        self.assertGreater(len(fi), 0)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_non_divisible_spatial_dims(self) -> None:
        """MaxViT must handle H,W not divisible by window_size via padding.

        224 is divisible all the way down, so it never reaches the padding
        branch — the input has to be chosen so a stage's feature map is not
        a multiple of the window.  240/32 = 7.5, so the last stage lands on
        an odd grid and the pad-then-crop path runs.
        """
        for size in (224, 240):
            with self.subTest(size=size):
                x = lucid.randn(1, 3, size, size)
                feat = self.model.forward_features(x)
                self.assertEqual(feat.shape[0], 1)
                self.assertTrue(bool(feat.isfinite().all().item()))


class TestMaxViTPaddedBatch(unittest.TestCase):
    """A padded feature map answers every sample of a batch (LCD-554).

    The validity plane that drives padded keys out of the softmax was built
    for one image, so the pad bias held one row per window of the first
    sample and failed to broadcast against the B·windows attention batch:
    every resolution that is not a multiple of the window grid ran at batch
    size 1 only.
    """

    def _model(self) -> MaxViTForImageClassification:
        lucid.manual_seed(0)
        return models.create_model(
            "maxvit_tiny_cls",
            depths=(1, 1, 1, 1),
            dims=(16, 32, 64, 128),
            window_size=2,
            head_dim=8,
            stem_width=16,
        )

    def test_a_batch_answers_as_each_sample_alone(self) -> None:
        # 100 pads from stage 0 (25x25), 96 only at the last stage (3x3).
        model = self._model().eval()
        for side in (100, 96, 65):
            with self.subTest(side=side), lucid.no_grad():
                x = lucid.randn(3, 3, side, side)
                batch = model(x).logits
                alone = lucid.cat([model(x[i : i + 1]).logits for i in range(3)])
                self.assertLess(float((batch - alone).abs().max()), 1e-5)

    def test_trains_on_a_padded_batch(self) -> None:
        model = self._model().train()
        opt = optim.SGD(model.parameters(), lr=0.1)
        out = model(lucid.randn(2, 3, 100, 100), labels=lucid.tensor([0, 7]))
        assert out.loss is not None
        out.loss.backward()
        opt.step()
        self.assertTrue(bool(out.loss.isfinite().item()))


class TestMaxViTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = maxvit_tiny_cls()
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
        m = MaxViTForImageClassification(MaxViTConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMaxViTRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="maxvit")
        self.assertIn("maxvit_tiny", names)
        self.assertIn("maxvit_tiny_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("maxvit_tiny")
        self.assertIsInstance(m, MaxViT)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("maxvit_tiny")
        self.assertIsInstance(cfg, MaxViTConfig)
        self.assertIsInstance(cfg.window_size, int)


class TestMaxViTSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = maxvit_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = MaxViTForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


_SHIPPED = (
    ("maxvit_tiny_cls", "maxvit-tiny", "maxvit_tiny_tf_224.in1k", 30_916_528),
    ("maxvit_small_cls", "maxvit-small", "maxvit_small_tf_224.in1k", 68_927_956),
    ("maxvit_base_cls", "maxvit-base", "maxvit_base_tf_224.in1k", 119_467_708),
    ("maxvit_large_cls", "maxvit-large", "maxvit_large_tf_224.in1k", 211_785_560),
)


class TestMaxViTWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.vision.maxvit import (
            MaxViTTinyWeights,
            MaxViTSmallWeights,
            MaxViTBaseWeights,
            MaxViTLargeWeights,
        )

        return (MaxViTTinyWeights, MaxViTSmallWeights, MaxViTBaseWeights, MaxViTLargeWeights)

    def test_default_aliases_in1k(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.IN1K)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams) in zip(self._enums(), _SHIPPED):
            e = cls.IN1K.entry
            self.assertEqual(e.num_classes, 1000)
            # 64-char hex once uploaded; sentinel until the hub patch lands.
            self.assertTrue(len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__")
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/IN1K/", e.url)
            self.assertEqual(cls.IN1K.meta["source"], f"timm/{src}")
            self.assertEqual(cls.IN1K.meta["license"], "apache-2.0")
            self.assertEqual(cls.IN1K.meta["num_params"], nparams)

    def test_transforms_bicubic_224(self) -> None:
        for cls in self._enums():
            tf = cls.IN1K.transforms()
            self.assertEqual(tf.crop_size, 224)
            self.assertEqual(tf.resize_size, 235)
            self.assertEqual(tf.interpolation, "bicubic")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("IN1K", list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestMaxViTPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_tiny_default(self) -> None:
        m = models.maxvit_tiny_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_small_string_tag(self) -> None:
        m = models.maxvit_small_cls(pretrained="IN1K")
        self.assertIsInstance(m, MaxViTForImageClassification)


if __name__ == "__main__":
    unittest.main()
