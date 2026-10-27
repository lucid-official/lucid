"""Unit tests for ResNeSt (Zhang et al., 2020)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.resnest import (
    ResNeStConfig,
    ResNeSt,
    ResNeStForImageClassification,
    resnest_50,
    resnest_50_cls,
    resnest_101,
)


class TestResNeStConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = ResNeStConfig()
        self.assertEqual(cfg.model_type, "resnest")

    def test_json_round_trip(self) -> None:
        import json
        import os
        import tempfile

        cfg = ResNeStConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ResNeStConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestResNeStParamCounts(unittest.TestCase):

    def test_101_more_params_than_50(self) -> None:
        self.assertGreater(resnest_101().num_parameters(), resnest_50().num_parameters())


class TestResNeStBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnest_50()
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


class TestResNeStClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnest_50_cls()
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
        m = ResNeStForImageClassification(ResNeStConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestResNeStRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        # 6 registered depths (14 / 26 / 50 / 101 / 200 / 269) x 2
        # (raw + ``_cls`` task wrapper) = 12 registered names.
        #
        # Not all six are paper variants: ResNeSt is Zhang et al., 2020
        # (arXiv:2004.08955), whose Table 1 is the ImageNet ablation study
        # (mixup/autoaug on the left, radix-vs-cardinality on the right) and
        # contains no depth table at all.  14 and 26 appear nowhere in the
        # paper — they come from timm.  The old comment credited "He 2020
        # Table 1", which is the wrong author and the wrong table.
        self.assertEqual(len(models.list_models(family="resnest")), 12)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="resnest")
        self.assertIn("resnest_50", names)
        self.assertIn("resnest_50_cls", names)
        self.assertIn("resnest_101", names)
        self.assertIn("resnest_101_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("resnest_50")
        self.assertIsInstance(m, ResNeSt)


# (factory, slug, timm-source, num_params, crop, resize, interpolation)
_SHIPPED = (
    ("resnest_50_cls", "resnest-50", "resnest50d.in1k", 27_483_240, 224, 256, "bilinear"),
    ("resnest_101_cls", "resnest-101", "resnest101e.in1k", 48_275_016, 256, 292, "bilinear"),
    ("resnest_200_cls", "resnest-200", "resnest200e.in1k", 70_201_544, 320, 352, "bicubic"),
    ("resnest_269_cls", "resnest-269", "resnest269e.in1k", 110_929_480, 416, 448, "bicubic"),
)


class TestResNeStWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.vision.resnest import (
            ResNeSt50Weights,
            ResNeSt101Weights,
            ResNeSt200Weights,
            ResNeSt269Weights,
        )

        return (ResNeSt50Weights, ResNeSt101Weights, ResNeSt200Weights, ResNeSt269Weights)

    def test_default_aliases_in1k(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.IN1K)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams, *_pre) in zip(self._enums(), _SHIPPED):
            e = cls.IN1K.entry
            self.assertEqual(e.num_classes, 1000)
            # sha256 is the upload placeholder until the conversion loop
            # patches the real 64-char digest post-Hub-upload.
            self.assertTrue(e.sha256 == "__PENDING_UPLOAD__" or len(e.sha256) == 64)
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/IN1K/", e.url)
            self.assertEqual(cls.IN1K.meta["source"], f"timm/{src}")
            self.assertEqual(cls.IN1K.meta["license"], "apache-2.0")
            self.assertEqual(cls.IN1K.meta["num_params"], nparams)

    def test_transforms_per_variant(self) -> None:
        for cls, (*_head, crop, resize, interp) in zip(self._enums(), _SHIPPED):
            tf = cls.IN1K.transforms()
            self.assertEqual(tf.crop_size, crop)
            self.assertEqual(tf.resize_size, resize)
            self.assertEqual(tf.interpolation, interp)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("IN1K", list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestResNeStPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_50_default(self) -> None:
        m = models.resnest_50_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_101_string_tag(self) -> None:
        m = models.resnest_101_cls(pretrained="IN1K")
        self.assertIsInstance(m, ResNeStForImageClassification)


if __name__ == "__main__":
    unittest.main()
