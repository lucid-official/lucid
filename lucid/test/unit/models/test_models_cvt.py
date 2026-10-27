"""Unit tests for CvT — Convolutional Vision Transformer (Wu et al., 2021)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.cvt import CvTConfig, CvT, CvTForImageClassification, cvt_13, cvt_13_cls


class TestCvTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CvTConfig()
        self.assertEqual(cfg.model_type, "cvt")


class TestCvTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cvt_13()
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


class TestCvTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cvt_13_cls()
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
        m = CvTForImageClassification(CvTConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestCvTRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="cvt")
        for n in ("cvt_13", "cvt_13_cls", "cvt_21", "cvt_21_cls", "cvt_w24", "cvt_w24_cls"):
            self.assertIn(n, names)

    def test_create_model(self) -> None:
        m = models.create_model("cvt_13")
        self.assertIsInstance(m, CvT)


class TestCvTClsToken(unittest.TestCase):
    """The reference CvT carries a CLS token only on the last stage."""

    def test_cls_token_only_last_stage(self) -> None:
        m = cvt_13()
        flags = [st.cls_token is not None for st in m.stages]
        self.assertEqual(flags, [False, False, True])

    def test_cls_token_shape(self) -> None:
        m = cvt_13()
        cls = m.stages[-1].cls_token
        self.assertIsNotNone(cls)
        self.assertEqual(cls.shape, (1, 1, 384))


class TestCvTWeightsEnums(unittest.TestCase):
    """Static contract of every per-variant ``<X>Weights`` enum."""

    def test_default_aliases(self) -> None:
        from lucid.models.weights import CvT13Weights, CvT21Weights, CvTW24Weights

        self.assertIs(CvT13Weights.DEFAULT, CvT13Weights.IN1K)
        self.assertIs(CvT21Weights.DEFAULT, CvT21Weights.IN1K)
        self.assertIs(CvTW24Weights.DEFAULT, CvTW24Weights.IN22K_FT_IN1K_384)

    def test_entry_fields(self) -> None:
        from lucid.models.weights import CvT13Weights

        e = CvT13Weights.IN1K.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)
        self.assertIn("lucid-dl/cvt-13", e.url)

    def test_provenance(self) -> None:
        from lucid.models.weights import CvTW24Weights

        meta = CvTW24Weights.IN22K_FT_IN1K_384.meta
        self.assertEqual(meta["source"], "transformers/microsoft/cvt-w24-384-22k")
        tf = CvTW24Weights.IN22K_FT_IN1K_384.transforms()
        self.assertEqual(tf.crop_size, 384)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        self.assertIn("IN1K", list_pretrained("cvt_13_cls"))
        self.assertIn("IN22K_FT_IN1K_384", list_pretrained("cvt_w24_cls"))


if __name__ == "__main__":
    unittest.main()
