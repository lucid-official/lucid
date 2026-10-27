"""Unit tests for the paper-faithful CSPNet rebuild (Wang et al., CVPRW 2020).

Pre-3.5 implementation was a single-variant toy with wrong channel
widths.  This file targets the three paper-cited variants:
``cspresnet_50``, ``cspresnext_50``, ``cspdarknet_53``.
"""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.cspnet import CSPNetConfig, CSPNet, cspresnet_50, cspresnet_50_cls


class TestCSPNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CSPNetConfig()
        self.assertEqual(cfg.model_type, "cspnet")
        self.assertEqual(cfg.stem_stride, 2)
        self.assertEqual(cfg.depths, (3, 3, 5, 2))
        self.assertEqual(cfg.out_chs, (128, 256, 512, 1024))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            CSPNetConfig(depths=(3, 3, 5), out_chs=(128, 256, 512, 1024))


class TestCSPNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cspresnet_50()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        feat = self.model.forward_features(lucid.randn(1, 3, 256, 256))
        self.assertEqual(feat.shape[0], 1)
        self.assertEqual(feat.shape[1], 1024)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        out = self.model(lucid.randn(1, 3, 256, 256))
        self.assertIsInstance(out, BaseModelOutput)


class TestCSPNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cspresnet_50_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        out = self.model(lucid.randn(2, 3, 256, 256))
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        self.assertIsNone(self.model(lucid.randn(1, 3, 256, 256)).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        out = self.model(lucid.randn(2, 3, 256, 256), labels=lucid.tensor([0, 999]))
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())


class TestCSPNetRegistry(unittest.TestCase):

    def test_all_variants_registered(self) -> None:
        names = models.list_models(family="cspnet")
        for n in (
            "cspresnet_50",
            "cspresnet_50_cls",
            "cspresnext_50",
            "cspresnext_50_cls",
            "cspdarknet_53",
            "cspdarknet_53_cls",
        ):
            self.assertIn(n, names)

    def test_create_model(self) -> None:
        self.assertIsInstance(models.create_model("cspresnet_50"), CSPNet)


class TestCSPNetParamCountsVsPaper(unittest.TestCase):
    """timm-reported param counts; ±1% margin."""

    def _check(self, factory_name: str, paper_M: float) -> None:
        n = float(getattr(models, factory_name)().num_parameters()) / 1e6
        delta = abs(n - paper_M) / paper_M
        self.assertLess(delta, 0.01, f"{factory_name}: {n:.2f}M vs paper {paper_M}M")

    def test_cspresnet_50(self) -> None:
        self._check("cspresnet_50_cls", 21.62)

    def test_cspresnext_50(self) -> None:
        self._check("cspresnext_50_cls", 20.57)

    def test_cspdarknet_53(self) -> None:
        self._check("cspdarknet_53_cls", 27.64)


class TestCSPNetWeightsEnums(unittest.TestCase):
    """Static contract of every per-variant ``<X>Weights`` enum."""

    def test_default_aliases_in1k(self) -> None:
        from lucid.models.weights import (
            CSPDarknet53Weights,
            CSPResNet50Weights,
            CSPResNeXt50Weights,
        )

        for cls in (CSPResNet50Weights, CSPResNeXt50Weights, CSPDarknet53Weights):
            self.assertIs(cls.DEFAULT, cls.RA_IN1K)

    def test_entry_fields(self) -> None:
        from lucid.models.weights import CSPResNet50Weights

        e = CSPResNet50Weights.RA_IN1K.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)
        self.assertIn("lucid-dl/cspresnet-50", e.url)
        self.assertIn("/RA_IN1K/", e.url)

    def test_meta_provenance(self) -> None:
        from lucid.models.weights import CSPDarknet53Weights

        meta = CSPDarknet53Weights.RA_IN1K.meta
        self.assertEqual(meta["source"], "timm/cspdarknet53.ra_in1k")
        self.assertEqual(meta["license"], "apache-2.0")
        self.assertIn("ImageNet-1k", meta["metrics"])

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for name in ("cspresnet_50_cls", "cspresnext_50_cls", "cspdarknet_53_cls"):
            self.assertIn("RA_IN1K", list_pretrained(name))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestCSPNetPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA verify + load + forward."""

    def test_cspresnet_default(self) -> None:
        m = cspresnet_50_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 256, 256))
        self.assertEqual(out.logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
