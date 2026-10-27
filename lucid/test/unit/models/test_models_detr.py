"""Unit tests for DETR — DEtection TRansformer (Carion et al., 2020)."""

import os
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.detr import (
    DETRConfig,
    DETRForObjectDetection,
    detr_resnet50,
)


class TestDETRConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = DETRConfig()
        self.assertEqual(cfg.model_type, "detr")
        self.assertEqual(cfg.num_classes, 80)
        self.assertEqual(cfg.d_model, 256)
        self.assertEqual(cfg.num_queries, 100)


class TestDETRForward(unittest.TestCase):

    def setUp(self) -> None:
        # Small config override to keep the test cheap, still real factory.
        self.model = detr_resnet50(num_classes=20, num_queries=10)
        self.model.eval()

    def test_output_shapes(self) -> None:
        x = lucid.randn(1, 3, 256, 256)
        out = self.model(x)
        # logits: (B, N, num_classes + 1); boxes: (B, N, 4)
        self.assertEqual(out.logits.shape, (1, 10, 21))
        self.assertEqual(out.pred_boxes.shape, (1, 10, 4))

    def test_boxes_in_unit_range(self) -> None:
        x = lucid.randn(1, 3, 256, 256)
        boxes = self.model(x).pred_boxes
        self.assertGreaterEqual(float(boxes.min().item()), 0.0)
        self.assertLessEqual(float(boxes.max().item()), 1.0)

    def test_no_targets_no_loss(self) -> None:
        x = lucid.randn(1, 3, 256, 256)
        self.assertIsNone(self.model(x).loss)


class TestDETRTransformerTopology(unittest.TestCase):
    """The rebuilt transformer mirrors the reference key layout."""

    def test_reference_key_layout(self) -> None:
        model = detr_resnet50(num_classes=91)
        keys = set(model.state_dict().keys())
        # Encoder layers nested under transformer.encoder.layers.N
        self.assertIn("transformer.encoder.layers.0.self_attn.in_proj_weight", keys)
        self.assertIn("transformer.encoder.layers.5.norm2.weight", keys)
        # No final encoder norm
        self.assertNotIn("transformer.encoder.norm.weight", keys)
        # Decoder layers + final decoder norm
        self.assertIn("transformer.decoder.layers.0.multihead_attn.in_proj_weight", keys)
        self.assertIn("transformer.decoder.norm.weight", keys)
        # Backbone frozen-BN: no num_batches_tracked
        self.assertIn("backbone.bn1.running_mean", keys)
        self.assertFalse(any(k.endswith("num_batches_tracked") for k in keys))

    def test_coco_class_embed_shape(self) -> None:
        model = detr_resnet50(num_classes=91)
        sd = model.state_dict()
        self.assertEqual(sd["class_embed.weight"].shape, (92, 256))


class TestDETRRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="detr")
        self.assertIn("detr_resnet50", names)
        self.assertIn("detr_resnet101", names)

    def test_create_model(self) -> None:
        m = models.create_model("detr_resnet50")
        self.assertIsInstance(m, DETRForObjectDetection)


_SHIPPED = (
    ("detr_resnet50", "detr-resnet-50", "detr_resnet50", 41_524_768, 42.0),
    ("detr_resnet101", "detr-resnet-101", "detr_resnet101", 60_464_672, 43.5),
)


class TestDETRWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.vision.detr import (
            DETRResNet50Weights,
            DETRResNet101Weights,
        )

        return (DETRResNet50Weights, DETRResNet101Weights)

    def test_default_aliases_coco(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.COCO_2017)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams, box_ap) in zip(self._enums(), _SHIPPED):
            e = cls.COCO_2017.entry
            self.assertEqual(e.num_classes, 91)
            # sha256 is either a real 64-hex digest or the upload placeholder.
            self.assertTrue(len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__")
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/COCO_2017/", e.url)
            self.assertEqual(cls.COCO_2017.meta["source"], f"facebookresearch/detr/{src}")
            self.assertEqual(cls.COCO_2017.meta["license"], "apache-2.0")
            self.assertEqual(cls.COCO_2017.meta["num_params"], nparams)
            self.assertEqual(cls.COCO_2017.meta["metrics"]["COCO"]["box mAP"], box_ap)

    def test_transforms_detection_preset(self) -> None:
        for cls in self._enums():
            tf = cls.COCO_2017.transforms()
            # Detection preset round-trips to a Detection-typed config.
            self.assertEqual(tf.to_dict()["preprocessor_type"], "Detection")
            self.assertEqual(tf.max_size, 1333)
            # Unlike the R-CNN presets, DETR's canvas is not rounded up.
            self.assertEqual(tf.canvas_size, 1333)
            # Its reference pads on the bottom and right: the image stays in
            # the top-left corner, the frame ``postprocess`` scales boxes into.
            self.assertEqual(tf.pad_position, "top_left")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("COCO_2017", list_pretrained(fac))


@unittest.skipUnless(
    os.environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestDETRPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_r50_default(self) -> None:
        m = models.detr_resnet50(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 512, 512))
        self.assertEqual(out.logits.shape, (1, 100, 92))

    def test_r101_string_tag(self) -> None:
        m = models.detr_resnet101(pretrained="COCO_2017")
        self.assertIsInstance(m, DETRForObjectDetection)


if __name__ == "__main__":
    unittest.main()
