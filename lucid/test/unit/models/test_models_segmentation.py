"""Unit tests for Wave 3d segmentation models — CPU + Metal parametrized.

Covers:
  FCN (resnet50/101), UNet (base/small/bilinear),
  Attention U-Net, MaskFormer (resnet50/101),
  Mask2Former (swin tiny/small/base/large)

Each test combines factory check, output type, shape, deterministic
self-consistency, and loss=None checks in ONE forward pass per device.

Tests are parametrized over the ``device`` fixture so they run on both
the CPU (Accelerate) and Metal (MLX) streams.
"""

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import SemanticSegmentationOutput

_B = 1
_C = 3
_H = 128
_W = 128


def _img(device: str, ch: int = _C) -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, ch, _H, _W), device=device)


def _build(factory, device: str):
    m = factory()
    m.eval()
    return m.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# FCN
# ─────────────────────────────────────────────────────────────────────────────


class TestFCNResNet50:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet50, FCNForSemanticSegmentation

        m = _build(fcn_resnet50, device)
        assert isinstance(m, FCNForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestFCNResNet101:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet101, FCNForSemanticSegmentation

        m = _build(fcn_resnet101, device)
        assert isinstance(m, FCNForSemanticSegmentation)
        out = m(_img(device))
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[0]) == _B
        assert int(out.logits.shape[-1]) == _W


# ─────────────────────────────────────────────────────────────────────────────
# UNet variants
# ─────────────────────────────────────────────────────────────────────────────


class TestUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet, UNetForSemanticSegmentation

        m = _build(unet, device)
        assert isinstance(m, UNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert int(out.logits.shape[1]) == K
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestResUNet2d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_2d

        m = _build(
            lambda: res_unet_2d(num_classes=4, base_channels=8, depth=2),
            device,
        )
        out = m(_img(device, ch=1))
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 4, _H, _W)


class TestUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet_3d
        import lucid

        m = _build(
            lambda: unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        # (B, K, D, H, W) — same spatial size as input
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


class TestResUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_3d
        import lucid

        m = _build(
            lambda: res_unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.attention_unet import (
            attention_unet,
            AttentionUNetForSemanticSegmentation,
        )

        m = _build(attention_unet, device)
        assert isinstance(m, AttentionUNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskFormer:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.maskformer import (
            maskformer_resnet50,
            MaskFormerForSemanticSegmentation,
        )

        # Slim transformer + queries for test speed
        m = _build(
            lambda: maskformer_resnet50(
                num_queries=20,
                num_decoder_layers=2,
            ),
            device,
        )
        assert isinstance(m, MaskFormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        # Semantic output drops the no-object slot (reference post-processing):
        # exactly num_classes channels, not num_classes + 1.
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former
# ─────────────────────────────────────────────────────────────────────────────


class TestMask2FormerSwinTiny:
    def test_factory_and_forward(self, device: str) -> None:
        # Use a small config override (fewer queries / decoder layers) so the
        # heavy deformable + masked-attention stack stays cheap on CPU/Metal.
        # The Swin backbone pads internally so any input size works.
        from lucid.models.vision.mask2former import (
            mask2former_swin_tiny,
            Mask2FormerForSemanticSegmentation,
        )

        m = _build(
            lambda: mask2former_swin_tiny(
                num_queries=20,
                num_decoder_layers=3,
            ),
            device,
        )
        assert isinstance(m, Mask2FormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        # Semantic output drops the no-object slot → K channels (matches the
        # reference post_process_semantic_segmentation).
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-test (device-independent)
# ─────────────────────────────────────────────────────────────────────────────


class TestSegmentationRegistry:
    def test_segmentation_models_registered(self) -> None:
        import lucid.models as M

        seg_models = M.list_models(task="semantic-segmentation")
        expected = [
            "fcn_resnet50",
            "fcn_resnet101",
            "unet",
            "res_unet_2d",
            "unet_3d",
            "res_unet_3d",
            "attention_unet",
            "maskformer_resnet50",
            "maskformer_resnet101",
            "mask2former_swin_tiny",
            "mask2former_swin_small",
            "mask2former_swin_base",
            "mask2former_swin_large",
        ]
        for name in expected:
            assert name in seg_models, f"{name!r} missing from registry"


# ─────────────────────────────────────────────────────────────────────────────
# FCN pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_FCN_SHIPPED = (
    ("fcn_resnet50", "fcn-resnet-50", "FCN_ResNet50_Weights", 35_322_218, 60.5),
    ("fcn_resnet101", "fcn-resnet-101", "FCN_ResNet101_Weights", 54_314_346, 63.7),
)


def _fcn_enums() -> tuple[type, ...]:
    from lucid.models.weights import FCNResNet50Weights, FCNResNet101Weights

    return (FCNResNet50Weights, FCNResNet101Weights)


def test_fcn_weights_default_aliases() -> None:
    for cls in _fcn_enums():
        assert cls.DEFAULT is cls.COCO_WITH_VOC_LABELS_V1


def test_fcn_weights_entry_fields() -> None:
    for cls, (_fac, slug, src, nparams, miou) in zip(_fcn_enums(), _FCN_SHIPPED):
        e = cls.COCO_WITH_VOC_LABELS_V1.entry
        assert e.num_classes == 21
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert f"lucid-dl/{slug}" in e.url
        assert "/COCO_WITH_VOC_LABELS_V1/" in e.url
        meta = cls.COCO_WITH_VOC_LABELS_V1.meta
        assert meta["source"] == f"reference_vision/{src}.COCO_WITH_VOC_LABELS_V1"
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == nparams
        assert meta["metrics"]["COCO-val2017-VOC-labels"]["mIoU"] == miou


def test_fcn_weights_segmentation_preset() -> None:
    for cls in _fcn_enums():
        tf = cls.COCO_WITH_VOC_LABELS_V1.transforms()
        d = tf.to_dict()
        assert d["preprocessor_type"] == "Segmentation"
        # The reference SemanticSegmentation preset only resizes the shortest
        # side — it does NOT centre-crop.  Cropping to 520x520 discarded ~a
        # quarter of every non-square image (a 500x375 VOC frame becomes
        # 693x520 and then loses 173 columns), so the advertised mIoU was
        # unreachable.  ``crop_size is None`` is the resize-only pipeline.
        assert d["init_kwargs"]["crop_size"] is None
        assert d["init_kwargs"]["resize_size"] == 520


def test_fcn_weights_registry_discoverable() -> None:
    from lucid.weights import list_pretrained

    for fac, *_ in _FCN_SHIPPED:
        assert "COCO_WITH_VOC_LABELS_V1" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_fcn_pretrained_load() -> None:
    import lucid.models as models

    m = models.fcn_resnet50(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 256, 256))
    assert out.logits.shape == (1, 21, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_MASKFORMER_SHIPPED = (
    (
        "maskformer_resnet50",
        "maskformer-resnet-50",
        "facebook/maskformer-resnet50-ade",
        41_307_863,
        44.5,
    ),
    (
        "maskformer_resnet101",
        "maskformer-resnet-101",
        "facebook/maskformer-resnet101-ade",
        60_299_991,
        45.5,
    ),
)


def _maskformer_enums() -> tuple[type, ...]:
    from lucid.models.vision.maskformer import (
        MaskFormerResNet50Weights,
        MaskFormerResNet101Weights,
    )

    return (MaskFormerResNet50Weights, MaskFormerResNet101Weights)


def test_maskformer_weights_default_aliases() -> None:
    for cls in _maskformer_enums():
        assert cls.DEFAULT is cls.ADE20K


def test_maskformer_weights_entry_fields() -> None:
    for cls, (_fac, slug, src, nparams, miou) in zip(_maskformer_enums(), _MASKFORMER_SHIPPED):
        e = cls.ADE20K.entry
        assert e.num_classes == 150
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert f"lucid-dl/{slug}" in e.url
        assert "/ADE20K/" in e.url
        meta = cls.ADE20K.meta
        assert meta["source"] == src
        assert meta["license"] == "other"
        assert meta["num_params"] == nparams
        assert meta["metrics"]["ADE20K"]["mIoU"] == miou


def test_maskformer_weights_segmentation_preset() -> None:
    for cls in _maskformer_enums():
        tf = cls.ADE20K.transforms()
        d = tf.to_dict()
        assert d["preprocessor_type"] == "Segmentation"
        assert d["init_kwargs"]["crop_size"] == 512


def test_maskformer_weights_registry_discoverable() -> None:
    from lucid.weights import list_pretrained

    for fac, *_ in _MASKFORMER_SHIPPED:
        assert "ADE20K" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_maskformer_pretrained_load() -> None:
    import lucid.models as models

    m = models.maskformer_resnet50(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 256, 256))
    # Semantic output: num_classes channels (no-object slot dropped).
    assert out.logits.shape == (1, 150, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former pretrained weights — static enum contract (no network)
# ─────────────────────────────────────────────────────────────────────────────

_MASK2FORMER_SHIPPED = (
    (
        "mask2former_swin_tiny",
        "mask2former-swin-tiny-ade",
        "facebook/mask2former-swin-tiny-ade-semantic",
        47_468_596,  # built + empty_weight + rel-pos index buffers
        47.7,
    ),
    (
        "mask2former_swin_small",
        "mask2former-swin-small-ade",
        "facebook/mask2former-swin-small-ade-semantic",
        68_815_312,
        51.3,
    ),
    (
        "mask2former_swin_base",
        "mask2former-swin-base-ade",
        "facebook/mask2former-swin-base-ade-semantic",
        107_420_006,
        52.4,  # MODEL_ZOO.md Swin-B; 53.9 is the IN21k row
    ),
    (
        "mask2former_swin_large",
        "mask2former-swin-large-ade",
        "facebook/mask2former-swin-large-ade-semantic",
        215_986_594,
        56.1,
    ),
)


def _mask2former_enums() -> tuple[type, ...]:
    from lucid.models.vision.mask2former import (
        Mask2FormerSwinTinyWeights,
        Mask2FormerSwinSmallWeights,
        Mask2FormerSwinBaseWeights,
        Mask2FormerSwinLargeWeights,
    )

    return (
        Mask2FormerSwinTinyWeights,
        Mask2FormerSwinSmallWeights,
        Mask2FormerSwinBaseWeights,
        Mask2FormerSwinLargeWeights,
    )


class TestMask2FormerWeightsEnums:
    def test_default_aliases(self) -> None:
        for cls in _mask2former_enums():
            assert cls.DEFAULT is cls.ADE20K

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams, miou) in zip(
            _mask2former_enums(), _MASK2FORMER_SHIPPED
        ):
            e = cls.ADE20K.entry
            assert e.num_classes == 150
            assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
            assert f"lucid-dl/{slug}" in e.url
            assert "/ADE20K/" in e.url
            meta = cls.ADE20K.meta
            assert meta["source"] == src
            assert meta["license"] == "other"
            assert meta["num_params"] == nparams
            assert meta["metrics"]["ADE20K"]["mIoU"] == miou

    def test_segmentation_preset(self) -> None:
        # The upstream processor is configured with an explicit
        # size={"height": 384, "width": 384}, so it resizes straight to
        # 384x384 with no crop -- hence stretch, and crop_size unset.
        for cls in _mask2former_enums():
            tf = cls.ADE20K.transforms()
            d = tf.to_dict()
            assert d["preprocessor_type"] == "Segmentation"
            assert d["init_kwargs"]["resize_size"] == 384
            assert d["init_kwargs"]["stretch"] is True
            assert d["init_kwargs"]["crop_size"] is None

    def test_preset_output_is_exactly_384_square(self) -> None:
        import lucid

        for cls in _mask2former_enums():
            out = cls.ADE20K.transforms()(lucid.rand(3, 500, 375))
            assert tuple(out.shape)[1:] == (384, 384)

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _MASK2FORMER_SHIPPED:
            assert "ADE20K" in list_pretrained(fac)


@pytest.mark.skipif(
    __import__("os").environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
def test_mask2former_pretrained_load() -> None:
    import lucid.models as models

    m = models.mask2former_swin_tiny(pretrained=True)
    m.eval()
    out = m(lucid.randn(1, 3, 384, 384))
    # Semantic output: num_classes channels (no-object slot dropped).
    assert out.logits.shape == (1, 150, 384, 384)


# ---------------------------------------------------------------------------
# Mask2Former training objective (3.2.2)
# ---------------------------------------------------------------------------


def _m2f_tiny_config(**over: object):
    from lucid.models.vision.mask2former._config import Mask2FormerConfig

    base = dict(
        num_classes=3,
        num_queries=4,
        swin_depths=(1, 1, 1, 1),
        swin_embed_dim=24,
        swin_num_heads=(1, 1, 1, 1),
        d_model=32,
        mask_feature_size=32,
        n_head=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        dim_feedforward=32,
        encoder_feedforward_dim=32,
        train_num_points=64,
    )
    base.update(over)
    return Mask2FormerConfig(**base)  # type: ignore[arg-type]


def _m2f_targets(side: int = 64) -> list[dict[str, object]]:
    square = [
        [1.0 if (8 <= r < 40 and 8 <= c < 40) else 0.0 for c in range(side)] for r in range(side)
    ]
    return [{"labels": lucid.tensor([1]).long(), "masks": lucid.tensor([square])}]


class TestMask2FormerTraining:
    def test_loss_is_finite_and_inference_unchanged(self) -> None:
        from lucid.models.vision.mask2former._model import (
            Mask2FormerForSemanticSegmentation,
        )

        lucid.manual_seed(0)
        m = Mask2FormerForSemanticSegmentation(_m2f_tiny_config())
        x = lucid.randn(1, 3, 64, 64)
        out = m(x, targets=_m2f_targets())
        assert out.loss is not None
        assert bool(out.loss.isfinite().all().item())
        assert m(x).loss is None

    def test_all_three_terms_are_alive(self) -> None:
        """A term stuck at zero is a term that is not being computed."""
        from lucid.models.vision.mask2former._model import _m2f_stage_loss

        lucid.manual_seed(0)
        cfg = _m2f_tiny_config()
        cls_loss, ce_loss, dice = _m2f_stage_loss(
            lucid.randn(1, 4, 4), lucid.randn(1, 4, 16, 16), _m2f_targets(), cfg
        )
        assert float(cls_loss.item()) > 0.0
        assert float(ce_loss.item()) > 0.0
        assert float(dice.item()) > 0.0

    def test_matcher_picks_the_query_that_fits(self) -> None:
        """Hand-built: only query 2 has both the right shape and class."""
        from lucid.models._utils._segmentation import sample_point_coords
        from lucid.models.vision.mask2former._model import _m2f_match

        lucid.manual_seed(0)
        cfg = _m2f_tiny_config(train_num_points=256)
        side = 8
        gt_masks = lucid.tensor(
            [[[1.0 if c < 4 else 0.0 for c in range(side)] for _ in range(side)]]
        )
        gt_labels = lucid.tensor([1]).long()
        right = [[20.0 if c < 4 else -20.0 for c in range(side)] for _ in range(side)]
        wrong = [[-20.0 if c < 4 else 20.0 for c in range(side)] for _ in range(side)]
        mask_logits = lucid.tensor([wrong, wrong, right, wrong])
        class_logits = lucid.tensor(
            [
                [8.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
            ]
        )
        coords = sample_point_coords(cfg.train_num_points)
        pred, gt = _m2f_match(class_logits, mask_logits, gt_labels, gt_masks, coords, cfg)
        assert list(zip(pred, gt)) == [(2, 0)]

    def test_deep_supervision_adds_the_auxiliary_layers(self) -> None:
        from dataclasses import replace

        from lucid.models.vision.mask2former._model import (
            Mask2FormerForSemanticSegmentation,
        )

        lucid.manual_seed(0)
        cfg = _m2f_tiny_config()
        deep = Mask2FormerForSemanticSegmentation(cfg)
        shallow = Mask2FormerForSemanticSegmentation(replace(cfg, deep_supervision=False))
        shallow.load_state_dict(deep.state_dict())
        x = lucid.randn(1, 3, 64, 64)
        targets = _m2f_targets()

        lucid.manual_seed(1)
        with_aux = float(deep(x, targets=targets).loss.item())
        lucid.manual_seed(1)
        without = float(shallow(x, targets=targets).loss.item())
        assert with_aux > without

    def test_importance_sampler_targets_the_boundary(self) -> None:
        """The point budget is only worth spending near the decision contour."""
        from lucid.models._utils._segmentation import (
            point_sample,
            uncertain_point_coords,
        )

        lucid.manual_seed(0)
        side = 32
        ramp = [[-20.0 + 40.0 * c / (side - 1) for c in range(side)] for _ in range(side)]
        logits = lucid.tensor([ramp])
        coords = uncertain_point_coords(logits, 400, 3.0, 0.75)
        values = [abs(v) for v in point_sample(logits, coords)[0].tolist()]
        importance, uniform = values[:300], values[300:]
        assert sum(importance) / len(importance) < sum(uniform) / len(uniform)

    def test_overfits_a_single_example(self) -> None:
        import lucid.optim as optim

        from lucid.models.vision.mask2former._model import (
            Mask2FormerForSemanticSegmentation,
        )

        lucid.manual_seed(0)
        m = Mask2FormerForSemanticSegmentation(_m2f_tiny_config(num_decoder_layers=2))
        m.train()
        opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        x = lucid.randn(1, 3, 64, 64)
        targets = _m2f_targets()
        losses: list[float] = []
        for _ in range(12):
            opt.zero_grad()
            out = m(x, targets=targets)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))
        assert all(v == v for v in losses)
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Attention U-Net rank (2-D images vs the paper's 3-D volumes)
# ---------------------------------------------------------------------------


class TestAttentionUNetRank:
    @staticmethod
    def _model(dims: int):
        from lucid.models.vision.attention_unet._config import AttentionUNetConfig
        from lucid.models.vision.attention_unet._model import (
            AttentionUNetForSemanticSegmentation,
        )

        return AttentionUNetForSemanticSegmentation(
            AttentionUNetConfig(num_classes=3, base_channels=8, depth=2, spatial_dims=dims)
        )

    def test_two_d_shape_is_unchanged(self) -> None:
        m = self._model(2)
        m.eval()
        assert tuple(m(lucid.randn(1, 1, 32, 32)).logits.shape) == (1, 3, 32, 32)

    def test_three_d_preserves_every_spatial_axis(self) -> None:
        m = self._model(3)
        m.eval()
        out = m(lucid.randn(1, 1, 16, 32, 32))
        assert tuple(out.logits.shape) == (1, 3, 16, 32, 32)

    def test_layers_do_not_mix_ranks(self) -> None:
        """A stray Conv2d in the 3-D model would only surface on odd shapes."""
        two = {type(mod).__name__ for mod in self._model(2).modules()}
        three = {type(mod).__name__ for mod in self._model(3).modules()}
        assert "Conv2d" in two and "Conv3d" not in two
        assert "Conv3d" in three and "Conv2d" not in three
        assert {"BatchNorm3d", "MaxPool3d", "ConvTranspose3d"} <= three

    def test_three_d_loss_path(self) -> None:
        m = self._model(3)
        targets = lucid.zeros(1, 16, 32, 32).long()
        out = m(lucid.randn(1, 1, 16, 32, 32), targets=targets)
        assert out.loss is not None
        assert bool(out.loss.isfinite().all().item())

    def test_factory_is_registered(self) -> None:
        import lucid.models as models

        m = models.attention_unet_3d(base_channels=8, depth=2)
        m.eval()
        assert tuple(m(lucid.randn(1, 1, 16, 32, 32)).logits.shape)[2:] == (16, 32, 32)


class TestAttentionUNetDeepSupervision:
    """``unet_CT_multi_att_dsv`` — per-level classifiers fused by a 1x1 conv.

    Despite the "deep supervision" name the reference does not add auxiliary
    losses; it concatenates each level's upsampled class map and fuses them,
    so the mechanism is part of the prediction, not only of training.
    """

    @staticmethod
    def _model(*, deep_supervision: bool, dims: int = 2):
        from lucid.models.vision.attention_unet._config import AttentionUNetConfig
        from lucid.models.vision.attention_unet._model import (
            AttentionUNetForSemanticSegmentation,
        )

        return AttentionUNetForSemanticSegmentation(
            AttentionUNetConfig(
                num_classes=3,
                base_channels=8,
                depth=3,
                spatial_dims=dims,
                deep_supervision=deep_supervision,
            )
        )

    def test_default_is_off(self) -> None:
        """Existing checkpoints have no ``dsv``/``fuse`` tensors, so the
        feature must not appear unless it is asked for."""
        from lucid.models.vision.attention_unet._config import AttentionUNetConfig

        assert AttentionUNetConfig().deep_supervision is False
        keys = self._model(deep_supervision=False).state_dict().keys()
        assert not [k for k in keys if "dsv" in k or "fuse" in k]

    def test_is_purely_additive(self) -> None:
        off = set(self._model(deep_supervision=False).state_dict().keys())
        on = set(self._model(deep_supervision=True).state_dict().keys())
        assert off < on
        assert sorted(on - off) == [
            "dsv_1.bias",
            "dsv_1.weight",
            "dsv_2.bias",
            "dsv_2.weight",
            "fuse.bias",
            "fuse.weight",
        ]

    def test_output_shape_is_unchanged(self) -> None:
        m = self._model(deep_supervision=True)
        m.eval()
        assert tuple(m(lucid.randn(1, 1, 64, 64)).logits.shape) == (1, 3, 64, 64)

    def test_each_level_reaches_the_output(self) -> None:
        """A level wired in but never read would still build, still train,
        and still produce the right shape — only the gradient shows it."""
        m = self._model(deep_supervision=True)
        out = m(lucid.randn(1, 1, 64, 64), targets=lucid.zeros(1, 64, 64).long())
        assert out.loss is not None
        out.loss.backward()
        for name in ("head", "dsv_1", "dsv_2", "fuse"):
            grad = getattr(m, name).weight.grad
            assert grad is not None, name
            assert float(grad.abs().max().item()) > 0.0, name

    def test_classifiers_match_their_decoder_widths(self) -> None:
        """``dsv_i`` reads decoder level ``i``, whose width is
        ``base_channels * 2**i`` — a mismatch here silently classifies the
        wrong stage."""
        m = self._model(deep_supervision=True)
        assert int(m.dsv_1.weight.shape[1]) == 16
        assert int(m.dsv_2.weight.shape[1]) == 32
        assert int(m.fuse.weight.shape[1]) == 3 * 3  # num_classes * depth

    def test_three_d_path(self) -> None:
        m = self._model(deep_supervision=True, dims=3)
        m.eval()
        out = m(lucid.randn(1, 1, 16, 32, 32))
        assert tuple(out.logits.shape) == (1, 3, 16, 32, 32)
