"""Unit tests for Wave 3d detection models — CPU + Metal parametrized.

Covers:
  R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, DETR,
  EfficientDet D0, YOLO v1/v2/v3/v4

For each model we run **one forward pass per device** and check shape /
type / loss / deterministic self-consistency in a single test method so
heavy models (Mask R-CNN, EfficientDet, DETR) don't get re-instantiated
for every assertion.

Tests are parametrized over the ``device`` fixture so they run on both
the CPU (Accelerate) and Metal (MLX) compute streams.
"""

import os
import unittest
from typing import Any

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import InstanceSegmentationOutput, ObjectDetectionOutput

# ─────────────────────────────────────────────────────────────────────────────
# Tiny inputs to keep tests fast
# ─────────────────────────────────────────────────────────────────────────────
_B = 1
_H = 128  # multiple of 32 for stride-32 backbones
_W = 128
_C = 3
_H_ED = 256  # EfficientDet 5-level BiFPN needs ≥256 (P7 = H/128 ≥ 2)
_W_ED = 256


def _img(device: str, h: int = _H, w: int = _W, ch: int = _C) -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, ch, h, w), device=device)


def _build(factory, device: str):
    """Instantiate `factory`, switch to eval, move to device."""
    m = factory()
    m.eval()
    return m.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Shared label assignment / sampling
# ─────────────────────────────────────────────────────────────────────────────


def _one_gt_iou() -> Tensor:
    """IoU of one 10x10 ground truth against five hand-picked anchors.

    The boxes are chosen so the overlaps are exact decimals — 1.00, 0.81,
    0.49, 0.25, 0.00 — which lets the expected assignment be written down
    rather than read off a run.
    """
    from lucid.models._utils._detection import box_iou

    gt = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
    anchors = lucid.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 9.0, 9.0],
            [0.0, 0.0, 7.0, 7.0],
            [0.0, 0.0, 5.0, 5.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    return box_iou(gt, anchors)


class TestMatcher:
    def test_overlaps_are_the_expected_decimals(self) -> None:
        vals = [round(v, 4) for v in _one_gt_iou().reshape(-1).tolist()]
        assert vals == [1.0, 0.81, 0.49, 0.25, 0.0]

    def test_rpn_thresholds(self) -> None:
        """Faster R-CNN 3.1.2: >0.7 object, <0.3 background, band ignored."""
        from lucid.models._utils._detection import Matcher

        got = Matcher(0.7, 0.3)(_one_gt_iou()).tolist()
        assert got == [0, 0, -2, -1, -1]

    def test_fast_rcnn_thresholds_put_negatives_in_the_band(self) -> None:
        """Fast R-CNN 2.3 draws negatives from [0.1, 0.5), not from <0.1."""
        from lucid.models._utils._detection import Matcher

        got = Matcher(0.5, 0.1)(_one_gt_iou()).tolist()
        # 0.49 and 0.25 land in the band; only the disjoint anchor is below.
        assert got == [0, 0, -2, -2, -1]

    def test_low_quality_rescues_the_best_anchor(self) -> None:
        """The half of 3.1.2 that a plain threshold cannot express.

        With both thresholds above every overlap, nothing is a positive and
        the ground truth would train no anchor at all — which is exactly the
        case the "anchor with the highest IoU" clause exists for.
        """
        from lucid.models._utils._detection import Matcher, box_iou

        gt = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
        anchors = lucid.tensor(
            [
                [0.0, 0.0, 9.0, 9.0],
                [0.0, 0.0, 7.0, 7.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        )
        iou = box_iou(gt, anchors)
        assert Matcher(0.90, 0.85)(iou).tolist() == [-1, -1, -1]
        forced = Matcher(0.90, 0.85, allow_low_quality_matches=True)
        assert forced(iou).tolist() == [0, -1, -1]

    def test_low_quality_keeps_every_tied_anchor(self) -> None:
        from lucid.models._utils._detection import Matcher, box_iou

        gt = lucid.tensor([[0.0, 0.0, 10.0, 10.0]])
        tied = lucid.tensor(
            [
                [0.0, 0.0, 9.0, 9.0],
                [1.0, 1.0, 10.0, 10.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        )
        m = Matcher(0.90, 0.85, allow_low_quality_matches=True)
        assert m(box_iou(gt, tied)).tolist() == [0, 0, -1]

    def test_each_ground_truth_keeps_its_own_best(self) -> None:
        from lucid.models._utils._detection import Matcher, box_iou

        gt = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
        anchors = lucid.tensor([[0.0, 0.0, 9.0, 9.0], [100.0, 100.0, 109.0, 109.0]])
        m = Matcher(0.90, 0.85, allow_low_quality_matches=True)
        assert m(box_iou(gt, anchors)).tolist() == [0, 1]

    def test_empty_ground_truth_is_refused(self) -> None:
        from lucid.models._utils._detection import Matcher, box_iou

        anchors = lucid.tensor([[0.0, 0.0, 9.0, 9.0]])
        with pytest.raises(ValueError, match="no ground-truth rows"):
            Matcher(0.7, 0.3)(box_iou(lucid.zeros(0, 4), anchors))

    def test_inverted_thresholds_are_refused(self) -> None:
        from lucid.models._utils._detection import Matcher

        with pytest.raises(ValueError, match="must not exceed"):
            Matcher(0.3, 0.7)


class TestBalancedPositiveNegativeSampler:
    @staticmethod
    def _labels(n_pos: int, n_neg: int, n_ignore: int = 0) -> Tensor:
        return lucid.tensor([1] * n_pos + [0] * n_neg + [-1] * n_ignore).long()

    def test_hits_the_requested_fraction(self) -> None:
        """Fast R-CNN 2.3: 64 RoIs per image, 25% of them foreground."""
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        lucid.manual_seed(0)
        labels = self._labels(30, 200, 10)
        pos, neg = BalancedPositiveNegativeSampler(64, 0.25)(labels)
        p, n = pos.tolist(), neg.tolist()
        assert (len(p), len(n)) == (16, 48)

    def test_samples_are_disjoint_and_correctly_labelled(self) -> None:
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        lucid.manual_seed(0)
        labels = self._labels(30, 200, 10)
        flat = labels.tolist()
        pos, neg = BalancedPositiveNegativeSampler(64, 0.25)(labels)
        p, n = pos.tolist(), neg.tolist()
        assert not set(p) & set(n)
        assert all(flat[i] == 1 for i in p)
        assert all(flat[i] == 0 for i in n)

    def test_shortfall_is_backfilled_with_negatives(self) -> None:
        """3.1.3: an object-poor image still yields a full minibatch."""
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        lucid.manual_seed(0)
        pos, neg = BalancedPositiveNegativeSampler(64, 0.25)(self._labels(3, 200))
        assert (len(pos.tolist()), len(neg.tolist())) == (3, 61)

    def test_takes_what_exists_when_the_image_is_tiny(self) -> None:
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        pos, neg = BalancedPositiveNegativeSampler(64, 0.25)(self._labels(1, 2))
        assert (len(pos.tolist()), len(neg.tolist())) == (1, 2)

    def test_reproducible_under_manual_seed(self) -> None:
        from lucid.models._utils._detection import BalancedPositiveNegativeSampler

        s = BalancedPositiveNegativeSampler(64, 0.25)
        labels = self._labels(30, 200)
        lucid.manual_seed(0)
        first = s(labels)[0].tolist()
        lucid.manual_seed(0)
        assert s(labels)[0].tolist() == first


# ─────────────────────────────────────────────────────────────────────────────
# R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.rcnn import rcnn, RCNNForObjectDetection

        m = _build(rcnn, device)
        assert isinstance(m, RCNNForObjectDetection)

        proposals = [
            lucid.tensor(
                [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]],
                device=device,
            )
        ]
        x = _img(device)
        out = m(x, proposals)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[0]) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        # Self-consistency
        out2 = m(x, proposals)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Fast R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFastRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fast_rcnn import (
            fast_rcnn,
            FastRCNNForObjectDetection,
        )

        m = _build(fast_rcnn, device)
        assert isinstance(m, FastRCNNForObjectDetection)

        proposals = [
            lucid.tensor(
                [[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]],
                device=device,
            )
        ]
        x = _img(device)
        out = m(x, proposals)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.shape[0]) == 2
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x, proposals)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFastRCNNSampling:
    """§2.3's minibatch: 64 RoIs per image, 25% foreground.

    Proposals are built so the three assignment classes are populated by
    construction — 40 exact overlaps, a graded ramp through the
    ``[0.1, 0.5)`` hard-negative band, and 110 disjoint boxes that the
    paper excludes from training entirely.
    """

    @staticmethod
    def _fixture() -> tuple[object, list[Tensor], Tensor]:
        from lucid.models.vision.fast_rcnn import FastRCNNForObjectDetection
        from lucid.models.vision.fast_rcnn._config import FastRCNNConfig

        m = FastRCNNForObjectDetection(FastRCNNConfig(num_classes=3))
        props = [[0.0, 0.0, 100.0, 100.0]] * 40
        props += [[0.0, 0.0, float(sz), float(sz)] for sz in range(55, 205)]
        props += [[500.0, 500.0, 520.0, 520.0]] * 110
        return m, [lucid.tensor(props)], lucid.tensor([[0.0, 0.0, 100.0, 100.0]])

    def test_disjoint_proposals_are_ignored_not_background(self) -> None:
        """A proposal overlapping nothing is too easy to be a negative."""
        m, proposals, gt = self._fixture()
        labels, _ = m._assign_proposals(  # type: ignore[attr-defined]
            proposals[0], gt, lucid.tensor([1]).long()
        )
        lab = labels.tolist()
        assert sum(1 for v in lab if v < 0) == 110

    def test_sampled_minibatch_has_the_paper_split(self) -> None:
        m, proposals, gt = self._fixture()
        labels, _ = m._assign_proposals(  # type: ignore[attr-defined]
            proposals[0], gt, lucid.tensor([1]).long()
        )
        lab = labels.tolist()
        lucid.manual_seed(0)
        keep = m._sample_proposals(labels).tolist()  # type: ignore[attr-defined]
        assert len(keep) == 64
        assert sum(1 for i in keep if lab[i] > 0) == 16
        assert sum(1 for i in keep if lab[i] == 0) == 48
        assert sum(1 for i in keep if lab[i] < 0) == 0

    def test_sampled_negatives_lie_in_the_hard_band(self) -> None:
        from lucid.models._utils._detection import box_iou

        m, proposals, gt = self._fixture()
        labels, _ = m._assign_proposals(  # type: ignore[attr-defined]
            proposals[0], gt, lucid.tensor([1]).long()
        )
        lab = labels.tolist()
        ious = box_iou(proposals[0], gt).reshape(-1).tolist()
        lucid.manual_seed(0)
        keep = m._sample_proposals(labels).tolist()  # type: ignore[attr-defined]
        assert all(0.1 <= ious[i] < 0.5 for i in keep if lab[i] == 0)

    def test_loss_depends_on_the_sample_and_is_seed_reproducible(self) -> None:
        m, proposals, gt = self._fixture()
        targets = [{"boxes": gt, "labels": lucid.tensor([1]).long()}]
        x = lucid.randn(1, 3, 128, 128)

        lucid.manual_seed(1)
        first = float(m(x, proposals=proposals, targets=targets).loss.item())
        lucid.manual_seed(1)
        again = float(m(x, proposals=proposals, targets=targets).loss.item())
        lucid.manual_seed(2)
        other = float(m(x, proposals=proposals, targets=targets).loss.item())

        assert first == again
        # If the sampler were a no-op the seed could not move the loss.
        assert first != other


class TestFasterRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.faster_rcnn import (
            faster_rcnn_resnet50_fpn,
            FasterRCNNForObjectDetection,
        )

        # Reduce RPN top-N to keep the python NMS loop tractable on metal.
        m = _build(
            lambda: faster_rcnn_resnet50_fpn(
                num_classes=91, rpn_pre_nms_top_n=200, rpn_post_nms_top_n=100
            ),
            device,
        )
        assert isinstance(m, FasterRCNNForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        # ResNet-50-FPN: per-RoI logits (N, num_classes), per-class boxes
        # (N, num_classes, 4).
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[-1]) == 91
        assert int(out.pred_boxes.ndim) == 3
        assert int(out.pred_boxes.shape[-2]) == 91
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.proposals is not None
        assert out.loss is None

        out2 = m(x)
        assert out.logits.shape == out2.logits.shape
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

        # postprocess returns one dict per image with boxes/scores/labels.
        dets = m.postprocess(out, image_sizes=[(_H, _W)])
        assert len(dets) == _B
        assert set(dets[0].keys()) == {"boxes", "scores", "labels"}


class TestFasterRCNNTopology:
    """The rebuilt detector mirrors the reference ResNet-50-FPN key layout."""

    def test_reference_key_layout(self) -> None:
        from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn

        m = faster_rcnn_resnet50_fpn(num_classes=91)
        keys = set(m.state_dict().keys())
        assert len(keys) == 295
        # Backbone body (ResNet) + FPN.
        assert "backbone.body.conv1.weight" in keys
        assert "backbone.body.layer4.2.bn3.running_mean" in keys
        assert "backbone.fpn.inner_blocks.0.0.weight" in keys
        assert "backbone.fpn.layer_blocks.3.0.weight" in keys
        # RPN head.
        assert "rpn.head.conv.0.0.weight" in keys
        assert "rpn.head.cls_logits.weight" in keys
        assert "rpn.head.bbox_pred.weight" in keys
        # RoI heads.
        assert "roi_heads.box_head.fc6.weight" in keys
        assert "roi_heads.box_predictor.cls_score.weight" in keys
        assert "roi_heads.box_predictor.bbox_pred.weight" in keys
        # Frozen BN: no num_batches_tracked anywhere.
        assert not any(k.endswith("num_batches_tracked") for k in keys)

    def test_frozen_bn_eps_zero(self) -> None:
        from lucid.models.vision.faster_rcnn import faster_rcnn_resnet50_fpn

        m = faster_rcnn_resnet50_fpn(num_classes=91)
        # Reference detection FrozenBatchNorm2d uses eps = 0.
        assert float(m.backbone.body.bn1.eps) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestFasterRCNNTraining:
    """§3.1.2 assignment / §3.1.3 sampling / the four-term loss."""

    @staticmethod
    def _model() -> object:
        from lucid.models.vision.faster_rcnn import FasterRCNNForObjectDetection
        from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig

        return FasterRCNNForObjectDetection(
            FasterRCNNConfig(
                num_classes=4,
                backbone_layers=(1, 1, 1, 1),
                rpn_pre_nms_top_n=100,
                rpn_post_nms_top_n=50,
            )
        )

    @staticmethod
    def _targets() -> list[dict[str, Tensor]]:
        return [
            {
                "boxes": lucid.tensor(
                    [[10.0, 10.0, 60.0, 60.0], [70.0, 70.0, 110.0, 110.0]]
                ),
                "labels": lucid.tensor([1, 2]).long(),
            }
        ]

    def test_loss_is_finite_and_inference_is_unchanged(self) -> None:
        lucid.manual_seed(0)
        m = self._model()
        x = lucid.randn(1, 3, 128, 128)
        out = m(x, targets=self._targets())
        assert out.loss is not None
        assert bool(out.loss.isfinite().all().item())
        assert m(x).loss is None

    def test_gradient_reaches_both_stages(self) -> None:
        """End-to-end means the RPN trains too, not just the box head."""
        lucid.manual_seed(0)
        m = self._model()
        m(lucid.randn(1, 3, 128, 128), targets=self._targets()).loss.backward()
        trained = {
            name.split(".")[0]
            for name, prm in m.named_parameters()
            if prm.grad is not None and float(prm.grad.abs().sum().item()) > 0
        }
        assert {"backbone", "rpn", "roi_heads"} <= trained

    def test_all_four_terms_are_alive(self) -> None:
        """A term stuck at zero is a term that is not being computed."""
        lucid.manual_seed(0)
        m = self._model()
        x = lucid.randn(1, 3, 128, 128)
        targets = self._targets()

        feats = m.backbone(x)
        strides = [max(1, 128 // int(f.shape[2])) for f in feats]
        logits, deltas = m.rpn.head.forward(feats)
        anchors = m._anchor_gen.forward(feats, strides)  # type: ignore[attr-defined]
        obj, reg = m._rpn_loss(  # type: ignore[attr-defined]
            logits, deltas, anchors, targets, (128, 128)
        )
        assert float(obj.item()) > 0.0
        assert float(reg.item()) > 0.0

        props = m._rpn_proposals(  # type: ignore[attr-defined]
            logits, deltas, anchors, (128, 128)
        )
        sampled, labels, _, _ = m._select_training_samples(  # type: ignore[attr-defined]
            props, targets
        )
        # Ground truths are appended to the proposals, so an untrained RPN
        # still yields positives to learn from.
        assert sum(1 for v in labels[0].tolist() if v > 0) >= 2
        assert int(sampled[0].shape[0]) <= 512

    def test_box_regression_is_class_specific(self) -> None:
        """Only the ground-truth class channel of a positive RoI may learn."""
        from lucid.nn import Parameter

        m = self._model()
        k = 4
        deltas = Parameter(lucid.zeros(3, k * 4))
        labels = [lucid.tensor([0, 2, 1]).long()]
        reg_targets = [lucid.tensor([[0.1, 0.1, 0.1, 0.1]] * 3)]
        _, reg_loss = m._roi_loss(  # type: ignore[attr-defined]
            lucid.zeros(3, k), deltas, labels, reg_targets
        )
        reg_loss.backward()
        per_class = deltas.grad.reshape(3, k, 4).abs().sum(dim=2).tolist()
        assert [c for c, v in enumerate(per_class[0]) if v > 0] == []
        assert [c for c, v in enumerate(per_class[1]) if v > 0] == [2]
        assert [c for c, v in enumerate(per_class[2]) if v > 0] == [1]

    def test_targets_with_precomputed_proposals_is_refused(self) -> None:
        m = self._model()
        x = lucid.randn(1, 3, 128, 128)
        props = [lucid.tensor([[0.0, 0.0, 32.0, 32.0]])]
        with pytest.raises(ValueError, match="only defined when the RPN"):
            m(x, targets=self._targets(), proposals=props)

    def test_cross_boundary_flag_changes_the_labels(self) -> None:
        """3.1.2's clause is off by default; turning it on must do something."""
        from lucid.models._utils._detection import Matcher
        from lucid.models.vision.faster_rcnn import FasterRCNNForObjectDetection
        from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig

        anchors = lucid.tensor(
            [
                [0.0, 0.0, 50.0, 50.0],  # inside
                [-20.0, 0.0, 50.0, 50.0],  # crosses the left edge
            ]
        )
        gt = lucid.tensor([[0.0, 0.0, 50.0, 50.0]])
        matcher = Matcher(0.7, 0.3, allow_low_quality_matches=True)

        off = FasterRCNNForObjectDetection(
            FasterRCNNConfig(num_classes=2, backbone_layers=(1, 1, 1, 1))
        )
        on = FasterRCNNForObjectDetection(
            FasterRCNNConfig(
                num_classes=2,
                backbone_layers=(1, 1, 1, 1),
                rpn_ignore_cross_boundary=True,
            )
        )
        lab_off, _ = off._assign_anchors(  # type: ignore[attr-defined]
            anchors, gt, matcher, (128, 128)
        )
        lab_on, _ = on._assign_anchors(  # type: ignore[attr-defined]
            anchors, gt, matcher, (128, 128)
        )
        assert lab_off.tolist()[1] != -1
        assert lab_on.tolist()[1] == -1
        # the in-bounds anchor is untouched either way
        assert lab_off.tolist()[0] == lab_on.tolist()[0] == 1

    def test_overfits_a_single_example(self) -> None:
        """The loss has to be trainable, not merely finite."""
        import lucid.optim as optim
        from lucid.models.vision.faster_rcnn import FasterRCNNForObjectDetection
        from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig

        lucid.manual_seed(0)
        m = FasterRCNNForObjectDetection(
            FasterRCNNConfig(
                num_classes=3,
                backbone_layers=(1, 1, 1, 1),
                rpn_pre_nms_top_n=60,
                rpn_post_nms_top_n=30,
                rpn_batch_size_per_image=64,
                roi_batch_size_per_image=64,
            )
        )
        m.train()
        opt = optim.SGD(m.parameters(), lr=0.005, momentum=0.9)
        x = lucid.randn(1, 3, 128, 128)
        targets = [
            {
                "boxes": lucid.tensor([[16.0, 16.0, 64.0, 64.0]]),
                "labels": lucid.tensor([1]).long(),
            }
        ]
        losses: list[float] = []
        for _ in range(20):
            opt.zero_grad()
            out = m(x, targets=targets)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))

        assert all(v == v for v in losses)
        # The sampler redraws every step, so the curve is noisy; compare the
        # ends rather than asking for monotonicity.
        assert losses[-1] < losses[0]


class TestMaskRCNN:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.mask_rcnn import (
            mask_rcnn,
            MaskRCNNForObjectDetection,
        )

        # Aggressively shrink RPN proposals to keep python-NMS tractable
        m = _build(
            lambda: mask_rcnn(rpn_pre_nms_top_n=100, rpn_post_nms_top_n=50),
            device,
        )
        assert isinstance(m, MaskRCNNForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, InstanceSegmentationOutput)
        # Raw RoI-head outputs: per-proposal class logits (N, num_classes),
        # per-class boxes (N, num_classes, 4), per-class masks (N, K, 28, 28).
        assert int(out.logits.ndim) == 2
        assert int(out.logits.shape[-1]) == 91  # COCO num_classes (incl. bg)
        assert int(out.pred_boxes.ndim) == 3
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.pred_masks.ndim) == 4
        assert int(out.pred_masks.shape[1]) == 91  # one mask per class
        assert int(out.pred_masks.shape[-1]) == 28
        assert int(out.pred_masks.shape[-2]) == 28
        assert int(out.logits.shape[0]) == int(out.pred_masks.shape[0])
        assert int(out.logits.shape[0]) == int(out.pred_boxes.shape[0])
        assert out.loss is None


class TestMaskRCNNPostprocessAndStrides:
    """``postprocess`` clips as Faster R-CNN does, empty results keep the mask
    shape, and ``forward`` derives its strides the way Faster R-CNN does."""

    # Parts shared with Faster R-CNN, then the mask branch -- the smallest
    # real configuration, built through the registered factory.
    _SHARED: dict[str, object] = {
        "num_classes": 3,
        "backbone_layers": (1, 1, 1, 1),
        "fpn_out_channels": 32,
        "rpn_post_nms_top_n": 10,
    }
    _MASK: dict[str, object] = {
        "roi_representation": 64,
        "mask_hidden_channels": 16,
        "mask_num_convs": 1,
        "mask_predictor_hidden": 16,
    }

    @classmethod
    def _model(cls, **overrides: object) -> Any:
        from lucid.models import create_model

        lucid.manual_seed(0)
        kw = {**cls._SHARED, **cls._MASK, **overrides}
        return create_model("mask_rcnn_resnet50_fpn", **kw).eval()

    def test_boxes_are_clipped_to_image_sizes(self) -> None:
        model = self._model()
        with lucid.no_grad():
            out = model(_img("cpu", 64, 64))
            loose = model.postprocess(out)[0]["boxes"]
            det = model.postprocess(out, image_sizes=[(48, 40)])[0]
        # Decoding clipped only to the 64-pixel canvas, so a box reaches into
        # what would be padding around a 48 x 40 image.
        assert bool((loose[:, 0::2] > 40).any())
        boxes = det["boxes"]
        assert int(boxes.shape[0]) == int(loose.shape[0]) > 0
        assert bool((boxes[:, 0::2] <= 40).all())
        assert bool((boxes[:, 1::2] <= 48).all())
        assert bool((boxes >= 0).all())
        assert tuple(det["masks"].shape) == (int(boxes.shape[0]), 1, 48, 40)

    def test_empty_results_keep_the_mask_shape(self) -> None:
        # No class clears a 0.99 score threshold, so the image comes back
        # empty through the post-NMS branch; precomputed empty proposals
        # reach the other empty branch.
        model = self._model(score_thresh=0.99)
        with lucid.no_grad():
            out = model(_img("cpu", 64, 64))
            no_props = model(_img("cpu", 64, 64), proposals=[lucid.zeros((0, 4))])
            for o in (out, no_props):
                pasted = model.postprocess(o, image_sizes=[(48, 40)])[0]
                raw = model.postprocess(o)[0]
                assert int(pasted["boxes"].shape[0]) == 0
                assert tuple(pasted["masks"].shape) == (0, 1, 48, 40)
                assert tuple(raw["masks"].shape) == (0, 1, 28, 28)

    def test_empty_roi_masks_follow_the_configured_size(self) -> None:
        model = self._model(score_thresh=0.99, roi_mask_size=7)
        with lucid.no_grad():
            raw = model.postprocess(model(_img("cpu", 64, 64)))[0]
        assert tuple(raw["masks"].shape) == (0, 1, 14, 14)

    def test_anchor_strides_come_from_the_feature_maps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """96 is a multiple of 32 but not of 64: the pool level is 2 wide, so
        its stride is 96 // 2 = 48 where the fixed table said 64."""
        model = self._model()
        seen: list[list[int]] = []
        real = model._anchor_gen.forward

        def spy(features: list[Tensor], strides: list[int]) -> list[Tensor]:
            seen.append(list(strides))
            return real(features, strides)

        monkeypatch.setattr(model._anchor_gen, "forward", spy)
        with lucid.no_grad():
            model(_img("cpu", 96, 96))
        assert seen == [[4, 8, 16, 32, 48]]

    def test_box_branch_matches_faster_rcnn_off_the_stride_grid(self) -> None:
        """At 100 x 100 no level divides evenly (strides 4, 7, 14, 25, 50).
        The RPN and box branch are Faster R-CNN's, so with the same weights
        the proposals, logits and boxes must come out identical."""
        from lucid.models import create_model

        mask = self._model()
        lucid.manual_seed(0)
        faster = create_model(
            "faster_rcnn_resnet50_fpn", **self._SHARED, roi_representation_size=64
        ).eval()
        mask.load_state_dict(faster.state_dict(), strict=False)
        x = _img("cpu", 100, 100)
        with lucid.no_grad():
            got = mask(x)
            want = faster(x)
        assert len(got.proposals) == len(want.proposals) == 1
        assert tuple(got.proposals[0].shape) == tuple(want.proposals[0].shape)
        assert bool((got.proposals[0] == want.proposals[0]).all())
        assert bool((got.logits == want.logits).all())
        assert bool((got.pred_boxes == want.pred_boxes).all())

    def test_postprocess_realigns_with_the_strides_forward_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """99 is not a multiple of 4.  P2 has 25 rows, as it would at 100, but
        forward's strides are 3, 7, 14, 24 rather than 4, 7, 14, 25 -- only
        the input size the output carries tells the two apart."""
        import lucid.models.vision.mask_rcnn._model as mask_module

        model = self._model()
        scales: list[list[float]] = []
        real = mask_module.multiscale_roi_align

        def spy(*args: Any, **kwargs: Any) -> Tensor:
            scales.append(list(kwargs["spatial_scales"]))
            return real(*args, **kwargs)

        monkeypatch.setattr(mask_module, "multiscale_roi_align", spy)
        with lucid.no_grad():
            out = model(_img("cpu", 99, 99))
            det = model.postprocess(out)[0]
        assert out.input_size == (99, 99)
        assert int(det["boxes"].shape[0]) > 0  # so the re-alignment ran
        box_crop, mask_crop, realign = scales
        expected = [1.0 / float(s) for s in (3, 7, 14, 24)]
        assert box_crop == mask_crop == realign == expected


class TestDetectionPresetCanvas:
    """Both R-CNN presets put the image in the top-left corner of a 1344
    canvas: where ``image_sizes`` clipping measures from, and where the
    strides the detectors derive are the network's own."""

    @staticmethod
    def _presets() -> list[Any]:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        return [
            FasterRCNNResNet50FPNWeights.COCO_V1.transforms(),
            MaskRCNNResNet50FPNWeights.COCO_V1.transforms(),
        ]

    def test_canvas_is_a_multiple_of_32(self) -> None:
        import lucid.utils.transforms as T

        for tf in self._presets():
            assert tf.canvas_size == 1344
            assert tf.pad_position == "top_left"
            out = tf(T.Image(lucid.rand(3, 30, 40)))
            assert tuple(out.data.shape) == (3, 1344, 1344)

    @staticmethod
    def _far_corner(tf: Any) -> tuple[list[float], tuple[int, int]]:
        """A box in the far corner of a 300 x 500 image, after ``tf``, and
        the image's extent on the canvas as ``postprocess`` wants it."""
        import lucid.utils.transforms as T
        from lucid.utils.transforms._datatypes import to_xyxy

        boxes = T.BoundingBoxes(
            lucid.tensor([[450.0, 250.0, 500.0, 300.0]]),
            "xyxy",
            (300, 500),
            labels=lucid.tensor([1.0]),
        )
        out = tf({"image": T.Image(lucid.rand(3, 300, 500)), "boxes": boxes})
        box = [float(v) for v in to_xyxy(out["boxes"]).numpy().reshape(-1)]
        return box, tf.image_size(300, 500)

    def _survives_clipping(self, model: Any, tf: Any, masks: bool) -> None:
        box, (h, w) = self._far_corner(tf)
        # The box ends where the image does, inside the clipping extent.
        assert w - 2 < box[2] <= w and h - 2 < box[3] <= h
        k = 3
        logits = lucid.tensor([[-10.0, 10.0, -10.0]])  # class 1, surely
        pred_boxes = lucid.tensor([[box] * k])
        proposals = (lucid.tensor([box]),)
        out: Any
        if masks:
            out = InstanceSegmentationOutput(
                logits=logits,
                pred_boxes=pred_boxes,
                pred_masks=lucid.zeros((1, k, 28, 28)),
                proposals=proposals,
            )
        else:
            out = ObjectDetectionOutput(
                logits=logits, pred_boxes=pred_boxes, proposals=proposals
            )
        det = model.postprocess(out, image_sizes=[(h, w)])[0]
        assert int(det["boxes"].shape[0]) == 1
        got = [float(v) for v in det["boxes"].numpy().reshape(-1)]
        assert got == pytest.approx(box, abs=1e-3)
        if masks:
            assert tuple(det["masks"].shape) == (1, 1, h, w)

    def test_a_far_corner_box_survives_faster_rcnn_clipping(self) -> None:
        from lucid.models import create_model

        model = create_model(
            "faster_rcnn_resnet50_fpn",
            **TestMaskRCNNPostprocessAndStrides._SHARED,
            roi_representation_size=64,
        ).eval()
        self._survives_clipping(model, self._presets()[0], masks=False)

    def test_a_far_corner_box_survives_mask_rcnn_clipping(self) -> None:
        model = TestMaskRCNNPostprocessAndStrides._model()
        self._survives_clipping(model, self._presets()[1], masks=True)

    def test_a_centred_image_is_where_clipping_went_wrong(self) -> None:
        """The same box on the old centred canvas sits 272 rows below where
        ``image_sizes`` ends, so clipping flattened it."""
        import lucid.utils.transforms as T

        centred = T.Detection(min_size=800, max_size=1333, size_divisible=32)
        box, (h, _) = self._far_corner(centred)
        assert box[1] >= h

    def test_both_detectors_derive_strides_one_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mask R-CNN imports Faster R-CNN's backbone and stride rule, and
        Faster R-CNN's own forward applies that rule: at 96 the pool level is
        2 wide, a stride of 48."""
        import lucid.models.vision.faster_rcnn._model as faster_module
        import lucid.models.vision.mask_rcnn._model as mask_module
        from lucid.models import create_model

        assert mask_module._level_strides is faster_module._level_strides
        assert mask_module._BackboneWithFPN is faster_module._BackboneWithFPN
        lucid.manual_seed(0)
        faster = create_model(
            "faster_rcnn_resnet50_fpn",
            **TestMaskRCNNPostprocessAndStrides._SHARED,
            roi_representation_size=64,
        ).eval()
        seen: list[list[int]] = []
        real = faster._anchor_gen.forward

        def spy(features: list[Tensor], strides: list[int]) -> list[Tensor]:
            seen.append(list(strides))
            return real(features, strides)

        monkeypatch.setattr(faster._anchor_gen, "forward", spy)
        with lucid.no_grad():
            faster(_img("cpu", 96, 96))
        assert seen == [[4, 8, 16, 32, 48]]

    @pytest.mark.slow
    def test_strides_are_exact_on_the_preset_canvas(self) -> None:
        """The real backbone on the preset's canvas: every level's stride is
        the network's own and the P2 anchors reach the far edge.  On the old
        1333 canvas the strides were 3, 7, 15, 31, 63 and the P2 anchors
        stopped at pixel 1022."""
        from lucid.models import create_model
        from lucid.models.vision.faster_rcnn._model import _level_strides

        side = int(self._presets()[0].canvas_size)
        lucid.manual_seed(0)
        model = create_model(
            "faster_rcnn_resnet50_fpn",
            num_classes=3,
            backbone_layers=(1, 1, 1, 1),
            fpn_out_channels=8,
            roi_representation_size=16,
        ).eval()
        with lucid.no_grad():
            feats = model.backbone(lucid.zeros((1, 3, side, side)))
            strides = _level_strides(side, feats)
            p2 = model._anchor_gen.forward(feats, strides)[0]
        assert strides == [4, 8, 16, 32, 64]
        # The last P2 anchor centre sits one stride inside the edge, and the
        # widest P2 anchor spans past it.
        assert (int(feats[0].shape[3]) - 1) * strides[0] == side - strides[0]
        assert float(p2[:, 2].max().item()) >= side
        assert float(p2[:, 3].max().item()) >= side


class TestMaskRCNNTopology:
    """The rebuilt detector mirrors the reference ResNet-50-FPN key layout."""

    def test_reference_key_layout(self) -> None:
        from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn

        m = mask_rcnn_resnet50_fpn(num_classes=91)
        keys = set(m.state_dict().keys())
        # 295 shared Faster R-CNN keys + 12 mask-branch keys.
        assert len(keys) == 307
        # Shared backbone / FPN / RPN / box-head keys (identity-mapped).
        assert "backbone.body.conv1.weight" in keys
        assert "backbone.fpn.inner_blocks.0.0.weight" in keys
        assert "rpn.head.conv.0.0.weight" in keys
        assert "roi_heads.box_head.fc6.weight" in keys
        assert "roi_heads.box_predictor.cls_score.weight" in keys
        # Mask branch keys.
        for i in range(4):
            assert f"roi_heads.mask_head.{i}.0.weight" in keys
            assert f"roi_heads.mask_head.{i}.0.bias" in keys
        assert "roi_heads.mask_predictor.conv5_mask.weight" in keys
        assert "roi_heads.mask_predictor.mask_fcn_logits.weight" in keys
        # Frozen BN: no num_batches_tracked anywhere.
        assert not any(k.endswith("num_batches_tracked") for k in keys)

    def test_frozen_bn_eps_zero(self) -> None:
        from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn

        m = mask_rcnn_resnet50_fpn(num_classes=91)
        # Reference detection FrozenBatchNorm2d uses eps = 0.
        assert float(m.backbone.body.bn1.eps) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DETR
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskRCNNTraining:
    """3's ``L = L_cls + L_box + L_mask`` on top of the inherited RPN terms."""

    @staticmethod
    def _model() -> object:
        from lucid.models.vision.mask_rcnn import MaskRCNNForObjectDetection
        from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig

        return MaskRCNNForObjectDetection(
            MaskRCNNConfig(
                num_classes=4,
                backbone_layers=(1, 1, 1, 1),
                rpn_pre_nms_top_n=80,
                rpn_post_nms_top_n=40,
                rpn_batch_size_per_image=64,
                roi_batch_size_per_image=32,
            )
        )

    @staticmethod
    def _targets(with_masks: bool = True) -> list[dict[str, Tensor]]:
        side = 128
        square = [
            [1.0 if (10 <= r < 60 and 10 <= c < 60) else 0.0 for c in range(side)]
            for r in range(side)
        ]
        tgt: dict[str, Tensor] = {
            "boxes": lucid.tensor([[10.0, 10.0, 60.0, 60.0]]),
            "labels": lucid.tensor([1]).long(),
        }
        if with_masks:
            tgt["masks"] = lucid.tensor([square])
        return [tgt]

    def test_mask_term_changes_the_loss(self) -> None:
        """Dropping "masks" must train the detector alone, not silently pass."""
        lucid.manual_seed(0)
        m = self._model()
        x = lucid.randn(1, 3, 128, 128)
        with_masks = float(m(x, targets=self._targets(True)).loss.item())
        without = float(m(x, targets=self._targets(False)).loss.item())
        assert with_masks != without
        assert with_masks > without
        assert m(x).loss is None

    def test_mask_loss_is_defined_on_the_gt_class_only(self) -> None:
        """3: "other mask outputs do not contribute to the loss"."""
        from lucid.models._utils._detection import maskrcnn_loss
        from lucid.nn import Parameter

        logits = Parameter(lucid.zeros(3, 4, 6, 6))
        labels = lucid.tensor([0, 2, 1]).long()
        maskrcnn_loss(logits, labels, lucid.zeros(3, 6, 6)).backward()
        per_channel = logits.grad.abs().sum(dim=(2, 3)).tolist()
        assert [c for c, v in enumerate(per_channel[0]) if v > 0] == []
        assert [c for c, v in enumerate(per_channel[1]) if v > 0] == [2]
        assert [c for c, v in enumerate(per_channel[2]) if v > 0] == [1]

    def test_mask_targets_are_cropped_into_the_roi_frame(self) -> None:
        """3.1: the target is the RoI-aligned intersection, not the raw mask."""
        from lucid.models._utils._detection import project_masks_on_boxes

        side = 20
        quadrant = [
            [1.0 if (r < 10 and c < 10) else 0.0 for c in range(side)]
            for r in range(side)
        ]
        gt_masks = lucid.tensor([quadrant])
        proposals = lucid.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
        matched = lucid.tensor([0, 0]).long()
        target = project_masks_on_boxes(gt_masks, proposals, matched, 4)

        assert tuple(target.shape) == (2, 4, 4)
        # The proposal sitting on the square is entirely inside it; the one on
        # the opposite quadrant sees none of it.
        assert float(target[0].mean().item()) == 1.0
        assert float(target[1].mean().item()) == 0.0

    def test_no_positive_roi_yields_zero_not_nan(self) -> None:
        from lucid.models._utils._detection import maskrcnn_loss

        loss = maskrcnn_loss(
            lucid.zeros(2, 4, 6, 6), lucid.tensor([0, 0]).long(), lucid.zeros(2, 6, 6)
        )
        assert float(loss.item()) == 0.0


class TestDETR:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.detr import detr_resnet50, DETRForObjectDetection

        # Slim transformer for test speed (paper default: 6/6 layers, d=256)
        m = _build(
            lambda: detr_resnet50(
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_queries=20,
            ),
            device,
        )
        assert isinstance(m, DETRForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        cfg = m._cfg
        assert tuple(out.logits.shape) == (_B, cfg.num_queries, cfg.num_classes + 1)
        assert tuple(out.pred_boxes.shape) == (_B, cfg.num_queries, 4)
        # Box coords in [0, 1]
        assert float(out.pred_boxes.min().item()) >= 0.0
        assert float(out.pred_boxes.max().item()) <= 1.0
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestDETRResNet101:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.detr import detr_resnet101

        m = _build(
            lambda: detr_resnet101(
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_queries=20,
            ),
            device,
        )
        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        cfg = m._cfg
        assert tuple(out.logits.shape) == (_B, cfg.num_queries, cfg.num_classes + 1)


# ─────────────────────────────────────────────────────────────────────────────
# EfficientDet D0
# ─────────────────────────────────────────────────────────────────────────────


class TestEfficientDetD0:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.efficientdet import (
            efficientdet_d0,
            EfficientDetForObjectDetection,
        )

        m = _build(efficientdet_d0, device)
        assert isinstance(m, EfficientDetForObjectDetection)

        x = _img(device, h=_H_ED, w=_W_ED)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.logits.shape[1]) > 0
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

    def test_a_side_the_bifpn_cannot_fuse_is_refused_by_name(self) -> None:
        """288 px used to fail as a broadcast ShapeMismatch inside the first
        BiFPN layer (a 5x5 map met the 2x up-sample of a 3x3 one), naming
        neither the input nor the rule (LCD-554)."""
        from lucid._C import engine as _C_engine
        import lucid.models as models

        lucid.manual_seed(0)
        m = models.create_model(
            "efficientdet_d0",
            num_classes=3,
            image_size=_H_ED,
            fpn_channels=16,
            fpn_repeats=1,
            head_repeats=1,
        ).eval()
        for side, named in ((288, ("256x256", "384x384")), (248, ("256x256",))):
            with pytest.raises(ValueError) as caught:
                m(lucid.randn(1, 3, side, side))
            # The engine's ShapeMismatch is a ValueError too.
            assert not isinstance(caught.value, _C_engine.LucidError)
            message = str(caught.value)
            assert f"Got {side}x{side} pixels" in message
            for size in named:
                assert size in message
        # A side that rounds up to a multiple of 128 still answers.
        for side in (384, 250):
            with lucid.no_grad():
                out = m(lucid.randn(1, 3, side, side))
            assert bool(out.logits.isfinite().all().item())


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv1
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLO:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo, YOLOForObjectDetection

        m = _build(yolo, device)
        assert isinstance(m, YOLOForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        S = m.config.split_size
        B_boxes = m.config.num_boxes
        C = m.config.num_classes
        expected = S * S * B_boxes
        assert tuple(out.logits.shape) == (_B, expected, C)
        assert tuple(out.pred_boxes.shape) == (_B, expected, 4)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

    def test_tiny_variant(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_tiny

        m = _build(yolo_tiny, device)
        out = m(_img(device))
        assert isinstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv2
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV2:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v2, YOLOV2ForObjectDetection

        m = _build(yolo_v2, device)
        assert isinstance(m, YOLOV2ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv3
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV3:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v3, YOLOV3ForObjectDetection

        m = _build(yolo_v3, device)
        assert isinstance(m, YOLOV3ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert int(out.logits.shape[1]) == int(out.pred_boxes.shape[1])
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5

    def test_tiny_variant(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v3_tiny

        m = _build(yolo_v3_tiny, device)
        out = m(_img(device))
        assert isinstance(out, ObjectDetectionOutput)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv4
# ─────────────────────────────────────────────────────────────────────────────


class TestYOLOV4:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.yolo import yolo_v4, YOLOV4ForObjectDetection

        m = _build(yolo_v4, device)
        assert isinstance(m, YOLOV4ForObjectDetection)

        x = _img(device)
        out = m(x)
        assert isinstance(out, ObjectDetectionOutput)
        assert int(out.logits.ndim) == 3
        assert int(out.logits.shape[0]) == _B
        assert int(out.pred_boxes.shape[-1]) == 4
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-tests (device-independent)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectionRegistry:
    def test_detection_models_registered(self) -> None:
        import lucid.models as M

        det_models = M.list_models(task="object-detection")
        expected = [
            "rcnn",
            "fast_rcnn",
            "faster_rcnn",
            "faster_rcnn_resnet50_fpn",
            "mask_rcnn",
            "mask_rcnn_resnet50_fpn",
            "detr_resnet50",
            "detr_resnet101",
            "efficientdet_d0",
            "efficientdet_d7",
            "yolo",
            "yolo_tiny",
            "yolo_v2",
            "yolo_v3",
            "yolo_v3_tiny",
            "yolo_v4",
        ]
        for name in expected:
            assert name in det_models, f"{name!r} missing from registry"

    def test_create_model_api(self) -> None:
        import lucid.models as M

        m = M.create_model("detr_resnet50")
        assert m is not None


class TestRoIAlign:
    """RoIAlign correctness — sub-bin sampling_ratio averaging + boundary
    clamp must match the reference op (regression guard for the
    grid_sample bilinear fix this builds on)."""

    def test_constant_feature_constant_output(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        feat = lucid.ones(1, 4, 16, 16) * 3.0
        boxes = [lucid.tensor([[2.0, 3.0, 11.0, 13.0], [0.0, 0.0, 15.0, 15.0]])]
        for ratio in (1, 2, -1):
            out = roi_align(
                feat, boxes, output_size=7, spatial_scale=1.0, sampling_ratio=ratio
            )
            assert tuple(out.shape) == (2, 4, 7, 7)
            # A constant feature must sample to that constant everywhere.
            assert abs(float(out.max().item()) - 3.0) < 1e-5
            assert abs(float(out.min().item()) - 3.0) < 1e-5

    def test_linear_ramp_ratio_invariant(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        # On a linear (horizontal) ramp, averaging symmetric sub-bin samples
        # equals the single centre sample, so RoIAlign is sampling_ratio-
        # invariant — a correctness property of bilinear sub-bin averaging.
        ramp = lucid.tensor([[[[float(c) for c in range(8)] for _ in range(8)]]])
        boxes = [lucid.tensor([[0.0, 0.0, 7.0, 7.0]])]
        r1 = float(roi_align(ramp, boxes, 1, sampling_ratio=1).item())
        r4 = float(roi_align(ramp, boxes, 1, sampling_ratio=4).item())
        assert abs(r1 - r4) < 1e-5

    def test_subbin_averaging_runs_for_2d_grid(self) -> None:
        import lucid
        from lucid.models._utils._detection import roi_align

        # Exercise the (out_h*ry, out_w*rx) sub-bin reshape/mean path on a
        # multi-bin output with ratio>1 — shape + finiteness guard.
        feat = lucid.randn(1, 2, 20, 20)
        boxes = [lucid.tensor([[1.0, 2.0, 17.0, 18.0]])]
        out = roi_align(feat, boxes, output_size=(5, 5), sampling_ratio=2)
        assert tuple(out.shape) == (1, 2, 5, 5)
        assert bool(lucid.isfinite(out).all().item())


class TestMSDeformAttn:
    """Multi-scale deformable attention (Deformable DETR / Mask2Former) —
    composite over the fixed grid_sample; reproduces the reference op."""

    def test_output_shape(self) -> None:
        import lucid
        from lucid.models._utils._detection import multi_scale_deformable_attention

        bs, nh, hd, nq, nl, npt = 1, 8, 32, 25, 3, 4
        shapes = [(8, 8), (4, 4), (2, 2)]
        s = sum(h * w for h, w in shapes)
        value = lucid.randn(bs, s, nh, hd)
        loc = lucid.rand(bs, nq, nh, nl, npt, 2)
        aw = lucid.rand(bs, nq, nh, nl, npt)
        out = multi_scale_deformable_attention(value, shapes, loc, aw)
        assert tuple(out.shape) == (bs, nq, nh * hd)
        assert bool(lucid.isfinite(out).all().item())

    def test_constant_value_with_unit_weights(self) -> None:
        import lucid
        from lucid.models._utils._detection import multi_scale_deformable_attention

        # Constant value C + attention weights that sum to 1 over (nl*npt)
        # per (query, head) → output must be C everywhere (interior samples,
        # no boundary zero-padding).
        bs, nh, hd, nq, nl, npt = 1, 2, 4, 3, 2, 2
        shapes = [(8, 8), (4, 4)]
        s = sum(h * w for h, w in shapes)
        value = lucid.ones(bs, s, nh, hd) * 2.5
        # interior sampling locations (0.5 = centre) avoid edge zero-pad
        loc = lucid.ones(bs, nq, nh, nl, npt, 2) * 0.5
        aw = lucid.ones(bs, nq, nh, nl, npt) / float(nl * npt)
        out = multi_scale_deformable_attention(value, shapes, loc, aw)
        assert abs(float(out.max().item()) - 2.5) < 1e-5
        assert abs(float(out.min().item()) - 2.5) < 1e-5


class TestFasterRCNNWeightsEnums:
    """Static contract of the Faster R-CNN ResNet-50-FPN Weights enum."""

    def test_default_aliases_coco(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        assert (
            FasterRCNNResNet50FPNWeights.DEFAULT is FasterRCNNResNet50FPNWeights.COCO_V1
        )

    def test_entry_fields(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        e = FasterRCNNResNet50FPNWeights.COCO_V1.entry
        assert e.num_classes == 91
        # sha256 is either a real 64-hex digest or the upload placeholder.
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert "lucid-dl/faster-rcnn-resnet-50-fpn" in e.url
        assert "/COCO_V1/" in e.url
        meta = FasterRCNNResNet50FPNWeights.COCO_V1.meta
        assert meta["source"] == (
            "reference_vision/FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
        )
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == 41_755_286
        assert meta["metrics"]["COCO"]["box mAP"] == 37.0

    def test_transforms_detection_preset(self) -> None:
        from lucid.models.vision.faster_rcnn import FasterRCNNResNet50FPNWeights

        tf = FasterRCNNResNet50FPNWeights.COCO_V1.transforms()
        assert tf.to_dict()["preprocessor_type"] == "Detection"
        assert tf.max_size == 1333

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        assert "COCO_V1" in list_pretrained("faster_rcnn")
        assert "COCO_V1" in list_pretrained("faster_rcnn_resnet50_fpn")


@unittest.skipUnless(
    os.environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestFasterRCNNPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        import lucid.models as M

        m = M.create_model("faster_rcnn_resnet50_fpn", pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 256, 256))
        assert int(out.logits.shape[-1]) == 91


class TestMaskRCNNWeightsEnums:
    """Static contract of the Mask R-CNN ResNet-50-FPN Weights enum."""

    def test_default_aliases_coco(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        assert MaskRCNNResNet50FPNWeights.DEFAULT is MaskRCNNResNet50FPNWeights.COCO_V1

    def test_entry_fields(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        e = MaskRCNNResNet50FPNWeights.COCO_V1.entry
        assert e.num_classes == 91
        # sha256 is either a real 64-hex digest or the upload placeholder.
        assert len(e.sha256) == 64 or e.sha256 == "__PENDING_UPLOAD__"
        assert "lucid-dl/mask-rcnn-resnet-50-fpn" in e.url
        assert "/COCO_V1/" in e.url
        meta = MaskRCNNResNet50FPNWeights.COCO_V1.meta
        assert meta["source"] == (
            "reference_vision/MaskRCNN_ResNet50_FPN_Weights.COCO_V1"
        )
        assert meta["license"] == "bsd-3-clause"
        assert meta["num_params"] == 44_401_393
        assert meta["metrics"]["COCO"]["box mAP"] == 37.9
        assert meta["metrics"]["COCO"]["mask mAP"] == 34.6

    def test_transforms_detection_preset(self) -> None:
        from lucid.models.vision.mask_rcnn import MaskRCNNResNet50FPNWeights

        tf = MaskRCNNResNet50FPNWeights.COCO_V1.transforms()
        assert tf.to_dict()["preprocessor_type"] == "Detection"
        assert tf.max_size == 1333

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        assert "COCO_V1" in list_pretrained("mask_rcnn")
        assert "COCO_V1" in list_pretrained("mask_rcnn_resnet50_fpn")


@unittest.skipUnless(
    os.environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestMaskRCNNPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        import lucid.models as M

        m = M.create_model("mask_rcnn_resnet50_fpn", pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 256, 256))
        assert int(out.logits.shape[-1]) == 91
        assert int(out.pred_masks.shape[-1]) == 28


class TestTruncatedSVD:
    """Fast R-CNN §3.1's post-training fc6/fc7 compression."""

    def test_full_rank_reproduces_the_layer(self) -> None:
        import lucid.nn as nn

        from lucid.models._utils._common import truncated_svd_linear

        lucid.manual_seed(0)
        layer = nn.Linear(64, 32)
        x = lucid.randn(4, 64)
        exact = truncated_svd_linear(layer, rank=32)
        assert float((exact(x) - layer(x)).abs().max().item()) < 1e-4

    def test_low_rank_shrinks_the_parameter_count(self) -> None:
        import math

        import lucid.nn as nn

        from lucid.models._utils._common import truncated_svd_linear

        lucid.manual_seed(0)
        layer = nn.Linear(64, 32)
        compressed = truncated_svd_linear(layer, rank=8)
        original = 64 * 32 + 32
        built = sum(math.prod(p.shape) for p in compressed.parameters())
        # t(u + v) + bias, and it must actually be a saving.
        assert built == 8 * 64 + 32 * 8 + 32
        assert built < original

    def test_rank_beyond_the_spectrum_is_refused(self) -> None:
        import lucid.nn as nn

        from lucid.models._utils._common import truncated_svd_linear

        with pytest.raises(ValueError, match=r"rank must lie in \[1, 32\]"):
            truncated_svd_linear(nn.Linear(64, 32), rank=33)


class TestRCNNTraining:
    """Stages 1 and 3 of R-CNN's three-stage recipe.

    The proposal set is built so the two IoU thresholds select *different*
    subsets — otherwise Appendix C's tighter 0.6 would be untested.
    """

    @staticmethod
    def _fixture():
        from lucid.models.vision.rcnn import RCNNForObjectDetection
        from lucid.models.vision.rcnn._config import RCNNConfig

        model = RCNNForObjectDetection(RCNNConfig(num_classes=3))
        gt = lucid.tensor([[10.0, 10.0, 60.0, 60.0]])
        proposals = lucid.tensor(
            [
                [12.0, 12.0, 62.0, 62.0],  # IoU .855 — both stages, offset
                [10.0, 10.0, 60.0, 101.0],  # IoU .549 — stage 1 only
                [100.0, 100.0, 120.0, 120.0],  # IoU .000 — background
            ]
        )
        targets = [{"boxes": gt, "labels": lucid.tensor([1]).long()}]
        return model, gt, proposals, targets

    def test_the_fixture_spans_both_thresholds(self) -> None:
        from lucid.models._utils._detection import box_iou

        _, gt, proposals, _ = self._fixture()
        ious = [round(v, 3) for v in box_iou(gt, proposals).reshape(-1).tolist()]
        assert ious == [0.855, 0.549, 0.0]

    def test_loss_is_finite_and_inference_unchanged(self) -> None:
        lucid.manual_seed(0)
        model, _, proposals, targets = self._fixture()
        x = lucid.randn(1, 3, 128, 128)
        out = model(x, proposals=[proposals], targets=targets)
        assert out.loss is not None
        assert bool(out.loss.isfinite().all().item())
        assert model(x, proposals=[proposals]).loss is None

    def test_both_stage_terms_are_alive(self) -> None:
        lucid.manual_seed(0)
        model, _, proposals, targets = self._fixture()
        x = lucid.randn(1, 3, 128, 128)
        feats = model.conv_features(
            model._warp_proposals(x, proposals)  # type: ignore[attr-defined]
        )
        logits, deltas = model.fc_head(feats)
        cls_loss, reg_loss = model._stage_losses(  # type: ignore[attr-defined]
            logits, deltas, proposals, targets[0]
        )
        assert float(cls_loss.item()) > 0.0
        assert float(reg_loss.item()) > 0.0

    def test_appendix_c_threshold_excludes_the_loose_proposal(self) -> None:
        """Stage 3 trains on IoU > 0.6, which is stricter than stage 1's 0.5."""
        import lucid.nn as nn

        model, _, proposals, targets = self._fixture()
        deltas = nn.Parameter(lucid.zeros(3, 3 * 4))
        _, reg_loss = model._stage_losses(  # type: ignore[attr-defined]
            lucid.zeros(3, 4), deltas, proposals, targets[0]
        )
        reg_loss.backward()
        per_prop = deltas.grad.reshape(3, 3, 4).abs().sum(dim=2).tolist()
        trained = [i for i, row in enumerate(per_prop) if sum(row) > 0]
        assert trained == [0]

    def test_regressor_is_class_specific(self) -> None:
        """Appendix C trains a separate regressor per class."""
        import lucid.nn as nn

        model, _, proposals, targets = self._fixture()
        deltas = nn.Parameter(lucid.zeros(3, 3 * 4))
        _, reg_loss = model._stage_losses(  # type: ignore[attr-defined]
            lucid.zeros(3, 4), deltas, proposals, targets[0]
        )
        reg_loss.backward()
        rows = deltas.grad.reshape(3, 3, 4).abs().sum(dim=2).tolist()[0]
        # label 1 is the first foreground class, so regressor row 0.
        assert [c for c, v in enumerate(rows) if v > 0] == [0]

    def test_targets_without_proposals_is_refused(self) -> None:
        model, _, _, targets = self._fixture()
        with pytest.raises(ValueError, match="does not generate its own regions"):
            model(lucid.randn(1, 3, 128, 128), targets=targets)


class TestLinearSVM:
    """R-CNN stage 2's optimisation (the mining loop is the caller's)."""

    def test_separates_a_linearly_separable_set(self) -> None:
        from lucid.models._utils._common import fit_linear_svm

        lucid.manual_seed(0)
        pos = [[2.0 + 0.1 * i, 0.3 * (i % 3)] for i in range(20)]
        neg = [[-2.0 - 0.1 * i, 0.3 * (i % 3)] for i in range(20)]
        x = lucid.tensor(pos + neg)
        y = lucid.tensor([1.0] * 20 + [-1.0] * 20)
        w, b = fit_linear_svm(x, y, steps=300, lr=0.02)

        scores = (x @ w.reshape(-1, 1)).reshape(-1) + b
        correct = sum((v > 0) == (i < 20) for i, v in enumerate(scores.tolist()))
        assert correct == 40
        # The clusters differ only along x, so the separator must too.
        assert abs(w.tolist()[0]) > abs(w.tolist()[1])

    def test_zero_one_labels_are_refused(self) -> None:
        from lucid.models._utils._common import fit_linear_svm

        with pytest.raises(ValueError, match=r"exactly \+1 and -1"):
            fit_linear_svm(
                lucid.tensor([[1.0, 0.0], [-1.0, 0.0]]), lucid.tensor([1.0, 0.0])
            )

    def test_single_class_is_refused(self) -> None:
        from lucid.models._utils._common import fit_linear_svm

        with pytest.raises(ValueError, match="both positive and negative"):
            fit_linear_svm(
                lucid.tensor([[1.0, 0.0], [-1.0, 0.0]]), lucid.tensor([1.0, 1.0])
            )


class TestHardNegativeMining:
    """R-CNN §2.3's stage-2 loop.

    The pool is built so the hard cases are *planted*: 360 trivially
    separable negatives far from the boundary, and 40 sitting right on it.
    A loop that never finds those 40 would still look fine on the easy ones,
    so the planted set is what makes the test able to fail.
    """

    @staticmethod
    def _pool() -> tuple[Tensor, Tensor]:
        positives = lucid.tensor([[3.0 + 0.1 * i, 0.0] for i in range(20)])
        easy = [[-6.0 - 0.1 * i, 0.0] for i in range(360)]
        hard = [[0.5, 0.05 * i] for i in range(40)]
        return positives, lucid.tensor(easy + hard)

    def test_working_set_grows_then_converges(self) -> None:
        from lucid.models._utils._common import mine_hard_negatives

        lucid.manual_seed(0)
        pos, neg = self._pool()
        _, _, history = mine_hard_negatives(pos, neg, rounds=4, keep_per_round=50)
        assert history[0] == 50
        assert history[-1] > history[0]
        # It stops early once no negative scores inside the margin, so it
        # must not have used all four rounds.
        assert len(history) < 4

    def test_separates_the_planted_hard_negatives(self) -> None:
        from lucid.models._utils._common import mine_hard_negatives

        lucid.manual_seed(0)
        pos, neg = self._pool()
        w, b, _ = mine_hard_negatives(pos, neg, rounds=4, keep_per_round=50)

        neg_scores = ((neg @ w.reshape(-1, 1)).reshape(-1) + b).tolist()
        pos_scores = ((pos @ w.reshape(-1, 1)).reshape(-1) + b).tolist()
        assert all(v < 0 for v in neg_scores)
        assert all(v > 0 for v in pos_scores)

    def test_mining_beats_fitting_on_an_easy_slice(self) -> None:
        """Without mining the boundary cases are simply never seen."""
        from lucid.models._utils._common import fit_linear_svm, mine_hard_negatives

        lucid.manual_seed(0)
        pos, neg = self._pool()
        w_mined, b_mined, _ = mine_hard_negatives(pos, neg, rounds=4, keep_per_round=50)
        w_plain, b_plain = fit_linear_svm(
            lucid.cat([pos, neg[:50]], dim=0),
            lucid.tensor([1.0] * 20 + [-1.0] * 50),
        )
        hard = neg[360:]
        mined_worst = max(
            ((hard @ w_mined.reshape(-1, 1)).reshape(-1) + b_mined).tolist()
        )
        plain_worst = max(
            ((hard @ w_plain.reshape(-1, 1)).reshape(-1) + b_plain).tolist()
        )
        assert plain_worst > 0.0  # the easy-slice fit gets them wrong
        assert mined_worst < plain_worst

    def test_empty_side_is_refused(self) -> None:
        from lucid.models._utils._common import mine_hard_negatives

        with pytest.raises(ValueError, match="both positives and negatives"):
            mine_hard_negatives(lucid.zeros(0, 4), lucid.randn(10, 4))


class TestYOLOV2MultiScale:
    """YOLOv2 §2 — a new input size drawn every 10 batches."""

    @staticmethod
    def _schedule():
        from lucid.models.vision.yolo import MultiScaleResolution

        return MultiScaleResolution()

    def test_candidate_set_is_the_papers(self) -> None:
        sched = self._schedule()
        assert sched.sizes == tuple(range(320, 609, 32))
        assert sched.period == 10

    def test_size_holds_for_ten_batches_then_redraws(self) -> None:
        """The point of the period: resizing every batch would thrash the
        data pipeline, and querying twice for one batch must not shift the
        size out from under the targets."""
        sched = self._schedule()
        window = [sched.size_for(i) for i in range(10)]
        assert len(set(window)) == 1
        assert sched.size_for(3) == window[0]  # idempotent within the window
        assert sched.size_for(10) in sched.sizes

    def test_rejects_sizes_off_the_stride_grid(self) -> None:
        """A side that is not a multiple of 32 gives a fractional feature
        map, which fails far from here with a confusing shape error."""
        from lucid.models.vision.yolo import MultiScaleResolution

        for bad in ((300,), (0,), (-32,)):
            with pytest.raises(ValueError, match="multiple of 32"):
                MultiScaleResolution(bad)
        with pytest.raises(ValueError, match="at least one"):
            MultiScaleResolution(())
        with pytest.raises(ValueError, match="period"):
            MultiScaleResolution(period=0)

    def test_model_runs_at_every_candidate_size(self) -> None:
        """The schedule is only usable because the network is fully
        convolutional — check that claim rather than assume it."""
        import lucid.nn.functional as F
        from lucid.models.vision.yolo import YOLOV2ForObjectDetection
        from lucid.models.vision.yolo._v2 import YOLOV2Config

        model = YOLOV2ForObjectDetection(YOLOV2Config(num_classes=4)).eval()
        base = lucid.randn(1, 3, 320, 320)
        for side in (320, 448, 608):
            x = F.interpolate(
                base, size=(side, side), mode="bilinear", align_corners=False
            )
            out = model(x)
            assert int(out.pred_boxes.shape[1]) == (side // 32) ** 2 * 5

    def test_loss_is_not_proportional_to_the_grid(self) -> None:
        """Multi-scale training is unusable if the loss tracks the grid.

        The objectness term is evaluated at every cell, so summing it raw
        made the total scale with resolution — 69.9 at 320, 239.3 at 608 —
        and no single learning rate serves both ends of the schedule.
        Dividing the objectness pair by the cell count and the localisation
        terms by the positive count removes that, without touching the
        obj/noobj ratio that ``lambda_noobj`` exists to set.

        Stated as ``loss / cells``: it used to be near-constant across
        resolutions, which is exactly what proportional means.
        """
        import lucid.nn.functional as F
        from lucid.models.vision.yolo import YOLOV2ForObjectDetection
        from lucid.models.vision.yolo._v2 import YOLOV2Config

        lucid.manual_seed(0)
        model = YOLOV2ForObjectDetection(YOLOV2Config(num_classes=4))
        base = lucid.randn(2, 3, 416, 416)
        targets = [
            {
                "boxes": lucid.tensor([[0.2, 0.2, 0.6, 0.7]]),
                "labels": lucid.tensor([1]).long(),
            },
            {
                "boxes": lucid.tensor([[0.1, 0.3, 0.4, 0.9]]),
                "labels": lucid.tensor([2]).long(),
            },
        ]

        per_cell = []
        for side in (320, 416, 608):
            x = F.interpolate(
                base, size=(side, side), mode="bilinear", align_corners=False
            )
            out = model(x, targets=targets)
            assert out.loss is not None
            per_cell.append(float(out.loss.item()) / ((side // 32) ** 2 * 5))

        # Proportional puts this near 1 (the defect measured 1.05).  A loss
        # that ignores the grid puts it near the cell ratio 3.61, less what
        # a random network's loss varies between resolutions — seen as low
        # as 2.6 across seeds, so 3.0 only held for some of them.
        assert max(per_cell) / min(per_cell) > 2.0, per_cell

    def test_training_path_is_live_off_the_default_resolution(self) -> None:
        import lucid.nn.functional as F
        from lucid.models.vision.yolo import YOLOV2ForObjectDetection
        from lucid.models.vision.yolo._v2 import YOLOV2Config

        model = YOLOV2ForObjectDetection(YOLOV2Config(num_classes=4))
        targets = [
            {
                "boxes": lucid.tensor([[0.2, 0.2, 0.6, 0.7]]),
                "labels": lucid.tensor([1]).long(),
            }
        ]
        x = F.interpolate(
            lucid.randn(1, 3, 416, 416),
            size=(320, 320),
            mode="bilinear",
            align_corners=False,
        )
        out = model(x, targets=targets)
        assert out.loss is not None
        assert bool(out.loss.isfinite().all().item())
        out.loss.backward()
        assert float(model.pred.weight.grad.abs().max().item()) > 0.0


class TestNms:
    """``nms`` against a plain greedy pass over the same IoU values.

    ``nms`` packs the thresholded IoU matrix into 32-bit words on the device
    and walks bit sets on the host; before, it fetched each kept box's IoU
    row one ``item()`` at a time — 25 of 26 seconds of a Faster R-CNN forward.
    The sizes straddle the word boundaries.
    """

    @staticmethod
    def _reference(boxes: Tensor, scores: Tensor, thr: float) -> list[int]:
        from lucid.models._utils._detection import box_iou

        order = [int(i) for i in lucid.argsort(-scores).tolist()]
        iou = box_iou(boxes, boxes).tolist()
        dead: set[int] = set()
        keep: list[int] = []
        for a, i in enumerate(order):
            if i in dead:
                continue
            keep.append(i)
            for j in order[a + 1 :]:
                if iou[i][j] > thr:
                    dead.add(j)
        return keep

    @staticmethod
    def _boxes(n: int, device: str) -> tuple[Tensor, Tensor]:
        lucid.manual_seed(n)
        xy = lucid.rand(n, 2) * 50.0
        wh = lucid.rand(n, 2) * 30.0 + 1.0
        boxes = lucid.cat([xy, xy + wh], dim=1).to(device)
        return boxes, lucid.rand(n).to(device)

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 64, 200, 1000])
    @pytest.mark.parametrize("thr", [0.3, 0.7])
    def test_matches_the_plain_greedy_pass(
        self, device: str, n: int, thr: float
    ) -> None:
        from lucid.models._utils._detection import nms

        boxes, scores = self._boxes(n, device)
        assert nms(boxes, scores, thr).tolist() == self._reference(boxes, scores, thr)

    def test_empty_and_all_overlapping(self, device: str) -> None:
        from lucid.models._utils._detection import nms

        empty = nms(lucid.zeros(0, 4).to(device), lucid.zeros(0).to(device), 0.5)
        assert empty.tolist() == []
        same = lucid.tensor([[0.0, 0.0, 10.0, 10.0]] * 40).to(device)
        scores = lucid.arange(40).float().to(device)
        assert nms(same, scores, 0.5).tolist() == [39]
