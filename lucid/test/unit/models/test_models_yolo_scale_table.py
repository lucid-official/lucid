"""YOLOv3 / YOLOv3-Tiny / YOLOv4 read one per-scale prior table from their config.

YOLOv3-Tiny predicts at two scales and shares ``YOLOV3Config`` with the
three-scale model.  Its loss used to hard-code the full model's three-scale
mask table, so the 2-entry head list was paired with a 3-entry table: a
silent mismatch (the stride-16 head scored against the full model's
medium priors, the argmax run over the full model's nine) until a lint
pass made the ``zip`` strict, after which the tiny model could not train
at all.  The table now lives in the config — anchors, a mask per scale,
a stride per scale — and the heads, the decode and the loss all read it.

These tests hold that from the public factory: both variants train and
infer on every device, the tiny table is ``yolov3-tiny.cfg``'s, its loss
assigns ground truth to its own priors, and a table whose parts disagree is
refused where it is written.
"""

import json
import math
from typing import Any

import pytest

import lucid
import lucid.models as models
import lucid.optim as optim
from lucid.models.vision.yolo import (
    YOLOV3Config,
    YOLOV3ForObjectDetection,
    YOLOV3TinyForObjectDetection,
    YOLOV4Config,
)
from lucid.models.vision.yolo._v3 import _scale_priors, _yolov3_loss

_CLASSES = 3
_SIDE = 64


def _targets(device: str) -> list[dict[str, Any]]:
    return [
        {
            "boxes": lucid.tensor(
                [[4.0, 6.0, 30.0, 40.0], [10.0, 10.0, 60.0, 62.0]], device=device
            ),
            "labels": lucid.tensor([1, 2], device=device).long(),
        },
        {
            "boxes": lucid.tensor([[0.0, 0.0, 20.0, 12.0]], device=device),
            "labels": lucid.tensor([0], device=device).long(),
        },
    ]


@pytest.mark.parametrize("name", ["yolo_v3_tiny", "yolo_v3"])
def test_trains_and_infers_from_the_factory(name: str, device: str) -> None:
    lucid.manual_seed(0)
    model = models.create_model(name, num_classes=_CLASSES).to(device)
    model.train()
    x = lucid.randn(2, 3, _SIDE, _SIDE, device=device)

    loss = model(x, targets=_targets(device)).loss
    assert loss is not None
    assert math.isfinite(float(loss.item()))
    head = model.p4_predict if name == "yolo_v3_tiny" else model.p3_predict
    weight = next(iter(head.parameters()))
    before = weight.detach().clone()
    opt = optim.SGD(model.parameters(), lr=1e-2)
    opt.zero_grad()
    loss.backward()
    assert weight.grad is not None
    assert math.isfinite(float(weight.grad.abs().sum().item()))
    opt.step()
    assert float((weight.detach() - before).abs().max().item()) > 0.0

    model.eval()
    with lucid.no_grad():
        out = model(x)
        dets = model.postprocess(out, image_sizes=[(_SIDE, _SIDE)] * 2)
    assert out.loss is None
    assert int(out.logits.shape[0]) == 2
    assert int(out.logits.shape[-1]) == _CLASSES
    assert len(dets) == 2
    assert sorted(dets[0]) == ["boxes", "labels", "scores"]
    assert dets[0]["boxes"].device.type == out.logits.device.type


def test_tiny_table_is_the_reference_cfg() -> None:
    cfg = models.create_model("yolo_v3_tiny").config
    assert isinstance(cfg, YOLOV3Config)
    # yolov3-tiny.cfg: six priors; mask 0,1,2 on the stride-16 head and
    # 3,4,5 on the stride-32 head.
    assert cfg.anchors == (
        (10.0, 14.0),
        (23.0, 27.0),
        (37.0, 58.0),
        (81.0, 82.0),
        (135.0, 169.0),
        (344.0, 319.0),
    )
    assert cfg.strides == (16, 32)
    assert cfg.anchor_masks == ((0, 1, 2), (3, 4, 5))
    # Coarse to fine, the order the heads emit.
    assert [(m, s) for m, _, s in _scale_priors(*_table(cfg))] == [
        ((3, 4, 5), 32),
        ((0, 1, 2), 16),
    ]


def _table(cfg: YOLOV3Config) -> tuple[Any, Any, Any]:
    return cfg.anchors, cfg.anchor_masks, cfg.strides


def test_full_table_is_unchanged() -> None:
    cfg = models.create_model("yolo_v3").config
    assert cfg.strides == (8, 16, 32)
    assert cfg.anchor_masks == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert len(cfg.anchors) == 9


@pytest.mark.parametrize("prior", range(6))
def test_tiny_loss_assigns_ground_truth_to_its_own_priors(prior: int) -> None:
    """A box shaped like any one of the six priors becomes a positive.

    With zero predictions every class logit is 0, so a positive anchor adds
    ``C * ln 2`` of class BCE to the localisation term, normalised by the
    one positive; the objectness term, normalised over every cell, stays
    below ``ln 2``.  A loss under ``C * ln 2`` means no anchor was assigned
    — what the full model's table did to the three small priors, whose
    best match among its nine sat on a scale the tiny network lacks.
    """
    cfg = models.create_model("yolo_v3_tiny", num_classes=_CLASSES).config
    side = 416
    channels = 3 * (5 + _CLASSES)
    raw = [
        lucid.zeros(1, channels, side // 32, side // 32),
        lucid.zeros(1, channels, side // 16, side // 16),
    ]
    w, h = cfg.anchors[prior]
    cx = cy = side / 2.0
    box = lucid.tensor([[cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]])
    loss = _yolov3_loss(raw, [{"boxes": box, "labels": lucid.tensor([0])}], cfg)
    assert float(loss.item()) >= _CLASSES * math.log(2.0) - 1e-4


class TestTableMismatchIsRefused:
    def test_stride_without_a_mask(self) -> None:
        with pytest.raises(ValueError, match=r"2 strides .* but 3 anchor masks"):
            YOLOV3Config(strides=(16, 32))

    def test_mask_past_the_anchors(self) -> None:
        with pytest.raises(ValueError, match=r"indexes outside the 6 anchors"):
            YOLOV3Config(
                anchors=((10.0, 14.0),) * 6,
                strides=(16, 32),
                anchor_masks=((0, 1, 2), (3, 4, 6)),
            )

    def test_empty_mask(self) -> None:
        with pytest.raises(ValueError, match=r"anchor mask is empty"):
            YOLOV3Config(strides=(16, 32), anchor_masks=((0, 1, 2), ()))

    def test_tiny_refuses_the_three_scale_table(self) -> None:
        with pytest.raises(ValueError, match=r"YOLOv3-Tiny predicts at 2 scales .* describes 3"):
            YOLOV3TinyForObjectDetection(YOLOV3Config(num_classes=_CLASSES))

    def test_full_model_refuses_the_two_scale_table(self) -> None:
        tiny = models.create_model("yolo_v3_tiny", num_classes=_CLASSES).config
        with pytest.raises(ValueError, match=r"YOLOv3 predicts at 3 scales .* 2"):
            YOLOV3ForObjectDetection(tiny)

    @pytest.mark.parametrize(
        "overrides",
        [{"strides": (8, 16)}, {"anchors": ((12.0, 16.0),) * 6}],
        ids=["two-strides", "six-anchors"],
    )
    def test_v4_needs_nine_anchors_and_three_strides(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"YOLOV4Config needs 9 anchors and 3"):
            YOLOV4Config(**overrides)


def test_tiny_config_survives_a_json_round_trip() -> None:
    cfg = models.create_model("yolo_v3_tiny").config
    again = YOLOV3Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    assert again.anchor_masks == ((0, 1, 2), (3, 4, 5))
