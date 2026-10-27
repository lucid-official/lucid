"""YOLO training losses run on the device, and detectors return int64 labels.

LCD-211.  The v1/v2 losses walked every predictor in Python and read its
box back to the host to match it against the ground truth (four ``.item()``
calls per predictor in v2), and the v3/v4 ignore mask did the same for
every anchor.  The losses are now tensor programs; the scalar references
below are the old algorithms, kept as the specification the vectorised
versions must reproduce.
"""

import ast
import math
from pathlib import Path

import pytest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection_results import (
    detection_result,
    empty_detection_result,
)
from lucid.models.vision.yolo._v1 import YOLOConfig, YOLOForObjectDetection
from lucid.models.vision.yolo._v2 import YOLOV2Config, YOLOV2ForObjectDetection
from lucid.models.vision.yolo._v3 import (
    YOLOV3Config,
    _decode_predictions,
    _ignore_mask,
    _postprocess_detections,
    _yolov3_loss,
)
from lucid.models.vision.yolo._v4 import YOLOV4Config, _yolov4_loss

_MODELS = Path(__file__).resolve().parents[3] / "models"


def _sig(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def _gt_px(target: dict[str, Tensor], hw: tuple[int, int]) -> list[tuple]:
    H, W = hw
    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()
    return [
        (x1 * W, y1 * H, x2 * W, y2 * H, int(c))
        for (x1, y1, x2, y2), c in zip(boxes, labels, strict=True)
    ]


def _targets(seed: int, n_per_image: list[int], ncls: int) -> list[dict]:
    """Normalised xyxy targets; image 0 carries a duplicated object so two
    claimants land on the same predictor, and an image may be empty."""
    lucid.manual_seed(seed)
    out = []
    for i, m in enumerate(n_per_image):
        xy = lucid.rand(m, 2) * 0.6
        wh = lucid.rand(m, 2) * 0.35 + 0.05
        boxes = lucid.cat([xy, xy + wh], dim=1)
        labels = lucid.randint(0, ncls, (m,))
        if i == 0 and m >= 2:
            boxes = lucid.cat([boxes, boxes[:1] * 0.999], dim=0)
            labels = lucid.cat([labels, (labels[:1] + 1) % ncls])
        out.append({"boxes": boxes, "labels": labels})
    return out


def _stub(cls: type, cfg: object) -> object:
    """The loss/postprocess methods read only ``self.config``."""
    names = ("_compute_loss", "_image_loss", "_decode_boxes", "postprocess")
    ns = {k: getattr(cls, k) for k in names if hasattr(cls, k)}
    ns["config"] = cfg
    return type("Stub", (), ns)()


# ── v1 ──────────────────────────────────────────────────────────────────────


def _ref_v1(raw: Tensor, targets: list[dict], cfg: YOLOConfig, hw) -> float:
    S, B = cfg.split_size, cfg.num_boxes
    H, W = hw
    cw, ch = W / S, H / S
    data = raw.tolist()
    total = 0.0
    for bi, tgt in enumerate(targets):
        cell = data[bi]
        assigned: dict[tuple[int, int, int], tuple] = {}
        for x1, y1, x2, y2, c in _gt_px(tgt, hw):
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            col, row = min(int(cx / cw), S - 1), min(int(cy / ch), S - 1)
            best_b, best = 0, -1.0
            for b in range(B):
                p = cell[row][col][b * 5 : b * 5 + 4]
                px, py = (p[0] + col) * cw, (p[1] + row) * ch
                pw, ph = p[2] ** 2 * W, p[3] ** 2 * H
                v = _iou(
                    (px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2),
                    (x1, y1, x2, y2),
                )
                if v > best:
                    best, best_b = v, b
            assigned[(row, col, best_b)] = (cx, cy, w, h, best, c)
        xy = wh = co = cn = cl = 0.0
        for row in range(S):
            for col in range(S):
                for b in range(B):
                    p = cell[row][col][b * 5 : b * 5 + 5]
                    if (row, col, b) not in assigned:
                        cn += p[4] ** 2
                        continue
                    cx, cy, w, h, v, c = assigned[(row, col, b)]
                    xy += (p[0] - (cx / cw - col)) ** 2 + (p[1] - (cy / ch - row)) ** 2
                    sw = math.sqrt(max(0.0, min(1.0, w / W)))
                    sh = math.sqrt(max(0.0, min(1.0, h / H)))
                    wh += (p[2] - sw) ** 2 + (p[3] - sh) ** 2
                    co += (p[4] - v) ** 2
                    cls = cell[row][col][B * 5 :]
                    cl += sum((s - (1.0 if k == c else 0.0)) ** 2 for k, s in enumerate(cls))
        total += cfg.lambda_coord * (xy + wh) + co + cfg.lambda_noobj * cn + cl
    return total / len(targets)


def test_v1_loss_matches_the_scalar_reference() -> None:
    cfg = YOLOConfig(num_classes=5)
    lucid.manual_seed(1)
    raw = lucid.randn(3, 7, 7, 2 * 5 + 5) * 0.5
    targets = _targets(11, [4, 0, 3], 5)
    got = YOLOForObjectDetection._compute_loss(
        _stub(YOLOForObjectDetection, cfg), raw, targets, (448, 448)
    )
    want = _ref_v1(raw, targets, cfg, (448, 448))
    assert float(got.item()) == pytest.approx(want, rel=1e-4)


# ── v2 ──────────────────────────────────────────────────────────────────────


def _ref_v2(raw: Tensor, targets: list[dict], cfg: YOLOV2Config, hw) -> float:
    A, C = cfg.num_anchors, cfg.num_classes
    H, W = hw
    fH, fW = int(raw.shape[2]), int(raw.shape[3])
    sh, sw = H / fH, W / fW
    data = raw.tolist()
    total = 0.0
    for bi, tgt in enumerate(targets):
        gts = _gt_px(tgt, hw)
        assigned: dict[tuple[int, int, int], list] = {}
        for x1, y1, x2, y2, c in gts:
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            col, row = min(int(cx / sw), fW - 1), min(int(cy / sh), fH - 1)
            best_a, best = 0, -1.0
            for a in range(A):
                aw, ah = cfg.anchors[a][0] * sw, cfg.anchors[a][1] * sh
                inter = min(w, aw) * min(h, ah)
                v = inter / max(w * h + aw * ah - inter, 1e-6)
                if v > best:
                    best, best_a = v, a
            assigned.setdefault((row, col, best_a), []).append((cx, cy, w, h, c))
        xy = wh = co = cn = cl = 0.0
        for row in range(fH):
            for col in range(fW):
                for a in range(A):
                    p = [data[bi][a * (5 + C) + k][row][col] for k in range(5 + C)]
                    bx, by = (_sig(p[0]) + col) * sw, (_sig(p[1]) + row) * sh
                    bw = cfg.anchors[a][0] * sw * math.exp(p[2])
                    bh = cfg.anchors[a][1] * sh * math.exp(p[3])
                    box = (bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2)
                    best = max((_iou(box, g[:4]) for g in gts), default=0.0)
                    if (row, col, a) not in assigned:
                        if best <= 0.6:
                            cn += _sig(p[4]) ** 2
                        continue
                    for cx, cy, w, h, c in assigned[(row, col, a)]:
                        tx = max(1e-6, min(1 - 1e-6, cx / sw - col))
                        ty = max(1e-6, min(1 - 1e-6, cy / sh - row))
                        xy += (_sig(p[0]) - tx) ** 2 + (_sig(p[1]) - ty) ** 2
                        aw, ah = cfg.anchors[a][0] * sw, cfg.anchors[a][1] * sh
                        wh += (p[2] - math.log(max(w, 1e-6) / aw)) ** 2
                        wh += (p[3] - math.log(max(h, 1e-6) / ah)) ** 2
                        co += (_sig(p[4]) - best) ** 2
                        z = sum(math.exp(v) for v in p[5:])
                        cl += sum(
                            (math.exp(v) / z - (1.0 if k == c else 0.0)) ** 2
                            for k, v in enumerate(p[5:])
                        )
        n_pos = max(1, len(gts))
        n_cells = fH * fW * A
        loc = cfg.lambda_coord * (xy + wh) + cl
        total += loc / n_pos + (co + cfg.lambda_noobj * cn) / n_cells
    return total / len(targets)


def test_v2_loss_matches_the_scalar_reference() -> None:
    cfg = YOLOV2Config(num_classes=4)
    lucid.manual_seed(2)
    raw = lucid.randn(3, 5 * (5 + 4), 6, 6) * 0.5
    targets = _targets(21, [4, 0, 2], 4)
    stub = _stub(YOLOV2ForObjectDetection, cfg)
    got = YOLOV2ForObjectDetection._compute_loss(stub, raw, targets, (192, 192))
    want = _ref_v2(raw, targets, cfg, (192, 192))
    assert float(got.item()) == pytest.approx(want, rel=1e-4)


# ── v3 / v4 ignore mask ────────────────────────────────────────────────────


def test_ignore_mask_matches_the_scalar_reference() -> None:
    anchors = [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)]
    stride, fH, fW, C = 8, 5, 6, 3
    lucid.manual_seed(3)
    raw = lucid.randn(1, 3 * (5 + C), fH, fW) * 0.7
    _, boxes, _ = _decode_predictions(raw, anchors, stride)
    # GTs copied from a few predicted boxes guarantee some overlaps > .5.
    gt = boxes[0][lucid.tensor([0, 17, 40]).long()] + 0.5
    got = _ignore_mask(boxes[0], gt, 0.5).tolist()

    gts = [tuple(g) for g in gt.tolist()]
    want = [float(max(_iou(tuple(b), g) for g in gts) > 0.5) for b in boxes[0].tolist()]
    assert got == want
    assert sum(want) >= 3
    assert _ignore_mask(boxes[0], lucid.zeros((0, 4)), 0.5).tolist() == [0.0] * len(want)


# ── no per-element host reads ───────────────────────────────────────────────


class _HostReads:
    """Counts ``.item()`` / ``.tolist()`` / ``.numpy()`` calls on tensors."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: dict[str, int] = {"item": 0, "tolist": 0, "numpy": 0}
        for name in self.calls:
            original = getattr(Tensor, name)

            def counted(t: Tensor, *a: object, _n: str = name, _f=original, **k: object):
                self.calls[_n] += 1
                return _f(t, *a, **k)

            monkeypatch.setattr(Tensor, name, counted)


def _v1_v2_reads(monkeypatch: pytest.MonkeyPatch, grid: int) -> dict[str, int]:
    lucid.manual_seed(4)
    raw1 = lucid.randn(2, grid, grid, 2 * 5 + 3)
    raw2 = lucid.randn(2, 5 * (5 + 3), grid, grid)
    targets = _targets(41, [3, 2], 3)
    with monkeypatch.context() as patch:
        reads = _HostReads(patch)
        YOLOForObjectDetection._compute_loss(
            _stub(YOLOForObjectDetection, YOLOConfig(num_classes=3, split_size=grid)),
            raw1,
            targets,
            (448, 448),
        )
        YOLOV2ForObjectDetection._compute_loss(
            _stub(YOLOV2ForObjectDetection, YOLOV2Config(num_classes=3)),
            raw2,
            targets,
            (32 * grid, 32 * grid),
        )
    return reads.calls


def test_v1_v2_losses_read_nothing_per_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only host reads left are ``one_hot``'s index-range guard (one per
    call on the CPU), so the count must not grow with the grid."""
    small = _v1_v2_reads(monkeypatch, 4)
    large = _v1_v2_reads(monkeypatch, 9)
    assert small == large
    assert small["tolist"] == 0 and small["numpy"] == 0


def _v3_v4_reads(version: str, monkeypatch: pytest.MonkeyPatch, grid: int) -> dict[str, int]:
    C, B = 3, 2
    cfg = YOLOV3Config(num_classes=C) if version == "v3" else YOLOV4Config(num_classes=C)
    loss_fn = _yolov3_loss if version == "v3" else _yolov4_loss
    lucid.manual_seed(5)
    raws = [lucid.randn(B, 3 * (5 + C), grid * s, grid * s) for s in (1, 2, 4)]
    targets = [
        {"boxes": t["boxes"] * float(32 * grid), "labels": t["labels"]}
        for t in _targets(51, [3, 2], C)
    ]
    with monkeypatch.context() as patch:
        reads = _HostReads(patch)
        loss_fn(raws, targets, cfg)
    return reads.calls


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_v3_v4_losses_read_targets_not_anchors(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target assignment indexes Python arrays, so it reads each image's
    boxes and labels once per scale — and nothing per anchor."""
    small = _v3_v4_reads(version, monkeypatch, 2)
    large = _v3_v4_reads(version, monkeypatch, 4)
    assert small == large
    assert small["item"] == 0 and small["numpy"] == 0
    assert small["tolist"] <= 2 * 2 * 3  # boxes + labels, per image, per scale


# ── postprocess labels are int64 ────────────────────────────────────────────


def _det_output(n: int, ncls: int) -> ObjectDetectionOutput:
    lucid.manual_seed(6)
    xy = lucid.rand(1, n, 2) * 50.0
    boxes = lucid.cat([xy, xy + 20.0], dim=-1)
    return ObjectDetectionOutput(
        logits=lucid.randn(1, n, ncls) * 3.0,
        pred_boxes=boxes,
        objectness=lucid.ones((1, n)),
    )


@pytest.mark.parametrize("empty", [False, True])
def test_yolo_postprocess_labels_are_int64(empty: bool) -> None:
    thresh = 1e3 if empty else 0.0  # no score clears 1e3
    out = _det_output(12, 4)
    v1 = _stub(YOLOForObjectDetection, YOLOConfig(num_classes=4, score_thresh=thresh))
    v2 = _stub(YOLOV2ForObjectDetection, YOLOV2Config(num_classes=4, score_thresh=thresh))
    results = [
        YOLOForObjectDetection.postprocess(v1, out, (96, 96))[0],
        YOLOV2ForObjectDetection.postprocess(v2, out, (96, 96))[0],
        _postprocess_detections(out, [(96, 96)], thresh, 0.5)[0],
    ]
    for r in results:
        assert r["labels"].dtype == lucid.int64
        assert (int(r["labels"].shape[0]) == 0) == empty


def test_detection_result_owns_the_label_dtype() -> None:
    r = detection_result(lucid.zeros((1, 4)), lucid.ones((1,)), lucid.tensor([2.0]))
    assert r["labels"].dtype == lucid.int64
    assert "masks" not in r
    e = empty_detection_result("cpu", (28, 28))
    assert e["labels"].dtype == lucid.int64
    assert tuple(e["masks"].shape) == (0, 1, 28, 28)


def _hand_built_results(source: str) -> list[int]:
    """Lines that build a detection result dict — one with both a
    ``"scores"`` and a ``"labels"`` key — by hand.

    Any dict literal, in any function: the empty-result helpers this batch
    deleted (``_empty_det``, ``_no_detections``) were not named
    ``postprocess`` and built exactly such dicts.  Targets carry ``boxes``
    and ``labels`` but no ``scores``, so they are not caught.
    """
    hits: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        if {"scores", "labels"} <= keys:
            hits.append(node.lineno)
    return hits


_RESULT_OWNER = _MODELS / "_utils" / "_detection_results.py"


def test_detectors_build_results_through_the_helper() -> None:
    offenders = {
        str(p.relative_to(_MODELS)): hits
        for p in sorted(_MODELS.rglob("*.py"))
        if p != _RESULT_OWNER and (hits := _hand_built_results(p.read_text()))
    }
    assert not offenders, (
        f"detection result dicts built by hand at {offenders}; use "
        "lucid.models._utils._detection_results.detection_result (or "
        "empty_detection_result) so labels stay int64"
    )


def test_the_helper_check_can_fail() -> None:
    src = (
        "def postprocess(x):\n"
        '    return {"boxes": x, "scores": x, "labels": x}\n'
        "def _empty_det(dev):\n"
        '    return {"boxes": z, "scores": z, "labels": z}\n'
        "def target(x):\n"
        '    return {"boxes": x, "labels": x}\n'
    )
    assert _hand_built_results(src) == [2, 4]
    assert _hand_built_results(_RESULT_OWNER.read_text())  # the owner does build one
