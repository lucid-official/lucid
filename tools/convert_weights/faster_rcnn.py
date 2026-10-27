"""Faster R-CNN weight converter — torchvision → Lucid.

Lucid's Faster R-CNN was rebuilt to mirror the reference
``fasterrcnn_resnet50_fpn`` submodule layout verbatim — ``backbone.body.*``
(ResNet-50 with FrozenBatchNorm2d, eps=0), ``backbone.fpn.inner_blocks /
layer_blocks``, ``rpn.head.conv.0.0`` / ``cls_logits`` / ``bbox_pred``, and
``roi_heads.box_head`` (TwoMLPHead) + ``roi_heads.box_predictor``
(FastRCNNPredictor).  A full key/shape diff shows 295 keys on each side with
an **identical** key-set and zero shape mismatches, so ``map_key`` is a pure
identity.  The FrozenBatchNorm2d buffers carry no ``num_batches_tracked``, so
there is nothing to drop.

Source: ``torchvision.models.detection.fasterrcnn_resnet50_fpn`` with the
``COCO_V1`` tag (91 classes; box AP 37.0; BSD-3 license).
"""

import dataclasses

import torchvision.models.detection as tvd

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_FRCNN_CITATION = (
    "@inproceedings{ren2015faster,\n"
    "  title={Faster R-CNN: Towards Real-Time Object Detection with "
    "Region Proposal Networks},\n"
    "  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and "
    "Sun, Jian},\n"
    "  booktitle={Advances in Neural Information Processing Systems "
    "(NeurIPS)},\n"
    "  year={2015}\n"
    "}"
)

_FRCNN_VARIANTS: dict[str, tuple[str, str, str, float]] = {
    # arch -> (lucid_factory, repo_id, title, COCO box AP)
    "faster_rcnn_resnet50_fpn": (
        "faster_rcnn_resnet50_fpn",
        "lucid-dl/faster-rcnn-resnet-50-fpn",
        "Faster R-CNN (ResNet-50-FPN)",
        37.0,
    )
}


class FasterRCNNArch(Architecture):
    """Converter for the torchvision Faster R-CNN ResNet-50-FPN variant."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _FRCNN_VARIANTS:
            raise KeyError(f"FasterRCNNArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._weights_enum = tvd.FasterRCNN_ResNet50_FPN_Weights
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = tvd.fasterrcnn_resnet50_fpn(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _FRCNN_VARIANTS[self.arch][0]
        return models.create_model(factory, num_classes=91)

    def map_key(self, src_key: str) -> str | None:
        # Identical key layout — pure identity map.
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title, box_ap = _FRCNN_VARIANTS[self.arch]
        model = models.create_model(factory_name, num_classes=91)
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        from lucid.utils.transforms import Detection

        # The preset the weights module ships (faster_rcnn/_weights.py):
        # shortest side 800, longest capped at 1333, the image in the top-left
        # corner of a 1344 square (a multiple of 32, as the reference pads its
        # batches), ImageNet normalisation (boxes ride with the image).
        preset = Detection(
            min_size=800, max_size=1333, size_divisible=32, pad_position="top_left", pad_value=0.0
        )
        preprocessing = preset.to_dict()

        tv_meta = dict(self._tv_weights.meta)
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": {"COCO": {"box mAP": box_ap}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="object-detection",
            model_type="faster_rcnn",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=91,
            config=config,
            preprocessing=preprocessing,
            citation=_FRCNN_CITATION,
            title=title,
            paper_url="Ren et al., 2015 — *Faster R-CNN: Towards Real-Time "
            "Object Detection with Region Proposal Networks* "
            "(arXiv:1506.01497)",
            categories=[],
            datasets=["coco"],
            meta=meta,
        )


@register_arch("faster_rcnn_resnet50_fpn")
def _frcnn_r50_fpn(tag: str) -> Architecture:
    return FasterRCNNArch("faster_rcnn_resnet50_fpn", tag)
