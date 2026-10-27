"""MobileNet v2 weight converter — torchvision → Lucid.

torchvision's MobileNetV2 wraps each conv+bn+relu triple in a
``Conv2dNormActivation`` sub-module and keeps the inverted-residual
blocks contiguous (``features.1`` … ``features.17``), whereas Lucid
flattens every ``Conv2dNormActivation`` into the parent ``Sequential``
and interleaves the blocks with the stem/head so the body lives at
``features.3`` … ``features.19``.  The mapping therefore needs three
pieces:

==========================================  ============================
torchvision                                 Lucid
==========================================  ============================
``features.0.0.weight``  (stem conv)        ``features.0.weight``
``features.0.1.*``       (stem bn)          ``features.1.*``
``features.N.conv.*``    (block, N=1..17)   ``features.N+2.conv.*`` (re-slotted)
``features.18.0.weight`` (head conv)        ``features.20.weight``
``features.18.1.*``      (head bn)          ``features.21.*``
``classifier.1.*``       (head linear)      ``classifier.*``
==========================================  ============================

Within a block, torchvision groups conv+bn under a
``Conv2dNormActivation`` (``conv.0.{0,1}`` for the expand, ``conv.1.{0,1}``
for the depthwise) while the linear-bottleneck projection is a bare
conv/bn (``conv.2`` / ``conv.3``).  Lucid uses one flat ``Sequential``
where the ReLU6 occupies its own slot, so the inner indices shift:

  * expand_ratio == 1 (only ``features.1``): no expand conv, so
    depthwise = ``conv.0.{0,1}`` → ``conv.{0,1}`` and projection =
    ``conv.{1,2}`` → ``conv.{3,4}``.
  * expand_ratio == 6 (all other blocks): expand = ``conv.0.{0,1}`` →
    ``conv.{0,1}``, depthwise = ``conv.1.{0,1}`` → ``conv.{3,4}``,
    projection = ``conv.{2,3}`` → ``conv.{6,7}``.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MOBILENET_V2_CITATION = (
    "@inproceedings{sandler2018mobilenetv2,\n"
    "  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},\n"
    "  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong "
    "and Zhmoginov, Andrey and Chen, Liang-Chieh},\n"
    "  booktitle={CVPR}, year={2018}\n"
    "}"
)

_MOBILENET_V2_PAPER_URL = (
    "Sandler et al., 2018 — *MobileNetV2: Inverted Residuals and "
    "Linear Bottlenecks* (arXiv:1801.04381)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_MOBILENET_V2_VARIANTS: dict[str, tuple[str, str, str]] = {
    "mobilenet_v2": ("mobilenet_v2_cls", "lucid-dl/mobilenet-v2", "MobileNet V2")
}

_TV_BUILDERS = {"mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights)}

# Inner conv-slot remap, keyed by which torchvision sub-index appears.
# expand_ratio == 6 block: conv.0.* (expand), conv.1.* (depthwise),
#   conv.2 / conv.3 (projection conv / bn).
_BLOCK6_INNER: dict[str, str] = {
    "0": "0",  # expand conv  (conv.0.0 -> conv.0)
    "1": "3",  # depthwise conv (conv.1.0 -> conv.3)
}
# expand_ratio == 1 block (features.1): no expand; conv.0.* (depthwise),
#   conv.1 / conv.2 (projection conv / bn).
_BLOCK1_PROJ_CONV = "1"  # torchvision projection conv slot for the er==1 block


class MobileNetV2Arch(Architecture):
    """Converter for the torchvision MobileNet-v2 variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _MOBILENET_V2_VARIANTS:
            raise KeyError(f"MobileNetV2Arch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _MOBILENET_V2_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Stem: features.0.0 (conv) -> features.0 ; features.0.1.* (bn) -> features.1.*
        if src_key.startswith("features.0.0."):
            return "features.0." + src_key[len("features.0.0.") :]
        if src_key.startswith("features.0.1."):
            return "features.1." + src_key[len("features.0.1.") :]

        # Head: features.18.0 (conv) -> features.20 ; features.18.1.* (bn) -> features.21.*
        if src_key.startswith("features.18.0."):
            return "features.20." + src_key[len("features.18.0.") :]
        if src_key.startswith("features.18.1."):
            return "features.21." + src_key[len("features.18.1.") :]

        # Classifier: classifier.1.* (linear) -> classifier.*  (Dropout at .0 has no params)
        if src_key.startswith("classifier.1."):
            return "classifier." + src_key[len("classifier.1.") :]

        # Inverted-residual blocks: features.N (N=1..17) -> features.N+2
        if src_key.startswith("features."):
            parts = src_key.split(".")
            # parts[0] == "features", parts[1] == block index, parts[2] == "conv"
            block_idx = int(parts[1])
            lucid_block = block_idx + 2
            rest = parts[2:]  # e.g. ["conv", "0", "0", "weight"]
            assert rest[0] == "conv", f"unexpected block sub-module {src_key!r}"

            if block_idx == 1:
                # expand_ratio == 1 block: depthwise = conv.0.{0,1} -> conv.{0,1},
                #   projection conv/bn = conv.{1,2} -> conv.{3,4}.
                inner = rest[1]
                if inner == "0":
                    # conv.0.{0|1}.<field> -> conv.{0|1}.<field>
                    sub = rest[2]  # "0" (conv) or "1" (bn)
                    tail = ".".join(rest[3:])
                    new_inner = "0" if sub == "0" else "1"
                    return f"features.{lucid_block}.conv.{new_inner}.{tail}"
                if inner == _BLOCK1_PROJ_CONV:  # "1" -> projection conv -> slot 3
                    tail = ".".join(rest[2:])
                    return f"features.{lucid_block}.conv.3.{tail}"
                if inner == "2":  # projection bn -> slot 4
                    tail = ".".join(rest[2:])
                    return f"features.{lucid_block}.conv.4.{tail}"
                raise AssertionError(f"unexpected er==1 inner slot {src_key!r}")

            # expand_ratio == 6 block.
            inner = rest[1]
            if inner in _BLOCK6_INNER:
                # Conv2dNormActivation: conv.{0|1}.{0|1}.<field>
                base = int(_BLOCK6_INNER[inner])  # 0 (expand) or 3 (depthwise)
                sub = rest[2]  # "0" (conv) or "1" (bn)
                tail = ".".join(rest[3:])
                new_inner = base + (0 if sub == "0" else 1)
                return f"features.{lucid_block}.conv.{new_inner}.{tail}"
            if inner == "2":  # projection conv -> slot 6
                tail = ".".join(rest[2:])
                return f"features.{lucid_block}.conv.6.{tail}"
            if inner == "3":  # projection bn -> slot 7
                tail = ".".join(rest[2:])
                return f"features.{lucid_block}.conv.7.{tail}"
            raise AssertionError(f"unexpected er==6 inner slot {src_key!r}")

        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _MOBILENET_V2_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        preset = ImageClassification(
            crop_size=int(tf.crop_size[0]),
            resize_size=int(tf.resize_size[0]),
            mean=tuple(float(m) for m in tf.mean),
            std=tuple(float(s) for s in tf.std),
            interpolation=str(tf.interpolation.value),
        )
        preprocessing = preset.to_dict()
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": dict(tv_meta.get("_metrics", {})),
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="mobilenet_v2",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_MOBILENET_V2_CITATION,
            title=title,
            paper_url=_MOBILENET_V2_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("mobilenet_v2")
def _build_mobilenet_v2(tag: str) -> Architecture:
    return MobileNetV2Arch("mobilenet_v2", tag)
