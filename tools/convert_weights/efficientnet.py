"""EfficientNet weight converter — reference framework → Lucid.

Maps torchvision's EfficientNet ``state_dict`` onto Lucid's module
layout.  This is a pure *rename*: the parameter count matches the
reference framework exactly (every tensor has an identical shape), only
the key naming differs.

torchvision nests the 16 MBConv blocks as seven ``features.{1..7}``
stages, each holding per-stage block indices ``features.{stage}.{block}``;
Lucid flattens all blocks into a single ``features`` Sequential where the
stem occupies indices 0–2, the 16 MBConv blocks occupy ``features.3``
through ``features.18``, and the 1×1 head occupies indices 19–20:

================================================  ===================================
torchvision                                       Lucid
================================================  ===================================
``features.0.0.weight`` (stem conv)               ``features.0.weight``
``features.0.1.*`` (stem BN)                       ``features.1.*``
``features.{s}.{b}.block.*`` (MBConv s,b)         ``features.{3..18}.*``  (flattened)
``features.8.0.weight`` (head conv)               ``features.19.weight``
``features.8.1.*`` (head BN)                        ``features.20.*``
``classifier.1.{weight,bias}``                    ``classifier.{weight,bias}``
================================================  ===================================

Per MBConv block the inner layout also differs, and there are two cases
depending on whether the block has an expansion 1×1 conv (every stage
except stage 1, whose ``expand_ratio == 1`` skips expansion):

* **No expand** (stage 1, torchvision ``features.1.*``)::

      block.0.0 (dwconv) → conv.0      block.0.1 (BN) → conv.1
      block.1.fc1 (SE)   → se.fc1      block.1.fc2     → se.fc2
      block.2.0 (project)→ project_conv block.2.1 (BN) → project_bn

* **With expand** (stages 2–7)::

      block.0.0 (expand) → conv.0      block.0.1 (BN) → conv.1
      block.1.0 (dwconv) → conv.3      block.1.1 (BN) → conv.4
      block.2.fc1 (SE)   → se.fc1      block.2.fc2     → se.fc2
      block.3.0 (project)→ project_conv block.3.1 (BN) → project_bn

The flat ``features`` index for ``(stage, block)`` is assigned by walking
the torchvision stages in order (``stage`` 1→7, ``block`` ascending) and
counting from Lucid index 3.  The per-block expand/no-expand layout is
detected from the torchvision keys themselves (presence of a depthwise
``block.1.0`` vs an SE ``block.1.fc1``), so the same mapping table works
unchanged across every B-variant regardless of its scaled depth.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_EFFICIENTNET_CITATION = (
    "@inproceedings{tan2019efficientnet,\n"
    "  title={EfficientNet: Rethinking Model Scaling for Convolutional "
    "Neural Networks},\n"
    "  author={Tan, Mingxing and Le, Quoc},\n"
    "  booktitle={ICML}, year={2019}\n"
    "}"
)

_EFFICIENTNET_PAPER_URL = (
    "Tan & Le, 2019 — *EfficientNet: Rethinking Model Scaling for "
    "Convolutional Neural Networks* (arXiv:1905.11946)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_EFFICIENTNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "efficientnet_b0": (
        "efficientnet_b0_cls",
        "lucid-dl/efficientnet-b0",
        "EfficientNet-B0",
    ),
    "efficientnet_b1": (
        "efficientnet_b1_cls",
        "lucid-dl/efficientnet-b1",
        "EfficientNet-B1",
    ),
    "efficientnet_b2": (
        "efficientnet_b2_cls",
        "lucid-dl/efficientnet-b2",
        "EfficientNet-B2",
    ),
    "efficientnet_b3": (
        "efficientnet_b3_cls",
        "lucid-dl/efficientnet-b3",
        "EfficientNet-B3",
    ),
    "efficientnet_b4": (
        "efficientnet_b4_cls",
        "lucid-dl/efficientnet-b4",
        "EfficientNet-B4",
    ),
    "efficientnet_b5": (
        "efficientnet_b5_cls",
        "lucid-dl/efficientnet-b5",
        "EfficientNet-B5",
    ),
    "efficientnet_b6": (
        "efficientnet_b6_cls",
        "lucid-dl/efficientnet-b6",
        "EfficientNet-B6",
    ),
    "efficientnet_b7": (
        "efficientnet_b7_cls",
        "lucid-dl/efficientnet-b7",
        "EfficientNet-B7",
    ),
}

# torchvision builders keyed by arch.
_TV_BUILDERS = {
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
    "efficientnet_b1": (tvm.efficientnet_b1, tvm.EfficientNet_B1_Weights),
    "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights),
    "efficientnet_b3": (tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights),
    "efficientnet_b4": (tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights),
    "efficientnet_b5": (tvm.efficientnet_b5, tvm.EfficientNet_B5_Weights),
    "efficientnet_b6": (tvm.efficientnet_b6, tvm.EfficientNet_B6_Weights),
    "efficientnet_b7": (tvm.efficientnet_b7, tvm.EfficientNet_B7_Weights),
}

# torchvision feature stage index for the head 1×1 conv (after the seven
# MBConv stages 1..7).  The stem is stage 0.
_TV_HEAD_STAGE = 8


class EfficientNetArch(Architecture):
    """Converter for one torchvision EfficientNet variant + tag.

    The MBConv block flattening requires a stage-aware key map, so the
    converter pre-computes the ``(stage, block) -> flat_index`` table
    from the source ``state_dict`` itself, then resolves each key against
    it.  ``expand`` vs ``no-expand`` layout is detected per block, so the
    one table serves every B-variant.
    """

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _EFFICIENTNET_VARIANTS:
            raise KeyError(f"EfficientNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]
        self._src: dict[str, object] | None = None
        # (stage, block) -> Lucid flat features index, filled on first use.
        self._flat_index: dict[tuple[int, int], int] = {}
        # (stage, block) -> True if the block has an expansion conv.
        self._has_expand: dict[tuple[int, int], bool] = {}

    # -- source / target --------------------------------------------------

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        self._build_block_tables(sd)
        self._src = sd
        return sd

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _EFFICIENTNET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    # -- block-flattening tables ------------------------------------------

    def _build_block_tables(self, sd: dict[str, object]) -> None:
        """Derive (stage, block) → flat index + expand flags from keys.

        Walks the torchvision MBConv stages (1..7) in order; within each
        stage walks the block indices in ascending numeric order.  The
        first Lucid MBConv slot is ``features.3`` (after the 3-element
        stem at indices 0–2).
        """
        pairs: set[tuple[int, int]] = set()
        expand: dict[tuple[int, int], bool] = {}
        for key in sd:
            if not key.startswith("features."):
                continue
            parts = key.split(".")
            stage = int(parts[1])
            if stage == 0 or stage == _TV_HEAD_STAGE:
                continue  # stem / head handled separately
            block = int(parts[2])
            pairs.add((stage, block))
            # Expand layout has a depthwise conv at block.1.0 (i.e. an inner
            # "0" sub-index after block.1); no-expand has SE (fc1/fc2) there.
            if parts[3] == "block" and parts[4] == "1" and parts[5] == "0":
                expand[(stage, block)] = True

        flat = 3
        for stage, block in sorted(pairs):
            self._flat_index[(stage, block)] = flat
            self._has_expand[(stage, block)] = expand.get((stage, block), False)
            flat += 1

    # -- key map ----------------------------------------------------------

    def map_key(self, src_key: str) -> str | None:
        # Classifier head: torchvision wraps the linear in a Sequential
        # (Dropout at index 0, Linear at index 1).
        if src_key.startswith("classifier.1."):
            return "classifier." + src_key[len("classifier.1.") :]

        if not src_key.startswith("features."):
            raise UnknownSourceKey(src_key)

        parts = src_key.split(".")
        stage = int(parts[1])

        # Stem: features.0.0 (conv) → features.0 ; features.0.1 (BN) → features.1
        if stage == 0:
            sub = parts[2]
            rest = ".".join(parts[3:])
            if sub == "0":
                return f"features.0.{rest}"
            if sub == "1":
                return f"features.1.{rest}"
            raise UnknownSourceKey(src_key)

        # Head: features.8.0 (conv) → features.19 ; features.8.1 (BN) → features.20
        if stage == _TV_HEAD_STAGE:
            # The head 1x1 conv sits right after the stem (3 slots) + all
            # MBConv blocks.  Its flat index is variant-dependent (B0 has
            # 16 blocks -> head at 19/20; B1 has 23 -> 26/27, etc.), so it
            # must be derived from the block-table size, not hard-coded.
            head_conv = 3 + len(self._flat_index)
            sub = parts[2]
            rest = ".".join(parts[3:])
            if sub == "0":
                return f"features.{head_conv}.{rest}"
            if sub == "1":
                return f"features.{head_conv + 1}.{rest}"
            raise UnknownSourceKey(src_key)

        # MBConv block.
        block = int(parts[2])
        flat = self._flat_index.get((stage, block))
        if flat is None:
            raise UnknownSourceKey(src_key)
        has_expand = self._has_expand[(stage, block)]
        # parts[3] == "block"; parts[4] is the inner block index.
        inner_idx = parts[4]
        rest = ".".join(parts[5:])
        mapped = self._map_block_inner(flat, has_expand, inner_idx, rest)
        if mapped is None:
            raise UnknownSourceKey(src_key)
        return mapped

    @staticmethod
    def _map_block_inner(
        flat: int, has_expand: bool, inner_idx: str, rest: str
    ) -> str | None:
        prefix = f"features.{flat}"
        if has_expand:
            # block.0 expand conv+bn → conv.0 / conv.1
            # block.1 dwconv  conv+bn → conv.3 / conv.4
            # block.2 SE (fc1/fc2)    → se.*
            # block.3 project conv+bn → project_conv / project_bn
            if inner_idx == "0":
                return EfficientNetArch._map_conv_bn(prefix, "conv", "0", "1", rest)
            if inner_idx == "1":
                return EfficientNetArch._map_conv_bn(prefix, "conv", "3", "4", rest)
            if inner_idx == "2":
                return f"{prefix}.se.{rest}"
            if inner_idx == "3":
                return EfficientNetArch._map_project(prefix, rest)
            return None
        # No-expand (stage 1): only depthwise + SE + project.
        # block.0 dwconv conv+bn → conv.0 / conv.1
        # block.1 SE (fc1/fc2)   → se.*
        # block.2 project conv+bn → project_conv / project_bn
        if inner_idx == "0":
            return EfficientNetArch._map_conv_bn(prefix, "conv", "0", "1", rest)
        if inner_idx == "1":
            return f"{prefix}.se.{rest}"
        if inner_idx == "2":
            return EfficientNetArch._map_project(prefix, rest)
        return None

    @staticmethod
    def _map_conv_bn(
        prefix: str, seq: str, conv_idx: str, bn_idx: str, rest: str
    ) -> str | None:
        # torchvision Conv2dNormActivation: ".0.weight" (conv), ".1.*" (BN).
        sub, _, tail = rest.partition(".")
        if sub == "0":
            return f"{prefix}.{seq}.{conv_idx}.{tail}"
        if sub == "1":
            return f"{prefix}.{seq}.{bn_idx}.{tail}"
        return None

    @staticmethod
    def _map_project(prefix: str, rest: str) -> str | None:
        sub, _, tail = rest.partition(".")
        if sub == "0":
            return f"{prefix}.project_conv.{tail}"
        if sub == "1":
            return f"{prefix}.project_bn.{tail}"
        return None

    # -- spec -------------------------------------------------------------

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _EFFICIENTNET_VARIANTS[self.arch]
        import lucid.models as models

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
            model_type="efficientnet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="apache-2.0",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_EFFICIENTNET_CITATION,
            title=title,
            paper_url=_EFFICIENTNET_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("efficientnet_b0")
def _build_efficientnet_b0(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b0", tag)


@register_arch("efficientnet_b1")
def _build_efficientnet_b1(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b1", tag)


@register_arch("efficientnet_b2")
def _build_efficientnet_b2(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b2", tag)


@register_arch("efficientnet_b3")
def _build_efficientnet_b3(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b3", tag)


@register_arch("efficientnet_b4")
def _build_efficientnet_b4(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b4", tag)


@register_arch("efficientnet_b5")
def _build_efficientnet_b5(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b5", tag)


@register_arch("efficientnet_b6")
def _build_efficientnet_b6(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b6", tag)


@register_arch("efficientnet_b7")
def _build_efficientnet_b7(tag: str) -> Architecture:
    return EfficientNetArch("efficientnet_b7", tag)
