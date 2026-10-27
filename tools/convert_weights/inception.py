"""Inception v3 weight converter — torchvision → Lucid.

torchvision's ``Inception3`` uses the historical Slim/TF block names
(``Conv2d_1a_3x3`` for the stem convs, ``Mixed_5b`` … ``Mixed_7c`` for
the Inception blocks, ``AuxLogits`` for the auxiliary head, and ``fc``
for the classifier).  Lucid's :class:`InceptionV3ForImageClassification`
uses descriptive names (``stem.N``, ``inception_a0`` …, ``reduction_a`` /
``reduction_b``, ``inception_c0`` …, ``inception_e0`` / ``inception_e1``,
``classifier``), so the converter rewrites both the top-level block name
and each branch sub-module name.

Two structural notes:

* Lucid keeps the *first* conv of each factorised branch as a fused
  ``_ConvBnReLU`` (``.conv`` / ``.bn``) but stores the trailing
  factorised convs as a bare ``nn.Conv2d`` plus a sibling BatchNorm
  named ``<branch>_bn``.  torchvision always fuses (``.conv`` / ``.bn``),
  so for those trailing branches the converter folds the ``conv`` /
  ``bn`` infix away (``…branch7x7_2.conv.weight`` → ``…branch2_b.weight``;
  ``…branch7x7_2.bn.weight`` → ``…branch2_b_bn.weight``).
* Lucid's default ``inception_v3_cls`` is built with ``aux_logits=False``
  (timm-aligned), so the entire ``AuxLogits.*`` sub-tree is dropped.

BatchNorm ``eps`` is pinned to 1e-3 on the Lucid side to match the
canonical Inception graph (see ``_model._BN_EPS``); without it the BN
denominators differ and parity drifts well above 1e-3.
"""

import dataclasses
import math

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_INCEPTION_CITATION = (
    "@inproceedings{szegedy2016rethinking,\n"
    "  title={Rethinking the Inception Architecture for Computer Vision},\n"
    "  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey "
    "and Shlens, Jon and Wojna, Zbigniew},\n"
    "  booktitle={CVPR}, year={2016}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_INCEPTION_VARIANTS: dict[str, tuple[str, str, str]] = {
    "inception_v3": ("inception_v3_cls", "lucid-dl/inception-v3", "Inception v3")
}

# Top-level block-name map: torchvision -> Lucid.
_BLOCK = {
    "Conv2d_1a_3x3": "stem.0",
    "Conv2d_2a_3x3": "stem.1",
    "Conv2d_2b_3x3": "stem.2",
    "Conv2d_3b_1x1": "stem.4",
    "Conv2d_4a_3x3": "stem.5",
    "Mixed_5b": "inception_a0",
    "Mixed_5c": "inception_a1",
    "Mixed_5d": "inception_a2",
    "Mixed_6a": "reduction_a",
    "Mixed_6b": "inception_c0",
    "Mixed_6c": "inception_c1",
    "Mixed_6d": "inception_c2",
    "Mixed_6e": "inception_c3",
    "Mixed_7a": "reduction_b",
    "Mixed_7b": "inception_e0",
    "Mixed_7c": "inception_e1",
}

# Inception-A branch map (Mixed_5b/5c/5d -> _InceptionA).
_A_BR = {
    "branch1x1": "branch1",
    "branch5x5_1": "branch2_a",
    "branch5x5_2": "branch2_b",
    "branch3x3dbl_1": "branch3_a",
    "branch3x3dbl_2": "branch3_b",
    "branch3x3dbl_3": "branch3_c",
    "branch_pool": "branch4_conv",
}

# Reduction-A branch map (Mixed_6a -> _InceptionB).
_B_BR = {
    "branch3x3": "branch1",
    "branch3x3dbl_1": "branch2_a",
    "branch3x3dbl_2": "branch2_b",
    "branch3x3dbl_3": "branch2_c",
}

# Inception-C branch map (Mixed_6b..6e -> _InceptionC).
_C_BR = {
    "branch1x1": "branch1",
    "branch7x7_1": "branch2_a",
    "branch7x7_2": "branch2_b",
    "branch7x7_3": "branch2_c",
    "branch7x7dbl_1": "branch3_a",
    "branch7x7dbl_2": "branch3_b",
    "branch7x7dbl_3": "branch3_c",
    "branch7x7dbl_4": "branch3_d",
    "branch7x7dbl_5": "branch3_e",
    "branch_pool": "branch4_conv",
}
# Lucid C-branches stored as bare conv + sibling ``<branch>_bn``.
_C_RAW = {"branch2_b", "branch2_c", "branch3_b", "branch3_c", "branch3_d", "branch3_e"}

# Reduction-B branch map (Mixed_7a -> _InceptionD).
_D_BR = {
    "branch3x3_1": "branch1_a",
    "branch3x3_2": "branch1_b",
    "branch7x7x3_1": "branch2_a",
    "branch7x7x3_2": "branch2_b",
    "branch7x7x3_3": "branch2_c",
    "branch7x7x3_4": "branch2_d",
}
_D_RAW = {"branch2_b", "branch2_c"}

# Inception-E branch map (Mixed_7b/7c -> _InceptionE).
_E_BR = {
    "branch1x1": "branch1",
    "branch3x3_1": "branch2_a",
    "branch3x3_2a": "branch2_b1",
    "branch3x3_2b": "branch2_b2",
    "branch3x3dbl_1": "branch3_a",
    "branch3x3dbl_2": "branch3_b",
    "branch3x3dbl_3a": "branch3_c1",
    "branch3x3dbl_3b": "branch3_c2",
    "branch_pool": "branch4_conv",
}
_E_RAW = {"branch2_b1", "branch2_b2", "branch3_c1", "branch3_c2"}

_STEM_BLOCKS = {"Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"}


def _branch(table: dict[str, str], tv_branch: str, src_key: str) -> str:
    """The Lucid branch for a torchvision one, refusing a branch no table names."""
    lucid_branch = table.get(tv_branch)
    if lucid_branch is None:
        raise UnknownSourceKey(src_key)
    return lucid_branch


class InceptionArch(Architecture):
    """Converter for one torchvision Inception v3 variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _INCEPTION_VARIANTS:
            raise KeyError(f"InceptionArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._weights_enum = tvm.Inception_V3_Weights
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        # Build with the aux head present (matches the published
        # checkpoint); the aux sub-tree is dropped in map_key.
        model = tvm.inception_v3(weights=self._tv_weights, aux_logits=True)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _INCEPTION_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        parts = src_key.split(".")
        blk = parts[0]

        if blk == "AuxLogits":
            return None  # aux_logits=False on the Lucid side — dropped.
        if blk == "fc":
            return "classifier." + ".".join(parts[1:])
        if blk in _STEM_BLOCKS:
            return _BLOCK[blk] + "." + ".".join(parts[1:])

        ltop = _BLOCK.get(blk)
        if ltop is None or len(parts) < 4:
            raise UnknownSourceKey(src_key)
        tv_br = parts[1]
        sub = parts[2]  # "conv" or "bn"
        rest = ".".join(parts[3:])  # weight / bias / running_mean / running_var / ...

        if blk in ("Mixed_5b", "Mixed_5c", "Mixed_5d"):
            return f"{ltop}.{_branch(_A_BR, tv_br, src_key)}.{sub}.{rest}"
        if blk == "Mixed_6a":
            return f"{ltop}.{_branch(_B_BR, tv_br, src_key)}.{sub}.{rest}"
        if blk in ("Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e"):
            lbr = _branch(_C_BR, tv_br, src_key)
            return self._map_factorised(ltop, lbr, _C_RAW, sub, rest)
        if blk == "Mixed_7a":
            lbr = _branch(_D_BR, tv_br, src_key)
            return self._map_factorised(ltop, lbr, _D_RAW, sub, rest)
        if blk in ("Mixed_7b", "Mixed_7c"):
            lbr = _branch(_E_BR, tv_br, src_key)
            return self._map_factorised(ltop, lbr, _E_RAW, sub, rest)
        raise UnknownSourceKey(src_key)

    @staticmethod
    def _map_factorised(ltop: str, lbr: str, raw_set: set[str], sub: str, rest: str) -> str:
        """Map a branch that may be a bare conv + sibling ``_bn`` in Lucid."""
        if lbr in raw_set:
            if sub == "conv":
                return f"{ltop}.{lbr}.{rest}"
            return f"{ltop}.{lbr}_bn.{rest}"  # sub == "bn"
        return f"{ltop}.{lbr}.{sub}.{rest}"

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _INCEPTION_VARIANTS[self.arch]
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

        # torchvision reports num_params for the aux-enabled build
        # (~27.2 M); the shipped Lucid head has aux_logits=False, so count
        # the actual converted model's parameters instead.
        num_params = sum(math.prod(int(d) for d in p.shape) for p in model.parameters())
        meta = {
            "num_params": num_params,
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
            model_type="inception_v3",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_INCEPTION_CITATION,
            title=title,
            paper_url="Szegedy et al., 2015 — *Rethinking the Inception "
            "Architecture for Computer Vision* (arXiv:1512.00567)",
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("inception_v3")
def _build_inception_v3(tag: str) -> Architecture:
    return InceptionArch("inception_v3", tag)
