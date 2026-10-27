"""Central parity-test registry for all Lucid vision models.

Every entry is a ``ParitySpec`` that declares:
  - which Lucid factory function to call
  - the timm model name to compare against (or None → self-consistency only)
  - input shape, tolerances, test tier, and any explicit key remaps

Single source of truth
-----------------------
Adding a new model = adding one ``ParitySpec`` here.  The parametrised
test in ``test_parity_models.py`` picks these up automatically; no
other file needs to change.

Tolerance policy
-----------------
  atol = base * depth_factor

  base        = 1e-3   (float32 per-op rounding)
  depth_factor
    = 1.0   for shallow CNNs (≤ 10 layers deep)
    = 2.0   for medium networks (11–30 layers)
    = 5.0   for deep / attention networks (> 30 layers, window attn, etc.)

These are *conservative* bounds — a model matching within 1e-3 absolute
on every logit is architecturally equivalent for all practical purposes.

Tier definitions
-----------------
  'default'  — always run (< 30 M params, fast)
  'slow'     — 30–100 M params;  pytest -m slow
  'heavy'    — > 100 M params;   pytest -m heavy  (needs ≥ 8 GB RAM)
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import lucid.models as M

# ── Key transform functions (Lucid key name → timm key name) ─────────────────


def _swin_key_transform(k: str) -> str:
    """Swin Transformer: Lucid key → timm key.

    Handles:
      stages.X.downsample.norm → layers.(X+1).downsample.norm  (index offset +1)
      stages.X.downsample.proj → layers.(X+1).downsample.reduction
      stages.X  → layers.X
      rel_pos_bias → relative_position_bias_table
      mlp.0.*  → mlp.fc1.*   (Lucid Sequential MLP → timm named Linear)
      mlp.3.*  → mlp.fc2.*

    Note: downsamplers are remapped BEFORE the general stages→layers substitution
    because Lucid's patch merging for stage X sits inside stages.X and feeds into
    timm's layers.X+1 (timm places it at the downstream stage boundary).
    """

    def _remap_ds(m: re.Match[str]) -> str:
        n = int(m.group(1))
        sub = "reduction" if m.group(2) == "proj" else m.group(2)
        return f"layers.{n + 1}.downsample.{sub}."

    k = re.sub(r"stages\.(\d+)\.downsample\.(norm|proj)\.", _remap_ds, k)
    k = k.replace("stages.", "layers.")
    k = k.replace("rel_pos_bias", "relative_position_bias_table")
    k = k.replace(".mlp.0.", ".mlp.fc1.")
    k = k.replace(".mlp.3.", ".mlp.fc2.")
    return k


def _cspnet_key_transform(k: str) -> str:
    """CSPNet: Lucid key → timm key.

    The two trees are otherwise identical (the converter's ``map_key`` is the
    identity), so the only rename is the classifier head:
      classifier.*  →  head.fc.*
    """
    if k.startswith("classifier."):
        return "head.fc." + k[len("classifier.") :]
    return k


def _convnext_key_transform(k: str) -> str:
    """ConvNeXt: Lucid key → timm key.

    Handles:
      stem.conv.N  → stem.N
      stem.norm.*  → stem.1.*
      stages.X.Y.  → stages.X.blocks.Y.
      .dwconv.     → .conv_dw.
      .blocks.Y.fc1/fc2.  → .blocks.Y.mlp.fc1/fc2.
      head_norm.   → head.norm.
      downsamplers.N.norm/conv.  → stages.(N+1).downsample.0/1.
    """
    # Stem
    k = re.sub(r"stem\.conv\.(\d+)", r"stem.\1", k)
    k = k.replace("stem.norm.", "stem.1.")
    # Stage blocks
    k = re.sub(r"stages\.(\d+)\.(\d+)\.", r"stages.\1.blocks.\2.", k)
    # Layer names inside blocks
    k = k.replace(".dwconv.", ".conv_dw.")
    k = re.sub(r"\.blocks\.(\d+)\.fc([12])\.", r".blocks.\1.mlp.fc\2.", k)
    # Head norm
    k = k.replace("head_norm.", "head.norm.")

    # Downsamplers: downsamplers.N.{norm,conv}.* → stages.(N+1).downsample.{0,1}.*
    def _remap_ds(m: re.Match[str]) -> str:
        n = int(m.group(1))
        sub = "0" if m.group(2) == "norm" else "1"
        return f"stages.{n + 1}.downsample.{sub}."

    k = re.sub(r"downsamplers\.(\d+)\.(norm|conv)\.", _remap_ds, k)
    return k


def _mobilenet_v4_branch_gains(model: object) -> None:
    """MobileNet v4: give every branch gain a trained-like magnitude.

    With the norm buffers randomised, normalisation stops rescaling, and
    the reference initialisation then fails in both directions.  In the
    conv variants every UIB branch adds about its input's size again, so
    Conv-Medium / -Large answered with logits of 3e5 / 4e6 and the
    comparison skipped.  In the hybrids every layer scale starts at 1e-5 —
    including on the strided blocks that have no residual, each of which
    therefore shrinks the main path by 1e-5 — so the logits came out near
    1e-12 and an ``atol`` of 1e-3 passed anything, attention included.

    Residual branches get a gain from U(0.1, 0.3) (their layer scale when
    there is one, else the branch's last norm weight); non-residual layer
    scales get U(0.5, 1).  Every variant then answers with logits between
    about 0.5 and 20.  Seeded, so the check stays deterministic.
    """
    import lucid
    import lucid.nn as nn

    lucid.manual_seed(0)
    for stage in getattr(model, "blocks"):
        for block in stage:
            residual = bool(getattr(block, "_has_skip", False)) or hasattr(block, "attn")
            lo, hi = (0.1, 0.3) if residual else (0.5, 1.0)
            gamma = getattr(getattr(block, "layer_scale", None), "gamma", None)
            if gamma is not None:
                nn.init.uniform_(gamma, lo, hi)
            elif residual and hasattr(block, "pw_proj"):
                nn.init.uniform_(block.pw_proj.bn.weight, lo, hi)
            elif residual and hasattr(block, "bn2"):
                nn.init.uniform_(block.bn2.weight, lo, hi)


def _norm_gains(lo: float, hi: float) -> Callable[[object], None]:
    """Draw every BatchNorm weight from U(lo, hi) before the transfer.

    With the reference's norm buffers randomised, a deep plain stack
    amplifies layer after layer: Inception v3 answered with logits of
    1.25e12 and SE-ResNet-152 with 9.1e3, and the comparison skipped — for
    as long as nobody looked.  Gains a little below one keep the signal
    alive without letting it grow; the ranges were measured to land the
    reference's logits at 108 and 22 (a range lower by 0.1 already
    collapses them below 0.25).  Seeded, so the check stays deterministic.
    """

    def prepare(model: object) -> None:
        import lucid
        import lucid.nn as nn

        lucid.manual_seed(0)
        for module in model.modules():  # type: ignore[attr-defined]
            if isinstance(module, nn.BatchNorm2d) and module.weight is not None:
                nn.init.uniform_(module.weight, lo, hi)

    return prepare


def _legacy_seresnet_key_transform(k: str) -> str:
    """Legacy SE-ResNet: Lucid key → timm ``legacy_seresnet*`` key.

    The 18/34/101/152 variants use the legacy stem-pool (``legacy_pool=True``)
    and were converted from timm's ``legacy_seresnet*`` checkpoints, so they
    must be compared against that family (the modern ``seresnet*`` uses a
    different stem pool + SE reduction).  Naming deltas:
      conv1.* / bn1.*  → layer0.conv1.* / layer0.bn1.*   (stem)
      .se.             → .se_module.                      (SE block)
    The classifier head (``fc`` → ``last_linear``) is handled via key_remap.
    """
    if k.startswith("conv1."):
        return "layer0.conv1." + k[len("conv1.") :]
    if k.startswith("bn1."):
        return "layer0.bn1." + k[len("bn1.") :]
    return k.replace(".se.", ".se_module.")


# Legacy SE-ResNet classifier head: lucid ``fc`` → timm ``last_linear``
# (not covered by the auto head-remap, which only knows ``fc``/``head``/...).
_LEGACY_SERESNET_HEAD = {"fc.weight": "last_linear.weight", "fc.bias": "last_linear.bias"}


# ── Spec type ─────────────────────────────────────────────────────────────────


@dataclass
class ParitySpec:
    """Complete specification for one model parity test."""

    # Lucid factory — callable returning the *classifier* variant
    lucid_factory: Callable[[], object]

    # timm model name.  None → only self-consistency (no timm reference).
    timm_name: str | None

    # Forward-pass input shape (B, C, H, W)
    input_shape: tuple[int, ...] = (1, 3, 224, 224)

    # Numerical tolerances for logit comparison
    atol: float = 1e-3
    rtol: float = 1e-3

    # Test tier controlling which pytest marker is applied
    tier: Literal["default", "slow", "heavy"] = "default"

    # Explicit Lucid→timm key remaps (supplement auto-alignment)
    key_remap: dict[str, str] = field(default_factory=dict)

    # Non-empty → test is skipped with this explanation
    skip_reason: str = ""

    # Optional function: lucid_key → timm_key name transform (applied before
    # explicit key_remap and auto-remap).  Use for systematic renaming patterns
    # (e.g. ``stages.`` → ``layers.``) that would require many remap entries.
    key_transform: Callable[[str], str] | None = None

    # Fall back to positional weight transfer when named alignment fails to
    # cover > 50 % of keys (e.g. when our attribute names differ from timm but
    # the parameter *shapes* and *order* are identical).  Positional is less
    # robust but avoids false-negative skips for structurally-correct models.
    use_positional_fallback: bool = False

    # Optional function applied to the freshly built Lucid model *before*
    # the transfer (so the reference receives the same values).  For
    # families whose default initialisation leaves the randomised-norm
    # forward numerically meaningless — logits that explode past the
    # skip ceiling, or collapse far below ``atol`` — it sets parameters
    # to trained-like magnitudes so the comparison measures the model.
    prepare: Callable[[object], None] | None = None

    @property
    def id(self) -> str:
        """Short label for pytest parametrize IDs."""
        return self.lucid_factory.__name__.removesuffix("_cls")


# ── Registry ─────────────────────────────────────────────────────────────────


SPECS: list[ParitySpec] = [
    # ── LeNet / AlexNet / ZFNet / GoogLeNet ─────────────────────────────────
    # ``timm_name=None`` means no numeric guard runs for these.  This harness
    # compares against timm (``_run_parity`` calls ``timm.create_model``), and
    # timm ships none of them — AlexNet and ZFNet live in the reference
    # *vision* library instead, LeNet nowhere.  Guarding AlexNet/ZFNet
    # numerically would mean teaching the harness a second source, not just
    # filling in a name here.
    ParitySpec(M.lenet_5_cls, None, input_shape=(1, 1, 32, 32)),
    ParitySpec(M.alexnet_cls, None, tier="slow"),
    ParitySpec(M.zfnet_cls, None, tier="slow"),
    ParitySpec(M.googlenet_cls, None),
    # ── VGG — heavy (> 100 M) ────────────────────────────────────────────────
    # Lucid names intermediate FC layers fc6/fc7; timm uses pre_logits.fc1/fc2.
    # The final Linear auto-remaps via _AUTO_HEAD_REMAP (classifier.* → head.fc.*).
    ParitySpec(
        M.vgg_11_cls,
        "vgg11",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_11_bn_cls,
        "vgg11_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_13_cls,
        "vgg13",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_13_bn_cls,
        "vgg13_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_16_cls,
        "vgg16",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_16_bn_cls,
        "vgg16_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_19_cls,
        "vgg19",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_19_bn_cls,
        "vgg19_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    # ── ResNet ────────────────────────────────────────────────────────────────
    # Lucid uses stem.0/stem.1 vs timm's conv1/bn1 — positional order matches.
    ParitySpec(M.resnet_18_cls, "resnet18", use_positional_fallback=True),
    ParitySpec(M.resnet_34_cls, "resnet34", use_positional_fallback=True),
    ParitySpec(M.resnet_50_cls, "resnet50", tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnet_101_cls, "resnet101", tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnet_152_cls, "resnet152", tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnet_200_cls, None, tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnet_269_cls, None, tier="slow", use_positional_fallback=True),
    ParitySpec(M.wide_resnet_50_cls, "wide_resnet50_2", tier="slow", use_positional_fallback=True),
    ParitySpec(
        M.wide_resnet_101_cls, "wide_resnet101_2", tier="slow", use_positional_fallback=True
    ),
    # ── ResNeXt ───────────────────────────────────────────────────────────────
    ParitySpec(
        M.resnext_50_32x4d_cls, "resnext50_32x4d", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.resnext_101_32x4d_cls, "resnext101_32x4d", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.resnext_101_32x8d_cls, "resnext101_32x8d", tier="slow", use_positional_fallback=True
    ),
    # ── SE-ResNet ─────────────────────────────────────────────────────────────
    # 18/34/101/152 use the legacy stem-pool (converted from timm
    # ``legacy_seresnet*``) → compare against that family, not modern
    # ``seresnet*`` (different stem pool + SE reduction).  50 keeps the
    # modern ``seresnet50`` reference (it is ``legacy_pool=False``).
    ParitySpec(
        M.se_resnet_18_cls,
        "legacy_seresnet18",
        key_transform=_legacy_seresnet_key_transform,
        key_remap=_LEGACY_SERESNET_HEAD,
    ),
    ParitySpec(
        M.se_resnet_34_cls,
        "legacy_seresnet34",
        key_transform=_legacy_seresnet_key_transform,
        key_remap=_LEGACY_SERESNET_HEAD,
    ),
    ParitySpec(M.se_resnet_50_cls, "seresnet50", tier="slow", use_positional_fallback=True),
    ParitySpec(
        M.se_resnet_101_cls,
        "legacy_seresnet101",
        tier="slow",
        key_transform=_legacy_seresnet_key_transform,
        key_remap=_LEGACY_SERESNET_HEAD,
    ),
    ParitySpec(
        M.se_resnet_152_cls,
        "legacy_seresnet152",
        tier="slow",
        key_transform=_legacy_seresnet_key_transform,
        key_remap=_LEGACY_SERESNET_HEAD,
        prepare=_norm_gains(0.8, 1.0),
    ),
    # ── SK-ResNet / SK-ResNeXt ────────────────────────────────────────────────
    # timm 1.0 only has skresnet18/34/50/50d and skresnext50_32x4d.
    # sk_resnet_101 has no timm counterpart → self-consistency only.
    ParitySpec(M.sk_resnet_18_cls, "skresnet18", use_positional_fallback=True),
    ParitySpec(M.sk_resnet_34_cls, "skresnet34", use_positional_fallback=True),
    # sk_resnet_50 is compared against nothing on purpose.  Eq. (4)'s
    # floor L = 32 gives it 32-wide attention at every stage; timm's
    # skresnet50 has no floor and runs 8/8/16/32, so 84 of its keys can
    # never line up and the spec had been skipping on "alignment
    # incomplete (84% coverage)" -- indefinitely, and quietly, which is
    # indistinguishable in a green run from a comparison that passed.
    #
    # Unlike 18/34 there is no checkpoint forcing the question: that pair
    # matched the reference because the weights they ship were trained
    # against it, and this one ships none.  Keeping the paper's geometry
    # and saying the comparison is not available beats pretending it
    # might run one day.
    ParitySpec(M.sk_resnet_50_cls, None, tier="slow"),
    ParitySpec(M.sk_resnet_101_cls, None, tier="slow"),
    ParitySpec(
        M.sk_resnext_50_32x4d_cls, "skresnext50_32x4d", tier="slow", use_positional_fallback=True
    ),
    # ── ResNeSt ───────────────────────────────────────────────────────────────
    ParitySpec(M.resnest_14_cls, None),
    ParitySpec(M.resnest_26_cls, None),
    ParitySpec(M.resnest_50_cls, "resnest50d", tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnest_101_cls, "resnest101e", tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnest_200_cls, None, tier="slow"),
    ParitySpec(M.resnest_269_cls, None, tier="slow"),
    # ── DenseNet ──────────────────────────────────────────────────────────────
    # timm DenseNet uses `classifier` head → exact name match (no remap).
    # Internal BN key order in transition layers differs from timm; named
    # alignment gets close but traversal order also differs, so both named
    # and positional transfer may be incomplete.  Accept partial match.
    ParitySpec(M.densenet_121_cls, "densenet121"),
    ParitySpec(M.densenet_169_cls, "densenet169"),
    ParitySpec(M.densenet_201_cls, "densenet201"),
    # DenseNet-161 is the only variant with growth_rate != 32 and
    # num_init_features != 64, so it is the only one that exercises the
    # 4*k bottleneck width (192) and the 2k stem arithmetic -- and it
    # ships a converted checkpoint.
    ParitySpec(M.densenet_161_cls, "densenet161"),
    ParitySpec(M.densenet_264_cls, None, tier="slow"),  # no timm equivalent
    # ── Inception ─────────────────────────────────────────────────────────────
    # Inception v3: attribute names differ (e.g. Conv2dNormActivation sub-modules
    # vs flat names in timm).  Positional fallback handles this when shapes match.
    ParitySpec(
        M.inception_v3_cls,
        "inception_v3",
        input_shape=(1, 3, 299, 299),
        use_positional_fallback=True,
        prepare=_norm_gains(0.4, 0.7),
    ),
    ParitySpec(
        M.inception_resnet_v2_cls, "inception_resnet_v2", input_shape=(1, 3, 299, 299), tier="slow"
    ),
    # Inception-v4 mirrors timm's ``features.N`` / ``last_linear`` layout
    # key-for-key (896 tensors), so the named transfer is complete.
    ParitySpec(M.inception_v4_cls, "inception_v4", input_shape=(1, 3, 299, 299), tier="slow"),
    # ── Xception ──────────────────────────────────────────────────────────────
    ParitySpec(
        M.xception_cls,
        "legacy_xception",  # timm deprecated name→legacy_xception
        input_shape=(1, 3, 299, 299),
        key_remap={"classifier.1.weight": "fc.weight", "classifier.1.bias": "fc.bias"},
        use_positional_fallback=True,
    ),
    # ── MobileNet v1 — no timm exact equivalent ───────────────────────────────
    ParitySpec(M.mobilenet_cls, None),
    ParitySpec(M.mobilenet_075_cls, None),
    ParitySpec(M.mobilenet_050_cls, None),
    ParitySpec(M.mobilenet_025_cls, None),
    # ── MobileNet v2 ──────────────────────────────────────────────────────────
    # timm head: classifier.weight (exact match).  Feature block names differ
    # from timm's MBConv naming → positional fallback for body alignment.
    ParitySpec(M.mobilenet_v2_cls, "mobilenetv2_100", use_positional_fallback=True),
    ParitySpec(M.mobilenet_v2_075_cls, "mobilenetv2_075", use_positional_fallback=True),
    # ── MobileNet v3 ──────────────────────────────────────────────────────────
    # No head remap: Lucid's classifier is a 6-element Sequential, so its head
    # keys are ``classifier.1.*`` (conv head) and ``classifier.5.*`` (linear)
    # — runtime-verified.  The remap that used to sit here named
    # ``classifier.weight`` on the left, which this model does not have, so it
    # matched nothing and the positional fallback was doing all the work
    # anyway.  An inert entry is worse than none: it reads as though the head
    # were explicitly aligned.
    ParitySpec(M.mobilenet_v3_large_cls, "mobilenetv3_large_100", use_positional_fallback=True),
    ParitySpec(M.mobilenet_v3_small_cls, "mobilenetv3_small_100", use_positional_fallback=True),
    # ── MobileNet v4 ──────────────────────────────────────────────────────────
    # The module tree mirrors timm's ``mobilenetv4_*`` key-for-key (conv_stem,
    # blocks.S.B.{dw_start,pw_exp,dw_mid,pw_proj,attn,...}, conv_head,
    # norm_head, classifier), so the named transfer is complete and no
    # positional fallback is allowed to paper over a mismatch.
    # ``prepare`` — see ``_mobilenet_v4_branch_gains``.
    ParitySpec(
        M.mobilenet_v4_conv_small_cls, "mobilenetv4_conv_small", prepare=_mobilenet_v4_branch_gains
    ),
    ParitySpec(
        M.mobilenet_v4_conv_medium_cls,
        "mobilenetv4_conv_medium",
        tier="slow",
        prepare=_mobilenet_v4_branch_gains,
    ),
    ParitySpec(
        M.mobilenet_v4_conv_large_cls,
        "mobilenetv4_conv_large",
        tier="slow",
        prepare=_mobilenet_v4_branch_gains,
    ),
    ParitySpec(
        M.mobilenet_v4_hybrid_medium_cls,
        "mobilenetv4_hybrid_medium",
        tier="slow",
        prepare=_mobilenet_v4_branch_gains,
    ),
    ParitySpec(
        M.mobilenet_v4_hybrid_large_cls,
        "mobilenetv4_hybrid_large",
        tier="slow",
        prepare=_mobilenet_v4_branch_gains,
    ),
    # ── EfficientNet — head: classifier.weight (exact match) ─────────────────
    # Lucid uses features.N vs timm's blocks.N.M — positional fallback for body.
    ParitySpec(M.efficientnet_b0_cls, "efficientnet_b0", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b1_cls, "efficientnet_b1", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b2_cls, "efficientnet_b2", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b3_cls, "efficientnet_b3", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b4_cls, "efficientnet_b4", tier="slow", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b5_cls, "efficientnet_b5", tier="slow", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b6_cls, "efficientnet_b6", tier="slow", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b7_cls, "efficientnet_b7", tier="slow", use_positional_fallback=True),
    # ── ViT ───────────────────────────────────────────────────────────────────
    # head: head.weight (auto-remapped)
    # Transformer accumulation → slightly looser tolerance
    ParitySpec(M.vit_base_16_cls, "vit_base_patch16_224", tier="slow", atol=5e-3),
    ParitySpec(M.vit_base_32_cls, "vit_base_patch32_224", tier="slow", atol=5e-3),
    ParitySpec(M.vit_large_16_cls, "vit_large_patch16_224", tier="heavy", atol=5e-3),
    ParitySpec(M.vit_large_32_cls, "vit_large_patch32_224", tier="heavy", atol=5e-3),
    ParitySpec(M.vit_huge_14_cls, None, tier="heavy"),  # 632 M — no timm parity
    # ── Swin Transformer ──────────────────────────────────────────────────────
    # Lucid: stages.X  vs  timm: layers.X  → key_transform handles this.
    # Relative position bias accumulation → looser tolerance (atol=2e-2).
    ParitySpec(
        M.swin_tiny_cls,
        "swin_tiny_patch4_window7_224",
        atol=2e-2,
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_small_cls,
        "swin_small_patch4_window7_224",
        atol=2e-2,
        tier="slow",
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_base_cls,
        "swin_base_patch4_window7_224",
        atol=2e-2,
        tier="heavy",
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_large_cls,
        "swin_large_patch4_window7_224",
        atol=2e-2,
        tier="heavy",
        key_transform=_swin_key_transform,
    ),
    # ── ConvNeXt ──────────────────────────────────────────────────────────────
    # Lucid: stem.conv.N / stem.norm / stages.X.Y
    # timm:  stem.N      / stem.1    / stages.X.blocks.Y
    ParitySpec(M.convnext_tiny_cls, "convnext_tiny", key_transform=_convnext_key_transform),
    ParitySpec(
        M.convnext_small_cls, "convnext_small", tier="slow", key_transform=_convnext_key_transform
    ),
    ParitySpec(
        M.convnext_base_cls, "convnext_base", tier="slow", key_transform=_convnext_key_transform
    ),
    ParitySpec(
        M.convnext_large_cls, "convnext_large", tier="heavy", key_transform=_convnext_key_transform
    ),
    ParitySpec(
        M.convnext_xlarge_cls,
        "convnext_xlarge",
        tier="heavy",
        key_transform=_convnext_key_transform,
    ),
    # ── CSPNet ───────────────────────────────────────────────────────────────
    # timm ships all three; the module and state-dict naming already mirror
    # ``timm.models.cspnet``, so the converter's ``map_key`` is the identity
    # apart from the head rename — which is exactly what ``_cspnet_key_transform``
    # reproduces here.
    ParitySpec(M.cspresnet_50_cls, "cspresnet50", key_transform=_cspnet_key_transform),
    ParitySpec(M.cspresnext_50_cls, "cspresnext50", key_transform=_cspnet_key_transform),
    ParitySpec(
        M.cspdarknet_53_cls, "cspdarknet53", key_transform=_cspnet_key_transform, tier="slow"
    ),
    # ── PVT v2 ───────────────────────────────────────────────────────────────
    # head: head.weight (auto-remapped)
    ParitySpec(M.pvt_v2_b0_cls, None),
    ParitySpec(M.pvt_v2_b1_cls, "pvt_v2_b1", atol=2e-3),
    ParitySpec(M.pvt_v2_b2_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b3_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b4_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b5_cls, None, tier="slow"),
    # ── CvT ──────────────────────────────────────────────────────────────────
    # Internal ordering may differ → named transfer, accept partial match
    ParitySpec(M.cvt_13_cls, None),  # no stable timm equiv with same arch
    ParitySpec(M.cvt_21_cls, None, tier="slow"),
    ParitySpec(M.cvt_w24_cls, None, tier="slow"),
    # ── CrossViT ─────────────────────────────────────────────────────────────
    ParitySpec(M.crossvit_9_cls, None),
    ParitySpec(M.crossvit_tiny_cls, None),
    ParitySpec(M.crossvit_small_cls, None),
    ParitySpec(M.crossvit_base_cls, None, tier="slow"),
    ParitySpec(M.crossvit_15_cls, None),
    ParitySpec(M.crossvit_18_cls, None),
    # ── CoAtNet ───────────────────────────────────────────────────────────────
    # 98 % of keys are unmatched (Lucid uses stem.N vs timm's stages.0.blocks.N);
    # positional shape mismatch confirms different module hierarchy.
    # Self-consistency only until a key_transform is written for CoAtNet.
    ParitySpec(M.coatnet_0_cls, None, tier="slow"),
    # ── EfficientFormer ───────────────────────────────────────────────────────
    ParitySpec(M.efficientformer_l1_cls, None),
    ParitySpec(M.efficientformer_l3_cls, None, tier="slow"),
    ParitySpec(M.efficientformer_l7_cls, None, tier="slow"),
    # ── MaxViT ───────────────────────────────────────────────────────────────
    # 100% key coverage — attribute names match timm's maxvit_tiny_tf_224 exactly.
    # Relative position bias accumulation → atol=2e-2.
    ParitySpec(M.maxvit_tiny_cls, "maxvit_tiny_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_small_cls, "maxvit_small_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_base_cls, "maxvit_base_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_large_cls, "maxvit_large_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_xlarge_cls, None, tier="slow"),
    # ── InceptionNeXt ────────────────────────────────────────────────────────
    # 100% named key coverage — no key_transform or positional fallback needed.
    ParitySpec(M.inception_next_tiny_cls, "inception_next_tiny", tier="slow"),
]


# ── Helpers for test parametrisation ─────────────────────────────────────────


def specs_for_tier(tier: str) -> list[ParitySpec]:
    """Return all specs that should run at the given tier."""
    return [s for s in SPECS if s.tier == tier]


def specs_with_timm(tier: str | None = None) -> list[ParitySpec]:
    """Return specs that have a timm reference (can do numeric parity)."""
    out = [s for s in SPECS if s.timm_name is not None and not s.skip_reason]
    if tier is not None:
        out = [s for s in out if s.tier == tier]
    return out


def specs_self_consistency(tier: str | None = None) -> list[ParitySpec]:
    """Return specs that only do self-consistency (no timm reference)."""
    out = [s for s in SPECS if s.timm_name is None and not s.skip_reason]
    if tier is not None:
        out = [s for s in out if s.tier == tier]
    return out
