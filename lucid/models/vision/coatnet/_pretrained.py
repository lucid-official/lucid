"""Registry factories for CoAtNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.coatnet._config import CoAtNetConfig
from lucid.models.vision.coatnet._model import CoAtNet, CoAtNetForImageClassification
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_0 = CoAtNetConfig(
    variant="coatnet_0",
    blocks_per_stage=(2, 3, 5, 2),
    dims=(96, 192, 384, 768),
    stem_width=64,
    attn_heads=(12, 24),
    mbconv_expand=4,
    head_hidden_size=768,
)

# Paper Table 5 (Dai et al., NeurIPS 2021):
# L = (L0, L1, L2, L3, L4), D = (D0, D1, D2, D3, D4).
# Lucid keeps L0 / D0 in `stem_width` (stem is 2 conv layers, paper L0=2 always).
# Lucid `blocks_per_stage` = (L1, L2, L3, L4); Lucid `dims` = (D1, D2, D3, D4).
# Attn heads derive from dim/32 (head_dim=32) for each of the two T-stages.
_CFG_1 = CoAtNetConfig(
    variant="coatnet_1",
    blocks_per_stage=(2, 6, 14, 2),
    dims=(96, 192, 384, 768),
    stem_width=64,
    attn_heads=(12, 24),
    mbconv_expand=4,
    head_hidden_size=768,
)
_CFG_2 = CoAtNetConfig(
    variant="coatnet_2",
    blocks_per_stage=(2, 6, 14, 2),
    dims=(128, 256, 512, 1024),
    stem_width=128,
    attn_heads=(16, 32),
    mbconv_expand=4,
    head_hidden_size=1024,
)
_CFG_3 = CoAtNetConfig(
    variant="coatnet_3",
    blocks_per_stage=(2, 6, 14, 2),
    dims=(192, 384, 768, 1536),
    stem_width=192,
    attn_heads=(24, 48),
    mbconv_expand=4,
    head_hidden_size=1536,
)
_CFG_4 = CoAtNetConfig(
    variant="coatnet_4",
    blocks_per_stage=(2, 12, 28, 2),
    dims=(192, 384, 768, 1536),
    stem_width=192,
    attn_heads=(24, 48),
    mbconv_expand=4,
    head_hidden_size=1536,
)
_CFG_5 = CoAtNetConfig(
    variant="coatnet_5",
    blocks_per_stage=(2, 12, 28, 2),
    dims=(256, 512, 1280, 2048),
    stem_width=192,
    # Table 11: CoAtNet-5 uses head size 64, not the standard 32 — dims
    # (1280, 2048) / 64 = (20, 32).
    attn_heads=(20, 32),
    mbconv_expand=4,
    head_hidden_size=2048,
)

# CoAtNet-6 / CoAtNet-7 from paper §A.2 / Table 12 — "we move 2/3 of the
# MBConv blocks of S2 into S3 and double its hidden dimension".  S3 becomes
# a *mixed-block* stage (MBConv L=8 then channel-expand to a wider width
# then Transformer L=42); the `mixed_s3` field on CoAtNetConfig spells this
# out.  `blocks_per_stage[2]` is set to L_mb + L_attn = 50 for invariance,
# but the actual stage 3 block layout is fully governed by `mixed_s3`.
_CFG_6 = CoAtNetConfig(
    variant="coatnet_6",
    blocks_per_stage=(2, 4, 50, 2),
    dims=(192, 384, 768, 2048),
    stem_width=192,
    # Section A.2 raises the head size to 128 for the scaled models:
    # (1536, 2048) / 128 = (12, 16).
    attn_heads=(12, 16),
    mbconv_expand=4,
    head_hidden_size=2048,
    mixed_s3=(8, 42, 1536),
)
_CFG_7 = CoAtNetConfig(
    variant="coatnet_7",
    blocks_per_stage=(2, 4, 50, 2),
    dims=(256, 512, 1024, 3072),
    stem_width=192,
    # Head size 128: (2048, 3072) / 128 = (16, 24).
    attn_heads=(16, 24),
    mbconv_expand=4,
    head_hidden_size=3072,
    mixed_s3=(8, 42, 2048),
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_0,
    params=25_000_000,
)
def coatnet_0(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-0 backbone (Dai et al., 2021).

    Builds the canonical *CoAtNet-0* configuration:
    ``blocks_per_stage=(2, 3, 5, 2)``, ``dims=(96, 192, 384, 768)``,
    ``stem_width=64``, ``attn_heads=(12, 24)``,
    ``head_hidden_size=768``.  The first two stages are MBConv with
    expansion ratio 4 and the last two stages are relative-attention
    transformers with head dim 32.  Approximately **25.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``coatnet_0``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-0
        config.  Common examples: ``image_size=384``, ``in_channels=1``,
        ``head_hidden_size=None``.  Each override must match a field of
        :class:`CoAtNetConfig`.

    Returns
    -------
    CoAtNet
        A :class:`CoAtNet` backbone returning a
        :math:`(B, 768, H/32, W/32)` spatial feature map.

    Notes
    -----
    CoAtNet-0 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Dai
    et al., 2021, Table 5).  See
    `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_0
    >>> model = coatnet_0()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_0")
    cfg = apply_overrides(_CFG_0, overrides)
    return CoAtNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_0,
    params=25_000_000,
)
def coatnet_0_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-0 image classifier (Dai et al., 2021).

    Combines the :func:`coatnet_0` backbone with the standard CoAtNet
    head: global average pool → LayerNorm → ``Linear(768) + Tanh`` →
    final linear classifier.  Default output is ``num_classes=1000``
    (ImageNet-1k).  Approximately **25.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        No checkpoint is published for ``coatnet_0_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides on top of the canonical CoAtNet-0 config.
        Commonly used: ``num_classes`` to switch label space,
        ``image_size`` to fine-tune at a different resolution.

    Returns
    -------
    CoAtNetForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CoAtNet-0 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Dai
    et al., 2021, Table 5).  Larger variants (CoAtNet-3 / -4) reach
    **84.5% / 85.8%** at 384x384 after JFT-3B pretraining
    (Table 1, headline result).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_0_cls
    >>> model = coatnet_0_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_0_cls")
    cfg = apply_overrides(_CFG_0, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-1 (Dai et al., 2021, Table 5) — ~42M params, 83.3% ImageNet top-1
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_1,
    params=42_000_000,
)
def coatnet_1(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-1 backbone (Dai et al., 2021).

    Same channel widths as CoAtNet-0 but with substantially deeper
    stages: ``blocks_per_stage=(2, 6, 14, 2)``, ``dims=(96, 192, 384, 768)``,
    ``stem_width=64``, ``attn_heads=(12, 24)``.  Approximately **42M
    parameters**.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_1``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-1
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Notes
    -----
    Reaches **83.3% ImageNet-1k top-1** at 224×224 (Table 5, NeurIPS
    2021).  See `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_1
    >>> model = coatnet_1()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 768, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_1")
    cfg = apply_overrides(_CFG_1, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_1,
    params=42_000_000,
)
def coatnet_1_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-1 image classifier (Dai et al., 2021).

    Backbone + standard CoAtNet head (GAP → LayerNorm → Linear+Tanh →
    classifier).  Approximately **42M parameters**, 83.3% ImageNet-1k
    top-1 at 224x224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_1_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-1
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_1_cls
    >>> model = coatnet_1_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_1_cls")
    cfg = apply_overrides(_CFG_1, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-2 (Dai et al., 2021, Table 5) — ~75M params, 84.1% ImageNet top-1
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_2,
    params=75_000_000,
)
def coatnet_2(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-2 backbone (Dai et al., 2021).

    Wider variant: ``blocks_per_stage=(2, 6, 14, 2)``,
    ``dims=(128, 256, 512, 1024)``, ``stem_width=128``,
    ``attn_heads=(16, 32)``.  Approximately **75M parameters**, **84.1%
    ImageNet-1k top-1** at 224×224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_2``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-2
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_2
    >>> model = coatnet_2()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 1024, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_2")
    cfg = apply_overrides(_CFG_2, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_2,
    params=75_000_000,
)
def coatnet_2_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-2 image classifier (Dai et al., 2021).

    Wider variant — ``dims=(128, 256, 512, 1024)`` — with the standard
    CoAtNet head.  Approximately **75M parameters**, 84.1% ImageNet-1k
    top-1 at 224x224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_2_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-2
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_2_cls
    >>> model = coatnet_2_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_2_cls")
    cfg = apply_overrides(_CFG_2, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-3 (Dai et al., 2021, Table 5) — ~168M params, 84.5% ImageNet top-1
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_3,
    params=168_000_000,
)
def coatnet_3(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-3 backbone (Dai et al., 2021).

    Larger variant: ``blocks_per_stage=(2, 6, 14, 2)``,
    ``dims=(192, 384, 768, 1536)``, ``stem_width=192``,
    ``attn_heads=(24, 48)``.  Approximately **168M parameters**, **84.5%
    ImageNet-1k top-1** at 224×224 (Table 5).  With ImageNet-21k or
    JFT-3B pretraining the paper reports 86.5% / 87.8% at 384x384.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_3``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-3
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_3
    >>> model = coatnet_3()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 1536, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_3")
    cfg = apply_overrides(_CFG_3, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_3,
    params=168_000_000,
)
def coatnet_3_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-3 image classifier (Dai et al., 2021).

    Larger variant — ``dims=(192, 384, 768, 1536)`` — with the standard
    CoAtNet head.  Approximately **168M parameters**, 84.5% ImageNet-1k
    top-1 at 224x224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_3_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-3
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_3_cls
    >>> model = coatnet_3_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_3_cls")
    cfg = apply_overrides(_CFG_3, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-4 (Dai et al., 2021, Table 5) — ~275M params, 85.0% ImageNet top-1
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_4,
    params=275_000_000,
)
def coatnet_4(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-4 backbone (Dai et al., 2021).

    Even deeper variant: ``blocks_per_stage=(2, 12, 28, 2)``,
    ``dims=(192, 384, 768, 1536)``, ``stem_width=192``,
    ``attn_heads=(24, 48)``.  Approximately **275M parameters**, **85.0%
    ImageNet-1k top-1** at 224×224 (Table 5).  Paper headline with
    JFT-3B pretrain reaches 88.4% at 512x512.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_4``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-4
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_4
    >>> model = coatnet_4()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 1536, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_4")
    cfg = apply_overrides(_CFG_4, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_4,
    params=275_000_000,
)
def coatnet_4_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-4 image classifier (Dai et al., 2021).

    Deepest of the CoAtNet-0..4 series — same dims as CoAtNet-3 but with
    ``blocks_per_stage=(2, 12, 28, 2)``.  Approximately **275M
    parameters**, 85.0% ImageNet-1k top-1 at 224x224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_4_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-4
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_4_cls
    >>> model = coatnet_4_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_4_cls")
    cfg = apply_overrides(_CFG_4, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-5 (Dai et al., 2021, Table 5) — ~688M params, 85.8% ImageNet top-1
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_5,
    params=688_000_000,
)
def coatnet_5(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-5 backbone (Dai et al., 2021).

    Widest paper-cited variant: ``blocks_per_stage=(2, 12, 28, 2)``,
    ``dims=(256, 512, 1280, 2048)``, ``stem_width=192``,
    ``attn_heads=(20, 32)``.  Approximately **688M parameters**, **85.8%
    ImageNet-1k top-1** at 224×224 (Table 5).  With JFT-3B pretraining
    the paper reports 89.0% at 512x512.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_5``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-5
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Notes
    -----
    Memory footprint at default ``image_size=224`` is non-trivial — a
    forward pass of a single 3×224×224 input materialises stage-3/4
    activations on the order of ``B × D × H/32 × W/32`` channels with
    D=2048.  Instantiating + running this variant on a 16 GB host is
    near-the-edge; prefer measuring on a larger GPU.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_5
    >>> model = coatnet_5()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_5")
    cfg = apply_overrides(_CFG_5, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_5,
    params=688_000_000,
)
def coatnet_5_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-5 image classifier (Dai et al., 2021).

    Widest paper-cited variant — ``dims=(256, 512, 1280, 2048)`` — with
    the standard CoAtNet head.  Approximately **688M parameters**,
    85.8% ImageNet-1k top-1 at 224x224 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_5_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-5
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_5_cls
    >>> model = coatnet_5_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_5_cls")
    cfg = apply_overrides(_CFG_5, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-6 (Dai et al., 2021, §A.2 / Table 12) — ~1.5B params
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_6,
    params=1_470_000_000,
)
def coatnet_6(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-6 backbone (Dai et al., 2021).

    Mixed-stage variant: S3 carries an MBConv sub-stage *and* a
    transformer sub-stage at a wider dimension.  Spec from paper §A.2 /
    Table 12:

    - S0 conv stem: D=192, L=2 (handled by ``stem_width``)
    - S1 MBConv: L=2, D=192
    - S2 MBConv: L=4, D=384
    - S3 *mixed*: MBConv L=8 D=768  →  expand→1536  →  TFMRel L=42 D=1536
    - S4 TFMRel: L=2, D=2048

    Approximately **1.5B parameters** — only meaningful with very large
    pretraining (the paper's headline 88.4-89.0% ImageNet numbers come
    from JFT-3B pretraining + 512×512 finetune).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_6``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-6
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Notes
    -----
    Footprint at default ``image_size=224`` exceeds what a 16 GB host
    can comfortably instantiate (≈ 6 GB of params alone before
    activations); use a larger GPU.  Building this variant is
    reserved for inference on dedicated hardware.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_6
    >>> model = coatnet_6()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_6")
    cfg = apply_overrides(_CFG_6, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_6,
    params=1_470_000_000,
)
def coatnet_6_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-6 image classifier (Dai et al., 2021).

    Mixed-S3 variant.  Approximately **1.5B parameters**.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_6_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-6
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_6_cls
    >>> model = coatnet_6_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_6_cls")
    cfg = apply_overrides(_CFG_6, overrides)
    return CoAtNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# CoAtNet-7 (Dai et al., 2021, §A.2 / Table 12) — ~2.4B params
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_7,
    params=2_440_000_000,
)
def coatnet_7(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-7 backbone (Dai et al., 2021).

    Widest mixed-stage variant.  Spec from paper §A.2 / Table 12:

    - S0 conv stem: D=192, L=2
    - S1 MBConv: L=2, D=256
    - S2 MBConv: L=4, D=512
    - S3 *mixed*: MBConv L=8 D=1024  →  expand→2048  →  TFMRel L=42 D=2048
    - S4 TFMRel: L=2, D=3072

    Approximately **2.4B parameters** — paper's strongest published
    CoAtNet, only usable with JFT-3B pretraining.  Not buildable on a
    16 GB host (param footprint alone is ≈ 9.7 GB before activations).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_7``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-7
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``image_size=384``, ``in_channels=1``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_7
    >>> model = coatnet_7()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.last_hidden_state.shape
    (1, 3072, 7, 7)

    Two and a half billion parameters — the largest variant in the
    paper, and worth knowing before instantiating one.
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_7")
    cfg = apply_overrides(_CFG_7, overrides)
    return CoAtNet(cfg)


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_7,
    params=2_440_000_000,
)
def coatnet_7_cls(pretrained: bool = False, **overrides: object) -> CoAtNetForImageClassification:
    r"""CoAtNet-7 image classifier (Dai et al., 2021).

    Widest mixed-S3 variant.  Approximately **2.4B parameters**.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No checkpoint is published for ``coatnet_7_cls``, so ``True`` raises
        ``NotImplementedError`` rather than return randomly initialised weights.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-7
        config; each must name a field of :class:`CoAtNetConfig` (e.g.
        ``num_classes=10`` to retarget the head).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_7_cls
    >>> model = coatnet_7_cls()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """
    if pretrained:
        reject_unavailable_pretrained("coatnet_7_cls")
    cfg = apply_overrides(_CFG_7, overrides)
    return CoAtNetForImageClassification(cfg)
