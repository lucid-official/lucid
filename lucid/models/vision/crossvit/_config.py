"""CrossViT configuration dataclass (Chen et al., ICCV 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="CrossViT",
    citation=(
        'Chen, Chun-Fu (Richard), et al. "CrossViT: Cross-Attention '
        'Multi-Scale Vision Transformer for Image Classification." '
        "Proceedings of the IEEE/CVF International Conference on "
        "Computer Vision, 2021, pp. 357-366."
    ),
    theory=r"""
    CrossViT explicitly models *multi-scale* visual information by
    running two parallel ViT branches with different patch sizes and
    fusing them through a lightweight cross-attention mechanism.  A
    *small-patch* branch (patch size 12) processes a long sequence of
    fine-grained tokens, while a *large-patch* branch (patch size 16)
    processes a short sequence of coarse tokens.  The two branches
    operate on *different* input resolutions — the small-patch branch
    on the full image (e.g. :math:`240\times240`), the large-patch
    branch on a slightly down-sampled version (e.g. :math:`224\times224`)
    — and each has its own class token plus learnable positional
    embedding.

    Each branch is independently transformer-encoded for
    :math:`N_s` / :math:`N_l` standard self-attention blocks; the two
    branches then exchange information at the **end of every stage**
    via a *MultiScaleBlock*: the CLS token from one branch is
    projected into the other branch's embedding space, cross-attends
    to that branch's patch tokens, and is projected back.  Concretely,
    for the small-to-large direction,

    .. math::

        \tilde{x}_{\text{cls}}^{\,l} = x_{\text{cls}}^{l} +
            \mathrm{CrossAttn}\!\bigl(W_q x_{\text{cls}}^{l},
            W_k x_{\text{patch}}^{s}, W_v x_{\text{patch}}^{s}\bigr),

    and symmetrically for large-to-small.  Because cross-attention is
    only :math:`\mathcal{O}(N_s + N_l)` per direction, multi-scale
    fusion is essentially free relative to the per-branch
    self-attention cost.  The complete network is :math:`K=3` such
    MultiScaleBlocks stacked back-to-back; the final classifier head
    averages the logits of the two per-branch linear layers.

    The variants CrossViT-Ti / -S / -B / -9 / -15 / -18 trade depth and
    width along this two-branch skeleton (paper Table 2).
    """,
)
@dataclass(frozen=True)
class CrossViTConfig(ModelConfig):
    r"""Configuration for every paper-cited CrossViT variant.

    Parameters
    ----------
    num_classes : int, optional, default=1000
        Output-class count of the classification heads.
    in_channels : int, optional, default=3
        Number of input image channels.
    image_size : int, optional, default=240
        Spatial side of the small-branch input.  The large-branch
        input is rescaled from this via :attr:`img_scale`.
    img_scale : tuple of float, optional, default=(1.0, 224/240)
        Per-branch scale relative to ``image_size`` — the first entry
        targets the small-patch branch (always 1.0 in paper-cited
        variants), the second the large-patch branch (224/240 in the
        canonical recipe so the large branch consumes 224×224 from a
        240×240 input).
    crop_scale : bool, optional, default=False
        Method used to rescale the large-branch input.  ``False`` ⇒
        bicubic resize to the absolute branch target (paper default);
        ``True`` ⇒ center-crop.
    patch_sizes : tuple of int, optional, default=(12, 16)
        Patch sizes for the (small, large) branches.
    embed_dims : tuple of int, optional, default=(96, 192)
        Per-branch embedding dimensions (small, large).
    depths : tuple of tuple of int, optional
        Per-stage block counts ``(N_s, N_l, N_cross)`` for each of the
        ``K`` MultiScaleBlocks.  Defaults to ``((1, 4, 0)) × 3`` (the
        CrossViT-Ti recipe).  ``N_cross`` is the number of *extra*
        cross-attention transformer blocks the stage runs after the
        single mandatory cross-attention fusion — paper-cited variants
        keep this at 0.
    num_heads : tuple of int, optional, default=(3, 3)
        Per-branch attention head counts (small, large).
    mlp_ratio : tuple of float, optional, default=(4.0, 4.0, 1.0)
        MLP expansion ratios for the (small self-attn, large
        self-attn, cross-attention) sub-modules within a
        MultiScaleBlock.
    qkv_bias : bool, optional, default=True
    drop_path_rate : float, optional, default=0.0
        Stochastic-depth rate, linearly scheduled across the whole
        trunk (sum of all blocks across all stages).
    dropout : float, optional, default=0.0
        Dropout probability inside attention + MLP modules.
    layer_norm_eps : float, optional, default=1e-6
        ``eps`` for every LayerNorm in the trunk (paper / timm use
        ``1e-6``; matches the reference checkpoints exactly).

    Examples
    --------
    >>> from lucid.models.vision.crossvit import CrossViTConfig
    >>> cfg = CrossViTConfig()
    >>> cfg.num_classes, cfg.image_size
    (1000, 240)
    >>> cfg.model_type
    'crossvit'
    """

    model_type: ClassVar[str] = "crossvit"

    num_classes: int = 1000
    in_channels: int = 3
    image_size: int = 240
    img_scale: tuple[float, float] = (1.0, 224.0 / 240.0)
    crop_scale: bool = False
    patch_sizes: tuple[int, int] = (12, 16)
    embed_dims: tuple[int, int] = (96, 192)
    depths: tuple[tuple[int, int, int], ...] = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
    num_heads: tuple[int, int] = (3, 3)
    mlp_ratio: tuple[float, float, float] = (4.0, 4.0, 1.0)
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        # ``depths[k][2]`` (the per-stage cross-attention count L) is only
        # honoured for 0 and 1: the fusion is one ``_CrossAttentionBlock`` per
        # branch, which is what the reference builds when L == 0, and what
        # every registered variant uses.  Larger values used to be accepted,
        # build a single block anyway, and still reserve L stochastic-depth
        # slots — over-advancing the drop-path cursor for the whole model.
        for k, stage in enumerate(self.depths):
            if len(stage) >= 3 and int(stage[2]) > 1:
                raise NotImplementedError(
                    f"depths[{k}][2]={stage[2]} (cross-attention depth) is not "
                    f"implemented; the fusion builds a single block per branch, "
                    f"so only 0 or 1 is supported."
                )
        object.__setattr__(self, "img_scale", tuple(self.img_scale))
        object.__setattr__(self, "patch_sizes", tuple(self.patch_sizes))
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
        object.__setattr__(self, "depths", tuple(tuple(d) for d in self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        object.__setattr__(self, "mlp_ratio", tuple(self.mlp_ratio))
