"""Drifting Model configuration — Deng, Li, Li, Du & He, 2026.

Every other one-step generator in this zoo starts from a multi-step
process and compresses it: MeanFlow learns the integral a flow would have
taken, TVM the transition a sampler would have walked.  A Drifting Model
never had a multi-step process.  The iteration is moved to *training*:
each optimizer step drifts every generated sample a little toward the
data and away from the other samples, and the network is regressed onto
that drifted copy.  When the two distributions match, the drift vanishes
and training is at a fixed point — inference was always one evaluation.

The fields below are the official release's, one for one.  Three groups:

* **Generator** — the ``DitGen`` / ``LightningDiT`` hyper-parameters of
  ``configs/gen/*.yaml :: model``.  ``input_size`` is :attr:`sample_size`
  here, the name the rest of the generative zoo uses.
* **Drifting loss** — ``train.forward_dict``, ``train.loss_kwargs`` and
  ``train.activation_kwargs``.  ``R_list`` is :attr:`temperatures`.
* **Feature space** — ``feature.*`` plus the MAE encoder's own
  ``model`` block.  The encoder is a frozen, training-time-only network
  and is built on request rather than with the generator; see
  :meth:`DriftingForImageGeneration.load_feature_encoder`.

Two release flags have no field because Lucid expresses them differently.
``use_bf16``, and the MAE's ``use_bf16`` likewise, is :class:`lucid.amp.autocast`
with ``dtype=lucid.bfloat16``.  Do not cast the model instead: the release
keeps its parameters in float32, and bfloat16 parameters round small optimizer
updates away (one AdamW step at ``lr=4e-4`` changed 80.6% of the elements,
against 100% in float32 and under autocast).  The optimizer, EMA and
sample-queue push rate are training-loop settings, not model settings.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_patch_grid,
    check_sample_space_channels,
    patch_grid_side,
)

# Counts a model cannot be built or trained with at zero (the field attracts
# toward positives, so it has nothing without them), and counts where zero is
# a switch (no style codebook, no in-context tokens, no guidance negatives).
_POSITIVE_COUNTS = (
    "depth",
    "cond_dim",
    "num_classes",
    "noise_coords",
    "gen_per_label",
    "pos_per_sample",
)
_NON_NEGATIVE_COUNTS = ("noise_classes", "n_cls_tokens", "neg_per_sample")


@model_family_meta(
    canonical_name="Drifting Model",
    citation=(
        "Deng, Mingyang, He Li, Tianhong Li, Yilun Du, and Kaiming He. "
        '"Generative Modeling via Drifting." arXiv preprint arXiv:2602.04770, 2026.'
    ),
    theory=r"""
    A generator :math:`f_\theta` maps noise to samples, :math:`x = f_\theta(\epsilon)`,
    and induces the pushforward :math:`q = f_{\theta\#}\,p_\epsilon`.  Training is a
    sequence of models, so it is also a sequence of pushforwards :math:`\{q_i\}`; a
    Drifting Model governs that sequence directly with a **drifting field**
    :math:`V_{p,q}` that says how each sample should move:

    .. math::

        x_{i+1} = x_i + V_{p,q_i}(x_i).

    The field is chosen **anti-symmetric**, :math:`V_{p,q} = -V_{q,p}`, which is all
    it takes for :math:`q = p \Rightarrow V = 0`: matched distributions are an
    equilibrium.  The instantiation is mean-shift attraction toward the data and
    repulsion from the current samples,

    .. math::

        V_{p,q}(x) = \frac{1}{Z_p Z_q}\,
            \mathbb{E}_{y^+\sim p,\; y^-\sim q}
            \bigl[k(x, y^+)\,k(x, y^-)\,(y^+ - y^-)\bigr],
        \qquad k(x, y) = \exp\!\Bigl(-\tfrac{1}{\tau}\lVert x - y\rVert\Bigr),

    with the normalisers :math:`Z` realised as a softmax over the :math:`y`
    samples (and, in practice, a second softmax over the batch of :math:`x`).

    The fixed point :math:`f_\theta(\epsilon) = f_\theta(\epsilon) + V(f_\theta(\epsilon))`
    becomes a regression onto a frozen target,

    .. math::

        \mathcal{L} = \mathbb{E}_\epsilon\Bigl\lVert f_\theta(\epsilon) -
            \mathrm{sg}\bigl(f_\theta(\epsilon) + V(f_\theta(\epsilon))\bigr)\Bigr\rVert^2,

    whose value is :math:`\mathbb{E}\lVert V\rVert^2` but whose gradient never passes
    through :math:`V` — back-propagating through a distribution is what the
    stop-gradient avoids.

    For images the loss is taken in the feature space of a frozen self-supervised
    encoder :math:`\phi` (a ResNet-style MAE, plus ConvNeXt-V2 for pixels), one term
    per scale and location, each with its own distance and drift normalisation and
    summed over temperatures :math:`\tau \in \{0.02, 0.05, 0.2\}`.  Guidance is a
    training-time behaviour: real samples of other classes join the negatives with
    a weight set by a sampled scale :math:`\alpha`, the generator is conditioned on
    :math:`\alpha`, and inference stays one evaluation.
    """,
)
@dataclass(frozen=True)
class DriftingConfig(GenerativeModelConfig):
    r"""Configuration for the Drifting Model family.

    Parameters
    ----------
    sample_size : int, default=32
        Side of the generated field — the official ``input_size``.  ``32``
        for the SD-VAE latent models, ``256`` for the pixel models.
    in_channels, out_channels : int, default=4, 4
        Channels of the noise and of the output.  The generator maps a
        noise field to a sample of the same shape, so they must match.
    patch_size : int, default=2
        Tokenisation patch — ``2`` for ``B/2`` / ``L/2``, ``16`` for
        ``B/16`` / ``L/16``.  Both give 256 tokens.
    hidden_size : int, default=768
        Transformer width.
    depth : int, default=12
        Number of blocks.
    num_heads : int, default=12
        Attention heads.
    mlp_ratio : float, default=4.0
        Feed-forward expansion before the SwiGLU two-thirds rule.
    cond_dim : int, default=768
        Width of the conditioning vector that drives adaLN and the
        in-context tokens.  Every released model sets it equal to
        ``hidden_size``.
    num_classes : int, default=1000
        Class vocabulary.  There is no null class: guidance is trained in
        by conditioning on :math:`\alpha`, not by label dropout.
    noise_classes : int, default=64
        Codebook size of the random *style embeddings* — the discrete half
        of the noise.  ``0`` disables them.
    noise_coords : int, default=32
        How many style indices are drawn per sample; their embeddings are
        summed into the conditioning vector.
    n_cls_tokens : int, default=16
        In-context conditioning tokens prepended to the sequence.
    use_qknorm, use_swiglu, use_rope, use_rmsnorm : bool, default=True
        The LightningDiT modernisations the paper adopts (A.2).
    attn_fp32 : bool, default=True
        Run attention in float32 when the model is in a narrower dtype.
        ``False`` for the released latent ``B/2`` and ``L/2``.
    use_remat : bool, default=False
        Recompute each block on the backward pass instead of storing its
        activations.  ``True`` for the released pixel models.
    frequency_embedding_size : int, default=256
        Width of the sinusoid the guidance scale is embedded through.
    temperatures : tuple of float, default=(0.2, 0.05, 0.02)
        Kernel temperatures :math:`\tau` (the release's ``R_list``).  One
        normalised drift per temperature, summed.
    gen_per_label : int, default=64
        Generated samples per class label, :math:`N_\mathrm{neg}`.
    cfg_min, cfg_max : float, default=1.0, 4.0
        Range of the training-time guidance scale :math:`\alpha`.
    neg_cfg_pw : float, default=5.0
        Exponent of the power law :math:`p(\alpha) \propto \alpha^{-k}` the
        scale is drawn from.
    no_cfg_frac : float, default=0.0
        Probability of replacing the drawn scale by ``1`` (no guidance).
        ``0.5`` for the released ``L/2``.
    patch_mean_size, patch_std_size : tuple of int, default=(2, 4)
        Patch sides over which local feature means / stds are pooled (A.5).
    use_mean, use_std : bool, default=True
        Whether each feature map also contributes its global mean / std.
    every_k_block : int or None, default=2
        Also take features from every ``k``-th residual block of each
        stage.  ``None`` keeps only the stage outputs.
    pos_per_sample : int, default=64
        Positive (same-class) samples drawn per label, :math:`N_\mathrm{pos}`.
    neg_per_sample : int, default=32
        Unconditional real samples drawn per label, :math:`N_\mathrm{unc}`.
        ``0`` trains without guidance negatives.
    positive_bank_size : int, default=128
        Per-class sample-queue length.
    negative_bank_size : int, default=1000
        Global unconditional queue length.
    use_mae : bool, default=True
        Compute the loss in the latent- / pixel-MAE feature space.  No trained
        encoder is published, so see
        :meth:`DriftingForImageGeneration.load_feature_encoder`.
    use_convnext : bool, default=False
        Also use ConvNeXt-V2-Base (ImageNet-22k) features.  Pixel models only.
    mae_base_channels : int, default=640
        MAE ResNet base width :math:`C`; stages are :math:`C, 2C, 4C, 8C`.
    mae_layers : tuple of int, default=(3, 4, 6, 3)
        Basic blocks per MAE stage — ResNet-34's.
    mae_input_patch_size : int, default=1
        Space-to-depth patch applied before the MAE ResNet: ``1`` for the
        32x32x4 latent, ``8`` for 256x256x3 pixels (giving 32x32x192).

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, *"Generative Modeling via
    Drifting"*, arXiv:2602.04770, 2026.  Architecture: Appendix A.2 and
    Table 8.  Loss: Alg. 2 and Appendices A.5–A.7.

    Where Table 8 and the released ``configs/gen/*.yaml`` disagree, the
    release wins, because the authors' checkpoints were trained by it: the
    latent ``B/2`` uses 64 positives per label (Table 8 says 128).

    Examples
    --------
    >>> from lucid.models.generative.drifting import DriftingConfig
    >>> config = DriftingConfig()
    >>> config.hidden_size, config.depth, config.patch_size
    (768, 12, 2)
    >>> config.num_patches, config.swiglu_hidden_size
    (256, 2048)
    """

    model_type: ClassVar[str] = "drifting"

    # ── generator (configs/gen/*.yaml :: model) ──────────────────────────────
    sample_size: int | tuple[int, int] = 32
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    cond_dim: int = 768
    num_classes: int = 1000
    noise_classes: int = 64
    noise_coords: int = 32
    n_cls_tokens: int = 16
    use_qknorm: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    use_rmsnorm: bool = True
    attn_fp32: bool = True
    use_remat: bool = False
    frequency_embedding_size: int = 256

    # ── drifting loss (train.forward_dict / loss_kwargs / activation_kwargs) ─
    temperatures: tuple[float, ...] = (0.2, 0.05, 0.02)
    gen_per_label: int = 64
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    neg_cfg_pw: float = 5.0
    no_cfg_frac: float = 0.0
    patch_mean_size: tuple[int, ...] = (2, 4)
    patch_std_size: tuple[int, ...] = (2, 4)
    use_mean: bool = True
    use_std: bool = True
    every_k_block: int | None = 2

    # ── sample queue (train.*) ───────────────────────────────────────────────
    pos_per_sample: int = 64
    neg_per_sample: int = 32
    positive_bank_size: int = 128
    negative_bank_size: int = 1000

    # ── feature space (feature.* and the MAE's model block) ──────────────────
    use_mae: bool = True
    use_convnext: bool = False
    mae_base_channels: int = 640
    mae_layers: tuple[int, int, int, int] = (3, 4, 6, 3)
    mae_input_patch_size: int = 1

    @property
    def grid_size(self) -> int:
        """Tokens per side of the patch grid."""
        return patch_grid_side(self.sample_size) // self.patch_size

    @property
    def num_patches(self) -> int:
        """Image tokens — ``(sample_size / patch_size)^2``, 256 for every release."""
        return self.grid_size**2

    @property
    def head_dim(self) -> int:
        """Per-head width."""
        return self.hidden_size // self.num_heads

    @property
    def swiglu_hidden_size(self) -> int:
        """SwiGLU inner width: two thirds of ``hidden * mlp_ratio``, rounded up to 32.

        The release computes ``int(2/3 * int(hidden * mlp_ratio))`` and then
        rounds up to a multiple of 32 — 2048 for ``B``, 2752 for ``L``.
        """
        inner = int(2 / 3 * int(self.hidden_size * self.mlp_ratio))
        return (inner + 31) // 32 * 32

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "temperatures", tuple(float(t) for t in self.temperatures))
        object.__setattr__(self, "patch_mean_size", tuple(int(s) for s in self.patch_mean_size))
        object.__setattr__(self, "patch_std_size", tuple(int(s) for s in self.patch_std_size))
        object.__setattr__(self, "mae_layers", tuple(int(n) for n in self.mae_layers))

        # One check per group of fields, in the order the groups are declared.
        check_patch_grid(self.sample_size, self.patch_size, self.hidden_size, self.num_heads)
        check_sample_space_channels(self.in_channels, self.out_channels, "the generated sample")
        self._check_counts()
        self._check_rope()
        self._check_drift_kernel()
        self._check_feature_sets()
        self._check_sample_queue()
        self._check_mae_encoder()
        self._check_convnext()

    def _check_counts(self) -> None:
        """Refuse a zero model dimension, or a negative count that may be zero."""
        for name in _POSITIVE_COUNTS:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in _NON_NEGATIVE_COUNTS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    def _check_rope(self) -> None:
        """Refuse an odd head width under RoPE, which rotates feature pairs."""
        if self.use_rope and self.head_dim % 2 != 0:
            raise ValueError(
                f"use_rope rotates feature pairs, so the head width must be even; "
                f"got hidden_size={self.hidden_size} over num_heads={self.num_heads}"
            )

    def _check_drift_kernel(self) -> None:
        """Refuse kernel temperatures or a guidance distribution Alg. 2 cannot use."""
        if not self.temperatures or any(t <= 0.0 for t in self.temperatures):
            raise ValueError(
                f"temperatures must be non-empty and positive, got {self.temperatures}"
            )
        if not 0.0 < self.cfg_min <= self.cfg_max:
            raise ValueError(
                f"the guidance range must satisfy 0 < cfg_min <= cfg_max, "
                f"got [{self.cfg_min}, {self.cfg_max}]"
            )
        if not math.isfinite(self.neg_cfg_pw):
            raise ValueError(f"neg_cfg_pw must be finite, got {self.neg_cfg_pw}")
        if not 0.0 <= self.no_cfg_frac <= 1.0:
            raise ValueError(f"no_cfg_frac is a probability, got {self.no_cfg_frac}")

    def _check_feature_sets(self) -> None:
        """Refuse a block stride or a pooling side no feature map can be cut by (A.5)."""
        if self.every_k_block is not None and self.every_k_block < 1:
            raise ValueError(
                f"every_k_block must be at least 1, or None to disable, got {self.every_k_block}"
            )
        if any(s <= 0 for s in self.patch_mean_size + self.patch_std_size):
            raise ValueError("patch_mean_size / patch_std_size entries must be positive")

    def _check_sample_queue(self) -> None:
        """Refuse an empty sample queue (A.8)."""
        if self.positive_bank_size <= 0 or self.negative_bank_size <= 0:
            raise ValueError("the sample-queue sizes must be positive")

    def _check_mae_encoder(self) -> None:
        """Refuse an MAE ResNet that cannot be built over this sample."""
        if self.mae_base_channels <= 0 or len(self.mae_layers) != 4:
            raise ValueError(
                f"the MAE encoder needs a positive width and four stages, got "
                f"mae_base_channels={self.mae_base_channels}, mae_layers={self.mae_layers}"
            )
        if any(n <= 0 for n in self.mae_layers):
            raise ValueError(f"mae_layers entries must be positive, got {self.mae_layers}")
        side = patch_grid_side(self.sample_size)
        if self.mae_input_patch_size <= 0 or side % self.mae_input_patch_size != 0:
            raise ValueError(
                f"mae_input_patch_size must be positive and divide sample_size {side}, "
                f"got {self.mae_input_patch_size}"
            )

    def _check_convnext(self) -> None:
        """Refuse ConvNeXt-V2 features for a sample that is not an RGB image."""
        if self.use_convnext and self.in_channels != 3:
            raise ValueError(
                "use_convnext reads RGB images; it applies to pixel-space models "
                f"(in_channels=3), got in_channels={self.in_channels}"
            )
