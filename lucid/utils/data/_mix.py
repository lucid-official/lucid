r"""Mixup + CutMix batch-level augmentation collators.

These are NOT image transforms (they don't belong in
:mod:`lucid.utils.transforms`): they operate on the assembled batch,
mixing pairs of samples and producing soft-label classification
targets.  Plug into :class:`~lucid.utils.data.DataLoader` via the
``collate_fn`` argument::

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=MixupCollator(alpha=0.2, num_classes=10),
    )

The collator first runs :func:`~lucid.utils.data.default_collate` to
build the standard batch, then samples :math:`\lambda \sim
\mathrm{Beta}(\alpha, \alpha)` and mixes the batch with a random
permutation of itself.  Soft targets are produced even when no mix is
applied (``p == 0`` or the per-batch probability gate fails) so
training-loop code sees a consistent target dtype/shape.

The loss is the standard :func:`~lucid.nn.functional.cross_entropy`,
which accepts soft per-class probabilities as targets.
"""

import math
from typing import override

import lucid
import lucid.distributions as dist
import lucid.nn.functional as nnF
from lucid._tensor import Tensor
from lucid.utils.data.dataloader import default_collate
from lucid.utils.transforms import _random

# ── helpers ─────────────────────────────────────────────────────────


def _sample_lambda(alpha: float) -> float:
    r"""Draw one :math:`\lambda \sim \mathrm{Beta}(\alpha, \alpha)` scalar."""
    if alpha <= 0.0:
        return 1.0
    return float(dist.Beta(alpha, alpha).sample(()).item())


def _validate_init(alpha: float, num_classes: int, p: float) -> None:
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")


def _to_soft_targets(labels: Tensor, num_classes: int) -> Tensor:
    """Encode integer labels ``(B,)`` as float ``(B, num_classes)`` one-hot."""
    return nnF.one_hot(labels, num_classes=num_classes).to(lucid.float32)


# ── Mixup ────────────────────────────────────────────────────────────


class MixupCollator:
    r"""Mixup collator (Zhang et al., 2018 — arXiv:1710.09412).

    For each batch:

    1. Run :func:`default_collate` to get ``(images, labels)``.
    2. One-hot encode ``labels`` to ``(B, num_classes)`` soft targets.
    3. With probability ``p``, sample
       :math:`\lambda \sim \mathrm{Beta}(\alpha, \alpha)`, choose a
       random permutation of the batch, and produce

       .. math::

           x_{\mathrm{mix}} &= \lambda x + (1 - \lambda) x_{\pi} \\
           y_{\mathrm{mix}} &= \lambda y + (1 - \lambda) y_{\pi}

       Otherwise the batch passes through with one-hot soft targets.

    Parameters
    ----------
    alpha : float, optional, default=0.2
        Beta-distribution concentration.  Reference-framework
        recipes use ``0.2`` for ImageNet, ``1.0`` for CIFAR.
    num_classes : int
        Output class count — required so soft targets have the right
        width.
    p : float, optional, default=1.0
        Probability of applying Mixup per batch.

    Notes
    -----
    The companion loss is :func:`~lucid.nn.functional.cross_entropy`
    with the soft target shape ``(B, num_classes)`` — Lucid's
    ``cross_entropy`` accepts that natively.
    """

    def __init__(self, alpha: float = 0.2, *, num_classes: int, p: float = 1.0) -> None:
        _validate_init(alpha, num_classes, p)
        self.alpha = alpha
        self.num_classes = num_classes
        self.p = p

    def __call__(self, batch: list[object]) -> tuple[Tensor, Tensor]:
        images_obj, labels_obj = default_collate(batch)
        images = images_obj if isinstance(images_obj, Tensor) else images_obj
        labels = labels_obj if isinstance(labels_obj, Tensor) else labels_obj
        if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
            raise TypeError(
                "MixupCollator expects each sample to yield (image_tensor, "
                f"label_tensor); got images={type(images).__name__}, "
                f"labels={type(labels).__name__}"
            )
        targets = _to_soft_targets(labels, self.num_classes)
        if float(lucid.rand(1).item()) >= self.p:
            return images, targets
        lam = _sample_lambda(self.alpha)
        perm = lucid.randperm(int(images.shape[0]))
        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_targets = lam * targets + (1.0 - lam) * targets[perm]
        return mixed_images, mixed_targets

    @override
    def __repr__(self) -> str:
        return f"MixupCollator(alpha={self.alpha}, " f"num_classes={self.num_classes}, p={self.p})"


# ── CutMix ───────────────────────────────────────────────────────────


class CutMixCollator:
    r"""CutMix collator (Yun et al., 2019 — arXiv:1905.04899).

    Same lambda-sampling scheme as :class:`MixupCollator`, but instead
    of blending pixels globally, a rectangular patch of one image is
    pasted onto another.  The effective :math:`\lambda` is recomputed
    from the actual patch area (after border clamping) so soft
    targets stay calibrated.

    Parameters
    ----------
    alpha : float, optional, default=1.0
        Beta-distribution concentration.  Reference-framework recipes
        use ``1.0`` for ImageNet (uniform :math:`\lambda`).
    num_classes : int
        Output class count.
    p : float, optional, default=1.0
        Probability of applying CutMix per batch.

    Notes
    -----
    Patch placement uses a multiplicative keep-mask + additive paste
    (the same composition pattern as
    :class:`~lucid.utils.transforms.RandomErasing`) — fully
    differentiable, no in-place writes.
    """

    def __init__(self, alpha: float = 1.0, *, num_classes: int, p: float = 1.0) -> None:
        _validate_init(alpha, num_classes, p)
        self.alpha = alpha
        self.num_classes = num_classes
        self.p = p

    def __call__(self, batch: list[object]) -> tuple[Tensor, Tensor]:
        images_obj, labels_obj = default_collate(batch)
        images = images_obj if isinstance(images_obj, Tensor) else images_obj
        labels = labels_obj if isinstance(labels_obj, Tensor) else labels_obj
        if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
            raise TypeError(
                "CutMixCollator expects each sample to yield (image_tensor, "
                f"label_tensor); got images={type(images).__name__}, "
                f"labels={type(labels).__name__}"
            )
        targets = _to_soft_targets(labels, self.num_classes)
        if float(lucid.rand(1).item()) >= self.p:
            return images, targets

        b, _, h, w = (int(d) for d in images.shape)
        lam = _sample_lambda(self.alpha)
        cut_w = int(w * math.sqrt(max(0.0, 1.0 - lam)))
        cut_h = int(h * math.sqrt(max(0.0, 1.0 - lam)))
        if cut_w == 0 or cut_h == 0:
            # Cut box collapsed to nothing → no mixing this batch.
            return images, targets

        cx = _random.randint(0, w)
        cy = _random.randint(0, h)
        bx1 = max(cx - cut_w // 2, 0)
        bx2 = min(cx + cut_w // 2, w)
        by1 = max(cy - cut_h // 2, 0)
        by2 = min(cy + cut_h // 2, h)
        # Effective lambda after border clamping (CutMix paper Eq. 1).
        lam_eff = 1.0 - ((bx2 - bx1) * (by2 - by1)) / float(w * h)

        # Build keep mask: 1 outside cut box, 0 inside; broadcasts over B + C.
        inner = lucid.zeros(1, by2 - by1, bx2 - bx1, dtype=images.dtype)
        from lucid.utils.transforms import functional as F

        keep_1c = F.pad(inner, (bx1, w - bx2, by1, h - by2), value=1.0)  # (1, H, W)
        keep = keep_1c[None]  # (1, 1, H, W) — broadcasts over batch + channels

        perm = lucid.randperm(b)
        mixed_images = images * keep + images[perm] * (1.0 - keep)
        mixed_targets = lam_eff * targets + (1.0 - lam_eff) * targets[perm]
        return mixed_images, mixed_targets

    @override
    def __repr__(self) -> str:
        return f"CutMixCollator(alpha={self.alpha}, " f"num_classes={self.num_classes}, p={self.p})"


# ── Random mixup-or-cutmix ───────────────────────────────────────────


class RandomMixupCutMixCollator:
    r"""Pick :class:`MixupCollator` or :class:`CutMixCollator` per batch.

    The reference-framework "RandomMixup+CutMix" recipe: each batch
    independently chooses one of the two collators uniformly, then
    that collator decides whether to fire (per its own ``p``).

    Parameters
    ----------
    mixup_alpha : float, optional, default=0.2
    cutmix_alpha : float, optional, default=1.0
    num_classes : int
    p : float, optional, default=1.0
        Probability that *either* collator runs (vs the batch being
        returned unchanged with soft labels).
    switch_prob : float, optional, default=0.5
        When the policy fires, probability of choosing CutMix over
        Mixup.
    """

    def __init__(
        self,
        *,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        num_classes: int,
        p: float = 1.0,
        switch_prob: float = 0.5,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not 0.0 <= switch_prob <= 1.0:
            raise ValueError(f"switch_prob must be in [0, 1], got {switch_prob}")
        self.mixup = MixupCollator(alpha=mixup_alpha, num_classes=num_classes, p=1.0)
        self.cutmix = CutMixCollator(alpha=cutmix_alpha, num_classes=num_classes, p=1.0)
        self.num_classes = num_classes
        self.p = p
        self.switch_prob = switch_prob

    def __call__(self, batch: list[object]) -> tuple[Tensor, Tensor]:
        if float(lucid.rand(1).item()) >= self.p:
            # Neither fires — soft-label the raw batch for shape consistency.
            images_obj, labels_obj = default_collate(batch)
            if not isinstance(images_obj, Tensor) or not isinstance(labels_obj, Tensor):
                raise TypeError("RandomMixupCutMixCollator expects (image, label) tuples")
            return images_obj, _to_soft_targets(labels_obj, self.num_classes)
        if float(lucid.rand(1).item()) < self.switch_prob:
            return self.cutmix(batch)
        return self.mixup(batch)

    @override
    def __repr__(self) -> str:
        return (
            f"RandomMixupCutMixCollator(mixup_alpha={self.mixup.alpha}, "
            f"cutmix_alpha={self.cutmix.alpha}, "
            f"num_classes={self.num_classes}, p={self.p}, "
            f"switch_prob={self.switch_prob})"
        )
