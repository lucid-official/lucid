"""Shared building blocks + policy classes for AutoAugment family.

All three policies sample from the same 14-op vocabulary and the same
magnitude → parameter mapping; only the *selection* differs:

* :class:`AutoAugment` — fixed 25 sub-policies (each = 2 ops with their
  own probabilities and magnitudes), one policy sampled per image.
* :class:`RandAugment` — ``num_ops`` ops sampled uniformly (with
  replacement), all sharing a single ``magnitude``.
* :class:`TrivialAugmentWide` — exactly 1 op sampled uniformly,
  magnitude sampled uniformly in ``[0, num_magnitude_bins)``.

All policies are :class:`PhotometricTransform` subclasses: even when
the sampled op is geometric (Shear / Translate / Rotate), it applies
to the image only — masks, boxes, and keypoints pass through.  This
matches the reference-framework convention; geometric augmentation of
non-image targets should be done separately with explicit transforms
such as :class:`~lucid.utils.transforms.Affine`.

All ops act on float images in ``[0, 1]`` with ``(C, H, W)`` or
``(B, C, H, W)`` layout — matching the rest of
:mod:`lucid.utils.transforms`.
"""

import math
from dataclasses import dataclass
from typing import Callable, final, override

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import PhotometricTransform
from lucid.utils.transforms._interpolation import Interpolation

# Full op vocabulary shared by all three policies.  Order matches the
# reference-framework ``_AUGMENTATION_SPACE`` so magnitude indices line
# up across implementations.
_OP_NAMES: tuple[str, ...] = (
    "Identity",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "Brightness",
    "Color",
    "Contrast",
    "Sharpness",
    "Posterize",
    "Solarize",
    "AutoContrast",
    "Equalize",
    "Invert",
)


def _magnitudes_for(op: str, num_bins: int) -> tuple[list[float], bool]:
    r"""Return the (magnitudes, signed) entry of the magnitude lookup table.

    ``num_bins`` evenly partitions each op's parameter range, mirroring
    the reference framework's table.  ``signed`` toggles whether a sign
    is sampled with 50/50 probability at call time (e.g. ShearX is
    symmetric, Posterize is not).

    Parameters
    ----------
    op : str
        One of :data:`_OP_NAMES`.
    num_bins : int
        Number of magnitude bins (``> 1``).  Standard recipes use
        ``31`` (so ``M ∈ {0, 1, ..., 30}``).

    Returns
    -------
    (magnitudes, signed) : tuple of (list of float, bool)
        ``magnitudes`` indexable by ``M`` to get the parameter for that
        op; empty list signals the op ignores magnitude (e.g.
        ``Identity``, ``AutoContrast``, ``Equalize``).
    """
    if num_bins <= 1:
        raise ValueError(f"num_bins must be > 1, got {num_bins}")
    last = num_bins - 1

    def lin(hi: float) -> list[float]:
        return [hi * i / last for i in range(num_bins)]

    if op == "Identity":
        return ([], False)
    if op in {"ShearX", "ShearY"}:
        return (lin(0.3), True)
    if op in {"TranslateX", "TranslateY"}:
        # Same fraction-of-image-size constant as the reference
        # framework (≈ 0.453 = 150 / 331).
        return (lin(150.0 / 331.0), True)
    if op == "Rotate":
        return (lin(30.0), True)  # degrees
    if op in {"Brightness", "Color", "Contrast", "Sharpness"}:
        return (lin(0.9), True)  # factor offset; factor = 1 + signed_mag
    if op == "Posterize":
        # Bits stepped 8 → 4 across the bins (rounded).
        return (
            [int(8 - round(i / (last / 4.0))) for i in range(num_bins)],
            False,
        )
    if op == "Solarize":
        # Threshold stepped 1.0 → 0.0.
        return (lin(1.0)[::-1], False)
    if op in {"AutoContrast", "Equalize", "Invert"}:
        return ([], False)
    raise KeyError(f"unknown op {op!r}; expected one of {_OP_NAMES}")


# Ops whose magnitude has no inherent sign — sign sampled per call.
SIGNED_OPS: frozenset[str] = frozenset(
    {
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Rotate",
        "Brightness",
        "Color",
        "Contrast",
        "Sharpness",
    }
)


# Ops that take no magnitude (the lookup list is empty).
NO_MAGNITUDE_OPS: frozenset[str] = frozenset({"Identity", "AutoContrast", "Equalize", "Invert"})


def _apply_affine(
    img: Tensor,
    *,
    shear_x_deg: float = 0.0,
    shear_y_deg: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    angle_deg: float = 0.0,
    interpolation: Interpolation,
    fill: float,
    cx: float | None = None,
    cy: float | None = None,
) -> Tensor:
    r"""Compose ``affine_matrix`` + ``warp_affine`` for the geometric ops.

    Optional ``cx`` / ``cy`` override the anchor point.  Default is the
    image centre ``((w-1)/2, (h-1)/2)`` — matching reference-framework
    convention for Rotate / Translate.  For Shear, the reference
    framework anchors at the top-left corner ``(0, 0)`` (legacy
    AutoAugment paper convention); pass ``cx=0, cy=0`` explicitly to
    reproduce that behaviour.

    Honours the ``fill`` arg as a *constant* value over the entire
    out-of-bounds region.  Lucid's :func:`warp_affine` plumbs ``fill``
    to ``grid_sample`` which only supports ``"zeros"`` / ``"border"``
    / ``"reflection"`` padding modes (no true constant fill), so this
    helper renders the constant fill explicitly via a mask-and-add:

    .. math::

        out(p) = \mathrm{warp}_0(p) + fill \cdot (1 - mask(p))

    where ``warp_0`` is the zero-padded warp of the image and ``mask``
    is the zero-padded warp of an all-ones image (the in-bounds
    sampling weight at every output pixel).  This is mathematically
    equivalent to grid-sampling with a true constant outside the
    source — fully in-bounds pixels (``mask = 1``) pass through, fully
    out-of-bounds pixels (``mask = 0``) become ``fill``, and bilinear
    border interpolations get the correct fractional blend.
    """
    h, w = F._spatial_hw(img)
    if cx is None:
        cx = (w - 1) / 2.0
    if cy is None:
        cy = (h - 1) / 2.0
    matrix = F.affine_matrix(
        cx=cx,
        cy=cy,
        scale=1.0,
        angle_deg=angle_deg,
        shear_x_deg=shear_x_deg,
        shear_y_deg=shear_y_deg,
        translate_x=translate_x,
        translate_y=translate_y,
        device=img.device,
    )
    mode = "nearest" if interpolation is Interpolation.NEAREST else "bilinear"
    warped = F.warp_affine(img, matrix, (h, w), mode=mode, fill=0.0)
    if fill == 0.0:
        return warped
    # Validity mask: warp an all-ones image with zero padding → 1 where
    # the source was fully in-bounds, fractional at borders, 0 outside.
    ones = lucid.ones_like(img)
    mask = F.warp_affine(ones, matrix, (h, w), mode=mode, fill=0.0)
    return warped + fill * (1.0 - mask)


def apply_op(
    img: Tensor,
    op: str,
    magnitude: float,
    *,
    interpolation: Interpolation = Interpolation.BILINEAR,
    fill: float = 0.0,
) -> Tensor:
    r"""Dispatch one AutoAugment-family op on ``img`` with the given magnitude.

    ``magnitude`` is the *signed* parameter value (already drawn from the
    lookup table and sign-flipped if applicable); ops in
    :data:`NO_MAGNITUDE_OPS` ignore it.  Magnitude conventions:

    * geometric ops — ``ShearX/Y`` use the magnitude as ``tan(angle)``
      (converted to degrees here); ``TranslateX/Y`` use it as a
      fraction of the spatial size; ``Rotate`` uses it directly as
      degrees.
    * photometric "adjust" ops — ``Brightness``, ``Color``,
      ``Contrast``, ``Sharpness`` interpret ``magnitude`` as a factor
      *offset* (factor = ``1 + magnitude``); ``magnitude == 0`` is the
      identity.
    * tone ops — ``Posterize`` uses ``int(magnitude)`` as the bit count;
      ``Solarize`` uses ``magnitude`` as the threshold in ``[0, 1]``.
    * ``AutoContrast``, ``Equalize``, ``Identity`` take no magnitude.

    Parameters
    ----------
    img : Tensor
        Float image in ``[0, 1]`` of shape ``(C, H, W)`` or
        ``(B, C, H, W)``.
    op : str
        One of :data:`_OP_NAMES`.
    magnitude : float
        Signed magnitude.  Ignored for ops in :data:`NO_MAGNITUDE_OPS`.
    interpolation : Interpolation, optional, keyword-only
        Used by the geometric ops.
    fill : float, optional, keyword-only, default=0.0
        Border fill value for the geometric warps.

    Returns
    -------
    Tensor
        Augmented image, same shape and dtype as ``img``.

    Raises
    ------
    KeyError
        If ``op`` is not in :data:`_OP_NAMES`.
    """
    if op == "Identity":
        return img
    if op == "ShearX":
        # Two convention differences from Lucid's default ``affine_matrix``:
        # 1. Reference framework anchors shear at ``center=[0, 0]`` (top-
        #    left, per legacy AutoAugment paper convention), not the image
        #    centre — pass ``cx=cy=0``.
        # 2. Reference framework's shear matrix uses the opposite sign
        #    convention from Lucid's forward-warp ``affine_matrix``
        #    (Lucid: ``y_out += tan·x``; reference: ``y_out -= tan·x``).
        #    Negate the angle to absorb this.
        deg = math.degrees(math.atan(magnitude))
        return _apply_affine(
            img,
            shear_x_deg=-deg,
            cx=0.0,
            cy=0.0,
            interpolation=interpolation,
            fill=fill,
        )
    if op == "ShearY":
        # Same two convention flips as ShearX (see comment above).
        deg = math.degrees(math.atan(magnitude))
        return _apply_affine(
            img,
            shear_y_deg=-deg,
            cx=0.0,
            cy=0.0,
            interpolation=interpolation,
            fill=fill,
        )
    if op == "TranslateX":
        _, w = F._spatial_hw(img)
        return _apply_affine(
            img,
            translate_x=magnitude * w,
            interpolation=interpolation,
            fill=fill,
        )
    if op == "TranslateY":
        h, _ = F._spatial_hw(img)
        return _apply_affine(
            img,
            translate_y=magnitude * h,
            interpolation=interpolation,
            fill=fill,
        )
    if op == "Rotate":
        # Negate to align with reference-framework / PIL rotation convention.
        # Lucid's ``affine_matrix`` uses the math convention (positive
        # degrees → counter-clockwise); PIL / reference_vision's ``F.rotate``
        # uses the image convention (positive degrees → clockwise).
        # Flipping the sign here makes ``apply_op("Rotate", +deg)`` agree
        # with the reference framework's ``_apply_op("Rotate", +deg)``.
        return _apply_affine(
            img,
            angle_deg=-magnitude,
            interpolation=interpolation,
            fill=fill,
        )
    if op == "Brightness":
        return F.adjust_brightness(img, 1.0 + magnitude)
    if op == "Color":
        return F.adjust_saturation(img, 1.0 + magnitude)
    if op == "Contrast":
        return F.adjust_contrast(img, 1.0 + magnitude)
    if op == "Sharpness":
        return F.adjust_sharpness(img, 1.0 + magnitude)
    if op == "Posterize":
        return F.posterize(img, int(magnitude))
    if op == "Solarize":
        return F.solarize(img, magnitude)
    if op == "AutoContrast":
        return F.autocontrast(img)
    if op == "Equalize":
        return F.equalize(img)
    if op == "Invert":
        return F.invert(img)
    raise KeyError(f"unknown op {op!r}; expected one of {_OP_NAMES}")


def sample_signed_magnitude(magnitudes: list[float], index: int) -> float:
    r"""Pick ``magnitudes[index]`` with a uniformly random sign.

    Used at call time for ops in :data:`SIGNED_OPS` so the geometric /
    adjust ops sweep both directions across the dataset.

    Parameters
    ----------
    magnitudes : list of float
        Lookup table for the op (from :func:`_magnitudes_for`).
    index : int
        Magnitude bin index, ``0 <= index < len(magnitudes)``.

    Returns
    -------
    float
        Signed magnitude.  Returns ``0.0`` if ``magnitudes`` is empty
        (no-magnitude ops should not reach this code path).
    """
    if not magnitudes:
        return 0.0
    value = float(magnitudes[index])
    return -value if float(lucid.rand(1).item()) < 0.5 else value


# Type alias for the dispatch signature (used by the policy classes).
OpFn = Callable[[Tensor, float], Tensor]


# ── policy classes ──────────────────────────────────────────────────


@final
@dataclass(frozen=True, slots=True)
class _SingleOpParams:
    """Sampled (op_name, signed_magnitude) for one call."""

    op_name: str
    magnitude: float


class TrivialAugmentWide(PhotometricTransform[_SingleOpParams]):
    r"""Trivial Augment Wide (Müller & Hutter, 2021 — arXiv:2103.10158).

    Each call draws one op uniformly from the 14-op vocabulary and one
    magnitude uniformly from ``[0, num_magnitude_bins)``.  Despite its
    simplicity it matches the accuracy of much more elaborate policies
    (AutoAugment, RandAugment) on ImageNet — making it the default
    "set-and-forget" augmentation in modern recipes.

    Parameters
    ----------
    num_magnitude_bins : int, optional, default=31
        Magnitude bin count.  The wide variant uses 31; the original
        TrivialAugment uses 10.  Larger values give a finer magnitude
        grid.
    interpolation : str or Interpolation, optional, default="bilinear"
        Resample mode for the geometric ops (Shear / Translate /
        Rotate).
    fill : float, optional, default=0.0
        Border fill value for the geometric warps.
    p : float, optional, default=1.0
        Probability of applying the policy at all.  TrivialAugment is
        designed to fire every call (Identity is one of the 14 ops, so
        ~7% of calls are no-ops by virtue of the op draw); the ``p``
        knob is provided for compositional consistency, not because
        ``p < 1`` is recommended.

    Notes
    -----
    Photometric (image-only) by Lucid convention — see the module
    docstring.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.transforms import TrivialAugmentWide
    >>> tf = TrivialAugmentWide()
    >>> x = lucid.rand(3, 224, 224)
    >>> y = tf(x)
    >>> tuple(y.shape)
    (3, 224, 224)
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        *,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
        fill: float = 0.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if num_magnitude_bins <= 1:
            raise ValueError(f"num_magnitude_bins must be > 1, got {num_magnitude_bins}")
        if isinstance(interpolation, str):
            interpolation = Interpolation(interpolation)
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    @override
    def make_params(self, img: Tensor) -> _SingleOpParams:
        r"""Sample per-call random parameters for :class:`TrivialAugmentWide`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        _SingleOpParams
            Carries the sampled ``op_name`` (uniform over the 14-op
            vocabulary) and ``magnitude`` — either ``0.0`` for ops in
            :data:`NO_MAGNITUDE_OPS`, or
            ``magnitudes[mag_idx]`` (with a random sign for ops in
            :data:`SIGNED_OPS`) where ``mag_idx`` is uniform over
            ``[0, num_magnitude_bins)``.
        """
        op_idx = _random.randint(0, len(_OP_NAMES))
        op_name = _OP_NAMES[op_idx]
        if op_name in NO_MAGNITUDE_OPS:
            return _SingleOpParams(op_name=op_name, magnitude=0.0)
        magnitudes, signed = _magnitudes_for(op_name, self.num_magnitude_bins)
        mag_idx = _random.randint(0, self.num_magnitude_bins)
        value = float(magnitudes[mag_idx])
        if signed and float(lucid.rand(1).item()) < 0.5:
            value = -value
        return _SingleOpParams(op_name=op_name, magnitude=value)

    @override
    def _apply_image(self, img: Tensor, params: _SingleOpParams) -> Tensor:
        return apply_op(
            img,
            params.op_name,
            params.magnitude,
            interpolation=self.interpolation,
            fill=self.fill,
        )

    @override
    def __repr__(self) -> str:
        return (
            f"TrivialAugmentWide(num_magnitude_bins={self.num_magnitude_bins}, "
            f"interpolation={self.interpolation.value!r}, fill={self.fill}, "
            f"p={self.p})"
        )


@final
@dataclass(frozen=True, slots=True)
class _RandAugmentParams:
    """Sampled (op_name, signed_magnitude) sequence for one call."""

    ops: tuple[tuple[str, float], ...]


class RandAugment(PhotometricTransform[_RandAugmentParams]):
    r"""RandAugment (Cubuk et al., 2020 — arXiv:1909.13719).

    Each call samples ``num_ops`` ops uniformly with replacement from
    the 14-op vocabulary and applies them sequentially, all sharing
    the same magnitude :math:`M` (signed per-op for symmetric ops).
    Reduces AutoAugment's two-level search space (policy + magnitude)
    to two scalar hyper-parameters that can be tuned with a simple
    grid sweep.

    Parameters
    ----------
    num_ops : int, optional, default=2
        Number of ops sampled per call (with replacement).  The
        reference-framework default for ImageNet is 2.
    magnitude : int, optional, default=9
        Integer index into the magnitude lookup table; must satisfy
        ``0 <= magnitude < num_magnitude_bins``.  Reference-framework
        recipes sweep this from 5 (small models) to 15 (large models).
    num_magnitude_bins : int, optional, default=31
        Magnitude bin count — same convention as
        :class:`TrivialAugmentWide`.
    interpolation : str or Interpolation, optional, default="bilinear"
    fill : float, optional, default=0.0
    p : float, optional, default=1.0

    Notes
    -----
    Photometric (image-only) by Lucid convention — see the module
    docstring.  The op draws are with replacement so a call may stack
    the same op twice (e.g. ``Rotate`` followed by ``Rotate``); this is
    the reference-framework behaviour.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.transforms import RandAugment
    >>> tf = RandAugment(num_ops=2, magnitude=9)
    >>> x = lucid.rand(3, 224, 224)
    >>> y = tf(x)
    >>> tuple(y.shape)
    (3, 224, 224)
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        *,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
        fill: float = 0.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if num_ops < 1:
            raise ValueError(f"num_ops must be >= 1, got {num_ops}")
        if num_magnitude_bins <= 1:
            raise ValueError(f"num_magnitude_bins must be > 1, got {num_magnitude_bins}")
        if not 0 <= magnitude < num_magnitude_bins:
            raise ValueError(f"magnitude must be in [0, {num_magnitude_bins}), got {magnitude}")
        if isinstance(interpolation, str):
            interpolation = Interpolation(interpolation)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    @override
    def make_params(self, img: Tensor) -> _RandAugmentParams:
        r"""Sample per-call random parameters for :class:`RandAugment`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        _RandAugmentParams
            Carries ``ops`` — a length-``num_ops`` tuple of
            ``(op_name, signed_magnitude)`` pairs.  Op names are drawn
            uniformly with replacement from the 14-op vocabulary; the
            magnitude shared across all ops is fixed at the
            ``self.magnitude`` lookup index, with a random sign for
            ops in :data:`SIGNED_OPS`.
        """
        ops: list[tuple[str, float]] = []
        for _ in range(self.num_ops):
            op_idx = _random.randint(0, len(_OP_NAMES))
            op_name = _OP_NAMES[op_idx]
            if op_name in NO_MAGNITUDE_OPS:
                ops.append((op_name, 0.0))
                continue
            magnitudes, signed = _magnitudes_for(op_name, self.num_magnitude_bins)
            value = float(magnitudes[self.magnitude])
            if signed and float(lucid.rand(1).item()) < 0.5:
                value = -value
            ops.append((op_name, value))
        return _RandAugmentParams(ops=tuple(ops))

    @override
    def _apply_image(self, img: Tensor, params: _RandAugmentParams) -> Tensor:
        out = img
        for op_name, mag in params.ops:
            out = apply_op(
                out,
                op_name,
                mag,
                interpolation=self.interpolation,
                fill=self.fill,
            )
        return out

    @override
    def __repr__(self) -> str:
        return (
            f"RandAugment(num_ops={self.num_ops}, magnitude={self.magnitude}, "
            f"num_magnitude_bins={self.num_magnitude_bins}, "
            f"interpolation={self.interpolation.value!r}, fill={self.fill}, "
            f"p={self.p})"
        )


# ── AutoAugment policy tables (Cubuk et al., 2019) ──────────────────
#
# Each sub-policy is two ``(op_name, probability, magnitude_idx)`` triples
# applied in sequence; the magnitudes are indices into the 10-bin lookup
# table (the original paper / reference framework's convention).  Each
# policy contains exactly 25 sub-policies — one is drawn uniformly per
# call.  Verbatim from the paper (arXiv:1805.09501) Tables 2, 6, 7.

_SubOp = tuple[str, float, int]
_SubPolicy = tuple[_SubOp, _SubOp]
_PolicyTable = tuple[_SubPolicy, ...]


_IMAGENET_POLICY: _PolicyTable = (
    (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
    (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
    (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
    (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
    (("Equalize", 0.4, 4), ("Rotate", 0.8, 8)),
    (("Solarize", 0.6, 3), ("Equalize", 0.6, 7)),
    (("Posterize", 0.8, 5), ("Equalize", 1.0, 2)),
    (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
    (("Equalize", 0.6, 8), ("Posterize", 0.4, 6)),
    (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
    (("Rotate", 0.4, 9), ("Equalize", 0.6, 2)),
    (("Equalize", 0.0, 7), ("Equalize", 0.8, 8)),
    (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
    (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
    (("Sharpness", 0.4, 7), ("Invert", 0.6, 8)),
    (("ShearX", 0.6, 5), ("Equalize", 1.0, 9)),
    (("Color", 0.4, 0), ("Equalize", 0.6, 3)),
    (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
    (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
)


_CIFAR10_POLICY: _PolicyTable = (
    (("Invert", 0.1, 7), ("Contrast", 0.2, 6)),
    (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
    (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
    (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)),
    (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
    (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
    (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
    (("Equalize", 0.6, 5), ("Equalize", 0.5, 1)),
    (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
    (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
    (("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)),
    (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
    (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
    (("Solarize", 0.5, 2), ("Invert", 0.0, 3)),
    (("Equalize", 0.2, 0), ("AutoContrast", 0.6, 0)),
    (("Equalize", 0.2, 8), ("Equalize", 0.6, 4)),
    (("Color", 0.9, 9), ("Equalize", 0.6, 6)),
    (("AutoContrast", 0.8, 4), ("Solarize", 0.2, 8)),
    (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
    (("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)),
    (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)),
    (("Equalize", 0.8, 8), ("Invert", 0.1, 3)),
    (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)),
)


_SVHN_POLICY: _PolicyTable = (
    (("ShearX", 0.9, 4), ("Invert", 0.2, 3)),
    (("ShearY", 0.9, 8), ("Invert", 0.7, 5)),
    (("Equalize", 0.6, 5), ("Solarize", 0.6, 6)),
    (("Invert", 0.9, 3), ("Equalize", 0.6, 3)),
    (("Equalize", 0.6, 1), ("Rotate", 0.9, 3)),
    (("ShearX", 0.9, 4), ("AutoContrast", 0.8, 3)),
    (("ShearY", 0.9, 8), ("Invert", 0.4, 5)),
    (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
    (("Invert", 0.9, 6), ("AutoContrast", 0.8, 1)),
    (("Equalize", 0.6, 3), ("Rotate", 0.9, 3)),
    (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
    (("ShearY", 0.8, 8), ("Invert", 0.7, 4)),
    (("Equalize", 0.9, 5), ("TranslateY", 0.6, 6)),
    (("Invert", 0.9, 4), ("Equalize", 0.6, 7)),
    (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
    (("Invert", 0.8, 5), ("TranslateY", 0.0, 2)),
    (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
    (("Invert", 0.6, 4), ("Rotate", 0.8, 4)),
    (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
    (("ShearX", 0.1, 6), ("Invert", 0.6, 5)),
    (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
    (("ShearY", 0.8, 4), ("Invert", 0.8, 8)),
    (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
    (("ShearY", 0.8, 5), ("AutoContrast", 0.7, 3)),
    (("ShearX", 0.7, 2), ("Invert", 0.1, 5)),
)


_POLICY_TABLES: dict[str, _PolicyTable] = {
    "imagenet": _IMAGENET_POLICY,
    "cifar10": _CIFAR10_POLICY,
    "svhn": _SVHN_POLICY,
}


@final
@dataclass(frozen=True, slots=True)
class _AutoAugmentParams:
    """Resolved ops to apply after sampling a sub-policy + probability gates."""

    ops: tuple[tuple[str, float], ...]


class AutoAugment(PhotometricTransform[_AutoAugmentParams]):
    r"""AutoAugment (Cubuk et al., 2019 — arXiv:1805.09501).

    Each call samples one of 25 fixed sub-policies uniformly; each
    sub-policy is two ``(op, probability, magnitude_idx)`` triples
    applied sequentially with their own probability gates.  Three
    policy tables are shipped, verbatim from the paper:

    * ``"imagenet"`` — Table 2 (default).  General-purpose vision
      classification.
    * ``"cifar10"`` — Table 6.  Small-image classification.
    * ``"svhn"`` — Table 7.  Digits / OCR-style data.

    Parameters
    ----------
    policy : str, optional, default="imagenet"
        Which policy table to sample sub-policies from.  Must be one
        of ``"imagenet"``, ``"cifar10"``, ``"svhn"``.
    num_magnitude_bins : int, optional, default=10
        Magnitude bin count.  The paper / reference framework uses
        ``10`` for AutoAugment (vs. ``31`` for RandAugment /
        TrivialAugmentWide); the policy tables index into this size.
    interpolation : str or Interpolation, optional, default="bilinear"
    fill : float, optional, default=0.0
    p : float, optional, default=1.0

    Notes
    -----
    Photometric (image-only) by Lucid convention — see the module
    docstring.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.transforms import AutoAugment
    >>> tf = AutoAugment(policy="imagenet")
    >>> x = lucid.rand(3, 224, 224)
    >>> y = tf(x)
    >>> tuple(y.shape)
    (3, 224, 224)
    """

    def __init__(
        self,
        policy: str = "imagenet",
        num_magnitude_bins: int = 10,
        *,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
        fill: float = 0.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if policy not in _POLICY_TABLES:
            raise ValueError(
                f"unknown policy {policy!r}; expected one of " f"{sorted(_POLICY_TABLES)}"
            )
        if num_magnitude_bins <= 1:
            raise ValueError(f"num_magnitude_bins must be > 1, got {num_magnitude_bins}")
        # Validate the table against the magnitude range we'll use.
        table = _POLICY_TABLES[policy]
        for sub in table:
            for op_name, prob, mag_idx in sub:
                if op_name not in _OP_NAMES:
                    raise ValueError(
                        f"policy {policy!r} references unknown op "
                        f"{op_name!r} (expected one of {_OP_NAMES})"
                    )
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"policy {policy!r}: probability {prob} out of [0, 1]")
                if not 0 <= mag_idx < num_magnitude_bins:
                    raise ValueError(
                        f"policy {policy!r}: magnitude_idx {mag_idx} out of "
                        f"[0, {num_magnitude_bins})"
                    )
        if isinstance(interpolation, str):
            interpolation = Interpolation(interpolation)
        self.policy = policy
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    @override
    def make_params(self, img: Tensor) -> _AutoAugmentParams:
        r"""Sample per-call random parameters for :class:`AutoAugment`.

        Parameters
        ----------
        img : Tensor
            Image tensor; not inspected, carried through for dispatch.

        Returns
        -------
        _AutoAugmentParams
            Carries ``ops`` — the resolved sequence of
            ``(op_name, signed_magnitude)`` pairs after sampling one
            sub-policy uniformly from the active policy table and
            gating each sub-op by its per-policy probability.
            Ops in :data:`NO_MAGNITUDE_OPS` are emitted with
            magnitude ``0.0``; ops in :data:`SIGNED_OPS` get a random
            sign on the table-lookup magnitude.
        """
        table = _POLICY_TABLES[self.policy]
        sub = table[_random.randint(0, len(table))]
        ops: list[tuple[str, float]] = []
        for op_name, prob, mag_idx in sub:
            if float(lucid.rand(1).item()) >= prob:
                continue
            if op_name in NO_MAGNITUDE_OPS:
                ops.append((op_name, 0.0))
                continue
            magnitudes, signed = _magnitudes_for(op_name, self.num_magnitude_bins)
            value = float(magnitudes[mag_idx])
            if signed and float(lucid.rand(1).item()) < 0.5:
                value = -value
            ops.append((op_name, value))
        return _AutoAugmentParams(ops=tuple(ops))

    @override
    def _apply_image(self, img: Tensor, params: _AutoAugmentParams) -> Tensor:
        out = img
        for op_name, mag in params.ops:
            out = apply_op(
                out,
                op_name,
                mag,
                interpolation=self.interpolation,
                fill=self.fill,
            )
        return out

    @override
    def __repr__(self) -> str:
        return (
            f"AutoAugment(policy={self.policy!r}, "
            f"num_magnitude_bins={self.num_magnitude_bins}, "
            f"interpolation={self.interpolation.value!r}, fill={self.fill}, "
            f"p={self.p})"
        )


# Public re-export — TrivialAugment / RandAugment / AutoAugment classes
# all import the helpers above, and the classes themselves are re-exported
# via ``lucid.utils.transforms``.
__all__ = [
    "_OP_NAMES",
    "SIGNED_OPS",
    "NO_MAGNITUDE_OPS",
    "_magnitudes_for",
    "apply_op",
    "sample_signed_magnitude",
    "TrivialAugmentWide",
    "RandAugment",
    "AutoAugment",
]
