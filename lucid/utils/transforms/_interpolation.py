"""Interpolation modes for resampling transforms.

A small ``str``-valued enum so callers get autocomplete + validation
while the value still flows straight into
:func:`lucid.nn.functional.interpolate` (which takes a string).
"""

import enum
from typing import override


class Interpolation(str, enum.Enum):
    """Resampling mode for resize-style transforms.

    Subclasses ``str`` so members compare equal to their wire value
    (``Interpolation.BILINEAR == "bilinear"``) and pass through to the
    engine's ``interpolate`` without conversion.

    Examples
    --------
    >>> import lucid.utils.transforms as T
    >>> [mode.name for mode in T.Interpolation][:4]
    ['NEAREST', 'LINEAR', 'BILINEAR', 'BICUBIC']
    >>> T.Resize(16, 16, interpolation=T.Interpolation.NEAREST)
    Resize(height=16, width=16, p=1.0)
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"

    @override
    def __str__(self) -> str:
        return self.value


# Modes for which ``interpolate`` accepts an ``align_corners`` flag.
_ALIGN_CORNERS_MODES = frozenset(
    {
        Interpolation.LINEAR,
        Interpolation.BILINEAR,
        Interpolation.BICUBIC,
        Interpolation.TRILINEAR,
    }
)


# OpenCV ``INTER_*`` codes → Lucid modes, so Albumentations-style
# ``interpolation=cv2.INTER_LINEAR`` (an int) is accepted verbatim.
_CV2_CODES = {
    0: Interpolation.NEAREST,  # INTER_NEAREST
    1: Interpolation.BILINEAR,  # INTER_LINEAR
    2: Interpolation.AREA,  # INTER_AREA
    3: Interpolation.BICUBIC,  # INTER_CUBIC
    4: Interpolation.BICUBIC,  # INTER_LANCZOS4 → nearest available
}


def as_interpolation(mode: str | int | Interpolation) -> Interpolation:
    r"""Coerce a string, OpenCV ``INTER_*`` int code, or enum member
    to an :class:`Interpolation` value.

    Accepts the three forms callers naturally reach for:

    * an :class:`Interpolation` member — returned unchanged;
    * a string (``"bilinear"`` / ``"nearest"`` / ...) — parsed via
      the enum's value lookup;
    * an OpenCV ``cv2.INTER_*`` int code — looked up in the
      built-in mapping for Albumentations parity.

    Parameters
    ----------
    mode : str or int or Interpolation
        Mode to coerce.  ``bool`` is rejected because it would
        otherwise satisfy ``isinstance(..., int)``.

    Returns
    -------
    Interpolation
        Canonical enum value usable with
        :func:`lucid.nn.functional.interpolate`.

    Raises
    ------
    ValueError
        Unknown string, unknown OpenCV code, or ``bool`` input.
    """
    if isinstance(mode, Interpolation):
        return mode
    if isinstance(mode, int) and not isinstance(mode, bool):
        if mode not in _CV2_CODES:
            raise ValueError(
                f"unknown OpenCV interpolation code {mode}; "
                f"expected one of {sorted(_CV2_CODES)}"
            )
        return _CV2_CODES[mode]
    try:
        return Interpolation(mode)
    except ValueError:
        valid = ", ".join(m.value for m in Interpolation)
        raise ValueError(f"unknown interpolation {mode!r}; expected one of: {valid}") from None
