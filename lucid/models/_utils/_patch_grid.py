"""The size check every fixed-grid patch model owes its input.

A ViT-style network holds one position entry per patch of the grid it was
built for, so it takes exactly one input size.  The check is shared across
domains — ViT and I-JEPA (vision), CLIP's image tower (multimodal), and the
DiT family's latents (generative) — so it lives beside neither: each
family's patch embedding calls it, and the generative ``check_latent_size``
delegates to it with its own wording.
"""

__all__ = ["check_patch_grid"]


def check_patch_grid(
    owner: str,
    height: int,
    width: int,
    *,
    image_size: int,
    patch_size: int,
    noun: str = "input",
    size_field: str = "image_size",
) -> None:
    """Refuse an input the position table of a fixed-grid patch model was not built for.

    The model takes exactly one size, ``image_size`` squared.  The reference
    ViT, CLIP and I-JEPA embeddings refuse every other size unless a caller
    opts into position-table interpolation, which this zoo does not offer.
    Without the check the patch embedding, a stride-``patch_size``
    convolution, failed in three ways:

    * a side that is not a multiple of the patch makes the convolution drop
      the remainder, so a 47x47 image answered as its 32x32 crop with the
      border unread and no error (a latent of 9 against an 8 build of 2 px
      patches did the same, with an output of the built size rather than the
      input's);
    * another size on the patch grid broadcast against the table and failed
      as an engine ``ShapeMismatch`` naming neither size;
    * the same patch count laid out another way (16x64 against a 32x32 build
      of 8 px patches) fit the table and answered with every patch reading
      another patch's position entry.

    The configs already refuse a built size that is not a multiple of the
    patch, so ``image_size`` here is always a whole grid.

    Parameters
    ----------
    owner : str
        The model's name, for the message.
    height, width : int
        The input's spatial size.
    image_size : int
        Side of the square input the position table was built for.
    patch_size : int
        Side of a patch.
    noun : str, default="input"
        What the model reads, in the message: ``"input"`` for an image,
        ``"latent"`` for a latent model.
    size_field : str, default="image_size"
        The config field a caller would change to rebuild for another size
        (``"sample_size"`` for a latent model).

    Raises
    ------
    ValueError
        If ``(height, width)`` is not ``(image_size, image_size)``.

    Examples
    --------
    >>> from lucid.models._utils._patch_grid import check_patch_grid
    >>> check_patch_grid("ViT", 32, 32, image_size=32, patch_size=16)
    >>> check_patch_grid("ViT", 47, 47, image_size=32, patch_size=16)
    Traceback (most recent call last):
        ...
    ValueError: ViT is built for 32x32 inputs: its position table holds one entry per patch of a 2x2 grid of 16px patches.  Got 47x47 pixels, which is not a multiple of the 16px patch, so the patch embedding would silently drop 15 rows and 15 columns at the border.  Resize the input to 32x32.
    """
    if height == image_size and width == image_size:
        return
    p = patch_size
    side = image_size // p
    if height % p or width % p:
        got = (
            f"which is not a multiple of the {p}px patch, so the patch embedding "
            f"would silently drop {height % p} rows and {width % p} columns at the border"
        )
    else:
        got = f"which is {height // p}x{width // p} patches"
    rebuild = (
        f", or build the model with {size_field}={height}, whose position "
        "table is then sized for that grid"
        if height == width and height % p == 0
        else ""
    )
    raise ValueError(
        f"{owner} is built for {image_size}x{image_size} {noun}s: its position "
        f"table holds one entry per patch of a {side}x{side} grid of {p}px "
        f"patches.  Got {height}x{width} pixels, {got}.  "
        f"Resize the {noun} to {image_size}x{image_size}{rebuild}."
    )
