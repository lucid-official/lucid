"""What several functions of one Core ML package share: the weight blob.

Split from :mod:`lucid.coreml._build`.  ``export_functions`` builds every
entry point into one package and one ``weight.bin``; a weight two of them
use is written once and both programs point at the same bytes.  This module
owns that bookkeeping, so every storage format (float, int8, palettized,
sparse) goes through the one lookup.
"""

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._C.engine import BlobWriter, PackagePaths, TensorImpl

__all__ = ["Float", "Int8", "Palettized", "Placement", "Shared", "Sparse"]


@dataclasses.dataclass(frozen=True)
class Placement:
    """Where a constant's bytes sit in the blob, in one storage format.

    One subclass per format.  Each holds exactly what a second function
    needs to declare its own constant over the same bytes, so reusing a
    placement never re-reads or re-compresses the tensor.
    """


@dataclasses.dataclass(frozen=True)
class Float(Placement):
    offset: int


@dataclasses.dataclass(frozen=True)
class Int8(Placement):
    offset: int
    scale_bytes: bytes
    zero_bytes: bytes
    channels: int


@dataclasses.dataclass(frozen=True)
class Palettized(Placement):
    offset: int
    lut_offset: int
    lut_shape: list[int]
    key_bits: int


@dataclasses.dataclass(frozen=True)
class Sparse(Placement):
    value_offset: int
    mask_offset: int
    kept: int
    mask_len: int


@dataclasses.dataclass
class Shared:
    """A package and blob several functions are being written into.

    Parameters
    ----------
    paths : PackagePaths
        The package directory being filled.
    blob : BlobWriter
        The one ``weight.bin`` every function appends to.
    """

    paths: PackagePaths
    blob: BlobWriter
    # The key holds the traced tensor itself.  ``id()`` of a tensor is
    # reused as soon as it is freed — which function one's host constants
    # are, once it has been traced — so an ``id`` key hands a later
    # function's unrelated constant the bytes of an earlier one's.
    placed: dict[tuple[TensorImpl, type[Placement], tuple[object, ...]], Placement | None] = (
        dataclasses.field(default_factory=dict, init=False)
    )

    def place[P: Placement](
        self,
        source: TensorImpl,
        kind: type[P],
        variant: tuple[object, ...],
        write: Callable[[], P | None],
    ) -> P | None:
        """``write()``'s result for ``source`` stored as ``kind``, run once.

        Parameters
        ----------
        source : TensorImpl
            The traced tensor; the key is the object, not its address.
        kind : type of Placement
            The storage format asked for.
        variant : tuple
            Whatever else the bytes depend on (precision, target).
        write : callable
            Appends the bytes and returns where they went, or ``None`` when
            the tensor is not worth this format; that answer is kept too, so
            it is not worked out again.

        Returns
        -------
        Placement or None
            The placement, as ``kind``.
        """
        key = (source, kind, variant)
        if key not in self.placed:
            self.placed[key] = write()
        found = self.placed[key]
        return found if isinstance(found, kind) else None
