r"""What FRoD derives from the weights: the shared basis of a category, the sparse positions, and a layer's start.

FRoD (Wan et al., 2025, arXiv:2512.23485, section "Hierarchical Joint Decomposition") finds one
orthonormal-ish basis :math:`V` for every *category* of layer (the query projections of all blocks, say),
from the weights of the whole category at once.  The category's weights are stacked,
:math:`W^{(c)} = [W_1; \dots; W_L] = Q R` (Eq. 1-2); with :math:`Q_i` the rows of :math:`Q` that belong to
layer :math:`i`, :math:`T = \frac1L\sum_i (Q_i^\top Q_i + \pi I)^{-1}` (Eq. 3) is diagonalised as
:math:`T = Z\Lambda Z^\top` (Eq. 4) and the basis is :math:`V = R^\top Z` (Eq. 5).  Each layer then has
:math:`B_i = W_i V^{-\top}`, :math:`\Sigma_i = \operatorname{diag}(\lVert B_i\rVert_{col})` and
:math:`U_i = B_i\Sigma_i^{-1}` (Eq. 6), so :math:`W_i = U_i\Sigma_i V^\top` (Eq. 7).  When the stack has
fewer rows than columns the basis is the right singular vectors of the stack instead (the reference's
rule).  A basis is run through on the CPU in float32, as the reference does; ``U`` and :math:`\Sigma` in
float64.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.peft.tuners.fourierft._locations import _spectrum_positions

#: A column whose norm is below this has no direction: its ``U`` column is zero.
_NORM_FLOOR = 1e-12
#: The cut of the pseudo-inverse used when the basis cannot be inverted.
_PINV_RTOL = 1e-6


def _category_from_key(key: str) -> str:
    """The role a module plays in its block, read off its dotted path: the last two components, joined.

    ``encoder.layer.0.attention.self.query`` is ``self_query``; a path ending in a block number is
    its last component; BERT's attention output (``...attention.output.dense``) is ``attention_output`` so it
    does not meet the feed-forward ``output.dense``.  Layers of one category share a basis.
    """
    parts = key.split(".")
    if len(parts) == 1:
        return parts[0]
    if parts[-2].isdigit():
        return parts[-1]
    category = f"{parts[-2]}_{parts[-1]}"
    if category == "output_dense" and len(parts) >= 3 and parts[-3] == "attention":
        return "attention_output"
    return category


def _layer_index_from_key(key: str, fallback: int) -> int:
    """The block a module sits in: the integer after ``layers``, else the first integer in the path, else ``fallback``."""
    parts = key.split(".")
    if "layers" in parts:
        at = parts.index("layers") + 1
        if at < len(parts) and parts[at].isdigit():
            return int(parts[at])
    for part in parts:
        if part.isdigit():
            return int(part)
    return fallback


def _projection_from_weights(matrices: Sequence[Tensor], regularization_alpha: float) -> Tensor:
    """The shared basis :math:`V` (``in x in``) of ``matrices``, which are ``(out_i, in)`` float32 CPU weights."""
    stacked = lucid.cat(list(matrices), dim=0)
    if stacked.shape[0] < stacked.shape[1]:
        _, _, vh = linalg.svd(stacked, full_matrices=True)
        return vh.mT
    q, r = linalg.qr(stacked)
    dim = int(r.shape[1])
    # Where the stack is, not where the default device is: the sum below mixes them with its pieces.
    eye = lucid.eye(dim, dtype=r.dtype, device=r.device)
    gram = lucid.zeros(dim, dim, dtype=r.dtype, device=r.device)
    start = 0
    for matrix in matrices:
        rows = int(matrix.shape[0])
        piece = q[start : start + rows]
        gram = gram + linalg.inv(piece.mT @ piece + regularization_alpha * eye)
        start += rows
    _, vectors = linalg.eigh(gram / len(matrices))
    return r.mT @ vectors


def _sparse_count(in_features: int, sparse_rate: float) -> int:
    """How many off-diagonal entries of the ``in x in`` matrix are trainable: the rate's share, at least one if the rate is positive."""
    off_diagonal = in_features * in_features - in_features
    count = min(int(in_features * in_features * sparse_rate), off_diagonal)
    return max(1, count) if sparse_rate > 0 and off_diagonal > 0 else count


def _positions(total: int, count: int, key: int) -> tuple[int, ...]:
    """The first ``count`` entries of a random permutation of ``range(total)`` seeded by ``key``.

    Below the position count where the reference framework's generator draws 32-bit words this is
    that permutation (it is FourierFT's, which is checked against it); above, the same shuffle
    draws 64-bit words from a Mersenne Twister seeded by ``key``: deterministic, not compared.
    """
    try:
        return _spectrum_positions(total, count, key)
    except NotImplementedError:
        generator = random.Random(key)
        moved: dict[int, int] = {}
        picked: list[int] = []
        for i in range(count):
            j = i + generator.getrandbits(64) % (total - i)
            picked.append(moved.get(j, j))
            moved[j] = moved.get(i, i)
        return tuple(picked)


def _draw_pattern(
    in_features: int, sparse_rate: float, key: int, device: DeviceLike
) -> tuple[Tensor, Tensor]:
    """``(indices, size)``: the ``(2, nnz)`` positions drawn from ``key`` among the off-diagonal ones, and ``[in, in]``.

    The off-diagonal entries are numbered row by row; the first ``nnz`` of a seeded permutation of them are
    the trainable ones.
    """
    count = _sparse_count(in_features, sparse_rate)
    width = in_features - 1
    flat = _positions(in_features * width, count, key) if count else ()
    rows = [p // width for p in flat]
    columns = [p % width + (1 if p % width >= p // width else 0) for p in flat]
    indices = lucid.tensor([rows, columns], dtype=lucid.int64, device=device).reshape(2, count)
    return indices, lucid.tensor([in_features, in_features], dtype=lucid.int64, device=device)


class _Sparse(NamedTuple):
    """Where the trainable entries of the ``in x in`` matrix :math:`S` sit: row and column of each, and the size."""

    rows: Tensor
    columns: Tensor
    size: int

    @classmethod
    def of(cls, indices: Tensor, side: int) -> _Sparse:
        """The positions in ``indices`` (``(2, nnz)``) of a square matrix of ``side`` rows."""
        return cls(indices[0], indices[1], side)

    def apply(self, z: Tensor, values: Tensor) -> Tensor:
        r""":math:`z\,S^\top` for the rows ``z`` (``(b, in)``) and the entries ``values``, without forming :math:`S`."""
        contribution = z[:, self.columns] * values
        out: Tensor = lucid.zeros(z.shape[0], self.size, dtype=z.dtype, device=z.device).index_add(
            1, self.rows, contribution
        )
        return out

    def dense(self, values: Tensor) -> Tensor:
        """:math:`S` as a dense ``in x in`` matrix."""
        out: Tensor = lucid.zeros(
            self.size, self.size, dtype=values.dtype, device=values.device
        ).index_put((self.rows, self.columns), values)
        return out


class _Start(NamedTuple):
    """A layer's frozen left factor :math:`U` (``out x in``, unit columns) and the initial column norms :math:`\\Sigma`."""

    u: Tensor
    norms: Tensor

    @classmethod
    def of(cls, weight: Tensor, basis: Tensor, dtype: DTypeLike, device: DeviceLike) -> _Start:
        r""":math:`B = W V^{-\top}` by a solve in float64 on the CPU (a pseudo-inverse if :math:`V` is singular)."""
        wide = weight.detach().to("cpu").to(lucid.float64)
        v = basis.detach().to("cpu").to(lucid.float64)
        try:
            columns = linalg.solve(v, wide.mT).mT
        except linalg.LinAlgError:
            columns = wide @ linalg.pinv(v, rtol=_PINV_RTOL).mT
        norms = linalg.vector_norm(columns, dim=0)
        present = norms > _NORM_FLOOR
        safe = lucid.where(present, norms, lucid.ones_like(norms))
        u = lucid.where(present.unsqueeze(0), columns / safe, lucid.zeros_like(columns))
        return cls(u.to(device=device, dtype=dtype), norms.to(device=device, dtype=dtype))


def _grouped_weights(keys_and_weights: Sequence[tuple[str, Tensor]]) -> dict[str, list[Tensor]]:
    """The weights of each category, one per block in block order, as float32 CPU matrices.

    A module is filed under its category and its block's index; two in the same place keep the later.
    A block that lacks a category is skipped for that category.
    """
    by_block: dict[int, dict[str, Tensor]] = defaultdict(dict)
    for fallback, (key, weight) in enumerate(keys_and_weights):
        by_block[_layer_index_from_key(key, fallback)][_category_from_key(key)] = weight
    grouped: dict[str, list[Tensor]] = {}
    for category in sorted({c for block in by_block.values() for c in block}):
        grouped[category] = [
            block[category].detach().to("cpu").to(lucid.float32)
            for _, block in sorted(by_block.items())
            if category in block
        ]
    return grouped
