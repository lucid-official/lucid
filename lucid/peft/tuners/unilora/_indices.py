r"""The projection of Uni-LoRA: which entry of the vector every LoRA parameter reads, and the scale it reads it with.

Uni-LoRA (Li et al., 2025, Sec. 3.2) lays the LoRA factors of every adapted layer end to end into one
vector :math:`\theta_D` and reads it out of :math:`\theta_d` as :math:`\theta_D = P\,\theta_d`, where each
row of :math:`P` is one-hot at an index drawn uniformly from the :math:`d` slots, the column of slot
:math:`j` scaled by :math:`1/\sqrt{n_j}` for the :math:`n_j` entries that read it.  Then
:math:`P^\top P = I_d` (Theorem 1): the map is an isometry.  ``P`` is never built; its indices and
scales are.

The draw is balanced: each slot is read ``floor(D / d)`` or ``ceil(D / d)`` times, the assignment
shuffled by the seed, so no slot is left unused for a small adapter.
"""

from collections.abc import Sequence
from typing import NamedTuple

import lucid
from lucid._tensor.tensor import Tensor


class _Assignment(NamedTuple):
    """One layer's share of the projection: where ``A`` (``(r, in)``) and ``B`` (``(out, r)``) read the vector, and their scales."""

    index_a: Tensor
    index_b: Tensor
    scale_a: Tensor
    scale_b: Tensor


def generate_index(lora_param_count: int, theta_d_length: int, proj_seed: int) -> Tensor:
    """The slot of ``theta_d`` each of ``lora_param_count`` LoRA parameters reads, balanced and shuffled by ``proj_seed``.

    Parameters
    ----------
    lora_param_count : int
        The number of LoRA parameters :math:`D`, non-negative.
    theta_d_length : int
        The length :math:`d` of the shared vector, positive.
    proj_seed : int
        Seed of the shuffle; the same seed gives the same assignment on every device.

    Returns
    -------
    Tensor
        ``(lora_param_count,)`` ``int64`` indices in ``[0, theta_d_length)``.

    Raises
    ------
    ValueError
        ``lora_param_count`` is negative or ``theta_d_length`` is not positive.
    """
    if lora_param_count < 0:
        raise ValueError(f"`lora_param_count` must be non-negative, got {lora_param_count}.")
    if theta_d_length <= 0:
        raise ValueError(f"`theta_d_length` must be positive, got {theta_d_length}.")
    generator = lucid.Generator(proj_seed, device="cpu")
    each, extra = divmod(lora_param_count, theta_d_length)
    slots = lucid.repeat_interleave(lucid.arange(theta_d_length, dtype=lucid.int64), each)
    if extra:
        # The slots read once more than the others: a random subset, as many as are left over.
        slots = lucid.cat([slots, lucid.randperm(theta_d_length, generator=generator)[:extra]])
    return slots[lucid.randperm(lora_param_count, generator=generator)]


def assign(
    shapes: Sequence[tuple[tuple[int, int], tuple[int, int]]], theta_d_length: int, proj_seed: int
) -> list[_Assignment]:
    """The projection of layers whose ``(A, B)`` shapes are ``shapes``, laid end to end in that order, ``A`` then ``B`` of each.

    The scales are :math:`1/\\sqrt{n_j}` for the ``n_j`` parameters of the whole model that read slot
    ``j``, so they are computed once over all layers, not per layer.

    Parameters
    ----------
    shapes : sequence of ((int, int), (int, int))
        ``((r, in), (out, r))`` of each layer.
    theta_d_length : int
        The length of the shared vector.
    proj_seed : int
        The seed of the draw.

    Returns
    -------
    list of _Assignment
        One per layer, on the CPU.
    """
    sizes = [(a[0] * a[1], b[0] * b[1]) for a, b in shapes]
    indices = generate_index(sum(n_a + n_b for n_a, n_b in sizes), theta_d_length, proj_seed)
    counts = lucid.bincount(indices, minlength=theta_d_length)
    # A slot nothing reads has no scale to speak of; one keeps the division defined.
    inverse = counts.to(lucid.float32).clamp(min=1.0).rsqrt()
    assigned: list[_Assignment] = []
    cursor = 0
    for (shape_a, shape_b), (n_a, n_b) in zip(shapes, sizes, strict=True):
        index_a = indices[cursor : cursor + n_a].reshape(*shape_a)
        index_b = indices[cursor + n_a : cursor + n_a + n_b].reshape(*shape_b)
        cursor += n_a + n_b
        assigned.append(_Assignment(index_a, index_b, inverse[index_a], inverse[index_b]))
    return assigned
