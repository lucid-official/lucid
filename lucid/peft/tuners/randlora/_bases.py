r"""The random bases RandLoRA shares: how they are drawn.

One pair :math:`(A, B)` serves every layer of an adapter, sized for the largest
layer.  :math:`A` has shape ``(r, 1, d_min)`` and :math:`B` ``(d_max, n, r)``
with :math:`n = \lceil d_{min} / r \rceil` bases; a layer slices them to its size.
Both are drawn on the CPU from a generator seeded with ``projection_prng_key`` and
then moved, so the same key gives the same bases on every device — which is what
lets an adapter saved without them (``save_projection=False``) be restored — and
both are divided by their standard deviation, as the reference does.
"""

import math
from typing import NamedTuple

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike


def _num_bases(min_dim: int, r: int) -> int:
    """How many rank-``r`` bases a layer whose smaller side is ``min_dim`` needs to reach full rank."""
    return math.ceil(min_dim / r)


def _ternary(uniform: Tensor, sparsity: float) -> Tensor:
    """``uniform`` (draws from ``[0, 1)``) as -1, 0 or 1: -1 below ``1 / 2s``, 1 above ``1 - 1 / 2s``."""
    edge = 1.0 / (2.0 * sparsity)
    return (uniform > 1.0 - edge).to(uniform.dtype) - (uniform < edge).to(uniform.dtype)


def _normalised(basis: Tensor) -> Tensor:
    return basis / basis.std()


class _Bases(NamedTuple):
    """The shared pair: ``randlora_A`` is ``(r, 1, d_min)`` and ``randlora_B`` is ``(d_max, n, r)``."""

    randlora_A: Tensor
    randlora_B: Tensor

    @staticmethod
    def draw(
        key: int,
        r: int,
        max_dim: int,
        min_dim: int,
        sparsity: float | None,
        device: DeviceLike,
        dtype: DTypeLike,
    ) -> _Bases:
        """The pair for a largest layer with sides ``max_dim`` and ``min_dim``, drawn from ``key``.

        ``sparsity`` is ``None`` for dense bases (Kaiming-uniform, ``a = sqrt 5``, so
        uniform in :math:`\\pm 1/\\sqrt{fan_{in}}`); a number ``s`` draws ternary bases
        that are non-zero with probability ``1 / s``.  ``A`` comes first, then
        ``B`` one basis at a time (dense) or whole (sparse), from one generator.
        """
        generator = lucid.Generator(key, device="cpu")
        bases = _num_bases(min_dim, r)

        def uniform(shape: tuple[int, ...], low: float, high: float) -> Tensor:
            tensor = lucid.empty(*shape, device="cpu")
            tensor.uniform_(low, high, generator=generator)
            return tensor

        if sparsity is None:
            first = uniform((r, 1, min_dim), -1.0 / math.sqrt(min_dim), 1.0 / math.sqrt(min_dim))
            bound = 1.0 / math.sqrt(r)
            second = lucid.cat(
                [uniform((max_dim, 1, r), -bound, bound) for _ in range(bases)], dim=1
            )
        else:
            first = _ternary(uniform((r, 1, min_dim), 0.0, 1.0), sparsity)
            second = _ternary(uniform((max_dim, bases, r), 0.0, 1.0), sparsity)
        return _Bases(
            _normalised(first).to(device=device, dtype=dtype),
            _normalised(second).to(device=device, dtype=dtype),
        )
