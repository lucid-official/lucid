r"""The trainable middle of PSOFT: an ``r x r`` orthogonal matrix between two frozen factors, with two diagonal scalings.

PSOFT (Wu et al., 2025, arXiv:2505.11235) puts the matrix between the factors of the weight's
principal subspace and trains only it (section 4.1, Eq. 6-8),

.. math:: C = \operatorname{diag}(\beta)\,R\,\operatorname{diag}(\alpha),

where :math:`R` stays orthogonal through the Cayley parameterisation of a skew-symmetric
:math:`Q` (section 4.2 and Appendix C), the vectors start at one (Eq. 8), and ``weight`` holds only
the :math:`r(r-1)/2` entries above the diagonal of :math:`Q`.  ``vector_b`` is :math:`\beta`, on the
left of :math:`R`, and ``vector_a`` is :math:`\alpha`, on its right: this is the matrix of the weight's
``out x in`` layout, the transpose of the paper's ``d x n`` one, where :math:`\alpha` is the one on the left.
"""

from typing import override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._cayley import _Cayley

#: The standard deviation of the ``init_weights=False`` start of ``R``, the reference's.
_RANDOM_STD = 0.1
#: Added to a norm before a generator is rescaled by it, so a zero generator is not divided by zero.
_NORM_FLOOR = 1e-12


class OrthLayer(nn.Module):
    r"""The matrix :math:`C = \operatorname{diag}(\beta)\,R\,\operatorname{diag}(\alpha)`, the parameters that make it, and how to apply it.

    Parameters
    ----------
    size : int
        The rank :math:`r` of the principal subspace.
    orth : bool, default=True
        Keep :math:`R` orthogonal with the Cayley transform of a skew-symmetric matrix whose
        :math:`r(r-1)/2` upper entries are trained; ``False`` trains :math:`R` itself.
    mag_b, mag_a : bool, default=True
        Whether the vector left of (``mag_b``) or right of (``mag_a``) :math:`R` is trained.
    use_cayley_neumann : bool, default=False
        Build :math:`R` from a truncated Neumann series instead of an exact solve.
    num_cayley_neumann_terms : int, default=5
        Terms of the series.
    cayley_neumann_eps : float, optional
        With the series, rescale the generator whenever its Frobenius norm exceeds this.
    init_weights : bool, default=True
        Start at the identity (the generator at zero, the vectors at one); ``False`` perturbs
        :math:`R` by a normal of standard deviation 0.1.
    device, dtype
        Where and in what dtype the parameters are made.
    """

    def __init__(
        self,
        size: int,
        orth: bool = True,
        mag_b: bool = True,
        mag_a: bool = True,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        cayley_neumann_eps: float | None = None,
        init_weights: bool = True,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Make the generator (or the free matrix) and the two vectors, on ``device`` in ``dtype``."""
        super().__init__()
        self.size = size
        self.orth = orth
        self.mag_b = mag_b
        self.mag_a = mag_a
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.cayley_neumann_eps = cayley_neumann_eps
        weight = (
            lucid.zeros(size * (size - 1) // 2, device=device, dtype=dtype)
            if orth
            else lucid.eye(size, device=device, dtype=dtype)
        )
        if not init_weights:
            with lucid.no_grad():
                weight = weight + _RANDOM_STD * lucid.randn(
                    *weight.shape, device=device, dtype=dtype
                )
        self.weight = Parameter(weight)
        self.register_parameter(
            "vector_b", Parameter(lucid.ones(size, device=device, dtype=dtype)) if mag_b else None
        )
        self.register_parameter(
            "vector_a", Parameter(lucid.ones(size, device=device, dtype=dtype)) if mag_a else None
        )

    def _generator(self) -> Tensor:
        """The ``(1, r, r)`` skew-symmetric :math:`Q` whose entries above the diagonal are ``weight``, row by row."""
        return _Cayley.skew(self.weight.reshape(1, -1), self.size)

    @staticmethod
    def _bounded(generator: Tensor, bound: float) -> Tensor:
        """``generator`` scaled down to a Frobenius norm of ``bound`` if it exceeds it (and is finite)."""
        norm = (generator * generator).sum().sqrt()
        shrink = lucid.where(
            lucid.isfinite(norm) & (norm > bound),
            bound / (norm + _NORM_FLOOR),
            lucid.ones_like(norm),
        )
        return generator * shrink

    def get_matrix(self) -> Tensor:
        r"""Return :math:`C`, the ``(r, r)`` matrix the layer applies between its factors.

        :math:`R = (I + Q)(I - Q)^{-1}`: exact, or the Neumann series
        :math:`I + 2Q + 2Q^2 + \dots + Q^{t-1}` with its last term halved, built from
        ``weight`` as :math:`Q`.  The vectors then scale its rows (``vector_b``) and columns
        (``vector_a``).
        """
        if not self.orth:
            matrix: Tensor = self.weight
        else:
            generator = self._generator()
            if self.use_cayley_neumann:
                if self.cayley_neumann_eps is not None:
                    generator = self._bounded(generator, self.cayley_neumann_eps)
                rotation = _Cayley.neumann(generator, int(self.num_cayley_neumann_terms))
            else:
                # (I + Q)(I - Q)^-1 is the exact transform of -Q, which is what ``_Cayley`` solves.
                rotation = _Cayley.exact(-generator)
            matrix = rotation[0]
        if self.vector_b is not None:
            matrix = self.vector_b.unsqueeze(1) * matrix
        if self.vector_a is not None:
            matrix = matrix * self.vector_a.unsqueeze(0)
        return matrix

    @override
    def forward(self, input: Tensor) -> Tensor:
        r""":math:`x\,C^\top`, with ``input`` of width ``r`` on its last axis."""
        matrix = self.get_matrix()
        return input.to(matrix.dtype) @ matrix.mT

    @override
    def extra_repr(self) -> str:
        """The size and the options that shape the matrix."""
        return (
            f"size={self.size}, orth={self.orth}, use_cayley_neumann={self.use_cayley_neumann}, "
            f"num_cayley_neumann_terms={int(self.num_cayley_neumann_terms)}, "
            f"cayley_neumann_eps={self.cayley_neumann_eps}, mag_a={self.mag_a}, mag_b={self.mag_b}"
        )
