r"""The 2-D rotations RoAd applies to the features of an output, in the form they are computed.

RoAd (Liao and Monz, 2024, arXiv:2409.00119, section 3.2) multiplies the output of a layer by a
matrix :math:`R` that is block diagonal with :math:`2 \times 2` blocks (Eq. 3).  The blocks do not
pair neighbouring features: the features are cut into groups of ``group_size`` and element
:math:`i` of a group is paired with element :math:`i + \text{group}/2` of the same group.  A block
for the pair :math:`(i, j)` is

.. math:: \begin{pmatrix} f_i & -s_i \\ s_j & f_j \end{pmatrix}, \qquad f = \alpha\cos\theta,\quad s = \alpha\sin\theta ,

and one learned :math:`\alpha, \theta` per pair (``road_1``), per element (``road_2``) or two per
element (``road_4``) fill it.  Only the two coefficient vectors :math:`f` and :math:`s` are ever
formed: the rotation of a vector is :math:`f \odot y + s \odot \operatorname{rot}(y)`, with
:math:`\operatorname{rot}` moving the second half of each group to the first with a sign
(Eq. 4), which is what makes a batch of different adapters as cheap as an elementwise product.
"""

from typing import NamedTuple

import lucid
from lucid._tensor.tensor import Tensor


class _Rotation(NamedTuple):
    """The coefficients of one adapter's matrix :math:`R`, shaped ``(groups, 2, group/2)``.

    ``first`` is :math:`f`, the diagonal; ``second`` is :math:`s`, which sits in the pair's
    off-diagonal positions.  A rotation is never turned into its dense matrix except to be
    inspected (:meth:`matrix`).
    """

    first: Tensor
    second: Tensor

    @classmethod
    def of(cls, variant: str, group_size: int, theta: Tensor, alpha: Tensor) -> _Rotation:
        """The rotation of a ``variant``'s angles ``theta`` and scales ``alpha``.

        ``road_1`` holds ``out/2`` of each, shared by the two elements of a pair; ``road_2`` one per
        element; ``road_4`` two per element, the first for the diagonal and the second for the
        off-diagonal.
        """
        half = group_size // 2
        if variant == "road_1":
            shared = (alpha * theta.cos(), alpha * theta.sin())
            first, second = (part.reshape(-1, 1, half).expand(-1, 2, half) for part in shared)
        elif variant == "road_2":
            first = (alpha * theta.cos()).reshape(-1, 2, half)
            second = (alpha * theta.sin()).reshape(-1, 2, half)
        elif variant == "road_4":
            angles = theta.reshape(-1, 2, group_size)
            scales = alpha.reshape(-1, 2, group_size)
            first = (scales[:, 0, :] * angles[:, 0, :].cos()).reshape(-1, 2, half)
            second = (scales[:, 1, :] * angles[:, 1, :].sin()).reshape(-1, 2, half)
        else:
            raise ValueError(
                f"Unsupported variant {variant} for RoadLayer. "
                "Supported variants are road_1, road_2, and road_4."
            )
        return cls(first, second)

    def apply(self, y: Tensor) -> Tensor:
        r""":math:`R y` along the last axis of ``y``, whose width is ``groups x group_size``."""
        half = self.first.shape[-1]
        grouped = y.reshape(*y.shape[:-1], -1, 2, half)
        turned = lucid.stack([-grouped[..., 1, :], grouped[..., 0, :]], dim=-2)
        return (grouped * self.first + turned * self.second).reshape(*y.shape)

    def inverse(self) -> _Rotation:
        r""":math:`R^{-1}`, as coefficients of the same form: each pair's block inverted by its determinant.

        Raises
        ------
        ValueError
            A block is singular (a scale of zero, or the two diagonal terms cancelling the
            off-diagonal ones): there is no way back.
        """
        f_i, f_j = self.first[:, 0, :], self.first[:, 1, :]
        s_i, s_j = self.second[:, 0, :], self.second[:, 1, :]
        determinant = f_i * f_j + s_i * s_j
        if bool((determinant == 0).any().item()):
            raise ValueError(
                "the rotation has a singular block (a scale of zero) and cannot be undone"
            )
        scale = determinant.unsqueeze(1)
        return _Rotation(lucid.stack([f_j, f_i], dim=1) / scale, -self.second / scale)

    def matrix(self) -> Tensor:
        r"""The dense :math:`R`, ``(out, out)``: the diagonal :math:`f` and each pair's two :math:`\mp s`."""
        out = int(self.first.numel())
        half = self.first.shape[-1]
        index = lucid.arange(out, device=self.first.device)
        leading = (index % (2 * half)) < half
        partner = lucid.where(leading, index + half, index - half)
        sign = lucid.where(leading, -1.0, 1.0).to(self.first.dtype)
        paired = (index.unsqueeze(0) == partner.unsqueeze(1)).to(self.first.dtype)
        eye = lucid.eye(out, dtype=self.first.dtype, device=self.first.device)
        return (
            eye * self.first.reshape(1, out)
            + (sign * self.second.reshape(out)).unsqueeze(1) * paired
        )
