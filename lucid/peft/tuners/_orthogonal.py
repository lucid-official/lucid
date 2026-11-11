r"""What the orthogonal-transform family (OFT, BOFT, HRA) shares: a rotation of the input space beside a frozen weight.

An adapter of this family is a square matrix :math:`D` over the base layer's
input units (for a convolution, over the flattened ``in x kernel`` axis), applied
to the base weight from the right,

.. math:: W' = (W D) \odot s, \qquad y = f_{W'}(x) + b,

where :math:`s` is an optional per-output-row scale (BOFT's ``boft_s``).  Merging
writes :math:`W'` into the base weight; unmerging multiplies back by
:math:`D^{-1}` and divides by :math:`s`.  The forward pass of an unmerged layer
applies the active adapters in turn to the weight, so it equals the merged
layer's for any number of adapters, and a method with a cheaper way to apply its
matrix (OFT rotates the input block by block) overrides ``_adapted``.

**The order of several adapters** is the order they are listed in (the active
adapters, or ``adapter_names`` of a merge): the weight is rewritten by the first,
then the second, :math:`W D_1 D_2 \cdots D_n`.  Rotations do not commute, so a
layer that turns the input instead of the weight must turn it by the *last*
adapter first, :math:`x D_n^\top \cdots D_1^\top`, to be the same function
(``_input_order``).  The weight route and the merge walk the list as given and
never decide this; ``_input_order`` is where it is decided.

A method supplies the matrix: ``_right_factor`` (and ``get_delta_weight``, the
matrix the reference calls by that name) and, for a scale, ``_row_scale``.
"""

from abc import abstractmethod
from typing import override

import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._kinds import _Geometry, _Kind
from lucid.peft.tuners._support import _ConfigLayer, _WeightRewrite


class _OrthogonalLayer(_ConfigLayer, _WeightRewrite):
    r"""A frozen base module whose weight an orthogonal matrix of its input space multiplies from the right.

    Per adapter a layer keeps the method's parameters in its tables
    (``adapter_layer_names``) and its scalar state in ``_specs``.  Only a
    ``nn.Linear`` or an ungrouped ``nn.Conv1d`` / ``nn.Conv2d`` is adapted:
    a grouped kernel has no single input space to rotate (``NotImplementedError``).
    """

    @property
    def _geometry(self) -> _Geometry:
        root = self._root()
        return _Kind.of(type(root)).geometry(root)

    @property
    def in_features(self) -> int:
        """Input units the matrix acts on: ``in_features``, or ``in_channels x kernel`` for a convolution."""
        return self._geometry.flat_in

    @property
    def out_features(self) -> int:
        """Output units of the base layer."""
        return self._geometry.out

    def _check_ungrouped(self) -> None:
        root = self._root()
        groups = getattr(root, "groups", 1)
        if groups > 1:
            raise NotImplementedError(
                f"{type(self).__name__} does not support {type(root).__name__} layers "
                f"with groups > 1 (got groups={groups})."
            )

    # ── what a method supplies ────────────────────────────────────────────

    @abstractmethod
    def _right_factor(self, adapter: str, *, stochastic: bool = False) -> Tensor:
        """The matrix :math:`D` with :math:`W' = W D` (before the row scale); ``stochastic`` may apply training-time dropout."""

    def _row_scale(self, adapter: str) -> Tensor | None:
        """The ``(out, 1)`` scale :math:`s` of the weight's rows, or ``None`` (no scale)."""
        del adapter  # most methods only rotate
        return None

    def _right_inverse(self, adapter: str) -> Tensor:
        """:math:`D^{-1}`; a method that knows its inverse in closed form overrides this."""
        return linalg.inv(self._right_factor(adapter))

    # ── the weight an adapter makes ───────────────────────────────────────

    def _flat(self, weight: Tensor) -> Tensor:
        return weight.reshape(weight.shape[0], -1)

    def _rotated(self, weight: Tensor, adapter: str, *, stochastic: bool = False) -> Tensor:
        """``weight`` with ``adapter`` applied: :math:`(W D) \\odot s`, in ``weight``'s shape."""
        factor = self._right_factor(adapter, stochastic=stochastic)
        flat = self._flat(weight).to(factor.dtype) @ factor
        scale = self._row_scale(adapter)
        if scale is not None:
            flat = flat * scale
        return flat.reshape(*weight.shape)

    def _restored(self, weight: Tensor, adapter: str) -> Tensor:
        """``weight`` with ``adapter`` taken back out: :math:`(W / s) D^{-1}`."""
        scale = self._row_scale(adapter)
        inverse = self._right_inverse(adapter)
        flat = self._flat(weight).to(inverse.dtype)
        if scale is not None:
            flat = flat / scale
        return (flat @ inverse).reshape(*weight.shape)

    @override
    def _rewritten(self, weight: Tensor, adapter: str) -> Tensor:
        return self._rotated(weight, adapter)

    @override
    def _reverted(self, weight: Tensor, adapter: str) -> Tensor:
        return self._restored(weight, adapter)

    # ── the forward pass ──────────────────────────────────────────────────

    @staticmethod
    def _input_order(adapters: list[str]) -> list[str]:
        """``adapters`` in the order a layer that turns the *input* applies them: the last listed first.

        :math:`W D_1 D_2` is the weight the adapters ``[1, 2]`` make, and
        :math:`x (W D_1 D_2)^\\top = (x D_2^\\top D_1^\\top) W^\\top` turns ``x`` by adapter 2
        before adapter 1.
        """
        return adapters[::-1]

    def _adapted(self, x: Tensor, adapters: list[str]) -> Tensor:
        """The layer's output with ``adapters`` applied, computed from the weight."""
        weight: Tensor = self.weight
        for adapter in adapters:
            weight = self._rotated(weight, adapter, stochastic=self.training)
        base = self.base_layer
        kind = _Kind.of(type(base))
        out = kind.apply(base, x.to(weight.dtype), weight)
        bias = self.bias
        if bias is not None:
            kernel = self._geometry.kernel
            out = out + (bias.reshape(1, -1, *([1] * len(kernel))) if kernel else bias)
        return out

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run the layer on the weight the active adapters turn.

        The base weight is rotated by each active adapter in turn (and row-scaled, for BOFT)
        and the layer is computed from that; the base layer alone when adapters are disabled,
        merged or none applies.
        """
        adapters = [name for name in self.active_adapters if name in self._specs]
        if self.disable_adapters or self.merged or not adapters:
            result: Tensor = self.base_layer(x)
            return result
        out = self._adapted(x, adapters)
        return out if out.dtype == x.dtype else out.to(x.dtype)
