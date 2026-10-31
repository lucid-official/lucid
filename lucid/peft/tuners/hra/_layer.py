r"""The HRA layers: a frozen base module whose weight a chain of Householder reflections multiplies.

``HRALayer`` is the family root.  An adapter holds one parameter, ``hra_u``
(``d x r``, the columns :math:`u_i`; ``d`` is the input size, or ``in x kernel``
for a convolution).  The layer computes with the weight :math:`W D`,
:math:`D = H_1 \cdots H_r`, :math:`H_i = I - 2\hat u_i\hat u_i^\top`.  A state dict
reads ``....hra_u.default`` and an adapter file ``....hra_u``, the reference's key.
"""

import math
import warnings
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._orthogonal import _OrthogonalLayer
from lucid.peft.tuners._support import _adapter_dtype, _config_as, _SpecField
from lucid.peft.tuners.hra._config import HRAConfig
from lucid.peft.tuners.hra._householder import _Householder


class _HraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameter.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    hra_r: int
    hra_apply_GS: bool


class HRALayer(_OrthogonalLayer, family=True):
    r"""Adapter bookkeeping and the reflection chain shared by the HRA layers.

    Per adapter: ``hra_u``, a ``(d, r)`` parameter, and the read-only mappings
    ``hra_r`` (number of reflections) and ``hra_apply_GS``.  ``get_delta_weight``
    is the orthogonal matrix :math:`D`; merging writes :math:`W D` into the base
    weight and unmerging multiplies by the chain in reverse.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` or ``nn.Conv2d`` whose weight the chain multiplies; its parameters stay shared.
    adapter_name : str
        Name of the first chain.
    config : HRAConfig
        ``r``, ``apply_GS`` and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is another module type.
    ValueError
        ``base_layer`` is lazy and has not run yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("hra_u",)
    _repr_prefix = "hra."
    hra_u: nn.ParameterDict

    hra_r = _SpecField[int]("Number of reflections of each adapter, by name (read-only).")
    hra_apply_GS = _SpecField[bool](
        "Whether each adapter orthonormalises its vectors first, by name (read-only)."
    )

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(HRAConfig, config)
        self._check_ungrouped()
        anchor = self.weight
        vectors = Parameter(self._initial(cfg, anchor))
        self._specs[adapter_name] = _HraSpec(cfg.r, cfg.apply_GS)
        self.hra_u[adapter_name] = vectors

    def _initial(self, config: HRAConfig, anchor: Tensor) -> Tensor:
        """The starting ``(d, r)`` vectors: Kaiming-uniform, repeated in pairs for the identity start."""
        rows, device, dtype = self.in_features, anchor.device, _adapter_dtype(anchor)
        if config.init_weights and config.r % 2 == 0:
            half = lucid.zeros(rows, config.r // 2, device=device, dtype=dtype)
            init.kaiming_uniform_(half, a=math.sqrt(5))
            return lucid.repeat_interleave(half, 2, dim=1)
        if config.init_weights:
            warnings.warn(
                "The symmetric initialization can NOT be performed when r is odd!", stacklevel=5
            )
        vectors = lucid.zeros(rows, config.r, device=device, dtype=dtype)
        init.kaiming_uniform_(vectors, a=math.sqrt(5))
        return vectors

    @override
    def get_delta_weight(self, adapter: str, reverse: bool = False) -> Tensor:
        r"""Return :math:`D = H_1 \cdots H_r`, the orthogonal matrix ``adapter`` multiplies the weight by.

        Parameters
        ----------
        adapter : str
            The adapter's name.
        reverse : bool, default=False
            Multiply the reflections in reverse order: :math:`D^{-1}`.

        Returns
        -------
        Tensor
            ``(d, d)``; ``d`` is ``in_features``, or ``in_channels x kernel`` for a convolution.
        """
        spec = self._specs[adapter]
        assert isinstance(spec, _HraSpec)
        return _Householder.product(
            _parameter_at(self.hra_u, adapter), gram_schmidt=spec.hra_apply_GS, reverse=reverse
        )

    @override
    def _right_factor(self, adapter: str, *, stochastic: bool = False) -> Tensor:
        del stochastic  # HRA has no dropout
        return self.get_delta_weight(adapter)

    @override
    def _right_inverse(self, adapter: str) -> Tensor:
        return self.get_delta_weight(adapter, reverse=True)


class Linear(HRALayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with HRA reflections of its input: :math:`y = (W D)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`HRALayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import HRAConfig
    >>> from lucid.peft.tuners.hra import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", HRAConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.hra_u["default"].shape)
    (8, 2)
    >>> x = lucid.randn(3, 8)
    >>> bool(lucid.allclose(layer(x), base(x), atol=1e-6))   # equal pairs of vectors cancel
    True
    """

    base_layer: nn.Linear


class Conv2d(HRALayer, base_type=nn.Conv2d):
    """An ``nn.Conv2d`` with HRA reflections of its flattened ``in x kernel`` axis: the convolution runs with the transformed filters.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`HRALayer`.
    """

    base_layer: nn.Conv2d
