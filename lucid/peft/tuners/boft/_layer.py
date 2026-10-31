r"""The BOFT layers: a frozen base module whose weight a butterfly of orthogonal factors turns and a vector scales.

``BOFTLayer`` is the family root.  An adapter holds ``boft_R``
(``m x D x b x b``: ``m`` butterfly factors of ``D`` blocks of size ``b``, each
block the Cayley transform of the skew part of its slice) and ``boft_s`` (the
scale of the output rows).  The layer computes with the weight

.. math:: W' = (W M^\top) \odot s, \qquad M = M_{m-1} \cdots M_0,\quad
          M_i = P_i\,\operatorname{diag}(B_{i1}, \dots, B_{iD})\,P_i^\top.

State dict keys read ``....boft_R.default`` / ``....boft_s.default`` and adapter
file keys ``....boft_R`` / ``....boft_s``, the reference's.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._cayley import _Cayley, _IdentityDropout
from lucid.peft.tuners._orthogonal import _OrthogonalLayer
from lucid.peft.tuners._support import _adapter_dtype, _config_as, _SpecField
from lucid.peft.tuners.boft._butterfly import _permutation_rows
from lucid.peft.tuners.boft._config import BOFTConfig

#: Standard deviation of the ``init_weights=False`` start, the reference's.
_RANDOM_STD = 0.1


class _BoftSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    boft_block_size: int
    boft_block_num: int
    boft_n_butterfly_factor: int
    boft_dropout: float


class BOFTLayer(_OrthogonalLayer, family=True):
    r"""Adapter bookkeeping and the butterfly rotation shared by the BOFT layers.

    Per adapter: ``boft_R`` and ``boft_s``, and the read-only mappings
    ``boft_block_size`` and ``boft_block_num``.  ``get_delta_weight`` is the
    butterfly matrix :math:`M` (the reference returns it with the scale; read the
    scale from ``boft_s``); merging writes :math:`(W M^\top) \odot s` into the base
    weight and unmerging multiplies by :math:`M` and divides by :math:`s`.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` or ``nn.Conv2d`` to rotate and scale; its parameters stay shared.
    adapter_name : str
        Name of the first butterfly.
    config : BOFTConfig
        Block size or count, butterfly factors, dropout, ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is another module type.
    ValueError
        The block sizes do not fit the input size, or ``base_layer`` is lazy and has not run yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("boft_R", "boft_s")
    _repr_prefix = "boft."
    boft_R: nn.ParameterDict
    boft_s: nn.ParameterDict

    boft_block_size = _SpecField[int]("Block size of each adapter, by name (read-only).")
    boft_block_num = _SpecField[int]("Blocks per factor of each adapter, by name (read-only).")

    def _scale_shape(self) -> tuple[int, int]:
        """The shape the reference gives ``boft_s`` for this layer type (a column or a row)."""
        return (self.out_features, 1)

    def _resolve_blocks(self, config: BOFTConfig) -> tuple[int, int]:
        """``(blocks per factor, block size)`` for this layer's input size; ``ValueError`` if they do not fit."""
        dim, depth = self.in_features, config.boft_n_butterfly_factor - 1
        size, num = config.boft_block_size, config.boft_block_num
        if size == 0:
            if dim % num != 0:
                raise ValueError(f"input size ({dim}) must be divisible by boft_block_num ({num})!")
            if depth != 0 and depth > int(math.log2(num)):
                raise ValueError(
                    f"Invalid combination of boft_n_butterfly_factor ({depth + 1}) "
                    f"and boft_block_num ({num})!"
                )
            if depth != 0 and num % 2**depth != 0:
                raise ValueError(
                    f"boft_block_num ({num}) must be a multiple of 2 raised to the power of "
                    f"boft_n_butterfly_factor ({depth + 1})!"
                )
            size = dim // num
        else:
            if dim % size != 0:
                raise ValueError(f"input size ({dim}) must be divisible by boft_block_size ({size})!")
            if depth != 0 and (dim < size * 2**depth or dim % (size * 2**depth) != 0):
                raise ValueError(
                    f"Invalid combination of input size ({dim}), boft_n_butterfly_factor "
                    f"({depth + 1}) and boft_block_size ({size})!"
                )
            num = dim // size
        if depth != 0 and num % 2 != 0:
            raise ValueError(f"boft_block_num ({num}) must be an even number!")
        if depth != 0 and size % 2 != 0:
            raise ValueError(f"boft_block_size ({size}) must be an even number!")
        return num, size

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(BOFTConfig, config)
        self._check_ungrouped()
        num, size = self._resolve_blocks(cfg)
        anchor = self.weight
        device, dtype = anchor.device, _adapter_dtype(anchor)
        shape = (cfg.boft_n_butterfly_factor, num, size, size)
        rotations = Parameter(lucid.zeros(*shape, device=device, dtype=dtype))
        scale = Parameter(lucid.ones(*self._scale_shape(), device=device, dtype=dtype))
        if not cfg.init_weights:
            with lucid.no_grad():
                rotations.copy_(_RANDOM_STD * lucid.randn(*shape, device=device, dtype=dtype))
                noise = lucid.randn(*scale.shape, device=device, dtype=dtype)
                scale.copy_(1.0 + _RANDOM_STD * noise)
        self._specs[adapter_name] = _BoftSpec(size, num, cfg.boft_n_butterfly_factor, cfg.boft_dropout)
        self.boft_R[adapter_name] = rotations
        self.boft_s[adapter_name] = scale

    # ── the butterfly ─────────────────────────────────────────────────────

    def _blocks(self, adapter: str, *, stochastic: bool) -> Tensor:
        """The ``(m, D, b, b)`` orthogonal blocks; ``stochastic`` applies block dropout."""
        spec = self._specs[adapter]
        assert isinstance(spec, _BoftSpec)
        rotations = _parameter_at(self.boft_R, adapter)
        factors, count, size = rotations.shape[0], rotations.shape[1], rotations.shape[2]
        flat = rotations.reshape(factors * count, size, size)
        skew = 0.5 * (flat - flat.mT)
        blocks = _Cayley.exact(skew).reshape(factors, count, size, size)
        if stochastic and spec.boft_dropout > 0:
            blocks = _IdentityDropout.butterfly(blocks, spec.boft_dropout)
        return blocks

    def _butterfly(self, adapter: str, *, stochastic: bool) -> Tensor:
        """:math:`M = M_{m-1} \\cdots M_0`, the ``(in, in)`` orthogonal matrix of the adapter."""
        spec = self._specs[adapter]
        assert isinstance(spec, _BoftSpec)
        blocks = self._blocks(adapter, stochastic=stochastic)
        order = _permutation_rows(
            self.in_features,
            spec.boft_block_num,
            spec.boft_block_size,
            spec.boft_n_butterfly_factor - 1,
            blocks.device,
        )
        product: Tensor | None = None
        for i in range(blocks.shape[0]):
            dense = lucid.block_diag(*blocks[i].unbind(0))
            factor = dense[order[i]][:, order[i]]  # P B P^T as an index map
            product = factor if product is None else factor @ product
        assert product is not None  # at least one factor
        return product

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`M`, the butterfly rotation ``adapter`` applies to the input.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            The ``(d, d)`` butterfly product, ``d`` being the input size.
        """
        return self._butterfly(adapter, stochastic=False)

    @override
    def _right_factor(self, adapter: str, *, stochastic: bool = False) -> Tensor:
        return self._butterfly(adapter, stochastic=stochastic).mT

    @override
    def _right_inverse(self, adapter: str) -> Tensor:
        return self._butterfly(adapter, stochastic=False)  # orthogonal: (M^T)^-1 = M

    @override
    def _row_scale(self, adapter: str) -> Tensor | None:
        return _parameter_at(self.boft_s, adapter).reshape(-1, 1)


class Linear(BOFTLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a BOFT rotation of its input and a scale of its output: :math:`y = s \odot (W M^\top x) + b`.

    ``boft_s`` is ``(out_features, 1)``.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`BOFTLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import BOFTConfig
    >>> from lucid.peft.tuners.boft import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", BOFTConfig(boft_block_size=4, target_modules=["x"]))
    >>> tuple(layer.boft_R["default"].shape), tuple(layer.boft_s["default"].shape)
    ((1, 2, 4, 4), (4, 1))
    >>> x = lucid.randn(3, 8)
    >>> bool(lucid.allclose(layer(x), base(x), atol=1e-6))   # identity rotation, unit scale
    True
    """

    base_layer: nn.Linear


class Conv2d(BOFTLayer, base_type=nn.Conv2d):
    """An ``nn.Conv2d`` with a BOFT rotation of its flattened ``in x kernel`` axis and a scale of its filters.

    ``boft_s`` is ``(1, out_channels)``, as in the reference.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`BOFTLayer`.
    """

    base_layer: nn.Conv2d

    @override
    def _scale_shape(self) -> tuple[int, int]:
        return (1, self.out_features)
