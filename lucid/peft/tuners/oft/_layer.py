r"""The OFT layers: a frozen base module whose input space a block-diagonal rotation turns.

``OFTLayer`` is the family root.  An adapter holds one parameter, ``oft_R``
(``r x b(b-1)/2`` free entries; one row when the blocks are shared), the strict
upper triangles of ``r`` skew-symmetric :math:`b \times b` matrices :math:`Q_i`.
Their Cayley transform :math:`R_i = (I - Q_i)(I + Q_i)^{-1}` is orthogonal, and
the layer computes

.. math:: y = f_W(x R) + b, \qquad R = \operatorname{diag}(R_1, \dots, R_r),

which is the same as the base layer with the weight :math:`W R^\top`.  A state
dict reads ``....oft_R.default`` and an adapter file ``....oft_R.weight``, the
reference's key.
"""

import math
import warnings
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._cayley import _block_diagonal, _Cayley, _IdentityDropout
from lucid.peft.tuners._orthogonal import _OrthogonalLayer
from lucid.peft.tuners._support import _adapter_dtype, _config_as, _SpecField
from lucid.peft.tuners.oft._config import OFTConfig

#: Standard deviation of the ``init_weights=False`` start, the reference's.
_RANDOM_STD = 0.1


class _OftSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameter.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the options below are baked into the graph.
    """

    r: int
    oft_block_size: int
    module_dropout: float
    coft: bool
    eps: float
    block_share: bool
    use_cayley_neumann: bool
    num_cayley_neumann_terms: int


def _nearest_divisor(total: int, wanted: int) -> int:
    """The divisor of ``total`` closest to ``wanted`` (the lower one on a tie): what an unfit block size or count becomes."""
    if wanted >= total:
        return total
    higher = wanted
    while higher <= total and total % higher != 0:
        higher += 1
    lower = wanted
    while lower > 1 and total % lower != 0:
        lower -= 1
    return lower if wanted - lower <= higher - wanted else higher


class OFTLayer(_OrthogonalLayer, family=True):
    r"""Adapter bookkeeping and the rotation shared by the OFT layers.

    Per adapter: ``oft_R``, a ``(r, b(b-1)/2)`` parameter (``(1, b(b-1)/2)`` with
    ``block_share``), and the read-only mappings ``r`` (number of blocks) and
    ``oft_block_size``.  ``get_delta_weight`` is the block-diagonal rotation
    :math:`R`; merging writes :math:`W R^\top` into the base weight.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` or ``nn.Conv2d`` whose input is rotated; its parameters stay shared.
    adapter_name : str
        Name of the first rotation.
    config : OFTConfig
        Block count or size, dropout, ``coft``, ``block_share``, the Cayley options, ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is another module type.
    ValueError
        The blocks do not fit the input size, or ``base_layer`` is lazy and has not run yet.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("oft_R",)
    _repr_prefix = "oft."
    oft_R: nn.ParameterDict

    r = _SpecField[int]("Number of blocks of each adapter, by name (read-only).")
    oft_block_size = _SpecField[int]("Block size of each adapter, by name (read-only).")

    @override
    def _disk_table(self, table_name: str) -> str:
        return f"{table_name}.weight"  # the reference keeps the parameter in a module named ``weight``

    def _resolve_blocks(self, config: OFTConfig) -> tuple[int, int]:
        """``(r, block size)`` for this layer's input size: the one given, moved to a divisor if it does not fit."""
        dim = self.in_features
        if config.r == 0:
            size = config.oft_block_size
            if dim % size != 0 or size > dim:
                adjusted = _nearest_divisor(dim, size)
                warnings.warn(
                    f"Invalid `oft_block_size` ({size})! Adjusted `oft_block_size` to ({adjusted}).",
                    stacklevel=4,
                )
                size = adjusted
            return dim // size, size
        count = config.r
        if dim % count != 0 or count > dim:
            adjusted = _nearest_divisor(dim, count)
            warnings.warn(f"Invalid `r` ({count})! Adjusted `r` to ({adjusted}).", stacklevel=4)
            count = adjusted
        return count, dim // count

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(OFTConfig, config)
        self._check_ungrouped()
        count, size = self._resolve_blocks(cfg)
        anchor = self.weight
        rotation = Parameter(
            lucid.zeros(
                1 if cfg.block_share else count,
                size * (size - 1) // 2,
                device=anchor.device,
                dtype=_adapter_dtype(anchor),
            )
        )
        if not cfg.init_weights:
            init.normal_(rotation, mean=0.0, std=_RANDOM_STD)
        self._specs[adapter_name] = _OftSpec(
            count,
            size,
            cfg.module_dropout,
            cfg.coft,
            cfg.eps,
            cfg.block_share,
            cfg.use_cayley_neumann,
            cfg.num_cayley_neumann_terms,
        )
        self.oft_R[adapter_name] = rotation

    # ── the rotation ──────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _OftSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _OftSpec)
        return spec

    def _constrain(self, adapter: str) -> None:
        r"""COFT: pull each block's :math:`Q` into the ball of radius ``eps / sqrt(r)``, in place.

        The parameter itself is clamped (the optimiser continues from the
        clamped value), as the reference does every time it builds the rotation.
        """
        spec = self._spec(adapter)
        rotation = _parameter_at(self.oft_R, adapter)
        with lucid.no_grad():
            skew = _Cayley.skew(rotation, spec.oft_block_size)
            radius = spec.eps / math.sqrt(skew.shape[0])
            norm = (skew * skew).sum(dim=(1, 2), keepdim=True).sqrt()
            pulled = skew * (radius / norm.clamp(min=radius))
            rotation.copy_(_Cayley.upper(pulled, spec.oft_block_size))

    def _blocks(self, adapter: str, *, stochastic: bool = False) -> Tensor:
        """The ``(r, b, b)`` orthogonal blocks (``(1, b, b)`` when shared); ``stochastic`` applies block dropout."""
        spec = self._spec(adapter)
        if spec.coft:
            self._constrain(adapter)
        skew = _Cayley.skew(_parameter_at(self.oft_R, adapter), spec.oft_block_size)
        blocks = (
            _Cayley.neumann(skew, spec.num_cayley_neumann_terms)
            if spec.use_cayley_neumann
            else _Cayley.exact(skew)
        )
        if stochastic and spec.module_dropout > 0:
            blocks = _IdentityDropout.blocks(blocks, spec.module_dropout)
        return blocks

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`R`, the block-diagonal rotation ``adapter`` applies to the input.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(d, d)`` and block-diagonal, ``d`` being the number of input units.
        """
        spec = self._spec(adapter)
        copies = self.in_features // spec.oft_block_size if spec.block_share else spec.r
        return _block_diagonal(self._blocks(adapter), copies)

    @override
    def _right_factor(self, adapter: str, *, stochastic: bool = False) -> Tensor:
        spec = self._spec(adapter)
        copies = self.in_features // spec.oft_block_size if spec.block_share else spec.r
        return _block_diagonal(self._blocks(adapter, stochastic=stochastic), copies).mT

    @override
    def _right_inverse(self, adapter: str) -> Tensor:
        # The blocks are inverted one by one: the inverse of a block-diagonal matrix is
        # block-diagonal, and a dense solve over the whole input size would cost far more.
        spec = self._spec(adapter)
        copies = self.in_features // spec.oft_block_size if spec.block_share else spec.r
        inverse = linalg.inv(self._blocks(adapter))
        return _block_diagonal(inverse, copies).mT


class Linear(OFTLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with an OFT rotation of its input: :math:`y = W(xR) + b`.

    The rotation is applied block by block to the input, which costs
    :math:`O(n\,d\,b)` for :math:`n` rows of width :math:`d` and blocks of size
    :math:`b` instead of the :math:`O(d_{out}\,d^2)` of building :math:`W R^\top`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`OFTLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import OFTConfig
    >>> from lucid.peft.tuners.oft import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", OFTConfig(oft_block_size=4, target_modules=["x"]))
    >>> layer.r["default"], tuple(layer.oft_R["default"].shape)
    (2, (2, 6))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the rotation starts at the identity
    True
    """

    base_layer: nn.Linear

    def _rotate(self, x: Tensor, adapter: str) -> Tensor:
        """``x R``: each width-``b`` slice of the last axis turned by its block."""
        spec = self._spec(adapter)
        blocks = self._blocks(adapter, stochastic=self.training)
        slices = x.to(blocks.dtype).reshape(-1, spec.r, spec.oft_block_size).transpose(0, 1)
        return (slices @ blocks).transpose(0, 1).reshape(*x.shape)

    @override
    def _adapted(self, x: Tensor, adapters: list[str]) -> Tensor:
        turned = x
        for adapter in adapters:
            turned = self._rotate(turned, adapter)
        out: Tensor = self.base_layer(turned.to(x.dtype))
        return out


class Conv2d(OFTLayer, base_type=nn.Conv2d):
    """An ``nn.Conv2d`` with an OFT rotation of its flattened ``in x kernel`` axis: the convolution runs with the rotated filters.

    The reference rotates every unfolded input patch and sums the overlapping patches back
    before convolving, which equals this only for 1x1 kernels.  Here the rotation is applied to
    the filters, so the adapted layer and its merged form are the same computation.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`OFTLayer`.
    """

    base_layer: nn.Conv2d
