r"""The HiRA layers: a frozen base module whose weight a low-rank matrix modulates elementwise.

HiRA (Huang et al., ICLR 2025, Hadamard High-Rank Adaptation) learns two thin factors
:math:`A \in \mathbb{R}^{r \times d_{in}}` and :math:`B \in \mathbb{R}^{d_{out} \times r}` and adds the
Hadamard product of the frozen weight with their product,

.. math:: \Delta W = W \odot (B A), \qquad W' = W \odot (1 + B A),

so the update has up to ``rank(W) * r`` rank at a LoRA's parameter count.  The update reads :math:`W`,
so merging stashes it (``_PristineMerge``) and unmerging gives it back bit for bit.

``HiraLayer`` is the family root.  An adapter of a ``Linear`` or a convolution holds ``hira_A`` and
``hira_B``; one of an ``Embedding`` holds ``hira_embedding_A`` and ``hira_embedding_B``, the reference's
keys.  The factors of a convolution are a ``(r, in, *kernel)`` kernel and a ``(out, r, 1, ...)`` one,
and the modulation is their composition, as in LoRA's convolution layers.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar, NamedTuple, override

import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.functional.conv import _ConvArgs, _convnd
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer, _Home
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._stash import _WEIGHT, _PristineMerge
from lucid.peft.tuners._support import _SpecField
from lucid.peft.tuners.hira._config import HiraConfig

type _Conv = nn.Conv1d | nn.Conv2d | nn.Conv3d


class _HiraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int


class HiraLayer(_DeltaLayer[HiraConfig], _PristineMerge, family=True):
    r"""Adapter bookkeeping and the modulated update shared by the HiRA layers.

    Per adapter: the two factor tables, a dropout of the input (not on a lookup table, which has
    no activations to drop), and the read-only mapping ``r``.  A layer type supplies the shapes of
    the factors, the matrix :math:`BA` they compose to, shaped like the weight (``_modulation``), and how a
    weight is applied to an input (``_through``); the rest is here.

    Parameters
    ----------
    base_layer : nn.Module
        The module the modulated copy of the weight is applied beside; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : HiraConfig
        ``r``, ``hira_dropout`` and ``init_weights`` (the per-module ranks are the tuner's to resolve).

    Raises
    ------
    TypeError
        ``base_layer`` is another module type.
    ValueError
        ``base_layer`` is lazy and has not run yet, or ``config.fan_in_fan_out`` is set.
    NotImplementedError
        ``base_layer`` is a grouped convolution.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("hira_A", "hira_B")
    _dropout_table = "hira_dropout"
    _repr_prefix = "hira."
    hira_dropout: nn.ModuleDict[nn.Module]

    r = _SpecField[int]("Rank of each adapter, by name (read-only).")

    @override
    def _prepare(self, config: HiraConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)

    # ── what a layer type supplies ────────────────────────────────────────

    @abstractmethod
    def _factor_shapes(self, r: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The shapes of ``A`` and ``B`` for rank ``r``."""

    @abstractmethod
    def _modulation(self, adapter: str) -> Tensor:
        """:math:`BA`, shaped like the base weight."""

    @abstractmethod
    def _through(self, x: Tensor, weight: Tensor) -> Tensor:
        """``x`` through ``weight`` the way the base layer applies its own (no bias)."""

    def _start(self, config: HiraConfig, home: _Home) -> tuple[Tensor, Tensor]:
        """The starting ``A`` and ``B``: Kaiming ``A`` and zero ``B``, ``"gaussian"`` ``A`` for that setting, random for a falsy one."""
        shape_a, shape_b = self._factor_shapes(config.r)
        a, b = home.zeros(*shape_a), home.zeros(*shape_b)
        if config.init_weights:
            if config.init_weights == "gaussian":
                init.normal_(a, std=1.0 / config.r)
            else:
                init.kaiming_uniform_(a, a=5**0.5)
        else:  # random factors, for tests
            init.normal_(a)
            init.normal_(b)
        return a, b

    def _factors(self, adapter: str) -> tuple[Parameter, Parameter]:
        first, second = type(self).adapter_layer_names
        return (
            _parameter_at(getattr(self, first), adapter),
            _parameter_at(getattr(self, second), adapter),
        )

    @override
    def _enter(self, adapter_name: str, config: HiraConfig) -> None:
        a, b = self._start(config, self._home())
        first, second = type(self).adapter_layer_names
        self._specs[adapter_name] = _HiraSpec(config.r)
        getattr(self, first)[adapter_name] = Parameter(a)
        getattr(self, second)[adapter_name] = Parameter(b)
        self._enter_dropout(adapter_name, config.hira_dropout)

    # ── the update ────────────────────────────────────────────────────────

    def _modulated(self, adapter: str, weight: Tensor) -> Tensor:
        """``weight`` Hadamard-multiplied by ``adapter``'s modulation, in the factors' dtype."""
        modulation = self._modulation(adapter)
        return weight.to(modulation.dtype) * modulation

    def _into(self, adapter: str, x: Tensor) -> Tensor:
        """The input of the modulated weight: ``x`` through the adapter's dropout, in the factors' dtype."""
        return self._dropped(adapter, x).to(self._factors(adapter)[0].dtype)

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        modulated = self._modulated(adapter, self._pristine_weight())
        return self._through(self._into(adapter, x), modulated)

    @override
    def _changes(self, adapter: str, base: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {_WEIGHT: self._modulated(adapter, base[_WEIGHT])}


class Linear(HiraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a HiRA update: :math:`y = (W + W \odot BA)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`HiraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import HiraConfig
    >>> from lucid.peft.tuners.hira import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", HiraConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.hira_A["default"].shape), tuple(layer.hira_B["default"].shape)
    ((2, 8), (4, 2))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear

    @override
    def _factor_shapes(self, r: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (r, self.base_layer.in_features), (self.base_layer.out_features, r)

    @override
    def _modulation(self, adapter: str) -> Tensor:
        a, b = self._factors(adapter)
        product: Tensor = b @ a
        return product

    @override
    def _through(self, x: Tensor, weight: Tensor) -> Tensor:
        out: Tensor = x @ weight.mT
        return out


class Embedding(HiraLayer, base_type=nn.Embedding):
    r"""An ``nn.Embedding`` with a HiRA update: the table :math:`E` is looked up as :math:`E + E \odot (BA)^\top`.

    The factors are :math:`A \in \mathbb{R}^{r \times V}` and :math:`B \in \mathbb{R}^{d \times r}`
    (``hira_embedding_A``, ``hira_embedding_B``).  They start as the reference starts them,
    :math:`A = 0` and :math:`B \sim \mathcal N(0, 1)`, so the update is zero; ``init_weights=False``
    leaves both Gaussian.  The modulated table is looked up as is: the base's ``padding_idx`` and
    ``max_norm`` apply to the base lookup only, and ``hira_dropout`` has nothing to act on.

    Parameters
    ----------
    base_layer : nn.Embedding
        The layer to adapt; the other parameters are those of :class:`HiraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import HiraConfig
    >>> from lucid.peft.tuners.hira import Embedding
    >>> base = nn.Embedding(10, 4)
    >>> layer = Embedding(base, "default", HiraConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.hira_embedding_A["default"].shape), tuple(layer.hira_embedding_B["default"].shape)
    ((2, 10), (4, 2))
    >>> ids = lucid.tensor([[1, 2, 3]])
    >>> bool((layer(ids) == base(ids)).all().item())   # A starts at zero
    True
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("hira_embedding_A", "hira_embedding_B")
    base_layer: nn.Embedding

    @override
    def _factor_shapes(self, r: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (r, self.base_layer.num_embeddings), (self.base_layer.embedding_dim, r)

    @override
    def _start(self, config: HiraConfig, home: _Home) -> tuple[Tensor, Tensor]:
        if config.init_weights is False:
            return super()._start(config, home)
        shape_a, shape_b = self._factor_shapes(config.r)
        b = home.zeros(*shape_b)
        init.normal_(b)
        return home.zeros(*shape_a), b

    @override
    def _modulation(self, adapter: str) -> Tensor:
        a, b = self._factors(adapter)
        table: Tensor = (b @ a).mT
        return table

    @override
    def _through(self, x: Tensor, weight: Tensor) -> Tensor:
        looked: Tensor = F.embedding(x, weight)
        return looked

    @override
    def _into(self, adapter: str, x: Tensor) -> Tensor:
        return x  # token ids: no dropout, no cast


class _ConvNd(HiraLayer):
    r"""A HiRA convolution: ``hira_A`` is a ``(r, in, *kernel)`` kernel, ``hira_B`` a ``(out, r, 1, ...)`` one.

    The modulation is their composition, :math:`M_{o,i,k} = \sum_s B_{o,s} A_{s,i,k}`, shaped like the base
    kernel; the convolution runs with the modulated kernel and the base's stride, padding,
    dilation and padding mode.  A grouped base is refused: the factors have no meaning across groups.
    """

    base_layer: _Conv

    @override
    def _prepare(self, config: HiraConfig) -> None:
        groups = self.base_layer.groups
        if groups > 1:
            raise NotImplementedError(
                f"HiRA does not support {type(self.base_layer).__name__} layers with groups > 1 "
                f"(got groups={groups})."
            )
        super()._prepare(config)

    @override
    def _factor_shapes(self, r: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        base = self.base_layer
        pointwise = (1,) * (base.weight.ndim - 2)
        return (r, base.in_channels, *base.weight.shape[2:]), (base.out_channels, r, *pointwise)

    @override
    def _modulation(self, adapter: str) -> Tensor:
        a, b = self._factors(adapter)
        rank = a.shape[0]
        composed = b.reshape(b.shape[0], rank) @ a.reshape(rank, -1)
        return composed.reshape(b.shape[0], *a.shape[1:])

    @override
    def _through(self, x: Tensor, weight: Tensor) -> Tensor:
        base = self.base_layer
        args = _ConvArgs(base.stride, base._padding_str or base.padding, base.dilation, base.groups)
        out: Tensor = _convnd(x, weight, None, args, weight.ndim - 2, base.padding_mode)
        return out


class Conv1d(_ConvNd, base_type=nn.Conv1d):
    r"""An ``nn.Conv1d`` with a HiRA update: the kernel is modulated by :math:`BA`.

    Parameters
    ----------
    base_layer : nn.Conv1d
        The layer to adapt; the other parameters are those of :class:`HiraLayer`.
    """

    base_layer: nn.Conv1d


class Conv2d(_ConvNd, base_type=nn.Conv2d):
    r"""An ``nn.Conv2d`` with a HiRA update: the kernel is modulated by :math:`BA`.

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; the other parameters are those of :class:`HiraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import HiraConfig
    >>> from lucid.peft.tuners.hira import Conv2d
    >>> base = nn.Conv2d(2, 6, kernel_size=3, padding=1)
    >>> layer = Conv2d(base, "default", HiraConfig(r=4, target_modules=["x"]))
    >>> tuple(layer.hira_A["default"].shape), tuple(layer.hira_B["default"].shape)
    ((4, 2, 3, 3), (6, 4, 1, 1))
    >>> x = lucid.randn(1, 2, 5, 5)
    >>> bool((layer(x) == base(x)).all().item())   # the modulation starts at zero
    True
    """

    base_layer: nn.Conv2d


class Conv3d(_ConvNd, base_type=nn.Conv3d):
    r"""An ``nn.Conv3d`` with a HiRA update: the kernel is modulated by :math:`BA`.

    Parameters
    ----------
    base_layer : nn.Conv3d
        The layer to adapt; the other parameters are those of :class:`HiraLayer`.
    """

    base_layer: nn.Conv3d
