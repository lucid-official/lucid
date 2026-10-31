"""LoRA on convolutions: ``nn.Conv{1,2,3}d`` and ``nn.ConvTranspose{1,2,3}d``.

The adapter is a narrow convolution followed by a 1x1 one, in both families; they differ in the first
factor (a convolution of the base layer's geometry, or a transposed one) and in how the pair
composes into one kernel.
"""

import warnings
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, Final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners.lora._layer import _Factored, _Factors, _Place

type _Conv = nn.Conv1d | nn.Conv2d | nn.Conv3d
type _ConvTranspose = nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d
type _Convolution = _Conv | _ConvTranspose

#: The regular convolution of each rank (by the kernel's number of axes): a second factor's class.
_POINTWISE: Final = {3: nn.Conv1d, 4: nn.Conv2d, 5: nn.Conv3d}


class _ConvLayer(_Factored[_Convolution]):
    r"""What every LoRA convolution shares: the 1x1 second factor, groups and the channel axis.

    ``lora_A`` takes :math:`C_i` channels to :math:`r` with the base layer's
    geometry (``_first_factor``), and ``lora_B`` is a
    :math:`1 \times \dots \times 1` convolution from :math:`r` to
    :math:`C_o` with the base layer's ``groups``.  Their composition is
    one convolution with the kernel :math:`B \circ A`, so with
    ``groups == 1`` the update merges into :math:`W` exactly; a grouped
    base works in the forward pass but cannot be merged (a rank-:math:`r`
    update mixes channels across groups, nor can DoRA, which needs that
    kernel).  ``lora_bias`` gives ``lora_B`` a bias, merged into the
    base bias.
    """

    #: The class of the first factor: the module type this layer wraps (its ``base_type=``).
    _conv_type: ClassVar[Callable[..., nn.Module]]
    base_layer: _Convolution

    @override
    def __init_subclass__(
        cls, *, base_type: type[nn.Module] | None = None, family: bool = False, **kwargs: object
    ) -> None:
        super().__init_subclass__(base_type=base_type, family=family, **kwargs)
        if base_type is not None:
            cls._conv_type = base_type

    @override
    def _setup(self) -> None:
        base = self.base_layer
        self.in_features = base.in_channels
        self.out_features = base.out_channels
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        if base.groups > 1:
            warnings.warn(
                "LoRA adapter added to a convolution with groups > 1. Merging is not supported.",
                stacklevel=4,
            )

    @override
    def _anchor(self) -> Tensor:
        return self.base_layer.weight

    @override
    def _check_rank_fits(self, r: int) -> None:
        groups = self.base_layer.groups
        if r % groups:
            raise ValueError(
                f"Targeting a {type(self.base_layer).__name__} with groups={groups} and rank {r}. "
                "Currently, support is limited to conv layers where the rank is divisible by "
                "groups. Either choose a different rank or do not target this specific layer."
            )

    @abstractmethod
    def _geometry(self) -> dict[str, object]:
        """The base layer's geometry as constructor arguments, which ``lora_A`` takes over."""

    def _first_factor(self, r: int, place: _Place) -> nn.Module:
        """``lora_A``: the base layer's geometry from its input channels to ``r``, without a bias."""
        return type(self)._conv_type(
            self.in_features,
            r,
            bias=False,
            device=place.device,
            dtype=place.dtype,
            **self._geometry(),
        )

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        base = self.base_layer
        return (
            self._first_factor(r, place),
            _POINTWISE[base.weight.ndim](
                r,
                self.out_features,
                kernel_size=1,
                groups=base.groups,
                bias=place.bias,
                device=place.device,
                dtype=place.dtype,
            ),
        )

    @override
    def _output_vector(self, vector: Tensor) -> Tensor:
        # Channels are axis 1 of a convolution's output.
        return vector.reshape(1, -1, *([1] * (self.base_layer.weight.ndim - 2)))

    @override
    def _check_weight_norm(self) -> None:
        if self.base_layer.groups > 1:
            raise NotImplementedError(
                "DoRA is not supported for a LoRA convolution with groups > 1: the update "
                "mixes channels across groups, which one grouped kernel cannot hold"
            )

    def _refuse_grouped_merge(self) -> None:
        if self.base_layer.groups > 1:
            raise NotImplementedError(
                "Merging is not supported for a LoRA convolution with groups > 1: the update "
                "mixes channels across groups, which one grouped kernel cannot hold"
            )

    @override
    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        pending = super()._pending_merges(adapter_names)
        if pending:
            self._refuse_grouped_merge()
        return pending


class _ConvNd(_ConvLayer):
    r"""A LoRA convolution over ``nn.Conv{1,2,3}d``: ``lora_A`` keeps the base's kernel, stride, padding and dilation."""

    base_layer: _Conv

    @override
    def _geometry(self) -> dict[str, object]:
        base = self.base_layer
        return {
            "kernel_size": base.kernel_size,
            "stride": base.stride,
            # ``"same"`` / ``"valid"`` live beside the numeric padding, which is 0 then.
            "padding": base._padding_str or base.padding,
            "dilation": base.dilation,
            "padding_mode": base.padding_mode,
        }

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,B \circ A`, the kernel of the two convolutions in a row.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base kernel, ``(out_channels, in_channels, *kernel_size)``,
            in the adapter's dtype.

        Raises
        ------
        NotImplementedError
            The base layer has ``groups > 1``.
        """
        self._refuse_grouped_merge()
        a = self.lora_A[adapter].weight  # (r, in, *k)
        b = self.lora_B[adapter].weight  # (out, r, 1, ...)
        out_channels, rank = b.shape[0], b.shape[1]
        composed = b.reshape(out_channels, rank) @ a.reshape(rank, -1)
        return composed.reshape(out_channels, *a.shape[1:]) * self._specs[adapter].scaling


class _ConvTransposeNd(_ConvLayer):
    r"""A LoRA convolution over ``nn.ConvTranspose{1,2,3}d``.

    The base kernel is :math:`W \in \mathbb{R}^{C_i \times C_o \times k}` (input
    channels first).  ``lora_A`` is a transposed convolution
    :math:`C_i \to r` of the base's kernel size, stride, padding, output
    padding and dilation, with kernel :math:`A \in \mathbb{R}^{C_i \times r \times k}`;
    ``lora_B`` is the 1x1 (regular) convolution :math:`r \to C_o`.  A pointwise
    map after a transposed convolution is the transposed convolution of the
    composed kernel, :math:`\Delta W_{i,o,k} = s \sum_{r} A_{i,r,k}\,B_{o,r}`, so
    ``groups == 1`` merges exactly, as for the forward convolutions.  An
    ``output_size`` given to the forward reaches the base layer and ``lora_A``
    alike, so the update has the output the base resolves.
    """

    base_layer: _ConvTranspose

    @override
    def _geometry(self) -> dict[str, object]:
        base = self.base_layer
        return {
            "kernel_size": base.kernel_size,
            "stride": base.stride,
            "padding": base.padding,
            "output_padding": base.output_padding,
            "dilation": base.dilation,
        }

    @override
    def _rows(self, weight: Tensor) -> Tensor:
        # The kernel is (in, out, *k): swap to one row per output channel; the swap is its own inverse.
        return weight.transpose(0, 1)

    @override
    def forward(self, x: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """Run the base layer and add the adapters' update, both at the output size asked for.

        ``output_size`` picks one of the sizes the stride allows, as for ``nn.ConvTranspose*d``.
        The update's first factor has the base's geometry, so it resolves the same size.
        """
        return self._forward(x, {"output_size": output_size})

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,\sum_r A_{i,r,k}B_{o,r}`, the kernel of the two layers in a row.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base kernel, ``(in_channels, out_channels, *kernel_size)``,
            in the adapter's dtype.

        Raises
        ------
        NotImplementedError
            The base layer has ``groups > 1``.
        """
        self._refuse_grouped_merge()
        a = self.lora_A[adapter].weight  # (in, r, *k)
        b = self.lora_B[adapter].weight  # (out, r, 1, ...)
        in_channels, rank = a.shape[0], a.shape[1]
        out_channels = b.shape[0]
        # (out, r) @ (in, r, K) broadcasts over the input channels: (in, out, K).
        composed = b.reshape(out_channels, rank) @ a.reshape(in_channels, rank, -1)
        return (
            composed.reshape(in_channels, out_channels, *a.shape[2:]) * self._specs[adapter].scaling
        )


class Conv1d(_ConvNd, base_type=nn.Conv1d):
    r"""An ``nn.Conv1d`` with a LoRA update: ``lora_A`` (kernel :math:`k`, :math:`r` channels), then ``lora_B`` (1x1).

    Parameters
    ----------
    base_layer : nn.Conv1d
        The layer to adapt; the other parameters are those of :class:`Conv2d`.
    """

    base_layer: nn.Conv1d


class Conv2d(_ConvNd, base_type=nn.Conv2d):
    r"""An ``nn.Conv2d`` with a LoRA update: ``lora_A`` (kernel :math:`k_h \times k_w`, :math:`r` channels), then ``lora_B`` (1x1).

    Parameters
    ----------
    base_layer : nn.Conv2d
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.
    r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
        The adapter's hyper-parameters, as ``LoraLayer.update_layer`` takes
        them; ``r`` must be positive and divisible by the base's ``groups``.
        ``init_lora_weights`` of ``"pissa"`` or ``"olora"`` is refused (the
        kernel is not a dense matrix).

    Notes
    -----
    A grouped or depthwise base (``groups > 1``) adapts in the forward pass
    (with a warning), but its adapter cannot be merged: a rank-:math:`r` update
    mixes channels across groups, which one grouped kernel cannot hold, so
    ``merge``, ``get_delta_weight`` and DoRA raise ``NotImplementedError``.  The
    reference's convolution LoRA has the same restriction, and the default
    targets of the model zoo leave such layers out.  The same holds for
    :class:`Conv1d`, :class:`Conv3d` and the transposed layers.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import Conv2d
    >>> base = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    >>> layer = Conv2d(base, "default", r=2, lora_alpha=4)
    >>> tuple(layer.lora_A["default"].weight.shape), tuple(layer.lora_B["default"].weight.shape)
    ((2, 3, 3, 3), (5, 2, 1, 1))
    >>> x = lucid.randn(2, 3, 6, 6)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Conv2d


class Conv3d(_ConvNd, base_type=nn.Conv3d):
    r"""An ``nn.Conv3d`` with a LoRA update: ``lora_A`` (a 3-D kernel, :math:`r` channels), then ``lora_B`` (1x1x1).

    Parameters
    ----------
    base_layer : nn.Conv3d
        The layer to adapt; the other parameters are those of :class:`Conv2d`.
    """

    base_layer: nn.Conv3d


class ConvTranspose1d(_ConvTransposeNd, base_type=nn.ConvTranspose1d):
    r"""An ``nn.ConvTranspose1d`` with a LoRA update: ``lora_A`` (a transposed convolution to :math:`r` channels), then ``lora_B`` (1x1).

    Parameters
    ----------
    base_layer : nn.ConvTranspose1d
        The layer to adapt; the other parameters are those of :class:`ConvTranspose2d`.
    """

    base_layer: nn.ConvTranspose1d


class ConvTranspose2d(_ConvTransposeNd, base_type=nn.ConvTranspose2d):
    r"""An ``nn.ConvTranspose2d`` with a LoRA update: ``lora_A`` (a transposed convolution to :math:`r` channels), then ``lora_B`` (1x1).

    Parameters
    ----------
    base_layer : nn.ConvTranspose2d
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.

    The remaining parameters are those of :class:`Conv2d`.  The initialisations
    ``"pissa"`` and ``"olora"`` are refused (the kernel is not a dense matrix).

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import ConvTranspose2d
    >>> base = nn.ConvTranspose2d(3, 5, kernel_size=4, stride=2, padding=1)
    >>> layer = ConvTranspose2d(base, "default", r=2, lora_alpha=4)
    >>> tuple(layer.lora_A["default"].weight.shape), tuple(layer.lora_B["default"].weight.shape)
    ((3, 2, 4, 4), (5, 2, 1, 1))
    >>> x = lucid.randn(2, 3, 6, 6)
    >>> layer(x).shape == base(x).shape
    True
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.ConvTranspose2d


class ConvTranspose3d(_ConvTransposeNd, base_type=nn.ConvTranspose3d):
    r"""An ``nn.ConvTranspose3d`` with a LoRA update: ``lora_A`` (a 3-D transposed convolution to :math:`r` channels), then ``lora_B`` (1x1x1).

    Parameters
    ----------
    base_layer : nn.ConvTranspose3d
        The layer to adapt; the other parameters are those of :class:`ConvTranspose2d`.
    """

    base_layer: nn.ConvTranspose3d
