r"""The C3A layers: a frozen base module beside a block-circulant update stored as its kernels.

``C3ALayer`` is the family root.  An adapter holds one parameter, ``c3a_kernel``
(``(d_out / b, d_in / b, b)``, float32 whatever the base's dtype), so a state dict reads
``....c3a_kernel.default`` and an adapter file ``....c3a_kernel``, the reference's key.
The update is the block-circulant matrix of the kernels divided by ``d_in``
(``lucid.peft.tuners.c3a._circulant``); the forward pass applies it by FFT, merging
builds the dense matrix.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._support import (
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.c3a._circulant import _circulant_apply, _circulant_matrix
from lucid.peft.tuners.c3a._config import C3AConfig


class _C3ASpec(NamedTuple):
    """What a layer remembers of one adapter besides its kernels.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    block_size: int


class C3ALayer(_ConfigLayer, _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the kernel-to-update map shared by the C3A layers.

    Per adapter: ``c3a_kernel``, a ``(d_out / b, d_in / b, b)`` parameter, and the read-only
    mapping ``block_size``.  The forward pass adds each active adapter's circulant update
    applied to the input (every adapter sees the layer's input); merging adds it to the weight.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : C3AConfig
        The adapter's config (its ``block_size`` is this layer's).

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.block_size`` does not divide the layer's input and output sizes, or
        ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("c3a_kernel",)
    _repr_prefix = "c3a."
    c3a_kernel: nn.ParameterDict

    block_size = _SpecField[int]("Block size of each adapter, by name (read-only).")

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(C3AConfig, config)
        size = cfg.block_size
        for label, features in (("input", self.in_features), ("output", self.out_features)):
            if features % size != 0:
                raise ValueError(
                    f"The block size should be a factor of the {label} size. However, the "
                    f"{label} size is {features} and the block size is {size}"
                )
        # FFTs run in float32 only, so the kernels do whatever the base's dtype is.
        kernel = lucid.zeros(
            self.out_features // size,
            self.in_features // size,
            size,
            device=self.weight.device,
            dtype=lucid.float32,
        )
        self._sample(kernel, cfg.init_weights)
        self._specs[adapter_name] = _C3ASpec(size)
        self.c3a_kernel[adapter_name] = Parameter(kernel)

    def _sample(self, kernel: Tensor, how: bool | str) -> None:
        """Fill ``kernel`` as ``init_weights`` says; ``True`` leaves the zeros."""
        if how is True:
            return
        with lucid.no_grad():
            if how == "gaussian":
                init.normal_(kernel)
            elif how == "kaiming_uniform":
                bound = math.sqrt(1.0 / self.in_features)
                init.uniform_(kernel, -bound, bound)
            else:  # "xavier_uniform", and False
                bound = math.sqrt(3.0) * math.sqrt(2.0 / (self.in_features + self.out_features))
                init.uniform_(kernel, -bound, bound)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the block-circulant update :math:`\mathcal C(w) / d_{in}`, shaped like the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            In the base weight's dtype.
        """
        kernel = _parameter_at(self.c3a_kernel, adapter)
        return (_circulant_matrix(kernel) / self.in_features).to(self.weight.dtype)

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        kernel = _parameter_at(self.c3a_kernel, adapter)
        applied = _circulant_apply(x.to(kernel.dtype), kernel)
        return applied / self.in_features


class Linear(C3ALayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a block-circulant update: :math:`y = W x + b + \mathcal C(w)\,x / d_{in}`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`C3ALayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.c3a import C3AConfig, Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", C3AConfig(block_size=4, init_weights=True, target_modules=["x"]))
    >>> tuple(layer.c3a_kernel["default"].shape)
    (1, 2, 4)
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the kernels start at zero
    True
    """

    base_layer: nn.Linear
