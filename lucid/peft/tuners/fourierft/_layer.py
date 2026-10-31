r"""The FourierFT layers: a frozen base module beside an update that lives in the frequency domain.

``FourierFTLayer`` is the family root.  An adapter holds ``fourierft_spectrum``,
``n`` coefficients, and the layer builds its update from them each time it is
needed: the coefficients are scattered into a zero ``d_out x d_in`` spectrum at
the adapter's fixed locations and the real part of the inverse 2-D Fourier
transform, times the scale, is :math:`\Delta W`.  A state dict reads
``....fourierft_spectrum.default`` and an adapter file ``....fourierft_spectrum``,
the reference's key.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.fft as fft
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._base import _parameter_at
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.fourierft._config import FourierFTConfig
from lucid.peft.tuners.fourierft._locations import _entry_indices


class _FourierSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameter.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the scale is baked into the graph and the seed decides the locations.
    """

    fourierft_n_frequency: int
    fourierft_scaling: float
    fourierft_random_loc_seed: int


class FourierFTLayer(_ConfigLayer, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the spectrum-to-weight map shared by the FourierFT layers.

    Per adapter: ``fourierft_spectrum``, an ``(n,)`` parameter, and the read-only
    mappings ``fourierft_n_frequency``, ``fourierft_scaling`` and
    ``fourierft_random_loc_seed``.  The forward pass adds each active adapter's
    :math:`\Delta W\,x` to the base output; merging adds :math:`\Delta W` to the weight.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : FourierFTConfig
        The adapter's config (its ``n_frequency`` is this layer's).

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.n_frequency`` exceeds the number of weight entries, or ``base_layer``
        builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("fourierft_spectrum",)
    _repr_prefix = "fourierft."
    fourierft_spectrum: nn.ParameterDict

    fourierft_n_frequency = _SpecField[int]("Coefficients of each adapter, by name (read-only).")
    fourierft_scaling = _SpecField[float]("Scale of each adapter's update, by name (read-only).")
    fourierft_random_loc_seed = _SpecField[int](
        "Seed that places each adapter's coefficients, by name (read-only)."
    )

    @property
    def in_features(self) -> int:
        """Columns of the base weight: the layer's input features."""
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        """Rows of the base weight: the layer's output features."""
        return int(self.weight.shape[0])

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(FourierFTConfig, config)
        entries = self.in_features * self.out_features
        if cfg.n_frequency > entries:
            raise ValueError(
                "`n_frequency` should be less than or equal to the product of the input and "
                f"output dimensions but the value passed is {cfg.n_frequency} and the product is "
                f"{entries}"
            )
        anchor = self.weight
        spectrum = Parameter(
            lucid.randn(cfg.n_frequency, device=anchor.device, dtype=_adapter_dtype(anchor))
        )
        if cfg.init_weights:
            init.zeros_(spectrum)
        self._specs[adapter_name] = _FourierSpec(
            cfg.n_frequency, cfg.scaling, cfg.random_loc_seed
        )
        self.fourierft_spectrum[adapter_name] = spectrum

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\Delta W = \alpha\,\operatorname{Re}\,\mathcal F^{-1}(S)`, the change ``adapter`` makes to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the dtype of the adapter's coefficients.
        """
        spec = self._specs[adapter]
        assert isinstance(spec, _FourierSpec)
        spectrum = _parameter_at(self.fourierft_spectrum, adapter)
        where = _entry_indices(
            self.out_features,
            self.in_features,
            spec.fourierft_n_frequency,
            spec.fourierft_random_loc_seed,
            spectrum.device,
        )
        dense = lucid.zeros(self.out_features, self.in_features, device=spectrum.device)
        dense = dense.index_put((where[0], where[1]), spectrum.to(lucid.float32))
        delta = fft.ifft2(dense).real * spec.fourierft_scaling
        return delta.to(spectrum.dtype)

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        delta = self.get_delta_weight(adapter)
        return x.to(delta.dtype) @ delta.mT


class Linear(FourierFTLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a FourierFT update: :math:`y = W x + b + \Delta W\,x`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`FourierFTLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import FourierFTConfig
    >>> from lucid.peft.tuners.fourierft import Linear
    >>> base = nn.Linear(8, 4)
    >>> config = FourierFTConfig(n_frequency=5, init_weights=True, target_modules=["x"])
    >>> layer = Linear(base, "default", config)
    >>> tuple(layer.fourierft_spectrum["default"].shape)
    (5,)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the coefficients start at zero
    True
    """

    base_layer: nn.Linear
