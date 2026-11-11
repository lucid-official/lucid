r"""The WaveFT layers: a frozen linear module beside a few trained wavelet coefficients of its weight update.

WaveFT (Bilican et al., 2025, arXiv:2505.12532) writes the update of a :math:`d_{out} \times d_{in}`
weight in an orthogonal wavelet basis and trains only ``n_frequency`` of the coefficients, at random
positions fixed by ``random_loc_seed`` (the same random permutation FourierFT's locations come from).
The coefficient matrix is padded so that one level of the inverse 2-D transform of the chosen wavelet
family returns exactly the weight's shape, split into its four quarter-size bands
:math:`cA, cH, cV, cD`, inverse-transformed and multiplied by ``scaling``:

.. math:: \Delta W = s\,\mathrm{IDWT}_2\bigl(cA, (cH, cV, cD)\bigr), \qquad y = (W + \Delta W)\,x + b .

With ``use_idwt=False`` the coefficients are the entries of :math:`\Delta W` itself.  Zero
coefficients start the update at zero.

``WaveFTLayer`` is the family root.  An adapter holds ``waveft_spectrum``, an ``(n_frequency,)``
parameter, the reference's key and shape, and the read-only mappings ``waveft_n_frequency``,
``waveft_scaling``, ``waveft_random_loc_seed``, ``waveft_wavelet_family`` and ``waveft_use_idwt``.
The positions are not stored: they are a function of the seed and the weight's shape.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._sparse import _matrix_from_entries
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.fourierft._locations import _entry_indices
from lucid.peft.tuners.waveft._config import WaveFTConfig
from lucid.peft.tuners.waveft._filters import _filter_bank
from lucid.peft.tuners.waveft._idwt import _idwt2

#: The standard deviation of the ``init_weights=False`` start, the reference's.
_RANDOM_STD = 0.01


class _WaveFTSpec(NamedTuple):
    """What a layer remembers of one adapter besides its coefficients.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the family, the scale
    and the positions' seed shape the graph.
    """

    waveft_n_frequency: int
    waveft_scaling: float
    waveft_random_loc_seed: int
    waveft_wavelet_family: str
    waveft_use_idwt: bool


def _padded(size: int, reduction: int) -> int:
    """The side a coefficient matrix is padded to so the inverse transform returns ``size``: ``size + reduction``, made even."""
    side = size + reduction
    return side + side % 2


class WaveFTLayer(_ConfigLayer[WaveFTConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the coefficients-to-weight map shared by the WaveFT layers.

    Per adapter: ``waveft_spectrum`` and the read-only mappings ``waveft_n_frequency``,
    ``waveft_scaling``, ``waveft_random_loc_seed``, ``waveft_wavelet_family`` and
    ``waveft_use_idwt``.  The forward pass adds each active adapter's :math:`\Delta W\,x` to the base
    output; merging adds :math:`\Delta W` to the weight.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight update is parametrised; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : WaveFTConfig
        The adapter's config (its ``n_frequency`` is this layer's).

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    ValueError
        ``config.n_frequency`` exceeds the number of weight entries, ``config.fan_in_fan_out`` is set
        (the layer's weight is stored ``(out, in)``), ``use_idwt`` is on for a wavelet this package
        has no filters for, or ``base_layer`` has lazy parameters.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("waveft_spectrum",)
    _repr_prefix = "waveft."
    waveft_spectrum: nn.ParameterDict

    waveft_n_frequency = _SpecField[int]("Coefficients of each adapter, by name (read-only).")
    waveft_scaling = _SpecField[float]("Scale of each adapter's update, by name (read-only).")
    waveft_random_loc_seed = _SpecField[int](
        "Seed that places each adapter's coefficients, by name (read-only)."
    )
    waveft_wavelet_family = _SpecField[str]("Wavelet of each adapter, by name (read-only).")
    waveft_use_idwt = _SpecField[bool](
        "Whether each adapter inverse-transforms its coefficients, by name (read-only)."
    )

    @override
    def _prepare(self, config: WaveFTConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)

    @override
    def _enter(self, adapter_name: str, config: WaveFTConfig) -> None:
        entries = self.in_features * self.out_features
        if config.n_frequency > entries:
            raise ValueError(
                "`n_frequency` should be less than or equal to the product of the input and "
                f"output dimensions but the value passed is {config.n_frequency} and the product "
                f"is {entries}"
            )
        if config.use_idwt:
            _filter_bank(
                config.wavelet_family
            )  # refuse a wavelet with no filters now, not at the first call
        anchor = self.weight
        dtype = _adapter_dtype(anchor)
        start = lucid.zeros(config.n_frequency, device=anchor.device, dtype=dtype)
        if not config.init_weights:
            start = _RANDOM_STD * lucid.randn(config.n_frequency, device=anchor.device, dtype=dtype)
        self._specs[adapter_name] = _WaveFTSpec(
            config.n_frequency,
            config.scaling,
            config.random_loc_seed,
            config.wavelet_family,
            config.use_idwt,
        )
        self.waveft_spectrum[adapter_name] = Parameter(start)

    def _spec(self, adapter: str) -> _WaveFTSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _WaveFTSpec)
        return spec

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\Delta W`, the change ``adapter`` makes to the weight, from its coefficients.

        The coefficients are laid at their positions in a zero matrix, which is padded and split
        into the four bands of the inverse transform; with ``use_idwt=False`` the matrix is
        :math:`\Delta W` itself.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the dtype of the adapter's coefficients.
        """
        spec = self._spec(adapter)
        spectrum = _parameter_at(self.waveft_spectrum, adapter)
        out, inn = self.out_features, self.in_features
        where = _entry_indices(
            out, inn, spec.waveft_n_frequency, spec.waveft_random_loc_seed, spectrum.device
        )
        rows, columns = where[0], where[1]
        if not spec.waveft_use_idwt:
            dense = _matrix_from_entries((out, inn), rows * inn + columns, spectrum)
            return dense * spec.waveft_scaling
        # The inverse transform shortens each side by the filters' reach, so the coefficients sit in
        # a matrix that much larger, with the weight's own rows and columns in its middle.
        reach = _filter_bank(spec.waveft_wavelet_family).length - 2
        height, width = _padded(out, reach), _padded(inn, reach)
        top, left = (height - out) // 2, (width - inn) // 2
        padded = _matrix_from_entries(
            (height, width), (rows + top) * width + (columns + left), spectrum
        )
        half_h, half_w = height // 2, width // 2
        delta = _idwt2(
            padded[:half_h, :half_w],
            padded[:half_h, half_w:],
            padded[half_h:, :half_w],
            padded[half_h:, half_w:],
            spec.waveft_wavelet_family,
        )
        first_row, first_column = (int(delta.shape[0]) - out) // 2, (int(delta.shape[1]) - inn) // 2
        window = delta[first_row : first_row + out, first_column : first_column + inn]
        return window * spec.waveft_scaling

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        delta = self.get_delta_weight(adapter)
        return x.to(delta.dtype) @ delta.mT


class Linear(WaveFTLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a WaveFT update: :math:`y = (W + s\,\mathrm{IDWT}_2(\cdot))\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`WaveFTLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import WaveFTConfig
    >>> from lucid.peft.tuners.waveft import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", WaveFTConfig(n_frequency=5, target_modules=["x"]))
    >>> tuple(layer.waveft_spectrum["default"].shape)
    (5,)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the coefficients start at zero
    True
    """

    base_layer: nn.Linear
