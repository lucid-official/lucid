"""Training against the compression the export will apply.

Fitting a palette to a finished model is the cheapest thing that could
work, and measured on a trained ResNet-50 it is not enough: four-bit
palettization changes every one of five top-1 predictions. Two ways of
recovering that were tried at export time and both failed — leaving the
stem and the classifier uncompressed did nothing, and a table per
channel was *worse* at six bits and cost 1.5 MB. The loss is spread
through the network rather than concentrated at its ends, which is the
signature of a problem no export setting can fix.

What is left is to make the network tolerate it, which means showing it
the compression while it is still learning. The weight the forward pass
uses is the compressed one; the weight the optimizer updates is the full
precision one behind it. Over a fine-tune the parameters move onto the
palette instead of being rounded onto it afterwards.

The gradient reaches the real parameter through
``w + (compress(w) - w).detach()``: forward it is ``compress(w)``, and
its derivative is one, so the update lands on ``w``. That estimator is
what makes this trainable at all — the compression itself has a zero
derivative almost everywhere and would stop every gradient.

The fitters are the export's own. Training against a palette fitted any
other way would optimise the model for a constraint the package does not
have, which is worse than not training at all: it would look like it
worked.
"""

import lucid
import lucid.quantization as _quant
from lucid._tensor import Tensor
from typing import override

from lucid.nn import Module, Parameter

from lucid.coreml._build import (
    _QUANTIZE_MIN_ELEMENTS,
    _assign,
    _edge_table,
    _palette_groups,
    _palettes_for,
)
from lucid.coreml._spec import Palettize, Sparsify, WeightPrecision

__all__ = ["CompressionAware"]


def _worth_compressing(weight: object) -> bool:
    """The export's own rule, so the two agree about what is covered.

    Rank at least two and enough elements to pay for the table. A bias
    and a normalisation's parameters are rank one and are left alone by
    both — palettizing a per-channel scale kills whole channels.
    """
    if not isinstance(weight, Parameter):
        return False
    return len(weight.shape) >= 2 and int(weight.numel()) >= _QUANTIZE_MIN_ELEMENTS


class _Palette:
    """A codebook, and the assignment onto it.

    The codebook is fitted when asked and the assignment is recomputed
    every forward. That split is deliberate: Lloyd's algorithm is the
    expensive half and the centroids barely move between steps, while
    which entry a weight has landed in changes as it trains and is the
    part the model has to see.
    """

    def __init__(self, weight: Tensor, bits: int) -> None:
        self.count = 1 << bits
        self.groups = _palette_groups(int(weight.shape[0]), self.count, int(weight.numel()))
        self.centres = self._fit(weight)

    def _fit(self, weight: Tensor) -> Tensor:
        rows = weight.to(lucid.float32).reshape(self.groups, -1)
        return _palettes_for(rows, self.count)

    def refit(self, weight: Tensor) -> None:
        self.centres = self._fit(weight)

    def apply(self, weight: Tensor) -> Tensor:
        rows = weight.to(lucid.float32).reshape(self.groups, -1)
        keys = _assign(rows, _edge_table(self.centres), self.count)
        picked = lucid.gather(self.centres, 1, keys)
        return picked.reshape(*[int(d) for d in weight.shape]).to(weight.dtype)


class _Symmetric:
    """Eight bits per weight with one scale per output channel.

    The scale is held across steps for the same reason a palette is: it
    is a property of the weight's spread rather than of any one step, and
    recomputing it every forward lets a single outlier move the whole
    grid.
    """

    def __init__(self, weight: Tensor) -> None:
        self.channels = int(weight.shape[0])
        self.scale, self.zero_point = self._fit(weight)

    def _fit(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        flat = weight.reshape(self.channels, -1)
        return _quant.calculate_qparams(
            flat.min(dim=1).values,
            flat.max(dim=1).values,
            _quant.per_channel_symmetric,
            _quant.qint8,
        )

    def refit(self, weight: Tensor) -> None:
        self.scale, self.zero_point = self._fit(weight)

    def apply(self, weight: Tensor) -> Tensor:
        codes = _quant.quantize(weight, self.scale, self.zero_point, _quant.qint8, ch_axis=0)
        return _quant.dequantize(codes, self.scale, self.zero_point, ch_axis=0).to(weight.dtype)


class _Mask:
    """Which weights survive, decided once and then held.

    A mask recomputed every step is not pruning — it is a threshold the
    network can walk around, since a weight pushed to zero is free to
    come back next step and nothing ever settles. Holding it is what
    makes the survivors learn to carry the load.
    """

    def __init__(self, weight: Tensor, ratio: float) -> None:
        self.ratio = ratio
        self.keep = self._fit(weight)

    def _fit(self, weight: Tensor) -> Tensor:
        flat = weight.reshape(-1).abs()
        threshold = float(lucid.quantile(flat, self.ratio).item())
        return (weight.abs() > threshold).to(weight.dtype)

    def refit(self, weight: Tensor) -> None:
        self.keep = self._fit(weight)

    def apply(self, weight: Tensor) -> Tensor:
        return weight * self.keep


class CompressionAware(Module):
    """A model that trains against the compression it will be exported with.

    Wrap a model, fine-tune the wrapper as if it were the model, then
    :meth:`settle` and export with the same ``weights`` argument. The
    forward pass uses compressed weights throughout, so the loss the
    optimizer sees is the loss the package will have.

    Parameters
    ----------
    model : nn.Module
        Model to fine-tune. It is used in place rather than copied — its
        parameters are what training updates — so keep a checkpoint if
        the uncompressed weights are still wanted.
    weights : WeightPrecision or Palettize or Sparsify
        The same value the export will be given. Anything else trains
        the model for a constraint the package will not have.

    Attributes
    ----------
    covered : list of str
        Qualified names of the weights being compressed. Rank-one
        parameters and anything under the export's size threshold are
        absent, because the export leaves those in floating point too.

    Examples
    --------
    >>> import shutil, tempfile
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.coreml as cml
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    ...     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
    ... )
    >>> loader = [
    ...     (lucid.randn(4, 3, 16, 16), lucid.randint(0, 10, (4,))) for _ in range(3)
    ... ]
    >>> aware = cml.CompressionAware(model, weights=cml.Palettize(bits=4))
    >>> aware.covered               # the one weight big enough to compress
    ['2.weight']
    >>> optimizer = lucid.optim.SGD(aware.parameters(), lr=0.01)
    >>> for x, y in loader:
    ...     loss = F.cross_entropy(aware(x), y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> aware.refit()               # follow the weights as they move
    >>> settled = aware.settle().eval()
    >>> x, room = lucid.randn(1, 3, 16, 16), tempfile.mkdtemp()
    >>> with cml.export(settled, x, f"{room}/aware.mlpackage",
    ...                 weights=cml.Palettize(bits=4)) as package:
    ...     print(package.verify(settled, x, relative=True) < 1e-5)
    True
    >>> shutil.rmtree(room)

    Notes
    -----
    ``refit`` is the caller's to schedule. The codebook is fitted once at
    construction and the assignment onto it is recomputed every forward,
    so a model drifts away from its palette over a long fine-tune;
    refitting every few hundred steps follows it back. Refitting every
    step is not better — it moves the target the model is chasing.
    """

    def __init__(
        self,
        model: Module,
        weights: WeightPrecision | Palettize | Sparsify,
    ) -> None:
        super().__init__()
        if weights == WeightPrecision.FLOAT:
            raise ValueError(
                "lucid.coreml: WeightPrecision.FLOAT compresses nothing, so there "
                "is nothing for the model to be trained against. Pass the value "
                "the export will be given — INT8, Palettize or Sparsify"
            )
        self.model = model
        self.weights = weights
        self._fits: dict[str, _Palette | _Symmetric | _Mask] = {}
        self._holders: dict[str, Module] = {}
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, Parameter) or not _worth_compressing(weight):
                continue
            qualified = f"{name}.weight" if name else "weight"
            self._holders[qualified] = module
            self._fits[qualified] = self._fit_for(weight)
        if not self._fits:
            raise ValueError(
                "lucid.coreml: nothing in this model is large enough to compress — "
                f"a weight needs rank two and {_QUANTIZE_MIN_ELEMENTS} elements, "
                "which is the export's own threshold. Training against it would "
                "change nothing"
            )

    def _fit_for(self, weight: Tensor) -> _Palette | _Symmetric | _Mask:
        if isinstance(self.weights, Palettize):
            return _Palette(weight, self.weights.bits)
        if isinstance(self.weights, Sparsify):
            return _Mask(weight, self.weights.ratio)
        return _Symmetric(weight)

    @property
    def covered(self) -> list[str]:
        """Weights this is training against, in the model's own names."""
        return sorted(self._fits)

    def refit(self) -> None:
        """Re-fit every codebook, scale or mask to the current weights."""
        for name, fit in self._fits.items():
            fit.refit(self._real(name))

    def _real(self, name: str) -> Parameter:
        # From the registry, which a forward's shadowing value never reaches.
        return self._holders[name].get_parameter("weight")

    def _compressed(self, name: str) -> Tensor:
        real = self._real(name)
        # Straight through: the forward value is the compressed weight and
        # the derivative is one, so the update lands on the parameter
        # behind it. Compression's own derivative is zero almost
        # everywhere and would stop the gradient dead.
        return real + (self._fits[name].apply(real) - real).detach()

    @override
    def forward(self, *args: Tensor, **kwargs: object) -> Tensor | tuple[Tensor, ...]:
        """Run the model with every covered weight compressed."""
        # The compressed value shadows the parameter from the layer's
        # ``__dict__``, which attribute lookup reads before
        # ``Module.__getattr__`` reaches the registry: the layer's forward
        # sees it while the Parameter keeps its slot, so ``parameters()``,
        # the optimizer and ``state_dict()`` see the real one throughout.
        # Assigning a plain tensor to the registered name is refused, and
        # used to evict the parameter and re-register it last.
        try:
            for name, module in self._holders.items():
                object.__setattr__(module, "weight", self._compressed(name))
            output: Tensor | tuple[Tensor, ...] = self.model(*args, **kwargs)
            return output
        finally:
            for module in self._holders.values():
                module.__dict__.pop("weight", None)

    def settle(self) -> Module:
        """Write the compressed values into the model and hand it back.

        Until this is called the parameters are still full precision and
        only the forward pass sees the compression; exporting then would
        fit a palette to weights that sit near one rather than on it.

        Returns
        -------
        nn.Module
            The model that was passed in, with every covered weight
            replaced by the value the compression gives it.
        """
        state = self.model.state_dict()
        for name in self._fits:
            state[name] = self._fits[name].apply(self._real(name))
        self.model.load_state_dict(state)
        return self.model
