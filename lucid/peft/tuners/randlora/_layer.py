r"""The RandLoRA layers: a frozen base module beside two trainable scales over a pair of shared random bases.

``RandLoraLayer`` is the family root.  A layer holds, per adapter name, the two
scales ``randlora_lambda`` (``(r, n)``) and ``randlora_gamma`` (``(n, d_min)``) —
``ParameterDict``\ s, so a state dict reads ``....randlora_lambda.default`` and an
adapter file ``....randlora_lambda`` — and reads the bases ``randlora_A`` /
``randlora_B`` from the tables the tuner shares between all its layers (``bind``).

With :math:`A \in \mathbb{R}^{r\times 1\times d_{min}}` and
:math:`B \in \mathbb{R}^{d_{max}\times n\times r}` sliced to the layer, the update is
:math:`\frac{\alpha}{r}\,B'\,(\lambda \odot A \odot \gamma)'` where the primes flatten
``(r, n)`` and ``(n, r)`` axes in the order the reference does (``A`` rank-major, ``B``
basis-major), so a file moves between the two libraries unchanged.

A layer made on its own, outside a tuner, draws bases of its own size from its first
adapter's key the first time it needs them.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._shared import _BufferDict, _SharingLayer
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.randlora._bases import _Bases, _num_bases
from lucid.peft.tuners.randlora._config import RandLoraConfig


class _RandLoraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    scaling: float
    projection_prng_key: int
    sparse: bool
    very_sparse: bool


def _sparsity(spec: _RandLoraSpec, min_dim: int) -> float | None:
    """The ``s`` the bases are drawn with: ``None`` (dense), 3 (sparse) or ``sqrt(d_min)`` (very sparse)."""
    if spec.very_sparse:
        return math.sqrt(min_dim)
    return 3.0 if spec.sparse else None


class RandLoraLayer(_SharingLayer, _ConfigLayer, _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the basis-to-update map shared by the RandLoRA layers.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : RandLoraConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``config.fan_in_fan_out`` is set (the layer's weight is stored ``(out, in)``), or
        ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("randlora_lambda", "randlora_gamma")
    _shared_names: ClassVar[tuple[str, ...]] = ("randlora_A", "randlora_B")
    _repr_prefix = "randlora."
    randlora_lambda: nn.ParameterDict
    randlora_gamma: nn.ParameterDict
    randlora_dropout: nn.ModuleDict[nn.Module]

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter's bases, by name (read-only).")
    scaling = _SpecField[float]("Scale ``alpha / r`` of each adapter, by name (read-only).")

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        cfg = _config_as(RandLoraConfig, config)
        if cfg.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        self._hold(
            {
                "randlora_A": _BufferDict(cfg.save_projection),
                "randlora_B": _BufferDict(cfg.save_projection),
            }
        )
        self.randlora_dropout = nn.ModuleDict()
        # Bases drawn for a layer used alone: plain state, not part of ``state_dict``.
        self._unbound: dict[str, _Bases] = {}

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(RandLoraConfig, config)
        min_dim = min(self.in_features, self.out_features)
        bases = _num_bases(min_dim, cfg.r)
        anchor = self.weight
        device, dtype = anchor.device, _adapter_dtype(anchor)
        if cfg.init_weights:
            scales = lucid.zeros(cfg.r, bases, device=device, dtype=dtype)
            gamma = lucid.full(
                (bases, min_dim), 1.0 / max(bases, min_dim), device=device, dtype=dtype
            )
        else:
            scales = lucid.randn(cfg.r, bases, device=device, dtype=dtype)
            gamma = lucid.full(
                (bases, min_dim),
                1.0 / max(self.in_features, self.out_features),
                device=device,
                dtype=dtype,
            )
        dropout: nn.Module = (
            nn.Dropout(p=cfg.randlora_dropout) if cfg.randlora_dropout > 0.0 else nn.Identity()
        )
        self._specs[adapter_name] = _RandLoraSpec(
            cfg.r, cfg.randlora_alpha / cfg.r, cfg.projection_prng_key, cfg.sparse, cfg.very_sparse
        )
        self.randlora_lambda[adapter_name] = Parameter(scales)
        self.randlora_gamma[adapter_name] = Parameter(gamma)
        self.randlora_dropout[adapter_name] = dropout

    @override
    def _rebound(self) -> None:
        self._unbound.clear()

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its scale vectors, its dropout and any projections it drew."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.randlora_dropout:
            del self.randlora_dropout[adapter_name]
        self._unbound.pop(adapter_name, None)

    # ── the bases ─────────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _RandLoraSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _RandLoraSpec)
        return spec

    def _bases(self, adapter: str) -> _Bases:
        """The ``(A, B)`` ``adapter`` reads: the bound tables', or ones drawn for this layer's size.

        A tuner fills the tables it binds before any layer reads them.  A layer used
        without one draws from its key and keeps the pair out of its state: a forward
        pass must not change what ``state_dict`` holds, or the layer's own checkpoint
        would not load into a fresh instance.
        """
        held_a, held_b = self._table("randlora_A"), self._table("randlora_B")
        assert isinstance(held_a, _BufferDict) and isinstance(held_b, _BufferDict)
        if adapter in held_a and adapter in held_b:
            return _Bases(held_a[adapter], held_b[adapter])
        drawn = self._unbound.get(adapter)
        if drawn is None:
            spec, anchor = self._spec(adapter), self.weight
            max_dim, min_dim = max(self.in_features, self.out_features), min(
                self.in_features, self.out_features
            )
            drawn = _Bases.draw(
                spec.projection_prng_key,
                spec.r,
                max_dim,
                min_dim,
                _sparsity(spec, min_dim),
                anchor.device,
                _adapter_dtype(anchor),
            )
            self._unbound[adapter] = drawn
        return drawn

    def _factors(self, adapter: str) -> tuple[Tensor, Tensor]:
        r"""The two matrices whose product (``second @ first``) is the update, before the scale.

        Applied to an input, ``first`` goes first.  The scales multiply the smaller-side basis
        :math:`A`; the pair is ordered so ``first`` acts on the input's side: ``(in, ...)`` when
        the layer's input is its smaller side, and the transposes otherwise.
        """
        bases = _num_bases(min(self.in_features, self.out_features), self._spec(adapter).r)
        min_dim = min(self.in_features, self.out_features)
        max_dim = max(self.in_features, self.out_features)
        lam = _parameter_at(self.randlora_lambda, adapter)
        gamma = _parameter_at(self.randlora_gamma, adapter)
        basis_a, basis_b = self._bases(adapter)
        sliced_a = basis_a[:, :bases, :min_dim].to(lam.dtype)
        sliced_b = basis_b[:max_dim, :bases, :].to(lam.dtype)
        update_b = sliced_b.flatten(1)  # (d_max, n * r)
        update_a = (lam.unsqueeze(-1) * sliced_a * gamma.unsqueeze(0)).flatten(
            0, 1
        )  # (r * n, d_min)
        if min_dim == self.in_features:
            return update_a, update_b
        return update_b.mT, update_a.mT

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\frac{\alpha}{r}` times the update, shaped like the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            The change merging ``adapter`` makes to the base weight, in the scales' dtype.
        """
        first, second = self._factors(adapter)
        return (second @ first) * self._spec(adapter).scaling

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        first, second = self._factors(adapter)
        dropped: Tensor = self.randlora_dropout[adapter](x)
        dropped = dropped if dropped.dtype == first.dtype else dropped.to(first.dtype)
        return ((dropped @ first.mT) @ second.mT) * self._spec(adapter).scaling


class Linear(RandLoraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a RandLoRA update: :math:`y = W x + b + \frac{\alpha}{r}\,\Delta\,\mathrm{drop}(x)`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`RandLoraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.randlora import Linear, RandLoraConfig
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", RandLoraConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.randlora_lambda["default"].shape), tuple(layer.randlora_gamma["default"].shape)
    ((2, 2), (2, 4))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # lambda starts at zero
    True
    """

    base_layer: nn.Linear
