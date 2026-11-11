r"""The PVeRA layers: a frozen base module beside two trained vectors and the random projections their tuner shares.

PVeRA (Fillioux et al., 2025, arXiv:2512.07703, Sec. 3.2) is VeRA whose rank-:math:`r` latent is a
Gaussian.  Two frozen random matrices :math:`A \in \mathbb{R}^{2r\times d_{in}}` and
:math:`B \in \mathbb{R}^{d_{out}\times r}`, shared by every layer, and two learned vectors per layer,
:math:`\lambda_d \in \mathbb{R}^{2r}` and :math:`\lambda_b \in \mathbb{R}^{d_{out}}`, give

.. math:: \mu,\ \log\sigma^2 = \mathrm{split}(\lambda_d \odot A x), \qquad
          z = \mu + \epsilon \odot \sigma, \qquad y = W x + b + \lambda_b \odot B z .

``z = mu`` unless the layer is training or ``sample_at_inference`` is set (Sec. 3.3), and then the
update folds into the weight exactly (Eq. 10): :math:`\Delta W = (\lambda_b \odot B)(\lambda_{d,\mu} \odot A_\mu)`
with the first ``r`` rows of each.

``PveraLayer`` is the family root.  A layer holds, per adapter name, the two vectors in ``pvera_lambda_b``
(``(out,)``) and ``pvera_lambda_d`` (``(2r,)``) — ``ParameterDict``\ s, so a state dict reads
``....pvera_lambda_d.default`` and an adapter file ``....pvera_lambda_d`` — and reads the projections
``pvera_A`` / ``pvera_B`` from the tables the tuner shares between all its layers (``bind``).

A layer made on its own, outside a tuner, draws projections of its own size from its first adapter's
key the first time it needs them.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict, _SharingLayer
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.pvera._config import PveraConfig
from lucid.peft.tuners.pvera._projection import _Projection


class _PveraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    projection_prng_key: int
    sample_at_inference: bool


class PveraLayer(
    _SharingLayer, _DeltaLayer[PveraConfig], _LinearSized, _ResidualUpdate, family=True
):
    r"""Adapter bookkeeping, the latent and the vectors-to-update map shared by the PVeRA layers.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : PveraConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is another kind of module than the layer class wraps.
    ValueError
        ``config.fan_in_fan_out`` is set, or ``base_layer`` has yet to build its parameters.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("pvera_lambda_b", "pvera_lambda_d")
    _shared_names: ClassVar[tuple[str, ...]] = ("pvera_A", "pvera_B")
    _dropout_table = "pvera_dropout"
    _repr_prefix = "pvera."
    pvera_lambda_b: nn.ParameterDict
    pvera_lambda_d: nn.ParameterDict
    pvera_dropout: nn.ModuleDict[nn.Module]

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter's latent, by name (read-only).")
    sample_at_inference = _SpecField[bool](
        "Whether each adapter samples its latent in evaluation too, by name (read-only)."
    )

    @override
    def _prepare(self, config: PveraConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)
        self._hold(
            {
                "pvera_A": _BufferDict(config.save_projection),
                "pvera_B": _BufferDict(config.save_projection),
            }
        )
        # Projections drawn for a layer used alone: plain state, not part of ``state_dict``.
        self._unbound: dict[str, _Projection] = {}
        # One sampler per adapter, seeded by ``generator_seed``; ``None`` draws from the global one.
        self._samplers: dict[str, lucid.Generator | None] = {}

    @override
    def _enter(self, adapter_name: str, config: PveraConfig) -> None:
        home = self._home()
        if config.init_weights:
            lambda_b = home.zeros(self.out_features)
            lambda_d = home.full((2 * config.r,), config.d_initial)
        else:
            lambda_b = home.full((self.out_features,), 1.0)
            lambda_d = lucid.randn(2 * config.r, device=home.device, dtype=home.dtype)
        self._specs[adapter_name] = _PveraSpec(
            config.r, config.projection_prng_key, config.sample_at_inference
        )
        self.pvera_lambda_b[adapter_name] = Parameter(lambda_b)
        self.pvera_lambda_d[adapter_name] = Parameter(lambda_d)
        self._enter_dropout(adapter_name, config.pvera_dropout)
        seed = config.generator_seed
        self._samplers[adapter_name] = None if seed is None else lucid.Generator(seed, device="cpu")

    @override
    def _rebound(self) -> None:
        self._unbound.clear()

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its vectors, its dropout, its sampler and any projections it drew."""
        super().delete_adapter(adapter_name)
        self._unbound.pop(adapter_name, None)
        self._samplers.pop(adapter_name, None)

    # ── the projections ───────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _PveraSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _PveraSpec)  # ``_enter`` is the only writer
        return spec

    def _projection(self, adapter: str) -> _Projection:
        """The ``(A, B)`` ``adapter`` reads: the bound tables', or ones drawn for this layer's size.

        A tuner fills the tables it binds before any layer reads them.  A layer used without one
        draws from its key and keeps the pair out of its state: a forward pass must not change what
        ``state_dict`` holds.
        """
        held_a, held_b = self._table("pvera_A"), self._table("pvera_B")
        assert isinstance(held_a, _BufferDict) and isinstance(held_b, _BufferDict)
        if adapter in held_a and adapter in held_b:
            return _Projection(held_a[adapter], held_b[adapter])
        drawn = self._unbound.get(adapter)
        if drawn is None:
            spec, anchor = self._spec(adapter), self.weight
            home = self._home()
            drawn = _Projection.draw(
                spec.projection_prng_key,
                spec.r,
                self.out_features,
                self.in_features,
                anchor.device,
                home.dtype,
            )
            self._unbound[adapter] = drawn
        return drawn

    def _sliced(self, adapter: str) -> tuple[Tensor, Tensor]:
        """The projections cut to this layer: ``A[:2r, :in]`` and ``B[:out, :r]``, in the vectors' dtype."""
        r = self._spec(adapter).r
        dtype = _parameter_at(self.pvera_lambda_d, adapter).dtype
        proj_a, proj_b = self._projection(adapter)
        return (
            proj_a[: 2 * r, : self.in_features].to(dtype=dtype),
            proj_b[: self.out_features, :r].to(dtype=dtype),
        )

    # ── the update ────────────────────────────────────────────────────────

    def _latent(self, adapter: str, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""The latent: :math:`\mu` in evaluation, :math:`\mu + \epsilon\,e^{\frac12\log\sigma^2}` while training or when ``sample_at_inference`` is set."""
        if not self.training and not self._spec(adapter).sample_at_inference:
            return mu
        std = (0.5 * logvar).exp()
        sampler = self._samplers.get(adapter)
        if sampler is None:
            noise = lucid.randn(*std.shape, dtype=std.dtype, device=std.device)
        else:
            # Drawn on the CPU, where the generator is, then moved beside the activations.
            noise = lucid.randn(*std.shape, dtype=std.dtype, generator=sampler).to(std.device)
        return mu + noise * std

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""The dense update of ``adapter`` as it computes with the latent at its mean (Eq. 10 of the paper).

        Parameters
        ----------
        adapter : str
            Name of the adapter whose update is wanted.

        Returns
        -------
        Tensor
            ``(out, in)``, in the dtype of the vectors.
        """
        r = self._spec(adapter).r
        proj_a, proj_b = self._sliced(adapter)
        lambda_d = _parameter_at(self.pvera_lambda_d, adapter)
        lambda_b = _parameter_at(self.pvera_lambda_b, adapter)
        return (lambda_b.unsqueeze(-1) * proj_b) @ (lambda_d[:r].unsqueeze(-1) * proj_a[:r])

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        r = self._spec(adapter).r
        proj_a, proj_b = self._sliced(adapter)
        lambda_d = _parameter_at(self.pvera_lambda_d, adapter)
        lambda_b = _parameter_at(self.pvera_lambda_b, adapter)
        dropped = self._dropped(adapter, x)
        dropped = dropped if dropped.dtype == lambda_d.dtype else dropped.to(lambda_d.dtype)
        hidden = (dropped @ proj_a.mT) * lambda_d
        latent = self._latent(adapter, hidden[..., :r], hidden[..., r:])
        return lambda_b * (latent @ proj_b.mT)


class Linear(PveraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a PVeRA update: :math:`y = W x + b + \lambda_b \odot B\,z`, :math:`z` the Gaussian latent.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`PveraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PveraConfig
    >>> from lucid.peft.tuners.pvera import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", PveraConfig(r=3, target_modules=["x"]))
    >>> tuple(layer.pvera_lambda_d["default"].shape), tuple(layer.pvera_lambda_b["default"].shape)
    ((6,), (4,))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # lambda_b starts at zero
    True
    """

    base_layer: nn.Linear
