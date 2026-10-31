r"""The VeRA layers: a frozen base module beside two trainable vectors and a pair of shared random matrices.

``VeraLayer`` is the family root.  A layer holds, per adapter name, the two
vectors in ``vera_lambda_d`` (``(r,)``) and ``vera_lambda_b`` (``(out,)``) —
``ParameterDict``\ s, so a state dict reads ``....vera_lambda_d.default`` and
an adapter file ``....vera_lambda_d`` — and reads the projections ``vera_A`` /
``vera_B`` from the containers the tuner shares between all its layers
(``bind``).  The update is :math:`\lambda_b \odot B_{[:out,:r]}
(\lambda_d \odot A_{[:r,:in]}\,x)`.

A layer made on its own, outside a tuner, draws a projection of its own size
from its first adapter's key the first time it needs one.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._base import _check_adapter_name, _parameter_at
from lucid.peft.tuners._support import _adapter_dtype, _ResidualUpdate, _SpecField
from lucid.peft.tuners.vera._config import VeraConfig
from lucid.peft.tuners.vera._projection import _BufferDict, _Projection


class _VeraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    projection_prng_key: int
    save_projection: bool


class VeraLayer(_ResidualUpdate, family=True):
    r"""Adapter bookkeeping shared by the VeRA layers.

    Holds the wrapped module as ``base_layer`` — the very module, so its
    parameters stay the same ``Parameter`` objects — and, per adapter name, the
    vectors of the adapter.  The forward pass adds the update of every active
    adapter this layer holds; a layer given no projection (``bind``) draws one
    from the key of its first adapter.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    config : VeraConfig
        The config; dropout, ``d_initial`` and ``init_weights`` are read.
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a positive rank.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``r`` is not positive, ``config.fan_in_fan_out`` is set (the layer's weight
        is stored ``(out, in)``), or ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("vera_lambda_b", "vera_lambda_d")

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter, by name (read-only).")

    def __init__(
        self, base_layer: nn.Module, adapter_name: str, config: VeraConfig, r: int = 0
    ) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super().__init__(base_layer)
        self._specs: dict[str, _VeraSpec] = {}
        self.vera_lambda_b: nn.ParameterDict = nn.ParameterDict()
        self.vera_lambda_d: nn.ParameterDict = nn.ParameterDict()
        self.vera_dropout: nn.ModuleDict[nn.Module] = nn.ModuleDict()
        # The shared projections: empty until the tuner (or the first forward pass) binds them.
        self.vera_A: _BufferDict | None = None
        self.vera_B: _BufferDict | None = None
        self.update_layer(adapter_name, r, config)

    # ── the layer ─────────────────────────────────────────────────────────

    @property
    def in_features(self) -> int:
        """Input features of the base layer."""
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        """Output features of the base layer."""
        return int(self.weight.shape[0])

    @override
    def __repr__(self) -> str:
        return "vera." + super().__repr__()

    @override
    def extra_repr(self) -> str:
        return (
            f"adapter={self.active_adapter!r}, r={dict(self.r)}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    @override
    def _adapter_names(self) -> list[str]:
        return list(self._specs)

    # ── creating, binding and removing an adapter ────────────────────────

    def update_layer(self, adapter_name: str, r: int, config: VeraConfig) -> None:
        r"""Create adapter ``adapter_name``: :math:`\lambda_d` of size ``r`` and :math:`\lambda_b` of the layer's output size.

        The vectors live on the base weight's device, in its dtype — float32 for a
        half-precision base.  A layer that holds an adapter already keeps applying
        the adapters it applied: the new one is inactive until ``set_adapter``
        names it.  The projection the adapter reads is the layer's ``bind``-ing.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each table.
        r : int
            Rank; positive.
        config : VeraConfig
            Read for ``vera_dropout``, ``d_initial``, ``init_weights`` and the projection's key.

        Raises
        ------
        ValueError
            The layer already holds an adapter of this name, or ``r`` is not positive.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self._specs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        device, dtype = self.weight.device, _adapter_dtype(self.weight)
        if config.init_weights:
            lambda_b = Parameter(lucid.zeros(self.out_features, device=device, dtype=dtype))
            lambda_d = Parameter(lucid.full((r,), config.d_initial, device=device, dtype=dtype))
        else:
            lambda_b = Parameter(lucid.ones(self.out_features, device=device, dtype=dtype))
            lambda_d = Parameter(lucid.randn(r, device=device, dtype=dtype))
        dropout: nn.Module = (
            nn.Dropout(p=config.vera_dropout) if config.vera_dropout > 0.0 else nn.Identity()
        )
        self._specs[adapter_name] = _VeraSpec(r, config.projection_prng_key, config.save_projection)
        self.vera_lambda_b[adapter_name] = lambda_b
        self.vera_lambda_d[adapter_name] = lambda_d
        self.vera_dropout[adapter_name] = dropout
        if not self.active_adapters:
            self.active_adapter = adapter_name

    def bind(self, vera_a: _BufferDict, vera_b: _BufferDict) -> None:
        """Read the projections from these containers — the ones the tuner shares between its layers."""
        self.vera_A = vera_a
        self.vera_B = vera_b

    def _projection(self, adapter: str) -> _Projection:
        """The ``(A, B)`` ``adapter`` reads; one drawn for this layer's size if nothing holds it yet.

        A tuner fills the containers it binds before any layer reads them, so
        this only draws for a layer used without one.
        """
        spec = self._specs[adapter]
        if self.vera_A is None or self.vera_B is None:
            self.bind(_BufferDict(spec.save_projection), _BufferDict(spec.save_projection))
        assert self.vera_A is not None and self.vera_B is not None
        if adapter not in self.vera_A:
            anchor = self.weight
            self.vera_A[adapter], self.vera_B[adapter] = _Projection.draw(
                spec.projection_prng_key,
                spec.r,
                self.out_features,
                self.in_features,
                anchor.device,
                _adapter_dtype(anchor),
            )
        return _Projection(self.vera_A[adapter], self.vera_B[adapter])

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        if adapter_name in self.vera_dropout:
            del self.vera_dropout[adapter_name]
        self._specs.pop(adapter_name, None)

    # ── the update ────────────────────────────────────────────────────────

    def _sliced(self, adapter: str) -> _Projection:
        """The projection cut to this layer: ``A[:r, :in]`` and ``B[:out, :r]``, in the vectors' dtype."""
        r = self._specs[adapter].r
        dtype = _parameter_at(self.vera_lambda_d, adapter).dtype
        vera_a, vera_b = self._projection(adapter)
        return _Projection(
            vera_a[:r, : self.in_features].to(dtype=dtype),
            vera_b[: self.out_features, :r].to(dtype=dtype),
        )

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\mathrm{diag}(\lambda_b)\,B\,\mathrm{diag}(\lambda_d)\,A`, shaped like the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            The change merging ``adapter`` makes to the base weight, in the vectors' dtype.
        """
        vera_a, vera_b = self._sliced(adapter)
        lambda_d = _parameter_at(self.vera_lambda_d, adapter)
        lambda_b = _parameter_at(self.vera_lambda_b, adapter)
        return (lambda_b.unsqueeze(-1) * vera_b) @ (lambda_d.unsqueeze(-1) * vera_a)


class Linear(VeraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a VeRA update: :math:`y = W x + b + \lambda_b \odot B(\lambda_d \odot A\,\mathrm{drop}(x))`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in every table.
    config : VeraConfig
        The adapter's config.
    r : int, default=0
        Rank; pass a positive one.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.vera import VeraConfig
    >>> from lucid.peft.tuners.vera import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", VeraConfig(target_modules=["x"]), r=16)
    >>> tuple(layer.vera_lambda_d["default"].shape), tuple(layer.vera_lambda_b["default"].shape)
    ((16,), (4,))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # lambda_b starts at zero
    True
    """

    base_layer: nn.Linear

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        vera_a, vera_b = self._sliced(adapter)
        lambda_d = _parameter_at(self.vera_lambda_d, adapter)
        lambda_b = _parameter_at(self.vera_lambda_b, adapter)
        dropped: Tensor = self.vera_dropout[adapter](x)
        dropped = dropped if dropped.dtype == lambda_d.dtype else dropped.to(lambda_d.dtype)
        return lambda_b * ((lambda_d * (dropped @ vera_a.mT)) @ vera_b.mT)
