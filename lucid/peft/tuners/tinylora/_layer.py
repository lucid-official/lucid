r"""The TinyLoRA layers: a frozen base module beside the frozen SVD of its weight and a vector that fits in a handful of numbers.

TinyLoRA (Morris et al., 2026, arXiv:2602.04118, Sec. 4.0.2) keeps the rank-:math:`r` truncated SVD
:math:`W \approx U \Sigma V^\top` of each adapted weight frozen and trains only a vector
:math:`v \in \mathbb{R}^u`.  Fixed random matrices :math:`P_i \in \mathbb{R}^{r\times r}` turn it into
:math:`R = \sum_i v_i P_i`, and the layer adds :math:`U \Sigma^{1/2}\,R\,\Sigma^{1/2} V^\top` — the
singular values split evenly over the two frozen factors :math:`B = U\Sigma^{1/2}` and
:math:`A = \Sigma^{1/2}V^\top`, so :math:`\Delta W = B R A`.  Modules may share one :math:`v`
(``weight_tying``): at full tying the whole model learns ``u`` numbers.

``TinyLoraLayer`` is the family root.  A layer holds, per adapter name, ``tinylora_A`` (``(r, in)``),
``tinylora_B`` (``(out, r)``) and ``tinylora_P`` (``(u, r, r)``) as buffers — the first two always saved,
``tinylora_P`` only when ``config.save_projection`` is set — and reads its vector from ``tinylora_v``,
the table its tuner shares between all its layers (``bind``): per adapter, a ``ParameterDict`` of the
groups of modules that share a vector, each ``(u,)``.

A layer made on its own, outside a tuner, holds a vector of its own and is the one module of its group.
"""

import math
from abc import abstractmethod
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn import Parameter
from lucid.peft.tuners._delta import _DeltaLayer, _Home
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict, _TrainedSharedLayer
from lucid.peft.tuners._support import _ResidualUpdate, _SpecField
from lucid.peft.tuners.tinylora._config import TinyLoraConfig


class _TinyLoraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    u: int
    group: str
    projection_seed: int


def _new_vector(config: TinyLoraConfig, home: _Home) -> Parameter:
    """A trained vector of ``config.u`` numbers: zero (the update starts at nothing), uniform in the bound, or standard normal."""
    start = home.zeros(config.u)
    if config.init_weights == "uniform":
        start.uniform_(-config.init_v_bound, config.init_v_bound)
    elif not config.init_weights:
        start = lucid.randn(config.u, device=home.device, dtype=home.dtype)
    return Parameter(start)


def _svd_factors(weight: Tensor, rank: int, home: _Home) -> tuple[Tensor, Tensor, int]:
    r"""The frozen factors :math:`A = \Sigma^{1/2}V^\top` (``(r, cols)``) and :math:`B = U\Sigma^{1/2}` (``(rows, r)``) of a weight, and the rank kept.

    The rank is ``rank``, or the weight's smaller side if that is less.  The decomposition is taken
    in float32, then cast to where the adapter lives.
    """
    with lucid.no_grad():
        u, s, vh = lucid.linalg.svd(weight.detach().to(lucid.float32), full_matrices=False)
    kept = min(rank, int(s.shape[0]))
    root = s[:kept].sqrt()
    a = (root.unsqueeze(1) * vh[:kept]).to(device=home.device, dtype=home.dtype)
    b = (u[:, :kept] * root.unsqueeze(0)).to(device=home.device, dtype=home.dtype)
    return a, b, kept


class TinyLoraLayer(_TrainedSharedLayer, _DeltaLayer[TinyLoraConfig], _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the vector-to-update map shared by the TinyLoRA layers.

    A layer type supplies how the update acts on an input (``_through``); the frozen factors, the
    projection, the vector and the merge are here.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : TinyLoraConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear`` or ``nn.Embedding`` as the layer class requires.
    ValueError
        ``config.fan_in_fan_out`` is set, or ``base_layer`` is lazy and has not been run.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ()
    _shared_names: ClassVar[tuple[str, ...]] = ("tinylora_v",)
    _dropout_table = "tinylora_dropout"
    _repr_prefix = "tinylora."
    tinylora_dropout: nn.ModuleDict[nn.Module]
    tinylora_v: nn.ModuleDict[nn.Module]
    tinylora_A: _BufferDict
    tinylora_B: _BufferDict
    tinylora_P: _BufferDict

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter's frozen SVD, by name (read-only).")
    u = _SpecField[int]("Length of each adapter's trained vector, by name (read-only).")

    @override
    def _prepare(self, config: TinyLoraConfig) -> None:
        if config.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        super()._prepare(config)
        self._hold({"tinylora_v": nn.ModuleDict()})
        self.tinylora_A = _BufferDict(persistent=True)
        self.tinylora_B = _BufferDict(persistent=True)
        self.tinylora_P = _BufferDict(persistent=config.save_projection)

    @override
    def _enter(self, adapter_name: str, config: TinyLoraConfig) -> None:
        home = self._home()
        a, b, kept = _svd_factors(self.weight, config.r, home)
        spec = _TinyLoraSpec(kept, config.u, "0", config.projection_seed)
        self._specs[adapter_name] = spec
        self.tinylora_A[adapter_name], self.tinylora_B[adapter_name] = a, b
        self.tinylora_P[adapter_name] = self._draw_projection(spec, 0, home)
        self._enter_dropout(adapter_name, config.tinylora_dropout)
        if self._bound is None:
            # A layer used alone holds the vector it reads; a tuner's layers read the tuner's.
            self.tinylora_v[adapter_name] = nn.ParameterDict({"0": _new_vector(config, home)})

    @override
    def _adapter_names(self) -> list[str]:
        return list(self._specs)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its factors, its projection, its dropout, and its vectors if this layer held them."""
        super().delete_adapter(adapter_name)
        for held in (self.tinylora_A, self.tinylora_B, self.tinylora_P):
            if adapter_name in held:
                del held[adapter_name]
        if self._bound is None and adapter_name in self.tinylora_v:
            del self.tinylora_v[adapter_name]

    # ── the group and the projection ──────────────────────────────────────

    def _spec(self, adapter: str) -> _TinyLoraSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _TinyLoraSpec)  # ``_enter`` is the only writer
        return spec

    @staticmethod
    def _draw_projection(spec: _TinyLoraSpec, index: int, home: _Home) -> Tensor:
        r"""The fixed :math:`P \in \mathbb{R}^{u\times r\times r}`: normal, standard deviation :math:`1/\sqrt r`, seeded by ``projection_seed + index``."""
        generator = lucid.Generator(spec.projection_seed + index, device="cpu")
        drawn = lucid.randn(spec.u, spec.r, spec.r, generator=generator) / math.sqrt(spec.r)
        return drawn.to(device=home.device, dtype=home.dtype)

    def place(self, adapter: str, index: int, group: str) -> None:
        """Give this layer its place among the adapter's modules: the group whose vector it reads, and the position that seeds its projection.

        Parameters
        ----------
        adapter : str
            The adapter's name.
        index : int
            The module's position among the adapter's targets, in model order.
        group : str
            The key of the vector in ``tinylora_v[adapter]``.
        """
        spec = self._spec(adapter)
        self._specs[adapter] = spec._replace(group=group)
        self.tinylora_P[adapter] = self._draw_projection(spec, index, self._home())

    def _vectors(self, adapter: str) -> nn.ParameterDict:
        table = self._table("tinylora_v")
        assert isinstance(table, nn.ModuleDict)
        vectors = table[adapter]
        assert isinstance(vectors, nn.ParameterDict)
        return vectors

    def _compute_r(self, adapter: str) -> Tensor:
        r"""The ``(r, r)`` matrix :math:`R = \sum_i v_i P_i` of the vector this layer's group trains."""
        spec = self._spec(adapter)
        v = _parameter_at(self._vectors(adapter), spec.group)
        p = self.tinylora_P[adapter].to(device=v.device, dtype=v.dtype)
        return (v @ p.reshape(spec.u, spec.r * spec.r)).reshape(spec.r, spec.r)

    # ── the update ────────────────────────────────────────────────────────

    @abstractmethod
    def _through(self, adapter: str, x: Tensor, a: Tensor, r: Tensor, b: Tensor) -> Tensor:
        """The update applied to the input ``x``: the base layer's way of using a weight ``B R A``."""

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""The dense update of ``adapter``: :math:`B R A`, shaped like the base weight.

        Parameters
        ----------
        adapter : str
            Name of the adapter whose update is wanted.

        Returns
        -------
        Tensor
            Like the base weight, in the dtype of the adapter's vector.
        """
        r = self._compute_r(adapter)
        a = self.tinylora_A[adapter].to(device=r.device, dtype=r.dtype)
        b = self.tinylora_B[adapter].to(device=r.device, dtype=r.dtype)
        return b @ r @ a

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        r = self._compute_r(adapter)
        a = self.tinylora_A[adapter].to(device=r.device, dtype=r.dtype)
        b = self.tinylora_B[adapter].to(device=r.device, dtype=r.dtype)
        return self._through(adapter, x, a, r, b)


class Linear(TinyLoraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a TinyLoRA update: :math:`y = W x + b + B R A\,\mathrm{drop}(x)`, :math:`R = \sum_i v_i P_i`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`TinyLoraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import TinyLoraConfig
    >>> from lucid.peft.tuners.tinylora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", TinyLoraConfig(r=2, u=3, target_modules=["x"]))
    >>> tuple(layer.tinylora_A["default"].shape), tuple(layer.tinylora_B["default"].shape)
    ((2, 8), (4, 2))
    >>> x = lucid.randn(2, 8)
    >>> bool((layer(x) == base(x)).all().item())   # v starts at zero
    True
    """

    base_layer: nn.Linear

    @override
    def _through(self, adapter: str, x: Tensor, a: Tensor, r: Tensor, b: Tensor) -> Tensor:
        dropped = self._dropped(adapter, x)
        dropped = dropped if dropped.dtype == a.dtype else dropped.to(a.dtype)
        return ((dropped @ a.mT) @ r.mT) @ b.mT


class Embedding(TinyLoraLayer, base_type=nn.Embedding):
    r"""An ``nn.Embedding`` with a TinyLoRA update: the table :math:`E` is looked up as :math:`E + B R A`.

    The frozen SVD is that of the table, ``(num_embeddings, embedding_dim)``: :math:`B` has one row
    per token and :math:`A` one column per feature, and the update of a token is the row of :math:`B`
    it indexes, through :math:`R` and :math:`A`.  The dropout acts on the looked-up rows.

    Parameters
    ----------
    base_layer : nn.Embedding
        The layer to adapt; the other parameters are those of :class:`TinyLoraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import TinyLoraConfig
    >>> from lucid.peft.tuners.tinylora import Embedding
    >>> base = nn.Embedding(10, 4)
    >>> layer = Embedding(base, "default", TinyLoraConfig(r=2, u=3, target_modules=["x"]))
    >>> tuple(layer.tinylora_A["default"].shape), tuple(layer.tinylora_B["default"].shape)
    ((2, 4), (10, 2))
    >>> ids = lucid.tensor([[1, 2, 3]])
    >>> bool((layer(ids) == base(ids)).all().item())   # v starts at zero
    True
    """

    base_layer: nn.Embedding

    @override
    def _through(self, adapter: str, x: Tensor, a: Tensor, r: Tensor, b: Tensor) -> Tensor:
        looked: Tensor = F.embedding(x, b)
        return (self._dropped(adapter, looked) @ r) @ a
