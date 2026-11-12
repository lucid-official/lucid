r"""The VB-LoRA layers: a frozen base module beside logits that pick, from a shared vector bank, the rows of its low-rank pair.

``VBLoRALayer`` is the family root.  A layer holds, per adapter name, the logits
``vblora_logits_A`` (``(r, in / b, N)``) and ``vblora_logits_B``
(``(out / b, r, N)``) — ``ParameterDict``\ s, so a state dict reads
``....vblora_logits_A.default`` and an adapter file ``....vblora_logits_A`` — and
reads the vector bank ``vblora_vector_bank`` (``(N, b)``, one per adapter) from the
table the tuner shares between all its layers (``bind``).

Each sub-vector of :math:`A` and :math:`B` is the softmax-weighted sum of the
top-:math:`k` bank vectors its logits pick, so :math:`A \in \mathbb{R}^{r \times d_{in}}`
and :math:`B \in \mathbb{R}^{d_{out} \times r}` are built from the bank on each use and
the update is :math:`B\,A\,x`.  The bank is trained too.

A layer made on its own, outside a tuner, draws a bank of its own for each adapter.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._shared import _TrainedSharedLayer
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.vblora._config import VBLoRAConfig


class _VBLoRASpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    topk: int


def _new_bank(config: VBLoRAConfig, like: Tensor) -> Parameter:
    """A bank of ``num_vectors`` vectors, uniform in ``±init_vector_bank_bound``, beside ``like``."""
    bound = config.init_vector_bank_bound
    bank = lucid.empty(
        config.num_vectors, config.vector_length, device=like.device, dtype=_adapter_dtype(like)
    )
    bank.uniform_(-bound, bound)
    return Parameter(bank)


def _mix(logits: Tensor, bank: Tensor, topk: int) -> Tensor:
    """The sub-vectors ``logits`` (``(..., N)``) pick from ``bank``: the top-``k`` rows, softmax-weighted (``(..., b)``)."""
    values, indices = logits.topk(topk, dim=-1)
    weights = F.softmax(values, dim=-1)
    return (weights.unsqueeze(-1) * bank[indices]).sum(-2)


class VBLoRALayer(_TrainedSharedLayer, _ConfigLayer, _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the bank-to-factors map shared by the VB-LoRA layers.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : VBLoRAConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``base_layer``'s sizes are not multiples of ``config.vector_length``,
        ``config.fan_in_fan_out`` is set (the layer's weight is stored ``(out, in)``), or
        ``base_layer`` builds parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("vblora_logits_A", "vblora_logits_B")
    _shared_names: ClassVar[tuple[str, ...]] = ("vblora_vector_bank",)
    _repr_prefix = "vblora."
    vblora_logits_A: nn.ParameterDict
    vblora_logits_B: nn.ParameterDict
    vblora_vector_bank: nn.ParameterDict
    vblora_dropout: nn.ModuleDict[nn.Module]

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter, by name (read-only).")
    topk = _SpecField[int](
        "Bank vectors mixed per sub-vector by each adapter, by name (read-only)."
    )

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        cfg = _config_as(VBLoRAConfig, config)
        if cfg.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); this layer's weight is "
                "stored (out, in)"
            )
        self._hold({"vblora_vector_bank": nn.ParameterDict()})
        self.vblora_dropout = nn.ModuleDict()

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(VBLoRAConfig, config)
        for label, size in (("in_features", self.in_features), ("out_features", self.out_features)):
            if size % cfg.vector_length != 0:
                raise ValueError(
                    f"`{label}` {size} must be divisible by `vector_length` {cfg.vector_length}"
                )
        anchor = self.weight
        device, dtype = anchor.device, _adapter_dtype(anchor)
        logits_a = lucid.randn(
            cfg.r,
            self.in_features // cfg.vector_length,
            cfg.num_vectors,
            device=device,
            dtype=dtype,
        )
        logits_b = lucid.randn(
            self.out_features // cfg.vector_length,
            cfg.r,
            cfg.num_vectors,
            device=device,
            dtype=dtype,
        )
        dropout: nn.Module = (
            nn.Dropout(p=cfg.vblora_dropout) if cfg.vblora_dropout > 0.0 else nn.Identity()
        )
        self._specs[adapter_name] = _VBLoRASpec(cfg.r, cfg.topk)
        self.vblora_logits_A[adapter_name] = Parameter(logits_a * cfg.init_logits_std)
        self.vblora_logits_B[adapter_name] = Parameter(logits_b * cfg.init_logits_std)
        self.vblora_dropout[adapter_name] = dropout
        if self._bound is None:
            # A layer used alone draws the bank it reads; a tuner's layers read the tuner's.
            self.vblora_vector_bank[adapter_name] = _new_bank(cfg, anchor)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its logits and its dropout."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.vblora_dropout:
            del self.vblora_dropout[adapter_name]

    # ── the factors ───────────────────────────────────────────────────────

    def _spec(self, adapter: str) -> _VBLoRASpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _VBLoRASpec)
        return spec

    def _factors(self, adapter: str) -> tuple[Tensor, Tensor]:
        r"""The pair :math:`(A, B)` ``adapter`` reads: ``(r, in)`` and ``(out, r)``, built from its logits and bank."""
        logits_a = _parameter_at(self.vblora_logits_A, adapter)
        logits_b = _parameter_at(self.vblora_logits_B, adapter)
        bank_table = self._table("vblora_vector_bank")
        assert isinstance(bank_table, nn.ParameterDict)
        bank = _parameter_at(bank_table, adapter).to(logits_a.device)
        topk = self._spec(adapter).topk
        # A: (r, in / b, b) -> (r, in).  B: (out / b, r, b) -> (out / b, b, r) -> (out, r).
        factor_a = _mix(logits_a, bank, topk).reshape(logits_a.shape[0], -1)
        factor_b = _mix(logits_b, bank, topk).transpose(1, 2).reshape(-1, logits_b.shape[1])
        return factor_a, factor_b

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`B A`, the change merging ``adapter`` makes to the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the logits' dtype.
        """
        factor_a, factor_b = self._factors(adapter)
        return factor_b @ factor_a

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        factor_a, factor_b = self._factors(adapter)
        dropped: Tensor = self.vblora_dropout[adapter](x)
        dropped = dropped if dropped.dtype == factor_a.dtype else dropped.to(factor_a.dtype)
        return (dropped @ factor_a.mT) @ factor_b.mT


class Linear(VBLoRALayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a VB-LoRA update: :math:`y = W x + b + B\,A\,\mathrm{drop}(x)`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`VBLoRALayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.vblora import Linear, VBLoRAConfig
    >>> base = nn.Linear(8, 4)
    >>> config = VBLoRAConfig(r=2, num_vectors=6, vector_length=4, target_modules=["x"])
    >>> layer = Linear(base, "default", config)
    >>> tuple(layer.vblora_logits_A["default"].shape), tuple(layer.vblora_logits_B["default"].shape)
    ((2, 2, 6), (1, 2, 6))
    >>> tuple(layer.vblora_vector_bank["default"].shape)
    (6, 4)
    >>> tuple(layer(lucid.randn(3, 8)).shape)
    (3, 4)
    """

    base_layer: nn.Linear
