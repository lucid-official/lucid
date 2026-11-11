r"""The Super-Tuning layers: a frozen linear module beside a sparse subset of its own weight and, optionally, a LoRA pair.

Super-Tuning (Ilin et al., 2026, arXiv:2607.09287) picks, once and without data, the
``1 - sparsity`` fraction of the entries of a weight with the largest (``select_top=False``: the
smallest) magnitude and trains those entries alone: :math:`W' = W + \Delta`, with :math:`\Delta` zero
off that support and zero at the start.  With ``r`` set (the paper's Supra hybrid) a LoRA update
:math:`\tfrac{\alpha}{r} B A` is added to it, :math:`B` zero at the start.

``SupertuningLayer`` is the family root.  An adapter holds ``supertuning_values`` (``(K,)``,
:math:`K = \lfloor n (1 - \text{sparsity}) \rfloor` for a weight of :math:`n` entries), the positions
of those entries ``supertuning_indices`` (``(K,)`` int32, flat and row-major, in the order of the
ranking: largest first, or smallest first), and, with ``r``, ``supertuning_lora_A`` and
``supertuning_lora_B`` (bias-free ``nn.Linear`` modules), the reference's names and shapes.  The
positions are saved with the adapter unless ``save_precomputed_indices`` is ``False``, in which case
they are chosen again from the base weight when the adapter is built.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn import init
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._sparse import _matrix_from_entries
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.supertuning._config import SupertuningConfig


class _SupertuningSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the rank and the
    scale of the LoRA part are baked into the graph.
    """

    supertuning_rank: int
    supertuning_lora_alpha: float


class SupertuningLayer(_DeltaLayer[SupertuningConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the sparse-plus-LoRA update shared by the Super-Tuning layers.

    Per adapter: the values and positions of the trained entries, the LoRA pair and its dropout
    when ``r`` is set, and the read-only mappings ``supertuning_rank`` (zero without LoRA) and
    ``supertuning_lora_alpha``.  ``get_delta_weight`` is the dense matrix of the sparse update plus
    :math:`\tfrac{\alpha}{r} B A`; the forward pass adds the same update applied to the input.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight is partly trained; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : SupertuningConfig
        ``sparsity``, ``select_top``, the LoRA options and ``save_precomputed_indices``.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    ValueError
        ``sparsity`` leaves no entry to train, or the module still has lazy parameters.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = (
        "supertuning_values",
        "supertuning_lora_A",
        "supertuning_lora_B",
    )
    _dropout_table = "supertuning_lora_dropout"
    _repr_prefix = "supertuning."
    supertuning_values: nn.ParameterDict
    supertuning_lora_A: nn.ModuleDict[nn.Module]
    supertuning_lora_B: nn.ModuleDict[nn.Module]
    supertuning_indices: _BufferDict

    supertuning_rank = _SpecField[int](
        "LoRA rank of each adapter, 0 without one, by name (read-only)."
    )
    supertuning_lora_alpha = _SpecField[float](
        "Scale numerator of each adapter's LoRA part, by name (read-only)."
    )

    @override
    def _prepare(self, config: SupertuningConfig) -> None:
        super()._prepare(config)
        # The LoRA halves are modules, as the reference holds them (``...lora_A.default.weight``);
        # the values are a bare parameter each.
        self.supertuning_lora_A = nn.ModuleDict()
        self.supertuning_lora_B = nn.ModuleDict()
        self.supertuning_indices = _BufferDict(persistent=config.save_precomputed_indices)

    @staticmethod
    def _support(weight: Tensor, config: SupertuningConfig) -> Tensor:
        """The flat positions of the entries to train: the ``1 - sparsity`` fraction by magnitude, best first."""
        scores = weight.detach().abs().flatten()
        count = int(scores.shape[0] * (1 - config.sparsity))
        if count < 1:
            raise ValueError(
                f"sparsity={config.sparsity} leaves no trainable entries for a weight with "
                f"{int(scores.shape[0])} elements — no parameter would be adapted. "
                "Lower the sparsity so at least one entry is trainable."
            )
        return lucid.topk(scores, count, largest=config.select_top).indices.to(lucid.int32)

    @override
    def _enter(self, adapter_name: str, config: SupertuningConfig) -> None:
        home = self._home()
        positions = self._support(self.weight, config)
        count = int(positions.shape[0])
        self.supertuning_indices[adapter_name] = positions
        values = (
            home.zeros(count)
            if config.init_weights
            else lucid.randn(count, device=home.device, dtype=home.dtype)
        )
        rank = config.r or 0
        alpha = float(config.lora_alpha) if rank and config.lora_alpha is not None else 0.0
        if rank:
            down, up = home.linear(self.in_features, rank), home.linear(rank, self.out_features)
            init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            if config.init_weights:  # B zero: the update starts at zero
                init.zeros_(up.weight)
            else:
                init.kaiming_uniform_(up.weight, a=math.sqrt(5))
            self.supertuning_lora_A[adapter_name] = down
            self.supertuning_lora_B[adapter_name] = up
            self._enter_dropout(adapter_name, config.lora_dropout)
        self._specs[adapter_name] = _SupertuningSpec(rank, alpha)
        self.supertuning_values[adapter_name] = Parameter(values)

    def _spec(self, adapter: str) -> _SupertuningSpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _SupertuningSpec)
        return spec

    def _low_rank(self, adapter: str) -> tuple[Tensor, Tensor, float] | None:
        """``(A, B, alpha / r)`` of the adapter's LoRA part, or ``None`` without one."""
        spec = self._spec(adapter)
        if not spec.supertuning_rank:
            return None
        down, up = self.supertuning_lora_A[adapter], self.supertuning_lora_B[adapter]
        assert isinstance(down, nn.Linear) and isinstance(up, nn.Linear)
        return down.weight, up.weight, spec.supertuning_lora_alpha / spec.supertuning_rank

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its values, positions, LoRA pair and dropout."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.supertuning_indices:
            del self.supertuning_indices[adapter_name]

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the update as a dense matrix shaped like the base weight: the sparse entries plus :math:`\tfrac{\alpha}{r} B A`.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            In the adapter's dtype.
        """
        values = _parameter_at(self.supertuning_values, adapter)
        delta = _matrix_from_entries(
            (self.out_features, self.in_features), self.supertuning_indices[adapter], values
        )
        low = self._low_rank(adapter)
        if low is None:
            return delta
        down, up, scale = low
        return delta + scale * (up @ down)

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        # The sparse part reads the input as it comes; only the LoRA part drops some of it out.
        values = _parameter_at(self.supertuning_values, adapter)
        sparse = _matrix_from_entries(
            (self.out_features, self.in_features), self.supertuning_indices[adapter], values
        )
        out: Tensor = x.to(sparse.dtype) @ sparse.mT
        low = self._low_rank(adapter)
        if low is None:
            return out
        down, up, scale = low
        dropped = self._dropped(adapter, x).to(down.dtype)
        return out + scale * ((dropped @ down.mT) @ up.mT)


class Linear(SupertuningLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a Super-Tuning update: :math:`y = (W + \Delta + \tfrac{\alpha}{r} B A)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`SupertuningLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import SupertuningConfig
    >>> from lucid.peft.tuners.supertuning import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", SupertuningConfig(sparsity=0.75, target_modules=["x"]))
    >>> tuple(layer.supertuning_values["default"].shape), tuple(layer.supertuning_indices["default"].shape)
    ((8,), (8,))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the values start at zero
    True
    """

    base_layer: nn.Linear
