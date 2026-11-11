r"""The Poly layers: a frozen linear layer beside a task-routed mixture of low-rank skills.

``PolyLayer`` is the family root.  An adapter holds ``poly_lora_A`` (``(n_splits, n_skills,
in / n_splits, r)``) and ``poly_lora_B`` (``(n_splits, n_skills, r, out / n_splits)``) as bare
parameters, and a router in ``poly_router`` (one logit vector per task).  The router turns each batch
row's task into skill weights; the row's factors are the weighted sums of the skills' factors, split
by split, and the layer adds :math:`\tfrac{1}{r}\,x\,A_{\text{row}}B_{\text{row}}` to the base output.
A state dict reads ``....poly_lora_A.default`` and an adapter file ``....poly_lora_A``, the
reference's key.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _config_as,
    _ConfigLayer,
    _LinearSized,
    _SpecField,
    _Unmergeable,
)
from lucid.peft.tuners.poly._config import PolyConfig
from lucid.peft.tuners.poly._router import _Router


class _PolySpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int
    n_tasks: int
    n_skills: int
    n_splits: int
    poly_type: str


def _mixed(weights: Tensor, factors: Tensor) -> Tensor:
    """``(batch, split, skill)`` weights applied to ``(split, skill, m, n)`` factors: ``(batch, split, m, n)``."""
    splits, skills, rows, cols = (int(n) for n in factors.shape)
    batch = int(weights.shape[0])
    stacked = factors.reshape(splits, skills, rows * cols)
    mixed = weights.permute(1, 0, 2) @ stacked  # (split, batch, rows * cols)
    return mixed.permute(1, 0, 2).reshape(batch, splits, rows, cols)


class PolyLayer(_LinearSized, _ConfigLayer, _Unmergeable, family=True):
    r"""Adapter bookkeeping shared by the Poly layers.

    Holds the wrapped module as ``base_layer`` — the very module, so its parameters stay the same
    ``Parameter`` objects — and, per adapter name, the skill factors and a router.  The forward pass
    adds the update of every active adapter this layer holds, for the tasks ``task_ids`` names.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    config : PolyConfig
        The adapter's config.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``n_splits`` does not divide the layer's input or output size, or ``base_layer`` builds
        parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("poly_lora_A", "poly_lora_B", "poly_router")
    _repr_prefix = "poly."
    _unmergeable_because = "a Poly layer, which mixes its skills per task"
    poly_lora_A: nn.ParameterDict
    poly_lora_B: nn.ParameterDict
    poly_router: nn.ModuleDict[_Router]

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter's skills, by name (read-only).")
    n_tasks = _SpecField[int]("Tasks each adapter's router has logits for, by name (read-only).")
    n_skills = _SpecField[int]("Skills of each adapter, by name (read-only).")
    n_splits = _SpecField[int]("Blocks each adapter's skills are split into, by name (read-only).")
    poly_type = _SpecField[str]("The router type of each adapter, by name (read-only).")

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        del config  # nothing here depends on it
        # The factors are bare parameters, the routers modules: the one table of a different kind.
        self.poly_router = nn.ModuleDict()

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(PolyConfig, config)
        for size, what in ((self.in_features, "input"), (self.out_features, "output")):
            if size % cfg.n_splits:
                raise ValueError(
                    f"n_splits={cfg.n_splits} must divide the layer's {what} size {size}"
                )
        anchor = self.weight
        dtype = _adapter_dtype(anchor)
        shape_a = (cfg.n_splits, cfg.n_skills, self.in_features // cfg.n_splits, cfg.r)
        shape_b = (cfg.n_splits, cfg.n_skills, cfg.r, self.out_features // cfg.n_splits)
        a = Parameter(lucid.empty(*shape_a, device=anchor.device, dtype=dtype))
        b = Parameter(lucid.empty(*shape_b, device=anchor.device, dtype=dtype))
        router = _Router.of(cfg).to(device=anchor.device, dtype=dtype)
        self._reset(a, b, router, cfg)
        self._specs[adapter_name] = _PolySpec(
            cfg.r, cfg.n_tasks, cfg.n_skills, cfg.n_splits, cfg.poly_type
        )
        self.poly_lora_A[adapter_name] = a
        self.poly_lora_B[adapter_name] = b
        self.poly_router[adapter_name] = router

    @staticmethod
    def _reset(a: Parameter, b: Parameter, router: _Router, config: PolyConfig) -> None:
        """Start ``A`` like a linear layer's weight, ``B`` at zero (or alike), the router near-uniform."""
        skills = config.n_splits * config.n_skills
        with lucid.no_grad():
            # Each (split, skill) factor is drawn as a linear layer's weight would be (fan-in the
            # feature size), then laid out feature-major.
            width = int(a.shape[2])
            weight = lucid.empty(skills * config.r, width, device=a.device, dtype=a.dtype)
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            a.copy_(weight.reshape(skills, config.r, width).permute(0, 2, 1).reshape(a.shape))
            if config.init_weights:
                init.zeros_(b)
            else:
                width = int(b.shape[3])
                weight = lucid.empty(skills * width, config.r, device=b.device, dtype=b.dtype)
                init.kaiming_uniform_(weight, a=math.sqrt(5))
                b.copy_(weight.reshape(skills, width, config.r).permute(0, 2, 1).reshape(b.shape))
            router.reset()

    def _spec(self, adapter: str) -> _PolySpec:
        spec = self._specs[adapter]
        assert isinstance(spec, _PolySpec)  # ``_enter`` is the only writer
        return spec

    def _update(self, adapter: str, x: Tensor, task_ids: Tensor | None) -> Tensor:
        """The adapter's addition to the base output for the input ``x`` (``(batch, ..., in)``)."""
        spec = self._spec(adapter)
        weights = self.poly_router[adapter](task_ids)
        batch = int(x.shape[0])
        if int(weights.shape[0]) != batch:
            raise ValueError(
                f"task_ids names {int(weights.shape[0])} tasks for a batch of {batch} rows"
            )
        a = _mixed(weights, _parameter_at(self.poly_lora_A, adapter))
        b = _mixed(weights, _parameter_at(self.poly_lora_B, adapter))
        a = a.reshape(batch, self.in_features, spec.r)  # splits laid end to end along the features
        b = b.transpose(1, 2).reshape(batch, spec.r, self.out_features)
        rows = x.to(a.dtype).reshape(batch, -1, self.in_features)
        return (rows @ a @ b / spec.r).reshape(*x.shape[:-1], self.out_features)

    @override
    def forward(self, x: Tensor, task_ids: Tensor | None = None) -> Tensor:
        """The base output plus each active adapter's mixture of skills for the tasks ``task_ids``.

        ``x`` is ``(batch, ..., in)`` and ``task_ids`` ``(batch,)`` integers, one task per row.
        The base output alone when adapters are disabled.

        Raises
        ------
        ValueError
            An active adapter applies and ``task_ids`` is missing, names a task the router lacks,
            or has another length than the batch.
        """
        result: Tensor = self.base_layer(x)
        if self.disable_adapters:
            return result
        out = result
        for adapter in self.active_adapters:
            if adapter in self._specs:  # an adapter aimed at other layers: this one runs its base
                out = out + self._update(adapter, x, task_ids).to(result.dtype)
        return out


class Linear(PolyLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a task-routed mixture of low-rank skills.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``poly_lora_A`` and the other tables.
    config : PolyConfig
        The adapter's config.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PolyConfig
    >>> from lucid.peft.tuners.poly import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", PolyConfig(r=2, n_tasks=2, target_modules=["x"]))
    >>> x = lucid.randn(3, 5, 8)
    >>> out = layer(x, task_ids=lucid.tensor([0, 1, 0]))
    >>> bool((out == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear
