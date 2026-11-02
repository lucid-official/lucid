"""``PolyModel``: the Poly tuner — it carries the task of each batch row to the layers; the rest is the generic tuner's."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.nn.hooks import RemovableHandle
from lucid.peft._config import PeftType, _TargetedConfig
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.poly._config import PolyConfig
from lucid.peft.tuners.poly._layer import PolyLayer


class PolyModel(_ConfigTuner, peft_type=PeftType.POLY):
    r"""A model with Poly layers in it, and the tools to take them out again.

    Every module ``config`` selects is wrapped in a :class:`~lucid.peft.tuners.poly.Linear`
    holding ``n_skills`` low-rank adapters and a router; the rest of the model is frozen.  The
    model is changed in place and kept as ``model``; a public name this wrapper lacks is read
    from it.  ``peft_config`` holds a copy of ``config`` with the default ``target_modules``
    filled in.

    The router needs to know the task of each batch row: pass ``task_ids`` (``(batch,)``
    integers in ``[0, n_tasks)``) to the call — ``PeftModel.forward(..., task_ids=)``,
    ``generate(..., task_ids=)`` or this tuner's own ``forward`` — and every Poly layer of the
    model receives it for that call.  Adapters add and switch as for any tuner; none merges.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    peft_config : PolyConfig
        What to adapt and how.
    adapter_name : str
        Name of the adapter, the key in every ``poly_lora_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match is not an ``nn.Linear``, or ``n_splits`` does
        not divide a matched layer's sizes.
    NotImplementedError
        The model already holds adapter layers of another method.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import PolyConfig, PolyModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> poly = PolyModel(net, PolyConfig(r=2, n_tasks=3, target_modules=["0"]), "default")
    >>> poly.targeted_module_names
    ['0']
    >>> poly(lucid.randn(4, 8), task_ids=lucid.tensor([0, 1, 2, 0])).shape
    (4, 2)
    """

    _layer_types = PolyLayer._layer_types
    _config_class: ClassVar[type[_TargetedConfig]] = PolyConfig

    @override
    @contextmanager
    def _task_scope(self, task_ids: Tensor | None) -> Iterator[None]:
        """Within the block, every Poly layer is called with ``task_ids`` (nothing changes without them)."""
        if task_ids is None:
            yield
            return

        def give(
            _: object, args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            return args, {**kwargs, "task_ids": task_ids}

        handles: list[RemovableHandle] = [
            layer.register_forward_pre_hook(give, with_kwargs=True)
            for _, layer in _tuner_layers(self.model)
            if isinstance(layer, PolyLayer)
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @override
    def forward(  # type: ignore[override]  # the reference names task_ids, which the base call takes by keyword
        self, *args: object, task_ids: Tensor | None = None, **kwargs: object
    ) -> object:
        """Call the model; ``task_ids``, if given, goes to every Poly layer for the call."""
        if task_ids is not None and not isinstance(task_ids, Tensor):
            raise TypeError(f"task_ids must be a Tensor, got {type(task_ids).__name__}")
        with self._task_scope(task_ids):
            return self.model(*args, **kwargs)
