r"""Mixed-adapter batches: each row of one batch goes through the adapter it is named for.

``model(x, adapter_names=["a", "b", "a", "__base__"])`` computes row 0 and 2
with adapter ``a``, row 1 with ``b`` and row 3 with the base model alone,
in one pass.  A serving process holding many adapters can then batch
requests for different adapters together.

The model is not told: while a :class:`_MixedBatch` is open, every slot is
put to its base module (``disable_adapters``) and a forward hook adds,
slot by slot, what each named adapter makes of its own rows to the base
output.  A hook per slot kind is a :class:`_Route`, registered by the slot
type it serves; the routes are the one place that knows how a slot adds an
adapter's contribution to a sub-batch.  Everything the model's own
forward does — dropout aside, which a mixed batch never has (it is
inference only) — is untouched.

The scope matches the layers whose output is a function of one input
tensor: ``nn.Linear``, ``nn.Embedding`` and convolutions, plus the
trained copies of ``modules_to_save``.  A layer that is not (an attention
whose weights are adapted, a raw parameter) refuses the call rather than
silently ignoring its rows' names.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import _ModuleOutput
from lucid.nn.hooks import RemovableHandle
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._slots import _adapter_slots
from lucid.peft.tuners._base import _BaseTunerLayer
from lucid.peft.tuners.lora._layer import _Residual

# The routes are reached through ``_Route.of``, by the slot type each registers for.
__all__ = [
    "BASE_ADAPTER",
    "_CopyRoute",
    "_MixedBatch",
    "_ResidualRoute",
    "_Route",
    "_UnroutableRoute",
]

#: The name that stands for "no adapter": that row runs the base model alone.
BASE_ADAPTER: Final = "__base__"


def _rows_by_adapter(names: Sequence[str]) -> dict[str, list[int]]:
    """``names`` inverted: the row indices each adapter is asked for, in order."""
    rows: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        rows[name].append(index)
    return dict(rows)


class _Route(ABC):
    """How one kind of slot answers a mixed batch; registered by the slot type it serves.

    A route is built around one slot.  ``check`` refuses a slot that
    cannot answer; ``apply`` takes the slot's output for the base module
    and returns it with every named adapter's rows replaced or updated.
    Routes are tried in the order they are defined.
    """

    _routes: ClassVar[list[type[_Route]]] = []
    _slot_type: ClassVar[type[nn.Module]]
    slot: _BaseTunerLayer | ModulesToSaveWrapper

    @override
    def __init_subclass__(
        cls, *, slot_type: type[nn.Module] | None = None, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)
        if slot_type is not None:
            cls._slot_type = slot_type
            cls._routes.append(cls)

    @classmethod
    def of(cls, slot: object) -> _Route | None:
        """The route for ``slot``, or ``None`` if it is not a slot a route serves."""
        if not isinstance(slot, _BaseTunerLayer | ModulesToSaveWrapper):
            return None
        for route in cls._routes:
            if isinstance(slot, route._slot_type):
                return route(slot)
        return None

    def __init__(self, slot: _BaseTunerLayer | ModulesToSaveWrapper) -> None:
        self.slot = slot

    @abstractmethod
    def check(self, named: frozenset[str]) -> None:
        """Raise if this slot cannot answer a mixed batch that names the adapters ``named``."""

    @abstractmethod
    def apply(self, x: Tensor, output: Tensor, rows: dict[str, list[int]]) -> Tensor:
        """``output``, the base module's, with each adapter's ``rows`` of ``x`` routed through it."""


def _index(rows: list[int], like: Tensor) -> Tensor:
    return lucid.tensor(rows, dtype=lucid.int64, device=like.device)


class _ResidualRoute(_Route, slot_type=_Residual):
    r"""A LoRA layer adds each adapter's low-rank update to its own rows: :math:`y_i \mathrel{+}= u_a(x_i)`."""

    slot: _Residual

    @override
    def check(self, named: frozenset[str]) -> None:
        if self.slot.merged:
            raise ValueError(
                "Cannot pass `adapter_names` when there are merged adapters, please call "
                "`unmerge_adapter` first."
            )
        if any(self.slot.use_dora.get(name, False) for name in named):
            raise ValueError("Cannot pass `adapter_names` when DoRA is enabled.")

    @override
    def apply(self, x: Tensor, output: Tensor, rows: dict[str, list[int]]) -> Tensor:
        layer = self.slot
        held = layer._adapter_names()
        for adapter, indices in rows.items():
            if adapter == BASE_ADAPTER or adapter not in held:
                continue
            index = _index(indices, x)
            update = layer._lowrank(adapter, layer._dropped(adapter, x.index_select(0, index)))
            output = output.index_add(0, index, update.to(output.dtype))
        return output


class _CopyRoute(_Route, slot_type=ModulesToSaveWrapper):
    """A ``modules_to_save`` module runs the named adapter's trained copy on its rows."""

    slot: ModulesToSaveWrapper

    @override
    def check(self, named: frozenset[str]) -> None:
        """Always answers (a mixed batch of tensors)."""
        del named

    @override
    def apply(self, x: Tensor, output: Tensor, rows: dict[str, list[int]]) -> Tensor:
        copies = self.slot.modules_to_save
        for adapter, indices in rows.items():
            if adapter == BASE_ADAPTER or adapter not in copies:
                continue
            index = _index(indices, x)
            trained = copies[adapter](x.index_select(0, index))
            if not isinstance(trained, Tensor):
                raise TypeError(
                    f"`adapter_names` needs a modules_to_save module that returns a Tensor, "
                    f"got {type(trained).__name__}"
                )
            output = output.index_copy(0, index, trained)
        return output


class _UnroutableRoute(_Route, slot_type=_BaseTunerLayer):
    """A tuner layer whose output is not one tensor of rows (attention, a raw parameter): refuses."""

    @override
    def check(self, named: frozenset[str]) -> None:
        del named
        raise NotImplementedError(
            f"`adapter_names` routes nn.Linear, nn.Embedding and convolution layers; "
            f"{type(self.slot).__name__} is not one of them"
        )

    @override
    def apply(self, x: Tensor, output: Tensor, rows: dict[str, list[int]]) -> Tensor:
        raise NotImplementedError("unreachable: check() refuses this slot first")


class _MixedBatch(AbstractContextManager[None]):
    """While open, a forward pass sends each batch row through the adapter it is named for.

    Entering checks the call as a whole (training mode, unknown names,
    every slot), puts every slot to its base module and hooks each to add
    its routed contribution; leaving removes the hooks and restores each
    slot's ``disable_adapters`` — also on an exception.  Slots inside the
    model's encoder, if it has one, keep the names as given where the rest
    take them repeated ``num_beams`` times: beam search repeats the decoder's
    rows but not the encoder's.

    Parameters
    ----------
    model : nn.Module
        The adapted model whose slots are routed.
    adapter_names : list of str
        One adapter per row; ``"__base__"`` for none.
    known : Sequence of str
        The adapters the model holds.
    num_beams : int, default=1
        Beam width of a ``generate`` call; each name is repeated this often.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_names: Sequence[str],
        known: Sequence[str],
        num_beams: int = 1,
    ) -> None:
        if model.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")
        unknown = {name for name in adapter_names if name != BASE_ADAPTER} - set(known)
        if unknown:
            raise ValueError(
                f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unknown))}"
            )
        self._model = model
        self._plain = list(adapter_names)
        self._beamed = [name for name in adapter_names for _ in range(num_beams)]
        self._handles: list[RemovableHandle] = []
        self._restore: list[tuple[_BaseTunerLayer | ModulesToSaveWrapper, bool]] = []

    def _encoder_slots(self) -> set[int]:
        """The ids of the slots under the model's encoder (none if it has no ``get_encoder``)."""
        get_encoder = getattr(self._model, "get_encoder", None)
        if get_encoder is None:
            return set()
        encoder = get_encoder()
        return {id(m) for m in encoder.modules()} if isinstance(encoder, nn.Module) else set()

    def _hook(self, route: _Route, names: Sequence[str], was_off: bool) -> None:
        """Add the forward hook of ``route`` that routes a batch named ``names``.

        ``was_off``: the slot's adapters were already off (``disable_adapter()``); it stays base-only.
        """
        rows = _rows_by_adapter(names)
        slot = route.slot

        def routed(
            module: object, args: tuple[Tensor, ...], output: _ModuleOutput
        ) -> _ModuleOutput:
            del module  # the slot this hook was made for
            if was_off or not isinstance(output, Tensor):
                # Adapters off before the batch, or not a tensor of rows: nothing to route.
                return output
            x = args[0]
            if len(x) != len(names):
                raise ValueError(
                    "Length of `adapter_names` should be the same as the number of inputs, but "
                    f"got {len(names)} and {len(x)} respectively."
                )
            return route.apply(x, output, rows)

        self._handles.append(slot.register_forward_hook(routed))

    @override
    def __enter__(self) -> None:
        in_encoder = self._encoder_slots()
        named = frozenset(self._plain) - {BASE_ADAPTER}
        routes: list[_Route] = []
        for _, slot in _adapter_slots(self._model):
            route = _Route.of(slot)
            if route is None:
                continue
            route.check(named)
            routes.append(route)
        for route in routes:
            slot = route.slot
            names = self._plain if id(slot) in in_encoder else self._beamed
            was_off = slot.disable_adapters
            self._restore.append((slot, was_off))
            self._hook(route, names, was_off)
            slot.disable_adapters = True

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb  # an exception passes through; the slots come back regardless
        for handle in self._handles:
            handle.remove()
        for slot, disabled in self._restore:
            slot.disable_adapters = disabled
        self._handles.clear()
        self._restore.clear()
