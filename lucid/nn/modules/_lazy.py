"""What every lazy layer shares: placeholder parameters, and where they will be made.

A lazy layer cannot build its weights until it sees an input.  What it
*can* do is exist: registering an
:class:`~lucid.nn.parameter.UninitializedParameter` up front means that
anything reading ``parameters()`` before the first forward — an
optimizer, an EMA, a parameter-server shard — is handed the objects the
real weights will later occupy, instead of an empty list it keeps
forever.

``fill`` is what turns one into the other.  It exists as a shared helper
because thirteen lazy classes across three files have to do exactly the
same thing, and the failure mode of each of them doing it separately is
that one of them assigns instead of materializing and quietly reverts to
the old behaviour for that layer alone.

:class:`_LazyModule` owns the other half: the device and dtype the layer
will materialise on.  ``Module.to`` before the first forward has nothing
to move but the placeholders, so the layer records the move itself, or
it would build its weights where it was constructed rather than where it
was sent.
"""

from collections.abc import Callable
from typing import ClassVar, override, Self

from lucid._factories.creation import empty
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter, UninitializedParameter


def fill(module: Module, name: str, value: Parameter) -> None:
    """Give ``module.<name>`` its real value without replacing the object.

    Parameters
    ----------
    module : Module
        The lazy layer being initialised.
    name : str
        Which registered parameter to fill.
    value : Parameter
        The real parameter — correct shape, already initialised.

    Notes
    -----
    Falls back to plain assignment when the slot does not hold a
    placeholder.  That happens on the second materialisation of a layer
    loaded from a checkpoint and then run, and there is nothing to
    preserve in that case; assigning is right.
    """
    current = module._parameters.get(name)
    if isinstance(current, UninitializedParameter):
        current.materialize(value)
    else:
        setattr(module, name, value)


class _LazyModule(Module):
    """The device and dtype a lazy layer will make its parameters on.

    ``_device`` / ``_dtype`` start as the constructor's and follow every
    conversion — ``to``, ``metal``, ``half``, ``type`` — made while the
    layer still waits for its first input, as the reference's lazy layers
    follow theirs.  Afterwards the real tensors carry it.
    """

    _device: DeviceLike
    _dtype: DTypeLike

    #: The attribute that stays ``None`` until the layer is sized
    #: (``in_features`` / ``in_channels`` / ``num_features``).
    _size_attr: ClassVar[str]
    #: The axis of an input that carries that size; the channel axis by default.
    _size_axis: ClassVar[int] = 1
    #: Allocate and initialise the parameters for a layer ``size`` wide.  A
    #: declaration only: each family already owns its ``_initialize``, and
    #: defining a stub here would shadow the one a mixin inherits behind it.
    _initialize: Callable[[int], None]

    def _extent(self, input: Tensor) -> int:
        """The extent of the axis of ``input`` that sizes this layer."""
        return int(input.shape[self._size_axis])

    def initialize_parameters(self, input: Tensor) -> None:
        """Size the layer from ``input`` and allocate its parameters, if it has not been yet.

        The step the first forward takes, exposed so a model can be sized
        without running it.  Does nothing once the layer is sized.

        Parameters
        ----------
        input : Tensor
            An input of the shape the layer will see; only the axis that
            carries the layer's size is read.
        """
        if getattr(self, self._size_attr) is None:
            self._initialize(self._extent(input))

    def _placeholder(self) -> UninitializedParameter:
        """A parameter slot to fill later, on the device and dtype it will have."""
        return UninitializedParameter(empty(0, dtype=self._dtype, device=self._device))

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Convert the tensors, and record the conversion while still pending."""
        if self.has_uninitialized_parameters(recurse=False):
            # What ``fn`` makes of an empty tensor on the target is what it
            # would make of the weights: only what changed is recorded, so
            # an unset dtype stays unset across a move.
            before = empty(0, dtype=self._dtype, device=self._device)
            after = fn(before)
            if after.device != before.device:
                self._device = after.device
            if after.dtype != before.dtype:
                self._dtype = after.dtype
        return super()._apply(fn)


__all__ = ["fill"]
