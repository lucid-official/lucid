"""Helpers the convolution and transposed-convolution modules share."""

from collections.abc import Sequence
from typing import ClassVar, override

from lucid._C import engine as _C_engine
from lucid._factories.creation import empty
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike, _Size2d, _Size3d
import lucid.nn.init as init
from lucid.nn.module import Module
from lucid.nn.modules._lazy import _LazyModule, fill as _fill
from lucid.nn.parameter import Parameter


def _ntuple(v: int | Sequence[int], n: int, name: str) -> tuple[int, ...]:
    """``v`` as ``n`` ints, as the reference reads a convolution's sizes.

    One int, or a sequence of one, is repeated; otherwise the sequence must
    have ``n``.  ``kernel_size`` is the exception: the reference spreads no
    one-element kernel across a 2-D or 3-D window, so it must have ``n``.
    """
    if isinstance(v, int):
        return (v,) * n
    values = tuple(v)
    if len(values) == 1 and name != "kernel_size":
        return values * n
    if len(values) != n:
        if n == 1:
            expected = "a sequence of one"
        elif name == "kernel_size":
            expected = f"{n} ints"
        else:
            expected = f"1 or {n} ints"
        raise ValueError(f"{name} must be an int or {expected}, got {values!r}")
    return values


def _single(v: int | Sequence[int], name: str) -> int:
    (a,) = _ntuple(v, 1, name)
    return a


def _pair(v: _Size2d, name: str) -> tuple[int, int]:
    a, b = _ntuple(v, 2, name)
    return a, b


def _triple(v: _Size3d, name: str) -> tuple[int, int, int]:
    a, b, c = _ntuple(v, 3, name)
    return a, b, c


def _check_groups(in_channels: int | None, out_channels: int, groups: int) -> None:
    """Refuse a ``groups`` the channels cannot be split into.

    ``in_channels`` is ``None`` for a lazy layer that has not seen its input
    yet; its divisibility is checked when it is sized.

    Raises
    ------
    lucid._C.engine.InvalidArgument
        ``groups`` is not positive, or does not divide a channel count; a
        ``ValueError`` when the layer is built, and a ``RuntimeError`` too for
        a lazy layer, which the reference refuses only at its first forward.
    """
    if groups <= 0:
        raise _C_engine.InvalidArgument(f"groups must be a positive integer, got {groups}")
    if in_channels is not None and in_channels % groups != 0:
        raise _C_engine.InvalidArgument(
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
    if out_channels % groups != 0:
        raise _C_engine.InvalidArgument(
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )


def _in_channels(x: Tensor, n: int) -> int:
    """The channel count of a batched or unbatched ``n``-D conv input."""
    if x.ndim not in (n + 1, n + 2):
        raise ValueError(
            f"expected {n + 1}D (unbatched) or {n + 2}D (batched) input, "
            f"got {x.ndim}D input of shape {tuple(x.shape)}"
        )
    return int(x.shape[0 if x.ndim == n + 1 else 1])


def _conv_parameters(
    shape: tuple[int, ...], out_channels: int, bias: bool, device: DeviceLike, dtype: DTypeLike
) -> tuple[Parameter, Parameter | None]:
    """A convolution's ``weight`` of ``shape`` and its bias, allocated and not yet drawn."""
    weight = Parameter(empty(*shape, dtype=dtype, device=device))
    b = Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
    return weight, b


class _ConvParameters(Module):
    """The ``weight`` / ``bias`` of a convolution, and how they are initialised.

    The eager, transposed and lazy convolutions all own the pair, so the
    rule for drawing it lives here once: :meth:`reset_parameters` is the
    default weight-and-bias initialisation, and :meth:`_make_parameters`
    allocates the pair and then calls it.
    """

    out_channels: int
    weight: Parameter
    bias: Parameter | None

    def _make_parameters(
        self, shape: tuple[int, ...], bias: bool, device: DeviceLike, dtype: DTypeLike
    ) -> None:
        """Allocate ``weight`` of ``shape`` and the bias, then draw their values."""
        self.weight, self.bias = _conv_parameters(shape, self.out_channels, bias, device, dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-draw ``weight`` (Kaiming uniform) and ``bias`` (bounded by the weight's fan-in)."""
        init._reset_weight_and_bias(self.weight, self.bias)


class _LazyConvMixin(_LazyModule, _ConvParameters):
    """What the six lazy convolutions share: deferring the weight to the first input.

    Each lazy class puts this before its eager counterpart, sets
    ``_lazy_rank`` (and ``_lazy_transposed`` for a transposed one), and
    calls :meth:`_defer` last in its ``__init__``.  The weight layout is
    ``(out, in // groups, *K)``, or ``(in, out // groups, *K)`` transposed.
    """

    _size_attr: ClassVar[str] = "in_channels"
    _lazy_rank: ClassVar[int]
    _lazy_transposed: ClassVar[bool] = False
    groups: int
    _has_bias: bool

    def _defer(self, bias: bool, device: DeviceLike, dtype: DTypeLike) -> None:
        """Leave ``in_channels`` unknown and the parameters as placeholders."""
        # Through ``Module.__setattr__`` and not ``self.x = ...``: the eager
        # base declares ``in_channels: int`` / ``bias``, and an assignment
        # here would declare them a second time, differently.
        Module.__setattr__(self, "in_channels", None)
        _check_groups(None, self.out_channels, self.groups)
        self._has_bias = bias
        self._device = device
        self._dtype = dtype
        # Placeholders, not ``None``: an optimizer built before the first
        # forward must be handed objects that survive being filled in.
        self.register_parameter("weight", self._placeholder())
        self.register_parameter("bias", self._placeholder() if bias else None)

    def _kernel(self) -> tuple[int, ...]:
        """The kernel's extent along each spatial axis."""
        raise NotImplementedError

    @override
    def _initialize(self, in_channels: int) -> None:
        """Allocate and initialise ``weight`` / ``bias`` for ``in_channels``."""
        _check_groups(in_channels, self.out_channels, self.groups)
        Module.__setattr__(self, "in_channels", in_channels)
        if self._lazy_transposed:
            lead = (in_channels, self.out_channels // self.groups)
        else:
            lead = (self.out_channels, in_channels // self.groups)
        weight, bias = _conv_parameters(
            (*lead, *self._kernel()), self.out_channels, self._has_bias, self._device, self._dtype
        )
        # ``fill`` and not assignment — the placeholder object is what
        # anything that read ``parameters()`` early is holding on to.
        _fill(self, "weight", weight)
        if bias is not None:
            _fill(self, "bias", bias)
        else:
            Module.__setattr__(self, "bias", None)
        self.reset_parameters()

    @override
    def reset_parameters(self) -> None:
        """Re-draw the values; nothing to do until the layer has been sized."""
        if self.in_channels is not None:
            super().reset_parameters()

    @override
    def _extent(self, input: Tensor) -> int:
        """The input's channel axis, whatever the rank (batched or not)."""
        return _in_channels(input, self._lazy_rank)

    @override
    def initialize_parameters(self, input: Tensor, *args: object, **kwargs: object) -> None:
        """Size ``in_channels`` from ``input`` and allocate the weights, if not sized yet.

        Takes the extra positional and keyword arguments the reference's lazy
        convolutions accept, and ignores them, as it does.

        Parameters
        ----------
        input : Tensor
            An input of the shape the layer will see.
        *args, **kwargs
            Accepted for signature parity; unused.
        """
        del args, kwargs
        super().initialize_parameters(input)

    def _sized_by(self, weight: Tensor, prefix: str, error_msgs: list[str]) -> bool:
        """Size the layer from a checkpoint's ``weight``; ``False`` if it cannot be."""
        name = type(self).__name__
        if len(weight.shape) != self._lazy_rank + 2:
            error_msgs.append(
                f"{name} expected {self._lazy_rank + 2}-D weight at '{prefix}weight', "
                f"got {tuple(weight.shape)}"
            )
            return False
        out_axis, expected = (
            (1, self.out_channels // self.groups)
            if self._lazy_transposed
            else (0, self.out_channels)
        )
        if int(weight.shape[out_axis]) != expected:
            error_msgs.append(
                f"{name} out_channels mismatch at '{prefix}weight': "
                f"expected {expected}, got {int(weight.shape[out_axis])}"
            )
            return False
        self._dtype = self._dtype or weight.dtype
        self._device = self._device or weight.device
        in_channels = int(weight.shape[1 - out_axis])
        self._initialize(in_channels if self._lazy_transposed else in_channels * self.groups)
        return True

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Size a still-lazy layer from the checkpoint's weight, then load it."""
        weight: Tensor | None = state_dict.get(f"{prefix}weight")
        if (
            self.in_channels is None
            and weight is not None
            and not self._sized_by(weight, prefix, error_msgs)
        ):
            return
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        padding = vars(self).get("_padding_str") or self.padding
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={padding}"
        )
        # A layer pickled before transposed convolutions had ``padding_mode``
        # has none; it was ``"zeros"``.
        mode = getattr(self, "padding_mode", "zeros")
        if mode != "zeros":
            s += f", padding_mode={mode!r}"
        return s
