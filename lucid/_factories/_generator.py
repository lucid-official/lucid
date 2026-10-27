r"""``lucid.Generator``: one Philox-4×32 stream, with the reference's methods.

``lucid.Generator`` used to be the engine class itself, so the methods
reference code reaches for — ``Generator().manual_seed(0)``,
``initial_seed()``, ``get_state()`` / ``set_state()`` — raised
``AttributeError`` (LCD-374).  This subclass adds them; the engine reads
the stream exactly as before, since every instance still *is* an engine
generator.

The state of a stream is ``(key, counter)``: Philox is counter-based, so
the pair determines every draw that follows.  :func:`lucid.get_rng_state`
and :meth:`Generator.get_state` both pack it as a length-2 ``int64`` tensor,
through the helpers here.
"""

import operator
import os
import weakref
from typing import TYPE_CHECKING, Self, SupportsIndex, override

from lucid._C import engine as _C_engine
from lucid._device import device as _device_cls
from lucid._dtype import int64
from lucid._engine_scope import untraced
from lucid._factories.converters import tensor
from lucid._types import DeviceLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_UNSIGNED_64 = 1 << 64
_SIGNED_64 = 1 << 63

#: The engine's own counter descriptor.  The engine stub declares it
#: read-only; the binding writes it, which restoring a state needs.
_ENGINE_COUNTER = _C_engine.Generator.__dict__["counter"]


def _seed_value(seed: object, op: str) -> int:
    """``seed`` as the engine's unsigned 64-bit key.

    The reference takes any 64-bit integer, signed or unsigned, and reads a
    negative one modulo :math:`2^{64}`; the engine refused it with a raw
    "incompatible constructor arguments".  A float or bool is refused.
    """
    if isinstance(seed, bool) or not isinstance(seed, SupportsIndex):
        raise TypeError(f"{op}: seed must be an int, got {type(seed).__name__}")
    value = operator.index(seed)
    if not -_SIGNED_64 <= value < _UNSIGNED_64:
        raise ValueError(f"{op}: seed must lie in [-2**63, 2**64), got {value}")
    return value % _UNSIGNED_64


def _key(g: _C_engine.Generator) -> int:
    """The key of ``g``, a :class:`Generator` or the engine's own singleton."""
    if isinstance(g, Generator):
        return g.initial_seed()
    return int(g.seed)


def _set_counter(g: _C_engine.Generator, counter: int) -> None:
    """Place ``g`` at block ``counter`` of its stream."""
    _ENGINE_COUNTER.__set__(g, counter % _UNSIGNED_64)


def _state_tensor(g: _C_engine.Generator) -> Tensor:
    """``[key, counter]`` as an ``int64`` tensor, each in two's complement."""
    words = [_key(g), int(g.counter)]
    with untraced():
        return tensor(
            [w - _UNSIGNED_64 if w >= _SIGNED_64 else w for w in words], dtype=int64
        )


def _unpack_state(state: Tensor, op: str) -> tuple[int, int]:
    """``(key, counter)`` from a state tensor :func:`_state_tensor` made."""
    if int(state.numel()) != 2:
        raise ValueError(
            f"{op}: expected a length-2 state tensor, got shape={tuple(state.shape)}"
        )
    with untraced():
        flat = state.reshape(-1)
        return int(flat[0].item()) % _UNSIGNED_64, int(flat[1].item()) % _UNSIGNED_64


class Generator(_C_engine.Generator):
    r"""A Philox-4×32 random stream of its own.

    Pass one as ``generator=`` to a random op to draw from it instead of the
    default stream that :func:`lucid.manual_seed` controls; draws through it
    leave the default stream where it was.

    Parameters
    ----------
    seed : int or str or lucid.device, optional
        The key, any integer in :math:`[-2^{63}, 2^{64})` — a negative one
        is read modulo :math:`2^{64}`, as the reference reads it.  When a
        device specifier is given as the single positional argument
        (``Generator("cpu")``), it is read as ``device`` and the key
        defaults to ``0``.
    device : str or lucid.device, optional
        The device recorded on :attr:`device`.  Defaults to ``device("cpu")``.

    Raises
    ------
    TypeError
        If ``seed`` is not an integer or device specifier.
    ValueError
        If ``seed`` is out of the 64-bit range or ``device`` names no
        supported device.

    Notes
    -----
    The constructor accepts both ``Generator(seed)`` and the reference's
    ``Generator(device=...)`` / ``Generator("cpu")``.  :meth:`seed` reseeds
    from entropy, as the reference's does; :meth:`initial_seed` reads the
    key.

    Examples
    --------
    >>> import lucid
    >>> g = lucid.Generator().manual_seed(7)
    >>> g.initial_seed()
    7
    >>> a = lucid.rand(3, generator=g)
    >>> state = g.get_state()
    >>> b = lucid.rand(3, generator=g)
    >>> c = lucid.rand(3, generator=g.set_state(state))
    >>> bool((b == c).all())
    True
    """

    _device: _device_cls

    def __init__(
        self,
        seed: int | DeviceLike = 0,
        device: DeviceLike = None,
    ) -> None:
        if isinstance(seed, (str, _device_cls, _C_engine.Device)):
            if device is not None:
                raise TypeError("Generator() got multiple values for argument 'device'")
            device, seed = seed, 0
        self._device = _device_cls("cpu" if device is None else device)
        super().__init__(_seed_value(seed, "Generator"))
        _LIVE.add(self)

    @property
    def device(self) -> _device_cls:
        """The device recorded when this generator was constructed."""
        return self._device

    def manual_seed(self, seed: int) -> Self:
        """Set the key to ``seed`` and rewind the stream to its start.

        Parameters
        ----------
        seed : int
            The new key, as for the constructor.

        Returns
        -------
        Generator
            This generator, so a call can be chained.
        """
        self.set_seed(_seed_value(seed, "manual_seed"))
        return self

    def initial_seed(self) -> int:
        """The key this stream was last seeded with.

        Returns
        -------
        int
            The key, in :math:`[0, 2^{64})`.
        """
        return int(super().seed)

    def get_state(self) -> Tensor:
        """The stream's position, ``[key, counter]``, as an ``int64`` tensor.

        Returns
        -------
        Tensor
            Shape ``(2,)``; :meth:`set_state` returns the stream to it.
        """
        return _state_tensor(self)

    def set_state(self, new_state: Tensor) -> Self:
        """Return the stream to a position :meth:`get_state` captured.

        Parameters
        ----------
        new_state : Tensor
            A length-2 ``int64`` tensor ``[key, counter]``.

        Returns
        -------
        Generator
            This generator.

        Raises
        ------
        ValueError
            If ``new_state`` does not hold exactly two elements.
        """
        key, counter = _unpack_state(new_state, "set_state")
        self.set_seed(key)
        _set_counter(self, counter)
        return self

    @override
    def seed(self) -> int:  # type: ignore[override]  # the reference's method replaces the engine's key property
        """Reseed the stream from the OS entropy source and return the new key.

        Returns
        -------
        int
            The key, in :math:`[0, 2^{63})`; :meth:`manual_seed` with it
            replays the stream from here.
        """
        key = int.from_bytes(os.urandom(8), "little") & (_SIGNED_64 - 1)
        self.manual_seed(key)
        return key

    #: How many Philox blocks the stream has drawn since it was seeded —
    #: the engine's own property, named here so it stays this class's
    #: member as it was when ``lucid.Generator`` was the engine class.
    counter = _ENGINE_COUNTER


#: Every :class:`Generator` alive, so a compile can return each stream to
#: where it was (:func:`_stream_states`).
_LIVE: weakref.WeakSet[Generator] = weakref.WeakSet()

#: One stream's position: the stream, its key, its counter.
type _StreamState = tuple[_C_engine.Generator, int, int]


def _stream_states() -> tuple[_StreamState, ...]:
    """The position of the default stream and of every live :class:`Generator`.

    Read from the streams' own fields, with no tensor op, so it is safe
    inside a compile recording.  :func:`_restore_stream_states` puts them
    back: a compile that records a call drawing random values advances
    the streams, and the call's first real run must draw what an eager
    call would have.

    Returns
    -------
    tuple of (Generator, int, int)
        One ``(stream, key, counter)`` per stream, the default one first.
    """
    streams = (_C_engine.default_generator(), *_LIVE)
    return tuple((g, _key(g), int(g.counter)) for g in streams)


def _restore_stream_states(states: tuple[_StreamState, ...]) -> None:
    """Return each stream to the position :func:`_stream_states` read.

    Parameters
    ----------
    states : tuple of (Generator, int, int)
        What :func:`_stream_states` returned.  A :class:`Generator` made
        since is left where it is.
    """
    for g, key, counter in states:
        g.set_seed(key)
        _set_counter(g, counter)
