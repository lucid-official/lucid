"""A snapshot of one device's autocast state, and the scope that puts it back.

Several things have to run a piece of code under the autocast state some
earlier code ran under: ``checkpoint`` recomputes a segment during backward,
and ``custom_bwd`` runs a ``backward`` after its ``forward``.  Each used to
read the state ad hoc, or not at all, and the recompute silently ran in a
different dtype from the forward.  :class:`AutocastState` is the one owner
of "what the state was" and "run under it again".
"""

from dataclasses import dataclass
from typing import Self, final

from lucid._dtype import dtype
from lucid.amp.autocast import _DEVICE_TYPES, autocast


@final
@dataclass(frozen=True, slots=True)
class AutocastState:
    """The autocast state of one device on the calling thread, at one moment.

    Parameters
    ----------
    device_type : str
        ``"metal"`` or ``"cpu"``.
    enabled : bool
        Whether autocast was on for that device.
    target : lucid.dtype or None
        The dtype it cast to; ``None`` when ``enabled`` is ``False`` (the
        engine keeps no dtype for a device whose autocast is off).
    """

    device_type: str
    enabled: bool
    target: dtype | None

    @classmethod
    def capture(cls, device_type: str) -> Self:
        """Read the state ``device_type`` is in now."""
        return cls(
            device_type,
            autocast.is_autocast_enabled(device_type),
            autocast.get_autocast_dtype(device_type),
        )

    @classmethod
    def capture_all(cls) -> tuple[Self, ...]:
        """Read the state of every device, in :data:`_DEVICE_TYPES` order."""
        return tuple(cls.capture(device_type) for device_type in _DEVICE_TYPES)

    def scope(self) -> autocast:
        """A scope that runs its body in this state — on or off, as captured.

        An off state gives an ``enabled=False`` scope, so a body that was
        recorded with autocast off stays off even when the code replaying it
        runs inside an autocast block.
        """
        return autocast(self.device_type, dtype=self.target, enabled=self.enabled)
