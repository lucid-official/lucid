"""Lucid device descriptors.

Defines :class:`device` — the runtime handle that pins a tensor to either
the CPU stream (Apple Accelerate) or the Metal GPU stream (MLX). Only
``"cpu"`` and ``"metal"`` are valid device types; Lucid is Apple Silicon
only.
"""

from typing import override

from lucid._C import engine as _C_engine


class device:
    """
    Lucid device descriptor.
    device("metal") → Apple GPU (Metal)
    device("cpu")   → CPU (Apple Accelerate)
    """

    _METAL_TYPES: frozenset[str] = frozenset({"metal"})
    _VALID_TYPES: frozenset[str] = frozenset({"metal", "cpu"})

    type: str
    index: int

    def __init__(self, type_or_str: device | _C_engine.Device | str, index: int = 0) -> None:
        """Parse a device specifier.

        Parameters
        ----------
        type_or_str : device, lucid._C.engine.Device, or str
            Either an existing :class:`device` (copied, as the reference's
            device constructor copies one), the engine's own ``Device``
            enum, a bare device type (``"cpu"`` / ``"metal"``), or a
            ``"<type>:<index>"`` string (e.g. ``"metal:0"``).
        index : int, optional
            Device index — only ``0`` is meaningful on current Apple
            Silicon hardware. Read only when ``type_or_str`` is a bare
            type string without its own index suffix; a :class:`device`
            or engine ``Device`` carries its own index instead.

        Raises
        ------
        ValueError
            If the device type is not ``"cpu"`` or ``"metal"``.

        Examples
        --------
        >>> import lucid
        >>> lucid.device("cpu")
        device('cpu')
        >>> lucid.device("metal:0").is_metal
        True
        >>> lucid.device(lucid.device("metal")) == lucid.device("metal")
        True
        """
        if isinstance(type_or_str, device):
            self.type = type_or_str.type
            self.index = type_or_str.index
        elif isinstance(type_or_str, _C_engine.Device):
            self.type = "metal" if type_or_str == _C_engine.Device.GPU else "cpu"
            self.index = 0
        elif isinstance(type_or_str, str):
            if ":" in type_or_str:
                t, idx = type_or_str.split(":", 1)
                self.type = t.strip()
                self.index = int(idx)
            else:
                self.type = type_or_str.strip()
                self.index = index
        else:
            raise TypeError(
                "lucid.device(): expected a device, lucid._C.engine.Device, "
                f"or str, got {type(type_or_str).__name__}"
            )
        if self.type not in self._VALID_TYPES:
            raise ValueError(f"Unknown device '{self.type}'. Use 'metal' or 'cpu'.")
        self._engine: _C_engine.Device = (
            _C_engine.Device.GPU if self.type in self._METAL_TYPES else _C_engine.Device.CPU
        )

    @property
    def is_metal(self) -> bool:
        """``True`` if this device targets the Metal GPU stream."""
        return self.type in self._METAL_TYPES

    @override
    def __repr__(self) -> str:
        """Return ``"device('<type>')"``."""
        return f"device('{self.type}')"

    @override
    def __eq__(self, other: object) -> bool:
        """Compare against another :class:`device` or a device-type string.

        Two devices are equal iff they map to the same engine stream
        (``CPU`` or ``GPU``).
        """
        if isinstance(other, str):
            return other == self.type
        if isinstance(other, device):
            return self._engine == other._engine
        return NotImplemented

    @override
    def __hash__(self) -> int:
        """Hash by engine stream so equal devices hash equal."""
        return hash(self._engine)


def _device_from_engine(d: _C_engine.Device) -> device:
    """Convert engine Device enum → lucid device."""
    return device(d)
