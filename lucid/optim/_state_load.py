"""Loading a checkpoint into an optimizer: the payload, and how its state reaches the engines.

``Optimizer.load_state_dict`` is a boundary: a checkpoint written by another
run, process or machine comes in, and the optimizer must take all of it or
none of it.  What can be decided without touching the optimizer — the payload's
shape, whether it fits the optimizer's groups, what each saved entry becomes
for its parameter — is decided here, so a malformed checkpoint is refused
before anything is staged, let alone installed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter

from lucid._C import engine as _C_engine

# Per-parameter scalars the engine reads as numbers wherever they live, so a
# loaded one follows neither the parameter's dtype nor its device: ``step`` is
# a count, and the others are rounded to the parameter's precision by the engine.
_SCALAR_STATE = frozenset({"step", "eta", "mu", "mu_product"})


class _StateLoader(Protocol):
    """The part of an engine optimizer a checkpoint is loaded through."""

    step_count: int

    def load_state_buffers(
        self, bufs: Sequence[tuple[str, Sequence[_C_engine.TensorImpl | None]]]
    ) -> None: ...


def _follow(src: _C_engine.TensorImpl, like: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """``src`` in ``like``'s dtype and on its device; ``src`` itself when it already is.

    The one rule by which optimizer state follows its parameter through a
    conversion (``module.to()``, ``.double()``, ``.half()``) or a checkpoint
    saved in another precision.  Engine optimizers apply it when their engine
    is rebuilt and when a checkpoint is loaded; optimizers that keep their
    state in Python apply it before they read that state in ``step``.
    """
    out = src
    if out.dtype != like.dtype:
        out = _C_engine.astype(out, like.dtype)
    if out.device != like.device:
        out = out.transfer_to_device(like.device, False)
    return out


def _loaded_state_like(
    name: str, value: object, like: _C_engine.TensorImpl
) -> _C_engine.TensorImpl:
    """A checkpointed state entry ``name`` as the engine takes it for parameter ``like``.

    A buffer follows its parameter (:func:`_follow`), whatever it was saved
    as: a checkpoint written in float64 loads into a float32 run, and one
    written in float32 into a half-precision run.  A host array is built on
    the host and converted there, because a float64 array cannot be built on
    the GPU.  A scalar (``step``, ``eta``, ``mu``, ``mu_product``) stays as it
    was saved.
    """
    if isinstance(value, Tensor):
        src = value._impl
    else:
        src = _C_engine.TensorImpl(value, _C_engine.Device.CPU, False)
    return src if name in _SCALAR_STATE else _follow(src, like)


def _load_group_state(
    engine: _StateLoader, params: list[Parameter], first: int, state: dict[int, dict[str, object]]
) -> None:
    """Give ``engine`` the saved state of its group's parameters, ``first`` being the first's index.

    Every entry, ``step`` included, goes to its own parameter through
    ``load_state_buffers``; a parameter without an entry keeps the fresh
    state ``engine`` was built with.  A ``step`` that is a plain number is
    the older layout, one count for the whole group, and is set as that
    group's count.
    """
    by_name: dict[str, list[_C_engine.TensorImpl | None]] = {}
    step_count: int = 0
    for slot, p in enumerate(params):
        for name, value in state.get(first + slot, {}).items():
            if name == "step" and isinstance(value, (int, float)):
                # Folding per-parameter arrays to their maximum here gave
                # every parameter the furthest one's count.
                step_count = max(step_count, int(value))
                continue
            by_name.setdefault(name, [None] * len(params))[slot] = _loaded_state_like(
                name, value, p._impl
            )
    if by_name:
        engine.load_state_buffers(list(by_name.items()))
    if step_count and hasattr(engine, "step_count"):
        engine.step_count = step_count


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A ``state_dict()`` payload that fits the optimizer it is loaded into.

    ``groups`` are the saved ``param_groups`` entries, one per live group, each
    listing as many parameters as its live group has.  ``state`` is the saved
    per-parameter state, keyed by flat parameter index as ``state_dict()``
    writes it.
    """

    groups: list[dict[str, object]]
    state: dict[int, dict[str, object]]

    @classmethod
    def read(cls, state_dict: dict[str, object], live_sizes: Sequence[int]) -> Self:
        """Check ``state_dict`` against the optimizer's groups and return it.

        Parameters
        ----------
        state_dict : dict
            An ``optimizer.state_dict()`` payload.
        live_sizes : sequence of int
            The number of parameters in each of the optimizer's groups.

        Raises
        ------
        TypeError
            ``param_groups`` or ``state`` is the wrong kind of container.
        ValueError
            The saved groups are not the optimizer's: another count, or a
            group of another size.  Every saved group is checked, so a
            checkpoint that moves parameters between groups (``[2, 2]`` saved,
            ``[1, 3]`` live) is refused, not loaded onto the wrong parameters.
        """
        raw_groups = state_dict["param_groups"]
        if not isinstance(raw_groups, list):
            raise TypeError("load_state_dict: 'param_groups' must be a list")
        if len(raw_groups) != len(live_sizes):
            raise ValueError(
                "loaded state dict has a different number of parameter groups: "
                f"{len(raw_groups)} saved, {len(live_sizes)} in the optimizer"
            )
        for idx, (saved, live) in enumerate(zip(raw_groups, live_sizes, strict=True)):
            held = cls._saved_size(idx, saved)
            if held != live:
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match "
                    f"the size of optimizer's group: group {idx} has {held} saved "
                    f"parameters, {live} in the optimizer"
                )
        raw_state = state_dict.get("state", {})
        if not isinstance(raw_state, dict):
            raise TypeError("load_state_dict: 'state' must be a dict")
        return cls(raw_groups, raw_state)

    @staticmethod
    def _saved_size(idx: int, saved: object) -> int:
        """The number of parameters saved group ``idx`` lists."""
        params = saved.get("params") if isinstance(saved, dict) else None
        if not isinstance(params, (list, tuple)):
            raise TypeError(
                f"load_state_dict: param_groups[{idx}] must be a dict with a 'params' list"
            )
        return len(params)
