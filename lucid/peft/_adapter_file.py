"""One adapter as its directory holds it."""

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.peft._config import PeftConfig


class _AdapterFile[C: PeftConfig](NamedTuple):
    """The config and the tensors (by file key) of one saved adapter.

    A leaf module: the init strategies (``lora._inits``), which the config
    imports, hand these around without importing the config back.
    """

    config: C
    state: dict[str, Tensor]
