"""The learnable slopes of a parametric activation, and how they are reset."""

from lucid._factories.creation import full
from lucid._types import DeviceLike, DTypeLike
import lucid.nn.init as init
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter


class _SlopeParameters(Module):
    """A ``weight`` of ``num_parameters`` slopes, every one ``init`` until trained.

    What a parametric activation (:class:`PReLU`) keeps: :meth:`reset_parameters`
    puts every slope back to the ``init`` the layer was built with.
    """

    num_parameters: int
    init: float
    weight: Parameter

    def _make_slopes(
        self, num_parameters: int, init_value: float, device: DeviceLike, dtype: DTypeLike
    ) -> None:
        """Record ``init_value`` and allocate ``weight`` holding it."""
        self.num_parameters = num_parameters
        self.init = init_value
        self.weight = Parameter(full((num_parameters,), init_value, dtype=dtype, device=device))

    def reset_parameters(self) -> None:
        """Put every slope back to ``init``."""
        init.constant_(self.weight, self.init)
