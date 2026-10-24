"""Shared configuration base for the multimodal domain."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig

__all__ = ["MultimodalActivation", "MultimodalModelConfig"]

MultimodalActivation = Literal["gelu", "quick_gelu", "relu", "silu"]
"""Activations a multimodal tower may use.

``quick_gelu`` is here rather than in the generic list because the
released contrastive checkpoints were trained with it and it is not
interchangeable with ``gelu`` — see
:class:`~lucid.models.multimodal.clip._towers.QuickGELU`.
"""


@dataclass(frozen=True)
class MultimodalModelConfig(ModelConfig):
    """Shared base for models whose input spans more than one modality.

    Parameters
    ----------
    embed_dim : int, default=512
        Width of the space every modality is projected into.  This is
        the dimension a downstream consumer sees, and it is deliberately
        separate from either tower's own width — the towers are free to
        differ from each other and from the space they meet in.
    act_fn : MultimodalActivation, default="quick_gelu"
        Activation inside the towers.

    Notes
    -----
    The field this base carries is the one that makes a model
    *multimodal* in the structural sense this zoo classifies by: two or
    more encoders that are trained to land in one representation.  A
    model that merely consumes two inputs without a shared space — a
    conditioned generator, say — belongs to the domain of whatever it
    generates, not here.

    Deliberately **not** carried: the contrastive temperature.  It is
    universal to contrastive families and meaningless to the rest, so it
    belongs to a tier below this one.  That tier is not created while
    :class:`~lucid.models.multimodal.clip.CLIPConfig` is its only
    member — the zoo's rule is that a shared base needs consumers before
    it is written, and the sibling domains' tiers (``DiffusionModelConfig``,
    ``NormalizingFlowConfig``) each had two or three when they appeared.

    Examples
    --------
    >>> from lucid.models.multimodal import MultimodalModelConfig
    >>> MultimodalModelConfig.model_type
    'multimodal'
    """

    model_type: ClassVar[str] = "multimodal"

    embed_dim: int = 512
    act_fn: MultimodalActivation = "quick_gelu"

    def __post_init__(self) -> None:
        """Reject a joint space that cannot hold anything."""
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
