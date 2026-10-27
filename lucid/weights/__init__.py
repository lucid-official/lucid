"""``lucid.weights`` — pretrained-weight system for the model zoo.

A standalone package (sibling to :mod:`lucid.models`) that provides the
*infrastructure* for pretrained checkpoints: the tagged-variant enum
base, hub download + SHA verification, loading, and a discovery
registry.  Preprocessing lives in :mod:`lucid.utils.transforms`; a
checkpoint's :class:`WeightEntry` carries one of those transforms.

It deliberately contains **no architecture-specific weight
declarations**.  Each model family declares its own checkpoints in a
``_weights.py`` module *inside the model package* (e.g.
``lucid.models.vision.resnet._weights.ResNet18Weights``), importing the
primitives below.  This keeps :mod:`lucid.weights` a pure porting
substrate and the dependency one-directional (``models`` → ``weights``,
never the reverse).

Design
------
Each architecture declares a :class:`WeightsEnum` (e.g.
``ResNet18Weights``) whose members are :class:`WeightEntry` manifests —
one per tagged checkpoint (``IMAGENET1K_V1``, …) plus a ``DEFAULT``
alias.  This mirrors the reference_vision ``Weights`` enum so call sites are
familiar::

    import lucid.models as models
    import lucid.weights as weights

    # default tag
    model = models.resnet_18(pretrained=True)

    # explicit tag — enum or string
    from lucid.models.vision.resnet import ResNet18Weights
    model = models.resnet_18_cls(weights=ResNet18Weights.IMAGENET1K_V1)
    model = models.resnet_18_cls(pretrained="IMAGENET1K_V1")

    # discover + preprocess (after the model package is imported)
    weights.list_pretrained("resnet_18")        # ['IMAGENET1K_V1']
    w = weights.get_weight("ResNet18Weights.IMAGENET1K_V1")
    x = w.transforms()(image)

The package depends only on :mod:`lucid.serialization` (for SafeTensors
loading); model factories depend on it, never the reverse, so there is
no circular import.  Crucially, :class:`WeightEntry` carries no model
config — named factories already pin their own.

Checkpoints are hosted on the Hugging Face Hub under the ``lucid-dl``
org, converted from reference_vision / timm / transformers sources by the
offline ``tools/convert_weights`` pipeline.
"""

from lucid.weights._base import WeightEntry, WeightsEnum
from lucid.weights._hub import download
from lucid.weights._loading import load_weight_entry, resolve_weights
from lucid.weights._registry import get_weight, list_pretrained, register_weights, weights_for

#: Hugging Face Hub URL root for every Lucid-hosted checkpoint.  Each
#: per-family ``_weights.py`` imports this and composes its repo URL as
#: ``f"{HUB_BASE}/<family-slug>/resolve/main/<TAG>/model.safetensors"``,
#: so the org name lives in exactly one place (e.g. if Lucid ever moves
#: off the ``lucid-dl`` org, only this constant changes).
HUB_BASE: str = "https://huggingface.co/lucid-dl"

__all__ = [
    # Core types
    "WeightsEnum",
    "WeightEntry",
    # Discovery
    "get_weight",
    "list_pretrained",
    "register_weights",
    "weights_for",
    # Fetch / load
    "download",
    "load_weight_entry",
    "resolve_weights",
    # Hub root
    "HUB_BASE",
]
