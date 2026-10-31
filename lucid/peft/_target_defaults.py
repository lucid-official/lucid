"""The modules an adapter takes when its config does not say: a default per model type.

``LoraConfig()`` names no ``target_modules``.  The reference library
answers from a table keyed by the base model's ``config.model_type`` —
attention's query and value projections, by the names that architecture
gives them — and refuses a model type it has no row for.  This is that
table for every model type Lucid's zoo registers.  The three the
reference lists (``t5``, ``gpt2``, ``bert``) keep its choice; the others
take the same projections under the zoo's own names, and the convolutional
families — which the reference leaves to the user — the convolutions
(or the fully connected layers) that carry the model.

A row is chosen to hold for **every factory of its type**, not only the
one the tests build: a name that lands on a grouped convolution is left
out (a grouped LoRA convolution cannot be merged), and so is a name that
also lands on a wrapper module the adapter has no layer for.
``lucid/test/unit/models/test_peft_zoo.py`` builds every registered
factory without allocating and checks both, and runs one factory of each
type end to end.  A model with an exponential-moving-average target network
(the JEPAs) or an auxiliary one (diamond's reward and actor) lists it in
``DEFAULT_LORA_EXCLUDE_MODULES``; the config stores the row's own short names
and the reference's ``exclude_modules`` (a pattern over those subtrees), so the
saved adapter follows later changes to the model's structure.  ``neural_ode`` has no module row: its vector field holds raw
parameters, with no ``nn.Linear`` or convolution to adapt, so its default is
``DEFAULT_LORA_TARGET_PARAMETERS``.

Grouped and depthwise convolutions (``groups > 1``) are left out of every row
on purpose: a rank-:math:`r` update mixes channels across groups, which one
grouped kernel cannot hold, so such a layer adapts in the forward pass but its
adapter cannot be merged (the reference's convolution LoRA has the same
restriction).
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import lucid.nn as nn


def _names(*names: str) -> frozenset[str]:
    return frozenset(names)


_QKV: Final = _names("qkv")
_ATTENTION: Final = _names("self_attn", "multihead_attn")
_QV_PROJ: Final = _names("q_proj", "v_proj")
_FC6_FC7: Final = _names("fc6", "fc7")
_BOTTLENECK_CONVS: Final = _names("conv1", "conv2", "conv3")
_UNET_ATTENTION: Final = _names("qkv", "proj")
_DREAMER_HEADS: Final = _names("prior_head.0", "posterior_head.0")
_DREAMER_DECODER: Final = frozenset(f"deconvs.{i}" for i in range(4))

#: ``model_type`` → the module names a LoRA adapter takes by default.  Read-only.
DEFAULT_LORA_TARGET_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        # text and multimodal
        "gpt": _names("c_attn"),
        "gpt2": _names("c_attn"),
        "bert": _names("query", "value"),
        "roformer": _names("query", "value"),
        "t5": _names("q", "v"),
        "transformer": _ATTENTION,
        "clip": _names("attn"),
        # vision transformers
        "vit": _QKV,
        "swin": _QKV,
        "pvt": _names("q", "kv"),
        "coatnet": _names("qkv", "proj"),
        "crossvit": _names("qkv", "wq", "wv"),
        "cvt": _names("proj_q.proj", "proj_v.proj"),
        "efficientformer": _names("qkv", "proj"),
        "maxvit": _names("qkv", "proj"),
        "ijepa": _QKV,
        "vjepa": _QKV,
        "vjepa2": _QKV,
        "vjepa2_ac": _QKV,
        # convolutional classifiers
        "alexnet": _names("features.0", "features.3", "features.6", "features.8", "features.10")
        | _FC6_FC7,
        "zfnet": _names("features.0", "features.4", "features.8", "features.10", "features.12")
        | _FC6_FC7,
        "lenet": _names("features.0", "features.3", "features.6", "f6"),
        "vgg": _names("features.0", "fc6", "fc7"),
        "densenet": _names("conv1", "conv2"),
        "resnet": _BOTTLENECK_CONVS,
        "resnext": _names("conv1", "conv3"),
        "resnest": _names("conv3", "fc1", "fc2"),
        "senet": _BOTTLENECK_CONVS,
        "sknet": _names("fc_reduce", "fc_select"),
        "convnext": _names("fc1", "fc2"),
        "inception_next": _names("fc1", "fc2"),
        "cspnet": _names("conv_exp.conv", "conv_transition.conv"),
        "efficientnet": _names("project_conv"),
        "mobilenet": frozenset(f"{i}.3" for i in range(3, 16)),
        "mobilenet_v2": _names("conv.6"),
        "mobilenet_v3": _names("fc1", "fc2"),
        "mobilenet_v4": _names("pw_exp.conv", "pw_proj.conv", "conv_exp", "conv_pwl"),
        "xception": _names("pointwise"),
        "googlenet": _names(
            "0.conv", "1.conv", "branch1.conv", "conv1.conv", "conv2.conv", "conv3.conv"
        ),
        "inception_v3": _names("conv"),
        "inception_resnet": _names("conv"),
        "inception_v4": _names(
            *("0.conv", "1.conv", "2.conv", "4.conv", "conv.conv", "branch0.conv"),
            *("branch1.3.conv", "branch1_0.conv", "branch1_1a.conv", "branch1_1b.conv"),
            *("branch2.3.conv", "branch2_0.conv", "branch2_1.conv", "branch2_2.conv"),
            *("branch2_3a.conv", "branch2_3b.conv"),
        ),
        # detection and segmentation
        "detr": _ATTENTION,
        "efficientdet": _names("pw"),
        "rcnn": _FC6_FC7,
        "fast_rcnn": _FC6_FC7,
        "faster_rcnn": _FC6_FC7,
        "mask_rcnn": _FC6_FC7,
        "yolo": _names("fc1", "fc2"),
        "yolo_v2": _names(
            *("0.0", "1.0", "2.0", "3.0", "4.0", "det1.0", "det2.0", "det3.0", "route_reduce.0")
        ),
        "yolo_v3": _names("conv"),
        "yolo_v3_tiny": _names("conv"),
        "yolo_v4": _names("conv"),
        "fcn": _BOTTLENECK_CONVS,
        "unet": _names("body.0", "body.3", "up"),
        "attention_unet": _names("net.0", "net.3", "up"),
        "maskformer": _QV_PROJ,
        "mask2former": _QV_PROJ,
        # generative
        "ddpm": _UNET_ATTENTION,
        "ncsn": _UNET_ATTENTION,
        "score_sde": _UNET_ATTENTION,
        "flow_matching": _UNET_ATTENTION,
        "rectified_flow": _names("query", "value"),
        "dit": _names("attn"),
        "mean_flow": _names("attn"),
        "tvm": _QKV,
        "stable_diffusion": _names("attn1", "attn2"),
        "diamond": _names("qkv_proj", "out_proj"),
        "genie": _names("query", "value"),
        "nice": _names("proj"),
        "realnvp": _names("conv1", "conv2"),
        "vqvae": _names("conv1", "conv2", "up_blocks.0", "up_blocks.1"),
        "vae": _names(*(f"{side}_blocks.{i}" for side in ("down", "up") for i in range(3))),
        "dreamer": _DREAMER_HEADS | _DREAMER_DECODER,
        "dreamer_v2": _DREAMER_HEADS | _DREAMER_DECODER,
        "dreamer_v3": _DREAMER_HEADS | _DREAMER_DECODER,
        "planet": _DREAMER_HEADS | _DREAMER_DECODER,
    }
)

#: ``model_type`` → the raw parameters (``target_parameters``) an adapter takes by default, for a
#: model whose weights are parameters of a plain module and not the weight of a layer.  Read-only.
DEFAULT_LORA_TARGET_PARAMETERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {"neural_ode": _names("field.w", "field.u")}
)

_EMA_TARGET: Final = _names("target_encoder")

#: ``model_type`` → the sub-networks a default adapter never goes into, each by the name of the
#: module that roots it: an exponential-moving-average target network is written by averaging the
#: live one and no forward trains it, and an auxiliary network is not the model's own.  A default
#: row names layers by suffix, so a name here removes a whole subtree.  Read-only.
DEFAULT_LORA_EXCLUDE_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "ijepa": _EMA_TARGET,
        "vjepa": _EMA_TARGET,
        "vjepa2": _EMA_TARGET,
        "diamond": _names("reward_end", "actor_critic"),
    }
)

#: The model type of a module that carries none, as the reference spells it.
_UNKNOWN_MODEL_TYPE: Final = "custom"


def _model_type(model: nn.Module) -> str:
    """``model.config.model_type``, or ``"custom"`` for a module without one."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type if isinstance(model_type, str) else _UNKNOWN_MODEL_TYPE


def _default_target_parameters(
    model: nn.Module, table: Mapping[str, frozenset[str]]
) -> frozenset[str]:
    """The parameter names ``table`` gives ``model``'s type; empty if it has no row."""
    return table.get(_model_type(model), frozenset())


def _default_target_modules(
    model: nn.Module, table: Mapping[str, frozenset[str]], parameters: frozenset[str] = frozenset()
) -> frozenset[str]:
    """The names ``table`` gives ``model``'s type; ``ValueError`` if it has no row.

    A type whose default is parameters alone has no module names, and is not an error.
    """
    model_type = _model_type(model)
    names = table.get(model_type)
    if names is None and parameters:
        return frozenset()
    if names is None:
        raise ValueError(
            "Please specify `target_modules` in `peft_config`: a list of module names, a "
            f"regex, or 'all-linear'.  There is no default for model type {model_type!r} "
            f"(the defaults cover {', '.join(sorted(table))})."
        )
    return names


def _default_skipped_subtrees(model: nn.Module) -> frozenset[str]:
    """The sub-networks a default adapter never goes into for ``model``'s type (none for most)."""
    return DEFAULT_LORA_EXCLUDE_MODULES.get(_model_type(model), frozenset())
