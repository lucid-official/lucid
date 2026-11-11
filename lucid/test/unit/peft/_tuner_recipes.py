"""A small model, a config and a forward pass for every method in ``_TUNERS``.

The class-wide contracts (``test_adapter_dtype_contract``) hold every registered tuner to one
rule, so each needs a recipe to be built and called the same way.  A method that adapts the
audit's two linear layers has its row in the audit's ``_EXAMPLE_CONFIGS`` and is covered by that
row alone; a method that adapts something else (a prompt, an attention module, a set of
experts) has a row in ``RECIPES``.  A registered method with neither fails the contract
tests loudly instead of being left out.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    AdaLoraConfig,
    CPTConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    VeraConfig,
    XLoraConfig,
    get_peft_model,
)
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.adaption_prompt import AdaptionPromptConfig
from lucid.test.audit._builders import _EXAMPLE_CONFIGS


def same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    """The largest absolute difference of ``a`` and ``b``, taken in float32."""
    return float((a.to(lucid.float32) - b.to(lucid.float32)).abs().max().item())


def magnitude(a: lucid.Tensor) -> float:
    return float(a.to(lucid.float32).abs().max().item())


def finite(tensor: lucid.Tensor) -> bool:
    return bool(lucid.isfinite(tensor.to(lucid.float32)).all().item())


@dataclass(frozen=True)
class Recipe:
    """A small model a method adapts, what it adapts it with, and how it is called."""

    #: A base model, the same weights on every call, in ``dtype`` on ``device``.
    base: Callable[[lucid.dtype, str], nn.Module]
    #: The method's config; ``scratch`` is a directory it may write to (X-LoRA's experts).
    config: Callable[[lucid.dtype, str, Path], PeftConfig]
    #: One forward pass of the adapted ``model``, as a tensor of the base's output dtype.
    call: Callable[[nn.Module, lucid.dtype, str], lucid.Tensor]
    #: Whether the adapter folds into the base weights (a prompt, a router, a gate cannot).
    mergeable: bool = True
    #: What the adapter trains in on a float64 base: the base's own, except for a prompt, which
    #: is float32 as the reference's is and is cast to the model's dtype where it enters.
    on_float64: lucid.dtype = lucid.float64
    #: Whether a second adapter can be added beside the first (X-LoRA mixes experts instead).
    adds: bool = True
    #: The fields a second adapter sets differently to sit beside the first: AdaLoRA trains one
    #: adapter at a time, and two trainable-token adapters may not name the same token.
    beside: Mapping[str, object] = field(default_factory=dict)

    def second(self, dtype: lucid.dtype, device: str, scratch: Path) -> PeftConfig:
        """The config of a second adapter of the method, which can be added beside its first."""
        config = self.config(dtype, device, scratch)
        for name, value in self.beside.items():
            setattr(config, name, value)
        return config


def _mlp(dtype: lucid.dtype, device: str) -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device).to(dtype)


def _call_mlp(model: nn.Module, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    lucid.manual_seed(1)
    out: lucid.Tensor = model(lucid.randn(3, 6, device=device).to(dtype))
    return out


def _call_routed(model: nn.Module, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    lucid.manual_seed(1)
    x = lucid.randn(3, 6, device=device).to(dtype)
    out: lucid.Tensor = model(x, task_ids=lucid.tensor([0, 1, 0], device=device))
    return out


def _module_recipe(
    make: Callable[[], PeftConfig],
    *,
    routed: bool = False,
    beside: Mapping[str, object] | None = None,
) -> Recipe:
    """An adapter of the ``nn.Linear`` layers of a two-layer MLP."""
    return Recipe(
        _mlp,
        lambda dtype, device, scratch: make(),
        _call_routed if routed else _call_mlp,
        mergeable=not routed,
        beside=beside or {},
    )


def _text(dtype: lucid.dtype, device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = create_model(
        "gpt2_small_lm",
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    return model.to(device).to(dtype)


def _call_text(model: nn.Module, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    ids = lucid.tensor([[1, 2, 3, 4]], device=device)
    logits: lucid.Tensor = model(input_ids=ids).logits
    return logits


def _call_tasks(model: nn.Module, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    ids = lucid.tensor([[1, 2, 3, 4]], device=device)
    logits: lucid.Tensor = model(input_ids=ids, task_ids=lucid.tensor([0], device=device)).logits
    return logits


def _text_recipe(
    make: Callable[[], PeftConfig], *, tasks: bool = False, prompt: bool = True
) -> Recipe:
    """A prompt (``prompt``) or an attention prompt in front of a one-layer language model."""
    return Recipe(
        _text,
        lambda dtype, device, scratch: make(),
        _call_tasks if tasks else _call_text,
        mergeable=False,
        on_float64=lucid.float32 if prompt else lucid.float64,
    )


def _xlora_config(dtype: lucid.dtype, device: str, scratch: Path) -> PeftConfig:
    """X-LoRA over two saved LoRA experts, written beside the test from a float32 copy of the base.

    Two, because one expert's weight is always one; each moved off its start (B = 0), where no
    classifier weight could change the output.
    """
    experts: dict[str, str] = {}
    for seed, name in enumerate("ab"):
        directory = scratch / name
        if not directory.exists():
            saved = get_peft_model(
                _text(lucid.float32, device), LoraConfig(r=2, target_modules=["c_attn"])
            )
            perturb(saved, seed=seed)
            saved.save_pretrained(directory)
        experts[name] = str(directory)
    return XLoraConfig(adapters=experts, task_type="CAUSAL_LM")


RECIPES: dict[PeftType, Recipe] = {
    PeftType.LORA: _module_recipe(lambda: LoraConfig(r=2, target_modules=["0", "2"])),
    PeftType.IA3: _module_recipe(
        lambda: IA3Config(target_modules=["0", "2"], feedforward_modules=["2"])
    ),
    PeftType.LOHA: _module_recipe(lambda: LoHaConfig(r=2, target_modules=["0", "2"])),
    PeftType.LOKR: _module_recipe(lambda: LoKrConfig(r=2, target_modules=["0", "2"])),
    PeftType.ADALORA: _module_recipe(
        lambda: AdaLoraConfig(init_r=2, total_step=10, target_modules=["0", "2"]),
        beside={"inference_mode": True},
    ),
    PeftType.VERA: _module_recipe(lambda: VeraConfig(r=2, target_modules=["0", "2"])),
    # Every other method that adapts the audit's two linear layers has its row in the audit table.
    **{
        peft_type: _module_recipe(
            make,
            routed=peft_type is PeftType.POLY,
            beside={"token_indices": [2]} if peft_type is PeftType.TRAINABLE_TOKENS else None,
        )
        for peft_type, make in _EXAMPLE_CONFIGS.items()
    },
    # A Lily layer mixes its up-projections by the batch's router average: no weight holds the update.
    PeftType.LILY: replace(_module_recipe(_EXAMPLE_CONFIGS[PeftType.LILY]), mergeable=False),
    PeftType.PROMPT_TUNING: _text_recipe(
        lambda: PromptTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PeftType.PREFIX_TUNING: _text_recipe(
        lambda: PrefixTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PeftType.P_TUNING: _text_recipe(
        lambda: PromptEncoderConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PeftType.MULTITASK_PROMPT_TUNING: _text_recipe(
        lambda: MultitaskPromptTuningConfig(
            num_virtual_tokens=2, num_tasks=2, task_type="CAUSAL_LM"
        ),
        tasks=True,
    ),
    PeftType.CPT: _text_recipe(
        lambda: CPTConfig(cpt_token_ids=[1, 2], cpt_tokens_type_mask=[1, 4], task_type="CAUSAL_LM")
    ),
    PeftType.ADAPTION_PROMPT: _text_recipe(
        lambda: AdaptionPromptConfig(adapter_len=2, adapter_layers=1, task_type="CAUSAL_LM"),
        prompt=False,  # the prompt sits in the attention layer, beside its weights
    ),
    PeftType.XLORA: Recipe(_text, _xlora_config, _call_text, mergeable=False, adds=False),
}


def adapted(
    recipe: Recipe, dtype: lucid.dtype, device: str, scratch: Path
) -> tuple[list[lucid.Tensor], PeftModel]:
    """``(own, model)``: the base's own parameters, and the method's adapter on that base."""
    base = recipe.base(dtype, device)
    own = list(base.parameters())  # the adapter is built into ``base`` itself
    return own, get_peft_model(base, recipe.config(dtype, device, scratch))


def trainable(model: nn.Module) -> list[lucid.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def perturb(model: nn.Module, seed: int = 3) -> None:
    """Move every trainable parameter well off its starting value, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in trainable(model):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


def perturb_adapter(model: nn.Module, name: str, seed: int) -> None:
    """Move the parameters of adapter ``name``, trainable or not, well off their starting values."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for _, layer in _tuner_layers(model):
            for p in layer._adapter_parameters(name):
                p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))
