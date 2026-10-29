r"""``lucid.peft`` — parameter-efficient fine-tuning: LoRA on ``nn.Linear``.

Fine-tuning a pretrained model end to end updates, stores and ships every
weight.  LoRA (Hu et al., 2021) freezes them and learns, beside each
chosen ``nn.Linear``, a rank-:math:`r` update :math:`s\,BA` — on GPT-2
small's ``c_attn`` at :math:`r = 8`, 294,912 parameters, 0.24 % of the
model — so a fine-tune is trained, saved and swapped as a few megabytes.

The API and the adapter file format are the reference adapter library's,
names and defaults included, so a script written for it reads the same
and an adapter directory moves between the two::

    from lucid.peft import LoraConfig, PeftModel, get_peft_model

    config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(base_model, config)    # base_model changed in place
    model.print_trainable_parameters()
    ...                                            # train: only the adapter learns
    model.save_pretrained("my-adapter")            # adapter_config.json + adapter_model.safetensors

    model = PeftModel.from_pretrained(fresh_base_model, "my-adapter")
    merged = model.merge_and_unload()              # a plain model again, adapter folded in

What a LoRA layer guarantees:

* **The base is untouched until merged.**  The wrapped ``nn.Linear`` is
  kept as ``base_layer`` with its own ``Parameter`` objects — not
  copies — so an optimizer built earlier, a tied embedding and Core ML's
  identity-keyed weight sharing all still see them, and with the
  default initialisation (:math:`B = 0`) the adapted model's output is
  the base model's, bit for bit.
* **Only the adapter trains.**  Every other parameter is frozen; the
  ``bias`` option and ``modules_to_save`` add biases and whole modules.
* **A half-precision base gets a float32 adapter**, computed in float32
  and cast back to the base's output dtype.
* **Merging refuses a shared weight.**  Folding the update into a weight
  another module also uses (a tied LM head) would change that module;
  ``merge_and_unload`` raises instead.

v1 adapts ``nn.Linear`` only and keeps one adapter per model.  Asking for
more — an embedding or convolution target, DoRA, a second adapter, an
adapter file of another method — raises ``NotImplementedError`` rather
than doing less than asked.  Adapters are read from a local directory;
hub ids are not resolved.
"""

from lucid.peft import tuners
from lucid.peft._config import LoraConfig, PeftType, TaskType
from lucid.peft._mapping import get_peft_model, inject_adapter_in_model
from lucid.peft._peft_model import PeftModel
from lucid.peft._state_dict import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft.tuners.lora._model import LoraModel

__all__ = [
    "LoraConfig",
    "PeftType",
    "TaskType",
    "LoraModel",
    "PeftModel",
    "get_peft_model",
    "inject_adapter_in_model",
    "get_peft_model_state_dict",
    "set_peft_model_state_dict",
]
