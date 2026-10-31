r"""``lucid.peft`` — parameter-efficient fine-tuning: LoRA, AdaLoRA, (IA)^3, LoHa, LoKr and VeRA.

Fine-tuning a pretrained model end to end updates, stores and ships every
weight.  LoRA (Hu et al., 2021) freezes them and learns, beside each
chosen ``nn.Linear``, a rank-:math:`r` update :math:`s\,BA` — on GPT-2
small's ``c_attn`` at :math:`r = 8`, 294,912 parameters, 0.24 % of the
model — so a fine-tune is trained, saved and swapped as a few megabytes.

The API and the adapter file format are the reference adapter library's,
names and defaults included, so a script written for it reads the same
and an adapter directory moves between the two::

    from lucid.peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

    config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(base_model, config)    # base_model changed in place
    model.print_trainable_parameters()
    ...                                            # train: only the adapter learns
    model.save_pretrained("my-adapter")            # adapter_config.json + adapter_model.safetensors

    PeftConfig.from_pretrained("my-adapter").base_model_name_or_path  # which base it needs
    model = PeftModel.from_pretrained(fresh_base_model, "my-adapter")
    merged = model.merge_and_unload()              # a plain model again, adapter folded in

Each method is a config class, a tuner and its layer classes, registered
by ``PeftType`` as they are defined (``lucid.peft.tuners._base``); the
entry points above dispatch on the method a config or a file names.  The
methods are LoRA (``LoraConfig`` / ``LoraModel``), AdaLoRA — LoRA as a
truncated SVD whose rank per layer is allocated during training
(``AdaLoraConfig`` / ``AdaLoraModel``), (IA)^3 — a learned rescaling vector per
layer (``IA3Config`` / ``IA3Model``), the LyCORIS pair LoHa (a Hadamard product
of low-rank products, ``LoHaConfig`` / ``LoHaModel``) and LoKr (a Kronecker
product, ``LoKrConfig`` / ``LoKrModel``), and VeRA — two learned vectors per
layer over one pair of frozen random projections (``VeraConfig`` /
``VeraModel``); the layer classes of each are in ``lucid.peft.tuners.<method>``.
Everything below about adapters, switching, merging and files holds for all six.

Three more methods learn an input rather than a weight and add no layer to the model:
prompt tuning (``PromptTuningConfig``: virtual-token embeddings in front of the input),
prefix tuning (``PrefixTuningConfig``: virtual keys and values in front of every attention
layer) and p-tuning (``PromptEncoderConfig``: the virtual tokens produced by an MLP or LSTM).
The prompt is applied by hooks on the model's own call, so ``model(...)`` and the model's
``generate`` both carry it; the adapter file holds the finished prompt as
``prompt_embeddings``.  They train one prompt at a time, have nothing to merge, and
support the causal-LM, sequence-to-sequence and sequence-classification tasks::

    config = PromptTuningConfig(num_virtual_tokens=20, task_type="CAUSAL_LM")
    model = get_peft_model(base_model, config)


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

A model holds any number of adapters, each under its own name::

    model.add_adapter("style", LoraConfig(r=4, target_modules=["c_attn"]))
    model.set_adapter("style")                     # compute with — and train — this one
    model.add_weighted_adapter(["default", "style"], [0.7, 0.3], "mix")
    model.save_pretrained("adapters")              # "default" in adapters/, "style" in adapters/style/
    model.load_adapter("adapters/style", "style-again")

The model class follows the task: ``get_peft_model`` returns a
``PeftModelForCausalLM`` for ``task_type="CAUSAL_LM"`` (and the matching
``PeftModelFor...`` class for the other tasks, which also train a
classification or question-answering head), and a config with no
``target_modules`` takes the default for the base model's ``model_type``.
An adapter directory that names its base model reopens in one call::

    model = AutoPeftModelForCausalLM.from_pretrained("my-adapter")

LoRA layers exist for ``nn.Linear``, ``nn.Embedding``, ``nn.Conv1d`` /
``Conv2d`` / ``Conv3d``, their ``nn.ConvTranspose`` counterparts and
``nn.MultiheadAttention`` (found by the module's
type in ``target_modules``), and ``target_parameters`` adapts a raw
``nn.Parameter`` of a module (``lucid.peft.tuners.lora``).  Asking for
more — ``add_weighted_adapter`` over anything but ``nn.Linear`` layers, an
adapter file of a method this package lacks — raises ``NotImplementedError``
rather than doing less than asked.  Adapters are read from a local directory;
hub ids are not resolved.

A model can hold several adapters, switch between them, combine them
(``add_weighted_adapter``: ``svd``, ``cat``, ``linear`` and the TIES / DARE /
magnitude-prune types), and compute a batch whose rows each use a different
adapter (``model(x, adapter_names=[...])``).  ``get_layer_status`` and
``get_model_status`` report what each adapter layer, and the model, is doing.
Adapting a model a second time adds the new adapter beside the first, with a
warning.
"""

from lucid.peft import tuners
from lucid.peft._config import PeftConfig, PeftType, PromptLearningConfig, TaskType
from lucid.peft._mapping import get_peft_model, inject_adapter_in_model
from lucid.peft._peft_model import PeftModel, get_layer_status, get_model_status
from lucid.peft._state_dict import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft._status import TunerLayerStatus, TunerModelStatus
from lucid.peft._task_models import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from lucid.peft.auto import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
)
from lucid.peft.tuners.adalora import AdaLoraConfig, AdaLoraModel
from lucid.peft.tuners.ia3._config import IA3Config
from lucid.peft.tuners.ia3._model import IA3Model
from lucid.peft.tuners.loha._config import LoHaConfig
from lucid.peft.tuners.loha._model import LoHaModel
from lucid.peft.tuners.lokr._config import LoKrConfig
from lucid.peft.tuners.lokr._model import LoKrModel
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.lora._model import LoraModel
from lucid.peft.tuners.p_tuning import (
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
)
from lucid.peft.tuners.prefix_tuning import PrefixEncoder, PrefixTuningConfig
from lucid.peft.tuners.prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from lucid.peft.tuners.vera import VeraConfig, VeraModel

__all__ = [
    "AutoPeftModel",
    "AutoPeftModelForCausalLM",
    "AutoPeftModelForFeatureExtraction",
    "AutoPeftModelForQuestionAnswering",
    "AutoPeftModelForSeq2SeqLM",
    "AutoPeftModelForSequenceClassification",
    "AutoPeftModelForTokenClassification",
    "MODEL_TYPE_TO_PEFT_MODEL_MAPPING",
    "PeftConfig",
    "PrefixEncoder",
    "PrefixTuningConfig",
    "PromptEmbedding",
    "PromptEncoder",
    "PromptEncoderConfig",
    "PromptEncoderReparameterizationType",
    "PromptLearningConfig",
    "PromptTuningConfig",
    "PromptTuningInit",
    "AdaLoraConfig",
    "IA3Config",
    "LoHaConfig",
    "LoKrConfig",
    "LoraConfig",
    "VeraConfig",
    "PeftType",
    "TaskType",
    "AdaLoraModel",
    "TunerLayerStatus",
    "TunerModelStatus",
    "IA3Model",
    "LoHaModel",
    "LoKrModel",
    "LoraModel",
    "VeraModel",
    "PeftModel",
    "PeftModelForCausalLM",
    "PeftModelForFeatureExtraction",
    "PeftModelForQuestionAnswering",
    "PeftModelForSeq2SeqLM",
    "PeftModelForSequenceClassification",
    "PeftModelForTokenClassification",
    "get_layer_status",
    "get_model_status",
    "get_peft_model",
    "inject_adapter_in_model",
    "get_peft_model_state_dict",
    "set_peft_model_state_dict",
]
