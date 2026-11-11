r"""``lucid.peft`` — parameter-efficient fine-tuning: LoRA and 25 other adapter methods, with one API.

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
layer classes of each are in ``lucid.peft.tuners.<method>``.  The methods,
by family:

* **LoRA family** — a low-rank update beside each frozen weight.  LoRA
  (``LoraConfig`` / ``LoraModel``) and AdaLoRA, which allocates the rank per
  layer during training (``AdaLoraConfig`` / ``AdaLoraModel``).
* **Orthogonal family** — a rotation of each layer's input space instead of
  an addition to its weight.  OFT (block-diagonal Cayley rotations,
  ``OFTConfig`` / ``OFTModel``), BOFT (their butterfly factorisation plus a
  row scale, ``BOFTConfig`` / ``BOFTModel``) and HRA (a chain of Householder
  reflections, ``HRAConfig`` / ``HRAModel``).
* **LyCORIS family** — a weight-shaped update from a structured
  factorisation.  LoHa (a Hadamard product of low-rank products,
  ``LoHaConfig`` / ``LoHaModel``) and LoKr (a Kronecker product,
  ``LoKrConfig`` / ``LoKrModel``).
* **Shared-table tuners** — one set of tensors for the whole model, held once
  by the tuner, and a few parameters per layer.  VeRA (two vectors per layer
  over frozen random projections, ``VeraConfig`` / ``VeraModel``), RandLoRA (a
  full-rank update from frozen random bases and learned scales,
  ``RandLoraConfig`` / ``RandLoraModel``) and VB-LoRA (every low-rank factor
  mixed from one vector bank, ``VBLoRAConfig`` / ``VBLoRAModel``).
* **Structured and sparse updates** — a weight update with a fixed structure
  other than low rank.  FourierFT (a few frequency-domain coefficients,
  ``FourierFTConfig`` / ``FourierFTModel``), C3A (block-circulant blocks,
  ``C3AConfig`` / ``C3AModel``), MiSS (one matrix shared across the shards of
  a weight, ``MissConfig`` / ``MissModel``), Bone (the method MiSS replaced,
  kept to read old adapter files, ``BoneConfig`` / ``BoneModel``) and SHiRA (a
  sparse subset of the weight itself, ``ShiraConfig`` / ``ShiraModel``).
* **Scaling and selective tuning** — a small part of the model itself.
  (IA)^3 (a learned rescaling vector per layer, ``IA3Config`` /
  ``IA3Model``), LN tuning (a trainable copy of each chosen normalisation
  layer, ``LNTuningConfig`` / ``LNTuningModel``) and trainable tokens (a few
  rows of an embedding table, ``TrainableTokensConfig`` /
  ``TrainableTokensModel``).

* **Routing and mixing** — several low-rank adapters combined per task or per token.
  Poly and Multi-Head Routing (every layer holds ``n_skills`` low-rank skills and a router
  with a logit vector per task, ``PolyConfig`` / ``PolyModel``; the call names its tasks with
  ``task_ids``) and X-LoRA (trained LoRA adapters loaded by name and weighed per token by a
  classifier on the final hidden states, ``XLoraConfig`` / ``XLoraModel``; only the classifier
  is saved).
* **Adaption prompts** — LLaMA-Adapter: the top attention layers of a ``gpt`` / ``gpt2`` decoder
  attend to learned prompts through a gate that starts at zero (``AdaptionPromptConfig`` /
  ``AdaptionPromptModel``).

Everything below about adapters, switching, merging and files holds for all
of them (LN tuning applies, and merges, one adapter at a time; adaption prompts apply one at a
time; adaption prompts, Poly and X-LoRA cannot be merged into the weights).

**Prompt learning** is the last family.  It learns an input rather than a weight and adds no
layer to the model: prompt tuning (``PromptTuningConfig``: virtual-token embeddings in front of
the input), prefix tuning (``PrefixTuningConfig``: virtual keys and values in front of every
attention layer) and p-tuning (``PromptEncoderConfig``: the virtual tokens produced by an MLP or
LSTM).  The prompt is applied by hooks on the model's own call, so ``model(...)`` and the model's
``generate`` both carry it; the adapter file holds the finished prompt as
``prompt_embeddings``.  They train one prompt at a time, have nothing to merge, and
support the causal-LM, sequence-to-sequence, sequence-classification, token-classification
and question-answering tasks (the last two get logits as long as the input, the prompt's rows
cut off at the head).  Multitask prompt tuning (``MultitaskPromptTuningConfig``) learns a shared
prompt and a low-rank factor pair per task, chosen per batch row by ``task_ids``; context-aware
prompt tuning (``CPTConfig``) tunes a prompt of in-context examples with a bounded offset and
a loss over the examples' outputs::

    config = PromptTuningConfig(num_virtual_tokens=20, task_type="CAUSAL_LM")
    model = get_peft_model(base_model, config)

Twenty more methods that the reference library has added are here as configs, with no adapter yet:
DeLoRA, GraLoRA, HiRA, GLoRA, DEFT and PEANuT (``DeloraConfig`` …), Uni-LoRA, PVeRA, TinyLoRA and
Lily, RoAd, PSOFT, OSF and FRoD, Super-Tuning, WaveFT, BEFT and Cartridge, and AdaMSS.  Each config has
every field of the reference's, in its order and with its defaults, and reads and writes
``adapter_config.json``; building an adapter with one raises ``NotImplementedError`` naming the issue
that tracks it, and ``PEFT_TYPE_TO_TUNER_MAPPING`` has a tuner class that raises it for each.  The
twentieth, ShadowPEFT, is refused for good — it needs the base model's decoder loop and a second
backbone beside it.  The option classes of LoRA's extensions (``ArrowConfig``, ``BdLoraConfig``,
``KasaConfig``, ``MontecloraConfig``, ``VeloraConfig``) are here too, for the ``LoraConfig`` fields
that hold them; a ``LoraConfig`` that sets one of those extensions is refused.


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

``modules_to_save`` trains whole modules (a classification head) beside the adapter; each
becomes a ``ModulesToSaveWrapper`` that keeps the original and a trained copy per adapter.  An
adapter written by the reference library for a model of the zoo is converted to the zoo's
module layout as it is loaded (``PeftModel.from_pretrained(..., convert=)``).  The reference
library and the zoo name some modules differently and lay a few out differently, so a
converter exists for each of these zoo model types (``model.config.model_type``):

* ``"gpt2"`` — the sequence-classification head ``score`` becomes ``classifier``; no other
  name or tensor changes.
* ``"roformer"`` — the layouts agree, so a file converts by identity.
* ``"t5"`` — the reference's ``block.N.layer.M.<Module>`` nesting becomes the zoo's roles
  (``blocks.N.self_attention``, ``cross_attention``, ``feed_forward``, ``<sub-layer>_norm``),
  and the text-to-text model's modules gain the ``t5.`` trunk prefix.
* ``"vit"`` — the separate ``query`` / ``key`` / ``value`` adapters are fused into the zoo's
  one ``qkv`` (exactly for LoRA; another method on those projections raises
  ``NotImplementedError``), and the remaining modules are renamed.

``convert="auto"`` (the default) converts when the model's type has a converter and the file is
in a layout it changes, and reads any other file as it is, such as one saved by
``save_pretrained``.  ``True`` always converts and raises ``ValueError`` for a model type with
no converter, or for a tensor the converter has no rule for.  ``False`` reads the file as it
is.  A model without a ``config.model_type`` is matched by the reference class that the
adapter's ``auto_mapping`` records.

A model can hold several adapters, switch between them, combine them
(``add_weighted_adapter``: ``svd``, ``cat``, ``linear`` and the TIES / DARE /
magnitude-prune types), and compute a batch whose rows each use a different
adapter (``model(x, adapter_names=[...])``).  ``get_layer_status`` and
``get_model_status`` report what each adapter layer, and the model, is doing.
Adapting a model a second time adds the new adapter beside the first, with a
warning.

The helpers a script moved from the reference library reaches for are here under its
names: ``PeftError`` / ``NoMatchingPeftModuleError`` / ``PeftWarning``, the method tables
``PEFT_TYPE_TO_CONFIG_MAPPING`` and ``PEFT_TYPE_TO_TUNER_MAPPING`` (read-only views),
``get_peft_config``, ``load_peft_weights``, ``get_base_model_state_dict`` /
``set_base_model_state_dict``, ``cast_mixed_precision_params``, ``shift_tokens_right``,
``prepare_model_for_kbit_training`` (freezes and upcasts; a quantized model is refused),
``find_kappa_target_modules`` and the bloom prefix-tuning reshaper.
"""

from lucid.peft import tuners
from lucid.peft._base_state_dict import get_base_model_state_dict, set_base_model_state_dict
from lucid.peft._config import PeftConfig, PeftType, PromptLearningConfig, TaskType
from lucid.peft._errors import NoMatchingPeftModuleError, PeftError, PeftWarning
from lucid.peft._kappa import find_kappa_target_modules
from lucid.peft._load_weights import load_peft_weights
from lucid.peft._mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
    get_peft_config,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft._model_prep import (
    cast_mixed_precision_params,
    prepare_model_for_kbit_training,
    shift_tokens_right,
)
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._peft_model import PeftModel, get_layer_status, get_model_status
from lucid.peft._prefix_postprocess import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    bloom_model_postprocess_past_key_value,
)
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
from lucid.peft.tuners.adamss import AdamssConfig, AdamssModel
from lucid.peft.tuners.adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
from lucid.peft.tuners.beft import BeftConfig, BeftModel
from lucid.peft.tuners.boft import BOFTConfig, BOFTModel
from lucid.peft.tuners.bone import BoneConfig, BoneModel
from lucid.peft.tuners.c3a import C3AConfig, C3AModel
from lucid.peft.tuners.cartridge import CartridgeConfig
from lucid.peft.tuners.cpt import CPTConfig, CPTEmbedding, CPTModel
from lucid.peft.tuners.deft import DeftConfig, DeftModel
from lucid.peft.tuners.delora import DeloraConfig, DeloraModel
from lucid.peft.tuners.fourierft import FourierFTConfig, FourierFTModel
from lucid.peft.tuners.frod import FrodConfig, FrodModel
from lucid.peft.tuners.glora import GloraConfig, GloraModel
from lucid.peft.tuners.gralora import GraloraConfig, GraloraModel
from lucid.peft.tuners.hira import HiraConfig, HiraModel
from lucid.peft.tuners.hra import HRAConfig, HRAModel
from lucid.peft.tuners.ia3._config import IA3Config
from lucid.peft.tuners.ia3._model import IA3Model
from lucid.peft.tuners.lily import LilyConfig, LilyModel
from lucid.peft.tuners.ln_tuning import LNTuningConfig, LNTuningModel
from lucid.peft.tuners.loha._config import LoHaConfig
from lucid.peft.tuners.loha._model import LoHaModel
from lucid.peft.tuners.lokr._config import LoKrConfig
from lucid.peft.tuners.lokr._model import LoKrModel
from lucid.peft.tuners.lora._config import (
    EvaConfig,
    LoftQConfig,
    LoraConfig,
    LoraGAConfig,
    LoraRuntimeConfig,
    replace_lora_weights_loftq,
)
from lucid.peft.tuners.lora._conversion import convert_to_lora, save_as_lora
from lucid.peft.tuners.lora._eva import get_eva_state_dict, initialize_lora_eva_weights
from lucid.peft.tuners.lora._loraga import preprocess_loraga
from lucid.peft.tuners.lora._model import LoraModel
from lucid.peft.tuners.lora._subconfigs import (
    ArrowConfig,
    BdLoraConfig,
    KasaConfig,
    MontecloraConfig,
    VeloraConfig,
)
from lucid.peft.tuners.miss import MissConfig, MissModel
from lucid.peft.tuners.multitask_prompt_tuning import (
    MultitaskPromptEmbedding,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
)
from lucid.peft.tuners.oft import OFTConfig, OFTModel
from lucid.peft.tuners.osf import OSFConfig, OSFModel
from lucid.peft.tuners.p_tuning import (
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
)
from lucid.peft.tuners.peanut import PeanutConfig, PeanutModel
from lucid.peft.tuners.poly import PolyConfig, PolyModel
from lucid.peft.tuners.prefix_tuning import PrefixEncoder, PrefixTuningConfig
from lucid.peft.tuners.prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from lucid.peft.tuners.psoft import PsoftConfig, PsoftModel
from lucid.peft.tuners.pvera import PveraConfig, PveraModel
from lucid.peft.tuners.randlora import RandLoraConfig, RandLoraModel
from lucid.peft.tuners.road import RoadConfig, RoadModel
from lucid.peft.tuners.shadow import ShadowConfig, ShadowModel
from lucid.peft.tuners.shira import ShiraConfig, ShiraModel
from lucid.peft.tuners.supertuning import SupertuningConfig, SupertuningModel
from lucid.peft.tuners.tinylora import TinyLoraConfig, TinyLoraModel
from lucid.peft.tuners.trainable_tokens import TrainableTokensConfig, TrainableTokensModel
from lucid.peft.tuners.unilora import UniLoraConfig, UniLoraModel
from lucid.peft.tuners.vblora import VBLoRAConfig, VBLoRAModel
from lucid.peft.tuners.vera import VeraConfig, VeraModel
from lucid.peft.tuners.waveft import WaveFTConfig, WaveFTModel
from lucid.peft.tuners.xlora import XLoraClassifier, XLoraConfig, XLoraModel

__all__ = [
    "AutoPeftModel",
    "AutoPeftModelForCausalLM",
    "AutoPeftModelForFeatureExtraction",
    "AutoPeftModelForQuestionAnswering",
    "AutoPeftModelForSeq2SeqLM",
    "AutoPeftModelForSequenceClassification",
    "AutoPeftModelForTokenClassification",
    "MODEL_TYPE_TO_PEFT_MODEL_MAPPING",
    "ModulesToSaveWrapper",
    "NoMatchingPeftModuleError",
    "PEFT_TYPE_TO_CONFIG_MAPPING",
    "PEFT_TYPE_TO_TUNER_MAPPING",
    "PeftError",
    "PeftWarning",
    "TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING",
    "MultitaskPromptEmbedding",
    "MultitaskPromptTuningConfig",
    "MultitaskPromptTuningInit",
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
    "AdaptionPromptConfig",
    "BOFTConfig",
    "BoneConfig",
    "C3AConfig",
    "CPTConfig",
    "FourierFTConfig",
    "HRAConfig",
    "IA3Config",
    "LNTuningConfig",
    "LoHaConfig",
    "EvaConfig",
    "LoftQConfig",
    "LoKrConfig",
    "LoraConfig",
    "LoraGAConfig",
    "LoraRuntimeConfig",
    "MissConfig",
    "OFTConfig",
    "PolyConfig",
    "RandLoraConfig",
    "ShiraConfig",
    "TrainableTokensConfig",
    "VBLoRAConfig",
    "VeraConfig",
    "XLoraConfig",
    "AdamssConfig",
    "ArrowConfig",
    "BdLoraConfig",
    "BeftConfig",
    "CartridgeConfig",
    "DeftConfig",
    "DeloraConfig",
    "FrodConfig",
    "GloraConfig",
    "GraloraConfig",
    "HiraConfig",
    "KasaConfig",
    "LilyConfig",
    "MontecloraConfig",
    "OSFConfig",
    "PeanutConfig",
    "PsoftConfig",
    "PveraConfig",
    "RoadConfig",
    "ShadowConfig",
    "SupertuningConfig",
    "TinyLoraConfig",
    "UniLoraConfig",
    "VeloraConfig",
    "WaveFTConfig",
    "PeftType",
    "TaskType",
    "AdaLoraModel",
    "AdaptionPromptModel",
    "TunerLayerStatus",
    "TunerModelStatus",
    "BOFTModel",
    "BoneModel",
    "C3AModel",
    "CPTEmbedding",
    "CPTModel",
    "FourierFTModel",
    "HRAModel",
    "IA3Model",
    "LNTuningModel",
    "LoHaModel",
    "LoKrModel",
    "LoraModel",
    "MissModel",
    "OFTModel",
    "PolyModel",
    "RandLoraModel",
    "ShiraModel",
    "TrainableTokensModel",
    "VBLoRAModel",
    "VeraModel",
    "AdamssModel",
    "BeftModel",
    "DeftModel",
    "DeloraModel",
    "FrodModel",
    "GloraModel",
    "GraloraModel",
    "HiraModel",
    "LilyModel",
    "OSFModel",
    "PeanutModel",
    "PsoftModel",
    "PveraModel",
    "RoadModel",
    "ShadowModel",
    "SupertuningModel",
    "TinyLoraModel",
    "UniLoraModel",
    "WaveFTModel",
    "XLoraClassifier",
    "XLoraModel",
    "PeftModel",
    "PeftModelForCausalLM",
    "PeftModelForFeatureExtraction",
    "PeftModelForQuestionAnswering",
    "PeftModelForSeq2SeqLM",
    "PeftModelForSequenceClassification",
    "PeftModelForTokenClassification",
    "bloom_model_postprocess_past_key_value",
    "cast_mixed_precision_params",
    "find_kappa_target_modules",
    "get_base_model_state_dict",
    "get_layer_status",
    "get_model_status",
    "get_peft_config",
    "get_peft_model",
    "inject_adapter_in_model",
    "get_peft_model_state_dict",
    "load_peft_weights",
    "prepare_model_for_kbit_training",
    "set_base_model_state_dict",
    "convert_to_lora",
    "get_eva_state_dict",
    "initialize_lora_eva_weights",
    "preprocess_loraga",
    "replace_lora_weights_loftq",
    "save_as_lora",
    "set_peft_model_state_dict",
    "shift_tokens_right",
]
