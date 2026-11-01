"""``lucid.peft.tuners`` — one sub-package per parameter-efficient method, 22 in all.

A method is a config class, a tuner (the model-level host that injects, switches, merges and
saves its adapters) and, for the methods that adapt weights, the layer types that wrap the
modules it targets.  Each sub-package holds those layer types together with the method's
config and tuner, registered by ``PeftType`` on import; the configs and tuners are also exported
from ``lucid.peft``, which is where ``get_peft_model`` and ``PeftModel.from_pretrained`` live.
The methods, by family:

* **LoRA family** — a low-rank update beside each frozen weight: ``lora`` (LoRA, with DoRA, rsLoRA
  and PiSSA / OLoRA starts) and ``adalora`` (AdaLoRA, rank allocated per layer during training).
* **Orthogonal family** — a rotation of each layer's input space instead of an addition to its
  weight: ``oft`` (block-diagonal Cayley rotations), ``boft`` (their butterfly factorisation) and
  ``hra`` (a chain of Householder reflections).
* **LyCORIS family** — a weight-shaped update from a structured factorisation: ``loha`` (a Hadamard
  product of low-rank products) and ``lokr`` (a Kronecker product).
* **Prompt learning** — an input learned instead of a weight, with no layer added to the model:
  ``prompt_tuning`` (virtual tokens in front of the embeddings), ``prefix_tuning`` (virtual keys and
  values in every attention layer), ``p_tuning`` (tokens an MLP or LSTM produces) and
  ``multitask_prompt_tuning`` (one prompt shared across tasks, scaled per task).
* **Shared-table tuners** — one set of tensors for the whole model, held once by the tuner, with a
  few parameters per layer: ``vera`` (shared random projections, two vectors per layer),
  ``randlora`` (shared random bases, a full-rank update) and ``vblora`` (a shared vector bank).
* **Structured and sparse updates** — a weight update with a fixed structure other than low rank:
  ``fourierft`` (a few frequency coefficients), ``c3a`` (block-circulant blocks), ``miss`` (shared
  matrix shards), ``bone`` (the method MiSS replaced, kept to read old adapter files) and ``shira``
  (a sparse subset of the weight itself).
* **Scaling and selective tuning** — a small part of the model itself: ``ia3`` (a rescaling vector
  per layer), ``ln_tuning`` (a trainable copy of each chosen normalisation layer) and
  ``trainable_tokens`` (a few rows of an embedding table).

The modules whose names begin with an underscore here (``_base``, ``_lycoris``, ``_orthogonal``,
``_shared``, ``_prompt`` …) are the machinery the methods share, not part of the public surface.
"""

from lucid.peft.tuners import (
    adalora,
    adaption_prompt,
    boft,
    bone,
    c3a,
    fourierft,
    hra,
    ia3,
    ln_tuning,
    loha,
    lokr,
    lora,
    miss,
    multitask_prompt_tuning,
    oft,
    p_tuning,
    prefix_tuning,
    prompt_tuning,
    randlora,
    shira,
    trainable_tokens,
    vblora,
    vera,
)

__all__ = [
    "adalora",
    "adaption_prompt",
    "boft",
    "bone",
    "c3a",
    "fourierft",
    "hra",
    "ia3",
    "ln_tuning",
    "loha",
    "lokr",
    "lora",
    "miss",
    "multitask_prompt_tuning",
    "oft",
    "p_tuning",
    "prefix_tuning",
    "prompt_tuning",
    "randlora",
    "shira",
    "trainable_tokens",
    "vblora",
    "vera",
]
