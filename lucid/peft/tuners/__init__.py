"""``lucid.peft.tuners`` — one sub-package per parameter-efficient method, 46 in all.

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
  values in every attention layer), ``p_tuning`` (tokens an MLP or LSTM produces),
  ``multitask_prompt_tuning`` (one prompt shared across tasks, scaled per task) and ``cpt`` (a
  prompt of in-context examples with a bounded learned offset and its own loss).
* **Shared-table tuners** — one set of tensors for the whole model, held once by the tuner, with a
  few parameters per layer: ``vera`` (shared random projections, two vectors per layer),
  ``randlora`` (shared random bases, a full-rank update) and ``vblora`` (a shared vector bank).
* **Structured and sparse updates** — a weight update with a fixed structure other than low rank:
  ``fourierft`` (a few frequency coefficients), ``c3a`` (block-circulant blocks), ``miss`` (shared
  matrix shards), ``bone`` (the method MiSS replaced, kept to read old adapter files) and ``shira``
  (a sparse subset of the weight itself).
* **Routing and mixing** — several low-rank adapters combined per task or per token: ``poly``
  (low-rank skills mixed by a per-task router) and ``xlora`` (LoRA experts weighed per token by a
  classifier on the hidden states).
* **Adaption prompts** — ``adaption_prompt`` (LLaMA-Adapter): learned prompts the top attention
  layers attend to through a zero-initialised gate.
* **Scaling and selective tuning** — a small part of the model itself: ``ia3`` (a rescaling vector
  per layer), ``ln_tuning`` (a trainable copy of each chosen normalisation layer) and
  ``trainable_tokens`` (a few rows of an embedding table).

Twenty more methods are full tuners like the ones above, grouped by the form of their update:
``delora``, ``gralora``, ``hira``, ``glora``, ``deft`` and ``peanut`` (a weight update of low-rank
or projection form), ``unilora``, ``pvera``, ``tinylora`` and ``lily`` (adapters read out of shared
banks), ``road``, ``psoft``, ``osf`` and ``frod`` (orthogonal and spectral updates),
``supertuning``, ``waveft``, ``beft`` and ``cartridge`` (sparse, wavelet, bias and key-value
updates) and ``adamss`` (subspaces of the weight's SVD).  ``shadow`` is refused for good: its
config reads and writes, but the method needs the base model's decoder loop and a second backbone
beside it, so building it raises ``NotImplementedError``.

``MixedModel`` is the host of a model that holds adapters of several of these methods at once, their
layers stacked on each module; only the methods declared ``mixed=True`` may take part.

The modules whose names begin with an underscore here (``_base``, ``_lycoris``, ``_orthogonal``,
``_shared``, ``_prompt`` …) are the machinery the methods share, not part of the public surface.
"""

from lucid.peft.tuners import (
    adalora,
    adamss,
    adaption_prompt,
    beft,
    boft,
    bone,
    c3a,
    cartridge,
    cpt,
    deft,
    delora,
    fourierft,
    frod,
    glora,
    gralora,
    hira,
    hra,
    ia3,
    lily,
    ln_tuning,
    loha,
    lokr,
    lora,
    miss,
    multitask_prompt_tuning,
    oft,
    osf,
    p_tuning,
    peanut,
    poly,
    prefix_tuning,
    prompt_tuning,
    psoft,
    pvera,
    randlora,
    road,
    shadow,
    shira,
    supertuning,
    tinylora,
    trainable_tokens,
    unilora,
    vblora,
    vera,
    waveft,
    xlora,
)
from lucid.peft.tuners._mixed import MixedModel

__all__ = [
    "MixedModel",
    "adalora",
    "adamss",
    "adaption_prompt",
    "beft",
    "boft",
    "bone",
    "c3a",
    "cartridge",
    "cpt",
    "deft",
    "delora",
    "fourierft",
    "frod",
    "glora",
    "gralora",
    "hira",
    "hra",
    "ia3",
    "lily",
    "ln_tuning",
    "loha",
    "lokr",
    "lora",
    "miss",
    "multitask_prompt_tuning",
    "oft",
    "osf",
    "p_tuning",
    "peanut",
    "poly",
    "prefix_tuning",
    "prompt_tuning",
    "psoft",
    "pvera",
    "randlora",
    "road",
    "shadow",
    "shira",
    "supertuning",
    "tinylora",
    "trainable_tokens",
    "unilora",
    "vblora",
    "vera",
    "waveft",
    "xlora",
]
