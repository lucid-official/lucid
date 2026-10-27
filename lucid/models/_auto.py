"""Auto classes — task-aware generic model / config loaders.

Each ``AutoModelFor{Task}`` is a thin shell that delegates to the registry,
filtered by ``_task``.  The filter is the whole mechanism: a backbone and the
head built on it are *separate* registry entries under separate names, so
``AutoModel`` resolves ``resnet_50`` to ``ResNet`` and
``AutoModelForImageClassification`` resolves ``resnet_50_cls`` to
``ResNetForImageClassification``.  Asking one Auto type for the other's name
is refused rather than coerced, and the refusal names the entry that fits.
"""

from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel


class _BaseAutoClass:
    r"""Internal mixin shared by every public ``AutoModelFor*`` shell.

    Concrete subclasses set a single class variable, ``_task``, naming the
    task tag (``"image-classification"``, ``"language-modeling"``, …).  The shared
    :meth:`from_pretrained` implementation then resolves a registered model
    name (or a local directory) through the registry and verifies that the
    registry entry's declared task matches ``_task``.

    Attributes
    ----------
    _task : ClassVar[str]
        Task tag every subclass must override.  Used to gate registry
        lookups so that, for example, ``AutoModelForLanguageModeling``
        refuses to return an image-classification head.

    Notes
    -----
    This class is **not** part of the public API — instantiating it raises
    ``EnvironmentError`` and even subclasses are not meant to be
    constructed.  Use the documented ``AutoModelFor*.from_pretrained(...)``
    class method instead.
    """

    _task: ClassVar[str]

    def __init__(self) -> None:
        raise EnvironmentError(
            f"{type(self).__name__} cannot be instantiated. "
            f"Use {type(self).__name__}.from_pretrained(...)."
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, strict: bool = True) -> PretrainedModel:
        r"""Resolve a name or local checkpoint directory and return a model.

        Two input modes are supported:

        1. **Registered model name** (e.g. ``"resnet_50"``) — looked up in
           the global registry.  The registry entry's ``_task`` must equal
           ``cls._task``; otherwise a :class:`ValueError` is raised.  The
           factory is then invoked with ``pretrained=True``.
        2. **Local directory** containing ``config.json`` and either
           ``model.safetensors`` or ``weights.lucid``.  The directory is
           dispatched to :func:`_load_from_directory`, which parses
           ``model_type`` from the config and finds the matching registry
           entry under the requested task.

        Parameters
        ----------
        name_or_path : str
            Either a registered model name (case-insensitive, ``-`` and
            ``_`` interchangeable) or a path to a directory written by
            :meth:`PretrainedModel.save_pretrained`.
        strict : bool, optional, keyword-only, default=True
            Forwarded to :meth:`PretrainedModel.load_state_dict`.  When
            ``False``, mismatched / missing keys are tolerated — useful
            when fine-tuning a backbone whose head shape differs from the
            checkpoint.

        Returns
        -------
        PretrainedModel
            A fully constructed concrete subclass with weights loaded.

        Raises
        ------
        ValueError
            If the name is unknown, or the registered task does not match
            ``cls._task``.
        FileNotFoundError
            If a local directory is supplied but ``config.json`` /
            ``weights`` files are absent.

        Notes
        -----
        The lookup is performed by :func:`_registry_lookup` and uses
        ``_normalize`` so case and hyphen / underscore differences are
        ignored.  When a near-match is found, the error message includes
        up to three Levenshtein-near suggestions.

        Examples
        --------
        >>> from lucid.models import AutoModelForImageClassification
        >>> model = AutoModelForImageClassification.from_pretrained(
        ...     "resnet_50_cls"
        ... )
        >>> type(model).__name__
        'ResNetForImageClassification'
        """
        from lucid.models._registry import _registry_lookup, is_model

        path = Path(name_or_path)
        if path.is_dir():
            return _load_from_directory(cls._task, path, strict=strict)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message, _REGISTRY

            raise ValueError(_unknown_model_message(name_or_path, list(_REGISTRY.keys())))

        entry = _registry_lookup(name_or_path, task=cls._task)
        return entry.factory(pretrained=True)


def _load_from_directory(task: str, path: Path, *, strict: bool) -> PretrainedModel:
    r"""Reconstruct a model from a ``config.json`` + weights directory.

    Two-step strategy:

    1. Parse ``model_type`` from ``config.json`` to identify the family.
    2. Find a registry entry whose ``(task, model_type)`` match.

    The registry entry may carry an optional ``model_class`` shortcut so
    the model can be instantiated directly with the saved config; when
    absent, the factory is called once with ``pretrained=False`` only to
    discover the concrete class, after which the saved config is used to
    re-instantiate it (this guarantees parameter shapes match the
    checkpoint exactly).

    Parameters
    ----------
    task : str
        Task tag the caller's Auto class requires (e.g.
        ``"image-classification"``).
    path : pathlib.Path
        Directory containing ``config.json`` and a weights file.
    strict : bool, keyword-only
        Forwarded to :meth:`PretrainedModel.load_state_dict`.

    Returns
    -------
    PretrainedModel
        A fresh model populated with the on-disk state dict.

    Raises
    ------
    FileNotFoundError
        If ``config.json`` or neither weights file (``model.safetensors``
        or ``weights.lucid``) exist under ``path``.
    ValueError
        If ``config.json`` is malformed, ``model_type`` is missing, or no
        registered entry matches the ``(task, model_type)`` pair.
    TypeError
        If the matching model class fails to declare ``config_class`` or
        the weights file does not contain a state-dict.

    Notes
    -----
    SafeTensors are preferred when present (``model.safetensors``); the
    pickle-based ``weights.lucid`` is the fallback.  Both are loaded via
    :func:`lucid.load` with ``weights_only=True``.
    """
    import json

    import lucid as _lucid
    from lucid.models._registry import _REGISTRY, _RegistryEntry

    cfg_file = path / "config.json"
    weights_st = path / "model.safetensors"
    weights_lucid = path / "weights.lucid"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {path}")
    if weights_st.exists():
        weights_file = weights_st
    elif weights_lucid.exists():
        weights_file = weights_lucid
    else:
        raise FileNotFoundError(
            f"No weights file found in {path}. " f"Expected 'model.safetensors' or 'weights.lucid'."
        )

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"{cfg_file} did not contain a JSON object")
    model_type = cfg_dict.get("model_type")
    if not isinstance(model_type, str):
        raise TypeError(f"{cfg_file} must declare 'model_type' (string)")

    matching: list[_RegistryEntry] = [
        entry
        for entry in _REGISTRY.values()
        if entry.task == task and entry.model_type == model_type
    ]
    if not matching:
        raise ValueError(f"No model registered for task={task!r}, model_type={model_type!r}")

    entry = matching[0]

    if entry.model_class is not None:
        # Fast path: instantiate directly — no factory call needed.
        model_cls = entry.model_class
        if model_cls.config_class is None:
            raise TypeError(f"{model_cls.__name__} must set config_class before loading")
        config = model_cls.config_class.from_dict(cfg_dict)
        model: PretrainedModel = model_cls(config)
    else:
        # Fallback: call factory once to discover the concrete class, then
        # rebuild with the saved config so dimensions match the checkpoint.
        template = entry.factory(pretrained=False)
        model_cls2 = type(template)
        cc2 = model_cls2.config_class
        if cc2 is None:
            raise TypeError(f"{model_cls2.__name__} must set config_class before loading")
        saved_config = cc2.from_dict(cfg_dict)
        model = model_cls2(saved_config)

    sd = _lucid.load(str(weights_file), weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"weights.lucid did not contain a state_dict, got {type(sd).__name__}")
    model.load_state_dict(sd, strict=strict)
    return model


class AutoConfig:
    r"""Generic config loader — return the :class:`ModelConfig` for any name.

    Use this when you only need the hyper-parameter dataclass for a
    registered model, without paying the cost of allocating its weights.
    Typical applications: introspecting layer widths, building parity
    tests, or feeding the config into a custom training loop.

    Notes
    -----
    :class:`AutoConfig` is not instantiable — call
    :meth:`AutoConfig.from_pretrained` directly.  Resolution has two
    fast paths:

    1. If the registry entry carries a ``default_config`` (recommended
       for all Phase 1+ registrations), that config object is returned
       immediately — zero model construction.
    2. Otherwise, the factory is invoked with ``pretrained=False`` and the
       returned model's ``.config`` is read; no checkpoint is downloaded.

    Local directories are also accepted: ``config.json`` is parsed and a
    matching ``ModelConfig`` subclass is reconstructed via
    :meth:`ModelConfig.from_dict`.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> cfg = AutoConfig.from_pretrained("resnet_50")
    >>> cfg.model_type
    'resnet'
    >>> cfg.num_classes
    1000
    """

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoConfig cannot be instantiated. Use AutoConfig.from_pretrained(...)."
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> ModelConfig:
        r"""Return the :class:`ModelConfig` for ``name_or_path``.

        Parameters
        ----------
        name_or_path : str
            Registered model name (e.g. ``"vit_base_16"``) or a directory
            saved by :meth:`PretrainedModel.save_pretrained`.

        Returns
        -------
        ModelConfig
            The default configuration dataclass for that family (no
            weights are downloaded or instantiated).

        Raises
        ------
        ValueError
            If the name is unknown to the registry.
        FileNotFoundError
            If a directory path is supplied but ``config.json`` is absent.

        Notes
        -----
        Resolution order:

        1. ``default_config`` fast path — O(1) when the registry entry
           pre-declared one at ``@register_model`` time.
        2. Factory fallback — invokes the factory with ``pretrained=False``
           and reads ``.config`` off the resulting model.  Still allocates
           parameters, but no checkpoint download.

        Examples
        --------
        >>> from lucid.models import AutoConfig
        >>> cfg = AutoConfig.from_pretrained("bert_base")
        >>> cfg.hidden_size
        768
        """
        from lucid.models._registry import _REGISTRY, _normalize, is_model

        path = Path(name_or_path)
        if path.is_dir():
            return _load_config_from_directory(path)

        if not is_model(name_or_path):
            from lucid.models._registry import _unknown_model_message

            raise ValueError(_unknown_model_message(name_or_path, list(_REGISTRY.keys())))

        # Fast path: default_config pre-registered, no model instantiation.
        entry = _REGISTRY.get(_normalize(name_or_path))
        if entry is not None and entry.default_config is not None:
            return entry.default_config

        # Fallback: build the model (pretrained=False) and return its config.
        from lucid.models._registry import model_entrypoint

        model = model_entrypoint(name_or_path)(pretrained=False)
        return model.config


def _load_config_from_directory(path: Path) -> ModelConfig:
    r"""Reconstruct a :class:`ModelConfig` from a ``config.json`` directory.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing a ``config.json`` produced by
        :meth:`ModelConfig.save`.

    Returns
    -------
    ModelConfig
        Concrete config subclass matching the on-disk ``model_type``.

    Raises
    ------
    FileNotFoundError
        If ``config.json`` is absent.
    ValueError
        If the JSON is malformed, ``model_type`` is missing, or no
        registered entry matches the declared ``model_type``.
    TypeError
        If the matching model class has no ``config_class`` attribute.

    Notes
    -----
    Fast path uses ``default_config`` from the registry entry to discover
    the config class without instantiating the model.  Fallback calls the
    factory with ``pretrained=False`` once just to identify the config
    class, then re-parses the saved JSON through ``from_dict``.
    """
    import json

    from lucid.models._registry import _REGISTRY

    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json not found in {path}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"{cfg_file} did not contain a JSON object")
    model_type = cfg_dict.get("model_type")
    if not isinstance(model_type, str):
        raise TypeError(f"{cfg_file} must declare 'model_type' (string)")

    # Fast path: find an entry with a matching model_type and use its
    # default_config if present.
    for entry in _REGISTRY.values():
        if entry.model_type == model_type and entry.default_config is not None:
            return type(entry.default_config).from_dict(cfg_dict)

    # Fallback: call factory to get the config class, then parse.
    matching = [e for e in _REGISTRY.values() if e.model_type == model_type]
    if not matching:
        raise ValueError(f"No model registered for model_type={model_type!r}")
    template = matching[0].factory(pretrained=False)
    cc_dir = type(template).config_class
    if cc_dir is None:
        raise TypeError(f"{type(template).__name__} must set config_class before loading")
    return cc_dir.from_dict(cfg_dict)


class AutoModel(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads the *backbone* of any family.

    "Backbone" means the family base class with no task head attached
    (e.g. ``ResNet``, not ``ResNetForImageClassification``).  Use this
    when you want raw feature outputs or intend to attach a custom head
    on top.

    Notes
    -----
    Each registered model carries a ``_task`` tag; this class filters on
    ``_task == "base"``.  A given name can resolve to different classes
    under different ``AutoModelFor*`` shells — for example,
    ``AutoModel`` resolves to the bare backbone (``ResNet``) whereas
    ``AutoModelForImageClassification.from_pretrained("resnet_50_cls")``
    returns ``ResNetForImageClassification``.  The two are separate
    registry entries, and only the classifier has published weights.

    Examples
    --------
    A classifier's backbone is not published on its own — the weights
    were trained with the head and are kept with it — so asking for one
    is refused, and the refusal names the factory that has them rather
    than handing back a model that is "pretrained" and random.

    >>> from lucid.models import AutoModel
    >>> AutoModel.from_pretrained("resnet_50")
    Traceback (most recent call last):
        ...
    NotImplementedError: No pretrained weights are published for ``resnet_50``; ``resnet_50(pretrained=True)`` cannot be honoured.  Use ``resnet_50_cls`` for the checkpointed variant.

    Families whose backbone *is* checkpointed load through the same call;
    ``gpt`` is one. For an architecture without weights, build it
    directly instead:

    >>> import lucid
    >>> from lucid.models import create_model
    >>> backbone = create_model("resnet_50").eval()
    >>> type(backbone).__name__
    'ResNet'
    >>> backbone(lucid.randn(1, 3, 224, 224)).last_hidden_state.shape
    (1, 2048, 7, 7)
    """

    _task: ClassVar[str] = "base"


class AutoModelForImageClassification(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a registered image classifier.

    The Auto* family lets you write framework-agnostic code that picks
    the right model at runtime — given a model name string, Lucid looks
    up the registry, verifies the model's declared task matches, and
    returns the concrete subclass with weights loaded.

    Notes
    -----
    Each registered model declares a class-level
    ``_task: ClassVar[str]``.  This Auto class restricts lookups to
    entries tagged ``"image-classification"`` and raises
    :class:`ValueError` on mismatch — preventing accidental loads of a
    detection backbone or a language model as a classifier.

    The returned object is a concrete subclass such as
    ``ResNetForImageClassification`` or ``ViTForImageClassification``;
    its ``forward(images)`` returns an :class:`ImageClassificationOutput`
    with ``logits`` of shape ``(B, num_classes)``.

    Examples
    --------
    >>> from lucid.models import AutoModelForImageClassification
    >>> model = AutoModelForImageClassification.from_pretrained("resnet_50_cls")
    >>> type(model).__name__
    'ResNetForImageClassification'
    >>> import lucid
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """

    _task: ClassVar[str] = "image-classification"


class AutoModelForObjectDetection(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a registered object detector.

    Resolves names like ``"faster_rcnn"`` / ``"detr_resnet50"`` /
    ``"yolo_v3"`` to their concrete ``{Family}ForObjectDetection``
    subclass.

    Notes
    -----
    The detector forward returns an :class:`ObjectDetectionOutput` with
    ``logits`` (class scores per proposal / query) and ``pred_boxes``
    (predicted box coordinates).  Two-stage detectors additionally
    populate ``proposals`` so postprocessing can be reconstructed from
    the output alone.

    Examples
    --------
    >>> from lucid.models import AutoModelForObjectDetection
    >>> model = AutoModelForObjectDetection.from_pretrained("detr_resnet50")
    >>> type(model).__name__
    'DETRForObjectDetection'
    """

    _task: ClassVar[str] = "object-detection"


class AutoModelForSemanticSegmentation(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a semantic-segmentation model.

    Resolves names like ``"fcn_resnet50"`` / ``"unet"`` /
    ``"maskformer_resnet50"`` to their concrete
    ``{Family}ForSemanticSegmentation`` subclass.

    Notes
    -----
    The forward returns a :class:`SemanticSegmentationOutput` whose
    ``logits`` carry per-pixel class scores of shape
    ``(B, num_classes, H, W)``.

    Examples
    --------
    >>> from lucid.models import AutoModelForSemanticSegmentation
    >>> model = AutoModelForSemanticSegmentation.from_pretrained("fcn_resnet50")
    >>> type(model).__name__
    'FCNForSemanticSegmentation'
    """

    _task: ClassVar[str] = "semantic-segmentation"


class AutoModelForLanguageModeling(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a language model.

    Covers the masked, causal and sequence-to-sequence objectives, which
    were three tasks until they were recognised as one — each trains a
    language model, and what differs is which positions are visible when
    a token is predicted.

    Notes
    -----
    The three objectives keep their own output types
    (:class:`MaskedLMOutput`, :class:`CausalLMOutput`,
    :class:`Seq2SeqLMOutput`), and **the forward signatures are not
    interchangeable** — an encoder-decoder model requires
    ``decoder_input_ids`` as a second positional argument where a causal
    model takes only ``input_ids``.  This class resolves a name to a
    class; it does not promise you can call two of them the same way.
    See :mod:`lucid.models._tasks`.

    Examples
    --------
    >>> from lucid.models import AutoModelForLanguageModeling
    >>> model = AutoModelForLanguageModeling.from_pretrained("gpt_lm")
    >>> type(model).__name__
    'GPTLMHeadModel'
    """

    _task: ClassVar[str] = "language-modeling"


class AutoModelForSequenceClassification(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a sentence-level classifier.

    Resolves to ``{Family}ForSequenceClassification`` for GLUE / sentiment
    / NLI fine-tunes across BERT / GPT / GPT-2 / RoFormer / Transformer.

    Notes
    -----
    The forward returns an output whose ``logits`` are shaped
    ``(B, num_labels)``.  Each backbone wraps a pooled representation
    (CLS for BERT-style, last non-pad token for GPT-style) through a
    Linear classifier head.

    Examples
    --------
    >>> from lucid.models import AutoModelForSequenceClassification
    >>> model = AutoModelForSequenceClassification.from_pretrained("bert_base_cls")
    >>> type(model).__name__
    'BERTForSequenceClassification'
    """

    _task: ClassVar[str] = "sequence-classification"


class AutoModelForTokenClassification(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a per-token classifier.

    Resolves to ``{Family}ForTokenClassification`` for NER / POS-tagging
    fine-tunes across BERT / RoFormer / Transformer.

    Notes
    -----
    The forward returns ``logits`` of shape ``(B, T, num_labels)`` and is
    typically trained against ``(B, T)`` label tensors via
    :meth:`MaskedLMMixin.compute_lm_loss`, which folds the sequence axis
    into the batch axis for cross-entropy.

    Examples
    --------
    >>> from lucid.models import AutoModelForTokenClassification
    >>> model = AutoModelForTokenClassification.from_pretrained("bert_base_token_cls")
    >>> type(model).__name__
    'BERTForTokenClassification'
    """

    _task: ClassVar[str] = "token-classification"


class AutoModelForImageGeneration(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads an image-generation model.

    Resolves names like ``"vae_gen"`` / ``"ddpm_cifar_gen"`` /
    ``"ncsn_cifar_gen"`` to their concrete ``{Family}ForImageGeneration``
    subclass — VAE / DDPM / NCSN today, future flow models tomorrow.

    Notes
    -----
    All ``ForImageGeneration`` subclasses expose a ``generate(...)``
    method (via :class:`DiffusionMixin` for diffusion models, or a
    family-specific method for VAEs).  See
    :class:`GenerationOutput` for the structure of the returned samples.

    Examples
    --------
    >>> from lucid.models import AutoModelForImageGeneration
    >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
    >>> type(model).__name__
    'DDPMForImageGeneration'
    """

    _task: ClassVar[str] = "image-generation"


class AutoModelForWorldModeling(_BaseAutoClass):
    r"""Auto-dispatching wrapper that loads a world model.

    Resolves names like ``"planet_world_model"`` to their concrete
    ``{Family}ForWorldModeling`` subclass.

    A world model here is a *latent dynamics* model: it consumes a
    trajectory of observations and actions and predicts how the latent
    state evolves, rather than mapping one input to one label.  Its
    forward therefore takes two sequences, not a single tensor.

    Notes
    -----
    All ``ForWorldModeling`` subclasses expose ``observe(...)`` — filter a
    trajectory into posterior states — and ``imagine(...)`` — roll the
    dynamics forward from a state with no observations at all.  The second
    is what planning and policy learning are built on: it is a forward pass
    over the model's own parameters, with no environment involved.

    Examples
    --------
    Only ``diamond_world_model`` has published weights; the PlaNet and
    Dreamer entries are architectures without checkpoints, so asking for
    one is refused rather than answered with random parameters.

    >>> from lucid.models import AutoModelForWorldModeling
    >>> AutoModelForWorldModeling.from_pretrained("planet_world_model")
    Traceback (most recent call last):
        ...
    NotImplementedError: No pretrained weights are published for ``planet_world_model``; ``planet_world_model(pretrained=True)`` cannot be honoured.

    Build one directly to work with the architecture itself:

    >>> from lucid.models import create_model
    >>> model = create_model("planet_world_model")
    >>> type(model).__name__
    'PlaNetForWorldModeling'
    """

    _task: ClassVar[str] = "world-modeling"
