"""Task-tagged global model registry with name normalization."""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, final

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel


class ModelFactory(Protocol):
    r"""Protocol implemented by every factory function registered with
    :func:`register_model`.

    A factory takes a single positional flag and an open-ended set of
    keyword overrides, and returns a fully constructed
    :class:`PretrainedModel`.  The ``pretrained`` flag controls whether
    the factory should download and load a checkpoint (``True``) or just
    allocate randomly initialised weights (``False``).

    Parameters
    ----------
    pretrained : bool, default=False
        Whether to download / load registered checkpoint weights into the
        model before returning it.
    **overrides : Any
        Optional config-field overrides.  Forwarded to the underlying
        :class:`ModelConfig` constructor — e.g.
        ``create_model("resnet_50", num_classes=10)``.  Typed ``Any``
        rather than ``object`` because keyword parameters are
        contravariant: a factory that adds a typed keyword such as
        ``weights: ResNet50Weights | None`` only satisfies the protocol
        when the protocol's keywords are ``Any``.

    Returns
    -------
    PretrainedModel
        Concrete model instance ready for ``.forward(...)``.

    Notes
    -----
    The ``__name__`` attribute is part of the protocol because
    :func:`register_model` uses it as the registry key when no explicit
    name is supplied (which is the common case — the decorator captures
    the function's defined name).

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> from lucid.models import ModelConfig, PretrainedModel
    >>> from lucid.models import register_model, create_model
    >>> @dataclass(frozen=True)
    ... class MyConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "myfamily"
    ...     hidden_size: int = 8
    ...     num_classes: int = 2
    >>> class MyModel(PretrainedModel):
    ...     config_class: ClassVar[type[MyConfig]] = MyConfig
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         self.linear = nn.Linear(config.hidden_size, config.num_classes)
    ...     def forward(self, x):
    ...         return self.linear(x)
    >>> @register_model(task="image-classification", family="myfamily")
    ... def my_model(pretrained: bool = False, **overrides: object):
    ...     return MyModel(MyConfig(**overrides))
    >>> model = create_model("my_model", num_classes=10)
    >>> model(lucid.randn(1, 8)).shape
    (1, 10)
    """

    __name__: str  # every Python function has __name__; Protocol must declare it

    def __call__(self, pretrained: bool = False, **overrides: Any) -> PretrainedModel: ...


@final
@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    r"""Internal record stored under each registered name.

    Attributes
    ----------
    factory : ModelFactory
        The decorated function that builds the model.
    task : str
        Task tag (``"base"``, ``"image-classification"``,
        ``"language-modeling"``, …) used by Auto classes for typed
        dispatch.  See :mod:`lucid.models._tasks` for the eight.
    family : str
        Architecture family identifier (e.g. ``"resnet"``, ``"bert"``).
        Inferred from the factory module path when not supplied.
    model_type : str
        Persistent ``ModelConfig.model_type`` value this factory produces.
        Used by directory-based loading to match ``config.json`` against
        registered factories.
    model_class : type[PretrainedModel] or None, default=None
        Optional fast-path pointer to the concrete model class — skips a
        factory call during ``from_pretrained`` from a directory.
    default_config : ModelConfig or None, default=None
        Optional fast-path default config — lets
        :meth:`AutoConfig.from_pretrained` return without invoking the
        factory.
    """

    factory: ModelFactory
    task: str
    family: str
    model_type: str
    # Optional fast-path fields: when provided, Auto classes avoid calling the
    # factory just to discover the class or default config.
    model_class: type[PretrainedModel] | None = field(default=None)
    default_config: ModelConfig | None = field(default=None)
    # Paper-cited trainable parameter count of the factory's default config
    # (e.g. ``61_100_840`` for ``alexnet_cls``).  Source: the original paper
    # / official release notes.  Surfaced as the "MODEL SIZE" card on the
    # docs site and as a ``"61.1M"``-style tag on factory cards.  ``None``
    # when no authoritative count is known — UI then omits the tag.
    params: int | None = field(default=None)
    # Layer-summary policy for the docs site's expandable model tree:
    #   "auto" (default) — ``tools/build_model_summaries.py`` instantiates
    #     the factory at build time, walks ``named_modules()``, compresses
    #     consecutive identical children with ``× N``, caches the result.
    #   None — explicit opt-out (e.g. factories too large to instantiate
    #     on a build machine, or models that require external resources).
    #   dict — pre-built declarative tree (for legacy / special cases).
    summary: object = field(default="auto")


_REGISTRY: dict[str, _RegistryEntry] = {}


def _normalize(name: str) -> str:
    r"""Canonicalise a model name for case- and separator-insensitive lookup.

    Parameters
    ----------
    name : str
        User-supplied name (e.g. ``"ResNet-50"``).

    Returns
    -------
    str
        Lowercase form with hyphens replaced by underscores
        (``"resnet_50"``).

    Notes
    -----
    All registry keys are stored normalised; every lookup runs through
    this function so ``"resnet_50"``, ``"resnet-50"``, and ``"ResNet-50"``
    all reach the same entry.
    """
    return name.replace("-", "_").lower()


def _naming(fn: ModelFactory, name: str) -> ModelFactory:
    r"""``fn`` wrapped so the model it builds carries ``name`` as its ``name_or_path``.

    The name is set only when it rebuilds the model: a call with any keyword
    override builds something else, and a source that rebuilds the registered
    default would hand ``lucid.peft`` (``AutoPeftModel``) the wrong base.  Such
    a model has none, as one built straight from a config has none.

    The wrapper keeps ``fn``'s name, module, docstring and signature
    (``functools.wraps``), so tools that read the factory by those see the
    function it wraps.
    """

    @functools.wraps(fn)
    def named(pretrained: bool = False, **overrides: object) -> PretrainedModel:
        model = fn(pretrained=pretrained, **overrides)
        if not overrides:
            model.name_or_path = name
        return model

    return named


def register_model(
    *,
    task: str = "base",
    family: str | None = None,
    model_type: str | None = None,
    model_class: type[PretrainedModel] | None = None,
    default_config: ModelConfig | None = None,
    params: int | None = None,
    summary: object = "auto",
) -> Callable[[ModelFactory], ModelFactory]:
    r"""Decorator that registers a factory function under its ``__name__``.

    The global registry is the single source of truth for which models
    Lucid exposes.  Every public factory in ``lucid/models/**/_pretrained.py``
    is decorated with ``@register_model`` so it becomes discoverable via
    :func:`create_model`, :func:`list_models`, and the ``AutoModelFor*``
    family.

    Parameters
    ----------
    task : str, default="base"
        Task tag used by Auto classes for typed dispatch.  One of
        ``"base"`` (a backbone with no head) or the eight task tags
        declared in :mod:`lucid.models._tasks`:
        ``"image-classification"``, ``"image-generation"``,
        ``"object-detection"``, ``"semantic-segmentation"``,
        ``"sequence-classification"``, ``"token-classification"``,
        ``"language-modeling"``, ``"world-modeling"``.  The tag a factory
        passes must equal its ``model_class.task``, which the suite
        asserts — the two used to be independent strings and drifted.
    family : str or None, optional
        Architecture family identifier (e.g. ``"resnet"``).  When ``None``,
        inferred from the factory's parent module name (the
        second-to-last component of ``fn.__module__``).
    model_type : str or None, optional
        Persistent ``ModelConfig.model_type`` value this factory produces.
        Defaults to ``family``.  Used by directory-based loading to match
        on-disk ``config.json`` files against registered factories.
    model_class : type[PretrainedModel] or None, optional
        The concrete :class:`PretrainedModel` subclass returned by this
        factory.  When supplied, directory-based loading
        (:meth:`AutoModel.from_pretrained` against a path) avoids a
        redundant factory call — strongly recommended.
    default_config : ModelConfig or None, optional
        The default config produced when ``pretrained=False``.  When
        supplied, :meth:`AutoConfig.from_pretrained` returns it without
        instantiating the model.
    params : int or None, optional
        Paper-cited trainable-parameter count for this factory's default
        config (e.g. ``61_100_840`` for ``alexnet_cls``).  Surfaced on
        the API docs site as a ``"61.1M"``-style tag on the factory
        card and as a "Model Size" section on the detail page.  Omit
        (``None``) when no authoritative count is available — the UI
        then hides the tag rather than guessing.
    summary : object, default="auto"
        Layer-tree summary used by the docs site's collapsible
        "Model Size" section.  ``"auto"`` (the canonical value) tells
        ``tools/build_model_summaries.py`` to introspect the model via
        ``shadow_alloc`` and emit a real tree.  Pass an already-built
        summary dict to override the introspection (rare — used only
        when shadow-alloc introspection fails for a model family).

    Returns
    -------
    Callable[[ModelFactory], ModelFactory]
        Decorator that registers the factory and returns it wrapped, so the
        model it builds carries the registered name as ``name_or_path``.

    Raises
    ------
    ValueError
        If a model is already registered under the same normalised name.

    Notes
    -----
    The decorator uses the factory's ``__name__`` (normalised through
    :func:`_normalize`) as the registry key.  Same name twice raises —
    this catches accidental shadowing across families.

    Examples
    --------
    >>> from lucid.models import register_model, PretrainedModel
    >>> # Inside lucid/models/vision/myfamily/_pretrained.py:
    >>> @register_model(
    ...     task="image-classification",
    ...     family="myfamily",
    ...     model_type="myfamily",
    ... )
    ... def myfamily_small(pretrained: bool = False, **overrides):
    ...     cfg = MyFamilyConfig(depth=12, **overrides)
    ...     return MyFamilyForImageClassification(cfg)
    """

    def decorator(fn: ModelFactory) -> ModelFactory:
        name = _normalize(fn.__name__)
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} already registered")

        if family is None:
            # Module path looks like ``lucid.models.vision.resnet._pretrained``;
            # the second-to-last component is the family name.
            parts = fn.__module__.split(".")
            family_resolved = parts[-2] if len(parts) >= 2 else fn.__module__
        else:
            family_resolved = family

        mt = model_type if model_type is not None else family_resolved

        _REGISTRY[name] = _RegistryEntry(
            factory=_naming(fn, name),
            task=task,
            family=family_resolved,
            model_type=mt,
            model_class=model_class,
            default_config=default_config,
            params=params,
            summary=summary,
        )
        return _REGISTRY[name].factory

    return decorator


def list_models(*, task: str | None = None, family: str | None = None) -> list[str]:
    r"""List all registered model names, optionally filtered by task / family.

    Parameters
    ----------
    task : str or None, optional, keyword-only
        Restrict to entries with this task tag (e.g.
        ``"image-classification"``).  ``None`` returns all tasks.
    family : str or None, optional, keyword-only
        Restrict to entries from this architecture family (e.g.
        ``"resnet"``).  ``None`` returns all families.

    Returns
    -------
    list[str]
        Sorted list of normalised model names matching the filters.

    Notes
    -----
    Use this for discovery — pair with :func:`create_model` to build any
    listed name.  The returned strings are the canonical (normalised)
    forms; either ``"resnet_50"`` or ``"ResNet-50"`` is accepted as input
    elsewhere, but the listing always returns the canonical spelling.

    Examples
    --------
    >>> from lucid.models import list_models
    >>> list_models(family="resnet")[:3]
    ['resnet_101', 'resnet_101_cls', 'resnet_152']

    A backbone and the classifier built on it are separate entries under
    separate names, so filtering by task returns the ``_cls`` one and not
    the bare architecture.

    >>> "vit_base_16" in list_models(task="image-classification")
    False
    >>> "vit_base_16_cls" in list_models(task="image-classification")
    True
    """
    out: list[str] = []
    for name, entry in _REGISTRY.items():
        if task is not None and entry.task != task:
            continue
        if family is not None and entry.family != family:
            continue
        out.append(name)
    return sorted(out)


def is_model(name: str) -> bool:
    r"""Return whether ``name`` is registered (case / separator insensitive).

    Parameters
    ----------
    name : str
        Candidate model name.

    Returns
    -------
    bool
        ``True`` if a registry entry exists under the normalised name.

    Examples
    --------
    >>> from lucid.models import is_model
    >>> is_model("resnet_50")
    True
    >>> is_model("ResNet-50")
    True
    >>> is_model("not_a_real_model")
    False
    """
    return _normalize(name) in _REGISTRY


def model_entrypoint(name: str) -> ModelFactory:
    r"""Return the registered factory callable for ``name``.

    Parameters
    ----------
    name : str
        Registered model name (any case, hyphens or underscores).

    Returns
    -------
    ModelFactory
        The factory function decorated with :func:`register_model`.

    Raises
    ------
    ValueError
        If no entry is registered under the normalised name; the error
        message includes up to three Levenshtein-near suggestions.

    Notes
    -----
    Task is **not** filtered here — :func:`model_entrypoint` returns the
    factory regardless of which Auto class it belongs to.  Use
    :func:`_registry_lookup` internally when you need task-filtered
    dispatch.

    Examples
    --------
    >>> from lucid.models import model_entrypoint
    >>> factory = model_entrypoint("resnet_50")
    >>> model = factory(pretrained=False)
    """
    norm = _normalize(name)
    entry = _REGISTRY.get(norm)
    if entry is None:
        raise ValueError(_unknown_model_message(name, list(_REGISTRY.keys())))
    return entry.factory


def create_model(name: str, *, pretrained: bool = False, **overrides: object) -> PretrainedModel:
    r"""Look up ``name`` and call its factory — the timm-style entry point.

    Parameters
    ----------
    name : str
        Registered model name (case-insensitive, ``-`` / ``_``
        interchangeable).
    pretrained : bool, optional, keyword-only, default=False
        Whether to download and load registered checkpoint weights.
    **overrides : object
        Forwarded as keyword arguments to the underlying factory.  Most
        factories pipe these into the ``ModelConfig`` constructor so
        callers can adjust hyper-parameters at creation time without
        defining a custom config (e.g. ``num_classes=10`` for transfer
        learning).

    Returns
    -------
    PretrainedModel
        A fully constructed concrete model.

    Raises
    ------
    ValueError
        If ``name`` is not registered.

    Notes
    -----
    Equivalent to ``model_entrypoint(name)(pretrained=pretrained,
    **overrides)``.  For task-aware dispatch with the Auto* shells, see
    :class:`AutoModel` and the ``AutoModelFor*`` family.

    Examples
    --------
    >>> from lucid.models import create_model
    >>> # CIFAR-10 transfer head — random init, 10 classes
    >>> model = create_model("resnet_50", num_classes=10)
    >>> model.config.num_classes
    10
    """
    factory = model_entrypoint(name)
    return factory(pretrained=pretrained, **overrides)


def _registry_lookup(name: str, *, task: str) -> _RegistryEntry:
    r"""Resolve ``name`` and require the entry's task to equal ``task``.

    Parameters
    ----------
    name : str
        Registered model name.
    task : str, keyword-only
        Required task tag.

    Returns
    -------
    _RegistryEntry
        The matching registry entry.

    Raises
    ------
    ValueError
        If the name is unknown, or if the entry's task does not match.

    Notes
    -----
    Used internally by :class:`_BaseAutoClass.from_pretrained` so that
    ``AutoModelForImageClassification.from_pretrained("gpt")`` raises a
    clean error rather than returning an inappropriate model.
    """
    norm = _normalize(name)
    entry = _REGISTRY.get(norm)
    if entry is None:
        raise ValueError(_unknown_model_message(name, list(_REGISTRY.keys())))
    if entry.task != task:
        raise ValueError(
            f"Model {name!r} is registered for task {entry.task!r}, "
            f"but task {task!r} was requested. Use the matching Auto class."
        )
    return entry


def _registry_resolve_saved(
    task: str, model_type: str, architectures: Sequence[str] | None
) -> _RegistryEntry:
    r"""Resolve the registry entry a saved ``config.json`` describes.

    ``(task, model_type)`` alone does not say which class was saved: a family
    registers several heads under one task (``BERTForQuestionAnswering`` and
    ``BERTForSequenceClassification`` are both ``sequence-classification``
    ``bert``), and the first of them is not the one on disk.  The
    ``architectures`` that :meth:`PretrainedModel.save_pretrained` records name
    the class, and decide.

    Parameters
    ----------
    task : str
        Task tag the calling Auto class requires.
    model_type : str
        ``model_type`` of the saved config.
    architectures : sequence of str or None
        Class names the saved config lists.  ``None`` for a directory that
        predates the key: the first entry for ``(task, model_type)``.

    Returns
    -------
    _RegistryEntry
        The first entry of ``task`` and ``model_type`` whose class is the
        first of ``architectures`` that any such entry has.

    Raises
    ------
    ValueError
        When nothing is registered for ``(task, model_type)``, or when
        ``architectures`` names no class registered for it — the message says
        which task the named class belongs to, if it is registered at all.
    """
    of_type = [e for e in _REGISTRY.values() if e.model_type == model_type]
    of_task = [e for e in of_type if e.task == task]
    if not of_task:
        raise ValueError(f"No model registered for task={task!r}, model_type={model_type!r}")
    if architectures is None:
        return of_task[0]

    def class_name(entry: _RegistryEntry) -> str:
        # An entry registered without ``model_class`` costs one factory call to learn its class.
        cls = entry.model_class if entry.model_class is not None else type(entry.factory())
        return cls.__name__

    for architecture in architectures:
        for entry in of_task:
            if class_name(entry) == architecture:
                return entry
    elsewhere = sorted(
        {e.task for e in of_type if e.model_class is not None and class_name(e) in architectures}
    )
    hint = (
        f"; it is registered for task {elsewhere[0]!r} — use the matching Auto class"
        if elsewhere
        else ""
    )
    raise ValueError(
        f"The saved architectures {list(architectures)!r} name no class registered for "
        f"task={task!r}, model_type={model_type!r}{hint}"
    )


def _unknown_model_message(name: str, candidates: list[str]) -> str:
    r"""Build a "did you mean" error message for an unknown model name.

    Parameters
    ----------
    name : str
        The (unrecognised) name the caller supplied.
    candidates : list[str]
        All registered names; the function scores each by Levenshtein
        distance to ``_normalize(name)``.

    Returns
    -------
    str
        Error message with up to three suggestions whose edit distance
        is at most 3, sorted by closeness.

    Notes
    -----
    The threshold is conservative so that, say, ``"resnet50"`` suggests
    ``"resnet_50"`` (distance 1) but unrelated names don't pollute the
    output.
    """
    norm = _normalize(name)
    nearby = sorted(((_edit_distance(c, norm), c) for c in candidates), key=lambda t: t[0])
    suggestions = [c for d, c in nearby[:3] if d <= 3]
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return f"Unknown model {name!r}.{hint}"


def _edit_distance(a: str, b: str) -> int:
    r"""Compute the Levenshtein edit distance between two strings.

    Parameters
    ----------
    a, b : str
        Input strings.

    Returns
    -------
    int
        Minimum number of single-character insertions, deletions, or
        substitutions required to transform ``a`` into ``b``.

    Notes
    -----
    Standard two-row dynamic programming implementation — O(len(a) *
    len(b)) time, O(min(len(a), len(b))) memory.  Used by
    :func:`_unknown_model_message` to suggest near-misses.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev: list[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr: list[int] = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]
