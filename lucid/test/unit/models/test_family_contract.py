"""Automatic protocol-conformance tests for the Lucid model zoo.

Walks every family directory under ``lucid/models/<domain>/`` and asserts
that the exported Config class structurally satisfies
``ModelConfigProtocol`` and the public model classes satisfy
``PretrainedModelProtocol``.  Runs on every ``pytest`` invocation — so
any future family that drops a slot is caught the moment its tests run,
without anyone having to remember to invoke ``tools/validate_model_zoo``.

Companion to the static AST validator at ``tools/validate_model_zoo.py``:
the validator catches *file-level* / *decorator-absence* mistakes,
these tests catch *runtime structural* drift (e.g. a refactor that
silently strips ``__model_family_meta__``).

Contract spec: ``obsidian/architecture/arch-models-family-contract.md``
"""

import importlib
import inspect
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest

import lucid
import lucid.weights._loading as weights_loading
from lucid.models import (
    CausalLMOutput,
    ImageClassificationOutput,
    InstanceSegmentationOutput,
    MaskedLMOutput,
    ModelOutput,
    ObjectDetectionOutput,
    QuestionAnsweringOutput,
    Seq2SeqLMOutput,
    SemanticSegmentationOutput,
    SequenceClassificationOutput,
    TokenClassificationOutput,
    create_model,
)
from lucid.models._protocols import ModelConfigProtocol, PretrainedModelProtocol
from lucid.models.multimodal.clip import CLIPZeroShotOutput
from lucid.models.text.bert import BERTForPreTrainingOutput
from lucid.models._registry import _REGISTRY, is_model, list_models, model_entrypoint
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.nn._shadow import shadow_alloc
from lucid.weights import weights_for

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = REPO_ROOT / "lucid" / "models"
DOMAINS = ("vision", "text", "generative", "multimodal")

# Abstract intermediate Config bases — these legitimately keep
# ``model_type == "base"`` and are not themselves family configs.
_INTERMEDIATE_CONFIG_NAMES = {
    "ModelConfig",
    "LanguageModelConfig",
    "DiffusionModelConfig",
    "GenerativeModelConfig",
}


def _discover_families() -> list[tuple[str, str]]:
    families: list[tuple[str, str]] = []
    for domain in DOMAINS:
        dom_dir = MODELS_DIR / domain
        if not dom_dir.is_dir():
            continue
        for fam in sorted(dom_dir.iterdir()):
            if not fam.is_dir() or fam.name.startswith("_"):
                continue
            families.append((domain, fam.name))
    return families


_FAMILIES = _discover_families()


def _import_family(domain: str, family: str):
    mod_path = f"lucid.models.{domain}.{family}"
    return importlib.import_module(mod_path)


@pytest.mark.parametrize("domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES])
def test_family_config_satisfies_protocol(domain: str, family: str) -> None:
    """Every family must export a ``<Family>Config`` class that
    structurally satisfies :class:`ModelConfigProtocol` (model_type
    ClassVar set, ``@model_family_meta`` applied, and is a dataclass)."""
    mod = _import_family(domain, family)
    config_classes = [
        obj
        for name in dir(mod)
        if isinstance(obj := getattr(mod, name), type)
        and name.endswith("Config")
        and name not in _INTERMEDIATE_CONFIG_NAMES
    ]
    assert config_classes, (
        f"family {domain}/{family}: no <Family>Config class exported " f"from __init__.py"
    )
    for cfg in config_classes:
        assert isinstance(cfg, ModelConfigProtocol), (
            f"{cfg.__name__} ({domain}/{family}): does not satisfy "
            f"ModelConfigProtocol.  Required attributes: "
            f"model_type (ClassVar[str]), __model_family_meta__ "
            f"(set by @model_family_meta), __dataclass_fields__ "
            f"(from @dataclass).  See arch-models-family-contract.md."
        )
        # Sanity: not the abstract default.
        assert cfg.model_type != "base", (
            f"{cfg.__name__} ({domain}/{family}): model_type is still "
            f"'base' — must override with a unique family identifier."
        )


@pytest.mark.parametrize("domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES])
def test_family_models_satisfy_protocol(domain: str, family: str) -> None:
    """Every public model class in a family (backbone or task wrapper)
    must satisfy :class:`PretrainedModelProtocol` — i.e. declare
    ``config_class``, ``__init__(self, config)`` and ``forward(...)``.
    """
    mod = _import_family(domain, family)
    model_classes = []
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if not isinstance(obj, type):
            continue
        # Skip Configs, Output dataclasses, and protocols themselves.
        if name.endswith("Config") or name.endswith("Output"):
            continue
        # Heuristic: only classes whose home module is in this family.
        home = getattr(obj, "__module__", "")
        if not home.startswith(f"lucid.models.{domain}.{family}"):
            continue
        # The protocol governs the 5-slot family-canonical classes that
        # take a Config object.  A family may still export a composition
        # primitive that carries no config (``ViTEmbeddings``); skip
        # anything lacking ``config_class``, which the protocol itself
        # requires, and pin the actual model classes.
        if not hasattr(obj, "config_class"):
            continue
        model_classes.append(obj)

    if not model_classes:
        pytest.skip(
            f"family {domain}/{family}: no public model class exported "
            f"— nothing to check (legitimate for re-export-only __init__)."
        )
    for cls in model_classes:
        assert isinstance(cls, PretrainedModelProtocol), (
            f"{cls.__name__} ({domain}/{family}): does not satisfy "
            f"PretrainedModelProtocol.  Required attributes: "
            f"config_class (ClassVar[type]), __init__(self, config), "
            f"forward(...).  See arch-models-family-contract.md."
        )


def test_family_count_matches_directory_scan() -> None:
    """Guards against discovery silently dropping families — e.g. a
    family that doesn't expose anything in ``__init__.py`` would still
    be a directory but invisible to the parametrised tests above."""
    actual = sum(
        1
        for domain in DOMAINS
        if (MODELS_DIR / domain).is_dir()
        for fam in (MODELS_DIR / domain).iterdir()
        if fam.is_dir() and not fam.name.startswith("_")
    )
    assert len(_FAMILIES) == actual, (
        f"family discovery saw {len(_FAMILIES)}, raw directory scan "
        f"sees {actual} — investigate."
    )


def _misspelt_versions(canonical_name: str) -> list[str]:
    """Version tokens in a display name that are not written ``-vN``.

    The zoo writes a family's version as a lowercase ``v`` joined by a
    hyphen — ``Dreamer-v3``, ``MobileNet-v2``, ``PVT-v2`` — so the
    sidebar reads consistently regardless of how each paper styled its
    own title (``DreamerV3``, ``MobileNet V2``, ``PVTv2`` are all in the
    literature).

    Only ``v``-prefixed tokens count.  A bare trailing numeral is part of
    the published name rather than a version suffix — ``GPT-2`` is what
    the model is called, and ``Mask2Former`` is not versioned at all.
    """
    bad: list[str] = []
    for match in re.finditer(r"[Vv]\s*\d+", canonical_name):
        token = match.group()
        preceded_by_hyphen = match.start() > 0 and canonical_name[match.start() - 1] == "-"
        if not (token.startswith("v") and preceded_by_hyphen):
            bad.append(token)
    return bad


@pytest.mark.parametrize("domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES])
def test_family_version_suffix_is_hyphen_v(domain: str, family: str) -> None:
    """Versioned families display as ``<Name>-vN`` on the docs site."""
    module = _import_family(domain, family)
    config = next((getattr(module, n) for n in module.__all__ if n.endswith("Config")), None)
    assert config is not None, f"{domain}/{family} exports no Config"
    meta = getattr(config, "__model_family_meta__", None)
    canonical = getattr(meta, "canonical_name", None) or ""
    bad = _misspelt_versions(canonical)
    assert not bad, (
        f"{domain}/{family} canonical_name {canonical!r} writes its version "
        f"as {bad} — the zoo's convention is '-vN' (lowercase v, hyphen "
        f"joined), e.g. 'Dreamer-v3'."
    )


def test_the_version_check_can_fail() -> None:
    """Guards the test above — every family already conforms, so a check
    that matched nothing would look identical to one that works."""
    assert _misspelt_versions("MobileNet V2") == ["V2"]
    assert _misspelt_versions("DreamerV3") == ["V3"]
    assert _misspelt_versions("PVTv2") == ["v2"]  # right letter, no hyphen
    assert _misspelt_versions("Dreamer-v3") == []
    assert _misspelt_versions("GPT-2") == []  # not a version suffix
    assert _misspelt_versions("Mask2Former") == []


def _stray_exported_functions(module: ModuleType) -> list[str]:
    """Names in ``module.__all__`` that are functions with no business
    being public.

    A family's public functions are its **model factories**, plus the
    occasional **config builder** (EfficientDet exposes one, keyed by the
    compound-scaling coefficient).  Anything else — a training objective,
    a dispatch helper — is an implementation detail of the family.

    Kept as a helper rather than inlined so the test below can be pointed
    at a module built to fail.
    """
    stray: list[str] = []
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if not inspect.isfunction(obj):
            continue
        if is_model(name):
            continue
        if getattr(obj, "__module__", "").endswith("._config"):
            continue
        stray.append(name)
    return stray


@pytest.mark.parametrize("domain,family", _FAMILIES, ids=[f"{d}/{f}" for d, f in _FAMILIES])
def test_family_exports_no_internal_helpers(domain: str, family: str) -> None:
    """A family's ``__all__`` must not leak its own machinery.

    This is a docs bug before it is an API bug.  The site renders a
    family leaf's module-level functions as its factory list, so an
    exported helper appears among the pretrained entries as though you
    could load weights with it — which is how ``free_bits_kl`` was
    spotted sitting next to the twelve DreamerV3 factories.

    Classes are exempt: they render in their own section, so a genuinely
    public building block (the SDE hierarchy) is fine there.
    """
    module = _import_family(domain, family)
    stray = _stray_exported_functions(module)
    assert not stray, (
        f"lucid.models.{domain}.{family} exports {stray} — not registered "
        f"factories and not config builders.  Drop them from __init__.py "
        f"(callers inside the family should import from the private "
        f"module directly), or register them if they really are factories."
    )


def test_the_export_check_can_fail() -> None:
    """Guards the test above — with every family already clean, a check
    that silently matched nothing would pass just as convincingly."""
    fake = ModuleType("lucid.models.generative.fake")

    def free_bits_kl() -> None: ...

    free_bits_kl.__module__ = "lucid.models.generative.fake._objectives"
    fake.free_bits_kl = free_bits_kl  # type: ignore[attr-defined]
    fake.__all__ = ["free_bits_kl"]  # type: ignore[attr-defined]

    assert _stray_exported_functions(fake) == ["free_bits_kl"]


# ── pretrained=True never returns an untrained model ────────────────────────
#
# ``from_pretrained`` resolves a name by calling ``factory(pretrained=True)``
# and trusts whatever comes back.  A factory that accepts the flag and does
# nothing with it therefore hands the caller random weights labelled as
# trained.  Every factory with no weights registered under its own name must
# refuse the flag with the exception ``reject_unavailable_pretrained`` raises
# — unless it reaches a checkpoint some other way (a task head that loads its
# backbone's weights, as ``bert_base_cls`` does).

# The type ``reject_unavailable_pretrained`` raises; pinned against the
# helper itself by ``test_rejection_type_is_the_helpers`` below.
_REJECTION = NotImplementedError


class _CheckpointRequested(Exception):
    """Raised in place of a download, so no test ever fetches weights."""


def _no_download(*args: object, **kwargs: object) -> NoReturn:
    raise _CheckpointRequested


@pytest.fixture
def _downloads_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weights_loading, "download", _no_download)
    monkeypatch.setattr(urllib.request, "urlopen", _no_download)


def _pretrained_outcome(factory: Callable[..., object]) -> str:
    """What ``factory(pretrained=True)`` does: refuses, loads, or ignores.

    Built under ``shadow_alloc`` so even the largest configuration costs
    no memory; the caller must block downloads so "loads" stops at the
    request rather than fetching the file.
    """
    with shadow_alloc():
        try:
            factory(pretrained=True)
        except _REJECTION:
            return "refuses"
        except _CheckpointRequested:
            return "loads"
    return "ignores"


_UNWEIGHTED = sorted(n for n in list_models() if weights_for(n) is None)


@pytest.mark.parametrize("name", _UNWEIGHTED)
def test_pretrained_without_weights_is_refused(name: str, _downloads_blocked: None) -> None:
    outcome = _pretrained_outcome(model_entrypoint(name))
    assert outcome != "ignores", (
        f"{name}(pretrained=True) returned a randomly initialised model.  "
        f"No weights are registered for {name!r}, so the factory must call "
        f"reject_unavailable_pretrained({name!r}) when pretrained is set."
    )


_STALE_PRETRAINED_DOC = re.compile(
    r"reserved for future|currently ignored|currently a no-op", re.IGNORECASE
)
_SAYS_REFUSED = re.compile(r"\braise|\brefuse|NotImplementedError")


def _pretrained_doc(factory: Callable[..., object]) -> str:
    """The ``pretrained`` entry of a factory's Parameters section, or ``""``."""
    doc = (inspect.getdoc(factory) or "") + "\n"
    match = re.search(r"^pretrained\s*:.*\n(?:[ \t]+.*\n?)*", doc, re.MULTILINE)
    return match.group(0) if match else ""


@pytest.mark.parametrize("name", sorted(list_models()))
def test_pretrained_doc_tells_the_truth(name: str, _downloads_blocked: None) -> None:
    """LCD-368: 167 factories documented ``pretrained`` as "reserved for
    future" or "currently ignored" while refusing it — or while loading a
    checkpoint — and 39 more said they loaded weights "when available" for
    a configuration that has none."""
    factory = model_entrypoint(name)
    doc = _pretrained_doc(factory)
    assert not _STALE_PRETRAINED_DOC.search(doc), f"{name}: {doc}"
    if doc and name in _UNWEIGHTED and _pretrained_outcome(factory) == "refuses":
        assert _SAYS_REFUSED.search(
            doc
        ), f"{name}(pretrained=True) raises, but its docstring says:\n{doc}"


def test_the_pretrained_doc_check_can_fail() -> None:
    def stale(pretrained: bool = False) -> None:
        """Build.

        Parameters
        ----------
        pretrained : bool, optional
            If ``True``, loads ImageNet weights when available.
        """

    doc = _pretrained_doc(stale)
    assert doc.startswith("pretrained : bool")
    assert not _SAYS_REFUSED.search(doc)


_DOC_SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}\n", re.MULTILINE)


def _undocumented_params(factory: Callable[..., object]) -> list[str]:
    """Signature parameters with no entry in the Parameters section.

    ``*args`` / ``**kwargs`` are written with their stars, as numpydoc
    does.  A factory with no Parameters section misses all of them.
    """
    doc = inspect.getdoc(factory) or ""
    heads = list(_DOC_SECTION.finditer(doc))
    section = ""
    for i, head in enumerate(heads):
        if head.group(1) == "Parameters":
            stop = heads[i + 1].start() if i + 1 < len(heads) else len(doc)
            section = doc[head.end() : stop]
            break
    stars = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}
    missing: list[str] = []
    for param in inspect.signature(factory).parameters.values():
        entry = stars.get(param.kind, "") + param.name
        if not re.search(rf"^{re.escape(entry)}\s*(:|$)", section, re.MULTILINE):
            missing.append(entry)
    return missing


@pytest.mark.parametrize("name", sorted(list_models()))
def test_factory_documents_every_parameter(name: str) -> None:
    """LCD-411: 23 factories had no Parameters section, so what
    ``pretrained=True`` does — refuse, or load a checkpoint — was written
    nowhere a reader of the docs site could see it."""
    missing = _undocumented_params(model_entrypoint(name))
    assert not missing, f"{name}'s docstring has no Parameters entry for {missing}"


def test_the_param_doc_check_can_fail() -> None:
    def bare(pretrained: bool = False, **overrides: object) -> None:
        """Build.

        Examples
        --------
        >>> pretrained = True
        """

    def partial(pretrained: bool = False, *, weights: object = None) -> None:
        """Build.

        Parameters
        ----------
        pretrained : bool, optional
            Refused.

        Returns
        -------
        weights : object
            Mentioned outside Parameters, which does not count.
        """

    assert _undocumented_params(bare) == ["pretrained", "**overrides"]
    assert _undocumented_params(partial) == ["weights"]


def test_rejection_type_is_the_helpers() -> None:
    with pytest.raises(_REJECTION):
        reject_unavailable_pretrained("probe")


def test_partially_weighted_families_are_covered() -> None:
    """A family with weights for one configuration only is still checked
    for the rest — DiT publishes XL/2 and nothing else."""
    assert "dit_xlarge_2" not in _UNWEIGHTED
    assert {"dit_xlarge_4", "dit_small_2_gen"} <= set(_UNWEIGHTED)


def test_the_pretrained_check_can_fail(_downloads_blocked: None) -> None:
    """Guards the test above — with every factory already conforming, a
    classifier that never said "ignores" would pass just as well."""

    def ignores(pretrained: bool = False) -> object:
        del pretrained  # the defect under test: the flag goes nowhere
        return object()

    def refuses(pretrained: bool = False) -> object:
        if pretrained:
            reject_unavailable_pretrained("refuses")
        return object()

    def loads(pretrained: bool = False) -> object:
        if pretrained:
            weights_loading.download("https://example.invalid/w", "0", name="w")
        return object()

    assert _pretrained_outcome(ignores) == "ignores"
    assert _pretrained_outcome(refuses) == "refuses"
    assert _pretrained_outcome(loads) == "loads"


# ── a task wrapper returns its task's shared output ─────────────────────────
#
# ``lucid.models._tasks`` fixes which output dataclass family each task
# returns, and ``lucid.models._output`` promises that every forward returns a
# ``ModelOutput``.  A family that declares its own output dataclass must
# therefore *inherit* the task's shared one, or ``isinstance`` checks, tuple
# unpacking and dict-style access all stop working for that one family
# (LCD-203: five classifiers returned plain dataclasses).  The static check
# covers every registered factory through its class's ``forward`` return
# annotation — mypy keeps the annotation honest — and the runtime check
# below builds the families that once broke.

_TASK_OUTPUTS: dict[str, tuple[type[ModelOutput], ...]] = {
    "image-classification": (ImageClassificationOutput, CLIPZeroShotOutput),
    "object-detection": (ObjectDetectionOutput, InstanceSegmentationOutput),
    "semantic-segmentation": (SemanticSegmentationOutput,),
    # Pre-training (MLM + NSP) is registered here because NSP is a pooled
    # sentence-pair head; its output carries both objectives.
    "sequence-classification": (
        SequenceClassificationOutput,
        QuestionAnsweringOutput,
        BERTForPreTrainingOutput,
    ),
    "token-classification": (TokenClassificationOutput,),
    "language-modeling": (CausalLMOutput, MaskedLMOutput, Seq2SeqLMOutput),
}


def _output_violation(task: str, model_class: type) -> str | None:
    """Why ``model_class.forward`` breaks its task's output contract."""
    ret = inspect.get_annotations(model_class.forward).get("return")
    allowed = _TASK_OUTPUTS.get(task, (ModelOutput,))
    if isinstance(ret, type) and issubclass(ret, allowed):
        return None
    shown = getattr(ret, "__name__", repr(ret))
    return (
        f"{model_class.__name__}.forward returns {shown}, not one of "
        f"{[c.__name__ for c in allowed]} (task {task!r})"
    )


def _registered_classes() -> list[tuple[str, type]]:
    seen: dict[str, tuple[str, type]] = {}
    for entry in _REGISTRY.values():
        cls = entry.model_class
        if cls is not None:
            seen.setdefault(cls.__name__, (entry.task, cls))
    return [seen[k] for k in sorted(seen)]


@pytest.mark.parametrize(
    "task,model_class", [pytest.param(t, c, id=c.__name__) for t, c in _registered_classes()]
)
def test_forward_returns_the_task_output(task: str, model_class: type) -> None:
    violation = _output_violation(task, model_class)
    assert violation is None, violation


def test_every_registered_factory_names_its_class() -> None:
    """The check above walks ``model_class``; a factory without one would
    escape it silently."""
    missing = sorted(n for n, e in _REGISTRY.items() if e.model_class is None)
    assert not missing, f"factories registered without model_class: {missing}"


def test_the_output_check_can_fail() -> None:
    """Guards the check above with the LCD-203 shape: a plain dataclass
    carrying the right fields but not inheriting the task output."""

    @dataclass
    class LooseOutput:
        logits: object = None

    class Loose:
        def forward(self) -> LooseOutput:
            return LooseOutput()

    class Bare:
        def forward(self) -> int:
            return 0

    class Fine:
        def forward(self) -> ImageClassificationOutput:
            raise NotImplementedError

    assert _output_violation("image-classification", Loose) is not None
    assert _output_violation("base", Bare) is not None
    assert _output_violation("image-classification", Fine) is None
    # One task's shared output does not satisfy another task.
    assert _output_violation("token-classification", Fine) is not None


# Families that once returned a non-``ModelOutput`` (LCD-203), built small
# enough to run: the static check trusts annotations, this one does not.
_SMALL_TEXT: dict[str, object] = {
    "num_labels": 3,
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 50,
}
_RUNTIME_PROBES: list[tuple[str, dict[str, object], tuple[int, ...]]] = [
    ("googlenet_cls", {"num_classes": 4}, (1, 3, 64, 64)),
    ("inception_v3_cls", {"num_classes": 4}, (1, 3, 75, 75)),
    ("inception_v4_cls", {"num_classes": 4}, (1, 3, 75, 75)),
    ("inception_resnet_v2_cls", {"num_classes": 4}, (1, 3, 75, 75)),
    ("xception_cls", {"num_classes": 4}, (1, 3, 71, 71)),
    ("gpt_cls", _SMALL_TEXT, (2, 5)),
    ("gpt2_small_cls", _SMALL_TEXT, (2, 5)),
    ("transformer_base_cls", _SMALL_TEXT, (2, 5)),
    ("transformer_base_token_cls", _SMALL_TEXT, (2, 5)),
]


@pytest.mark.parametrize(
    "name,overrides,shape", _RUNTIME_PROBES, ids=[p[0] for p in _RUNTIME_PROBES]
)
def test_forward_output_is_the_task_output_at_runtime(
    name: str, overrides: dict[str, object], shape: tuple[int, ...]
) -> None:
    model = create_model(name, **overrides).eval()
    is_text = len(shape) == 2
    x = lucid.randint(0, 50, shape) if is_text else lucid.randn(*shape)
    out = model(x)
    allowed = _TASK_OUTPUTS[_REGISTRY[name].task]
    assert isinstance(out, allowed), (name, type(out).__mro__)
    assert out[0] is out.logits  # the tuple protocol ModelOutput provides
