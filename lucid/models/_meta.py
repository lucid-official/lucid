"""Decorator-based metadata for model-family Config classes.

Each family's ``ModelConfig`` subclass is wrapped with
``@model_family_meta(canonical_name=..., citation=..., theory=...)`` to attach
the three docs-facing fields used by the API docs site:

- ``canonical_name`` — paper-style display name (``"ResNet"``, ``"BERT"``)
- ``citation``       — MLA-formatted citation of the defining paper
- ``theory``         — rST-formatted theoretical description (math OK)

The decorator stores the metadata on the class as ``__model_family_meta__``
for runtime introspection.  The docs build script parses the decorator
statically from each family's ``_config.py`` source — it does not rely on
importing the class — so the metadata is the single source of truth for
both runtime and docs.

Convention rationale: see ``arch-models-canonical-name``.
"""

from dataclasses import dataclass, is_dataclass, field, fields
from typing import Callable, dataclass_transform

from lucid.models._base import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelFamilyMeta:
    r"""Immutable container for a family's docs-facing metadata.

    Attributes
    ----------
    canonical_name : str
        Paper-style family display name (e.g. ``"ResNet"``,
        ``"ConvNeXt"``, ``"BERT"``).
    citation : str
        MLA-formatted citation of the family's defining paper.
    theory : str
        rST-formatted theoretical description.  May include
        ``:math:`...`` inline math and ``.. math::`` blocks; rendered
        through the same pipeline as docstring extended-description
        sections on the API docs site.

    Examples
    --------
    >>> from lucid.models._meta import ModelFamilyMeta
    >>> meta = ModelFamilyMeta(
    ...     canonical_name="ResNet",
    ...     citation="He et al., CVPR 2016",
    ...     theory="Residual connections let depth stop hurting.",
    ... )
    >>> meta.canonical_name
    'ResNet'

    Attached to a family's config by the decorator rather than written
    into each factory, so the docs site reads one source instead of
    however many entries the family registers.
    """

    canonical_name: str
    citation: str
    theory: str


@dataclass_transform(frozen_default=True, field_specifiers=(field,))
def model_family_meta[C: type[ModelConfig]](
    *, canonical_name: str, citation: str, theory: str
) -> Callable[[C], C]:
    r"""Class decorator: attach paper / theory / display metadata to a
    ``ModelConfig`` subclass.

    The decorator stores a :class:`ModelFamilyMeta` instance on the
    decorated class as the ``__model_family_meta__`` attribute, so
    runtime code may introspect family metadata without parsing source.

    The docs site does **not** rely on the runtime attribute — instead
    it parses the decorator call statically from each family's
    ``_config.py`` file (Python ``ast`` module).  This means:

    - Adding a new family is purely a source-level operation; nothing
      needs to be re-imported or re-registered at runtime.
    - The decorator arguments **must be string literals** (or implicit
      concatenations of literals).  f-strings, expressions, and
      references to module-level constants are not recognised by the
      static parser.

    Parameters
    ----------
    canonical_name : str, keyword-only
        Paper-style family display name (``"ResNet"``, ``"BERT"``).
    citation : str, keyword-only
        MLA-formatted citation of the defining paper.  Use ``\"...\"``
        around the title to embed double-quotes.
    theory : str, keyword-only
        rST-formatted theoretical description.  Prefer raw triple-quoted
        strings (``r\"\"\"...\"\"\"``) so LaTeX escapes survive verbatim.

    Returns
    -------
    Callable[[C], C]
        The decorator function.  The returned class is *the same class
        object*, with ``__model_family_meta__`` set.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from lucid.models._base import ModelConfig
    >>> from lucid.models._meta import model_family_meta
    >>> @model_family_meta(
    ...     canonical_name="ResNet",
    ...     citation=(
    ...         "He, Kaiming, et al. \"Deep Residual Learning for Image "
    ...         "Recognition.\" CVPR, 2016, pp. 770-778."
    ...     ),
    ...     theory=r'''
    ...     Deep residual network — adds an identity shortcut around
    ...     every convolution stack:
    ...
    ...     .. math::
    ...
    ...         y = F(x, \{W_i\}) + x
    ...     ''',
    ... )
    ... @dataclass(frozen=True)
    ... class ResNetConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "resnet"
    """

    # Layer-2 contract checks (see arch-models-family-contract).
    # Validate decorator arguments *before* class wrap so misuse raises
    # at module-import time with a clear message instead of producing a
    # silently-malformed JSON.
    for _name, _val in (
        ("canonical_name", canonical_name),
        ("citation", citation),
        ("theory", theory),
    ):
        if not isinstance(_val, str) or not _val.strip():
            raise ValueError(
                f"@model_family_meta: '{_name}' must be a non-empty string "
                f"literal (got {_val!r}).  See "
                f"arch-models-family-contract.md."
            )

    meta = ModelFamilyMeta(canonical_name=canonical_name, citation=citation, theory=theory)

    def _wrap(cls: C) -> C:
        # Per-class structural checks at decorator-application time.
        if not isinstance(cls, type) or not issubclass(cls, ModelConfig):
            raise TypeError(
                f"@model_family_meta can only decorate ModelConfig subclasses; "
                f"got {cls!r}.  See arch-models-family-contract.md."
            )
        if not is_dataclass(cls):
            raise TypeError(
                f"@model_family_meta on {cls.__name__}: class must be a "
                f"@dataclass.  Place @dataclass(frozen=True) BELOW "
                f"@model_family_meta so the dataclass transformation runs first."
            )
        # Frozen-dataclass check — ``fields()`` works on any dataclass, so
        # we inspect via the ``__dataclass_params__`` marker instead.
        params = getattr(cls, "__dataclass_params__", None)
        if params is not None and not getattr(params, "frozen", False):
            raise TypeError(
                f"@model_family_meta on {cls.__name__}: dataclass must be "
                f"frozen — use @dataclass(frozen=True) (see "
                f"arch-models-family-contract.md)."
            )
        if cls.model_type == "base":
            raise TypeError(
                f"@model_family_meta on {cls.__name__}: family Config must "
                f"override 'model_type' ClassVar with a unique family "
                f"identifier (e.g. 'resnet', 'bert')."
            )
        # ``fields(cls)`` access guards against ABC-only declarations that
        # bypass the dataclass transform — accessing it surfaces a clear
        # TypeError early.
        _ = fields(cls)
        cls.__model_family_meta__ = meta
        return cls

    return _wrap
