"""Reading a saved checkpoint into a model, and saying so briefly when it does not fit.

:meth:`PretrainedModel.from_pretrained` and the ``AutoModelFor*`` classes load the
same directory layout and the same registered names, so they share one reader and one
loader here.  The loader is the one place that decides what a mismatch is -- a weight
of another shape, a key the checkpoint lacks, a key the model has no place for -- and
the one place that words it: a count and the first few names, so a checkpoint of the
wrong class reads as a sentence rather than as hundreds of lines.
"""

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import lucid
from lucid._tensor.tensor import Tensor

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel
    from lucid.models._registry import ModelFactory

#: How many names a message gives before it says "and N more".
_SHOWN: Final = 5

#: A weight whose checkpoint shape and model shape differ: ``(key, checkpoint, model)``.
type _ShapeMismatch = tuple[str, tuple[int, ...], tuple[int, ...]]


def replaced_config(config: ModelConfig, **changes: object) -> ModelConfig:
    """``config`` with ``changes`` applied, which re-runs the config's own validation.

    The one place a frozen config is edited from outside its class: a checkpoint's
    keyword overrides, a resized vocabulary, a swapped head.

    Parameters
    ----------
    config : ModelConfig
        A dataclass config instance.
    **changes : object
        Field replacements.  An unknown field raises ``TypeError``, as a config
        constructor does.

    Returns
    -------
    ModelConfig
        A new config of the same class.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"{type(config).__name__} must be a dataclass instance")
    return replace(config, **changes)


@dataclass(frozen=True)
class Checkpoint:
    """A ``save_pretrained`` directory: its ``config.json`` read, its weights file located.

    Parameters
    ----------
    path : pathlib.Path
        The directory.
    config_file : pathlib.Path
        Its ``config.json``.
    weights_file : pathlib.Path
        ``model.safetensors`` if the directory has one, else ``weights.lucid``.
    document : dict[str, object]
        The JSON object ``config.json`` holds: the config's fields beside
        ``model_type`` and ``architectures``.
    """

    path: Path
    config_file: Path
    weights_file: Path
    document: dict[str, object]

    @classmethod
    def open(cls, path: Path) -> Checkpoint:
        """Locate and read the directory at ``path``.

        Raises
        ------
        FileNotFoundError
            If ``config.json`` is absent, or neither weights file is.
        TypeError
            If ``config.json`` is not a JSON object.
        """
        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        for name in ("model.safetensors", "weights.lucid"):
            if (path / name).exists():
                weights_file = path / name
                break
        else:
            raise FileNotFoundError(
                f"No weights file found in {path}. "
                f"Expected 'model.safetensors' or 'weights.lucid'."
            )
        with open(config_file, encoding="utf-8") as f:
            document = json.load(f)
        if not isinstance(document, dict):
            raise TypeError(f"{config_file} did not contain a JSON object")
        return cls(path, config_file, weights_file, document)

    @property
    def architectures(self) -> list[str] | None:
        """The class names ``config.json`` lists under ``architectures``, or ``None`` without the key."""
        architectures = self.document.get("architectures")
        if architectures is None:
            return None
        if not isinstance(architectures, list) or not all(isinstance(a, str) for a in architectures):
            raise TypeError(f"{self.config_file}: 'architectures' must be a list of class names")
        return architectures

    def load_into[M: PretrainedModel](
        self,
        model_cls: type[M],
        overrides: Mapping[str, object],
        *,
        strict: bool,
        ignore_mismatched_sizes: bool,
    ) -> M:
        """Build ``model_cls`` from the saved config and load the saved weights into it.

        Parameters
        ----------
        model_cls : type[PretrainedModel]
            The class to build.
        overrides : Mapping[str, object]
            Fields of the saved config to change first; an unknown one raises ``TypeError``.
        strict : bool
            Whether missing and unexpected keys are errors.
        ignore_mismatched_sizes : bool
            Keep a mismatched weight's fresh initialisation instead of failing.

        Returns
        -------
        PretrainedModel
            An instance of ``model_cls``, with ``name_or_path`` set to the directory.
        """
        if model_cls.config_class is None:
            raise TypeError(f"{model_cls.__name__} must set config_class before loading")
        config = replaced_config(model_cls.config_class.from_dict(self.document), **overrides)
        model = model_cls(config)
        state = lucid.load(str(self.weights_file), weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Weights file did not contain a state_dict, got {type(state).__name__}")
        load_state_dict_checked(
            model,
            state,
            strict=strict,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            source=str(self.path),
            architectures=self.architectures,
        )
        model.name_or_path = str(self.path)
        return model


def _listed(names: Sequence[str]) -> str:
    """The first few ``names`` and how many more there were."""
    shown = ", ".join(names[:_SHOWN])
    left_out = len(names) - _SHOWN
    return f"{shown}, and {left_out} more" if left_out > 0 else shown


def _shape_mismatches(
    own: Mapping[str, Tensor], state_dict: Mapping[str, Tensor]
) -> list[_ShapeMismatch]:
    """The keys both hold whose shapes differ.

    A 0-d weight takes a ``(1,)`` checkpoint value as its scalar, which is how
    checkpoints from before 0-d tensors stored it; the loader accepts that, so
    this does not call it a mismatch.
    """
    found: list[_ShapeMismatch] = []
    for key, source in state_dict.items():
        target = own.get(key)
        if target is None or not isinstance(source, Tensor):
            continue
        theirs, ours = tuple(source.shape), tuple(target.shape)
        if theirs != ours and not (ours == () and theirs == (1,)):
            found.append((key, theirs, ours))
    return found


def _incompatibility(
    model: PretrainedModel,
    source: str,
    architectures: Sequence[str] | None,
    *,
    sizes: Sequence[_ShapeMismatch],
    missing: Sequence[str],
    unexpected: Sequence[str],
) -> str:
    """One sentence per kind of mismatch: its count and the first few names."""
    name = type(model).__name__
    parts: list[str] = []
    if sizes:
        shown = [f"{k!r} (checkpoint {a}, model {b})" for k, a, b in sizes]
        parts.append(
            f"{len(sizes)} weight(s) differ in shape from the checkpoint: {_listed(shown)}"
        )
    if missing:
        parts.append(
            f"{len(missing)} weight(s) of the model are not in the checkpoint: "
            f"{_listed([repr(k) for k in missing])}"
        )
    if unexpected:
        parts.append(
            f"{len(unexpected)} checkpoint weight(s) match nothing in the model: "
            f"{_listed([repr(k) for k in unexpected])}"
        )
    saved = (
        f" The checkpoint was saved from {', '.join(architectures)}, not {name}."
        if architectures and name not in architectures
        else ""
    )
    hint = (
        " Pass ignore_mismatched_sizes=True to keep a mismatched weight's fresh initialisation."
        if sizes
        else ""
    )
    return f"Error(s) loading {name} from {source}: " + "; ".join(parts) + "." + saved + hint


def load_state_dict_checked(
    model: PretrainedModel,
    state_dict: Mapping[str, Tensor],
    *,
    strict: bool,
    ignore_mismatched_sizes: bool,
    source: str,
    architectures: Sequence[str] | None = None,
) -> None:
    """Load ``state_dict`` into ``model``, refusing in one short message what does not fit.

    A weight whose shape differs from the model's is an error naming the count and
    the first few, unless ``ignore_mismatched_sizes``: then it is left at the
    initialisation the model was built with and reported in a warning, and is not
    counted as missing.  With ``strict``, a key the model has and the checkpoint
    lacks, or the reverse, is an error too; without it they are tolerated.

    Parameters
    ----------
    model : PretrainedModel
        Destination; a mismatch found before loading leaves it untouched.
    state_dict : Mapping[str, Tensor]
        The checkpoint's weights.
    strict : bool
        Whether missing and unexpected keys are errors.
    ignore_mismatched_sizes : bool
        Keep a mismatched weight's fresh initialisation instead of failing.
    source : str
        Where the checkpoint came from (a directory, a registered name), for the messages.
    architectures : sequence of str or None, optional
        The class names the checkpoint's ``config.json`` lists, which a message
        names when none of them is ``model``'s class.

    Raises
    ------
    RuntimeError
        On a mismatch it may not ignore.
    """
    mismatched = _shape_mismatches(model.state_dict(), state_dict)
    if mismatched and not ignore_mismatched_sizes:
        raise RuntimeError(
            _incompatibility(
                model, source, architectures, sizes=mismatched, missing=[], unexpected=[]
            )
        )
    skipped = {key for key, _, _ in mismatched}
    if skipped:
        warnings.warn(
            f"Some weights of {type(model).__name__} were not loaded from {source} because "
            f"their shapes differ, and keep their fresh initialisation: "
            f"{_listed([f'{k!r} (checkpoint {a}, model {b})' for k, a, b in mismatched])}.",
            UserWarning,
            stacklevel=3,
        )
    kept = {key: value for key, value in state_dict.items() if key not in skipped}
    result = model.load_state_dict(kept, strict=False)
    missing = [key for key in result.missing_keys if key not in skipped]
    if strict and (missing or result.unexpected_keys):
        raise RuntimeError(
            _incompatibility(
                model,
                source,
                architectures,
                sizes=[],
                missing=missing,
                unexpected=result.unexpected_keys,
            )
        )


def load_registered(
    factory: ModelFactory,
    name: str,
    overrides: Mapping[str, object],
    *,
    strict: bool,
    ignore_mismatched_sizes: bool,
) -> PretrainedModel:
    """The model a registered factory builds with its published weights.

    With no ``overrides`` the factory loads its own weights.  With some, the
    model is built at the overridden config and takes whatever the published
    weights, which fit the default config, can give it: each weight whose shape
    survived the override.  A transfer head (``num_classes=10``) is the
    ``ignore_mismatched_sizes`` case: every other weight loads and the head keeps
    its fresh initialisation.

    Parameters
    ----------
    factory : ModelFactory
        The registered factory.
    name : str
        Its registered name, for messages.
    overrides : Mapping[str, object]
        Config fields to change; an unknown one raises ``TypeError`` from the
        factory before any weight is fetched.
    strict : bool
        Whether missing and unexpected keys are errors, when there are overrides.
    ignore_mismatched_sizes : bool
        Keep a mismatched weight's fresh initialisation instead of failing, when
        there are overrides.

    Returns
    -------
    PretrainedModel
        The model.
    """
    if not overrides:
        return factory(pretrained=True)
    model = factory(pretrained=False, **overrides)
    published = factory(pretrained=True)
    load_state_dict_checked(
        model,
        published.state_dict(),
        strict=strict,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        source=repr(name),
    )
    return model
