"""``AutoPeftModel*``: the base model and its adapter, loaded from the adapter's directory.

An adapter directory names the model it was trained on
(``base_model_name_or_path`` in ``adapter_config.json``) and the task
(``task_type``).  ``AutoPeftModelFor<Task>.from_pretrained(directory)``
builds that base model through the zoo's ``AutoModelFor<Task>`` and loads
the adapter onto it, so a fine-tune is reopened with one call::

    from lucid.peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained("my-adapter")

``base_model_name_or_path`` is anything the zoo's Auto class accepts: a
registered model name, or a directory written by
``PretrainedModel.save_pretrained``.  A task is one of the zoo's tasks
(the zoo files causal, masked and encoder-decoder language models under
``language-modeling`` and extractive question answering under
``sequence-classification``), so ``AutoPeftModelFor<Task>`` inherits the
zoo's resolution of a directory by ``(task, model_type)``.

``PeftModel.save_pretrained`` also records the class of the base model
(``auto_mapping``).  When the class is in ``lucid.models`` it is used in
place of the Auto class, which picks the first model of a ``model_type``
and task (a question-answering model's directory would load as the
sequence classifier before it), and an adapter with no task, which has no
Auto class to ask, is loaded by ``AutoPeftModel`` through it alone.
"""

import importlib
import os
from collections.abc import Sequence
from typing import ClassVar, NoReturn, Protocol, override, runtime_checkable

import lucid.nn as nn
from lucid.peft._config import PeftConfig, TaskType, _adapter_directory
from lucid.peft._convert import ConvertMode
from lucid.peft._peft_model import PeftModel
from lucid.peft._task_models import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)


@runtime_checkable
class _ZooLoader(Protocol):
    """What builds a base model from a name or directory: the zoo's Auto and model classes."""

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, strict: bool = True) -> nn.Module: ...


#: Packages an adapter's ``auto_mapping`` may name by default.
_DEFAULT_IMPORT_ALLOWLIST = ("lucid.models",)


def _allowed(package: str, allowlist: Sequence[str]) -> bool:
    return any(package == name or package.startswith(f"{name}.") for name in allowlist)


def _class_named(package: str, name: str) -> type[_ZooLoader]:
    """``package.name``, imported when asked for: a class that builds a model from a name or directory.

    The zoo is not imported with ``lucid.peft``: the Auto classes load it the first time
    one of them is used.
    """
    found = getattr(importlib.import_module(package), name, None)
    if not (isinstance(found, type) and issubclass(found, _ZooLoader)):
        raise TypeError(f"{package}.{name} is not a model class with `from_pretrained`")
    return found


class _BaseAutoPeftModel:
    """Load a base model and its adapter from the adapter's directory; never instantiated.

    A subclass names the zoo's Auto class for its task (``_zoo_auto``) and the
    wrapper class it returns (``_target_peft_class``).  The Auto classes are
    shells — the model is built by the zoo, the adapter by the wrapper — so
    there is nothing to construct.
    """

    #: Name of the ``lucid.models`` class that builds the base model.
    _zoo_auto: ClassVar[str]
    #: The model class returned.
    _target_peft_class: ClassVar[type[PeftModel]] = PeftModel

    def __init__(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs  # accepted so the error below is the one raised, not a TypeError
        raise OSError(
            f"{type(self).__name__} is designed to be instantiated using the "
            f"`{type(self).__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        *,
        subfolder: str | None = None,
        import_allowlist: list[str] | None = None,
        convert: ConvertMode = "auto",
    ) -> PeftModel:
        r"""Build the base model the adapter names and load the adapter onto it.

        Reads ``adapter_config.json`` in ``pretrained_model_name_or_path``,
        builds the base model from its ``base_model_name_or_path`` with the
        zoo's Auto class for the task, and returns
        ``PeftModel.from_pretrained(base, ...)`` — the task's wrapper
        class, the adapter loaded and frozen unless ``is_trainable``.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            A local adapter directory.  Hub ids are not resolved.
        adapter_name : str, default="default"
            Name to give the adapter.
        is_trainable : bool, default=False
            Keep the adapter trainable.  By default it is frozen and the
            model is in ``eval()`` mode.
        config : PeftConfig, optional
            Use this config for the adapter instead of the directory's.  The
            base model is still the one the directory's config names.
        subfolder : str, optional
            The directory under ``pretrained_model_name_or_path`` that holds
            the adapter.
        import_allowlist : list of str, optional
            Only for ``AutoPeftModel``: the packages whose classes the
            adapter's ``auto_mapping`` may name; ``["lucid.models"]`` by
            default.  The mapping is data from a file, and importing what it
            names would run that package's code.
        convert : bool or "auto", default="auto"
            Read an adapter written for the reference library's model in the
            zoo's layout; see :meth:`PeftModel.from_pretrained`.

        Returns
        -------
        PeftModel
            The base model, adapted and loaded.

        Raises
        ------
        ValueError
            The adapter's task is not this class's, the adapter names no base
            model, or (``AutoPeftModel``) it has a task, or no usable
            ``auto_mapping``, or the mapping names a package outside the
            allowlist.
        TypeError
            The ``auto_mapping`` names something that is not a model class.
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.

        Examples
        --------
        >>> import tempfile
        >>> from lucid.models import create_model
        >>> from lucid.peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
        >>> small = dict(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        ...              num_attention_heads=2, intermediate_size=32, max_position_embeddings=16)
        >>> with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as adapter:
        ...     model = create_model("gpt2_small_lm", **small)
        ...     model.save_pretrained(base)
        ...     config = LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM",
        ...                         base_model_name_or_path=base)
        ...     get_peft_model(model, config).save_pretrained(adapter)
        ...     loaded = AutoPeftModelForCausalLM.from_pretrained(adapter)
        >>> type(loaded).__name__
        'PeftModelForCausalLM'
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, subfolder)
        loader = cls._loader_for(peft_config, import_allowlist)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError(
                f"{_adapter_directory(pretrained_model_name_or_path, subfolder)!r} names no base "
                "model (`base_model_name_or_path` is empty); build the base model yourself and "
                "use PeftModel.from_pretrained(model, directory)"
            )
        base_model = loader.from_pretrained(base_name)
        return cls._target_peft_class.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            subfolder=subfolder,
            convert=convert,
        )

    @classmethod
    def _loader_for(
        cls, peft_config: PeftConfig, import_allowlist: Sequence[str] | None
    ) -> type[_ZooLoader]:
        """What builds the adapter's base model, once its task is checked against this class's."""
        task_type = peft_config.task_type
        if task_type is not None:
            expected = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[TaskType(task_type)]
            if cls._target_peft_class is not expected:
                raise ValueError(
                    f"Expected target PEFT class: {expected.__name__}, but you have asked for: "
                    f"{cls._target_peft_class.__name__} make sure that you are loading the "
                    "correct model for your task type."
                )
        mapping = peft_config.auto_mapping
        allowed = _DEFAULT_IMPORT_ALLOWLIST if import_allowlist is None else import_allowlist
        # A mapping into another package (a file the reference wrote) is not ours to follow.
        if (
            mapping
            and {"base_model_class", "parent_library"} <= mapping.keys()
            and _allowed(mapping["parent_library"], allowed)
        ):
            return _class_named(mapping["parent_library"], mapping["base_model_class"])
        return _class_named("lucid.models", cls._zoo_auto)


class AutoPeftModel(_BaseAutoPeftModel):
    r"""Load an adapter that has no task, with its base model.

    ``AutoPeftModel.from_pretrained(directory)`` is for an adapter saved
    with no ``task_type`` — a vision model's, for one, since there is no
    image task — whose ``adapter_config.json`` carries an ``auto_mapping``
    naming the class of the base model.  An adapter with a task is loaded by
    the class of its task (``AutoPeftModelForCausalLM`` and the others).

    The mapping is data read from a file, and building the model imports
    the package it names.  Only packages on ``import_allowlist``
    (``["lucid.models"]`` by default) are imported, and what is found there
    must be a class with ``from_pretrained`` (else ``TypeError``).

    Examples
    --------
    >>> from lucid.peft import AutoPeftModel
    >>> AutoPeftModel()
    Traceback (most recent call last):
        ...
    OSError: AutoPeftModel is designed to be instantiated using the `AutoPeftModel.from_pretrained(pretrained_model_name_or_path)` method.
    """

    @classmethod
    @override
    def _loader_for(
        cls, peft_config: PeftConfig, import_allowlist: Sequence[str] | None
    ) -> type[_ZooLoader]:
        if peft_config.task_type is not None:
            raise ValueError(
                "Cannot use `AutoPeftModel` with a task type, please use a specific class for "
                "your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)"
            )
        mapping = peft_config.auto_mapping
        if mapping is None or {"base_model_class", "parent_library"} - mapping.keys():
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are "
                "loading the correct model for your task type."
            )
        allowed = _DEFAULT_IMPORT_ALLOWLIST if import_allowlist is None else import_allowlist
        parent = mapping["parent_library"]
        if not _allowed(parent, allowed):
            raise ValueError(
                f"The auto mapping wants to import {parent!r} which is not in the import "
                "allowlist. If you are sure that importing this package is safe, specify "
                f"`import_allowlist=[{parent!r}]` in `from_pretrained()`."
            )
        return _class_named(parent, mapping["base_model_class"])


class AutoPeftModelForCausalLM(_BaseAutoPeftModel):
    r"""Load a causal language model's adapter with its base model.

    The base model is built by ``AutoModelForLanguageModeling``; the result
    is a :class:`~lucid.peft.PeftModelForCausalLM`.  The adapter must be for
    ``task_type="CAUSAL_LM"`` or for no task.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForCausalLM
    >>> AutoPeftModelForCausalLM._target_peft_class.__name__
    'PeftModelForCausalLM'
    """

    _zoo_auto = "AutoModelForLanguageModeling"
    _target_peft_class = PeftModelForCausalLM


class AutoPeftModelForSeq2SeqLM(_BaseAutoPeftModel):
    r"""Load an encoder-decoder model's adapter with its base model.

    The base model is built by ``AutoModelForLanguageModeling`` (the zoo's
    task for every language model); the result is a
    :class:`~lucid.peft.PeftModelForSeq2SeqLM`.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForSeq2SeqLM
    >>> AutoPeftModelForSeq2SeqLM._target_peft_class.__name__
    'PeftModelForSeq2SeqLM'
    """

    _zoo_auto = "AutoModelForLanguageModeling"
    _target_peft_class = PeftModelForSeq2SeqLM


class AutoPeftModelForSequenceClassification(_BaseAutoPeftModel):
    r"""Load a sequence classifier's adapter with its base model.

    The base model is built by ``AutoModelForSequenceClassification``; the
    result is a :class:`~lucid.peft.PeftModelForSequenceClassification`.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForSequenceClassification
    >>> AutoPeftModelForSequenceClassification._target_peft_class.__name__
    'PeftModelForSequenceClassification'
    """

    _zoo_auto = "AutoModelForSequenceClassification"
    _target_peft_class = PeftModelForSequenceClassification


class AutoPeftModelForTokenClassification(_BaseAutoPeftModel):
    r"""Load a token classifier's adapter with its base model.

    The base model is built by ``AutoModelForTokenClassification``; the
    result is a :class:`~lucid.peft.PeftModelForTokenClassification`.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForTokenClassification
    >>> AutoPeftModelForTokenClassification._target_peft_class.__name__
    'PeftModelForTokenClassification'
    """

    _zoo_auto = "AutoModelForTokenClassification"
    _target_peft_class = PeftModelForTokenClassification


class AutoPeftModelForQuestionAnswering(_BaseAutoPeftModel):
    r"""Load a question-answering model's adapter with its base model.

    The zoo files extractive question answering under its
    ``sequence-classification`` task, so the base model is built by
    ``AutoModelForSequenceClassification``; the result is a
    :class:`~lucid.peft.PeftModelForQuestionAnswering`.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForQuestionAnswering
    >>> AutoPeftModelForQuestionAnswering._target_peft_class.__name__
    'PeftModelForQuestionAnswering'
    """

    _zoo_auto = "AutoModelForSequenceClassification"
    _target_peft_class = PeftModelForQuestionAnswering


class AutoPeftModelForFeatureExtraction(_BaseAutoPeftModel):
    r"""Load a feature encoder's adapter with its base model.

    The base model is built by ``AutoModel``, the zoo's loader for a
    backbone; the result is a :class:`~lucid.peft.PeftModelForFeatureExtraction`.

    Examples
    --------
    >>> from lucid.peft import AutoPeftModelForFeatureExtraction
    >>> AutoPeftModelForFeatureExtraction._target_peft_class.__name__
    'PeftModelForFeatureExtraction'
    """

    _zoo_auto = "AutoModel"
    _target_peft_class = PeftModelForFeatureExtraction
