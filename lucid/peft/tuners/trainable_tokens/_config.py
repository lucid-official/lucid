"""``TrainableTokensConfig``: which rows of an embedding table to train."""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.peft._config import PeftType, _Check, _check_bool, _TargetedConfig


def _check_indices(name: str, value: object) -> list[int]:
    """A list of non-negative ``int`` token ids (``bool`` excluded)."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"`{name}` must be a list of int, got {value!r}")
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"`{name}` holds {item!r}, which is not an int")
        if item < 0:
            raise ValueError(f"`{name}` holds {item}, but token ids are non-negative")
        ids.append(item)
    return ids


@dataclass
class TrainableTokensConfig(_TargetedConfig, peft_type=PeftType.TRAINABLE_TOKENS):
    r"""How :func:`lucid.peft.get_peft_model` trains a few rows of an embedding table.

    Trainable tokens re-train existing token embeddings, or the fresh ones a resized
    vocabulary added, **without training the whole table**: the listed rows are copied
    into a small parameter, the layer reads those rows from it instead of from the
    frozen table, and only the copy is trained and saved,

    .. math:: E'_{t} = \theta_k \text{ if } t \text{ is the } k\text{-th trainable token, else } E_t .

    A file holds ``len(token_indices) x embedding_dim`` numbers whatever the vocabulary's
    size, and the adapted model is the base until the rows move.  Merging writes the rows
    into the table; unmerging puts the originals back.

    Field names, their order, defaults and ``adapter_config.json`` are the reference adapter
    library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    token_indices : list of int
        The ids of the tokens to train (a tokenizer's ``input_ids``).  The closer their number
        is to the vocabulary's, the less this saves.  Must not be empty by the time an adapter
        is built, and must not repeat across the adapters of one layer.
    target_modules : str or list of str, optional
        The embedding (or, for a tied output head, ``nn.Linear``) modules to adapt, matched
        as ``LoraConfig.target_modules`` is.  ``None`` takes the model's input embedding
        (``get_input_embeddings()``), else a module named ``embed_tokens``.
    init_weights : bool, default=True
        ``True``: the rows start as the table's (the adapter is a no-op until trained).
        ``False``: a standard normal sample (for tests).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    modules_to_save : list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.

    Raises
    ------
    TypeError
        A field is not of its type.
    ValueError
        A token id is negative.

    Examples
    --------
    >>> from lucid.peft import TrainableTokensConfig
    >>> config = TrainableTokensConfig(token_indices=[3, 7], target_modules=["wte"])
    >>> config.token_indices, config.peft_type
    ([3, 7], <PeftType.TRAINABLE_TOKENS: 'TRAINABLE_TOKENS'>)
    >>> TrainableTokensConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset(
        {"exclude_modules", "modules_to_save", "layers_to_transform", "layers_pattern"}
    )
    _field_checks: ClassVar[dict[str, _Check]] = {
        "token_indices": _check_indices,
        "init_weights": _check_bool,
    }

    token_indices: list[int] = field(default_factory=list)
    target_modules: str | list[str] | set[str] | None = None
    init_weights: bool = True
    exclude_modules: str | list[str] | set[str] | None = field(default=None, kw_only=True)
    modules_to_save: list[str] | None = field(default=None, kw_only=True)
    layers_to_transform: int | list[int] | None = field(default=None, kw_only=True)
    layers_pattern: str | list[str] | None = field(default=None, kw_only=True)
