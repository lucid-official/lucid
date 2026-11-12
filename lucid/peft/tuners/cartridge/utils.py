"""Starting a Cartridge from a document, and joining Cartridges: the four utilities that go with ``CartridgeEncoder``.

A Cartridge begins as noise unless it is started from what the model itself makes of a text: run the
model over the document and keep the keys and values of its first tokens, which are what a cache
that stood for the document would hold.  ``initialize_kv_prefix_from_text`` does the whole of it,
``initialize_kv_prefix_from_past_key_values`` starts from a cache you already have, and
``prompt_embeddings_from_past_key_values`` is the packing between them: a cache of one sequence as
the matrix of rows ``(num_virtual_tokens, num_layers * 2 * token_dim)`` an adapter file holds.
``compose_cartridge_adapters`` joins saved Cartridges into one by laying their caches end to end,
which the paper shows serves several documents at once.
"""

import dataclasses
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType, PromptLearningConfig, _output_directory
from lucid.peft._load_weights import load_peft_weights
from lucid.peft._peft_model import _WEIGHTS_NAME, PeftModel
from lucid.peft._refusals import refuse_value
from lucid.peft.tuners._prompt._tuner import _PROMPT_KEY, _PromptLearningTuner
from lucid.peft.tuners.cartridge._config import CartridgeConfig
from lucid.peft.tuners.cartridge._model import CartridgeEncoder
from lucid.peft.tuners.prefix_tuning._model import PrefixEncoder
from lucid.peft.tuners.prompt_tuning._model import _TextEncoder
from lucid.serialization import save_safetensors

#: The configs a prefix-style cache can be loaded into.
_CACHE_TYPES = (PeftType.CARTRIDGE, PeftType.PREFIX_TUNING)
#: The sizes two Cartridges must share to be laid end to end.
_COMPOSABLE_FIELDS = (
    "task_type",
    "token_dim",
    "num_layers",
    "num_attention_heads",
    "num_transformer_submodules",
)


@runtime_checkable
class _LegacyCache(Protocol):
    """A cache object that can give itself as the tuple of ``(key, value)`` pairs, one per layer."""

    def to_legacy_cache(self) -> tuple[tuple[Tensor, ...], ...]: ...


def _legacy_layers(past_key_values: object) -> Sequence[object]:
    """The per-layer entries of a cache: a tuple or list as it is, a cache object through ``to_legacy_cache``, or any iterable of ``(key, value)`` tuples."""
    if isinstance(past_key_values, tuple | list):
        return past_key_values
    if isinstance(past_key_values, _LegacyCache):
        return past_key_values.to_legacy_cache()
    if isinstance(past_key_values, Iterable):
        layers = list(past_key_values)
        if layers and isinstance(layers[0], tuple | list) and len(layers[0]) >= 2:
            return [(layer[0], layer[1]) for layer in layers]
    raise TypeError(
        "Unsupported `past_key_values` type. Expected a legacy tuple/list, an object with "
        "`to_legacy_cache()`, or an iterable of (key, value) tuples."
    )


def _key_value(layer: object) -> tuple[Tensor, Tensor]:
    """One layer's ``(key, value)``, each ``(batch, heads, tokens, head_dim)``."""
    if (
        isinstance(layer, tuple | list)
        and len(layer) == 2
        and isinstance(layer[0], Tensor)
        and isinstance(layer[1], Tensor)
    ):
        return layer[0], layer[1]
    raise TypeError("Each layer of `past_key_values` must be a (key, value) pair of tensors.")


def prompt_embeddings_from_past_key_values(
    past_key_values: object, *, num_virtual_tokens: int
) -> Tensor:
    r"""Pack the first keys and values of a cache into the matrix an adapter file holds.

    Row :math:`t` is the key and the value of token :math:`t` for layer 0, then for layer 1, and so
    on, each of the ``heads * head_dim`` numbers a layer keeps per token; that is the layout prefix
    tuning's and a Cartridge's ``prompt_embeddings`` have, so the matrix can be loaded into either.

    Parameters
    ----------
    past_key_values : object
        The cache of one sequence: a legacy tuple (or list) of ``(key, value)`` per layer, an object
        with ``to_legacy_cache()`` (``lucid.utils.cache.DynamicCache``), or an iterable of
        ``(key, value)`` tuples.  Each tensor is ``(1, heads, tokens, head_dim)``.
    num_virtual_tokens : int
        How many leading tokens to keep.

    Returns
    -------
    Tensor
        ``(num_virtual_tokens, num_layers * 2 * heads * head_dim)``, in the cache's dtype and device.

    Raises
    ------
    TypeError
        ``past_key_values`` is none of the three accepted kinds, or a layer of it is not a pair of tensors.
    ValueError
        The cache is empty, is of a batch (not one sequence), holds fewer than ``num_virtual_tokens``
        tokens, or its layers disagree on heads, head size or length.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import prompt_embeddings_from_past_key_values
    >>> layer = (lucid.zeros(1, 2, 5, 4), lucid.ones(1, 2, 5, 4))
    >>> prompt_embeddings_from_past_key_values((layer, layer), num_virtual_tokens=3).shape
    (3, 32)
    """
    layers = [_key_value(layer) for layer in _legacy_layers(past_key_values)]
    if not layers:
        raise ValueError("Empty `past_key_values`.")
    first = layers[0][0]
    if first.ndim != 4:
        raise ValueError(f"Expected key/value tensors with rank 4, got key.ndim={first.ndim}.")
    if int(first.shape[0]) != 1:
        raise ValueError(
            "This helper expects `past_key_values` from a single-sequence prefill (batch=1). "
            f"Got batch={int(first.shape[0])}."
        )
    heads, tokens, head_dim = (int(n) for n in first.shape[1:])
    if tokens < num_virtual_tokens:
        raise ValueError(f"Need at least {num_virtual_tokens} cached tokens, got {tokens}.")
    rows: list[Tensor] = []
    for key, value in layers:
        for tensor in (key, value):
            if tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1, heads):
                raise ValueError("Inconsistent head shapes across layers in `past_key_values`.")
            if int(tensor.shape[2]) < num_virtual_tokens:
                raise ValueError(
                    "Not enough cached tokens in `past_key_values` for the requested cartridge length."
                )
            if int(tensor.shape[3]) != head_dim:
                raise ValueError("Inconsistent head_dim across layers in `past_key_values`.")
            rows.append(
                tensor[0, :, :num_virtual_tokens, :].permute(1, 0, 2)
            )  # (tokens, heads, head_dim)
    return lucid.stack(rows, dim=1).reshape(num_virtual_tokens, -1)


def _cache_adapter(
    model: PeftModel, adapter_name: str | None
) -> tuple[str, PromptLearningConfig, PrefixEncoder | CartridgeEncoder]:
    """The adapter a cache is loaded into, its config and its encoder, after checking it can hold one."""
    active = model.active_adapters
    if adapter_name is None and len(active) != 1:
        raise ValueError(f"name the adapter to initialise: the model has {active} active")
    name = active[0] if adapter_name is None else adapter_name
    config = model.peft_config[name]
    if config.peft_type not in _CACHE_TYPES or not isinstance(config, PromptLearningConfig):
        raise ValueError(
            f"Adapter '{name}' must be a CARTRIDGE or PREFIX_TUNING adapter (got {config.peft_type})."
        )
    if getattr(config, "prefix_projection", False):
        raise ValueError(
            "Initialization from KV cache is not supported for prefix tuning with `prefix_projection=True`."
        )
    holder = model.base_model
    assert isinstance(holder, _PromptLearningTuner)  # a cache type is a prompt-learning adapter
    encoder = holder.prompt_encoder[name]
    assert isinstance(encoder, PrefixEncoder | CartridgeEncoder)  # what a cache type builds
    return name, config, encoder


def initialize_kv_prefix_from_past_key_values(
    model: PeftModel,
    *,
    adapter_name: str | None = None,
    past_key_values: object,
    num_virtual_tokens: int | None = None,
) -> Tensor:
    """Start a Cartridge (or an unprojected prefix) from the keys and values of a cache you have.

    The first ``num_virtual_tokens`` tokens of the cache of one sequence become the adapter's rows
    (``prompt_embeddings_from_past_key_values``), frozen ones first.

    Parameters
    ----------
    model : PeftModel
        A model with a ``CARTRIDGE`` or ``PREFIX_TUNING`` adapter.
    adapter_name : str, optional
        The adapter to initialise; the active one when left out.
    past_key_values : object
        The cache, in a form ``prompt_embeddings_from_past_key_values`` takes.
    num_virtual_tokens : int, optional
        How many leading tokens to take; the adapter's own count when left out.

    Returns
    -------
    Tensor
        The rows that were loaded.

    Raises
    ------
    ValueError
        The adapter is neither a Cartridge nor a prefix, is a prefix with ``prefix_projection``, or the
        cache does not fit (see ``prompt_embeddings_from_past_key_values``); the adapter's rows are
        untouched then.
    """
    name, config, encoder = _cache_adapter(model, adapter_name)
    count = config.num_virtual_tokens if num_virtual_tokens is None else num_virtual_tokens
    assert count is not None  # the tuner resolved it
    prompt = prompt_embeddings_from_past_key_values(past_key_values, num_virtual_tokens=count)
    encoder.load_prompt_embeddings(prompt)
    return prompt


def _token_ids(
    tokenizer: _TextEncoder, text: str, use_chat_template: bool, max_length: int | None
) -> list[int]:
    """The ids of ``text``: through the tokenizer's chat template as a system message when it has one and it works, else its plain ``encode``."""
    ids: object = None
    template = getattr(tokenizer, "apply_chat_template", None)
    if use_chat_template and callable(template):
        try:
            ids = template(
                [{"role": "system", "content": text}], tokenize=True, add_generation_prompt=False
            )
        except (TypeError, ValueError):
            ids = None  # a tokenizer with no chat template, or with another signature: plain text
    if ids is None:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError(
                "`tokenizer` must offer `encode(text) -> list[int]` (every tokenizer of "
                "lucid.utils.tokenizer does) or an `apply_chat_template`; got "
                f"{type(tokenizer).__name__}"
            )
        ids = encode(text)
    flat = _flat_ids(ids)
    if max_length is not None:
        flat = flat[:max_length]
    if not flat:
        raise ValueError("`text` produced no tokens, so there is no cache to read a prefix from.")
    return flat


def _flat_ids(ids: object) -> list[int]:
    """One sequence of token ids from a list, a ``(1, T)`` list of lists, a tensor or a mapping with ``input_ids``."""
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if isinstance(ids, Tensor):
        ids = ids.reshape(-1).tolist()
    if isinstance(ids, Sequence) and len(ids) == 1 and isinstance(ids[0], Sequence):
        ids = ids[0]
    if isinstance(ids, Sequence) and all(isinstance(i, int) for i in ids):
        return [int(i) for i in ids]
    raise TypeError(
        f"expected one sequence of token ids from the tokenizer, got {type(ids).__name__}"
    )


def initialize_kv_prefix_from_text(
    model: PeftModel,
    tokenizer: _TextEncoder,
    *,
    text: str,
    adapter_name: str | None = None,
    num_virtual_tokens: int | None = None,
    use_chat_template: bool = True,
    max_length: int | None = None,
) -> Tensor:
    """Start a Cartridge from a text: read it with the model, and keep the first keys and values.

    The base model runs over the text with the adapters off (``use_cache=True``), and the first
    ``num_virtual_tokens`` entries of the cache it returns are loaded into the adapter
    (``initialize_kv_prefix_from_past_key_values``).  No tokenizer package is needed: ``tokenizer``
    is any object with ``encode(text) -> list[int]``, as the ones of ``lucid.utils.tokenizer`` are,
    and one that also has ``apply_chat_template`` reads the text as a system message.

    Parameters
    ----------
    model : PeftModel
        A model with a ``CARTRIDGE`` or ``PREFIX_TUNING`` adapter, whose base model returns its
        cache as ``past_key_values`` when called with ``use_cache=True``.
    tokenizer : object with ``encode(text) -> list[int]``
        Turns the text into token ids.
    text : str
        The document, or the first part of it.
    adapter_name : str, optional
        The adapter to initialise; the active one when left out.
    num_virtual_tokens : int, optional
        How many leading tokens to keep; the adapter's own count when left out.
    use_chat_template : bool, default=True
        Read the text through the tokenizer's chat template as a system message when it has one.
    max_length : int, optional
        Cut the token ids to this many before the model reads them.

    Returns
    -------
    Tensor
        The rows that were loaded.

    Raises
    ------
    TypeError
        ``tokenizer`` offers neither ``encode`` nor a chat template, or returns something that is
        not token ids.
    ValueError
        The adapter cannot be initialised from a cache (see
        ``initialize_kv_prefix_from_past_key_values``), the text has no tokens, or the model returned
        no cache.
    """
    name, _, _ = _cache_adapter(model, adapter_name)
    ids = _token_ids(tokenizer, text, use_chat_template, max_length)
    device = next(iter(model.get_base_model().parameters())).device
    input_ids = lucid.tensor([ids], dtype=lucid.int64, device=device)
    with lucid.no_grad(), model.disable_adapter():
        outputs = model(
            input_ids=input_ids, attention_mask=lucid.ones_like(input_ids), use_cache=True
        )
    cache = getattr(outputs, "past_key_values", None)
    if cache is None:
        raise ValueError(
            "the model returned no `past_key_values` for use_cache=True, so there is no cache to "
            "read a prefix from"
        )
    return initialize_kv_prefix_from_past_key_values(
        model, adapter_name=name, past_key_values=cache, num_virtual_tokens=num_virtual_tokens
    )


def compose_cartridge_adapters(
    adapter_paths: Sequence[str | os.PathLike[str]],
    *,
    output_path: str | os.PathLike[str],
    safe_serialization: bool = True,
) -> None:
    """Join saved Cartridges into one by laying their caches end to end.

    The composed cache is every cache one after another, so a model that reads it attends to all the
    documents at once (the paper's composition by concatenation, done at the adapter, with no
    several adapters at run time).  Its config is the first Cartridge's with ``num_virtual_tokens``
    the total; the frozen tokens are the first Cartridge's only, one attention sink for the whole.

    Parameters
    ----------
    adapter_paths : sequence of str or PathLike
        Directories of at least two saved Cartridges.
    output_path : str or PathLike
        The directory to write the composed adapter to; created if missing.
    safe_serialization : bool, default=True
        Write ``adapter_model.safetensors``, the one format here; ``False`` raises
        ``NotImplementedError``.

    Raises
    ------
    ValueError
        Fewer than two paths, an adapter that is not a Cartridge, Cartridges whose sizes differ, or a
        directory without a ``prompt_embeddings`` tensor.
    NotImplementedError
        ``safe_serialization`` is ``False``.
    """
    refuse_value(
        "compose_cartridge_adapters",
        "safe_serialization",
        safe_serialization,
        True,
        "lucid.peft writes safetensors only",
    )
    paths = [os.fspath(path) for path in adapter_paths]
    if len(paths) < 2:
        raise ValueError("Need at least 2 adapters to compose.")
    configs = [PeftConfig.from_pretrained(path) for path in paths]
    for path, config in zip(paths, configs, strict=True):
        if config.peft_type != PeftType.CARTRIDGE:
            raise ValueError(
                f"Adapter at '{path}' is not a CARTRIDGE adapter (got {config.peft_type})."
            )
    base = configs[0]
    assert isinstance(base, CartridgeConfig)  # its peft_type says so
    for other in configs[1:]:
        for field in _COMPOSABLE_FIELDS:
            if getattr(other, field, None) != getattr(base, field, None):
                raise ValueError(f"Incompatible CARTRIDGE configs for attribute '{field}'.")
    caches = []
    for path in paths:
        weights = load_peft_weights(path, device="cpu")
        if _PROMPT_KEY not in weights:
            raise ValueError(f"Adapter at '{path}' holds no `{_PROMPT_KEY}` tensor.")
        caches.append(weights[_PROMPT_KEY])
    composed = lucid.cat(caches, dim=0)
    root = _output_directory(output_path)
    composed_config = dataclasses.replace(
        base, num_virtual_tokens=int(composed.shape[0]), num_frozen_tokens=base.num_frozen_tokens
    )
    composed_config.save_pretrained(root)
    save_safetensors(
        {_PROMPT_KEY: composed}, os.path.join(root, _WEIGHTS_NAME), metadata={"format": "pt"}
    )
