"""Steps taken on a base model before it is trained with an adapter: freezing, dtypes, shifted labels."""

from typing import Final

import lucid
import lucid.metal
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.nn.quantized.dynamic as nnqd
from lucid._tensor.tensor import Tensor

#: Attribute of a model that names an external quantization method lucid.peft does not support.
_UNSUPPORTED_QUANTIZATION_METHODS: Final = frozenset({"gptq", "aqlm", "eetq", "hqq"})
_QUANTIZED_MODULE_TYPES: Final = (nnq.QuantizedLinearMLX, nnq.Linear, nnqd.Linear)
_HALF_PRECISION: Final = (lucid.float16, lucid.bfloat16)


def _unsupported_quantization_of(model: nn.Module) -> str | None:
    """Which unsupported third-party quantization ``model`` reports (``"gptq"``, ``"hqq"``), or ``None``."""
    method = getattr(model, "quantization_method", None)
    if method in _UNSUPPORTED_QUANTIZATION_METHODS:
        return str(method)
    if getattr(model, "hqq_quantized", False):
        return "hqq"
    return None


def _is_quantized_model(model: nn.Module) -> bool:
    """Whether ``model`` is quantized via Lucid's quantized layers or ``is_loaded_in_4bit`` / ``is_loaded_in_8bit``."""
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        return True
    return any(isinstance(m, _QUANTIZED_MODULE_TYPES) for m in model.modules())


def _enable_input_require_grads(model: nn.Module) -> None:
    """Ensure the model's embedding output (or first floating-point input) requires gradients for checkpointing."""
    enable = getattr(model, "enable_input_require_grads", None)
    if callable(enable):
        enable()
        return
    get_emb = getattr(model, "get_input_embeddings", None)
    emb = get_emb() if callable(get_emb) else None
    if isinstance(emb, nn.Module):

        def _make_inputs_require_grad(
            _module: object, _args: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]
        ) -> None:
            if isinstance(output, Tensor) and output.is_floating_point():
                output.requires_grad_(True)

        emb.register_forward_hook(_make_inputs_require_grad)
        return

    def _require_grad_pre_hook(_module: object, args: tuple[Tensor, ...]) -> None:
        if args and args[0].is_floating_point():
            args[0].requires_grad_(True)

    model.register_forward_pre_hook(_require_grad_pre_hook)


def cast_mixed_precision_params(model: nn.Module, dtype: lucid.dtype) -> None:
    r"""Cast the model's frozen parameters to ``dtype`` and its trainable ones to float32.

    For half-precision mixed-precision training with an adapter: the frozen base
    weights, which are most of the model, are kept in the smaller type to save memory,
    and the adapter's weights stay in full precision, which keeps their updates stable.
    The change is in place.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are cast.
    dtype : lucid.dtype
        The type for the parameters that do not require a gradient — ``lucid.float16``
        or ``lucid.bfloat16``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, cast_mixed_precision_params, get_peft_model
    >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
    >>> cast_mixed_precision_params(peft, lucid.float16)
    >>> peft.get_base_model()[0].base_layer.weight.dtype
    lucid.float16
    >>> peft.get_base_model()[0].lora_A["default"].weight.dtype
    lucid.float32
    """
    for param in model.parameters():
        param.data = param.data.to(lucid.float32 if param.requires_grad else dtype)


def shift_tokens_right(
    input_ids: Tensor, pad_token_id: int | None, decoder_start_token_id: int
) -> Tensor:
    r"""Shift token ids one place to the right, to make the decoder inputs of labels.

    The first column becomes ``decoder_start_token_id``, every other column is the one
    before it in ``input_ids``, and the ``-100`` that marks a position to ignore in a
    label is replaced by ``pad_token_id``.  ``input_ids`` is not changed.

    Parameters
    ----------
    input_ids : Tensor
        Token ids of shape ``(batch_size, sequence_length)``.
    pad_token_id : int
        The id of the padding token.
    decoder_start_token_id : int
        The id the shifted sequences start with.

    Returns
    -------
    Tensor
        The shifted ids, of the same shape and dtype.

    Raises
    ------
    ValueError
        ``pad_token_id`` is ``None``, or ``input_ids`` is not two-dimensional.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import shift_tokens_right
    >>> labels = lucid.tensor([[5, 6, -100]])
    >>> shift_tokens_right(labels, pad_token_id=0, decoder_start_token_id=2)
    tensor([[2, 5, 6]], dtype=lucid.int64)
    >>> shift_tokens_right(lucid.tensor([[-100, 7]]), pad_token_id=0, decoder_start_token_id=2)
    tensor([[2, 0]], dtype=lucid.int64)
    """
    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined to shift tokens right")
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape (batch_size, sequence_length), got {input_ids.shape}"
        )
    start = lucid.full(
        (input_ids.shape[0], 1),
        decoder_start_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    shifted = lucid.cat([start, input_ids[:, :-1]], dim=1)
    return lucid.where(shifted == -100, pad_token_id, shifted)


def prepare_model_for_kbit_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, object] | None = None,
    auto_clear_cache: bool = True,
) -> nn.Module:
    r"""Freeze the base model, upcast its half-precision parameters to float32, and prepare quantized models for QLoRA.

    The preparation step run before training an adapter on a base model loaded for
    low-memory or k-bit training: every parameter stops requiring a gradient (the adapter
    adds its own), and each ``float16`` or ``bfloat16`` parameter becomes ``float32``, so
    that the layer norms, embeddings, and output head compute at full precision.  When the
    model is quantized (:class:`~lucid.nn.quantized.QuantizedLinearMLX`,
    :class:`~lucid.nn.quantized.Linear`, :class:`~lucid.nn.quantized.dynamic.Linear`, or
    ``is_loaded_in_4bit`` / ``is_loaded_in_8bit``) and ``use_gradient_checkpointing`` is
    ``True``, input embeddings are configured to require gradients and
    ``gradient_checkpointing_enable`` is invoked if the model provides it.  The model is
    changed in place and returned.

    Parameters
    ----------
    model : nn.Module
        The base model.
    use_gradient_checkpointing : bool, default=True
        When ``True`` on a quantized model, enable gradients on the input embeddings and
        call ``model.gradient_checkpointing_enable`` if present.
    gradient_checkpointing_kwargs : dict, optional
        Keyword arguments forwarded to ``model.gradient_checkpointing_enable``; defaults to
        ``{"use_reentrant": False}`` when ``None``.
    auto_clear_cache : bool, default=True
        Release the Metal allocator's cached buffers after the casts, which return the
        memory the half-precision copies held.

    Returns
    -------
    nn.Module
        ``model`` itself.

    Raises
    ------
    NotImplementedError
        ``model`` reports an external quantization scheme (``gptq``, ``aqlm``, ``eetq``, ``hqq``)
        that ``lucid.peft`` does not support.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import prepare_model_for_kbit_training
    >>> net = nn.Sequential(nn.Linear(4, 4)).to(lucid.float16)
    >>> net = prepare_model_for_kbit_training(net)
    >>> net[0].weight.dtype, net[0].weight.requires_grad
    (lucid.float32, False)
    """
    unsupported = _unsupported_quantization_of(model)
    if unsupported is not None:
        raise NotImplementedError(
            f"prepare_model_for_kbit_training: the model reports {unsupported}; lucid.peft does "
            "not support this quantization method"
        )
    is_quantized = _is_quantized_model(model)
    model.requires_grad_(False)
    for param in model.parameters():
        if param.dtype in _HALF_PRECISION:
            param.data = param.data.to(lucid.float32)
    if is_quantized and use_gradient_checkpointing:
        _enable_input_require_grads(model)
        enable_gc = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable_gc):
            gc_kwargs = (
                {"use_reentrant": False}
                if gradient_checkpointing_kwargs is None
                else gradient_checkpointing_kwargs
            )
            enable_gc(gradient_checkpointing_kwargs=gc_kwargs)
    if auto_clear_cache:
        lucid.metal.empty_cache()
    return model
