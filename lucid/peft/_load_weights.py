"""``load_peft_weights``: an adapter's tensors, read from its directory without building a model."""

from lucid._tensor.tensor import Tensor
from lucid.peft._config import _adapter_directory
from lucid.peft._peft_model import _adapter_weights, _remapped
from lucid.peft._refusals import NO_HUB, refuse_kwargs


def load_peft_weights(
    model_id: str,
    device: str | None = None,
    key_mapping: dict[str, str] | None = None,
    **hf_hub_download_kwargs: object,
) -> dict[str, Tensor]:
    r"""Read the tensors of a saved adapter, by their file keys.

    What ``PeftModel.load_adapter`` reads before it copies the tensors into a model, for
    code that wants the tensors alone: the keys are the adapter file's
    (``base_model.model.q.lora_A.weight``), with no adapter name in them.

    Parameters
    ----------
    model_id : str
        The directory that holds ``adapter_model.safetensors`` (the one ``save_pretrained``
        wrote).
    device : str, optional
        The device to read the tensors onto; the default device when ``None``.
    key_mapping : dict of str to str, optional
        Renames applied to the keys: the ``base_model.model.`` prefix is set aside, the
        patterns are tried in order with ``re.sub`` on the rest of the key (the first that
        matches wins), and the prefix goes back.
    **hf_hub_download_kwargs : object
        The reference forwards these to its hub download.  ``subfolder`` names a directory
        under ``model_id``; any other key raises ``NotImplementedError``, as there is no hub.

    Returns
    -------
    dict of str to Tensor
        The adapter's tensors.

    Raises
    ------
    FileNotFoundError
        The directory holds no ``adapter_model.safetensors``.
    NotImplementedError
        The directory holds a pickle (``adapter_model.bin``) and no safetensors file, or a
        hub download option was passed.
    ValueError
        ``key_mapping`` is given and a key does not start with ``base_model.``.

    Examples
    --------
    >>> import tempfile
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_peft_model, load_peft_weights
    >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     peft.save_pretrained(directory)
    ...     weights = load_peft_weights(directory)
    >>> sorted(weights)
    ['base_model.model.0.lora_A.weight', 'base_model.model.0.lora_B.weight']
    """
    subfolder = hf_hub_download_kwargs.pop("subfolder", None)
    if subfolder is not None and not isinstance(subfolder, str):
        raise TypeError(f"subfolder must be a str, got {type(subfolder).__name__}")
    refuse_kwargs("load_peft_weights", hf_hub_download_kwargs, NO_HUB)
    return _remapped(_adapter_weights(_adapter_directory(model_id, subfolder), device), key_mapping)
