r""":func:`create_arrow_model`: a model that routes over a library of saved LoRA adapters (Arrow)."""

import copy
import os
from collections.abc import Mapping
from typing import TypedDict, Unpack

import lucid.nn as nn
from lucid.peft._config import _CONFIG_NAME, _adapter_directory
from lucid.peft._convert import ConvertMode
from lucid.peft._peft_model import PeftModel
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.lora._arrow import (
    GKS_ADAPTER_PREFIX,
    ROUTER_ADAPTER_NAME,
    TASK_ADAPTER_PREFIX,
)
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.lora._layer import Linear, LoraLayer
from lucid.peft.tuners.lora._subconfigs import ArrowConfig


class _LoaderOptions(TypedDict, total=False):
    """The options both loaders take, which ``create_arrow_model`` hands to each of its loads."""

    is_trainable: bool
    subfolder: str | None
    key_mapping: Mapping[str, str] | None
    convert: ConvertMode
    autocast_adapter_dtype: bool
    ephemeral_gpu_offload: bool
    low_cpu_mem_usage: bool


def _check_source(path: str, subfolder: str | None) -> None:
    """Raise if ``path`` (under ``subfolder``) is not a directory holding an adapter, before anything is loaded."""
    directory = _adapter_directory(path, subfolder)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"adapter directory {directory!r} does not exist (lucid.peft reads local "
            "directories; hub ids are not resolved)"
        )
    if not os.path.isfile(os.path.join(directory, _CONFIG_NAME)):
        raise ValueError(f"Local adapter path '{path}' does not contain '{_CONFIG_NAME}'.")


def _library(model: nn.Module, adapters: list[str]) -> dict[str, dict[str, tuple[int, ...]]]:
    """Each adapter's factors by module, ``{adapter: {module: (r, in_features, out_features)}}``.

    Raises ``TypeError`` for an adapter on a layer an Arrow router cannot route (anything but
    ``nn.Linear``) and ``ValueError`` for one that is not plain LoRA (DoRA, ``lora_bias``).
    """
    found: dict[str, dict[str, tuple[int, ...]]] = {name: {} for name in adapters}
    offenders: list[str] = []
    for key, layer in _tuner_layers(model):
        if not isinstance(layer, LoraLayer):
            continue
        for name in adapters:
            if name not in layer._specs:
                continue
            if not isinstance(layer, Linear):
                offenders.append(
                    f"  - adapter '{name}' on module '{key}' of type {type(layer._root()).__name__}"
                )
                continue
            spec = layer._specs[name]
            if spec.variant != "lora" or spec.lora_bias:
                raise ValueError(
                    f"Arrow routes plain LoRA adapters: adapter '{name}' on module '{key}' "
                    "is DoRA or has lora_bias"
                )
            a, b = layer._factor_pairs(name)[0]
            found[name][key] = (a.shape[0], a.shape[1], b.shape[0])
    if offenders:
        raise TypeError(
            "LoRA adapters must only target nn.Linear layers (the only ones an Arrow router "
            "can route). Found:\n" + "\n".join(offenders)
        )
    return found


def _check_compatibility(model: nn.Module, adapters: list[str]) -> tuple[list[str], int]:
    """The module names and the rank the loaded adapters agree on; ``ValueError`` if they do not share them.

    They must target the same modules, with the same rank and shapes, as the reference requires.
    """
    library = _library(model, adapters)
    reference_name = adapters[0]
    reference = library[reference_name]
    reference_modules = {key.rsplit(".", 1)[-1] for key in reference}
    for name in adapters[1:]:
        modules = {key.rsplit(".", 1)[-1] for key in library[name]}
        if modules != reference_modules or library[name].keys() != reference.keys():
            raise ValueError(
                f"[{name}] target_modules mismatch:\n"
                f"  this adapter -> {sorted(modules)}\n"
                f"  reference   -> {sorted(reference_modules)}"
            )
        ranks, expected = {s[0] for s in library[name].values()}, {s[0] for s in reference.values()}
        if ranks != expected:
            raise ValueError(f"[{name}] rank mismatch: {sorted(ranks)} != {sorted(expected)}")
        if library[name] != reference:
            raise ValueError(f"[{name}] shape mismatch: {library[name]} != {reference}")
    return sorted(reference_modules), max(s[0] for s in reference.values())


def _plan(
    task_paths: list[str],
    general_paths: list[str] | None,
    arrow_config: ArrowConfig,
    options: _LoaderOptions,
) -> tuple[list[str], list[str], list[str]]:
    """The names the adapters are loaded under and every path to read, task adapters first; refuse a bad call before anything is loaded."""
    if not task_paths:
        raise ValueError("`task_specific_adapter_paths` should contain at least one adapter path")
    if arrow_config.use_gks and not general_paths:
        raise ValueError("You should provide general LoRA paths if you want to use GenKnowSub.")
    general = list(general_paths or []) if arrow_config.use_gks else []
    for path in [*task_paths, *general]:
        _check_source(path, options.get("subfolder"))
    return (
        [f"{TASK_ADAPTER_PREFIX}{i}" for i in range(len(task_paths))],
        [f"{GKS_ADAPTER_PREFIX}{i}" for i in range(len(general))],
        [*task_paths, *general],
    )


def create_arrow_model(
    base_model: nn.Module,
    task_specific_adapter_paths: list[str],
    arrow_config: ArrowConfig,
    general_adapter_paths: list[str] | None = None,
    **adapter_kwargs: Unpack[_LoaderOptions],
) -> PeftModel:
    r"""Load saved LoRA adapters into ``base_model`` and add an Arrow router over them.

    Arrow (Ostapenko et al., 2024, arXiv:2405.11157) reuses a library of task-specific adapters:
    at every adapted ``nn.Linear`` a token goes through the ``top_k`` adapters whose
    prototype — the top singular vector of the adapter's update — it is most aligned with,
    combined by a softmax with temperature ``arrow_config.router_temperature``.  With
    ``arrow_config.use_gks`` the mean of the general-knowledge adapters' factors is first
    subtracted from every task adapter's (GenKnowSub, ACL 2025).

    The task adapters are loaded as ``task_0``, ``task_1``, … and the general ones as ``gks_0``,
    ``gks_1``, …; the router is the adapter ``arrow_router``, and it is the one in use.  They
    must target the same modules with the same rank.  The model is changed in place (as
    :meth:`~lucid.peft.PeftModel.from_pretrained` changes it) and is undone, adapters included,
    if anything fails.

    Parameters
    ----------
    base_model : nn.Module
        The model the adapters were trained on.
    task_specific_adapter_paths : list of str
        Local adapter directories (an ``adapter_config.json`` and its weights), at least one.
    arrow_config : ArrowConfig
        The router's options.  Its ``task_adapter_names`` and ``gks_adapter_names`` are filled
        in, as the reference does.
    general_adapter_paths : list of str, optional
        The general-knowledge adapters; required with ``arrow_config.use_gks``.
    **adapter_kwargs
        The options the two loaders share (``is_trainable``, ``subfolder``, ``key_mapping``,
        ``convert`` …), handed to :meth:`~lucid.peft.PeftModel.from_pretrained` for the first
        adapter and to :meth:`~lucid.peft.PeftModel.load_adapter` for the others.  The reference
        also forwards its hub options (``revision``, ``token`` …); here they are refused.

    Returns
    -------
    PeftModel
        ``base_model`` adapted, with the router active.

    Raises
    ------
    ValueError
        No task adapter, ``use_gks`` without general adapters, a directory without an
        ``adapter_config.json``, adapters that do not target the same modules with the same
        rank, or one that is DoRA or has ``lora_bias``.
    FileNotFoundError
        A path is not a directory (hub ids are not resolved).
    TypeError
        An adapter targets a layer that is not ``nn.Linear``.

    Examples
    --------
    >>> import tempfile
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import ArrowConfig, LoraConfig, create_arrow_model, get_peft_model
    >>> def base() -> nn.Module:
    ...     lucid.manual_seed(1)
    ...     return nn.Sequential(nn.Linear(6, 6))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     for task in ("a", "b"):
    ...         config = LoraConfig(r=2, target_modules=["0"], init_lora_weights=False)
    ...         get_peft_model(base(), config).save_pretrained(f"{d}/{task}")
    ...     model = create_arrow_model(
    ...         base(), [f"{d}/a", f"{d}/b"], ArrowConfig(top_k=1, rng_seed=0)
    ...     )
    >>> model.active_adapter, sorted(model.peft_config)
    ('arrow_router', ['arrow_router', 'task_0', 'task_1'])
    >>> tuple(model(lucid.randn(3, 6)).shape)
    (3, 6)
    """
    task_names, general_names, paths = _plan(
        task_specific_adapter_paths, general_adapter_paths, arrow_config, adapter_kwargs
    )
    # The caller's config is filled in once the model is built, as the reference does.
    routed = copy.copy(arrow_config)
    routed.task_adapter_names, routed.gks_adapter_names = task_names, general_names
    model = PeftModel.from_pretrained(
        base_model, paths[0], adapter_name=task_names[0], **adapter_kwargs
    )
    try:
        for name, path in zip([*task_names[1:], *general_names], paths[1:], strict=True):
            model.load_adapter(path, name, **adapter_kwargs)
        modules, rank = _check_compatibility(model.get_base_model(), [*task_names, *general_names])
        router = LoraConfig(arrow_config=routed, target_modules=modules, r=rank)
        model.add_adapter(ROUTER_ADAPTER_NAME, router)
        model.set_adapter(ROUTER_ADAPTER_NAME)
    except BaseException:
        # What the caller passed in goes back as it was, the adapters loaded so far included.
        model.unload()
        raise
    arrow_config.task_adapter_names, arrow_config.gks_adapter_names = task_names, general_names
    return model


__all__ = ["create_arrow_model"]
