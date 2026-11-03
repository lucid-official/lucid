r"""LoRA-GA gradient preprocessing (:func:`preprocess_loraga`)."""

from collections.abc import Callable
import os
import pathlib

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._targeting import _selects
from lucid.peft.tuners._plan import _selected_modules
from lucid.peft.tuners.lora._config import LoraConfig, LoraGAConfig
from lucid.peft.tuners.lora._model import LoraModel


def _target_linears(model: nn.Module, config: LoraConfig) -> list[tuple[str, nn.Linear]]:
    prepared = LoraModel._prepare_config(config, model)
    if prepared.target_modules is None:
        return [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
    selected_names = {name for name, _ in _selected_modules(model, prepared)}
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (name in selected_names or _selects(prepared, name))
    ]


def _load_cached_gradients(
    targets: list[tuple[str, nn.Linear]],
    ga_cfg: LoraGAConfig | None,
    cache_file: str | os.PathLike[str],
) -> None:
    path = pathlib.Path(cache_file)
    sample_device = targets[0][1].weight.device
    loaded = lucid.load(path, map_location=sample_device.type)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a dict of gradients in {path}, got {type(loaded).__name__}")
    for name, module in targets:
        grad = loaded.get(name)
        if isinstance(grad, Tensor):
            module.__dict__["_peft_loraga_grad"] = grad.to(
                device=module.weight.device, dtype=lucid.float32
            )
            module.__dict__["_peft_loraga_config"] = ga_cfg


def _save_cached_gradients(
    targets: list[tuple[str, nn.Linear]], cache_file: str | os.PathLike[str]
) -> None:
    path = pathlib.Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Tensor] = {}
    for name, module in targets:
        grad = module.__dict__.get("_peft_loraga_grad")
        if isinstance(grad, Tensor):
            cached[name] = grad.detach().cpu()
    lucid.save(cached, path)


def _make_grad_counter(counts: dict[int, int], mod_id: int) -> Callable[[Tensor], Tensor | None]:
    def _hook(grad: Tensor) -> Tensor | None:
        del grad
        counts[mod_id] += 1
        return None

    return _hook


def _capture_target_gradients(
    model: nn.Module,
    targets: list[tuple[str, nn.Linear]],
    train_step: Callable[[], None],
    ga_cfg: LoraGAConfig | None,
) -> None:
    counts: dict[int, int] = {id(module): 0 for _, module in targets}
    saved_req_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for _, param in model.named_parameters():
        param.requires_grad = False

    hooks = []
    for _, module in targets:
        module.weight.requires_grad = True
        module.weight.grad = None
        hooks.append(module.weight.register_hook(_make_grad_counter(counts, id(module))))

    try:
        with lucid.enable_grad():
            train_step()
    finally:
        for handle in hooks:
            handle.remove()
        for name, param in model.named_parameters():
            if name in saved_req_grad:
                param.requires_grad = saved_req_grad[name]

    for _, module in targets:
        count = counts[id(module)]
        if count > 0 and module.weight.grad is not None:
            module.__dict__["_peft_loraga_grad"] = module.weight.grad.detach() / count
            module.__dict__["_peft_loraga_config"] = ga_cfg
        module.weight.grad = None


def preprocess_loraga(
    model: nn.Module,
    lora_config: LoraConfig,
    train_step: Callable[[], None],
    cache_file: str | os.PathLike[str] | None = None,
) -> None:
    r"""Estimate and attach per-layer weight gradients on ``model`` before ``get_peft_model(..., LoraConfig(init_lora_weights='lora_ga'))``.

    Parameters
    ----------
    model : nn.Module
        Unadapted base model whose target ``nn.Linear`` layers will be adapted by LoRA-GA.
    lora_config : LoraConfig
        A :class:`LoraConfig` with ``init_lora_weights="lora_ga"``.
    train_step : callable
        Zero-argument callback that runs forward and ``loss.backward()`` on calibration batches.
    cache_file : str or PathLike, optional
        If provided and the file already exists, cached gradients are loaded from it without
        running ``train_step``; otherwise the captured gradients are saved to ``cache_file``.
    """
    if lora_config.init_lora_weights != "lora_ga":
        raise ValueError(
            "`preprocess_loraga` requires `lora_config.init_lora_weights == 'lora_ga'`, "
            f"got {lora_config.init_lora_weights!r}."
        )
    targets = _target_linears(model, lora_config)
    if not targets:
        raise ValueError("No target `nn.Linear` modules matched `lora_config.target_modules`.")

    if cache_file is not None and pathlib.Path(cache_file).exists():
        _load_cached_gradients(targets, lora_config.lora_ga_config, cache_file)
        return

    was_training = model.training
    try:
        _capture_target_gradients(model, targets, train_step, lora_config.lora_ga_config)
    finally:
        if not was_training:
            model.eval()

    if cache_file is not None:
        _save_cached_gradients(targets, cache_file)
