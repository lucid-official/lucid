r"""Convert additive non-LoRA adapters into LoRA (:func:`convert_to_lora`, :func:`save_as_lora`)."""

import copy
import os
import pathlib
import warnings

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft.tuners._layer import _BaseTunerLayer
from lucid.peft.tuners.lora._config import LoraConfig
from lucid.peft.tuners.miss._layer import Linear as MissLinear, _MissSpec
from lucid.serialization import save_safetensors

_WEIGHTS_NAME = "adapter_model.safetensors"


def _find_cutoff_index(s: Tensor, threshold: float) -> int:
    if s.ndim != 1:
        raise ValueError("Input vector must be 1d.")
    energy = s * s
    cs = lucid.cumsum(energy, dim=0)
    total = float(cs[-1].item())
    if total <= 0.0:
        return 1
    cutoff = threshold * total
    for idx in range(int(cs.shape[0])):
        if float(cs[idx].item()) >= cutoff:
            return idx + 1
    return int(s.shape[0])


def _svd_lowrank_factors(delta_weight: Tensor, rank: int | float) -> tuple[Tensor, Tensor, int]:
    orig_dtype = delta_weight.dtype
    u, s, v = linalg.svd(delta_weight.to(lucid.float32), full_matrices=False)
    effective_rank = rank if isinstance(rank, int) else _find_cutoff_index(s, threshold=rank)
    if effective_rank > u.shape[1]:
        raise ValueError(
            f"The chosen rank {effective_rank} is larger than the weight shape ({u.shape[1]}), "
            "please choose a lower rank."
        )
    lora_b = u[:, :effective_rank] * s[:effective_rank]
    lora_a = v[:effective_rank]
    return (
        lora_a.to(orig_dtype).contiguous(),
        lora_b.to(orig_dtype).contiguous(),
        effective_rank,
    )


def _convert_miss_module_to_lora(
    module: MissLinear, rank: int | float, adapter_name: str = "default"
) -> tuple[Tensor, Tensor, int]:
    spec = module._specs[adapter_name]
    assert isinstance(spec, _MissSpec)
    if spec.mode == "bat":
        return _svd_lowrank_factors(module.get_delta_weight(adapter_name), rank)

    miss_block = module.miss_block[adapter_name]
    assert miss_block is not None
    in_features = module.in_features
    out_features = module.out_features
    orig_dtype = miss_block.dtype
    device = miss_block.device

    miss = miss_block.to(lucid.float32)
    r = int(miss.shape[0])
    if spec.mode == "mini":
        mini_r = spec.miss_mini_r
        miss = miss.repeat(1, out_features // mini_r)

    lora_a = lucid.zeros(r, in_features, device=device, dtype=lucid.float32)
    indices = lucid.arange(in_features, device=device, dtype=lucid.int64)
    ones = lucid.ones(in_features, device=device, dtype=lucid.float32)
    lora_a = lora_a.index_put((indices % r, indices), ones)
    lora_b = miss.mT
    return lora_a.to(orig_dtype).contiguous(), lora_b.to(orig_dtype).contiguous(), r


def _convert_module_to_lora(
    module: _BaseTunerLayer, rank: int | float, adapter_name: str = "default"
) -> tuple[Tensor, Tensor, int]:
    if isinstance(module, MissLinear):
        return _convert_miss_module_to_lora(module, rank, adapter_name)
    return _svd_lowrank_factors(module.get_delta_weight(adapter_name), rank)


def _validate_convertible_model(
    model: nn.Module, rank: int | float, adapter_name: str
) -> PeftConfig | None:
    if isinstance(rank, int) and rank <= 0:
        raise ValueError(f"Expected rank > 0, got {rank}")
    if isinstance(rank, float) and not (0.0 <= rank <= 1.0):
        raise ValueError(f"Expected 0.0 <= rank <= 1.0 when rank is a float, got {rank}")

    configs = getattr(model, "peft_config", None)
    peft_config: PeftConfig | None = (
        configs[adapter_name] if isinstance(configs, dict) and adapter_name in configs else None
    )
    if peft_config is not None and peft_config.peft_type == PeftType.LORA:
        warnings.warn(
            "Model already uses LoRA, converting it to LoRA does not generally make sense.",
            UserWarning,
            stacklevel=3,
        )
    unsupported = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, _BaseTunerLayer) and not mod.supports_lora_conversion(adapter_name)
    ]
    if unsupported:
        raise ValueError(
            "Model does not support conversion to LoRA because the following modules do not: "
            f"{', '.join(unsupported)}"
        )
    supports_fn = getattr(model, "supports_lora_conversion", None)
    if callable(supports_fn) and not supports_fn(adapter_name=adapter_name):
        raise ValueError(f"Model does not support conversion to LoRA for adapter {adapter_name}.")
    return peft_config


def _build_lora_config(peft_config: PeftConfig | None, rank: int | float) -> LoraConfig:
    base_r = rank if isinstance(rank, int) else 1
    if peft_config is None:
        return LoraConfig(r=base_r, lora_alpha=base_r, target_modules=[])

    raw_targets = getattr(peft_config, "target_modules", None)
    targets: str | list[str] | set[str] | None
    if isinstance(raw_targets, str):
        targets = raw_targets
    elif isinstance(raw_targets, (list, set)):
        targets = set()
    else:
        targets = None

    raw_exclude = getattr(peft_config, "exclude_modules", None)
    exclude: str | list[str] | set[str]
    if isinstance(raw_exclude, (list, set)):
        exclude = set(raw_exclude)
    elif isinstance(raw_exclude, str) and raw_exclude:
        exclude = {raw_exclude}
    else:
        exclude = set()

    if isinstance(peft_config, LoraConfig):
        lora_config = copy.deepcopy(peft_config)
        lora_config.r = base_r
        lora_config.lora_alpha = base_r
        lora_config.target_modules = targets
        lora_config.exclude_modules = exclude
        lora_config.rank_pattern = {}
        lora_config.alpha_pattern = {}
        return lora_config

    return LoraConfig(
        r=base_r,
        lora_alpha=base_r,
        target_modules=targets,
        exclude_modules=exclude,
        task_type=peft_config.task_type,
        inference_mode=peft_config.inference_mode,
        base_model_name_or_path=peft_config.base_model_name_or_path,
    )


def _record_converted_layer(
    lora_config: LoraConfig,
    state_dict: dict[str, Tensor],
    name: str,
    peft_prefix: str,
    factors: tuple[Tensor, Tensor, int],
    rank: int | float,
) -> None:
    lora_a, lora_b, effective_rank = factors
    clean_name = name.removeprefix(peft_prefix)
    if effective_rank == 0:
        assert isinstance(lora_config.exclude_modules, set)
        lora_config.exclude_modules.add(clean_name)
        return
    if (effective_rank != rank) or isinstance(lora_config.target_modules, str):
        lora_config.rank_pattern[clean_name] = effective_rank
        lora_config.alpha_pattern[clean_name] = float(effective_rank)
    if isinstance(lora_config.target_modules, set):
        lora_config.target_modules.add(clean_name)
    elif isinstance(lora_config.target_modules, list):
        lora_config.target_modules.append(clean_name)
    state_dict[f"{name}.lora_A.weight"] = lora_a
    state_dict[f"{name}.lora_B.weight"] = lora_b


def _copy_modules_to_save(
    model: nn.Module,
    peft_config: PeftConfig | None,
    lora_config: LoraConfig,
    state_dict: dict[str, Tensor],
    adapter_name: str,
) -> None:
    mts = getattr(peft_config, "modules_to_save", None) if peft_config is not None else None
    if not mts:
        return
    lora_config.modules_to_save = copy.copy(mts)
    for module_name, module in model.named_modules():
        if isinstance(module, ModulesToSaveWrapper) and adapter_name in module.modules_to_save:
            for param_name, param in module.modules_to_save[adapter_name].named_parameters():
                state_dict[f"{module_name}.{param_name}"] = param.data


def convert_to_lora(
    model: nn.Module,
    rank: int | float,
    adapter_name: str = "default",
    progressbar: bool = False,
    compile_kwargs: dict[str, object] | None = None,
) -> tuple[LoraConfig, dict[str, Tensor]]:
    r"""Convert an additive PEFT adapter on ``model`` into a LoRA config and state dictionary.

    Parameters
    ----------
    model : nn.Module
        A PEFT model (or module containing PEFT tuner layers) whose additive delta weights
        can be decomposed into rank-``r`` LoRA factors.
    rank : int or float
        Target LoRA rank (positive integer) or cumulative squared singular-value threshold
        in ``[0.0, 1.0]`` used to pick each layer's rank dynamically.
    adapter_name : str, default="default"
        Name of the adapter to convert.
    progressbar : bool, default=False
        Accepted for signature parity with the reference adapter library.
    compile_kwargs : dict, optional
        Optional keyword arguments forwarded to :func:`lucid.compile` around the per-layer
        SVD conversion helper.

    Returns
    -------
    tuple of (LoraConfig, dict of str to Tensor)
        The converted :class:`LoraConfig` and a LoRA state dict ready to save or load.
    """
    del progressbar
    peft_config = _validate_convertible_model(model, rank, adapter_name)
    peft_prefix = "base_model.model." if hasattr(model, "peft_config") else ""
    lora_config = _build_lora_config(peft_config, rank)

    convert_fn = _convert_module_to_lora
    if compile_kwargs is not None:
        convert_fn = lucid.compile(
            _convert_module_to_lora,
            fullgraph=bool(compile_kwargs.get("fullgraph", False)),
            dynamic=(
                bool(compile_kwargs["dynamic"])
                if compile_kwargs.get("dynamic") is not None
                else None
            ),
        )

    state_dict: dict[str, Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, _BaseTunerLayer) or adapter_name not in module._adapter_names():
            continue
        with lucid.no_grad():
            factors = convert_fn(module, rank=rank, adapter_name=adapter_name)
        _record_converted_layer(lora_config, state_dict, name, peft_prefix, factors, rank)

    if not state_dict:
        raise ValueError("Did not convert a single layer.")
    _copy_modules_to_save(model, peft_config, lora_config, state_dict, adapter_name)
    return lora_config, state_dict


def save_as_lora(
    path: str | os.PathLike[str],
    model: nn.Module,
    rank: int | float,
    adapter_name: str = "default",
    progressbar: bool = False,
    compile_kwargs: dict[str, object] | None = None,
) -> None:
    r"""Convert an additive PEFT adapter on ``model`` into LoRA and save it to ``path``."""
    target_path = pathlib.Path(path)
    if target_path.is_file():
        raise ValueError(f"Provided path ({path!s}) should be a directory, not a file")

    lora_config, state_dict = convert_to_lora(
        model,
        rank=rank,
        adapter_name=adapter_name,
        progressbar=progressbar,
        compile_kwargs=compile_kwargs,
    )
    out_dir = target_path if adapter_name == "default" else target_path / adapter_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors(state_dict, str(out_dir / _WEIGHTS_NAME), metadata={"format": "pt"})
    lora_config.save_pretrained(str(out_dir))
