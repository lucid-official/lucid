r"""Explained Variance Adaptation (EVA) data-driven LoRA initialisation."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from copy import copy, deepcopy
from itertools import cycle
import math
import warnings

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn.hooks import RemovableHandle
from lucid.peft._targeting import _pattern_value, _selects
from lucid.peft.tuners._plan import _selected_modules, _set_submodule
from lucid.peft.tuners._spectral import _svd_lowrank
from lucid.peft.tuners.lora._attention import MultiheadAttention
from lucid.peft.tuners.lora._config import EvaConfig, LoraConfig
from lucid.peft.tuners.lora._conv import _ConvLayer
from lucid.peft.tuners.lora._embedding import Embedding
from lucid.peft.tuners.lora._layer import LoraLayer
from lucid.peft.tuners.lora._model import LoraModel

_UNSUPPORTED_LORA_MODULES = (Embedding, MultiheadAttention, _ConvLayer)


class _IncrementalPCA:
    """Streaming incremental PCA / SVD over 2-D activation batches."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.n_features_: int | None = None
        self.n_samples_seen_: int = 0
        self.mean_: Tensor | None = None
        self.var_: Tensor | None = None
        self.components_: Tensor
        self.singular_values_: Tensor
        self.explained_variance_ratio_: Tensor

    def _check_batch(self, x_fp32: Tensor, first_pass: bool) -> tuple[int, int]:
        n_samples, n_features = int(x_fp32.shape[0]), int(x_fp32.shape[1])
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} invalid for n_features={n_features}, "
                "need more columns than components for IncrementalPCA."
            )
        if self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal to the batch number of samples {n_samples}"
            )
        if first_pass:
            self.n_features_ = n_features
        elif n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match the number of features of the first batch."
            )
        return n_samples, n_features

    def partial_fit(self, x: Tensor) -> _IncrementalPCA:
        first_pass = not hasattr(self, "components_")
        x_fp32 = x.to(lucid.float32).clone()
        n_samples, _ = self._check_batch(x_fp32, first_pass)

        col_mean, col_var, n_total = self._incremental_mean_and_var(
            x_fp32, self.mean_, self.var_, self.n_samples_seen_
        )
        if first_pass:
            x_centered = x_fp32 - col_mean
        else:
            assert self.mean_ is not None
            batch_mean = lucid.mean(x_fp32, dim=0)
            x_centered = x_fp32 - batch_mean
            corr_scale = math.sqrt((self.n_samples_seen_ / n_total) * n_samples)
            mean_correction = (self.mean_ - batch_mean).unsqueeze(0) * corr_scale
            x_centered = lucid.cat(
                [
                    self.singular_values_.reshape(-1, 1) * self.components_,
                    x_centered,
                    mean_correction,
                ],
                dim=0,
            )

        q = min(2 * self.n_components, min(int(x_centered.shape[0]), int(x_centered.shape[1])))
        if q >= min(int(x_centered.shape[0]), int(x_centered.shape[1])):
            u, s, vt = linalg.svd(x_centered, full_matrices=False)
        else:
            u, s, v = _svd_lowrank(x_centered, q, 4)
            vt = v.mT
        _, vt = self._svd_flip(u, vt)
        total_var = lucid.sum(col_var * n_total).clamp(min=1e-12)
        explained_variance_ratio = (s * s) / total_var

        self.n_samples_seen_ = n_total
        self.components_ = vt[: self.n_components]
        self.singular_values_ = s[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        return self

    @staticmethod
    def _incremental_mean_and_var(
        x: Tensor, last_mean: Tensor | None, last_variance: Tensor | None, last_sample_count: int
    ) -> tuple[Tensor, Tensor, int]:
        new_count = int(x.shape[0])
        new_sum = lucid.sum(x, dim=0)
        if last_sample_count == 0 or last_mean is None or last_variance is None:
            mean = new_sum / new_count
            diff = x - mean
            var = lucid.mean(diff * diff, dim=0)
            return mean, var, new_count

        last_sum = last_mean * last_sample_count
        updated_count = last_sample_count + new_count
        updated_mean = (last_sum + new_sum) / updated_count
        new_unnorm_var = lucid.sum((x - (new_sum / new_count)) ** 2, dim=0)
        last_unnorm_var = last_variance * last_sample_count
        ratio = last_sample_count / new_count
        corr = (last_sum / ratio - new_sum) ** 2 * (ratio / updated_count)
        updated_var = (last_unnorm_var + new_unnorm_var + corr) / updated_count
        return updated_mean, updated_var, updated_count

    @staticmethod
    def _svd_flip(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        abs_v = lucid.abs(v)
        max_cols = lucid.argmax(abs_v, dim=1)
        rows = lucid.arange(int(v.shape[0]), device=v.device, dtype=lucid.int64)
        vals = v[rows, max_cols]
        signs = lucid.where(vals >= 0, lucid.ones_like(vals), -lucid.ones_like(vals))
        u = u * signs.unsqueeze(0)
        v = v * signs.unsqueeze(1)
        return u, v


class _Hook:
    def __init__(
        self,
        name: str,
        prepare_layer_inputs_fn: Callable[..., Tensor] | None = None,
        gather_distributed_inputs: bool = True,
    ) -> None:
        self.name = name
        self._prepare_layer_inputs_fn = (
            prepare_layer_inputs_fn
            if prepare_layer_inputs_fn is not None
            else self._default_prepare_layer_inputs_fn
        )
        self.gather_distributed_inputs = gather_distributed_inputs
        self.model_input: object = None

    @staticmethod
    def _default_prepare_layer_inputs_fn(
        layer_input: object, model_input: object, layer_name: str
    ) -> Tensor:
        del model_input
        if isinstance(layer_input, (tuple, list)):
            layer_input = layer_input[0]
        if not isinstance(layer_input, Tensor):
            raise TypeError(
                f"unsupported input type {type(layer_input)} for prepare_layer_inputs_fn in layer {layer_name}, "
                "please provide a custom prepare_layer_inputs_fn"
            )
        if layer_input.ndim > 2:
            layer_input = layer_input.reshape(-1, layer_input.shape[-1])
        return layer_input

    def prepare_layer_inputs(self, layer_input: object) -> Tensor:
        return self._prepare_layer_inputs_fn(layer_input, self.model_input, self.name)


class _HashHook(_Hook):
    def __init__(
        self,
        name: str,
        prepare_layer_inputs_fn: Callable[..., Tensor] | None = None,
        gather_distributed_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
        )
        self.hashed_inputs: list[int] = []

    def __call__(
        self, module: object, inputs: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]
    ) -> None:
        del module, output
        states = self.prepare_layer_inputs(inputs)
        flat = states.reshape(-1)
        self.hashed_inputs.append(
            hash(tuple(float(flat[i].item()) for i in range(int(flat.shape[0]))))
        )


class _SVDHook(_Hook):
    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: float | Tensor,
        prepare_layer_inputs_fn: Callable[..., Tensor] | None = None,
        gather_distributed_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
        )
        self.n_components = n_components
        self.sim_thresh = sim_thresh
        if (
            isinstance(sim_thresh, Tensor)
            and sim_thresh.ndim > 0
            and (sim_thresh.ndim != 1 or sim_thresh.shape[0] not in (1, n_components))
        ):
            raise ValueError(
                "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"
            )
        self.svd = _IncrementalPCA(n_components=n_components)
        self.converged = lucid.zeros((n_components,), dtype=lucid.bool)

    def __call__(
        self, module: object, inputs: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]
    ) -> None:
        del module, output
        previous = (
            self.svd.components_.clone().detach() if hasattr(self.svd, "components_") else None
        )
        states = self.prepare_layer_inputs(inputs)
        if states.shape[0] < self.n_components:
            return
        self.svd.partial_fit(states)
        if previous is None:
            return
        components = self.svd.components_
        if components.ndim == 1:
            components = components.reshape(1, -1)
            previous = previous.reshape(1, -1)
        sim = F.cosine_similarity(components, previous, dim=-1)
        thresh = (
            self.sim_thresh.to(sim.device)
            if isinstance(self.sim_thresh, Tensor)
            else self.sim_thresh
        )
        self.converged = (sim >= thresh).to(lucid.bool)


def _find_equal_values(mapping: dict[str, int]) -> list[list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for k, v in mapping.items():
        grouped[v].append(k)
    return [names for names in grouped.values() if len(names) > 1]


def _model_device(model: nn.Module) -> lucid.device | None:
    devices = {p.device for p in model.parameters()}
    if len(devices) == 1:
        return next(iter(devices))
    return None


def _move_inputs_to_device(inputs: object, device: lucid.device) -> object:
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {k: _move_inputs_to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, tuple):
        return tuple(_move_inputs_to_device(v, device) for v in inputs)
    if isinstance(inputs, list):
        return [_move_inputs_to_device(v, device) for v in inputs]
    return inputs


def _prepare_model_inputs_fn_language_modeling(
    model_input: object, peft_config: LoraConfig
) -> Tensor:
    """Extract the boolean non-masked token mask from a language-modeling batch dictionary."""
    if not isinstance(model_input, Mapping):
        raise TypeError(
            "When using `prepare_model_inputs_fn_language_modeling` inputs must be a dictionary"
        )
    eva_cfg = peft_config.eva_config
    mask_value = eva_cfg.label_mask_value if eva_cfg is not None else None
    mask = None
    if (
        eva_cfg is not None
        and eva_cfg.use_label_mask
        and "labels" in model_input
        and mask_value is not None
    ):
        labels = model_input["labels"]
        if isinstance(labels, Tensor):
            mask = lucid.nonzero(labels != mask_value, as_tuple=True)
    if mask is None and "attention_mask" in model_input:
        attn = model_input["attention_mask"]
        if isinstance(attn, Tensor):
            mask = lucid.nonzero(attn != 0, as_tuple=True)
    if mask is not None:
        return mask
    first = next(iter(model_input.values()))
    assert isinstance(first, Tensor)
    return lucid.ones_like(first).to(lucid.bool)


def _prepare_layer_inputs_fn_language_modeling(
    layer_input: tuple[Tensor, ...] | Tensor, model_input: Tensor, layer_name: str
) -> Tensor:
    """Select active token representations from ``layer_input`` using the boolean mask ``model_input``."""
    del layer_name
    if isinstance(layer_input, (tuple, list)):
        x = layer_input[0]
    elif isinstance(layer_input, Tensor):
        x = layer_input
    else:
        raise TypeError(
            f"unsupported input type {type(layer_input)} for prepare_layer_inputs_fn, "
            "please provide a custom prepare_layer_inputs_fn"
        )
    return x[model_input]


def _forward_fn_dict(model: nn.Module, inputs: object) -> object:
    """Default forward callback for EVA: unpack a mapping as keyword arguments, or pass a Tensor directly."""
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def _rank_distribution(
    hooks: dict[str, tuple[_SVDHook, RemovableHandle | None]],
    layer_hook_map: dict[str, str],
    equal_inputs_map: dict[str, str],
    rank_budget: int,
    max_components: dict[str, int],
) -> dict[str, int]:
    exp_vars = {
        k: h.svd.explained_variance_ratio_[: max_components[k]] for k, (h, _) in hooks.items()
    }
    pairs = [(k, c) for k, name in layer_hook_map.items() for c in exp_vars[name]]
    keys = [p[0] for p in pairs]
    values = lucid.stack([p[1] for p in pairs])
    order = lucid.argsort(values, descending=True)
    top_k = min(rank_budget, int(order.shape[0]))
    counts_raw = Counter(keys[int(order[i].item())] for i in range(top_k))
    counts = {k: counts_raw.get(k, 0) for k in layer_hook_map}
    for k, k_hook in equal_inputs_map.items():
        rank, rank_hook = counts[k], counts[k_hook]
        if rank_hook < rank:
            counts[k_hook], counts[k] = rank, rank_hook
    return counts


def _resolve_eva_config_and_targets(
    model: nn.Module, peft_config: LoraConfig | None, adapter_name: str
) -> tuple[LoraConfig, EvaConfig, Callable[[str, nn.Module], bool]]:
    is_peft_model = hasattr(model, "peft_config")
    if peft_config is None:
        if not is_peft_model:
            raise ValueError("if model is not a PeftModel, peft_config is required")
        peft_config = model.__dict__.get("peft_config", getattr(model, "peft_config", {}))[
            adapter_name
        ]
    assert isinstance(peft_config, LoraConfig)
    if peft_config.eva_config is None:
        peft_config = copy(peft_config)
        peft_config.eva_config = EvaConfig()

    if is_peft_model:

        def _is_peft_target(name: str, module: nn.Module) -> bool:
            del name
            return isinstance(module, LoraLayer) and not isinstance(
                module, _UNSUPPORTED_LORA_MODULES
            )

        return peft_config, peft_config.eva_config, _is_peft_target

    prepared = LoraModel._prepare_config(peft_config, model)
    selected_names = (
        {name for name, _ in _selected_modules(model, prepared)}
        if prepared.target_modules is not None
        else set()
    )

    def _is_bare_target(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if peft_config.target_modules is None:
            return True
        return name in selected_names or _selects(prepared, name)

    return peft_config, peft_config.eva_config, _is_bare_target


def _prepare_step_inputs(
    raw_inputs: object,
    device: lucid.device | None,
    prepare_model_inputs_fn: Callable[..., object] | None,
    peft_config: LoraConfig,
) -> tuple[object, object]:
    inputs = _move_inputs_to_device(raw_inputs, device) if device is not None else raw_inputs
    model_inputs = (
        prepare_model_inputs_fn(inputs, peft_config)
        if prepare_model_inputs_fn is not None
        else deepcopy(inputs)
    )
    return inputs, model_inputs


type _ForwardFn = Callable[..., object] | None
type _ModelInputFn = Callable[..., object] | None
type _LayerInputFn = Callable[..., Tensor] | dict[str, Callable[..., Tensor]] | None


class _EvaSession:
    def __init__(
        self,
        model: nn.Module,
        peft_config: LoraConfig,
        eva_cfg: EvaConfig,
        device: lucid.device | None,
        run_forward: Callable[..., object],
        prepare_model_inputs_fn: _ModelInputFn,
    ) -> None:
        self.model = model
        self.peft_config = peft_config
        self.eva_cfg = eva_cfg
        self.device = device
        self.run_forward = run_forward
        self.prepare_model_inputs_fn = prepare_model_inputs_fn
        self.svd_hooks: dict[str, tuple[_SVDHook, RemovableHandle | None]] = {}
        self.layer_hook_map: dict[str, str] = {}
        self.equal_inputs_map: dict[str, str] = {}
        self.max_components: dict[str, int] = {}
        self.rank_budget: int = 0


def _register_hash_hooks(
    session: _EvaSession,
    rho: float,
    is_target: Callable[[str, nn.Module], bool],
    prepare_layer_inputs_fn: _LayerInputFn,
    gather_distributed_inputs: bool,
    model_inputs_for_hooks: object,
) -> dict[str, tuple[_HashHook, RemovableHandle]]:
    layer_fn_map = (
        dict(prepare_layer_inputs_fn) if isinstance(prepare_layer_inputs_fn, Mapping) else None
    )
    hash_hooks: dict[str, tuple[_HashHook, RemovableHandle]] = {}
    for name, module in session.model.named_modules():
        if not is_target(name, module):
            continue
        fn = (
            layer_fn_map.pop(name, None)
            if layer_fn_map is not None
            else (
                prepare_layer_inputs_fn
                if not isinstance(prepare_layer_inputs_fn, Mapping)
                else None
            )
        )
        hook = _HashHook(
            name=name,
            prepare_layer_inputs_fn=fn,
            gather_distributed_inputs=gather_distributed_inputs,
        )
        hook.model_input = model_inputs_for_hooks
        handle = module.register_forward_hook(hook)
        hash_hooks[name] = (hook, handle)
        clean_name = name.removeprefix("base_model.model.")
        layer_rank = _pattern_value(
            session.peft_config.rank_pattern, clean_name, session.peft_config.r
        )
        session.max_components[name] = round(layer_rank * rho)
        session.rank_budget += layer_rank

    if layer_fn_map:
        for _, h_rem in hash_hooks.values():
            h_rem.remove()
        raise ValueError(
            "prepare_layer_inputs_fn is a mapping but the following module names were not found in the model: "
            f"{list(layer_fn_map.keys())}"
        )
    return hash_hooks


def _build_svd_hooks(
    session: _EvaSession,
    hash_hooks: dict[str, tuple[_HashHook, RemovableHandle]],
    gather_distributed_inputs: bool,
) -> None:
    hash_dict = {k: h.hashed_inputs[0] for k, (h, _) in hash_hooks.items() if h.hashed_inputs}
    equal_inputs = _find_equal_values(hash_dict)
    session.equal_inputs_map = {vv: group[0] for group in equal_inputs for vv in group[1:]}
    for names in equal_inputs:
        max_val = max(session.max_components[n] for n in names)
        for n in names:
            session.max_components[n] = max_val

    for name in list(hash_hooks.keys()):
        old_hook, old_handle = hash_hooks.pop(name)
        old_handle.remove()
        if name in session.equal_inputs_map:
            continue
        svd_hook = _SVDHook(
            name=name,
            n_components=session.max_components[name],
            sim_thresh=session.eva_cfg.tau,
            prepare_layer_inputs_fn=old_hook._prepare_layer_inputs_fn,
            gather_distributed_inputs=gather_distributed_inputs,
        )
        module = session.model.get_submodule(name)
        new_handle = module.register_forward_hook(svd_hook)
        session.svd_hooks[name] = (svd_hook, new_handle)

    session.layer_hook_map = {**{k: k for k in session.svd_hooks}, **session.equal_inputs_map}


def _update_hook_convergence(
    session: _EvaSession,
    convergence_dict: dict[str, bool],
    rank_dist: dict[str, int],
    model_inputs_for_hooks: object,
) -> None:
    for name in list(session.svd_hooks.keys()):
        svd_hook, cur_handle = session.svd_hooks[name]
        converged = bool(lucid.all(svd_hook.converged[: rank_dist[name]]).item())
        if (not convergence_dict[name]) and converged and cur_handle is not None:
            cur_handle.remove()
            cur_handle = None
            convergence_dict[name] = True
            session.svd_hooks[name] = (svd_hook, cur_handle)
            continue
        if convergence_dict[name] and not converged:
            module = session.model.get_submodule(name)
            cur_handle = module.register_forward_hook(svd_hook)
            convergence_dict[name] = False
        svd_hook.model_input = model_inputs_for_hooks
        session.svd_hooks[name] = (svd_hook, cur_handle)


def _converge_svd_hooks(session: _EvaSession, batches: list[object]) -> dict[str, int]:
    convergence_dict = {k: False for k in session.svd_hooks}
    rank_dist = session.max_components.copy()
    for step_idx, raw_inputs in enumerate(cycle(batches)):
        inputs, model_inputs_for_hooks = _prepare_step_inputs(
            raw_inputs, session.device, session.prepare_model_inputs_fn, session.peft_config
        )
        _update_hook_convergence(session, convergence_dict, rank_dist, model_inputs_for_hooks)
        if all(convergence_dict.values()):
            break
        session.run_forward(session.model, inputs)
        if not all(hasattr(h.svd, "components_") for h, _ in session.svd_hooks.values()):
            if step_idx + 1 >= len(batches):
                unfit = [
                    k
                    for k, (h, _) in session.svd_hooks.items()
                    if not hasattr(h.svd, "components_")
                ]
                raise ValueError(
                    f"EVA SVD hooks for {unfit} received fewer samples than n_components across all batches."
                )
            continue
        rank_dist = _rank_distribution(
            session.svd_hooks,
            session.layer_hook_map,
            session.equal_inputs_map,
            session.rank_budget,
            session.max_components,
        )
    return rank_dist


def _extract_eva_state_dict(session: _EvaSession, rank_dist: dict[str, int]) -> dict[str, Tensor]:
    eva_state_dict: dict[str, Tensor] = {}
    for name, rank in rank_dist.items():
        svd_hook = session.svd_hooks[session.layer_hook_map[name]][0]
        if rank > 0 and not bool(lucid.all(svd_hook.converged[:rank]).item()):
            raise ValueError(f"Layer {name} has not converged but was assigned rank {rank}.")
        u = svd_hook.svd.components_[:rank].clone()
        if session.eva_cfg.whiten and rank > 0:
            u = u / svd_hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
        eva_state_dict[name] = u.to(session.device) if session.device is not None else u
    return eva_state_dict


def get_eva_state_dict(
    model: nn.Module,
    dataloader: Iterable[object],
    peft_config: LoraConfig | None = None,
    forward_fn: _ForwardFn = _forward_fn_dict,
    prepare_model_inputs_fn: _ModelInputFn = _prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: _LayerInputFn = _prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
) -> dict[str, Tensor]:
    r"""Compute top right singular vectors and redistributed ranks for EVA LoRA initialisation."""
    del show_progress_bar
    batches = list(dataloader)
    if not batches:
        raise ValueError("dataloader is empty")

    peft_config, eva_cfg, is_target = _resolve_eva_config_and_targets(
        model, peft_config, adapter_name
    )
    rho = eva_cfg.rho
    if rho > 1000:
        max_dim = max((max(p.shape) for p in model.parameters()), default=peft_config.r)
        rho = min(rho, max_dim // peft_config.r)

    was_training = model.training
    device = _model_device(model)
    model.eval()
    run_forward = forward_fn if forward_fn is not None else _forward_fn_dict
    session = _EvaSession(model, peft_config, eva_cfg, device, run_forward, prepare_model_inputs_fn)

    with lucid.no_grad():
        first_inputs, model_inputs_for_hooks = _prepare_step_inputs(
            batches[0], device, prepare_model_inputs_fn, peft_config
        )
        hash_hooks = _register_hash_hooks(
            session,
            rho,
            is_target,
            prepare_layer_inputs_fn,
            gather_distributed_inputs,
            model_inputs_for_hooks,
        )
        run_forward(model, first_inputs)
        _build_svd_hooks(session, hash_hooks, gather_distributed_inputs)
        try:
            rank_dist = _converge_svd_hooks(session, batches)
        finally:
            for _, rem_handle in session.svd_hooks.values():
                if rem_handle is not None:
                    rem_handle.remove()
            model.train(was_training)

    return _extract_eva_state_dict(session, rank_dist)


def _apply_eva_layer(
    model: nn.Module,
    name: str,
    module: LoraLayer,
    eva_state_dict: dict[str, Tensor],
    peft_config: LoraConfig,
    adapter_name: str,
) -> tuple[int, float, bool]:
    eva_cfg = peft_config.eva_config
    spec = module._specs[adapter_name]
    r = spec.r
    alpha = float(spec.lora_alpha)
    w = eva_state_dict.get(name)
    if w is None:
        w = eva_state_dict.get(name.removeprefix("base_model.model."))
    if w is None:
        module.reset_lora_parameters(adapter_name, init_lora_weights=True)
        return r, alpha, False

    new_rank = int(w.shape[0])
    if new_rank == 0:
        _set_submodule(model, name, module.get_base_layer())
        return 0, alpha, False

    if eva_cfg is not None and eva_cfg.adjust_scaling_factors and new_rank != r:
        if peft_config.use_rslora:
            alpha = alpha * math.sqrt(new_rank / r)
        else:
            alpha = alpha * (new_rank / r)
    if new_rank != r or alpha != float(spec.lora_alpha):
        module._resize_adapter(adapter_name, new_rank, alpha, peft_config.use_rslora)
    with lucid.no_grad():
        lora_a_mod = module.lora_A[adapter_name]
        assert isinstance(lora_a_mod, nn.Module)
        target_weight = lora_a_mod.__dict__.get("_parameters", {}).get("weight")
        assert isinstance(target_weight, Tensor)
        target_weight.copy_(w.to(device=target_weight.device, dtype=target_weight.dtype))
    return new_rank, alpha, True


def _load_eva_state_dict(
    model: nn.Module, eva_state_dict: dict[str, Tensor], adapter_name: str
) -> None:
    peft_configs = getattr(model, "peft_config", None)
    assert isinstance(peft_configs, dict)
    peft_config: LoraConfig = peft_configs[adapter_name]
    base_model = getattr(model, "base_model", None)
    prefix = "base_model.model." if isinstance(base_model, LoraModel) else "base_model."

    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, float] = {}
    missing_eva_inits: list[str] = []
    new_target_modules: list[str] = []

    target_layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, LoraLayer) and adapter_name in mod._adapter_names()
    ]
    merged = [name for name, mod in target_layers if adapter_name in mod.merged_adapters]
    if merged:
        raise ValueError(
            f"cannot initialise adapter {adapter_name!r} with EVA: it is merged into the base "
            f"weights of {merged}, which would keep the update of the factors this replaces; "
            "call unmerge_adapter() first"
        )
    for name, module in target_layers:
        name_in_base = name.removeprefix(prefix)
        new_rank, alpha, found = _apply_eva_layer(
            model, name, module, eva_state_dict, peft_config, adapter_name
        )
        if new_rank == 0:
            continue
        if found:
            new_target_modules.append(name_in_base)
        else:
            missing_eva_inits.append(name_in_base)
        if new_rank != peft_config.r:
            rank_pattern[name_in_base] = new_rank
        if alpha != float(peft_config.lora_alpha):
            alpha_pattern[name_in_base] = alpha

    peft_config.target_modules = new_target_modules + missing_eva_inits
    peft_config.rank_pattern = rank_pattern
    peft_config.alpha_pattern = alpha_pattern

    if missing_eva_inits:
        warnings.warn(
            "the following layers were initialized with init_lora_weights=True because they "
            f"were not found in the eva state_dict: {missing_eva_inits}\ncurrently the "
            f"following lora modules are not supported by EVA: {_UNSUPPORTED_LORA_MODULES}",
            stacklevel=2,
        )


def initialize_lora_eva_weights(
    model: nn.Module,
    dataloader: Iterable[object] | None = None,
    eva_state_dict: dict[str, Tensor] | None = None,
    forward_fn: _ForwardFn = _forward_fn_dict,
    prepare_model_inputs_fn: _ModelInputFn = _prepare_model_inputs_fn_language_modeling,
    prepare_layer_inputs_fn: _LayerInputFn = _prepare_layer_inputs_fn_language_modeling,
    adapter_name: str = "default",
    gather_distributed_inputs: bool = True,
    show_progress_bar: bool = True,
) -> None:
    r"""Initialise LoRA layers of ``model`` in place with EVA singular vectors and redistributed ranks.

    A layer whose rank changes gets new factors (``lora_A`` the singular vectors, ``lora_B``
    zero), so make the optimizer afterwards; ``rank_pattern`` and ``alpha_pattern`` of the
    adapter's config record the new ranks, and a saved adapter is rebuilt at them.

    Parameters
    ----------
    model : nn.Module
        A ``PeftModel`` holding an adapter made with ``init_lora_weights='eva'``.
    dataloader : iterable, optional
        The calibration batches; required unless ``eva_state_dict`` is given.
    eva_state_dict : dict of str to Tensor, optional
        The result of :func:`get_eva_state_dict`, to skip the forward passes.
    forward_fn : callable, optional
        Runs the model on a batch; by default a mapping is passed as keyword arguments.
    prepare_model_inputs_fn : callable, optional
        Turns a batch into what ``prepare_layer_inputs_fn`` is given as ``model_input``; by
        default the token mask of a language-modelling batch.
    prepare_layer_inputs_fn : callable or dict of str to callable, optional
        Selects the activations a layer's singular vectors are taken from, one function or one
        per module name; by default the unmasked tokens.
    adapter_name : str, default="default"
        The adapter to initialise.
    gather_distributed_inputs : bool, default=True
        Kept for the reference's signature: there is no process group to gather across.
    show_progress_bar : bool, default=True
        Kept for the reference's signature: nothing is displayed.

    Raises
    ------
    ValueError
        ``model`` is not a ``PeftModel`` of an ``init_lora_weights='eva'`` adapter, more than one
        adapter is active, no ``dataloader`` or ``eva_state_dict`` is given, or the adapter is
        merged into the base weights (``unmerge_adapter()`` first).
    """
    if not hasattr(model, "peft_config"):
        raise ValueError("initialize_lora_eva_weights requires a `PeftModel`.")
    active_adapters = getattr(model, "active_adapters", [adapter_name])
    if len(active_adapters) > 1:
        raise ValueError(
            "`initialize_lora_eva_weights` currently only works with a single active adapter"
        )
    peft_configs = getattr(model, "peft_config", {})
    peft_config = peft_configs[adapter_name]
    if peft_config.init_lora_weights != "eva":
        raise ValueError(
            "`initialize_lora_eva_weights` can only be used with `init_lora_weights='eva'`"
        )
    if eva_state_dict is None:
        if dataloader is None:
            raise ValueError("dataloader is required if eva_state_dict is not provided")
        eva_state_dict = get_eva_state_dict(
            model,
            dataloader,
            None,
            forward_fn,
            prepare_model_inputs_fn,
            prepare_layer_inputs_fn,
            adapter_name,
            gather_distributed_inputs,
            show_progress_bar,
        )
    with lucid.no_grad():
        _load_eva_state_dict(model, eva_state_dict, adapter_name)


_prepare_model_inputs_fn_language_modeling.__name__ = "prepare_model_inputs_fn_language_modeling"
_prepare_layer_inputs_fn_language_modeling.__name__ = "prepare_layer_inputs_fn_language_modeling"
_forward_fn_dict.__name__ = "forward_fn_dict"
