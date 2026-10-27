"""state_dict save/load infrastructure for Module.

The protocol mirrors the reference framework:

* ``Module._save_to_state_dict`` writes own params / buffers into the dest dict.
  The top-level ``state_dict()`` walker recurses into children.
* ``Module._load_from_state_dict`` consumes its own keys from the supplied
  state_dict, mutating ``missing_keys``, ``unexpected_keys`` and
  ``error_msgs`` in place.  The top-level ``load_state_dict()`` driver
  recurses, runs pre/post hooks, and raises a single ``RuntimeError`` at
  the end when ``strict=True``.
* ``state_dict()`` attaches an ``_metadata`` attribute to the returned
  OrderedDict mapping ``module_path → {"version": <int>}`` for every module
  that defines a ``_version`` class attribute.  This metadata is forwarded
  to each child's ``_load_from_state_dict`` via the ``local_metadata`` arg
  for migration hooks to inspect.
"""

from collections import OrderedDict, namedtuple
from typing import final

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _impl_with_grad, _wrap
from lucid.nn.hooks import _GLOBAL_LOAD_STATE_DICT_POST_HOOKS, _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS
from lucid.nn.module import Module, _tensors_replaced
from lucid._types import StateDict


@final
class _IncompatibleKeys(namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    """Result tuple returned by ``Module.load_state_dict``.

    Lightweight named tuple summarising the diff between the
    checkpoint's ``state_dict`` keys and the module's actual parameter
    / buffer names.  Each field is a list of strings; the loader is
    advisory — keys reported here are merely *not loaded*, not raised.

    Attributes
    ----------
    missing_keys : list[str]
        Module parameter / buffer names not present in the checkpoint.
        These end up using the module's existing (init-time) values.
    unexpected_keys : list[str]
        Checkpoint entries with no corresponding parameter / buffer in
        the module — typically a sign of a stale checkpoint or a
        rename.
    """

    __slots__ = ()


# ── state_dict (save side) ────────────────────────────────────────────────────


def _build_metadata(module: Module, prefix: str = "") -> dict[str, dict[str, object]]:
    """Walk the module tree and collect per-module ``_version`` tags.

    Every module now carries at least ``{"version": 1}`` (base class default),
    matching the reference framework.  Subclasses may set ``_version = N``
    (N ≥ 2) to signal that their ``_load_from_state_dict`` performs key
    migration for older checkpoints.
    """
    metadata: dict[str, dict[str, object]] = {}
    version: int | None = getattr(type(module), "_version", None)
    key: str = prefix.rstrip(".") if prefix else ""
    # Always record when version is a non-None int (includes the base default 1).
    if isinstance(version, int):
        metadata[key] = {"version": version}
    for name, child in module._modules.items():
        if child is None:
            continue
        metadata.update(_build_metadata(child, f"{prefix}{name}."))
    return metadata


def _save_to_state_dict(
    module: Module, destination: dict[str, Tensor], prefix: str = "", keep_vars: bool = False
) -> None:
    """Recursive walker — let each module write its own state, then recurse."""
    module._save_to_state_dict(destination, prefix, keep_vars)
    for mname, child in module._modules.items():
        if child is None:
            continue
        _save_to_state_dict(child, destination, f"{prefix}{mname}.", keep_vars)


# ── state_dict (load side) ────────────────────────────────────────────────────


def _checked_source(
    src: object, key: str, attr: Tensor, assign: bool, error_msgs: list[str]
) -> Tensor | None:
    """``src`` if it can be loaded into ``attr``, else ``None`` with the reason
    appended to ``error_msgs``.

    The shapes must agree, on an ``assign`` load too — only a parameter not
    yet shaped (a lazy layer's) is replaced whatever the checkpoint's shape,
    as in the reference.  One exception, the reference's too: a 0-d tensor
    takes a shape ``(1,)`` one as its scalar, which is how checkpoints from
    before 0-d tensors stored it.
    """
    from lucid.nn.parameter import UninitializedParameter

    if not isinstance(src, Tensor):
        error_msgs.append(
            f"While copying parameter '{key}', expected Tensor, " f"got {type(src).__name__}"
        )
        return None
    unshaped = assign and isinstance(attr, UninitializedParameter)
    if not unshaped and attr.ndim == 0 and tuple(src.shape) == (1,):
        src = src.reshape(())
    if not unshaped and tuple(src.shape) != tuple(attr.shape):
        error_msgs.append(
            f"size mismatch for {key}: copying a param with shape "
            f"{tuple(src.shape)} from checkpoint, the shape in current model "
            f"is {tuple(attr.shape)}."
        )
        return None
    return src


def _put(module: Module, name: str, attr: Tensor, src: Tensor, assign: bool) -> None:
    """Load ``src`` into the module's ``name``, which holds ``attr`` now."""
    from lucid.nn.parameter import Parameter

    if not assign:
        # Copy into the tensor the module already holds, with its own
        # dtype and device, as the reference does.  Rebinding it to a new
        # impl left everything that held the old one — a view of the
        # weight, an alias, a hook — reading the old values.
        converted: Tensor = src.to(device=attr.device, dtype=attr.dtype)
        attr._impl.copy_from(_C_engine.contiguous(converted._impl))
        return
    # Replace the parameter/buffer object, taking the checkpoint's dtype and
    # device.  A Parameter keeps the module's requires_grad; a buffer is the
    # loaded tensor, flag included, as in the reference (LCD-486).
    is_param = name in module._parameters
    needs_grad = attr.requires_grad if is_param else src.requires_grad
    new_impl = _impl_with_grad(_C_engine.contiguous(src._impl), needs_grad)
    if is_param:
        module._parameters[name] = Parameter(_wrap(new_impl), requires_grad=needs_grad)
    else:
        module._buffers[name] = _wrap(new_impl)


def _default_load_from_state_dict(
    module: Module,
    state_dict: StateDict,
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Default ``_load_from_state_dict`` impl — params + persistent buffers.

    Subclasses should call this from their override (after any pre-processing)
    to share the standard copy / dtype-convert / shape-check logic.

    ``local_metadata["assign_to_params_buffers"]``, which an ``assign=True``
    load sets as the reference sets it, puts the loaded tensor in place of
    the parameter / buffer with the checkpoint's dtype and device; otherwise
    the data is copied into the existing tensor, which keeps its own.
    """
    del strict, unexpected_keys  # _load_from_state_dict hook signature parity
    assign = bool(local_metadata.get("assign_to_params_buffers", False))
    persistent_buffers: dict[str, Tensor] = {
        name: b
        for name, b in module._buffers.items()
        if b is not None and name not in module._non_persistent_buffers
    }
    local_state: dict[str, Tensor] = {
        **{name: p for name, p in module._parameters.items() if p is not None},
        **persistent_buffers,
    }

    for name, attr in local_state.items():
        key: str = f"{prefix}{name}"
        if key not in state_dict:
            missing_keys.append(key)
            continue
        src = _checked_source(state_dict[key], key, attr, assign, error_msgs)
        if src is not None:
            _put(module, name, attr, src, assign)

    # extra_state (rare hook for opaque per-module state).
    extra_key: str = f"{prefix}_extra_state"
    if extra_key in state_dict:
        module.set_extra_state(state_dict[extra_key])
    elif module.get_extra_state() is not None:
        # Module advertises extra_state but checkpoint lacks it.
        missing_keys.append(extra_key)


def _enumerate_local_keys(module: Module, prefix: str) -> list[str]:
    """Flat list of keys this module owns at ``prefix`` (params + persistent buffers)."""
    keys: list[str] = []
    for name, p in module._parameters.items():
        if p is not None:
            keys.append(f"{prefix}{name}")
    for name, b in module._buffers.items():
        if b is not None and name not in module._non_persistent_buffers:
            keys.append(f"{prefix}{name}")
    if module.get_extra_state() is not None:
        keys.append(f"{prefix}_extra_state")
    return keys


def _walk_load(
    module: Module,
    state_dict: StateDict,
    prefix: str,
    metadata: dict[str, dict[str, object]],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
    post_hook_modules: list[Module],
    assign: bool = False,
) -> None:
    """Pre-order walk: handle this module, then recurse into children.

    Post-hooks are deferred — modules with registered post-hooks are
    appended to `post_hook_modules` and fired by the top-level driver
    after the full walk completes, so each hook sees the final
    _IncompatibleKeys rather than a per-module snapshot.
    """
    local_meta: dict[str, object] = metadata.get(prefix.rstrip("."), {}) if metadata else {}
    if assign:
        # Told to every loader the way the reference tells it, so a module's
        # own _load_from_state_dict (lazy sizing, key and version migration)
        # runs on an assign load too.  A copy: the checkpoint's metadata is
        # the caller's.
        local_meta = {**local_meta, "assign_to_params_buffers": True}

    # Pre-hooks (global → instance).
    for hook in _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS.values():
        hook(
            module,
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    for hook in module._load_state_dict_pre_hooks.values():
        hook(
            module,
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    module._load_from_state_dict(
        state_dict, prefix, local_meta, strict, missing_keys, unexpected_keys, error_msgs
    )
    if assign:
        # New tensor objects now stand where the old ones did; a copy writes
        # into the old ones, which every holder sees.
        _tensors_replaced(module)

    if module._load_state_dict_post_hooks:
        post_hook_modules.append(module)

    for name, child in module._modules.items():
        if child is None:
            continue
        _walk_load(
            child,
            state_dict,
            f"{prefix}{name}.",
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            post_hook_modules,
            assign,
        )


def _load_state_dict(
    module: Module, state_dict: StateDict, strict: bool = True, assign: bool = False
) -> _IncompatibleKeys:
    """Driver for ``Module.load_state_dict``.

    Pre-order recursion: at each module, run pre-hooks, then
    ``_load_from_state_dict``, then post-hooks, then recurse into
    children.  ``state_dict`` may carry an ``_metadata`` attribute (set
    by ``Module.state_dict()``) mapping module paths to per-module
    metadata such as ``{"version": N}``; each module's hook receives
    the relevant slice as ``local_metadata``.

    Parameters
    ----------
    module : Module
        Destination module tree.  Its parameter / buffer values are
        overwritten in place by matching keys from ``state_dict``.
    state_dict : StateDict
        Source mapping ``str → Tensor``.  Typically the output of a
        prior :meth:`Module.state_dict` call or a deserialised
        checkpoint.
    strict : bool, optional
        When ``True`` (default), raise if ``state_dict`` contains keys
        not present on the module or vice versa.  When ``False``, both
        kinds of mismatch are reported via the returned
        :class:`_IncompatibleKeys`.
    assign : bool, optional
        When ``True``, replace the module's parameter / buffer tensors
        wholesale (preserving the checkpoint's dtype / device).  When
        ``False`` (default), copy values into the existing tensors so
        downstream optimiser / grad state stays attached.

    Returns
    -------
    _IncompatibleKeys
        Named tuple ``(missing_keys, unexpected_keys)`` summarising the
        load.  Empty fields indicate a clean match.
    """
    metadata: dict[str, dict[str, object]] = getattr(state_dict, "_metadata", None) or {}
    # We must operate on a mutable copy so hooks may rename keys safely.
    state_dict = OrderedDict(state_dict)
    state_dict._metadata = metadata  # type: ignore[attr-defined]

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []
    post_hook_modules: list[Module] = []

    _walk_load(
        module,
        state_dict,
        "",
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        post_hook_modules,
        assign,
    )

    # Compute final unexpected_keys based on the post-hook state_dict
    # (pre-hooks may have renamed/added/removed keys).  Anything still
    # present that no module claimed is "unexpected".
    expected_keys: set[str] = set()
    _collect_expected(module, "", expected_keys)
    unexpected_keys = [k for k in state_dict.keys() if k not in expected_keys]

    incompatible: _IncompatibleKeys = _IncompatibleKeys(missing_keys, unexpected_keys)

    # Fire post-hooks (instance + global) with the final _IncompatibleKeys.
    for m in post_hook_modules:
        for hook in m._load_state_dict_post_hooks.values():
            hook(m, incompatible)
    if _GLOBAL_LOAD_STATE_DICT_POST_HOOKS:

        def _walk_post(mod: Module) -> None:
            for hook in _GLOBAL_LOAD_STATE_DICT_POST_HOOKS.values():
                hook(mod, incompatible)
            for child in mod._modules.values():
                if child is not None:
                    _walk_post(child)

        _walk_post(module)

    # Strict mode: turn missing/unexpected into errors.
    if strict:
        if missing_keys:
            error_msgs.insert(
                0, f"Missing key(s) in state_dict: {', '.join(repr(k) for k in missing_keys)}."
            )
        if unexpected_keys:
            error_msgs.insert(
                0,
                f"Unexpected key(s) in state_dict: {', '.join(repr(k) for k in unexpected_keys)}.",
            )

    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                type(module).__name__, "\n\t".join(error_msgs)
            )
        )

    return incompatible


def _collect_expected(module: Module, prefix: str, out: set[str]) -> None:
    out.update(_enumerate_local_keys(module, prefix))
    for name, child in module._modules.items():
        if child is None:
            continue
        _collect_expected(child, f"{prefix}{name}.", out)
