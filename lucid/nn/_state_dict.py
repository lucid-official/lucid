"""state_dict save/load infrastructure for Module.

The protocol mirrors the reference framework:

* ``Module._save_to_state_dict`` writes own params / buffers into the dest dict.
  The top-level ``state_dict()`` walker recurses into children.
* ``Module._load_from_state_dict`` consumes its own keys from the supplied
  state_dict, mutating ``missing_keys``, ``unexpected_keys`` and
  ``error_msgs`` in place.  The top-level ``load_state_dict()`` driver
  recurses, runs pre/post hooks, and raises a single ``RuntimeError`` at
  the end when ``strict=True``.
* ``state_dict()`` (``Module.state_dict``, which also runs the state-dict
  hooks) attaches an ``_metadata`` attribute to the returned OrderedDict
  mapping ``module_path → {"version": <int>}`` for every module.  This
  metadata is forwarded
  to each child's ``_load_from_state_dict`` via the ``local_metadata`` arg
  for migration hooks to inspect.
"""

from collections import OrderedDict

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _impl_with_grad, _wrap
from lucid.nn.hooks import _GLOBAL_LOAD_STATE_DICT_POST_HOOKS, _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS
from lucid.nn._incompatible_keys import _IncompatibleKeys
from lucid.nn.module import Module, _tensors_replaced
from lucid._types import StateDict

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


def _unexpected_here(module: Module, state_dict: StateDict, prefix: str) -> list[str]:
    """Checkpoint keys under ``prefix`` that neither this module nor a child claims.

    Run right after the module's own ``_load_from_state_dict`` (so after its
    pre-hooks have renamed what they rename), as the reference does; a key
    under a child is the child's to judge.
    """
    own = set(_enumerate_local_keys(module, prefix))
    children = {name for name, child in module._modules.items() if child is not None}
    unexpected: list[str] = []
    for key in state_dict:
        if not key.startswith(prefix) or key in own:
            continue
        head, dot, _ = key[len(prefix) :].partition(".")
        if not (dot and head in children):
            unexpected.append(key)
    return unexpected


def _walk_load(
    module: Module,
    state_dict: StateDict,
    prefix: str,
    metadata: dict[str, dict[str, object]],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
    assign: bool = False,
) -> None:
    """Handle this module, recurse into its children, then run its post-hooks.

    As in the reference, a module's post-hooks run right after its own
    subtree has loaded and see ``missing_keys`` / ``unexpected_keys`` as they
    stand then — the lists are shared across the walk, and a hook may edit
    them in place; ``strict`` is judged on what is left once every hook has
    run.
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
    # Wrapped hooks: each one decides whether it is handed the module.
    for hook in module._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_meta, strict, missing_keys, unexpected_keys, error_msgs)

    module._load_from_state_dict(
        state_dict, prefix, local_meta, strict, missing_keys, unexpected_keys, error_msgs
    )
    if assign:
        # New tensor objects now stand where the old ones did; a copy writes
        # into the old ones, which every holder sees.
        _tensors_replaced(module)

    unexpected_keys.extend(_unexpected_here(module, state_dict, prefix))

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
            assign,
        )

    incompatible: _IncompatibleKeys = _IncompatibleKeys(missing_keys, unexpected_keys)
    for hook in module._load_state_dict_post_hooks.values():
        hook(module, incompatible)
    for hook in _GLOBAL_LOAD_STATE_DICT_POST_HOOKS.values():
        hook(module, incompatible)


def _load_state_dict(
    module: Module, state_dict: StateDict, strict: bool = True, assign: bool = False
) -> _IncompatibleKeys:
    """Driver for ``Module.load_state_dict``.

    Pre-order recursion: at each module, run pre-hooks, then
    ``_load_from_state_dict``, then recurse into children, then run the
    module's post-hooks (each right after its own subtree, as in the
    reference).  ``state_dict`` may carry an ``_metadata`` attribute (set
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

    _walk_load(
        module, state_dict, "", metadata, strict, missing_keys, unexpected_keys, error_msgs, assign
    )

    # Strict mode: turn what the hooks left of missing/unexpected into errors.
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

    return _IncompatibleKeys(missing_keys, unexpected_keys)
