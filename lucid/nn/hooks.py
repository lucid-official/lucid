"""Hook registration helpers for Module."""

from collections import OrderedDict
from typing import Callable

_GLOBAL_FORWARD_PRE_HOOKS: OrderedDict[int, tuple[Callable[..., object], bool]] = OrderedDict()
_GLOBAL_FORWARD_HOOKS: OrderedDict[int, tuple[Callable[..., object], bool, bool]] = OrderedDict()
_GLOBAL_BACKWARD_PRE_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()
_GLOBAL_BACKWARD_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()
# load_state_dict pre-hook signature:
#   hook(module, state_dict, prefix, local_metadata, strict, missing_keys,
#        unexpected_keys, error_msgs) -> None
_GLOBAL_LOAD_STATE_DICT_PRE_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()
# load_state_dict post-hook signature:
#   hook(module, incompatible_keys) -> None
_GLOBAL_LOAD_STATE_DICT_POST_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()

_HOOK_ID = 0


def _next_hook_id() -> int:
    global _HOOK_ID
    _HOOK_ID += 1
    return _HOOK_ID


class RemovableHandle:
    """Lightweight handle returned by every ``register_*_hook`` call.

    Holds a reference to the dict (or set) that owns the registered hook
    plus the unique key under which it lives.  Calling :meth:`remove`
    deregisters the hook — subsequent forward / backward / load-state-dict
    passes will skip it.  Also supports the context manager protocol so
    hooks can be scoped to a ``with`` block.

    Parameters
    ----------
    hooks : dict[int, object]
        The hook registry dict (e.g. ``_GLOBAL_FORWARD_HOOKS``) whose
        entry should be removed when :meth:`remove` is called.
    key : int
        Unique hook identifier returned by ``_next_hook_id()``.
    extra_sets : tuple[set[int], ...], optional
        Auxiliary sets (e.g. for "always-called" hooks) from which the
        same key should also be discarded on removal.

    Notes
    -----
    The handle does NOT own the hook callable — it only knows where to
    delete the registration.  Holding a handle therefore prevents nothing
    on its own; once :meth:`remove` is called the hook stops firing
    regardless of any references the caller still holds.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn import hooks
    >>> model, x = nn.Linear(3, 2), lucid.zeros((1, 3))
    >>> def my_hook(mod, inputs):
    ...     print(f'about to call {type(mod).__name__}')
    >>> h = hooks.register_module_forward_pre_hook(my_hook)
    >>> _ = model(x)
    about to call Linear
    >>> h.remove()                    # explicit deregistration
    >>> _ = model(x)                  # silent: the hook is gone
    >>>
    >>> # Or via context manager for scoped activation:
    >>> with hooks.register_module_forward_pre_hook(my_hook):
    ...     _ = model(x)              # hook fires here
    about to call Linear
    >>> _ = model(x)                  # auto-removed on context exit
    """

    def __init__(
        self,
        hooks: dict[int, object],
        key: int,
        extra_sets: tuple[set[int], ...] = (),
    ) -> None:
        """Initialise the handle.  See the class docstring for parameter semantics."""
        self._hooks: dict[int, object] = (
            hooks  # covariant dict usage: we only pop/read, never write incompatible types
        )
        self._key = key
        self._extra_sets = extra_sets

    def remove(self) -> None:
        """Deregister the hook this handle refers to.

        Pops the hook entry from the owning registry dict and discards the
        key from any auxiliary sets.  Idempotent — calling ``remove`` a
        second time is a no-op (``pop(..., None)``).
        """
        self._hooks.pop(self._key, None)
        for extra_set in self._extra_sets:
            extra_set.discard(self._key)

    def __enter__(self) -> RemovableHandle:
        """Enter the ``with`` block; return ``self`` for binding via ``as``."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the ``with`` block, auto-removing the hook registration."""
        self.remove()


def register_module_forward_pre_hook(
    hook: Callable[..., object],
    *,
    with_kwargs: bool = False,
) -> RemovableHandle:
    r"""Register a global pre-forward hook fired on every ``Module.__call__``.

    The hook is invoked **before** the wrapped module's ``forward`` method
    runs.  It receives the module instance and the positional inputs (and,
    optionally, the keyword inputs) and may return a tuple that replaces
    the positional inputs for the upcoming forward pass — useful for
    automatic input normalisation, logging, or shape validation.

    Parameters
    ----------
    hook : Callable
        The pre-hook function.  Signature:

        * ``hook(module, inputs) -> None | tuple``                (default)
        * ``hook(module, inputs, kwargs) -> None | tuple``        (``with_kwargs=True``)

        Returning ``None`` leaves the inputs untouched; returning a
        non-``None`` tuple replaces the positional inputs.
    with_kwargs : bool, optional
        If ``True``, the hook also receives the kwargs dict and may
        replace it (return ``(new_inputs, new_kwargs)``).  Default
        ``False`` for compatibility with reference-framework idioms.

    Returns
    -------
    RemovableHandle
        Handle whose :meth:`~RemovableHandle.remove` deregisters the hook.

    Notes
    -----
    Global hooks fire for **every** Module instance.  For per-module hooks
    use :meth:`Module.register_forward_pre_hook` instead.  The order of
    hook execution is registration order (FIFO).

    Examples
    --------
    >>> from lucid.nn.hooks import register_module_forward_pre_hook
    >>> def log_call(mod, inputs):
    ...     print(f'{type(mod).__name__}({inputs[0].shape})')
    >>> handle = register_module_forward_pre_hook(log_call)
    >>> # ... forward passes trigger the log ...
    >>> handle.remove()
    """
    key = _next_hook_id()
    _GLOBAL_FORWARD_PRE_HOOKS[key] = (hook, with_kwargs)
    return RemovableHandle(_GLOBAL_FORWARD_PRE_HOOKS, key)  # type: ignore[arg-type]


def register_module_forward_hook(
    hook: Callable[..., object],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    r"""Register a global post-forward hook fired after every ``Module.__call__``.

    The hook runs **after** ``forward`` returns and may inspect or replace
    the output.  Common uses: activation logging / visualisation, output
    transformation (e.g. moving outputs to a different device), or
    side-effect bookkeeping.

    Parameters
    ----------
    hook : Callable
        The post-hook function.  Signature:

        * ``hook(module, inputs, output) -> None | new_output``                (default)
        * ``hook(module, inputs, kwargs, output) -> None | new_output``        (``with_kwargs=True``)

        Returning ``None`` leaves the output unchanged; returning any
        other value replaces the forward output.
    with_kwargs : bool, optional
        If ``True``, the hook also receives the kwargs dict that was
        passed to ``forward``.
    always_call : bool, optional
        If ``True``, the hook fires even when ``forward`` raises an
        exception (the ``output`` argument is then ``None``).  Useful for
        cleanup-style hooks.

    Returns
    -------
    RemovableHandle
        Handle for later deregistration.

    Notes
    -----
    Global forward hooks fire for **every** Module call.  When multiple
    hooks are registered they run in registration order and each may see
    the previous hook's replacement output — a chain of transforms.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.hooks import register_module_forward_hook
    >>> model, x = nn.Linear(3, 2), lucid.zeros((1, 3))
    >>> activations = {}
    >>> def capture(mod, inputs, output):
    ...     activations[type(mod).__name__] = output.detach()
    >>> with register_module_forward_hook(capture):
    ...     _ = model(x)
    >>> sorted(activations)
    ['Linear']
    """
    key = _next_hook_id()
    _GLOBAL_FORWARD_HOOKS[key] = (hook, with_kwargs, always_call)
    return RemovableHandle(_GLOBAL_FORWARD_HOOKS, key)  # type: ignore[arg-type]


def register_module_full_backward_pre_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    r"""Register a global pre-hook fired before any module's backward pass.

    Invoked at the moment Lucid's autograd engine is about to compute
    gradients for a Module's parameters.  The hook receives the module and
    the gradient(s) of the module's output(s); it may return a modified
    tuple of grad-outputs to inject custom backward behaviour (e.g.
    gradient clipping, sign-SGD-style modifications).

    Parameters
    ----------
    hook : Callable
        Signature: ``hook(module, grad_output) -> None | tuple``.
        Returning a tuple replaces the gradient flowing into the module
        before any registered post-hook runs.

    Returns
    -------
    RemovableHandle
        Handle for later deregistration.

    Notes
    -----
    "Full" backward pre-hooks are paired with **full** backward hooks
    (:func:`register_module_full_backward_hook`).  Both fire only when
    autograd actually visits the module's backward node — modules whose
    output gradient is never requested (e.g. frozen branches) are skipped.

    Examples
    --------
    >>> from lucid.nn.hooks import register_module_full_backward_pre_hook
    >>> def scale_grads(mod, grad_output):
    ...     return tuple(g * 0.5 for g in grad_output)
    >>> h = register_module_full_backward_pre_hook(scale_grads)
    >>> h.remove()                   # global: leave nothing behind
    """
    key = _next_hook_id()
    _GLOBAL_BACKWARD_PRE_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_BACKWARD_PRE_HOOKS, key)  # type: ignore[arg-type]


def register_module_full_backward_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    r"""Register a global post-hook fired after a module's backward pass.

    Invoked once Lucid's autograd engine has finished accumulating
    gradients for a Module.  The hook receives the module along with the
    input-gradients (gradients flowing back into the module's inputs) and
    the output-gradients (gradients that came from above).  Mutating these
    in place propagates to subsequent backward computation; returning a
    new tuple of input-gradients replaces them.

    Parameters
    ----------
    hook : Callable
        Signature: ``hook(module, grad_input, grad_output) -> None | tuple``.

    Returns
    -------
    RemovableHandle
        Handle for later deregistration.

    Notes
    -----
    Use the "full" variant whenever a Module has multiple inputs and you
    need access to every per-input gradient.  Pairs naturally with
    :func:`register_module_full_backward_pre_hook` for modify-then-observe
    workflows.

    Examples
    --------
    >>> from lucid.nn.hooks import register_module_full_backward_hook
    >>> def log_grad_norm(mod, grad_input, grad_output):
    ...     for i, g in enumerate(grad_input):
    ...         if g is not None:
    ...             print(f'{type(mod).__name__}.in[{i}].grad_norm = {g.norm().item()}')
    >>> handle = register_module_full_backward_hook(log_grad_norm)
    >>> handle.remove()              # global: leave nothing behind
    """
    key = _next_hook_id()
    _GLOBAL_BACKWARD_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_BACKWARD_HOOKS, key)  # type: ignore[arg-type]


def register_module_load_state_dict_pre_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    r"""Register a global pre-hook fired for every Module during ``load_state_dict``.

    Invoked **before** a module attempts to copy values from a state-dict
    into its own parameters / buffers.  The hook may mutate ``state_dict``
    in place to rename keys, transform tensors (e.g. cast dtype, adjust
    shape after architectural changes), or pre-populate
    ``missing_keys``/``unexpected_keys`` to influence ``strict=True``
    behaviour.

    Parameters
    ----------
    hook : Callable
        Signature::

            hook(module, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None

        All list / dict arguments may be mutated in place.

    Returns
    -------
    RemovableHandle
        Handle for later deregistration.

    Notes
    -----
    Common application: backward-compatible checkpoint loading after
    refactors — e.g. rename ``"layer.weight"`` to ``"linear.weight"`` to
    accommodate a renamed submodule without forcing users to regenerate
    checkpoints.

    Examples
    --------
    >>> from lucid.nn.hooks import register_module_load_state_dict_pre_hook
    >>> def rename_keys(mod, sd, prefix, *args):
    ...     # Rename legacy keys on the fly.
    ...     for k in list(sd):
    ...         if k.startswith(prefix + 'fc.'):
    ...             sd[k.replace('fc.', 'linear.', 1)] = sd.pop(k)
    >>> h = register_module_load_state_dict_pre_hook(rename_keys)
    >>> # model.load_state_dict(legacy_checkpoint) now succeeds
    >>> h.remove()                   # global: leave nothing behind
    """
    key = _next_hook_id()
    _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_LOAD_STATE_DICT_PRE_HOOKS, key)  # type: ignore[arg-type]


def register_module_load_state_dict_post_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    r"""Register a global post-hook fired after every Module's ``load_state_dict``.

    Invoked **after** the copy from state-dict to parameters has happened.
    The hook receives the module and a NamedTuple-like ``incompatible_keys``
    object summarising any ``missing_keys`` / ``unexpected_keys`` that
    occurred — useful for warning the user, clearing those keys to
    silence ``strict=True`` errors, or running follow-up reinitialisation
    on any parameter that wasn't loaded.

    Parameters
    ----------
    hook : Callable
        Signature: ``hook(module, incompatible_keys) -> None``.

        ``incompatible_keys`` has two list attributes / fields:
        ``missing_keys`` and ``unexpected_keys``.  Mutating either list
        is reflected in the caller's error handling.

    Returns
    -------
    RemovableHandle
        Handle for later deregistration.

    Notes
    -----
    Common application: re-initialise newly-added layers after loading a
    legacy checkpoint that doesn't contain entries for them — keeps the
    rest of the model from being clobbered while bringing the new
    parameters to a sensible starting point.

    Examples
    --------
    >>> from lucid.nn.hooks import register_module_load_state_dict_post_hook
    >>> def reinit_missing(mod, incompatible):
    ...     for key in incompatible.missing_keys:
    ...         print(f'(post-load) {key} not in checkpoint — keeping init values')
    >>> h = register_module_load_state_dict_post_hook(reinit_missing)
    >>> h.remove()                   # global: leave nothing behind
    """
    key = _next_hook_id()
    _GLOBAL_LOAD_STATE_DICT_POST_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_LOAD_STATE_DICT_POST_HOOKS, key)  # type: ignore[arg-type]
