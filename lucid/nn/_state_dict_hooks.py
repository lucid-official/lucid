"""The ``state_dict`` / ``load_state_dict`` hook registration API of :class:`Module`.

Kept apart from ``module.py``: six registration methods that share one shape
(add to a registry, hand back a removable handle) and nothing else of the
module's machinery.  The calling side lives in ``Module.state_dict`` and in
``lucid.nn._state_dict`` (``_walk_load``).
"""

from collections import OrderedDict
from collections.abc import Callable

from lucid.nn.hooks import RemovableHandle, _next_hook_id, _WrappedHook


def _add_hook(
    registry: OrderedDict[int, Callable[..., object]], hook: Callable[..., object]
) -> RemovableHandle:
    """Put ``hook`` last in ``registry`` and return the handle that removes it."""
    key = _next_hook_id()
    registry[key] = hook
    return RemovableHandle(registry, key)


class _StateDictHookMixin:
    """Registers the hooks around ``state_dict`` and ``load_state_dict``."""

    _state_dict_hooks: OrderedDict[int, Callable[..., object]]
    _state_dict_pre_hooks: OrderedDict[int, Callable[..., object]]
    _load_state_dict_pre_hooks: OrderedDict[int, Callable[..., object]]
    _load_state_dict_post_hooks: OrderedDict[int, Callable[..., object]]

    def register_state_dict_pre_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a hook run at the start of this module's ``state_dict``.

        Parameters
        ----------
        hook : Callable
            ``hook(module, prefix, keep_vars) -> None``, called before the
            module writes its own entries, with the module it is registered
            on.  Hooks run in registration order.

        Returns
        -------
        RemovableHandle
            Call ``remove()`` on it to deregister the hook.
        """
        return _add_hook(self._state_dict_pre_hooks, hook)

    def register_state_dict_post_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a hook run at the end of this module's ``state_dict``.

        Parameters
        ----------
        hook : Callable
            ``hook(module, state_dict, prefix, local_metadata) -> None``,
            called once the module and its children have written their
            entries.  It edits ``state_dict`` (and ``local_metadata``, the
            ``_metadata`` entry of this module) in place and must return
            ``None``; any other return value raises ``RuntimeError``.  Hooks
            run in registration order.

        Returns
        -------
        RemovableHandle
            Call ``remove()`` on it to deregister the hook.

        Notes
        -----
        The hook is tagged ``_from_public_api`` so ``_register_state_dict_hook``
        refuses the same callable, as in the reference.
        """
        vars(hook)["_from_public_api"] = True  # the tag the private form and state_dict read
        return _add_hook(self._state_dict_hooks, hook)

    def _register_state_dict_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """The older form of :meth:`register_state_dict_post_hook`.

        ``hook(module, state_dict, prefix, local_metadata)`` may edit
        ``state_dict`` in place or return a new one; a returned dict is only
        honoured when this module is the one ``state_dict`` was called on.

        Raises
        ------
        RuntimeError
            If ``hook`` was already registered through
            :meth:`register_state_dict_post_hook`.
        """
        if getattr(hook, "_from_public_api", False):
            raise RuntimeError(
                "Cannot register the same function as the state dict post hook that was "
                "previously registered via register_state_dict_post_hook"
            )
        return _add_hook(self._state_dict_hooks, hook)

    def _register_load_state_dict_pre_hook(
        self, hook: Callable[..., object], with_module: bool = False
    ) -> RemovableHandle:
        """Register a load pre-hook, optionally without the module argument.

        With ``with_module=False`` the hook is called as ``hook(state_dict,
        prefix, local_metadata, strict, missing_keys, unexpected_keys,
        error_msgs)``; with ``True`` the module comes first, which is what
        :meth:`register_load_state_dict_pre_hook` registers.
        """
        return _add_hook(
            self._load_state_dict_pre_hooks, _WrappedHook(hook, self if with_module else None)
        )

    def register_load_state_dict_pre_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a pre-hook called when this module loads state_dict.

        Hook signature:
            hook(module, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None

        The hook may mutate state_dict/missing/unexpected/error_msgs.
        """
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    def register_load_state_dict_post_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a post-hook called after this module loads state_dict.

        Hook signature: ``hook(module, incompatible_keys) -> None``.
        """
        return _add_hook(self._load_state_dict_post_hooks, hook)
