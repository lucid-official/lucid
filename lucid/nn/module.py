"""
nn.Module: base class for all neural network layers.
"""

import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Self,
    overload,
    override,
)

from lucid._C import engine as _C_engine
from lucid import _promotion
from lucid._dtype import _ENGINE_TO_DTYPE, dtype as _dtype_cls
from lucid._tensor.tensor import Tensor
from lucid._tensor._to import _to_engine_target, _type_target
from lucid._dispatch import _unwrap, _wrap
from lucid.autograd._grad_inplace import zero_grads_
from lucid.nn.parameter import Buffer, Parameter, UninitializedParameter
from lucid.nn.hooks import (
    _GLOBAL_BACKWARD_HOOKS,
    _GLOBAL_BACKWARD_PRE_HOOKS,
    _GLOBAL_FORWARD_HOOKS,
    _GLOBAL_FORWARD_PRE_HOOKS,
    RemovableHandle,
)
from lucid.nn._module_backward_state import _ModuleBackwardState
from lucid._tree_version import _TrackedDict, _tree_changed
from lucid._types import (
    _ModuleOutput,
    _ForwardPreHook,
    _ForwardPreHookWithKwargs,
    _ForwardHook,
    _ForwardHookWithKwargs,
    _BackwardHook,
)

if TYPE_CHECKING:
    from lucid._types import _HasForward


class UninitializedParameterWarning(UserWarning):
    """A lazy layer's parameters were read before they existed.

    Its own category so it can be promoted, filtered or counted on its
    own — a blanket ``UserWarning`` filter would either swallow this or
    turn every unrelated warning into an error.
    """


# _state_dict is imported lazily inside state_dict()/load_state_dict() to
# break the Module ↔ _state_dict circular dependency.


_HOOK_ID: int = 0

# The registries ``lucid.compile`` keys an executable on, held as
# ``_TrackedDict``s so every write to one moves the tree version
# (``nn/_tree_version.py``) — they are written directly, not set as attributes.
# The tensor registries are among them: an executable reads the tensors it was
# traced with, and ``register_parameter``, an ``assign=True`` load or a
# ``ParameterList.append`` put new ones there without a ``setattr`` (LCD-560).
_TRACKED_REGISTRIES = (
    "_parameters",
    "_buffers",
    "_modules",
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)


def _next_hook_id() -> int:
    global _HOOK_ID
    _HOOK_ID += 1
    return _HOOK_ID


class _ModuleWrapper:
    """Answers calls on behalf of a module that stays in its parent's tree.

    :func:`lucid.compile` returns one.  Assigned as a module attribute, the
    wrapped :attr:`model` is what the parent registers — so its parameters,
    ``state_dict`` keys, ``.to()`` and ``.train()`` are exactly what they
    were before it was wrapped — while the attribute itself keeps answering
    with the wrapper, so ``self.encoder(x)`` still takes the wrapper's path.
    """

    @property
    def model(self) -> Module:
        """The module this wrapper answers for."""
        raise NotImplementedError

    def _tensors_replaced(self) -> None:
        """The model's parameters or buffers were swapped for new tensors."""


# Wrappers holding work derived from a module's tensors, keyed by that module.
# ``_apply`` (``.to()`` / ``.half()``) and ``load_state_dict`` put new tensors
# in place of the old ones, and a compiled executable fed the old ones keeps
# answering with the old weights — so both tell the module's wrappers.
_WRAPPERS: weakref.WeakKeyDictionary[Module, weakref.WeakSet[_ModuleWrapper]] = (
    weakref.WeakKeyDictionary()
)


def _watch_tensors(module: Module, wrapper: _ModuleWrapper) -> None:
    """Tell ``wrapper`` whenever ``module``'s tensors are replaced."""
    _WRAPPERS.setdefault(module, weakref.WeakSet()).add(wrapper)


def _tensors_replaced(module: Module) -> None:
    if _WRAPPERS:
        for wrapper in tuple(_WRAPPERS.get(module, ())):
            wrapper._tensors_replaced()


def _apply_to_parameter(param: Parameter, fn: Callable[[Tensor], Tensor]) -> bool:
    """Apply ``fn`` to ``param`` and its gradient; whether either got a new tensor."""
    grad = param.grad
    converted = fn(param)
    # ``.data =`` keeps the Parameter and records no graph, so backward still
    # accumulates on it.  An unchanged tensor stays, or an optimizer would see a
    # replaced parameter and a held ``p.grad`` would stop being the gradient.
    replaced = converted._impl is not param._impl
    if replaced:
        param.data = converted
    if grad is not None:
        g = fn(grad)
        if g._impl is not grad._impl:
            param.grad = g if g._impl.is_leaf else g.detach()
            replaced = True
    return replaced


def _refuse_unless_none(value: object, name: str, kind: str, expected: str) -> None:
    """Reject assigning ``value`` to the registered ``kind`` slot ``name``.

    Only ``None`` — which empties the slot — or a value of the slot's own
    kind may go there; the kind is checked by the caller first.
    """
    if value is not None:
        raise TypeError(
            f"cannot assign '{type(value).__name__}' as {kind} '{name}' "
            f"({expected} or None expected)"
        )


def _before_init_kind(value: object) -> str | None:
    """Which registry ``value`` would need, or ``None`` for a plain value."""
    if isinstance(value, Parameter):
        return "parameters"
    if isinstance(value, Buffer):
        return "buffer"
    if isinstance(value, (Module, _ModuleWrapper)):
        return "module"
    return None


class Module:
    """Base class for all neural network modules.

    Every custom model should subclass this and implement :meth:`forward`.
    Submodules assigned as attributes are tracked automatically.

    Notes
    -----
    **Attribute routing**: Setting an attribute follows this priority order:

    1. If the value is a :class:`~lucid.nn.Parameter` → stored in ``_parameters``.
    2. If the name is a registered parameter → the value must be ``None``,
       which empties the slot; anything else is a ``TypeError``.
    3. If the value is a :class:`Module` → stored in ``_modules``.
    4. If the name is a registered submodule → the value must be ``None``.
    5. If the value is a :class:`~lucid.nn.Buffer` → registered as a buffer,
       persistent or not as the ``Buffer`` says.
       If the name is a registered buffer → the value must be a
       :class:`~lucid.Tensor` or ``None``, and the buffer keeps its
       persistence.
    6. Otherwise → plain Python attribute.

    To register a non-parameter tensor (e.g. a running mean), call
    :meth:`register_buffer` explicitly.  Assigning to a name that is
    already registered replaces the value *in its slot*: the name keeps
    its position in :meth:`named_parameters`, :meth:`named_buffers`,
    :meth:`named_children` and :meth:`state_dict`, so
    ``layer.weight += 1`` under :func:`lucid.no_grad` leaves the order an
    optimizer or a position-paired average was built from untouched.  A
    name leaves its registry through ``del``, not through assignment.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> class MLP(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10, 20)
    ...         self.fc2 = nn.Linear(20, 1)
    ...
    ...     def forward(self, x):
    ...         return self.fc2(lucid.relu(self.fc1(x)))
    ...
    >>> model = MLP()
    >>> model(lucid.randn(4, 10)).shape
    (4, 1)
    """

    training: bool
    _parameters: OrderedDict[str, Parameter | None]
    _buffers: OrderedDict[str, Tensor | None]
    _modules: OrderedDict[str, Module | None]
    _non_persistent_buffers: set[str]
    _forward_pre_hooks: OrderedDict[int, Callable[..., object]]
    _forward_hooks: OrderedDict[int, Callable[..., object]]
    _forward_pre_hooks_with_kwargs: set[int]
    _forward_hooks_with_kwargs: set[int]
    _forward_hooks_always_called: set[int]
    _backward_pre_hooks: OrderedDict[int, Callable[..., object]]
    _backward_hooks: OrderedDict[int, Callable[..., object]]
    _load_state_dict_pre_hooks: OrderedDict[int, Callable[..., object]]
    _load_state_dict_post_hooks: OrderedDict[int, Callable[..., object]]

    def __init__(self) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        object.__setattr__(self, "_non_persistent_buffers", set())
        object.__setattr__(self, "_forward_pre_hooks_with_kwargs", set())
        object.__setattr__(self, "_forward_hooks_with_kwargs", set())
        object.__setattr__(self, "_forward_hooks_always_called", set())
        for name in _TRACKED_REGISTRIES:
            object.__setattr__(self, name, _TrackedDict())
        object.__setattr__(self, "_load_state_dict_pre_hooks", OrderedDict())
        object.__setattr__(self, "_load_state_dict_post_hooks", OrderedDict())
        object.__setattr__(self, "training", True)

    # ``(*Any, **Any) -> Any`` is the one signature every narrower override is
    # compatible with; the precise type of a call comes from ``__call__``
    # below, which reads it off the subclass's own ``forward``.
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Override in subclasses to define the computation."""
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")

    def _call_impl(self, *args: Tensor, **kwargs: object) -> _ModuleOutput:
        """Run :meth:`forward` with the registered hooks around it.

        This is what ``module(...)`` executes; it is bound as ``__call__``.
        """
        # 3.2.2 fast path: 99 % of training-loop ``forward()`` calls have
        # zero hooks attached.  Short-circuit when neither global nor
        # local hooks are registered so the four dict-iteration loops
        # below are skipped entirely.  Measured: ~15–20 µs / call saved
        # on hookless modules; over a ResNet-18 forward (50+ module
        # calls) that's ~1 ms / batch.  Hooks land via
        # ``register_forward_pre_hook`` / ``register_forward_hook`` /
        # ``register_full_backward_pre_hook`` / etc.; the cached
        # ``_has_backward_hooks()`` already covers the backward side.
        if (
            not _GLOBAL_FORWARD_PRE_HOOKS
            and not self._forward_pre_hooks
            and not _GLOBAL_FORWARD_HOOKS
            and not self._forward_hooks
            and not self._has_backward_hooks()
        ):
            result: _ModuleOutput = self.forward(*args, **kwargs)
            return result

        for hook, with_kwargs in _GLOBAL_FORWARD_PRE_HOOKS.values():
            args, kwargs = self._call_forward_pre_hook(hook, args, kwargs, with_kwargs=with_kwargs)
        for key, hook in self._forward_pre_hooks.items():
            args, kwargs = self._call_forward_pre_hook(
                hook, args, kwargs, with_kwargs=key in self._forward_pre_hooks_with_kwargs
            )

        backward_state = None
        if self._has_backward_hooks():
            args, backward_state = self._prepare_backward_hooks(args)

        output: _ModuleOutput | None = None
        try:
            output = self.forward(*args, **kwargs)
        except Exception:
            self._call_always_forward_hooks(args, kwargs, output)
            raise

        for hook, with_kwargs, _ in _GLOBAL_FORWARD_HOOKS.values():
            output = self._call_forward_hook(hook, args, kwargs, output, with_kwargs=with_kwargs)
        for key, hook in self._forward_hooks.items():
            output = self._call_forward_hook(
                hook, args, kwargs, output, with_kwargs=key in self._forward_hooks_with_kwargs
            )
        if backward_state is not None:
            output = self._attach_output_backward_hooks(output, backward_state)
        return output

    if TYPE_CHECKING:

        def __call__[**P, R](self: _HasForward[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
            """Run :meth:`forward` with the registered hooks around it.

            Always call the module rather than :meth:`forward` directly:
            forward pre-hooks, forward hooks and backward hooks only run on
            this path.  The arguments and the return type are those of the
            subclass's own :meth:`forward`.
            """
            ...

    else:
        __call__ = _call_impl

    def _call_forward_pre_hook(
        self,
        hook: Callable[..., object],
        args: tuple[Tensor, ...],
        kwargs: dict[str, object],
        *,
        with_kwargs: bool,
    ) -> tuple[tuple[Tensor, ...], dict[str, object]]:
        if with_kwargs:
            result = hook(self, args, kwargs)
            if result is None:
                return args, kwargs
            if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict):
                raise RuntimeError(
                    "forward pre-hook with kwargs must return None or (args, kwargs)"
                )
            new_args = result[0]
            if not isinstance(new_args, tuple):
                new_args = (new_args,)
            return new_args, result[1]

        result = hook(self, args)
        if result is None:
            return args, kwargs
        if not isinstance(result, tuple):
            result = (result,)
        return result, kwargs

    def _call_forward_hook(
        self,
        hook: Callable[..., object],
        args: tuple[Tensor, ...],
        kwargs: dict[str, object],
        output: _ModuleOutput,
        *,
        with_kwargs: bool,
    ) -> _ModuleOutput:
        if with_kwargs:
            hook_out = hook(self, args, kwargs, output)
        else:
            hook_out = hook(self, args, output)
        return output if hook_out is None else hook_out  # type: ignore[return-value]

    def _call_always_forward_hooks(
        self, args: tuple[Tensor, ...], kwargs: dict[str, object], output: _ModuleOutput | None
    ) -> None:
        """Run the ``always_call`` forward hooks after ``forward`` raised.

        Each hook runs even when an earlier one fails, and none may mask
        the forward error the caller re-raises: a hook's own error is
        reported as a warning instead.
        """
        calls: list[tuple[Callable[..., object], bool]] = [
            (hook, with_kwargs)
            for hook, with_kwargs, always_call in _GLOBAL_FORWARD_HOOKS.values()
            if always_call
        ]
        calls.extend(
            (hook, key in self._forward_hooks_with_kwargs)
            for key, hook in self._forward_hooks.items()
            if key in self._forward_hooks_always_called
        )
        for hook, with_kwargs in calls:
            try:
                if with_kwargs:
                    hook(self, args, kwargs, output)
                else:
                    hook(self, args, output)
            # A user callback may raise any class, its own included, and
            # whatever escapes here replaces the forward error (LCD-464);
            # the reference catches Exception at this site too.
            except Exception as err:
                warnings.warn(
                    f"forward hook {hook!r} with always_call=True raised {err!r}; "
                    "it was suppressed because forward() had already raised",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _has_backward_hooks(self) -> bool:
        return bool(
            self._backward_hooks
            or self._backward_pre_hooks
            or _GLOBAL_BACKWARD_HOOKS
            or _GLOBAL_BACKWARD_PRE_HOOKS
        )

    def _prepare_backward_hooks(
        self, args: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], object]:
        # The engine state is owned by the call's graph alone (LCD-167): the
        # runners keep the module alive as long as the graph is, and nothing
        # Python holds points back at the engine state, so dropping the
        # graph frees the call.  A runner object that also held the engine
        # state was a cycle through C++ the garbage collector cannot see.
        runners = _ModuleBackwardState(self)
        engine_state: object = _C_engine._create_module_backward_hook_state(  # type: ignore[attr-defined]
            len(args), runners.run_pre_hooks, runners.run_full_hooks
        )
        given: tuple[object, ...] = args  # a forward is given anything at run time
        entries: list[tuple[int, _C_engine.TensorImpl]] = [
            (idx, _unwrap(arg))
            for idx, arg in enumerate(given)
            if isinstance(arg, Tensor) and arg.requires_grad
        ]
        if not entries:
            return args, engine_state
        wrapped_impls = _C_engine._wrap_module_backward_inputs(  # type: ignore[attr-defined]
            engine_state, entries
        )
        wrapped_args = list(args)
        for (idx, _), impl in zip(entries, wrapped_impls, strict=True):
            wrapped_args[idx] = _wrap(impl)
        return tuple(wrapped_args), engine_state

    def _attach_output_backward_hooks(
        self, output: _ModuleOutput, engine_state: object
    ) -> _ModuleOutput:
        if isinstance(output, tuple):
            items: tuple[object, ...] = output  # it returns anything, too
            n_outputs = sum(1 for item in items if isinstance(item, Tensor))
            entries: list[tuple[int, _C_engine.TensorImpl]] = []
            output_idx = 0
            positions: list[int] = []
            for pos, item in enumerate(items):
                if isinstance(item, Tensor):
                    if item.requires_grad:
                        entries.append((output_idx, _unwrap(item)))
                        positions.append(pos)
                    output_idx += 1
            if not entries:
                return output
            wrapped_impls = _C_engine._wrap_module_backward_outputs(  # type: ignore[attr-defined]
                engine_state, entries, n_outputs
            )
            wrapped_output = list(output)
            for pos, impl in zip(positions, wrapped_impls, strict=True):
                wrapped_output[pos] = _wrap(impl)
            return tuple(wrapped_output)

        result: object = output
        if not isinstance(result, Tensor) or not result.requires_grad:
            return output
        wrapped_impl = _C_engine._wrap_module_backward_outputs(  # type: ignore[attr-defined]
            engine_state, [(0, _unwrap(output))], 1
        )[0]
        return _wrap(wrapped_impl)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Attribute setter; routes Parameter / Module / buffer assignments into their registries.

        The rules are *Attribute routing* on the class.  A name already in
        a registry is written over in its slot, never deleted and re-added:
        re-adding moved it to the end, so ``weight += 1`` under ``no_grad``
        — which Python spells as assigning the same Parameter back — turned
        a Linear's ``['weight', 'bias']`` into ``['bias', 'weight']``, and
        whatever paired two modules' parameters by position (an EMA target,
        an optimizer's per-parameter state, a checkpoint's key order) paired
        the wrong tensors without a word (LCD-551).
        """
        if not isinstance(name, str):
            raise TypeError("module attribute name must be a string")
        if "." in name:
            raise KeyError("module attribute name cannot contain '.'")
        if name == "":
            raise KeyError("module attribute name cannot be empty")

        if self._setattr_before_init(name, value) or self._write_traced_buffer(name, value):
            return

        if isinstance(value, Parameter):
            self._vacate(name, keep=self._parameters)
            self._parameters[name] = value
        elif name in self._parameters:
            _refuse_unless_none(value, name, "parameter", "Parameter")
            self._parameters[name] = None
        elif isinstance(value, Module):
            self._vacate(name, keep=self._modules)
            self._modules[name] = value
        elif isinstance(value, _ModuleWrapper):
            # The wrapped model joins the tree under this name — every walk
            # (parameters, state_dict, ``.to()``, ``.train()``) reaches it as
            # if it had been assigned bare — and the attribute, found in
            # ``__dict__`` before ``__getattr__`` is consulted, answers with
            # the wrapper.  Held only as a plain attribute, the model was
            # outside the tree: an optimizer built from the parent missed
            # its parameters and a checkpoint missed its weights.
            self._vacate(name, keep=self._modules)
            self._modules[name] = value.model
            object.__setattr__(self, name, value)
        elif name in self._modules:
            _refuse_unless_none(value, name, "child module", "Module")
            self.__dict__.pop(name, None)  # a compiled wrapper's
            self._modules[name] = None
        elif isinstance(value, Buffer) or name in self._buffers:
            self._assign_buffer(name, value)
        else:
            object.__setattr__(self, name, value)
        _tree_changed()

    def _assign_buffer(self, name: str, value: object) -> None:
        """Assign to buffer ``name``, or register ``value`` as one if it is a :class:`Buffer`.

        A ``Buffer`` says how it is kept, as ``register_buffer`` would be told.
        Any other value goes to a registered buffer's slot: the buffer stays
        a buffer, with its persistence, until it is deleted — spectral_norm's
        power iteration and every running statistic are kept by assignment,
        and a buffer demoted to a plain attribute drops out of ``state_dict``
        and ``.to()``.  ``None`` empties the slot, the way optional
        statistics are spelled.
        """
        if isinstance(value, Buffer):
            self.register_buffer(name, value, value.persistent)
            return
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"cannot assign '{type(value).__name__}' as buffer "
                f"'{name}' (Tensor or None expected)"
            )
        self._buffers[name] = value

    def _setattr_before_init(self, name: str, value: object) -> bool:
        """True, and already written, if ``Module.__init__`` hasn't run yet.

        None of the tracked registries exist before that -- every branch
        in ``__setattr__`` past this point reads one -- so a plain value
        is written directly and a value that would need a registry is
        refused with the reference's own wording (LCD-556).
        """
        if "_parameters" in self.__dict__:
            return False
        kind = _before_init_kind(value)
        if kind is not None:
            raise AttributeError(f"cannot assign {kind} before Module.__init__() call")
        object.__setattr__(self, name, value)
        return True

    def _write_traced_buffer(self, name: str, value: object) -> bool:
        """Under a compile trace, write ``value`` into buffer ``name`` in place.

        The executable reads the buffer it was traced against, so a
        replacement — spectral_norm's ``u`` / ``v`` every forward, a running
        average kept by reassignment — left every compiled call reading the
        trace-time value.  Written in place, the trace records the write and
        the compiled call carries it back
        (``lucid/compile/_core/buffer_writes.py``).  Returns whether it wrote.
        """
        current = self._buffers.get(name)
        if (
            isinstance(value, Tensor)
            and current is not None
            and current is not value
            and _C_engine.compile.current_tracer() is not None
            and current.shape == value.shape
            and current.dtype == value.dtype
            and current.device == value.device
        ):
            current.copy_(value)
            return True
        return False

    def _vacate(self, name: str, keep: Mapping[str, object]) -> None:
        """Clear ``name`` from ``__dict__`` and from every registry but ``keep``.

        ``keep``'s own entry is left where it is, so the write that follows
        replaces it in place.  ``__dict__`` is found before ``__getattr__``
        is asked, so a plain entry left there — ``Conv2d(bias=False)`` sets
        ``self.bias = None`` before any Parameter, a compiled submodule's
        wrapper sits beside its model — would go on answering for the name.
        """
        self.__dict__.pop(name, None)
        if keep is not self._parameters and name in self._parameters:
            del self._parameters[name]
        if keep is not self._buffers and name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        if keep is not self._modules and name in self._modules:
            del self._modules[name]

    def __setstate__(self, state: object) -> None:
        """Restore a pickled or copied module, its registries tracked.

        ``state`` is what the default reduce produced: the instance
        ``__dict__``, or a ``(dict, slots)`` pair for a subclass with
        ``__slots__``.  A module pickled before its registries were
        ``_TrackedDict``\\ s brings plain ones back, whose writes the
        compiled cache would not see; they are converted.
        """
        dict_state, slot_state = state if isinstance(state, tuple) else (state, None)
        if isinstance(dict_state, dict):
            self.__dict__.update(dict_state)
        if isinstance(slot_state, dict):
            for name, value in slot_state.items():
                object.__setattr__(self, name, value)
        for name in _TRACKED_REGISTRIES:
            registry = self.__dict__.get(name)
            if isinstance(registry, dict) and not isinstance(registry, _TrackedDict):
                self.__dict__[name] = _TrackedDict(registry)
        _tree_changed()

    def __getattr__(self, name: str) -> Tensor | Parameter | Module:
        """Attribute lookup fallback; resolves Parameter / buffer / submodule registries when ordinary lookup fails.

        Reads via ``self.__dict__.get(...)`` so a registry missing before
        ``Module.__init__`` doesn't recurse back here about itself."""
        d = self.__dict__
        p: dict[str, Tensor | Parameter | Module] | None = d.get("_parameters")
        if p is not None and name in p:
            return p[name]
        m: dict[str, Tensor | Parameter | Module] | None = d.get("_modules")
        if m is not None and name in m:
            return m[name]
        b: dict[str, Tensor | Parameter | Module] | None = d.get("_buffers")
        if b is not None and name in b:
            return b[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @override
    def __delattr__(self, name: str) -> None:
        """Attribute deletion; mirrors :meth:`__setattr__` by removing the entry from the corresponding registry.

        This, not assignment, is how a name leaves its registry; assigning
        it again afterwards registers it anew, at the end.
        """
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        elif name in self._modules:
            del self._modules[name]
            # A wrapped model's wrapper sits in ``__dict__`` beside it.
            self.__dict__.pop(name, None)
        else:
            object.__delattr__(self, name)
        _tree_changed()

    # ── parameter / module / buffer traversal ─────────────────────────────

    #: Attributes a lazy module leaves at ``None`` until its first forward.
    #: Derived rather than declared: the thirteen lazy classes span three
    #: files and three base classes, and a marker each of them has to
    #: remember to set is a marker one of them will forget.
    _PENDING_ATTRS = ("in_features", "in_channels", "num_features")

    def has_uninitialized_parameters(self, recurse: bool = True) -> bool:
        """Whether any module here is still waiting to see an input.

        A lazy layer has no parameters until its first forward, and the
        parameters it eventually builds are new objects.  Anything that
        took a snapshot of :meth:`parameters` before then — an optimiser,
        an EMA, a parameter-server shard — holds a list those objects are
        not in, and will never touch them again.
        """
        modules = self.modules() if recurse else [self]
        return any(
            hasattr(m, "_initialize")
            and any(getattr(m, a, 0) is None for a in Module._PENDING_ATTRS)
            for m in modules
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield all Parameters in this module (and children if recurse=True).

        Warns when a lazy layer has not seen an input yet.

        The list is complete — a lazy layer registers
        :class:`~lucid.nn.parameter.UninitializedParameter` placeholders
        at construction, and those objects are the ones the real weights
        later occupy, so an optimizer built from this list does train
        them.  What is not yet true is their *shape*: they are ``(0,)``
        until the first forward, so anything reading shapes, counting
        elements or flattening them into a vector is reading nothing.

        A warning rather than a refusal because ``zero_grad`` and
        ``requires_grad_`` legitimately walk a tree in this state.  The
        test suite promotes it to an error (see ``filterwarnings`` in
        ``pyproject.toml``), so the gate is strict where a running
        program is merely told.
        """
        if self.has_uninitialized_parameters(recurse=recurse):
            warnings.warn(
                f"{type(self).__name__}.parameters() was read before a lazy "
                "layer had seen an input.  The objects are real and an "
                "optimizer built from them will train normally once the "
                "shapes are known — but they are placeholders of shape "
                "(0,) until then, so anything that reads their shapes or "
                "flattens them now is reading nothing.  Run one forward "
                "pass first, or give the layer its input size explicitly.",
                UninitializedParameterWarning,
                stacklevel=2,
            )
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield ``(qualified_name, Parameter)`` pairs from this module's tree.

        Parameters
        ----------
        prefix : str, optional
            String prepended to every yielded name — used internally by
            the recursive walk to namespace child modules.  Default
            ``""``.
        recurse : bool, optional
            When ``True`` (default), descend into submodules.  When
            ``False``, yield only this module's directly-attached
            parameters.
        remove_duplicate : bool, optional
            When ``True`` (default), each unique :class:`Parameter`
            object is yielded only once even if referenced by multiple
            attributes — matches the reference framework's contract.

        Yields
        ------
        tuple[str, Parameter]
            ``(qualified_dotted_name, Parameter)`` for every parameter in
            scope.
        """
        seen: set[int] = set()
        for name, p in self._parameters.items():
            if p is not None:
                if remove_duplicate:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                yield (f"{prefix}.{name}" if prefix else name), p
        if recurse:
            for mname, m in self._modules.items():
                if m is None:
                    continue
                subprefix = f"{prefix}.{mname}" if prefix else mname
                for full_name, p in m.named_parameters(
                    subprefix, recurse=True, remove_duplicate=False
                ):
                    if remove_duplicate:
                        if id(p) in seen:
                            continue
                        seen.add(id(p))
                    yield full_name, p

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Yield all buffer tensors."""
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Tensor]]:
        """Yield (name, buffer) pairs."""
        seen: set[int] = set()
        for name, b in self._buffers.items():
            if b is not None:
                if remove_duplicate:
                    if id(b) in seen:
                        continue
                    seen.add(id(b))
                yield (f"{prefix}.{name}" if prefix else name), b
        if recurse:
            for mname, m in self._modules.items():
                if m is None:
                    continue
                subprefix = f"{prefix}.{mname}" if prefix else mname
                for full_name, b in m.named_buffers(
                    subprefix, recurse=True, remove_duplicate=False
                ):
                    if remove_duplicate:
                        if id(b) in seen:
                            continue
                        seen.add(id(b))
                    yield full_name, b

    def modules(self) -> Iterator[Module]:
        """Yield this module and all submodules (depth-first)."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self, memo: set[int] | None = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Module]]:
        """Yield (name, module) pairs."""
        if memo is None:
            memo = set()
        if remove_duplicate and id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, m in self._modules.items():
            if m is None:
                continue
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_modules(
                memo=memo, prefix=subprefix, remove_duplicate=remove_duplicate
            )

    def children(self) -> Iterator[Module]:
        """Yield direct child modules."""
        for module in self._modules.values():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield (name, child_module) pairs."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    # ── dotted-path accessors ─────────────────────────────────────────────

    def get_submodule(self, target: str) -> Module:
        """Return submodule at dotted path, e.g. 'encoder.layer.0'."""
        if not target:
            return self
        parts = target.split(".")
        mod: Module = self
        for part in parts:
            if part not in mod._modules:
                raise AttributeError(f"'{type(mod).__name__}' has no submodule '{part}'")
            next_mod = mod._modules[part]
            if next_mod is None:
                raise AttributeError(f"'{type(mod).__name__}' submodule '{part}' is None")
            mod = next_mod
        return mod

    def get_parameter(self, target: str) -> Parameter:
        """Return parameter at dotted path, e.g. 'fc.weight'."""
        parts = target.split(".")
        mod = self.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else self
        name = parts[-1]
        if name not in mod._parameters:
            raise AttributeError(f"'{type(mod).__name__}' has no parameter '{name}'")
        p = mod._parameters[name]
        if p is None:
            raise AttributeError(f"Parameter '{target}' is None")
        return p

    def get_buffer(self, target: str) -> Tensor:
        """Return buffer at dotted path, e.g. 'bn.running_mean'."""
        parts = target.split(".")
        mod = self.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else self
        name = parts[-1]
        if name not in mod._buffers:
            raise AttributeError(f"'{type(mod).__name__}' has no buffer '{name}'")
        b = mod._buffers[name]
        if b is None:
            raise AttributeError(f"Buffer '{target}' is None")
        return b

    # ── registration ─────────────────────────────────────────────────────

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a Parameter under the given name."""
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        self._validate_child_name(name, "parameter")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' object as parameter '{name}'")
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        if name in self._modules:
            del self._modules[name]
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None:
        """Register a buffer tensor. Non-persistent buffers are excluded from state_dict."""
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        self._validate_child_name(name, "buffer")
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"cannot assign '{type(tensor).__name__}' object as buffer '{name}'")
        if name in self._parameters:
            del self._parameters[name]
        if name in self._modules:
            del self._modules[name]
        if not persistent:
            self._non_persistent_buffers.add(name)
        else:
            self._non_persistent_buffers.discard(name)
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Module | None) -> None:
        """Add a child module."""
        self._validate_child_name(name, "module")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        if module is not None and not isinstance(module, Module):
            raise TypeError(
                f"cannot assign '{type(module).__name__}' object as child module '{name}'"
            )
        if name in self._parameters:
            del self._parameters[name]
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        # Past the check above, an entry in ``__dict__`` can only be a
        # compiled wrapper of the child being replaced, and it would go on
        # answering for the name.
        self.__dict__.pop(name, None)
        self._modules[name] = module

    def register_module(self, name: str, module: Module | None) -> None:
        """Alias for add_module."""
        self.add_module(name, module)

    def _validate_child_name(self, name: str, kind: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"{kind} name must be a string")
        if name == "":
            raise KeyError(f"{kind} name cannot be empty")
        if "." in name:
            raise KeyError(f"{kind} name cannot contain '.'")

    # ── training mode ────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> Self:
        """Set this module and all children to training mode."""
        if not isinstance(mode, bool):
            raise TypeError(f"train() requires a bool, got {type(mode).__name__}")
        self.training = mode
        for m in self._modules.values():
            if m is not None:
                m.train(mode)
        return self

    def eval(self) -> Self:
        """Set this module and all children to evaluation mode."""
        return self.train(False)

    # ── device / dtype conversion via _apply ─────────────────────────────

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply fn to every parameter/buffer in-place (preserves object identity)."""
        for module in self.children():
            module._apply(fn)
        return self._apply_local(fn)

    def _apply_local(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply fn to direct params and buffers; unchanged ones stay (LCD-601)."""
        replaced = False
        for param in self._parameters.values():
            if param is not None:
                replaced = _apply_to_parameter(param, fn) or replaced
        for key, buf in self._buffers.items():
            if buf is None:
                continue
            converted = fn(buf)
            if converted._impl is buf._impl:
                continue
            if not buf._impl.is_phantom and converted._impl.is_phantom:
                raise _C_engine.LucidError("phantom tensor has no values: _apply")
            self._buffers[key] = converted
            replaced = True
        if replaced:
            _tensors_replaced(self)
        return self

    def to(self, *args: object, **kwargs: object) -> Self:
        """Move/cast all parameters and buffers, preserving Parameter object identity.

        Takes what :meth:`lucid.Tensor.to` takes: a device, a dtype, both, or
        a tensor whose device and dtype to match.  A device moves every
        tensor.  A dtype must be floating point or complex, and it casts the
        floating and complex tensors only — an integer or bool buffer such as
        ``BatchNorm.num_batches_tracked`` keeps its dtype, as in the reference
        framework, so a checkpoint round-trip does not change its type.

        Raises
        ------
        TypeError
            If the dtype is bool or an integer, or the arguments are not one
            of the ``.to`` forms.
        """
        device, dtype, copy = _to_engine_target(args, kwargs, Tensor)
        if dtype is not None and _promotion.is_integral(dtype):
            raise TypeError(
                "nn.Module.to only accepts floating point or complex dtypes, "
                f"but got desired dtype={_ENGINE_TO_DTYPE[dtype]}"
            )

        def _convert(t: Tensor) -> Tensor:
            keeps_dtype = _promotion.is_integral(t._impl.dtype)
            return t.to(device, None if keeps_dtype else dtype, copy=copy)

        return self._apply(_convert)

    def metal(self) -> Self:
        """Move all parameters and buffers to Apple Metal GPU."""
        return self.to("metal")

    def cpu(self) -> Self:
        """Move all parameters and buffers to CPU."""
        return self.to("cpu")

    def half(self) -> Self:
        """Cast all parameters and buffers to float16."""
        from lucid._dtype import float16

        return self.to(float16)

    def float(self) -> Self:
        """Cast all parameters and buffers to float32."""
        from lucid._dtype import float32

        return self.to(float32)

    def double(self) -> Self:
        """Cast all parameters and buffers to float64."""
        from lucid._dtype import float64

        return self.to(float64)

    def bfloat16(self) -> Self:
        """Cast all parameters and buffers to bfloat16."""
        from lucid._dtype import bfloat16

        return self.to(bfloat16)

    def type(self, dst_type: _dtype_cls | _C_engine.Dtype | str) -> Self:
        """Cast every parameter and buffer with :meth:`lucid.Tensor.type`.

        Unlike :meth:`to`, this casts the integer and bool buffers too —
        ``BatchNorm.num_batches_tracked`` included — as the reference does.

        Parameters
        ----------
        dst_type : lucid.dtype or str
            A dtype, which keeps each tensor's device, or a legacy type name,
            which moves it to the name's: ``"lucid.DoubleTensor"`` to the CPU,
            ``"lucid.metal.HalfTensor"`` to Metal, as the reference's names do.

        Returns
        -------
        Module
            ``self``, its Parameter objects kept.

        Raises
        ------
        ValueError
            If ``dst_type`` names no legacy type (``"float64"``, ``float``).
        TypeError
            If ``dst_type`` is neither a dtype, a string nor a type.
        RuntimeError
            If ``dst_type`` is an integer or bool dtype and a parameter
            requires grad, which only a floating or complex tensor can.
            Nothing is cast then.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> nn.Linear(2, 2).type("lucid.HalfTensor").weight.dtype
        lucid.float16
        """
        _, target = _type_target(dst_type)
        if _promotion.is_integral(target):
            trainable = [n for n, p in self.named_parameters() if p.requires_grad]
            if trainable:
                raise RuntimeError(
                    f"Module.type: parameter {trainable[0]!r} requires grad, which "
                    "only a floating point or complex tensor can, so it cannot "
                    f"become {_ENGINE_TO_DTYPE[target]}"
                )
        return self._apply(lambda t: t.type(dst_type))

    def apply(self, fn: Callable[[Module], None]) -> Self:
        """Apply fn recursively to every submodule (including self)."""
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all parameters; see :meth:`lucid.optim.Optimizer.zero_grad`."""
        zero_grads_(self.parameters(), set_to_none=set_to_none)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad for all parameters."""
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def share_memory(self) -> Self:
        """No-op on Apple Silicon (unified memory is always shared)."""
        return self

    def compile(self, *args: object, **kwargs: object) -> Self:
        """No-op compatibility stub.

        External codepaths often call ``model.compile()`` to opt into JIT
        acceleration; Lucid has no such layer, so this returns ``self``
        unchanged rather than crashing the caller.  Any positional or
        keyword arguments are accepted and ignored.
        """
        del args, kwargs
        return self

    def to_empty(self, *, device: object = None, recurse: bool = True) -> Self:
        """Move parameters and buffers to ``device`` without copying data.

        The reference framework uses ``to_empty`` to materialise a
        model originally constructed on the meta device.  Lucid has no
        meta device, so this method exists for API parity and
        moves parameters and buffers when a device is supplied.

        Parameters
        ----------
        device : object, optional
            Target placement (string, :class:`device`, or engine enum).
            When ``None`` (default) the call is a no-op returning
            ``self``.
        recurse : bool, optional
            Honoured by the underlying :meth:`_apply` call.  Default
            ``True`` — every child module is moved as well.

        Returns
        -------
        Self
            The same module, parameters / buffers now on ``device``.
        """
        if device is None:
            return self
        target, _, _ = _to_engine_target((device,), {}, Tensor)
        if target is None:
            return self

        def _move(t: Tensor) -> Tensor:
            return _wrap(_C_engine.to_device(t._impl, target))

        return self._apply(_move) if recurse else self._apply_local(_move)

    # ── extra state ───────────────────────────────────────────────────────

    def get_extra_state(self) -> object:
        """Return extra state to include in state_dict. Override in subclasses."""
        return None

    def set_extra_state(self, state: object) -> None:
        """Restore extra state loaded from state_dict. Override in subclasses."""
        pass

    # ── state_dict ────────────────────────────────────────────────────────

    # Per-class version tag.  The base class starts at 1 (matching the
    # reference framework), so every module appears in ``_metadata`` with at
    # least ``{"version": 1}``.  Subclasses override to a higher integer when
    # they need backward-compatible key migration in _load_from_state_dict.
    _version: ClassVar[int] = 1

    def state_dict(
        self,
        destination: dict[str, Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Tensor]:
        """Return an ordered dict mapping parameter / buffer names to tensors.

        Includes every learnable parameter, every persistent buffer
        (``persistent=True`` at register time), and every nested
        submodule's contribution under a dotted-path prefix.

        The returned :class:`OrderedDict` carries a ``_metadata``
        attribute mapping ``module_path → {"version": int}`` for every
        module that defines a ``_version`` class attribute.
        :func:`lucid.save` preserves this attribute across disk
        round-trips so version-aware ``_load_from_state_dict`` hooks
        can migrate older checkpoints.

        Parameters
        ----------
        destination : dict, optional
            When supplied, results are written into this dict instead
            of a fresh one.  Useful when composing state dicts from
            multiple modules.  Default ``None``.
        prefix : str, optional
            String prepended to every key — used internally by the
            recursive walk to namespace child modules.  Default ``""``.
        keep_vars : bool, optional
            When ``True`` keep tensors attached to autograd (return
            them as-is); when ``False`` (default) detach them so the
            state dict is safe to serialise / cross threads.

        Returns
        -------
        dict
            OrderedDict mapping qualified parameter / buffer paths to
            their tensor values.
        """
        from lucid.nn._state_dict import _save_to_state_dict, _build_metadata

        if destination is None:
            destination = OrderedDict()
            destination._metadata = _build_metadata(self, prefix)  # type: ignore[attr-defined]
        _save_to_state_dict(self, destination, prefix=prefix, keep_vars=keep_vars)
        return destination

    def _save_to_state_dict(
        self, destination: dict[str, Tensor], prefix: str, keep_vars: bool
    ) -> None:
        """Write this module's own params + persistent buffers into destination.

        Default implementation iterates `_parameters` and persistent buffers.
        Subclasses may override for custom save logic.  Recursion into
        children is handled by the top-level walker, not by this method.
        """
        for name, p in self._parameters.items():
            # A placeholder has no values to save, and writing its
            # zero-element buffer would put an entry in the checkpoint
            # that looks like a real one and restores to shape ``(0,)``.
            # A checkpoint of a layer that has not run yet is empty, and
            # says so.
            if p is not None and not isinstance(p, UninitializedParameter):
                destination[f"{prefix}{name}"] = p if keep_vars else p.detach()
        for name, b in self._buffers.items():
            if b is not None and name not in self._non_persistent_buffers:
                destination[f"{prefix}{name}"] = b if keep_vars else b.detach()
        extra = self.get_extra_state()
        if extra is not None:
            destination[f"{prefix}_extra_state"] = extra  # type: ignore[assignment]

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Default per-module loader.

        Mutates state_dict / missing_keys / unexpected_keys / error_msgs
        in place.  Override in subclasses for custom migration logic
        (e.g. lazy materialization, buffer rename, version migration).
        Recursion into children is handled by the top-level driver.
        """
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Load parameters from a state_dict.

        Calls each module's ``_load_from_state_dict`` recursively.
        Returns ``_IncompatibleKeys(missing_keys, unexpected_keys)`` on success.
        Raises ``RuntimeError`` if ``strict=True`` and any keys are missing
        or unexpected, or if any error_msgs accumulated during loading.

        Parameters
        ----------
        state_dict : dict
            A mapping from parameter/buffer names to tensors.
        strict : bool
            If ``True`` (default) require an exact key match; raise on any
            missing or unexpected keys.
        assign : bool
            If ``True`` replace each parameter/buffer object with the
            loaded tensor, taking its dtype and device; each module's
            ``_load_from_state_dict`` sees it as
            ``local_metadata["assign_to_params_buffers"]``.  If ``False``
            (default) copy data into the existing parameter preserving its
            dtype and device.  Either way a shape that differs from the
            module's is an error.
        """
        from lucid.nn._state_dict import _load_state_dict as _driver

        return _driver(self, state_dict, strict=strict, assign=assign)

    def register_load_state_dict_pre_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a pre-hook called when this module loads state_dict.

        Hook signature:
            hook(module, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None

        The hook may mutate state_dict/missing/unexpected/error_msgs.
        """
        key = _next_hook_id()
        self._load_state_dict_pre_hooks[key] = hook
        return RemovableHandle(self._load_state_dict_pre_hooks, key)  # type: ignore[arg-type]

    def register_load_state_dict_post_hook(self, hook: Callable[..., object]) -> RemovableHandle:
        """Register a post-hook called after this module loads state_dict.

        Hook signature: ``hook(module, incompatible_keys) -> None``.
        """
        key = _next_hook_id()
        self._load_state_dict_post_hooks[key] = hook
        return RemovableHandle(self._load_state_dict_post_hooks, key)  # type: ignore[arg-type]

    # ── hooks ─────────────────────────────────────────────────────────────

    @overload
    def register_forward_pre_hook(
        self, hook: _ForwardPreHook, *, prepend: bool = False, with_kwargs: Literal[False] = False
    ) -> RemovableHandle: ...

    @overload
    def register_forward_pre_hook(
        self, hook: _ForwardPreHookWithKwargs, *, prepend: bool = False, with_kwargs: Literal[True]
    ) -> RemovableHandle: ...

    def register_forward_pre_hook(
        self,
        hook: _ForwardPreHook | _ForwardPreHookWithKwargs,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        """Register a hook called before forward().

        Parameters
        ----------
        hook : callable
            ``hook(module, args)`` returning ``None`` or the new ``args``; with
            ``with_kwargs``, ``hook(module, args, kwargs)`` returning ``None`` or the
            new ``(args, kwargs)``.
        prepend : bool, default=False
            Run before the hooks already registered.
        with_kwargs : bool, default=False
            Pass the keyword arguments to the hook too.

        Returns
        -------
        RemovableHandle
            ``handle.remove()`` unregisters the hook.
        """
        key = _next_hook_id()
        self._forward_pre_hooks[key] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs.add(key)
        if prepend:
            self._forward_pre_hooks.move_to_end(key, last=False)
        return RemovableHandle(
            self._forward_pre_hooks,  # type: ignore[arg-type]
            key,
            (self._forward_pre_hooks_with_kwargs,),
        )

    def register_forward_hook(
        self,
        hook: _ForwardHook | _ForwardHookWithKwargs,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        """Register a hook called after forward()."""
        key = _next_hook_id()
        self._forward_hooks[key] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs.add(key)
        if always_call:
            self._forward_hooks_always_called.add(key)
        if prepend:
            self._forward_hooks.move_to_end(key, last=False)
        return RemovableHandle(
            self._forward_hooks,  # type: ignore[arg-type]  # safe: RemovableHandle only pops by key
            key,
            (self._forward_hooks_with_kwargs, self._forward_hooks_always_called),
        )

    def register_full_backward_pre_hook(
        self, hook: _BackwardHook, *, prepend: bool = False
    ) -> RemovableHandle:
        """Register a hook to be called before backward hooks."""
        key = _next_hook_id()
        self._backward_pre_hooks[key] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(key, last=False)
        return RemovableHandle(self._backward_pre_hooks, key)  # type: ignore[arg-type]

    def register_full_backward_hook(
        self, hook: _BackwardHook, *, prepend: bool = False
    ) -> RemovableHandle:
        """Register a backward hook. Returns a RemovableHandle."""
        key = _next_hook_id()
        self._backward_hooks[key] = hook
        if prepend:
            self._backward_hooks.move_to_end(key, last=False)
        return RemovableHandle(self._backward_hooks, key)  # type: ignore[arg-type]

    def register_backward_hook(self, hook: _BackwardHook) -> RemovableHandle:
        """Deprecated alias for register_full_backward_hook."""
        return self.register_full_backward_hook(hook)

    # ── repr ──────────────────────────────────────────────────────────────

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        extra = self.extra_repr()
        cls_name = type(self).__name__
        if not self._modules:
            return f"{cls_name}({extra})" if extra else f"{cls_name}()"
        lines = [f"{cls_name}("]
        if extra:
            lines.append(f"  {extra}")
        for name, m in self._modules.items():
            mod_repr = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Override to add extra repr info (e.g. Linear shows in_features, etc.)."""
        return ""
