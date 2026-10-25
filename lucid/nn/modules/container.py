"""
Container modules: Sequential, ModuleList, ModuleDict.
"""

import operator
from collections import OrderedDict
from collections.abc import (
    Iterable,
    ItemsView,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Iterator, cast, overload, override
from lucid.nn.module import Module, _ModuleWrapper
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor


def _key_at(keys: list[str], idx: int) -> str:
    """The key at list position ``idx``, found the way a list finds it.

    ``idx`` goes through ``operator.index``, so a string or a float is a
    ``TypeError``; a position outside ``[-len, len)`` is an ``IndexError``.
    """
    position = operator.index(idx)
    if not -len(keys) <= position < len(keys):
        raise IndexError(f"index {position} is out of range")
    return keys[position]


@overload
def _as_parameter(value: Tensor) -> Parameter: ...
@overload
def _as_parameter[T](value: T) -> T: ...


def _as_parameter(value: object) -> object:
    """A plain tensor as a :class:`Parameter`; anything else unchanged.

    The reference wraps a plain tensor at every entry point that adds one
    to a :class:`ParameterList` / :class:`ParameterDict` -- the constructor,
    ``append``/``extend``, item assignment and ``update`` -- so this is
    the one place that rule lives, called from each of them.  The two
    overloads above are the honest contract every typed caller relies on
    (``Tensor`` in, ``Parameter`` out); a value that defies its own
    parameter's type hint at runtime (not a tensor at all) comes back
    unchanged below, for the registry's own check to refuse with its usual
    message rather than this function's.
    """
    if isinstance(value, Tensor) and not isinstance(value, Parameter):
        return Parameter(value)
    return value


class Sequential(Module):
    r"""An ordered container of modules applied one after another in sequence.

    ``Sequential`` composes a pipeline of modules so that the output of each
    module is fed as the sole input to the next.  Given modules
    :math:`f_1, f_2, \dots, f_n` the computation is:

    .. math::

        \text{output} = f_n(\cdots f_2(f_1(x)) \cdots)

    This is the most common way to build feedforward models in Lucid.
    Modules are stored internally in an ``OrderedDict`` and indexed either
    by integer position or by string key.

    Parameters
    ----------
    *args : Module or OrderedDict[str, Module]
        - If a single ``OrderedDict[str, Module]`` is passed, the modules are
          registered under their dict keys.
        - If multiple positional ``Module`` arguments are passed, they are
          registered under string-formatted integer indices ``'0'``, ``'1'``,
          ``'2'``, …

    Attributes
    ----------
    _modules : OrderedDict[str, Module | None]
        Internal ordered mapping from string key to child module, inherited
        from ``Module``.  Direct mutation is discouraged — prefer ``append``,
        ``insert``, ``__setitem__``, and ``__delitem__``.

    Methods
    -------
    forward(x)
        Pass ``x`` through each child module in order and return the result.
    append(module)
        Add ``module`` at the end, keyed by the current length.
    extend(modules)
        Append each module in ``modules`` in order.
    insert(index, module)
        Insert ``module`` at position ``index``, shifting subsequent modules.
    __getitem__(idx)
        Return the module at integer position ``idx`` or a new ``Sequential``
        for a slice.
    __setitem__(idx, module)
        Replace the module at integer position ``idx``.
    __delitem__(idx)
        Delete the module at position ``idx`` or the slice, renumbering keys.
    __len__()
        Number of registered modules.
    __iter__()
        Iterate over registered modules in order.

    Notes
    -----
    - Slicing (``seq[1:3]``) returns a new ``Sequential`` containing only
      the sliced modules — keys are preserved from the original.
    - After a deletion the internal keys are renumbered (``_renumber_modules``)
      so that indices remain contiguous integers starting at ``0``.
    - ``Sequential`` does **not** implement custom parameter grouping; every
      ``Parameter`` in every child module is returned by ``parameters()``.

    Examples
    --------
    **CNN feature extractor followed by classifier head:**

    >>> import lucid.nn as nn
    >>> backbone = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ...     nn.Conv2d(64, 128, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d((1, 1)),
    ... )
    >>> head = nn.Linear(128, 10)
    >>> # backbone: (N, 3, H, W) -> (N, 128, 1, 1)

    **Named modules via OrderedDict:**

    >>> from collections import OrderedDict
    >>> model = nn.Sequential(OrderedDict([
    ...     ("conv1", nn.Conv2d(1, 32, 3, padding=1)),
    ...     ("relu1", nn.ReLU()),
    ...     ("conv2", nn.Conv2d(32, 64, 3, padding=1)),
    ...     ("pool",  nn.MaxPool2d(2)),
    ... ]))
    >>> # Access by attribute name, or by position:
    >>> conv = model.conv1
    >>> model[0] is conv
    True

    **Dynamic construction and mutation:**

    >>> layers = [nn.Linear(128, 128) for _ in range(4)]
    >>> mlp = nn.Sequential(*layers)
    >>> mlp.append(nn.Linear(128, 10))   # add output layer
    >>> mlp.insert(0, nn.Flatten())      # prepend flatten
    >>> del mlp[1]                        # remove first hidden layer
    """

    @overload
    def __init__(self, *args: Module) -> None: ...
    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Module | OrderedDict[str, Module]) -> None:  # type: ignore[misc]
        """Initialise the Sequential module. See the class docstring for parameter semantics."""
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):  # type: ignore[assignment]
                self.add_module(str(idx), module)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Apply the contained modules to the input.

        Parameters
        ----------
        input : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor produced by the contained modules.
        """
        for m in self._modules.values():
            assert m is not None
            input = cast(Tensor, m(input))
        return input

    @overload
    def __getitem__(self, idx: int) -> Module: ...
    @overload
    def __getitem__(self, idx: slice) -> Sequential: ...

    def __getitem__(self, idx: int | slice) -> Module | Sequential:
        """Return the child module(s) at the given index or slice."""
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            return Sequential(
                OrderedDict(
                    (k, m) for k in keys[idx] if (m := self._modules[k]) is not None
                )
            )
        m = self._modules[keys[idx]]
        assert m is not None
        return m

    def __setitem__(self, idx: int, module: Module) -> None:
        """Replace the child module at the given index.

        Assigned through ``setattr`` like any other child: it keeps its
        turn in the forward, a value that is not a :class:`Module` is a
        ``TypeError``, and a compiled wrapper of the child it replaces
        stops answering for the index.
        """
        setattr(self, _key_at(list(self._modules), idx), module)

    def __delitem__(self, idx: int | slice) -> None:
        """Remove the child module at the given index or slice.

        Through ``delattr`` -- not a direct ``_modules`` deletion -- so a
        compiled child's wrapper (the ``_ModuleWrapper`` branch of
        ``Module.__setattr__`` keeps it in ``__dict__`` under this same
        key) is cleared along with the registry entry, instead of
        lingering to clash with whatever key the renumbering below
        assigns next (LCD-571).
        """
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                delattr(self, key)
        else:
            delattr(self, keys[idx])
        self._renumber_modules()

    def __len__(self) -> int:
        """Return the number of registered child modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over the registered child modules."""
        for m in self._modules.values():
            assert m is not None
            yield m

    def append(self, module: Module) -> None:
        """Append a module to the end of the Sequential."""
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: Iterable[Module]) -> None:
        """Append each module from an iterable to the Sequential."""
        for module in modules:
            self.append(module)

    def insert(self, index: int, module: Module) -> None:
        """Insert a module at the given position in the Sequential."""
        n = len(self._modules)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, item in enumerate(items):
            self.add_module(str(i), item)

    def _renumber_modules(self) -> None:
        """Reassign every remaining child to a fresh ``'0', '1', ...`` key.

        A compiled child's wrapper sits in ``__dict__`` under its *old*
        key, separate from the unwrapped model ``_modules`` holds
        (LCD-571): carried along here, through ``setattr`` with the
        wrapper itself, so the wrapper -- not the bare model inside it --
        answers under the new key too, and the old key is cleared before
        any new key is assigned, so it can never clash with one.
        """
        keys = list(self._modules.keys())
        modules = list(self._modules.values())
        wrappers = [self.__dict__.get(key) for key in keys]
        for key in keys:
            delattr(self, key)
        for i, (module, wrapper) in enumerate(zip(modules, wrappers, strict=True)):
            if isinstance(wrapper, _ModuleWrapper):
                setattr(self, str(i), wrapper)
            else:
                self.add_module(str(i), module)


class ModuleList[M: Module = Module](Module):
    r"""A list-like container that registers all child modules with the parent.

    ``ModuleList`` stores an ordered collection of ``Module`` objects and
    makes them visible to the Lucid module system (``parameters()``,
    ``state_dict()``, device transfer, etc.) exactly like named sub-modules
    defined as class attributes.  Unlike ``Sequential``, it does **not**
    define ``forward`` — the user iterates over the list manually and
    controls the data flow.

    Parameters
    ----------
    modules : list[Module] or None, optional
        Initial list of modules to register.  Each module is stored under
        its integer index converted to a string key (``'0'``, ``'1'``, …).
        Pass ``None`` (default) to start with an empty list.

    Attributes
    ----------
    _modules : OrderedDict[str, Module | None]
        Internal ordered mapping from string index to child module.

    Methods
    -------
    append(module)
        Append ``module`` at the end of the list.
    extend(modules)
        Append each module from an iterable in order.
    insert(index, module)
        Insert ``module`` at position ``index``, renumbering subsequent keys.
    __getitem__(idx)
        Return the module at integer position ``idx`` or a new ``ModuleList``
        for a slice.
    __setitem__(idx, module)
        Replace the module at integer position ``idx``.
    __delitem__(idx)
        Delete the module at position ``idx`` (or slice), renumbering keys.
    __len__()
        Number of registered modules.
    __iter__()
        Iterate over registered modules in order.

    Notes
    -----
    - ``forward`` is intentionally **not** implemented — calling it raises
      ``NotImplementedError``.  Design your own data flow in the enclosing
      module's ``forward`` method.
    - Parameters of all registered modules participate in gradient computation
      and are returned by ``parameters()`` and ``named_parameters()`` on the
      parent module.
    - After insertion or deletion the internal indices are kept contiguous by
      ``_renumber_modules``.

    Examples
    --------
    **Ensemble of encoders — manual iteration:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> class EnsembleEncoder(nn.Module):
    ...     def __init__(self, n: int, in_dim: int, out_dim: int) -> None:
    ...         super().__init__()
    ...         self.encoders = nn.ModuleList(
    ...             [nn.Linear(in_dim, out_dim) for _ in range(n)]
    ...         )
    ...
    ...     def forward(self, x: lucid.Tensor) -> lucid.Tensor:
    ...         # Stack outputs from each encoder along a new axis
    ...         outputs = [enc(x) for enc in self.encoders]
    ...         return lucid.stack(outputs, dim=0)
    >>>
    >>> enc = EnsembleEncoder(n=4, in_dim=256, out_dim=64)

    **Dynamic layer collection built incrementally:**

    >>> class DynamicMLP(nn.Module):
    ...     def __init__(self, dims: list[int]) -> None:
    ...         super().__init__()
    ...         self.layers = nn.ModuleList()
    ...         for in_d, out_d in zip(dims[:-1], dims[1:]):
    ...             self.layers.append(nn.Linear(in_d, out_d))
    ...
    ...     def forward(self, x: lucid.Tensor) -> lucid.Tensor:
    ...         for layer in self.layers[:-1]:
    ...             x = lucid.nn.functional.relu(layer(x))
    ...         return self.layers[-1](x)
    >>>
    >>> mlp = DynamicMLP([784, 512, 256, 10])
    """

    def __init__(self, modules: Iterable[M] | None = None) -> None:
        """Initialise the ModuleList module. See the class docstring for parameter semantics."""
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self.add_module(str(i), m)

    @overload
    def __getitem__(self, idx: int) -> M: ...
    @overload
    def __getitem__(self, idx: slice) -> ModuleList[M]: ...

    def __getitem__(self, idx: int | slice) -> M | ModuleList[M]:
        """Return the child module(s) at the given index or slice."""
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            return ModuleList([cast(M, self._modules[k]) for k in keys[idx]])
        return cast(M, self._modules[keys[idx]])

    def __setitem__(self, idx: int, module: M) -> None:
        """Replace the child module at the given index.

        Assigned through ``setattr`` like any other child: a value that is
        not a :class:`Module` is a ``TypeError``, and a compiled wrapper of
        the child it replaces stops answering for the index.
        """
        setattr(self, _key_at(list(self._modules), idx), module)

    def __delitem__(self, idx: int | slice) -> None:
        """Remove the child module at the given index or slice.

        Through ``delattr`` -- not a direct ``_modules`` deletion -- so a
        compiled child's wrapper (the ``_ModuleWrapper`` branch of
        ``Module.__setattr__`` keeps it in ``__dict__`` under this same
        key) is cleared along with the registry entry, instead of
        lingering to clash with whatever key the renumbering below
        assigns next (LCD-571).
        """
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                delattr(self, key)
        else:
            delattr(self, keys[idx])
        self._renumber_modules()

    def __len__(self) -> int:
        """Return the number of registered child modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[M]:
        """Iterate over the registered child modules."""
        return cast(Iterator[M], iter(self._modules.values()))

    def append(self, module: M) -> None:
        """Append a module to the end of the ModuleList."""
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: Iterable[M]) -> None:
        """Append each module from an iterable to the ModuleList."""
        for module in modules:
            self.append(module)

    def insert(self, index: int, module: M) -> None:
        """Insert a module at the given position in the ModuleList."""
        n = len(self._modules)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, item in enumerate(items):
            self.add_module(str(i), item)

    def _renumber_modules(self) -> None:
        """Reassign every remaining child to a fresh ``'0', '1', ...`` key.

        A compiled child's wrapper sits in ``__dict__`` under its *old*
        key, separate from the unwrapped model ``_modules`` holds
        (LCD-571): carried along here, through ``setattr`` with the
        wrapper itself, so the wrapper -- not the bare model inside it --
        answers under the new key too, and the old key is cleared before
        any new key is assigned, so it can never clash with one.
        """
        keys = list(self._modules.keys())
        modules = list(self._modules.values())
        wrappers = [self.__dict__.get(key) for key in keys]
        for key in keys:
            delattr(self, key)
        for i, (module, wrapper) in enumerate(zip(modules, wrappers, strict=True)):
            if isinstance(wrapper, _ModuleWrapper):
                setattr(self, str(i), wrapper)
            else:
                self.add_module(str(i), module)

    @override
    def forward(self, *args: object) -> Tensor:
        """Apply the contained modules to the input.

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
            Output tensor produced by the contained modules.
        """
        raise NotImplementedError("ModuleList has no forward; iterate manually.")


class ModuleDict[M: Module = Module](Module):
    r"""A dict-like container that registers all child modules under string keys.

    ``ModuleDict`` maps arbitrary string keys to ``Module`` objects and
    integrates them fully with the Lucid module system: ``parameters()``,
    ``state_dict()``, and device/dtype transfers all traverse into the
    registered modules transparently.

    Unlike ``Sequential`` and ``ModuleList``, the keys are user-defined
    strings rather than integers, making ``ModuleDict`` well-suited for
    multi-task or multi-branch architectures where branches have semantic
    names.

    ``forward`` is **not** defined — the user dispatches to specific branches
    by key in the enclosing module's ``forward``.

    Parameters
    ----------
    modules : dict[str, Module] or None, optional
        Initial ``{name: module}`` mapping.  Each entry is registered via
        ``add_module(key, module)``.  Pass ``None`` (default) for an
        empty dict.

    Attributes
    ----------
    _modules : OrderedDict[str, Module | None]
        Internal ordered mapping from string key to child module.

    Methods
    -------
    __getitem__(key)
        Return the module stored under ``key``.
    __setitem__(key, module)
        Register ``module`` under ``key`` (calls ``add_module``).
    __delitem__(key)
        Remove the entry for ``key``.
    __contains__(key)
        Test membership.
    __len__()
        Number of registered modules.
    __iter__()
        Iterate over registered keys.
    keys()
        View of all registered keys.
    values()
        View of all registered modules.
    items()
        View of all ``(key, module)`` pairs.
    get(key, default)
        Return the module for ``key`` or ``default`` if absent.
    pop(key)
        Remove and return the module for ``key``.
    clear()
        Remove all entries.
    update(modules)
        Merge another mapping or iterable of ``(key, module)`` pairs.

    Notes
    -----
    - Insertion order is preserved (backed by ``OrderedDict``).
    - ``update`` accepts both ``Mapping[str, Module]`` and
      ``Iterable[tuple[str, Module]]``.

    Examples
    --------
    **Multi-task prediction heads keyed by task name:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> class MultiTaskModel(nn.Module):
    ...     def __init__(self, shared_dim: int) -> None:
    ...         super().__init__()
    ...         self.backbone = nn.Linear(shared_dim, 256)
    ...         self.heads = nn.ModuleDict({
    ...             "classification": nn.Linear(256, 10),
    ...             "regression":     nn.Linear(256, 1),
    ...             "segmentation":   nn.Linear(256, 64),
    ...         })
    ...
    ...     def forward(self, x: lucid.Tensor, task: str) -> lucid.Tensor:
    ...         feat = lucid.nn.functional.relu(self.backbone(x))
    ...         return self.heads[task](feat)
    >>>
    >>> model = MultiTaskModel(shared_dim=512)
    >>> x = lucid.randn(4, 512)
    >>> # Dispatch dynamically at runtime:
    >>> logits = model(x, task="classification")
    >>> logits.shape
    (4, 10)

    **Conditional gating — adding/removing branches at runtime:**

    >>> router = nn.ModuleDict({"low": nn.Linear(64, 32)})
    >>> router["high"] = nn.Linear(64, 128)   # register a new branch
    >>> router.pop("low")                      # remove old branch
    Linear(in_features=64, out_features=32, bias=True)
    >>> for name, branch in router.items():
    ...     print(name, branch)
    high Linear(in_features=64, out_features=128, bias=True)
    """

    def __init__(self, modules: Mapping[str, M] | None = None) -> None:
        """Initialise the ModuleDict module. See the class docstring for parameter semantics."""
        super().__init__()
        if modules is not None:
            for k, m in modules.items():
                self.add_module(k, m)

    def __getitem__(self, key: str) -> M:
        """Return the child module(s) at the given index or slice."""
        return cast(M, self._modules[key])

    def __setitem__(self, key: str, module: M) -> None:
        """Replace the child module at the given index."""
        self.add_module(key, module)

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if the key/module is registered."""
        return key in self._modules

    def __len__(self) -> int:
        """Return the number of registered child modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered child modules."""
        yield from self._modules.keys()

    def __delitem__(self, key: str) -> None:
        """Remove the child module at the given index or slice.

        Through ``delattr`` once ``key`` is known to be present, so a
        compiled child's wrapper (``Module.__setattr__``'s
        ``_ModuleWrapper`` branch keeps it in ``__dict__`` under this
        same key) is cleared along with the registry entry rather than
        left dangling (LCD-571) -- the missing-key case is checked first
        so it still raises ``KeyError``, as a mapping does, rather than
        ``delattr``'s ``AttributeError``.
        """
        if key not in self._modules:
            raise KeyError(key)
        delattr(self, key)

    def keys(self) -> KeysView[str]:
        """Return an iterable over the keys of the ModuleDict."""
        return self._modules.keys()

    def items(self) -> ItemsView[str, M]:
        """Return an iterable of ``(key, module)`` pairs in the ModuleDict."""
        return cast(ItemsView[str, M], self._modules.items())

    def values(self) -> ValuesView[M]:
        """Return an iterable over the modules in the ModuleDict."""
        return cast(ValuesView[M], self._modules.values())

    def get(self, key: str, default: M | None = None) -> M | None:
        """Return the module at ``key``, or ``default`` if ``key`` is absent.

        Checked by membership, not by the stored value being ``None`` --
        an entry explicitly set to ``None`` (a cleared slot that keeps
        its place in :meth:`named_children`) is a present value in its
        own right and must come back as ``None``, not silently swapped
        for ``default``.

        Parameters
        ----------
        key : str
            Name to look up.
        default : Module, optional
            Returned when ``key`` is not registered.  Default ``None``.

        Returns
        -------
        Module or None
            The registered module, ``default`` if ``key`` is absent, or
            ``None`` if the slot at ``key`` was explicitly cleared.
        """
        if key not in self._modules:
            return default
        return cast(M, self._modules[key])

    def pop(self, key: str) -> M:
        """Remove and return the module at the given index from the ModuleDict."""
        return cast(M, self._modules.pop(key))

    def clear(self) -> None:
        """Remove all modules from the ModuleDict."""
        self._modules.clear()

    def update(self, modules: Mapping[str, M] | Iterable[tuple[str, M]]) -> None:
        """Update the ModuleDict with another mapping of modules."""
        items = modules.items() if isinstance(modules, Mapping) else modules
        for key, module in items:
            self.add_module(key, module)

    @override
    def forward(self, *args: object) -> Tensor:
        """Apply the contained modules to the input.

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
            Output tensor produced by the contained modules.
        """
        raise NotImplementedError("ModuleDict has no forward; use indexing.")


class ParameterList(Module):
    r"""A list-like container that registers ``Parameter`` objects with the module.

    ``ParameterList`` stores an ordered sequence of ``Parameter`` leaf tensors
    and registers each one via ``register_parameter``.  This means every stored
    parameter appears in ``parameters()``, participates in gradient tracking,
    and is included in ``state_dict()`` / ``load_state_dict``.

    It is semantically equivalent to assigning a list of ``Parameter``\s as
    attributes of a module, but is the correct approach when the number of
    parameters is data-dependent or only known at runtime.

    Parameters
    ----------
    values : Iterable[Tensor] or None, optional
        Initial iterable of ``Parameter`` objects, or plain tensors -- each
        wrapped in a :class:`Parameter` first, as every entry point that
        adds one does.  Each is registered under its zero-based integer
        index formatted as a string.  Pass ``None`` (default) for an empty
        list.

    Attributes
    ----------
    _parameters : OrderedDict[str, Parameter | None]
        Internal ordered mapping from string index to ``Parameter``, inherited
        from ``Module``.

    Methods
    -------
    append(param)
        Append a ``Parameter`` at the end.
    extend(parameters)
        Append each ``Parameter`` from an iterable in order.
    __getitem__(idx)
        Return the ``Parameter`` at integer position ``idx``.
    __setitem__(idx, param)
        Replace the ``Parameter`` at integer position ``idx``.
    __delitem__(idx)
        Delete the ``Parameter`` at position ``idx`` (or slice), renumbering.
    __len__()
        Number of registered parameters.
    __iter__()
        Iterate over registered ``Parameter`` objects in order.

    Notes
    -----
    - ``forward`` is intentionally **not** implemented.  Access individual
      parameters via indexing: ``self.scales[i]``.
    - Unlike ``ModuleList``, this container holds ``Parameter`` objects
      (leaf tensors with ``requires_grad=True`` by default), not arbitrary
      modules.

    Examples
    --------
    **Per-layer learnable temperature scales (e.g. knowledge distillation):**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> class ScaledDistiller(nn.Module):
    ...     def __init__(self, n_layers: int) -> None:
    ...         super().__init__()
    ...         self.scales = nn.ParameterList(
    ...             [nn.Parameter(lucid.ones(1)) for _ in range(n_layers)]
    ...         )
    ...
    ...     def forward(self, feats: list[lucid.Tensor]) -> list[lucid.Tensor]:
    ...         return [f * self.scales[i] for i, f in enumerate(feats)]
    >>>
    >>> distiller = ScaledDistiller(n_layers=6)

    **Dynamically growing learnable biases:**

    >>> bias_bank = nn.ParameterList()
    >>> for _ in range(4):
    ...     bias_bank.append(nn.Parameter(lucid.zeros(128)))
    >>> # All 4 biases visible to optimizer:
    >>> n_params = sum(p.numel() for p in bias_bank.parameters())
    """

    def __init__(self, values: Iterable[Tensor] | None = None) -> None:
        """Initialise the ParameterList module. See the class docstring for parameter semantics."""
        super().__init__()
        if values is not None:
            for i, p in enumerate(values):
                self.register_parameter(str(i), _as_parameter(p))

    def __getitem__(self, idx: int) -> Parameter | None:
        """Return the parameter at the given index, as the reference does --
        ``None`` if that slot was explicitly cleared by assignment, never
        an assertion failure for reading back what was legitimately put
        there (LCD-571)."""
        keys = list(self._parameters.keys())
        return self._parameters[keys[idx]]

    def __setitem__(self, idx: int, param: Tensor) -> None:
        """Replace the parameter at the given index.

        A plain tensor is wrapped in a :class:`Parameter`, as the reference
        does; anything else that is not a Parameter is a ``TypeError``
        from the assignment.
        """
        setattr(self, _key_at(list(self._parameters), idx), _as_parameter(param))

    def __delitem__(self, idx: int | slice) -> None:
        """Remove the child module at the given index or slice."""
        keys = list(self._parameters.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                del self._parameters[key]
        else:
            del self._parameters[keys[idx]]
        self._renumber_parameters()

    def __len__(self) -> int:
        """Return the number of registered child modules."""
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter | None]:
        """Iterate over the registered parameters, ``None`` included for a
        slot explicitly cleared by assignment (LCD-571)."""
        yield from self._parameters.values()

    def append(self, param: Tensor) -> None:
        """Append a module to the end of the ParameterList.

        A plain tensor is wrapped in a :class:`Parameter`, as the
        reference does for every entry point that adds one.
        """
        self.register_parameter(str(len(self._parameters)), _as_parameter(param))

    def extend(self, parameters: Iterable[Tensor]) -> None:
        """Append each module from an iterable to the ParameterList."""
        for param in parameters:
            self.append(param)

    def _renumber_parameters(self) -> None:
        """Internal helper for the ParameterList module."""
        items = list(self._parameters.values())
        self._parameters.clear()
        for i, param in enumerate(items):
            self.register_parameter(str(i), param)

    @override
    def forward(self, *args: object) -> Tensor:
        """Apply the contained modules to the input.

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
            Output tensor produced by the contained modules.
        """
        raise NotImplementedError(
            "ParameterList has no forward(); it is a container for Parameters only. "
            "Access individual parameters via indexing: self.params[i]"
        )


class ParameterDict(Module):
    r"""A dict-like container that registers ``Parameter`` objects under string keys.

    ``ParameterDict`` maps arbitrary string keys to ``Parameter`` leaf tensors
    and registers each one via ``register_parameter``, making them first-class
    citizens of the Lucid module system: they appear in ``parameters()``,
    ``named_parameters()``, and ``state_dict()``/``load_state_dict``.

    This is the preferred pattern when parameters have meaningful names or when
    the set of parameters is determined programmatically (e.g., one weight
    matrix per attention head type, one scale per branch name).

    Parameters
    ----------
    parameters : dict[str, Tensor] or None, optional
        Initial ``{name: parameter}`` mapping.  Each entry is registered via
        ``register_parameter``; a plain tensor is wrapped in a
        :class:`Parameter` first, as every entry point that adds one does.
        Pass ``None`` (default) for an empty dict.

    Attributes
    ----------
    _parameters : OrderedDict[str, Parameter | None]
        Internal ordered mapping from string key to ``Parameter``, inherited
        from ``Module``.

    Methods
    -------
    __getitem__(key)
        Return the ``Parameter`` stored under ``key``.
    __setitem__(key, param)
        Register ``param`` under ``key`` (calls ``register_parameter``).
    __delitem__(key)
        Remove the entry for ``key``.
    __contains__(key)
        Test membership.
    __len__()
        Number of registered parameters.
    __iter__()
        Iterate over registered keys.
    keys()
        View of all registered keys.
    values()
        View of all registered ``Parameter`` objects.
    items()
        View of all ``(key, Parameter)`` pairs.
    get(key, default)
        Return the ``Parameter`` for ``key`` or ``default`` if absent.
    pop(key)
        Remove and return the ``Parameter`` for ``key``.
    clear()
        Remove all entries.
    update(parameters)
        Merge another mapping or iterable of ``(key, Parameter)`` pairs.

    Notes
    -----
    - ``forward`` is intentionally **not** implemented.  Access parameters
      by key in the enclosing module's ``forward``: ``self.params['query']``.
    - Insertion order is preserved (backed by ``OrderedDict``).

    Examples
    --------
    **Named projection matrices for a custom multi-head attention layer:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> class CustomAttention(nn.Module):
    ...     def __init__(self, dim: int) -> None:
    ...         super().__init__()
    ...         self.weights = nn.ParameterDict({
    ...             "query":  nn.Parameter(lucid.randn(dim, dim)),
    ...             "key":    nn.Parameter(lucid.randn(dim, dim)),
    ...             "value":  nn.Parameter(lucid.randn(dim, dim)),
    ...             "output": nn.Parameter(lucid.randn(dim, dim)),
    ...         })
    ...
    ...     def forward(self, x: lucid.Tensor) -> lucid.Tensor:
    ...         q = x @ self.weights["query"]
    ...         k = x @ self.weights["key"]
    ...         v = x @ self.weights["value"]
    ...         # ... attention logic ...
    ...         return v @ self.weights["output"]

    **Runtime-configured bias bank with named entries:**

    >>> bias_bank = nn.ParameterDict()
    >>> for name in ["low_freq", "mid_freq", "high_freq"]:
    ...     bias_bank[name] = nn.Parameter(lucid.zeros(64))
    >>> bias_bank["low_freq"].shape  # retrieve by name
    (64,)
    >>> _ = bias_bank.pop("mid_freq")  # remove dynamically
    >>> list(bias_bank.keys())
    ['low_freq', 'high_freq']
    """

    def __init__(self, parameters: dict[str, Tensor] | None = None) -> None:
        """Initialise the ParameterDict module. See the class docstring for parameter semantics."""
        super().__init__()
        if parameters is not None:
            for k, p in parameters.items():
                self.register_parameter(k, _as_parameter(p))

    def __getitem__(self, key: str) -> Parameter | None:
        """Return the parameter at ``key``, as the reference does -- ``None``
        if that slot was explicitly cleared by assignment, never an
        assertion failure for reading back what was legitimately put
        there (LCD-571)."""
        return self._parameters[key]

    def __setitem__(self, key: str, param: Tensor) -> None:
        """Register ``param`` under ``key``.

        A plain tensor is wrapped in a :class:`Parameter`, as the
        reference does for every entry point that adds one.
        """
        self.register_parameter(key, _as_parameter(param))

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if the key/module is registered."""
        return key in self._parameters

    def __len__(self) -> int:
        """Return the number of registered child modules."""
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered child modules."""
        yield from self._parameters.keys()

    def __delitem__(self, key: str) -> None:
        """Remove the child module at the given index or slice."""
        del self._parameters[key]

    def keys(self) -> KeysView[str]:
        """Return an iterable over the keys of the ParameterDict."""
        return self._parameters.keys()

    def items(self) -> ItemsView[str, Parameter]:
        """Return an iterable of ``(key, module)`` pairs in the ParameterDict."""
        return cast(ItemsView[str, Parameter], self._parameters.items())

    def values(self) -> ValuesView[Parameter]:
        """Return an iterable over the modules in the ParameterDict."""
        return cast(ValuesView[Parameter], self._parameters.values())

    def get(self, key: str, default: Parameter | None = None) -> Parameter | None:
        """Method on the ParameterDict module."""
        return self._parameters.get(key, default)

    def pop(self, key: str) -> Parameter | None:
        """Remove and return the parameter at ``key``, as the reference does
        -- ``None`` if that slot was explicitly cleared by assignment,
        never an assertion failure (LCD-571)."""
        return self._parameters.pop(key)

    def clear(self) -> None:
        """Remove all modules from the ParameterDict."""
        self._parameters.clear()

    def update(
        self, parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]]
    ) -> None:
        """Update the ParameterDict with another mapping of modules.

        A plain tensor is wrapped in a :class:`Parameter`, as the
        reference does for every entry point that adds one.
        """
        items = parameters.items() if isinstance(parameters, Mapping) else parameters
        for key, param in items:
            self.register_parameter(key, _as_parameter(param))

    @override
    def forward(self, *args: object) -> Tensor:
        """Apply the contained modules to the input.

        Parameters
        ----------
        None

        Returns
        -------
        Tensor
            Output tensor produced by the contained modules.
        """
        raise NotImplementedError(
            "ParameterDict has no forward(); it is a container for Parameters only. "
            "Access individual parameters via key: self.params['key']"
        )
