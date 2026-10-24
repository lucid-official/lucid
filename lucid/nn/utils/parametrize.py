"""Parametrization API — re-derive a tensor of a module from trainable originals.

``register_parametrization(module, "weight", f)`` moves the trainable leaf
to ``module.parametrizations.weight.original`` and turns ``module.weight``
into a property that computes ``f(original)`` on every read.  The originals
are registered tensors of a :class:`ParametrizationList` held in a
registered :class:`~lucid.nn.ModuleDict`, so ``parameters()``,
``state_dict()``, ``.to()``, ``zero_grad()`` and every other walk of the
module tree reach them under ``parametrizations.<name>.original``.

The property lives on a subclass made for the one module
(``Parametrized<Class>``): a property on the shared class would
reparametrise every instance of it.
"""

import copy
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import override

import lucid
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.modules.container import ModuleDict, ModuleList
from lucid.nn.parameter import Parameter

# ``cached()`` nests; the cache lives while any level is open.
_cache_enabled: int = 0
_cache: dict[tuple[int, str], Tensor] = {}


@contextmanager
def cached() -> Iterator[None]:
    r"""Compute each parametrized tensor once while the context is open.

    Inside the block, the first read of ``module.<name>`` computes the
    parametrization and later reads return that tensor, until the
    outermost ``cached()`` exits.  Use it around a forward pass that reads
    a parametrized weight more than once (a recurrent layer reading it each
    time step) when the parametrization is expensive.

    Yields
    ------
    None
        The context has no value; it only switches the cache on.

    Raises
    ------
    RuntimeError
        On a read inside a :func:`lucid.compile` trace: the trace would
        record the cached tensor as a constant and the compiled call would
        never see the originals change.

    Notes
    -----
    Contexts nest.  The cache is emptied when the outermost one exits, so
    a tensor computed inside never outlives it.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import parametrize
    >>> class Double(nn.Module):
    ...     def forward(self, X):
    ...         return 2 * X
    >>> layer = nn.Linear(3, 3)
    >>> _ = parametrize.register_parametrization(layer, "weight", Double())
    >>> with parametrize.cached():
    ...     layer.weight is layer.weight
    True
    """
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache.clear()


def _set_storage(dest: Tensor, src: Tensor) -> None:
    """Give ``dest`` the values of ``src`` in place, keeping ``dest`` the same object.

    The object an optimizer or a hook already holds has to stay the
    trainable leaf, so its storage is swapped rather than the tensor
    replaced.
    """
    if src is not dest:
        dest.data = src


def _register_tensor(module: Module, name: str, value: Tensor) -> None:
    """Register ``value`` as a parameter if it is one, else as a buffer."""
    if isinstance(value, Parameter):
        module.register_parameter(name, value)
    else:
        module.register_buffer(name, value)


def _right_inverse_of(step: Module) -> Callable[..., object] | None:
    """The step's ``right_inverse`` method, or ``None`` if it has none."""
    inverse = getattr(step, "right_inverse", None)
    return inverse if callable(inverse) else None


def _check_like(value: object, reference: Tensor, what: str) -> Tensor:
    """Refuse ``value`` unless it is a tensor of ``reference``'s dtype."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{what} must return a Tensor, got {type(value).__name__}")
    if value.dtype != reference.dtype:
        raise ValueError(
            f"{what} may not change the dtype of the tensor, unless `unsafe` "
            f"flag is enabled.\nunparametrized dtype: {reference.dtype}\n"
            f"parametrized dtype: {value.dtype}"
        )
    return value


def _check_same_shape(value: Tensor, reference: Tensor, what: str) -> None:
    """Refuse ``value`` unless it has ``reference``'s shape."""
    if tuple(value.shape) != tuple(reference.shape):
        raise ValueError(
            f"{what} may not change the shape of the tensor, unless `unsafe` "
            f"flag is enabled.\nunparametrized shape: {tuple(reference.shape)}\n"
            f"parametrized shape: {tuple(value.shape)}"
        )


class ParametrizationList(ModuleList):
    r"""The parametrizations of one tensor, and the originals they read.

    ``module.parametrizations[name]`` is one of these.  It is a
    :class:`~lucid.nn.ModuleList` of the parametrizations in the order
    they were registered, and it holds the trainable originals: one tensor
    ``original``, or ``original0``, ``original1``, … when the first
    parametrization's ``right_inverse`` returns a sequence.  Calling it
    folds the chain over the originals — the value ``module.<name>`` reads.

    Parameters
    ----------
    modules : Sequence[Module]
        The parametrizations, applied first to last.  At least one.
    original : Tensor or Parameter
        The tensor being parametrized.  It becomes ``original`` in place —
        the same object, given the values of the chain's right inverse —
        so an optimizer built before registration still trains it.  A
        :class:`~lucid.nn.Parameter` is registered as a parameter, any
        other tensor as a buffer.
    unsafe : bool, optional
        If ``True``, skip the check that the chain returns a tensor of
        ``original``'s shape and dtype.  Default ``False``.

    Attributes
    ----------
    original : Tensor
        The single original, when ``right_inverse`` returned a tensor.
    is_tensor : bool
        ``True`` when there is one original, ``False`` when there are
        ``ntensors`` of them.
    ntensors : int
        How many originals there are.
    unsafe : bool
        Whether shape and dtype checks are skipped.

    Raises
    ------
    TypeError
        If a right inverse returns something other than a tensor or a
        sequence of tensors, or the chain does not return a tensor.
    ValueError
        If ``modules`` is empty, or (unless ``unsafe``) the chain changes
        the tensor's shape or dtype.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrize import ParametrizationList
    >>> class Double(nn.Module):
    ...     def forward(self, X):
    ...         return 2 * X
    >>> chain = ParametrizationList([Double()], nn.Parameter(lucid.ones(2)))
    >>> chain().tolist()
    [2.0, 2.0]
    """

    original: Tensor
    unsafe: bool
    is_tensor: bool
    ntensors: int

    def __init__(
        self,
        modules: Sequence[Module],
        original: Tensor | Parameter,
        unsafe: bool = False,
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        if len(modules) == 0:
            raise ValueError("ParametrizationList requires one or more modules.")
        super().__init__(modules)
        self.unsafe = unsafe
        # The values ``original`` has now: what the checks compare against,
        # and what it gets back if they refuse, so a refused registration
        # leaves the module's tensor as it was.
        before = original.data
        new: object = original
        with lucid.no_grad():
            for step in reversed(list(self)):
                inverse = _right_inverse_of(step)
                if inverse is None:
                    continue
                # A step whose inverse raises NotImplementedError keeps none;
                # the value passes through it unchanged.
                with suppress(NotImplementedError):
                    new = inverse(new)
        if isinstance(new, Tensor):
            self._hold_one(original, new)
        elif isinstance(new, Sequence):
            self._hold_many(original, new)
        else:
            raise TypeError(
                "'right_inverse' must return a Tensor or a Sequence of tensors "
                f"(list, tuple...). Got {type(new).__name__}"
            )
        if unsafe:
            return
        what = "Registering a parametrization"
        try:
            _check_same_shape(_check_like(self(), before, what), before, what)
        except TypeError, ValueError:
            with lucid.no_grad():
                _set_storage(original, before)
            raise

    def _hold_one(self, original: Tensor, new: Tensor) -> None:
        self.is_tensor = True
        self.ntensors = 1
        _check_like(new, original, "When `right_inverse` outputs one tensor, it")
        with lucid.no_grad():
            _set_storage(original, new)
        _register_tensor(self, "original", original)

    def _hold_many(self, original: Tensor, new: Sequence[object]) -> None:
        self.is_tensor = False
        self.ntensors = len(new)
        for i, piece in enumerate(new):
            if not isinstance(piece, Tensor):
                raise TypeError(
                    "'right_inverse' must return a Tensor or a Sequence of "
                    f"tensors. Got element {i} of type {type(piece).__name__}"
                )
            if isinstance(original, Parameter):
                held: Tensor = Parameter(piece.detach(), original.requires_grad)
            else:
                held = piece.detach().requires_grad_(original.requires_grad)
            _register_tensor(self, f"original{i}", held)

    def _originals(self) -> list[Tensor]:
        if self.is_tensor:
            return [self.original]
        return [getattr(self, f"original{i}") for i in range(self.ntensors)]

    def right_inverse(self, value: Tensor) -> None:
        r"""Set the originals so the chain computes ``value``.

        Runs each parametrization's ``right_inverse`` from last to first
        and writes the result into the originals in place.  This is what
        ``module.<name> = value`` calls.

        Parameters
        ----------
        value : Tensor
            The value ``module.<name>`` should read afterwards (up to what
            the right inverses can represent).

        Raises
        ------
        RuntimeError
            If a parametrization in the chain has no ``right_inverse``.
        TypeError
            If the inverses return something other than tensors.
        ValueError
            If they return the wrong number of tensors, or a dtype other
            than the originals'.
        """
        result: object = value
        with lucid.no_grad():
            for step in reversed(list(self)):
                inverse = _right_inverse_of(step)
                if inverse is None:
                    raise RuntimeError(
                        f"parametrization {type(step).__name__} does not "
                        "implement right_inverse."
                    )
                result = inverse(result)
            if self.is_tensor:
                what = "`right_inverse` should return a tensor; it"
                _set_storage(self.original, _check_like(result, self.original, what))
                return
            if not isinstance(result, Sequence):
                raise TypeError(
                    f"'right_inverse' must return a sequence of {self.ntensors} "
                    f"tensors, got {type(result).__name__}"
                )
            if len(result) != self.ntensors:
                raise ValueError(
                    f"'right_inverse' must return a sequence of {self.ntensors} "
                    f"tensors, got {len(result)}"
                )
            for original, piece in zip(self._originals(), result, strict=True):
                what = "`right_inverse` should return tensors; it"
                _set_storage(original, _check_like(piece, original, what))

    @override
    def forward(self, *args: object) -> Tensor:
        """Fold the parametrizations over the originals.

        The chain reads its own originals, so a call takes no input.
        """
        if args:
            raise TypeError(
                f"ParametrizationList takes no input ({len(args)} given); "
                "it reads its own originals"
            )
        steps = list(self)
        out: object = steps[0](*self._originals())
        for step in steps[1:]:
            out = step(out)
        if not isinstance(out, Tensor):
            raise TypeError(
                f"a parametrization must return a Tensor, got {type(out).__name__}"
            )
        return out


def _parametrizations_of(module: Module) -> ModuleDict | None:
    """``module.parametrizations`` if it is the registry this file installs."""
    held = module._modules.get("parametrizations")
    return held if isinstance(held, ModuleDict) else None


def _chain_in(
    module: Module, tensor_name: str
) -> tuple[ModuleDict, ParametrizationList]:
    """The registry and the chain of ``tensor_name``; ``ValueError`` if absent."""
    registry = _parametrizations_of(module)
    chain = registry.get(tensor_name) if registry is not None else None
    if registry is None or not isinstance(chain, ParametrizationList):
        raise ValueError(
            f"Module {type(module).__name__} does not have a parametrization "
            f"on {tensor_name}"
        )
    return registry, chain


def _chain(module: Module, tensor_name: str) -> ParametrizationList:
    return _chain_in(module, tensor_name)[1]


def _deepcopy_parametrized(self: Module, memo: dict[int, object]) -> Module:
    """Deep-copy a parametrized module, which ``__getstate__`` keeps from pickling."""
    found = memo.get(id(self))
    if isinstance(found, Module):
        return found
    replica = self.__new__(type(self))
    memo[id(self)] = replica
    object.__setattr__(replica, "__dict__", copy.deepcopy(self.__dict__, memo))
    return replica


def _refuse_pickle(self: Module) -> object:
    raise RuntimeError(
        f"Serialization of parametrized modules ({type(self).__name__}) is "
        "only supported through state_dict()."
    )


def _inject_new_class(module: Module) -> None:
    """Move ``module`` onto a subclass of its own, to carry the properties."""
    cls = type(module)

    def setattr_parametrized(self: Module, name: str, value: object) -> None:
        # The property answers for a parametrized name, so a Parameter or
        # Module registered under it would never be read — a weight
        # assigned that way would train nothing.  A plain tensor goes on
        # to the property, which writes the originals.
        registry = _parametrizations_of(self)
        if (
            isinstance(value, (Parameter, Module))
            and registry is not None
            and name in registry
        ):
            raise KeyError(f"attribute '{name}' already exists")
        cls.__setattr__(self, name, value)

    namespace: dict[str, object] = {
        "__getstate__": _refuse_pickle,
        "__setattr__": setattr_parametrized,
    }
    if not hasattr(cls, "__deepcopy__"):
        namespace["__deepcopy__"] = _deepcopy_parametrized
    module.__class__ = type(f"Parametrized{cls.__name__}", (cls,), namespace)


def _read_parametrized(module: Module, tensor_name: str) -> Tensor:
    chain = _chain(module, tensor_name)
    if not _cache_enabled:
        return chain()
    if _C_engine.compile.current_tracer() is not None:
        raise RuntimeError("Cannot trace a model while caching parametrizations.")
    key = (id(module), tensor_name)
    held = _cache.get(key)
    if held is None:
        held = _cache[key] = chain()
    return held


def _inject_property(module: Module, tensor_name: str) -> None:
    def get(self: Module) -> Tensor:
        return _read_parametrized(self, tensor_name)

    def put(self: Module, value: Tensor) -> None:
        _chain(self, tensor_name).right_inverse(value)

    setattr(type(module), tensor_name, property(get, put))


def _check_appended(parametrization: Module, current: Tensor) -> None:
    """Refuse a chained step that changes the tensor's shape or dtype."""
    what = "Registering a parametrization"
    produced = _check_like(parametrization(current), current, what)
    _check_same_shape(produced, current, what)
    inverse = _right_inverse_of(parametrization)
    if inverse is None:
        return
    try:
        restored = inverse(produced)
    except NotImplementedError:
        return  # no inverse to check
    what = "parametrization.right_inverse"
    _check_same_shape(_check_like(restored, current, what), current, what)


def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    r"""Install a parametrization on a parameter or buffer of ``module``.

    After registration ``module.<tensor_name>`` is computed on every read
    as ``parametrization(original)``.  The trainable tensor is
    ``module.parametrizations[tensor_name].original`` — the same object
    that was ``module.<tensor_name>``, so an optimizer built earlier keeps
    training it — and it appears in :meth:`~lucid.nn.Module.parameters`
    and :meth:`~lucid.nn.Module.state_dict` as
    ``parametrizations.<tensor_name>.original``.

    If ``parametrization`` has a ``right_inverse(X)`` method, the original
    is set to ``right_inverse(module.<tensor_name>)`` at registration, so
    the tensor reads the same value before and after.  A ``right_inverse``
    returning a sequence of tensors makes one original per element
    (``original0``, ``original1``, …), and ``forward`` takes them as
    separate arguments.  Assigning ``module.<tensor_name> = value`` calls
    the inverses to set the originals.

    Registering on a name that is already parametrized appends to its
    chain: the read becomes ``g(f(original))`` for registration order
    ``f`` then ``g``.

    Parameters
    ----------
    module : Module
        Module holding the tensor.  Mutated in place; its class becomes a
        ``Parametrized<Class>`` subclass of the original class.
    tensor_name : str
        Name of the parameter or buffer to parametrize.
    parametrization : Module
        Module whose ``forward`` maps the original(s) to the tensor.  It is
        put in ``module``'s training mode.
    unsafe : bool, keyword-only, optional
        If ``True``, skip the checks that the parametrization keeps the
        tensor's shape and dtype.  Default ``False``.

    Returns
    -------
    Module
        ``module``.

    Raises
    ------
    TypeError
        If the parametrization or its right inverse returns something
        other than a tensor (or, for ``right_inverse``, a sequence of them).
    ValueError
        If ``module`` has no parameter, buffer or parametrized tensor named
        ``tensor_name``, or (unless ``unsafe``) the parametrization or its
        right inverse changes the tensor's shape or dtype.

    Notes
    -----
    Mathematically the trained quantity becomes

    .. math::

        \mathbf{W} \;=\; f(\boldsymbol\theta),

    with :math:`\boldsymbol\theta` the original and :math:`f` the
    parametrization; gradients reach :math:`\boldsymbol\theta` through
    :math:`f`.  A parametrized module pickles only through its
    ``state_dict``; :func:`copy.deepcopy` works.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrize import register_parametrization
    >>> class Symmetric(nn.Module):
    ...     def forward(self, X):
    ...         return 0.5 * (X + X.mT)
    >>> layer = nn.Linear(3, 3)
    >>> _ = register_parametrization(layer, "weight", Symmetric())
    >>> [name for name, _ in layer.named_parameters()]
    ['bias', 'parametrizations.weight.original']
    >>> bool((layer.weight == layer.weight.mT).all())
    True
    """
    parametrization.train(module.training)
    registry = _parametrizations_of(module)
    if registry is not None and tensor_name in registry:
        chain = _chain(module, tensor_name)
        if not unsafe:
            _check_appended(parametrization, _read_parametrized(module, tensor_name))
        chain.append(parametrization)
        chain.unsafe |= unsafe
        return module
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(
            f"Module '{type(module).__name__}' does not have a parameter, a "
            f"buffer, or a parametrized element with name '{tensor_name}'"
        )
    held = module._parameters.get(tensor_name, module._buffers.get(tensor_name))
    if held is None:
        raise ValueError(f"cannot parametrize '{tensor_name}': it is None")
    chain = ParametrizationList([parametrization], held, unsafe=unsafe)
    delattr(module, tensor_name)
    if registry is None:
        _inject_new_class(module)
        registry = ModuleDict()
        module.add_module("parametrizations", registry)
    _inject_property(module, tensor_name)
    registry[tensor_name] = chain
    return module


def is_parametrized(module: Module, tensor_name: str | None = None) -> bool:
    r"""Whether ``module`` has a parametrization (on ``tensor_name``, if given).

    Parameters
    ----------
    module : Module
        Module to inspect.
    tensor_name : str, optional
        If given, whether that tensor is parametrized.  When ``None`` (the
        default), whether any tensor of ``module`` is.

    Returns
    -------
    bool
        ``True`` if a parametrization matches the query.

    Notes
    -----
    Reads the registry only; no parametrization is computed.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrize import (
    ...     is_parametrized,
    ...     register_parametrization,
    ... )
    >>> class Symmetric(nn.Module):
    ...     def forward(self, X):
    ...         return 0.5 * (X + X.mT)
    >>> m = nn.Linear(3, 3)
    >>> _ = register_parametrization(m, "weight", Symmetric())
    >>> is_parametrized(m)
    True
    >>> is_parametrized(m, "bias")
    False
    """
    registry = _parametrizations_of(module)
    if registry is None:
        return False
    if tensor_name is None:
        return len(registry) > 0
    return tensor_name in registry


def remove_parametrizations(
    module: Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> Module:
    r"""Undo :func:`register_parametrization` on ``tensor_name``.

    ``module.<tensor_name>`` becomes a plain parameter (or buffer) again,
    registered under its own name.  With one original, that is the
    original object itself, so an optimizer holding it keeps training it.
    When the last parametrization goes, ``module.parametrizations`` is
    deleted and ``module`` returns to its original class.

    Parameters
    ----------
    module : Module
        Module previously passed to :func:`register_parametrization`.
    tensor_name : str
        The parametrized tensor to restore.
    leave_parametrized : bool, optional
        If ``True`` (the default), the tensor keeps its current
        parametrized value.  If ``False``, it gets the original's values —
        possible only when there is a single original.

    Returns
    -------
    Module
        ``module``.

    Raises
    ------
    ValueError
        If ``tensor_name`` is not parametrized, or ``leave_parametrized``
        is ``False`` for a tensor parametrized by several originals.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrize import (
    ...     is_parametrized,
    ...     register_parametrization,
    ...     remove_parametrizations,
    ... )
    >>> class Symmetric(nn.Module):
    ...     def forward(self, X):
    ...         return 0.5 * (X + X.mT)
    >>> layer = nn.Linear(3, 3)
    >>> _ = register_parametrization(layer, "weight", Symmetric())
    >>> _ = remove_parametrizations(layer, "weight")
    >>> is_parametrized(layer)
    False
    >>> type(layer).__name__
    'Linear'
    """
    registry, chain = _chain_in(module, tensor_name)
    if chain.is_tensor:
        original = chain.original
        if leave_parametrized:
            with lucid.no_grad():
                _set_storage(original, chain())
    elif leave_parametrized:
        value = chain()
        original = (
            Parameter(value.detach(), value.requires_grad)
            if value.requires_grad
            else value.detach()
        )
    else:
        raise ValueError(
            "Cannot leave unparametrized (`leave_parametrized=False`) a tensor "
            "that is parametrized in terms of a sequence of tensors."
        )
    delattr(type(module), tensor_name)
    del registry[tensor_name]
    _register_tensor(module, tensor_name, original)
    if not registry:
        delattr(module, "parametrizations")
        module.__class__ = type(module).__bases__[0]
    return module


def type_before_parametrizations(module: Module) -> type:
    r"""The class ``module`` had before any parametrization was registered.

    A parametrized module's class is a generated ``Parametrized<Class>``
    subclass; this returns ``Class``.  For a module without
    parametrizations it is ``type(module)``.

    Parameters
    ----------
    module : Module
        Module to inspect.

    Returns
    -------
    type
        The module's class before parametrization.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import parametrize
    >>> class Double(nn.Module):
    ...     def forward(self, X):
    ...         return 2 * X
    >>> layer = nn.Linear(2, 2)
    >>> _ = parametrize.register_parametrization(layer, "weight", Double())
    >>> type(layer).__name__, parametrize.type_before_parametrizations(layer).__name__
    ('ParametrizedLinear', 'Linear')
    """
    if is_parametrized(module):
        return type(module).__bases__[0]
    return type(module)
