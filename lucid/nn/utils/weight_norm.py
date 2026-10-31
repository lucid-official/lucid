"""Weight Normalization (Salimans & Kingma, 2016).

Replaces a module's ``weight`` parameter with the reparametrisation

    weight = g * v / ||v||_dim

where ``g`` is a learnable scale (one entry per slice along ``dim``) and ``v``
is a learnable direction with the same shape as the original weight. The
reparametrised weight is recomputed before each forward call via a pre-hook.

Mirrors ``reference framework.nn.utils.weight_norm`` (the legacy non-parametrize variant).
"""

__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]

from typing import override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid.nn.utils._checks import _require_module, _require_parameter
from lucid.nn.utils._reparam import _NamedHook


def _norm_along_dim(v: Tensor, dim: int) -> Tensor:
    """L2 norm of ``v`` over every axis except ``dim``, kept broadcastable.

    ``dim == -1`` is the reference's "no axis": the norm of the whole tensor, a
    0-d scalar (so ``weight_norm(m, dim=-1)`` and ``dim=None`` share one scalar
    magnitude).  Any other negative ``dim`` counts from the end.
    """
    ndim: int = len(v.shape)
    if dim == -1:
        return lucid.sqrt((v * v).sum())
    if not -ndim <= dim < ndim:
        raise IndexError(f"dim {dim} is out of range for a weight of shape {tuple(v.shape)}")
    dim %= ndim
    keep_axes: list[int] = [i for i in range(ndim) if i != dim]
    if not keep_axes:
        # A 1-D weight normalised along its only axis: every slice is one element.
        return lucid.abs(v)
    return lucid.sqrt((v * v).sum(dim=keep_axes, keepdim=True))


def _compute_weight(g: Tensor, v: Tensor, dim: int) -> Tensor:
    return v * (g / _norm_along_dim(v, dim))


class WeightNorm(_NamedHook):
    r"""The forward pre-hook behind :func:`weight_norm`.

    Holds which parameter is reparametrised and along which axis; before every
    forward it recomputes :math:`\mathbf{W} = g\,\mathbf{v}/\lVert\mathbf{v}\rVert`
    from the module's ``<name>_g`` and ``<name>_v``.  A plain instance, so a
    module carrying one pickles and deep-copies.

    Parameters
    ----------
    name : str
        Name of the reparametrised parameter.
    dim : int or None
        Axis keeping its own magnitude; ``None`` or ``-1`` normalises the whole
        tensor to one scalar magnitude.
    """

    _LABEL = "weight_norm"

    def __init__(self, name: str, dim: int | None) -> None:
        super().__init__(name)
        self.dim: int = -1 if dim is None else dim

    def compute_weight(self, module: Module) -> Tensor:
        """The weight ``g * v / ||v||`` from the module's current ``g`` and ``v``."""
        g: Tensor = getattr(module, self.name + "_g")
        v: Tensor = getattr(module, self.name + "_v")
        return _compute_weight(g, v, self.dim)

    @classmethod
    def apply(cls, module: Module, name: str, dim: int | None) -> WeightNorm:
        """Reparametrise ``module.<name>`` and install the hook; the constructor of record.

        Parameters
        ----------
        module : Module
            Host of the parameter.
        name : str
            Name of the parameter to normalise.
        dim : int or None
            Axis keeping its own magnitude; ``None`` / ``-1`` for one scalar.

        Returns
        -------
        WeightNorm
            The hook now installed on ``module``.

        Raises
        ------
        RuntimeError
            If ``name`` is already weight-normalised.
        """
        cls._require_free(module, name)
        weight: Parameter = _require_parameter(module, name, "apply weight_norm")
        fn = cls(name, dim)
        # The magnitude is the first thing computed, so a ``dim`` out of range is
        # refused before the module is touched.
        g: Parameter = Parameter(_norm_along_dim(weight.detach(), fn.dim))
        v: Parameter = Parameter(weight.detach())
        del module._parameters[name]
        module.register_parameter(name + "_g", g)
        module.register_parameter(name + "_v", v)
        fn._publish(module, fn.compute_weight(module))
        cls._register(module, fn)
        return fn

    @override
    def refresh(self, module: Module) -> None:
        """Recompute ``module.<name>`` from the current ``g`` and ``v``."""
        self._publish(module, self.compute_weight(module))

    @override
    def remove(self, module: Module) -> None:
        """Materialise the weight one last time as a plain leaf Parameter."""
        weight: Tensor = self.compute_weight(module).detach()
        self._vacate(module)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        module.register_parameter(self.name, Parameter(weight))


def weight_norm(module: Module, name: str = "weight", dim: int | None = 0) -> Module:
    r"""Reparametrise a weight as direction times magnitude (Salimans & Kingma 2016).

    Replaces ``module.<name>`` with the derived tensor

    .. math::

        \mathbf{W} \;=\; g \,\cdot\, \frac{\mathbf{v}}{\|\mathbf{v}\|}

    where ``g`` is a learnable scale (one entry per slice along ``dim``)
    and ``v`` is a learnable direction sharing the original weight's
    shape.  Decoupling magnitude from direction often accelerates
    convergence and improves conditioning, especially in deep
    convolutional and recurrent networks.

    Parameters
    ----------
    module : Module
        The module whose parameter is to be reparametrised.  Mutated in
        place — the original leaf parameter is removed and replaced with
        a derived tensor recomputed before every forward call.
    name : str, optional
        Attribute name of the parameter to normalise.  Default ``"weight"``.
    dim : int or None, optional
        Axis along which the scale ``g`` has its own entry; norms are
        reduced over every other axis.  ``0`` (the default) is correct
        for ``Linear`` and ``Conv*`` (one scale per output channel).
        ``None`` or ``-1`` normalises the whole tensor against one scalar
        ``g`` (the reference's convention); any other negative ``dim``
        counts from the end.

    Returns
    -------
    Module
        The same module, now carrying ``<name>_g`` and ``<name>_v`` as
        leaf parameters in place of the original ``<name>``.

    Notes
    -----
    Two new parameters are registered:

    * ``<name>_g`` — broadcast-shaped tensor (``1`` along every axis
      except ``dim``; a 0-d scalar for ``dim=None`` / ``-1``) holding the
      per-slice magnitudes.
    * ``<name>_v`` — same shape as the original weight, holding the
      unnormalised direction.

    A forward pre-hook recomputes :math:`\mathbf{W} = g \cdot \mathbf{v}/\|\mathbf{v}\|`
    so subsequent autograd / inference sees the same shape as before.
    Invert with :func:`remove_weight_norm`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import weight_norm
    >>> layer = weight_norm(nn.Linear(128, 64))
    >>> layer.weight_g.shape, layer.weight_v.shape
    ((64, 1), (64, 128))
    """
    _require_module(module, "weight_norm")
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: Module, name: str = "weight") -> Module:
    r"""Reverse :func:`weight_norm` and restore a plain leaf parameter.

    Materialises :math:`\mathbf{W} = g \cdot \mathbf{v}/\|\mathbf{v}\|`
    one last time, writes the result back as a single leaf parameter
    ``module.<name>``, drops ``<name>_g`` / ``<name>_v``, and detaches
    the forward pre-hook.  Typical use is right before exporting a model
    for inference where the reparametrised form is not needed.

    Parameters
    ----------
    module : Module
        Module previously passed through :func:`weight_norm`.
    name : str, optional
        Attribute name of the parameter to un-normalise.  Must match the
        ``name`` used at registration.  Default ``"weight"``.

    Returns
    -------
    Module
        The same module, with ``<name>`` restored as a plain
        :class:`~lucid.nn.parameter.Parameter` and the ``g``/``v``
        helpers removed.

    Raises
    ------
    ValueError
        If no weight-norm registration exists on ``<name>``.

    Notes
    -----
    The materialised parameter is detached from the autograd graph — any
    history accumulated through the reparametrisation is discarded.  If
    you need the original ``v`` direction back, copy it out *before*
    calling this.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import weight_norm, remove_weight_norm
    >>> layer = weight_norm(nn.Linear(128, 64))
    >>> _ = remove_weight_norm(layer)
    >>> "weight_g" in dict(layer.named_parameters())
    False
    """
    return WeightNorm.uninstall(module, name)
