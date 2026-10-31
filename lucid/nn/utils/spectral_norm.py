"""Spectral Normalization (Miyato et al., 2018).

Reparametrises a module's ``weight`` parameter as

    weight = W / σ(W)

where ``σ(W)`` is the largest singular value of ``W``, estimated cheaply by
running one or more power-iteration steps on a pair of buffers ``u`` and
``v``.  Constraining the spectral norm of the linear/conv weight bounds
the layer's Lipschitz constant, which stabilises GAN discriminators and
some Transformer variants.

Reproduction follows the reference framework's legacy ``spectral_norm``
API (the parametrize-based variant is a separate piece of work).  Each
forward call:

  1. Reads the current ``weight_orig`` parameter.
  2. Runs ``n_power_iterations`` updates of ``u`` and ``v``:
         v ← normalize(W^T u)
         u ← normalize(W   v)
  3. Computes ``sigma = u^T W v`` and writes ``weight = W / sigma``.

The buffers ``u`` / ``v`` are recomputed in training mode and held fixed
in eval mode — bit-identical to the reference framework's behaviour.
"""

from typing import Self, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid.nn.utils._checks import _require_module, _require_parameter
from lucid.nn.utils._reparam import _NamedHook

__all__ = ["SpectralNorm", "spectral_norm", "remove_spectral_norm"]


def _l2_normalize(v: Tensor, eps: float = 1e-12) -> Tensor:
    """Return ``v / max(‖v‖, eps)`` — kept tiny so it inlines into the graph."""
    norm: Tensor = lucid.sqrt((v * v).sum() + eps)
    return v / norm


def _mv(mat: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector product; ``matmul`` wants ≥ 2-D, so the vector rides as a column."""
    col: Tensor = vec.reshape([int(vec.shape[0]), 1])
    return lucid.matmul(mat, col).reshape([-1])


class SpectralNorm(_NamedHook):
    r"""The forward pre-hook behind :func:`spectral_norm`.

    Holds the parameter name, the row axis of the flattened matrix and the
    power-iteration settings; before every forward it refreshes the ``u`` /
    ``v`` buffers (training mode) and sets ``module.<name>`` to
    :math:`\mathbf{W}_{\text{orig}} / \sigma`.  A plain instance, so a module
    carrying one pickles and deep-copies.

    Parameters
    ----------
    name : str, optional
        Name of the reparametrised parameter.  Default ``"weight"``.
    n_power_iterations : int, optional
        Power-iteration steps per training-mode forward; positive.  Default ``1``.
    dim : int, optional
        Axis that becomes the rows of the flattened matrix.  Default ``0``.
    eps : float, optional
        Floor inside the :math:`\ell_2` normalisation.  Default ``1e-12``.
    """

    _LABEL = "spectral_norm"
    _version: int = 1

    def __init__(
        self, name: str = "weight", n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12
    ) -> None:
        if n_power_iterations <= 0:
            raise ValueError(
                f"Expected n_power_iterations to be positive, but got n_power_iterations="
                f"{n_power_iterations}"
            )
        super().__init__(name)
        self.dim: int = dim
        self.n_power_iterations: int = n_power_iterations
        self.eps: float = eps

    def reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        """``weight`` as a 2-D matrix with axis :attr:`dim` as its rows.

        Spectral norm is defined for matrices, so every other axis collapses
        into one column axis.  ``dim=0`` is right for ``Linear`` / ``Conv*``
        (rows = output channels, columns = fan-in).
        """
        if self.dim != 0:
            perm: list[int] = [self.dim] + [i for i in range(len(weight.shape)) if i != self.dim]
            weight = weight.permute(perm)
        return weight.reshape([int(weight.shape[0]), -1])

    def compute_weight(self, module: Module, do_power_iteration: bool) -> Tensor:
        """``W_orig / sigma``, refreshing ``u`` / ``v`` first when asked to."""
        w_orig: Tensor = getattr(module, self.name + "_orig")
        u: Tensor = getattr(module, self.name + "_u")
        v: Tensor = getattr(module, self.name + "_v")
        w_mat: Tensor = self.reshape_weight_to_matrix(w_orig)
        if do_power_iteration:
            # No-grad, and into the buffers in place as the reference writes them (LCD-497).
            with lucid.no_grad():
                for _ in range(self.n_power_iterations):
                    v.copy_(_l2_normalize(_mv(w_mat.mT, u), eps=self.eps))
                    u.copy_(_l2_normalize(_mv(w_mat, v), eps=self.eps))
        # ``sigma`` reads copies in every mode: the graph saves its operands, and a
        # later iteration writing the live buffers in place would reach into it
        # (eval then train used to raise on the write).
        u, v = u.clone(), v.clone()
        sigma: Tensor = (u * _mv(w_mat, v)).sum()
        return w_orig / sigma

    @classmethod
    def apply(
        cls, module: Module, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> Self:
        r"""Reparametrise ``module.<name>`` and install the hook; the constructor of record.

        The original Parameter stays, as the same object, under ``<name>_orig``
        — an optimizer built before this call keeps updating it.

        Parameters
        ----------
        module : Module
            Host of the parameter.
        name : str
            Name of the parameter to normalise.
        n_power_iterations : int
            Power-iteration steps per training-mode forward.
        dim : int
            Axis that becomes the rows of the flattened matrix.
        eps : float
            Floor inside the :math:`\ell_2` normalisation of ``u`` and ``v``.

        Returns
        -------
        SpectralNorm
            The hook now installed on ``module``.

        Raises
        ------
        RuntimeError
            If ``name`` is already spectral-normalised.
        """
        cls._require_free(module, name)
        fn = cls(name, n_power_iterations, dim, eps)
        weight: Parameter = _require_parameter(module, name, "apply spectral_norm")
        if not 0 <= dim < len(weight.shape):
            raise ValueError(f"dim {dim} out of range for weight of shape {tuple(weight.shape)}")
        with lucid.no_grad():
            rows, cols = (int(n) for n in fn.reshape_weight_to_matrix(weight).shape)
            # Both iterates are contracted with the weight every forward: same device and dtype.
            u: Tensor = _l2_normalize(
                lucid.randn(rows, dtype=weight.dtype, device=weight.device), eps=eps
            )
            v: Tensor = _l2_normalize(
                lucid.randn(cols, dtype=weight.dtype, device=weight.device), eps=eps
            )
        del module._parameters[name]
        module.register_parameter(name + "_orig", weight)
        module.register_buffer(name + "_u", u)
        module.register_buffer(name + "_v", v)
        # Readable before the first forward, normalised by the seeded vectors;
        # the first forward is the first power iteration.
        fn._publish(module, fn.compute_weight(module, do_power_iteration=False))
        cls._register(module, fn)
        return fn

    @override
    def refresh(self, module: Module) -> None:
        """Recompute ``module.<name>``; the power iteration runs only in training mode."""
        self._publish(module, self.compute_weight(module, do_power_iteration=module.training))

    @override
    def remove(self, module: Module) -> None:
        """Hand the unnormalised ``W_orig`` back as ``<name>`` and drop ``_u`` / ``_v``."""
        orig: Parameter = _require_parameter(module, self.name + "_orig", "remove spectral_norm")
        self._vacate(module)
        del module._parameters[self.name + "_orig"]
        del module._buffers[self.name + "_u"]
        del module._buffers[self.name + "_v"]
        module.register_parameter(self.name, orig)


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    r"""Constrain a weight's spectral norm via power iteration (Miyato et al. 2018).

    Reparametrises ``module.<name>`` as

    .. math::

        \mathbf{W} \;=\; \frac{\mathbf{W}_{\text{orig}}}{\sigma(\mathbf{W}_{\text{orig}})},

    where :math:`\sigma(\mathbf{W})` is the largest singular value
    (operator 2-norm) of the weight viewed as a matrix.  This bounds the
    layer's Lipschitz constant by ``1`` and is the canonical
    stabilisation trick for GAN discriminators; it also sees use in
    certified-robust classifiers and contractive Transformer variants.

    Parameters
    ----------
    module : Module
        The module whose parameter is to be normalised.  Mutated in
        place.
    name : str, optional
        Attribute name of the parameter.  Default ``"weight"``.
    n_power_iterations : int, optional
        Number of power-iteration updates of the ``u`` / ``v`` vectors
        performed before each training-mode forward.  Defaults to ``1``
        — sufficient because the state is preserved between calls and
        converges over the course of training.
    eps : float, optional
        Numerical floor added inside the :math:`\ell_2` normalisation of
        ``u`` and ``v``.  Default ``1e-12``.
    dim : int, optional
        Axis to treat as the row dimension of the flattened matrix.  If
        ``None`` (the default), uses ``0`` for everything except
        ``ConvTranspose*`` modules, which use ``1`` (so output channels
        end up as rows).

    Returns
    -------
    Module
        The same module, with ``<name>_orig`` (the trainable underlying
        matrix), ``<name>_u`` / ``<name>_v`` (buffers), and a forward
        pre-hook performing the normalisation.  ``<name>_orig`` is the
        original :class:`~lucid.nn.parameter.Parameter` object itself, so an
        optimizer built before this call keeps updating it.

    Raises
    ------
    RuntimeError
        If ``<name>`` is already spectral-normalised.
    ValueError
        If ``n_power_iterations`` is not positive or ``dim`` is out of range.

    Notes
    -----
    Each forward in training mode runs ``n_power_iterations`` updates

    .. math::

        \mathbf{v} \leftarrow \frac{\mathbf{W}^{\!\top}\mathbf{u}}{\|\mathbf{W}^{\!\top}\mathbf{u}\|}, \qquad
        \mathbf{u} \leftarrow \frac{\mathbf{W}\,\mathbf{v}}{\|\mathbf{W}\,\mathbf{v}\|},

    estimates :math:`\sigma \approx \mathbf{u}^{\!\top}\mathbf{W}\mathbf{v}`,
    and exposes :math:`\mathbf{W}_{\text{orig}} / \sigma` as the active
    weight.  In eval mode no updates occur — the cached buffers are used
    as-is for deterministic inference.  Invert with
    :func:`remove_spectral_norm`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import spectral_norm
    >>> disc = spectral_norm(nn.Conv2d(3, 64, 3, padding=1))
    """
    _require_module(module, "spectral_norm")
    if dim is None:
        # ConvTranspose* stores weights as (in_channels, out_channels, ...): its
        # output-channel axis, the matrix rows, is 1.
        dim = 1 if type(module).__name__.startswith("ConvTranspose") else 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module: Module, name: str = "weight") -> Module:
    r"""Reverse :func:`spectral_norm` and restore a plain leaf parameter.

    Copies ``module.<name>_orig`` back into a fresh leaf parameter
    ``module.<name>``, removes the ``<name>_u`` / ``<name>_v`` power-
    iteration buffers, and detaches the forward pre-hook.  Used before
    exporting for inference when the normalisation overhead is no
    longer wanted, or before fine-tuning without the spectral cap.

    Parameters
    ----------
    module : Module
        Module previously passed through :func:`spectral_norm`.
    name : str, optional
        Attribute name of the parameter to restore.  Default ``"weight"``.

    Returns
    -------
    Module
        The same module, with ``<name>`` as a plain
        :class:`~lucid.nn.parameter.Parameter` carrying the *unnormalised*
        weight that was being trained behind the parametrisation.

    Raises
    ------
    ValueError
        If no spectral-norm registration exists on ``<name>``.

    Notes
    -----
    The restored parameter is :math:`\mathbf{W}_{\text{orig}}` — *not*
    the rescaled :math:`\mathbf{W}_{\text{orig}} / \sigma` that was
    visible during training.  If you want the normalised matrix in the
    final checkpoint, copy it out *before* calling this function.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import spectral_norm, remove_spectral_norm
    >>> disc = spectral_norm(nn.Linear(16, 1))
    >>> _ = remove_spectral_norm(disc)
    >>> sorted(name for name, _ in disc.named_parameters())
    ['bias', 'weight']
    """
    return SpectralNorm.uninstall(module, name)
