"""Weight norm, spectral norm and orthogonal — built on ``register_parametrization``.

The reference framework's ``parametrizations`` package re-implements
``weight_norm`` and ``spectral_norm`` (and adds ``orthogonal``) on top of
the modern :func:`~lucid.nn.utils.parametrize.register_parametrization` API
rather than the legacy forward-pre-hook machinery in
:mod:`lucid.nn.utils.weight_norm` / :mod:`lucid.nn.utils.spectral_norm`.
This module mirrors that surface: the trainable leaves land under
``module.parametrizations.<name>.original0`` / ``original1`` (or
``original`` for ``orthogonal``), so a checkpoint saved against the
reference's ``parametrizations.*`` loads here and vice versa — unlike the
legacy ``<name>_g`` / ``<name>_v`` / ``<name>_orig`` layout, which stays
exactly as it was for code that already depends on it.
"""

from typing import override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.linalg import householder_product, matrix_exp, qr, solve
from lucid.nn.functional.normalization import normalize
from lucid.nn.module import Module
from lucid.nn.utils.parametrize import register_parametrization

__all__ = ["orthogonal", "spectral_norm", "weight_norm"]


def _mH(x: Tensor) -> Tensor:
    """Conjugate transpose of the last two axes; plain transpose if real."""
    return x.conj().mT if x.is_complex() else x.mT


def _mv(mat: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector product, without the batched ``matmul``'s own axis."""
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)


def _tensor_attr(module: Module, name: str) -> Tensor:
    """``module.<name>`` as a Tensor — the guard ``spectral_norm`` and
    ``orthogonal`` share, worded as the reference's does."""
    weight = getattr(module, name, None)
    if isinstance(weight, Tensor):
        return weight
    raise ValueError(f"Module '{module}' has no parameter or buffer with name '{name}'")


# ── weight norm ────────────────────────────────────────────────────────────────


def _norm_except_dim(v: Tensor, dim: int) -> Tensor:
    r"""The reference's ``norm_except_dim``: an L2 norm over every axis but one.

    ``dim == -1`` is its special case for "no axis survives" — the norm of
    the whole tensor, a 0-d result.  For a non-negative ``dim`` with no
    other axis to reduce (a 1-D tensor normalised along its only axis), no
    reduction happens at all and the result is ``|v|`` elementwise.
    """
    if dim == -1:
        return lucid.sqrt((v * v).sum())
    ndim = v.ndim
    if dim < 0:
        dim += ndim
    keep_axes = [i for i in range(ndim) if i != dim]
    if not keep_axes:
        return v.abs()
    return lucid.sqrt((v * v).sum(dim=keep_axes, keepdim=True))


class _WeightNorm(Module):
    """``weight = g * v / ||v||`` as a registered parametrization.

    ``forward`` takes the two originals positionally — the chain calls a
    parametrization with as many arguments as ``right_inverse`` returned
    tensors — and ``right_inverse`` splits an existing weight into them.
    """

    dim: int

    def __init__(self, dim: int | None = 0) -> None:
        super().__init__()
        self.dim = -1 if dim is None else dim

    @override
    def forward(self, weight_g: Tensor, weight_v: Tensor) -> Tensor:
        return weight_g * (weight_v / _norm_except_dim(weight_v, self.dim))

    def right_inverse(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        return _norm_except_dim(weight, self.dim), weight


def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    r"""Reparametrise a weight as magnitude times direction, via a parametrization.

    .. math::

        \mathbf{w} \;=\; g \,\cdot\, \frac{\mathbf{v}}{\|\mathbf{v}\|}

    This is :func:`lucid.nn.utils.weight_norm` rebuilt on
    :func:`~lucid.nn.utils.parametrize.register_parametrization`: ``g`` and
    ``v`` land as ``module.parametrizations.<name>.original0`` /
    ``original1`` rather than ``<name>_g`` / ``<name>_v``, and no forward
    pre-hook is installed — ``module.<name>`` is a property that recomputes
    the product on every read.

    Parameters
    ----------
    module : Module
        The module whose parameter is to be reparametrised.  Mutated in
        place.
    name : str, optional
        Attribute name of the parameter to normalise.  Default ``"weight"``.
    dim : int, optional
        Axis along which the scale ``g`` has its own entry; norms are
        reduced over every other axis.  ``0`` (the default) is correct for
        ``Linear`` and ``Conv*`` — one scale per output channel.

    Returns
    -------
    Module
        ``module``, now carrying ``parametrizations.<name>.original0``
        (the magnitude ``g``) and ``original1`` (the direction ``v``).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrizations import weight_norm
    >>> layer = weight_norm(nn.Linear(20, 40))
    >>> layer.parametrizations.weight.original0.shape
    (40, 1)
    >>> layer.parametrizations.weight.original1.shape
    (40, 20)
    """
    register_parametrization(module, name, _WeightNorm(dim), unsafe=True)
    return module


# ── spectral norm ──────────────────────────────────────────────────────────────


class _SpectralNorm(Module):
    """:math:`W / \\sigma(W)` as a registered parametrization.

    ``_u`` / ``_v`` are registered buffers holding the running power-
    iteration estimate of the dominant left/right singular vector; they
    update in training mode on every read and hold still in eval mode.
    """

    dim: int
    eps: float
    n_power_iterations: int
    _u: Tensor
    _v: Tensor

    def __init__(
        self, weight: Tensor, n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but got "
                f"n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = int(weight_mat.shape[0]), int(weight_mat.shape[1])
            u = lucid.randn(h, device=weight.device)
            v = lucid.randn(w, device=weight.device)
            self.register_buffer("_u", normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", normalize(v, dim=0, eps=self.eps))
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        if weight.ndim <= 1:
            raise ValueError(f"Expected weight to have more than 1 dimension, got {weight.ndim}")
        if self.dim != 0:
            ndim = weight.ndim
            weight = weight.permute([self.dim] + [d for d in range(ndim) if d != self.dim])
        return weight.flatten(1)

    def _power_method(self, weight_mat: Tensor, n_power_iterations: int) -> None:
        if weight_mat.ndim <= 1:
            raise ValueError(
                f"Expected weight_mat to have more than 1 dimension, got {weight_mat.ndim}"
            )
        with lucid.no_grad():
            for _ in range(n_power_iterations):
                self._u = normalize(_mv(weight_mat, self._v), dim=0, eps=self.eps)
                self._v = normalize(_mv(_mH(weight_mat), self._u), dim=0, eps=self.eps)

    @override
    def forward(self, weight: Tensor) -> Tensor:
        if weight.ndim == 1:
            return normalize(weight, dim=0, eps=self.eps)
        weight_mat = self._reshape_weight_to_matrix(weight)
        if self.training:
            self._power_method(weight_mat, self.n_power_iterations)
        u, v = self._u.clone(), self._v.clone()
        sigma = (u * _mv(weight_mat, v)).sum()
        return weight / sigma

    def right_inverse(self, value: Tensor) -> Tensor:
        return value


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    r"""Constrain a weight's spectral norm via power iteration, as a parametrization.

    .. math::

        \mathbf{W}_{SN} \;=\; \frac{\mathbf{W}}{\sigma(\mathbf{W})}

    This is :func:`lucid.nn.utils.spectral_norm` rebuilt on
    :func:`~lucid.nn.utils.parametrize.register_parametrization`: the
    trainable matrix lands at ``module.parametrizations.<name>.original``
    (not ``<name>_orig``), and ``_u`` / ``_v`` are buffers of the
    parametrization module itself rather than of ``module``.  Each read in
    training mode runs ``n_power_iterations`` more updates of ``_u`` /
    ``_v``; eval mode leaves them untouched, so repeated reads are
    deterministic.

    Parameters
    ----------
    module : Module
        Holder of the tensor to constrain.  Mutated in place, as
        ``register_parametrization`` always does.
    name : str, optional
        Which attribute of ``module`` to read.  Default ``"weight"``.
    n_power_iterations : int, optional
        How many power-iteration updates a training-mode read performs.
        Default ``1``.
    eps : float, optional
        Numerical floor inside the :math:`\ell_2` normalisation of ``_u``
        and ``_v``.  Default ``1e-12``.
    dim : int, optional
        Axis treated as the row dimension of the flattened matrix.  If
        ``None`` (the default), ``0`` for everything except
        ``ConvTranspose*`` modules, which use ``1``.

    Returns
    -------
    Module
        ``module``, with the normalised matrix at
        ``parametrizations.<name>.original``.

    Raises
    ------
    IndexError
        If ``dim`` is out of range for ``module.<name>``'s rank.
    ValueError
        If ``n_power_iterations`` is not positive.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrizations import spectral_norm
    >>> disc = spectral_norm(nn.Conv2d(3, 64, 3, padding=1))
    """
    weight = _tensor_attr(module, name)
    if dim is None:
        dim = 1 if type(module).__name__.startswith("ConvTranspose") else 0
    register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps))
    return module


# ── orthogonal ─────────────────────────────────────────────────────────────────

_ORTHOGONAL_MAPS = ("matrix_exp", "cayley", "householder")


def _is_orthogonal(Q: Tensor, atol: float) -> bool:
    k = int(Q.shape[-1])
    identity = lucid.eye(k, dtype=Q.dtype, device=Q.device)
    return bool((_mH(Q) @ Q).allclose(identity, atol=atol))


def _make_orthogonal(A: Tensor) -> Tensor:
    r"""Orthogonalise a tall matrix via QR, with a canonical (sign-fixed) ``Q``.

    ``A = Q R``; LAPACK's ``R`` diagonal can carry either sign, so each
    column of ``Q`` is flipped by the sign of its ``R`` diagonal entry to
    fix the ambiguity — the reference does the same correction over the
    packed Householder form from ``geqrf``, which :func:`lucid.linalg.qr`
    is built on, so the two agree up to floating-point rounding.
    """
    Q, R = qr(A, mode="reduced")
    diag_r = R.diagonal(dim1=-2, dim2=-1)
    return Q * diag_r.sign().unsqueeze(-2)


class _Orthogonal(Module):
    """An orthogonal (or, for a complex tensor, unitary) matrix, via one of
    three maps from a skew-symmetric / Householder-reflector generator."""

    shape: tuple[int, ...]
    orthogonal_map: str
    base: Tensor

    def __init__(
        self, weight: Tensor, orthogonal_map: str, *, use_trivialization: bool = True
    ) -> None:
        super().__init__()
        if weight.is_complex() and orthogonal_map == "householder":
            raise ValueError("The householder parametrization does not support complex tensors.")
        self.shape = tuple(int(s) for s in weight.shape)
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)

    @override
    def forward(self, X: Tensor) -> Tensor:
        n, k = int(X.shape[-2]), int(X.shape[-1])
        transposed = n < k
        if transposed:
            X = X.mT
            n, k = k, n
        if self.orthogonal_map in ("matrix_exp", "cayley"):
            X = X.tril()
            if n != k:
                pad = X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)
                X = lucid.cat([X, pad], dim=-1)
            A = X - _mH(X)
            if self.orthogonal_map == "matrix_exp":
                Q = matrix_exp(A)
            else:
                identity = lucid.eye(n, dtype=A.dtype, device=A.device)
                Q = solve(identity - 0.5 * A, identity + 0.5 * A)
            if n != k:
                Q = Q[..., :k]
        else:
            A = X.tril(-1)
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            Q = householder_product(A, tau)
            Q = Q * X.diagonal(dim1=-2, dim2=-1).sign().unsqueeze(-2)
        if hasattr(self, "base"):
            Q = self.base @ Q
        if transposed:
            Q = Q.mT
        return Q

    def right_inverse(self, Q: Tensor) -> Tensor:
        if tuple(int(s) for s in Q.shape) != self.shape:
            raise ValueError(
                f"Expected a matrix or batch of matrices of shape {self.shape}. "
                f"Got a tensor of shape {tuple(Q.shape)}."
            )
        with lucid.no_grad():
            return self._right_inverse(Q)

    def _right_inverse(self, Q_init: Tensor) -> Tensor:
        Q = Q_init
        n, k = int(Q.shape[-2]), int(Q.shape[-1])
        transpose = n < k
        if transpose:
            Q = Q.mT
            n, k = k, n
        if not hasattr(self, "base"):
            if self.orthogonal_map in ("cayley", "matrix_exp"):
                raise NotImplementedError(
                    "It is not possible to assign to the matrix exponential "
                    "or the Cayley parametrizations when use_trivialization=False."
                )
            raise NotImplementedError(
                "orthogonal(..., orthogonal_map='householder', "
                "use_trivialization=False) needs the packed-Householder form "
                "of an arbitrary matrix (LAPACK geqrf); lucid.linalg does not "
                "expose that factorisation yet. Register with "
                "use_trivialization=True (the default) instead."
            )
        atol = 10.0 * n * lucid.finfo(Q.dtype).eps
        if n == k:
            Q = Q.clone() if _is_orthogonal(Q, atol) else _make_orthogonal(Q)
        else:
            tail = lucid.randn(
                *(tuple(int(s) for s in Q.shape[:-2]) + (n, n - k)), dtype=Q.dtype, device=Q.device
            )
            Q = _make_orthogonal(lucid.cat([Q, tail], dim=-1))
        self.base = Q
        # ``-I`` of ``Q_init``'s own (pre-swap) shape: the diagonal doubles as
        # the sign the Householder branch reads off the originals' diagonal.
        neg_identity = -lucid.eye(
            int(Q_init.shape[-2]), int(Q_init.shape[-1]), dtype=Q_init.dtype, device=Q_init.device
        )
        if Q_init.ndim > 2:
            neg_identity = lucid.broadcast_to(neg_identity, Q_init.shape).contiguous()
        return neg_identity


def orthogonal(
    module: Module,
    name: str = "weight",
    orthogonal_map: str | None = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    r"""Parametrise a matrix (or batch of matrices) to stay orthogonal.

    :math:`Q \in \mathbb{K}^{m \times n}` (:math:`\mathbb{K} = \mathbb{R}`
    or :math:`\mathbb{C}`) is orthogonal/unitary as :math:`Q^{H}Q = I_n`
    when :math:`m \ge n`, or :math:`QQ^{H} = I_m` otherwise.  Three maps
    from an unconstrained generator give such a :math:`Q`:

    * ``"matrix_exp"`` — :math:`Q = \exp(A)` for skew-symmetric :math:`A`.
    * ``"cayley"`` — the Cayley retraction
      :math:`Q = (I + A/2)(I - A/2)^{-1}`.
    * ``"householder"`` — a product of Householder reflectors.

    With ``use_trivialization=True`` (the default), an extra matrix
    :math:`B` is stored under
    ``module.parametrizations.<name>[0].base`` and the parametrized value
    is :math:`BQ`, which converges faster at the cost of that extra buffer.

    Parameters
    ----------
    module : Module
        The module whose tensor is to be made orthogonal.  Mutated in
        place.
    name : str, optional
        Attribute name of the tensor.  Default ``"weight"``.
    orthogonal_map : {"matrix_exp", "cayley", "householder"}, optional
        The map to use.  ``None`` (the default) picks ``"matrix_exp"`` for
        a square or complex tensor and ``"householder"`` otherwise.
    use_trivialization : bool, keyword-only, optional
        Whether to use the dynamic-trivialization framework described
        above.  Default ``True``.

    Returns
    -------
    Module
        ``module``, with the orthogonal matrix at
        ``parametrizations.<name>.original``.

    Raises
    ------
    ValueError
        If ``module.<name>`` is not a tensor of rank at least 2, or
        ``orthogonal_map`` is not one of the three named maps.
    NotImplementedError
        For ``orthogonal_map="householder"`` with
        ``use_trivialization=False``: that path needs an arbitrary
        matrix's packed Householder-reflector form (LAPACK ``geqrf``),
        which :mod:`lucid.linalg` does not expose.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.parametrizations import orthogonal
    >>> layer = orthogonal(nn.Linear(20, 40))
    >>> Q = layer.weight
    >>> bool(lucid.allclose(Q.mT @ Q, lucid.eye(20), atol=1e-4))
    True
    """
    weight = _tensor_attr(module, name)
    if weight.ndim < 2:
        raise ValueError(
            f"Expected a matrix or batch of matrices. Got a tensor of {weight.ndim} dimensions."
        )
    if orthogonal_map is None:
        orthogonal_map = (
            "matrix_exp"
            if int(weight.shape[-2]) == int(weight.shape[-1]) or weight.is_complex()
            else "householder"
        )
    if orthogonal_map not in _ORTHOGONAL_MAPS:
        raise ValueError(
            'orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
            f"Got: {orthogonal_map}"
        )
    orth = _Orthogonal(weight, orthogonal_map, use_trivialization=use_trivialization)
    register_parametrization(module, name, orth, unsafe=True)
    return module
