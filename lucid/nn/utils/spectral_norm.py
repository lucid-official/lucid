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

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid.nn.utils._checks import _require_module, _require_parameter

# Sentinel attribute used by ``remove_spectral_norm`` to find an active
# registration on a module.
_SN_HOOK_ATTR: str = "_spectral_norm_hooks"


def _l2_normalize(v: Tensor, eps: float = 1e-12) -> Tensor:
    """Return ``v / max(‖v‖, eps)`` — kept tiny so it inlines into the graph."""
    norm: Tensor = lucid.sqrt((v * v).sum() + eps)
    return v / norm


def _flatten_weight(w: Tensor, dim: int) -> Tensor:
    """Reshape ``w`` to 2-D with the chosen ``dim`` as rows.

    Spectral norm is defined for matrices, so we collapse all the other
    axes into a single column dimension.  The default ``dim=0`` is correct
    for ``Linear`` / ``Conv*``: rows = output channels, columns = fan-in.
    """
    if dim != 0:
        ndim: int = len(w.shape)
        perm: list[int] = [dim] + [i for i in range(ndim) if i != dim]
        w = w.permute(perm)
    rows: int = int(w.shape[0])
    return w.reshape([rows, -1])


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
        pre-hook performing the normalisation.

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
    weight: Parameter = _require_parameter(module, name, "apply spectral_norm")

    # ConvTranspose* stores weights as (in_channels, out_channels, ...) — the
    # spectral norm should be taken with the output-channel axis (1) on top.
    if dim is None:
        cls_name: str = type(module).__name__
        dim = 1 if cls_name.startswith("ConvTranspose") else 0
    if dim < 0 or dim >= len(weight.shape):
        raise ValueError(f"dim {dim} out of range for weight of shape {tuple(weight.shape)}")

    # Move the original weight to ``{name}_orig`` and seed the iteration
    # vectors.  ``u`` is the row-side iterate (size = rows of the flattened
    # weight); ``v`` is the column-side iterate.
    weight_mat: Tensor = _flatten_weight(weight.detach(), dim)
    rows: int = int(weight_mat.shape[0])
    cols: int = int(weight_mat.shape[1])
    # Power-iteration vectors are contracted with the weight every forward,
    # so they must start on the weight's device.
    dev = weight.device
    u_init: Tensor = _l2_normalize(lucid.randn(rows, device=dev), eps=eps)
    v_init: Tensor = _l2_normalize(lucid.randn(cols, device=dev), eps=eps)

    del module._parameters[name]
    module.register_parameter(name + "_orig", Parameter(weight.detach()))
    module.register_buffer(name + "_u", u_init)
    module.register_buffer(name + "_v", v_init)

    def _pre_hook(mod: Module, _inputs: object) -> None:
        w_orig: Parameter = getattr(mod, name + "_orig")
        u: Tensor = getattr(mod, name + "_u")
        v: Tensor = getattr(mod, name + "_v")
        w_mat: Tensor = _flatten_weight(w_orig, dim)

        # ``matmul`` only handles ≥ 2-D inputs; treat each iterate as a
        # column vector ``(N, 1)`` and squeeze the trailing axis after.
        def _mv(mat: Tensor, vec: Tensor) -> Tensor:
            col: Tensor = vec.reshape([int(vec.shape[0]), 1])
            return lucid.matmul(mat, col).reshape([-1])

        # Power iteration runs in no-grad, into the buffers in place as the
        # reference writes them (LCD-497).  ``sigma`` reads copies, so the
        # next forward's writes cannot reach what this graph saved.
        with lucid.no_grad():
            iters: int = n_power_iterations if mod.training else 0
            for _ in range(iters):
                v.copy_(_l2_normalize(_mv(w_mat.mT, u), eps=eps))
                u.copy_(_l2_normalize(_mv(w_mat, v), eps=eps))
            if iters > 0:
                u, v = u.clone(), v.clone()

        # ``sigma = u^T (W v)`` — keep the autograd graph here so the
        # gradient w.r.t. ``weight_orig`` flows through the normalisation.
        sigma: Tensor = (u * _mv(w_mat, v)).sum()
        normalised: Tensor = w_orig / sigma
        object.__setattr__(mod, name, normalised)

    handle = module.register_forward_pre_hook(_pre_hook)  # type: ignore[arg-type]

    hooks: dict[str, object] = getattr(module, _SN_HOOK_ATTR, {})
    hooks[name] = (handle, dim)
    object.__setattr__(module, _SN_HOOK_ATTR, hooks)

    # Run the hook once so ``module.{name}`` has a sensible initial value
    # without waiting for the first forward call.
    _pre_hook(module, ())
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
    hooks: dict[str, object] = getattr(module, _SN_HOOK_ATTR, {})
    if name not in hooks:
        raise ValueError(f"spectral_norm not registered on '{name}'")
    handle, _dim = hooks.pop(name)  # type: ignore[misc]
    handle.remove()  # type: ignore[has-type]

    w_orig: Parameter = getattr(module, name + "_orig")
    materialised: Tensor = w_orig.detach()
    del module._parameters[name + "_orig"]

    # Drop the cached non-leaf attribute so ``register_parameter`` succeeds.
    if name in module.__dict__:
        del module.__dict__[name]
    module.register_parameter(name, Parameter(materialised))

    # Buffers are stored on the module's ``_buffers`` dict — clear them.
    for buf_suffix in ("_u", "_v"):
        full_name: str = name + buf_suffix
        if hasattr(module, "_buffers") and full_name in module._buffers:
            del module._buffers[full_name]

    if not hooks:
        object.__delattr__(module, _SN_HOOK_ATTR)
    return module
