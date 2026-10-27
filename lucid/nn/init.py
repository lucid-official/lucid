"""
nn.init: parameter initialization functions.
All functions operate in-place and return the tensor.
"""

import functools
import math
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import lucid as _lucid
from lucid import _promotion
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_P = ParamSpec("_P")
_R = TypeVar("_R")

# While positive, every initialiser below returns its tensor untouched.
# Per thread, so a skip_init on one thread never skips another's.
_skip = threading.local()


@contextmanager
def _skipping() -> Iterator[None]:
    """Make the initialisers no-ops for the duration — what ``skip_init`` needs.

    Construction then costs the allocation and nothing else: no random
    draw, no fill.  ``skip_init`` used to construct normally and swap in
    empty storage afterwards, which paid for every initialiser and then
    for a second allocation.
    """
    _skip.depth = getattr(_skip, "depth", 0) + 1
    try:
        yield
    finally:
        _skip.depth -= 1


def _skippable(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def initialise(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if getattr(_skip, "depth", 0):
            return args[0] if args else kwargs["tensor"]  # type: ignore[return-value]
        return fn(*args, **kwargs)

    return initialise


def _fill_from_impl(tensor: Tensor, src_impl: _C_engine.TensorImpl) -> Tensor:
    """Write ``src_impl`` into ``tensor``, in place.

    It used to *replace* ``tensor._impl`` with a fresh one carrying the
    same ``requires_grad``.  That silently detached the tensor from any
    graph it was in: initialising something mid-computation left every
    node that had saved it pointing at storage the tensor no longer used,
    and the backward pass went on to give an answer built from values
    that had been overwritten.  Nothing raised, because the version
    counter never saw a write — there was none.

    Written through instead, so the counter fires and a later backward
    says so.  Under ``no_grad``, because an initialiser is not a step in a
    computation and the target is nearly always a leaf that requires
    grad, which an in-place write is otherwise refused on.  That is the
    same shape the reference gives its own ``nn.init``.
    """
    if not src_impl.is_contiguous():
        src_impl = _C_engine.contiguous(src_impl)
    reshaped = _C_engine.reshape(src_impl, list(tensor.shape))
    # The draw does not always land on the target's dtype —
    # ``orthogonal_`` builds its factor at float32.  Replacing the impl
    # hid that by adopting the source's dtype wholesale, so initialising a
    # float64 parameter quietly turned it into a float32 one.  Writing
    # through cannot: the parameter keeps the dtype it was declared with.
    if reshaped.dtype != tensor._impl.dtype:
        reshaped = _C_engine.astype(reshaped, tensor._impl.dtype)
    with _lucid.no_grad():
        if (
            tensor._impl.is_aliased()
            or not tensor.is_contiguous()
            or tensor._impl.storage_offset_bytes() != 0
        ):
            _C_engine.assign_inplace(tensor._impl, reshaped, "nn.init")
        else:
            tensor._impl.assign_from(reshaped, "nn.init")
    return tensor


@_skippable
def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with samples from a uniform distribution.

    Each entry is drawn independently from :math:`\mathcal{U}(a, b)`.  The
    function mutates ``tensor`` in place and returns the same object so
    calls can be chained or used as a one-liner inside a model
    constructor.

    Parameters
    ----------
    tensor : Tensor
        Tensor to fill in place; any shape is accepted.
    a : float, optional
        Lower bound of the uniform interval.  Default ``0.0``.
    b : float, optional
        Upper bound of the uniform interval.  Default ``1.0``.
    generator : Generator, optional
        PRNG for uniform draws.

    Returns
    -------
    Tensor
        ``tensor`` after uniform fill.

    Notes
    -----
    The resulting distribution has mean and variance

    .. math::

        \mathbb{E}[W] = \frac{a + b}{2}, \qquad
        \mathrm{Var}(W) = \frac{(b - a)^2}{12}.

    Higher-level schemes such as :func:`xavier_uniform_` and
    :func:`kaiming_uniform_` are thin wrappers over this function with
    bounds computed from the fan factors of ``tensor``.

    The draw is :meth:`Tensor.uniform_`'s under :func:`lucid.no_grad`, as
    the reference's is, so the two cannot disagree: the same seed gives the
    same values, ``a == b`` moves the stream on as any other bounds do, and
    a complex ``tensor`` draws both parts from the interval.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import uniform_
    >>> w = lucid.empty(4, 4)
    >>> _ = uniform_(w, -0.1, 0.1)
    >>> bool(((w >= -0.1) & (w <= 0.1)).all().item())
    True
    """
    with _lucid.no_grad():
        return tensor.uniform_(a, b, generator=generator)


@_skippable
def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with samples from a Gaussian distribution.

    Each entry is drawn independently from
    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.  This is the simplest
    initialisation scheme and serves as the building block for
    :func:`xavier_normal_` / :func:`kaiming_normal_`, which choose ``std``
    so that the activation variance is preserved layer-by-layer.

    Parameters
    ----------
    tensor : Tensor
        Tensor to fill in place; any shape is accepted.
    mean : float, optional
        Mean of the Gaussian distribution.  Default ``0.0``.
    std : float, optional
        Standard deviation of the Gaussian distribution.  Must be
        non-negative.  Default ``1.0``.
    generator : Generator, optional
        PRNG for Gaussian draws.

    Returns
    -------
    Tensor
        ``tensor`` after Gaussian fill.

    Notes
    -----
    The samples have

    .. math::

        \mathbb{E}[W] = \text{mean}, \qquad
        \mathrm{Var}(W) = \text{std}^2.

    For unbounded tails consider :func:`trunc_normal_` instead, which
    truncates extreme draws and is often preferred in transformer
    pre-training recipes.

    The draw is :meth:`Tensor.normal_`'s under :func:`lucid.no_grad`, as the
    reference's is: the same seed gives the same values, and a complex
    ``tensor`` takes ``mean`` on both parts (it used to reach the real part
    alone).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import normal_
    >>> w = lucid.empty(64, 32)
    >>> lucid.manual_seed(0)
    >>> _ = normal_(w, mean=0.0, std=0.02)
    >>> abs(w.std().item() - 0.02) < 0.002       # sample std close to the target
    True
    """
    with _lucid.no_grad():
        return tensor.normal_(mean, std, generator=generator)


@_skippable
def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill ``tensor`` in-place with a single scalar value.

    Every entry of ``tensor`` is overwritten with ``val``.  Most often
    used to initialise bias terms to zero or to a small positive value
    (e.g. ``0.01`` to keep ReLU units active at the start of training).

    Parameters
    ----------
    tensor : Tensor
        Tensor to fill in place; any shape is accepted.
    val : float
        Scalar to broadcast into every entry.

    Returns
    -------
    Tensor
        ``tensor`` (mutated) for chaining.

    Notes
    -----
    Constant-filling a weight matrix breaks the network's symmetry-breaking
    property — every output unit would compute the same function and
    learn identical gradients.  Use this for biases only.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import constant_
    >>> b = lucid.empty(128)
    >>> _ = constant_(b, 0.0)
    >>> bool((b == 0.0).all().item())
    True
    """
    return _fill_from_impl(
        tensor,
        _C_engine.full(list(tensor.shape), val, tensor._impl.dtype, tensor._impl.device),
    )


@_skippable
def ones_(tensor: Tensor) -> Tensor:
    r"""Fill ``tensor`` in-place with ones.

    Convenience wrapper around :func:`constant_` with ``val=1.0``.  The
    canonical use case is the gain (gamma) parameter of normalisation
    layers (BatchNorm, LayerNorm, GroupNorm) so the layer initially acts
    as the identity scaling.

    Parameters
    ----------
    tensor : Tensor
        Tensor to fill in place; any shape is accepted.

    Returns
    -------
    Tensor
        ``tensor`` (mutated) for chaining.

    Notes
    -----
    Equivalent to :func:`constant_` ``(tensor, 1.0)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import ones_
    >>> gamma = lucid.empty(64)
    >>> _ = ones_(gamma)
    >>> float(gamma.sum().item())
    64.0
    """
    return _fill_from_impl(
        tensor,
        _C_engine.ones(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


@_skippable
def zeros_(tensor: Tensor) -> Tensor:
    r"""Fill ``tensor`` in-place with zeros.

    Convenience wrapper around :func:`constant_` with ``val=0.0``.  The
    canonical use case is bias initialisation in linear and convolution
    layers, and the shift (beta) parameter of normalisation layers.

    Parameters
    ----------
    tensor : Tensor
        Tensor to fill in place; any shape is accepted.

    Returns
    -------
    Tensor
        ``tensor`` (mutated) for chaining.

    Notes
    -----
    Equivalent to :func:`constant_` ``(tensor, 0.0)``.  Note that
    zero-filling a *weight* matrix would prevent symmetry breaking and
    block learning — use only for biases or normalisation shifts.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import zeros_
    >>> bias = lucid.empty(128)
    >>> _ = zeros_(bias)
    >>> bool((bias == 0.0).all().item())
    True
    """
    return _fill_from_impl(
        tensor,
        _C_engine.zeros(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


@_skippable
def eye_(tensor: Tensor) -> Tensor:
    r"""Fill a 2-D ``tensor`` in-place with the identity matrix.

    Writes ones on the main diagonal and zeros elsewhere.  Identity
    initialisation is useful for square linear layers in residual blocks
    or for testing — at initialisation the layer is the identity map and
    gradients propagate untransformed.

    Parameters
    ----------
    tensor : Tensor
        Two-dimensional tensor to fill in place.  Need not be square; the
        identity is written into the leading ``min(n, m)`` diagonal
        entries.

    Returns
    -------
    Tensor
        ``tensor`` (mutated) for chaining.

    Raises
    ------
    ValueError
        If ``tensor.ndim != 2``.

    Notes
    -----
    For convolution weights that should act as the identity see
    :func:`dirac_`, which is the spatial analogue.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import eye_
    >>> w = lucid.empty(4, 4)
    >>> eye_(w)
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    """
    if tensor.ndim != 2:
        raise ValueError("eye_() requires a 2D tensor")
    n, m = tensor.shape
    return _fill_from_impl(tensor, _C_engine.eye(n, m, 0, tensor._impl.dtype, tensor._impl.device))


@_skippable
def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with Xavier (Glorot) uniform initialisation.

    Fills the tensor with values drawn uniformly from :math:`[-a, a]`
    where the limit ``a`` is chosen so that the variance of activations
    is preserved across a stack of linear / mildly-nonlinear layers.
    Introduced in Glorot & Bengio (2010), this scheme is well-suited to
    ``tanh`` and ``sigmoid`` networks; for ReLU networks prefer
    :func:`kaiming_uniform_`, which corrects for the half-truncation of
    negative pre-activations.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; must have at least 2 dimensions
        so ``fan_in`` and ``fan_out`` can be computed.
    gain : float, optional
        Multiplicative gain factor — typically the value returned by
        :func:`calculate_gain` for the downstream nonlinearity.
        Default ``1.0``.
    generator : Generator, optional
        PRNG used for the Xavier uniform draw.  Default ``None``.

    Returns
    -------
    Tensor
        The Xavier-uniform-initialised ``tensor``.

    Notes
    -----
    Let :math:`n_\text{in}` and :math:`n_\text{out}` be the fan-in and
    fan-out of ``tensor`` (see :func:`_calculate_fan_in_and_fan_out`).
    The uniform range is

    .. math::

        a = \text{gain} \cdot \sqrt{\frac{6}{n_\text{in} + n_\text{out}}},

    which yields variance

    .. math::

        \mathrm{Var}(W) = \frac{2 \cdot \text{gain}^2}{n_\text{in} + n_\text{out}}.

    This is the value that approximately preserves the variance of
    activations forwards and gradients backwards in a linear layer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import xavier_uniform_, calculate_gain
    >>> w = lucid.empty(64, 32)
    >>> _ = xavier_uniform_(w, gain=calculate_gain('tanh'))
    >>> bound = calculate_gain('tanh') * (6 / (32 + 64)) ** 0.5
    >>> bool((w.abs() <= bound).all().item())
    True
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a, generator=generator)


@_skippable
def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with Xavier (Glorot) normal initialisation.

    Gaussian counterpart of :func:`xavier_uniform_`.  Draws each entry
    from :math:`\mathcal{N}(0, \sigma^2)` with a standard deviation
    chosen to preserve activation variance through a stack of linear /
    mildly-nonlinear layers, as proposed in Glorot & Bengio (2010).
    Prefer this for ``tanh`` / ``sigmoid`` networks; use
    :func:`kaiming_normal_` for ReLU networks.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; must have at least 2 dimensions.
    gain : float, optional
        Multiplicative gain factor — see :func:`calculate_gain`.
        Default ``1.0``.
    generator : Generator, optional
        PRNG used for the Xavier Gaussian draw.  Default ``None``.

    Returns
    -------
    Tensor
        The Xavier-normal-initialised ``tensor``.

    Notes
    -----
    The standard deviation is

    .. math::

        \sigma = \text{gain} \cdot \sqrt{\frac{2}{n_\text{in} + n_\text{out}}},

    which yields variance

    .. math::

        \mathrm{Var}(W) = \frac{2 \cdot \text{gain}^2}{n_\text{in} + n_\text{out}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import xavier_normal_, calculate_gain
    >>> w = lucid.empty(64, 32)
    >>> lucid.manual_seed(0)
    >>> _ = xavier_normal_(w, gain=calculate_gain('tanh'))
    >>> std = calculate_gain('tanh') * (2 / (32 + 64)) ** 0.5
    >>> abs(w.std().item() - std) < 0.1 * std
    True
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std, generator=generator)


@_skippable
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with Kaiming (He) uniform initialisation.

    Draws each entry uniformly from :math:`[-b, b]` with the bound
    chosen so that the variance of activations is preserved across a
    stack of ReLU-family layers.  Introduced in He et al. (2015), this
    scheme corrects the Xavier formula for the fact that ReLU zeroes
    half of its pre-activations, which would otherwise halve the
    forward variance at every layer.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; must have at least 2 dimensions.
    a : float, optional
        Negative slope of the rectifier used after this layer (only
        used when ``nonlinearity='leaky_relu'``).  Default ``0``.
    mode : {'fan_in', 'fan_out'}, optional
        Which fan to use as the variance scaler.  ``'fan_in'`` (default)
        preserves the magnitude of activations in the forward pass;
        ``'fan_out'`` preserves the magnitude of gradients in the
        backward pass.
    nonlinearity : str, optional
        Nonlinearity name forwarded to :func:`calculate_gain`.
        Default ``'leaky_relu'``.
    generator : Generator, optional
        PRNG used for the Kaiming uniform draw.  Default ``None``.

    Returns
    -------
    Tensor
        The Kaiming-uniform-initialised ``tensor``.

    Notes
    -----
    With :math:`n = \text{fan}` (selected by ``mode``) the bound is

    .. math::

        b = \sqrt{\frac{6}{n}} \cdot \text{gain},

    giving variance

    .. math::

        \mathrm{Var}(W) = \frac{\text{gain}^2}{n} = \frac{2}{n}
        \quad\text{for ReLU.}

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import kaiming_uniform_
    >>> w = lucid.empty(64, 32)
    >>> _ = kaiming_uniform_(w, nonlinearity='relu')
    >>> bound = 2.0**0.5 * (3 / 32) ** 0.5   # gain * sqrt(3 / fan_in)
    >>> bool((w.abs() <= bound).all().item())
    True
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, generator=generator)


@_skippable
def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with Kaiming (He) normal initialisation.

    Gaussian counterpart of :func:`kaiming_uniform_`.  Draws each entry
    from :math:`\mathcal{N}(0, \sigma^2)` with a standard deviation
    chosen to preserve activation (or gradient) variance across a stack
    of ReLU-family layers, as proposed in He et al. (2015).

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; must have at least 2 dimensions.
    a : float, optional
        Negative slope of the rectifier (only used when
        ``nonlinearity='leaky_relu'``).  Default ``0``.
    mode : {'fan_in', 'fan_out'}, optional
        ``'fan_in'`` preserves forward-pass variance, ``'fan_out'``
        preserves backward-pass gradient variance.  Default ``'fan_in'``.
    nonlinearity : str, optional
        Activation name forwarded to :func:`calculate_gain`.  Default
        ``'leaky_relu'``.
    generator : Generator, optional
        PRNG used for the Kaiming Gaussian draw.  Default ``None``.

    Returns
    -------
    Tensor
        The Kaiming-normal-initialised ``tensor``.

    Notes
    -----
    With :math:`n = \text{fan}` the standard deviation is

    .. math::

        \sigma = \frac{\text{gain}}{\sqrt{n}},

    so that for ReLU :math:`\mathrm{Var}(W) = 2 / n`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import kaiming_normal_
    >>> w = lucid.empty(64, 32)
    >>> lucid.manual_seed(0)
    >>> _ = kaiming_normal_(w, nonlinearity='relu')
    >>> std = 2.0**0.5 / 32**0.5                 # gain / sqrt(fan_in) = 0.25
    >>> abs(w.std().item() - std) < 0.1 * std
    True
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std, generator=generator)


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@_skippable
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with truncated normal samples.

    Each entry is drawn from :math:`\mathcal{N}(\text{mean},
    \text{std}^2)` truncated to ``[a, b]`` via the inverse-CDF transform
    (:func:`~lucid.erfinv` applied to uniform draws in
    ``[2*Phi((a-mean)/std)-1, 2*Phi((b-mean)/std)-1]``).
    This is the preferred initialiser for transformer weights (e.g.
    ViT, BERT) where unbounded Gaussian tails would otherwise produce
    rare but disruptive activations.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; any shape is accepted.
    mean : float, optional
        Mean of the underlying (untruncated) Gaussian.  Default ``0.0``.
    std : float, optional
        Standard deviation of the underlying Gaussian.  Default ``1.0``.
    a : float, optional
        Lower truncation bound.  Default ``-2.0``.
    b : float, optional
        Upper truncation bound.  Default ``2.0``.
    generator : Generator, optional
        PRNG used for the inverse-CDF uniform quantile draw.  Default ``None``.

    Returns
    -------
    Tensor
        ``tensor`` populated with truncated normal draws.

    Notes
    -----
    The conditional density is

    .. math::

        p(x \mid a \le x \le b) =
        \frac{1}{Z}\,
        \frac{1}{\sqrt{2\pi}\,\text{std}}\,
        \exp\!\left(-\frac{(x - \text{mean})^2}{2\,\text{std}^2}\right),
        \quad x \in [a, b],

    where :math:`Z = \Phi((b - \text{mean})/\text{std}) -
    \Phi((a - \text{mean})/\text{std})` is the normalising constant.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import trunc_normal_
    >>> w = lucid.empty(64, 32)
    >>> _ = trunc_normal_(w, mean=0.0, std=0.02, a=-0.04, b=0.04)
    >>> bool(((w >= -0.04) & (w <= 0.04)).all().item())
    True
    """
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    l = _norm_cdf((a - mean) / std)
    u = _norm_cdf((b - mean) / std)
    lo = 2.0 * l - 1.0
    hi = 2.0 * u - 1.0
    dt = tensor._impl.dtype
    dev = tensor._impl.device
    if a > b or lo > hi:
        _C_engine.uniform([1], 1.0, 0.0, dt, dev, generator)
    shape = list(tensor.shape)
    if 0 in shape:
        _C_engine.uniform(shape, 0.0, 1.0, dt, dev, generator)
        return tensor
    work_dt = _C_engine.F32 if dt in (_C_engine.F16, _C_engine.BF16) else dt
    if (
        dev == _C_engine.Device.CPU
        and work_dt in (_C_engine.F32, _C_engine.F64)
        and (abs(lo) > 0.999999 or abs(hi) > 0.999999)
    ):
        work_dt = _C_engine.F64
    if lo == hi:
        _C_engine.uniform(shape, 0.0, 1.0, work_dt, dev, generator)
        clamped = _C_engine.full(shape, a, work_dt, dev)
    else:
        u_impl = _C_engine.uniform(shape, lo, hi, work_dt, dev, generator)
        z = _C_engine.erfinv(u_impl)
        scale = _C_engine.full([], std * math.sqrt(2.0), work_dt, dev)
        loc = _C_engine.full([], mean, work_dt, dev)
        scaled = _C_engine.add(_C_engine.mul(z, scale), loc)
        clamped = _C_engine.clip(scaled, a, b)
    return _fill_from_impl(tensor, clamped)


@_skippable
def orthogonal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise ``tensor`` in-place with a (semi-)orthogonal matrix.

    A random Gaussian matrix is drawn and orthonormalised via QR
    decomposition; the resulting matrix ``Q`` is multiplied by ``gain``
    and written into ``tensor``.  For tensors with rank > 2 the leading
    axis is flattened against the remaining axes, the 2-D matrix is
    orthogonalised, and the original shape is restored.

    Orthogonal initialisation, proposed by Saxe et al. (2013), preserves
    vector norms exactly through deep *linear* networks
    (:math:`\|Wx\|_2 = \|x\|_2` because ``W`` has orthonormal columns),
    keeping the spectrum of the forward Jacobian on the unit sphere.
    This is particularly valuable for RNNs and pre-norm transformer
    blocks where activation magnitudes can otherwise drift exponentially
    with depth.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in place; must have at least 2 dimensions.
    gain : float, optional
        Multiplicative scaling applied to ``Q``.  Default ``1.0``.
    generator : Generator, optional
        PRNG used to draw the pre-QR Gaussian matrix.  Default ``None``.

    Returns
    -------
    Tensor
        ``tensor`` populated with a Haar-distributed orthogonal matrix.

    Raises
    ------
    ValueError
        If ``tensor.ndim < 2``.

    Notes
    -----
    The matrix satisfies :math:`Q^\top Q = \text{gain}^2 I` (or
    :math:`Q Q^\top = \text{gain}^2 I` when fan-out exceeds fan-in).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import orthogonal_
    >>> w = lucid.empty(64, 64)
    >>> _ = orthogonal_(w, gain=1.0)
    >>> bool(lucid.allclose(w.mT @ w, lucid.eye(64), atol=1e-4))
    True
    """
    if tensor.ndim < 2:
        raise ValueError("orthogonal_() requires at least a 2D tensor")
    if tensor.numel() == 0:
        return tensor
    rows: int = int(tensor.shape[0])
    cols: int = int(tensor.numel() // rows)
    # All work on the engine — no numpy.
    flat: Tensor = _lucid.randn(rows, cols, device=tensor._impl.device, generator=generator)
    # Use QR on the (max-dim × min-dim) shape so we always get an
    # orthonormal Q with the right number of columns, and multiply by
    # sign(diag(R)) so Q is Haar-distributed on O(n).
    if rows < cols:
        flat = flat.mT
    q, r = _lucid.linalg.qr(flat)
    d = r.diagonal(dim1=-2, dim2=-1)
    q = q * d.sign()
    if rows < cols:
        q = q.mT
    if gain != 1.0:
        q = q * gain
    src_t: Tensor = _lucid.tensor(q, dtype=tensor._impl.dtype, device=tensor._impl.device)
    return _fill_from_impl(tensor, src_t._impl)


@_skippable
def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = 0.01,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Initialise a 2-D ``tensor`` in-place with a sparse random matrix.

    For each column, a random subset of ``ceil(sparsity * rows)`` rows
    is set to zero; the remaining entries are drawn from
    :math:`\mathcal{N}(0, \text{std}^2)`.  Sparse initialisation, due to
    Martens (2010), encourages each output unit to depend on only a
    small number of inputs at the start of training and helps in
    ill-conditioned regimes — particularly for very wide layers where a
    dense Gaussian would have an unfavourable condition number.

    Parameters
    ----------
    tensor : Tensor
        Two-dimensional tensor to fill in place.
    sparsity : float
        Fraction of entries in each column that are zeroed.  Must lie in
        ``[0, 1]``.
    std : float, optional
        Standard deviation of the Gaussian used for the non-zero
        entries.  Default ``0.01``.
    generator : Generator, optional
        PRNG used for both the Gaussian fill and the column zero-mask.
        Default ``None``.

    Returns
    -------
    Tensor
        ``tensor`` populated with a column-sparse Gaussian matrix.

    Raises
    ------
    ValueError
        If ``tensor.ndim != 2`` or ``sparsity`` is outside ``[0, 1]``.

    Notes
    -----
    The expected per-entry variance is

    .. math::

        \mathrm{Var}(W_{ij}) = (1 - \text{sparsity}) \cdot \text{std}^2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import sparse_
    >>> w = lucid.empty(256, 256)
    >>> _ = sparse_(w, sparsity=0.9, std=0.01)
    >>> int((w == 0).sum().item())   # ceil(0.9 * 256) = 231 zeros in each column
    59136
    """
    if tensor.ndim != 2:
        raise ValueError("sparse_() requires a 2D tensor")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity!r}")
    if tensor.numel() == 0:
        return tensor

    rows: int = int(tensor.shape[0])
    cols: int = int(tensor.shape[1])
    n_zero: int = int(math.ceil(sparsity * rows))
    dev = tensor._impl.device
    # Per-column random row selection: ``argsort`` of uniform draws gives a
    # uniform permutation; take the first ``n_zero`` rows of each column.
    src: Tensor = _lucid.normal(0.0, std, size=(rows, cols), device=dev, generator=generator)
    if n_zero > 0:
        # noise is shape (rows, cols); per-column argsort along dim=0.
        noise: Tensor = _lucid.rand(rows, cols, device=dev, generator=generator)
        perm: Tensor = noise.argsort(dim=0)  # (rows, cols) int.
        zero_rows: Tensor = perm.narrow(0, 0, min(n_zero, rows))  # (n_zero, cols).
        # Build a (rows, cols) bool mask: 1 where a row was chosen.
        mask: Tensor = _lucid.zeros(rows, cols, device=dev)
        ones: Tensor = _lucid.ones(zero_rows.shape[0], cols, device=dev)
        mask = mask.scatter_add(0, zero_rows, ones)
        src = _lucid.where(mask > 0.0, _lucid.zeros_like(src), src)
    src_t: Tensor = _lucid.tensor(src, dtype=tensor._impl.dtype, device=dev)
    return _fill_from_impl(tensor, src_t._impl)


@_skippable
def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    r"""Initialise a 3/4/5-D convolution weight in-place as a Dirac delta.

    The kernel is filled with zeros except for a single ``1`` at the
    spatial centre of each ``(out_channel, in_channel)`` matched pair,
    so that — at initialisation — the convolution acts as the identity
    function on its input channels (subject to channel-count matching
    per group).  Useful for residual networks where one wants gradients
    to flow unchanged through deep stacks at step 0.

    Parameters
    ----------
    tensor : Tensor
        Convolution weight of shape
        ``(out_channels, in_channels // groups, *K)`` where ``K`` is
        the 1-D, 2-D, or 3-D spatial kernel shape.
    groups : int, optional
        Number of groups in the convolution.  ``out_channels`` must be
        divisible by ``groups``.  Default ``1``.

    Returns
    -------
    Tensor
        ``tensor`` (mutated) for chaining.

    Raises
    ------
    ValueError
        If ``tensor.ndim`` is not in ``{3, 4, 5}`` or if
        ``out_channels`` is not divisible by ``groups``.

    Notes
    -----
    For convolution input :math:`x` with the resulting weight,
    :math:`(W \star x)_{:n} = x_{:n}` where
    :math:`n = \min(\text{in\_channels}/\text{groups},\,
    \text{out\_channels}/\text{groups})`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.init import dirac_
    >>> w = lucid.empty(16, 16, 3, 3)  # (out, in, kH, kW)
    >>> _ = dirac_(w)
    >>> bool((w[:, :, 1, 1] == lucid.eye(16)).all().item())   # a 1 at each centre
    True
    >>> float(w.sum().item())
    16.0
    """
    if tensor.ndim not in (3, 4, 5):
        raise ValueError(f"dirac_() expects a 3/4/5-D tensor; got ndim={tensor.ndim}")
    out_ch: int = int(tensor.shape[0])
    in_ch_per_group: int = int(tensor.shape[1])
    if out_ch % groups != 0:
        raise ValueError(f"out_channels ({out_ch}) must be divisible by groups ({groups})")
    if tensor.numel() == 0:
        return tensor

    out_per_group: int = out_ch // groups
    min_dim: int = min(in_ch_per_group, out_per_group)
    spatial_centres: tuple[int, ...] = tuple(int(s) // 2 for s in tensor.shape[2:])
    dev = tensor._impl.device
    shape: tuple[int, ...] = tuple(int(s) for s in tensor.shape)
    src: Tensor = _lucid.zeros(*shape, device=dev)
    # Each (out_idx, d, *centres) gets a 1.  Build per-axis index tensors of
    # length ``groups·min_dim`` and use ``index_put_`` for the in-place write.
    n_writes: int = groups * min_dim
    if n_writes > 0:
        out_idx_list: list[int] = []
        in_idx_list: list[int] = []
        for g in range(groups):
            for d in range(min_dim):
                out_idx_list.append(g * out_per_group + d)
                in_idx_list.append(d)
        idx_tensors: list[Tensor] = [
            _lucid.tensor(out_idx_list, dtype=_lucid.int64, device=dev),
            _lucid.tensor(in_idx_list, dtype=_lucid.int64, device=dev),
        ]
        for c in spatial_centres:
            idx_tensors.append(_lucid.tensor([c] * n_writes, dtype=_lucid.int64, device=dev))
        src = _lucid.index_put(
            src,
            tuple(idx_tensors),
            _lucid.ones(n_writes, device=dev),
        )
    src_t: Tensor = _lucid.tensor(src, dtype=tensor._impl.dtype, device=dev)
    return _fill_from_impl(tensor, src_t._impl)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    r"""Return the recommended variance-preserving gain for an activation.

    The gain is a multiplicative correction applied to the standard
    deviation of Xavier / Kaiming initialisation so the per-layer
    activation variance is preserved after the nonlinearity.  Values
    follow the original recommendations in Glorot & Bengio (2010) and
    He et al. (2015).

    Parameters
    ----------
    nonlinearity : str
        Name of the activation function.  One of ``'linear'``,
        ``'conv1d'``, ``'conv2d'``, ``'conv3d'``,
        ``'conv_transpose1d'``, ``'conv_transpose2d'``,
        ``'conv_transpose3d'``, ``'sigmoid'``, ``'tanh'``, ``'relu'``,
        ``'leaky_relu'``, ``'selu'``.
    param : float, optional
        Negative slope of the rectifier — only consulted when
        ``nonlinearity='leaky_relu'``.  Default ``0.01``.

    Returns
    -------
    float
        Recommended gain factor.

    Raises
    ------
    ValueError
        If ``nonlinearity`` is not in the supported set, or if ``param``
        is not a valid number when ``nonlinearity='leaky_relu'``.

    Notes
    -----
    Mapping table:

    ===============  ============================================
    nonlinearity     gain
    ===============  ============================================
    ``linear``       :math:`1`
    ``conv{1,2,3}d`` :math:`1`
    ``sigmoid``      :math:`1`
    ``tanh``         :math:`5/3 \approx 1.667`
    ``relu``         :math:`\sqrt{2} \approx 1.414`
    ``leaky_relu``   :math:`\sqrt{2 / (1 + a^2)}`
    ``selu``         :math:`3/4`
    ===============  ============================================

    Examples
    --------
    >>> from lucid.nn.init import calculate_gain
    >>> calculate_gain('relu')
    1.4142135623730951
    >>> calculate_gain('leaky_relu', 0.2)
    1.3867504905630728
    """
    _gains: dict[str, float] = {
        "linear": 1.0,
        "conv1d": 1.0,
        "conv2d": 1.0,
        "conv3d": 1.0,
        "conv_transpose1d": 1.0,
        "conv_transpose2d": 1.0,
        "conv_transpose3d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "selu": 3.0 / 4.0,
    }
    if nonlinearity == "leaky_relu":
        param_num = _promotion.as_number(param)
        if param_num is None:
            slope: float = 0.01
        elif not isinstance(param_num, bool) and isinstance(param_num, (int, float)):
            slope = float(param_num)
        else:
            raise ValueError(f"negative_slope {param!r} not a valid number")
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity in _gains:
        return _gains[nonlinearity]
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity!r}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Compute fan_in and fan_out for Linear and Conv weight tensors."""
    ndim = tensor.ndim
    if ndim < 2:
        raise ValueError("fan_in/fan_out requires at least 2D tensor")
    receptive = 1
    if ndim > 2:
        for s in tensor.shape[2:]:
            receptive *= s
    fan_in = tensor.shape[1] * receptive
    fan_out = tensor.shape[0] * receptive
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    raise ValueError(f"Unknown mode: {mode!r}")


# Non-inplace aliases (deprecated in the reference framework but still widely used).
constant = constant_
dirac = dirac_
eye = eye_
kaiming_normal = kaiming_normal_
kaiming_uniform = kaiming_uniform_
normal = normal_
orthogonal = orthogonal_
sparse = sparse_
uniform = uniform_
xavier_normal = xavier_normal_
xavier_uniform = xavier_uniform_

__all__ = [
    "calculate_gain",
    "constant_",
    "constant",
    "dirac_",
    "dirac",
    "eye_",
    "eye",
    "kaiming_normal_",
    "kaiming_normal",
    "kaiming_uniform_",
    "kaiming_uniform",
    "normal_",
    "normal",
    "ones_",
    "orthogonal_",
    "orthogonal",
    "sparse_",
    "sparse",
    "trunc_normal_",
    "uniform_",
    "uniform",
    "xavier_normal_",
    "xavier_normal",
    "xavier_uniform_",
    "xavier_uniform",
    "zeros_",
]
