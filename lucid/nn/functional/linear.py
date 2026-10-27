"""
nn.functional linear (fully-connected) operations.

Includes fused forward kernels:
  ``fused_linear_relu``  — y = max(0, x @ W.T + b)   (BLAS + vDSP, single pass)
  ``fused_linear_gelu``  — y = GELU(x @ W.T + b)     (BLAS + vForce tanh, single pass)

Both kernels bypass the standard two-op graph (matmul → activation) in
**inference mode** (``no_grad`` context or when no inputs require grad).
In training mode they transparently fall back to the unfused path so
autograd gradients remain correct.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._types import GeluApproximate

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    r"""Apply an affine linear transformation :math:`y = x W^\top + b`.

    The canonical fully-connected layer kernel — every dense / MLP /
    transformer block in Lucid ultimately routes through this call.
    ``weight`` is stored in row-major ``(out_features, in_features)``
    orientation (matching the rest of the framework's matmul-of-transpose
    convention), and the leading batch dims of ``input`` are preserved
    untouched.

    Parameters
    ----------
    input : Tensor
        Input activations, shape ``(..., in_features)``.  Any number of
        leading batch dimensions is allowed.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor, optional
        Bias vector, shape ``(out_features,)``.  When ``None`` (default)
        the kernel runs with an implicit zero bias allocated on the same
        device / dtype as ``weight`` — equivalent to the unbiased form
        :math:`y = x W^\top`.

    Returns
    -------
    Tensor
        Output activations, shape ``(..., out_features)``.

    Notes
    -----
    Implementation dispatches to the engine's :func:`nn.linear` kernel:
    a single BLAS SGEMM on the CPU stream (Accelerate) or an MLX matmul
    on the GPU stream, fused with the bias addition.  Backward stores
    the saved ``input`` / ``weight`` so the chain-rule path

    .. math::

        \frac{\partial y}{\partial x} = W,\quad
        \frac{\partial y}{\partial W} = x^\top,\quad
        \frac{\partial y}{\partial b} = \mathbf{1}

    can be evaluated with two more SGEMMs in :class:`LinearBackward`.
    For inference-only ReLU / GELU follow-up, callers can invoke
    :func:`fused_linear_relu` / :func:`fused_linear_gelu` to skip the
    activation's intermediate buffer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import linear
    >>> x = lucid.randn(4, 16)            # batch of 4
    >>> w = lucid.randn(32, 16)
    >>> b = lucid.zeros(32)
    >>> linear(x, w, b).shape
    (4, 32)
    >>> linear(x, w).shape                # bias=None, unbiased form
    (4, 32)

    See Also
    --------
    lucid.nn.Linear : ``nn.Module`` wrapper that owns the parameters.
    fused_linear_relu : fused linear + ReLU inference kernel.
    fused_linear_gelu : fused linear + GELU inference kernel.
    bilinear : two-input quadratic-interaction variant.
    """
    w_impl = _unwrap(weight)
    out_features = w_impl.shape[0]
    if bias is not None:
        b_impl = _unwrap(bias)
    else:
        b_impl = _C_engine.zeros([out_features], w_impl.dtype, w_impl.device)
    return _wrap(_C_engine.nn.linear(_unwrap(input), w_impl, b_impl))


def bilinear(
    input1: Tensor,
    input2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    r"""Bilinear transformation applied to two inputs.

    Computes a quadratic interaction between every pair of features from
    ``input1`` and ``input2``.  Common in fine-grained recognition (bilinear
    pooling), tensor-decomposition attention, and gated cross-modal
    fusion layers.

    Parameters
    ----------
    input1 : Tensor
        First input, shape ``(..., in1_features)``.
    input2 : Tensor
        Second input, shape ``(..., in2_features)``.
    weight : Tensor
        Weight tensor, shape ``(out_features, in1_features, in2_features)``.
    bias : Tensor, optional
        Bias vector, shape ``(out_features,)``.  Default ``None``.

    Returns
    -------
    Tensor
        Output tensor, shape ``(..., out_features)``.

    Notes
    -----
    For each output channel :math:`k`,

    .. math::

        y_k = x_1^\top W_k x_2 + b_k,

    where :math:`W_k` is the :math:`k`-th slice of the weight tensor.
    The parameter count grows multiplicatively in
    :math:`\text{in1} \times \text{in2}`, so bilinear layers are usually
    kept narrow.  Reduces to two ordinary linear layers when
    :math:`W_k = u_k v_k^\top` is rank-one (the *factorised bilinear*
    trick).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import bilinear
    >>> x1 = lucid.randn(2, 3)
    >>> x2 = lucid.randn(2, 4)
    >>> w = lucid.randn(5, 3, 4)
    >>> bilinear(x1, x2, w).shape
    (2, 5)
    """
    # The bias goes to the engine with the other operands, so the op's
    # autocast row (WIDEST on Metal) covers it too.
    bias_impl = _unwrap(bias) if bias is not None else None
    return _wrap(
        _C_engine.nn.bilinear_layer(_unwrap(input1), _unwrap(input2), _unwrap(weight), bias_impl)
    )


# ── Phase 19: Fused forward kernels ──────────────────────────────────────────


def fused_linear_relu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    r"""Fused linear + ReLU forward kernel.

    Computes :math:`\text{ReLU}(xW^\top + b)` in a single CPU pass — the
    pre-activation tensor is never materialised, halving memory traffic on
    large MLP blocks.  Falls back to the unfused two-op path whenever any
    input requires gradient so that autograd remains correct.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., in_features)``.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor
        Bias vector, shape ``(out_features,)``.

    Returns
    -------
    Tensor
        ``relu(x @ weight.T + bias)``, shape ``(..., out_features)``.

    Notes
    -----
    .. math::

        y = \max\!\big(0,\, x W^\top + b\big)

    Implementation: a single BLAS SGEMM followed by vDSP ``vrelu`` over
    the GEMM output buffer for CPU float32; on Metal and in the half
    formats, the backend's linear followed by its ReLU.  Runs the fused
    kernel when grad-mode is disabled or none of ``x`` / ``weight`` /
    ``bias`` has ``requires_grad=True``.  In training mode the call routes
    through the standard :func:`linear` + :func:`relu` graph so autograd's
    backward derivations are exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fused_linear_relu
    >>> x = lucid.randn(8, 16)
    >>> w = lucid.randn(32, 16)
    >>> b = lucid.zeros(32)
    >>> fused_linear_relu(x, w, b).shape
    (8, 32)
    """
    needs_grad: bool = _C_engine.grad_enabled() and (
        x.requires_grad or weight.requires_grad or bias.requires_grad
    )
    if needs_grad:
        # Training path: unfused standard ops — autograd graph is correct.
        return _wrap(
            _C_engine.relu(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
        )

    # Inference path: single-pass BLAS+vDSP fused kernel.
    return _wrap(_C_engine._fused_linear_relu(_unwrap(x), _unwrap(weight), _unwrap(bias)))


def fused_linear_gelu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    approximate: GeluApproximate = "tanh",
) -> Tensor:
    r"""Fused linear + GELU forward kernel.

    Computes :math:`\text{GELU}(xW^\top + b)` in a single CPU pass.  Used
    extensively in transformer feed-forward blocks where the linear-then-
    GELU pattern dominates inference cost; fusing eliminates the
    intermediate activation buffer and improves cache behaviour.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., in_features)``.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor
        Bias vector, shape ``(out_features,)``.
    approximate : str, optional
        GELU approximation.  ``"tanh"`` (default) is what the fused
        kernel implements; ``"none"`` falls back to the unfused exact-erf
        path.

    Returns
    -------
    Tensor
        ``gelu(x @ weight.T + bias)``, shape ``(..., out_features)``.

    Notes
    -----
    .. math::

        y = \text{GELU}\!\big(x W^\top + b\big)

    Implementation: a single BLAS SGEMM followed by vForce ``vtanh``
    over the GEMM output to evaluate the tanh-form GELU
    :math:`\tfrac{x}{2}\big[1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715 x^3))\big]`
    for CPU float32; on Metal and in the half formats, the backend's linear
    followed by the same tanh-form GELU.  Engaged only in inference mode
    (no grad-tracking inputs) and only when ``approximate="tanh"``;
    otherwise the call falls back to the
    standard :func:`linear` + :func:`gelu` graph so gradients remain
    exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fused_linear_gelu
    >>> x = lucid.randn(8, 16)
    >>> w = lucid.randn(32, 16)
    >>> b = lucid.zeros(32)
    >>> fused_linear_gelu(x, w, b).shape
    (8, 32)
    """
    from lucid.nn.functional.activations import gelu as _gelu

    needs_grad: bool = _C_engine.grad_enabled() and (
        x.requires_grad or weight.requires_grad or bias.requires_grad
    )
    if needs_grad:
        # Training path: standard ops — autograd graph is correct.
        pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
        return _gelu(pre, approximate=approximate)

    if approximate == "tanh":
        return _wrap(_C_engine._fused_linear_gelu(_unwrap(x), _unwrap(weight), _unwrap(bias)))
    # exact erf path — no fused kernel available, fall back to two-op path.
    pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
    return _gelu(pre, approximate="none")


def scaled_noise(x: Tensor) -> Tensor:
    r"""The factorised-noise transform of Fortunato et al. (2017).

    .. math::

        f(x) = \mathrm{sign}(x)\sqrt{|x|}

    Parameters
    ----------
    x : Tensor
        Unit Gaussian samples, any shape.

    Returns
    -------
    Tensor
        The same shape, magnitude-compressed with the sign kept.

    Notes
    -----
    Written out because :func:`noisy_linear` applies it to both noise
    vectors and the paper is explicit that the *bias* uses it too: "for
    the bias we could have set :math:`f(x) = x`, but we decided to keep
    the same output noise for weights and biases."

    Composed rather than fused, matching :func:`~lucid.nn.functional.symlog`
    — the same :math:`\mathrm{sign}(x)\,g(|x|)` shape, and the same
    decision.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import scaled_noise
    >>> scaled_noise(lucid.tensor([-4.0, 0.0, 9.0]))
    tensor([-2., 0., 3.])
    """
    import lucid as _l

    return _l.sign(x) * _l.sqrt(_l.abs(x))


def noisy_linear(
    x: Tensor,
    weight_mu: Tensor,
    weight_sigma: Tensor,
    bias_mu: Tensor | None = None,
    bias_sigma: Tensor | None = None,
    epsilon_in: Tensor | None = None,
    epsilon_out: Tensor | None = None,
) -> Tensor:
    r"""A linear layer whose parameters carry learnable noise.

    .. math::

        y = (\mu^w + \sigma^w \odot \varepsilon^w)\,x
            + (\mu^b + \sigma^b \odot \varepsilon^b)

    Parameters
    ----------
    x : Tensor
        Input, ``(*, in_features)``.
    weight_mu, weight_sigma : Tensor
        Both ``(out_features, in_features)`` — the mean weight and the
        scale of the noise on it.
    bias_mu, bias_sigma : Tensor or None, optional
        Both ``(out_features,)``, or ``None`` for an unbiased layer.
    epsilon_in : Tensor or None, optional
        ``(in_features,)`` unit Gaussian samples.  ``None`` draws them.
    epsilon_out : Tensor or None, optional
        ``(out_features,)`` unit Gaussian samples.  ``None`` draws them.

    Returns
    -------
    Tensor
        ``(*, out_features)``.

    Raises
    ------
    ValueError
        If exactly one of ``bias_mu`` and ``bias_sigma`` is given.

    Notes
    -----
    Reference: Fortunato, Azar, Piot, Menick, Hessel, Osband, Graves,
    Mnih, Munos, Hassabis, Pietquin, Blundell, and Legg, *"Noisy Networks
    for Exploration"*, ICLR, 2018 (arXiv:1706.10295), equations 9-11.

    **Factorised** noise, which is the variant the paper uses for the
    single-threaded agents: rather than one sample per weight, it draws
    :math:`p` for the inputs and :math:`q` for the outputs and forms

    .. math::

        \varepsilon^w_{i,j} = f(\varepsilon_i) f(\varepsilon_j),
        \qquad
        \varepsilon^b_j = f(\varepsilon_j),

    so a layer costs :math:`p + q` random numbers instead of :math:`pq`.
    The paper's stated reason is compute time, and it is the reason to
    prefer this form here too.

    The two noise vectors are arguments rather than always drawn, because
    a *sample* has to be held fixed across a whole forward pass — and,
    for an agent, often across a whole episode.  Drawing inside the
    function would make every call a different network.

    Gradients reach :math:`\sigma` through the noise, which is the point:
    the scale of the exploration is learned rather than annealed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import noisy_linear
    >>> x = lucid.zeros((2, 4))
    >>> mu, sigma = lucid.zeros((3, 4)), lucid.ones((3, 4))
    >>> noisy_linear(x, mu, sigma).shape
    (2, 3)

    See Also
    --------
    lucid.nn.NoisyLinear : ``nn.Module`` wrapper that owns the parameters.
    linear : the deterministic layer this perturbs.
    """
    import lucid as _l

    if (bias_mu is None) != (bias_sigma is None):
        raise ValueError("bias_mu and bias_sigma must be given together or not at all")

    out_features, in_features = (int(v) for v in weight_mu.shape)
    if epsilon_in is None:
        epsilon_in = _l.randn((in_features,), device=weight_mu.device, dtype=weight_mu.dtype)
    if epsilon_out is None:
        epsilon_out = _l.randn((out_features,), device=weight_mu.device, dtype=weight_mu.dtype)

    f_in = scaled_noise(epsilon_in)
    f_out = scaled_noise(epsilon_out)
    # The outer product is the factorisation: one draw per input and per
    # output, combined into the (q, p) perturbation the weight needs.
    weight = weight_mu + weight_sigma * (
        f_out.reshape(out_features, 1) * f_in.reshape(1, in_features)
    )

    bias: Tensor | None = None
    if bias_mu is not None and bias_sigma is not None:
        bias = bias_mu + bias_sigma * f_out
    return linear(x, weight, bias)
