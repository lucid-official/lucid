"""
Linear and related fully-connected layers.
"""

import math
from typing import override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid.nn.modules._lazy import _LazyModule, fill as _fill
import lucid.nn.init as init
from lucid._factories.creation import empty
from lucid.nn.functional.linear import (
    linear,
    noisy_linear,
    bilinear,
    fused_linear_relu,
    fused_linear_gelu,
)
from lucid._types import StateDict


class Linear(Module):
    r"""Apply a learnable affine transformation to incoming data.

    Computes the linear map

    .. math::

        \mathbf{y} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    where :math:`\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}`
    is the weight matrix and :math:`\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}` is
    the optional bias vector.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample (:math:`d_{\text{in}}`).
    out_features : int
        Dimensionality of each output sample (:math:`d_{\text{out}}`).
    bias : bool, optional
        If ``True`` (default) a learnable bias :math:`\mathbf{b}` is added to the
        output.  Set to ``False`` when a subsequent normalization layer already
        absorbs the bias (e.g. ``BatchNorm1d``).
    device : DeviceLike, optional
        Device on which the initial parameters are allocated (``'cpu'`` or
        ``'metal'``).  Defaults to the global default device.
    dtype : DTypeLike, optional
        Floating-point dtype for the initial parameters.  Defaults to the global
        default dtype (``float32``).

    Attributes
    ----------
    weight : Parameter
        Learnable weight matrix of shape ``(out_features, in_features)``.
        Initialized with **Kaiming uniform** sampling:

        .. math::

            \mathbf{W}_{ij} \sim \mathcal{U}\!\left(
                -\sqrt{\tfrac{6}{(1 + a^2)\,d_{\text{in}}}},\;
                \sqrt{\tfrac{6}{(1 + a^2)\,d_{\text{in}}}}
            \right)

        where :math:`a = \sqrt{5}` is the default negative-slope parameter.
        This keeps gradient variance roughly constant across layers at
        initialization — critical for training stability in deep networks.

    bias : Parameter or None
        Learnable bias vector of shape ``(out_features,)``.  Initialized with
        uniform sampling over :math:`\left[-\tfrac{1}{\sqrt{d_{\text{in}}}},\;
        \tfrac{1}{\sqrt{d_{\text{in}}}}\right]`.
        ``None`` when ``bias=False``.

    Shape
    -----
    - **Input**: :math:`(\ast, d_{\text{in}})` — any number of leading batch
      dimensions followed by ``in_features``.
    - **Output**: :math:`(\ast, d_{\text{out}})` — same leading dimensions,
      last axis replaced by ``out_features``.

    Notes
    -----
    ``Linear`` is the most common building block in feed-forward sub-layers
    (e.g. the MLP inside a Transformer block uses two ``Linear`` layers with a
    non-linearity in between).  When composing many layers in sequence the
    Kaiming initialization ensures that neither the forward activations nor the
    backward gradients explode or vanish at the start of training.

    Examples
    --------
    Basic usage with a 2-D input:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.Linear(20, 10)
    >>> x = lucid.randn(4, 20)   # batch of 4, 20 features each
    >>> y = m(x)
    >>> y.shape
    (4, 10)

    Higher-dimensional inputs (batch + sequence):

    >>> m = nn.Linear(512, 256)
    >>> x = lucid.randn(2, 32, 512)   # (batch, seq_len, d_model)
    >>> m(x).shape
    (2, 32, 256)

    Disable bias for use before a normalization layer:

    >>> m_no_bias = nn.Linear(128, 64, bias=False)
    >>> m_no_bias.bias is None
    True
    >>> lucid.randn(8, 128).shape == (8, 128)
    True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Linear module. See the class docstring for parameter semantics."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias: Parameter | None = Parameter(empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight with Kaiming uniform and bias with uniform fan_in bound."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the linear transformation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(*, \text{in\_features})`.

        Returns
        -------
        Tensor
            Output tensor of shape :math:`(*, \text{out\_features})`.
        """
        return linear(input, self.weight, self.bias)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Identity(Module):
    r"""Pass-through layer that returns its input unchanged.

    For any input :math:`\mathbf{x}` the layer computes

    .. math::

        f(\mathbf{x}) = \mathbf{x}

    with no learnable parameters.

    Notes
    -----
    ``Identity`` is primarily useful as a **structural placeholder** when a
    layer slot must be filled but no transformation is desired.  Common
    scenarios include:

    * **Ablation studies** — swap out a component (e.g. a dropout or
      normalization layer) with ``Identity`` to isolate its effect without
      refactoring the surrounding ``nn.Sequential``.
    * **Conditional composition** — build models where a sub-network is
      either a real layer or a no-op depending on a configuration flag::

          head = nn.Linear(256, num_classes) if use_head else nn.Identity()

    * **Skip connections** — act as the identity branch in a residual block
      when the channel dimensions already match, avoiding any projection.

    Shape
    -----
    - **Input**: any shape :math:`(\ast)`.
    - **Output**: identical shape :math:`(\ast)` — same data, same storage.

    Examples
    --------
    Use as a drop-in replacement for a disabled component:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> layer = nn.Identity()
    >>> x = lucid.randn(3, 64)
    >>> y = layer(x)
    >>> (y - x).abs().max().item() == 0.0
    True

    Conditional head in an ``nn.Sequential``-style pipeline:

    >>> use_projection = False
    >>> proj = nn.Linear(512, 512) if use_projection else nn.Identity()
    >>> x = lucid.randn(1, 512)
    >>> proj(x).shape
    (1, 512)
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any arguments.

        So ``Identity`` can stand in wherever a layer is built from
        arguments — ``norm_layer(dim)``, ``act(inplace=True)`` — without a
        special case at the call site.  It raised on any argument before.
        """
        super().__init__()
        del args, kwargs

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Return the input unchanged.

        Parameters
        ----------
        input : Tensor
            Input of any shape.

        Returns
        -------
        Tensor
            ``input`` itself.
        """
        return input


class FusedLinear(Module):
    r"""Linear layer with a kernel-fused non-linear activation.

    Computes

    .. math::

        \mathbf{y} = \sigma\!\left(\mathbf{x}\mathbf{W}^{\top} + \mathbf{b}\right)

    where :math:`\sigma` is one of the supported pointwise activations.

    **Inference mode** dispatches to a single BLAS + Accelerate pass that
    avoids allocating the intermediate pre-activation tensor.  **Training
    mode** falls back to unfused, differentiable ops so that the autograd
    engine can compute correct gradients through both the linear projection
    and the activation.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample (:math:`d_{\text{in}}`).
    out_features : int
        Dimensionality of each output sample (:math:`d_{\text{out}}`).
    activation : str, optional
        Fused activation function.  Supported values:

        * ``'relu'`` (default) — rectified linear unit,
          :math:`\sigma(z) = \max(0, z)`.  Fastest; preferred for
          intermediate hidden layers.
        * ``'gelu'`` — Gaussian error linear unit (tanh approximation),
          :math:`\sigma(z) = z \cdot \tfrac{1}{2}
          \left[1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}
          \left(z + 0.044715 z^3\right)\right)\right]`.
          Preferred in Transformer MLP blocks.
    bias : bool, optional
        If ``True`` (default) a learnable bias is added before the
        activation.  When ``bias=False`` the fused kernel is unavailable
        and the layer falls back to standard unfused ops even during
        inference.
    device : DeviceLike, optional
        Device for initial parameters.
    dtype : DTypeLike, optional
        Dtype for initial parameters.

    Attributes
    ----------
    weight : Parameter
        Weight matrix of shape ``(out_features, in_features)``.
        Initialized with Kaiming uniform (same scheme as ``Linear``).
    bias : Parameter or None
        Bias vector of shape ``(out_features,)``.
        ``None`` when ``bias=False``.
    activation : str
        Name of the fused activation (``'relu'`` or ``'gelu'``).
    in_features : int
        Stored input dimensionality.
    out_features : int
        Stored output dimensionality.

    Shape
    -----
    - **Input**: :math:`(\ast, d_{\text{in}})`.
    - **Output**: :math:`(\ast, d_{\text{out}})` after activation.

    Notes
    -----
    The kernel fusion benefit is most pronounced for **inference-only
    workloads** (e.g. model serving with ``lucid.no_grad()``).  During
    training, the unfused fallback ensures that every intermediate value
    needed for backpropagation is materialised correctly.

    For ``bias=False`` the fused path is always skipped; prefer using
    ``bias=True`` to take advantage of fusion.

    Examples
    --------
    Inference with ReLU activation:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.FusedLinear(64, 256, activation='relu')
    >>> x = lucid.randn(4, 64)
    >>> with lucid.no_grad():
    ...     y = m(x)   # single-pass fused kernel on CPU
    >>> y.shape
    (4, 256)

    GELU activation for a Transformer MLP block:

    >>> mlp = nn.FusedLinear(768, 3072, activation='gelu')
    >>> x = lucid.randn(2, 16, 768)   # (batch, seq_len, d_model)
    >>> with lucid.no_grad():
    ...     out = mlp(x)
    >>> out.shape
    (2, 16, 3072)
    """

    _SUPPORTED = frozenset({"relu", "gelu"})

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the FusedLinear module. See the class docstring for parameter semantics."""
        super().__init__()
        if activation not in self._SUPPORTED:
            raise ValueError(
                f"FusedLinear: unsupported activation '{activation}'. "
                f"Choose from {sorted(self._SUPPORTED)}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        import math as _math
        import lucid.nn.init as _init

        self.weight = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias: Parameter | None = Parameter(empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

        _init.kaiming_uniform_(self.weight, a=_math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / _math.sqrt(fan_in) if fan_in > 0 else 0.0
            _init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the linear transformation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(*, \text{in\_features})`.

        Returns
        -------
        Tensor
            Output tensor of shape :math:`(*, \text{out\_features})`.
        """
        if self.bias is None:
            # No fused kernel for bias=False — fall back to standard ops.
            import lucid.nn.functional as F

            act = F.relu if self.activation == "relu" else F.gelu
            return act(linear(x, self.weight, None))

        if self.activation == "relu":
            return fused_linear_relu(x, self.weight, self.bias)
        return fused_linear_gelu(x, self.weight, self.bias)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"activation={self.activation!r}, bias={self.bias is not None}"
        )


class Bilinear(Module):
    r"""Apply a bilinear transformation to a pair of input tensors.

    For each output unit :math:`k` the layer computes

    .. math::

        y_k = \mathbf{x}_1^{\top} \mathbf{W}_{k,:,:}\, \mathbf{x}_2 + b_k,
        \qquad k = 1, \dots, d_{\text{out}}

    where :math:`\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}_1}
    \times d_{\text{in}_2}}` is the weight tensor and
    :math:`\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}` is the optional bias.

    Parameters
    ----------
    in1_features : int
        Dimensionality of the first input (:math:`d_{\text{in}_1}`).
    in2_features : int
        Dimensionality of the second input (:math:`d_{\text{in}_2}`).
    out_features : int
        Number of output units (:math:`d_{\text{out}}`).
    bias : bool, optional
        If ``True`` (default) add a learnable bias term :math:`\mathbf{b}`.
    device : DeviceLike, optional
        Device for initial parameters.
    dtype : DTypeLike, optional
        Dtype for initial parameters.

    Attributes
    ----------
    weight : Parameter
        Weight tensor of shape ``(out_features, in1_features, in2_features)``.
        Each slice ``weight[k]`` is a matrix that mixes the two input spaces
        for the :math:`k`-th output unit.  Initialized with uniform sampling
        over :math:`\left[-\tfrac{1}{\sqrt{d_{\text{in}_1}}},\;
        \tfrac{1}{\sqrt{d_{\text{in}_1}}}\right]`.
    bias : Parameter or None
        Bias vector of shape ``(out_features,)``.
        ``None`` when ``bias=False``.

    Shape
    -----
    - **Input 1** (``x1``): :math:`(\ast,\, d_{\text{in}_1})`.
    - **Input 2** (``x2``): :math:`(\ast,\, d_{\text{in}_2})`.
    - **Output**: :math:`(\ast,\, d_{\text{out}})`.

    Notes
    -----
    ``Bilinear`` captures *multiplicative interactions* between two feature
    vectors that a plain ``Linear`` layer cannot express.  Typical use cases
    include:

    * **Attention scoring** — compute compatibility between query and key
      vectors using a learned interaction matrix instead of the dot product.
    * **Relation networks** — model pair-wise relationships in graph
      neural networks or visual question answering.
    * **Similarity scoring** — learn an asymmetric distance metric between
      two embeddings (e.g. in contrastive or metric-learning settings).

    Examples
    --------
    Scoring query–key compatibility in an attention mechanism:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> attn_score = nn.Bilinear(64, 64, 1)
    >>> q = lucid.randn(8, 64)   # queries
    >>> k = lucid.randn(8, 64)   # keys
    >>> scores = attn_score(q, k)
    >>> scores.shape
    (8, 1)

    Multi-output relation layer with different input spaces:

    >>> rel = nn.Bilinear(128, 256, 32)
    >>> x1 = lucid.randn(4, 128)
    >>> x2 = lucid.randn(4, 256)
    >>> rel(x1, x2).shape
    (4, 32)
    """

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Bilinear module. See the class docstring for parameter semantics."""
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            empty(out_features, in1_features, in2_features, dtype=dtype, device=device)
        )
        self.bias: Parameter | None = (
            Parameter(empty(out_features, dtype=dtype, device=device)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with Kaiming uniform."""
        import math

        bound = 1.0 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        r"""Apply the linear transformation to the input tensor.

        Parameters
        ----------
        input1 : Tensor
            First input tensor of shape :math:`(*, \text{in1\_features})`.
        input2 : Tensor
            Second input tensor of shape :math:`(*, \text{in2\_features})`.

        Returns
        -------
        Tensor
            Output tensor of shape :math:`(*, \text{out\_features})`.
        """
        return bilinear(input1, input2, self.weight, self.bias)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(_LazyModule):
    r"""Linear layer whose input dimension is inferred on the first forward call.

    ``LazyLinear`` defers weight allocation until it receives its first input
    tensor.  At that point it reads ``x.shape[-1]`` to determine
    ``in_features``, allocates and initializes ``weight`` and ``bias``, and
    then performs the standard affine transformation

    .. math::

        \mathbf{y} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    All subsequent calls behave identically to :class:`Linear`.

    Parameters
    ----------
    out_features : int
        Dimensionality of each output sample (:math:`d_{\text{out}}`).
    bias : bool, optional
        If ``True`` (default) a learnable bias is added to the output.
    device : DeviceLike, optional
        Device on which the parameters will be allocated once materialized.
    dtype : DTypeLike, optional
        Dtype for the materialized parameters.

    Attributes
    ----------
    weight : Parameter or None
        ``None`` before the first forward call.  After materialization,
        a ``Parameter`` of shape ``(out_features, in_features)`` initialized
        with Kaiming uniform.
    bias : Parameter or None
        ``None`` before the first forward call (and permanently ``None`` when
        ``bias=False``).  After materialization, a ``Parameter`` of shape
        ``(out_features,)`` initialized with uniform fan-in bounds.
    in_features : int or None
        ``None`` until the layer is materialized.  Afterwards stores the
        inferred input dimensionality.
    out_features : int
        The output dimensionality supplied at construction time.

    Notes
    -----
    **When to prefer** ``LazyLinear`` **over** ``Linear``:

    * The input width is only known at runtime (e.g. it depends on a
      preceding convolutional feature extractor whose spatial size varies
      with the input image resolution).
    * You want to prototype model architectures without tracking every
      intermediate feature dimension by hand.

    **State-dict loading** — If :meth:`load_state_dict` is called while the
    layer is still uninitialized, the implementation reads the saved weight
    shape, materializes the parameters to the correct size, and then
    proceeds with the standard copy.  This means a serialized
    :class:`LazyLinear` checkpoint can be restored even without a forward
    pass.

    **Important**: once materialized, the layer behaves *exactly* like a
    :class:`Linear` with the same ``in_features``.  There is no runtime
    overhead after the first call.

    Examples
    --------
    Infer input size from actual data:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> m = nn.LazyLinear(64)
    >>> m.in_features is None          # the shape is not known yet ...
    True
    >>> m.weight.shape                 # ... but the object already exists
    (0,)
    >>> x = lucid.randn(4, 128)
    >>> y = m(x)              # triggers materialization
    >>> m.in_features
    128
    >>> y.shape
    (4, 64)

    Works with arbitrary leading batch dimensions:

    >>> m2 = nn.LazyLinear(32)
    >>> x2 = lucid.randn(2, 10, 256)
    >>> m2(x2).shape
    (2, 10, 32)

    Restore from a checkpoint without running a forward pass first:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> # Suppose we saved a trained LazyLinear that had in_features=512.
    >>> src = nn.Linear(512, 64)
    >>> ckpt = src.state_dict()
    >>> lazy = nn.LazyLinear(64)
    >>> lazy.in_features is None
    True
    >>> _ = lazy.load_state_dict(ckpt)  # materializes to (64, 512) from ckpt shape
    >>> lazy.in_features
    512
    """

    weight: Parameter
    bias: Parameter | None

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LazyLinear module. See the class docstring for parameter semantics."""
        super().__init__()
        self.out_features = out_features
        self.in_features: int | None = None
        self._has_bias = bias
        self._device = device
        self._dtype = dtype
        # Placeholders rather than ``None``: an optimizer built before the
        # first forward has to be given objects that survive being filled
        # in, or the layer never trains.  ``bias=False`` still registers
        # ``None`` — a parameter that will never exist is not a pending
        # one, and conflating the two makes every check downstream wrong.
        self.register_parameter("weight", self._placeholder())
        self.register_parameter("bias", self._placeholder() if bias else None)

    def _initialize(self, in_features: int) -> None:
        """Internal helper for the LazyLinear module."""
        self.in_features = in_features
        weight = Parameter(
            empty(self.out_features, in_features, dtype=self._dtype, device=self._device)
        )
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        # ``materialize`` and not assignment: the placeholder object is
        # what anything that read ``parameters()`` early is holding.
        _fill(self, "weight", weight)

        if self._has_bias:
            bound = 1.0 / math.sqrt(in_features)
            bias = Parameter(empty(self.out_features, dtype=self._dtype, device=self._device))
            init.uniform_(bias, -bound, bound)
            _fill(self, "bias", bias)
        else:
            self.bias = None

    @override
    def _load_from_state_dict(
        self,
        state_dict: StateDict,
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Size a still-lazy layer from the checkpoint's weight, then load it."""
        weight = state_dict.get(f"{prefix}weight")
        if self.in_features is None and weight is not None:
            if len(weight.shape) != 2:
                error_msgs.append(
                    f"LazyLinear expected 2-D weight at '{prefix}weight', "
                    f"got {tuple(weight.shape)}"
                )
                return
            if int(weight.shape[0]) != self.out_features:
                error_msgs.append(
                    f"LazyLinear out_features mismatch at '{prefix}weight': "
                    f"expected {self.out_features}, got {int(weight.shape[0])}"
                )
                return
            # Made where the layer was asked for, else where the checkpoint is;
            # ``_initialize`` fills the placeholders an optimizer may hold.
            self._dtype = self._dtype or weight.dtype
            self._device = self._device or weight.device
            self._initialize(int(weight.shape[1]))
        # Delegate the actual copy / shape-check to the default loader.
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

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the linear transformation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(*, \text{in\_features})`.

        Returns
        -------
        Tensor
            Output tensor of shape :math:`(*, \text{out\_features})`.
        """
        # ``in_features`` and not ``weight is None``: the weight is now a
        # placeholder object from construction, so its absence no longer
        # signals "not yet built".  The recorded input size does.
        if self.in_features is None:
            self._initialize(input.shape[-1])
        return linear(input, self.weight, self.bias)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._has_bias}"
        )


class NoisyLinear(Module):
    r"""A linear layer whose weights carry learnable exploration noise.

    Parameters
    ----------
    in_features, out_features : int
        Widths, as :class:`Linear`.
    bias : bool, default=True
        Whether to carry a noisy bias.
    sigma_zero : float, default=0.5
        :math:`\sigma_0`, the constant the noise scales are initialised
        from.  The paper's value, and it notes it was not tuned.
    device, dtype : optional
        As elsewhere.

    Attributes
    ----------
    weight_mu, weight_sigma : Parameter
        ``(out_features, in_features)``.
    bias_mu, bias_sigma : Parameter or None
        ``(out_features,)``.
    epsilon_in, epsilon_out : Tensor
        The current noise sample, held as buffers so that a forward pass
        — and, for an agent, a whole episode — sees one fixed network.
        Refreshed by :meth:`resample`.

    Raises
    ------
    ValueError
        If ``sigma_zero`` is not positive.

    Notes
    -----
    Reference: Fortunato et al., *"Noisy Networks for Exploration"*,
    ICLR, 2018 (arXiv:1706.10295).

    .. math::

        y = (\mu^w + \sigma^w \odot \varepsilon^w)\,x
            + (\mu^b + \sigma^b \odot \varepsilon^b)

    with factorised noise, so the layer draws ``in_features +
    out_features`` random numbers rather than their product.

    Initialised as the paper's factorised case specifies: "each element
    :math:`\mu_{i,j}` was initialised by a sample from an independent
    uniform distribution :math:`U[-1/\sqrt{p}, +1/\sqrt{p}]` and each
    element :math:`\sigma_{i,j}` was initialised to a constant
    :math:`\sigma_0/\sqrt{p}`", with :math:`p` the input width.

    **The noise is a buffer, not a per-call draw.**  Redrawing inside
    ``forward`` would make two calls on the same input two different
    networks, which breaks both the Monte-Carlo gradient the paper
    derives — one sample per optimisation step, its equation 13 — and
    any agent that needs a fixed policy for an episode.  Call
    :meth:`resample` when a new sample is wanted.  In ``eval()`` the
    noise is not used at all, so a deterministic policy is just the mean
    network.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> layer = nn.NoisyLinear(4, 3)
    >>> layer(lucid.zeros((2, 4))).shape
    (2, 3)
    """

    epsilon_in: Tensor
    epsilon_out: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sigma_zero: float = 0.5,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        if sigma_zero <= 0.0:
            raise ValueError(f"sigma_zero must be positive, got {sigma_zero}")
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero

        self.weight_mu = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        self.weight_sigma = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias_mu: Parameter | None = Parameter(
                empty(out_features, dtype=dtype, device=device)
            )
            self.bias_sigma: Parameter | None = Parameter(
                empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.bias_mu = None
            self.bias_sigma = None

        self.register_buffer("epsilon_in", empty(in_features, dtype=dtype, device=device))
        self.register_buffer("epsilon_out", empty(out_features, dtype=dtype, device=device))
        self.reset_parameters()
        self.resample()

    def reset_parameters(self) -> None:
        r"""The paper's factorised initialisation.

        ``mu ~ U[-1/sqrt(p), 1/sqrt(p)]`` and ``sigma = sigma_zero /
        sqrt(p)``, with ``p`` the input width.
        """
        bound = 1.0 / math.sqrt(self.in_features)
        scale = self.sigma_zero / math.sqrt(self.in_features)
        init.uniform_(self.weight_mu, -bound, bound)
        init.constant_(self.weight_sigma, scale)
        if self.bias_mu is not None and self.bias_sigma is not None:
            init.uniform_(self.bias_mu, -bound, bound)
            init.constant_(self.bias_sigma, scale)

    @property
    def noise_in(self) -> Tensor:
        """The input noise sample, narrowed from the buffer registry."""
        return self.epsilon_in

    @property
    def noise_out(self) -> Tensor:
        """The output noise sample, narrowed from the buffer registry."""
        return self.epsilon_out

    def resample(self) -> None:
        """Draw a fresh noise sample — one per optimisation step, or episode.

        Notes
        -----
        Writes the buffers in place with ``[:]``.  Assigning ``.data``
        writes to a copy and silently does nothing, which is the mistake
        this codebase has made before.
        """
        import lucid as _l

        noise_in, noise_out = self.noise_in, self.noise_out
        with _l.no_grad():
            noise_in[:] = _l.randn(
                (self.in_features,), device=noise_in.device, dtype=noise_in.dtype
            )
            noise_out[:] = _l.randn(
                (self.out_features,), device=noise_out.device, dtype=noise_out.dtype
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the layer — noisy while training, the mean network in eval.

        Parameters
        ----------
        x : Tensor
            ``(*, in_features)``.

        Returns
        -------
        Tensor
            ``(*, out_features)``.
        """
        if not self.training:
            return linear(x, self.weight_mu, self.bias_mu)
        return noisy_linear(
            x,
            self.weight_mu,
            self.weight_sigma,
            self.bias_mu,
            self.bias_sigma,
            self.noise_in,
            self.noise_out,
        )
