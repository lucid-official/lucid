r"""
Random tensor creation: rand, randn, randint, bernoulli, normal, manual_seed,
the ``*_like`` forms, and the draws behind ``Tensor.normal_`` / ``uniform_``.
"""

import math
import operator
import os
from collections.abc import Callable
from typing import TYPE_CHECKING
from lucid._ctx import check_random_allowed as _check_random

from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _wrap
from lucid._factories._build import _leaf
from lucid._factories._generator import (
    Generator as Generator,
    _key,
    _seed_value,
    _set_counter,
    _state_tensor,
    _unpack_state,
)
from lucid._dtype import int64
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _default_stream() -> _C_engine.Generator:
    r"""The one stream every draw that names no generator reads.

    It is the engine's process-wide generator, the one its ops fall back
    to when handed ``None``.  :func:`manual_seed` and :func:`set_rng_state`
    rewrite it in place rather than install a generator of their own.  An
    installed one is read only by the callers that remember to fetch it:
    ``rand`` / ``randn`` did, while ``rand_like``, ``nn.init`` and the
    dropout masks passed ``None`` and kept reading this one, seeded alike,
    so the two streams repeated each other's numbers.
    """
    return _C_engine.default_generator()


def manual_seed(seed: int) -> None:
    r"""Set the seed of the default Philox counter-based random number generator.

    The Lucid RNG uses a **Philox-4×32** counter-based PRNG (Salmon et al.,
    2011).  Unlike stateful RNGs (e.g. Mersenne Twister), Philox is a
    keyed bijection: given a 64-bit key $k$ (the seed) and a 64-bit counter
    $c$, it produces a deterministic 128-bit output block via 10 rounds of
    the Philox permutation:

    .. math::

        (r_0, r_1, r_2, r_3) = \text{Philox}_{4 \times 32}(k,\, c)

    Setting the seed to a fixed value $k_0$ fully determines the entire
    subsequent sampling stream, enabling **reproducible** experiments.

    Parameters
    ----------
    seed : int
        A 64-bit integer seed, in :math:`[-2^{63}, 2^{64})`; a negative one
        is read modulo :math:`2^{64}`.  Anything ``int()`` takes is read
        through it first — ``True``, ``2.0`` — as the reference reads it;
        :meth:`Generator.manual_seed` is strict.  The same seed always
        produces the same sequence of random values on the same hardware.

    Notes
    -----
    Calling ``manual_seed`` resets the counter to $0$ in addition to
    updating the key $k$.  Two calls with the same ``seed`` produce
    identical streams regardless of how many samples were drawn in between.

    There is one default stream, and it is device-independent.  Every
    draw that names no ``generator`` reads and advances it, whatever
    spells the draw: the factories (``rand``, ``randn_like``, ``randint``,
    ``bernoulli``, ``randperm``, …), the ``nn.init`` initialisers, the
    dropout masks and the samplers built on them, on CPU and Metal alike.
    A seed therefore fixes their *interleaving*, not each spelling's own
    sequence: ``randn`` then ``randn_like`` gives two different draws.
    :func:`get_rng_state` / :func:`set_rng_state` read and write the same
    stream, and :func:`lucid.metal.manual_seed` seeds it too.  A
    :class:`Generator` passed as ``generator=`` is a stream of its own and
    leaves this one where it was.

    For multi-process reproducibility, seed each worker with a distinct
    derived seed, e.g. ``manual_seed(base_seed + worker_id)``, to avoid
    correlated streams across processes.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(42)
    >>> a = lucid.rand(3)
    >>> lucid.manual_seed(42)
    >>> b = lucid.rand(3)
    >>> (a - b).abs().max().item()
    0.0

    Every spelling draws from the one stream, so they do not repeat each
    other:

    >>> lucid.manual_seed(42)
    >>> x = lucid.randn(3)
    >>> y = lucid.randn_like(x)
    >>> bool((x == y).all())
    False
    """
    _default_stream().set_seed(_seed_value(int(seed), "manual_seed"))


def seed() -> int:
    r"""Seed the default generator from the OS cryptographic entropy source.

    Reads 8 bytes from the operating system's secure random pool
    (``/dev/urandom`` on macOS / Linux, ``CryptGenRandom`` on Windows)
    and uses them as the 63-bit Philox seed key $k$:

    .. math::

        k = \text{OS\_urandom}(8) \;\bmod\; 2^{63}

    The 63-bit mask ensures the seed fits comfortably in a signed
    ``int64``, which is required by the C++ ``Generator`` API.
    After seeding, the RNG counter $c$ is reset to $0$.

    Returns
    -------
    int
        The 63-bit seed value that was applied, so the caller can
        log or checkpoint it for later reproducibility.

    Notes
    -----
    Unlike `manual_seed`, successive calls to `seed` produce different
    streams with overwhelming probability ($2^{63}$ possible seeds).
    Use this at the start of non-reproducible training runs and log
    the return value so you can reproduce the exact run later with
    ``lucid.manual_seed(returned_value)`` if needed.

    The OS entropy pool on Apple Silicon is hardware-backed (the Secure
    Enclave contributes entropy), so the seeds are cryptographically
    unpredictable.

    Examples
    --------
    >>> import lucid
    >>> s = lucid.seed()
    >>> isinstance(s, int) and 0 <= s < 2**63
    True

    Log and reproduce:

    >>> s = lucid.seed()
    >>> print(f"Run seed: {s}")
    Run seed: ...
    >>> # Later, to reproduce:
    >>> lucid.manual_seed(s)
    """
    s: int = int.from_bytes(os.urandom(8), "little", signed=False)
    # Mask to the 63-bit non-negative range so the int round-trips through
    # signed APIs cleanly.
    s &= 0x7FFF_FFFF_FFFF_FFFF
    manual_seed(s)
    return s


def initial_seed() -> int:
    r"""Return the seed (key) of the default generator as it was last set.

    Reads the 64-bit Philox key $k$ from the default ``Generator`` object
    without advancing its counter.  The returned value is the seed that
    was last applied via `manual_seed` or `seed`.

    Returns
    -------
    int
        Current seed of the default generator.  This is the Philox key
        $k \in [0,\, 2^{63})$ — a non-negative 63-bit integer.

    Notes
    -----
    ``initial_seed`` does **not** reset or alter the generator state;
    it is a pure read.  Together with ``get_rng_state``, it forms the
    minimal information needed to checkpoint and restore the full
    sampling stream:

    .. math::

        \text{state} = (k,\, c)
        \;\Longrightarrow\;
        \text{next sample} = \text{Philox}(k,\, c)

    where $c$ is the counter returned by ``get_rng_state``.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(1337)
    >>> lucid.initial_seed()
    1337
    """
    return _key(_default_stream())


def get_rng_state() -> Tensor:
    r"""Snapshot the full state of the default random generator.

    Returns an ``int64`` tensor of length 2, ``[seed, counter]``,
    that completely characterises the default Philox generator.

    The Philox-4×32 PRNG is a **stateless counter-based** design: the
    entire infinite sample stream is determined by the two-tuple

    .. math::

        \text{state} = (k,\; c)
        \quad k \in [0,\, 2^{63}),\; c \in [0,\, 2^{64})

    where $k$ is the seed key and $c$ is the call counter.  After each
    draw of a 128-bit output block, $c$ is incremented by 1.  The next
    sample is always $\text{Philox}(k, c)$, so saving $(k, c)$ is a
    lossless checkpoint of the generator.

    Returns
    -------
    Tensor
        1-D ``int64`` tensor ``[seed, counter]`` of shape ``(2,)``.
        Pass this directly to `set_rng_state` to restore the stream.

    Notes
    -----
    Unlike Mersenne Twister (which requires 624 × 32-bit words of state),
    Philox's full state is just 128 bits.  This makes checkpointing
    essentially free and enables **exact reproducibility** of training
    runs when saved alongside model weights.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> _ = lucid.rand(10)           # advance counter by drawing 10 values
    >>> state = lucid.get_rng_state()
    >>> a = lucid.rand(5)
    >>> lucid.set_rng_state(state)   # rewind
    >>> b = lucid.rand(5)
    >>> (a - b).abs().max().item()
    0.0
    """
    return _state_tensor(_default_stream())


def set_rng_state(new_state: Tensor) -> None:
    r"""Restore the default random generator to a previously captured state.

    Rewinds (or fast-forwards) the Philox generator to the exact
    ``(seed, counter)`` pair encoded in ``new_state``, so that the next
    sample call produces exactly the same value as it did immediately
    after the `get_rng_state` call that created ``new_state``.

    Formally, after ``set_rng_state(state)`` the generator satisfies:

    .. math::

        k \leftarrow \texttt{state}[0], \quad
        c \leftarrow \texttt{state}[1]

    and the next draw returns $\text{Philox}(k, c)$.

    Parameters
    ----------
    new_state : Tensor
        Length-2 ``int64`` tensor ``[seed, counter]`` as returned by
        `get_rng_state`.  Passing a tensor of any other shape or dtype
        raises ``ValueError``.

    Raises
    ------
    ValueError
        If ``new_state`` does not have exactly 2 elements.

    Notes
    -----
    ``set_rng_state`` + `get_rng_state` together provide a lightweight
    **fork–join** pattern for reproducible stochastic computation:

    .. math::

        s^* = \text{get\_rng\_state}()
        \;\to\;
        \text{sample branch A}
        \;\to\;
        \text{set\_rng\_state}(s^*)
        \;\to\;
        \text{sample branch B}

    Both branches then draw from the same sub-stream, enabling
    controlled ablations where only the model changes, not the data
    augmentation noise.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(99)
    >>> state = lucid.get_rng_state()
    >>> x = lucid.randn(4)
    >>> lucid.set_rng_state(state)
    >>> y = lucid.randn(4)
    >>> (x - y).abs().max().item()
    0.0
    """
    s_seed, s_counter = _unpack_state(new_state, "set_rng_state")
    g = _default_stream()
    g.set_seed(s_seed)
    _set_counter(g, s_counter)


def rand(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Return a tensor of samples drawn from the continuous uniform distribution.

    Each element is drawn independently from $U[0, 1)$, the uniform
    distribution on the half-open unit interval:

    .. math::

        X_i \sim U[0, 1), \quad
        f_X(x) = \begin{cases} 1 & 0 \le x < 1 \\ 0 & \text{otherwise} \end{cases}

    The distribution has mean $\mathbb{E}[X] = \tfrac{1}{2}$ and variance
    $\operatorname{Var}[X] = \tfrac{1}{12}$.

    Parameters
    ----------
    *size : int or tuple[int, ...]
        Shape of the output tensor.  Accepts varargs ``rand(2, 3)`` or a
        single tuple ``rand((2, 3))``.
    dtype : lucid.dtype, optional
        Floating-point data type.  Defaults to the global default
        (``lucid.float32``).  Integer dtypes are not supported.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.
    generator : lucid._C.engine.Generator, optional
        Explicit generator to use.  When ``None`` (default), the global
        default generator is used.

    Returns
    -------
    Tensor
        Tensor of shape ``size`` with values in $[0, 1)$.

    Notes
    -----
    Internally the engine applies the Philox bijection and converts the
    raw 32-bit output to a float via the standard IEEE 754 mantissa trick:

    .. math::

        x = (u \;\&\; \texttt{0x007FFFFF}) \;\big|\; \texttt{0x3F800000}
        \;\text{(reinterpreted as float32)} \;-\; 1.0

    yielding $2^{23}$ equally spaced values in $[0, 1)$ with spacing
    $2^{-23} \approx 1.19 \times 10^{-7}$.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> x = lucid.rand(3, 4)
    >>> x.shape
    (3, 4)
    >>> float(x.min()) >= 0.0 and float(x.max()) < 1.0
    True

    Monte-Carlo estimate of $\pi$:

    >>> lucid.manual_seed(0)
    >>> n = 1_000_000
    >>> pts = lucid.rand(n, 2)
    >>> inside = ((pts ** 2).sum(dim=-1) < 1.0).float().mean()
    >>> abs(float(inside) * 4 - 3.1416) < 0.01
    True
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = _size_to_list(*size)
    impl = _C_engine.rand(shape, _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def randn(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Return a tensor of samples drawn from the standard normal distribution.

    Each element is drawn independently from $\mathcal{N}(0, 1)$, the
    standard Gaussian with zero mean and unit variance:

    .. math::

        X_i \sim \mathcal{N}(0, 1), \quad
        f_X(x) = \frac{1}{\sqrt{2\pi}}\, e^{-x^2 / 2}

    The distribution has mean $\mathbb{E}[X] = 0$, variance
    $\operatorname{Var}[X] = 1$, and all odd central moments zero.

    Parameters
    ----------
    *size : int or tuple[int, ...]
        Shape of the output tensor.
    dtype : lucid.dtype, optional
        Floating-point data type.  Defaults to the global default.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the global default generator.

    Returns
    -------
    Tensor
        Tensor of shape ``size`` with $\mathcal{N}(0, 1)$ samples.

    Notes
    -----
    The engine uses the **Box–Muller transform** to convert pairs of
    uniform samples $(u_1, u_2) \sim U(0,1)^2$ into standard normals:

    .. math::

        z_1 = \sqrt{-2\ln u_1}\,\cos(2\pi u_2), \quad
        z_2 = \sqrt{-2\ln u_1}\,\sin(2\pi u_2)

    This is exact (not approximate) and is highly efficient on SIMD
    hardware because the two outputs $z_1, z_2$ can be produced
    simultaneously from one pair of uniform draws.

    **Xavier / Glorot initialisation** (Glorot & Bengio, 2010) uses
    $\mathcal{N}(0, \sigma^2)$ with

    .. math::

        \sigma = \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}

    which can be obtained as ``randn(fan_in, fan_out) * sigma``.

    **He / Kaiming initialisation** (He et al., 2015) for ReLU networks
    uses $\sigma = \sqrt{2 / n_\text{in}}$.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> x = lucid.randn(1000)
    >>> abs(float(x.mean())) < 0.1       # ≈ 0
    True
    >>> abs(float(x.std()) - 1.0) < 0.1  # ≈ 1
    True

    Kaiming initialisation for a linear layer:

    >>> fan_in = 256
    >>> w = lucid.randn(fan_in, 512) * (2.0 / fan_in) ** 0.5
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = _size_to_list(*size)
    impl = _C_engine.randn(shape, _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def randint(
    low: int,
    high: int,
    size: list[int] | tuple[int, ...],
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Return a tensor of samples drawn from the discrete uniform distribution.

    Each element is drawn independently and uniformly from the integer set
    $\{l, l+1, \ldots, h-1\}$, i.e. the discrete uniform distribution:

    .. math::

        X_i \sim \mathcal{U}\{l,\, h-1\}, \quad
        P(X_i = k) = \frac{1}{h - l}
        \quad \forall\; k \in \{l, \ldots, h-1\}

    The distribution has mean $\mathbb{E}[X] = \tfrac{l + h - 1}{2}$ and
    variance $\operatorname{Var}[X] = \tfrac{(h-l)^2 - 1}{12}$.

    Parameters
    ----------
    low : int
        Lower bound of the interval (inclusive).
    high : int
        Upper bound of the interval (exclusive).  Must satisfy
        ``high > low``.
    size : list[int] or tuple[int, ...]
        Shape of the output tensor.  Unlike `rand` / `randn`, ``size``
        is a single required keyword-positional argument, not varargs.
    dtype : lucid.dtype, optional
        Integer data type.  Defaults to ``lucid.int64``.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.  Gradients through
        integer tensors are always zero; this flag is accepted for
        compatibility with downstream wrappers.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the global default generator.

    Returns
    -------
    Tensor
        Integer tensor of shape ``size`` with values in $[l, h)$.

    Notes
    -----
    The engine generates the integers via

    .. math::

        X_i = l + \lfloor U_i \cdot (h - l) \rfloor,
        \quad U_i \sim U[0, 1)

    which gives an exact discrete uniform distribution with no
    acceptance–rejection overhead.

    Common uses include:

    * Sampling mini-batch indices for data loaders
    * Generating random class labels for testing
    * Stochastic depth / drop-path masks

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> x = lucid.randint(0, 10, (5,))
    >>> x.shape, x.dtype
    ((5,), lucid.int64)
    >>> bool(((x >= 0) & (x < 10)).all())
    True

    Shuffle indices for a dataset of 1024 examples:

    >>> idx = lucid.randint(0, 1024, (64,))  # sample 64 indices

    Random binary mask:

    >>> mask = lucid.randint(0, 2, (3, 4))  # values in {0, 1}
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else int64, device
    )
    shape = list(size)
    impl = _C_engine.randint(shape, low, high, _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def bernoulli(
    input: Tensor,
    p: float | None = None,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Draw binary random numbers (``0`` or ``1``) from a Bernoulli distribution.

    The ``input`` tensor holds probabilities in :math:`[0, 1]` to be used for
    drawing the binary random number, unless scalar ``p`` is supplied, in which
    case ``input`` provides only the output shape, dtype, and device while each
    element is drawn from :math:`\text{Bernoulli}(p)`.

    .. math::

        \text{out}_i \sim \text{Bernoulli}(p_i), \quad
        P(\text{out}_i = 1) = p_i, \quad P(\text{out}_i = 0) = 1 - p_i

    Parameters
    ----------
    input : Tensor
        Probability values for the Bernoulli distribution when ``p`` is ``None``,
        or shape/dtype/device template when ``p`` is given.
    p : float, optional
        If provided, uniform success probability in :math:`[0, 1]` used for all
        elements.
    generator : lucid._C.engine.Generator, optional
        Explicit generator. Defaults to the global default.

    Returns
    -------
    Tensor
        Tensor of the same shape, dtype, and device as ``input`` with values in
        :math:`\{0, 1\}`.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> probs = lucid.full((10_000,), 0.3)
    >>> x = lucid.bernoulli(probs)
    >>> sorted(set(x.tolist()))                      # only zeros and ones
    [0.0, 1.0]
    >>> abs(float(x.mean()) - 0.3) < 0.02            # ≈ 30 % ones
    True
    """
    from lucid import _promotion

    _check_random()
    impl = _promotion.try_unwrap(input)
    if impl is None:
        raise TypeError(
            f"bernoulli(): argument 'input' (position 1) must be Tensor, "
            f"not {type(input).__name__}"
        )
    if input.is_complex():
        raise _C_engine.NotImplementedError(
            f'"bernoulli" not implemented for {input.dtype}'
        )
    dt, dev, gen = impl.dtype, impl.device, generator
    if p is not None:
        p_num = _promotion.as_number(p)
        if isinstance(p_num, bool) or not isinstance(p_num, (int, float)):
            raise TypeError(
                f"bernoulli(): argument 'p' must be float, not {type(p).__name__}"
            )
        pf = float(p_num)
        if not 0.0 <= pf <= 1.0:
            raise _C_engine.InvalidArgument(
                f"bernoulli_ expects p to be in [0, 1], but got p={pf}"
            )
        draw_dt = dt if input.is_floating_point() else _C_engine.Dtype.F32
        impl = _C_engine.bernoulli(list(input.shape), pf, draw_dt, dev, gen)
        if draw_dt != dt:
            impl = _C_engine.astype(impl, dt)
        return _wrap(_leaf(impl, False))
    if not input.is_floating_point():
        raise _C_engine.NotImplementedError(
            f'"bernoulli_tensor" not implemented for {input.dtype}'
        )
    if input.numel() > 0 and not hasattr(input._impl, "_in_dim"):
        from lucid._engine_scope import untraced

        with untraced():
            ok = bool(((input >= 0.0) & (input <= 1.0)).all().item())
        if not ok:
            raise _C_engine.InvalidArgument(
                "Expected p_in >= 0 && p_in <= 1 to be true, but got false."
            )
    u = _wrap(_C_engine.uniform(list(input.shape), 0.0, 1.0, dt, dev, gen))
    return (u < input).to(dtype=input.dtype).detach()


def normal(
    mean: float = 0.0,
    std: float = 1.0,
    *,
    size: list[int] | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Return a tensor of samples drawn from a parametric normal distribution.

    Each element is drawn independently from $\mathcal{N}(\mu, \sigma^2)$:

    .. math::

        X_i \sim \mathcal{N}(\mu,\, \sigma^2), \quad
        f_X(x) = \frac{1}{\sigma\sqrt{2\pi}}\,
                 \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    with mean $\mathbb{E}[X_i] = \mu$ and standard deviation
    $\sqrt{\operatorname{Var}[X_i]} = \sigma$.

    Parameters
    ----------
    mean : float, optional
        Mean $\mu$ of the distribution.  Default: ``0.0``.
    std : float, optional
        Standard deviation $\sigma > 0$.  Default: ``1.0``.
    size : list[int] or tuple[int, ...]
        Shape of the output tensor.  Required (keyword-only).
    dtype : lucid.dtype, optional
        Floating-point data type.  Defaults to the global default.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the global default.

    Returns
    -------
    Tensor
        Tensor of shape ``size`` with $\mathcal{N}(\mu, \sigma^2)$ samples.

    Notes
    -----
    This is a convenience wrapper around `randn` with an affine shift:

    .. math::

        X = \mu + \sigma \cdot Z, \quad Z \sim \mathcal{N}(0, 1)

    The output is therefore equivalent to
    ``mean + std * lucid.randn(*size, ...)``.

    **Weight initialisation** recipes commonly parameterise the normal
    distribution directly:

    * **LeCun normal** (LeCun et al., 1998):
      $\sigma = \sqrt{1 / n_\text{in}}$
    * **Glorot normal** (Glorot & Bengio, 2010):
      $\sigma = \sqrt{2 / (n_\text{in} + n_\text{out})}$
    * **He normal** (He et al., 2015):
      $\sigma = \sqrt{2 / n_\text{in}}$

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> x = lucid.normal(mean=5.0, std=2.0, size=(1000,))
    >>> abs(float(x.mean()) - 5.0) < 0.2
    True
    >>> abs(float(x.std()) - 2.0) < 0.2
    True

    He-normal initialisation for a Conv2d kernel:

    >>> fan_in = 3 * 3 * 64          # kernel_h × kernel_w × C_in
    >>> w = lucid.normal(0.0, (2.0 / fan_in) ** 0.5, size=(128, 64, 3, 3))
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    shape = list(size)
    impl = _C_engine.normal(shape, mean, std, _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def rand_like(
    input: Tensor,
    *,
    generator: _C_engine.Generator | None = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a $U[0,1)$ random tensor with the same shape, dtype, and device as ``input``.

    Equivalent to ``lucid.rand(t.shape, dtype=t.dtype, device=t.device)``,
    but infers metadata from an existing tensor so the caller need not
    repeat shape and type information.

    Each element is drawn independently from

    .. math::

        X_i \sim U[0, 1)

    Parameters
    ----------
    input : Tensor
        Reference tensor.  Its ``shape``, ``dtype``, and ``device`` are
        inherited by the output unless overridden.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the default stream that
        :func:`manual_seed` controls — the one :func:`rand` reads.
    dtype : lucid.dtype, optional
        Override the data type.  Defaults to ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  Defaults to ``t.device``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.

    Returns
    -------
    Tensor
        Uniform random tensor shaped like ``input``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros(3, 4, device="metal")
    >>> noise = lucid.rand_like(x)
    >>> noise.shape, noise.device
    ((3, 4), device('metal'))

    Notes
    -----
    The returned tensor mirrors ``input``'s ``shape``, ``dtype``, and
    ``device`` exactly unless overridden via the corresponding keyword
    arguments.  Sampling follows the same dtype-default rule as
    :func:`lucid.rand`: only floating-point dtypes are supported.  Calling
    this on an integer tensor without overriding ``dtype`` raises — use
    :func:`lucid.randint` with an explicit shape for integer noise.
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else input.dtype,
        device if device is not None else input.device,
    )
    impl = _C_engine.rand(list(input.shape), _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def randn_like(
    input: Tensor,
    *,
    generator: _C_engine.Generator | None = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a $\mathcal{N}(0,1)$ random tensor with the same shape, dtype, and device as ``input``.

    Equivalent to ``lucid.randn(t.shape, dtype=t.dtype, device=t.device)``,
    inferring all metadata from an existing tensor.

    Each element is drawn independently from the standard normal:

    .. math::

        X_i \sim \mathcal{N}(0, 1), \quad
        f_X(x) = \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}

    Parameters
    ----------
    input : Tensor
        Reference tensor whose ``shape``, ``dtype``, and ``device`` are
        inherited unless overridden.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the default stream that
        :func:`manual_seed` controls — the one :func:`randn` reads.
    dtype : lucid.dtype, optional
        Override the data type.  Defaults to ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  Defaults to ``t.device``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.

    Returns
    -------
    Tensor
        Standard-normal random tensor shaped like ``input``.

    Notes
    -----
    A common pattern in **noise injection** regularisation is to add
    scaled Gaussian noise to activations or weights during training:

    .. math::

        \tilde{x} = x + \epsilon \cdot \mathcal{N}(0, I),
        \quad \epsilon \ll 1

    ``randn_like`` makes this concise and device-safe:

    .. code-block:: python

        x_noisy = x + noise_scale * lucid.randn_like(x)

    Examples
    --------
    >>> import lucid
    >>> w = lucid.zeros(256, 512, dtype=lucid.float16, device="metal")
    >>> noise = lucid.randn_like(w)
    >>> noise.shape, noise.dtype, noise.device
    ((256, 512), lucid.float16, device('metal'))
    """
    _check_random()
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else input.dtype,
        device if device is not None else input.device,
    )
    impl = _C_engine.randn(list(input.shape), _dt, _dev, generator)
    return _wrap(_leaf(impl, requires_grad))


def randint_like(
    input: Tensor,
    low: int = 0,
    high: int | None = None,
    *,
    generator: _C_engine.Generator | None = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return random integers in $[\text{low}, \text{high})$ shaped like ``input``.

    Each element is drawn independently and uniformly from
    $\{l, l+1, \ldots, h-1\}$, as :func:`randint` draws them, in ``input``'s
    shape, dtype and device unless overridden.  Called with one bound,
    ``randint_like(x, 10)``, the bound is ``high`` and ``low`` is ``0``.

    Parameters
    ----------
    input : Tensor
        Reference tensor whose ``shape``, ``dtype`` and ``device`` the
        result takes.
    low : int, optional
        Lowest integer drawn (inclusive).  Default ``0``; the one bound
        when ``high`` is not given.
    high : int
        One above the highest integer drawn (exclusive).
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the default stream that
        :func:`manual_seed` controls — the one :func:`randint` reads.
    dtype : lucid.dtype, optional
        Override the data type.  Defaults to ``input.dtype``: a floating
        ``input`` gets floating values that are whole numbers.
    device : str or lucid.device, optional
        Override the device.  Defaults to ``input.device``.
    requires_grad : bool, optional
        Enable autograd tracking — floating dtypes only.  Default ``False``.

    Returns
    -------
    Tensor
        Tensor of ``input``'s shape with whole-number values in
        $[\text{low}, \text{high})$.

    Raises
    ------
    TypeError
        A bound is not an integer, or no bound is given.
    lucid._C.engine.InvalidArgument
        ``low >= high`` (a ``RuntimeError`` and a ``ValueError``).

    Notes
    -----
    The integers are drawn as ``int64`` and cast to the result's dtype, so
    the draw is the same on either device whatever that dtype is.

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> x = lucid.zeros(2, 3)
    >>> r = lucid.randint_like(x, 10)
    >>> r.shape, r.dtype
    ((2, 3), lucid.float32)
    >>> bool(((r >= 0) & (r < 10)).all())
    True
    >>> lucid.randint_like(x, 3, 4, dtype=lucid.int64).tolist()
    [[3, 3, 3], [3, 3, 3]]
    """
    _check_random()
    if high is None:
        low, high = 0, low
    bounds = []
    for what, bound in (("low", low), ("high", high)):
        if isinstance(bound, bool):
            raise TypeError(f"randint_like(): {what} must be an int, got bool")
        try:
            bounds.append(operator.index(bound))
        except TypeError:
            raise TypeError(
                f"randint_like(): {what} must be an int, got {type(bound).__name__}"
            ) from None
    lo, hi = bounds
    if lo >= hi:
        raise _C_engine.InvalidArgument(
            f"randint_like(): low must be less than high, got low={lo} >= high={hi}"
        )
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else input.dtype,
        device if device is not None else input.device,
    )
    impl = _C_engine.randint(list(input.shape), lo, hi, _C_engine.I64, _dev, generator)
    if _dt != _C_engine.I64:
        impl = _C_engine.astype(impl, _dt)
    return _wrap(_leaf(impl, requires_grad))


#: A complex tensor's real dtype: its two parts are drawn in it, pairwise.
_PART_DTYPE = {_C_engine.C64: _C_engine.F32, _C_engine.C128: _C_engine.F64}


def _draw_like(
    input: Tensor,
    name: str,
    draw: Callable[[list[int], _C_engine.Dtype, _C_engine.Device], _C_engine.TensorImpl],
) -> Tensor:
    """``draw``'s samples in ``input``'s shape, dtype and device.

    The draw behind ``Tensor.normal_`` / ``Tensor.uniform_``.  A complex
    ``input`` takes its samples in pairs, as its real and imaginary parts,
    which is how the reference fills one.  An integer or bool ``input``
    has no continuous values to take, and is refused under ``name``.
    """
    _check_random()
    impl = input._impl
    part = _PART_DTYPE.get(impl.dtype)
    if part is not None:
        pairs = draw([*impl.shape, 2], part, impl.device)
        return _wrap(_C_engine.view_as_complex(pairs))
    if impl.dtype not in (
        _C_engine.F16,
        _C_engine.BF16,
        _C_engine.F32,
        _C_engine.F64,
    ):
        raise _C_engine.NotImplementedError(
            f'"{name}" not implemented for {input.dtype}'
        )
    return _wrap(draw(list(impl.shape), impl.dtype, impl.device))


def _normal_like(
    input: Tensor,
    mean: float,
    std: float,
    generator: _C_engine.Generator | None,
) -> Tensor:
    r"""Samples of $\mathcal{N}(\text{mean}, \text{std}^2)$ shaped like ``input``.

    A complex ``input`` draws each part with standard deviation
    $\text{std} / \sqrt{2}$, so the complex value has variance
    $\text{std}^2$; both parts take ``mean``, as the reference's do.
    """
    if not std >= 0.0:
        raise _C_engine.InvalidArgument(f"normal_: std must be >= 0, got {std}")
    scale = std / math.sqrt(2.0) if input.is_complex() else std
    return _draw_like(
        input,
        "normal_",
        lambda shape, dt, dev: _C_engine.normal(shape, mean, scale, dt, dev, generator),
    )


def _uniform_like(
    input: Tensor,
    low: float,
    high: float,
    generator: _C_engine.Generator | None,
) -> Tensor:
    r"""Samples of $U[\text{low}, \text{high})$ shaped like ``input``.

    ``low == high`` is every sample ``low``, as the reference allows; a
    complex ``input`` draws both parts from the interval.
    """
    if not low <= high:
        raise _C_engine.InvalidArgument(
            f"uniform_: expects a [from, to) range, got from={low} > to={high}"
        )

    def draw(
        shape: list[int], dt: _C_engine.Dtype, dev: _C_engine.Device
    ) -> _C_engine.TensorImpl:
        if low == high:
            return _C_engine.full(shape, low, dt, dev)
        return _C_engine.uniform(shape, low, high, dt, dev, generator)

    return _draw_like(input, "uniform_", draw)


from lucid._factories.creation import _size_to_list


def randperm(
    n: int,
    *,
    generator: _C_engine.Generator | None = None,
    dtype: DTypeLike = int64,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a uniformly random permutation of the integers $0, 1, \ldots, n-1$.

    Produces a rank-1 tensor containing a uniformly random element of
    $S_n$, the symmetric group of permutations on $n$ symbols.  Every
    one of the $n!$ orderings is equally likely:

    .. math::

        P(\sigma) = \frac{1}{n!}
        \quad \forall\; \sigma \in S_n

    Parameters
    ----------
    n : int
        Number of elements.  Must be $\geq 0$.  Returns an empty tensor
        when ``n = 0``.
    generator : lucid._C.engine.Generator, optional
        Explicit generator.  Defaults to the global default.
    dtype : lucid.dtype, optional
        Integer data type of the output.  Defaults to ``lucid.int64``.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.

    Returns
    -------
    Tensor
        1-D integer tensor of shape ``(n,)`` containing a permutation of
        $\{0, 1, \ldots, n-1\}$.

    Notes
    -----
    The implementation uses the **lottery-ticket argsort trick**: assign
    each integer $k$ a uniform random key $u_k \sim U[0,1)$ and return
    the argsort of those keys:

    .. math::

        \pi = \operatorname{argsort}(u), \quad u_k \sim U[0,1)^n

    This is equivalent to the Fisher–Yates shuffle in expectation and
    is more cache-friendly on SIMD hardware because the sort can be
    vectorised without branch-dependent swaps.

    ``randperm`` is the standard building block for:

    * **Dataset shuffling** — ``dataset[lucid.randperm(len(dataset))]``
    * **k-fold cross-validation** — split a shuffled index tensor
    * **Contrastive learning** — negative-sample mining via permuted
      indices

    Examples
    --------
    >>> import lucid
    >>> lucid.manual_seed(0)
    >>> sorted(lucid.randperm(5).tolist())   # every index exactly once
    [0, 1, 2, 3, 4]

    Shuffle a dataset of 1024 examples in one line:

    >>> data = lucid.randn(1024, 16)
    >>> idx = lucid.randperm(1024)
    >>> shuffled = data[idx]
    >>> shuffled.shape
    (1024, 16)

    Edge cases:

    >>> lucid.randperm(0).shape
    (0,)
    >>> lucid.randperm(1).tolist()
    [0]
    """
    _check_random()
    if n < 0:
        raise _C_engine.InvalidArgument(
            f"randperm: Trying to create tensor with negative dimension {n}: [{n}]"
        )
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    if n == 0:
        impl = _C_engine.zeros([0], _dt, _dev)
        return _wrap(_leaf(impl, requires_grad))
    keys_impl = _C_engine.randint(
        [n], -(2**63), 2**63 - 1, _C_engine.I64, _dev, generator
    )
    perm_impl = _C_engine.argsort(keys_impl, -1)
    if perm_impl.dtype != _dt:
        perm_impl = _C_engine.astype(perm_impl, _dt)
    return _wrap(_leaf(perm_impl, requires_grad))
