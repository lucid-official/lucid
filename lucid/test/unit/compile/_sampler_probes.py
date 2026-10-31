"""One tiny trained model per traced sampler / custom VJP, and the child that compiles them.

Behind ``test_traced_sampler_compile_budget.py`` (LCD-786).  The parent test
enumerates the targets from the packages -- every ``lucid.distributions``
class with ``rsample``, the top-level random ops, every ``Function`` listed in
``TRACED_BACKWARD`` -- and hands their ids to this module run as a script in
a fresh process, where the first ``make_step`` compile of each is timed.  The
compile is native and cannot be interrupted, so it must not run in the test
process.

A target is a ``draw(w) -> Tensor`` over the model's one trainable scalar
``w`` (value 1.0): everything the target builds is multiplied by ``w`` or
built from it, so the sampler sits on the path from a parameter to the loss
and the compiler cannot fold it away.
"""

import copy
import inspect
import sys
import time
from collections.abc import Callable

import lucid
import lucid.distributions as D
import lucid.linalg as LA
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.special as S
from lucid._tensor import Tensor
from lucid.test.audit._axes_data import _EXCLUSIVE, _construct

DEVICE = "metal"
Draw = Callable[[Tensor], Tensor]  # a plain function, or a Module whose forward ignores ``w``

# ── distributions ────────────────────────────────────────────────────────────

# Classes the audit's generic constructor cannot build.
_DIST_OVERRIDES: dict[str, Callable[[], D.Distribution]] = {
    "LowRankMultivariateNormal": lambda: D.LowRankMultivariateNormal(
        lucid.zeros(3), lucid.ones(3, 2) * 0.3, lucid.ones(3)
    ),
}

#: ``Function`` -> the public call that reaches it.  A ``TRACED_BACKWARD``
#: member missing here fails the guard instead of going unprobed.
FUNCTION_ENTRY: dict[str, str] = {
    "_StandardGamma": "dist:Gamma",
    "_MatrixPowerZero": "fn:matrix_power0",
    "_LogAbsDet": "fn:slogdet",
    "_FakeQuantizeAffine": "fn:fake_quantize",
    "_CholeskyAutograd": "fn:cholesky",
    "_SVD": "fn:svd",
    "_QR": "fn:qr",
    "_Pinv": "fn:pinv",
    "_Eigh": "fn:eigh",
}


def _spd() -> Tensor:
    return lucid.tensor([[2.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 2.0]], device=DEVICE)


FUNCTIONS: dict[str, Draw] = {
    "matrix_power0": lambda w: LA.matrix_power(_spd() * w, 0),
    "slogdet": lambda w: LA.slogdet(_spd() * w)[1],
    "fake_quantize": lambda w: lucid.quantization.fake_quantize(
        lucid.linspace(-1.0, 1.0, 6, device=DEVICE) * w, 0.05, 0, -128, 127
    ),
    "cholesky": lambda w: LA.cholesky(_spd() * w),
    "svd": lambda w: LA.svd(_spd() * w)[1],
    "qr": lambda w: LA.qr(_spd() * w)[1],
    "pinv": lambda w: LA.pinv(_spd() * w),
    "eigh": lambda w: LA.eigh(_spd() * w)[0],
}

# ── top-level random ops ─────────────────────────────────────────────────────

_RANDOM: dict[str, Draw] = {
    "rand": lambda w: lucid.rand(3, device=DEVICE) * w,
    "randn": lambda w: lucid.randn(3, device=DEVICE) * w,
    "randint": lambda w: lucid.randint(0, 5, (3,), device=DEVICE).float() * w,
    "bernoulli": lambda w: lucid.bernoulli(lucid.full((3,), 0.4, device=DEVICE) * w),
    "normal": lambda w: lucid.normal(0.0, 1.0, size=(3,), device=DEVICE) * w,
    "rand_like": lambda w: lucid.rand_like(w.reshape(1).expand(3)) * w,
    "randn_like": lambda w: lucid.randn_like(w.reshape(1).expand(3)) * w,
    "randint_like": lambda w: lucid.randint_like(w.reshape(1).expand(3), 0, 5).float() * w,
    "randperm": lambda w: lucid.randperm(5, device=DEVICE).float() * w,
    "multinomial": lambda w: lucid.multinomial(
        lucid.full((4,), 0.25, device=DEVICE) * w, 3, replacement=True
    ).float()
    * w,
    "poisson": lambda w: lucid.poisson(lucid.full((3,), 2.0, device=DEVICE) * w) * w,
}

#: Public ``lucid.*`` names that look random and are not draws.
_RANDOM_NOT_DRAWS = frozenset({"manual_seed", "seed", "initial_seed"})
_RANDOM_PREFIXES = ("rand", "normal", "bernoulli", "multinomial", "poisson")


def random_names_in_package() -> set[str]:
    """Every ``lucid.*`` name that draws, found by prefix; the probe table must cover it."""
    return {
        name
        for name in lucid.__all__
        if name.startswith(_RANDOM_PREFIXES) and name not in _RANDOM_NOT_DRAWS
    }


def random_table() -> frozenset[str]:
    return frozenset(_RANDOM)


def _map_tensors(dist: D.Distribution, fn: Callable[[Tensor], Tensor]) -> D.Distribution:
    """A copy of ``dist`` (and of the distributions it holds) with ``fn`` on its float tensors."""
    clone = copy.copy(dist)
    for key, value in vars(clone).items():
        if isinstance(value, Tensor) and value.dtype == lucid.float32:
            setattr(clone, key, fn(value))
        elif isinstance(value, D.Distribution):
            setattr(clone, key, _map_tensors(value, fn))
    return clone


def _lifted(value: Tensor) -> Tensor:
    """Give a parameter a leading batch axis: the shape a model's ``Linear`` output has."""
    if value.ndim == 0:
        return value.reshape(1, 1).expand(2, 2)
    if value.ndim == 1:
        return value.reshape(1, -1).expand(2, value.shape[0])
    return value


def build_distribution(name: str) -> D.Distribution:
    """The distribution with batched (rank >= 2) float parameters.

    The compile blow-up needs the sampler's inputs to come out of a matrix product at
    rank 2, as a ``Linear`` layer's do; a rank-1 or 0-d parameter compiled in under a
    second even with the exponential sampler (LCD-786 measurements).
    """
    cls = getattr(D, name)
    if name in _DIST_OVERRIDES:
        return _DIST_OVERRIDES[name]()
    plain, why = _construct(cls)
    if plain is None:
        raise RuntimeError(f"cannot build D.{name}: {why}")
    named: dict[str, Tensor] = {}
    claimed: set[int] = set()
    for key, value in vars(plain).items():
        if key not in inspect.signature(cls).parameters:
            continue
        if not (isinstance(value, Tensor) and value.dtype == lucid.float32):
            continue
        group = next((i for i, names in enumerate(_EXCLUSIVE) if key in names), None)
        if group is not None:
            if group in claimed:
                continue  # the class takes exactly one of the group
            claimed.add(group)
        named[key] = _lifted(value)
    lifted, why = _construct(cls, **named)
    if lifted is None:
        raise RuntimeError(f"cannot build D.{name} with batched parameters: {why}")
    assert isinstance(lifted, D.Distribution)
    return lifted


def distribution_names() -> list[str]:
    """Every exported distribution class whose ``rsample`` exists -- from the package."""
    names: list[str] = []
    for name in D.__all__:
        cls = getattr(D, name)
        if not (inspect.isclass(cls) and issubclass(cls, D.Distribution)):
            continue
        if cls in (D.Distribution, D.ExponentialFamily):
            continue
        has = getattr(cls, "has_rsample", False)
        if has is True or isinstance(has, property):
            names.append(name)
    return names


def _required(fn: Callable[..., Tensor]) -> list[inspect.Parameter]:
    return [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


def special_names() -> list[str]:
    """Every ``lucid.special`` function of one or two tensors -- composites with series loops.

    A function whose required argument is an ``int`` (``polygamma``'s order,
    ``multigammaln``'s dimension) takes no tensor there, so the shared feed does not fit.
    """
    names: list[str] = []
    for name in S.__all__:
        fn = getattr(S, name)
        if not inspect.isroutine(fn):
            continue
        required = _required(fn)
        if len(required) in (1, 2) and all(p.annotation is not int for p in required):
            names.append(name)
    return names


class _SpecialDraw(nn.Module):
    """A special function of ``softplus(const @ W) + 0.5``, the same rank-2 trainable feed."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._fn = getattr(S, name)
        self._arity = len(_required(self._fn))
        self.mix = nn.Parameter(lucid.eye(4))

    def forward(self, _w: Tensor) -> Tensor:
        a = F.softplus(lucid.full((4, 4), 1.2, device=DEVICE) @ self.mix) + 0.5
        out = self._fn(*[a] * self._arity)
        assert isinstance(out, Tensor)
        return out


class _DistributionDraw(nn.Module):
    """``rsample`` of a distribution whose every float tensor is ``softplus(t @ W) + 0.5``.

    A parameter that reaches the sampler through a matrix product and a softplus is how a
    real model feeds one (``Gamma(softplus(Linear(x)) + 0.5, ...)``), and it is what made
    the LCD-784 compile exponential: the same sampler fed by ``const * w`` compiled in
    under a second.  The values are not the plain distribution's (a loc of 0 becomes 1.2);
    the guard times the compile, it does not look at the draw.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        # Moved before the trace: a transfer inside it is a mixed-device trace, which
        # runs eagerly and never reaches the graph compiler.
        self._base = _map_tensors(build_distribution(name), lambda t: t.to(DEVICE))
        widths: list[int] = []

        def record(t: Tensor) -> Tensor:
            widths.append(t.shape[-1])
            return t

        _map_tensors(self._base, record)
        self.mix = nn.ParameterList([nn.Parameter(lucid.eye(n)) for n in widths])

    def forward(self, _w: Tensor) -> Tensor:
        slots = iter(self.mix)

        def tie(t: Tensor) -> Tensor:
            weight = next(slots)
            return F.softplus(t @ weight) + 0.5

        tied = _map_tensors(self._base, tie)
        if not tied.has_rsample:
            raise RuntimeError("the distribution built here has no rsample")
        return tied.rsample().float()


# ── the model and the timing ─────────────────────────────────────────────────


def draw_for(target: str) -> Draw:
    kind, _, key = target.partition(":")
    if kind == "dist":
        return _DistributionDraw(key)
    if kind == "special":
        return _SpecialDraw(key)
    if kind == "fn":
        return FUNCTIONS[key]
    if kind == "random":
        return _RANDOM[key]
    raise KeyError(target)


class _Probe(nn.Module):
    def __init__(self, draw: Draw) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(1))
        self._draw = draw

    def forward(self, x: Tensor) -> Tensor:
        sample = self._draw(self.w.reshape(()))
        return x + sample.float().sum()


def compile_seconds(target: str) -> tuple[float, bool]:
    """Time the first ``make_step`` call -- trace plus compile -- of one target."""
    from lucid.compile import make_step

    lucid.manual_seed(0)
    model = _Probe(draw_for(target)).to(DEVICE)
    x = lucid.randn(4).to(DEVICE)
    t = lucid.randn(4).to(DEVICE)
    step = make_step(model, F.mse_loss)
    start = time.perf_counter()
    step(x, t).backward()
    elapsed = time.perf_counter() - start
    return elapsed, not getattr(step, "eager_only")


def main(targets: list[str]) -> None:
    for target in targets:
        print(f"START {target}", flush=True)
        try:
            seconds, compiled = compile_seconds(target)
        except NotImplementedError as exc:
            # A function whose gradient w.r.t. this argument is not implemented has no
            # backward to compile; the guard cannot say anything about it.
            print(f"SKIP {target} {str(exc)[:120]!r}", flush=True)
            continue
        print(f"DONE {target} {seconds:.3f} {int(compiled)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
