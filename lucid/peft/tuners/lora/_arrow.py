r"""Arrow: route each token over the LoRA adapters a layer already holds.

Arrow (Ostapenko et al., 2024, arXiv:2405.11157) reuses a library of task-specific LoRA adapters
instead of training a new one.  At every adapted layer it gives each adapter a *prototype*, the
top right singular vector :math:`p_e` of its update :math:`\Delta W_e = B_e A_e`, and sends each
token :math:`x` through the ``top_k`` adapters whose prototypes it is most aligned with:

.. math::

    s_e = \lvert x \cdot p_e \rvert, \qquad
    c = \mathrm{softmax}\bigl(\mathrm{mask}_{\mathrm{top}\text{-}k}(s) / T\bigr), \qquad
    y = W x + \sum_e c_e \, \sigma_e \, B_e A_e x,

with :math:`T` the router temperature and :math:`\sigma_e` the adapter's own LoRA scale.  The
absolute value makes the router blind to the sign of a singular vector, which is only defined up
to one.  GenKnowSub (ACL 2025, short paper) first subtracts the mean of a few general-knowledge
adapters' factors, :math:`A_e - \bar A,\ B_e - \bar B`, from every task adapter, so what is left
of each is what is specific to its task.

In this package an Arrow router is one more adapter of a layer, built from ``LoraConfig(
arrow_config=ArrowConfig(...))`` after its task adapters (``create_arrow_model`` does both).
It is a ``_Variant`` like DoRA: the layer asks it for the update the adapter adds, and it asks
the layer for the other adapters' factors.  What it keeps per layer is the ``_ArrowRouter``, a
module with no parameters: the prototypes, in a buffer that is not saved, and the names of the
adapters they are of.

* The task adapters are the layer's other plain LoRA adapters (not DoRA, none with a ``lora_bias``,
  no other router), less the general-knowledge ones, in the order the layer holds them.  They are
  read from the layer each pass, so adding or deleting one takes effect at the next, and the
  prototypes are rebuilt then.  They are also rebuilt when the values of an adapter they were
  built from change in place (``load_adapter`` over a name, an optimizer step), so the router
  never routes by a stale prototype; the reference builds them once.
* The general-knowledge adapters are the ones ``ArrowConfig.gks_adapter_names`` names, or,
  where it is not filled in (a config read from a file does not keep it), the ones named
  ``gks_<i>`` as ``create_arrow_model`` names them.
* GenKnowSub is applied on the fly.  The reference subtracts the mean from the task adapters'
  stored weights once, at the first forward pass, so what an adapter file holds depends on
  whether the model has run; here the stored weights are never changed and the same router
  can be rebuilt as often as the adapters change.
* The router cannot be merged: no single weight is what per-token routing computes.
"""

import functools
import operator
import re
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, override

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.lora._subconfigs import ArrowConfig
from lucid.peft.tuners.lora._variants import _Variant

if TYPE_CHECKING:
    from lucid.peft.tuners.lora._layer import LoraLayer, _Extras, _Residual

#: The per-adapter table the router of each Arrow adapter is kept in (the reference's name).
ARROW_TABLE: Final = "lora_arrow"
#: How ``create_arrow_model`` names the adapters it loads, and so how a layer tells them apart.
TASK_ADAPTER_PREFIX: Final = "task_"
GKS_ADAPTER_PREFIX: Final = "gks_"
ROUTER_ADAPTER_NAME: Final = "arrow_router"

_GKS_NAME: Final = re.compile(rf"{re.escape(GKS_ADAPTER_PREFIX)}\d+")
#: Power iterations and the guard against dividing by zero, the reference's.
_POWER_ITERATIONS: Final = 15
_EPS: Final = 1e-8


class _Roster(NamedTuple):
    """The adapters a router reads from a layer: the task adapters it routes over, the general ones it subtracts."""

    experts: tuple[str, ...] = ()
    general: tuple[str, ...] = ()


#: The factors of the roster's adapters, by name: ``(A, B)`` of each.
type _Pairs = dict[str, tuple[Parameter, Parameter]]


def _options(arrow_config: ArrowConfig | None) -> ArrowConfig:
    """``arrow_config``, which an Arrow adapter is always built with."""
    if arrow_config is None:
        raise TypeError("an Arrow adapter is built from an ArrowConfig, got None")
    return arrow_config


def _plain(layer: LoraLayer) -> list[str]:
    """The adapters of ``layer`` a router can read: plain LoRA, in the order held."""
    return [n for n, spec in layer._specs.items() if spec.variant == "lora" and not spec.lora_bias]


class _ArrowRouter(nn.Module):
    r"""The Arrow state of one adapter on one layer: its settings, its roster and the prototypes of that roster.

    Holds no parameter.  ``prototypes`` (``(experts, in_features)``, one row per task adapter,
    in the factors' dtype) is a buffer outside the state dict.  It is rebuilt from the layer when
    ``sync`` finds a different roster or a factor of it changed in place (a load, an optimizer
    step: the tensors' version counters say so).  The settings are plain attributes so that
    ``lucid.compile`` keys a trace on them.

    Parameters
    ----------
    config : ArrowConfig
        The router's options; the adapter names it holds are read as the roster's explicit part.
    """

    prototypes: Tensor

    def __init__(self, config: ArrowConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.temperature = config.router_temperature
        self.use_gks = config.use_gks
        self.rng_seed = config.rng_seed
        self.task_names = (
            None if config.task_adapter_names is None else tuple(config.task_adapter_names)
        )
        self.general_names = (
            None if config.gks_adapter_names is None else tuple(config.gks_adapter_names)
        )
        self._seen = _Roster()
        self._stamp: tuple[int, ...] = ()

    @override
    def extra_repr(self) -> str:
        return (
            f"top_k={self.top_k}, temperature={self.temperature}, use_gks={self.use_gks}, "
            f"experts={self._seen.experts}"
        )

    # -- what the layer holds ------------------------------------------------

    def _derive(self, layer: LoraLayer) -> _Roster:
        """The roster the layer holds now, by name alone."""
        plain = _plain(layer)
        general = (
            [n for n in plain if _GKS_NAME.fullmatch(n)]
            if self.general_names is None
            else list(self.general_names)
        )
        return _Roster(tuple(n for n in plain if n not in general), tuple(general))

    @staticmethod
    def _pairs(layer: LoraLayer, roster: _Roster) -> _Pairs:
        return {n: layer._factor_pairs(n)[0] for n in (*roster.experts, *roster.general)}

    def _validate(self, layer: LoraLayer, roster: _Roster, pairs: _Pairs) -> None:
        """Raise ``ValueError`` if ``roster`` cannot be routed or subtracted: nothing to route, ``top_k`` too large, a named adapter missing, ranks that differ."""
        where = type(layer).__name__
        plain = _plain(layer)
        for label, named in (("task", self.task_names), ("general-knowledge", self.general_names)):
            missing = [n for n in named or () if n not in plain]
            if missing:
                raise ValueError(
                    f"Arrow names the {label} adapters {missing}, but this {where} holds no plain "
                    "LoRA adapter of those names (DoRA, lora_bias and router adapters cannot be routed)"
                )
        if not roster.experts:
            raise ValueError(
                f"Arrow routes the plain LoRA adapters a layer already holds, and this {where} "
                "holds none besides the router: add the task adapters first "
                "(create_arrow_model does)"
            )
        if self.top_k > len(roster.experts):
            raise ValueError(
                f"top_k={self.top_k} exceeds the {len(roster.experts)} task adapter(s) "
                f"{list(roster.experts)} this {where} holds; lower top_k"
            )
        if not self.use_gks:
            return
        if not roster.general:
            raise ValueError(
                "use_gks=True needs general-knowledge adapters: name them in "
                "ArrowConfig.gks_adapter_names, or call them gks_<i>"
            )
        if len({tuple(tuple(w.shape) for w in pair) for pair in pairs.values()}) > 1:
            raise ValueError(
                "GenKnowSub subtracts the general adapters' factors from the task adapters', "
                "so they must all have the same rank and shape"
            )

    def check(self, layer: LoraLayer) -> None:
        """Raise ``ValueError`` if this router cannot be built on ``layer`` as it is now."""
        roster = self._derive(layer)
        self._validate(layer, roster, self._pairs(layer, roster))

    # -- prototypes ----------------------------------------------------------

    def _mean_factors(self, roster: _Roster, pairs: _Pairs) -> tuple[Tensor, Tensor] | None:
        """The mean of the general adapters' ``A`` and of their ``B`` (``None`` unless ``use_gks``)."""
        if not self.use_gks:
            return None
        held = [pairs[n] for n in roster.general]
        return (
            lucid.stack([a for a, _ in held]).mean(dim=0),
            lucid.stack([b for _, b in held]).mean(dim=0),
        )

    def _top_right_singular_vector(self, a: Tensor, b: Tensor) -> Tensor:
        r"""The dominant right singular vector of :math:`BA`, by power iteration on :math:`A^\top (B^\top B) A`.

        It lives in the input space, which is what a token is compared with, and
        :math:`\Delta W = BA` is never formed.  The iteration runs in float32 from a random
        start (drawn on the CPU, so it is the same on every device; ``rng_seed`` fixes it, and
        ``None`` draws from the global generator) and is cut at the reference's fixed count: for
        a spectrum without a gap it has not converged, as the reference's has not.  The sign is
        arbitrary, and the routing ignores it.
        """
        a32, b32 = a.to(lucid.float32), b.to(lucid.float32)
        gram = b32.mT @ b32
        generator = None if self.rng_seed is None else lucid.Generator(self.rng_seed, device="cpu")
        v = lucid.randn(a32.shape[1], 1, generator=generator).to(a32.device)
        v = v / (linalg.norm(v) + _EPS)
        for _ in range(_POWER_ITERATIONS):
            w = a32.mT @ (gram @ (a32 @ v))
            v = w / (linalg.norm(w) + _EPS)
        return v.reshape(-1).to(a.dtype)

    def _rebuild(self, roster: _Roster, pairs: _Pairs) -> None:
        with lucid.no_grad():
            mean = self._mean_factors(roster, pairs)
            rows: list[Tensor] = []
            for name in roster.experts:
                a, b = pairs[name]
                rows.append(
                    self._top_right_singular_vector(
                        *((a, b) if mean is None else (a - mean[0], b - mean[1]))
                    )
                )
            self.register_buffer("prototypes", lucid.stack(rows, dim=0), persistent=False)

    def sync(self, layer: LoraLayer) -> tuple[_Roster, _Pairs]:
        """Make the prototypes those of the roster ``layer`` holds now; return the roster and its factors."""
        roster = self._derive(layer)
        pairs = self._pairs(layer, roster)
        stamp = tuple(p._version for pair in pairs.values() for p in pair)
        if roster != self._seen or stamp != self._stamp:
            self._validate(layer, roster, pairs)
            self._rebuild(roster, pairs)
            self._seen, self._stamp = roster, stamp
        return roster, pairs

    # -- routing -------------------------------------------------------------

    def delta(self, layer: LoraLayer, x: Tensor) -> Tensor:
        r"""What the router adds to ``x``'s base output: :math:`\sum_e c_e\,\sigma_e\,B_e A_e x`, per token.

        ``x`` is flattened to tokens (``(..., in)`` to ``(tokens, in)``) and the result reshaped
        back to ``(..., out)``.  The coefficients carry the gradient of the token, as the reference's
        do; the prototypes do not.
        """
        roster, pairs = self.sync(layer)
        dtype = pairs[roster.experts[0]][0].dtype
        tokens = x.reshape(-1, x.shape[-1])
        tokens = tokens if tokens.dtype == dtype else tokens.to(dtype)
        similarity = (tokens @ self.prototypes.mT).abs()
        top_values, top_index = lucid.topk(similarity, self.top_k, dim=1)
        scores = lucid.full(similarity.shape, float("-inf"), dtype=dtype, device=similarity.device)
        coefficients = F.softmax(scores.scatter(1, top_index, top_values) / self.temperature, dim=1)
        mean = self._mean_factors(roster, pairs)
        parts: list[Tensor] = []
        for e, name in enumerate(roster.experts):
            a, b = pairs[name]
            down, up = (a, b) if mean is None else (a - mean[0], b - mean[1])
            weight = coefficients[:, e : e + 1] * layer._specs[name].scaling
            parts.append(((tokens @ down.mT) @ up.mT) * weight)
        routed = functools.reduce(operator.add, parts)
        return routed.reshape(*x.shape[:-1], routed.shape[-1])


class _Arrow(_Variant, variant="arrow"):
    """The adapter is a router over the layer's other adapters; it keeps no update of its own to merge."""

    __slots__ = ()
    label = "Arrow routing (arrow_config)"
    mergeable: ClassVar[bool] = False

    @override
    def check(self, layer: LoraLayer, arrow_config: ArrowConfig | None = None) -> None:
        super().check(layer, arrow_config)
        _ArrowRouter(_options(arrow_config)).check(layer)

    @override
    def attach(
        self, layer: _Residual, adapter: str, arrow_config: ArrowConfig | None = None
    ) -> None:
        table: nn.ModuleDict[_ArrowRouter] = getattr(layer, ARROW_TABLE)
        router = _ArrowRouter(_options(arrow_config))
        router.sync(layer)
        table[adapter] = router

    @staticmethod
    def _router(layer: _Residual, adapter: str) -> _ArrowRouter:
        table: nn.ModuleDict[_ArrowRouter] = getattr(layer, ARROW_TABLE)
        return table[adapter]

    @override
    def contribution(
        self, layer: _Residual, adapter: str, x: Tensor, result: Tensor, extras: _Extras
    ) -> Tensor:
        del result, extras  # the routed update follows the layer's input alone
        # Dropout acts on the routed update, not on the tokens, as in the reference.
        update: Tensor = layer.lora_dropout[adapter](self._router(layer, adapter).delta(layer, x))
        return update

    @override
    def shift(
        self, layer: _Residual, adapter: str, rows: Tensor, delta: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        del (
            layer,
            adapter,
            rows,
            delta,
        )  # unreachable: ``mergeable`` is False, a merge is refused first
        raise RuntimeError("Cannot merge an active Arrow router adapter. Remove it first.")


__all__ = [
    "ARROW_TABLE",
    "GKS_ADAPTER_PREFIX",
    "ROUTER_ADAPTER_NAME",
    "TASK_ADAPTER_PREFIX",
    "_Arrow",
    "_ArrowRouter",
]
