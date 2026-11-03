"""How a new LoRA adapter's factors are initialised: one strategy class per ``init_lora_weights`` spelling.

``init_lora_weights`` is ``True``, ``False`` or a name; each spelling is a
class that registers itself as it is defined (``init=`` in the class
statement), so a new initialisation is one subclass — no branch on the
spelling anywhere else.  ``_Init.of`` turns what the user wrote into the
strategy, ``_Init.value`` is the canonical spelling a config stores, and
``apply`` writes the factors of one adapter a layer has just built.

PiSSA and OLoRA are not "just a different random draw": they read the base
weight, put its principal directions (an SVD) or an orthogonal basis (a QR)
into the factors, and write the residual back into the base weight, so the
adapted layer starts as the original.  They are the reason this runs after
the factors exist and before anything else reads the base weight — and the
reason ``on_load`` exists: an adapter read from a file gets its factors from
the file, so it is built with the strategy that leaves the base alone.
"""

import dataclasses
import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final, Self, override

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid.peft._adapter_file import _AdapterFile
from lucid.peft.tuners._spectral import _svd_lowrank

if TYPE_CHECKING:
    from lucid.peft.tuners.lora._config import LoraConfig
    from lucid.peft.tuners.lora._layer import LoraLayer

#: Initialisations of the reference that need machinery lucid.peft does not have yet.
_PLANNED: Final = {
    "corda": "it is data-driven: it needs a calibration pass over batches of inputs",
    "loftq": "it needs a quantized base weight",
    "orthogonal": "it is not implemented",
    "mica": "it is not implemented",
}

_PISSA_ITERATIONS: Final = re.compile(r"pissa_niter_(\d+)")

#: The axis a factor is stacked along to add a rank: ``lora_A`` along its rows, ``lora_B`` along its columns.
_RANK_AXIS: Final = {"lora_A": 0, "lora_B": 1}


class _Init(ABC):
    """One way to initialise an adapter's factors, registered under its spelling.

    A subclass names itself with ``init=`` and supplies ``apply``; one that
    stands for several spellings (``pissa``, ``pissa_niter_4``) overrides
    ``parse``.  Strategies are immutable values.
    """

    __slots__ = ()

    _strategies: ClassVar[dict[str, type[_Init]]] = {}
    #: The spelling, for a strategy with one; the instance's ``value`` otherwise.
    spelling: ClassVar[str]
    #: Whether the strategy factorises the base weight — the layer must hold a dense ``(out, in)`` one —
    #: and writes the residual back into it, so an adapter saved beside that base needs ``to_plain``.
    factorises: ClassVar[bool] = False

    @override
    def __init_subclass__(cls, *, init: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if init is not None:
            cls.spelling = init
            cls._strategies[init] = cls

    @classmethod
    def of(cls, spec: bool | str) -> _Init:
        """The strategy ``spec`` asks for.

        Raises
        ------
        NotImplementedError
            ``spec`` names an initialisation lucid.peft does not have.
        ValueError
            ``spec`` names none.
        """
        if isinstance(spec, bool):
            return cls._strategies["default" if spec else "random"]()
        text = spec.lower()
        for strategy in cls._strategies.values():
            found = strategy.parse(text)
            if found is not None:
                return found
        if text in _PLANNED:
            raise NotImplementedError(
                f"init_lora_weights={spec!r} is not implemented in lucid.peft yet: {_PLANNED[text]}"
            )
        raise ValueError(f"Unknown initialization init_lora_weights={spec!r}")

    @classmethod
    def parse(cls, text: str) -> Self | None:
        """The strategy spelled ``text`` (lower case), or ``None`` if ``text`` is not its spelling."""
        return cls() if text == cls.spelling else None

    @property
    @abstractmethod
    def value(self) -> bool | str:
        """The spelling a config stores: ``True``, ``False`` or the lower-case name."""

    def on_load(self) -> _Init:
        """The strategy to build an adapter with when a file is about to overwrite its factors.

        An initialisation that only fills the factors can run as it is; one
        that rewrites the base weight (``factorises``) overrides this, for
        loading must not change the base the adapter is put on.
        """
        return self

    def check(self, layer: LoraLayer, r: int) -> None:
        """Raise ``ValueError`` if ``layer`` cannot be initialised this way at rank ``r``."""
        if not self.factorises:
            return
        if not layer._dense_weight:
            raise ValueError(
                f"init_lora_weights={self.value!r} factorises the base weight, which "
                f"{type(layer).__name__} does not hold as a dense (out, in) matrix; use True, "
                "False or 'gaussian'"
            )
        biggest = min(layer.weight.shape)
        if r > biggest:
            raise ValueError(
                f"init_lora_weights={self.value!r} needs r <= min(out_features, in_features) = "
                f"{biggest}, got r={r}"
            )

    @abstractmethod
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        """Initialise the factors of ``adapter``, which ``layer`` has built."""

    def to_plain(
        self, trained: _AdapterFile[LoraConfig], initial: _AdapterFile[LoraConfig]
    ) -> _AdapterFile[LoraConfig]:
        r"""``trained`` as a plain LoRA that fits the original base weight.

        A strategy that rewrites the base weight (``factorises``) gives an
        adapter that is only right beside the rewritten base.  Its change
        since initialisation is :math:`\Delta W = s(BA - B_0A_0)`, which a
        plain LoRA of rank :math:`2r` holds exactly as
        :math:`[B\;{-B_0}]\,[A;\,A_0]` (``lora_alpha`` doubled, so :math:`s`
        is unchanged) and which adds to the original weight.  Any other
        strategy left the base alone: its adapter is returned as it is, with
        a warning.

        Parameters
        ----------
        trained : _AdapterFile
            The adapter after training: its config and file tensors.
        initial : _AdapterFile
            The same adapter right after initialisation.

        Returns
        -------
        _AdapterFile
            The converted config (``init_lora_weights=True``) and tensors.

        Raises
        ------
        ValueError
            The adapter uses DoRA, or ``use_rslora`` with a rank or alpha
            pattern, or ``initial`` does not match ``trained``.
        """
        if not self.factorises:
            warnings.warn(
                "`path_initial_model_for_weight_conversion` only converts an adapter whose "
                "initialisation rewrote the base weight (PiSSA, OLoRA); "
                f"init_lora_weights={self.value!r} did not, so the adapter is saved as it is.",
                stacklevel=2,
            )
            return trained
        _check_convertible(trained.config, initial.config)
        return _AdapterFile(_doubled(trained.config), _stacked(trained.state, initial.state))


class _Default(_Init, init="default"):
    r"""``True``: Kaiming-uniform :math:`A` (as ``nn.Linear``), zero :math:`B`; the update starts at zero."""

    __slots__ = ()

    @property
    @override
    def value(self) -> bool:
        return True

    @override
    @classmethod
    def parse(cls, text: str) -> Self | None:
        return None  # a bool is its only spelling

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        layer._reset_factors(adapter, gaussian=False)

    @override
    def to_plain(
        self, trained: _AdapterFile[LoraConfig], initial: _AdapterFile[LoraConfig]
    ) -> _AdapterFile[LoraConfig]:
        del initial  # B starts at zero and the base is untouched: it is a plain LoRA already
        return trained


class _Random(_Init, init="random"):
    """``False``: both factors keep the layers' own initialisation (the update is not zero)."""

    __slots__ = ()

    @property
    @override
    def value(self) -> bool:
        return False

    @override
    @classmethod
    def parse(cls, text: str) -> Self | None:
        return None

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        del layer, adapter  # the factors are already as the layers made them


class _Gaussian(_Init, init="gaussian"):
    r"""``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`, zero :math:`B`."""

    __slots__ = ()

    @property
    @override
    def value(self) -> str:
        return self.spelling

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        layer._reset_factors(adapter, gaussian=True)


def _widen(weight: Tensor) -> Tensor:
    """``weight`` as float32 — the decompositions run there, whatever the base's dtype."""
    return weight if weight.dtype == lucid.float32 else weight.to(lucid.float32)


def _install(
    layer: LoraLayer, adapter: str, lora_a: Tensor, lora_b: Tensor, residual: Tensor
) -> None:
    """Write the factors into ``adapter`` and the residual into the base weight, in place."""
    with lucid.no_grad():
        ((a, b),) = layer._factor_pairs(adapter)
        a.copy_(lora_a.to(a.dtype))
        b.copy_(lora_b.to(b.dtype))
        weight = layer.weight
        weight.copy_(residual.to(weight.dtype))


class _Factorising(_Init):
    """A strategy that factorises the base weight and writes the residual back into it."""

    __slots__ = ()
    factorises = True

    @override
    def on_load(self) -> _Init:
        # The file holds the factors, and the base it is loaded onto already holds (or must keep) its weight.
        return _Random()


class _Pissa(_Factorising, init="pissa"):
    r"""``"pissa"`` / ``"pissa_niter_<k>"``: the principal singular components, residual in the base.

    With :math:`W = U\Sigma V^\top` and rank :math:`r`, scale :math:`s`:
    :math:`B = U_r\sqrt{\Sigma_r/s}`, :math:`A = \sqrt{\Sigma_r/s}\,V_r^\top`
    and the base becomes :math:`W - s\,BA` — so :math:`W' + sBA = W` and the
    adapter trains the directions that matter most.  The exact SVD for
    ``"pissa"``, a randomised one with ``k`` power iterations for
    ``"pissa_niter_<k>"`` (cheaper on a large weight).  The base weight is
    **changed in place**.
    """

    __slots__ = ("niter",)

    def __init__(self, niter: int | None = None) -> None:
        self.niter = niter

    @property
    @override
    def value(self) -> str:
        return "pissa" if self.niter is None else f"pissa_niter_{self.niter}"

    @override
    @classmethod
    def parse(cls, text: str) -> Self | None:
        if text == "pissa":
            return cls()
        if text.startswith("pissa"):
            found = _PISSA_ITERATIONS.fullmatch(text)
            if found is None:
                raise ValueError(
                    "init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', "
                    f"got {text!r} instead."
                )
            return cls(int(found.group(1)))
        return None

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        spec = layer._specs[adapter]
        with lucid.no_grad():
            weight = _widen(layer.weight)
            if self.niter is None:
                u, s, vh = linalg.svd(weight, full_matrices=False)
                u_r, s_r, vh_r = u[:, : spec.r], s[: spec.r], vh[: spec.r]
            else:
                u_r, s_r, v_r = _svd_lowrank(weight, spec.r, self.niter)
                vh_r = v_r.mT
            root = (s_r / spec.scaling).sqrt()
            lora_a = root.unsqueeze(1) * vh_r
            lora_b = u_r * root.unsqueeze(0)
            _install(layer, adapter, lora_a, lora_b, weight - spec.scaling * (lora_b @ lora_a))


class _Olora(_Factorising, init="olora"):
    r"""``"olora"``: an orthonormal basis of the weight's column space (QR), residual in the base.

    With :math:`W = QR`: :math:`B = Q_{:,\,:r}`, :math:`A = R_{:r}` and the
    base becomes :math:`W - s\,BA`, so :math:`W' + sBA = W`.  :math:`B` starts
    with orthonormal columns (Büyükakyüz, 2024).  The base weight is
    **changed in place**.
    """

    __slots__ = ()

    @property
    @override
    def value(self) -> str:
        return self.spelling

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        spec = layer._specs[adapter]
        with lucid.no_grad():
            weight = _widen(layer.weight)
            q, r_factor = linalg.qr(weight)
            lora_a, lora_b = r_factor[: spec.r], q[:, : spec.r]
            _install(layer, adapter, lora_a, lora_b, weight - spec.scaling * (lora_b @ lora_a))


class _Eva(_Init, init="eva"):
    """Leave factors untouched at construction; :func:`initialize_lora_eva_weights` populates them from activations."""

    @property
    @override
    def value(self) -> str:
        return "eva"

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        layer._reset_factors(adapter, gaussian=False)


def _loraga_direction_factors(
    u: Tensor, v: Tensor, r: int, direction: str, device: lucid.device
) -> tuple[Tensor, Tensor]:
    if direction == "ArBr":
        return v[r : 2 * r, :], u[:, :r]
    if direction == "A2rBr":
        return v[:r, :], u[:, r : 2 * r]
    if direction == "ArB2r":
        return v[r : 2 * r, :], u[:, r : 2 * r]
    rand_idx = lucid.randperm(2 * r, device=device)
    a_idx = rand_idx[:r]
    b_idx = rand_idx[r : 2 * r]
    return v[a_idx, :], u[:, b_idx]


def _loraga_rescale(
    lora_a: Tensor, lora_b: Tensor, weight: Tensor, r: int, ga_cfg: object, scaling: float
) -> tuple[Tensor, Tensor]:
    scale = getattr(ga_cfg, "scale", "stable")
    if scale == "gd_scale":
        return lora_a / scaling, lora_b / scaling
    if scale == "unit":
        _, s_a, _ = linalg.svd(lora_a, full_matrices=False)
        _, s_b, _ = linalg.svd(lora_b, full_matrices=False)
        norm_a = s_a[0].clamp(min=1e-12)
        norm_b = s_b[0].clamp(min=1e-12)
        return lora_a / norm_a, lora_b / norm_b
    if scale == "stable":
        d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
        gamma = float(getattr(ga_cfg, "stable_gamma", 16))
        return lora_a * (d_out**0.25 / gamma**0.5), lora_b * (d_in**0.25 / gamma**0.5)
    _, s_b, _ = linalg.svd(lora_b, full_matrices=False)
    _, weight_s, _ = _svd_lowrank(weight, r, 4)
    if float(s_b[0].item()) > 0.0:
        lora_b = lora_b * (weight_s[0] / s_b[0])
    return lora_a, lora_b


class _LoraGa(_Factorising, init="lora_ga"):
    @property
    @override
    def value(self) -> str:
        return "lora_ga"

    r"""Gradient-approximated LoRA initialisation (:func:`preprocess_loraga`).

    Decomposes the cached base-weight gradient :math:`G = U S V^	op` at rank
    :math:`2r`, selects the :math:`(A, B)` subspace per ``LoraGAConfig.direction``,
    rescales per ``LoraGAConfig.scale``, and subtracts :math:`s B A` from the
    base weight so the initial forward output matches the unadapted model.
    """

    @override
    def apply(self, layer: LoraLayer, adapter: str) -> None:
        base_layer = layer.get_base_layer()
        grad = base_layer.__dict__.get("_peft_loraga_grad")
        if not isinstance(grad, Tensor):
            warnings.warn(
                "Tune the model with `preprocess_loraga` before calling `get_peft_model` when "
                "`init_lora_weights='lora_ga'`. Falling back to standard LoRA initialization.",
                UserWarning,
                stacklevel=3,
            )
            layer._reset_factors(adapter, gaussian=False)
            return

        spec = layer._specs[adapter]
        r = spec.r
        ga_cfg = base_layer.__dict__.get("_peft_loraga_config")
        direction = getattr(ga_cfg, "direction", "ArB2r")

        with lucid.no_grad():
            weight = layer.weight.to(lucid.float32)
            grad_f32 = grad.to(device=weight.device, dtype=lucid.float32)
            d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
            if min(d_out, d_in) < 2 * r:
                raise ValueError(
                    f"LoRA-GA requires min(in_features, out_features) >= 2 * r ({2 * r}), "
                    f"got shape ({d_out}, {d_in})."
                )
            u, _, v = _svd_lowrank(grad_f32, 2 * r, 4)
            lora_a, lora_b = _loraga_direction_factors(u, v.mT, r, direction, weight.device)
            lora_a, lora_b = _loraga_rescale(lora_a, lora_b, weight, r, ga_cfg, spec.scaling)
            target_dtype = layer.weight.dtype
            lora_a_cast = lora_a.to(target_dtype)
            lora_b_cast = lora_b.to(target_dtype)
            residual = weight - spec.scaling * (
                lora_b_cast.to(lucid.float32) @ lora_a_cast.to(lucid.float32)
            )
            _install(layer, adapter, lora_a_cast, lora_b_cast, residual)
            base_layer.__dict__.pop("_peft_loraga_grad", None)
            base_layer.__dict__.pop("_peft_loraga_config", None)


def _check_convertible(trained: LoraConfig, initial: LoraConfig) -> None:
    """Raise ``ValueError`` unless ``trained`` can become a plain LoRA against ``initial``."""
    if trained.use_dora:
        raise ValueError(
            "path_initial_model_for_weight_conversion cannot convert a DoRA adapter: its "
            "magnitude vector has no place in a plain LoRA"
        )
    if trained.use_rslora and (trained.rank_pattern or trained.alpha_pattern):
        raise ValueError(
            "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not "
            "supported when using `rank_pattern` or `alpha_pattern` at the same time as "
            "`use_rslora=True`."
        )
    for name in ("r", "lora_alpha", "use_rslora", "rank_pattern", "alpha_pattern"):
        if getattr(trained, name) != getattr(initial, name):
            raise ValueError(
                f"the initial adapter does not match the trained one: {name} is "
                f"{getattr(initial, name)!r}, the trained adapter has {getattr(trained, name)!r}"
            )


def _doubled(config: LoraConfig) -> LoraConfig:
    r"""``config`` at twice the rank with the same scaling, initialised plainly.

    ``lora_alpha`` doubles with ``r``; under ``use_rslora`` (scaling
    :math:`\alpha/\sqrt r`) it grows by :math:`\sqrt 2` instead.
    """
    gain = 2**0.5 if config.use_rslora else 2
    return dataclasses.replace(
        config,
        r=2 * config.r,
        lora_alpha=config.lora_alpha * gain,
        rank_pattern={key: 2 * rank for key, rank in config.rank_pattern.items()},
        alpha_pattern={key: 2 * alpha for key, alpha in config.alpha_pattern.items()},
        init_lora_weights=True,
    )


def _rank_axis(key: str) -> int | None:
    """The axis ``key``'s factor grows along when a rank is added, or ``None`` if ``key`` is no factor weight."""
    *_, table, leaf = ("", "", *key.split("."))
    return _RANK_AXIS.get(table) if leaf == "weight" else None


def _stacked(trained: dict[str, Tensor], initial: dict[str, Tensor]) -> dict[str, Tensor]:
    r"""``trained`` with each factor widened by the initial one: :math:`[A;\,A_0]` and :math:`[B\;{-B_0}]`.

    What is not a factor (a trained bias, a ``modules_to_save`` copy) is kept.

    Raises
    ------
    ValueError
        The two hold factors for different layers, or of different shapes.
    """
    axes = {key: _rank_axis(key) for key in trained}
    factors = [key for key, axis in axes.items() if axis is not None]
    unmatched = sorted(set(factors) ^ {key for key in initial if _rank_axis(key) is not None})
    if unmatched:
        raise ValueError(
            "the initial adapter does not match the trained one: they hold LoRA factors "
            f"for different layers ({unmatched[:4]})"
        )
    stacked = dict(trained)
    for key in factors:
        now, start = trained[key], initial[key]
        if now.shape != start.shape:
            raise ValueError(
                f"the initial adapter does not match the trained one: {key} has shape "
                f"{tuple(start.shape)}, the trained adapter's is {tuple(now.shape)}"
            )
        start = start.to(device=now.device, dtype=now.dtype)
        axis = axes[key]
        stacked[key] = lucid.cat([now, start if axis == 0 else -start], dim=axis)
    return stacked


def _check_init(name: str, value: object) -> bool | str:
    """The canonical ``init_lora_weights``: ``True``, ``False`` or a lower-case name.

    Raises
    ------
    TypeError
        ``value`` is neither a bool nor a str.
    NotImplementedError, ValueError
        See ``_Init.of``.
    """
    if not isinstance(value, (bool, str)):
        raise TypeError(f"`{name}` must be a bool or a str, got {value!r}")
    return _Init.of(value).value


# The strategies are reached through ``_Init.of``, by the spelling each registers under.
__all__ = [
    "_Default",
    "_Eva",
    "_Factorising",
    "_Gaussian",
    "_Init",
    "_LoraGa",
    "_Olora",
    "_Pissa",
    "_Random",
    "_check_init",
]
