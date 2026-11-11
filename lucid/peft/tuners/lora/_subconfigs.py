"""The option dataclasses a ``LoraConfig`` field holds, and the one check that reads them.

``LoraConfig`` has a field per LoRA extension (``arrow_config``, ``kasa_config``, ``use_bdlora``,
``monteclora_config``, ``velora_config``, and the older ``eva_config`` / ``lora_ga_config``); each
holds the extension's own options as a small dataclass defined here and in ``_config``.  A file
writes such a field as a plain dict, so ``_subconfig`` turns the dict back into the dataclass.
The extensions themselves are not implemented: a ``LoraConfig`` that sets one is refused
(``_LORA_UNSUPPORTED``) once its options have been read and validated.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, Protocol


class _Dataclass(Protocol):
    """What ``@dataclass`` leaves on a class: the table of its fields."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


def _subconfig[T: _Dataclass](config_class: type[T]) -> Callable[[str, object], T | None]:
    """A field check: ``None``, an instance of ``config_class``, or the dict of its options.

    A dict is the shape a config file holds.  Its keys for fields the dataclass does not take
    through its constructor (``init=False``: the ones a library fills in itself, which a file
    still records) are dropped; any other key the class lacks raises ``TypeError``.
    """
    derived = {name for name, f in config_class.__dataclass_fields__.items() if not f.init}

    def check(name: str, value: object) -> T | None:
        if value is None or isinstance(value, config_class):
            return value
        if isinstance(value, Mapping):
            return config_class(**{k: v for k, v in value.items() if k not in derived})
        raise TypeError(
            f"`{name}` must be a {config_class.__name__}, a dict, or None, got {value!r}"
        )

    return check


_VELORA_INITS: Final = ("batch_average", "batch_average_once", "random")


@dataclass
class VeloraConfig:
    r"""Options of VeLoRA, which keeps a compressed copy of each activation for the backward pass.

    VeLoRA (Miles et al., 2024, arXiv:2405.17991) trains with less memory by splitting each
    layer's input activation into ``num_groups`` sub-tokens along its depth and storing only a
    rank-1 projection of every group; the backward pass reconstructs the activation from it,
    scaled by ``scale``.  ``LoraConfig(velora_config=...)`` is refused: the extension is not
    implemented.

    Parameters
    ----------
    num_groups : int, default=64
        How many groups the input activation's depth is cut into before it is compressed.
    scale : float, default=1.0
        Multiplier applied to the reconstructed activations in the backward pass.
    init_type : {"batch_average", "batch_average_once", "random"}, default="batch_average"
        How the projection is set.  ``"batch_average"`` takes it from the batch's statistics in
        every forward pass; ``"batch_average_once"`` does so in the first pass only (cheaper, a
        less exact gradient); ``"random"`` draws it at random.

    Raises
    ------
    ValueError
        ``num_groups`` or ``scale`` is not positive, or ``init_type`` is not one of the three.

    Examples
    --------
    >>> from lucid.peft import VeloraConfig
    >>> config = VeloraConfig(num_groups=32, init_type="random")
    >>> config.num_groups, config.scale
    (32, 1.0)
    """

    num_groups: int = 64
    scale: float = 1.0
    init_type: Literal["batch_average", "batch_average_once", "random"] = "batch_average"

    def __post_init__(self) -> None:
        if self.num_groups <= 0:
            raise ValueError(f"`num_groups` should be positive, got {self.num_groups}.")
        if self.scale <= 0:
            raise ValueError(f"`scale` should be positive, got {self.scale}.")
        if self.init_type not in _VELORA_INITS:
            raise ValueError(
                f"Unsupported `init_type` {self.init_type!r}. Supported values are "
                f"{', '.join(map(repr, _VELORA_INITS))}."
            )


@dataclass
class ArrowConfig:
    r"""Options of Arrow, which routes each token to the best few of several LoRA adapters.

    Arrow (Ostapenko et al., 2024, arXiv:2405.11157) builds a library of task-specific LoRA
    adapters and, at every layer, scores them against the token by the principal direction of
    each one's update, combining the ``top_k`` best with softmax weights.  With ``use_gks``
    general-knowledge adapters are subtracted first (GenKnowSub).  ``task_adapter_names`` and
    ``gks_adapter_names`` are filled in by the function that assembles the library, never passed.
    ``LoraConfig(arrow_config=...)`` is refused: the extension is not implemented.

    Parameters
    ----------
    top_k : int, default=3
        How many adapters are combined for each token.
    router_temperature : float, default=1.0
        Softmax temperature of the combination weights; must be positive.
    use_gks : bool, default=False
        Subtract the general-knowledge adapters first (GenKnowSub).
    rng_seed : int, optional
        Seed of the sampling the router does; ``None`` leaves it non-deterministic.

    Raises
    ------
    ValueError
        ``top_k`` or ``router_temperature`` is not positive.

    Examples
    --------
    >>> from lucid.peft import ArrowConfig
    >>> config = ArrowConfig(top_k=2, router_temperature=0.5)
    >>> config.top_k, config.use_gks, config.task_adapter_names
    (2, False, None)
    """

    top_k: int = 3
    router_temperature: float = 1.0
    use_gks: bool = False
    task_adapter_names: list[str] | None = field(default=None, init=False)
    gks_adapter_names: list[str] | None = field(default=None, init=False)
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"`top_k` must be positive, got {self.top_k}.")
        if self.router_temperature <= 0:
            raise ValueError(
                f"`router_temperature` must be greater than 0, got {self.router_temperature}."
            )


@dataclass
class BdLoraConfig:
    r"""Options of block-diagonal LoRA, which makes chosen LoRA factors block-diagonal.

    In BD-LoRA a LoRA factor :math:`A` or :math:`B` is a block-diagonal matrix of ``nblocks``
    blocks, so the adapter splits across devices (or serving shards) without communication.
    ``target_modules_bd_a`` names the modules whose :math:`A` is block-diagonal (the query, key,
    value, up and gate projections) and ``target_modules_bd_b`` those whose :math:`B` is (the
    output and down projections).  ``LoraConfig(use_bdlora=...)`` is refused: the extension is
    not implemented.

    Parameters
    ----------
    target_modules_bd_a : list of str, optional
        Modules whose LoRA-A is block-diagonal; a pattern matches a module whose name contains it.
    target_modules_bd_b : list of str, optional
        Modules whose LoRA-B is block-diagonal, matched the same way.
    nblocks : int, default=1
        Blocks per block-diagonal matrix.  Set it to the sharding degree to serve with.
    match_strict : bool, default=True
        Require every targeted module to have a block-diagonal A or B and raise otherwise.
        ``False`` allows mixing plain LoRA and BD-LoRA layers.

    Raises
    ------
    ValueError
        A module is listed for both A and B.

    Examples
    --------
    >>> from lucid.peft import BdLoraConfig
    >>> config = BdLoraConfig(target_modules_bd_a=["q_proj"], target_modules_bd_b=["o_proj"], nblocks=2)
    >>> config.nblocks, config.match_strict
    (2, True)
    """

    target_modules_bd_a: list[str] | None = None
    target_modules_bd_b: list[str] | None = None
    nblocks: int = 1
    match_strict: bool = True

    def __post_init__(self) -> None:
        overlap = set(self.target_modules_bd_a or []) & set(self.target_modules_bd_b or [])
        if overlap:
            raise ValueError(
                f"Found overlapping modules in the block-diagonal target lists: {sorted(overlap)} "
                f"are in both {self.target_modules_bd_a} (A) and {self.target_modules_bd_b} (B)."
            )


@dataclass
class KasaConfig:
    r"""Options of KaSA, which gives the LoRA update a learned singular-value diagonal.

    KaSA (Wang et al., 2024, arXiv:2412.06071) reads a LoRA update as an SVD,
    :math:`\Delta W = B\,\mathrm{diag}(\sigma)\,A`, and adds two regularisers to the training
    loss: an L2 penalty on :math:`\sigma` weighted by ``beta`` and a semi-orthogonality penalty
    :math:`\lVert B^\top B - I\rVert_F + \lVert A A^\top - I\rVert_F` weighted by ``gamma``.
    ``LoraConfig(kasa_config=...)`` is refused: the extension is not implemented.

    Parameters
    ----------
    beta : float, default=1e-4
        Weight of the L2 penalty on the singular values; not negative.
    gamma : float, default=1e-3
        Weight of the orthogonality penalty on the factors; not negative.

    Raises
    ------
    ValueError
        ``beta`` or ``gamma`` is negative.

    Examples
    --------
    >>> from lucid.peft import KasaConfig
    >>> config = KasaConfig(beta=1e-3)
    >>> config.beta, config.gamma
    (0.001, 0.001)
    """

    beta: float = 0.0001
    gamma: float = 0.001

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError(f"`beta` must be non-negative, got {self.beta}.")
        if self.gamma < 0:
            raise ValueError(f"`gamma` must be non-negative, got {self.gamma}.")


@dataclass
class MontecloraConfig:
    r"""Options of MonteCLoRA, which averages the LoRA update over Monte Carlo samples.

    MonteCLoRA draws ``num_samples`` perturbed copies of the low-rank update per forward pass
    and mixes them with Dirichlet-sampled expert weights, so the adapter is trained as a
    variational posterior rather than a point estimate; ``kl_loss_weight`` and ``use_entropy``
    regularise it.  ``LoraConfig(monteclora_config=...)`` is refused: the extension is not
    implemented.

    Parameters
    ----------
    num_samples : int, default=8
        Monte Carlo samples drawn per forward pass; positive.
    use_entropy : bool, default=False
        Add an entropy term that keeps the sampled weights from collapsing onto one sample.
    dirichlet_prior : float, default=0.1
        Concentration of the Dirichlet prior the expert weights are sampled from; positive.
    sample_scaler : float, default=1e-4
        Scale of the perturbation each sample adds.
    kl_loss_weight : float, default=1e-5
        Weight of the KL-divergence term of the variational objective.
    buffer_size : int, default=150
        Size of the sampler's buffer; positive.

    Raises
    ------
    ValueError
        ``num_samples``, ``dirichlet_prior`` or ``buffer_size`` is not positive.

    Examples
    --------
    >>> from lucid.peft import MontecloraConfig
    >>> config = MontecloraConfig(num_samples=4)
    >>> config.num_samples, config.buffer_size
    (4, 150)
    """

    num_samples: int = 8
    use_entropy: bool = False
    dirichlet_prior: float = 0.1
    sample_scaler: float = 0.0001
    kl_loss_weight: float = 1e-05
    buffer_size: int = 150

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError(f"`num_samples` must be greater than 0, got {self.num_samples}.")
        if self.dirichlet_prior <= 0:
            raise ValueError(
                f"`dirichlet_prior` must be greater than 0, got {self.dirichlet_prior}."
            )
        if self.buffer_size <= 0:
            raise ValueError(f"`buffer_size` must be greater than 0, got {self.buffer_size}.")
