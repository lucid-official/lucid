"""``PeanutConfig``: the PEANuT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_number,
    _check_rank,
    _TargetedConfig,
)
from lucid.peft.tuners.peanut._activations import _check_act_fn


@dataclass
class PeanutConfig(_TargetedConfig, peft_type=PeftType.PEANUT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with PEANuT.

    PEANuT (Zhong et al., 2024, arXiv:2410.01870, section 3.3; later versions of the paper call it
    Neat) writes the update as a function of the frozen weight, :math:`W' = W + f(W; \theta)`, where
    :math:`f` is a small trained network, not a fixed low-rank product.  At depth zero
    :math:`f(W) = s\,\sigma(W^\top A^\top)\,B` with :math:`A \in \mathbb{R}^{r \times d_{out}}` and
    :math:`B \in \mathbb{R}^{d_{out} \times r}` (transposed back to the shape of :math:`W`); each
    unit of ``depth`` puts an encoder layer :math:`r \to r` before and a decoder layer after the
    middle, joined by residuals (appendix F), with the non-linearity ``act_fn`` between all of
    them.  :math:`B` is zero when training begins, which makes the first update zero too.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Hidden width of the adapter network.
    depth : int, default=0
        Hidden layers per side, beyond the input and output projections; not negative.
    act_fn : str, default="relu"
        Name of the non-linearity between the layers: ``"relu"``, ``"gelu"``, ``"tanh"``,
        ``"silu"``, ``"leaky_relu"``, ``"mish"``, ``"sigmoid"``, ``"linear"`` or another name of the
        reference's activation table (an unknown name raises ``ValueError``; ``"xielu"``, whose slopes
        are parameters of the activation, raises ``NotImplementedError``).
    scaling : float, default=1.0
        Multiplier on the adapter's output before it is added to the base output.
    init_weights : bool, default=True
        ``True``: the output projection starts at zero.  ``False``: all weights start random.
    target_modules : str or list of str, optional
        The modules PEANuT adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules PEANuT never adapts, matched like ``target_modules``.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import PeanutConfig
    >>> config = PeanutConfig(r=8, depth=1, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.PEANUT: 'PEANUT'>)
    >>> PeanutConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, read
    and written as ``adapter_config.json``.  A field of the wrong type raises ``TypeError``.
    ``ValueError`` is raised when ``r`` is not positive, ``depth`` is negative or ``act_fn`` names no
    activation.  The tuner is :class:`~lucid.peft.PeanutModel`; it adapts ``nn.Linear`` modules.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "depth": _check_count,
        "act_fn": _check_act_fn,
        "scaling": _check_number,
        "init_weights": _check_bool,
    }

    r: int = 32
    depth: int = 0
    act_fn: str = "relu"
    scaling: float = 1.0
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    init_weights: bool = True
