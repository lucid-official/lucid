"""``HRAConfig``: the HRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import PeftType, _Check, _check_bool, _check_rank, _choice, _TargetedConfig

_BIASES = ("none", "all", "hra_only")


@dataclass(kw_only=True)
class HRAConfig(_TargetedConfig, peft_type=PeftType.HRA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with HRA.

    HRA (Yuan et al., 2024, arXiv:2405.17484 — Householder Reflection Adaptation)
    multiplies each targeted weight from the right by a chain of :math:`r`
    Householder reflections,

    .. math:: W' = W\,\prod_{i=1}^{r}\bigl(I - 2\,\hat u_i \hat u_i^\top\bigr),
              \qquad \hat u_i = u_i / \lVert u_i \rVert,

    an orthogonal transform of the input space whose :math:`r` learned vectors
    :math:`u_i` are its only parameters (:math:`r\,d_{in}` per layer).  With an even
    ``r`` the vectors start in equal pairs, so the chain is the identity and the
    adapted model starts as the base; the transform folds into the weight.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    r : int, default=8
        Number of reflections.  Best even: an odd ``r`` cannot start at the identity.
    apply_GS : bool, default=False
        Orthonormalise the vectors with Gram-Schmidt before reflecting, so the
        chain is one :math:`I - 2UU^\top`.
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` modules to adapt, matched as
        ``LoraConfig.target_modules`` is.  ``None`` takes the default for the base
        model's ``model_type`` (the LoRA table, as the reference does).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    init_weights : bool, default=True
        ``True``: the vectors start in equal pairs (the identity).  ``False``: every
        vector Kaiming-uniform (for tests).
    layers_to_transform, layers_pattern, modules_to_save
        As in ``LoraConfig``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"hra_only"``; here the adapter
        trains its own vectors only, and those raise ``NotImplementedError``.

    Raises
    ------
    TypeError
        A field is not of its type.
    ValueError
        ``r`` is not positive.
    NotImplementedError
        ``bias`` asks for trained biases.

    Examples
    --------
    >>> from lucid.peft import HRAConfig
    >>> config = HRAConfig(r=4, apply_GS=True, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.HRA: 'HRA'>)
    >>> HRAConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "apply_GS": _check_bool,
        "init_weights": _check_bool,
        "bias": _choice(_BIASES),
    }

    r: int = 8
    apply_GS: bool = False
    init_weights: bool = True
    bias: Literal["none", "all", "hra_only"] = "none"

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: an HRA adapter in lucid.peft trains its own vectors only "
                "(bias='none')"
            )
