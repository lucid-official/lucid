"""``RoadConfig``: the RoAd hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import PeftType, _Check, _check_bool, _check_rank, _choice, _TargetedConfig


@dataclass
class RoadConfig(_TargetedConfig, peft_type=PeftType.ROAD):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with RoAd.

    RoAd (Liao and Monz, 2024, arXiv:2409.00119 — 2D Rotary Adaptation) pairs the output features of a
    layer into 2-D vectors and rotates and scales each pair,

    .. math:: \begin{pmatrix} y_i' \\ y_j' \end{pmatrix} = \alpha \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} y_i \\ y_j \end{pmatrix} .

    Applied to the output, the rotation folds into the weight and the bias, so the adapter costs nothing at
    inference; the variants ``road_1``, ``road_2`` and ``road_4`` learn one, two or four times as many numbers per
    layer, as many as the layer's output width in the smallest.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    variant : {"road_1", "road_2", "road_4"}, default="road_1"
        ``"road_1"``: one scale and angle per pair.  ``"road_2"``: one per element (twice the
        parameters).  ``"road_4"``: two scales and angles per element (four times).
    group_size : int, default=64
        Features are grouped by this size into pairs: within a group, element ``i`` is paired
        with element ``i + group_size / 2``.  Even; the output width must be divisible by it.
    init_weights : bool, default=True
        Use the default initialisation (no rotation, unit scale).
    target_modules : str or list of str, optional
        The modules RoAd adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import RoadConfig
    >>> config = RoadConfig(variant='road_2', group_size=32, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.ROAD: 'ROAD'>)
    >>> RoadConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    RoAd's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``variant`` is unknown, or ``group_size`` is not a
    positive even number.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-825, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "variant": _choice(("road_1", "road_2", "road_4")),
        "group_size": _check_rank,
        "init_weights": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.group_size % 2 != 0:
            raise ValueError(
                f"The group_size must be divisible by 2 when using RoadLayer, but got {self.group_size}."
            )

    variant: Literal["road_1", "road_2", "road_4"] = "road_1"
    group_size: int = 64
    init_weights: bool = True
    target_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
