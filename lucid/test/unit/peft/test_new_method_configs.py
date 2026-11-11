"""The configs of the twenty methods that have no adapter yet: the reference's fields, checked, and a file.

Each config is the reference's dataclass — fields, order, defaults (``test_reference_signatures`` holds
that) — and validates what the reference's constructor validates, so a script or an adapter file that
was wrong for the reference is wrong here too, and a right one reads.  What is held per config:

* its default writes the reference's keys and no others;
* it round-trips through ``adapter_config.json`` and ``from_dict``;
* it names its own method, and refuses another's name;
* each check the reference's ``__post_init__`` makes raises here.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid.peft as peft
from lucid.peft import PeftConfig, PeftType
from lucid.peft._config import _CONFIG_CLASSES

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: The twenty methods without a working tuner, by the name of their config class.
_NAMES = [
    "AdamssConfig",
    "BeftConfig",
    "CartridgeConfig",
    "DeftConfig",
    "DeloraConfig",
    "FrodConfig",
    "GloraConfig",
    "GraloraConfig",
    "HiraConfig",
    "LilyConfig",
    "OSFConfig",
    "PeanutConfig",
    "PsoftConfig",
    "PveraConfig",
    "RoadConfig",
    "ShadowConfig",
    "SupertuningConfig",
    "TinyLoraConfig",
    "UniLoraConfig",
    "WaveFTConfig",
]
_CLASSES: list[type[PeftConfig]] = [getattr(peft, name) for name in _NAMES]


def test_every_config_without_a_working_tuner_is_among_these() -> None:
    """The refused method still has its config here."""
    from lucid.peft.tuners._base import _REFUSED_TUNERS

    assert set(_REFUSED_TUNERS) <= {cls._peft_type for cls in _CLASSES}


@pytest.mark.parametrize("config_class", _CLASSES, ids=_NAMES)
class TestEveryConfig:
    def test_the_default_writes_the_reference_keys(self, config_class: type[PeftConfig]) -> None:
        reference = {row[0] for row in _SNAPSHOT["fields"][config_class.__name__]}
        assert set(config_class().to_dict()) == reference - {"peft_version"}

    def test_it_is_registered_under_its_method(self, config_class: type[PeftConfig]) -> None:
        peft_type = config_class._peft_type
        assert peft_type is not None
        assert _CONFIG_CLASSES[peft_type] is config_class
        assert config_class().peft_type == peft_type

    def test_it_refuses_another_methods_name(self, config_class: type[PeftConfig]) -> None:
        other = "LORA" if config_class._peft_type != PeftType.LORA else "IA3"
        with pytest.raises(ValueError, match="peft_type"):
            config_class(peft_type=other)

    def test_it_round_trips_through_its_file(
        self, config_class: type[PeftConfig], tmp_path: Path
    ) -> None:
        config = config_class(task_type="CAUSAL_LM", base_model_name_or_path="base")
        config.save_pretrained(tmp_path)
        assert PeftConfig.from_pretrained(tmp_path) == config
        assert config_class.from_dict(json.loads(json.dumps(config.to_dict()))) == config

    def test_a_wrong_type_is_a_type_error(self, config_class: type[PeftConfig]) -> None:
        build: Callable[..., PeftConfig] = config_class
        with pytest.raises(TypeError):
            build(inference_mode="yes")

    def test_every_field_has_a_default(self, config_class: type[PeftConfig]) -> None:
        for f in dataclasses.fields(config_class):
            assert (
                f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            ), f.name


# ── the checks of each constructor ───────────────────────────────────────────

#: (config, fields that must be refused, the exception).  Each is a check the reference makes.
_REFUSED: list[tuple[str, dict[str, object], type[Exception]]] = [
    ("DeloraConfig", {"r": 0}, ValueError),
    ("DeloraConfig", {"module_dropout": 1.0}, ValueError),
    ("DeloraConfig", {"bias": "everything"}, ValueError),
    ("DeloraConfig", {"layers_pattern": "h"}, ValueError),
    ("DeloraConfig", {"target_modules": "q", "layers_to_transform": [0]}, ValueError),
    ("GraloraConfig", {"r": 8, "gralora_k": 3}, ValueError),
    ("GraloraConfig", {"gralora_k": 0}, ValueError),
    ("GraloraConfig", {"hybrid_r": -1}, ValueError),
    ("HiraConfig", {"init_weights": "uniform"}, ValueError),
    ("GloraConfig", {"config_A_B": "matrix"}, ValueError),
    ("GloraConfig", {"config_C": "constant"}, ValueError),
    ("GloraConfig", {"config_D_E": "lora"}, ValueError),
    ("DeftConfig", {"decomposition_method": "svd"}, ValueError),
    ("PeanutConfig", {"depth": -1}, ValueError),
    ("PeanutConfig", {"r": 0}, ValueError),
    ("UniLoraConfig", {"theta_d_length": 0}, ValueError),
    ("UniLoraConfig", {"init_theta_d_bound": 0.0}, ValueError),
    ("UniLoraConfig", {"bias": "unilora"}, ValueError),
    ("PveraConfig", {"r": 0}, ValueError),
    ("TinyLoraConfig", {"weight_tying": 1.5}, ValueError),
    ("TinyLoraConfig", {"u": 0}, ValueError),
    ("TinyLoraConfig", {"init_weights": "normal"}, ValueError),
    ("LilyConfig", {"num_B": 1}, ValueError),
    ("LilyConfig", {"stride_A": 0}, ValueError),
    ("RoadConfig", {"variant": "road_3"}, ValueError),
    ("RoadConfig", {"group_size": 63}, ValueError),
    ("RoadConfig", {"group_size": 0}, ValueError),
    ("PsoftConfig", {"ab_svd_init": "random"}, ValueError),
    ("PsoftConfig", {"psoft_svd": "sparse"}, ValueError),
    ("PsoftConfig", {"use_cayley_neumann": True, "num_cayley_neumann_terms": 0}, ValueError),
    ("PsoftConfig", {"use_cayley_neumann": True, "cayley_neumann_eps": 1.5}, ValueError),
    ("OSFConfig", {"effective_rank": "half"}, TypeError),
    ("OSFConfig", {"rank_pattern": {"q": "full"}}, TypeError),
    ("FrodConfig", {"sparse_rate": 1.5}, ValueError),
    ("SupertuningConfig", {"sparsity": 1.0}, ValueError),
    ("SupertuningConfig", {"lora_alpha": 4.0}, ValueError),
    ("SupertuningConfig", {"r": 0}, ValueError),
    ("WaveFTConfig", {"n_frequency": 0}, ValueError),
    ("WaveFTConfig", {"wavelet_family": 3}, TypeError),
    ("BeftConfig", {"init_weights": "zeros"}, TypeError),
    ("CartridgeConfig", {"num_virtual_tokens": 4, "num_frozen_tokens": 5}, ValueError),
    ("CartridgeConfig", {"num_frozen_tokens": -1}, ValueError),
    ("AdamssConfig", {"init_weights": "random"}, ValueError),
    ("AdamssConfig", {"use_asa": True, "init_warmup": 10, "final_warmup": 5}, ValueError),
    ("AdamssConfig", {"use_asa": True, "mask_interval": 0}, ValueError),
    ("AdamssConfig", {"use_asa": True, "asa_target_subspaces": 0}, ValueError),
    ("AdamssConfig", {"use_asa": True, "asa_schedule_exponent": 0.0}, ValueError),
    ("ShadowConfig", {"shadow_model": "elsewhere", "shadow_hidden_size": 8}, ValueError),
    ("ShadowConfig", {"r": 0}, ValueError),
]


@pytest.mark.parametrize(
    ("name", "fields", "error"),
    _REFUSED,
    ids=[f"{name}-{'-'.join(map(str, fields))}" for name, fields, _ in _REFUSED],
)
def test_a_value_the_reference_refuses_is_refused(
    name: str, fields: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        getattr(peft, name)(**fields)


def test_every_config_has_a_refusal_row() -> None:
    assert {name for name, _, _ in _REFUSED} == set(_NAMES)


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("HiraConfig", {"init_weights": "gaussian"}),
        ("HiraConfig", {"init_weights": None}),
        ("TinyLoraConfig", {"init_weights": "uniform"}),
        ("RoadConfig", {"variant": "road_4", "group_size": 8}),
        ("OSFConfig", {"effective_rank": 0.25, "rank_pattern": {"q": 4, "v": 0.5}}),
        ("PsoftConfig", {"cayley_neumann_eps": 1.5}),  # only read with use_cayley_neumann
        ("AdamssConfig", {"mask_interval": 0}),  # only read with use_asa
        ("ShadowConfig", {"shadow_model": "elsewhere"}),
        ("CartridgeConfig", {"num_virtual_tokens": 4, "num_frozen_tokens": 4}),
    ],
)
def test_a_value_the_reference_accepts_is_accepted(name: str, fields: dict[str, object]) -> None:
    config = getattr(peft, name)(**fields)
    assert type(config).from_dict(config.to_dict()) == config


def test_supertuning_takes_twice_the_rank_as_alpha_when_it_has_none() -> None:
    config = peft.SupertuningConfig(r=4)
    assert config.lora_alpha == 8
    assert peft.SupertuningConfig(r=4, lora_alpha=3.0).lora_alpha == 3.0
    assert peft.SupertuningConfig().lora_alpha is None


def test_adamss_reads_false_as_no_initialisation() -> None:
    build: Callable[..., PeftConfig] = peft.AdamssConfig
    assert build(init_weights=False).init_weights is None
    assert peft.AdamssConfig().init_weights == "orthogonal"


def test_the_targeting_fields_a_config_declares_are_checked() -> None:
    """``layers_pattern`` without ``layers_to_transform`` and a regex that does not compile fail at the config."""
    with pytest.raises(ValueError, match="layers_to_transform"):
        peft.LilyConfig(layers_pattern="layers")
    with pytest.raises(ValueError, match="regular expression"):
        peft.RoadConfig(target_modules="(")


def test_a_target_list_becomes_a_set_of_names() -> None:
    config = peft.DeloraConfig(target_modules=["v", "q"])
    assert config.target_modules == {"q", "v"}
    assert config.to_dict()["target_modules"] == ["q", "v"]
