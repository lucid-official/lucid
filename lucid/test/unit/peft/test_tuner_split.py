"""The tuner base lives in four concern modules behind one facade, and is generic in its config.

``lucid.peft.tuners._base`` re-exports the layer base, the planning helpers, the host and the
generic tuner, so an import written against the old single module still names the same objects
(``_TUNERS`` above all: a copy would split the registry in two).  ``_BaseTuner[C]`` derives
``_config_class`` from the class it is parametrised by, and a tuner that names ``_config_class``
itself keeps doing so.
"""

import ast
import inspect
import types

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import HRAConfig, LoraConfig, LoraModel, PeftType
from lucid.peft.tuners import _base, _host, _layer, _plan, _tuner
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.hra._layer import HRALayer

_HOMES = (_layer, _plan, _host, _tuner)


class TestFacade:
    def test_every_name_is_the_object_its_home_module_defines(self) -> None:
        for name in _base.__all__:
            owners = [module for module in _HOMES if hasattr(module, name)]
            assert owners, f"{name} is defined in none of the concern modules"
            assert all(getattr(module, name) is getattr(_base, name) for module in owners)

    def test_the_facade_lists_every_name_it_holds(self) -> None:
        held = {name for name in vars(_base) if not name.startswith("__")}
        assert set(_base.__all__) == held

    def test_the_registry_is_one_object(self) -> None:
        assert _base._TUNERS is _host._TUNERS
        assert _base._TUNERS[PeftType.LORA] is LoraModel

    def test_a_concern_module_imports_only_earlier_ones(self) -> None:
        names = [module.__name__.rpartition(".")[2] for module in _HOMES]
        for position, module in enumerate(_HOMES):
            tree = ast.parse(inspect.getsource(module))
            imported = {
                node.module.rpartition(".")[2]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
                if node.module.startswith("lucid.peft.tuners._")
            }
            assert imported <= set(names[:position]), (
                f"{names[position]} imports {sorted(imported - set(names[:position]))}; "
                "the order is layer -> plan -> host -> tuner"
            )


class _Clone(_ConfigTuner[HRAConfig]):
    """HRA's tuner by parametrisation alone: no ``_config_class``, and not registered."""

    _layer_types = HRALayer._layer_types


class TestConfigClassDerivation:
    def test_a_parametrised_subclass_gets_its_config_class(self) -> None:
        assert _Clone._config_class is HRAConfig

    def test_the_base_parametrised_directly_is_derived_too(self) -> None:
        class Direct(_tuner._BaseTuner[LoraConfig]):
            pass

        assert Direct._config_class is LoraConfig

    def test_a_subclass_of_a_derived_class_inherits_it(self) -> None:
        class Below(_Clone):
            pass

        assert Below._config_class is HRAConfig

    def test_an_explicit_config_class_still_works(self) -> None:
        assert LoraModel._config_class is LoraConfig
        assert _TunerNamingItself._config_class is HRAConfig

    def test_an_explicit_config_class_agreeing_with_the_parameter_is_fine(self) -> None:
        class Both(_ConfigTuner[HRAConfig]):
            _config_class = HRAConfig

        assert Both._config_class is HRAConfig

    def test_an_explicit_config_class_disagreeing_with_the_parameter_is_refused(self) -> None:
        with pytest.raises(TypeError, match="parametrised by HRAConfig but names"):

            class Both(_ConfigTuner[HRAConfig]):
                _config_class = LoraConfig

    def test_a_parameter_that_is_not_a_config_is_refused(self) -> None:
        wrong = types.GenericAlias(_tuner._BaseTuner, (nn.Linear,))
        with pytest.raises(TypeError, match="Linear, which is not a _TargetedConfig"):
            types.new_class("Wrong", (wrong,))

    def test_a_type_variable_names_nothing(self) -> None:
        class Open[C: LoraConfig](_tuner._BaseTuner[C]):
            pass

        assert "_config_class" not in vars(Open)

    def test_deriving_registers_nothing(self) -> None:
        before = dict(_host._TUNERS)

        class Quiet(_ConfigTuner[HRAConfig]):
            pass

        assert _host._TUNERS == before
        assert Quiet not in _host._TUNERS.values() and _Clone not in _host._TUNERS.values()


class _TunerNamingItself(_tuner._BaseTuner):
    """The bare base keeps its default ``C`` and says what it takes itself."""

    _config_class = HRAConfig


class TestDerivedTunerWorks:
    def test_it_adapts_a_model_and_takes_the_adapter_back_out(self, device: str) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)
        tuner = _Clone(net, HRAConfig(r=2, target_modules=["0"]), "default")
        assert tuner.targeted_module_names == ["0"]
        assert isinstance(net[0], HRALayer)
        assert tuner.unload() is net
        assert type(net[0]) is nn.Linear

    def test_another_methods_config_is_refused_by_name(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 3)).to(device)
        with pytest.raises(TypeError, match="config must be a HRAConfig, got LoraConfig"):
            _Clone(net, LoraConfig(r=2, target_modules=["0"]), "default")
        assert type(net[0]) is nn.Linear
