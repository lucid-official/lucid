"""The weight-conversion pipeline refuses what it cannot account for.

Two layers: the shared engine (``tools/convert_weights/_base.py``) refuses
a source key no recipe rule covers, and the TVM recipe's unpickler reads
the release format — a compile wrapper around persistence records —
without running anything the file names.  No download: every checkpoint
here is built in the test.
"""

import importlib
import pickle
import pkgutil
import sys
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import lucid.nn as nn
from lucid.models import AutoConfig
from lucid.test._fixtures.ref_framework import require_ref

require_ref(module_level=True)
pytest.importorskip("huggingface_hub")

from tools.convert_weights import tvm as convert_tvm  # noqa: E402
from tools.convert_weights._base import (  # noqa: E402
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    convert,
)

# What a persistence record carries as its module's source.  The
# unpickler must never run it.
_POISON = "raise SystemExit('a checkpoint module source ran')"


# ── the shared engine ────────────────────────────────────────────────────────


class _LinearRecipe(Architecture):
    """Two known keys, one deliberate drop, everything else unknown."""

    def __init__(self, source: dict[str, object]) -> None:
        self.source = source

    def source_state_dict(self) -> dict[str, object]:
        return self.source

    def target_model(self) -> nn.Module:
        return nn.Linear(2, 2)

    def map_key(self, src_key: str) -> str | None:
        if src_key == "w":
            return "weight"
        if src_key == "b":
            return "bias"
        if src_key == "dropped":
            return None
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        return ConversionSpec(
            model_name="linear",
            architecture="linear",
            repo_id="test/linear",
            tag="T",
            task="base",
            model_type="linear",
            source="test",
            license="mit",
            num_classes=2,
            config={},
            preprocessing={},
            citation="",
            title="Linear",
        )


def _linear_source(**extra: object) -> dict[str, object]:
    source: dict[str, object] = {
        "w": np.ones((2, 2), dtype=np.float32),
        "b": np.zeros(2, dtype=np.float32),
    }
    source.update(extra)
    return source


def test_a_deliberately_dropped_key_converts() -> None:
    state, _ = convert(_LinearRecipe(_linear_source(dropped=np.ones(3))))
    assert sorted(state) == ["bias", "weight"]


def test_every_unknown_source_key_is_refused_together() -> None:
    recipe = _LinearRecipe(_linear_source(zeta=np.ones(1), alpha=np.ones(1)))
    with pytest.raises(RuntimeError, match=r"2 source key\(s\).*'alpha', 'zeta'"):
        convert(recipe)


# ── every registered recipe ──────────────────────────────────────────────────


def _recipe_classes() -> list[type[Architecture]]:
    import tools.convert_weights as package

    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"tools.convert_weights.{info.name}")
    found: list[type[Architecture]] = []
    pending = list(Architecture.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("tools.convert_weights."):
            found.append(cls)
    return sorted(found, key=lambda c: (c.__module__, c.__name__))


# What a recipe's ``map_key`` reads off itself before it decides anything;
# built for real, each would download its checkpoint.
_BARE_STATE: dict[str, dict[str, object]] = {
    "BERTArch": {"_kind": "mlm"},
    "SENetArch": {"arch": ""},
    "MobileNetV3Arch": {"_key_map": {}},
}


@pytest.mark.parametrize("recipe", _recipe_classes(), ids=lambda c: c.__name__)
def test_no_recipe_drops_a_key_it_has_no_rule_for(recipe: type[Architecture]) -> None:
    """LCD-630: a recipe that fell through to ``return None`` dropped any key
    it never decided about, and the conversion succeeded without it.  A key
    no rule covers must raise :class:`UnknownSourceKey`, or carry through
    under a name that :func:`convert`'s key-set check then refuses."""
    bare = recipe.__new__(recipe)
    for name, value in _BARE_STATE.get(recipe.__name__, {}).items():
        setattr(bare, name, value)
    try:
        mapped = bare.map_key("model.unexpected.weight")
    except UnknownSourceKey:
        return
    assert mapped is not None


# ── the TVM release format ───────────────────────────────────────────────────


class _Call:
    """Pickles as a call of ``function`` on ``args`` — a global the file names."""

    def __init__(self, function: object, *args: object) -> None:
        self.function = function
        self.args = args

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (self.function, self.args)


class _Persistent:
    """Pickles the way a layer written by the authors' persistence scheme does.

    Its reduce is ``(_reconstruct_persistent_obj, (meta,))`` with the
    layer's attributes under ``meta["state"]`` and its module's source
    under ``meta["module_src"]``.
    """

    def __init__(self, hook: Callable[[object], object], **attrs: object) -> None:
        children = {k: v for k, v in attrs.items() if not isinstance(v, (bool, int))}
        self.hook = hook
        self.state: dict[str, object] = {
            "training": False,
            "_parameters": OrderedDict(),
            "_buffers": OrderedDict(),
            "_modules": children,
            **{k: v for k, v in attrs.items() if k not in children},
        }

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        meta = {
            "type": "class",
            "version": 6,
            "module_src": _POISON,
            "class_name": "Layer",
            "state": self.state,
        }
        return (self.hook, (meta,))


@pytest.fixture
def hook(
    ref: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Callable[[object], object]:
    """The persistence hook, importable under the name the release uses.

    Calling it fails the test: the unpickler must substitute its own.
    """
    package = ref.__name__ + "_utils"
    module = ModuleType(package + ".persistence")

    def _reconstruct_persistent_obj(meta: object) -> object:
        raise AssertionError("the unpickler called the checkpoint's own hook")

    _reconstruct_persistent_obj.__module__ = module.__name__
    _reconstruct_persistent_obj.__qualname__ = "_reconstruct_persistent_obj"
    setattr(module, "_reconstruct_persistent_obj", _reconstruct_persistent_obj)
    monkeypatch.setitem(sys.modules, package, ModuleType(package))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return _reconstruct_persistent_obj


def _release(
    ref: ModuleType,
    hook: Callable[[object], object],
    *,
    num_heads: int = 18,
    do_mod_norm: bool = True,
    do_scale_param: bool = True,
) -> tuple[_Call, dict[str, object]]:
    """A compile-wrapped, persistence-written wrapper — the release's shape.

    Returns the object to pickle and the tensors it holds, by the names
    the converter must recover.
    """
    qkv = ref.nn.Linear(4, 12)
    modulation = ref.nn.Sequential(ref.nn.SiLU(), ref.nn.Linear(4, 24))
    final = ref.nn.Linear(4, 4)
    attn = _Persistent(hook, qkv=qkv, num_heads=num_heads)
    block = _Persistent(
        hook, attn=attn, adaLN_modulation=modulation, do_mod_norm=do_mod_norm
    )
    network = _Persistent(
        hook,
        blocks=_Persistent(hook, **{"0": block}),
        final_layer=_Persistent(hook, linear=final, do_mod_norm=do_mod_norm),
        num_heads=num_heads,
        learn_guidance=True,
    )
    wrapper = _Persistent(hook, model=network, do_scale_param=do_scale_param)
    wrapper.state["dtype"] = ref.bfloat16
    compiled = _Call(ref._dynamo.eval_frame.OptimizedModule, wrapper, None)
    tensors = {
        "model.blocks.0.attn.qkv.weight": qkv.weight,
        "model.blocks.0.attn.qkv.bias": qkv.bias,
        "model.blocks.0.adaLN_modulation.1.weight": modulation[1].weight,
        "model.blocks.0.adaLN_modulation.1.bias": modulation[1].bias,
        "model.final_layer.linear.weight": final.weight,
        "model.final_layer.linear.bias": final.bias,
    }
    return compiled, tensors


def _dump(tmp_path: Path, net: object) -> str:
    path = tmp_path / "checkpoint.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"net": net}, handle)
    return str(path)


def _arch_reading(path: str, monkeypatch: pytest.MonkeyPatch) -> convert_tvm.TVMArch:
    monkeypatch.setattr(convert_tvm, "hf_hub_download", lambda repo, name: path)
    return convert_tvm.TVMArch("tvm_imagenet256", "IMAGENET1K_256_W2")


def test_a_compiled_module_yields_its_tensors(ref: ModuleType, tmp_path: Path) -> None:
    # LCD-619: the compile wrapper's state was dropped, leaving 0 tensors.
    linear = ref.nn.Linear(2, 2)
    for net in (linear, ref.compile(linear)):
        state, _, _ = convert_tvm._load_ema(_dump(tmp_path, net))
        assert sorted(state) == ["bias", "weight"]
        np.testing.assert_array_equal(state["weight"], linear.weight.detach().numpy())


def test_the_release_format_loads_without_running_its_code(
    ref: ModuleType, hook: Callable[[object], object], tmp_path: Path
) -> None:
    net, tensors = _release(ref, hook)
    state, wrapper, network = convert_tvm._load_ema(_dump(tmp_path, net))
    assert sorted(state) == sorted(tensors)
    for key, tensor in tensors.items():
        np.testing.assert_array_equal(state[key], tensor.detach().numpy())
    assert wrapper["do_scale_param"] is True
    assert network["num_heads"] == 18


def test_the_release_flags_match_the_factory(
    ref: ModuleType,
    hook: Callable[[object], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net, tensors = _release(ref, hook)
    state = _arch_reading(_dump(tmp_path, net), monkeypatch).source_state_dict()
    assert sorted(state) == sorted(tensors)


@pytest.mark.parametrize(
    ("num_heads", "do_mod_norm", "do_scale_param", "message"),
    [
        (16, True, True, "num_heads"),
        (18, False, True, "do_mod_norm"),
        (18, True, False, "do_scale_param=False"),
    ],
)
def test_a_release_built_differently_is_refused(
    ref: ModuleType,
    hook: Callable[[object], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    do_mod_norm: bool,
    do_scale_param: bool,
    message: str,
) -> None:
    net, _ = _release(
        ref,
        hook,
        num_heads=num_heads,
        do_mod_norm=do_mod_norm,
        do_scale_param=do_scale_param,
    )
    arch = _arch_reading(_dump(tmp_path, net), monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        arch.source_state_dict()


def test_a_tensorless_checkpoint_is_refused_before_its_flags(
    ref: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It used to reach the flag check and blame do_scale_param.
    path = _dump(tmp_path, ref.compile(ref.nn.Identity()))
    with pytest.raises(RuntimeError, match="no tensors"):
        _arch_reading(path, monkeypatch).source_state_dict()


def _calling(module: str, name: str, *args: object) -> bytes:
    """``{"net": module.name(*args)}`` as pickle bytes, importing nothing.

    The GLOBAL opcode names the callable as text, so the test does not
    depend on the callable existing, or living where it does today.
    """
    arguments = pickle.dumps(args, protocol=3)[2:-1]  # the tuple, no PROTO / STOP
    target = f"{module}\n{name}\n".encode()
    return b"\x80\x03}X\x03\x00\x00\x00netc" + target + arguments + b"Rs."


def _disallowed(ref_name: str) -> dict[str, bytes]:
    # A nested storage whose own pickle names a global: the weights-only
    # loader, not the outer table, has to refuse it.
    nested = b"\x80\x02cos\ngetcwd\n)R."
    return {
        "eval": _calling("builtins", "eval", "1 + 1"),
        "getcwd": _calling("os", "getcwd"),
        "numpy runstring": _calling(
            "numpy.testing._private.utils", "runstring", "x = 1", {}
        ),
        "unrestricted load": _calling(
            f"{ref_name}.serialization", "load", "/nonexistent.pt"
        ),
        "nested storage naming getcwd": _calling(
            f"{ref_name}.storage", "_load_from_bytes", nested
        ),
    }


@pytest.mark.parametrize("name", sorted(_disallowed("ref")))
def test_a_global_outside_the_tables_is_refused(
    ref: ModuleType, tmp_path: Path, name: str
) -> None:
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(_disallowed(ref.__name__)[name])
    with pytest.raises(pickle.UnpicklingError):
        convert_tvm._load_ema(str(path))


def test_an_old_framework_reads_no_storage(
    ref: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Before 2.6 the weights-only loader could be bypassed (CVE-2025-32434).
    path = _dump(tmp_path, ref.nn.Linear(2, 2))
    monkeypatch.setattr(ref, "__version__", "2.5.1")
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        convert_tvm._load_ema(path)


@pytest.mark.parametrize(
    ("version", "release"),
    [
        ("2.5.1", (2, 5)),
        ("2.6.0+cpu", (2, 6)),
        ("2.10.0a0+git1234", (2, 10)),
        ("2.13.0", (2, 13)),
    ],
)
def test_a_release_is_compared_as_numbers(
    version: str, release: tuple[int, int]
) -> None:
    assert convert_tvm._release_of(version) == release
    assert (convert_tvm._release_of(version) >= (2, 6)) == (release >= (2, 6))


def test_the_key_map_refuses_what_the_release_does_not_hold() -> None:
    arch = convert_tvm.TVMArch("tvm_imagenet256", "IMAGENET1K_256_W2")
    assert arch.map_key("model.blocks.3.attn.qkv.weight") == "blocks.3.attn.qkv.weight"
    assert arch.map_key("model.final_layer.linear.bias") == "final.proj.bias"
    assert arch.map_key("model.pos_embed") is None
    for key in (
        "model.blocks.3.attn.extra.weight",
        "model.blocks.x.attn.qkv.weight",
        "model.unknown.weight",
        "sigma_data",
    ):
        with pytest.raises(UnknownSourceKey):
            arch.map_key(key)


def test_each_release_config_is_the_registered_one_plus_its_overrides() -> None:
    for tag, release in convert_tvm._RELEASES.items():
        config = convert_tvm.TVMArch(release.factory, tag)._config()
        registered = AutoConfig.from_pretrained(release.factory)
        for field, value in vars(registered).items():
            assert getattr(config, field) == release.overrides.get(field, value), field
