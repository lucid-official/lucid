"""The dead-code collectors: vulture with its whitelist, and unbound engine bindings.

DS-15 / LCD-356 (the whitelist and decorators), DS-18 (both are gate
contract), DS-24 (engine bindings no production module reaches).
"""

from pathlib import Path

import pytest

from tools.quality_gate import config
from tools.quality_gate.collectors import all_collectors
from tools.quality_gate.collectors.bindings import (
    EngineBindingsCollector,
    stub_functions,
    unbound_bindings,
)
from tools.quality_gate.collectors.deadcode import VultureCollector
from tools.quality_gate.core import Scope, scope_of


def _collect_single_file(tmp_path: Path, source: str) -> dict[tuple[str, str, str], int]:
    pkg = tmp_path / "lucid"
    pkg.mkdir(parents=True, exist_ok=True)
    target = pkg / "sample.py"
    target.write_text(source)
    scope = Scope(root=tmp_path, py=["lucid/sample.py"], cpp=[], workdir=tmp_path)
    return VultureCollector().collect(scope)


@pytest.mark.parametrize(
    "decorator",
    [
        "@_emitter('add')",
        "@register_kl(int, int)",
        "@register_model('my_model')",
        "@register_weights(object)",
        "@_register_preset",
        "@_register(frozenset({'zeros'}))",
    ],
)
def test_registry_decorators_are_not_flagged_while_unused_function_is(
    tmp_path: Path, decorator: str
) -> None:
    """Functions registered via decorator callbacks must not count as dead code."""
    src = (
        f"{decorator}\n"
        "def _registered_callback() -> int:\n"
        "    return 1\n\n"
        "def _genuinely_unused_function() -> int:\n"
        "    return 2\n"
    )
    counts = _collect_single_file(tmp_path, src)
    assert counts == {("vulture", "unused-function", "lucid/sample.py"): 1}


@pytest.mark.parametrize(
    ("kind", "snippet", "dead_snippet", "expected_rule"),
    [
        (
            "tensor_public_method",
            "class Tensor:\n"
            "    def clamp_min_(self) -> int:\n"
            "        return 0\n"
            "    def share_memory_(self) -> int:\n"
            "        return 0\n"
            "Tensor()\n",
            "class _Holder:\n"
            "    def _truly_dead_method(self) -> int:\n"
            "        return 0\n"
            "_Holder()\n",
            "unused-method",
        ),
        (
            "weights_enum_members",
            "class Weights:\n"
            "    ALIEN = 1\n"
            "    UPNDOWN = 2\n"
            "    IMAGENET1K_512 = 3\n"
            "Weights()\n",
            "class _DeadWeights:\n" "    TRULY_DEAD_MEMBER = 99\n" "_DeadWeights()\n",
            "unused-variable",
        ),
        (
            "cpp_and_stdlib_hooks",
            "class Hooks:\n"
            "    def __init__(self) -> None:\n"
            "        self.once_differentiable = True\n"
            "        self.redraw_rng = False\n"
            "    def persistent_id(self) -> None:\n"
            "        return None\n"
            "    def _on_queue_feeder_error(self) -> None:\n"
            "        return None\n"
            "Hooks()\n",
            "class _AttrHolder:\n"
            "    def __init__(self) -> None:\n"
            "        self._truly_dead_attr = 1\n"
            "_AttrHolder()\n",
            "unused-attribute",
        ),
    ],
)
def test_whitelisted_indirect_symbols_are_not_flagged(
    tmp_path: Path, kind: str, snippet: str, dead_snippet: str, expected_rule: str
) -> None:
    """Whitelisted public API, WeightsEnum members, and runtime hooks are ignored."""
    del kind
    assert config.VULTURE_WHITELIST.is_file()
    assert "@_emitter" in config.VULTURE_IGNORE_DECORATORS
    assert "@register_kl" in config.VULTURE_IGNORE_DECORATORS

    clean_counts = _collect_single_file(tmp_path, snippet)
    assert clean_counts == {}

    seeded_counts = _collect_single_file(tmp_path, snippet + "\n" + dead_snippet)
    assert seeded_counts == {("vulture", expected_rule, "lucid/sample.py"): 1}


# ── DS-18: the whitelist and the allowlists are contract ────────────────────


def test_whitelist_digest_follows_its_names_not_its_layout(tmp_path: Path) -> None:
    head = "from unittest.mock import Mock\n\n_ = Mock()\n"
    plain = tmp_path / "plain.py"
    plain.write_text(head + "_.alpha  # read by C++\n")
    restyled = tmp_path / "restyled.py"
    restyled.write_text(head + "\n# ── a section ──\n_.alpha  # reworded reason\n\n")
    grown = tmp_path / "grown.py"
    grown.write_text(head + "_.alpha  # read by C++\n_.beta  # a new entry\n")
    assert config.ast_digest(plain) == config.ast_digest(restyled)
    assert config.ast_digest(plain) != config.ast_digest(grown)


def test_allowlist_edits_change_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    before = config.contract()
    monkeypatch.setattr(
        config, "VULTURE_IGNORE_DECORATORS", (*config.VULTURE_IGNORE_DECORATORS, "@_hides_more")
    )
    monkeypatch.setattr(
        config,
        "ENGINE_TEST_ONLY_BINDINGS",
        {**config.ENGINE_TEST_ONLY_BINDINGS, "cube": "a test calls it"},
    )
    after = config.contract()
    assert after["vulture"] != before["vulture"]
    assert after["engine"] != before["engine"]
    assert after["lizard"] == before["lizard"]


def test_every_contract_belongs_to_a_collector() -> None:
    """A contract filed under a name no collector has would never be checked;
    the baseline records the entries flat, so their keys must be unique."""
    names = {c.name for c in all_collectors()}
    assert set(config.contract()) <= names
    flat = [key for entries in config.contract().values() for key in entries]
    assert len(flat) == len(set(flat))


# ── DS-24: engine bindings no production module reaches ─────────────────────

STUB = """\
class TensorImpl:
    def item(self) -> float: ...

def cube(x: TensorImpl) -> TensorImpl: ...
def cube_root(x: TensorImpl) -> TensorImpl: ...
def diag(x: TensorImpl) -> TensorImpl: ...
def host_sync_count() -> int: ...

class _nn_namespace:
    @staticmethod
    def bce_loss(x: TensorImpl) -> TensorImpl: ...
    @staticmethod
    def rotate(x: TensorImpl) -> TensorImpl: ...
nn: _nn_namespace

class _linalg_namespace:
    @staticmethod
    def solve(a: TensorImpl) -> TensorImpl: ...
    @staticmethod
    def lstsq(a: TensorImpl) -> TensorImpl: ...
linalg: _linalg_namespace

class _utils_tokenizer_namespace:
    @staticmethod
    def encode(s: str) -> list[int]: ...

class _utils_namespace:
    tokenizer: _utils_tokenizer_namespace
utils: _utils_namespace
"""

# Each production file reaches the engine one way.  The look-alike only names
# the same attributes, and the test is not production.
TREE = {
    "lucid/ops.py": (
        "from lucid._C import engine as _C_engine\n\n"
        "_R = _C_engine\n\n\n"
        "def f(x):\n"
        "    return _R.cube(_C_engine.nn.bce_loss(x))\n"
    ),
    "lucid/linalg/__init__.py": (
        "from lucid._C import engine as _C_engine\n\n_la = _C_engine.linalg\n"
    ),
    "lucid/linalg/_ex.py": (
        "from lucid.linalg import _la\n\n\ndef g(a):\n    return _la.solve(a)\n"
    ),
    "lucid/linalg/_rel.py": "from . import _la\n\n\ndef r(a):\n    return _la.lstsq(a)\n",
    "lucid/tok.py": (
        "from lucid._C.engine import utils as _C_utils\n\n\n"
        "def h(s):\n"
        "    return _C_utils.tokenizer.encode(s)\n"
    ),
    "lucid/dyn.py": (
        "from lucid._C import engine as _C_engine\n\n\n"
        "def k(x):\n"
        '    return getattr(_C_engine, "diag")(x)\n'
    ),
    "lucid/look_alike.py": (
        "class T:\n"
        "    def __init__(self, F):\n"
        "        self.rotate = F.rotate\n"
        "        self.cube_root = F.cube_root\n"
    ),
    "lucid/test/test_engine.py": (
        "from lucid._C import engine as _C_engine\n\n"
        "_C_engine.cube_root(1)\n"
        "_C_engine.nn.rotate(1)\n"
    ),
}


def _engine_tree(root: Path) -> Scope:
    (root / config.ENGINE_STUB).parent.mkdir(parents=True)
    (root / config.ENGINE_STUB).write_text(STUB)
    for rel, text in TREE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return scope_of(root, [config.ENGINE_STUB, *TREE], root)


def test_stub_functions_cover_submodules_but_not_methods() -> None:
    assert stub_functions(STUB) == {
        "cube",
        "cube_root",
        "diag",
        "host_sync_count",
        "nn.bce_loss",
        "nn.rotate",
        "linalg.solve",
        "linalg.lstsq",
        "utils.tokenizer.encode",
    }


def test_unbound_bindings_follow_engine_paths_only(tmp_path: Path) -> None:
    """Aliases, cross-module and relative imports and ``getattr`` reach a
    binding; a same-named attribute elsewhere and a test do not, and the
    test-only allowlist (``host_sync_count``) is not counted."""
    scope = _engine_tree(tmp_path)
    assert "host_sync_count" in config.ENGINE_TEST_ONLY_BINDINGS
    assert "lucid/test/test_engine.py" not in scope.py
    assert unbound_bindings(tmp_path, scope.py) == ["cube_root", "nn.rotate"]
    counts = EngineBindingsCollector().collect(scope)
    assert counts == {("engine", "unbound", config.ENGINE_STUB): 2}


def test_a_tree_without_the_engine_stub_measures_nothing(tmp_path: Path) -> None:
    (tmp_path / "lucid").mkdir()
    (tmp_path / "lucid" / "a.py").write_text("x = 1\n")
    scope = scope_of(tmp_path, ["lucid/a.py"], tmp_path)
    assert EngineBindingsCollector().collect(scope) == {}
