"""Unit tests for the export rule of ``tools/validate_model_zoo.py`` (MZ080 / MZ081).

A public name in a family module is either exported from the family package
or it is not public.  The rule is exercised on seeded families — a clean one,
and one per way of breaking it — and then on the real tree, so a family that
adds an unexported helper fails here, in the default suite, not only when
someone runs the validator by hand.
"""

import textwrap
from pathlib import Path

import pytest

from tools import validate_model_zoo as zoo

_INIT = '__all__ = ["Widget", "make_widget"]\n'


def _family(root: Path, files: dict[str, str], init: str = _INIT) -> Path:
    family = root / "vision" / "widget"
    family.mkdir(parents=True)
    (family / "__init__.py").write_text(init)
    for name, body in files.items():
        path = family / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return family


def _issues(family: Path) -> list[zoo.Issue]:
    found: list[zoo.Issue] = []
    zoo._check_exports(family, found)
    return found


def test_a_family_that_exports_its_public_names_passes(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {
            "_model.py": """
                class Widget: ...
                class _Block: ...
                def _helper() -> None: ...
            """,
            "_pretrained.py": "def make_widget() -> None: ...\n",
        },
    )
    assert _issues(family) == []


def test_an_exported_component_in_its_own_module_passes(tmp_path: Path) -> None:
    """The family-component tier (a tokenizer, a scheduler) is public by export."""
    family = _family(
        tmp_path,
        {
            "_model.py": "class Widget: ...\n",
            "_pretrained.py": "def make_widget() -> None: ...\n",
            "_scheduler.py": "class WidgetScheduler: ...\n",
        },
        init='__all__ = ["Widget", "make_widget", "WidgetScheduler"]\n',
    )
    assert _issues(family) == []


def test_an_unexported_class_is_an_error(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {
            "_model.py": "class Widget: ...\n",
            "_pretrained.py": "def make_widget() -> None: ...\n",
            "_heads.py": "class TwoHotHead: ...\n",
        },
    )
    (issue,) = _issues(family)
    assert (issue.code, issue.severity) == ("MZ080", "error")
    assert issue.path.name == "_heads.py" and "TwoHotHead" in issue.message
    assert "class" in issue.message


def test_an_unexported_function_is_an_error(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {
            "_model.py": "class Widget: ...\n",
            "_pretrained.py": "def make_widget() -> None: ...\n\ndef balanced_kl() -> None: ...\n",
        },
    )
    (issue,) = _issues(family)
    assert (issue.code, issue.severity) == ("MZ080", "error")
    assert issue.path.name == "_pretrained.py" and "balanced_kl" in issue.message
    assert "function" in issue.message


def test_an_unexported_output_is_an_error(tmp_path: Path) -> None:
    """The case that made FCN and EfficientFormer export their ``*Output``."""
    family = _family(
        tmp_path,
        {
            "_model.py": "class Widget: ...\nclass WidgetOutput: ...\n",
            "_pretrained.py": "def make_widget() -> None: ...\n",
        },
    )
    (issue,) = _issues(family)
    assert issue.code == "MZ080" and "WidgetOutput" in issue.message


def test_modules_below_the_family_directory_are_checked(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {
            "_model.py": "class Widget: ...\n",
            "_pretrained.py": "def make_widget() -> None: ...\n",
            "_parts/_attention.py": "class Attention: ...\n",
        },
    )
    (issue,) = _issues(family)
    assert issue.code == "MZ080" and issue.path.name == "_attention.py"


def test_names_other_than_top_level_definitions_are_not_judged(tmp_path: Path) -> None:
    """A nested class, a constant and an import are not definitions the rule polices."""
    family = _family(
        tmp_path,
        {
            "_model.py": """
                from collections import OrderedDict

                LIMIT = 3

                class Widget:
                    class Inner: ...
                    def method(self) -> None: ...
            """,
            "_pretrained.py": "def make_widget() -> None: ...\n",
        },
    )
    assert _issues(family) == []


def test_the_package_init_itself_is_not_scanned(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {"_model.py": "class Widget: ...\n", "_pretrained.py": "def make_widget() -> None: ...\n"},
        init='__all__ = ["Widget", "make_widget"]\n\ndef lazy() -> None: ...\n',
    )
    assert _issues(family) == []


@pytest.mark.parametrize(
    "init",
    ["", "from x import y\n", '__all__ = ["a"] + other\n', '__all__ = ["a", name]\n'],
    ids=["empty", "no-all", "computed-all", "non-string-entry"],
)
def test_an_all_the_rule_cannot_read_is_an_error(tmp_path: Path, init: str) -> None:
    """Failing open would let the rule pass by not being able to look."""
    family = _family(tmp_path, {"_model.py": "class Widget: ...\n"}, init=init)
    (issue,) = _issues(family)
    assert (issue.code, issue.severity) == ("MZ081", "error")
    assert issue.path.name == "__init__.py"


def test_a_tuple_all_and_an_annotated_all_are_read(tmp_path: Path) -> None:
    family = _family(
        tmp_path,
        {"_model.py": "class Widget: ...\n"},
        init='__all__: tuple[str, ...] = ("Widget",)\n',
    )
    assert _issues(family) == []


def test_a_missing_init_is_left_to_the_layout_check(tmp_path: Path) -> None:
    family = tmp_path / "vision" / "widget"
    family.mkdir(parents=True)
    (family / "_model.py").write_text("class Widget: ...\n")
    assert _issues(family) == []


def test_shared_substrate_is_not_a_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``<domain>/_common`` holds what several families share; the rule is not for it."""
    _family(tmp_path, {"_model.py": "class Widget: ...\n"})
    common = tmp_path / "vision" / "_common"
    common.mkdir()
    (common / "__init__.py").write_text("")
    (common / "_blocks.py").write_text("class ConvBlock: ...\n\ndef conv_bn() -> None: ...\n")
    monkeypatch.setattr(zoo, "MODELS_DIR", tmp_path)

    assert [f.name for f in zoo.iter_families(None, None)] == ["widget"]

    issues: list[zoo.Issue] = []
    for fam in zoo.iter_families(None, None):
        zoo._check_exports(fam, issues)
    assert issues == []


def test_the_rule_runs_in_a_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _family(
        tmp_path, {"_model.py": "class Widget: ...\n", "_helpers.py": "def stray() -> None: ...\n"}
    )
    monkeypatch.setattr(zoo, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(zoo, "REPO_ROOT", tmp_path)

    zoo.validate("vision", None, strict=False)
    out = capsys.readouterr().out
    assert "error MZ080 stray: public function in _helpers.py" in out


def test_every_real_family_exports_every_public_name() -> None:
    """The zoo as shipped: nothing public-named is left out of a family ``__all__``."""
    families = list(zoo.iter_families(None, None))
    assert len(families) > 50, "the zoo scan found almost no families — the path is wrong"
    issues: list[zoo.Issue] = []
    for fam in families:
        zoo._check_exports(fam, issues)
    assert [i.format() for i in issues] == []
