"""``load_state_dict`` is typed as the named tuple it returns (LCD-706).

It was annotated ``-> object``, so ``result.missing_keys`` and
``missing, unexpected = ...`` were errors for a caller under a type checker
even though they are the documented use.  Only a checker can see that, so these
tests run mypy on a small client program, as ``test_module_call_typing`` does.
"""

import sys
import textwrap

import pytest

mypy_api = pytest.importorskip("mypy.api")

_BAD = "result.missing_keyz  # the only error expected"

_CLIENT = textwrap.dedent(f"""
    from typing import assert_type

    import lucid
    import lucid.nn as nn
    from lucid.nn._incompatible_keys import _IncompatibleKeys
    from lucid.peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    lin = nn.Linear(3, 4)
    result = lin.load_state_dict(lin.state_dict(), strict=False)
    assert_type(result, _IncompatibleKeys)
    assert_type(result.missing_keys, list[str])
    assert_type(result.unexpected_keys, list[str])
    missing, unexpected = result
    assert_type(missing, list[str])
    assert_type(unexpected, list[str])

    peft = get_peft_model(nn.Sequential(nn.Linear(4, 2)), LoraConfig(r=2, target_modules=["0"]))
    assert_type(set_peft_model_state_dict(peft, {{}}), _IncompatibleKeys)
    assert_type(peft.load_adapter("dir", "other"), _IncompatibleKeys)

    {_BAD}
    """)

_BAD_LINE = _CLIENT.splitlines().index(_BAD) + 1


@pytest.fixture(scope="module")
def mypy_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, int]:
    tmp = tmp_path_factory.mktemp("load_state_dict_typing")
    client = tmp / "client.py"
    client.write_text(_CLIENT)
    config = tmp / "mypy.ini"
    config.write_text("[mypy]\n")
    stdout, stderr, status = mypy_api.run(
        [
            str(client),
            "--config-file",
            str(config),
            "--python-executable",
            sys.executable,
            "--cache-dir",
            str(tmp / "cache"),
            "--follow-imports=silent",
            "--no-error-summary",
            "--show-error-codes",
            "--hide-error-context",
            "--no-color-output",
        ]
    )
    assert not stderr, stderr
    return stdout, status


def test_the_result_is_the_named_tuple(mypy_report: tuple[str, int]) -> None:
    stdout, _ = mypy_report
    wanted = f"client.py:{_BAD_LINE}:"
    unexpected = [e for e in stdout.splitlines() if e.strip() and wanted not in e]
    assert unexpected == [], "\n".join(unexpected)


def test_a_wrong_field_is_caught(mypy_report: tuple[str, int]) -> None:
    stdout, status = mypy_report
    assert status == 1
    bad = [e for e in stdout.splitlines() if f"client.py:{_BAD_LINE}:" in e]
    assert len(bad) == 1, stdout
    assert "[attr-defined]" in bad[0], bad[0]
