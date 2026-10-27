"""``module(...)`` is typed from that module's own ``forward``.

``Module.__call__`` binds ``self`` to a protocol over ``forward``, so a type
checker reads the parameters and the return type of each call off the
concrete subclass instead of the base class's catch-all signature.  Only a
checker can see that, so these tests run mypy on a small client program.
"""

import sys
import textwrap

import pytest

mypy_api = pytest.importorskip("mypy.api")

_CLIENT = textwrap.dedent("""
    from typing import assert_type, override

    import lucid
    import lucid.nn as nn
    from lucid import Tensor

    x = lucid.randn(2, 3)
    lin = nn.Linear(3, 4)
    assert_type(lin(x), Tensor)
    assert_type(nn.Sequential(nn.Linear(3, 4), nn.ReLU())(x), Tensor)

    q = lucid.randn(5, 2, 4)
    attn = nn.MultiheadAttention(4, 2)
    assert_type(attn(q, q, q), tuple[Tensor, Tensor | None])

    compiled = lucid.compile(nn.Linear(3, 4))
    assert_type(compiled(x), Tensor)
    reveal_type(compiled)  # the compiled module keeps forward's signature

    class Twice(nn.Module):
        @override
        def forward(self, x: Tensor, scale: float = 2.0) -> tuple[Tensor, Tensor]:
            return x, x * scale

    assert_type(Twice()(x), tuple[Tensor, Tensor])
    assert_type(Twice()(x, scale=3.0), tuple[Tensor, Tensor])

    layers: nn.ModuleList[nn.Linear] = nn.ModuleList([nn.Linear(3, 4)])
    assert_type(layers[0](x), Tensor)
    assert_type(layers[:1], nn.ModuleList[nn.Linear])

    named: nn.ModuleDict[nn.Linear] = nn.ModuleDict({"head": nn.Linear(3, 4)})
    assert_type(named["head"](x), Tensor)

    seq = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    assert_type(seq[:1](x), Tensor)

    img = lucid.randn(1, 3, 8, 8)
    assert_type(nn.MaxPool2d(2)(img), Tensor)
    assert_type(nn.AdaptiveMaxPool2d((4, 4))(img), Tensor)

    lin("bad")  # the only error expected
    """)

_LINES = _CLIENT.splitlines()
_BAD_LINE = _LINES.index('lin("bad")  # the only error expected') + 1
_REVEAL_LINE = (
    _LINES.index("reveal_type(compiled)  # the compiled module keeps forward's signature") + 1
)


@pytest.fixture(scope="module")
def mypy_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, int]:
    tmp = tmp_path_factory.mktemp("module_call_typing")
    client = tmp / "client.py"
    client.write_text(_CLIENT)
    # An empty config, so the result does not depend on which mypy.ini
    # the working directory happens to hold.
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


def _at(stdout: str, line: int) -> list[str]:
    return [entry for entry in stdout.splitlines() if f"client.py:{line}:" in entry]


def test_calls_take_the_subclass_forward_types(mypy_report: tuple[str, int]) -> None:
    stdout, _ = mypy_report
    expected = set(_at(stdout, _BAD_LINE)) | set(_at(stdout, _REVEAL_LINE))
    unexpected = [e for e in stdout.splitlines() if e.strip() and e not in expected]
    assert unexpected == [], "\n".join(unexpected)


def test_compile_keeps_the_forward_signature(mypy_report: tuple[str, int]) -> None:
    stdout, _ = mypy_report
    revealed = _at(stdout, _REVEAL_LINE)
    assert len(revealed) == 1, stdout
    tensor = "lucid._tensor.tensor.Tensor"
    assert f"CompiledModule[[input: {tensor}], {tensor}]" in revealed[0], revealed[0]


def test_a_wrong_argument_is_caught(mypy_report: tuple[str, int]) -> None:
    stdout, status = mypy_report
    assert status == 1
    bad = _at(stdout, _BAD_LINE)
    assert len(bad) == 1, stdout
    assert "[arg-type]" in bad[0], bad[0]
