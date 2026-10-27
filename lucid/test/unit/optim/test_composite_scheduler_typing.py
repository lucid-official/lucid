"""A list of schedulers held in a variable type-checks as a composite's children.

``SequentialLR`` and ``ChainedScheduler`` took ``list[A | B]``.  ``list`` is
invariant, so a user's ``kids = [ExponentialLR(...), StepLR(...)]`` — a
``list[_LRScheduler]`` — no longer matched once the children's type grew to
a union, and ``mypy --strict`` failed on code that had passed.  They take a
``Sequence``, as the reference does.  Only a checker can see that, so this
runs mypy on a small client program.
"""

import sys
import textwrap

import pytest

mypy_api = pytest.importorskip("mypy.api")

_CLIENT = textwrap.dedent("""
    import lucid.nn as nn
    import lucid.optim as optim
    from lucid.optim import lr_scheduler as S

    opt = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    kids = [S.ExponentialLR(opt, gamma=0.9), S.StepLR(opt, step_size=2)]
    chain = S.ChainedScheduler(kids)
    S.SequentialLR(opt, kids, milestones=[2])
    S.SequentialLR(opt, [S.LinearLR(opt), chain], milestones=[3])
    S.ChainedScheduler([chain, S.SequentialLR(opt, kids, milestones=[2])])
    S.ChainedScheduler((S.ExponentialLR(opt, gamma=0.9),))

    S.ChainedScheduler([S.ReduceLROnPlateau(opt)])  # the only error expected
    """)

_BAD_LINE = (
    _CLIENT.splitlines().index(
        "S.ChainedScheduler([S.ReduceLROnPlateau(opt)])  # the only error expected"
    )
    + 1
)


@pytest.fixture(scope="module")
def mypy_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, int]:
    tmp = tmp_path_factory.mktemp("composite_scheduler_typing")
    client = tmp / "client.py"
    client.write_text(_CLIENT)
    # An empty config, so the result does not depend on which mypy.ini
    # the working directory happens to hold.
    config = tmp / "mypy.ini"
    config.write_text("[mypy]\n")
    stdout, stderr, status = mypy_api.run(
        [
            str(client),
            "--strict",
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


def test_a_list_of_schedulers_in_a_variable_is_accepted(mypy_report: tuple[str, int]) -> None:
    stdout, _ = mypy_report
    unexpected = [
        e for e in stdout.splitlines() if e.strip() and f"client.py:{_BAD_LINE}:" not in e
    ]
    assert unexpected == [], "\n".join(unexpected)


def test_a_child_that_steps_on_a_metric_is_still_a_type_error(mypy_report: tuple[str, int]) -> None:
    stdout, status = mypy_report
    assert status == 1
    assert f"client.py:{_BAD_LINE}:" in stdout, stdout
