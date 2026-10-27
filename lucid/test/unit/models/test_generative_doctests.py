"""Every generative family's ``Examples`` block has to actually run.

Doctests are in no other gate — not in ``pyproject.toml``'s ``addopts``,
not in any workflow — so an example that raises, or that states a value
the code does not produce, ships silently.  That is a real defect rather
than a cosmetic one: these blocks are what the documentation site renders
and what a reader copies out.

Found the hard way.  ``check_docstring_coverage`` was green throughout
while ``NeuralODEConfig``'s example raised ``ValueError`` on the first
line and claimed a ``data_dim`` the code never returns — that checker
validates presence and shape, not whether an example is true.

Scoped to ``lucid.models.generative`` because that is where this file
lives, not because the rest is unclean — ``lucid/linalg``,
``lucid/autograd`` and ``lucid/optim`` were swept and now run too.  A
package-wide gate is the obvious next step; it needs the model zoo's
slower examples costed first.
"""

import contextlib
import doctest
import importlib
import io
import pkgutil

import pytest

import lucid.models.generative as _generative


def _generative_modules() -> list[str]:
    """Every module under the generative package, including sub-packages."""
    found = [_generative.__name__]
    for info in pkgutil.walk_packages(_generative.__path__, _generative.__name__ + "."):
        found.append(info.name)
    return sorted(found)


@pytest.mark.slow
@pytest.mark.parametrize("module_name", _generative_modules())
def test_examples_run(module_name: str) -> None:
    module = importlib.import_module(module_name)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        # ELLIPSIS matches what ``pytest --doctest-modules`` applies by
        # default, so running this file and running that flag agree.
        result = doctest.testmod(module, optionflags=doctest.ELLIPSIS, report=False, verbose=False)
    assert result.failed == 0, f"{module_name}:\n{captured.getvalue()}"
