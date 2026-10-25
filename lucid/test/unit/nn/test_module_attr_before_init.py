"""A plain attribute set before ``super().__init__()`` is allowed.

``Module.__setattr__`` touched ``self._parameters`` / ``self._buffers`` on
every assignment, before checking what kind of value was being set.  None
of the tracked registries exist until ``Module.__init__`` has run, so even
``self.x = 3`` before ``super().__init__()`` fell into ``__getattr__``,
which raised about the *registry's* own missing name (``'_parameters'``)
rather than the one actually being looked up.  The reference tolerates a
plain value there and refuses only a ``Parameter`` / ``Module`` / buffer,
which needs a registry that is not there yet (LCD-556).
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn


class _PlainBeforeInit(nn.Module):
    def __init__(self):
        self.x = 3
        super().__init__()


class _ParameterBeforeInit(nn.Module):
    def __init__(self):
        self.w = nn.Parameter(lucid.zeros(2))
        super().__init__()


class _ModuleBeforeInit(nn.Module):
    def __init__(self):
        self.lin = nn.Linear(2, 2)
        super().__init__()


def test_a_plain_attribute_before_init_is_kept():
    m = _PlainBeforeInit()
    assert m.x == 3


def test_a_parameter_before_init_is_refused():
    with pytest.raises(AttributeError, match="before Module.__init__"):
        _ParameterBeforeInit()


def test_a_submodule_before_init_is_refused():
    with pytest.raises(AttributeError, match="before Module.__init__"):
        _ModuleBeforeInit()


def test_register_parameter_before_init_is_refused():
    class M(nn.Module):
        def __init__(self):
            self.register_parameter("p", nn.Parameter(lucid.zeros(2)))
            super().__init__()

    with pytest.raises(AttributeError, match="before Module.__init__"):
        M()


def test_register_buffer_before_init_is_refused():
    class M(nn.Module):
        def __init__(self):
            self.register_buffer("b", lucid.zeros(2))
            super().__init__()

    with pytest.raises(AttributeError, match="before Module.__init__"):
        M()


def test_a_missing_attribute_before_init_names_itself_not_the_registry():
    class M(nn.Module):
        def __init__(self):
            # A read, not a write: ``__getattr__`` must not crash about
            # ``_parameters`` while probing for an unrelated name.
            _ = getattr(self, "missing", "default")
            super().__init__()

    M()  # must not raise


@pytest.mark.parity
def test_plain_attribute_before_init_matches_the_reference(ref: ModuleType):
    class Lucid(nn.Module):
        def __init__(self):
            self.x = 3
            super().__init__()

    class Ref(ref.nn.Module):
        def __init__(self):
            self.x = 3
            super().__init__()

    assert Lucid().x == Ref().x == 3
