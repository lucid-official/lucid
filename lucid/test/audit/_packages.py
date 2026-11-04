"""The packages of Lucid, and what each one means for each stage of the audit.

A run used to be scoped three different ways: a ``--subsystem`` string for the
sweep, stage flags for the rest, and a separate script for the model zoo.  Two
of those could not say "just the linear algebra, docs included", and the third
was not in the audit at all — so "the audit is clean" said nothing about the
zoo, which is the largest thing the framework ships.

This module owns the question.  An :class:`AuditPackage` states, for itself:

==========  =================================================================
sweep       the ``SUBSYSTEMS`` keys whose symbols the probe sweep walks
suite       the test subtrees pytest runs, and the source files the
            line-coverage floor is read from
doctests    the module prefix its docstring examples live under
zoo         the model zoo is the one package a family-by-family stage covers
==========  =================================================================

The command line is *generated* from the registry — ``--<key>`` selects a
package, ``--no-<key>`` removes it — so a new ``SUBSYSTEMS`` entry has its
flags the moment it is declared, and nothing else needs editing.  A parent
takes its children with it (``--nn`` is ``nn.functional``, ``nn.init``, …),
because a ``.`` in a key is a containment claim, not a naming accident.

Selection is deliberately small:

* no package flag selects every package;
* positive flags select exactly those (and their children);
* ``--no-*`` flags subtract;
* asking for a package and excluding it — or something that contains it — is
  a contradiction, :class:`PackageSelectionError`, not a silent winner.
"""

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from lucid.test.audit import _suite, _surface

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

#: The key of the one package the family-by-family zoo stage verifies.
MODELS = "models"

#: What ``--suite-path`` defaults to, and so what "the whole tree" means.
DEFAULT_SUITE_PATH = "lucid/test"


class PackageSelectionError(ValueError):
    """The package flags contradict each other, or select nothing."""


@dataclass(frozen=True)
class AuditPackage(ABC):
    """One selectable part of Lucid and what taking part in each stage means for it."""

    key: str

    @property
    def flag(self) -> str:
        return f"--{self.key}"

    @property
    def negated_flag(self) -> str:
        return f"--no-{self.key}"

    @property
    def dest(self) -> str:
        return "pkg_" + self.key.replace(".", "_")

    @property
    def negated_dest(self) -> str:
        return "no_" + self.dest

    @property
    @abstractmethod
    def stages(self) -> tuple[str, ...]:
        """The stages this package takes part in, by name."""

    @abstractmethod
    def summary(self) -> str:
        """One line for ``--help`` and ``--list-packages``."""

    @abstractmethod
    def symbol_keys(self) -> tuple[str, ...]:
        """The ``SUBSYSTEMS`` keys whose symbols the sweep walks for this package."""

    @abstractmethod
    def test_trees(self, root: Path) -> tuple[str, ...]:
        """The test subtrees that exist for this package, relative to ``root``."""

    @abstractmethod
    def file_prefixes(self) -> tuple[str, ...]:
        """The source paths line coverage is attributed to this package by."""

    @abstractmethod
    def module_prefix(self) -> str:
        """The dotted module path its docstring examples live under."""


@dataclass(frozen=True)
class SubsystemPackage(AuditPackage):
    """A ``SUBSYSTEMS`` entry: swept, tested by subtree, documented by module."""

    @property
    @override
    def stages(self) -> tuple[str, ...]:
        return ("sweep", "suite", "doctests")

    @override
    def summary(self) -> str:
        module, kind = _surface.SUBSYSTEMS[self.key]
        return f"{module} ({kind})"

    @override
    def symbol_keys(self) -> tuple[str, ...]:
        return (self.key,)

    @override
    def test_trees(self, root: Path) -> tuple[str, ...]:
        return tuple(_suite.subsystem_test_dirs(root, self.key))

    @override
    def file_prefixes(self) -> tuple[str, ...]:
        return _suite.file_prefixes_for_subsystems(self.key) or ()

    @override
    def module_prefix(self) -> str:
        return _surface.SUBSYSTEMS[self.key][0]


@dataclass(frozen=True)
class ModelZooPackage(AuditPackage):
    """``lucid.models``: no symbol sweep and no doctests, a stage of its own.

    The zoo is composition over the primitives the sweep audits, with its own
    contract validator, device sweep and real-size builds, so the sweep leaves
    it out (:data:`~lucid.test.audit._surface.EXCLUDED`).  What it does not
    leave out is *responsibility*: this package is how a run says "and the
    zoo", and its stage is :mod:`~lucid.test.audit._zoo`.
    """

    key: str = MODELS

    #: Where the zoo's own tests live.  Files elsewhere that merely build a
    #: model (peft, coreml, compile) belong to those packages' trees; the zoo
    #: stage still runs them once, with the markers cleared.
    _TREES = ("lucid/test/unit/models", "lucid/test/parity/models")

    @property
    @override
    def stages(self) -> tuple[str, ...]:
        return ("zoo",)

    @override
    def summary(self) -> str:
        return "lucid.models (the model zoo, verified family by family)"

    @override
    def symbol_keys(self) -> tuple[str, ...]:
        return ()

    @override
    def test_trees(self, root: Path) -> tuple[str, ...]:
        return tuple(rel for rel in self._TREES if (root / rel).exists())

    @override
    def file_prefixes(self) -> tuple[str, ...]:
        return ("lucid/models/",)

    @override
    def module_prefix(self) -> str:
        return "lucid.models"


@dataclass(frozen=True)
class SuitePlan:
    """What the suite stage hands pytest, leaves out, and is held to."""

    paths: tuple[str, ...]
    ignore: tuple[str, ...]
    #: The source files the line floor may be compared on; ``None`` is all.
    in_scope: Callable[[str], bool] | None


class PackageRegistry:
    """Every package, in a stable order, and the flags that select them."""

    def __init__(self, packages: Iterable[AuditPackage]) -> None:
        self._packages: dict[str, AuditPackage] = {}
        for package in packages:
            if package.key in self._packages:
                raise ValueError(f"package {package.key!r} is declared twice")
            self._packages[package.key] = package
        self._check_dests()

    @classmethod
    def default(cls) -> PackageRegistry:
        """One :class:`SubsystemPackage` per ``SUBSYSTEMS`` key, then the zoo."""
        return cls([*(SubsystemPackage(key) for key in _surface.SUBSYSTEMS), ModelZooPackage()])

    def _check_dests(self) -> None:
        dests = [d for p in self for d in (p.dest, p.negated_dest)]
        if len(set(dests)) != len(dests):
            raise ValueError("two packages would store their flag in the same attribute")

    def __iter__(self) -> Iterator[AuditPackage]:
        return iter(self._packages.values())

    def __len__(self) -> int:
        return len(self._packages)

    def __contains__(self, key: object) -> bool:
        return key in self._packages

    def __getitem__(self, key: str) -> AuditPackage:
        return self._packages[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._packages)

    def flags(self) -> frozenset[str]:
        """Every option string this registry generates."""
        return frozenset(f for p in self for f in (p.flag, p.negated_flag))

    def subtree(self, key: str) -> frozenset[str]:
        """``key`` and every package it contains."""
        return frozenset(k for k in self._packages if k == key or k.startswith(key + "."))

    def code_keys(self) -> frozenset[str]:
        """The packages the sweep, suite and doctests carry: everything but the zoo."""
        return frozenset(k for k in self._packages if k != MODELS)

    # ── the command line ─────────────────────────────────────────────────────

    def add_flags(self, parser: argparse.ArgumentParser) -> None:
        """Generate ``--<key>`` and ``--no-<key>`` for every package.

        A flag that is already declared is argparse's conflict to raise: that
        is the failure this must not hide, since a silently shadowed flag
        changes what an existing command line means.
        """
        group = parser.add_argument_group(
            "packages",
            "--<package> runs only those packages, --no-<package> leaves one out; "
            "with neither, every package runs, the model zoo included.  "
            "A package takes its children (--nn is nn.functional, nn.init, ...).  "
            "--list-packages names them all.",
        )
        for package in self:
            group.add_argument(
                package.flag, dest=package.dest, action="store_true", help=package.summary()
            )
            group.add_argument(
                package.negated_flag,
                dest=package.negated_dest,
                action="store_true",
                help=argparse.SUPPRESS,
            )

    def select(self, args: argparse.Namespace) -> Selection:
        """The packages ``args`` ask for, or :class:`PackageSelectionError`."""
        wanted = [p.key for p in self if getattr(args, p.dest, False)]
        unwanted = [p.key for p in self if getattr(args, p.negated_dest, False)]
        for ask in wanted:
            for drop in unwanted:
                if ask == drop or ask.startswith(drop + "."):
                    raise PackageSelectionError(
                        f"{self[ask].flag} asks for a package that {self[drop].negated_flag} "
                        "leaves out; pass one or the other"
                    )
        chosen = {k for key in wanted for k in self.subtree(key)} if wanted else set(self._packages)
        chosen -= {k for key in unwanted for k in self.subtree(key)}
        if not chosen:
            raise PackageSelectionError("the package flags leave nothing to run")
        return Selection(self, frozenset(chosen), explicit=bool(wanted))

    # ── ownership ────────────────────────────────────────────────────────────

    def file_owners(self, name: str) -> frozenset[str]:
        """The packages whose most specific source prefix covers ``name``."""
        best = 0
        owners: set[str] = set()
        for package in self:
            for prefix in package.file_prefixes():
                if name.startswith(prefix) and len(prefix) >= best:
                    if len(prefix) > best:
                        owners.clear()
                    best = len(prefix)
                    owners.add(package.key)
        return frozenset(owners)

    def module_owners(self, name: str) -> frozenset[str]:
        """The packages whose most specific module prefix covers ``name``."""
        best = 0
        owners: set[str] = set()
        for package in self:
            prefix = package.module_prefix()
            if (name == prefix or name.startswith(prefix + ".")) and len(prefix) >= best:
                if len(prefix) > best:
                    owners.clear()
                best = len(prefix)
                owners.add(package.key)
        return frozenset(owners)


@dataclass(frozen=True)
class Selection:
    """Which packages a run covers, and what that means for each stage."""

    registry: PackageRegistry
    keys: frozenset[str]
    #: A positive flag was given, so the run is *only* those packages.
    explicit: bool

    def includes(self, key: str) -> bool:
        return key in self.keys

    @property
    def runs_zoo(self) -> bool:
        return MODELS in self.keys

    @property
    def code_keys(self) -> frozenset[str]:
        """The selected packages the sweep, suite and doctests carry."""
        return self.keys & self.registry.code_keys()

    @property
    def has_code(self) -> bool:
        return bool(self.code_keys)

    @property
    def complete(self) -> bool:
        """Whether every package those three stages carry is selected."""
        return self.code_keys == self.registry.code_keys()

    def describe(self) -> str:
        """A phrase for the banner and the transcript."""
        if len(self.keys) == len(self.registry):
            return "every package"
        if not self.explicit:
            left_out = [k for k in self.registry.keys() if k not in self.keys]
            return "every package except " + ", ".join(left_out)
        shown = [k for k in self.registry.keys() if k in self.keys]
        return ", ".join(shown[:6]) + (f" and {len(shown) - 6} more" if len(shown) > 6 else "")

    # ── the sweep ────────────────────────────────────────────────────────────

    def sweep_keys(self) -> list[str] | None:
        """The ``SUBSYSTEMS`` keys to sweep, or ``None`` for the whole surface."""
        if self.complete:
            return None
        return [k for p in self.registry if p.key in self.code_keys for k in p.symbol_keys()]

    # ── the suite ────────────────────────────────────────────────────────────

    def owns_file(self, name: str) -> bool:
        """Whether line coverage of the source file ``name`` is this run's to judge."""
        owners = self.registry.file_owners(name)
        if not owners:
            return not self.explicit
        return bool(owners & self.code_keys)

    def owns_module(self, name: str) -> bool:
        """Whether the docstring examples of module ``name`` are this run's to judge."""
        return bool(self.registry.module_owners(name) & self.code_keys)

    @property
    def file_scope(self) -> Callable[[str], bool] | None:
        return None if self.complete else self.owns_file

    @property
    def module_scope(self) -> Callable[[str], bool] | None:
        return None if self.complete else self.owns_module

    def suite_plan(
        self, root: Path, default: str = DEFAULT_SUITE_PATH, *, zoo_stage: bool = False
    ) -> SuitePlan:
        """What the suite stage runs.

        The zoo's own test trees are left out exactly when the zoo is not this
        stage's business: when the zoo stage runs them family by family with the
        markers cleared, or when the zoo is excluded.  A ``--tests-only`` run
        has neither, so it carries them as it always did.
        """
        scope = self.file_scope
        if default != DEFAULT_SUITE_PATH:
            return SuitePlan((default,), (), scope)
        if self.explicit and not self.complete:
            trees = (
                t for p in self.registry if p.key in self.code_keys for t in p.test_trees(root)
            )
            return SuitePlan(tuple(dict.fromkeys(trees)), (), scope)
        carried = self.code_keys | (self.keys & {MODELS} if not zoo_stage else set())
        kept = {t for p in self.registry if p.key in carried for t in p.test_trees(root)}
        left = (t for p in self.registry if p.key not in carried for t in p.test_trees(root))
        ignore = tuple(dict.fromkeys(t for t in left if t not in kept))
        return SuitePlan((default,), ignore, scope)


#: The registry every command line and every run is built from.
PACKAGES = PackageRegistry.default()

__all__ = [
    "DEFAULT_SUITE_PATH",
    "MODELS",
    "PACKAGES",
    "AuditPackage",
    "ModelZooPackage",
    "PackageRegistry",
    "PackageSelectionError",
    "Selection",
    "SubsystemPackage",
    "SuitePlan",
]
