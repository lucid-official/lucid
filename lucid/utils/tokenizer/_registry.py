"""Tokenizers by name: the registry a class statement fills and a name or directory is resolved by.

A tokenizer class registers itself under its own name when it is defined, and
under any alias its class statement gives
(``class BERTTokenizer(WordPieceTokenizer, names=("bert",))``), so the zoo's
tokenizers are found without a table kept elsewhere.  Nothing is fetched: a name
resolves to a class, and the class is given what it needs to be built — the
vocabulary in the keyword arguments, or a directory it reads its files from.

A directory is resolved the way a published one is: its ``tokenizer_config.json``
names the class in ``tokenizer_class``.
"""

import inspect
import json
from collections.abc import Callable
from pathlib import Path

from lucid.utils.tokenizer._base import Tokenizer, registry_key

_CONFIG_FILE = "tokenizer_config.json"


class _Registry:
    """The tokenizer classes by name."""

    @classmethod
    def names(cls) -> list[str]:
        """The registered keys, sorted."""
        return sorted(Tokenizer._named)

    @classmethod
    def lookup(cls, name: str) -> type[Tokenizer]:
        """The class registered under ``name``.

        Raises
        ------
        ValueError
            No tokenizer is registered under ``name``.
        """
        found = Tokenizer._named.get(registry_key(name))
        if found is None:
            raise ValueError(
                f"no tokenizer is registered as {name!r}; registered: {', '.join(cls.names())}. "
                "A name is resolved without a download, so a hub id is not one; the zoo's "
                "tokenizers register when lucid.models is imported"
            )
        return found

    @classmethod
    def build(cls, name_or_path: str, **kwargs: object) -> Tokenizer:
        """The tokenizer ``name_or_path`` names, built from ``kwargs``.

        Parameters
        ----------
        name_or_path : str
            A registered name, or a directory whose ``tokenizer_config.json`` has the
            registered name of its class under ``tokenizer_class``.
        **kwargs
            For a name, the class's constructor arguments (its ``vocab``, say); for a
            directory, the keyword arguments of the class's ``from_pretrained``.

        Raises
        ------
        ValueError
            The name is not registered, the directory does not name its class, or the
            class cannot be built from ``kwargs`` (a name alone has no vocabulary).
        """
        directory = Path(name_or_path)
        if directory.is_dir():
            return cls._read(directory, kwargs)
        # Each algorithm's constructor takes its own arguments, not the base's.
        tokenizer: Callable[..., Tokenizer] = cls.lookup(name_or_path)
        try:
            inspect.signature(tokenizer).bind(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"tokenizer {name_or_path!r} cannot be built from {sorted(kwargs)}: {exc}; a name "
                "ships no vocabulary, so give it in the keyword arguments or name a directory"
            ) from exc
        return tokenizer(**kwargs)

    @classmethod
    def _read(cls, directory: Path, kwargs: dict[str, object]) -> Tokenizer:
        config = directory / _CONFIG_FILE
        if not config.is_file():
            raise ValueError(
                f"{directory} has no {_CONFIG_FILE}, which names the tokenizer's class "
                "(tokenizer_class) in a directory"
            )
        named = json.loads(config.read_text(encoding="utf-8")).get("tokenizer_class")
        if not isinstance(named, str):
            raise TypeError(f"{config}: tokenizer_class must be a string, got {named!r}")
        tokenizer = cls.lookup(named)
        # ``from_pretrained`` is each algorithm's own (their keyword arguments differ), not the base's.
        loader = getattr(tokenizer, "from_pretrained", None)
        if loader is None:
            raise ValueError(f"{tokenizer.__name__} cannot be read from a directory")
        built: Tokenizer = loader(str(directory), **kwargs)
        return built


def resolve_tokenizer(name_or_path: str, **kwargs: object) -> Tokenizer:
    """A tokenizer by registered name (built from ``kwargs``) or by directory (see :class:`_Registry`)."""
    return _Registry.build(name_or_path, **kwargs)
