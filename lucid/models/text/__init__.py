"""Text model families — Phase 4 of the model zoo.

Houses BERT / GPT / GPT-2 / RoFormer and the shared base config /
activation alias.  Positional-encoding primitives (RoPE, sinusoidal PE) live
in :mod:`lucid.nn` — text families just import them from there.
"""

from lucid.models.text._common._config import LanguageModelConfig, TextActivation

__all__ = ["LanguageModelConfig", "TextActivation"]
