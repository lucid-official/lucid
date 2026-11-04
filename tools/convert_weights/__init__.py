"""One-off pretrained-weight conversion scripts.

Conversion scripts are one-off.  A port's script is written here and deleted
once its weights are published and ``tools.check_pretrained_parity`` agrees with
the source.  Nothing outside this package may import one: it will be gone.

A script may import the reference framework, because ``tools/`` sits outside the
runtime's no-external-dependency rule.  What it produces is pure Lucid
safetensors, and nothing here ships in the installed package.

Earlier converters and the shared base they built on can be found in git at
``e1858082ce92cf60debac4ff91811adf3debba06``::

    git ls-tree --name-only e1858082ce92cf60debac4ff91811adf3debba06 tools/convert_weights/
    git show e1858082ce92cf60debac4ff91811adf3debba06:tools/convert_weights/_base.py
"""
