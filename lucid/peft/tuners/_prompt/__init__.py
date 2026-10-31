"""The machinery prompt tuning, prefix tuning and p-tuning share (private).

``_encoder`` (what is learned), ``_sites`` (the hooks that put it in a call),
``_layout`` (which modules of a model get which hook, by task) and ``_tuner``
(the adapter holder).  A method is a config, an encoder and a three-line tuner
class in its own sub-package.
"""
