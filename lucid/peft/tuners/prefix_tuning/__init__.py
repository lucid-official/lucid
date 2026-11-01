"""``lucid.peft.tuners.prefix_tuning`` — prefix tuning: learned keys and values per attention layer.

Prefix tuning (Li and Liang, 2021, arXiv:2101.00190) learns ``num_virtual_tokens`` key and value
vectors for every attention layer and puts them in front of that layer's own, so the whole network
can attend to them: ``num_virtual_tokens * 2 * num_layers * token_dim`` numbers.  With
``prefix_projection`` they are produced from a smaller embedding by a two-layer MLP while training,
which the paper found steadier; the file still stores the finished vectors.  Pick it over prompt
tuning when a prompt at the input is too weak (generation tasks, smaller models), at the price of
more parameters.

The prefix is applied by hooks on the model's own call.  Here: ``PrefixEncoder`` (what is learned)
and ``PrefixTuningModel`` (the tuner).  ``PrefixTuningConfig`` is here too and exported from
``lucid.peft``.
"""

from lucid.peft.tuners.prefix_tuning._config import PrefixTuningConfig
from lucid.peft.tuners.prefix_tuning._model import PrefixEncoder, PrefixTuningModel

__all__ = ["PrefixEncoder", "PrefixTuningConfig", "PrefixTuningModel"]
