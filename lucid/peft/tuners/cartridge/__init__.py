r"""``lucid.peft.tuners.cartridge`` — Cartridge: a trained key-value cache that stands in for a long document.

Cartridge (Eyuboglu et al., 2025, arXiv:2506.06266) is a trained key-value cache: ``num_virtual_tokens``
learned key and value pairs per attention layer, put in front of the model's own in place of a long
document it would otherwise read at every query.  It is trained by self-study, with the model
answering questions about the document, and a Cartridge for a corpus costs a fraction of the memory
of the corpus in context.  Pick it for serving many queries over one large, unchanging document.

The cache is laid out and applied as prefix tuning's is, but it is stored directly, not produced by
a projection, so a Cartridge can be started from the keys and values the model makes of a text
(``initialize_kv_prefix_from_text``) and several can be joined (``compose_cartridge_adapters``).  The
first ``num_frozen_tokens`` rows are never trained: the paper keeps one as an attention sink.

Here: ``CartridgeEncoder`` (the cache, trained) and ``CartridgeModel`` (the tuner), the config, and
the four utilities of ``lucid.peft.tuners.cartridge.utils``.  ``CartridgeConfig``, ``CartridgeEncoder``
and the utilities are exported from ``lucid.peft``, as the reference exports them; the tuner is not,
the reference having no model class.  Importing the package registers the method (its config under
``PeftType.CARTRIDGE``, its tuner and encoder), which is what makes ``get_peft_model`` and
``PeftModel.from_pretrained`` know it.
"""

from lucid.peft.tuners.cartridge._config import CartridgeConfig
from lucid.peft.tuners.cartridge._model import CartridgeEncoder, CartridgeModel
from lucid.peft.tuners.cartridge.utils import (
    compose_cartridge_adapters,
    initialize_kv_prefix_from_past_key_values,
    initialize_kv_prefix_from_text,
    prompt_embeddings_from_past_key_values,
)

__all__ = [
    "CartridgeConfig",
    "CartridgeEncoder",
    "CartridgeModel",
    "compose_cartridge_adapters",
    "initialize_kv_prefix_from_past_key_values",
    "initialize_kv_prefix_from_text",
    "prompt_embeddings_from_past_key_values",
]
