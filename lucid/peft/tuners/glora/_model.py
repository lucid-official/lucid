"""``GloraModel``: the GLoRA tuner — its config and layer family; the rest is the generic tuner's."""

from lucid.peft._config import PeftType
from lucid.peft.tuners._support import _ConfigTuner
from lucid.peft.tuners.glora._config import GloraConfig
from lucid.peft.tuners.glora._layer import GloraLayer


class GloraModel(_ConfigTuner[GloraConfig], peft_type=PeftType.GLORA):
    r"""A model with GLoRA layers in it, and the tools to take them out again.

    Each ``nn.Linear`` ``config`` selects becomes a :class:`~lucid.peft.tuners.glora.Linear` whose
    weight and bias five learned support tensors correct, :math:`W + W A + B` and
    :math:`b + b D + E + W C`; the rest of the model is frozen.  Every support is a low-rank
    product, a vector, a scalar or absent as ``config_A_B``, ``config_C`` and ``config_D_E``
    say.  The model is edited in place and stays reachable as ``model``; several adapters
    coexist (``add_adapter``, ``set_adapter``, ``delete_adapter``), but the rows of a batch cannot
    be sent through different ones (the reference's ``adapter_names`` is not supported).

    Parameters
    ----------
    model : nn.Module
        The model to put GLoRA layers into.
    peft_config : GloraConfig
        Rank, the form of each support and which modules.
    adapter_name : str
        Name of the first adapter, the key in every ``glora_A``.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        Nothing matches ``target_modules``, a match is not an ``nn.Linear``, or nothing matches
        ``modules_to_save``.
    NotImplementedError
        The model already holds layers of another method.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import GloraConfig, GloraModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> glora = GloraModel(net, GloraConfig(r=4, target_modules=["0"]), "default")
    >>> glora.targeted_module_names
    ['0']
    >>> net[0].base_layer.weight.requires_grad, net[0].glora_A["default"].Xu.requires_grad
    (False, True)
    >>> glora.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _layer_types = GloraLayer._layer_types
