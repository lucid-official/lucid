"""Reference parameters that ``lucid.peft`` accepts by name and refuses by value.

The API follows the reference adapter library's signatures, so a parameter
that means nothing here (a flag for a feature the library lacks) still has
its place in the signature, with the reference's default.  Passing it at that
default changes nothing; passing anything else is refused with
``NotImplementedError`` naming the parameter, never silently ignored.
"""

from collections.abc import Mapping
from typing import NoReturn


def refuse_value(where: str, name: str, value: object, default: object, why: str) -> None:
    """Raise ``NotImplementedError`` unless ``value`` is the reference's ``default``.

    Parameters
    ----------
    where : str
        The callable or class the parameter belongs to (``"PeftModel.from_pretrained"``).
    name : str
        The parameter's name, as the reference spells it.
    value : object
        What the caller passed.
    default : object
        The reference's default for the parameter.
    why : str
        One clause saying what the parameter would do and why it cannot here.
    """
    if value is default or value == default:
        return
    raise NotImplementedError(f"{where}: `{name}={value!r}` is not supported by lucid.peft: {why}")


def refuse_method(where: str, why: str) -> NoReturn:
    """Raise ``NotImplementedError`` for a reference method that has no meaning here.

    For a member the reference offers whose whole job is a feature this library lacks (a hub model
    card).  The member stays, with the reference's signature, so that code which probes for it
    finds it and is told why calling it cannot work, rather than meeting an ``AttributeError``.

    Parameters
    ----------
    where : str
        The method (``"PeftModel.create_or_update_model_card"``).
    why : str
        One clause saying what it would do and why it cannot here.
    """
    raise NotImplementedError(f"{where}() is not supported by lucid.peft: {why}")


def refuse_kwargs(where: str, kwargs: Mapping[str, object], why: str) -> None:
    """Raise ``NotImplementedError`` naming every key of ``kwargs``, if there is one.

    For a reference ``**kwargs`` that only forwards options of a feature this
    library lacks (hub download, ``push_to_hub``, a transformers loader).

    Parameters
    ----------
    where : str
        The callable the ``**kwargs`` belong to.
    kwargs : Mapping of str to object
        The extra keyword arguments the caller passed.
    why : str
        One clause saying what they would have configured.
    """
    if kwargs:
        raise NotImplementedError(
            f"{where}: {', '.join(f'`{k}`' for k in sorted(kwargs))} "
            f"{'is' if len(kwargs) == 1 else 'are'} not supported by lucid.peft: {why}"
        )


#: Why the options several reference signatures share are refused, said once.
NO_META_DEVICE = (
    "adapters are built in place on the model's device; there is no meta-device initialisation"
)
NO_STATE_DICT_MATCHING = "targets are matched against the model's modules, not against a state dict"
NO_FLOAT_AUTOCAST_OFF = "a half-precision base always gets a float32 adapter"
NO_OFFLOAD = "adapter layers are never offloaded between devices"
NO_HUB = "adapters are read from a local directory; there is no hub download"
NO_MISMATCH_SKIPPING = "a tensor whose shape does not fit the model raises; it is never skipped"
NO_MODEL_CARD = (
    "a model card is the README of a hub repository, which an adapter saved to a local directory "
    "does not have"
)
NO_HYBRID_CACHE = "a prompt's cache grows with the sequence; there is no fixed-size cache to size"
NO_MIXED_SAVE = (
    "a mixed model has no file layout of its own (the reference has none either): write each "
    "adapter from a PeftModel of its own method, and read it back with load_adapter"
)


def reject_unexpected(where: str, kwargs: Mapping[str, object]) -> None:
    """Raise ``TypeError`` naming every key of ``kwargs``, as Python does for an unknown keyword.

    For a reference ``**kwargs`` whose known keys the caller has already taken out.

    Parameters
    ----------
    where : str
        The callable the keywords were passed to.
    kwargs : Mapping of str to object
        What was left over.
    """
    if kwargs:
        raise TypeError(f"{where}() got unexpected keyword arguments {sorted(kwargs)}")


def refuse_build_options(
    where: str,
    *,
    autocast_adapter_dtype: bool = True,
    low_cpu_mem_usage: bool = False,
    ephemeral_gpu_offload: bool = False,
) -> None:
    """Refuse the options a reference signature offers for building an adapter that this library cannot honour.

    The three travel together through the reference's ``PeftModel`` constructor, ``add_adapter``,
    ``from_pretrained`` and ``load_adapter``; each is checked against its default.

    Parameters
    ----------
    where : str
        The callable they were passed to.
    autocast_adapter_dtype : bool, default=True
        Cast a half-precision base's adapter weights to float32.  Always the case here.
    low_cpu_mem_usage : bool, default=False
        Build the adapter on the meta device.
    ephemeral_gpu_offload : bool, default=False
        Keep the adapter off the accelerator between uses.
    """
    refuse_value(
        where, "autocast_adapter_dtype", autocast_adapter_dtype, True, NO_FLOAT_AUTOCAST_OFF
    )
    refuse_value(where, "ephemeral_gpu_offload", ephemeral_gpu_offload, False, NO_OFFLOAD)
    refuse_value(where, "low_cpu_mem_usage", low_cpu_mem_usage, False, NO_META_DEVICE)
