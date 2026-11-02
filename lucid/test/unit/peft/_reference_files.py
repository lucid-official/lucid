"""An adapter directory as the reference library writes it, for tests that load one.

``adapter_config.json`` carries every field of the reference's ``LoraConfig`` (read from
``_reference_signatures.json``, the snapshot of release 0.21.2), not the handful a test
cares about, so a field this library cannot read fails here and not on a user's first
``from_pretrained``.  ``adapter_model.safetensors`` holds ``base_model.model.<module>.lora_A.weight``
style keys with the format metadata the reference writes.
"""

import ast
import json
from pathlib import Path
from typing import Final

import lucid
from lucid.serialization import load_safetensors, save_safetensors

MODEL_PREFIX: Final = "base_model.model."

_SNAPSHOT: Final = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())
_FACTORY: Final = "<factory>:"
_EMPTY: Final = frozenset({"dict", "list", "set"})


def _fields(class_name: str) -> dict[str, object]:
    """The defaults of the reference dataclass ``class_name``, as ``to_dict`` writes them."""
    return {
        name: _default(default)
        for name, _kind, has_default, default in _SNAPSHOT["fields"][class_name]
        if has_default
    }


def _default(spelled: str) -> object:
    """A dataclass field default as the snapshot spells it: a literal or a ``<factory>:`` call."""
    if not spelled.startswith(_FACTORY):
        return ast.literal_eval(spelled)
    made = spelled.removeprefix(_FACTORY)
    return {"dict": {}, "list": [], "set": []}.get(made) if made in _EMPTY else _fields(made)


def reference_lora_config(
    class_name: str, library: str, targets: list[str], **fields: object
) -> dict[str, object]:
    """The ``adapter_config.json`` of a LoRA adapter, every field the reference's config has.

    Parameters
    ----------
    class_name : str
        The reference model class the adapter was trained on, recorded in ``auto_mapping``.
    library : str
        The module that class lives in (``parent_library``).
    targets : list[str]
        ``target_modules`` as the reference names them.
    **fields : object
        Fields to set over the reference's defaults.
    """
    config = _fields("LoraConfig")
    config.update(
        peft_type="LORA",
        peft_version=_SNAPSHOT["peft_version"],
        auto_mapping={"base_model_class": class_name, "parent_library": library},
        base_model_name_or_path=f"reference/{class_name}",
        inference_mode=True,
        target_modules=sorted(targets),
    )
    config.update(fields)
    return config


def lora_tensors(
    module: str, a: lucid.Tensor, b: lucid.Tensor, prefix: str = MODEL_PREFIX
) -> dict[str, lucid.Tensor]:
    """The two factor tensors of the LoRA module ``module`` under the keys the reference writes."""
    return {f"{prefix}{module}.lora_A.weight": a, f"{prefix}{module}.lora_B.weight": b}


def write_adapter(
    directory: Path, config: dict[str, object], tensors: dict[str, lucid.Tensor]
) -> Path:
    """``directory`` holding ``adapter_config.json`` and ``adapter_model.safetensors``."""
    directory.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        tensors, str(directory / "adapter_model.safetensors"), metadata={"format": "pt"}
    )
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return directory


def read_keys(directory: Path) -> set[str]:
    """The tensor keys of the adapter ``directory`` holds."""
    return set(load_safetensors(str(directory / "adapter_model.safetensors")))
