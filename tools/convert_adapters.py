"""Convert an adapter directory written for the reference library's model to the zoo's layout.

::

    python -m tools.convert_adapters SRC DST [--model-type t5] [--reference-class T5ForConditionalGeneration]

``SRC`` holds ``adapter_config.json`` and ``adapter_model.safetensors`` as the
reference library wrote them; ``DST`` receives the same two files with the
module names (and, where the layouts differ, the tensors — ViT's separate
``query`` / ``key`` / ``value`` factors fused into the zoo's ``qkv``) the zoo's
models carry, and ``target_modules`` rewritten to match.  The converter is
the one ``PeftModel.from_pretrained(..., convert=...)`` uses; it is picked by
``--model-type`` (a zoo ``model_type``) or, without it, by the class name the
adapter's ``auto_mapping`` records.

A conversion that needs the model's weights (DoRA on some of ``query`` /
``key`` / ``value``) is refused here: load the directory with
``PeftModel.from_pretrained`` instead, which converts as it loads.
"""

import argparse
import os
import sys
from collections.abc import Sequence

from lucid.peft._config import PeftConfig, _output_directory
from lucid.peft._convert import AdapterConverter, AdapterFile, ConversionContext, convert_adapter
from lucid.peft._peft_model import _WEIGHTS_NAME, _adapter_weights
from lucid.serialization import save_safetensors


def convert_directory(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    model_type: str | None = None,
    reference_class: str | None = None,
) -> AdapterFile:
    """Write the zoo-layout copy of the adapter in ``source`` into ``destination``; return it.

    Raises
    ------
    ValueError
        ``destination`` is ``source``, or no converter serves the adapter.
    UnmappedKey
        The adapter holds a tensor the converter has no rule for.
    NotImplementedError
        The adapter's method, or a case of it, cannot be converted.
    """
    source_dir = os.fspath(source)
    if os.path.realpath(source_dir) == os.path.realpath(os.fspath(destination)):
        raise ValueError("the destination is the source directory; converting is not in place")
    config = PeftConfig.from_pretrained(source_dir)
    named = reference_class or (config.auto_mapping or {}).get("base_model_class")
    context = ConversionContext(model_type=model_type, reference_class=named)
    converted = convert_adapter(AdapterFile(config, _adapter_weights(source_dir)), context, True)
    target = _output_directory(destination)
    save_safetensors(
        dict(converted.tensors), os.path.join(target, _WEIGHTS_NAME), metadata={"format": "pt"}
    )
    converted.config.save_pretrained(target)
    return converted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="convert_adapters", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="adapter directory in the reference layout")
    parser.add_argument("destination", help="directory to write the zoo-layout adapter into")
    parser.add_argument(
        "--model-type",
        choices=AdapterConverter.registered(),
        help="the zoo model_type to convert for (default: from the adapter's auto_mapping)",
    )
    parser.add_argument(
        "--reference-class",
        help="the reference library's class the adapter was trained on, when auto_mapping lacks it "
        "(T5: T5ForConditionalGeneration nests the trunk under 't5.')",
    )
    args = parser.parse_args(argv)
    try:
        converted = convert_directory(
            args.source,
            args.destination,
            model_type=args.model_type,
            reference_class=args.reference_class,
        )
    except (ValueError, NotImplementedError, FileNotFoundError) as error:
        print(f"convert_adapters: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(converted.tensors)} tensors to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
