"""CLI entry point for the weight-conversion pipeline.

Examples
--------
::

    # convert + write locally (inspect before uploading)
    python -m tools.convert_weights resnet_18 --tag IMAGENET1K_V1

    # convert + upload to the Hub
    python -m tools.convert_weights resnet_18 --tag IMAGENET1K_V1 --upload
"""

import argparse
import sys

# Importing the per-architecture modules registers their converters.
import tools.convert_weights.alexnet  # noqa: F401
import tools.convert_weights.bert  # noqa: F401
import tools.convert_weights.clip  # noqa: F401
import tools.convert_weights.convnext  # noqa: F401
import tools.convert_weights.crossvit  # noqa: F401
import tools.convert_weights.cspnet  # noqa: F401
import tools.convert_weights.cvt  # noqa: F401
import tools.convert_weights.ddpm  # noqa: F401
import tools.convert_weights.densenet  # noqa: F401
import tools.convert_weights.diamond  # noqa: F401
import tools.convert_weights.dit  # noqa: F401
import tools.convert_weights.detr  # noqa: F401
import tools.convert_weights.efficientformer  # noqa: F401
import tools.convert_weights.efficientnet  # noqa: F401
import tools.convert_weights.faster_rcnn  # noqa: F401
import tools.convert_weights.fcn  # noqa: F401
import tools.convert_weights.googlenet  # noqa: F401
import tools.convert_weights.gpt  # noqa: F401
import tools.convert_weights.gpt2  # noqa: F401
import tools.convert_weights.inception  # noqa: F401
import tools.convert_weights.mask2former  # noqa: F401
import tools.convert_weights.mask_rcnn  # noqa: F401
import tools.convert_weights.maskformer  # noqa: F401
import tools.convert_weights.inception_next  # noqa: F401
import tools.convert_weights.inception_resnet  # noqa: F401
import tools.convert_weights.inception_v4  # noqa: F401
import tools.convert_weights.maxvit  # noqa: F401
import tools.convert_weights.mobilenet  # noqa: F401
import tools.convert_weights.mobilenet_v2  # noqa: F401
import tools.convert_weights.mobilenet_v3  # noqa: F401
import tools.convert_weights.mobilenet_v4  # noqa: F401
import tools.convert_weights.pvt  # noqa: F401
import tools.convert_weights.resnest  # noqa: F401
import tools.convert_weights.resnet  # noqa: F401
import tools.convert_weights.resnext  # noqa: F401
import tools.convert_weights.roformer  # noqa: F401
import tools.convert_weights.senet  # noqa: F401
import tools.convert_weights.sknet  # noqa: F401
import tools.convert_weights.swin  # noqa: F401
import tools.convert_weights.tvm  # noqa: F401
import tools.convert_weights.vgg  # noqa: F401
import tools.convert_weights.vit  # noqa: F401
import tools.convert_weights.vjepa2  # noqa: F401
import tools.convert_weights.vjepa2_ac  # noqa: F401
import tools.convert_weights.xception  # noqa: F401
import tools.convert_weights.yolo  # noqa: F401
from tools.convert_weights._base import convert, get_arch, upload, write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert_weights",
        description="Convert upstream pretrained weights to Lucid safetensors.",
    )
    parser.add_argument("model", help="Architecture name, e.g. 'resnet_18'.")
    parser.add_argument("--tag", required=True, help="Weight tag, e.g. 'IMAGENET1K_V1'.")
    parser.add_argument(
        "--out", default="_converted", help="Output root directory (default: ./_converted)."
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload the converted folder to the Hugging Face Hub."
    )
    args = parser.parse_args(argv)

    arch = get_arch(args.model, args.tag)
    print(f"[convert] {args.model} :: {args.tag}")
    state_dict, spec = convert(arch)
    print(f"[convert] OK — {len(state_dict)} tensors, keys verified + loaded")

    tag_dir = write(state_dict, spec, args.out)
    sha = spec.meta.get("sha256", "")
    size = spec.meta.get("file_size_mb", "")
    print(f"[write]   {tag_dir}")
    print(f"[write]   sha256 = {sha}")
    print(f"[write]   size   = {size} MB")
    arch_pkg = args.model.split("_")[0]
    print(f"\n  → paste into lucid/models/<domain>/{arch_pkg}/_weights.py:")
    print(f'      sha256="{sha}",')

    if args.upload:
        url = upload(tag_dir, spec)
        print(f"[upload]  {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
