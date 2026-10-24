"""Conversion engine: source state_dict → verified Lucid safetensors.

Architecture-agnostic core.  Concrete architectures (e.g.
:mod:`tools.convert_weights.resnet`) subclass :class:`Architecture` to
supply three things: how to fetch the upstream weights, how to build the
matching empty Lucid model, and how to remap each key.  Everything else
— key 1:1 verification, shape checks, a real ``load_state_dict`` smoke
load, safetensors writing with embedded metadata, and ``config.json`` /
``README.md`` rendering — is handled here.
"""

import abc
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import lucid
import lucid.serialization as _serial
from lucid.nn import Module

from tools.convert_weights._templates import (
    _SIBLING_LOOKUP,
    render_config_json,
    render_model_card,
)


@dataclass
class ConversionSpec:
    """Static metadata describing one checkpoint conversion.

    Carries everything the templates + safetensors header need that is
    not derivable from the weights themselves.
    """

    model_name: str
    """Lucid factory the weights load into, e.g. ``"resnet_18_cls"``."""
    architecture: str
    """Canonical short name used in URLs / cards, e.g. ``"resnet_18"``."""
    repo_id: str
    """Hub repo, e.g. ``"lucid-dl/resnet-18"``."""
    tag: str
    """Variant tag, e.g. ``"IMAGENET1K_V1"``."""
    task: str
    """HF pipeline tag, e.g. ``"image-classification"``."""
    model_type: str
    """Lucid config ``model_type``, e.g. ``"resnet"``."""
    source: str
    """Provenance string, e.g. ``"torchvision/ResNet18_Weights.IMAGENET1K_V1"``."""
    license: str
    """SPDX-ish license id inherited from the source weights."""
    num_classes: int
    """Output-class count of the head."""
    config: dict[str, object]
    """Lucid config dataclass dump (architecture hyperparameters)."""
    preprocessing: dict[str, object]
    """Inference transform spec (resize/crop/mean/std/interpolation)."""
    citation: str
    """Paper citation (BibTeX or free text) for the model card."""
    title: str
    """Human title, e.g. ``"ResNet-18"``."""
    paper_url: str = ""
    """Optional paper link rendered in the card header."""
    categories: list[str] = field(default_factory=list)
    """Class label names (e.g. ImageNet-1k), used to build id2label."""
    datasets: list[str] = field(default_factory=list)
    """Dataset slugs used during training (e.g. ``["imagenet-22k",
    "imagenet-1k"]`` for an in22k → in1k finetune).  Emitted into the
    Hub model-card's YAML ``datasets:`` frontmatter so HF dataset
    discovery picks up every dataset the checkpoint touched, not just
    the evaluation set.  Empty list = fall back to the metrics' first
    key (backward-compat for the simple single-dataset case)."""
    meta: dict[str, object] = field(default_factory=dict)
    """Metrics + recipe + flops, merged into config.json + card."""


class UnknownSourceKey(KeyError):
    """A source key that :meth:`Architecture.map_key` has no rule for.

    ``map_key`` returning ``None`` is a decision to drop a key; raising
    this says the recipe never decided anything about it.  :func:`convert`
    collects every such key and refuses the checkpoint, because a key the
    recipe does not know is a tensor the converted model would silently
    lack or a layout the recipe was not written for.
    """


class Architecture(abc.ABC):
    """Per-architecture conversion recipe.

    Subclasses bind a concrete upstream checkpoint (via ``__init__``
    args like ``tag``) and implement the three abstract hooks.
    """

    @abc.abstractmethod
    def source_state_dict(self) -> Mapping[str, object]:
        """Return the upstream ``state_dict`` as ``{name: numpy.ndarray}``."""
        raise NotImplementedError

    @abc.abstractmethod
    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        raise NotImplementedError

    @abc.abstractmethod
    def map_key(self, src_key: str) -> str | None:
        """Map an upstream key to its Lucid name.

        ``None`` drops the key on purpose.  A recipe that knows its
        source's whole key set raises :class:`UnknownSourceKey` for
        anything else, so that :func:`convert` refuses it instead of
        dropping it.
        """
        raise NotImplementedError

    def transform_value(self, src_key: str, arr: object) -> object:
        """Optional hook — reshape / cast the raw upstream array per key.

        Default identity.  Override on subclasses that need to massage a
        few specific tensors before they land in Lucid (e.g. ConvNeXt's
        ``layer_scale`` ships as ``(C, 1, 1)`` for use in NCHW
        broadcasting but Lucid stores it as ``(C,)`` for an explicit
        elementwise multiply).
        """
        return arr

    @abc.abstractmethod
    def spec(self) -> ConversionSpec:
        """Return the static :class:`ConversionSpec` for this checkpoint."""
        raise NotImplementedError


def convert(arch: Architecture) -> tuple[dict[str, lucid.Tensor], ConversionSpec]:
    """Remap + verify an architecture's weights into a Lucid state dict.

    Performs the full safety gauntlet: key remap, 1:1 key-set check
    against a freshly-built Lucid model, per-key shape check, and a real
    ``load_state_dict(strict=True)`` smoke load.  Any discrepancy raises
    before anything is written.

    Parameters
    ----------
    arch : Architecture
        The conversion recipe.

    Returns
    -------
    (dict, ConversionSpec)
        The Lucid state dict (``{lucid_key: Tensor}``) and the spec.

    Raises
    ------
    RuntimeError
        On source keys the recipe has no rule for, missing / extra keys,
        or a shape mismatch.
    """
    src = arch.source_state_dict()
    model = arch.target_model()
    target_sd = model.state_dict()
    target_keys = set(target_sd.keys())

    out: dict[str, lucid.Tensor] = {}
    unknown: list[str] = []
    for src_key, arr in src.items():
        try:
            lucid_key = arch.map_key(src_key)
        except UnknownSourceKey:
            unknown.append(src_key)
            continue
        if lucid_key is None:
            continue
        if lucid_key in out:
            raise RuntimeError(
                f"convert: key collision — two source keys map to {lucid_key!r}"
            )
        out[lucid_key] = lucid.tensor(arch.transform_value(src_key, arr))

    if unknown:
        raise RuntimeError(
            f"convert: {len(unknown)} source key(s) the "
            f"{arch.spec().model_name} recipe has no rule for: "
            f"{sorted(unknown)[:10]}"
        )

    got = set(out)
    missing = sorted(target_keys - got)
    extra = sorted(got - target_keys)
    if missing or extra:
        raise RuntimeError(
            f"convert: key-set mismatch for {arch.spec().model_name}.\n"
            f"  missing ({len(missing)}): {missing[:10]}\n"
            f"  extra   ({len(extra)}): {extra[:10]}"
        )

    for key in sorted(got):
        want = tuple(int(d) for d in target_sd[key].shape)
        have = tuple(int(d) for d in out[key].shape)
        if want != have:
            raise RuntimeError(
                f"convert: shape mismatch at {key!r}: target {want}, got {have}"
            )

    # Real load — the ultimate correctness gate.
    model.load_state_dict(out, strict=True)
    return out, arch.spec()


def _sha256_of(path: Path) -> str:
    """Stream-hash a file in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(
    state_dict: Mapping[str, lucid.Tensor],
    spec: ConversionSpec,
    out_root: str,
) -> Path:
    """Write ``model.safetensors`` + ``config.json`` + ``README.md``.

    Layout::

        {out_root}/{repo_id}/{tag}/model.safetensors
        {out_root}/{repo_id}/{tag}/config.json
        {out_root}/{repo_id}/README.md            (repo-level card)

    The safetensors file embeds a flat ``__metadata__`` header (format,
    architecture, tag, source, license, sha-less scalar facts); the
    full rich ``meta`` is JSON-encoded into ``config.json``.  ``sha256``
    is computed post-write and injected into both the card and the
    returned spec's meta.

    Parameters
    ----------
    state_dict : dict
        Verified Lucid state dict from :func:`convert`.
    spec : ConversionSpec
        Static metadata.
    out_root : str
        Output root directory (e.g. ``"./_converted"``).

    Returns
    -------
    pathlib.Path
        The per-tag directory that was written.
    """
    repo_dir = Path(out_root) / spec.repo_id
    tag_dir = repo_dir / spec.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    st_path = tag_dir / "model.safetensors"
    header = {
        "format": "lucid",
        "lucid_version": str(lucid.__version__),
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "tag": spec.tag,
        "source": spec.source,
        "license": spec.license,
        "num_classes": str(spec.num_classes),
    }
    # save_safetensors checks each value itself, so it takes a
    # ``dict[str, object]``, which a dict of tensors is not (dict is
    # invariant); this copy is that boundary.
    tensors: dict[str, object] = dict(state_dict)
    _serial.save_safetensors(tensors, str(st_path), metadata=header)

    sha = _sha256_of(st_path)
    file_size_mb = round(st_path.stat().st_size / (1024 * 1024), 2)
    spec.meta = {**spec.meta, "sha256": sha, "file_size_mb": file_size_mb}

    (tag_dir / "config.json").write_text(render_config_json(spec), encoding="utf-8")
    # The card is repo-level but rendered from one tag's spec, so with two
    # tags the second overwrote the first and the table advertised the
    # wrong default.  Sibling tag directories are what the repo actually
    # holds, so they are what the table lists.
    siblings = sorted(
        d.name
        for d in repo_dir.iterdir()
        if d.is_dir() and (d / "config.json").is_file()
    )
    lookup: dict[str, tuple[dict[str, object], str]] = {}
    for tag in siblings:
        if tag == spec.tag:
            continue
        payload = json.loads((repo_dir / tag / "config.json").read_text())
        weights = payload.get("weights", {})
        lookup[tag] = (dict(weights), str(weights.get("source", "")))
    _SIBLING_LOOKUP.clear()
    _SIBLING_LOOKUP.update(lookup)
    (repo_dir / "README.md").write_text(
        render_model_card(spec, siblings), encoding="utf-8"
    )
    return tag_dir


def upload(tag_dir: Path, spec: ConversionSpec) -> str:
    """Upload a converted checkpoint folder to the Hugging Face Hub.

    Pushes the per-tag folder into ``spec.repo_id`` under the tag
    subfolder, and the repo-level ``README.md`` to the repo root.
    Requires ``huggingface-cli login`` or ``HF_TOKEN`` in the env.

    Parameters
    ----------
    tag_dir : pathlib.Path
        The per-tag directory produced by :func:`write`.
    spec : ConversionSpec
        Static metadata (provides ``repo_id`` + ``tag``).

    Returns
    -------
    str
        The Hub URL of the updated repo.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(spec.repo_id, repo_type="model", exist_ok=True)
    api = HfApi()
    # Tag subfolder (weights + per-tag config).
    api.upload_folder(
        repo_id=spec.repo_id,
        folder_path=str(tag_dir),
        path_in_repo=spec.tag,
        commit_message=f"Add {spec.tag} weights",
    )
    # Repo-level card.
    readme = tag_dir.parent / "README.md"
    if readme.is_file():
        api.upload_file(
            repo_id=spec.repo_id,
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message="Update model card",
        )
    return f"https://huggingface.co/{spec.repo_id}"


# Registry of convertible architectures, populated by the per-arch modules.
_ARCH_BUILDERS: dict[str, Callable[[str], Architecture]] = {}


def register_arch(
    model_name: str,
) -> Callable[[Callable[[str], Architecture]], Callable[[str], Architecture]]:
    """Register an ``Architecture`` builder under a model name for the CLI."""

    def _decorator(
        builder: Callable[[str], Architecture],
    ) -> Callable[[str], Architecture]:
        _ARCH_BUILDERS[model_name] = builder
        return builder

    return _decorator


def get_arch(model_name: str, tag: str) -> Architecture:
    """Look up + instantiate a registered architecture for ``tag``."""
    if model_name not in _ARCH_BUILDERS:
        raise KeyError(
            f"get_arch: no converter registered for {model_name!r}. "
            f"Known: {sorted(_ARCH_BUILDERS)}"
        )
    return _ARCH_BUILDERS[model_name](tag)
