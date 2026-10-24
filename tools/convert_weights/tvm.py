"""TVM weight converter — the authors' training pickles → Lucid.

The three released checkpoints (``lumaai/tvm`` on the Hub) are not state
dicts.  Each is ``{"net": ...}``, a pickle of the training loop's
evaluation EMA, and two layers of wrapping sit between that entry and the
tensors:

* **The compile wrapper.**  The training loop compiles the network before
  it deep-copies the EMA, so the pickled object is an ``OptimizedModule``,
  which pickles as ``OptimizedModule(wrapped_module, compile_context)``.
* **The persistence scheme.**  The wrapped ``TVMWrapper`` and every layer
  of its DiT are written with the ``torch_utils.persistence`` scheme
  StyleGAN introduced: each decorated object is stored *with the source
  of its module*, and unpickling ``exec``\\s that source.  Loading one the
  ordinary way therefore needs the authors' repository importable — and
  its ``training/dit.py`` imports a Triton attention kernel at module
  level, which does not build on Apple Silicon.

Nothing from the checkpoint runs here.  :class:`_ParameterUnpickler`
resolves every global the file names through three tables and refuses
any global that is in none of them:

* ``_REAL`` — the tensor rebuilders ``OrderedDict``,
  ``torch._utils._rebuild_parameter`` and ``_rebuild_tensor_v2``, which
  only assemble tensors from their arguments.
* ``_SPECIAL`` — three hooks replaced by functions of this module: the
  persistence hook keeps an object's pickled attributes and ignores its
  source; the compile wrapper returns the module it wraps; and the
  storage loader reads a storage's nested bytes with the framework's
  weights-only loader.  The release's own storage loader,
  ``torch.storage._load_from_bytes``, is an unrestricted ``torch.load``.
* ``_INERT`` — the classes the network tree, the training configuration
  its arguments carry, and the compile context are built from.  Each
  resolves to :class:`_Record`, which stores attributes and runs nothing.
  The compile context's own classes change with every compiler release
  and the context is discarded, so its package is admitted by prefix
  (``_INERT_PACKAGES``); every other entry is an exact name.

The tables are what the released 512-pixel file names — read from its
first 400 kB and last 60 kB — plus the layer classes of a replica pickled
with the authors' code, which is where the rest of the file's globals
come from.  An unexpected global means the file is not shaped like the
release, and it is reported by name rather than guessed at.

:func:`_flatten` walks ``_parameters`` / ``_buffers`` / ``_modules``
through the resulting tree into the usual dotted names.  Three checks
then guard what a key map cannot see:

* **The training flags match the factory.**  The wrapper records its
  ``do_scale_param`` and the network its ``learn_guidance``; a checkpoint
  whose scaled parameterisation disagrees with the config it is loaded
  into would produce velocities off by a factor of ``w`` with no shape
  error to say so.
* **The layers ran the factory's architecture.**  Attention heads and the
  modulation norm have no parameters of their own, so a checkpoint with
  the wrong ``num_heads`` or ``do_mod_norm`` loads cleanly and computes a
  different function.  Both are read from the layers themselves
  (``Attention.num_heads``, ``DiTBlock.do_mod_norm``,
  ``FinalLayer.do_mod_norm``) rather than from the persistence
  ``_init_kwargs``: the released wrapper hands the configuration's
  ``model_kwargs`` to the DiT as one keyword its ``**kwargs`` swallows, so
  the recorded arguments do not say what the layers were built with.
* **The positional table matches Lucid's.**  The reference stores its
  frozen sine-cosine table as a parameter; Lucid rebuilds it from the
  config.  The stored one is compared against
  :func:`~lucid.models.generative._common._transformers.sincos_position_embedding`
  and then dropped, so a convention drift (the column-first ordering)
  fails here rather than as a quietly degraded sample.

No released checkpoint has been converted yet: the comparison against
the authors' sampler on real weights (H12 step 11, the published-weights
check) waits on the first real run.  ``tools/check_tvm_upstream_parity.py``
compares the two implementations on random weights.

Licence: the Hub card says Apache-2.0 but the ``LICENSE`` file beside the
weights, and the code repository's, is CC-BY-NC-SA-4.0.  The converted
card carries the stricter of the two.
"""

import dataclasses
import io
import pickle
from collections.abc import Callable
from typing import override

import numpy as np
import numpy.typing as npt
import torch
from huggingface_hub import hf_hub_download

from lucid.models import AutoConfig
from lucid.models._utils._common import apply_overrides
from lucid.models.generative._common._config import patch_grid_side
from lucid.models.generative._common._transformers import sincos_position_embedding
from lucid.models.generative.tvm import TVMConfig, TVMModel
from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CITATION = (
    "@inproceedings{zhou2026terminal,\n"
    "  title={Terminal Velocity Matching},\n"
    "  author={Zhou, Linqi and Parger, Mathias and Haque, Ayaan and "
    "Song, Jiaming},\n"
    "  booktitle={International Conference on Learning Representations "
    "(ICLR)},\n"
    "  year={2026},\n"
    "  eprint={2511.19797},\n"
    "  archivePrefix={arXiv}\n"
    "}"
)

_REPO = "lumaai/tvm"


@dataclasses.dataclass(frozen=True)
class _Release:
    """One released checkpoint and the Lucid configuration it loads into."""

    factory: str
    filename: str
    overrides: dict[str, object]
    guidance: float
    fid: dict[str, float]


# Table 6 and the authors' ``configs/*.yaml``.  The ``w = 2.25`` 512-pixel
# run in the paper was never released.  ``overrides`` are applied to the
# factory's registered default config.
_RELEASES: dict[str, _Release] = {
    "IMAGENET1K_256_W2": _Release(
        factory="tvm_imagenet256",
        filename="im256_w2.pkl",
        overrides={},
        guidance=2.0,
        fid={"fid_1nfe": 3.29, "fid_2nfe": 2.80},
    ),
    "IMAGENET1K_256_W1P75": _Release(
        factory="tvm_imagenet256",
        filename="im256_w1p75.pkl",
        overrides={"guidance_scale": 1.75, "time_embedding_init": "normal"},
        guidance=1.75,
        fid={"fid_1nfe": 4.58, "fid_2nfe": 2.61, "fid_4nfe": 1.99},
    ),
    "IMAGENET1K_512_W2P5": _Release(
        factory="tvm_imagenet512",
        filename="im512_w2p5.pkl",
        overrides={},
        guidance=2.5,
        fid={"fid_1nfe": 4.32, "fid_2nfe": 3.50},
    ),
}


# ── loading a persistence pickle without its source ──────────────────────────


class _Record(dict[str, object]):
    """Inert stand-in for every class the converter will not import.

    A ``dict`` so that dictionary subclasses (``EasyDict``, a
    ``defaultdict``) unpickle their items into it; a ``__setstate__`` so
    that objects unpickle their attributes into ``__dict__``.  Either way
    nothing of the original class runs.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs  # the class it stands in for would have used them
        super().__init__()

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, dict):
            self.__dict__.update(state)


def _reconstruct_persistent(meta: object) -> _Record:
    """Replacement for ``torch_utils.persistence._reconstruct_persistent_obj``.

    The original ``exec``\\s ``meta["module_src"]`` and instantiates the
    class it defines; this keeps only ``meta["state"]``, the object's
    ``__dict__`` — which, for a network, is its parameters and children.
    """
    record = _Record()
    if isinstance(meta, dict):
        record.__setstate__(meta.get("state") or {})
    return record


def _unwrap_compiled(module: object, *context: object) -> object:
    """Replacement for the compile wrapper's class: the module it wraps.

    ``OptimizedModule`` pickles as ``(OptimizedModule, (module,
    compile_context))``.  The context is the compiler's configuration,
    which a conversion has no use for.
    """
    del context
    return module


# Before 2.6 the weights-only loader could be bypassed through the legacy
# format a nested storage uses (CVE-2025-32434).
_WEIGHTS_ONLY_SAFE_SINCE = (2, 6)


def _release_of(version: str) -> tuple[int, int]:
    """``(major, minor)`` of a version string such as ``"2.13.0+cpu"``."""
    major, minor = version.split("+", 1)[0].split(".")[:2]
    return int(major), int(minor)


def _load_storage(data: object) -> torch.storage.TypedStorage:
    """Replacement for ``torch.storage._load_from_bytes``.

    A pickled storage is the bytes of a nested legacy ``torch.save``.  The
    framework's hook reads them with an unrestricted ``torch.load``, which
    would run whatever the nested pickle names; the weights-only loader
    admits tensor data and plain containers only, and is refused outright
    on a framework release old enough to be bypassed.
    """
    if _release_of(torch.__version__) < _WEIGHTS_ONLY_SAFE_SINCE:
        raise RuntimeError(
            f"convert(tvm): framework {torch.__version__} predates 2.6, whose "
            f"weights-only loader closes CVE-2025-32434; upgrade it to read a "
            f"checkpoint's storages"
        )
    if not isinstance(data, bytes):
        raise pickle.UnpicklingError(
            f"convert(tvm): a storage record holds {type(data).__name__}, not bytes"
        )
    storage = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(storage, torch.storage.TypedStorage):
        raise pickle.UnpicklingError(
            f"convert(tvm): a storage record decodes to "
            f"{type(storage).__name__}, not a storage"
        )
    return storage


# Resolved for real: they assemble tensors from their arguments and call
# nothing else.
_REAL: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_tensor_v2"),
    }
)

_SPECIAL: dict[tuple[str, str], Callable[..., object]] = {
    ("torch_utils.persistence", "_reconstruct_persistent_obj"): _reconstruct_persistent,
    ("torch._dynamo.eval_frame", "OptimizedModule"): _unwrap_compiled,
    ("torch.storage", "_load_from_bytes"): _load_storage,
}

_INERT: frozenset[tuple[str, str]] = frozenset(
    {
        # The network tree: the framework's layers, timm's patch embedding
        # and MLP, and the one undecorated class of the authors' module.
        ("torch.nn.modules.activation", "GELU"),
        ("torch.nn.modules.activation", "SiLU"),
        ("torch.nn.modules.container", "ModuleList"),
        ("torch.nn.modules.container", "Sequential"),
        ("torch.nn.modules.conv", "Conv2d"),
        ("torch.nn.modules.dropout", "Dropout"),
        ("torch.nn.modules.linear", "Identity"),
        ("torch.nn.modules.linear", "Linear"),
        ("torch.nn.modules.normalization", "RMSNorm"),
        ("torch.nn.modules.sparse", "Embedding"),
        ("timm.layers.format", "Format"),
        ("timm.layers.mlp", "Mlp"),
        ("timm.layers.patch_embed", "PatchEmbed"),
        ("training.dit", "Attention"),
        # The wrapper's autocast dtype.
        ("torch", "bfloat16"),
        # The training configuration the recorded arguments carry.
        ("builtins", "dict"),
        ("builtins", "int"),
        ("builtins", "list"),
        ("collections", "defaultdict"),
        ("omegaconf.base", "ContainerMetadata"),
        ("omegaconf.base", "Metadata"),
        ("omegaconf.dictconfig", "DictConfig"),
        ("omegaconf.listconfig", "ListConfig"),
        ("omegaconf.nodes", "AnyNode"),
        ("typing", "Any"),
        # The compile context outside ``torch._dynamo``.
        ("builtins", "getattr"),
        ("contextlib", "nullcontext"),
        ("torch", "_TorchCompileInductorWrapper"),
    }
)

_INERT_PACKAGES = ("torch._dynamo.",)


class _ParameterUnpickler(pickle.Unpickler):
    """Resolve what a parameter tree needs; refuse everything else."""

    @override
    def find_class(self, module: str, name: str, /) -> object:
        special = _SPECIAL.get((module, name))
        if special is not None:
            return special
        if (module, name) in _REAL:
            return super().find_class(module, name)
        if (module, name) in _INERT or module.startswith(_INERT_PACKAGES):
            return _Record
        raise pickle.UnpicklingError(
            f"convert(tvm): the checkpoint names {module}.{name}, which the "
            f"converter does not admit — inspect what it is before adding "
            f"it to a table in tools/convert_weights/tvm.py"
        )


def _attrs(obj: object) -> dict[str, object]:
    """An unpickled object's attribute dictionary (empty for a non-record)."""
    return vars(obj) if isinstance(obj, _Record) else {}


def _children(attrs: dict[str, object]) -> dict[str, object]:
    """The ``_modules`` dictionary of a module record's attributes."""
    modules = attrs.get("_modules")
    return modules if isinstance(modules, dict) else {}


def _flatten(obj: object, prefix: str, out: dict[str, npt.NDArray[np.float32]]) -> None:
    """Collect every parameter and buffer under ``obj`` by dotted name."""
    attrs = _attrs(obj)
    for group in ("_parameters", "_buffers"):
        tensors = attrs.get(group)
        if not isinstance(tensors, dict):
            continue
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"convert(tvm): {prefix}{name} is a "
                    f"{type(tensor).__name__}, not a tensor"
                )
            out[prefix + name] = tensor.detach().to(torch.float32).cpu().numpy()
    for name, child in _children(attrs).items():
        _flatten(child, f"{prefix}{name}.", out)


def _module_attribute(
    attrs: dict[str, object], attr: str, prefix: str = ""
) -> dict[str, object]:
    """``attr`` of every module in the tree that records one, by dotted path."""
    found: dict[str, object] = {}
    if attr in attrs:
        found[prefix.rstrip(".") or "<network>"] = attrs[attr]
    for name, child in _children(attrs).items():
        found.update(_module_attribute(_attrs(child), attr, f"{prefix}{name}."))
    return found


def _load_ema(
    path: str,
) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, object], dict[str, object]]:
    """The evaluation EMA's tensors, and the wrapper's and network's attributes.

    Raises
    ------
    pickle.UnpicklingError
        If the file names a global outside the converter's tables.
    RuntimeError
        If the file holds no evaluation EMA, or the EMA holds no tensors.
    """
    with open(path, "rb") as handle:
        data = _ParameterUnpickler(handle).load()
    # The authors' sampler reads ``data["ema"]``; their training loop's
    # snapshots store the same evaluation EMA under ``"net"``.  Either is
    # the 0.9999 average.
    key = next(
        (k for k in ("ema", "net") if isinstance(data, dict) and k in data), None
    )
    if key is None:
        raise RuntimeError(
            f"convert(tvm): {path} holds neither an 'ema' nor a 'net' entry"
        )
    wrapper = data[key]
    state: dict[str, npt.NDArray[np.float32]] = {}
    _flatten(wrapper, "", state)
    if not state:
        raise RuntimeError(
            f"convert(tvm): {path} unpickled to an object holding no tensors "
            f"— its {key!r} entry is not a module tree this converter reads"
        )
    network = _children(_attrs(wrapper)).get("model")
    return state, _attrs(wrapper), _attrs(network)


# ── the recipe ───────────────────────────────────────────────────────────────


class TVMArch(Architecture):
    """Conversion recipe for one released TVM checkpoint."""

    def __init__(self, model_name: str, tag: str) -> None:
        """Bind a factory name to one of its releases.

        Parameters
        ----------
        model_name : str
            Lucid factory the weights load into.
        tag : str
            A key of ``_RELEASES`` whose factory is ``model_name``.

        Raises
        ------
        ValueError
            If ``tag`` names no release of ``model_name``.
        """
        mine = sorted(k for k, r in _RELEASES.items() if r.factory == model_name)
        if tag not in mine:
            raise ValueError(f"no {model_name} release tagged {tag!r}; known: {mine}")
        self.model_name = model_name
        self.tag = tag
        self.release = _RELEASES[tag]

    def _config(self) -> TVMConfig:
        base = AutoConfig.from_pretrained(self.model_name)
        if not isinstance(base, TVMConfig):
            raise TypeError(
                f"{self.model_name} is registered with a {type(base).__name__}, "
                f"not a TVMConfig"
            )
        return apply_overrides(base, self.release.overrides)

    @override
    def source_state_dict(self) -> dict[str, npt.NDArray[np.float32]]:
        """Download the pickle, load its EMA, and check what keys cannot.

        Returns
        -------
        dict
            Upstream names to arrays, ``pos_embed`` already verified.

        Raises
        ------
        RuntimeError
            If the checkpoint holds no tensors, its scaled
            parameterisation, guidance embedder, attention heads or
            modulation norm disagree with the target config, or its
            positional table differs from the one Lucid builds.
        """
        path = hf_hub_download(_REPO, self.release.filename)
        state, wrapper, network = _load_ema(path)
        config = self._config()
        self._check_training_flags(wrapper, network, config)
        self._check_layers(network, "num_heads", config.num_heads)
        self._check_layers(network, "do_mod_norm", config.modulation_norm)

        stored = state.get("model.pos_embed")
        if stored is not None:
            side = patch_grid_side(config.sample_size) // config.patch_size
            built = sincos_position_embedding(config.hidden_size, side).numpy()
            if not np.allclose(stored, built, atol=1e-5):
                raise RuntimeError(
                    "convert(tvm): the checkpoint's positional table differs "
                    "from Lucid's sincos_position_embedding — the axis order "
                    "or frequency ladder has drifted"
                )
        return state

    def _check_training_flags(
        self, wrapper: dict[str, object], network: dict[str, object], config: TVMConfig
    ) -> None:
        scaled = wrapper.get("do_scale_param")
        if not isinstance(scaled, bool):
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} records no "
                f"do_scale_param on its wrapper"
            )
        if scaled != config.scaled_parameterization:
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} was trained with "
                f"do_scale_param={scaled}, but the target config has "
                f"scaled_parameterization={config.scaled_parameterization}"
            )
        if not network.get("learn_guidance", True):
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} has no guidance "
                f"embedder; Lucid's TVM always conditions on 1/w"
            )

    def _check_layers(
        self, network: dict[str, object], attr: str, expected: object
    ) -> None:
        """Every layer that records ``attr`` ran with the config's value."""
        found = _module_attribute(network, attr)
        if not found:
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} records no {attr} on "
                f"any layer of its network"
            )
        wrong = {path: value for path, value in found.items() if value != expected}
        if wrong:
            sample = dict(sorted(wrong.items())[:3])
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} ran {len(wrong)} "
                f"layer(s) with a different {attr} than the target config's "
                f"{expected!r}: {sample}"
            )

    @override
    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return TVMModel(self._config())

    @override
    def map_key(self, src_key: str) -> str | None:
        """Map one upstream key to its Lucid name.

        Parameters
        ----------
        src_key : str
            Upstream name, ``model.``-prefixed as the wrapper stores it.

        Returns
        -------
        str or None
            The Lucid name, or ``None`` for the positional table, which
            :meth:`source_state_dict` has checked against the one Lucid
            rebuilds.

        Raises
        ------
        UnknownSourceKey
            For any other key: the release holds the network and nothing
            else, so a key outside this map is a layout it was not
            written for.
        """
        if src_key == "model.pos_embed":
            return None
        key = src_key.removeprefix("model.")
        leaf = key.rsplit(".", 1)[-1]
        simple = {
            "x_embedder.proj.": "patch_embed.",
            "t_embedder.mlp.0.": "time_embed.fc1.",
            "t_embedder.mlp.2.": "time_embed.fc2.",
            "s_embedder.mlp.0.": "interval_embed.fc1.",
            "s_embedder.mlp.2.": "interval_embed.fc2.",
            "guidance_embedder.mlp.0.": "guidance_embed.fc1.",
            "guidance_embedder.mlp.2.": "guidance_embed.fc2.",
            "y_embedder.embedding_table.": "label_embed.",
            "final_layer.linear.": "final.proj.",
            "final_layer.adaLN_modulation.1.": "final.ada_ln.1.",
        }
        inner = {
            "attn.qkv.": "attn.qkv.",
            "attn.q_norm.": "attn.q_norm.",
            "attn.k_norm.": "attn.k_norm.",
            "attn.proj.": "attn.proj.",
            "mlp.fc1.": "mlp.fc1.",
            "mlp.fc2.": "mlp.fc2.",
            "adaLN_modulation.1.": "ada_ln.1.",
        }
        if key != src_key:
            for upstream, lucid_prefix in simple.items():
                if key == upstream + leaf:
                    return lucid_prefix + leaf
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[0] == "blocks" and parts[1].isdigit():
                for upstream, lucid_prefix in inner.items():
                    if parts[2] == upstream + leaf:
                        return f"blocks.{parts[1]}.{lucid_prefix}{leaf}"
        raise UnknownSourceKey(src_key)

    @override
    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        pixels = 256 if config.sample_size == 32 else 512
        return ConversionSpec(
            model_name=self.model_name,
            architecture="tvm",
            repo_id="lucid-dl/tvm",
            tag=self.tag,
            task="image-generation",
            model_type="tvm",
            source=f"https://huggingface.co/{_REPO}/blob/main/{self.release.filename}",
            license="cc-by-nc-sa-4.0",
            num_classes=config.num_classes,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": pixels,
                "rescale": 1 / 127.5,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            citation=_CITATION,
            title=f"TVM-XL/2 ({pixels}x{pixels}, w={self.release.guidance})",
            paper_url="https://arxiv.org/abs/2511.19797",
            categories=["generative", "flow-map", "class-conditional"],
            datasets=["imagenet-1k"],
            meta={
                "num_params": 678_068_496,
                "metrics": {"imagenet-1k": dict(self.release.fid)},
                "guidance_scale": self.release.guidance,
                "latent_size": config.sample_size,
                "vae": "stabilityai/sd-vae-ft-ema",
                "latent_whitening": {
                    "mean": list(config.latent_mean),
                    "std": list(config.latent_std),
                },
                "sampler": "x_s = x_t + (s - t) * model(x_t, t, s), t on linspace(1, 0, n+1)",
            },
        )


@register_arch("tvm_imagenet256")
def _tvm_256(tag: str) -> TVMArch:
    """Build the recipe for a 256-pixel release."""
    return TVMArch("tvm_imagenet256", tag)


@register_arch("tvm_imagenet512")
def _tvm_512(tag: str) -> TVMArch:
    """Build the recipe for a 512-pixel release."""
    return TVMArch("tvm_imagenet512", tag)
