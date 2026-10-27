"""DIAMOND weight converter — the released agents → Lucid.

Twenty-six checkpoints, one per Atari 100k game, each a flat state dict
holding all three networks.  The mapping is almost a rename, because the
Lucid model was rebuilt against these files: the module tree matches
stage for stage and block for block.

Three renames and one drop are all that is left.

``inner_model`` disappears — the released denoiser wraps its U-Net in an
extra module that only exists to hold the EDM preconditioners, and those
live on :class:`~lucid.models.generative.diamond.DIAMONDModel` here.
``act_emb`` becomes ``action_embed`` and ``lstm`` becomes ``cell``,
matching Lucid's names for an embedding and an LSTM cell.  The released
``act_emb`` is a ``Sequential`` whose only parameter-bearing member is
the embedding, so its ``.0.`` disappears with it.

Each game has its own action count, which is why the tag is the game
name: ``Breakout`` has 4 actions and ``Alien`` 18, and a checkpoint will
not load into a model built for the wrong one.
"""

import dataclasses
from typing import Any, cast

import torch
from huggingface_hub import hf_hub_download

from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    "@inproceedings{alonso2024diffusion,\n"
    "  title={Diffusion for World Modeling: Visual Details Matter in "
    "Atari},\n"
    "  author={Alonso, Eloi and Jelley, Adam and Micheli, Vincent and "
    "Kanervisto, Anssi and Storkey, Amos and Pearce, Tim and Fleuret, "
    "Fran{\\c{c}}ois},\n"
    "  booktitle={Advances in Neural Information Processing Systems "
    "(NeurIPS)},\n"
    "  year={2024}\n"
    "}"
)

_REPO = "eloialonso/diamond"

# The CS:GO world model, from the same release.  It is a different model
# rather than a bigger one: no reward head, no actor-critic, non-square
# frames, per-resolution attention, and a second diffusion model that
# magnifies 30x56 to 150x280.  Its values are the released
# ``config/agent/csgo.yaml`` verbatim.
_CSGO = "CSGO"
_CSGO_CONFIG: dict[str, object] = {
    "sample_size": (30, 56),
    "unet_channels": (128, 256, 512, 1024),
    "unet_layers": (2, 2, 2, 2),
    "attn_depths": (0, 0, 1, 1),
    "cond_dim": 2048,
    "num_actions": 51,
    "with_agent": False,
    "noise_previous_obs": True,
    "sigma_offset_noise": 0.1,
    "upsampler_channels": (64, 64, 128, 256),
    "upsampler_layers": (2, 2, 2, 2),
    "upsampler_attn_depths": (0, 0, 0, 1),
    "upsampling_factor": 5,
}

# The 26 games of Atari 100k, with the action count each checkpoint was
# trained at.  Atari exposes a per-game minimal action set, so this is a
# property of the environment rather than a choice — and it is read back
# from the checkpoint at conversion time rather than trusted blindly.
_GAMES: tuple[str, ...] = (
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
)


def _num_actions(state: dict[str, Any]) -> int:
    """Read the action count off the checkpoint itself.

    Parameters
    ----------
    state : dict
        The released state dict.

    Returns
    -------
    int
        Rows of the policy's output layer.

    Raises
    ------
    RuntimeError
        If the key that carries it is missing, which would mean the
        layout changed and every other assumption here needs re-checking.
    """
    key = "actor_critic.actor_linear.weight"
    if key not in state:
        raise RuntimeError(
            f"convert(diamond): {key!r} is missing, so the action count "
            f"cannot be read and the released layout has changed"
        )
    return int(state[key].shape[0])


class DIAMONDArch(Architecture):
    """Conversion recipe for one released DIAMOND agent."""

    def __init__(self, model_name: str, tag: str = "Breakout") -> None:
        """Bind a factory name to one game's checkpoint.

        Parameters
        ----------
        model_name : str
            Lucid factory the weights load into.
        tag : str, default="Breakout"
            Game name, which is also the checkpoint's file stem.

        Raises
        ------
        ValueError
            If ``tag`` names no released game.
        """
        if tag not in _GAMES and tag != _CSGO:
            raise ValueError(f"no released agent for {tag!r}; known: {[*_GAMES, _CSGO]}")
        self.model_name = model_name
        self.tag = tag
        self._actions = 0

    def _config(self) -> DIAMONDConfig:
        if self.tag == _CSGO:
            return DIAMONDConfig(**cast(dict[str, Any], _CSGO_CONFIG))
        return DIAMONDConfig(num_actions=self._actions or 18)

    def source_state_dict(self) -> dict[str, object]:
        """Download one game's agent.

        Returns
        -------
        dict
            The released names to arrays, with the action count cached so
            :meth:`target_model` can build the right policy head.
        """
        if self.tag == _CSGO:
            path = hf_hub_download(_REPO, "csgo/model/csgo.pt")
            raw = torch.load(path, map_location="cpu", weights_only=True)
            self._actions = _CSGO_CONFIG["num_actions"]  # type: ignore[assignment]
            return {k: v.detach().cpu().numpy() for k, v in raw.items()}
        path = hf_hub_download(_REPO, f"atari_100k/models/{self.tag}.pt")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self._actions = _num_actions(raw)
        return {k: v.detach().cpu().numpy() for k, v in raw.items()}

    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return DIAMONDModel(self._config())

    def map_key(self, src_key: str) -> str | None:
        """Map one released key to its Lucid name.

        Parameters
        ----------
        src_key : str
            The released name.

        Returns
        -------
        str or None
            The Lucid name; never ``None``, because every released
            parameter has a home here and a silent drop would be a bug
            rather than a design.
        """
        key = src_key.replace("denoiser.inner_model.", "denoiser.")
        key = key.replace("upsampler.inner_model.", "upsampler.")
        key = key.replace("upsampler.act_emb.0.", "upsampler.action_embed.")
        key = key.replace("rew_end_model.", "reward_end.")
        key = key.replace("reward_end.act_emb.", "reward_end.action_embed.")
        key = key.replace("denoiser.act_emb.0.", "denoiser.action_embed.")
        key = key.replace("reward_end.lstm.", "reward_end.cell.")
        key = key.replace("actor_critic.lstm.", "actor_critic.cell.")
        key = key.replace("actor_critic.encoder.encoder.", "actor_critic.encoder.")
        # The released LSTM names carry a layer suffix; Lucid's cell is a
        # single step and does not.
        key = key.replace("_l0", "")
        return key

    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        return ConversionSpec(
            model_name=self.model_name,
            architecture="diamond",
            repo_id="lucid-dl/diamond",
            tag=self.tag,
            task="world-modeling",
            model_type="diamond",
            source=f"https://huggingface.co/{_REPO}",
            license="mit",
            num_classes=config.num_actions,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": 64,
                "rescale": 1 / 127.5,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            citation=_CITATION,
            title=f"DIAMOND ({self.tag})",
            paper_url="https://arxiv.org/abs/2405.12399",
            categories=["generative", "diffusion", "world-model"],
            datasets=["atari-100k"],
            meta={
                "num_params": 0,
                "game": self.tag,
                "num_actions": config.num_actions,
                "denoising_steps": config.denoise_steps,
            },
        )


@register_arch("diamond")
def _diamond(tag: str) -> DIAMONDArch:
    """Build the recipe the CLI dispatches to, for one game."""
    return DIAMONDArch("diamond", tag)


@register_arch("diamond_world_model")
def _diamond_world_model(tag: str) -> DIAMONDArch:
    """The same weights, for the imagination-capable factory."""
    return DIAMONDArch("diamond_world_model", tag)


def games() -> tuple[str, ...]:
    """The 26 released games, for a caller converting all of them."""
    return _GAMES
