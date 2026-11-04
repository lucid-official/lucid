"""Pretrained-weight declarations for the DIAMOND family.

One agent per game.  The Atari 100k benchmark trains a separate agent on
each of its 26 games from 100k environment steps, so there is no single
"DIAMOND checkpoint" — there are twenty-six, and they differ in more
than weights: each game exposes its own minimal action set, so the
policy head's width changes with the tag.  The factory reads that from
the entry rather than making the caller remember it.

The weights are converted from ``eloialonso/diamond`` and re-hosted under
the ``lucid-dl`` org.

⚠️ These are **agents**, not just world models.  Each file carries the
denoiser, the reward/termination model and the actor-critic together,
which is why loading one gives something that can act as well as
imagine.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = ["DIAMONDWeights"]

# Frames reach the world model in [-1, 1]; there is no dataset mean to
# subtract, because what consumes them is a diffusion model rather than
# a classifier.
_PRESET = ImageClassification(
    crop_size=64, resize_size=64, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)

# CS:GO diffuses at 30x56 and upsamples to 150x280; the transform names
# the resolution the world model works at, not the one it ends on.
_PRESET_CSGO = ImageClassification(
    crop_size=30, resize_size=30, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)


@register_weights("diamond")
@register_weights("diamond_world_model")
# Without this, list_pretrained("diamond_csgo") answers [] -- "no
# weights", which is wrong: the CSGO checkpoint is that factory's and
# loads there.  The registry maps a factory to an enum, not to a
# subset of its tags, so each factory refuses the other's tags by name
# (see _tag / _csgo_tag) rather than by not listing them.
@register_weights("diamond_csgo")
class DIAMONDWeights(WeightsEnum):
    """Pretrained agents for :func:`lucid.models.diamond`.

    One member per Atari 100k game.  ``num_classes`` is the game's action
    count, not a label set — a world model has no classes, and this is
    the number the policy head and both action embeddings are built to.

    Notes
    -----
    Reference: Alonso et al., NeurIPS 2024 (arXiv:2405.12399), Table 1
    for the per-game scores these agents reach.

    ``DEFAULT`` is *Breakout*, which is the game the paper's own
    analysis figures use.

    ``CSGO`` is the odd one out: a world model with no agent, at a
    different resolution and nearly thirty times the size, from the
    paper's Section 6.  It loads into :func:`lucid.models.diamond_csgo`
    rather than the Atari factories.

    Examples
    --------
    >>> from lucid.models.generative.diamond._weights import DIAMONDWeights
    >>> len(DIAMONDWeights.__members__)
    28
    >>> "BREAKOUT" in DIAMONDWeights.__members__
    True

    Twenty-seven Atari agents and one CS:GO world model share this enum,
    and the two are not interchangeable: ``diamond`` and
    ``diamond_world_model`` refuse the CS:GO tag, and ``diamond_csgo``
    refuses the Atari ones.

    >>> DIAMONDWeights.BREAKOUT.num_classes   # Breakout has four actions
    4
    """

    ALIEN = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Alien/model.safetensors",
        sha256="2faa3e875f37df5c8a8c6138393d5a3f68b67453d852e31eb91c42a3407e6715",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Alien",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Alien",
        },
    )
    AMIDAR = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Amidar/model.safetensors",
        sha256="82069fb40916b548502bc729d224283d990eb7dfbce39dfe393f4f16cba177ec",
        num_classes=10,
        transforms=_PRESET,
        meta={
            "tag": "Amidar",
            "num_actions": 10,
            "num_params": 13_536_584,
            "file_size_mb": 51.7,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Amidar",
        },
    )
    ASSAULT = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Assault/model.safetensors",
        sha256="3ce2170373d79571da2fe2a518cd7f79b499a0002c405dfd5d3482d9a77760c1",
        num_classes=7,
        transforms=_PRESET,
        meta={
            "tag": "Assault",
            "num_actions": 7,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Assault",
        },
    )
    ASTERIX = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Asterix/model.safetensors",
        sha256="9d4fb8f74d0af3173611b0da0e38bd20200a58554feca227fffcb3e3bbe6ece2",
        num_classes=9,
        transforms=_PRESET,
        meta={
            "tag": "Asterix",
            "num_actions": 9,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Asterix",
        },
    )
    BANKHEIST = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/BankHeist/model.safetensors",
        sha256="0410d595ef2cde9f5b9defd339e45237b055db4aaca368a35cf701f928ad81df",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "BankHeist",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "BankHeist",
        },
    )
    BATTLEZONE = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/BattleZone/model.safetensors",
        sha256="67b68ed57f130995a6b2b4ae6b872505dd920b9a703a1709488f0cfe86350b55",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "BattleZone",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "BattleZone",
        },
    )
    BOXING = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Boxing/model.safetensors",
        sha256="94979c2f4d10bd6ef41340d9b49ec43b80ebbac93bb8b9ac1722578495086c8d",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Boxing",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Boxing",
        },
    )
    BREAKOUT = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Breakout/model.safetensors",
        sha256="aea58ec7ff53cd9aceb4f1758fbf8280952ac15f9c200c6494bdde8c2256e141",
        num_classes=4,
        transforms=_PRESET,
        meta={
            "tag": "Breakout",
            "num_actions": 4,
            "num_params": 13_536_584,
            "file_size_mb": 51.68,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Breakout",
        },
    )
    CHOPPERCOMMAND = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/ChopperCommand/model.safetensors",
        sha256="79e6e4fd93aa0b4f0e2f7e6fc7e5dbc96c9fc3076b0a005123a37bda345041cd",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "ChopperCommand",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "ChopperCommand",
        },
    )
    CRAZYCLIMBER = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/CrazyClimber/model.safetensors",
        sha256="0caa904e7421e76b945c520e24d1ea10a4fdb33ff202a09764e9cad43add1243",
        num_classes=9,
        transforms=_PRESET,
        meta={
            "tag": "CrazyClimber",
            "num_actions": 9,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "CrazyClimber",
        },
    )
    DEMONATTACK = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/DemonAttack/model.safetensors",
        sha256="9aba493d0206ad27f817b7b52741eec569c23c6475300b4719ec269cc6e3531b",
        num_classes=6,
        transforms=_PRESET,
        meta={
            "tag": "DemonAttack",
            "num_actions": 6,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "DemonAttack",
        },
    )
    FREEWAY = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Freeway/model.safetensors",
        sha256="e228061d2364ba4526fbfa787eac3def20da28e5b6d05e16ee96ff41881dc8af",
        num_classes=3,
        transforms=_PRESET,
        meta={
            "tag": "Freeway",
            "num_actions": 3,
            "num_params": 13_536_584,
            "file_size_mb": 51.68,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Freeway",
        },
    )
    FROSTBITE = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Frostbite/model.safetensors",
        sha256="d54c4ba81a0db8f623611c4c1516034b67531cf84b38744e002fb9abcd30915a",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Frostbite",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Frostbite",
        },
    )
    GOPHER = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Gopher/model.safetensors",
        sha256="59de7a600d548852f97068c4083c2bd1624169e2ed7119451a61e578b9720d29",
        num_classes=8,
        transforms=_PRESET,
        meta={
            "tag": "Gopher",
            "num_actions": 8,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Gopher",
        },
    )
    HERO = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Hero/model.safetensors",
        sha256="53af3965df5c7d8dc984a56a94f2ebd21301b5c98eee554404e6c71df93f91e8",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Hero",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Hero",
        },
    )
    JAMESBOND = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Jamesbond/model.safetensors",
        sha256="3cdcc78133c9d96bab9ecf9b63db341f848a917809ff7a26ae1830489840128b",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Jamesbond",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Jamesbond",
        },
    )
    KANGAROO = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Kangaroo/model.safetensors",
        sha256="a68f90ca7c1c4a55692f9076872246264014624fe00807b91cbd9c70653984e2",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Kangaroo",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Kangaroo",
        },
    )
    KRULL = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Krull/model.safetensors",
        sha256="424db1f8766cda7e6b133a3840f628824f62f6412e6ffb6089f41bc63b2a6d73",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Krull",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Krull",
        },
    )
    KUNGFUMASTER = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/KungFuMaster/model.safetensors",
        sha256="2434e5b6b6a514b8d0c8d7ad1313e28e0832a8fa9da39447bda7f95b4a8dfcae",
        num_classes=14,
        transforms=_PRESET,
        meta={
            "tag": "KungFuMaster",
            "num_actions": 14,
            "num_params": 13_536_584,
            "file_size_mb": 51.71,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "KungFuMaster",
        },
    )
    MSPACMAN = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/MsPacman/model.safetensors",
        sha256="53bc103976157262ab00a3fd79622ad5c1b53be01d5ac04b2442d7bd9e0cf404",
        num_classes=9,
        transforms=_PRESET,
        meta={
            "tag": "MsPacman",
            "num_actions": 9,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "MsPacman",
        },
    )
    PONG = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Pong/model.safetensors",
        sha256="6be517a6e8a7ef277e18c1153155d9139f6085c2e7942ae0975f5bfbcd921245",
        num_classes=6,
        transforms=_PRESET,
        meta={
            "tag": "Pong",
            "num_actions": 6,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Pong",
        },
    )
    PRIVATEEYE = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/PrivateEye/model.safetensors",
        sha256="aa23a0f855109c89579781c31d44a62b72ab904105f43371eb8f03771cbf4e10",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "PrivateEye",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "PrivateEye",
        },
    )
    QBERT = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Qbert/model.safetensors",
        sha256="f443caee739b6b608bcd2dd13c4a1545b1ea3363fa5e964793bd42c7ede6fe0a",
        num_classes=6,
        transforms=_PRESET,
        meta={
            "tag": "Qbert",
            "num_actions": 6,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Qbert",
        },
    )
    ROADRUNNER = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/RoadRunner/model.safetensors",
        sha256="2ba663897128db2f244b81a4e346f73f5975fc9a3731273a6ffa118760cbfb95",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "RoadRunner",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "RoadRunner",
        },
    )
    SEAQUEST = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/Seaquest/model.safetensors",
        sha256="ca9110ab47b3bc54128717034e44b6160116f70eace50eaed9b113c5b2698a8e",
        num_classes=18,
        transforms=_PRESET,
        meta={
            "tag": "Seaquest",
            "num_actions": 18,
            "num_params": 13_536_584,
            "file_size_mb": 51.72,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "Seaquest",
        },
    )
    UPNDOWN = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/UpNDown/model.safetensors",
        sha256="4dd98e8eede190ce6f75a4f74f2774f8fecfbc6cb596c3a74f3b149af904d450",
        num_classes=6,
        transforms=_PRESET,
        meta={
            "tag": "UpNDown",
            "num_actions": 6,
            "num_params": 13_536_584,
            "file_size_mb": 51.69,
            "source": "eloialonso/diamond (atari_100k)",
            "license": "mit",
            "game": "UpNDown",
        },
    )
    CSGO = WeightEntry(
        url=f"{HUB_BASE}/diamond/resolve/main/CSGO/model.safetensors",
        sha256="20e8a9cc87da27e6af4d47694c318b411dd014617c217705dea3b228e60828b5",
        num_classes=51,
        transforms=_PRESET_CSGO,
        meta={
            "tag": "CSGO",
            "num_actions": 51,
            "num_params": 381_642_502,
            "file_size_mb": 1455.92,
            "source": "eloialonso/diamond (csgo)",
            "license": "mit",
            "game": "Counter-Strike: Global Offensive",
            "frame_shape": [30, 56],
            "upsampled_shape": [150, 280],
        },
    )
    DEFAULT = BREAKOUT
