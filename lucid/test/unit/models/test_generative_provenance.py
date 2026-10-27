"""Where every generative-family default came from, checked exhaustively.

This file exists because of how the defects in this family were actually
found. Seven wrong defaults were fixed across PlaNet, Dreamer, DreamerV2
and DreamerV3, and **not one of them was a field that had been checked
and got wrong.** Every one was a field nobody had looked at:

* ``horizon`` was 16 where the paper's Table 4 says 15
* ``num_bins`` was 41 where the reference uses 255
* ``reward_layers`` and ``pcont_layers`` were 3 where the reference's
  ``rewhead`` and ``conhead`` have one
* Dreamer's ``reward_layers`` was 2, from a *paraphrase* of the paper
  that narrowed "all other functions" to "the action and value models"
* DreamerV2's Control Suite config was missing ``free_nats``

Checking fields opportunistically cannot find that class of defect,
because the thing that goes wrong is the choice of what to check. So the
check here is inverted: the table below must name **every field of every
config**, and a field with no recorded source fails the suite. Adding a
config field without writing down where its value came from is now a
build break rather than a thing someone might notice later.

Three kinds of source, and the prefix is part of the contract:

``paper:``
    Stated in the family's own paper. Quote it, do not paraphrase it —
    the Dreamer defect above lived entirely in a paraphrase, and the
    paraphrase was in a test whose name was ``test_defaults_match_paper``.
``code:``
    From the released implementation, because the paper is silent. Name
    the file and key.
``lucid:``
    Neither — a convention of this zoo, or a deliberate divergence. Say
    which, and why.

Where paper and code disagree the citation says so outright, so that a
reader can see the disagreement without going to look for it.
"""

from dataclasses import fields
from typing import Any

import pytest

from lucid.models.generative.dreamer import DreamerConfig
from lucid.models.generative.dreamer_v2 import DreamerV2Config
from lucid.models.generative.dreamer_v3 import DreamerV3Config
from lucid.models.generative.genie import GenieConfig
from lucid.models.generative.planet import PlaNetConfig
from lucid.models.generative.score_sde import ScoreSDEConfig

# A Lucid-wide convention shared by every generative config; recorded once
# and reused, because repeating it four times would invite it drifting.
_SAMPLE = "lucid: 64x64 frames, the resolution all four papers train on"
_RGB = "lucid: RGB, the zoo's channel convention for pixel models"
_ACTION_DIM = "lucid: placeholder — the environment sets this, not the paper"
_MEAN_ONLY = "lucid: zoo convention — draw the latent unless asked not to"
# Not a pure convention: each paper names both settings and says which
# benchmarks use which.  The *default* is the continuous one; the
# discrete factories override it.
_ACTION_SPACE = (
    "paper: the continuous setting is the default; each family's discrete "
    "factory overrides it, citing that paper's own discrete-control section"
)


PLANET: dict[str, tuple[Any, str]] = {
    "sample_size": (64, _SAMPLE),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "action_dim": (1, _ACTION_DIM),
    "mean_only": (False, _MEAN_ONLY),
    "act_fn": (
        "relu",
        'paper: Appendix A, "two fully connected layers of size 200 with ' 'ReLU activations"',
    ),
    "stoch_size": (
        30,
        'paper: Appendix A, "Distributions in latent space are '
        '30-dimensional diagonal Gaussians"',
    ),
    "deter_size": (
        200,
        'paper: Appendix A, "a GRU with 200 units as deterministic path in ' 'the dynamics model"',
    ),
    "hidden_size": (
        200,
        'paper: Appendix A, "two fully connected layers of size 200"',
    ),
    "cnn_depth": (
        32,
        'paper: Appendix A, "the convolutional and deconvolutional networks '
        'from Ha & Schmidhuber (2018)" — depth 32',
    ),
    "min_std": (
        0.1,
        "code: the RSSM lineage floors the latent scale at 0.1 — "
        "dreamer models.py, std = softplus(std) + 0.1.  PlaNet's paper "
        "does not state it",
    ),
    "free_nats": (
        3.0,
        'paper: Appendix A, "grant the model 3 free nats by clipping the '
        'divergence loss below this value"',
    ),
    "kl_weight": (
        1.0,
        'paper: Appendix A, "We do not scale the KL divergence terms '
        'relatively to the reconstruction terms"',
    ),
    "reward_hidden": (200, 'paper: Appendix A, "all other functions" — size 200'),
    "reward_layers": (
        2,
        'paper: Appendix A, "implement all other functions as **two** fully '
        "connected layers of size 200\".  Note Dreamer's appendix says "
        "three; the two families genuinely differ here",
    ),
    "reward_loss_scale": (1.0, "paper: Appendix A, the reward term is unscaled"),
    "overshoot_distance": (
        None,
        'paper: Appendix A, "In a previous version of the agent, we used '
        'latent overshooting ... but we found this to not be necessary" — '
        "off by default, implemented because the paper derives it",
    ),
    "overshoot_weight": (
        1.0,
        "lucid: inert while overshoot_distance is None; the scale only has "
        "meaning once overshooting is switched on",
    ),
    "overshoot_reward_weight": (1.0, "lucid: inert, as overshoot_weight"),
}


DREAMER: dict[str, tuple[Any, str]] = {
    "sample_size": (64, _SAMPLE),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "action_dim": (1, _ACTION_DIM),
    "mean_only": (False, _MEAN_ONLY),
    "act_fn": (
        "elu",
        'paper: Appendix A, "three dense layers of size 300 with ELU activations"',
    ),
    "cnn_act": (
        "relu",
        "code: dreamer.py cnn_act = 'relu', which the released config keeps "
        "separate from dense_act = 'elu'; the paper states the dense "
        "activation and is silent on the convolutional one",
    ),
    "stoch_size": (
        30,
        'paper: Appendix A, "Distributions in latent space are '
        '30-dimensional diagonal Gaussians"',
    ),
    "deter_size": (
        200,
        'paper: Appendix A, "the RSSM of Hafner et al. (2018)" — PlaNet\'s '
        "200-unit GRU; code: dreamer.py deter_size = 200",
    ),
    "hidden_size": (
        200,
        "code: dreamer.py builds RSSM(stoch_size, deter_size, deter_size), so "
        "the hidden width is the deterministic one",
    ),
    "cnn_depth": (
        32,
        'paper: Appendix A, "the convolutional encoder and decoder networks '
        'from Ha and Schmidhuber (2018)"; code: cnn_depth = 32',
    ),
    "min_std": (
        0.1,
        "code: models.py, std = tf.nn.softplus(std) + 0.1 in both the "
        "prior and the posterior head",
    ),
    "free_nats": (
        3.0,
        'paper: Appendix A, "clip them below 3 free nats as in PlaNet"',
    ),
    "kl_weight": (
        1.0,
        'paper: Appendix A, "We do not scale the KL regularizers (beta = 1)"',
    ),
    "horizon": (15, 'paper: Appendix A, "The imagination horizon is H = 15"'),
    "discount": (
        0.99,
        'paper: Appendix A, "We compute the V-lambda targets with gamma = '
        '0.99 and lambda = 0.95"',
    ),
    "lambda_": (0.95, "paper: Appendix A, the same sentence as discount"),
    "actor_hidden": (300, 'paper: Appendix A, "three dense layers of size 300"'),
    "actor_layers": (3, "paper: Appendix A, the same sentence"),
    "value_hidden": (300, "paper: Appendix A, the same sentence"),
    "value_layers": (3, "paper: Appendix A, the same sentence"),
    "reward_hidden": (300, "paper: Appendix A, the same sentence"),
    "reward_layers": (
        3,
        'paper: Appendix A, "implement **all other functions** as three '
        'dense layers of size 300".  All other functions is reward, action '
        "and value alike.  code: disagrees on every count (400 units; "
        "reward 2, value 3, actor 4) and the paper is followed",
    ),
    "actor_min_std": (
        1e-4,
        "code: models.py ActionDecoder(..., min_std=1e-4, init_std=5, mean_scale=5)",
    ),
    "actor_init_std": (
        5.0,
        "code: dreamer.py action_init_std = 5.0, threaded into " "models.py ActionDecoder init_std",
    ),
    "actor_mean_scale": (
        5.0,
        'paper: Appendix A, "The action model outputs a tanh mean scaled by ' 'a factor of 5"',
    ),
    "detach_actor_input": (
        True,
        "code: dreamer.py _imagine_ahead, "
        "self._actor(tf.stop_gradient(self._dynamics.get_feat(state)))",
    ),
    "pcont": (
        False,
        "code: dreamer.py pcont = False.  paper: the discount head is "
        "introduced for discrete control, and DMC episodes do not terminate",
    ),
    "pcont_scale": (10.0, "code: dreamer.py pcont_scale = 10.0"),
    "pcont_layers": (
        3,
        "code: dreamer.py DenseDecoder((), 3, num_units, 'binary')",
    ),
    "action_space": (
        "continuous",
        "paper: Appendix A gives two settings.  The main paragraph is the "
        "Control Suite's, and its policy is the tanh-squashed Gaussian; the "
        '*Discrete control* paragraph says "the action model predicts the '
        'logits of a categorical distribution" for Atari and DeepMind Lab, '
        "which dreamer_discrete selects",
    ),
}


DREAMER_V2: dict[str, tuple[Any, str]] = {
    "sample_size": (64, _SAMPLE),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "action_dim": (1, _ACTION_DIM),
    "mean_only": (False, _MEAN_ONLY),
    "act_fn": ("elu", "code: configs.yaml rssm.act: elu"),
    "stoch_size": (
        32,
        "paper: Table D.1, Discrete latent dimensions 32; code: rssm.stoch 32",
    ),
    "discrete": (
        32,
        "paper: Table D.1, Discrete latent classes 32; code: rssm.discrete 32",
    ),
    "deter_size": (
        1024,
        "code: configs.yaml defaults rssm.deter 1024.  Table D.1's 600 is the "
        "Atari column and lives in dreamer_v2_atari",
    ),
    "hidden_size": (1024, "code: configs.yaml defaults rssm.hidden 1024"),
    "cnn_depth": (48, "code: configs.yaml encoder.cnn_depth 48"),
    "min_std": (0.1, "code: configs.yaml rssm.min_std 0.1"),
    "free_nats": (
        0.0,
        "code: configs.yaml kl.free 0.0 — v2 balances the divergence instead "
        "of flooring it.  The Control Suite config overrides this to 1.0",
    ),
    "kl_weight": (
        1.0,
        "code: configs.yaml loss_scales.kl 1.0.  Table D.1's beta = 0.1 is " "the Atari column",
    ),
    "kl_balance": (0.8, "paper: Table D.1, KL balancing alpha = 0.8"),
    "reward_hidden": (400, "paper: Table D.1, MLP number of units 400"),
    "reward_layers": (4, "paper: Table D.1, MLP number of layers 4"),
    "actor_hidden": (400, "paper: Table D.1, MLP number of units 400"),
    "actor_layers": (4, "paper: Table D.1, MLP number of layers 4"),
    "value_hidden": (400, "paper: Table D.1, MLP number of units 400"),
    "value_layers": (4, "paper: Table D.1, MLP number of layers 4"),
    "horizon": (15, "paper: Table D.1, Imagination horizon H = 15"),
    "discount": (
        0.99,
        "code: configs.yaml defaults discount 0.99.  Table D.1's 0.995 is the "
        "Atari column, where the released config in turn uses 0.999",
    ),
    "lambda_": (0.95, "paper: Table D.1, lambda-target parameter 0.95"),
    "actor_grad": (
        "auto",
        "code: configs.yaml actor_grad: auto — dynamics for a box, reinforce " "for buttons",
    ),
    "actor_grad_mix": (
        0.1,
        "code: configs.yaml actor_grad_mix 0.1.  Table D.1's rho = 1 is the "
        "Atari column, where the estimator is pure reinforce anyway",
    ),
    "actor_entropy": (
        2e-3,
        "code: configs.yaml actor_ent 2e-3.  Table D.1's 1e-3 is the Atari column",
    ),
    "actor_min_std": (0.1, "code: configs.yaml actor.min_std 0.1"),
    "slow_target_update": (
        100,
        "paper: Table D.1, Slow critic update interval 100",
    ),
    "slow_target_fraction": (
        1.0,
        "code: configs.yaml slow_target_fraction 1 — a hard copy, not a blend",
    ),
    "action_space": ("continuous", _ACTION_SPACE),
    "pcont": (True, "code: configs.yaml pred_discount True"),
    "pcont_scale": (1.0, "code: configs.yaml loss_scales.discount 1.0"),
    "pcont_layers": (4, "code: configs.yaml discount_head.layers 4"),
}


DREAMER_V3: dict[str, tuple[Any, str]] = {
    "sample_size": (64, _SAMPLE),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "action_dim": (1, _ACTION_DIM),
    "mean_only": (False, _MEAN_ONLY),
    "action_space": ("continuous", _ACTION_SPACE),
    "act_fn": ("silu", "paper: Table 4, Activation RMSNorm + SiLU"),
    "stoch_size": (
        32,
        "code: configs.yaml rssm.stoch 32.  paper: Table 3 says the number of "
        "latents is constant across sizes but does not print it",
    ),
    "discrete": (32, "paper: Table 3, Codes per latent (d/16) at the 50m rung"),
    "deter_size": (4096, "paper: Table 3, Recurrent units (8d) at the 50m rung"),
    "hidden_size": (512, "paper: Table 3, Hidden size (d) at the 50m rung"),
    "cnn_depth": (32, "paper: Table 3, Base CNN channels (d/16) at the 50m rung"),
    "blocks": (
        8,
        'paper: Table 3 text, "The sequence model has 8 times the number of '
        'recurrent units, split into 8 blocks"; code: rssm.blocks 8',
    ),
    "min_std": (
        0.1,
        "lucid: inherited from WorldModelConfig and unused — this family's "
        "latent is categorical, so there is no Gaussian scale to floor",
    ),
    "unimix": (0.01, "paper: Table 4, Latent unimix 1%"),
    "free_nats": (1.0, "paper: Table 4, Free nats 1"),
    "kl_weight": (
        1.0,
        "lucid: a multiplier over both divergence halves, inherited from "
        "WorldModelConfig.  The paper scales them separately, so at 1.0 this "
        "is a no-op and dyn_scale/rep_scale carry the meaning",
    ),
    "dyn_scale": (1.0, "paper: Table 4, Dynamics loss scale beta_dyn = 1"),
    "rep_scale": (0.1, "paper: Table 4, Representation loss scale beta_rep = 0.1"),
    "pred_scale": (1.0, "paper: Table 4, Reconstruction loss scale beta_pred = 1"),
    "num_bins": (
        255,
        "code: configs.yaml rewhead.bins and value.bins 255.  The paper gives "
        "the grid's range but not its resolution",
    ),
    "bin_range": (
        20.0,
        "paper: equation (10), B = symexp(-20 ... +20)",
    ),
    "reward_hidden": (512, "paper: Table 3, the MLPs are d wide"),
    "reward_layers": (
        1,
        "code: configs.yaml rewhead.layers 1.  Three is the actor's and the "
        "critic's depth, not every head's",
    ),
    "actor_hidden": (512, "paper: Table 3, the MLPs are d wide"),
    "actor_layers": (3, "code: configs.yaml policy.layers 3"),
    "value_hidden": (512, "paper: Table 3, the MLPs are d wide"),
    "value_layers": (3, "code: configs.yaml value.layers 3"),
    "horizon": (15, "paper: Table 4, Imagination horizon H = 15"),
    "discount": (
        0.997,
        "paper: Table 4, Discount horizon 1/(1-gamma) = 333, so gamma = 1 - 1/333",
    ),
    "lambda_": (0.95, "paper: Table 4, Return lambda 0.95"),
    "actor_entropy": (3e-4, "paper: Table 4, Actor entropy regularizer 3e-4"),
    "actor_min_std": (0.1, "code: configs.yaml policy.minstd 0.1"),
    "return_ema_decay": (0.99, "paper: Table 4, Actor RetNorm decay 0.99"),
    "return_low": (5.0, "paper: Table 4, Actor RetNorm scale Per(R,95)-Per(R,5)"),
    "return_high": (95.0, "paper: Table 4, the same entry"),
    "critic_ema": (
        0.02,
        "paper: Table 4, Critic EMA decay 0.98 — expressed here as the rate "
        "1 - 0.98; code: slowvalue.rate 0.02",
    ),
    "critic_slowreg": (1.0, "paper: Table 4, Critic EMA regularizer 1"),
    "replay_value_scale": (
        0.3,
        "paper: Table 4, Critic replay loss scale beta_repval = 0.3",
    ),
    "pcont": (True, "code: configs.yaml contdisc True"),
    "pcont_scale": (
        1.0,
        "code: configs.yaml loss_scales.con 1.0, the beta_pred family",
    ),
    "pcont_layers": (1, "code: configs.yaml conhead.layers 1"),
}


SCORE_SDE: dict[str, tuple[Any, str]] = {
    "sample_size": (
        32,
        "code: 32x32, following the DDPM backbone this family reuses and "
        "the CIFAR-10 setting the paper reports its Table 2 likelihoods on",
    ),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "act_fn": (
        "silu",
        "lucid: the generative domain's default activation; the paper "
        "specifies the backbone, which this family does not use",
    ),
    "sde_type": (
        "vp",
        "paper: Table 3 reports rows per process; VP is DDPM's continuous "
        "limit and the natural default for a DDPM backbone",
    ),
    "sigma_min": (
        0.01,
        'paper: Appendix C, "in SMLD, the noise scales is typically a '
        'geometric sequence where sigma_min is fixed to 0.01"',
    ),
    "sigma_max": (
        50.0,
        "paper: Appendix C — chosen per dataset by Song & Ermon (2020)'s "
        "Technique 1; 50 is the CIFAR-10 value",
    ),
    "beta_min": (
        0.1,
        'paper: Appendix C, "we let beta_min = 0.1 and beta_max = 20 to '
        'match the settings in Ho et al. (2020)"',
    ),
    "beta_max": (20.0, "paper: Appendix C, the same sentence as beta_min"),
    "num_scales": (
        1000,
        'paper: Appendix C, "the SMLD and DDPM models both use N = 1000 '
        'noise scales".  A sampler property here, not a trained-in one',
    ),
    "snr": (
        0.16,
        "code: the reference configs' Predictor-Corrector signal-to-noise " "ratio for CIFAR-10",
    ),
    "corrector_steps": (
        1,
        "code: configs.py n_steps_each = 1 — one Langevin step per "
        "predictor step, the reference default",
    ),
    "base_channels": (
        128,
        "code: DDPMConfig.base_channels 128 — the backbone this family reuses",
    ),
    "channel_mult": (
        (1, 2, 2, 2),
        "code: DDPMConfig.channel_mult, the CIFAR-10 multipliers",
    ),
    "num_res_blocks": (
        2,
        "paper: DDPM Appendix B, two residual blocks per resolution",
    ),
    "attention_resolutions": (
        (16,),
        "code: DDPMConfig.attention_resolutions — attention at 16x16",
    ),
    "num_heads": (4, "code: DDPMConfig.num_heads 4"),
    "dropout": (0.1, "code: DDPMConfig.dropout 0.1, the CIFAR-10 value"),
    "resnet_groups": (32, "code: DDPMConfig.resnet_groups 32"),
}


# Genie's paper tabulates layers, widths, heads and query/key sizes, and
# leaves the rest unstated.  The ``lucid:`` entries are those gaps.
_GENIE_UNSTATED_HEAD = (
    "lucid: the paper gives no query/key size for the latent action model "
    "(Table 5 lists layers, d_model and heads only), so dim // heads"
)
GENIE: dict[str, tuple[Any, str]] = {
    "sample_size": (
        (90, 160),
        'paper: Section 3, Datasets: "16s video clips at 10FPS, with 160x90 resolution"',
    ),
    "in_channels": (3, _RGB),
    "out_channels": (3, _RGB),
    "act_fn": (
        "gelu",
        "lucid: not stated — GELU, as in the ViT encoders the ST-ViViT " "tokenizer is built from",
    ),
    "num_frames": (
        16,
        'paper: Section 3, "For all modelling components we use a sequence '
        'length of 16 frames with an FPS of 10"',
    ),
    "mlp_ratio": (
        4.0,
        "lucid: not stated — the transformer convention, and what Jasmine "
        "(arXiv 2510.27002, ffn_dim 2048 at d_model 512) and 1X's ST "
        "transformer (mlp_ratio 4.0) both use; Jafar uses 1x.  No ratio "
        "reproduces the paper's parameter counts: the latent action model's "
        "attention alone exceeds its 300M, and the tokenizer and dynamics "
        "model would need ratios of 0.10 and 0.41",
    ),
    "commitment_cost": (
        0.25,
        "lucid: not stated — the commitment weight of the VQ-VAE paper the "
        "tokenizer's objective is named after",
    ),
    "code_reset_threshold": (
        0.1,
        "lucid: not stated — a codebook of eight entries collapses to a few "
        "live codes and the loss does not show it.  Jafar resets dead codes "
        "on a schedule (vq_reset_thresh); this reads the same failure as a "
        "share of the uniform one, so it does not depend on codebook size",
    ),
    "tokenizer_patch_size": (4, "paper: Table 7, patch size 4"),
    "num_codes": (1024, "paper: Table 7, num codes 1024"),
    "code_dim": (32, "paper: Table 7, latent dim 32"),
    "tokenizer_encoder_layers": (12, "paper: Table 7, encoder num layers 12"),
    "tokenizer_encoder_dim": (512, "paper: Table 7, encoder d model 512"),
    "tokenizer_encoder_heads": (8, "paper: Table 7, encoder num heads 8"),
    "tokenizer_encoder_head_dim": (64, "paper: Table 7, encoder k/q size 64"),
    "tokenizer_decoder_layers": (20, "paper: Table 7, decoder num layers 20"),
    "tokenizer_decoder_dim": (1024, "paper: Table 7, decoder d model 1024"),
    "tokenizer_decoder_heads": (16, "paper: Table 7, decoder num heads 16"),
    "tokenizer_decoder_head_dim": (64, "paper: Table 7, decoder k/q size 64"),
    "action_patch_size": (16, "paper: Table 5, patch size 16"),
    "num_latent_actions": (8, "paper: Table 5, num codes 8"),
    "action_dim": (32, "paper: Table 5, latent dim 32"),
    "action_encoder_layers": (20, "paper: Table 5, encoder num layers 20"),
    "action_encoder_dim": (1024, "paper: Table 5, encoder d model 1024"),
    "action_encoder_heads": (16, "paper: Table 5, encoder num heads 16"),
    "action_encoder_head_dim": (None, _GENIE_UNSTATED_HEAD),
    "action_decoder_layers": (20, "paper: Table 5, decoder num layers 20"),
    "action_decoder_dim": (1024, "paper: Table 5, decoder d model 1024"),
    "action_decoder_heads": (16, "paper: Table 5, decoder num heads 16"),
    "action_decoder_head_dim": (None, _GENIE_UNSTATED_HEAD),
    "dynamics_layers": (48, "paper: Table 12, num layers 48"),
    "dynamics_dim": (5120, "paper: Table 12, d model 5120"),
    "dynamics_heads": (36, "paper: Table 12, num heads 36"),
    "dynamics_head_dim": (128, "paper: Table 12, k/q size 128"),
    "dynamics_qk_norm": (
        True,
        'paper: Section 3, "we employ bfloat16 and QK norm for training our ' 'dynamics model"',
    ),
    "mask_ratio_min": (
        0.5,
        'paper: Section 2.1, "a Bernoulli distribution masking rate sampled '
        'uniformly between 0.5 and 1"',
    ),
    "mask_ratio_max": (1.0, "paper: Section 2.1, the upper end of that range, 1"),
    "maskgit_steps": (
        25,
        'paper: Section 3, "we perform 25 MaskGIT steps for the sampling of each frame"',
    ),
    "temperature": (
        2.0,
        'paper: Section 3, "with a temperature of 2 using random sampling"',
    ),
}


_FAMILIES = [
    ("planet", PlaNetConfig, PLANET),
    ("dreamer", DreamerConfig, DREAMER),
    ("dreamer_v2", DreamerV2Config, DREAMER_V2),
    ("dreamer_v3", DreamerV3Config, DREAMER_V3),
    ("score_sde", ScoreSDEConfig, SCORE_SDE),
    ("genie", GenieConfig, GENIE),
]
_KINDS = ("paper:", "code:", "lucid:")
_LOCATORS = (
    "Table",
    "Appendix",
    "equation",
    "Figure",
    "section",
    ".yaml",
    ".py",
    "configs",
)


@pytest.mark.parametrize("name,config_class,table", _FAMILIES)
class TestProvenance:
    def test_every_field_has_a_recorded_source(
        self, name: str, config_class: type, table: dict[str, tuple[Any, str]]
    ) -> None:
        """The check that matters — omission is what actually goes wrong.

        Every defect this family has shipped was a field nobody looked at.
        A new config field fails here until someone writes down where its
        value came from.
        """
        declared = {f.name for f in fields(config_class)}
        recorded = set(table)
        assert not declared - recorded, (
            f"{name}: no recorded source for {sorted(declared - recorded)} — "
            f"add it to the table with a paper:/code:/lucid: citation"
        )
        assert not recorded - declared, (
            f"{name}: table names fields that no longer exist: " f"{sorted(recorded - declared)}"
        )

    def test_every_recorded_value_is_the_configured_one(
        self, name: str, config_class: type, table: dict[str, tuple[Any, str]]
    ) -> None:
        """A citation attached to the wrong number is worse than none."""
        config = config_class()
        wrong = {
            field: (getattr(config, field), expected)
            for field, (expected, _) in table.items()
            if getattr(config, field) != expected
        }
        assert not wrong, f"{name}: configured vs recorded {wrong}"

    def test_every_citation_points_somewhere(
        self, name: str, config_class: type, table: dict[str, tuple[Any, str]]
    ) -> None:
        """A source citation must say *where*, not just that there is one.

        Length is the wrong test — "Table 4, Free nats 1" is short and
        completely sufficient, while a long sentence can still fail to
        name a location.  So the rule is a locator: a table, an appendix,
        an equation, a file, or a number.  ``lucid:`` entries are exempt
        because there is nowhere external to point; they owe a reason
        instead.
        """
        for field, (_, citation) in table.items():
            assert citation.startswith(_KINDS), f"{name}.{field}: {citation!r}"
            kind, body = citation.split(":", 1)
            body = body.strip()
            assert body, f"{name}.{field}: empty citation"
            if kind == "lucid":
                assert len(body) > 25, f"{name}.{field}: a convention owes a reason"
                continue
            assert any(token in body for token in _LOCATORS) or any(
                character.isdigit() for character in body
            ), f"{name}.{field}: {citation!r} names no location"


class TestTheTableIsNotVacuous:
    """Guards the suite above.  A table of only ``lucid:`` entries would
    pass every test while recording nothing about either source."""

    @pytest.mark.parametrize("name,_,table", _FAMILIES)
    def test_most_fields_trace_to_a_source_outside_lucid(
        self, name: str, _: type, table: dict[str, tuple[Any, str]]
    ) -> None:
        external = sum(1 for _, citation in table.values() if not citation.startswith("lucid:"))
        assert external / len(table) > 0.6, (
            f"{name}: only {external}/{len(table)} fields trace to a paper or " f"a released config"
        )

    @pytest.mark.parametrize("name,config_class,table", _FAMILIES)
    def test_an_unrecorded_field_would_fail(
        self, name: str, config_class: type, table: dict[str, tuple[Any, str]]
    ) -> None:
        """Guards the instrument.

        The coverage check is a set difference, and a set difference that
        is empty for the wrong reason — a table built *from* the config,
        say — would pass forever while checking nothing.  Drop an entry
        and the difference must name it.
        """
        for dropped in list(table)[:3]:
            incomplete = {k: v for k, v in table.items() if k != dropped}
            declared = {f.name for f in fields(config_class)}
            assert declared - set(incomplete) == {
                dropped
            }, f"{name}: dropping {dropped} did not register as missing"

    def test_the_families_disagree_where_their_papers_do(self) -> None:
        """PlaNet says two dense layers, Dreamer says three.

        Pinned because collapsing them is exactly the mistake that was
        made: Dreamer's reward head carried PlaNet's depth for a while,
        justified by a comment that had quietly rewritten Dreamer's own
        appendix.
        """
        assert PLANET["reward_layers"][0] == 2
        assert DREAMER["reward_layers"][0] == 3
        assert DREAMER["actor_layers"][0] == DREAMER["reward_layers"][0]
