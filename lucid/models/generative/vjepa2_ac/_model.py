"""Latent action-conditioned V-JEPA 2 world model."""

from dataclasses import dataclass
from typing import ClassVar, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, FeatureInfo
from lucid.models._output import ModelOutput
from lucid.models._tasks import WorldModelingModel
from lucid.models.generative.vjepa2_ac._config import VJEPA2ACConfig
from lucid.models.vision.vjepa2._model import _ActionConditionedPredictor, _RoPEVideoEncoder

__all__ = ["VJEPA2ACModel", "VJEPA2ACForWorldModeling", "VJEPA2ACOutput"]


@dataclass(slots=True)
class VJEPA2ACOutput(ModelOutput):
    r"""Latent transition output of V-JEPA 2-AC.

    Attributes
    ----------
    prediction : Tensor
        Predicted latent grids, ``(B, S * P, D)``.
    context : Tensor
        Encoded context grids consumed by the predictor.
    target : Tensor
        The latents actually observed at the predicted frames.
    loss : Tensor
        Representation-space L1 between the two.
    """

    prediction: Tensor
    context: Tensor
    target: Tensor
    loss: Tensor


class VJEPA2ACModel(PretrainedModel, BackboneMixin):
    r"""V-JEPA 2-AC encoder plus action-conditioned latent predictor.

    Parameters
    ----------
    config : VJEPA2ACConfig
        Frozen model configuration.

    Notes
    -----
    The encoder is applied independently to each frame, repeated to fill
    the patch embedding's tubelet.  A video with ``T`` frames therefore
    produces ``T`` action-model steps, each containing
    ``(image_size // patch_size) ** 2`` spatial tokens — one step per
    frame, as the released training loop and energy-landscape example
    both do.  :meth:`forward` takes one conditioning row per *transition*,
    so ``T - 1`` of them.  Planning and controller search stay outside the
    model.
    """

    config_class: ClassVar[type[VJEPA2ACConfig]] = VJEPA2ACConfig

    def __init__(self, config: VJEPA2ACConfig) -> None:
        super().__init__(config)
        self.config: VJEPA2ACConfig = config
        self.encoder = _RoPEVideoEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            tubelet_size=config.tubelet_size,
            in_channels=config.in_channels,
            dim=config.encoder_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            layer_norm_eps=config.layer_norm_eps,
            qkv_bias=config.qkv_bias,
            init_std=config.init_std,
            use_silu=config.use_silu,
            wide_silu=config.wide_silu,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
        )
        self.predictor = _ActionConditionedPredictor(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.encoder_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            extrinsics_dim=config.extrinsics_dim,
            use_extrinsics=config.use_extrinsics,
            frame_causal=config.frame_causal,
            mlp_ratio=config.predictor_mlp_ratio,
            layer_norm_eps=config.layer_norm_eps,
            qkv_bias=config.qkv_bias,
            init_std=config.init_std,
            use_silu=config.use_silu,
            wide_silu=config.wide_silu,
            rope_base=config.rope_base,
        )
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.encoder_dim, reduction=config.patch_size)
        ]

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        """The spatial latent-grid feature stage."""
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        """Encode a video into flattened per-step spatial tokens."""
        return self.encode(x)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return encoder and predictor parameters for post-training."""
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    def _check_video(self, video: Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must have shape (B, T, C, H, W), got {video.shape}")
        _, frames, channels, height, width = (int(s) for s in video.shape)
        if frames < 1:
            raise ValueError(f"video must carry at least one frame, got {frames}")
        expected = (self.config.in_channels, self.config.image_size, self.config.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"video frames must have shape (C, H, W) = {expected}, got "
                f"{(channels, height, width)}"
            )

    def encode(self, video: Tensor) -> Tensor:
        r"""Encode every frame to its own latent grid, ``(B, T * P, D)``.

        One action-model step is **one frame**, not one two-frame tubelet.
        The encoder's patch embedding is three-dimensional and wants
        ``tubelet_size`` frames, so each frame is repeated to fill one —
        which is what the released code does in both its training loop
        and its energy-landscape example::

            c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)

        Pairing two *distinct* frames instead would halve the temporal
        resolution and take one action per two frames, which is not the
        rate the released checkpoint was post-trained at.
        """
        self._check_video(video)
        batch, frames, channels, height, width = (int(s) for s in video.shape)
        clips = video.reshape(batch * frames, 1, channels, height, width).repeat(
            1, self.config.tubelet_size, 1, 1, 1
        )
        tokens = self.encoder(clips)
        spatial = self.config.tokens_per_step
        return tokens.reshape(batch, frames * spatial, self.config.encoder_dim)

    def normalize(self, tokens: Tensor) -> Tensor:
        """Layer-normalise latents, as the released loop does to both sides."""
        if not self.config.normalize_targets:
            return tokens
        return F.layer_norm(tokens, (int(tokens.shape[-1]),))

    def predict(
        self, context: Tensor, actions: Tensor, states: Tensor, extrinsics: Tensor | None = None
    ) -> Tensor:
        """Predict the next latent grids from encoded context and controls."""
        if extrinsics is None:
            prediction = self.predictor(context, actions, states)
        else:
            prediction = self.predictor(context, actions, states, extrinsics)
        return prediction

    @override
    def forward(
        self, video: Tensor, actions: Tensor, states: Tensor, extrinsics: Tensor | None = None
    ) -> VJEPA2ACOutput:
        r"""One teacher-forced transition step over a clip.

        The clip's first ``T - 1`` frames are the context and its last
        ``T - 1`` are the target: the predictor is asked what the *next*
        frame looks like, which is the post-training objective.  A
        conditioning row is therefore one per transition, so ``actions``,
        ``states`` and ``extrinsics`` carry ``T - 1`` of them.

        Parameters
        ----------
        video : Tensor
            Clip ``(B, T, C, H, W)`` with ``T >= 2``.
        actions, states : Tensor
            ``(B, T - 1, action_dim)`` and ``(B, T - 1, state_dim)``.
        extrinsics : Tensor or None, optional
            ``(B, T - 1, extrinsics_dim)`` when the config enables them.

        Returns
        -------
        VJEPA2ACOutput
            The predicted next latents, the context they came from, the
            latents actually observed at those frames, and the L1 between
            them.

        Notes
        -----
        Both sides are layer-normalised before the loss when
        ``normalize_targets`` is set, as the released loop does.  Comparing
        a prediction against the frame it was *given* — which is what
        scoring the unshifted latents would do — is satisfied by copying
        the input, since the predictor's attention is frame-causal.
        """
        frames = int(video.shape[1])
        if frames < 2:
            raise ValueError(f"a transition needs at least two frames, got {frames}")
        spatial = self.config.tokens_per_step
        encoded = self.normalize(self.encode(video))
        context = encoded[:, :-spatial]
        prediction = self.predict(context, actions, states, extrinsics)
        target = encoded[:, spatial:].detach()
        return VJEPA2ACOutput(
            prediction=self.normalize(prediction),
            context=context,
            target=target,
            loss=F.l1_loss(self.normalize(prediction), target),
        )


class VJEPA2ACForWorldModeling(WorldModelingModel):
    r"""V-JEPA 2-AC under the zoo's world-modeling task.

    Parameters
    ----------
    config : VJEPA2ACConfig
        Frozen model configuration.

    Attributes
    ----------
    vjepa2_ac : VJEPA2ACModel
        The encoder and the action-conditioned predictor.  The wrapper
        adds no parameters of its own.

    Notes
    -----
    It exists so ``AutoModelForWorldModeling`` can reach the family and so
    the registry can tag it by task.  The forward is one teacher-forced
    latent transition; a rollout is that forward applied to its own
    output, and the controller that decides the actions stays outside.
    """

    config_class: ClassVar[type[VJEPA2ACConfig]] = VJEPA2ACConfig

    def __init__(self, config: VJEPA2ACConfig) -> None:
        super().__init__(config)
        self.config: VJEPA2ACConfig = config
        self.vjepa2_ac = VJEPA2ACModel(config)

    @override
    def forward(
        self, video: Tensor, actions: Tensor, states: Tensor, extrinsics: Tensor | None = None
    ) -> VJEPA2ACOutput:
        """Return an action-conditioned latent transition."""
        if extrinsics is None:
            result = self.vjepa2_ac(video, actions, states)
        else:
            result = self.vjepa2_ac(video, actions, states, extrinsics)
        return result
