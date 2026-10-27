"""Lucid 3.0 model zoo — public surface."""

# fmt: off

from lucid.models._auto import (
    AutoConfig, AutoModel,
    AutoModelForImageClassification,
    AutoModelForImageGeneration,
    AutoModelForLanguageModeling,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForWorldModeling,
)
from lucid.models._base   import ModelConfig, PretrainedModel
# The eight tasks a wrapper can be built for.  Exported so a caller can
# ask what a model is for (``isinstance(m, ImageClassificationModel)``)
# without matching on the class name — see ``lucid.models._tasks``.
from lucid.models._tasks import (
    TaskModel,
    ImageClassificationModel, ImageGenerationModel,
    LanguageModelingModel, ObjectDetectionModel,
    SemanticSegmentationModel, SequenceClassificationModel,
    TokenClassificationModel, WorldModelingModel,
)
from lucid.models._output import (
    ModelOutput, BaseModelOutput, BaseModelOutputWithPooling,
    ImageClassificationOutput, ObjectDetectionOutput, InstanceSegmentationOutput,
    SemanticSegmentationOutput, CausalLMOutput, MaskedLMOutput, Seq2SeqLMOutput,
    SequenceClassificationOutput, TokenClassificationOutput,
    QuestionAnsweringOutput,
    DiffusionModelOutput, VAEOutput, NormalizingFlowOutput, GenerationOutput,
)
from lucid.models._registry import create_model, is_model, list_models, model_entrypoint, register_model

# Text-domain infrastructure (Phase 4 base layer).
# RoPE / sinusoidal PE live in lucid.nn — import from there directly.
from lucid.models.text import LanguageModelConfig
# Generative-domain infrastructure (Phase 5 base layer).
from lucid.models.generative import (
    DiffusionModelConfig, GenerativeModelConfig,
    NormalizingFlowConfig, )
# 2013 — VAE (Kingma & Welling)
from lucid.models.generative.vae import (
    VAEConfig, VAEModel, VAEForImageGeneration,
    vae, hvae, vae_gen, hvae_gen,
)
# 2014 — NICE (Dinh, Krueger & Bengio)
from lucid.models.generative.nice import (
    NICEConfig, NICEModel, NICEForImageGeneration,
    nice_mnist, nice_tfd, nice_svhn, nice_cifar,
    nice_mnist_gen, nice_tfd_gen, nice_svhn_gen, nice_cifar_gen,
)
# 2017 — VQ-VAE (van den Oord, Vinyals & Kavukcuoglu)
from lucid.models.generative.vqvae import (
    VQVAEConfig, VQVAEModel, VQVAEForImageGeneration, vqvae, vqvae_gen,
)
# 2019 — PlaNet (Hafner et al.)
from lucid.models.generative.planet import (
    PlaNetConfig, PlaNetModel, PlaNetForWorldModeling, planet, planet_world_model,
)
# 2020 — Dreamer (Hafner et al.)
from lucid.models.generative.score_sde import (
    ScoreSDEConfig, ScoreSDEModel, ScoreSDEForImageGeneration, score_sde_vp, score_sde_vp_gen, score_sde_ve, score_sde_ve_gen,
    score_sde_subvp, score_sde_subvp_gen,
)
from lucid.models.generative.dreamer import (
    DreamerConfig, DreamerModel, DreamerForWorldModeling, dreamer, dreamer_world_model,
    dreamer_discrete, dreamer_discrete_world_model,
)
# 2021 — DreamerV2 (Hafner et al.)
from lucid.models.generative.dreamer_v2 import (
    DreamerV2Config, DreamerV2Model, DreamerV2ForWorldModeling, dreamer_v2, dreamer_v2_world_model,
    dreamer_v2_atari, dreamer_v2_atari_world_model,
    dreamer_v2_dmc, dreamer_v2_dmc_world_model,
)
from lucid.models.generative.dreamer_v3 import (
    DreamerV3Config, DreamerV3Model, DreamerV3ForWorldModeling, dreamer_v3_12m, dreamer_v3_12m_world_model,
    dreamer_v3_25m, dreamer_v3_25m_world_model,
    dreamer_v3_50m, dreamer_v3_50m_world_model,
    dreamer_v3_100m, dreamer_v3_100m_world_model,
    dreamer_v3_200m, dreamer_v3_200m_world_model,
    dreamer_v3_400m, dreamer_v3_400m_world_model,
)
# 2022 — Stable Diffusion / LDM (Rombach et al.)
from lucid.models.generative.stable_diffusion import (
    StableDiffusionConfig, StableDiffusionModel,
    StableDiffusionForImageGeneration,
    stable_diffusion, stable_diffusion_gen,
)
# 2021 — CLIP (Radford et al.)
from lucid.models.multimodal.clip import (
    CLIPConfig, CLIPModel, CLIPForZeroShotImageClassification,
    clip_vit_base_32, clip_vit_base_16,
    clip_vit_large_14, clip_vit_large_14_336,
    clip_vit_base_32_zero_shot, clip_vit_base_16_zero_shot,
    clip_vit_large_14_zero_shot, clip_vit_large_14_336_zero_shot,
)
# 2020 — DDPM (Ho et al.)
from lucid.models.generative.ddpm import (
    DDPMConfig, DDPMModel, DDPMForImageGeneration, ddpm_cifar, ddpm_lsun, ddpm_imagenet64,
    ddpm_cifar_gen, ddpm_lsun_gen, ddpm_imagenet64_gen,
)
# 2016 — RealNVP (Dinh, Sohl-Dickstein & Bengio)
from lucid.models.generative.realnvp import (
    RealNVPConfig, RealNVPModel, RealNVPForImageGeneration,
    realnvp_cifar, realnvp_imagenet32, realnvp_imagenet64,
    realnvp_lsun, realnvp_celeba,
    realnvp_cifar_gen, realnvp_imagenet32_gen, realnvp_imagenet64_gen,
    realnvp_lsun_gen, realnvp_celeba_gen,
)
# 2018 — Neural ODE (Chen, Rubanova, Bettencourt & Duvenaud)
from lucid.models.generative.neural_ode import (
    NeuralODEConfig, NeuralODEModel, NeuralODEForImageGeneration,
    neural_ode, neural_ode_gen,
)
# 2023 — Flow Matching (Lipman, Chen, Ben-Hamu, Nickel & Le)
from lucid.models.generative.flow_matching import (
    FlowMatchingConfig, FlowMatchingModel, FlowMatchingForImageGeneration,
    flow_matching_cifar, flow_matching_imagenet32,
    flow_matching_imagenet64, flow_matching_imagenet128,
    flow_matching_cifar_gen, flow_matching_imagenet32_gen,
    flow_matching_imagenet64_gen, flow_matching_imagenet128_gen,
)
# 2023 — Rectified Flow (Liu, Gong & Liu)
from lucid.models.generative.rectified_flow import (
    RectifiedFlowConfig, RectifiedFlowModel, RectifiedFlowForImageGeneration,
    rectified_flow_cifar, rectified_flow_bedroom, rectified_flow_church,
    rectified_flow_celeba_hq, rectified_flow_afhq_cat,
    rectified_flow_cifar_gen, rectified_flow_bedroom_gen,
    rectified_flow_church_gen, rectified_flow_celeba_hq_gen,
    rectified_flow_afhq_cat_gen,
)
# 2023 — DiT (Peebles & Xie)
from lucid.models.generative.dit import (
    DiTConfig, DiTModel, DiTForImageGeneration,
    dit_small_2,
    dit_small_4,
    dit_small_8,
    dit_base_2,
    dit_base_4,
    dit_base_8,
    dit_large_2,
    dit_large_4,
    dit_large_8,
    dit_xlarge_2,
    dit_xlarge_4,
    dit_xlarge_8,
    dit_small_2_gen,
    dit_small_4_gen,
    dit_small_8_gen,
    dit_base_2_gen,
    dit_base_4_gen,
    dit_base_8_gen,
    dit_large_2_gen,
    dit_large_4_gen,
    dit_large_8_gen,
    dit_xlarge_2_gen,
    dit_xlarge_4_gen,
    dit_xlarge_8_gen,
)
# 2024 — DIAMOND (Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce & Fleuret)
from lucid.models.generative.diamond import (
    DIAMONDConfig, DIAMONDModel, DIAMONDForWorldModeling,
    diamond, diamond_csgo, diamond_world_model,
)
# 2024 — Genie (Bruce, Dennis, Edwards, Parker-Holder, Shi, Hughes, Lai, et al.)
from lucid.models.generative.genie import (
    GenieConfig, GenieModel, GenieForWorldModeling,
    genie, genie_world_model, genie_coinrun, genie_coinrun_world_model,
)
# 2025 — V-JEPA 2-AC (Assran et al.)
from lucid.models.generative.vjepa2_ac import (
    VJEPA2ACConfig, VJEPA2ACModel, VJEPA2ACForWorldModeling,
    VJEPA2ACOutput, vjepa2_ac_vit_giant, vjepa2_ac_vit_giant_world_model,
)
# 2025 — MeanFlow (Geng, Deng, Bai, Kolter & He)
from lucid.models.generative.mean_flow import (
    MeanFlowConfig, MeanFlowModel, MeanFlowForImageGeneration,
    mean_flow_base_4, mean_flow_base_2, mean_flow_medium_2,
    mean_flow_large_2, mean_flow_xlarge_2,
    mean_flow_base_4_gen, mean_flow_base_2_gen, mean_flow_medium_2_gen,
    mean_flow_large_2_gen, mean_flow_xlarge_2_gen,
)
# 2026 — Terminal Velocity Matching (Zhou, Parger, Haque & Song)
from lucid.models.generative.tvm import (
    TVMConfig, TVMModel, TVMForImageGeneration,
    tvm_imagenet256, tvm_imagenet512,
    tvm_imagenet256_gen, tvm_imagenet512_gen,
)
# 2019 — NCSN (Song & Ermon)
from lucid.models.generative.ncsn import (
    NCSNConfig, NCSNModel, NCSNForImageGeneration,
    ncsn_cifar, ncsn_celeba,
    ncsn_cifar_gen, ncsn_celeba_gen,
)
# 2017 — Transformer (Vaswani et al.)
from lucid.models.text.transformer import (
    TransformerConfig, TransformerModel,
    TransformerForSeq2SeqLM,
    TransformerForSequenceClassification, TransformerForTokenClassification,
    transformer_base, transformer_large,
    transformer_base_seq2seq, transformer_large_seq2seq,
    transformer_base_cls, transformer_base_token_cls,
)
# 2018 — BERT (Devlin et al.)
from lucid.models.text.bert import (
    BERTConfig, BERTModel,
    BERTForCausalLM,
    BERTForMaskedLM,
    BERTForNextSentencePrediction,
    BERTForPreTraining, BERTForQuestionAnswering,
    BERTForSequenceClassification, BERTForTokenClassification,
    bert_tiny, bert_mini, bert_small, bert_medium, bert_base, bert_large,
    bert_base_mlm, bert_large_mlm,
    bert_base_cls, bert_large_cls,
    bert_base_token_cls, bert_base_qa, bert_large_qa,
    bert_base_causal_lm, bert_base_nsp, bert_base_pretrain,
)
# 2018 — GPT-1 (Radford et al.)
from lucid.models.text.gpt import (
    GPTConfig, GPTModel, GPTLMHeadModel,
    GPTDoubleHeadsModel, GPTForSequenceClassification,
    gpt, gpt_lm, gpt_cls,
)
# 2019 — GPT-2 (Radford et al.)
from lucid.models.text.gpt2 import (
    GPT2Config, GPT2Model, GPT2LMHeadModel,
    GPT2DoubleHeadsModel, GPT2ForSequenceClassification,
    gpt2_small, gpt2_medium, gpt2_large, gpt2_xlarge,
    gpt2_small_lm, gpt2_medium_lm, gpt2_large_lm, gpt2_xlarge_lm,
    gpt2_small_cls,
)
# 2020 — T5 (Raffel et al.)
from lucid.models.text.t5 import (
    T5Config, T5Model, T5ForSeq2SeqLM,
    t5_small, t5_base, t5_large, t5_3b, t5_11b,
    t5_small_seq2seq, t5_base_seq2seq, t5_large_seq2seq,
    t5_3b_seq2seq, t5_11b_seq2seq,
)
# 2021 — RoFormer (Su et al.)
from lucid.models.text.roformer import (
    RoFormerConfig, RoFormerModel,
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification, RoFormerForTokenClassification,
    roformer,
    roformer_mlm, roformer_cls, roformer_token_cls,
    roformer_multiple_choice, roformer_qa,
)

# Vision families — chronological order of publication
# 1998 — LeNet-5 (LeCun et al.)
from lucid.models.vision.lenet import (
    LeNetConfig, LeNet, LeNetForImageClassification,
    lenet_5, lenet_5_cls,
)
# 2012 — AlexNet (Krizhevsky, Sutskever & Hinton)
from lucid.models.vision.alexnet import (
    AlexNetConfig, AlexNet, AlexNetForImageClassification,
    alexnet, alexnet_cls,
)
# 2013 — ZFNet (Zeiler & Fergus)
from lucid.models.vision.zfnet import (
    ZFNetConfig, ZFNet, ZFNetForImageClassification,
    zfnet, zfnet_cls,
)
# 2014 — VGG (Simonyan & Zisserman)
from lucid.models.vision.vgg import (
    VGGConfig, VGG, VGGForImageClassification,
    vgg_11, vgg_11_bn, vgg_11_cls, vgg_11_bn_cls,
    vgg_13, vgg_13_bn, vgg_13_cls, vgg_13_bn_cls,
    vgg_16, vgg_16_bn, vgg_16_cls, vgg_16_bn_cls,
    vgg_19, vgg_19_bn, vgg_19_cls, vgg_19_bn_cls,
)
# 2014 — GoogLeNet / Inception v1 (Szegedy et al.)
from lucid.models.vision.googlenet import (
    GoogLeNetConfig, GoogLeNet, GoogLeNetForImageClassification, googlenet, googlenet_cls,
)
# 2015 — ResNet (He et al.)
from lucid.models.vision.resnet import (
    ResNetConfig, ResNet, ResNetForImageClassification,
    resnet_18, resnet_18_cls,
    resnet_34, resnet_34_cls,
    resnet_50, resnet_50_cls,
    resnet_101, resnet_101_cls,
    resnet_152, resnet_152_cls,
    resnet_200, resnet_200_cls,
    resnet_269, resnet_269_cls,
    wide_resnet_50, wide_resnet_50_cls,
    wide_resnet_101, wide_resnet_101_cls,
)
# 2015 — Inception v3 (Szegedy et al.)
from lucid.models.vision.inception import (
    InceptionConfig, InceptionV3, InceptionV3ForImageClassification, inception_v3, inception_v3_cls,
)
# 2016 — DenseNet (Huang et al.)
from lucid.models.vision.densenet import (
    DenseNetConfig, DenseNet, DenseNetForImageClassification,
    densenet_121, densenet_121_cls,
    densenet_161, densenet_161_cls,
    densenet_169, densenet_169_cls,
    densenet_201, densenet_201_cls,
    densenet_264, densenet_264_cls,
)
# 2016 — Inception-ResNet v2 (Szegedy et al.)
from lucid.models.vision.inception_resnet import (
    InceptionResNetConfig, InceptionResNetV2, InceptionResNetV2ForImageClassification,
    inception_resnet_v2, inception_resnet_v2_cls,
)
# 2016 — Inception-v4 (Szegedy et al.)
from lucid.models.vision.inception_v4 import (
    InceptionV4Config, InceptionV4, InceptionV4ForImageClassification,
    inception_v4, inception_v4_cls,
)
# 2017 — ResNeXt (Xie et al.)
from lucid.models.vision.resnext import (
    ResNeXtConfig, ResNeXt, ResNeXtForImageClassification,
    resnext_50_32x4d, resnext_50_32x4d_cls,
    resnext_101_32x4d, resnext_101_32x4d_cls,
    resnext_101_32x8d,
    resnext_101_64x4d,
    resnext_101_64x4d_cls, resnext_101_32x8d_cls,
)
# 2017 — Xception (Chollet)
from lucid.models.vision.xception import (
    XceptionConfig, Xception, XceptionForImageClassification, xception, xception_cls,
)
# 2017 — MobileNet v1 (Howard et al.)
from lucid.models.vision.mobilenet import (
    MobileNetConfig, MobileNet, MobileNetForImageClassification,
    mobilenet,     mobilenet_cls,
    mobilenet_075, mobilenet_075_cls,
    mobilenet_050, mobilenet_050_cls,
    mobilenet_025, mobilenet_025_cls,
)
# 2018 — SENet (Hu et al.)
from lucid.models.vision.senet import (
    SENetConfig, SENet, SENetForImageClassification,
    se_resnet_18, se_resnet_18_cls,
    se_resnet_34, se_resnet_34_cls,
    se_resnet_50, se_resnet_50_cls,
    se_resnet_101, se_resnet_101_cls,
    se_resnet_152, se_resnet_152_cls,
)
# 2018 — MobileNet v2 (Sandler et al.)
from lucid.models.vision.mobilenet_v2 import (
    MobileNetV2Config, MobileNetV2, MobileNetV2ForImageClassification,
    mobilenet_v2, mobilenet_v2_cls,
    mobilenet_v2_075, mobilenet_v2_075_cls,
)
# 2019 — SKNet (Li et al.)
from lucid.models.vision.sknet import (
    SKNetConfig, SKNet, SKNetForImageClassification,
    sk_resnet_18, sk_resnet_18_cls,
    sk_resnet_34, sk_resnet_34_cls,
    sk_resnet_50, sk_resnet_50_cls,
    sk_resnet_101, sk_resnet_101_cls,
    sk_resnext_50_32x4d, sk_resnext_50_32x4d_cls,
)
# 2019 — EfficientNet (Tan & Le)
from lucid.models.vision.efficientnet import (
    EfficientNetConfig, EfficientNet, EfficientNetForImageClassification,
    efficientnet_b0, efficientnet_b0_cls,
    efficientnet_b1, efficientnet_b1_cls,
    efficientnet_b2, efficientnet_b2_cls,
    efficientnet_b3, efficientnet_b3_cls,
    efficientnet_b4, efficientnet_b4_cls,
    efficientnet_b5, efficientnet_b5_cls,
    efficientnet_b6, efficientnet_b6_cls,
    efficientnet_b7, efficientnet_b7_cls,
)
# 2019 — MobileNet v3 (Howard et al.)
from lucid.models.vision.mobilenet_v3 import (
    MobileNetV3Config, MobileNetV3, MobileNetV3ForImageClassification,
    mobilenet_v3_large, mobilenet_v3_large_cls,
    mobilenet_v3_small, mobilenet_v3_small_cls,
)
# 2019 — CSPNet (Wang et al.)
from lucid.models.vision.cspnet import (
    CSPNetConfig, CSPNet, CSPNetForImageClassification,
    cspresnet_50, cspresnet_50_cls,
    cspresnext_50, cspresnext_50_cls,
    cspdarknet_53, cspdarknet_53_cls,
)
# 2020 — ResNeSt (Zhang et al.)
from lucid.models.vision.resnest import (
    ResNeStConfig, ResNeSt, ResNeStForImageClassification,
    resnest_14, resnest_14_cls,
    resnest_26, resnest_26_cls,
    resnest_50, resnest_50_cls,
    resnest_101, resnest_101_cls,
    resnest_200, resnest_200_cls,
    resnest_269, resnest_269_cls,
)
# 2020 — ViT (Dosovitskiy et al.)
from lucid.models.vision.vit import (
    ViTConfig, ViT, ViTForImageClassification,
    vit_base_16, vit_base_16_cls,
    vit_base_32, vit_base_32_cls,
    vit_large_16, vit_large_16_cls,
    vit_large_32, vit_large_32_cls,
    vit_huge_14, vit_huge_14_cls,
)
# 2021 — Swin Transformer (Liu et al.)
from lucid.models.vision.swin import (
    SwinConfig, SwinTransformer, SwinTransformerForImageClassification,
    swin_tiny, swin_tiny_cls, swin_small, swin_small_cls,
    swin_base, swin_base_cls, swin_large, swin_large_cls,
)
# 2021 — CoAtNet (Dai et al.)
from lucid.models.vision.coatnet import (
    CoAtNetConfig, CoAtNet, CoAtNetForImageClassification,
    coatnet_0, coatnet_0_cls,
    coatnet_1, coatnet_1_cls,
    coatnet_2, coatnet_2_cls,
    coatnet_3, coatnet_3_cls,
    coatnet_4, coatnet_4_cls,
    coatnet_5, coatnet_5_cls,
    coatnet_6, coatnet_6_cls,
    coatnet_7, coatnet_7_cls,
)
# 2021 — CvT (Wu et al.)
from lucid.models.vision.cvt import (
    CvTConfig, CvT, CvTForImageClassification,
    cvt_13, cvt_13_cls,
    cvt_21, cvt_21_cls,
    cvt_w24, cvt_w24_cls,
)
# 2021 — CrossViT (Chen et al.)
from lucid.models.vision.crossvit import (
    CrossViTConfig, CrossViT, CrossViTForImageClassification,
    crossvit_9, crossvit_9_cls,
    crossvit_tiny, crossvit_tiny_cls,
    crossvit_small, crossvit_small_cls,
    crossvit_base, crossvit_base_cls,
    crossvit_15, crossvit_15_cls,
    crossvit_18, crossvit_18_cls,
)
# 2021 — PVT (Wang et al.)
from lucid.models.vision.pvt import (
    PVTConfig, PVT, PVTForImageClassification,
    pvt_v2_b0, pvt_v2_b0_cls,
    pvt_v2_b1, pvt_v2_b1_cls,
    pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b2_cls,
    pvt_v2_b3, pvt_v2_b3_cls,
    pvt_v2_b4, pvt_v2_b4_cls,
    pvt_v2_b5, pvt_v2_b5_cls,
)
# 2022 — ConvNeXt (Liu et al.)
from lucid.models.vision.convnext import (
    ConvNeXtConfig, ConvNeXt, ConvNeXtForImageClassification,
    convnext_tiny, convnext_tiny_cls, convnext_small, convnext_small_cls,
    convnext_base, convnext_base_cls, convnext_large, convnext_large_cls,
    convnext_xlarge, convnext_xlarge_cls,
)
# 2022 — EfficientFormer (Li et al.)
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig, EfficientFormer, EfficientFormerForImageClassification,
    efficientformer_l1, efficientformer_l1_cls,
    efficientformer_l3, efficientformer_l3_cls,
    efficientformer_l7, efficientformer_l7_cls,
)
# 2022 — MaxViT (Tu et al.)
from lucid.models.vision.maxvit import (
    MaxViTConfig, MaxViT, MaxViTForImageClassification,
    maxvit_tiny, maxvit_tiny_cls,
    maxvit_small, maxvit_small_cls,
    maxvit_base, maxvit_base_cls,
    maxvit_large, maxvit_large_cls,
    maxvit_xlarge, maxvit_xlarge_cls,
)
# 2023 — InceptionNeXt (Yu et al.)
from lucid.models.vision.inception_next import (
    InceptionNeXtConfig, InceptionNeXt, InceptionNeXtForImageClassification,
    inception_next_tiny, inception_next_tiny_cls,
    inception_next_small, inception_next_small_cls,
    inception_next_base, inception_next_base_cls,
)
# 2023 — I-JEPA (Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun & Ballas)
from lucid.models.vision.ijepa import (
    IJEPAConfig, IJEPAModel, IJEPAForImageClassification,
    ijepa_base_16, ijepa_base_16_cls,
    ijepa_large_16, ijepa_large_16_cls,
    ijepa_huge_14, ijepa_huge_14_cls,
    ijepa_huge_16_448, ijepa_huge_16_448_cls,
)
# 2024 — V-JEPA (Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran & Ballas)
from lucid.models.vision.vjepa import (
    VJEPAConfig, VJEPAModel, VJEPAForVideoClassification,
    vjepa_large_16, vjepa_large_16_cls,
    vjepa_huge_16, vjepa_huge_16_cls,
    vjepa_huge_16_384, vjepa_huge_16_384_cls,
)
# 2024 — MobileNet v4 (Qin et al.)
from lucid.models.vision.mobilenet_v4 import (
    MobileNetV4Config, MobileNetV4, MobileNetV4ForImageClassification,
    mobilenet_v4_conv_small, mobilenet_v4_conv_small_cls,
    mobilenet_v4_conv_medium, mobilenet_v4_conv_medium_cls,
    mobilenet_v4_conv_large, mobilenet_v4_conv_large_cls,
    mobilenet_v4_hybrid_medium, mobilenet_v4_hybrid_medium_cls,
    mobilenet_v4_hybrid_large, mobilenet_v4_hybrid_large_cls,
)
# 2025 — V-JEPA 2 (Assran et al.)
from lucid.models.vision.vjepa2 import (
    VJEPA2Config, VJEPA2Model, VJEPA2ForVideoClassification, VJEPA2Output,
    vjepa2_vit_large, vjepa2_vit_large_cls,
    vjepa2_vit_huge, vjepa2_vit_huge_cls,
    vjepa2_vit_giant, vjepa2_vit_giant_cls,
    vjepa2_vit_giant_384, vjepa2_vit_giant_384_cls,
)
# 2014 — R-CNN (Girshick et al.)
from lucid.models.vision.rcnn import (
    RCNNConfig, RCNNForObjectDetection,
    rcnn,
)
# 2015 — Fast R-CNN (Girshick)
from lucid.models.vision.fast_rcnn import (
    FastRCNNConfig, FastRCNNForObjectDetection,
    fast_rcnn,
)
# 2015 — Faster R-CNN (Ren et al.)
from lucid.models.vision.faster_rcnn import (
    FasterRCNNConfig, FasterRCNNForObjectDetection,
    faster_rcnn, faster_rcnn_resnet50_fpn,
)
# 2017 — Mask R-CNN (He et al.)
from lucid.models.vision.mask_rcnn import (
    MaskRCNNConfig, MaskRCNNForObjectDetection,
    mask_rcnn, mask_rcnn_resnet50_fpn,
)
# 2015 — FCN (Long et al.)
from lucid.models.vision.fcn import (
    FCNConfig, FCNForSemanticSegmentation,
    fcn_resnet50, fcn_resnet101,
)
# 2015 — U-Net (Ronneberger et al.)
from lucid.models.vision.unet import (
    UNetConfig, UNetForSemanticSegmentation,
    unet,
    res_unet_2d, unet_3d, res_unet_3d,
)
# 2018 — Attention U-Net (Oktay et al.)
from lucid.models.vision.attention_unet import (
    AttentionUNetConfig, AttentionUNetForSemanticSegmentation,
    attention_unet,
    attention_unet_3d,
)
# 2020 — DETR (Carion et al.)
from lucid.models.vision.detr import (
    DETRConfig, DETRForObjectDetection,
    detr_resnet50, detr_resnet101,
)
# 2020 — EfficientDet (Tan et al.)
from lucid.models.vision.efficientdet import (
    EfficientDetConfig, EfficientDetForObjectDetection, efficientdet_d0, efficientdet_d1, efficientdet_d2, efficientdet_d3,
    efficientdet_d4, efficientdet_d5, efficientdet_d6, efficientdet_d7,
)
# 2021 — MaskFormer (Cheng et al.)
from lucid.models.vision.maskformer import (
    MaskFormerConfig, MaskFormerForSemanticSegmentation,
    maskformer_resnet50, maskformer_resnet101,
)
# 2022 — Mask2Former (Cheng et al.)
from lucid.models.vision.mask2former import (
    Mask2FormerConfig, Mask2FormerForSemanticSegmentation,
    mask2former_swin_tiny, mask2former_swin_small,
    mask2former_swin_base, mask2former_swin_large,
)
# 2016–2020 — YOLO family (Redmon et al., Bochkovskiy et al.)
from lucid.models.vision.yolo import (
    YOLOConfig, YOLOForObjectDetection,
    yolo, yolo_tiny,
    YOLOV2Config, YOLOV2ForObjectDetection,
    yolo_v2,
    YOLOV3Config, YOLOV3ForObjectDetection,
    yolo_v3, yolo_v3_tiny,
    YOLOV4Config, YOLOV4ForObjectDetection,
    yolo_v4,
)

__all__ = [    # ── Infrastructure ────────────────────────────────────────────────────────
    "ModelConfig", "PretrainedModel",
    # Task bases — one per registered task tag.
    "TaskModel", "ImageClassificationModel", "ImageGenerationModel", "LanguageModelingModel", "ObjectDetectionModel", "SemanticSegmentationModel", "SequenceClassificationModel", "TokenClassificationModel", "WorldModelingModel",
    "LanguageModelConfig", "GenerativeModelConfig", "DiffusionModelConfig", "NormalizingFlowConfig",
    "ModelOutput", "BaseModelOutput", "BaseModelOutputWithPooling",
    "ImageClassificationOutput", "ObjectDetectionOutput", "InstanceSegmentationOutput",
    "SemanticSegmentationOutput", "CausalLMOutput", "MaskedLMOutput", "Seq2SeqLMOutput",
    "SequenceClassificationOutput", "TokenClassificationOutput",
    "QuestionAnsweringOutput",
    "DiffusionModelOutput", "VAEOutput", "NormalizingFlowOutput", "GenerationOutput",
    "AutoConfig", "AutoModel",
    "AutoModelForImageClassification",
    "AutoModelForObjectDetection", "AutoModelForSemanticSegmentation",
    "AutoModelForLanguageModeling",
    "AutoModelForSequenceClassification", "AutoModelForTokenClassification",
    "AutoModelForImageGeneration",
    "AutoModelForWorldModeling",
    "CLIPConfig",
    "CLIPModel",
    "StableDiffusionConfig",
    "StableDiffusionModel",
    "StableDiffusionForImageGeneration",
    "stable_diffusion",
    "stable_diffusion_gen",
    "CLIPForZeroShotImageClassification",
    "clip_vit_base_32",
    "clip_vit_base_16",
    "clip_vit_large_14",
    "clip_vit_large_14_336",
    "clip_vit_base_32_zero_shot",
    "clip_vit_base_16_zero_shot",
    "clip_vit_large_14_zero_shot",
    "clip_vit_large_14_336_zero_shot",
    "create_model", "is_model", "list_models", "model_entrypoint", "register_model",
    # Structural-typing contracts (advisory; see arch-models-family-contract).
    # ── Vision (1998) LeNet ───────────────────────────────────────────────────
    "LeNetConfig", "LeNet", "LeNetForImageClassification",
    "lenet_5", "lenet_5_cls",
    # ── Vision (2012) AlexNet ─────────────────────────────────────────────────
    "AlexNetConfig", "AlexNet", "AlexNetForImageClassification",
    "alexnet", "alexnet_cls",
    # ── Vision (2013) ZFNet ───────────────────────────────────────────────────
    "ZFNetConfig", "ZFNet", "ZFNetForImageClassification",
    "zfnet", "zfnet_cls",
    # ── Vision (2014) VGG ────────────────────────────────────────────────────
    "VGGConfig", "VGG", "VGGForImageClassification",
    "vgg_11", "vgg_11_bn", "vgg_11_cls", "vgg_11_bn_cls",
    "vgg_13", "vgg_13_bn", "vgg_13_cls", "vgg_13_bn_cls",
    "vgg_16", "vgg_16_bn", "vgg_16_cls", "vgg_16_bn_cls",
    "vgg_19", "vgg_19_bn", "vgg_19_cls", "vgg_19_bn_cls",
    # ── Vision (2014) GoogLeNet ───────────────────────────────────────────────
    "GoogLeNetConfig", "GoogLeNet", "GoogLeNetForImageClassification", "googlenet", "googlenet_cls",
    # ── Vision (2015) ResNet ──────────────────────────────────────────────────
    "ResNetConfig", "ResNet", "ResNetForImageClassification",
    "resnet_18", "resnet_18_cls", "resnet_34", "resnet_34_cls",
    "resnet_50", "resnet_50_cls", "resnet_101", "resnet_101_cls", "resnet_152", "resnet_152_cls",
    "resnet_200", "resnet_200_cls", "resnet_269", "resnet_269_cls",
    "wide_resnet_50", "wide_resnet_50_cls", "wide_resnet_101", "wide_resnet_101_cls",
    # ── Vision (2015) Inception v3 ────────────────────────────────────────────
    "InceptionConfig", "InceptionV3", "InceptionV3ForImageClassification", "inception_v3", "inception_v3_cls",
    # ── Vision (2016) DenseNet ────────────────────────────────────────────────
    "DenseNetConfig", "DenseNet", "DenseNetForImageClassification",
    "densenet_121", "densenet_121_cls", "densenet_161", "densenet_161_cls",
    "densenet_169", "densenet_169_cls",
    "densenet_201", "densenet_201_cls", "densenet_264", "densenet_264_cls",
    # ── Vision (2016) Inception-ResNet v2 ────────────────────────────────────
    "InceptionResNetConfig", "InceptionResNetV2", "InceptionResNetV2ForImageClassification",
    "inception_resnet_v2", "inception_resnet_v2_cls",
    # ── Vision (2016) Inception-v4 ────────────────────────────────────────────
    "InceptionV4Config", "InceptionV4", "InceptionV4ForImageClassification",
    "inception_v4", "inception_v4_cls",
    # ── Vision (2017) ResNeXt ─────────────────────────────────────────────────
    "ResNeXtConfig", "ResNeXt", "ResNeXtForImageClassification",
    "resnext_50_32x4d", "resnext_50_32x4d_cls",
    "resnext_101_32x4d", "resnext_101_32x4d_cls",
    "resnext_101_32x8d",
    "resnext_101_64x4d",
    "resnext_101_64x4d_cls", "resnext_101_32x8d_cls",
    # ── Vision (2017) Xception ────────────────────────────────────────────────
    "XceptionConfig", "Xception", "XceptionForImageClassification", "xception", "xception_cls",
    # ── Vision (2017) MobileNet v1 ────────────────────────────────────────────
    "MobileNetConfig", "MobileNet", "MobileNetForImageClassification",
    "mobilenet", "mobilenet_cls",
    "mobilenet_075", "mobilenet_075_cls",
    "mobilenet_050", "mobilenet_050_cls",
    "mobilenet_025", "mobilenet_025_cls",
    # ── Vision (2018) SENet ───────────────────────────────────────────────────
    "SENetConfig", "SENet", "SENetForImageClassification",
    "se_resnet_18", "se_resnet_18_cls", "se_resnet_34", "se_resnet_34_cls",
    "se_resnet_50", "se_resnet_50_cls", "se_resnet_101", "se_resnet_101_cls",
    "se_resnet_152", "se_resnet_152_cls",
    # ── Vision (2018) MobileNet v2 ────────────────────────────────────────────
    "MobileNetV2Config", "MobileNetV2", "MobileNetV2ForImageClassification",
    "mobilenet_v2", "mobilenet_v2_cls", "mobilenet_v2_075", "mobilenet_v2_075_cls",
    # ── Vision (2019) SKNet ───────────────────────────────────────────────────
    "SKNetConfig", "SKNet", "SKNetForImageClassification",
    "sk_resnet_18", "sk_resnet_18_cls", "sk_resnet_34", "sk_resnet_34_cls",
    "sk_resnet_50", "sk_resnet_50_cls", "sk_resnet_101", "sk_resnet_101_cls",
    "sk_resnext_50_32x4d", "sk_resnext_50_32x4d_cls",
    # ── Vision (2019) EfficientNet ────────────────────────────────────────────
    "EfficientNetConfig", "EfficientNet", "EfficientNetForImageClassification",
    "efficientnet_b0", "efficientnet_b0_cls", "efficientnet_b1", "efficientnet_b1_cls",
    "efficientnet_b2", "efficientnet_b2_cls", "efficientnet_b3", "efficientnet_b3_cls",
    "efficientnet_b4", "efficientnet_b4_cls", "efficientnet_b5", "efficientnet_b5_cls",
    "efficientnet_b6", "efficientnet_b6_cls", "efficientnet_b7", "efficientnet_b7_cls",
    # ── Vision (2019) MobileNet v3 ────────────────────────────────────────────
    "MobileNetV3Config", "MobileNetV3", "MobileNetV3ForImageClassification",
    "mobilenet_v3_large", "mobilenet_v3_large_cls",
    "mobilenet_v3_small", "mobilenet_v3_small_cls",
    # ── Vision (2019) CSPNet ─────────────────────────────────────────────────
    "CSPNetConfig", "CSPNet", "CSPNetForImageClassification",
    "cspresnet_50", "cspresnet_50_cls",
    "cspresnext_50", "cspresnext_50_cls",
    "cspdarknet_53", "cspdarknet_53_cls",
    # ── Vision (2020) ResNeSt ────────────────────────────────────────────────
    "ResNeStConfig", "ResNeSt", "ResNeStForImageClassification",
    "resnest_14", "resnest_14_cls", "resnest_26", "resnest_26_cls",
    "resnest_50", "resnest_50_cls", "resnest_101", "resnest_101_cls",
    "resnest_200", "resnest_200_cls", "resnest_269", "resnest_269_cls",
    # ── Vision (2020) ViT ────────────────────────────────────────────────────
    "ViTConfig", "ViT", "ViTForImageClassification",
    "vit_base_16", "vit_base_16_cls", "vit_base_32", "vit_base_32_cls",
    "vit_large_16", "vit_large_16_cls", "vit_large_32", "vit_large_32_cls",
    "vit_huge_14", "vit_huge_14_cls",
    # ── Vision (2021) Swin Transformer ───────────────────────────────────────
    "SwinConfig", "SwinTransformer", "SwinTransformerForImageClassification",
    "swin_tiny", "swin_tiny_cls", "swin_small", "swin_small_cls",
    "swin_base", "swin_base_cls", "swin_large", "swin_large_cls",
    # ── Vision (2021) CoAtNet ────────────────────────────────────────────────
    "CoAtNetConfig", "CoAtNet", "CoAtNetForImageClassification",
    "coatnet_0", "coatnet_0_cls", "coatnet_1", "coatnet_1_cls",
    "coatnet_2", "coatnet_2_cls", "coatnet_3", "coatnet_3_cls",
    "coatnet_4", "coatnet_4_cls", "coatnet_5", "coatnet_5_cls",
    "coatnet_6", "coatnet_6_cls", "coatnet_7", "coatnet_7_cls",
    # ── Vision (2021) CvT ────────────────────────────────────────────────────
    "CvTConfig", "CvT", "CvTForImageClassification",
    "cvt_13", "cvt_13_cls", "cvt_21", "cvt_21_cls", "cvt_w24", "cvt_w24_cls",
    # ── Vision (2021) CrossViT ───────────────────────────────────────────────
    "CrossViTConfig", "CrossViT", "CrossViTForImageClassification",
    "crossvit_9", "crossvit_9_cls",
    "crossvit_tiny", "crossvit_tiny_cls", "crossvit_small", "crossvit_small_cls",
    "crossvit_base", "crossvit_base_cls", "crossvit_15", "crossvit_15_cls",
    "crossvit_18", "crossvit_18_cls",
    # ── Vision (2021) PVT ────────────────────────────────────────────────────
    "PVTConfig", "PVT", "PVTForImageClassification",
    "pvt_v2_b0", "pvt_v2_b0_cls", "pvt_v2_b1", "pvt_v2_b1_cls",
    "pvt_v2_b2", "pvt_v2_b2_li", "pvt_v2_b2_cls", "pvt_v2_b3", "pvt_v2_b3_cls",
    "pvt_v2_b4", "pvt_v2_b4_cls", "pvt_v2_b5", "pvt_v2_b5_cls",
    # ── Vision (2022) ConvNeXt ────────────────────────────────────────────────
    "ConvNeXtConfig", "ConvNeXt", "ConvNeXtForImageClassification",
    "convnext_tiny", "convnext_tiny_cls", "convnext_small", "convnext_small_cls",
    "convnext_base", "convnext_base_cls", "convnext_large", "convnext_large_cls",
    "convnext_xlarge", "convnext_xlarge_cls",
    # ── Vision (2022) EfficientFormer ────────────────────────────────────────
    "EfficientFormerConfig", "EfficientFormer", "EfficientFormerForImageClassification",
    "efficientformer_l1", "efficientformer_l1_cls",
    "efficientformer_l3", "efficientformer_l3_cls",
    "efficientformer_l7", "efficientformer_l7_cls",
    # ── Vision (2022) MaxViT ─────────────────────────────────────────────────
    "MaxViTConfig", "MaxViT", "MaxViTForImageClassification",
    "maxvit_tiny", "maxvit_tiny_cls",
    "maxvit_small", "maxvit_small_cls", "maxvit_base", "maxvit_base_cls",
    "maxvit_large", "maxvit_large_cls", "maxvit_xlarge", "maxvit_xlarge_cls",
    # ── Vision (2023) InceptionNeXt ──────────────────────────────────────────
    "InceptionNeXtConfig", "InceptionNeXt", "InceptionNeXtForImageClassification",
    "inception_next_tiny", "inception_next_tiny_cls",
    "inception_next_small", "inception_next_small_cls",
    "inception_next_base", "inception_next_base_cls",
    # ── Vision (2023) I-JEPA ─────────────────────────────────────────────────
    "IJEPAConfig", "IJEPAModel", "IJEPAForImageClassification",
    "ijepa_base_16", "ijepa_base_16_cls",
    "ijepa_large_16", "ijepa_large_16_cls",
    "ijepa_huge_14", "ijepa_huge_14_cls",
    "ijepa_huge_16_448", "ijepa_huge_16_448_cls",
    # ── Vision (2024) V-JEPA ─────────────────────────────────────────────────
    "VJEPAConfig", "VJEPAModel", "VJEPAForVideoClassification",
    "vjepa_large_16", "vjepa_large_16_cls",
    "vjepa_huge_16", "vjepa_huge_16_cls",
    "vjepa_huge_16_384", "vjepa_huge_16_384_cls",
    # ── Vision (2024) MobileNet v4 ───────────────────────────────────────────
    "MobileNetV4Config", "MobileNetV4", "MobileNetV4ForImageClassification",
    "mobilenet_v4_conv_small", "mobilenet_v4_conv_small_cls",
    "mobilenet_v4_conv_medium", "mobilenet_v4_conv_medium_cls",
    "mobilenet_v4_conv_large", "mobilenet_v4_conv_large_cls",
    "mobilenet_v4_hybrid_medium", "mobilenet_v4_hybrid_medium_cls",
    "mobilenet_v4_hybrid_large", "mobilenet_v4_hybrid_large_cls",
    # ── Vision (2025) V-JEPA 2 ───────────────────────────────────────────────
    "VJEPA2Config", "VJEPA2Model", "VJEPA2ForVideoClassification", "VJEPA2Output",
    "vjepa2_vit_large", "vjepa2_vit_large_cls",
    "vjepa2_vit_huge", "vjepa2_vit_huge_cls",
    "vjepa2_vit_giant", "vjepa2_vit_giant_cls",
    "vjepa2_vit_giant_384", "vjepa2_vit_giant_384_cls",
    # ── Vision (2015) FCN ────────────────────────────────────────────────────
    "FCNConfig", "FCNForSemanticSegmentation",
    "fcn_resnet50", "fcn_resnet101",
    # ── Vision (2015) U-Net ──────────────────────────────────────────────────
    "UNetConfig", "UNetForSemanticSegmentation",
    "unet",
    "res_unet_2d", "unet_3d", "res_unet_3d",
    # ── Vision (2018) Attention U-Net ────────────────────────────────────────
    "AttentionUNetConfig", "AttentionUNetForSemanticSegmentation",
    "attention_unet",
    "attention_unet_3d",
    # ── Vision (2014) R-CNN ───────────────────────────────────────────────────
    "RCNNConfig", "RCNNForObjectDetection",
    "rcnn",
    # ── Vision (2015) Fast R-CNN ──────────────────────────────────────────────
    "FastRCNNConfig", "FastRCNNForObjectDetection",
    "fast_rcnn",
    # ── Vision (2015) Faster R-CNN ────────────────────────────────────────────
    "FasterRCNNConfig", "FasterRCNNForObjectDetection",
    "faster_rcnn",
    "faster_rcnn_resnet50_fpn",
    # ── Vision (2017) Mask R-CNN ──────────────────────────────────────────────
    "MaskRCNNConfig", "MaskRCNNForObjectDetection",
    "mask_rcnn",
    "mask_rcnn_resnet50_fpn",
    # ── Vision (2020) DETR ────────────────────────────────────────────────────
    "DETRConfig", "DETRForObjectDetection",
    "detr_resnet50", "detr_resnet101",
    # ── Vision (2020) EfficientDet ────────────────────────────────────────────
    "EfficientDetConfig", "EfficientDetForObjectDetection", "efficientdet_d0", "efficientdet_d1", "efficientdet_d2", "efficientdet_d3",
    "efficientdet_d4", "efficientdet_d5", "efficientdet_d6", "efficientdet_d7",
    # ── Vision (2021) MaskFormer ──────────────────────────────────────────────
    "MaskFormerConfig", "MaskFormerForSemanticSegmentation",
    "maskformer_resnet50", "maskformer_resnet101",
    # ── Vision (2022) Mask2Former ─────────────────────────────────────────────
    "Mask2FormerConfig", "Mask2FormerForSemanticSegmentation",
    "mask2former_swin_tiny", "mask2former_swin_small",
    "mask2former_swin_base", "mask2former_swin_large",
    # ── Text (2017) Transformer (Vaswani et al.) ──────────────────────────────
    "TransformerConfig", "TransformerModel",
    "TransformerForSeq2SeqLM",
    "TransformerForSequenceClassification", "TransformerForTokenClassification",
    "transformer_base", "transformer_large",
    "transformer_base_seq2seq", "transformer_large_seq2seq",
    "transformer_base_cls", "transformer_base_token_cls",
    # ── Text (2018) BERT ──────────────────────────────────────────────────────
    "BERTConfig", "BERTModel",
    "BERTForCausalLM",
    "BERTForMaskedLM",
    "BERTForNextSentencePrediction",
    "BERTForPreTraining", "BERTForQuestionAnswering",
    "BERTForSequenceClassification", "BERTForTokenClassification",
    "bert_tiny", "bert_mini", "bert_small", "bert_medium", "bert_base", "bert_large",
    "bert_base_mlm", "bert_large_mlm",
    "bert_base_cls", "bert_large_cls",
    "bert_base_token_cls", "bert_base_causal_lm", "bert_base_nsp", "bert_base_pretrain", "bert_base_qa", "bert_large_qa",
    # ── Text (2018) GPT-1 ─────────────────────────────────────────────────────
    "GPTConfig", "GPTModel", "GPTLMHeadModel",
    "GPTDoubleHeadsModel", "GPTForSequenceClassification",
    "gpt", "gpt_lm", "gpt_cls",
    # ── Text (2019) GPT-2 ─────────────────────────────────────────────────────
    "GPT2Config", "GPT2Model", "GPT2LMHeadModel",
    "GPT2DoubleHeadsModel", "GPT2ForSequenceClassification",
    "gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xlarge",
    "gpt2_small_lm", "gpt2_medium_lm", "gpt2_large_lm", "gpt2_xlarge_lm",
    "gpt2_small_cls",
    # ── Text (2020) T5 ────────────────────────────────────────────────────────
    "T5Config", "T5Model", "T5ForSeq2SeqLM",
    "t5_small", "t5_base", "t5_large", "t5_3b", "t5_11b",
    "t5_small_seq2seq", "t5_base_seq2seq", "t5_large_seq2seq",
    "t5_3b_seq2seq", "t5_11b_seq2seq",
    # ── Text (2021) RoFormer ──────────────────────────────────────────────────
    "RoFormerConfig", "RoFormerModel",
    "RoFormerForMaskedLM",
    "RoFormerForMultipleChoice",
    "RoFormerForQuestionAnswering",
    "RoFormerForSequenceClassification", "RoFormerForTokenClassification",
    "roformer",
    "roformer_multiple_choice", "roformer_qa", "roformer_mlm", "roformer_cls", "roformer_token_cls",
    # ── Vision (2016) YOLOv1 ──────────────────────────────────────────────────
    "YOLOConfig", "YOLOForObjectDetection",
    "yolo", "yolo_tiny",
    # ── Vision (2017) YOLOv2 ──────────────────────────────────────────────────
    "YOLOV2Config", "YOLOV2ForObjectDetection",
    "yolo_v2",
    # ── Vision (2018) YOLOv3 ──────────────────────────────────────────────────
    "YOLOV3Config", "YOLOV3ForObjectDetection",
    "yolo_v3", "yolo_v3_tiny",
    # ── Vision (2020) YOLOv4 ──────────────────────────────────────────────────
    "YOLOV4Config", "YOLOV4ForObjectDetection",
    "yolo_v4",
    # ── Generative (2013) VAE ─────────────────────────────────────────────────
    "VAEConfig", "VAEModel", "VAEForImageGeneration",
    "vae", "hvae", "vae_gen", "hvae_gen",
    # ── Generative (2014) NICE ────────────────────────────────────────────────
    "NICEConfig", "NICEModel", "NICEForImageGeneration",
    "nice_mnist", "nice_tfd", "nice_svhn", "nice_cifar",
    "nice_mnist_gen", "nice_tfd_gen", "nice_svhn_gen", "nice_cifar_gen",
    # ── Generative (2017) VQ-VAE ──────────────────────────────────────────────
    "VQVAEConfig", "VQVAEModel", "VQVAEForImageGeneration", "vqvae", "vqvae_gen",
    # ── Generative (2019) PlaNet ──────────────────────────────────────────────
    "PlaNetConfig", "PlaNetModel", "PlaNetForWorldModeling", "planet", "planet_world_model",
    # ── Generative (2020) Dreamer ─────────────────────────────────────────────
    "DreamerConfig", "DreamerModel", "DreamerForWorldModeling", "ScoreSDEConfig", "ScoreSDEModel", "ScoreSDEForImageGeneration",
    "score_sde_vp", "score_sde_vp_gen", "score_sde_ve", "score_sde_ve_gen",
    "score_sde_subvp", "score_sde_subvp_gen",
    "dreamer", "dreamer_world_model",
    "dreamer_discrete", "dreamer_discrete_world_model",
    # ── Generative (2021) DreamerV2 ───────────────────────────────────────────
    "DreamerV2Config", "DreamerV2Model", "DreamerV2ForWorldModeling",
    "dreamer_v2", "dreamer_v2_world_model",
    "dreamer_v2_atari", "dreamer_v2_atari_world_model",
    "dreamer_v2_dmc", "dreamer_v2_dmc_world_model",
    "DreamerV3Config", "DreamerV3Model", "DreamerV3ForWorldModeling",
    "dreamer_v3_12m", "dreamer_v3_12m_world_model",
    "dreamer_v3_25m", "dreamer_v3_25m_world_model",
    "dreamer_v3_50m", "dreamer_v3_50m_world_model",
    "dreamer_v3_100m", "dreamer_v3_100m_world_model",
    "dreamer_v3_200m", "dreamer_v3_200m_world_model",
    "dreamer_v3_400m", "dreamer_v3_400m_world_model",
    # ── Generative (2020) DDPM ────────────────────────────────────────────────
    "DDPMConfig", "DDPMModel", "DDPMForImageGeneration", "ddpm_cifar", "ddpm_lsun", "ddpm_imagenet64",
    "ddpm_cifar_gen", "ddpm_lsun_gen", "ddpm_imagenet64_gen",
    # ── Generative (2016) RealNVP ─────────────────────────────────────────────
    "RealNVPConfig", "RealNVPModel", "RealNVPForImageGeneration",
    "realnvp_cifar", "realnvp_imagenet32", "realnvp_imagenet64",
    "realnvp_lsun", "realnvp_celeba",
    "realnvp_cifar_gen", "realnvp_imagenet32_gen", "realnvp_imagenet64_gen",
    "realnvp_lsun_gen", "realnvp_celeba_gen",
    # ── Generative (2018) Neural ODE ──────────────────────────────────────────
    "NeuralODEConfig", "NeuralODEModel", "NeuralODEForImageGeneration",
    "neural_ode", "neural_ode_gen",
    # ── Generative (2023) Flow Matching ───────────────────────────────────────
    "FlowMatchingConfig", "FlowMatchingModel", "FlowMatchingForImageGeneration",
    "flow_matching_cifar", "flow_matching_imagenet32",
    "flow_matching_imagenet64", "flow_matching_imagenet128",
    "flow_matching_cifar_gen", "flow_matching_imagenet32_gen",
    "flow_matching_imagenet64_gen", "flow_matching_imagenet128_gen",
    # ── Generative (2023) Rectified Flow ──────────────────────────────────────
    "RectifiedFlowConfig", "RectifiedFlowModel", "RectifiedFlowForImageGeneration",
    "rectified_flow_cifar", "rectified_flow_bedroom", "rectified_flow_church",
    "rectified_flow_celeba_hq", "rectified_flow_afhq_cat",
    "rectified_flow_cifar_gen", "rectified_flow_bedroom_gen",
    "rectified_flow_church_gen", "rectified_flow_celeba_hq_gen",
    "rectified_flow_afhq_cat_gen",
    # ── Generative (2023) DiT ─────────────────────────────────────────────────
    "DiTConfig", "DiTModel", "DiTForImageGeneration",
    "dit_small_2",
    "dit_small_4",
    "dit_small_8",
    "dit_base_2",
    "dit_base_4",
    "dit_base_8",
    "dit_large_2",
    "dit_large_4",
    "dit_large_8",
    "dit_xlarge_2",
    "dit_xlarge_4",
    "dit_xlarge_8",
    "dit_small_2_gen",
    "dit_small_4_gen",
    "dit_small_8_gen",
    "dit_base_2_gen",
    "dit_base_4_gen",
    "dit_base_8_gen",
    "dit_large_2_gen",
    "dit_large_4_gen",
    "dit_large_8_gen",
    "dit_xlarge_2_gen",
    "dit_xlarge_4_gen",
    "dit_xlarge_8_gen",
    # ── Generative (2024) DIAMOND ─────────────────────────────────────────────
    "DIAMONDConfig", "DIAMONDModel", "DIAMONDForWorldModeling",
    "diamond", "diamond_world_model", "diamond_csgo",
    # ── Generative (2024) Genie ───────────────────────────────────────────────
    "GenieConfig", "GenieModel", "GenieForWorldModeling",
    "genie", "genie_world_model", "genie_coinrun", "genie_coinrun_world_model",
    # ── Generative (2025) V-JEPA 2-AC ─────────────────────────────────────────
    "VJEPA2ACConfig", "VJEPA2ACModel", "VJEPA2ACForWorldModeling", "VJEPA2ACOutput",
    "vjepa2_ac_vit_giant", "vjepa2_ac_vit_giant_world_model",
    # ── Generative (2025) MeanFlow ────────────────────────────────────────────
    "MeanFlowConfig", "MeanFlowModel", "MeanFlowForImageGeneration",
    "mean_flow_base_4", "mean_flow_base_2", "mean_flow_medium_2",
    "mean_flow_large_2", "mean_flow_xlarge_2",
    "mean_flow_base_4_gen", "mean_flow_base_2_gen", "mean_flow_medium_2_gen",
    "mean_flow_large_2_gen", "mean_flow_xlarge_2_gen",
    # ── Generative (2026) Terminal Velocity Matching ──────────────────────────
    "TVMConfig", "TVMModel", "TVMForImageGeneration",
    "tvm_imagenet256", "tvm_imagenet512",
    "tvm_imagenet256_gen", "tvm_imagenet512_gen",
    # ── Generative (2019) NCSN ────────────────────────────────────────────────
    "NCSNConfig", "NCSNModel", "NCSNForImageGeneration",
    "ncsn_cifar", "ncsn_celeba",
    "ncsn_cifar_gen", "ncsn_celeba_gen",
]
