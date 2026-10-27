"""GPT-1 configuration (Radford et al., 2018).

Paper: "Improving Language Understanding by Generative Pre-Training".  Single
canonical size in the original paper (L=12, H=768, A=12, T=512); we expose a
``GPTConfig`` parametrised the same way as ``BERTConfig`` so downstream code
can build smaller variants for tests / quick experiments without monkey-
patching.

Differences from BERT:
    - decoder-only (causal mask applied inside attention)
    - no segment embedding (``type_vocab_size`` is unused)
    - default activation is ``gelu_new`` (Hendrycks tanh approximation, per the
      GPT-1 reference code)
    - tied input/output embeddings by default
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.text._common._config import LanguageModelConfig, TextActivation


@model_family_meta(
    canonical_name="GPT",
    citation=(
        'Radford, Alec, et al. "Improving Language Understanding by '
        'Generative Pre-Training." OpenAI Technical Report, 2018.'
    ),
    theory=r"""
    GPT — *Generative Pre-Training* — is a decoder-only transformer trained
    on an autoregressive (causal) language-modelling objective.  Given a
    token sequence :math:`x = (x_1, \dots, x_T)`, the network factorises
    the joint distribution left-to-right and maximises

    .. math::

        \mathcal{L}_{\text{LM}}(\theta)
            = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}),

    where :math:`p_\theta(x_t \mid x_{<t}) =
    \operatorname{softmax}(W_e h_t^{(L)})` and :math:`h_t^{(L)}` is the
    final-layer hidden state at position :math:`t`.  Causality is enforced
    by an upper-triangular mask inside scaled dot-product attention, so
    position :math:`t` only attends to :math:`\{1, \dots, t\}`.

    The architecture is a stack of :math:`L=12` transformer blocks
    (``hidden_size=768``, 12 heads, ``intermediate_size=3072``) with
    learned absolute position embeddings and the tanh-approximated GELU
    activation (``gelu_new``).  Input and output token embeddings are
    tied — the LM head reuses :math:`W_e^\top` — which halves the
    parameter count of the softmax layer.  Unlike BERT, there is no
    segment embedding and no masked-LM objective; downstream tasks are
    handled by appending a task-specific delimiter sequence and reading
    off the final hidden state.

    Empirically, GPT established that **unsupervised pre-training on a
    large unlabeled corpus** (BookCorpus, ~800M tokens) followed by
    supervised fine-tuning transfers across a wide range of NLP tasks,
    foreshadowing the scaling-law regime later formalised by Kaplan et
    al. (2020).  Increasing depth, width, and corpus size yields smooth,
    power-law improvements in held-out perplexity — the foundation that
    GPT-2 and GPT-3 scale up.
    """,
)
@dataclass(frozen=True)
class GPTConfig(LanguageModelConfig):
    """Configuration for every GPT-1 variant.

    Args mirror the HuggingFace ``OpenAIGPTConfig`` field set so checkpoint
    porting is a flat key rename.

    Examples
    --------
    >>> from lucid.models.text.gpt._config import GPTConfig
    >>> cfg = GPTConfig()
    >>> cfg.model_type
    'gpt'
    >>> cfg.hidden_size, cfg.vocab_size
    (768, 40478)
    """

    model_type: ClassVar[str] = "gpt"

    # GPT-1 defaults (Radford et al., 2018).
    vocab_size: int = 40_478  # BPE vocab from the paper
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 512
    hidden_act: TextActivation = "gelu_new"
    # GPT-1 uses LayerNorm eps 1e-5 (Radford et al., 2018; matches GPT-2),
    # overriding the 1e-12 base default inherited from LanguageModelConfig.
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = None  # GPT-1 has no pad token

    # Classification fine-tuning head — used by ``GPTForSequenceClassification``.
    num_labels: int = 2
    classifier_dropout: float | None = None  # falls back to ``hidden_dropout``

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.classifier_dropout is not None and not (0.0 <= self.classifier_dropout < 1.0):
            raise ValueError(
                "classifier_dropout must be in [0, 1) or None, got " f"{self.classifier_dropout}"
            )
