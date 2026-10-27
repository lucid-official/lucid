"""T5 configuration — Raffel et al., JMLR 2020.

Paper: "Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer".  An encoder-decoder Transformer whose
positions enter as a learned scalar per head and log-bucketed relative
distance, with a scale-only (RMS) layer norm in front of every sub-layer
and no 1/sqrt(d_kv) in the attention logits.

The config inherits :class:`LanguageModelConfig` and keeps its names for
the shared dimensions — ``hidden_size`` is d_model, ``intermediate_size``
is d_ff, ``num_hidden_layers`` counts the encoder blocks — and adds the
knobs T5 does not share with the other text families:

    * ``head_dim``                — d_kv, decoupled from
      ``hidden_size / num_attention_heads`` (T5-3B and T5-11B attend at a
      width four and sixteen times their model width).
    * ``num_decoder_layers``      — decoder blocks.
    * ``relative_attention_*``    — the bucketing of relative distance.
    * ``decoder_start_token_id``  — what the decoder reads first.
    * ``initializer_factor``      — the scale of the Mesh TensorFlow init.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.text._common._config import LanguageModelConfig, TextActivation


@model_family_meta(
    canonical_name="T5",
    citation=(
        'Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with '
        'a Unified Text-to-Text Transformer." Journal of Machine Learning '
        "Research, vol. 21, no. 140, 2020, pp. 1–67."
    ),
    theory=r"""
    T5 casts every task — translation, summarisation, classification,
    regression — as **text in, text out**: a task prefix such as
    ``"summarize: "`` is prepended to the input and the answer is decoded
    as tokens, so one encoder-decoder, one loss and one decoding procedure
    serve them all.

    **Relative position bias.**  No position is added to the tokens.
    Each attention head instead adds a learned scalar to its logits,
    looked up by the bucketed distance :math:`j - i` from query to key:

    .. math::

        A_{ij} = q_i^{\top} k_j + b_{h,\,\beta(j - i)},

    with **no** :math:`1/\sqrt{d_{kv}}` scaling — the Mesh TensorFlow
    initialisation folds it into the query projection.  The bucket map
    :math:`\beta` is exact for the :math:`m` nearest distances and
    logarithmic beyond, saturating at ``relative_attention_max_distance``
    :math:`D`.  For :math:`n` buckets per direction and :math:`m = n/2`,

    .. math::

        \beta(r) =
        \begin{cases}
            |r| & |r| < m, \\
            \min\!\Bigl(n - 1,\; m + \Bigl\lfloor
                \dfrac{\log(|r| / m)}{\log(D / m)}\,(n - m)
            \Bigr\rfloor\Bigr) & \text{otherwise.}
        \end{cases}

    The encoder splits its 32 buckets between the two signs of
    :math:`r` (:math:`n = 16`); the causal decoder uses all 32 for
    :math:`r \le 0`.  One table per stack, computed once and shared by
    every layer.

    **Simplified layer norm.**  Every sub-layer reads
    :math:`x / \mathrm{RMS}(x) \odot \gamma` — no mean subtraction, no
    bias — and adds its output back to the unnormalised stream
    (pre-norm residual), with a final norm closing each stack.

    **Objective.**  Pre-training corrupts 15% of the input tokens in
    spans of mean length 3, replaces each span with a unique sentinel
    and asks the decoder for the dropped spans, each after its sentinel.
    Every task then trains the same per-token cross-entropy

    .. math::

        \mathcal{L} = -\sum_{t} \log p_\theta(y_t \mid x,\, y_{<t}),

    the decoder reading the target shifted right behind
    ``decoder_start_token_id``.  The paper's five sizes span 60M to 11B
    parameters; T5-11B set the state of the art on GLUE, SuperGLUE,
    SQuAD and CNN/Daily Mail at publication.
    """,
)
@dataclass(frozen=True)
class T5Config(LanguageModelConfig):
    """Configuration for every T5 variant (Raffel et al., 2020, §3.7).

    The defaults are T5-Small.  ``vocab_size`` is the 32 000-piece
    SentencePiece vocabulary with 100 sentinels, padded to 32 128 rows.

    Examples
    --------
    >>> from lucid.models.text.t5._config import T5Config
    >>> cfg = T5Config()
    >>> cfg.model_type
    't5'
    >>> cfg.hidden_size, cfg.inner_dim, cfg.vocab_size
    (512, 512, 32128)

    The attention width is ``num_attention_heads * head_dim``, which need
    not be the model width:

    >>> T5Config(hidden_size=1024, num_attention_heads=32, head_dim=128).inner_dim
    4096
    """

    model_type: ClassVar[str] = "t5"

    # T5-Small — Raffel et al. (2020) §3.7 and the released checkpoint.
    vocab_size: int = 32_128
    hidden_size: int = 512  # d_model
    num_hidden_layers: int = 6  # encoder blocks
    num_attention_heads: int = 8
    intermediate_size: int = 2_048  # d_ff
    # §3.1.1: the feed-forward is a dense layer, a ReLU, and a dense layer.
    hidden_act: TextActivation = "relu"
    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    layer_norm_eps: float = 1e-6

    head_dim: int = 64  # d_kv
    num_decoder_layers: int = 6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    # The decoder starts from the pad id: T5 has no beginning-of-sequence.
    decoder_start_token_id: int = 0
    # Scales every init standard deviation (Mesh TensorFlow's
    # ``initializer_factor``); ``initializer_range`` is not read by T5.
    initializer_factor: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.num_decoder_layers <= 0:
            raise ValueError(f"num_decoder_layers must be positive, got {self.num_decoder_layers}")
        buckets = self.relative_attention_num_buckets
        if buckets < 4:
            raise ValueError(
                "relative_attention_num_buckets must be at least 4 — the encoder "
                "splits them by sign and keeps half of each side exact, got "
                f"{buckets}"
            )
        if self.relative_attention_max_distance <= buckets // 2:
            raise ValueError(
                "relative_attention_max_distance must exceed the exact range "
                f"({buckets // 2} positions for {buckets} buckets), got "
                f"{self.relative_attention_max_distance}"
            )
        if not 0 <= self.decoder_start_token_id < self.vocab_size:
            raise ValueError(
                f"decoder_start_token_id must be a token id below vocab_size "
                f"{self.vocab_size}, got {self.decoder_start_token_id}"
            )
        if self.initializer_factor <= 0.0:
            raise ValueError(f"initializer_factor must be positive, got {self.initializer_factor}")

    @property
    def inner_dim(self) -> int:
        """Attention width — ``num_attention_heads * head_dim``."""
        return self.num_attention_heads * self.head_dim
