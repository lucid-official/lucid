"""Names vulture cannot see used, listed so they are not counted as dead.

vulture reads this file as Python and treats every name in it as used — in
*every* module: its used-name set is global, so an entry also hides any dead
symbol of the same name elsewhere.  An entry therefore has to earn its place:

- vulture needs it: without it, a production definition is reported;
- something outside Python's view reaches that definition — the C++ engine
  through pybind11, a lookup by string, a protocol the runtime invokes, a
  test or tool that probes the public surface by name — named on its line.

A name that is merely unused belongs in a deletion, not here, and a name
Python code already uses needs no entry.  The file is part of the gate's
contract (``config.contract``): adding or removing a name is
``--rebaseline --reason``, while comments and layout are free.

Dunders are already ignored by the gate (``--ignore-names "__*__"``).
"""

from decimal import Context
from unittest.mock import Mock

Context.prec  # set through getcontext() in lucid/diffeq/_collocation.py

_ = Mock()

# ── C++ engine attributes and stdlib hooks ────────────────────────────────
_.once_differentiable  # read by C++ PythonNode in lucid/_C/autograd/CustomFunction.cpp
_._register_node_prehook  # called by name from lucid/_C/bindings/bind_autograd.cpp
_._register_node_hook  # called by name from lucid/_C/bindings/bind_autograd.cpp
_.redraw_rng  # read by C++ Tracer in lucid/_C/compile/Tracer.h
_.persistent_id  # pickle.Pickler.persistent_id hook in lucid/serialization/__init__.py
_._on_queue_feeder_error  # multiprocessing.queues.Queue hook in lucid/utils/data/dataloader.py

# ── TYPE_CHECKING protocols referenced in string-quoted cast(...) ─────────
_._FloatConv  # cast("_FloatConv", mod) in lucid/nn/quantized/conv.py
_._FloatConvT  # cast("_FloatConvT", mod) in lucid/nn/quantized/conv_transpose.py
_._FloatLinear  # cast("_FloatLinear", mod) in lucid/nn/quantized/{,dynamic/}linear.py
_._FloatEmbedding  # cast("_FloatEmbedding", mod) in lucid/nn/quantized/sparse.py

# ── GenieConfig fields read dynamically via getattr(self, f"{stack}_...") ─
_.tokenizer_encoder_head_dim  # GenieConfig.attention_head_dim / __post_init__
_.tokenizer_decoder_head_dim  # GenieConfig.attention_head_dim / __post_init__
_.action_encoder_layers  # _STTransformer.__init__ / GenieConfig.__post_init__
_.action_encoder_heads  # _STTransformer.__init__ / GenieConfig.attention_head_dim
_.action_encoder_head_dim  # GenieConfig.attention_head_dim / __post_init__
_.action_decoder_layers  # _STTransformer.__init__ / GenieConfig.__post_init__
_.action_decoder_heads  # _STTransformer.__init__ / GenieConfig.attention_head_dim
_.action_decoder_head_dim  # GenieConfig.attention_head_dim / __post_init__

# ── WeightsEnum members, reached by tag through weights.resolve_weights ──
_.ALIEN  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.AMIDAR  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.ASSAULT  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.ASTERIX  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.BANKHEIST  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.BATTLEZONE  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.BOXING  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.CHOPPERCOMMAND  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.CRAZYCLIMBER  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.DEMONATTACK  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.FREEWAY  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.FROSTBITE  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.GOPHER  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.HERO  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.JAMESBOND  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.KANGAROO  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.KRULL  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.KUNGFUMASTER  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.MSPACMAN  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.PONG  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.PRIVATEEYE  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.QBERT  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.ROADRUNNER  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.SEAQUEST  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.UPNDOWN  # DIAMOND_Weights member in lucid/models/generative/diamond/_weights.py
_.IMAGENET1K_512  # DiT_XL_2_Weights member in lucid/models/generative/dit/_weights.py

# ── Public Tensor methods (probed via inspect in lucid/test/audit) ────────
_.clamp_min_  # Tensor.clamp_min_ in lucid/_tensor/tensor.py
_.clamp_max_  # Tensor.clamp_max_ in lucid/_tensor/tensor.py
_.share_memory_  # Tensor.share_memory_ in lucid/_tensor/tensor.py

# ── Public Module / RNN / pruning / autograd / compile / metal / profiler API ─
_.register_module  # Module.register_module in lucid/nn/module.py
_.share_memory  # Module.share_memory in lucid/nn/module.py
_.register_load_state_dict_post_hook  # Module hook API in lucid/nn/module.py
_.flatten_parameters  # RNNBase.flatten_parameters in lucid/nn/modules/rnn.py
_.random_unstructured  # lucid.nn.utils.prune.random_unstructured public API
_.is_pruned  # lucid.nn.utils.prune.is_pruned public API
_.elapsed_time  # MetalEvent.elapsed_time in lucid/metal/__init__.py
_.time_ms  # OpEvent.time_ms property in lucid/profiler/__init__.py
_.export_chrome_trace  # Profiler.export_chrome_trace in lucid/profiler/__init__.py
_.smallest_normal  # finfo.smallest_normal field in lucid/_dtype.py

# ── Public distributions / models / tokenizer / transforms surface ────────
_.open_unit_interval  # lucid.distributions.constraints.open_unit_interval
_.greater_than  # lucid.distributions.constraints.greater_than
_.less_than  # lucid.distributions.constraints.less_than
_.BackboneProtocol  # model contract protocol in lucid/models/_protocols.py
_.TaskWrapperProtocol  # model contract protocol in lucid/models/_protocols.py
_.OutputDataclassProtocol  # model contract protocol in lucid/models/_protocols.py
_.ConfigT  # model config TypeVar in lucid/models/_protocols.py
_.compute_model_summary  # called by tools/build_model_summaries.py
_.world_model_loss  # DIAMONDModel.world_model_loss in lucid/models/generative/diamond/_model.py
_.infer_actions  # GenieModel.infer_actions in lucid/models/generative/genie/_model.py
_.get_output_embeddings  # TransformerModel.get_output_embeddings in lucid/models/text/transformer/_model.py
_.stage_features  # ConvNeXt.stage_features property in lucid/models/vision/convnext/_model.py
_.decoder_hidden_states  # Seq2SeqLMOutput field in lucid/models/_output.py
_.decoder_attentions  # Seq2SeqLMOutput field in lucid/models/_output.py
_.encoder_attentions  # Seq2SeqLMOutput field in lucid/models/_output.py
_.overshoot_kl_loss  # PlaNetOutput field in lucid/models/generative/planet/_model.py
_.overshoot_reward_loss  # PlaNetOutput field in lucid/models/generative/planet/_model.py
_.class_queries_logits  # MaskFormerOutput field in lucid/models/vision/maskformer/_model.py
_.masks_queries_logits  # MaskFormerOutput field in lucid/models/vision/maskformer/_model.py
_.short_context  # VJEPAPredictorOutput field in lucid/models/vision/vjepa/_model.py
_.short_targets  # VJEPAPredictorOutput field in lucid/models/vision/vjepa/_model.py
_.long_context  # VJEPAPredictorOutput field in lucid/models/vision/vjepa/_model.py
_.long_targets  # VJEPAPredictorOutput field in lucid/models/vision/vjepa/_model.py
_.mask_token_id  # TokenizerBase.mask_token_id property in lucid/utils/tokenizer/_base.py
_.sep_token_id  # TokenizerBase.sep_token_id property in lucid/utils/tokenizer/_base.py
_.label_fields  # BboxParams.label_fields field in lucid/utils/transforms/_base.py

# ── LoraConfig fields the library accepts at their default and refuses otherwise ──
_.megatron_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported (lucid/peft/_config.py)
_.megatron_core  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.trainable_token_indices  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.loftq_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.eva_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.corda_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.lora_ga_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.velora_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.alora_invocation_tokens  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.use_qalora  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.qalora_group_size  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.monteclora_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.layer_replication  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.use_bdlora  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.arrow_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.kasa_config  # LoraConfig field read by name through PeftConfig._refuse_unsupported
_.ensure_weight_tying  # LoraConfig field read by name through PeftConfig._refuse_unsupported
