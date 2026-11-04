// lucid/_C/ops/quantized/QuantizedMatmul.h
//
// GPU-stream low-precision (int4 / int8) matmul primitives, thin wrappers over
// MLX's group-wise affine quantization kernels.
//
// GPU stream only (H3): all three ops require Device::GPU inputs and run on the
// default MLX stream.  The packed weight returned by ``quantize_op`` is MLX's
// uint32-packed layout; Lucid has no uint32 dtype, so it is tagged ``I32``
// (same 32-bit width) and treated as an opaque buffer — only ever handed back
// to ``dequantize_op`` / ``quantized_matmul_op``.
//
// ``QuantizedMatmulBackward`` propagates gradients with respect to the floating
// activation ``x`` (``dX = dY @ dequantize(w)`` when ``transpose=true``) while
// leaving the frozen packed weight and per-group scales / biases untouched,
// enabling QLoRA fine-tuning through quantized base linear layers.

#pragma once

#include <optional>
#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for :func:`quantized_matmul_op` with respect to the input
// activation ``x``.  The packed integer weight ``w`` and its per-group
// ``scales`` and ``biases`` are frozen quantization buffers (never trainable)
// and are held by value so backward can dequantize on the fly on Metal.
class LUCID_API QuantizedMatmulBackward : public FuncOp<QuantizedMatmulBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Storage saved_w_;
    Storage saved_scales_;
    std::optional<Storage> saved_biases_;
    Shape w_shape_;
    Shape scales_shape_;
    Shape biases_shape_;
    bool transpose_ = true;
    int group_size_ = 64;
    int bits_ = 8;

    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
    void release_own_saved() { clear_saved(saved_w_, saved_scales_, saved_biases_); }
};

// Quantize a float weight to MLX group-wise affine.  Returns
// ``{packed_weight (I32-tagged uint32), scales (F32), biases (F32)}``.
LUCID_API std::vector<TensorImplPtr> quantize_op(const TensorImplPtr& w, int group_size, int bits);

// Reconstruct a float weight from its packed form (inverse of quantize_op).
LUCID_API TensorImplPtr dequantize_op(const TensorImplPtr& w,
                                      const TensorImplPtr& scales,
                                      const TensorImplPtr& biases,
                                      int group_size,
                                      int bits);

// ``x @ w`` where ``w`` is packed (from quantize_op).  ``transpose`` matches
// MLX semantics (weight stored as ``(out, in)`` → ``transpose=true``).
LUCID_API TensorImplPtr quantized_matmul_op(const TensorImplPtr& x,
                                            const TensorImplPtr& w,
                                            const TensorImplPtr& scales,
                                            const TensorImplPtr& biases,
                                            bool transpose,
                                            int group_size,
                                            int bits);

}  // namespace lucid
