// lucid/_C/nn/Attention.cpp
//
// Implementation of Scaled Dot-Product Attention forward and backward.
//
// The shared run_forward() helper validates inputs, dispatches to
// IBackend::sdpa_forward, and returns a ForwardCore bundle containing the
// output tensor, the attention weight storage, batch/sequence/dim sizes, and
// the output/weights shapes.  Both public entry points call run_forward and
// then either wire the backward node (forward()) or expose the weights tensor
// as a second return value (scaled_dot_product_attention_with_weights_op()).
//
// FLOP estimate: 2 * B * Lq * Lk * (Dk + Dv) — covers Q@K^T and W@V.

#include "Attention.h"

#include <cmath>
#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/Softmax.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/Tri.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema ScaledDotProductAttentionBackward::schema_v1{"scaled_dot_product_attention", 1,
                                                            AmpPolicy::ForceFP32, true};

namespace {

// Batch size B plus the last two dims (L, D) extracted from a Q/K/V shape.
struct Flat3 {
    std::size_t B;
    std::size_t L;
    std::size_t D;
};

// Flatten all leading dims of s (except last two) into a single batch B.
// Throws if s has fewer than 2 dimensions.
Flat3 flatten_qkv(const Shape& s, const char* name) {
    if (s.size() < 2) {
        ErrorBuilder("attention").fail(std::string(name) + " must be at least 2-D ([..., L, d])");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 2 < s.size(); ++i)
        b *= static_cast<std::size_t>(s[i]);
    return {b, static_cast<std::size_t>(s[s.size() - 2]), static_cast<std::size_t>(s.back())};
}

// Construct the SDPA output shape (..., Lq, Dv) from the Q and V shapes.
Shape build_output_shape(const Shape& q_shape, const Shape& v_shape) {
    Shape out;
    out.reserve(q_shape.size());
    for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
        out.push_back(q_shape[i]);
    out.push_back(q_shape[q_shape.size() - 2]);
    out.push_back(v_shape.back());
    return out;
}

// All data produced by the shared forward kernel, bundled for the two callers.
struct ForwardCore {
    TensorImplPtr output;
    Storage weights_storage;  // Attention weights W; may be {1} on GPU path.
    std::size_t B;
    std::size_t Lq;
    std::size_t Lk;
    std::size_t Dk;
    std::size_t Dv;
    Shape out_shape;
    Shape weights_shape;  // (..., Lq, Lk).
};

// Validate Q/K/V shapes and dtypes, dispatch to IBackend::sdpa_forward, and
// return a ForwardCore.  The backend returns [weights, output] in results[].
ForwardCore run_forward(const TensorImplPtr& q,
                        const TensorImplPtr& k,
                        const TensorImplPtr& v,
                        const TensorImplPtr& attn_mask,
                        double scale,
                        bool is_causal,
                        bool need_weights) {
    if (!q || !k || !v)
        ErrorBuilder("attention").fail("null input");
    // Pairwise against Q, so the error names the operand that differs: one
    // three-way test reported Q and K even when V was the odd one out, and
    // read "expected float32, got float32".
    Validator::pair(q, k, "attention: Q/K device mismatch").same_device();
    Validator::pair(q, v, "attention: Q/V device mismatch").same_device();
    Validator::pair(q, k, "attention: Q/K dtype mismatch").same_dtype();
    Validator::pair(q, v, "attention: Q/V dtype mismatch").same_dtype();
    if (attn_mask) {
        // A bool mask selects; an additive one is summed into the scores, so
        // both kernels read it in q's dtype.  The CPU read a mismatched one
        // as q's bytes and MLX refused it, so it is refused here for both.
        Validator::pair(q, attn_mask, "attention: Q/attn_mask").same_device();
        if (attn_mask->dtype() != Dtype::Bool)
            Validator::pair(q, attn_mask, "attention: Q/attn_mask").same_dtype();
    }
    if (q->shape().size() < 2 || k->shape().size() < 2 || v->shape().size() < 2)
        ErrorBuilder("attention").invalid_argument("Q/K/V must be at least 2-D");

    const auto fq = flatten_qkv(q->shape(), "Q");
    const auto fk = flatten_qkv(k->shape(), "K");
    const auto fv = flatten_qkv(v->shape(), "V");
    if (fq.B != fk.B || fq.B != fv.B)
        throw ShapeMismatch(q->shape(), k->shape(),
                            "attention: leading dims of Q/K/V must be equal");
    if (fq.D != fk.D)
        throw ShapeMismatch(q->shape(), k->shape(), "attention: Q.last_dim must equal K.last_dim");
    if (fk.L != fv.L)
        throw ShapeMismatch(k->shape(), v->shape(), "attention: K.L_k must equal V.L_k");
    // ``is_causal`` has one meaning only where every backend agrees on it: a
    // square score matrix and no other mask.  With a mask the Accelerate
    // kernel applied both while the MLX one let the mask win; with
    // ``L_q != L_k`` the first aligned the triangle top-left and the second
    // bottom-right — the same call answered differently per device.
    // ``F.scaled_dot_product_attention`` folds causality into the additive
    // mask in those cases, so only a direct engine caller reaches this.
    if (is_causal && (attn_mask || fq.L != fk.L))
        ErrorBuilder("attention")
            .fail("is_causal needs a square score matrix and no attn_mask; fold the causal "
                  "triangle into attn_mask instead (F.scaled_dot_product_attention does)");

    OpScopeFull scope{ScaledDotProductAttentionBackward::schema_v1.name, q->device(), q->dtype(),
                      build_output_shape(q->shape(), v->shape())};
    scope.set_attr("scale", scale);
    scope.set_attr("is_causal", is_causal);
    scope.set_attr("has_mask", attn_mask != nullptr);

    Shape out_shape = build_output_shape(q->shape(), v->shape());
    Shape weights_shape;
    weights_shape.reserve(q->shape().size());
    for (std::size_t i = 0; i + 2 < q->shape().size(); ++i)
        weights_shape.push_back(q->shape()[i]);
    weights_shape.push_back(q->shape()[q->shape().size() - 2]);
    weights_shape.push_back(k->shape()[k->shape().size() - 2]);

    const Storage* mask_storage = attn_mask ? &attn_mask->storage() : nullptr;
    // sdpa_forward returns {weights, output}.  On the GPU path weights may be a
    // dummy {1}-shape tensor; the backward detects this and recomputes W.
    auto results =
        backend::Dispatcher::for_device(q->device())
            .sdpa_forward(q->storage(), k->storage(), v->storage(), mask_storage, q->shape(),
                          k->shape(), v->shape(), attn_mask ? attn_mask->dtype() : Dtype::F32,
                          attn_mask ? static_cast<std::size_t>(attn_mask->numel()) : std::size_t{0},
                          scale, is_causal, need_weights, q->dtype());

    // results[0] = weights storage; results[1] = output storage.
    auto out = std::make_shared<TensorImpl>(std::move(results[1]), out_shape, q->dtype(),
                                            q->device(), false);

    scope.set_flops(static_cast<std::int64_t>(2) * static_cast<std::int64_t>(fq.B) *
                    static_cast<std::int64_t>(fq.L) * static_cast<std::int64_t>(fk.L) *
                    static_cast<std::int64_t>(fq.D + fv.D));

    // Caller wire_autograd records on_op_io internally — no explicit call.

    return ForwardCore{std::move(out),       std::move(results[0]),   fq.B, fq.L, fk.L, fq.D, fv.D,
                       std::move(out_shape), std::move(weights_shape)};
}

}  // namespace

TensorImplPtr ScaledDotProductAttentionBackward::forward(const TensorImplPtr& q_in,
                                                         const TensorImplPtr& k_in,
                                                         const TensorImplPtr& v_in,
                                                         const TensorImplPtr& mask_in,
                                                         double scale,
                                                         bool is_causal) {
    // The row (LOWER on both devices) casts Q, K, V and an additive mask
    // together; a Bool mask is not a float and stays.  Before it, a float32
    // Q from RoPE met a float16 V under Metal autocast and was refused.
    const AmpScope amp{schema_v1.name, q_in ? q_in->device() : Device::CPU};
    const auto operands = amp.cast_widest({q_in, k_in, v_in, mask_in});
    const TensorImplPtr &q = operands[0], &k = operands[1], &v = operands[2];
    const TensorImplPtr& attn_mask = operands[3];
    // Output-only op: use the memory-efficient fused path (no dense weights).
    auto core = run_forward(q, k, v, attn_mask, scale, is_causal, /*need_weights=*/false);

    auto bwd = std::make_shared<ScaledDotProductAttentionBackward>();
    bwd->saved_weights_ = std::move(core.weights_storage);
    bwd->scale_ = scale;
    bwd->is_causal_ = is_causal;
    bwd->orig_q_shape_ = q->shape();
    bwd->orig_k_shape_ = k->shape();
    bwd->orig_v_shape_ = v->shape();
    if (attn_mask) {
        // Persist the mask so the GPU VJP backward can replay the exact masked
        // attention.  Copying the Storage retains the underlying buffer
        // independently of the mask tensor.
        bwd->has_mask_ = true;
        bwd->saved_mask_ = attn_mask->storage();
        bwd->mask_dtype_ = attn_mask->dtype();
        bwd->orig_mask_shape_ = attn_mask->shape();
    }

    // An *additive* (float) mask is a differentiable input when the caller
    // actually wants its gradient: relative-position bias tables are learned
    // parameters that reach the loss only through this argument.  A Bool
    // keep-mask selects rather than adds, and a constant padding or causal
    // mask needs no gradient — both get a null edge, which keeps the node
    // (and the compile trace) exactly as it was for those callers.
    const bool mask_is_differentiable =
        attn_mask && attn_mask->dtype() != Dtype::Bool && attn_mask->requires_grad();
    const TensorImplPtr& mask_edge = mask_is_differentiable ? attn_mask : TensorImplPtr{};
    bwd->mask_differentiable_ = mask_is_differentiable;

    kernel::NaryKernel<ScaledDotProductAttentionBackward, 4>::wire_autograd(
        std::move(bwd), {q, k, v, mask_edge}, core.output);

    // ``wire_autograd`` records its raw input array — including the null 4th
    // slot when there is no mask, or a Bool one that carries no gradient.  The
    // trace must not see that null, and a Bool mask must still appear so the
    // sdpa emitter can fold it in rather than bailing to eager.  Re-record the
    // exact operand list (``Tracer::on_op_io`` is last-write-wins), mirroring
    // the embedding op's handling of its non-diff ``indices`` input.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        if (attn_mask) {
            trc->on_op_io({q, k, v, attn_mask}, core.output);
        } else {
            trc->on_op_io({q, k, v}, core.output);
        }
    }
    return core.output;
}

std::vector<Storage> ScaledDotProductAttentionBackward::apply(Storage grad_out) {
    const Storage* mask_ptr = has_mask_ ? &saved_mask_ : nullptr;
    auto grads = backend::Dispatcher::for_device(device_).sdpa_backward(
        grad_out, saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], saved_weights_, mask_ptr,
        orig_q_shape_, orig_k_shape_, orig_v_shape_, mask_dtype_, scale_, is_causal_, dtype_);

    // The backends hand back dM at the *full* score shape (…, L_q, L_k); the
    // caller's mask may have broadcast into it, so sum the expanded axes away.
    // A Bool keep-mask has a null edge and a placeholder slot — leave it be.
    if (grads.size() >= 4 && mask_differentiable_ && !orig_mask_shape_.empty()) {
        Shape score_shape;
        score_shape.reserve(orig_q_shape_.size());
        for (std::size_t i = 0; i + 2 < orig_q_shape_.size(); ++i)
            score_shape.push_back(orig_q_shape_[i]);
        score_shape.push_back(orig_q_shape_[orig_q_shape_.size() - 2]);
        score_shape.push_back(orig_k_shape_[orig_k_shape_.size() - 2]);

        if (score_shape != orig_mask_shape_) {
            grads[3] =
                reduce_grad_to_shape(grads[3], score_shape, orig_mask_shape_, dtype_, device_);
        }
    }
    return grads;
}

std::vector<TensorImplPtr>
ScaledDotProductAttentionBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& q = saved_impl_inputs_[0];
    const auto& k = saved_impl_inputs_[1];
    const auto& v = saved_impl_inputs_[2];
    if (!q || !k || !v || !grad_out)
        ErrorBuilder("scaled_dot_product_attention")
            .fail("graph-mode backward is missing a saved input");

    const int last = static_cast<int>(orig_q_shape_.size()) - 1;
    auto scaled = [this](const TensorImplPtr& t) { return mul_op(t, full_like_op(t, scale_)); };

    // Scores exactly as the forward formed them.  The mask is the storage
    // the kernel read, or the live input when its gradient was asked for.
    auto scores = scaled(matmul_op(q, mT_op(k)));
    if (has_mask_) {
        TensorImplPtr mask = mask_differentiable_ ? saved_impl_inputs_[3] : TensorImplPtr{};
        if (!mask)
            mask = std::make_shared<TensorImpl>(saved_mask_, orig_mask_shape_, mask_dtype_, device_,
                                                false);
        if (mask_dtype_ == Dtype::Bool)
            scores = where_op(mask, scores, full_like_op(scores, -INFINITY));
        else
            scores = add_op(scores, mask);
    }
    if (is_causal_) {
        // Square and mask-free (``run_forward`` refuses anything else), so
        // the lower triangle is the whole rule.
        const std::int64_t lq = orig_q_shape_[orig_q_shape_.size() - 2];
        const std::int64_t lk = orig_k_shape_[orig_k_shape_.size() - 2];
        auto keep = tril_op(ones_op(Shape{lq, lk}, scores->dtype(), device_), 0);
        scores = where_op(greater_op(keep, zeros_like_op(keep)), scores,
                          full_like_op(scores, -INFINITY));
    }
    auto p = softmax_op(scores, last);

    // dV = Pᵀ g;  dP = g Vᵀ;  dS = P ∘ (dP − Σ dP∘P);  dQ = s·dS K;  dK = s·dSᵀ Q.
    auto dv = matmul_op(mT_op(p), grad_out);
    auto dp = matmul_op(grad_out, mT_op(v));
    auto ds = mul_op(p, sub_op(dp, sum_op(mul_op(dp, p), std::vector<int>{last}, true)));
    auto dq = scaled(matmul_op(ds, k));
    auto dk = scaled(matmul_op(mT_op(ds), q));

    // S = s·QKᵀ + M, so dM is dS summed over whatever the mask broadcast along.
    TensorImplPtr dm;
    if (mask_differentiable_) {
        dm = ds;
        const Shape& full = ds->shape();
        if (full != orig_mask_shape_) {
            const std::size_t lead = full.size() - orig_mask_shape_.size();
            std::vector<int> axes;
            for (std::size_t i = 0; i < full.size(); ++i)
                if (i < lead || (orig_mask_shape_[i - lead] == 1 && full[i] != 1))
                    axes.push_back(static_cast<int>(i));
            dm = reshape_op(
                sum_op(ds, axes, true),
                std::vector<std::int64_t>(orig_mask_shape_.begin(), orig_mask_shape_.end()));
        }
    }
    return {dq, dk, dv, dm};
}

TensorImplPtr scaled_dot_product_attention_op(const TensorImplPtr& q,
                                              const TensorImplPtr& k,
                                              const TensorImplPtr& v,
                                              const TensorImplPtr& attn_mask_or_null,
                                              double scale,
                                              bool is_causal) {
    return ScaledDotProductAttentionBackward::forward(q, k, v, attn_mask_or_null, scale, is_causal);
}

std::vector<TensorImplPtr>
scaled_dot_product_attention_with_weights_op(const TensorImplPtr& q_in,
                                             const TensorImplPtr& k_in,
                                             const TensorImplPtr& v_in,
                                             const TensorImplPtr& mask_in,
                                             double scale,
                                             bool is_causal) {
    // The row, as in ScaledDotProductAttentionBackward::forward.
    const AmpScope amp{ScaledDotProductAttentionBackward::schema_v1.name,
                       q_in ? q_in->device() : Device::CPU};
    const auto operands = amp.cast_widest({q_in, k_in, v_in, mask_in});
    const TensorImplPtr &q = operands[0], &k = operands[1], &v = operands[2];
    const TensorImplPtr& attn_mask_or_null = operands[3];
    // Weights variant: materialize the dense softmax weight matrix for the caller.
    auto core = run_forward(q, k, v, attn_mask_or_null, scale, is_causal, /*need_weights=*/true);

    Shape weights_shape = core.weights_shape;
    auto weights = std::make_shared<TensorImpl>(
        std::move(core.weights_storage), std::move(weights_shape), q->dtype(), q->device(), false);

    if (GradMode::is_enabled() &&
        (q->requires_grad() || k->requires_grad() || v->requires_grad())) {
        auto with_grad = ScaledDotProductAttentionBackward::forward(q, k, v, attn_mask_or_null,
                                                                    scale, is_causal);
        return {std::move(with_grad), std::move(weights)};
    }
    return {std::move(core.output), std::move(weights)};
}

LUCID_REGISTER_OP(ScaledDotProductAttentionBackward)

}  // namespace lucid
