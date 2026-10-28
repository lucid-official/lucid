// lucid/_C/compile/VjpEmitters/linalg/Linear.mm
//
// VJPs for ``linear`` (nn.Linear forward) and ``matmul``.
//
// Linear forward: y = x @ W^T + b
//   - x : (..., in_features)
//   - W : (out_features, in_features)
//   - b : (out_features,)            optional
//   - y : (..., out_features)
//
// Backward:
//   dx = grad @ W                                       (shape (..., in_features))
//   dW = reshape(grad, (-1, out)) ^T @ reshape(x, (-1, in))
//                                                       (shape (out, in))
//   db = sum(grad, all_leading_axes)                    (shape (out,))
//
// Matmul (a @ b, both 2-D or batched-N-D): the canonical formula
//   dA = grad @ b^T,  dB = a^T @ grad.  For batched inputs (rank ≥ 3),
// we let MPSGraph's matmul handle the leading batch broadcasting.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <vector>

#include "../../OpEmitters/_SignedZero.h"
#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// ────────────────────────────────────────────────────────────────────
// Linear
// ────────────────────────────────────────────────────────────────────
class LinearVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "linear"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() < 2 || node.inputs.size() > 3 ||
            grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        TensorId W_id = node.inputs[1];
        TensorId b_id = (node.inputs.size() == 3)
            ? node.inputs[2]
            : TraceId::external_feed();
        if (x_id < 0 || W_id < 0) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        MPSGraphTensor* W = as_tensor(bctx.forward(W_id));
        if (g == nil || grad == nil || x == nil || W == nil) return false;

        // Mixed-dtype reconciliation (autocast): MPSGraph's matmul
        // requires matching operand dtypes.  Cast forward x / W to
        // grad's dtype so dx = grad @ W and dW = grad^T @ x both
        // execute in one precision.
        const MPSDataType chain_dt = grad.dataType;
        x = cast_if_needed(g, x, chain_dt);
        W = cast_if_needed(g, W, chain_dt);

        std::vector<std::int64_t> x_shape = shape_of_mps(x);
        std::vector<std::int64_t> W_shape = shape_of_mps(W);  // (out, in)
        if (x_shape.empty() || W_shape.size() != 2) return false;
        const std::int64_t out_feat = W_shape[0];
        const std::int64_t in_feat = W_shape[1];

        // dx = grad @ W   (W is (out, in), no transpose needed since
        //                  grad @ W gives shape (..., in_features).)
        MPSGraphTensor* dx =
            [g matrixMultiplicationWithPrimaryTensor:grad
                                      secondaryTensor:W
                                                 name:@"linear_vjp_dx"];
        bctx.accumulate_grad(x_id, from_tensor(dx));

        // dW: collapse leading axes of grad / x to a single batch dim,
        // then dW = grad_2d^T @ x_2d.
        NSArray<NSNumber*>* g_flat_shape =
            @[ @(-1), [NSNumber numberWithLongLong:out_feat] ];
        NSArray<NSNumber*>* x_flat_shape =
            @[ @(-1), [NSNumber numberWithLongLong:in_feat] ];
        MPSGraphTensor* g_flat =
            [g reshapeTensor:grad withShape:g_flat_shape name:@"linear_vjp_g_flat"];
        MPSGraphTensor* x_flat =
            [g reshapeTensor:x withShape:x_flat_shape name:@"linear_vjp_x_flat"];
        MPSGraphTensor* g_flat_T =
            [g transposeTensor:g_flat permutation:@[ @1, @0 ] name:nil];
        MPSGraphTensor* dW =
            [g matrixMultiplicationWithPrimaryTensor:g_flat_T
                                      secondaryTensor:x_flat
                                                 name:@"linear_vjp_dW"];
        bctx.accumulate_grad(W_id, from_tensor(dW));

        // db = sum(grad over all leading axes) → (out,).
        if (b_id >= 0) {
            // Reduce all axes except the last.
            const std::size_t rank = x_shape.size();
            // grad has shape (...x_shape[:-1], out_feat), so rank-1 axes
            // need reducing.
            NSMutableArray<NSNumber*>* axes =
                [NSMutableArray arrayWithCapacity:rank - 1];
            for (std::size_t i = 0; i + 1 < rank; ++i)
                [axes addObject:[NSNumber numberWithLongLong:(long long)i]];
            MPSGraphTensor* db_keep = grad;
            if (axes.count > 0) {
                db_keep = [g reductionSumWithTensor:grad
                                                axes:axes
                                                name:@"linear_vjp_db_sum"];
            }
            // Reshape to bare (out,).
            NSArray<NSNumber*>* db_shape =
                @[ [NSNumber numberWithLongLong:out_feat] ];
            MPSGraphTensor* db =
                [g reshapeTensor:db_keep withShape:db_shape name:@"linear_vjp_db"];
            bctx.accumulate_grad(b_id, from_tensor(db));
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// Matmul (2-D case primary; batched defers to MPSGraph's broadcasting).
// ────────────────────────────────────────────────────────────────────
class MatmulVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "matmul"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 2 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        MPSGraphTensor* a = as_tensor(bctx.forward(a_id));
        MPSGraphTensor* b = as_tensor(bctx.forward(b_id));
        if (g == nil || grad == nil || a == nil || b == nil) return false;

        // Mixed-dtype reconciliation (autocast).
        const MPSDataType chain_dt = grad.dataType;
        a = cast_if_needed(g, a, chain_dt);
        b = cast_if_needed(g, b, chain_dt);

        std::vector<std::int64_t> a_shape = shape_of_mps(a);
        std::vector<std::int64_t> b_shape = shape_of_mps(b);
        if (a_shape.size() < 2 || b_shape.size() < 2) return false;

        // dA = grad @ b^T (last two dims swapped on b).
        // dB = a^T @ grad (last two dims swapped on a).
        const std::size_t b_rank = b_shape.size();
        const std::size_t a_rank = a_shape.size();
        NSMutableArray<NSNumber*>* b_perm =
            [NSMutableArray arrayWithCapacity:b_rank];
        for (std::size_t i = 0; i + 2 < b_rank; ++i)
            [b_perm addObject:[NSNumber numberWithLongLong:(long long)i]];
        [b_perm addObject:[NSNumber numberWithLongLong:(long long)(b_rank - 1)]];
        [b_perm addObject:[NSNumber numberWithLongLong:(long long)(b_rank - 2)]];
        MPSGraphTensor* b_T = [g transposeTensor:b permutation:b_perm name:nil];
        b_T = materialize_matmul_splat(g, b_T, is_const_splat(bctx.forward_ctx(), b_id));

        NSMutableArray<NSNumber*>* a_perm =
            [NSMutableArray arrayWithCapacity:a_rank];
        for (std::size_t i = 0; i + 2 < a_rank; ++i)
            [a_perm addObject:[NSNumber numberWithLongLong:(long long)i]];
        [a_perm addObject:[NSNumber numberWithLongLong:(long long)(a_rank - 1)]];
        [a_perm addObject:[NSNumber numberWithLongLong:(long long)(a_rank - 2)]];
        MPSGraphTensor* a_T = [g transposeTensor:a permutation:a_perm name:nil];
        a_T = materialize_matmul_splat(g, a_T, is_const_splat(bctx.forward_ctx(), a_id));

        MPSGraphTensor* dA =
            [g matrixMultiplicationWithPrimaryTensor:grad
                                      secondaryTensor:b_T
                                                 name:@"matmul_vjp_dA"];
        MPSGraphTensor* dB =
            [g matrixMultiplicationWithPrimaryTensor:a_T
                                      secondaryTensor:grad
                                                 name:@"matmul_vjp_dB"];

        // For batched matmul where one operand has fewer leading dims,
        // MPSGraph broadcasts forward; the backward grad has the
        // broadcast shape, and we need to unreduce back to each input's
        // shape.  For 2-D case this is a no-op.
        std::vector<std::int64_t> dA_shape = shape_of_mps(dA);
        std::vector<std::int64_t> dB_shape = shape_of_mps(dB);
        void* dA_red = bctx.unreduce(from_tensor(dA), a_shape, dA_shape);
        void* dB_red = bctx.unreduce(from_tensor(dB), b_shape, dB_shape);

        bctx.accumulate_grad(a_id, dA_red);
        bctx.accumulate_grad(b_id, dB_red);
        return true;
    }
};

struct LinearVjpRegistrar {
    LinearVjpRegistrar() {
        register_vjp_emitter(std::make_unique<LinearVjp>());
        register_vjp_emitter(std::make_unique<MatmulVjp>());
    }
};

[[maybe_unused]] static const LinearVjpRegistrar g_linear_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
