// lucid/_C/compile/VjpEmitters/elementwise/Arith.mm
//
// VJPs for the binary-elementwise family.  Mirrors the eager
// implementations in :file:`lucid/_C/ops/bfunc/{Add,Sub,Mul,Div,Pow}.cpp`,
// translated to MPSGraph subgraph emission.
//
// Standard recipe:
//   1. Pull ``grad_out = grad_outs[0]`` (single-output ops).
//   2. Look up forward activations a / b via :func:`BackwardContext::forward`.
//   3. Emit the gradient subgraph.
//   4. Call :func:`BackwardContext::unreduce` to broadcast-back to each
//      input's shape, then :func:`accumulate_grad`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// Pair of full-shape gradients (before unreduce) for a binary VJP.
struct BinaryGradPair {
    MPSGraphTensor* da = nil;
    MPSGraphTensor* db = nil;
};

template <class Body>
inline bool accumulate_binary(BackwardContext& bctx,
                              const OpNode& node,
                              const std::vector<void*>& grad_outs,
                              Body body) {
    BinaryVjpCtx c = unpack_binary_vjp(bctx, node, grad_outs);
    if (!c.ok)
        return false;
    BinaryGradPair g = body(c);
    if (g.da == nil || g.db == nil)
        return false;
    MPSGraphTensor* da = as_tensor(bctx.unreduce(from_tensor(g.da), c.a_shape, c.out_shape));
    MPSGraphTensor* db = as_tensor(bctx.unreduce(from_tensor(g.db), c.b_shape, c.out_shape));
    if (da == nil || db == nil)
        return false;  // the grad cannot be reduced to an input's shape
    bctx.accumulate_grad(c.a_id, from_tensor(da));
    bctx.accumulate_grad(c.b_id, from_tensor(db));
    return true;
}

// ────────────────────────────────────────────────────────────────────
// add: dA = dB = grad_out (unreduce to each input shape).
// ────────────────────────────────────────────────────────────────────
class AddVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "add"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair { return {c.go, c.go}; });
    }
};

// ────────────────────────────────────────────────────────────────────
// sub: dA = grad_out, dB = -grad_out (unreduce).
// ────────────────────────────────────────────────────────────────────
class SubVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sub"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
                                 [](const BinaryVjpCtx& c) -> BinaryGradPair {
                                     return {c.go, [c.g negativeWithTensor:c.go name:nil]};
                                 });
    }
};

// ────────────────────────────────────────────────────────────────────
// nextafter: dA = grad, dB = 0.  One ulp from ``a`` follows ``a`` and does
// not move with ``b``.  Mirrors ``NextafterBackward``.
// ────────────────────────────────────────────────────────────────────
class NextafterVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "nextafter"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs, [](const BinaryVjpCtx& c) -> BinaryGradPair {
                MPSGraphTensor* zero = [c.g constantWithScalar:0.0 dataType:c.go.dataType];
                return {c.go, [c.g multiplicationWithPrimaryTensor:c.go
                                                   secondaryTensor:zero
                                                              name:nil]};
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// mul: dA = grad * b, dB = grad * a (unreduce).  Product rule.
// ────────────────────────────────────────────────────────────────────
class MulVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mul"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs, [](const BinaryVjpCtx& c) -> BinaryGradPair {
                return {[c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.b name:nil],
                        [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.a name:nil]};
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// div: dA = grad / b, dB = -(grad * (a / b)) / b (unreduce).  Quotient rule.
//
// dB is the textbook -(grad * a) / b² with the square never formed, as eager's
// ``DivBackward`` (LCD-998): b² leaves float32 for a |b| outside ~1e-19 .. 1e19, so the
// textbook form returns 0 or inf there for a gradient that is an ordinary float whenever
// grad and a are of b's scale.  Dividing by b twice never forms a square of b, so the
// range is that of ``grad * (a / b)`` rather than of b², and ``a / b`` is 0 for an a that
// is 0, which ``grad * a / b²`` turns into 0 / 0 once b² underflows.
// ────────────────────────────────────────────────────────────────────
class DivVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "div"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs, [](const BinaryVjpCtx& c) -> BinaryGradPair {
                MPSGraphTensor* da = [c.g divisionWithPrimaryTensor:c.go
                                                    secondaryTensor:c.b
                                                               name:nil];
                MPSGraphTensor* quotient = [c.g divisionWithPrimaryTensor:c.a
                                                          secondaryTensor:c.b
                                                                     name:nil];
                MPSGraphTensor* g_times_q = [c.g multiplicationWithPrimaryTensor:c.go
                                                                 secondaryTensor:quotient
                                                                            name:nil];
                MPSGraphTensor* div_by_b = [c.g divisionWithPrimaryTensor:g_times_q
                                                          secondaryTensor:c.b
                                                                     name:nil];
                MPSGraphTensor* db = [c.g negativeWithTensor:div_by_b name:nil];
                return {da, db};
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// pow: dA = b * a^(b-1) * grad, dB = log(a) * a^b * grad (unreduce).
// ────────────────────────────────────────────────────────────────────
class PowVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "pow"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs, [](const BinaryVjpCtx& c) -> BinaryGradPair {
                // c.a / c.b already cast to chain dtype by accumulate_binary.
                MPSDataType dt = c.go.dataType;
                MPSGraphTensor* zero = [c.g constantWithScalar:0.0 dataType:dt];
                MPSGraphTensor* one = [c.g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* bm1 = [c.g subtractionWithPrimaryTensor:c.b
                                                        secondaryTensor:one
                                                                   name:nil];
                MPSGraphTensor* a_pow_bm1 = [c.g powerWithPrimaryTensor:c.a
                                                        secondaryTensor:bm1
                                                                   name:nil];
                MPSGraphTensor* b_times_apow = [c.g multiplicationWithPrimaryTensor:c.b
                                                                    secondaryTensor:a_pow_bm1
                                                                               name:nil];
                MPSGraphTensor* b_eq_zero = [c.g equalWithPrimaryTensor:c.b
                                                        secondaryTensor:zero
                                                                   name:nil];
                b_times_apow = [c.g selectWithPredicateTensor:b_eq_zero
                                          truePredicateTensor:zero
                                         falsePredicateTensor:b_times_apow
                                                         name:nil];
                MPSGraphTensor* da = [c.g multiplicationWithPrimaryTensor:b_times_apow
                                                          secondaryTensor:c.go
                                                                     name:nil];

                MPSGraphTensor* log_a = [c.g logarithmWithTensor:c.a name:nil];
                MPSGraphTensor* a_pow_b = [c.g powerWithPrimaryTensor:c.a
                                                      secondaryTensor:c.b
                                                                 name:nil];
                MPSGraphTensor* prod = [c.g multiplicationWithPrimaryTensor:log_a
                                                            secondaryTensor:a_pow_b
                                                                       name:nil];
                MPSGraphTensor* a_eq_zero = [c.g equalWithPrimaryTensor:c.a
                                                        secondaryTensor:zero
                                                                   name:nil];
                MPSGraphTensor* b_ge_zero = [c.g greaterThanOrEqualToWithPrimaryTensor:c.b
                                                                       secondaryTensor:zero
                                                                                  name:nil];
                MPSGraphTensor* mask_db = [c.g logicalANDWithPrimaryTensor:a_eq_zero
                                                           secondaryTensor:b_ge_zero
                                                                      name:nil];
                prod = [c.g selectWithPredicateTensor:mask_db
                                  truePredicateTensor:zero
                                 falsePredicateTensor:prod
                                                 name:nil];
                MPSGraphTensor* db = [c.g multiplicationWithPrimaryTensor:prod
                                                          secondaryTensor:c.go
                                                                     name:nil];
                return {da, db};
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// maximum / minimum: the winner takes the whole gradient and the loser none; a
// tie splits it in half and an unordered pair (a NaN on either side) gives both
// operands the whole of it — eager's rule (ops/bfunc/_Extremum.h), read off the
// reference's kernels.  An operand's weight is 0 where it lost the comparison
// (``p < q``), ½ where the two are equal (-0 and +0, inf and inf included) and
// 1 otherwise, NaN included, since a NaN fails both tests (LCD-874).
// CLIP clamps its learnt logit scale with ``minimum``; without this its
// training step ran eager.
// ────────────────────────────────────────────────────────────────────
template <bool IS_MAX>
class MinMaxVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return IS_MAX ? "maximum" : "minimum"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        return accumulate_binary(
            bctx, node, grad_outs, [](const BinaryVjpCtx& c) -> BinaryGradPair {
                MPSGraphTensor* lhs = IS_MAX ? c.a : c.b;
                MPSGraphTensor* rhs = IS_MAX ? c.b : c.a;
                const MPSDataType dt = c.go.dataType;
                MPSGraphTensor* zero = [c.g constantWithScalar:0.0 dataType:dt];
                MPSGraphTensor* one = [c.g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* half = [c.g constantWithScalar:0.5 dataType:dt];
                MPSGraphTensor* tie = [c.g equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
                MPSGraphTensor* unless_lost = [c.g selectWithPredicateTensor:tie
                                                         truePredicateTensor:half
                                                        falsePredicateTensor:one
                                                                        name:nil];
                auto share = [&](MPSGraphTensor* p, MPSGraphTensor* q) {
                    MPSGraphTensor* lost = [c.g lessThanWithPrimaryTensor:p secondaryTensor:q name:nil];
                    MPSGraphTensor* w = [c.g selectWithPredicateTensor:lost
                                                   truePredicateTensor:zero
                                                  falsePredicateTensor:unless_lost
                                                                  name:nil];
                    return [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:w name:nil];
                };
                return {share(lhs, rhs), share(rhs, lhs)};
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// neg (unary): dA = -grad.
// ────────────────────────────────────────────────────────────────────
class NegVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "neg"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 1 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* neg = [g negativeWithTensor:go name:nil];
        bctx.accumulate_grad(x_id, from_tensor(neg));
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// where(cond, a, b): da = select(cond, g, 0), db = select(cond, 0, g), each
// unreduced to its input's shape; cond takes no gradient.  masked_fill(x,
// mask) is the same with the filled branch a constant.  MPSGraph's autodiff
// aborts on both — the predicate is a comparison — so without these every
// attention mask, and every activation composed with ``where``, kept a
// model from training compiled.
// ────────────────────────────────────────────────────────────────────
inline MPSGraphTensor* zeros_like_t(MPSGraph* g, MPSGraphTensor* t) {
    return [g constantWithScalar:0.0 dataType:t.dataType];
}

class WhereVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "where"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 3 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* c = as_tensor(bctx.forward(node.inputs[0]));
        if (g == nil || go == nil || c == nil)
            return false;
        const auto out_shape = shape_of_mps(go);
        MPSGraphTensor* zero = zeros_like_t(g, go);
        for (int side = 1; side <= 2; ++side) {
            const TensorId id = node.inputs[static_cast<std::size_t>(side)];
            if (id < 0)
                continue;
            MPSGraphTensor* fwd = as_tensor(bctx.forward(id));
            if (fwd == nil)
                return false;
            MPSGraphTensor* d = side == 1 ? [g selectWithPredicateTensor:c
                                                     truePredicateTensor:go
                                                    falsePredicateTensor:zero
                                                                    name:nil]
                                          : [g selectWithPredicateTensor:c
                                                     truePredicateTensor:zero
                                                    falsePredicateTensor:go
                                                                    name:nil];
            void* red = bctx.unreduce(from_tensor(d), shape_of_mps(fwd), out_shape);
            if (red == nullptr)
                return false;
            bctx.accumulate_grad(id, red);
        }
        return true;
    }
};

class MaskedFillVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "masked_fill"; }
    bool
    emit(BackwardContext& bctx, const OpNode& node, const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 2 || grad_outs.empty() || grad_outs[0] == nullptr ||
            node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* mask = as_tensor(bctx.forward(node.inputs[1]));
        if (g == nil || go == nil || mask == nil)
            return false;
        MPSGraphTensor* d = [g selectWithPredicateTensor:mask
                                     truePredicateTensor:zeros_like_t(g, go)
                                    falsePredicateTensor:go
                                                    name:@"masked_fill_vjp"];
        bctx.accumulate_grad(node.inputs[0], from_tensor(d));
        return true;
    }
};

struct ArithVjpRegistrar {
    ArithVjpRegistrar() {
        register_vjp_emitter(std::make_unique<AddVjp>());
        register_vjp_emitter(std::make_unique<WhereVjp>());
        register_vjp_emitter(std::make_unique<MaskedFillVjp>());
        register_vjp_emitter(std::make_unique<SubVjp>());
        register_vjp_emitter(std::make_unique<NextafterVjp>());
        register_vjp_emitter(std::make_unique<MulVjp>());
        register_vjp_emitter(std::make_unique<DivVjp>());
        register_vjp_emitter(std::make_unique<PowVjp>());
        register_vjp_emitter(std::make_unique<NegVjp>());
        register_vjp_emitter(std::make_unique<MinMaxVjp<true>>());
        register_vjp_emitter(std::make_unique<MinMaxVjp<false>>());
    }
};

[[maybe_unused]] static const ArithVjpRegistrar g_arith_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
