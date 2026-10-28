// lucid/_C/compile/OpEmitters/elementwise/Arith.mm
//
// Two-tensor element-wise arithmetic emitters: add / sub / mul / div /
// pow / maximum / minimum / floordiv / nextafter + scalar-form
// pow_scalar / rpow_scalar.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/bfunc/{Add,Sub,Mul,Div,Pow,...}.cpp
//
// MPSGraph's primary/secondary binary builders broadcast natively, so
// the emitters don't need to thread broadcast-shape logic — the
// builder ensures the input MPSGraphTensors carry the correct shapes
// from their producing ops (Lucid's broadcast happens before
// dispatch, so each side has already been reshaped/copied as needed).
//
// ``matmul`` used to live in this file because the engine ships it
// under ``bfunc/``, but conceptually it's a matrix-algebra primitive
// — it now lives in ``linalg/Matmul.mm`` alongside Linear / Inner /
// Outer / Tensordot.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string_view>
#include <variant>

#include "../../../core/Half.h"
#include "../../MpsDtype.h"
#include "../OpEmitter.h"
#include "../_SignedZero.h"

namespace lucid::compile {

namespace {

inline bool is_0d_f32_tensor(MPSGraphTensor* t) {
    return t.dataType == MPSDataTypeFloat32 && t.shape != nil && t.shape.count == 0;
}

inline bool is_half_f32_0d_pair(const OpNode& node,
                                MPSDataType target_dt,
                                MPSGraphTensor* a_t,
                                MPSGraphTensor* b_t) {
    const Dtype out_dt = node.outputs[0].dtype;
    if (out_dt != Dtype::F16 && out_dt != Dtype::BF16)
        return false;
    if (node.name != "mul" && node.name != "div" && node.name != "floordiv")
        return false;
    return (a_t.dataType == target_dt && is_0d_f32_tensor(b_t)) ||
           (b_t.dataType == target_dt && is_0d_f32_tensor(a_t));
}

template <class BuilderBlock>
inline bool emit_binary(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 2 || node.outputs.empty())
        return false;
    TensorId a_id = node.inputs[0];
    TensorId b_id = node.inputs[1];
    if (a_id < 0 || b_id < 0)
        return false;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
    MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
    if (a_t == nil || b_t == nil || graph == nil)
        return false;

    // **AMP/mixed-dtype reconciliation.**  Under autocast, Lucid's
    // eager dispatch auto-downcasts F32+F16 binary ops to the lower
    // (autocast-target) dtype, so the trace's recorded output dtype
    // can disagree with what MPSGraph's binary builders produce
    // (MPSGraph silently *upcasts* mixed-dtype inputs to the higher
    // precision).  Reconcile by casting any operand whose dtype
    // doesn't match the recorded output dtype before the binary op.
    //
    // **Half op-math (F16/BF16 + 0-d F32 scalar/tensor).**  Eager keeps a
    // half tensor paired with a 0-d float32 operand at the half output dtype
    // while computing ``mul`` / ``div`` / ``floordiv`` in float32 before
    // rounding back to half (``_Opmath.h``).
    const auto held = mps_dtype_of(node.outputs[0].dtype);
    if (!held)
        return false;
    const MPSDataType target_dt = *held;
    const MPSDataType op_dt =
        is_half_f32_0d_pair(node, target_dt, a_t, b_t) ? MPSDataTypeFloat32 : target_dt;

    if (a_t.dataType != op_dt) {
        a_t = cast_mps_tensor(graph, a_t, op_dt, @"binop_cast_a");
    }
    if (b_t.dataType != op_dt) {
        b_t = cast_mps_tensor(graph, b_t, op_dt, @"binop_cast_b");
    }

    MPSGraphTensor* y = builder(graph, a_t, b_t);
    if (y == nil)
        return false;
    if (y.dataType != target_dt) {
        y = cast_mps_tensor(graph, y, target_dt, @"binop_cast_out");
    }
    ctx.bind(node.outputs[0].id, (__bridge void*)y);
    return true;
}

// ``add``, ``sub``, ``mul`` and ``div`` route ±0 / ±1 splat operands through
// ``add_pos_zero`` / ``sub_pos_zero`` — MPSGraph would otherwise fold the
// constant away and preserve f32/bf16 subnormals or -0 that hardware
// arithmetic flushes (_SignedZero.h, LCD-446 / LCD-452).
class AddEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "add"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool a_pos_zero = node.inputs.size() == 2 && is_pos_zero_splat(ctx, node.inputs[0]);
        const bool b_pos_zero = node.inputs.size() == 2 && is_pos_zero_splat(ctx, node.inputs[1]);
        const bool a_neg_zero = node.inputs.size() == 2 && is_neg_zero_splat(ctx, node.inputs[0]);
        const bool b_neg_zero = node.inputs.size() == 2 && is_neg_zero_splat(ctx, node.inputs[1]);
        return emit_binary(ctx, node,
                           [a_pos_zero, b_pos_zero, a_neg_zero,
                            b_neg_zero](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
                               if (b_pos_zero)
                                   return add_pos_zero(g, a, b);
                               if (a_pos_zero)
                                   return add_pos_zero(g, b, a);
                               if (b_neg_zero)
                                   return sub_pos_zero(g, a, zero_splat_like(g, b));
                               if (a_neg_zero)
                                   return sub_pos_zero(g, b, zero_splat_like(g, a));
                               return [g additionWithPrimaryTensor:a secondaryTensor:b name:@"add"];
                           });
    }
};

class SubEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sub"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool a_neg_zero = node.inputs.size() == 2 && is_neg_zero_splat(ctx, node.inputs[0]);
        const bool b_pos_zero = node.inputs.size() == 2 && is_pos_zero_splat(ctx, node.inputs[1]);
        const bool b_neg_zero = node.inputs.size() == 2 && is_neg_zero_splat(ctx, node.inputs[1]);
        return emit_binary(ctx, node,
                           [a_neg_zero, b_pos_zero, b_neg_zero](MPSGraph* g, MPSGraphTensor* a,
                                                                MPSGraphTensor* b) {
                               if (b_neg_zero)
                                   return add_pos_zero(g, a, zero_splat_like(g, b));
                               if (b_pos_zero)
                                   return sub_pos_zero(g, a, b);
                               if (a_neg_zero)
                                   return [g
                                       negativeWithTensor:sub_pos_zero(g, b, zero_splat_like(g, a))
                                                     name:@"sub_from_neg_zero"];
                               return [g subtractionWithPrimaryTensor:a
                                                      secondaryTensor:b
                                                                 name:@"sub"];
                           });
    }
};

inline MPSGraphTensor*
multiply_avoiding_i64_square(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, NSString* name) {
    // MPSGraph's algebraic simplifier pattern-matches ``mul(x, x)`` (after CSE
    // of identical subgraphs such as ``abs(a) * abs(a)`` or ``y.argmax(1) *
    // y.argmax(1)``) and rewrites it to ``mps.square(x)``.  On 64-bit integers
    // MPSGraph has a working ``mul_i64`` kernel but no ``square_i64`` kernel,
    // so the rewritten executable throws an Objective-C exception ("object
    // cannot be nil (key: square_i64)") on its first run.
    //
    // Wrapping the primary operand in ``maximum(a, a)`` and the secondary
    // operand in ``minimum(b, b)`` is an exact identity for all 64-bit integer
    // bit patterns (integers have no NaNs or signed zeros), and MPSGraph does
    // not fold ``maximum(x, x)`` or ``minimum(x, x)``.  Because the defining
    // MLIR ops of the two operands are distinct (``mps.maximum`` vs
    // ``mps.minimum``), CSE can never unify them into the same SSA value even
    // if ``a`` and ``b`` already end in ``maximum`` or ``minimum``, so
    // ``mul_i64`` always runs directly.
    if (a.dataType == MPSDataTypeInt64 || a.dataType == MPSDataTypeUInt64) {
        MPSGraphTensor* lhs = [g maximumWithPrimaryTensor:a secondaryTensor:a name:nil];
        MPSGraphTensor* rhs = [g minimumWithPrimaryTensor:b secondaryTensor:b name:nil];
        return [g multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:name];
    }
    return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:name];
}

class MulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "mul"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool a_pos_one = node.inputs.size() == 2 && is_pos_one_splat(ctx, node.inputs[0]);
        const bool b_pos_one = node.inputs.size() == 2 && is_pos_one_splat(ctx, node.inputs[1]);
        const bool a_neg_one = node.inputs.size() == 2 && is_neg_one_splat(ctx, node.inputs[0]);
        const bool b_neg_one = node.inputs.size() == 2 && is_neg_one_splat(ctx, node.inputs[1]);
        return emit_binary(
            ctx, node,
            [a_pos_one, b_pos_one, a_neg_one, b_neg_one](MPSGraph* g, MPSGraphTensor* a,
                                                         MPSGraphTensor* b) {
                if ((a.dataType & MPSDataTypeFloatBit) != 0) {
                    if (b_pos_one)
                        return sub_pos_zero(g, a, zero_splat_like(g, b));
                    if (a_pos_one)
                        return sub_pos_zero(g, b, zero_splat_like(g, a));
                    if (b_neg_one)
                        return [g negativeWithTensor:sub_pos_zero(g, a, zero_splat_like(g, b))
                                                name:@"mul_neg_one"];
                    if (a_neg_one)
                        return [g negativeWithTensor:sub_pos_zero(g, b, zero_splat_like(g, a))
                                                name:@"mul_neg_one"];
                }
                return multiply_avoiding_i64_square(g, a, b, @"mul");
            });
    }
};

class DivEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "div"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool b_pos_one = node.inputs.size() == 2 && is_pos_one_splat(ctx, node.inputs[1]);
        const bool b_neg_one = node.inputs.size() == 2 && is_neg_one_splat(ctx, node.inputs[1]);
        return emit_binary(
            ctx, node, [b_pos_one, b_neg_one](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
                if ((a.dataType & MPSDataTypeFloatBit) != 0) {
                    if (b_pos_one)
                        return sub_pos_zero(g, a, zero_splat_like(g, b));
                    if (b_neg_one)
                        return [g negativeWithTensor:sub_pos_zero(g, a, zero_splat_like(g, b))
                                                name:@"div_neg_one"];
                    // Prevent MPSGraph's MLIR canonicalizer from rewriting
                    // ``divide(a, const)`` into ``multiply(a, 1.0 / const)``
                    // when ``b`` is a constant splat (e.g. ``amax / 127.5``),
                    // which shifts 1 ULP away from IEEE-754 division.
                    MPSGraphTensor* zero_unless_nan =
                        [g minimumWithPrimaryTensor:[g absoluteWithTensor:a name:nil]
                                    secondaryTensor:zero_splat_like(g, a)
                                               name:nil];
                    b = [g subtractionWithPrimaryTensor:b
                                        secondaryTensor:[g floorWithTensor:zero_unless_nan name:nil]
                                                   name:nil];
                }
                return [g divisionWithPrimaryTensor:a secondaryTensor:b name:@"div"];
            });
    }
};

inline float round_scalar_for_mps_dtype(double v, MPSDataType dt) {
    const auto f = static_cast<float>(v);
    if (dt == MPSDataTypeFloat16)
        return backend::detail::half_bits_to_float(backend::detail::float_to_half_bits(f));
    if (dt == MPSDataTypeBFloat16)
        return backend::detail::bfloat16_bits_to_float(backend::detail::float_to_bfloat16_bits(f));
    return f;
}

inline bool is_special_pow_exponent(float e) {
    return e == 0.0f || e == 1.0f || e == 2.0f || e == 3.0f || e == -1.0f || e == -2.0f ||
           e == 0.5f || e == -0.5f;
}

inline MPSGraphTensor* cast_from_f32_if_needed(MPSGraph* g, MPSGraphTensor* t, MPSDataType dt) {
    return (t.dataType == dt) ? t : [g castTensor:t toType:dt name:nil];
}

inline MPSGraphTensor*
emit_pow_scalar_roots(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* xf, float e) {
    const MPSDataType orig_dt = a.dataType;
    const double inf_d = std::numeric_limits<double>::infinity();
    MPSGraphTensor* c_zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* c_pos_inf = [g constantWithScalar:inf_d dataType:MPSDataTypeFloat32];
    MPSGraphTensor* c_neg_inf = [g constantWithScalar:-inf_d dataType:MPSDataTypeFloat32];
    MPSGraphTensor* is_neg_inf = [g equalWithPrimaryTensor:xf secondaryTensor:c_neg_inf name:nil];
    if (e == 0.5f) {
        MPSGraphTensor* s =
            add_pos_zero(g, [g squareRootWithTensor:xf name:nil], zero_splat_like(g, xf));
        MPSGraphTensor* res = [g selectWithPredicateTensor:is_neg_inf
                                       truePredicateTensor:c_pos_inf
                                      falsePredicateTensor:s
                                                      name:@"pow_half"];
        return cast_from_f32_if_needed(g, res, orig_dt);
    }
    MPSGraphTensor* is_zero = [g equalWithPrimaryTensor:xf secondaryTensor:c_zero name:nil];
    if (orig_dt == MPSDataTypeFloat16) {
        MPSGraphTensor* rsq = [g reciprocalSquareRootWithTensor:a name:nil];
        MPSGraphTensor* h_zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat16];
        MPSGraphTensor* h_inf = [g constantWithScalar:inf_d dataType:MPSDataTypeFloat16];
        MPSGraphTensor* non_zero = [g selectWithPredicateTensor:is_neg_inf
                                            truePredicateTensor:h_zero
                                           falsePredicateTensor:rsq
                                                           name:nil];
        return [g selectWithPredicateTensor:is_zero
                        truePredicateTensor:h_inf
                       falsePredicateTensor:non_zero
                                       name:@"pow_neg_half"];
    }
    MPSGraphTensor* inv_s = [g reciprocalWithTensor:[g squareRootWithTensor:xf name:nil] name:nil];
    MPSGraphTensor* non_zero = [g selectWithPredicateTensor:is_neg_inf
                                        truePredicateTensor:c_zero
                                       falsePredicateTensor:inv_s
                                                       name:nil];
    MPSGraphTensor* res = [g selectWithPredicateTensor:is_zero
                                   truePredicateTensor:c_pos_inf
                                  falsePredicateTensor:non_zero
                                                  name:@"pow_neg_half"];
    return cast_from_f32_if_needed(g, res, orig_dt);
}

// Match ``lucid/_C/backend/gpu/Pow.h`` for a scalar special exponent ``e``:
// computed in float32 and rounded once to ``a``'s dtype, except ``e == 1.0f``
// (identity) and float16 ``e == -0.5f`` (float16's own correctly rounded rsqrt).
inline MPSGraphTensor* emit_pow_special_scalar(MPSGraph* g, MPSGraphTensor* a, float e) {
    const MPSDataType orig_dt = a.dataType;
    if (e == 0.0f)
        return constant_splat_like(g, a, 1.0);
    if (e == 1.0f)
        return [g identityWithTensor:a name:@"pow_one"];
    MPSGraphTensor* xf =
        (orig_dt == MPSDataTypeFloat32) ? a : [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
    if (e == 2.0f) {
        MPSGraphTensor* sq = [g multiplicationWithPrimaryTensor:xf
                                                secondaryTensor:xf
                                                           name:@"pow_sq"];
        return cast_from_f32_if_needed(g, sq, orig_dt);
    }
    if (e == 3.0f) {
        MPSGraphTensor* sq = [g multiplicationWithPrimaryTensor:xf secondaryTensor:xf name:nil];
        MPSGraphTensor* cube = [g multiplicationWithPrimaryTensor:sq
                                                  secondaryTensor:xf
                                                             name:@"pow_cube"];
        return cast_from_f32_if_needed(g, cube, orig_dt);
    }
    if (e == -1.0f) {
        MPSGraphTensor* inv = [g reciprocalWithTensor:xf name:@"pow_inv"];
        return cast_from_f32_if_needed(g, inv, orig_dt);
    }
    if (e == -2.0f) {
        MPSGraphTensor* sq = [g multiplicationWithPrimaryTensor:xf secondaryTensor:xf name:nil];
        MPSGraphTensor* c_tiny = [g constantWithScalar:static_cast<double>(0x1p-126f)
                                              dataType:MPSDataTypeFloat32];
        MPSGraphTensor* not_tiny = [g greaterThanOrEqualToWithPrimaryTensor:sq
                                                            secondaryTensor:c_tiny
                                                                       name:nil];
        MPSGraphTensor* inv_x = [g reciprocalWithTensor:xf name:nil];
        MPSGraphTensor* inv_x_sq = [g multiplicationWithPrimaryTensor:inv_x
                                                      secondaryTensor:inv_x
                                                                 name:nil];
        MPSGraphTensor* res = [g selectWithPredicateTensor:not_tiny
                                       truePredicateTensor:[g reciprocalWithTensor:sq name:nil]
                                      falsePredicateTensor:inv_x_sq
                                                      name:@"pow_neg_two"];
        return cast_from_f32_if_needed(g, res, orig_dt);
    }
    return emit_pow_scalar_roots(g, a, xf, e);
}

inline MPSGraphTensor* emit_pow_tensor_special_f32(MPSGraph* g,
                                                   MPSGraphTensor* x,
                                                   MPSGraphTensor* y,
                                                   MPSGraphTensor* native_pow) {
    auto f32 = [g](double v) { return [g constantWithScalar:v dataType:MPSDataTypeFloat32]; };
    const double inf_d = std::numeric_limits<double>::infinity();
    MPSGraphTensor* c_zero = f32(0.0);
    MPSGraphTensor* c_neg_two = f32(-2.0);
    MPSGraphTensor* c_neg_half = f32(-0.5);
    MPSGraphTensor* c_pos_inf = f32(inf_d);
    MPSGraphTensor* c_neg_inf = f32(-inf_d);
    MPSGraphTensor* sq = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor* cube = [g multiplicationWithPrimaryTensor:sq secondaryTensor:x name:nil];
    MPSGraphTensor* s = [g squareRootWithTensor:x name:nil];
    MPSGraphTensor* y_is_neg_two = [g equalWithPrimaryTensor:y secondaryTensor:c_neg_two name:nil];
    MPSGraphTensor* y_is_neg_half = [g equalWithPrimaryTensor:y
                                              secondaryTensor:c_neg_half
                                                         name:nil];
    MPSGraphTensor* not_tiny = [g greaterThanOrEqualToWithPrimaryTensor:sq
                                                        secondaryTensor:f32(0x1p-126)
                                                                   name:nil];
    MPSGraphTensor* tiny = [g logicalANDWithPrimaryTensor:y_is_neg_two
                                          secondaryTensor:[g notWithTensor:not_tiny name:nil]
                                                     name:nil];
    MPSGraphTensor* denom = [g selectWithPredicateTensor:y_is_neg_two
                                     truePredicateTensor:[g selectWithPredicateTensor:tiny
                                                                  truePredicateTensor:x
                                                                 falsePredicateTensor:sq
                                                                                 name:nil]
                                    falsePredicateTensor:[g selectWithPredicateTensor:y_is_neg_half
                                                                  truePredicateTensor:s
                                                                 falsePredicateTensor:x
                                                                                 name:nil]
                                                    name:nil];
    MPSGraphTensor* inv = [g reciprocalWithTensor:denom name:nil];
    MPSGraphTensor* inv_sq = [g multiplicationWithPrimaryTensor:inv secondaryTensor:inv name:nil];
    MPSGraphTensor* val_neg_two = [g selectWithPredicateTensor:tiny
                                           truePredicateTensor:inv_sq
                                          falsePredicateTensor:inv
                                                          name:nil];
    MPSGraphTensor* x_is_neg_inf = [g equalWithPrimaryTensor:x secondaryTensor:c_neg_inf name:nil];
    MPSGraphTensor* x_is_zero = [g equalWithPrimaryTensor:x secondaryTensor:c_zero name:nil];
    MPSGraphTensor* val_half =
        [g selectWithPredicateTensor:x_is_neg_inf
                 truePredicateTensor:c_pos_inf
                falsePredicateTensor:add_pos_zero(g, s, zero_splat_like(g, s))
                                name:nil];
    MPSGraphTensor* val_neg_half =
        [g selectWithPredicateTensor:x_is_zero
                 truePredicateTensor:c_pos_inf
                falsePredicateTensor:[g selectWithPredicateTensor:x_is_neg_inf
                                              truePredicateTensor:c_zero
                                             falsePredicateTensor:inv
                                                             name:nil]
                                name:nil];
    MPSGraphTensor* res = [g selectWithPredicateTensor:y_is_neg_half
                                   truePredicateTensor:val_neg_half
                                  falsePredicateTensor:native_pow
                                                  name:nil];
    res = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:y
                                                 secondaryTensor:f32(0.5)
                                                            name:nil]
                   truePredicateTensor:val_half
                  falsePredicateTensor:res
                                  name:nil];
    res = [g selectWithPredicateTensor:y_is_neg_two
                   truePredicateTensor:val_neg_two
                  falsePredicateTensor:res
                                  name:nil];
    res = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:y
                                                 secondaryTensor:f32(-1.0)
                                                            name:nil]
                   truePredicateTensor:inv
                  falsePredicateTensor:res
                                  name:nil];
    res = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:y
                                                 secondaryTensor:f32(3.0)
                                                            name:nil]
                   truePredicateTensor:cube
                  falsePredicateTensor:res
                                  name:nil];
    return [g selectWithPredicateTensor:[g equalWithPrimaryTensor:y
                                                  secondaryTensor:f32(2.0)
                                                             name:nil]
                    truePredicateTensor:sq
                   falsePredicateTensor:res
                                   name:nil];
}

// Match ``gpu_pow(a, b)`` in ``lucid/_C/backend/gpu/Pow.h``: float32,
// float16, and bfloat16 evaluate the special-exponent table element-by-element
// in float32, while integer and complex dtypes use ``powerWithPrimaryTensor``.
inline MPSGraphTensor* emit_pow_tensor(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    const MPSDataType orig_dt = a.dataType;
    if (orig_dt != MPSDataTypeFloat32 && orig_dt != MPSDataTypeFloat16 &&
        orig_dt != MPSDataTypeBFloat16) {
        return [g powerWithPrimaryTensor:a secondaryTensor:b name:@"pow"];
    }
    MPSGraphTensor* xf =
        (orig_dt == MPSDataTypeFloat32) ? a : [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* yf = (b.dataType == MPSDataTypeFloat32)
                             ? b
                             : [g castTensor:b toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* c_zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* c_one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* zero_ab = [g multiplicationWithPrimaryTensor:[g additionWithPrimaryTensor:xf
                                                                              secondaryTensor:yf
                                                                                         name:nil]
                                                 secondaryTensor:c_zero
                                                            name:nil];
    MPSGraphTensor* x = [g additionWithPrimaryTensor:xf secondaryTensor:zero_ab name:nil];
    MPSGraphTensor* y = [g additionWithPrimaryTensor:yf secondaryTensor:zero_ab name:nil];
    MPSGraphTensor* native_pow = [g powerWithPrimaryTensor:xf secondaryTensor:yf name:nil];
    MPSGraphTensor* res = emit_pow_tensor_special_f32(g, x, y, native_pow);
    MPSGraphTensor* res_dt = cast_from_f32_if_needed(g, res, orig_dt);
    MPSGraphTensor* a_bcast =
        (a.shape != nil && res_dt.shape != nil && [a.shape isEqualToArray:res_dt.shape])
            ? a
            : [g broadcastTensor:a toShapeTensor:[g shapeOfTensor:res_dt name:nil] name:nil];
    res_dt = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:y
                                                    secondaryTensor:c_one
                                                               name:nil]
                      truePredicateTensor:a_bcast
                     falsePredicateTensor:res_dt
                                     name:nil];
    MPSGraphTensor* one_out =
        [g logicalORWithPrimaryTensor:[g equalWithPrimaryTensor:y secondaryTensor:c_zero name:nil]
                      secondaryTensor:[g equalWithPrimaryTensor:x secondaryTensor:c_one name:nil]
                                 name:nil];
    return [g selectWithPredicateTensor:one_out
                    truePredicateTensor:[g constantWithScalar:1.0 dataType:orig_dt]
                   falsePredicateTensor:res_dt
                                   name:@"pow"];
}

class PowEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pow"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return emit_pow_tensor(g, a, b);
        });
    }
};

// ``matmul`` lives in ``linalg/Matmul.mm`` — see file header.

// R1 — additional binary ops.

class MaximumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "maximum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g maximumWithPrimaryTensor:a secondaryTensor:b name:@"maximum"];
        });
    }
};

class MinimumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "minimum"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return [g minimumWithPrimaryTensor:a secondaryTensor:b name:@"minimum"];
        });
    }
};

enum class DivModKind : std::uint8_t { FloorDiv, Remainder, Fmod };

inline MPSGraphTensor*
emit_float_divmod(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, DivModKind kind) {
    // Match eager ``DivMod.h`` / ``gpu/DivMod.h``: floating-point ``//``,
    // ``%`` (remainder), and ``fmod`` run in float32 starting from exact
    // ``fmod(|a|, |b|)`` with ``a``'s sign bit restored.
    const MPSDataType orig_dt = a.dataType;
    MPSGraphTensor* af =
        (orig_dt == MPSDataTypeFloat32) ? a : [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* bf = (b.dataType == MPSDataTypeFloat32)
                             ? b
                             : [g castTensor:b toType:MPSDataTypeFloat32 name:nil];

    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* r = [g moduloWithPrimaryTensor:[g absoluteWithTensor:af name:nil]
                                   secondaryTensor:[g absoluteWithTensor:bf name:nil]
                                              name:nil];
    MPSGraphTensor* inv_a = [g reciprocalWithTensor:af name:nil];
    MPSGraphTensor* neg_zero = [g logicalANDWithPrimaryTensor:[g equalWithPrimaryTensor:af
                                                                        secondaryTensor:zero
                                                                                   name:nil]
                                              secondaryTensor:[g lessThanWithPrimaryTensor:inv_a
                                                                           secondaryTensor:zero
                                                                                      name:nil]
                                                         name:nil];
    MPSGraphTensor* a_sign = [g logicalORWithPrimaryTensor:[g lessThanWithPrimaryTensor:af
                                                                        secondaryTensor:zero
                                                                                   name:nil]
                                           secondaryTensor:neg_zero
                                                      name:nil];
    MPSGraphTensor* mod = [g selectWithPredicateTensor:a_sign
                                   truePredicateTensor:[g negativeWithTensor:r name:nil]
                                  falsePredicateTensor:r
                                                  name:@"fmod"];
    if (kind == DivModKind::Fmod) {
        return (orig_dt == MPSDataTypeFloat32)
                   ? mod
                   : [g castTensor:mod toType:orig_dt name:@"fmod_cast"];
    }

    MPSGraphTensor* mod_nonzero = [g notEqualWithPrimaryTensor:mod secondaryTensor:zero name:nil];
    MPSGraphTensor* mod_neg = [g lessThanWithPrimaryTensor:mod secondaryTensor:zero name:nil];
    MPSGraphTensor* div_neg = [g lessThanWithPrimaryTensor:bf secondaryTensor:zero name:nil];
    MPSGraphTensor* signs_differ = [g logicalXORWithPrimaryTensor:mod_neg
                                                  secondaryTensor:div_neg
                                                             name:nil];
    MPSGraphTensor* needs_adjust = [g logicalANDWithPrimaryTensor:mod_nonzero
                                                  secondaryTensor:signs_differ
                                                             name:nil];
    if (kind == DivModKind::Remainder) {
        MPSGraphTensor* mod_plus_b = [g additionWithPrimaryTensor:mod secondaryTensor:bf name:nil];
        MPSGraphTensor* res = [g selectWithPredicateTensor:needs_adjust
                                       truePredicateTensor:mod_plus_b
                                      falsePredicateTensor:mod
                                                      name:@"remainder"];
        return (orig_dt == MPSDataTypeFloat32)
                   ? res
                   : [g castTensor:res toType:orig_dt name:@"remainder_cast"];
    }

    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* raw_div = [g divisionWithPrimaryTensor:af secondaryTensor:bf name:nil];
    MPSGraphTensor* adjust = [g selectWithPredicateTensor:needs_adjust
                                      truePredicateTensor:one
                                     falsePredicateTensor:zero
                                                     name:nil];
    MPSGraphTensor* num = [g subtractionWithPrimaryTensor:af secondaryTensor:mod name:nil];
    MPSGraphTensor* q_exact = [g divisionWithPrimaryTensor:num secondaryTensor:bf name:nil];
    MPSGraphTensor* div = [g subtractionWithPrimaryTensor:q_exact secondaryTensor:adjust name:nil];
    MPSGraphTensor* whole = [g floorWithTensor:div name:nil];
    MPSGraphTensor* frac = [g subtractionWithPrimaryTensor:div secondaryTensor:whole name:nil];
    MPSGraphTensor* round_up = [g greaterThanWithPrimaryTensor:frac secondaryTensor:half name:nil];
    MPSGraphTensor* whole_plus_one = [g additionWithPrimaryTensor:whole
                                                  secondaryTensor:one
                                                             name:nil];
    whole = [g selectWithPredicateTensor:round_up
                     truePredicateTensor:whole_plus_one
                    falsePredicateTensor:whole
                                    name:nil];
    MPSGraphTensor* signed_zero = [g multiplicationWithPrimaryTensor:raw_div
                                                     secondaryTensor:zero
                                                                name:nil];
    MPSGraphTensor* div_is_zero = [g equalWithPrimaryTensor:div secondaryTensor:zero name:nil];
    MPSGraphTensor* res = [g selectWithPredicateTensor:div_is_zero
                                   truePredicateTensor:signed_zero
                                  falsePredicateTensor:whole
                                                  name:nil];
    MPSGraphTensor* b_is_zero = [g equalWithPrimaryTensor:bf secondaryTensor:zero name:nil];
    res = [g selectWithPredicateTensor:b_is_zero
                   truePredicateTensor:raw_div
                  falsePredicateTensor:res
                                  name:@"floordiv"];
    return (orig_dt == MPSDataTypeFloat32)
               ? res
               : [g castTensor:res toType:orig_dt name:@"floordiv_cast"];
}

inline MPSGraphTensor*
emit_int_divmod(MPSGraph* g, MPSGraphTensor* a_in, MPSGraphTensor* b_in, DivModKind kind) {
    // Match eager ``DivMod.h`` / ``gpu/DivMod.h``: integer division truncates
    // toward zero, so ``fmod`` is the raw remainder ``a - q * b``, ``floordiv``
    // subtracts 1 when ``rem != 0`` and ``(rem < 0) != (b < 0)``, and
    // ``remainder`` adds ``b`` back in the same case.  A zero divisor on Metal
    // substitutes 1 during division and selects ``-1`` (``floordiv``) or ``a``
    // (``remainder`` / ``fmod``) without host synchronisation.  Widen int8 to
    // int32 to avoid MPSGraph's broken int8 ``a - (a / b) * b`` fold.
    const MPSDataType orig_dt = a_in.dataType;
    const MPSDataType dt = (orig_dt == MPSDataTypeInt8) ? MPSDataTypeInt32 : orig_dt;
    MPSGraphTensor* a = (orig_dt == dt) ? a_in : [g castTensor:a_in toType:dt name:nil];
    MPSGraphTensor* b_raw = (b_in.dataType == dt) ? b_in : [g castTensor:b_in toType:dt name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
    MPSGraphTensor* neg_one = [g constantWithScalar:-1.0 dataType:dt];
    MPSGraphTensor* by_zero = [g equalWithPrimaryTensor:b_raw secondaryTensor:zero name:nil];
    MPSGraphTensor* b = [g selectWithPredicateTensor:by_zero
                                 truePredicateTensor:one
                                falsePredicateTensor:b_raw
                                                name:nil];
    MPSGraphTensor* q = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    MPSGraphTensor* prod = multiply_avoiding_i64_square(g, q, b, nil);
    MPSGraphTensor* rem = [g subtractionWithPrimaryTensor:a secondaryTensor:prod name:nil];
    MPSGraphTensor* zero_b = [g multiplicationWithPrimaryTensor:b_raw
                                                secondaryTensor:zero
                                                           name:nil];
    MPSGraphTensor* a_bcast = [g additionWithPrimaryTensor:a secondaryTensor:zero_b name:nil];
    if (kind == DivModKind::Fmod) {
        MPSGraphTensor* res = [g selectWithPredicateTensor:by_zero
                                       truePredicateTensor:a_bcast
                                      falsePredicateTensor:rem
                                                      name:@"fmod"];
        return (orig_dt == dt) ? res : [g castTensor:res toType:orig_dt name:@"fmod_cast"];
    }

    MPSGraphTensor* rem_nonzero = [g notEqualWithPrimaryTensor:rem secondaryTensor:zero name:nil];
    MPSGraphTensor* rem_neg = [g lessThanWithPrimaryTensor:rem secondaryTensor:zero name:nil];
    MPSGraphTensor* div_neg = [g lessThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
    MPSGraphTensor* signs_differ = [g logicalXORWithPrimaryTensor:rem_neg
                                                  secondaryTensor:div_neg
                                                             name:nil];
    MPSGraphTensor* needs_adjust = [g logicalANDWithPrimaryTensor:rem_nonzero
                                                  secondaryTensor:signs_differ
                                                             name:nil];
    if (kind == DivModKind::Remainder) {
        MPSGraphTensor* rem_plus_b = [g additionWithPrimaryTensor:rem secondaryTensor:b name:nil];
        MPSGraphTensor* floored_rem = [g selectWithPredicateTensor:needs_adjust
                                               truePredicateTensor:rem_plus_b
                                              falsePredicateTensor:rem
                                                              name:nil];
        MPSGraphTensor* res = [g selectWithPredicateTensor:by_zero
                                       truePredicateTensor:a_bcast
                                      falsePredicateTensor:floored_rem
                                                      name:@"remainder"];
        return (orig_dt == dt) ? res : [g castTensor:res toType:orig_dt name:@"remainder_cast"];
    }

    MPSGraphTensor* adjust = [g selectWithPredicateTensor:needs_adjust
                                      truePredicateTensor:one
                                     falsePredicateTensor:zero
                                                     name:nil];
    MPSGraphTensor* floored_q = [g subtractionWithPrimaryTensor:q secondaryTensor:adjust name:nil];
    MPSGraphTensor* res = [g selectWithPredicateTensor:by_zero
                                   truePredicateTensor:neg_one
                                  falsePredicateTensor:floored_q
                                                  name:@"floordiv"];
    return (orig_dt == dt) ? res : [g castTensor:res toType:orig_dt name:@"floordiv_cast"];
}

inline MPSGraphTensor*
emit_divmod(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, DivModKind kind) {
    const MPSDataType dt = a.dataType;
    if (dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16 || dt == MPSDataTypeFloat32) {
        return emit_float_divmod(g, a, b, kind);
    }
    return emit_int_divmod(g, a, b, kind);
}

class FloordivEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "floordiv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return emit_divmod(g, a, b, DivModKind::FloorDiv);
        });
    }
};

class RemainderEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "remainder"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return emit_divmod(g, a, b, DivModKind::Remainder);
        });
    }
};

class FmodEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "fmod"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            return emit_divmod(g, a, b, DivModKind::Fmod);
        });
    }
};

class NextafterEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nextafter"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        // The same bit step as the eager Metal kernel (``nextafter_gpu_f32``):
        // one ULP is one unit of the float's bit pattern, not a fixed
        // epsilon.  This emitter used to add ``sign(b - a) * 1.19e-7``,
        // which leaves 100.0 unchanged and jumps 1e-10 by three orders of
        // magnitude — it was unreachable only because the op never
        // recorded its trace inputs.  Eager accepts F32 / F64 and F64 never
        // reaches Metal, so anything but F32 declines.
        if (node.outputs.empty() || node.outputs[0].dtype != Dtype::F32)
            return false;
        return emit_binary(ctx, node, [](MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
            auto i32 = [g](double v) { return [g constantWithScalar:v dataType:MPSDataTypeInt32]; };
            MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
            MPSGraphTensor* a_bits = [g reinterpretCastTensor:a toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* b_bits = [g reinterpretCastTensor:b toType:MPSDataTypeInt32 name:nil];

            // Toward b is +1 in value; a negative float's bits order the
            // other way, so the bit step flips sign with a.
            MPSGraphTensor* direction =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:b
                                                             secondaryTensor:a
                                                                        name:nil]
                         truePredicateTensor:i32(1)
                        falsePredicateTensor:i32(-1)
                                        name:nil];
            MPSGraphTensor* delta =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:a
                                                             secondaryTensor:zero
                                                                        name:nil]
                         truePredicateTensor:direction
                        falsePredicateTensor:[g negativeWithTensor:direction name:nil]
                                        name:nil];
            MPSGraphTensor* step = [g additionWithPrimaryTensor:a_bits
                                                secondaryTensor:delta
                                                           name:nil];

            // ±0 steps to the smallest subnormal carrying b's sign.
            MPSGraphTensor* from_zero =
                [g selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:b
                                                             secondaryTensor:zero
                                                                        name:nil]
                         truePredicateTensor:i32(1)
                        falsePredicateTensor:i32(static_cast<std::int32_t>(0x80000001u))
                                        name:nil];
            step = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:a
                                                          secondaryTensor:zero
                                                                     name:nil]
                            truePredicateTensor:from_zero
                           falsePredicateTensor:step
                                           name:nil];

            // a == b returns b itself, which keeps the sign of a zero.
            step = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:a
                                                          secondaryTensor:b
                                                                     name:nil]
                            truePredicateTensor:b_bits
                           falsePredicateTensor:step
                                           name:nil];

            MPSGraphTensor* any_nan = [g logicalORWithPrimaryTensor:[g isNaNWithTensor:a name:nil]
                                                    secondaryTensor:[g isNaNWithTensor:b name:nil]
                                                               name:nil];
            step = [g selectWithPredicateTensor:any_nan
                            truePredicateTensor:i32(static_cast<std::int32_t>(0x7FC00000u))
                           falsePredicateTensor:step
                                           name:nil];
            return [g reinterpretCastTensor:step toType:MPSDataTypeFloat32 name:@"nextafter"];
        });
    }
};

// pow_scalar — single-tensor input + ``exp`` (double) attr.
class PowScalarEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "pow_scalar"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        double exp_v = 1.0;
        auto it = node.attrs.find("exp");
        if (it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second))
                exp_v = *p;
        }
        const MPSDataType dt = x_t.dataType;
        const bool is_float =
            (dt == MPSDataTypeFloat32 || dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16);
        const float er = round_scalar_for_mps_dtype(exp_v, dt);
        MPSGraphTensor* y = nil;
        if (is_float && is_special_pow_exponent(er)) {
            y = emit_pow_special_scalar(graph, x_t, er);
        } else {
            MPSGraphTensor* e = [graph constantWithScalar:static_cast<double>(er) dataType:dt];
            y = [graph powerWithPrimaryTensor:x_t secondaryTensor:e name:@"pow_scalar"];
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

// rpow_scalar — base^x with ``base`` (double) attr.
class RPowScalarEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rpow_scalar"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;
        double base_v = std::exp(1.0);
        auto it = node.attrs.find("base");
        if (it != node.attrs.end()) {
            if (const auto* p = std::get_if<double>(&it->second))
                base_v = *p;
        }
        const float br = round_scalar_for_mps_dtype(base_v, x_t.dataType);
        MPSGraphTensor* b = [graph constantWithScalar:static_cast<double>(br)
                                             dataType:x_t.dataType];
        MPSGraphTensor* y = emit_pow_tensor(graph, b, x_t);
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

struct ElementwiseEmitterRegistrar {
    ElementwiseEmitterRegistrar() {
        register_emitter(std::make_unique<AddEmitter>());
        register_emitter(std::make_unique<SubEmitter>());
        register_emitter(std::make_unique<MulEmitter>());
        register_emitter(std::make_unique<DivEmitter>());
        register_emitter(std::make_unique<PowEmitter>());
        // ``matmul`` registered in linalg/Matmul.mm.
        // R1 additions.
        register_emitter(std::make_unique<MaximumEmitter>());
        register_emitter(std::make_unique<MinimumEmitter>());
        register_emitter(std::make_unique<FloordivEmitter>());
        register_emitter(std::make_unique<RemainderEmitter>());
        register_emitter(std::make_unique<FmodEmitter>());
        register_emitter(std::make_unique<NextafterEmitter>());
        register_emitter(std::make_unique<PowScalarEmitter>());
        register_emitter(std::make_unique<RPowScalarEmitter>());
    }
};

[[maybe_unused]] static const ElementwiseEmitterRegistrar g_elementwise_registrar;

}  // namespace

}  // namespace lucid::compile
