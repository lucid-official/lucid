// lucid/_C/test/backend/test_shapeless_fused.cpp
//
// LCD-1062: ``ShapelessFused`` answers an empty operand with the steps run
// eagerly.  MLX's shapeless trace broadcasts a 0 extent against a 1 as 1, so
// ``maximum`` of a ``(0,)`` and a ``(1,)`` array compiled shapeless came back
// ``(1,)`` holding one element.  What is held here is the shape and dtype of
// each output, which are read from the array and need no evaluation, and that a
// call with no empty operand still runs the compiled trace and agrees with the
// eager answer.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "backend/gpu/ShapelessFused.h"

namespace mx = ::mlx::core;
using lucid::gpu::FusedArrays;
using lucid::gpu::ShapelessFused;

namespace {

FusedArrays max_steps(const FusedArrays& in) {
    return {mx::maximum(in[0], in[1])};
}

// Two outputs of different dtypes, as ``AxisIndex``'s index check returns.
FusedArrays max_and_less_steps(const FusedArrays& in) {
    return {mx::maximum(in[0], in[1]), mx::less(in[0], in[1])};
}

using Dims = std::vector<mx::ShapeElem>;

mx::array filled(const Dims& dims, float value) {
    return mx::full(mx::Shape(dims.begin(), dims.end()), value, mx::float32);
}

Dims dims_of(const mx::array& a) {
    return Dims(a.shape().begin(), a.shape().end());
}

struct Broadcast {
    Dims a;
    Dims b;
    Dims want;
};

const std::vector<Broadcast>& empty_broadcasts() {
    static const std::vector<Broadcast> cases = {
        {{0}, {1}, {0}},
        {{1}, {0}, {0}},
        {{0, 3}, {1, 3}, {0, 3}},
        {{2, 0}, {2, 1}, {2, 0}},
        {{3, 1, 0}, {1, 4, 1}, {3, 4, 0}},
        {{0}, {}, {0}},
    };
    return cases;
}

}  // namespace

TEST(ShapelessFusedTest, AnEmptyBroadcastKeepsItsExtent) {
    const ShapelessFused fused(&max_steps);
    for (const auto& c : empty_broadcasts()) {
        const auto out = fused({filled(c.a, 0.0f), filled(c.b, 1.0f)});
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(dims_of(out[0]), c.want);
        EXPECT_EQ(out[0].size(), 0u);
    }
}

TEST(ShapelessFusedTest, EveryOutputKeepsItsShapeAndDtype) {
    const ShapelessFused fused(&max_and_less_steps);
    for (const auto& c : empty_broadcasts()) {
        const auto out = fused({filled(c.a, 0.0f), filled(c.b, 1.0f)});
        ASSERT_EQ(out.size(), 2u);
        EXPECT_EQ(dims_of(out[0]), c.want);
        EXPECT_EQ(dims_of(out[1]), c.want);
        EXPECT_EQ(out[0].dtype(), mx::float32);
        EXPECT_EQ(out[1].dtype(), mx::bool_);
    }
}

TEST(ShapelessFusedTest, ANonEmptyBroadcastRunsTheTraceAndAgreesWithEager) {
    const ShapelessFused fused(&max_steps);
    // An empty call first: it must not leave the trace unable to be built.
    (void)fused({filled({0}, 0.0f), filled({1}, 1.0f)});

    const mx::array a = mx::array({1.0f, -2.0f, 3.0f}, mx::Shape{3, 1});
    const mx::array b = mx::array({0.0f, 2.0f}, mx::Shape{1, 2});
    auto out = fused({a, b});
    auto eager = max_steps({a, b});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(dims_of(out[0]), (Dims{3, 2}));
    mx::eval(out[0], eager[0]);
    for (std::size_t i = 0; i < out[0].size(); ++i)
        EXPECT_EQ(out[0].data<float>()[i], eager[0].data<float>()[i]) << "element " << i;
}

TEST(ShapelessFusedTest, HasEmptyOperandReadsElementCountsNotRanks) {
    EXPECT_FALSE(lucid::gpu::has_empty_operand({filled({}, 1.0f), filled({3}, 1.0f)}));
    EXPECT_TRUE(lucid::gpu::has_empty_operand({filled({3}, 1.0f), filled({2, 0}, 1.0f)}));
    EXPECT_FALSE(lucid::gpu::has_empty_operand({}));
}
