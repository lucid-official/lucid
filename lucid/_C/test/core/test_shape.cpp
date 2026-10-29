// lucid/_C/test/core/test_shape.cpp
// Tests for Shape arithmetic: numel, strides, broadcasting.

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "../../core/Error.h"
#include "../../core/Shape.h"

using namespace lucid;

TEST(ShapeNumel, ScalarIsOne) {
    EXPECT_EQ(shape_numel({}), 1u);
}

TEST(ShapeNumel, VectorCorrect) {
    EXPECT_EQ(shape_numel({5}), 5u);
}

TEST(ShapeNumel, MatrixCorrect) {
    EXPECT_EQ(shape_numel({3, 4}), 12u);
}

TEST(ShapeNumel, HigherDimCorrect) {
    EXPECT_EQ(shape_numel({2, 3, 4, 5}), 120u);
}

// LCD-669: the product used to wrap — {2^33, 2^33} counted 0 elements, and
// the tensor built over no buffer segfaulted on the first read.
TEST(ShapeNumel, OverflowIsRefused) {
    EXPECT_THROW(shape_numel({1LL << 33, 1LL << 33}), InvalidArgument);
    EXPECT_THROW(shape_numel({1LL << 62, 4}), InvalidArgument);
    EXPECT_THROW(shape_numel({1LL << 31, 1LL << 31, 1LL << 31}), InvalidArgument);
}

TEST(ShapeNumel, LimitIsInclusive) {
    EXPECT_EQ(shape_numel({static_cast<std::int64_t>(kMaxNumel)}), kMaxNumel);
    EXPECT_THROW(shape_numel({static_cast<std::int64_t>(kMaxNumel) + 1}), InvalidArgument);
    // Every dtype's byte size of the largest count still fits int64.
    EXPECT_LE(kMaxNumel * 16, static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
}

TEST(ShapeNumel, ZeroExtentWinsOverAnyProduct) {
    EXPECT_EQ(shape_numel({1LL << 62, 1LL << 62, 0}), 0u);
    EXPECT_EQ(shape_numel({0, 1LL << 62, 1LL << 62}), 0u);
}

TEST(ShapeNumel, NegativeIsStillRefused) {
    EXPECT_THROW(shape_numel({3, -1}), InvalidArgument);
}

TEST(CheckConcreteShape, RefusesOverflow) {
    EXPECT_THROW(check_concrete_shape({1LL << 40, 1LL << 40}), InvalidArgument);
    EXPECT_NO_THROW(check_concrete_shape({1LL << 20, 1LL << 20}));
}

TEST(MulExtent, OverflowIsRefused) {
    EXPECT_EQ(mul_extent(3, 1LL << 40), 3LL << 40);
    EXPECT_EQ(mul_extent(0, std::numeric_limits<std::int64_t>::max()), 0);
    EXPECT_THROW(mul_extent(274177, 67280421310721), InvalidArgument);  // 2^64 + 1
}

TEST(CountNbytes, CountIsHeldToTheLimit) {
    EXPECT_EQ(count_nbytes(10, 8), 80u);
    EXPECT_THROW(count_nbytes(1ULL << 62, 8), InvalidArgument);  // wrapped to 0 bytes
}

TEST(ContiguousStride, ScalarEmpty) {
    auto s = contiguous_stride({}, 4);
    EXPECT_TRUE(s.empty());
}

TEST(ContiguousStride, VectorStride) {
    auto s = contiguous_stride({5}, 4);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0], 4);  // stride in bytes = element_size * next_dim_product
}

TEST(ContiguousStride, MatrixRowMajor) {
    // (3, 4) float32: row stride = 4*4=16, col stride = 4
    auto s = contiguous_stride({3, 4}, 4);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0], 4 * 4);  // 16 bytes per row
    EXPECT_EQ(s[1], 4);      // 4 bytes per element
}

TEST(ContiguousStride, OverflowingStrideIsRefused) {
    // Zero elements, but the outer stride (4 * 2^62 bytes) cannot be stored.
    EXPECT_THROW(contiguous_stride({0, 1LL << 62, 4}, 4), InvalidArgument);
    // The outermost extent is in no stride.
    auto s = contiguous_stride({1LL << 62, 0}, 4);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0], 4);
    EXPECT_EQ(s[1], 4);
}

TEST(ShapeEquality, SameShapeEqual) {
    Shape a{2, 3, 4};
    Shape b{2, 3, 4};
    EXPECT_EQ(a, b);
}

TEST(ShapeEquality, DifferentShapeNotEqual) {
    Shape a{2, 3};
    Shape b{3, 2};
    EXPECT_NE(a, b);
}
