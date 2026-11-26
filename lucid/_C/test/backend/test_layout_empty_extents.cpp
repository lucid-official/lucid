// lucid/_C/test/backend/test_layout_empty_extents.cpp
//
// Empty-extent and 0-d guards for CPU layout ops (LCD-1119).
//
// CpuBackend::pad previously divided ``in_numel / shape.back()`` without
// checking for an empty input, causing integer division by zero under UBSan
// when the trailing extent was zero, and segfaulted on 0-d inputs.  Sibling
// CPU layout kernels also passed null empty-tensor buffers to std::memcpy and
// std::memset.  These tests exercise each layout op on CPU directly in C++.

#include <gtest/gtest.h>

#include "../../ops/utils/Concat.h"
#include "../../ops/utils/Layout.h"
#include "../../ops/utils/Pad.h"
#include "../../ops/utils/Repeat.h"
#include "numeric_assert.h"
#include "tensor_factory.h"

using namespace lucid;
using namespace lucid::test;

TEST(LayoutEmptyExtents, PadEmptyLastDimFillsConstant) {
    auto x = cpu_zeros({2, 0}, Dtype::F32);
    auto y = pad_op(x, {{0, 0}, {1, 1}}, 3.0);
    EXPECT_TENSOR_SHAPE(y, (Shape{2, 2}));
    EXPECT_TENSOR_NEAR(y, 3.0f, 1e-6f);
}

TEST(LayoutEmptyExtents, PadEmptyLeadingDimFillsConstant) {
    auto x = cpu_zeros({0, 3}, Dtype::F32);
    auto y = pad_op(x, {{1, 1}, {0, 0}}, 2.5);
    EXPECT_TENSOR_SHAPE(y, (Shape{2, 3}));
    EXPECT_TENSOR_NEAR(y, 2.5f, 1e-6f);
}

TEST(LayoutEmptyExtents, PadEmptyOutputRemainsEmpty) {
    auto x = cpu_zeros({0, 2}, Dtype::F32);
    auto y = pad_op(x, {{0, 0}, {1, 1}}, 0.0);
    EXPECT_TENSOR_SHAPE(y, (Shape{0, 4}));
    EXPECT_EQ(y->numel(), 0u);
}

TEST(LayoutEmptyExtents, PadScalarReturnsCopy) {
    auto x = cpu_full({}, 5.0, Dtype::F32);
    auto y = pad_op(x, {}, 3.0);
    EXPECT_TENSOR_SHAPE(y, (Shape{}));
    EXPECT_TENSOR_NEAR(y, 5.0f, 1e-6f);
}

TEST(LayoutEmptyExtents, PadHalfAndIntEmptyLastDim) {
    auto f16 = cpu_zeros({2, 0}, Dtype::F16);
    auto yf16 = pad_op(f16, {{0, 0}, {1, 2}}, 1.5);
    EXPECT_TENSOR_SHAPE(yf16, (Shape{2, 3}));
    EXPECT_TENSOR_DTYPE(yf16, Dtype::F16);

    auto i32 = cpu_zeros({2, 0}, Dtype::I32);
    auto yi32 = pad_op(i32, {{0, 0}, {1, 1}}, 7.0);
    EXPECT_TENSOR_SHAPE(yi32, (Shape{2, 2}));
    EXPECT_TENSOR_DTYPE(yi32, Dtype::I32);
}

TEST(LayoutEmptyExtents, ConcatMixingEmptyAndNonEmpty) {
    auto e = cpu_zeros({2, 0}, Dtype::F32);
    auto f = cpu_full({2, 3}, 4.0, Dtype::F32);
    auto y = concatenate_op({e, f, e}, 1);
    EXPECT_TENSOR_SHAPE(y, (Shape{2, 3}));
    EXPECT_TENSOR_NEAR(y, 4.0f, 1e-6f);
}

TEST(LayoutEmptyExtents, StackRepeatTileBroadcastEmpty) {
    auto e = cpu_zeros({2, 3, 0}, Dtype::F32);
    EXPECT_TENSOR_SHAPE(stack_op({e, e}, 0), (Shape{2, 2, 3, 0}));
    EXPECT_TENSOR_SHAPE(repeat_op(e, 2, 1), (Shape{2, 6, 0}));
    EXPECT_TENSOR_SHAPE(tile_op(e, {2, 1, 3}), (Shape{4, 3, 0}));
    EXPECT_TENSOR_SHAPE(broadcast_to_op(e, {4, 2, 3, 0}), (Shape{4, 2, 3, 0}));
}
